#!/usr/bin/env python3
"""
Fast local cross-encoder reranker with TTL cache and tenant isolation.

Defaults to BAAI/BGE reranker models when available. If the model cannot be
loaded (e.g., no weights available due to offline environment), the reranker
falls back to a lightweight heuristic that preserves ordering and applies a
score-knee based dynamic-k trim. This ensures requests never block on model
downloads and avoids Anthropic calls entirely.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _stable_doc_id(doc: Dict[str, Any]) -> str:
    """Derive a stable identifier for a result item.

    Prefers existing IDs; otherwise hashes the content and path for stability.
    """
    for key in ("id", "doc_id", "chunk_id"):
        if key in doc and doc[key]:
            return str(doc[key])
    base = f"{doc.get('file_path','')}|{doc.get('content','')}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


class _TTLCache:
    """Very small in-memory TTL cache for pairwise scores.

    Keyed by (tenant, query_hash, doc_id). Keeps up to `max_items` entries.
    """

    def __init__(self, ttl_seconds: int = 600, max_items: int = 50_000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, Tuple[float, float]] = {}
        # value = (score, expiry_ts)

    def _prune(self):
        if len(self._store) <= self.max_items:
            return
        # Drop oldest 10% by expiry
        items = sorted(self._store.items(), key=lambda kv: kv[1][1])
        cutoff = max(1, int(len(items) * 0.1))
        for k, _ in items[:cutoff]:
            self._store.pop(k, None)

    def _now(self) -> float:
        return time.time()

    def make_key(self, tenant: Optional[str], query_hash: str, doc_id: str) -> str:
        return f"{tenant or 'default'}::{query_hash}::{doc_id}"

    def get(self, key: str) -> Optional[float]:
        val = self._store.get(key)
        if not val:
            return None
        score, expiry = val
        if self._now() > expiry:
            self._store.pop(key, None)
            return None
        return score

    def set(self, key: str, score: float) -> None:
        self._store[key] = (score, self._now() + self.ttl)
        self._prune()


@dataclass
class RerankConfig:
    model_name: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    model_path: Optional[str] = os.getenv("RERANKER_MODEL_PATH")
    latency_budget_ms: int = int(os.getenv("RERANK_BUDGET_MS", "120"))
    batch_size: int = int(os.getenv("RERANK_BATCH", "48"))
    cache_ttl_s: int = int(os.getenv("RERANK_CACHE_TTL", "600"))


class CrossEncoderReranker:
    """Local cross-encoder reranker with latency budget and caching.

    - Attempts to load a sentence-transformers CrossEncoder model lazily.
    - Uses an LRU-ish TTL cache for pairwise (query, doc) scores to avoid
      recomputation within a session and across nearby queries.
    - If the model isn't available, falls back to a fast heuristic.
    """

    def __init__(self, config: Optional[RerankConfig] = None, tenant_id: Optional[str] = None):
        self.config = config or RerankConfig()
        self.tenant_id = tenant_id
        self._model = None  # lazy
        self._cache = _TTLCache(ttl_seconds=self.config.cache_ttl_s)
        self._load_error: Optional[str] = None
        # Metrics collection
        self._metrics = {
            'total_requests': 0,
            'budget_skipped': 0,
            'heuristic_used': 0,
            'cross_encoder_used': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'latency_p50_ms': [],
            'latency_p95_ms': [],
            'latency_max_window': []  # Keep last 100 latencies for percentile calculation
        }

    def _query_hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()

    def _ensure_model(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            # Late import to avoid heavy deps when unused
            from sentence_transformers import CrossEncoder  # type: ignore
            name_or_path = self.config.model_path or self.config.model_name
            # trust_remote_code False for safety; allow local path via env
            self._model = CrossEncoder(name_or_path, trust_remote_code=False)
            logger.info(f"Loaded cross-encoder reranker: {name_or_path}")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                f"Cross-encoder not available; using heuristic fallback. Error: {e}"
            )

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        latency_budget_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank results within a latency budget; never blocks beyond budget.

        Returns a list of dicts, preserving original fields and updating scores
        when reranked. Falls back to heuristic ordering if model unavailable
        or on timeout/errors.
        """
        start_time = time.perf_counter()
        self._metrics['total_requests'] += 1
        
        if not results:
            return []

        # Early bound: only rerank small candidate set
        candidates = results[: max(top_k * 3, min(len(results), 50))]
        query_h = self._query_hash(query)
        tenant = self.tenant_id
        latency_ms = latency_budget_ms or self.config.latency_budget_ms

        # Try model path; if missing use heuristic
        self._ensure_model()
        if self._model is None:
            self._metrics['heuristic_used'] += 1
            result = self._heuristic_rerank(query, candidates, top_k)
            self._record_latency_metrics(start_time)
            return result

        # Prepare pairs and collect cached scores
        pairs: List[Tuple[str, str]] = []
        scores: List[Optional[float]] = []
        cache_hits = 0
        cache_misses = 0
        
        for doc in candidates:
            doc_id = _stable_doc_id(doc)
            key = self._cache.make_key(tenant, query_h, doc_id)
            cached = self._cache.get(key)
            if cached is not None:
                scores.append(float(cached))
                cache_hits += 1
            else:
                text = self._pair_text(doc)
                pairs.append((query, text))
                scores.append(None)
                cache_misses += 1
        
        self._metrics['cache_hits'] += cache_hits
        self._metrics['cache_misses'] += cache_misses

        async def _infer_missing() -> List[float]:
            # CrossEncoder expects list[(q, d)] and returns relevance scores
            if not pairs:
                return []
            # Run in default loop executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self._model.predict(pairs, batch_size=self.config.batch_size).tolist()
            )

        try:
            preds: List[float] = await asyncio.wait_for(_infer_missing(), timeout=max(0.01, latency_ms / 1000.0))
            self._metrics['cross_encoder_used'] += 1
        except asyncio.TimeoutError:
            logger.debug(f"Cross-encoder rerank exceeded budget ({latency_ms}ms); using heuristic.")
            self._metrics['budget_skipped'] += 1
            result = self._heuristic_rerank(query, candidates, top_k)
            self._record_latency_metrics(start_time)
            return result
        except Exception as e:
            logger.debug(f"Cross-encoder rerank failed ({e}); using heuristic.")
            self._metrics['heuristic_used'] += 1
            result = self._heuristic_rerank(query, candidates, top_k)
            self._record_latency_metrics(start_time)
            return result

        # Merge scores back and cache
        pred_idx = 0
        for i, s in enumerate(scores):
            if s is None:
                s = float(preds[pred_idx]) if pred_idx < len(preds) else 0.0
                pred_idx += 1
            # Cache
            doc_id = _stable_doc_id(candidates[i])
            key = self._cache.make_key(tenant, query_h, doc_id)
            self._cache.set(key, s)
            # Attach rerank score
            candidates[i]["rerank_score"] = s

        # Sort by rerank_score (desc), then original score as tie-breaker
        candidates.sort(key=lambda d: (d.get("rerank_score", 0.0), d.get("score", 0.0)), reverse=True)
        result = candidates[:top_k]
        self._record_latency_metrics(start_time)
        return result

    def _pair_text(self, doc: Dict[str, Any]) -> str:
        """Compact text used for pair scoring to reduce tokenization cost."""
        meta = doc.get("metadata", {})
        title = meta.get("title") or meta.get("symbol") or meta.get("file_path") or ""
        header = meta.get("header") or meta.get("function") or meta.get("class") or ""
        snippet = doc.get("content", "")
        # Trim the snippet to a small window
        if isinstance(snippet, str) and len(snippet) > 512:
            snippet = snippet[:512]
        return f"{title}\n{header}\n{snippet}".strip()

    def _heuristic_rerank(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Heuristic: keep original order, dynamic-k knee trim, tiny boosts."""
        # Knee over vector scores if present
        scores = [float(r.get("score", 0.0)) for r in results]
        k = self._find_knee(scores, default=min(len(results), top_k))
        k = min(k, top_k)

        # Lightweight keyword overlap boost
        q_terms = set(t for t in query.lower().split() if len(t) > 3)
        def _boost(r: Dict[str, Any]) -> float:
            text = (r.get("content") or "")
            if not isinstance(text, str):
                text = str(text)
            tokens = set(w for w in text.lower().split() if len(w) > 3)
            overlap = len(q_terms & tokens)
            return 0.01 * overlap

        for r in results:
            r["rerank_score"] = r.get("score", 0.0) + _boost(r)

        results.sort(key=lambda d: (d.get("rerank_score", 0.0), d.get("score", 0.0)), reverse=True)
        return results[:k]

    def _find_knee(self, scores: List[float], default: int) -> int:
        if not scores:
            return 0
        drops = [scores[i-1] - scores[i] for i in range(1, len(scores))]
        if not drops:
            return default
        max_i = max(range(len(drops)), key=lambda i: drops[i])
        return max(1, min(len(scores), max_i + 1))

    def _record_latency_metrics(self, start_time: float):
        """Record latency metrics for performance tracking."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Keep a rolling window of last 100 latencies for percentile calculation
        self._metrics['latency_max_window'].append(elapsed_ms)
        if len(self._metrics['latency_max_window']) > 100:
            self._metrics['latency_max_window'].pop(0)
        
        # Calculate and store current percentiles
        if self._metrics['latency_max_window']:
            sorted_latencies = sorted(self._metrics['latency_max_window'])
            n = len(sorted_latencies)
            
            p50_idx = int(0.50 * n)
            p95_idx = int(0.95 * n)
            
            self._metrics['latency_p50_ms'] = sorted_latencies[p50_idx] if p50_idx < n else sorted_latencies[-1]
            self._metrics['latency_p95_ms'] = sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive reranker statistics and performance metrics."""
        cache_size = len(self._cache._store)
        total_cache_ops = self._metrics['cache_hits'] + self._metrics['cache_misses']
        
        # Calculate real cache hit rate
        cache_hit_rate = (
            self._metrics['cache_hits'] / total_cache_ops if total_cache_ops > 0 else 0.0
        )
        
        # Calculate usage distribution
        total_requests = self._metrics['total_requests']
        usage_stats = {}
        if total_requests > 0:
            usage_stats = {
                'heuristic_usage_rate': self._metrics['heuristic_used'] / total_requests,
                'cross_encoder_usage_rate': self._metrics['cross_encoder_used'] / total_requests,
                'budget_skip_rate': self._metrics['budget_skipped'] / total_requests,
            }
        
        stats = {
            # Basic info
            'tenant_id': self.tenant_id or 'default',
            'model_available': self._model is not None,
            'load_error': self._load_error,
            
            # Request metrics
            'total_requests': total_requests,
            'budget_skipped': self._metrics['budget_skipped'],
            'heuristic_used': self._metrics['heuristic_used'],
            'cross_encoder_used': self._metrics['cross_encoder_used'],
            
            # Cache metrics
            'cache_size': cache_size,
            'cache_max_items': self._cache.max_items,
            'cache_ttl_seconds': self._cache.ttl,
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            
            # Performance metrics
            'latency_p50_ms': self._metrics['latency_p50_ms'],
            'latency_p95_ms': self._metrics['latency_p95_ms'],
            'latency_window_size': len(self._metrics['latency_max_window']),
        }
        
        # Add usage distribution if we have data
        stats.update(usage_stats)
            
        return stats

