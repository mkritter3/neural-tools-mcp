#!/usr/bin/env python3
"""
L9 Feature Flags System for Neural Embeddings
Manages feature flags for embedding model selection, A/B testing, and performance monitoring
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool
    description: str
    rollout_percentage: float = 100.0
    conditions: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.conditions is None:
            self.conditions = {}

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    variants: Dict[str, float]  # variant_name -> weight
    enabled: bool = True
    traffic_percentage: float = 100.0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class FeatureFlagManager:
    """Manages feature flags for neural embedding system"""
    
    def __init__(self, config_path: str = ".claude/neural-system/feature_flags.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default feature flags for L9 implementation
        self.default_flags = {
            # Phase 1: Embedding Model Upgrades
            "use_qodo_embed": FeatureFlag(
                name="use_qodo_embed",
                enabled=os.getenv("USE_QODO_EMBED", "false").lower() == "true",
                description="Enable Qodo-Embed-1-1.5B for code-specific embeddings",
                rollout_percentage=10.0  # Conservative rollout
            ),
            
            "use_codestral_embed": FeatureFlag(
                name="use_codestral_embed", 
                enabled=os.getenv("USE_CODESTRAL_EMBED", "false").lower() == "true",
                description="Enable Codestral Embed for code understanding"
            ),
            
            "use_openai_embeddings": FeatureFlag(
                name="use_openai_embeddings",
                enabled=os.getenv("OPENAI_API_KEY") is not None,
                description="Enable OpenAI text-embedding-3-small for general text",
                conditions={"requires_api_key": True}
            ),
            
            # Shadow Indexing and A/B Testing
            "shadow_indexing": FeatureFlag(
                name="shadow_indexing",
                enabled=True,
                description="Enable shadow indexing for multiple embedding models"
            ),
            
            "enable_ab_testing": FeatureFlag(
                name="enable_ab_testing",
                enabled=os.getenv("ENABLE_AB_TESTING", "false").lower() == "true",
                description="Enable A/B testing between embedding models",
                rollout_percentage=50.0
            ),
            
            # Performance Monitoring
            "performance_monitoring": FeatureFlag(
                name="performance_monitoring",
                enabled=os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true", 
                description="Enable detailed performance monitoring and metrics"
            ),
            
            "debug_embeddings": FeatureFlag(
                name="debug_embeddings",
                enabled=os.getenv("DEBUG_EMBEDDINGS", "false").lower() == "true",
                description="Enable detailed embedding generation debugging"
            ),
            
            # Phase 2 Features (disabled by default)
            "ast_aware_chunking": FeatureFlag(
                name="ast_aware_chunking",
                enabled=False,
                description="Enable AST-aware code chunking (Phase 2)",
                conditions={"phase": 2, "requires_tree_sitter": True}
            ),
            
            "cross_domain_search": FeatureFlag(
                name="cross_domain_search", 
                enabled=False,
                description="Enable cross-domain semantic search (Phase 2)",
                conditions={"phase": 2}
            )
        }
        
        # A/B test configurations
        self.default_ab_tests = {
            "embedding_model_comparison": ABTestConfig(
                test_name="embedding_model_comparison",
                variants={
                    "onnx_baseline": 0.4,      # 40% - current system
                    "qodo_embed": 0.3,         # 30% - code-specific model
                    "openai_hybrid": 0.3       # 30% - OpenAI + ONNX hybrid
                },
                enabled=os.getenv("ENABLE_AB_TESTING", "false").lower() == "true",
                traffic_percentage=25.0  # Only 25% of traffic participates in A/B test
            )
        }
        
        self.flags = {}
        self.ab_tests = {}
        self.load_config()
    
    def load_config(self):
        """Load feature flags from config file or create defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Load feature flags
                for flag_name, flag_data in config.get('flags', {}).items():
                    self.flags[flag_name] = FeatureFlag(**flag_data)
                
                # Load A/B tests
                for test_name, test_data in config.get('ab_tests', {}).items():
                    self.ab_tests[test_name] = ABTestConfig(**test_data)
                
                logger.info(f"Loaded {len(self.flags)} feature flags and {len(self.ab_tests)} A/B tests")
            else:
                # Use defaults
                self.flags = self.default_flags.copy()
                self.ab_tests = self.default_ab_tests.copy()
                self.save_config()
                logger.info("Created default feature flag configuration")
                
        except Exception as e:
            logger.warning(f"Failed to load feature flags, using defaults: {e}")
            self.flags = self.default_flags.copy()
            self.ab_tests = self.default_ab_tests.copy()
    
    def save_config(self):
        """Save current feature flags to config file"""
        try:
            config = {
                'flags': {name: asdict(flag) for name, flag in self.flags.items()},
                'ab_tests': {name: asdict(test) for name, test in self.ab_tests.items()},
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled"""
        if flag_name not in self.flags:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        
        # Check basic enabled status
        if not flag.enabled:
            return False
        
        # Check rollout percentage (simple hash-based distribution)
        if flag.rollout_percentage < 100.0:
            import hashlib
            hash_input = f"{flag_name}_{context.get('user_id', 'default') if context else 'default'}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            percentage = (hash_value % 100) + 1
            
            if percentage > flag.rollout_percentage:
                return False
        
        # Check conditions
        if flag.conditions:
            if context:
                for condition_key, condition_value in flag.conditions.items():
                    if condition_key in context and context[condition_key] != condition_value:
                        return False
        
        return True
    
    def get_ab_test_variant(self, test_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get A/B test variant for a user/context"""
        if test_name not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_name]
        if not test_config.enabled:
            return None
        
        # Check if user is in test traffic
        user_id = context.get('user_id', 'default') if context else 'default'
        import hashlib
        
        # Traffic percentage check
        hash_input = f"{test_name}_traffic_{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        traffic_percentage = (hash_value % 100) + 1
        
        if traffic_percentage > test_config.traffic_percentage:
            return None  # User not in test traffic
        
        # Variant assignment
        hash_input = f"{test_name}_variant_{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = hash_value % 100
        
        cumulative_weight = 0
        for variant_name, weight in test_config.variants.items():
            cumulative_weight += weight * 100
            if percentage < cumulative_weight:
                return variant_name
        
        # Fallback to first variant
        return list(test_config.variants.keys())[0]
    
    def update_flag(self, flag_name: str, enabled: bool, rollout_percentage: Optional[float] = None):
        """Update a feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = enabled
            self.flags[flag_name].updated_at = datetime.now().isoformat()
            
            if rollout_percentage is not None:
                self.flags[flag_name].rollout_percentage = rollout_percentage
            
            self.save_config()
            logger.info(f"Updated feature flag {flag_name}: enabled={enabled}")
        else:
            logger.warning(f"Cannot update unknown flag: {flag_name}")
    
    def get_embedding_model_priority(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get embedding model priority based on feature flags and A/B tests"""
        priority_list = []
        
        # Check A/B test first
        ab_variant = self.get_ab_test_variant("embedding_model_comparison", context)
        if ab_variant:
            if ab_variant == "qodo_embed" and self.is_enabled("use_qodo_embed", context):
                priority_list = ["qodo", "openai", "onnx"]
            elif ab_variant == "openai_hybrid" and self.is_enabled("use_openai_embeddings", context):
                priority_list = ["openai", "qodo", "onnx"]
            elif ab_variant == "onnx_baseline":
                priority_list = ["onnx"]
        
        # Fallback to feature flag-based priority
        if not priority_list:
            if self.is_enabled("use_qodo_embed", context):
                priority_list.append("qodo")
            if self.is_enabled("use_openai_embeddings", context):
                priority_list.append("openai")
            priority_list.append("onnx")  # Always include ONNX as fallback
        
        return priority_list
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature flag statistics"""
        enabled_flags = sum(1 for flag in self.flags.values() if flag.enabled)
        active_ab_tests = sum(1 for test in self.ab_tests.values() if test.enabled)
        
        return {
            'total_flags': len(self.flags),
            'enabled_flags': enabled_flags,
            'active_ab_tests': active_ab_tests,
            'config_path': str(self.config_path),
            'flags': {name: {'enabled': flag.enabled, 'rollout': flag.rollout_percentage} 
                     for name, flag in self.flags.items()},
            'ab_tests': {name: {'enabled': test.enabled, 'variants': len(test.variants)}
                        for name, test in self.ab_tests.items()}
        }

# Global instance
_feature_manager = None

def get_feature_manager() -> FeatureFlagManager:
    """Get or create global feature flag manager"""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureFlagManager()
    return _feature_manager

def is_enabled(flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to check if feature is enabled"""
    return get_feature_manager().is_enabled(flag_name, context)

def get_ab_variant(test_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Convenience function to get A/B test variant"""
    return get_feature_manager().get_ab_test_variant(test_name, context)

def get_model_priority(context: Optional[Dict[str, Any]] = None) -> List[str]:
    """Convenience function to get embedding model priority"""
    return get_feature_manager().get_embedding_model_priority(context)

if __name__ == "__main__":
    # Test the feature flag system
    print("🚩 Testing Feature Flag System")
    print("=" * 40)
    
    manager = get_feature_manager()
    
    print("Feature Flags:")
    stats = manager.get_stats()
    for flag_name, flag_info in stats['flags'].items():
        status = "✅" if flag_info['enabled'] else "❌"
        print(f"  {status} {flag_name}: {flag_info['rollout']}% rollout")
    
    print(f"\nA/B Tests:")
    for test_name, test_info in stats['ab_tests'].items():
        status = "🧪" if test_info['enabled'] else "💤"
        print(f"  {status} {test_name}: {test_info['variants']} variants")
    
    print(f"\nModel Priority: {get_model_priority()}")
    print(f"A/B Variant: {get_ab_variant('embedding_model_comparison')}")