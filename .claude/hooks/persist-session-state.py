#!/usr/bin/env python3
# Persist session state into memory/session-sticky.md
# Adds: Conversation summary, MCP/tool exchanges, and continuation/thread/run IDs.

import sys, os, json, re, datetime
from collections import deque

# ---------- IO helpers ----------

def read_stdin_json():
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"[persist-session-state] Failed to parse stdin JSON: {e}", file=sys.stderr)
        return {}

def load_transcript_lines(jsonl_path, max_bytes=6_000_000):
    if not jsonl_path or not os.path.exists(jsonl_path):
        return []
    try:
        size = os.path.getsize(jsonl_path)
        start = 0
        if size > max_bytes:
            start = size - max_bytes
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
            if start:
                f.seek(start)
                f.readline()
            return f.readlines()
    except Exception as e:
        print(f"[persist-session-state] Failed reading transcript: {e}", file=sys.stderr)
        return []

# ---------- parsing helpers ----------

def extract_text_from_msg(obj):
    if not isinstance(obj, dict):
        return ""
    if isinstance(obj.get("text"), str):
        return obj["text"]
    content = obj.get("content")
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
        if parts:
            return "\n".join(parts)
    if isinstance(obj.get("message"), str):
        return obj["message"]
    return ""

def role_of(obj):
    # Common fields across transcripts
    for k in ("role", "author", "speaker"):
        if k in obj and isinstance(obj[k], str):
            return obj[k].lower()
    # Fallback guesses
    if obj.get("type") in ("assistant", "user", "system"):
        return obj.get("type")
    return "unknown"

def is_tool_event(obj):
    # Heuristics to detect MCP / tool calls
    if isinstance(obj.get("type"), str) and obj["type"].lower() in {"tool_call", "tool_result", "mcp_call", "mcp_result"}:
        return True
    if any(k in obj for k in ("tool_name", "mcp", "call", "toolCallId", "tool_call_id")):
        return True
    # Sometimes tool traces appear under content blocks:
    content = obj.get("content")
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and (c.get("type") in {"tool_use", "tool_result"} or "mcp" in c):
                return True
    return False

def get_tool_brief(obj, max_chars=220):
    name = obj.get("tool_name") or obj.get("name") or "tool"
    call_id = obj.get("tool_call_id") or obj.get("call_id") or obj.get("invocation_id") or obj.get("id")
    payload = extract_text_from_msg(obj)
    if not payload:
        # Try nested payloads commonly seen in tool calls
        for k in ("arguments", "result", "data", "output"):
            v = obj.get(k)
            if isinstance(v, (dict, list)):
                try:
                    payload = json.dumps(v)[:max_chars]
                except Exception:
                    payload = str(v)[:max_chars]
            elif isinstance(v, str):
                payload = v[:max_chars]
            if payload:
                break
    payload = (payload or "").strip().replace("\n", " ")
    if len(payload) > max_chars:
        payload = payload[:max_chars] + "…"
    base = f"{name}"
    if call_id:
        base += f"#{call_id}"
    return f"{base}: {payload}" if payload else base

def harvest_items(text, pattern, min_len=3):
    out = []
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            cleaned = line.strip()
            if len(cleaned) >= min_len and cleaned not in out:
                out.append(cleaned)
    return out

def extract_state(transcript_lines, tail_messages=120):
    # Parse JSONL → recent objects
    msgs = []
    for ln in transcript_lines[-4000:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            msgs.append(obj)
        except Exception:
            continue

    recent = msgs[-tail_messages:]

    # Flatten text
    buf = []
    for m in recent:
        buf.append(extract_text_from_msg(m))
    text = "\n".join([b for b in buf if b])

    # TODOs
    todo_pat = re.compile(r"^(\s*[-*]\s+\[ \]\s+.+|.*\bTODO\b[: ]?.+)$", re.IGNORECASE)
    todos = harvest_items(text, todo_pat)

    # Decisions
    decision_pat = re.compile(r"^(##?\s*Decisions?.*|.*\bDecision(s)?\b[:\-].+|.*\b(We|I)\s+(decided|chose)\b.+)$", re.IGNORECASE)
    decisions = harvest_items(text, decision_pat)

    # Current task
    task_pat = re.compile(r"^(##?\s*(Current\s+)?Task|Goal)\b[:\-]\s*(.+)$", re.IGNORECASE)
    current_task = None
    for line in reversed(text.splitlines()):
        m = task_pat.search(line.strip())
        if m:
            current_task = m.group(0).strip()
            break
    if not current_task:
        for m in reversed(recent):
            if role_of(m) == "user":
                t = extract_text_from_msg(m).strip()
                if t:
                    current_task = f"Task (inferred from last user prompt): {t[:240]}..."
                    break

    # Conversation summary: compact extract of the last ~24 turns
    convo = deque(maxlen=24)
    for m in recent:
        r = role_of(m)
        t = extract_text_from_msg(m).strip()
        if not t:
            continue
        # strip long blocks
        t_one = re.sub(r"\s+", " ", t)[:280]
        if len(t_one) == 280:
            t_one += "…"
        if r in ("user", "assistant", "system"):
            convo.append(f"- **{r.capitalize()}**: {t_one}")

    # MCP/tool exchanges: last 10 tool events
    mcp = deque(maxlen=10)
    for m in recent:
        if is_tool_event(m):
            mcp.append(f"- {get_tool_brief(m)}")

    # Continuation / thread / run IDs: best-effort scan of recent messages
    ids = set()
    keys = ("continuation_id", "thread_id", "run_id", "conversation_id", "parent_id", "tool_call_id")
    for m in reversed(recent):
        for k in keys:
            v = m.get(k)
            if isinstance(v, str) and len(v) >= 8:
                ids.add(f"{k}={v}")
        # sometimes nested under metadata or context
        for k in ("metadata", "context", "trace", "tool"):
            v = m.get(k)
            if isinstance(v, dict):
                for kk in keys:
                    vv = v.get(kk)
                    if isinstance(vv, str) and len(vv) >= 8:
                        ids.add(f"{kk}={vv}")

    convo_summary = "\n".join(convo) if convo else "_(none found)_"
    mcp_log = "\n".join(mcp) if mcp else "_(none found)_"
    cont_ids = "\n".join(sorted(ids)) if ids else "_(none found)_"

    return current_task, todos, decisions, convo_summary, mcp_log, cont_ids

# ---------- formatting / writing ----------

def ensure_dir(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

def replace_block(text, marker, new_block):
    start_tag = f"<!-- START:{marker} -->"
    end_tag   = f"<!-- END:{marker} -->"
    if start_tag in text and end_tag in text:
        pre, rest = text.split(start_tag, 1)
        _, post = rest.split(end_tag, 1)
        return pre + start_tag + "\n" + new_block + "\n" + end_tag + post
    else:
        return text + f"\n\n{start_tag}\n{new_block}\n{end_tag}\n"

def format_md_snapshot(now, trigger, current_task, todos, decisions, convo_summary, mcp_log, cont_ids):
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    inv_text = "- Keep project invariants and rules as defined in CLAUDE.md and imports."

    todo_text = "\n".join([f"- [ ] {re.sub(r'^[-*]\\s*\\[ \\]\\s*', '', t, flags=re.I)}" for t in todos]) or "_(none found)_"
    dec_text  = "\n".join([f"- {t}" for t in decisions]) or "_(none found)_"

    meta = []
    meta.append(f"**Last capture:** {ts}  \n**Trigger:** `{trigger}`")
    if current_task:
        meta.append(f"**Current task:** {current_task}")
    meta_body = "\n\n".join(meta)

    return inv_text, todo_text, dec_text, meta_body, convo_summary, mcp_log, cont_ids

def main():
    hook = read_stdin_json()
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or hook.get("cwd") or os.getcwd()
    transcript_path = hook.get("transcript_path")
    trigger = hook.get("trigger", "unknown")

    lines = load_transcript_lines(transcript_path)
    (current_task, todos, decisions,
     convo_summary, mcp_log, cont_ids) = extract_state(lines)

    now = datetime.datetime.now()
    (inv_text, todo_text, dec_text, meta_body,
     convo_text, mcp_text, cont_text) = format_md_snapshot(
        now, trigger, current_task, todos, decisions, convo_summary, mcp_log, cont_ids
     )

    sticky_path = os.path.join(project_dir, "memory", "session-sticky.md")
    ensure_dir(sticky_path)
    if os.path.exists(sticky_path):
        with open(sticky_path, "r", encoding="utf-8", errors="ignore") as f:
            existing = f.read()
    else:
        existing = "# Session Sticky State\n\nThis file is auto-maintained by the PreCompact hook.\n"

    updated = existing
    updated = replace_block(updated, "INVARIANTS", inv_text)
    updated = replace_block(updated, "ACTIVE_TODOS", todo_text)
    updated = replace_block(updated, "RECENT_DECISIONS", dec_text)
    updated = replace_block(updated, "CONVO_SUMMARY", convo_text)
    updated = replace_block(updated, "MCP_LOG", mcp_text)
    updated = replace_block(updated, "CONTINUATIONS", cont_text)
    updated = replace_block(updated, "SNAPSHOT_META", meta_body)

    with open(sticky_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"[persist-session-state] Wrote {sticky_path} "
          f"(TODOs={len(todos)}, decisions={len(decisions)}).")

if __name__ == "__main__":
    main()
