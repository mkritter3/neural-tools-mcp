#!/usr/bin/env python3
import sys, json, re, pathlib


hook = json.load(sys.stdin)
if hook.get("stop_hook_active"):
sys.exit(0)


# Load config
cfg = {"mock_policy": {"require_flagging": True, "forbid_prod_fallback": True, "prod_env_keys": [], "trigger_terms": ["mock","fake","stub","simulated","test double"]}}
try:
import os, json as _json, pathlib as _p
p = _p.Path(__file__).with_name("config.json")
cfg.update(_json.loads(p.read_text()))
except Exception:
pass


# Fetch last assistant message
last = ""
try:
p = pathlib.Path(hook["transcript_path"]) # ndjson lines
for line in p.read_text().splitlines()[::-1]:
if '"role":"assistant"' in line:
last = line.lower()
break
except Exception:
pass


issues = []
trigs = cfg["mock_policy"]["trigger_terms"]
if any(t in last for t in trigs):
# Require an explicit flagging line
if cfg["mock_policy"].get("require_flagging", True) and "mock usage:" not in last:
issues.append("Add a 'Mock Usage:' line specifying scope (unit/integration/local), data provenance, and gaps vs prod.")
# Forbid suggesting prod fallbacks to mocks
bad_phrases = ["fallback to mock", "gracefully fallback to mock", "use mock in production", "use a mock in prod"]
if any(bp in last for bp in bad_phrases):
issues.append("Do not propose or allow mock fallbacks in prod/staging; fail fast and alert instead.")


if issues:
sys.stderr.write("Revise with Mocking Policy rules:
- " + "
- ".join(issues))
sys.exit(2)


sys.exit(0)