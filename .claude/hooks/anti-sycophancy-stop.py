#!/usr/bin/env python3
import sys, json, re, pathlib


hook = json.load(sys.stdin)
if hook.get("stop_hook_active"):
sys.exit(0) # avoid loops


# Load config if present
cfg = {
"require_confidence": True,
"require_assumptions": True,
"hedge_regex": r"\\b(i think|maybe|it seems|i feel|as an ai)\\b",
"needs_citation_triggers": [
"ceo","price","version","release","benchmark","law","election",
"api change","deprecation","breaking change","security cve","advisory"
]
}
try:
cfg_path = pathlib.Path(__file__).with_name("config.json")
cfg.update(json.loads(cfg_path.read_text()))
except Exception:
pass


# Pull last assistant message from transcript (ndjson lines)
last = ""
try:
p = pathlib.Path(hook["transcript_path"])
for line in p.read_text().splitlines()[::-1]:
if '"role":"assistant"' in line:
last = line.lower()
break
except Exception:
pass


issues = []
if re.search(cfg["hedge_regex"], last):
issues.append("Replace hedging with a clear claim + numeric uncertainty.")
if cfg.get("require_confidence", True) and "confidence:" not in last:
issues.append("Add final 'Confidence: NN%' with 1–2 assumptions.")
if any(k in last for k in cfg["needs_citation_triggers"]) and not ("cite" in last or "http" in last or "citation" in last):
issues.append("Add citations for time-sensitive or niche facts.")


if issues:
sys.stderr.write("Revise with Truth-First L9 rules:\n- " + "\n- ".join(issues))
sys.exit(2) # block and send feedback for self-repair


sys.exit(0)