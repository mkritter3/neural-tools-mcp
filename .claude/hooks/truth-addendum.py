#!/usr/bin/env python3
import sys, json
_ = json.load(sys.stdin) # hook payload not used here
msg = (
"### Truth-First L9 Addendum\n"
"- Prioritize correctness over agreeableness; challenge incorrect premises.\n"
"- Provide evidence or citations for time-sensitive/niche claims.\n"
"- End with 'Confidence: NN%' and 1–2 key assumptions.\n"
)
sys.stdout.write(msg)