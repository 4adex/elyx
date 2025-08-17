# Python script to simplify the journey JSON into per-week combined conversation histories.
# This will be executed and the output shown below as a demonstration.
# Save this snippet as `simplify_journey.py` for reuse; it reads JSON from a file or stdin and writes simplified JSON to stdout or a file.
#
# Usage (command line):
#   python simplify_journey.py --in journey.json --out simplified.json
#   cat journey.json | python simplify_journey.py          # reads from stdin, writes to stdout
#
# Behavior:
# - Collects all `conversation_history` entries from every conversation in each week.
# - Produces a dict with keys like "week_1", "week_2", ... where values are lists of conversation-history dicts
# - If a week_number is missing, it will fall back to the week's index (1-based).

import json
import argparse
import sys
from typing import Any, Dict, List

def simplify_journey(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert the input journey JSON (with `weekly_conversations`) into a simplified mapping:
      { "week_1": [<conversation_history entries...>], "week_2": [...], ... }
    Each entry in the lists is one dict from the original `conversation_history` (e.g. {'sender','role','message'}).
    """
    simplified = {}
    weekly = data.get("weekly_conversations", []) or []
    for idx, week in enumerate(weekly, start=1):
        meta = week.get("week_metadata", {})
        week_number = meta.get("week_number") or idx
        key = f"week_{week_number}"
        combined = []
        for conv in week.get("conversations", []):
            # Each conversation may have a `conversation_history` which is a list of message dicts
            history = conv.get("conversation_history", []) or []
            for entry in history:
                # Keep the entry as-is. If you prefer only message text, replace `entry` with entry.get("message")
                combined.append(entry)
        simplified[key] = combined
    return simplified

def main():
    parser = argparse.ArgumentParser(description="Simplify a journey JSON into combined per-week conversation histories.")
    parser.add_argument("--in", dest="infile", help="Input JSON file (default: stdin)", default=None)
    parser.add_argument("--out", dest="outfile", help="Output JSON file (default: stdout)", default=None)
    args = parser.parse_args()

    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    simplified = simplify_journey(data)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        print(f"Simplified JSON written to {args.outfile}")
    else:
        json.dump(simplified, sys.stdout, indent=2, ensure_ascii=False)
        print()  # newline after JSON output

main()
