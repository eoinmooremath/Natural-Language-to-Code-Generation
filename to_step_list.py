#!/usr/bin/env python3
"""
Convert newline/semicolon-chained robot actions into a list.

- Splits on:
  * '\n'
  * ';'
  * '\n;'  (explicitly normalized)
- Trims whitespace and trailing periods.
- Keeps only lines that look like robot.* calls.
- Preserves original "output" and adds "output_list".
- If "output" is already a list, it is normalized and placed in "output_list".

Usage:
  python to_step_list.py /path/robot_full_dataset.json /path/robot_full_dataset.list.json
"""

import json
import re
import sys
from pathlib import Path
from typing import List

SEP_PATTERN = re.compile(r"[;\n]+")

def normalize_line(line: str) -> str:
    """Strip whitespace, drop trailing periods, and ensure startswith 'robot.' if possible."""
    ln = line.strip()
    # drop trailing period(s)
    ln = re.sub(r"\.*\s*$", "", ln)
    # best-effort rescue if the line contains a robot.*(...) call but extra text around it
    if not ln.startswith("robot."):
        m = re.search(r"(robot\.[A-Za-z0-9_\.]+\s*\([^()]*\))", ln)
        if m:
            ln = m.group(1).strip()
    return ln

def split_actions(text: str) -> List[str]:
    """Split by separators, normalize, and filter."""
    if not text:
        return []
    # normalize the explicit combo '\n;'
    text = text.replace("\n;", "\n")
    # split by any run of ; or \n
    parts = [normalize_line(p) for p in SEP_PATTERN.split(text)]
    steps = [p for p in parts if p and p.startswith("robot.")]
    return steps

def main(inp: Path, outp: Path):
    data = json.loads(inp.read_text())
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {input, output, ...} objects.")

    converted = 0
    kept = 0
    out_rows = []

    for rec in data:
        rec = dict(rec)  # shallow copy
        out_field = rec.get("output", "")

        if isinstance(out_field, list):
            # normalize list entries
            steps = []
            for item in out_field:
                if isinstance(item, str):
                    for s in split_actions(item):
                        steps.append(s)
            # de-duplicate consecutive blanks
            steps = [s for s in (normalize_line(x) for x in steps) if s]
            rec["output_list"] = steps
            kept += 1
        elif isinstance(out_field, str):
            steps = split_actions(out_field)
            rec["output_list"] = steps
            converted += 1
        else:
            # unknown type; just add empty list
            rec["output_list"] = []

        out_rows.append(rec)

    outp.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2))
    print(f"✅ Wrote: {outp}")
    print(f"   Converted string outputs -> list: {converted}")
    print(f"   Normalized pre-existing lists: {kept}")
    print(f"   Total records: {len(out_rows)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python to_step_list.py /path/robot_full_dataset.json /path/robot_full_dataset.list.json")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
