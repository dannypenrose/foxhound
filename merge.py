#!/usr/bin/env python3
"""Merge multiple evidence JSON files into one deduplicated set.

Combines results from different semantic searches, keeping the highest
relevance score per unique email. No emails are lost.

Usage:
  python merge.py evidence1.json evidence2.json evidence3.json --output merged.json
  python merge.py evidence/*.json --output merged.json
  python merge.py evidence/*.json --output merged.json --min-score 0.25
"""

import json
import sys
from pathlib import Path


def merge_evidence(files: list[str], min_score: float = 0.0) -> list[dict]:
    """Merge multiple evidence JSON files, deduplicating by message_id."""
    seen: dict[str, dict] = {}  # message_id → best result

    total_loaded = 0
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue

        with open(path) as f:
            documents = json.load(f)

        print(f"  {path.name}: {len(documents)} results")
        total_loaded += len(documents)

        for doc in documents:
            # Use message_id as primary key, fall back to source_file
            key = doc.get("message_id", "") or doc.get("source_file", "")
            if not key:
                continue

            score = float(doc.get("relevance_score", 0) or 0)

            if min_score and score < min_score:
                continue

            if key not in seen or score > float(seen[key].get("relevance_score", 0) or 0):
                seen[key] = doc

    # Sort by relevance score (highest first)
    merged = sorted(seen.values(), key=lambda r: float(r.get("relevance_score", 0) or 0), reverse=True)

    print(f"\n  Total loaded:  {total_loaded}")
    print(f"  After dedup:   {len(merged)}")
    print(f"  Duplicates:    {total_loaded - len(merged)}")

    return merged


def main():
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print("Usage: python merge.py file1.json file2.json ... --output merged.json")
        print()
        print("Options:")
        print("  --output FILE     Output file (default: merged_evidence.json)")
        print("  --min-score N     Minimum relevance score to include (e.g. 0.25)")
        return

    # Parse args
    output_path = "merged_evidence.json"
    min_score = 0.0
    files = []

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        elif argv[i] == "--min-score" and i + 1 < len(argv):
            min_score = float(argv[i + 1])
            i += 2
        else:
            files.append(argv[i])
            i += 1

    if not files:
        print("  No input files specified.")
        return

    print(f"\n  Merging {len(files)} evidence files...")
    merged = merge_evidence(files, min_score)

    if not merged:
        print("  No results to write.")
        return

    # Stats
    all_dates = sorted(set(r.get("date", "") for r in merged if r.get("date")))
    if all_dates:
        print(f"  Date range:    {all_dates[0]} → {all_dates[-1]}")

    scores = [float(r.get("relevance_score", 0) or 0) for r in merged]
    if scores:
        print(f"  Score range:   {min(scores):.3f} → {max(scores):.3f}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, default=str)

    print(f"\n  Saved to: {output_path}")
    print(f"  Ready for: uv run python analyze.py {output_path} --dry-run")


if __name__ == "__main__":
    main()
