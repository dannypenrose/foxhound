#!/usr/bin/env python3
"""Stage 7: Export analysis — structured, evidence-foregrounded output.

Formats analysis results for easy copy-paste into any LLM for
report writing. Foregrounds direct quotes and source references,
with AI observations clearly labelled as analytical aids.

Usage:
  python export.py analysis_output.md                      # Display formatted output
  python export.py analysis_output.md --clipboard          # Copy to clipboard
  python export.py analysis_output.md --output report.md   # Save to file
  python export.py results.json --raw                      # Export raw evidence package
  python export.py results.json --raw --clipboard          # Raw evidence → clipboard
"""

import json
import subprocess
import sys
from pathlib import Path


def format_raw_evidence(documents: list[dict]) -> str:
    """Format raw documents into a structured evidence package."""
    lines = [
        "# Evidence Package for Analysis",
        "",
        f"**Total Documents:** {len(documents)}",
        "",
        "---",
        "",
    ]

    # Group by date
    by_date = {}
    for doc in documents:
        date = doc.get("date", "unknown")
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(doc)

    # Build timeline
    lines.append("## Chronological Evidence")
    lines.append("")

    for date in sorted(by_date.keys()):
        docs = by_date[date]
        lines.append(f"### {date}")
        lines.append("")

        for doc in docs:
            sender = doc.get("sender", "unknown")
            subject = doc.get("subject", "(no subject)")
            recipient = doc.get("recipient", "")
            cc = doc.get("cc", "")
            folder = doc.get("folder", "inbox")
            body = doc.get("text_full", doc.get("text_preview", ""))

            lines.append(f"**From:** {sender}")
            if recipient:
                lines.append(f"**To:** {recipient}")
            if cc and cc != "[]":
                lines.append(f"**CC:** {cc}")
            lines.append(f"**Subject:** {subject}")
            lines.append(f"**Folder:** {folder}")

            # Include triage info if available
            triage = doc.get("triage", {})
            if triage and triage.get("relevance_score"):
                lines.append(f"**Relevance Score:** {triage['relevance_score']}/10 *(triage estimate)*")
                if triage.get("category"):
                    lines.append(f"**Category:** {triage['category']}")

            lines.append("")
            lines.append("```")
            lines.append(body[:2000] if body else "(empty)")
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Participant summary
    lines.append("## Participants")
    lines.append("")
    participants = {}
    for doc in documents:
        sender = doc.get("sender", "")
        if sender:
            if sender not in participants:
                participants[sender] = {"sent": 0, "received": 0, "cc": 0}
            participants[sender]["sent"] += 1

        recipients = doc.get("recipient", "[]")
        if isinstance(recipients, str):
            try:
                recipients = json.loads(recipients)
            except json.JSONDecodeError:
                recipients = []
        for r in recipients:
            if r:
                if r not in participants:
                    participants[r] = {"sent": 0, "received": 0, "cc": 0}
                participants[r]["received"] += 1

    lines.append("| Participant | Sent | Received | CC'd |")
    lines.append("|---|---|---|---|")
    for p, counts in sorted(participants.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]:
        lines.append(f"| {p} | {counts['sent']} | {counts['received']} | {counts['cc']} |")
    lines.append("")

    return "\n".join(lines)


def format_analysis_for_llm(analysis_text: str) -> str:
    """Wrap analysis output with instructions for LLM report writing."""
    return (
        "# Document Analysis — For Report Writing\n\n"
        "Below is a structured analysis of document evidence produced by an automated pipeline.\n"
        "All findings are based on direct quotes and metadata from the source documents.\n"
        "Items marked as 'ANALYTICAL NOTE' are AI-generated observations for your consideration,\n"
        "not standalone evidence.\n\n"
        "---\n\n"
        f"{analysis_text}\n\n"
        "---\n\n"
        "## Instructions for Report\n\n"
        "Please use the above evidence to produce a structured report. Options:\n\n"
        "1. **Chronological summary** — timeline with citations\n"
        "2. **Thematic analysis** — key themes with supporting evidence\n"
        "3. **Pattern analysis** — behavioral or procedural patterns\n"
        "4. **Executive summary** — high-level findings and recommendations\n"
        "5. **Custom format** — describe what you need\n\n"
        "For each finding, cite the specific document date, source, and relevant quote.\n"
    )


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard (macOS)."""
    try:
        process = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python export.py <analysis_output.md|results.json> [options]")
        print()
        print("Options:")
        print("  --clipboard    Copy to clipboard (macOS)")
        print("  --output FILE  Save to file")
        print("  --raw          Format as raw evidence package (for JSON input)")
        return

    input_file = sys.argv[1]
    to_clipboard = "--clipboard" in sys.argv
    raw_mode = "--raw" in sys.argv
    output_file = None

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_file = sys.argv[idx + 1]

    input_path = Path(input_file)

    if raw_mode or input_path.suffix == ".json":
        # Format raw evidence from JSON
        with open(input_path) as f:
            documents = json.load(f)
        formatted = format_raw_evidence(documents)
    else:
        # Wrap existing analysis
        analysis_text = input_path.read_text(encoding="utf-8")
        formatted = format_analysis_for_llm(analysis_text)

    # Character and token counts
    char_count = len(formatted)
    token_estimate = char_count // 4

    print(f"\n  Export Summary")
    print(f"  {'─'*40}")
    print(f"  Characters:    {char_count:,}")
    print(f"  Est. tokens:   {token_estimate:,}")
    print(f"  Source:         {input_file}")

    if to_clipboard:
        if copy_to_clipboard(formatted):
            print(f"  Copied to clipboard!")
            print(f"\n  Paste into your preferred LLM for report writing.")
        else:
            print(f"  ERROR: Could not copy to clipboard.")
            print(f"  Content saved to export_output.md instead.")
            output_file = output_file or "export_output.md"

    if output_file:
        Path(output_file).write_text(formatted, encoding="utf-8")
        print(f"  Saved to:      {output_file}")

    if not to_clipboard and not output_file:
        # Print to stdout
        print(f"\n{'='*60}")
        print(formatted)
        print(f"{'='*60}")
        print(f"\n  To copy: python export.py {input_file} --clipboard")
        print(f"  To save: python export.py {input_file} --output report.md")


if __name__ == "__main__":
    main()
