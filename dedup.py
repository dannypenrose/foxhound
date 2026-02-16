#!/usr/bin/env python3
"""Stage 0: Email deduplication & preprocessing.

Strips quoted/forwarded text from reply chains, groups emails into threads,
and writes deduplicated content to rag_ready/ for ingestion.

Two-tier storage:
  - Original .eml files are preserved untouched (evidence tier)
  - rag_ready/ contains only unique content per message (ingestion tier)

Usage:
  python dedup.py                          # Use paths from config.yaml
  python dedup.py --config config.yaml     # Explicit config
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from parsers.email_parser import parse_eml


def build_threads(emails: list[dict]) -> dict[str, list[dict]]:
    """Group emails into threads using Message-ID / In-Reply-To / References."""
    # Detect Message-ID collisions across folders (same email in multiple sources)
    seen_ids: dict[str, list[dict]] = defaultdict(list)
    for e in emails:
        if e["message_id"]:
            seen_ids[e["message_id"]].append(e)

    cross_folder_dupes = {
        mid: entries for mid, entries in seen_ids.items()
        if len(entries) > 1 and len({e["folder"] for e in entries}) > 1
    }
    same_folder_dupes = {
        mid: entries for mid, entries in seen_ids.items()
        if len(entries) > 1 and len({e["folder"] for e in entries}) == 1
    }

    if cross_folder_dupes:
        print(f"\n  NOTICE: {len(cross_folder_dupes)} email(s) found in multiple folders (duplicates in output):")
        for mid, entries in list(cross_folder_dupes.items())[:10]:
            folders = [e["folder"] for e in entries]
            subject = entries[0].get("subject", "(no subject)")[:60]
            print(f"    - {subject} | folders: {folders} | Message-ID: {mid[:50]}")
        if len(cross_folder_dupes) > 10:
            print(f"    ... and {len(cross_folder_dupes) - 10} more")

    if same_folder_dupes:
        print(f"\n  NOTICE: {len(same_folder_dupes)} duplicate Message-ID(s) within same folder:")
        for mid, entries in list(same_folder_dupes.items())[:5]:
            print(f"    - {entries[0].get('subject', '(no subject)')[:60]} | count: {len(entries)}")

    id_to_email = {e["message_id"]: e for e in emails if e["message_id"]}
    thread_map = defaultdict(list)

    def find_thread_root(msg_id: str, visited: set | None = None) -> str:
        if visited is None:
            visited = set()
        if msg_id in visited:
            return msg_id
        visited.add(msg_id)

        em = id_to_email.get(msg_id)
        if not em:
            return msg_id
        if em["in_reply_to"] and em["in_reply_to"] in id_to_email:
            return find_thread_root(em["in_reply_to"], visited)
        if em["references"]:
            return em["references"][0]
        return msg_id

    # Also group by Thread-Topic as fallback for Outlook emails
    topic_to_root = {}

    for em in emails:
        if em["message_id"]:
            root_id = find_thread_root(em["message_id"])
        elif em.get("thread_topic"):
            # Fallback: use Thread-Topic header for threading
            topic = em["thread_topic"].strip().lower()
            if topic not in topic_to_root:
                topic_to_root[topic] = f"topic_{hashlib.md5(topic.encode()).hexdigest()[:12]}"
            root_id = topic_to_root[topic]
        else:
            # Standalone email — its own thread
            root_id = hashlib.md5(em["source_file"].encode()).hexdigest()[:12]

        thread_id = hashlib.md5(root_id.encode()).hexdigest()[:12]
        thread_map[thread_id].append(em)

    # Sort each thread chronologically
    for tid in thread_map:
        thread_map[tid].sort(key=lambda e: e.get("date", ""))

    return dict(thread_map)


def write_rag_ready(threads: dict, output_dir: Path):
    """Write deduplicated thread files to rag_ready/ directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"threads": 0, "messages": 0, "skipped_empty": 0}

    for thread_id, emails in threads.items():
        thread_dir = output_dir / f"thread_{thread_id}"
        thread_dir.mkdir(exist_ok=True)

        thread_metadata = {
            "thread_id": thread_id,
            "subject": emails[0].get("subject", ""),
            "participants": list({e["sender"] for e in emails if e["sender"]}),
            "all_recipients": list({r for e in emails for r in e.get("recipient", []) if r}),
            "cc_participants": list({c for e in emails for c in e.get("cc", []) if c}),
            "date_range": {
                "first": emails[0].get("date", ""),
                "last": emails[-1].get("date", ""),
            },
            "message_count": len(emails),
            "folder": emails[0].get("folder", "inbox"),
            "messages": [],
        }

        for idx, em in enumerate(emails, 1):
            if not em["body_new"].strip():
                stats["skipped_empty"] += 1
                continue

            # Safe filename
            date_part = em.get("date", "unknown")[:10]
            sender_part = em["sender"].split("@")[0] if em["sender"] else "unknown"
            sender_part = sender_part.replace(".", "_")[:20]
            filename = f"{idx:03d}_{date_part}_{sender_part}.txt"

            (thread_dir / filename).write_text(em["body_new"], encoding="utf-8")

            thread_metadata["messages"].append({
                "index": idx,
                "file": filename,
                "from": em["sender"],
                "from_name": em.get("sender_name", ""),
                "to": em.get("recipient", []),
                "cc": em.get("cc", []),
                "date": em.get("date", ""),
                "date_raw": em.get("date_raw", ""),
                "subject": em.get("subject", ""),
                "message_id": em["message_id"],
                "has_attachments": em.get("has_attachments", False),
                "attachment_names": em.get("attachment_names", []),
                "folder": em.get("folder", "inbox"),
                "source_file": em.get("source_file", ""),
            })
            stats["messages"] += 1

        (thread_dir / "metadata.json").write_text(
            json.dumps(thread_metadata, indent=2, default=str),
            encoding="utf-8",
        )
        stats["threads"] += 1

    return stats


def main():
    # Load config
    config_path = Path("config.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = Path(sys.argv[idx + 1])

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["deduplication"]["rag_ready_path"])

    # Collect all email source directories from config
    sources = config["ingestion"]["sources"]
    email_sources = [s for s in sources if s["type"] == "email"]

    all_emails = []
    total_files = 0

    for source in email_sources:
        source_path = Path(source["path"]).expanduser()
        folder = source.get("folder", "inbox")

        if not source_path.exists():
            print(f"  WARNING: Source path does not exist: {source_path}")
            continue

        eml_files = list(source_path.rglob("*.eml"))
        total_files += len(eml_files)
        print(f"Parsing {len(eml_files)} .eml files from {source_path} (folder: {folder})...")

        for i, eml_file in enumerate(eml_files, 1):
            if i % 500 == 0:
                print(f"  Parsed {i}/{len(eml_files)}...")
            try:
                parsed = parse_eml(eml_file, folder=folder)
                all_emails.append(parsed)
            except Exception as e:
                print(f"  ERROR parsing {eml_file.name}: {e}")

    print(f"\nTotal parsed: {len(all_emails)} emails from {total_files} files")

    print("Building threads...")
    threads = build_threads(all_emails)
    print(f"  Found {len(threads)} threads")

    print(f"Writing deduplicated output to {output_dir}...")
    stats = write_rag_ready(threads, output_dir)

    # Calculate reduction
    total_raw = sum(len(e["body_full"]) for e in all_emails)
    total_dedup = sum(len(e["body_new"]) for e in all_emails)
    reduction = (1 - total_dedup / total_raw) * 100 if total_raw else 0

    # Per-folder breakdown
    folder_counts = defaultdict(int)
    for e in all_emails:
        folder_counts[e.get("folder", "unknown")] += 1

    # Standalone emails (single-message threads) — important for sent-no-reply visibility
    standalone_threads = {tid: msgs for tid, msgs in threads.items() if len(msgs) == 1}
    standalone_by_folder = defaultdict(int)
    for msgs in standalone_threads.values():
        standalone_by_folder[msgs[0].get("folder", "unknown")] += 1

    # Integrity check: every parsed email must appear in exactly one thread
    emails_in_threads = sum(len(msgs) for msgs in threads.values())
    integrity_ok = emails_in_threads == len(all_emails)

    print(f"\n{'='*50}")
    print(f"DEDUPLICATION COMPLETE")
    print(f"{'='*50}")
    print(f"  Emails parsed:      {len(all_emails):,}")
    print(f"  Threads created:    {stats['threads']:,}")
    print(f"  Messages written:   {stats['messages']:,}")
    print(f"  Empty skipped:      {stats['skipped_empty']:,}")
    print(f"  Content reduction:  {reduction:.0f}% ({total_raw:,} → {total_dedup:,} chars)")
    print(f"  Output directory:   {output_dir}")

    print(f"\n  Per-folder breakdown:")
    for folder, count in sorted(folder_counts.items()):
        print(f"    {folder}: {count:,} emails")

    print(f"\n  Standalone emails (single-message threads): {len(standalone_threads):,}")
    for folder, count in sorted(standalone_by_folder.items()):
        print(f"    {folder}: {count:,}")

    print(f"\n  INTEGRITY CHECK: {'PASS' if integrity_ok else 'FAIL'} "
          f"({emails_in_threads:,} in threads / {len(all_emails):,} parsed)")
    if not integrity_ok:
        print(f"  WARNING: {len(all_emails) - emails_in_threads} email(s) missing from threads!")


if __name__ == "__main__":
    main()
