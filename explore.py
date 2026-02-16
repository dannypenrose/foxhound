#!/usr/bin/env python3
"""Stage 2a: Explore your email corpus — free stats and breakdowns.

All operations are local ChromaDB queries. No AI, no API calls, no cost.

Usage:
  python explore.py                    # Full stats
  python explore.py --year 2025        # Stats for 2025 only
  python explore.py --sender scott@company.com  # Stats for one sender
  python explore.py --interactive      # Interactive exploration mode
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import yaml


def get_collection(config: dict) -> chromadb.Collection:
    db_path = Path(config["database"]["path"])
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(config["database"]["collection"])


def get_all_metadata(collection: chromadb.Collection) -> list[dict]:
    """Fetch all metadata from collection."""
    result = collection.get(include=["metadatas"])
    return result["metadatas"]


def print_stats(metadatas: list[dict], title: str = "Corpus Stats"):
    """Print comprehensive stats from metadata."""
    sender_counter = Counter()
    recipient_counter = Counter()
    date_counter = Counter()
    month_counter = Counter()
    year_counter = Counter()
    subject_counter = Counter()
    folder_counter = Counter()
    source_type_counter = Counter()
    has_attachments_count = 0

    for m in metadatas:
        sender = m.get("sender", "")
        if sender:
            sender_counter[sender] += 1

        recipients = json.loads(m.get("recipient", "[]"))
        for r in recipients:
            recipient_counter[r] += 1

        date = m.get("date", "")
        if date:
            date_counter[date] += 1
            if len(date) >= 7:
                month_counter[date[:7]] += 1
            if len(date) >= 4:
                year_counter[date[:4]] += 1

        subject = m.get("subject", "")
        if subject:
            # Normalise subject (strip Re:/Fwd: prefixes)
            clean_subj = subject.strip()
            for prefix in ["Re: ", "RE: ", "Fwd: ", "FW: ", "Fw: "]:
                while clean_subj.startswith(prefix):
                    clean_subj = clean_subj[len(prefix):].strip()
            if clean_subj:
                subject_counter[clean_subj] += 1

        folder = m.get("folder", "inbox")
        folder_counter[folder] += 1

        source_type = m.get("source_type", "email")
        source_type_counter[source_type] += 1

        if m.get("has_attachments"):
            has_attachments_count += 1

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\n  Total documents (chunks): {len(metadatas):,}")
    print(f"  With attachments: {has_attachments_count:,}")

    if len(source_type_counter) > 1:
        print(f"\n  By Source Type:")
        for stype, count in source_type_counter.most_common():
            print(f"    {stype}: {count:,}")

    print(f"\n  Folders:")
    for folder, count in folder_counter.most_common():
        print(f"    {folder}: {count:,}")

    print(f"\n  By Year:")
    for year, count in sorted(year_counter.items()):
        print(f"    {year}: {count:,}")

    print(f"\n  By Month (top 12):")
    for month, count in sorted(month_counter.items())[-12:]:
        print(f"    {month}: {count:,}")

    print(f"\n  Top 20 Senders:")
    for sender, count in sender_counter.most_common(20):
        print(f"    {sender}: {count:,}")

    print(f"\n  Top 20 Recipients:")
    for recipient, count in recipient_counter.most_common(20):
        print(f"    {recipient}: {count:,}")

    print(f"\n  Top 20 Subjects (normalised):")
    for subject, count in subject_counter.most_common(20):
        display = subject[:60] + "..." if len(subject) > 60 else subject
        print(f"    [{count:,}] {display}")

    print(f"\n  Unique senders: {len(sender_counter):,}")
    print(f"  Unique recipients: {len(recipient_counter):,}")
    print(f"  Unique dates: {len(date_counter):,}")
    print(f"  Date range: {min(date_counter.keys()) if date_counter else 'N/A'} → {max(date_counter.keys()) if date_counter else 'N/A'}")
    print()

    return {
        "senders": sender_counter,
        "recipients": recipient_counter,
        "dates": date_counter,
        "months": month_counter,
        "years": year_counter,
        "subjects": subject_counter,
        "folders": folder_counter,
    }


def filter_metadata(metadatas: list[dict], **kwargs) -> list[dict]:
    """Filter metadata by criteria."""
    filtered = metadatas

    if "year" in kwargs and kwargs["year"]:
        year_str = str(kwargs["year"])
        filtered = [m for m in filtered if m.get("date", "").startswith(year_str)]

    if "sender" in kwargs and kwargs["sender"]:
        sender = kwargs["sender"].lower()
        filtered = [m for m in filtered if sender in m.get("sender", "").lower()]

    if "recipient" in kwargs and kwargs["recipient"]:
        recipient = kwargs["recipient"].lower()
        filtered = [m for m in filtered
                    if recipient in json.dumps(m.get("recipient", "[]")).lower()]

    if "keyword" in kwargs and kwargs["keyword"]:
        keyword = kwargs["keyword"].lower()
        filtered = [m for m in filtered
                    if keyword in m.get("subject", "").lower()]

    if "folder" in kwargs and kwargs["folder"]:
        folder = kwargs["folder"].lower()
        filtered = [m for m in filtered if m.get("folder", "").lower() == folder]

    if "source_type" in kwargs and kwargs["source_type"]:
        stype = kwargs["source_type"].lower()
        filtered = [m for m in filtered if m.get("source_type", "").lower() == stype]

    if "date_from" in kwargs and kwargs["date_from"]:
        filtered = [m for m in filtered if m.get("date", "") >= kwargs["date_from"]]

    if "date_to" in kwargs and kwargs["date_to"]:
        filtered = [m for m in filtered if m.get("date", "") <= kwargs["date_to"]]

    return filtered


def keyword_search(metadatas: list[dict], documents: list[str], keyword: str) -> list[dict]:
    """Search for keyword in both subject and body text."""
    kw = keyword.lower()
    results = []
    for meta, doc in zip(metadatas, documents):
        if kw in meta.get("subject", "").lower() or kw in doc.lower():
            results.append(meta)
    return results


def interactive_mode(collection: chromadb.Collection):
    """Interactive exploration shell."""
    print("\n  Interactive Explore Mode")
    print("  Type queries like:")
    print("    stats                    — Full corpus stats")
    print("    year 2025                — Stats for 2025")
    print("    sender scott@company.com — Emails from sender")
    print("    keyword performance      — Keyword in subject/body")
    print("    count sender=X date=2025 — Quick count")
    print("    quit                     — Exit")
    print()

    all_data = collection.get(include=["metadatas", "documents"])
    metadatas = all_data["metadatas"]
    documents = all_data["documents"]

    while True:
        try:
            query = input("explore> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        parts = query.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "stats":
            print_stats(metadatas)
        elif cmd == "year":
            filtered = filter_metadata(metadatas, year=arg)
            print_stats(filtered, title=f"Stats for {arg}")
        elif cmd == "sender":
            filtered = filter_metadata(metadatas, sender=arg)
            print_stats(filtered, title=f"Stats for sender: {arg}")
        elif cmd == "recipient":
            filtered = filter_metadata(metadatas, recipient=arg)
            print_stats(filtered, title=f"Stats for recipient: {arg}")
        elif cmd == "keyword":
            results = keyword_search(metadatas, documents, arg)
            print(f"\n  Found {len(results)} documents containing '{arg}'")
            if results:
                print_stats(results, title=f"Keyword: {arg}")
        elif cmd == "folder":
            filtered = filter_metadata(metadatas, folder=arg)
            print_stats(filtered, title=f"Stats for folder: {arg}")
        elif cmd == "count":
            # Parse key=value pairs
            kwargs = {}
            for kv in arg.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    kwargs[k] = v
            filtered = filter_metadata(metadatas, **kwargs)
            print(f"\n  Count: {len(filtered):,} documents")
        else:
            # Try as keyword search
            results = keyword_search(metadatas, documents, query)
            print(f"\n  Found {len(results)} documents matching '{query}'")
            if results:
                # Show quick summary
                senders = Counter(m.get("sender", "") for m in results)
                print(f"  Top senders:")
                for s, c in senders.most_common(5):
                    print(f"    {s}: {c}")


def main():
    config_path = Path("config.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = Path(sys.argv[idx + 1])

    with open(config_path) as f:
        config = yaml.safe_load(f)

    collection = get_collection(config)

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode(collection)
        return

    # Collect filters from CLI args
    kwargs = {}
    for flag, key in [("--year", "year"), ("--sender", "sender"),
                      ("--recipient", "recipient"), ("--folder", "folder"),
                      ("--source-type", "source_type"), ("--keyword", "keyword"),
                      ("--from", "date_from"), ("--to", "date_to")]:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            kwargs[key] = sys.argv[idx + 1]

    metadatas = get_all_metadata(collection)

    if kwargs:
        metadatas = filter_metadata(metadatas, **kwargs)
        title = f"Filtered Stats ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
    else:
        title = "Email Corpus Stats"

    print_stats(metadatas, title=title)


if __name__ == "__main__":
    main()
