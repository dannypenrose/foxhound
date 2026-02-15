#!/usr/bin/env python3
"""Stage 2b: Query and filter emails — automated filtering with CSV export.

All operations are local ChromaDB queries. No AI, no API calls, no cost.
Supports both metadata filtering and semantic similarity search.

Usage:
  python query.py --sender scott@company.com --year 2025
  python query.py --sender scott@company.com --date-range 2025-01-01 2025-03-31
  python query.py --keywords performance review --export results.csv
  python query.py --count-only --sender scott@company.com
  python query.py --semantic "hostile work environment" --top-k 200
  python query.py --cc hr@company.com --date-range 2025-03-01 2025-05-31

  # Show body previews inline
  python query.py --semantic "reasonable adjustment" --show

  # Read specific result in full (by result number from listing)
  python query.py --semantic "reasonable adjustment" --read 4

  # Open original .eml file for a result
  python query.py --semantic "reasonable adjustment" --open 4
"""

import csv
import json
import sys
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


def dedup_chunks(metadatas: list, documents: list, scores: list) -> tuple[list, list, list]:
    """Collapse multiple chunks from the same email into one result.

    Groups by message_id (or source_file fallback), keeps the highest-scoring
    chunk's metadata and score, and concatenates all chunk texts for the full body.
    This ensures no emails are lost while removing chunk-level duplicates.
    """
    seen: dict[str, dict] = {}  # key → {meta, chunks: [(score, text)]}

    for m, d, s in zip(metadatas, documents, scores):
        # Use message_id as primary key, fall back to source_file
        key = m.get("message_id", "") or m.get("source_file", "") or id(m)
        key = str(key)

        if key not in seen:
            seen[key] = {"meta": m, "chunks": [], "best_score": s}

        seen[key]["chunks"].append((s, d))

        # Track best score for ranking
        if s is not None and (seen[key]["best_score"] is None or s > seen[key]["best_score"]):
            seen[key]["best_score"] = s

    # Rebuild deduplicated lists
    deduped_metas = []
    deduped_docs = []
    deduped_scores = []

    for entry in seen.values():
        deduped_metas.append(entry["meta"])
        deduped_scores.append(entry["best_score"])
        # Sort chunks by score (best first) and join for full text
        sorted_chunks = sorted(entry["chunks"], key=lambda x: x[0] if x[0] is not None else 0, reverse=True)
        deduped_docs.append("\n".join(c[1] for c in sorted_chunks))

    return deduped_metas, deduped_docs, deduped_scores


def build_where_filter(args: dict) -> dict | None:
    """Build ChromaDB where filter from CLI args."""
    conditions = []

    if args.get("sender"):
        senders = [s.strip() for s in args["sender"].split(",")]
        if len(senders) == 1:
            conditions.append({"sender": {"$eq": senders[0]}})
        else:
            conditions.append({"sender": {"$in": senders}})

    if args.get("source_type"):
        conditions.append({"source_type": {"$eq": args["source_type"]}})

    if args.get("folder"):
        conditions.append({"folder": {"$eq": args["folder"]}})

    if args.get("date_from"):
        conditions.append({"date": {"$gte": args["date_from"]}})

    if args.get("date_to"):
        conditions.append({"date": {"$lte": args["date_to"]}})

    if args.get("year"):
        year = str(args["year"])
        conditions.append({"date": {"$gte": f"{year}-01-01"}})
        conditions.append({"date": {"$lte": f"{year}-12-31"}})

    if args.get("has_attachments"):
        conditions.append({"has_attachments": {"$eq": True}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_where_document_filter(args: dict) -> dict | None:
    """Build ChromaDB where_document filter for text content search."""
    conditions = []

    keywords = args.get("keywords", [])
    for kw in keywords:
        conditions.append({"$contains": kw})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def format_result(meta: dict, doc: str, score: float | None = None) -> dict:
    """Format a single result for display/export."""
    return {
        "date": meta.get("date", ""),
        "sender": meta.get("sender", ""),
        "sender_name": meta.get("sender_name", ""),
        "recipient": meta.get("recipient", "[]"),
        "cc": meta.get("cc", "[]"),
        "subject": meta.get("subject", ""),
        "folder": meta.get("folder", ""),
        "thread_id": meta.get("thread_id", ""),
        "has_attachments": meta.get("has_attachments", False),
        "message_id": meta.get("message_id", ""),
        "source_file": meta.get("source_file", ""),
        "relevance_score": f"{score:.3f}" if score is not None else "",
        "text_preview": doc[:300].replace("\n", " ") if doc else "",
        "text_full": doc or "",
    }


def export_csv(results: list[dict], output_path: str):
    """Export results to CSV."""
    if not results:
        print("  No results to export.")
        return

    fieldnames = [k for k in results[0].keys() if k != "text_full"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k != "text_full"}
            writer.writerow(row)

    print(f"  Exported {len(results)} results to {output_path}")


def export_full(results: list[dict], output_path: str):
    """Export results with full text to JSON (for analysis stage)."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Exported {len(results)} results (with full text) to {output_path}")


def parse_args() -> dict:
    """Parse CLI arguments."""
    args = {
        "sender": None,
        "recipient": None,
        "cc": None,
        "folder": None,
        "source_type": None,
        "year": None,
        "date_from": None,
        "date_to": None,
        "keywords": [],
        "has_attachments": False,
        "semantic": None,
        "top_k": 200,
        "count_only": False,
        "show": False,
        "read": None,
        "open": None,
        "export": None,
        "export_json": None,
        "config": "config.yaml",
    }

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--sender" and i + 1 < len(argv):
            args["sender"] = argv[i + 1].lower()
            i += 2
        elif arg == "--recipient" and i + 1 < len(argv):
            args["recipient"] = argv[i + 1].lower()
            i += 2
        elif arg == "--cc" and i + 1 < len(argv):
            args["cc"] = argv[i + 1].lower()
            i += 2
        elif arg == "--folder" and i + 1 < len(argv):
            args["folder"] = argv[i + 1]
            i += 2
        elif arg == "--source-type" and i + 1 < len(argv):
            args["source_type"] = argv[i + 1]
            i += 2
        elif arg == "--year" and i + 1 < len(argv):
            args["year"] = argv[i + 1]
            i += 2
        elif arg == "--date-range" and i + 2 < len(argv):
            args["date_from"] = argv[i + 1]
            args["date_to"] = argv[i + 2]
            i += 3
        elif arg == "--from" and i + 1 < len(argv):
            args["date_from"] = argv[i + 1]
            i += 2
        elif arg == "--to" and i + 1 < len(argv):
            args["date_to"] = argv[i + 1]
            i += 2
        elif arg == "--keywords":
            # Collect all following non-flag args as keywords
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                args["keywords"].append(argv[i])
                i += 1
        elif arg == "--has-attachments":
            args["has_attachments"] = True
            i += 1
        elif arg == "--semantic" and i + 1 < len(argv):
            args["semantic"] = argv[i + 1]
            i += 2
        elif arg == "--top-k" and i + 1 < len(argv):
            args["top_k"] = int(argv[i + 1])
            i += 2
        elif arg == "--count-only":
            args["count_only"] = True
            i += 1
        elif arg == "--show":
            args["show"] = True
            i += 1
        elif arg == "--read" and i + 1 < len(argv):
            args["read"] = int(argv[i + 1])
            i += 2
        elif arg == "--open" and i + 1 < len(argv):
            args["open"] = int(argv[i + 1])
            i += 2
        elif arg == "--export" and i + 1 < len(argv):
            args["export"] = argv[i + 1]
            i += 2
        elif arg == "--export-json" and i + 1 < len(argv):
            args["export_json"] = argv[i + 1]
            i += 2
        elif arg == "--config" and i + 1 < len(argv):
            args["config"] = argv[i + 1]
            i += 2
        else:
            i += 1

    return args


def main():
    args = parse_args()

    with open(args["config"]) as f:
        config = yaml.safe_load(f)

    db_path = Path(config["database"]["path"])
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(config["database"]["collection"])

    # Build filters
    where_filter = build_where_filter(args)
    where_doc_filter = build_where_document_filter(args)

    if args["semantic"]:
        # Semantic similarity search
        model_name = config["ingestion"].get("embedding_model", "all-MiniLM-L6-v2")
        print(f"Loading embedding model: {model_name}...")
        model = SentenceTransformer(model_name)

        query_embedding = model.encode([args["semantic"]])[0].tolist()

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": args["top_k"],
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        if where_doc_filter:
            query_kwargs["where_document"] = where_doc_filter

        result = collection.query(**query_kwargs)

        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]

        # Convert cosine distance to similarity score
        scores = [1 - d for d in distances]

    else:
        # Metadata-only filtering
        get_kwargs = {
            "include": ["documents", "metadatas"],
        }
        if where_filter:
            get_kwargs["where"] = where_filter
        if where_doc_filter:
            get_kwargs["where_document"] = where_doc_filter

        result = collection.get(**get_kwargs)
        metadatas = result["metadatas"]
        documents = result["documents"]
        scores = [None] * len(metadatas)

    # Post-filter for fields ChromaDB can't handle natively (recipient, cc contain)
    if args.get("recipient"):
        filtered = []
        for m, d, s in zip(metadatas, documents, scores):
            recipients = json.loads(m.get("recipient", "[]"))
            if args["recipient"] in [r.lower() for r in recipients]:
                filtered.append((m, d, s))
        metadatas = [f[0] for f in filtered]
        documents = [f[1] for f in filtered]
        scores = [f[2] for f in filtered]

    if args.get("cc"):
        filtered = []
        for m, d, s in zip(metadatas, documents, scores):
            cc_list = json.loads(m.get("cc", "[]"))
            if args["cc"] in [c.lower() for c in cc_list]:
                filtered.append((m, d, s))
        metadatas = [f[0] for f in filtered]
        documents = [f[1] for f in filtered]
        scores = [f[2] for f in filtered]

    # Deduplicate chunks — collapse multiple chunks from same email into one result
    chunk_count = len(metadatas)
    metadatas, documents, scores = dedup_chunks(metadatas, documents, scores)
    if chunk_count != len(metadatas):
        print(f"  (Collapsed {chunk_count} chunks → {len(metadatas)} unique emails)")

    # Count-only mode
    if args["count_only"]:
        print(f"\n  Found: {len(metadatas)} documents")
        if metadatas:
            from collections import Counter
            senders = Counter(m.get("sender", "") for m in metadatas)
            print(f"\n  Top senders:")
            for s, c in senders.most_common(10):
                print(f"    {s}: {c}")
            dates = sorted(set(m.get("date", "") for m in metadatas if m.get("date")))
            if dates:
                print(f"\n  Date range: {dates[0]} → {dates[-1]}")
        print(f"\n  Cost: $0 (local query)")
        return

    # Format results
    results = []
    for m, d, s in zip(metadatas, documents, scores):
        results.append(format_result(m, d, s))

    # Sort: by relevance score for semantic search, by date for metadata queries
    if args["semantic"]:
        results.sort(key=lambda r: float(r.get("relevance_score", 0) or 0), reverse=True)
    else:
        results.sort(key=lambda r: r.get("date", ""), reverse=True)

    # Display summary
    print(f"\n{'='*60}")
    print(f"  Query Results")
    print(f"{'='*60}")
    print(f"  Found: {len(results)} documents")

    if args["semantic"]:
        print(f"  Search: '{args['semantic']}'")
        print(f"  Top-K: {args['top_k']}")

    filter_desc = []
    if args["sender"]:
        filter_desc.append(f"sender={args['sender']}")
    if args["recipient"]:
        filter_desc.append(f"recipient={args['recipient']}")
    if args["cc"]:
        filter_desc.append(f"cc={args['cc']}")
    if args["year"]:
        filter_desc.append(f"year={args['year']}")
    if args["date_from"]:
        filter_desc.append(f"from={args['date_from']}")
    if args["date_to"]:
        filter_desc.append(f"to={args['date_to']}")
    if args["keywords"]:
        filter_desc.append(f"keywords={args['keywords']}")
    if args["folder"]:
        filter_desc.append(f"folder={args['folder']}")
    if args.get("source_type"):
        filter_desc.append(f"source_type={args['source_type']}")

    if filter_desc:
        print(f"  Filters: {', '.join(filter_desc)}")

    # Date range across all results
    all_dates = sorted(set(r.get("date", "") for r in results if r.get("date")))
    if all_dates:
        print(f"  Date range: {all_dates[0]} → {all_dates[-1]}")

    # Year breakdown
    from collections import Counter
    year_counts = Counter(r.get("date", "")[:4] for r in results if r.get("date"))
    if year_counts:
        year_str = ", ".join(f"{y}: {c}" for y, c in sorted(year_counts.items()))
        print(f"  By year: {year_str}")

    # Sender breakdown
    sender_counts = Counter(r.get("sender", "") for r in results)
    if sender_counts:
        print(f"  Top senders: {', '.join(f'{s} ({c})' for s, c in sender_counts.most_common(5))}")

    # --read N: show a single result in full detail
    if args.get("read"):
        idx = args["read"]
        if 1 <= idx <= len(results):
            r = results[idx - 1]
            print(f"\n  {'='*60}")
            print(f"  Result #{idx}")
            print(f"  {'='*60}")
            print(f"  Date:        {r['date']}")
            print(f"  From:        {r['sender']} ({r.get('sender_name', '')})")
            print(f"  To:          {r['recipient']}")
            print(f"  CC:          {r['cc']}")
            print(f"  Subject:     {r['subject']}")
            print(f"  Folder:      {r['folder']}")
            print(f"  Thread:      {r['thread_id']}")
            print(f"  Attachments: {r['has_attachments']}")
            if r.get("relevance_score"):
                print(f"  Score:       {r['relevance_score']}")
            print(f"  Source:      {r['source_file']}")
            print(f"  {'─'*60}")
            print(f"\n{r.get('text_full', r.get('text_preview', '(no body)'))}\n")
            print(f"  {'─'*60}")
        else:
            print(f"\n  Invalid result number {idx}. Range: 1-{len(results)}")
        return

    # --open N: open the original .eml file in default app
    if args.get("open"):
        import subprocess
        idx = args["open"]
        if 1 <= idx <= len(results):
            r = results[idx - 1]
            source = r.get("source_file", "")
            if source and Path(source).exists():
                print(f"\n  Opening: {source}")
                subprocess.run(["open", source])
            else:
                print(f"\n  Source file not found: {source}")
        else:
            print(f"\n  Invalid result number {idx}. Range: 1-{len(results)}")
        return

    # Show all results
    sort_label = "by relevance" if args["semantic"] else "by date (newest first)"
    print(f"\n  All {len(results)} results ({sort_label}):")
    print(f"  {'─'*56}")

    for i, r in enumerate(results, 1):
        score_str = f" [{r['relevance_score']}]" if r["relevance_score"] else ""
        subject = r['subject'][:45] + "..." if len(r.get('subject', '')) > 45 else r.get('subject', '')
        print(f"  {i:3d}. {r['date']} | {r['sender'][:28]:<28} | {subject}{score_str}")

        # --show: include body preview under each result
        if args.get("show"):
            body = r.get("text_full", r.get("text_preview", ""))
            if body:
                # Show first 200 chars of body, indented
                preview = body[:200].replace("\n", " ").strip()
                if len(body) > 200:
                    preview += "..."
                print(f"       {preview}")
                print()

    print(f"\n  Cost: $0 (local query)")

    # Export
    if args["export"]:
        export_csv(results, args["export"])

    if args["export_json"]:
        export_full(results, args["export_json"])

    if not args["export"] and not args["export_json"]:
        print(f"\n  Tip: --read N to read email #N in full | --show for body previews")
        print(f"       --export results.csv | --export-json results.json for analysis")


if __name__ == "__main__":
    main()
