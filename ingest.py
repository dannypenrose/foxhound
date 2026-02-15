#!/usr/bin/env python3
"""Stage 1: Ingest all sources into ChromaDB with embeddings.

Supports multiple source types: emails (from rag_ready/), diary entries,
meeting notes, and documents. Each source type has its own parser.

Usage:
  python ingest.py                     # Use config.yaml
  python ingest.py --config config.yaml
  python ingest.py --reset             # Clear DB and re-ingest
"""

import json
import sys
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from parsers.base_parser import make_document
from parsers.diary_parser import DiaryParser
from parsers.document_parser import DocumentParser
from parsers.meeting_parser import MeetingParser


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def load_rag_ready(rag_ready_path: Path) -> list[dict]:
    """Load all deduplicated messages from rag_ready/ directories."""
    documents = []
    thread_dirs = sorted(rag_ready_path.glob("thread_*"))

    for thread_dir in thread_dirs:
        metadata_file = thread_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            thread_meta = json.load(f)

        for msg in thread_meta.get("messages", []):
            txt_file = thread_dir / msg["file"]
            if not txt_file.exists():
                continue

            body = txt_file.read_text(encoding="utf-8").strip()
            if not body:
                continue

            documents.append({
                "text": body,
                "metadata": {
                    "source_type": "email",
                    "thread_id": thread_meta["thread_id"],
                    "subject": msg.get("subject", thread_meta.get("subject", "")),
                    "sender": msg.get("from", ""),
                    "sender_name": msg.get("from_name", ""),
                    "recipient": json.dumps(msg.get("to", [])),
                    "cc": json.dumps(msg.get("cc", [])),
                    "date": msg.get("date", ""),
                    "date_raw": msg.get("date_raw", ""),
                    "message_id": msg.get("message_id", ""),
                    "has_attachments": msg.get("has_attachments", False),
                    "attachment_names": json.dumps(msg.get("attachment_names", [])),
                    "folder": msg.get("folder", "inbox"),
                    "source_file": msg.get("source_file", ""),
                    "participants": json.dumps(
                        list(set(
                            [msg.get("from", "")] +
                            msg.get("to", []) +
                            msg.get("cc", [])
                        ))
                    ),
                },
            })

    return documents


def load_non_email_source(source_config: dict) -> list[dict]:
    """Load documents from a non-email source using the appropriate parser."""
    source_type = source_config["type"]
    source_path = Path(source_config["path"]).expanduser()

    if not source_path.exists():
        print(f"  Warning: Source path does not exist: {source_path}")
        return []

    # Select parser
    if source_type == "diary":
        parser = DiaryParser()
        kwargs = {"entry_separator": source_config.get("entry_separator", "## ")}
    elif source_type == "meeting_note":
        parser = MeetingParser()
        kwargs = {
            "parse_attendees": source_config.get("parse_attendees", True),
            "parse_action_items": source_config.get("parse_action_items", True),
        }
    elif source_type == "document":
        parser = DocumentParser()
        kwargs = {}
    else:
        print(f"  Warning: Unknown source type '{source_type}', skipping")
        return []

    # Parse directory
    if source_path.is_dir():
        raw_docs = parser.parse_directory(source_path, **kwargs)
    else:
        raw_docs = parser.parse_file(source_path, **kwargs)

    # Convert unified schema to ingest format (text + metadata dict)
    documents = []
    for doc in raw_docs:
        text = doc.pop("text", "")
        if not text or len(text.strip()) < 20:
            continue

        # ChromaDB metadata must be str, int, float, or bool — serialise lists
        metadata = {}
        for key, value in doc.items():
            if isinstance(value, list):
                metadata[key] = json.dumps(value)
            elif value is None:
                metadata[key] = ""
            else:
                metadata[key] = value

        documents.append({"text": text, "metadata": metadata})

    return documents


def main():
    config_path = Path("config.yaml")
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        config_path = Path(sys.argv[idx + 1])

    reset = "--reset" in sys.argv

    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_path = Path(config["database"]["path"])
    collection_name = config["database"]["collection"]
    rag_ready_path = Path(config["deduplication"]["rag_ready_path"])
    model_name = config["ingestion"].get("embedding_model", "all-MiniLM-L6-v2")
    chunk_size = config["ingestion"].get("chunk_size", 1000)
    chunk_overlap = config["ingestion"].get("chunk_overlap", 200)

    # Initialize ChromaDB
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    if reset:
        print("Resetting database...")
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection '{collection_name}' already has {existing_count} documents.")
        print("Use --reset to clear and re-ingest.")
        return

    # Load embedding model
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # ── Load all sources ──────────────────────────────────────────────
    all_documents = []
    source_counts = {}

    # 1. Email sources (from rag_ready/)
    print(f"\nLoading emails from {rag_ready_path}...")
    email_docs = load_rag_ready(rag_ready_path)
    all_documents.extend(email_docs)
    source_counts["email"] = len(email_docs)
    print(f"  Loaded {len(email_docs)} email messages")

    # 2. Non-email sources from config
    sources = config["ingestion"].get("sources", [])
    for source_config in sources:
        if source_config["type"] == "email":
            continue  # Emails handled via rag_ready/

        source_type = source_config["type"]
        source_path = source_config.get("path", "")
        print(f"\nLoading {source_type} from {source_path}...")

        docs = load_non_email_source(source_config)
        all_documents.extend(docs)
        source_counts[source_type] = source_counts.get(source_type, 0) + len(docs)
        print(f"  Loaded {len(docs)} {source_type} documents")

    # ── Chunk and embed ───────────────────────────────────────────────
    print(f"\nTotal documents across all sources: {len(all_documents)}")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc_idx, doc in enumerate(all_documents):
        source_type = doc["metadata"].get("source_type", "unknown")
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{source_type}_{doc_idx}_{chunk_idx}"
            all_chunks.append(chunk)
            all_metadatas.append(doc["metadata"])
            all_ids.append(doc_id)

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Batch embed and insert
    batch_size = 256
    total_inserted = 0

    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        batch_texts = all_chunks[i:batch_end]
        batch_metas = all_metadatas[i:batch_end]
        batch_ids = all_ids[i:batch_end]

        print(f"  Embedding batch {i // batch_size + 1} ({i+1}-{batch_end} of {len(all_chunks)})...")
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=embeddings,
        )
        total_inserted += len(batch_texts)

    print(f"\n{'='*50}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*50}")
    for stype, count in sorted(source_counts.items()):
        print(f"  {stype:<15} {count:>6,} documents")
    print(f"  {'─'*30}")
    print(f"  {'total':<15} {len(all_documents):>6,} documents")
    print(f"  Chunks embedded:    {total_inserted:,}")
    print(f"  ChromaDB path:      {db_path}")
    print(f"  Collection:         {collection_name}")
    print(f"  Embedding model:    {model_name}")


if __name__ == "__main__":
    main()
