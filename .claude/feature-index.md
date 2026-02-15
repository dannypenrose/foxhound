# Feature Index

Quick reference for locating files by feature area.

## Pipeline Scripts

- `dedup.py` - Stage 0: Email deduplication, threading, quoted text removal
- `ingest.py` - Stage 1: ChromaDB embedding with sentence-transformers
- `explore.py` - Stage 2a: Free corpus stats, sender breakdown, interactive mode
- `query.py` - Stage 2b: Metadata filtering, semantic search, CSV/JSON export
- `analyze.py` - Stage 3-6: AI analysis (DeepSeek, Gemini, Haiku, Ollama)
- `pseudonymise.py` - Privacy: Name/email → alias mapping before API calls
- `merge.py` - Combine multiple evidence JSON files, dedup by message_id
- `export.py` - Stage 7: Format output for LLM clipboard paste

## Parsing

- `parsers/__init__.py` - Package init, exports all parsers
- `parsers/base_parser.py` - Base class + unified `make_document()` schema helper
- `parsers/email_parser.py` - EML parsing, body extraction, quoted text stripping, signature removal
- `parsers/diary_parser.py` - Markdown diary entries, date/tag/mood extraction
- `parsers/meeting_parser.py` - Word/Markdown meeting notes, attendees, action items
- `parsers/document_parser.py` - PDF/Word/text documents, metadata extraction

## Configuration

- `config.example.yaml` - Configuration template (copy to `config.yaml`)
- `config.yaml` - Local config (gitignored)
- `pyproject.toml` - Python project and dependency management (uv)

## Data (gitignored)

- `data/rag_ready/` - Deduplicated thread directories (`thread_*/metadata.json` + `*.txt`)
- `data/chroma_db/` - ChromaDB persistent vector database
- `data/alias_map.json` - Pseudonymisation forward/reverse mapping

## Documentation

- `README.md` - Quick start, model pricing, full pipeline workflow
- `CLAUDE.md` - Claude Code project context
- `.claude/feature-index.md` - This file
