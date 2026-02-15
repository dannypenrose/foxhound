# Foxhound - CLAUDE.md

> **Feature Index**: [.claude/feature-index.md](.claude/feature-index.md) | **Docs**: [README.md](README.md)

---

## Quick Reference

| Resource | Location |
| -------- | -------- |
| Config | `config.yaml` (gitignored, copy from `config.example.yaml`) |
| Vector DB | `data/chroma_db/` |
| Deduplicated sources | `data/rag_ready/` |
| Alias map | `data/alias_map.json` |
| Run any script | `uv run python <script>.py` |

**No dev servers** — all scripts are CLI tools run on-demand.

---

## Architecture

Local-first multi-source analysis pipeline. Ingests emails (.eml), diary entries (Markdown), meeting notes (Word/MD), and documents (PDF/Word/text). Deduplicates reply chains, embeds into ChromaDB for semantic search, and provides optional paid AI analysis via OpenRouter or free local analysis via Ollama.

**Two-stage scoring**: (1) Cosine similarity from embeddings for initial retrieval, (2) LLM triage (Mistral 7B free local, or paid models) for contextual relevance scoring 1-10.

```text
rag/
├── config.example.yaml     # Configuration template
├── config.yaml             # Local config (gitignored)
├── dedup.py                # Stage 0: Email dedup + threading
├── ingest.py               # Stage 1: Embed into ChromaDB (multi-source)
├── explore.py              # Stage 2a: Free stats + exploration
├── query.py                # Stage 2b: Filter + semantic search
├── merge.py                # Combine multiple evidence JSON files
├── analyze.py              # Stage 3-6: AI analysis (paid/local)
├── pseudonymise.py         # Privacy: Name/email redaction
├── export.py               # Stage 7: Format for LLM export
├── parsers/
│   ├── base_parser.py      # Unified schema + BaseParser ABC
│   ├── email_parser.py     # EML parsing + text cleaning
│   ├── diary_parser.py     # Markdown diary entries
│   ├── meeting_parser.py   # Word/Markdown meeting notes
│   └── document_parser.py  # PDF/Word/text documents
├── evidence/               # Saved search results (gitignored)
└── data/
    ├── rag_ready/          # Deduplicated thread files
    ├── chroma_db/          # ChromaDB vector database
    └── alias_map.json      # Pseudonym mappings (local only)
```

**Stack**: Python 3.12 + ChromaDB + sentence-transformers + OpenRouter API + Ollama + python-docx + pymupdf

**Key Patterns**:

- **Pipeline stages**: Each script is a standalone CLI tool (0 → 1 → 2a/2b → 3-6 → 7)
- **Cost safety**: All paid API calls require explicit `y/n` confirmation with cost estimate shown first
- **Privacy-first**: Pseudonymisation applied before any cloud API call, restored locally after
- **Two-tier storage**: Raw .eml preserved untouched; deduplicated text in `rag_ready/`

---

## Sub-Agent Strategy

Deploy sub-agents when: **>10K tokens** OR **>5 files** OR **complexity >0.7**

| Task Type | Agent | When to Use |
| --------- | ----- | ----------- |
| Python code | `python-expert` | Script modifications, new features |
| Research | `Explore` | File discovery, codebase questions |
| Planning | `Plan` | Architecture, implementation design |
| Quality | `quality-engineer` | Testing, edge cases |

---

## MCP Servers

**Priority**: Context7 (docs) → Sequential (analysis)

**Fallback**: MCP unavailable → WebSearch → Manual implementation

---

## Quality Gates

**Before coding**: Read files first. Plan with TodoWrite. Validate approach.

**During coding**: ONE task in_progress. Mark complete immediately.

**After coding**:

```bash
uv run python dedup.py      # Verify dedup still works
uv run python explore.py    # Verify stats are correct
uv run python query.py --count-only --sender <known-sender>  # Verify queries work
```

---

## Documentation Maintenance (CRITICAL)

**Update docs BEFORE marking any task complete.** This is not optional.

1. **Feature Index** (`.claude/feature-index.md`) — update if files added/moved/renamed
2. **CLAUDE.md** — update if patterns, commands, or structure changed
3. **README.md** — update if features or CLI flags changed

**Workflow**: Complete task → Update docs → Run quality checks → Mark complete

---

## Task Completion Checklist

Before marking ANY task as complete:

- [ ] Code changes tested and working
- [ ] Feature index updated (if files changed)
- [ ] CLAUDE.md updated (if patterns changed)
- [ ] README updated (if CLI flags/features changed)
- [ ] No secrets or sensitive data in commits (check .env, API keys)

---

## Project-Specific Notes

- **Source paths** configured in `config.yaml` (gitignored) — email paths under `deduplication.source_paths`, others under `ingestion.sources`
- **ChromaDB collection** is named `all_sources` — single collection for all source types
- **Embedding model**: `all-MiniLM-L6-v2` — free, local, no API key needed
- **Chunking**: 1000 chars with 200 overlap per chunk
- **Thread detection**: Uses Message-ID → In-Reply-To → References chain, with Thread-Topic fallback for Outlook
- **Multi-source parsers**: All return unified schema via `parsers/base_parser.py` — `make_document()` fills defaults for missing fields
- **Chunk dedup in query.py**: Long texts split into overlapping chunks during ingestion; `dedup_chunks()` collapses them back by `message_id` at query time
- **Multi-sender filter**: `--sender "a@x.com,b@x.com"` uses ChromaDB `$in` operator
- **Evidence merge workflow**: Multiple semantic searches → `merge.py` dedup → `analyze.py --full-pipeline`
- **Cost controls in config.yaml**: `confirm_before_api_call: true`, `max_cost_per_query: 1.00`, `warn_above: 0.10`
- **OpenRouter API key**: Set via `export OPENROUTER_API_KEY=sk-or-...` (only needed for paid analysis)
- **Do NOT search for user file paths** — user will provide paths when ready to add new sources
