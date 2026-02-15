# Foxhound

A local, privacy-first document search and analysis pipeline. Ingest emails, diary entries, meeting notes, and documents into a single searchable corpus — then explore with semantic search and optional AI-powered analysis.

Foxhound tracks down what matters across large, messy document collections without sending anything to the cloud (unless you choose to).

## Use Cases

- **Workplace disputes** — Gather email trails, meeting notes, and diary entries to build a chronological evidence pack for HR or legal review
- **Due diligence** — Search across thousands of documents for red flags, inconsistencies, or undisclosed risks before a deal closes
- **Subject access requests (SARs)** — Quickly locate every document mentioning a specific person across all source types
- **Internal investigations** — Cross-reference communications between multiple parties to establish who knew what and when
- **Contract review** — Semantic search across large document sets to find relevant clauses, commitments, or obligations
- **Research synthesis** — Merge notes, papers, and meeting minutes into a single searchable corpus, then use AI to surface key themes

## What's Included

| Stage | Script | What It Does | Cost |
|---|---|---|---|
| **Stage 0** | `dedup.py` | Strips quoted text from email reply chains, groups into threads | Free |
| **Stage 1** | `ingest.py` | Embeds all sources (email, diary, meetings, docs) into ChromaDB | Free |
| **Stage 2a** | `explore.py` | Stats, sender breakdowns, date ranges, interactive exploration | Free |
| **Stage 2b** | `query.py` | Filter by sender/date/keywords + semantic search + CSV export | Free |
| **Merge** | `merge.py` | Combine + deduplicate results from multiple searches | Free |
| **Stage 3-6** | `analyze.py` | AI-powered triage + deep analysis (multiple model choices) | Paid or Free (local) |
| **Privacy** | `pseudonymise.py` | Replaces real names with aliases before any API call | Free |
| **Stage 7** | `export.py` | Formats analysis for pasting into any LLM | Free |

## Supported Source Types

| Type | Format | Parser |
|---|---|---|
| Email | `.eml` files | `parsers/email_parser.py` |
| Diary / Logs | Markdown (`.md`) | `parsers/diary_parser.py` |
| Meeting Notes | Word (`.docx`) or Markdown | `parsers/meeting_parser.py` |
| Documents | PDF, Word, plain text | `parsers/document_parser.py` |

All source types share a unified schema and are stored in the same ChromaDB collection, so semantic search works across all of them.

---

## Quick Start

### 1. Configure Sources

Copy `config.example.yaml` to `config.yaml` and edit the paths:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` to point at your source directories:

```yaml
ingestion:
  sources:
    - type: diary
      path: "~/path/to/diary"
      entry_separator: "## "
    - type: meeting_note
      path: "~/path/to/meeting-notes"
    - type: document
      path: "~/path/to/documents"
```

Email sources are configured separately under `deduplication.source_paths`.

### 2. Ingest

```bash
# Stage 0: Deduplicate email threads (skip if no emails)
uv run python dedup.py

# Stage 1: Embed all sources into ChromaDB
uv run python ingest.py

# Re-ingest from scratch if you add new sources
uv run python ingest.py --reset
```

### 3. Explore (Free)

```bash
# Full corpus stats — sender breakdown, date ranges, source types
uv run python explore.py

# Stats for a specific year
uv run python explore.py --year 2025

# Stats for a specific sender
uv run python explore.py --sender someone@example.com

# Filter by source type
uv run python explore.py --source-type meeting_note

# Interactive exploration mode
uv run python explore.py -i
```

### 4. Search & Filter (Free)

```bash
# Count documents matching a filter
uv run python query.py --count-only --sender someone@example.com

# Filter by sender + date range
uv run python query.py --sender someone@example.com --date-range 2025-01-01 2025-06-30

# Semantic search — finds conceptually similar content, not just keyword matches
uv run python query.py --semantic "project deadline concerns" --top-k 50

# Combine semantic search with filters
uv run python query.py --semantic "budget discussion" --sender someone@example.com

# Show body previews inline
uv run python query.py --semantic "budget discussion" --show

# Read a specific result in full (by result number)
uv run python query.py --semantic "budget discussion" --read 4

# Open the original source file (macOS)
uv run python query.py --semantic "budget discussion" --open 4

# Filter by multiple senders (comma-separated)
uv run python query.py --sender "alice@example.com,bob@example.com" --year 2025

# Filter by source type
uv run python query.py --semantic "action items" --source-type meeting_note

# Export to CSV for manual review
uv run python query.py --sender someone@example.com --year 2025 --export results.csv

# Export to JSON (required input for the analysis stage)
uv run python query.py --semantic "project concerns" --top-k 100 --export-json results.json
```

> **Note:** `--top-k` defaults to 200 for semantic searches. Adjust higher or lower as needed.

### 5. AI Analysis (Paid or Free Local)

**Before any API call, the pipeline will:**

1. Show you the exact model and its description
2. Calculate estimated input and output tokens
3. Display the itemised cost breakdown
4. Ask you to confirm with `y/n` before proceeding

You will never be charged without explicit confirmation.

```bash
# Dry-run first to see cost estimate
uv run python analyze.py results.json \
  --context "Summarise key themes and concerns" \
  --model deepseek \
  --dry-run

# Run the analysis (will ask y/n before spending)
uv run python analyze.py results.json \
  --context "Summarise key themes and concerns" \
  --model deepseek

# Full pipeline: free local triage → filter high-relevance → paid deep analysis
uv run python analyze.py results.json \
  --full-pipeline \
  --context "Identify key decisions and action items" \
  --model deepseek

# Fully local analysis (free, private — nothing leaves your machine)
uv run python analyze.py results.json \
  --context "Summarise findings" \
  --local
```

### 6. Export for LLM Report Writing

```bash
# Copy analysis to clipboard
uv run python export.py analysis_output.md --clipboard

# Save to a file
uv run python export.py analysis_output.md --output report.md

# Export raw evidence package (from JSON query results)
uv run python export.py results.json --raw --clipboard
```

---

## Two-Stage Scoring

The pipeline uses **two different scoring methods**:

| Stage | Tool | Scoring Method | What It Measures | Cost |
|---|---|---|---|---|
| **Search** | `query.py` | Cosine similarity (0.0-1.0) | How textually similar the content is to your search phrase | Free |
| **Triage** | `analyze.py --full-pipeline` | LLM relevance score (1-10) | How contextually relevant the content actually is | Free (local) or Paid |

**Cosine similarity** (from `query.py`) is a rough initial filter based on text similarity. It's good for casting a wide net but can miss contextually relevant results that use different wording.

**LLM triage** (from `analyze.py --full-pipeline`) uses Mistral 7B (free, local) to actually read each document and score relevance against your specific context. The `--full-pipeline` flag runs triage first, filters to documents scoring 7+, then sends only those to the paid model for deep analysis.

### Recommended Multi-Search Workflow

Run several searches with different phrasings to maximise recall, then merge and deduplicate:

```bash
mkdir -p evidence

uv run python query.py \
  --semantic "project delays timeline concerns" \
  --top-k 200 --export-json evidence/delays.json

uv run python query.py \
  --semantic "budget overrun cost escalation" \
  --top-k 200 --export-json evidence/budget.json

uv run python query.py \
  --semantic "stakeholder feedback complaints" \
  --top-k 200 --export-json evidence/feedback.json

# Merge and deduplicate (same document found by multiple searches is kept once)
uv run python merge.py evidence/*.json --output evidence/merged.json

# Triage + analysis
uv run python analyze.py evidence/merged.json \
  --full-pipeline \
  --context "Key project risks and stakeholder concerns" \
  --model deepseek \
  --dry-run
```

---

## Choosing an Analysis Model

Specify the model with the `--model` flag. Defaults to `deepseek` if not specified.

| Flag | Model | Input Cost | Output Cost | Context Window | Best For |
|---|---|---|---|---|---|
| `--model deepseek` | DeepSeek V3.2 | $0.25/M tokens | $0.38/M tokens | 164K | **Default.** Best value for most use cases |
| `--model gemini-flash` | Gemini 2.0 Flash | $0.10/M tokens | $0.40/M tokens | 1M | Large document sets — 1M context window |
| `--model gemini-free` | Gemini 2.0 Flash Exp | Free | Free | 1M | Testing and experimentation |
| `--model haiku` | Claude 3.5 Haiku | $0.80/M tokens | $4.00/M tokens | 200K | High-quality scoring and classification |
| `--local` | Ollama Mistral 7B | Free | Free | Local | **Maximum privacy** — nothing leaves your machine |

### Typical Costs (DeepSeek V3.2)

| Query Type | Documents | Estimated Cost | Time |
|---|---|---|---|
| Quick targeted (50-100) | 50-100 | $0.01-0.02 | 3-5 min |
| Standard (100-300) | 100-300 | $0.02-0.05 | 5-8 min |
| Exploratory (300-600) | 300-600 | $0.05-0.10 | 8-12 min |
| Comprehensive multi-stage | 500-1500 | $0.10-0.25 | 15-25 min |

---

## Cost Controls

Built-in safety limits in `config.yaml`:

- **`confirm_before_api_call: true`** — Always asks before spending
- **`max_cost_per_query: 1.00`** — Hard block above $1.00
- **`warn_above: 0.10`** — Extra warning above $0.10

---

## Privacy & Pseudonymisation

When using cloud API models, the pipeline automatically:

1. Replaces all email addresses and names with aliases (e.g. `alice@example.com` -> `Person-A`)
2. Sends only the pseudonymised text to the API
3. Restores real names in the returned analysis locally

The alias map is stored locally at `data/alias_map.json` and never sent to any API.

```bash
# View current alias map
uv run python pseudonymise.py --show

# Pre-register a known identity
uv run python pseudonymise.py --add "Alice Smith" "alice@example.com"

# Reset all aliases
uv run python pseudonymise.py --reset
```

For maximum privacy, use `--local` — nothing leaves your machine.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.ai) with `mistral:7b` model (for free local triage/analysis)
- OpenRouter API key (only for paid cloud analysis): `export OPENROUTER_API_KEY=sk-or-...`

All Python dependencies are managed by `uv` and installed automatically.

---

## File Structure

```
rag/
├── config.example.yaml     # Configuration template (copy to config.yaml)
├── dedup.py                # Stage 0: Email deduplication
├── ingest.py               # Stage 1: Embed all sources into ChromaDB
├── explore.py              # Stage 2a: Free stats and exploration
├── query.py                # Stage 2b: Filtering + semantic search
├── merge.py                # Combine + deduplicate results from multiple searches
├── analyze.py              # Stage 3-6: AI triage + deep analysis
├── pseudonymise.py         # Privacy: Name/email redaction
├── export.py               # Stage 7: Format for LLM export
├── parsers/
│   ├── base_parser.py      # Unified schema + base class
│   ├── email_parser.py     # EML parsing and cleaning
│   ├── diary_parser.py     # Markdown diary/log entries
│   ├── meeting_parser.py   # Word/Markdown meeting notes
│   └── document_parser.py  # PDF/Word/text documents
├── evidence/               # Search result files (gitignored)
└── data/
    ├── rag_ready/          # Deduplicated thread files
    ├── chroma_db/          # Vector database
    └── alias_map.json      # Pseudonym mappings (local only)
```
