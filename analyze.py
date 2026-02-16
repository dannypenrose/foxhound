#!/usr/bin/env python3
"""Stage 3-6: AI-powered analysis with cost controls and model flexibility.

Supports multiple analysis backends:
  - OpenRouter API (DeepSeek V3.2, Gemini Flash, Claude Haiku)
  - Ollama local models (Mistral 7B for triage, Mixtral/Llama for analysis)
  - Dry-run mode for cost estimation without spending

Usage:
  # Analyze from query results (JSON export)
  python analyze.py results.json --context "Performance review manipulation"

  # Analyze with cost estimate only (no API call)
  python analyze.py results.json --dry-run

  # Use local model (free, private)
  python analyze.py results.json --local --context "HR complaint investigation"

  # Use specific model
  python analyze.py results.json --model deepseek --context "Timeline analysis"
  python analyze.py results.json --model gemini --context "Pattern detection"

  # Triage mode (local Mistral 7B, free, fast)
  python analyze.py results.json --triage

  # Full pipeline: triage → verify high-relevance → deep analysis
  python analyze.py results.json --full-pipeline --context "Constructive dismissal evidence"
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

from pseudonymise import Pseudonymiser

# Model pricing (per million tokens, USD) — updated Feb 2026
MODEL_PRICING = {
    "gemini-flash": {
        "id": "google/gemini-2.0-flash-001",
        "input": 0.10,
        "output": 0.40,
        "context_window": 1000000,
        "description": "Gemini 2.0 Flash — cheapest, fast, good for triage",
    },
    "deepseek": {
        "id": "deepseek/deepseek-v3.2",
        "input": 0.25,
        "output": 0.38,
        "context_window": 164000,
        "description": "DeepSeek V3.2 — best value for deep analysis",
    },
    "gemini-2.5": {
        "id": "google/gemini-2.5-flash",
        "input": 0.30,
        "output": 2.50,
        "context_window": 1048576,
        "description": "Gemini 2.5 Flash — newer, thinking model",
    },
    "gemini-3": {
        "id": "google/gemini-3-flash-preview",
        "input": 0.50,
        "output": 3.00,
        "context_window": 1048576,
        "description": "Gemini 3 Flash — latest, agentic workflows",
    },
    "gemini-free": {
        "id": "google/gemini-2.0-flash-exp:free",
        "input": 0.0,
        "output": 0.0,
        "context_window": 1000000,
        "description": "Gemini 2.0 Flash Exp — FREE (rate limited)",
    },
    "haiku": {
        "id": "anthropic/claude-3.5-haiku",
        "input": 0.80,
        "output": 4.00,
        "context_window": 200000,
        "description": "Claude 3.5 Haiku — high quality, most expensive",
    },
}


def get_triage_score(doc: dict) -> int:
    """Get triage relevance score from a document, handling both key formats."""
    triage = doc.get("triage", {})
    # LLM returns "Relevance Score", code uses "relevance_score"
    return triage.get("relevance_score", 0) or triage.get("Relevance Score", 0)


# Analysis prompt templates
TRIAGE_PROMPT = """You are analysing emails for relevance to an investigation.

CONTEXT: {context}

For each email below, extract:
1. **Relevance Score** (1-10): How relevant is this to the investigation context?
2. **Key Claims**: Any factual claims, promises, or assertions made
3. **People Mentioned**: Names and roles
4. **Dates Referenced**: Any dates mentioned in the text
5. **Tone Assessment**: Professional/neutral/hostile/defensive/evasive
6. **Red Flags**: Anything suspicious, contradictory, or policy-violating
7. **Category**: performance_review | hr_complaint | retaliation | exclusion | policy_violation | neutral | other

Respond in JSON format for each email.

EMAILS:
{emails}"""

ANALYSIS_PROMPT = """You are a thorough investigative analyst reviewing email evidence.

INVESTIGATION CONTEXT: {context}

Analyse the following emails and produce a structured report containing:

## 1. Executive Summary
- Brief overview of key findings
- Overall assessment of evidence strength

## 2. Timeline of Events
- Chronological list of significant events with dates
- Note any gaps or suspicious timing

## 3. Key Evidence
For each significant piece of evidence:
- **Date**: When it occurred
- **Source**: Who sent it, who received it
- **Direct Quote**: Exact relevant text from the email
- **Significance**: Why this matters to the investigation
- **Evidence Strength**: Strong / Supporting / Weak
- **Corroboration**: Does other evidence support or contradict this?

## 4. Pattern Analysis
- Communication patterns (frequency changes, tone shifts)
- Behavioral patterns (exclusion, retaliation indicators, policy inconsistencies)
- Relationship dynamics between participants

## 5. Contradictions & Inconsistencies
- Where statements contradict each other
- Where actions contradict stated policies
- Timeline inconsistencies

## 6. Gaps & Missing Information
- What's missing that would strengthen the analysis
- Suggested follow-up queries to fill gaps

## 7. Witness Map
- Who was involved in or aware of key events
- CC/BCC patterns that indicate awareness

IMPORTANT: Base all findings on direct quotes and verifiable facts from the emails.
Mark any inferences or interpretations explicitly as "INVESTIGATIVE NOTE" — these
are leads for human review, not conclusions.

EMAILS ({count} documents):
{emails}"""

SCORING_PROMPT = """Rate the relevance of each email excerpt to this investigation context:

CONTEXT: {context}

For each email, respond with ONLY a JSON object:
{{"email_index": N, "relevance_score": 1-10, "reason": "brief explanation"}}

EMAILS:
{emails}"""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def estimate_cost(input_text: str, model_key: str, estimated_output_tokens: int = 2000) -> dict:
    """Estimate cost for an API call."""
    model = MODEL_PRICING[model_key]
    input_tokens = estimate_tokens(input_text)
    input_cost = (input_tokens / 1_000_000) * model["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * model["output"]
    total = input_cost + output_cost

    return {
        "model": model_key,
        "model_id": model["id"],
        "description": model["description"],
        "input_tokens": input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total,
        "within_context": input_tokens < model["context_window"],
    }


def call_openrouter(prompt: str, model_key: str, config: dict) -> str:
    """Call OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY", config.get("api_keys", {}).get("openrouter", ""))
    if not api_key or api_key.startswith("${"):
        raise ValueError(
            "OPENROUTER_API_KEY not set. Export it:\n"
            "  export OPENROUTER_API_KEY=sk-or-..."
        )

    model = MODEL_PRICING[model_key]

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model["id"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0.1,
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_ollama(prompt: str, model: str = "mistral:7b") -> str:
    """Call local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr}")
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError("Ollama not found. Install from https://ollama.ai")


def format_emails_for_prompt(documents: list[dict], max_chars: int = 500000,
                             truncate_body: int = 0) -> str:
    """Format documents into prompt-ready text with metadata headers.

    Args:
        truncate_body: If >0, truncate each document body to this many chars.
                       Useful for triage where full text isn't needed.
    """
    parts = []
    total_chars = 0

    for i, doc in enumerate(documents):
        header = (
            f"--- Email #{i+1} ---\n"
            f"Date: {doc.get('date', 'unknown')}\n"
            f"From: {doc.get('sender', 'unknown')}\n"
            f"To: {doc.get('recipient', 'unknown')}\n"
            f"CC: {doc.get('cc', '')}\n"
            f"Subject: {doc.get('subject', '')}\n"
            f"---\n"
        )
        body = doc.get("text_full", doc.get("text_preview", ""))
        if truncate_body > 0 and len(body) > truncate_body:
            body = body[:truncate_body] + "\n[... truncated for triage ...]"
        entry = header + body + "\n\n"

        if total_chars + len(entry) > max_chars:
            parts.append(f"\n[... {len(documents) - i} more emails truncated for context limit ...]")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "".join(parts)


def run_triage(documents: list[dict], context: str, config: dict,
               model_key: str = None,
               checkpoint_file: str = "evidence/triage_checkpoint.json",
               truncate_body: int = 0,
               concurrency: int = 1) -> list[dict]:
    """Run triage extraction to score document relevance.

    Uses local Mistral 7B by default (free), or a paid model via --model.
    Saves progress to a checkpoint file every 10 batches so work isn't lost
    if the process crashes. On restart, resumes from the last checkpoint.
    """
    import re

    use_api = model_key is not None and model_key != "local"

    checkpoint_path = Path(checkpoint_file)
    start_index = 0
    all_results = []

    # Resume from checkpoint if available
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            all_results = checkpoint.get("results", [])
            start_index = checkpoint.get("next_index", 0)
            if start_index > 0:
                print(f"\n  Resuming triage from document {start_index}/{len(documents)} "
                      f"({len(all_results)} already triaged)")
        except (json.JSONDecodeError, KeyError):
            print(f"  Warning: Corrupt checkpoint file, starting fresh")
            start_index = 0
            all_results = []

    if start_index >= len(documents):
        print(f"\n  Triage already complete ({len(all_results)} documents)")
        return all_results

    remaining = len(documents) - start_index

    if use_api:
        model_info = MODEL_PRICING.get(model_key, {})
        print(f"\n  Running triage on {remaining} documents ({model_key})...")
        print(f"  Model: {model_info.get('description', model_key)}")
        # Sample-based cost estimate
        batch_size = 10
        est_batches = (remaining + batch_size - 1) // batch_size
        sample_batch = documents[start_index:start_index + batch_size]
        sample_text = format_emails_for_prompt(sample_batch, truncate_body=truncate_body)
        sample_prompt = TRIAGE_PROMPT.format(context=context, emails=sample_text)
        avg_input_tokens = estimate_tokens(sample_prompt)
        est_input_tokens = avg_input_tokens * est_batches
        est_output_tokens = est_batches * 500
        est_cost = (est_input_tokens / 1_000_000 * model_info.get("input", 0) +
                    est_output_tokens / 1_000_000 * model_info.get("output", 0))
        print(f"  Estimated cost: ${est_cost:.4f}")
    else:
        print(f"\n  Running triage on {remaining} documents (local Mistral 7B)...")
        print(f"  Cost: $0 (local model)")
    if truncate_body > 0:
        print(f"  Truncating bodies to {truncate_body} chars (faster, lower cost)")
    print(f"  Checkpoint: {checkpoint_file}")

    # Process in batches of 10 for better quality
    batch_size = 10
    total_batches = (len(documents) + batch_size - 1) // batch_size
    checkpoint_interval = 10  # Save every 10 batches

    def _process_batch(batch_info):
        """Process a single batch — returns (batch_index, batch_docs, response_or_error)."""
        idx, batch_docs = batch_info
        batch_text = format_emails_for_prompt(batch_docs, truncate_body=truncate_body)
        prompt = TRIAGE_PROMPT.format(context=context, emails=batch_text)
        try:
            if use_api:
                response = call_openrouter(prompt, model_key, config)
            else:
                response = call_ollama(prompt)
            return (idx, batch_docs, response, None)
        except Exception as e:
            return (idx, batch_docs, None, e)

    def _parse_response(idx, batch_docs, response):
        """Parse triage response and attach scores to documents."""
        results = []
        try:
            json_blocks = re.findall(r'\{[^{}]+\}', response, re.DOTALL)
            for j, block in enumerate(json_blocks):
                try:
                    parsed = json.loads(block)
                    if j < len(batch_docs):
                        documents[idx + j]["triage"] = parsed
                        documents[idx + j]["triage"]["confidence"] = "triage"
                        results.append(documents[idx + j])
                except json.JSONDecodeError:
                    pass
        except Exception:
            for j, doc in enumerate(batch_docs):
                doc["triage"] = {
                    "raw_response": response[:500],
                    "confidence": "triage",
                    "relevance_score": 5,
                }
                results.append(doc)
        return results

    # Build all batch work items
    batch_items = []
    for i in range(start_index, len(documents), batch_size):
        batch_items.append((i, documents[i:i + batch_size]))

    if use_api and concurrency > 1:
        # Concurrent API calls
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"  Concurrency: {concurrency} parallel requests")

        processed = 0
        for wave_start in range(0, len(batch_items), concurrency):
            wave = batch_items[wave_start:wave_start + concurrency]
            wave_num = wave_start // concurrency + 1
            total_waves = (len(batch_items) + concurrency - 1) // concurrency
            batch_from = wave_start + 1
            batch_to = min(wave_start + concurrency, len(batch_items))
            print(f"  Wave {wave_num}/{total_waves} (batches {batch_from}-{batch_to}/{len(batch_items)})...")

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_process_batch, item): item for item in wave}
                for future in as_completed(futures):
                    idx, batch_docs, response, error = future.result()
                    if error:
                        print(f"    WARNING: Batch at doc {idx} failed: {error}")
                        for doc in batch_docs:
                            doc["triage"] = {"confidence": "failed", "relevance_score": 0}
                            all_results.append(doc)
                    else:
                        results = _parse_response(idx, batch_docs, response)
                        all_results.extend(results)
                    processed += 1

            # Checkpoint after each wave
            if processed % checkpoint_interval == 0 or wave_start + concurrency >= len(batch_items):
                last_idx = wave[-1][0] + batch_size
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as f:
                    json.dump({"next_index": last_idx, "results": all_results}, f)
                print(f"  (Checkpoint saved: {len(all_results)} documents triaged)")
    else:
        # Sequential processing (local model or concurrency=1)
        for batch_idx, (i, batch_docs) in enumerate(batch_items):
            batch_num = batch_idx + 1
            print(f"  Triage batch {batch_num}/{len(batch_items)}...")

            idx, batch_docs, response, error = _process_batch((i, batch_docs))
            if error:
                print(f"  WARNING: Triage batch failed: {error}")
                for doc in batch_docs:
                    doc["triage"] = {"confidence": "failed", "relevance_score": 0}
                    all_results.append(doc)
            else:
                results = _parse_response(idx, batch_docs, response)
                all_results.extend(results)

            # Checkpoint every N batches
            if batch_num % checkpoint_interval == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as f:
                    json.dump({"next_index": i + batch_size, "results": all_results}, f)
                print(f"  (Checkpoint saved: {len(all_results)} documents triaged)")

    # Final save and clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  (Checkpoint removed — triage complete)")

    return all_results


def run_analysis(documents: list[dict], context: str, model_key: str, config: dict,
                 pseudonymise: bool = True, dry_run: bool = False) -> str:
    """Run deep analysis with API or local model."""

    # Pseudonymise if using API
    ps = None
    if pseudonymise and model_key not in ("local",):
        ps = Pseudonymiser()
        documents = ps.redact_documents(documents)
        context = ps.redact_text(context)
        print("  Applied pseudonymisation before API call")

    emails_text = format_emails_for_prompt(documents)
    prompt = ANALYSIS_PROMPT.format(
        context=context,
        count=len(documents),
        emails=emails_text,
    )

    if model_key == "local":
        # Local analysis
        if dry_run:
            tokens = estimate_tokens(prompt)
            print(f"\n  DRY RUN — Local Analysis")
            print(f"  Input tokens: ~{tokens:,}")
            print(f"  Model: Ollama (local)")
            print(f"  Cost: $0")
            print(f"  Note: Slower than API but fully private")
            return ""

        print(f"\n  Running local analysis ({len(documents)} documents)...")
        print(f"  Cost: $0 (local model)")
        print(f"  This may take 10-20 minutes...")
        response = call_ollama(prompt, model="mistral:7b")
    else:
        # API analysis
        cost_info = estimate_cost(prompt, model_key, estimated_output_tokens=4000)

        if not cost_info["within_context"]:
            print(f"\n  WARNING: Input ({cost_info['input_tokens']:,} tokens) exceeds "
                  f"context window ({MODEL_PRICING[model_key]['context_window']:,})")
            print(f"  Consider reducing the document set or using gemini-flash (1M context)")
            return ""

        print(f"\n  {'─'*50}")
        print(f"  COST ESTIMATE")
        print(f"  {'─'*50}")
        print(f"  Model:          {cost_info['description']}")
        print(f"  Input tokens:   ~{cost_info['input_tokens']:,}")
        print(f"  Output tokens:  ~{cost_info['estimated_output_tokens']:,}")
        print(f"  Input cost:     ${cost_info['input_cost']:.4f}")
        print(f"  Output cost:    ${cost_info['output_cost']:.4f}")
        print(f"  TOTAL COST:     ${cost_info['total_cost']:.4f}")
        print(f"  {'─'*50}")

        if dry_run:
            print(f"  DRY RUN — no API call made")
            return ""

        # Check cost controls
        max_cost = config.get("cost_controls", {}).get("max_cost_per_query", 1.0)
        warn_above = config.get("cost_controls", {}).get("warn_above", 0.10)

        if cost_info["total_cost"] > max_cost:
            print(f"  BLOCKED: Estimated cost exceeds max_cost_per_query (${max_cost})")
            return ""

        if cost_info["total_cost"] > warn_above:
            print(f"  WARNING: Cost exceeds warn threshold (${warn_above})")

        # Confirm
        if config.get("cost_controls", {}).get("confirm_before_api_call", True):
            confirm = input(f"  Proceed with API call? (y/n) > ").strip().lower()
            if confirm != "y":
                print("  Cancelled.")
                return ""

        print(f"  Calling {model_key}...")
        response = call_openrouter(prompt, model_key, config)

    # Restore real names if pseudonymised
    if ps:
        response = ps.restore_text(response)
        print("  Restored real identities in output")

    return response


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print("Usage: python analyze.py <results.json> [options]")
        print()
        print("Options:")
        print("  --context 'description'   Investigation context (required for analysis)")
        print("  --dry-run                 Estimate cost without API call")
        print("  --model MODEL             deepseek | gemini-flash | gemini-free | haiku")
        print("  --local                   Use local Ollama model (free, private)")
        print("  --triage                  Triage only (local Mistral 7B by default, or use --model)")
        print("  --full-pipeline           Triage → filter → deep analysis")
        print("  --deep-only               Skip triage, run deep analysis on already-triaged data")
        print("  --retry-failed            Re-triage only failed documents from a previous run")
        print("  --truncate N              Truncate each doc body to N chars for triage (e.g. 500)")
        print("  --concurrency N           Run N triage batches in parallel (default: 1, try 5)")
        print("  --no-pseudonymise         Skip pseudonymisation (not recommended)")
        print("  --min-relevance N         Minimum relevance score for analysis (default: 7)")
        print("  --output FILE             Output file (default: analysis_output.md)")
        print("  --config FILE             Config file (default: config.yaml)")
        print()
        print("Models and costs:")
        for key, info in MODEL_PRICING.items():
            print(f"  {key:<15} ${info['input']:.2f}/${info['output']:.2f} per M tokens — {info['description']}")
        return

    input_file = sys.argv[1]
    config_path = "config.yaml"
    context = ""
    model_key = "deepseek"
    dry_run = False
    local = False
    triage_only = False
    full_pipeline = False
    deep_only = False
    retry_failed = False
    pseudonymise = True
    min_relevance = 7
    truncate_body = 0
    concurrency = 1
    output_file = "analysis_output.md"

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--context" and i + 1 < len(sys.argv):
            context = sys.argv[i + 1]
            i += 2
        elif arg == "--model" and i + 1 < len(sys.argv):
            model_key = sys.argv[i + 1]
            i += 2
        elif arg == "--dry-run":
            dry_run = True
            i += 1
        elif arg == "--local":
            local = True
            i += 1
        elif arg == "--triage":
            triage_only = True
            i += 1
        elif arg == "--full-pipeline":
            full_pipeline = True
            i += 1
        elif arg == "--deep-only":
            deep_only = True
            i += 1
        elif arg == "--retry-failed":
            retry_failed = True
            i += 1
        elif arg == "--truncate" and i + 1 < len(sys.argv):
            truncate_body = int(sys.argv[i + 1])
            i += 2
        elif arg == "--concurrency" and i + 1 < len(sys.argv):
            concurrency = int(sys.argv[i + 1])
            i += 2
        elif arg == "--no-pseudonymise":
            pseudonymise = False
            i += 1
        elif arg == "--min-relevance" and i + 1 < len(sys.argv):
            min_relevance = int(sys.argv[i + 1])
            i += 2
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load documents
    with open(input_file) as f:
        documents = json.load(f)

    print(f"\n  Loaded {len(documents)} documents from {input_file}")

    if not context:
        print("  WARNING: No --context provided. Analysis quality depends on clear context.")
        context = "General email investigation"

    if local:
        model_key = "local"

    # Determine triage model: None = local Mistral, otherwise use --model
    triage_model = None if local or model_key == "local" else model_key

    # Retry failed triage batches
    if retry_failed:
        failed = [d for d in documents if d.get("triage", {}).get("confidence") == "failed"]
        passed = [d for d in documents if d.get("triage", {}).get("confidence") != "failed"]
        if not failed:
            print("  No failed documents to retry.")
            return
        print(f"  Found {len(failed)} failed documents to re-triage")
        retried = run_triage(failed, context, config, model_key=triage_model,
                             checkpoint_file="evidence/retry_checkpoint.json",
                             truncate_body=truncate_body, concurrency=concurrency)
        all_docs = passed + retried
        all_docs.sort(key=lambda d: d.get("date", ""))
        with open(output_file, "w") as f:
            json.dump(all_docs, f, indent=2, default=str)
        still_failed = sum(1 for d in retried if d.get("triage", {}).get("confidence") == "failed")
        print(f"\n  Retry complete. Output: {output_file}")
        print(f"  Re-triaged: {len(retried)}, still failed: {still_failed}")
        return

    if triage_only:
        # Triage only mode
        if dry_run and triage_model:
            # Show cost estimate by sampling real batch sizes
            batch_size = 10
            est_batches = (len(documents) + batch_size - 1) // batch_size
            # Sample first 3 batches to get average prompt size
            sample_tokens = []
            for s in range(min(3, est_batches)):
                batch = documents[s * batch_size:(s + 1) * batch_size]
                batch_text = format_emails_for_prompt(batch, truncate_body=truncate_body)
                prompt = TRIAGE_PROMPT.format(context=context, emails=batch_text)
                sample_tokens.append(estimate_tokens(prompt))
            avg_input_tokens = sum(sample_tokens) // len(sample_tokens)
            est_input_tokens = avg_input_tokens * est_batches
            est_output_tokens = est_batches * 500
            model_info = MODEL_PRICING.get(triage_model, {})
            est_cost = (est_input_tokens / 1_000_000 * model_info.get("input", 0) +
                        est_output_tokens / 1_000_000 * model_info.get("output", 0))
            print(f"\n  DRY RUN — Triage cost estimate:")
            print(f"  Model:          {model_info.get('description', triage_model)}")
            print(f"  Batches:        {est_batches}")
            print(f"  Avg tokens/batch: ~{avg_input_tokens:,}")
            print(f"  Total input:    ~{est_input_tokens:,} tokens")
            print(f"  Total output:   ~{est_output_tokens:,} tokens")
            print(f"  TOTAL COST:     ${est_cost:.4f}")
            if truncate_body > 0:
                print(f"  Truncation:     {truncate_body} chars/doc")
            else:
                print(f"  Tip: Use --truncate 500 to reduce cost and speed up triage")
            return
        results = run_triage(documents, context, config, model_key=triage_model,
                             truncate_body=truncate_body, concurrency=concurrency)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Triage complete. Output: {output_file}")
        if not triage_model:
            print(f"  Cost: $0")
        return

    if deep_only:
        # Skip triage — use existing triage scores from input file
        high_relevance = [
            d for d in documents
            if get_triage_score(d) >= min_relevance
        ]
        print(f"\n  Filtered to {len(high_relevance)} documents with triage score >= {min_relevance}"
              f" (from {len(documents)} total)")

        if not high_relevance:
            print("  No documents meet the threshold. Try lowering --min-relevance")
            documents.sort(key=lambda d: get_triage_score(d), reverse=True)
            high_relevance = documents[:50]
            print(f"  Using top {len(high_relevance)} by score instead")

        print(f"\n  Deep Analysis ({model_key})")
        analysis = run_analysis(high_relevance, context, model_key, config,
                                pseudonymise=pseudonymise, dry_run=dry_run)
    elif full_pipeline:
        # Stage 1: Triage
        if triage_model:
            print(f"\n  STAGE 1: Triage ({triage_model})")
        else:
            print("\n  STAGE 1: Triage (local, free)")
        triaged = run_triage(documents, context, config, model_key=triage_model,
                             truncate_body=truncate_body, concurrency=concurrency)

        # Stage 2: Filter high-relevance
        high_relevance = [
            d for d in triaged
            if get_triage_score(d) >= min_relevance
        ]
        print(f"\n  STAGE 2: Filtered to {len(high_relevance)} high-relevance documents "
              f"(score >= {min_relevance})")

        if not high_relevance:
            print("  No high-relevance documents found. Try lowering --min-relevance")
            # Fall back to top N by score
            triaged.sort(key=lambda d: get_triage_score(d), reverse=True)
            high_relevance = triaged[:50]
            print(f"  Using top {len(high_relevance)} by score instead")

        # Stage 3: Deep analysis
        print(f"\n  STAGE 3: Deep Analysis ({model_key})")
        analysis = run_analysis(high_relevance, context, model_key, config,
                                pseudonymise=pseudonymise, dry_run=dry_run)
    else:
        # Direct analysis
        analysis = run_analysis(documents, context, model_key, config,
                                pseudonymise=pseudonymise, dry_run=dry_run)

    if analysis:
        Path(output_file).write_text(analysis, encoding="utf-8")
        print(f"\n  Analysis complete. Output: {output_file}")


if __name__ == "__main__":
    main()
