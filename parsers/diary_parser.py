"""Diary/log parser — splits Markdown files into dated entries."""

import re
from pathlib import Path

from .base_parser import BaseParser, make_document


# Date patterns to extract from headers
DATE_PATTERNS = [
    # ISO: 2025-03-15
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    # UK: 15/03/2025 or 15-03-2025
    re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})"),
    # Written: March 15, 2025 or 15 March 2025
    re.compile(
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
]

# Tag/mood extraction patterns
TAG_PATTERN = re.compile(r"^tags?:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
MOOD_PATTERN = re.compile(r"^mood:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def extract_date(text: str) -> str:
    """Try to extract a date from text, returning ISO format or empty string."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            # If already ISO format, return as-is
            if re.match(r"\d{4}-\d{2}-\d{2}", raw):
                return raw
            # Try to parse other formats
            try:
                from dateutil.parser import parse as dateparse
                dt = dateparse(raw, dayfirst=True)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return raw
    return ""


def extract_tags(text: str) -> list[str]:
    """Extract tags from text (tags: tag1, tag2 or #tag patterns)."""
    tags = []
    match = TAG_PATTERN.search(text)
    if match:
        raw = match.group(1)
        tags.extend(t.strip().strip("#") for t in raw.split(",") if t.strip())

    # Also find #hashtag patterns
    hashtags = re.findall(r"#(\w+)", text)
    tags.extend(h for h in hashtags if h not in tags)

    return tags


def extract_mood(text: str) -> str:
    """Extract mood from text (mood: frustrated)."""
    match = MOOD_PATTERN.search(text)
    return match.group(1).strip() if match else ""


class DiaryParser(BaseParser):
    """Parse Markdown diary/log files into individual entries."""

    def parse_directory(self, dir_path: Path, **kwargs) -> list[dict]:
        """Parse all Markdown files in directory."""
        separator = kwargs.get("entry_separator", "## ")
        documents = []

        md_files = sorted(dir_path.glob("**/*.md"))
        for md_file in md_files:
            # Skip dotfiles, templates, READMEs
            if md_file.name.startswith(".") or md_file.name.lower() in ("readme.md", "template.md"):
                continue
            documents.extend(self.parse_file(md_file, entry_separator=separator))

        return documents

    def parse_file(self, file_path: Path, **kwargs) -> list[dict]:
        """Parse a single Markdown file, splitting by entry separator."""
        separator = kwargs.get("entry_separator", "## ")
        documents = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        if not content.strip():
            return []

        # Try to extract date from filename first (common pattern: 2025-03-15.md)
        file_date = extract_date(file_path.stem)

        # Split by separator
        if separator in content:
            parts = content.split(separator)
            # First part is preamble (before first heading)
            entries = []
            for i, part in enumerate(parts):
                if i == 0 and not part.strip():
                    continue
                if part.strip():
                    entries.append((separator + part) if i > 0 else part)
        else:
            # Whole file is one entry
            entries = [content]

        for entry_idx, entry_text in enumerate(entries):
            text = entry_text.strip()
            if not text or len(text) < 20:
                continue

            # Extract metadata from entry
            entry_date = extract_date(text) or file_date
            tags = extract_tags(text)
            mood = extract_mood(text)

            # Extract participants mentioned (simple @mention or email pattern)
            participants = re.findall(r"@(\w+)", text)
            email_mentions = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
            participants.extend(email_mentions)

            documents.append(make_document(
                source_type="diary",
                date=entry_date,
                text=text,
                participants=list(set(participants)),
                tags=tags,
                source_file=str(file_path),
                entry_number=entry_idx + 1,
                mood=mood,
            ))

        return documents
