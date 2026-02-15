"""Meeting notes parser — extracts attendees, action items from Word/Markdown files."""

import re
from pathlib import Path

from .base_parser import BaseParser, make_document

# Attendee extraction patterns
ATTENDEE_PATTERNS = [
    re.compile(r"^(?:attendees?|present|participants?):\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:who|people|team):\s*(.+)$", re.IGNORECASE | re.MULTILINE),
]

# Action item patterns
ACTION_PATTERNS = [
    re.compile(r"^[-*]\s*\[[ x]\]\s*(.+)$", re.MULTILINE),  # - [ ] or - [x] task
    re.compile(r"^(?:action|todo|task):\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:action items?|next steps?):\s*$\n((?:[-*]\s+.+\n?)+)", re.IGNORECASE | re.MULTILINE),
]

# Date extraction from content
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_WRITTEN = re.compile(
    r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})",
    re.IGNORECASE,
)

# Agenda patterns
AGENDA_PATTERN = re.compile(
    r"^(?:agenda|topics?):\s*$\n((?:[-*\d.]\s+.+\n?)+)",
    re.IGNORECASE | re.MULTILINE,
)


def extract_date(text: str, filename: str = "") -> str:
    """Extract date from text or filename."""
    # Try filename first
    match = DATE_PATTERN.search(filename)
    if match:
        return match.group(1)

    # Try content
    match = DATE_PATTERN.search(text)
    if match:
        return match.group(1)

    match = DATE_WRITTEN.search(text)
    if match:
        try:
            from dateutil.parser import parse as dateparse
            dt = dateparse(match.group(1), dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return match.group(1)

    return ""


def extract_attendees(text: str) -> list[str]:
    """Extract attendee names from meeting notes."""
    for pattern in ATTENDEE_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            # Split by comma, semicolon, or "and"
            names = re.split(r"[,;]|\band\b", raw)
            return [n.strip() for n in names if n.strip()]
    return []


def extract_action_items(text: str) -> list[str]:
    """Extract action items from meeting notes."""
    items = []
    for pattern in ACTION_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1) if match.lastindex else match.group(0)
            # Handle multi-line action items section
            if "\n" in raw:
                for line in raw.strip().split("\n"):
                    line = re.sub(r"^[-*\d.]\s+", "", line).strip()
                    if line:
                        items.append(line)
            else:
                items.append(raw.strip())
    return items


def extract_agenda(text: str) -> list[str]:
    """Extract agenda items from meeting notes."""
    match = AGENDA_PATTERN.search(text)
    if match:
        raw = match.group(1)
        items = []
        for line in raw.strip().split("\n"):
            line = re.sub(r"^[-*\d.]\s+", "", line).strip()
            if line:
                items.append(line)
        return items
    return []


def read_docx(file_path: Path) -> str:
    """Extract text from a .docx file."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        print(f"  Warning: python-docx not installed, skipping {file_path.name}")
        return ""
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return ""


def read_markdown(file_path: Path) -> str:
    """Read a Markdown file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return ""


class MeetingParser(BaseParser):
    """Parse meeting notes from Word (.docx) and Markdown (.md) files."""

    def parse_directory(self, dir_path: Path, **kwargs) -> list[dict]:
        """Parse all meeting note files in directory."""
        parse_attendees = kwargs.get("parse_attendees", True)
        parse_action_items = kwargs.get("parse_action_items", True)
        documents = []

        patterns = ["**/*.docx", "**/*.md", "**/*.txt"]
        files = set()
        for pattern in patterns:
            files.update(dir_path.glob(pattern))

        for file_path in sorted(files):
            if file_path.name.startswith(".") or file_path.name.startswith("~$"):
                continue
            if file_path.name.lower() in ("readme.md", "template.md"):
                continue
            documents.extend(self.parse_file(
                file_path,
                parse_attendees=parse_attendees,
                parse_action_items=parse_action_items,
            ))

        return documents

    def parse_file(self, file_path: Path, **kwargs) -> list[dict]:
        """Parse a single meeting notes file."""
        parse_attendees_flag = kwargs.get("parse_attendees", True)
        parse_action_items_flag = kwargs.get("parse_action_items", True)

        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            text = read_docx(file_path)
        elif suffix in (".md", ".txt"):
            text = read_markdown(file_path)
        else:
            return []

        if not text or len(text.strip()) < 20:
            return []

        date = extract_date(text, file_path.stem)
        attendees = extract_attendees(text) if parse_attendees_flag else []
        action_items = extract_action_items(text) if parse_action_items_flag else []
        agenda = extract_agenda(text)

        # Derive subject from first heading or filename
        subject = ""
        heading_match = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
        if heading_match:
            subject = heading_match.group(1).strip()
        else:
            subject = file_path.stem.replace("-", " ").replace("_", " ")

        # Extract email mentions as participants
        email_mentions = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        participants = list(set(attendees + email_mentions))

        return [make_document(
            source_type="meeting_note",
            date=date,
            text=text,
            participants=participants,
            tags=[],
            source_file=str(file_path),
            subject=subject,
            attendees=attendees,
            agenda=agenda,
            action_items=action_items,
        )]
