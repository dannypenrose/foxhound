"""Base parser — defines the unified document schema all parsers must return."""

from abc import ABC, abstractmethod
from pathlib import Path


# Default values for source-type-specific fields
EMPTY_SCHEMA = {
    # Universal fields (all source types)
    "source_type": "",
    "date": "",
    "text": "",
    "participants": [],
    "tags": [],
    "source_file": "",

    # Email-specific
    "sender": "",
    "sender_name": "",
    "recipient": [],
    "cc": [],
    "thread_id": "",
    "subject": "",
    "message_id": "",
    "folder": "",
    "has_attachments": False,
    "attachment_names": [],

    # Diary-specific
    "entry_number": None,
    "mood": "",

    # Meeting-specific
    "attendees": [],
    "agenda": [],
    "action_items": [],

    # Transcript-specific
    "speakers": [],
    "duration_minutes": None,
    "utterance_count": None,

    # Document-specific
    "title": "",
    "author": "",
    "page_count": None,
}


def make_document(**overrides) -> dict:
    """Create a document dict with unified schema, filling defaults for missing fields."""
    doc = dict(EMPTY_SCHEMA)
    doc.update(overrides)
    return doc


class BaseParser(ABC):
    """Base class for all source type parsers."""

    @abstractmethod
    def parse_directory(self, dir_path: Path, **kwargs) -> list[dict]:
        """Parse all files in a directory, returning documents in unified schema.

        Each document should be created via make_document() to ensure
        all fields are present with sensible defaults.
        """

    @abstractmethod
    def parse_file(self, file_path: Path, **kwargs) -> list[dict]:
        """Parse a single file, returning one or more documents in unified schema."""
