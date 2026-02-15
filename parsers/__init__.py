"""Source parsers — one per content type, all returning unified schema."""

from .base_parser import BaseParser, make_document
from .diary_parser import DiaryParser
from .document_parser import DocumentParser
from .email_parser import parse_eml
from .meeting_parser import MeetingParser

__all__ = [
    "BaseParser",
    "make_document",
    "parse_eml",
    "DiaryParser",
    "DocumentParser",
    "MeetingParser",
]
