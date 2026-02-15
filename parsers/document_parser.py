"""Document parser — extracts text from PDF, Word, and plain text files."""

import re
from pathlib import Path

from .base_parser import BaseParser, make_document

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def read_pdf(file_path: Path) -> tuple[str, dict]:
    """Extract text and metadata from a PDF file.

    Returns (text, metadata_dict) where metadata may contain title, author, page_count.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(file_path))
        text = "\n\n".join(page.get_text() for page in doc)
        meta = doc.metadata or {}
        page_count = len(doc)
        doc.close()
        return text, {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "page_count": page_count,
        }
    except ImportError:
        print(f"  Warning: pymupdf not installed, skipping {file_path.name}")
        return "", {}
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return "", {}


def read_docx(file_path: Path) -> tuple[str, dict]:
    """Extract text and metadata from a Word document."""
    try:
        from docx import Document
        doc = Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        meta = {}
        if doc.core_properties:
            meta["title"] = doc.core_properties.title or ""
            meta["author"] = doc.core_properties.author or ""
        return text, meta
    except ImportError:
        print(f"  Warning: python-docx not installed, skipping {file_path.name}")
        return "", {}
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return "", {}


def read_text(file_path: Path) -> tuple[str, dict]:
    """Read a plain text file."""
    try:
        text = file_path.read_text(encoding="utf-8")
        return text, {}
    except Exception:
        return "", {}


class DocumentParser(BaseParser):
    """Parse documents (PDF, Word, plain text) into unified schema."""

    def parse_directory(self, dir_path: Path, **kwargs) -> list[dict]:
        """Parse all document files in directory."""
        documents = []
        patterns = ["**/*.pdf", "**/*.docx", "**/*.doc", "**/*.txt"]
        files = set()
        for pattern in patterns:
            files.update(dir_path.glob(pattern))

        for file_path in sorted(files):
            if file_path.name.startswith(".") or file_path.name.startswith("~$"):
                continue
            documents.extend(self.parse_file(file_path))

        return documents

    def parse_file(self, file_path: Path, **kwargs) -> list[dict]:
        """Parse a single document file."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text, meta = read_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            text, meta = read_docx(file_path)
        elif suffix == ".txt":
            text, meta = read_text(file_path)
        else:
            return []

        if not text or len(text.strip()) < 20:
            return []

        title = meta.get("title", "") or file_path.stem.replace("-", " ").replace("_", " ")
        author = meta.get("author", "")
        page_count = meta.get("page_count")

        # Try to extract date from filename or content
        date = ""
        date_match = DATE_PATTERN.search(file_path.stem)
        if date_match:
            date = date_match.group(1)
        else:
            date_match = DATE_PATTERN.search(text[:500])
            if date_match:
                date = date_match.group(1)

        # Extract email mentions as participants
        participants = list(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)))

        return [make_document(
            source_type="document",
            date=date,
            text=text,
            participants=participants,
            tags=[],
            source_file=str(file_path),
            title=title,
            author=author,
            page_count=page_count,
        )]
