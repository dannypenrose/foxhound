"""Email parser for .eml files — extracts text and metadata into unified schema."""

import email
import re
from email import policy
from pathlib import Path

from bs4 import BeautifulSoup

# Quoted text patterns (Outlook, Gmail, Apple Mail, Thunderbird)
QUOTE_PATTERNS = [
    r"^-{2,}\s*Original Message\s*-{2,}",
    r"^On .+ wrote:$",
    r"^From:\s+.+\nSent:\s+.+\nTo:\s+.+",
    r"^>{1,}\s",
    r"^_{2,}",
    r"^\*From:\*\s",
]
QUOTE_RE = re.compile("|".join(QUOTE_PATTERNS), re.MULTILINE | re.IGNORECASE)

# Email signature patterns
SIGNATURE_PATTERNS = [
    r"^--\s*$",  # Standard sig delimiter
    r"^Sent from my (iPhone|iPad|Android|Samsung)",
    r"^Get Outlook for",
]
SIGNATURE_RE = re.compile("|".join(SIGNATURE_PATTERNS), re.MULTILINE | re.IGNORECASE)

# Confidentiality disclaimer patterns
DISCLAIMER_PATTERNS = [
    r"This email is CONFIDENTIAL",
    r"This email and any attachments are confidential",
    r"DISCLAIMER:",
    r"LEGAL NOTICE:",
    r"If you are not the intended recipient",
]
DISCLAIMER_RE = re.compile("|".join(DISCLAIMER_PATTERNS), re.IGNORECASE)


def extract_body(msg: email.message.EmailMessage) -> str:
    """Extract plain text body from email, converting HTML if needed."""
    body = msg.get_body(preferencelist=("plain",))
    if body:
        try:
            return body.get_content()
        except Exception:
            pass

    html_body = msg.get_body(preferencelist=("html",))
    if html_body:
        try:
            soup = BeautifulSoup(html_body.get_content(), "html.parser")
            for tag in soup(["style", "script"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            pass

    # Fallback: walk all parts
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            try:
                return part.get_content()
            except Exception:
                continue
        elif content_type == "text/html":
            try:
                soup = BeautifulSoup(part.get_content(), "html.parser")
                for tag in soup(["style", "script"]):
                    tag.decompose()
                return soup.get_text(separator="\n", strip=True)
            except Exception:
                continue

    return ""


def strip_quoted_text(body: str) -> str:
    """Remove quoted/forwarded content, keeping only new text."""
    lines = body.split("\n")
    clean_lines = []

    for line in lines:
        if QUOTE_RE.search(line):
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def strip_signature(body: str) -> str:
    """Remove email signatures."""
    lines = body.split("\n")
    clean_lines = []

    for line in lines:
        if SIGNATURE_RE.search(line):
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def strip_disclaimer(body: str) -> str:
    """Remove confidentiality disclaimers."""
    lines = body.split("\n")
    clean_lines = []

    for i, line in enumerate(lines):
        if DISCLAIMER_RE.search(line):
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def clean_body(body: str, strip_quotes: bool = True) -> str:
    """Full cleaning pipeline for email body text."""
    text = body
    if strip_quotes:
        text = strip_quoted_text(text)
    text = strip_signature(text)
    text = strip_disclaimer(text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_address(addr: str | None) -> str:
    """Extract email address from 'Name <email>' format."""
    if not addr:
        return ""
    match = re.search(r"<([^>]+)>", addr)
    if match:
        return match.group(1).lower()
    return addr.strip().lower()


def parse_address_list(addr: str | None) -> list[str]:
    """Parse comma-separated address list."""
    if not addr:
        return []
    parts = addr.split(",")
    return [parse_address(p) for p in parts if parse_address(p)]


def parse_display_name(addr: str | None) -> str:
    """Extract display name from 'Name <email>' format."""
    if not addr:
        return ""
    match = re.search(r"^(.+?)\s*<", addr)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        return name
    return addr.strip()


def parse_eml(eml_path: Path, folder: str = "inbox") -> dict:
    """Parse a single .eml file into structured metadata + clean body."""
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    message_id = msg.get("Message-ID", "")
    in_reply_to = msg.get("In-Reply-To", "")
    references_raw = msg.get("References", "")
    references = references_raw.split() if references_raw else []

    raw_body = extract_body(msg)
    new_content = clean_body(raw_body, strip_quotes=True)

    from_addr = parse_address(msg.get("From", ""))
    from_name = parse_display_name(msg.get("From", ""))
    to_addrs = parse_address_list(msg.get("To", ""))
    cc_addrs = parse_address_list(msg.get("Cc", ""))
    bcc_addrs = parse_address_list(msg.get("Bcc", ""))

    subject = msg.get("Subject", "") or ""
    date_str = msg.get("Date", "") or ""

    # Parse date to ISO format
    date_iso = ""
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            date_iso = dt.strftime("%Y-%m-%d")
        except Exception:
            date_iso = date_str

    # Check for attachments
    has_attachments = False
    attachment_names = []
    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            has_attachments = True
            fname = part.get_filename()
            if fname:
                attachment_names.append(fname)

    # All participants
    participants = list(set([from_addr] + to_addrs + cc_addrs))
    participants = [p for p in participants if p]

    return {
        "source_type": "email",
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "references": references,
        "date": date_iso,
        "date_raw": date_str,
        "sender": from_addr,
        "sender_name": from_name,
        "recipient": to_addrs,
        "cc": cc_addrs,
        "bcc": bcc_addrs,
        "subject": subject,
        "body_full": raw_body,
        "body_new": new_content,
        "participants": participants,
        "has_attachments": has_attachments,
        "attachment_names": attachment_names,
        "folder": folder,
        "source_file": str(eml_path),
        "thread_topic": msg.get("Thread-Topic", ""),
        "thread_index": msg.get("Thread-Index", ""),
    }
