#!/usr/bin/env python3
"""Privacy layer: Replace real names/emails with aliases before API calls.

Maps real identities to pseudonyms, applies substitution to documents,
and provides reverse mapping to restore real names after analysis.

Usage:
  # As a library (called by analyze.py)
  from pseudonymise import Pseudonymiser
  ps = Pseudonymiser()
  redacted_docs = ps.redact(documents)
  # ... send to API ...
  restored = ps.restore(api_response)

  # CLI: inspect or manage alias map
  python pseudonymise.py --show          # Show current alias map
  python pseudonymise.py --add "Scott Mackintosh" "s.mackintosh@converta.co.uk"
  python pseudonymise.py --reset         # Clear alias map
"""

import json
import re
import sys
from pathlib import Path

import yaml

# Pseudonym pools
ROLE_PREFIXES = [
    "Manager", "Director", "Colleague", "HR-Rep", "Senior-Leader",
    "Team-Lead", "External", "Admin", "Executive", "Supervisor",
    "Coordinator", "Analyst", "Specialist", "Consultant", "Partner",
]

LETTER_SUFFIXES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class Pseudonymiser:
    """Manages identity pseudonymisation for privacy-safe API calls."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.alias_map_path = Path(config["privacy"]["alias_map_path"])
        self.alias_map: dict[str, str] = {}  # real → alias
        self.reverse_map: dict[str, str] = {}  # alias → real
        self._next_id = 0

        self._load_map()

    def _load_map(self):
        if self.alias_map_path.exists():
            with open(self.alias_map_path) as f:
                data = json.load(f)
            self.alias_map = data.get("forward", {})
            self.reverse_map = data.get("reverse", {})
            self._next_id = data.get("next_id", len(self.alias_map))

    def _save_map(self):
        self.alias_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alias_map_path, "w") as f:
            json.dump({
                "forward": self.alias_map,
                "reverse": self.reverse_map,
                "next_id": self._next_id,
            }, f, indent=2)

    def _generate_alias(self) -> str:
        """Generate next pseudonym like Person-A, Person-B, etc."""
        idx = self._next_id
        self._next_id += 1

        if idx < 26:
            return f"Person-{LETTER_SUFFIXES[idx]}"
        else:
            prefix_idx = idx // 26
            letter_idx = idx % 26
            prefix = ROLE_PREFIXES[prefix_idx % len(ROLE_PREFIXES)]
            return f"{prefix}-{LETTER_SUFFIXES[letter_idx]}"

    def get_or_create_alias(self, real_identity: str) -> str:
        """Get existing alias or create a new one."""
        key = real_identity.lower().strip()
        if key in self.alias_map:
            return self.alias_map[key]

        alias = self._generate_alias()
        self.alias_map[key] = alias
        self.reverse_map[alias] = key
        self._save_map()
        return alias

    def add_known_identity(self, name: str, email: str = "", role_hint: str = ""):
        """Pre-register an identity with optional role hint."""
        alias = self.get_or_create_alias(email or name)
        # Also map the name to the same alias
        if name:
            name_key = name.lower().strip()
            if name_key not in self.alias_map:
                self.alias_map[name_key] = alias
                self.reverse_map[alias] = email or name  # Prefer email as canonical
                self._save_map()
        return alias

    def redact_text(self, text: str) -> str:
        """Replace all known identities in text with their aliases."""
        result = text
        # Sort by length (longest first) to avoid partial replacements
        sorted_identities = sorted(self.alias_map.keys(), key=len, reverse=True)

        for real in sorted_identities:
            alias = self.alias_map[real]
            # Case-insensitive replacement
            pattern = re.compile(re.escape(real), re.IGNORECASE)
            result = pattern.sub(alias, result)

        return result

    def redact_documents(self, documents: list[dict]) -> list[dict]:
        """Redact all documents, replacing identities in text and metadata."""
        redacted = []
        for doc in documents:
            r = dict(doc)
            if "text_full" in r:
                r["text_full"] = self.redact_text(r["text_full"])
            if "text_preview" in r:
                r["text_preview"] = self.redact_text(r["text_preview"])
            if "sender" in r:
                r["sender"] = self.get_or_create_alias(r["sender"]) if r["sender"] else ""
            if "sender_name" in r:
                r["sender_name"] = self.redact_text(r["sender_name"]) if r["sender_name"] else ""
            if "subject" in r:
                r["subject"] = self.redact_text(r["subject"])

            # Handle JSON-encoded lists (email fields + meeting/transcript fields)
            for field in ["recipient", "cc", "attendees", "speakers"]:
                if field in r and r[field]:
                    try:
                        items = json.loads(r[field]) if isinstance(r[field], str) else r[field]
                        r[field] = json.dumps([self.get_or_create_alias(i) for i in items if i])
                    except (json.JSONDecodeError, TypeError):
                        r[field] = self.redact_text(str(r[field]))

            redacted.append(r)

        return redacted

    def restore_text(self, text: str) -> str:
        """Reverse pseudonymisation — restore real identities."""
        result = text
        # Sort by length (longest first)
        sorted_aliases = sorted(self.reverse_map.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            real = self.reverse_map[alias]
            result = result.replace(alias, real)

        return result

    def show_map(self):
        """Display current alias map."""
        if not self.alias_map:
            print("  No aliases registered yet.")
            return

        print(f"\n  Alias Map ({len(self.alias_map)} entries):")
        print(f"  {'Real Identity':<40} → {'Alias':<20}")
        print(f"  {'─'*40}   {'─'*20}")
        for real, alias in sorted(self.alias_map.items()):
            print(f"  {real:<40} → {alias:<20}")

    def reset(self):
        """Clear all aliases."""
        self.alias_map = {}
        self.reverse_map = {}
        self._next_id = 0
        self._save_map()
        print("  Alias map reset.")


def main():
    ps = Pseudonymiser()

    if "--show" in sys.argv:
        ps.show_map()
    elif "--reset" in sys.argv:
        ps.reset()
    elif "--add" in sys.argv:
        idx = sys.argv.index("--add")
        name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        email = sys.argv[idx + 2] if idx + 2 < len(sys.argv) else ""
        alias = ps.add_known_identity(name, email)
        print(f"  Registered: {name} / {email} → {alias}")
    else:
        print("Usage:")
        print("  python pseudonymise.py --show")
        print("  python pseudonymise.py --add 'Name' 'email@example.com'")
        print("  python pseudonymise.py --reset")


if __name__ == "__main__":
    main()
