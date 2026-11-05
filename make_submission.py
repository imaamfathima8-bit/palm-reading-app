"""
make_submission.py
------------------
Builds a clean ZIP of the project for submission.

Run from the project folder:   python make_submission.py

WHY this script exists:
    Zipping the folder by hand includes the private `.env` file (your real
    API key) and the large `.venv` folder. This script includes only the
    source code and documentation, and refuses to build the ZIP if anything
    that looks like a real API key is found in the included files.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT.parent / "palm_reading_app_submission.zip"

# Folders and files that must never be submitted.
EXCLUDED_DIRS = {".venv", "venv", "env", "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache"}
EXCLUDED_FILES = {".env", "secrets.toml", ".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}

# Patterns of real keys: Google (AIza...), Google OAuth-style tokens (AQ....),
# Groq (gsk_...), OpenAI (sk-...).
KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bAQ\.[0-9A-Za-z_\-]{20,}"),
    re.compile(r"gsk_[0-9A-Za-z]{20,}"),
    re.compile(r"sk-[0-9A-Za-z_\-]{20,}"),
]

# A downloaded ".env.example" sometimes loses its leading dot.
EXAMPLE_ALIASES = {"env.example", "_env.example", ".env.example"}


def is_excluded(path: Path) -> bool:
    """True for secrets, environments, caches, generated outputs and personal images."""
    rel = path.relative_to(PROJECT)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return True
    if rel.parts[0] == "assets" and path.name != ".gitkeep":
        return True   # saved outputs may contain people's palm photos
    name = path.name
    if name in EXAMPLE_ALIASES:
        return False
    if name in EXCLUDED_FILES or name.endswith(".env") or name.startswith(".env") or ".env." in name:
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def find_keys(path: Path) -> list[str]:
    """Return any key-like strings found in a text file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [m.group(0)[:8] + "..." for pattern in KEY_PATTERNS for m in pattern.finditer(text)]


def main() -> int:
    files = sorted(p for p in PROJECT.rglob("*") if p.is_file() and not is_excluded(p))

    problems = {str(p.relative_to(PROJECT)): keys for p in files if (keys := find_keys(p))}
    if problems:
        print("STOPPED: these files contain something that looks like a real API key:")
        for name, keys in problems.items():
            print(f"  {name}: {', '.join(keys)}")
        print("Remove the key from the file (keys belong only in .env) and run this again.")
        return 1

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = Path("palm_reading_app") / path.relative_to(PROJECT)
            if path.name in EXAMPLE_ALIASES:
                arcname = arcname.with_name(".env.example")
            zf.write(path, arcname.as_posix())

    print(f"Created {OUTPUT}")
    print(f"{len(files)} files included. Left out: .env, .venv, caches and saved outputs.")
    for path in files:
        print("  ", path.relative_to(PROJECT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
