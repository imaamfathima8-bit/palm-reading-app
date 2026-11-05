"""
knowledge_base.py  (Task 04 - the "R" in RAG)
---------------------------------------------
Loads the curated palmistry knowledge base and retrieves the entries that
are relevant to a set of detected features or a chat question.

Why RAG instead of letting the LLM use its own knowledge:
    A general LLM knows a lot of loosely-related, sometimes contradictory
    palmistry folklore, and may confidently add claims (for example about
    lifespan or health) that responsible readings must avoid. Retrieval
    restricts the model to a small, reviewed corpus: the prompt contains
    ONLY the entries that match what the computer-vision stage observed,
    each with an ID, so every interpretation can be traced back to a source.

Retrieval strategy:
    Deterministic tag matching. Each feature in the JSON (e.g. a long,
    curved heart line) is converted to tags ("heart:long", "heart:curved"),
    and entries carrying those tags are returned. This is transparent and
    reproducible - the same palm always retrieves the same entries - which
    matters more for an explainable academic project than the fuzzy
    matching that embeddings would give.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import config

# Entry categories in display order, so prompts read from general to specific.
CATEGORY_ORDER = ("general", "comparison", "line", "line-attribute", "shape", "mount", "theme")

# Chat keywords -> knowledge-base tags. WHY: users ask in everyday words
# ("money", "job"); entries are tagged with palmistry terms.
KEYWORD_TAGS: dict[str, tuple[str, ...]] = {
    "heart": ("line:heart",), "love": ("theme:relationships", "line:heart"),
    "relationship": ("theme:relationships", "line:heart"), "relationships": ("theme:relationships", "line:heart"),
    "marriage": ("theme:relationships",), "partner": ("theme:relationships",),
    "emotion": ("theme:relationships", "line:heart"), "emotional": ("theme:relationships", "line:heart"),
    "feelings": ("line:heart",),
    "head": ("line:head",), "mind": ("line:head",), "thinking": ("line:head",), "intelligence": ("line:head",),
    "life": ("line:life",), "vitality": ("line:life",), "energy": ("line:life",),
    "lifespan": ("line:life",), "die": ("line:life",), "death": ("line:life",), "health": ("line:life",),
    "fate": ("line:fate",), "destiny": ("line:fate",),
    "career": ("theme:career", "line:fate", "line:head"), "job": ("theme:career", "line:fate"),
    "work": ("theme:career", "line:fate"), "profession": ("theme:career", "line:fate"),
    "professional": ("theme:career",), "business": ("theme:finance", "mount:mercury"),
    "money": ("theme:finance",), "financial": ("theme:finance",), "finance": ("theme:finance",),
    "wealth": ("theme:finance",), "rich": ("theme:finance",), "prosperity": ("theme:finance",),
    "future": ("theme:future",), "predict": ("theme:future",), "prediction": ("theme:future",),
    "personality": ("theme:personality",), "character": ("theme:personality",),
    "shape": ("shape:square", "shape:rectangular", "shape:oval", "shape:spatulate"),
    "element": ("element:earth", "element:air", "element:fire", "element:water"),
    "left": ("comparison",), "right": ("comparison",), "difference": ("comparison",),
    "differences": ("comparison",), "compare": ("comparison",), "both": ("comparison",),
    "jupiter": ("mount:jupiter",), "saturn": ("mount:saturn",), "apollo": ("mount:apollo",),
    "mercury": ("mount:mercury",), "venus": ("mount:venus",), "luna": ("mount:luna",),
    "mars": ("mount:mars",), "mount": ("mount:jupiter", "mount:venus", "mount:luna"),
    "missing": ("absent",), "absent": ("absent",), "visible": ("absent",),
}

# Position notes from feature_extraction.py -> short attribute tags.
POSITION_TAGS: dict[str, str] = {
    "sloping downward": "sloping", "fairly horizontal": "horizontal",
    "wide arc into the palm": "wide_arc", "close to the thumb": "close_thumb",
    "near the wrist": "wrist", "mid-palm": "mid_palm",
}


class KnowledgeBaseError(Exception):
    """Raised when the knowledge-base file is missing or malformed."""


@lru_cache(maxsize=1)
def load_kb(path: str = str(config.KNOWLEDGE_PATH)) -> tuple[dict, ...]:
    """Load and validate the knowledge base (cached - it never changes at runtime).

    WHY validate: a typo in the JSON should give a clear message in the app,
    not a confusing KeyError halfway through generating a report.
    """
    file = Path(path)
    if not file.exists():
        raise KnowledgeBaseError(f"Knowledge base not found at {file}. Check the 'knowledge' folder.")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge base JSON is invalid (line {exc.lineno}): {exc.msg}") from exc

    entries = data.get("entries", [])
    required = {"id", "category", "title", "tags", "text"}
    for i, entry in enumerate(entries):
        missing = required - entry.keys()
        if missing:
            raise KnowledgeBaseError(f"Knowledge base entry #{i} is missing: {', '.join(sorted(missing))}")
    return tuple(entries)


def _mount_tag(region_text: str) -> str | None:
    """Turn 'thumb base (Mount of Venus)' into 'mount:venus'."""
    match = re.search(r"(?:Mount|Plain) of (\w+)", region_text)
    return f"mount:{match.group(1).lower()}" if match else None


def feature_tags(features: dict) -> set[str]:
    """Convert one palm's feature JSON into retrieval tags.

    WHY only detected attributes produce attribute tags: an undetected line
    retrieves only its general overview and the 'absent' note, so the LLM
    has nothing to build an invented description from.
    """
    tags: set[str] = set()
    for name, line in features.get("lines", {}).items():
        tags.add(f"line:{name}")
        if not line.get("detected"):
            tags.add("absent")
            tags.add(f"{name}:absent")
            continue
        for attr in ("length", "curvature", "depth"):
            value = line.get(attr)
            if value and value != "unknown":
                tags.add(f"{name}:{value}")
        if "ends_under" in line and (mount := _mount_tag(line["ends_under"])):
            tags.add(f"{name}:ends_{mount.split(':')[1]}")
        for key in ("slope", "arc", "starts_from"):
            if key in line and line[key] in POSITION_TAGS:
                tags.add(f"{name}:{POSITION_TAGS[line[key]]}")

    for key, prefix in (("palm_shape", "shape"), ("palm_element", "element")):
        value = features.get(key)
        if value and value != "unknown":
            tags.add(f"{prefix}:{value}")

    for crease in features.get("prominent_creases", []):
        if mount := _mount_tag(crease.get("region", "")):
            tags.add(mount)
    return tags


def _sort_entries(entries: list[dict]) -> list[dict]:
    """Order entries general -> specific so the prompt reads logically."""
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(entries, key=lambda e: (rank.get(e["category"], 99), e["id"]))


def retrieve_for_features(*palms: dict) -> list[dict]:
    """Retrieve every entry relevant to one or more palms' features.

    Entries tagged 'always' (disclaimer, left/right comparison, the theme
    guides) are included for every report, because every report must cover
    those sections.
    """
    wanted: set[str] = {"always"}
    for palm in palms:
        wanted |= feature_tags(palm)
    hits = [e for e in load_kb() if wanted & set(e["tags"])]
    return _sort_entries(hits)


def search(query: str, limit: int = 8) -> list[dict]:
    """Keyword retrieval for chat questions.

    WHY needed on top of the report entries: a follow-up such as "tell me
    about the Mount of Venus" may need an entry the report never used.
    Scoring: +2 for each mapped tag the entry carries, +1 for each query
    word in the entry title.
    """
    words = set(re.findall(r"[a-z]+", query.lower()))
    tags = {t for w in words for t in KEYWORD_TAGS.get(w, ())}
    if not tags and not words:
        return []

    scored: list[tuple[int, dict]] = []
    for entry in load_kb():
        score = 2 * len(tags & set(entry["tags"]))
        score += sum(1 for w in words if len(w) > 3 and w in entry["title"].lower())
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [entry for _, entry in scored[:limit]]


def format_entries(entries: list[dict]) -> str:
    """Render entries for a prompt, each with its ID so claims can be cited."""
    if not entries:
        return "(no knowledge-base entries retrieved)"
    return "\n".join(f"[KB:{e['id']}] {e['title']}: {e['text']}" for e in entries)
