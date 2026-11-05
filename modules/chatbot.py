"""
chatbot.py  (Task 05)
---------------------
Context-aware follow-up chat about the user's palms.

Grounding: every request contains
    1. a system prompt holding BOTH palms' feature JSON and the generated
       report (the conversation's permanent context);
    2. the recent conversation history (memory for follow-up questions);
    3. per-question retrieval: knowledge-base entries matching the question,
       plus a code-generated "detection facts" note for any line the user
       mentions. If the user asks about a line that was NOT detected, the
       note says so explicitly, so the model cannot drift into describing it.

The extra context is attached only to the current question. History keeps
the user's plain words, so old retrieval results don't pile up in the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from modules.knowledge_base import format_entries, search
from modules.llm_engine import LLMClient, LLMError, detection_table, features_for_prompt, unsupported_claims

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What does my heart line traditionally represent?",
    "What differences are visible between my left and right palms?",
    "What does palmistry say about my career?",
    "Can you explain my life line in more detail?",
    "What are the traditional interpretations of my financial prospects?",
)

# Words users might use for each line -> the line's name.
LINE_ALIASES: dict[str, str] = {
    "heart": "heart", "love line": "heart",
    "head": "head", "wisdom line": "head",
    "life": "life", "vitality": "life",
    "fate": "fate", "destiny": "fate", "career line": "fate", "saturn line": "fate",
}

# Questions asking for predictions get the disclaimer appended by code.
PREDICTION_WORDS = re.compile(
    r"\b(will i|going to|future|predict|when will|rich|marry|marriage|die|death|"
    r"how long|lifespan|health|sick|lucky|luck|wealth|money)\b",
    re.IGNORECASE,
)

CHAT_SYSTEM_PROMPT = """You are the follow-up assistant of an educational ENTERTAINMENT palm-reading app.
Palmistry is cultural tradition, not science.

You can see: (1) feature data for the user's LEFT and RIGHT palms measured by computer vision,
(2) the reading already generated, and (3) knowledge-base (KB) entries attached to each question.

RULES
1. Ground every answer in the feature data, the reading, and the KB entries. Cite KB entries
   you use, e.g. [KB:heart-long]. Do not bring in palmistry meanings from elsewhere.
2. NEVER invent palm features. If a line has "status": "not_detected" for a palm, say it was not
   clearly detected in that photo, then explain only its general traditional meaning. If its
   status is "uncertain", say the detection is uncertain and do not treat it as confirmed.
3. Separate observation from interpretation: say what was [Detected] and what is traditional
   [Interpretation]. Use phrasing like "palmistry traditionally associates...".
4. NEVER predict or comment on health, illness, lifespan, death, pregnancy, dates, amounts of
   money, or specific partners, and give no medical, financial, legal, career or relationship
   advice. If asked, kindly explain palmistry cannot do this and offer the traditional theme instead.
5. Keep answers clear and under about 200 words unless the user asks for more detail.
6. If a question is unrelated to the palms or palmistry, briefly steer back to the reading.
7. When asked about the future or prospects, remind the user it is symbolic entertainment.

=== LEFT PALM FEATURES ===
{left_json}

=== RIGHT PALM FEATURES ===
{right_json}

=== DETECTION STATUS TABLE ===
{table}

=== READING ALREADY GIVEN TO THE USER ===
{report}
"""


@dataclass
class ChatTurn:
    """One answered question, with what was retrieved for it (for transparency)."""

    question: str
    answer: str
    kb_ids: list[str] = field(default_factory=list)
    grounding_notes: list[str] = field(default_factory=list)


@dataclass
class ChatSession:
    """Conversation state for one pair of palms.

    WHY a class: Streamlit reruns the whole script on every interaction, so
    the session object lives in st.session_state and carries the context
    and memory between reruns.
    """

    left: dict
    right: dict
    report_text: str
    history: list[dict[str, str]] = field(default_factory=list)
    turns: list[ChatTurn] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        """The permanent grounding context for every chat request."""
        return CHAT_SYSTEM_PROMPT.format(
            left_json=features_for_prompt(self.left),
            right_json=features_for_prompt(self.right),
            table=detection_table(self.left, self.right),
            report=self.report_text,
        )

    def recent_history(self) -> list[dict[str, str]]:
        """Last N exchanges only.

        WHY trim: every turn is resent to the model. An unbounded history
        would eventually exceed the context window and the free-tier token
        limits; recent turns are what follow-up questions refer to.
        """
        return self.history[-2 * config.CHAT_HISTORY_TURNS:]

    def reset(self) -> None:
        """Forget the conversation but keep the palm context."""
        self.history.clear()
        self.turns.clear()


def mentioned_lines(question: str) -> list[str]:
    """Which major lines the question refers to (by name or common alias)."""
    q = question.lower()
    found = [line for alias, line in LINE_ALIASES.items() if re.search(rf"\b{alias}\b", q)]
    return list(dict.fromkeys(found))


def grounding_notes(question: str, left: dict, right: dict) -> list[str]:
    """Code-generated facts about each line the user mentions.

    WHY in code: the strongest protection against invented features is to
    state, right next to the question, exactly what was and was not seen.
    """
    notes = []
    for line in mentioned_lines(question):
        for side, palm in (("LEFT", left), ("RIGHT", right)):
            data = palm["lines"][line]
            if data.get("detected") and data.get("status") == "uncertain":
                notes.append(f"{side} palm {line} line: UNCERTAIN (weak image evidence, confidence "
                             f"{data.get('confidence')}) - say it is uncertain, do not treat it as confirmed.")
            elif data.get("detected"):
                desc = ", ".join(str(data[k]) for k in ("length", "curvature", "depth") if data.get(k))
                notes.append(f"{side} palm {line} line: DETECTED ({desc}; confidence {data.get('confidence')}).")
            else:
                notes.append(f"{side} palm {line} line: NOT clearly detected - do not describe its "
                             "appearance; give only the general traditional meaning.")
    return notes


def build_user_message(question: str, notes: list[str], entries: list[dict]) -> str:
    """The question plus this turn's retrieved context."""
    parts = [question.strip(), "", "--- Context attached by the app (not written by the user) ---"]
    if notes:
        parts.append("Detection facts for lines mentioned:")
        parts.extend(f"- {n}" for n in notes)
    parts.append("Relevant knowledge-base entries:")
    parts.append(format_entries(entries))
    return "\n".join(parts)


def ask(session: ChatSession, question: str, client: LLMClient) -> ChatTurn:
    """Answer one follow-up question and store it in the session's memory.

    Raises LLMError with a user-friendly message on failure; the history is
    only updated on success, so a failed request can simply be retried.
    """
    question = question.strip()
    if not question:
        raise LLMError("Please type a question.")
    if (problem := config.check_llm_config(client.settings)):
        raise LLMError(problem)

    entries = search(question)
    notes = grounding_notes(question, session.left, session.right)
    message = build_user_message(question, notes, entries)

    answer = client.generate(session.system_prompt, [*session.recent_history(),
                                                     {"role": "user", "content": message}])

    if unsupported_claims(answer):
        answer += ("\n\n*Note: part of this answer sounds like a factual prediction. Palmistry "
                   "cannot predict health, lifespan, money or relationships - read it as symbolic only.*")
    if PREDICTION_WORDS.search(question) and "entertainment" not in answer.lower():
        answer += ("\n\n*Reminder: palmistry is a cultural and entertainment practice. "
                   "It cannot predict the future.*")

    session.history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])
    turn = ChatTurn(question, answer, [e["id"] for e in entries], notes)
    session.turns.append(turn)
    return turn
