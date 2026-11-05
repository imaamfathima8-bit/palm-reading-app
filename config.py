"""
config.py
---------
Central configuration for the Palm Reading application.

WHY this file exists:
    Every threshold used by the image-processing pipeline lives here, in one
    place, so that (a) each value can be justified in the viva, (b) tuning
    does not require hunting through modules, and (c) secrets are read only
    from the environment and never appear in source code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
KNOWLEDGE_PATH: Path = BASE_DIR / "knowledge" / "palmistry_kb.json"
ASSETS_DIR: Path = BASE_DIR / "assets"

# Load .env from the project folder explicitly, so the app works no matter
# which directory `streamlit run` is launched from.
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false environment variable.

    WHY: .env values are always strings; "false" would otherwise be truthy.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on bad input.

    WHY: a typo in .env should not crash the whole app at import time.
    """
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int environment variable, falling back on bad input."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# Privacy
# ------------------------------------------------------------------
# Off by default: the brief asks us to minimise retention of uploaded images.
SAVE_OUTPUTS: bool = _env_bool("SAVE_OUTPUTS", False)

# ------------------------------------------------------------------
# Task 01 - Upload validation thresholds
# ------------------------------------------------------------------
ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png")
# Must match the formats PIL reports after actually decoding the bytes,
# so a .txt renamed to .jpg is still caught.
ALLOWED_PIL_FORMATS: tuple[str, ...] = ("JPEG", "PNG")
MAX_FILE_SIZE_MB: int = 15

# Shorter side below this is too coarse to resolve palm creases.
MIN_IMAGE_SIDE_PX: int = 400

# Variance of the Laplacian: sharp images have many strong second-order
# edges (high variance); blurred ones do not. 40 is a deliberately lenient
# value measured on the resized working image - phone photos of skin are
# naturally low-texture, and WhatsApp-compressed photos of clearly visible
# palms scored 55-110 in testing, while genuinely blurred photos score < 10.
BLUR_VARIANCE_THRESHOLD: float = 40.0

# Mean grayscale brightness (0-255). Outside this band the lines are
# lost in shadow or washed out by glare.
MIN_MEAN_BRIGHTNESS: float = 60.0
MAX_MEAN_BRIGHTNESS: float = 220.0

# Standard deviation of grayscale intensities = histogram spread.
# A very narrow histogram means a flat, low-contrast image.
MIN_CONTRAST_STD: float = 25.0

# ------------------------------------------------------------------
# Task 02 - Preprocessing parameters
# ------------------------------------------------------------------
# All images are resized so the longer side equals this. A fixed scale makes
# kernel sizes and pixel-length thresholds mean the same thing for every photo.
WORKING_LONG_SIDE_PX: int = 800

GAUSSIAN_KERNEL: tuple[int, int] = (5, 5)   # must be odd
GAUSSIAN_SIGMA: float = 0

# Bilateral filter: smooths flat skin but keeps sharp crease edges because
# it also weights neighbours by intensity similarity.
BILATERAL_DIAMETER: int = 9
BILATERAL_SIGMA_COLOR: float = 75.0
BILATERAL_SIGMA_SPACE: float = 75.0

# CLAHE: clip limit caps noise amplification; 8x8 tiles adapt to the uneven
# lighting typical of hand photos.
CLAHE_CLIP_LIMIT: float = 2.5
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)

CANNY_LOW: int = 30
CANNY_HIGH: int = 90
SOBEL_KERNEL: int = 3

# Adaptive threshold: local mean over a block, minus C. Palm lines are
# darker than their immediate surroundings, so a local threshold works where
# a single global one fails under shadows.
ADAPTIVE_BLOCK_SIZE: int = 21              # must be odd
ADAPTIVE_C: int = 6

MORPH_OPEN_KERNEL: tuple[int, int] = (3, 3)   # removes tiny skin-texture specks
MORPH_CLOSE_KERNEL: tuple[int, int] = (7, 7)  # bridges small gaps along a line

# ------------------------------------------------------------------
# Task 03 - Hand detection & feature extraction
# ------------------------------------------------------------------
MP_MIN_DETECTION_CONFIDENCE: float = 0.5
MP_MAX_NUM_HANDS: int = 1

# Fraction of the image the hand must cover; smaller usually means the hand
# is far away or only partly in frame.
MIN_HAND_AREA_RATIO: float = 0.08

# A line candidate is reported as "detected" only above this score (0-1).
LINE_CONFIDENCE_THRESHOLD: float = 0.35

# Line length as a fraction of palm width -> length label.
LINE_LENGTH_LONG: float = 0.70
LINE_LENGTH_MEDIUM: float = 0.40

# Minimum pixel length for a segment to count as a "prominent crease".
MIN_CREASE_LENGTH_PX: int = 40
MAX_REPORTED_CREASES: int = 8

# BGR colours for annotation overlays (OpenCV uses BGR order).
LINE_COLORS: dict[str, tuple[int, int, int]] = {
    "heart": (60, 60, 230),     # red
    "head": (230, 160, 40),     # blue
    "life": (60, 190, 60),      # green
    "fate": (40, 200, 230),     # yellow
    "crease": (200, 200, 200),  # light grey
    "outline": (255, 0, 255),   # magenta
}

# ------------------------------------------------------------------
# Task 04/05 - LLM settings
# ------------------------------------------------------------------
SUPPORTED_PROVIDERS: tuple[str, ...] = ("gemini", "groq", "openai")

# Recent turns kept in chat memory. Older turns are dropped so the prompt
# stays within the model's context window and free-tier limits.
CHAT_HISTORY_TURNS: int = 10

# Inserted by code, never left to the model, so it cannot be omitted.
DISCLAIMER: str = (
    "⚠️ **Entertainment only.** Palmistry is a cultural and entertainment "
    "practice. It is not scientifically validated and cannot predict "
    "personality, health, wealth, relationships or future events. Do not use "
    "this reading for medical, financial, employment, relationship or other "
    "important decisions."
)


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration for the selected provider.

    WHY a dataclass: llm_engine.py receives one typed object instead of
    reading environment variables itself, which keeps it easy to test.
    """

    provider: str
    api_key: str
    model: str
    base_url: str | None
    temperature: float
    max_tokens: int


def _clean(value: str | None) -> str:
    """Strip spaces, quotes and backticks that often sneak into .env values.

    WHY: a model name typed as `"gemini-3.6-flash"` or with a trailing space
    is rejected by the API with a confusing 'unexpected model name format'.
    """
    return (value or "").strip().strip("\"'`").strip()


def get_llm_settings() -> LLMSettings:
    """Build LLM settings from environment variables.

    WHY: keys are read with os.getenv() at runtime only - nothing secret is
    stored in the code or in version control.
    """
    provider = _clean(os.getenv("MODEL_PROVIDER", "gemini")).lower()

    defaults = {
        "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.6-flash"),
        "groq": ("GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    }
    key_var, model_var, default_model = defaults.get(provider, defaults["gemini"])

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    return LLMSettings(
        provider=provider,
        api_key=_clean(os.getenv(key_var)),
        model=_clean(os.getenv(model_var)) or default_model,
        base_url=base_url if provider == "openai" else None,
        temperature=_env_float("LLM_TEMPERATURE", 0.4),
        max_tokens=_env_int("LLM_MAX_TOKENS", 4096),
    )


def check_llm_config(settings: LLMSettings) -> str | None:
    """Return a user-facing error if the LLM config is unusable, else None.

    WHY: the app can still run the full image pipeline without a key, so we
    surface a clear message in the UI instead of crashing on startup.
    """
    if settings.provider not in SUPPORTED_PROVIDERS:
        return (
            f"MODEL_PROVIDER='{settings.provider}' is not supported. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    placeholder = settings.api_key.startswith("your_") or "PASTE" in settings.api_key
    if not settings.api_key or placeholder:
        if (misnamed := misnamed_env_file()):
            return (
                f"Found a file called '{misnamed}', but the app only reads a file named exactly "
                "'.env'. Rename it to .env (turn on View > Show > File name extensions in File "
                "Explorer to see the full name), then restart the app."
            )
        if not (BASE_DIR / ".env").exists():
            return (
                f"No .env file found in {BASE_DIR.name}. Copy .env.example, rename the copy to "
                f".env, put your {settings.provider} API key in it, then restart the app."
            )
        return (
            f"No API key found for '{settings.provider}'. Open the .env file in "
            "the palm_reading_app folder and paste your key after the '=' sign."
        )
    return None


def misnamed_env_file() -> str | None:
    """Spot common .env naming mistakes (e.g. '.env.txt', 'New Text Document.env').

    WHY: on Windows, Notepad and File Explorer often hide or add extensions,
    so the key file ends up with a name the app never reads.
    """
    if (BASE_DIR / ".env").exists():
        return None
    for path in BASE_DIR.iterdir():
        name = path.name.lower()
        if not path.is_file() or "example" in name:
            continue
        if name in {"env", "env.txt", ".env.txt"} or name.endswith(".env") or name.endswith(".env.txt"):
            return path.name
    return None
