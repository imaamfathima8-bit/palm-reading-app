"""
app.py  (Task 06 - full integration)
------------------------------------
AI-Powered Palm Reading and Intelligent Chatbot Application.

Run with:   streamlit run app.py

Workflow (one app, no separate scripts):
    1. Upload LEFT and RIGHT palm photos        -> modules/validation.py
    2. Image processing (original vs processed) -> modules/preprocessing.py
    3. Hand detection + line annotation         -> modules/hand_detection.py, feature_extraction.py
    4. RAG palm reading                         -> modules/knowledge_base.py, llm_engine.py
    5. Grounded follow-up chat                  -> modules/chatbot.py

This file only orchestrates and displays; all processing lives in modules/.
The visual design (CSS below + .streamlit/config.toml) keeps the interface
quiet so the coloured palm-line overlays are the focus.
"""

from __future__ import annotations

import hashlib
import html
import json

import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Palm Reader", page_icon="✋", layout="wide")

import config  # noqa: E402
from modules.chatbot import EXAMPLE_QUESTIONS, ChatSession, ask  # noqa: E402
from modules.feature_extraction import extract_features, save_outputs  # noqa: E402
from modules.hand_detection import detect_hand, draw_hand_overlay, hand_check  # noqa: E402
from modules.knowledge_base import KnowledgeBaseError  # noqa: E402
from modules.llm_engine import LLMError, generate_report, get_client  # noqa: E402
from modules.preprocessing import run_preprocessing, to_display_rgb  # noqa: E402
from modules.validation import validate_upload  # noqa: E402

SIDES = ("left", "right")
LINE_ORDER = ("heart", "head", "life", "fate")
STAGE_GROUPS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("Resize, grayscale and normalise", "A fixed working size and a single intensity channel.", (0, 1, 2)),
    ("Noise reduction", "Gaussian blur compared with the edge-preserving bilateral filter.", (3, 4)),
    ("Contrast enhancement", "Global histogram equalisation compared with CLAHE.", (5, 6)),
    ("Edge detection", "Sobel gradient magnitude and Canny edges.", (7, 8)),
    ("Thresholding and segmentation", "Skin segmentation, adaptive threshold, black-hat with Otsu.", (9, 10, 11)),
    ("Morphological operations", "Opening removes specks; closing joins broken lines.", (12, 13)),
)


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    """OpenCV colours are BGR; CSS needs RGB hex."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# The same colours used on the annotated photos, so legend and image match.
LINE_HEX = {name: bgr_to_hex(config.LINE_COLORS[name]) for name in LINE_ORDER}

# ------------------------------------------------------------------
# Visual design
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Literata:opsz,wght@7..72,400;7..72,600&display=swap');

:root {
  --paper: #F3F5F7;
  --surface: #FFFFFF;
  --ink: #18202B;
  --muted: #5B6676;
  --rule: #D8DEE6;
  --accent: #2C4C8C;
  --ok: #2E7D5B;
  --warn: #A8741A;
  --bad: #B3403A;
}

html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: 'Public Sans', system-ui, sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
h1, h2, h3 { font-family: 'Public Sans', sans-serif; color: var(--ink); letter-spacing: -0.01em; }
h3 { font-size: 1.15rem; font-weight: 600; }
[data-testid="stHeader"] { background: transparent; }

/* Header */
.pr-header { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.4rem; }
.pr-header svg { flex: none; margin-top: 0.15rem; }
.pr-title { font-size: 1.9rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.015em; color: var(--ink); }
.pr-header p { margin: 0.3rem 0 0; color: var(--muted); max-width: 78ch; font-size: 0.98rem; line-height: 1.5; }
.pr-course { margin-left: auto; text-align: right; color: var(--muted); font-size: 0.82rem; line-height: 1.4; }

/* Stepper: a real sequence, so it is numbered */
.pr-steps { display: flex; gap: 0; margin: 1.3rem 0 1.1rem; border: 1px solid var(--rule);
            border-radius: 10px; background: var(--surface); overflow: hidden; }
.pr-step { flex: 1; display: flex; align-items: center; gap: 0.6rem; padding: 0.7rem 0.9rem;
           color: var(--muted); font-size: 0.9rem; border-right: 1px solid var(--rule); }
.pr-step:last-child { border-right: none; }
.pr-step .n { width: 1.55rem; height: 1.55rem; border-radius: 50%; display: grid; place-items: center;
              font-size: 0.8rem; font-weight: 600; border: 1.5px solid var(--rule); flex: none; }
.pr-step.done { color: var(--ink); }
.pr-step.done .n { background: var(--ok); border-color: var(--ok); color: #fff; }
.pr-step.current { color: var(--ink); font-weight: 600; box-shadow: inset 0 -3px 0 var(--accent); }
.pr-step.current .n { border-color: var(--accent); color: var(--accent); }

/* Disclaimer strip */
.pr-note { border-left: 3px solid var(--warn); background: #FBF6EC; color: #5A4513;
           padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; font-size: 0.88rem; margin-bottom: 1.2rem; }

/* Section headings */
.pr-section { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.2rem 0.75rem; margin: 1.6rem 0 0.6rem;
              padding-bottom: 0.45rem; border-bottom: 1px solid var(--rule); }
.pr-section .pr-h { font-size: 1.3rem; font-weight: 650; color: var(--ink); letter-spacing: -0.01em; }
.pr-section .pr-sub { color: var(--muted); font-size: 0.9rem; }

/* Palm panels */
.st-key-panel_left, .st-key-panel_right, .st-key-reading, .st-key-chatbox {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 1.1rem 1.2rem 0.6rem;
}
.pr-panel-title { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }
.pr-panel-title .pr-ph { font-size: 1.15rem; font-weight: 650; color: var(--ink); }
.st-key-panel_left [data-testid="stImage"] img, .st-key-panel_right [data-testid="stImage"] img {
  border-radius: 8px;
}
.pr-badge { font-size: 0.78rem; padding: 0.15rem 0.55rem; border-radius: 99px; font-weight: 600; }
.pr-badge.ok { background: #E3F1EA; color: var(--ok); }
.pr-badge.bad { background: #F6E4E2; color: var(--bad); }
.pr-badge.wait { background: #ECEFF3; color: var(--muted); }

[data-testid="stFileUploaderDropzone"] { background: var(--paper); border: 1.5px dashed #B9C3D0; border-radius: 10px; }

/* Quality meters */
.pr-meters { display: grid; gap: 0.55rem; margin: 0.4rem 0 0.8rem; }
.pr-meter { display: grid; grid-template-columns: 6.2rem 1fr 3.2rem; align-items: center; gap: 0.7rem; font-size: 0.86rem; }
.pr-meter .label { color: var(--muted); }
.pr-meter .val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.pr-track { position: relative; height: 6px; background: #E6EAF0; border-radius: 3px; }
.pr-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
.pr-fill.ok { background: var(--ok); } .pr-fill.bad { background: var(--bad); }
.pr-mark { position: absolute; top: -3px; width: 2px; height: 12px; background: var(--ink); opacity: 0.55; }

/* Line legend / table */
.pr-lines { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 0.4rem 0 0.6rem; }
.pr-lines th { text-align: left; color: var(--muted); font-weight: 500; padding: 0.35rem 0.5rem;
               border-bottom: 1px solid var(--rule); }
.pr-lines td { padding: 0.45rem 0.5rem; border-bottom: 1px solid #EEF1F5; }
.pr-lines td:nth-child(-n+2) { white-space: nowrap; }
.pr-sw { display: inline-block; width: 1.1rem; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 0.5rem; }
.pr-dim { color: #9AA4B2; }
.pr-conf { font-variant-numeric: tabular-nums; }
.pr-facts { display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; font-size: 0.88rem; color: var(--muted); margin-bottom: 0.4rem; }
.pr-facts b { color: var(--ink); font-weight: 600; }

/* Reading: set as a document */
.st-key-reading { padding: 1.6rem 2rem 1rem; }
.st-key-reading [data-testid="stMarkdownContainer"] p,
.st-key-reading [data-testid="stMarkdownContainer"] li {
  font-family: 'Literata', Georgia, serif; font-size: 1.02rem; line-height: 1.7; max-width: 72ch;
}
.st-key-reading [data-testid="stMarkdownContainer"] h2 {
  font-size: 1.15rem; margin-top: 1.6rem; padding-top: 0.9rem; border-top: 1px solid var(--rule);
}
.st-key-reading table { font-size: 0.88rem; }

[data-testid="stExpander"] [data-testid="stImage"] img { max-height: 560px; width: auto !important;
  max-width: 100%; object-fit: contain; margin: 0 auto; display: block; }
[data-testid="stExpander"] [data-testid="stImage"] { display: flex; flex-direction: column; align-items: center; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] { padding: 0.6rem 0.1rem; font-weight: 500; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
.pr-side h4 { font-size: 0.82rem; color: var(--muted); font-weight: 600; margin: 1.1rem 0 0.35rem; }
.pr-side p { font-size: 0.86rem; margin: 0 0 0.3rem; line-height: 1.45; }
.pr-dot { display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%; margin-right: 0.45rem; }

[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; }
.stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

@media (max-width: 760px) {
  .pr-course { display: none; }
  .pr-step { padding: 0.55rem 0.5rem; font-size: 0.78rem; }
  .pr-step .t { display: none; }
  .pr-section .pr-sub { flex-basis: 100%; }
  .pr-step.current .t { display: inline; }
  .st-key-reading { padding: 1rem; }
}
</style>
"""

LOGO_SVG = (
    '<svg width="46" height="46" viewBox="0 0 46 46" aria-hidden="true">'
    '<rect x="1" y="1" width="44" height="44" rx="11" fill="#18202B"/>'
    f'<path d="M33 15 C 26 12, 18 13, 12 17" stroke="{LINE_HEX["heart"]}" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
    f'<path d="M12 22 C 19 21, 26 23, 33 27" stroke="{LINE_HEX["head"]}" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
    f'<path d="M13 21 C 17 26, 19 32, 19 38" stroke="{LINE_HEX["life"]}" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
    f'<path d="M26 38 C 25 32, 25 26, 25 12" stroke="{LINE_HEX["fate"]}" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    "</svg>"
)


def crop_to_hand(image, detection, pad: float = 0.12):
    """Crop a pipeline image (working-image size) to the hand, with a margin.

    WHY: phone photos are mostly table and background; cropping makes the
    palm and its traced lines large enough to read. Returns the image
    unchanged if no hand outline is available.
    """
    pts = detection.landmarks if detection.landmarks is not None else detection.hand_contour
    if pts is None:
        return image
    pts = pts.reshape(-1, 2)
    h, w = image.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    return image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image


def render_html(markup: str) -> None:
    """Render trusted, app-generated HTML. User text is always escaped first."""
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    """Section heading with an optional one-line explanation."""
    render_html(f'<div class="pr-section"><span class="pr-h">{title}</span><span class="pr-sub">{subtitle}</span></div>')


# ------------------------------------------------------------------
# Cached processing
# WHY cache: Streamlit reruns this whole script on every click. Caching on
# the file bytes means each photo is validated and processed once, and a
# replaced photo is processed fresh automatically.
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def cached_validation(filename: str, data: bytes):
    """Validate one upload, including the hand-presence check."""
    return validate_upload(filename, data, hand_check=hand_check)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_pipeline(filename: str, data: bytes, side: str) -> dict:
    """Preprocess, detect the hand and extract features for one palm."""
    image = cached_validation(filename, data).image_bgr
    prep = run_preprocessing(image)
    detection = detect_hand(prep.resized_bgr, side)
    features = extract_features(prep, detection, side)
    return {
        "prep": prep,
        "detection": detection,
        "features": features,
        "overlay": draw_hand_overlay(prep.resized_bgr, detection),
    }


def init_state() -> None:
    """Create session keys once per browser session."""
    for key, default in (("signature", None), ("report", None), ("chat", None), ("saved_sig", None),
                         ("upload_gen", 0)):
        st.session_state.setdefault(key, default)


def uploader_key(side: str) -> str:
    """Widget key for a palm uploader.

    WHY a generation number: Streamlit does not reliably clear a file
    uploader by deleting its session state. Changing the key creates a brand
    new, empty uploader, which is the dependable way to reset it.
    """
    return f"upload_{side}_{st.session_state.upload_gen}"


def start_over() -> None:
    """Clear both photos, all processing results, the reading and the chat."""
    st.session_state.upload_gen += 1
    cached_validation.clear()
    cached_pipeline.clear()
    reset_results()
    for key in ("signature", "saved_sig", "proc_side"):
        st.session_state.pop(key, None)
    st.session_state.signature = None


def reset_results() -> None:
    """Drop the reading and chat - used when a photo changes or on reset."""
    st.session_state.report = None
    st.session_state.chat = None


# ------------------------------------------------------------------
# Header, stepper, sidebar
# ------------------------------------------------------------------
def render_header() -> None:
    """Brand mark, title and one-sentence description of what the app does."""
    render_html(
        f'<div class="pr-header">{LOGO_SVG}<div><div class="pr-title">Palm Reader</div>'
        "<p>Upload both palms. The app checks each photo, traces your palm lines with "
        "computer vision, then writes a traditional reading based only on what it found.</p></div>"
        '<div class="pr-course">Digital Image Processing<br>NIBM KIC, ICA 01</div></div>'
    )


def render_stepper(steps: dict[str, bool]) -> None:
    """Progress through the five workflow steps; the first unfinished one is current."""
    current_found = False
    items = []
    for i, (label, done) in enumerate(steps.items(), start=1):
        if done:
            state, mark = "done", "✓"
        elif not current_found:
            state, mark, current_found = "current", str(i), True
        else:
            state, mark = "", str(i)
        items.append(f'<div class="pr-step {state}"><span class="n">{mark}</span><span class="t">{label}</span></div>')
    render_html(f'<div class="pr-steps">{"".join(items)}</div>')


def render_sidebar(llm_problem: str | None, model_label: str) -> None:
    """Quiet status panel: model, privacy and reset."""
    with st.sidebar:
        model_state = (f'<span class="pr-dot" style="background:var(--bad)"></span>Not ready'
                       if llm_problem else
                       '<span class="pr-dot" style="background:var(--ok)"></span>Ready')
        saving = ("Annotated outputs are saved to the assets folder (SAVE_OUTPUTS is on)."
                  if config.SAVE_OUTPUTS else "Nothing is saved to disk.")
        render_html(
            '<div class="pr-side">'
            f"<h4>Language model</h4><p>{model_state}</p><p><code>{html.escape(model_label)}</code></p>"
            "<p>Grounded in a curated palmistry knowledge base (retrieval-augmented generation).</p>"
            "<h4>Privacy</h4><p>Photos are processed in memory. Only the measured line features, "
            f"never the images, are sent to the language model. {saving}</p>"
            "<h4>About the reading</h4><p>Palmistry is a cultural tradition, not science. "
            "Readings are for entertainment only.</p></div>"
        )
        if llm_problem:
            st.error(llm_problem)
        st.write("")
        # on_click runs before the next script run, so the new empty
        # uploaders are drawn immediately.
        st.button("Start over with new photos", width="stretch", on_click=start_over)


# ------------------------------------------------------------------
# Step 1: upload & validation
# ------------------------------------------------------------------
def meter(label: str, value: float, lo: float | None, hi: float | None, scale_max: float) -> str:
    """One quality bar: the fill is the measured value, the tick is the threshold."""
    passed = (lo is None or value >= lo) and (hi is None or value <= hi)
    pct = max(2.0, min(100.0, value / scale_max * 100))
    marks = "".join(
        f'<span class="pr-mark" style="left:{min(99.0, t / scale_max * 100):.1f}%"></span>'
        for t in (lo, hi) if t is not None
    )
    return (f'<div class="pr-meter"><span class="label">{label}</span>'
            f'<div class="pr-track"><div class="pr-fill {"ok" if passed else "bad"}" style="width:{pct:.1f}%"></div>{marks}</div>'
            f'<span class="val">{value:.0f}</span></div>')


def render_quality(metrics: dict) -> None:
    """Sharpness, brightness and contrast as threshold meters."""
    render_html(
        '<div class="pr-meters">'
        + meter("Sharpness", float(metrics["sharpness"]), config.BLUR_VARIANCE_THRESHOLD, None,
                max(300.0, float(metrics["sharpness"])))
        + meter("Brightness", float(metrics["brightness"]), config.MIN_MEAN_BRIGHTNESS, config.MAX_MEAN_BRIGHTNESS, 255)
        + meter("Contrast", float(metrics["contrast"]), config.MIN_CONTRAST_STD * 0.6, None, 100)
        + "</div>"
    )


def render_upload_column(side: str):
    """One labelled upload panel with preview, quality meters and messages.

    Returns (filename, bytes, ValidationResult) or None if nothing uploaded.
    """
    with st.container(key=f"panel_{side}"):
        upload = st.session_state.get(uploader_key(side))
        result = cached_validation(upload.name, upload.getvalue()) if upload is not None else None
        badge = ('<span class="pr-badge wait">Waiting for photo</span>' if result is None else
                 '<span class="pr-badge ok">Ready</span>' if result.ok else
                 '<span class="pr-badge bad">Needs a new photo</span>')
        render_html(f'<div class="pr-panel-title"><span class="pr-ph">{side.title()} palm</span>{badge}</div>')

        upload = st.file_uploader(
            f"Photo of your {side} palm",
            type=list(config.ALLOWED_EXTENSIONS),
            key=uploader_key(side),
            help="JPG or PNG. Palm facing the camera, fingers together, whole hand and wrist in the frame.",
        )
        if upload is None:
            return None

        data = upload.getvalue()
        with st.spinner(f"Checking the {side} palm photo..."):
            result = cached_validation(upload.name, data)

        if result.image_bgr is not None:
            # A narrow centre column keeps tall phone photos from dominating the panel.
            _, mid, _ = st.columns([1, 1.6, 1])
            mid.image(to_display_rgb(result.image_bgr), width="stretch")
        if "sharpness" in result.metrics:
            render_quality(result.metrics)

        if result.ok:
            st.success("Photo passed all quality checks.")
        else:
            for error in result.errors:
                st.error(error)
            st.caption("Upload a new photo above to replace this one.")
        for warning in result.warnings:
            st.warning(warning)
        return upload.name, data, result


def render_photo_tips() -> None:
    """Short guidance shown until both photos are accepted."""
    with st.expander("How to take a good palm photo"):
        st.markdown(
            "- Open palm facing the camera, fingers together and pointing up\n"
            "- Whole hand and wrist inside the frame\n"
            "- Even daylight; no flash and no harsh shadows\n"
            "- A plain background that contrasts with your skin\n"
            "- Hold the phone steady and tap the screen to focus on your palm"
        )


# ------------------------------------------------------------------
# Step 2: image processing
# ------------------------------------------------------------------
def render_processing(results: dict[str, dict]) -> None:
    """Original-vs-processed comparison for every technique."""
    st.caption("Each technique is shown next to the original photo, in the order the pipeline applies them.")
    side = st.segmented_control("Palm", SIDES, format_func=str.title, default="left", key="proc_side") or "left"
    prep = results[side]["prep"]
    det = results[side]["detection"]
    original = to_display_rgb(crop_to_hand(prep.resized_bgr, det))

    for i, (title, summary, indices) in enumerate(STAGE_GROUPS, start=1):
        with st.expander(f"{i}. {title}", expanded=(i == 1)):
            st.caption(summary)
            for idx in indices:
                stage = prep.stages[idx]
                left_col, right_col = st.columns(2)
                left_col.image(original, caption="Original", width="stretch")
                right_col.image(to_display_rgb(crop_to_hand(stage.image, det)), caption=stage.title,
                                width="stretch")
                right_col.caption(stage.explanation)


# ------------------------------------------------------------------
# Step 3: line detection
# ------------------------------------------------------------------
def line_table(features: dict) -> str:
    """HTML table of the four major lines, with swatches matching the overlay colours."""
    rows = []
    for name in LINE_ORDER:
        line = features["lines"][name]
        swatch = f'<span class="pr-sw" style="background:{LINE_HEX[name]}"></span>'
        if line["detected"]:
            status = "Detected" if line.get("status") != "uncertain" else "Uncertain (weak evidence)"
            cells = [line.get("length", ""), line.get("curvature", ""), line.get("depth", "")]
            rows.append(f"<tr><td>{swatch}{name.title()}</td><td>{status}</td>"
                        + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
                        + f'<td class="pr-conf">{line["confidence"]:.2f}</td></tr>')
        else:
            rows.append(f'<tr><td>{swatch}{name.title()}</td><td class="pr-dim">Not clearly detected</td>'
                        f'<td class="pr-dim">-</td><td class="pr-dim">-</td><td class="pr-dim">-</td>'
                        f'<td class="pr-conf pr-dim">{line["confidence"]:.2f}</td></tr>')
    return ('<table class="pr-lines"><thead><tr><th>Line</th><th>Status</th><th>Length</th>'
            '<th>Curvature</th><th>Depth</th><th>Confidence</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def render_features(results: dict[str, dict]) -> None:
    """Annotated images, detection details and JSON for both palms."""
    st.caption("Everything here is measured by computer vision. No AI interpretation is involved yet.")
    columns = st.columns(2, gap="large")
    for side, col in zip(SIDES, columns):
        res = results[side]
        det, feat = res["detection"], res["features"]
        features = feat.features
        with col:
            st.subheader(f"{side.title()} palm")
            st.image(to_display_rgb(crop_to_hand(feat.annotated_bgr, det)), width="stretch")
            if det.method != "mediapipe":
                st.warning(det.message)
            for warning in det.warnings:
                st.warning(warning)

            render_html(line_table(features))
            render_html(
                '<div class="pr-facts">'
                f"<span>Palm shape <b>{html.escape(features['palm_shape'])}</b></span>"
                f"<span>Element <b>{html.escape(features['palm_element'])}</b></span>"
                f"<span>Other creases <b>{len(features['prominent_creases'])}</b></span>"
                f"<span>Hand located by <b>{html.escape(det.method)}</b></span></div>"
            )

            with st.expander("How the lines were found"):
                st.image(to_display_rgb(crop_to_hand(res["overlay"], det)),
                         caption="MediaPipe hand landmarks and palm region",
                         width="stretch")
                for stage in feat.stages:
                    st.image(to_display_rgb(stage.image), caption=stage.title, width="stretch")
                    st.caption(stage.explanation)
            with st.expander("Feature data (JSON)"):
                st.json(features)
                st.download_button("Download JSON", json.dumps(features, indent=2),
                                   file_name=f"{side}_palm_features.json", mime="application/json",
                                   key=f"dl_json_{side}")


# ------------------------------------------------------------------
# Step 4: reading
# ------------------------------------------------------------------
def render_report(results: dict[str, dict], llm_problem: str | None) -> None:
    """Generate and display the RAG palm reading."""
    left = results["left"]["features"].features
    right = results["right"]["features"].features
    report = st.session_state.report

    label = "Write a new reading" if report else "Generate palm reading"
    if st.button(label, type="primary", disabled=bool(llm_problem)):
        with st.spinner("Looking up the palmistry knowledge base and writing your reading..."):
            try:
                report = generate_report(left, right, get_client())
                st.session_state.report = report
                st.session_state.chat = ChatSession(left, right, report.llm_text)
            except (LLMError, KnowledgeBaseError) as exc:
                st.error(str(exc))
    if llm_problem:
        st.error(llm_problem)

    if not report:
        st.caption("The reading compares both palms, using only the lines the app detected.")
        return

    for warning in report.warnings:
        st.warning(warning)
    with st.container(key="reading"):
        st.markdown(report.markdown, unsafe_allow_html=True)
    st.download_button("Download reading", report.markdown, file_name="palm_reading.md", mime="text/markdown")

    with st.expander("How this reading was generated"):
        st.markdown(
            f"- **Model:** `{report.model_label}`\n"
            "- **Method:** retrieval-augmented generation over `knowledge/palmistry_kb.json`\n"
            "- **Input:** feature data for both palms plus the retrieved knowledge entries (no images)\n"
            "- **Output:** a reading with 8 fixed sections\n"
            f"- **Knowledge entries used ({len(report.kb_ids)}):** {', '.join(report.kb_ids)}"
        )
        st.markdown("**System prompt**")
        st.code(report.system_prompt, language="text")
        st.markdown("**User prompt (features and retrieved knowledge)**")
        st.code(report.user_prompt, language="text")


# ------------------------------------------------------------------
# Step 5: chat
# ------------------------------------------------------------------
def render_chat(llm_problem: str | None) -> None:
    """Multi-turn chat grounded in both palms' features and the reading."""
    session: ChatSession | None = st.session_state.chat
    if session is None:
        st.caption("Generate the palm reading first. The chat uses it, and your detected lines, as context.")
        return

    st.caption("Ask follow-up questions about your reading. Answers stick to the lines that were "
               "actually detected. Entertainment only.")
    clicked = None
    if not session.turns:
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for i, (col, question) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
            if col.button(question, key=f"example_{i}", width="stretch"):
                clicked = question

    for turn in session.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)

    question = st.chat_input("Ask about your palms", disabled=bool(llm_problem)) or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Writing an answer..."):
                try:
                    turn = ask(session, question, get_client())
                    st.markdown(turn.answer)
                    with st.expander("Sources used for this answer"):
                        st.markdown("**Knowledge entries:** " + (", ".join(turn.kb_ids) or "none"))
                        for note in turn.grounding_notes:
                            st.markdown(f"- {note}")
                except (LLMError, KnowledgeBaseError) as exc:
                    st.error(str(exc))

    if session.turns and st.button("Clear chat"):
        session.reset()
        st.rerun()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    """Lay out the workflow top to bottom."""
    render_html(CSS)
    init_state()
    client = get_client()
    llm_problem = config.check_llm_config(client.settings)

    render_header()
    stepper_slot = st.empty()
    render_html(f'<div class="pr-note">{html.escape(config.DISCLAIMER.replace("⚠️ ", "").replace("**", ""))}</div>')

    section("Your palms", "One photo of each palm, facing the camera")
    uploads = {}
    for side, col in zip(SIDES, st.columns(2, gap="large")):
        with col:
            uploads[side] = render_upload_column(side)
    both_valid = all(u is not None and u[2].ok for u in uploads.values())
    if not both_valid:
        render_photo_tips()

    # A changed photo invalidates the old reading and chat.
    signature = None
    if both_valid:
        signature = hashlib.sha256(uploads["left"][1] + b"|" + uploads["right"][1]).hexdigest()
    if signature != st.session_state.signature:
        st.session_state.signature = signature
        reset_results()

    results: dict[str, dict] = {}
    if both_valid:
        with st.spinner("Processing images and tracing palm lines..."):
            for side in SIDES:
                name, data, _ = uploads[side]
                results[side] = cached_pipeline(name, data, side)
        if config.SAVE_OUTPUTS and st.session_state.saved_sig != signature:
            for side in SIDES:
                feat = results[side]["features"]
                save_outputs(side, feat.annotated_bgr, feat.features)
            st.session_state.saved_sig = signature

    render_sidebar(llm_problem, client.label)

    section("Analysis", "From image processing to your reading")
    tabs = st.tabs(["Image processing", "Line detection", "Reading", "Chat"])
    if both_valid:
        with tabs[0]:
            render_processing(results)
        with tabs[1]:
            render_features(results)
        with tabs[2]:
            render_report(results, llm_problem)
        with tabs[3]:
            render_chat(llm_problem)
    else:
        for tab in tabs:
            tab.info("Upload two photos that pass the quality checks to unlock this step.")

    # Drawn last so it reflects anything that happened during this run
    # (e.g. a reading generated by the button above).
    with stepper_slot:
        render_stepper({
            "Upload": all(u is not None for u in uploads.values()),
            "Check photos": both_valid,
            "Trace lines": bool(results),
            "Reading": st.session_state.report is not None,
            "Chat": bool(st.session_state.chat and st.session_state.chat.turns),
        })


main()
