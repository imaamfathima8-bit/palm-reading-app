# AI-Powered Palm Reading and Intelligent Chatbot Application

**Module:** Digital Image Processing — NIBM Kandy Innovation Centre (ICA 01)

> ⚠️ **Disclaimer.** Palmistry is a cultural and entertainment practice. It is
> **not** a scientifically validated method for predicting personality, health,
> wealth, relationships or future events. Every interpretation produced by this
> application is labelled accordingly and must not be used for medical,
> financial, employment, relationship or any other consequential decision.

---

## 1. What the application does

A single Streamlit app that runs the full workflow with no separate scripts:

```
Upload LEFT + RIGHT palm
        │
        ▼
1. Validate      → file type, corruption, size, resolution, blur, lighting,
                   hand present / open / fully in frame (MediaPipe)
        │
        ▼
2. Preprocess    → resize/normalise, grayscale, Gaussian + bilateral denoise,
                   histogram equalisation / CLAHE, Sobel + Canny edges,
                   segmentation, adaptive + Otsu thresholding, morphology
        │
        ▼
3. Detect        → palm ROI from landmarks → upright palm frame →
                   minimal-path tracing of heart / head / life / fate lines,
                   evidence check, palm shape, prominent creases
        │
        ▼
4. Annotate      → coloured overlays + labels, structured JSON per palm
        │
        ▼
5. Retrieve      → matching entries from the local palmistry knowledge base (RAG)
        │
        ▼
6. Report        → LLM writes an 8-section comparative reading
        │
        ▼
7. Chat          → multi-turn assistant grounded in features + report
```

Every stage is visible in the app (original vs processed images side by
side, and a "How the lines were found" view), so the computer-vision pipeline
is transparent and can be explained in the viva.

---

## 2. Project structure

```
palm_reading_app/
├── app.py                    # Streamlit UI + workflow orchestration
├── config.py                 # .env loading, constants, thresholds
├── modules/
│   ├── __init__.py
│   ├── validation.py         # file format + quality checks
│   ├── preprocessing.py      # resize, gray, denoise, CLAHE, edges, threshold, morphology
│   ├── hand_detection.py     # MediaPipe landmarks, hand validation, palm ROI
│   ├── feature_extraction.py # line tracing, evidence check, annotation, JSON
│   ├── knowledge_base.py     # loads palmistry KB, retrieval function
│   ├── llm_engine.py         # LLM client, RAG prompt builder, report generator
│   └── chatbot.py            # chat state + grounded follow-up logic
├── knowledge/
│   └── palmistry_kb.json     # curated knowledge base (57 entries)
├── assets/                   # optional saved outputs (off by default)
├── .streamlit/
│   └── config.toml           # UI theme, upload size limit
├── .env.example              # settings template - placeholders only
├── .gitignore                # keeps .env, .venv and outputs out of Git
├── make_submission.py        # builds a clean submission ZIP (no secrets)
├── requirements.txt
└── README.md
```

---

## 3. Setup

### 3.1 Prerequisites
- **Python 3.10, 3.11 or 3.12.** MediaPipe 0.10.14 has no installers for 3.13+.
  Check with `python --version`.
- An API key for **one** LLM provider. Gemini has a free tier:
  create a key at <https://aistudio.google.com/apikey>.

### 3.2 Install

Open a terminal in the `palm_reading_app` folder (on Windows: type `cmd` in
File Explorer's address bar and press Enter), then:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The install takes 5–10 minutes the first time. Three versions are pinned on
purpose (see comments in `requirements.txt`): `mediapipe==0.10.14`,
`protobuf<5` (MediaPipe crashes with newer protobuf) and `streamlit==1.55.0`
(newer versions fail to display images with this protobuf version).

### 3.3 Configure your API key

1. Make a copy of **`.env.example`** and name the copy exactly **`.env`**:
   ```bash
   copy .env.example .env      # Windows
   cp .env.example .env        # macOS / Linux
   ```
   > Some browsers drop the leading dot when downloading, so the template may
   > arrive as `env.example`. That is fine - copy it to `.env` all the same.
   > On Windows, turn on **View → Show → File name extensions** in File
   > Explorer so you can see the full name; a file shown as `.env` must not
   > really be `.env.txt` or `New Text Document.env`.
2. Open `.env` in a text editor and set:
   ```
   MODEL_PROVIDER=gemini
   GEMINI_API_KEY=your-real-key
   GEMINI_MODEL=gemini-3.6-flash
   ```
   No quotes and no spaces. If the app finds a misnamed key file it tells you
   which file to rename.

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PROVIDER` | Which backend to call | `gemini` (`groq` and `openai` also supported) |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Google Gemini | `gemini-3.6-flash` |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq | `llama-3.3-70b-versatile` |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | OpenAI or a compatible server | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Lower = closer to the knowledge base (Groq/OpenAI) | `0.4` |
| `LLM_MAX_TOKENS` | Reply length ceiling (Groq/OpenAI) | `4096` |
| `SAVE_OUTPUTS` | Save annotated images + JSON to `assets/` | `false` |

**Model names.** As of October 2026 Google lists `gemini-3.6-flash` as the
current Flash model; the older `gemini-3.5-flash` still works because Google
routes it to 3.6. Model names change over time - if the app reports that the
model was not found, put a current name from Google's model list in `.env`.

API keys are read only through `os.getenv()` in `config.py` and are never
shown in the app or written into source code.

### 3.4 Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Keep the terminal open while using
it; press `Ctrl + C` to stop. To use it from a phone on the same Wi-Fi, open
the "Network URL" printed in the terminal.

### 3.5 Troubleshooting

| Symptom | Fix |
|---|---|
| "No API key found" / "Found a file called ..." | Check the `.env` name and contents (section 3.3), then restart the app. |
| "The model name ... was rejected" / "not found" | Set `GEMINI_MODEL=gemini-3.6-flash` in `.env`, restart. |
| `ImportError` for `cv2` | `pip uninstall -y opencv-python opencv-python-headless` then `pip install --force-reinstall opencv-contrib-python` |
| `MessageFactory ... GetPrototype` error | `pip install -r requirements.txt` again (installs `protobuf<5`). |
| Images show as "0" | `pip install -r requirements.txt` again (installs `streamlit==1.55.0`). |

---

## 4. How to use it

1. Upload a **left palm** and a **right palm** photo (JPG, JPEG or PNG, up to 15 MB).
2. Each panel shows the photo, sharpness / brightness / contrast meters (the
   tick marks the pass level) and either "Photo passed all quality checks" or
   a message saying exactly what to fix. Upload a new photo to replace it.
3. **Image processing** tab - every technique next to the original.
4. **Line detection** tab - annotated lines, a table with each line's status
   and confidence, palm shape, and "How the lines were found".
5. **Reading** tab - click **Generate palm reading**.
6. **Chat** tab - ask follow-up questions.
7. **Start over with new photos** (sidebar) clears both photos, all results,
   the reading and the chat.

### Photo tips
- Open hand, palm flat and facing the camera, fingers straight.
- Whole hand **and wrist** inside the frame.
- Even daylight; no flash, no harsh shadows. Plain background.
- Hold the phone steady and tap to focus on the palm.

---

## 5. Upload validation (Task 01)

| Check | How | Threshold |
|---|---|---|
| File type | extension **and** decoded format (a renamed text file is caught) | JPG, JPEG, PNG |
| Corruption | `PIL.Image.verify()` | must decode |
| Size | bytes | 1 byte – 15 MB |
| Resolution | shorter side | ≥ 400 px |
| Blur | variance of the Laplacian | ≥ 40 |
| Lighting | mean grey level | 60 – 220 |
| Contrast | standard deviation of grey levels | ≥ 15 (warning below 25) |
| Hand present | MediaPipe Hands landmarks | required |
| Hand fully visible | palm landmarks inside the frame | required |
| Hand large enough | landmark hull area | ≥ 8% of image |
| Hand open, palm facing camera | fingertips farther from wrist than middle joints; finger bases spread | required |

**Why MediaPipe is required.** Without landmarks the finger bases, thumb and
wrist cannot be located, so palm-line analysis would rest on guesses. When
MediaPipe finds no hand, a skin-segmentation outline (counting the gaps
between fingers) is used **only** to choose a more helpful error message -
e.g. "part of a hand, cut off by the edge" - never to analyse the image. A
skin-coloured oval or other object is therefore rejected.

---

## 6. Image processing techniques (Task 02)

| # | Technique | Why it is used here |
|---|---|---|
| 1 | Resize + normalise, RGB → grayscale | A fixed working size makes thresholds comparable across photos; palm lines are intensity features, so colour adds noise rather than information. |
| 2 | Gaussian blur and bilateral filter | Removes sensor noise and skin texture. The bilateral filter smooths flat skin while preserving the sharp edges of creases. |
| 3 | Histogram equalisation and CLAHE | Palm lines are low-contrast. CLAHE boosts local contrast tile by tile without blowing out bright regions as global equalisation does. |
| 4 | Sobel and Canny edge detection | Sobel gives gradient magnitude (edge strength); Canny gives thin, connected edges after non-maximum suppression and hysteresis. |
| 5 | Segmentation + thresholding | YCrCb skin segmentation separates the hand; adaptive thresholding and black-hat + Otsu isolate dark creases under uneven lighting. |
| 6 | Morphological opening / closing | Opening removes small specks of texture; closing bridges small gaps so each line becomes one connected structure. |

---

## 7. Palm feature detection (Task 03)

1. **Palm ROI.** MediaPipe Hands returns 21 landmarks. The palm ROI is the
   polygon through the wrist, thumb base and the four finger bases.
2. **Upright palm frame.** An affine warp from three landmarks (index base,
   little-finger base, wrist) turns every palm upright with the thumb on the
   left. It is warped from the original full-resolution photo, so the palm
   is always about 330 px wide and every filter size means the same thing
   for every photo.
3. **Crease response.** Bilateral filter → CLAHE → morphological black-hat
   (17 px kernel), which responds to thin dark creases and ignores broad
   shadows.
4. **Minimal-path tracing.** Each line is traced as the cheapest path
   (Dijkstra's algorithm, scikit-image `MCP_Geometric`) between two regions
   on the palm edge where palmistry diagrams place it - e.g. heart line: from
   the little-finger edge to below the index/middle fingers; life line: from
   between thumb and index down to the wrist. Walking along a crease is cheap,
   crossing plain skin is expensive, and lines pay a penalty for re-using
   another line's crease.
5. **Evidence check.** A path always exists, so only the longest stretch that
   follows a real crease is kept. A line is reported as **detected** only if
   that stretch is:
   - long enough (≥ 30% of the expected length),
   - **dark enough in absolute grey levels** (mean black-hat ≥ 12; real creases
     measured 30–60, a palm with its creases removed measured 1–2), and
   - pointing the expected way (e.g. a fate line must be mostly vertical).

   **Confidence** = length support × coverage × crease strength, so it falls
   as creases get fainter. Each line gets a `status`:
   `detected` (confidence ≥ 0.5), `uncertain` (weak evidence) or
   `not_detected` - shown as **"Not clearly detected"**, with no line drawn.
6. **Attributes** of detected lines: length (relative to palm size), curvature
   (sagitta-to-chord ratio), depth (local contrast - an approximation, not
   physical depth) and position notes (e.g. where the heart line ends).
7. **Prominent creases** - probabilistic Hough transform on the remaining
   crease map, labelled by palm region (mount).
8. **Palm shape** - from landmark proportions: palm length / width and width
   at the knuckles versus the lower palm; the element (earth/air/fire/water)
   also uses finger length. These are heuristics.

Everything on the Line detection tab is **computer-vision observation**.
Everything the LLM writes is marked **[Interpretation]**, so observation and
interpretation are never mixed.

### Feature JSON format (per palm)

```json
{
  "palm_side": "left",
  "source": "computer_vision",
  "detection_method": "mediapipe",
  "palm_shape": "oval|square|rectangular|spatulate|unknown",
  "palm_element": "earth|air|fire|water|unknown",
  "lines": {
    "heart": {"detected": true, "status": "detected", "confidence": 0.82,
              "strength": 62.2, "length": "medium", "curvature": "curved",
              "depth": "deep", "ends_under": "middle finger (Mount of Saturn)",
              "notes": "visually detected by computer vision"},
    "head":  {"...": "..."},
    "life":  {"...": "..."},
    "fate":  {"detected": false, "status": "not_detected", "confidence": 0.18,
              "notes": "not clearly detected"}
  },
  "prominent_creases": [{"region": "...", "orientation": "...", "length_px": 0}],
  "confidence_notes": "...",
  "warnings": []
}
```

---

## 8. LLM specialisation (Task 04)

| Item | Detail |
|---|---|
| **Model / provider** | Google Gemini `gemini-3.6-flash` (default) via the `google-genai` SDK. Groq (`llama-3.3-70b-versatile`) or any OpenAI-compatible model can be used instead by changing `.env`. |
| **Specialisation mechanism** | Retrieval-Augmented Generation (RAG) over a curated local knowledge base, `knowledge/palmistry_kb.json`. No fine-tuning. |
| **Knowledge base** | 57 entries: the heart, head, life and fate lines and their length / curvature / depth / position variants; palm shapes and elements; mounts; career, finance, relationship, personality and future themes; left-vs-right comparison; caveats. |
| **Retrieval strategy** | Deterministic tag matching: each detected attribute (e.g. `heart:long`, `heart:curved`) maps to tagged entries; undetected lines retrieve only their overview and the "absent line" note. Chat questions are matched by keyword. Entries are inserted into the prompt with IDs so claims can be cited, e.g. `[KB:heart-long]`. |
| **Input** | Feature JSON for both palms (no images) + retrieved entries. Only palms that passed hand detection can be sent. |
| **Output** | Markdown reading with 8 fixed sections: (1) major lines and characteristics, (2) personality, (3) career, (4) financial themes, (5) relationships and emotions, (6) future-oriented narratives (labelled non-scientific), (7) left vs right comparison, (8) uncertainty statements - plus disclaimer and a detection-status table added by code. |
| **Prompt template** | `REPORT_SYSTEM_PROMPT` and `REPORT_USER_TEMPLATE` in `modules/llm_engine.py`; the full prompts used for each reading are shown in the app under "How this reading was generated". |

### Safeguards against unsupported claims
1. **Grounding rules:** only lines with status `detected` or `uncertain` may
   be described; `uncertain` lines must be called uncertain; `not_detected`
   lines get only their general meaning.
2. **Knowledge-base restriction:** meanings must come from retrieved entries,
   cited by ID.
3. **Disclaimer added by code**, at the top and bottom of every reading.
4. **Detection-status table added by code**, independent of the LLM.
5. **Section check:** the app warns if any of the 8 sections is missing.
6. **Prediction check:** a code-level filter flags phrases such as "you will
   become rich" or "your lifespan" in readings and chat answers.
7. **No consequential advice:** the prompt forbids medical, financial, legal,
   career or relationship advice and any health/lifespan prediction.
8. **Validated input only:** a reading cannot be generated unless both palms
   passed validation and hand detection.
9. **Error handling:** missing/misnamed key, invalid model, rate limits,
   timeouts (90 s), network and server errors all produce a plain message;
   API keys are redacted from any error text.

---

## 9. Chatbot (Task 05)

- **Context:** both feature JSONs, the detection-status table and the
  generated reading are in the system prompt.
- **Per-question grounding:** the app retrieves knowledge-base entries for each
  question and, for every line the question mentions, attaches the detection
  facts (detected / uncertain / not detected) so the model cannot describe a
  line that was not seen.
- **Memory:** the last 10 exchanges are kept in `st.session_state` and sent
  with each turn.
- **Disclaimer:** shown above the chat; a reminder is appended to answers about
  the future, money or health.

Example questions (shown as buttons in the app):
- *What does my heart line traditionally represent?*
- *What differences are visible between my left and right palms?*
- *What does palmistry say about my career?*
- *Can you explain my life line in more detail?*
- *What are the traditional interpretations of my financial prospects?*

---

## 10. Testing

### Automated checks run during development
Run on generated images and one consenting volunteer's palm photos:
JPG / JPEG / PNG accepted; text file, corrupted JPEG, blurred, dark and
low-resolution images rejected with specific messages; skin-coloured ovals,
textured walls, random noise and a cut-off hand rejected; a real palm with
its creases digitally removed gave **no** detected lines; confidence fell
step by step as crease contrast was reduced; report and chat flows,
validated-input guard, prediction check and Start Over tested with a stubbed
LLM. Live Gemini calls require your own key and are tested manually.

### Manual tests required by the brief
At least three different people's left/right palm pairs, **with consent**, and
at least one error or low-quality case. Suggested matrix:

| Test | Input | Expected result |
|---|---|---|
| T1–T3 | Three clear left/right pairs | Checks pass; lines annotated; reading and chat work |
| T4 | Blurry photo | Rejected with blur message |
| T5 | Dark photo | Rejected with lighting message |
| T6 | Renamed text file or `.gif` | Rejected as unsupported/corrupted |
| T7 | Photo with no hand (wall, desk) | Rejected: no hand detected |
| T8 | Closed fist or curled fingers | Rejected: open your hand |
| T9 | Hand partly outside the frame | Rejected: hand cut off |
| T10 | Chat: ask about a line marked "Not clearly detected" | Bot says it was not detected |
| T11 | Start over, then a new pair | Both photos, results and chat cleared |

Record screenshots of each for the technical report.

---

## 11. Privacy, security and submission

- Uploaded images are processed **in memory**; nothing is written to disk
  unless `SAVE_OUTPUTS=true`. Only derived feature JSON (no images, no names)
  is sent to the LLM provider. No accounts or personal data are collected.
- The API key lives only in `.env`, which is listed in `.gitignore` and must
  never be shared, committed or zipped.
- **Build the submission ZIP with `python make_submission.py`** (from the
  project folder, with the virtual environment active). It creates
  `palm_reading_app_submission.zip` next to the project folder, leaves out
  `.env`, `.venv`, caches and saved outputs, and stops if any file contains
  something that looks like a real API key.
- If a key was ever shared (in a ZIP, chat or repository), delete it in
  Google AI Studio and create a new one.

---

## 12. Limitations

- Classical CV line detection is sensitive to lighting, camera angle, skin
  tone and very fine or very deep skin texture; a strong minor crease can be
  traced instead of a faint major line.
- Line search regions assume typical palm anatomy; unusual line positions may
  be reported as "Not clearly detected".
- Thresholds were calibrated on a small number of photos and should be
  re-checked on the three-person test set.
- Depth is approximated from image contrast, not true 3D depth. Palm-shape
  classification uses simple geometric ratios.
- MediaPipe needs an open hand, palm facing the camera and fully in frame.
- The LLM can still paraphrase loosely; grounding rules and code checks reduce
  but cannot eliminate this.
- Palmistry itself has no scientific predictive validity.

---

## 13. Acknowledgements (third-party components)

- [OpenCV](https://opencv.org/) — image processing
- [MediaPipe Hands](https://ai.google.dev/edge/mediapipe) — hand landmark detection (Google)
- [scikit-image](https://scikit-image.org/) — minimal-cost path search (`skimage.graph.MCP_Geometric`)
- [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/)
- [Streamlit](https://streamlit.io/) — user interface
- Google Gemini (`google-genai`), Groq and OpenAI APIs — language model inference
- Fonts: Public Sans and Literata (Google Fonts)
- Parts of the source code were generated with the assistance of an AI model
  (Anthropic Claude) and reviewed, tested and adapted by the student(s).
