"""
feature_extraction.py  (Task 03, part 2)
----------------------------------------
Finds the major palm lines, palm shape and prominent creases, draws them on
the photo, and returns a structured JSON record per palm.

Method:
    1. NORMALISE THE PALM. The palm is warped from the ORIGINAL photo into
       an upright "canonical" frame (hand_detection.py): finger bases at the
       top, wrist at the bottom, thumb on the LEFT for both hands, and the
       palm always about 330 px wide. Every filter size below therefore
       means the same thing for every photo, whatever its resolution.

    2. CREASE RESPONSE. The Task 02 techniques are applied again to this
       palm ROI: bilateral denoising -> CLAHE -> morphological black-hat,
       which responds to thin dark lines (creases) and ignores broad
       shadows. The response is normalised to 0-1 inside the palm.

    3. MINIMAL-PATH LINE TRACING. Each major line is traced as the cheapest
       path between two anatomical regions, where walking along a crease is
       cheap and crossing plain skin is expensive (Dijkstra's algorithm via
       scikit-image's MCP). E.g. the heart line runs from the little-finger
       edge to below the index/middle fingers; the life line from between
       thumb and index finger down towards the wrist.
       WHY: palms have hundreds of small creases. Searching for "any long
       line" picks the wrong ones; constraining each line's start and end
       to where palmistry diagrams place it makes the search robust.

    4. VERIFY. A path always exists, so only the longest stretch that truly
       follows a crease is kept. A line is DETECTED only if that stretch is
       long enough, well supported and dark enough in absolute grey levels
       (so faint skin texture cannot pass as a crease); otherwise "not
       clearly detected".

    5. DESCRIBE detected lines by measurable attributes: length (relative to
       palm size), curvature (sagitta / chord ratio), depth (local contrast)
       and position notes. Other long creases are found with the
       probabilistic Hough transform. Palm shape comes from landmark ratios.

Everything in this file is OBSERVATION from computer vision. Interpretation
happens only later, in llm_engine.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

import config
from modules.hand_detection import (
    CANON_H,
    CANON_W,
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_CMC,
    WRIST,
    HandDetection,
    canonical_to_image,
    palm_polygon_canonical,
    warp_to_canonical,
)
from modules.preprocessing import (
    PreprocessResult,
    Stage,
    apply_clahe,
    bilateral_denoise,
    blackhat_enhance,
    morph_close,
    morph_open,
    remove_small_components,
    to_grayscale,
)

LINE_NAMES: tuple[str, ...] = ("heart", "head", "life", "fate")
Box = tuple[int, int, int, int]   # (x0, y0, x1, y1) in canonical pixels


@dataclass(frozen=True)
class LineSpec:
    """Where a line starts and ends in the canonical frame, and its typical length.

    Paths start and end on the palm's EDGE inside these boxes, then get
    trimmed to the part that follows a crease. WHY the edge: if paths could
    start anywhere in a box, they would start at the box border nearest the
    other box and cut every line short.

    Canonical landmarks for reference: index base (160, 60), middle base
    (~255, 35), ring base (~340, 50), little-finger base (420, 100),
    wrist (270, 470), thumb base joint (~130, 410).
    """

    name: str
    start: Box
    end: Box
    expected_px: float
    angle_range: tuple[float, float]   # allowed direction of the accepted line, degrees from horizontal


# WHY these regions: they follow the standard palmistry diagram, expressed
# relative to the finger-base and wrist landmarks.
LINE_SPECS: dict[str, LineSpec] = {
    # little-finger edge  ->  top edge below the index / middle fingers
    "heart": LineSpec("heart", (410, 100, 500, 280), (90, 20, 300, 170), 270, (0, 40)),
    # edge between thumb and index  ->  little-finger edge, mid palm
    "head": LineSpec("head", (10, 60, 170, 250), (400, 160, 500, 390), 280, (0, 50)),
    # edge between thumb and index  ->  around the thumb mound to the wrist
    "life": LineSpec("life", (10, 60, 170, 250), (80, 390, 330, 520), 330, (35, 90)),
    # wrist edge  ->  top edge below the middle / ring fingers
    "fate": LineSpec("fate", (190, 410, 350, 520), (190, 10, 340, 150), 260, (55, 90)),
}
EDGE_BAND_PX = 16           # how close to the palm outline a path may start/end

# Palm reference sizes in canonical pixels (index-to-little base x 1.25, and
# wrist to middle base). Line length is reported relative to these.
PALM_WIDTH_REF = 330.0
PALM_LENGTH_REF = 435.0

BLACKHAT_KERNEL = 17        # ~5% of palm width: wider than a crease, narrower than a shadow
# Absolute evidence thresholds, in grey levels of the black-hat response.
# Measured on test photos: real palm creases average 30-60 grey levels along
# the traced line; a palm with its creases smoothed away averages 1-2.
BLACKHAT_REFERENCE = 40.0   # response is scaled by at least this, so faint texture is never stretched to 1
MIN_LINE_STRENGTH = 12.0    # a line must be at least this dark on average to count as detected
CONFIDENT_LEVEL = 0.5       # detected lines below this confidence are reported as "uncertain"
ON_LINE_LEVEL = 0.22        # normalised crease response counted as "on a crease"
MAX_GAP_PX = 18             # breaks in a crease shorter than this are bridged
MIN_LENGTH_FRACTION = 0.30  # shortest acceptable line, as a fraction of expected length
REUSE_PENALTY = 4.0         # extra cost for a line re-using another line's crease
OFF_PALM_COST = 20.0        # discourages paths leaving the palm

CURVE_STRAIGHT = 0.05       # sagitta/chord below this -> "straight"
CURVE_SLIGHT = 0.12         # below this -> "slightly curved", above -> "curved"
DEPTH_DEEP = 18.0           # ring-minus-line contrast (grey levels) -> "deep"
DEPTH_MODERATE = 8.0        # -> "moderate"; below -> "faint"

# Palm-shape heuristics (documented as approximate in the README).
SQUARE_MAX_LW = 1.12        # palm length / width below this -> square
LONG_FINGER_RATIO = 0.85    # middle finger / palm length above this -> long fingers
# Taper thresholds were calibrated on sample photos: typical adult palms
# measured 1.7-1.9 with this landmark-based definition.
SPATULATE_TAPER = 2.15      # width at knuckles / lower palm above this -> spatulate
OVAL_TAPER = 1.45           # below this (lower palm relatively wide) -> oval


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------
@dataclass
class LineResult:
    """A detected (or not detected) major line."""

    name: str
    detected: bool
    confidence: float
    points: np.ndarray | None = None     # polyline, canonical frame
    path: np.ndarray | None = None       # full traced path (for the explanation view)
    attributes: dict[str, str | float] = field(default_factory=dict)


@dataclass
class FeatureResult:
    """Everything feature extraction produces for one palm."""

    features: dict
    annotated_bgr: np.ndarray            # lines drawn on the user's photo
    zones_bgr: np.ndarray                # canonical view with search regions and paths
    stages: list[Stage] = field(default_factory=list)   # palm-ROI processing steps


# ------------------------------------------------------------------
# Steps 1-2: palm ROI and crease response
# ------------------------------------------------------------------
def palm_region_mask(detection: HandDetection) -> np.ndarray:
    """Binary palm ROI in the canonical frame, shrunk slightly from its edges.

    WHY shrink: the boundary between palm and background is a strong edge
    that must not be traced as a palm line.
    """
    mask = np.zeros((CANON_H, CANON_W), np.uint8)
    poly = palm_polygon_canonical(detection)
    if len(poly):
        cv2.fillPoly(mask, [poly], 255)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))


def crease_response(canon_bgr: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bilateral -> CLAHE -> black-hat, scaled to 0-1 inside the palm.

    Returns (enhanced_gray, blackhat_display, response_0_to_1, blackhat_raw).
    `blackhat_raw` keeps real grey levels (how much darker a pixel is than
    its surroundings) and is the absolute evidence used to accept a line.
    WHY scale by max(99th percentile, a fixed reference): the percentile
    stops one very dark spot from squashing every crease; the fixed floor
    stops a smooth palm's faint skin texture from being stretched until it
    looks like strong creases - the cause of lines being "found" on palms
    that have none.
    """
    gray = to_grayscale(canon_bgr)
    enhanced = apply_clahe(bilateral_denoise(gray))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL, BLACKHAT_KERNEL))
    raw = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    raw[region == 0] = 0
    values = raw[region > 0]
    top = max(float(np.percentile(values, 99)) if values.size else 0.0, BLACKHAT_REFERENCE)
    response = np.clip(raw.astype(np.float32) / top, 0.0, 1.0)
    display = blackhat_enhance(enhanced, BLACKHAT_KERNEL)
    display[region == 0] = 0
    return enhanced, display, response, raw


def crease_map(response: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Binary map of strong creases: threshold, then opening and closing.

    Used for prominent-crease detection and shown in the UI.
    """
    binary = np.where(response >= ON_LINE_LEVEL + 0.1, 255, 0).astype(np.uint8)
    binary = morph_close(morph_open(binary))
    return cv2.bitwise_and(remove_small_components(binary, 60), region)


# ------------------------------------------------------------------
# Step 3: minimal-path tracing
# ------------------------------------------------------------------
def cost_image(response: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Walking cost per pixel: cheap on creases, expensive on plain skin.

    WHY a cubic: it makes the cost of a strong crease (response ~1) about
    30x lower than plain skin (response ~0), so paths stick to creases, but
    a short crossing of skin to bridge a break in a line is still allowed.
    """
    cost = 0.03 + (1.0 - response) ** 3
    cost[region == 0] = OFF_PALM_COST
    return cost.astype(np.float64)


def edge_band(region: np.ndarray) -> np.ndarray:
    """Pixels of the palm ROI that lie within EDGE_BAND_PX of its outline."""
    k = 2 * EDGE_BAND_PX + 1
    inner = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.subtract(region, inner)


def box_pixels(box: Box, allowed: np.ndarray, fallback: np.ndarray, step: int = 2) -> list[tuple[int, int]]:
    """Grid of (row, col) points inside a box where `allowed` is set.

    Falls back to `fallback` (the whole palm) if the box misses the palm edge,
    which can happen with unusual hand poses.
    """
    x0, y0, x1, y1 = box
    ys = range(max(0, y0), min(CANON_H, y1), step)
    xs = range(max(0, x0), min(CANON_W, x1), step)
    for mask in (allowed, fallback):
        pts = [(y, x) for y in ys for x in xs if mask[y, x]]
        if pts:
            return pts
    return []


def trace_path(cost: np.ndarray, spec: LineSpec, region: np.ndarray) -> np.ndarray | None:
    """Cheapest path from the line's start region to its end region.

    Returns an (N, 2) array of (x, y) points, or None if no path exists.
    """
    band = edge_band(region)
    starts = box_pixels(spec.start, band, region)
    ends = box_pixels(spec.end, band, region)
    if not starts or not ends:
        return None
    mcp = MCP_Geometric(cost, fully_connected=True)
    cumulative, _ = mcp.find_costs(starts, ends, find_all_ends=False)
    end_costs = [cumulative[e] for e in ends]
    best = int(np.argmin(end_costs))
    if not np.isfinite(end_costs[best]):
        return None
    path = np.array(mcp.traceback(ends[best]))           # (row, col)
    return path[:, ::-1].astype(np.float32)              # -> (x, y)


# ------------------------------------------------------------------
# Step 4: keep only the part that follows a real crease
# ------------------------------------------------------------------
def supported_segment(path: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, float]:
    """Longest stretch of the path lying on a crease, bridging small gaps.

    Returns (segment_points, coverage) where coverage is the share of the
    segment's pixels that are on a crease.
    WHY: the path is forced to connect both regions, so where the real line
    stops, the path continues across plain skin. Trimming to the supported
    part means the reported length reflects the actual crease.
    """
    # A 3x3 max-filter tolerates the path running a pixel beside the crease centre.
    local = cv2.dilate(response, np.ones((3, 3), np.uint8))
    xs = np.clip(path[:, 0].astype(int), 0, CANON_W - 1)
    ys = np.clip(path[:, 1].astype(int), 0, CANON_H - 1)
    on = local[ys, xs] >= ON_LINE_LEVEL

    best, best_len = (0, 0), 0.0
    start, gap = None, 0
    steps = np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1)]
    for i, flag in enumerate(on):
        if flag:
            if start is None:
                start = i
            gap = 0
            length = float(steps[start + 1:i + 1].sum())
            if length > best_len:
                best, best_len = (start, i), length
        elif start is not None:
            gap += 1
            if gap > MAX_GAP_PX:
                start, gap = None, 0

    a, b = best
    if b <= a:
        return path[:1], 0.0
    segment = path[a:b + 1]
    return segment, float(on[a:b + 1].mean())


def smooth_polyline(points: np.ndarray, window: int = 15, samples: int = 40) -> np.ndarray:
    """Moving-average smoothing and even resampling of a pixel path.

    WHY: pixel paths zig-zag by one pixel at a time, which would inflate
    curvature and look jagged when drawn. Unlike a polynomial fit, a moving
    average keeps the line's real shape (e.g. an S-bend).
    """
    if len(points) < 3:
        return points.astype(np.float32)
    window = max(3, min(window, len(points) // 2 * 2 - 1))
    kernel = np.ones(window) / window
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    smooth = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in (0, 1)])
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
    if dist[-1] == 0:
        return smooth.astype(np.float32)
    target = np.linspace(0, dist[-1], samples)
    return np.column_stack([np.interp(target, dist, smooth[:, k]) for k in (0, 1)]).astype(np.float32)


# ------------------------------------------------------------------
# Step 5: describing a detected line
# ------------------------------------------------------------------
def polyline_length(curve: np.ndarray) -> float:
    """Arc length = sum of distances between consecutive points."""
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def curvature_label(curve: np.ndarray) -> tuple[str, float]:
    """Classify curvature using the sagitta-to-chord ratio.

    WHY: the sagitta (largest distance between the curve and the straight
    chord joining its ends) divided by the chord length is scale-free, so a
    long and a short line with the same bend get the same label.
    """
    start, end = curve[0], curve[-1]
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-6:
        return "unknown", 0.0
    normal = np.array([-chord[1], chord[0]]) / chord_len
    ratio = float(np.abs((curve - start) @ normal).max()) / chord_len
    if ratio < CURVE_STRAIGHT:
        return "straight", ratio
    if ratio < CURVE_SLIGHT:
        return "slightly curved", ratio
    return "curved", ratio


def depth_label(curve: np.ndarray, enhanced: np.ndarray, region: np.ndarray) -> tuple[str, float]:
    """Estimate how deep/clear a line is from its local contrast.

    WHY contrast: a photo has no real depth, but deeper creases cast more
    shadow and look darker than the skin beside them. We compare the mean
    CLAHE intensity ON the line with a thin ring of skin around it.
    """
    line_mask = np.zeros(enhanced.shape, np.uint8)
    cv2.polylines(line_mask, [curve.astype(np.int32)], False, 255, 3)
    near = cv2.dilate(line_mask, np.ones((7, 7), np.uint8))
    far = cv2.dilate(line_mask, np.ones((17, 17), np.uint8))
    ring = cv2.bitwise_and(cv2.subtract(far, near), region)
    on_line = cv2.bitwise_and(line_mask, region)
    if cv2.countNonZero(ring) == 0 or cv2.countNonZero(on_line) == 0:
        return "unknown", 0.0
    contrast = float(enhanced[ring > 0].mean() - enhanced[on_line > 0].mean())
    if contrast >= DEPTH_DEEP:
        return "deep", contrast
    if contrast >= DEPTH_MODERATE:
        return "moderate", contrast
    return "faint", contrast


def length_label(curve: np.ndarray, name: str) -> tuple[str, float]:
    """Long / medium / short relative to palm width (heart, head) or length (life, fate)."""
    reference = PALM_WIDTH_REF if name in ("heart", "head") else PALM_LENGTH_REF
    ratio = polyline_length(curve) / reference
    if ratio >= config.LINE_LENGTH_LONG:
        return "long", ratio
    if ratio >= config.LINE_LENGTH_MEDIUM:
        return "medium", ratio
    return "short", ratio


def finger_region(x: float) -> str:
    """Which finger a canonical x-position sits beneath (index is left)."""
    if x < 207:
        return "index finger (Mount of Jupiter)"
    if x < 297:
        return "middle finger (Mount of Saturn)"
    if x < 380:
        return "ring finger (Mount of Apollo)"
    return "little finger (Mount of Mercury)"


def position_notes(name: str, curve: np.ndarray) -> dict[str, str]:
    """Position-based attributes that palmistry traditions refer to.

    WHY: tradition reads more than length - e.g. where the heart line ends,
    or whether the head line slopes. These are measurable geometric facts,
    so they belong to the observation stage, not the LLM.
    """
    notes: dict[str, str] = {}
    if name == "heart":
        notes["ends_under"] = finger_region(float(curve[:, 0].min()))
    elif name == "head":
        left, right = curve[np.argmin(curve[:, 0])], curve[np.argmax(curve[:, 0])]
        slope = abs(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
        notes["slope"] = "fairly horizontal" if slope < 15 else "sloping downward"
    elif name == "life":
        notes["arc"] = "wide arc into the palm" if curve[:, 0].max() > 210 else "close to the thumb"
    elif name == "fate":
        notes["starts_from"] = "near the wrist" if curve[:, 1].max() > 400 else "mid-palm"
    return notes


def line_strength(points: np.ndarray, raw: np.ndarray) -> float:
    """Mean black-hat value (grey levels darker than surroundings) along a line."""
    mask = np.zeros(raw.shape, np.uint8)
    cv2.polylines(mask, [points.astype(np.int32)], False, 255, 3)
    values = raw[mask > 0]
    return float(values.mean()) if values.size else 0.0


def detect_line(name: str, cost: np.ndarray, response: np.ndarray,
                enhanced: np.ndarray, region: np.ndarray, raw: np.ndarray) -> LineResult:
    """Trace, verify and describe one major line.

    Confidence combines three kinds of image evidence:
        length   - how much of the expected line length a crease supports,
        coverage - share of that stretch actually lying on a crease,
        strength - how dark the crease is in absolute grey levels.
    A line that fails any minimum is reported as not clearly detected; its
    coordinates are never shown or passed on.
    """
    spec = LINE_SPECS[name]
    path = trace_path(cost, spec, region)
    if path is None:
        return LineResult(name, False, 0.0)

    segment, coverage = supported_segment(path, response)
    length = polyline_length(segment) if len(segment) > 1 else 0.0
    strength = line_strength(segment, raw) if len(segment) > 1 else 0.0
    strength_factor = min(1.0, strength / BLACKHAT_REFERENCE)
    confidence = round(min(1.0, length / spec.expected_px) * coverage * (0.4 + 0.6 * strength_factor), 2)
    if (length < MIN_LENGTH_FRACTION * spec.expected_px or strength < MIN_LINE_STRENGTH
            or confidence < config.LINE_CONFIDENCE_THRESHOLD):
        return LineResult(name, False, confidence, path=path)

    # Orientation check. WHY: with no real fate line, the cheapest "fate"
    # path can run sideways along the head line. A fate line must be mostly
    # vertical, a heart line mostly horizontal, and so on.
    chord = segment[-1] - segment[0]
    direction = abs(np.degrees(np.arctan2(chord[1], chord[0]))) % 180
    direction = min(direction, 180 - direction)
    lo, hi = spec.angle_range
    if not lo <= direction <= hi:
        return LineResult(name, False, round(confidence * 0.5, 2), path=path)

    curve = smooth_polyline(segment)
    length_lbl, length_ratio = length_label(curve, name)
    curvature, sagitta = curvature_label(curve)
    depth, contrast = depth_label(curve, enhanced, region)
    attrs: dict[str, str | float] = {
        "strength": round(strength, 1),
        "length": length_lbl,
        "curvature": curvature,
        "depth": depth,
        "length_ratio": round(length_ratio, 2),
        "sagitta_ratio": round(sagitta, 3),
        "contrast": round(contrast, 1),
    }
    attrs.update(position_notes(name, curve))
    return LineResult(name, True, confidence, curve, path, attrs)


def line_mask(points: np.ndarray, thickness: int = 13) -> np.ndarray:
    """Thick mask along a line, used to penalise or exclude re-use of its crease."""
    mask = np.zeros((CANON_H, CANON_W), np.uint8)
    cv2.polylines(mask, [points.astype(np.int32)], False, 255, thickness)
    return mask


def detect_all_lines(response: np.ndarray, enhanced: np.ndarray, region: np.ndarray,
                     raw: np.ndarray) -> dict[str, LineResult]:
    """Detect the four major lines, preventing two lines from sharing one crease.

    WHY the order and penalties: the heart line is found first; the head
    line then pays extra to re-use the heart line's crease (they are
    different lines). The life line pays extra to re-use the head line,
    except near their shared starting point, where the two lines genuinely
    begin together in many hands. The fate line pays extra to run along any
    other line; crossing one (which fate lines do naturally) adds only a
    small cost because the crossing is short.
    """
    base = cost_image(response, region)
    lines: dict[str, LineResult] = {}

    lines["heart"] = detect_line("heart", base, response, enhanced, region, raw)

    head_cost = base.copy()
    if lines["heart"].detected:
        head_cost[line_mask(lines["heart"].points) > 0] += REUSE_PENALTY
    lines["head"] = detect_line("head", head_cost, response, enhanced, region, raw)

    life_cost = base.copy()
    if lines["head"].detected:
        penalty = line_mask(lines["head"].points)
        x0, y0, x1, y1 = LINE_SPECS["life"].start
        penalty[y0:y1 + 40, x0:x1 + 40] = 0
        life_cost[penalty > 0] += REUSE_PENALTY
    lines["life"] = detect_line("life", life_cost, response, enhanced, region, raw)

    fate_cost = base.copy()
    for other in ("heart", "head", "life"):
        if lines[other].detected:
            fate_cost[line_mask(lines[other].points) > 0] += REUSE_PENALTY
    lines["fate"] = detect_line("fate", fate_cost, response, enhanced, region, raw)

    # Sanity rule: the heart line lies above the head line. If it does not,
    # the heart candidate was probably the head line, so it is not trusted.
    heart, head = lines["heart"], lines["head"]
    if heart.detected and head.detected and heart.points[:, 1].mean() > head.points[:, 1].mean():
        lines["heart"] = LineResult("heart", False, heart.confidence, path=heart.path)
    return lines


# ------------------------------------------------------------------
# Palm shape
# ------------------------------------------------------------------
def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to the line through a and b."""
    ab = b - a
    return float(abs(ab[0] * (p - a)[1] - ab[1] * (p - a)[0]) / (np.linalg.norm(ab) + 1e-6))


def classify_palm_shape(detection: HandDetection) -> dict:
    """Approximate palm shape from hand-landmark proportions.

    WHY landmarks: they are measured on the original (undistorted) photo and
    do not depend on the background, unlike a skin-colour outline, which a
    wooden table can confuse. All rules are documented heuristics:
        - length / width separates square from rectangular palms;
        - taper = palm width at the knuckles / width of the lower palm
          (thumb base joint to the little-finger edge). A palm much wider at
          the fingers is spatulate; one noticeably wider lower down is oval;
        - the traditional "element" combines squareness with finger length.
    """
    m = detection.measurements
    result: dict = {"length_to_width": m.get("length_to_width"), "finger_to_palm": m.get("finger_to_palm")}
    lm = detection.landmarks
    if lm is None or not m.get("length_to_width"):
        result.update(shape="unknown", element="unknown")
        return result

    knuckle_width = float(np.linalg.norm(lm[INDEX_MCP] - lm[PINKY_MCP])) * 1.25
    lower_width = _point_line_distance(lm[THUMB_CMC], lm[WRIST], lm[PINKY_MCP]) * 1.15
    taper = knuckle_width / lower_width if lower_width else 0.0
    lw = float(m["length_to_width"])
    result["taper"] = round(taper, 3)

    if taper > SPATULATE_TAPER:
        shape = "spatulate"
    elif 0 < taper < OVAL_TAPER:
        shape = "oval"
    elif lw < SQUARE_MAX_LW:
        shape = "square"
    else:
        shape = "rectangular"

    finger = m.get("finger_to_palm")
    if finger is None:
        element = "unknown"
    else:
        element = {
            (True, False): "earth", (True, True): "air",
            (False, False): "fire", (False, True): "water",
        }[(lw < SQUARE_MAX_LW, float(finger) > LONG_FINGER_RATIO)]
    result.update(shape=shape, element=element)
    return result


# ------------------------------------------------------------------
# Prominent creases (Hough)
# ------------------------------------------------------------------
def palm_region_name(x: float, y: float) -> str:
    """Name the traditional palm area (mount) at a canonical position."""
    if y < 150:
        return finger_region(x)
    if x < 190 and y > 260:
        return "thumb base (Mount of Venus)"
    if x > 330 and y > 260:
        return "outer lower palm (Mount of Luna)"
    return "centre of the palm (Plain of Mars)"


def find_creases(creases_binary: np.ndarray, exclude: np.ndarray) -> list[dict]:
    """Detect other prominent straight creases with the probabilistic Hough transform.

    WHY Hough: it votes for straight segments in parameter space, so it
    finds long straight creases even when they are broken into dashes.
    Pixels already assigned to a major line are masked out first, so the
    same crease is not reported twice.
    """
    remaining = cv2.bitwise_and(creases_binary, cv2.bitwise_not(exclude))
    segments = cv2.HoughLinesP(remaining, rho=1, theta=np.pi / 180, threshold=30,
                               minLineLength=config.MIN_CREASE_LENGTH_PX * 1.5, maxLineGap=8)
    if segments is None:
        return []

    creases: list[dict] = []
    for x1, y1, x2, y2 in sorted(segments[:, 0], key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        angle = min(angle, 180 - angle)
        # Skip near-duplicates: Hough often returns overlapping segments.
        if any(np.linalg.norm(mid - c["_mid"]) < 20 and abs(angle - c["_angle"]) < 15 for c in creases):
            continue
        creases.append({
            "region": palm_region_name(*mid),
            "orientation": "horizontal" if angle < 30 else "vertical" if angle > 60 else "diagonal",
            "length_px": int(np.hypot(x2 - x1, y2 - y1)),
            "_mid": mid, "_angle": angle, "_seg": (int(x1), int(y1), int(x2), int(y2)),
        })
        if len(creases) >= config.MAX_REPORTED_CREASES:
            break
    return creases


# ------------------------------------------------------------------
# Annotation
# ------------------------------------------------------------------
LINE_LABELS = {"heart": "Heart", "head": "Head", "life": "Life", "fate": "Fate"}


def _text(img: np.ndarray, text: str, org: tuple[int, int], colour: tuple[int, int, int], scale: float = 0.55) -> None:
    """Draw readable text: a dark outline first, then the colour on top."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate_image(image_bgr: np.ndarray, detection: HandDetection, lines: dict[str, LineResult],
                   creases: list[dict], shape: str) -> np.ndarray:
    """Draw detected lines, creases, the palm ROI and a legend on the photo.

    WHY the legend lists undetected lines too: the brief requires absent or
    uncertain features to be marked explicitly, not silently left out.
    """
    out = image_bgr.copy()
    if detection.palm_polygon is not None:
        cv2.polylines(out, [detection.palm_polygon], True, config.LINE_COLORS["outline"], 1, cv2.LINE_AA)

    for crease in creases:
        x1, y1, x2, y2 = crease["_seg"]
        pts = canonical_to_image(np.float32([[x1, y1], [x2, y2]]), detection)
        cv2.line(out, tuple(pts[0]), tuple(pts[1]), config.LINE_COLORS["crease"], 1, cv2.LINE_AA)

    for name, line in lines.items():
        if not line.detected or line.points is None:
            continue
        pts = canonical_to_image(line.points, detection)
        colour = config.LINE_COLORS[name]
        cv2.polylines(out, [pts], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, colour, 2, cv2.LINE_AA)
        mid = pts[len(pts) // 2]
        _text(out, LINE_LABELS[name], (int(mid[0]) + 6, int(mid[1]) - 6), colour, 0.45)

    # Legend: every line is listed, detected or not.
    rows = [(f"{LINE_LABELS[n]}: " + ("detected" if lines[n].detected else "not clearly detected"),
             config.LINE_COLORS[n] if lines[n].detected else (160, 160, 160)) for n in LINE_NAMES]
    rows.append((f"Palm shape: {shape}", (255, 255, 255)))
    rows.append(("Source: computer vision", (255, 255, 255)))
    box_h = 20 * len(rows) + 10
    overlay = out.copy()
    cv2.rectangle(overlay, (6, 6), (250, 6 + box_h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)
    for i, (text, colour) in enumerate(rows):
        _text(out, text, (12, 26 + 20 * i), colour, 0.45)
    return out


def draw_search_view(canon_bgr: np.ndarray, lines: dict[str, LineResult]) -> np.ndarray:
    """Upright palm with each line's start/end regions, traced path and accepted part.

    WHY: explains HOW each line was found - useful for the viva and report.
    Thin line = full minimal path; thick line = the part supported by a crease.
    """
    view = cv2.addWeighted(canon_bgr, 0.55, np.zeros_like(canon_bgr), 0.45, 0)
    for name, spec in LINE_SPECS.items():
        colour = config.LINE_COLORS[name]
        for tag, (x0, y0, x1, y1) in (("start", spec.start), ("end", spec.end)):
            cv2.rectangle(view, (x0, y0), (x1, y1), colour, 1)
            _text(view, f"{name} {tag}", (x0 + 3, y0 + 13), colour, 0.38)
        line = lines[name]
        if line.path is not None:
            cv2.polylines(view, [line.path.astype(np.int32)], False, colour, 1, cv2.LINE_AA)
        if line.detected and line.points is not None:
            cv2.polylines(view, [line.points.astype(np.int32)], False, colour, 4, cv2.LINE_AA)
    return view


# ------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------
def build_feature_json(palm_side: str, detection: HandDetection, lines: dict[str, LineResult],
                       shape_info: dict, creases: list[dict]) -> dict:
    """Assemble the structured feature record sent to the LLM.

    WHY include confidence and source: the LLM and the chatbot must be able
    to tell what was measured, how sure the measurement is, and what was not
    seen at all.
    """
    line_json: dict[str, dict] = {}
    for name in LINE_NAMES:
        line = lines[name]
        if line.detected:
            confident = line.confidence >= CONFIDENT_LEVEL
            entry = {"detected": True, "status": "detected" if confident else "uncertain",
                     "confidence": line.confidence}
            entry.update(dict(line.attributes))
            entry["notes"] = ("visually detected by computer vision" if confident
                              else "weak evidence - treat as uncertain, not as a confirmed feature")
        else:
            entry = {"detected": False, "status": "not_detected", "confidence": line.confidence,
                     "notes": "not clearly detected"}
        line_json[name] = entry

    missing = [n for n in LINE_NAMES if not lines[n].detected]
    uncertain = [n for n in LINE_NAMES if lines[n].detected and lines[n].confidence < CONFIDENT_LEVEL]
    notes = [f"Palm located with {detection.method}."]
    if missing:
        notes.append("Not clearly detected: " + ", ".join(missing) + " line(s).")
    if uncertain:
        notes.append("Low confidence: " + ", ".join(uncertain) + " line(s).")
    notes.append("Depth is estimated from image contrast, not physical depth.")

    return {
        "palm_side": palm_side,
        "source": "computer_vision",
        "detection_method": detection.method,
        "palm_shape": shape_info.get("shape", "unknown"),
        "palm_element": shape_info.get("element", "unknown"),
        "shape_measurements": {k: v for k, v in shape_info.items() if k not in ("shape", "element")},
        "lines": line_json,
        "prominent_creases": [{k: v for k, v in c.items() if not k.startswith("_")} for c in creases],
        "confidence_notes": " ".join(notes),
        "warnings": list(detection.warnings),
    }


def empty_features(palm_side: str, reason: str) -> dict:
    """Feature record when no hand could be analysed - every line marked absent."""
    return {
        "palm_side": palm_side,
        "source": "computer_vision",
        "detection_method": "none",
        "palm_shape": "unknown",
        "palm_element": "unknown",
        "shape_measurements": {},
        "lines": {n: {"detected": False, "status": "not_detected", "confidence": 0.0,
                      "notes": "not clearly detected"} for n in LINE_NAMES},
        "prominent_creases": [],
        "confidence_notes": reason,
        "warnings": [reason],
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def extract_features(prep: PreprocessResult, detection: HandDetection, palm_side: str) -> FeatureResult:
    """Run line detection, shape analysis and annotation for one palm."""
    if not detection.found or detection.to_canonical is None:
        reason = detection.message or "Hand not detected."
        return FeatureResult(empty_features(palm_side, reason), prep.resized_bgr.copy(),
                             np.zeros((CANON_H, CANON_W, 3), np.uint8))

    canon_bgr = warp_to_canonical(prep.source_bgr, detection, scale=prep.scale)
    region = palm_region_mask(detection)
    enhanced, blackhat, response, raw = crease_response(canon_bgr, region)
    creases_binary = crease_map(response, region)

    lines = detect_all_lines(response, enhanced, region, raw)

    exclude = np.zeros((CANON_H, CANON_W), np.uint8)
    for line in lines.values():
        if line.detected:
            exclude |= line_mask(line.points, 21)
    creases = find_creases(creases_binary, exclude)

    shape_info = classify_palm_shape(detection)
    features = build_feature_json(palm_side, detection, lines, shape_info, creases)
    annotated = annotate_image(prep.resized_bgr, detection, lines, creases, features["palm_shape"])
    search_view = draw_search_view(canon_bgr, lines)

    stages = [
        Stage("A. Palm ROI warped upright (full resolution)", canon_bgr,
              "Affine warp from three landmarks: same size and orientation for every palm, thumb on the left."),
        Stage("B. Bilateral filter + CLAHE", enhanced,
              "Task 02 techniques re-applied to the palm only, at a consistent scale."),
        Stage("C. Morphological black-hat", blackhat,
              "Bright = thin dark creases. Broad shadows are removed."),
        Stage("D. Crease map (threshold + open/close)", creases_binary,
              "Strong creases inside the palm ROI; used for prominent-crease detection."),
        Stage("E. Minimal-path line search", search_view,
              "Boxes = where each line starts/ends. Thin = cheapest path along creases; thick = part accepted as the line."),
    ]
    return FeatureResult(features, annotated, search_view, stages)


def save_outputs(side: str, annotated_bgr: np.ndarray, features: dict) -> list[Path]:
    """Save the annotated image and JSON to assets/ (only when SAVE_OUTPUTS=true).

    WHY opt-in: the brief asks for minimal retention of personal images.
    """
    if not config.SAVE_OUTPUTS:
        return []
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = config.ASSETS_DIR / f"{stamp}_{side}_annotated.png"
    json_path = config.ASSETS_DIR / f"{stamp}_{side}_features.json"
    cv2.imwrite(str(img_path), annotated_bgr)
    json_path.write_text(json.dumps(features, indent=2), encoding="utf-8")
    return [img_path, json_path]
