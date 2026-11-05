"""
hand_detection.py  (Task 03, part 1)
------------------------------------
Finds the hand, extracts the palm region of interest (ROI), and maps the palm
into an upright "canonical" frame so every palm can be analysed the same way.

Method:
    1. MediaPipe Hands (preferred) returns 21 landmarks per hand:
           0 = wrist, 1-4 = thumb, 5/9/13/17 = finger bases (MCP joints),
           8/12/16/20 = fingertips.
    2. Palm ROI = polygon through the wrist, thumb base and four finger bases.
    3. Canonical frame = an affine warp that puts the index base top-left,
       the little-finger base top-right and the wrist bottom-centre. Rotated
       or tilted photos, and left vs right hands, all end up in the same
       orientation, so feature_extraction.py can look for the heart line
       "near the top" and the life line "around the thumb" with one set of
       rules.
    4. Validation: the hand must be fully in frame, large enough, open and
       facing the camera. When MediaPipe finds no hand, a skin-segmentation
       outline (counting the gaps between fingers) is used only to give a
       more specific error message - never to analyse a palm, because without
       landmarks the palm's geometry would be guesswork.

Landmark reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

import config
from modules.preprocessing import resize_image, segment_hand

# MediaPipe landmark indices used below.
WRIST, THUMB_CMC, THUMB_MCP, THUMB_TIP = 0, 1, 2, 4
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
MIDDLE_TIP = 12
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (WRIST, THUMB_CMC, THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Bone connections for drawing the skeleton overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# Canonical palm frame (pixels). Index base top-left, little-finger base
# top-right, wrist bottom-centre; the thumb side is therefore always on the LEFT.
# The frame has a margin around the anchors because the palm extends beyond
# them - especially the thumb mound, where the life line runs.
CANON_W, CANON_H = 500, 520
CANON_ANCHORS = np.float32([
    [160, 60],    # index MCP (5)
    [420, 100],   # pinky MCP (17) - sits slightly lower than the index base
    [270, 470],   # wrist (0)
])

# Palm polygon is pushed outward from its centroid by this factor. WHY: the
# MCP landmarks sit on the knuckle joints, slightly inside the true palm edge,
# so without expansion the ends of the heart and life lines are cut off.
PALM_POLYGON_EXPAND = 1.15

# A palm landmark this close to the edge (fraction of width/height) means the
# palm is probably cut off by the frame.
EDGE_MARGIN = 0.01

# MediaPipe's graph is not safe for concurrent calls, and Streamlit serves
# each browser session from its own thread.
_MP_LOCK = threading.Lock()


@dataclass
class HandDetection:
    """Everything known about the hand in one image.

    WHY keep `method` and `warnings`: the UI and the LLM must know whether
    the palm was located precisely (MediaPipe) or roughly (contour
    fallback), so uncertainty is reported rather than hidden.
    """

    found: bool
    method: str                                  # "mediapipe" | "contour" | "none"
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    landmarks: np.ndarray | None = None          # (21, 2) pixel coordinates
    palm_polygon: np.ndarray | None = None       # (N, 2) int32, original image coords
    hand_contour: np.ndarray | None = None       # OpenCV contour of the whole hand
    to_canonical: np.ndarray | None = None       # 2x3 affine: image -> canonical
    from_canonical: np.ndarray | None = None     # 2x3 affine: canonical -> image
    measurements: dict[str, float] = field(default_factory=dict)
    mp_handedness: str | None = None             # raw MediaPipe label (see note below)
    mp_score: float | None = None
    appears_as: str | None = None                # "left"/"right" from thumb position


# ------------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_hands_model() -> Any:
    """Create the MediaPipe Hands model once and reuse it.

    WHY cached: loading the model takes about a second, and Streamlit reruns
    the whole script on every click. static_image_mode=True runs full
    detection on every image instead of tracking between video frames.
    """
    import mediapipe as mp  # imported lazily so a broken install gives a clean error

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=config.MP_MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
    )


def detect_landmarks(image_bgr: np.ndarray) -> tuple[np.ndarray, str, float] | None:
    """Run MediaPipe Hands and return (landmarks_px, handedness, score) or None.

    WHY convert to pixels: MediaPipe returns coordinates normalised to 0-1.
    Every later step (polygons, warps, drawing) works in pixels.
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with _MP_LOCK:
        results = _get_hands_model().process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark], dtype=np.float32)

    label, score = "unknown", 0.0
    if results.multi_handedness:
        cls = results.multi_handedness[0].classification[0]
        label, score = cls.label.lower(), float(cls.score)
    return pts, label, score


# ------------------------------------------------------------------
# Geometry from landmarks
# ------------------------------------------------------------------
def palm_polygon_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Build the palm ROI polygon from the wrist, thumb base and finger bases.

    WHY these points: together they outline the palm and exclude the
    fingers, which have their own creases that are not palm lines.
    The polygon is expanded slightly because the landmarks sit inside the
    real palm edge.
    """
    pts = landmarks[list(PALM_POINTS)]
    hull = cv2.convexHull(pts).reshape(-1, 2)
    centroid = hull.mean(axis=0)
    expanded = centroid + (hull - centroid) * PALM_POLYGON_EXPAND
    return expanded.astype(np.int32)


def canonical_transforms(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Affine transforms between the image and the canonical palm frame.

    WHY affine from three anchors: three point pairs define an affine map
    exactly (rotation, scale, shear, translation and, when needed, a mirror
    flip). Because the anchors are the index base, the little-finger base
    and the wrist, a left palm and a right palm are both mapped with the
    thumb on the left - the flip happens automatically.
    """
    src = np.float32([landmarks[INDEX_MCP], landmarks[PINKY_MCP], landmarks[WRIST]])
    forward = cv2.getAffineTransform(src, CANON_ANCHORS)
    inverse = cv2.invertAffineTransform(forward)
    return forward, inverse


def palm_measurements(landmarks: np.ndarray) -> dict[str, float]:
    """Measure the palm and fingers in pixels.

    WHY: palm shape in palmistry is judged by palm width vs length and finger
    length vs palm length. These distances are measured on the ORIGINAL
    landmarks, because the canonical warp deliberately distorts proportions.
    The index-to-pinky base distance underestimates full palm width (the
    knuckles sit inside the palm edges), so it is scaled by 1.25.
    """
    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(landmarks[a] - landmarks[b]))

    width = dist(INDEX_MCP, PINKY_MCP) * 1.25
    length = dist(WRIST, MIDDLE_MCP)
    finger = dist(MIDDLE_MCP, MIDDLE_TIP)
    return {
        "palm_width_px": round(width, 1),
        "palm_length_px": round(length, 1),
        "middle_finger_px": round(finger, 1),
        "length_to_width": round(length / width, 3) if width else 0.0,
        "finger_to_palm": round(finger / length, 3) if length else 0.0,
    }


def thumb_side(landmarks: np.ndarray) -> str:
    """Infer which hand this is from where the thumb is, assuming the palm faces the camera.

    WHY not trust MediaPipe's handedness label: MediaPipe assumes a mirrored
    selfie image, so on a normal back-camera photo its label is often
    reversed. Geometry is more dependable: with the fingers pointing "up"
    (wrist -> middle base), a RIGHT palm facing the camera has its thumb on
    the image's right, and a LEFT palm on the left. The sign of the 2-D
    cross product tells us which side the thumb is on, at any rotation.
    """
    up = landmarks[MIDDLE_MCP] - landmarks[WRIST]
    thumb = landmarks[THUMB_MCP] - landmarks[WRIST]
    cross = up[0] * thumb[1] - up[1] * thumb[0]
    # Image y points down, so a positive cross product means "to the right".
    return "right" if cross > 0 else "left"


def completeness_issues(landmarks: np.ndarray, shape: tuple[int, int]) -> tuple[list[str], list[str]]:
    """Check whether the palm or fingers run off the edge of the photo.

    WHY: MediaPipe estimates landmarks even when they fall outside the frame.
    A palm cut off by the frame means lines cannot be measured (error); a
    clipped fingertip only affects finger length (warning).
    Returns (errors, warnings).
    """
    h, w = shape
    mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h

    def outside(idx: int) -> bool:
        x, y = landmarks[idx]
        return not (mx <= x <= w - mx and my <= y <= h - my)

    errors, warnings = [], []
    if any(outside(i) for i in PALM_POINTS):
        errors.append(
            "Part of the palm is outside the photo. Move the camera back so your "
            "whole hand, including the wrist, fits inside the frame."
        )
    elif any(outside(i) for i in FINGERTIPS):
        warnings.append(
            "Some fingertips are cut off. Lines can still be read, but palm-shape "
            "estimates that use finger length will be less reliable."
        )
    return errors, warnings


FINGER_CHAINS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))   # (MCP, PIP, TIP)


def open_hand_problem(landmarks: np.ndarray) -> str | None:
    """Return an error if the hand is not open with the palm visible.

    WHY: palm lines can only be read on an open, flat hand. A fingertip that
    is not clearly farther from the wrist than its middle joint means that
    finger is curled; with two or more curled fingers (a fist, a pinch, a
    side view) the palm is hidden or folded.
    """
    wrist = landmarks[WRIST]
    curled = 0
    for mcp, pip, tip in FINGER_CHAINS:
        if np.linalg.norm(landmarks[tip] - wrist) < 1.05 * np.linalg.norm(landmarks[pip] - wrist):
            curled += 1
    if curled >= 2:
        return ("Your fingers look curled or folded. Open your hand fully with the palm flat "
                "and facing the camera, then take the photo again.")

    # A palm seen edge-on has its finger bases bunched together.
    palm_len = np.linalg.norm(landmarks[MIDDLE_MCP] - wrist)
    knuckles = np.linalg.norm(landmarks[INDEX_MCP] - landmarks[PINKY_MCP])
    if palm_len > 0 and knuckles / palm_len < 0.35:
        return ("The palm seems to be turned sideways. Hold your palm flat and facing the "
                "camera directly, then take the photo again.")
    return None


def hand_area_ratio(landmarks: np.ndarray, shape: tuple[int, int]) -> float:
    """Fraction of the image covered by the convex hull of all landmarks.

    WHY: a hand that fills only a tiny part of the photo has too few pixels
    across each palm line to detect it reliably.
    """
    hull = cv2.convexHull(landmarks.astype(np.float32))
    return float(cv2.contourArea(hull)) / float(shape[0] * shape[1])


# ------------------------------------------------------------------
# Diagnosis when MediaPipe finds no hand
# ------------------------------------------------------------------
# A finger gap is a convexity defect deeper than this fraction of the
# outline's bounding-box diagonal. An open hand has 3-4 such gaps; an oval,
# a wall or a face has none.
FINGER_GAP_DEPTH = 0.06
MIN_FINGER_GAPS = 3


def count_finger_gaps(contour: np.ndarray) -> int:
    """Count deep convexity defects (gaps between fingers) in an outline.

    WHY: the convex hull of an open hand spans the fingertips, so each gap
    between two fingers is a deep "bay" between hull and outline. Blobs such
    as ovals or skin-coloured objects have no deep bays.
    """
    if len(contour) < 5:
        return 0
    hull = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    x, y, w, h = cv2.boundingRect(contour)
    min_depth = FINGER_GAP_DEPTH * float(np.hypot(w, h))
    return int(np.sum(defects[:, 0, 3] / 256.0 > min_depth))


def diagnose_missing_hand(image_bgr: np.ndarray) -> HandDetection:
    """Explain why no usable hand was found, as specifically as possible.

    WHY no analysis here: without landmarks the finger bases, thumb and wrist
    cannot be located, so any palm-line analysis would rest on guesses. An
    earlier version analysed the largest skin-coloured blob, which wrongly
    accepted skin-coloured ovals and cut-off hands. The outline is now used
    only to choose the most helpful message.
    """
    h, w = image_bgr.shape[:2]
    mask = segment_hand(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return HandDetection(False, "none", message=_no_hand_message())

    contour = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(contour) / float(h * w)
    if ratio < config.MIN_HAND_AREA_RATIO or ratio > 0.90:
        return HandDetection(False, "none", message=_no_hand_message())

    x, y, bw, bh = cv2.boundingRect(contour)
    touches = sum([x <= 2, y <= 2, x + bw >= w - 2, y + bh >= h - 2])
    gaps = count_finger_gaps(contour)
    if gaps >= MIN_FINGER_GAPS and touches >= 2:
        message = ("This looks like part of a hand, but the hand is cut off by the edge of the "
                   "photo. Move the camera back so your whole hand and wrist are inside the frame.")
    elif gaps >= MIN_FINGER_GAPS:
        message = ("Something hand-shaped was found, but its finger and wrist positions could not "
                   "be identified. Retake the photo with your palm flat and facing the camera, "
                   "fingers straight, in even light without strong shadows.")
    else:
        message = _no_hand_message()
    return HandDetection(False, "none", message=message, hand_contour=contour)


def _no_hand_message() -> str:
    """Single source for the 'no hand' advice so the wording stays consistent."""
    return (
        "No hand was detected. Hold your open palm facing the camera, fingers "
        "together and pointing up, against a plain background, with the whole "
        "hand inside the frame."
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def detect_hand(image_bgr: np.ndarray, palm_side: str | None = None) -> HandDetection:
    """Detect the hand and build the palm ROI for one image.

    `image_bgr` should be the pipeline's resized working image, so the
    landmark pixels line up with every preprocessed stage.
    `palm_side` ("left"/"right") is the slot the user uploaded into; it is
    used to warn about swapped uploads.
    """
    h, w = image_bgr.shape[:2]
    try:
        detected = detect_landmarks(image_bgr)
    except Exception:  # noqa: BLE001 - any MediaPipe failure falls back cleanly
        detected = None

    if detected is None:
        return diagnose_missing_hand(image_bgr)

    landmarks, mp_label, mp_score = detected
    errors, warnings = completeness_issues(landmarks, (h, w))
    if errors:
        return HandDetection(False, "mediapipe", message=errors[0], landmarks=landmarks)

    if (problem := open_hand_problem(landmarks)):
        return HandDetection(False, "mediapipe", message=problem, landmarks=landmarks)

    area = hand_area_ratio(landmarks, (h, w))
    if area < config.MIN_HAND_AREA_RATIO:
        return HandDetection(
            False, "mediapipe", landmarks=landmarks,
            message=(
                f"The hand is too small in the photo (covers {area:.0%} of the image). "
                "Move the camera closer so your hand fills most of the frame."
            ),
        )

    appears_as = thumb_side(landmarks)
    if palm_side and appears_as != palm_side:
        warnings.append(
            f"This was uploaded as the {palm_side.upper()} palm, but the thumb position "
            f"looks like a {appears_as.upper()} palm. Check that the photos are not "
            "swapped and that the palm (not the back of the hand) faces the camera."
        )

    forward, inverse = canonical_transforms(landmarks)
    hull = cv2.convexHull(landmarks.astype(np.int32))

    return HandDetection(
        found=True,
        method="mediapipe",
        message=f"Hand detected by MediaPipe (confidence {mp_score:.0%}).",
        warnings=warnings,
        landmarks=landmarks,
        palm_polygon=palm_polygon_from_landmarks(landmarks),
        hand_contour=hull,
        to_canonical=forward,
        from_canonical=inverse,
        measurements=palm_measurements(landmarks),
        mp_handedness=mp_label,
        mp_score=mp_score,
        appears_as=appears_as,
    )


def hand_check(image_bgr: np.ndarray) -> tuple[bool, str]:
    """Validation hook: is there a usable hand in this photo?

    WHY a separate wrapper: validation.py only needs yes/no plus a message,
    and works on the full-size upload, so this resizes first for speed.
    """
    detection = detect_hand(resize_image(image_bgr))
    return detection.found, detection.message


def palm_mask(shape: tuple[int, int], detection: HandDetection) -> np.ndarray:
    """Binary mask (255 inside the palm ROI) in image coordinates."""
    mask = np.zeros(shape, dtype=np.uint8)
    if detection.palm_polygon is not None:
        cv2.fillPoly(mask, [detection.palm_polygon], 255)
    return mask


def warp_to_canonical(image: np.ndarray, detection: HandDetection, binary: bool = False,
                      scale: float = 1.0) -> np.ndarray:
    """Warp any pipeline image into the canonical palm frame.

    `scale` is (working image size / this image's size). Landmarks are found
    on the 800 px working image, but the palm is warped from the ORIGINAL
    photo (scale < 1) so no detail is lost - small palms in the working
    image are only ~150 px wide, too few pixels to separate nearby creases.

    WHY nearest-neighbour for binary images: interpolating a 0/255 mask
    would create grey values that are neither line nor background.
    """
    if detection.to_canonical is None:
        raise ValueError("No palm transform available - hand was not detected.")
    matrix = detection.to_canonical.copy()
    matrix[:, :2] *= scale
    interp = cv2.INTER_NEAREST if binary else cv2.INTER_CUBIC
    return cv2.warpAffine(image, matrix, (CANON_W, CANON_H), flags=interp,
                          borderMode=cv2.BORDER_REPLICATE if not binary else cv2.BORDER_CONSTANT)


def palm_polygon_canonical(detection: HandDetection) -> np.ndarray:
    """The palm ROI polygon expressed in canonical-frame pixels."""
    if detection.palm_polygon is None or detection.to_canonical is None:
        return np.empty((0, 2), np.int32)
    pts = detection.palm_polygon.astype(np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, detection.to_canonical).reshape(-1, 2).astype(np.int32)


def canonical_to_image(points: np.ndarray, detection: HandDetection) -> np.ndarray:
    """Map (N, 2) points from the canonical frame back to image pixels.

    WHY: lines are found in the canonical frame but must be drawn on the
    user's own photo.
    """
    if detection.from_canonical is None or len(points) == 0:
        return points.astype(np.int32)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, detection.from_canonical).reshape(-1, 2).astype(np.int32)


def draw_hand_overlay(image_bgr: np.ndarray, detection: HandDetection) -> np.ndarray:
    """Draw landmarks, skeleton and palm ROI for the 'Hand detection' display.

    WHY: makes the detection stage visible and explainable - the examiner
    can see exactly which region the line search was restricted to.
    """
    out = image_bgr.copy()
    outline = config.LINE_COLORS["outline"]

    if detection.hand_contour is not None and detection.method == "contour":
        cv2.drawContours(out, [detection.hand_contour], -1, (0, 200, 255), 2)

    if detection.landmarks is not None:
        pts = detection.landmarks.astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            colour = (0, 0, 255) if i in PALM_POINTS else (0, 255, 0)
            cv2.circle(out, tuple(p), 4, colour, -1, cv2.LINE_AA)

    if detection.palm_polygon is not None:
        cv2.polylines(out, [detection.palm_polygon], True, outline, 2, cv2.LINE_AA)
        x, y = detection.palm_polygon.min(axis=0)
        cv2.putText(out, "Palm ROI", (int(x), max(15, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, outline, 2, cv2.LINE_AA)
    return out
