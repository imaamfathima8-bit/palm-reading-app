"""
validation.py  (Task 01)
------------------------
Checks every uploaded palm photo BEFORE any processing happens.

WHY validate first:
    Classical computer vision is sensitive to input quality. A blurry, dark
    or tiny photo produces broken edges and false "lines", which would then
    be passed to the LLM as if they were real features. Rejecting bad input
    early - with a message that tells the user how to fix it - protects the
    whole pipeline and keeps the reading honest.

Checks performed (in order):
    1. File extension        - only .jpg / .jpeg / .png
    2. File size             - not empty, not oversized
    3. Decoding              - the bytes really are a valid JPEG/PNG
    4. Resolution            - enough pixels to resolve palm creases
    5. Blur                  - variance of the Laplacian
    6. Lighting              - mean brightness + histogram spread (std-dev)
    7. Hand present          - optional callback into hand_detection.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

# A hand check takes a BGR image and returns (hand_found, message).
# It is passed in rather than imported so validation.py has no dependency on
# MediaPipe and can be unit-tested on its own.
HandCheck = Callable[[np.ndarray], tuple[bool, str]]


@dataclass
class ValidationResult:
    """Outcome of validating one uploaded image.

    WHY a dataclass: the UI needs the pass/fail flag, every error (not just
    the first), the measured numbers for display, and the decoded image so
    it is not decoded twice.
    """

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    image_bgr: np.ndarray | None = None

    def fail(self, message: str) -> None:
        """Record an error and mark the result as failed."""
        self.ok = False
        self.errors.append(message)


# ------------------------------------------------------------------
# 1-3. File-level checks
# ------------------------------------------------------------------
def check_extension(filename: str) -> str | None:
    """Return an error if the file extension is not JPG/JPEG/PNG.

    WHY: a fast first filter with a clear message. It is not trusted on its
    own - decode_image() also checks the real format from the file bytes.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(e.upper() for e in config.ALLOWED_EXTENSIONS)
        return (
            f"'{filename}' is not a supported file type. "
            f"Please upload a {allowed} photo."
        )
    return None


def check_file_size(data: bytes) -> str | None:
    """Return an error if the file is empty or larger than the size limit.

    WHY: an empty upload means a failed transfer; a huge one slows the app
    and is never needed, because images are resized to 800 px anyway.
    """
    if not data:
        return "The file is empty. Please choose the photo again."
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return (
            f"The file is {size_mb:.1f} MB, which is over the "
            f"{config.MAX_FILE_SIZE_MB} MB limit. Please upload a smaller photo."
        )
    return None


def decode_image(data: bytes) -> tuple[np.ndarray | None, str | None]:
    """Decode uploaded bytes into an OpenCV BGR image.

    WHY Pillow first:
        - Image.verify() detects truncated or corrupted files.
        - img.format reports the REAL format, so a .txt file renamed to
          .jpg is caught even though its extension looks valid.
        - exif_transpose() applies the phone's rotation tag. Without it,
          portrait phone photos arrive sideways and the finger/wrist
          geometry used later would be wrong.

    Returns (image, None) on success or (None, error_message) on failure.
    """
    corrupted_msg = (
        "This file could not be read as an image - it may be corrupted or "
        "not really a JPG/PNG. Please export or take the photo again."
    )
    try:
        # verify() leaves the object unusable, so we open the bytes twice.
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
            real_format = probe.format

        if real_format not in config.ALLOWED_PIL_FORMATS:
            return None, (
                f"The file contents are {real_format}, not JPG or PNG. "
                "Please convert the photo to JPG or PNG and upload it again."
            )

        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            # Drops alpha and palette modes, which OpenCV cannot use directly.
            rgb = np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None, corrupted_msg

    # OpenCV expects BGR channel order.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None


# ------------------------------------------------------------------
# 4-6. Image-quality checks
# ------------------------------------------------------------------
def _to_analysis_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Resize to the working size and convert to grayscale.

    WHY: blur and lighting scores depend on image scale. Measuring on the
    same 800 px working size used by the pipeline makes the thresholds in
    config.py apply equally to a 12 MP phone photo and a small webcam shot.
    """
    h, w = image_bgr.shape[:2]
    scale = config.WORKING_LONG_SIDE_PX / max(h, w)
    if scale < 1.0:
        # INTER_AREA averages pixels when shrinking, avoiding aliasing.
        image_bgr = cv2.resize(
            image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def check_resolution(image_bgr: np.ndarray) -> tuple[str | None, int]:
    """Check that the shorter side is large enough.

    WHY the shorter side: a 2000x300 strip has plenty of pixels but cannot
    show a whole palm. Returns (error_or_None, shorter_side_px).
    """
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side < config.MIN_IMAGE_SIDE_PX:
        return (
            f"The image is too small ({w}x{h} px). Palm lines need at least "
            f"{config.MIN_IMAGE_SIDE_PX} px on the shorter side. Use your "
            "phone's main camera at full resolution and don't crop too tightly.",
            short_side,
        )
    return None, short_side


def compute_blur_score(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian - higher means sharper.

    WHY: the Laplacian is a second-derivative filter that responds strongly
    at sharp edges. A focused image has many strong responses, so their
    variance is high; a blurred image has weak, uniform responses and a low
    variance. It needs no reference image and is cheap to compute.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_blur(gray: np.ndarray) -> tuple[str | None, float]:
    """Reject the image if the blur score is below the threshold."""
    score = compute_blur_score(gray)
    if score < config.BLUR_VARIANCE_THRESHOLD:
        return (
            f"The photo looks blurry (sharpness score {score:.0f}, needs "
            f"{config.BLUR_VARIANCE_THRESHOLD:.0f}+). Hold the phone steady, "
            "tap the screen on your palm to focus, and take it again.",
            score,
        )
    return None, score


# Below this fraction of MIN_CONTRAST_STD the image is rejected; between it
# and MIN_CONTRAST_STD it only gets a warning. WHY: a close-up palm photo is
# mostly skin and naturally has a narrow histogram, and CLAHE in the
# preprocessing stage can recover moderate low contrast. Only truly flat
# images are hopeless.
HARD_CONTRAST_FRACTION: float = 0.6


def check_lighting(gray: np.ndarray) -> tuple[list[str], list[str], float, float]:
    """Check exposure (mean brightness) and contrast (histogram spread).

    WHY two measures:
        - Mean brightness catches photos that are too dark (lines hidden in
          shadow) or overexposed (lines washed out by glare).
        - Standard deviation measures how spread out the histogram is. A
          photo can have a normal mean yet be flat and grey, with every
          pixel at a similar value; then the lines have no contrast to detect.

    Returns (errors, warnings, mean_brightness, contrast_std).
    """
    mean = float(gray.mean())
    std = float(gray.std())
    errors: list[str] = []
    warnings: list[str] = []

    if mean < config.MIN_MEAN_BRIGHTNESS:
        errors.append(
            f"The photo is too dark (brightness {mean:.0f}/255). Move to a "
            "brighter spot - daylight near a window works well - and avoid "
            "shading your palm with the phone."
        )
    elif mean > config.MAX_MEAN_BRIGHTNESS:
        errors.append(
            f"The photo is overexposed (brightness {mean:.0f}/255). Avoid "
            "direct sunlight or flash, which wash out the palm lines."
        )

    hard_limit = config.MIN_CONTRAST_STD * HARD_CONTRAST_FRACTION
    if std < hard_limit:
        errors.append(
            f"The photo has very low contrast (spread {std:.0f}, needs "
            f"{hard_limit:.0f}+). Use a plain background that contrasts with "
            "your skin, and even lighting."
        )
    elif std < config.MIN_CONTRAST_STD:
        warnings.append(
            f"Contrast is on the low side (spread {std:.0f}). The app will "
            "enhance it, but faint lines may not be detected."
        )
    return errors, warnings, mean, std


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------
def validate_upload(
    filename: str,
    data: bytes,
    hand_check: HandCheck | None = None,
) -> ValidationResult:
    """Run every check on one uploaded file.

    WHY collect all errors instead of stopping at the first:
        If a photo is both dark and blurry, telling the user only about the
        darkness means they retake it, hit the blur error, and retake again.
        Reporting everything at once saves a round trip.

    File-level failures (wrong type, corrupted) DO stop early, because no
    image exists to run the quality checks on.
    """
    result = ValidationResult()

    for file_error in (check_extension(filename), check_file_size(data)):
        if file_error:
            result.fail(file_error)
    if not result.ok:
        return result

    image_bgr, decode_error = decode_image(data)
    if decode_error or image_bgr is None:
        result.fail(decode_error or "The image could not be decoded.")
        return result

    result.image_bgr = image_bgr
    h, w = image_bgr.shape[:2]
    result.metrics["width_px"] = w
    result.metrics["height_px"] = h

    res_error, short_side = check_resolution(image_bgr)
    result.metrics["shorter_side_px"] = short_side
    if res_error:
        result.fail(res_error)

    gray = _to_analysis_gray(image_bgr)

    blur_error, blur_score = check_blur(gray)
    result.metrics["sharpness"] = round(blur_score, 1)
    if blur_error:
        result.fail(blur_error)

    light_errors, light_warnings, mean, std = check_lighting(gray)
    result.metrics["brightness"] = round(mean, 1)
    result.metrics["contrast"] = round(std, 1)
    for err in light_errors:
        result.fail(err)
    result.warnings.extend(light_warnings)

    # The hand check is the most expensive step and is meaningless on an
    # unusable image, so it runs only if every quality check has passed.
    if hand_check is not None and result.ok:
        try:
            found, message = hand_check(image_bgr)
        except Exception:  # noqa: BLE001 - surface any detector failure cleanly
            found, message = False, (
                "Hand detection failed unexpectedly. Please try a different photo."
            )
        result.metrics["hand_detected"] = "yes" if found else "no"
        if not found:
            result.fail(message)

    return result
