"""
preprocessing.py  (Task 02)
---------------------------
Digital image processing pipeline that turns a raw palm photo into a clean
binary "line map" for feature extraction.

Pipeline (each step is a separate function so each can be justified):

    1. resize_image          fixed working scale
    2. to_grayscale          colour -> intensity
       normalize_intensity   stretch to the full 0-255 range
    3. gaussian_denoise      remove sensor noise
       bilateral_denoise     smooth skin texture, keep crease edges
    4. equalize_histogram    global contrast (shown for comparison)
       apply_clahe           local contrast (used downstream)
    5. sobel_edges           gradient magnitude
       canny_edges           thin, connected edges
    6. segment_hand          skin segmentation -> hand mask
       adaptive_threshold    dark creases -> white pixels
       blackhat_enhance      morphological isolation of thin dark lines
    7. morph_open            remove specks
       morph_close           join broken line pieces

run_preprocessing() executes everything and returns every intermediate
image, so the UI can show original-vs-processed for each technique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import config


# ------------------------------------------------------------------
# 1. Resize
# ------------------------------------------------------------------
def resize_image(image: np.ndarray, long_side: int = config.WORKING_LONG_SIDE_PX) -> np.ndarray:
    """Scale the image so its longer side equals `long_side`, keeping aspect ratio.

    WHY: kernel sizes (e.g. a 7x7 closing) and pixel-length thresholds only
    mean the same thing if every photo is at the same scale. 800 px keeps
    palm creases several pixels wide while staying fast to process.
    INTER_AREA when shrinking avoids aliasing; INTER_CUBIC when enlarging
    gives smoother results than nearest-neighbour.
    """
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image.copy()
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interp)


# ------------------------------------------------------------------
# 2. Grayscale + normalisation
# ------------------------------------------------------------------
def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR colour to a single intensity channel.

    WHY: palm lines are darker folds of skin - an intensity feature, not a
    colour feature. One channel also makes every later filter 3x cheaper.
    OpenCV weights the channels (0.299 R + 0.587 G + 0.114 B) to match
    human brightness perception.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def normalize_intensity(gray: np.ndarray) -> np.ndarray:
    """Linearly stretch intensities so the darkest pixel is 0 and brightest 255.

    WHY: photos taken in different light occupy different parts of the
    0-255 range. Min-max normalisation puts them on a common scale before
    filtering. It is linear, so it changes no relative brightness.
    """
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


# ------------------------------------------------------------------
# 3. Noise reduction
# ------------------------------------------------------------------
def gaussian_denoise(gray: np.ndarray) -> np.ndarray:
    """Smooth with a Gaussian kernel.

    WHY: camera sensor noise creates random bright/dark pixels that edge
    detectors mistake for lines. A Gaussian weights nearby pixels most, so
    it suppresses noise with less blurring than a plain box average.
    Limitation: it also softens real edges, which is why we compare it with
    the bilateral filter.
    """
    return cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, config.GAUSSIAN_SIGMA)


def bilateral_denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing with a bilateral filter.

    WHY: each neighbour is weighted by distance AND by intensity similarity.
    Across flat skin the neighbours are similar, so texture is smoothed;
    across a crease the intensity jumps, so those neighbours get little
    weight and the crease edge survives. This is the denoised image we
    use downstream.
    """
    return cv2.bilateralFilter(
        gray,
        config.BILATERAL_DIAMETER,
        config.BILATERAL_SIGMA_COLOR,
        config.BILATERAL_SIGMA_SPACE,
    )


# ------------------------------------------------------------------
# 4. Contrast enhancement
# ------------------------------------------------------------------
def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalisation.

    WHY it is shown: it is the classic contrast technique and a useful
    baseline. It spreads the intensity histogram across 0-255 using one
    mapping for the whole image, so it often over-brightens the background
    and amplifies noise - which is why CLAHE is used for the real pipeline.
    """
    return cv2.equalizeHist(gray)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    WHY: palm lines are low-contrast and lighting across a hand is uneven.
    CLAHE equalises each 8x8 tile separately (adaptive) so a shadowed part
    of the palm is enhanced as much as a bright part, and it clips the
    histogram (contrast limited) so noise in flat regions is not blown up.
    """
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    return clahe.apply(gray)


# ------------------------------------------------------------------
# 5. Edge detection
# ------------------------------------------------------------------
def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude from horizontal and vertical Sobel filters.

    WHY: Sobel estimates the first derivative in x and y. The magnitude
    sqrt(Gx^2 + Gy^2) is high wherever brightness changes quickly, i.e. on
    both sides of a crease. It keeps edge STRENGTH (useful as a proxy for
    line depth), but edges come out thick.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=config.SOBEL_KERNEL)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=config.SOBEL_KERNEL)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edge detection.

    WHY: Canny improves on raw gradients with non-maximum suppression (edges
    1 px thin) and hysteresis thresholding (a weak edge is kept only if it
    connects to a strong one). That keeps long continuous lines and drops
    isolated texture, which suits palm lines well.
    """
    return cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)


# ------------------------------------------------------------------
# 6. Thresholding and segmentation
# ------------------------------------------------------------------
def segment_hand(image_bgr: np.ndarray) -> np.ndarray:
    """Segment the hand from the background using skin colour in YCrCb.

    WHY YCrCb: it separates brightness (Y) from colour (Cr, Cb). Skin of
    every tone clusters in a narrow Cr/Cb range, so thresholding only the
    colour channels works across skin tones and lighting levels. If skin
    thresholding finds too little (unusual lighting), Otsu's threshold on
    the grayscale image is used as a fallback.

    Returns a binary mask (255 = hand) containing only the largest region,
    with holes filled.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    if cv2.countNonZero(mask) < 0.05 * mask.size:
        gray = cv2.GaussianBlur(to_grayscale(image_bgr), (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu picks a side arbitrarily; the hand is usually the central object.
        h, w = mask.shape
        if mask[h // 3: 2 * h // 3, w // 3: 2 * w // 3].mean() < 127:
            mask = cv2.bitwise_not(mask)

    # Close small gaps (e.g. specular highlights), then keep the largest blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Local (adaptive) thresholding - dark creases become white pixels.

    WHY adaptive: a single global threshold fails because one side of the
    palm is often in shadow. Each pixel is compared with the Gaussian-
    weighted mean of its 21x21 neighbourhood minus C, so a crease is found
    wherever it is darker than its OWN surroundings. THRESH_BINARY_INV makes
    the lines white, which is the convention for morphology and contours.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.ADAPTIVE_BLOCK_SIZE,
        config.ADAPTIVE_C,
    )


def blackhat_enhance(gray: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Morphological black-hat: closing(image) - image.

    WHY: closing fills in dark structures NARROWER than the kernel. Subtracting
    the original therefore leaves only those thin dark structures - exactly
    what palm creases are - while broad shadows (wider than the kernel)
    cancel out. This is a classic way to isolate thin dark lines on an
    uneven background.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarise with Otsu's automatically chosen global threshold.

    WHY on the black-hat image: after black-hat the background is near zero
    and creases are bright, so the histogram is bimodal - the case where
    Otsu's method (which minimises within-class variance) works best.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ------------------------------------------------------------------
# 7. Morphological clean-up
# ------------------------------------------------------------------
def morph_open(binary: np.ndarray) -> np.ndarray:
    """Opening = erosion then dilation.

    WHY: removes white specks smaller than the kernel (pores, skin texture,
    noise) while leaving larger structures such as line segments roughly
    unchanged in size.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_OPEN_KERNEL)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morph_close(binary: np.ndarray) -> np.ndarray:
    """Closing = dilation then erosion.

    WHY: palm lines often appear as a chain of short dashes after
    thresholding. Closing bridges gaps smaller than the kernel so each major
    line becomes one connected component, which feature extraction needs in
    order to measure its length and shape.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_CLOSE_KERNEL)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def remove_small_components(binary: np.ndarray, min_area: int = 40) -> np.ndarray:
    """Delete connected components with fewer than `min_area` pixels.

    WHY: opening removes tiny specks, but blobs a little larger than the
    kernel survive. Filtering by component area removes them without
    shrinking real lines.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def erode_mask(mask: np.ndarray, pixels: int = 12) -> np.ndarray:
    """Shrink the hand mask inward by a few pixels.

    WHY: the boundary between hand and background is a strong edge, and
    without this step it would be detected as a giant "palm line". Eroding
    the mask keeps the analysis strictly inside the hand.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.erode(mask, kernel)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------
@dataclass
class Stage:
    """One displayable pipeline step: the image plus a one-line explanation."""

    title: str
    image: np.ndarray
    explanation: str


@dataclass
class PreprocessResult:
    """Everything produced by the pipeline.

    `stages` is ordered for display. `enhanced`, `line_map` and `hand_mask`
    are the outputs feature_extraction.py actually consumes.
    """

    resized_bgr: np.ndarray
    source_bgr: np.ndarray      # original full-resolution photo
    scale: float                # resized size / original size
    enhanced: np.ndarray        # CLAHE grayscale
    line_map: np.ndarray        # cleaned binary crease map inside the hand
    hand_mask: np.ndarray       # binary hand segmentation
    stages: list[Stage] = field(default_factory=list)


def run_preprocessing(image_bgr: np.ndarray) -> PreprocessResult:
    """Run the full Task 02 pipeline and keep every intermediate image.

    WHY the order matters:
        resize -> grayscale -> normalise gives a consistent input;
        denoising comes BEFORE contrast enhancement, otherwise CLAHE would
        amplify the noise; edges and thresholds run on the enhanced image;
        morphology runs last because it works on binary images.
    """
    resized = resize_image(image_bgr)
    gray = to_grayscale(resized)
    normalized = normalize_intensity(gray)

    gaussian = gaussian_denoise(normalized)
    bilateral = bilateral_denoise(normalized)

    hist_eq = equalize_histogram(bilateral)
    clahe = apply_clahe(bilateral)

    sobel = sobel_edges(clahe)
    canny = canny_edges(gaussian_denoise(clahe))

    hand_mask = segment_hand(resized)
    inner_mask = erode_mask(hand_mask)
    adaptive = cv2.bitwise_and(adaptive_threshold(clahe), inner_mask)

    blackhat = cv2.bitwise_and(blackhat_enhance(clahe), inner_mask)
    line_binary = otsu_threshold(blackhat)

    opened = morph_open(line_binary)
    closed = morph_close(opened)
    line_map = cv2.bitwise_and(remove_small_components(closed), inner_mask)

    stages = [
        Stage("1. Resized (original colour)", resized,
              f"Scaled to {config.WORKING_LONG_SIDE_PX} px on the long side so all thresholds use the same scale."),
        Stage("2a. Grayscale", gray,
              "Colour removed - palm lines are differences in brightness, not colour."),
        Stage("2b. Normalised", normalized,
              "Intensities stretched to the full 0-255 range."),
        Stage("3a. Gaussian blur", gaussian,
              "Removes sensor noise, but also softens edges."),
        Stage("3b. Bilateral filter", bilateral,
              "Smooths skin texture while keeping crease edges sharp (used downstream)."),
        Stage("4a. Histogram equalisation", hist_eq,
              "Global contrast stretch - note the over-brightened areas."),
        Stage("4b. CLAHE", clahe,
              "Local, clip-limited contrast: lines stand out evenly across the palm."),
        Stage("5a. Sobel magnitude", sobel,
              "Gradient strength in x and y - thick edges, keeps edge strength."),
        Stage("5b. Canny edges", canny,
              "Thin, connected edges via non-max suppression and hysteresis."),
        Stage("6a. Hand segmentation", hand_mask,
              "Skin-colour segmentation in YCrCb; largest region kept."),
        Stage("6b. Adaptive threshold", adaptive,
              "Pixels darker than their local neighbourhood -> white."),
        Stage("6c. Black-hat + Otsu", line_binary,
              "Black-hat isolates thin dark creases; Otsu binarises them."),
        Stage("7a. Morphological opening", opened,
              "Removes small specks of skin texture."),
        Stage("7b. Closing + clean-up (final line map)", line_map,
              "Bridges gaps so each line is one connected structure."),
    ]

    return PreprocessResult(
        resized_bgr=resized,
        source_bgr=image_bgr,
        scale=resized.shape[1] / image_bgr.shape[1],
        enhanced=clahe,
        line_map=line_map,
        hand_mask=hand_mask,
        stages=stages,
    )


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or BGR image to RGB for Streamlit.

    WHY: OpenCV stores colour as BGR, but Streamlit and Pillow expect RGB;
    without this, skin would appear blue.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
