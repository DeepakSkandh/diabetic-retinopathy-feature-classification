"""
Red-lesion candidate features.

Detects small, compact regions that are darker than their local background
in the green channel (the appearance of microaneurysm- and hemorrhage-like
spots), after shade correction.

IMPORTANT: these are CANDIDATE regions, not confirmed lesions. They can also
include vessel fragments, pigmentation, noise and other dark structures.
Their correspondence to real lesions has not been validated (the IDRiD
"A. Segmentation" masks can be used for that).

Resolution
----------
Microaneurysms are tiny. At 512 x 512 (the main pipeline's size) they
shrink to about 1-3 pixels, and the non-uniform resize distorts their
shape. This module therefore works on its own copy of the image, resized
to WORK_WIDTH pixels wide with the aspect ratio preserved.
All size parameters below are in pixels at WORK_WIDTH.
"""

from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 1024

# Starting values: tune them by inspecting overlays (save_red_lesion_overlay).
PARAMS = {
    "mask_threshold": 10,          # same threshold as the main retinal mask
    "border_erosion": 15,          # px removed from the FOV rim (vignetting)
    "background_kernel": 61,       # median-blur size; must exceed lesion size
    "k_sigma": 3.0,                # darkness threshold, in robust std units
    "min_area": 3,                 # smaller blobs are treated as noise
    "max_area": 3000,              # larger blobs are mostly vessel trees
    "small_max_area": 50,          # <= this: microaneurysm-sized candidate
    "max_elongation": 3.0,         # longer/thinner shapes are vessel-like
    "min_solidity": 0.5,           # very irregular shapes are rejected
    "shape_check_min_area": 10,    # shape measures unreliable below this
    "quadrant_count_threshold": 5, # for red_quadrants_over_thresh
}

RED_FEATURE_NAMES = [
    "red_count",
    "red_small_count",
    "red_large_count",
    "red_total_area",
    "red_area_ratio",
    "red_mean_area",
    "red_mean_contrast",
    "red_quadrant_min_count",
    "red_quadrants_over_thresh",
]


def _params(params):
    return {**PARAMS, **(params or {})}


def load_retina(image_path, width=WORK_WIDTH, threshold=PARAMS["mask_threshold"]):
    """Load an image as RGB at `width` px (aspect preserved) plus a retinal mask."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    height = int(round(h * width / w))
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest component (the field of view)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return rgb, mask


def shade_correct_green(rgb, mask, background_kernel=PARAMS["background_kernel"]):
    """
    Shade correction: estimate the slowly varying background of the green
    channel with a large median blur and subtract the image from it.

    Returns a float "darkness" map: positive where a pixel is darker than its
    local background, ~0 on uniform background, independent of overall
    image brightness.
    """
    green = rgb[:, :, 1].copy()
    inside = mask > 0

    # Fill outside the FOV with the median retinal value so the blur does not
    # drag the background estimate down near the rim.
    green[~inside] = np.uint8(np.median(green[inside]))

    background = cv2.medianBlur(green, background_kernel)
    darkness = background.astype(np.float32) - green.astype(np.float32)
    darkness[~inside] = 0
    return darkness


def detect_red_candidates(rgb, mask, params=None, return_debug=False):
    """
    Returns
    -------
    candidates : list of dict  (area, cx, cy, contrast)
    candidate_mask : uint8 mask of kept candidates
    inner_mask : bool mask of the analysed retinal area (rim removed)
    debug : dict (only if return_debug=True) with the background estimate,
        darkness map, robust z-map and all dark pixels before shape filtering
    """
    p = _params(params)

    darkness = shade_correct_green(rgb, mask, p["background_kernel"])

    size = 2 * p["border_erosion"] + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    inner = cv2.erode(mask, erode_kernel) > 0

    # Robust per-image normalisation (median / MAD), so the threshold adapts
    # to each image's noise level instead of using a fixed intensity value.
    values = darkness[inner]
    med = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - med)) + 1e-6
    z = (darkness - med) / sigma

    dark_mask = ((z > p["k_sigma"]) & inner).astype(np.uint8) * 255

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    candidates = []
    candidate_mask = np.zeros_like(dark_mask)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)

        if area >= p["shape_check_min_area"]:
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)

            # Orientation-independent elongation (rotated bounding box)
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            elongation = max(rw, rh) / max(min(rw, rh), 1.0)

            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = contour_area / hull_area if hull_area > 0 else 1.0

            if elongation > p["max_elongation"] or solidity < p["min_solidity"]:
                continue

        contrast = float(z[y:y + h, x:x + w][comp > 0].mean())
        cx, cy = centroids[i]
        candidates.append({"area": area, "cx": float(cx), "cy": float(cy),
                           "contrast": contrast})
        candidate_mask[y:y + h, x:x + w][comp > 0] = 255

    if return_debug:
        background = np.where(mask > 0, darkness + rgb[:, :, 1].astype(np.float32), 0)
        debug = {
            "background": background,
            "darkness": darkness,
            "z": z,
            "all_dark": dark_mask,
        }
        return candidates, candidate_mask, inner, debug

    return candidates, candidate_mask, inner


def extract_red_lesion_features(rgb, mask, params=None):
    """Numerical red-lesion candidate features for one image."""
    p = _params(params)
    candidates, _, inner = detect_red_candidates(rgb, mask, p)

    if not candidates:
        return {name: 0 for name in RED_FEATURE_NAMES}

    areas = np.array([c["area"] for c in candidates])
    contrasts = np.array([c["contrast"] for c in candidates])

    # Image quadrants around the centre of the retinal mask
    m = cv2.moments(mask, binaryImage=True)
    mx, my = m["m10"] / m["m00"], m["m01"] / m["m00"]
    quadrant = np.array([(c["cx"] >= mx) + 2 * (c["cy"] >= my) for c in candidates])
    q_counts = np.bincount(quadrant, minlength=4)

    retinal_area = int(inner.sum())

    return {
        "red_count": len(candidates),
        "red_small_count": int((areas <= p["small_max_area"]).sum()),
        "red_large_count": int((areas > p["small_max_area"]).sum()),
        "red_total_area": int(areas.sum()),
        "red_area_ratio": areas.sum() / retinal_area if retinal_area > 0 else 0,
        "red_mean_area": float(areas.mean()),
        "red_mean_contrast": float(contrasts.mean()),
        "red_quadrant_min_count": int(q_counts.min()),
        "red_quadrants_over_thresh": int((q_counts >= p["quadrant_count_threshold"]).sum()),
    }


def save_red_lesion_overlay(image_path, out_path, params=None):
    """
    Save the image with candidates circled, for visual checking.
    Yellow = small (microaneurysm-sized) candidate, magenta = larger candidate.
    Returns the number of candidates.
    """
    p = _params(params)
    rgb, mask = load_retina(image_path, threshold=p["mask_threshold"])
    candidates, _, _ = detect_red_candidates(rgb, mask, p)

    overlay = rgb.copy()
    for c in candidates:
        radius = int(np.sqrt(c["area"] / np.pi)) + 5
        color = (255, 255, 0) if c["area"] <= p["small_max_area"] else (255, 0, 255)
        cv2.circle(overlay, (int(c["cx"]), int(c["cy"])), radius, color, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return len(candidates)