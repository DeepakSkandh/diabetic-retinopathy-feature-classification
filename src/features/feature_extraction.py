import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_color_features(masked_rgb, retinal_mask):
    """
    Extract color/intensity features from the retinal region.
    """

    pixels = masked_rgb[retinal_mask > 0]

    features = {
        "mean_r": np.mean(pixels[:, 0]),
        "mean_g": np.mean(pixels[:, 1]),
        "mean_b": np.mean(pixels[:, 2]),

        "std_r": np.std(pixels[:, 0]),
        "std_g": np.std(pixels[:, 1]),
        "std_b": np.std(pixels[:, 2]),

        "median_r": np.median(pixels[:, 0]),
        "median_g": np.median(pixels[:, 1]),
        "median_b": np.median(pixels[:, 2]),
    }

    return features


def extract_texture_features(green_enhanced):
    """
    Extract GLCM texture features from the
    CLAHE-enhanced green channel.
    """

    # Quantize 0-255 image into 8 gray levels
    green_quantized = (green_enhanced / 32).astype(np.uint8)

    # Create GLCM
    glcm = graycomatrix(
        green_quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    features = {
        "glcm_contrast": graycoprops(
            glcm, "contrast"
        )[0, 0],

        "glcm_dissimilarity": graycoprops(
            glcm, "dissimilarity"
        )[0, 0],

        "glcm_homogeneity": graycoprops(
            glcm, "homogeneity"
        )[0, 0],

        "glcm_energy": graycoprops(
            glcm, "energy"
        )[0, 0],
    }

    return features


def extract_morphology_features(candidate_mask, retinal_mask):
    """
    Extract morphological features from candidate abnormal regions.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask,
        connectivity=8
    )

    retinal_area = np.sum(retinal_mask > 0)

    areas = []
    aspect_ratios = []
    circularities = []
    solidities = []

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        # Ignore very small regions
        if area < 5:
            continue

        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        areas.append(area)

        # Aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        aspect_ratios.append(aspect_ratio)

        # Create mask for this component
        component = (labels == label).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)

        # Geometric area of the contour
        contour_area = cv2.contourArea(contour)

        # Perimeter
        perimeter = cv2.arcLength(contour, True)

        # Circularity
        if perimeter > 0:
            circularity = (
                4 * np.pi * contour_area /
                (perimeter ** 2)
            )
        else:
            circularity = 0

        circularities.append(circularity)

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # Solidity
        if hull_area > 0:
            solidity = contour_area / hull_area
        else:
            solidity = 0

        solidities.append(solidity)

    if len(areas) == 0:
        return {
            "candidate_area": 0,
            "candidate_area_ratio": 0,
            "num_regions": 0,
            "largest_region_area": 0,
            "mean_region_area": 0,
            "std_region_area": 0,
            "mean_aspect_ratio": 0,
            "mean_circularity": 0,
            "mean_solidity": 0,
        }

    return {
        "candidate_area": np.sum(areas),

        "candidate_area_ratio": (
            np.sum(areas) / retinal_area
            if retinal_area > 0 else 0
        ),

        "num_regions": len(areas),

        "largest_region_area": np.max(areas),

        "mean_region_area": np.mean(areas),

        "std_region_area": np.std(areas),

        "mean_aspect_ratio": np.mean(aspect_ratios),

        "mean_circularity": np.mean(circularities),

        "mean_solidity": np.mean(solidities),
    }


def extract_vessel_features(vessel_mask, retinal_mask):
    """
    Extract simple morphological features from the vessel mask.

    Parameters
    ----------
    vessel_mask : numpy.ndarray
        Binary vessel mask.

    retinal_mask : numpy.ndarray
        Binary retinal-region mask.

    Returns
    -------
    dict
        Numerical vessel features.
    """

    # Retinal area
    retinal_area = np.sum(retinal_mask > 0)

    # Vessel area
    vessel_area = np.sum(vessel_mask > 0)

    # Vessel density
    vessel_density = (
        vessel_area / retinal_area
        if retinal_area > 0 else 0
    )

    # Find connected vessel regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vessel_mask,
        connectivity=8
    )

    component_areas = []

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        # Ignore tiny isolated noise
        if area < 5:
            continue

        component_areas.append(area)

    # No valid vessel components
    if len(component_areas) == 0:
        return {
            "vessel_area": 0,
            "vessel_density": 0,
            "num_vessel_components": 0,
            "mean_vessel_component_area": 0,
            "std_vessel_component_area": 0,
        }

    return {
        "vessel_area": vessel_area,

        "vessel_density": vessel_density,

        "num_vessel_components": len(component_areas),

        "mean_vessel_component_area": np.mean(
            component_areas
        ),

        "std_vessel_component_area": np.std(
            component_areas
        ),
    }



# ======================================================
# Mask builders (shared by extract_all_features and the
# visualization notebook, so both use identical logic)
# ======================================================

def build_bright_candidate_mask(green_enhanced, retinal_mask, threshold=180):
    """
    Bright candidate regions: pixels of the CLAHE-enhanced green channel
    above `threshold`, restricted to the retinal region.
    These are candidates, not confirmed lesions (the optic disc and
    illumination artefacts are also bright).
    """

    _, candidate_mask = cv2.threshold(
        green_enhanced,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    candidate_mask = cv2.bitwise_and(
        candidate_mask,
        candidate_mask,
        mask=retinal_mask
    )

    return candidate_mask


def build_vessel_mask(green_enhanced, retinal_mask, kernel_size=15,
                      return_intermediates=False):
    """
    Candidate vessel mask: black-hat transform -> Otsu threshold ->
    3x3 morphological opening.

    If return_intermediates is True, also returns a dict with the
    black-hat image, the mask before opening and the Otsu threshold.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        green_enhanced,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    blackhat = cv2.bitwise_and(
        blackhat,
        blackhat,
        mask=retinal_mask
    )

    otsu_threshold, otsu_mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    small_kernel = np.ones((3, 3), np.uint8)

    vessel_mask = cv2.morphologyEx(
        otsu_mask,
        cv2.MORPH_OPEN,
        small_kernel
    )

    if return_intermediates:
        return vessel_mask, {
            "blackhat": blackhat,
            "otsu_mask": otsu_mask,
            "otsu_threshold": otsu_threshold,
        }

    return vessel_mask


def extract_all_features(image_path, include_red_lesions=True, red_params=None):
    """
    Run the complete preprocessing and feature extraction
    pipeline for one fundus image.

    Parameters
    ----------
    image_path : str or Path
        Path to the fundus image.

    include_red_lesions : bool
        If True, also extract the red-lesion candidate family
        (computed at higher resolution, see red_lesion_features.py).
        Set to False to reproduce the original 27 features.

    red_params : dict, optional
        Overrides for red_lesion_features.PARAMS.

    Returns
    -------
    dict
        Dictionary containing all extracted features.
    """

    from src.preprocessing.image_preprocessing import preprocess_image

    # --------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------

    result = preprocess_image(image_path)

    masked_rgb = result["masked_rgb"]
    retinal_mask = result["retinal_mask"]
    green_enhanced = result["green_enhanced"]

    # --------------------------------------------------
    # 2. Candidate abnormal-region mask
    # --------------------------------------------------

    candidate_mask = build_bright_candidate_mask(
        green_enhanced,
        retinal_mask
    )

    # --------------------------------------------------
    # 3. Vessel mask
    # --------------------------------------------------

    vessel_mask = build_vessel_mask(
        green_enhanced,
        retinal_mask
    )

    # --------------------------------------------------
    # 4. Extract feature families
    # --------------------------------------------------

    color_features = extract_color_features(
        masked_rgb,
        retinal_mask
    )

    texture_features = extract_texture_features(
        green_enhanced
    )

    morphology_features = extract_morphology_features(
        candidate_mask,
        retinal_mask
    )

    vessel_features = extract_vessel_features(
        vessel_mask,
        retinal_mask
    )

    # --------------------------------------------------
    # 5. Combine everything
    # --------------------------------------------------

    features = {}

    features.update(color_features)
    features.update(texture_features)
    features.update(morphology_features)
    features.update(vessel_features)

    # --------------------------------------------------
    # 6. Red-lesion candidate features (new family)
    #    Uses its own higher-resolution, aspect-preserving copy of the
    #    image because microaneurysm-sized spots vanish at 512 x 512.
    # --------------------------------------------------

    if include_red_lesions:
        from src.features.red_lesion_features import (
            load_retina,
            extract_red_lesion_features,
        )

        rgb_hr, mask_hr = load_retina(image_path)
        features.update(
            extract_red_lesion_features(rgb_hr, mask_hr, red_params)
        )

    return features