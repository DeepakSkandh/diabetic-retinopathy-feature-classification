import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np


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