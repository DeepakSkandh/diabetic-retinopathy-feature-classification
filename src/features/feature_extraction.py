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