from pathlib import Path

import cv2
import numpy as np


def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess an IDRiD fundus image.

    Steps:
    1. Load image
    2. Convert BGR -> RGB
    3. Resize to target size
    4. Create retinal-region mask
    5. Clean the mask using morphology
    6. Apply the mask to the RGB image
    7. Extract the green channel
    8. Enhance the green channel using CLAHE

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the fundus image.

    target_size : tuple
        Target image size as (width, height).
        Default is (512, 512).

    Returns
    -------
    dict
        Dictionary containing the processed images.
    """

    image_path = Path(image_path)

    # --------------------------------------------------
    # 1. Load image
    # --------------------------------------------------
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # --------------------------------------------------
    # 2. Convert BGR -> RGB
    # --------------------------------------------------
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------
    # 3. Resize
    # --------------------------------------------------
    image_rgb = cv2.resize(
        image_rgb,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------
    # 4. Create retinal-region mask
    # --------------------------------------------------
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    _, retinal_mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------
    # 5. Clean the retinal mask
    # --------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    # Closing: fills small gaps
    retinal_mask = cv2.morphologyEx(
        retinal_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Opening: removes small isolated noise
    retinal_mask = cv2.morphologyEx(
        retinal_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------
    # 6. Apply retinal mask
    # --------------------------------------------------
    masked_rgb = cv2.bitwise_and(
        image_rgb,
        image_rgb,
        mask=retinal_mask
    )

    # --------------------------------------------------
    # 7. Extract green channel
    # --------------------------------------------------
    green_channel = masked_rgb[:, :, 1]

    # --------------------------------------------------
    # 8. CLAHE enhancement
    # --------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    green_enhanced = clahe.apply(
        green_channel
    )

    return {
        "image_rgb": image_rgb,
        "retinal_mask": retinal_mask,
        "masked_rgb": masked_rgb,
        "green_channel": green_channel,
        "green_enhanced": green_enhanced,
    }