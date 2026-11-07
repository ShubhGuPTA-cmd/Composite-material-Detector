import cv2
import numpy as np

from preprocessing import (
    denoise_gaussian, denoise_median, denoise_bilateral, enhance_clahe,
)

import numpy as np


def threshold_otsu(image, dark_defects=True):
    """
    Apply Otsu thresholding to a grayscale image.

    dark_defects=True:
        Dark pixels become white in the mask.

    dark_defects=False:
        Bright pixels become white in the mask.

    Returns:
        threshold_value, binary_mask
    """
    if image is None:
        raise ValueError("Input image is empty or could not be loaded.")

    # Convert to grayscale if a color image is supplied.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu automatically selects a threshold value.
    threshold_type = cv2.THRESH_BINARY_INV if dark_defects else cv2.THRESH_BINARY

    threshold_value, mask = cv2.threshold(
        gray,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU
    )

    return threshold_value, mask


def clean_mask(mask, open_size=3, close_size=7):
    """
    Clean a binary mask using morphological opening and closing.

    Opening removes small white specks.
    Closing fills small black gaps inside white regions.
    """
    if mask is None:
        raise ValueError("Input mask is empty.")

    # Ensure the mask is binary and single-channel.
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    if open_size < 1 or close_size < 1:
        raise ValueError("Kernel sizes must be positive integers.")

    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size)
    )

    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size)
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_k
    )

    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        close_k
    )

    return cleaned


def find_regions(mask, min_area=30):
    """
    Find candidate regions in a binary mask.

    Each region contains:
        contour: region outline
        area: contour area in pixels
        bbox: (x, y, width, height)

    Regions are sorted from largest to smallest.
    """
    if mask is None:
        raise ValueError("Input mask is empty.")

    if min_area < 0:
        raise ValueError("min_area cannot be negative.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    result = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Compatible with OpenCV 3 and OpenCV 4.
    contours = result[-2]

    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore regions smaller than the minimum area.
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        regions.append({
            "contour": contour,
            "area": area,
            "bbox": (x, y, width, height)
        })

    # Put the largest regions first.
    regions.sort(
        key=lambda region: region["area"],
        reverse=True
    )

    return regions


def draw_regions(image, regions):
    """
    Draw green contour outlines and red bounding boxes.

    Returns a new image without modifying the original.
    """
    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 2:
        output = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )
    else:
        output = image.copy()

    for region in regions:
        x, y, width, height = region["bbox"]

        # Draw the contour in green.
        cv2.drawContours(
            output,
            [region["contour"]],
            -1,
            (0, 255, 0),
            1
        )

        # Draw the bounding box in red.
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            1
        )

    return output


def baseline_detect(gray, dark_defects=True, min_area=30):
    """Full baseline: blur, CLAHE, Otsu threshold, clean, find regions."""
    enhanced = enhance_clahe(denoise_gaussian(gray, 5))
    _, mask = threshold_otsu(enhanced, dark_defects=dark_defects)
    cleaned = clean_mask(mask)
    return find_regions(cleaned, min_area)


def pipeline_score(gray, blur="gaussian", use_clahe=True, dark_defects=True,
                   open_size=3, close_size=7, min_area=30):
    """Configurable baseline. Returns total defect-region area in pixels."""

    if blur == "gaussian":
        img = denoise_gaussian(gray, 5)
    elif blur == "median":
        img = denoise_median(gray, 5)
    elif blur == "bilateral":
        img = denoise_bilateral(gray)
    elif blur == "none":
        img = gray
    else:
        raise ValueError(f"Unknown blur type: {blur}")

    if use_clahe:
        img = enhance_clahe(img)

    _, mask = threshold_otsu(img, dark_defects=dark_defects)
    cleaned = clean_mask(mask, open_size, close_size)
    regions = find_regions(cleaned, min_area)

    return float(sum(r["area"] for r in regions))


def detect_edges(gray):
    """Detect image edges using the Canny edge detector."""
    import cv2

    return cv2.Canny(gray, 100, 200)


def white_fraction(mask):
    """Return the fraction of nonzero pixels in a binary mask."""
    import numpy as np

    if mask.size == 0:
        return 0.0

    return float(np.count_nonzero(mask)) / mask.size


def edge_density_map(gray, window=31):
    """For each pixel, calculate the local share of edge pixels."""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and at least 3")

    enhanced = enhance_clahe(denoise_gaussian(gray, 5))
    edges = detect_edges(enhanced)

    return cv2.blur(
        (edges > 0).astype(np.float32),
        (window, window)
    )



def hot_spots(density_map, k=1.5, min_area=60, margin=15):
    """Find areas with unusually high local edge density.
    Ignore the border where the density window is less reliable.
    Returns (cutoff_used, mask, regions).
    """
    h, w = density_map.shape

    inner = np.zeros((h, w), dtype=bool)
    inner[margin:h - margin, margin:w - margin] = True

    values = density_map[inner]
    cutoff = float(values.mean() + k * values.std())

    mask = ((density_map > cutoff) & inner).astype(np.uint8) * 255
    mask = clean_mask(mask)

    return cutoff, mask, find_regions(mask, min_area)




