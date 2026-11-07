import cv2
import numpy as np


def attention_overlay(result):
    """Original image with the attention heat map blended in (red = strongest)."""
    base = cv2.cvtColor(result["image_gray"], cv2.COLOR_GRAY2RGB)
    if result["cam"] is None:
        return base
    heat = cv2.applyColorMap((result["cam"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
    outlines = cv2.findContours(result["region_mask"], cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(overlay, outlines, -1, (0, 255, 0), 1)
    return overlay