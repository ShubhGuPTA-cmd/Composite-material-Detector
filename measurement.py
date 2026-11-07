
import cv2
import numpy as np


def measure_mask(mask, mm_per_pixel=None, min_area=1):
    """Measure separate white regions in a mask, largest first."""
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    image_pixels = binary.shape[0] * binary.shape[1]
    regions = []

    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])

        if area < min_area:
            continue

        points = np.column_stack(
            np.where(labels == i)
        )[:, ::-1].astype(np.float32)

        hull = cv2.convexHull(points)[:, 0, :]
        gaps = hull[:, None, :] - hull[None, :, :]
        length = float(np.sqrt((gaps ** 2).sum(axis=2)).max())

        region = {
            "area_px": area,
            "area_percent": 100.0 * area / image_pixels,
            "bbox": (x, y, w, h),
            "bbox_width_px": w,
            "bbox_height_px": h,
            "length_px": length,
        }

        if mm_per_pixel is not None:
            region["area_mm2"] = area * mm_per_pixel ** 2
            region["length_mm"] = length * mm_per_pixel

        regions.append(region)

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions
