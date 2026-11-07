
import cv2


def load_gray(path):
    """Read an image file and return it as a grayscale array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _check_odd(ksize):
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(
            "ksize must be an odd number of 3 or more (3, 5, 7...)"
        )


def denoise_gaussian(gray, ksize=5):
    _check_odd(ksize)
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def denoise_median(gray, ksize=5):
    _check_odd(ksize)
    return cv2.medianBlur(gray, ksize)


def denoise_bilateral(gray, d=7, sigma_color=50, sigma_space=50):
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def enhance_hist_eq(gray):
    return cv2.equalizeHist(gray)


def enhance_clahe(gray, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )
    return clahe.apply(gray)
