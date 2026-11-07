
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from dataset import CLASS_NAMES, standardize_image
from gradcam import GradCAM
from measurement import measure_mask
from model import build_model
from severity import DISCLAIMER, assess


MODEL_FILE = "models/resnet18_neutral.pt"
SUPPORTED = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 64
SIZE = 224
CAM_THRESHOLD = 0.5
MAX_REGION_PERCENT = 50.0

LOCALIZATION_NOTE = (
    "The highlighted region shows where the model's attention was strongest. "
    "On 10 validation images it did not outperform marking the whole image "
    "(IoU 0.105 vs 0.131), so treat it as a cue for human review, not a "
    "defect outline."
)

MEASUREMENT_NOTE = (
    "Sizes are in pixels of a 224 x 224 version of the image. "
    "No physical scale is known, so no millimetre values are given."
)

SCOPE_NOTE = (
    "The model was trained on photos of carbon-look components. Results on "
    "other materials or imaging methods are not meaningful, and surface "
    "photos cannot show hidden internal defects."
)

PREPARE = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Lambda(standardize_image),
])


class InspectionError(Exception):
    """A problem the user can understand and fix."""


def load_image(source):
    """Load and validate an image path or an uploaded file."""
    name = str(getattr(source, "name", source))

    if Path(name).suffix.lower() not in SUPPORTED:
        raise InspectionError(
            "Unsupported file type. Please upload a "
            + ", ".join(sorted(SUPPORTED))
            + " image."
        )

    if isinstance(source, (str, Path)) and not Path(source).exists():
        raise InspectionError(f"File not found: {source}")

    try:
        with Image.open(source) as im:
            image = im.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise InspectionError(
            "This file could not be read as an image. It may be damaged."
        ) from None

    width, height = image.size

    if min(width, height) < MIN_SIDE:
        raise InspectionError(
            f"The image is too small ({width} x {height}). "
            f"At least {MIN_SIDE} pixels on each side are needed."
        )

    return image


class Inspector:
    def __init__(self, model_file=MODEL_FILE):
        if not Path(model_file).exists():
            raise InspectionError(
                f"Model file not found: {model_file}. Train the model first."
            )

        self.model = build_model(pretrained=False)
        self.model.load_state_dict(
            torch.load(model_file, map_location="cpu")
        )
        self.model.eval()
        self.cam_tool = GradCAM(self.model)

    def inspect(self, source, rules=None):
        start = time.perf_counter()

        image = load_image(source)

        # Centre-crop to a square and resize to 224 x 224.
        square = ImageOps.fit(image, (SIZE, SIZE))
        gray = np.asarray(square.convert("L"))

        if gray.std() < 2:
            raise InspectionError(
                "The image looks blank or almost uniform. "
                "Please upload a clear photo of the surface."
            )

        notes = []

        if gray.mean() < 15 or gray.mean() > 240:
            notes.append(
                "The image is extremely dark or bright. "
                "Results may be unreliable."
            )

        tensor = PREPARE(square)

        # Initial classification does not need gradients.
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0].tolist()

        verdict = assess(probs, rules=rules)

        cam = None
        region_mask = np.zeros(
            (SIZE, SIZE), dtype=np.uint8
        )
        measurements = []
        localized = False
        area_percent = 0.0

        if verdict["severity"] != "None":
            # Select the more likely of the two defect classes.
            target = 1 + int(np.argmax(probs[1:]))

            # Grad-CAM calculates attention using gradients.
            cam, _, _ = self.cam_tool(tensor, target)

            candidate = (
                (cam >= CAM_THRESHOLD).astype(np.uint8) * 255
            )

            area_percent = float(
                (candidate > 0).mean() * 100.0
            )

            if area_percent <= MAX_REGION_PERCENT:
                # Accept this as a candidate localized attention region.
                localized = True
                region_mask = candidate
                measurements = measure_mask(region_mask)

                verdict = assess(
                    probs,
                    region_area_percent=area_percent,
                    rules=rules,
                )

                notes += [
                    LOCALIZATION_NOTE,
                    MEASUREMENT_NOTE,
                ]

            else:
                # Do not report an over-large region as a defect measurement.
                notes.append(
                    f"The model's attention is spread over "
                    f"{area_percent:.0f}% of the image, so no localized "
                    "region is reported and nothing is measured."
                )

        notes.append(SCOPE_NOTE)

        return {
            "image_gray": gray,
            "class_probs": dict(zip(CLASS_NAMES, probs)),
            "predicted_class": CLASS_NAMES[
                int(np.argmax(probs))
            ],
            "verdict": verdict,
            "cam": cam,
            "region_mask": region_mask,
            "measurements": measurements,
            "localized": localized,
            "cam_area_percent": area_percent,
            "notes": notes,
            "method": (
                "ResNet18 classifier (grayscale, per-image standardized "
                "input) + Grad-CAM attention region"
            ),
            "seconds": time.perf_counter() - start,
            "disclaimer": DISCLAIMER,
            "inspected_at": datetime.now(),
        }
