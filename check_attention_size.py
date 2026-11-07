
import numpy as np
import torch

from dataset import CompositeDataset
from gradcam import GradCAM
from model import build_model
from pipeline import CAM_THRESHOLD, MAX_REGION_PERCENT, MODEL_FILE


model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

cam_tool = GradCAM(model)
dataset = CompositeDataset("val", neutral=True)

areas = {1: [], 2: []}

for i in range(len(dataset)):
    tensor, label = dataset[i]

    if label == 0:
        continue

    cam, _, _ = cam_tool(tensor, label)
    area_percent = float(
        (cam >= CAM_THRESHOLD).mean() * 100
    )
    areas[label].append(area_percent)

print(
    "Share of the image covered by the Grad-CAM region "
    "(validation, defective images):"
)

for label, name in [
    (1, "recoverable"),
    (2, "non-recoverable"),
]:
    a = np.array(areas[label])

    if len(a) == 0:
        print(f"  {name:16s} images=0 -- check the validation dataset labels")
        continue

    print(
        f"  {name:16s} images={len(a)}  "
        f"median={np.median(a):.0f}%  "
        f"middle half={np.percentile(a, 25):.0f}%-"
        f"{np.percentile(a, 75):.0f}%  "
        f"over {MAX_REGION_PERCENT:.0f}% (would be dropped): "
        f"{int((a > MAX_REGION_PERCENT).sum())}"
    )
