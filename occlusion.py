
import numpy as np
import torch


def occlusion_map(model, image, target_class, patch=48, n_steps=8, batch_size=32):
    """Measure how hiding image patches changes model confidence."""
    model.eval()
    _, h, w = image.shape

    ys = np.linspace(0, h - patch, n_steps).astype(int)
    xs = np.linspace(0, w - patch, n_steps).astype(int)
    boxes = [(y, x) for y in ys for x in xs]

    drops = []
    with torch.no_grad():
        base = torch.softmax(
            model(image.unsqueeze(0)), dim=1
        )[0, target_class].item()

        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start:start + batch_size]
            batch = image.unsqueeze(0).repeat(len(chunk), 1, 1, 1)

            for k, (y, x) in enumerate(chunk):
                batch[k, :, y:y + patch, x:x + patch] = 0.0

            probs = torch.softmax(model(batch), dim=1)[:, target_class]
            drops.extend((base - probs).tolist())

    total = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for (y, x), d in zip(boxes, drops):
        total[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1

    heat = np.clip(total / np.maximum(count, 1), 0, None)
    if heat.max() > 0:
        heat = heat / heat.max()

    return heat, base
