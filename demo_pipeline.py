import csv
import tempfile
from pathlib import Path
from PIL import Image
from pipeline import Inspector, InspectionError

inspector = Inspector()

# ---- Part 1: bad inputs must give clear messages, not crashes ----
print("Part 1: error handling")
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "notes.txt").write_text("hello")
    (tmp / "broken.jpg").write_bytes(b"this is not an image")
    Image.new("RGB", (32, 32), "gray").save(tmp / "tiny.png")
    Image.new("RGB", (300, 300), (128, 128, 128)).save(tmp / "blank.png")
    cases = [("missing file", tmp / "nothing.jpg"), ("wrong file type", tmp / "notes.txt"),
             ("damaged image", tmp / "broken.jpg"), ("too small", tmp / "tiny.png"),
             ("blank image", tmp / "blank.png")]
    for label, path in cases:
        try:
            inspector.inspect(path)
            print(f"  FAIL  {label}: no error was raised")
        except InspectionError as err:
            print(f"  PASS  {label}: {err}")

# ---- Part 2: real images, one per class, from the validation split ----
print("\nPart 2: one validation image per class")
with open("data/splits.csv", newline="") as fh:
    val_rows = [r for r in csv.DictReader(fh) if r["split"] == "val"]
seconds = []
for label in ["negative", "recoverable_defects", "non_recoverable_defects"]:
    row = next(r for r in val_rows if r["label"] == label)
    result = inspector.inspect(row["path"])
    v = result["verdict"]
    area = sum(m["area_px"] for m in result["measurements"])
    mask_pixels = int((result["region_mask"] > 0).sum())
    seconds.append(result["seconds"])
    print(f"  true={label:24s} predicted={result['predicted_class']:24s} "
          f"severity={v['severity']:6s} review={v['needs_review']}")
    print(f"      localized={result['localized']}  attention covers "
          f"{result['cam_area_percent']:.0f}% of the image  "
          f"regions={len(result['measurements'])}  area={area} px  "
          f"{result['seconds'] * 1000:.0f} ms  "
          f"{'PASS' if area == mask_pixels else 'FAIL'} measurement matches mask pixels")
print(f"\nAverage time per image: {sum(seconds) / len(seconds) * 1000:.0f} ms "
      "(the first image includes start-up time)")
print("Disclaimer in every result:", result["disclaimer"])