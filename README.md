# CompositeGuard AI

Automated visual inspection for carbon-fibre-look composite surfaces. Upload a photo and CompositeGuard classifies the surface, rates its severity, highlights where the model looked, measures that region, and exports a PDF report.

> **Research prototype. Not a certified safety tool.** Output is an inspection *aid*, not a structural safety assessment. Surface photos cannot reveal hidden internal defects such as delamination or voids; those need methods like ultrasound or thermography.

## Features

| | |
|---|---|
| **Classification** | Fine-tuned ResNet18 sorts each image into *good*, *recoverable defect*, or *non-recoverable defect*. |
| **Severity** | None / Low / Medium / High from adjustable rules, plus a **human review recommended** flag when the model is unsure. |
| **Attention map** | Grad-CAM overlay beside the original, with the highlighted region measured in pixels (area, box size, longest span). |
| **Input checks** | Clear messages for unsupported, unreadable, tiny, blank, or extremely dark/bright images. |
| **PDF report** | Images, results, settings used, and known limitations in one export. |
| **History** | Past results stored in a local SQLite database (results only, never the images). |

## Quick start

Requires Python 3 (developed on 3.14). Commands below work on macOS, Linux, and Windows; only the venv activation line differs.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Get the data

Download the dataset from the [authors' repository](https://github.com/airtlab/surface-defect-classification-in-carbon-look-components-dataset) (check its terms of use) and place the multi-class images at:

```
data/raw/carbon-dataset/carbon-multiclass/<class>/*.jpg
```

Dataset images and trained weights are **not** included in this repo.

### Train and run

```bash
python src/make_splits.py          # only if data/splits.csv is missing (seed 42)
python src/train_cnn.py            # normal-input model  -> models/resnet18_best.pt
python src/train_cnn.py neutral    # neutral-input model -> models/resnet18_neutral.pt
streamlit run app.py               # dashboard at http://localhost:8501
python -m pytest -q                # run the tests
```

Training takes roughly 5–15 minutes on CPU. The app uses the **neutral** model (`models/resnet18_neutral.pt`), so train that one at minimum.

## How it works

```
upload → validate → centre-crop to 224×224 → grayscale + per-image standardisation
       → ResNet18 → class probabilities → severity rules (+ review flag)
       → Grad-CAM region (only if a defect is reported) → pixel measurements
       → dashboard / PDF report / history
```

- **Neutral input** means grayscale with per-image brightness and contrast removed, which makes the model more stable under lighting changes.
- **Severity rules** (defaults in `src/severity.py`): report a defect at ≥ 0.50 defect confidence; Medium from 0.25 and High from 0.50 non-recoverable confidence; recommend review when top confidence is below 0.70. All are adjustable in the app sidebar. The size-based upgrade rule is **off by default** because it is not validated.
- **Attention regions** are the Grad-CAM area above 50% of its maximum. If that area covers more than half the image, no region is reported and nothing is measured.

## Results

All figures are from the **held-out test set (225 images, 75 per class), used once** after settings were frozen (`reports/frozen_settings.txt`). Full output, including classical baselines and 95% bootstrap ranges, is in `reports/test_results.txt`.

| Model | AUC, defect vs good | AUC, non-recoverable vs good | Balanced acc. (task 1 / 2) | 3-class accuracy |
|---|---|---|---|---|
| CNN, neutral input (primary) | 0.958 | 0.961 | 0.870 / 0.880 | 85.8% |
| CNN, normal input | 0.963 | 0.949 | 0.887 / 0.887 | 88.4% |

- The two CNNs are within the noise of a 225-image test set.
- Classical baselines (thresholding, edge density, sharpness) reached AUCs of only 0.51–0.67.
- 34 of 225 test images were flagged for human review.
- Full pipeline: median **284 ms** per image on CPU (95th percentile 326 ms).

**Lighting robustness** (test accuracy under synthetic lighting changes):

| Change | Normal input | Neutral input |
|---|---|---|
| None (original) | 0.884 | 0.858 |
| Darker (×0.5) | 0.818 | 0.867 |
| Brighter (×1.6) | 0.724 | 0.778 |
| Low contrast (×0.5) | 0.707 | 0.862 |
| Gamma dark (1.8) | 0.827 | 0.849 |
| Gamma bright (0.55) | 0.849 | 0.853 |
| Shade left to right | 0.862 | 0.836 |

Neutral input helped a lot under darker and low-contrast images, did not help under shading, and strong over-exposure still cost it about 8 points.

**Localization** (10 validation images with hand-drawn boxes): attention regions did **not** beat simply marking the whole image (mean IoU: Grad-CAM 0.105, occlusion 0.047, edge-density 0.008, mark-everything 0.131). Treat the highlighted region as a cue for human review, not a defect outline.

## Limitations

- One dataset from one manufacturer; other components, cameras, or materials may behave differently.
- The train/val/test split is random and the dataset does not say whether multiple photos show the same component, so near-duplicates may span splits and real-world results may be lower.
- Localization is not validated, and the boxes were drawn by one non-expert.
- Sizes are in pixels of a 224×224 image; no physical scale is known, so no millimetre values are given.
- Severity levels are proxies for the dataset's recoverable / non-recoverable labels, not physical damage measures.
- Lighting tests were synthetic, and each model was trained once.
- Displayed "confidence" values are softmax scores. Neural networks are often overconfident, so these are not true probabilities.

## Reproducing the experiments

Run from the project root, in this order.

**Validation-set experiments**

1. `eval_baseline.py`, `eval_variants.py`, `diagnose_scores.py`, `eval_features.py`: thresholding and edge-density baselines
2. `train_cnn.py` and `train_cnn.py neutral`: CNN training
3. `eval_lighting.py`: lighting-change test (add `test` to run on the test split)
4. `eval_localization_val.py`: localization against `data/annotations_val`
5. `eval_severity.py`: severity table

**Final evaluation (run once)**. It creates `reports/test_run.lock` and refuses to run twice.

```bash
python src/final_eval.py           # writes reports/test_results.txt
python src/failure_gallery.py      # error and correct-prediction galleries
python src/eval_lighting.py test
```

## Project structure

```
compositeguard-ai/
├── app.py                Streamlit dashboard (Inspect + History pages)
├── requirements.txt      Pinned package versions
├── src/
│   ├── pipeline.py       Inspector: validation, inference, Grad-CAM, measurement
│   ├── model.py          ResNet18 builder
│   ├── dataset.py        Data loading, class names, standardisation
│   ├── severity.py       Severity rules and disclaimer
│   ├── gradcam.py        Grad-CAM attention maps
│   ├── occlusion.py      Occlusion-based attention (comparison)
│   ├── measurement.py    Region size in pixels
│   ├── report.py         PDF report
│   ├── history.py        SQLite history
│   ├── visuals.py        Overlay drawing
│   ├── preprocessing.py  Filters and contrast
│   ├── detection.py      Thresholding, edges, regions (classical baseline)
│   ├── evaluation.py     Metrics
│   └── ...               Training, eval_*, demo_*, and check_* scripts
├── tests/                pytest suite
├── data/
│   ├── splits.csv        Train / val / test split
│   ├── annotations/      Hand-drawn boxes (5 training images)
│   ├── annotations_val/  Hand-drawn boxes (10 validation images)
│   ├── raw/              Dataset images (not tracked)
│   └── inspections.db    Inspection history (created by the app)
├── models/               Trained weights (not tracked)
└── reports/              Results, figures, galleries, frozen settings
```

## Dataset and citation

Images come from the *Surface defect classification in carbon look components* dataset (images provided by HP Composites S.p.A.). This project uses the multi-class folder: 1,500 images of 224×224 pixels, 500 each for no defect, recoverable defects, and non-recoverable defects.

> Silenzi, A.; Castorani, V.; Tomassini, S.; Falcionelli, N.; Contardo, P.; Bonci, A.; Dragoni, A.F.; Sernani, P. Quality Control of Carbon Look Components via Surface Defect Classification with Deep Neural Networks. *Sensors* 2023, 23(17), 7607. https://doi.org/10.3390/s23177607

The dataset authors report about 97% accuracy for DenseNet121 on an *augmented* version of the data, and 86–93% per-class precision and recall for their best models on the original images. Their splits, models, and protocol differ from this project's, so those numbers are **not directly comparable** with the table above.

## License

No license has been added yet. Add one (for example MIT) before publishing. The dataset has its own terms; see the dataset repository.
