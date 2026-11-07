# CompositeGuard AI

An automated visual inspection aid for carbon-fibre-look composite surfaces. Given a photo, it classifies the surface as **good**, **recoverable defect**, or **non-recoverable defect**, assigns a severity level, shows where the model's attention was strongest, measures that region in pixels, and exports a PDF report.

> **Research prototype. Not a certified safety tool.** The output is an automated inspection aid, not a structural safety assessment. Surface photos cannot show hidden internal defects such as delamination or voids; those need specialised methods such as ultrasound or thermography.

## What it does

- **Upload and inspect** a JPG or PNG in a Streamlit dashboard, with clear messages for unreadable, tiny, blank, or wrong-type files.
- **Classify** the image with a fine-tuned ResNet18 (good / recoverable / non-recoverable).
- **Severity** (None, Low, Medium, High) from configurable rules, plus a "human review recommended" flag when the model is not confident.
- **Attention map** (Grad-CAM) beside the original image, with the attention region measured in pixels (area, box size, longest span).
- **PDF report** with the images, results, settings used, and the known limits.
- **History** of past inspections in a local SQLite database (results only, never the images).

## Results

All numbers below come from the **held-out test set (225 images, 75 per class), which was used once** after all settings were frozen (`reports/frozen_settings.txt`). Full output, including the classical baselines and 95% ranges, is in `reports/test_results.txt`.

| Model | AUC, defect vs good | AUC, non-recoverable vs good | Balanced accuracy (task 1 / task 2) | 3-class accuracy |
|---|---|---|---|---|
| CNN, neutral input (primary) | 0.958 | 0.961 | 0.870 / 0.880 | 85.8% |
| CNN, normal input | 0.963 | 0.949 | 0.887 / 0.887 | 88.4% |

- The two CNNs perform similarly; the difference is within the noise of a 225-image test set.
- 34 of 225 test images were flagged "human review recommended".
- The full pipeline takes a median of 284 ms per image on the development laptop's CPU (95th percentile 326 ms).
- "Neutral input" means grayscale with per-image brightness and contrast removed.

**Lighting robustness** (test accuracy when the lighting of the test images is changed artificially):

| Change | Normal input | Neutral input |
|---|---|---|
| None (original) | 0.884 | 0.858 |
| Darker (x0.5) | 0.818 | 0.867 |
| Brighter (x1.6) | 0.724 | 0.778 |
| Low contrast (x0.5) | 0.707 | 0.862 |
| Gamma dark (1.8) | 0.827 | 0.849 |
| Gamma bright (0.55) | 0.849 | 0.853 |
| Shade left to right | 0.862 | 0.836 |

Neutral input was much more stable under darker and lower-contrast images. It did not help under shading, and strong over-exposure still cost it 8 points.

**Localization** (10 validation images with hand-drawn boxes): the attention regions did **not** beat simply marking the whole image (mean IoU: Grad-CAM 0.105, occlusion 0.047, edge-density hot spots 0.008, mark-everything 0.131). The highlighted region is therefore a cue for human review, not a defect outline.

## Limitations

- One dataset from one manufacturer. Results on other components, cameras, or materials may differ.
- The train/validation/test split is random. The dataset does not say whether several photos show the same component, so similar photos may appear in both training and test images, and results on new components may be lower.
- Localization is not validated (see above). Boxes were drawn by one non-expert.
- Sizes are in pixels of a 224 x 224 image. No physical scale is known, so no millimetre values are given.
- Severity levels are proxies for the dataset's labels (recoverable / non-recoverable). They are not physical damage measures, and the size rule is off by default because it is not validated.
- Lighting changes were synthetic. Each model was trained once.
- Model scores are shown as "confidence", but neural-network scores are often overconfident and are not true probabilities.

## Dataset and citation

Images: *Surface defect classification in carbon look components* dataset, released openly by the authors: https://github.com/airtlab/surface-defect-classification-in-carbon-look-components-dataset (check the repository page for its terms of use). This project uses the multi-class folder: 1,500 images of 224 x 224 pixels, 500 each for no defect, recoverable defects, and non-recoverable defects.

> Silenzi, A.; Castorani, V.; Tomassini, S.; Falcionelli, N.; Contardo, P.; Bonci, A.; Dragoni, A.F.; Sernani, P. Quality Control of Carbon Look Components via Surface Defect Classification with Deep Neural Networks. *Sensors* 2023, 23(17), 7607. https://doi.org/10.3390/s23177607

The dataset authors report about 97% accuracy for a DenseNet121 on an **augmented** version of the data, and precision and recall between 86% and 93% per class for their best models on the original images. Their splits, models, and protocol differ from ours, so those numbers are **not directly comparable** with the table above.

## Setup (Windows, VS Code)

Developed with Python 3.14 on Windows. Run these in the VS Code terminal, in the project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Command Prompt: .venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Download the dataset ZIP from the repository above, extract it, and place it so the images are at `data/raw/carbon-dataset/carbon-multiclass/<class>/*.jpg`. Trained weights and dataset images are **not** included in this repository.

## Run

```powershell
python src\make_splits.py             # only if data/splits.csv is missing (seed 42)
python src\train_cnn.py               # normal model  -> models/resnet18_best.pt   (about 5 to 15 min on CPU)
python src\train_cnn.py neutral       # neutral model -> models/resnet18_neutral.pt
streamlit run app.py                  # dashboard at http://localhost:8501
python -m pytest -q                   # 44 tests
```

## Reproducing the experiments

Run from the project root, in this order. Validation-set experiments:

1. `python src\eval_baseline.py`, `eval_variants.py`, `diagnose_scores.py`, `eval_features.py`: thresholding baseline and edge-density features.
2. `python src\train_cnn.py` and `python src\train_cnn.py neutral`: CNN training and validation results.
3. `python src\eval_lighting.py`: lighting-change test (add `test` to run on the test split).
4. `python src\eval_localization_val.py`: localization on the hand-drawn validation boxes (`data/annotations_val`).
5. `python src\eval_severity.py`: severity table.

Final evaluation (**once**; it creates `reports/test_run.lock` and refuses to run twice):

```powershell
python src\final_eval.py              # writes reports/test_results.txt
python src\failure_gallery.py         # error and correct-prediction galleries
python src\eval_lighting.py test
```

## Project structure

```
compositeguard-ai/
  app.py              Streamlit dashboard
  requirements.txt    package versions used
  src/                code (modules and experiment scripts, see below)
  tests/              44 pytest tests
  data/
    raw/              dataset images (not in version control)
    splits.csv        train / val / test split
    annotations/      hand-drawn boxes, 5 training images
    annotations_val/  hand-drawn boxes, 10 validation images
    inspections.db    inspection history (created by the app)
  models/             trained weights (not in version control)
  reports/            results, figures, example PDFs, frozen settings
  notebooks/          unused
```

**Modules used by the app (`src/`):** `preprocessing.py` (filters, contrast), `detection.py` (thresholding, edges, regions, edge-density map), `measurement.py` (region size in pixels), `evaluation.py` (metrics), `dataset.py` and `model.py` (data loading, ResNet18), `gradcam.py` and `occlusion.py` (attention maps), `severity.py` (rules), `pipeline.py` (one `inspect` function with input checks), `visuals.py`, `report.py` (PDF), `history.py` (SQLite).

**Experiment and check scripts (`src/`):** `explore_images.py`, `demo_preprocessing.py`, `demo_detection.py`, `demo_regions.py` (learning the image-processing steps); `make_splits.py` (the split); `eval_baseline.py`, `eval_variants.py`, `diagnose_scores.py`, `eval_features.py` (classical methods); `demo_heatmap.py`, `check_border.py`, `eval_localization.py`, `annotate.py` (classic localization and box drawing); `check_torch.py`, `check_data.py`, `train_cnn.py` (CNN); `eval_lighting.py`, `demo_gradcam.py`, `eval_localization_val.py`, `check_attention_size.py` (lighting and attention experiments); `demo_measurement.py`, `eval_severity.py`, `demo_pipeline.py`, `demo_report.py`, `demo_history.py` (module checks); `list_val_examples.py`, `make_requirements.py` (helpers); `final_eval.py`, `failure_gallery.py` (final evaluation).

## Credits and license

Dataset: Silenzi et al. (above), images provided by HP Composites S.p.A. Add a license for your own code before publishing it.