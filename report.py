import io
from datetime import datetime
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from pipeline import CAM_THRESHOLD, MAX_REGION_PERCENT
from severity import DEFAULT_RULES
from visuals import attention_overlay

NICE_NAMES = {
    "negative": "No defect (good)",
    "recoverable_defects": "Recoverable defect",
    "non_recoverable_defects": "Non-recoverable defect",
}

# Update these numbers after the final test-set evaluation (Stage 32).

# Final numbers from reports/test_results.txt (test set used once, 225 images).
KNOWN_LIMITS = [
    "Detection (held-out test set, 225 images): AUC 0.958 for defect vs good and 0.961 for "
    "non-recoverable vs good; 3-class accuracy 85.8%; 34 of 225 images were flagged for "
    "human review. Results on new components or cameras may differ.",
    "The photos come from one public dataset. Similar photos of the same component may be "
    "in both the training and test images, so results on new components may be lower.",
    "Lighting: the model is much more stable under darker or lower-contrast images than a "
    "version without per-image normalization, but strong over-exposure still lowers accuracy "
    "(test accuracy 77.8% at 1.6 times brightness, against 85.8% unchanged).",
    "Localization (10 validation images, hand-drawn boxes): the attention region did not "
    "beat marking the whole image (mean IoU 0.105 vs 0.131).",
    "Severity levels are proxies for the dataset labels (recoverable / non-recoverable). "
    "They are not physical damage measures.",
]


styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)
WARNING = ParagraphStyle("warning", parent=BODY, backColor=colors.HexColor("#FFF4CE"),
                         borderColor=colors.HexColor("#C9A400"), borderWidth=0.8,
                         borderPadding=6, spaceAfter=12)


def _p(text, style=BODY):
    return Paragraph(text, style)


def _picture(array, size_cm=7.5):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    buffer.seek(0)
    return RLImage(buffer, width=size_cm * cm, height=size_cm * cm)


def _table(rows, widths, header=True):
    table = Table(rows, colWidths=[w * cm for w in widths])
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF4")))
    table.setStyle(TableStyle(style))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(2 * cm, 1 * cm, "CompositeGuard AI - automated inspection aid, "
                                      "not a certified structural safety assessment.")
    canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def make_report(result, file_name, rules):
    """Return the PDF report as bytes."""
    rules = {**DEFAULT_RULES, **rules}
    verdict = result["verdict"]
    when = result.get("inspected_at") or datetime.now()

    story = [_p("CompositeGuard AI - Inspection Report", styles["Title"]),
             _p(escape(result["disclaimer"]), WARNING)]
    story.append(_table([
        [_p("<b>Inspection date</b>", SMALL), _p(when.strftime("%Y-%m-%d %H:%M"), SMALL)],
        [_p("<b>Image file</b>", SMALL), _p(escape(file_name), SMALL)],
        [_p("<b>Method</b>", SMALL), _p(escape(result["method"]), SMALL)],
        [_p("<b>Processing time</b>", SMALL), _p(f"{result['seconds'] * 1000:.0f} ms", SMALL)],
    ], [4, 13], header=False))
    story.append(Spacer(1, 10))
    story.append(_table([
        [_picture(result["image_gray"]), _picture(attention_overlay(result))],
        [_p("Inspected image (224 x 224 pixels, grayscale)", SMALL),
         _p("Model attention. Not a validated defect outline.", SMALL)],
    ], [8.5, 8.5], header=False))

    story += [Spacer(1, 10),
              _p(f"<b>Severity: {escape(verdict['severity'])}</b>", styles["Heading2"]),
              _p("Predicted class: " + NICE_NAMES[result["predicted_class"]])]
    if verdict["needs_review"]:
        story.append(_p("<b>Human review recommended:</b> the model is not very confident."))
    for reason in verdict["reasons"]:
        story.append(_p("- " + escape(reason)))
    story.append(Spacer(1, 6))
    rows = [[_p("<b>Class</b>", SMALL), _p("<b>Model confidence</b>", SMALL)]]
    rows += [[_p(NICE_NAMES[n], SMALL), _p(f"{p:.0%}", SMALL)]
             for n, p in result["class_probs"].items()]
    story.append(_table(rows, [8, 4]))

    story += [Spacer(1, 10), _p("<b>Attention region size</b> (pixels only, no physical scale)",
                                styles["Heading3"])]
    if result["measurements"]:
        heads = ["Region", "Area (px)", "Share of image", "Box w x h (px)", "Longest span (px)"]
        rows = [[_p(f"<b>{h}</b>", SMALL) for h in heads]]
        for i, m in enumerate(result["measurements"], 1):
            rows.append([_p(str(i), SMALL), _p(str(m["area_px"]), SMALL),
                         _p(f"{m['area_percent']:.1f}%", SMALL),
                         _p(f"{m['bbox_width_px']} x {m['bbox_height_px']}", SMALL),
                         _p(f"{m['length_px']:.1f}", SMALL)])
        story.append(_table(rows, [2, 3, 3.5, 4, 4.5]))
    elif result["cam"] is not None:
        story.append(_p(f"No region reported: attention was spread over "
                        f"{result['cam_area_percent']:.0f}% of the image."))
    else:
        story.append(_p("No defect reported, so no region was measured."))

    story += [Spacer(1, 10), _p("<b>Settings used</b>", styles["Heading3"]),
              _p(escape(
                  f"Defect reported from {rules['defect_threshold']:.0%} confidence; Medium from "
                  f"{rules['medium_threshold']:.0%} and High from {rules['high_threshold']:.0%} "
                  f"non-recoverable confidence; human review below {rules['review_below']:.0%} "
                  f"top confidence; size rule {'ON (not validated)' if rules['use_size_rule'] else 'off'}. "
                  f"Attention region = at least {CAM_THRESHOLD:.0%} of the maximum; regions over "
                  f"{MAX_REGION_PERCENT:.0f}% of the image are not reported."), SMALL)]

    story += [Spacer(1, 10), _p("<b>Notes and limits</b>", styles["Heading3"])]
    for text in result["notes"] + KNOWN_LIMITS:
        story.append(_p("- " + escape(text), SMALL))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="CompositeGuard AI inspection report")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()