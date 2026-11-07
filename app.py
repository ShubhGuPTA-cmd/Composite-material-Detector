
import sys
import sqlite3
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from history import delete_all, list_inspections, save_inspection  # noqa: E402
from pipeline import InspectionError, Inspector  # noqa: E402
from report import make_report  # noqa: E402
from severity import DEFAULT_RULES, DISCLAIMER  # noqa: E402
from visuals import attention_overlay  # noqa: E402


NICE_NAMES = {
    "negative": "No defect (good)",
    "recoverable_defects": "Recoverable defect",
    "non_recoverable_defects": "Non-recoverable defect",
}

st.set_page_config(
    page_title="CompositeGuard AI",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the model...")
def get_inspector():
    return Inspector()


def show_history():
    """Display saved inspection results."""
    st.subheader("Inspection history")

    try:
        rows = list_inspections(limit=100)
    except sqlite3.Error as err:
        st.error(f"Could not load inspection history: {err}")
        return

    if not rows:
        st.info("No inspections saved yet.")
        st.write("Go to Inspect and run an inspection to create history.")
        return

    display_rows = []

    for row in rows:
        probability = row["p_good"]

        display_rows.append({
            "Date": row["inspected_at"],
            "File": row["file_name"],
            "Predicted class": NICE_NAMES.get(
                row["predicted_class"],
                row["predicted_class"],
            ),
            "Severity": row["severity"],
            "Human review": (
                "Yes" if row["needs_review"] else "No"
            ),
            "Defect confidence": (
                f"{1 - probability:.0%}"
                if probability is not None
                else "N/A"
            ),
            "Attention region": (
                f"{row['attention_percent']:.0f}% of image"
                if row["localized"]
                and row["attention_percent"] is not None
                else "None"
            ),
        })

    st.dataframe(
        display_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The database stores inspection results, not uploaded images. "
        "History is stored locally in data/inspections.db."
    )

    st.divider()
    st.subheader("Delete history")

    confirm_delete = st.checkbox(
        "I understand that deleting all history cannot be undone."
    )

    if st.button(
        "Delete all history",
        disabled=not confirm_delete,
        type="secondary",
    ):
        try:
            delete_all()
            st.success("Inspection history deleted.")
            st.rerun()
        except sqlite3.Error as err:
            st.error(f"Could not delete history: {err}")


# ---------------- sidebar navigation ----------------

page = st.sidebar.radio(
    "Page",
    ["Inspect", "History"],
)

st.sidebar.header("Severity rules")

st.sidebar.caption(
    "Defaults are not tuned or validated. Changing them changes the "
    "labels, not the model."
)

rules = {
    "defect_threshold": st.sidebar.slider(
        "Report a defect from this confidence",
        0.30,
        0.90,
        float(DEFAULT_RULES["defect_threshold"]),
        0.05,
    ),
    "medium_threshold": st.sidebar.slider(
        "Medium severity from non-recoverable confidence",
        0.05,
        0.60,
        float(DEFAULT_RULES["medium_threshold"]),
        0.05,
    ),
    "high_threshold": st.sidebar.slider(
        "High severity from non-recoverable confidence",
        0.30,
        0.90,
        float(DEFAULT_RULES["high_threshold"]),
        0.05,
    ),
    "review_below": st.sidebar.slider(
        "Recommend human review if top confidence is below",
        0.50,
        0.95,
        float(DEFAULT_RULES["review_below"]),
        0.05,
    ),
    "use_size_rule": st.sidebar.checkbox(
        "Use size rule (not validated)",
        value=False,
    ),
}

if rules["medium_threshold"] > rules["high_threshold"]:
    st.sidebar.error(
        "The Medium threshold must not be above the High threshold."
    )
    st.stop()


# ---------------- application header ----------------

st.title("CompositeGuard AI")

st.caption(
    "Inspection aid for carbon-fibre-look composite surfaces "
    "(research prototype)"
)

st.info(DISCLAIMER)


# ---------------- history page ----------------

if page == "History":
    show_history()
    st.stop()


# ---------------- image upload ----------------

uploaded = st.file_uploader(
    "Upload a photo of the composite surface (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.write("Upload an image to begin.")
    st.stop()


signature = (
    uploaded.name,
    uploaded.size,
    tuple(sorted(rules.items())),
)


# ---------------- original image ----------------

left, right = st.columns(2)

with left:
    st.subheader("Original")

    try:
        st.image(uploaded, width=380)
    except Exception:
        st.warning(
            "The preview could not be shown. The file may be damaged."
        )


# ---------------- run inspection ----------------

if st.button("Run inspection", type="primary"):
    try:
        uploaded.seek(0)

        with st.spinner("Inspecting..."):
            new_result = get_inspector().inspect(
                uploaded,
                rules=rules,
            )

        st.session_state["saved"] = (
            signature,
            new_result,
        )

        # Save the inspection result to SQLite.
        # A database error should not invalidate the inspection.
        try:
            save_inspection(
                new_result,
                uploaded.name,
                rules,
            )
            st.session_state["history_save_error"] = None

        except sqlite3.Error as err:
            st.session_state["history_save_error"] = str(err)

    except InspectionError as err:
        st.session_state.pop("saved", None)
        st.error(str(err))
        st.stop()

    except Exception as err:
        st.session_state.pop("saved", None)
        st.error(f"Something unexpected went wrong: {err}")
        st.stop()


# ---------------- retrieve current result ----------------

saved = st.session_state.get("saved")

if saved is None:
    st.stop()

if saved[0] != signature:
    st.info(
        "The image or the rules changed. "
        "Press 'Run inspection' again."
    )
    st.stop()

result = saved[1]
verdict = result["verdict"]


if st.session_state.get("history_save_error"):
    st.warning(
        "The inspection completed, but its history could not be saved. "
        "Database error: "
        f"{st.session_state['history_save_error']}"
    )


# ---------------- model attention visualization ----------------

with right:
    st.subheader("Model attention (not a defect outline)")

    st.image(
        attention_overlay(result),
        width=380,
    )

    if result["localized"]:
        st.caption(
            "Red = strongest attention. "
            "Green outline = attention region."
        )

    elif result["cam"] is not None:
        st.caption(
            "Attention was spread over most of the image, "
            "so no region is outlined."
        )

    else:
        st.caption(
            "No defect reported, so no attention map is shown."
        )


# ---------------- inspection results ----------------

st.divider()

banner = {
    "None": st.success,
    "Low": st.info,
    "Medium": st.warning,
    "High": st.error,
}

banner[verdict["severity"]](
    f"Severity: {verdict['severity']}"
)

if verdict["needs_review"]:
    st.warning(
        "Human review recommended: the model is not very confident."
    )

st.write("**Why:**")

for reason in verdict["reasons"]:
    st.write(f"- {reason}")


# ---------------- model confidence ----------------

st.subheader("Model confidence by class")

for name, probability in result["class_probs"].items():
    st.progress(
        float(probability),
        text=f"{NICE_NAMES.get(name, name)}: {probability:.0%}",
    )


# ---------------- attention measurements ----------------

st.subheader("Attention region size")

if result["measurements"]:
    rows = [
        {
            "Region": i + 1,
            "Area (px)": measurement["area_px"],
            "Share of image (%)": round(
                measurement["area_percent"],
                1,
            ),
            "Box width x height (px)": (
                f"{measurement['bbox_width_px']} x "
                f"{measurement['bbox_height_px']}"
            ),
            "Longest span (px)": round(
                measurement["length_px"],
                1,
            ),
        }
        for i, measurement in enumerate(result["measurements"])
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.write("No region to measure.")


# ---------------- important notes ----------------

with st.expander("Important notes and limits", expanded=True):
    for note in result["notes"]:
        st.write(f"- {note}")


st.caption(
    f"Method: {result['method']}. "
    f"Inspection time: {result['seconds'] * 1000:.0f} ms."
)


# ---------------- PDF report download ----------------

st.divider()

st.download_button(
    "Download inspection report (PDF)",
    data=make_report(
        result,
        uploaded.name,
        rules,
    ),
    file_name=f"inspection_{Path(uploaded.name).stem}.pdf",
    mime="application/pdf",
)
