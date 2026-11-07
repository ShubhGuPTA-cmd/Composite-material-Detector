
DISCLAIMER = (
    "Automated inspection aid only. This is not a certified structural "
    "safety assessment. A qualified person must review the part."
)

LEVELS = ["None", "Low", "Medium", "High"]

DEFAULT_RULES = {
    "defect_threshold": 0.50,
    "medium_threshold": 0.25,
    "high_threshold": 0.50,
    "review_below": 0.70,
    "use_size_rule": False,
    "size_percent_for_upgrade": 10.0,
}


def assess(probs, region_area_percent=None, rules=None):
    """Assign a preliminary severity from three class probabilities."""

    if (
        len(probs) != 3
        or any(p < 0 or p > 1 for p in probs)
        or abs(sum(probs) - 1.0) > 1e-3
    ):
        raise ValueError(
            "probs must be three probabilities that add up to 1"
        )

    rules = {**DEFAULT_RULES, **(rules or {})}

    p_good, p_rec, p_non = probs
    p_defect = 1.0 - p_good

    reasons = [f"Model confidence that a defect is present: {p_defect:.0%}"]

    if p_defect < rules["defect_threshold"]:
        severity = "None"
        reasons.append(
            "Below the defect threshold, so no defect is reported."
        )
    else:
        reasons.append(
            f"Model confidence that the defect is non-recoverable: {p_non:.0%}"
        )

        if p_non >= rules["high_threshold"]:
            severity = "High"
        elif p_non >= rules["medium_threshold"]:
            severity = "Medium"
            reasons.append(
                "The model is unsure whether the defect is recoverable."
            )
        else:
            severity = "Low"
            reasons.append(
                "The model favours a recoverable defect."
            )

        if (
            rules["use_size_rule"]
            and region_area_percent is not None
            and region_area_percent >= rules["size_percent_for_upgrade"]
            and severity in ("Low", "Medium")
        ):
            severity = LEVELS[LEVELS.index(severity) + 1]
            reasons.append(
                f"Size rule (not validated): region covers "
                f"{region_area_percent:.1f}% of the image, severity raised."
            )

    needs_review = max(probs) < rules["review_below"]

    if needs_review:
        reasons.append(
            "Low model confidence: human review recommended."
        )

    return {
        "severity": severity,
        "reasons": reasons,
        "needs_review": needs_review,
        "p_defect": p_defect,
        "p_nonrecoverable": p_non,
    }
