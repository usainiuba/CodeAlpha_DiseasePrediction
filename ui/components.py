"""Small functions that return HTML for the app. No machine learning here.

Every string is built without line breaks or leading spaces, because
Streamlit's markdown would treat indented lines as a code block.
"""
from html import escape

from ui.labels import title

RING_LENGTH = 339.29


def fmt_percent(value):
    """Show a probability as a percent, never a false 0% or 100%."""
    if value > 0.999:
        return ">99.9%"
    if value < 0.001:
        return "<0.1%"
    return f"{value * 100:.1f}%"
  # circumference of the gauge circle (radius 54)

ECG_PATH = (
    "M0 40 L50 40 L62 40 L72 14 L82 66 L92 28 L100 40 L160 40 "
    "L172 40 L182 10 L192 70 L202 30 L210 40 L300 40"
)


def hero_html(metrics):
    """Top banner with a few real facts about the trained model."""
    chips = []
    if metrics:
        chips.append(f"Model: {escape(str(metrics.get('selected_model', 'n/a')))}")
        if "test" in metrics:
            chips.append(f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}")
        if "dataset" in metrics:
            chips.append(f"{metrics['dataset']['rows']} cases in the dataset")
    chip_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips)
    return (
        '<div class="hero fade-in">'
        '<div class="hero-text">'
        '<div class="hero-eyebrow">CodeAlpha ML Internship · Task 4</div>'
        "<h1>Cell Analysis Screening</h1>"
        "<p>See how a machine learning model reads breast tissue cell measurements "
        "and estimates whether they look benign or malignant. "
        "A learning project, not a medical tool.</p>"
        f'<div class="chips">{chip_html}</div>'
        "</div>"
        '<svg class="ecg" viewBox="0 0 300 80" aria-hidden="true">'
        f'<path class="ecg-track" d="{ECG_PATH}"/>'
        f'<path class="ecg-line" pathLength="100" d="{ECG_PATH}"/>'
        "</svg>"
        "</div>"
    )


def disclaimer_html():
    return (
        '<div class="disclaimer fade-in delay-1">'
        '<span class="disclaimer-icon" aria-hidden="true">!</span>'
        "<div><b>Not medical advice.</b> This is a student project trained on a small "
        "research dataset. It cannot diagnose anything. If you are worried about your "
        "health, please see a doctor.</div>"
        "</div>"
    )


def section_title_html(text, hint=""):
    hint_html = f'<span class="section-hint">{escape(hint)}</span>' if hint else ""
    return f'<div class="section-title">{escape(text)}{hint_html}</div>'


def steps_html():
    items = [
        ("1", "Pick a starting point", "Load a typical benign or malignant profile, or move the sliders yourself."),
        ("2", "Analyse", "Press the button and the model reads all ten measurements."),
        ("3", "Read the result", "See the probability and where the key values sit."),
    ]
    cards = "".join(
        f'<div class="step fade-in delay-{index + 1}"><div class="step-num">{num}</div>'
        f"<div><b>{escape(head)}</b><p>{escape(text)}</p></div></div>"
        for index, (num, head, text) in enumerate(items)
    )
    return f'<div class="steps">{cards}</div>'


def empty_state_html():
    return (
        '<div class="card empty fade-in delay-1">'
        '<div class="empty-icon" aria-hidden="true">🔬</div>'
        "<h3>No result yet</h3>"
        "<p>Choose a starting profile, then press <b>Analyse measurements</b>. "
        "Your result will appear here.</p>"
        "</div>"
    )


def result_html(prediction):
    """Animated ring with the probability, the band and two bars."""
    malignant, benign = prediction["malignant"], prediction["benign"]
    band = prediction["band"]
    tone = band["tone"]
    percent = fmt_percent(malignant)
    offset = RING_LENGTH * (1 - malignant)
    return (
        '<div class="card result fade-in">'
        '<div class="eyebrow">Model estimate</div>'
        f'<svg class="ring" viewBox="0 0 140 140" role="img" '
        f'aria-label="Probability of malignant: {percent}">'
        '<circle class="ring-track" cx="70" cy="70" r="54" transform="rotate(-90 70 70)"/>'
        f'<circle class="ring-fill tone-{tone}" cx="70" cy="70" r="54" '
        f'transform="rotate(-90 70 70)" style="stroke-dashoffset:{offset:.2f}"/>'
        f'<text x="70" y="70" text-anchor="middle" class="ring-number">{percent}</text>'
        '<text x="70" y="90" text-anchor="middle" class="ring-caption">malignant</text>'
        "</svg>"
        f'<div class="band tone-{tone}">{escape(band["label"])}</div>'
        f'<p class="muted center">{escape(band["message"])}</p>'
        '<div class="bars">'
        f'{_prob_bar("Benign", benign, "low")}'
        f'{_prob_bar("Malignant", malignant, "high")}'
        "</div>"
        "</div>"
    )


def _prob_bar(label, value, tone):
    return (
        '<div class="bar-row">'
        f'<div class="bar-head"><span>{escape(label)}</span><b>{fmt_percent(value)}</b></div>'
        f'<div class="bar-track"><div class="bar-fill tone-bg-{tone}" '
        f'style="width:{value * 100:.1f}%"></div></div>'
        "</div>"
    )


def position_html(rows):
    """Where the person's values sit between typical benign and typical malignant."""
    if not rows:
        return ""
    items = "".join(
        '<div class="pos-row">'
        f'<div class="pos-head"><span>{escape(title(r["feature"]))}</span>'
        f'<b>{escape(r["value_text"])}</b></div>'
        '<div class="pos-track">'
        '<span class="pos-mark" style="left:25%"></span>'
        '<span class="pos-mark" style="left:75%"></span>'
        f'<span class="pos-pin" style="left:{r["percent"]:.1f}%"></span>'
        "</div></div>"
        for r in rows
    )
    return (
        '<div class="card fade-in delay-2">'
        '<div class="eyebrow">Where your values sit</div>'
        '<div class="pos-legend"><span style="left:25%">Typical benign</span>'
        '<span style="left:75%">Typical malignant</span></div>'
        f"{items}"
        '<p class="muted small">The dot shows your value. The two ticks mark the typical '
        "value in benign and malignant cases from the dataset.</p>"
        "</div>"
    )


def models_html(cv, selected, skipped):
    """Cross-validated ROC-AUC for each model, as animated bars."""
    rows = []
    for name, scores in cv.items():
        auc = scores["roc_auc"]
        width = max(0.0, min(100.0, (auc["mean"] - 0.5) / 0.5 * 100))
        badge = '<span class="badge">Selected</span>' if name == selected else ""
        chosen = " chosen" if name == selected else ""
        rows.append(
            f'<div class="model-row{chosen}">'
            f'<div class="bar-head"><span>{escape(name)}{badge}</span>'
            f'<b>{auc["mean"]:.3f} ± {auc["std"]:.3f}</b></div>'
            f'<div class="bar-track"><div class="bar-fill tone-bg-brand" style="width:{width:.1f}%"></div></div>'
            f'<div class="muted small">Recall {scores["recall"]["mean"]:.3f} · '
            f'Precision {scores["precision"]["mean"]:.3f} · F1 {scores["f1"]["mean"]:.3f}</div>'
            "</div>"
        )
    note = ""
    if skipped:
        names = ", ".join(escape(n) for n in skipped)
        note = f'<p class="muted small">Not compared: {names}.</p>'
    return (
        '<div class="card fade-in">'
        '<div class="eyebrow">Model comparison (5-fold cross-validation, ROC-AUC)</div>'
        f'{"".join(rows)}'
        '<p class="muted small">Bars start at 0.5, which is what random guessing scores.</p>'
        f"{note}</div>"
    )


def tiles_html(test):
    items = [
        ("ROC-AUC", f"{test['roc_auc']:.3f}"),
        ("Accuracy", f"{test['accuracy'] * 100:.1f}%"),
        ("Sensitivity", f"{test['recall'] * 100:.1f}%"),
        ("Specificity", f"{test['specificity'] * 100:.1f}%"),
        ("Precision", f"{test['precision'] * 100:.1f}%"),
    ]
    tiles = "".join(
        f'<div class="tile fade-in delay-{i % 4 + 1}"><div class="tile-value">{value}</div>'
        f'<div class="tile-label">{label}</div></div>'
        for i, (label, value) in enumerate(items)
    )
    return f'<div class="tiles">{tiles}</div>'


def confusion_html(test):
    (tn, fp), (fn, tp) = test["confusion_matrix"]
    return (
        '<div class="card fade-in delay-2">'
        '<div class="eyebrow">Final test: what the model got right and wrong</div>'
        '<div class="cm">'
        '<div class="cm-h"></div><div class="cm-h">Predicted benign</div>'
        '<div class="cm-h">Predicted malignant</div>'
        f'<div class="cm-h">Actually benign</div><div class="cm-c good">{tn}</div>'
        f'<div class="cm-c bad">{fp}</div>'
        f'<div class="cm-h">Actually malignant</div><div class="cm-c bad">{fn}</div>'
        f'<div class="cm-c good">{tp}</div>'
        "</div>"
        f'<p class="muted small">Of {fn + tp} malignant cases the model caught {tp} and '
        f"missed {fn}. Of {tn + fp} benign cases it wrongly flagged {fp}.</p>"
        "</div>"
    )


def factor_bars_html(features, count=6):
    top = features[:count]
    biggest = max([f["importance"] for f in top] + [1e-9])
    rows = "".join(
        '<div class="bar-row">'
        f'<div class="bar-head"><span>{escape(title(f["feature"]))}</span>'
        f'<b>{max(f["importance"], 0):.3f}</b></div>'
        '<div class="bar-track"><div class="bar-fill tone-bg-brand" '
        f'style="width:{max(f["importance"], 0) / biggest * 100:.1f}%"></div></div></div>'
        for f in top
    )
    return (
        '<div class="card fade-in delay-3">'
        '<div class="eyebrow">What the model relies on most</div>'
        f"{rows}"
        '<p class="muted small">Drop in ROC-AUC when that measurement is shuffled. '
        "Size measurements overlap, so their credit is shared.</p>"
        "</div>"
    )


def about_html(metrics):
    dataset = (metrics or {}).get("dataset", {})
    rows = dataset.get("rows", 569)
    malignant = dataset.get("malignant", 212)
    return (
        '<div class="card fade-in">'
        "<h3>About this project</h3>"
        "<p>The model learns from the <b>Breast Cancer Wisconsin (Diagnostic)</b> dataset. "
        f"It holds {rows} cases ({malignant} malignant). Each case is a set of measurements "
        "taken from a digital image of cells from a breast mass. "
        "Only the ten average measurements are used. In real life these numbers come from software that analyses a lab image of the cells,not from patient. Here you set them with sliders, or load a typical profile, to see how the model reacts.</p>"
        "<p>Four models were compared with cross-validation: Logistic Regression, SVM, "
        "Random Forest and XGBoost. The best one was tested once on cases it had never seen.</p>"
        "<h3>What it cannot do</h3>"
        "<p>It cannot look at a patient. It only sees ten numbers. The dataset is small and old, "
        "so results would not carry over to new hospitals or equipment. "
        "A real diagnosis needs a doctor, a scan and a lab test.</p>"
        "</div>"
    )


def footer_html():
    return (
        '<div class="footer">Built for the CodeAlpha Machine Learning Internship · '
        "Python, scikit-learn, XGBoost and Streamlit</div>"
    )
