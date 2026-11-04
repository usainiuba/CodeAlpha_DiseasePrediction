"""Streamlit app: cell measurement screening demo (educational).

Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from predict import (
    METRICS_PATH,
    MODEL_PATH,
    OPTIONS_PATH,
    load_json,
    load_model,
    position_rows,
    predict_one,
)
from ui.components import (
    about_html,
    confusion_html,
    disclaimer_html,
    empty_state_html,
    factor_bars_html,
    footer_html,
    hero_html,
    models_html,
    position_html,
    result_html,
    section_title_html,
    steps_html,
    tiles_html,
)
from ui.labels import FIELD_GROUPS, HELP, title

st.set_page_config(page_title="Cell Analysis Screening", page_icon="🔬", layout="wide")
css = (Path(__file__).parent / "ui" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if not (MODEL_PATH.exists() and METRICS_PATH.exists() and OPTIONS_PATH.exists()):
    st.error("The model has not been trained yet. Run `python train_model.py` first, "
             "then start the app again.")
    st.stop()


@st.cache_resource
def get_model():
    """Load the model once and reuse it between interactions."""
    return load_model()


model = get_model()
metrics = load_json(METRICS_PATH)
options = load_json(OPTIONS_PATH)


def key_of(name):
    return f"field_{name}"


# Give every slider a starting value the first time the page loads.
for field, spec in options.items():
    st.session_state.setdefault(key_of(field), float(spec["median"]))


def apply_preset(kind, spec_by_field):
    """Fill every slider with the typical benign, typical malignant or average value."""
    for field, spec in spec_by_field.items():
        st.session_state[key_of(field)] = float(spec[kind])


st.markdown(hero_html(metrics), unsafe_allow_html=True)
tab_assess, tab_perf, tab_about = st.tabs(["Assessment", "Model performance", "About"])

with tab_assess:
    st.markdown(disclaimer_html(), unsafe_allow_html=True)
    st.markdown(section_title_html("Start from a profile", "Optional. You can still change any slider."),
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.button("Typical benign", on_click=apply_preset, args=("benign_median", options), key="p_benign")
    p2.button("Typical malignant", on_click=apply_preset, args=("malignant_median", options), key="p_malignant")
    p3.button("Dataset average", on_click=apply_preset, args=("median", options), key="p_median")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.form("measurements"):
            values = {}
            for group_title, fields in FIELD_GROUPS:
                st.markdown(section_title_html(group_title), unsafe_allow_html=True)
                for field in fields:
                    spec = options[field]
                    values[field] = st.slider(
                        title(field),
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        step=float(spec["step"]),
                        format=f"%.{spec['decimals']}f",
                        help=HELP.get(field),
                        key=key_of(field),
                    )
            submitted = st.form_submit_button("Analyse measurements", type="primary")

    with right:
        if submitted:
            prediction = predict_one(model, values)
            st.markdown(result_html(prediction), unsafe_allow_html=True)
            rows = position_rows(values, options, metrics["top_features"])
            st.markdown(position_html(rows), unsafe_allow_html=True)
        else:
            st.markdown(empty_state_html(), unsafe_allow_html=True)
            st.markdown(steps_html(), unsafe_allow_html=True)

with tab_perf:
    st.markdown(tiles_html(metrics["test"]), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(models_html(metrics["cv"], metrics["selected_model"], metrics.get("skipped", {})),
                    unsafe_allow_html=True)
        st.markdown(factor_bars_html(metrics["top_features"]), unsafe_allow_html=True)
    with c2:
        st.markdown(confusion_html(metrics["test"]), unsafe_allow_html=True)
        st.markdown(
            '<div class="card fade-in delay-3"><div class="eyebrow">Reading these numbers</div>'
            "<p><b>Sensitivity</b> is the share of malignant cases the model catches. "
            "<b>Specificity</b> is the share of benign cases it correctly clears. "
            "In screening, missing a malignant case is the costlier mistake, "
            "so sensitivity matters most.</p></div>",
            unsafe_allow_html=True,
        )

with tab_about:
    st.markdown(about_html(metrics), unsafe_allow_html=True)

st.markdown(footer_html(), unsafe_allow_html=True)
