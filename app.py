"""GET 324 Group 19 Leopard versus Tiger Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from inference_utils import interpret_probability, prepare_image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "leopard_tiger_model.keras"
CONFIG_PATH = PROJECT_DIR / "model_info.json"
DEFAULT_CONFIG = {
    "class_names": ["leopard", "tiger"],
    "image_size": [224, 224],
    "threshold": 0.5,
    "minimum_confidence": 0.80,
}


def inject_styles() -> None:
    """Apply the project's wildlife-inspired visual design."""
    st.markdown(
        """
        <style>
        :root {
            --forest: #10231b;
            --forest-soft: #1b382b;
            --cream: #fff8e8;
            --amber: #f5b544;
            --rust: #c96c3b;
            --muted: #65736b;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(255,248,232,.80) 0%, rgba(255,248,232,.68) 50%, rgba(238,243,237,.75) 100%),
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1600' height='1000' viewBox='0 0 1600 1000'%3E%3Crect width='1600' height='1000' fill='%2310231b'/%3E%3Cpath d='M240 780c90-190 220-320 370-390 102-48 234-76 306-48 77 29 130 97 165 176 33 74 44 164 32 248-12 77-49 153-110 208-62 56-139 77-225 77-170 0-329-75-472-210-36-34-70-74-93-120-13-25-16-52-10-77 7-34 22-60 47-74Z' fill='%23f5b544' fill-opacity='.25'/%3E%3Cpath d='M920 310c-60 0-110 44-128 101-19 61-5 130 32 182 47 66 124 100 214 96 98-4 179-54 223-137 31-58 41-125 24-187-15-57-56-101-111-124-53-23-116-25-171-2-38 16-62 43-83 71Z' fill='%23ef8b48' fill-opacity='.28'/%3E%3Cpath d='M640 420c80-28 167-39 255-37 76 2 150 17 214 50 63 33 111 84 137 148 27 67 29 144 13 214-18 77-63 148-132 195-68 46-154 70-243 69-92-1-183-25-262-76-70-45-123-115-144-195-24-94-5-196 53-274 26-34 60-57 99-74Z' fill='%23fff8e8' fill-opacity='.14'/%3E%3C/svg%3E") center/cover no-repeat;
            background-attachment: fixed;
        }

        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.5rem;
            border-radius: 26px;
            color: var(--cream);
            background:
                linear-gradient(125deg, rgba(9,25,18,.97), rgba(27,56,43,.94)),
                repeating-linear-gradient(45deg, transparent 0 20px, rgba(255,255,255,.02) 20px 22px);
            box-shadow: 0 20px 55px rgba(16,35,27,.18);
            margin-bottom: 1.3rem;
        }

        .hero::after {
            content: "🐆";
            position: absolute;
            right: 2rem;
            bottom: -1.6rem;
            font-size: 8.5rem;
            opacity: .18;
            transform: rotate(-7deg);
        }

        .eyebrow {
            color: var(--amber);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #fffaf0;
            font-size: clamp(2rem, 6vw, 4.2rem);
            line-height: .98;
            margin: .65rem 0 1rem;
            max-width: 720px;
        }

        .hero p {
            color: #dce7df;
            max-width: 650px;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .status-row {
            display: flex;
            gap: .6rem;
            flex-wrap: wrap;
            margin-top: 1.4rem;
        }

        .status-pill {
            border: 1px solid rgba(255,255,255,.18);
            background: rgba(255,255,255,.08);
            border-radius: 999px;
            padding: .42rem .78rem;
            color: #edf4ef;
            font-size: .82rem;
        }

        .stApp .stMarkdownContainer,
        .stApp .stTextInput,
        .stApp .stTextArea,
        .stApp .stNumberInput,
        .stApp .stSelectbox,
        .stApp .stCheckbox,
        .stApp .stRadio,
        .stApp .stSlider,
        .stApp .stFileUploader,
        .stApp .stExpander,
        .stApp .stAlert {
            color: var(--forest) !important;
            background: rgba(255, 248, 232, 0.96) !important;
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
        }

        .stApp .stAlert {
            border: 1px solid rgba(16, 35, 27, 0.14);
        }

        .control-card,
        .result-card,
        .prob-card,
        .result-summary {
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            background: rgba(255,255,255,.92);
            border: 1px solid rgba(16,35,27,.10);
            box-shadow: 0 18px 40px rgba(16,35,27,.08);
            margin-bottom: 1rem;
        }

        .result-summary {
            display: grid;
            gap: .65rem;
        }

        .detail-text {
            color: #375141;
            opacity: .86;
            margin-top: .3rem;
            font-size: .92rem;
        }

        .detail-value {
            color: #10231b;
            font-size: 1.85rem;
            font-weight: 800;
            margin-top: .25rem;
        }

        .summary-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 5.5rem;
            padding: .6rem 1rem;
            border-radius: 999px;
            font-weight: 800;
            color: #10231b;
        }

        .summary-badge.leopard {
            background: #f5b544;
        }

        .summary-badge.tiger {
            background: #ef8b48;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,.82);
            border: 1px solid rgba(16,35,27,.10);
            border-radius: 20px;
            padding: 1.1rem;
            box-shadow: 0 12px 35px rgba(16,35,27,.07);
        }

        div.stButton > button {
            min-height: 3.2rem;
            border: 0;
            border-radius: 14px;
            color: #14251c;
            font-weight: 800;
            background: linear-gradient(90deg, #f7c45d, #ee9e38);
            box-shadow: 0 10px 24px rgba(201,108,59,.20);
        }

        div.stButton > button:hover {
            color: #14251c;
            border: 0;
            transform: translateY(-1px);
        }

        .result-card {
            padding: 1.35rem 1.45rem;
            border-radius: 18px;
            background: #10231b;
            color: white;
            box-shadow: 0 14px 36px rgba(16,35,27,.15);
        }

        .result-card .label {
            color: #f5b544;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .result-card .animal {
            font-size: 2.25rem;
            font-weight: 850;
            margin: .25rem 0;
        }

        .result-card .score {
            color: #dce7df;
            font-size: 1rem;
        }

        .prob-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255,255,255,.78);
            border: 1px solid rgba(16,35,27,.10);
        }

        [data-testid="stSidebar"] {
            background: #10231b;
        }

        [data-testid="stSidebar"] * {
            color: #f6f3e8;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_config() -> dict:
    """Load the settings produced by the Colab notebook."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"class_names", "image_size", "threshold"}
    if not required.issubset(config):
        raise ValueError(
            "model_info.json must contain class_names, image_size and threshold."
        )
    if config["class_names"] != ["leopard", "tiger"]:
        raise ValueError("Expected class order: ['leopard', 'tiger'].")
    config.setdefault("minimum_confidence", 0.80)
    return config


@st.cache_resource(show_spinner="Preparing the wildlife model...")
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load the trained model once for reuse across app reruns."""
    return tf.keras.models.load_model(model_path, compile=False)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    config: dict,
) -> tuple[str, float, float]:
    """Run one image through the trained binary classifier."""
    batch = prepare_image(image, tuple(config["image_size"]))
    tiger_probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
    label, confidence = interpret_probability(
        tiger_probability,
        config["class_names"],
        float(config["threshold"]),
    )
    return label, confidence, tiger_probability


def rerun_app() -> None:
    """Trigger a page refresh using a Streamlit API available in this runtime."""
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()
        return

    raise AttributeError("Streamlit does not expose a supported rerun API.")


def render_sidebar(config: dict) -> tuple[float, float, bool]:
    """Display project info and interactive analysis controls."""
    with st.sidebar:
        st.markdown("## 🐾 Field Guide")
        st.caption("GET 324 · Laboratory Exercise 10")
        st.markdown("**Group:** 19")
        st.markdown("**Task:** Leopard versus Tiger")
        st.markdown("**Model:** MobileNetV3Small")
        st.markdown("**Input:** RGB image, 224 × 224")
        st.divider()
        st.markdown("### Best image")
        st.write("• One visible animal")
        st.write("• Clear daylight")
        st.write("• Minimal obstruction")
        st.write("• JPG, PNG or WEBP")

        st.divider()
        st.markdown("### Analysis controls")
        decision_threshold = st.slider(
            "Decision threshold",
            min_value=0.20,
            max_value=0.80,
            value=min(max(float(config.get("threshold", 0.50)), 0.20), 0.80),
            step=0.01,
            help=(
                "Adjust the decision boundary used to classify leopard vs tiger. "
                "A higher threshold makes tiger predictions more conservative."
            ),
        )
        minimum_confidence = st.slider(
            "Uncertainty confidence",
            min_value=0.50,
            max_value=0.95,
            value=float(config.get("minimum_confidence", 0.80)),
            step=0.01,
            help=(
                "Raise this slider to require stronger model confidence before "
                "the app reports a definitive result."
            ),
        )
        show_raw_scores = st.checkbox(
            "Show raw model scores",
            value=False,
            help="Reveal the underlying tiger and leopard probability values.",
        )

        st.divider()
        st.markdown("### Model notes")
        st.caption(
            "The model is trained only for leopard and tiger. Unsupported images "
            "can still receive one of those labels with high confidence."
        )
    return decision_threshold, minimum_confidence, show_raw_scores


def main() -> None:
    st.set_page_config(
        page_title="WildSpot · Leopard or Tiger",
        page_icon="🐆",
        layout="wide",
    )
    inject_styles()

    try:
        config = load_config()
    except Exception as error:
        st.error(f"Configuration error: {error}")
        return

    decision_threshold, minimum_confidence, show_raw_scores = render_sidebar(config)

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Group 19 · Wildlife Vision Lab</div>
            <h1>Leopard or Tiger?</h1>
            <p>
                Upload a clear wildlife photograph and let the trained vision
                model compare the coat pattern, colour and visible form.
            </p>
            <div class="status-row">
                <span class="status-pill">MobileNetV3Small</span>
                <span class="status-pill">Binary classification</span>
                <span class="status-pill">Confidence aware</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    intro_left, intro_right = st.columns([1.35, 1], gap="large")
    with intro_left:
        st.subheader("Upload observation")
        uploaded_file = st.file_uploader(
            "Choose one wildlife image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear image containing one leopard or tiger.",
            label_visibility="collapsed",
        )

    with intro_right:
        st.subheader("Analysis review")
        st.write(
            "This app distinguishes only leopards and tigers. If your image "
            "contains another animal, the model will still choose its closest match."
        )
        st.markdown(
            f"**Decision threshold:** {decision_threshold:.2f}  \\"
            f"**Confidence threshold:** {minimum_confidence:.2f}"
        )
        st.caption(
            "Use the sidebar sliders to explore how stricter or looser criteria "
            "affect the final classification."
        )

    if uploaded_file is None:
        st.info("Upload an image to preview it and enable the analysis button.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"The uploaded file could not be opened as an image: {error}")
        return

    image_column, action_column = st.columns([1.25, 1], gap="large")
    with image_column:
        st.image(image, caption="Selected wildlife image", use_container_width=True)

    with action_column:
        st.markdown("### Ready for analysis")
        st.write(
            "Press the button once. The model will resize the image, extract "
            "visual features and calculate predicted probabilities."
        )
        analyse = st.button(
            "Run Wildlife Analysis",
            type="primary",
            use_container_width=True,
        )
        if st.button("Reset page", type="secondary", use_container_width=True):
            rerun_app()

    if not analyse:
        st.caption("No prediction has been made yet.")
        return

    if not MODEL_PATH.exists():
        st.error(
            "The trained model file is missing. Run the Colab notebook and put "
            "leopard_tiger_model.keras beside app.py."
        )
        return

    try:
        model = load_trained_model(str(MODEL_PATH))
        display_config = config.copy()
        display_config["threshold"] = decision_threshold
        with st.spinner("Examining visual patterns..."):
            label, confidence, tiger_probability = predict_image(
                model, image, display_config
            )
    except Exception as error:
        st.exception(error)
        return

    leopard_probability = 1.0 - tiger_probability

    st.divider()
    st.subheader("Analysis result")

    result_section, details_section = st.columns([1.1, 0.9], gap="large")
    with result_section:
        if confidence < minimum_confidence:
            st.warning(
                "Uncertain observation. Upload a clearer and closer photograph "
                "containing one leopard or tiger."
            )
            st.metric("Highest model score", f"{confidence * 100:.2f}%")
        else:
            icon = "🐆" if label == "leopard" else "🐅"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Predicted animal</div>
                    <div class="animal">{icon} {label.title()}</div>
                    <div class="score">Model confidence: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="prob-card">
                <b>Model decision</b><br>
                <span class="detail-text">Threshold</span>
                <div class="detail-value">{decision_threshold:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with details_section:
        st.markdown(
            f"""
            <div class="result-summary">
                <div><strong>Tiger probability</strong></div>
                <div class="summary-badge tiger">{tiger_probability * 100:.2f}%</div>
                <div class="detail-text">Raw sigmoid output from the model.</div>
            </div>
            <div class="result-summary">
                <div><strong>Leopard probability</strong></div>
                <div class="summary-badge leopard">{leopard_probability * 100:.2f}%</div>
                <div class="detail-text">Complementary probability value.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(round(tiger_probability * 100)))
        st.progress(int(round(leopard_probability * 100)))

    if show_raw_scores:
        with st.expander("Raw model scores and thresholds", expanded=True):
            st.write(
                "Tiger probability is the raw model output. The app uses the "
                "decision threshold to convert it to a class label and the "
                "uncertainty confidence level to mark low-confidence images."
            )
            st.metric("Tiger raw score", f"{tiger_probability * 100:.2f}%")
            st.metric("Leopard raw score", f"{leopard_probability * 100:.2f}%")
            st.write(f"Configured minimum confidence: {minimum_confidence:.2f}")

    with st.expander("Interpret this result"):
        st.write(
            "The displayed confidence is the model's measured preference between "
            "its two learned classes. It is not a biological identification guarantee."
        )
        st.write(
            f"The app is currently using a decision threshold of {decision_threshold:.2f} "
            f"and an uncertainty cutoff of {minimum_confidence:.2f}."
        )


if __name__ == "__main__":
    main()
