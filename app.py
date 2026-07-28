"""GET 324 Group 19 Streamlit application.

The application loads the model trained in the accompanying Google Colab
notebook and classifies one uploaded image as a leopard or a tiger.
"""

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
}


def load_config() -> dict:
    """Load model metadata, using safe defaults if the file is unavailable."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = {"class_names", "image_size", "threshold"}
    if not required_keys.issubset(config):
        raise ValueError(
            "model_info.json must contain class_names, image_size and threshold."
        )
    if config["class_names"] != ["leopard", "tiger"]:
        raise ValueError("The expected class order is ['leopard', 'tiger'].")
    return config


@st.cache_resource(show_spinner="Loading the trained model...")
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load the saved Keras model once and reuse it across Streamlit reruns."""
    return tf.keras.models.load_model(model_path, compile=False)


def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
    config: dict,
) -> tuple[str, float, float]:
    """Prepare an image, run inference and return the prediction details."""
    image_size = tuple(config["image_size"])
    batch = prepare_image(image, image_size)
    tiger_probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
    label, confidence = interpret_probability(
        tiger_probability,
        config["class_names"],
        float(config["threshold"]),
    )
    return label, confidence, tiger_probability


def main() -> None:
    st.set_page_config(
        page_title="Leopard or Tiger Classifier",
        page_icon="🐆",
        layout="centered",
    )

    st.title("Leopard or Tiger?")
    st.caption("GET 324 Laboratory Exercise 10 · Group 19")
    st.write(
        "Upload a clear photograph of one animal. The trained model will classify "
        "the image as a leopard or a tiger and display its confidence score."
    )

    with st.expander("How the classifier works"):
        st.write(
            "The project uses MobileNetV3Small transfer learning. The convolutional "
            "base extracts visual features, while a binary classification layer "
            "produces the probability used for the final prediction."
        )
        st.info(
            "Confidence is the model's score, not a guarantee. Images containing "
            "other animals, drawings, heavy obstruction or unusual angles may be "
            "classified incorrectly."
        )

    uploaded_file = st.file_uploader(
        "Choose a leopard or tiger image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear colour image in JPG, JPEG, PNG or WEBP format.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"The uploaded file could not be opened as an image: {error}")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if not MODEL_PATH.exists():
        st.error(
            "The trained model file is missing. Run the Colab notebook, download "
            "`leopard_tiger_model.keras`, and place it beside `app.py`."
        )
        return

    try:
        config = load_config()
        model = load_trained_model(str(MODEL_PATH))
        with st.spinner("Analysing the image..."):
            label, confidence, tiger_probability = predict_image(
                model, image, config
            )
    except Exception as error:
        st.exception(error)
        return

    MINIMUM_CONFIDENCE = 0.80

if confidence < MINIMUM_CONFIDENCE:
    st.warning(
        "Image not recognized. Please upload a clear photograph "
        "containing only a leopard or tiger."
    )
    st.metric("Highest model score", f"{confidence * 100:.2f}%")
    st.caption(
        "The model could not identify the image with sufficient confidence."
    )
    return

st.success(f"Prediction: {label.title()}")
st.metric("Model confidence", f"{confidence * 100:.2f}%")
st.progress(int(round(confidence * 100)))

    leopard_probability = 1.0 - tiger_probability
    st.write(
        {
            "Leopard probability": f"{leopard_probability * 100:.2f}%",
            "Tiger probability": f"{tiger_probability * 100:.2f}%",
        }
    )


if __name__ == "__main__":
    main()
