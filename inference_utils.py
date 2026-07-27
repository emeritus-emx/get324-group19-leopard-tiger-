"""Model-independent image preparation and probability interpretation."""

from __future__ import annotations

import numpy as np
from PIL import Image


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Convert an uploaded image into the model's expected input batch."""
    image = image.convert("RGB")
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def interpret_probability(
    tiger_probability: float,
    class_names: list[str],
    threshold: float,
) -> tuple[str, float]:
    """Convert the sigmoid output into a class label and confidence score."""
    tiger_probability = float(np.clip(tiger_probability, 0.0, 1.0))
    if tiger_probability >= threshold:
        return class_names[1], tiger_probability
    return class_names[0], 1.0 - tiger_probability
