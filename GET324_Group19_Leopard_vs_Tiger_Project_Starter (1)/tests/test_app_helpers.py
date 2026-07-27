"""Small tests for preprocessing-independent application logic."""

import numpy as np
from PIL import Image

from inference_utils import interpret_probability, prepare_image


def test_prepare_image_returns_expected_batch_shape():
    image = Image.new("RGB", (400, 300), color=(200, 150, 100))
    batch = prepare_image(image, (224, 224))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_interpret_probability_returns_leopard_below_threshold():
    label, confidence = interpret_probability(
        0.20, ["leopard", "tiger"], 0.5
    )
    assert label == "leopard"
    assert confidence == 0.80


def test_interpret_probability_returns_tiger_at_threshold():
    label, confidence = interpret_probability(
        0.50, ["leopard", "tiger"], 0.5
    )
    assert label == "tiger"
    assert confidence == 0.50
