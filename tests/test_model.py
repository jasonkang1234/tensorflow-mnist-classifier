import numpy as np
from model import build_model


def test_model_output_shape() -> None:
    model = build_model()
    x = np.random.rand(4, 28, 28, 1).astype("float32")
    output = model.predict(x, verbose=0)
    assert output.shape == (4, 10)


def test_model_output_probabilities() -> None:
    model = build_model()
    x = np.random.rand(2, 28, 28, 1).astype("float32")
    output = model.predict(x, verbose=0)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)
    assert (output >= 0).all()
