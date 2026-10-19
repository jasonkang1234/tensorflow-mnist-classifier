import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_model(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> keras.Model:
    """CNN model for MNIST digit classification."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model
