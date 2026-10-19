import logging
import tensorflow as tf
from model import build_model
from data import load_data

logger = logging.getLogger(__name__)


def train(
    batch_size: int = 128,
    epochs: int = 15,
    learning_rate: float = 0.001,
    model_path: str = "mnist_cnn.keras",
) -> tf.keras.callbacks.History:
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f  Test loss: %.4f", test_acc, test_loss)
    return history
