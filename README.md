# TensorFlow MNIST Classifier

A convolutional neural network (CNN) built with TensorFlow/Keras to classify MNIST handwritten digits.
Inspired by the [TensorFlow official MNIST example](https://www.tensorflow.org/datasets/keras_example).

## Architecture

```
Input (28x28x1)
  → Conv2D(32) → Conv2D(64) → MaxPool → Dropout(0.25)
  → Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python main.py
python main.py --epochs 20 --batch-size 64 --model-path my_model.keras
```

## Test

```bash
pytest
```

## Results

Typical accuracy: **~99.2%** on the MNIST test set after 15 epochs.

## License

MIT — based on TensorFlow open source examples.
