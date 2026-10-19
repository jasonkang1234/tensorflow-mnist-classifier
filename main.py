import argparse
import logging
from train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorFlow MNIST CNN Classifier")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model-path", type=str, default="mnist_cnn.keras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
