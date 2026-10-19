# LLM Wiki — TensorFlow MNIST Classifier

## Table of Contents
1. [What is a Large Language Model (LLM)?](#1-what-is-a-large-language-model-llm)
2. [CNN vs Transformer Architecture](#2-cnn-vs-transformer-architecture)
3. [How TensorFlow Supports LLMs](#3-how-tensorflow-supports-llms)
4. [From MNIST CNNs to LLMs: The Learning Path](#4-from-mnist-cnns-to-llms-the-learning-path)
5. [Popular LLMs Built on TensorFlow/Keras](#5-popular-llms-built-on-tensorflowkeras)
6. [Key Concepts Glossary](#6-key-concepts-glossary)
7. [Further Reading](#7-further-reading)

---

## 1. What is a Large Language Model (LLM)?

A **Large Language Model (LLM)** is a deep neural network trained on massive text corpora to understand and generate human language. Unlike this MNIST CNN (which classifies 28×28 images into 10 digit classes), LLMs process sequences of tokens and predict the next token — a deceptively simple objective that produces emergent reasoning capabilities at scale.

### Key Characteristics

| Property | This MNIST CNN | Typical LLM |
|---|---|---|
| Input | 28×28 pixel image | Sequence of text tokens |
| Output | 10-class probability | Next token probability over vocabulary |
| Parameters | ~1.2M | 7B – 1T+ |
| Architecture | CNN | Transformer (self-attention) |
| Training data | 60,000 images | Trillions of tokens |
| Training time | Minutes (CPU) | Weeks (thousands of GPUs) |

---

## 2. CNN vs Transformer Architecture

### CNN (this repo)
```
Input image
  → Conv2D (local feature extraction)
  → MaxPooling (spatial downsampling)
  → Flatten → Dense (classification)
```
- **Inductive bias:** Locality and translation invariance
- **Strength:** Efficient for fixed-size grid inputs (images)
- **Limitation:** Fixed receptive field; no global context

### Transformer (LLMs)
```
Input tokens
  → Embedding + Positional Encoding
  → Multi-Head Self-Attention (global context)
  → Feed-Forward Network
  → [repeat N layers]
  → Output head
```
- **Inductive bias:** None (learns all relationships from data)
- **Strength:** Global context, scales with data and compute
- **Limitation:** Quadratic attention cost O(n²) in sequence length

---

## 3. How TensorFlow Supports LLMs

### KerasNLP
High-level library for NLP and LLM fine-tuning:
```python
import keras_nlp

# Load a pretrained GPT-2
model = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")
model.generate("The future of AI is", max_length=50)
```

### TensorFlow Hub
Pretrained model repository:
```python
import tensorflow_hub as hub

# BERT embedding
embed = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
```

### Fine-tuning BERT with Keras
```python
import keras_nlp

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=2,
)
classifier.fit(train_dataset, epochs=3)
```

---

## 4. From MNIST CNNs to LLMs: The Learning Path

```
MNIST CNN (this repo)
    ↓  Add recurrence
RNN / LSTM (sequential data)
    ↓  Replace recurrence with attention
Transformer Encoder (BERT, classification)
    ↓  Add decoder, train on text prediction
GPT-style LLM (text generation)
    ↓  Scale up parameters + data
Large Language Model (GPT-4, Claude, Gemini)
```

The CNN you are running right now teaches the foundational ideas:
- **Layers** — stacking transformations to learn hierarchical features
- **Dropout** — regularisation to prevent overfitting
- **Softmax output** — probability distribution over classes
- **Backpropagation** — gradient-based weight updates

All of these carry directly into LLM architectures.

---

## 5. Popular LLMs Built on TensorFlow/Keras

| Model | Organisation | Notes |
|---|---|---|
| BERT | Google | Bidirectional encoder, fine-tuning tasks |
| T5 | Google | Text-to-text framework |
| PaLM 2 | Google | Multilingual, reasoning |
| Gemini | Google DeepMind | Multimodal (text + image + audio) |
| GPT-2 | OpenAI | Available via KerasNLP |

---

## 6. Key Concepts Glossary

| Term | Definition |
|---|---|
| **Token** | Smallest unit of text (word piece, character, or byte) |
| **Embedding** | Dense vector representation of a token |
| **Attention** | Mechanism to weigh the importance of each token relative to others |
| **Self-attention** | Attention where keys, queries, and values all come from the same sequence |
| **Fine-tuning** | Continuing training of a pretrained model on a smaller task-specific dataset |
| **Prompt** | Input text given to an LLM to elicit a response |
| **Context window** | Maximum number of tokens an LLM can process in one pass |
| **Hallucination** | When an LLM generates plausible-sounding but factually incorrect output |
| **RLHF** | Reinforcement Learning from Human Feedback — used to align LLMs with human preferences |
| **RAG** | Retrieval-Augmented Generation — grounding LLM responses with retrieved documents |

---

## 7. Further Reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [KerasNLP documentation](https://keras.io/keras_nlp/)
- [TensorFlow NLP tutorials](https://www.tensorflow.org/tutorials/text)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
