# Derivative-LLM: Training a Transformer to Differentiate Polynomials 🧮

[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-blue.svg)]()

A **miniature GPT** trained from scratch to perform **symbolic differentiation of polynomials** — not by implementing the power rule, but by learning the input-output mapping from examples.

> 🔗 This model is used as a **tool** by the [Derivative-Agent](https://github.com/AlexGit31/Derivative-Agent) project, where an LLM-powered agent calls it to solve math problems.

---

## 🎯 What It Does

Given a polynomial string like `3x^2+2x+1`, the model predicts its derivative: `6x+2`.

```
Input:  f=3x^2+2x+1|df=
Output: 6x+2
```

It's a **character-level transformer** trained on 50,000 synthetic `(polynomial, derivative)` pairs. The model learns to:
- Parse polynomial notation (`x^3`, `-5x`, `+7`)
- Apply the power rule implicitly through pattern recognition
- Handle coefficients, multi-term polynomials, and constant terms

---

## 🏗️ Architecture

A standard GPT-style decoder-only transformer, implemented in PyTorch:

| Component | Value |
|-----------|-------|
| Layers | 4 transformer blocks |
| Embedding dim | 128 |
| Attention heads | 4 |
| Context length | 64 tokens |
| Parameters | ~1.3M |
| Vocabulary | 16 chars (`0123456789+-^x=\|df\n`) |

```
Input tokens → Embedding + Position → 4× Transformer Block → LayerNorm → LM Head → Output tokens
                 ↑                                                              ↑
          "f=3x^2+2x+1|df="                                              "6x+2\n"
```

---

## 📊 Training

| Metric | Value |
|--------|-------|
| Dataset | 50,000 synthetic polynomials (degree 1–10) |
| Format | `f=POLYNOMIAL\|df=DERIVATIVE\n` |
| Train/Test split | Degrees 1–3 / 4–15 (out-of-distribution) |
| Optimizer | AdamW |
| Hardware | Apple M1 (MPS) |

The model generalizes to **out-of-distribution degrees** — it correctly differentiates polynomials of degree 4–15 despite only training on degrees 1–3, showing it learns the differentiation rule rather than memorizing.

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch sympy
```

### Training

Run the notebook **`LLM_Derivation.ipynb`** — it handles everything:
1. Generates the training data (50K polynomials with SymPy)
2. Defines the GPT architecture
3. Trains the model
4. Evaluates on held-out test polynomials

### Using the Pre-Trained Weights

```python
import torch
from math_model import GPTLanguageModel, encode, decode, device

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load("math_gpt_weights.pth", map_location=device))
model.eval()

# Derive a polynomial
poly = "f=3x^2+2x+1|df="
context = torch.tensor(encode(poly), device=device).unsqueeze(0)
output = model.generate(context, max_new_tokens=20)
print(decode(output[0].tolist()))
# → "f=3x^2+2x+1|df=6x+2\n"
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `LLM_Derivation.ipynb` | Complete notebook: data generation, training, evaluation |
| `math_model.py` | GPT model architecture definition |
| `math_gpt_weights.pth` | Pre-trained weights (3.5 MB) |
| `math_train.txt` | Training data (50K polynomial-derivative pairs) |
| `math_test_ood.txt` | Test data (out-of-distribution degrees) |

---

## 🤔 Why This Exists

In early 2024, when LLM-powered agents were just emerging, this was an experiment to answer: *"Can a tiny transformer learn a mathematical operation from examples alone?"* The answer is **yes** — even a 1.3M-parameter model generalizes the power rule to unseen polynomial degrees.

This project later evolved into the **[Derivative-Agent](https://github.com/AlexGit31/Derivative-Agent)** , where a larger LLM (Microsoft Phi) acts as a reasoning agent that decides when to call this model as a tool.

---

*Built as an educational experiment in LLM training and AI agents. Not production code — just pure curiosity.*
