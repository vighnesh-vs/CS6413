# CS6413
CS6413 Group Project

steps to compile:
1. go to build directory (cd build)
2. run cmake command wih the following flags: cmake -DWITH_PROCPS=OFF -DWITH_SUPERCOP=OFF -DOPENSSL_ROOT_DIR="/opt/homebrew/opt/openssl@3" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_ASM=OFF -DCURVE=ALT_BN128 ..
3. then run 'make' from inside build directory
4. if successful, the executable can be run as './src/main' (from within build directory)


# ZKMLP — ZK-Friendly Image Classifier

A lightweight, **Zero-Knowledge Proof-compatible** MLP trained on MNIST. Designed for verifiable inference via zkSNARK circuits — every operation is polynomial, making the model directly arithmetizable in R1CS-based proof systems.

---

## Why ZK-Friendly?

Standard neural networks use ReLU activations, which are piecewise-linear and require expensive lookup-table gadgets inside arithmetic circuits. This model replaces ReLU with a **squared activation** (σ(x) = x²) — a degree-2 polynomial that maps to a single multiplication gate in R1CS, keeping the constraint count tractable.

```
Input (28×28)
    │
flatten → fc1 (784→128) → x² activation → fc2 (128→10) → logits
```

No softmax at inference — argmax over raw logits is equivalent and simpler to constrain in-circuit.

---

## Model Architecture

```python
class SquareActivation(nn.Module):
    def forward(self, x):
        return x * x

class ZKMLP(nn.Module):
    def __init__(self):
        super(ZKMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28*28, 128)   # 784 → 128
        self.act     = SquareActivation()
        self.fc2     = nn.Linear(128, 10)       # 128 → 10

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x   # raw logits — no softmax
```

| Parameter | Value |
|---|---|
| Input size | 28 × 28 = 784 |
| Hidden units | 128 |
| Output classes | 10 (digits 0–9) |
| Activation | Square (x²) |
| Total parameters | 101,770 |

---

## Training

| Hyperparameter | Value |
|---|---|
| Dataset | MNIST |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 7 |
| Loss | Cross-Entropy |
| **Test accuracy** | **~97%** |

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib
```

### Download MNIST and Train

```bash
# Clone the repo
git clone https://github.com/<your-username>/zkmlp.git
cd zkmlp

# Edit img_classifier.py — set download=True for first run
# train_dataset = datasets.MNIST(..., download=True, ...)
# test_dataset  = datasets.MNIST(..., download=True, ...)

python img_classifier.py
```

Training takes ~2–3 minutes on CPU, under 1 minute on GPU.

### Expected Output

```
Epoch 1, Loss: 142.3821
Epoch 2, Loss: 89.4102
...
Epoch 7, Loss: 41.2287
Test Accuracy: 97.12%
```

---

## Exporting to ONNX (for circuit generation)

To export the trained model for use with zkSNARK toolchains (e.g. [`ezkl`](https://github.com/zkonduit/ezkl)):

Uncomment the relevant block at the bottom of `img_classifier.py`:

```python
real_image, _ = test_dataset[0]
real_input = real_image.unsqueeze(0).to(device)

torch.onnx.export(
    model,
    real_input,
    "zk_mlp.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
    do_constant_folding=True
)
```

This produces `zk_mlp.onnx`, which can be passed directly to `ezkl` for automatic R1CS circuit generation and proof setup.

---

## Saving Prediction Screenshots

To save the prediction plots (useful for the project report):

```python
# In show_image_prediction(), replace plt.show() with:
plt.savefig(f"prediction_{index}.png", bbox_inches='tight', dpi=150)
plt.close()
```

---

## Project Context

This model is the machine learning component of a larger **zkSNARK Proof-of-Inference** pipeline. The full system proves that a given image was classified by this exact model — without revealing the input image or model weights — using a zkSNARK proof.

```
[MNIST image] → ZKMLP → logits → ONNX → ezkl circuit → zkSNARK proof → verifier
```

| Pipeline Stage | Tool / Component |
|---|---|
| Model training | PyTorch (`img_classifier.py`) |
| Model export | ONNX (`zk_mlp.onnx`) |
| Circuit generation | `ezkl` |
| Proof backend | Groth16 / Halo2 |
| Verification | On-chain or off-chain verifier |

---

## File Structure

```
zkmlp/
├── img_classifier.py   # model definition, training, evaluation
├── zk_mlp.onnx         # exported model (generated after training)
├── data/               # MNIST data (auto-downloaded)
└── prediction_*.png    # sample output plots (generated after training)
```

---

## References

- LeCun et al. (1998) — [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- Kingma & Ba (2014) — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Gilad-Bachrach et al. (2016) — [CryptoNets](https://proceedings.mlr.press/v48/gilad-bachrach16.html) (polynomial activations for encrypted inference)
- Liu et al. (2021) — [zkCNN](https://eprint.iacr.org/2021/673) (ZK proofs for CNN inference)
- [ezkl](https://github.com/zkonduit/ezkl) — Easy ZK for neural networks

---

## License

MIT

