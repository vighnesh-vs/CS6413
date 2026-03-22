# CS6413 — zkSNARK Proof-of-Inference for MNIST MLP

## Overview

python -m pip install onnx
python -m pip install torchvision


This project proves the correctness of an MNIST image classification model using zero-knowledge proofs (zkSNARKs). A prover can demonstrate that a specific image was classified correctly by a specific model — without revealing the raw image pixels or the model's internal weights.

The system works in two phases:

```
Phase 1 (Python)                              Phase 2 (C++ / libsnark)
┌──────────────┐    ┌───────────────────┐     ┌──────────────────┐    ┌────────────┐
│  your_model  │───>│ onnx_to_circuit_  │────>│ onnx_circuit_    │───>│  zkSNARK   │
│  .onnx       │    │ params.py         │     │ builder.hpp      │    │  Proof +   │
│              │    │                   │     │ + main.cpp       │    │  Verify    │
└──────────────┘    └───────────────────┘     └──────────────────┘    └────────────┘
  (existing)         Parse graph, quantize      Build R1CS circuit,     Trusted setup,
                     → circuit_params.json      assign witness          prove, verify
```

**What is private (hidden by the proof):** the input image pixels and the model weights.

**What is public (revealed to the verifier):** the predicted class ID (0–9).


## Repository Structure

```
CS6413/
├── depends/                        # libsnark + libff (git submodules)
├── src/
│   ├── main.cpp                    # Entry point: loads JSON, builds circuit, runs zkSNARK
│   ├── onnx_circuit_builder.hpp    # R1CS circuit builder for MLP inference
│   └── json_parser.hpp             # Minimal JSON parser (no external dependency)
├── build/                          # CMake build directory
├── onnx_to_circuit_params.py       # Python: ONNX → quantized JSON converter
├── CMakeLists.txt                  # Build configuration
└── README.md                       # This file
```


## Prerequisites

### Python (for ONNX conversion)

```bash
pip install onnx numpy torchvision
```

`torchvision` is needed only to bundle a sample MNIST test image into the JSON for testing. `onnx` and `numpy` are the core dependencies.

### C++ (for zkSNARK circuit)

- A C++11 compatible compiler (GCC or Clang)
- CMake >= 3.5
- GMP library (GNU Multiple Precision Arithmetic)
- OpenSSL development headers
- libsnark and libff in the `depends/` directory (already set up as git submodules)

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential cmake libgmp-dev libssl-dev
```

On macOS (Homebrew):
```bash
brew install cmake gmp openssl@3
```


## Step-by-Step Usage

### Step 1 — Convert your ONNX model to circuit parameters

Run the Python converter against your existing `.onnx` file:

```bash
python onnx_to_circuit_params.py --onnx mnist_mlp.onnx --output circuit_params.json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--onnx` | *(required)* | Path to your trained ONNX model file |
| `--scale` | `1000` | Fixed-point quantization scale factor |
| `--output` | `circuit_params.json` | Output path for the JSON file |

**Expected output:**
```
[*] Parsing ONNX model: mnist_mlp.onnx
[✓] Parsed 3 layer(s) from ONNX graph:
    Layer 0: Linear  weight=(128, 784)  bias=yes
    Layer 1: RELU
    Layer 2: Linear  weight=(10, 128)   bias=yes

[*] Quantizing with scale=1000 ...
    linear_0: weight range [-312, 298], bias range [-187, 203]
    linear_2: weight range [-541, 487], bias range [-102, 156]

[*] Exporting sample MNIST test image...

[✓] Circuit parameters written to: circuit_params.json
    Scale factor : 1000
    Layers       : 3
    Sample label : 7
```

### Step 2 — Build the C++ project

Copy `circuit_params.json` into the `build/` directory, then compile:

```bash
cd build
cmake -DWITH_PROCPS=OFF -DWITH_SUPERCOP=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DUSE_ASM=OFF -DCURVE=ALT_BN128 ..
make
```

### Step 3 — Run the proof-of-inference

```bash
./src/main ../circuit_params.json
```

**Expected output:**
```
============================================
  zkSNARK Proof-of-Inference for MNIST MLP
============================================
Circuit params: ../circuit_params.json

[Step 1] Loading circuit parameters...
[load] Loaded 3 layers, scale=1000, sample_label=7

[Step 2] Building R1CS circuit...
[alloc] Total protoboard variables: ~5000
  [linear] Layer 0: 128 neurons, 128 constraints added
  [relu]   Layer 1: 128 neurons, ~4480 constraints added
  [linear] Layer 2: 10 neurons, 10 constraints added
[constraints] Total R1CS constraints: ~4700

[Step 3] Generating witness (forward pass)...
  Witness generation time: 0.002 seconds
  True label of sample: 7
  Predicted class: 7

[Step 4] Running zkSNARK pipeline...
[1/3] Running trusted setup (key generation)...
[2/3] Generating proof...
[3/3] Verifying proof...

Verification result: PASS ✓

============= SUMMARY =============
Model:            MLP (784 → 128 → 10)
Scale factor:     1000
R1CS variables:   ~5000
R1CS constraints: ~4700
-----------------------------------
True label:       7
Predicted class:  7
Match:            YES
-----------------------------------
Proof verified:   YES ✓
=====================================
```


## How It Works

### Fixed-point quantization (`onnx_to_circuit_params.py`)

zkSNARKs operate over finite fields (large integers), not floating-point numbers. The Python script bridges this gap by scaling every weight, bias, and pixel value to an integer:

```
quantized_value = round(float_value × SCALE)
```

With `SCALE = 1000`, a weight of `0.1347` becomes `135`. This preserves roughly 3 decimal digits of precision, which is sufficient for MNIST classification accuracy.

### R1CS circuit construction (`onnx_circuit_builder.hpp`)

The C++ circuit builder reads the JSON and translates each MLP layer into R1CS constraints:

**Linear layers** — For each output neuron `j` of `y = Wx + b`, one constraint:
```
SCALE × y[j]  =  Σᵢ(W[j][i] × x[i])  +  b[j] × SCALE
```
The weights are baked into the constraint system as constants. Total: one constraint per output neuron.

**ReLU layers** — For each neuron computing `y = max(0, x)`:
1. A boolean sign bit `is_pos ∈ {0, 1}` (1 constraint)
2. Conditional output `y = is_pos × x` (1 constraint)
3. A 32-bit decomposition of `(x + 2³¹)` to prove the sign bit is honest (34 constraints: 32 bit-boolean checks + 1 sum check + 1 sign-bit equality)

Total per ReLU neuron: ~36 constraints.

**Constraint budget for the default architecture (784 → 128 → 10):**
- Linear layer 1: 128 constraints
- ReLU: 128 × 36 = ~4,608 constraints
- Linear layer 2: 10 constraints
- Output equality: 10 constraints
- **Total: ~4,756 constraints**

### zkSNARK pipeline (`main.cpp`)

1. **Trusted setup** — Generates a proving key (PK) and verifying key (VK) from the circuit. This is a one-time operation per circuit.
2. **Witness generation** — Runs the MLP forward pass with real pixel values, assigning every intermediate variable.
3. **Proof generation** — The prover uses PK + witness to produce a succinct proof.
4. **Verification** — The verifier uses VK + public inputs (the output class) to confirm the proof. This is fast and does not require the private data.


## File Descriptions

### `onnx_to_circuit_params.py`

The Python converter. It does three things: parses the ONNX computation graph to extract `Gemm`/`MatMul`/`Add`/`Relu` nodes along with their weight tensors, quantizes all floating-point parameters to scaled integers, and bundles a sample MNIST test image for witness testing. Output is a single `circuit_params.json`.

### `onnx_circuit_builder.hpp`

The core C++ header. Contains `MLPInferenceCircuit`, which allocates libsnark protoboard variables, generates R1CS constraints for linear and ReLU layers, performs witness assignment via forward-pass computation, and exposes `run_zksnark()` for the full prove/verify pipeline.

### `json_parser.hpp`

A minimal, self-contained JSON parser sufficient for reading `circuit_params.json`. Supports objects, arrays, strings, integers, and floats. No external dependency needed. For larger projects, this can be replaced with nlohmann/json.

### `main.cpp`

Entry point. Initializes the elliptic curve parameters, loads `circuit_params.json`, builds the circuit, generates the witness, determines the predicted class via argmax, runs the zkSNARK pipeline, and prints a timing/accuracy summary.

### `CMakeLists.txt`

Build configuration. Links against libsnark from the `depends/` directory. Uses ALT_BN128 curve by default, with procps disabled for portability.


## Tuning Guide

### If proof generation is too slow or runs out of memory

- Reduce hidden layer size from 128 to 64 when training your model. This cuts the ReLU constraint count in half (~2,300 fewer constraints).
- Reduce the scale factor: `--scale 100` (less precision, but values fit in fewer bits).

### If classification accuracy drops after quantization

- Increase the scale factor: `--scale 10000` for ~4 digits of precision.
- Inspect `circuit_params.json` to verify weight ranges look reasonable (no overflow to very large integers).

### If the witness fails to satisfy (`pb.is_satisfied() == false`)

This usually means a mismatch between constraint generation and witness assignment. Common causes:
- The ONNX export changed the layer order or transposed weights differently than expected. Check the `[✓] Parsed N layer(s)` output to verify the graph was read correctly.
- Signed-value interpretation in the finite field is off. The `as_ulong()` call in `assign_relu_witness` interprets field elements as unsigned; very large values (close to the field modulus) represent negative numbers. The threshold `(1ULL << 62)` may need adjustment depending on your value range.
- The scale factor is too large, causing intermediate dot-product accumulations to overflow `int64`. Keep `SCALE × max_weight × input_dim` within the `int64` range (~9.2 × 10¹⁸).


## References

- Liu et al., "zkCNN: Zero Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy", CCS 2021
- Modulus Labs, "Scaling up Trustless DNN Inference with Zero-Knowledge Proofs", 2023
- Hao et al., "Scalable Zero-knowledge Proofs for Non-linear Functions in Neural Networks", USENIX Security 2024
- South et al., "Verifiable Evaluations of Machine Learning Models using zkSNARKs", MIT/Microsoft Research, 2024
- Wang et al., "Zero-Knowledge Proof Based Verifiable Inference of Models", November 2025
