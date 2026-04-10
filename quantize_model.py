#!/usr/bin/env python3
"""
===========================================================================
 Phase 1 — Python: ONNX → Quantized JSON → (feed to C++ R1CS builder)
===========================================================================

Pipeline:
    Train model  →  Serialize graph (ONNX)  →  Quantize to int  →  JSON

Reads the trained ONNX model (zk_mlp.onnx + zk_mlp.onnx.data),
parses the computation graph (Gemm, Mul/Pow for x²), extracts weight
matrices and bias vectors, then performs fixed-point quantization.

Quantization:
    Scale all floating-point values to integers:  round(value × 1000)
    This bridges the gap between neural net floats and finite field
    arithmetic used by the zkSNARK circuit.

R1CS constraint formulas:
    Linear layer:    SCALE × y[j] = Σᵢ( W[j][i] × x[i] ) + b[j] × SCALE
    Square activation:     x[j] × x[j]  =  y[j] × SCALE

    In practice, intermediate values accumulate scale factors:
        z1  at scale S²     (W_q @ x_q + b_q × S)
        a1  at scale S⁴     (z1 × z1)
        out at scale S⁵     (W2_q @ a1 + b2_q × S⁴)
    All values fit comfortably in the BN128 prime field (~2²⁵⁴).

Output: model_data/circuit.json — consumed by Phase 2 C++ (R1CS builder).
"""

import os
import sys
import json
import struct
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("ERROR: onnx package required.  pip install onnx")
    sys.exit(1)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SCALE       = 1000          # Fixed-point scale factor
ONNX_MODEL  = "zk_mlp.onnx"
OUTPUT_DIR  = "model_data"

INPUT_SIZE  = 784           # 28 × 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10


# ═════════════════════════════════════════════
# Step 1: Parse ONNX graph 
# ═════════════════════════════════════════════

def parse_onnx_graph(model_path):
    """
    Walk the computation graph and extract each layer's operation type
    (Gemm, MatMul, Mul/Pow for x²), weight matrices, and bias vectors.
    """
    model = onnx.load(model_path)
    graph = model.graph

    # Extract initializers (weight tensors)
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    # Walk nodes to identify layers
    layers = []
    for node in graph.node:
        layer_info = {
            "op_type": node.op_type,
            "inputs":  list(node.input),
            "outputs": list(node.output),
        }
        # Extract weight/bias shapes for Gemm nodes
        if node.op_type == "Gemm":
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            layer_info["weight_name"]  = w_name
            layer_info["weight_shape"] = list(initializers[w_name].shape)
            if b_name and b_name in initializers:
                layer_info["bias_name"]  = b_name
                layer_info["bias_shape"] = list(initializers[b_name].shape)
        elif node.op_type == "Mul":
            layer_info["description"] = "Square activation (x * x)"
        layers.append(layer_info)

    return initializers, layers, model


# ═════════════════════════════════════════════
# Step 2: Fixed-point quantization
# ═════════════════════════════════════════════

def quantize_weights(inits, scale):
    """
    Scale all floating-point values to integers: round(value × 1000).
    
    Weights and biases are both quantized at scale S = round(val × S).
    In the R1CS circuit, biases are multiplied by appropriate scale
    powers to match the accumulated scale at each layer.
    """
    W1 = inits['fc1.weight']   # (128, 784)
    b1 = inits['fc1.bias']     # (128,)
    W2 = inits['fc2.weight']   # (10, 128)
    b2 = inits['fc2.bias']     # (10,)

    S = scale
    W1_q = np.round(W1 * S).astype(np.int64)   # at scale S
    b1_q = np.round(b1 * S).astype(np.int64)    # at scale S
    W2_q = np.round(W2 * S).astype(np.int64)    # at scale S
    b2_q = np.round(b2 * S).astype(np.int64)    # at scale S

    return W1_q, b1_q, W2_q, b2_q


# ═══════════════════════════════════════════════
# Inference (quantized, for witness generation)
# ═══════════════════════════════════════════════

def quantized_inference(W1_q, b1_q, W2_q, b2_q, x_float, scale):
    """
    Run MLP inference in quantized integers to produce the witness
    (all intermediate values the prover needs).

    Scale accumulation per layer:
        x_q           at scale S
        z1 = W1·x+b·S at scale S²   
        a1 = z1²       at scale S⁴ 
        out= W2·a1+b·S⁴ at scale S⁵  
    """
    S = scale
    x_q = np.round(x_float * S).astype(np.int64)

    # FC1:  z1 = W1_q @ x_q + b1_q * S          
    z1_q = W1_q @ x_q + b1_q * S                 # at scale S²

    # Square activation:  a1 = z1 * z1          
    a1_q = z1_q * z1_q                            # at scale S⁴

    # FC2:  out = W2_q @ a1 + b2_q * S⁴    
    out_q = W2_q @ a1_q + b2_q * (S ** 4)        # at scale S⁵

    predicted = int(np.argmax(out_q))
    return x_q, z1_q, a1_q, out_q, predicted


def float_inference(inits, x_float):
    """Reference float-point inference for validation."""
    W1, b1 = inits['fc1.weight'], inits['fc1.bias']
    W2, b2 = inits['fc2.weight'], inits['fc2.bias']
    z1  = W1 @ x_float + b1
    a1  = z1 * z1
    out = W2 @ a1 + b2
    return int(np.argmax(out)), out


# ═══════════════════════════════════════════════
# Step 3: Export as JSON 
# ═══════════════════════════════════════════════

def export_circuit_json(W1_q, b1_q, W2_q, b2_q,
                        x_q, z1_q, a1_q, out_q,
                        predicted, true_label,
                        scale, graph_layers, output_dir):
    """
    Export quantized model and witness as JSON for the C++ R1CS builder.
    Format: ONNX → JSON → R1CS circuit.
    """
    os.makedirs(output_dir, exist_ok=True)

    circuit_data = {
        "metadata": {
            "description": "Quantized MLP for zkSNARK proof-of-inference",
            "pipeline":    "ONNX -> JSON -> R1CS circuit",
            "scale":       scale,
            "input_size":  INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
        },
        "graph": graph_layers,
        "quantized_weights": {
            "W1": W1_q.tolist(),        # (128, 784) at scale S
            "b1": b1_q.tolist(),        # (128,)     at scale S
            "W2": W2_q.tolist(),        # (10, 128)  at scale S
            "b2": b2_q.tolist(),        # (10,)      at scale S
        },
        "test_witness": {
            "true_label":  true_label,
            "predicted":   predicted,
            "input":       x_q.tolist(),     # (784,)  at scale S
            "z1":          z1_q.tolist(),     # (128,)  at scale S²
            "a1":          a1_q.tolist(),     # (128,)  at scale S⁴
            "output":      out_q.tolist(),    # (10,)   at scale S⁵
        },
        "r1cs_info": {
            "constraints": {
                "fc1_linear":        HIDDEN_SIZE,
                "square_activation": HIDDEN_SIZE,
                "fc2_linear":        OUTPUT_SIZE,
                "output_equality":   OUTPUT_SIZE,
                "total":             HIDDEN_SIZE * 2 + OUTPUT_SIZE * 2,
            },
            "variables": {
                "public_outputs":    OUTPUT_SIZE,
                "private_input":     INPUT_SIZE,
                "private_z1":        HIDDEN_SIZE,
                "private_a1":        HIDDEN_SIZE,
                "private_fc2_out":   OUTPUT_SIZE,
                "total":             OUTPUT_SIZE + INPUT_SIZE + HIDDEN_SIZE * 2 + OUTPUT_SIZE,
            },
            "constraint_formulas": {
                "linear_layer": "SCALE * y[j] = sum_i(W[j][i] * x[i]) + b[j] * SCALE",
                "square_activation": "x[j] * x[j] = y[j] * SCALE",
                "note": "In practice, SCALE factors accumulate: z1 at S^2, a1 at S^4, out at S^5"
            }
        }
    }

    filepath = os.path.join(output_dir, "circuit.json")
    with open(filepath, 'w') as f:
        json.dump(circuit_data, f, separators=(',', ':'))
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  JSON exported to: {filepath}  ({size_mb:.1f} MB)")
    return filepath


# ═══════════════════════════════════════════════
# Test utilities
# ═══════════════════════════════════════════════

# test image for number 1 
# def generate_test_image():
#     """Synthetic MNIST-like test image (digit-1 shape)."""
#     img = np.zeros((28, 28), dtype=np.float32)
#     img[4:24, 13:16] = 1.0
#     img[5:7, 11:14] = 0.8
#     img[23:25, 11:18] = 0.9
#     return img.flatten()

# test image for number 7
def generate_test_image():
    """Synthetic MNIST-like test image (digit-7 shape)."""
    img = np.zeros((28, 28), dtype=np.float32)
    # Top horizontal bar
    img[5:8, 7:21] = 1.0
    # Diagonal stroke going down-left
    for row in range(8, 25):
        col = int(20 - (row - 8) * 0.65)
        img[row, max(0, col-1):col+2] = 1.0
    return img.flatten()


def try_load_mnist_sample(data_dir="data"):
    """Try to load a real MNIST test sample."""
    try:
        img_path = os.path.join(data_dir, "MNIST", "raw", "t10k-images-idx3-ubyte")
        lbl_path = os.path.join(data_dir, "MNIST", "raw", "t10k-labels-idx1-ubyte")
        with open(img_path, 'rb') as f:
            _, num, _, _ = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784)
        with open(lbl_path, 'rb') as f:
            _, _ = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return images[0].astype(np.float32) / 255.0, int(labels[0])
    except FileNotFoundError:
        return None, None


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Phase 1 — Python: ONNX → Quantized JSON")
    print("=" * 60)

    # ── Step 1: Parse ONNX graph ──
    print(f"\n[Step 1] Parsing ONNX graph from '{ONNX_MODEL}' ...")
    if not os.path.exists(ONNX_MODEL):
        print(f"ERROR: '{ONNX_MODEL}' not found.")
        print(f"       Place zk_mlp.onnx and zk_mlp.onnx.data in this directory.")
        sys.exit(1)

    inits, graph_layers, model = parse_onnx_graph(ONNX_MODEL)

    print(f"  Graph nodes:")
    for layer in graph_layers:
        desc = layer.get("description", layer["op_type"])
        print(f"    {desc}  {layer['inputs']} → {layer['outputs']}")
    print(f"  fc1.weight: {inits['fc1.weight'].shape}")
    print(f"  fc1.bias:   {inits['fc1.bias'].shape}")
    print(f"  fc2.weight: {inits['fc2.weight'].shape}")
    print(f"  fc2.bias:   {inits['fc2.bias'].shape}")

    # ── Step 2: Fixed-point quantization ──
    print(f"\n[Step 2] Fixed-point quantization: round(value × {SCALE}) ...")
    W1_q, b1_q, W2_q, b2_q = quantize_weights(inits, SCALE)
    print(f"  W1_q range: [{W1_q.min()}, {W1_q.max()}]  (scale S)")
    print(f"  b1_q range: [{b1_q.min()}, {b1_q.max()}]  (scale S)")
    print(f"  W2_q range: [{W2_q.min()}, {W2_q.max()}]  (scale S)")
    print(f"  b2_q range: [{b2_q.min()}, {b2_q.max()}]  (scale S)")

    # ── Prepare test input ──
    print(f"\n[Step 3] Preparing test input ...")
    x_float, true_label = try_load_mnist_sample()
    if x_float is not None:
        print(f"  Loaded MNIST test sample (true label = {true_label})")
    else:
        print(f"  MNIST data not found — using synthetic test image")
        x_float = generate_test_image()
        true_label = -1

    # ── Validate quantized inference ──
    print(f"\n[Step 4] Validating quantized inference ...")
    float_pred, _ = float_inference(inits, x_float)
    x_q, z1_q, a1_q, out_q, quant_pred = quantized_inference(
        W1_q, b1_q, W2_q, b2_q, x_float, SCALE)

    print(f"  Float  prediction: {float_pred}")
    print(f"  Quant  prediction: {quant_pred}")
    if true_label >= 0:
        print(f"  True label:        {true_label}")
    match = float_pred == quant_pred
    print(f"  Predictions match: {'YES' if match else 'NO'}")
    if not match:
        print("  WARNING: quantization changed the prediction!")

    print(f"\n  Intermediate value ranges:")
    print(f"    x_q:   [{x_q.min()}, {x_q.max()}]          (scale S)")
    print(f"    z1_q:  [{z1_q.min()}, {z1_q.max()}]    (scale S²)")
    print(f"    a1_q:  [{a1_q.min()}, {a1_q.max()}]  (scale S⁴)")
    print(f"    out_q: [{out_q.min()}, {out_q.max()}]  (scale S⁵)")

    # ── Export to JSON ──
    print(f"\n[Step 5] Exporting quantized model → JSON ...")
    export_circuit_json(W1_q, b1_q, W2_q, b2_q,
                        x_q, z1_q, a1_q, out_q,
                        quant_pred, true_label,
                        SCALE, graph_layers, OUTPUT_DIR)

    # ── Batch accuracy test ──
    print(f"\n[Step 6] Batch accuracy validation ...")
    try:
        img_path = os.path.join("data", "MNIST", "raw", "t10k-images-idx3-ubyte")
        lbl_path = os.path.join("data", "MNIST", "raw", "t10k-labels-idx1-ubyte")
        with open(img_path, 'rb') as f:
            _, num, _, _ = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784)
        with open(lbl_path, 'rb') as f:
            _, _ = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        n_test = min(1000, num)
        float_correct = quant_correct = mismatches = 0
        for idx in range(n_test):
            xf = images[idx].astype(np.float32) / 255.0
            fp, _ = float_inference(inits, xf)
            _, _, _, _, qp = quantized_inference(
                W1_q, b1_q, W2_q, b2_q, xf, SCALE)
            float_correct += (fp == labels[idx])
            quant_correct += (qp == labels[idx])
            mismatches    += (fp != qp)
        print(f"  Tested {n_test} samples:")
        print(f"    Float accuracy:      {100.0*float_correct/n_test:.2f}%")
        print(f"    Quantized accuracy:  {100.0*quant_correct/n_test:.2f}%")
        print(f"    Prediction mismatches: {mismatches}/{n_test}")
    except FileNotFoundError:
        print("  MNIST test data not available — skipping batch test.")

    # ── Summary ──
    total_c = HIDDEN_SIZE * 2 + OUTPUT_SIZE * 2
    total_v = OUTPUT_SIZE + INPUT_SIZE + HIDDEN_SIZE * 2 + OUTPUT_SIZE
    print(f"\n{'='*60}")
    print(f"  Phase 1 complete!")
    print(f"  Output:        {OUTPUT_DIR}/circuit.json")
    print(f"  Scale factor:  {SCALE}")
    print(f"  Constraints:   ~{total_c}  (128 + 128 + 10 + 10)")
    print(f"  Variables:     ~{total_v}  ({OUTPUT_SIZE} public + {total_v-OUTPUT_SIZE} private)")
    print(f"\n  Next → Phase 2: C++ (circuit.json → R1CS → Prove → Verify)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
