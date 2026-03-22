#!/usr/bin/env python3
"""
onnx_to_circuit_params.py
=========================
Takes an existing ONNX file (.onnx) of your MNIST MLP and converts it
into the quantized JSON format that the C++ libsnark circuit builder reads.

Pipeline:
  your_model.onnx  →  parse graph  →  quantize to fixed-point  →  circuit_params.json

Usage:
  python onnx_to_circuit_params.py --onnx mnist_mlp.onnx --output circuit_params.json
  python onnx_to_circuit_params.py --onnx mnist_mlp.onnx --scale 1000 --output circuit_params.json
"""

import argparse
import json
import numpy as np
import onnx
from onnx import numpy_helper


# ---------------------------------------------------------------------------
# 1. Parse ONNX graph — extract layer topology, weights, and biases
# ---------------------------------------------------------------------------
def parse_onnx(onnx_path: str):
    """
    Walk the ONNX graph and extract:
      - Layer ordering and types (Gemm/MatMul+Add, Relu)
      - Weight matrices and bias vectors as numpy arrays

    Returns a list of layer dicts.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    graph = model.graph

    # Build lookup: initializer name → numpy array
    initializers = {}
    for tensor in graph.initializer:
        initializers[tensor.name] = numpy_helper.to_array(tensor)

    layers = []
    for node in graph.node:
        op = node.op_type

        if op in ("Gemm", "MatMul"):
            # Linear layer: y = x @ W^T + b
            weight_name = node.input[1]
            W = initializers[weight_name]

            # Check for bias (Gemm bundles it as input[2])
            bias = None
            if len(node.input) >= 3 and node.input[2] in initializers:
                bias = initializers[node.input[2]]

            # Gemm attributes: check for transB
            transB = False
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1:
                    transB = True

            if transB:
                W = W.T  # Circuit expects W in [out_features, in_features] order

            layers.append({
                "type": "linear",
                "weight": W,
                "bias": bias,
                "name": node.name or f"linear_{len(layers)}",
            })

        elif op == "Relu":
            layers.append({
                "type": "relu",
                "name": node.name or f"relu_{len(layers)}",
            })

        elif op == "Add":
            # Standalone Add following a MatMul (bias addition)
            # Fold it into the previous linear layer if bias is missing
            if layers and layers[-1]["type"] == "linear" and layers[-1]["bias"] is None:
                for inp in node.input:
                    if inp in initializers:
                        layers[-1]["bias"] = initializers[inp]
                        break

        # Reshape / Flatten nodes are structural — the circuit operates
        # on flat vectors, so we skip them.

    print(f"[✓] Parsed {len(layers)} layer(s) from ONNX graph:")
    for i, l in enumerate(layers):
        if l["type"] == "linear":
            w_shape = l["weight"].shape
            print(f"    Layer {i}: Linear  weight={w_shape}  "
                  f"bias={'yes' if l['bias'] is not None else 'no'}")
        else:
            print(f"    Layer {i}: {l['type'].upper()}")

    return layers


# ---------------------------------------------------------------------------
# 2. Fixed-point quantization
# ---------------------------------------------------------------------------
def quantize_to_fixed_point(layers, scale: int = 1000):
    """
    Convert every float weight/bias to a scaled integer.

    With SCALE = 1000:
      quantized_w = round(w * 1000)

    After matmul in the circuit, the result is in SCALE^2 domain,
    so the constraint divides by SCALE once to bring it back.

    Returns a JSON-serializable list of layer descriptors.
    """
    quantized_layers = []

    for layer in layers:
        if layer["type"] == "linear":
            W = layer["weight"]
            b = layer["bias"]

            W_int = np.round(W * scale).astype(np.int64)
            b_int = np.round(b * scale).astype(np.int64) if b is not None else None

            out_features, in_features = W_int.shape

            quantized_layers.append({
                "type": "linear",
                "name": layer["name"],
                "in_features": int(in_features),
                "out_features": int(out_features),
                "weights": W_int.flatten().tolist(),   # row-major
                "biases": b_int.tolist() if b_int is not None else [0] * out_features,
            })

            print(f"    {layer['name']}: weight range [{W_int.min()}, {W_int.max()}], "
                  f"bias range [{b_int.min() if b_int is not None else 0}, "
                  f"{b_int.max() if b_int is not None else 0}]")

        elif layer["type"] == "relu":
            quantized_layers.append({
                "type": "relu",
                "name": layer["name"],
            })

    return quantized_layers


# ---------------------------------------------------------------------------
# 3. Export a sample MNIST test image for witness generation / testing
# ---------------------------------------------------------------------------
def export_sample_input(scale: int = 1000):
    """
    Export a single MNIST test image as a quantized flat pixel vector.
    This gets bundled into the JSON so the C++ side can immediately
    test the full prove/verify pipeline.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    image, label = test_ds[0]
    flat = image.view(-1).numpy()
    quantized = np.round(flat * scale).astype(np.int64)
    return {
        "pixels": quantized.tolist(),
        "true_label": int(label),
        "num_pixels": len(quantized.tolist()),
    }


# ---------------------------------------------------------------------------
# 4. Write the JSON consumed by the C++ circuit builder
# ---------------------------------------------------------------------------
def write_circuit_json(quantized_layers, sample_input, scale, output_path):
    """
    Produce a single JSON file with everything the C++ side needs:
      - scale factor
      - layer descriptors (type, dimensions, quantized weights/biases)
      - a sample input image for witness generation / testing
    """
    payload = {
        "scale": scale,
        "num_layers": len(quantized_layers),
        "layers": quantized_layers,
        "sample_input": sample_input,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] Circuit parameters written to: {output_path}")
    print(f"    Scale factor : {scale}")
    print(f"    Layers       : {len(quantized_layers)}")
    print(f"    Sample label : {sample_input['true_label']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert an ONNX MLP into quantized zkSNARK circuit parameters"
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to your existing ONNX model file (.onnx)")
    parser.add_argument("--scale", type=int, default=1000,
                        help="Fixed-point scale factor (default: 1000)")
    parser.add_argument("--output", type=str, default="circuit_params.json",
                        help="Output JSON for C++ circuit builder")
    args = parser.parse_args()

    # ---- Stage 1: Parse ONNX graph ----
    print(f"[*] Parsing ONNX model: {args.onnx}")
    layers = parse_onnx(args.onnx)

    # ---- Stage 2: Quantize to fixed-point ----
    print(f"\n[*] Quantizing with scale={args.scale} ...")
    quantized = quantize_to_fixed_point(layers, scale=args.scale)

    # ---- Stage 3: Export sample input ----
    print("\n[*] Exporting sample MNIST test image...")
    sample = export_sample_input(scale=args.scale)

    # ---- Write JSON ----
    write_circuit_json(quantized, sample, args.scale, args.output)


if __name__ == "__main__":
    main()
