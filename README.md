# Zero-Knowledge Proof for AI Model Validation

The project is split into two phases:

## Prerequisites

**Python (Phase 1):**
- Python 3.8+
- `pip install onnx numpy`
- (Optional) `pip install onnxruntime` for ONNX runtime validation

**C++ (Phase 2):**
- CMake 3.5+
- GCC or Clang with C++11 support
- GMP (`libgmp-dev`)
- Boost (`libboost-all-dev`)
- OpenSSL (`libssl-dev`)
- (Optional) libprocps (`libprocps-dev`) for memory profiling

### macOS (Homebrew)

```bash
brew install cmake gmp boost openssl@3
```

## Build & Run

### 1. Clone and initialize submodules

```bash
git clone https://github.com/vighnesh-vs/CS6413.git
cd CS6413
git checkout Khurana7
git submodule update --init --recursive
```

### 2. Run Phase 1 — Quantize model (Python)

Make sure `zk_mlp.onnx` and `zk_mlp.onnx.data` are in the repository root, then:

```bash
python3 quantize_model.py
```

This parses the ONNX graph, quantizes weights with `round(value × 1000)`, and writes `model_data/circuit.json`.

### 3. Build Phase 2 (C++)

**macOS:**

```bash
mkdir -p build && cd build
cmake -DWITH_PROCPS=OFF -DWITH_SUPERCOP=OFF \
      -DOPENSSL_ROOT_DIR="/opt/homebrew/opt/openssl@3" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DUSE_ASM=OFF -DCURVE=ALT_BN128 ..
make -j$(sysctl -n hw.ncpu)
```

or USE THIS if the above doesn't work

```bash
rm -rf build
mkdir build
cd build
cmake -DWITH_PROCPS=OFF -DWITH_SUPERCOP=OFF \
      -DOPENSSL_ROOT_DIR="/opt/homebrew/opt/openssl@3" \
      -DOPENSSL_CRYPTO_LIBRARY="/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib" \
      -DOPENSSL_INCLUDE_DIR="/opt/homebrew/opt/openssl@3/include" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DUSE_ASM=OFF -DCURVE=ALT_BN128 \
      -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/openssl@3/lib" ..
make -j$(sysctl -n hw.ncpu)
```


### 4. Run Phase 2 — Prove & Verify

From the repository root:
come to main project directory

```bash
cd ..
./build/src/main
```

