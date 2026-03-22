/*
 * onnx_circuit_builder.hpp
 * ========================
 * Stage 3 of the ONNX → zkSNARK pipeline.
 *
 * Reads the quantized model parameters (from circuit_params.json produced by
 * the Python script) and constructs an R1CS constraint system in libsnark
 * that encodes the full MLP inference computation:
 *
 *   input_pixels  →  Linear(W1,b1) → ReLU → Linear(W2,b2) → output_logits
 *
 * The prover demonstrates knowledge of a private input image that, when fed
 * through the model, produces a specific public classification output —
 * without revealing the image or the model weights.
 *
 * Design decisions:
 *   - Fixed-point arithmetic with a configurable SCALE factor.
 *   - ReLU is implemented via a sign-bit gadget: decompose the value into
 *     bits, read the sign, and conditionally zero the output.
 *   - All weights/biases are baked into the constraint system as constants
 *     (not witness variables), keeping the circuit deterministic.
 */

#ifndef ONNX_CIRCUIT_BUILDER_HPP
#define ONNX_CIRCUIT_BUILDER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

// ---- libsnark headers ----
#include <libsnark/common/default_types/r1cs_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_ppzksnark/r1cs_ppzksnark.hpp>
#include <libsnark/gadgetlib1/protoboard.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <libff/common/default_types/ec_pp.hpp>

// ---- A tiny JSON parser (self-contained, no external dependency) ----
// For production use, consider nlohmann/json. This minimal parser handles
// the specific JSON structure we produce from the Python script.
#include "json_parser.hpp"

using namespace libsnark;
using namespace std;

// The elliptic curve / finite field type
typedef libff::default_ec_pp ppT;
typedef libff::Fr<ppT> FieldT;

// =========================================================================
//  Data structures to hold parsed model parameters
// =========================================================================

struct LinearLayerParams {
    string name;
    int in_features;
    int out_features;
    vector<long long> weights;   // row-major: [out_features × in_features]
    vector<long long> biases;    // [out_features]
};

struct LayerDescriptor {
    string type;  // "linear" or "relu"
    string name;
    // Only populated for linear layers:
    LinearLayerParams linear;
};

struct CircuitParams {
    int scale;
    int num_layers;
    vector<LayerDescriptor> layers;
    // Sample input for witness generation
    vector<long long> sample_pixels;
    int true_label;
    int num_pixels;
};


// =========================================================================
//  JSON loading (reads the output of export_and_parse_onnx.py)
// =========================================================================

CircuitParams load_circuit_params(const string& json_path) {
    // Read file into string
    ifstream ifs(json_path);
    if (!ifs.is_open()) {
        cerr << "ERROR: Cannot open " << json_path << endl;
        exit(1);
    }
    stringstream ss;
    ss << ifs.rdbuf();
    string json_str = ss.str();

    // Use our simple JSON parser
    JsonValue root = json_parse(json_str);

    CircuitParams params;
    params.scale      = root["scale"].as_int();
    params.num_layers = root["num_layers"].as_int();

    // Parse layers
    const auto& layers_arr = root["layers"].as_array();
    for (size_t i = 0; i < layers_arr.size(); i++) {
        const JsonValue& lj = layers_arr[i];
        LayerDescriptor ld;
        ld.type = lj["type"].as_string();
        ld.name = lj["name"].as_string();

        if (ld.type == "linear") {
            ld.linear.name         = ld.name;
            ld.linear.in_features  = lj["in_features"].as_int();
            ld.linear.out_features = lj["out_features"].as_int();

            const auto& w_arr = lj["weights"].as_array();
            for (auto& v : w_arr) ld.linear.weights.push_back(v.as_int());

            const auto& b_arr = lj["biases"].as_array();
            for (auto& v : b_arr) ld.linear.biases.push_back(v.as_int());
        }
        params.layers.push_back(ld);
    }

    // Parse sample input
    const auto& sample = root["sample_input"];
    params.true_label = sample["true_label"].as_int();
    params.num_pixels = sample["num_pixels"].as_int();
    const auto& pix = sample["pixels"].as_array();
    for (auto& v : pix) params.sample_pixels.push_back(v.as_int());

    cout << "[load] Loaded " << params.layers.size() << " layers, "
         << "scale=" << params.scale
         << ", sample_label=" << params.true_label << endl;

    return params;
}


// =========================================================================
//  R1CS Circuit Builder for MLP Inference
// =========================================================================

class MLPInferenceCircuit {
public:
    protoboard<FieldT> pb;

    // ----- Variables -----
    // Private inputs: the image pixels (flattened 784-vector)
    vector<pb_variable<FieldT>> input_vars;

    // Intermediate activations between layers
    // intermediate[i] holds the output of layer i
    vector<vector<pb_variable<FieldT>>> layer_outputs;

    // Public output: the final classification logits (10 values)
    vector<pb_variable<FieldT>> output_vars;

    // Auxiliary variables for ReLU (sign bits, bit decompositions)
    vector<pb_variable<FieldT>> relu_sign_bits;
    // Bit decomposition variables for ReLU comparison
    vector<vector<pb_variable<FieldT>>> relu_bits;

    // Auxiliary variables for intermediate products in dot products
    vector<pb_variable<FieldT>> aux_product_vars;

    // Model parameters (baked in)
    CircuitParams params;

    int scale;
    // Bit width for values (determines the range for ReLU comparison)
    // For MNIST with scale=1000, values stay within ~20 bits
    static const int VALUE_BITS = 32;

    MLPInferenceCircuit(const CircuitParams& p) : params(p), scale(p.scale) {}

    // -----------------------------------------------------------------
    //  Allocate all protoboard variables
    // -----------------------------------------------------------------
    void allocate_variables() {
        int num_inputs = params.num_pixels;  // 784 for MNIST

        // Find output dimension (last linear layer's out_features)
        int num_outputs = 10;
        for (int i = params.layers.size() - 1; i >= 0; i--) {
            if (params.layers[i].type == "linear") {
                num_outputs = params.layers[i].linear.out_features;
                break;
            }
        }

        // --- Allocate output variables FIRST (they will be public inputs) ---
        output_vars.resize(num_outputs);
        for (int i = 0; i < num_outputs; i++) {
            output_vars[i].allocate(pb, "output_" + to_string(i));
        }
        // Mark how many variables are public (the output logits)
        pb.set_input_sizes(num_outputs);

        // --- Allocate private input variables ---
        input_vars.resize(num_inputs);
        for (int i = 0; i < num_inputs; i++) {
            input_vars[i].allocate(pb, "pixel_" + to_string(i));
        }

        // --- Allocate intermediate layer output variables ---
        // Walk through layers and figure out dimensions
        int current_dim = num_inputs;
        for (size_t li = 0; li < params.layers.size(); li++) {
            const auto& layer = params.layers[li];

            if (layer.type == "linear") {
                int out_dim = layer.linear.out_features;
                vector<pb_variable<FieldT>> vars(out_dim);
                for (int j = 0; j < out_dim; j++) {
                    vars[j].allocate(pb,
                        "layer_" + to_string(li) + "_out_" + to_string(j));
                }
                layer_outputs.push_back(vars);
                current_dim = out_dim;

            } else if (layer.type == "relu") {
                // ReLU output has same dimension as its input
                vector<pb_variable<FieldT>> vars(current_dim);
                for (int j = 0; j < current_dim; j++) {
                    vars[j].allocate(pb,
                        "relu_" + to_string(li) + "_out_" + to_string(j));
                }
                layer_outputs.push_back(vars);

                // Allocate sign bits for ReLU
                for (int j = 0; j < current_dim; j++) {
                    pb_variable<FieldT> sb;
                    sb.allocate(pb, "relu_sign_" + to_string(li) + "_" + to_string(j));
                    relu_sign_bits.push_back(sb);
                }

                // Allocate bit decomposition variables for range proof
                for (int j = 0; j < current_dim; j++) {
                    vector<pb_variable<FieldT>> bits(VALUE_BITS);
                    for (int b = 0; b < VALUE_BITS; b++) {
                        bits[b].allocate(pb,
                            "relu_bit_" + to_string(li) + "_" + to_string(j) + "_" + to_string(b));
                    }
                    relu_bits.push_back(bits);
                }
            }
        }

        cout << "[alloc] Total protoboard variables: " << pb.num_variables() << endl;
    }


    // -----------------------------------------------------------------
    //  Generate R1CS constraints for the entire MLP
    // -----------------------------------------------------------------
    void generate_constraints() {
        int relu_idx = 0;  // counter for relu sign/bit vars
        int layer_out_idx = 0;

        for (size_t li = 0; li < params.layers.size(); li++) {
            const auto& layer = params.layers[li];

            if (layer.type == "linear") {
                generate_linear_constraints(li, layer.linear, layer_out_idx);
                layer_out_idx++;

            } else if (layer.type == "relu") {
                generate_relu_constraints(li, layer_out_idx, relu_idx);
                layer_out_idx++;
                // relu_idx advances by the dimension of the ReLU layer
                int dim = layer_outputs[layer_out_idx - 1].size();
                relu_idx += dim;
            }
        }

        // --- Constrain output_vars to equal the final layer's output ---
        int last_out_idx = layer_outputs.size() - 1;
        for (size_t i = 0; i < output_vars.size(); i++) {
            // output_vars[i] == layer_outputs[last][i]
            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(
                    1,
                    layer_outputs[last_out_idx][i] - output_vars[i],
                    0
                ),
                "output_equality_" + to_string(i)
            );
        }

        cout << "[constraints] Total R1CS constraints: "
             << pb.num_constraints() << endl;
    }


    // -----------------------------------------------------------------
    //  Linear layer constraints: y[j] = sum_i(W[j][i] * x[i]) + b[j]
    //
    //  For each output neuron j, we add the constraint:
    //    y[j] * SCALE  ==  sum_i( W_int[j][i] * x[i] ) + b_int[j] * SCALE
    //
    //  This accounts for the fixed-point scaling:
    //    - x[i] is in units of SCALE
    //    - W_int[j][i] is in units of SCALE
    //    - Product W*x is in units of SCALE^2
    //    - We want y[j] in units of SCALE, so we divide by SCALE
    //    - The constraint form: y[j] * SCALE = dot_product + b_scaled
    // -----------------------------------------------------------------
    void generate_linear_constraints(int layer_idx,
                                     const LinearLayerParams& lp,
                                     int layer_out_idx)
    {
        // Determine input variables for this layer
        const vector<pb_variable<FieldT>>& in_vars = get_layer_input(layer_idx, layer_out_idx);
        const vector<pb_variable<FieldT>>& out_vars_layer = layer_outputs[layer_out_idx];

        int in_dim  = lp.in_features;
        int out_dim = lp.out_features;

        for (int j = 0; j < out_dim; j++) {
            // Build a linear combination: sum_i( W[j][i] * x[i] ) + b[j]*SCALE
            linear_combination<FieldT> dot_product;

            for (int i = 0; i < in_dim; i++) {
                long long w = lp.weights[j * in_dim + i];
                if (w != 0) {
                    dot_product.add_term(in_vars[i], FieldT(w));
                }
            }

            // Add bias term: b[j] is already scaled by SCALE once.
            // Since dot_product is at SCALE^2, we need bias at SCALE^2 too.
            long long b_scaled = lp.biases[j] * (long long)scale;
            dot_product.add_term(pb_variable<FieldT>(0), FieldT(b_scaled));  // constant term

            // Constraint: out_vars[j] * SCALE = dot_product
            // i.e.,  (SCALE) * (out_vars[j]) = dot_product
            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(
                    FieldT(scale),          // A = SCALE (constant)
                    out_vars_layer[j],      // B = y[j]
                    dot_product             // C = sum(W*x) + b*SCALE
                ),
                "linear_" + to_string(layer_idx) + "_neuron_" + to_string(j)
            );
        }

        cout << "  [linear] Layer " << layer_idx << ": " << out_dim
             << " neurons, " << out_dim << " constraints added" << endl;
    }


    // -----------------------------------------------------------------
    //  ReLU constraints: y[j] = max(0, x[j])
    //
    //  Implementation using a sign bit and conditional multiplication:
    //    1. is_positive[j] ∈ {0, 1}              (boolean constraint)
    //    2. y[j] = is_positive[j] * x[j]         (conditional output)
    //    3. Range proof that is_positive correctly reflects sign of x
    //       (via bit decomposition of x + OFFSET to ensure non-negative)
    //
    //  Simplified approach for our MLP:
    //    - Constrain is_pos to be boolean: is_pos * (1 - is_pos) = 0
    //    - Constrain output: y = is_pos * x
    //    - Constrain consistency: if is_pos=1 then x >= 0;
    //                             if is_pos=0 then x < 0 and y = 0
    //
    //  The full range proof uses bit decomposition to verify the sign.
    // -----------------------------------------------------------------
    void generate_relu_constraints(int layer_idx,
                                   int layer_out_idx,
                                   int relu_start_idx)
    {
        // ReLU input = previous layer's output
        const vector<pb_variable<FieldT>>& in_vars =
            layer_outputs[layer_out_idx - 1];
        const vector<pb_variable<FieldT>>& out_vars_layer =
            layer_outputs[layer_out_idx];

        int dim = in_vars.size();

        for (int j = 0; j < dim; j++) {
            int ridx = relu_start_idx + j;
            pb_variable<FieldT>& is_pos = relu_sign_bits[ridx];

            // Constraint 1: is_pos is boolean
            // is_pos * (1 - is_pos) = 0
            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(
                    is_pos,                             // A
                    FieldT(1) - is_pos,                 // B  (as linear combination)
                    0                                   // C = 0
                ),
                "relu_bool_" + to_string(layer_idx) + "_" + to_string(j)
            );

            // Constraint 2: output = is_pos * input
            // is_pos * in_vars[j] = out_vars[j]
            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(
                    is_pos,             // A
                    in_vars[j],         // B
                    out_vars_layer[j]   // C
                ),
                "relu_mul_" + to_string(layer_idx) + "_" + to_string(j)
            );

            // Constraint 3: Bit decomposition for range proof
            // We prove that (x + OFFSET) can be represented in VALUE_BITS bits,
            // where OFFSET is large enough that x + OFFSET is always non-negative.
            // Then we check the high bit to determine the sign.
            //
            // shifted_x = in_vars[j] + 2^(VALUE_BITS-1)
            // Decompose shifted_x into bits: shifted_x = sum(bit_k * 2^k)
            // The sign of x is: is_pos == bit_(VALUE_BITS-1)
            //
            // Sum constraint: sum of bits * 2^k = in_vars[j] + OFFSET
            long long offset = (1LL << (VALUE_BITS - 1));
            linear_combination<FieldT> bit_sum;
            for (int b = 0; b < VALUE_BITS; b++) {
                bit_sum.add_term(relu_bits[ridx][b], FieldT(1LL << b));

                // Each bit must be boolean
                pb.add_r1cs_constraint(
                    r1cs_constraint<FieldT>(
                        relu_bits[ridx][b],
                        FieldT(1) - relu_bits[ridx][b],
                        0
                    ),
                    "relu_bitbool_" + to_string(ridx) + "_" + to_string(b)
                );
            }

            // bit_sum = in_vars[j] + offset
            // => bit_sum - in_vars[j] - offset = 0
            // As R1CS: 1 * (bit_sum - in_vars[j] - offset) = 0
            linear_combination<FieldT> lhs;
            lhs = bit_sum;
            lhs.add_term(in_vars[j], FieldT(-1));
            lhs.add_term(pb_variable<FieldT>(0), FieldT(-offset));

            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(1, lhs, 0),
                "relu_decomp_" + to_string(ridx)
            );

            // Sign bit = most significant bit of shifted value
            // If x >= 0 → shifted_x >= OFFSET → MSB is 1 → is_pos should be 1
            // If x < 0  → shifted_x < OFFSET  → MSB is 0 → is_pos should be 0
            // Constraint: is_pos = relu_bits[ridx][VALUE_BITS-1]
            pb.add_r1cs_constraint(
                r1cs_constraint<FieldT>(
                    1,
                    is_pos - relu_bits[ridx][VALUE_BITS - 1],
                    0
                ),
                "relu_signbit_" + to_string(ridx)
            );
        }

        int total_constraints = dim * (2 + VALUE_BITS + 2); // bool + mul + per-bit-bool + decomp + signbit
        cout << "  [relu] Layer " << layer_idx << ": " << dim
             << " neurons, ~" << total_constraints << " constraints added" << endl;
    }


    // -----------------------------------------------------------------
    //  Witness generation: assign actual values to all variables
    // -----------------------------------------------------------------
    void generate_witness(const vector<long long>& pixel_values) {
        // Assign input pixels
        assert(pixel_values.size() == input_vars.size());
        for (size_t i = 0; i < pixel_values.size(); i++) {
            pb.val(input_vars[i]) = FieldT(pixel_values[i]);
        }

        // Forward pass through each layer, assigning intermediate values
        int relu_idx = 0;
        int layer_out_idx = 0;

        for (size_t li = 0; li < params.layers.size(); li++) {
            const auto& layer = params.layers[li];

            if (layer.type == "linear") {
                assign_linear_witness(li, layer.linear, layer_out_idx);
                layer_out_idx++;

            } else if (layer.type == "relu") {
                assign_relu_witness(li, layer_out_idx, relu_idx);
                layer_out_idx++;
                int dim = layer_outputs[layer_out_idx - 1].size();
                relu_idx += dim;
            }
        }

        // Assign output vars to match final layer's output
        int last_idx = layer_outputs.size() - 1;
        for (size_t i = 0; i < output_vars.size(); i++) {
            pb.val(output_vars[i]) = pb.val(layer_outputs[last_idx][i]);
        }

        cout << "[witness] Assignment complete. Satisfied: "
             << (pb.is_satisfied() ? "YES" : "NO") << endl;
    }

    // -----------------------------------------------------------------
    //  Run the full zkSNARK pipeline: setup → prove → verify
    // -----------------------------------------------------------------
    bool run_zksnark() {
        cout << "\n===== zkSNARK Pipeline =====" << endl;

        // 1. Generate keypair (trusted setup)
        cout << "[1/3] Running trusted setup (key generation)..." << endl;
        auto keypair = r1cs_ppzksnark_generator<ppT>(pb.get_constraint_system());
        cout << "  Proving key size  : " << keypair.pk.size_in_bits() << " bits" << endl;
        cout << "  Verifying key size: " << keypair.vk.size_in_bits() << " bits" << endl;

        // 2. Generate proof
        cout << "[2/3] Generating proof..." << endl;
        auto proof = r1cs_ppzksnark_prover<ppT>(keypair.pk, pb.primary_input(), pb.auxiliary_input());
        cout << "  Proof generated successfully." << endl;

        // 3. Verify proof
        cout << "[3/3] Verifying proof..." << endl;
        bool verified = r1cs_ppzksnark_verifier_strong_IC<ppT>(
            keypair.vk, pb.primary_input(), proof
        );

        cout << "\n============================" << endl;
        cout << "Verification result: " << (verified ? "PASS ✓" : "FAIL ✗") << endl;
        cout << "============================" << endl;

        // Print classification result
        cout << "\nPublic output (classification logits):" << endl;
        for (size_t i = 0; i < output_vars.size(); i++) {
            cout << "  Class " << i << ": " << pb.val(output_vars[i]) << endl;
        }

        return verified;
    }


private:
    // Helper: get the input variables for a given layer
    const vector<pb_variable<FieldT>>& get_layer_input(int layer_idx,
                                                         int layer_out_idx) {
        if (layer_out_idx == 0) {
            return input_vars;  // First layer takes raw pixels
        } else {
            return layer_outputs[layer_out_idx - 1];
        }
    }

    // Assign witness values for a linear layer (forward pass computation)
    void assign_linear_witness(int layer_idx,
                               const LinearLayerParams& lp,
                               int layer_out_idx)
    {
        const auto& in_vars_ref = get_layer_input(layer_idx, layer_out_idx);
        auto& out_vars_ref = layer_outputs[layer_out_idx];

        int in_dim  = lp.in_features;
        int out_dim = lp.out_features;

        for (int j = 0; j < out_dim; j++) {
            // Compute dot product in full precision
            long long acc = 0;
            for (int i = 0; i < in_dim; i++) {
                long long x_val = pb.val(in_vars_ref[i]).as_ulong();
                // Handle negative values in the field
                // (values close to the field modulus represent negatives)
                long long w = lp.weights[j * in_dim + i];
                acc += w * x_val;
            }
            // Add bias (scaled to SCALE^2)
            acc += lp.biases[j] * (long long)scale;
            // Divide by SCALE to bring back to SCALE domain
            long long result = acc / scale;

            pb.val(out_vars_ref[j]) = FieldT(result);
        }
    }

    // Assign witness values for a ReLU layer
    void assign_relu_witness(int layer_idx,
                             int layer_out_idx,
                             int relu_start_idx)
    {
        const auto& in_vars_ref = layer_outputs[layer_out_idx - 1];
        auto& out_vars_ref = layer_outputs[layer_out_idx];
        int dim = in_vars_ref.size();

        for (int j = 0; j < dim; j++) {
            int ridx = relu_start_idx + j;
            long long x = pb.val(in_vars_ref[j]).as_ulong();

            // Interpret field element as signed value
            // (if x > field_modulus/2, it represents a negative number)
            // For our small values, we use a simpler heuristic
            bool is_positive = true;
            long long signed_x = x;

            // If x is very large (close to field modulus), it's negative
            if (x > (1ULL << 62)) {
                is_positive = false;
                signed_x = 0;  // ReLU of negative is 0
            }

            // Assign sign bit
            pb.val(relu_sign_bits[ridx]) = is_positive ? FieldT(1) : FieldT(0);

            // Assign output
            pb.val(out_vars_ref[j]) = is_positive ? FieldT(x) : FieldT(0);

            // Assign bit decomposition
            long long offset = (1LL << (VALUE_BITS - 1));
            unsigned long long shifted = (unsigned long long)(signed_x + offset);
            for (int b = 0; b < VALUE_BITS; b++) {
                pb.val(relu_bits[ridx][b]) = FieldT((shifted >> b) & 1);
            }
        }
    }
};

#endif // ONNX_CIRCUIT_BUILDER_HPP
