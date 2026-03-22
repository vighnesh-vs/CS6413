/*
 * main.cpp
 * ========
 * Entry point for the ONNX → zkSNARK proof-of-inference system.
 *
 * Usage:
 *   ./zk_inference <path_to_circuit_params.json>
 *
 * This program:
 *   1. Loads the quantized model parameters from JSON
 *   2. Builds the R1CS constraint system (circuit) in libsnark
 *   3. Generates the witness (assigns actual pixel values)
 *   4. Runs trusted setup → proof generation → verification
 *   5. Reports the classification result and verification status
 */

#include <iostream>
#include <chrono>

#include <libff/common/default_types/ec_pp.hpp>
#include "onnx_circuit_builder.hpp"

using namespace std;


int main(int argc, char* argv[]) {
    // ---- Parse command-line arguments ----
    string json_path = "circuit_params.json";
    if (argc >= 2) {
        json_path = argv[1];
    }

    cout << "============================================" << endl;
    cout << "  zkSNARK Proof-of-Inference for MNIST MLP  " << endl;
    cout << "============================================" << endl;
    cout << "Circuit params: " << json_path << endl << endl;

    // ---- Initialize the elliptic curve parameters ----
    ppT::init_public_params();

    // ---- Load model parameters from JSON ----
    cout << "[Step 1] Loading circuit parameters..." << endl;
    CircuitParams params = load_circuit_params(json_path);

    // ---- Build the circuit ----
    cout << "\n[Step 2] Building R1CS circuit..." << endl;
    MLPInferenceCircuit circuit(params);

    auto t0 = chrono::high_resolution_clock::now();
    circuit.allocate_variables();
    circuit.generate_constraints();
    auto t1 = chrono::high_resolution_clock::now();

    double circuit_time = chrono::duration<double>(t1 - t0).count();
    cout << "  Circuit build time: " << circuit_time << " seconds" << endl;
    cout << "  Variables    : " << circuit.pb.num_variables() << endl;
    cout << "  Constraints  : " << circuit.pb.num_constraints() << endl;

    // ---- Generate witness using the sample input ----
    cout << "\n[Step 3] Generating witness (forward pass)..." << endl;
    auto t2 = chrono::high_resolution_clock::now();
    circuit.generate_witness(params.sample_pixels);
    auto t3 = chrono::high_resolution_clock::now();

    double witness_time = chrono::duration<double>(t3 - t2).count();
    cout << "  Witness generation time: " << witness_time << " seconds" << endl;
    cout << "  True label of sample: " << params.true_label << endl;

    // ---- Determine predicted class (argmax of output logits) ----
    int predicted_class = 0;
    FieldT max_logit = circuit.pb.val(circuit.output_vars[0]);
    for (size_t i = 1; i < circuit.output_vars.size(); i++) {
        FieldT val = circuit.pb.val(circuit.output_vars[i]);
        // Simple comparison using field element as unsigned long
        if (val.as_ulong() > max_logit.as_ulong()) {
            max_logit = val;
            predicted_class = i;
        }
    }
    cout << "  Predicted class: " << predicted_class << endl;

    // ---- Run full zkSNARK pipeline ----
    cout << "\n[Step 4] Running zkSNARK pipeline..." << endl;
    auto t4 = chrono::high_resolution_clock::now();
    bool verified = circuit.run_zksnark();
    auto t5 = chrono::high_resolution_clock::now();

    double zksnark_time = chrono::duration<double>(t5 - t4).count();

    // ---- Summary ----
    cout << "\n============= SUMMARY =============" << endl;
    cout << "Model:            MLP (784 → " << params.layers[0].linear.out_features
         << " → 10)" << endl;
    cout << "Scale factor:     " << params.scale << endl;
    cout << "R1CS variables:   " << circuit.pb.num_variables() << endl;
    cout << "R1CS constraints: " << circuit.pb.num_constraints() << endl;
    cout << "-----------------------------------" << endl;
    cout << "True label:       " << params.true_label << endl;
    cout << "Predicted class:  " << predicted_class << endl;
    cout << "Match:            " << (predicted_class == params.true_label ? "YES" : "NO") << endl;
    cout << "-----------------------------------" << endl;
    cout << "Circuit build:    " << circuit_time << " s" << endl;
    cout << "Witness gen:      " << witness_time << " s" << endl;
    cout << "zkSNARK total:    " << zksnark_time << " s" << endl;
    cout << "Proof verified:   " << (verified ? "YES ✓" : "NO ✗") << endl;
    cout << "=====================================" << endl;

    return verified ? 0 : 1;
}
