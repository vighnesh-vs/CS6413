/**
 * ============================================================================
 *  Phase 2 — C++ : circuit.json → R1CS → Prove → Verify
 * ============================================================================
 *
 *  System architecture:
 *
 *    Phase 1 (Python):  Train model → ONNX export → Quantize → JSON
 *    Phase 2 (C++):     Trusted Setup → Generate Witness → Create Proof → Verify
 *
 *  This program implements Phase 2:
 *    1. Load quantized model from circuit.json (produced by Phase 1)
 *    2. Build the R1CS circuit encoding the full MLP inference
 *    3. Trusted Setup  → Proving Key (PK) + Verifying Key (VK)
 *    4. Generate Witness (forward pass with actual values)
 *    5. Create Proof: Prover(PK, witness) → proof π
 *    6. Verify Proof: Verifier(VK, proof, public output) → accept/reject
 *
 *  R1CS constraint formulas:
 *
 *    Linear layer:       SCALE × y[j] = Σᵢ( W[j][i] × x[i] ) + b[j] × SCALE
 *    Square activation:  x[j] × x[j]  =  y[j] × SCALE
 *
 *    In the circuit, scales accumulate through layers:
 *      z1   at scale S²   : FC1 output
 *      a1   at scale S⁴   : after square activation (z1 × z1)
 *      out  at scale S⁵   : FC2 output
 *
 *  Constraint budget ~276:
 *    FC1 (linear layer 1):    128 constraints
 *    Square activation (x²):  128 constraints
 *    FC2 (linear layer 2):     10 constraints
 *    Output equality:           10 constraints
 *    Total:                    276 constraints
 *
 *  Privacy model:
 *    Private:  image pixels + model weights + all intermediate activations
 *    Public:   predicted output logits (10 values, class 0–9)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <algorithm>

#include <depends/libsnark/libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <depends/libsnark/libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <depends/libsnark/libsnark/gadgetlib1/pb_variable.hpp>

using namespace libsnark;
using namespace std;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;


// ==========================================================================
//  Model dimensions 
// ==========================================================================
static const int INPUT_SIZE  = 784;
static const int HIDDEN_SIZE = 128;
static const int OUTPUT_SIZE = 10;


// ==========================================================================
//  Data Structures
// ==========================================================================

struct QuantizedModel {
    int scale;                          // SCALE = 1000
    vector<vector<long long>> W1;       // (128 × 784) at scale S
    vector<long long> b1;               // (128)       at scale S
    vector<vector<long long>> W2;       // (10 × 128)  at scale S
    vector<long long> b2;               // (10)        at scale S
};

struct TestWitness {
    int true_label;
    int predicted;
    vector<long long> x_q;              // (784) input pixels    at scale S
    vector<long long> z1_q;             // (128) FC1 output      at scale S²
    vector<long long> a1_q;             // (128) after square    at scale S⁴
    vector<long long> out_q;            // (10)  FC2 output      at scale S⁵
};


// ==========================================================================
//  Field element helpers
// ==========================================================================

/** Convert signed int64 to field element (negative → p - |x|). */
FieldT to_field(long long x) {
    if (x >= 0)
        return FieldT(static_cast<unsigned long>(x));
    else
        return FieldT::zero() - FieldT(static_cast<unsigned long>(-x));
}

/** Add coefficient × variable to a linear combination (handles sign). */
void add_term(linear_combination<FieldT>& lc,
              long long coeff,
              const pb_variable<FieldT>& var) {
    if (coeff > 0)
        lc = lc + FieldT(static_cast<unsigned long>(coeff)) * var;
    else if (coeff < 0)
        lc = lc - FieldT(static_cast<unsigned long>(-coeff)) * var;
}

/** Add a constant (bias × scale) to a linear combination via the ONE wire. */
void add_constant(linear_combination<FieldT>& lc, long long constant) {
    // ONE is a libsnark macro: pb_variable<FieldT>(0), the constant-1 wire
    if (constant > 0)
        lc = lc + FieldT(static_cast<unsigned long>(constant)) * ONE;
    else if (constant < 0)
        lc = lc - FieldT(static_cast<unsigned long>(-constant)) * ONE;
}


// ==========================================================================
//  JSON parser (minimal, for circuit.json)
// ==========================================================================

/** Skip whitespace in a string starting at pos. */
static void skip_ws(const string& s, size_t& p) {
    while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t')) p++;
}

/** Parse a JSON number (integer, possibly negative). */
static long long parse_number(const string& s, size_t& p) {
    skip_ws(s, p);
    size_t start = p;
    if (s[p] == '-') p++;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9') p++;
    return stoll(s.substr(start, p - start));
}

/** Parse a JSON array of numbers: [n1, n2, ...] */
static vector<long long> parse_array(const string& s, size_t& p) {
    skip_ws(s, p);
    assert(s[p] == '['); p++;
    vector<long long> result;
    skip_ws(s, p);
    if (s[p] == ']') { p++; return result; }
    while (true) {
        result.push_back(parse_number(s, p));
        skip_ws(s, p);
        if (s[p] == ']') { p++; break; }
        assert(s[p] == ','); p++;
    }
    return result;
}

/** Parse a JSON 2D array: [[...], [...], ...] */
static vector<vector<long long>> parse_2d_array(const string& s, size_t& p) {
    skip_ws(s, p);
    assert(s[p] == '['); p++;
    vector<vector<long long>> result;
    skip_ws(s, p);
    while (s[p] != ']') {
        result.push_back(parse_array(s, p));
        skip_ws(s, p);
        if (s[p] == ',') p++;
        skip_ws(s, p);
    }
    p++; // skip ']'
    return result;
}

/** Find a JSON key and position cursor after the colon. */
static bool find_key(const string& s, size_t& p, const string& key) {
    string needle = "\"" + key + "\":";
    size_t pos = s.find(needle, p);
    if (pos == string::npos) {
        // Try with space after colon
        needle = "\"" + key + "\" :";
        pos = s.find(needle, p);
    }
    if (pos == string::npos) return false;
    p = pos + needle.size();
    skip_ws(s, p);
    return true;
}


// ==========================================================================
//  Load circuit.json  (Phase 1 output → Phase 2 input)
// ==========================================================================

void load_circuit_json(const string& path,
                       QuantizedModel& model,
                       TestWitness& witness) {
    // Read entire file
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "ERROR: Cannot open " << path << endl;
        exit(1);
    }
    string json_str((istreambuf_iterator<char>(fin)),
                     istreambuf_iterator<char>());
    fin.close();

    size_t p = 0;

    // Parse metadata
    find_key(json_str, p, "scale");
    model.scale = (int)parse_number(json_str, p);

    // Parse quantized weights
    p = 0;
    find_key(json_str, p, "quantized_weights");

    size_t pw = p;
    find_key(json_str, pw, "W1");
    model.W1 = parse_2d_array(json_str, pw);

    find_key(json_str, pw, "b1");
    model.b1 = parse_array(json_str, pw);

    find_key(json_str, pw, "W2");
    model.W2 = parse_2d_array(json_str, pw);

    find_key(json_str, pw, "b2");
    model.b2 = parse_array(json_str, pw);

    // Parse test witness
    p = 0;
    find_key(json_str, p, "test_witness");

    size_t pt = p;
    find_key(json_str, pt, "true_label");
    witness.true_label = (int)parse_number(json_str, pt);

    find_key(json_str, pt, "predicted");
    witness.predicted = (int)parse_number(json_str, pt);

    find_key(json_str, pt, "input");
    witness.x_q = parse_array(json_str, pt);

    find_key(json_str, pt, "z1");
    witness.z1_q = parse_array(json_str, pt);

    find_key(json_str, pt, "a1");
    witness.a1_q = parse_array(json_str, pt);

    find_key(json_str, pt, "output");
    witness.out_q = parse_array(json_str, pt);

    // Validate dimensions
    assert((int)model.W1.size() == HIDDEN_SIZE);
    assert((int)model.W1[0].size() == INPUT_SIZE);
    assert((int)model.b1.size() == HIDDEN_SIZE);
    assert((int)model.W2.size() == OUTPUT_SIZE);
    assert((int)model.W2[0].size() == HIDDEN_SIZE);
    assert((int)model.b2.size() == OUTPUT_SIZE);
    assert((int)witness.x_q.size() == INPUT_SIZE);
    assert((int)witness.z1_q.size() == HIDDEN_SIZE);
    assert((int)witness.a1_q.size() == HIDDEN_SIZE);
    assert((int)witness.out_q.size() == OUTPUT_SIZE);
}


// ==========================================================================
//  Main — Phase 2 Pipeline
// ==========================================================================

int main() {
    using hrc = chrono::high_resolution_clock;
    auto t_start = hrc::now();

    cout << "============================================================" << endl;
    cout << "  Phase 2 — C++: R1CS Circuit → zkSNARK Proof" << endl;
    cout << "  zkSNARK Proof-of-Inference for MNIST MLP" << endl;
    cout << "============================================================" << endl;

    // ------------------------------------------------------------------
    //  Step 0: Initialize BN128 curve parameters (libff / libsnark)
    // ------------------------------------------------------------------
    cout << "\n[0] Initializing BN128 curve parameters ..." << endl;
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    libff::inhibit_profiling_info = true;

    // ------------------------------------------------------------------
    //  Step 1: Load circuit.json (output of Phase 1)
    // ------------------------------------------------------------------
    cout << "\n[1] Loading circuit.json (Phase 1 output) ..." << endl;
    QuantizedModel model;
    TestWitness witness;
    load_circuit_json("model_data/circuit.json", model, witness);

    cout << "    Model architecture: " << INPUT_SIZE << " -> "
         << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << endl;
    cout << "    Scale factor (SCALE): " << model.scale << endl;
    cout << "    Test sample predicted class: " << witness.predicted << endl;
    if (witness.true_label >= 0)
        cout << "    True label: " << witness.true_label << endl;

    long long S = model.scale;

    // ==================================================================
    //  Step 2: Build R1CS Circuit 
    //
    //  Translate each layer into R1CS constraints.
    //  Linear layers → dot-product constraints
    //  Square activation → one multiply constraint per neuron (y = x × x)
    // ==================================================================
    cout << "\n[2] Building R1CS circuit ..." << endl;
    auto t_circuit = hrc::now();

    protoboard<FieldT> pb;

    // --- PUBLIC variables: output logits (10) ---
    //     The verifier sees these; computes argmax for predicted class.
    //     Allocated first: libsnark treats the first N variables as public.
    pb_variable_array<FieldT> out_public;
    out_public.allocate(pb, OUTPUT_SIZE, "output_logits_public");
    pb.set_input_sizes(OUTPUT_SIZE);

    // --- PRIVATE (witness) variables ---
    pb_variable_array<FieldT> x_var;
    x_var.allocate(pb, INPUT_SIZE, "input_pixels");       // 784

    pb_variable_array<FieldT> z1_var;
    z1_var.allocate(pb, HIDDEN_SIZE, "fc1_output");       // 128

    pb_variable_array<FieldT> a1_var;
    a1_var.allocate(pb, HIDDEN_SIZE, "square_activation"); // 128

    pb_variable_array<FieldT> fc2_out_var;
    fc2_out_var.allocate(pb, OUTPUT_SIZE, "fc2_output");   // 10

    int total_vars = OUTPUT_SIZE + INPUT_SIZE + HIDDEN_SIZE * 2 + OUTPUT_SIZE;
    cout << "    Variables: " << total_vars
         << "  (" << OUTPUT_SIZE << " public, "
         << (total_vars - OUTPUT_SIZE) << " private)" << endl;

    // ── FC1: 128 linear layer constraints ──
    //    Conceptual formula: SCALE × y[j] = Σᵢ(W1[j][i] × x[i]) + b1[j] × SCALE
    //    R1CS form: ( Σᵢ W1[j][i]·x[i] + b1[j]·S ) × 1 = z1[j]
    //    z1[j] is at scale S² (because W at S × x at S = S², plus bias×S at S²)
    cout << "    Adding FC1 linear layer constraints (" << HIDDEN_SIZE << ") ..." << endl;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        linear_combination<FieldT> lc;
        for (int i = 0; i < INPUT_SIZE; i++) {
            add_term(lc, model.W1[j][i], x_var[i]);
        }
        // b[j] × SCALE 
        add_constant(lc, model.b1[j] * S);

        pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(lc, 1, z1_var[j]),
            "fc1_" + to_string(j));
    }

    // ── Square activation: 128 constraints ──
    //    Formula: x[j] × x[j] = y[j] × SCALE
    //    R1CS form: z1[j] × z1[j] = a1[j]
    //    a1[j] is at scale S⁴ (because S² × S² = S⁴)
    cout << "    Adding square activation constraints (" << HIDDEN_SIZE << ") ..." << endl;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(z1_var[j], z1_var[j], a1_var[j]),
            "square_" + to_string(j));
    }

    // ── FC2: 10 linear layer constraints ──
    //    R1CS form: ( Σⱼ W2[i][j]·a1[j] + b2[i]·S⁴ ) × 1 = fc2_out[i]
    //    fc2_out[i] is at scale S⁵ (because W at S × a1 at S⁴ = S⁵)
    cout << "    Adding FC2 linear layer constraints (" << OUTPUT_SIZE << ") ..." << endl;
    long long S4 = S * S * S * S;  // S⁴ for bias scaling in FC2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        linear_combination<FieldT> lc;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            add_term(lc, model.W2[i][j], a1_var[j]);
        }
        // b2[i] × S⁴ to match scale S⁵
        add_constant(lc, model.b2[i] * S4);

        pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(lc, 1, fc2_out_var[i]),
            "fc2_" + to_string(i));
    }

    // ── Output equality: 10 constraints ──
    //    Enforce: fc2_out[i] = out_public[i]
    //    This copies the private FC2 result to the public output variables
    //    so the verifier can inspect the logits.
    cout << "    Adding output equality constraints (" << OUTPUT_SIZE << ") ..." << endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        pb.add_r1cs_constraint(
            r1cs_constraint<FieldT>(fc2_out_var[i], 1, out_public[i]),
            "out_eq_" + to_string(i));
    }

    int total_constraints = HIDDEN_SIZE + HIDDEN_SIZE + OUTPUT_SIZE + OUTPUT_SIZE;
    auto t_circuit_end = hrc::now();
    double circuit_ms = chrono::duration<double, milli>(t_circuit_end - t_circuit).count();

    cout << "    ────────────────────────────────────────" << endl;
    cout << "    Total R1CS constraints: " << total_constraints
         << "  (128 + 128 + 10 + 10)" << endl;
    cout << "    Circuit build time: " << circuit_ms << " ms" << endl;

    // ==================================================================
    //  Step 3: Assign Witness 
    //
    //  The prover fills in all private variables with the actual
    //  quantized values computed during the forward pass.
    // ==================================================================
    cout << "\n[3] Generating witness (forward pass) ..." << endl;
    auto t_wit = hrc::now();

    // Private: input image pixels
    for (int i = 0; i < INPUT_SIZE; i++)
        pb.val(x_var[i]) = to_field(witness.x_q[i]);

    // Private: FC1 output (z1, at scale S²)
    for (int j = 0; j < HIDDEN_SIZE; j++)
        pb.val(z1_var[j]) = to_field(witness.z1_q[j]);

    // Private: squared activation output (a1, at scale S⁴)
    for (int j = 0; j < HIDDEN_SIZE; j++)
        pb.val(a1_var[j]) = to_field(witness.a1_q[j]);

    // Private: FC2 raw output (at scale S⁵)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        pb.val(fc2_out_var[i]) = to_field(witness.out_q[i]);

    // Public: output logits (same values, visible to verifier)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        pb.val(out_public[i]) = to_field(witness.out_q[i]);

    auto t_wit_end = hrc::now();
    double wit_ms = chrono::duration<double, milli>(t_wit_end - t_wit).count();
    cout << "    Witness assignment time: " << wit_ms << " ms" << endl;

    // Sanity check: all constraints must be satisfied
    cout << "    Checking constraint satisfaction ... ";
    if (pb.is_satisfied()) {
        cout << "SATISFIED" << endl;
    } else {
        cout << "NOT SATISFIED" << endl;
        cerr << "ERROR: Constraints not satisfied. "
             << "Check quantization or witness values." << endl;
        return 1;
    }

    // ==================================================================
    //  Step 4: Trusted Setup 
    //
    //  Circuit-specific generation of keypair (PK, VK).
    //  Needs to be done only once per model architecture.
    // ==================================================================
    cout << "\n[4] Trusted Setup → PK + VK ..." << endl;
    auto t_setup = hrc::now();

    const auto keypair =
        r1cs_gg_ppzksnark_generator<default_r1cs_gg_ppzksnark_pp>(
            pb.get_constraint_system());

    auto t_setup_end = hrc::now();
    double setup_ms = chrono::duration<double, milli>(t_setup_end - t_setup).count();
    cout << "    Trusted setup time: " << setup_ms << " ms" << endl;

    // ==================================================================
    //  Step 5: Create Proof 
    //
    //  Prover(PK, witness) → proof π
    //  The prover knows: input pixels, model weights, intermediate
    //  activations (all private).
    // ==================================================================
    cout << "\n[5] Creating proof: Prover(PK, witness) → proof π ..." << endl;
    auto t_prove = hrc::now();

    const auto proof =
        r1cs_gg_ppzksnark_prover<default_r1cs_gg_ppzksnark_pp>(
            keypair.pk,
            pb.primary_input(),
            pb.auxiliary_input());

    auto t_prove_end = hrc::now();
    double prove_ms = chrono::duration<double, milli>(t_prove_end - t_prove).count();
    cout << "    Proof generation time: " << prove_ms << " ms" << endl;

    // ==================================================================
    //  Step 6: Verify Proof
    //
    //  Verifier(VK, proof, public output) → accept / reject
    //
    //  The verifier knows ONLY:
    //    • The verifying key (VK)
    //    • The claimed output logits (public)
    //  The verifier does NOT know:
    //    • The input image
    //    • The model weights
    //    • Any intermediate activations
    // ==================================================================
    cout << "\n[6] Verifying proof: Verifier(VK, proof, output) ..." << endl;
    auto t_verify = hrc::now();

    bool verified = r1cs_gg_ppzksnark_verifier_strong_IC<
        default_r1cs_gg_ppzksnark_pp>(
            keypair.vk,
            pb.primary_input(),
            proof);

    auto t_verify_end = hrc::now();
    double verify_ms = chrono::duration<double, milli>(t_verify_end - t_verify).count();
    cout << "    Verification time: " << verify_ms << " ms" << endl;

    // ==================================================================
    //  Determine predicted class from public outputs
    // ==================================================================
    int predicted_class = 0;
    long long max_logit = witness.out_q[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (witness.out_q[i] > max_logit) {
            max_logit = witness.out_q[i];
            predicted_class = i;
        }
    }

    // ==================================================================
    //  Results 
    // ==================================================================
    auto t_end = hrc::now();
    double total_ms = chrono::duration<double, milli>(t_end - t_start).count();

    cout << "\n============================================================" << endl;
    cout << "  RESULTS  " << endl;
    cout << "============================================================" << endl;
    cout << "  Proof verification:          "
         << (verified ? "SUCCESS" : "FAILURE") << endl;
    cout << "  Predicted class (0-9):       " << predicted_class << endl;
    if (witness.true_label >= 0) {
        cout << "  True label:                  " << witness.true_label << endl;
        cout << "  Classification correct:      "
             << (predicted_class == witness.true_label ? "YES" : "NO") << endl;
    }
    cout << endl;
    cout << "  --- Performance Metrics ---" << endl;
    cout << "  Circuit construction:        " << circuit_ms << " ms" << endl;
    cout << "  Witness generation:          " << wit_ms << " ms" << endl;
    cout << "  Trusted setup:               " << setup_ms << " ms" << endl;
    cout << "  Proof generation:            " << prove_ms << " ms" << endl;
    cout << "  Proof verification:          " << verify_ms << " ms" << endl;
    cout << "  Total wall-clock time:       " << total_ms << " ms" << endl;
    cout << endl;
    cout << "  --- Circuit Statistics ---" << endl;
    cout << "  R1CS constraints:            ~" << total_constraints << endl;
    cout << "    Linear layer 1 (FC1):      " << HIDDEN_SIZE << endl;
    cout << "    Square activation (x^2):   " << HIDDEN_SIZE << endl;
    cout << "    Linear layer 2 (FC2):      " << OUTPUT_SIZE << endl;
    cout << "    Output equality:           " << OUTPUT_SIZE << endl;
    cout << "  R1CS variables:              ~" << total_vars << endl;
    cout << "    Public (output logits):    " << OUTPUT_SIZE << endl;
    cout << "    Private (witness):         " << (total_vars - OUTPUT_SIZE) << endl;
    cout << "  Scale factor:                " << model.scale << endl;
    cout << "  Activation:                  Square (x^2)" << endl;
    cout << "============================================================" << endl;

    return verified ? 0 : 1;
}
