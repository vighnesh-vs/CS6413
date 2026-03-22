#include <iostream>
#include <vector>
#include <depends/libsnark/libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <depends/libsnark/libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <depends/libsnark/libsnark/gadgetlib1/pb_variable.hpp>


#include <depends/libsnark/libsnark/common/default_types/r1cs_ppzksnark_pp.hpp>
#include <depends/libsnark/libsnark/zk_proof_systems/ppzksnark/r1cs_ppzksnark/r1cs_ppzksnark.hpp>

using namespace libsnark;
using namespace std;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

/*
template<typename ppT>
bool run_r1cs_ppzksnark(const r1cs_example<libff::Fr<ppT> > &example)
{
    libff::print_header("R1CS ppzkSNARK Generator");
    
    r1cs_ppzksnark_keypair<ppT> keypair = r1cs_ppzksnark_generator<ppT>(example.constraint_system);
    
    printf("\n"); libff::print_indent(); libff::print_mem("after generator");

    libff::print_header("Preprocess verification key");
    
    r1cs_ppzksnark_processed_verification_key<ppT> pvk = r1cs_ppzksnark_verifier_process_vk<ppT>(keypair.vk);

    libff::print_header("R1CS ppzkSNARK Prover");
    r1cs_ppzksnark_proof<ppT> proof = r1cs_ppzksnark_prover<ppT>(keypair.pk, example.primary_input, example.auxiliary_input);
    printf("\n"); libff::print_indent(); libff::print_mem("after prover");

    libff::print_header("R1CS ppzkSNARK Verifier");
    const bool ans = r1cs_ppzksnark_verifier_strong_IC<ppT>(keypair.vk, example.primary_input, proof);
    
    printf("\n"); libff::print_indent(); libff::print_mem("after verifier");
    printf("* The verification result is: %s\n", (ans ? "PASS" : "FAIL"));

    libff::print_header("R1CS ppzkSNARK Online Verifier");
    
    const bool ans2 = r1cs_ppzksnark_online_verifier_strong_IC<ppT>(pvk, example.primary_input, proof);
    assert(ans == ans2);

    return ans;
}

template<typename ppT>
void test_r1cs_ppzksnark(size_t num_constraints, size_t input_size)
{
    r1cs_example<libff::Fr<ppT> > example = generate_r1cs_example_with_binary_input<libff::Fr<ppT> >(num_constraints, input_size);
    const bool bit = run_r1cs_ppzksnark<ppT>(example);
    assert(bit);
}
*/


int main() {
    
    default_r1cs_gg_ppzksnark_pp::init_public_params();

    protoboard<FieldT> pb;
    
    /*
    pb_variable_array<FieldT> input_image;
    pb_variable_array<FieldT> weights;
    pb_variable<FieldT> output_class;

    input_image.allocate(pb, 784, "input"); // For 28x28 MNIST
    weights.allocate(pb, 784, "weights");
    output_class.allocate(pb, "output");
    
    // Set output_class as the public input
    pb.set_input_sizes(1); 

    linear_combination<FieldT> res;
    for(int i=0; i<784; ++i) {
        res = res + (weights[i] * input_image[i]);
    }
    // R1CS Constraint: (Inner Product) * (1) = (Output)
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(res, 1, output_class));

    const r1cs_constraint_system<FieldT> cs = pb.get_constraint_system();

    auto keypair = r1cs_ppzksnark_generator<default_r1cs_ppzksnark_pp>(cs);

    // Load quantized data into variables
    pb.val(input_image[0]) = FieldT(125); // Example pixel value

    // ... load all values ...

    auto proof = r1cs_ppzksnark_prover<default_r1cs_ppzksnark_pp>(keypair.pk, pb.primary_input(), pb.auxiliary_input());

    bool is_valid = r1cs_ppzksnark_verifier_strong_IC<default_r1cs_ppzksnark_pp>(keypair.vk, pb.primary_input(), proof);
    */
    
    //test_r1cs_ppzksnark<default_r1cs_ppzksnark_pp>(1000, 100);

/**
 * The follwing code demonstrates a zk-SNARK for the computation: x^3 + x + 5 = y
 * The Prover wants to prove they know an 'x' that satisfies the equation 
 * for a public 'y', without revealing 'x'.
 */

    pb_variable<FieldT> x;
    pb_variable<FieldT> y;
    pb_variable<FieldT> x_sq;
    pb_variable<FieldT> x_cube;

    // Allocate variables on the protoboard
    // 'y' is public input, 'x' is private witness
    y.allocate(pb, "y");
    x.allocate(pb, "x");
    x_sq.allocate(pb, "x_sq");
    x_cube.allocate(pb, "x_cube");

    // Set the public input gap (everything allocated before this is public)
    pb.set_input_sizes(1);

    // Add R1CS constraints: A * B = C
    // Constraint 1: x * x = x_sq
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(x, x, x_sq));

    // Constraint 2: x_sq * x = x_cube
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(x_sq, x, x_cube));

    // Constraint 3: (x_cube + x + 5) * 1 = y
    // We use a linear combination for (x_cube + x + 5)
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(x_cube + x + 5, 1, y));

    // Generate the keypair (Trusted Setup)
    const r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> keypair =
        r1cs_gg_ppzksnark_generator<default_r1cs_gg_ppzksnark_pp>(pb.get_constraint_system());

    // Prover sets values and generates proof
    // Let's say x = 3, so y = 3^3 + 3 + 5 = 35
    pb.val(x) = 3;
    pb.val(x_sq) = 9;
    pb.val(x_cube) = 27;
    pb.val(y) = 35;

    if (!pb.is_satisfied()) {
        cout << "Constraints not satisfied!" << endl;
        return 1;
    }

    const r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> proof =
        r1cs_gg_ppzksnark_prover<default_r1cs_gg_ppzksnark_pp>(keypair.pk, pb.primary_input(), pb.auxiliary_input());

    // Verifier checks the proof
    // The verifier only knows the public input (y=35) and the verification key
    bool result = r1cs_gg_ppzksnark_verifier_strong_IC<default_r1cs_gg_ppzksnark_pp>(keypair.vk, pb.primary_input(), proof);

    cout << "Proof verification: " << (result ? "SUCCESS" : "FAILURE") << endl;


    return 0;

}
