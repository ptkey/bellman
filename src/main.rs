#![allow(unused_imports)]
#![allow(unused_variables)]
extern crate bellman;
extern crate pairing;
extern crate rand;
use bellman::{Circuit, ConstraintSystem, SynthesisError};
use pairing::{Engine, Field, PrimeField};

mod dummy;

fn main(){
    use pairing::bls12_381::{Bls12, Fr};
    use rand::thread_rng;
    use bellman::groth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof, Proof,
    };

    println!("I know the value of 2^(2^1000)");

    let rng = &mut thread_rng();

    println!("Creating parameters...");

    // Create parameters for our circuit
    let params = {
        let c = dummy::DummyDemo::<Bls12> {
            xx: None
        };

        generate_random_parameters(c, rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    println!("Creating proofs...");

    // Create an instance of circuit
    let c = dummy::DummyDemo::<Bls12> {
        xx: Fr::from_str("3")
    };

    // Create a groth16 proof with our parameters.
    let proof = create_random_proof(c, &params, rng).unwrap();

    println!("{}", verify_proof(
        &pvk,
        &proof,
        &[]
    ).unwrap());
}
