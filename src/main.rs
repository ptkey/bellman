#![allow(unused_imports)]
#![allow(unused_variables)]
extern crate bellman;
extern crate pairing;
extern crate rand;
use bellman::{Circuit, ConstraintSystem, SynthesisError};
use bellman::groth16::{Parameters};
use pairing::{Engine, Field, PrimeField};

use std::fs::File;
use std::io::prelude::*;

mod dummy;
mod gpu;

fn main(){

    println!("Running a sample OpenCL program...");
    gpu::trivial();
    println!("=================================================");

    use pairing::bls12_381::{Bls12, Fr};
    use rand::thread_rng;
    use bellman::groth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof, Proof,
    };

    println!("I know the value of 2^(2^1000)");

    let rng = &mut thread_rng();

    println!("Creating parameters...");

    let load_parameters = true;
    let parameters_path = "parameters.dat";

    // Create parameters for our circuit
    let params = if load_parameters {
        let mut param_file = File::open(parameters_path).expect("Unable to open parameters file!");
        Parameters::<Bls12>::read(param_file, false /* false for better performance*/)
            .expect("Unable to read parameters file!")
    } else {
        let c = dummy::DummyDemo::<Bls12> {
            xx: None
        };

        let p = generate_random_parameters(c, rng).unwrap();
        let mut param_file = File::create(parameters_path).expect("Unable to create parameters file!");
        p.write(param_file).expect("Unable to write parameters file!");
        p
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
