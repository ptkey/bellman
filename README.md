# GPU Accelerated Bellman

```bash
cargo build
cargo run
```

# Target Goal

- Provide up to 40x prover speeds for larger circuits with FFT parallel processing.

# Roadmap 
### March 26th - May 15th

- [x] Decide whether to use OpenCL or CUDA
	- start with OCL as it is supported by rust and FFT kernel is noted to be as performant as CUDA.

### OpenCL Implementation
- opencl prover 
	- [kernel](https://github.com/clMathLibraries/clFFT)
- [ ] clFFT support bellman inputs
- [ ] Bellman PR for GPU flagged prover mode
- [ ] Initial benchmarks against AMD Radeon 480x
	- [ ] Once stable, test against other devices

## Potential Future work
- [ ] Rust CUDA Supported
	- C version of bellman or rustc LLVM NVPTX upgrades and support
- [ ] Create a large circuit in Bellman and use a profiling tool to find the CPU-heavy parts of the code
    - We can use [this](https://github.com/TyOverby/flame) library to extract a flamegraph, showing exactly which functions took the longest time to run.
- [ ] Find the parallelizable spots of the Bellman source code.
    - Some potential candidates are:
    - [Fast-Fourier-Transform function](https://github.com/finalitylabs/bellman/blob/437664fa9dc2a5103a664407d2f8f01a4fd5b748/src/domain.rs#L272)
    - `mul_assign`, `sub_assign` and other [methods of Bellman's finite-field struct](https://github.com/finalitylabs/bellman/blob/437664fa9dc2a5103a664407d2f8f01a4fd5b748/src/groth16/prover.rs#L250) which are defined in [Bellman's pairing library](https://github.com/zkcrypto/pairing/blob/183a64b08e9dc7067f78624ec161371f1829623e/src/bls12_381/ec.rs#L534)
    - [Multi-exponentiation](https://github.com/finalitylabs/bellman/blob/437664fa9dc2a5103a664407d2f8f01a4fd5b748/src/groth16/prover.rs#L262)
- [ ] Implement a finite field arithmetic library (For large integers) in OpenCL/CUDA (Or find and use an existing library instead)
    - Some useful links:
    - https://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda
    - https://github.com/blawar/biginteger
    - https://brage.bibsys.no/xmlui/bitstream/handle/11250/143956/Fredrik%20Gundersen.pdf?sequence=4&isAllowed=y
