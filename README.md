# GPU Accelerated Bellman

```bash
cargo build
cargo run
```

# Target

- Provide up to 40x [Bellman](https://github.com/zkcrypto/bellman) prover speeds for larger circuits with FFT parallel processing.

# Roadmap 

### March 26th - May 15th

## OpenCL Implementation
- opencl prover 
	- [kernel](https://github.com/clMathLibraries/clFFT)
- Bellman PR for GPU flagged prover mode
- Test against AMD Radeon 480x

## Potential Future work
- CUDA Support
	- C bellman port
	- rustc LLVM NVPTX upgrades and support
