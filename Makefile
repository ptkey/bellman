all:
	cargo build --release --features "gpu-test"
	RUST_LOG=info ./target/release/bellperson
	#cargo test --release --features "gpu-test" -- --nocapture --exact groth16::test_with_bls12_381::serialization
	#cargo test --release --features "gpu-test" -- --nocapture --exact groth16::tests::test_xordemo
	#RUST_LOG=info cargo test --release --features "gpu-test" -- --nocapture --exact domain::gpu_fft_consistency
