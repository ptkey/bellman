extern crate bellman;

fn main(){
    use bellman::domain::{gpu_fft_supported, gpu_fft_consistency};
    gpu_fft_supported(20).expect("Error!");
    gpu_fft_consistency();
}
