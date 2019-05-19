mod error;
pub use self::error::*;

#[cfg(feature = "ocl")]
mod fft;
#[cfg(feature = "ocl")]
pub use self::fft::*;

#[cfg(not (feature = "ocl"))]
mod nofft;
#[cfg(not (feature = "ocl"))]
pub use self::nofft::*;
