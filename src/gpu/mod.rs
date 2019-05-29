mod error;
pub use self::error::*;

#[cfg(feature = "ocl")]
mod fft;
#[cfg(feature = "ocl")]
pub use self::fft::*;

#[cfg(feature = "ocl")]
mod multi_exp;
#[cfg(feature = "ocl")]
pub use self::multi_exp::*;

#[cfg(not (feature = "ocl"))]
mod nofft;
#[cfg(not (feature = "ocl"))]
pub use self::nofft::*;
