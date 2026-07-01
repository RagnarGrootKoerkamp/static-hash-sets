mod kphf_set;
mod traits;

#[cfg(feature = "ext")]
pub mod cuckoo;
#[cfg(feature = "ext")]
pub mod kphf_trait;
#[cfg(feature = "ext")]
pub mod phf_set;
#[cfg(feature = "ext")]
pub mod phf_trait;
#[cfg(feature = "ext")]
pub mod u64_hashset;

pub use kphf_set::KphfSet;
pub use traits::HashSet;
