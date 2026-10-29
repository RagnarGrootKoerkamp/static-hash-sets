mod bin;
pub mod braided_cuckoo;
mod kphf_set;
mod kphf_trait;
mod traits;

pub mod cuckoo;
pub mod cuckoo_quotienting;
#[cfg(feature = "ext")]
pub mod phf_set;
#[cfg(feature = "ext")]
pub mod phf_trait;
#[cfg(feature = "ext")]
pub mod u64_hashset;

use bin::Bin;
pub use kphf_set::KphfSet;
pub use kphf_trait::Kphf;
pub use traits::HashSet;

type T = u64;
const BIN_SIZE: usize = 8;
type S = wide::i64x4;
