use std::path::PathBuf;

fn main() {
    let third_party = PathBuf::from("third_party/engineering-k-perfect-hashing");
    let extlib = third_party.join("extlib");
    let ribbon_dir = extlib.join("simple-ribbon").join("extlib").join("ribbon");
    let ips2ra_include = ribbon_dir.join("ips2ra").join("include");
    let tlx_include = ribbon_dir.join("tlx");
    let ribbon_include = extlib.join("simple-ribbon").join("extlib").join("ribbon");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .flag("-march=native")
        // .flag("-lto")
        .flag("-w") // suppress warnings
        // Disable parallel ips2ra (no TBB dependency)
        .define("IPS2RA_DISABLE_PARALLEL", None)
        // Include paths
        .include(third_party.join("include").join("common"))
        .include(third_party.join("include").join("threshold-based-bumping"))
        .include(third_party.join("include").join("hash-displace"))
        .include(extlib.join("gcem").join("include"))
        .include(extlib.join("sux"))
        .include(extlib.join("util").join("include"))
        .include(extlib.join("fips").join("include"))
        .include(extlib.join("simple-ribbon").join("include"))
        .include(&ribbon_include)
        .include(ribbon_include.join("DySECT").join("include"))
        .include(ribbon_include.join("DySECT"))
        .include(&ips2ra_include)
        .include(ips2ra_include.join("ips2ra"))
        .include(&tlx_include)
        .include(extlib.join("simple-ribbon").join("extlib").join("xxhash"))
        .include("../../.spack-env/view/include")
        // SimpleRibbon compiled sources
        .file(
            extlib
                .join("simple-ribbon")
                .join("src")
                .join("SimpleRibbon.cpp"),
        )
        .file(ribbon_include.join("sorter.cpp"))
        // Main wrapper
        .file("csrc/ekphf_wrapper.cpp")
        .compile("ekphf");

    println!("cargo:rerun-if-changed=csrc/ekphf_wrapper.cpp");
    println!(
        "cargo:rerun-if-changed={}",
        third_party
            .join("include")
            .join("threshold-based-bumping")
            .join("ThresholdBasedBumping.hpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        third_party
            .join("include")
            .join("hash-displace")
            .join("HashDisplace.hpp")
            .display()
    );
}
