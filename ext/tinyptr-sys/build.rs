fn main() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("-march=native")
        .flag("-mavx2")
        .include("third_party/tinyptr/src")
        .file("csrc/tinyptr_wrapper.cpp")
        .file("third_party/tinyptr/src/nonconc_blast_ht.cpp")
        .compile("tinyptr");

    println!("cargo:rerun-if-changed=csrc/tinyptr_wrapper.cpp");
    println!("cargo:rerun-if-changed=third_party/tinyptr/src/nonconc_blast_ht.h");
    println!("cargo:rerun-if-changed=third_party/tinyptr/src/nonconc_blast_ht.cpp");
}
