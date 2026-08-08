fn main() {
    cc::Build::new()
        .std("c11")
        .opt_level(3)
        .flag("-march=native")
        .file("third_party/tpht/tpht.c")
        .include("third_party/tpht")
        .compile("tpht");

    println!("cargo:rerun-if-changed=third_party/tpht/tpht.c");
    println!("cargo:rerun-if-changed=third_party/tpht/tpht.h");
}
