fn main() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .flag("-march=native")
        // .flag("-flto")
        .include("third_party/fph-table/include")
        .file("csrc/fph_wrapper.cpp")
        .compile("fph_wrapper");
    println!("cargo:rerun-if-changed=csrc/fph_wrapper.cpp");
}
