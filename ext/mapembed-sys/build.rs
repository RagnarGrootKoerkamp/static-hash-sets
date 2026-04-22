fn main() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .flag("-march=native")
        .include("third_party/mapembed/CPU/MapEmbed")
        .file("csrc/mapembed_wrapper.cpp")
        .compile("mapembed_wrapper");
    println!("cargo:rerun-if-changed=csrc/mapembed_wrapper.cpp");
}
