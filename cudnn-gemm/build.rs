use std::path::PathBuf;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rerun-if-changed=kernels/bf16_fp8_cudnn.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![format!("{}/kernels/bf16_fp8_cudnn.cu", manifest_dir)])
        .arg("-arch=sm_90a")
        .arg("-I/usr/local/cuda/include")
        .arg("-Icudnn-frontend/include");

    let target = std::env::var("TARGET").unwrap();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let out_file = if target.contains("msvc") {
        // Windows case
        build_dir.join("modeldcudnn.lib")
    } else {
        build_dir.join("libmodeldcudnn.a")
    };

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=modeldcudnn");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    builder.build_lib(out_file);
}
