use std::path::PathBuf;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rerun-if-changed=kernels/bf16_fp8_cutlass_sm90.cu");
    println!("cargo:rerun-if-changed=kernels/fp8_fp8_cutlass_sm90.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![format!(
            "{}/kernels/bf16_fp8_cutlass_sm90.cu",
            manifest_dir
        ),
        format!(
            "{}/kernels/fp8_fp8_cutlass_sm90.cu",
            manifest_dir
        ),
    ])
        .arg("-arch=sm_90a")
        // .arg("-gencode=arch=compute_120,code=sm_120")
        // .arg("-gencode=arch=compute_120,code=compute_120")
        .arg("-Icutlass/include")
        .arg("-Icutlass/tools/util/include");

    let target = std::env::var("TARGET").unwrap();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let out_file = if target.contains("msvc") {
        // Windows case
        build_dir.join("modeldcutlass.lib")
    } else {
        build_dir.join("libmodeldcutlass.a")
    };

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=modeldcutlass");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    builder.build_lib(out_file);
}
