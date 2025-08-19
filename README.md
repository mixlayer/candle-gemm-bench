# Candle GEMM Benchmarks

Benchmark suite for various different GEMM engines: 

* Candle built-in cuBLAS gemm 
* CUTLASS BF16 @ FP8 mixed precision
* CUTLASS FP8 @ FP8 

### Current Results 

```
Benchmarking candle_builtin_cublas_gemm_bf16/8192x8192@8192x1: Collecting 100 samples in estimated 5.103candle_builtin_cublas_gemm_bf16/8192x8192@8192x1
                        time:   [58.572 µs 58.625 µs 58.692 µs]
                        change: [+1.3768% +1.4752% +1.5720%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 100 measurements (2.00%)                                                           2 (2.00%) high mild                                                                                   

Benchmarking candle_builtin_cublas_gemm_bf16/1024x8192@8192x1: Collecting 100 samples in estimated 5.061candle_builtin_cublas_gemm_bf16/1024x8192@8192x1
                        time:   [22.167 µs 22.186 µs 22.209 µs]
                        change: [+0.9880% +1.1829% +1.3903%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 12 outliers among 100 measurements (12.00%)
  5 (5.00%) low mild
  2 (2.00%) high mild                                                                                     5 (5.00%) high severe
  
Benchmarking candle_builtin_cublas_gemm_bf16/28672x8192@8192x1: Collecting 100 samples in estimated 5.05candle_builtin_cublas_gemm_bf16/28672x8192@8192x1                                                                               time:   [166.70 µs 166.75 µs 166.80 µs]
                        change: [+0.0231% +0.0605% +0.1057%] (p = 0.00 < 0.05)
                        Change within noise threshold.                                                  Found 18 outliers among 100 measurements (18.00%)
  2 (2.00%) low mild
  8 (8.00%) high mild
  8 (8.00%) high severe

Benchmarking cutlass_gemm_bf16_fp8/8192x8192@8192x1: Collecting 100 samples in estimated 5.1662 s (56k icutlass_gemm_bf16_fp8/8192x8192@8192x1
                        time:   [91.810 µs 91.870 µs 91.934 µs]                                                                 
                        change: [+0.1684% +0.2507% +0.3252%] (p = 0.00 < 0.05)
                        Change within noise threshold.                                                  Found 9 outliers among 100 measurements (9.00%)
  6 (6.00%) high mild
  3 (3.00%) high severe                                                                                 Benchmarking cutlass_gemm_bf16_fp8/1024x8192@8192x1: Collecting 100 samples in estimated 5.4244 s (61k icutlass_gemm_bf16_fp8/1024x8192@8192x1
                        time:   [89.383 µs 89.425 µs 89.472 µs]                                                                 
                        change: [−0.2705% −0.1867% −0.1111%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 7 outliers among 100 measurements (7.00%)                                                           
3 (3.00%) high mild
  4 (4.00%) high severe
Benchmarking cutlass_gemm_bf16_fp8/28672x8192@8192x1: Collecting 100 samples in estimated 5.4159 s (30k cutlass_gemm_bf16_fp8/28672x8192@8192x1
                        time:   [178.69 µs 178.78 µs 178.88 µs]
                        change: [−0.1288% −0.0093% +0.0964%] (p = 0.88 > 0.05)
                        No change in performance detected.
Found 12 outliers among 100 measurements (12.00%)                                                         5 (5.00%) high mild
  7 (7.00%) high severe

Benchmarking cutlass_gemm_fp8_fp8/8192x8192@8192x1: Collecting 100 samples in estimated 5.2691 s (86k itcutlass_gemm_fp8_fp8/8192x8192@8192x1
                        time:   [61.353 µs 61.392 µs 61.437 µs]
                        change: [−0.6186% −0.4456% −0.2839%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 9 outliers among 100 measurements (9.00%)
  7 (7.00%) high mild
  2 (2.00%) high severe                                                                                 Benchmarking cutlass_gemm_fp8_fp8/1024x8192@8192x1: Collecting 100 samples in estimated 5.1965 s (101k icutlass_gemm_fp8_fp8/1024x8192@8192x1
                        time:   [49.361 µs 49.404 µs 49.455 µs]                                                                 change: [−0.0026% +0.1229% +0.2470%] (p = 0.05 > 0.05)
                        No change in performance detected.
Found 6 outliers among 100 measurements (6.00%)
  4 (4.00%) high mild
  2 (2.00%) high severe
Benchmarking cutlass_gemm_fp8_fp8/28672x8192@8192x1: Collecting 100 samples in estimated 5.1574 s (40k icutlass_gemm_fp8_fp8/28672x8192@8192x1
                        time:   [127.72 µs 127.78 µs 127.84 µs]
                        change: [−0.0136% +0.0628% +0.1436%] (p = 0.11 > 0.05)
                        No change in performance detected.                                              Found 8 outliers among 100 measurements (8.00%)                                                           
                        1 (1.00%) low mild                                                                                      4 (4.00%) high mild
  3 (3.00%) high severe
```
