| 顺序 | LeetCUDA kernel / 主题                                     | 为什么学                                   | 重要度   |
| -- | -------------------------------------------------------- | -------------------------------------- | ----- |
| 1  | `elementwise_f32`                                        | 最基本的 thread → element 映射               | ★★★   |
| 2  | `elementwise_f32x4`                                      | 第一次理解 vectorized load/store            | ★★★   |
| 3  | `mat_trans_f32_col2row`                                  | 二维 block/grid + 矩阵 indexing            | ★★★★★ |
| 4  | `mat_trans_f32_diagonal2d`                               | coalescing、block mapping               | ★★★★  |
| 5  | `mat_trans_f32x4_*`                                      | transpose + vectorized IO              | ★★★★  |
| 6  | `warp_reduce_{all}`                                      | warp shuffle，后面所有 reduction 基础         | ★★★★★ |
| 7  | `block_all_reduce_f32_f32`                               | warp → block reduction                 | ★★★★★ |
| 8  | `block_all_reduce_f32x4_f32`                             | reduction + vectorized load            | ★★★★★ |
| 9  | `dot_product_f32x4`                                      | reduction + memory throughput 综合练习     | ★★★   |
| 10 | `safe_softmax_f32_per_tok`                               | stable softmax + reduction             | ★★★★★ |
| 11 | `safe_softmax_f32x4_per_tok`                             | softmax + vectorized IO                | ★★★★★ |
| 12 | `online_safe_softmax_f32_per_token`                      | FlashAttention 数学基础                    | ★★★★★ |
| 13 | `layer_norm_f32`                                         | mean/variance + reduction              | ★★★★  |
| 14 | `layer_norm_f32x4`                                       | norm 的实际高性能写法                          | ★★★★  |
| 15 | `rms_norm_f32`                                           | LLM 高频算子                               | ★★★★★ |
| 16 | `rms_norm_f32x4`                                         | vLLM/LLM inference 很相关                 | ★★★★★ |
| 17 | `rope_f32` / `rope_f32x4_pack`                           | Transformer 特有算子                       | ★★★★  |
| 18 | `sgemv_k32_f32`                                          | decoding 场景、小 batch GEMV 思维            | ★★★★★ |
| 19 | `sgemv_k128_f32x4`                                       | warp mapping + vector load + reduction | ★★★★★ |
| 20 | `sgemm_naive_f32`                                        | 从最朴素 GEMM 建立基线                         | ★★★★★ |
| 21 | `sgemm_sliced_k_f32`                                     | shared-memory tiling                   | ★★★★★ |
| 22 | `sgemm_t_8x8_sliced_k_f32x4`                             | register tiling + 每线程多个输出              | ★★★★★ |
| 23 | `sgemm_*dbuf / *async`                                   | double buffering + async pipeline      | ★★★★★ |
| 24 | `hgemm_wmma...naive → mma4x2/4x4`                        | Tensor Core / WMMA                     | ★★★★★ |
| 25 | `flash_attn_cute(naive) → flash_attn_mma_stages_split_q` | 把前面所有知识合并成真实推理 kernel                  | ★★★★★ |
