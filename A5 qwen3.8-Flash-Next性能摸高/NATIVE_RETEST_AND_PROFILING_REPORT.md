# Qwen3.8-Flash-Next 原生 `cpu_offload=true` 复测、Torch-NPU Profiling 瓶颈分析与零改码定向优化报告

## 1. 测试背景与环境基线

`codex-177.42.22.1` 容器 `qwen3.8_flash_next_w00893744_1117` 中的 `vllm-ascend` 已由旧提交 `568d12495` 升级至最新 `main` 提交 **`5d90273879b19c30ab770fe41e2ee4e626a58eab`**（搭配 `vllm` `4ba787ec9959` 与 CANN `9.3.0` `V100R001C25B114`）。本次升级合入了多项针对 `Qwen4Exp` 的关键特性与修复：

1. **原生 PLE UVA 主机卸载与跨 DP 共享内存（PR `#21751` / `470dd81ed`）**：注册 `@Qwen4ExpPLEPinnedHostEmbedding.register_oot -> AscendPLEHostEmbedding`，通过 `SharedUvaBuffer`（`/dev/shm` POSIX 共享内存 + `MADV_HUGEPAGE` + `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer`）及 `_ple_uva_lookup_kernel` 原生支持 `cpu_offload=true`，无需任何外部 `PYTHONPATH` 垫片或源码改动。
2. **QSA 异步解码主机同步消除（PR `#21758` / `b858314e3`）**：将 `AscendQSAMetadataBuilder` 纳入免同步列表，消除异步调度/投机解码路径中每步 `num_computed_tokens_event.synchronize()` 主机气泡。
3. **MTP 长上下文位置缓存修复（PR `#21362` / `da205aa80`）**：修复 `Qwen4ExpMTP` 长序列 RoPE 位置索引越界问题。

> **公平性与零改码审计约束**：全程保持 `gpu_memory_utilization=0.9`、`BF16` 权重、`FP32` SSM 状态（`--mamba-ssm-cache-dtype float32`）、`TP=4, DP=2, EP=true`、关闭前缀缓存、`ignore-eos`、`temperature=0`。所有正式吞吐压测与 `--profiler-config` 剖析运行 **100% 物理隔离**；测试前后 `vllm` / `vllm-ascend` `git status` 均为空，编译扩展 `.so` 与模型 `config.json` SHA256 完全未变。

---

## 2. 第一阶段：原生 `cpu_offload=true` 复测结果

使用标准原生配置 `--engram-config '{"cpu_offload": true, "embedding_across_dp": false, "dp_shared_memory": true}'` 复测：

- **原生共享内存与显存验证**：
  - 8 个 Worker 全部打印：`PLE host offload: 80000384 rows x 160, 23.842 GiB mapped per rank, BF16, shared=True`。
  - 开启 `dp_shared_memory=true` 后，`DP0` 与 `DP1` 共享同一份 `/dev/shm` 表，宿主机共享内存占用仅为 **`96G`（`1000G` 的 `10%`）**，较不共享方案（`191G`）减半；停服后自动释放回 **`0%`**。
  - 每张 NPU 权重占用由 `embedding_across_dp=true` 时的 `38.68 GiB`（不跨 DP 时 `44.72 GiB`）降至 **`32.63~32.77 GiB`**，KV Cache 池扩大至 **`46.04~46.83 GiB`**（纯文本 `4,155` 块/DP，多模态 `4,226` 块/DP）。

| 工作负载 | 方案来源 | 并发 `C` | 抢占数 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | 平均 `TTFT` (ms) | 平均 `TPOT` (ms) | 耗时 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 纯文本 `2K->6K` | 上一轮外挂 `uva_shim` | 544 | 0 | 9,190.42 | 12,253.90 | 17,056.4 | 56.08 | 363.68 |
| 纯文本 `2K->6K` | **本轮第一阶段原生 `5d9027387`** | **544** | **0** | **9,264.03** | **12,352.04** | **16,990.3** | **55.59** | **360.79** |
| 纯文本 `2K->6K` | 上一轮外挂 `uva_shim` | 576 | 18 | 9,318.92 | 12,425.23 | 18,452.1 | 58.15 | 379.76 |
| 纯文本 `2K->6K` | **本轮第一阶段原生 `5d9027387`** | **576** | **18** | **9,322.72** | **12,430.29** | **18,605.2** | **58.12** | **379.60** |
| 多模态 `4K+1080p->32K` | **本轮第一阶段原生 `5d9027387`** | **160** | **0** | **4,670.45** | **5,552.78** | **23,671.3** | **33.47** | **1,122.56** |
| 多模态 `4K+1080p->32K` | **本轮第一阶段原生 `5d9027387`** | **176** | **4** | **4,852.72** | **5,769.48** | **25,994.7** | **34.85** | **1,188.44** |

> **复测结论**：最新版 `vllm-ascend` 原生实现不仅消除了外挂垫片依赖、将 `/dev/shm` 内存减半，还在纯文本 `C=544` 零抢占场景因省去冗余 `output.copy_()`（`supports_prefetch=False`）并启用大页建议（`MADV_HUGEPAGE`）取得 **`+0.80%`** 净吞吐增益；多模态 `C=176` 则暴露出默认图档位越过 `4,226` 块上限引发 `4` 次 `32K` 尾部重计算惩罚（`~65s` 尾部拖尾）。

---

## 3. 第二阶段：稳态解码 Torch-NPU Profiling 深度拆解

在独立隔离服务上，待全量并发请求（纯文本 `C=544` 即 `272`/DP；多模态 `C=160` 即 `80`/DP）100% 进入稳态 `FULL_DECODE_ONLY` ACLGraph 回放后（校验 `gdn_recurrent_calls = 720 = 36 层 × 20 步`、`ple_lookup_calls = 20 = 1 × 20 步`，零预填充混入），采集 `20` 步硬件级轨迹并执行 CANN 离线解析。

### 3.1 硬件流时间轴拆解 (`step_trace_time.csv`，Rank 0)

| 指标项 | 纯文本稳态解码 (`C=544`, `272`/DP) | 多模态稳态解码 (`C=160`, `80`/DP) | 瓶颈诊断结论 |
|---|---:|---:|---|
| 20 步总耗时 `Stage` | `1,083.52 ms` (`54.18 ms/步`) | `640.61 ms` (`32.03 ms/步`) | 与端到端实测 `TPOT`（`55.59 ms` / `33.47 ms`）高度吻合 |
| 计算耗时 `Computing` | `921.42 ms` (**`85.04%`**) | `537.64 ms` (**`83.93%`**) | 主导耗时，但大量由碎片化小算子构成 |
| 未掩盖通信 `Comm(Not Overlapped)` | `107.78 ms` (**`9.95%`**, `5.39 ms/步`) | `76.27 ms` (**`11.91%`**, `3.81 ms/步`) | **`Overlapped = 0.0 ms`**，集合通信与共享专家完全串行 |
| 设备空闲间隙 `Free` | `54.32 ms` (**`5.01%`**, `2.72 ms/步`) | `26.70 ms` (**`4.17%`**, `1.33 ms/步`) | 单步高达 `2,904~3,000` 个算子，图内调度与同步间隙显著 |
| 加权 Cube `aic_mac_ratio` | `10.78%` | `7.78%` | 内存带宽与访存延迟受限（Memory-Bound） |
| 加权 Vector `aiv_vec_ratio` | `15.59%` | `11.48%` | 大量小粒度逐元素/切片/路由算子拉低向量核有效饱和度 |

### 3.2 九大算子家族单步耗时拆解 (`kernel_details.csv`，Rank 0，20 步均值)

| 算子家族 | 纯文本 `C=544` 单步耗时 (ms) | 纯文本占比 (%) | 多模态 `C=160` 单步耗时 (ms) | 多模态占比 (%) | 单步调用数 | 核心热点算子（单步耗时：纯文本 / 多模态） |
|---|---:|---:|---:|---:|---:|---|
| `9_Norm_Activation_Elementwise_Other` | **10.77** | **18.97%** | **7.96** | **23.01%** | 1,505 / 1,409 | `Slice` (`0.88 / 0.73 ms`), `SwiGlu` (`0.83 / 0.53 ms`), `TensorMove` (`0.39 ms`), `ZerosLike` (`0.31 / 0.28 ms`) |
| `2_MoE_Compute_GMM_Routing` | **9.51** | **16.76%** | **5.91** | **17.07%** | 144 | `aclnnGroupedMatmulV5` (`9.24 / 5.71 ms`), `MoeGatingTopK` (`0.20 ms`) |
| `4_GDN_Linear_Attention_36L` | **8.92** | **15.71%** | **3.39** | **9.78%** | 108 | `RecurrentGatedDeltaRule` (`6.64 / 1.96 ms`), `CausalConv1d` (`1.32 / 0.52 ms`), `fused_gdn_gating` (`0.96 / 0.91 ms`) |
| `5_QSA_Sparse_Attention_12L` | **7.16** | **12.60%** | **4.09** | **11.81%** | 168 | `GenericBlockSparseAttention` (`4.77 / 2.36 ms`), `LightningIndexer` (`0.62 / 0.45 ms`), `GBSAMetadata` (`0.52 / 0.41 ms`) |
| `8_Dense_Projections_MatMul` | **6.11** | **10.77%** | **4.43** | **12.81%** | 532 | `aclnnMatmul_MatMulV3` (`5.72 / 4.13 ms`，484 次/步，单次仅 `8.5~11.8 us`) |
| `3_TP_DP_HCCL_Collective_Comm` | **5.57** | **9.82%** | **4.00** | **11.56%** | 148 | `hcom_allReduce` (`3.77 / 2.91 ms`, 98 次/步), `hcom_allGather` (`1.35 / 0.81 ms`, 48 次/步) |
| `7_HyperConnection_48L` | **4.15** | **7.31%** | **1.93** | **5.58%** | 292 | `_hc_combine_norm` (`2.38 / 0.90 ms`), `_hc_gate_mix` (`1.19 / 0.63 ms`), `_hc_silu` (`0.54 / 0.38 ms`) |
| `1_MoE_Dispatch_Combine_Comm` | **3.84** | **6.76%** | **2.65** | **7.66%** | 96 | `MoeDistributeCombineV3` (`1.96 / 1.49 ms`), `MoeDistributeDispatchV3` (`1.88 / 1.16 ms`) |
| `6_PLE_Host_UVA_Offload_1L` | **0.74** | **1.30%** | **0.25** | **0.72%** | 7 | `_ple_conv_decode` (`0.42 ms`), `_ple_uva_lookup` (**仅 `0.32 ms / <0.15 ms`！**) |

### 3.3 四大核心 Profiling 发现

1. **原生 PLE UVA 主机查表代价极低**：即便将 `95.37 GiB` PLE 词表放在主机 `/dev/shm` 中，通过 UB/HCCS 直连页锁定映射，`_ple_uva_lookup` 单步耗时仅 **`0.15~0.32 ms`（占单步 `<0.6%`）**，却换取每卡 **`+11.9 GiB` HBM**，直接解释了 `cpu_offload=true` 吞吐全面碾压 HBM 驻留方案的物理根因。
2. **MoE 分发/聚合与共享专家零重叠（`Overlapped = 0.0 ms`）**：48 层 `MoeDistributeDispatchV3 + CombineV3` 占 `2.65~3.84 ms/步`，而 `SharedExpertMLP`（含 `Qwen4Exp` 特有的 `expert_gate`）默认在主流上串行执行；源码核查确认 `additional_config.multistream_overlap_shared_expert=true` 已适配 `expert_gate` 与 ACLGraph 双流同步。
3. **静态图捕获档位断层与 KV 块边界错位**：
   - 混合 KV 块公式（`TP=4, FP32 SSM`）为 $B_{\text{req}}(L) = 5 + \lceil L / 896 \rceil$。
   - 多模态单请求总长约 `38,961` 词元，占 `49` 块；默认 ACLGraph 档位从 `80` 直接跳到 `88`，导致 `C=160`（`80`/DP）闲置 `6.5%` KV 池，而 `C=176`（`88`/DP，需 `4,312` 块）超出池上限触发 `4` 次 `32K` 尾部抢占重算。通过自定义 `cudagraph_capture_sizes` 注入精确档位（`85, 86, 87, 88`）并将 `max_num_batched_tokens` 收紧至 `2560`（释放峰值激活显存，使可用块增至 `4,242` 块/DP），即可精确跑满 `86`/DP（`C=172`）零抢占边界！
4. **`FULL_DECODE_ONLY` 下预填充批上限 `max_num_batched_tokens` 的双向耦合定律**：
   - 只要某一步混入哪怕 `1` 个预填充词元，引擎就会退回非静态图（Eager）模式（单步 `~160 ms` vs ACLGraph `53 ms`）。
   - 在纯文本 `2K -> 6K`（总耗时仅 `~350s`）中，减小 `max_num_batched_tokens`（如 `3072`）会把每条 `2048` 词元提示切碎，使 Eager 混合阶段从 `33.36s` 暴涨至 `47.86s`（吃掉解码收益）；反之将其调大至 **`6528`**（$= 3 \times 2048 + 384$，即使已有 `288` 条解码序列仍能每步整吞 `3` 条完整提示、零切分），可将 `P99 TTFT` 压降至 **`29.96s`**，快速切入全静态图解码！

---

## 4. 第三阶段：零改码消融与全量吞吐优化验证

### 4.1 优化旋钮消融结果

1. **`multistream_overlap_shared_expert=true`（采纳 ✅）**：
   - 优化后复采 `20` 步硬件 Profiling（`text_decode_opt_c544`）证实：8 张卡的 `Stage_us` 全面下降 **`1.7%~2.3%`**（Rank 0 从 `1,083.52 ms` 降至 **`1,058.51 ms`**，`Computing_us` 从 `921.42 ms` 降至 **`898.72 ms`**，`Free_us` 从 `54.32 ms` 降至 **`47.57 ms`，降幅 `-12.4%`**），共享专家在辅助流上与主干 MoE 实现硬件并发。
2. **自定义紧凑精确 `cudagraph_capture_sizes` + 场景化 `max_num_batched_tokens`（采纳 ✅）**：
   - **纯文本**：精简捕获档位使 `npu_graph` 显存从 `0.55 GiB` 降至 `0.35 GiB`，配合 `max_num_batched_tokens=6528` 消除 `2048` 提示切分，`C=544` 端到端耗时从 `360.79s` 缩短至 **`349.89s`**。
   - **多模态**：`max_num_batched_tokens=2560` 将峰值激活从 `0.82 GiB` 压至 `0.69 GiB`，KV 池扩至 **`47.00 GiB`（`4,242` 块/DP）**，配合精确档位 `86, 88` 同时刷新零抢占与峰值纪录。
3. **`enable_static_kernel=true`（不采纳 ❌）**：多档位全模型静态编译触发 `bisheng`/`opc` 编译超过 `multiproc_executor` 600s 通信超时被取消，当前版本高并发多档位不建议直接开启。
4. **`MTP=1` 投机解码 `qwen4_exp_mtp`（高并发吞吐场景不采纳 ❌）**：`#21758` + `#21362` 下功能完全正常，但草稿头权重（`+0.75 GiB`）与缓存使可用块降至 `3,797` 块/DP，高并发探测（`C=256/384, out=512`）输出吞吐（`2,304 / 1,830 tok/s`）低于非投机模式（`3,629 tok/s`），更适合低并发延迟敏感场景。

### 4.2 全部 12 组正式全长压测完整对比表（100% 通过逐请求输出长度独立校验）

| 场景 | 阶段 | 配置标签 | `C` | 可用块/DP | 抢占数 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | `P99 TTFT` (ms) | 平均 `TPOT` (ms) | 耗时 (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 纯文本 `2K->6K` | Phase 1 | `native_offload_dp_shared` | 544 | 4,155 | 0 | 9,264.03 | 12,352.04 | 33,357.3 | 55.59 | 360.79 |
| 纯文本 `2K->6K` | Phase 1 | `native_offload_dp_shared` | 576 | 4,155 | 18 | 9,322.72 | 12,430.29 | 36,427.1 | 58.12 | 379.60 |
| 纯文本 `2K->6K` | Phase 3 消融 | `opt_exact_graph_b3072_ms_ovlp` | 544 | 4,171 | 0 | 8,989.55 | 11,986.07 | 47,856.6 | 56.17 | 371.80 |
| 纯文本 `2K->6K` | Phase 3 消融 | `opt_exact_graph_b3072_ms_ovlp` | 556 | 4,171 | 0 | 8,981.74 | 11,975.65 | 50,561.5 | 57.27 | 380.33 |
| 纯文本 `2K->6K` | Phase 3 消融 | `opt_exact_graph_b3072_ms_ovlp` | 594 | 4,171 | 26 | 8,766.13 | 11,688.18 | 54,816.2 | 60.86 | 416.32 |
| **纯文本 `2K->6K`** | **Phase 3 最优** | **`opt_exact_graph_b6528_ms_ovlp`** | **544** | **4,130** | **0** | **9,552.60** | **12,736.80** | **29,958.7** | **54.21** | **349.89** |
| 纯文本 `2K->6K` | Phase 3 最优 | `opt_exact_graph_b6528_ms_ovlp` | 550 | 4,130 | 1 | 9,528.73 | 12,704.97 | 30,172.6 | 54.86 | 354.63 |
| **纯文本 `2K->6K`** | **Phase 3 最优** | **`opt_exact_graph_b6528_ms_ovlp`** | **576** | **4,130** | **26** | **9,580.70** | **12,774.26** | **31,585.0** | **56.84** | **369.38** |
| 多模态 `4K+1080p->32K` | Phase 1 | `native_offload_dp_shared` | 160 | 4,226 | 0 | 4,670.45 | 5,552.78 | 45,138.6 | 33.47 | 1,122.56 |
| 多模态 `4K+1080p->32K` | Phase 1 | `native_offload_dp_shared` | 176 | 4,226 | 4 | 4,852.72 | 5,769.48 | 50,076.8 | 34.85 | 1,188.44 |
| **多模态 `4K+1080p->32K`** | **Phase 3 最优** | **`opt_exact_graph_b2560_ms_ovlp`** | **172** | **4,242** | **0** | **4,956.85** | **5,893.28** | **57,845.1** | **33.69** | **1,137.03** |
| **多模态 `4K+1080p->32K`** | **Phase 3 最优** | **`opt_exact_graph_b2560_ms_ovlp`** | **176** | **4,242** | **4** | **5,023.21** | **5,972.17** | **60,499.1** | **33.81** | **1,148.10** |

---

## 5. 后续引擎源码级优化演进建议（基于 Profiling 量化证据）

在不修改业务配置之外，本次硬件级 Profiling 还定位出以下值得向上游 `vllm-ascend` 演进的源码级算子融合点（预计可进一步带来 `8%~15%` 稳态解码提速）：

1. **补齐 `causal_conv1d_update_npu` Triton 算子导出**：服务端日志明确提示 `NPU Triton causal_conv1d_update is unavailable`，当前 36 层 GDN 回退至 `CausalConv1d` 算子耗时 `1.32 ms/步`（`720` 次/20 步），与 `fused_gdn_gating_kernel`（`0.96 ms/步`）融合可省去约 `1.0~1.3 ms/步`。
2. **4 流 HyperConnection 算子纵向融合**：`_hc_combine_norm_kernel`（`2.38 ms/步`）、`_hc_gate_mix_kernel`（`1.19 ms/步`）与 `_hc_silu_kernel`（`0.54 ms/步`）每步启动 `292` 次，单步共占 `4.15 ms`（`7.31%`），可将残差门控与归一化合并为单次驻留 L1/UB 的 Triton 算子。
3. **消除小粒度 `Slice` / `TensorMove` / `ZerosLike` 碎片算子**：`9_Norm_Activation_Elementwise_Other` 每步高达 `1,409~1,505` 个小算子，占总核时 `18.97%~23.01%`，并直接推高 `Free` 间隙（`4.2%~5.0%`）；通过视图原地切片（零拷贝 stride view）与预分配清零复用可显著削减启动开销。
4. **支持 `enable_static_kernel` 增量编译缓存与多进程并行编译**：解决当前多档位 ACLGraph 下 `bisheng`/`opc` 编译超 `600s` 的冷启动瓶颈，使 `super_kernel` 图内算子级联下发得以在大批次生产配置中落地。
