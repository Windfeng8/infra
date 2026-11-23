# vLLM / SGLang IPC、NCCL、Sparse NCCL 社区周报

> **统计窗口**：2026-11-13 12:00 ～ 2026-11-20 12:00（UTC+8）  
> **数据来源**：GitHub 官方 PR 元数据与 PR 作者说明；核验时间：2026-11-20 12:00（UTC+8）附近。  
> **统计口径**：以窗口内新建、更新或合并的相关 PR 为主；状态以核验时点为准。PR 页面最后更新时间并不必然意味着本期有实质代码变化。所有 benchmark 均为**作者报告，非独立复测**。

## 一、本周摘要

1. **vLLM Sparse NCCL 的 OOM/通信开销修复已合并**：[#66375](https://github.com/vllm-project/vllm/pull/66375) 于 11 月 14 日合并，512 MiB 分批、每批按 dtype 聚合广播，解决大比例稀疏更新的显存峰值问题。作者在 Qwen3-30B-A3B TP4 报告广播次数 37,734 → 2～22，3% 更新耗时下降约 50%。
2. **vLLM Packed NCCL / IPC 正在统一协议**：[#68043](https://github.com/vllm-project/vllm/pull/68043) 11 月 19 日合并，统一 chunk 布局并允许超出缓冲区的单个张量跨 chunk 拆分；[#67840](https://github.com/vllm-project/vllm/pull/67840) 的 planned receive 仍为 Draft。
3. **vLLM 稀疏更新正确性风险仍未完全关闭**：[#66944](https://github.com/vllm-project/vllm/pull/66944) 仍 Open，针对 GDN/Mamba2 稀疏 patch 中 NaN 标记与派生权重二次变换问题。
4. **SGLang 权重缓存 VMM FD 传输已合并**：[#46953](https://github.com/sgl-project/sglang/pull/46953) 于 11 月 19 日合并；这是 CUDA VMM/FD 共享权重机制，**不是 Sparse NCCL**。
5. **SGLang IPC 与权重更新安全性推进**：[#51237](https://github.com/sgl-project/sglang/pull/51237) 修复 target/draft 重复重建 CUDA IPC 张量；[#50229](https://github.com/sgl-project/sglang/pull/50229) 防止 TP 某个 rank 失败却错误报告更新成功；两者均仍 Open。

## 二、vLLM 重点 PR

| PR | 分类 | 创建（UTC） | 最近更新（UTC） | 核验状态 | 本周关键变化 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [#66375](https://github.com/vllm-project/vllm/pull/66375) | **Sparse NCCL** | 11-08 08:07 | 11-14 12:46 | **Merged 11-14** | 512 MiB 分批传输；按 dtype 聚合 indices/values；修复非连续 patch |
| [#68043](https://github.com/vllm-project/vllm/pull/68043) | Packed NCCL + Packed CUDA IPC | 11-17 01:15 | 11-19 05:42 | **Merged 11-19** | 两种传输共享 chunk 布局；跨 chunk 大张量；修复 IPC 非对齐条目 |
| [#67840](https://github.com/vllm-project/vllm/pull/67840) | Packed NCCL / 接收数据面 | 11-16 03:07 | 11-18 05:27 | Open · Draft | 基于 load plan 直接写入模型、合并拷贝、减少缓冲占用 |
| [#68578](https://github.com/vllm-project/vllm/pull/68578) | Unpacked CUDA IPC | 11-19 04:26 | 11-19 22:17 | Open · Draft | 单 trainer rank 发 IPC handle 前也必须同步计算流 |
| [#68608](https://github.com/vllm-project/vllm/pull/68608) | Packed IPC 可扩展性 | 11-19 07:19 | 11-19 07:37 | Open · Draft | 抽象设备、handle、映射重建接口；尚需适配 #68043 |
| [#62853](https://github.com/vllm-project/vllm/pull/62853) | **NCCL M2N** | 10-21 07:54 | 11-17 19:42 | Open | PP trainer 不同 source group 分别向推理 worker 传参数 |
| [#68132](https://github.com/vllm-project/vllm/pull/68132) | Dense NCCL / 生命周期 | 11-17 09:46 | 11-20 03:05 | Open | FT scale-down 后清理失效 PyNCCL 与 PyTorch NCCL communicator |
| [#66944](https://github.com/vllm-project/vllm/pull/66944) | Sparse patch 正确性 | 11-11 03:47 | 11-20 02:50 | Open | 修复 GDN/Mamba2 NaN mask 被写入和派生权重重复变换 |

### 2.1 Sparse NCCL：#66375 已合并

**机制**：发送方和接收方依据 patch metadata 一致切分批次，每批最多约 512 MiB（超限单 patch 单独成批）；同类型索引和数值合并后广播，worker 逐批接收并应用。修复非连续 tensor 直接广播导致的字节错误；失败时仍排空后续广播以避免通信组失配。

**作者测试**：4×H20、TP4、Qwen3-30B-A3B，trainer 位于第五张 H20。主干版本 1.25% 变化比例发生 OOM；该 PR 最高验证到 50%。0.1% / 1% / 3% 变化比例的更新耗时分别降低 21% / 33% / 50%，约为 2.1 秒；广播次数从 37,734 降到 2～22。以上均为 PR 作者自测，不能推断 A3/A5 或其他模型具有相同收益。

### 2.2 Packed NCCL / IPC：#68043 已合并，#67840 仍在评审

[#68043](https://github.com/vllm-project/vllm/pull/68043) 定义共同 `chunks(specs, buffer_size)` 协议，按 16 字节对齐，支持单个张量拆分为多个 chunk，避免为 16 GiB expert stack 分配同等大小的打包缓冲；unpacked NCCL/IPC 不变。

[#67840](https://github.com/vllm-project/vllm/pull/67840) 在接收侧复用 load plan，将相邻规则写入合并为 strided copy。作者报告 DeepSeek-V4-Flash GB200 steady-state 更新 2.7 s → 0.62 s，8,180 → 933 次 copy；但对照包含相关栈式 PR，且约 0.5 s 收益来自另一 FP4 prep 改动，**不可全部归因于 #67840**。此 PR 仍为 Draft，不能视为主干能力。

### 2.3 IPC 同步和 NPU 扩展边界

[#68578](https://github.com/vllm-project/vllm/pull/68578) 发现 unpacked IPC 在单 trainer rank 时跳过 all-gather，原本依赖其中的 stream synchronize，可能将未完成写入的权重句柄发送给 worker。作者在 GB200 报告原主干复现失败、修复分支 IPC 测试 22 passed。该问题是 **CUDA IPC 的数据就绪顺序**，并非 NCCL 通信吞吐问题。

[#68608](https://github.com/vllm-project/vllm/pull/68608) 拟将 Packed IPC 的 handle 重建接口抽象为可插拔 backend；作者 CPU bootstrap 测试 5 passed / 30 skipped，**尚无 NPU 端到端验证**。其描述明确指出需基于已合并 #68043 重新适配，不能认定当前已支持 torch_npu IPC。

### 2.4 M2N、通信组生命周期与稀疏正确性

- [#62853](https://github.com/vllm-project/vllm/pull/62853) 是 **NCCL M2N**：允许 PP trainer 的参数分属不同、互不相交的 source group，每组各建传输 communicator；不是 sparse NCCL 的增量编码。
- [#68132](https://github.com/vllm-project/vllm/pull/68132) 是 **NCCL 生命周期修复**：作者在 Qwen3-30B-A3B DP4 弹性 EP 故障缩容场景报告额外回收约 1 GiB 显存；另约 1 GiB 残留与 UCX CUDA IPC cache 有关，不能混为 NCCL 泄漏。
- [#66944](https://github.com/vllm-project/vllm/pull/66944) 指出 sparse patch 的 NaN sentinel 对使用 `__setitem__` 的 loader 和复合变换 loader 不安全。作者报告单测原主干 6 failed / 31 passed，修复分支 37 passed；截至本次核验仍未合并。

## 三、SGLang 重点 PR

| PR | 分类 | 创建（UTC） | 最近更新（UTC） | 核验状态 | 本周关键变化 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [#46953](https://github.com/sgl-project/sglang/pull/46953) | CUDA VMM FD / 权重缓存 | 10-28 05:45 | 11-19 01:00 | **Merged 11-19** | daemon 通过 FD 共享 CUDA VMM allocation，engine 自行映射地址 |
| [#50910](https://github.com/sgl-project/sglang/pull/50910) | ROCm/HIP IPC + NCCL 对照 | 11-17 03:05 | 11-20 03:53 | Open | diffusion 双 rank K/V gather 用 IPC P2P 数据复制替代两次大 all-gather |
| [#51237](https://github.com/sgl-project/sglang/pull/51237) | CUDA IPC / RL weight update | 11-18 12:19 | 11-19 23:52 | Open | target 与 draft 复用同一次 IPC tensor 重建，减少句柄持有 |
| [#46208](https://github.com/sgl-project/sglang/pull/46208) | **NPU IPC** / 多模态 | 10-24 02:03 | 11-19 11:29 | Open | 新增 `mm_feature_transport=npu_ipc`，复用有界 IPC pool |
| [#42210](https://github.com/sgl-project/sglang/pull/42210) | GPU delta / weight sync | 10-02 07:41 | 11-18 01:52 | Open | 对接 Miles GPU delta 解码和原地更新；非 Sparse NCCL |
| [#51419](https://github.com/sgl-project/sglang/pull/51419) | 分块更新事务正确性 | 11-19 03:16 | 11-19 06:23 | Open · Draft | 防止跨 chunk loader 状态丢失、部分提交 |
| [#50229](https://github.com/sgl-project/sglang/pull/50229) | TP weight update 一致性 | 11-13 04:32 | 11-19 07:01 | Open | 某 TP rank 更新失败时统一返回失败，避免假成功 |
| [#44883](https://github.com/sgl-project/sglang/pull/44883) | NPU IPC 旧方案 | 10-17 10:33 | 11-13 06:05 | **Closed，未合并** | 复用 `cuda_ipc` CLI 的 NPU IPC 方案；不可视为已交付 |

### 3.1 CUDA VMM FD：#46953 已合并

`weight_cache` 新增 `--weight-cache-transport {auto,torch_ipc,vmm_fd}`；daemon 将权重打包至可导出的 CUDA VMM allocation，通过 Unix socket `SCM_RIGHTS` 发送 FD；engine 预留虚拟地址区间后映射。作者说明 VMM 映射在 daemon 退出后仍能保留 allocation，改善 daemon 退出导致 engine 失效的问题。**当前 `auto` 的 daemon 端仍默认 torch_ipc**；地址不变的卸载/重映射属于后续规划，不是本 PR 已实现的功能。

### 3.2 HIP IPC 替代 RCCL gather：#50910

ROCm diffusion 双 rank K/V gather 在具备 peer access 时，使用设备 IPC 共享双缓冲，实际 K/V 数据走 P2P copy，仅用一个标量 all-gather 做顺序屏障。功能由 `SGLANG_DIFFUSION_IPC_KV_GATHER` 显式启用，失败回退原路径。

作者给出的动机数据：gfx1250 socket 上 25 MB/rank all-gather 446 μs（有效 56 GB/s），设备 copy 9.8 μs；FLUX.2 两步 trace 的 432 个 RCCL kernel 占 1183 ms GPU 时间中的 219 ms（18.5%）。**这些是原路径的 profiling 与微基准，不应直接当成该 PR 端到端加速比**。

### 3.3 CUDA IPC target/draft 重建：#51237

旧路径对 target、draft 分别调用 `LocalSerializedTensor.get()`，重复打开同一 CUDA IPC handle。修复将重建移至 runner 循环外，一次重建后共享。作者在 Qwen3.6-35B-A3B TP4 共置 RL 场景报告首次同步后恢复 KV cache 发生 OOM；GPU 测试示例表明旧路径有约 716 MiB 的 producer allocation 残留。PR 仍 Open，未构成已合入保证。

### 3.4 NPU IPC：#46208 与 #44883 的区别

[#46208](https://github.com/sgl-project/sglang/pull/46208) 新增明确的 `npu_ipc` 选项，采用 `torch_npu` IPC handle 和 CANN value write/wait，实现单机多模态特征设备侧传输，并对 pool 从 KV 预算中预留容量。它仍为 Open。

[#44883](https://github.com/sgl-project/sglang/pull/44883) 则尝试在 NPU 上复用原 `cuda_ipc` 入口，采用 `torch.multiprocessing` reducer/rebuilder，且不使用 NPU pool；11 月 13 日 **Closed、未合并**。两种设计不能合并描述为“SGlang 已支持 NPU IPC”。

### 3.5 权重更新事务与 GPU delta

[#50229](https://github.com/sgl-project/sglang/pull/50229) 作者在 CPU Gloo 故障注入中测试 disk、distributed、serialized tensor、IPC 四条路径，原主干 12/12 peer-failure case 错报成功，分支 0/12；这是注入测试，**不是线上故障发生率**。

[#51419](https://github.com/sgl-project/sglang/pull/51419) 讨论 QKV-A 融合和 FP8 indexer 权重/scale 跨 chunk 更新时的 loader 状态保持，仍 Draft，正文测试栏尚无实质验证。

[#42210](https://github.com/sgl-project/sglang/pull/42210) 依赖外部 [Miles #3894](https://github.com/radixark/miles/pull/3894)，实现 GPU delta 解码与原地权重更新 hook；这是**差分权重算法与外部接收/更新框架**，不能称为 NCCL sparse transport。

## 四、分类边界与本周覆盖

| 类别 | 典型机制 | 本期核验结论 |
| :-- | :-- | :-- |
| CUDA IPC / HIP IPC | 共享 device allocation/handle，设备间映射 | vLLM #68578、#68608；SGLang #50910、#51237 |
| NPU IPC | torch_npu IPC / CANN 设备侧同步 | SGLang #46208 Open；#44883 Closed 未合并；本期未验证 vLLM 原生 NPU IPC 落地 |
| 消息 IPC | 控制消息、进程间序列化/MessagePack | 与设备内存共享不同；本期未核验出需列为核心突破的合并项 |
| Dense NCCL | 常规广播、all-gather、reduce 等 | vLLM #68132 生命周期；SGLang #50910 的替代基线 |
| **Sparse NCCL** | 只传 changed indices + values 的稀疏权重 patch | **vLLM #66375 Merged**；#66944 是应用正确性修复（Open） |
| **NCCL M2N** | 多 trainer source ranks → 多 inference ranks 的权重分发 | vLLM #62853 Open；与 Sparse NCCL 正交 |
| RDMA / WeightBridge / Connex | 跨节点传输/外部权重桥接方案 | 本次标题关键词检索未取得足以确认新增落地的核心 PR；不据此断言社区无更新 |
| sparse_disk | 从磁盘稀疏载入/差分文件 | 与 sparse NCCL 的通信通路不同；本期无经核验的核心变更 |
| GPU delta / 数据面优化 | XOR/delta decode、load plan、tensor copy 合并 | SGLang #42210；vLLM #67840；不能直接等同 NCCL 带宽优化 |

## 五、两项目对比、风险与建议

| 维度 | vLLM | SGLang |
| :-- | :-- | :-- |
| Sparse NCCL 权重同步 | 本期已合并 #66375，降低广播数量与 OOM 风险 | 本期未核验到同类原生 Sparse NCCL 已合并项；GPU delta 依赖 Miles |
| Packed NCCL / IPC | #68043 已统一 chunk 协议；planned receive 尚 Draft | 主要聚焦 CUDA IPC 重建/权重缓存 VMM FD |
| NPU IPC | #68608 仅 backend 抽象 Draft，无 NPU 实测 | #46208 提供明确 NPU 方案但仍 Open |
| 事务/一致性 | #66944 sparse loader 正确性仍 Open | #50229 TP 错报成功、#51419 chunk 事务仍待合入 |
| 可迁移性 | #66375 的 batching 思路可评估用于 HCCL Sparse；不能照搬 CUDA 实现 | NPU IPC 的 CANN 事件和共享句柄实现值得跟踪 |

**主要风险**：Sparse NCCL 分批后虽然控制显存峰值，但失败后的部分权重已应用，不能自动理解为原子回滚；#66944 暴露部分 loader 不遵守 NaN mask 语义；Packed IPC 的 GPU stream 同步与 handle 生命周期仍在修补；NPU 方案仍需设备端到端和多 rank 验证。

## 六、旧 PR 本期状态说明

- [vLLM #66375](https://github.com/vllm-project/vllm/pull/66375) 虽创建于上期（11 月 8 日），但 **11 月 14 日合并**，属于本期实质更新。
- [SGLang #46953](https://github.com/sgl-project/sglang/pull/46953) 虽创建于 10 月 28 日，但 **11 月 19 日合并**，属于本期实质更新。
- [vLLM #62853](https://github.com/vllm-project/vllm/pull/62853)、[SGLang #42210](https://github.com/sgl-project/sglang/pull/42210) 均早于本期创建，最近更新发生在窗口内；**未逐 commit 比对，不能将全部方案内容视为本周新增代码**。
- [SGLang #44883](https://github.com/sgl-project/sglang/pull/44883) 11 月 13 日关闭且未合并；不作为已落地能力。

## 七、下周跟踪（11 月 20～27 日）

1. **vLLM**：#66944 是否合并及其对 GDN/Mamba2、Qwen3.8 系列的覆盖；#67840 planned receive 的主干化；#68578 单 rank IPC 同步修复。
2. **vLLM NCCL M2N**：#62853 是否完成 PP trainer 分组验证；#68132 FT 缩容后的 NCCL/UCX 显存回收。
3. **SGLang**：#46208 NPU IPC 的 CANN 版本兼容、跨 NPU 映射和多 rank E2E；#51237 target/draft IPC OOM 修复；#50229 事务错误传播。
4. **性能复测**：针对 A3/A5，单独测 `indices/values` 聚合通信耗时、HCCL broadcast 数量、pack/index_copy、显存峰值及 0.1%～50% sparsity；避免将 H20/GB200 作者数据直接迁移到 Ascend。
5. **拓宽检索**：继续检查 Issue/RFC、commit 与 RDMA/WeightBridge/Connex/sparse_disk 关键词，避免只依赖 PR 标题导致漏报。

---

**证据限制**：本报告的状态、创建/更新时间来自 GitHub PR 元数据，测试与性能来自作者 PR 描述；本次未逐项核对全部 commit diff、Issue/RFC 讨论时间线或独立运行 benchmark。部分 PR 最近更新时间可能来自评论、rebase 或编辑，不能等同新增功能。统计为重点进展清单，非所有相关 PR 的穷举。
