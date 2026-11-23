# vLLM / SGLang 社区进展：IPC、NCCL 与 Sparse NCCL

- **统计区间：** 2026-10-30 至 2026-11-06（含）
- **统计截止：** 2026-11-06（UTC+8）
- **仓库：** `vllm-project/vllm`、`sgl-project/sglang`
- **口径：** 统计本期新建、更新或合并的 PR / RFC /
  Issue。较早创建但本期有实质更新的项目标注为“延续跟踪”。
- **状态：** 以本次核查时 GitHub 页面为准。性能数字来自 PR
  作者公开测试，不代表本报告独立复测。

------------------------------------------------------------------------

## 1. 执行摘要

1.  **vLLM Sparse NCCL 有直接性能优化。**
    [\#66021](https://github.com/vllm-project/vllm/pull/66021) 将逐
    patch 校验改为批量验证。Qwen3-30B-A3B、TP4、H20、18,867 个 tensor
    测试中，作者报告 pause 时间降低 **40%–48%**。PR 于 11 月 6
    日新建，尚未合并。
2.  **vLLM 统一权重更新后的模型处理。**
    [\#65914](https://github.com/vllm-project/vllm/pull/65914) 于 11 月
    6 日合并。Sparse NCCL、kernel-format reload 等路径统一通过
    `finalize_model_weights()` 执行 post-load
    hook，修复部分派生权重未刷新的问题。
3.  **vLLM ROCm IPC collective 持续推进。**
    [\#63890](https://github.com/vllm-project/vllm/pull/63890) 使用
    AITER IPC all-gather 替代部分 RCCL TP
    all-gather；[\#65628](https://github.com/vllm-project/vllm/pull/65628)
    为 Qwen4Exp Sequence Parallel 增加 AITER IPC reduce-scatter /
    all-gather 和可选 INT4 QuickReduce。两项均未合并。
4.  **vLLM NCCL 可靠性和分片加载架构演进。**
    [\#64374](https://github.com/vllm-project/vllm/issues/64374)
    讨论进度超时、失败中止和整轮重试；[\#65235](https://github.com/vllm-project/vllm/pull/65235)
    已合并 `plan_loads` / `TraceTensor`，为 NCCL M2N、sharded RDT
    等提供统一加载计划。
5.  **SGLang IPC 重点是正确性和设备扩展。**
    [\#48485](https://github.com/sgl-project/sglang/pull/48485)
    已合并，修复 Mooncake HIP IPC
    子分配偏移错误；[\#41495](https://github.com/sgl-project/sglang/pull/41495)
    推进 Ascend `torch_npu` 零拷贝
    IPC；[\#46953](https://github.com/sgl-project/sglang/pull/46953)
    推进 CUDA VMM + POSIX fd 权重共享。
6.  **SGLang 新增 NCCL EP，但不是 Sparse NCCL。**
    [\#47600](https://github.com/sgl-project/sglang/pull/47600) 增加可选
    `nccl_ep` MoE dispatch/combine 后端。
7.  **SGLang 本期未发现直接实现 Sparse NCCL 的 PR。**
    [\#42210](https://github.com/sgl-project/sglang/pull/42210) 的 GPU
    delta / Bitcomp 是压缩差分 + GPU decode + 原地 apply，不是 Sparse
    NCCL。

**一句话结论：** 本周 vLLM 的 Sparse NCCL 有直接性能优化；SGLang
主要推进 IPC 正确性、NPU IPC、VMM fd、NCCL EP 和 GPU delta
更新。两边都在从“把数据传过去”转向同时保证失败可恢复、模型状态正确，以及通信与推理更好地重叠。

## 2. 术语边界

| 名称                 | 本报告中的含义                                                 | 不等同于                       |
|----------------------|----------------------------------------------------------------|--------------------------------|
| CUDA / HIP / NPU IPC | 跨进程共享设备内存或导入设备 tensor storage                    | NCCL collective                |
| VMM + POSIX fd       | 通过设备虚拟内存映射和文件描述符共享 allocation                | 传统 CUDA IPC handle           |
| Dense NCCL           | 传输完整 tensor / bucket / weight                              | Sparse NCCL                    |
| Sparse NCCL          | 传输 indices + values 等 sparse weight patch，并在接收端 apply | 稀疏注意力、稀疏 KV cache      |
| NCCL EP              | NCCL Expert Parallel dispatch / combine                        | trainer-to-rollout weight sync |
| NCCL M2N             | 不同 trainer / inference GPU mesh 间重分片传输                 | Sparse NCCL                    |
| GPU delta / Bitcomp  | 压缩权重差分并在 GPU 上解码、原地 apply                        | Sparse NCCL                    |

本报告中的 Sparse NCCL 只统计权重 patch 传输、验证、apply
和生命周期相关改动，不把 sparse MLA / DSA attention、QSA cache sharding
算作 Sparse NCCL。

------------------------------------------------------------------------

## 3. vLLM 社区进展

### 3.1 Sparse NCCL：批量验证 sparse weight patches

**PR：** [\#66021 — Validate sparse weight patches in one
pass](https://github.com/vllm-project/vllm/pull/66021)  
**作者：** `ShuoleiWang`  
**创建：** 2026-11-06 09:09（UTC+8）  
**更新：** 2026-11-06 14:22（UTC+8）  
**状态：** Open，非 Draft，尚未合并。

旧实现逐个 patch 执行 NaN、indices 范围、排序和重复检查，每个 patch
约触发十个小型 GPU 操作。模型 tensor 数量较多时，小操作启动开销会成为
pause 阶段的重要组成部分。

本 PR 将多个 patch 合并为批次统一校验。

#### 作者公开结果

环境：Qwen3-30B-A3B、TP4、H20、18,867 个 tensor；32 个并发 streaming
client。

| 更新比例 | 传输 payload | main pause | PR pause | pause 降幅 |
|---------:|-------------:|-----------:|---------:|-----------:|
|     0.1% |     0.17 GiB |     5.75 s |   3.06 s |    **47%** |
|       1% |     1.71 GiB |     7.14 s |   3.73 s |    **48%** |
|       3% |     5.12 GiB |     8.99 s |   5.35 s |    **40%** |

作者另报告：

- 单 tensor 平均验证耗时约从 125 µs 降至 5 µs；
- 同一 engine 交替测试的 30 组对比全部更快；
- `/weight_checker` 的 1,740 个权重摘要和 greedy per-token logprobs
  一致；
- patch 校验单测 31 passed、reload 测试 1 passed、weight-transfer 测试
  20 passed / 1 skipped；
- 120,000 轮随机 CPU 校验和 21 种故障注入与旧实现判定一致。

**解读：** 这是本周最直接的 Sparse NCCL 性能进展，但优化点是 patch
validation，不是 NCCL broadcast 算法或 sparse wire format。pause
数据覆盖 pause → send → resume 周期，不应当作纯网络传输耗时。作者也注明
serving 吞吐测试的更新频率不同，不能直接解释为同等更新频率下的吞吐提升。

### 3.2 Sparse NCCL 更新后的派生权重刷新统一

**PR：** [\#65914 — One finish for every update path; drop
refresh_derived_weights](https://github.com/vllm-project/vllm/pull/65914)  
**作者：** `aoshen02`  
**合并：** 2026-11-06 13:59（UTC+8）  
**状态：** Merged。

旧的不同更新路径结束处理并不统一：

| 更新路径                   | 量化处理   | Attention 类层处理 | 其它模块 hook |
|----------------------------|------------|--------------------|---------------|
| Cold start / direct reload | 有         | 有                 | 有            |
| Layerwise reload           | 有         | 有                 | 有            |
| Sparse NCCL finish         | 不完整     | **缺失**           | 有            |
| Kernel-format reload       | 不重复量化 | **缺失**           | **缺失**      |

例如 Kimi-K3 ROCm 的 KDA 层会从权重派生 decode copy。Sparse NCCL
原地写入权重后，如果不重新执行 hook，decode kernel 可能继续读取旧值。

本 PR 统一通过 `finalize_model_weights()` 执行
`process_weights_after_loading()`：

- 所有模块使用 children-first traversal；
- Sparse NCCL finish 只需调用统一 finalize；
- kernel-format `reload_weights` 也接入统一 finalize；
- 删除临时增加的第二套 `refresh_derived_weights()` hook；
- 移除部分 forward-time lazy build。

作者报告 MI355X ROCm 的 Kimi-K3 / reload 相关测试 137 passed、17
skipped、1 xpassed；两个 NVFP4A16 DeepSeek-V3-debug 测试在 base
上也失败，不属于本 PR 新问题。

**解读：**
不减少通信字节，但解决“权重已更新、派生状态仍旧”的一致性问题。

### 3.3 Kimi-K3 ROCm 派生权重修复前序 PR

**PR：** [\#64886 — Refresh derived weight copies after in-place weight
writes](https://github.com/vllm-project/vllm/pull/64886)  
**合并：** 2026-11-05；后续统一方案由 \#65914 接管。

曾为
`SparseNCCLWeightTransferEngine.finish_weight_update`、kernel-format
reload 和 Elastic EP 增加 `refresh_derived_weights()` hook。主要修复
Kimi-K3 ROCm 的 `decode_conv1d_weight`、`decode_norm_weight` 和
`_w_fold`。其中 `_w_fold` 必须原地写入，避免 CUDA Graph 继续引用旧地址。

### 3.4 NCCL 权重更新超时、失败中止和整轮重试 RFC

**Issue：** [\#64374 — Bounded, retryable NCCL weight
updates](https://github.com/vllm-project/vllm/issues/64374)  
**创建：** 2026-10-29；**本期更新：** 2026-11-05  
**状态：** Open RFC，尚未实现为合并 PR。

当前 NCCL 权重更新中，如果 worker 中途退出或 RPC 没有送达，trainer
和其它 worker 可能一直卡在 collective。现有 rollback 需要 update
抛异常；collective 卡死时不会抛异常，因此无法恢复。

RFC 方案：

1.  增加 `stall_timeout_s`，基于最近一次进度而非整轮更新时间；
2.  slot / event 等待采用 query + timeout；
3.  出错时 abort communicator；
4.  worker 执行 layerwise rollback；
5.  trainer 重新初始化 communicator，从头发送完整权重；
6.  权重不一致时禁止 generation，必须完整重传并 commit。

讨论中倾向按 backend 区分：broadcast 可考虑 deadline + abort +
retry；M2N / GIN 的设备端工作在 abort 时更复杂，不能假设普通
`ncclCommAbort` 一定中止所有 kernel。RFC 仍需明确失败传播、超时默认值和
M2N 恢复语义。

### 3.5 NCCL M2N / Sharded RDT 的统一 Load Plan

**PR：** [\#65235 — Load plan: plan_loads and
TraceTensor](https://github.com/vllm-project/vllm/pull/65235)  
**合并：** 2026-11-06；**状态：** Merged。

不同 rank 需要知道 trainer tensor 中哪些区域属于自己，以及数据应写到
live model 的什么位置。此前 NCCL M2N 和 sharded RDT
各自维护参数分片规则。

本 PR 新增：

- `TraceTensor`：追踪模型 loader 的 view / copy 行为；
- `plan_loads(model, metadata)`：生成每个 rank 的 load plan；
- `get_boxes(name)`：返回当前 rank 需要的 checkpoint tensor 区域；
- `get_dest(name)`：返回目标 live storage 区域；
- `load_planned`：按计划写入。

NCCL broadcast engine 可 opt-in 使用；NCCL M2N 和 sharded RDT
后续可复用。

作者测试 Qwen3-30B-A3B BF16：首次 sync 7.4 s，其中 trace 4.2 s；后续
sync 中位数 2.9 s。

**解读：** 这是分片加载架构进展，不是 Sparse NCCL 算法优化。

### 3.6 IPC / NCCL 权重更新失败后的 rollback

**PR：** [\#57854 — Fail a weight update without taking the engine
down](https://github.com/vllm-project/vllm/pull/57854)  
**创建：** 2026-09-21；**本期更新：** 2026-11-06  
**状态：** Open，延续跟踪。

覆盖 IPC、NCCL、NCCL M2N、Sparse NCCL 和 sharded RDT 等更新路径。

主要解决：

- 权重仍处于 sleep / unmap 状态时发起 update，可能触发非法设备拷贝；
- layerwise reload 的某个 chunk 失败后，未完成 layer 可能仍保留 meta
  tensor，导致后续 forward 崩溃。

本 PR 增加 sleep 状态 guard 和
`abort_layerwise_reload()`，失败时恢复未完成 layer 的原始 kernel
tensors，清理 partial weights 与 reload metadata。

### 3.7 ROCm AITER IPC all-gather

**PR：** [\#63890 — Use AITER's all-gather for small TP
all-gathers](https://github.com/vllm-project/vllm/pull/63890)  
**创建：** 2026-10-27；**本期更新：** 2026-11-05  
**状态：** Open。

在 ROCm gfx950 / gfx942 上，如果 TP communicator 已使用 AITER custom
all-reduce，则符合条件的小尺寸 TP all-gather 使用 AITER IPC all-gather
替代 RCCL。

| TP size | 每 rank 尺寸上限 |
|--------:|-----------------:|
|     TP2 |           32 MiB |
|     TP4 |           16 MiB |
|     TP8 |            8 MiB |

作者报告 MI355X gfx950：

- TP4 decode c1–c4：约快 1.2%–1.9%；
- TP4 c8：约快 1.4%，但作者提示该点波动较大；
- TP8 decode c1–c4：约快 0.8%–1.2%；
- 768 个 sweep 点覆盖不同维度、dtype、eager 和 graph replay；
- AITER / RCCL 输出 bit-exact；
- GSM8K 1319 题平均准确率均为 0.916。

**解读：** 这是 TP all-gather 的 IPC collective 优化，不是 Sparse NCCL
权重传输。

### 3.8 Qwen4Exp Sequence Parallel：AITER IPC + INT4 QuickReduce

**PR：** [\#65628 — Opt-in exact AITER IPC and INT4 QuickReduce
collectives](https://github.com/vllm-project/vllm/pull/65628)  
**创建：** 2026-11-04；**最近更新：** 2026-11-06  
**状态：** Draft。

面向 ROCm gfx950、TP2、Qwen4Exp 的
`--enable-hc-sp`，增加默认关闭的配置：

- `--enable-hc-sp-aiter-ipc`：AITER exact IPC reduce-scatter /
  all-gather；
- `--enable-hc-sp-quick-reduce`：大尺寸走 FlyDSL INT4
  QuickReduce，小尺寸 decode 仍走 exact IPC。

作者报告，只开 HC-SP 时，98 次 AITER custom all-reduce 被替换为 195 次
RCCL collective，ITL 回退 7%–18%。启用 IPC 后：

| 工作负载           | HC-SP + RCCL 相对默认 ITL | HC-SP + IPC 相对默认 ITL |
|--------------------|--------------------------:|-------------------------:|
| W2 concurrency 16  |                    +18.0% |                    +7.6% |
| W2 concurrency 64  |                    +11.2% |                    -0.4% |
| W2 concurrency 256 |                     +7.2% |                    -0.3% |
| W1 concurrency 32  |                    +15.5% |                    +4.8% |

INT4 QuickReduce 在 1,633–8,192 rows 大 step 上，将作者测试的 GPU busy
time 从 IPC-only 的 224.3 ms 降到 174.0 ms，约降低 22%。

限制：当前只验证 TP2 / gfx950；gfx942、TP4/TP8 和 MTP draft acceptance
尚未验证。INT4 是有损路径，作者 GSM8K
样本没有观察到明确精度损失，但不能外推为所有模型均无风险。

### 3.9 PCIe IPC all-reduce P2P fallback

**PR：** [\#65192 — Fall back from FlashInfer PCIe IPC all-reduce
without peer access](https://github.com/vllm-project/vllm/pull/65192)  
**创建：** 2026-11-02；**更新：** 2026-11-06  
**状态：** Open。

当设置 `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`，但多 GPU
主机不具备完整 CUDA peer access 时，旧实现可能在初始化阶段失败。本 PR
增加 peer-access 检查，并让所有 rank 协同决定是否启用 IPC；不满足条件时
warning 并回退到后续 backend。

作者在 RTX PRO 6000 Blackwell、L40S、L4、A10G 等环境验证回退行为，CPU
单测 11 passed。

**解读：** 这是 IPC 拓扑适配与可靠性修复，不是性能提升。

### 3.10 LMCache MP 使用 VMM POSIX fd 扩展 KV cache

**PR：** [\#65507 — Enable LMCache MP connector for the extensible KV
cache](https://github.com/vllm-project/vllm/pull/65507)  
**创建：** 2026-11-04；**更新：** 2026-11-06  
**状态：** Open。

该 PR 让 LMCache MP 在 extensible KV cache 下通过 CUDA VMM / POSIX fd
共享 allocation。作者在单 H200、Qwen3-0.6B 上报告：

| 配置                   | GPU KV cache 容量 |
|------------------------|------------------:|
| 传统 CUDA IPC          |  1,172,144 tokens |
| Extensible KV + VMM fd |  1,256,960 tokens |
| 变化                   |         **+7.2%** |

这是 KV cache 共享，不是权重 Sparse NCCL；没有提供 VMM fd 与传统 IPC
的独立传输耗时对比。

------------------------------------------------------------------------

## 4. SGLang 社区进展

### 4.1 Mooncake HIP IPC 子分配偏移修复

**PR：** [\#48485 — Bump Mooncake to pick up the HIP IPC sub-allocation
fix](https://github.com/sgl-project/sglang/pull/48485)  
**创建：** 2026-11-05；**合并：** 2026-11-05  
**状态：** Merged。

旧 Mooncake HIP IPC 导出整个 `hipMalloc` allocation 的 handle，但登记
buffer 时使用 tensor 子分配地址作为 base。若 tensor 不是 allocator
segment 的第一个对象，peer 映射后就会按错误 offset 读取。transfer
可能返回成功，但权重数据错误。

本 PR 升级 Mooncake 到包含 `Mooncake#3548` 的 commit，按 allocation
粒度登记地址。作者报告旧实现 AMD `transfer_engine` CI 失败 10/10，修复后
MI300 ROCm 10 的 Engine / Server 测试通过。

**判断：** 本周 SGLang 最重要的 IPC 数据正确性修复之一，说明必须同时验证
allocation base、storage offset 和实际读写内容。

### 4.2 Ascend NPU Weight Cache 零拷贝 IPC

**PR：** [\#41495 — Add NPU weight cache
support](https://github.com/sgl-project/sglang/pull/41495)  
**创建：** 2026-09-28；**最近更新：** 2026-11-05  
**状态：** Open，延续跟踪。

CUDA weight cache 依赖 `_share_cuda_`，Ascend 则使用 `torch_npu` 注册的
`reduce_tensor` / `rebuild_npu_tensor`。本 PR 通过原生 multiprocessing
reducer 复用现有 `TorchIpcTransportBackend`，不需要独立 NPU backend。

平台限制：

- 零拷贝需要 HDK ≥ 25.3.RC1；
- CANN ≥ 8.3.RC1；
- **Ascend 950DT 不支持该零拷贝路径**，需要 fallback 到 CPU copy；
- reducer probe 不成功时 warning 并使用 CPU-copy 路径。

新增 Qwen3-32B TP2 startup test，比较 cold disk load、daemon disk load
和 warm client IPC
import。作者表示零拷贝路径不会引入额外数据拷贝，但没有提供 per-token
throughput 提升数据。

### 4.3 CUDA VMM + POSIX fd：Weight Cache 新传输方式

**PR：** [\#46953 — Add the vmm_fd
transport](https://github.com/sgl-project/sglang/pull/46953)  
**创建：** 2026-10-28；**更新：** 2026-11-06  
**状态：** Open。

daemon 将权重 pack 到少量可导出的 VMM allocation，通过 Unix socket
`SCM_RIGHTS` 发送 POSIX fd；engine 预留 VA、导入 fd 并映射
allocation，再接入现有 tensor loader。

核心能力：daemon 被 SIGKILL 后，已经 attach 的 engine
仍可以继续使用映射中的权重。

作者在 B200 上覆盖 Qwen3-0.6B、Qwen3-30B-A3B TP / EP / PP
等配置，报告参数和 buffer 的 shape、dtype、stride、storage offset /
size、SHA-256 与 disk / torch_ipc 一致，daemon 被杀后 greedy token
仍一致。

限制：

- POSIX fd 方案仅同节点；
- 尚未实现释放后在相同地址重新 attach；
- weight cache active 时仍拒绝 weight update；
- import time 尚未测量；
- rebase 后主要重跑 CPU tests 和 lint，GPU parity 数据来自较早
  revision。

### 4.4 UMBP GPU buffer 注册设备类型修复

**PR：** [\#48764 — Register UMBP GPU buffers with device
metadata](https://github.com/sgl-project/sglang/pull/48764)  
**创建：** 2026-11-06  
**状态：** Draft。

UMBP direct linker 原来只注册 pointer 和 size，MoRI embedded client
默认将其解释为 CPU memory，导致选择 CPU memcpy，而非 GPU transfer
engine。

本 PR 显式传递 `MemoryLocationType.GPU` 和 local device index，并补充
allocation-base 去重、host buffer 拒绝和注册失败处理。

作者给出历史压力测试：MI355X output tok/s 从 31.65 / 31.76 到 183.09 /
183.74；MI455X 从 155.95 / 152.84 到 162.00 / 161.59。但作者说明 MI355X
结果还包含其它集成变化，不能将全部收益归因于本 PR。完整 upstream CI
仍待完成。

### 4.5 Pipeline Parallel 下 CUDA IPC payload rank 映射修复

**PR：** [\#48657 — One tensor weight payload per engine rank under
pipeline
parallelism](https://github.com/sgl-project/sglang/pull/48657)  
**创建：** 2026-11-05  
**状态：** Open。

旧实现按 TP rank 发送 CUDA IPC payload。PP\>1 时，不同 stage 的同一 TP
rank 可能打开同一个 handle，读取到前一 stage 的参数；多个 consumer
共用一个 IPC handle 也会破坏 producer 侧 refcount。

本 PR 改为每个 engine rank 一个 payload：

``` text
payload index = TP * PP_rank + TP_rank
```

作者在 4×H100、Qwen3-0.6B、PP2×TP2 验证：修复前 stage 1
权重没有更新，并出现 shared CUDA tensor release 错误；修复后 stage 1
`down_proj` 权重正确变化。

### 4.6 自定义平台 Weight Transport 插件化

**PR：** [\#48433 — Let platforms serve weight transports through
/plugin_weight_update](https://github.com/sgl-project/sglang/pull/48433)  
**创建：** 2026-11-05；**更新：** 2026-11-06  
**状态：** Open。

将 weight update 拆成 receive transport 和 load policy：

- transport 负责把 tensor bucket 传到当前 TP rank；
- core 负责模型权重写入、derived weight、draft runner、version、cache
  flush 等。

平台可通过 `SRTPlatform.get_weight_update_handlers()` 注册自己的
transport，支持 ICI、P2P / RDMA、shared memory、disk、DMA engine
等方式。

B200 devbox 上 scheduler weight-version tracking 测试 32
passed。没有单独性能 benchmark。

### 4.7 NCCL EP：MoE A2A 新后端

**PR：** [\#47600 — Add NCCL EP as optional
moe-a2a-backend](https://github.com/sgl-project/sglang/pull/47600)  
**创建：** 2026-10-31；**更新：** 2026-11-06  
**状态：** Open，非 Draft。

新增可选参数：

``` bash
--moe-a2a-backend nccl_ep
```

支持 `expert_major` 和 `rank_major` 两种布局。rank-major 保留 NCCL
source-rank slot 顺序，减少 wire 上的重排，并使用 Triton fused gather /
route-weight / fp32 reduce。

运行要求：

- `nccl4py[cu13]`；
- NCCL ≥ 2.29。

后续还需要 HT prefill、CUDA Graph、EPLB、SBO / TBO 和 deterministic
fallback。当前 PR 的 Base CI 因缺少 `run-ci` label 尚未启动，PR
描述没有提供完整端到端性能结果。

**判断：** 这是 NCCL Expert Parallel dispatch / combine，不是 Sparse
NCCL 权重更新。

### 4.8 Gated Background Transfers

**PR：** [\#45945 — Gated background
transfers](https://github.com/sgl-project/sglang/pull/45945)  
**合并：** 2026-11-06  
**状态：** Merged。

为低优先级后台传输增加 chunk-based gate：

- 后台 KV gather、PCIe H2D / D2H 等任务在独立 stream 上分块；
- 高优先级计算或 collective 可暂停后台任务；
- 后台任务在下一个 chunk boundary 停止；
- 高优先级操作完成后自动恢复。

作者在 8×H200、CUDA 13.0、NCCL 2.30.7 上测得：每 chunk gate wait 约 2.0
µs；pause + resume 约 4.7 µs（device word）或 5.7 µs（host
word）。测试覆盖 2–4 GPU 的 NCCL broadcast / all-gather chunk，以及
foreground all-reduce 暂停后台 NCCL chunk。

**判断：** 这是通信调度和 overlap 控制，不是 NCCL 算法或 payload 优化。

### 4.9 ROCm Custom All-Reduce V2

**PR：** [\#47964 — Run the JIT custom all-reduce v2 on
ROCm](https://github.com/sgl-project/sglang/pull/47964)  
**创建：** 2026-11-03；**更新：** 2026-11-06  
**状态：** Open。

该 PR 将 JIT Lamport push / pull 算法移植到 ROCm gfx950，并通过 HIP
VMM + POSIX fd 建立 rank 间 symmetric
workspace。8×MI350X、TP8、BF16、CUDA Graph 测试中：

|   Size |     RCCL |   AITER | carv2 最优 | carv2 vs AITER |
|-------:|---------:|--------:|-----------:|---------------:|
|   4 KB |  14.5 µs |  5.3 µs |     4.2 µs |           -20% |
|  16 KB |  14.4 µs |  5.6 µs |     4.4 µs |           -23% |
|  64 KB |  14.3 µs |  8.7 µs |     5.5 µs |           -37% |
| 128 KB |  14.1 µs |  7.3 µs |     7.1 µs |            -3% |
| 256 KB |  16.7 µs |  7.6 µs |     7.8 µs |            +3% |
| 512 KB |  21.7 µs |  8.7 µs |     8.7 µs |             0% |
|   1 MB |  25.0 µs | 11.3 µs |    10.3 µs |            -9% |
|   2 MB |  31.2 µs | 16.7 µs |    16.4 µs |            -2% |
|   4 MB |  48.5 µs | 25.8 µs |    25.3 µs |            -2% |
|   8 MB |  73.1 µs | 44.5 µs |    49.0 µs |           +10% |
|  16 MB | 112.0 µs | 82.6 µs |    90.8 µs |           +10% |

作者报告 carv2 所有尺寸均快于 RCCL；相对 AITER
的优势主要集中在小尺寸和部分中等尺寸。

**重要限制：** 本 PR 没有改变 ROCm 生产 dispatch，当前 serving 仍然选择
AITER。因此这些数字不代表默认 SGLang serving 已获得对应收益。

### 4.10 GPU Delta / Bitcomp：稀疏压缩差分，但不是 Sparse NCCL

**PR：** [\#42210 — Add GPU delta decode and in-place weight
updates](https://github.com/sgl-project/sglang/pull/42210)  
**创建：** 2026-10-02；**更新：** 2026-11-06  
**状态：** Open，延续跟踪。

路线是：pause 前准备 payload；进入 update session 后 fence reader；使用
nvCOMP GPU decode 解压 delta；通过 Triton 对现有 inference buffer 原地
XOR / apply；最后更新 version 并恢复 generation。

作者在 8×B300、GLM-5.2 NVFP4 W4A16 proxy 上报告：

| Codec        | Warm update |    Pause |
|--------------|-------------|---------:|
| Snappy–Zstd  | v2          | 232.3 ms |
| Snappy–Zstd  | v3          | 240.9 ms |
| Bitcomp–Zstd | v2          | 168.7 ms |
| Bitcomp–Zstd | v3          | 173.7 ms |

Bitcomp–Zstd stored payload 约比 Snappy–Zstd 小 61.9%。Bitcomp/default
配置 507 个 focused tests 通过。

限制：三次 update 是累计版本，不是独立重复实验；没有最终 rollout
throughput / training-step speedup。Bitcomp delta
是压缩差分路径，不能计入 Sparse NCCL。

### 4.11 Out-of-process HiCache Device IPC RFC

**Issue：** [\#37372 — Out-of-process HiCache data plane with
device-memory IPC](https://github.com/sgl-project/sglang/issues/37372)  
**创建：** 2026-09-01；**最近更新：** 2026-11-01  
**状态：** Open RFC，本期未发现对应实现 PR。

提议把 HiCache data plane 从 scheduler 进程拆出，使用 CUDA IPC（初期
NVIDIA）让独立 daemon 直接映射 scheduler 的 GPU KV allocation。Scheduler
继续负责 cache metadata、prefix match、lock/refcount；daemon 负责
GPU↔Host transfer、异步队列、host cache 和外部 L3。

这是 IPC 架构方向讨论，不是本期已落地的性能改进。

------------------------------------------------------------------------

## 5. vLLM 与 SGLang 横向比较

| 维度                        | vLLM                                                                 | SGLang                                                   |
|-----------------------------|----------------------------------------------------------------------|----------------------------------------------------------|
| 直接 Sparse NCCL            | \#66021 批量 patch validation，pause 降低 40%–48%                    | 本期未检索到直接 Sparse NCCL 实现 PR                     |
| Sparse NCCL 更新后正确性    | \#65914 统一 finalize，刷新派生权重                                  | 无对应 Sparse NCCL 路径                                  |
| Dense NCCL / reliability    | \#64374 讨论 timeout / abort / retry；#57854 处理 rollback           | \#47600 推进 NCCL EP                                     |
| NCCL M2N / sharded transfer | \#65235 Load Plan 已合并                                             | NCCL 2.30 roadmap 继续推进                               |
| CUDA IPC                    | \#65192 P2P fallback；#65507 VMM fd KV cache                         | \#46953 VMM fd weight cache；#48657 PP payload rank 修复 |
| ROCm IPC                    | \#63890 AITER IPC all-gather；#65628 IPC reduce-scatter / all-gather | \#48485 修复 Mooncake HIP IPC 子分配偏移                 |
| Ascend NPU IPC              | 本期未发现对应主仓新实现                                             | \#41495 推进 torch_npu zero-copy IPC                     |
| 后台通信 overlap            | 主要关注 slot / stream 生命周期                                      | \#45945 gated background transfers 已合并                |
| 非 NCCL 差分更新            | Sparse NCCL 仍是主要方案                                             | \#42210 GPU delta / Bitcomp，非 Sparse NCCL              |

------------------------------------------------------------------------

## 6. 本周结论与技术判断

### 6.1 Sparse NCCL

vLLM 本周的重点不是改变 sparse wire format，而是降低每个 patch
的验证成本，并保证更新完成后模型派生状态同步刷新。#66021、#65914、#64886、#57854、#64374
分别覆盖 patch 校验、post-load、派生状态、rollback 和 retry。

SGLang 本周没有直接 Sparse NCCL 权重传输实现。GPU delta / Bitcomp
是另一种压缩差分路线。

### 6.2 IPC

IPC 工作从“能共享”转向“共享正确、可恢复、跨设备”：

1.  地址和 offset 正确性：Mooncake HIP IPC；
2.  设备适配：torch_npu IPC、UMBP GPU metadata；
3.  生命周期：VMM fd，daemon 退出后 engine 仍能使用映射；
4.  rank ownership：PP 下每个 engine rank 独立 IPC payload；
5.  collective：AITER IPC all-gather / reduce-scatter。

### 6.3 NCCL

vLLM 重点在分片加载、失败检测、rollback 和 retry；SGLang 重点在 NCCL
EP、后台通信调度和 ROCm custom all-reduce。

### 6.4 性能数据不能直接横向比较

测试涉及不同模型、硬件和 workload：

- vLLM Sparse NCCL：Qwen3-30B-A3B、H20；
- vLLM AITER IPC：Qwen3.8 / DeepSeek-V4.1、MI355X；
- SGLang CustomAllReduceV2：MI350X；
- SGLang GPU delta：GLM-5.2、B300；
- SGLang Mooncake IPC：MI300 / ROCm。

这些数据适合说明各自路径的改进方向，不足以判断某个项目的 NCCL 或 IPC
整体带宽更高。

------------------------------------------------------------------------

## 7. 下一步重点跟踪

### P0：Sparse NCCL 与权重更新正确性

1.  **vLLM \#66021：** 是否合并；是否在更大模型、更多 TP/EP
    组合下验证批量 validation。
2.  **vLLM \#65914：** Kimi-K3 / DeepSeek-V4 等模型的 post-load hook 和
    CUDA Graph 地址稳定性。
3.  **vLLM \#57854：** rollback 对多 TP / PP 和真实 IPC / Sparse NCCL
    trainer 的覆盖。
4.  **vLLM \#64374：** progress-based timeout、NCCL abort 语义和整轮
    retry 是否拆分 broadcast / M2N。

### P1：IPC 和设备互联

5.  **vLLM \#63890 / \#65628：** AITER IPC 在 gfx942、TP4/TP8
    和更大并发下的稳定性；注意 \#65628 的 INT4 有损路径。
6.  **SGLang \#41495：** 核实 Ascend 950DT 的 CPU-copy fallback
    与实际部署限制。
7.  **SGLang \#46953：** VMM fd import latency、release / re-attach 和
    weight update 支持。
8.  **SGLang \#48764：** UMBP GPU registration 的完整 upstream CI。
9.  **SGLang \#48657：** PP 多 stage、多个 rank 和 producer lifetime 的
    CI。

### P2：NCCL / 非 NCCL 通信路线

10. **SGLang \#47600：** NCCL EP 的 HT prefill、CUDA Graph、EPLB
    和端到端 benchmark。
11. **SGLang \#47964：** CustomAllReduceV2 何时进入 ROCm serving
    dispatch；目前 benchmark 不代表默认生产路径。
12. **SGLang \#45945：** gated background transfer 与真实 KV offload /
    NCCL workload 的 overlap 收益。
13. **SGLang \#42210：** Bitcomp delta 是否提供更多重复测试和真实 RL
    training throughput 数据。

------------------------------------------------------------------------

## 8. PR / RFC 来源索引

### vLLM

| 编号    | 主题                                | 状态   | 链接                                                       |
|---------|-------------------------------------|--------|------------------------------------------------------------|
| \#66021 | Sparse patch 批量验证               | Open   | [PR](https://github.com/vllm-project/vllm/pull/66021)      |
| \#65914 | 统一 post-load finalize             | Merged | [PR](https://github.com/vllm-project/vllm/pull/65914)      |
| \#64886 | Kimi-K3 ROCm 派生权重刷新           | Merged | [PR](https://github.com/vllm-project/vllm/pull/64886)      |
| \#64374 | NCCL bounded / retryable update RFC | Open   | [Issue](https://github.com/vllm-project/vllm/issues/64374) |
| \#65235 | Load Plan / TraceTensor             | Merged | [PR](https://github.com/vllm-project/vllm/pull/65235)      |
| \#57854 | 更新失败 rollback                   | Open   | [PR](https://github.com/vllm-project/vllm/pull/57854)      |
| \#63890 | AITER IPC TP all-gather             | Open   | [PR](https://github.com/vllm-project/vllm/pull/63890)      |
| \#65628 | Qwen4Exp AITER IPC / QuickReduce    | Draft  | [PR](https://github.com/vllm-project/vllm/pull/65628)      |
| \#65192 | PCIe IPC all-reduce P2P fallback    | Open   | [PR](https://github.com/vllm-project/vllm/pull/65192)      |
| \#65507 | LMCache MP + VMM fd                 | Open   | [PR](https://github.com/vllm-project/vllm/pull/65507)      |

### SGLang

| 编号    | 主题                               | 状态     | 链接                                                        |
|---------|------------------------------------|----------|-------------------------------------------------------------|
| \#48485 | Mooncake HIP IPC 子分配偏移修复    | Merged   | [PR](https://github.com/sgl-project/sglang/pull/48485)      |
| \#41495 | Ascend NPU weight-cache IPC        | Open     | [PR](https://github.com/sgl-project/sglang/pull/41495)      |
| \#46953 | CUDA VMM + POSIX fd weight cache   | Open     | [PR](https://github.com/sgl-project/sglang/pull/46953)      |
| \#48764 | UMBP GPU buffer metadata           | Draft    | [PR](https://github.com/sgl-project/sglang/pull/48764)      |
| \#48657 | PP 每 engine rank 一个 IPC payload | Open     | [PR](https://github.com/sgl-project/sglang/pull/48657)      |
| \#48433 | Platform plugin weight transport   | Open     | [PR](https://github.com/sgl-project/sglang/pull/48433)      |
| \#47600 | NCCL EP MoE backend                | Open     | [PR](https://github.com/sgl-project/sglang/pull/47600)      |
| \#45945 | Gated background transfers         | Merged   | [PR](https://github.com/sgl-project/sglang/pull/45945)      |
| \#47964 | ROCm CustomAllReduceV2             | Open     | [PR](https://github.com/sgl-project/sglang/pull/47964)      |
| \#42210 | GPU delta / Bitcomp weight update  | Open     | [PR](https://github.com/sgl-project/sglang/pull/42210)      |
| \#37372 | Out-of-process HiCache IPC RFC     | Open RFC | [Issue](https://github.com/sgl-project/sglang/issues/37372) |

------------------------------------------------------------------------

## 9. 最终结论

截至 2026-11-06：

- **Sparse NCCL：** vLLM \#66021 是本周最直接的性能优化，patch
  validation pause 开销降低约 40%–48%；#65914
  补齐权重更新后的模型状态刷新。SGLang 本期没有直接 Sparse NCCL 实现。
- **IPC：** vLLM 推进 ROCm AITER IPC collective 和 PCIe IPC
  fallback；SGLang 修复 HIP IPC offset 错误，并继续推进 VMM fd、Ascend
  NPU IPC 和 PP rank payload ownership。
- **NCCL：** vLLM 重点在分片加载、失败检测、rollback 和 retry；SGLang
  重点在 NCCL EP 和后台通信调度。
- **替代路线：** SGLang GPU delta / Bitcomp 可以压缩差分 payload，但不是
  Sparse NCCL，不能混为同一类实现。
- **总体判断：** 当前最值得持续跟踪的是 vLLM Sparse NCCL patch
  validation、NCCL update failure recovery，以及 SGLang 在 HIP / NPU /
  VMM fd 上的 IPC 正确性和生命周期管理。
