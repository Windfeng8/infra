# vLLM / SGLang 社区周报：IPC、NCCL、Sparse NCCL 与 RDMA

-   **统计区间：** 2026-11-07 00:00 至 2026-11-13 12:00（UTC+8）
-   **复核时间：** 2026-11-21 09:20（UTC+8）
-   **仓库：**
    [vllm-project/vllm](https://github.com/vllm-project/vllm)、[sgl-project/sglang](https://github.com/sgl-project/sglang)
-   **统计口径：** 本期新建、更新、合并的 PR / Issue / RFC /
    commits；较早创建但本期有实质更新的项目列为"延续跟踪"。
-   **状态口径：** 同时给出本期截止时状态与本次复核时状态。合并发生在 11
    月 13 日 12:00 之后的，明确标注为"截止后合并"。
-   **性能口径：** 数字均来自 PR / Issue
    作者公开的测试描述；本报告没有在对应硬件上独立复测。作者测试、CI
    测试和独立验证分开表述。

------------------------------------------------------------------------

## 1. 执行摘要

### 本期最重要的变化

1.  **vLLM Sparse NCCL 从逐 tensor 传输走向 bounded batching。**
    -   [#66021](https://github.com/vllm-project/vllm/pull/66021) 于 11
        月 7 日合并：将 sparse patch 的 NaN、indices
        范围、排序和重复检查批量化，作者报告 pause 时间降低 40%--48%。
    -   [#66375](https://github.com/vllm-project/vllm/pull/66375) 于 11
        月 8 日创建，截止时仍 Open；11 月 14 日才合并。它将 indices /
        values 按最多 512 MiB 分批广播、逐批 apply，作者报告广播次数从
        37,734 降至 2--22 次，3% 更新比例下 `send_weights` 从 4.29 s
        降至 2.14 s。
    -   [#66944](https://github.com/vllm-project/vllm/pull/66944) 暴露
        GDN / Mamba2 sparse patch loader 的正确性问题：部分 loader 绕过
        `Tensor.copy_`，导致 NaN marker 被写入权重或 composed transform
        重复执行。
2.  **vLLM IPC 方向同时覆盖 HIP IPC 权重同步和 CUDA IPC KV pull。**
    -   [#67082](https://github.com/vllm-project/vllm/pull/67082) 修复
        ROCm 首次 HIP IPC 权重更新后 KFD queue eviction，作者报告首次
        `send_weights` 从 13.7--23.7 s 降至约 0.28 s。
    -   [#65639](https://github.com/vllm-project/vllm/pull/65639) 是同机
        NIXL + CUDA IPC KV pull，仍为 Draft；性能数字来自下游 H20
        移植观察，不是 upstream 分支的受控端到端对比。
3.  **vLLM ROCm AITER IPC collective 扩展。**
    -   [#63890](https://github.com/vllm-project/vllm/pull/63890)
        的小尺寸 TP AITER IPC all-gather 于 11 月 7 日 02:18 合并。
    -   [#67010](https://github.com/vllm-project/vllm/pull/67010) 增加
        AITER IPC reduce-scatter，仍为 Draft。
    -   [#65628](https://github.com/vllm-project/vllm/pull/65628) 推进
        Qwen4Exp sequence-parallel 的 AITER INT4 QuickReduce，属于有损
        collective 路径，不是 Sparse NCCL 权重传输。
4.  **SGLang 的重点是 NPU weight-cache IPC、消息 payload 去重、ROCm
    custom all-reduce 和 NPU HiCache。**
    -   [#41495](https://github.com/sgl-project/sglang/pull/41495) 于 11
        月 11 日合并，使用 `torch_npu` 原生 reducer 支持 Ascend NPU
        weight-cache IPC。
    -   [#48241](https://github.com/sgl-project/sglang/pull/48241) 于 11
        月 7 日合并，让 HTTP weight-sync 的一个序列化 payload 服务所有
        TP rank；这是消息 / 序列化层优化，不是 CUDA IPC。
    -   [#49001](https://github.com/sgl-project/sglang/pull/49001) 为
        MI450 增加 HIP FABRIC handles 和 gfx1250 custom all-reduce
        参数表；它不是 NCCL，默认 dispatch 仍是 AITER。
    -   [#48031](https://github.com/sgl-project/sglang/pull/48031) 将
        Ascend HiCache MLA / DeepSeek V4 路径接入 AIV
        `kv_exchange`，这是 NPU cache data-plane transfer，不是 NPU
        IPC。
5.  **RDMA / NIXL / Mooncake 的进展主要在 KV cache
    与存储数据面，不应计作权重 Sparse NCCL。**
    -   vLLM [#56192](https://github.com/vllm-project/vllm/pull/56192)
        合入 RDMA-capable OBJ
        tier；[#65943](https://github.com/vllm-project/vllm/pull/65943)
        复用 Mooncake PD TransferEngine。
    -   SGLang
        [#44615](https://github.com/sgl-project/sglang/pull/44615) 合入
        NIXL TP-heterogeneous KV
        packing；[#49075](https://github.com/sgl-project/sglang/pull/49075)
        仍为 Draft，重点是 Mooncake transfer drain 前的 buffer
        生命周期保护。

### 一句话结论

**vLLM 本期最直接的权重同步进展是 Sparse NCCL batching、patch validation
和 loader 正确性；SGLang 本期更突出 NPU IPC weight cache、HTTP payload
去重、ROCm custom collective 与 HiCache / NIXL
数据面。两边没有形成可直接横向比较的同硬件、同模型 Sparse NCCL
性能结果。**

------------------------------------------------------------------------

## 2. 术语边界：先区分传输机制与数据面优化

  ---------------------------------------------------------------------------
  分类                 本报告中的含义                   不应混为
  -------------------- -------------------------------- ---------------------
  CUDA IPC             同机不同进程通过 CUDA IPC handle NCCL collective、跨机
                       共享 GPU allocation / tensor     RDMA
                       storage                          

  HIP IPC              ROCm 进程间共享 HIP allocation / CUDA IPC、RCCL
                       handle                           collective

  NPU IPC              通过 `torch_npu` reducer / NPU   HCCL collective、AIV
                       IPC handle 共享 NPU tensor       cache copy
                       storage                          

  消息 / 序列化        HTTP、RPC 或进程间消息中传递 CPU GPU IPC 零拷贝
  payload              tensor /                         
                       metadata，减少重复序列化和转发   

  VMM + POSIX fd       CUDA / HIP VMM allocation 通过   传统 CUDA IPC handle
                       POSIX fd 导出、导入和映射        

  Dense NCCL           完整 tensor / bucket / weight 的 Sparse NCCL
                       NCCL broadcast 或 collective     

  Sparse NCCL          发送 checkpoint-coordinate       Sparse
                       sparse weight patches（如        MLA、HiSparse、稀疏
                       indices + values），接收端验证并 KV、磁盘缓存
                       apply                            

  NCCL M2N             trainer 与 inference 使用不同    普通 NCCL
                       GPU mesh / sharding layout 时的  broadcast、Sparse
                       NCCL reshard 传输                NCCL

  RCCL / AITER /       ROCm 上的 collective 实现或      NVIDIA NCCL
  CustomAllReduce      kernel 路径                      

  RDMA / NIXL /        跨设备或跨节点的直接内存 / KV    CUDA IPC、NCCL
  Mooncake / MoRIIO    数据传输及其控制面               M2N、Sparse NCCL

  AIV transfer         Ascend HiCache 使用 AIV          NPU IPC、HCCL weight
                       `kv_exchange` 在 device / host   broadcast
                       cache 间搬运                     

  HiSparse / sparse KV 稀疏 KV cache                    Sparse NCCL 权重更新
                       的驻留、备份、swap-in / swap-out 

  sparse_disk          磁盘 /                           Sparse NCCL wire
                       文件层的稀疏缓存或数据组织       format
  ---------------------------------------------------------------------------

**本期边界判断：**

-   vLLM 官方 weight-transfer backend 列表将
    `NCCL`、`IPC`、`sparse_nccl`、`sharded_rdt`、`nccl_m2n`
    分开列出；其中 `sharded_rdt` 是 NIXL / Ray Direct Transport pull
    path。参考：[vLLM Weight Transfer
    文档](https://docs.vllm.ai/en/latest/training/weight_transfer/)。
-   `#48241` 的 HTTP 请求承载的是普通 CPU tensor，没有 CUDA IPC
    handle；只有进程内 `Engine` 路径仍然需要按 rank 发送 IPC handles。
-   `#49001` 的 HIP FABRIC handle 用于 ROCm VMM allocation 共享与 custom
    collective，不应仅因名称中有 FABRIC 就归类为跨节点 RDMA。
-   `#48031` 是 AIV HiCache transfer；`#48835` 是 NPU EPLB 的 P2P weight
    migration；`#49264` 是 HCCL communication-domain
    生命周期管理。这三者都不是 NPU IPC weight-cache transport。
-   本期在两个主仓中未检索到明确以 **WeightBridge、Connex 或
    `sparse_disk`** 命名的新 PR / Issue / RFC / commit。相关
    RDMA、NIXL、Mooncake、HiSparse 和 disk-tier
    项目仍按各自真实机制分类，不用这些名称代称。

------------------------------------------------------------------------

## 3. vLLM 社区进展

### 3.1 Sparse NCCL：patch validation 与 bounded batching

#### [#66021 --- Validate sparse weight patches in one pass](https://github.com/vllm-project/vllm/pull/66021)

-   **作者：** `ShuoleiWang`

-   **创建：** 2026-11-06 09:09（UTC+8）

-   **合并：** 2026-11-07 16:41（UTC+8）

-   **状态：** Merged

-   **技术方案：** 将逐 patch 的 NaN、indices
    范围、排序和重复检查改成批量验证。旧实现每个 patch 会触发多个小型
    GPU 操作，模型 tensor 数量大时验证启动开销明显。

-   **作者测试：** Qwen3-30B-A3B、TP4、H20、18,867 个 tensor、32 个
    streaming clients。

      新比例 P   ayload Ma   in pause PR     pause 降 幅
      ---------- ----------- ------------- ---------- ----
      0.1% 0.    17 GiB      5.75 s          3.06 s 4 7%
      1% 1.      71 GiB      7.14 s          3.73 s 4 8%
      3% 5.      12 GiB      8.99 s          5.35 s 4 0%

作者还报告： - 单 tensor 平均验证耗时约从 125 µs 降至 5 µs； - 30
组交替对比全部更快； - `/weight_checker` 的 1,740 个权重摘要和 greedy
per-token logprobs 一致； - patch 校验单测 31 passed，reload 测试 1
passed，weight-transfer 测试 20 passed / 1 skipped； - 120,000 轮随机
CPU 校验和 21 种故障注入与旧实现判定一致。

**判断：** 这是 Sparse NCCL 的接收端 patch validation 优化，不是 NCCL
broadcast 算法、网络带宽或 wire format 的优化。pause 时间覆盖 pause →
send → resume 周期，不等于纯通信时间。

#### [#66375 --- Fix sparse NCCL update OOM and improve performance](https://github.com/vllm-project/vllm/pull/66375)

-   **作者：** `ShuoleiWang`
-   **创建：** 2026-11-08 16:07（UTC+8）
-   **本期截止状态：** Open
-   **后续状态：** 2026-11-14 20:46
    合并；属于截止后合并，不能回填为本期已合并。
-   **技术方案：**
    1.  trainer / worker 根据相同 metadata 将 patch 切为最大 512 MiB 的
        batches；
    2.  每批 indices 一次 broadcast，values 按 dtype 合并 broadcast；
    3.  worker 收到一批后立即 apply，不再把整轮 patch 全部留在 GPU；
    4.  trainer 先将 batch 内非连续 patch 拼成连续字节，修复非连续 patch
        直接广播错误字节的问题；
    5.  单批校验或加载失败时，worker 继续接收后续广播，最后再抛错，避免
        collective group 因某个 rank 提前退出而失步。

**作者公开结果：**

   更新比例     Payload   Main `send_weights`   Batched  变化
  ---------- ---------- --------------------- --------- ------
     0.1%      0.17 GiB                2.63 s    2.08 s  -21%
      1%       1.71 GiB                3.14 s    2.11 s  -33%
      3%       5.12 GiB                4.29 s    2.14 s  -50%

其他结果： - 广播次数：37,734 → 2--22； - Qwen3-30B-A3B TP4
默认配置下，Main 在 1.25% changed elements 时 OOM；PR 最高测试到 50%； -
3% 更新时额外 GPU memory：6.65 GiB → 2.02 GiB； - 32 streaming clients
的 HTTP under-load pause：7.24 s → 4.87 s（3% case）； - 全部 1,740 个
weight digests 和 greedy per-token logprobs 与 Main 匹配。

**独立验证：** 本报告未在 H20 或 Ascend
上复测。上述数据均为作者报告。另需注意，PR 最后一次 11 月 13 日 14:00
的分支同步晚于本期 12:00 截止，合并发生在 11 月 14 日。

**判断：** 这是本期最值得研究的 Sparse NCCL 传输数据面变化：bounded
memory、batch apply、减少 collective 次数和保持 collective 对齐。对 HCCL
Sparse 权重同步有参考价值，但 H20 结果不能直接外推到 A3/A5。

#### [#66944 --- Stop sparse weight patches from storing NaN markers](https://github.com/vllm-project/vllm/pull/66944)

-   **作者：** `saikrn112`
-   **创建：** 2026-11-11 11:47（UTC+8）
-   **本期截止状态：** Open
-   **本次复核状态：** Open；最后更新时间 2026-11-21
    04:17（UTC+8），该更新时间在本期之后。
-   **问题：** Qwen3.5 / Qwen3-Next GatedDeltaNet 与 Mamba2 sparse
    updates 可能返回成功，但权重错误。
    -   `mamba_v2_sharded_weight_loader` 使用 slice assignment，绕过
        sparse patch 对 `Tensor.copy_` 的 NaN-marker 过滤；
    -   `composed_weight_loader` 在 patch 后再次执行 transform，可能对
        unchanged elements 重复变换，例如 Mamba2 的
        `A = -exp(-exp(A_log))`。
-   **修复：** Mamba sharded loader 改为对目标 slice 使用
    `copy_`；composed loader 增加 `elementwise=True`
    opt-in，先提取变化元素再执行 transform；不支持该语义的 composed
    loaders 在 sparse update 时显式报错。
-   **测试：**
    -   `test_checkpoint_weight_patch.py`：Main 6 failed / 31 passed；PR
        37 passed；
    -   Qwen3.5-0.8B TP1/2 错误参数数：18 / 248 → 0；
    -   Mamba2-370m TP1/2：44--172 / 434 → 0；
    -   Qwen3.5-35B-A3B TP4/8：29 / 573 → 0。
-   **作者性能数据：** Dense load path 的 Qwen3.5-0.8B median 21.80 ms →
    22.06 ms；作者认为差异在 run-to-run noise 内。Mamba2 为 35.39 ms →
    37.50 ms，但 server runs 较少，结论不确定。
-   **判断：** 这是 Sparse NCCL 的模型 loader
    正确性问题，不是通信性能问题。对 GDN / Mamba 权重同步尤其值得跟踪。

### 3.2 CUDA / HIP IPC 与 IPC collective

#### [#67082 --- Avoid KFD queue eviction after the first IPC weight update](https://github.com/vllm-project/vllm/pull/67082)

-   **作者：** `jhinpan`
-   **创建：** 2026-11-12 03:07（UTC+8）
-   **状态：** Open
-   **机制：** ROCm HIP IPC 权重更新。worker 在 `Worker.init_device`
    中、分配设备内存前，从短生命周期 child process 导入一个 HIP IPC
    handle，避免首次 IPC import 触发 KFD queue eviction。
-   **作者测试：** 8×MI355X、gfx950、HIP 7.15、torch 2.13 ROCm 10。
    -   第一次 `send_weights`：13.7--23.7 s → 0.28--0.29 s；
    -   KFD `evicted_ms`：原来 10 万至 35 万量级，修复后为 0--116；
    -   更新后 generation：原来 1.4--18 s / batch，修复后 0.64--0.66 s；
    -   greedy outputs 更新前后相同；
    -   代价：符合该配置时每个 worker 启动增加约 1.9 s。
-   **测试边界：** 作者在 ROCm image 中挂载 PR
    文件进行复现；不是本报告独立运行的 upstream CI。
-   **判断：** 这是 HIP IPC handle import / allocator
    初始化顺序问题，不是 RCCL 或 Sparse NCCL。

#### [#65639 --- Add stream-ordered same-host CUDA IPC pulls](https://github.com/vllm-project/vllm/pull/65639)

-   **作者：** `zhangxin81`
-   **创建：** 2026-11-04 17:42（UTC+8）
-   **更新：** 2026-11-09 14:52（UTC+8）
-   **状态：** Draft
-   **机制：** NIXL connector 的同机 CUDA IPC pull。Decode 在 compute
    batch 中直接从 producer cache 通过 `swap_blocks_batch` 复制，并使用
    CUDA event 保证消费前 copy 完成；保留 NIXL handshake、descriptor
    mapping 和 completion notification。默认关闭，跨机仍走普通 NIXL
    pull。
-   **数据性质：** PR 作者提交并确认了一份 2026-11-04 的下游 H20 serving
    profile，不是 upstream branch 上的受控对照。
    -   每个 D rank copied bytes：176.099 MiB → 101.180 MiB；
    -   GPU P2P copy events：832 → 1 batch event；
    -   四个 D rank copy window：13.05 ms → 0.713 ms；
    -   P 最后 GPU work 到最早 D graph kernel：47.26 ms → 25.94 ms。
-   **重要限制：** baseline 与 IPC arm 使用不同节点；P 调度模式、采样
    seed 等不完全相同；draft acceptance 和 output equivalence
    尚未建立。不能把两条请求的 client latency 差值当作受控吞吐提升。
-   **判断：** 这是 KV cache 的 CUDA IPC pull，不是 trainer-to-rollout
    weight IPC。

#### [#64420 --- Close the early-PDL completion race on the trtllm IPC backend](https://github.com/vllm-project/vllm/pull/64420)

-   **创建：** 2026-10-29；**本期更新：** 2026-11-07 06:10（UTC+8）
-   **状态：** Open
-   **机制：** 修复 FlashInfer one-shot Lamport all-reduce 在单机
    `trtllm` IPC workspace 下提前发出 PDL completion 的竞态，避免后续
    kernel 读取尚未提交的 output buffer。
-   **测试 / 性能：** PR 主要给出 API safety
    条件和代码路径分析；未提供可量化性能收益。
-   **边界：** 仅影响单机 `trtllm` IPC workspace；MNNVL backend
    忽略该参数，不受此改动影响。

#### [#63890 --- Use AITER's all-gather for small TP all-gathers](https://github.com/vllm-project/vllm/pull/63890)

-   **创建：** 2026-10-27；**合并：** 2026-11-07 02:18（UTC+8）
-   **状态：** Merged
-   **机制：** ROCm gfx950 / gfx942，TP communicator 已使用 AITER custom
    all-reduce 时，小尺寸 TP all-gather 路由到 AITER IPC
    all-gather，替代 RCCL。
-   **payload 上限：** TP2 32 MiB、TP4 16 MiB、TP8 8 MiB。
-   **作者测试：** MI355X gfx950；768 个 sweep 点，覆盖不同 shape /
    dtype / eager / graph replay；AITER 与 RCCL 输出 bit-exact；GSM8K
    1319 题平均准确率均为 0.916。
-   **作者性能数据：** TP4 decode c1--c4 约快 1.2%--1.9%；TP8 decode
    c1--c4 约快 0.8%--1.2%；TP4 c8 约快 1.4%，但作者提示波动较大。
-   **判断：** 这是 ROCm IPC collective 优化，不是权重 Sparse NCCL。

#### [#67010 --- Route TP reduce-scatter through AITER's IPC kernel](https://github.com/vllm-project/vllm/pull/67010)

-   **创建：** 2026-11-11 17:18；**更新：** 17:46（UTC+8）
-   **状态：** Draft
-   **机制：** gfx942 / gfx950、TP2/4/8，在满足 AITER size /
    divisibility 条件时，将二维 BF16 / FP16、dim 0 的 TP reduce-scatter
    路由到 AITER copy-in IPC kernel；其余情况继续走 RCCL。
-   **作者 microbenchmark：** gfx950、BF16、hidden 8192、CUDA graph
    replay、100 次 median。
    -   TP2 32 KiB：6.2 µs vs RCCL 7.1 µs；
    -   TP2 1 MiB：15.2 µs vs 20.6 µs；
    -   TP2 64 MiB：614.6 µs vs 662.8 µs；
    -   TP2 在 13 个尺寸上 bitwise equal；TP4 在 4-ulp bound 内但不保证
        bitwise equal。
-   **作者 E2E：** Qwen3.8-Flash-Next-Quark-MXFP4 TP2，吞吐约提升
    2.0%--3.3%；W1 c64 约 +2.6%。每个 decode step 的 copy-in 增加约 395
    µs，AITER kernel 节省约 565 µs。
-   **限制：** gfx950、TP2/4 microbenchmark 与 TP2 E2E；gfx942 / TP8
    尚未验证。registered capture path 在 expandable segments
    下失败，因此采用 copy-in。

#### [#65628 --- Qwen4Exp AITER INT4 sequence-parallel collectives](https://github.com/vllm-project/vllm/pull/65628)

-   **创建：** 2026-11-04；**更新：** 2026-11-12 23:23（UTC+8）
-   **状态：** Draft
-   **机制：** Qwen4Exp `--enable-hc-sp` 下，根据 payload size 将部分
    sequence-parallel reduce-scatter / all-gather 路由到 AITER FlyDSL
    INT4 QuickReduce；不满足条件时走 exact collective。
-   **作者测试：** 8×MI355X、TP2。大 prefill step 的 GPU busy 从 236.2
    ms 降至 180.1 ms（-23.8%）；W1 吞吐在部分并发下提升约
    25%--30%，decode 小尺寸基本不变。
-   **精度：** GSM8K 5276 paired questions，strict accuracy 差值 -0.34
    percentage point，95% CI \[-0.75, +0.07\]；通过作者预先定义的 -1.0
    point 下界门槛。Needle test 40/40。
-   **限制：** INT4 是有损路径；测试只覆盖特定 TP2 / gfx950 / Qwen4Exp
    配置。INT4 路径只作用于一部分较大 step，不能概括为全 workload
    性能提升。
-   **判断：** 这是 ROCm sequence-parallel collective 数据面优化，不是
    Sparse NCCL 权重更新。

### 3.3 Dense NCCL / RCCL 与 M2N

#### [#66303 --- NCCL symmetric all-reduce uses the wrong group when TP and ETP differ](https://github.com/vllm-project/vllm/issues/66303)

-   **创建：** 2026-11-08 03:56（UTC+8）
-   **状态：** Open Issue
-   **复现：** 4×RTX 6000D、TP2 / DP2 /
    ETP4、`VLLM_USE_NCCL_SYMM_MEM=1`。Engram ETP all-reduce 返回 TP2
    subgroup 的结果，而不是四 rank ETP 结果。
-   **作者定位：** process-wide custom op 注册时闭包捕获第一个 PyNCCL
    communicator；后续调用未传递当前 group identity，可能复用错误
    communicator。
-   **风险：** 这是 correctness blocker，不是性能问题；需要检查 TP / ETP
    / DP group 交叉场景。

#### [#61109 --- Two-level all-reduce for TP=8 across two XGMI islands](https://github.com/vllm-project/vllm/pull/61109)

-   **创建：** 2026-10-12；**本期更新：** 2026-11-12 04:58（UTC+8）
-   **状态：** Open
-   **机制：** ROCm 8×MI210 两个 4-GPU XGMI islands，增加 ISLAND
    backend：小尺寸先 island 内 custom all-reduce，再跨岛 GPU
    pair；大尺寸先 island 内 RCCL reduce-scatter，再跨岛 shard
    all-reduce，最后 island 内 all-gather。
-   **作者测试：** Qwen2.5-72B、8×MI210、TP8。1k/1k 与 8k/1k workload 的
    TPOT p50 下降约 11%--26%，output throughput 提升约 13%--34%。
-   **限制：** 这是 ROCm RCCL + custom all-reduce 分层方案，不是 NVIDIA
    NCCL。作者没有运行 accuracy benchmark；跨 arm greedy outputs 在 16
    个 prompts 中有 10--12 个相同，不能据此声称完整输出等价。

#### NCCL M2N：本期没有新的已合并传输实现

-   [#65235 --- Load plan: `plan_loads` and
    \`TraceTensor](https://github.com/vllm-project/vllm/pull/65235) 已于
    11 月 6 日
    14:02（UTC+8）合并，属于上一期截止附近的基础架构，不计为本期新合并。
-   [#64374 --- Bounded, retryable NCCL weight
    updates](https://github.com/vllm-project/vllm/issues/64374) 是旧
    RFC，本期没有实质更新；当前页面更新时间为 11 月 19
    日，晚于本期截止。
-   [#66231 --- P/D disaggregation for continuous RL
    rollouts](https://github.com/vllm-project/vllm/issues/66231)
    是本期新 RFC，重点是 pause / lease / in-flight transfer
    control，不是新的 NCCL M2N wire protocol。H20 1P1D 作者测试报告 NIXL
    与 Mooncake 场景 8/8 通过，P pause 0.9--1.7 s、D pause 0.03 s、45 s
    pause 无 lease expiry。

### 3.4 RDMA / NIXL / Mooncake：主要是 KV data plane，不是 weight Sparse NCCL

#### [#56192 --- Add RDMA-capable OBJ tier](https://github.com/vllm-project/vllm/pull/56192)

-   **合并：** 2026-11-09 18:45（UTC+8）；状态 Merged。
-   **机制：** KV offload OBJ secondary tier 通过 NIXL `backend_params`
    传递 accelerated object-store options，并将 CPU DRAM pool
    分块注册，每块不超过 4 GiB - 64 KiB。
-   **测试：** `test_obj_tier.py` 50 passed；9 GiB CPU tier 注册成功。
-   **作者 NIC 证据：** standard S3 / accelerated without vendor `type`
    的 NIC `rx_write_requests` 为 0、netdev RX 1656 MiB；测试用 vendor
    S3-RDMA engine 时 `rx_write_requests` 增加 4400、netdev RX 4
    MiB。1.6 GiB KV 从 OBJ tier 提供。
-   **限制：** NIXL 1.5.2 只有 vendor engine（`type: dell`）；generic
    S3-RDMA engine 在 NIXL 1.6 / main，不属于该 PR 的正式验证。小模型
    OBJ reload TTFT 不优于 recompute，作者没有声称 TTFT 提升。
-   **分类：** RDMA KV offload，不是 trainer-to-rollout weight
    transfer。

#### [#65943 --- Reuse the Mooncake PD TransferEngine in the store connector](https://github.com/vllm-project/vllm/pull/65943)

-   **创建：** 2026-11-06；**合并：** 2026-11-11 09:24（UTC+8）
-   **状态：** Merged
-   **方案：** `MooncakeStoreConnector` 可选复用同一 worker 中
    `MooncakeConnector` 已创建的 TransferEngine，避免第二次 RDMA
    init、GPU KV registration、P2P handshake endpoint 和 CPU segment
    mount。
-   **测试：** 两个 unit suites 418 passed；2×H20 本地 E2E，reuse /
    separate 模式分别 5.60 / 5.59 req/s，TTFT 65.6 / 69.9 ms；sleep /
    wake 后输出一致。
-   **判断：** 是 Mooncake KV store / PD engine 资源复用，没有新的
    weight transfer wire protocol。

#### [#62832 --- Support sleep mode with MoRIIOConnector in WRITE mode over RDMA](https://github.com/vllm-project/vllm/pull/62832)

-   **创建：** 2026-10-21；**本期更新：** 2026-11-09 23:05（UTC+8）
-   **状态：** Draft
-   **方案：** MoRIIO WRITE + RDMA 在 sleep / wake 后使用 epoch 重新注册
    KV cache，并在下一次写入时重新 handshake；支持 P、D 或两边 sleep。
-   **测试：** MI355X + Pollara RDMA NIC；TP1 P + TP1 D；8/8 prompts
    输出与 awake baseline 一致，D 收到全部 25,862 prompt tokens；sleep
    释放 17.9--18.8 GiB。
-   **限制：** 依赖 #62857；READ mode 和 XGMI
    仍不支持。Draft，不能视为已合并能力。

#### NIXL KV transport correctness / layout

-   [#59746](https://github.com/vllm-project/vllm/pull/59746) 于 11 月 7
    日 03:16 合并：P/D block size 不同时，改为按 request last token 配对
    block，CPU 测试从 44 failed / 182 passed 变为 226 passed。
-   [#65922](https://github.com/vllm-project/vllm/pull/65922) 于 11 月 7
    日 03:32 更新：修复 NVFP4 KV 在 P16 → D64 等异构 block size
    场景的接收端 corruption；H100 serving / native-writer checks 2
    passed，CPU connector tests 155 passed；当前仍 Open。
-   [#64067](https://github.com/vllm-project/vllm/pull/64067) 于 11 月 7
    日 04:49 更新：为不同 block size 的 HMA attention group 注册 dense
    NIXL pages；三组相关 suites 446
    passed，另有一个环境依赖模型下载失败；当前仍 Open。
-   [#66837](https://github.com/vllm-project/vllm/pull/66837) 于 11 月
    10--11 日更新：Draft，尝试把 spec-driven TP slicing 用于 NIXL；PR
    测试计划仍未完成。
-   [#60523](https://github.com/vllm-project/vllm/issues/60523)
    MooncakeConnector RFC 于 11 月 10 日更新，讨论 push-only、重复
    metadata、bootstrap / GIL、RDMA write correctness debug 和 failure
    propagation；仍是 RFC，不是已合入实现。

------------------------------------------------------------------------

## 4. SGLang 社区进展

### 4.1 Weight sync：消息 payload、NPU IPC 与 VMM fd

#### [#48241 --- Let one serialized payload serve every TP rank](https://github.com/sgl-project/sglang/pull/48241)

-   **创建：** 2026-11-04；**合并：** 2026-11-07 08:49（UTC+8）
-   **状态：** Merged
-   **机制：** HTTP `update_weights_from_tensor` 原来按 TP rank 索引
    `serialized_named_tensors`，trainer 必须给每个 rank 发送一份相同
    payload；修改后 HTTP caller 只需发送一份普通 CPU tensor
    payload，所有 TP ranks 读取 entry 0。
-   **明确边界：** 只有 HTTP caller 使用单 payload；in-process `Engine`
    / CUDA IPC 路径仍保留 per-rank list，因为每个 rank 不能打开其它 rank
    的 CUDA IPC handles。
-   **作者性能数据：** GLM-5.3 744B full MoE LoRA，4.1 GB
    adapter，rollout MI355X、trainer B300：
    -   2 engines × TP4，4 copies：409 s；
    -   1 engine × TP8，8 copies：超过 600 s，触发 timeout；
    -   TP8 + one copy：127 s（其中 POST 103 s）；
    -   另一组 TP8 one-copy：84 s；连续三次 84.2 / 84.1 / 84.9 s。
-   **测试：** 新增 2 个 unit cases；logprob diff 0.036 → 0.033，KL
    5.4e-3 → 4.6e-3。
-   **判断：** 这是消息 / HTTP serialization payload 去重，不是 IPC
    零拷贝，也没有更换底层传输协议。

#### [#41495 --- Add NPU weight cache support](https://github.com/sgl-project/sglang/pull/41495)

-   **创建：** 2026-09-28；**合并：** 2026-11-11 15:05（UTC+8）
-   **状态：** Merged
-   **机制：** Ascend NPU 使用 `torch_npu` 原生 multiprocessing
    reducer（`reduce_tensor` / `rebuild_npu_tensor`），复用现有
    `TorchIpcTransportBackend`，支持 weight-cache daemon 与 engine 间
    NPU tensor IPC。
-   **测试：** Qwen3-32B TP2 startup test 比较 cold disk load、daemon
    disk load、warm IPC import；作者称 zero-copy path 不引入额外 data
    copy，warm IPC import 快于 cold load。
-   **性能边界：** 作者没有提供 per-token throughput 提升数据；该能力是
    weight-cache import，不等同于分布式 trainer-to-rollout HCCL
    broadcast。
-   **平台限制：** 零拷贝需要 HDK ≥ 25.3.RC1、CANN ≥ 8.3.RC1；950DT
    不支持该 zero-copy path，需要 CPU-copy fallback。

#### [#46953 --- Add the vmm_fd transport](https://github.com/sgl-project/sglang/pull/46953)

-   **创建：** 2026-10-28；本期无实质更新。
-   **本期截止状态：** Open；**后续状态：** 2026-11-19 09:00 合并。
-   **机制：** CUDA VMM allocation 通过 Unix socket `SCM_RIGHTS` 发送
    POSIX fd；engine 预留 VA 并导入映射。daemon 被 SIGKILL 后，已 attach
    的 engine 可继续使用映射中的权重。
-   **分类：** VMM + POSIX fd，不是传统 CUDA IPC handle。当前 PR
    只做一次 attach，尚未实现释放后同地址 re-attach，也未支持 cache
    active 时 weight update。
-   **本期处理：** 作为旧 PR 无实质更新列出；11 月 19
    日的合并是截止后变化。

#### 旧 PR：NPU multimodal IPC 与 PP CUDA IPC

-   [#46208](https://github.com/sgl-project/sglang/pull/46208)
    `mm_feature_transport=npu_ipc`：本期无实质更新；当前 Open，11 月 19
    日更新。它是 NPU multimodal feature transport，不是 weight-cache
    IPC。
-   [#48657](https://github.com/sgl-project/sglang/pull/48657) PP 下每
    engine rank 一个 CUDA IPC weight payload：本期无实质更新；当前
    Open，11 月 21 日更新。修复 PP\>1 时 stage 1 rank 错误复用 stage 0
    IPC handle 和 producer refcount 的问题；4×H100 PP2×TP2
    的作者测试中，修复后 stage 1 `down_proj` 权重正确更新。

### 4.2 ROCm collective：CustomAllReduceV2、FABRIC handle 与 NVLink multicast

#### [#47964 --- Run the JIT custom all-reduce v2 on ROCm](https://github.com/sgl-project/sglang/pull/47964)

-   **创建：** 2026-11-03；**合并：** 2026-11-07 07:39（UTC+8）
-   **状态：** Merged
-   **方案：** ROCm gfx950 上使用 HIP VMM + POSIX fd 建立 symmetric
    workspace，移植 JIT Lamport push / pull custom all-reduce v2。
-   **限制：** 合入后 ROCm serving 默认仍走 AITER；该 PR 的 benchmark
    不能当作默认 serving 已获得的性能提升。

#### [#49001 --- Run carv2 on MI450 with FABRIC handles and a gfx1250 table](https://github.com/sgl-project/sglang/pull/49001)

-   **创建：** 2026-11-07 09:56；**合并：** 2026-11-08 05:27（UTC+8）
-   **状态：** Merged
-   **方案：** MI450 / gfx1250 支持 HIP `hipMemFabricHandle_t`，并增加
    gfx1250 专属 `1shot_push` / `2shot_push` threshold 和 block
    table。通过 `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1` 显式
    opt-in；默认仍为 AITER。
-   **作者测试：** 8×MI450、TP8、BF16、CUDA graph replay。
    -   4 KB：carv2 5.8 µs，RCCL 9.9 µs（-41%）；
    -   512 KB：13.8 µs vs 28.7 µs（-52%）；
    -   8 MB：62.0 µs vs 89.3 µs（-31%）；
    -   16 MB 回退 RCCL。
-   **Serving：** DeepSeek-V4-Flash TP8，C64 output tok/s +11.1%，C256
    +5.5%。
-   **正确性：** 每个 world size 310 passed；TP8 stress eager / graph
    replay 无 mismatch；与 RCCL bit-exact。
-   **重要限制：** serving 数字使用本地修复了一个与该 PR 无关的 HIP
    compile issue 的分支；必须与默认 AITER serving 区分。FABRIC handle
    是 ROCm VMM allocation 共享方式，不等同于跨节点 RDMA。

#### [#48563 --- Fuse Gemma4 TP all-reduce with layer epilogue and FP8 quant](https://github.com/sgl-project/sglang/pull/48563)

-   **创建：** 2026-11-05；**合并：** 2026-11-07 16:00（UTC+8）
-   **状态：** Merged
-   **方案：** Gemma-4 extend path 用 NVLink SHARP /
    `multimem.ld_reduce` 与 `multimem.st` 将 TP
    all-reduce、RMSNorm、residual、FP8 quant 融合；不是 NCCL transport
    backend 的替换。
-   **作者数据：** H200 kernel benchmark 中 TP4 2,048 tokens 从 156.9 µs
    降至 95.4 µs，TP8 从 189.0 µs 降至 91.1 µs；TTFT 各 cell 下降
    13%--22%，prefill-heavy throughput 提升约 9%--13%。
-   **精度：** GSM8K paired 0.9704 → 0.9712，差异置信区间跨 0；作者报告
    greedy KL drift 较小，但比 unfused noise floor 高。
-   **判断：** 这是 collective + epilogue kernel
    fusion，属于数据面融合，不应记作 NCCL 算法或传输机制更新。

#### [#45947 --- Add gated allgather in KV cache sharding](https://github.com/sgl-project/sglang/pull/45947)

-   **创建：** 2026-10-23；**合并：** 2026-11-12 15:03（UTC+8）
-   **状态：** Merged
-   **方案：** 将 background KV allgather 拆成 8 MiB multicast
    chunks；`TransferChain(Link.NVLINK)` gate 在 foreground collective
    到来时暂停后续 chunks，已发出的 chunk 继续完成，foreground
    完成后再恢复。
-   **对照：** 原生 NCCL 使用一个 `ncclAllGather` +
    `NCCL_CTA_POLICY_ZERO`；custom 路径不是 NCCL collective。
-   **作者数据：** 256 MiB/rank background gather + 32 MiB foreground
    all-reduce：
    -   foreground latency：Native NCCL 1,272 µs；custom chunked +
        paused 177 µs；
    -   total completion：Native 1.501 ms；custom chunked + paused 1.617
        ms。
-   **判断：** 显著降低前台延迟，但 total completion 变长；是通信调度 /
    latency priority tradeoff，不是总吞吐提升。

### 4.3 Ascend NPU：AIV HiCache、HCCL lifecycle 与 EPLB weight migration

#### [#48031 --- Support AIV transfers for MLA and DeepSeek V4](https://github.com/sgl-project/sglang/pull/48031)

-   **创建：** 2026-11-03；**合并：** 2026-11-12 19:50（UTC+8）
-   **状态：** Merged
-   **方案：** Ascend HiCache MLA / DeepSeek V4 的 L2 host transfer
    路由到 in-tree AIV `kv_exchange` operator；host memory 可用 ACL
    registration，也可选 MemFabric。
-   **分类：** NPU cache transfer，不是 NPU IPC，也不是 HCCL weight
    broadcast。
-   **作者 TTFT 数据：** 32k input / 1 output、100% cache、8 NPU。
    -   A3 GLM-5.2：baseline 2097.56 ms；AIV page-first 2049.37
        ms；layer-first 2073.46 ms。
    -   A5 GLM-5.2：baseline 1834.34 ms；AIV page-first 1847.80
        ms；layer-first 1854.45 ms。
    -   A3 DeepSeek V4：baseline 724.94 ms；AIV page-first 429.32
        ms；layer-first 409.74 ms。
    -   A5 DeepSeek V4：baseline 1912.46 ms；AIV page-first 232.98
        ms；layer-first 231.05 ms。
-   **测试：** 新增 layout dispatch、layer / Indexer selection、host
    allocation accounting tests；GSM8K 128 samples，0.984375。
-   **注意：** 数据是作者给出的特定模型 / cache 配置结果，A3/A5、模型和
    baseline 路径不同，不应横向直接归因于 AIV 单一因素。

#### [#49264 --- Release device communication domains on NPU sleep mode](https://github.com/sgl-project/sglang/pull/49264)

-   **创建：** 2026-11-09 15:51；**更新：** 2026-11-11 06:19（UTC+8）
-   **状态：** Open
-   **方案：** Ascend 每个 parallel group 的 HCCL communication domain
    会持有大量 device buffers；新增 `comm` memory-saver tag，通过
    `torch_npu.distributed.reinit_process_group(rebuild_link=True)` 在
    sleep 时释放通信域、resume 时重建，同时保留 `ProcessGroup`
    对象，使已有 device-group 引用继续有效。
-   **生命周期：** 先释放 CUDA graphs，再销毁 / 重建实际存在的 HCCL
    comm；按 group/world 做一致性投票；resume 要求 weights / KV cache
    先恢复。
-   **正确性：** 作者声明 wake 前后 greedy outputs 一致。
-   **性能：** PR 明确说明无性能 benchmark；当前 CI 状态不完整 /
    有失败项。
-   **判断：** 这是 HCCL resource lifecycle，不是新的 HCCL 数据传输
    backend。

#### [#48835 --- Preserve expert storage and reuse TP communication for NPU EPLB](https://github.com/sgl-project/sglang/pull/48835)

-   **创建：** 2026-11-06；**更新：** 2026-11-10 20:56（UTC+8）
-   **状态：** Open
-   **方案：** Ascend A3 / A5 统一使用 storage-preserving expert
    migration；直接按物理 storage bytes 访问 ND / batched NZ，保留 FP4 /
    FP8 bit pattern、NZ padding、原 tensor 地址和 captured graph
    pointer；避免 NZ→ND→NZ 转换及 full-weight snapshots；复用 TP
    communication。
-   **作者 device test：**
    -   A3 单层 local update wall time：6.036 ms → 4.114 ms（-31.85%）；
    -   peak extra allocated memory：1272.270 MiB → 840.269 MiB（减少
        432 MiB）；
    -   writeback coalescing 后，16/17 expert slots migration 的三层
        wall time：10.411 ms → 8.934 ms（-14.2%）；
    -   两 rank mixed local/remote migration：19.142 ms → 16.519
        ms（-13.7%）。
-   **HCCL 测试限制：** 48-case real-NPU matrix 和两 rank HCCL
    migration/restore checks 通过，但默认 task queue 环境下 coalesced
    HCCL P2P 会报 `HcclGroupEnd: not in a group call`；作者使用
    `TASK_QUEUE_ENABLE=0` 使 focused HCCL correctness tests 通过。local
    timing 没有执行 P2P。
-   **E2E EPLB：** 作者另报告 DeepSeek-V4-Flash W8A8 1P1D、8
    cards，9K/1K case throughput +26.94%；这是 dynamic EPLB OFF/ON
    的端到端差异，不是单独的 HCCL transport speedup。
-   **判断：** 这是 NPU expert weight migration / P2P 数据面优化，不是
    trainer-to-rollout IPC 或 Sparse NCCL。

#### [#49277 --- Optimize Ascend MoE AllGather](https://github.com/sgl-project/sglang/pull/49277)

-   **创建：** 2026-11-09 17:27；**合并：** 2026-11-11 20:12（UTC+8）
-   **状态：** Merged
-   **方案：** 复用 DP/CP token metadata 避免 row-count AllGather；遵循
    low-latency MXFP8 communication dtype；按 local expert range
    路由，避免完整 FP32 buffer unpermute。
-   **测试：** PR 给出 A3 / DeepSeek-V4-Flash、TP8、DP8、200
    prompts、concurrency 64 的 benchmark 命令与 accuracy / speed
    图片；文本没有给出可复核的数值表。
-   **判断：** NPU MoE collective 优化；PR 描述使用 `ascend_ops` / CP
    transport，不应未经代码核验就直接标为"新增 HCCL backend"。

### 4.4 UMBP / NIXL / Mooncake KV data plane

#### [#48764 --- Register UMBP GPU buffers with device metadata](https://github.com/sgl-project/sglang/pull/48764)

-   **创建：** 2026-11-06 09:11；**更新：** 2026-11-11 04:51（UTC+8）
-   **状态：** Open
-   **问题：** UMBP direct linker 注册 GPU KV allocation 时只传 pointer
    / size，MoRI embedded client 默认按 CPU memory 处理，导致选择 CPU
    memcpy 而不是 GPU transfer engine。
-   **方案：** 传入 `MemoryLocationType.GPU` 和 local device index；保留
    allocation-base dedup；拒绝 host buffer；传播 registration failure。
-   **作者历史性能数据：** MI355X output tok/s 31.65 / 31.76 → 183.09 /
    183.74；MI455X 155.95 / 152.84 → 162.00 / 161.59。
-   **验证限制：** 作者说明 MI355X 结果包含其它集成变化；完整 upstream
    CI 在 rebase 后仍待完成。该 PR 的 code commits 主要在 11 月 6
    日，本期更新主要是验证 / PR 状态。

#### [#44615 --- NIXL KV packing for TP-heterogeneous transfer](https://github.com/sgl-project/sglang/pull/44615)

-   **创建：** 2026-10-16；**合并：** 2026-11-11 06:25（UTC+8）
-   **状态：** Merged
-   **方案：** P attention TP \< D attention TP 时，先把每个 decode rank
    的 KV head slices gather 到连续 GPU buffer，再用 page-sized NIXL
    descriptors 发送，减少 per-token small descriptors。
-   **作者数据：** TEP4 prefill / TEP8 decode，Muse Spark Open
    NVFP4；concurrency 128 output throughput +46%、TTFT p50 13.63 s →
    1.97 s；concurrency 64 throughput +9%。concurrency 2--16
    时吞吐基本不变，TTFT p50 约增加 90 ms。
-   **分类：** NIXL KV transport packing，不是 weight transfer。

#### [#49075 --- Retain Mooncake buffers until transfer drain](https://github.com/sgl-project/sglang/pull/49075)

-   **创建：** 2026-11-07 20:47；**更新：** 20:51（UTC+8）
-   **状态：** Draft
-   **问题：** Mooncake PD cancellation 可能在 native writes 尚未 drain
    时释放源 KV locks、metadata 或 device destination slots。
-   **方案：** 保留 source ownership 到 queued chunks / worker
    bookkeeping drain；device destination 必须收到 generation-bound
    drain ACK；未知完成状态 quarantine，不能仅按 elapsed time 释放。
-   **测试：** CPU-only 6 个 native suites 共 94 tests + 54 subtests
    passed；真实 4-rank Gloo roster mismatch test 通过。
-   **限制：** CUDA / DMA fault injection、serving cancellation under
    load、asymmetric TP serving 和性能仍未验证。当前是 Draft。

#### [#49317 --- Remove synchronous host-backup stall in HiSparse spec decode](https://github.com/sgl-project/sglang/pull/49317)

-   **创建：** 2026-11-09；**关闭：** 2026-11-10；**状态：**
    Closed，未合并。
-   **方案：** 删除 speculative decode 每次 commit 后的同步
    `wait_for_pending_backup()`，让 D2H KV backup 在独立 stream
    上继续；swap-in 时再等待 event。
-   **测试：** CPU regression 5 tests、coordinator 10 tests；H100
    数据完整性 / event ordering；4×H100 PD-disaggregated E2E 输出一致。
-   **分类：** HiSparse KV host-backup overlap，不是 Sparse NCCL，也不是
    sparse_disk weight transfer。

------------------------------------------------------------------------

## 5. 本期无实质更新的旧 PR / RFC

以下项目与主题相关，但在 11 月 7 日 00:00 至 11 月 13 日 12:00
之间没有可确认的实质代码 / 设计更新。表中"当前状态"按 11 月 21
日复核时页面记录。

  -------------------------------------------------------------------------------------------------------------------
  仓库       项目                                                           本期情况           当前状态 / 后续变化
  ---------- -------------------------------------------------------------- ------------------ ----------------------
  vLLM       [#57854](https://github.com/vllm-project/vllm/pull/57854)      本期无实质更新     Open；11/20
             weight update rollback                                                            更新，超出本期

  vLLM       [#64374](https://github.com/vllm-project/vllm/issues/64374)    本期无实质更新     Open；11/19
             bounded / retryable NCCL update RFC                                               更新，超出本期

  vLLM       [#65192](https://github.com/vllm-project/vllm/pull/65192) PCIe 本期无实质更新     Open；11/19
             IPC all-reduce P2P fallback                                                       更新，超出本期

  SGLang     [#46953](https://github.com/sgl-project/sglang/pull/46953)     本期无实质更新     11/19
             CUDA VMM fd weight cache                                                          合并，属于截止后变化

  SGLang     [#46208](https://github.com/sgl-project/sglang/pull/46208) NPU 本期无实质更新     Open；11/19
             multimodal `npu_ipc`                                                              更新，超出本期

  SGLang     [#48657](https://github.com/sgl-project/sglang/pull/48657) PP  本期无实质更新     Open；11/21
             CUDA IPC payload rank mapping                                                     更新，超出本期

  SGLang     [#47600](https://github.com/sgl-project/sglang/pull/47600)     最近更新在         Open
             NCCL EP backend                                                11/6，本期无更新   

  SGLang     [#42210](https://github.com/sgl-project/sglang/pull/42210) GPU 本期无实质更新     Open；11/18
             delta / Bitcomp                                                                   更新，超出本期

  SGLang     [#37372](https://github.com/sgl-project/sglang/issues/37372)   本期无实质更新     Open；最近更新在 11/1
             HiCache out-of-process device IPC RFC                                             
  -------------------------------------------------------------------------------------------------------------------

补充： - vLLM [#65507](https://github.com/vllm-project/vllm/pull/65507)
LMCache MP + extensible KV cache 于 11 月 9
日关闭、未合并；不要把它记作已落地的 VMM fd backend。 - vLLM
[#65956](https://github.com/vllm-project/vllm/pull/65956) "S3 over RDMA
accelerated option" 于 11 月 9 日关闭、未合并；已合入的 RDMA OBJ
能力应以 [#56192](https://github.com/vllm-project/vllm/pull/56192)
的真实实现和测试为准。 - vLLM
[#64393](https://github.com/vllm-project/vllm/pull/64393) 是
SimpleCPUOffload disk tier 文件清理 Draft，不是 `sparse_disk`
权重传输。 - vLLM commit
[#65607](https://github.com/vllm-project/vllm/commit/508e5784b18a0addf1d607bb7eeaf6792f24c142)
是 HiSparse coordinator per-page residency bookkeeping；SGLang
[#49317](https://github.com/sgl-project/sglang/pull/49317) 是 HiSparse
host backup async。它们属于 sparse KV cache 管理，不属于 Sparse NCCL。

------------------------------------------------------------------------

## 6. 两项目横向比较

  -----------------------------------------------------------------------
  维度                    vLLM                    SGLang
  ----------------------- ----------------------- -----------------------
  Sparse NCCL 权重更新    本期有 #66021 批量      本期没有发现直接 Sparse
                          validation、#66375      NCCL 权重传输实现
                          bounded                 
                          batching、#66944 loader 
                          correctness             

  Dense NCCL / collective #66303 NCCL symmetric   #47600 NCCL EP
  correctness             all-reduce group        本期无更新；#45947 是
                          mismatch；#64420 IPC    NVLink multicast gated
                          PDL completion race     allgather，不是 NCCL
                                                  实现

  CUDA / HIP IPC          #67082 HIP IPC weight   #48657 PP CUDA IPC 旧
                          sync；#65639 CUDA IPC   PR；#46953 VMM fd 旧
                          KV pull；#64420 trtllm  PR（截止后合并）
                          IPC race                

  NPU IPC                 本期未发现 vLLM 主仓新  #41495 NPU weight-cache
                          NPU IPC weight-transfer IPC 已合并；#46208 NPU
                          实现                    multimodal IPC
                                                  本期无更新

  NPU HCCL / AIV          本期没有直接 HCCL       #49264 HCCL domain
                          Sparse weight-sync 合并 sleep lifecycle；#49277
                                                  NPU MoE
                                                  AllGather；#48031 AIV
                                                  HiCache

  NCCL M2N                本期无新的 M2N 传输     本期未发现对应 NCCL M2N
                          PR；#64374 RFC          权重传输
                          本期无更新              

  RDMA / NIXL / Mooncake  #56192 RDMA OBJ、#65943 #44615 NIXL KV
                          Mooncake TransferEngine packing、#49075
                          reuse、#62832 MoRIIO    Mooncake buffer
                          RDMA sleep draft        lifetime、#48764 UMBP
                                                  GPU registration

  非 NCCL collective      ROCm AITER IPC / RCCL   ROCm carv2 /
                          两级 all-reduce         FABRIC、NVLink
                                                  multicast fusion、gated
                                                  allgather

  本期最值得借鉴的方向    Sparse patch            NPU native IPC、HCCL
                          batching、collective    lifecycle、NZ/ND
                          failure                 storage-preserving
                          alignment、GDN/Mamba    migration
                          loader semantics        
  -----------------------------------------------------------------------

### 对昇腾 A3 / A5 的可借鉴点

1.  **Sparse HCCL batching：** 可研究 #66375 的 bounded batch +
    apply-after-receive 思路；关键不是简单切 chunk，而是 trainer /
    worker 两侧 batch metadata 一致、失败 rank 仍完成后续
    collective、buffer 复用前有 event / stream 生命周期保护。
2.  **Sparse patch correctness：** #66944 说明 NaN marker 机制依赖所有
    loader 都遵守 `copy_` 语义。需要重点审计 GDN / Mamba、slice
    assignment、composed transform、NZ / ND view 等路径。
3.  **NPU IPC 与 HCCL 的边界：** #41495 是同机 weight-cache
    IPC；不能替代跨节点 HCCL。#49264 是 HCCL comm domain
    生命周期；#48031 是 AIV HiCache；三者不是同一个 backend。
4.  **NZ / ND storage preservation：** #48835 的 raw physical
    bytes、descriptor
    borrowing、`record_stream`、scratch-before-writeback 对 A5 MXFP4 /
    MXFP8 NZ layout 有参考价值；但作者报告的 HCCL P2P 测试依赖
    `TASK_QUEUE_ENABLE=0`，必须进一步验证真实任务队列配置。
5.  **RDMA / Mooncake：** #56192、#62832、#65943 的价值主要在 KV offload
    / P-D cache transfer 和 lifecycle，不要把其 RDMA 带宽或成功案例记成
    weight Sparse NCCL 结果。

------------------------------------------------------------------------

## 7. 风险与待验证问题

1.  **Sparse NCCL 正确性：** #66375 解决了 batching / OOM /
    non-contiguous patch 问题，但 #66944 表明 loader
    层仍可能产生静默错误；必须以最终权重 digest 和模型输出校验为准。
2.  **HCCL collective 对齐：** batch apply 失败时继续接收后续 broadcast
    的设计值得借鉴；但 HCCL group、event、stream 和 task queue 行为需在
    A3 / A5 真实验证。
3.  **IPC handle 生命周期：** HIP / CUDA / NPU IPC 都有
    allocator、refcount、process lifetime 和 stream completion 约束；PP
    rank、sleep / wake、daemon exit、跨 rank handle ownership
    是高风险点。
4.  **有损 collective：** #65628 INT4 QuickReduce 有精度门槛，不能仅看
    throughput；需要覆盖真实模型、真实长上下文和 RL rollout accuracy。
5.  **RDMA 版本依赖：** #56192 的通用 S3-RDMA engine 并未在 NIXL 1.5.2
    中完整落地；vendor plugin 的 NIC counter 结果不能泛化到所有 NIXL
    版本。
6.  **性能对比不等价：** H20、H100、MI355X、MI450、A3、A5
    的拓扑、并发、模型、量化、cache
    命中率不同；本报告只整理作者数据，不做跨硬件排名。
7.  **CI 与作者本地测试：** PR body 中的作者 benchmark 不等于 upstream
    CI 全绿；例如 #49264 CI 不完整，#48764 upstream CI pending，#49075
    尚未做 GPU fault injection。

------------------------------------------------------------------------

## 8. 下周跟踪项

### P0：权重同步正确性与 Sparse NCCL

-   [#66375](https://github.com/vllm-project/vllm/pull/66375)：已在
    11/14 合并；跟踪 merge 后回归、更多模型 / TP / EP 配置，以及 A3/A5
    HCCL 可移植性。
-   [#66944](https://github.com/vllm-project/vllm/pull/66944)：GDN /
    Mamba2 sparse patch loader correctness；重点确认 composed loader
    对不支持 elementwise transform 的显式拒绝策略。
-   [#67082](https://github.com/vllm-project/vllm/pull/67082)：ROCm HIP
    IPC KFD queue eviction 修复是否合并，以及不同 visible-device mask /
    sleep mode 的覆盖。
-   [#66303](https://github.com/vllm-project/vllm/issues/66303)：NCCL
    symmetric all-reduce 的 TP / ETP group identity 修复。

### P1：IPC / AITER / NPU 通信

-   [#65639](https://github.com/vllm-project/vllm/pull/65639)：CUDA IPC
    same-host NIXL pull 的 upstream GPU test、output equivalence、draft
    acceptance。
-   [#67010](https://github.com/vllm-project/vllm/pull/67010)：AITER IPC
    reduce-scatter 的 gfx942 / TP4 / TP8 和 graph capture 验证。
-   [#65628](https://github.com/vllm-project/vllm/pull/65628)：INT4
    QuickReduce 的精度 / workload exposure 和 AITER router release。
-   [#46208](https://github.com/sgl-project/sglang/pull/46208)：NPU
    multimodal IPC 是否进入可合并状态。
-   [#48657](https://github.com/sgl-project/sglang/pull/48657)：PP × TP
    CUDA IPC payload rank ownership 和 producer refcount。
-   [#49264](https://github.com/sgl-project/sglang/pull/49264)：HCCL
    comm domain sleep / wake 的完整 NPU CI 与失败恢复。

### P2：RDMA / NIXL / Mooncake

-   [#62832](https://github.com/vllm-project/vllm/pull/62832)：MoRIIO
    RDMA sleep / wake 是否解除 Draft，以及 READ / XGMI 支持计划。
-   [#64374](https://github.com/vllm-project/vllm/issues/64374)：NCCL /
    M2N bounded retry RFC 的 timeout、abort、rollback 和 communicator
    rebuild 语义。
-   [#49075](https://github.com/sgl-project/sglang/pull/49075)：Mooncake
    transfer drain 的 CUDA / DMA fault injection 和 serving cancellation
    under load。
-   [#48764](https://github.com/sgl-project/sglang/pull/48764)：UMBP GPU
    registration 的完整 upstream CI。
-   [#48835](https://github.com/sgl-project/sglang/pull/48835)：A3/A5
    EPLB storage-preserving path 在默认 task queue 下的 HCCL P2P
    correctness。
-   [#46953](https://github.com/sgl-project/sglang/pull/46953)：VMM fd
    已在 11/19 合并，跟踪 release / re-attach 和 weight update
    后续能力。

------------------------------------------------------------------------

## 9. 本期重点 PR / Issue / Commit 索引

### vLLM

  ---------------------------------------------------------------------------------------------------
  编号            主题                   链接
  --------------- ---------------------- ------------------------------------------------------------
  #66021          Sparse patch 批量      [PR](https://github.com/vllm-project/vllm/pull/66021)
                  validation，已合并     

  #66375          Sparse NCCL bounded    [PR](https://github.com/vllm-project/vllm/pull/66375)
                  batching，截止后合并   

  #66944          GDN / Mamba2 sparse    [PR](https://github.com/vllm-project/vllm/pull/66944)
                  patch correctness      

  #67082          ROCm HIP IPC KFD queue [PR](https://github.com/vllm-project/vllm/pull/67082)
                  eviction               

  #63890          AITER IPC TP           [PR](https://github.com/vllm-project/vllm/pull/63890)
                  all-gather，已合并     

  #67010          AITER IPC              [PR](https://github.com/vllm-project/vllm/pull/67010)
                  reduce-scatter         

  #65628          Qwen4Exp AITER INT4 SP [PR](https://github.com/vllm-project/vllm/pull/65628)
                  collectives            

  #65639          Same-host CUDA IPC     [PR](https://github.com/vllm-project/vllm/pull/65639)
                  NIXL KV pull           

  #66303          NCCL symmetric         [Issue](https://github.com/vllm-project/vllm/issues/66303)
                  all-reduce group       
                  mismatch               

  #61109          ROCm two-level         [PR](https://github.com/vllm-project/vllm/pull/61109)
                  all-reduce             

  #66231          Continuous RL P/D      [Issue](https://github.com/vllm-project/vllm/issues/66231)
                  pause RFC              

  #64374          Bounded / retryable    [Issue](https://github.com/vllm-project/vllm/issues/64374)
                  NCCL update RFC        

  #56192          RDMA-capable OBJ tier  [PR](https://github.com/vllm-project/vllm/pull/56192)

  #65943          Mooncake PD            [PR](https://github.com/vllm-project/vllm/pull/65943)
                  TransferEngine reuse   

  #62832          MoRIIO RDMA sleep mode [PR](https://github.com/vllm-project/vllm/pull/62832)
  ---------------------------------------------------------------------------------------------------

### SGLang

  ------------------------------------------------------------------------------------------------
  编号            主题                    链接
  --------------- ----------------------- --------------------------------------------------------
  #48241          HTTP weight-sync one    [PR](https://github.com/sgl-project/sglang/pull/48241)
                  serialized payload per  
                  TP group                

  #41495          Ascend NPU weight-cache [PR](https://github.com/sgl-project/sglang/pull/41495)
                  IPC，已合并             

  #46953          CUDA VMM fd             [PR](https://github.com/sgl-project/sglang/pull/46953)
                  weight-cache            
                  transport，截止后合并   

  #47964          ROCm CustomAllReduceV2  [PR](https://github.com/sgl-project/sglang/pull/47964)

  #49001          MI450 FABRIC handles /  [PR](https://github.com/sgl-project/sglang/pull/49001)
                  gfx1250 carv2           

  #48563          Gemma4 NVLink multicast [PR](https://github.com/sgl-project/sglang/pull/48563)
                  all-reduce fusion       

  #45947          Gated multicast         [PR](https://github.com/sgl-project/sglang/pull/45947)
                  allgather               

  #48031          Ascend AIV HiCache      [PR](https://github.com/sgl-project/sglang/pull/48031)
                  transfer                

  #49264          NPU HCCL                [PR](https://github.com/sgl-project/sglang/pull/49264)
                  communication-domain    
                  sleep lifecycle         

  #48835          NPU EPLB                [PR](https://github.com/sgl-project/sglang/pull/48835)
                  storage-preserving      
                  migration               

  #49277          Ascend MoE AllGather    [PR](https://github.com/sgl-project/sglang/pull/49277)
                  optimization            

  #48764          UMBP GPU buffer         [PR](https://github.com/sgl-project/sglang/pull/48764)
                  metadata                

  #44615          NIXL KV packing for     [PR](https://github.com/sgl-project/sglang/pull/44615)
                  TP-heterogeneous        
                  transfer                

  #49075          Mooncake transfer drain [PR](https://github.com/sgl-project/sglang/pull/49075)
                  buffer lifetime         
  ------------------------------------------------------------------------------------------------

### 本期可定位的代表性 commits

  -------------------------------------------------------------------------------------------------------------------------------------
  仓库        Commit                                                                                              说明
  ----------- --------------------------------------------------------------------------------------------------- ---------------------
  vLLM        [88bdd833](https://github.com/vllm-project/vllm/commit/88bdd83329c9036892fe1d21cb1c13d3840055f6)    #66021
                                                                                                                  分支同步提交，11/7

  vLLM        [3c88e0a4](https://github.com/vllm-project/vllm/commit/3c88e0a4b360c828959e51b1333ef4569782161d)    #66375
                                                                                                                  分支同步提交，11/13
                                                                                                                  14:00，晚于本期截止

  vLLM        [e4c2b369](https://github.com/vllm-project/vllm/commit/e4c2b369718f7070ddf8db138d47f9a3c88e0ea7)    #67082 HIP IPC import
                                                                                                                  warm-up

  vLLM        [4f4dab57](https://github.com/vllm-project/vllm/commit/4f4dab57a9f2756b9326478b104078cab094cf64)    #65628 INT4 SP router
                                                                                                                  typing / tests

  vLLM        [f3615081](https://github.com/vllm-project/vllm/commit/f361508151c8f2e7b50e19256c83d3204f9e4d91)    #67010 AITER IPC
                                                                                                                  reduce-scatter

  vLLM        [41d8e838](https://github.com/vllm-project/vllm/commit/41d8e8380d0f8b021e87bde9ef9fc83e39f8dbeb)    #65943 Mooncake
                                                                                                                  TransferEngine reuse

  vLLM        [8b549478](https://github.com/vllm-project/vllm/commit/8b5494786a61eda21dbaa5edaf2944984c63254d)    #56192 RDMA OBJ
                                                                                                                  backend options

  SGLang      [44eba9a2](https://github.com/sgl-project/sglang/commit/44eba9a2703984c18111027a1f725ac0230e0df5)   #48241 shared
                                                                                                                  serialized payload

  SGLang      [5213d710](https://github.com/sgl-project/sglang/commit/5213d71080aa9bdc298d640eef0ce12285e5776b)   #48507 CUDA VMM
                                                                                                                  feature receive
                                                                                                                  budget

  SGLang      [69a16a90](https://github.com/sgl-project/sglang/commit/69a16a9053ddb0e19b238ac25410068a8742c946)   #49001 MI450 FABRIC
                                                                                                                  carv2

  SGLang      [38bdb79b](https://github.com/sgl-project/sglang/commit/38bdb79bafa996fa691eab39a5d93081a351345a)   #41495 NPU
                                                                                                                  weight-cache IPC

  SGLang      [d3b150f5](https://github.com/sgl-project/sglang/commit/d3b150f5631469402551d6a1eff4e234e2e9d2ad)   #45947 gated
                                                                                                                  multicast allgather

  SGLang      [dd142548](https://github.com/sgl-project/sglang/commit/dd14254811ab494efbc718258182db274af640e2)   #48031 NPU MLA sparse
                                                                                                                  AIV transfer

  SGLang      [4bf9ca2a](https://github.com/sgl-project/sglang/commit/4bf9ca2a3bf7f8dd2e1c2ea6e63d8d8059b46426)   #49277 Ascend MoE
                                                                                                                  AllGather

  SGLang      [3d2e518e](https://github.com/sgl-project/sglang/commit/3d2e518e94d0f33d9ae19d819c107c3867f741d1)   #49264 NPU sleep-comm
                                                                                                                  refusal guard

  SGLang      [65e9a4ff](https://github.com/sgl-project/sglang/commit/65e9a4fff06a0fb9bd083378c441ef20f9e695a0)   #48835 NPU EPLB
                                                                                                                  storage-preserving
                                                                                                                  path
  -------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 10. 统计结论

-   **vLLM：** Sparse NCCL 的批量验证和批量传输是本期主线；IPC 方面既有
    HIP IPC 权重同步问题修复，也有 CUDA IPC KV pull；M2N 的 retry RFC
    尚未在本期形成新实现。
-   **SGLang：** 本期没有直接 Sparse NCCL 权重传输 PR；进展集中在 NPU
    weight-cache IPC、HTTP payload 去重、ROCm custom all-reduce、NPU
    HCCL lifecycle、AIV HiCache 和 NIXL / Mooncake KV transfer。
-   **RDMA / WeightBridge / Connex / sparse_disk：** 本期有明确的 NIXL /
    Mooncake / MoRIIO RDMA 或 KV data-plane
    更新；未发现两个主仓中新增或更新的 WeightBridge、Connex 或
    `sparse_disk` 直接实现。不要把 HiSparse、SimpleCPUOffload disk tier
    或 NIXL OBJ storage 算作 Sparse NCCL。
-   **独立验证：** 本报告没有对任何 PR
    进行硬件复测。所有性能数据均按作者报告标注；CI pass / fail 也只按 PR
    页面公开记录描述。
