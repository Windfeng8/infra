---
title: 基于 HCCL 的 Checkpoint 坐标稀疏权重更新方案（RFC）
---

**相关 Issue/PR：**
[vllm-project/vllm-ascend#18915](https://github.com/vllm-project/vllm-ascend/pull/18915)\
**上游依赖：** [vllm-project/vllm 中的 sparse NCCL weight
transfer](https://github.com/vllm-project/vllm/blob/c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c/vllm/distributed/weight_transfer/sparse_nccl_engine.py)


------------------------------------------------------------------------

# 1. 概述

## 1.1 简介

本提案为 vLLM-Ascend 增加 `sparse_hccl` 权重传输后端，使 RL Trainer
能够向运行在独立昇腾 NPU 上的 vLLM 推理 Worker 发送 **Checkpoint
坐标下的稀疏权重更新**。方案复用 vLLM 上游的 Sparse NCCL
数据结构、更新元数据、Patch 校验和原生 Checkpoint
loader，将底层通信适配为 HCCL，并处理 Ascend MoE 推理权重与 Checkpoint
权重布局不一致的问题。

与全量 HCCL 权重同步相比，稀疏路径传输选中权重的扁平索引 `indices`
与新值 `values`，并通过 Checkpoint
名称、完整形状等元数据让接收端调用原生 loader。原生 loader 继续负责 TP
分片、专家归属和参数映射。当前方案要求 Trainer 与推理服务从相同的初始
Checkpoint 启动；更新原地生效且不是事务操作。

## 1.2 动机

RL 训练与推理部署通常运行在不同设备上。常规 HCCL
权重同步面向全量权重传输；当一次训练迭代只改变少量参数时，传输完整权重会产生不必要的通信量。

本方案针对以下问题：

1.  **减少稀疏更新的通信数据量。**
    当更新比例较低时，仅传输变更索引和值，避免每轮发送完整权重张量。
2.  **复用上游 Patch 语义。** 不另行发明一套模型参数寻址规则，而是沿用
    `SparseWeightPatch` 与 Checkpoint 原生 loader。
3.  **适配 Ascend 通信与执行环境。** 使用 HCCL 建立 Trainer---Worker
    通信组，并将上游依赖的 CUDA stream 调用映射到 NPU 环境。
4.  **保持 TP/EP 参数映射逻辑。** 使用 Checkpoint
    参数名称和形状表达更新，由原生 loader 处理推理侧参数分片与专家放置。
5.  **处理 MoE 权重视图差异。** Ascend 推理侧部分 MoE
    权重经过转置，接收更新时临时提供 Checkpoint loader
    期望的视图，结束后恢复原视图且不替换底层存储。

不实现该方案时，使用独立 Trainer NPU 的 RL
工作流仍需依赖全量权重传输，或自行维护稀疏 HCCL 的传输与参数映射逻辑。

## 1.3 目标

### 目标

-   注册 `sparse_hccl` Trainer 与 Worker 权重传输引擎。
-   复用上游 `SparseWeightPatch`、更新元数据和 Checkpoint patch
    加载逻辑。
-   通过 HCCL 传输每个非空 Patch 的 `indices` 与 `values`。
-   支持当前实现覆盖的 TP/DP 组合，并由原生 loader 处理专家权重放置。
-   在初始化失败和正常关闭时释放 HCCL communicator。
-   在稀疏更新完成后，按现有 RL 生命周期控制推理恢复。
-   通过单元测试、Checkpoint E2E 测试和示例验证参数一致性。


# 2. 用例分析

  -------------------------------------------------------------------------------------------------------------------------------------
  用例                    功能要求                                                   关键约束 / 验收点
  ----------------------- ---------------------------------------------------------- --------------------------------------------------
  初始化 Trainer 与       两侧交换初始化信息并加入同一个 HCCL 通信组                 Trainer 与 Worker 使用不同 NPU；通信组规模为
  Worker                                                                             `1 + TP * DP`

  发送稀疏 Patch          传递参数名、完整形状、数据类型、更新数量，再广播索引和值   两侧 Patch 顺序及 collective 次数必须一致

  应用 Patch              通过原生 Checkpoint loader 将逻辑 Checkpoint               验证每个 Worker 的参数摘要与 dense reference 一致
                          更新映射到运行时参数                                       

  更新 MoE 权重           临时恢复 Checkpoint loader 所需的权重视图                  原始参数存储地址与推理侧视图需在退出上下文后恢复

  空更新                  空 Patch 列表或无索引 Patch 不触发无效 collective          Sender 与 Receiver 不应发生 collective 次数不一致

  暂停与恢复推理          更新前暂停生成，成功后恢复                                 失败后继续保持暂停，直到恢复基线或重启服务

  服务关闭                释放 Worker 与 Trainer 的 HCCL communicator                `shutdown()` 可重复调用且不重复关闭资源

  性能评估                比较稀疏发送、索引应用和全量发送成本                       按更新比例测量；不能仅凭稀疏 payload
                                                                                     较小推断端到端更快
  -------------------------------------------------------------------------------------------------------------------------------------



# 3. 方案设计

## 3.1 总体方案

### 3.1.1 架构组成

``` mermaid
flowchart TB
    subgraph Trainer["Trainer 进程（独立 NPU，HCCL rank 0）"]
        A["训练后的模型 / Checkpoint 坐标"]
        B["生成 SparseWeightPatch"]
        C["SparseHCCLTrainerWeightTransferEngine"]
        A --> B --> C
    end

    subgraph Control["vLLM 权重同步控制面"]
        D["HTTPVLLMWeightSyncClient"]
        E["初始化 / 暂停 / 更新 / 完成 / 恢复 RPC"]
        D <--> E
    end

    subgraph Workers["vLLM-Ascend 推理 Worker（TP × DP 个 rank）"]
        F["SparseHCCLWeightTransferEngine"]
        G["HCCL communicator"]
        H["checkpoint_loader_views"]
        I["原生 Checkpoint patch loader"]
        J["运行时模型参数"]
        F <--> G
        F --> H --> I --> J
    end

    C <--> D
    C <--> G
    E <--> F
```

**架构说明：**

1.  Trainer 负责确定哪些 Checkpoint 坐标发生变化，并构造
    `SparseWeightPatch`。
2.  控制面通过现有 vLLM 权重同步客户端调用 Worker
    的初始化、开始更新、接收更新和结束更新接口。
3.  HCCL 负责传输 Patch 的索引与值；每个非空 Patch
    对应一次索引广播和一次值广播。
4.  Worker 复用上游接收和 Checkpoint patch 加载逻辑。Ascend
    侧在加载期间临时暴露 Checkpoint 期望的权重视图。
5.  更新成功后才由 RL
    控制流程恢复推理；异常时保留暂停状态，避免服务使用不完整或不一致的参数。

### 3.1.2 组件与代码位置

  ---------------------------------------------------------------------------------------------------------------------------------------------
  组件                    代码位置                                                             职责
  ----------------------- -------------------------------------------------------------------- ------------------------------------------------
  后端注册                `vllm_ascend/distributed/weight_transfer/__init__.py`                将 `sparse_hccl` 分别注册到 Worker 与 Trainer
                                                                                               factory

  Sparse HCCL 引擎        `vllm_ascend/distributed/weight_transfer/sparse_hccl_engine.py`      HCCL 初始化、稀疏发送、接收侧布局适配、资源关闭

  HCCL 通信引擎           `vllm_ascend/distributed/weight_transfer/hccl_engine.py`             复用现有 HCCL communicator 初始化与传输基础能力

  MoE 权重视图            `vllm_ascend/ops/fused_moe/routed_experts.py`                        校验专家权重格式，并临时将推理侧转置视图切换为
                                                                                               loader 期望的视图

  RL 示例                 `examples/rl/rlhf_sparse_hccl.py`                                    展示 Trainer 如何生成
                                                                                               Patch、暂停推理、发送更新并恢复

  单元测试                `tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py`    验证配置限制、空 Patch、NPU 同步与资源清理

  端到端测试              `tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py`   验证实际 HCCL、TP/EP 映射、参数摘要与生成结果

  文档                    `docs/source/user_guide/feature_guide/rl.md`                         说明 RL 权重传输后端、配置与使用限制
  ---------------------------------------------------------------------------------------------------------------------------------------------

### 3.1.3 Patch 数据模型

方案复用上游 `SparseWeightPatch`：

  ----------------------------------------------------------------------------------
  字段                    类型                    含义
  ----------------------- ----------------------- ----------------------------------
  `name`                  `str`                   Checkpoint 中的完整参数名称

  `full_shape`            `tuple[int, ...]`       参数在完整 Checkpoint 中的逻辑形状

  `indices`               `torch.Tensor`，一维    被更新元素在扁平逻辑张量中的索引
                          `int32`                 

  `values`                `torch.Tensor`，一维    与 `indices` 一一对应的新值
  ----------------------------------------------------------------------------------

`indices` 是逻辑 Tensor 的扁平元素索引，不是设备内存地址，也不是 HCCL
rank。接收端利用 `name`、`full_shape` 和索引和值构造 Checkpoint
patch，再由原生 loader 映射到对应的运行时参数。

更新元数据 `SparseHCCLWeightTransferUpdateInfo` 复用上游结构，包含：

-   `names`：本次 Patch 的参数名称列表；
-   `dtype_names`：各 Patch 的值数据类型；
-   `shapes`：各参数的完整 Checkpoint 形状；
-   `num_updates_list`：各 Patch 需要接收的索引/值数量。

元数据通过权重同步客户端发送；索引和值由 HCCL collective
传输。当前实现**不支持 packed sparse updates**，不能将其描述为已经实现的
Patch 打包优化。

### 3.1.4 HCCL 通信组设计

-   Trainer sender 使用 rank 0。
-   推理 Worker 从 rank offset 1 开始。
-   当前通信组成员数为 `1 + TP * DP`：一个 Trainer sender 加上 `TP × DP`
    个推理 Worker。
-   EP 不额外增加通信组成员数；PR 的 E2E 案例覆盖 `TP=4, DP=1, EP=4` 与
    `TP=2, DP=2, EP=4`，两种场景均使用 5 张 NPU（1 张 Trainer NPU + 4 个
    Worker rank）。
-   Trainer 与推理 Worker 必须使用相同的初始
    Checkpoint，但运行在不同设备上。
-   当前不支持 PP；不得将 PP
    维度直接计入当前实现的通信组规模并假设其可用。

### 3.1.5 初始化时序图

``` mermaid
sequenceDiagram
    autonumber
    participant T as Trainer（rank 0）
    participant C as HTTPVLLMWeightSyncClient
    participant S as vLLM Server / Worker RPC
    participant W as Worker Sparse HCCL Engine
    participant H as HCCL 通信组

    T->>S: POST /pause
    S-->>T: 暂停成功
    T->>C: trainer_init(world_size = 1 + TP × DP)
    par Worker 侧初始化 RPC
        C->>S: init_weight_transfer_engine(worker_init_info)
        S->>W: init_transfer_engine()
        W->>H: 加入 HCCL（rank offset = 1）
        H-->>W: Worker communicator ready
        W-->>S: 初始化完成
        S-->>C: RPC 成功
    and Trainer 侧初始化
        T->>H: trainer_init(rank = 0)
        H-->>T: Trainer communicator ready
    end
    C-->>T: 初始化握手成功
    Note over T,H: Trainer 与所有 Worker 必须加入同一 rendezvous；初始化失败时清理已创建的本地 communicator
```

说明：Trainer 端会在线程中发起 Worker 初始化 RPC，同时创建本地 rank 0
communicator，因为两侧需要并发完成 HCCL rendezvous。实现使用 `ExitStack`
注册清理动作，只有本地 communicator 创建和 Worker 初始化 RPC
均成功后才释放该清理回调。

### 3.1.6 稀疏权重更新时序图

``` mermaid
sequenceDiagram
    autonumber
    participant T as Trainer
    participant C as Trainer Weight Transfer Engine
    participant API as vLLM 权重同步 RPC
    participant W as Worker Sparse HCCL Engine
    participant H as HCCL collective
    participant L as 原生 Checkpoint loader
    participant M as Worker 模型参数

    T->>T: 根据训练结果生成 Patch 列表
    T->>API: POST /pause
    API-->>T: 推理已暂停
    T->>C: send_weights(patches)
    C->>API: start_weight_update()
    C->>API: update_weights(update_info)（异步发起）
    API->>W: receive_weights(update_info)

    loop 每个非空 Patch，按相同顺序执行
        C->>H: broadcast(indices, src=0)
        H-->>W: indices
        C->>H: broadcast(values, src=0)
        H-->>W: values
    end

    W->>W: 构造 CheckpointWeightPatch 列表
    W->>L: load_checkpoint_weight_patches(model, patches)
    Note over W,L: Ascend MoE 参数在临时 loader view 中应用更新
    L->>M: 更新对应运行时参数（原地修改）
    L-->>W: 加载完成
    W-->>API: update_weights RPC 返回
    API-->>C: 更新调用完成
    C->>C: 同步 Trainer 当前 NPU stream
    C->>API: finish_weight_update()
    API-->>C: 更新生命周期完成
    C-->>T: send_weights 返回成功
    T->>API: POST /resume
    API-->>T: 恢复推理
```

**关键时序约束：**

1.  Trainer 和 Worker 必须以相同顺序执行相同数量的 HCCL collective。
2.  每个非空 Patch 先广播 `indices`，再广播 `values`；元数据中的 Patch
    顺序、dtype、shape 和更新数量必须与发送端一致。
3.  空 Patch 在发送端被过滤；若过滤后没有
    Patch，则不进入无意义的广播流程。
4.  `send_weights()`
    成功返回后才进入恢复推理步骤。失败路径不应无条件调用 `/resume`。
5.  PR 的 Trainer 侧 `_post_send_sync()` 会同步当前 NPU stream；E2E
    仍需持续覆盖接收侧异步执行与恢复推理之间的时序，避免将 RPC
    返回简单等同于所有设备工作均已完成。

## 3.2 技术选型

  -------------------------------------------------------------------------------------------------------------------
  方案                    优点                                 缺点 / 不选择原因
  ----------------------- ------------------------------------ ------------------------------------------------------
  全量 HCCL (`hccl`)      路径成熟、语义简单，不需要稀疏索引   每轮发送完整权重；当更新比例低时会传输大量未变化数据

  稀疏 HCCL               仅传输更新索引和值；复用上游 Patch   增加索引广播、loader
  (`sparse_hccl`)         语义和原生 loader；适合低更新比例    应用和布局适配复杂度；更新比例高时不一定更快

  NPU IPC (`ipc`)         Trainer 与推理使用同一物理 NPU       不适用于本方案要求的独立 Trainer NPU 与推理 Worker NPU
                          时可避免跨设备 collective            场景

  完全自研稀疏传输引擎    可以从头控制元数据、传输和布局映射   会重复实现上游已有的 Patch 校验、接收与 loader
                                                               逻辑，增加维护成本和语义分叉风险
  -------------------------------------------------------------------------------------------------------------------

因此，本 PR 采用"**上游 sparse engine + Ascend HCCL 适配 + Ascend MoE
loader view**"的设计，而不是重新实现整套稀疏权重同步。

## 3.3 功能与性能设计

### 3.3.1 功能流程

1.  启动推理服务并注册 `sparse_hccl` 后端。
2.  Trainer 与推理 Worker 使用同一初始 Checkpoint；Trainer 使用独立
    NPU。
3.  初始化权重同步客户端和 HCCL 通信组。
4.  Trainer 生成 `SparseWeightPatch`，验证名称、完整形状、索引和值。
5.  暂停推理，开始权重更新生命周期。
6.  发送更新元数据；Worker 在 RPC 路径进入接收流程。
7.  每个非空 Patch 依次广播 `indices` 与 `values`。
8.  Worker 构造 Checkpoint patch，并调用原生 loader 应用更新。
9.  更新生命周期完成且必要的同步完成后恢复推理。
10. 关闭时显式关闭 Trainer 与 Worker 的 HCCL communicator。

### 3.3.2 Ascend MoE 权重视图适配

Ascend 的部分 MoE 推理权重采用适合 GMM 执行的转置视图，而上游 Checkpoint
loader 期望 Checkpoint 布局。PR 在 `routed_experts.py` 中增加：

-   `validate_checkpoint_weight_loader_view(layer)`：检查
    `w13_weight`、`w2_weight` 存在且是当前实现支持的
    ND、未拆分专家权重；
-   `checkpoint_weight_loader_view(layer)`：临时将参数视图切换为
    `original.transpose(1, 2)`，在 `finally` 中恢复原始 `.data` 视图；
-   `checkpoint_loader_views(model)`：遍历模型模块，进入支持该上下文的量化方法/权重方法所提供的视图上下文。

此设计的目的不是复制一份新权重，而是在更新期间提供 loader
需要的视图，并在结束后恢复推理侧布局。当前方案不支持需要不同映射方式的
split expert lists 和 NZ tensors。

### 3.3.3 配置与调用示例

PR 文档给出的基本选择方式如下：

``` bash
vllm serve /path/to/checkpoint \
  --weight-transfer-config '{"backend": "sparse_hccl"}' \
  --additional-config '{"rl_config": {"enabled": true}}
```

当前实现要求有效配置禁用 FRACTAL_NZ；RL 模式会将 `weight_nz_mode`
强制设为 `0`。PR 的实际示例也显式传入：

``` bash
--additional-config '{"weight_nz_mode":0}' \
--weight-transfer-config '{"backend":"sparse_hccl"}'
```

Trainer 示例命令形态：

``` bash
ASCEND_RT_VISIBLE_DEVICES=0 \
  python examples/rl/rlhf_sparse_hccl.py \
    --model /path/to/checkpoint \
    --served-model-name sparse-oracle \
    --url http://127.0.0.1:8000 \
    --tp 2 --dp 1
```

`--model` 必须指向与推理服务相同的初始 Checkpoint；`--tp`、`--dp`
必须与服务配置一致。实际部署前应按所用版本的 RL
文档配置暂停/恢复接口与设备可见性。

### 3.3.4 性能数据

PR #18915 给出了在 Ascend A5 上运行 Qwen3-1.7B 的通信延迟数据：

  -------------------------------------------------------------------------------------------
     更新比例      Sparse  Dense send Sparse send   `index_copy`   `index_copy / sparse send`
                  payload        (ms)        (ms)           (ms) 
                    (MiB)                                        
  ----------- ----------- ----------- ----------- -------------- ----------------------------
         0.1%        9.85      191.41       84.65          11.03                       13.03%

         0.3%       29.54      191.41       86.25          10.95                       12.70%

           1%       98.45      191.41       86.52          10.90                       12.60%

          10%      984.52      191.41       98.77          10.56                       10.69%

          50%     4922.60      191.41      252.66           9.89                        3.91%

         100%     9845.21      191.41      435.87          15.40                        3.53%
  -------------------------------------------------------------------------------------------

**数据解读：**

-   在 0.1%--10% 更新比例下，Sparse send 明显低于表中 Dense
    send，符合稀疏传输减少传输量的预期。
-   更新比例达到 50% 后，Sparse send 已高于 Dense send；100%
    更新时差距更明显。稀疏路径不是所有更新比例下都更优。
-   `index_copy` 是额外的更新应用开销，评估端到端耗时不能只比较
    `Sparse send`。简单相加 `Sparse send + index_copy` 的参考值分别约为
    95.68、97.20、97.42、109.33、262.55、451.27
    ms；该简单相加仅是分项参考，不等同于完整服务端到端延迟。
-   该表是特定模型与硬件下的结果。正式性能结论还应报告运行环境、重复次数、统计方式、方差、Patch
    分布以及暂停/恢复和 loader 开销。

### 3.3.5 验收标准

-   Trainer 与所有 Worker 成功建立 HCCL 通信组，rank/world size 与 TP/DP
    配置一致。
-   对每个 Worker 比较更新后参数摘要，与 native dense checkpoint loading
    的参考结果一致。
-   对确定性生成场景比较 greedy generation 结果与 dense reference 一致。
-   空 Patch 不触发无效 collective；非空 Patch
    的索引和值长度、dtype、形状不匹配时明确失败。
-   MoE loader view 在正常返回与异常退出时均恢复原始视图。
-   初始化 RPC 失败时，已创建的本地 communicator
    被关闭，原始异常被保留。
-   正常关闭时 communicator 释放一次；重复 `shutdown()` 不重复关闭资源。
-   不支持的配置在初始化 HCCL 通信前被拒绝。
-   失败的更新不会自动恢复生成；需验证服务仍保持暂停并提供恢复流程。



### 3.5.2 接口定义与设计

#### 3.5.2.1 后端配置（`sparse_hccl`）

-   **接口描述：** 选择 Ascend 稀疏 HCCL 权重同步后端。
-   **接口原型：**

``` bash
--weight-transfer-config '{"backend":"sparse_hccl"}'
```

-   **输入参数：**

  ----------------------------------------------------------------------------------------------------------
  参数名称              输入/输出       类型           描述                   取值范围
  --------------------- --------------- -------------- ---------------------- ------------------------------
  `backend`             输入            `str`          权重传输后端选择       `sparse_hccl`

  `weight_nz_mode`      输入/有效配置   `int`          禁用 FRACTAL_NZ        `0`
                                                       布局转换               

  `rl_config.enabled`   输入/有效配置   `bool`         启用 RL                `true`（按所用版本文档配置）
                                                       更新生命周期相关配置   
  ----------------------------------------------------------------------------------------------------------

-   **异常处理：** 当前实现会拒绝 PP、EPLB、量化权重以及不满足
    ND/专家权重视图要求的配置。
-   **约束说明：** 必须使用与当前实现兼容的 vLLM-Ascend 版本；Trainer 与
    Worker 需要使用相同初始 Checkpoint。
-   **变更说明：** PR 在 `WeightTransferEngineFactory` 和
    `WeightTransferTrainerFactory` 中注册 `sparse_hccl`。

#### 3.5.2.2 Patch 数据结构（`SparseWeightPatch`）

-   **接口描述：** 描述一个 Checkpoint 参数的稀疏更新。
-   **接口原型：**

``` python
@dataclass
class SparseWeightPatch:
    name: str
    indices: torch.Tensor
    values: torch.Tensor
    full_shape: tuple[int, ...]
```

-   **输入参数：**

  ------------------------------------------------------------------------------------
  参数名称       输入/输出      类型                描述           取值范围
  -------------- -------------- ------------------- -------------- -------------------
  `name`         输入           `str`               完整           必须与模型 loader
                                                    Checkpoint     可识别的名称对应
                                                    参数名         

  `indices`      输入           `torch.Tensor`      扁平逻辑索引   一维
                                                                   `int32`；元素数与
                                                                   `values` 一致

  `values`       输入           `torch.Tensor`      更新值         一维；dtype
                                                                   与目标参数兼容

  `full_shape`   输入           `tuple[int, ...]`   完整           与目标 Checkpoint
                                                    Checkpoint     参数一致
                                                    张量形状       
  ------------------------------------------------------------------------------------

-   **异常处理：** 缺失
    `full_shape`、索引类型错误、非一维输入或索引和值长度不匹配时应拒绝发送。
-   **约束说明：** 需遵循上游 patch 校验与 loader
    约束；索引是逻辑坐标，不是物理地址。
-   **调用参考：** 参见 `examples/rl/rlhf_sparse_hccl.py` 的
    `make_patches()`。

#### 3.5.2.3 Trainer 初始化与发送

-   **接口描述：** 初始化 Trainer 侧 HCCL engine 并发送一轮稀疏更新。
-   **调用参考：**

``` python
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=SparseHCCLTrainerInitInfo(
        master_address=master_address,
        master_port=master_port,
        world_size=1 + tp * dp,
        rank=0,
    ),
    client=HTTPVLLMWeightSyncClient(url),
)

engine.send_weights(patches)
engine.shutdown()
```

以上为接口形态示意；具体初始化字段、factory 注册方式与 import
路径应以当前 PR 对应的 vLLM/vLLM-Ascend 版本为准。

-   **异常处理：** 初始化失败时需关闭已创建的本地
    communicator；更新失败时保持推理暂停，并按运维策略恢复基线或重启。
-   **约束说明：** 只有指定的 Trainer sender
    加入该稀疏发送通信组；当前稀疏 Patch 每轮直接传入
    `send_weights()`，不是稳定的 `WeightSource`。


# 5. 现有技术

1.  **vLLM Sparse NCCL：** 提供
    `SparseWeightPatch`、稀疏更新元数据、Patch 校验、Trainer
    发送逻辑和原生 Checkpoint patch loader。PR #18915
    复用该实现，并替换通信器初始化与通信后端。
2.  **vLLM-Ascend HCCL 权重传输：** 现有 `hccl` 后端面向独立 NPU
    间的全量权重传输。`sparse_hccl` 复用 HCCL communicator
    基础设施，但传输的是稀疏索引和值。
3.  **NPU IPC：** 适用于 Trainer 与 rollout engine 使用同一物理 NPU
    的情况，不是当前独立 Trainer NPU 场景的替代方案。
4.  **Ascend MoE 权重布局：** 推理执行所需的权重视图可能与 Checkpoint
    loader 的预期布局不同。PR 新增临时 loader
    view，以减少对上游参数映射逻辑的重复实现。

差异总结：本方案不是把 HCCL 变成"稀疏 collective"；它是在 HCCL
上传输稀疏 Patch 的索引和值，并由 Worker 的原生 loader
完成参数更新。HCCL 本身仍执行标准 collective 调用。



------------------------------------------------------------------------

# 附录

## A. PR 当前报告的测试结果

PR 描述中列出的测试结果如下；这些是 PR 提供的验证记录，并不意味着当前 PR
已合入主分支。

### 单元测试

``` bash
python -m pytest \
  tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py \
  tests/ut/ops/test_routed_experts.py -q
```

PR 报告：`35 passed, 15 warnings in 0.16s`。覆盖 Sparse HCCL engine 与
routed-expert 相关测试，包括不支持配置、communicator 清理、空 Patch、NPU
同步和 loader view 恢复。

### Checkpoint E2E

``` bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7,12 \
  python -m pytest -x -rs \
  tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py
```

PR 报告：`2 passed, 14 warnings in 198.19s`。

    TP   DP   EP   NPU 总数
  ---- ---- ---- ----------
     4    1    4          5
     2    2    4          5

两组用例分别执行。总 NPU 数包括 1 个独立 Trainer 与 `TP × DP` 个
Worker。测试比较每个 Worker 的参数 hash 和 greedy generation 与 native
dense checkpoint loading 的结果。

### 实际 HCCL 示例

PR 使用一个两层、八专家的合成 Qwen3 MoE
Checkpoint。示例交换首尾层中两个专家的 gate projection
的前两行。对该示例，四个 Patch 共传输 `8192` 个稀疏值。PR 报告四个
Worker 的参数摘要和生成结果与 dense reference
一致；这验证了小型合成模型上的 Patch 生成、HCCL
传输与恢复推理，不代表预训练模型准确率或大模型性能结论。

## B. 参考链接

-   [PR #18915：Add checkpoint sparse HCCL weight
    updates](https://github.com/vllm-project/vllm-ascend/pull/18915)
-   [PR
    文件变更](https://github.com/vllm-project/vllm-ascend/pull/18915/files)
-   [Sparse HCCL RL 用户指南（PR
    中的文档变更）](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/rl.md)
-   [Sparse HCCL
    示例](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rl/rlhf_sparse_hccl.py)
-   [上游 Sparse NCCL engine（PR 描述引用的
    commit）](https://github.com/vllm-project/vllm/blob/c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c/vllm/distributed/weight_transfer/sparse_nccl_engine.py)
-   [Ascend
    社区安全设计规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)

## C. 术语表

  ----------------------------------------------------------------------------------
  术语                                说明
  ----------------------------------- ----------------------------------------------
  HCCL                                昇腾设备间通信库

  Sparse Patch                        对一个 Checkpoint 参数的稀疏更新描述

  `indices`                           扁平逻辑张量中待更新元素的位置，不是物理地址

  `values`                            与 `indices` 一一对应的新权重值

  Checkpoint 坐标                     以完整模型 Checkpoint
                                      参数名称和逻辑形状表示的位置体系

  TP                                  Tensor Parallelism，张量并行

  DP                                  Data Parallelism，数据并行

  EP                                  Expert Parallelism，专家并行

  EPLB                                Expert Parallelism Load
                                      Balancing，专家并行负载均衡

  ND                                  普通 N 维 Tensor 布局

  FRACTAL_NZ                          昇腾矩阵计算使用的特殊分块权重布局

  Native loader                       vLLM 原生 Checkpoint
                                      权重加载逻辑，负责将逻辑权重映射到运行时参数
  ----------------------------------------------------------------------------------


