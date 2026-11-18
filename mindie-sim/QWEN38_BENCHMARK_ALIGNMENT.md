# Qwen3.8-Flash-Next (`Qwen4Exp`) 8×Ascend 950DT Modeling Alignment

This directory contains `mindie-sim` with five physical mechanism fixes aligning high-concurrency `2K -> 6K` decode throughput and KV/PLE HBM capacity on `8×Ascend 950DT` (`Ascend950DT_9582`, `BF16` weights, `FP32` GDN state, `TP=4/DP=2 + EP=8`).

## 1. Root Causes & Mechanism Fixes

1. **`Qwen4Exp` Sequence Parallel Default (`va_profiles.py`, `compose.py`)**
   - `vllm_ascend/models/qwen4_exp/qwen4_exp.py` runs TP-replicated low-rank gated HyperConnection (`hc_lowrank > 0`) without FlashComm/SP (`sp=False`). Disabling spurious `sp=True` auto-enabling removes phantom `reduce_scatter` / `allgather` pairs while restoring full token rows on TP-sharded norms/projections.
2. **QSA `GenericBlockSparseAttention` (`qsa_gbsa.py`, `attention.py`, `program.py`)**
   - Implements `QSAGBSADecodeRecipe` (`qsa_gbsa_decode`) matching `vllm_ascend/models/qwen4_exp/gbsa.py` (`GBSA_KEY_TILE = 128`, separate BF16 `K`/`V` planes, 16-row Cube fractal alignment, split-core `mte2_aiv` block gather -> GM scratch -> `mte2_aic` QK/PV -> `fixpipe` -> `vector` online softmax), plus per-step `qsa_compress_groups` (`4:1` key-pool update) and `gbsa_metadata` preparation.
3. **Continuous `mte2_aic_dram` / `mte2_aiv_dram` & `hbm_atomic_rmw` Service Bounds (`matrix_steady.py`, `physical_recipe_base.py`, `vector_pipeline.py`, `gdn_recurrent.py`)**
   - Extends `SteadyMatmul` per-owner `mte2_aic_dram` read bounding smoothly from `nm = 1` to `nm >= 2` (`total_m > 128`), bounds AIV DRAM load phases and `RecurrentGatedDeltaRule` cold state loads by declared `mte2_aiv_dram`, fixes `io_weight(..., "hbm_read")` across phased pipelines, and enforces `hbm_atomic_rmw` on `route_unpack` (`top_k = 10` MoE atomic scatter).
4. **HCCL AIV Local Staging & Atomic Reduction (`topology_collective.py`)**
   - Models local AIV GM staging and `OneShot`/`TwoShot` `hbm_atomic_rmw` reductions (`hccl_aiv.py`) on split-core Ascend collectives (`allreduce`, `tp_allreduce`, `alltoall`).
5. **`Qwen4KVCacheLayout` Shared `896`-Token Pool & `engram_across_dp` (`qwen4_exp.py`, `weights.py`, `capacity.py`)**
   - Wires the shared `12`-slot QSA + `3` GDN state group + `1` raw circular page pool (`11,895,936 B/block`, `896 tokens/block`) into `Memory` and adds `engram_across_dp` (`ETP = TP * DP`) sharding and collectives for PLE n-gram tables.

## 2. Verification Summary (`8×Ascend 950DT`, `BF16`, `FP32` SSM)

### 2.1 Torch-NPU Silicon Profiler (`TP=8, DP=1, EP=8`, 48 Layers)

| Workload | Measured (ms) | Predicted (ms) | Error |
| :--- | :---: | :---: | :---: |
| `TP=8, C=4` | `17.49` | `17.23` | `-1.5%` |
| `TP=8, C=56` | `29.63` | `29.69` | `+0.2%` |

### 2.2 Pure Text `2048 -> 6144` End-to-End Serving (`TP=4, DP=2, EP=8`)

| Concurrency & Mode | Measured TPOT (ms) | Predicted TPOT (ms) | TPOT Error | Measured E2E (tok/s) | Before Fix E2E Error | After Fix E2E (tok/s) | After Fix E2E Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C=256 (ETP=4)` | `37.93` | `40.72` | `+7.4%` | `6,502.9` | `+3.1%` | `6,001.2` | `-7.7%` |
| `C=272 (ETP=4)` | `39.66` | `41.59` | `+4.9%` | `6,582.8` | `+5.8%` | `6,233.4` | `-5.3%` |
| `C=368 (ETP=8)` | `44.71` | `46.33` | `+3.6%` | `7,918.2` | `+11.8%` | `7,493.5` | `-5.4%` |
| `C=384 (ETP=8)` | `45.78` | `46.30` | `+1.1%` | `8,026.4` | `+13.3%` | `7,802.2` | `-2.8%` |
| `C=416 (ETP=8)` | `48.02` | `48.35` | `+0.7%` | `8,233.2` | `+16.7%` | `8,078.2` | `-1.9%` |
| `C=416 (Host UVA)` | `48.72` | `48.16` | `-1.1%` | `8,100.6` | `+18.7%` | `8,108.1` | `+0.1%` |
| `C=544 (Host UVA)` | `55.96` | `52.99` | `-5.3%` | `9,190.4` | `+24.3%` | `9,526.9` | `+3.7%` |
| `C=576 (Host UVA)` | `58.17` | `54.23` | `-6.8%` | `9,318.9` | `+27.9%` | `9,832.8` | `+5.5%` |

### 2.3 HBM Weights & Max Concurrency (`TP=4, DP=2, EP=8`, `mem=0.90`)

| PLE Placement Mode | Measured Weights/Card | Predicted Weights/Card | Measured MM Max `C` | Predicted MM Max `C` | Measured Text Max `C` | Predicted Text Max `C` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ETP=4` (TP-local) | `56.61 GiB` | `56.18 GiB` | `84` | `82` | `270` | `282` |
| `ETP=8` (`engram_across_dp=True`) | `44.69 GiB` | `44.26 GiB` | `126` | `126` | `406` | `436` |
| `Host UVA` (`engram_cpu_offload=True`) | `32.77 GiB` | `32.34 GiB` | `172` | `172` | `554` | `592` |
