# 最优配置复现命令手册 (`COMMANDS.md`)

所有测试均在宿主机 `codex-177.42.22.1` 的容器 `qwen3.8_flash_next_w00893744_1117` 内执行，无需修改任何 `vllm` 或 `vllm-ascend` 源码，不注入任何 `PYTHONPATH`。

## 1. 公共环境变量初始化

```bash
source /mnt/share/w00893744/cann_b020/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_NPU_SOCKET_PORT_RANGE=60100-60150
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

---

## 2. 纯文本 `2K -> 6K` 最优服务端与压测命令

### 服务端启动（对应 `configs/candidates/text_opt_prefill_boost.json`）

```bash
vllm serve /mnt/weight/Qwen3.8-Flash-Next \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 18599 \
  --trust-remote-code \
  --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
  --api-server-count 4 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-batched-tokens 6528 \
  --max-num-seqs 296 \
  --disable-prefix-caching --mamba-cache-mode none \
  --skip-mm-profiling \
  --engram-config '{"cpu_offload": true, "embedding_across_dp": false, "dp_shared_memory": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 272, 274, 276, 278, 288, 294, 296]}' \
  --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": false, "multistream_overlap_shared_expert": true}' \
  --generation-config vllm --seed 20261030
```

### 客户端压测（`C=544` 零抢占 `12,736.80 tok/s`；将 `544` 改为 `576` 即峰值 `12,774.26 tok/s`）

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:18599 \
  --model qwen3.8-flash-next \
  --tokenizer /mnt/weight/Qwen3.8-Flash-Next \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 2048 --random-output-len 6144 --random-range-ratio 0 \
  --ignore-eos --temperature 0 --seed 20261030 \
  --num-prompts 544 --max-concurrency 544 \
  --metric-percentiles 50,90,99
```

---

## 3. 多模态 `4K + 1×(1920×1080) -> 32K` 最优服务端与压测命令

### 服务端启动（对应 `configs/candidates/mm_opt_winner.json`）

```bash
vllm serve /mnt/weight/Qwen3.8-Flash-Next \
  --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 18598 \
  --trust-remote-code \
  --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel \
  --api-server-count 4 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 39424 \
  --max-num-batched-tokens 2560 \
  --max-num-seqs 88 \
  --disable-prefix-caching --mamba-cache-mode none \
  --mm-encoder-tp-mode data --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"min_pixels": 65536, "max_pixels": 2088960}' \
  --engram-config '{"cpu_offload": true, "embedding_across_dp": false, "dp_shared_memory": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 72, 80, 84, 85, 86, 87, 88]}' \
  --additional-config '{"enable_cpu_binding": true, "enable_flashcomm1": false, "multistream_overlap_shared_expert": true}' \
  --generation-config vllm --seed 20261030
```

### 客户端压测（`C=172` 精确零抢占 `5,893.28 tok/s`；将 `172` 改为 `176` 即峰值 `5,972.17 tok/s`）

```bash
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:18598 \
  --model qwen3.8-flash-next \
  --tokenizer /mnt/weight/Qwen3.8-Flash-Next \
  --trust-remote-code \
  --dataset-name random-mm \
  --random-input-len 4096 --random-output-len 32768 --random-range-ratio 0 \
  --random-mm-base-items-per-request 1 \
  --random-mm-num-mm-items-range-ratio 0 \
  --random-mm-bucket-config '{(1080, 1920, 1): 1.0}' \
  --ignore-eos --temperature 0 --seed 20261030 \
  --num-prompts 172 --max-concurrency 172 \
  --metric-percentiles 50,90,99
```

---

## 4. 稳态解码 Profiling 触发与离线解析命令

```bash
# 1. 使用 configs/text_profiler_ready.json 或 configs/mm_profiler_ready.json 启动隔离服务（需设置 VLLM_SERVER_DEV_MODE=1 且 api_server_count=1）
# 2. 在全部并发请求进入纯稳态解码后触发 20 步采集：
curl -X POST http://127.0.0.1:18582/start_profile
sleep 4.5
curl -X POST http://127.0.0.1:18582/stop_profile

# 3. 停服后运行 CANN 9.3.0 离线轨迹解析与算子家族聚合：
python /mnt/share/w00893744/qwen3.8_flash_next/project_1117_native_offload_profiling/scripts/profile_collector_and_parser.py --analyse-only
```
