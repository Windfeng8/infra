"""Split-core GenericBlockSparseAttention (GBSA) decode recipe for Qwen4Exp QSA.

[Role]
  Price the split-core MIX_AIC GenericBlockSparseAttention decode kernel used by
  ``vllm_ascend/models/qwen4_exp/gbsa.py`` on Ascend 910C / 950DT dies.
[Invariants]
  * Separate BF16 K and V planes of ``head_dim`` (not latent MLA single-plane).
  * Physical page gather runs on AIV (``mte2_aiv`` -> UB -> ``mte3`` GM scratch),
    followed by AIC ``qk_load``/``qk``/``fixpipe`` scores, AIV ``softmax``, and
    AIC ``pv_load``/``pv``/``fixpipe`` partial output accumulation.
  * Cube operands and score matrices align query-head and key tiles to 16-row
    Cube fractals; ``GBSA_PAGE_SIZE = 128`` is the default key tile.
  * Targets without split AIC/AIV copy pipes fall back to the generic tiled
    attention mechanism so non-Ascend hardware profiles remain valid.
[Timeouts & Locks]
  Bounded by query owners and selected top-k key tiles; no I/O or locks.
[Side-Effect Policy]
  Pure OperatorPlan construction and virtual Engine submissions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_pipeline import TiledFlashAndPagedAttentionRecipe
from .indexed_attention import _align, build_pipeline
from .mechanism import OperatorPlan, PlannedRecipe, Work

QSA_GBSA_RECIPE = "qsa_gbsa_decode"
GBSA_KEY_TILE = 128
GBSA_SLOTS = 2
SYNC_FLAG_BYTES = 1024
DESCRIPTOR_BYTES_PER_KEY = 8
ROW_ALIGN_BYTES = 32
BF16_BYTES = 2
FP32_BYTES = 4
CUBE_FRACTAL = 16


SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "path": "vllm_ascend/models/qwen4_exp/gbsa.py",
    "scope": "BF16 separate K/V GenericBlockSparseAttention decode, GBSA_PAGE_SIZE=128",
}


def _has_split_pipes(geometry) -> bool:
    units = dict(getattr(geometry, "resource_units", {}) or {})
    cores = int(getattr(geometry, "compute_units", 0) or 0)
    return (
        cores > 0
        and units.get("mte2_aiv", 0) > 0
        and units.get("mte2_aic", 0) > 0
        and units.get("mte3", 0) > 0
        and units.get("fixpipe", 0) > 0
        and units.get("vector", 0) > 0
    )


def _gbsa_phases(
    keys: int, heads: int, head_dim: int, _rope_unused: int, *,
    first: bool, last: bool, scratch_policy: str, operand_modes: bool = False,
):
    """Physical tile work for GenericBlockSparseAttention (separate K & V, BF16)."""
    padded_h = _align(heads, CUBE_FRACTAL)
    padded_k = _align(keys, CUBE_FRACTAL)
    query_bytes = heads * head_dim * BF16_BYTES
    k_payload = keys * head_dim * BF16_BYTES
    v_payload = keys * head_dim * BF16_BYTES
    kv_payload = k_payload + v_payload
    row_trans = _align(head_dim * BF16_BYTES, ROW_ALIGN_BYTES)
    kv_trans = 2 * keys * row_trans
    score = padded_h * padded_k * FP32_BYTES
    probability = heads * padded_k * BF16_BYTES
    partial_output = padded_h * head_dim * FP32_BYTES
    output = heads * head_dim * BF16_BYTES if last else 0
    offchip = scratch_policy == "offchip"

    def scratch(read: int = 0, write: int = 0, *, vector: bool = False, fixpipe: bool = False):
        resource = "mte2_aiv" if vector else "mte2_aic"
        if not offchip:
            resource += "_cached"
        work = {resource: read} if read else {}
        if write:
            work["fixpipe" if fixpipe else "mte3"] = write
        if offchip:
            work["hbm"] = read + write
        else:
            if read:
                work["l2_read"] = read
            if write:
                work["l2_write"] = write
        return work

    def atomic(write: int, *, fixpipe: bool = False):
        work = scratch(write=write, fixpipe=fixpipe)
        role = "hbm" if offchip else "l2_read"
        work[role] = work.get(role, 0) + write
        return work

    def operands(a: int, b: int):
        return ({"mte1_l0a": a, "mte1_l0b": b} if operand_modes else {"mte1": a + b})

    qk_inner = operands(
        padded_h * ((keys + GBSA_KEY_TILE - 1) // GBSA_KEY_TILE) * head_dim * BF16_BYTES,
        padded_k * head_dim * BF16_BYTES,
    )
    pv_reads = v_payload + probability
    pv_inner = operands(
        padded_h * ((head_dim + GBSA_KEY_TILE - 1) // GBSA_KEY_TILE) * padded_k * BF16_BYTES,
        head_dim * padded_k * BF16_BYTES,
    )
    return (
        ("gather", Work(resources={
            "hbm": kv_trans + keys * DESCRIPTOR_BYTES_PER_KEY,
            "mte2_aiv": kv_trans + keys * DESCRIPTOR_BYTES_PER_KEY,
        })),
        ("publish", Work(resources=scratch(write=kv_payload + SYNC_FLAG_BYTES))),
        ("qk_load", Work(resources=scratch(read=k_payload)) + Work(resources={
            "hbm": query_bytes, "mte2_aic": query_bytes, **qk_inner,
        })),
        ("qk", Work(matrix={"bf16": 2 * padded_h * padded_k * head_dim})),
        ("scores", Work(resources=scratch(write=score, fixpipe=True))),
        ("softmax", Work(resources={
            **scratch(read=heads * padded_k * FP32_BYTES + SYNC_FLAG_BYTES, vector=True),
            "vector": 8 * heads * padded_k + 26 * heads,
        })),
        ("probability", Work(resources=scratch(write=probability))),
        ("pv_load", Work(resources={**scratch(read=pv_reads), **pv_inner})),
        ("pv", Work(matrix={"bf16": 2 * padded_h * padded_k * head_dim})),
        ("nupdate", Work(resources={
            **(scratch(write=partial_output) if first else atomic(partial_output)),
            "vector": padded_h * (head_dim if first else GBSA_KEY_TILE),
        })),
        ("partial", Work(resources=atomic(partial_output, fixpipe=True))),
        ("update", Work(resources={
            **scratch(read=heads * head_dim * FP32_BYTES if last else 0, vector=True),
            "vector": heads * head_dim * 5 if last else 0,
        })),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    )


@dataclass(frozen=True)
class QSAGBSADecodeRecipe:
    """Split-core GBSA decode recipe with automatic fallback on non-split targets."""

    key_tile: int = GBSA_KEY_TILE
    slots: int = GBSA_SLOTS
    scratch_policy: str = "offchip"
    operand_modes: bool = True

    def build(self, op, geometry) -> OperatorPlan:
        meta = dict(getattr(op, "metadata", ()) or ())
        gqa_dec = meta.get("gqa_decode")
        if (
            not isinstance(gqa_dec, dict)
            or not _has_split_pipes(geometry)
            or str(gqa_dec.get("key_format", "bf16")) not in {"bf16", "fp16", "bfloat16", "float16"}
        ):
            return TiledFlashAndPagedAttentionRecipe().build(op, geometry)
        requests = tuple(gqa_dec.get("requests", ()))
        heads = max(1, int(gqa_dec.get("query_heads", 1)))
        kv_heads = max(1, int(gqa_dec.get("kv_heads", 1)))
        head_dim = max(1, int(gqa_dec.get("head_dim", 128)))
        topk = max(1, int(gqa_dec.get("window", 2048)))
        pool_size = max(1, int(meta.get("qsa_index_kpool", 4)))
        sequences = tuple((int(q) * kv_heads, int(c)) for q, c in requests if int(q) > 0)
        if not sequences:
            return TiledFlashAndPagedAttentionRecipe().build(op, geometry)
        directional = bool(
            self.operand_modes
            and {"mte1_l0a", "mte1_l0b"} <= set(dict(getattr(geometry, "resource_units", {}) or {}))
        )
        shape = {
            "sequences": sequences,
            "topk": topk,
            "heads": max(1, heads // kv_heads),
            "latent": head_dim,
            "rope": 0,
            "index_pool_size": pool_size,
            "index_pool_tail": True,
        }
        plan = build_pipeline(
            shape,
            geometry,
            scratch_policy=self.scratch_policy,
            key_tile=self.key_tile,
            slots=self.slots,
            compact=True,
            operand_modes=directional,
            phase_factory=_gbsa_phases,
        )
        plan_meta = dict(plan.metadata)
        plan_meta["source"] = dict(SOURCE)
        return OperatorPlan(plan.work, plan.program, metadata=plan_meta, native=plan.native)


def build_gbsa(op, geometry, **options) -> OperatorPlan:
    return QSAGBSADecodeRecipe(**options).build(op, geometry)


def recipe(**options) -> PlannedRecipe:
    return PlannedRecipe(lambda op, geometry: QSAGBSADecodeRecipe(**options).build(op, geometry))


RECIPE = PlannedRecipe(build_gbsa)
