"""Topology collective physical recipe.

[Role] Lower a routed collective into pack, directed peer phases, reduction and
  unpack phases on the shared link and memory resources.
[Invariants] Per-rank wire work comes from the topology plan; this recipe never
  re-applies a bus factor. A topology-declared direct mesh is consumed as its
  physical plan: ReduceScatter then AllGather for AllReduce, one peer phase for
  ReduceScatter/AllGather, each issuing P/W per directed peer concurrently.
  Non-mesh collectives retain the ring/NHR/CCU phase split. AllReduce retains
  two dependent phases; other collectives retain one peer phase.
[Timeouts & Locks] Bounded phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import dtype_bytes
from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, phased_program)
from .program import KernelProgram, Stage

HALF_SHARE = 0.5
PACK_READ_SHARE = 0.25
UNPACK_WRITE_SHARE = 0.25
REDUCE_VECTOR_SHARE = 0.50
ONESHOT_PAYLOAD_LIMIT_BYTES = 2 * 1024 * 1024
MESH_ALGORITHM = "direct_mesh"
ALLREDUCE_FAMILIES = frozenset({"allreduce", "tp_allreduce"})
SINGLE_PHASE_FAMILIES = frozenset({
    "allgather", "reduce_scatter", "alltoall", "engram_broadcast", "engram_ids",
})


def _pack_phase(read_share):
    return Phase("pack", {IO_HBM_CATEGORY: read_share, "mte2": 1.0})


def _unpack_phase(write_share, after, *, atomic_rmw: bool = False):
    weights = {
        IO_HBM_CATEGORY: write_share,
        "mte3": 1.0,
    }
    if atomic_rmw:
        weights["hbm_atomic_rmw"] = 1.0
    return Phase("unpack_store", weights, after=after)


def _with_hccl_aiv_local_work(context):
    """Derive local AIV buffer staging and atomic reduction on split-core dies."""
    op = context.op
    if float(getattr(op, "hbm_bytes", 0.0) or 0.0) > 0.0:
        return (context, io_weight(context, "hbm_read"),
                io_weight(context, "hbm_write"), False)
    link_bytes = float(getattr(op, "link_bytes", 0.0) or 0.0)
    names = dict(getattr(context, "resource_names", ()) or {})
    local_shape = tuple(getattr(op, "local_shape", ()) or ())
    elems = math.prod(int(d) for d in local_shape) if local_shape else 0
    if (link_bytes <= 0.0 or elems <= 1
            or "mte2_aiv" not in names or "mte3" not in names
            or "hbm" not in names):
        return (context, io_weight(context, "hbm_read"),
                io_weight(context, "hbm_write"), False)
    family = str(getattr(op, "family", ""))
    elem_bytes = float(dtype_bytes(str(getattr(op, "dtype", "") or "bf16")))
    payload = float(elems) * elem_bytes
    atomic_rmw = False
    if family in ALLREDUCE_FAMILIES:
        meta = dict(getattr(op, "metadata", ()) or ())
        steps = int(meta.get("steps", 0) or 0)
        if steps >= 2:
            ranks = steps // 2 + 1
        else:
            ratio = link_bytes / max(2.0 * payload, 1e-9)
            ranks = max(2, int(round(1.0 / max(1e-3, 1.0 - min(0.999, ratio)))))
        hbm_read = payload
        mte2_aiv = payload
        if payload <= ONESHOT_PAYLOAD_LIMIT_BYTES:
            mte3 = float(ranks - 1) * payload
            hbm_write = 2.0 * float(ranks - 1) * payload
        else:
            shard = payload * float(ranks - 1) / float(ranks)
            mte3 = shard + payload
            hbm_write = 2.0 * shard + payload
        atomic_rmw = True
    elif family in {"alltoall", "alltoallv", "allgather", "reduce_scatter"}:
        hbm_read = link_bytes
        mte2_aiv = link_bytes
        hbm_write = link_bytes
        mte3 = link_bytes
    else:
        return (context, io_weight(context, "hbm_read"),
                io_weight(context, "hbm_write"), False)
    total_hbm = hbm_read + hbm_write
    work = dict(context.work)
    work[names["hbm"]] = work.get(names["hbm"], 0.0) + total_hbm
    work[names["mte2_aiv"]] = work.get(names["mte2_aiv"], 0.0) + mte2_aiv
    work[names["mte3"]] = work.get(names["mte3"], 0.0) + mte3
    updated = replace(context, work=work)
    return updated, hbm_read / total_hbm, hbm_write / total_hbm, atomic_rmw


def _mesh_phases(context, phases, family, *, read_share: float | None = None,
                 write_share: float | None = None, atomic_rmw: bool = False):
    """Physical phase weights around the plan-owned peer transports."""
    r_share = io_weight(context, "hbm_read") if read_share is None else read_share
    w_share = io_weight(context, "hbm_write") if write_share is None else write_share
    result = [_pack_phase(r_share)]
    reduce = (REDUCE_VECTOR_SHARE if family in ALLREDUCE_FAMILIES
              else 1.0)
    for name in phases:
        result.append(Phase(name, {"vector": reduce}, after=(result[(-1)].name,)))
    result.append(_unpack_phase(w_share, after=(result[(-1)].name,),
                                atomic_rmw=atomic_rmw))
    return tuple(result)


@dataclass(frozen=True)
class TopologyCollectiveRecipe:
    """Class E: direct-mesh or hierarchical collective phase plan."""

    traffic_rewrite_capable: bool = True
    #: The topology may declare a direct full-mesh plan for this invocation.
    mesh_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        binding = context.collective_plan
        if getattr(binding, "algorithm", None) == MESH_ALGORITHM:
            return _lower_mesh(context, binding())
        context, read_share, write_share, atomic_rmw = _with_hccl_aiv_local_work(context)
        family = str(getattr(context.op, "family", ""))
        if family in ALLREDUCE_FAMILIES:
            phases = (
                _pack_phase(read_share),
                Phase("reduce_scatter", {"link": HALF_SHARE,
                                         "vector": REDUCE_VECTOR_SHARE},
                      after=("pack",)),
                Phase("all_gather", {"link": HALF_SHARE,
                                     "vector": REDUCE_VECTOR_SHARE},
                      after=("reduce_scatter",)),
                _unpack_phase(write_share, after=("all_gather",),
                              atomic_rmw=atomic_rmw),
            )
            return _with_setup(context, phases)
        phases = (
            _pack_phase(read_share),
            Phase("peer_exchange", {"link": 1.0},
                  after=("pack",)),
            _unpack_phase(write_share, after=("peer_exchange",),
                          atomic_rmw=atomic_rmw),
        )
        return _with_setup(context, phases)


def _lower_mesh(context, plan):
    """Submit the selected plan's dependent phases; never re-split ring demand."""
    plan.validate_path_model("direct")
    context, read_share, write_share, atomic_rmw = _with_hccl_aiv_local_work(context)
    family = str(getattr(context.op, "family", ""))
    # The plan owns all link work.  Keep only local packaging/reduction work in
    # the stages that wrap the physical peer programs, so no byte is counted
    # twice.
    link_names = frozenset(plan.wire_by_resource())
    work = {name: value for name, value in dict(context.work).items()
            if name not in link_names}
    phases = _mesh_phases(context, tuple(plan.phase_names), family,
                          read_share=read_share, write_share=write_share,
                          atomic_rmw=atomic_rmw)
    base = phased_program(replace(context, work=work), phases)
    stages = []
    for stage in base.stages:
        if stage.name in plan.phase_names:
            # A dependent phase is a real program boundary: its startup and
            # every directed peer message come from the topology plan.
            stage = replace(stage, program=plan.phase_program(stage.name),
                            keep_empty=True)
        stages.append(stage)
    return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)


def _with_setup(context, phases):
    # The setup stage is part of the invocation, so the outer program owns the
    # inclusive floor. Keep the nested body floor-free: charging it again after
    # setup would add setup_ms to the declared completion response.
    program = phased_program(
        context, phases, minimum_ms=0.0 if context.setup_ms > 0.0 else None)
    if context.setup_ms > 0.0:
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",), program=program),
        ), minimum_ms=context.minimum_ms)
    return program


CollectiveRecipe = TopologyCollectiveRecipe
