"""Recursive path-relative linearization, component recursion, similarity clustering, and SV genotyping.

Implements PanFlat v1 specification:
- §3.5 Subproblem state `S = (B_id, [lo, hi), M)`, progress metric `(|M|, hi - lo)`,
       local re-anchoring (Rule a), termination & candidate block extraction (`F14`, `F15`, `T4`, `T9`, `T11`, `T16`, `T18`, `T19`)
- §3.6.1–§3.6.5 Parallel branch groups `G(a,b)`, exact step deduplication, Union-Find component
       partitioning, secondary backbone selection (`support` / `longest`), component recursion (Rule b),
       4-criterion similarity clustering (`edlib` <=1 kb / `mappy` >1 kb), and unified right-anchor
       output layout (`F16`, `F17`, `F18`, `T1`–`T5`, `T12`, `T18`, `T20`, `C14`)
- §3.6.6 Universal repeat-guard integration prior to cycle folding and component recursion (`F19`, `C1`, `C16`–`C20`)
- §3.7 & §5.1–§5.5 Top-level SV event calling (`INS`, `DEL`, `CNV`, `INV`, `INV_DISPLACED`,
       `INV_EDGE`, `INV_CONFLICT`, `DUP_TRA`, `LARGE_JUMP`, `CEN_ISLAND`), nested allele signature
       genotyping, piece coordinate assembly, and single-chromosome pipeline orchestration (`F20`, `T6`–`T13`, `T22`, `T24`)
"""

from __future__ import annotations

import bisect
import collections
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass, field, replace
import gc
import multiprocessing
import os
import sys
import threading
import time
from typing import AbstractSet, Any, Iterator, Literal, Mapping, Sequence

import mappy
import numpy as np

from panflat.progress import log_progress

from panflat.centromere import (
    delimit_centromere_regions,
    partition_fragments_by_centromere,
    process_centromere_regions,
)
from panflat.gfa_io import (
    revcomp,
    spell_steps,
    validate_build_config,
)
from panflat.models import (
    BuildConfig,
    CentromereRegion,
    CentromereRow,
    ChromosomeBuildResult,
    CNVCall,
    CompactLinkSet,
    CompactNodeStore,
    CompactRankSet,
    CompactRefNodeSet,
    CompactVisitCounts,
    DeviationSegment,
    GFAGraph,
    InsertionSlotBlock,
    LazyProjectedAnchorSequence,
    LazyProvenanceTuple,
    LazyStepOffsetSequence,
    LazyStepTuple,
    PathFragment,
    Piece,
    ReverseChainRecord,
    Step,
    StepOffsetRow,
    StepProvenance,
    SVEvent,
    UnresolvedRecord,
    is_compact_nid_str,
)
from panflat.normalizer import (
    _extract_full_period,
    _get_lazy_backbone_anchor_arrays,
    _step_boundary_contiguous,
    build_backbone_anchor_map,
    compute_raw_visit_counts,
    compute_weighted_lis,
    detect_canonical_cnv,
    extract_deviations,
    fold_tandem_cycles,
    normalize_inversions_top_level,
    orient_contig_top_level,
    project_steps_onto_backbone,
)
from panflat.repeat_guard import (
    EDLIB_MAX_LEN,
    REF_SUB_BLOCK_SIZE,
    REF_SUB_MIN_ALIGNED_RATIO,
    REF_SUB_MIN_IDENTITY,
    REPEAT_KMER_SIZE,
    REPEAT_MIN_SEQ_LEN,
    REPEAT_SINGLETON_THRESHOLD,
    REPEAT_VALID_POS_RATIO,
    REPEAT_WINDOW_SIZE,
    REPEAT_WINDOW_STEP,
    _check_similarity_merge_with_span,
    _repeat_mask_intervals_cached,
    check_similarity_merge,
    complement_intervals,
    merge_contiguous_provenance,
    repeat_mask_intervals,
    slice_candidate_by_intervals,
    subtract_reference_intervals_mappy,
)


# ============================================================================
# §3.5 Subproblem State & Progress Metric
# ============================================================================

ProvenanceIntervalKey = tuple[str, int, int, Literal["+", "-"]]
BackboneId = tuple[ProvenanceIntervalKey, ...]
MemberId = tuple[ProvenanceIntervalKey, ...]
StepSliceKey = tuple[tuple[str, Literal["+", "-"], int, int | None, bool], ...]


def _is_full_step(step: Step) -> bool:
    """Return True iff `step` is a full, unsliced GFA node traversal."""
    return (not step.is_partial) and step.slice_start == 0 and step.slice_end is None


def _step_bp_len(step: Step, nodes: Mapping[str, str]) -> int:
    """Return effective base length of `step` (full or partial slice)."""
    full_len = (
        nodes.node_len(step.node_id)
        if isinstance(nodes, CompactNodeStore)
        else len(nodes[step.node_id])
    )
    end = step.slice_end if step.slice_end is not None else full_len
    return end - step.slice_start


def make_provenance_key(
    provenance: Sequence[StepProvenance],
) -> tuple[ProvenanceIntervalKey, ...]:
    """Build a canonical immutable provenance key from a sequence of `StepProvenance`."""
    if not provenance:
        return ()
    merged = merge_contiguous_provenance(provenance)
    return tuple(
        (prov.src_path, prov.src_start, prov.src_end, prov.src_strand)
        for prov, _ in merged
    )


def make_step_slice_key(steps: Sequence[Step]) -> StepSliceKey:
    """Build a directed step+slice key for exact deduplication (§3.6.2 Step 1)."""
    return tuple(
        (st.node_id, st.orient, st.slice_start, st.slice_end, st.is_partial)
        for st in steps
    )


def make_token_str(steps: Sequence[Step]) -> str:
    """Deterministic string representation of directed steps (including partial slice bounds)."""
    parts: list[str] = []
    for st in steps:
        if _is_full_step(st):
            parts.append(f"{st.node_id}{st.orient}")
        else:
            s_end = "" if st.slice_end is None else str(st.slice_end)
            parts.append(f"{st.node_id}{st.orient}[{st.slice_start}:{s_end}]")
    return ",".join(parts)


def _nested_key_sort_key(k: Any) -> Any:
    """Total-order sort key for nested signature keys (`int`, `str`, or nested `tuple`)."""
    if isinstance(k, int):
        return (0, k)
    if isinstance(k, str):
        return (1, k)
    if isinstance(k, tuple):
        return (2, tuple(_nested_key_sort_key(x) for x in k))
    return (3, repr(k))


@dataclass(frozen=True)
class SubproblemStateKey:
    """
    §3.5 Subproblem state `S = (B_id, [lo, hi), M)`.
    """

    b_id: BackboneId
    lo: int
    hi: int
    has_logical_start: bool
    has_logical_end: bool
    m_key: tuple[MemberId, ...]

    @property
    def metric(self) -> tuple[int, int]:
        return (len(self.m_key), self.hi - self.lo)


def can_transition(
    parent_key: SubproblemStateKey,
    child_key: SubproblemStateKey,
    ancestor_keys: AbstractSet[SubproblemStateKey],
) -> bool:
    """Return True iff `child_key` is not in `ancestor_keys` and its metric strictly decreases."""
    if child_key in ancestor_keys or child_key == parent_key:
        return False
    return child_key.metric < parent_key.metric


# ============================================================================
# §3.5 Local Re-anchoring & Candidate Block Extraction Data Models
# ============================================================================


@dataclass(frozen=True)
class CandidateBlock:
    """A raw contiguous non-reference candidate block extracted from a terminal deviation (§3.1, §3.5)."""

    candidate_id: str
    hap_id: str
    src_path: str
    left_anchor_rank: int | Literal["START"]
    right_anchor_rank: int | Literal["END"]
    source_order: int
    steps: tuple[Step, ...]
    provenance: tuple[StepProvenance, ...]
    seq: str
    piece_boundaries: tuple[int, ...] = ()
    repeat_screened: bool = False
    partial_content: bool = False
    repeat_residual: bool = False
    multi_hit: bool = False
    left_anchor_contiguous: bool = True


@dataclass(frozen=True)
class ExtractedCandidate:
    """A repeat-screened, cycle-folded candidate ready for §3.6 group solving."""

    candidate_id: str
    parent_id: str | None
    hap_id: str
    src_path: str
    left_anchor_rank: int | Literal["START"]
    right_anchor_rank: int | Literal["END"]
    source_order: int
    steps: tuple[Step, ...]
    provenance: tuple[StepProvenance, ...]
    seq: str
    piece_boundaries: tuple[int, ...] = ()
    repeat_screened: bool = True
    partial_content: bool = False
    repeat_residual: bool = False
    multi_hit: bool = False
    left_anchor_contiguous: bool = True
    contig_idx: int = -1


@dataclass
class GroupRepeatGuardResult:
    """Output of §3.6.6 repeat-guard and §3.4.3 cycle folding for a parallel branch group."""

    kept_candidates: list[ExtractedCandidate] = field(default_factory=list)
    unresolved_records: list[UnresolvedRecord] = field(default_factory=list)
    json_log_entries: list[dict[str, Any]] = field(default_factory=list)
    hap_had_masked_repeat: dict[str, bool] = field(default_factory=dict)
    hap_had_kept_candidate: dict[str, bool] = field(default_factory=dict)
    contig_had_masked_repeat: set[int] = field(default_factory=set)
    contig_had_kept_candidate: set[int] = field(default_factory=set)
    contig_had_subtracted_homolog: set[int] = field(default_factory=set)
    ref_masked_bp: int = 0
    short_residual_bp: int = 0
    has_partial_content: bool = False
    cand_dev_loc: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(slots=True)
class _CandidateDedupStats:
    """Per-candidate Module A reference subtraction accounting."""

    had_subtracted_homolog: bool = False
    ref_masked_bp: int = 0
    short_residual_bp: int = 0


def _make_temp_fragment(
    hap_id: str,
    src_path: str,
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    nodes: Mapping[str, str],
) -> PathFragment:
    """Build a lightweight `PathFragment` wrapper for `extract_deviations`."""
    sample, hap = hap_id.split("#", 1) if "#" in hap_id else (hap_id, "1")
    base_offset = provenance[0].src_start if provenance else 0
    if isinstance(steps, LazyStepTuple) and isinstance(
        provenance, LazyProvenanceTuple
    ):
        total_len = provenance.total_bp()
        out_steps: Any = steps
        out_prov: Any = provenance
    else:
        total_len = sum(st.step_len(nodes) for st in steps)
        out_steps = tuple(steps)
        out_prov = tuple(provenance)
    return PathFragment(
        sample=sample,
        hap=hap,
        hap_id=hap_id,
        seqid=src_path,
        src_path=src_path,
        base_offset=base_offset,
        steps=out_steps,
        provenance=out_prov,
        is_w_line=":" in src_path,
        total_len=total_len,
    )


def _shares_canonical_tandem_unit(
    ref_steps: Sequence[Step],
    alt_steps: Sequence[Step],
) -> bool:
    """Return True iff `ref_steps` and `alt_steps` share the same canonical period U with max(k) >= 2."""
    if not ref_steps or not alt_steps:
        return False
    if isinstance(ref_steps, LazyStepTuple) and isinstance(alt_steps, LazyStepTuple):
        r_codes = ref_steps._codes
        a_codes = alt_steps._codes
        if int(r_codes.shape[0]) < 2 and int(a_codes.shape[0]) < 2:
            return False
        if (
            int(r_codes.min()) != int(a_codes.min())
            or int(r_codes.max()) != int(a_codes.max())
        ):
            return False
        r_u, r_cnt = np.unique(r_codes, return_counts=True)
        a_u, a_cnt = np.unique(a_codes, return_counts=True)
        if not bool(np.array_equal(r_u, a_u)):
            return False
        if int(r_cnt.max()) < 2 and int(a_cnt.max()) < 2:
            return False
    p_ref = _extract_full_period(ref_steps)
    if p_ref is None:
        return False
    p_alt = _extract_full_period(alt_steps)
    if p_alt is None:
        return False
    u_ref, k_ref = p_ref
    u_alt, k_alt = p_alt
    return u_ref == u_alt and (k_ref >= 2 or k_alt >= 2)


def reanchor_deviation(
    dev: DeviationSegment,
    backbone_steps: Sequence[Step],
    nodes: Mapping[str, str],
    lo: int = 0,
    hi: int | None = None,
    parent_key: SubproblemStateKey | None = None,
    ancestor_keys: frozenset[SubproblemStateKey] = frozenset(),
    covered_ranks_out: Any = None,
) -> list[DeviationSegment]:
    """
    §3.5 Progress condition (a) (`F14`, `T16`): Local re-anchoring on `B(a, b)` via explicit
    LIFO iterative DFS (zero Python recursion depth limit, O(D) ancestor set maintenance).
    """
    n_backbone = len(backbone_steps)
    init_hi = n_backbone if hi is None else hi
    use_lazy_b = isinstance(backbone_steps, LazyStepTuple)
    is_compact_cov = isinstance(covered_ranks_out, CompactRankSet)
    active_ancestors: set[SubproblemStateKey] = set(ancestor_keys)
    refined: list[DeviationSegment] = []
    stack: list[tuple[Any, ...]] = [("WORK", dev, lo, init_hi, parent_key)]

    while stack:
        frame = stack.pop()
        if frame[0] == "POP_ANC":
            active_ancestors.discard(frame[1])
            continue

        _, cur_dev, cur_lo, cur_hi, cur_parent_key = frame
        if cur_dev.is_unanchored_start or cur_dev.is_unanchored_end:
            refined.append(cur_dev)
            continue

        lo_ab = (
            cur_lo
            if cur_dev.left_anchor_rank == "START"
            else int(cur_dev.left_anchor_rank) + 1
        )
        hi_ab = (
            cur_hi
            if cur_dev.right_anchor_rank == "END"
            else int(cur_dev.right_anchor_rank)
        )

        if hi_ab - lo_ab <= 0 or hi_ab - lo_ab >= cur_hi - cur_lo:
            refined.append(cur_dev)
            continue

        if not cur_dev.branch_steps:
            refined.append(cur_dev)
            continue

        # Fast check before allocating state keys: project onto B[lo_ab:hi_ab)
        sub_anchors = project_steps_onto_backbone(
            cur_dev.branch_steps,
            cur_dev.branch_provenance,
            backbone_steps,
            nodes,
            lo=lo_ab,
            hi=hi_ab,
        )
        if isinstance(sub_anchors, LazyProjectedAnchorSequence):
            has_fwd = bool(np.any(sub_anchors.rel_sign > 0))
        else:
            has_fwd = any(anc.rel_orient == "+" for anc in sub_anchors)
        if not has_fwd:
            refined.append(cur_dev)
            continue

        if cur_parent_key is None:
            cur_parent_key = SubproblemStateKey(
                b_id=(("BACKBONE", 0, cur_hi, "+"),),
                lo=cur_lo,
                hi=cur_hi,
                has_logical_start=(cur_lo > 0),
                has_logical_end=(cur_hi < n_backbone),
                m_key=(make_provenance_key(cur_dev.branch_provenance),),
            )

        child_key = SubproblemStateKey(
            b_id=cur_parent_key.b_id,
            lo=lo_ab,
            hi=hi_ab,
            has_logical_start=True,
            has_logical_end=True,
            m_key=cur_parent_key.m_key,
        )
        if not can_transition(cur_parent_key, child_key, active_ancestors):
            refined.append(cur_dev)
            continue

        sub_chain, _ = compute_weighted_lis(sub_anchors)
        if not sub_chain:
            refined.append(cur_dev)
            continue

        if covered_ranks_out is not None:
            if isinstance(sub_chain, LazyProjectedAnchorSequence) and is_compact_cov:
                covered_ranks_out.add_array(sub_chain.rank_in_b)
            else:
                for anc in sub_chain:
                    covered_ranks_out.add(anc.rank_in_b)

            first_anc = sub_chain[0]
            b_pref = (
                backbone_steps[lo_ab : first_anc.rank_in_b]
                if use_lazy_b
                else tuple(backbone_steps[lo_ab : first_anc.rank_in_b])
            )
            if cur_dev.branch_steps[0 : first_anc.branch_step_idx] == b_pref:
                if is_compact_cov:
                    covered_ranks_out.add_range(lo_ab, first_anc.rank_in_b + 1)
                else:
                    for r in range(lo_ab, first_anc.rank_in_b + 1):
                        covered_ranks_out.add(r)

            for k in range(len(sub_chain) - 1):
                a_l = sub_chain[k]
                a_r = sub_chain[k + 1]
                b_mid = (
                    backbone_steps[a_l.rank_in_b + 1 : a_r.rank_in_b]
                    if use_lazy_b
                    else tuple(backbone_steps[a_l.rank_in_b + 1 : a_r.rank_in_b])
                )
                if (
                    cur_dev.branch_steps[a_l.branch_step_idx + 1 : a_r.branch_step_idx]
                    == b_mid
                ):
                    if is_compact_cov:
                        covered_ranks_out.add_range(a_l.rank_in_b, a_r.rank_in_b + 1)
                    else:
                        for r in range(a_l.rank_in_b, a_r.rank_in_b + 1):
                            covered_ranks_out.add(r)

            last_anc = sub_chain[-1]
            b_suff = (
                backbone_steps[last_anc.rank_in_b + 1 : hi_ab]
                if use_lazy_b
                else tuple(backbone_steps[last_anc.rank_in_b + 1 : hi_ab])
            )
            if cur_dev.branch_steps[last_anc.branch_step_idx + 1 :] == b_suff:
                if is_compact_cov:
                    covered_ranks_out.add_range(last_anc.rank_in_b, hi_ab)
                else:
                    for r in range(last_anc.rank_in_b, hi_ab):
                        covered_ranks_out.add(r)

        temp_frag = _make_temp_fragment(
            cur_dev.hap_id,
            cur_dev.src_path,
            cur_dev.branch_steps,
            cur_dev.branch_provenance,
            nodes,
        )
        raw_sub_devs = extract_deviations(
            fragment=temp_frag,
            main_chain=sub_chain,
            backbone_steps=backbone_steps,
            is_top_level=False,
            lo=lo_ab,
            hi=hi_ab,
            all_projected_anchors=sub_anchors,
        )
        if not raw_sub_devs:
            continue

        mapped_devs: list[DeviationSegment] = []
        for sdev in raw_sub_devs:
            mapped_left = (
                cur_dev.left_anchor_rank
                if sdev.left_anchor_rank == "START"
                else sdev.left_anchor_rank
            )
            mapped_right = (
                cur_dev.right_anchor_rank
                if sdev.right_anchor_rank == "END"
                else sdev.right_anchor_rank
            )
            if (
                cur_dev.multi_hit_nodes == sdev.multi_hit_nodes
                or not sdev.multi_hit_nodes
            ):
                merged_mh = cur_dev.multi_hit_nodes
            elif not cur_dev.multi_hit_nodes:
                merged_mh = sdev.multi_hit_nodes
            else:
                merged_mh = tuple(
                    sorted(set(cur_dev.multi_hit_nodes) | set(sdev.multi_hit_nodes))
                )
            mapped_devs.append(
                DeviationSegment(
                    hap_id=cur_dev.hap_id,
                    src_path=cur_dev.src_path,
                    left_anchor_rank=mapped_left,
                    right_anchor_rank=mapped_right,
                    branch_start_idx=cur_dev.branch_start_idx + sdev.branch_start_idx,
                    branch_end_idx=cur_dev.branch_start_idx + sdev.branch_end_idx,
                    branch_steps=sdev.branch_steps,
                    branch_provenance=sdev.branch_provenance,
                    backbone_steps=sdev.backbone_steps,
                    multi_hit_nodes=merged_mh,
                    is_unanchored_start=False,
                    is_unanchored_end=False,
                    left_anchor_contiguous=(
                        cur_dev.left_anchor_contiguous
                        if sdev.left_anchor_rank == "START"
                        else sdev.left_anchor_contiguous
                    ),
                )
            )

        if cur_parent_key not in active_ancestors:
            active_ancestors.add(cur_parent_key)
            stack.append(("POP_ANC", cur_parent_key))
        for mdev in reversed(mapped_devs):
            stack.append(("WORK", mdev, lo_ab, hi_ab, child_key))

    return refined


def extract_candidate_blocks(
    dev: DeviationSegment,
    backbone_steps: Sequence[Step],
    nodes: Mapping[str, str],
    global_ref_nodes: AbstractSet[str],
    min_sv_len: int = 50,
    is_top_level: bool = True,
    lo: int = 0,
    hi: int | None = None,
    already_screened: bool = False,
    cand_id_prefix: str = "cand",
    base_source_order: int = 0,
    claimed_source_intervals: AbstractSet[tuple[str, int, int]] = frozenset(),
    precomputed_b_sets: tuple[AbstractSet[str], set[str], set[str]] | None = None,
    out_run_starts: list[int] | None = None,
) -> list[CandidateBlock]:
    """
    §3.1 & §3.5 (`F15`, `T4`, `T9`, `T11`, `T19`): Extract candidate blocks from a terminal deviation.
    """
    if is_top_level and (dev.is_unanchored_start or dev.is_unanchored_end):
        return []
    if not dev.branch_steps:
        return []

    if isinstance(dev.branch_provenance, LazyProvenanceTuple):
        if dev.branch_provenance.total_bp() < min_sv_len:
            return []

    eff_hi = len(backbone_steps) if hi is None else hi
    lo_ab = lo if dev.left_anchor_rank == "START" else int(dev.left_anchor_rank) + 1
    hi_ab = eff_hi if dev.right_anchor_rank == "END" else int(dev.right_anchor_rank)

    # Fast path when backbone_steps is the top-level reference LazyStepTuple:
    # every node in backbone_steps is already in global_ref_nodes!
    if (
        is_top_level
        and isinstance(backbone_steps, LazyStepTuple)
        and isinstance(global_ref_nodes, CompactRefNodeSet)
        and len(global_ref_nodes._extra) == 0
    ):
        b_interval_nodes: AbstractSet[str] = global_ref_nodes
        multicopy_local_nonref_nodes: set[str] = set()
        singlecopy_local_nonref_nodes: set[str] = set()

        if (
            isinstance(dev.branch_steps, LazyStepTuple)
            and isinstance(dev.branch_provenance, LazyProvenanceTuple)
            and isinstance(nodes, CompactNodeStore)
            and len(nodes._extra) == 0
            and not dev.branch_provenance.has_internal_discontinuity()
        ):
            br_codes = dev.branch_steps._codes
            br_nids = np.abs(br_codes)
            if int(br_nids.max()) <= nodes._max_nid:
                m_len = int(global_ref_nodes._mask.shape[0])
                ref_m = (br_nids < m_len) & global_ref_nodes._mask[
                    np.minimum(br_nids, max(0, m_len - 1))
                ]
                nonref_idx = np.flatnonzero(~ref_m)
                if int(nonref_idx.shape[0]) == 0:
                    return []
                gaps_nr = np.flatnonzero(np.diff(nonref_idx) > 1)
                n_nr_runs = int(gaps_nr.shape[0]) + 1
                nr_starts = np.empty(n_nr_runs, dtype=np.int32)
                nr_ends = np.empty(n_nr_runs, dtype=np.int32)
                nr_starts[0] = nonref_idx[0]
                if int(gaps_nr.shape[0]) > 0:
                    nr_starts[1:] = nonref_idx[gaps_nr + 1]
                    nr_ends[:-1] = nonref_idx[gaps_nr] + 1
                nr_ends[-1] = nonref_idx[-1] + 1

                node_lens_arr = nodes._node_lens
                kept_nr_bounds: list[tuple[int, int]] = []
                for s_i, e_i in zip(nr_starts.tolist(), nr_ends.tolist()):
                    if int(node_lens_arr[br_nids[s_i:e_i]].sum()) >= min_sv_len:
                        kept_nr_bounds.append((s_i, e_i))
                if not kept_nr_bounds:
                    return []

                if hi_ab <= lo_ab:
                    top_undecomposable_repeat = False
                else:
                    b_ab_nids = np.abs(backbone_steps._codes[lo_ab:hi_ab])
                    u_ab, cnt_ab = np.unique(b_ab_nids, return_counts=True)
                    b_ab_has_multicopy = bool(np.any(cnt_ab > 1))
                    if not b_ab_has_multicopy:
                        top_undecomposable_repeat = False
                    else:
                        b_ab_has_no_unique = not bool(np.any(cnt_ab == 1))
                        if b_ab_has_no_unique:
                            top_undecomposable_repeat = True
                        else:
                            top_undecomposable_repeat = bool(
                                np.any(np.isin(br_nids, u_ab[cnt_ab > 1]))
                            )

                candidates_fast: list[CandidateBlock] = []
                order_idx_fast = base_source_order
                for s_i, e_i in kept_nr_bounds:
                    sub_st_lazy = dev.branch_steps[s_i:e_i]
                    sub_pr_lazy = dev.branch_provenance[s_i:e_i]
                    if not already_screened and claimed_source_intervals:
                        merged_p = merge_contiguous_provenance(sub_pr_lazy)
                        if any(
                            (p.src_path, p.src_start, p.src_end)
                            in claimed_source_intervals
                            for p, _ in merged_p
                        ):
                            continue
                    seq = sub_st_lazy.spell(nodes)
                    if len(seq) < min_sv_len:
                        continue
                    cand_id = f"{cand_id_prefix}_{order_idx_fast + 1}"
                    candidates_fast.append(
                        CandidateBlock(
                            candidate_id=cand_id,
                            hap_id=dev.hap_id,
                            src_path=dev.src_path,
                            left_anchor_rank=dev.left_anchor_rank,
                            right_anchor_rank=dev.right_anchor_rank,
                            source_order=order_idx_fast,
                            steps=tuple(sub_st_lazy),
                            provenance=tuple(sub_pr_lazy),
                            seq=seq,
                            piece_boundaries=(),
                            repeat_screened=already_screened,
                            partial_content=False,
                            repeat_residual=top_undecomposable_repeat,
                            multi_hit=bool(dev.multi_hit_nodes),
                            left_anchor_contiguous=(
                                s_i == 0 and dev.left_anchor_contiguous
                            ),
                        )
                    )
                    if out_run_starts is not None:
                        out_run_starts.append(int(s_i))
                    order_idx_fast += 1
                return candidates_fast

        if hi_ab <= lo_ab:
            top_undecomposable_repeat = False
        else:
            b_ab_nids = np.abs(backbone_steps._codes[lo_ab:hi_ab])
            u_ab, cnt_ab = np.unique(b_ab_nids, return_counts=True)
            b_ab_has_multicopy = bool(np.any(cnt_ab > 1))
            if not b_ab_has_multicopy:
                top_undecomposable_repeat = False
            else:
                b_ab_has_no_unique = not bool(np.any(cnt_ab == 1))
                if b_ab_has_no_unique:
                    top_undecomposable_repeat = True
                else:
                    mc_nids_set = {str(x) for x in u_ab[cnt_ab > 1].tolist()}
                    branch_touches_b_ab_multicopy = any(
                        _is_full_step(st) and st.node_id in mc_nids_set
                        for st in dev.branch_steps
                    )
                    top_undecomposable_repeat = branch_touches_b_ab_multicopy
    else:
        if precomputed_b_sets is not None:
            b_interval_nodes, base_mc_nonref, base_sc_nonref = precomputed_b_sets
        else:
            b_interval_counts = collections.Counter(
                st.node_id for st in backbone_steps[lo:eff_hi] if _is_full_step(st)
            )
            b_interval_nodes = set(b_interval_counts.keys())
            base_mc_nonref = {
                nid
                for nid, cnt in b_interval_counts.items()
                if cnt > 1 and nid not in global_ref_nodes
            }
            base_sc_nonref = {
                nid
                for nid in b_interval_nodes
                if nid not in global_ref_nodes and nid not in base_mc_nonref
            }

        if hi_ab <= lo_ab:
            multicopy_local_nonref_nodes = base_mc_nonref
            singlecopy_local_nonref_nodes = base_sc_nonref
            top_undecomposable_repeat = False
        else:
            b_ab_counts = collections.Counter(
                st.node_id for st in backbone_steps[lo_ab:hi_ab] if _is_full_step(st)
            )
            b_ab_has_multicopy = any(cnt > 1 for cnt in b_ab_counts.values())
            if lo <= lo_ab and hi_ab <= eff_hi:
                multicopy_local_nonref_nodes = base_mc_nonref
                singlecopy_local_nonref_nodes = base_sc_nonref
            else:
                multicopy_local_nonref_nodes = base_mc_nonref | {
                    nid
                    for nid, cnt in b_ab_counts.items()
                    if cnt > 1 and nid not in global_ref_nodes
                }
                singlecopy_local_nonref_nodes = {
                    nid
                    for nid in b_interval_nodes
                    if nid not in global_ref_nodes
                    and nid not in multicopy_local_nonref_nodes
                }
            if not b_ab_has_multicopy:
                top_undecomposable_repeat = False
            else:
                b_ab_has_no_unique = not bool(
                    build_backbone_anchor_map(backbone_steps, lo=lo_ab, hi=hi_ab)
                )
                if b_ab_has_no_unique:
                    top_undecomposable_repeat = True
                else:
                    branch_touches_b_ab_multicopy = any(
                        _is_full_step(st) and b_ab_counts.get(st.node_id, 0) > 1
                        for st in dev.branch_steps
                    )
                    top_undecomposable_repeat = branch_touches_b_ab_multicopy

    def _is_separator(st: Step) -> bool:
        if st.node_id in global_ref_nodes:
            return True
        if singlecopy_local_nonref_nodes and _is_full_step(st) and (
            st.node_id in singlecopy_local_nonref_nodes
        ):
            return True
        return False

    runs: list[tuple[int, list[Step], list[StepProvenance]]] = []
    cur_start_idx = -1
    cur_steps: list[Step] = []
    cur_prov: list[StepProvenance] = []

    for step_idx, (st, pv) in enumerate(
        zip(dev.branch_steps, dev.branch_provenance)
    ):
        if _is_separator(st):
            if cur_steps:
                runs.append((cur_start_idx, cur_steps, cur_prov))
                cur_start_idx = -1
                cur_steps = []
                cur_prov = []
            continue

        if not already_screened and cur_prov:
            prev = cur_prov[-1]
            same_source = (
                prev.src_hap == pv.src_hap
                and prev.src_path == pv.src_path
                and prev.src_strand == pv.src_strand
            )
            contiguous = (
                (prev.src_end == pv.src_start)
                if prev.src_strand == "+"
                else (prev.src_start == pv.src_end)
            )
            if not (same_source and contiguous):
                runs.append((cur_start_idx, cur_steps, cur_prov))
                cur_start_idx = -1
                cur_steps = []
                cur_prov = []

        if not cur_steps:
            cur_start_idx = step_idx
        cur_steps.append(st)
        cur_prov.append(pv)

    if cur_steps:
        runs.append((cur_start_idx, cur_steps, cur_prov))

    candidates: list[CandidateBlock] = []
    order_idx = base_source_order
    for r_start_idx, r_steps, r_prov in runs:
        has_new_step = any(
            (not _is_full_step(st)) or (st.node_id not in b_interval_nodes)
            for st in r_steps
        )
        if not has_new_step:
            continue

        if not already_screened and claimed_source_intervals:
            merged_p = merge_contiguous_provenance(r_prov)
            if any(
                (p.src_path, p.src_start, p.src_end) in claimed_source_intervals
                for p, _ in merged_p
            ):
                continue

        seq = spell_steps(r_steps, nodes)
        if len(seq) < min_sv_len:
            continue

        has_embedded_local_multicopy = bool(multicopy_local_nonref_nodes) and any(
            _is_full_step(st) and st.node_id in multicopy_local_nonref_nodes
            for st in r_steps
        )
        is_rep_residual = has_embedded_local_multicopy or top_undecomposable_repeat

        cand_id = f"{cand_id_prefix}_{order_idx + 1}"
        candidates.append(
            CandidateBlock(
                candidate_id=cand_id,
                hap_id=dev.hap_id,
                src_path=dev.src_path,
                left_anchor_rank=dev.left_anchor_rank,
                right_anchor_rank=dev.right_anchor_rank,
                source_order=order_idx,
                steps=tuple(r_steps),
                provenance=tuple(r_prov),
                seq=seq,
                piece_boundaries=(),
                repeat_screened=already_screened,
                partial_content=False,
                repeat_residual=is_rep_residual,
                multi_hit=bool(dev.multi_hit_nodes),
                left_anchor_contiguous=(
                    r_start_idx == 0
                    and dev.left_anchor_contiguous
                    and r_steps[0].slice_start == 0
                ),
            )
        )
        if out_run_starts is not None:
            out_run_starts.append(int(r_start_idx))
        order_idx += 1

    return candidates


def _extract_contiguous_nonref_runs_for_unresolved(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    nodes: Mapping[str, str],
    global_ref_nodes: AbstractSet[str],
    min_sv_len: int,
    claimed_source_intervals: AbstractSet[tuple[str, int, int]] = frozenset(),
) -> list[tuple[StepProvenance, str]]:
    """Extract maximal contiguous global non-reference source intervals `>= min_sv_len` for `unresolved.tsv`."""
    results: list[tuple[StepProvenance, str]] = []
    if len(steps) == 0:
        return results

    if (
        isinstance(steps, LazyStepTuple)
        and isinstance(global_ref_nodes, CompactRefNodeSet)
        and len(global_ref_nodes._extra) == 0
    ):
        nids = np.abs(steps._codes)
        m_len = int(global_ref_nodes._mask.shape[0])
        ref_mask = (nids < m_len) & global_ref_nodes._mask[
            np.minimum(nids, max(0, m_len - 1))
        ]
        nonref_indices = np.flatnonzero(~ref_mask)
        if int(nonref_indices.shape[0]) == 0:
            return results
        gaps = np.flatnonzero(np.diff(nonref_indices) > 1)
        run_starts = np.empty(int(gaps.shape[0]) + 1, dtype=np.int32)
        run_ends = np.empty(int(gaps.shape[0]) + 1, dtype=np.int32)
        run_starts[0] = nonref_indices[0]
        if int(gaps.shape[0]) > 0:
            run_starts[1:] = nonref_indices[gaps + 1]
            run_ends[:-1] = nonref_indices[gaps] + 1
        run_ends[-1] = nonref_indices[-1] + 1
        for s_i, e_i in zip(run_starts.tolist(), run_ends.tolist()):
            sub_st = steps[s_i:e_i]
            sub_pr = provenance[s_i:e_i]
            if isinstance(sub_pr, LazyProvenanceTuple) and sub_pr.total_bp() < min_sv_len:
                continue
            seq = spell_steps(sub_st, nodes)
            for m_prov, m_seq in merge_contiguous_provenance(sub_pr, seq):
                if m_prov.length >= min_sv_len:
                    if (
                        m_prov.src_path,
                        m_prov.src_start,
                        m_prov.src_end,
                    ) not in claimed_source_intervals:
                        results.append((m_prov, m_seq))
        return results

    cur_steps: list[Step] = []
    cur_prov: list[StepProvenance] = []

    def _flush() -> None:
        if not cur_steps:
            return
        seq = spell_steps(cur_steps, nodes)
        for m_prov, m_seq in merge_contiguous_provenance(cur_prov, seq):
            if m_prov.length >= min_sv_len:
                if (
                    m_prov.src_path,
                    m_prov.src_start,
                    m_prov.src_end,
                ) not in claimed_source_intervals:
                    results.append((m_prov, m_seq))

    for st, pv in zip(steps, provenance):
        if st.node_id not in global_ref_nodes:
            cur_steps.append(st)
            cur_prov.append(pv)
        else:
            _flush()
            cur_steps = []
            cur_prov = []
    _flush()
    return results


# ============================================================================
# §3.6.6 Universal Repeat Guard & §3.4.3 Cycle Folding
# ============================================================================


def screen_and_fold_candidate(
    raw_cand: CandidateBlock,
    graph: GFAGraph,
    config: BuildConfig,
    region_id: str,
    next_source_order: int,
    unres_counter_ref: list[int],
    seen_exported_intervals: dict[tuple[str, int, int, str], str],
    node_visit_counts: Mapping[str, int] | None = None,
    ref_dedup_aligner: Any | None = None,
    ref_dedup_short_aligner: Any | None = None,
    ref_dedup_buf: Any | None = None,
    ref_dedup_short_buf: Any | None = None,
    dedup_stats_out: _CandidateDedupStats | None = None,
) -> tuple[
    list[ExtractedCandidate],
    list[UnresolvedRecord],
    list[dict[str, Any]],
    bool,
    int,
]:
    """
    §3.6.6, §3.4.3 & DEDUP_PLAN Module A.2 (`F19`, `F12`, `C1`, `C16`–`C20`, `T17`):
    1. When `config.dedup_mode in ("ref", "pan")`, subtract reference-homologous 50 bp blocks
       (`subtract_reference_intervals_mappy`) without stitching across masked intervals.
    2. Screen surviving `>= 500 bp` residuals with `repeat_mask_intervals` (when
       `config.repeat_guard == "on"`), recording `REPEAT_LIKE` in `unresolved_records`.
    3. Apply `fold_tandem_cycles` to each kept residual `>= config.min_sv_len`.

    Returns `(kept_candidates, unresolved_records, json_logs, had_masked_repeat, updated_next_source_order)`.
    """
    kept_out: list[ExtractedCandidate] = []
    unres_out: list[UnresolvedRecord] = []
    logs_out: list[dict[str, Any]] = []
    had_masked = False

    def _calc_rev_frac(steps: Sequence[Step], seq_len: int) -> float:
        if node_visit_counts is None or seq_len <= 0:
            return 0.0
        if isinstance(node_visit_counts, CompactVisitCounts):
            m_set = node_visit_counts._multi_set
            return (
                sum(
                    st.step_len(graph.nodes)
                    for st in steps
                    if is_compact_nid_str(st.node_id) and int(st.node_id) in m_set
                )
                / seq_len
            )
        return (
            sum(
                st.step_len(graph.nodes)
                for st in steps
                if node_visit_counts.get(st.node_id, 0) > 1
            )
            / seq_len
        )

    raw_len = len(raw_cand.seq)
    raw_contig_prov = merge_contiguous_provenance(raw_cand.provenance, raw_cand.seq)[0][0]
    rev_frac: float = _calc_rev_frac(raw_cand.steps, raw_len)

    def _make_log(
        cand_id: str,
        parent_id: str | None,
        prov: StepProvenance,
        length: int,
        revisited_fraction: float | None,
        masked_ivs: Sequence[tuple[int, int]] | None,
        win_ratios: list[float | None] | None,
        final_reason: str,
        rep_event_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "region_id": region_id,
            "hap": raw_cand.hap_id,
            "candidate_id": cand_id,
            "parent_id": parent_id,
            "src_path": prov.src_path,
            "src_start": prov.src_start,
            "src_end": prov.src_end,
            "src_strand": prov.src_strand,
            "length": length,
            "revisited_fraction": revisited_fraction,
            "masked_intervals": (
                [[int(s), int(e)] for s, e in masked_ivs]
                if masked_ivs is not None
                else None
            ),
            "window_singleton_ratios": win_ratios,
            "final_reason": final_reason,
            "representative_event_id": rep_event_id,
        }

    # Step 1 (Module A.2): Full-reference 50 bp block homology subtraction
    step1_items: list[
        tuple[
            str,
            str | None,
            tuple[Step, ...],
            tuple[StepProvenance, ...],
            str,
            bool,
            bool,
        ]
    ] = []

    if (
        not raw_cand.repeat_screened
        and config.dedup_mode in ("ref", "pan")
        and raw_len >= config.min_sv_len
        and graph.ref_seq
    ):
        if ref_dedup_aligner is None:
            ref_seq_up = graph.ref_seq.upper()
            ref_dedup_aligner = mappy.Aligner(
                seq=ref_seq_up, preset="asm20", best_n=50
            )
            if ref_dedup_short_aligner is None:
                ref_dedup_short_aligner = mappy.Aligner(
                    seq=ref_seq_up,
                    k=13,
                    w=5,
                    min_cnt=2,
                    min_chain_score=20,
                    best_n=50,
                )

        ref_masked_ivs, ref_unmasked_ivs, b_ref_bp = subtract_reference_intervals_mappy(
            query_seq=raw_cand.seq,
            ref_region_seq="",
            min_sv_len=config.min_sv_len,
            aligner=ref_dedup_aligner,
            buf=ref_dedup_buf,
            min_identity=config.dedup_id,
            short_aligner=ref_dedup_short_aligner,
            short_buf=ref_dedup_short_buf,
        )

        if ref_masked_ivs:
            if dedup_stats_out is not None:
                dedup_stats_out.had_subtracted_homolog = True
                dedup_stats_out.ref_masked_bp += b_ref_bp

            if not any(u_e - u_s >= config.min_sv_len for u_s, u_e in ref_unmasked_ivs):
                if dedup_stats_out is not None:
                    dedup_stats_out.short_residual_bp += sum(
                        u_e - u_s for u_s, u_e in ref_unmasked_ivs
                    )
                logs_out.append(
                    _make_log(
                        cand_id=raw_cand.candidate_id,
                        parent_id=None,
                        prov=raw_contig_prov,
                        length=raw_len,
                        revisited_fraction=rev_frac,
                        masked_ivs=ref_masked_ivs,
                        win_ratios=None,
                        final_reason="REFERENCE_LIKE",
                    )
                )
                return kept_out, unres_out, logs_out, had_masked, next_source_order

            logs_out.append(
                _make_log(
                    cand_id=raw_cand.candidate_id,
                    parent_id=None,
                    prov=raw_contig_prov,
                    length=raw_len,
                    revisited_fraction=rev_frac,
                    masked_ivs=ref_masked_ivs,
                    win_ratios=None,
                    final_reason="SPLIT_BY_REFERENCE_SUBTRACTION",
                )
            )
            ref_kept_slices, _ = slice_candidate_by_intervals(
                steps=raw_cand.steps,
                provenance=raw_cand.provenance,
                nodes=graph.nodes,
                keep_intervals=ref_unmasked_ivs,
                mask_intervals=(),
            )
            for r_idx, (r_start, _, r_steps, r_prov, r_seq) in enumerate(
                ref_kept_slices, start=1
            ):
                r_len = len(r_seq)
                r_id = f"{raw_cand.candidate_id}_r{r_idx}"
                if r_len < config.min_sv_len:
                    if dedup_stats_out is not None:
                        dedup_stats_out.short_residual_bp += r_len
                    r_contig = merge_contiguous_provenance(r_prov, r_seq)[0][0]
                    r_rev_frac = _calc_rev_frac(r_steps, r_len)
                    logs_out.append(
                        _make_log(
                            cand_id=r_id,
                            parent_id=raw_cand.candidate_id,
                            prov=r_contig,
                            length=r_len,
                            revisited_fraction=r_rev_frac,
                            masked_ivs=[],
                            win_ratios=None,
                            final_reason="SHORT_RESIDUAL",
                        )
                    )
                    continue
                step1_items.append(
                    (
                        r_id,
                        raw_cand.candidate_id,
                        r_steps,
                        r_prov,
                        r_seq,
                        True,
                        (r_start == 0 and raw_cand.left_anchor_contiguous),
                    )
                )
        else:
            step1_items.append(
                (
                    raw_cand.candidate_id,
                    None,
                    raw_cand.steps,
                    raw_cand.provenance,
                    raw_cand.seq,
                    raw_cand.partial_content,
                    raw_cand.left_anchor_contiguous,
                )
            )
    else:
        step1_items.append(
            (
                raw_cand.candidate_id,
                None,
                raw_cand.steps,
                raw_cand.provenance,
                raw_cand.seq,
                raw_cand.partial_content,
                raw_cand.left_anchor_contiguous,
            )
        )

    # Step 2 (15-mer repeat guard) & Step 3 (tandem cycle folding)
    for (
        r_id,
        r_parent,
        r_steps,
        r_prov,
        r_seq,
        r_partial,
        r_left_contig,
    ) in step1_items:
        r_len = len(r_seq)
        r_contig_prov = (
            raw_contig_prov
            if r_parent is None
            else merge_contiguous_provenance(r_prov, r_seq)[0][0]
        )
        r_rev_frac = rev_frac if r_parent is None else _calc_rev_frac(r_steps, r_len)

        if (
            not raw_cand.repeat_screened
            and config.repeat_guard == "on"
            and r_len >= REPEAT_MIN_SEQ_LEN
        ):
            masked_intervals, window_diags = repeat_mask_intervals(r_seq)
            win_ratios: list[float | None] | None = (
                [d["singleton_ratio"] for d in window_diags] if window_diags else None
            )
            unmasked_intervals = complement_intervals(r_len, masked_intervals)
            kept_slices, masked_slices = slice_candidate_by_intervals(
                steps=r_steps,
                provenance=r_prov,
                nodes=graph.nodes,
                keep_intervals=unmasked_intervals,
                mask_intervals=masked_intervals,
            )

            if masked_slices:
                had_masked = True
                for _, _, m_prov, m_seq in masked_slices:
                    m_len = len(m_seq)
                    if m_len >= config.min_sv_len:
                        m_contig = merge_contiguous_provenance(m_prov, m_seq)[0][0]
                        rec_id = f"{graph.ref_chrom}_UNRES_{unres_counter_ref[0]}"
                        unres_counter_ref[0] += 1
                        if config.dump_repeat_sequences:
                            key = (
                                m_contig.src_path,
                                m_contig.src_start,
                                m_contig.src_end,
                                m_contig.src_strand,
                            )
                            exp_id = seen_exported_intervals.setdefault(key, rec_id)
                        else:
                            exp_id = "."
                        unres_out.append(
                            UnresolvedRecord(
                                record_id=rec_id,
                                region_id=region_id,
                                event_id=".",
                                src_path=m_contig.src_path,
                                src_start=m_contig.src_start,
                                src_end=m_contig.src_end,
                                src_strand=m_contig.src_strand,
                                length=m_contig.length,
                                reason="REPEAT_LIKE",
                                seq=m_seq if config.dump_repeat_sequences else "",
                                export_to_fasta=config.dump_repeat_sequences,
                                exported_fasta_id=exp_id,
                            )
                        )

            if not kept_slices:
                logs_out.append(
                    _make_log(
                        cand_id=r_id,
                        parent_id=r_parent,
                        prov=r_contig_prov,
                        length=r_len,
                        revisited_fraction=r_rev_frac,
                        masked_ivs=masked_intervals,
                        win_ratios=win_ratios,
                        final_reason="REPEAT_LIKE",
                    )
                )
                continue

            if masked_intervals:
                logs_out.append(
                    _make_log(
                        cand_id=r_id,
                        parent_id=r_parent,
                        prov=r_contig_prov,
                        length=r_len,
                        revisited_fraction=r_rev_frac,
                        masked_ivs=masked_intervals,
                        win_ratios=win_ratios,
                        final_reason="SPLIT_BY_REPEAT_MASK",
                    )
                )

            for c_idx, (u_start, _, k_steps, k_prov, k_seq) in enumerate(
                kept_slices, start=1
            ):
                k_len = len(k_seq)
                c_id = r_id if not masked_intervals else f"{r_id}_c{c_idx}"
                c_parent = r_parent if not masked_intervals else r_id
                k_contig = merge_contiguous_provenance(k_prov, k_seq)[0][0]
                k_rev_frac = _calc_rev_frac(k_steps, k_len)

                if k_len < config.min_sv_len:
                    logs_out.append(
                        _make_log(
                            cand_id=c_id,
                            parent_id=c_parent,
                            prov=k_contig,
                            length=k_len,
                            revisited_fraction=k_rev_frac,
                            masked_ivs=[],
                            win_ratios=None,
                            final_reason="SHORT_RESIDUAL",
                        )
                    )
                    continue

                fold_res = fold_tandem_cycles(
                    k_steps, k_prov, graph.global_ref_nodes, config.max_copy
                )
                folded_seq = spell_steps(fold_res.folded_steps, graph.nodes)
                if len(folded_seq) < config.min_sv_len:
                    continue

                kept_out.append(
                    ExtractedCandidate(
                        candidate_id=c_id,
                        parent_id=c_parent,
                        hap_id=raw_cand.hap_id,
                        src_path=raw_cand.src_path,
                        left_anchor_rank=raw_cand.left_anchor_rank,
                        right_anchor_rank=raw_cand.right_anchor_rank,
                        source_order=next_source_order,
                        steps=fold_res.folded_steps,
                        provenance=fold_res.folded_provenance,
                        seq=folded_seq,
                        piece_boundaries=fold_res.piece_boundaries,
                        repeat_screened=True,
                        partial_content=(r_partial or bool(masked_intervals)),
                        repeat_residual=raw_cand.repeat_residual,
                        multi_hit=raw_cand.multi_hit,
                        left_anchor_contiguous=(u_start == 0 and r_left_contig),
                    )
                )
                logs_out.append(
                    _make_log(
                        cand_id=c_id,
                        parent_id=c_parent,
                        prov=k_contig,
                        length=len(folded_seq),
                        revisited_fraction=k_rev_frac,
                        masked_ivs=[] if masked_intervals else masked_intervals,
                        win_ratios=None if masked_intervals else win_ratios,
                        final_reason="ACCEPTED_CANDIDATE",
                    )
                )
                next_source_order += 1
            continue

        # Repeat-guard off or residual < 500 bp (or already screened)
        if r_len < config.min_sv_len:
            continue

        fold_res = fold_tandem_cycles(
            r_steps, r_prov, graph.global_ref_nodes, config.max_copy
        )
        folded_seq = spell_steps(fold_res.folded_steps, graph.nodes)
        if len(folded_seq) < config.min_sv_len:
            continue

        kept_out.append(
            ExtractedCandidate(
                candidate_id=r_id,
                parent_id=r_parent,
                hap_id=raw_cand.hap_id,
                src_path=raw_cand.src_path,
                left_anchor_rank=raw_cand.left_anchor_rank,
                right_anchor_rank=raw_cand.right_anchor_rank,
                source_order=next_source_order,
                steps=fold_res.folded_steps,
                provenance=fold_res.folded_provenance,
                seq=folded_seq,
                piece_boundaries=fold_res.piece_boundaries,
                repeat_screened=True,
                partial_content=r_partial,
                repeat_residual=raw_cand.repeat_residual,
                multi_hit=raw_cand.multi_hit,
                left_anchor_contiguous=r_left_contig,
            )
        )
        logs_out.append(
            _make_log(
                cand_id=r_id,
                parent_id=r_parent,
                prov=r_contig_prov,
                length=len(folded_seq),
                revisited_fraction=r_rev_frac,
                masked_ivs=None,
                win_ratios=None,
                final_reason="ACCEPTED_CANDIDATE",
            )
        )
        next_source_order += 1

    return kept_out, unres_out, logs_out, had_masked, next_source_order


# ============================================================================
# §3.6.1–§3.6.5 Parallel Branch Group Solver, Component Recursion & Clustering
# ============================================================================


@dataclass
class DedupCandidate:
    """Unique directed step+slice path within a parallel branch group (§3.6.2 Step 1)."""

    step_key: StepSliceKey
    steps: tuple[Step, ...]
    seq: str
    pieces: tuple[Piece, ...]
    support_haps: frozenset[str]
    occurrences: tuple[ExtractedCandidate, ...]
    exemplar: ExtractedCandidate
    token_str: str
    min_hap: str


@dataclass
class ComponentResult:
    """Result of solving a single-member or multi-member Union-Find component (§3.6.2 Step 3)."""

    is_synthetic: bool
    seq: str
    length_bp_no_n: int
    pieces: tuple[Piece, ...]
    support_haps: set[str]
    token_str: str
    min_hap: str
    exemplar: ExtractedCandidate
    member_occurrences: list[ExtractedCandidate]
    nested_choices_by_cand_id: dict[str, tuple[tuple[Any, tuple[int, ...]], ...]]
    reconstructed_len_by_cand_id: dict[str, int]
    piece_cand_ids: tuple[frozenset[str], ...]
    partial_content: bool = False
    repeat_residual: bool = False
    multi_hit: bool = False
    cand_to_flags: dict[str, tuple[bool, bool, bool]] = field(default_factory=dict)


@dataclass
class ClusterRepresentative:
    """A final cluster representative `R_k` (`k >= 1`) in `G(a, b)` (§3.6.2 Step 4, §3.6.4)."""

    rep_idx: int
    is_synthetic: bool
    seq: str
    length_bp_no_n: int
    pieces: tuple[Piece, ...]
    support_haps: set[str]
    token_str: str
    min_hap: str
    exemplar: ExtractedCandidate
    member_occurrences: list[ExtractedCandidate]
    nested_choices_by_cand_id: dict[str, tuple[tuple[Any, tuple[int, ...]], ...]]
    reconstructed_len_by_cand_id: dict[str, int]
    piece_cand_ids: tuple[frozenset[str], ...]
    partial_content: bool = False
    repeat_residual: bool = False
    multi_hit: bool = False
    cand_to_flags: dict[str, tuple[bool, bool, bool]] = field(default_factory=dict)
    aligner: Any | None = None
    support_hap_count: int = 0


@dataclass
class ParallelGroupSolution:
    """Complete solution of a parallel branch group `G(a, b)`."""

    left_anchor_rank: int | Literal["START"]
    right_anchor_rank: int | Literal["END"]
    has_nonempty_backbone: bool
    backbone_seq: str
    non_backbone_reps: tuple[ClusterRepresentative, ...]
    cand_to_rep_idx: dict[str, int]
    cand_to_nested_sig: dict[str, tuple[tuple[Any, tuple[int, ...]], ...]]
    cand_to_reconstructed_len: dict[str, int]
    cand_to_flags: dict[str, tuple[bool, bool, bool]] = field(default_factory=dict)
    rep0_cand_spans: dict[str, tuple[int, int, int, int]] = field(
        default_factory=dict
    )


def steps_to_pieces(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    piece_boundaries: Sequence[int],
    nodes: Mapping[str, str],
    members: Sequence[str],
    first_join_before: Literal["source_contiguous", "spacer", "synthetic"] = (
        "source_contiguous"
    ),
) -> tuple[Piece, ...]:
    """
    Convert a sequence of `steps` and `provenance` into `Piece`s, splitting at cycle-folding
    `piece_boundaries` (with `join_before = "synthetic"`) and at any source discontinuities.
    """
    if not steps:
        return ()

    cut_set = set(piece_boundaries)
    chunks: list[tuple[list[Step], list[StepProvenance]]] = []
    cur_s: list[Step] = []
    cur_p: list[StepProvenance] = []
    for idx, (st, pv) in enumerate(zip(steps, provenance)):
        if idx in cut_set and cur_s:
            chunks.append((cur_s, cur_p))
            cur_s = []
            cur_p = []
        cur_s.append(st)
        cur_p.append(pv)
    if cur_s:
        chunks.append((cur_s, cur_p))

    members_tuple = tuple(sorted(set(members)))
    pieces: list[Piece] = []
    for chunk_s, chunk_p in chunks:
        chunk_seq = spell_steps(chunk_s, nodes)
        merged = merge_contiguous_provenance(chunk_p, chunk_seq)
        for m_prov, m_seq in merged:
            join_type: Literal["source_contiguous", "spacer", "synthetic"] = (
                first_join_before if len(pieces) == 0 else "synthetic"
            )
            pieces.append(
                Piece(
                    src_hap=m_prov.src_hap,
                    src_path=m_prov.src_path,
                    src_start=m_prov.src_start,
                    src_end=m_prov.src_end,
                    src_strand=m_prov.src_strand,
                    seq=m_seq,
                    join_before=join_type,
                    members=members_tuple,
                )
            )
    return tuple(pieces)


def _merge_adjacent_source_contiguous_pieces(
    pieces: Sequence[Piece],
    piece_cands: Sequence[frozenset[str]],
    require_same_cands: bool = False,
) -> tuple[tuple[Piece, ...], tuple[frozenset[str], ...]]:
    """Merge adjacent `Piece`s when `curr.join_before == 'source_contiguous'` and source intervals touch."""
    if not pieces:
        return (), ()
    out_p: list[Piece] = [pieces[0]]
    out_c: list[frozenset[str]] = [piece_cands[0]]

    for p, cset in zip(pieces[1:], piece_cands[1:]):
        prev = out_p[-1]
        if (
            p.join_before == "source_contiguous"
            and prev.src_hap == p.src_hap
            and prev.src_path == p.src_path
            and prev.src_strand == p.src_strand
            and prev.members == p.members
            and (not require_same_cands or out_c[-1] == cset)
        ):
            if prev.src_strand == "+" and prev.src_end == p.src_start:
                out_p[-1] = replace(
                    prev,
                    src_end=p.src_end,
                    seq=prev.seq + p.seq,
                )
                out_c[-1] = out_c[-1] | cset
                continue
            if prev.src_strand == "-" and prev.src_start == p.src_end:
                out_p[-1] = replace(
                    prev,
                    src_start=p.src_start,
                    seq=prev.seq + p.seq,
                )
                out_c[-1] = out_c[-1] | cset
                continue
        out_p.append(p)
        out_c.append(cset)

    return tuple(out_p), tuple(out_c)


def _solve_multi_member_component(
    comp: Sequence[DedupCandidate],
    graph: GFAGraph,
    config: BuildConfig,
    parent_state_key: SubproblemStateKey | None,
    ancestor_keys: frozenset[SubproblemStateKey],
) -> list[ComponentResult]:
    """
    §3.6.2 Step 3: Select secondary backbone `B_sec` (`support` or `longest`), verify progress
    metric `(|M_child|, len(B_sec)) < parent_state_key.metric`, recursively solve child groups
    along `B_sec`, and assemble the resulting synthetic sequence and pieces.
    """
    node_haps: dict[str, set[str]] = collections.defaultdict(set)
    for d in comp:
        full_nids = {st.node_id for st in d.steps if _is_full_step(st)}
        for nid in full_nids:
            node_haps[nid].update(d.support_haps)

    def _sec_backbone_key(
        d: DedupCandidate,
    ) -> tuple[int, int, str, str, str, int, str]:
        if config.sec_backbone == "support":
            full_nids = {st.node_id for st in d.steps if _is_full_step(st)}
            full_score = sum(
                len(graph.nodes[nid]) * len(node_haps[nid]) for nid in full_nids
            )
            partial_score = sum(
                st.step_len(graph.nodes) * len(d.support_haps)
                for st in d.steps
                if not _is_full_step(st)
            )
            score = full_score + partial_score
        else:
            score = len(d.seq)
        p0 = d.exemplar.provenance[0]
        return (
            -score,
            -len(d.seq),
            d.token_str,
            d.min_hap,
            p0.src_path,
            p0.src_start,
            p0.src_strand,
        )

    b_sec = min(comp, key=_sec_backbone_key)
    m_child = [d for d in comp if d is not b_sec]

    b_sec_prov_key = make_provenance_key(b_sec.exemplar.provenance)
    m_child_keys = tuple(
        sorted({make_provenance_key(d.exemplar.provenance) for d in m_child})
    )
    child_state_key = SubproblemStateKey(
        b_id=b_sec_prov_key,
        lo=0,
        hi=len(b_sec.steps),
        has_logical_start=True,
        has_logical_end=True,
        m_key=m_child_keys,
    )

    if parent_state_key is not None and not can_transition(
        parent_state_key, child_state_key, ancestor_keys
    ):
        fallback: list[ComponentResult] = []
        for d in comp:
            all_occ_ids = frozenset(occ.candidate_id for occ in d.occurrences)
            fallback.append(
                ComponentResult(
                    is_synthetic=False,
                    seq=d.seq,
                    length_bp_no_n=len(d.seq),
                    pieces=d.pieces,
                    support_haps=set(d.support_haps),
                    token_str=d.token_str,
                    min_hap=d.min_hap,
                    exemplar=d.exemplar,
                    member_occurrences=list(d.occurrences),
                    nested_choices_by_cand_id={
                        occ.candidate_id: () for occ in d.occurrences
                    },
                    reconstructed_len_by_cand_id={
                        occ.candidate_id: len(d.seq) for occ in d.occurrences
                    },
                    piece_cand_ids=tuple(all_occ_ids for _ in d.pieces),
                    partial_content=any(occ.partial_content for occ in d.occurrences),
                    repeat_residual=any(occ.repeat_residual for occ in d.occurrences),
                    multi_hit=any(occ.multi_hit for occ in d.occurrences),
                    cand_to_flags={
                        occ.candidate_id: (
                            occ.partial_content,
                            occ.repeat_residual,
                            occ.multi_hit,
                        )
                        for occ in d.occurrences
                    },
                )
            )
        return fallback

    next_ancestors = (
        ancestor_keys | {parent_state_key} if parent_state_key is not None else ancestor_keys
    )

    b_steps = b_sec.steps
    b_prov = b_sec.exemplar.provenance
    n_b = len(b_steps)
    track_exact_b_sec = config.dedup_mode != "off"
    sec_b_counts = collections.Counter(
        st.node_id for st in b_steps if _is_full_step(st)
    )
    sec_b_interval_nodes = set(sec_b_counts.keys())
    sec_mc_nonref = {
        nid
        for nid, cnt in sec_b_counts.items()
        if cnt > 1 and nid not in graph.global_ref_nodes
    }
    sec_sc_nonref = {
        nid
        for nid in sec_b_interval_nodes
        if nid not in graph.global_ref_nodes and nid not in sec_mc_nonref
    }
    sec_b_sets: tuple[AbstractSet[str], set[str], set[str]] = (
        sec_b_interval_nodes,
        sec_mc_nonref,
        sec_sc_nonref,
    )

    child_group_cands: dict[
        tuple[int | Literal["START"], int | Literal["END"]], list[ExtractedCandidate]
    ] = collections.defaultdict(list)
    child_cid_to_parent_cids: dict[str, set[str]] = collections.defaultdict(set)
    member_cands_at_group: dict[
        tuple[int, tuple[int | Literal["START"], int | Literal["END"]]],
        list[ExtractedCandidate],
    ] = collections.defaultdict(list)
    member_covered_anchors: dict[int, set[int | Literal["START", "END"]]] = {}
    member_local_flags: dict[int, tuple[bool, bool]] = {}
    extra_repeat_residual = any(
        occ.repeat_residual for d in comp for occ in d.occurrences
    )
    extra_multi_hit = any(occ.multi_hit for d in comp for occ in d.occurrences)
    extra_partial_content = any(
        occ.partial_content for d in comp for occ in d.occurrences
    )

    next_child_order = 0
    for m_idx, d in enumerate(m_child):
        d_ex = d.exemplar
        anchors = project_steps_onto_backbone(
            d.steps, d_ex.provenance, b_steps, graph.nodes, lo=0, hi=n_b
        )
        main_chain, d_mh = compute_weighted_lis(anchors)
        m_rep_res = False
        m_mh = bool(d_mh)
        if d_mh:
            extra_multi_hit = True

        cov: set[int | Literal["START", "END"]] = {"START", "END"}
        for anc in main_chain:
            cov.add(anc.rank_in_b)

        if not main_chain:
            if d.steps == b_steps:
                for r in range(n_b):
                    cov.add(r)
        else:
            first_a = main_chain[0]
            if d.steps[0 : first_a.branch_step_idx] == b_steps[0 : first_a.rank_in_b]:
                for r in range(0, first_a.rank_in_b + 1):
                    cov.add(r)
            for k in range(len(main_chain) - 1):
                al = main_chain[k]
                ar = main_chain[k + 1]
                if (
                    d.steps[al.branch_step_idx + 1 : ar.branch_step_idx]
                    == b_steps[al.rank_in_b + 1 : ar.rank_in_b]
                ):
                    for r in range(al.rank_in_b, ar.rank_in_b + 1):
                        cov.add(r)
            last_a = main_chain[-1]
            if (
                d.steps[last_a.branch_step_idx + 1 :]
                == b_steps[last_a.rank_in_b + 1 : n_b]
            ):
                for r in range(last_a.rank_in_b, n_b):
                    cov.add(r)

        d_frag = _make_temp_fragment(
            d_ex.hap_id, d_ex.src_path, d.steps, d_ex.provenance, graph.nodes
        )
        raw_devs = extract_deviations(
            fragment=d_frag,
            main_chain=main_chain,
            backbone_steps=b_steps,
            is_top_level=False,
            lo=0,
            hi=n_b,
            all_projected_anchors=anchors,
        )

        local_cov: set[int] = set()
        term_devs: list[DeviationSegment] = []
        for rdev in raw_devs:
            term_devs.extend(
                reanchor_deviation(
                    dev=rdev,
                    backbone_steps=b_steps,
                    nodes=graph.nodes,
                    lo=0,
                    hi=n_b,
                    parent_key=child_state_key,
                    ancestor_keys=next_ancestors,
                    covered_ranks_out=local_cov,
                )
            )
        cov.update(local_cov)
        member_covered_anchors[m_idx] = cov

        for tdev in term_devs:
            if tdev.multi_hit_nodes:
                extra_multi_hit = True
                m_mh = True
            raw_blocks = extract_candidate_blocks(
                dev=tdev,
                backbone_steps=b_steps,
                nodes=graph.nodes,
                global_ref_nodes=graph.global_ref_nodes,
                min_sv_len=config.min_sv_len,
                is_top_level=False,
                lo=0,
                hi=n_b,
                already_screened=True,
                cand_id_prefix=f"{d_ex.candidate_id}_sub",
                base_source_order=next_child_order,
                precomputed_b_sets=sec_b_sets,
            )
            next_child_order += len(raw_blocks)

            g_key = (tdev.left_anchor_rank, tdev.right_anchor_rank)
            for rb in raw_blocks:
                if rb.repeat_residual:
                    extra_repeat_residual = True
                    m_rep_res = True
                fold_res = fold_tandem_cycles(
                    rb.steps,
                    rb.provenance,
                    graph.global_ref_nodes,
                    config.max_copy,
                )
                f_seq = spell_steps(fold_res.folded_steps, graph.nodes)
                if len(f_seq) < config.min_sv_len:
                    continue
                for occ_idx, parent_occ in enumerate(d.occurrences):
                    ext_c = ExtractedCandidate(
                        candidate_id=f"{rb.candidate_id}_o{occ_idx}",
                        parent_id=parent_occ.candidate_id,
                        hap_id=parent_occ.hap_id,
                        src_path=parent_occ.src_path,
                        left_anchor_rank=rb.left_anchor_rank,
                        right_anchor_rank=rb.right_anchor_rank,
                        source_order=rb.source_order,
                        steps=fold_res.folded_steps,
                        provenance=fold_res.folded_provenance,
                        seq=f_seq,
                        piece_boundaries=fold_res.piece_boundaries,
                        repeat_screened=True,
                        partial_content=parent_occ.partial_content,
                        repeat_residual=rb.repeat_residual or parent_occ.repeat_residual,
                        multi_hit=rb.multi_hit or m_mh or parent_occ.multi_hit,
                        left_anchor_contiguous=(
                            (
                                rb.left_anchor_contiguous
                                and d.exemplar.left_anchor_contiguous
                            )
                            if rb.left_anchor_rank == "START"
                            else rb.left_anchor_contiguous
                        ),
                    )
                    child_group_cands[g_key].append(ext_c)
                    child_cid_to_parent_cids[ext_c.candidate_id].add(
                        parent_occ.candidate_id
                    )
                    if occ_idx == 0:
                        member_cands_at_group[(m_idx, g_key)].append(ext_c)
        member_local_flags[m_idx] = (m_rep_res, m_mh)

    def _child_group_sort_key(
        g_key: tuple[int | Literal["START"], int | Literal["END"]],
    ) -> tuple[int, int, int]:
        a_r, b_r = g_key
        a_pos = -1 if a_r == "START" else int(a_r)
        b_pos = n_b if b_r == "END" else int(b_r)
        return (b_pos, b_pos - a_pos, a_pos)

    sorted_child_keys = sorted(child_group_cands.keys(), key=_child_group_sort_key)
    active_child_solutions: list[
        tuple[
            tuple[int | Literal["START"], int | Literal["END"]],
            int,
            int,
            int,
            ParallelGroupSolution,
        ]
    ] = []

    old_rec_limit = sys.getrecursionlimit()
    needed_rec_limit = min(100_000, len(comp) * 4 + 1024)
    raised_rec_limit = needed_rec_limit > old_rec_limit
    if raised_rec_limit:
        sys.setrecursionlimit(needed_rec_limit)
    try:
        for g_key in sorted_child_keys:
            a_r, b_r = g_key
            lo_ab = 0 if a_r == "START" else int(a_r) + 1
            hi_ab = n_b if b_r == "END" else int(b_r)
            b_slice_steps = b_steps[lo_ab:hi_ab]
            sol = solve_parallel_group(
                left_anchor_rank=a_r,
                right_anchor_rank=b_r,
                backbone_slice_steps=b_slice_steps,
                candidates=child_group_cands[g_key],
                graph=graph,
                config=config,
                parent_state_key=child_state_key,
                ancestor_keys=next_ancestors,
            )
            if sol.non_backbone_reps:
                a_pos = -1 if a_r == "START" else int(a_r)
                b_pos = n_b if b_r == "END" else int(b_r)
                active_child_solutions.append((g_key, b_pos, b_pos - a_pos, a_pos, sol))
                for r in sol.non_backbone_reps:
                    if r.repeat_residual:
                        extra_repeat_residual = True
                    if r.multi_hit:
                        extra_multi_hit = True
                    if r.partial_content:
                        extra_partial_content = True
    finally:
        if raised_rec_limit:
            sys.setrecursionlimit(old_rec_limit)

    all_comp_occurrences: list[ExtractedCandidate] = []
    for d in comp:
        all_comp_occurrences.extend(d.occurrences)
    all_comp_cand_ids = frozenset(occ.candidate_id for occ in all_comp_occurrences)
    comp_support_haps = set().union(*(d.support_haps for d in comp))

    nested_choices_by_cand_id: dict[
        str, tuple[tuple[Any, tuple[int, ...]], ...]
    ] = {}
    reconstructed_len_by_cand_id: dict[str, int] = {}
    cand_flags: dict[str, tuple[bool, bool, bool]] = {}
    rep_used_by_parent_cands: dict[tuple[Any, int], set[str]] = collections.defaultdict(
        set
    )
    step_excluded_cands: list[set[str]] = (
        [set() for _ in range(n_b)] if track_exact_b_sec else []
    )
    b_step_lens: list[int] = (
        [st.step_len(graph.nodes) for st in b_steps] if track_exact_b_sec else []
    )

    for occ in b_sec.occurrences:
        choices_b: list[tuple[Any, tuple[int, ...]]] = [
            (g_key, (0,)) for g_key, _, _, _, _ in active_child_solutions
        ]
        nested_choices_by_cand_id[occ.candidate_id] = tuple(choices_b)
        reconstructed_len_by_cand_id[occ.candidate_id] = len(b_sec.seq)
        cand_flags[occ.candidate_id] = (
            occ.partial_content,
            occ.repeat_residual,
            occ.multi_hit,
        )

    for m_idx, d in enumerate(m_child):
        cov = member_covered_anchors.get(m_idx, {"START", "END"})
        d_choices: list[tuple[Any, tuple[int, ...]]] = []
        rec_len = len(b_sec.seq)
        child_non_rep0_bp = 0
        m_rep_res, m_mh = member_local_flags.get(m_idx, (False, False))
        m_part = False
        d_parent_cids = [p_occ.candidate_id for p_occ in d.occurrences]

        for g_key, _, _, _, sol in active_child_solutions:
            a_r, b_r = g_key
            lo_ab = 0 if a_r == "START" else int(a_r) + 1
            hi_ab = n_b if b_r == "END" else int(b_r)
            cands_here = member_cands_at_group.get((m_idx, g_key), [])
            if cands_here:
                rep_list: list[int] = []
                sub_nested_entries: list[tuple[Any, tuple[int, ...]]] = []
                rep_len_sum = 0
                for ch_c in sorted(cands_here, key=lambda x: x.source_order):
                    ch_part, ch_rep, ch_mh = sol.cand_to_flags.get(
                        ch_c.candidate_id,
                        (
                            ch_c.partial_content,
                            ch_c.repeat_residual,
                            ch_c.multi_hit,
                        ),
                    )
                    m_part = m_part or ch_part
                    m_rep_res = m_rep_res or ch_rep
                    m_mh = m_mh or ch_mh
                    r_idx = sol.cand_to_rep_idx.get(ch_c.candidate_id, 0)
                    rep_list.append(r_idx)
                    if r_idx > 0:
                        for p_cid in d_parent_cids:
                            rep_used_by_parent_cands[(g_key, r_idx)].add(p_cid)
                        ch_len = sol.cand_to_reconstructed_len.get(
                            ch_c.candidate_id,
                            sol.non_backbone_reps[r_idx - 1].length_bp_no_n,
                        )
                        rep_len_sum += ch_len
                        child_non_rep0_bp += ch_len
                    else:
                        rep_len_sum += len(sol.backbone_seq)
                    sub_sig = sol.cand_to_nested_sig.get(ch_c.candidate_id, ())
                    for sub_k, sub_v in sub_sig:
                        sub_nested_entries.append(((g_key, sub_k), sub_v))

                d_choices.append((g_key, tuple(rep_list)))
                d_choices.extend(sub_nested_entries)
                if rep_list != [0]:
                    rec_len += rep_len_sum - len(sol.backbone_seq)
                    if track_exact_b_sec and lo_ab < hi_ab and all(r > 0 for r in rep_list):
                        for st_i in range(lo_ab, hi_ab):
                            step_excluded_cands[st_i].update(d_parent_cids)
            elif a_r in cov and b_r in cov:
                d_choices.append((g_key, (0,)))
            else:
                d_choices.append((g_key, (-1,)))

        if track_exact_b_sec and active_child_solutions:
            d_rep_cid = d_parent_cids[0]
            retained_b_bp = sum(
                b_step_lens[st_i]
                for st_i in range(n_b)
                if d_rep_cid not in step_excluded_cands[st_i]
            )
            exact_rec_len = retained_b_bp + child_non_rep0_bp
        else:
            exact_rec_len = max(len(d.seq), rec_len)

        for p_occ in d.occurrences:
            nested_choices_by_cand_id[p_occ.candidate_id] = tuple(d_choices)
            reconstructed_len_by_cand_id[p_occ.candidate_id] = exact_rec_len
            cand_flags[p_occ.candidate_id] = (
                p_occ.partial_content or m_part,
                p_occ.repeat_residual or m_rep_res,
                p_occ.multi_hit or m_mh,
            )

    if not active_child_solutions:
        b_pieces = tuple(
            replace(p, members=tuple(sorted(comp_support_haps))) for p in b_sec.pieces
        )
        return [
            ComponentResult(
                is_synthetic=(config.dedup_mode == "off"),
                seq=b_sec.seq,
                length_bp_no_n=len(b_sec.seq),
                pieces=b_pieces,
                support_haps=comp_support_haps,
                token_str=b_sec.token_str,
                min_hap=min(comp_support_haps),
                exemplar=b_sec.exemplar,
                member_occurrences=all_comp_occurrences,
                nested_choices_by_cand_id=nested_choices_by_cand_id,
                reconstructed_len_by_cand_id=reconstructed_len_by_cand_id,
                piece_cand_ids=tuple(all_comp_cand_ids for _ in b_pieces),
                partial_content=extra_partial_content,
                repeat_residual=extra_repeat_residual,
                multi_hit=extra_multi_hit,
                cand_to_flags=cand_flags,
            )
        ]

    groups_by_slot: dict[
        int,
        list[
            tuple[
                tuple[int | Literal["START"], int | Literal["END"]],
                int,
                int,
                ParallelGroupSolution,
            ]
        ],
    ] = collections.defaultdict(list)
    for g_key, b_pos, g_span, a_pos, sol in active_child_solutions:
        groups_by_slot[b_pos].append((g_key, g_span, a_pos, sol))
    for slot_list in groups_by_slot.values():
        slot_list.sort(key=lambda item: (item[1], item[2]))

    def _provs_contiguous(
        p_prev: StepProvenance, p_next: StepProvenance
    ) -> bool:
        if (
            p_prev.src_hap != p_next.src_hap
            or p_prev.src_path != p_next.src_path
            or p_prev.src_strand != p_next.src_strand
        ):
            return False
        if p_prev.src_strand == "+":
            return p_prev.src_end == p_next.src_start
        return p_prev.src_start == p_next.src_end

    b_sec_cut_set = set(b_sec.exemplar.piece_boundaries)
    has_excluded_b_steps = track_exact_b_sec and any(step_excluded_cands)
    comp_members_sorted = tuple(sorted(comp_support_haps))
    cand_id_to_hap = {occ.candidate_id: occ.hap_id for occ in all_comp_occurrences}

    raw_out_pieces: list[Piece] = []
    raw_out_cands: list[frozenset[str]] = []
    cur_b_steps: list[Step] = []
    cur_b_prov: list[StepProvenance] = []
    cur_b_cands: frozenset[str] | None = None
    cur_b_first_join: Literal["source_contiguous", "spacer", "synthetic"] = (
        b_sec.pieces[0].join_before if b_sec.pieces else "synthetic"
    )

    def _flush_b_run() -> None:
        nonlocal cur_b_steps, cur_b_prov, cur_b_cands, cur_b_first_join
        if not cur_b_steps:
            return
        run_cands = cur_b_cands if cur_b_cands is not None else all_comp_cand_ids
        run_members = (
            comp_members_sorted
            if run_cands == all_comp_cand_ids
            else tuple(sorted({cand_id_to_hap[c] for c in run_cands}))
        )
        run_seq = spell_steps(cur_b_steps, graph.nodes)
        merged = merge_contiguous_provenance(cur_b_prov, run_seq)
        for idx_m, (m_prov, m_seq) in enumerate(merged):
            j_type: Literal["source_contiguous", "spacer", "synthetic"] = (
                cur_b_first_join if idx_m == 0 else "synthetic"
            )
            raw_out_pieces.append(
                Piece(
                    src_hap=m_prov.src_hap,
                    src_path=m_prov.src_path,
                    src_start=m_prov.src_start,
                    src_end=m_prov.src_end,
                    src_strand=m_prov.src_strand,
                    seq=m_seq,
                    join_before=j_type,
                    members=run_members,
                )
            )
            raw_out_cands.append(run_cands)
        cur_b_steps = []
        cur_b_prov = []
        cur_b_cands = None
        cur_b_first_join = "source_contiguous"

    for idx in range(n_b + 1):
        if idx in b_sec_cut_set and cur_b_steps:
            _flush_b_run()
            cur_b_first_join = "synthetic"

        slot_groups = groups_by_slot.get(idx, [])
        if slot_groups:
            _flush_b_run()
            for m_g, (g_key, _, _, sol) in enumerate(slot_groups):
                for j_r, rep in enumerate(sol.non_backbone_reps):
                    if m_g > 0 or j_r > 0 or sol.has_nonempty_backbone:
                        first_join: Literal[
                            "source_contiguous", "spacer", "synthetic"
                        ] = "spacer"
                    else:
                        first_join = (
                            "synthetic"
                            if len(raw_out_pieces) > 0
                            else (
                                rep.pieces[0].join_before
                                if rep.pieces
                                else "synthetic"
                            )
                        )
                    used_by = frozenset(
                        rep_used_by_parent_cands.get(
                            (g_key, rep.rep_idx), set(all_comp_cand_ids)
                        )
                    )
                    for p_i, rp in enumerate(rep.pieces):
                        translated_cands: set[str] = set()
                        if p_i < len(rep.piece_cand_ids):
                            for cc in rep.piece_cand_ids[p_i]:
                                p_set = child_cid_to_parent_cids.get(cc)
                                if p_set:
                                    translated_cands.update(p_set)
                        p_cands = (
                            frozenset(translated_cands) & used_by
                            if translated_cands
                            else used_by
                        )
                        if not p_cands:
                            p_cands = used_by
                        if p_i == 0:
                            raw_out_pieces.append(replace(rp, join_before=first_join))
                        else:
                            raw_out_pieces.append(rp)
                        raw_out_cands.append(p_cands or all_comp_cand_ids)
            if idx < n_b:
                cur_b_first_join = "synthetic"

        if idx < n_b:
            step_cands = (
                all_comp_cand_ids - step_excluded_cands[idx]
                if has_excluded_b_steps and step_excluded_cands[idx]
                else all_comp_cand_ids
            )
            if cur_b_steps and step_cands != cur_b_cands:
                prev_prov = cur_b_prov[-1]
                _flush_b_run()
                cur_b_first_join = (
                    "source_contiguous"
                    if _provs_contiguous(prev_prov, b_prov[idx])
                    else "synthetic"
                )
            if not cur_b_steps:
                cur_b_cands = step_cands
            cur_b_steps.append(b_steps[idx])
            cur_b_prov.append(b_prov[idx])

    _flush_b_run()

    merged_pieces, merged_cands = _merge_adjacent_source_contiguous_pieces(
        raw_out_pieces,
        raw_out_cands,
        require_same_cands=has_excluded_b_steps,
    )
    seq_parts: list[str] = []
    for p_i, p in enumerate(merged_pieces):
        if p_i > 0 and p.join_before == "spacer":
            seq_parts.append("N" * config.spacer_n)
        seq_parts.append(p.seq)
    full_seq = "".join(seq_parts)
    len_no_n = sum(len(p.seq) for p in merged_pieces)

    return [
        ComponentResult(
            is_synthetic=True,
            seq=full_seq,
            length_bp_no_n=len_no_n,
            pieces=merged_pieces,
            support_haps=comp_support_haps,
            token_str=b_sec.token_str,
            min_hap=min(comp_support_haps),
            exemplar=b_sec.exemplar,
            member_occurrences=all_comp_occurrences,
            nested_choices_by_cand_id=nested_choices_by_cand_id,
            reconstructed_len_by_cand_id=reconstructed_len_by_cand_id,
            piece_cand_ids=merged_cands,
            partial_content=extra_partial_content,
            repeat_residual=extra_repeat_residual,
            multi_hit=extra_multi_hit,
            cand_to_flags=cand_flags,
        )
    ]


def solve_parallel_group(
    left_anchor_rank: int | Literal["START"],
    right_anchor_rank: int | Literal["END"],
    backbone_slice_steps: Sequence[Step],
    candidates: Sequence[ExtractedCandidate],
    graph: GFAGraph,
    config: BuildConfig,
    parent_state_key: SubproblemStateKey | None = None,
    ancestor_keys: frozenset[SubproblemStateKey] = frozenset(),
) -> ParallelGroupSolution:
    """
    §3.6.1–§3.6.4 (`F16`, `F17`, `F18`, `T1`–`T5`, `T12`, `T18`, `T20`):
    Solve a parallel branch group `G(a, b)`:
    1. Exact deduplication of identical directed step+slice paths.
    2. Union-Find component partitioning on shared full non-reference node IDs.
    3. Multi-member component recursion via secondary backbone (`support` or `longest`).
    4. 4-criterion similarity clustering on real sequences (`is_synthetic == False`) and
       exact deduplication on synthetic sequences (`is_synthetic == True`).
    5. Representative ordering by support count DESC, length DESC, token string ASC.
    """
    rep0_seq = spell_steps(backbone_slice_steps, graph.nodes)
    has_rep0 = len(rep0_seq) > 0

    if not candidates:
        return ParallelGroupSolution(
            left_anchor_rank=left_anchor_rank,
            right_anchor_rank=right_anchor_rank,
            has_nonempty_backbone=has_rep0,
            backbone_seq=rep0_seq,
            non_backbone_reps=(),
            cand_to_rep_idx={},
            cand_to_nested_sig={},
            cand_to_reconstructed_len={},
            cand_to_flags={},
        )

    def _build_components_from_candidates(
        cands_subset: Sequence[ExtractedCandidate],
    ) -> list[ComponentResult]:
        # Step 1: Exact deduplication by directed step+slice key
        by_step_key: dict[StepSliceKey, list[ExtractedCandidate]] = (
            collections.defaultdict(list)
        )
        for c in cands_subset:
            by_step_key[make_step_slice_key(c.steps)].append(c)

        dedup_list: list[DedupCandidate] = []
        for skey, occs in by_step_key.items():
            occs_sorted = sorted(
                occs,
                key=lambda c: (
                    c.hap_id,
                    c.provenance[0].src_path,
                    c.provenance[0].src_start,
                    c.provenance[0].src_strand,
                    c.source_order,
                    c.candidate_id,
                ),
            )
            exemplar = occs_sorted[0]
            sup_haps = frozenset(c.hap_id for c in occs_sorted)
            min_hap = min(sup_haps)
            t_str = make_token_str(exemplar.steps)
            pieces = steps_to_pieces(
                steps=exemplar.steps,
                provenance=exemplar.provenance,
                piece_boundaries=exemplar.piece_boundaries,
                nodes=graph.nodes,
                members=tuple(sorted(sup_haps)),
                first_join_before=(
                    "source_contiguous"
                    if exemplar.left_anchor_contiguous
                    else "synthetic"
                ),
            )
            dedup_list.append(
                DedupCandidate(
                    step_key=skey,
                    steps=exemplar.steps,
                    seq=exemplar.seq,
                    pieces=pieces,
                    support_haps=sup_haps,
                    occurrences=tuple(occs_sorted),
                    exemplar=exemplar,
                    token_str=t_str,
                    min_hap=min_hap,
                )
            )

        dedup_list.sort(
            key=lambda d: (
                d.token_str,
                d.min_hap,
                d.exemplar.provenance[0].src_path,
                d.exemplar.provenance[0].src_start,
                d.exemplar.provenance[0].src_strand,
            )
        )

        # Step 2: Union-Find component partitioning on shared full non-reference node IDs
        n_dedup = len(dedup_list)
        uf_parent = list(range(n_dedup))

        def _find(x: int) -> int:
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                if ra < rb:
                    uf_parent[rb] = ra
                else:
                    uf_parent[ra] = rb

        first_by_nid: dict[str, int] = {}
        for idx_d, d in enumerate(dedup_list):
            for st in d.steps:
                if _is_full_step(st) and st.node_id not in graph.global_ref_nodes:
                    if st.node_id in first_by_nid:
                        _union(idx_d, first_by_nid[st.node_id])
                    else:
                        first_by_nid[st.node_id] = idx_d

        comp_map: dict[int, list[DedupCandidate]] = collections.defaultdict(list)
        for idx_d, d in enumerate(dedup_list):
            comp_map[_find(idx_d)].append(d)

        # Step 3: Solve each component (single-member -> real; multi-member -> recursive synthetic)
        built_results: list[ComponentResult] = []
        for root_idx in sorted(comp_map.keys()):
            comp = comp_map[root_idx]
            if len(comp) == 1:
                d = comp[0]
                occ_ids = frozenset(occ.candidate_id for occ in d.occurrences)
                built_results.append(
                    ComponentResult(
                        is_synthetic=False,
                        seq=d.seq,
                        length_bp_no_n=len(d.seq),
                        pieces=d.pieces,
                        support_haps=set(d.support_haps),
                        token_str=d.token_str,
                        min_hap=d.min_hap,
                        exemplar=d.exemplar,
                        member_occurrences=list(d.occurrences),
                        nested_choices_by_cand_id={
                            occ.candidate_id: () for occ in d.occurrences
                        },
                        reconstructed_len_by_cand_id={
                            occ.candidate_id: len(d.seq) for occ in d.occurrences
                        },
                        piece_cand_ids=tuple(occ_ids for _ in d.pieces),
                        partial_content=any(
                            occ.partial_content for occ in d.occurrences
                        ),
                        repeat_residual=any(
                            occ.repeat_residual for occ in d.occurrences
                        ),
                        multi_hit=any(occ.multi_hit for occ in d.occurrences),
                        cand_to_flags={
                            occ.candidate_id: (
                                occ.partial_content,
                                occ.repeat_residual,
                                occ.multi_hit,
                            )
                            for occ in d.occurrences
                        },
                    )
                )
            else:
                built_results.extend(
                    _solve_multi_member_component(
                        comp=comp,
                        graph=graph,
                        config=config,
                        parent_state_key=parent_state_key,
                        ancestor_keys=ancestor_keys,
                    )
                )
        return built_results

    def _real_comp_sort_key(c: ComponentResult) -> tuple[Any, ...]:
        return (
            -c.length_bp_no_n,
            c.token_str,
            c.min_hap,
            c.exemplar.provenance[0].src_path,
            c.exemplar.provenance[0].src_start,
            c.exemplar.provenance[0].src_strand,
        )

    def _synth_comp_sort_key(c: ComponentResult) -> tuple[Any, ...]:
        return (
            -len(c.support_haps),
            -c.length_bp_no_n,
            c.token_str,
            c.min_hap,
            c.exemplar.provenance[0].src_path,
            c.exemplar.provenance[0].src_start,
            c.exemplar.provenance[0].src_strand,
        )

    comp_results = _build_components_from_candidates(candidates)

    # Step 4: 4-criterion similarity clustering for real sequences & exact dedup for synthetic
    real_comps = [c for c in comp_results if not c.is_synthetic]
    synth_comps = [c for c in comp_results if c.is_synthetic]
    real_comps.sort(key=_real_comp_sort_key)
    synth_comps.sort(key=_synth_comp_sort_key)

    rep0_aligner: Any | None = None
    rep0_occurrences: list[ExtractedCandidate] = []
    rep0_cand_flags: dict[str, tuple[bool, bool, bool]] = {}
    rep0_cand_spans: dict[str, tuple[int, int, int, int]] = {}
    rep0_occ_span_cache: dict[str, tuple[bool, int, int, int]] = {}

    def _get_occ_rep0_span(occ_seq: str) -> tuple[bool, int, int, int]:
        nonlocal rep0_aligner
        cached = rep0_occ_span_cache.get(occ_seq)
        if cached is None:
            if (
                len(occ_seq) > EDLIB_MAX_LEN or len(rep0_seq) > EDLIB_MAX_LEN
            ) and rep0_aligner is None:
                rep0_aligner = mappy.Aligner(seq=rep0_seq, preset="asm20")
            cached = _check_similarity_merge_with_span(
                seq_s=occ_seq,
                seq_r=rep0_seq,
                min_sv_len=config.min_sv_len,
                cluster_id=config.cluster_id,
                cluster_win_id=config.cluster_win_id,
                aligner=rep0_aligner,
            )
            rep0_occ_span_cache[occ_seq] = cached
        return cached

    # Step 4a: Match against rep0 (splitting non-merging occurrences under dedup_mode != "off")
    remaining_real_comps: list[ComponentResult] = []
    remaining_synth_comps: list[ComponentResult] = []
    had_rep0_split = False

    if has_rep0:
        len_rep0 = len(rep0_seq)
        for rc in real_comps:
            if (
                len(rc.seq) > EDLIB_MAX_LEN or len_rep0 > EDLIB_MAX_LEN
            ) and rep0_aligner is None:
                rep0_aligner = mappy.Aligner(seq=rep0_seq, preset="asm20")
            merged_rep0, rep0_ts, rep0_te, rep0_del_bp = (
                _check_similarity_merge_with_span(
                    seq_s=rc.seq,
                    seq_r=rep0_seq,
                    min_sv_len=config.min_sv_len,
                    cluster_id=config.cluster_id,
                    cluster_win_id=config.cluster_win_id,
                    aligner=rep0_aligner,
                )
            )
            if not merged_rep0:
                remaining_real_comps.append(rc)
                continue

            span_info = (len(rc.seq), rep0_ts, rep0_te, rep0_del_bp)
            rc_failing_occs: list[ExtractedCandidate] = []
            for occ in rc.member_occurrences:
                if occ.seq == rc.seq:
                    rep0_cand_spans[occ.candidate_id] = span_info
                    rep0_occurrences.append(occ)
                    if occ.candidate_id in rc.cand_to_flags:
                        rep0_cand_flags[occ.candidate_id] = rc.cand_to_flags[
                            occ.candidate_id
                        ]
                else:
                    merged_occ, o_ts, o_te, o_del = _get_occ_rep0_span(occ.seq)
                    if merged_occ:
                        rep0_cand_spans[occ.candidate_id] = (
                            len(occ.seq),
                            o_ts,
                            o_te,
                            o_del,
                        )
                        rep0_occurrences.append(occ)
                        if occ.candidate_id in rc.cand_to_flags:
                            rep0_cand_flags[occ.candidate_id] = rc.cand_to_flags[
                                occ.candidate_id
                            ]
                    elif config.dedup_mode == "off" or len(occ.seq) <= len(rc.seq):
                        rep0_occurrences.append(occ)
                        if occ.candidate_id in rc.cand_to_flags:
                            rep0_cand_flags[occ.candidate_id] = rc.cand_to_flags[
                                occ.candidate_id
                            ]
                    else:
                        rc_failing_occs.append(occ)
            if rc_failing_occs:
                had_rep0_split = True
                for sub_c in _build_components_from_candidates(rc_failing_occs):
                    if sub_c.is_synthetic:
                        remaining_synth_comps.append(sub_c)
                    else:
                        remaining_real_comps.append(sub_c)

        for sc in synth_comps:
            if sc.seq != rep0_seq:
                remaining_synth_comps.append(sc)
                continue
            for occ in sc.member_occurrences:
                if occ.seq == rep0_seq:
                    rep0_cand_spans[occ.candidate_id] = (len_rep0, 0, len_rep0, 0)
                else:
                    merged_occ, o_ts, o_te, o_del = _get_occ_rep0_span(occ.seq)
                    if merged_occ:
                        rep0_cand_spans[occ.candidate_id] = (
                            len(occ.seq),
                            o_ts,
                            o_te,
                            o_del,
                        )
            rep0_occurrences.extend(sc.member_occurrences)
            rep0_cand_flags.update(sc.cand_to_flags)

        if had_rep0_split:
            remaining_real_comps.sort(key=_real_comp_sort_key)
            remaining_synth_comps.sort(key=_synth_comp_sort_key)
    else:
        remaining_real_comps = real_comps
        remaining_synth_comps = synth_comps

    # Step 4b: 4-criterion similarity clustering among surviving real sequences
    real_reps: list[ClusterRepresentative] = []
    for rc in remaining_real_comps:
        merged_into: ClusterRepresentative | None = None
        for rep in real_reps:
            if (
                config.dedup_mode == "pan"
                and abs(len(rep.seq) - len(rc.seq)) >= config.min_sv_len
            ):
                continue
            if (
                len(rc.seq) > EDLIB_MAX_LEN or len(rep.seq) > EDLIB_MAX_LEN
            ) and rep.aligner is None:
                rep.aligner = mappy.Aligner(seq=rep.seq, preset="asm20")
            if config.dedup_mode == "pan":
                merged_ok, m_ts, m_te, m_del = _check_similarity_merge_with_span(
                    seq_s=rc.seq,
                    seq_r=rep.seq,
                    min_sv_len=config.min_sv_len,
                    cluster_id=config.cluster_id,
                    cluster_win_id=config.cluster_win_id,
                    aligner=rep.aligner,
                )
                if (
                    merged_ok
                    and (m_ts + (len(rep.seq) - m_te) + m_del) < config.min_sv_len
                ):
                    merged_into = rep
                    break
            elif check_similarity_merge(
                seq_s=rc.seq,
                seq_r=rep.seq,
                min_sv_len=config.min_sv_len,
                cluster_id=config.cluster_id,
                cluster_win_id=config.cluster_win_id,
                aligner=rep.aligner,
            ):
                merged_into = rep
                break

        if merged_into is not None:
            merged_into.support_haps.update(rc.support_haps)
            merged_into.member_occurrences.extend(rc.member_occurrences)
            merged_into.nested_choices_by_cand_id.update(rc.nested_choices_by_cand_id)
            merged_into.reconstructed_len_by_cand_id.update(
                {
                    occ.candidate_id: merged_into.length_bp_no_n
                    for occ in rc.member_occurrences
                }
            )
            new_occ_ids = frozenset(occ.candidate_id for occ in rc.member_occurrences)
            merged_into.piece_cand_ids = tuple(
                cset | new_occ_ids for cset in merged_into.piece_cand_ids
            )
            merged_into.partial_content = (
                merged_into.partial_content or rc.partial_content
            )
            merged_into.repeat_residual = (
                merged_into.repeat_residual or rc.repeat_residual
            )
            merged_into.multi_hit = merged_into.multi_hit or rc.multi_hit
            merged_into.cand_to_flags.update(rc.cand_to_flags)
            merged_into.min_hap = min(merged_into.support_haps)
        else:
            real_reps.append(
                ClusterRepresentative(
                    rep_idx=0,
                    is_synthetic=False,
                    seq=rc.seq,
                    length_bp_no_n=rc.length_bp_no_n,
                    pieces=rc.pieces,
                    support_haps=set(rc.support_haps),
                    token_str=rc.token_str,
                    min_hap=rc.min_hap,
                    exemplar=rc.exemplar,
                    member_occurrences=list(rc.member_occurrences),
                    nested_choices_by_cand_id=dict(rc.nested_choices_by_cand_id),
                    reconstructed_len_by_cand_id=dict(rc.reconstructed_len_by_cand_id),
                    piece_cand_ids=rc.piece_cand_ids,
                    partial_content=rc.partial_content,
                    repeat_residual=rc.repeat_residual,
                    multi_hit=rc.multi_hit,
                    cand_to_flags=dict(rc.cand_to_flags),
                )
            )

    # Synthetic sequences: exact sequence deduplication only (§3.6.2 Step 4)
    synth_reps: list[ClusterRepresentative] = []
    for sc in remaining_synth_comps:
        exact_target: ClusterRepresentative | None = None
        for rep in real_reps + synth_reps:
            if rep.seq == sc.seq:
                exact_target = rep
                break
        if exact_target is not None:
            exact_target.support_haps.update(sc.support_haps)
            exact_target.member_occurrences.extend(sc.member_occurrences)
            exact_target.nested_choices_by_cand_id.update(sc.nested_choices_by_cand_id)
            exact_target.reconstructed_len_by_cand_id.update(
                sc.reconstructed_len_by_cand_id
            )
            if (
                len(exact_target.pieces) == len(sc.pieces)
                and len(exact_target.piece_cand_ids) == len(exact_target.pieces)
                and len(sc.piece_cand_ids) == len(sc.pieces)
                and all(
                    len(p1.seq) == len(p2.seq)
                    for p1, p2 in zip(exact_target.pieces, sc.pieces)
                )
            ):
                exact_target.piece_cand_ids = tuple(
                    c1 | c2
                    for c1, c2 in zip(exact_target.piece_cand_ids, sc.piece_cand_ids)
                )
            else:
                new_occ_ids = frozenset(
                    occ.candidate_id for occ in sc.member_occurrences
                )
                exact_target.piece_cand_ids = tuple(
                    cset | new_occ_ids for cset in exact_target.piece_cand_ids
                )
            exact_target.partial_content = (
                exact_target.partial_content or sc.partial_content
            )
            exact_target.repeat_residual = (
                exact_target.repeat_residual or sc.repeat_residual
            )
            exact_target.multi_hit = exact_target.multi_hit or sc.multi_hit
            exact_target.cand_to_flags.update(sc.cand_to_flags)
            exact_target.min_hap = min(exact_target.support_haps)
        else:
            synth_reps.append(
                ClusterRepresentative(
                    rep_idx=0,
                    is_synthetic=True,
                    seq=sc.seq,
                    length_bp_no_n=sc.length_bp_no_n,
                    pieces=sc.pieces,
                    support_haps=set(sc.support_haps),
                    token_str=sc.token_str,
                    min_hap=sc.min_hap,
                    exemplar=sc.exemplar,
                    member_occurrences=list(sc.member_occurrences),
                    nested_choices_by_cand_id=dict(sc.nested_choices_by_cand_id),
                    reconstructed_len_by_cand_id=dict(sc.reconstructed_len_by_cand_id),
                    piece_cand_ids=sc.piece_cand_ids,
                    partial_content=sc.partial_content,
                    repeat_residual=sc.repeat_residual,
                    multi_hit=sc.multi_hit,
                    cand_to_flags=dict(sc.cand_to_flags),
                )
            )

    all_non_backbone = real_reps + synth_reps
    for rep in all_non_backbone:
        rep.support_hap_count = len(rep.support_haps)

    all_non_backbone.sort(
        key=lambda r: (
            -r.support_hap_count,
            -r.length_bp_no_n,
            r.token_str,
            r.min_hap,
            r.exemplar.provenance[0].src_path,
            r.exemplar.provenance[0].src_start,
            r.exemplar.provenance[0].src_strand,
        )
    )

    cand_to_rep_idx: dict[str, int] = {}
    cand_to_nested_sig: dict[str, tuple[tuple[Any, tuple[int, ...]], ...]] = {}
    cand_to_reconstructed_len: dict[str, int] = {}
    cand_to_flags: dict[str, tuple[bool, bool, bool]] = {}

    for occ in rep0_occurrences:
        cand_to_rep_idx[occ.candidate_id] = 0
        cand_to_nested_sig[occ.candidate_id] = ()
        cand_to_reconstructed_len[occ.candidate_id] = len(rep0_seq)
        cand_to_flags[occ.candidate_id] = rep0_cand_flags.get(
            occ.candidate_id,
            (occ.partial_content, occ.repeat_residual, occ.multi_hit),
        )

    rep0_aligner = None
    for idx_r, rep in enumerate(all_non_backbone, start=1):
        rep.rep_idx = idx_r
        rep.aligner = None
        if not rep.is_synthetic:
            sup_sorted = tuple(sorted(rep.support_haps))
            rep.pieces = tuple(
                replace(p, members=sup_sorted) for p in rep.pieces
            )
        for occ in rep.member_occurrences:
            cand_to_rep_idx[occ.candidate_id] = idx_r
            cand_to_nested_sig[occ.candidate_id] = rep.nested_choices_by_cand_id.get(
                occ.candidate_id, ()
            )
            cand_to_reconstructed_len[occ.candidate_id] = (
                rep.reconstructed_len_by_cand_id.get(
                    occ.candidate_id, rep.length_bp_no_n
                )
            )
            cand_to_flags[occ.candidate_id] = rep.cand_to_flags.get(
                occ.candidate_id,
                (occ.partial_content, occ.repeat_residual, occ.multi_hit),
            )

    return ParallelGroupSolution(
        left_anchor_rank=left_anchor_rank,
        right_anchor_rank=right_anchor_rank,
        has_nonempty_backbone=has_rep0,
        backbone_seq=rep0_seq,
        non_backbone_reps=tuple(all_non_backbone),
        cand_to_rep_idx=cand_to_rep_idx,
        cand_to_nested_sig=cand_to_nested_sig,
        cand_to_reconstructed_len=cand_to_reconstructed_len,
        cand_to_flags=cand_to_flags,
        rep0_cand_spans=rep0_cand_spans,
    )

# ============================================================================
# §3.6.4, §3.8.4-D & §5.1–§5.3 Right-Anchor Slot Assembly & Coordinate Backfilling
# ============================================================================


def assemble_pan_and_offsets(
    graph: GFAGraph,
    slot_blocks_by_step: Mapping[int, Sequence[InsertionSlotBlock]],
    sv_events: Sequence[SVEvent],
    spacer_n: int = 100,
) -> tuple[
    str,
    Sequence[StepOffsetRow],
    tuple[SVEvent, ...],
    tuple[dict[str, Any], ...],
]:
    """
    §3.6.4, §3.8.4-D & §5.1–§5.3:
    Walk reference steps `0 .. len(ref_steps) - 1`, mount all `InsertionSlotBlock`s immediately
    before right anchor `r_idx` (with `CEN_ISLAND` blocks first, followed by normal groups ordered
    by `group_span = b - a` ASC, `a` ASC, separated by single `N * spacer_n` blocks), emit
    `StepOffsetRow`s and `inserts_rows` dicts with final `[pan_start, pan_end)` coordinates, and
    backfill `pan_start` and `pan_end` on all `SVEvent`s.
    """
    pan_chunks: list[str] = []
    pan_cursor = 0
    inserts_rows: list[dict[str, Any]] = []
    ins_event_spans: dict[str, tuple[int, int]] = {}
    pieces_by_event: dict[str, list[Piece]] = collections.defaultdict(list)

    # Pre-index CEN_ISLAND event IDs for each island slot block
    cen_events_by_step: dict[int, list[SVEvent]] = collections.defaultdict(list)
    ref_cum_off = graph._ref_cum_offsets
    for ev in sv_events:
        if ev.event_type == "CEN_ISLAND":
            if ref_cum_off is not None and int(ref_cum_off.shape[0]) > 1:
                starts_arr = ref_cum_off[:-1]
                pos = int(np.searchsorted(starts_arr, ev.ref_start, side="left"))
                if pos < int(starts_arr.shape[0]) and int(starts_arr[pos]) == ev.ref_start:
                    cen_events_by_step[pos].append(ev)
                    continue
            for r_i, (rs, _) in enumerate(graph.ref_step_ranges):
                if rs == ev.ref_start:
                    cen_events_by_step[r_i].append(ev)
                    break

    n_ref = len(graph.ref_fragment.steps)
    frags_by_src_path: dict[str, list[PathFragment]] | None = None
    frag_cum_cache: dict[int, np.ndarray | list[int]] = {}

    def _find_orig_frag(
        src_path: str, src_start: int, src_end: int
    ) -> PathFragment | None:
        nonlocal frags_by_src_path
        if frags_by_src_path is None:
            frags_by_src_path = collections.defaultdict(list)
            frags_by_src_path[graph.ref_fragment.src_path].append(
                graph.ref_fragment
            )
            for af in graph.alt_fragments:
                frags_by_src_path[af.src_path].append(af)
        cands = frags_by_src_path.get(src_path, ())
        for f in cands:
            if f.base_offset <= src_start <= src_end <= f.base_offset + f.total_len:
                return f
        return cands[0] if cands else None

    def _verify_first_piece_source_contiguous(r_idx: int, p: Piece) -> bool:
        if r_idx == 0:
            if graph.alt_fragments:
                frag = _find_orig_frag(p.src_path, p.src_start, p.src_end)
                if frag is not None:
                    return (
                        p.src_start == frag.base_offset
                        if p.src_strand == "+"
                        else p.src_end == frag.base_offset + frag.total_len
                    )
            else:
                if ":" in p.src_path:
                    try:
                        w_s_str, w_e_str = p.src_path.rsplit(":", 1)[1].split(
                            "-", 1
                        )
                        w_s, w_e = int(w_s_str), int(w_e_str)
                        return (
                            p.src_start == w_s
                            if p.src_strand == "+"
                            else p.src_end == w_e
                        )
                    except ValueError:
                        return False
                return p.src_start == 0 if p.src_strand == "+" else False
            return False

        if 1 <= r_idx <= n_ref:
            if not graph.alt_fragments:
                return True
            frag = _find_orig_frag(p.src_path, p.src_start, p.src_end)
            if frag is None or len(frag.steps) == 0:
                return True
            ref_prev = graph.ref_fragment.steps[r_idx - 1]
            l_prev = (
                graph.ref_step_ranges[r_idx - 1][1]
                - graph.ref_step_ranges[r_idx - 1][0]
            )
            if p.src_strand == "+":
                want_s = p.src_start - l_prev
                want_e = p.src_start
                want_orient = ref_prev.orient
            else:
                want_s = p.src_end
                want_e = p.src_end + l_prev
                want_orient = "-" if ref_prev.orient == "+" else "+"
            if (
                want_s < frag.base_offset
                or want_e > frag.base_offset + frag.total_len
            ):
                return False
            fid = id(frag)
            cum = frag_cum_cache.get(fid)
            if cum is None:
                if isinstance(frag.provenance, LazyProvenanceTuple):
                    cum = frag.provenance._get_cum()
                else:
                    cum = [pv.src_start for pv in frag.provenance] + [
                        frag.base_offset + frag.total_len
                    ]
                frag_cum_cache[fid] = cum
            if isinstance(cum, np.ndarray):
                k = int(np.searchsorted(cum, want_s, side="left"))
            else:
                k = bisect.bisect_left(cum, want_s)
            if (
                0 <= k < len(frag.steps)
                and int(cum[k]) == want_s
                and int(cum[k + 1]) == want_e
            ):
                pred_st = frag.steps[k]
                return (
                    _is_full_step(pred_st)
                    and pred_st.node_id == ref_prev.node_id
                    and pred_st.orient == want_orient
                )
            return False

        return False

    def _emit_slot_blocks_at_step(
        r_idx: int,
        blocks: list[InsertionSlotBlock],
    ) -> int:
        nonlocal pan_cursor
        slot_start_pan = pan_cursor
        blocks.sort(
            key=lambda blk: (
                0 if blk.is_centromere_island else 1,
                blk.group_span,
                blk.left_anchor_ref_step,
                blk.event_id,
            )
        )
        prev_trailing_spacer = False
        cen_ev_cursor = 0
        cen_evs_here = cen_events_by_step.get(r_idx, [])

        for b_idx, blk in enumerate(blocks):
            if not blk.representatives:
                continue

            has_nonempty_backbone = (
                (not blk.is_centromere_island) and (blk.group_span > 1)
            )
            need_leading_spacer = (
                blk.is_centromere_island
                or has_nonempty_backbone
                or (b_idx > 0)
                or prev_trailing_spacer
            )
            if need_leading_spacer and spacer_n > 0:
                pan_chunks.append("N" * spacer_n)
                pan_cursor += spacer_n

            blk_first_piece_start = pan_cursor
            for rep_i, rep_pieces in enumerate(blk.representatives):
                if rep_i > 0 and spacer_n > 0:
                    pan_chunks.append("N" * spacer_n)
                    pan_cursor += spacer_n

                if blk.is_centromere_island and cen_ev_cursor < len(cen_evs_here):
                    ev_id_rep = cen_evs_here[cen_ev_cursor].event_id
                    cen_ev_cursor += 1
                else:
                    ev_id_rep = blk.event_id

                rep_start_pan = pan_cursor
                for p_i, p in enumerate(rep_pieces):
                    if p_i > 0 and p.join_before == "spacer" and spacer_n > 0:
                        pan_chunks.append("N" * spacer_n)
                        pan_cursor += spacer_n
                    if p_i == 0:
                        if need_leading_spacer or rep_i > 0:
                            eff_join: Literal[
                                "source_contiguous", "spacer", "synthetic"
                            ] = "spacer"
                        elif (
                            p.join_before == "source_contiguous"
                            and not _verify_first_piece_source_contiguous(
                                r_idx, p
                            )
                        ):
                            eff_join = "synthetic"
                        else:
                            eff_join = p.join_before
                    else:
                        eff_join = p.join_before

                    p_start = pan_cursor
                    p_end = p_start + len(p.seq)
                    pan_chunks.append(p.seq.upper())
                    pan_cursor = p_end

                    placed_piece = replace(p, join_before=eff_join)
                    pieces_by_event[ev_id_rep].append(placed_piece)
                    inserts_rows.append(
                        {
                            "chrom": graph.ref_chrom,
                            "pan_start": p_start,
                            "pan_end": p_end,
                            "event_id": ev_id_rep,
                            "allele": placed_piece.allele_str,
                            "piece": placed_piece.piece_idx,
                            "src_hap": placed_piece.src_hap,
                            "src_path": placed_piece.src_path,
                            "src_intervals": placed_piece.src_intervals_str,
                            "src_strand": placed_piece.src_strand,
                            "join_before": placed_piece.join_before,
                            "members": ",".join(placed_piece.members),
                            "seq": placed_piece.seq.upper(),
                        }
                    )

                rep_end_pan = pan_cursor
                if blk.is_centromere_island and ev_id_rep:
                    ins_event_spans[ev_id_rep] = (rep_start_pan, rep_end_pan)

            blk_last_piece_end = pan_cursor
            if not blk.is_centromere_island and blk.event_id:
                ins_event_spans[blk.event_id] = (
                    blk_first_piece_start,
                    blk_last_piece_end,
                )

            prev_trailing_spacer = blk.trailing_spacer

        if prev_trailing_spacer and spacer_n > 0:
            pan_chunks.append("N" * spacer_n)
            pan_cursor += spacer_n

        return pan_cursor - slot_start_pan

    # Fast vectorized path on chromosome-scale LazyStepTuple reference backbone
    if (
        isinstance(graph.ref_fragment.steps, LazyStepTuple)
        and ref_cum_off is not None
        and n_ref > 10_000
    ):
        slot_ins_lens = np.zeros(n_ref, dtype=np.int64)
        active_slot_steps = sorted(
            r_idx for r_idx, blks in slot_blocks_by_step.items() if blks and 0 <= r_idx < n_ref
        )
        prev_r_idx = 0
        ref_seq_upper = graph.ref_seq.upper()
        for r_idx in active_slot_steps:
            if r_idx > prev_r_idx:
                r_s = int(ref_cum_off[prev_r_idx])
                r_e = int(ref_cum_off[r_idx])
                chunk = ref_seq_upper[r_s:r_e]
                pan_chunks.append(chunk)
                pan_cursor += len(chunk)
            blocks = list(slot_blocks_by_step[r_idx])
            slot_ins_lens[r_idx] = _emit_slot_blocks_at_step(r_idx, blocks)
            prev_r_idx = r_idx

        if prev_r_idx < n_ref:
            r_s = int(ref_cum_off[prev_r_idx])
            r_e = int(ref_cum_off[n_ref])
            chunk = ref_seq_upper[r_s:r_e]
            pan_chunks.append(chunk)
            pan_cursor += len(chunk)

        trailing_blocks_fast = list(slot_blocks_by_step.get(n_ref, ()))
        if trailing_blocks_fast:
            _emit_slot_blocks_at_step(n_ref, trailing_blocks_fast)

        pan_seq = "".join(pan_chunks)
        ref_starts = ref_cum_off[:-1]
        ref_ends = ref_cum_off[1:]
        cum_slot_ins = np.cumsum(slot_ins_lens)
        pan_starts = ref_starts + cum_slot_ins
        pan_ends = ref_ends + cum_slot_ins
        ref_nids = np.abs(graph.ref_fragment.steps._codes)

        step_offsets_seq: Sequence[StepOffsetRow] = LazyStepOffsetSequence(
            chrom=graph.ref_chrom,
            ref_nids=ref_nids,
            ref_starts=ref_starts,
            ref_ends=ref_ends,
            pan_starts=pan_starts,
            pan_ends=pan_ends,
            slot_ins_lens=slot_ins_lens,
        )

        def _map_ref_coord_start_fast(r_coord: int) -> int:
            idx = int(np.searchsorted(ref_ends, r_coord, side="right"))
            if idx < n_ref and int(ref_starts[idx]) <= r_coord < int(ref_ends[idx]):
                return int(pan_starts[idx]) + (r_coord - int(ref_starts[idx]))
            if n_ref > 0 and r_coord == int(ref_ends[-1]):
                return int(pan_ends[-1])
            return 0

        def _map_ref_coord_end_fast(r_coord: int) -> int:
            idx = int(np.searchsorted(ref_ends, r_coord, side="left"))
            if idx < n_ref and int(ref_starts[idx]) < r_coord <= int(ref_ends[idx]):
                return int(pan_starts[idx]) + (r_coord - int(ref_starts[idx]))
            if n_ref > 0 and r_coord == 0:
                return int(pan_starts[0])
            return int(pan_ends[-1]) if n_ref > 0 else len(pan_seq)

        updated_events_fast: list[SVEvent] = []
        for ev in sv_events:
            if ev.event_type in ("INS", "CEN_ISLAND") and ev.event_id in ins_event_spans:
                p_s, p_e = ins_event_spans[ev.event_id]
                ev_pieces = tuple(pieces_by_event.get(ev.event_id, ev.pieces))
                updated_events_fast.append(
                    replace(ev, pan_start=p_s, pan_end=p_e, pieces=ev_pieces)
                )
            else:
                if ev.ref_start == ev.ref_end:
                    p_s = _map_ref_coord_start_fast(ev.ref_start)
                    p_e = p_s
                else:
                    p_s = _map_ref_coord_start_fast(ev.ref_start)
                    p_e = _map_ref_coord_end_fast(ev.ref_end)
                updated_events_fast.append(replace(ev, pan_start=p_s, pan_end=p_e))

        return (
            pan_seq,
            step_offsets_seq,
            tuple(updated_events_fast),
            tuple(inserts_rows),
        )

    step_offsets: list[StepOffsetRow] = []
    for r_idx, ref_st in enumerate(graph.ref_fragment.steps):
        blocks = list(slot_blocks_by_step.get(r_idx, ()))
        if blocks:
            slot_ins_len = _emit_slot_blocks_at_step(r_idx, blocks)
        else:
            slot_ins_len = 0

        ref_s, ref_e = graph.ref_step_ranges[r_idx]
        step_seq = graph.ref_seq[ref_s:ref_e].upper()
        pan_s = pan_cursor
        pan_e = pan_s + len(step_seq)
        pan_chunks.append(step_seq)
        pan_cursor = pan_e

        step_offsets.append(
            StepOffsetRow(
                chrom=graph.ref_chrom,
                ref_step=r_idx,
                node_id=ref_st.node_id,
                ref_start=ref_s,
                ref_end=ref_e,
                pan_start=pan_s,
                pan_end=pan_e,
                slot_ins_len=slot_ins_len,
            )
        )

    trailing_blocks = list(slot_blocks_by_step.get(n_ref, ()))
    if trailing_blocks:
        _emit_slot_blocks_at_step(n_ref, trailing_blocks)

    pan_seq = "".join(pan_chunks)

    ref_start_to_step = {row.ref_start: row for row in step_offsets}
    ref_end_to_step = {row.ref_end: row for row in step_offsets}

    def _map_ref_coord_start(r_coord: int) -> int:
        if r_coord in ref_start_to_step:
            return ref_start_to_step[r_coord].pan_start
        for row in step_offsets:
            if row.ref_start <= r_coord < row.ref_end:
                return row.pan_start + (r_coord - row.ref_start)
        if step_offsets and r_coord == step_offsets[-1].ref_end:
            return step_offsets[-1].pan_end
        return 0

    def _map_ref_coord_end(r_coord: int) -> int:
        if r_coord in ref_end_to_step:
            return ref_end_to_step[r_coord].pan_end
        for row in step_offsets:
            if row.ref_start < r_coord <= row.ref_end:
                return row.pan_start + (r_coord - row.ref_start)
        if step_offsets and r_coord == 0:
            return step_offsets[0].pan_start
        return len(pan_seq)

    updated_events: list[SVEvent] = []
    for ev in sv_events:
        if ev.event_type in ("INS", "CEN_ISLAND") and ev.event_id in ins_event_spans:
            p_s, p_e = ins_event_spans[ev.event_id]
            ev_pieces = tuple(pieces_by_event.get(ev.event_id, ev.pieces))
            updated_events.append(
                replace(ev, pan_start=p_s, pan_end=p_e, pieces=ev_pieces)
            )
        else:
            if ev.ref_start == ev.ref_end:
                p_s = _map_ref_coord_start(ev.ref_start)
                p_e = p_s
            else:
                p_s = _map_ref_coord_start(ev.ref_start)
                p_e = _map_ref_coord_end(ev.ref_end)
            updated_events.append(replace(ev, pan_start=p_s, pan_end=p_e))

    return pan_seq, tuple(step_offsets), tuple(updated_events), tuple(inserts_rows)


# ============================================================================
# §3.3–§3.8, §5.1–§5.5 Single-Chromosome Pipeline Orchestrator
# ============================================================================


@dataclass(frozen=True, slots=True)
class _ContigFragMeta:
    """Lightweight per-contig metadata avoiding retention of full PathFragment in _ContigAnalysis."""

    hap_id: str
    src_path: str


@dataclass
class _ContigAnalysis:
    """Per-contig post-normalization main-chain and deviation analysis at top level."""

    raw_fragment: _ContigFragMeta | PathFragment
    lis_multi_hit: bool
    covered_ranks: set[int] | CompactRankSet
    unnormalized_rank_spans: list[tuple[int, int]]
    large_jump_rank_spans: list[tuple[int, int]]
    node_visit_counts: Mapping[str, int]
    has_main_chain: bool = False
    contig_idx: int = -1
    normalized_inv_rank_spans: list[tuple[int, int]] = field(
        default_factory=list
    )
    small_dev_bp_spans: list[tuple[int, int, int]] = field(
        default_factory=list
    )
    unnormalized_gap_spans: list[tuple[int, int]] = field(
        default_factory=list
    )


def _supports_fork() -> bool:
    return "fork" in multiprocessing.get_all_start_methods()


def _force_fork_enabled() -> bool:
    return os.environ.get("PANFLAT_FORCE_FORK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_max_fork_workers(threads: int) -> int:
    cap = max(1, int(threads))
    raw = os.environ.get("PANFLAT_MAX_FORK_WORKERS", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed >= 1:
                cap = min(cap, parsed)
        except ValueError:
            pass
    return cap


@contextlib.contextmanager
def _fork_gc_frozen() -> Iterator[None]:
    _repeat_mask_intervals_cached.cache_clear()
    gc.collect()
    froze = False
    if hasattr(gc, "freeze") and hasattr(gc, "unfreeze"):
        gc.freeze()
        froze = True
    try:
        yield
    finally:
        if froze:
            gc.unfreeze()


def _resolve_graph_node_lens(graph: GFAGraph) -> np.ndarray | None:
    if graph._node_lens is not None:
        return graph._node_lens
    if isinstance(graph.nodes, CompactNodeStore):
        return graph.nodes._node_lens
    return None


def _strip_dev_node_lens(
    dev: DeviationSegment,
    target_node_lens: np.ndarray | None,
) -> DeviationSegment:
    if (
        target_node_lens is not None
        and isinstance(dev.branch_provenance, LazyProvenanceTuple)
        and dev.branch_provenance._node_lens is target_node_lens
    ):
        return replace(
            dev,
            branch_provenance=dev.branch_provenance.without_node_lens(),
        )
    return dev


def _strip_chain_node_lens(
    ch: ReverseChainRecord,
    target_node_lens: np.ndarray | None,
) -> ReverseChainRecord:
    if target_node_lens is None or not ch.novel_unresolved_slices:
        return ch
    changed = False
    new_slices: list[tuple[Any, Any]] = []
    for sub_steps, sub_provs in ch.novel_unresolved_slices:
        if (
            isinstance(sub_provs, LazyProvenanceTuple)
            and sub_provs._node_lens is target_node_lens
        ):
            new_slices.append((sub_steps, sub_provs.without_node_lens()))
            changed = True
        else:
            new_slices.append((sub_steps, sub_provs))
    if not changed:
        return ch
    return replace(ch, novel_unresolved_slices=tuple(new_slices))


@dataclass(slots=True)
class _Stage4ForkContext:
    graph: GFAGraph
    config: BuildConfig
    normal_fragments_sorted: list[PathFragment]
    ref_steps: Any
    n_ref: int
    top_state_key: SubproblemStateKey
    claimed_source_intervals: set[tuple[str, int, int]]
    target_node_lens: np.ndarray | None


@dataclass(slots=True)
class _FragmentStage4Result:
    frag_idx: int
    analysis: _ContigAnalysis
    visit_counts_ipc: np.ndarray | dict[str, int] | None
    inv_chains: list[ReverseChainRecord]
    terminal_devs: list[tuple[tuple[int, int], DeviationSegment]]
    large_jump_devs: list[tuple[tuple[int, int], DeviationSegment]]
    pending_unanchored_unresolved: list[tuple[StepProvenance, str]]


_STAGE4_FORK_CTX: _Stage4ForkContext | None = None


def _rebind_stage4_result_after_ipc(
    res: _FragmentStage4Result,
    target_node_lens: np.ndarray | None,
) -> None:
    if target_node_lens is not None:
        for ch in res.inv_chains:
            for _, sub_provs in ch.novel_unresolved_slices:
                if (
                    isinstance(sub_provs, LazyProvenanceTuple)
                    and int(sub_provs._node_lens.shape[0]) == 0
                ):
                    sub_provs.rebind_node_lens(target_node_lens)
        for _, dev in res.terminal_devs:
            if (
                isinstance(dev.branch_provenance, LazyProvenanceTuple)
                and int(dev.branch_provenance._node_lens.shape[0]) == 0
            ):
                dev.branch_provenance.rebind_node_lens(target_node_lens)
        for _, dev in res.large_jump_devs:
            if (
                isinstance(dev.branch_provenance, LazyProvenanceTuple)
                and int(dev.branch_provenance._node_lens.shape[0]) == 0
            ):
                dev.branch_provenance.rebind_node_lens(target_node_lens)
    vc_ipc = res.visit_counts_ipc
    if isinstance(vc_ipc, np.ndarray):
        res.analysis.node_visit_counts = CompactVisitCounts(multi_nids=vc_ipc)
    elif isinstance(vc_ipc, dict):
        res.analysis.node_visit_counts = vc_ipc
    res.visit_counts_ipc = None


def _analyze_single_normal_fragment(
    frag_idx: int,
    ctx: _Stage4ForkContext,
    *,
    for_ipc: bool,
) -> _FragmentStage4Result:
    """
    Pure per-fragment Stage 2.4 analysis: inversion normalization, weighted LIS,
    deviation extraction, local re-anchoring, micro-variant filtering, and IPC stripping.
    """
    graph = ctx.graph
    config = ctx.config
    normal_fragments_sorted = ctx.normal_fragments_sorted
    ref_steps = ctx.ref_steps
    n_ref = ctx.n_ref
    top_state_key = ctx.top_state_key
    claimed_source_intervals = ctx.claimed_source_intervals
    target_node_lens = ctx.target_node_lens
    frag = normal_fragments_sorted[frag_idx]

    had_cum = (
        isinstance(frag.provenance, LazyProvenanceTuple)
        and frag.provenance._cum_cache[0] is not None
    )

    inv_res = normalize_inversions_top_level(frag, graph, config)
    norm_frag = inv_res.normalized_fragment
    proj_anchors = project_steps_onto_backbone(
        norm_frag.steps,
        norm_frag.provenance,
        ref_steps,
        graph.nodes,
        lo=0,
        hi=n_ref,
    )
    main_chain, lis_mh = compute_weighted_lis(proj_anchors)

    if (
        isinstance(main_chain, LazyProjectedAnchorSequence)
        and isinstance(norm_frag.steps, LazyStepTuple)
        and isinstance(ref_steps, LazyStepTuple)
    ):
        cov_ranks: set[int] | CompactRankSet = CompactRankSet(n_ref)
        mc_b = main_chain.branch_step_idx
        mc_r = main_chain.rank_in_b
        cov_ranks.add_array(mc_r)
        exact_ref_match_mask: np.ndarray | None = None
        if int(mc_b.shape[0]) > 1:
            b_starts = mc_b[:-1] + 1
            b_ends = mc_b[1:]
            r_starts = mc_r[:-1] + 1
            r_ends = mc_r[1:]
            b_lens = b_ends - b_starts
            r_lens = r_ends - r_starts
            exact_ref_match_mask = (b_lens == 0) & (r_lens == 0)
            unnorm_ranges = inv_res.unnormalized_branch_ranges
            f_codes_arr = norm_frag.steps._codes
            r_codes_arr = ref_steps._codes

            is_1x1 = (b_lens == 1) & (r_lens == 1)
            if bool(np.any(is_1x1)):
                idx_1x1 = np.flatnonzero(is_1x1)
                b_1x1 = b_starts[idx_1x1]
                r_1x1 = r_starts[idx_1x1]
                eq_1x1_sub = f_codes_arr[b_1x1] == r_codes_arr[r_1x1]
                if bool(np.any(eq_1x1_sub)):
                    eq_1x1_k = idx_1x1[eq_1x1_sub]
                    if unnorm_ranges:
                        kept_1x1_k = [
                            k
                            for k in eq_1x1_k.tolist()
                            if not any(
                                max(int(b_starts[k]), u_s)
                                < min(int(b_ends[k]), u_e)
                                for u_s, u_e in unnorm_ranges
                            )
                        ]
                        if kept_1x1_k:
                            kept_arr = np.asarray(kept_1x1_k, dtype=np.int32)
                            exact_ref_match_mask[kept_arr] = True
                            cov_ranks.add_array(r_starts[kept_arr])
                    else:
                        exact_ref_match_mask[eq_1x1_k] = True
                        cov_ranks.add_array(r_starts[eq_1x1_k])

            eq_gt1_k = np.flatnonzero(
                (r_lens > 1) & (b_lens == r_lens)
            ).tolist()
            for k in eq_gt1_k:
                b_s = int(b_starts[k])
                b_e = int(b_ends[k])
                overlaps_unnorm = any(
                    max(b_s, u_s) < min(b_e, u_e) for u_s, u_e in unnorm_ranges
                )
                if not overlaps_unnorm:
                    r_s = int(r_starts[k])
                    r_e = int(r_ends[k])
                    if bool(
                        np.array_equal(
                            f_codes_arr[b_s:b_e], r_codes_arr[r_s:r_e]
                        )
                    ):
                        exact_ref_match_mask[k] = True
                        cov_ranks.add_range(r_s, r_e)
    else:
        exact_ref_match_mask = None
        cov_ranks = {anc.rank_in_b for anc in main_chain}
        for k in range(len(main_chain) - 1):
            al = main_chain[k]
            ar = main_chain[k + 1]
            b_s = al.branch_step_idx + 1
            b_e = ar.branch_step_idx
            overlaps_unnorm = any(
                max(b_s, u_s) < min(b_e, u_e)
                for u_s, u_e in inv_res.unnormalized_branch_ranges
            )
            if not overlaps_unnorm and norm_frag.steps[b_s:b_e] == tuple(
                ref_steps[al.rank_in_b + 1 : ar.rank_in_b]
            ):
                for r in range(al.rank_in_b, ar.rank_in_b + 1):
                    cov_ranks.add(r)

    unnorm_rank_spans: list[tuple[int, int]] = []
    unnorm_gap_spans: list[tuple[int, int]] = []
    norm_inv_rank_spans: list[tuple[int, int]] = []
    for ch in inv_res.chains:
        if not ch.normalized:
            unnorm_rank_spans.append((ch.rmin, ch.rmax))
        else:
            norm_inv_rank_spans.append((ch.rmin, ch.rmax))
    if inv_res.unnormalized_branch_ranges and main_chain:
        u_ranges_sp = inv_res.unnormalized_branch_ranges
        if isinstance(main_chain, LazyProjectedAnchorSequence):
            mc_b_u = main_chain.branch_step_idx
            mc_r_u = main_chain.rank_in_b
            for k_u in range(int(mc_b_u.shape[0]) - 1):
                bs_u = int(mc_b_u[k_u]) + 1
                be_u = int(mc_b_u[k_u + 1])
                if any(max(bs_u, us) < min(be_u, ue) for us, ue in u_ranges_sp):
                    r_lo_u = int(mc_r_u[k_u])
                    r_hi_u = int(mc_r_u[k_u + 1])
                    if r_lo_u + 1 <= r_hi_u - 1:
                        unnorm_rank_spans.append((r_lo_u + 1, r_hi_u - 1))
                    else:
                        unnorm_gap_spans.append((r_lo_u, r_hi_u))
        else:
            for k_u in range(len(main_chain) - 1):
                bs_u = main_chain[k_u].branch_step_idx + 1
                be_u = main_chain[k_u + 1].branch_step_idx
                if any(max(bs_u, us) < min(be_u, ue) for us, ue in u_ranges_sp):
                    r_lo_u = main_chain[k_u].rank_in_b
                    r_hi_u = main_chain[k_u + 1].rank_in_b
                    if r_lo_u + 1 <= r_hi_u - 1:
                        unnorm_rank_spans.append((r_lo_u + 1, r_hi_u - 1))
                    else:
                        unnorm_gap_spans.append((r_lo_u, r_hi_u))
    if len(unnorm_rank_spans) > 1:
        unnorm_rank_spans = sorted(set(unnorm_rank_spans))

    visit_counts = compute_raw_visit_counts(norm_frag.steps)
    analysis = _ContigAnalysis(
        raw_fragment=_ContigFragMeta(
            hap_id=frag.hap_id, src_path=frag.src_path
        ),
        lis_multi_hit=lis_mh,
        covered_ranks=cov_ranks,
        unnormalized_rank_spans=unnorm_rank_spans,
        large_jump_rank_spans=[],
        node_visit_counts=visit_counts,
        has_main_chain=bool(main_chain),
        contig_idx=frag_idx,
        normalized_inv_rank_spans=norm_inv_rank_spans,
        unnormalized_gap_spans=unnorm_gap_spans,
    )

    if for_ipc:
        inv_chains = [
            _strip_chain_node_lens(ch, target_node_lens)
            for ch in inv_res.chains
        ]
    else:
        inv_chains = list(inv_res.chains)

    # Fast vectorized filtering of micro-SNP deviations (< min_sv_len with 0 ref nodes)
    # on chromosome-scale contigs before allocating DeviationSegment objects
    if (
        isinstance(main_chain, LazyProjectedAnchorSequence)
        and isinstance(norm_frag.steps, LazyStepTuple)
        and isinstance(ref_steps, LazyStepTuple)
        and graph._node_lens is not None
        and (
            int(norm_frag.steps._codes.shape[0]) == 0
            or int(np.abs(norm_frag.steps._codes).max())
            < int(graph._node_lens.shape[0])
        )
        and graph._ref_cum_offsets is not None
        and graph._nid_to_ref_rank is not None
        and isinstance(graph.global_ref_nodes, CompactRefNodeSet)
        and len(graph.global_ref_nodes._extra) == 0
    ):
        raw_devs: list[DeviationSegment] = []
        n_frag_steps = len(norm_frag.steps)
        unnorm_ranges = inv_res.unnormalized_branch_ranges

        def _ov_unnorm(bs: int, be: int) -> bool:
            if not unnorm_ranges:
                return False
            for us, ue in unnorm_ranges:
                if bs == be:
                    if us <= bs <= ue:
                        return True
                elif max(bs, us) < min(be, ue):
                    return True
            return False

        mh_flag = (
            ("1",)
            if (
                isinstance(proj_anchors, LazyProjectedAnchorSequence)
                and proj_anchors.has_multi_hit
            )
            else ()
        )

        mc_b = main_chain.branch_step_idx
        mc_r = main_chain.rank_in_b
        if int(mc_b.shape[0]) == 0:
            if n_frag_steps > 0 and not _ov_unnorm(0, n_frag_steps):
                raw_devs.append(
                    DeviationSegment(
                        hap_id=norm_frag.hap_id,
                        src_path=norm_frag.src_path,
                        left_anchor_rank="START",
                        right_anchor_rank="END",
                        branch_start_idx=0,
                        branch_end_idx=n_frag_steps,
                        branch_steps=norm_frag.steps[0:n_frag_steps],
                        branch_provenance=norm_frag.provenance[0:n_frag_steps],
                        backbone_steps=(),
                        multi_hit_nodes=mh_flag,
                        is_unanchored_start=True,
                        is_unanchored_end=True,
                    )
                )
        else:
            first_b = int(mc_b[0])
            if first_b > 0 and not _ov_unnorm(0, first_b):
                raw_devs.append(
                    DeviationSegment(
                        hap_id=norm_frag.hap_id,
                        src_path=norm_frag.src_path,
                        left_anchor_rank="START",
                        right_anchor_rank=int(mc_r[0]),
                        branch_start_idx=0,
                        branch_end_idx=first_b,
                        branch_steps=norm_frag.steps[0:first_b],
                        branch_provenance=norm_frag.provenance[0:first_b],
                        backbone_steps=(),
                        multi_hit_nodes=mh_flag,
                        is_unanchored_start=True,
                        is_unanchored_end=False,
                    )
                )

            if int(mc_b.shape[0]) > 1:
                f_nids = np.abs(norm_frag.steps._codes)
                f_lens = graph._node_lens[f_nids]
                frag_cum_bp = np.empty(n_frag_steps + 1, dtype=np.int64)
                frag_cum_bp[0] = 0
                np.cumsum(f_lens, dtype=np.int64, out=frag_cum_bp[1:])

                m_len_ref = int(graph.global_ref_nodes._mask.shape[0])
                f_is_ref = (
                    (f_nids < m_len_ref)
                    & graph.global_ref_nodes._mask[
                        np.minimum(f_nids, max(0, m_len_ref - 1))
                    ]
                ).astype(np.int32)
                frag_cum_ref = np.empty(n_frag_steps + 1, dtype=np.int32)
                frag_cum_ref[0] = 0
                np.cumsum(f_is_ref, dtype=np.int32, out=frag_cum_ref[1:])

                r_rank_len = int(graph._nid_to_ref_rank.shape[0])
                f_is_uniq_ref = (
                    (f_nids < r_rank_len)
                    & (
                        graph._nid_to_ref_rank[
                            np.minimum(f_nids, max(0, r_rank_len - 1))
                        ]
                        >= 0
                    )
                ).astype(np.int32)
                frag_cum_uniq_ref = np.empty(
                    n_frag_steps + 1, dtype=np.int32
                )
                frag_cum_uniq_ref[0] = 0
                np.cumsum(
                    f_is_uniq_ref, dtype=np.int32, out=frag_cum_uniq_ref[1:]
                )

                b_starts = mc_b[:-1] + 1
                b_ends = mc_b[1:]
                r_starts = mc_r[:-1] + 1
                r_ends = mc_r[1:]

                b_bp = frag_cum_bp[b_ends] - frag_cum_bp[b_starts]
                r_bp = (
                    graph._ref_cum_offsets[r_ends]
                    - graph._ref_cum_offsets[r_starts]
                )
                b_ref_hits = frag_cum_ref[b_ends] - frag_cum_ref[b_starts]
                b_uniq_ref_hits = (
                    frag_cum_uniq_ref[b_ends] - frag_cum_uniq_ref[b_starts]
                )

                # Keep only non-identical pairs that can produce an SV >= min_sv_len, LARGE_JUMP, DUP_TRA, or local re-anchoring
                cand_mask = (
                    (b_bp >= config.min_sv_len)
                    | (r_bp >= config.min_sv_len)
                    | (r_bp > config.max_jump)
                    | (b_uniq_ref_hits > 0)
                    | ((r_bp > 0) & (b_ref_hits > 0))
                )
                if exact_ref_match_mask is not None:
                    cand_mask &= ~exact_ref_match_mask

                dropped_small_mask = (~cand_mask) & (b_bp > 0)
                if exact_ref_match_mask is not None:
                    dropped_small_mask &= ~exact_ref_match_mask
                for k_sm in np.flatnonzero(dropped_small_mask).tolist():
                    if not _ov_unnorm(int(b_starts[k_sm]), int(b_ends[k_sm])):
                        analysis.small_dev_bp_spans.append(
                            (int(mc_r[k_sm]), int(mc_r[k_sm + 1]), int(b_bp[k_sm]))
                        )

                ref_codes_abs = np.abs(ref_steps._codes)
                for k in np.flatnonzero(cand_mask).tolist():
                    b_s = int(b_starts[k])
                    b_e = int(b_ends[k])
                    if _ov_unnorm(b_s, b_e):
                        continue
                    r_left = int(mc_r[k])
                    r_right = int(mc_r[k + 1])
                    r_s = r_left + 1
                    if (
                        int(b_bp[k]) < config.min_sv_len
                        and int(r_bp[k]) < config.min_sv_len
                        and int(r_bp[k]) <= config.max_jump
                        and int(b_uniq_ref_hits[k]) == 0
                    ):
                        if b_e - b_s == 1 and r_right - r_s == 1:
                            if int(f_nids[b_s]) != int(ref_codes_abs[r_s]):
                                if int(b_bp[k]) > 0:
                                    analysis.small_dev_bp_spans.append(
                                        (r_left, r_right, int(b_bp[k]))
                                    )
                                continue
                        elif set(f_nids[b_s:b_e].tolist()).isdisjoint(
                            ref_codes_abs[r_s:r_right].tolist()
                        ):
                            if int(b_bp[k]) > 0:
                                analysis.small_dev_bp_spans.append(
                                    (r_left, r_right, int(b_bp[k]))
                                )
                            continue
                    br_st = norm_frag.steps[b_s:b_e]
                    b_st: Any = ref_steps[r_s:r_right]
                    if br_st != b_st:
                        raw_devs.append(
                            DeviationSegment(
                                hap_id=norm_frag.hap_id,
                                src_path=norm_frag.src_path,
                                left_anchor_rank=r_left,
                                right_anchor_rank=r_right,
                                branch_start_idx=b_s,
                                branch_end_idx=b_e,
                                branch_steps=br_st,
                                branch_provenance=norm_frag.provenance[b_s:b_e],
                                backbone_steps=b_st,
                                multi_hit_nodes=mh_flag,
                                is_unanchored_start=False,
                                is_unanchored_end=False,
                                left_anchor_contiguous=(
                                    True
                                    if b_s >= b_e
                                    else _step_boundary_contiguous(
                                        norm_frag.provenance, b_s
                                    )
                                ),
                            )
                        )

            last_b = int(mc_b[-1])
            if last_b + 1 < n_frag_steps and not _ov_unnorm(last_b + 1, n_frag_steps):
                raw_devs.append(
                    DeviationSegment(
                        hap_id=norm_frag.hap_id,
                        src_path=norm_frag.src_path,
                        left_anchor_rank=int(mc_r[-1]),
                        right_anchor_rank="END",
                        branch_start_idx=last_b + 1,
                        branch_end_idx=n_frag_steps,
                        branch_steps=norm_frag.steps[last_b + 1 : n_frag_steps],
                        branch_provenance=norm_frag.provenance[
                            last_b + 1 : n_frag_steps
                        ],
                        backbone_steps=(),
                        multi_hit_nodes=mh_flag,
                        is_unanchored_start=False,
                        is_unanchored_end=True,
                    )
                )
    else:
        raw_devs = extract_deviations(
            fragment=norm_frag,
            main_chain=main_chain,
            backbone_steps=ref_steps,
            is_top_level=True,
            lo=0,
            hi=n_ref,
            unnormalized_branch_ranges=inv_res.unnormalized_branch_ranges,
            all_projected_anchors=proj_anchors,
        )

    pending_unanchored_unresolved: list[tuple[StepProvenance, str]] = []
    large_jump_devs: list[tuple[tuple[int, int], DeviationSegment]] = []
    terminal_devs: list[tuple[tuple[int, int], DeviationSegment]] = []

    for dev in raw_devs:
        if dev.is_unanchored_start or dev.is_unanchored_end:
            unanch_runs = _extract_contiguous_nonref_runs_for_unresolved(
                steps=dev.branch_steps,
                provenance=dev.branch_provenance,
                nodes=graph.nodes,
                global_ref_nodes=graph.global_ref_nodes,
                min_sv_len=config.min_sv_len,
                claimed_source_intervals=claimed_source_intervals,
            )
            for u_prov, u_seq in unanch_runs:
                pending_unanchored_unresolved.append((u_prov, u_seq))
            continue

        rank_a = int(dev.left_anchor_rank)
        rank_b = int(dev.right_anchor_rank)
        span_bp = graph.ref_step_ranges[rank_b][0] - graph.ref_step_ranges[rank_a][1]

        if span_bp > config.max_jump:
            analysis.large_jump_rank_spans.append((rank_a, rank_b))
            det_lj = dev.detached()
            if for_ipc:
                det_lj = _strip_dev_node_lens(det_lj, target_node_lens)
            large_jump_devs.append(((rank_a, rank_b), det_lj))
            continue

        can_reanchor = rank_a + 1 < rank_b and bool(dev.branch_steps)
        if (
            can_reanchor
            and isinstance(dev.branch_steps, LazyStepTuple)
            and isinstance(graph.global_ref_nodes, CompactRefNodeSet)
            and len(graph.global_ref_nodes._extra) == 0
        ):
            br_nids_ra = np.abs(dev.branch_steps._codes)
            m_len_ra = int(graph.global_ref_nodes._mask.shape[0])
            if int(br_nids_ra.shape[0]) == 0:
                can_reanchor = False
            elif int(br_nids_ra.max()) < m_len_ra:
                can_reanchor = bool(
                    np.any(graph.global_ref_nodes._mask[br_nids_ra])
                )

        if can_reanchor and not _shares_canonical_tandem_unit(
            ref_steps[rank_a + 1 : rank_b], dev.branch_steps
        ):
            reanchored_devs = reanchor_deviation(
                dev=dev,
                backbone_steps=ref_steps,
                nodes=graph.nodes,
                lo=0,
                hi=n_ref,
                parent_key=top_state_key,
                ancestor_keys=frozenset(),
                covered_ranks_out=analysis.covered_ranks,
            )
        else:
            reanchored_devs = [dev]

        for tdev in reanchored_devs:
            ta = int(tdev.left_anchor_rank)
            tb = int(tdev.right_anchor_rank)
            t_span_bp = (
                graph.ref_step_ranges[tb][0] - graph.ref_step_ranges[ta][1]
            )
            if t_span_bp > config.max_jump:
                analysis.large_jump_rank_spans.append((ta, tb))
                det_lj = tdev.detached()
                if for_ipc:
                    det_lj = _strip_dev_node_lens(det_lj, target_node_lens)
                large_jump_devs.append(((ta, tb), det_lj))
            else:
                if t_span_bp < config.min_sv_len:
                    if (
                        isinstance(tdev.branch_steps, LazyStepTuple)
                        and isinstance(ref_steps, LazyStepTuple)
                        and graph._node_lens is not None
                        and (
                            int(tdev.branch_steps._codes.shape[0]) == 0
                            or int(np.abs(tdev.branch_steps._codes).max())
                            < int(graph._node_lens.shape[0])
                        )
                        and graph._nid_to_ref_rank is not None
                    ):
                        t_nids = np.abs(tdev.branch_steps._codes)
                        t_alt_bp = int(graph._node_lens[t_nids].sum())
                        if t_alt_bp < config.min_sv_len:
                            r_rank_len_t = int(
                                graph._nid_to_ref_rank.shape[0]
                            )
                            has_uniq_ref = bool(
                                np.any(
                                    (t_nids < r_rank_len_t)
                                    & (
                                        graph._nid_to_ref_rank[
                                            np.minimum(
                                                t_nids,
                                                max(0, r_rank_len_t - 1),
                                            )
                                        ]
                                        >= 0
                                    )
                                )
                            )
                            if not has_uniq_ref:
                                r_slice_c = ref_steps._codes[ta + 1 : tb]
                                t_slice_c = tdev.branch_steps._codes
                                same_tokens = (
                                    (ta + 1 < tb)
                                    and (int(t_nids.shape[0]) > 0)
                                    and (
                                        int(t_slice_c.min())
                                        == int(r_slice_c.min())
                                    )
                                    and (
                                        int(t_slice_c.max())
                                        == int(r_slice_c.max())
                                    )
                                    and bool(
                                        np.array_equal(
                                            np.unique(t_slice_c),
                                            np.unique(r_slice_c),
                                        )
                                    )
                                )
                                if not same_tokens:
                                    if t_alt_bp > 0:
                                        analysis.small_dev_bp_spans.append(
                                            (ta, tb, t_alt_bp)
                                        )
                                    continue
                    else:
                        t_alt_bp = sum(
                            _step_bp_len(st, graph.nodes)
                            for st in tdev.branch_steps
                        )
                        if t_alt_bp < config.min_sv_len:
                            has_uniq_ref = any(
                                _is_full_step(st)
                                and st.node_id in graph.unique_ref_step_by_node
                                for st in tdev.branch_steps
                            )
                            if not has_uniq_ref:
                                r_slice_steps = ref_steps[ta + 1 : tb]
                                same_tokens = (
                                    (ta + 1 < tb)
                                    and bool(tdev.branch_steps)
                                    and {
                                        (st.node_id, st.orient)
                                        for st in tdev.branch_steps
                                        if _is_full_step(st)
                                    }
                                    == {
                                        (st.node_id, st.orient)
                                        for st in r_slice_steps
                                        if _is_full_step(st)
                                    }
                                )
                                if not same_tokens:
                                    if t_alt_bp > 0:
                                        analysis.small_dev_bp_spans.append(
                                            (ta, tb, t_alt_bp)
                                        )
                                    continue
                keep_prov = True
                if (
                    isinstance(tdev.branch_steps, LazyStepTuple)
                    and isinstance(graph.global_ref_nodes, CompactRefNodeSet)
                    and len(graph.global_ref_nodes._extra) == 0
                    and graph._node_lens is not None
                ):
                    t_nids_p = np.abs(tdev.branch_steps._codes)
                    m_len_p = int(graph.global_ref_nodes._mask.shape[0])
                    nl_len_p = int(graph._node_lens.shape[0])
                    if int(t_nids_p.shape[0]) == 0:
                        keep_prov = False
                    elif int(t_nids_p.max()) < min(m_len_p, nl_len_p):
                        nonref_m = ~graph.global_ref_nodes._mask[t_nids_p]
                        if (
                            not bool(np.any(nonref_m))
                            or int(
                                graph._node_lens[t_nids_p[nonref_m]].sum()
                            )
                            < config.min_sv_len
                        ):
                            keep_prov = False
                det_tdev = tdev.detached(keep_provenance=keep_prov)
                if for_ipc:
                    det_tdev = _strip_dev_node_lens(det_tdev, target_node_lens)
                terminal_devs.append(((ta, tb), det_tdev))

    empty_i32 = np.empty(0, dtype=np.int32)
    visit_counts_ipc: np.ndarray | dict[str, int] | None = None
    if for_ipc:
        if isinstance(visit_counts, CompactVisitCounts):
            visit_counts_ipc = visit_counts._multi_nids
        else:
            visit_counts_ipc = {
                k: 2 for k, v in visit_counts.items() if v > 1
            }
        analysis.node_visit_counts = {}
        # Worker-private cleanup only: NEVER mutate shared input `frag.steps` or `frag.provenance._orig_codes`
        if norm_frag is not frag:
            if (
                isinstance(norm_frag.steps, LazyStepTuple)
                and norm_frag.steps._codes is not getattr(frag.steps, "_codes", None)
            ):
                norm_frag.steps._codes = empty_i32
                norm_frag.steps._anchor_cache = None
            if (
                isinstance(norm_frag.provenance, LazyProvenanceTuple)
                and norm_frag.provenance._orig_codes
                is not getattr(frag.provenance, "_orig_codes", None)
            ):
                norm_frag.provenance._orig_codes = empty_i32
        if not had_cum and isinstance(frag.provenance, LazyProvenanceTuple):
            frag.provenance._cum_cache[0] = None
    else:
        if isinstance(norm_frag.steps, LazyStepTuple):
            norm_frag.steps._codes = empty_i32
            norm_frag.steps._anchor_cache = None
        if isinstance(norm_frag.provenance, LazyProvenanceTuple):
            norm_frag.provenance._cum_cache[0] = None
            if norm_frag.provenance._orig_codes is not getattr(
                frag.provenance, "_orig_codes", None
            ):
                norm_frag.provenance._orig_codes = empty_i32
        if isinstance(frag.provenance, LazyProvenanceTuple):
            frag.provenance._cum_cache[0] = None
            is_last_for_src = (
                frag_idx + 1 == len(normal_fragments_sorted)
                or normal_fragments_sorted[frag_idx + 1].src_path != frag.src_path
            )
            if isinstance(frag.steps, LazyStepTuple) and n_ref > 100_000:
                if (
                    frag.steps._codes is not frag.provenance._orig_codes
                    or is_last_for_src
                ):
                    frag.steps._codes = empty_i32
                    frag.steps._anchor_cache = None
                if is_last_for_src:
                    frag.provenance._orig_codes = empty_i32

    return _FragmentStage4Result(
        frag_idx=frag_idx,
        analysis=analysis,
        visit_counts_ipc=visit_counts_ipc,
        inv_chains=inv_chains,
        terminal_devs=terminal_devs,
        large_jump_devs=large_jump_devs,
        pending_unanchored_unresolved=pending_unanchored_unresolved,
    )


def _stage4_analyze_fragment_worker(frag_idx: int) -> _FragmentStage4Result:
    ctx = _STAGE4_FORK_CTX
    assert ctx is not None
    return _analyze_single_normal_fragment(frag_idx, ctx, for_ipc=True)


@dataclass(slots=True)
class _Stage5ForkContext:
    graph: GFAGraph
    config: BuildConfig
    ref_steps: Any
    n_ref: int
    top_state_key: SubproblemStateKey
    terminal_devs_by_pair: dict[
        tuple[int, int], list[tuple[_ContigAnalysis, DeviationSegment]]
    ]
    claimed_source_intervals: set[tuple[str, int, int]]
    ref_dedup_aligner: Any | None = None
    ref_dedup_short_aligner: Any | None = None


_STAGE5_FORK_CTX: _Stage5ForkContext | None = None
_STAGE5_WORKER_TBUF_CACHE: tuple[int, int, Any, Any] | None = None


def _screen_and_solve_one_pair(
    pair: tuple[int, int],
    ctx: _Stage5ForkContext,
    *,
    unres_counter_ref: list[int],
    seen_exported_intervals: dict[tuple[str, int, int, str], str],
    ref_dedup_buf: Any | None = None,
    ref_dedup_short_buf: Any | None = None,
) -> tuple[GroupRepeatGuardResult, ParallelGroupSolution | None]:
    """
    Fused pure Stage 2.5a (`extract_candidate_blocks` + `screen_and_fold_candidate`) and
    Stage 2.5b (`solve_parallel_group`) for one active `(rank_a, rank_b)` group.
    Never mutates `ctx.terminal_devs_by_pair` so COW pages stay clean in forked workers.
    """
    rank_a, rank_b = pair
    graph = ctx.graph
    config = ctx.config
    ref_steps = ctx.ref_steps
    n_ref = ctx.n_ref
    top_state_key = ctx.top_state_key
    devs_at_pair = ctx.terminal_devs_by_pair[pair]
    claimed_source_intervals = ctx.claimed_source_intervals

    guard_res = GroupRepeatGuardResult()
    next_src_order = 0
    cand_seq_idx = 1

    for analysis, tdev in devs_at_pair:
        if not tdev.branch_provenance:
            cand_seq_idx += 1
            continue
        rb_run_starts: list[int] = []
        raw_blocks = extract_candidate_blocks(
            dev=tdev,
            backbone_steps=ref_steps,
            nodes=graph.nodes,
            global_ref_nodes=graph.global_ref_nodes,
            min_sv_len=config.min_sv_len,
            is_top_level=True,
            lo=0,
            hi=n_ref,
            already_screened=False,
            cand_id_prefix=(
                f"{graph.ref_chrom}_G_{rank_a}_{rank_b}_{tdev.hap_id}_{cand_seq_idx}"
            ),
            base_source_order=next_src_order,
            claimed_source_intervals=claimed_source_intervals,
            out_run_starts=rb_run_starts,
        )
        cand_seq_idx += 1

        for rb, rb_start_idx in zip(raw_blocks, rb_run_starts):
            c_dedup_stats = _CandidateDedupStats()
            kept_c, unres_c, logs_c, had_mask, next_src_order = (
                screen_and_fold_candidate(
                    raw_cand=rb,
                    graph=graph,
                    config=config,
                    region_id=".",
                    next_source_order=next_src_order,
                    unres_counter_ref=unres_counter_ref,
                    seen_exported_intervals=seen_exported_intervals,
                    node_visit_counts=analysis.node_visit_counts,
                    ref_dedup_aligner=ctx.ref_dedup_aligner,
                    ref_dedup_short_aligner=ctx.ref_dedup_short_aligner,
                    ref_dedup_buf=ref_dedup_buf,
                    ref_dedup_short_buf=ref_dedup_short_buf,
                    dedup_stats_out=c_dedup_stats,
                )
            )
            if had_mask:
                guard_res.hap_had_masked_repeat[rb.hap_id] = True
                guard_res.contig_had_masked_repeat.add(analysis.contig_idx)
            if c_dedup_stats.had_subtracted_homolog:
                guard_res.contig_had_subtracted_homolog.add(analysis.contig_idx)
                guard_res.ref_masked_bp += c_dedup_stats.ref_masked_bp
                guard_res.short_residual_bp += c_dedup_stats.short_residual_bp
            if kept_c:
                guard_res.hap_had_kept_candidate[rb.hap_id] = True
                guard_res.contig_had_kept_candidate.add(analysis.contig_idx)
                if any(kc.partial_content for kc in kept_c):
                    guard_res.has_partial_content = True
                for kc in kept_c:
                    guard_res.cand_dev_loc[kc.candidate_id] = (
                        analysis.contig_idx,
                        rb_start_idx,
                    )
            guard_res.kept_candidates.extend(kept_c)
            guard_res.unresolved_records.extend(unres_c)
            guard_res.json_log_entries.extend(logs_c)

    sol: ParallelGroupSolution | None = None
    if guard_res.kept_candidates:
        cands = guard_res.kept_candidates
        b_slice = ref_steps[rank_a + 1 : rank_b]
        m_keys = tuple(
            sorted({make_provenance_key(c.provenance) for c in cands if c.provenance})
        )
        g_state_key = SubproblemStateKey(
            b_id=(("REF", rank_a + 1, rank_b, "+"),),
            lo=rank_a + 1,
            hi=rank_b,
            has_logical_start=True,
            has_logical_end=True,
            m_key=m_keys,
        )
        sol = solve_parallel_group(
            left_anchor_rank=rank_a,
            right_anchor_rank=rank_b,
            backbone_slice_steps=b_slice,
            candidates=cands,
            graph=graph,
            config=config,
            parent_state_key=g_state_key,
            ancestor_keys=frozenset({top_state_key}),
        )
        sol.backbone_seq = ""
        keep_token_str = config.dedup_mode == "pan"
        for rep in sol.non_backbone_reps:
            rep.aligner = None
            rep.member_occurrences = []
            rep.nested_choices_by_cand_id = {}
            rep.reconstructed_len_by_cand_id = {}
            rep.cand_to_flags = {}
            rep.support_haps = set()
            rep.exemplar = replace(rep.exemplar, steps=(), provenance=(), seq="")
            rep.seq = ""
            if not keep_token_str:
                rep.token_str = ""
        cand_contig_idx = {
            cid: loc[0] for cid, loc in guard_res.cand_dev_loc.items()
        }
        if not sol.non_backbone_reps:
            sol.cand_to_rep_idx = {}
            sol.cand_to_nested_sig = {}
            sol.cand_to_reconstructed_len = {}
            sol.cand_to_flags = {}
            sol.rep0_cand_spans = {}
            guard_res.cand_dev_loc = {}
        else:
            rep0_cids = {
                cid
                for cid, r_idx in sol.cand_to_rep_idx.items()
                if r_idx == 0
            }
            if rep0_cids:
                guard_res.cand_dev_loc = {
                    cid: loc
                    for cid, loc in guard_res.cand_dev_loc.items()
                    if cid in rep0_cids
                }
            else:
                guard_res.cand_dev_loc = {}
        guard_res.kept_candidates = [
            replace(
                kc,
                steps=(),
                provenance=(),
                seq="",
                contig_idx=cand_contig_idx.get(kc.candidate_id, -1),
            )
            for kc in guard_res.kept_candidates
        ]

    return guard_res, sol


def _stage5_screen_and_solve_batch_worker(
    batch_pairs: list[tuple[int, int]],
) -> list[
    tuple[tuple[int, int], GroupRepeatGuardResult, ParallelGroupSolution | None]
]:
    global _STAGE5_WORKER_TBUF_CACHE
    ctx = _STAGE5_FORK_CTX
    assert ctx is not None
    ref_buf: Any | None = None
    ref_short_buf: Any | None = None
    if ctx.ref_dedup_aligner is not None and hasattr(mappy, "ThreadBuffer"):
        pid = os.getpid()
        aln_id = id(ctx.ref_dedup_aligner)
        if (
            _STAGE5_WORKER_TBUF_CACHE is not None
            and _STAGE5_WORKER_TBUF_CACHE[0] == pid
            and _STAGE5_WORKER_TBUF_CACHE[1] == aln_id
        ):
            ref_buf = _STAGE5_WORKER_TBUF_CACHE[2]
            ref_short_buf = _STAGE5_WORKER_TBUF_CACHE[3]
        else:
            ref_buf = mappy.ThreadBuffer()
            ref_short_buf = (
                mappy.ThreadBuffer()
                if ctx.ref_dedup_short_aligner is not None
                else None
            )
            _STAGE5_WORKER_TBUF_CACHE = (pid, aln_id, ref_buf, ref_short_buf)

    local_counter = [1]
    local_seen: dict[tuple[str, int, int, str], str] = {}
    out: list[
        tuple[
            tuple[int, int],
            GroupRepeatGuardResult,
            ParallelGroupSolution | None,
        ]
    ] = []
    for pair in batch_pairs:
        g_res, sol = _screen_and_solve_one_pair(
            pair,
            ctx,
            unres_counter_ref=local_counter,
            seen_exported_intervals=local_seen,
            ref_dedup_buf=ref_buf,
            ref_dedup_short_buf=ref_short_buf,
        )
        out.append((pair, g_res, sol))
    return out


def _contig_covers_anchor_pair(
    analysis: _ContigAnalysis,
    rank_a: int,
    rank_b: int,
) -> bool:
    """Return True iff `analysis` covers both `rank_a` and `rank_b` in its forward chain without an unnormalized/large-jump break."""
    if rank_a not in analysis.covered_ranks or rank_b not in analysis.covered_ranks:
        return False
    lo_r, hi_r = min(rank_a, rank_b), max(rank_a, rank_b)
    for u_lo, u_hi in analysis.unnormalized_rank_spans:
        if max(lo_r, u_lo) <= min(hi_r, u_hi):
            return False
    for g_lo, g_hi in analysis.unnormalized_gap_spans:
        if lo_r <= g_lo and g_hi <= hi_r:
            return False
    for lj_lo, lj_hi in analysis.large_jump_rank_spans:
        if lo_r <= lj_lo and lj_hi <= hi_r:
            return False
    return True


def _contig_covers_rank_interval(
    analysis: _ContigAnalysis,
    rmin: int,
    rmax: int,
    n_ref: int,
) -> bool:
    """Return True iff `analysis`'s forward main chain strictly flanks `[rmin, rmax]` with no overlapping inversion or large-jump break."""
    if not analysis.has_main_chain or not analysis.covered_ranks:
        return False
    if rmin <= 0 and rmax >= n_ref - 1:
        return False
    for u_lo, u_hi in analysis.unnormalized_rank_spans:
        if max(rmin, u_lo) <= min(rmax, u_hi):
            return False
    for g_lo, g_hi in analysis.unnormalized_gap_spans:
        if rmin <= g_lo and g_hi <= rmax:
            return False
    for n_lo, n_hi in analysis.normalized_inv_rank_spans:
        if max(rmin, n_lo) <= min(rmax, n_hi):
            return False
    for lj_lo, lj_hi in analysis.large_jump_rank_spans:
        if max(rmin, lj_lo) <= min(rmax, lj_hi):
            return False

    if isinstance(analysis.covered_ranks, CompactRankSet):
        min_cov = analysis.covered_ranks._min_val
        max_cov = analysis.covered_ranks._max_val
    else:
        min_cov = min(analysis.covered_ranks)
        max_cov = max(analysis.covered_ranks)

    left_ok = (min_cov < rmin) if rmin > 0 else (0 in analysis.covered_ranks)
    right_ok = (
        (max_cov > rmax)
        if rmax < n_ref - 1
        else ((n_ref - 1) in analysis.covered_ranks)
    )
    return left_ok and right_ok


def _branch_steps_bp(
    branch_steps: Sequence[Step],
    graph: GFAGraph,
) -> int:
    """Compute total base length of `branch_steps` without depending on `branch_provenance`."""
    if not branch_steps:
        return 0
    if (
        isinstance(branch_steps, LazyStepTuple)
        and graph._node_lens is not None
    ):
        br_nids = np.abs(branch_steps._codes)
        if int(br_nids.max()) < int(graph._node_lens.shape[0]):
            return int(graph._node_lens[br_nids].sum(dtype=np.int64))
    return sum(_step_bp_len(st, graph.nodes) for st in branch_steps)


def _resolve_haplotype_genotypes(
    all_haplotypes: Sequence[str],
    ref_hap_id: str,
    ref_default: str,
    contig_calls_by_hap: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """
    §5.2: Combine per-contig genotype calls for each haplotype.
    If multiple contigs of the same haplotype give contradictory non-`.` calls, emit `'.'`.
    """
    genotypes: dict[str, str] = {}
    for hap_id in all_haplotypes:
        if hap_id == ref_hap_id:
            genotypes[hap_id] = ref_default
            continue
        calls = [c for c in contig_calls_by_hap.get(hap_id, ()) if c != "."]
        if not calls:
            genotypes[hap_id] = "."
        elif len(set(calls)) == 1:
            genotypes[hap_id] = calls[0]
        else:
            genotypes[hap_id] = "."
    return genotypes


# ============================================================================
# DEDUP_PLAN Module C & D: Stage 5.5 Cross-Group Deterministic Pan Deduplication
# ============================================================================

_PAN_DEDUP_WAVE_BUDGET_BP: int = 1_500_000
_PAN_DEDUP_SUBWAVE_BP: int = 384_000
_PAN_DEDUP_SUBWAVE_MAX_PIECES: int = 256
_PAN_DEDUP_QUERY_BATCH_BP: int = 2_000_000
_PAN_DEDUP_WAVE_SEP: str = "N" * 200
_PAN_DEDUP_SEED_K: int = 13
_PAN_DEDUP_LONG_SEED_K: int = 19
_PAN_DEDUP_BLOOM_BITS: int = 24  # 16 MiB lazy byte-Bloom; freed at Stage 5.5 exit
_PAN_DEDUP_BLOOM_SHIFT = np.uint64(64 - _PAN_DEDUP_BLOOM_BITS)
_PAN_DEDUP_BLOOM_M1 = np.uint64(0x9E3779B97F4A7C15)
_PAN_DEDUP_BLOOM_M2 = np.uint64(0xBF58476D1CE4E5B9)
_PAN_DEDUP_BLOOM_M3 = np.uint64(0x94D049BB133111EB)
_PAN_DEDUP_BLOOM_M4 = np.uint64(0xD6E8FEB86659FD93)

_PAN_DEDUP_LUT_FWD_U32 = np.full(256, 4, dtype=np.uint32)
for _ch_u, _ch_l, _code in (
    ("A", "a", 0),
    ("C", "c", 1),
    ("G", "g", 2),
    ("T", "t", 3),
):
    _PAN_DEDUP_LUT_FWD_U32[ord(_ch_u)] = _code
    _PAN_DEDUP_LUT_FWD_U32[ord(_ch_l)] = _code

_PAN_DEDUP_LUT_RC_U32 = np.full(256, 4, dtype=np.uint32)
for _ch_u, _ch_l, _code in (
    ("A", "a", 3),
    ("C", "c", 2),
    ("G", "g", 1),
    ("T", "t", 0),
):
    _PAN_DEDUP_LUT_RC_U32[ord(_ch_u)] = _code
    _PAN_DEDUP_LUT_RC_U32[ord(_ch_l)] = _code


def _extract_canon_19mers(seq: str) -> tuple[np.ndarray | None, np.ndarray]:
    """Return `(valid_positions_or_None, canonical_19mer_uint64)` for all non-N 19-mers in `seq`.

    When `valid_positions_or_None` is `None`, all `len(seq) - 18` positions `0..n-19` are valid
    so array index equals sequence offset directly.
    """
    n = len(seq)
    if n < _PAN_DEDUP_LONG_SEED_K:
        return None, np.empty(0, dtype=np.uint64)
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    f1 = _PAN_DEDUP_LUT_FWD_U32[raw]
    f2 = (f1[:-1] << np.uint32(2)) | f1[1:]
    f4 = (f2[:-2] << np.uint32(4)) | f2[2:]
    f8 = (f4[:-4] << np.uint32(8)) | f4[4:]
    f16 = (f8[:-8] << np.uint32(16)) | f8[8:]
    f19 = (
        (f16[:-3].astype(np.uint64) << np.uint64(6))
        | (f2[16:-1].astype(np.uint64) << np.uint64(2))
        | f1[18:].astype(np.uint64)
    )

    r1 = _PAN_DEDUP_LUT_RC_U32[raw[::-1]]
    r2 = (r1[:-1] << np.uint32(2)) | r1[1:]
    r4 = (r2[:-2] << np.uint32(4)) | r2[2:]
    r8 = (r4[:-4] << np.uint32(8)) | r4[4:]
    r16 = (r8[:-8] << np.uint32(16)) | r8[8:]
    r19 = (
        (r16[:-3].astype(np.uint64) << np.uint64(6))
        | (r2[16:-1].astype(np.uint64) << np.uint64(2))
        | r1[18:].astype(np.uint64)
    )[::-1]

    canon = np.minimum(f19, r19)
    bad1 = f1 >= np.uint32(4)
    if np.any(bad1):
        b2 = bad1[:-1] | bad1[1:]
        b4 = b2[:-2] | b2[2:]
        b8 = b4[:-4] | b4[4:]
        b16 = b8[:-8] | b8[8:]
        b19 = b16[:-3] | b2[16:-1] | bad1[18:]
        valid = ~b19
        return np.flatnonzero(valid).astype(np.int32), canon[valid]
    return None, canon


class _TailSeedGate:
    """
    Dual-regime mathematically zero-false-negative seed gate for Stage 5.5 `tail_pieces`:
    - `len(sub_seq) > 500`: primary pass runs only `asm20` (`k=19, w=10, min_cnt=3, min_chain_score=40`);
      uses a lazy byte-Bloom over all stride-1 canonical non-N 19-mers and requires >=3 hits
      spanning >=21 bp (`19 + 21 = 40`).
    - `len(sub_seq) <= 500`: primary pass runs `short_aligner` (`k=13, w=5, min_cnt=2, min_chain_score=20,
      bw=500`) plus `asm20`; indexes forward non-N 13-mers with target coordinate (`-1` if multi-copy)
      and requires >=2 same-strand hits with `i - pi >= 7` (`13 + 7 = 20`) and diagonal consistency
      `r > pr and abs((r - pr) - (i - pi)) <= 600` (or multi-copy `-1`).
    """

    __slots__ = (
        "_bloom",
        "_idx_19",
        "_tail_13",
        "_idx_13",
        "_off_13",
        "_last_19_seq",
        "_last_19_arr",
    )

    def __init__(self) -> None:
        self._bloom: np.ndarray | None = None
        self._idx_19: int = 0
        self._tail_13: dict[str, int] = {}
        self._idx_13: int = 0
        self._off_13: int = 0
        self._last_19_seq: str | None = None
        self._last_19_arr: np.ndarray | None = None

    def clear(self) -> None:
        if self._bloom is not None and self._idx_19 > 0:
            self._bloom.fill(0)
        self._idx_19 = 0
        self._tail_13.clear()
        self._idx_13 = 0
        self._off_13 = 0
        self._last_19_seq = None
        self._last_19_arr = None

    def release(self) -> None:
        self._bloom = None
        self._idx_19 = 0
        self._tail_13.clear()
        self._idx_13 = 0
        self._off_13 = 0
        self._last_19_seq = None
        self._last_19_arr = None

    def _ensure_19mers(self, tail_pieces: Sequence[str]) -> None:
        n_tail = len(tail_pieces)
        if self._idx_19 >= n_tail:
            return
        if self._bloom is None:
            self._bloom = np.zeros(1 << _PAN_DEDUP_BLOOM_BITS, dtype=np.uint8)
        bloom = self._bloom
        sh = _PAN_DEDUP_BLOOM_SHIFT
        m1 = _PAN_DEDUP_BLOOM_M1
        m2 = _PAN_DEDUP_BLOOM_M2
        m3 = _PAN_DEDUP_BLOOM_M3
        m4 = _PAN_DEDUP_BLOOM_M4
        while self._idx_19 < n_tail:
            s = tail_pieces[self._idx_19]
            self._idx_19 += 1
            if s is self._last_19_seq and self._last_19_arr is not None:
                arr = self._last_19_arr
            else:
                _, arr = _extract_canon_19mers(s)
            if arr.size:
                bloom[(arr * m1) >> sh] = 1
                bloom[(arr * m2) >> sh] = 1
                bloom[(arr * m3) >> sh] = 1
                bloom[(arr * m4) >> sh] = 1

    def _ensure_13mers(self, tail_pieces: Sequence[str]) -> None:
        n_tail = len(tail_pieces)
        if self._idx_13 >= n_tail:
            return
        sep_len = len(_PAN_DEDUP_WAVE_SEP)
        t13 = self._tail_13
        k = _PAN_DEDUP_SEED_K
        while self._idx_13 < n_tail:
            s = tail_pieces[self._idx_13]
            off = self._off_13 + (sep_len if self._idx_13 > 0 else 0)
            self._idx_13 += 1
            if len(s) >= k:
                has_n = "N" in s
                end = len(s) - k + 1
                for i in range(end):
                    km = s[i : i + k]
                    if has_n and "N" in km:
                        continue
                    t13[km] = -1 if km in t13 else (off + i)
            self._off_13 = off + len(s)

    def has_seed_hit(self, sub_seq: str, tail_pieces: Sequence[str]) -> bool:
        sub_len = len(sub_seq)
        if not tail_pieces or sub_len < _PAN_DEDUP_SEED_K:
            return False
        if sub_len > 500:
            self._ensure_19mers(tail_pieces)
            bloom = self._bloom
            if bloom is None:
                return False
            q_pos, q_arr = _extract_canon_19mers(sub_seq)
            self._last_19_seq = sub_seq
            self._last_19_arr = q_arr
            if not q_arr.size:
                return False
            sh = _PAN_DEDUP_BLOOM_SHIFT
            m = bloom[(q_arr * _PAN_DEDUP_BLOOM_M1) >> sh] != 0
            if not np.any(m):
                return False
            idx = np.flatnonzero(m)
            ks = q_arr[idx]
            for mult in (
                _PAN_DEDUP_BLOOM_M2,
                _PAN_DEDUP_BLOOM_M3,
                _PAN_DEDUP_BLOOM_M4,
            ):
                keep = bloom[(ks * mult) >> sh] != 0
                if not np.any(keep):
                    return False
                idx = idx[keep]
                ks = ks[keep]
            if idx.size < 3:
                return False
            span = (
                int(q_pos[idx[-1]] - q_pos[idx[0]])
                if q_pos is not None
                else int(idx[-1] - idx[0])
            )
            return span >= 21
        self._ensure_13mers(tail_pieces)
        t13 = self._tail_13
        if not t13:
            return False
        t13_get = t13.get
        k = _PAN_DEDUP_SEED_K
        end = sub_len - k + 1
        for qs in (sub_seq, revcomp(sub_seq)):
            hits: list[tuple[int, int]] = []
            for i in range(end):
                r = t13_get(qs[i : i + k])
                if r is not None:
                    for pi, pr in hits:
                        if i - pi >= 7 and (
                            r == -1
                            or pr == -1
                            or (r > pr and abs((r - pr) - (i - pi)) <= 600)
                        ):
                            return True
                    hits.append((i, r))
        return False


@dataclass(slots=True)
class _WaveIndex:
    """Read-only concatenated mappy index (asm20 + lazy thread-safe short k=13,w=5) for a pan-dedup wave."""

    aligner_main: Any
    wave_seq: str
    t_starts: tuple[int, ...]
    t_ends: tuple[int, ...]
    total_bp: int
    _aligner_short: Any = None
    _short_lock: Any = field(default_factory=threading.Lock)

    def get_short_aligner(self) -> Any:
        saln = self._aligner_short
        if saln is not None:
            return saln
        with self._short_lock:
            saln = self._aligner_short
            if saln is None:
                saln = mappy.Aligner(
                    seq=self.wave_seq,
                    k=13,
                    w=5,
                    min_cnt=2,
                    min_chain_score=20,
                    best_n=50,
                )
                self._aligner_short = saln
                self.wave_seq = ""
            return saln


@dataclass(slots=True)
class _PendingRepState:
    """Per-representative surviving interval state tracked across Stage 5.5 batches."""

    pair: tuple[int, int]
    rep: ClusterRepresentative
    piece_seqs_up: tuple[str, ...]
    piece_ivs: list[list[tuple[int, int]]]
    screened_sealed_waves: int = 0


def _build_pan_dedup_wave_index(piece_seqs: Sequence[str]) -> _WaveIndex:
    """Concatenate accepted piece sequences separated by `'N' * 200` and build `asm20` index (short index built lazily)."""
    sep = _PAN_DEDUP_WAVE_SEP
    sep_len = len(sep)
    parts: list[str] = []
    t_starts: list[int] = []
    t_ends: list[int] = []
    cursor = 0
    total_bp = 0
    for raw_s in piece_seqs:
        s = raw_s.upper()
        if not s:
            continue
        if parts:
            parts.append(sep)
            cursor += sep_len
        t_starts.append(cursor)
        cursor += len(s)
        t_ends.append(cursor)
        parts.append(s)
        total_bp += len(s)
    wave_seq = "".join(parts)
    aligner_main = mappy.Aligner(seq=wave_seq, preset="asm20", best_n=50)
    return _WaveIndex(
        aligner_main=aligner_main,
        wave_seq=wave_seq,
        t_starts=tuple(t_starts),
        t_ends=tuple(t_ends),
        total_bp=total_bp,
    )


def _subslice_piece(
    p: Piece,
    iv_s: int,
    iv_e: int,
    join_before: Literal["source_contiguous", "spacer", "synthetic"],
) -> Piece:
    """Safely sub-slice `Piece` `p` to half-open query coordinates `[iv_s, iv_e)` honoring strand and Global Invariant II."""
    if p.src_strand == "+":
        new_start = p.src_start + iv_s
        new_end = p.src_start + iv_e
    else:
        new_start = p.src_end - iv_e
        new_end = p.src_end - iv_s
    return replace(
        p,
        src_start=new_start,
        src_end=new_end,
        seq=p.seq[iv_s:iv_e],
        join_before=join_before,
    )


def _subtract_intervals_against_wave(
    seq_up: str,
    cur_ivs: Sequence[tuple[int, int]],
    wave: _WaveIndex,
    min_sv_len: int,
    dedup_id: float,
    buf_main: Any | None,
    buf_short: Any | None,
    collect_short_slices: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Subtract homologous 50 bp blocks from `cur_ivs` (coordinates on `seq_up`) against
    a single `_WaveIndex`. Untouched intervals (including `< min_sv_len` pieces) remain
    intact; masked intervals keep residuals `>= min_sv_len` and count `< min_sv_len`
    residuals in `short_res_bp`.
    """
    next_ivs: list[tuple[int, int]] = []
    masked_bp = 0
    short_res_bp = 0
    for iv_s, iv_e in cur_ivs:
        if iv_e - iv_s < min_sv_len:
            next_ivs.append((iv_s, iv_e))
            continue
        sub_seq = seq_up[iv_s:iv_e]
        sub_short_slices: list[tuple[int, int]] | None = (
            [] if (collect_short_slices is not None and (iv_e - iv_s) > 500) else None
        )
        masked_sub, unmasked_sub, b_masked = subtract_reference_intervals_mappy(
            query_seq=sub_seq,
            ref_region_seq="",
            min_sv_len=min_sv_len,
            aligner=wave.aligner_main,
            buf=buf_main,
            min_identity=dedup_id,
            short_aligner=wave.get_short_aligner,
            short_buf=buf_short,
            t_starts=wave.t_starts,
            t_ends=wave.t_ends,
            collect_short_slices=sub_short_slices,
        )
        if sub_short_slices:
            assert collect_short_slices is not None
            for ss, se in sub_short_slices:
                collect_short_slices.append((iv_s + ss, iv_s + se))
        if masked_sub:
            masked_bp += b_masked
            for u_s, u_e in unmasked_sub:
                if u_e - u_s >= min_sv_len:
                    next_ivs.append((iv_s + u_s, iv_s + u_e))
                else:
                    short_res_bp += u_e - u_s
        else:
            next_ivs.append((iv_s, iv_e))
    return next_ivs, masked_bp, short_res_bp


def _deduplicate_representatives_across_chromosome(
    graph: GFAGraph,
    config: BuildConfig,
    sorted_group_pairs: Sequence[tuple[int, int]],
    group_guard_by_pair: Mapping[tuple[int, int], GroupRepeatGuardResult],
    group_solutions: Mapping[tuple[int, int], ParallelGroupSolution],
    cen_out: Any,
) -> tuple[int, int, int]:
    """
    DEDUP_PLAN Modules C & D (Stage 5.5, `--dedup-mode pan`):
    Execute deterministic global-priority cross-group/cross-representative homologous
    subtraction strictly AFTER all `fork` pools have exited (`MULTICORE_PLAN.md` §2.3),
    then compact/reindex surviving reps and update per-allele flags & lengths.

    Returns `(pan_masked_bp, pan_short_residual_bp, module_c_dropped_groups)`.
    """
    min_sv_len = config.min_sv_len
    dedup_id = config.dedup_id
    has_tbuf = hasattr(mappy, "ThreadBuffer")

    # Step 1: Priority-0 seeding from CEN_ISLAND pieces
    accepted_exact_seqs: set[str] = set()
    sealed_waves: list[_WaveIndex] = []
    cen_chunk: list[str] = []
    cen_chunk_bp = 0
    for c_blk in cen_out.island_slot_blocks:
        for rep_pieces in c_blk.representatives:
            for p in rep_pieces:
                s_up = p.seq.upper()
                if not s_up:
                    continue
                if len(s_up) >= min_sv_len:
                    accepted_exact_seqs.add(s_up)
                    accepted_exact_seqs.add(revcomp(s_up))
                    cen_chunk.append(s_up)
                    cen_chunk_bp += len(s_up)
                    if cen_chunk_bp >= _PAN_DEDUP_WAVE_BUDGET_BP:
                        sealed_waves.append(_build_pan_dedup_wave_index(cen_chunk))
                        cen_chunk = []
                        cen_chunk_bp = 0
    if cen_chunk:
        sealed_waves.append(_build_pan_dedup_wave_index(cen_chunk))
        cen_chunk = []

    # Step 2: Global deterministic priority queue (T22 total order, input-order invariant)
    entries: list[
        tuple[
            tuple[int, int, int, int, int, int, str],
            tuple[int, int],
            ClusterRepresentative,
        ]
    ] = []
    for pair in sorted_group_pairs:
        sol = group_solutions.get(pair)
        if sol is None or not sol.non_backbone_reps:
            continue
        rank_a, rank_b = pair
        ref_s_open = graph.ref_step_ranges[rank_a][1]
        ref_e_open = graph.ref_step_ranges[rank_b][0]
        span_bp = ref_e_open - ref_s_open
        for rep in sol.non_backbone_reps:
            priority_key = (
                -rep.support_hap_count,
                span_bp,
                -rep.length_bp_no_n,
                rank_a,
                rank_b,
                rep.rep_idx,
                rep.token_str,
            )
            entries.append((priority_key, pair, rep))

    if not entries:
        sealed_waves.clear()
        accepted_exact_seqs.clear()
        return 0, 0, 0

    entries.sort(key=lambda item: item[0])

    # Step 3: Wave-buffered parallel pre-screen + priority-ordered sequential active subtraction
    pan_masked_bp = 0
    pan_short_residual_bp = 0

    active_all_pieces: list[str] = []
    active_bp = 0
    active_subwaves: list[_WaveIndex] = []
    tail_pieces: list[str] = []
    tail_bp = 0
    tail_gate = _TailSeedGate()
    tail_delta_pieces: list[str] = []
    tail_delta_gate = _TailSeedGate()
    tail_wave: _WaveIndex | None = None
    tail_dirty = False

    rep_results: dict[
        tuple[tuple[int, int], int],
        tuple[list[Piece], list[frozenset[str]], bool, bool, set[str] | None],
    ] = {}

    seq_buf_main = mappy.ThreadBuffer() if has_tbuf else None
    seq_buf_short = mappy.ThreadBuffer() if has_tbuf else None

    def _screen_ivs_against_active(
        seq_up: str,
        ivs: list[tuple[int, int]],
        start_sub_idx: int,
    ) -> list[tuple[int, int]]:
        nonlocal pan_masked_bp, pan_short_residual_bp, tail_wave, tail_dirty
        if ivs and accepted_exact_seqs:
            kept_ex: list[tuple[int, int]] = []
            for iv_s, iv_e in ivs:
                if seq_up[iv_s:iv_e] in accepted_exact_seqs:
                    pan_masked_bp += iv_e - iv_s
                else:
                    kept_ex.append((iv_s, iv_e))
            ivs = kept_ex
        for sw_idx in range(start_sub_idx, len(active_subwaves)):
            if not ivs:
                return ivs
            ivs, dm, dsr = _subtract_intervals_against_wave(
                seq_up,
                ivs,
                active_subwaves[sw_idx],
                min_sv_len,
                dedup_id,
                seq_buf_main,
                seq_buf_short,
            )
            pan_masked_bp += dm
            pan_short_residual_bp += dsr
        if ivs and tail_pieces:
            if any(
                (iv_e - iv_s) >= min_sv_len
                and tail_gate.has_seed_hit(seq_up[iv_s:iv_e], tail_pieces)
                for iv_s, iv_e in ivs
            ):
                if tail_wave is None or not tail_dirty:
                    if tail_wave is None:
                        tail_wave = _build_pan_dedup_wave_index(tail_pieces)
                        tail_delta_pieces.clear()
                        tail_delta_gate.clear()
                        tail_dirty = False
                    ivs, dm, dsr = _subtract_intervals_against_wave(
                        seq_up,
                        ivs,
                        tail_wave,
                        min_sv_len,
                        dedup_id,
                        seq_buf_main,
                        seq_buf_short,
                    )
                    pan_masked_bp += dm
                    pan_short_residual_bp += dsr
                else:
                    ge_bp = sum(
                        iv_e - iv_s
                        for iv_s, iv_e in ivs
                        if iv_e - iv_s >= min_sv_len
                    )
                    short_slices_old: list[tuple[int, int]] = []
                    ivs_old, dm_old, dsr_old = _subtract_intervals_against_wave(
                        seq_up,
                        ivs,
                        tail_wave,
                        min_sv_len,
                        dedup_id,
                        seq_buf_main,
                        seq_buf_short,
                        collect_short_slices=short_slices_old,
                    )
                    if dm_old == ge_bp and dsr_old == 0:
                        ivs = ivs_old
                        pan_masked_bp += dm_old
                    else:
                        need_rebuild = any(
                            (iv_e - iv_s) >= min_sv_len
                            and tail_delta_gate.has_seed_hit(
                                seq_up[iv_s:iv_e], tail_delta_pieces
                            )
                            for iv_s, iv_e in ivs
                        )
                        if not need_rebuild and short_slices_old:
                            for u_s, u_e in short_slices_old:
                                if tail_delta_gate.has_seed_hit(
                                    seq_up[u_s:u_e], tail_delta_pieces
                                ):
                                    need_rebuild = True
                                    break
                        if need_rebuild:
                            tail_wave = _build_pan_dedup_wave_index(tail_pieces)
                            tail_delta_pieces.clear()
                            tail_delta_gate.clear()
                            tail_dirty = False
                            ivs, dm, dsr = _subtract_intervals_against_wave(
                                seq_up,
                                ivs,
                                tail_wave,
                                min_sv_len,
                                dedup_id,
                                seq_buf_main,
                                seq_buf_short,
                            )
                            pan_masked_bp += dm
                            pan_short_residual_bp += dsr
                        else:
                            ivs = ivs_old
                            pan_masked_bp += dm_old
                            pan_short_residual_bp += dsr_old
        return ivs

    def _accept_piece_seq(sp_up: str) -> None:
        nonlocal active_bp, tail_bp, tail_wave, tail_dirty
        if len(sp_up) < min_sv_len:
            return
        accepted_exact_seqs.add(sp_up)
        accepted_exact_seqs.add(revcomp(sp_up))
        active_all_pieces.append(sp_up)
        active_bp += len(sp_up)
        tail_pieces.append(sp_up)
        tail_bp += len(sp_up)
        if tail_wave is not None:
            tail_delta_pieces.append(sp_up)
        tail_dirty = True
        if (
            tail_bp >= _PAN_DEDUP_SUBWAVE_BP
            or len(tail_pieces) >= _PAN_DEDUP_SUBWAVE_MAX_PIECES
        ) and active_bp < _PAN_DEDUP_WAVE_BUDGET_BP:
            active_subwaves.append(_build_pan_dedup_wave_index(tail_pieces))
            tail_pieces.clear()
            tail_bp = 0
            tail_gate.clear()
            tail_delta_pieces.clear()
            tail_delta_gate.clear()
            tail_wave = None
            tail_dirty = False

    queue_pos = 0
    n_entries = len(entries)
    carryover: list[_PendingRepState] = []

    while carryover or queue_pos < n_entries:
        batch: list[_PendingRepState] = []
        batch_bp = 0
        if carryover:
            for c_item in carryover:
                batch.append(c_item)
                batch_bp += sum(e - s for ivs in c_item.piece_ivs for s, e in ivs)
            carryover = []
        while queue_pos < n_entries and (
            not batch or batch_bp < _PAN_DEDUP_QUERY_BATCH_BP
        ):
            _, pair, rep = entries[queue_pos]
            queue_pos += 1
            p_seqs_up = tuple(p.seq.upper() for p in rep.pieces)
            p_ivs = [[(0, len(s))] if s else [] for s in p_seqs_up]
            batch.append(
                _PendingRepState(
                    pair=pair,
                    rep=rep,
                    piece_seqs_up=p_seqs_up,
                    piece_ivs=p_ivs,
                    screened_sealed_waves=0,
                )
            )
            batch_bp += sum(len(s) for s in p_seqs_up)

        n_sealed = len(sealed_waves)
        n_sub = len(active_subwaves)

        # Step 2.1: Parallel pre-screen against immutable sealed_waves + active_subwaves
        work_items: list[
            tuple[int, int, str, list[tuple[int, int]], int, int]
        ] = []
        for batch_i, item in enumerate(batch):
            start_sealed = item.screened_sealed_waves
            has_waves = (start_sealed < n_sealed) or (n_sub > 0)
            for orig_p_idx, seq_up in enumerate(item.piece_seqs_up):
                cur_ivs = item.piece_ivs[orig_p_idx]
                if not cur_ivs or len(seq_up) < min_sv_len:
                    continue
                if accepted_exact_seqs:
                    kept_ex: list[tuple[int, int]] = []
                    for iv_s, iv_e in cur_ivs:
                        if seq_up[iv_s:iv_e] in accepted_exact_seqs:
                            pan_masked_bp += iv_e - iv_s
                        else:
                            kept_ex.append((iv_s, iv_e))
                    cur_ivs = kept_ex
                    item.piece_ivs[orig_p_idx] = cur_ivs
                if cur_ivs and has_waves:
                    q_bp = sum(e - s for s, e in cur_ivs)
                    work_items.append(
                        (batch_i, orig_p_idx, seq_up, cur_ivs, start_sealed, q_bp)
                    )

        n_work = len(work_items)
        if n_work > 0:
            slot_out: list[tuple[list[tuple[int, int]], int, int]] = [
                ([], 0, 0) for _ in range(n_work)
            ]

            def _run_work_item(
                w_idx: int,
                buf_m: Any | None,
                buf_s: Any | None,
            ) -> tuple[list[tuple[int, int]], int, int]:
                _, _, w_seq_up, w_ivs, w_start_sealed, _ = work_items[w_idx]
                ivs = list(w_ivs)
                m_bp = 0
                sr_bp = 0
                for sw_i in range(w_start_sealed, n_sealed):
                    if not ivs:
                        break
                    ivs, dm, dsr = _subtract_intervals_against_wave(
                        w_seq_up,
                        ivs,
                        sealed_waves[sw_i],
                        min_sv_len,
                        dedup_id,
                        buf_m,
                        buf_s,
                    )
                    m_bp += dm
                    sr_bp += dsr
                for sub_i in range(n_sub):
                    if not ivs:
                        break
                    ivs, dm, dsr = _subtract_intervals_against_wave(
                        w_seq_up,
                        ivs,
                        active_subwaves[sub_i],
                        min_sv_len,
                        dedup_id,
                        buf_m,
                        buf_s,
                    )
                    m_bp += dm
                    sr_bp += dsr
                return ivs, m_bp, sr_bp

            workers = min(max(1, config.threads), n_work) if has_tbuf else 1
            if workers <= 1:
                for w_idx in range(n_work):
                    slot_out[w_idx] = _run_work_item(
                        w_idx, seq_buf_main, seq_buf_short
                    )
            else:
                ordered_w = sorted(
                    range(n_work),
                    key=lambda idx: (-work_items[idx][5], idx),
                )
                cursor_w = 0
                cursor_lock = threading.Lock()

                def _thread_worker() -> None:
                    nonlocal cursor_w
                    t_buf_main = mappy.ThreadBuffer()
                    t_buf_short = mappy.ThreadBuffer()
                    while True:
                        with cursor_lock:
                            start = cursor_w
                            if start >= n_work:
                                return
                            if work_items[ordered_w[start]][5] >= 2000:
                                step = 1
                            else:
                                step = min(
                                    8, max(1, (n_work - start) // (workers * 4))
                                )
                            end = min(n_work, start + step)
                            cursor_w = end
                        for pos in range(start, end):
                            w_i = ordered_w[pos]
                            slot_out[w_i] = _run_work_item(
                                w_i, t_buf_main, t_buf_short
                            )

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = [pool.submit(_thread_worker) for _ in range(workers)]
                    for fut in futs:
                        fut.result()

            for w_idx, (batch_i, orig_p_idx, _, _, _, _) in enumerate(work_items):
                ivs_res, m_bp, sr_bp = slot_out[w_idx]
                batch[batch_i].piece_ivs[orig_p_idx] = ivs_res
                pan_masked_bp += m_bp
                pan_short_residual_bp += sr_bp

        # Step 2.2: Strict priority-ordered sequential active-buffer subtraction & acceptance
        for batch_i, item in enumerate(batch):
            new_pieces: list[Piece] = []
            new_piece_cand_ids: list[frozenset[str]] = []
            pieces_changed = False
            seq_trimmed = False
            modified_piece_cands: set[str] | None = set()
            gap_since_last_kept = False

            orig_pieces = item.rep.pieces
            orig_pc_list = item.rep.piece_cand_ids
            for orig_p_idx, p in enumerate(orig_pieces):
                seq_up = item.piece_seqs_up[orig_p_idx]
                orig_pc = (
                    orig_pc_list[orig_p_idx]
                    if orig_p_idx < len(orig_pc_list)
                    else frozenset()
                )
                if len(seq_up) < min_sv_len:
                    surv_ivs = item.piece_ivs[orig_p_idx]
                else:
                    surv_ivs = _screen_ivs_against_active(
                        seq_up, item.piece_ivs[orig_p_idx], n_sub
                    )

                emitted_sub_for_piece = 0
                last_end_for_piece = -1
                sub_idx_before_p = len(active_subwaves)
                iv_queue: collections.deque[tuple[tuple[int, int], bool]] = (
                    collections.deque(
                        (iv, i > 0) for i, iv in enumerate(surv_ivs)
                    )
                )
                while iv_queue:
                    raw_iv, need_rescreen = iv_queue.popleft()
                    if need_rescreen:
                        sub_final_ivs = _screen_ivs_against_active(
                            seq_up, [raw_iv], sub_idx_before_p
                        )
                        if not sub_final_ivs:
                            continue
                        iv_s, iv_e = sub_final_ivs[0]
                        if len(sub_final_ivs) > 1:
                            for rem_iv in reversed(sub_final_ivs[1:]):
                                iv_queue.appendleft((rem_iv, True))
                    else:
                        iv_s, iv_e = raw_iv

                    is_full_orig = iv_s == 0 and iv_e == len(p.seq)
                    if not new_pieces:
                        if (
                            orig_p_idx == 0
                            and iv_s == 0
                            and p.join_before != "spacer"
                        ):
                            jb: Literal[
                                "source_contiguous", "spacer", "synthetic"
                            ] = p.join_before
                        else:
                            jb = "synthetic"
                    else:
                        if (
                            emitted_sub_for_piece > 0
                            or gap_since_last_kept
                            or iv_s > 0
                        ):
                            jb = "spacer"
                        else:
                            jb = p.join_before
                    if is_full_orig and jb == p.join_before:
                        sp = p
                    else:
                        sp = _subslice_piece(p, iv_s, iv_e, jb)
                        pieces_changed = True
                    new_pieces.append(sp)
                    new_piece_cand_ids.append(orig_pc)
                    _accept_piece_seq(seq_up if is_full_orig else sp.seq.upper())
                    emitted_sub_for_piece += 1
                    last_end_for_piece = iv_e

                piece_seq_modified = (
                    emitted_sub_for_piece != 1
                    or last_end_for_piece != len(p.seq)
                    or len(new_pieces[-1].seq) != len(p.seq)
                )
                if piece_seq_modified:
                    pieces_changed = True
                    seq_trimmed = True
                    if not orig_pc:
                        modified_piece_cands = None
                    elif modified_piece_cands is not None:
                        modified_piece_cands.update(orig_pc)

                if emitted_sub_for_piece == 0:
                    if new_pieces:
                        gap_since_last_kept = True
                else:
                    gap_since_last_kept = last_end_for_piece < len(p.seq)

            rep_results[(item.pair, item.rep.rep_idx)] = (
                new_pieces,
                new_piece_cand_ids,
                pieces_changed,
                seq_trimmed,
                modified_piece_cands,
            )

            # Mid-batch wave seal: seal active_all_pieces and carry remaining batch items
            if active_bp >= _PAN_DEDUP_WAVE_BUDGET_BP:
                sealed_waves.append(_build_pan_dedup_wave_index(active_all_pieces))
                active_all_pieces.clear()
                active_bp = 0
                active_subwaves.clear()
                tail_pieces.clear()
                tail_bp = 0
                tail_gate.clear()
                tail_delta_pieces.clear()
                tail_delta_gate.clear()
                tail_wave = None
                tail_dirty = False
                if batch_i + 1 < len(batch):
                    for rem_item in batch[batch_i + 1 :]:
                        rem_item.screened_sealed_waves = n_sealed
                        carryover.append(rem_item)
                break

    # Step 4 (Module D): Compact surviving reps densely, remap cand_to_rep_idx / flags / lengths
    module_c_dropped_groups = 0
    for pair in sorted_group_pairs:
        sol = group_solutions.get(pair)
        if sol is None or not sol.non_backbone_reps:
            continue
        guard_res = group_guard_by_pair[pair]
        kept_reps: list[ClusterRepresentative] = []
        old_to_new_rep_idx: dict[int, int] = {}
        trimmed_old_rep_mod_cands: dict[int, set[str] | None] = {}
        for rep in sol.non_backbone_reps:
            old_r = rep.rep_idx
            rep.token_str = ""
            (
                new_pieces,
                new_piece_cand_ids,
                pieces_changed,
                seq_trimmed,
                mod_cands,
            ) = rep_results[(pair, old_r)]
            if not new_pieces:
                old_to_new_rep_idx[old_r] = -1
            else:
                new_r = len(kept_reps) + 1
                old_to_new_rep_idx[old_r] = new_r
                rep.rep_idx = new_r
                if pieces_changed:
                    rep.pieces = tuple(new_pieces)
                    rep.piece_cand_ids = tuple(new_piece_cand_ids)
                    rep.length_bp_no_n = sum(len(p.seq) for p in new_pieces)
                if seq_trimmed:
                    trimmed_old_rep_mod_cands[old_r] = mod_cands
                    rep.partial_content = True
                kept_reps.append(rep)

        if not kept_reps:
            sol.non_backbone_reps = ()
            module_c_dropped_groups += 1
            dropped_cids = {
                cid for cid, r_idx in sol.cand_to_rep_idx.items() if r_idx > 0
            }
            for j_entry in guard_res.json_log_entries:
                if (
                    j_entry.get("final_reason") == "ACCEPTED_CANDIDATE"
                    and (
                        not dropped_cids
                        or j_entry.get("candidate_id") in dropped_cids
                    )
                ):
                    j_entry["final_reason"] = "PAN_HOMOLOG_SUBTRACTED"
            sol.cand_to_rep_idx = {}
            sol.cand_to_nested_sig = {}
            sol.cand_to_reconstructed_len = {}
            sol.cand_to_flags = {}
            sol.rep0_cand_spans = {}
        else:
            had_orig_cands = {
                old_r for old_r in sol.cand_to_rep_idx.values() if old_r > 0
            }
            new_cand_to_rep_idx: dict[str, int] = {}
            dropped_cids = set()
            for cid, old_r in sol.cand_to_rep_idx.items():
                if old_r <= 0:
                    new_cand_to_rep_idx[cid] = old_r
                    continue
                new_r = old_to_new_rep_idx[old_r]
                if new_r == -1:
                    new_cand_to_rep_idx[cid] = -1
                    dropped_cids.add(cid)
                    prev_f = sol.cand_to_flags.get(cid, (False, False, False))
                    sol.cand_to_flags[cid] = (True, prev_f[1], prev_f[2])
                    sol.cand_to_nested_sig.pop(cid, None)
                    sol.cand_to_reconstructed_len.pop(cid, None)
                elif old_r in trimmed_old_rep_mod_cands:
                    mod_cands = trimmed_old_rep_mod_cands[old_r]
                    kept_rep = kept_reps[new_r - 1]
                    if kept_rep.is_synthetic or (
                        kept_rep.piece_cand_ids
                        and len(kept_rep.piece_cand_ids) == len(kept_rep.pieces)
                    ):
                        cand_len = sum(
                            len(p.seq)
                            for p_i, p in enumerate(kept_rep.pieces)
                            if (
                                p_i >= len(kept_rep.piece_cand_ids)
                                or not kept_rep.piece_cand_ids[p_i]
                                or cid in kept_rep.piece_cand_ids[p_i]
                            )
                        )
                        if cand_len == 0:
                            new_cand_to_rep_idx[cid] = -1
                            dropped_cids.add(cid)
                            prev_f = sol.cand_to_flags.get(
                                cid, (False, False, False)
                            )
                            sol.cand_to_flags[cid] = (
                                True,
                                prev_f[1],
                                prev_f[2],
                            )
                            sol.cand_to_nested_sig.pop(cid, None)
                            sol.cand_to_reconstructed_len.pop(cid, None)
                        else:
                            new_cand_to_rep_idx[cid] = new_r
                            if mod_cands is None or cid in mod_cands:
                                prev_f = sol.cand_to_flags.get(
                                    cid, (False, False, False)
                                )
                                sol.cand_to_flags[cid] = (
                                    True,
                                    prev_f[1],
                                    prev_f[2],
                                )
                                sol.cand_to_reconstructed_len[cid] = cand_len
                    else:
                        new_cand_to_rep_idx[cid] = new_r
                        if mod_cands is None or cid in mod_cands:
                            prev_f = sol.cand_to_flags.get(
                                cid, (False, False, False)
                            )
                            sol.cand_to_flags[cid] = (
                                True,
                                prev_f[1],
                                prev_f[2],
                            )
                            sol.cand_to_reconstructed_len[cid] = (
                                kept_rep.length_bp_no_n
                            )
                else:
                    new_cand_to_rep_idx[cid] = new_r

            # Defensive compaction if cand_len==0 remapping orphaned a kept_rep
            surviving_new_rs = {
                r for r in new_cand_to_rep_idx.values() if r > 0
            }
            new_to_old_rep_idx = {
                nr: or_ for or_, nr in old_to_new_rep_idx.items() if nr > 0
            }
            if any(
                (nr not in surviving_new_rs)
                and (new_to_old_rep_idx.get(nr) in had_orig_cands)
                for nr in range(1, len(kept_reps) + 1)
            ):
                final_reps: list[ClusterRepresentative] = []
                remap_nr: dict[int, int] = {}
                for old_nr, rep in enumerate(kept_reps, start=1):
                    if (old_nr not in surviving_new_rs) and (
                        new_to_old_rep_idx.get(old_nr) in had_orig_cands
                    ):
                        continue
                    final_nr = len(final_reps) + 1
                    remap_nr[old_nr] = final_nr
                    rep.rep_idx = final_nr
                    final_reps.append(rep)
                kept_reps = final_reps
                new_cand_to_rep_idx = {
                    cid: (remap_nr[r] if r > 0 else r)
                    for cid, r in new_cand_to_rep_idx.items()
                }

            if not kept_reps:
                sol.non_backbone_reps = ()
                module_c_dropped_groups += 1
                for j_entry in guard_res.json_log_entries:
                    if (
                        j_entry.get("final_reason") == "ACCEPTED_CANDIDATE"
                        and (
                            not dropped_cids
                            or j_entry.get("candidate_id") in dropped_cids
                        )
                    ):
                        j_entry["final_reason"] = "PAN_HOMOLOG_SUBTRACTED"
                sol.cand_to_rep_idx = {}
                sol.cand_to_nested_sig = {}
                sol.cand_to_reconstructed_len = {}
                sol.cand_to_flags = {}
                sol.rep0_cand_spans = {}
            else:
                sol.non_backbone_reps = tuple(kept_reps)
                sol.cand_to_rep_idx = new_cand_to_rep_idx
                if dropped_cids:
                    for j_entry in guard_res.json_log_entries:
                        if (
                            j_entry.get("final_reason") == "ACCEPTED_CANDIDATE"
                            and j_entry.get("candidate_id") in dropped_cids
                        ):
                            j_entry["final_reason"] = "PAN_HOMOLOG_SUBTRACTED"

    # Explicit cleanup of Stage 5.5 indexes and scratch sets
    sealed_waves.clear()
    active_subwaves.clear()
    active_all_pieces.clear()
    tail_pieces.clear()
    tail_gate.release()
    tail_delta_pieces.clear()
    tail_delta_gate.release()
    tail_wave = None
    accepted_exact_seqs.clear()
    rep_results.clear()
    entries.clear()

    return pan_masked_bp, pan_short_residual_bp, module_c_dropped_groups


def run_chromosome_pipeline(
    graph: GFAGraph,
    config: BuildConfig,
) -> ChromosomeBuildResult:
    """
    Execute the complete PanFlat v1 chromosome build pipeline (§3.0–§3.8, §5.1–§5.5).
    """
    validate_build_config(config)

    if (
        isinstance(graph.links, CompactLinkSet)
        and int(graph.links._canon_links.shape[0]) > 100_000
    ):
        graph.links._canon_links = np.empty(0, dtype=np.uint64)
        graph.links._first_v = None

    t_pipe = time.monotonic()
    log_lines: list[str] = [
        f"INFO\tPanFlat build starting for chrom={graph.ref_chrom} "
        f"gfa={graph.gfa_path} gfa_sha256={graph.gfa_sha256} "
        f"ref_len={len(graph.ref_seq)} haplotypes={len(graph.all_haplotypes)}"
    ]

    # 1. Delimit centromere regions (§3.8.1)
    raw_regions, cen_delim_logs = delimit_centromere_regions(graph, config)
    regions: list[CentromereRegion] = []
    for reg in raw_regions:
        prefixed_id = (
            reg.region_id
            if reg.region_id.startswith(f"{graph.ref_chrom}_")
            else f"{graph.ref_chrom}_{reg.region_id}"
        )
        regions.append(replace(reg, region_id=prefixed_id))
    log_lines.extend(cen_delim_logs)

    # 2. Top-level whole-contig orientation (§3.3.0) on deterministically sorted alt fragments
    alt_sorted = sorted(
        graph.alt_fragments,
        key=lambda f: (f.hap_id, f.src_path, f.base_offset),
    )
    if len(graph.ref_fragment.steps) > 100_000:
        object.__setattr__(graph, "alt_fragments", ())
    oriented_fragments: list[PathFragment] = []
    for frag in alt_sorted:
        oriented = orient_contig_top_level(frag, graph)
        if oriented.is_flipped_contig:
            log_lines.append(
                f"INFO\tORIENT_CONTIG\thap={oriented.hap_id}\tsrc_path={oriented.src_path}\tsrc_strand=-"
            )
            if (
                len(graph.ref_fragment.steps) > 100_000
                and isinstance(frag.steps, LazyStepTuple)
                and isinstance(oriented.provenance, LazyProvenanceTuple)
                and frag.steps._codes is not oriented.provenance._orig_codes
            ):
                frag.steps._codes = np.empty(0, dtype=np.int32)
        oriented_fragments.append(oriented)

    # 3. Partition fragments by centromere regions & process centromere mode (§3.8.1–3.8.5)
    partition = partition_fragments_by_centromere(
        oriented_fragments, regions, graph, config
    )
    cen_out = process_centromere_regions(partition, graph, config)
    if regions:
        log_progress(
            f"  Centromere processing done ({time.monotonic() - t_pipe:.2f}s): "
            f"regions={len(regions)}, mode={config.effective_centromere_mode}, "
            f"islands={len(cen_out.island_sv_events)}"
        )
    t_stage4 = time.monotonic()

    centromere_rows: list[CentromereRow] = list(cen_out.centromere_rows)
    unresolved_records: list[UnresolvedRecord] = list(cen_out.unresolved_records)
    json_log_entries: list[dict[str, Any]] = list(cen_out.json_log_entries)
    claimed_source_intervals: set[tuple[str, int, int]] = set(
        partition.claimed_source_intervals
    )

    seen_exported_intervals: dict[tuple[str, int, int, str], str] = {}
    for rec in unresolved_records:
        if rec.export_to_fasta and rec.exported_fasta_id != ".":
            seen_exported_intervals[
                (rec.src_path, rec.src_start, rec.src_end, rec.src_strand)
            ] = rec.exported_fasta_id

    unres_counter_ref = [len(unresolved_records) + 1]

    def _emit_normal_unresolved(
        region_id: str,
        event_id: str,
        prov: StepProvenance,
        seq: str,
        reason: str,
        export_to_fasta: bool,
    ) -> int:
        rec_id = f"{graph.ref_chrom}_UNRES_{unres_counter_ref[0]}"
        unres_counter_ref[0] += 1
        if export_to_fasta:
            key = (prov.src_path, prov.src_start, prov.src_end, prov.src_strand)
            exp_id = seen_exported_intervals.setdefault(key, rec_id)
        else:
            exp_id = "."
        rec = UnresolvedRecord(
            record_id=rec_id,
            region_id=region_id,
            event_id=event_id,
            src_path=prov.src_path,
            src_start=prov.src_start,
            src_end=prov.src_end,
            src_strand=prov.src_strand,
            length=prov.length,
            reason=reason,
            seq=seq,
            export_to_fasta=export_to_fasta,
            exported_fasta_id=exp_id,
        )
        idx = len(unresolved_records)
        unresolved_records.append(rec)
        return idx

    # 4. Normal fragments: Top-level inversion normalization, weighted LIS, and deviation extraction
    normal_fragments_sorted = sorted(
        partition.normal_fragments,
        key=lambda f: (
            f.hap_id,
            f.src_path,
            f.provenance[0].src_start if f.provenance else f.base_offset,
            f.provenance[-1].src_end if f.provenance else f.base_offset,
        ),
    )
    del alt_sorted, oriented_fragments, partition

    ref_steps = graph.ref_fragment.steps
    n_ref = len(ref_steps)
    ref_hap_id = graph.ref_fragment.hap_id
    top_state_key = SubproblemStateKey(
        b_id=(("REF", 0, n_ref, "+"),),
        lo=0,
        hi=n_ref,
        has_logical_start=False,
        has_logical_end=False,
        m_key=tuple(
            make_provenance_key(f.provenance)
            for f in normal_fragments_sorted
            if f.provenance
        ),
    )

    contig_analyses: list[_ContigAnalysis] = []
    inv_chains_by_key: dict[
        tuple[str, int, int],
        list[tuple[_ContigAnalysis, ReverseChainRecord]],
    ] = collections.defaultdict(list)
    terminal_devs_by_pair: dict[
        tuple[int, int], list[tuple[_ContigAnalysis, DeviationSegment]]
    ] = collections.defaultdict(list)
    large_jump_devs_by_pair: dict[
        tuple[int, int], list[tuple[_ContigAnalysis, DeviationSegment]]
    ] = collections.defaultdict(list)

    force_fork = _force_fork_enabled()
    max_fork_workers = _resolve_max_fork_workers(config.threads)
    can_fork = config.threads > 1 and _supports_fork()
    target_node_lens = _resolve_graph_node_lens(graph)
    n_frags = len(normal_fragments_sorted)
    use_fork_s4 = (
        can_fork
        and (max_fork_workers > 1 or force_fork)
        and (n_frags >= 1 if force_fork else n_frags >= 2)
        and (n_ref >= 5_000 or force_fork)
    )

    stage4_ctx = _Stage4ForkContext(
        graph=graph,
        config=config,
        normal_fragments_sorted=normal_fragments_sorted,
        ref_steps=ref_steps,
        n_ref=n_ref,
        top_state_key=top_state_key,
        claimed_source_intervals=claimed_source_intervals,
        target_node_lens=target_node_lens,
    )

    def _merge_stage4_result(res: _FragmentStage4Result) -> None:
        analysis = res.analysis
        contig_analyses.append(analysis)
        for ch in res.inv_chains:
            inv_chains_by_key[(ch.classification, ch.rmin, ch.rmax)].append(
                (analysis, ch)
            )
        for u_prov, u_seq in res.pending_unanchored_unresolved:
            _emit_normal_unresolved(
                region_id=".",
                event_id=".",
                prov=u_prov,
                seq=u_seq,
                reason="UNANCHORED_END",
                export_to_fasta=True,
            )
        for lj_pair, lj_dev in res.large_jump_devs:
            large_jump_devs_by_pair[lj_pair].append((analysis, lj_dev))
        for t_pair, t_dev in res.terminal_devs:
            terminal_devs_by_pair[t_pair].append((analysis, t_dev))

    if use_fork_s4:
        if (
            isinstance(ref_steps, LazyStepTuple)
            and isinstance(graph.nodes, CompactNodeStore)
            and len(graph.nodes._extra) == 0
            and n_ref > 10_000
        ):
            _get_lazy_backbone_anchor_arrays(ref_steps, graph.nodes._max_nid)

        frag_step_lens = [len(f.steps) for f in normal_fragments_sorted]
        ordered_frag_indices = sorted(
            range(n_frags),
            key=lambda i: (-frag_step_lens[i], i),
        )
        n_workers_s4 = min(max_fork_workers, n_frags)
        results_by_idx: dict[int, _FragmentStage4Result] = {}
        global _STAGE4_FORK_CTX
        _STAGE4_FORK_CTX = stage4_ctx
        t_s4_pool = time.monotonic()
        last_s4_log = t_s4_pool
        done_s4 = 0
        try:
            with _fork_gc_frozen():
                with ProcessPoolExecutor(
                    max_workers=n_workers_s4,
                    mp_context=multiprocessing.get_context("fork"),
                ) as pool:
                    futs = [
                        pool.submit(_stage4_analyze_fragment_worker, f_idx)
                        for f_idx in ordered_frag_indices
                    ]
                    for fut in as_completed(futs):
                        f_res = fut.result()
                        results_by_idx[f_res.frag_idx] = f_res
                        done_s4 += 1
                        if n_frags >= 8:
                            now_t = time.monotonic()
                            if (
                                done_s4 == n_frags
                                or now_t - last_s4_log >= 10.0
                            ):
                                log_progress(
                                    f"  Normalizing & LIS contigs (fork, workers={n_workers_s4}): "
                                    f"{done_s4:,}/{n_frags:,} ({100.0 * done_s4 / n_frags:.1f}%, "
                                    f"{now_t - t_s4_pool:.1f}s)"
                                )
                                last_s4_log = now_t
        finally:
            _STAGE4_FORK_CTX = None

        for frag_idx in range(n_frags):
            f_res = results_by_idx.pop(frag_idx)
            _rebind_stage4_result_after_ipc(f_res, target_node_lens)
            _merge_stage4_result(f_res)
        del results_by_idx
    else:
        for frag_idx in range(n_frags):
            f_res = _analyze_single_normal_fragment(
                frag_idx, stage4_ctx, for_ipc=False
            )
            _merge_stage4_result(f_res)

    del stage4_ctx, normal_fragments_sorted

    # 5. Prepare candidate screening & group solving for all terminal normal groups G(a, b)
    sorted_group_pairs = sorted(
        terminal_devs_by_pair.keys(),
        key=lambda ab: (ab[1], ab[1] - ab[0], ab[0]),
    )
    log_progress(
        f"  Inversion normalization & weighted LIS done ({time.monotonic() - t_stage4:.2f}s): "
        f"contigs={len(contig_analyses)}, terminal_groups={len(sorted_group_pairs):,}"
    )
    t_stage5 = time.monotonic()

    group_guard_by_pair: dict[tuple[int, int], GroupRepeatGuardResult] = {}
    group_solutions: dict[tuple[int, int], ParallelGroupSolution] = {}

    active_pairs: list[tuple[int, int]] = []
    for pair in sorted_group_pairs:
        devs_at_pair = terminal_devs_by_pair[pair]
        if not any(tdev.branch_provenance for _, tdev in devs_at_pair):
            group_guard_by_pair[pair] = GroupRepeatGuardResult()
        else:
            active_pairs.append(pair)

    ref_dedup_aligner: Any | None = None
    ref_dedup_short_aligner: Any | None = None
    if config.dedup_mode in ("ref", "pan") and active_pairs and graph.ref_seq:
        t_ref_idx = time.monotonic()
        ref_seq_up = graph.ref_seq.upper()
        ref_dedup_aligner = mappy.Aligner(
            seq=ref_seq_up, preset="asm20", best_n=50
        )
        ref_dedup_short_aligner = mappy.Aligner(
            seq=ref_seq_up,
            k=13,
            w=5,
            min_cnt=2,
            min_chain_score=20,
            best_n=50,
        )
        del ref_seq_up
        log_progress(
            f"  Built read-only reference dedup indexes ({time.monotonic() - t_ref_idx:.2f}s, "
            f"ref_len={len(graph.ref_seq):,} bp, dedup_mode={config.dedup_mode}, "
            f"dedup_id={config.dedup_id})"
        )

    stage5_ctx = _Stage5ForkContext(
        graph=graph,
        config=config,
        ref_steps=ref_steps,
        n_ref=n_ref,
        top_state_key=top_state_key,
        terminal_devs_by_pair=terminal_devs_by_pair,
        claimed_source_intervals=claimed_source_intervals,
        ref_dedup_aligner=ref_dedup_aligner,
        ref_dedup_short_aligner=ref_dedup_short_aligner,
    )

    n_active = len(active_pairs)
    use_fork_s5 = (
        can_fork
        and (max_fork_workers > 1 or force_fork)
        and n_active >= 1
        and (n_active >= 32 or force_fork)
    )

    if use_fork_s5:
        pair_weights: dict[tuple[int, int], int] = {}
        for pair in active_pairs:
            ra, rb = pair
            cand_bp = 0
            for _, tdev in terminal_devs_by_pair[pair]:
                bp_prov = tdev.branch_provenance
                if bp_prov:
                    if isinstance(bp_prov, LazyProvenanceTuple):
                        cand_bp += bp_prov.total_bp()
                    else:
                        cand_bp += sum(p.length for p in bp_prov)
            ref_bp = max(
                0, graph.ref_step_ranges[rb][0] - graph.ref_step_ranges[ra][1]
            )
            pair_weights[pair] = cand_bp + ref_bp

        ordered_pairs = sorted(
            active_pairs, key=lambda p: (-pair_weights[p], p)
        )
        n_workers_s5 = min(max_fork_workers, n_active)
        eff_max_groups = min(64, max(1, n_active // (4 * n_workers_s5)))
        batches: list[list[tuple[int, int]]] = []
        cur_batch: list[tuple[int, int]] = []
        cur_bp = 0
        for pair in ordered_pairs:
            w = pair_weights[pair]
            if w >= 10_000:
                if cur_batch:
                    batches.append(cur_batch)
                    cur_batch = []
                    cur_bp = 0
                batches.append([pair])
            else:
                if cur_batch and (
                    len(cur_batch) >= eff_max_groups or cur_bp + w > 20_000
                ):
                    batches.append(cur_batch)
                    cur_batch = [pair]
                    cur_bp = w
                else:
                    cur_batch.append(pair)
                    cur_bp += w
        if cur_batch:
            batches.append(cur_batch)

        n_workers_s5 = min(n_workers_s5, len(batches))
        raw_results_by_pair: dict[
            tuple[int, int],
            tuple[GroupRepeatGuardResult, ParallelGroupSolution | None],
        ] = {}
        global _STAGE5_FORK_CTX
        _STAGE5_FORK_CTX = stage5_ctx
        t_solve = time.monotonic()
        last_solve_log = t_solve
        solved_count = 0
        try:
            with _fork_gc_frozen():
                with ProcessPoolExecutor(
                    max_workers=n_workers_s5,
                    mp_context=multiprocessing.get_context("fork"),
                ) as pool:
                    futs = [
                        pool.submit(_stage5_screen_and_solve_batch_worker, batch)
                        for batch in batches
                    ]
                    for fut in as_completed(futs):
                        batch_out = fut.result()
                        for pair, g_res, sol in batch_out:
                            raw_results_by_pair[pair] = (g_res, sol)
                        solved_count += len(batch_out)
                        if n_active >= 20:
                            now_t = time.monotonic()
                            if (
                                solved_count == n_active
                                or now_t - last_solve_log >= 10.0
                            ):
                                pct = 100.0 * solved_count / n_active
                                log_progress(
                                    f"  Screening & solving parallel groups (fork, "
                                    f"workers={n_workers_s5}): {solved_count:,}/{n_active:,} "
                                    f"({pct:.1f}%, {now_t - t_solve:.1f}s)"
                                )
                                last_solve_log = now_t
        finally:
            _STAGE5_FORK_CTX = None

        for pair in active_pairs:
            g_res, sol = raw_results_by_pair.pop(pair)
            if g_res.unresolved_records:
                remapped: list[UnresolvedRecord] = []
                for u_rec in g_res.unresolved_records:
                    rec_id = f"{graph.ref_chrom}_UNRES_{unres_counter_ref[0]}"
                    unres_counter_ref[0] += 1
                    if u_rec.export_to_fasta:
                        key = (
                            u_rec.src_path,
                            u_rec.src_start,
                            u_rec.src_end,
                            u_rec.src_strand,
                        )
                        exp_id = seen_exported_intervals.setdefault(key, rec_id)
                    else:
                        exp_id = "."
                    remapped.append(
                        replace(
                            u_rec,
                            record_id=rec_id,
                            exported_fasta_id=exp_id,
                        )
                    )
                g_res.unresolved_records = remapped
            group_guard_by_pair[pair] = g_res
            if sol is not None:
                group_solutions[pair] = sol
        del raw_results_by_pair
    else:
        has_tbuf_s5 = hasattr(mappy, "ThreadBuffer")
        serial_ref_buf = (
            mappy.ThreadBuffer()
            if ref_dedup_aligner is not None and has_tbuf_s5
            else None
        )
        serial_ref_short_buf = (
            mappy.ThreadBuffer()
            if ref_dedup_short_aligner is not None and has_tbuf_s5
            else None
        )
        try:
            for pair in active_pairs:
                g_res, sol = _screen_and_solve_one_pair(
                    pair,
                    stage5_ctx,
                    unres_counter_ref=unres_counter_ref,
                    seen_exported_intervals=seen_exported_intervals,
                    ref_dedup_buf=serial_ref_buf,
                    ref_dedup_short_buf=serial_ref_short_buf,
                )
                group_guard_by_pair[pair] = g_res
                if sol is not None:
                    group_solutions[pair] = sol
        finally:
            del serial_ref_buf, serial_ref_short_buf

    del stage5_ctx
    if ref_dedup_aligner is not None or ref_dedup_short_aligner is not None:
        ref_dedup_aligner = None
        ref_dedup_short_aligner = None
        gc.collect()
    for pair in active_pairs:
        devs_at_pair = terminal_devs_by_pair[pair]
        for d_idx, (analysis, tdev) in enumerate(devs_at_pair):
            if tdev.branch_provenance:
                devs_at_pair[d_idx] = (
                    analysis,
                    replace(tdev, branch_provenance=()),
                )
    for ca in contig_analyses:
        ca.node_visit_counts = {}

    log_progress(
        f"  Candidate screening & group solving done ({time.monotonic() - t_stage5:.2f}s): "
        f"solved_groups={len(group_solutions):,}"
    )

    # 5.5 Cross-group / cross-representative deterministic priority pan-dedup (strictly post-fork)
    dedup_ref_masked_bp = sum(
        g.ref_masked_bp for g in group_guard_by_pair.values()
    )
    dedup_short_residual_bp = sum(
        g.short_residual_bp for g in group_guard_by_pair.values()
    )
    dedup_dropped_groups = sum(
        1
        for pair in active_pairs
        if group_guard_by_pair[pair].contig_had_subtracted_homolog
        and not (pair in group_solutions and group_solutions[pair].non_backbone_reps)
    )
    dedup_pan_masked_bp = 0
    if config.dedup_mode == "pan" and group_solutions:
        t_stage55 = time.monotonic()
        pan_m_bp, pan_sr_bp, pan_drop_g = (
            _deduplicate_representatives_across_chromosome(
                graph=graph,
                config=config,
                sorted_group_pairs=sorted_group_pairs,
                group_guard_by_pair=group_guard_by_pair,
                group_solutions=group_solutions,
                cen_out=cen_out,
            )
        )
        dedup_pan_masked_bp = pan_m_bp
        dedup_short_residual_bp += pan_sr_bp
        dedup_dropped_groups += pan_drop_g
        log_progress(
            f"  Cross-group pan deduplication done ({time.monotonic() - t_stage55:.2f}s): "
            f"pan_masked_bp={dedup_pan_masked_bp:,}, dropped_groups={pan_drop_g:,}"
        )

    # 6. Collect all top-level normal SV events before assigning region_id and event_id
    @dataclass
    class _PendingNormalEvent:
        locus_start: int
        locus_end: int
        event: SVEvent
        slot_block: InsertionSlotBlock | None = None
        guard_pair: tuple[int, int] | None = None
        chain_unres_indices: list[int] = field(default_factory=list)

    pending_events: list[_PendingNormalEvent] = []
    contigs_by_hap: dict[str, list[_ContigAnalysis]] = collections.defaultdict(list)
    for ca in contig_analyses:
        contigs_by_hap[ca.raw_fragment.hap_id].append(ca)

    # 6a. Inversion events (INV, INV_DISPLACED, INV_EDGE, INV_CONFLICT)
    for inv_key in sorted(inv_chains_by_key.keys(), key=lambda x: (x[1], x[2], x[0])):
        cls_name, rmin, rmax = inv_key
        entries = inv_chains_by_key[inv_key]
        ref_s = graph.ref_step_ranges[rmin][0]
        ref_e = graph.ref_step_ranges[rmax][1]
        span_len = ref_e - ref_s

        chain_unres_idxs: list[int] = []
        if cls_name in ("INV_EDGE", "INV_CONFLICT"):
            for ca, ch in entries:
                for sub_steps, sub_provs in ch.novel_unresolved_slices:
                    sub_seq = spell_steps(sub_steps, graph.nodes)
                    for sl_prov, sl_seq in merge_contiguous_provenance(
                        sub_provs, sub_seq
                    ):
                        if (
                            sl_prov.src_path,
                            sl_prov.src_start,
                            sl_prov.src_end,
                        ) not in claimed_source_intervals:
                            idx_u = _emit_normal_unresolved(
                                region_id=".",
                                event_id=".",
                                prov=sl_prov,
                                seq=sl_seq,
                                reason=cls_name,
                                export_to_fasta=True,
                            )
                            chain_unres_idxs.append(idx_u)

        if span_len < config.min_sv_len:
            continue

        carrier_contig_ids = {ca.contig_idx for ca, _ in entries}
        first_ch = entries[0][1]
        l_flank = first_ch.left_flank_rank
        r_flank = first_ch.right_flank_rank

        contig_calls: dict[str, list[str]] = collections.defaultdict(list)
        for hap_id, ca_list in contigs_by_hap.items():
            for ca in ca_list:
                if ca.contig_idx in carrier_contig_ids:
                    contig_calls[hap_id].append("1")
                elif _contig_covers_rank_interval(ca, rmin, rmax, n_ref):
                    contig_calls[hap_id].append("0")
                else:
                    contig_calls[hap_id].append(".")

        genotypes = _resolve_haplotype_genotypes(
            graph.all_haplotypes, ref_hap_id, "0", contig_calls
        )
        carriers = sorted(h for h, g in genotypes.items() if g == "1")
        if not carriers:
            carriers = sorted({ca.raw_fragment.hap_id for ca, _ in entries})
        n_carrier = sum(1 for g in genotypes.values() if g not in ("0", "."))
        n_called = sum(1 for g in genotypes.values() if g != ".")
        exemplar_hap = carriers[0]
        has_mh = any(ca.lis_multi_hit for ca, _ in entries)
        mh_tag = ",multi_hit=1" if has_mh else ""
        l_str = str(l_flank) if l_flank is not None else str(rmin)
        r_str = str(r_flank) if r_flank is not None else str(rmax)

        ev = SVEvent(
            chrom=graph.ref_chrom,
            pan_start=0,
            pan_end=0,
            event_id="",
            event_type=cls_name,  # type: ignore[arg-type]
            ref_start=ref_s,
            ref_end=ref_e,
            anchors=f"{l_str}..{r_str}",
            alleles_str=f"1:len={span_len},src={exemplar_hap},n={len(carriers)}{mh_tag}",
            n_carrier=n_carrier,
            n_called=n_called,
            region_id=".",
            genotypes=genotypes,
            pieces=(),
        )
        pending_events.append(
            _PendingNormalEvent(
                locus_start=ref_s,
                locus_end=ref_e,
                event=ev,
                chain_unres_indices=chain_unres_idxs,
            )
        )

    # 6b. Large jump events (`LARGE_JUMP`)
    for pair in sorted(
        large_jump_devs_by_pair.keys(), key=lambda ab: (ab[0], ab[1])
    ):
        rank_a, rank_b = pair
        lj_entries = large_jump_devs_by_pair[pair]
        ref_s = graph.ref_step_ranges[rank_a][1]
        ref_e = graph.ref_step_ranges[rank_b][0]
        span_len = ref_e - ref_s

        lj_unres_idxs: list[int] = []
        for ca, dev in lj_entries:
            lj_runs = _extract_contiguous_nonref_runs_for_unresolved(
                steps=dev.branch_steps,
                provenance=dev.branch_provenance,
                nodes=graph.nodes,
                global_ref_nodes=graph.global_ref_nodes,
                min_sv_len=config.min_sv_len,
                claimed_source_intervals=claimed_source_intervals,
            )
            for u_prov, u_seq in lj_runs:
                idx_u = _emit_normal_unresolved(
                    region_id=".",
                    event_id=".",
                    prov=u_prov,
                    seq=u_seq,
                    reason="LARGE_JUMP",
                    export_to_fasta=True,
                )
                lj_unres_idxs.append(idx_u)

        carrier_contig_ids = {ca.contig_idx for ca, _ in lj_entries}
        contig_calls = collections.defaultdict(list)
        for hap_id, ca_list in contigs_by_hap.items():
            for ca in ca_list:
                if ca.contig_idx in carrier_contig_ids:
                    contig_calls[hap_id].append("1")
                elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
                    contig_calls[hap_id].append("0")
                else:
                    contig_calls[hap_id].append(".")

        genotypes = _resolve_haplotype_genotypes(
            graph.all_haplotypes, ref_hap_id, "0", contig_calls
        )
        carriers = sorted(h for h, g in genotypes.items() if g == "1")
        if not carriers:
            carriers = sorted({ca.raw_fragment.hap_id for ca, _ in lj_entries})
        n_carrier = sum(1 for g in genotypes.values() if g not in ("0", "."))
        n_called = sum(1 for g in genotypes.values() if g != ".")
        exemplar_hap = carriers[0]
        has_mh = any(bool(dev.multi_hit_nodes) for _, dev in lj_entries)
        mh_tag = ",multi_hit=1" if has_mh else ""

        ev = SVEvent(
            chrom=graph.ref_chrom,
            pan_start=0,
            pan_end=0,
            event_id="",
            event_type="LARGE_JUMP",
            ref_start=ref_s,
            ref_end=ref_e,
            anchors=f"{rank_a}..{rank_b}",
            alleles_str=f"1:len={span_len},src={exemplar_hap},n={len(carriers)}{mh_tag}",
            n_carrier=n_carrier,
            n_called=n_called,
            region_id=".",
            genotypes=genotypes,
            pieces=(),
        )
        pending_events.append(
            _PendingNormalEvent(
                locus_start=ref_s,
                locus_end=ref_e,
                event=ev,
                chain_unres_indices=lj_unres_idxs,
            )
        )

    # 6c. Terminal normal groups G(a, b): DUP_TRA, CNV, DEL, and INS
    # Precompute reference step lengths & per-contig sorted terminal deviation index for §5.2 INS genotyping
    ref_step_lens_arr: np.ndarray | None = (
        np.diff(graph._ref_cum_offsets)
        if graph._ref_cum_offsets is not None
        else None
    )
    dev_spans_by_contig: dict[int, list[tuple[int, int, int]]] = (
        collections.defaultdict(list)
    )
    for (ta, tb), pair_entries in terminal_devs_by_pair.items():
        for ca_d, tdev_d in pair_entries:
            br_bp = _branch_steps_bp(tdev_d.branch_steps, graph)
            if br_bp > 0:
                dev_spans_by_contig[ca_d.contig_idx].append((ta, tb, br_bp))
    for ca_s in contig_analyses:
        if ca_s.small_dev_bp_spans:
            dev_spans_by_contig[ca_s.contig_idx].extend(
                ca_s.small_dev_bp_spans
            )
    dev_starts_by_contig: dict[int, list[int]] = {}
    for c_idx, entries_c in dev_spans_by_contig.items():
        entries_c.sort(key=lambda x: (x[0], x[1]))
        dev_starts_by_contig[c_idx] = [e[0] for e in entries_c]

    def _uncovered_ref_bp_open(
        ca_obj: _ContigAnalysis,
        r_a: int,
        r_b: int,
    ) -> int:
        lo_r = max(0, r_a + 1)
        hi_r = min(n_ref, r_b)
        if lo_r >= hi_r:
            return 0
        cov = ca_obj.covered_ranks
        if isinstance(cov, CompactRankSet):
            uncov_mask = ~cov._mask[lo_r:hi_r]
            if not bool(np.any(uncov_mask)):
                return 0
            if ref_step_lens_arr is not None:
                return int(
                    ref_step_lens_arr[lo_r:hi_r][uncov_mask].sum(dtype=np.int64)
                )
            return sum(
                graph.ref_step_ranges[lo_r + off][1]
                - graph.ref_step_ranges[lo_r + off][0]
                for off in np.flatnonzero(uncov_mask).tolist()
            )
        return sum(
            graph.ref_step_ranges[r][1] - graph.ref_step_ranges[r][0]
            for r in range(lo_r, hi_r)
            if r not in cov
        )

    def _dev_branch_bp_in_span(
        contig_idx: int,
        r_a: int,
        r_b: int,
    ) -> int:
        entries_c = dev_spans_by_contig.get(contig_idx)
        if not entries_c:
            return 0
        starts_c = dev_starts_by_contig[contig_idx]
        pos = bisect.bisect_left(starts_c, r_a)
        total_bp = 0
        n_ent = len(entries_c)
        while pos < n_ent:
            ta_i, tb_i, br_bp_i = entries_c[pos]
            if ta_i >= r_b:
                break
            if tb_i <= r_b:
                total_bp += br_bp_i
            pos += 1
        return total_bp

    def _case_a_rep0_matches_backbone(
        tdev_here: DeviationSegment,
        b_steps_slice: Sequence[Step],
        c_list_here: Sequence[ExtractedCandidate],
        cand_dev_loc: Mapping[str, tuple[int, int]],
        rep0_cand_spans: Mapping[str, tuple[int, int, int, int]],
    ) -> bool:
        b_step_starts: list[int] = []
        b_step_ends: list[int] = []
        b_positions_by_tok: dict[tuple[str, str], list[int]] = (
            collections.defaultdict(list)
        )
        cur_b_off = 0
        for b_i, b_st in enumerate(b_steps_slice):
            b_len = _step_bp_len(b_st, graph.nodes)
            b_step_starts.append(cur_b_off)
            cur_b_off += b_len
            b_step_ends.append(cur_b_off)
            b_positions_by_tok[(b_st.node_id, b_st.orient)].append(b_i)
        len_b = cur_b_off

        cands_at_step: dict[int, list[tuple[int, int, int, int]]] = (
            collections.defaultdict(list)
        )
        kept_nonref_bp = 0
        for kc in c_list_here:
            span = rep0_cand_spans.get(kc.candidate_id)
            loc = cand_dev_loc.get(kc.candidate_id)
            if span is None or loc is None:
                return False
            kept_nonref_bp += span[0]
            cands_at_step[loc[1]].append(span)

        nonref_br_bp = 0
        for st in tdev_here.branch_steps:
            if not (_is_full_step(st) and st.node_id in graph.global_ref_nodes):
                nonref_br_bp += _step_bp_len(st, graph.nodes)

        dropped_nonref_bp = max(0, nonref_br_bp - kept_nonref_bp)
        if dropped_nonref_bp >= config.min_sv_len:
            return False

        cursor_b = 0
        b_step_cursor = 0
        covered_b_bp = 0
        extra_ref_br_bp = 0
        total_internal_del_bp = 0
        ins_excess_bp = 0

        for step_idx, st in enumerate(tdev_here.branch_steps):
            if step_idx in cands_at_step:
                for kept_len, t_start, t_end, internal_del_bp in cands_at_step[
                    step_idx
                ]:
                    eff_start = max(t_start, cursor_b)
                    new_b_cov = max(0, t_end - eff_start)
                    if new_b_cov > 0:
                        covered_b_bp += new_b_cov
                        cursor_b = t_end
                        total_internal_del_bp += internal_del_bp
                        ins_excess_bp += max(
                            0, kept_len + internal_del_bp - new_b_cov
                        )
                    else:
                        ins_excess_bp += kept_len
            if _is_full_step(st) and st.node_id in graph.global_ref_nodes:
                slen = _step_bp_len(st, graph.nodes)
                tok = (st.node_id, st.orient)
                pos_list = b_positions_by_tok.get(tok)
                matched_b_idx = -1
                if pos_list:
                    p_lo = bisect.bisect_left(pos_list, b_step_cursor)
                    while p_lo < len(pos_list):
                        cand_b_i = pos_list[p_lo]
                        if b_step_ends[cand_b_i] > cursor_b:
                            matched_b_idx = cand_b_i
                            break
                        p_lo += 1
                if matched_b_idx >= 0:
                    bs_start = b_step_starts[matched_b_idx]
                    bs_end = b_step_ends[matched_b_idx]
                    ov_prev = max(0, cursor_b - bs_start)
                    new_cov = bs_end - max(bs_start, cursor_b)
                    covered_b_bp += new_cov
                    cursor_b = bs_end
                    b_step_cursor = matched_b_idx + 1
                    ins_excess_bp += ov_prev
                else:
                    extra_ref_br_bp += slen

        uncovered_b_bp = (len_b - covered_b_bp) + total_internal_del_bp
        return (
            extra_ref_br_bp + dropped_nonref_bp + uncovered_b_bp
            < config.min_sv_len
            and extra_ref_br_bp + dropped_nonref_bp + ins_excess_bp
            < config.min_sv_len
        )

    guard_unres_indices_by_pair: dict[tuple[int, int], list[int]] = (
        collections.defaultdict(list)
    )
    use_lazy_ref_slice = isinstance(ref_steps, LazyStepTuple)
    n_stage6_pairs = len(sorted_group_pairs)
    t_stage6 = time.monotonic()
    last_stage6_log = t_stage6
    for p_idx_6, pair in enumerate(sorted_group_pairs, start=1):
        if n_stage6_pairs >= 1000 and (p_idx_6 & 255) == 0:
            now_6 = time.monotonic()
            if now_6 - last_stage6_log >= 10.0:
                log_progress(
                    f"  Calling events & genotyping: {p_idx_6:,}/{n_stage6_pairs:,} "
                    f"({100.0 * p_idx_6 / n_stage6_pairs:.1f}%, {now_6 - t_stage6:.1f}s)"
                )
                last_stage6_log = now_6
        rank_a, rank_b = pair
        devs_at_pair = terminal_devs_by_pair[pair]
        guard_res = group_guard_by_pair[pair]
        sol = group_solutions.get(pair)
        ref_s_open = graph.ref_step_ranges[rank_a][1]
        ref_e_open = graph.ref_step_ranges[rank_b][0]
        b_slice_steps: Sequence[Step] = (
            ref_steps[rank_a + 1 : rank_b]
            if use_lazy_ref_slice
            else tuple(ref_steps[rank_a + 1 : rank_b])
        )

        for u_rec in guard_res.unresolved_records:
            guard_unres_indices_by_pair[pair].append(len(unresolved_records))
            unresolved_records.append(u_rec)
        for j_entry in guard_res.json_log_entries:
            json_log_entries.append(j_entry)

        # (i) DUP_TRA check
        dup_tra_entries: list[tuple[_ContigAnalysis, DeviationSegment]] = []
        a_start, a_end = graph.ref_step_ranges[rank_a]
        for ca, dev in devs_at_pair:
            is_dt = False
            if (
                isinstance(dev.branch_steps, LazyStepTuple)
                and graph._nid_to_ref_rank is not None
                and graph._ref_cum_offsets is not None
            ):
                br_nids_dt = np.abs(dev.branch_steps._codes)
                r_rank_len_dt = int(graph._nid_to_ref_rank.shape[0])
                r_ancs = (
                    np.where(
                        br_nids_dt < r_rank_len_dt,
                        graph._nid_to_ref_rank[
                            np.minimum(br_nids_dt, max(0, r_rank_len_dt - 1))
                        ],
                        -1,
                    )
                    if r_rank_len_dt > 0
                    else np.full(br_nids_dt.shape, -1, dtype=np.int32)
                )
                out_mask = (r_ancs >= 0) & ((r_ancs < rank_a) | (r_ancs > rank_b))
                if bool(np.any(out_mask)):
                    cand_r = r_ancs[out_mask]
                    r_starts_dt = graph._ref_cum_offsets[cand_r]
                    r_ends_dt = graph._ref_cum_offsets[cand_r + 1]
                    dists = np.where(
                        cand_r > rank_a, r_starts_dt - a_end, a_start - r_ends_dt
                    )
                    if bool(np.any(dists > config.max_jump)):
                        is_dt = True
            else:
                for st in dev.branch_steps:
                    if _is_full_step(st) and st.node_id in graph.unique_ref_step_by_node:
                        r_anc = graph.unique_ref_step_by_node[st.node_id]
                        if r_anc < rank_a or r_anc > rank_b:
                            r_s, r_e = graph.ref_step_ranges[r_anc]
                            dist = (r_s - a_end) if r_anc > rank_a else (a_start - r_e)
                            if dist > config.max_jump:
                                is_dt = True
                                break
            if is_dt:
                dup_tra_entries.append((ca, dev))

        if dup_tra_entries:
            dt_contig_ids = {ca.contig_idx for ca, _ in dup_tra_entries}
            dt_calls: dict[str, list[str]] = collections.defaultdict(list)
            for hap_id, ca_list in contigs_by_hap.items():
                for ca in ca_list:
                    if ca.contig_idx in dt_contig_ids:
                        dt_calls[hap_id].append("1")
                    elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
                        dt_calls[hap_id].append("0")
                    else:
                        dt_calls[hap_id].append(".")
            dt_gt = _resolve_haplotype_genotypes(
                graph.all_haplotypes, ref_hap_id, "0", dt_calls
            )
            dt_carriers = sorted(h for h, g in dt_gt.items() if g == "1")
            if not dt_carriers:
                dt_carriers = sorted({ca.raw_fragment.hap_id for ca, _ in dup_tra_entries})
            n_car = sum(1 for g in dt_gt.values() if g not in ("0", "."))
            n_cal = sum(1 for g in dt_gt.values() if g != ".")
            has_mh = any(bool(dev.multi_hit_nodes) for _, dev in dup_tra_entries)
            mh_tag = ",multi_hit=1" if has_mh else ""
            dt_ev = SVEvent(
                chrom=graph.ref_chrom,
                pan_start=0,
                pan_end=0,
                event_id="",
                event_type="DUP_TRA",
                ref_start=ref_s_open,
                ref_end=ref_e_open,
                anchors=f"{rank_a}..{rank_b}",
                alleles_str=f"1:len={ref_e_open - ref_s_open},src={dt_carriers[0]},n={len(dt_carriers)}{mh_tag}",
                n_carrier=n_car,
                n_called=n_cal,
                region_id=".",
                genotypes=dt_gt,
                pieces=(),
            )
            pending_events.append(
                _PendingNormalEvent(
                    locus_start=ref_s_open,
                    locus_end=ref_e_open,
                    event=dt_ev,
                )
            )

        # (ii) Pre-folding Canonical CNV & DEL evaluation on B(a, b)
        if b_slice_steps:
            full_cnv_calls: list[tuple[_ContigAnalysis, DeviationSegment, CNVCall]] = []
            full_cnv_dev_indices: set[int] = set()
            for d_idx, (ca, dev) in enumerate(devs_at_pair):
                cnv_c = detect_canonical_cnv(
                    b_slice_steps,
                    dev.branch_steps,
                    graph.nodes,
                    config.min_sv_len,
                )
                if cnv_c is not None:
                    full_cnv_calls.append((ca, dev, cnv_c))
                    full_cnv_dev_indices.add(d_idx)

            emitted_full_cnv_keys: set[
                tuple[int, int, tuple[tuple[str, str], ...]]
            ] = set()
            if full_cnv_calls:
                first_cnv = full_cnv_calls[0][2]
                target_u = first_cnv.unit_tokens
                k_ref = first_cnv.k_ref
                unit_bp = first_cnv.unit_bp
                emitted_full_cnv_keys.add((rank_a + 1, rank_b, target_u))

                dev_by_contig_id = {ca.contig_idx: dev for ca, dev in devs_at_pair}
                cnv_calls_by_hap: dict[str, list[str]] = collections.defaultdict(list)
                for hap_id, ca_list in contigs_by_hap.items():
                    for ca in ca_list:
                        if ca.contig_idx in dev_by_contig_id:
                            d_seg = dev_by_contig_id[ca.contig_idx]
                            p_alt = _extract_full_period(d_seg.branch_steps)
                            if p_alt is not None and p_alt[0] == target_u:
                                cnv_calls_by_hap[hap_id].append(str(p_alt[1]))
                            else:
                                cnv_calls_by_hap[hap_id].append(".")
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b) and all(
                            r in ca.covered_ranks for r in range(rank_a + 1, rank_b)
                        ):
                            cnv_calls_by_hap[hap_id].append(str(k_ref))
                        else:
                            cnv_calls_by_hap[hap_id].append(".")

                cnv_gt = _resolve_haplotype_genotypes(
                    graph.all_haplotypes, ref_hap_id, str(k_ref), cnv_calls_by_hap
                )
                n_car = sum(
                    1 for g in cnv_gt.values() if g not in (".", str(k_ref))
                )
                n_cal = sum(1 for g in cnv_gt.values() if g != ".")
                has_mh = any(bool(dev.multi_hit_nodes) for _, dev, _ in full_cnv_calls)
                mh_tag = ",multi_hit=1" if has_mh else ""
                unit_str = ",".join(f"{nid}{o}" for nid, o in target_u)
                alt_counts: dict[int, list[str]] = collections.defaultdict(list)
                for h_id, g_val in sorted(cnv_gt.items()):
                    if g_val not in (".", str(k_ref)):
                        alt_counts[int(g_val)].append(h_id)
                if not alt_counts:
                    alt_counts[first_cnv.k_alt].append(
                        full_cnv_calls[0][0].raw_fragment.hap_id
                    )
                allele_parts: list[str] = []
                for a_i, (k_val, h_list) in enumerate(
                    sorted(alt_counts.items(), key=lambda x: (-len(x[1]), x[0])),
                    start=1,
                ):
                    allele_parts.append(
                        f"{a_i}:len={k_val * unit_bp},src={h_list[0]},n={len(h_list)},"
                        f"unit={unit_str},k_ref={k_ref},k_alt={k_val}{mh_tag}"
                    )

                cnv_ev = SVEvent(
                    chrom=graph.ref_chrom,
                    pan_start=0,
                    pan_end=0,
                    event_id="",
                    event_type="CNV",
                    ref_start=ref_s_open,
                    ref_end=ref_e_open,
                    anchors=f"{rank_a}..{rank_b}",
                    alleles_str=";".join(allele_parts),
                    n_carrier=n_car,
                    n_called=n_cal,
                    region_id=".",
                    genotypes=cnv_gt,
                    pieces=(),
                )
                pending_events.append(
                    _PendingNormalEvent(
                        locus_start=ref_s_open,
                        locus_end=ref_e_open,
                        event=cnv_ev,
                    )
                )

            del_intervals: dict[
                tuple[int, int], list[tuple[_ContigAnalysis, DeviationSegment]]
            ] = collections.defaultdict(list)
            partial_cnv_intervals: dict[
                tuple[int, int, tuple[tuple[str, str], ...]],
                list[tuple[_ContigAnalysis, DeviationSegment, CNVCall]],
            ] = collections.defaultdict(list)

            is_fast_b = (
                isinstance(b_slice_steps, LazyStepTuple)
                and graph._ref_cum_offsets is not None
            )
            if is_fast_b:
                assert isinstance(b_slice_steps, LazyStepTuple)
                assert graph._ref_cum_offsets is not None
                b_codes_abs = np.abs(b_slice_steps._codes)
                b_u_nids = np.unique(b_codes_abs)
                ref_cum = graph._ref_cum_offsets
            else:
                b_codes_abs = np.empty(0, dtype=np.int32)
                b_u_nids = np.empty(0, dtype=np.int32)
                ref_cum = np.empty(0, dtype=np.int64)

            b_nids_str: set[str] | None = None

            for d_idx, (ca, dev) in enumerate(devs_at_pair):
                if d_idx in full_cnv_dev_indices or _shares_canonical_tandem_unit(
                    b_slice_steps, dev.branch_steps
                ):
                    continue

                if is_fast_b and isinstance(dev.branch_steps, LazyStepTuple):
                    assert isinstance(b_slice_steps, LazyStepTuple)
                    br_codes_abs = np.abs(dev.branch_steps._codes)
                    br_u_nids = np.unique(br_codes_abs)
                    shared_int = np.intersect1d(
                        b_u_nids, br_u_nids, assume_unique=True
                    )
                    if int(shared_int.shape[0]) == 0:
                        del_bp = ref_e_open - ref_s_open
                        if del_bp >= config.min_sv_len:
                            del_intervals[(rank_a + 1, rank_b)].append(
                                (ca, dev)
                            )
                    else:
                        b_is_shared = np.isin(b_codes_abs, shared_int)
                        unsh_rel = np.flatnonzero(~b_is_shared)
                        if int(unsh_rel.shape[0]) > 0:
                            gaps_u = np.flatnonzero(np.diff(unsh_rel) > 1)
                            n_u_runs = int(gaps_u.shape[0]) + 1
                            u_starts = np.empty(n_u_runs, dtype=np.int32)
                            u_ends = np.empty(n_u_runs, dtype=np.int32)
                            u_starts[0] = unsh_rel[0]
                            if int(gaps_u.shape[0]) > 0:
                                u_starts[1:] = unsh_rel[gaps_u + 1]
                                u_ends[:-1] = unsh_rel[gaps_u] + 1
                            u_ends[-1] = unsh_rel[-1] + 1
                            for u_s_rel, u_e_rel in zip(
                                u_starts.tolist(), u_ends.tolist()
                            ):
                                d_lo = rank_a + 1 + u_s_rel
                                d_hi = rank_a + 1 + u_e_rel
                                if (
                                    int(ref_cum[d_hi]) - int(ref_cum[d_lo])
                                    >= config.min_sv_len
                                ):
                                    del_intervals[(d_lo, d_hi)].append(
                                        (ca, dev)
                                    )

                        sh_rel = np.flatnonzero(b_is_shared)
                        n_sh = int(sh_rel.shape[0])
                        if (
                            n_sh > 0
                            and int(sh_rel[-1]) - int(sh_rel[0]) + 1 == n_sh
                        ):
                            b_sh_lazy = LazyStepTuple(
                                b_slice_steps._codes[
                                    int(sh_rel[0]) : int(sh_rel[-1]) + 1
                                ]
                            )
                            br_is_sh = np.isin(br_codes_abs, shared_int)
                            br_sh_rel = np.flatnonzero(br_is_sh)
                            n_br_sh = int(br_sh_rel.shape[0])
                            if (
                                n_br_sh > 0
                                and int(br_sh_rel[-1]) - int(br_sh_rel[0]) + 1
                                == n_br_sh
                            ):
                                br_sh_lazy = LazyStepTuple(
                                    dev.branch_steps._codes[
                                        int(br_sh_rel[0]) : int(br_sh_rel[-1]) + 1
                                    ]
                                )
                                sh_cnv = detect_canonical_cnv(
                                    b_sh_lazy,
                                    br_sh_lazy,
                                    graph.nodes,
                                    config.min_sv_len,
                                )
                                if sh_cnv is not None:
                                    p_key = (
                                        rank_a + 1 + int(sh_rel[0]),
                                        rank_a + 1 + int(sh_rel[-1]) + 1,
                                        sh_cnv.unit_tokens,
                                    )
                                    if p_key not in emitted_full_cnv_keys:
                                        partial_cnv_intervals[p_key].append(
                                            (ca, dev, sh_cnv)
                                        )
                else:
                    if b_nids_str is None:
                        if isinstance(b_slice_steps, LazyStepTuple):
                            b_nids_str = {
                                str(x)
                                for x in np.unique(
                                    np.abs(b_slice_steps._codes)
                                ).tolist()
                            }
                        else:
                            b_nids_str = {
                                st.node_id
                                for st in b_slice_steps
                                if _is_full_step(st)
                            }

                    if isinstance(dev.branch_steps, LazyStepTuple):
                        br_nids = {
                            str(x)
                            for x in np.unique(
                                np.abs(dev.branch_steps._codes)
                            ).tolist()
                        }
                    else:
                        br_nids = {
                            st.node_id
                            for st in dev.branch_steps
                            if _is_full_step(st)
                        }
                    shared_nids = b_nids_str & br_nids

                    if not shared_nids:
                        del_bp = ref_e_open - ref_s_open
                        if del_bp >= config.min_sv_len:
                            del_intervals[(rank_a + 1, rank_b)].append(
                                (ca, dev)
                            )
                    else:
                        cur_del_start: int | None = None
                        for offset_idx, b_st in enumerate(b_slice_steps):
                            r_step = rank_a + 1 + offset_idx
                            if b_st.node_id not in shared_nids:
                                if cur_del_start is None:
                                    cur_del_start = r_step
                            else:
                                if cur_del_start is not None:
                                    d_s = graph.ref_step_ranges[
                                        cur_del_start
                                    ][0]
                                    d_e = graph.ref_step_ranges[
                                        r_step - 1
                                    ][1]
                                    if d_e - d_s >= config.min_sv_len:
                                        del_intervals[
                                            (cur_del_start, r_step)
                                        ].append((ca, dev))
                                    cur_del_start = None
                        if cur_del_start is not None:
                            d_s = graph.ref_step_ranges[cur_del_start][0]
                            d_e = graph.ref_step_ranges[rank_b - 1][1]
                            if d_e - d_s >= config.min_sv_len:
                                del_intervals[
                                    (cur_del_start, rank_b)
                                ].append((ca, dev))

                        b_shared_indices = [
                            rank_a + 1 + i
                            for i, st in enumerate(b_slice_steps)
                            if st.node_id in shared_nids
                        ]
                        if (
                            b_shared_indices
                            and b_shared_indices[-1]
                            - b_shared_indices[0]
                            + 1
                            == len(b_shared_indices)
                        ):
                            br_shared_indices = [
                                i
                                for i, st in enumerate(dev.branch_steps)
                                if _is_full_step(st)
                                and st.node_id in shared_nids
                            ]
                            if (
                                br_shared_indices
                                and br_shared_indices[-1]
                                - br_shared_indices[0]
                                + 1
                                == len(br_shared_indices)
                            ):
                                b_sh_steps = [
                                    st
                                    for st in b_slice_steps
                                    if st.node_id in shared_nids
                                ]
                                br_sh_steps = list(
                                    dev.branch_steps[
                                        br_shared_indices[0] : br_shared_indices[
                                            -1
                                        ]
                                        + 1
                                    ]
                                )
                                sh_cnv = detect_canonical_cnv(
                                    b_sh_steps,
                                    br_sh_steps,
                                    graph.nodes,
                                    config.min_sv_len,
                                )
                                if sh_cnv is not None:
                                    p_key = (
                                        b_shared_indices[0],
                                        b_shared_indices[-1] + 1,
                                        sh_cnv.unit_tokens,
                                    )
                                    if p_key not in emitted_full_cnv_keys:
                                        partial_cnv_intervals[p_key].append(
                                            (ca, dev, sh_cnv)
                                        )

            for (del_r_lo, del_r_hi), del_entries in sorted(
                del_intervals.items()
            ):
                d_ref_s = graph.ref_step_ranges[del_r_lo][0]
                d_ref_e = graph.ref_step_ranges[del_r_hi - 1][1]
                d_len = d_ref_e - d_ref_s
                del_contig_ids = {ca.contig_idx for ca, _ in del_entries}
                del_calls: dict[str, list[str]] = collections.defaultdict(list)
                for hap_id, ca_list in contigs_by_hap.items():
                    for ca in ca_list:
                        if ca.contig_idx in del_contig_ids:
                            del_calls[hap_id].append("1")
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
                            del_calls[hap_id].append("0")
                        else:
                            del_calls[hap_id].append(".")
                del_gt = _resolve_haplotype_genotypes(
                    graph.all_haplotypes, ref_hap_id, "0", del_calls
                )
                del_carriers = sorted(h for h, g in del_gt.items() if g == "1")
                if not del_carriers:
                    del_carriers = sorted(
                        {ca.raw_fragment.hap_id for ca, _ in del_entries}
                    )
                n_car = sum(1 for g in del_gt.values() if g not in ("0", "."))
                n_cal = sum(1 for g in del_gt.values() if g != ".")
                has_mh = any(
                    bool(dev.multi_hit_nodes) or ca.lis_multi_hit
                    for ca, dev in del_entries
                )
                mh_tag = ",multi_hit=1" if has_mh else ""
                del_ev = SVEvent(
                    chrom=graph.ref_chrom,
                    pan_start=0,
                    pan_end=0,
                    event_id="",
                    event_type="DEL",
                    ref_start=d_ref_s,
                    ref_end=d_ref_e,
                    anchors=f"{del_r_lo - 1}..{del_r_hi}",
                    alleles_str=f"1:len={d_len},src={del_carriers[0]},n={len(del_carriers)}{mh_tag}",
                    n_carrier=n_car,
                    n_called=n_cal,
                    region_id=".",
                    genotypes=del_gt,
                    pieces=(),
                )
                pending_events.append(
                    _PendingNormalEvent(
                        locus_start=d_ref_s,
                        locus_end=d_ref_e,
                        event=del_ev,
                    )
                )

            for (
                cnv_r_lo,
                cnv_r_hi,
                u_toks,
            ), p_cnv_entries in sorted(partial_cnv_intervals.items()):
                c_ref_s = graph.ref_step_ranges[cnv_r_lo][0]
                c_ref_e = graph.ref_step_ranges[cnv_r_hi - 1][1]
                first_c = p_cnv_entries[0][2]
                k_ref = first_c.k_ref
                unit_bp = first_c.unit_bp
                carrier_map = {
                    ca.contig_idx: c_call.k_alt for ca, _, c_call in p_cnv_entries
                }
                p_cnv_calls: dict[str, list[str]] = collections.defaultdict(list)
                for hap_id, ca_list in contigs_by_hap.items():
                    for ca in ca_list:
                        if ca.contig_idx in carrier_map:
                            p_cnv_calls[hap_id].append(str(carrier_map[ca.contig_idx]))
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b) and all(
                            r in ca.covered_ranks for r in range(cnv_r_lo, cnv_r_hi)
                        ):
                            p_cnv_calls[hap_id].append(str(k_ref))
                        else:
                            p_cnv_calls[hap_id].append(".")
                p_cnv_gt = _resolve_haplotype_genotypes(
                    graph.all_haplotypes, ref_hap_id, str(k_ref), p_cnv_calls
                )
                n_car = sum(
                    1 for g in p_cnv_gt.values() if g not in (".", str(k_ref))
                )
                n_cal = sum(1 for g in p_cnv_gt.values() if g != ".")
                has_mh = any(
                    bool(dev.multi_hit_nodes) or ca.lis_multi_hit
                    for ca, dev, _ in p_cnv_entries
                )
                mh_tag = ",multi_hit=1" if has_mh else ""
                unit_str = ",".join(f"{nid}{o}" for nid, o in u_toks)
                p_alt_counts: dict[int, list[str]] = collections.defaultdict(list)
                for h_id, g_val in sorted(p_cnv_gt.items()):
                    if g_val not in (".", str(k_ref)):
                        p_alt_counts[int(g_val)].append(h_id)
                if not p_alt_counts:
                    p_alt_counts[first_c.k_alt].append(
                        p_cnv_entries[0][0].raw_fragment.hap_id
                    )
                p_allele_parts: list[str] = []
                for a_i, (k_val, h_list) in enumerate(
                    sorted(p_alt_counts.items(), key=lambda x: (-len(x[1]), x[0])),
                    start=1,
                ):
                    p_allele_parts.append(
                        f"{a_i}:len={k_val * unit_bp},src={h_list[0]},n={len(h_list)},"
                        f"unit={unit_str},k_ref={k_ref},k_alt={k_val}{mh_tag}"
                    )
                p_cnv_ev = SVEvent(
                    chrom=graph.ref_chrom,
                    pan_start=0,
                    pan_end=0,
                    event_id="",
                    event_type="CNV",
                    ref_start=c_ref_s,
                    ref_end=c_ref_e,
                    anchors=f"{cnv_r_lo - 1}..{cnv_r_hi}",
                    alleles_str=";".join(p_allele_parts),
                    n_carrier=n_car,
                    n_called=n_cal,
                    region_id=".",
                    genotypes=p_cnv_gt,
                    pieces=(),
                )
                pending_events.append(
                    _PendingNormalEvent(
                        locus_start=c_ref_s,
                        locus_end=c_ref_e,
                        event=p_cnv_ev,
                    )
                )

        # (iii) INS event & InsertionSlotBlock when G(a, b) has >= 1 non-backbone representative
        if sol is not None and sol.non_backbone_reps:
            ins_ref_pos = graph.ref_step_ranges[rank_b][0]

            cands_by_contig: dict[int, list[ExtractedCandidate]] = (
                collections.defaultdict(list)
            )
            for kc in guard_res.kept_candidates:
                cands_by_contig[kc.contig_idx].append(kc)

            SignatureType = tuple[
                tuple[int, ...],
                tuple[tuple[Any, tuple[int, ...]], ...],
            ]
            contig_sig_by_hap: dict[
                str, list[SignatureType | Literal["0", "."]]
            ] = collections.defaultdict(list)
            sig_reconstructed_len: dict[SignatureType, int] = {}
            sig_exemplar_hap: dict[SignatureType, str] = {}
            sig_has_partial: dict[SignatureType, bool] = {}
            sig_has_rep_res: dict[SignatureType, bool] = {}
            sig_has_mh: dict[SignatureType, bool] = {}
            sig_cand_ids: dict[SignatureType, set[str]] = collections.defaultdict(set)

            dev_by_contig_at_pair = {
                ca_d.contig_idx: tdev_d for ca_d, tdev_d in devs_at_pair
            }

            for hap_id, ca_list in contigs_by_hap.items():
                for ca in ca_list:
                    ckey = ca.contig_idx
                    c_list = sorted(
                        cands_by_contig.get(ckey, ()),
                        key=lambda x: x.source_order,
                    )
                    if c_list:
                        top_reps: list[int] = []
                        kept_top_cids: list[str] = []
                        nested_all: list[tuple[Any, tuple[int, ...]]] = []
                        total_rec_len = 0
                        had_subtracted_rep = False
                        for kc in c_list:
                            r_idx = sol.cand_to_rep_idx.get(kc.candidate_id, 0)
                            if r_idx > 0:
                                kc_rec_len = sol.cand_to_reconstructed_len.get(
                                    kc.candidate_id,
                                    sol.non_backbone_reps[r_idx - 1].length_bp_no_n,
                                )
                                if config.dedup_mode != "off" and kc_rec_len <= 0:
                                    had_subtracted_rep = True
                                    continue
                                top_reps.append(r_idx)
                                kept_top_cids.append(kc.candidate_id)
                                nested_all.extend(
                                    sol.cand_to_nested_sig.get(kc.candidate_id, ())
                                )
                                total_rec_len += kc_rec_len
                            elif r_idx == -1:
                                had_subtracted_rep = True
                        if top_reps:
                            sig_val: SignatureType = (
                                tuple(top_reps),
                                tuple(nested_all),
                            )
                            contig_sig_by_hap[hap_id].append(sig_val)
                            if config.dedup_mode != "off":
                                prev_rec = sig_reconstructed_len.get(sig_val)
                                if prev_rec is None or total_rec_len < prev_rec:
                                    sig_reconstructed_len[sig_val] = total_rec_len
                            else:
                                sig_reconstructed_len.setdefault(sig_val, total_rec_len)
                            if (
                                sig_val not in sig_exemplar_hap
                                or hap_id < sig_exemplar_hap[sig_val]
                            ):
                                sig_exemplar_hap[sig_val] = hap_id
                            if (
                                ckey in guard_res.contig_had_masked_repeat
                                or ckey in guard_res.contig_had_subtracted_homolog
                                or had_subtracted_rep
                            ):
                                sig_has_partial[sig_val] = True
                            sig_cand_ids[sig_val].update(kept_top_cids)
                            for kc in c_list:
                                f_part, f_rep, f_mh = sol.cand_to_flags.get(
                                    kc.candidate_id,
                                    (
                                        kc.partial_content,
                                        kc.repeat_residual,
                                        kc.multi_hit,
                                    ),
                                )
                                if kc.partial_content or f_part:
                                    sig_has_partial[sig_val] = True
                                if kc.repeat_residual or f_rep:
                                    sig_has_rep_res[sig_val] = True
                                if kc.multi_hit or f_mh:
                                    sig_has_mh[sig_val] = True
                        elif (
                            ckey in guard_res.contig_had_masked_repeat
                            or ckey in guard_res.contig_had_subtracted_homolog
                            or had_subtracted_rep
                        ):
                            contig_sig_by_hap[hap_id].append(".")
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
                            tdev_here = dev_by_contig_at_pair.get(ca.contig_idx)
                            c_list_here = [
                                kc
                                for kc in c_list
                                if guard_res.cand_dev_loc.get(
                                    kc.candidate_id, (-1, -1)
                                )[0]
                                == ca.contig_idx
                            ]
                            if tdev_here is not None and c_list_here:
                                is_ref_like = _case_a_rep0_matches_backbone(
                                    tdev_here,
                                    b_slice_steps,
                                    c_list_here,
                                    guard_res.cand_dev_loc,
                                    sol.rep0_cand_spans,
                                )
                            else:
                                is_ref_like = (
                                    _uncovered_ref_bp_open(ca, rank_a, rank_b)
                                    < config.min_sv_len
                                    and _dev_branch_bp_in_span(
                                        ca.contig_idx, rank_a, rank_b
                                    )
                                    < config.min_sv_len
                                )
                            contig_sig_by_hap[hap_id].append(
                                "0" if is_ref_like else "."
                            )
                        else:
                            contig_sig_by_hap[hap_id].append(".")
                    elif (
                        ckey in guard_res.contig_had_masked_repeat
                        or ckey in guard_res.contig_had_subtracted_homolog
                    ):
                        contig_sig_by_hap[hap_id].append(".")
                    elif (
                        _contig_covers_anchor_pair(ca, rank_a, rank_b)
                        and _uncovered_ref_bp_open(ca, rank_a, rank_b)
                        < config.min_sv_len
                        and _dev_branch_bp_in_span(
                            ca.contig_idx, rank_a, rank_b
                        )
                        < config.min_sv_len
                    ):
                        contig_sig_by_hap[hap_id].append("0")
                    else:
                        contig_sig_by_hap[hap_id].append(".")

            hap_resolved_sig: dict[str, SignatureType | Literal["0", "."]] = {
                ref_hap_id: "0"
            }
            sig_carrier_haps: dict[SignatureType, list[str]] = collections.defaultdict(
                list
            )
            for hap_id in graph.all_haplotypes:
                if hap_id == ref_hap_id:
                    continue
                raw_calls = [
                    c for c in contig_sig_by_hap.get(hap_id, ()) if c != "."
                ]
                if not raw_calls:
                    hap_resolved_sig[hap_id] = "."
                elif len(set(raw_calls)) == 1:
                    chosen = raw_calls[0]
                    hap_resolved_sig[hap_id] = chosen
                    if chosen != "0":
                        sig_carrier_haps[chosen].append(hap_id)  # type: ignore[index]
                else:
                    hap_resolved_sig[hap_id] = "."

            all_observed_sigs = set(sig_reconstructed_len.keys())
            ordered_sigs = sorted(
                all_observed_sigs,
                key=lambda s: (
                    -len(sig_carrier_haps.get(s, ())),
                    s[0],
                    tuple((_nested_key_sort_key(k), v) for k, v in s[1]),
                ),
            )
            sig_to_allele_idx = {
                s: idx_a for idx_a, s in enumerate(ordered_sigs, start=1)
            }

            ins_genotypes: dict[str, str] = {}
            for hap_id in graph.all_haplotypes:
                res_val = hap_resolved_sig.get(hap_id, ".")
                if res_val in ("0", "."):
                    ins_genotypes[hap_id] = str(res_val)
                else:
                    ins_genotypes[hap_id] = str(sig_to_allele_idx[res_val])  # type: ignore[index]

            allele_entries: list[str] = []
            for s in ordered_sigs:
                a_idx = sig_to_allele_idx[s]
                a_len = sig_reconstructed_len[s]
                c_haps = sorted(sig_carrier_haps.get(s, ()))
                ex_h = c_haps[0] if c_haps else sig_exemplar_hap[s]
                n_h = len(c_haps)
                tags = ""
                if sig_has_partial.get(s, False):
                    tags += ",partial_content=1"
                if sig_has_rep_res.get(s, False):
                    tags += ",repeat_residual=1"
                if sig_has_mh.get(s, False):
                    tags += ",multi_hit=1"
                allele_entries.append(
                    f"{a_idx}:len={a_len},src={ex_h},n={n_h}{tags}"
                )

            slot_reps_pieces: list[tuple[Piece, ...]] = []
            flat_group_pieces: list[Piece] = []
            piece_seq_counter = 1

            for rep in sol.non_backbone_reps:
                rep_alleles = [
                    sig_to_allele_idx[s]
                    for s in ordered_sigs
                    if rep.rep_idx in s[0]
                ]
                if not rep_alleles:
                    rep_alleles = [1]

                updated_rep_pieces: list[Piece] = []
                for p_i, p in enumerate(rep.pieces):
                    p_cands = (
                        rep.piece_cand_ids[p_i]
                        if p_i < len(rep.piece_cand_ids)
                        else frozenset()
                    )
                    matching_alleles = sorted(
                        {
                            sig_to_allele_idx[s]
                            for s in ordered_sigs
                            if rep.rep_idx in s[0]
                            and (not p_cands or bool(p_cands & sig_cand_ids.get(s, set())))
                        }
                    )
                    if not matching_alleles:
                        matching_alleles = sorted(set(rep_alleles))
                    alleles_tup = tuple(matching_alleles)
                    up_p = replace(
                        p,
                        allele_idx=alleles_tup[0],
                        piece_idx=piece_seq_counter,
                        alleles=alleles_tup,
                    )
                    piece_seq_counter += 1
                    updated_rep_pieces.append(up_p)
                    flat_group_pieces.append(up_p)

                slot_reps_pieces.append(tuple(updated_rep_pieces))

            n_car = sum(
                1 for g in ins_genotypes.values() if g not in ("0", ".")
            )
            n_cal = sum(1 for g in ins_genotypes.values() if g != ".")

            ins_ev = SVEvent(
                chrom=graph.ref_chrom,
                pan_start=0,
                pan_end=0,
                event_id="",
                event_type="INS",
                ref_start=ins_ref_pos,
                ref_end=ins_ref_pos,
                anchors=f"{rank_a}..{rank_b}",
                alleles_str=";".join(allele_entries),
                n_carrier=n_car,
                n_called=n_cal,
                region_id=".",
                genotypes=ins_genotypes,
                pieces=tuple(flat_group_pieces),
            )
            slot_blk = InsertionSlotBlock(
                right_anchor_ref_step=rank_b,
                is_centromere_island=False,
                group_span=rank_b - rank_a,
                left_anchor_ref_step=rank_a,
                event_id="",
                representatives=tuple(slot_reps_pieces),
                trailing_spacer=False,
            )
            pending_events.append(
                _PendingNormalEvent(
                    locus_start=ref_s_open,
                    locus_end=ref_e_open,
                    event=ins_ev,
                    slot_block=slot_blk,
                    guard_pair=pair,
                )
            )

    # 7. Group co-located/overlapping normal events into shared `region_id`s and assign `event_id`s
    pending_events.sort(
        key=lambda pe: (
            pe.locus_start,
            pe.locus_end,
            pe.event.ref_start,
            pe.event.ref_end,
            pe.event.event_type,
            pe.event.anchors,
        )
    )

    reg_counter = 1
    cur_reg_end = -1
    cur_reg_id = ""
    type_counters: dict[str, int] = collections.defaultdict(int)

    normal_sv_events: list[SVEvent] = []
    slot_blocks_by_step: dict[int, list[InsertionSlotBlock]] = collections.defaultdict(
        list
    )
    for c_blk in cen_out.island_slot_blocks:
        slot_blocks_by_step[c_blk.right_anchor_ref_step].append(c_blk)

    for pe in pending_events:
        if not cur_reg_id or pe.locus_start > cur_reg_end:
            cur_reg_id = f"{graph.ref_chrom}_REG_{reg_counter}"
            reg_counter += 1
            cur_reg_end = pe.locus_end
        else:
            cur_reg_end = max(cur_reg_end, pe.locus_end)

        ev_type = pe.event.event_type
        type_counters[ev_type] += 1
        ev_id = f"{graph.ref_chrom}_{ev_type}_{type_counters[ev_type]}"

        final_ev = replace(
            pe.event,
            event_id=ev_id,
            region_id=cur_reg_id,
        )
        normal_sv_events.append(final_ev)

        if pe.slot_block is not None:
            upd_blk = replace(
                pe.slot_block,
                event_id=ev_id,
            )
            slot_blocks_by_step[upd_blk.right_anchor_ref_step].append(upd_blk)

        for idx_u in pe.chain_unres_indices:
            unresolved_records[idx_u] = replace(
                unresolved_records[idx_u],
                region_id=cur_reg_id,
                event_id=ev_id,
            )

        if pe.guard_pair is not None:
            for idx_u in guard_unres_indices_by_pair.get(pe.guard_pair, ()):
                unresolved_records[idx_u] = replace(
                    unresolved_records[idx_u],
                    region_id=cur_reg_id,
                    event_id=ev_id,
                )
            g_res = group_guard_by_pair[pe.guard_pair]
            for j_entry in g_res.json_log_entries:
                j_entry["region_id"] = cur_reg_id
                if j_entry.get("final_reason") == "ACCEPTED_CANDIDATE":
                    j_entry["representative_event_id"] = ev_id

    all_sv_events_unsorted: list[SVEvent] = (
        list(cen_out.island_sv_events) + normal_sv_events
    )

    pan_seq, step_offsets, backfilled_events, inserts_rows = assemble_pan_and_offsets(
        graph=graph,
        slot_blocks_by_step=slot_blocks_by_step,
        sv_events=all_sv_events_unsorted,
        spacer_n=config.spacer_n,
    )

    sorted_sv_events = tuple(
        sorted(
            backfilled_events,
            key=lambda ev: (
                ev.ref_start,
                ev.ref_end,
                0 if ev.event_type == "CEN_ISLAND" else 1,
                ev.pan_start,
                ev.pan_end,
                ev.event_id,
            ),
        )
    )

    summary_metrics: dict[str, Any] = {
        **cen_out.summary_metrics,
        "chrom": graph.ref_chrom,
        "ref_path": graph.ref_fragment.src_path,
        "ref_hap": graph.ref_fragment.hap_id,
        "gfa_path": str(graph.gfa_path),
        "gfa_sha256": graph.gfa_sha256,
        "ref_len": len(graph.ref_seq),
        "pan_len": len(pan_seq),
        "haplotype_count": len(graph.all_haplotypes),
        "sv_total_count": len(sorted_sv_events),
        "ins_count": sum(1 for e in sorted_sv_events if e.event_type == "INS"),
        "del_count": sum(1 for e in sorted_sv_events if e.event_type == "DEL"),
        "cnv_count": sum(1 for e in sorted_sv_events if e.event_type == "CNV"),
        "inv_count": sum(
            1
            for e in sorted_sv_events
            if e.event_type in ("INV", "INV_DISPLACED", "INV_EDGE", "INV_CONFLICT")
        ),
        "dup_tra_count": sum(1 for e in sorted_sv_events if e.event_type == "DUP_TRA"),
        "large_jump_count": sum(
            1 for e in sorted_sv_events if e.event_type == "LARGE_JUMP"
        ),
        "cen_island_count": sum(
            1 for e in sorted_sv_events if e.event_type == "CEN_ISLAND"
        ),
        "insert_piece_count": len(inserts_rows),
        "unresolved_count": len(unresolved_records),
        "repeat_window_size": REPEAT_WINDOW_SIZE,
        "repeat_window_step": REPEAT_WINDOW_STEP,
        "repeat_kmer_size": REPEAT_KMER_SIZE,
        "repeat_singleton_threshold": REPEAT_SINGLETON_THRESHOLD,
        "repeat_valid_pos_ratio": REPEAT_VALID_POS_RATIO,
        "repeat_min_seq_len": REPEAT_MIN_SEQ_LEN,
        "ref_sub_block_size": REF_SUB_BLOCK_SIZE,
        "ref_sub_min_aligned_ratio": REF_SUB_MIN_ALIGNED_RATIO,
        "ref_sub_min_identity": REF_SUB_MIN_IDENTITY,
        "dedup_mode": config.dedup_mode,
        "dedup_id": config.dedup_id,
        "dedup_ref_masked_bp": dedup_ref_masked_bp,
        "dedup_pan_masked_bp": dedup_pan_masked_bp,
        "dedup_short_residual_bp": dedup_short_residual_bp,
        "dedup_dropped_groups": dedup_dropped_groups,
    }

    log_progress(
        f"  Linearization complete ({time.monotonic() - t_pipe:.2f}s): "
        f"pan_len={len(pan_seq):,} bp (+{len(pan_seq) - len(graph.ref_seq):,} bp), "
        f"events={len(sorted_sv_events):,}, pieces={len(inserts_rows):,}, "
        f"unresolved={len(unresolved_records):,}"
    )

    return ChromosomeBuildResult(
        chrom=graph.ref_chrom,
        pan_seq=pan_seq,
        step_offsets=step_offsets,
        sv_events=sorted_sv_events,
        all_haplotypes=graph.all_haplotypes,
        inserts_rows=inserts_rows,
        unresolved_records=tuple(unresolved_records),
        centromere_rows=tuple(centromere_rows),
        lifted_gff_lines=(),
        summary_metrics=summary_metrics,
        log_lines=tuple(log_lines),
        json_log_entries=tuple(json_log_entries),
        gfa_path=str(graph.gfa_path),
        gfa_sha256=graph.gfa_sha256,
        ref_fa_path=str(config.ref_fa),
    )


solve_chromosome = run_chromosome_pipeline
