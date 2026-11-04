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

import collections
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import AbstractSet, Any, Literal, Mapping, Sequence

import mappy

from panflat.centromere import (
    delimit_centromere_regions,
    partition_fragments_by_centromere,
    process_centromere_regions,
)
from panflat.gfa_io import (
    spell_steps,
    validate_build_config,
)
from panflat.models import (
    BuildConfig,
    CentromereRegion,
    CentromereRow,
    ChromosomeBuildResult,
    CNVCall,
    DeviationSegment,
    GFAGraph,
    InsertionSlotBlock,
    PathFragment,
    Piece,
    ProjectedAnchor,
    ReverseChainRecord,
    Step,
    StepOffsetRow,
    StepProvenance,
    SVEvent,
    UnresolvedRecord,
)
from panflat.normalizer import (
    _extract_full_period,
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
    REF_SUB_BLOCK_SIZE,
    REF_SUB_MIN_ALIGNED_RATIO,
    REF_SUB_MIN_IDENTITY,
    REPEAT_KMER_SIZE,
    REPEAT_MIN_SEQ_LEN,
    REPEAT_SINGLETON_THRESHOLD,
    REPEAT_VALID_POS_RATIO,
    REPEAT_WINDOW_SIZE,
    REPEAT_WINDOW_STEP,
    check_similarity_merge,
    complement_intervals,
    merge_contiguous_provenance,
    repeat_mask_intervals,
    slice_candidate_by_intervals,
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


@dataclass(frozen=True)
class SubproblemStateKey:
    """
    §3.5 Subproblem state `S = (B_id, [lo, hi), M)`.

    The progress metric `(|M|, hi - lo)` must strictly decrease lexicographically
    under every recursive transition (Rule a: local re-anchoring; Rule b: component recursion),
    and no descendant state may equal any ancestor state.
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


@dataclass
class GroupRepeatGuardResult:
    """Output of §3.6.6 repeat-guard and §3.4.3 cycle folding for a parallel branch group."""

    kept_candidates: list[ExtractedCandidate] = field(default_factory=list)
    unresolved_records: list[UnresolvedRecord] = field(default_factory=list)
    json_log_entries: list[dict[str, Any]] = field(default_factory=list)
    hap_had_masked_repeat: dict[str, bool] = field(default_factory=dict)
    hap_had_kept_candidate: dict[str, bool] = field(default_factory=dict)
    contig_had_masked_repeat: set[tuple[str, str]] = field(default_factory=set)
    contig_had_kept_candidate: set[tuple[str, str]] = field(default_factory=set)
    has_partial_content: bool = False


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
    total_len = sum(st.step_len(nodes) for st in steps)
    return PathFragment(
        sample=sample,
        hap=hap,
        hap_id=hap_id,
        seqid=src_path,
        src_path=src_path,
        base_offset=base_offset,
        steps=tuple(steps),
        provenance=tuple(provenance),
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
    p_ref = _extract_full_period(ref_steps)
    p_alt = _extract_full_period(alt_steps)
    if p_ref is None or p_alt is None:
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
    covered_ranks_out: set[int] | None = None,
) -> list[DeviationSegment]:
    """
    §3.5 Progress condition (a) (`F14`, `T16`): Local re-anchoring on `B(a, b)`.

    Projects `dev.branch_steps` onto `backbone_steps[lo_ab:hi_ab)` where `(lo_ab, hi_ab)` is
    the open backbone interval `(a, b)`. If `lo_ab..hi_ab` is a proper sub-interval of `lo..hi`
    (`hi_ab - lo_ab < hi - lo`), finds relative-forward (`+`) locally unique anchors, computes
    weighted LIS, and recursively refines each resulting sub-deviation.
    """
    eff_hi = len(backbone_steps) if hi is None else hi
    if dev.is_unanchored_start or dev.is_unanchored_end:
        return [dev]

    lo_ab = lo if dev.left_anchor_rank == "START" else int(dev.left_anchor_rank) + 1
    hi_ab = eff_hi if dev.right_anchor_rank == "END" else int(dev.right_anchor_rank)

    # Must be a non-empty proper sub-interval of [lo, eff_hi) to shrink hi - lo
    if hi_ab - lo_ab <= 0 or hi_ab - lo_ab >= eff_hi - lo:
        return [dev]

    if parent_key is None:
        parent_key = SubproblemStateKey(
            b_id=(("BACKBONE", 0, eff_hi, "+"),),
            lo=lo,
            hi=eff_hi,
            has_logical_start=(lo > 0),
            has_logical_end=(eff_hi < len(backbone_steps)),
            m_key=(make_provenance_key(dev.branch_provenance),),
        )

    child_key = SubproblemStateKey(
        b_id=parent_key.b_id,
        lo=lo_ab,
        hi=hi_ab,
        has_logical_start=True,
        has_logical_end=True,
        m_key=parent_key.m_key,
    )
    if not can_transition(parent_key, child_key, ancestor_keys):
        return [dev]

    if not dev.branch_steps:
        return [dev]

    sub_anchors = project_steps_onto_backbone(
        dev.branch_steps,
        dev.branch_provenance,
        backbone_steps,
        nodes,
        lo=lo_ab,
        hi=hi_ab,
    )
    fwd_anchors = [anc for anc in sub_anchors if anc.rel_orient == "+"]
    if not fwd_anchors:
        return [dev]

    sub_chain, _ = compute_weighted_lis(sub_anchors)
    if not sub_chain:
        return [dev]

    if covered_ranks_out is not None:
        for anc in sub_chain:
            covered_ranks_out.add(anc.rank_in_b)
        first_anc = sub_chain[0]
        if dev.branch_steps[0 : first_anc.branch_step_idx] == tuple(
            backbone_steps[lo_ab : first_anc.rank_in_b]
        ):
            for r in range(lo_ab, first_anc.rank_in_b + 1):
                covered_ranks_out.add(r)
        for k in range(len(sub_chain) - 1):
            a_l = sub_chain[k]
            a_r = sub_chain[k + 1]
            if dev.branch_steps[a_l.branch_step_idx + 1 : a_r.branch_step_idx] == tuple(
                backbone_steps[a_l.rank_in_b + 1 : a_r.rank_in_b]
            ):
                for r in range(a_l.rank_in_b, a_r.rank_in_b + 1):
                    covered_ranks_out.add(r)
        last_anc = sub_chain[-1]
        if dev.branch_steps[last_anc.branch_step_idx + 1 :] == tuple(
            backbone_steps[last_anc.rank_in_b + 1 : hi_ab]
        ):
            for r in range(last_anc.rank_in_b, hi_ab):
                covered_ranks_out.add(r)

    temp_frag = _make_temp_fragment(
        dev.hap_id, dev.src_path, dev.branch_steps, dev.branch_provenance, nodes
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

    next_ancestors = ancestor_keys | {parent_key}
    refined: list[DeviationSegment] = []
    for sdev in raw_sub_devs:
        mapped_left = (
            dev.left_anchor_rank
            if sdev.left_anchor_rank == "START"
            else sdev.left_anchor_rank
        )
        mapped_right = (
            dev.right_anchor_rank
            if sdev.right_anchor_rank == "END"
            else sdev.right_anchor_rank
        )
        merged_mh = tuple(sorted(set(dev.multi_hit_nodes) | set(sdev.multi_hit_nodes)))
        mapped_dev = DeviationSegment(
            hap_id=dev.hap_id,
            src_path=dev.src_path,
            left_anchor_rank=mapped_left,
            right_anchor_rank=mapped_right,
            branch_start_idx=dev.branch_start_idx + sdev.branch_start_idx,
            branch_end_idx=dev.branch_start_idx + sdev.branch_end_idx,
            branch_steps=sdev.branch_steps,
            branch_provenance=sdev.branch_provenance,
            backbone_steps=sdev.backbone_steps,
            multi_hit_nodes=merged_mh,
            is_unanchored_start=False,
            is_unanchored_end=False,
        )
        refined.extend(
            reanchor_deviation(
                dev=mapped_dev,
                backbone_steps=backbone_steps,
                nodes=nodes,
                lo=lo_ab,
                hi=hi_ab,
                parent_key=child_key,
                ancestor_keys=next_ancestors,
                covered_ranks_out=covered_ranks_out,
            )
        )
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
) -> list[CandidateBlock]:
    """
    §3.1 & §3.5 (`F15`, `T4`, `T9`, `T11`, `T19`): Extract candidate blocks from a terminal deviation.

    - Top-level reference nodes (`node_id in global_ref_nodes`) are always strict separators and
      never enter any candidate block (`T9`, `T19`).
    - Single-copy current-backbone nodes (`count == 1` in `B[lo:hi)`) are also separators.
    - Multi-copy local non-reference backbone nodes (`count > 1` in `B[lo:hi)` or `B(a,b)` and
      `node_id not in global_ref_nodes`) embedded with at least one current-level new node are
      kept whole and flagged `repeat_residual = True` (`test_f15_05`, `test_f15_boundary_repeat_residual`).
    - When `B(a,b)` at top level contains multi-copy reference nodes that prevented decomposition
      (or were traversed by `dev.branch_steps`), extracted novel blocks also set `repeat_residual = True`.
    - Candidate blocks shorter than `min_sv_len` are discarded.
    """
    if is_top_level and (dev.is_unanchored_start or dev.is_unanchored_end):
        return []
    if not dev.branch_steps:
        return []

    eff_hi = len(backbone_steps) if hi is None else hi
    lo_ab = lo if dev.left_anchor_rank == "START" else int(dev.left_anchor_rank) + 1
    hi_ab = eff_hi if dev.right_anchor_rank == "END" else int(dev.right_anchor_rank)

    b_interval_counts = collections.Counter(
        st.node_id for st in backbone_steps[lo:eff_hi] if _is_full_step(st)
    )
    b_ab_counts = collections.Counter(
        st.node_id for st in backbone_steps[lo_ab:hi_ab] if _is_full_step(st)
    )
    b_interval_nodes = set(b_interval_counts.keys())
    multicopy_local_nonref_nodes = {
        nid
        for nid, cnt in b_interval_counts.items()
        if cnt > 1 and nid not in global_ref_nodes
    } | {
        nid
        for nid, cnt in b_ab_counts.items()
        if cnt > 1 and nid not in global_ref_nodes
    }
    singlecopy_local_nonref_nodes = (
        b_interval_nodes - set(global_ref_nodes) - multicopy_local_nonref_nodes
    )

    b_ab_has_multicopy = any(cnt > 1 for cnt in b_ab_counts.values())
    b_ab_has_no_unique = (hi_ab > lo_ab) and not bool(
        build_backbone_anchor_map(backbone_steps, lo=lo_ab, hi=hi_ab)
    )
    branch_touches_b_ab_multicopy = any(
        _is_full_step(st) and b_ab_counts.get(st.node_id, 0) > 1
        for st in dev.branch_steps
    )
    top_undecomposable_repeat = b_ab_has_multicopy and (
        b_ab_has_no_unique or branch_touches_b_ab_multicopy
    )

    def _is_separator(st: Step) -> bool:
        if st.node_id in global_ref_nodes:
            return True
        if _is_full_step(st) and st.node_id in singlecopy_local_nonref_nodes:
            return True
        return False

    runs: list[tuple[list[Step], list[StepProvenance]]] = []
    cur_steps: list[Step] = []
    cur_prov: list[StepProvenance] = []

    for st, pv in zip(dev.branch_steps, dev.branch_provenance):
        if _is_separator(st):
            if cur_steps:
                runs.append((cur_steps, cur_prov))
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
                runs.append((cur_steps, cur_prov))
                cur_steps = []
                cur_prov = []

        cur_steps.append(st)
        cur_prov.append(pv)

    if cur_steps:
        runs.append((cur_steps, cur_prov))

    candidates: list[CandidateBlock] = []
    order_idx = base_source_order
    for r_steps, r_prov in runs:
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

        has_embedded_local_multicopy = any(
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
            )
        )
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
) -> tuple[
    list[ExtractedCandidate],
    list[UnresolvedRecord],
    list[dict[str, Any]],
    bool,
    int,
]:
    """
    §3.6.6 & §3.4.3 (`F19`, `F12`, `C1`, `C16`–`C20`, `T17`):
    Screen a raw `CandidateBlock` with `repeat_mask_intervals` (when `config.repeat_guard == "on"`
    and `len(raw_cand.seq) >= 500`), slice kept/masked intervals, record `REPEAT_LIKE` in
    `unresolved_records`, and apply `fold_tandem_cycles` to each kept residual `>= config.min_sv_len`.

    Returns `(kept_candidates, unresolved_records, json_logs, had_masked_repeat, updated_next_source_order)`.
    """
    kept_out: list[ExtractedCandidate] = []
    unres_out: list[UnresolvedRecord] = []
    logs_out: list[dict[str, Any]] = []
    had_masked = False

    raw_len = len(raw_cand.seq)
    raw_contig_prov = merge_contiguous_provenance(raw_cand.provenance, raw_cand.seq)[0][0]
    if node_visit_counts is not None and raw_len > 0:
        rev_frac: float | None = (
            sum(
                st.step_len(graph.nodes)
                for st in raw_cand.steps
                if node_visit_counts.get(st.node_id, 0) > 1
            )
            / raw_len
        )
    else:
        rev_frac = 0.0

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

    if (
        not raw_cand.repeat_screened
        and config.repeat_guard == "on"
        and raw_len >= REPEAT_MIN_SEQ_LEN
    ):
        masked_intervals, window_diags = repeat_mask_intervals(raw_cand.seq)
        win_ratios: list[float | None] | None = (
            [d["singleton_ratio"] for d in window_diags] if window_diags else None
        )
        unmasked_intervals = complement_intervals(raw_len, masked_intervals)
        kept_slices, masked_slices = slice_candidate_by_intervals(
            steps=raw_cand.steps,
            provenance=raw_cand.provenance,
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
                            seq=m_seq,
                            export_to_fasta=config.dump_repeat_sequences,
                            exported_fasta_id=exp_id,
                        )
                    )

        if not kept_slices:
            logs_out.append(
                _make_log(
                    cand_id=raw_cand.candidate_id,
                    parent_id=None,
                    prov=raw_contig_prov,
                    length=raw_len,
                    revisited_fraction=rev_frac,
                    masked_ivs=masked_intervals,
                    win_ratios=win_ratios,
                    final_reason="REPEAT_LIKE",
                )
            )
            return kept_out, unres_out, logs_out, had_masked, next_source_order

        if masked_intervals:
            logs_out.append(
                _make_log(
                    cand_id=raw_cand.candidate_id,
                    parent_id=None,
                    prov=raw_contig_prov,
                    length=raw_len,
                    revisited_fraction=rev_frac,
                    masked_ivs=masked_intervals,
                    win_ratios=win_ratios,
                    final_reason="SPLIT_BY_REPEAT_MASK",
                )
            )

        for c_idx, (_, _, k_steps, k_prov, k_seq) in enumerate(kept_slices, start=1):
            k_len = len(k_seq)
            c_id = (
                raw_cand.candidate_id
                if not masked_intervals
                else f"{raw_cand.candidate_id}_c{c_idx}"
            )
            c_parent = None if not masked_intervals else raw_cand.candidate_id
            k_contig = merge_contiguous_provenance(k_prov, k_seq)[0][0]
            if node_visit_counts is not None and k_len > 0:
                k_rev_frac = (
                    sum(
                        st.step_len(graph.nodes)
                        for st in k_steps
                        if node_visit_counts.get(st.node_id, 0) > 1
                    )
                    / k_len
                )
            else:
                k_rev_frac = 0.0

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
                    partial_content=bool(masked_intervals),
                    repeat_residual=raw_cand.repeat_residual,
                    multi_hit=raw_cand.multi_hit,
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

        return kept_out, unres_out, logs_out, had_masked, next_source_order

    # Repeat-guard off or candidate < 500 bp (or already screened)
    if raw_len < config.min_sv_len:
        return kept_out, unres_out, logs_out, False, next_source_order

    fold_res = fold_tandem_cycles(
        raw_cand.steps, raw_cand.provenance, graph.global_ref_nodes, config.max_copy
    )
    folded_seq = spell_steps(fold_res.folded_steps, graph.nodes)
    if len(folded_seq) < config.min_sv_len:
        return kept_out, unres_out, logs_out, False, next_source_order

    kept_out.append(
        ExtractedCandidate(
            candidate_id=raw_cand.candidate_id,
            parent_id=None,
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
            partial_content=raw_cand.partial_content,
            repeat_residual=raw_cand.repeat_residual,
            multi_hit=raw_cand.multi_hit,
        )
    )
    logs_out.append(
        _make_log(
            cand_id=raw_cand.candidate_id,
            parent_id=None,
            prov=raw_contig_prov,
            length=len(folded_seq),
            revisited_fraction=rev_frac,
            masked_ivs=None,
            win_ratios=None,
            final_reason="ACCEPTED_CANDIDATE",
        )
    )
    next_source_order += 1
    return kept_out, unres_out, logs_out, False, next_source_order


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
    aligner: Any | None = None


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


def steps_to_pieces(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    piece_boundaries: Sequence[int],
    nodes: Mapping[str, str],
    members: Sequence[str],
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
                "source_contiguous" if len(pieces) == 0 else "synthetic"
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
                )
            )
        return fallback

    next_ancestors = (
        ancestor_keys | {parent_state_key} if parent_state_key is not None else ancestor_keys
    )

    b_steps = b_sec.steps
    b_prov = b_sec.exemplar.provenance
    n_b = len(b_steps)

    child_group_cands: dict[
        tuple[int | Literal["START"], int | Literal["END"]], list[ExtractedCandidate]
    ] = collections.defaultdict(list)
    member_cands_at_group: dict[
        tuple[int, tuple[int | Literal["START"], int | Literal["END"]]],
        list[ExtractedCandidate],
    ] = collections.defaultdict(list)
    member_covered_anchors: dict[int, set[int | Literal["START", "END"]]] = {}
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
            )
            next_child_order += len(raw_blocks)

            g_key = (tdev.left_anchor_rank, tdev.right_anchor_rank)
            for rb in raw_blocks:
                if rb.repeat_residual:
                    extra_repeat_residual = True
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
                        multi_hit=rb.multi_hit or parent_occ.multi_hit,
                    )
                    child_group_cands[g_key].append(ext_c)
                    if occ_idx == 0:
                        member_cands_at_group[(m_idx, g_key)].append(ext_c)

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

    all_comp_occurrences: list[ExtractedCandidate] = []
    for d in comp:
        all_comp_occurrences.extend(d.occurrences)
    all_comp_cand_ids = frozenset(occ.candidate_id for occ in all_comp_occurrences)
    comp_support_haps = set().union(*(d.support_haps for d in comp))

    nested_choices_by_cand_id: dict[
        str, tuple[tuple[Any, tuple[int, ...]], ...]
    ] = {}
    reconstructed_len_by_cand_id: dict[str, int] = {}
    rep_used_by_parent_cands: dict[tuple[Any, int], set[str]] = collections.defaultdict(
        set
    )

    for occ in b_sec.occurrences:
        choices_b: list[tuple[Any, tuple[int, ...]]] = []
        for g_key, _, _, _, sol in active_child_solutions:
            choices_b.append((g_key, (0,)))
        nested_choices_by_cand_id[occ.candidate_id] = tuple(choices_b)
        reconstructed_len_by_cand_id[occ.candidate_id] = len(b_sec.seq)

    for m_idx, d in enumerate(m_child):
        cov = member_covered_anchors.get(m_idx, {"START", "END"})
        d_choices: list[tuple[Any, tuple[int, ...]]] = []
        rec_len = len(b_sec.seq)

        for g_key, _, _, _, sol in active_child_solutions:
            a_r, b_r = g_key
            cands_here = member_cands_at_group.get((m_idx, g_key), [])
            if cands_here:
                rep_list: list[int] = []
                sub_nested_entries: list[tuple[Any, tuple[int, ...]]] = []
                rep_len_sum = 0
                for ch_c in sorted(cands_here, key=lambda x: x.source_order):
                    r_idx = sol.cand_to_rep_idx.get(ch_c.candidate_id, 0)
                    rep_list.append(r_idx)
                    if r_idx > 0:
                        for p_occ in d.occurrences:
                            rep_used_by_parent_cands[(g_key, r_idx)].add(
                                p_occ.candidate_id
                            )
                        rep_len_sum += sol.cand_to_reconstructed_len.get(
                            ch_c.candidate_id,
                            sol.non_backbone_reps[r_idx - 1].length_bp_no_n,
                        )
                    else:
                        rep_len_sum += len(sol.backbone_seq)
                    sub_sig = sol.cand_to_nested_sig.get(ch_c.candidate_id, ())
                    for sub_k, sub_v in sub_sig:
                        sub_nested_entries.append(((g_key, sub_k), sub_v))

                d_choices.append((g_key, tuple(rep_list)))
                d_choices.extend(sub_nested_entries)
                if rep_list != [0]:
                    rec_len += rep_len_sum - len(sol.backbone_seq)
            elif a_r in cov and b_r in cov:
                d_choices.append((g_key, (0,)))
            else:
                d_choices.append((g_key, (-1,)))

        for p_occ in d.occurrences:
            nested_choices_by_cand_id[p_occ.candidate_id] = tuple(d_choices)
            reconstructed_len_by_cand_id[p_occ.candidate_id] = max(
                len(d.seq), rec_len
            )

    if not active_child_solutions:
        b_pieces = tuple(
            replace(p, members=tuple(sorted(comp_support_haps))) for p in b_sec.pieces
        )
        return [
            ComponentResult(
                is_synthetic=True,
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

    b_sec_cut_set = set(b_sec.exemplar.piece_boundaries)
    comp_members_sorted = tuple(sorted(comp_support_haps))

    raw_out_pieces: list[Piece] = []
    raw_out_cands: list[frozenset[str]] = []
    cur_b_steps: list[Step] = []
    cur_b_prov: list[StepProvenance] = []
    cur_b_first_join: Literal["source_contiguous", "spacer", "synthetic"] = (
        "source_contiguous"
    )

    def _flush_b_run() -> None:
        nonlocal cur_b_steps, cur_b_prov, cur_b_first_join
        if not cur_b_steps:
            return
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
                    members=comp_members_sorted,
                )
            )
            raw_out_cands.append(all_comp_cand_ids)
        cur_b_steps = []
        cur_b_prov = []
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
                            "synthetic" if len(raw_out_pieces) > 0 else "source_contiguous"
                        )
                    used_by = frozenset(
                        rep_used_by_parent_cands.get(
                            (g_key, rep.rep_idx), set(all_comp_cand_ids)
                        )
                    )
                    for p_i, rp in enumerate(rep.pieces):
                        p_cands = (
                            used_by & rep.piece_cand_ids[p_i]
                            if p_i < len(rep.piece_cand_ids)
                            and (used_by & rep.piece_cand_ids[p_i])
                            else used_by
                        )
                        if p_i == 0:
                            raw_out_pieces.append(replace(rp, join_before=first_join))
                        else:
                            raw_out_pieces.append(rp)
                        raw_out_cands.append(p_cands or all_comp_cand_ids)
            if idx < n_b:
                cur_b_first_join = "synthetic"

        if idx < n_b:
            cur_b_steps.append(b_steps[idx])
            cur_b_prov.append(b_prov[idx])

    _flush_b_run()

    merged_pieces, merged_cands = _merge_adjacent_source_contiguous_pieces(
        raw_out_pieces, raw_out_cands
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
        )

    # Step 1: Exact deduplication by directed step+slice key
    by_step_key: dict[StepSliceKey, list[ExtractedCandidate]] = collections.defaultdict(
        list
    )
    for c in candidates:
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
    comp_results: list[ComponentResult] = []
    for root_idx in sorted(comp_map.keys()):
        comp = comp_map[root_idx]
        if len(comp) == 1:
            d = comp[0]
            occ_ids = frozenset(occ.candidate_id for occ in d.occurrences)
            comp_results.append(
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
                    partial_content=any(occ.partial_content for occ in d.occurrences),
                    repeat_residual=any(occ.repeat_residual for occ in d.occurrences),
                    multi_hit=any(occ.multi_hit for occ in d.occurrences),
                )
            )
        else:
            comp_results.extend(
                _solve_multi_member_component(
                    comp=comp,
                    graph=graph,
                    config=config,
                    parent_state_key=parent_state_key,
                    ancestor_keys=ancestor_keys,
                )
            )

    # Step 4: 4-criterion similarity clustering for real sequences & exact dedup for synthetic
    real_comps = [c for c in comp_results if not c.is_synthetic]
    synth_comps = [c for c in comp_results if c.is_synthetic]

    real_comps.sort(
        key=lambda c: (
            -c.length_bp_no_n,
            c.token_str,
            c.min_hap,
            c.exemplar.provenance[0].src_path,
            c.exemplar.provenance[0].src_start,
            c.exemplar.provenance[0].src_strand,
        )
    )
    synth_comps.sort(
        key=lambda c: (
            -len(c.support_haps),
            -c.length_bp_no_n,
            c.token_str,
            c.min_hap,
            c.exemplar.provenance[0].src_path,
            c.exemplar.provenance[0].src_start,
            c.exemplar.provenance[0].src_strand,
        )
    )

    rep0_aligner: Any | None = None
    rep0_occurrences: list[ExtractedCandidate] = []
    real_reps: list[ClusterRepresentative] = []

    for rc in real_comps:
        if has_rep0:
            if (len(rc.seq) > 1000 or len(rep0_seq) > 1000) and rep0_aligner is None:
                rep0_aligner = mappy.Aligner(seq=rep0_seq, preset="asm20")
            if check_similarity_merge(
                seq_s=rc.seq,
                seq_r=rep0_seq,
                min_sv_len=config.min_sv_len,
                cluster_id=config.cluster_id,
                cluster_win_id=config.cluster_win_id,
                aligner=rep0_aligner,
            ):
                rep0_occurrences.extend(rc.member_occurrences)
                continue

        merged_into: ClusterRepresentative | None = None
        for rep in real_reps:
            if (len(rc.seq) > 1000 or len(rep.seq) > 1000) and rep.aligner is None:
                rep.aligner = mappy.Aligner(seq=rep.seq, preset="asm20")
            if check_similarity_merge(
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
                )
            )

    # Synthetic sequences: exact sequence deduplication only (§3.6.2 Step 4)
    synth_reps: list[ClusterRepresentative] = []
    for sc in synth_comps:
        if has_rep0 and sc.seq == rep0_seq:
            rep0_occurrences.extend(sc.member_occurrences)
            continue

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
            new_occ_ids = frozenset(occ.candidate_id for occ in sc.member_occurrences)
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
                )
            )

    all_non_backbone = real_reps + synth_reps
    all_non_backbone.sort(
        key=lambda r: (
            -len(r.support_haps),
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

    for occ in rep0_occurrences:
        cand_to_rep_idx[occ.candidate_id] = 0
        cand_to_nested_sig[occ.candidate_id] = ()
        cand_to_reconstructed_len[occ.candidate_id] = len(rep0_seq)

    for idx_r, rep in enumerate(all_non_backbone, start=1):
        rep.rep_idx = idx_r
        if not rep.is_synthetic:
            sup_sorted = tuple(sorted(rep.support_haps))
            rep.pieces = tuple(replace(p, members=sup_sorted) for p in rep.pieces)
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

    return ParallelGroupSolution(
        left_anchor_rank=left_anchor_rank,
        right_anchor_rank=right_anchor_rank,
        has_nonempty_backbone=has_rep0,
        backbone_seq=rep0_seq,
        non_backbone_reps=tuple(all_non_backbone),
        cand_to_rep_idx=cand_to_rep_idx,
        cand_to_nested_sig=cand_to_nested_sig,
        cand_to_reconstructed_len=cand_to_reconstructed_len,
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
    tuple[StepOffsetRow, ...],
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
    step_offsets: list[StepOffsetRow] = []
    inserts_rows: list[dict[str, Any]] = []
    ins_event_spans: dict[str, tuple[int, int]] = {}
    pieces_by_event: dict[str, list[Piece]] = collections.defaultdict(list)

    # Pre-index CEN_ISLAND event IDs for each island slot block
    cen_events_by_step: dict[int, list[SVEvent]] = collections.defaultdict(list)
    for ev in sv_events:
        if ev.event_type == "CEN_ISLAND":
            for r_i, (rs, _) in enumerate(graph.ref_step_ranges):
                if rs == ev.ref_start:
                    cen_events_by_step[r_i].append(ev)
                    break

    for r_idx, ref_st in enumerate(graph.ref_fragment.steps):
        blocks = list(slot_blocks_by_step.get(r_idx, ()))
        slot_start_pan = pan_cursor

        if blocks:
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
                            eff_join: Literal[
                                "source_contiguous", "spacer", "synthetic"
                            ] = (
                                "spacer"
                                if (need_leading_spacer or rep_i > 0)
                                else p.join_before
                            )
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

        slot_ins_len = pan_cursor - slot_start_pan
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


@dataclass
class _ContigAnalysis:
    """Per-contig post-normalization main-chain and deviation analysis at top level."""

    raw_fragment: PathFragment
    normalized_fragment: PathFragment
    chains: tuple[ReverseChainRecord, ...]
    unnormalized_ranges: tuple[tuple[int, int], ...]
    main_chain: tuple[ProjectedAnchor, ...]
    lis_multi_hit: bool
    covered_ranks: set[int]
    unnormalized_rank_spans: list[tuple[int, int]]
    large_jump_rank_spans: list[tuple[int, int]]
    node_visit_counts: dict[str, int]


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
    for lj_lo, lj_hi in analysis.large_jump_rank_spans:
        if lo_r <= lj_lo and lj_hi <= hi_r:
            return False
    return True


def _contig_covers_rank_interval(
    analysis: _ContigAnalysis,
    rmin: int,
    rmax: int,
    left_flank: int | None = None,
    right_flank: int | None = None,
) -> bool:
    """Return True iff `analysis`'s forward main chain spans across `[rmin, rmax]` (or its flanks)."""
    if not analysis.main_chain:
        return False
    for u_lo, u_hi in analysis.unnormalized_rank_spans:
        if max(rmin, u_lo) <= min(rmax, u_hi):
            return False
    for lj_lo, lj_hi in analysis.large_jump_rank_spans:
        if max(rmin, lj_lo) < min(rmax, lj_hi):
            return False
    if left_flank is not None and right_flank is not None:
        if (
            left_flank in analysis.covered_ranks
            and right_flank in analysis.covered_ranks
        ):
            return True
    min_cov = min(analysis.covered_ranks) if analysis.covered_ranks else 10**9
    max_cov = max(analysis.covered_ranks) if analysis.covered_ranks else -1
    return min_cov <= rmin and max_cov >= rmax


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


def run_chromosome_pipeline(
    graph: GFAGraph,
    config: BuildConfig,
) -> ChromosomeBuildResult:
    """
    Execute the complete PanFlat v1 chromosome build pipeline (§3.0–§3.8, §5.1–§5.5).
    """
    validate_build_config(config)

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
    oriented_fragments: list[PathFragment] = []
    for frag in alt_sorted:
        oriented = orient_contig_top_level(frag, graph)
        if oriented.is_flipped_contig:
            log_lines.append(
                f"INFO\tORIENT_CONTIG\thap={oriented.hap_id}\tsrc_path={oriented.src_path}\tsrc_strand=-"
            )
        oriented_fragments.append(oriented)

    # 3. Partition fragments by centromere regions & process centromere mode (§3.8.1–3.8.5)
    partition = partition_fragments_by_centromere(
        oriented_fragments, regions, graph, config
    )
    cen_out = process_centromere_regions(partition, graph, config)

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

    for frag in normal_fragments_sorted:
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

        cov_ranks: set[int] = {anc.rank_in_b for anc in main_chain}
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
        for ch in inv_res.chains:
            if not ch.normalized:
                unnorm_rank_spans.append((ch.rmin, ch.rmax))

        visit_counts = compute_raw_visit_counts(norm_frag.steps)
        analysis = _ContigAnalysis(
            raw_fragment=frag,
            normalized_fragment=norm_frag,
            chains=inv_res.chains,
            unnormalized_ranges=inv_res.unnormalized_branch_ranges,
            main_chain=main_chain,
            lis_multi_hit=lis_mh,
            covered_ranks=cov_ranks,
            unnormalized_rank_spans=unnorm_rank_spans,
            large_jump_rank_spans=[],
            node_visit_counts=visit_counts,
        )
        contig_analyses.append(analysis)

        for ch in inv_res.chains:
            inv_chains_by_key[(ch.classification, ch.rmin, ch.rmax)].append(
                (analysis, ch)
            )

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
                    _emit_normal_unresolved(
                        region_id=".",
                        event_id=".",
                        prov=u_prov,
                        seq=u_seq,
                        reason="UNANCHORED_END",
                        export_to_fasta=True,
                    )
                continue

            rank_a = int(dev.left_anchor_rank)
            rank_b = int(dev.right_anchor_rank)
            span_bp = graph.ref_step_ranges[rank_b][0] - graph.ref_step_ranges[rank_a][1]

            if span_bp > config.max_jump:
                analysis.large_jump_rank_spans.append((rank_a, rank_b))
                large_jump_devs_by_pair[(rank_a, rank_b)].append((analysis, dev))
                continue

            if rank_a + 1 < rank_b and not _shares_canonical_tandem_unit(
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
                    large_jump_devs_by_pair[(ta, tb)].append((analysis, tdev))
                else:
                    terminal_devs_by_pair[(ta, tb)].append((analysis, tdev))

    # 5. Prepare candidate screening & group solving for all terminal normal groups G(a, b)
    sorted_group_pairs = sorted(
        terminal_devs_by_pair.keys(),
        key=lambda ab: (ab[1], ab[1] - ab[0], ab[0]),
    )

    group_guard_by_pair: dict[tuple[int, int], GroupRepeatGuardResult] = {}
    for pair in sorted_group_pairs:
        rank_a, rank_b = pair
        devs_at_pair = terminal_devs_by_pair[pair]
        guard_res = GroupRepeatGuardResult()
        next_src_order = 0
        cand_seq_idx = 1

        for analysis, tdev in devs_at_pair:
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
                cand_id_prefix=f"{graph.ref_chrom}_G_{rank_a}_{rank_b}_{tdev.hap_id}_{cand_seq_idx}",
                base_source_order=next_src_order,
                claimed_source_intervals=claimed_source_intervals,
            )
            cand_seq_idx += 1

            for rb in raw_blocks:
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
                    )
                )
                if had_mask:
                    guard_res.hap_had_masked_repeat[rb.hap_id] = True
                    guard_res.contig_had_masked_repeat.add((rb.hap_id, rb.src_path))
                if kept_c:
                    guard_res.hap_had_kept_candidate[rb.hap_id] = True
                    guard_res.contig_had_kept_candidate.add((rb.hap_id, rb.src_path))
                    if any(kc.partial_content for kc in kept_c):
                        guard_res.has_partial_content = True
                guard_res.kept_candidates.extend(kept_c)
                guard_res.unresolved_records.extend(unres_c)
                guard_res.json_log_entries.extend(logs_c)

        group_guard_by_pair[pair] = guard_res

    active_cand_pairs = [
        pair
        for pair in sorted_group_pairs
        if group_guard_by_pair[pair].kept_candidates
    ]

    def _solve_one_top_pair(pair: tuple[int, int]) -> ParallelGroupSolution:
        ra, rb = pair
        cands = group_guard_by_pair[pair].kept_candidates
        b_slice = ref_steps[ra + 1 : rb]
        m_keys = tuple(
            sorted({make_provenance_key(c.provenance) for c in cands if c.provenance})
        )
        g_state_key = SubproblemStateKey(
            b_id=(("REF", ra + 1, rb, "+"),),
            lo=ra + 1,
            hi=rb,
            has_logical_start=True,
            has_logical_end=True,
            m_key=m_keys,
        )
        return solve_parallel_group(
            left_anchor_rank=ra,
            right_anchor_rank=rb,
            backbone_slice_steps=b_slice,
            candidates=cands,
            graph=graph,
            config=config,
            parent_state_key=g_state_key,
            ancestor_keys=frozenset({top_state_key}),
        )

    group_solutions: dict[tuple[int, int], ParallelGroupSolution] = {}
    if config.threads > 1 and len(active_cand_pairs) > 1:
        with ThreadPoolExecutor(
            max_workers=min(config.threads, len(active_cand_pairs))
        ) as pool:
            solved_list = list(pool.map(_solve_one_top_pair, active_cand_pairs))
        for pair, sol in zip(active_cand_pairs, solved_list):
            group_solutions[pair] = sol
    else:
        for pair in active_cand_pairs:
            group_solutions[pair] = _solve_one_top_pair(pair)

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

        carrier_contig_ids = {id(ca) for ca, _ in entries}
        first_ch = entries[0][1]
        l_flank = first_ch.left_flank_rank
        r_flank = first_ch.right_flank_rank

        contig_calls: dict[str, list[str]] = collections.defaultdict(list)
        for hap_id, ca_list in contigs_by_hap.items():
            for ca in ca_list:
                if id(ca) in carrier_contig_ids:
                    contig_calls[hap_id].append("1")
                elif _contig_covers_rank_interval(ca, rmin, rmax, l_flank, r_flank):
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

        carrier_contig_ids = {id(ca) for ca, _ in lj_entries}
        contig_calls = collections.defaultdict(list)
        for hap_id, ca_list in contigs_by_hap.items():
            for ca in ca_list:
                if id(ca) in carrier_contig_ids:
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
    guard_unres_indices_by_pair: dict[tuple[int, int], list[int]] = (
        collections.defaultdict(list)
    )
    for pair in sorted_group_pairs:
        rank_a, rank_b = pair
        devs_at_pair = terminal_devs_by_pair[pair]
        guard_res = group_guard_by_pair[pair]
        sol = group_solutions.get(pair)
        ref_s_open = graph.ref_step_ranges[rank_a][1]
        ref_e_open = graph.ref_step_ranges[rank_b][0]
        b_slice_steps = tuple(ref_steps[rank_a + 1 : rank_b])

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
            dt_contig_ids = {id(ca) for ca, _ in dup_tra_entries}
            dt_calls: dict[str, list[str]] = collections.defaultdict(list)
            for hap_id, ca_list in contigs_by_hap.items():
                for ca in ca_list:
                    if id(ca) in dt_contig_ids:
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
            for ca, dev in devs_at_pair:
                cnv_c = detect_canonical_cnv(
                    b_slice_steps,
                    dev.branch_steps,
                    graph.nodes,
                    config.min_sv_len,
                )
                if cnv_c is not None:
                    full_cnv_calls.append((ca, dev, cnv_c))

            if full_cnv_calls:
                first_cnv = full_cnv_calls[0][2]
                target_u = first_cnv.unit_tokens
                k_ref = first_cnv.k_ref
                unit_bp = first_cnv.unit_bp

                dev_by_contig_id = {id(ca): dev for ca, dev in devs_at_pair}
                cnv_calls_by_hap: dict[str, list[str]] = collections.defaultdict(list)
                for hap_id, ca_list in contigs_by_hap.items():
                    for ca in ca_list:
                        if id(ca) in dev_by_contig_id:
                            d_seg = dev_by_contig_id[id(ca)]
                            p_alt = _extract_full_period(d_seg.branch_steps)
                            if p_alt is not None and p_alt[0] == target_u:
                                cnv_calls_by_hap[hap_id].append(str(p_alt[1]))
                            else:
                                cnv_calls_by_hap[hap_id].append(".")
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
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
            else:
                b_nids = {st.node_id for st in b_slice_steps if _is_full_step(st)}
                del_intervals: dict[
                    tuple[int, int], list[tuple[_ContigAnalysis, DeviationSegment]]
                ] = collections.defaultdict(list)
                partial_cnv_intervals: dict[
                    tuple[int, int, tuple[tuple[str, str], ...]],
                    list[tuple[_ContigAnalysis, DeviationSegment, CNVCall]],
                ] = collections.defaultdict(list)

                for ca, dev in devs_at_pair:
                    if _shares_canonical_tandem_unit(b_slice_steps, dev.branch_steps):
                        continue

                    br_nids = {
                        st.node_id for st in dev.branch_steps if _is_full_step(st)
                    }
                    shared_nids = b_nids & br_nids

                    if not shared_nids:
                        del_bp = ref_e_open - ref_s_open
                        if del_bp >= config.min_sv_len:
                            del_intervals[(rank_a + 1, rank_b)].append((ca, dev))
                    else:
                        cur_del_start: int | None = None
                        for offset_idx, b_st in enumerate(b_slice_steps):
                            r_step = rank_a + 1 + offset_idx
                            if b_st.node_id not in shared_nids:
                                if cur_del_start is None:
                                    cur_del_start = r_step
                            else:
                                if cur_del_start is not None:
                                    d_s = graph.ref_step_ranges[cur_del_start][0]
                                    d_e = graph.ref_step_ranges[r_step - 1][1]
                                    if d_e - d_s >= config.min_sv_len:
                                        del_intervals[(cur_del_start, r_step)].append(
                                            (ca, dev)
                                        )
                                    cur_del_start = None
                        if cur_del_start is not None:
                            d_s = graph.ref_step_ranges[cur_del_start][0]
                            d_e = graph.ref_step_ranges[rank_b - 1][1]
                            if d_e - d_s >= config.min_sv_len:
                                del_intervals[(cur_del_start, rank_b)].append((ca, dev))

                        b_shared_indices = [
                            rank_a + 1 + i
                            for i, st in enumerate(b_slice_steps)
                            if st.node_id in shared_nids
                        ]
                        if (
                            b_shared_indices
                            and b_shared_indices[-1] - b_shared_indices[0] + 1
                            == len(b_shared_indices)
                        ):
                            b_sh_steps = [
                                st for st in b_slice_steps if st.node_id in shared_nids
                            ]
                            br_sh_steps = [
                                st
                                for st in dev.branch_steps
                                if _is_full_step(st) and st.node_id in shared_nids
                            ]
                            sh_cnv = detect_canonical_cnv(
                                b_sh_steps,
                                br_sh_steps,
                                graph.nodes,
                                config.min_sv_len,
                            )
                            if sh_cnv is not None:
                                partial_cnv_intervals[
                                    (
                                        b_shared_indices[0],
                                        b_shared_indices[-1] + 1,
                                        sh_cnv.unit_tokens,
                                    )
                                ].append((ca, dev, sh_cnv))

                for (del_r_lo, del_r_hi), del_entries in sorted(
                    del_intervals.items()
                ):
                    d_ref_s = graph.ref_step_ranges[del_r_lo][0]
                    d_ref_e = graph.ref_step_ranges[del_r_hi - 1][1]
                    d_len = d_ref_e - d_ref_s
                    del_contig_ids = {id(ca) for ca, _ in del_entries}
                    del_calls: dict[str, list[str]] = collections.defaultdict(list)
                    for hap_id, ca_list in contigs_by_hap.items():
                        for ca in ca_list:
                            if id(ca) in del_contig_ids:
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
                        id(ca): c_call.k_alt for ca, _, c_call in p_cnv_entries
                    }
                    p_cnv_calls: dict[str, list[str]] = collections.defaultdict(list)
                    for hap_id, ca_list in contigs_by_hap.items():
                        for ca in ca_list:
                            if id(ca) in carrier_map:
                                p_cnv_calls[hap_id].append(str(carrier_map[id(ca)]))
                            elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
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
                    ex_hap = p_cnv_entries[0][0].raw_fragment.hap_id
                    p_cnv_ev = SVEvent(
                        chrom=graph.ref_chrom,
                        pan_start=0,
                        pan_end=0,
                        event_id="",
                        event_type="CNV",
                        ref_start=c_ref_s,
                        ref_end=c_ref_e,
                        anchors=f"{cnv_r_lo - 1}..{cnv_r_hi}",
                        alleles_str=(
                            f"1:len={first_c.k_alt * unit_bp},src={ex_hap},n={max(1, n_car)},"
                            f"unit={unit_str},k_ref={k_ref},k_alt={first_c.k_alt}{mh_tag}"
                        ),
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

            cands_by_contig: dict[tuple[str, str], list[ExtractedCandidate]] = (
                collections.defaultdict(list)
            )
            for kc in guard_res.kept_candidates:
                cands_by_contig[(kc.hap_id, kc.src_path)].append(kc)

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

            for hap_id, ca_list in contigs_by_hap.items():
                for ca in ca_list:
                    ckey = (hap_id, ca.raw_fragment.src_path)
                    c_list = sorted(
                        cands_by_contig.get(ckey, ()),
                        key=lambda x: x.source_order,
                    )
                    if c_list:
                        top_reps: list[int] = []
                        nested_all: list[tuple[Any, tuple[int, ...]]] = []
                        total_rec_len = 0
                        for kc in c_list:
                            r_idx = sol.cand_to_rep_idx.get(kc.candidate_id, 0)
                            if r_idx > 0:
                                top_reps.append(r_idx)
                                nested_all.extend(
                                    sol.cand_to_nested_sig.get(kc.candidate_id, ())
                                )
                                total_rec_len += sol.cand_to_reconstructed_len.get(
                                    kc.candidate_id,
                                    sol.non_backbone_reps[r_idx - 1].length_bp_no_n,
                                )
                        if top_reps:
                            sig_val: SignatureType = (
                                tuple(top_reps),
                                tuple(nested_all),
                            )
                            contig_sig_by_hap[hap_id].append(sig_val)
                            sig_reconstructed_len.setdefault(sig_val, total_rec_len)
                            if (
                                sig_val not in sig_exemplar_hap
                                or hap_id < sig_exemplar_hap[sig_val]
                            ):
                                sig_exemplar_hap[sig_val] = hap_id
                            for kc in c_list:
                                sig_cand_ids[sig_val].add(kc.candidate_id)
                                if kc.partial_content:
                                    sig_has_partial[sig_val] = True
                                if kc.repeat_residual:
                                    sig_has_rep_res[sig_val] = True
                                if kc.multi_hit:
                                    sig_has_mh[sig_val] = True
                        elif ckey in guard_res.contig_had_masked_repeat:
                            contig_sig_by_hap[hap_id].append(".")
                        elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
                            contig_sig_by_hap[hap_id].append("0")
                        else:
                            contig_sig_by_hap[hap_id].append(".")
                    elif ckey in guard_res.contig_had_masked_repeat:
                        contig_sig_by_hap[hap_id].append(".")
                    elif _contig_covers_anchor_pair(ca, rank_a, rank_b):
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
                key=lambda s: (-len(sig_carrier_haps.get(s, ())), s),
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

            group_has_partial = guard_res.has_partial_content or any(
                r.partial_content for r in sol.non_backbone_reps
            )
            group_has_rep_res = any(
                r.repeat_residual for r in sol.non_backbone_reps
            )
            group_has_mh = any(r.multi_hit for r in sol.non_backbone_reps) or any(
                bool(dev.multi_hit_nodes) for _, dev in devs_at_pair
            )

            allele_entries: list[str] = []
            for s in ordered_sigs:
                a_idx = sig_to_allele_idx[s]
                a_len = sig_reconstructed_len[s]
                c_haps = sorted(sig_carrier_haps.get(s, ()))
                ex_h = c_haps[0] if c_haps else sig_exemplar_hap[s]
                n_h = len(c_haps)
                tags = ""
                if sig_has_partial.get(s, False) or group_has_partial:
                    tags += ",partial_content=1"
                if sig_has_rep_res.get(s, False) or group_has_rep_res:
                    tags += ",repeat_residual=1"
                if sig_has_mh.get(s, False) or group_has_mh:
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
    }

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
