"""Top-level contig orientation, inversion normalization, weighted LIS, deviation extraction, cycle folding, and canonical CNV detection (§3.1–§3.5, §3.6.6, §3.7)."""

from __future__ import annotations

import collections
from typing import AbstractSet, Any, Literal, Mapping, Sequence

import numpy as np

from panflat.gfa_io import spell_steps
from panflat.models import (
    BuildConfig,
    CNVCall,
    CompactNodeStore,
    CompactRefNodeSet,
    CompactVisitCounts,
    CycleFoldResult,
    DeviationSegment,
    GFAGraph,
    InputContractError,
    InversionNormalizationResult,
    LazyProjectedAnchorSequence,
    LazyProvenanceTuple,
    LazyStepTuple,
    PathFragment,
    ProjectedAnchor,
    ReverseChainRecord,
    Step,
    StepProvenance,
    UnresolvedRecord,
)
from panflat.repeat_guard import merge_contiguous_provenance


def _is_full_step(step: Step) -> bool:
    """Return True iff `step` is an unsliced full node traversal (§3.6.6)."""
    return not step.is_partial and step.slice_start == 0 and step.slice_end is None


def build_backbone_anchor_map(
    backbone_steps: Sequence[Step],
    lo: int = 0,
    hi: int | None = None,
) -> dict[str, tuple[int, Literal["+", "-"]]]:
    """
    Build a map `node_id -> (rank_in_b, b_orient)` for nodes appearing **exactly once**
    as full steps within `backbone_steps[lo:hi]` (§3.1 & §3.6.6).

    Partial steps (`step.is_partial == True`) never participate as anchors.
    """
    n_steps = len(backbone_steps)
    eff_hi = n_steps if hi is None else hi
    if lo < 0 or eff_hi > n_steps or lo > eff_hi:
        raise InputContractError(
            f"Invalid backbone slice bounds [{lo}, {eff_hi}) for backbone of length {n_steps}"
        )

    if isinstance(backbone_steps, LazyStepTuple):
        b_codes = backbone_steps._codes[lo:eff_hi]
        if int(b_codes.shape[0]) == 0:
            return {}
        b_nids = np.abs(b_codes)
        u_nids, first_idx, counts = np.unique(
            b_nids, return_index=True, return_counts=True
        )
        uniq_mask = counts == 1
        out_map: dict[str, tuple[int, Literal["+", "-"]]] = {}
        for nid_val, rel_idx in zip(
            u_nids[uniq_mask].tolist(), first_idx[uniq_mask].tolist()
        ):
            c = int(b_codes[rel_idx])
            out_map[str(nid_val)] = (lo + int(rel_idx), "+" if c > 0 else "-")
        return out_map

    counts_c: collections.Counter[str] = collections.Counter()
    first_hit: dict[str, tuple[int, Literal["+", "-"]]] = {}
    for rank_idx in range(lo, eff_hi):
        st = backbone_steps[rank_idx]
        if not _is_full_step(st):
            continue
        counts_c[st.node_id] += 1
        if st.node_id not in first_hit:
            first_hit[st.node_id] = (rank_idx, st.orient)

    return {
        node_id: first_hit[node_id]
        for node_id, cnt in counts_c.items()
        if cnt == 1
    }


def _get_lazy_backbone_anchor_arrays(
    backbone: LazyStepTuple,
    max_nid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and cache `(nid_to_rank, nid_to_sign)` on full `backbone` `LazyStepTuple`."""
    cached = backbone._anchor_cache
    if cached is not None and cached[0] == max_nid:
        return cached[1], cached[2]

    b_codes = backbone._codes
    b_nids = np.abs(b_codes)
    counts = np.bincount(b_nids, minlength=max_nid + 1)
    nid_to_rank = np.full(max_nid + 1, -1, dtype=np.int32)
    nid_to_sign = np.zeros(max_nid + 1, dtype=np.int8)
    uniq_step_mask = counts[b_nids] == 1
    uniq_step_indices = np.flatnonzero(uniq_step_mask).astype(np.int32)
    uniq_nids = b_nids[uniq_step_indices]
    nid_to_rank[uniq_nids] = uniq_step_indices
    nid_to_sign[uniq_nids] = np.where(
        b_codes[uniq_step_indices] > 0, 1, -1
    ).astype(np.int8)
    backbone._anchor_cache = (max_nid, nid_to_rank, nid_to_sign)
    return nid_to_rank, nid_to_sign


def project_step_tuples_onto_backbone(
    branch_steps: Sequence[Step],
    backbone_steps: Sequence[Step],
    lo: int = 0,
    hi: int | None = None,
) -> list[tuple[int | None, Literal["+", "-"] | None]]:
    """
    Project each step in `branch_steps` to `(rank_in_b | None, rel_orient | None)` (§3.2).
    """
    anchor_map = build_backbone_anchor_map(backbone_steps, lo=lo, hi=hi)
    projected: list[tuple[int | None, Literal["+", "-"] | None]] = []
    for st in branch_steps:
        if _is_full_step(st) and st.node_id in anchor_map:
            rank_in_b, b_orient = anchor_map[st.node_id]
            rel_orient: Literal["+", "-"] = "+" if st.orient == b_orient else "-"
            projected.append((rank_in_b, rel_orient))
        else:
            projected.append((None, None))
    return projected


def project_steps_onto_backbone(
    fragment_or_steps: PathFragment | Sequence[Step],
    backbone_or_provenance: Sequence[Step] | Sequence[StepProvenance] | GFAGraph,
    nodes_or_backbone: Mapping[str, str] | Sequence[Step] | int | None = None,
    nodes: Mapping[str, str] | int | None = None,
    lo: int = 0,
    hi: int | None = None,
    *,
    provenance: Sequence[StepProvenance] | None = None,
    backbone_steps: Sequence[Step] | None = None,
) -> Sequence[ProjectedAnchor]:
    """
    §3.1–§3.2 (F7): Project branch steps onto unique full-step anchors in `backbone_steps[lo:hi)`.

    Supports flexible calling forms:
    1. `project_steps_onto_backbone(fragment, graph, lo=0, hi=None)`
    2. `project_steps_onto_backbone(fragment, backbone_steps, nodes, lo=0, hi=None)`
    3. `project_steps_onto_backbone(branch_steps, branch_provenance, backbone_steps, nodes, lo=0, hi=None)`
    4. `project_steps_onto_backbone(branch_steps, backbone_steps, nodes, lo=0, hi=None)`
    """
    resolved_branch_steps: Sequence[Step]
    resolved_branch_prov: Sequence[StepProvenance] | None = provenance
    resolved_backbone: Sequence[Step] | None = backbone_steps
    resolved_nodes: Mapping[str, str] | None = None
    eff_lo: int = lo
    eff_hi: int | None = hi

    if isinstance(fragment_or_steps, PathFragment):
        resolved_branch_steps = fragment_or_steps.steps
        resolved_branch_prov = fragment_or_steps.provenance
        if isinstance(backbone_or_provenance, GFAGraph):
            resolved_backbone = backbone_or_provenance.ref_fragment.steps
            resolved_nodes = backbone_or_provenance.nodes
            if isinstance(nodes_or_backbone, int):
                eff_lo = nodes_or_backbone
            if isinstance(nodes, int):
                eff_hi = nodes
        else:
            resolved_backbone = backbone_or_provenance  # type: ignore[assignment]
            if isinstance(nodes_or_backbone, Mapping):
                resolved_nodes = nodes_or_backbone
            elif isinstance(nodes, Mapping):
                resolved_nodes = nodes
            if isinstance(nodes, int):
                eff_lo = nodes
    else:
        resolved_branch_steps = fragment_or_steps
        if isinstance(backbone_or_provenance, GFAGraph):
            resolved_backbone = backbone_or_provenance.ref_fragment.steps
            resolved_nodes = backbone_or_provenance.nodes
            if isinstance(nodes_or_backbone, int):
                eff_lo = nodes_or_backbone
            if isinstance(nodes, int):
                eff_hi = nodes
        elif isinstance(nodes_or_backbone, Mapping):
            # Form 4: (branch_steps, backbone_steps, nodes, [lo, [hi]])
            resolved_backbone = backbone_or_provenance  # type: ignore[assignment]
            resolved_nodes = nodes_or_backbone
            if isinstance(nodes, int):
                eff_lo = nodes
        else:
            # Form 3: (branch_steps, branch_provenance, backbone_steps, nodes, [lo, [hi]])
            resolved_branch_prov = backbone_or_provenance  # type: ignore[assignment]
            if nodes_or_backbone is not None and not isinstance(nodes_or_backbone, int):
                resolved_backbone = nodes_or_backbone  # type: ignore[assignment]
            if isinstance(nodes, Mapping):
                resolved_nodes = nodes

    if resolved_backbone is None or resolved_nodes is None:
        raise InputContractError(
            "project_steps_onto_backbone requires backbone_steps and nodes mapping"
        )

    if resolved_branch_prov is None:
        if isinstance(resolved_branch_steps, LazyStepTuple) and isinstance(
            resolved_nodes, CompactNodeStore
        ):
            resolved_branch_prov = LazyProvenanceTuple(
                src_hap=" synthetic#1",
                src_path="synthetic#1#ctg",
                base_offset=0,
                orig_codes=resolved_branch_steps._codes,
                node_lens=resolved_nodes._node_lens,
            )
        else:
            synth_prov: list[StepProvenance] = []
            cur_pos = 0
            for st in resolved_branch_steps:
                slen = st.step_len(resolved_nodes)
                synth_prov.append(
                    StepProvenance(
                        src_hap=" synthetic#1",
                        src_path="synthetic#1#ctg",
                        src_start=cur_pos,
                        src_end=cur_pos + slen,
                        src_strand="+",
                    )
                )
                cur_pos += slen
            resolved_branch_prov = synth_prov

    if len(resolved_branch_steps) != len(resolved_branch_prov):
        raise InputContractError(
            f"branch_steps length ({len(resolved_branch_steps)}) != "
            f"branch_provenance length ({len(resolved_branch_prov)})"
        )

    n_b_total = len(resolved_backbone)
    eff_hi_val = n_b_total if eff_hi is None else eff_hi
    if eff_lo < 0 or eff_hi_val > n_b_total or eff_lo > eff_hi_val:
        raise InputContractError(
            f"Invalid backbone slice bounds [{eff_lo}, {eff_hi_val}) for backbone of length {n_b_total}"
        )

    # Fast vectorized path when branch, backbone, and nodes are compact NumPy-backed
    if (
        isinstance(resolved_branch_steps, LazyStepTuple)
        and isinstance(resolved_backbone, LazyStepTuple)
        and isinstance(resolved_nodes, CompactNodeStore)
        and len(resolved_nodes._extra) == 0
    ):
        br_codes = resolved_branch_steps._codes
        if int(br_codes.shape[0]) == 0 or eff_lo >= eff_hi_val:
            return []

        max_nid = resolved_nodes._max_nid
        if eff_lo == 0 and eff_hi_val == n_b_total and n_b_total > 10_000:
            nid_to_rank, nid_to_sign = _get_lazy_backbone_anchor_arrays(
                resolved_backbone, max_nid
            )
            br_nids = np.abs(br_codes)
            ranks = nid_to_rank[br_nids]
            hit_mask = ranks >= 0
            branch_step_idx = np.flatnonzero(hit_mask).astype(np.int32)
            if int(branch_step_idx.shape[0]) == 0:
                return []
            rank_in_b = ranks[branch_step_idx]
            hit_codes = br_codes[branch_step_idx]
            hit_nids = br_nids[branch_step_idx]
            b_signs = nid_to_sign[hit_nids]
            br_signs = np.where(hit_codes > 0, 1, -1).astype(np.int8)
            rel_sign = np.where(br_signs == b_signs, 1, -1).astype(np.int8)
            weights = resolved_nodes._node_lens[hit_nids]
        else:
            b_codes = resolved_backbone._codes[eff_lo:eff_hi_val]
            b_nids = np.abs(b_codes)
            u_nids, first_idx, counts = np.unique(
                b_nids, return_index=True, return_counts=True
            )
            uniq_mask = counts == 1
            uniq_b_nids = u_nids[uniq_mask]
            if int(uniq_b_nids.shape[0]) == 0:
                return []
            uniq_rel_idx = first_idx[uniq_mask]
            uniq_ranks = (eff_lo + uniq_rel_idx).astype(np.int32)
            uniq_b_signs = np.where(b_codes[uniq_rel_idx] > 0, 1, -1).astype(np.int8)

            br_nids = np.abs(br_codes)
            pos = np.searchsorted(uniq_b_nids, br_nids)
            n_u = int(uniq_b_nids.shape[0])
            hit_mask = (pos < n_u) & (
                uniq_b_nids[np.minimum(pos, n_u - 1)] == br_nids
            )
            branch_step_idx = np.flatnonzero(hit_mask).astype(np.int32)
            if int(branch_step_idx.shape[0]) == 0:
                return []
            hit_pos = pos[branch_step_idx]
            rank_in_b = uniq_ranks[hit_pos]
            b_signs = uniq_b_signs[hit_pos]
            hit_codes = br_codes[branch_step_idx]
            hit_nids = br_nids[branch_step_idx]
            br_signs = np.where(hit_codes > 0, 1, -1).astype(np.int8)
            rel_sign = np.where(br_signs == b_signs, 1, -1).astype(np.int8)
            weights = resolved_nodes._node_lens[hit_nids]

        n_hits = int(branch_step_idx.shape[0])
        if n_hits > 100_000:
            hit_cnts = np.bincount(hit_nids)
            dup_nids = np.flatnonzero(hit_cnts > 1)
        elif n_hits > 1:
            s_nids = np.sort(hit_nids)
            dup_nids = np.unique(s_nids[:-1][s_nids[1:] == s_nids[:-1]])
        else:
            dup_nids = np.empty(0, dtype=np.int32)

        has_mh = int(dup_nids.shape[0]) > 0
        mh_tuple = (
            tuple(sorted(str(x) for x in dup_nids.tolist())) if has_mh else ()
        )
        lazy_seq = LazyProjectedAnchorSequence(
            branch_step_idx=branch_step_idx,
            rank_in_b=rank_in_b,
            rel_sign=rel_sign,
            weights=weights,
            steps=resolved_branch_steps,
            provenance=resolved_branch_prov,
            multi_hit_nids=mh_tuple,
            has_multi_hit=has_mh,
        )
        if n_hits <= 5_000:
            return list(lazy_seq)
        return lazy_seq

    anchor_map = build_backbone_anchor_map(resolved_backbone, lo=eff_lo, hi=eff_hi_val)
    anchors: list[ProjectedAnchor] = []
    for idx, (st, prov) in enumerate(zip(resolved_branch_steps, resolved_branch_prov)):
        if not _is_full_step(st):
            continue
        if st.node_id not in anchor_map:
            continue
        rank_in_b_val, b_orient = anchor_map[st.node_id]
        rel_orient: Literal["+", "-"] = "+" if st.orient == b_orient else "-"
        weight = st.step_len(resolved_nodes)
        anchors.append(
            ProjectedAnchor(
                branch_step_idx=idx,
                rank_in_b=rank_in_b_val,
                rel_orient=rel_orient,
                node_id=st.node_id,
                weight=weight,
                step=st,
                provenance=prov,
            )
        )
    return anchors


def orient_contig_top_level(
    fragment: PathFragment,
    graph: GFAGraph,
) -> PathFragment:
    """
    §3.3.0 (F8, T23): Top-level whole-contig orientation.

    - Projects `fragment` onto `graph.ref_fragment.steps`.
    - Sums `forward_bp` (`rel_orient == "+"`) and `reverse_bp` (`rel_orient == "-"`).
    - Strictly when `reverse_bp > forward_bp` (ties are NOT flipped), reverse-complements
      the entire fragment:
      - Reverses `fragment.steps` and flips each step's orientation.
      - Reverses `fragment.provenance` and flips each entry's `src_strand` while
        preserving original 0-based `[src_start, src_end)` coordinates on the unflipped path.
      - Sets `is_flipped_contig = not fragment.is_flipped_contig`.
    - Whole-contig orientation is never recorded as an `INV` event.
    """
    if fragment.src_path == graph.ref_fragment.src_path:
        return fragment

    anchors = project_steps_onto_backbone(
        fragment.steps,
        fragment.provenance,
        graph.ref_fragment.steps,
        graph.nodes,
    )
    if isinstance(anchors, LazyProjectedAnchorSequence):
        forward_bp = int(anchors.weights[anchors.rel_sign > 0].sum(dtype=np.int64))
        reverse_bp = int(anchors.weights[anchors.rel_sign < 0].sum(dtype=np.int64))
    else:
        forward_bp = sum(a.weight for a in anchors if a.rel_orient == "+")
        reverse_bp = sum(a.weight for a in anchors if a.rel_orient == "-")

    if reverse_bp <= forward_bp:
        return fragment

    if isinstance(fragment.steps, LazyStepTuple) and isinstance(
        fragment.provenance, LazyProvenanceTuple
    ):
        flipped_steps: Any = fragment.steps.flipped_all()
        flipped_prov: Any = fragment.provenance.flipped_contig()
    else:
        flipped_steps = tuple(
            st.flipped(st.step_len(graph.nodes) if not st.is_partial else len(graph.nodes[st.node_id]))
            for st in reversed(fragment.steps)
        )
        flipped_prov = tuple(pr.flipped() for pr in reversed(fragment.provenance))

    return PathFragment(
        sample=fragment.sample,
        hap=fragment.hap,
        hap_id=fragment.hap_id,
        seqid=fragment.seqid,
        src_path=fragment.src_path,
        base_offset=fragment.base_offset,
        steps=flipped_steps,
        provenance=flipped_prov,
        is_w_line=fragment.is_w_line,
        total_len=fragment.total_len,
        is_flipped_contig=not fragment.is_flipped_contig,
    )


def _extract_contiguous_nonref_slices(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    nodes: Mapping[str, str],
    global_ref_nodes: AbstractSet[str],
    min_sv_len: int,
) -> list[tuple[tuple[Step, ...], tuple[StepProvenance, ...], StepProvenance, str]]:
    """
    Extract maximal contiguous runs of global non-reference steps from `(steps, provenance)`,
    split at any non-contiguous source boundaries (§3.7), and keep slices with
    `length >= min_sv_len`.
    """
    results: list[
        tuple[tuple[Step, ...], tuple[StepProvenance, ...], StepProvenance, str]
    ] = []
    n = len(steps)
    if n == 0:
        return results

    if isinstance(steps, LazyStepTuple) and isinstance(
        global_ref_nodes, CompactRefNodeSet
    ):
        codes = steps._codes
        nids = np.abs(codes)
        ref_mask = global_ref_nodes._mask[nids]
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
        run_pairs = list(zip(run_starts.tolist(), run_ends.tolist()))
    else:
        run_pairs = []
        i = 0
        while i < n:
            if steps[i].node_id in global_ref_nodes:
                i += 1
                continue
            j = i + 1
            while j < n and steps[j].node_id not in global_ref_nodes:
                j += 1
            run_pairs.append((i, j))
            i = j

    for i, j in run_pairs:
        run_steps = steps[i:j]
        run_prov = provenance[i:j]

        if isinstance(run_prov, LazyProvenanceTuple) and not run_prov.has_internal_discontinuity():
            if run_prov.total_bp() < min_sv_len:
                continue
            det_steps = (
                run_steps.detach()
                if isinstance(run_steps, LazyStepTuple)
                else run_steps
            )
            det_prov = run_prov.detach()
            sub_steps = tuple(det_steps) if len(det_steps) <= 64 else det_steps
            sub_provs = tuple(det_prov) if len(det_prov) <= 64 else det_prov
            seq_str = spell_steps(det_steps, nodes)
            merged_list = merge_contiguous_provenance(det_prov, seq_str)
            if merged_list:
                merged_prov, merged_seq = merged_list[0]
                if merged_prov.length >= min_sv_len:
                    results.append((sub_steps, sub_provs, merged_prov, merged_seq))
            continue

        grp_start = 0
        while grp_start < len(run_steps):
            grp_end = grp_start + 1
            while grp_end < len(run_steps):
                prev_p = run_prov[grp_end - 1]
                curr_p = run_prov[grp_end]
                is_contig = (
                    prev_p.src_hap == curr_p.src_hap
                    and prev_p.src_path == curr_p.src_path
                    and prev_p.src_strand == curr_p.src_strand
                    and (
                        (prev_p.src_strand == "+" and prev_p.src_end == curr_p.src_start)
                        or (
                            prev_p.src_strand == "-"
                            and prev_p.src_start == curr_p.src_end
                        )
                    )
                )
                if is_contig:
                    grp_end += 1
                else:
                    break

            sub_steps = tuple(run_steps[grp_start:grp_end])
            sub_provs = tuple(run_prov[grp_start:grp_end])
            seq_str = spell_steps(sub_steps, nodes)
            merged_list = merge_contiguous_provenance(sub_provs, seq_str)
            if merged_list:
                merged_prov, merged_seq = merged_list[0]
                if merged_prov.length >= min_sv_len:
                    results.append((sub_steps, sub_provs, merged_prov, merged_seq))
            grp_start = grp_end

    return results


def normalize_inversions_top_level(
    fragment: PathFragment,
    graph: GFAGraph,
    config: BuildConfig,
    region_id_prefix: str = "INV_REG",
    record_id_start: int = 1,
) -> InversionNormalizationResult:
    """
    §3.3.1–§3.3.2 (F9, T6, T7, T8, T24): Top-level reverse chain detection & inversion normalization.
    """
    anchors = project_steps_onto_backbone(
        fragment.steps,
        fragment.provenance,
        graph.ref_fragment.steps,
        graph.nodes,
    )
    if not anchors:
        return InversionNormalizationResult(
            normalized_fragment=fragment,
            chains=(),
            unresolved_records=(),
            unnormalized_branch_ranges=(),
        )

    if isinstance(anchors, LazyProjectedAnchorSequence):
        b_idxs = anchors.branch_step_idx
        ranks = anchors.rank_in_b
        signs = anchors.rel_sign
        minus_pos = np.flatnonzero(signs < 0)
        if int(minus_pos.shape[0]) == 0:
            return InversionNormalizationResult(
                normalized_fragment=fragment,
                chains=(),
                unresolved_records=(),
                unnormalized_branch_ranges=(),
            )
        m_list = minus_pos.tolist()
        raw_run_bounds: list[tuple[int, int, int, int]] = []
        i_m = 0
        n_m = len(m_list)
        while i_m < n_m:
            s_anc = m_list[i_m]
            e_anc = s_anc
            while (
                i_m + 1 < n_m
                and m_list[i_m + 1] == e_anc + 1
                and int(ranks[m_list[i_m + 1]]) < int(ranks[e_anc])
            ):
                e_anc = m_list[i_m + 1]
                i_m += 1
            raw_run_bounds.append(
                (
                    int(b_idxs[s_anc]),
                    int(b_idxs[e_anc]),
                    int(ranks[e_anc]),
                    int(ranks[s_anc]),
                )
            )
            i_m += 1

        plus_mask = signs > 0
        plus_b_idxs = b_idxs[plus_mask]
        plus_ranks = ranks[plus_mask]
        n_plus = int(plus_b_idxs.shape[0])

        chain_bounds: list[tuple[int, int, int, int, int | None, int | None, int, int]] = []
        for b_start, b_end, rmin, rmax in raw_run_bounds:
            ref_start = graph.ref_step_ranges[rmin][0]
            ref_end = graph.ref_step_ranges[rmax][1]
            l_pos = int(np.searchsorted(plus_b_idxs, b_start, side="left")) - 1
            left_flank = int(plus_ranks[l_pos]) if l_pos >= 0 else None
            r_pos = int(np.searchsorted(plus_b_idxs, b_end, side="right"))
            right_flank = int(plus_ranks[r_pos]) if r_pos < n_plus else None
            chain_bounds.append(
                (b_start, b_end, rmin, rmax, left_flank, right_flank, ref_start, ref_end)
            )
    else:
        raw_runs: list[list[ProjectedAnchor]] = []
        idx = 0
        n_anc = len(anchors)
        while idx < n_anc:
            if anchors[idx].rel_orient != "-":
                idx += 1
                continue
            run = [anchors[idx]]
            while (
                idx + 1 < n_anc
                and anchors[idx + 1].rel_orient == "-"
                and anchors[idx + 1].rank_in_b < anchors[idx].rank_in_b
            ):
                run.append(anchors[idx + 1])
                idx += 1
            raw_runs.append(run)
            idx += 1

        if not raw_runs:
            return InversionNormalizationResult(
                normalized_fragment=fragment,
                chains=(),
                unresolved_records=(),
                unnormalized_branch_ranges=(),
            )

        plus_anchors = [a for a in anchors if a.rel_orient == "+"]
        chain_bounds = []
        for run in raw_runs:
            b_start = run[0].branch_step_idx
            b_end = run[-1].branch_step_idx
            rmin = run[-1].rank_in_b
            rmax = run[0].rank_in_b
            ref_start = graph.ref_step_ranges[rmin][0]
            ref_end = graph.ref_step_ranges[rmax][1]

            left_flank = None
            for pa in plus_anchors:
                if pa.branch_step_idx < b_start:
                    left_flank = pa.rank_in_b
                else:
                    break

            right_flank = None
            for pa in plus_anchors:
                if pa.branch_step_idx > b_end:
                    right_flank = pa.rank_in_b
                    break

            chain_bounds.append(
                (b_start, b_end, rmin, rmax, left_flank, right_flank, ref_start, ref_end)
            )

    n_chains = len(chain_bounds)
    classifications: list[
        tuple[Literal["INV", "INV_DISPLACED", "INV_EDGE", "INV_CONFLICT"], bool]
    ] = []
    for i in range(n_chains):
        _, _, rmin_i, rmax_i, l_rank, f_rank, _, _ = chain_bounds[i]
        has_conflict = any(
            i != j
            and max(rmin_i, chain_bounds[j][2]) <= min(rmax_i, chain_bounds[j][3])
            for j in range(n_chains)
        )
        if has_conflict:
            classifications.append(("INV_CONFLICT", False))
        elif l_rank is None or f_rank is None:
            classifications.append(("INV_EDGE", False))
        elif l_rank < rmin_i and f_rank > rmax_i:
            classifications.append(("INV", True))
        else:
            classifications.append(("INV_DISPLACED", True))

    # Step 4: Normalize INV and INV_DISPLACED closed intervals [b_start, b_end]
    if isinstance(fragment.steps, LazyStepTuple) and isinstance(
        fragment.provenance, LazyProvenanceTuple
    ):
        flipped_ivs: list[tuple[int, int]] = []
        work_codes = fragment.steps._codes.copy()
        for i in range(n_chains):
            _, is_norm = classifications[i]
            if not is_norm:
                continue
            b_start, b_end = chain_bounds[i][0], chain_bounds[i][1]
            s = b_start
            e = b_end + 1
            work_codes[s:e] = -work_codes[s:e][::-1]
            flipped_ivs.append((s, e))
        norm_steps: Any = LazyStepTuple(work_codes)
        norm_prov: Any = fragment.provenance.with_flipped_intervals(flipped_ivs)
    else:
        work_steps = list(fragment.steps)
        work_prov = list(fragment.provenance)
        for i in range(n_chains):
            _, is_norm = classifications[i]
            if not is_norm:
                continue
            b_start, b_end = chain_bounds[i][0], chain_bounds[i][1]
            s = b_start
            e = b_end + 1
            work_steps[s:e] = [
                st.flipped(len(graph.nodes[st.node_id]))
                for st in reversed(work_steps[s:e])
            ]
            work_prov[s:e] = [pr.flipped() for pr in reversed(work_prov[s:e])]
        norm_steps = tuple(work_steps)
        norm_prov = tuple(work_prov)

    normalized_fragment = PathFragment(
        sample=fragment.sample,
        hap=fragment.hap,
        hap_id=fragment.hap_id,
        seqid=fragment.seqid,
        src_path=fragment.src_path,
        base_offset=fragment.base_offset,
        steps=norm_steps,
        provenance=norm_prov,
        is_w_line=fragment.is_w_line,
        total_len=fragment.total_len,
        is_flipped_contig=fragment.is_flipped_contig,
    )

    # Step 5: Handle unnormalized chains (INV_EDGE, INV_CONFLICT)
    gap_to_unnorm_chains: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    if any(not is_norm for _, is_norm in classifications):
        norm_anchors = project_steps_onto_backbone(
            normalized_fragment.steps,
            normalized_fragment.provenance,
            graph.ref_fragment.steps,
            graph.nodes,
        )
        if isinstance(norm_anchors, LazyProjectedAnchorSequence):
            norm_plus_arr = norm_anchors.branch_step_idx[norm_anchors.rel_sign > 0]
            n_np = int(norm_plus_arr.shape[0])
            for i in range(n_chains):
                if classifications[i][1]:
                    continue
                b_start, b_end = chain_bounds[i][0], chain_bounds[i][1]
                lp = int(np.searchsorted(norm_plus_arr, b_start, side="left")) - 1
                prev_plus = int(norm_plus_arr[lp]) if lp >= 0 else -1
                rp = int(np.searchsorted(norm_plus_arr, b_end, side="right"))
                next_plus = int(norm_plus_arr[rp]) if rp < n_np else len(fragment.steps)
                gap_to_unnorm_chains[(prev_plus, next_plus)].append(i)
        else:
            norm_plus_indices = sorted(
                a.branch_step_idx for a in norm_anchors if a.rel_orient == "+"
            )
            for i in range(n_chains):
                if classifications[i][1]:
                    continue
                b_start, b_end = chain_bounds[i][0], chain_bounds[i][1]
                prev_plus = -1
                for p_idx in norm_plus_indices:
                    if p_idx < b_start:
                        prev_plus = p_idx
                    else:
                        break
                next_plus = len(fragment.steps)
                for p_idx in norm_plus_indices:
                    if p_idx > b_end:
                        next_plus = p_idx
                        break
                gap_to_unnorm_chains[(prev_plus, next_plus)].append(i)

    unnormalized_branch_ranges: list[tuple[int, int]] = []
    unresolved_records: list[UnresolvedRecord] = []
    chain_novel_slices: dict[
        int, list[tuple[tuple[Step, ...], tuple[StepProvenance, ...]]]
    ] = collections.defaultdict(list)
    rec_counter = record_id_start

    for (prev_plus, next_plus), chain_indices in sorted(gap_to_unnorm_chains.items()):
        span_start = prev_plus + 1
        span_end = next_plus
        unnormalized_branch_ranges.append((span_start, span_end))

        m = len(chain_indices)
        for k, c_idx in enumerate(chain_indices):
            b_start, b_end, _, _, _, _, ref_start, ref_end = chain_bounds[c_idx]
            cls_name = classifications[c_idx][0]
            u_start = span_start if k == 0 else chain_bounds[chain_indices[k - 1]][1] + 1
            u_end = b_end + 1 if k < m - 1 else span_end

            if u_start >= u_end:
                continue

            slices = _extract_contiguous_nonref_slices(
                steps=fragment.steps[u_start:u_end],
                provenance=fragment.provenance[u_start:u_end],
                nodes=graph.nodes,
                global_ref_nodes=graph.global_ref_nodes,
                min_sv_len=config.min_sv_len,
            )
            reg_id = f"{region_id_prefix}_{c_idx + 1}"
            ev_id = (
                f"{graph.ref_chrom}_{cls_name}_{c_idx + 1}"
                if (ref_end - ref_start) >= config.min_sv_len
                else "."
            )
            for sub_steps, sub_provs, merged_prov, seq_str in slices:
                chain_novel_slices[c_idx].append((sub_steps, sub_provs))
                rec_id = f"UNRES_{rec_counter}"
                rec_counter += 1
                unresolved_records.append(
                    UnresolvedRecord(
                        record_id=rec_id,
                        region_id=reg_id,
                        event_id=ev_id,
                        src_path=merged_prov.src_path,
                        src_start=merged_prov.src_start,
                        src_end=merged_prov.src_end,
                        src_strand=merged_prov.src_strand,
                        length=merged_prov.length,
                        reason=cls_name,
                        seq=seq_str,
                        export_to_fasta=True,
                        exported_fasta_id=rec_id,
                    )
                )

    chain_records: list[ReverseChainRecord] = []
    for i in range(n_chains):
        b_start, b_end, rmin_i, rmax_i, l_rank, f_rank, ref_s, ref_e = chain_bounds[i]
        cls_name, is_norm = classifications[i]
        chain_records.append(
            ReverseChainRecord(
                chain_idx=i,
                branch_start_idx=b_start,
                branch_end_idx=b_end,
                rmin=rmin_i,
                rmax=rmax_i,
                left_flank_rank=l_rank,
                right_flank_rank=f_rank,
                classification=cls_name,
                ref_start=ref_s,
                ref_end=ref_e,
                normalized=is_norm,
                novel_unresolved_slices=tuple(chain_novel_slices.get(i, ())),
            )
        )

    return InversionNormalizationResult(
        normalized_fragment=normalized_fragment,
        chains=tuple(chain_records),
        unresolved_records=tuple(unresolved_records),
        unnormalized_branch_ranges=tuple(unnormalized_branch_ranges),
    )


def _solve_fenwick_lis_block(
    b_indices: Sequence[int] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    weights: Sequence[int] | np.ndarray,
) -> list[int]:
    """
    Exact Fenwick-tree weighted strictly increasing LIS on a block of forward anchors,
    with monotone unique-rank run contraction and LCA lexicographical `branch_step_idx` tie-breaking.
    Returns selected 0-based relative indices within the block.
    """
    n = len(ranks)
    if n == 0:
        return []
    if n == 1:
        return [0]

    b_arr = np.asarray(b_indices, dtype=np.int64)
    r_arr = np.asarray(ranks, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.int64)

    u_ranks, inv_cid_raw, counts = np.unique(
        r_arr, return_inverse=True, return_counts=True
    )
    inv_cid = inv_cid_raw.reshape(-1)

    can_fuse = (
        (counts[inv_cid[:-1]] == 1)
        & (counts[inv_cid[1:]] == 1)
        & (inv_cid[1:] == inv_cid[:-1] + 1)
        & (w_arr[:-1] > 0)
        & (w_arr[1:] > 0)
        & (b_arr[1:] > b_arr[:-1])
    )

    run_starts_list: list[int] | None = None
    run_ends_list: list[int] | None = None

    if bool(np.any(can_fuse)):
        split_pos = np.flatnonzero(~can_fuse) + 1
        n_runs = int(split_pos.shape[0]) + 1
        if n_runs == 1:
            return list(range(n))
        run_starts = np.empty(n_runs, dtype=np.int32)
        run_ends = np.empty(n_runs, dtype=np.int32)
        run_starts[0] = 0
        run_starts[1:] = split_pos
        run_ends[:-1] = split_pos
        run_ends[-1] = n

        b_idx_list = b_arr[run_starts].tolist()
        w_list = np.add.reduceat(w_arr, run_starts).tolist()
        u_run_ranks, run_inv_raw = np.unique(
            r_arr[run_starts], return_inverse=True
        )
        cid_list = (run_inv_raw.reshape(-1) + 1).tolist()
        u_len = int(u_run_ranks.shape[0])
        m = n_runs
        run_starts_list = run_starts.tolist()
        run_ends_list = run_ends.tolist()
    else:
        b_idx_list = b_arr.tolist()
        w_list = w_arr.tolist()
        cid_list = (inv_cid + 1).tolist()
        u_len = int(u_ranks.shape[0])
        m = n

    dp_weight: list[int] = [0] * m
    parent: list[int] = [-1] * m
    depth: list[int] = [0] * m
    first_step_idx: list[int] = [0] * m
    bit_best: list[int | None] = [None] * (u_len + 1)

    def _is_better(idx_a: int, idx_b: int) -> bool:
        if idx_a == idx_b:
            return False
        w_a = dp_weight[idx_a]
        w_b = dp_weight[idx_b]
        if w_a != w_b:
            return w_a > w_b
        f_a = first_step_idx[idx_a]
        f_b = first_step_idx[idx_b]
        if f_a != f_b:
            return f_a < f_b
        da = depth[idx_a]
        db = depth[idx_b]
        ca = idx_a
        cb = idx_b
        while da > db:
            ca = parent[ca]
            da -= 1
        while db > da:
            cb = parent[cb]
            db -= 1
        if ca == cb:
            return depth[idx_a] < depth[idx_b]
        while parent[ca] != parent[cb]:
            ca = parent[ca]
            cb = parent[cb]
        return b_idx_list[ca] < b_idx_list[cb]

    def _bit_query(max_cid: int) -> int | None:
        best: int | None = None
        idx = max_cid
        while idx > 0:
            cand = bit_best[idx]
            if cand is not None:
                if best is None or _is_better(cand, best):
                    best = cand
            idx -= idx & -idx
        return best

    def _bit_update(cid: int, elem_idx: int) -> None:
        idx = cid
        while idx <= u_len:
            cur = bit_best[idx]
            if cur is None or _is_better(elem_idx, cur):
                bit_best[idx] = elem_idx
            idx += idx & -idx

    for j in range(m):
        cid = cid_list[j]
        best_prev = _bit_query(cid - 1)
        w_j = w_list[j]
        if best_prev is None:
            dp_weight[j] = w_j
            parent[j] = -1
            depth[j] = 1
            first_step_idx[j] = b_idx_list[j]
        else:
            dp_weight[j] = dp_weight[best_prev] + w_j
            parent[j] = best_prev
            depth[j] = depth[best_prev] + 1
            first_step_idx[j] = first_step_idx[best_prev]
        _bit_update(cid, j)

    overall_best = _bit_query(u_len)
    if overall_best is None:
        return []

    chosen_runs: list[int] = []
    cur_idx = overall_best
    while cur_idx != -1:
        chosen_runs.append(cur_idx)
        cur_idx = parent[cur_idx]
    chosen_runs.reverse()

    if run_starts_list is not None and run_ends_list is not None:
        chosen: list[int] = []
        for r_i in chosen_runs:
            chosen.extend(range(run_starts_list[r_i], run_ends_list[r_i]))
        return chosen
    return chosen_runs


def compute_weighted_lis(
    anchors: Sequence[ProjectedAnchor],
    total_unique_anchors_in_b: int | None = None,
) -> tuple[Sequence[ProjectedAnchor], bool]:
    """
    §3.4.1 (F10): Node-length-weighted strictly increasing LIS on relative-forward anchors.

    - `multi_hit` is True iff any anchor (`rank_in_b` / `node_id`) appears more than once
      in `anchors` (across both `+` and `-` visits).
    - Filters to `rel_orient == "+"` anchors sorted by ascending `branch_step_idx`.
    - Finds a strictly rank-increasing subsequence (`r_1 < r_2 < ... < r_m`) maximizing
      total node base weight `sum(a.weight)`.
    - Tie-breaker: among chains achieving maximum weight, selects the one with the
      lexicographically smallest tuple of `branch_step_idx` values `(idx_1, idx_2, ...)`.
    """
    del total_unique_anchors_in_b
    if not anchors:
        return [], False

    if isinstance(anchors, LazyProjectedAnchorSequence):
        multi_hit = anchors.has_multi_hit
        fwd_indices = np.flatnonzero(anchors.rel_sign > 0).astype(np.int32)
        n = int(fwd_indices.shape[0])
        if n == 0:
            return [], multi_hit

        fwd_b_idx = anchors.branch_step_idx[fwd_indices]
        # Ensure sorted by ascending branch_step_idx
        if n > 1 and not bool(np.all(fwd_b_idx[:-1] <= fwd_b_idx[1:])):
            order = np.argsort(fwd_b_idx, kind="mergesort")
            fwd_indices = fwd_indices[order]
            fwd_b_idx = fwd_b_idx[order]

        fwd_ranks = anchors.rank_in_b[fwd_indices]
        fwd_weights = anchors.weights[fwd_indices]

        if n == 1:
            res_seq = anchors.subseq_by_indices(fwd_indices)
            return list(res_seq), multi_hit

        pref_max = np.maximum.accumulate(fwd_ranks)
        suff_min = np.minimum.accumulate(fwd_ranks[::-1])[::-1]
        is_cut = pref_max[:-1] < suff_min[1:]

        if bool(np.all(is_cut)):
            chosen_anc_indices = fwd_indices
        else:
            in_chain = np.ones(n, dtype=bool)
            not_cut = np.flatnonzero(~is_cut)
            gaps = np.flatnonzero(np.diff(not_cut) > 1)
            blk_starts = np.empty(int(gaps.shape[0]) + 1, dtype=np.int32)
            blk_ends = np.empty(int(gaps.shape[0]) + 1, dtype=np.int32)
            blk_starts[0] = not_cut[0]
            if int(gaps.shape[0]) > 0:
                blk_starts[1:] = not_cut[gaps + 1]
                blk_ends[:-1] = not_cut[gaps] + 2
            blk_ends[-1] = not_cut[-1] + 2

            for bs, be in zip(blk_starts.tolist(), blk_ends.tolist()):
                in_chain[bs:be] = False
                chosen_rel = _solve_fenwick_lis_block(
                    fwd_b_idx[bs:be],
                    fwd_ranks[bs:be],
                    fwd_weights[bs:be],
                )
                for rel_i in chosen_rel:
                    in_chain[bs + rel_i] = True

            chosen_anc_indices = fwd_indices[np.flatnonzero(in_chain)]

        res_seq = anchors.subseq_by_indices(chosen_anc_indices)
        if len(res_seq) <= 5_000:
            return list(res_seq), multi_hit
        return res_seq, multi_hit

    rank_counts = collections.Counter(a.rank_in_b for a in anchors)
    node_counts = collections.Counter(a.node_id for a in anchors)
    multi_hit = any(c > 1 for c in rank_counts.values()) or any(
        c > 1 for c in node_counts.values()
    )

    forward_anchors = sorted(
        (a for a in anchors if a.rel_orient == "+"),
        key=lambda a: a.branch_step_idx,
    )
    n = len(forward_anchors)
    if n == 0:
        return [], multi_hit

    chosen_indices = _solve_fenwick_lis_block(
        [a.branch_step_idx for a in forward_anchors],
        [a.rank_in_b for a in forward_anchors],
        [a.weight for a in forward_anchors],
    )
    return [forward_anchors[idx] for idx in chosen_indices], multi_hit


def extract_deviations(
    fragment: PathFragment,
    main_chain: Sequence[ProjectedAnchor],
    backbone_steps: Sequence[Step],
    is_top_level: bool = True,
    lo: int = 0,
    hi: int | None = None,
    unnormalized_branch_ranges: Sequence[tuple[int, int]] = (),
    all_projected_anchors: Sequence[ProjectedAnchor] | None = None,
    all_anchors: Sequence[ProjectedAnchor] | None = None,
) -> list[DeviationSegment]:
    """
    §3.4.2 & §3.5 (F11, T15): Extract deviation segments `D(a, b)` between adjacent main-chain anchors.
    """
    eff_hi = len(backbone_steps) if hi is None else hi
    provided_anchors = (
        all_projected_anchors if all_projected_anchors is not None else all_anchors
    )
    if isinstance(provided_anchors, LazyProjectedAnchorSequence):
        multi_hit_sorted = provided_anchors.multi_hit_nids
        multi_hit_set = set(multi_hit_sorted)
    elif provided_anchors is not None:
        anchor_counts = collections.Counter(a.node_id for a in provided_anchors)
        multi_hit_set = {nid for nid, cnt in anchor_counts.items() if cnt > 1}
        multi_hit_sorted = tuple(sorted(multi_hit_set))
    else:
        anchor_map = build_backbone_anchor_map(backbone_steps, lo=lo, hi=eff_hi)
        anchor_counts = collections.Counter(
            st.node_id
            for st in fragment.steps
            if _is_full_step(st) and st.node_id in anchor_map
        )
        multi_hit_set = {nid for nid, cnt in anchor_counts.items() if cnt > 1}
        multi_hit_sorted = tuple(sorted(multi_hit_set))

    fast_mh_tuple: tuple[str, ...] | None = (
        (multi_hit_sorted[0],) if len(multi_hit_sorted) > 64 else None
    )

    def _overlaps_unnormalized(b_start: int, b_end: int) -> bool:
        if not unnormalized_branch_ranges:
            return False
        for u_s, u_e in unnormalized_branch_ranges:
            if b_start == b_end:
                if u_s <= b_start <= u_e:
                    return True
            else:
                if max(b_start, u_s) < min(b_end, u_e):
                    return True
        return False

    def _local_multi_hit_nodes(
        left_anc: ProjectedAnchor | None,
        right_anc: ProjectedAnchor | None,
        br_steps: Sequence[Step],
        b_steps: Sequence[Step],
    ) -> tuple[str, ...]:
        if not multi_hit_sorted:
            return ()
        if fast_mh_tuple is not None:
            return fast_mh_tuple
        touched: set[str] = set()
        if left_anc is not None and left_anc.node_id in multi_hit_set:
            touched.add(left_anc.node_id)
        if right_anc is not None and right_anc.node_id in multi_hit_set:
            touched.add(right_anc.node_id)
        for st in br_steps:
            if _is_full_step(st) and st.node_id in multi_hit_set:
                touched.add(st.node_id)
        for st in b_steps:
            if _is_full_step(st) and st.node_id in multi_hit_set:
                touched.add(st.node_id)
        if not touched:
            return multi_hit_sorted
        return tuple(sorted(touched))

    deviations: list[DeviationSegment] = []
    n_frag = len(fragment.steps)
    use_lazy_b_slice = isinstance(backbone_steps, LazyStepTuple)

    if is_top_level:
        if not main_chain:
            if n_frag > 0 and not _overlaps_unnormalized(0, n_frag):
                br_steps = fragment.steps[0:n_frag]
                br_prov = fragment.provenance[0:n_frag]
                deviations.append(
                    DeviationSegment(
                        hap_id=fragment.hap_id,
                        src_path=fragment.src_path,
                        left_anchor_rank="START",
                        right_anchor_rank="END",
                        branch_start_idx=0,
                        branch_end_idx=n_frag,
                        branch_steps=br_steps,
                        branch_provenance=br_prov,
                        backbone_steps=(),
                        multi_hit_nodes=_local_multi_hit_nodes(None, None, br_steps, ()),
                        is_unanchored_start=True,
                        is_unanchored_end=True,
                    )
                )
            return deviations

        # Unanchored start prefix before main_chain[0]
        first_anc = main_chain[0]
        if first_anc.branch_step_idx > 0 and not _overlaps_unnormalized(
            0, first_anc.branch_step_idx
        ):
            br_steps = fragment.steps[0 : first_anc.branch_step_idx]
            br_prov = fragment.provenance[0 : first_anc.branch_step_idx]
            deviations.append(
                DeviationSegment(
                    hap_id=fragment.hap_id,
                    src_path=fragment.src_path,
                    left_anchor_rank="START",
                    right_anchor_rank=first_anc.rank_in_b,
                    branch_start_idx=0,
                    branch_end_idx=first_anc.branch_step_idx,
                    branch_steps=br_steps,
                    branch_provenance=br_prov,
                    backbone_steps=(),
                    multi_hit_nodes=_local_multi_hit_nodes(
                        None, first_anc, br_steps, ()
                    ),
                    is_unanchored_start=True,
                    is_unanchored_end=False,
                )
            )

        # Internal anchored pairs
        if (
            isinstance(main_chain, LazyProjectedAnchorSequence)
            and isinstance(fragment.steps, LazyStepTuple)
            and isinstance(backbone_steps, LazyStepTuple)
        ):
            mc_b = main_chain.branch_step_idx
            mc_r = main_chain.rank_in_b
            if int(mc_b.shape[0]) > 1:
                b_starts = mc_b[:-1] + 1
                b_ends = mc_b[1:]
                r_starts = mc_r[:-1] + 1
                r_ends = mc_r[1:]
                b_lens = b_ends - b_starts
                r_lens = r_ends - r_starts
                diff_mask = (b_lens != r_lens) | (b_lens > 0)
                one_step = (b_lens == 1) & (r_lens == 1)
                if bool(np.any(one_step)):
                    one_idx = np.flatnonzero(one_step)
                    same_one = (
                        fragment.steps._codes[b_starts[one_idx]]
                        == backbone_steps._codes[r_starts[one_idx]]
                    )
                    diff_mask[one_idx[same_one]] = False

                for k in np.flatnonzero(diff_mask).tolist():
                    b_start = int(b_starts[k])
                    b_end = int(b_ends[k])
                    if _overlaps_unnormalized(b_start, b_end):
                        continue
                    r_left = int(mc_r[k])
                    r_right = int(mc_r[k + 1])
                    br_steps = fragment.steps[b_start:b_end]
                    b_steps: Any = backbone_steps[r_left + 1 : r_right]
                    if br_steps != b_steps:
                        br_prov = fragment.provenance[b_start:b_end]
                        if fast_mh_tuple is not None:
                            mh_nodes = fast_mh_tuple
                        elif not multi_hit_sorted:
                            mh_nodes = ()
                        else:
                            mh_nodes = _local_multi_hit_nodes(
                                main_chain[k], main_chain[k + 1], br_steps, b_steps
                            )
                        deviations.append(
                            DeviationSegment(
                                hap_id=fragment.hap_id,
                                src_path=fragment.src_path,
                                left_anchor_rank=r_left,
                                right_anchor_rank=r_right,
                                branch_start_idx=b_start,
                                branch_end_idx=b_end,
                                branch_steps=br_steps,
                                branch_provenance=br_prov,
                                backbone_steps=b_steps,
                                multi_hit_nodes=mh_nodes,
                                is_unanchored_start=False,
                                is_unanchored_end=False,
                            )
                        )
        else:
            for k in range(len(main_chain) - 1):
                a_left = main_chain[k]
                a_right = main_chain[k + 1]
                b_start = a_left.branch_step_idx + 1
                b_end = a_right.branch_step_idx
                if _overlaps_unnormalized(b_start, b_end):
                    continue
                br_steps = fragment.steps[b_start:b_end]
                br_prov = fragment.provenance[b_start:b_end]
                b_steps = (
                    backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b]
                    if use_lazy_b_slice
                    else tuple(backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b])
                )
                if br_steps != b_steps:
                    deviations.append(
                        DeviationSegment(
                            hap_id=fragment.hap_id,
                            src_path=fragment.src_path,
                            left_anchor_rank=a_left.rank_in_b,
                            right_anchor_rank=a_right.rank_in_b,
                            branch_start_idx=b_start,
                            branch_end_idx=b_end,
                            branch_steps=br_steps,
                            branch_provenance=br_prov,
                            backbone_steps=b_steps,
                            multi_hit_nodes=_local_multi_hit_nodes(
                                a_left, a_right, br_steps, b_steps
                            ),
                            is_unanchored_start=False,
                            is_unanchored_end=False,
                        )
                    )

        # Unanchored end suffix after main_chain[-1]
        last_anc = main_chain[-1]
        if last_anc.branch_step_idx + 1 < n_frag and not _overlaps_unnormalized(
            last_anc.branch_step_idx + 1, n_frag
        ):
            b_start = last_anc.branch_step_idx + 1
            br_steps = fragment.steps[b_start:n_frag]
            br_prov = fragment.provenance[b_start:n_frag]
            deviations.append(
                DeviationSegment(
                    hap_id=fragment.hap_id,
                    src_path=fragment.src_path,
                    left_anchor_rank=last_anc.rank_in_b,
                    right_anchor_rank="END",
                    branch_start_idx=b_start,
                    branch_end_idx=n_frag,
                    branch_steps=br_steps,
                    branch_provenance=br_prov,
                    backbone_steps=(),
                    multi_hit_nodes=_local_multi_hit_nodes(
                        last_anc, None, br_steps, ()
                    ),
                    is_unanchored_start=False,
                    is_unanchored_end=True,
                )
            )
        return deviations

    # Recursive subproblem (is_top_level == False): START (at lo) and END (at eff_hi) are logical anchors
    if not main_chain:
        br_steps = fragment.steps[0:n_frag]
        br_prov = fragment.provenance[0:n_frag]
        b_steps = (
            backbone_steps[lo:eff_hi]
            if use_lazy_b_slice
            else tuple(backbone_steps[lo:eff_hi])
        )
        if br_steps != b_steps:
            deviations.append(
                DeviationSegment(
                    hap_id=fragment.hap_id,
                    src_path=fragment.src_path,
                    left_anchor_rank="START",
                    right_anchor_rank="END",
                    branch_start_idx=0,
                    branch_end_idx=n_frag,
                    branch_steps=br_steps,
                    branch_provenance=br_prov,
                    backbone_steps=b_steps,
                    multi_hit_nodes=_local_multi_hit_nodes(
                        None, None, br_steps, b_steps
                    ),
                    is_unanchored_start=False,
                    is_unanchored_end=False,
                )
            )
        return deviations

    # Prefix D(START, a_0)
    first_anc = main_chain[0]
    br_start_steps = fragment.steps[0 : first_anc.branch_step_idx]
    br_start_prov = fragment.provenance[0 : first_anc.branch_step_idx]
    b_start_steps: Any = (
        backbone_steps[lo : first_anc.rank_in_b]
        if use_lazy_b_slice
        else tuple(backbone_steps[lo : first_anc.rank_in_b])
    )
    if br_start_steps != b_start_steps:
        deviations.append(
            DeviationSegment(
                hap_id=fragment.hap_id,
                src_path=fragment.src_path,
                left_anchor_rank="START",
                right_anchor_rank=first_anc.rank_in_b,
                branch_start_idx=0,
                branch_end_idx=first_anc.branch_step_idx,
                branch_steps=br_start_steps,
                branch_provenance=br_start_prov,
                backbone_steps=b_start_steps,
                multi_hit_nodes=_local_multi_hit_nodes(
                    None, first_anc, br_start_steps, b_start_steps
                ),
                is_unanchored_start=False,
                is_unanchored_end=False,
            )
        )

    # Internal pairs D(a_k, a_{k+1})
    for k in range(len(main_chain) - 1):
        a_left = main_chain[k]
        a_right = main_chain[k + 1]
        b_start = a_left.branch_step_idx + 1
        b_end = a_right.branch_step_idx
        if b_start == b_end and a_left.rank_in_b + 1 == a_right.rank_in_b:
            continue
        br_steps = fragment.steps[b_start:b_end]
        br_prov = fragment.provenance[b_start:b_end]
        b_steps = (
            backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b]
            if use_lazy_b_slice
            else tuple(backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b])
        )
        if br_steps != b_steps:
            deviations.append(
                DeviationSegment(
                    hap_id=fragment.hap_id,
                    src_path=fragment.src_path,
                    left_anchor_rank=a_left.rank_in_b,
                    right_anchor_rank=a_right.rank_in_b,
                    branch_start_idx=b_start,
                    branch_end_idx=b_end,
                    branch_steps=br_steps,
                    branch_provenance=br_prov,
                    backbone_steps=b_steps,
                    multi_hit_nodes=_local_multi_hit_nodes(
                        a_left, a_right, br_steps, b_steps
                    ),
                    is_unanchored_start=False,
                    is_unanchored_end=False,
                )
            )

    # Suffix D(a_last, END)
    last_anc = main_chain[-1]
    b_start = last_anc.branch_step_idx + 1
    br_end_steps = fragment.steps[b_start:n_frag]
    br_end_prov = fragment.provenance[b_start:n_frag]
    b_end_steps: Any = (
        backbone_steps[last_anc.rank_in_b + 1 : eff_hi]
        if use_lazy_b_slice
        else tuple(backbone_steps[last_anc.rank_in_b + 1 : eff_hi])
    )
    if br_end_steps != b_end_steps:
        deviations.append(
            DeviationSegment(
                hap_id=fragment.hap_id,
                src_path=fragment.src_path,
                left_anchor_rank=last_anc.rank_in_b,
                right_anchor_rank="END",
                branch_start_idx=b_start,
                branch_end_idx=n_frag,
                branch_steps=br_end_steps,
                branch_provenance=br_end_prov,
                backbone_steps=b_end_steps,
                multi_hit_nodes=_local_multi_hit_nodes(
                    last_anc, None, br_end_steps, b_end_steps
                ),
                is_unanchored_start=False,
                is_unanchored_end=False,
            )
        )

    return deviations


def fold_tandem_cycles(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    global_ref_nodes: AbstractSet[str],
    max_copy: int = 1,
) -> CycleFoldResult:
    """
    §3.4.3 (F12, T17): Greedy left-to-right tandem repeat cycle folding on global non-reference tokens.
    """
    if len(steps) != len(provenance):
        raise InputContractError(
            f"steps length ({len(steps)}) != provenance length ({len(provenance)})"
        )
    if max_copy < 1:
        raise InputContractError(f"max_copy must be >= 1, got {max_copy}")

    n = len(steps)
    if n == 0:
        return CycleFoldResult(
            folded_steps=(),
            folded_provenance=(),
            piece_boundaries=(),
            deleted_intervals=(),
        )

    is_eligible = [
        _is_full_step(st) and (st.node_id not in global_ref_nodes) for st in steps
    ]
    tokens = [st.token for st in steps]

    # Index positions of each eligible token so we only test periods p where tokens[i] == tokens[i + p]
    pos_by_tok: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for idx_t, tok in enumerate(tokens):
        if is_eligible[idx_t]:
            pos_by_tok[tok].append(idx_t)

    # Fast return when no eligible token appears >= 2 times
    if all(len(plist) < 2 for plist in pos_by_tok.values()):
        return CycleFoldResult(
            folded_steps=tuple(steps),
            folded_provenance=tuple(provenance),
            piece_boundaries=(),
            deleted_intervals=(),
        )

    non_elig_prefix = [0] * (n + 1)
    for idx_e in range(n):
        non_elig_prefix[idx_e + 1] = non_elig_prefix[idx_e] + (
            0 if is_eligible[idx_e] else 1
        )

    folded_steps: list[Step] = []
    folded_prov: list[StepProvenance] = []
    piece_boundaries: list[int] = []
    deleted_intervals: list[tuple[StepProvenance, ...]] = []
    pending_break = False

    i = 0
    while i < n:
        found_p: int | None = None
        if is_eligible[i]:
            max_p = (n - i) // 2
            occ_list = pos_by_tok.get(tokens[i], ())
            if len(occ_list) >= 2:
                for next_pos in occ_list:
                    if next_pos <= i:
                        continue
                    p = next_pos - i
                    if p > max_p:
                        break
                    if (
                        non_elig_prefix[i + 2 * p] == non_elig_prefix[i]
                        and tokens[i : i + p] == tokens[i + p : i + 2 * p]
                    ):
                        found_p = p
                        break

        if found_p is None:
            if pending_break and len(folded_steps) > 0:
                piece_boundaries.append(len(folded_steps))
                pending_break = False
            folded_steps.append(steps[i])
            folded_prov.append(provenance[i])
            i += 1
            continue

        p = found_p
        k = 2
        while i + (k + 1) * p <= n:
            base = i + k * p
            if (
                non_elig_prefix[base + p] == non_elig_prefix[base]
                and tokens[base : base + p] == tokens[i : i + p]
            ):
                k += 1
            else:
                break

        keep_copies = min(k, max_copy)
        if pending_break and len(folded_steps) > 0 and keep_copies > 0:
            piece_boundaries.append(len(folded_steps))
            pending_break = False

        folded_steps.extend(steps[i : i + keep_copies * p])
        folded_prov.extend(provenance[i : i + keep_copies * p])

        if k > max_copy:
            deleted_intervals.append(
                tuple(provenance[i + max_copy * p : i + k * p])
            )
            pending_break = True

        i += k * p

    return CycleFoldResult(
        folded_steps=tuple(folded_steps),
        folded_provenance=tuple(folded_prov),
        piece_boundaries=tuple(piece_boundaries),
        deleted_intervals=tuple(deleted_intervals),
    )


def canonical_rotation(
    tokens: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Return the lexicographically smallest cyclic rotation of `tokens` (§3.4.4)."""
    if not tokens:
        return ()
    tok_tuple = tuple((str(nid), str(orient)) for nid, orient in tokens)
    m = len(tok_tuple)
    if m == 1:
        return tok_tuple
    min_tok = min(tok_tuple)
    cand_indices = [r for r, t in enumerate(tok_tuple) if t == min_tok]
    return min(tok_tuple[r:] + tok_tuple[:r] for r in cand_indices)


def _extract_full_period(
    steps: Sequence[Step],
) -> tuple[tuple[tuple[str, str], ...], int] | None:
    """
    Run the §3.4.3 greedy minimal period detector on `steps` (not restricted to non-ref nodes).

    - Excludes partial steps (`is_partial == True`).
    - If `steps` is `k >= 2` exact consecutive copies of minimal period `U`, returns
      `(canonical_rotation(U), k)`.
    - If `steps` has a tandem repeat starting at `i > 0` or leaving a trailing incomplete
      remainder (`k * p < len(steps)`), returns `None` (mixed / aperiodic).
    - If `steps` has no internal tandem repeat anywhere, returns `(canonical_rotation(tokens), 1)`
      so a 1-copy side can be compared against a `k >= 2` copy side.
    """
    n = len(steps)
    if n == 0:
        return None
    if not isinstance(steps, LazyStepTuple) and any(
        not _is_full_step(st) for st in steps
    ):
        return None

    tokens = tuple(st.token for st in steps)

    # Index positions of each token to avoid O(n^2) scans when tokens are distinct
    pos_by_tok: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for idx_t, tok in enumerate(tokens):
        pos_by_tok[tok].append(idx_t)

    if len(pos_by_tok) == n:
        # All tokens are distinct -> no tandem repeat anywhere
        return canonical_rotation(tokens), 1

    # Check if a tandem repeat starts at i = 0
    max_p = n // 2
    found_p: int | None = None
    first_occ = pos_by_tok[tokens[0]]
    if len(first_occ) >= 2:
        for next_pos in first_occ[1:]:
            p = next_pos
            if p > max_p:
                break
            if tokens[0:p] == tokens[p : 2 * p]:
                found_p = p
                break

    if found_p is not None:
        p = found_p
        k = 2
        while (k + 1) * p <= n and tokens[k * p : (k + 1) * p] == tokens[0:p]:
            k += 1
        if k * p == n:
            return canonical_rotation(tokens[0:p]), k
        return None

    # Ensure there is no tandem repeat starting at i > 0 (only check tokens appearing >= 2 times)
    for occ_list in pos_by_tok.values():
        if len(occ_list) < 2:
            continue
        m_occ = len(occ_list)
        for a_idx in range(m_occ - 1):
            i = occ_list[a_idx]
            if i == 0:
                continue
            rem_max_p = (n - i) // 2
            if rem_max_p < 1:
                break
            for b_idx in range(a_idx + 1, m_occ):
                p = occ_list[b_idx] - i
                if p > rem_max_p:
                    break
                if tokens[i : i + p] == tokens[i + p : i + 2 * p]:
                    return None

    return canonical_rotation(tokens), 1


def detect_canonical_cnv(
    ref_steps: Sequence[Step],
    alt_steps: Sequence[Step],
    nodes: Mapping[str, str],
    min_sv_len: int = 50,
) -> CNVCall | None:
    """
    §3.4.4 (F13, T10): Pre-folding canonical period CNV detector.
    """
    if not ref_steps or not alt_steps:
        return None

    if isinstance(ref_steps, LazyStepTuple) and isinstance(alt_steps, LazyStepTuple):
        r_codes = ref_steps._codes
        a_codes = alt_steps._codes
        # Fast filter: both sides must share the exact same set of signed step codes,
        # and at least one side must have length >= 2 with a repeated code.
        if int(r_codes.shape[0]) < 2 and int(a_codes.shape[0]) < 2:
            return None
        r_u, r_cnt = np.unique(r_codes, return_counts=True)
        a_u, a_cnt = np.unique(a_codes, return_counts=True)
        if not bool(np.array_equal(r_u, a_u)):
            return None
        if int(r_cnt.max()) < 2 and int(a_cnt.max()) < 2:
            return None

    ref_period = _extract_full_period(ref_steps)
    if ref_period is None:
        return None
    alt_period = _extract_full_period(alt_steps)
    if alt_period is None:
        return None

    u_ref, k_ref = ref_period
    u_alt, k_alt = alt_period
    if u_ref != u_alt:
        return None
    if k_ref < 2 and k_alt < 2:
        return None
    if k_ref == k_alt:
        return None

    if hasattr(nodes, "node_len"):
        unit_bp = sum(int(nodes.node_len(nid)) for nid, _ in u_ref)  # type: ignore[attr-defined]
    else:
        unit_bp = sum(len(nodes[nid]) for nid, _ in u_ref)
    delta_bp = abs(k_alt - k_ref) * unit_bp
    if delta_bp < min_sv_len:
        return None

    return CNVCall(
        unit_tokens=u_ref,
        unit_bp=unit_bp,
        k_ref=k_ref,
        k_alt=k_alt,
        delta_bp=delta_bp,
    )


def compute_raw_visit_counts(steps: Sequence[Step]) -> Mapping[str, int]:
    """Return raw visit counts per node_id across full steps in `steps` (§3.7)."""
    if isinstance(steps, LazyStepTuple) and len(steps) > 50_000:
        nids = np.abs(steps._codes)
        cnts = np.bincount(nids)
        multi_nids = np.flatnonzero(cnts >= 2).astype(np.int32)
        multi_counts = cnts[multi_nids].astype(np.int32)
        return CompactVisitCounts(multi_nids=multi_nids, multi_counts=multi_counts)

    counts: collections.Counter[str] = collections.Counter()
    for st in steps:
        if _is_full_step(st):
            counts[st.node_id] += 1
    return dict(counts)

