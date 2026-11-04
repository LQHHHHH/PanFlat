"""Top-level contig orientation, inversion normalization, weighted LIS, deviation extraction, cycle folding, and canonical CNV detection (§3.1–§3.5, §3.6.6, §3.7)."""

from __future__ import annotations

import collections
from typing import AbstractSet, Literal, Mapping, Sequence

from panflat.gfa_io import spell_steps
from panflat.models import (
    BuildConfig,
    CNVCall,
    CycleFoldResult,
    DeviationSegment,
    GFAGraph,
    InputContractError,
    InversionNormalizationResult,
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

    counts: collections.Counter[str] = collections.Counter()
    first_hit: dict[str, tuple[int, Literal["+", "-"]]] = {}
    for rank_idx in range(lo, eff_hi):
        st = backbone_steps[rank_idx]
        if not _is_full_step(st):
            continue
        counts[st.node_id] += 1
        if st.node_id not in first_hit:
            first_hit[st.node_id] = (rank_idx, st.orient)

    return {
        node_id: first_hit[node_id]
        for node_id, cnt in counts.items()
        if cnt == 1
    }


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
) -> list[ProjectedAnchor]:
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
        # Synthesize contiguous 0-based provenance when called via Form 4
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

    anchor_map = build_backbone_anchor_map(resolved_backbone, lo=eff_lo, hi=eff_hi)
    anchors: list[ProjectedAnchor] = []
    for idx, (st, prov) in enumerate(zip(resolved_branch_steps, resolved_branch_prov)):
        if not _is_full_step(st):
            continue
        if st.node_id not in anchor_map:
            continue
        rank_in_b, b_orient = anchor_map[st.node_id]
        rel_orient: Literal["+", "-"] = "+" if st.orient == b_orient else "-"
        weight = st.step_len(resolved_nodes)
        anchors.append(
            ProjectedAnchor(
                branch_step_idx=idx,
                rank_in_b=rank_in_b,
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
    forward_bp = sum(a.weight for a in anchors if a.rel_orient == "+")
    reverse_bp = sum(a.weight for a in anchors if a.rel_orient == "-")

    if reverse_bp <= forward_bp:
        return fragment

    flipped_steps = tuple(
        st.flipped(len(graph.nodes[st.node_id])) for st in reversed(fragment.steps)
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
    i = 0
    while i < n:
        if steps[i].node_id in global_ref_nodes:
            i += 1
            continue
        j = i + 1
        while j < n and steps[j].node_id not in global_ref_nodes:
            j += 1

        run_steps = steps[i:j]
        run_prov = provenance[i:j]

        # Group adjacent steps in run_steps that merge into a single contiguous StepProvenance
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

        i = j

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

    1. Projects `fragment` onto `graph.ref_fragment.steps`.
    2. Identifies maximal contiguous runs `R` of projected anchors with `rel_orient == "-"`
       and strictly decreasing `rank_in_b` (`r_1 > r_2 > ... > r_k`), allowing non-anchor
       steps in between and including single-anchor (`k = 1`) runs.
    3. Finds nearest preceding (`L`) and succeeding (`F`) forward anchors (`rel_orient == "+"`)
       on the same fragment, and checks rank-interval overlap `[rmin_i, rmax_i]` across all
       reverse chains on the fragment.
    4. Classifies each chain using the §3.3.2 4-row table:
       - Conflict -> `INV_CONFLICT` (`normalized = False`)
       - Missing `L` or `F` (no conflict) -> `INV_EDGE` (`normalized = False`)
       - `rank(L) < rmin` and `rank(F) > rmax` (no conflict) -> `INV` (`normalized = True`)
       - Otherwise (no conflict) -> `INV_DISPLACED` (`normalized = True`)
    5. Normalizes `INV` and `INV_DISPLACED` chains by reversing the closed branch step
       interval `[branch_start_idx, branch_end_idx]` and flipping each step's sign and
       each provenance entry's `src_strand` (preserving pre-flip `src_start..src_end`).
    6. Routes novel global non-reference runs (`>= config.min_sv_len`) inside unnormalized
       chains (`INV_EDGE`, `INV_CONFLICT`) to `UnresolvedRecord` (`export_to_fasta = True`).
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

    # Step 1: Extract maximal strictly decreasing relative-reverse anchor runs R
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

    # Precompute forward anchors in branch order for L / F lookup
    plus_anchors = [a for a in anchors if a.rel_orient == "+"]

    # Step 2 & 3: Compute bounds, L, F, conflicts, and 4-row classification
    chain_bounds: list[tuple[int, int, int, int, int | None, int | None, int, int]] = []
    for run in raw_runs:
        b_start = run[0].branch_step_idx
        b_end = run[-1].branch_step_idx
        rmin = run[-1].rank_in_b
        rmax = run[0].rank_in_b
        ref_start = graph.ref_step_ranges[rmin][0]
        ref_end = graph.ref_step_ranges[rmax][1]

        left_flank: int | None = None
        for pa in plus_anchors:
            if pa.branch_step_idx < b_start:
                left_flank = pa.rank_in_b
            else:
                break

        right_flank: int | None = None
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
    work_steps = list(fragment.steps)
    work_prov = list(fragment.provenance)
    for i in range(n_chains):
        cls_name, is_norm = classifications[i]
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

    normalized_fragment = PathFragment(
        sample=fragment.sample,
        hap=fragment.hap,
        hap_id=fragment.hap_id,
        seqid=fragment.seqid,
        src_path=fragment.src_path,
        base_offset=fragment.base_offset,
        steps=tuple(work_steps),
        provenance=tuple(work_prov),
        is_w_line=fragment.is_w_line,
        total_len=fragment.total_len,
        is_flipped_contig=fragment.is_flipped_contig,
    )

    # Step 5: Handle unnormalized chains (INV_EDGE, INV_CONFLICT)
    # Identify all forward ('+') anchor positions on normalized_fragment
    norm_anchors = project_steps_onto_backbone(
        normalized_fragment.steps,
        normalized_fragment.provenance,
        graph.ref_fragment.steps,
        graph.nodes,
    )
    norm_plus_indices = sorted(
        a.branch_step_idx for a in norm_anchors if a.rel_orient == "+"
    )

    # Group unnormalized chain indices by their enclosing (prev_plus_idx, next_plus_idx) gap
    gap_to_unnorm_chains: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
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


def compute_weighted_lis(
    anchors: Sequence[ProjectedAnchor],
    total_unique_anchors_in_b: int | None = None,
) -> tuple[list[ProjectedAnchor], bool]:
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

    sorted_ranks = sorted({a.rank_in_b for a in forward_anchors})
    u_len = len(sorted_ranks)
    rank_to_cid = {r: idx + 1 for idx, r in enumerate(sorted_ranks)}

    dp_weight: list[int] = [0] * n
    parent: list[int] = [-1] * n
    first_step_idx: list[int] = [0] * n
    bit_best: list[int | None] = [None] * (u_len + 1)

    def _reconstruct_indices(end_idx: int) -> tuple[int, ...]:
        seq_indices: list[int] = []
        cur = end_idx
        while cur != -1:
            seq_indices.append(forward_anchors[cur].branch_step_idx)
            cur = parent[cur]
        seq_indices.reverse()
        return tuple(seq_indices)

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
        return _reconstruct_indices(idx_a) < _reconstruct_indices(idx_b)

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

    for j, anc in enumerate(forward_anchors):
        cid = rank_to_cid[anc.rank_in_b]
        best_prev = _bit_query(cid - 1)
        if best_prev is None:
            dp_weight[j] = anc.weight
            parent[j] = -1
            first_step_idx[j] = anc.branch_step_idx
        else:
            dp_weight[j] = dp_weight[best_prev] + anc.weight
            parent[j] = best_prev
            first_step_idx[j] = first_step_idx[best_prev]
        _bit_update(cid, j)

    overall_best = _bit_query(u_len)
    if overall_best is None:
        return [], multi_hit

    chain: list[ProjectedAnchor] = []
    cur_idx = overall_best
    while cur_idx != -1:
        chain.append(forward_anchors[cur_idx])
        cur_idx = parent[cur_idx]
    chain.reverse()
    return chain, multi_hit


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

    - Compares open branch step slice `fragment.steps[a.branch_step_idx + 1 : b.branch_step_idx]`
      against `backbone_steps[a.rank_in_b + 1 : b.rank_in_b]`, emitting a `DeviationSegment`
      only when they differ (`T15`: identical repeat traversal emits no deviation).
    - Skips intervals that overlap `unnormalized_branch_ranges`.
    - Handles top-level unanchored contig ends (`is_top_level=True`) vs recursive subproblem
      logical `START`/`END` boundaries (`is_top_level=False`).
    """
    eff_hi = len(backbone_steps) if hi is None else hi
    provided_anchors = (
        all_projected_anchors if all_projected_anchors is not None else all_anchors
    )
    if provided_anchors is not None:
        anchor_counts = collections.Counter(a.node_id for a in provided_anchors)
    else:
        anchor_map = build_backbone_anchor_map(backbone_steps, lo=lo, hi=eff_hi)
        anchor_counts = collections.Counter(
            st.node_id
            for st in fragment.steps
            if _is_full_step(st) and st.node_id in anchor_map
        )
    multi_hit_set = {nid for nid, cnt in anchor_counts.items() if cnt > 1}

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
        if not multi_hit_set:
            return ()
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
            touched = set(multi_hit_set)
        return tuple(sorted(touched))

    deviations: list[DeviationSegment] = []
    n_frag = len(fragment.steps)

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
        for k in range(len(main_chain) - 1):
            a_left = main_chain[k]
            a_right = main_chain[k + 1]
            b_start = a_left.branch_step_idx + 1
            b_end = a_right.branch_step_idx
            if _overlaps_unnormalized(b_start, b_end):
                continue
            br_steps = fragment.steps[b_start:b_end]
            br_prov = fragment.provenance[b_start:b_end]
            b_steps = tuple(
                backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b]
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
        b_steps = tuple(backbone_steps[lo:eff_hi])
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
    b_start_steps = tuple(backbone_steps[lo : first_anc.rank_in_b])
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
        br_steps = fragment.steps[b_start:b_end]
        br_prov = fragment.provenance[b_start:b_end]
        b_steps = tuple(backbone_steps[a_left.rank_in_b + 1 : a_right.rank_in_b])
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
    b_end_steps = tuple(backbone_steps[last_anc.rank_in_b + 1 : eff_hi])
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

    - Only full steps (`is_partial == False`) whose `node_id not in global_ref_nodes` can
      participate in a repeat period `s[i : i + p]`.
    - At each index `i`, finds the smallest `p >= 1` such that `s[i : i + p] == s[i + p : i + 2*p]`
      and all steps in `s[i : i + 2*p]` are eligible.
    - Counts `k >= 2` consecutive complete repeats of period `p` starting at `i`.
    - Keeps `min(k, max_copy)` copies; if `k > max_copy`, deletes copies `max_copy + 1 .. k`,
      records their provenance in `deleted_intervals`, and records a split index in
      `piece_boundaries` if any subsequent step is emitted after the deletion.
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

    folded_steps: list[Step] = []
    folded_prov: list[StepProvenance] = []
    piece_boundaries: list[int] = []
    deleted_intervals: list[tuple[StepProvenance, ...]] = []
    pending_break = False

    i = 0
    while i < n:
        found_p: int | None = None
        max_p = (n - i) // 2
        for p in range(1, max_p + 1):
            if all(is_eligible[i + offset] for offset in range(2 * p)) and (
                tokens[i : i + p] == tokens[i + p : i + 2 * p]
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
            if all(is_eligible[base + offset] for offset in range(p)) and (
                tokens[base : base + p] == tokens[i : i + p]
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
    return min(tok_tuple[r:] + tok_tuple[:r] for r in range(m))


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
    if not steps or any(not _is_full_step(st) for st in steps):
        return None

    tokens = tuple(st.token for st in steps)
    n = len(tokens)

    # Check if a tandem repeat starts at i = 0
    max_p = n // 2
    found_p: int | None = None
    for p in range(1, max_p + 1):
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

    # Ensure there is no tandem repeat starting at i > 0 (which would mean a non-repeat prefix)
    for i in range(1, n):
        rem_max_p = (n - i) // 2
        for p in range(1, rem_max_p + 1):
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

    - Runs greedy minimal period detection on `ref_steps` (`B(a,b)`) and `alt_steps` (`D(a,b)`)
      without restricting to non-reference nodes (partial steps `is_partial=True` excluded).
    - Supports `k = 1` vs `k >= 2` when the 1-copy side matches a cyclic rotation of the
      multi-copy unit `U`.
    - Normalizes both repeat units to `canonical_u = canonical_rotation(U)`.
    - Emits `CNVCall` iff both sides share the same `canonical_u`, at least one side has
      `k >= 2`, `k_ref != k_alt`, and `abs(k_alt - k_ref) * unit_bp >= min_sv_len`.
    """
    ref_period = _extract_full_period(ref_steps)
    alt_period = _extract_full_period(alt_steps)
    if ref_period is None or alt_period is None:
        return None

    u_ref, k_ref = ref_period
    u_alt, k_alt = alt_period
    if u_ref != u_alt:
        return None
    if k_ref < 2 and k_alt < 2:
        return None
    if k_ref == k_alt:
        return None

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


def compute_raw_visit_counts(steps: Sequence[Step]) -> dict[str, int]:
    """Return raw visit counts per node_id across full steps in `steps` (§3.7)."""
    counts: collections.Counter[str] = collections.Counter()
    for st in steps:
        if _is_full_step(st):
            counts[st.node_id] += 1
    return dict(counts)
