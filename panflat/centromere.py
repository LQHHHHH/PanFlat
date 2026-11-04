"""Two-tier centromere region delimitation, completeness accounting, bypass mode, and extract-islands Steps A–D (§3.8, §5.1–§5.5)."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import AbstractSet, Any, Literal, Mapping, Sequence

import mappy
import numpy as np

from panflat.gfa_io import parse_bed, spell_steps, validate_build_config
from panflat.models import (
    BuildConfig,
    CentromereHapSlice,
    CentromereOutput,
    CentromerePartitionResult,
    CentromereRegion,
    CentromereRow,
    CompactRefNodeSet,
    GFAGraph,
    InsertionSlotBlock,
    InvariantViolationError,
    LazyProvenanceTuple,
    LazyStepTuple,
    PathFragment,
    Piece,
    Step,
    StepProvenance,
    SVEvent,
    UnresolvedRecord,
)
from panflat.repeat_guard import (
    REPEAT_KMER_SIZE,
    REPEAT_MIN_SEQ_LEN,
    REPEAT_SINGLETON_THRESHOLD,
    REPEAT_VALID_POS_RATIO,
    REPEAT_WINDOW_SIZE,
    REPEAT_WINDOW_STEP,
    REF_SUB_BLOCK_SIZE,
    REF_SUB_MIN_ALIGNED_RATIO,
    REF_SUB_MIN_IDENTITY,
    check_similarity_merge,
    complement_intervals,
    merge_contiguous_provenance,
    repeat_mask_intervals,
    slice_candidate_by_intervals,
    subtract_reference_intervals_mappy,
)


def _is_full_step(step: Step) -> bool:
    return not step.is_partial and step.slice_start == 0 and step.slice_end is None


def _dedup_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def delimit_centromere_regions(
    graph: GFAGraph,
    config: BuildConfig,
) -> tuple[list[CentromereRegion], list[str]]:
    """
    §3.8.1 (F21, C2, C5): Delimit centromere regions from `--centromere-bed`.
    """
    validate_build_config(config)
    eff_mode = config.effective_centromere_mode
    if eff_mode == "off" or config.centromere_bed is None:
        return [], []

    ref_len = len(graph.ref_seq)
    raw_bed = parse_bed(config.centromere_bed, graph.ref_chrom, ref_len)

    # Sort raw BED intervals by (start, end, original_index) and merge overlapping/touching
    indexed_bed = sorted(
        [(s, e, bid, idx) for idx, (_, s, e, bid) in enumerate(raw_bed)],
        key=lambda x: (x[0], x[1], x[3]),
    )
    merged_bed: list[tuple[int, int, list[str]]] = []
    for s, e, bid, _ in indexed_bed:
        if not merged_bed:
            merged_bed.append((s, e, [bid]))
            continue
        prev_s, prev_e, prev_ids = merged_bed[-1]
        if s <= prev_e:
            merged_bed[-1] = (
                prev_s,
                max(prev_e, e),
                _dedup_preserve_order([*prev_ids, bid]),
            )
        else:
            merged_bed.append((s, e, [bid]))

    if graph._nid_to_ref_rank is not None and graph._ref_cum_offsets is not None:
        uniq_steps_arr = np.sort(
            graph._nid_to_ref_rank[graph._nid_to_ref_rank >= 0]
        )
        uniq_starts = graph._ref_cum_offsets[uniq_steps_arr]
        uniq_ends = graph._ref_cum_offsets[uniq_steps_arr + 1]
    else:
        uniq_steps_arr = None
        uniq_starts = None
        uniq_ends = None
        unique_ref_steps = sorted(graph.unique_ref_step_by_node.values())

    def _expand_interval(
        bed_s: int,
        bed_e: int,
    ) -> tuple[int | None, int | None, int, int]:
        if uniq_steps_arr is not None and uniq_starts is not None and uniq_ends is not None:
            n_u = int(uniq_steps_arr.shape[0])
            u_pos = int(np.searchsorted(uniq_ends, bed_s, side="right")) - 1
            u_step = int(uniq_steps_arr[u_pos]) if u_pos >= 0 else None
            v_pos = int(np.searchsorted(uniq_starts, bed_e, side="left"))
            v_step = int(uniq_steps_arr[v_pos]) if v_pos < n_u else None
        else:
            u_step = None
            for r_idx in unique_ref_steps:
                if graph.ref_step_ranges[r_idx][1] <= bed_s:
                    u_step = r_idx
                else:
                    break

            v_step = None
            for r_idx in unique_ref_steps:
                if graph.ref_step_ranges[r_idx][0] >= bed_e:
                    v_step = r_idx
                    break

        eff_start = graph.ref_step_ranges[u_step][1] if u_step is not None else 0
        eff_end = graph.ref_step_ranges[v_step][0] if v_step is not None else ref_len
        return u_step, v_step, eff_start, eff_end

    # Initial outward expansion
    expanded: list[tuple[int, int, list[str], int | None, int | None, int, int]] = []
    for bed_s, bed_e, bids in merged_bed:
        u_st, v_st, eff_s, eff_e = _expand_interval(bed_s, bed_e)
        expanded.append((bed_s, bed_e, bids, u_st, v_st, eff_s, eff_e))

    # Iterative merging of overlapping or touching expanded intervals
    while True:
        expanded.sort(key=lambda item: (item[5], item[6], item[0], item[1]))
        next_expanded: list[
            tuple[int, int, list[str], int | None, int | None, int, int]
        ] = []
        merged_any = False
        for item in expanded:
            if not next_expanded:
                next_expanded.append(item)
                continue
            prev = next_expanded[-1]
            if item[5] <= prev[6]:
                new_bed_s = min(prev[0], item[0])
                new_bed_e = max(prev[1], item[1])
                new_bids = _dedup_preserve_order([*prev[2], *item[2]])
                u_st, v_st, eff_s, eff_e = _expand_interval(new_bed_s, new_bed_e)
                next_expanded[-1] = (
                    new_bed_s,
                    new_bed_e,
                    new_bids,
                    u_st,
                    v_st,
                    eff_s,
                    eff_e,
                )
                merged_any = True
            else:
                next_expanded.append(item)
        expanded = next_expanded
        if not merged_any:
            break

    regions: list[CentromereRegion] = []
    log_lines: list[str] = []
    for idx, (bed_s, bed_e, bids, u_st, v_st, eff_s, eff_e) in enumerate(
        expanded, start=1
    ):
        reg_id = f"CEN_{idx}"
        u_name = str(u_st) if u_st is not None else "VIRTUAL_START"
        v_name = str(v_st) if v_st is not None else "VIRTUAL_END"
        anchor_not_found = (u_st is None) or (v_st is None)
        expanded_whole = (u_st is None) and (v_st is None)
        reg_mode: Literal["bypass", "extract-islands"] = (
            "bypass" if anchor_not_found else eff_mode
        )

        reg = CentromereRegion(
            region_id=reg_id,
            chrom=graph.ref_chrom,
            input_bed_ids=tuple(bids),
            bed_start=bed_s,
            bed_end=bed_e,
            ref_start=eff_s,
            ref_end=eff_e,
            anchor_u_step=u_st,
            anchor_v_step=v_st,
            anchor_u_name=u_name,
            anchor_v_name=v_name,
            effective_mode=reg_mode,
            anchor_not_found=anchor_not_found,
            expanded_to_whole_chromosome=expanded_whole,
        )
        regions.append(reg)

        if anchor_not_found:
            log_lines.append(
                f"WARNING CEN_ANCHOR_NOT_FOUND region_id={reg_id} chrom={graph.ref_chrom} "
                f"anchor_u={u_name} anchor_v={v_name} effective_mode={reg_mode} "
                f"expanded_to_whole_chromosome={expanded_whole}"
            )
        log_lines.append(
            f"INFO CENTROMERE_REGION region_id={reg_id} chrom={graph.ref_chrom} "
            f"input_bed_ids={','.join(bids)} bed={bed_s}-{bed_e} ref={eff_s}-{eff_e} "
            f"anchor_u={u_name} anchor_v={v_name} mode={reg_mode}"
        )

    return regions, log_lines


def _slice_path_fragment(
    frag: PathFragment,
    start_idx: int,
    end_idx: int,
) -> PathFragment:
    """Return a sub-fragment covering `frag.steps[start_idx:end_idx]`."""
    sub_steps = frag.steps[start_idx:end_idx]
    sub_prov = frag.provenance[start_idx:end_idx]
    if isinstance(sub_prov, LazyProvenanceTuple):
        total_len = sub_prov.total_bp()
        base_offset = sub_prov.min_src_start() if len(sub_prov) > 0 else frag.base_offset
    else:
        total_len = sum(p.length for p in sub_prov)
        base_offset = min((p.src_start for p in sub_prov), default=frag.base_offset)
    return PathFragment(
        sample=frag.sample,
        hap=frag.hap,
        hap_id=frag.hap_id,
        seqid=frag.seqid,
        src_path=frag.src_path,
        base_offset=base_offset,
        steps=sub_steps,
        provenance=sub_prov,
        is_w_line=frag.is_w_line,
        total_len=total_len,
        is_flipped_contig=frag.is_flipped_contig,
    )


def _prov_min_start_max_end(
    prov: Sequence[StepProvenance],
) -> tuple[int, int]:
    if isinstance(prov, LazyProvenanceTuple):
        return prov.min_src_start(), prov.max_src_end()
    return min(p.src_start for p in prov), max(p.src_end for p in prov)


@dataclass(frozen=True)
class _NonRefRun:
    steps: tuple[Step, ...] | Any
    prov: tuple[StepProvenance, ...] | Any
    seq: str
    src_hap: str
    src_path: str
    src_start: int
    src_end: int
    src_strand: Literal["+", "-"]


def _extract_nonref_runs(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    nodes: Mapping[str, str],
    global_ref_nodes: AbstractSet[str],
) -> list[_NonRefRun]:
    """
    Extract maximal contiguous runs of global non-reference steps (`node_id not in global_ref_nodes`),
    splitting at any non-contiguous source provenance boundaries (§3.8.4-A).
    """
    runs: list[_NonRefRun] = []
    n = len(steps)
    if n == 0:
        return runs

    if isinstance(steps, LazyStepTuple) and isinstance(
        global_ref_nodes, CompactRefNodeSet
    ):
        codes = steps._codes
        nids = np.abs(codes)
        ref_mask = global_ref_nodes._mask[nids]
        nonref_indices = np.flatnonzero(~ref_mask)
        if int(nonref_indices.shape[0]) == 0:
            return runs
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
        raw_steps = steps[i:j]
        raw_prov = provenance[i:j]

        if isinstance(raw_prov, LazyProvenanceTuple) and not raw_prov.has_internal_discontinuity():
            sub_seq = spell_steps(raw_steps, nodes)
            merged = merge_contiguous_provenance(raw_prov, sub_seq)
            if merged:
                m_prov, m_seq = merged[0]
                runs.append(
                    _NonRefRun(
                        steps=raw_steps,
                        prov=raw_prov,
                        seq=m_seq,
                        src_hap=m_prov.src_hap,
                        src_path=m_prov.src_path,
                        src_start=m_prov.src_start,
                        src_end=m_prov.src_end,
                        src_strand=m_prov.src_strand,
                    )
                )
            continue

        grp_s = 0
        while grp_s < len(raw_steps):
            grp_e = grp_s + 1
            while grp_e < len(raw_steps):
                p_prev = raw_prov[grp_e - 1]
                p_curr = raw_prov[grp_e]
                is_adj = (
                    p_prev.src_hap == p_curr.src_hap
                    and p_prev.src_path == p_curr.src_path
                    and p_prev.src_strand == p_curr.src_strand
                    and (
                        (p_prev.src_strand == "+" and p_prev.src_end == p_curr.src_start)
                        or (
                            p_prev.src_strand == "-"
                            and p_prev.src_start == p_curr.src_end
                        )
                    )
                )
                if is_adj:
                    grp_e += 1
                else:
                    break

            sub_steps = tuple(raw_steps[grp_s:grp_e])
            sub_prov = tuple(raw_prov[grp_s:grp_e])
            sub_seq = spell_steps(sub_steps, nodes)
            merged = merge_contiguous_provenance(sub_prov, sub_seq)
            if merged:
                m_prov, m_seq = merged[0]
                runs.append(
                    _NonRefRun(
                        steps=sub_steps,
                        prov=sub_prov,
                        seq=m_seq,
                        src_hap=m_prov.src_hap,
                        src_path=m_prov.src_path,
                        src_start=m_prov.src_start,
                        src_end=m_prov.src_end,
                        src_strand=m_prov.src_strand,
                    )
                )
            grp_s = grp_e

    return runs


def partition_fragments_by_centromere(
    oriented_fragments: Sequence[PathFragment],
    regions: Sequence[CentromereRegion],
    graph: GFAGraph,
    config: BuildConfig | None = None,
) -> CentromerePartitionResult:
    """
    §3.8.1–§3.8.2 (F21, F22, C3, C4, C5, C14):
    Partition oriented fragments at centromere region anchors `(u, v)`.
    """
    alt_oriented = [
        f for f in oriented_fragments if f.src_path != graph.ref_fragment.src_path
    ]
    if not regions:
        return CentromerePartitionResult(
            regions=(),
            hap_slices={},
            normal_fragments=tuple(alt_oriented),
            claimed_source_intervals=set(),
            boundary_ambiguous_records=(),
        )

    min_sv_len = config.min_sv_len if config is not None else 50

    frags_by_hap: dict[str, list[PathFragment]] = collections.defaultdict(list)
    frags_by_hap[graph.ref_fragment.hap_id].append(graph.ref_fragment)
    for frag in alt_oriented:
        frags_by_hap[frag.hap_id].append(frag)

    hap_slices: dict[tuple[str, str], CentromereHapSlice] = {}
    claimed_source_intervals: set[tuple[str, int, int]] = set()
    clean_cuts_by_frag: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)

    is_lazy_ref = isinstance(graph.ref_fragment.steps, LazyStepTuple) and (
        graph._node_lens is not None
    )

    for reg in regions:
        l_ref = reg.ref_end - reg.ref_start
        ref_subseq = graph.ref_seq[reg.ref_start : reg.ref_end]
        ref_status: Literal["complete", "gapped"] = (
            "gapped" if "N" in ref_subseq.upper() else "complete"
        )

        u_ref_step = reg.anchor_u_step
        v_ref_step = reg.anchor_v_step
        u_node = (
            graph.ref_fragment.steps[u_ref_step].node_id
            if u_ref_step is not None
            else None
        )
        u_orient = (
            graph.ref_fragment.steps[u_ref_step].orient
            if u_ref_step is not None
            else None
        )
        v_node = (
            graph.ref_fragment.steps[v_ref_step].node_id
            if v_ref_step is not None
            else None
        )
        v_orient = (
            graph.ref_fragment.steps[v_ref_step].orient
            if v_ref_step is not None
            else None
        )

        r_inner_start = 0 if u_ref_step is None else u_ref_step + 1
        r_inner_end = (
            len(graph.ref_fragment.steps) if v_ref_step is None else v_ref_step
        )

        if is_lazy_ref:
            ref_nids = np.abs(graph.ref_fragment.steps._codes)
            max_nid_len = int(graph._node_lens.shape[0])
            outside_mask = np.zeros(max_nid_len, dtype=bool)
            if r_inner_start > 0:
                outside_mask[ref_nids[:r_inner_start]] = True
            if r_inner_end < int(ref_nids.shape[0]):
                outside_mask[ref_nids[r_inner_end:]] = True
            internal_only_mask = np.zeros(max_nid_len, dtype=bool)
            if r_inner_start < r_inner_end:
                internal_only_mask[ref_nids[r_inner_start:r_inner_end]] = True
            internal_only_mask &= ~outside_mask
            outside_ref_nodes: AbstractSet[str] = CompactRefNodeSet(outside_mask)
            internal_only_ref_nodes: AbstractSet[str] = CompactRefNodeSet(
                internal_only_mask
            )
            u_nid_int = int(u_node) if (u_node is not None and u_node.isdigit()) else -1
            v_nid_int = int(v_node) if (v_node is not None and v_node.isdigit()) else -1
        else:
            outside_mask = None
            internal_only_mask = None
            u_nid_int = -1
            v_nid_int = -1
            outside_ref_nodes = {
                graph.ref_fragment.steps[idx].node_id
                for idx in range(len(graph.ref_fragment.steps))
                if idx < r_inner_start or idx >= r_inner_end
            }
            internal_only_ref_nodes = (
                {
                    graph.ref_fragment.steps[idx].node_id
                    for idx in range(r_inner_start, r_inner_end)
                }
                - outside_ref_nodes
            )

        for hap_id in graph.all_haplotypes:
            hap_frags = frags_by_hap.get(hap_id, [])

            if (
                hap_id == graph.ref_fragment.hap_id
                and len(hap_frags) == 1
                and hap_frags[0].src_path == graph.ref_fragment.src_path
                and (u_ref_step is None or v_ref_step is None)
            ):
                inner_steps = graph.ref_fragment.steps[r_inner_start:r_inner_end]
                inner_prov = graph.ref_fragment.provenance[r_inner_start:r_inner_end]
                ratio = (
                    (l_ref / l_ref)
                    if (ref_status == "complete" and l_ref > 0)
                    else None
                )
                hap_slices[(reg.region_id, hap_id)] = CentromereHapSlice(
                    region_id=reg.region_id,
                    hap_id=hap_id,
                    status=ref_status,
                    ref_status=ref_status,
                    fragment=graph.ref_fragment,
                    u_step_idx=u_ref_step,
                    v_step_idx=v_ref_step,
                    inner_steps=inner_steps,
                    inner_provenance=inner_prov,
                    src_path=graph.ref_fragment.src_path,
                    src_start=reg.ref_start,
                    src_end=reg.ref_end,
                    src_strand="+",
                    l_ref=l_ref,
                    l_path=l_ref,
                    length_ratio=ratio,
                    extra_inner_slices=(),
                )
                continue

            valid_traversals: list[tuple[PathFragment, int, int]] = []
            any_frag_ambiguous = False
            single_side_frags: list[tuple[PathFragment, list[int], list[int]]] = []
            total_u_hits = 0
            total_v_hits = 0
            any_internal_hit = False

            for frag in hap_frags:
                if is_lazy_ref and isinstance(frag.steps, LazyStepTuple):
                    f_nids = np.abs(frag.steps._codes)
                    u_hits = (
                        np.flatnonzero(f_nids == u_nid_int).tolist()
                        if u_nid_int > 0
                        else []
                    )
                    v_hits = (
                        np.flatnonzero(f_nids == v_nid_int).tolist()
                        if v_nid_int > 0
                        else []
                    )
                    assert internal_only_mask is not None
                    if bool(np.any(internal_only_mask[f_nids])):
                        any_internal_hit = True
                else:
                    u_hits = [
                        i
                        for i, st in enumerate(frag.steps)
                        if _is_full_step(st) and u_node is not None and st.node_id == u_node
                    ]
                    v_hits = [
                        i
                        for i, st in enumerate(frag.steps)
                        if _is_full_step(st) and v_node is not None and st.node_id == v_node
                    ]
                    if any(
                        _is_full_step(st) and st.node_id in internal_only_ref_nodes
                        for st in frag.steps
                    ):
                        any_internal_hit = True

                total_u_hits += len(u_hits)
                total_v_hits += len(v_hits)

                is_valid = (
                    u_ref_step is not None
                    and v_ref_step is not None
                    and len(u_hits) == 1
                    and len(v_hits) == 1
                    and u_hits[0] < v_hits[0]
                    and frag.steps[u_hits[0]].orient == u_orient
                    and frag.steps[v_hits[0]].orient == v_orient
                )
                if is_valid:
                    valid_traversals.append((frag, u_hits[0], v_hits[0]))
                elif (
                    len(u_hits) > 1
                    or len(v_hits) > 1
                    or (len(u_hits) == 1 and len(v_hits) == 1 and not is_valid)
                ):
                    any_frag_ambiguous = True
                elif (len(u_hits) == 1 and len(v_hits) == 0) or (
                    len(u_hits) == 0 and len(v_hits) == 1
                ):
                    single_side_frags.append((frag, u_hits, v_hits))

            is_ambiguous = (
                u_ref_step is not None
                and v_ref_step is not None
                and (
                    any_frag_ambiguous
                    or len(valid_traversals) >= 2
                    or (
                        len(valid_traversals) == 1
                        and (
                            len(single_side_frags) > 0
                            or total_u_hits > 1
                            or total_v_hits > 1
                        )
                    )
                )
            )

            def _collect_single_side_slice(
                frag: PathFragment,
                u_hits: list[int],
                v_hits: list[int],
                extra_slices: list[tuple[Any, Any]],
            ) -> None:
                if len(u_hits) == 1 and len(v_hits) == 0:
                    u_i = u_hits[0]
                    if frag.steps[u_i].orient == u_orient and u_i + 1 < len(frag.steps):
                        left_bound = u_i
                        if (
                            u_ref_step is not None
                            and v_ref_step is not None
                            and u_ref_step + 1 < v_ref_step
                        ):
                            adj_ref_st = graph.ref_fragment.steps[u_ref_step + 1]
                            if adj_ref_st.node_id in graph.unique_ref_step_by_node:
                                for cand_idx in range(u_i + 1, len(frag.steps)):
                                    st_c = frag.steps[cand_idx]
                                    if (
                                        _is_full_step(st_c)
                                        and st_c.node_id == adj_ref_st.node_id
                                        and st_c.orient == adj_ref_st.orient
                                    ):
                                        if all(
                                            s.node_id not in graph.global_ref_nodes
                                            for s in frag.steps[u_i + 1 : cand_idx]
                                        ):
                                            left_bound = cand_idx
                                        break
                        tail_steps = frag.steps[left_bound + 1 :]
                        if (
                            tail_steps
                            and not any(
                                st.node_id in outside_ref_nodes for st in tail_steps
                            )
                            and any(
                                st.node_id in internal_only_ref_nodes
                                for st in tail_steps
                            )
                        ):
                            last_int = max(
                                idx
                                for idx in range(left_bound + 1, len(frag.steps))
                                if frag.steps[idx].node_id in internal_only_ref_nodes
                            )
                            sub_st = frag.steps[left_bound + 1 : last_int]
                            sub_pr = frag.provenance[left_bound + 1 : last_int]
                            if sub_st:
                                extra_slices.append((sub_st, sub_pr))
                                s_start, s_end = _prov_min_start_max_end(sub_pr)
                                claimed_source_intervals.add(
                                    (frag.src_path, s_start, s_end)
                                )
                            if frag.src_path != graph.ref_fragment.src_path:
                                clean_cuts_by_frag[id(frag)].append(
                                    (left_bound, last_int)
                                )
                elif len(v_hits) == 1 and len(u_hits) == 0:
                    v_i = v_hits[0]
                    if frag.steps[v_i].orient == v_orient and v_i > 0:
                        right_bound = v_i
                        if (
                            u_ref_step is not None
                            and v_ref_step is not None
                            and u_ref_step + 1 < v_ref_step
                        ):
                            adj_ref_st = graph.ref_fragment.steps[v_ref_step - 1]
                            if adj_ref_st.node_id in graph.unique_ref_step_by_node:
                                for cand_idx in range(v_i - 1, -1, -1):
                                    st_c = frag.steps[cand_idx]
                                    if (
                                        _is_full_step(st_c)
                                        and st_c.node_id == adj_ref_st.node_id
                                        and st_c.orient == adj_ref_st.orient
                                    ):
                                        if all(
                                            s.node_id not in graph.global_ref_nodes
                                            for s in frag.steps[cand_idx + 1 : v_i]
                                        ):
                                            right_bound = cand_idx
                                        break
                        head_steps = frag.steps[:right_bound]
                        if (
                            head_steps
                            and not any(
                                st.node_id in outside_ref_nodes for st in head_steps
                            )
                            and any(
                                st.node_id in internal_only_ref_nodes
                                for st in head_steps
                            )
                        ):
                            first_int = min(
                                idx
                                for idx in range(0, right_bound)
                                if frag.steps[idx].node_id in internal_only_ref_nodes
                            )
                            sub_st = frag.steps[first_int + 1 : right_bound]
                            sub_pr = frag.provenance[first_int + 1 : right_bound]
                            if sub_st:
                                extra_slices.append((sub_st, sub_pr))
                                s_start, s_end = _prov_min_start_max_end(sub_pr)
                                claimed_source_intervals.add(
                                    (frag.src_path, s_start, s_end)
                                )
                            if frag.src_path != graph.ref_fragment.src_path:
                                clean_cuts_by_frag[id(frag)].append(
                                    (first_int, right_bound)
                                )

            if is_ambiguous:
                extra_slices: list[tuple[Any, Any]] = []
                for frag, u_i, v_i in valid_traversals:
                    s_steps = frag.steps[u_i + 1 : v_i]
                    s_prov = frag.provenance[u_i + 1 : v_i]
                    if s_steps:
                        extra_slices.append((s_steps, s_prov))
                        s_start, s_end = _prov_min_start_max_end(s_prov)
                        claimed_source_intervals.add((frag.src_path, s_start, s_end))
                    if frag.src_path != graph.ref_fragment.src_path:
                        clean_cuts_by_frag[id(frag)].append((u_i, v_i))

                for frag, u_hits, v_hits in single_side_frags:
                    if not any(frag is vt[0] for vt in valid_traversals):
                        _collect_single_side_slice(frag, u_hits, v_hits, extra_slices)

                for frag in hap_frags:
                    if any(frag is vt[0] for vt in valid_traversals):
                        continue
                    if is_lazy_ref and isinstance(frag.steps, LazyStepTuple):
                        f_nids = np.abs(frag.steps._codes)
                        uv_mask = np.zeros(int(f_nids.shape[0]), dtype=bool)
                        if u_nid_int > 0:
                            uv_mask |= f_nids == u_nid_int
                        if v_nid_int > 0:
                            uv_mask |= f_nids == v_nid_int
                        uv_indices = np.flatnonzero(uv_mask).tolist()
                    else:
                        uv_indices = [
                            idx
                            for idx, st in enumerate(frag.steps)
                            if _is_full_step(st)
                            and (
                                (u_node is not None and st.node_id == u_node)
                                or (v_node is not None and st.node_id == v_node)
                            )
                        ]
                    for k in range(len(uv_indices) - 1):
                        i_a, i_b = uv_indices[k], uv_indices[k + 1]
                        if i_a + 1 < i_b:
                            sub_st = frag.steps[i_a + 1 : i_b]
                            if is_lazy_ref and isinstance(sub_st, LazyStepTuple):
                                assert outside_mask is not None
                                has_out = bool(np.any(outside_mask[np.abs(sub_st._codes)]))
                            else:
                                has_out = any(
                                    st.node_id in outside_ref_nodes for st in sub_st
                                )
                            if not has_out:
                                sub_pr = frag.provenance[i_a + 1 : i_b]
                                extra_slices.append((sub_st, sub_pr))
                                s_start, s_end = _prov_min_start_max_end(sub_pr)
                                claimed_source_intervals.add(
                                    (frag.src_path, s_start, s_end)
                                )
                                if frag.src_path != graph.ref_fragment.src_path:
                                    clean_cuts_by_frag[id(frag)].append((i_a, i_b))

                first_steps = extra_slices[0][0] if extra_slices else ()
                first_prov = extra_slices[0][1] if extra_slices else ()
                hap_slices[(reg.region_id, hap_id)] = CentromereHapSlice(
                    region_id=reg.region_id,
                    hap_id=hap_id,
                    status="ambiguous",
                    ref_status=ref_status,
                    fragment=None,
                    u_step_idx=None,
                    v_step_idx=None,
                    inner_steps=first_steps,
                    inner_provenance=first_prov,
                    src_path=".",
                    src_start=None,
                    src_end=None,
                    src_strand=None,
                    l_ref=l_ref,
                    l_path=None,
                    length_ratio=None,
                    extra_inner_slices=tuple(extra_slices),
                )
            elif len(valid_traversals) == 1:
                v_frag, u_idx, v_idx = valid_traversals[0]
                inner_steps = v_frag.steps[u_idx + 1 : v_idx]
                inner_prov = v_frag.provenance[u_idx + 1 : v_idx]
                inner_seq = spell_steps(inner_steps, graph.nodes)
                l_path = len(inner_seq)

                if inner_prov:
                    src_start, src_end = _prov_min_start_max_end(inner_prov)
                    src_strand: Literal["+", "-"] = inner_prov[0].src_strand
                    if src_start < src_end:
                        claimed_source_intervals.add(
                            (v_frag.src_path, src_start, src_end)
                        )
                else:
                    u_p = v_frag.provenance[u_idx]
                    boundary = (
                        u_p.src_end if u_p.src_strand == "+" else u_p.src_start
                    )
                    src_start = boundary
                    src_end = boundary
                    src_strand = u_p.src_strand

                if v_frag.src_path != graph.ref_fragment.src_path:
                    clean_cuts_by_frag[id(v_frag)].append((u_idx, v_idx))

                if "N" in inner_seq.upper():
                    status: Literal["complete", "gapped"] = "gapped"
                    length_ratio: float | None = None
                else:
                    status = "complete"
                    length_ratio = (
                        (l_path / l_ref)
                        if (ref_status == "complete" and l_ref > 0)
                        else None
                    )

                hap_slices[(reg.region_id, hap_id)] = CentromereHapSlice(
                    region_id=reg.region_id,
                    hap_id=hap_id,
                    status=status,
                    ref_status=ref_status,
                    fragment=v_frag,
                    u_step_idx=u_idx,
                    v_step_idx=v_idx,
                    inner_steps=inner_steps,
                    inner_provenance=inner_prov,
                    src_path=v_frag.src_path,
                    src_start=src_start,
                    src_end=src_end,
                    src_strand=src_strand,
                    l_ref=l_ref,
                    l_path=l_path,
                    length_ratio=length_ratio,
                    extra_inner_slices=(),
                )
            elif (
                len(single_side_frags) >= 1
                or total_u_hits > 0
                or total_v_hits > 0
                or any_internal_hit
            ):
                extra_slices = []
                for frag, u_hits, v_hits in single_side_frags:
                    _collect_single_side_slice(frag, u_hits, v_hits, extra_slices)

                first_steps = extra_slices[0][0] if extra_slices else ()
                first_prov = extra_slices[0][1] if extra_slices else ()
                hap_slices[(reg.region_id, hap_id)] = CentromereHapSlice(
                    region_id=reg.region_id,
                    hap_id=hap_id,
                    status="partial",
                    ref_status=ref_status,
                    fragment=None,
                    u_step_idx=None,
                    v_step_idx=None,
                    inner_steps=first_steps,
                    inner_provenance=first_prov,
                    src_path=".",
                    src_start=None,
                    src_end=None,
                    src_strand=None,
                    l_ref=l_ref,
                    l_path=None,
                    length_ratio=None,
                    extra_inner_slices=tuple(extra_slices),
                )
            else:
                hap_slices[(reg.region_id, hap_id)] = CentromereHapSlice(
                    region_id=reg.region_id,
                    hap_id=hap_id,
                    status="unmapped",
                    ref_status=ref_status,
                    fragment=None,
                    u_step_idx=None,
                    v_step_idx=None,
                    inner_steps=(),
                    inner_provenance=(),
                    src_path=".",
                    src_start=None,
                    src_end=None,
                    src_strand=None,
                    l_ref=l_ref,
                    l_path=None,
                    length_ratio=None,
                    extra_inner_slices=(),
                )

    # Step 4a: Slice alt_oriented at clean (u_idx, v_idx) cuts
    stage1_subfrags: list[PathFragment] = []
    for frag in alt_oriented:
        cuts = sorted(set(clean_cuts_by_frag.get(id(frag), [])))
        if not cuts:
            stage1_subfrags.append(frag)
            continue

        cur_start = 0
        n_steps = len(frag.steps)
        for u_i, v_i in cuts:
            if u_i >= 0 and cur_start <= u_i:
                stage1_subfrags.append(_slice_path_fragment(frag, cur_start, u_i + 1))
            cur_start = v_i
        if cur_start < n_steps:
            stage1_subfrags.append(_slice_path_fragment(frag, cur_start, n_steps))

    # Step 4b: Check remaining sub-fragments for boundary-ambiguous jumps across centromere regions
    normal_fragments: list[PathFragment] = []
    boundary_ambiguous_records: list[UnresolvedRecord] = []
    amb_counter = 1

    for sub_frag in stage1_subfrags:
        if (
            isinstance(sub_frag.steps, LazyStepTuple)
            and graph._nid_to_ref_rank is not None
            and graph._ref_cum_offsets is not None
        ):
            sf_nids = np.abs(sub_frag.steps._codes)
            sf_ranks = graph._nid_to_ref_rank[sf_nids]
            hit_pos = np.flatnonzero(sf_ranks >= 0).astype(np.int32)
            if int(hit_pos.shape[0]) < 2:
                normal_fragments.append(sub_frag)
                continue
            hit_ranks = sf_ranks[hit_pos]
            r_lo_arr = np.minimum(hit_ranks[:-1], hit_ranks[1:])
            r_hi_arr = np.maximum(hit_ranks[:-1], hit_ranks[1:])
            cum_off = graph._ref_cum_offsets
            span_s_arr = cum_off[r_lo_arr]
            # Wait: graph.ref_step_ranges[r_lo][1] is cum_off[r_lo + 1]!
            span_s_arr = cum_off[r_lo_arr + 1]
            span_e_arr = cum_off[r_hi_arr]
            crossed_mask = np.zeros(int(r_lo_arr.shape[0]), dtype=bool)
            non_eq = r_lo_arr < r_hi_arr
            for reg in regions:
                cond_interval = (span_s_arr < span_e_arr) & (
                    np.maximum(span_s_arr, reg.ref_start)
                    < np.minimum(span_e_arr, reg.ref_end)
                )
                cond_point = (span_s_arr >= span_e_arr) & (
                    (reg.ref_start < span_s_arr) & (span_s_arr < reg.ref_end)
                )
                crossed_mask |= non_eq & (cond_interval | cond_point)
            cand_k_list = np.flatnonzero(crossed_mask).tolist()
            anc_positions_lazy: tuple[np.ndarray, np.ndarray] | None = (
                hit_pos,
                hit_ranks,
            )
            anc_positions: list[tuple[int, int]] = []
        else:
            anc_positions_lazy = None
            anc_positions = [
                (idx, graph.unique_ref_step_by_node[st.node_id])
                for idx, st in enumerate(sub_frag.steps)
                if _is_full_step(st) and st.node_id in graph.unique_ref_step_by_node
            ]
            if len(anc_positions) < 2:
                normal_fragments.append(sub_frag)
                continue
            cand_k_list = list(range(len(anc_positions) - 1))

        amb_cuts: list[tuple[int, int]] = []
        for k in cand_k_list:
            if anc_positions_lazy is not None:
                idx_a = int(anc_positions_lazy[0][k])
                r_a = int(anc_positions_lazy[1][k])
                idx_b = int(anc_positions_lazy[0][k + 1])
                r_b = int(anc_positions_lazy[1][k + 1])
            else:
                idx_a, r_a = anc_positions[k]
                idx_b, r_b = anc_positions[k + 1]
            r_lo, r_hi = min(r_a, r_b), max(r_a, r_b)
            if r_lo == r_hi:
                continue
            span_start = graph.ref_step_ranges[r_lo][1]
            span_end = graph.ref_step_ranges[r_hi][0]

            crossed_reg: CentromereRegion | None = None
            for reg in regions:
                if span_start < span_end:
                    if max(span_start, reg.ref_start) < min(span_end, reg.ref_end):
                        crossed_reg = reg
                        break
                else:
                    if reg.ref_start < span_start < reg.ref_end:
                        crossed_reg = reg
                        break

            if crossed_reg is not None:
                amb_cuts.append((idx_a, idx_b))
                runs = _extract_nonref_runs(
                    steps=sub_frag.steps[idx_a + 1 : idx_b],
                    provenance=sub_frag.provenance[idx_a + 1 : idx_b],
                    nodes=graph.nodes,
                    global_ref_nodes=graph.global_ref_nodes,
                )
                for r in runs:
                    if len(r.seq) >= min_sv_len:
                        iv_key = (r.src_path, r.src_start, r.src_end)
                        if iv_key not in claimed_source_intervals:
                            claimed_source_intervals.add(iv_key)
                            rec_id = f"{crossed_reg.region_id}_UNRES_AMB_{amb_counter}"
                            amb_counter += 1
                            boundary_ambiguous_records.append(
                                UnresolvedRecord(
                                    record_id=rec_id,
                                    region_id=crossed_reg.region_id,
                                    event_id=".",
                                    src_path=r.src_path,
                                    src_start=r.src_start,
                                    src_end=r.src_end,
                                    src_strand=r.src_strand,
                                    length=r.src_end - r.src_start,
                                    reason="CEN_BOUNDARY_AMBIGUOUS",
                                    seq=r.seq,
                                    export_to_fasta=True,
                                    exported_fasta_id=rec_id,
                                )
                            )

        if not amb_cuts:
            normal_fragments.append(sub_frag)
        else:
            cur_s = 0
            for idx_a, idx_b in amb_cuts:
                if cur_s <= idx_a:
                    normal_fragments.append(
                        _slice_path_fragment(sub_frag, cur_s, idx_a + 1)
                    )
                cur_s = idx_b
            if cur_s < len(sub_frag.steps):
                normal_fragments.append(
                    _slice_path_fragment(sub_frag, cur_s, len(sub_frag.steps))
                )

    return CentromerePartitionResult(
        regions=tuple(regions),
        hap_slices=hap_slices,
        normal_fragments=tuple(normal_fragments),
        claimed_source_intervals=claimed_source_intervals,
        boundary_ambiguous_records=tuple(boundary_ambiguous_records),
    )


@dataclass
class _AcceptedIslandCandidate:
    candidate_id: str
    parent_id: str | None
    hap_id: str
    steps: tuple[Step, ...]
    prov: StepProvenance
    seq: str
    revisited_fraction: float
    masked_intervals: list[tuple[int, int]] | None
    window_singleton_ratios: list[float | None] | None


@dataclass
class _IslandCluster:
    rep_seq: str
    exemplar: _AcceptedIslandCandidate
    member_haps: set[str]
    member_candidates: list[_AcceptedIslandCandidate]
    aligner: Any | None = None


def process_centromere_regions(
    partition: CentromerePartitionResult,
    graph: GFAGraph,
    config: BuildConfig,
) -> CentromereOutput:
    """
    §3.8.3–§3.8.5 & §5.1–§5.5 (F23, F24, C2–C16, C18):
    Process all centromere regions in `partition` under `bypass` or `extract-islands` mode.
    """
    if not partition.regions:
        return CentromereOutput(
            centromere_rows=(),
            island_slot_blocks=(),
            island_sv_events=(),
            unresolved_records=partition.boundary_ambiguous_records,
            json_log_entries=(),
            summary_metrics={},
        )

    centromere_rows: list[CentromereRow] = []
    island_slot_blocks: list[InsertionSlotBlock] = []
    island_sv_events: list[SVEvent] = []
    unresolved_records: list[UnresolvedRecord] = list(
        partition.boundary_ambiguous_records
    )
    json_log_entries: list[dict[str, Any]] = []

    seen_exported_intervals: dict[tuple[str, int, int, str], str] = {}
    for rec in unresolved_records:
        if rec.export_to_fasta and rec.exported_fasta_id != ".":
            seen_exported_intervals[
                (rec.src_path, rec.src_start, rec.src_end, rec.src_strand)
            ] = rec.exported_fasta_id

    unres_counter = len(unresolved_records) + 1
    global_island_idx = 1

    def _emit_unresolved(
        region_id: str,
        src_path: str,
        src_start: int,
        src_end: int,
        src_strand: Literal["+", "-"],
        seq: str,
        reason: str,
        export_to_fasta: bool,
    ) -> UnresolvedRecord:
        nonlocal unres_counter
        rec_id = f"{region_id}_UNRES_{unres_counter}"
        unres_counter += 1
        if export_to_fasta:
            key = (src_path, src_start, src_end, src_strand)
            exp_id = seen_exported_intervals.setdefault(key, rec_id)
        else:
            exp_id = "."
        rec = UnresolvedRecord(
            record_id=rec_id,
            region_id=region_id,
            event_id=".",
            src_path=src_path,
            src_start=src_start,
            src_end=src_end,
            src_strand=src_strand,
            length=src_end - src_start,
            reason=reason,
            seq=seq,
            export_to_fasta=export_to_fasta,
            exported_fasta_id=exp_id,
        )
        unresolved_records.append(rec)
        return rec

    def _emit_json_log(
        region_id: str,
        hap_id: str,
        candidate_id: str,
        parent_id: str | None,
        src_path: str,
        src_start: int,
        src_end: int,
        src_strand: str,
        length: int,
        revisited_fraction: float | None,
        masked_intervals: Sequence[tuple[int, int]] | None,
        window_singleton_ratios: list[float | None] | None,
        final_reason: str,
        representative_event_id: str | None,
    ) -> None:
        json_log_entries.append(
            {
                "region_id": region_id,
                "hap": hap_id,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "src_path": src_path,
                "src_start": src_start,
                "src_end": src_end,
                "src_strand": src_strand,
                "length": length,
                "revisited_fraction": revisited_fraction,
                "masked_intervals": (
                    [[int(s), int(e)] for s, e in masked_intervals]
                    if masked_intervals is not None
                    else None
                ),
                "window_singleton_ratios": window_singleton_ratios,
                "final_reason": final_reason,
                "representative_event_id": representative_event_id,
            }
        )

    for reg in partition.regions:
        input_bed_ids_str = ",".join(reg.input_bed_ids)

        if reg.effective_mode == "bypass":
            for hap_id in graph.all_haplotypes:
                hslice = partition.hap_slices[(reg.region_id, hap_id)]
                if hslice.status in ("complete", "gapped"):
                    runs = _extract_nonref_runs(
                        steps=hslice.inner_steps,
                        provenance=hslice.inner_provenance,
                        nodes=graph.nodes,
                        global_ref_nodes=graph.global_ref_nodes,
                    )
                    nonref_input_bp = sum(len(r.seq) for r in runs)
                    for cand_idx, r in enumerate(runs, start=1):
                        cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_idx}"
                        if len(r.seq) >= config.min_sv_len:
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=r.src_path,
                                src_start=r.src_start,
                                src_end=r.src_end,
                                src_strand=r.src_strand,
                                seq=r.seq,
                                reason="CEN_BYPASS",
                                export_to_fasta=config.dump_repeat_sequences,
                            )
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=cand_id,
                            parent_id=None,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            length=len(r.seq),
                            revisited_fraction=None,
                            masked_intervals=None,
                            window_singleton_ratios=None,
                            final_reason="CEN_BYPASS",
                            representative_event_id=None,
                        )

                    ratio_str = (
                        f"{hslice.length_ratio:.6f}"
                        if hslice.length_ratio is not None
                        else "."
                    )
                    centromere_rows.append(
                        CentromereRow(
                            chrom=reg.chrom,
                            region_id=reg.region_id,
                            input_bed_ids=input_bed_ids_str,
                            bed_start=reg.bed_start,
                            bed_end=reg.bed_end,
                            ref_start=reg.ref_start,
                            ref_end=reg.ref_end,
                            anchor_u=reg.anchor_u_name,
                            anchor_v=reg.anchor_v_name,
                            mode=reg.effective_mode,
                            hap=hap_id,
                            status=hslice.status,
                            ref_status=hslice.ref_status,
                            src_path=hslice.src_path,
                            src_start=(
                                hslice.src_start if hslice.src_start is not None else "."
                            ),
                            src_end=(
                                hslice.src_end if hslice.src_end is not None else "."
                            ),
                            src_strand=hslice.src_strand or ".",
                            L_ref=hslice.l_ref,
                            L_path=hslice.l_path if hslice.l_path is not None else ".",
                            length_ratio=ratio_str,
                            nonref_input_bp=nonref_input_bp,
                            reference_like_bp=0,
                            repeat_like_bp=0,
                            short_bp=0,
                            unresolved_bp=nonref_input_bp,
                            accepted_source_bp=0,
                            supported_events=".",
                        )
                    )
                else:
                    cand_counter = 1
                    for sl_steps, sl_prov in hslice.extra_inner_slices:
                        runs = _extract_nonref_runs(
                            steps=sl_steps,
                            provenance=sl_prov,
                            nodes=graph.nodes,
                            global_ref_nodes=graph.global_ref_nodes,
                        )
                        for r in runs:
                            cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_counter}"
                            cand_counter += 1
                            if len(r.seq) >= config.min_sv_len:
                                _emit_unresolved(
                                    region_id=reg.region_id,
                                    src_path=r.src_path,
                                    src_start=r.src_start,
                                    src_end=r.src_end,
                                    src_strand=r.src_strand,
                                    seq=r.seq,
                                    reason="CEN_BYPASS",
                                    export_to_fasta=config.dump_repeat_sequences,
                                )
                                _emit_json_log(
                                    region_id=reg.region_id,
                                    hap_id=hap_id,
                                    candidate_id=cand_id,
                                    parent_id=None,
                                    src_path=r.src_path,
                                    src_start=r.src_start,
                                    src_end=r.src_end,
                                    src_strand=r.src_strand,
                                    length=len(r.seq),
                                    revisited_fraction=None,
                                    masked_intervals=None,
                                    window_singleton_ratios=None,
                                    final_reason="CEN_BYPASS",
                                    representative_event_id=None,
                                )

                    centromere_rows.append(
                        CentromereRow(
                            chrom=reg.chrom,
                            region_id=reg.region_id,
                            input_bed_ids=input_bed_ids_str,
                            bed_start=reg.bed_start,
                            bed_end=reg.bed_end,
                            ref_start=reg.ref_start,
                            ref_end=reg.ref_end,
                            anchor_u=reg.anchor_u_name,
                            anchor_v=reg.anchor_v_name,
                            mode=reg.effective_mode,
                            hap=hap_id,
                            status=hslice.status,
                            ref_status=hslice.ref_status,
                            src_path=".",
                            src_start=".",
                            src_end=".",
                            src_strand=".",
                            L_ref=hslice.l_ref,
                            L_path=".",
                            length_ratio=".",
                            nonref_input_bp=".",
                            reference_like_bp=".",
                            repeat_like_bp=".",
                            short_bp=".",
                            unresolved_bp=".",
                            accepted_source_bp=".",
                            supported_events=".",
                        )
                    )
            continue

        # Case II: reg.effective_mode == "extract-islands"
        ref_region_seq = graph.ref_seq[reg.ref_start : reg.ref_end].upper()
        region_aligner = (
            mappy.Aligner(seq=ref_region_seq, preset="asm20")
            if ref_region_seq and "N" not in ref_region_seq
            else None
        )

        hap_accounting: dict[str, dict[str, int | str]] = {}
        accepted_candidates: list[_AcceptedIslandCandidate] = []

        for hap_id in graph.all_haplotypes:
            hslice = partition.hap_slices[(reg.region_id, hap_id)]

            if hslice.status in ("partial", "ambiguous", "unmapped"):
                unres_reason = (
                    "CEN_AMBIGUOUS"
                    if hslice.status == "ambiguous"
                    else "CEN_INCOMPLETE"
                )
                cand_counter = 1
                for sl_steps, sl_prov in hslice.extra_inner_slices:
                    runs = _extract_nonref_runs(
                        steps=sl_steps,
                        provenance=sl_prov,
                        nodes=graph.nodes,
                        global_ref_nodes=graph.global_ref_nodes,
                    )
                    for r in runs:
                        cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_counter}"
                        cand_counter += 1
                        if len(r.seq) >= config.min_sv_len:
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=r.src_path,
                                src_start=r.src_start,
                                src_end=r.src_end,
                                src_strand=r.src_strand,
                                seq=r.seq,
                                reason=unres_reason,
                                export_to_fasta=True,
                            )
                            _emit_json_log(
                                region_id=reg.region_id,
                                hap_id=hap_id,
                                candidate_id=cand_id,
                                parent_id=None,
                                src_path=r.src_path,
                                src_start=r.src_start,
                                src_end=r.src_end,
                                src_strand=r.src_strand,
                                length=len(r.seq),
                                revisited_fraction=None,
                                masked_intervals=None,
                                window_singleton_ratios=None,
                                final_reason=unres_reason,
                                representative_event_id=None,
                            )

                hap_accounting[hap_id] = {
                    "nonref_input_bp": ".",
                    "reference_like_bp": ".",
                    "repeat_like_bp": ".",
                    "short_bp": ".",
                    "unresolved_bp": ".",
                    "accepted_source_bp": ".",
                }
                continue

            if hslice.status == "gapped":
                runs = _extract_nonref_runs(
                    steps=hslice.inner_steps,
                    provenance=hslice.inner_provenance,
                    nodes=graph.nodes,
                    global_ref_nodes=graph.global_ref_nodes,
                )
                nonref_input_bp = sum(len(r.seq) for r in runs)
                for cand_idx, r in enumerate(runs, start=1):
                    cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_idx}"
                    if len(r.seq) >= config.min_sv_len:
                        _emit_unresolved(
                            region_id=reg.region_id,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            seq=r.seq,
                            reason="CEN_INCOMPLETE",
                            export_to_fasta=True,
                        )
                    _emit_json_log(
                        region_id=reg.region_id,
                        hap_id=hap_id,
                        candidate_id=cand_id,
                        parent_id=None,
                        src_path=r.src_path,
                        src_start=r.src_start,
                        src_end=r.src_end,
                        src_strand=r.src_strand,
                        length=len(r.seq),
                        revisited_fraction=None,
                        masked_intervals=None,
                        window_singleton_ratios=None,
                        final_reason="CEN_INCOMPLETE",
                        representative_event_id=None,
                    )

                hap_accounting[hap_id] = {
                    "nonref_input_bp": nonref_input_bp,
                    "reference_like_bp": 0,
                    "repeat_like_bp": 0,
                    "short_bp": 0,
                    "unresolved_bp": nonref_input_bp,
                    "accepted_source_bp": 0,
                }
                continue

            # hslice.status == "complete"
            node_visit_counts = collections.Counter(
                st.node_id for st in hslice.inner_steps
            )
            runs = _extract_nonref_runs(
                steps=hslice.inner_steps,
                provenance=hslice.inner_provenance,
                nodes=graph.nodes,
                global_ref_nodes=graph.global_ref_nodes,
            )

            nonref_input_bp = 0
            reference_like_bp = 0
            repeat_like_bp = 0
            short_bp = 0
            unresolved_bp = 0
            accepted_source_bp = 0

            if hslice.ref_status == "gapped":
                for cand_idx, r in enumerate(runs, start=1):
                    cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_idx}"
                    c_len = len(r.seq)
                    nonref_input_bp += c_len
                    rev_frac = (
                        sum(
                            st.step_len(graph.nodes)
                            for st in r.steps
                            if node_visit_counts[st.node_id] > 1
                        )
                        / c_len
                    )
                    if c_len < config.centromere_min_island:
                        short_bp += c_len
                        if c_len >= config.min_sv_len:
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=r.src_path,
                                src_start=r.src_start,
                                src_end=r.src_end,
                                src_strand=r.src_strand,
                                seq=r.seq,
                                reason="CEN_SHORT_RESIDUAL",
                                export_to_fasta=True,
                            )
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=cand_id,
                            parent_id=None,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            length=c_len,
                            revisited_fraction=rev_frac,
                            masked_intervals=None,
                            window_singleton_ratios=None,
                            final_reason="CEN_SHORT_RESIDUAL",
                            representative_event_id=None,
                        )
                    else:
                        unresolved_bp += c_len
                        _emit_unresolved(
                            region_id=reg.region_id,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            seq=r.seq,
                            reason="CEN_REFERENCE_GAP",
                            export_to_fasta=True,
                        )
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=cand_id,
                            parent_id=None,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            length=c_len,
                            revisited_fraction=rev_frac,
                            masked_intervals=None,
                            window_singleton_ratios=None,
                            final_reason="CEN_REFERENCE_GAP",
                            representative_event_id=None,
                        )

                hap_accounting[hap_id] = {
                    "nonref_input_bp": nonref_input_bp,
                    "reference_like_bp": reference_like_bp,
                    "repeat_like_bp": repeat_like_bp,
                    "short_bp": short_bp,
                    "unresolved_bp": unresolved_bp,
                    "accepted_source_bp": accepted_source_bp,
                }
                continue

            # Full Steps A–C for status == "complete" and ref_status == "complete"
            for cand_idx, r in enumerate(runs, start=1):
                cand_id = f"{reg.region_id}_{hap_id}_cand_{cand_idx}"
                cand_len = len(r.seq)
                nonref_input_bp += cand_len
                rev_frac = (
                    sum(
                        st.step_len(graph.nodes)
                        for st in r.steps
                        if node_visit_counts[st.node_id] > 1
                    )
                    / cand_len
                )

                # Step A length check
                if cand_len < config.centromere_min_island:
                    short_bp += cand_len
                    if cand_len >= config.min_sv_len:
                        _emit_unresolved(
                            region_id=reg.region_id,
                            src_path=r.src_path,
                            src_start=r.src_start,
                            src_end=r.src_end,
                            src_strand=r.src_strand,
                            seq=r.seq,
                            reason="CEN_SHORT_RESIDUAL",
                            export_to_fasta=True,
                        )
                    _emit_json_log(
                        region_id=reg.region_id,
                        hap_id=hap_id,
                        candidate_id=cand_id,
                        parent_id=None,
                        src_path=r.src_path,
                        src_start=r.src_start,
                        src_end=r.src_end,
                        src_strand=r.src_strand,
                        length=cand_len,
                        revisited_fraction=rev_frac,
                        masked_intervals=None,
                        window_singleton_ratios=None,
                        final_reason="CEN_SHORT_RESIDUAL",
                        representative_event_id=None,
                    )
                    continue

                # Step B: Reference interval subtraction (mappy 50bp block grid)
                b_masked, b_unmasked, b_ref_bp = subtract_reference_intervals_mappy(
                    query_seq=r.seq,
                    ref_region_seq=ref_region_seq,
                    min_sv_len=config.min_sv_len,
                    aligner=region_aligner,
                )
                reference_like_bp += b_ref_bp

                if not b_unmasked:
                    _emit_json_log(
                        region_id=reg.region_id,
                        hap_id=hap_id,
                        candidate_id=cand_id,
                        parent_id=None,
                        src_path=r.src_path,
                        src_start=r.src_start,
                        src_end=r.src_end,
                        src_strand=r.src_strand,
                        length=cand_len,
                        revisited_fraction=rev_frac,
                        masked_intervals=b_masked,
                        window_singleton_ratios=None,
                        final_reason="CEN_REFERENCE_LIKE",
                        representative_event_id=None,
                    )
                    continue

                if b_masked:
                    _emit_json_log(
                        region_id=reg.region_id,
                        hap_id=hap_id,
                        candidate_id=cand_id,
                        parent_id=None,
                        src_path=r.src_path,
                        src_start=r.src_start,
                        src_end=r.src_end,
                        src_strand=r.src_strand,
                        length=cand_len,
                        revisited_fraction=rev_frac,
                        masked_intervals=b_masked,
                        window_singleton_ratios=None,
                        final_reason="SPLIT_BY_REFERENCE_SUBTRACTION",
                        representative_event_id=None,
                    )

                b_kept_slices, _ = slice_candidate_by_intervals(
                    steps=r.steps,
                    provenance=r.prov,
                    nodes=graph.nodes,
                    keep_intervals=b_unmasked,
                    mask_intervals=b_masked,
                )

                for b_idx, (_, _, b_steps, b_prov, b_seq) in enumerate(
                    b_kept_slices, start=1
                ):
                    b_len = len(b_seq)
                    b_id = cand_id if not b_masked else f"{cand_id}_b{b_idx}"
                    b_parent = None if not b_masked else cand_id
                    b_contig_prov = merge_contiguous_provenance(b_prov, b_seq)[0][0]
                    b_rev_frac = (
                        sum(
                            st.step_len(graph.nodes)
                            for st in b_steps
                            if node_visit_counts[st.node_id] > 1
                        )
                        / b_len
                    )

                    if b_len < config.centromere_min_island:
                        short_bp += b_len
                        if b_len >= config.min_sv_len:
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=b_contig_prov.src_path,
                                src_start=b_contig_prov.src_start,
                                src_end=b_contig_prov.src_end,
                                src_strand=b_contig_prov.src_strand,
                                seq=b_seq,
                                reason="CEN_SHORT_RESIDUAL",
                                export_to_fasta=True,
                            )
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=b_id,
                            parent_id=b_parent,
                            src_path=b_contig_prov.src_path,
                            src_start=b_contig_prov.src_start,
                            src_end=b_contig_prov.src_end,
                            src_strand=b_contig_prov.src_strand,
                            length=b_len,
                            revisited_fraction=b_rev_frac,
                            masked_intervals=None,
                            window_singleton_ratios=None,
                            final_reason="CEN_SHORT_RESIDUAL",
                            representative_event_id=None,
                        )
                        continue

                    # Step C: Local repeat masking (always executed in extract-islands)
                    c_masked, c_diags = repeat_mask_intervals(b_seq)
                    c_win_ratios: list[float | None] | None = (
                        [d["singleton_ratio"] for d in c_diags] if c_diags else None
                    )
                    c_unmasked = complement_intervals(b_len, c_masked)
                    c_kept_slices, c_masked_slices = slice_candidate_by_intervals(
                        steps=b_steps,
                        provenance=b_prov,
                        nodes=graph.nodes,
                        keep_intervals=c_unmasked,
                        mask_intervals=c_masked,
                    )

                    for _, _, m_prov, m_seq in c_masked_slices:
                        m_len = len(m_seq)
                        repeat_like_bp += m_len
                        if m_len >= config.min_sv_len:
                            m_contig_prov = merge_contiguous_provenance(m_prov, m_seq)[
                                0
                            ][0]
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=m_contig_prov.src_path,
                                src_start=m_contig_prov.src_start,
                                src_end=m_contig_prov.src_end,
                                src_strand=m_contig_prov.src_strand,
                                seq=m_seq,
                                reason="CEN_REPEAT_LIKE",
                                export_to_fasta=config.dump_repeat_sequences,
                            )

                    if not c_kept_slices:
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=b_id,
                            parent_id=b_parent,
                            src_path=b_contig_prov.src_path,
                            src_start=b_contig_prov.src_start,
                            src_end=b_contig_prov.src_end,
                            src_strand=b_contig_prov.src_strand,
                            length=b_len,
                            revisited_fraction=b_rev_frac,
                            masked_intervals=c_masked,
                            window_singleton_ratios=c_win_ratios,
                            final_reason="CEN_REPEAT_LIKE",
                            representative_event_id=None,
                        )
                        continue

                    if c_masked:
                        _emit_json_log(
                            region_id=reg.region_id,
                            hap_id=hap_id,
                            candidate_id=b_id,
                            parent_id=b_parent,
                            src_path=b_contig_prov.src_path,
                            src_start=b_contig_prov.src_start,
                            src_end=b_contig_prov.src_end,
                            src_strand=b_contig_prov.src_strand,
                            length=b_len,
                            revisited_fraction=b_rev_frac,
                            masked_intervals=c_masked,
                            window_singleton_ratios=c_win_ratios,
                            final_reason="SPLIT_BY_REPEAT_MASK",
                            representative_event_id=None,
                        )

                    for c_idx, (_, _, c_steps, c_prov, c_seq) in enumerate(
                        c_kept_slices, start=1
                    ):
                        c_len = len(c_seq)
                        c_id = b_id if not c_masked else f"{b_id}_c{c_idx}"
                        # Child residuals always reference raw_block_candidate_id (cand_id)
                        c_parent = b_parent if not c_masked else cand_id
                        c_contig_prov = merge_contiguous_provenance(c_prov, c_seq)[0][0]
                        c_rev_frac = (
                            sum(
                                st.step_len(graph.nodes)
                                for st in c_steps
                                if node_visit_counts[st.node_id] > 1
                            )
                            / c_len
                        )

                        if c_len < config.centromere_min_island:
                            short_bp += c_len
                            if c_len >= config.min_sv_len:
                                _emit_unresolved(
                                    region_id=reg.region_id,
                                    src_path=c_contig_prov.src_path,
                                    src_start=c_contig_prov.src_start,
                                    src_end=c_contig_prov.src_end,
                                    src_strand=c_contig_prov.src_strand,
                                    seq=c_seq,
                                    reason="CEN_SHORT_RESIDUAL",
                                    export_to_fasta=True,
                                )
                            _emit_json_log(
                                region_id=reg.region_id,
                                hap_id=hap_id,
                                candidate_id=c_id,
                                parent_id=c_parent,
                                src_path=c_contig_prov.src_path,
                                src_start=c_contig_prov.src_start,
                                src_end=c_contig_prov.src_end,
                                src_strand=c_contig_prov.src_strand,
                                length=c_len,
                                revisited_fraction=c_rev_frac,
                                masked_intervals=[],
                                window_singleton_ratios=None,
                                final_reason="CEN_SHORT_RESIDUAL",
                                representative_event_id=None,
                            )
                            continue

                        if any(ch not in "ACGTacgt" for ch in c_seq):
                            unresolved_bp += c_len
                            _emit_unresolved(
                                region_id=reg.region_id,
                                src_path=c_contig_prov.src_path,
                                src_start=c_contig_prov.src_start,
                                src_end=c_contig_prov.src_end,
                                src_strand=c_contig_prov.src_strand,
                                seq=c_seq,
                                reason="CEN_SEQUENCE_UNCERTAIN",
                                export_to_fasta=True,
                            )
                            _emit_json_log(
                                region_id=reg.region_id,
                                hap_id=hap_id,
                                candidate_id=c_id,
                                parent_id=c_parent,
                                src_path=c_contig_prov.src_path,
                                src_start=c_contig_prov.src_start,
                                src_end=c_contig_prov.src_end,
                                src_strand=c_contig_prov.src_strand,
                                length=c_len,
                                revisited_fraction=c_rev_frac,
                                masked_intervals=None,
                                window_singleton_ratios=None,
                                final_reason="CEN_SEQUENCE_UNCERTAIN",
                                representative_event_id=None,
                            )
                            continue

                        # Accepted candidate island
                        accepted_source_bp += c_len
                        if not c_masked:
                            sub_ratios = c_win_ratios
                        else:
                            _, sub_diags = repeat_mask_intervals(c_seq)
                            sub_ratios = (
                                [d["singleton_ratio"] for d in sub_diags]
                                if sub_diags
                                else None
                            )
                        accepted_candidates.append(
                            _AcceptedIslandCandidate(
                                candidate_id=c_id,
                                parent_id=c_parent,
                                hap_id=hap_id,
                                steps=c_steps,
                                prov=c_contig_prov,
                                seq=c_seq.upper(),
                                revisited_fraction=c_rev_frac,
                                masked_intervals=[],
                                window_singleton_ratios=sub_ratios,
                            )
                        )

            total_accounted = (
                reference_like_bp
                + repeat_like_bp
                + short_bp
                + unresolved_bp
                + accepted_source_bp
            )
            if total_accounted != nonref_input_bp:
                raise InvariantViolationError(
                    f"5-way centromere bp accounting mismatch for ({reg.region_id}, {hap_id}): "
                    f"{total_accounted} != {nonref_input_bp}"
                )

            hap_accounting[hap_id] = {
                "nonref_input_bp": nonref_input_bp,
                "reference_like_bp": reference_like_bp,
                "repeat_like_bp": repeat_like_bp,
                "short_bp": short_bp,
                "unresolved_bp": unresolved_bp,
                "accepted_source_bp": accepted_source_bp,
            }

        # Step D: Exact deduplication, 4-criterion clustering, and mounting at ref_start(v)
        hap_supported_events: dict[str, list[str]] = collections.defaultdict(list)
        if accepted_candidates and reg.anchor_v_step is not None and reg.anchor_u_step is not None:
            # Group by exact sequence
            seq_groups: dict[str, list[_AcceptedIslandCandidate]] = (
                collections.defaultdict(list)
            )
            for cand in accepted_candidates:
                seq_groups[cand.seq].append(cand)

            unique_groups: list[tuple[str, _AcceptedIslandCandidate, list[_AcceptedIslandCandidate]]] = []
            for seq_str, cands in seq_groups.items():
                exemplar = min(
                    cands,
                    key=lambda c: (
                        c.prov.src_path,
                        c.prov.src_start,
                        c.prov.src_strand,
                        c.candidate_id,
                    ),
                )
                unique_groups.append((seq_str, exemplar, cands))

            # Sort unique sequence groups by length descending, then sequence, then exemplar source
            unique_groups.sort(
                key=lambda item: (
                    -len(item[0]),
                    item[0],
                    item[1].prov.src_path,
                    item[1].prov.src_start,
                    item[1].prov.src_strand,
                )
            )

            clusters: list[_IslandCluster] = []
            for seq_str, exemplar, cands in unique_groups:
                merged_into: _IslandCluster | None = None
                for cl in clusters:
                    if len(seq_str) > 1000 or len(cl.rep_seq) > 1000:
                        if cl.aligner is None:
                            cl.aligner = mappy.Aligner(
                                seq=cl.rep_seq, preset="asm20"
                            )
                    if check_similarity_merge(
                        seq_s=seq_str,
                        seq_r=cl.rep_seq,
                        min_sv_len=config.min_sv_len,
                        cluster_id=config.cluster_id,
                        cluster_win_id=config.cluster_win_id,
                        aligner=cl.aligner,
                    ):
                        merged_into = cl
                        break

                if merged_into is not None:
                    for c in cands:
                        merged_into.member_haps.add(c.hap_id)
                        merged_into.member_candidates.append(c)
                else:
                    clusters.append(
                        _IslandCluster(
                            rep_seq=seq_str,
                            exemplar=exemplar,
                            member_haps={c.hap_id for c in cands},
                            member_candidates=list(cands),
                        )
                    )

            # Order clusters by support haplotype count desc, length desc, sequence, exemplar source
            clusters.sort(
                key=lambda cl: (
                    -len(cl.member_haps),
                    -len(cl.rep_seq),
                    cl.rep_seq,
                    cl.exemplar.prov.src_path,
                    cl.exemplar.prov.src_start,
                    cl.exemplar.prov.src_strand,
                )
            )

            region_representatives: list[tuple[Piece, ...]] = []
            first_ev_id = ""
            for cl in clusters:
                ev_id = f"{graph.ref_chrom}_CEN_ISLAND_{global_island_idx}"
                global_island_idx += 1
                if not first_ev_id:
                    first_ev_id = ev_id

                members = tuple(sorted(cl.member_haps))
                for m_hap in members:
                    hap_supported_events[m_hap].append(ev_id)

                ex = cl.exemplar
                piece = Piece(
                    src_hap=ex.hap_id,
                    src_path=ex.prov.src_path,
                    src_start=ex.prov.src_start,
                    src_end=ex.prov.src_end,
                    src_strand=ex.prov.src_strand,
                    seq=cl.rep_seq,
                    join_before="spacer",
                    members=members,
                    allele_idx=1,
                    piece_idx=1,
                    alleles=(1,),
                )
                region_representatives.append((piece,))

                genotypes: dict[str, str] = {}
                for h_id in graph.all_haplotypes:
                    h_sl = partition.hap_slices[(reg.region_id, h_id)]
                    if h_id in cl.member_haps:
                        genotypes[h_id] = "1"
                    elif h_sl.status == "complete" and h_sl.ref_status == "complete":
                        genotypes[h_id] = "0"
                    else:
                        genotypes[h_id] = "."

                n_called = sum(1 for g in genotypes.values() if g != ".")
                island_sv_events.append(
                    SVEvent(
                        chrom=graph.ref_chrom,
                        pan_start=0,
                        pan_end=len(cl.rep_seq),
                        event_id=ev_id,
                        event_type="CEN_ISLAND",
                        ref_start=reg.ref_end,
                        ref_end=reg.ref_end,
                        anchors=f"{reg.anchor_u_name}..{reg.anchor_v_name}",
                        alleles_str=(
                            f"1:len={len(cl.rep_seq)},src={ex.hap_id},n={len(members)}"
                        ),
                        n_carrier=len(members),
                        n_called=n_called,
                        region_id=reg.region_id,
                        genotypes=genotypes,
                        pieces=(piece,),
                    )
                )

                for cand in cl.member_candidates:
                    _emit_json_log(
                        region_id=reg.region_id,
                        hap_id=cand.hap_id,
                        candidate_id=cand.candidate_id,
                        parent_id=cand.parent_id,
                        src_path=cand.prov.src_path,
                        src_start=cand.prov.src_start,
                        src_end=cand.prov.src_end,
                        src_strand=cand.prov.src_strand,
                        length=len(cand.seq),
                        revisited_fraction=cand.revisited_fraction,
                        masked_intervals=cand.masked_intervals,
                        window_singleton_ratios=cand.window_singleton_ratios,
                        final_reason="ACCEPTED_ISLAND",
                        representative_event_id=ev_id,
                    )

            if region_representatives:
                island_slot_blocks.append(
                    InsertionSlotBlock(
                        right_anchor_ref_step=reg.anchor_v_step,
                        is_centromere_island=True,
                        group_span=reg.anchor_v_step - reg.anchor_u_step,
                        left_anchor_ref_step=reg.anchor_u_step,
                        event_id=first_ev_id,
                        representatives=tuple(region_representatives),
                        trailing_spacer=True,
                    )
                )

        # Emit CentromereRow for each haplotype in graph.all_haplotypes
        for hap_id in graph.all_haplotypes:
            hslice = partition.hap_slices[(reg.region_id, hap_id)]
            acc = hap_accounting[hap_id]

            if hslice.status == "complete" and hslice.ref_status == "complete":
                supp_str = ",".join(hap_supported_events.get(hap_id, []))
            else:
                supp_str = "."

            if hslice.status in ("complete", "gapped"):
                ratio_str = (
                    f"{hslice.length_ratio:.6f}"
                    if hslice.length_ratio is not None
                    else "."
                )
                centromere_rows.append(
                    CentromereRow(
                        chrom=reg.chrom,
                        region_id=reg.region_id,
                        input_bed_ids=input_bed_ids_str,
                        bed_start=reg.bed_start,
                        bed_end=reg.bed_end,
                        ref_start=reg.ref_start,
                        ref_end=reg.ref_end,
                        anchor_u=reg.anchor_u_name,
                        anchor_v=reg.anchor_v_name,
                        mode=reg.effective_mode,
                        hap=hap_id,
                        status=hslice.status,
                        ref_status=hslice.ref_status,
                        src_path=hslice.src_path,
                        src_start=(
                            hslice.src_start if hslice.src_start is not None else "."
                        ),
                        src_end=hslice.src_end if hslice.src_end is not None else ".",
                        src_strand=hslice.src_strand or ".",
                        L_ref=hslice.l_ref,
                        L_path=hslice.l_path if hslice.l_path is not None else ".",
                        length_ratio=ratio_str,
                        nonref_input_bp=acc["nonref_input_bp"],
                        reference_like_bp=acc["reference_like_bp"],
                        repeat_like_bp=acc["repeat_like_bp"],
                        short_bp=acc["short_bp"],
                        unresolved_bp=acc["unresolved_bp"],
                        accepted_source_bp=acc["accepted_source_bp"],
                        supported_events=supp_str,
                    )
                )
            else:
                centromere_rows.append(
                    CentromereRow(
                        chrom=reg.chrom,
                        region_id=reg.region_id,
                        input_bed_ids=input_bed_ids_str,
                        bed_start=reg.bed_start,
                        bed_end=reg.bed_end,
                        ref_start=reg.ref_start,
                        ref_end=reg.ref_end,
                        anchor_u=reg.anchor_u_name,
                        anchor_v=reg.anchor_v_name,
                        mode=reg.effective_mode,
                        hap=hap_id,
                        status=hslice.status,
                        ref_status=hslice.ref_status,
                        src_path=".",
                        src_start=".",
                        src_end=".",
                        src_strand=".",
                        L_ref=hslice.l_ref,
                        L_path=".",
                        length_ratio=".",
                        nonref_input_bp=".",
                        reference_like_bp=".",
                        repeat_like_bp=".",
                        short_bp=".",
                        unresolved_bp=".",
                        accepted_source_bp=".",
                        supported_events=".",
                    )
                )

    summary_metrics: dict[str, Any] = {
        "centromere_region_count": len(partition.regions),
        "centromere_island_event_count": len(island_sv_events),
        "centromere_unresolved_count": len(unresolved_records),
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

    return CentromereOutput(
        centromere_rows=tuple(centromere_rows),
        island_slot_blocks=tuple(island_slot_blocks),
        island_sv_events=tuple(island_sv_events),
        unresolved_records=tuple(unresolved_records),
        json_log_entries=tuple(json_log_entries),
        summary_metrics=summary_metrics,
    )
