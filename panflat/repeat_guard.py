"""Shared repeat-guard, partial-node slicing, mappy 50bp reference subtraction, and 4-criterion similarity clustering primitives (§3.6.3, §3.6.6, §3.8.4-B, §3.8.4-C)."""

from __future__ import annotations

import bisect
import collections
import functools
import re
from typing import Any, Literal, Mapping, Sequence

import mappy
import numpy as np

from panflat.edlib_similarity import (
    EDLIB_MAX_LEN,
    _coalesce_ops,
    _evaluate_edlib_similarity_branch,
    _has_distinct_edlib_del_split_copy,
    parse_edlib_cigar_to_ops,
)
from panflat.gfa_io import revcomp
from panflat.models import (
    InputContractError,
    LazyProvenanceTuple,
    Step,
    StepProvenance,
)

__all__ = [
    "EDLIB_MAX_LEN",
    "REF_SUB_BLOCK_SIZE",
    "REF_SUB_MIN_ALIGNED_RATIO",
    "REF_SUB_MIN_IDENTITY",
    "REPEAT_KMER_SIZE",
    "REPEAT_MIN_SEQ_LEN",
    "REPEAT_SINGLETON_THRESHOLD",
    "REPEAT_VALID_POS_RATIO",
    "REPEAT_WINDOW_SIZE",
    "REPEAT_WINDOW_STEP",
    "_check_similarity_merge_with_span",
    "_evaluate_mappy_hits_on_blocks",
    "_has_distinct_edlib_del_split_copy",
    "_repeat_mask_intervals_cached",
    "check_similarity_merge",
    "complement_intervals",
    "merge_contiguous_provenance",
    "merge_intervals",
    "parse_cs_to_query_ops",
    "parse_edlib_cigar_to_ops",
    "repeat_mask_intervals",
    "slice_candidate_by_intervals",
    "subtract_intervals",
    "subtract_reference_intervals_mappy",
]

# Fixed initial screening constants (§3.8.4-C item 8, reported in summary.tsv)
REPEAT_MIN_SEQ_LEN: int = 500
REPEAT_WINDOW_SIZE: int = 1000
REPEAT_WINDOW_STEP: int = 500
REPEAT_KMER_SIZE: int = 15
REPEAT_VALID_POS_RATIO: float = 0.90
REPEAT_SINGLETON_THRESHOLD: float = 0.30

REF_SUB_BLOCK_SIZE: int = 50
REF_SUB_MIN_ALIGNED_RATIO: float = 0.90
REF_SUB_MIN_IDENTITY: float = 0.85


_BASE_LUT = np.full(256, 4, dtype=np.uint8)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("a")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("c")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("g")] = 2
_BASE_LUT[ord("T")] = 3
_BASE_LUT[ord("t")] = 3

_CS_TOKEN_RE = re.compile(
    r":([0-9]+)|=([ACGTNacgtn]+)|\*([acgtn][acgtn])|\+([acgtn]+)|-([acgtn]+)|~[acgtn]{2}([0-9]+)[acgtn]{2}"
)

def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Sort and merge overlapping or adjacent half-open intervals `[start, end)`.
    Empty intervals (`start >= end`) are discarded.
    """
    valid = [(int(s), int(e)) for s, e in intervals if s < e]
    if not valid:
        return []
    valid.sort(key=lambda iv: (iv[0], iv[1]))
    merged: list[tuple[int, int]] = [valid[0]]
    for s, e in valid[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            if e > prev_e:
                merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def complement_intervals(
    total_len: int,
    masked_intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Return maximal contiguous unmasked intervals in `[0, total_len)` after removing
    `masked_intervals`.
    """
    if total_len <= 0:
        return []
    clamped = [
        (max(0, s), min(total_len, e))
        for s, e in masked_intervals
        if max(0, s) < min(total_len, e)
    ]
    merged_mask = merge_intervals(clamped)
    if not merged_mask:
        return [(0, total_len)]
    unmasked: list[tuple[int, int]] = []
    cursor = 0
    for s, e in merged_mask:
        if cursor < s:
            unmasked.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_len:
        unmasked.append((cursor, total_len))
    return unmasked


def subtract_intervals(
    base_intervals: Sequence[tuple[int, int]],
    remove_intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Compute `(union of base_intervals) \\ (union of remove_intervals)` as sorted
    disjoint half-open intervals.
    """
    merged_base = merge_intervals(base_intervals)
    if not merged_base:
        return []
    merged_rem = merge_intervals(remove_intervals)
    if not merged_rem:
        return merged_base

    result: list[tuple[int, int]] = []
    for b_s, b_e in merged_base:
        cur_s = b_s
        for r_s, r_e in merged_rem:
            if r_e <= cur_s:
                continue
            if r_s >= b_e:
                break
            if r_s > cur_s:
                result.append((cur_s, min(b_e, r_s)))
            cur_s = max(cur_s, r_e)
            if cur_s >= b_e:
                break
        if cur_s < b_e:
            result.append((cur_s, b_e))
    return result


def merge_contiguous_provenance(
    provenance: Sequence[StepProvenance],
    sub_seq: str | None = None,
) -> list[tuple[StepProvenance, str]]:
    """
    Merge adjacent `StepProvenance` entries that belong to the same `(src_hap, src_path, src_strand)`
    and are contiguous on the original unflipped path (`prev.src_end == curr.src_start` for `+`,
    or `prev.src_start == curr.src_end` for `-`).
    """
    if isinstance(provenance, LazyProvenanceTuple):
        runs = provenance.merged_runs()
        if not runs:
            return []
        seq_str = sub_seq if sub_seq is not None else ""
        total_prov_len = runs[-1][2]
        has_seq = len(seq_str) == total_prov_len
        if has_seq:
            return [(prov, seq_str[s_bp:e_bp]) for prov, s_bp, e_bp in runs]
        return [(prov, "") for prov, _, _ in runs]

    if not provenance:
        return []
    seq_str = sub_seq if sub_seq is not None else ""
    total_prov_len = sum(p.length for p in provenance)
    has_seq = len(seq_str) == total_prov_len

    merged: list[tuple[StepProvenance, str]] = []
    offset = 0
    for prov in provenance:
        plen = prov.length
        p_seq = seq_str[offset : offset + plen] if has_seq else ""
        offset += plen
        if not merged:
            merged.append((prov, p_seq))
            continue
        prev_prov, prev_seq = merged[-1]
        if (
            prev_prov.src_hap == prov.src_hap
            and prev_prov.src_path == prov.src_path
            and prev_prov.src_strand == prov.src_strand
        ):
            if prev_prov.src_strand == "+" and prev_prov.src_end == prov.src_start:
                merged[-1] = (
                    StepProvenance(
                        src_hap=prev_prov.src_hap,
                        src_path=prev_prov.src_path,
                        src_start=prev_prov.src_start,
                        src_end=prov.src_end,
                        src_strand="+",
                    ),
                    prev_seq + p_seq,
                )
                continue
            if prev_prov.src_strand == "-" and prev_prov.src_start == prov.src_end:
                merged[-1] = (
                    StepProvenance(
                        src_hap=prev_prov.src_hap,
                        src_path=prev_prov.src_path,
                        src_start=prov.src_start,
                        src_end=prev_prov.src_end,
                        src_strand="-",
                    ),
                    prev_seq + p_seq,
                )
                continue
        merged.append((prov, p_seq))
    return merged


_RC5_LUT = np.array(
    [
        ((3 - ((x >> 8) & 3)) << 0)
        | ((3 - ((x >> 6) & 3)) << 2)
        | ((3 - ((x >> 4) & 3)) << 4)
        | ((3 - ((x >> 2) & 3)) << 6)
        | ((3 - (x & 3)) << 8)
        for x in range(1024)
    ],
    dtype=np.uint32,
)
_REPEAT_CACHE_MAX_SEQ_LEN: int = 200_000


def _compute_repeat_mask_immutable(
    seq: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int, int, int, int, float | None, str], ...],
]:
    seq_len = len(seq)
    if seq_len <= REPEAT_WINDOW_SIZE:
        window_bounds: list[tuple[int, int]] = [(0, seq_len)]
    else:
        starts = sorted(
            set(range(0, seq_len - REPEAT_WINDOW_SIZE + 1, REPEAT_WINDOW_STEP))
            | {seq_len - REPEAT_WINDOW_SIZE}
        )
        window_bounds = [(s, s + REPEAT_WINDOW_SIZE) for s in starts]

    # Vectorized 30-bit canonical 15-mer encoding via three 5-mer blocks
    arr = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    codes = _BASE_LUT[arr]

    invalid_prefix = np.zeros(seq_len + 1, dtype=np.int32)
    np.cumsum(codes > 3, out=invalid_prefix[1:])
    valid_mask = (
        invalid_prefix[REPEAT_KMER_SIZE:] - invalid_prefix[:-REPEAT_KMER_SIZE]
    ) == 0

    safe = np.where(codes <= 3, codes, 0).astype(np.uint32)
    s5 = (
        (safe[:-4] << np.uint32(8))
        | (safe[1:-3] << np.uint32(6))
        | (safe[2:-2] << np.uint32(4))
        | (safe[3:-1] << np.uint32(2))
        | safe[4:]
    )
    fwd = (
        (s5[:-10] << np.uint32(20))
        | (s5[5:-5] << np.uint32(10))
        | s5[10:]
    )
    r5 = _RC5_LUT[s5]
    rev = (
        (r5[10:] << np.uint32(20))
        | (r5[5:-5] << np.uint32(10))
        | r5[:-10]
    )
    canon = np.minimum(fwd, rev, out=fwd)

    diag_tuples: list[
        tuple[int, int, int, int, int, int, float | None, str]
    ] = []
    for w_s, w_e in window_bounds:
        total_pos = (w_e - w_s) - (REPEAT_KMER_SIZE - 1)
        w_valid = valid_mask[w_s : w_s + total_pos]
        valid_pos = int(w_valid.sum())

        # Strictly < 90% valid 15-mer positions -> "unknown"
        if valid_pos * 10 < total_pos * 9:
            state: Literal["low", "non-low", "unknown"] = "unknown"
            singleton_count = 0
            unique_keys = 0
            singleton_ratio: float | None = None
        else:
            w_canon = canon[w_s : w_s + total_pos]
            if valid_pos < total_pos:
                w_canon = w_canon[w_valid]
            if valid_pos == 1:
                unique_keys = 1
                singleton_count = 1
            else:
                w_sorted = np.sort(w_canon)
                diff = w_sorted[1:] != w_sorted[:-1]
                unique_keys = int(np.count_nonzero(diff)) + 1
                left_ok = np.empty(valid_pos, dtype=bool)
                left_ok[0] = True
                left_ok[1:] = diff
                right_ok = np.empty(valid_pos, dtype=bool)
                right_ok[:-1] = diff
                right_ok[-1] = True
                singleton_count = int(np.count_nonzero(left_ok & right_ok))
            singleton_ratio = singleton_count / valid_pos
            # Strictly < 0.30 -> "low", >= 0.30 -> "non-low"
            if singleton_count * 10 < valid_pos * 3:
                state = "low"
            else:
                state = "non-low"

        diag_tuples.append(
            (
                w_s,
                w_e,
                total_pos,
                valid_pos,
                unique_keys,
                singleton_count,
                singleton_ratio,
                state,
            )
        )

    # Mask interval construction
    if len(diag_tuples) == 1:
        # Short-block exception (500 <= L <= 1000)
        if diag_tuples[0][7] == "low":
            masked_intervals: list[tuple[int, int]] = [(0, seq_len)]
        else:
            masked_intervals = []
    else:
        raw_masked: list[tuple[int, int]] = []
        current_run: list[tuple[int, int]] = []
        for w_diag in diag_tuples:
            if w_diag[7] == "low":
                current_run.append((w_diag[0], w_diag[1]))
            else:
                if len(current_run) >= 2:
                    raw_masked.append((current_run[0][0], current_run[-1][1]))
                current_run = []
        if len(current_run) >= 2:
            raw_masked.append((current_run[0][0], current_run[-1][1]))
        masked_intervals = merge_intervals(raw_masked)

    return tuple(masked_intervals), tuple(diag_tuples)


@functools.lru_cache(maxsize=32)
def _repeat_mask_intervals_cached(
    seq: str,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, int, int, int, int, float | None, str], ...],
]:
    return _compute_repeat_mask_immutable(seq)


def repeat_mask_intervals(
    seq: str,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """
    Shared local repeat masking function (§3.8.4-C & §3.6.6).

    - Only evaluates sequences of length >= 500 bp (`L < 500` returns `([], [])`).
    - For `500 <= L <= 1000`, evaluates a single full window `[0, L)`.
    - For `L > 1000`, evaluates 1000 bp windows at 500 bp steps plus a right-aligned
      tail window `[L - 1000, L)`, deduplicated by window start and sorted ascending.
    - In each window, computes canonical 15-mer keys `min(kmer, revcomp(kmer))` across
      pure ACGT 15-mers. If valid 15-mer positions < 90% of `W - 14`, window state is
      `"unknown"`. Otherwise `singleton_ratio = singletons / valid_positions`;
      strictly `< 0.30` is `"low"`, `>= 0.30` is `"non-low"`.
    - For `L > 1000`, at least 2 adjacent `"low"` windows (`"unknown"` breaks runs)
      form a masked interval equal to their union.
    - Short-block exception: for `500 <= L <= 1000` (single window), if the single
      window is `"low"`, masks `[0, L)`.
    """
    seq_len = len(seq)
    if seq_len < REPEAT_MIN_SEQ_LEN:
        return [], []
    if seq_len <= _REPEAT_CACHE_MAX_SEQ_LEN:
        masked_tup, diag_tup = _repeat_mask_intervals_cached(seq)
    else:
        masked_tup, diag_tup = _compute_repeat_mask_immutable(seq)

    window_diagnostics: list[dict[str, Any]] = [
        {
            "start": w_s,
            "end": w_e,
            "total_positions": total_pos,
            "valid_positions": valid_pos,
            "unique_keys": unique_keys,
            "singleton_count": singleton_count,
            "singleton_ratio": singleton_ratio,
            "state": state,
        }
        for (
            w_s,
            w_e,
            total_pos,
            valid_pos,
            unique_keys,
            singleton_count,
            singleton_ratio,
            state,
        ) in diag_tup
    ]
    return list(masked_tup), window_diagnostics


def slice_candidate_by_intervals(
    steps: Sequence[Step],
    provenance: Sequence[StepProvenance],
    nodes: Mapping[str, str],
    keep_intervals: Sequence[tuple[int, int]],
    mask_intervals: Sequence[tuple[int, int]],
) -> tuple[
    list[tuple[int, int, tuple[Step, ...], tuple[StepProvenance, ...], str]],
    list[tuple[int, int, tuple[StepProvenance, ...], str]],
]:
    """
    Slice a candidate walk (`steps` + `provenance`) at base-coordinate intervals
    `keep_intervals` and `mask_intervals` (§3.6.6 & §3.8.4).

    When an interval boundary falls inside a GFA node, creates a partial-node `Step`
    (`is_partial=True`, `slice_start`, `slice_end`) and computes exact 0-based
    `StepProvenance` coordinates respecting both `step.orient` and `prov.src_strand`.
    """
    if len(steps) != len(provenance):
        raise InputContractError(
            f"steps length ({len(steps)}) != provenance length ({len(provenance)})"
        )

    step_records: list[tuple[int, int, Step, StepProvenance, str, int]] = []
    cand_offset = 0
    for st, prov in zip(steps, provenance):
        if st.node_id not in nodes:
            raise InputContractError(f"Unknown node ID '{st.node_id}' in candidate walk")
        raw_seq = nodes[st.node_id]
        full_node_len = len(raw_seq)
        oriented_seq = raw_seq if st.orient == "+" else revcomp(raw_seq)
        s_start = st.slice_start
        s_end = full_node_len if st.slice_end is None else st.slice_end
        if s_start < 0 or s_end > full_node_len or s_start >= s_end:
            raise InputContractError(
                f"Invalid step slice [{s_start}, {s_end}) on node '{st.node_id}' "
                f"of length {full_node_len}"
            )
        step_seq = oriented_seq[s_start:s_end]
        step_len = len(step_seq)
        if prov.length != step_len:
            raise InputContractError(
                f"Provenance length ({prov.length}) != step length ({step_len}) "
                f"for node '{st.node_id}'"
            )
        step_records.append(
            (cand_offset, cand_offset + step_len, st, prov, step_seq, full_node_len)
        )
        cand_offset += step_len

    total_cand_len = cand_offset

    def _slice_single_interval(
        iv_start: int,
        iv_end: int,
    ) -> tuple[tuple[Step, ...], tuple[StepProvenance, ...], str]:
        if iv_start < 0 or iv_end > total_cand_len or iv_start >= iv_end:
            raise InputContractError(
                f"Slice interval [{iv_start}, {iv_end}) out of bounds for "
                f"candidate of length {total_cand_len}"
            )
        out_steps: list[Step] = []
        out_provs: list[StepProvenance] = []
        seq_parts: list[str] = []

        for st_start, st_end, st, prov, step_seq, full_node_len in step_records:
            if st_end <= iv_start:
                continue
            if st_start >= iv_end:
                break
            ov_start = max(iv_start, st_start)
            ov_end = min(iv_end, st_end)
            rel_s = ov_start - st_start
            rel_e = ov_end - st_start

            sliced_step = st.subslice(rel_s, rel_e, full_node_len)
            sliced_prov = prov.subslice(rel_s, rel_e)
            out_steps.append(sliced_step)
            out_provs.append(sliced_prov)
            seq_parts.append(step_seq[rel_s:rel_e])

        return tuple(out_steps), tuple(out_provs), "".join(seq_parts)

    kept_result: list[
        tuple[int, int, tuple[Step, ...], tuple[StepProvenance, ...], str]
    ] = []
    for s, e in keep_intervals:
        if s >= e:
            continue
        st_tuple, pr_tuple, sub_seq = _slice_single_interval(s, e)
        kept_result.append((s, e, st_tuple, pr_tuple, sub_seq))

    masked_result: list[tuple[int, int, tuple[StepProvenance, ...], str]] = []
    for s, e in mask_intervals:
        if s >= e:
            continue
        _, pr_tuple, sub_seq = _slice_single_interval(s, e)
        masked_result.append((s, e, pr_tuple, sub_seq))

    return kept_result, masked_result


def parse_cs_to_query_ops(
    cs_str: str,
    strand: int,
) -> list[tuple[str, int]]:
    """
    Parse a `mappy` `cs` string into coalesced `(op_type, length)` operations
    ordered along forward query coordinates `[hit.q_st, hit.q_en)`.

    `op_type` is one of `'M'`, `'X'`, `'I'`, `'D'`.
    When `strand == -1`, `mappy` emits `cs` in reverse-complement query order
    (`q_en` down to `q_st`), so reversing the token list restores ascending
    forward query order (`q_st` to `q_en`).
    """
    raw_ops: list[tuple[str, int]] = []
    for m in _CS_TOKEN_RE.finditer(cs_str):
        if m.group(1) is not None:
            raw_ops.append(("M", int(m.group(1))))
        elif m.group(2) is not None:
            raw_ops.append(("M", len(m.group(2))))
        elif m.group(3) is not None:
            raw_ops.append(("X", 1))
        elif m.group(4) is not None:
            raw_ops.append(("I", len(m.group(4))))
        elif m.group(5) is not None:
            raw_ops.append(("D", len(m.group(5))))
        elif m.group(6) is not None:
            raw_ops.append(("D", int(m.group(6))))
    if strand == -1:
        raw_ops.reverse()
    return _coalesce_ops(raw_ops)


def _partition_bridged_hit_ops(
    ops: Sequence[tuple[str, int]],
    q_start: int,
    r_st: int,
    r_en: int,
    strand: int,
    t_starts: Sequence[int],
    t_ends: Sequence[int],
) -> dict[int, list[tuple[int, int, int, str, int]]]:
    """
    Partition query-ordered alignment `ops` of a hit bridging concatenated target pieces
    `[t_starts[k], t_ends[k])` into per-piece `(q_pos, r_lo, r_hi, op, sub_len)` runs,
    discarding separator gaps. For `strand == -1`, reference intervals are emitted in
    forward target coordinates `[r_pos - step, r_pos)`.
    """
    piece_runs: dict[int, list[tuple[int, int, int, str, int]]] = (
        collections.defaultdict(list)
    )
    n_t = len(t_starts)
    q_cur = q_start
    r_cur = r_st if strand == 1 else r_en

    for op, length in ops:
        if op == "I":
            k_i = bisect.bisect_right(t_starts, r_cur) - 1
            if k_i >= 0 and t_starts[k_i] < r_cur < t_ends[k_i]:
                piece_runs[k_i].append((q_cur, r_cur, r_cur, "I", length))
            q_cur += length
        elif op in ("M", "X", "D"):
            rem = length
            r_pos = r_cur
            q_pos = q_cur
            if strand == 1:
                r_lim = r_cur + length
                while rem > 0:
                    k_p = bisect.bisect_right(t_starts, r_pos) - 1
                    if k_p >= 0 and r_pos < t_ends[k_p]:
                        step = min(rem, t_ends[k_p] - r_pos)
                        piece_runs[k_p].append((q_pos, r_pos, r_pos + step, op, step))
                        r_pos += step
                        if op != "D":
                            q_pos += step
                        rem -= step
                    else:
                        next_k = k_p + 1
                        if next_k >= n_t or t_starts[next_k] >= r_lim:
                            r_pos += rem
                            if op != "D":
                                q_pos += rem
                            rem = 0
                        else:
                            gap_step = t_starts[next_k] - r_pos
                            r_pos += gap_step
                            if op != "D":
                                q_pos += gap_step
                            rem -= gap_step
            else:
                r_low = r_cur - length
                while rem > 0:
                    k_p = bisect.bisect_right(t_starts, r_pos - 1) - 1
                    if k_p >= 0 and r_pos - 1 < t_ends[k_p]:
                        step = min(rem, r_pos - t_starts[k_p])
                        piece_runs[k_p].append((q_pos, r_pos - step, r_pos, op, step))
                        r_pos -= step
                        if op != "D":
                            q_pos += step
                        rem -= step
                    else:
                        if k_p < 0 or t_ends[k_p] <= r_low:
                            r_pos -= rem
                            if op != "D":
                                q_pos += rem
                            rem = 0
                        else:
                            gap_step = r_pos - t_ends[k_p]
                            r_pos -= gap_step
                            if op != "D":
                                q_pos += gap_step
                            rem -= gap_step
            r_cur = r_pos
            q_cur = q_pos

    cleaned: dict[int, list[tuple[int, int, int, str, int]]] = {}
    for k_p, runs_k in piece_runs.items():
        lo = 0
        hi = len(runs_k)
        while lo < hi and runs_k[lo][3] == "D":
            lo += 1
        while hi > lo and runs_k[hi - 1][3] == "D":
            hi -= 1
        if lo < hi:
            cleaned[k_p] = runs_k[lo:hi]
    return cleaned


def _accumulate_block_op(
    q_pos: int,
    op: str,
    length: int,
    q_len: int,
    num_blocks: int,
    block_m: list[int],
    block_x: list[int],
    block_i: list[int],
    block_d: list[int],
) -> None:
    if op in ("M", "X", "I"):
        q_next = q_pos + length
        start_blk = max(0, q_pos // REF_SUB_BLOCK_SIZE)
        end_blk = min(num_blocks - 1, (q_next - 1) // REF_SUB_BLOCK_SIZE)
        for k in range(start_blk, end_blk + 1):
            b_s = k * REF_SUB_BLOCK_SIZE
            b_e = min((k + 1) * REF_SUB_BLOCK_SIZE, q_len)
            ov = min(q_next, b_e) - max(q_pos, b_s)
            if ov > 0:
                if op == "M":
                    block_m[k] += ov
                elif op == "X":
                    block_x[k] += ov
                else:
                    block_i[k] += ov
    elif op == "D":
        # Assign D to the block of its right-hand query base, or last block if at query end
        k = min(max(0, q_pos // REF_SUB_BLOCK_SIZE), num_blocks - 1)
        block_d[k] += length


def _finalize_hit_block_counts(
    block_m: Sequence[int],
    block_x: Sequence[int],
    block_i: Sequence[int],
    block_d: Sequence[int],
    protected_ins: Sequence[tuple[int, int]],
    cov_spans: Sequence[tuple[int, int]],
    q_len: int,
    num_blocks: int,
    min_identity: float,
    hit_masked_intervals: list[tuple[int, int]],
    hit_aligned_intervals: list[tuple[int, int]],
) -> None:
    hit_qualifying_blocks: list[tuple[int, int]] = []
    hit_aligned_blocks: list[tuple[int, int]] = []
    for k in range(num_blocks):
        b_s = k * REF_SUB_BLOCK_SIZE
        b_e = min((k + 1) * REF_SUB_BLOCK_SIZE, q_len)
        l_k = b_e - b_s
        m_cnt = block_m[k]
        x_cnt = block_x[k]
        i_cnt = block_i[k]
        d_cnt = block_d[k]
        denom = m_cnt + x_cnt + i_cnt + d_cnt

        # Rule (i): blocks >=90% spanned by (M + X + I) are marked already aligned;
        # protected_ins is subtracted below so >=min_sv_len insertions remain re-mappable
        # for embedded ectopic homologs without dragging adjacent flanks into short_aligner.
        if (m_cnt + x_cnt + i_cnt) * 10 >= 9 * l_k:
            hit_aligned_blocks.append((b_s, b_e))
        elif protected_ins and any(
            b_s <= pe and ps <= b_e for ps, pe in protected_ins
        ):
            # Rule (ii-clean): even when mid-block soft-clipping leaves (M+X+I) < 90%
            # inside a block touching a protected_ins boundary (closed endpoint), mark
            # this hit's covered flank bases in that block as aligned so zero adjacent
            # flank bases leak into unaligned alongside protected_ins.
            for cs_i, ce_i in cov_spans:
                ov_s = max(b_s, cs_i)
                ov_e = min(b_e, ce_i)
                if ov_e > ov_s:
                    hit_aligned_blocks.append((ov_s, ov_e))

        # Masking Criterion (§3.8.4-B — unchanged):
        # Condition 1: (M + X) / L_k >= 0.90
        if (m_cnt + x_cnt) * 10 >= 9 * l_k:
            # Condition 2: M / (M + X + I + D) >= min_identity
            if denom > 0 and (m_cnt + 1e-12) >= min_identity * denom:
                hit_qualifying_blocks.append((b_s, b_e))

    if hit_qualifying_blocks:
        hit_masked = subtract_intervals(hit_qualifying_blocks, protected_ins)
        hit_masked_intervals.extend(hit_masked)
    if hit_aligned_blocks:
        hit_aligned = subtract_intervals(hit_aligned_blocks, protected_ins)
        hit_aligned_intervals.extend(hit_aligned)


def _evaluate_mappy_hits_on_blocks(
    hits: Sequence[Any],
    query_offset: int,
    q_len: int,
    num_blocks: int,
    min_sv_len: int,
    min_identity: float = REF_SUB_MIN_IDENTITY,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Evaluate `mappy` hits on the global 50 bp block grid `[50*k, min(50*(k+1), q_len))`.
    When `t_starts` and `t_ends` are supplied (concatenated target pieces separated by
    `'N' * 200`), any hit bridging across piece boundaries is split per target piece so
    separator gaps are discarded and adjacent target pieces are evaluated independently.
    Returns `(hit_masked_intervals, hit_aligned_intervals)`.
    """
    hit_masked_intervals: list[tuple[int, int]] = []
    hit_aligned_intervals: list[tuple[int, int]] = []
    check_targets = t_starts is not None and t_ends is not None

    for hit in hits:
        if not getattr(hit, "cs", None):
            continue
        ops = parse_cs_to_query_ops(hit.cs, hit.strand)

        use_direct = True
        if check_targets:
            assert t_starts is not None and t_ends is not None
            pos_t = bisect.bisect_right(t_starts, int(hit.r_st)) - 1
            if pos_t < 0 or int(hit.r_en) > t_ends[pos_t]:
                use_direct = False

        if use_direct:
            block_m = [0] * num_blocks
            block_x = [0] * num_blocks
            block_i = [0] * num_blocks
            block_d = [0] * num_blocks
            protected_ins: list[tuple[int, int]] = []

            q_st_abs = query_offset + int(hit.q_st)
            q_pos = q_st_abs
            for op, length in ops:
                if op == "I" and length >= min_sv_len:
                    protected_ins.append((q_pos, q_pos + length))
                _accumulate_block_op(
                    q_pos,
                    op,
                    length,
                    q_len,
                    num_blocks,
                    block_m,
                    block_x,
                    block_i,
                    block_d,
                )
                if op in ("M", "X", "I"):
                    q_pos += length

            _finalize_hit_block_counts(
                block_m=block_m,
                block_x=block_x,
                block_i=block_i,
                block_d=block_d,
                protected_ins=protected_ins,
                cov_spans=((q_st_abs, q_pos),) if protected_ins else (),
                q_len=q_len,
                num_blocks=num_blocks,
                min_identity=min_identity,
                hit_masked_intervals=hit_masked_intervals,
                hit_aligned_intervals=hit_aligned_intervals,
            )
        else:
            # Bridged hit across concatenated target pieces: partition ops strictly by
            # target piece [t_starts[k], t_ends[k]), discarding separator gaps.
            assert t_starts is not None and t_ends is not None
            piece_runs = _partition_bridged_hit_ops(
                ops=ops,
                q_start=query_offset + int(hit.q_st),
                r_st=int(hit.r_st),
                r_en=int(hit.r_en),
                strand=int(hit.strand),
                t_starts=t_starts,
                t_ends=t_ends,
            )

            for k_piece in sorted(piece_runs):
                block_m = [0] * num_blocks
                block_x = [0] * num_blocks
                block_i = [0] * num_blocks
                block_d = [0] * num_blocks
                protected_ins = []
                runs_k = piece_runs[k_piece]
                for q_p, _r_lo, _r_hi, op, sub_len in runs_k:
                    if op == "I" and sub_len >= min_sv_len:
                        protected_ins.append((q_p, q_p + sub_len))
                    _accumulate_block_op(
                        q_p,
                        op,
                        sub_len,
                        q_len,
                        num_blocks,
                        block_m,
                        block_x,
                        block_i,
                        block_d,
                    )
                cov_k: Sequence[tuple[int, int]] = (
                    [
                        (q_p, q_p + sub_len)
                        for q_p, _r_lo, _r_hi, op, sub_len in runs_k
                        if op in ("M", "X", "I")
                    ]
                    if protected_ins
                    else ()
                )
                _finalize_hit_block_counts(
                    block_m=block_m,
                    block_x=block_x,
                    block_i=block_i,
                    block_d=block_d,
                    protected_ins=protected_ins,
                    cov_spans=cov_k,
                    q_len=q_len,
                    num_blocks=num_blocks,
                    min_identity=min_identity,
                    hit_masked_intervals=hit_masked_intervals,
                    hit_aligned_intervals=hit_aligned_intervals,
                )

    return hit_masked_intervals, hit_aligned_intervals


def _filter_shadowed_short_hits(
    main_hits: Sequence[Any],
    short_hits: Sequence[Any],
) -> list[Any]:
    """Drop short-aligner hits strictly contained in a wider hit on the same `(ctg, strand)`."""
    if not short_hits:
        return []
    kept: list[Any] = []
    for sh in short_hits:
        s_ctg = getattr(sh, "ctg", None)
        s_str = int(sh.strand)
        s_qs, s_qe = int(sh.q_st), int(sh.q_en)
        s_rs, s_re = int(sh.r_st), int(sh.r_en)
        s_qspan = s_qe - s_qs
        shadowed = False
        for is_main, group in ((True, main_hits), (False, short_hits)):
            for ah in group:
                if ah is sh:
                    continue
                if getattr(ah, "ctg", None) != s_ctg or int(ah.strand) != s_str:
                    continue
                a_qs, a_qe = int(ah.q_st), int(ah.q_en)
                a_rs, a_re = int(ah.r_st), int(ah.r_en)
                if (
                    a_qs <= s_qs
                    and s_qe <= a_qe
                    and a_rs <= s_rs
                    and s_re <= a_re
                    and (is_main or (a_qe - a_qs) > s_qspan)
                ):
                    shadowed = True
                    break
            if shadowed:
                break
        if not shadowed:
            kept.append(sh)
    return kept


def subtract_reference_intervals_mappy(
    query_seq: str,
    ref_region_seq: str = "",
    min_sv_len: int = 50,
    aligner: Any | None = None,
    buf: Any | None = None,
    min_identity: float = REF_SUB_MIN_IDENTITY,
    short_aligner: Any | None = None,
    short_buf: Any | None = None,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
    collect_short_slices: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """
    Reference / homologous interval subtraction using `mappy` (`preset='asm20'`, `cs=True`) (§3.8.4-B).

    - Evaluates fixed non-overlapping 50 bp query blocks (`[0, 50), [50, 100), ...`,
      final tail block evaluated at actual length `L_k`).
    - Both forward (`+`) and reverse (`-`) hits are mapped to forward query coordinates.
    - Deletions (`D`) are assigned to the 50 bp block of the query base immediately to
      their right, or to the last block if located at the end of the query.
    - Any contiguous query insertion (`I`) with `length >= min_sv_len` within a hit is
      protected from being masked by that hit.
    - A block `k` is masked by a hit iff:
      1. `(M + X) / L_k >= 0.90` (at least 90% aligned query bases), AND
      2. `M / (M + X + I + D) >= min_identity` (default `0.85`, I/D counted in bases).
    - When `short_aligner` is supplied, queries or unaligned slices with length `<= 500 bp`
      also query `short_aligner` (with optional `short_buf`) and merge hits.
    - Returns `(masked_intervals, unmasked_intervals, reference_like_bp)`.
    """
    q_len = len(query_seq)
    if q_len == 0:
        return [], [], 0
    if aligner is None and short_aligner is None and not ref_region_seq:
        return [], [(0, q_len)], 0

    q_upper = query_seq.upper()
    if aligner is None and ref_region_seq:
        aligner = mappy.Aligner(seq=ref_region_seq.upper(), preset="asm20")

    num_blocks = (q_len + REF_SUB_BLOCK_SIZE - 1) // REF_SUB_BLOCK_SIZE

    def _map_and_evaluate_slice(
        sub_seq: str,
        query_offset: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        main_hits: list[Any] = []
        if aligner is not None:
            main_hits = list(
                aligner.map(sub_seq, buf=buf, cs=True)
                if buf is not None
                else aligner.map(sub_seq, cs=True)
            )
        short_hits: list[Any] = []
        if short_aligner is not None and len(sub_seq) <= 500:
            if collect_short_slices is not None:
                collect_short_slices.append((query_offset, query_offset + len(sub_seq)))
            saln = short_aligner() if callable(short_aligner) else short_aligner
            if saln is not None:
                short_hits = list(
                    saln.map(sub_seq, buf=short_buf, cs=True)
                    if short_buf is not None
                    else saln.map(sub_seq, cs=True)
                )
        if not main_hits and not short_hits:
            return [], []
        kept_short = _filter_shadowed_short_hits(main_hits, short_hits)
        combined_hits = main_hits + kept_short if main_hits else kept_short
        return _evaluate_mappy_hits_on_blocks(
            hits=combined_hits,
            query_offset=query_offset,
            q_len=q_len,
            num_blocks=num_blocks,
            min_sv_len=min_sv_len,
            min_identity=min_identity,
            t_starts=t_starts,
            t_ends=t_ends,
        )

    all_hit_masked, all_hit_aligned = _map_and_evaluate_slice(q_upper, 0)
    masked_intervals = merge_intervals(all_hit_masked)
    aligned_intervals = merge_intervals(all_hit_aligned)

    # When one or more large query insertions (e.g. >= 600 bp in C8) separate homologous
    # flanks, minimap2's RMQ chainer may chain flanks across the insertions and clip
    # downstream flanks at zdrop during DP. Iteratively re-mapping only the unaligned
    # sub-intervals (without rounding outward across already-aligned flank bases adjacent
    # to protected_ins boundaries) discovers clipped flanks while preserving protected_ins
    # and blocks already aligned and rejected due to < min_identity.
    seen_slices: set[tuple[int, int]] = {(0, q_len)}
    while aligned_intervals:
        unaligned = complement_intervals(q_len, aligned_intervals)
        for u_s, u_e in unaligned:
            if (
                u_e - u_s >= REF_SUB_BLOCK_SIZE
                and (u_s, u_e) not in seen_slices
            ):
                seen_slices.add((u_s, u_e))
                extra_masked, extra_aligned = _map_and_evaluate_slice(
                    q_upper[u_s:u_e], u_s
                )
                if extra_masked:
                    all_hit_masked.extend(
                        (max(a, u_s), min(b, u_e))
                        for a, b in extra_masked
                        if min(b, u_e) > max(a, u_s)
                    )
                if extra_aligned:
                    all_hit_aligned.extend(
                        (max(a, u_s), min(b, u_e))
                        for a, b in extra_aligned
                        if min(b, u_e) > max(a, u_s)
                    )
        prev_aligned = aligned_intervals
        masked_intervals = merge_intervals(all_hit_masked)
        aligned_intervals = merge_intervals(all_hit_aligned)
        if aligned_intervals == prev_aligned:
            break

    unmasked_intervals = complement_intervals(q_len, masked_intervals)
    reference_like_bp = sum(e - s for s, e in masked_intervals)
    return masked_intervals, unmasked_intervals, reference_like_bp


def _check_similarity_merge_with_span(
    seq_s: str,
    seq_r: str,
    min_sv_len: int = 50,
    cluster_id: float = 0.95,
    cluster_win_id: float = 0.70,
    aligner: Any | None = None,
) -> tuple[bool, int, int, int]:
    """
    Evaluate the 4-criterion similarity clustering check (§3.6.3) and return
    `(merged, t_start, t_end, del_bp_internal)` on `seq_r` so callers (e.g. Case A
    `rep0` backbone genotyping) can reuse exact target alignment coordinates without
    running a second alignment.
    """
    if not seq_s or not seq_r:
        return False, 0, 0, 0

    s = seq_s.upper()
    r = seq_r.upper()
    len_s = len(s)
    len_r = len(r)

    if s == r:
        return True, 0, len_r, 0

    # Fast bound check for Criterion 3: len_s - len_r <= unaligned_ends(S) + internal_ins_bp(S)
    if min_sv_len <= 0 or (len_s - len_r >= min_sv_len):
        return False, 0, 0, 0

    is_edlib_branch = len_s <= EDLIB_MAX_LEN and len_r <= EDLIB_MAX_LEN
    if is_edlib_branch:
        ok_edlib, ops, q_start, q_end, t_start, t_end, del_bp_internal = (
            _evaluate_edlib_similarity_branch(
                s=s,
                r=r,
                min_sv_len=min_sv_len,
                cluster_id=cluster_id,
            )
        )
        if not ok_edlib:
            return False, 0, 0, 0
    else:
        if aligner is None:
            aligner = mappy.Aligner(seq=r, preset="asm20")
        hits = list(aligner.map(s, cs=True))
        # Criterion 1: must have a single forward-strand alignment
        if len(hits) != 1:
            return False, 0, 0, 0
        hit = hits[0]
        if (
            hit.strand != 1
            or not getattr(hit, "is_primary", True)
            or not getattr(hit, "cs", None)
        ):
            return False, 0, 0, 0
        t_start = int(hit.r_st)
        t_end = int(hit.r_en)
        q_start = int(hit.q_st)
        q_end = int(hit.q_en)
        ops = parse_cs_to_query_ops(hit.cs, strand=1)

        # Criterion 1d: Reject split-alignment internal SV-scale deletions ('D')
        del_bp_internal = 0
        for op, length in ops:
            if op == "D":
                if length >= min_sv_len:
                    return False, 0, 0, 0
                del_bp_internal += length
        if del_bp_internal >= min_sv_len:
            return False, 0, 0, 0

        # Criterion 2: Gap-compressed identity ID_gc >= cluster_id
        m_total = sum(length for op, length in ops if op == "M")
        x_total = sum(length for op, length in ops if op == "X")
        gap_open_i = sum(1 for op, _ in ops if op == "I")
        gap_open_d = sum(1 for op, _ in ops if op == "D")
        denom_gc = m_total + x_total + gap_open_i + gap_open_d
        if denom_gc <= 0:
            return False, 0, 0, 0
        id_gc = m_total / denom_gc
        if id_gc + 1e-12 < cluster_id:
            return False, 0, 0, 0

        # Criterion 3: unaligned_ends(S) + internal_ins_bp(S) < min_sv_len
        unaligned_ends_s = q_start + (len_s - q_end)
        ins_bp_s = sum(length for op, length in ops if op == "I")
        if unaligned_ends_s + ins_bp_s >= min_sv_len:
            return False, 0, 0, 0

    # Criterion 4: Local sliding window protection along S
    win_len = min(len_s, min_sv_len)
    if win_len <= 0:
        return False, 0, 0, 0

    match_arr = np.zeros(len_s, dtype=np.int32)
    mis_arr = np.zeros(len_s, dtype=np.int32)
    ins_arr = np.zeros(len_s, dtype=np.int32)
    ins_open_arr = np.zeros(len_s, dtype=np.int32)
    del_bp_arr = np.zeros(len_s, dtype=np.int32)
    del_open_arr = np.zeros(len_s, dtype=np.int32)

    if q_start > 0:
        ins_arr[:q_start] = 1
        ins_open_arr[q_start - 1] += 1
    if q_end < len_s:
        ins_arr[q_end:] = 1
        ins_open_arr[q_end] += 1

    q_pos = q_start
    for op, length in ops:
        if op == "M":
            match_arr[q_pos : q_pos + length] = 1
            q_pos += length
        elif op == "X":
            mis_arr[q_pos : q_pos + length] = 1
            q_pos += length
        elif op == "I":
            ins_arr[q_pos : q_pos + length] = 1
            ins_open_arr[q_pos] += 1
            q_pos += length
        elif op == "D":
            target_q = min(max(0, q_pos), len_s - 1)
            del_bp_arr[target_q] += length
            del_open_arr[target_q] += 1

    match_csum = np.zeros(len_s + 1, dtype=np.int32)
    mis_csum = np.zeros(len_s + 1, dtype=np.int32)
    ins_csum = np.zeros(len_s + 1, dtype=np.int32)
    ins_open_csum = np.zeros(len_s + 1, dtype=np.int32)
    del_bp_csum = np.zeros(len_s + 1, dtype=np.int32)
    del_open_csum = np.zeros(len_s + 1, dtype=np.int32)

    np.cumsum(match_arr, out=match_csum[1:])
    np.cumsum(mis_arr, out=mis_csum[1:])
    np.cumsum(ins_arr, out=ins_csum[1:])
    np.cumsum(ins_open_arr, out=ins_open_csum[1:])
    np.cumsum(del_bp_arr, out=del_bp_csum[1:])
    np.cumsum(del_open_arr, out=del_open_csum[1:])

    win_m = match_csum[win_len:] - match_csum[:-win_len]
    win_x = mis_csum[win_len:] - mis_csum[:-win_len]
    win_i = ins_csum[win_len:] - ins_csum[:-win_len]
    win_i_open = ins_open_csum[win_len:] - ins_open_csum[:-win_len]
    win_d_bp = del_bp_csum[win_len:] - del_bp_csum[:-win_len]
    win_d_open = del_open_csum[win_len:] - del_open_csum[:-win_len]

    # Pure small insertions/deletions (win_x == 0) and T5-style scattered SNPs
    # (single gap with win_x < 4, or 0-SNP multi-insertions) remain governed by
    # Criterion 3 (< min_sv_len). Compensating indel substitutions (I+D), dense
    # multi-deletion clusters (win_d_open >= 4), and symmetric asymmetric
    # substitution blocks (insertion- or deletion-direction with win_x >= 4, or
    # >= 2 gaps + >= 3 mismatches, exceeding (1 - cluster_win_id)*win_len)
    # are penalized against `win_len + win_d_bp`.
    asym_sub_limit = (1.0 - cluster_win_id) * win_len + 1e-12
    is_compensating_indel_sub = (
        ((win_i > 0) & (win_d_bp > 0))
        | (win_d_open >= 4)
        | (
            (win_i > 0)
            & ((win_x >= 4) | ((win_x >= 3) & (win_i_open >= 2)))
            & ((win_i + win_x) > asym_sub_limit)
        )
        | (
            (win_d_bp > 0)
            & ((win_x >= 4) | ((win_x >= 3) & (win_d_open >= 2)))
            & ((win_d_bp + win_x) > asym_sub_limit)
        )
    )
    win_aligned_cols = np.where(
        is_compensating_indel_sub,
        win_len + win_d_bp,
        np.maximum(1, win_m + win_x + win_d_open),
    )

    if np.any(win_m + 1e-12 < cluster_win_id * win_aligned_cols):
        return False, 0, 0, 0

    return True, t_start, t_end, del_bp_internal


def check_similarity_merge(
    seq_s: str,
    seq_r: str,
    min_sv_len: int = 50,
    cluster_id: float = 0.95,
    cluster_win_id: float = 0.70,
    aligner: Any | None = None,
) -> bool:
    """
    4-criterion similarity clustering check (§3.6.3) testing whether candidate `seq_s`
    can be merged into representative `seq_r`:

    1. Unique single forward-strand alignment (`strand == +1`, no tied/secondary
       forward hits, no competitive/distinct reverse-complement hits, and no internal
       SV-scale split-alignment deletion `>= min_sv_len`).
    2. Gap-compressed identity `ID_gc = M / (M + X + gap_open_I + gap_open_D) >= cluster_id`.
    3. `unaligned_ends(S) + internal_ins_bp(S) < min_sv_len`.
    4. Sliding window protection: for every window of length `min(len(S), min_sv_len)`
       sliding along `S`, `matched_columns / aligned_columns >= cluster_win_id`.

    Boundary between Criterion 3 and Criterion 4:
    - Pure small insertions (`1 .. min_sv_len - 1` bp, e.g., `200=30I200=`, `225=49I225=`,
      without compensating `D` or dense mismatches) are governed by Criterion 3
      (`unaligned_ends + internal_ins_bp < min_sv_len`) and intentionally excluded from
      the Criterion 4 window denominator so `16–49 bp` pure small insertions are absorbed
      as required by §0 / §3.6.3 / T5 / T13.
    - Compensating indel substitutions (`win_i > 0` with `win_d_bp > 0`), dense multi-deletion
      clusters (`win_d_open >= 4`), and symmetric insertion/deletion asymmetric substitution
      blocks (`win_i > 0` or `win_d_bp > 0` with dense mismatches exceeding
      `(1 - cluster_win_id) * win_len`) are evaluated against the full window denominator
      `win_len + win_d_bp`.

    Uses `edlib` (`mode='HW'`, `task='path'`) with Criterion 1 multi-hit/revcomp/split-deletion
    guards, local `asm20` two-piece affine edit-cluster normalization, and affine terminal
    soft-clipping when both sequences are `<= 1000 bp`, and `mappy` (`preset='asm20'`, `cs=True`)
    when either sequence is `> 1000 bp`.
    """
    return _check_similarity_merge_with_span(
        seq_s=seq_s,
        seq_r=seq_r,
        min_sv_len=min_sv_len,
        cluster_id=cluster_id,
        cluster_win_id=cluster_win_id,
        aligner=aligner,
    )[0]
