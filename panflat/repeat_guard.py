"""Shared repeat-guard, partial-node slicing, mappy 50bp reference subtraction, and 4-criterion similarity clustering primitives (§3.6.3, §3.6.6, §3.8.4-B, §3.8.4-C)."""

from __future__ import annotations

import bisect
import collections
import functools
import itertools
import math
import operator
import re
from typing import Any, Literal, Mapping, NamedTuple, Sequence

import edlib
import mappy
import numpy as np

from panflat.gfa_io import revcomp
from panflat.models import (
    InputContractError,
    LazyProvenanceTuple,
    Step,
    StepProvenance,
)

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

EDLIB_MAX_LEN: int = 1000

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
_EDLIB_CIGAR_RE = re.compile(r"([0-9]+)([=XID])")


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


def _coalesce_ops(
    ops: Sequence[tuple[str, int]],
) -> list[tuple[str, int]]:
    if not ops:
        return []
    coalesced: list[tuple[str, int]] = [ops[0]]
    for op, length in ops[1:]:
        if length <= 0:
            continue
        prev_op, prev_len = coalesced[-1]
        if op == prev_op:
            coalesced[-1] = (prev_op, prev_len + length)
        else:
            coalesced.append((op, length))
    return coalesced


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


def _emit_mx_runs(out: list[tuple[str, int]], s1: str, s2: str) -> None:
    """Append coalesced `('M'|'X', length)` operations comparing equal-length strings `s1` and `s2`."""
    n = len(s1)
    if n == 0:
        return
    cur_op = "M" if s1[0] == s2[0] else "X"
    cur_len = 1
    for p in range(1, n):
        op = "M" if s1[p] == s2[p] else "X"
        if op == cur_op:
            cur_len += 1
        else:
            out.append((cur_op, cur_len))
            cur_op = op
            cur_len = 1
    out.append((cur_op, cur_len))


_ASM20_MATCH = 1
_ASM20_MISMATCH = 4
_ASM20_GAP_O1 = 6
_ASM20_GAP_E1 = 2
_ASM20_GAP_O2 = 26
_ASM20_GAP_E2 = 1
_EDLIB_CLUSTER_ANCHOR_M = 15
_EDLIB_CLUSTER_DP_MAX_SPAN = 160
_EDLIB_CLUSTER_DP_MAX_CELLS = (
    _EDLIB_CLUSTER_DP_MAX_SPAN * _EDLIB_CLUSTER_DP_MAX_SPAN
)
_EDLIB_CLUSTER_DP_SKINNY_MIN_SPAN = 64
_EDLIB_MIN_CLEAN_INS_ARM = 8
_EDLIB_MIN_ALIGNED_CORE = 19
_EDLIB_RC_BLEED_TOL = (
    _EDLIB_MIN_CLEAN_INS_ARM - 1
)  # 7 bp / 7 asm20 score pts per trimmed overhang end


def _asm20_gap_penalty(length: int) -> int:
    """Two-piece affine gap penalty matching `mappy` (`preset='asm20'`)."""
    if length <= 0:
        return 0
    return min(
        _ASM20_GAP_O1 + _ASM20_GAP_E1 * length,
        _ASM20_GAP_O2 + _ASM20_GAP_E2 * length,
    )


def _asm20_ops_score(ops: Sequence[tuple[str, int]]) -> int:
    """Compute exact `mappy asm20` alignment score of coalesced `ops`."""
    score = 0
    for op, length in ops:
        if op == "M":
            score += _ASM20_MATCH * length
        elif op == "X":
            score -= _ASM20_MISMATCH * length
        elif op == "I" or op == "D":
            score -= _asm20_gap_penalty(length)
    return score


def _collapse_edlib_gap_spans(
    ops: list[tuple[str, int]],
    seq_s: str,
    seq_r: str,
) -> list[tuple[str, int]]:
    """
    Lossless Pass 1 exact 0-mismatch coalescing on pure `('M', gap_type)` runs
    (`'I'` or `'D'`). Restores `edlib`-shattered 0-SNP indels into a single
    contiguous gap in O(span) time without introducing mismatches.
    """
    if len(ops) < 3:
        return ops

    normalized: list[tuple[str, int]] = []
    q_idx = 0
    t_idx = 0
    i = 0
    n_ops = len(ops)

    while i < n_ops:
        op, length = ops[i]
        if op in ("D", "I"):
            gap_type = op
            allowed = ("M", "D") if gap_type == "D" else ("M", "I")
            j = i
            q_end = q_idx
            t_end = t_idx
            gap_count = 0
            candidates: list[tuple[int, int, int]] = []
            while j < n_ops and ops[j][0] in allowed:
                gop, glen = ops[j]
                if gop == "M":
                    q_end += glen
                    t_end += glen
                elif gop == "D":
                    t_end += glen
                    gap_count += 1
                else:
                    q_end += glen
                    gap_count += 1
                j += 1
                if gop == gap_type and gap_count >= 2:
                    candidates.append((j, q_end, t_end))

            collapsed = False
            for cand_j, cand_q_end, cand_t_end in reversed(candidates):
                q_span = cand_q_end - q_idx
                t_span = cand_t_end - t_idx
                if gap_type == "D":
                    span_len = q_span
                    d_total = t_span - q_span
                    s_sub = seq_s[q_idx:cand_q_end]
                    r_left = seq_r[t_idx : t_idx + span_len]
                    r_right = seq_r[cand_t_end - span_len : cand_t_end]
                    pref = [0] * (span_len + 1)
                    acc = 0
                    for p in range(span_len):
                        if s_sub[p] != r_left[p]:
                            acc += 1
                        pref[p + 1] = acc
                    min_x = pref[span_len]
                    best_split = span_len
                    suff = 0
                    for p in range(span_len - 1, -1, -1):
                        if s_sub[p] != r_right[p]:
                            suff += 1
                        tot = pref[p] + suff
                        if tot <= min_x:
                            min_x = tot
                            best_split = p
                    if min_x == 0:
                        _emit_mx_runs(
                            normalized, s_sub[:best_split], r_left[:best_split]
                        )
                        normalized.append(("D", d_total))
                        _emit_mx_runs(
                            normalized, s_sub[best_split:], r_right[best_split:]
                        )
                        collapsed = True
                else:
                    span_len = t_span
                    i_total = q_span - t_span
                    r_sub = seq_r[t_idx:cand_t_end]
                    s_left = seq_s[q_idx : q_idx + span_len]
                    s_right = seq_s[cand_q_end - span_len : cand_q_end]
                    pref = [0] * (span_len + 1)
                    acc = 0
                    for p in range(span_len):
                        if s_left[p] != r_sub[p]:
                            acc += 1
                        pref[p + 1] = acc
                    min_x = pref[span_len]
                    best_split = span_len
                    suff = 0
                    for p in range(span_len - 1, -1, -1):
                        if s_right[p] != r_sub[p]:
                            suff += 1
                        tot = pref[p] + suff
                        if tot <= min_x:
                            min_x = tot
                            best_split = p
                    if min_x == 0:
                        _emit_mx_runs(
                            normalized, s_left[:best_split], r_sub[:best_split]
                        )
                        normalized.append(("I", i_total))
                        _emit_mx_runs(
                            normalized, s_right[best_split:], r_sub[best_split:]
                        )
                        collapsed = True
                if collapsed:
                    q_idx = cand_q_end
                    t_idx = cand_t_end
                    i = cand_j
                    break
            if collapsed:
                continue

        normalized.append((op, length))
        if op == "M" or op == "X":
            q_idx += length
            t_idx += length
        elif op == "I":
            q_idx += length
        elif op == "D":
            t_idx += length
        i += 1

    return _coalesce_ops(normalized)


def _align_two_piece_affine_asm20(
    s_sub: str,
    r_sub: str,
) -> list[tuple[str, int]]:
    """
    Exact global two-piece affine DP matching `mappy` (`preset='asm20'`:
    `A=1, B=4, O1=6, E1=2, O2=26, E2=1`) on a bounded local edit cluster
    `(s_sub, r_sub)`.
    """
    n = len(s_sub)
    m = len(r_sub)
    if n == 0:
        return [("D", m)] if m > 0 else []
    if m == 0:
        return [("I", n)]

    neg = -1_000_000_000
    o1e1 = _ASM20_GAP_O1 + _ASM20_GAP_E1
    o2e2 = _ASM20_GAP_O2 + _ASM20_GAP_E2
    e1 = _ASM20_GAP_E1
    e2 = _ASM20_GAP_E2
    match_sc = _ASM20_MATCH
    mis_sc = -_ASM20_MISMATCH

    # States: 0=H(M/X), 1=I1(short ins), 2=I2(long ins), 3=D1(short del), 4=D2(long del)
    h_dp = [[neg] * (m + 1) for _ in range(n + 1)]
    i1_dp = [[neg] * (m + 1) for _ in range(n + 1)]
    i2_dp = [[neg] * (m + 1) for _ in range(n + 1)]
    d1_dp = [[neg] * (m + 1) for _ in range(n + 1)]
    d2_dp = [[neg] * (m + 1) for _ in range(n + 1)]

    bt_h = [bytearray(m + 1) for _ in range(n + 1)]
    bt_i1 = [bytearray(m + 1) for _ in range(n + 1)]
    bt_i2 = [bytearray(m + 1) for _ in range(n + 1)]
    bt_d1 = [bytearray(m + 1) for _ in range(n + 1)]
    bt_d2 = [bytearray(m + 1) for _ in range(n + 1)]

    h_dp[0][0] = 0
    for i in range(1, n + 1):
        i1_dp[i][0] = -(_ASM20_GAP_O1 + e1 * i)
        i2_dp[i][0] = -(_ASM20_GAP_O2 + e2 * i)
        bt_i1[i][0] = 0 if i == 1 else 1
        bt_i2[i][0] = 0 if i == 1 else 2
    for j in range(1, m + 1):
        d1_dp[0][j] = -(_ASM20_GAP_O1 + e1 * j)
        d2_dp[0][j] = -(_ASM20_GAP_O2 + e2 * j)
        bt_d1[0][j] = 0 if j == 1 else 3
        bt_d2[0][j] = 0 if j == 1 else 4

    for i in range(1, n + 1):
        si = s_sub[i - 1]
        h_i = h_dp[i]
        h_p = h_dp[i - 1]
        i1_i = i1_dp[i]
        i1_p = i1_dp[i - 1]
        i2_i = i2_dp[i]
        i2_p = i2_dp[i - 1]
        d1_i = d1_dp[i]
        d1_p = d1_dp[i - 1]
        d2_i = d2_dp[i]
        d2_p = d2_dp[i - 1]
        bh = bt_h[i]
        bi1 = bt_i1[i]
        bi2 = bt_i2[i]
        bd1 = bt_d1[i]
        bd2 = bt_d2[i]
        for j in range(1, m + 1):
            best = h_p[j - 1]
            src = 0
            v = i1_p[j - 1]
            if v > best:
                best = v
                src = 1
            v = i2_p[j - 1]
            if v > best:
                best = v
                src = 2
            v = d1_p[j - 1]
            if v > best:
                best = v
                src = 3
            v = d2_p[j - 1]
            if v > best:
                best = v
                src = 4
            h_i[j] = best + (match_sc if si == r_sub[j - 1] else mis_sc)
            bh[j] = src

            open_h = h_p[j]
            open_src = 0
            if d1_p[j] > open_h:
                open_h = d1_p[j]
                open_src = 3
            if d2_p[j] > open_h:
                open_h = d2_p[j]
                open_src = 4
            v_o1 = open_h - o1e1
            v_e1 = i1_p[j] - e1
            if v_e1 >= v_o1:
                i1_i[j] = v_e1
                bi1[j] = 1
            else:
                i1_i[j] = v_o1
                bi1[j] = open_src
            v_o2 = open_h - o2e2
            v_e2 = i2_p[j] - e2
            if v_e2 >= v_o2:
                i2_i[j] = v_e2
                bi2[j] = 2
            else:
                i2_i[j] = v_o2
                bi2[j] = open_src

            open_h = h_i[j - 1]
            open_src = 0
            if i1_i[j - 1] > open_h:
                open_h = i1_i[j - 1]
                open_src = 1
            if i2_i[j - 1] > open_h:
                open_h = i2_i[j - 1]
                open_src = 2
            v_o1 = open_h - o1e1
            v_e1 = d1_i[j - 1] - e1
            if v_e1 >= v_o1:
                d1_i[j] = v_e1
                bd1[j] = 3
            else:
                d1_i[j] = v_o1
                bd1[j] = open_src
            v_o2 = open_h - o2e2
            v_e2 = d2_i[j - 1] - e2
            if v_e2 >= v_o2:
                d2_i[j] = v_e2
                bd2[j] = 4
            else:
                d2_i[j] = v_o2
                bd2[j] = open_src

    best = h_dp[n][m]
    st = 0
    for cand_st, val in (
        (1, i1_dp[n][m]),
        (2, i2_dp[n][m]),
        (3, d1_dp[n][m]),
        (4, d2_dp[n][m]),
    ):
        if val >= best:
            best = val
            st = cand_st

    out_ops: list[tuple[str, int]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        if st == 0:
            out_ops.append(("M" if s_sub[i - 1] == r_sub[j - 1] else "X", 1))
            st = bt_h[i][j]
            i -= 1
            j -= 1
        elif st == 1:
            out_ops.append(("I", 1))
            st = bt_i1[i][j]
            i -= 1
        elif st == 2:
            out_ops.append(("I", 1))
            st = bt_i2[i][j]
            i -= 1
        elif st == 3:
            out_ops.append(("D", 1))
            st = bt_d1[i][j]
            j -= 1
        else:
            out_ops.append(("D", 1))
            st = bt_d2[i][j]
            j -= 1
    out_ops.reverse()
    return _coalesce_ops(out_ops)


def _realign_edlib_edit_clusters(
    ops: list[tuple[str, int]],
    seq_s: str,
    seq_r: str,
    anchor_m: int = _EDLIB_CLUSTER_ANCHOR_M,
) -> list[tuple[str, int]]:
    """
    Re-align non-trivial local edit clusters between exact `'M'` anchors
    (`>= anchor_m` bp) using exact `mappy asm20` two-piece affine DP.

    Clusters with `n_gaps == 0` (pure SNPs) or `n_gaps == 1 and n_x == 0` (already
    globally optimal single 0-mismatch indel) skip DP with zero overhead.
    Oversized clusters that still need DP are recursively sub-split at
    `anchor_m = 15 -> 8 -> 4` (max depth 2), and skinny indel clusters bounded by
    `_EDLIB_CLUSTER_DP_MAX_CELLS` (`25,600` cells) run DP directly.
    """
    if len(ops) < 2:
        return ops
    len_s = len(seq_s)
    len_r = len(seq_r)
    out: list[tuple[str, int]] = []
    q_pos = 0
    t_pos = 0
    i = 0
    n_ops = len(ops)
    while i < n_ops:
        op, length = ops[i]
        if op == "M" and length >= anchor_m:
            out.append((op, length))
            q_pos += length
            t_pos += length
            i += 1
            continue
        j = i
        q_end = q_pos
        t_end = t_pos
        n_gaps = 0
        n_x = 0
        while j < n_ops and not (ops[j][0] == "M" and ops[j][1] >= anchor_m):
            cop, clen = ops[j]
            if cop == "M":
                q_end += clen
                t_end += clen
            elif cop == "X":
                q_end += clen
                t_end += clen
                n_x += clen
            elif cop == "I":
                q_end += clen
                n_gaps += 1
            elif cop == "D":
                t_end += clen
                n_gaps += 1
            j += 1
        q_span = q_end - q_pos
        t_span = t_end - t_pos
        needs_dp = (n_gaps >= 2) or (n_gaps >= 1 and n_x > 0)
        fits_square = (
            q_span <= _EDLIB_CLUSTER_DP_MAX_SPAN
            and t_span <= _EDLIB_CLUSTER_DP_MAX_SPAN
        )
        fits_skinny = (
            min(q_span, t_span) <= _EDLIB_CLUSTER_DP_SKINNY_MIN_SPAN
            and max(q_span, t_span) <= EDLIB_MAX_LEN
            and q_span * t_span <= _EDLIB_CLUSTER_DP_MAX_CELLS
        )
        if (
            needs_dp
            and (fits_square or fits_skinny)
            and q_end <= len_s
            and t_end <= len_r
        ):
            out.extend(
                _align_two_piece_affine_asm20(
                    seq_s[q_pos:q_end], seq_r[t_pos:t_end]
                )
            )
        elif needs_dp and anchor_m > 4 and q_end <= len_s and t_end <= len_r:
            next_anchor = 8 if anchor_m > 8 else 4
            out.extend(
                _realign_edlib_edit_clusters(
                    ops[i:j],
                    seq_s[q_pos:q_end],
                    seq_r[t_pos:t_end],
                    anchor_m=next_anchor,
                )
            )
        else:
            out.extend(ops[i:j])
        q_pos = q_end
        t_pos = t_end
        i = j
    return _coalesce_ops(out)


def _extend_edlib_exact_matches_outward(
    ops: list[tuple[str, int]],
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    seq_s: str,
    seq_r: str,
) -> tuple[list[tuple[str, int]], int, int, int, int]:
    """Greedily re-extend exact `'M'` matches outward to enforce local Smith-Waterman maximality."""
    if not ops:
        return ops, q_start, q_end, t_start, t_end
    len_s = len(seq_s)
    len_r = len(seq_r)
    ext_left = 0
    while (
        q_start > 0
        and t_start > 0
        and seq_s[q_start - 1] == seq_r[t_start - 1]
    ):
        q_start -= 1
        t_start -= 1
        ext_left += 1
    if ext_left > 0:
        if ops[0][0] == "M":
            ops[0] = ("M", ops[0][1] + ext_left)
        else:
            ops.insert(0, ("M", ext_left))

    ext_right = 0
    while (
        q_end < len_s
        and t_end < len_r
        and seq_s[q_end] == seq_r[t_end]
    ):
        q_end += 1
        t_end += 1
        ext_right += 1
    if ext_right > 0:
        if ops[-1][0] == "M":
            ops[-1] = ("M", ops[-1][1] + ext_right)
        else:
            ops.append(("M", ext_right))
    return ops, q_start, q_end, t_start, t_end


def _trim_edlib_terminal_overhangs(
    ops: list[tuple[str, int]],
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    seq_s: str | None = None,
    seq_r: str | None = None,
) -> tuple[list[tuple[str, int]], int, int, int, int]:
    """
    Trim non-positive-scoring terminal prefixes/suffixes of `edlib` `mode='HW'`
    `ops` under `mappy asm20` local affine scoring (`M: +1`, `X: -4`,
    `I/D: -min(6 + 2L, 26 + L)`), converting force-aligned query overhangs into
    soft-clipped `(q_start, q_end)` and accurate target core bounds `(t_start, t_end)`.

    When `seq_s` and `seq_r` (matching the `(q_start, q_end)` / `(t_start, t_end)`
    coordinate frame) are provided, exact `'M'` matches are greedily re-extended
    outward after trimming to preserve local Smith-Waterman maximality.
    """
    if not ops:
        return [], q_start, q_start, t_start, t_start

    def _op_score(op: str, length: int) -> int:
        if op == "M":
            return _ASM20_MATCH * length
        if op == "X":
            return -_ASM20_MISMATCH * length
        return -_asm20_gap_penalty(length)

    cur = 0
    min_s = 0
    cut_left = 0
    trim_q_left = 0
    trim_t_left = 0
    q_acc = 0
    t_acc = 0
    m_acc = 0
    x_acc = 0
    d_acc = 0
    i_open_acc = 0
    for idx, (op, length) in enumerate(ops):
        if op == "M" and length >= _EDLIB_CLUSTER_ANCHOR_M:
            break
        cur += _op_score(op, length)
        if op == "M":
            q_acc += length
            t_acc += length
            m_acc += length
        elif op == "X":
            q_acc += length
            t_acc += length
            x_acc += length
        elif op == "I":
            q_acc += length
            i_open_acc += 1
        elif op == "D":
            t_acc += length
            d_acc += length
        if cur - min_s >= _EDLIB_CLUSTER_ANCHOR_M:
            break
        if cur <= min_s and not (
            cut_left == 0
            and x_acc == 0
            and d_acc == 0
            and i_open_acc <= 1
            and m_acc >= _EDLIB_MIN_CLEAN_INS_ARM
        ):
            min_s = cur
            cut_left = idx + 1
            trim_q_left = q_acc
            trim_t_left = t_acc
    if cut_left >= len(ops):
        return [], q_start, q_start, t_start, t_start
    ops = ops[cut_left:]
    q_start += trim_q_left
    t_start += trim_t_left

    cur = 0
    min_s = 0
    n_ops = len(ops)
    cut_right = n_ops
    trim_q_right = 0
    trim_t_right = 0
    q_acc = 0
    t_acc = 0
    m_acc = 0
    x_acc = 0
    d_acc = 0
    i_open_acc = 0
    for idx in range(n_ops - 1, -1, -1):
        op, length = ops[idx]
        if op == "M" and length >= _EDLIB_CLUSTER_ANCHOR_M:
            break
        cur += _op_score(op, length)
        if op == "M":
            q_acc += length
            t_acc += length
            m_acc += length
        elif op == "X":
            q_acc += length
            t_acc += length
            x_acc += length
        elif op == "I":
            q_acc += length
            i_open_acc += 1
        elif op == "D":
            t_acc += length
            d_acc += length
        if cur - min_s >= _EDLIB_CLUSTER_ANCHOR_M:
            break
        if cur <= min_s and not (
            cut_right == n_ops
            and x_acc == 0
            and d_acc == 0
            and i_open_acc <= 1
            and m_acc >= _EDLIB_MIN_CLEAN_INS_ARM
        ):
            min_s = cur
            cut_right = idx
            trim_q_right = q_acc
            trim_t_right = t_acc
    if cut_right <= 0:
        return [], q_start, q_start, t_start, t_start
    ops = ops[:cut_right]
    q_end -= trim_q_right
    t_end -= trim_t_right

    if ops and seq_s is not None and seq_r is not None:
        ops, q_start, q_end, t_start, t_end = (
            _extend_edlib_exact_matches_outward(
                ops, q_start, q_end, t_start, t_end, seq_s, seq_r
            )
        )
    return ops, q_start, q_end, t_start, t_end


def parse_edlib_cigar_to_ops(
    cigar_str: str,
    seq_s: str | None = None,
    seq_r: str | None = None,
) -> list[tuple[str, int]]:
    """
    Parse an `edlib` extended CIGAR string (`=`, `X`, `I`, `D`) into coalesced
    `('M'|'X'|'I'|'D', length)` operations along `seq_s`.

    Because `edlib` uses unit Levenshtein edit distance without an affine gap-open
    penalty, indels (especially near 1-3 SNPs) or asymmetric replacement blocks can
    be shattered into many tiny `'I'`/`'D'` runs separated by 1-2 bp `'='` matches.
    When `seq_s` and `seq_r` are provided, Pass 1 performs lossless exact 0-mismatch
    single-gap coalescing and Pass 2 re-aligns remaining multi-gap/mixed local edit
    clusters with exact `mappy` `asm20` two-piece affine DP, achieving parity with
    the `> 1000 bp` `mappy` branch without fabricating mismatches on 0-SNP multi-indels.
    """
    raw_ops: list[tuple[str, int]] = []
    for m in _EDLIB_CIGAR_RE.finditer(cigar_str):
        length = int(m.group(1))
        code = m.group(2)
        op = "M" if code == "=" else code
        raw_ops.append((op, length))
    ops = _coalesce_ops(raw_ops)
    if seq_s is None or seq_r is None or len(ops) < 2:
        return ops

    s_up = seq_s.upper()
    r_up = seq_r.upper()
    ops = _collapse_edlib_gap_spans(ops, s_up, r_up)
    ops = _realign_edlib_edit_clusters(ops, s_up, r_up)
    return ops


def _realign_and_trim_edlib_slice(
    query: str,
    target_sub: str,
    q_offset: int,
    t_offset: int,
) -> tuple[list[tuple[str, int]], int, int, int, int]:
    """
    Re-align `query` against `target_sub` globally (`edlib mode='NW'`), normalize
    edit clusters with exact `mappy asm20` two-piece affine DP, trim terminal
    overhangs + outward re-extend exact matches on `[0, len(query))` /
    `[0, len(target_sub))`, and shift coordinates by `(q_offset, t_offset)`.
    """
    if not query or not target_sub:
        return [], q_offset, q_offset, t_offset, t_offset
    res = edlib.align(query, target_sub, mode="NW", task="path")
    cigar = res.get("cigar")
    if cigar is None or res.get("editDistance", -1) < 0:
        return [], q_offset, q_offset, t_offset, t_offset
    ops = parse_edlib_cigar_to_ops(cigar, seq_s=query, seq_r=target_sub)
    ops, qs, qe, ts, te = _trim_edlib_terminal_overhangs(
        ops, 0, len(query), 0, len(target_sub), seq_s=query, seq_r=target_sub
    )
    if not ops or qs >= qe or ts >= te:
        return [], q_offset, q_offset, t_offset, t_offset
    return ops, q_offset + qs, q_offset + qe, t_offset + ts, t_offset + te


def _recover_edlib_abandoned_arms(
    s: str,
    r: str,
    ops: list[tuple[str, int]],
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    min_sv_len: int,
) -> tuple[list[tuple[str, int]], int, int, int, int, bool, bool]:
    """
    Stage 4 bounded short-arm recovery + Criterion 1 split-alignment guard for
    the `<= 1 kb` `edlib` branch.

    Unit-Levenshtein `edlib mode='HW'` abandons a homologous flanking arm when
    `~0.42 * arm < gap_len` on `< min_sv_len` internal deletions or flanked
    insertions. Searches bounded target windows on the left and right for
    DEL-bridged (Geometries A1/A2/A3) and INS-bridged (Geometry B) arms, bridges
    at most 4 deduplicated endpoints per side via `_realign_and_trim_edlib_slice`,
    and guards against unrecovered `>= 19 bp` split alignments across `>= min_sv_len`
    target gaps.

    Returns `(ops, q_start, q_end, t_start, t_end, arm_recovered, split_detected)`.
    """
    len_s = len(s)
    len_r = len(r)
    if not ops or (q_end - q_start) < min(6, len_s):
        return ops, q_start, q_end, t_start, t_end, False, False

    def _ops_stats(
        cur_ops: Sequence[tuple[str, int]],
    ) -> tuple[int, int, int, int]:
        m_cnt = 0
        i_bp = 0
        d_bp = 0
        max_d = 0
        for op, ln in cur_ops:
            if op == "M":
                m_cnt += ln
            elif op == "I":
                i_bp += ln
            elif op == "D":
                d_bp += ln
                if ln > max_d:
                    max_d = ln
        return m_cnt, i_bp, d_bp, max_d

    def _mx_in_q_range(
        cur_ops: Sequence[tuple[str, int]],
        ops_qs: int,
        rq_lo: int,
        rq_hi: int,
    ) -> tuple[int, int]:
        rec_m = 0
        rec_x = 0
        qp = ops_qs
        for op, ln in cur_ops:
            if op == "M" or op == "X":
                ov = min(qp + ln, rq_hi) - max(qp, rq_lo)
                if ov > 0:
                    if op == "M":
                        rec_m += ov
                    else:
                        rec_x += ov
                qp += ln
            elif op == "I":
                qp += ln
        return rec_m, rec_x

    def _pick_locs(
        locs: Sequence[tuple[int, int]],
        *,
        use_end: bool,
    ) -> list[tuple[int, int]]:
        picked: list[tuple[int, int]] = []
        for loc in locs:
            coord = loc[1] if use_end else loc[0]
            if not any(
                abs(coord - (p[1] if use_end else p[0])) <= 4 for p in picked
            ):
                picked.append(loc)
                if len(picked) >= 2:
                    break
        return picked

    arm_recovered = False

    # Left pass
    if q_start > 0 and t_start > 0:
        cur_m, cur_i_bp, cur_d_bp, _ = _ops_stats(ops)
        del_budget = (min_sv_len - 1) - cur_d_bp
        ins_budget = (min_sv_len - 1) - cur_i_bp
        if del_budget >= 0 and ins_budget >= 0:
            l_len = q_start
            q_left = s[:l_len]
            win_l = max(0, t_start - (l_len + del_budget + 8))
            r_left = r[win_l:t_start]
            cand_ts_list: list[int] = []

            def _add_ts(ts_val: int) -> None:
                if (
                    0 <= ts_val < t_start
                    and ts_val not in cand_ts_list
                    and len(cand_ts_list) < 4
                ):
                    cand_ts_list.append(ts_val)

            a1_locs: list[tuple[int, int]] = []
            if r_left:
                k_a1 = 0 if l_len < 15 else int(0.15 * l_len)
                res_a1 = edlib.align(
                    q_left, r_left, mode="HW", task="locations", k=k_a1
                )
                if 0 <= res_a1.get("editDistance", -1) <= k_a1:
                    a1_locs = list(res_a1.get("locations") or [])
                    for ls, _ in _pick_locs(a1_locs, use_end=False):
                        _add_ts(win_l + ls)

            if not a1_locs and r_left and l_len > max(ins_budget, 16):
                inner_len = max(16, l_len - ins_budget)
                k_a2 = 0 if inner_len < 15 else int(0.15 * inner_len)
                res_a2 = edlib.align(
                    q_left[-inner_len:],
                    r_left,
                    mode="HW",
                    task="locations",
                    k=k_a2,
                )
                if 0 <= res_a2.get("editDistance", -1) <= k_a2:
                    for ls, _ in _pick_locs(
                        res_a2.get("locations") or [], use_end=False
                    ):
                        _add_ts(max(0, win_l + ls - (l_len - inner_len)))

            if not a1_locs and r_left and l_len >= 24 and del_budget > 0:
                p_len = min(l_len - 1, 24)
                k_a3 = int(0.15 * p_len)
                res_a3 = edlib.align(
                    q_left[:p_len], r_left, mode="HW", task="locations", k=k_a3
                )
                if 0 <= res_a3.get("editDistance", -1) <= k_a3:
                    for ls, _ in _pick_locs(
                        res_a3.get("locations") or [], use_end=False
                    ):
                        _add_ts(win_l + ls)

            min_k = max(_EDLIB_MIN_CLEAN_INS_ARM, l_len - ins_budget)
            max_k = min(t_start, l_len - 1)
            if min_k <= max_k:
                b_win_l = t_start - max_k
                b_win_r = min(t_end, t_start + 4)
                r_b = r[b_win_l:b_win_r]
                k_b = 0 if min_k < 10 else max(1, int(0.15 * min_k))
                res_b = edlib.align(
                    s[:min_k], r_b, mode="HW", task="locations", k=k_b
                )
                b_locs: list[tuple[int, int]] = (
                    list(res_b.get("locations") or [])
                    if 0 <= res_b.get("editDistance", -1) <= k_b
                    else []
                )
                if not b_locs and _EDLIB_MIN_CLEAN_INS_ARM < min_k <= 14:
                    res_b8 = edlib.align(
                        s[:_EDLIB_MIN_CLEAN_INS_ARM],
                        r_b,
                        mode="HW",
                        task="locations",
                        k=0,
                    )
                    if res_b8.get("editDistance", -1) == 0:
                        b_locs = list(res_b8.get("locations") or [])
                for ls, _ in _pick_locs(b_locs, use_end=False):
                    _add_ts(b_win_l + ls)

            best_left: tuple[
                int, list[tuple[str, int]], int, int, int, int
            ] | None = None
            for cand_ts in cand_ts_list:
                b_ops, b_qs, b_qe, b_ts, b_te = _realign_and_trim_edlib_slice(
                    s[:q_end], r[cand_ts:t_end], 0, cand_ts
                )
                if not b_ops or b_qs >= q_start:
                    continue
                b_m, b_i_bp, b_d_bp, b_max_d = _ops_stats(b_ops)
                if (
                    b_m <= cur_m
                    or b_max_d >= min_sv_len
                    or b_d_bp >= min_sv_len
                    or (b_qs + b_i_bp) >= min_sv_len
                ):
                    continue
                rec_m, rec_x = _mx_in_q_range(b_ops, b_qs, b_qs, q_start)
                if rec_m < _EDLIB_MIN_CLEAN_INS_ARM or rec_m * 100 < 85 * (
                    rec_m + rec_x
                ):
                    continue
                if best_left is None or b_m > best_left[0]:
                    best_left = (b_m, b_ops, b_qs, b_qe, b_ts, b_te)

            if best_left is not None:
                _, ops, q_start, q_end, t_start, t_end = best_left
                ops, q_start, q_end, t_start, t_end = (
                    _extend_edlib_exact_matches_outward(
                        ops, q_start, q_end, t_start, t_end, s, r
                    )
                )
                arm_recovered = True

    # Right pass (with updated budgets)
    if q_end < len_s and t_end < len_r:
        cur_m, cur_i_bp, cur_d_bp, _ = _ops_stats(ops)
        del_budget = (min_sv_len - 1) - cur_d_bp
        ins_budget = (min_sv_len - 1) - (q_start + cur_i_bp)
        if del_budget >= 0 and ins_budget >= 0:
            r_len = len_s - q_end
            q_right = s[q_end:]
            win_r = min(len_r, t_end + r_len + del_budget + 8)
            r_right = r[t_end:win_r]
            cand_te_list: list[int] = []

            def _add_te(te_val: int) -> None:
                if (
                    t_end < te_val <= len_r
                    and te_val not in cand_te_list
                    and len(cand_te_list) < 4
                ):
                    cand_te_list.append(te_val)

            a1_r_locs: list[tuple[int, int]] = []
            if r_right:
                k_a1 = 0 if r_len < 15 else int(0.15 * r_len)
                res_a1 = edlib.align(
                    q_right, r_right, mode="HW", task="locations", k=k_a1
                )
                if 0 <= res_a1.get("editDistance", -1) <= k_a1:
                    a1_r_locs = list(res_a1.get("locations") or [])
                    for _, le_incl in _pick_locs(a1_r_locs, use_end=True):
                        _add_te(t_end + le_incl + 1)

            if not a1_r_locs and r_right and r_len > max(ins_budget, 16):
                inner_len = max(16, r_len - ins_budget)
                k_a2 = 0 if inner_len < 15 else int(0.15 * inner_len)
                res_a2 = edlib.align(
                    q_right[:inner_len],
                    r_right,
                    mode="HW",
                    task="locations",
                    k=k_a2,
                )
                if 0 <= res_a2.get("editDistance", -1) <= k_a2:
                    for _, le_incl in _pick_locs(
                        res_a2.get("locations") or [], use_end=True
                    ):
                        _add_te(
                            min(
                                len_r,
                                t_end + le_incl + 1 + (r_len - inner_len),
                            )
                        )

            if not a1_r_locs and r_right and r_len >= 24 and del_budget > 0:
                p_len = min(r_len - 1, 24)
                k_a3 = int(0.15 * p_len)
                res_a3 = edlib.align(
                    q_right[-p_len:],
                    r_right,
                    mode="HW",
                    task="locations",
                    k=k_a3,
                )
                if 0 <= res_a3.get("editDistance", -1) <= k_a3:
                    for _, le_incl in _pick_locs(
                        res_a3.get("locations") or [], use_end=True
                    ):
                        _add_te(t_end + le_incl + 1)

            min_k = max(_EDLIB_MIN_CLEAN_INS_ARM, r_len - ins_budget)
            max_k = min(len_r - t_end, r_len - 1)
            if min_k <= max_k:
                b_win_l = max(t_start, t_end - 4)
                b_win_r = t_end + max_k
                r_b = r[b_win_l:b_win_r]
                k_b = 0 if min_k < 10 else max(1, int(0.15 * min_k))
                res_b = edlib.align(
                    s[len_s - min_k :],
                    r_b,
                    mode="HW",
                    task="locations",
                    k=k_b,
                )
                b_r_locs: list[tuple[int, int]] = (
                    list(res_b.get("locations") or [])
                    if 0 <= res_b.get("editDistance", -1) <= k_b
                    else []
                )
                if not b_r_locs and _EDLIB_MIN_CLEAN_INS_ARM < min_k <= 14:
                    res_b8 = edlib.align(
                        s[len_s - _EDLIB_MIN_CLEAN_INS_ARM :],
                        r_b,
                        mode="HW",
                        task="locations",
                        k=0,
                    )
                    if res_b8.get("editDistance", -1) == 0:
                        b_r_locs = list(res_b8.get("locations") or [])
                for _, le_incl in _pick_locs(b_r_locs, use_end=True):
                    _add_te(b_win_l + le_incl + 1)

            best_right: tuple[
                int, list[tuple[str, int]], int, int, int, int
            ] | None = None
            for cand_te in cand_te_list:
                b_ops, b_qs, b_qe, b_ts, b_te = _realign_and_trim_edlib_slice(
                    s[q_start:], r[t_start:cand_te], q_start, t_start
                )
                if not b_ops or b_qe <= q_end:
                    continue
                b_m, b_i_bp, b_d_bp, b_max_d = _ops_stats(b_ops)
                if (
                    b_m <= cur_m
                    or b_max_d >= min_sv_len
                    or b_d_bp >= min_sv_len
                    or (b_qs + (len_s - b_qe) + b_i_bp) >= min_sv_len
                ):
                    continue
                rec_m, rec_x = _mx_in_q_range(b_ops, b_qs, q_end, b_qe)
                if rec_m < _EDLIB_MIN_CLEAN_INS_ARM or rec_m * 100 < 85 * (
                    rec_m + rec_x
                ):
                    continue
                if best_right is None or b_m > best_right[0]:
                    best_right = (b_m, b_ops, b_qs, b_qe, b_ts, b_te)

            if best_right is not None:
                _, ops, q_start, q_end, t_start, t_end = best_right
                ops, q_start, q_end, t_start, t_end = (
                    _extend_edlib_exact_matches_outward(
                        ops, q_start, q_end, t_start, t_end, s, r
                    )
                )
                arm_recovered = True

    # Split-alignment guard (Criterion 1): reject unrecovered >= 19 bp outer
    # arm(s) (allowing up to 3 bp exact boundary-collision overlap with ops,
    # inner-core probing for outer novel overhangs, and complexity-gated
    # outer/mid 19 bp probing for inner junctional insertions `arm+ins` /
    # `ov+arm+ins`) whose cumulative deletion gap(s) plus existing `cur_d_bp`
    # reach >= min_sv_len.
    _, _, cur_d_bp, _ = _ops_stats(ops)
    if cur_d_bp < min_sv_len:
        split_del_bp = cur_d_bp
        right_rem = len_s - q_end
        col_min = _EDLIB_MIN_ALIGNED_CORE - 3
        k_in = max(1, int(0.15 * _EDLIB_MIN_ALIGNED_CORE))

        def _bridge_inward_micro_indel(
            is_left: bool,
            v_qs: int,
            v_qe: int,
            v_ts: int,
            v_te: int,
            q_lo: int,
            q_hi: int,
            t_lo: int,
            t_hi: int,
        ) -> tuple[int, int]:
            # Bridge a 1-2 bp micro-indel inside the inner portion of a
            # homologous outer arm (requiring >= 5 exact bases across the
            # shift) so a valid < min_sv_len gap (e.g. 47D+ins15+arm30_1bpDel)
            # is never misattributed as deleted target bases.
            if is_left:
                for dq, dt in ((0, 1), (1, 0), (0, 2), (2, 0)):
                    nq = v_qe + dq
                    nt = v_te + dt
                    if nq + 5 <= q_hi and nt + 5 <= t_hi:
                        if s[nq : nq + 5] == r[nt : nt + 5]:
                            ex = 5
                            max_ex = min(q_hi - nq, t_hi - nt)
                            while ex < max_ex and s[nq + ex] == r[nt + ex]:
                                ex += 1
                            return nt + ex, dt
                return v_te, 0
            for dq, dt in ((0, 1), (1, 0), (0, 2), (2, 0)):
                nq = v_qs - dq
                nt = v_ts - dt
                if nq - 5 >= q_lo and nt - 5 >= t_lo:
                    if s[nq - 5 : nq] == r[nt - 5 : nt]:
                        ex = 5
                        max_ex = min(nq - q_lo, nt - t_lo)
                        while (
                            ex < max_ex
                            and s[nq - 1 - ex] == r[nt - 1 - ex]
                        ):
                            ex += 1
                        return nt - ex, dt
            return v_ts, 0

        def _probe_verified_19bp_gap(
            is_left: bool,
            p_qs: int,
            p_qe: int,
            q_lo: int,
            q_hi: int,
            t_lo: int,
            t_hi: int,
        ) -> int | None:
            q_probe = s[p_qs:p_qe]
            if (
                len(
                    dict.fromkeys(
                        q_probe[i : i + 3]
                        for i in range(_EDLIB_MIN_ALIGNED_CORE - 2)
                    )
                )
                <= 6
            ):
                return None
            at_cnt = q_probe.count("A") + q_probe.count("T")
            gc_cnt = q_probe.count("G") + q_probe.count("C")
            k_out = (
                0
                if (
                    at_cnt + gc_cnt < _EDLIB_MIN_ALIGNED_CORE
                    or max(at_cnt, gc_cnt) * 100
                    > 68 * _EDLIB_MIN_ALIGNED_CORE
                )
                else 1
            )
            res_out = edlib.align(
                q_probe,
                r[t_lo:t_hi],
                mode="HW",
                task="locations",
                k=k_out,
            )
            if not (0 <= res_out.get("editDistance", -1) <= k_out):
                return None
            raw_locs = res_out.get("locations") or []
            if not raw_locs:
                return None
            # Deduplicate up to 8 closest hits to the core boundary first so
            # tandem-dup flanks (P1-G) always inspect gap=0 copies before far hits.
            picked_out: list[tuple[int, int]] = []
            scan_locs = reversed(raw_locs) if is_left else raw_locs
            for loc in scan_locs:
                coord = loc[1] if is_left else loc[0]
                if not any(
                    abs(coord - (p[1] if is_left else p[0])) <= 4
                    for p in picked_out
                ):
                    picked_out.append(loc)
                    if len(picked_out) >= 8:
                        break
            min_gap: int | None = None
            for ls, le_incl in picked_out:
                abs_ts = t_lo + ls
                abs_te = t_lo + le_incl + 1
                # Greedy +1/-4 diagonal extension leftward and rightward so
                # the true arm boundary is recovered (even across a 1-SNP
                # inner arm) without inflating gap by q_ins or random bases.
                max_l = min(p_qs - q_lo, abs_ts - t_lo)
                sc = 0
                best_sc = 0
                best_l = 0
                for ex in range(1, max_l + 1):
                    if s[p_qs - ex] == r[abs_ts - ex]:
                        sc += 1
                        if sc > best_sc:
                            best_sc = sc
                            best_l = ex
                    else:
                        sc -= 4
                        if sc < best_sc - 12:
                            break
                max_r = min(q_hi - p_qe, t_hi - abs_te)
                sc = 0
                best_sc = 0
                best_r = 0
                for ex in range(1, max_r + 1):
                    if s[p_qe + ex - 1] == r[abs_te + ex - 1]:
                        sc += 1
                        if sc > best_sc:
                            best_sc = sc
                            best_r = ex
                    else:
                        sc -= 4
                        if sc < best_sc - 12:
                            break
                a_qs = p_qs - best_l
                a_qe = p_qe + best_r
                a_ts = abs_ts - best_l
                a_te = abs_te + best_r
                v_ops, v_qs, v_qe, v_ts, v_te = _realign_and_trim_edlib_slice(
                    s[a_qs:a_qe], r[a_ts:a_te], a_qs, a_ts
                )
                if (
                    not v_ops
                    or (v_qe - v_qs) < _EDLIB_MIN_ALIGNED_CORE
                    or _asm20_ops_score(v_ops) <= 0
                ):
                    continue
                _, _, v_d_bp, _ = _ops_stats(v_ops)
                cand_gap = (
                    (t_hi - v_te) + v_d_bp
                    if is_left
                    else (v_ts - t_lo) + v_d_bp
                )
                if cand_gap >= max(1, min_sv_len - split_del_bp):
                    new_t, add_d = _bridge_inward_micro_indel(
                        is_left, v_qs, v_qe, v_ts, v_te, q_lo, q_hi, t_lo, t_hi
                    )
                    if add_d > 0:
                        br_gap = (
                            (t_hi - new_t) + v_d_bp + add_d
                            if is_left
                            else (new_t - t_lo) + v_d_bp + add_d
                        )
                        if br_gap < cand_gap:
                            cand_gap = br_gap
                if cand_gap <= 0:
                    return 0
                if min_gap is None or cand_gap < min_gap:
                    min_gap = cand_gap
            return min_gap

        def _probe_pigeonhole_10mer_gap(
            is_left: bool,
            q_lo: int,
            q_hi: int,
            t_lo: int,
            t_hi: int,
        ) -> int | None:
            # Pigeonhole exact 10-mer anchors at step 10 inside [q_lo, q_hi):
            # every exact >= 19 bp homologous arm at ANY asymmetric offset
            # (e.g. ins18 + arm19 + ov6) contains at least one such 10-mer.
            rem_len = q_hi - q_lo
            if rem_len < 24 or t_hi - t_lo < _EDLIB_MIN_ALIGNED_CORE:
                return None
            t_slice = r[t_lo:t_hi]
            t_sl_len = t_hi - t_lo
            min_gap: int | None = None
            for off in range(10, rem_len - 9, 10):
                p_qs = q_lo + off
                p_qe = p_qs + 10
                km = s[p_qs:p_qe]
                if len(dict.fromkeys(km[i : i + 3] for i in range(8))) <= 4:
                    continue
                hit_starts: list[int] = []
                if is_left:
                    pos = t_slice.rfind(km)
                    while pos != -1 and len(hit_starts) < 8:
                        hit_starts.append(pos)
                        pos = t_slice.rfind(km, 0, pos) if pos > 0 else -1
                else:
                    pos = t_slice.find(km)
                    while pos != -1 and len(hit_starts) < 8:
                        hit_starts.append(pos)
                        pos = (
                            t_slice.find(km, pos + 1)
                            if pos + 1 < t_sl_len
                            else -1
                        )
                for hs in hit_starts:
                    abs_ts = t_lo + hs
                    abs_te = abs_ts + 10
                    ex_l = 0
                    max_l = min(p_qs - q_lo, abs_ts - t_lo)
                    while (
                        ex_l < max_l
                        and s[p_qs - 1 - ex_l] == r[abs_ts - 1 - ex_l]
                    ):
                        ex_l += 1
                    ex_r = 0
                    max_r = min(q_hi - p_qe, t_hi - abs_te)
                    while (
                        ex_r < max_r
                        and s[p_qe + ex_r] == r[abs_te + ex_r]
                    ):
                        ex_r += 1
                    arm_len_ex = 10 + ex_l + ex_r
                    if arm_len_ex < _EDLIB_MIN_ALIGNED_CORE:
                        continue
                    a_qs = p_qs - ex_l
                    a_qe = p_qe + ex_r
                    a_ts = abs_ts - ex_l
                    a_te = abs_te + ex_r
                    arm_sub = s[a_qs:a_qe]
                    if (
                        len(
                            dict.fromkeys(
                                arm_sub[i : i + 3]
                                for i in range(arm_len_ex - 2)
                            )
                        )
                        <= 6
                    ):
                        continue
                    # Strict AT/GC skew guard (prevents random exact 19-mer
                    # matches in >= 83% AT/GC flanks while preserving all real
                    # complex homologous arms).
                    at_c = arm_sub.count("A") + arm_sub.count("T")
                    gc_c = arm_sub.count("G") + arm_sub.count("C")
                    if (
                        at_c + gc_c < arm_len_ex
                        or max(at_c, gc_c) * 100 > 84 * arm_len_ex
                    ):
                        continue
                    cand_gap = (t_hi - a_te) if is_left else (a_ts - t_lo)
                    if cand_gap >= max(1, min_sv_len - split_del_bp):
                        new_t, add_d = _bridge_inward_micro_indel(
                            is_left,
                            a_qs,
                            a_qe,
                            a_ts,
                            a_te,
                            q_lo,
                            q_hi,
                            t_lo,
                            t_hi,
                        )
                        if add_d > 0:
                            br_gap = (
                                (t_hi - new_t) + add_d
                                if is_left
                                else (new_t - t_lo) + add_d
                            )
                            if br_gap < cand_gap:
                                cand_gap = br_gap
                    if cand_gap <= 0:
                        return 0
                    if min_gap is None or cand_gap < min_gap:
                        min_gap = cand_gap
            return min_gap

        has_right_cand = (
            right_rem >= col_min and (len_r - t_end) >= 1 + col_min
        )
        min_left_gap = (
            1 if has_right_cand else max(1, min_sv_len - split_del_bp)
        )
        if q_start >= col_min and t_start >= min_left_gap + col_min:
            need_col_l = max(0, _EDLIB_MIN_ALIGNED_CORE - q_start)
            if need_col_l == 0 or (
                q_start + need_col_l <= q_end
                and ops[0][0] == "M"
                and ops[0][1] >= need_col_l
            ):
                k_sp = max(1, int(0.15 * q_start))
                sp_res = edlib.align(
                    s[:q_start],
                    r[:t_start],
                    mode="HW",
                    task="locations",
                    k=k_sp,
                )
                sp_locs: list[tuple[int, int]] = []
                q_arm_len = q_start
                if 0 <= sp_res.get("editDistance", -1) <= k_sp:
                    sp_locs = list(sp_res.get("locations") or [])
                    if need_col_l > 0:
                        col_seq = s[q_start : q_start + need_col_l]
                        sp_locs = [
                            (ls, le_incl)
                            for ls, le_incl in sp_locs
                            if le_incl + 1 + need_col_l <= len_r
                            and r[le_incl + 1 : le_incl + 1 + need_col_l]
                            == col_seq
                        ]
                if not sp_locs and q_start > _EDLIB_MIN_ALIGNED_CORE:
                    sp_in = edlib.align(
                        s[q_start - _EDLIB_MIN_ALIGNED_CORE : q_start],
                        r[:t_start],
                        mode="HW",
                        task="locations",
                        k=k_in,
                    )
                    if 0 <= sp_in.get("editDistance", -1) <= k_in:
                        sp_locs = list(sp_in.get("locations") or [])
                        q_arm_len = _EDLIB_MIN_ALIGNED_CORE
                if (
                    not sp_locs
                    and q_start > col_min
                    and ops[0][0] == "M"
                    and ops[0][1] >= 1
                    and q_end - q_start >= 1
                ):
                    sp_col = edlib.align(
                        s[q_start - col_min : q_start],
                        r[:t_start],
                        mode="HW",
                        task="locations",
                        k=k_in,
                    )
                    if 0 <= sp_col.get("editDistance", -1) <= k_in:
                        max_in = min(3, ops[0][1], q_end - q_start)
                        cand_col: list[tuple[int, int]] = []
                        for ls, le_incl in sp_col.get("locations") or []:
                            ex_in = 0
                            while (
                                ex_in < max_in
                                and le_incl + 1 + ex_in < len_r
                                and r[le_incl + 1 + ex_in]
                                == s[q_start + ex_in]
                            ):
                                ex_in += 1
                            if ex_in < 1:
                                continue
                            need_out = 3 - ex_in
                            ex_out = 0
                            while (
                                ex_out < need_out
                                and ls - 1 - ex_out >= 0
                                and q_start - col_min - 1 - ex_out >= 0
                                and r[ls - 1 - ex_out]
                                == s[q_start - col_min - 1 - ex_out]
                            ):
                                ex_out += 1
                            if ex_in + ex_out >= 3:
                                cand_col.append((ls, le_incl))
                        if cand_col:
                            sp_locs = cand_col
                            q_arm_len = col_min
                left_gap: int | None = None
                if sp_locs:
                    left_gap = min(
                        max(t_start - (le_incl + 1), t_start - ls - q_arm_len)
                        for ls, le_incl in sp_locs
                    )
                elif q_start > _EDLIB_MIN_ALIGNED_CORE:
                    left_gap = _probe_verified_19bp_gap(
                        True,
                        0,
                        _EDLIB_MIN_ALIGNED_CORE,
                        0,
                        q_start,
                        0,
                        t_start,
                    )
                    if left_gap is None and q_start >= 30:
                        mid_qs = (q_start - _EDLIB_MIN_ALIGNED_CORE) // 2
                        left_gap = _probe_verified_19bp_gap(
                            True,
                            mid_qs,
                            mid_qs + _EDLIB_MIN_ALIGNED_CORE,
                            0,
                            q_start,
                            0,
                            t_start,
                        )
                    if left_gap is None and q_start >= 24:
                        left_gap = _probe_pigeonhole_10mer_gap(
                            True, 0, q_start, 0, t_start
                        )
                if left_gap is not None and left_gap > 0:
                    split_del_bp += left_gap
                    if split_del_bp >= min_sv_len:
                        return (
                            ops,
                            q_start,
                            q_end,
                            t_start,
                            t_end,
                            arm_recovered,
                            True,
                        )

        rem_del_r = max(1, min_sv_len - split_del_bp)
        if right_rem >= col_min and (len_r - t_end) >= rem_del_r + col_min:
            need_col_r = max(0, _EDLIB_MIN_ALIGNED_CORE - right_rem)
            if need_col_r == 0 or (
                q_end - need_col_r >= q_start
                and ops[-1][0] == "M"
                and ops[-1][1] >= need_col_r
            ):
                k_sp = max(1, int(0.15 * right_rem))
                sp_res = edlib.align(
                    s[q_end:],
                    r[t_end:],
                    mode="HW",
                    task="locations",
                    k=k_sp,
                )
                sp_locs = []
                q_arm_len = right_rem
                if 0 <= sp_res.get("editDistance", -1) <= k_sp:
                    sp_locs = list(sp_res.get("locations") or [])
                    if need_col_r > 0:
                        col_seq = s[q_end - need_col_r : q_end]
                        sp_locs = [
                            (ls, le_incl)
                            for ls, le_incl in sp_locs
                            if t_end + ls >= need_col_r
                            and r[t_end + ls - need_col_r : t_end + ls]
                            == col_seq
                        ]
                if not sp_locs and right_rem > _EDLIB_MIN_ALIGNED_CORE:
                    sp_in = edlib.align(
                        s[q_end : q_end + _EDLIB_MIN_ALIGNED_CORE],
                        r[t_end:],
                        mode="HW",
                        task="locations",
                        k=k_in,
                    )
                    if 0 <= sp_in.get("editDistance", -1) <= k_in:
                        sp_locs = list(sp_in.get("locations") or [])
                        q_arm_len = _EDLIB_MIN_ALIGNED_CORE
                if (
                    not sp_locs
                    and right_rem > col_min
                    and ops[-1][0] == "M"
                    and ops[-1][1] >= 1
                    and q_end - q_start >= 1
                ):
                    sp_col = edlib.align(
                        s[q_end : q_end + col_min],
                        r[t_end:],
                        mode="HW",
                        task="locations",
                        k=k_in,
                    )
                    if 0 <= sp_col.get("editDistance", -1) <= k_in:
                        max_in = min(3, ops[-1][1], q_end - q_start)
                        cand_col = []
                        for ls, le_incl in sp_col.get("locations") or []:
                            ex_in = 0
                            while (
                                ex_in < max_in
                                and t_end + ls - 1 - ex_in >= 0
                                and r[t_end + ls - 1 - ex_in]
                                == s[q_end - 1 - ex_in]
                            ):
                                ex_in += 1
                            if ex_in < 1:
                                continue
                            need_out = 3 - ex_in
                            ex_out = 0
                            while (
                                ex_out < need_out
                                and t_end + le_incl + 1 + ex_out < len_r
                                and q_end + col_min + ex_out < len_s
                                and r[t_end + le_incl + 1 + ex_out]
                                == s[q_end + col_min + ex_out]
                            ):
                                ex_out += 1
                            if ex_in + ex_out >= 3:
                                cand_col.append((ls, le_incl))
                        if cand_col:
                            sp_locs = cand_col
                            q_arm_len = col_min
                right_gap: int | None = None
                if sp_locs:
                    right_gap = min(
                        max(ls, (le_incl + 1) - q_arm_len)
                        for ls, le_incl in sp_locs
                    )
                elif right_rem > _EDLIB_MIN_ALIGNED_CORE:
                    right_gap = _probe_verified_19bp_gap(
                        False,
                        len_s - _EDLIB_MIN_ALIGNED_CORE,
                        len_s,
                        q_end,
                        len_s,
                        t_end,
                        len_r,
                    )
                    if right_gap is None and right_rem >= 30:
                        mid_qs = (
                            q_end + (right_rem - _EDLIB_MIN_ALIGNED_CORE) // 2
                        )
                        right_gap = _probe_verified_19bp_gap(
                            False,
                            mid_qs,
                            mid_qs + _EDLIB_MIN_ALIGNED_CORE,
                            q_end,
                            len_s,
                            t_end,
                            len_r,
                        )
                    if right_gap is None and right_rem >= 24:
                        right_gap = _probe_pigeonhole_10mer_gap(
                            False, q_end, len_s, t_end, len_r
                        )
                if (
                    right_gap is not None
                    and right_gap > 0
                    and split_del_bp + right_gap >= min_sv_len
                ):
                    return (
                        ops,
                        q_start,
                        q_end,
                        t_start,
                        t_end,
                        arm_recovered,
                        True,
                    )

    return ops, q_start, q_end, t_start, t_end, arm_recovered, False


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


def _parse_cs_to_ops_and_local_dx(
    cs_str: str,
    strand: int,
    q_st: int,
    dx_lo: int,
    dx_hi: int,
) -> tuple[
    list[tuple[str, int]],
    dict[tuple[int, int], str],
    dict[int, tuple[str, str]],
]:
    """
    Single-pass `cs` parser producing:
      1. `ops`: forward-query-oriented `(op, length)` list (matching `parse_cs_to_query_ops`).
      2. `d_map`: forward-query-oriented explicit deleted sequences for `D` ops in `[dx_lo, dx_hi]`.
      3. `x_map`: forward-query-oriented `(ref_base, query_base)` for `X` ops in `[dx_lo, dx_hi)`.
    Avoids double regex scanning of `hit.cs` and avoids building dicts outside `[dx_lo, dx_hi]`.
    """
    raw_ops: list[tuple[str, int, str]] = []
    append_raw = raw_ops.append
    for m in _CS_TOKEN_RE.finditer(cs_str):
        li = m.lastindex
        if li == 1:
            append_raw(("M", int(m.group(1)), ""))
        elif li == 3:
            append_raw(("X", 1, m.group(3)))
        elif li == 4:
            append_raw(("I", len(m.group(4)), ""))
        elif li == 5:
            g5 = m.group(5)
            append_raw(("D", len(g5), g5))
        elif li == 2:
            append_raw(("M", len(m.group(2)), ""))
        elif li == 6:
            append_raw(("D", int(m.group(6)), ""))

    if strand == -1:
        raw_ops.reverse()

    ops: list[tuple[str, int]] = []
    d_map: dict[tuple[int, int], str] = {}
    x_map: dict[int, tuple[str, str]] = {}
    q_cur = q_st
    i = 0
    n_tok = len(raw_ops)
    while i < n_tok:
        op, length, tok_s = raw_ops[i]
        if op == "M" or op == "I":
            if ops and ops[-1][0] == op:
                ops[-1] = (op, ops[-1][1] + length)
            else:
                ops.append((op, length))
            q_cur += length
            i += 1
        elif op == "X":
            if ops and ops[-1][0] == "X":
                ops[-1] = ("X", ops[-1][1] + 1)
            else:
                ops.append(("X", 1))
            if dx_lo <= q_cur < dx_hi and len(tok_s) == 2:
                tu = tok_s.upper()
                r_b, q_b = (
                    (revcomp(tu[0]), revcomp(tu[1]))
                    if strand == -1
                    else (tu[0], tu[1])
                )
                if r_b != "N" and q_b != "N":
                    x_map[q_cur] = (r_b, q_b)
            q_cur += 1
            i += 1
        else:  # op == "D"
            tot_len = length
            in_dx = dx_lo <= q_cur <= dx_hi
            parts: list[str] | None = None
            if in_dx and tok_s:
                tu = tok_s.upper()
                parts = [revcomp(tu) if strand == -1 else tu]
            j = i + 1
            while j < n_tok and raw_ops[j][0] == "D":
                tot_len += raw_ops[j][1]
                if parts is not None:
                    dj = raw_ops[j][2]
                    if dj:
                        du = dj.upper()
                        parts.append(revcomp(du) if strand == -1 else du)
                    else:
                        parts = None
                j += 1
            if ops and ops[-1][0] == "D":
                ops[-1] = ("D", ops[-1][1] + tot_len)
            else:
                ops.append(("D", tot_len))
            if parts is not None:
                full_d = "".join(parts)
                if len(full_d) == tot_len and "N" not in full_d:
                    d_map[(q_cur, tot_len)] = full_d
            i = j

    return ops, d_map, x_map


def _extract_cs_dx_info(
    cs_str: str,
    strand: int,
    q_st: int,
) -> tuple[dict[tuple[int, int], str], dict[int, tuple[str, str]]]:
    """
    Extract forward-query-oriented `(d_map, x_map)` across the full `cs` string.
    """
    _, d_map, x_map = _parse_cs_to_ops_and_local_dx(
        cs_str, strand, q_st, -(10**18), 10**18
    )
    return d_map, x_map


def _extract_d_seqs_from_cs(
    cs_str: str,
    strand: int,
    q_st: int,
) -> dict[tuple[int, int], str]:
    """
    Extract forward-query-oriented deleted reference sequences for explicit `-<seq>`
    `D` ops (excluding `'N'` separators and `~` intron gaps), keyed by `(q_pos, d_len)`.
    """
    return _extract_cs_dx_info(cs_str, strand, q_st)[0]


def hit_covers_subinterval_mappy(
    hit: Any,
    p_s: int,
    p_e: int,
    min_identity: float = REF_SUB_MIN_IDENTITY,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
    min_hit_len: int = 0,
    query_seq: str | None = None,
    allow_softclip: bool = True,
    blocked_t_pieces: set[int] | None = None,
    ins_block_t_pieces_out: set[int] | None = None,
) -> bool:
    """
    Return True iff `hit` (direct or bridged across `t_starts`/`t_ends` target pieces)
    contains a single co-linear segment (within one target piece, without internal
    `D >= min_hit_len` structural gaps when `min_hit_len > 0`) of query span
    `>= min_hit_len` that covers `[p_s, p_e)` under exact, strand- & mirror-invariant
    fixed-gap-structure equivalence-class semantics:
      1. Unaligned bases are counted in denominators (`denom = p_len + d_p`,
         `w_denom = w_size + w_d`) alongside `(M + X) * 10 >= 9 * L`. When
         `allow_softclip=False`, segments not spanning `[p_s, p_e)` (`seg_q_st > p_s`
         or `seg_q_en < p_e`) are rejected.
      2. When `p_len < max(50, min_hit_len)` and `query_seq` is supplied, local
         `(Q_loc, T_loc)` reconstruction (including pure-`M`-connected outer tandem
         `I` injection for long asymmetric STRs) + `O(K * n)` Forward/Backward
         diagonal DP computes the exact, initial-position-independent, mirror-invariant
         marginal optimal placement sets `S*_j` for all `K` indels:
         - Rule D: Deletions `D` may occupy any `q in S*_j` (exact `M`/`X` counts).
         - Rule I: Inescapable `I` (`min_ov > 0`) and conserved tandem/quasi-tandem
           `I` (`has_tandem`) record the target piece in `ins_block_t_pieces_out`
           when a spanning segment returns `False`; rejects if `L >= min_hit_len`
           (§3A.1 `protected_ins`), else pulls canonically to `S_max = argmax ov`
           (conservative symmetric `ALL` on disconnected endpoints); exitable `I`
           (`min_ov == 0`) with `s_in` non-empty conserves ALL full-unit tandems
           and quasi-tandems in `w_size` (`W_allowed = s_in`), except the provably
           non-tandem zero-double-drop `L == 1` `X`-crossing `0`-mismatch-in-`P`
           exit (`BUG-R10-4`), and otherwise exits cleanly (`S_out`).
      3. Structural `D >= min_hit_len` splits symmetrically over `left_q_lim < q_p <
         right_q_lim`, forbids un-exited structural `D` strictly inside `(p_s, p_e)`,
         and allows bounded zero-penalty microhomology extension with deficit-repaying
         lookahead while trimming trailing/leading `X` pads.
    """
    p_len = p_e - p_s
    q_st = int(hit.q_st)
    q_en = int(hit.q_en)
    if (
        p_len <= 0
        or not getattr(hit, "cs", None)
        or q_en <= p_s
        or q_st >= p_e
        or (min_hit_len > 0 and (q_en - q_st) < min_hit_len)
        or (not allow_softclip and (q_st > p_s or q_en < p_e))
    ):
        return False

    strand = int(hit.strand)
    shift_w_size = max(REF_SUB_BLOCK_SIZE, min_hit_len)
    q_seq_len = len(query_seq) if query_seq is not None else -1

    # Single-pass cs parse extracting D/X sequence details strictly in a generous
    # symmetric window around [p_s, p_e) (covers shift_w_size + 2*max_i_len
    # expansion + structural D lookahead in one pass).
    dx_margin = shift_w_size * 2 + 320
    dx_lo = max(q_st, p_e - dx_margin)
    dx_hi = min(q_en, p_s + dx_margin)
    ops, d_seqs_local, x_map_local = _parse_cs_to_ops_and_local_dx(
        hit.cs, strand, q_st, dx_lo, dx_hi
    )
    dx_full_cache: (
        tuple[dict[tuple[int, int], str], dict[int, tuple[str, str]]] | None
    ) = None

    def _get_dx_for(
        req_lo: int, req_hi: int
    ) -> tuple[dict[tuple[int, int], str], dict[int, tuple[str, str]]]:
        nonlocal dx_full_cache
        if req_lo >= dx_lo and req_hi <= dx_hi:
            return d_seqs_local, x_map_local
        if dx_full_cache is None:
            _, d_all, x_all = _parse_cs_to_ops_and_local_dx(
                hit.cs, strand, q_st, q_st, q_en
            )
            dx_full_cache = (d_all, x_all)
        return dx_full_cache

    def _slice_q_up(a: int, b: int) -> str | None:
        if query_seq is None or a < 0 or b > q_seq_len:
            return None
        return query_seq[a:b].upper()

    check_targets = t_starts is not None and t_ends is not None
    use_direct = True
    direct_t_piece = 0
    if check_targets:
        assert t_starts is not None and t_ends is not None
        pos_t = bisect.bisect_right(t_starts, int(hit.r_st)) - 1
        if pos_t < 0 or int(hit.r_en) > t_ends[pos_t]:
            use_direct = False
        else:
            direct_t_piece = pos_t
            if blocked_t_pieces and direct_t_piece in blocked_t_pieces:
                return False
    elif blocked_t_pieces and 0 in blocked_t_pieces:
        return False
    cur_t_piece = direct_t_piece

    # ------------------------------------------------------------------
    # Structural D >= min_hit_len bounded zero-penalty extension helpers
    # ------------------------------------------------------------------
    def _collect_mx_right(
        ops_seq: Sequence[tuple[int, str, int]],
        start_idx: int,
        hi_lim: int,
        q_start: int,
        x_map: dict[int, tuple[str, str]],
        max_steps: int,
    ) -> tuple[list[str], list[str | None], str]:
        if max_steps <= 0 or query_seq is None or q_start >= q_seq_len:
            return [], [], ""
        take_lim = min(max_steps, q_seq_len - q_start)
        q_sub = query_seq[q_start : q_start + take_lim].upper()
        orig_ops: list[str] = []
        ref_bases: list[str | None] = []
        k = 0
        j = start_idx
        while j < hi_lim and k < take_lim:
            q_j, op_j, len_j = ops_seq[j]
            if op_j not in ("M", "X") or q_j != q_start + k:
                break
            take = min(len_j, take_lim - k)
            if take <= 0:
                break
            if op_j == "M":
                for off in range(take):
                    orig_ops.append("M")
                    ref_bases.append(q_sub[k + off])
            else:
                for off in range(take):
                    q_cur = q_start + k + off
                    orig_ops.append("X")
                    xq = x_map.get(q_cur)
                    if xq is not None and xq[1] == q_sub[k + off]:
                        ref_bases.append(xq[0])
                    else:
                        ref_bases.append(None)
            k += take
            if take < len_j:
                break
            j += 1
        return orig_ops, ref_bases, q_sub

    def _collect_mx_left(
        ops_seq: Sequence[tuple[int, str, int]],
        start_idx: int,
        lo_lim: int,
        q_end: int,
        x_map: dict[int, tuple[str, str]],
        max_steps: int,
    ) -> tuple[list[str], list[str | None], str]:
        if max_steps <= 0 or query_seq is None or q_end <= 0:
            return [], [], ""
        take_lim = min(max_steps, q_end)
        q_sub = query_seq[q_end - take_lim : q_end].upper()
        orig_ops: list[str] = []
        ref_bases: list[str | None] = []
        k = 0
        j = start_idx
        while j >= lo_lim and k < take_lim:
            q_j, op_j, len_j = ops_seq[j]
            if op_j not in ("M", "X") or q_j + len_j != q_end - k:
                break
            take = min(len_j, take_lim - k)
            if take <= 0:
                break
            if op_j == "M":
                for off in range(take):
                    orig_ops.append("M")
                    ref_bases.append(q_sub[take_lim - 1 - k - off])
            else:
                for off in range(take):
                    q_cur = q_end - 1 - k - off
                    orig_ops.append("X")
                    xq = x_map.get(q_cur)
                    if xq is not None and xq[1] == q_sub[take_lim - 1 - k - off]:
                        ref_bases.append(xq[0])
                    else:
                        ref_bases.append(None)
            k += take
            if take < len_j:
                break
            j -= 1
        return orig_ops, ref_bases, q_sub

    def _scan_struct_d_ext(
        orig_ops: Sequence[str],
        ref_bases: Sequence[str | None],
        d_stream: Sequence[str],
        q_sub: str,
        direction: int,
        base_steps: int,
    ) -> list[str]:
        """Longest zero-penalty M/X prefix within `base_steps`, extending past
        `base_steps` ONLY while repaying an active deficit (`orig_m > new_m`)."""
        n = len(orig_ops)
        if n == 0 or base_steps <= 0:
            return []
        rem_x = [0] * (n + 1)
        for idx_r in range(n - 1, -1, -1):
            rem_x[idx_r] = rem_x[idx_r + 1] + (1 if orig_ops[idx_r] == "X" else 0)
        t_stream: list[str | None] = list(d_stream) + list(ref_bases)
        q_sub_len = len(q_sub)
        new_ops: list[str] = []
        best_len = 0
        orig_m = 0
        new_m = 0
        for k in range(n):
            if k >= base_steps and new_m >= orig_m:
                break
            if orig_ops[k] == "M":
                orig_m += 1
            exp_b = t_stream[k]
            q_ch = q_sub[k] if direction == 1 else q_sub[q_sub_len - 1 - k]
            if exp_b is not None and q_ch == exp_b:
                new_ops.append("M")
                new_m += 1
            else:
                new_ops.append("X")
            if orig_m - new_m > rem_x[k + 1]:
                break
            if new_m >= orig_m:
                best_len = k + 1
                if k + 1 >= base_steps:
                    break
        return new_ops[:best_len]

    def _coalesce_base_ops(
        q_start: int, base_ops: Sequence[str]
    ) -> list[tuple[int, str, int]]:
        if not base_ops:
            return []
        out: list[tuple[int, str, int]] = []
        cur_op = base_ops[0]
        cur_s = q_start
        cur_l = 1
        for op_ch in base_ops[1:]:
            if op_ch == cur_op:
                cur_l += 1
            else:
                out.append((cur_s, cur_op, cur_l))
                cur_s += cur_l
                cur_op = op_ch
                cur_l = 1
        out.append((cur_s, cur_op, cur_l))
        return out

    def _extend_seg_right_ops(
        seg: Sequence[tuple[int, str, int]],
        add_ops: Sequence[tuple[int, str, int]],
    ) -> list[tuple[int, str, int]]:
        out = list(seg)
        for q_a, op_a, l_a in add_ops:
            if out and out[-1][1] == op_a and out[-1][0] + out[-1][2] == q_a:
                out[-1] = (out[-1][0], op_a, out[-1][2] + l_a)
            else:
                out.append((q_a, op_a, l_a))
        return out

    def _extend_seg_left_ops(
        seg: Sequence[tuple[int, str, int]],
        add_ops: Sequence[tuple[int, str, int]],
    ) -> list[tuple[int, str, int]]:
        out = list(add_ops)
        for q_a, op_a, l_a in seg:
            if out and out[-1][1] == op_a and out[-1][0] + out[-1][2] == q_a:
                out[-1] = (out[-1][0], op_a, out[-1][2] + l_a)
            else:
                out.append((q_a, op_a, l_a))
        return out

    # ------------------------------------------------------------------
    # Raw (indel-free / p_len >= shift_w_size / query_seq is None) evaluator
    # ------------------------------------------------------------------
    def _eval_raw_window(
        seg_ops: Sequence[tuple[int, str, int]],
        lo: int,
        hi: int,
        seg_q_st: int,
        seg_q_en: int,
        w_size: int,
    ) -> bool:
        if (seg_q_en - seg_q_st) < w_size:
            return False
        w_lo = max(min(p_s, seg_q_st), p_e - w_size)
        w_hi = min(p_s, max(p_e, seg_q_en) - w_size)
        if w_lo > w_hi:
            return False
        span_end = w_hi + w_size
        span_len = span_end - w_lo
        b_m = [0] * span_len
        b_x = [0] * span_len
        b_d = [0] * (span_len + 1)
        for idx_op in range(lo, hi):
            q_p, op, sub_len = seg_ops[idx_op]
            if op == "M" or op == "X":
                ov_s = max(q_p, w_lo)
                ov_e = min(q_p + sub_len, span_end)
                if ov_s < ov_e:
                    arr = b_m if op == "M" else b_x
                    for j_pos in range(ov_s - w_lo, ov_e - w_lo):
                        arr[j_pos] = 1
            elif op == "D":
                if w_lo <= q_p <= span_end:
                    b_d[q_p - w_lo] += sub_len
        pref_m = [0] * (span_len + 1)
        pref_x = [0] * (span_len + 1)
        pref_d = [0] * (span_len + 2)
        for j_pos in range(span_len):
            pref_m[j_pos + 1] = pref_m[j_pos] + b_m[j_pos]
            pref_x[j_pos + 1] = pref_x[j_pos] + b_x[j_pos]
        for j_pos in range(span_len + 1):
            pref_d[j_pos + 1] = pref_d[j_pos] + b_d[j_pos]
        for a in range(w_hi - w_lo + 1):
            b = a + w_size
            w_m = pref_m[b] - pref_m[a]
            w_x = pref_x[b] - pref_x[a]
            if (w_m + w_x) * 10 < 9 * w_size:
                continue
            w_d = pref_d[b] - pref_d[a + 1]
            w_denom = w_size + w_d
            if w_denom > 0 and (w_m + 1e-12) >= min_identity * w_denom:
                return True
        return False

    def _eval_raw_segment(
        seg_ops: Sequence[tuple[int, str, int]],
        lo: int,
        hi: int,
        seg_q_st: int,
        seg_q_en: int,
        m_p: int,
        x_p: int,
        i_p: int,
        d_p: int,
    ) -> bool:
        if (m_p + x_p) * 10 < 9 * p_len:
            return False
        denom = p_len + d_p
        if denom <= 0:
            return False
        raw_id_ok = (m_p + 1e-12) >= min_identity * denom
        if raw_id_ok:
            if min_hit_len <= 0 or p_len >= min_hit_len:
                return True
            if _eval_raw_window(seg_ops, lo, hi, seg_q_st, seg_q_en, min_hit_len):
                return True
            if min_hit_len < shift_w_size and p_len < REF_SUB_BLOCK_SIZE and i_p == 0 and d_p == 0:
                return _eval_raw_window(seg_ops, lo, hi, seg_q_st, seg_q_en, shift_w_size)
            return False
        if p_len >= REF_SUB_BLOCK_SIZE or i_p != 0 or d_p != 0:
            return False
        return _eval_raw_window(seg_ops, lo, hi, seg_q_st, seg_q_en, shift_w_size)

    # ------------------------------------------------------------------
    # Exact Fixed-Gap-Structure Diagonal-DP Equivalence-Class Evaluator
    # ------------------------------------------------------------------
    def _segment_covers_subinterval(
        seg_ops: Sequence[tuple[int, str, int]],
    ) -> bool:
        lo = 0
        hi = len(seg_ops)
        while lo < hi and seg_ops[lo][1] == "D":
            lo += 1
        while hi > lo and seg_ops[hi - 1][1] == "D":
            hi -= 1
        if lo >= hi:
            return False
        seg_q_st = seg_ops[lo][0]
        seg_q_en = seg_ops[hi - 1][0] + seg_ops[hi - 1][2]
        if seg_q_en <= p_s or seg_q_st >= p_e:
            return False
        if not allow_softclip and (seg_q_st > p_s or seg_q_en < p_e):
            return False
        if min_hit_len > 0 and (seg_q_en - seg_q_st) < min_hit_len:
            return False

        p_cov_seg = min(p_e, seg_q_en) - max(p_s, seg_q_st)
        if p_cov_seg * 10 < 9 * p_len:
            return False

        seg_spans_p = seg_q_st <= p_s and seg_q_en >= p_e
        max_w_i_budget = shift_w_size - ((9 * shift_w_size + 9) // 10)

        w_lo_full = max(min(p_s, seg_q_st), p_e - shift_w_size)
        w_hi_full = min(p_s, max(p_e, seg_q_en) - shift_w_size)
        if w_lo_full <= w_hi_full:
            base_lo = w_lo_full
            base_hi = w_hi_full + shift_w_size
        else:
            base_lo = seg_q_st
            base_hi = seg_q_en

        pad = shift_w_size
        scan_lo = max(seg_q_st, base_lo - pad)
        scan_hi = min(seg_q_en, base_hi + pad)

        def _slice_op_bounds(s_lo: int, s_hi: int) -> tuple[int, int]:
            b_lo, b_hi = lo, hi
            while b_lo < b_hi:
                mid = (b_lo + b_hi) // 2
                q_m, op_m, l_m = seg_ops[mid]
                end_m = q_m if op_m == "D" else q_m + l_m
                if end_m < s_lo or (op_m != "D" and end_m == s_lo):
                    b_lo = mid + 1
                else:
                    b_hi = mid
            o_lo = b_lo
            b_lo, b_hi = o_lo, hi
            while b_lo < b_hi:
                mid = (b_lo + b_hi) // 2
                q_m, op_m, _l_m = seg_ops[mid]
                if q_m < s_hi or (op_m == "D" and q_m == s_hi):
                    b_lo = mid + 1
                else:
                    b_hi = mid
            return o_lo, b_lo

        op_lo, op_hi = _slice_op_bounds(scan_lo, scan_hi)

        # Compute raw counts on [p_s, p_e) and detect any local indels
        m_p = 0
        x_p = 0
        i_p = 0
        d_p = 0
        has_local_indel = False
        max_i_len = 0
        for idx_op in range(op_lo, op_hi):
            q_p, op, sub_len = seg_ops[idx_op]
            if op in ("M", "X", "I"):
                ov = min(q_p + sub_len, p_e) - max(q_p, p_s)
                if ov > 0:
                    if op == "M":
                        m_p += ov
                    elif op == "X":
                        x_p += ov
                    else:
                        i_p += ov
                if op == "I":
                    has_local_indel = True
                    if sub_len > max_i_len:
                        max_i_len = sub_len
            elif op == "D":
                if p_s < q_p < p_e:
                    d_p += sub_len
                if scan_lo < q_p < scan_hi:
                    has_local_indel = True

        def _tandem_matches(s_abs: int, l_len: int) -> int:
            """Mirror-invariant O(L) single-cut tandem homology of Q[s_abs:s_abs+L]
            against its +L right neighbor (prefix) and -L left neighbor (suffix)."""
            if l_len <= 0:
                return 0
            q_mid = _slice_q_up(s_abs, s_abs + l_len)
            if q_mid is None:
                return 0
            r_s = s_abs + l_len
            r_e = min(seg_q_en, r_s + l_len)
            r_avail = max(0, r_e - r_s)
            l_e = s_abs
            l_s = max(seg_q_st, l_e - l_len)
            l_avail = max(0, l_e - l_s)
            if r_avail == 0 and l_avail == 0:
                return 0
            q_r: str | None = None
            if r_avail > 0:
                qr_raw = _slice_q_up(r_s, r_s + r_avail)
                if qr_raw is not None:
                    q_r = (
                        qr_raw
                        if r_avail == l_len
                        else qr_raw + (" " * (l_len - r_avail))
                    )
            q_l: str | None = None
            if l_avail > 0:
                ql_raw = _slice_q_up(l_s, l_s + l_avail)
                if ql_raw is not None:
                    q_l = (
                        ql_raw
                        if l_avail == l_len
                        else (" " * (l_len - l_avail)) + ql_raw
                    )
            if q_r is None and q_l is None:
                return 0
            if q_r is not None and q_mid == q_r:
                return l_len
            if q_l is not None and q_mid == q_l:
                return l_len
            if q_l is None:
                assert q_r is not None
                return sum(1 for a, b in zip(q_mid, q_r) if a == b)
            if q_r is None:
                return sum(1 for a, b in zip(q_mid, q_l) if a == b)
            cur = sum(1 for a, b in zip(q_mid, q_l) if a == b)
            best = cur
            for a, b_r, b_l in zip(q_mid, q_r, q_l):
                if a == b_r:
                    cur += 1
                if a == b_l:
                    cur -= 1
                if cur > best:
                    best = cur
            return best

        inject_left_i = 0
        inject_right_i = 0
        if (
            p_len < shift_w_size
            and query_seq is not None
            and q_seq_len >= seg_q_en
        ):
            # ----------------------------------------------------------
            # Step A: Cap-free, D-transparent, full-run exact & quasi-tandem
            # evaluator for ANY M/X/D-connected I(L) with L > p_len and
            # (L >= min_hit_len or L > max_w_i_budget). Eliminates L >= 101 /
            # 128 / 179 ceilings, sc_span < L quasi-tandem vaporization,
            # D+I vs I+D large-L blind spots, and N-copy distant-SNP truncation.
            # ----------------------------------------------------------
            def _check_unbounded_tandem_i(
                idx_i: int, check_tandem_rej: bool = True
            ) -> tuple[bool, int | None, int | None, bool]:
                q_i, _, l_i = seg_ops[idx_i]
                # Walk left across contiguous M/X/D ops (D consumes 0 query bases)
                mx_l_idx = idx_i
                mx_l_start = q_i
                j_l = idx_i - 1
                while j_l >= lo:
                    q_j, op_j, len_j = seg_ops[j_l]
                    if op_j == "D" and q_j == mx_l_start:
                        mx_l_idx = j_l
                        j_l -= 1
                        continue
                    if op_j not in ("M", "X") or q_j + len_j != mx_l_start:
                        break
                    mx_l_start = q_j
                    mx_l_idx = j_l
                    j_l -= 1
                # Walk right across contiguous M/X/D ops
                mx_r_idx = idx_i + 1
                mx_r_end = q_i + l_i
                j_r = idx_i + 1
                while j_r < hi:
                    q_j, op_j, len_j = seg_ops[j_r]
                    if op_j == "D" and q_j == mx_r_end:
                        mx_r_idx = j_r + 1
                        j_r += 1
                        continue
                    if op_j not in ("M", "X") or q_j != mx_r_end:
                        break
                    mx_r_end = q_j + len_j
                    mx_r_idx = j_r + 1
                    j_r += 1
                if check_tandem_rej:
                    if mx_l_start > p_s or mx_r_end < p_e:
                        return False, None, None, False
                elif mx_l_start >= base_hi or mx_r_end <= base_lo:
                    return False, None, None, False

                # Full unbounded I-free run domain (exact for 2-copy, 3-copy, N-copy)
                dom_lo = mx_l_start
                dom_hi = mx_r_end - l_i
                if dom_lo > dom_hi:
                    return False, None, None, False

                # Count X bases across full left & right domains
                x_l = 0
                for j_op in range(mx_l_idx, idx_i):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "X":
                        x_l += len_j
                x_r = 0
                for j_op in range(idx_i + 1, mx_r_idx):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "X":
                        x_r += len_j

                # Fast Path 1: zero-X across ENTIRE I-free run -> Delta(s) <= 0 globally
                # (contiguous S*; for 1 <= l_i <= pad without tandem reject, exit bounds
                # already lie inside [base_lo - pad, base_hi + pad]).
                if x_l == 0 and x_r == 0:
                    if not check_tandem_rej:
                        return False, None, None, False
                    if q_i <= p_s:
                        r_end_chk = max(q_i + 2 * l_i, p_e)
                        if r_end_chk <= mx_r_end:
                            q_sub = _slice_q_up(q_i, r_end_chk)
                            if (
                                q_sub is not None
                                and len(q_sub) > l_i
                                and q_sub[:-l_i] == q_sub[l_i:]
                            ):
                                return True, None, None, False
                    if q_i + l_i >= p_e:
                        l_st_chk = min(q_i - l_i, p_s)
                        if l_st_chk >= mx_l_start:
                            q_sub = _slice_q_up(l_st_chk, q_i + l_i)
                            if (
                                q_sub is not None
                                and len(q_sub) > l_i
                                and q_sub[:-l_i] == q_sub[l_i:]
                            ):
                                return True, None, None, False

                # Exact necessary narrowing of [l_eff, r_eff] before any cs re-parse
                r_eff = dom_hi
                r_q_lim = dom_hi + l_i
                if dom_hi > q_i:
                    m_mm = 0
                    for j_op in range(idx_i + 1, mx_r_idx):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if q_j >= r_q_lim:
                            break
                        if op_j != "M":
                            continue
                        seg_s = max(q_j, q_i + l_i)
                        seg_e = min(q_j + len_j, r_q_lim)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s - l_i, seg_e - l_i)
                        s_b = _slice_q_up(seg_s, seg_e)
                        if s_a is None or s_b is None:
                            return False, None, None, False
                        if s_a != s_b:
                            broke = False
                            for off, (ca, cb) in enumerate(zip(s_a, s_b)):
                                if ca != cb:
                                    m_mm += 1
                                    if m_mm > x_r:
                                        r_eff = (seg_s + off) - l_i
                                        broke = True
                                        break
                            if broke:
                                break

                l_eff = dom_lo
                if dom_lo < q_i:
                    m_mm = 0
                    for j_op in range(idx_i - 1, mx_l_idx - 1, -1):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if op_j == "D":
                            continue
                        if q_j + len_j <= dom_lo:
                            break
                        if op_j != "M":
                            continue
                        seg_s = max(q_j, dom_lo)
                        seg_e = min(q_j + len_j, q_i)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s, seg_e)
                        s_b = _slice_q_up(seg_s + l_i, seg_e + l_i)
                        if s_a is None or s_b is None:
                            return False, None, None, False
                        if s_a != s_b:
                            broke = False
                            span_m = seg_e - seg_s
                            for off in range(span_m - 1, -1, -1):
                                if s_a[off] != s_b[off]:
                                    m_mm += 1
                                    if m_mm > x_l:
                                        l_eff = seg_s + off + 1
                                        broke = True
                                        break
                            if broke:
                                break

                cov_lo = max(l_eff, p_e - l_i)
                cov_hi = min(r_eff, p_s)
                if l_eff > r_eff or l_eff >= base_hi or r_eff + l_i <= base_lo:
                    return False, None, None, False

                if x_l > 0 or x_r > 0:
                    _, x_map_a = _get_dx_for(l_eff, r_eff + l_i)
                else:
                    x_map_a = {}
                q_full = _slice_q_up(l_eff, r_eff + l_i)
                if q_full is None:
                    return False, None, None, False

                # O(1)-memory streaming exact 1-D Delta scan around q_i:
                # Delta(s) = m0_cum(s) - m1_cum(s) identically for ALL s in [l_eff, r_eff].
                max_delta = 0
                opt_lo = q_i
                opt_lo_m0 = 0
                opt_lo_m1 = 0
                opt_hi = q_i
                opt_hi_m0 = 0
                opt_hi_m1 = 0
                el_lim = base_lo - l_i
                er_lim = base_hi
                cov_opt_lo: int | None = q_i if (cov_lo <= q_i <= cov_hi) else None
                cov_opt_hi: int | None = q_i if (cov_lo <= q_i <= cov_hi) else None
                exit_l_max: int | None = q_i if q_i <= el_lim else None
                exit_r_min: int | None = q_i if q_i >= er_lim else None
                max_ov_opt = max(0, min(q_i + l_i, p_e) - max(q_i, p_s))
                has_base_opt = q_i < base_hi and q_i + l_i > base_lo

                def _upd_range(
                    d_val: int, ra: int, m0_a: int, m1_a: int, rb: int, m0_b: int, m1_b: int
                ) -> None:
                    nonlocal max_delta, opt_lo, opt_lo_m0, opt_lo_m1
                    nonlocal opt_hi, opt_hi_m0, opt_hi_m1, cov_opt_lo, cov_opt_hi
                    nonlocal exit_l_max, exit_r_min, max_ov_opt, has_base_opt
                    if d_val < max_delta:
                        return
                    c_a = max(ra, cov_lo)
                    c_b = min(rb, cov_hi)
                    l_cand = min(rb, el_lim) if ra <= el_lim else None
                    r_cand = max(ra, er_lim) if rb >= er_lim else None
                    r_has_base = ra < base_hi and rb + l_i > base_lo
                    if c_a <= c_b:
                        r_ov = p_len
                    elif rb < p_e - l_i:
                        r_ov = max(0, rb + l_i - p_s)
                    elif ra > p_s:
                        r_ov = max(0, p_e - ra)
                    else:
                        r_ov = p_len
                    if d_val > max_delta:
                        max_delta = d_val
                        opt_lo, opt_lo_m0, opt_lo_m1 = ra, m0_a, m1_a
                        opt_hi, opt_hi_m0, opt_hi_m1 = rb, m0_b, m1_b
                        if c_a <= c_b:
                            cov_opt_lo = c_a
                            cov_opt_hi = c_b
                        else:
                            cov_opt_lo = None
                            cov_opt_hi = None
                        exit_l_max = l_cand
                        exit_r_min = r_cand
                        max_ov_opt = r_ov
                        has_base_opt = r_has_base
                    else:
                        if ra < opt_lo:
                            opt_lo, opt_lo_m0, opt_lo_m1 = ra, m0_a, m1_a
                        if rb > opt_hi:
                            opt_hi, opt_hi_m0, opt_hi_m1 = rb, m0_b, m1_b
                        if c_a <= c_b:
                            if cov_opt_lo is None or c_a < cov_opt_lo:
                                cov_opt_lo = c_a
                            if cov_opt_hi is None or c_b > cov_opt_hi:
                                cov_opt_hi = c_b
                        if l_cand is not None and (exit_l_max is None or l_cand > exit_l_max):
                            exit_l_max = l_cand
                        if r_cand is not None and (exit_r_min is None or r_cand < exit_r_min):
                            exit_r_min = r_cand
                        if r_ov > max_ov_opt:
                            max_ov_opt = r_ov
                        if r_has_base:
                            has_base_opt = True

                # Scan RIGHT: u = q_i .. r_eff - 1, pos = u + l_i
                m0_c = 0
                m1_c = 0
                r_pos_lim = r_eff + l_i
                for j_op in range(idx_i + 1, mx_r_idx):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j >= r_pos_lim:
                        break
                    seg_s = max(q_j, q_i + l_i)
                    seg_e = min(q_j + len_j, r_pos_lim)
                    if seg_s >= seg_e:
                        continue
                    span_k = seg_e - seg_s
                    if op_j == "M":
                        o_u = seg_s - l_i - l_eff
                        o_p = seg_s - l_eff
                        if q_full[o_u : o_u + span_k] == q_full[o_p : o_p + span_k]:
                            ra = seg_s - l_i + 1
                            rb = seg_e - l_i
                            _upd_range(
                                m0_c - m1_c,
                                ra,
                                m0_c + 1,
                                m1_c + 1,
                                rb,
                                m0_c + span_k,
                                m1_c + span_k,
                            )
                            m0_c += span_k
                            m1_c += span_k
                        else:
                            for off in range(span_k):
                                if q_full[o_u + off] == q_full[o_p + off]:
                                    m0_c += 1
                                m1_c += 1
                                s_cur = seg_s + off - l_i + 1
                                _upd_range(
                                    m0_c - m1_c, s_cur, m0_c, m1_c, s_cur, m0_c, m1_c
                                )
                    else:  # op_j == "X"
                        for pos in range(seg_s, seg_e):
                            xq = x_map_a.get(pos)
                            if (
                                xq is not None
                                and xq[1] == q_full[pos - l_eff]
                                and q_full[pos - l_i - l_eff] == xq[0]
                            ):
                                m0_c += 1
                            s_cur = pos - l_i + 1
                            _upd_range(
                                m0_c - m1_c, s_cur, m0_c, m1_c, s_cur, m0_c, m1_c
                            )

                # Scan LEFT: u = q_i - 1 .. l_eff, s = u
                m0_c = 0
                m1_c = 0
                for j_op in range(idx_i - 1, mx_l_idx - 1, -1):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j + len_j <= l_eff:
                        break
                    seg_s = max(q_j, l_eff)
                    seg_e = min(q_j + len_j, q_i)
                    if seg_s >= seg_e:
                        continue
                    span_k = seg_e - seg_s
                    if op_j == "M":
                        o_u = seg_s - l_eff
                        o_p = seg_s + l_i - l_eff
                        if q_full[o_u : o_u + span_k] == q_full[o_p : o_p + span_k]:
                            ra = seg_s
                            rb = seg_e - 1
                            _upd_range(
                                m0_c - m1_c,
                                ra,
                                m0_c - span_k,
                                m1_c - span_k,
                                rb,
                                m0_c - 1,
                                m1_c - 1,
                            )
                            m0_c -= span_k
                            m1_c -= span_k
                        else:
                            for off in range(span_k - 1, -1, -1):
                                m0_c -= 1
                                if q_full[o_u + off] == q_full[o_p + off]:
                                    m1_c -= 1
                                s_cur = seg_s + off
                                _upd_range(
                                    m0_c - m1_c, s_cur, m0_c, m1_c, s_cur, m0_c, m1_c
                                )
                    else:  # op_j == "X"
                        for u_pos in range(seg_e - 1, seg_s - 1, -1):
                            xq = x_map_a.get(u_pos)
                            if (
                                xq is not None
                                and xq[1] == q_full[u_pos - l_eff]
                                and q_full[u_pos + l_i - l_eff] == xq[0]
                            ):
                                m1_c -= 1
                            _upd_range(
                                m0_c - m1_c, u_pos, m0_c, m1_c, u_pos, m0_c, m1_c
                            )

                if check_tandem_rej:
                    # Gem #1: Inescapable optimal placement (min_ov > 0 across ALL
                    # optimal placements in [opt_lo, opt_hi]) -> protected_ins / reject_early
                    if opt_lo + l_i > p_s and opt_hi < p_e:
                        if min_hit_len > 0 and l_i >= min_hit_len:
                            return True, None, None, has_base_opt
                        if (p_cov_seg - max_ov_opt) * 10 < 9 * p_len:
                            return True, None, None, has_base_opt

                    if cov_opt_lo is not None and cov_opt_hi is not None:
                        sc_span = opt_hi - opt_lo
                        if (
                            sc_span > 0
                            and min(opt_hi_m0 - opt_lo_m0, opt_hi_m1 - opt_lo_m1) * 2
                            > sc_span
                        ):
                            if sc_span >= l_i:
                                return True, None, None, has_base_opt
                            tm = _tandem_matches(cov_opt_lo, l_i)
                            if cov_opt_hi != cov_opt_lo:
                                tm = max(tm, _tandem_matches(cov_opt_hi, l_i))
                            if tm * 10 >= 7 * l_i:
                                return True, None, None, has_base_opt

                # Gem #2: When I(l_i) is local/overlapping P and can exit P optimally
                # (without s_cover), pass exact exit bounds so Diagonal-DP scan includes
                # the optimal exit (and any higher-scoring distant SNP if max_delta > 0).
                need_lo = (
                    (exit_l_max if exit_l_max is not None else opt_lo)
                    if opt_lo + l_i <= base_hi
                    else None
                )
                need_hi = (
                    ((exit_r_min if exit_r_min is not None else opt_hi) + l_i)
                    if opt_hi >= base_lo
                    else None
                )
                return False, need_lo, need_hi, has_base_opt

            def _find_outer_left_i(
                o_lo: int, o_hi: int
            ) -> tuple[int, int, int] | None:
                if o_lo > lo and o_lo < o_hi and seg_ops[o_lo][1] in ("M", "X", "D"):
                    cur_q = seg_ops[o_lo][0]
                    j_w = o_lo - 1
                    while j_w >= lo:
                        q_j, op_j, len_j = seg_ops[j_w]
                        if op_j == "D" and q_j == cur_q:
                            j_w -= 1
                        elif op_j in ("M", "X") and q_j + len_j == cur_q:
                            cur_q = q_j
                            j_w -= 1
                        elif op_j == "I" and q_j + len_j == cur_q:
                            return j_w, q_j, len_j
                        else:
                            return None
                return None

            def _find_outer_right_i(
                o_lo: int, o_hi: int
            ) -> tuple[int, int, int] | None:
                if o_hi < hi and o_lo < o_hi and seg_ops[o_hi - 1][1] in ("M", "X", "D"):
                    q_last, op_last, l_last = seg_ops[o_hi - 1]
                    cur_q = q_last if op_last == "D" else q_last + l_last
                    j_w = o_hi
                    while j_w < hi:
                        q_j, op_j, len_j = seg_ops[j_w]
                        if op_j == "D" and q_j == cur_q:
                            j_w += 1
                        elif op_j in ("M", "X") and q_j == cur_q:
                            cur_q = q_j + len_j
                            j_w += 1
                        elif op_j == "I" and q_j == cur_q:
                            return j_w, q_j, len_j
                        else:
                            return None
                return None

            # Identify the <= 2 M/X/D-connected candidate I ops around P
            p_op_lo, p_op_hi = _slice_op_bounds(
                max(seg_q_st, p_s), min(seg_q_en, p_e)
            )
            p_i_idxs = [
                j_p for j_p in range(p_op_lo, p_op_hi) if seg_ops[j_p][1] == "I"
            ]
            step_a_cands: list[int] = []
            if len(p_i_idxs) == 1:
                step_a_cands.append(p_i_idxs[0])
            elif len(p_i_idxs) == 0 and p_op_lo < p_op_hi:
                if seg_ops[p_op_lo][1] in ("M", "X", "D"):
                    cur_q = seg_ops[p_op_lo][0]
                    j_wl = p_op_lo - 1
                    while j_wl >= lo:
                        q_j, op_j, len_j = seg_ops[j_wl]
                        if op_j == "D" and q_j == cur_q:
                            j_wl -= 1
                        elif op_j in ("M", "X") and q_j + len_j == cur_q:
                            cur_q = q_j
                            j_wl -= 1
                        elif op_j == "I" and q_j + len_j == cur_q:
                            step_a_cands.append(j_wl)
                            break
                        else:
                            break
                if seg_ops[p_op_hi - 1][1] in ("M", "X", "D"):
                    q_last_p, op_last_p, l_last_p = seg_ops[p_op_hi - 1]
                    cur_q = q_last_p if op_last_p == "D" else q_last_p + l_last_p
                    j_wr = p_op_hi
                    while j_wr < hi:
                        q_j, op_j, len_j = seg_ops[j_wr]
                        if op_j == "D" and q_j == cur_q:
                            j_wr += 1
                        elif op_j in ("M", "X") and q_j == cur_q:
                            cur_q = q_j + len_j
                            j_wr += 1
                        elif op_j == "I" and q_j == cur_q:
                            step_a_cands.append(j_wr)
                            break
                        else:
                            break
            step_a_need_lo: int | None = None
            step_a_need_hi: int | None = None
            checked_step_a: set[int] = set()
            for idx_ca in step_a_cands:
                checked_step_a.add(idx_ca)
                q_ca, _, l_ca = seg_ops[idx_ca]
                is_big_i = l_ca > p_len and (
                    (min_hit_len > 0 and l_ca >= min_hit_len)
                    or l_ca > max_w_i_budget
                )
                is_rej_a, n_lo_a, n_hi_a, has_base_a = _check_unbounded_tandem_i(
                    idx_ca, check_tandem_rej=is_big_i
                )
                if is_rej_a:
                    if ins_block_t_pieces_out is not None and seg_spans_p:
                        ins_block_t_pieces_out.add(cur_t_piece)
                    return False
                # Propagate optimal exit expansion if I overlaps initial scan OR
                # its global optimal set S* intersects [base_lo, base_hi) (l_ca <= pad).
                if (q_ca < scan_hi and q_ca + l_ca > scan_lo) or (
                    has_base_a and l_ca <= pad
                ):
                    if n_lo_a is not None:
                        step_a_need_lo = (
                            n_lo_a
                            if step_a_need_lo is None
                            else min(step_a_need_lo, n_lo_a)
                        )
                    if n_hi_a is not None:
                        step_a_need_hi = (
                            n_hi_a
                            if step_a_need_hi is None
                            else max(step_a_need_hi, n_hi_a)
                        )

            # Symmetric boundary-I exit-bound discovery for multi-I alignments (K >= 2)
            # not already inspected via step_a_cands (leftmost/rightmost in-scan & outer I).
            extra_exit_cands: list[int] = []
            if max_i_len > 0:
                for idx_op in range(op_lo, op_hi):
                    if seg_ops[idx_op][1] == "I":
                        if idx_op not in checked_step_a:
                            extra_exit_cands.append(idx_op)
                        break
                for idx_op in range(op_hi - 1, op_lo - 1, -1):
                    if seg_ops[idx_op][1] == "I":
                        if idx_op not in checked_step_a and idx_op not in extra_exit_cands:
                            extra_exit_cands.append(idx_op)
                        break
            init_out_l = _find_outer_left_i(op_lo, op_hi)
            if (
                init_out_l is not None
                and init_out_l[0] not in checked_step_a
                and init_out_l[0] not in extra_exit_cands
            ):
                extra_exit_cands.append(init_out_l[0])
            init_out_r = _find_outer_right_i(op_lo, op_hi)
            if (
                init_out_r is not None
                and init_out_r[0] not in checked_step_a
                and init_out_r[0] not in extra_exit_cands
            ):
                extra_exit_cands.append(init_out_r[0])
            for idx_ex in extra_exit_cands:
                q_ex, _, l_ex = seg_ops[idx_ex]
                if 1 <= l_ex <= pad:
                    _, n_lo_e, n_hi_e, has_base_e = _check_unbounded_tandem_i(
                        idx_ex, check_tandem_rej=False
                    )
                    if (q_ex < scan_hi and q_ex + l_ex > scan_lo) or has_base_e:
                        if n_lo_e is not None:
                            step_a_need_lo = (
                                n_lo_e
                                if step_a_need_lo is None
                                else min(step_a_need_lo, n_lo_e)
                            )
                        if n_hi_e is not None:
                            step_a_need_hi = (
                                n_hi_e
                                if step_a_need_hi is None
                                else max(step_a_need_hi, n_hi_e)
                            )

            # ----------------------------------------------------------
            # Step B: Cap-free boundary adjustment & globally optimal
            # arbitrary-M/X/D outer-I reachability injection (scans full
            # I-free run left, inside, and right of scan; zero 128-bp or
            # fixed-pattern ceilings).
            # ----------------------------------------------------------
            def _adjust_boundary_i() -> None:
                nonlocal scan_lo, scan_hi, op_lo, op_hi, has_local_indel
                adj = False
                for idx_op in range(op_lo, op_hi):
                    q_p, op, sub_len = seg_ops[idx_op]
                    if op == "I":
                        if q_p < scan_lo < q_p + sub_len:
                            if q_p + sub_len <= base_lo and sub_len > 128:
                                scan_lo = q_p + sub_len
                            else:
                                scan_lo = max(seg_q_st, q_p - shift_w_size)
                                has_local_indel = True
                            adj = True
                        if q_p < scan_hi < q_p + sub_len:
                            if q_p >= base_hi and sub_len > 128:
                                scan_hi = q_p
                            else:
                                scan_hi = min(seg_q_en, q_p + sub_len + shift_w_size)
                                has_local_indel = True
                            adj = True
                if adj:
                    op_lo, op_hi = _slice_op_bounds(scan_lo, scan_hi)
                    adj2 = False
                    for idx_op in range(op_lo, op_hi):
                        q_p, op, sub_len = seg_ops[idx_op]
                        if op == "I":
                            if q_p < scan_lo < q_p + sub_len:
                                scan_lo = q_p + sub_len if q_p + sub_len <= base_lo else q_p
                                adj2 = True
                            if q_p < scan_hi < q_p + sub_len:
                                scan_hi = q_p if q_p >= base_hi else q_p + sub_len
                                adj2 = True
                    if adj2:
                        op_lo, op_hi = _slice_op_bounds(scan_lo, scan_hi)

            scan_expanded = False
            if max_i_len > 0:
                extra_pad = pad + 2 * min(max_i_len, 128)
                scan_lo = max(seg_q_st, base_lo - extra_pad)
                scan_hi = min(seg_q_en, base_hi + extra_pad)
                scan_expanded = True
            if step_a_need_lo is not None or step_a_need_hi is not None:
                cand_lo = scan_lo
                cand_hi = scan_hi
                if step_a_need_lo is not None and step_a_need_lo < cand_lo:
                    cand_lo = max(seg_q_st, step_a_need_lo)
                if step_a_need_hi is not None and step_a_need_hi > cand_hi:
                    cand_hi = min(seg_q_en, step_a_need_hi)
                if cand_lo < scan_lo or cand_hi > scan_hi:
                    if cand_hi - cand_lo <= 8000:
                        scan_lo = cand_lo
                        scan_hi = cand_hi
                    else:
                        ok_l = cand_lo < scan_lo and (scan_hi - cand_lo) <= 8000
                        ok_r = cand_hi > scan_hi and (cand_hi - scan_lo) <= 8000
                        if ok_l and not ok_r:
                            scan_lo = cand_lo
                        elif ok_r and not ok_l:
                            scan_hi = cand_hi
                        elif ok_l and ok_r:
                            d_l = scan_lo - cand_lo
                            d_r = cand_hi - scan_hi
                            if d_l < d_r:
                                scan_lo = cand_lo
                            elif d_r < d_l:
                                scan_hi = cand_hi
                    scan_expanded = True
            if scan_expanded:
                op_lo, op_hi = _slice_op_bounds(scan_lo, scan_hi)

            _adjust_boundary_i()

            if not has_local_indel:
                for idx_op in range(op_lo, op_hi):
                    q_p, op, _ = seg_ops[idx_op]
                    if op == "I" or (op == "D" and scan_lo < q_p < scan_hi):
                        has_local_indel = True
                        break

            def _check_outer_left_inject(
                idx_il: int, q_il: int, l_il: int, s_lo: int, s_hi: int
            ) -> bool:
                if l_il <= 0 or l_il > pad or s_lo + l_il > s_hi:
                    return False
                # Phase 1: Check if a strictly better placement exists LEFT of q_il
                mx_l_start = q_il
                mx_l_idx = idx_il
                x_left = 0
                j_l = idx_il - 1
                while j_l >= lo:
                    q_j, op_j, len_j = seg_ops[j_l]
                    if op_j == "D" and q_j == mx_l_start:
                        mx_l_idx = j_l
                        j_l -= 1
                        continue
                    if op_j not in ("M", "X") or q_j + len_j != mx_l_start:
                        break
                    if op_j == "X":
                        x_left += len_j
                    mx_l_start = q_j
                    mx_l_idx = j_l
                    j_l -= 1

                m_outside_l = 0
                if x_left > 0 and mx_l_start < q_il:
                    l_eff_b = mx_l_start
                    m_mm_l = 0
                    for j_op in range(idx_il - 1, mx_l_idx - 1, -1):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if op_j != "M":
                            continue
                        seg_s = max(q_j, mx_l_start)
                        seg_e = min(q_j + len_j, q_il)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s, seg_e)
                        s_b = _slice_q_up(seg_s + l_il, seg_e + l_il)
                        if s_a is None or s_b is None:
                            return False
                        if s_a != s_b:
                            broke = False
                            for off in range(seg_e - seg_s - 1, -1, -1):
                                if s_a[off] != s_b[off]:
                                    m_mm_l += 1
                                    if m_mm_l > x_left:
                                        l_eff_b = seg_s + off + 1
                                        broke = True
                                        break
                            if broke:
                                break
                    if l_eff_b < q_il:
                        _, x_map_l = _get_dx_for(l_eff_b, q_il)
                        q_ls = _slice_q_up(l_eff_b, q_il + l_il)
                        if q_ls is None:
                            return False
                        dl = 0
                        rem_xl = x_left
                        for j_op in range(idx_il - 1, mx_l_idx - 1, -1):
                            q_j, op_j, len_j = seg_ops[j_op]
                            if op_j == "D":
                                continue
                            if q_j + len_j <= l_eff_b:
                                break
                            seg_s = max(q_j, l_eff_b)
                            seg_e = min(q_j + len_j, q_il)
                            if seg_s >= seg_e:
                                continue
                            span_k = seg_e - seg_s
                            if op_j == "M":
                                o_u = seg_s - l_eff_b
                                o_p = seg_s + l_il - l_eff_b
                                if q_ls[o_u : o_u + span_k] != q_ls[o_p : o_p + span_k]:
                                    for off in range(span_k - 1, -1, -1):
                                        if q_ls[o_u + off] != q_ls[o_p + off]:
                                            dl -= 1
                                            if dl + rem_xl <= m_outside_l:
                                                break
                                    if dl + rem_xl <= m_outside_l:
                                        break
                            else:  # op_j == "X"
                                for u_pos in range(seg_e - 1, seg_s - 1, -1):
                                    rem_xl -= 1
                                    xq = x_map_l.get(u_pos)
                                    if (
                                        xq is not None
                                        and xq[1] == q_ls[u_pos - l_eff_b]
                                        and q_ls[u_pos + l_il - l_eff_b] == xq[0]
                                    ):
                                        dl += 1
                                        if dl > m_outside_l:
                                            m_outside_l = dl
                                    if dl + rem_xl <= m_outside_l:
                                        break
                                if dl + rem_xl <= m_outside_l:
                                    break

                # Phase 2: Walk RIGHT across full M/X/D run (including past s_hi)
                mx_r_end = q_il + l_il
                has_x_before_eff = False
                x_before_shi = 0
                x_after_shi = 0
                j_end = idx_il + 1
                while j_end < hi:
                    q_j, op_j, len_j = seg_ops[j_end]
                    if op_j == "D":
                        if q_j != mx_r_end:
                            break
                        j_end += 1
                        continue
                    if op_j not in ("M", "X") or q_j != mx_r_end:
                        break
                    op_e = q_j + len_j
                    if op_j == "X":
                        if q_j < s_hi:
                            x_before_shi += min(op_e, s_hi) - q_j
                        if op_e > s_hi:
                            x_after_shi += op_e - max(q_j, s_hi)
                        if q_j < s_lo + l_il:
                            has_x_before_eff = True
                    mx_r_end = op_e
                    j_end += 1
                if mx_r_end < s_lo + l_il:
                    return False
                if m_outside_l == 0 and not has_x_before_eff and x_after_shi == 0:
                    q_sub = _slice_q_up(q_il, s_lo + l_il)
                    if (
                        q_sub is not None
                        and len(q_sub) > l_il
                        and q_sub[:-l_il] == q_sub[l_il:]
                    ):
                        return True
                    if x_before_shi == 0:
                        return False
                # Fast necessary rejection on [q_il + l_il, s_lo + l_il)
                m_mm = 0
                eff_cut = s_lo + l_il
                for j_op in range(idx_il + 1, j_end):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j >= eff_cut:
                        break
                    if op_j != "M":
                        continue
                    seg_e = min(q_j + len_j, eff_cut)
                    s_a = _slice_q_up(q_j - l_il, seg_e - l_il)
                    s_b = _slice_q_up(q_j, seg_e)
                    if s_a is None or s_b is None:
                        return False
                    if s_a != s_b:
                        for ca, cb in zip(s_a, s_b):
                            if ca != cb:
                                m_mm += 1
                                if x_before_shi - m_mm < m_outside_l:
                                    return False
                # Narrow r_lim (past s_hi only if x_after_shi > 0)
                r_lim = min(mx_r_end, s_hi) if x_after_shi == 0 else mx_r_end
                if x_after_shi > 0:
                    m_mm_all = m_mm
                    x_tot_r = x_before_shi + x_after_shi
                    for j_op in range(idx_il + 1, j_end):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if op_j != "M" or q_j + len_j <= eff_cut:
                            continue
                        seg_s = max(q_j, eff_cut)
                        seg_e = min(q_j + len_j, mx_r_end)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s - l_il, seg_e - l_il)
                        s_b = _slice_q_up(seg_s, seg_e)
                        if s_a is None or s_b is None:
                            return False
                        if s_a != s_b:
                            broke = False
                            for off, (ca, cb) in enumerate(zip(s_a, s_b)):
                                if ca != cb:
                                    m_mm_all += 1
                                    if x_tot_r - m_mm_all < m_outside_l:
                                        r_lim = seg_s + off
                                        broke = True
                                        break
                            if broke:
                                break
                    if r_lim < s_lo + l_il:
                        return False
                _, x_map_b = _get_dx_for(q_il + l_il, r_lim)
                q_full = _slice_q_up(q_il, r_lim)
                if q_full is None:
                    return False
                delta = 0
                m_left = m_outside_l
                m_in = -(10**9)
                rem_x_in = x_before_shi
                rem_x_all = x_before_shi + x_after_shi
                s_in_hi = s_hi - l_il
                for j_op in range(idx_il + 1, j_end):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j >= r_lim:
                        break
                    seg_e = min(q_j + len_j, r_lim)
                    if seg_e <= q_j:
                        continue
                    span_k = seg_e - q_j
                    if op_j == "M":
                        o_u = q_j - l_il - q_il
                        o_p = q_j - q_il
                        if q_full[o_u : o_u + span_k] == q_full[o_p : o_p + span_k]:
                            ra = q_j - l_il + 1
                            rb = seg_e - l_il
                            if ra < s_lo and delta > m_left:
                                m_left = delta
                            if rb >= s_lo and ra <= s_in_hi:
                                if delta > m_in:
                                    m_in = delta
                                if m_in >= m_left:
                                    if x_after_shi == 0 or delta + rem_x_all <= m_in:
                                        return True
                                elif delta + rem_x_in < m_left:
                                    return False
                            if rb > s_in_hi:
                                if m_in < m_left or delta > m_in:
                                    return False
                                if delta + rem_x_all <= m_in:
                                    return True
                        else:
                            for off in range(span_k):
                                if q_full[o_u + off] != q_full[o_p + off]:
                                    delta -= 1
                                s_after = q_j + off - l_il + 1
                                if s_after < s_lo:
                                    if delta > m_left:
                                        m_left = delta
                                    if delta + rem_x_in < m_left:
                                        return False
                                elif s_after <= s_in_hi:
                                    if delta > m_in:
                                        m_in = delta
                                    if m_in >= m_left:
                                        if x_after_shi == 0 or delta + rem_x_all <= m_in:
                                            return True
                                    elif delta + rem_x_in < m_left:
                                        return False
                                else:
                                    if m_in < m_left or delta > m_in:
                                        return False
                                    if delta + rem_x_all <= m_in:
                                        return True
                    else:  # op_j == "X"
                        for pos in range(q_j, seg_e):
                            rem_x_all -= 1
                            if pos < s_hi:
                                rem_x_in -= 1
                            xq = x_map_b.get(pos)
                            if (
                                xq is not None
                                and xq[1] == q_full[pos - q_il]
                                and q_full[pos - l_il - q_il] == xq[0]
                            ):
                                delta += 1
                            s_after = pos - l_il + 1
                            if s_after < s_lo:
                                if delta > m_left:
                                    m_left = delta
                                if delta + rem_x_in < m_left:
                                    return False
                            elif s_after <= s_in_hi:
                                if delta > m_in:
                                    m_in = delta
                                if m_in >= m_left:
                                    if x_after_shi == 0 or delta + rem_x_all <= m_in:
                                        return True
                                elif delta + rem_x_in < m_left:
                                    return False
                            else:
                                if m_in < m_left or delta > m_in:
                                    return False
                                if delta + rem_x_all <= m_in:
                                    return True
                return m_in >= m_left

            def _check_outer_right_inject(
                idx_ir: int, q_ir: int, l_ir: int, s_lo: int, s_hi: int
            ) -> bool:
                if l_ir <= 0 or l_ir > pad or s_lo + l_ir > s_hi:
                    return False
                # Phase 1: Check if a strictly better placement exists RIGHT of q_ir
                mx_r_end = q_ir + l_ir
                mx_r_idx = idx_ir + 1
                x_right = 0
                j_r = idx_ir + 1
                while j_r < hi:
                    q_j, op_j, len_j = seg_ops[j_r]
                    if op_j == "D" and q_j == mx_r_end:
                        mx_r_idx = j_r + 1
                        j_r += 1
                        continue
                    if op_j not in ("M", "X") or q_j != mx_r_end:
                        break
                    if op_j == "X":
                        x_right += len_j
                    mx_r_end = q_j + len_j
                    mx_r_idx = j_r + 1
                    j_r += 1

                m_outside_r = 0
                if x_right > 0 and mx_r_end > q_ir + l_ir:
                    r_eff_b = mx_r_end
                    m_mm_r = 0
                    for j_op in range(idx_ir + 1, mx_r_idx):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if op_j != "M":
                            continue
                        seg_s = max(q_j, q_ir + l_ir)
                        seg_e = min(q_j + len_j, mx_r_end)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s - l_ir, seg_e - l_ir)
                        s_b = _slice_q_up(seg_s, seg_e)
                        if s_a is None or s_b is None:
                            return False
                        if s_a != s_b:
                            broke = False
                            for off, (ca, cb) in enumerate(zip(s_a, s_b)):
                                if ca != cb:
                                    m_mm_r += 1
                                    if m_mm_r > x_right:
                                        r_eff_b = seg_s + off
                                        broke = True
                                        break
                            if broke:
                                break
                    if r_eff_b > q_ir + l_ir:
                        _, x_map_r = _get_dx_for(q_ir + l_ir, r_eff_b)
                        q_rs = _slice_q_up(q_ir, r_eff_b)
                        if q_rs is None:
                            return False
                        dr = 0
                        rem_xr = x_right
                        for j_op in range(idx_ir + 1, mx_r_idx):
                            q_j, op_j, len_j = seg_ops[j_op]
                            if op_j == "D":
                                continue
                            if q_j >= r_eff_b:
                                break
                            seg_s = max(q_j, q_ir + l_ir)
                            seg_e = min(q_j + len_j, r_eff_b)
                            if seg_s >= seg_e:
                                continue
                            span_k = seg_e - seg_s
                            if op_j == "M":
                                o_u = seg_s - l_ir - q_ir
                                o_p = seg_s - q_ir
                                if q_rs[o_u : o_u + span_k] != q_rs[o_p : o_p + span_k]:
                                    for off in range(span_k):
                                        if q_rs[o_u + off] != q_rs[o_p + off]:
                                            dr -= 1
                                            if dr + rem_xr <= m_outside_r:
                                                break
                                    if dr + rem_xr <= m_outside_r:
                                        break
                            else:  # op_j == "X"
                                for pos in range(seg_s, seg_e):
                                    rem_xr -= 1
                                    xq = x_map_r.get(pos)
                                    if (
                                        xq is not None
                                        and xq[1] == q_rs[pos - q_ir]
                                        and q_rs[pos - l_ir - q_ir] == xq[0]
                                    ):
                                        dr += 1
                                        if dr > m_outside_r:
                                            m_outside_r = dr
                                    if dr + rem_xr <= m_outside_r:
                                        break
                                if dr + rem_xr <= m_outside_r:
                                    break

                # Phase 2: Walk LEFT across full M/X/D run (including past s_lo)
                mx_l_start = q_ir
                has_x_after_eff = False
                x_after_slo = 0
                x_before_slo = 0
                j_start = idx_ir - 1
                while j_start >= lo:
                    q_j, op_j, len_j = seg_ops[j_start]
                    if op_j == "D":
                        if q_j != mx_l_start:
                            break
                        j_start -= 1
                        continue
                    if op_j not in ("M", "X") or q_j + len_j != mx_l_start:
                        break
                    if op_j == "X":
                        if mx_l_start > s_lo:
                            x_after_slo += mx_l_start - max(q_j, s_lo)
                        if q_j < s_lo:
                            x_before_slo += min(mx_l_start, s_lo) - q_j
                        if mx_l_start > s_hi - l_ir:
                            has_x_after_eff = True
                    mx_l_start = q_j
                    j_start -= 1
                if mx_l_start > s_hi - l_ir:
                    return False
                if m_outside_r == 0 and not has_x_after_eff and x_before_slo == 0:
                    q_sub = _slice_q_up(s_hi - l_ir, q_ir + l_ir)
                    if (
                        q_sub is not None
                        and len(q_sub) > l_ir
                        and q_sub[:-l_ir] == q_sub[l_ir:]
                    ):
                        return True
                    if x_after_slo == 0:
                        return False
                m_mm = 0
                eff_cut = s_hi - l_ir
                for j_op in range(idx_ir - 1, j_start, -1):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j + len_j <= eff_cut:
                        break
                    if op_j != "M":
                        continue
                    seg_s = max(q_j, eff_cut)
                    seg_e = min(q_j + len_j, q_ir)
                    s_a = _slice_q_up(seg_s, seg_e)
                    s_b = _slice_q_up(seg_s + l_ir, seg_e + l_ir)
                    if s_a is None or s_b is None:
                        return False
                    if s_a != s_b:
                        for off in range(seg_e - seg_s - 1, -1, -1):
                            if s_a[off] != s_b[off]:
                                m_mm += 1
                                if x_after_slo - m_mm < m_outside_r:
                                    return False
                l_lim = max(mx_l_start, s_lo) if x_before_slo == 0 else mx_l_start
                if x_before_slo > 0:
                    m_mm_all = m_mm
                    x_tot_l = x_after_slo + x_before_slo
                    for j_op in range(idx_ir - 1, j_start, -1):
                        q_j, op_j, len_j = seg_ops[j_op]
                        if op_j != "M" or q_j >= eff_cut:
                            continue
                        seg_s = max(q_j, mx_l_start)
                        seg_e = min(q_j + len_j, eff_cut)
                        if seg_s >= seg_e:
                            continue
                        s_a = _slice_q_up(seg_s, seg_e)
                        s_b = _slice_q_up(seg_s + l_ir, seg_e + l_ir)
                        if s_a is None or s_b is None:
                            return False
                        if s_a != s_b:
                            broke = False
                            for off in range(seg_e - seg_s - 1, -1, -1):
                                if s_a[off] != s_b[off]:
                                    m_mm_all += 1
                                    if x_tot_l - m_mm_all < m_outside_r:
                                        l_lim = seg_s + off + 1
                                        broke = True
                                        break
                            if broke:
                                break
                    if l_lim > s_hi - l_ir:
                        return False
                _, x_map_b = _get_dx_for(l_lim, q_ir)
                q_full = _slice_q_up(l_lim, q_ir + l_ir)
                if q_full is None:
                    return False
                delta = 0
                m_right = m_outside_r
                m_in = -(10**9)
                rem_x_in = x_after_slo
                rem_x_all = x_after_slo + x_before_slo
                s_in_lim = s_hi - l_ir
                for j_op in range(idx_ir - 1, j_start, -1):
                    q_j, op_j, len_j = seg_ops[j_op]
                    if op_j == "D":
                        continue
                    if q_j + len_j <= l_lim:
                        break
                    seg_s = max(q_j, l_lim)
                    seg_e = min(q_j + len_j, q_ir)
                    if seg_s >= seg_e:
                        continue
                    span_k = seg_e - seg_s
                    if op_j == "M":
                        o_u = seg_s - l_lim
                        o_p = seg_s + l_ir - l_lim
                        if q_full[o_u : o_u + span_k] == q_full[o_p : o_p + span_k]:
                            ra = seg_s
                            rb = seg_e - 1
                            if rb > s_in_lim and delta > m_right:
                                m_right = delta
                            if ra <= s_in_lim and rb >= s_lo:
                                if delta > m_in:
                                    m_in = delta
                                if m_in >= m_right:
                                    if x_before_slo == 0 or delta + rem_x_all <= m_in:
                                        return True
                                elif delta + rem_x_in < m_right:
                                    return False
                            if ra < s_lo:
                                if m_in < m_right or delta > m_in:
                                    return False
                                if delta + rem_x_all <= m_in:
                                    return True
                        else:
                            for off in range(span_k - 1, -1, -1):
                                if q_full[o_u + off] != q_full[o_p + off]:
                                    delta -= 1
                                s_after = seg_s + off
                                if s_after > s_in_lim:
                                    if delta > m_right:
                                        m_right = delta
                                    if delta + rem_x_in < m_right:
                                        return False
                                elif s_after >= s_lo:
                                    if delta > m_in:
                                        m_in = delta
                                    if m_in >= m_right:
                                        if x_before_slo == 0 or delta + rem_x_all <= m_in:
                                            return True
                                    elif delta + rem_x_in < m_right:
                                        return False
                                else:
                                    if m_in < m_right or delta > m_in:
                                        return False
                                    if delta + rem_x_all <= m_in:
                                        return True
                    else:  # op_j == "X"
                        for u_pos in range(seg_e - 1, seg_s - 1, -1):
                            rem_x_all -= 1
                            if u_pos >= s_lo:
                                rem_x_in -= 1
                            xq = x_map_b.get(u_pos)
                            if (
                                xq is not None
                                and xq[1] == q_full[u_pos - l_lim]
                                and q_full[u_pos + l_ir - l_lim] == xq[0]
                            ):
                                delta += 1
                            if u_pos > s_in_lim:
                                if delta > m_right:
                                    m_right = delta
                                if delta + rem_x_in < m_right:
                                    return False
                            elif u_pos >= s_lo:
                                if delta > m_in:
                                    m_in = delta
                                if m_in >= m_right:
                                    if x_before_slo == 0 or delta + rem_x_all <= m_in:
                                        return True
                                elif delta + rem_x_in < m_right:
                                    return False
                            else:
                                if m_in < m_right or delta > m_in:
                                    return False
                                if delta + rem_x_all <= m_in:
                                    return True
                return m_in >= m_right

            # Symmetric two-sided outer-I evaluation & tie-break (BS-7)
            out_l = _find_outer_left_i(op_lo, op_hi)
            out_r = _find_outer_right_i(op_lo, op_hi)
            cand_l = 0
            cand_r = 0
            if out_l is not None:
                idx_il, q_il, l_il = out_l
                if _check_outer_left_inject(idx_il, q_il, l_il, scan_lo, scan_hi):
                    cand_l = l_il
            if out_r is not None:
                idx_ir, q_ir, l_ir = out_r
                if _check_outer_right_inject(idx_ir, q_ir, l_ir, scan_lo, scan_hi):
                    cand_r = l_ir
            if cand_l > 0 and cand_r > 0:
                if scan_lo + cand_l + cand_r <= scan_hi:
                    inject_left_i = cand_l
                    inject_right_i = cand_r
                    has_local_indel = True
                elif cand_l > cand_r:
                    inject_left_i = cand_l
                    has_local_indel = True
                elif cand_r > cand_l:
                    inject_right_i = cand_r
                    has_local_indel = True
            elif cand_l > 0:
                inject_left_i = cand_l
                has_local_indel = True
            elif cand_r > 0:
                inject_right_i = cand_r
                has_local_indel = True

        if (
            p_len >= shift_w_size
            or not has_local_indel
            or query_seq is None
            or q_seq_len < seg_q_en
            or scan_lo >= scan_hi
        ):
            return _eval_raw_segment(
                seg_ops, op_lo, op_hi, seg_q_st, seg_q_en, m_p, x_p, i_p, d_p
            )

        q_loc = _slice_q_up(scan_lo, scan_hi)
        if q_loc is None:
            return _eval_raw_segment(
                seg_ops, op_lo, op_hi, seg_q_st, seg_q_en, m_p, x_p, i_p, d_p
            )
        d_seqs, x_map = _get_dx_for(scan_lo, scan_hi)
        n_q = scan_hi - scan_lo
        t_loc: list[str | None] = []
        indels: list[tuple[str, int, int, bool]] = []
        q_cur = scan_lo
        contiguous = True
        eff_lo = scan_lo + inject_left_i
        eff_hi = scan_hi - inject_right_i
        if inject_left_i > 0:
            indels.append(("I", inject_left_i, 0, False))
            q_cur = eff_lo
        for idx_op in range(op_lo, op_hi):
            q_p, op, sub_len = seg_ops[idx_op]
            if op == "D":
                if eff_lo < q_p < eff_hi:
                    if q_p != q_cur:
                        contiguous = False
                        break
                    d_seq = d_seqs.get((q_p, sub_len))
                    if d_seq is not None:
                        t_loc.extend(d_seq)
                        indels.append(("D", sub_len, q_p - scan_lo, False))
                    else:
                        t_loc.extend([None] * sub_len)
                        indels.append(("D", sub_len, q_p - scan_lo, True))
            else:
                ov_s = max(q_p, eff_lo)
                ov_e = min(q_p + sub_len, eff_hi)
                if ov_s >= ov_e:
                    continue
                if ov_s != q_cur:
                    contiguous = False
                    break
                if op == "M":
                    t_loc.extend(q_loc[ov_s - scan_lo : ov_e - scan_lo])
                elif op == "X":
                    for qc in range(ov_s, ov_e):
                        xq = x_map.get(qc)
                        if xq is not None and xq[1] == q_loc[qc - scan_lo]:
                            t_loc.append(xq[0])
                        else:
                            t_loc.append(None)
                else:  # op == "I"
                    if q_p < eff_lo or q_p + sub_len > eff_hi:
                        contiguous = False
                        break
                    indels.append(("I", sub_len, q_p - scan_lo, False))
                q_cur = ov_e
        if contiguous and inject_right_i > 0:
            if q_cur != eff_hi:
                contiguous = False
            else:
                indels.append(("I", inject_right_i, eff_hi - scan_lo, False))
                q_cur = scan_hi

        if not contiguous or q_cur != scan_hi or not indels:
            return _eval_raw_segment(
                seg_ops, op_lo, op_hi, seg_q_st, seg_q_en, m_p, x_p, i_p, d_p
            )

        n_t = len(t_loc)
        k_cnt = len(indels)

        _diag_pref_cache: dict[int, list[int]] = {}
        _t_loc_str = "".join(c if c is not None else "\x00" for c in t_loc)
        _acc = itertools.accumulate
        _add = operator.add
        _sub = operator.sub
        _eq = operator.eq
        _ne = operator.ne
        _n_orbit_perms = 1
        if 2 <= k_cnt <= 4:
            if not any(x[3] for x in indels):
                _n_i_orb = sum(1 for x in indels if x[0] == "I")
                if 0 < _n_i_orb < k_cnt:
                    _n_orbit_perms = math.comb(k_cnt, _n_i_orb)
            elif (
                sum(
                    1
                    for j_sw in range(k_cnt - 1)
                    if indels[j_sw][0] != indels[j_sw + 1][0]
                    and not indels[j_sw][3]
                    and not indels[j_sw + 1][3]
                )
                == 1
            ):
                _n_orbit_perms = 2

        def _eval_indels_order(
            indels: list[tuple[str, int, int, bool]],
            min_m_max: int = -1,
        ) -> tuple[int, bool, bool]:
            offs = [0] * (k_cnt + 1)
            for j in range(k_cnt):
                op_j, l_j, _, _ = indels[j]
                offs[j + 1] = offs[j] + (l_j if op_j == "I" else -l_j)
            if n_q - n_t != offs[k_cnt]:
                return -2, False, False

            q_lo_d = [max(0, o) for o in offs]
            q_hi_d = [min(n_q, n_t + o) for o in offs]
            total_cells = sum(q_hi_d[j] - q_lo_d[j] + 1 for j in range(k_cnt + 1))
            if total_cells > 16384 or k_cnt > 12:
                return -2, False, False

            # Precompute match prefix sums along each diagonal j = 0 .. K
            # (cached by offset o_j across Component C indel-order permutations)
            diag_pref: list[list[int]] = []
            for j in range(k_cnt + 1):
                o_j = offs[j]
                pref_j = _diag_pref_cache.get(o_j)
                if pref_j is None:
                    lo_j = q_lo_d[j]
                    hi_j = q_hi_d[j]
                    if hi_j <= lo_j:
                        pref_j = [0] if hi_j == lo_j else []
                    else:
                        q_sl = q_loc[lo_j:hi_j]
                        t_sl = _t_loc_str[lo_j - o_j : hi_j - o_j]
                        if q_sl == t_sl:
                            pref_j = list(range(hi_j - lo_j + 1))
                        else:
                            pref_j = list(_acc(map(_eq, q_sl, t_sl), initial=0))
                    _diag_pref_cache[o_j] = pref_j
                diag_pref.append(pref_j)

            def _diag_m(j: int, a: int, b: int) -> int:
                if a >= b:
                    return 0
                lo_j = q_lo_d[j]
                pj = diag_pref[j]
                return pj[b - lo_j] - pj[a - lo_j]

            # Exact Forward F & Backward B DP over the K+1 diagonals
            neg_inf = -(10**9)
            f_dp: list[list[int]] = [list(diag_pref[0])]
            for j in range(1, k_cnt + 1):
                op_j, l_j, orig_i, pinned_j = indels[j - 1]
                prev_f = f_dp[j - 1]
                prev_lo = q_lo_d[j - 1]
                prev_hi = q_hi_d[j - 1]
                cur_lo = q_lo_d[j]
                cur_hi = q_hi_d[j]
                pj = diag_pref[j]
                cur_len = cur_hi - cur_lo + 1
                cur_f = [neg_inf] * cur_len
                if pinned_j:
                    s_lo = orig_i
                    s_hi = orig_i
                elif op_j == "D":
                    s_lo = max(prev_lo, cur_lo)
                    s_hi = min(prev_hi, cur_hi)
                else:
                    s_lo = max(prev_lo, cur_lo - l_j)
                    s_hi = min(prev_hi, cur_hi - l_j)
                shift_e = l_j if op_j == "I" else 0
                if s_lo <= s_hi:
                    p0 = s_lo - prev_lo
                    e0 = s_lo + shift_e - cur_lo
                    cnt_s = s_hi - s_lo + 1
                    e1 = e0 + cnt_s
                    cur_f[e0:e1] = prev_f[p0 : p0 + cnt_s]
                    best_g = neg_inf
                    for idx in range(e0, e1):
                        v = cur_f[idx]
                        p = pj[idx]
                        if v > neg_inf:
                            g = v - p
                            if g > best_g:
                                best_g = g
                        if best_g > neg_inf:
                            cur_f[idx] = best_g + p
                    if best_g > neg_inf and e1 < cur_len:
                        cur_f[e1:] = map(_add, itertools.repeat(best_g), pj[e1:])
                f_dp.append(cur_f)

            m_max = f_dp[k_cnt][n_q - q_lo_d[k_cnt]]
            if m_max < 0:
                return -2, False, False
            if m_max < min_m_max:
                return m_max, False, False

            b_dp: list[list[int]] = [[] for _ in range(k_cnt + 1)]
            pk = diag_pref[k_cnt]
            b_dp[k_cnt] = list(map(_sub, itertools.repeat(pk[-1]), pk))
            for j in range(k_cnt - 1, -1, -1):
                op_j, l_j, orig_i, pinned_j = indels[j]
                next_b = b_dp[j + 1]
                next_lo = q_lo_d[j + 1]
                next_hi = q_hi_d[j + 1]
                cur_lo = q_lo_d[j]
                cur_hi = q_hi_d[j]
                pj = diag_pref[j]
                cur_b = [neg_inf] * (cur_hi - cur_lo + 1)
                if pinned_j:
                    s_lo = orig_i
                    s_hi = orig_i
                elif op_j == "D":
                    s_lo = max(cur_lo, next_lo)
                    s_hi = min(cur_hi, next_hi)
                else:
                    s_lo = max(cur_lo, next_lo - l_j)
                    s_hi = min(cur_hi, next_hi - l_j)
                shift_e = l_j if op_j == "I" else 0
                if s_lo <= s_hi:
                    s0 = s_lo - cur_lo
                    e0 = s_lo + shift_e - next_lo
                    cnt_s = s_hi - s_lo + 1
                    s1 = s0 + cnt_s
                    cur_b[s0:s1] = next_b[e0 : e0 + cnt_s]
                    best_h = neg_inf
                    for idx in range(s1 - 1, s0 - 1, -1):
                        v = cur_b[idx]
                        p = pj[idx]
                        if v > neg_inf:
                            h = v + p
                            if h > best_h:
                                best_h = h
                        if best_h > neg_inf:
                            cur_b[idx] = best_h - p
                    if best_h > neg_inf and s0 > 0:
                        cur_b[:s0] = map(_sub, itertools.repeat(best_h), pj[:s0])
                b_dp[j] = cur_b

            # Exact marginal optimal placement sets S*_j
            s_opt: list[list[int]] = []
            for j in range(k_cnt):
                op_j, l_j, orig_i, pinned_j = indels[j]
                prev_f = f_dp[j]
                next_b = b_dp[j + 1]
                prev_lo = q_lo_d[j]
                prev_hi = q_hi_d[j]
                next_lo = q_lo_d[j + 1]
                next_hi = q_hi_d[j + 1]
                if pinned_j:
                    s_lo = orig_i
                    s_hi = orig_i
                elif op_j == "D":
                    s_lo = max(prev_lo, next_lo)
                    s_hi = min(prev_hi, next_hi)
                else:
                    s_lo = max(prev_lo, next_lo - l_j)
                    s_hi = min(prev_hi, next_hi - l_j)
                shift_e = l_j if op_j == "I" else 0
                if s_lo <= s_hi:
                    p0 = s_lo - prev_lo
                    e0 = s_lo + shift_e - next_lo
                    cnt_s = s_hi - s_lo + 1
                    opt_j = [
                        s_lo + off
                        for off, tot in enumerate(
                            map(
                                _add,
                                prev_f[p0 : p0 + cnt_s],
                                next_b[e0 : e0 + cnt_s],
                            )
                        )
                        if tot == m_max
                    ]
                else:
                    opt_j = []
                if not opt_j:
                    return -2, False, False
                s_opt.append(opt_j)

            # Classify each indel purely from its symmetric optimal set S*_j
            r_p_s = p_s - scan_lo
            r_p_e = p_e - scan_lo
            p_cov = min(r_p_e, n_q) - max(r_p_s, 0)
            if p_cov * 10 < 9 * p_len:
                return m_max, False, False

            p_allowed: list[list[int]] = []
            w_allowed: list[list[int]] = []
            split_inesc_indices: list[tuple[int, tuple[int, int]]] = []
            s_cover_indices: list[tuple[int, int, list[int]]] = []
            has_tandem = False
            force_window = False
            has_inesc_i = False
            reject_early = False

            for j in range(k_cnt):
                op_j, l_j, _orig_i, _pinned_j = indels[j]
                opt_j = s_opt[j]
                if op_j == "D":
                    p_allowed.append(opt_j)
                    w_allowed.append(opt_j)
                    if any(r_p_s < s < r_p_e for s in opt_j):
                        force_window = True
                else:
                    ovs = [max(0, min(s + l_j, r_p_e) - max(s, r_p_s)) for s in opt_j]
                    min_ov = min(ovs)
                    max_ov = max(ovs)
                    if min_ov > 0:
                        has_inesc_i = True
                        if min_hit_len > 0 and l_j >= min_hit_len:
                            reject_early = True  # §3A.1 protected_ins
                        elif (p_cov - max_ov) * 10 < 9 * p_len:
                            reject_early = True
                        else:
                            s_max = [s for s, ov in zip(opt_j, ovs) if ov == max_ov]
                            force_window = True
                            if len(s_max) > 1 and max_ov < min(l_j, p_len):
                                ep = (s_max[0], s_max[-1])
                                split_inesc_indices.append((j, ep))
                                p_allowed.append([ep[0]])
                                w_allowed.append([ep[0]])
                            else:
                                p_allowed.append(s_max)
                                w_allowed.append(s_max)
                    else:
                        s_out = [s for s, ov in zip(opt_j, ovs) if ov == 0]
                        s_in = [s for s in opt_j if r_p_s <= s and s + l_j <= r_p_e]
                        if s_in:
                            x_outs: list[int] = []
                            if l_j == 1:
                                lo_j = q_lo_d[j]
                                hi_j = q_hi_d[j]
                                lo_j1 = q_lo_d[j + 1]
                                hi_j1 = q_hi_d[j + 1]
                                fj = f_dp[j]
                                bj1 = b_dp[j + 1]
                                s_in_r = s_in[-1]
                                s_in_l = s_in[0]
                                for s_o in s_out:
                                    if s_o >= r_p_e:
                                        if (
                                            s_in_r >= lo_j
                                            and s_o <= hi_j
                                            and r_p_e <= hi_j
                                            and fj[s_in_r - lo_j]
                                            + _diag_m(j, s_in_r, s_o)
                                            + bj1[s_o + 1 - lo_j1]
                                            == m_max
                                            and _diag_m(j, s_in_r, r_p_e)
                                            == r_p_e - s_in_r
                                            and _diag_m(j, r_p_e, s_o) < s_o - r_p_e
                                        ):
                                            x_outs.append(s_o)
                                    else:  # s_o + 1 <= r_p_s
                                        if (
                                            s_o + 1 >= lo_j1
                                            and s_in_l + 1 <= hi_j1
                                            and r_p_s >= lo_j1
                                            and fj[s_o - lo_j]
                                            + _diag_m(j + 1, s_o + 1, s_in_l + 1)
                                            + bj1[s_in_l + 1 - lo_j1]
                                            == m_max
                                            and _diag_m(j + 1, r_p_s, s_in_l + 1)
                                            == (s_in_l + 1) - r_p_s
                                            and _diag_m(j + 1, s_o + 1, r_p_s)
                                            < r_p_s - (s_o + 1)
                                        ):
                                            x_outs.append(s_o)
                            if x_outs:
                                force_window = True
                                p_allowed.append(x_outs)
                                w_allowed.append(x_outs)
                            else:
                                has_tandem = True
                                force_window = True
                                p_allowed.append(s_out)
                                w_allowed.append(s_in)
                        else:
                            s_cover: list[int] = []
                            if l_j > p_len:
                                cand_cov = [
                                    s
                                    for s in opt_j
                                    if s <= r_p_s and s + l_j >= r_p_e
                                ]
                                if cand_cov:
                                    sc_span = opt_j[-1] - opt_j[0]
                                    if sc_span > 0:
                                        sc_a = opt_j[0]
                                        sc_b = opt_j[-1]
                                        m_sh0 = _diag_m(j, sc_a, sc_b)
                                        m_sh1 = _diag_m(j + 1, sc_a + l_j, sc_b + l_j)
                                        if min(m_sh0, m_sh1) * 2 > sc_span:
                                            if sc_span >= l_j:
                                                s_cover = cand_cov
                                            elif (
                                                min_hit_len > 0 and l_j >= min_hit_len
                                            ) or l_j > max_w_i_budget:
                                                tm = _tandem_matches(
                                                    cand_cov[0] + scan_lo, l_j
                                                )
                                                if cand_cov[-1] != cand_cov[0]:
                                                    tm = max(
                                                        tm,
                                                        _tandem_matches(
                                                            cand_cov[-1] + scan_lo, l_j
                                                        ),
                                                    )
                                                if tm * 10 >= 7 * l_j:
                                                    s_cover = cand_cov
                            if s_cover:
                                has_tandem = True
                                if (min_hit_len > 0 and l_j >= min_hit_len) or (
                                    l_j > max_w_i_budget
                                ):
                                    reject_early = True
                                else:
                                    force_window = True
                                    p_allowed.append(s_out)
                                    w_allowed.append(s_cover)
                                    s_cover_indices.append((j, l_j, s_cover))
                            else:
                                p_allowed.append(s_out)
                                w_allowed.append(s_out)
                                if max_ov > 0:
                                    force_window = True

            if not reject_early and s_cover_indices:
                if (
                    len(s_cover_indices) >= 2
                    or sum(lj_c for _, lj_c, _ in s_cover_indices) > max_w_i_budget
                ):
                    reject_early = True

            # Exact universal window I-budget early reject: any window covering P
            # must include at least min_ov_p(j) inserted bases per I op in w_allowed.
            if not reject_early and (
                has_tandem or force_window or (min_hit_len > 0 and p_len < min_hit_len)
            ):
                min_forced_i = 0
                for j_fi in range(k_cnt):
                    op_fi, l_fi, _, _ = indels[j_fi]
                    if op_fi == "I" and w_allowed[j_fi]:
                        min_forced_i += min(
                            max(0, min(s_fi + l_fi, r_p_e) - max(s_fi, r_p_s))
                            for s_fi in w_allowed[j_fi]
                        )
                if min_forced_i > max_w_i_budget:
                    reject_early = True

            should_mark_ins_block = (
                ins_block_t_pieces_out is not None
                and seg_spans_p
                and (has_inesc_i or has_tandem)
            )
            if reject_early:
                return m_max, False, should_mark_ins_block

            def _eval_k1(
                p_set: list[int], w_set: list[int], req_lo: int, req_hi: int
            ) -> bool:
                op_0, l_0, _, _ = indels[0]
                r_pa = max(r_p_s, 0)
                r_pb = min(r_p_e, n_q)

                def _stats_on(u: int, v: int, s: int) -> tuple[int, int, int, int]:
                    if op_0 == "D":
                        mid = min(max(s, u), v)
                        m_c = _diag_m(0, u, mid) + _diag_m(1, mid, v)
                        d_c = l_0 if u < s < v else 0
                        return m_c, (v - u) - m_c, 0, d_c
                    e = s + l_0
                    mid_l = min(max(s, u), v)
                    mid_r = min(max(e, u), v)
                    m_c = _diag_m(0, u, mid_l) + _diag_m(1, mid_r, v)
                    i_c = max(0, min(e, v) - max(s, u))
                    return m_c, (v - u) - i_c - m_c, i_c, 0

                def _p_status(s: int) -> int:
                    m_c, x_c, i_c, d_c = _stats_on(r_pa, r_pb, s)
                    if (m_c + x_c) * 10 < 9 * p_len:
                        return 0
                    den = p_len + d_c
                    if den <= 0:
                        return 0
                    if (m_c + 1e-12) >= min_identity * den:
                        return 3 if (i_c == 0 and d_c == 0) else 2
                    if i_c == 0 and d_c == 0:
                        return 1
                    return 0

                def _window_ok_for(s: int, w_size: int) -> bool:
                    if (seg_q_en - seg_q_st) < w_size:
                        return False
                    w_lo = max(min(req_lo, seg_q_st), req_hi - w_size)
                    w_hi = min(req_lo, max(req_hi, seg_q_en) - w_size)
                    if w_lo > w_hi:
                        return False
                    for w_s in range(w_lo, w_hi + 1):
                        w_e = w_s + w_size
                        u = max(w_s, seg_q_st) - scan_lo
                        v = min(w_e, seg_q_en) - scan_lo
                        m_c, x_c, _i_c, d_c = _stats_on(u, v, s)
                        if (m_c + x_c) * 10 < 9 * w_size:
                            continue
                        den = w_size + d_c
                        if den > 0 and (m_c + 1e-12) >= min_identity * den:
                            return True
                    return False

                if has_tandem:
                    if not any(_p_status(s) > 0 for s in p_set):
                        return False
                    return all(_window_ok_for(s, shift_w_size) for s in w_set)

                for s in w_set:
                    st = _p_status(s)
                    if st == 0:
                        continue
                    if st >= 2:
                        if not force_window and (min_hit_len <= 0 or p_len >= min_hit_len):
                            return True
                        w_size = shift_w_size if force_window else min_hit_len
                        if _window_ok_for(s, w_size):
                            return True
                        if st == 3 and w_size < shift_w_size and _window_ok_for(s, shift_w_size):
                            return True
                    elif _window_ok_for(s, shift_w_size):
                        return True
                return False

            _opt_break_pref_cache: list[list[int]] | None = None
            _p_eval_cache: dict[
                tuple[int, ...],
                tuple[
                    tuple[list[int], ...],
                    list[bytearray],
                    list[bytearray],
                    list[list[int]],
                    list[list[int]],
                    bool,
                    bool | None,
                ],
            ] = {}
            _n_work_leaves = 1

            def _eval_k_general(
                p_sets: list[list[int]],
                w_sets: list[list[int]],
                req_lo: int,
                req_hi: int,
            ) -> bool:
                def _prepare_reach(
                    allowed_sets: list[list[int]],
                ) -> tuple[
                    list[bytearray],
                    list[bytearray],
                    list[list[int]],
                    list[list[int]],
                ]:
                    nonlocal _opt_break_pref_cache
                    if _opt_break_pref_cache is None:
                        _rep_m = itertools.repeat(m_max)
                        _bp_all: list[list[int]] = []
                        for j_b in range(k_cnt + 1):
                            span_b = q_hi_d[j_b] - q_lo_d[j_b]
                            if span_b > 0:
                                pj_b = diag_pref[j_b]
                                fj_b = f_dp[j_b]
                                bj_b = b_dp[j_b]
                                _bp_all.append(
                                    list(
                                        _acc(
                                            map(
                                                _ne,
                                                map(
                                                    _add,
                                                    map(_sub, fj_b[:-1], pj_b[:-1]),
                                                    map(_add, bj_b[1:], pj_b[1:]),
                                                ),
                                                _rep_m,
                                            ),
                                            initial=0,
                                        )
                                    )
                                )
                            else:
                                _bp_all.append([0] if span_b == 0 else [])
                        _opt_break_pref_cache = _bp_all
                    opt_break_pref = _opt_break_pref_cache

                    reach_p: list[bytearray] = [
                        bytearray(q_hi_d[j] - q_lo_d[j] + 1)
                        for j in range(k_cnt + 1)
                    ]
                    if f_dp[0][0] + b_dp[0][0] == m_max:
                        reach_p[0][0] = 1
                    for j in range(k_cnt + 1):
                        rp_j = reach_p[j]
                        bp_j = opt_break_pref[j]
                        span_j = q_hi_d[j] - q_lo_d[j]
                        if span_j > 0:
                            pos = rp_j.find(1, 0)
                            while 0 <= pos < span_j:
                                end_c = bisect.bisect_right(
                                    bp_j, bp_j[pos], pos + 1
                                )
                                if end_c > pos + 1:
                                    rp_j[pos + 1 : end_c] = b"\x01" * (
                                        end_c - pos - 1
                                    )
                                pos = rp_j.find(1, end_c)
                        if j < k_cnt:
                            a_j = allowed_sets[j]
                            if not a_j:
                                continue
                            op_j, l_j, _, _ = indels[j]
                            shift_e = l_j if op_j == "I" else 0
                            lo_j = q_lo_d[j]
                            lo_j1 = q_lo_d[j + 1]
                            rp_j1 = reach_p[j + 1]
                            cnt_a = len(a_j)
                            if cnt_a > 1 and a_j[-1] - a_j[0] == cnt_a - 1:
                                s0 = a_j[0] - lo_j
                                e0 = a_j[0] + shift_e - lo_j1
                                if rp_j.find(0, s0, s0 + cnt_a) == -1:
                                    rp_j1[e0 : e0 + cnt_a] = b"\x01" * cnt_a
                                    continue
                                if rp_j.find(1, s0, s0 + cnt_a) == -1:
                                    continue
                            for s in a_j:
                                if rp_j[s - lo_j]:
                                    rp_j1[s + shift_e - lo_j1] = 1

                    reach_s: list[bytearray] = [
                        bytearray(q_hi_d[j] - q_lo_d[j] + 1)
                        for j in range(k_cnt + 1)
                    ]
                    end_idx_k = n_q - q_lo_d[k_cnt]
                    if f_dp[k_cnt][end_idx_k] + b_dp[k_cnt][end_idx_k] == m_max:
                        reach_s[k_cnt][end_idx_k] = 1
                    opt_edges: list[list[int]] = [[] for _ in range(k_cnt)]
                    for j in range(k_cnt, -1, -1):
                        rs_j = reach_s[j]
                        bp_j = opt_break_pref[j]
                        span_j = q_hi_d[j] - q_lo_d[j]
                        if span_j > 0:
                            pos = rs_j.rfind(1, 0, span_j + 1)
                            while pos > 0:
                                start_c = bisect.bisect_left(
                                    bp_j, bp_j[pos], 0, pos
                                )
                                if start_c < pos:
                                    rs_j[start_c:pos] = b"\x01" * (pos - start_c)
                                if start_c <= 1:
                                    break
                                pos = rs_j.rfind(1, 0, start_c)
                        if j > 0:
                            a_p = allowed_sets[j - 1]
                            if not a_p:
                                continue
                            op_p, l_p, _, _ = indels[j - 1]
                            shift_e = l_p if op_p == "I" else 0
                            lo_p = q_lo_d[j - 1]
                            lo_c = q_lo_d[j]
                            rp_p = reach_p[j - 1]
                            rs_p = reach_s[j - 1]
                            cnt_p = len(a_p)
                            if cnt_p > 1 and a_p[-1] - a_p[0] == cnt_p - 1:
                                s0 = a_p[0] - lo_p
                                e0 = a_p[0] + shift_e - lo_c
                                if rs_j.find(0, e0, e0 + cnt_p) == -1:
                                    rs_p[s0 : s0 + cnt_p] = b"\x01" * cnt_p
                                    if rp_p.find(0, s0, s0 + cnt_p) == -1:
                                        opt_edges[j - 1] = a_p
                                        continue
                                    if rp_p.find(1, s0, s0 + cnt_p) == -1:
                                        continue
                                elif rs_j.find(1, e0, e0 + cnt_p) == -1:
                                    continue
                            edges_j: list[int] = []
                            for s in a_p:
                                if rs_j[s + shift_e - lo_c]:
                                    s_i = s - lo_p
                                    rs_p[s_i] = 1
                                    if rp_p[s_i]:
                                        edges_j.append(s)
                            opt_edges[j - 1] = edges_j
                    return reach_p, reach_s, opt_edges, opt_break_pref

                def _dag_best_on_uv(
                    u: int,
                    v: int,
                    target_len: int,
                    reach_p: list[bytearray],
                    reach_s: list[bytearray],
                    opt_edges: list[list[int]],
                    opt_break_pref: list[list[int]],
                    require_zero_indel: bool = False,
                ) -> bool:
                    cov_len = v - u
                    req_mx = (9 * target_len + 9) // 10
                    max_i = cov_len - req_mx
                    if max_i < 0:
                        return False
                    if require_zero_indel:
                        max_i = 0

                    # Exact upper bound on I bases overlapping [u, v)
                    max_pos_i = 0
                    for j_i in range(k_cnt):
                        op_i, l_i, _, _ = indels[j_i]
                        if op_i == "I":
                            ed_i = opt_edges[j_i]
                            if ed_i:
                                i_lo = bisect.bisect_right(ed_i, u - l_i)
                                if i_lo < len(ed_i) and ed_i[i_lo] < v:
                                    max_pos_i += min(l_i, cov_len)

                    neg_f = -1e18
                    half_neg = neg_f * 0.5
                    best_obj = neg_f

                    if max_pos_i <= max_i:
                        # Exact O(|active_edges|) two-pointer diagonal-jumping 1D DP
                        in_pts: list[tuple[int, float]] = (
                            [(u, 0.0)]
                            if (
                                q_lo_d[0] <= u <= q_hi_d[0]
                                and reach_p[0][u - q_lo_d[0]]
                            )
                            else []
                        )
                        for j in range(k_cnt + 1):
                            lo_j = q_lo_d[j]
                            hi_j = q_hi_d[j]
                            pj = diag_pref[j]
                            bp_j = opt_break_pref[j]
                            n_in = len(in_pts)
                            ip = 0
                            cur_brk = -1
                            best_adj = neg_f
                            next_in: list[tuple[int, float]] = []
                            if j < k_cnt:
                                lo_j1 = q_lo_d[j + 1]
                                hi_j1 = q_hi_d[j + 1]
                                if lo_j1 <= u <= hi_j1 and reach_p[j + 1][u - lo_j1]:
                                    next_in.append((u, 0.0))
                                op_n, l_n, _, _ = indels[j]
                                ed_n = opt_edges[j]
                                if op_n == "D":
                                    if not require_zero_indel and ed_n and n_in > 0:
                                        pen = -(min_identity * l_n)
                                        i_a = bisect.bisect_right(ed_n, u)
                                        i_b = bisect.bisect_left(ed_n, v)
                                        for idx_e in range(i_a, i_b):
                                            s = ed_n[idx_e]
                                            while ip < n_in and in_pts[ip][0] <= s:
                                                p_i, v_i = in_pts[ip]
                                                pi_off = p_i - lo_j
                                                b_i = bp_j[pi_off]
                                                adj_i = v_i - pj[pi_off]
                                                if b_i != cur_brk:
                                                    cur_brk = b_i
                                                    best_adj = adj_i
                                                elif adj_i > best_adj:
                                                    best_adj = adj_i
                                                ip += 1
                                            s_off = s - lo_j
                                            if cur_brk == bp_j[s_off] and best_adj > half_neg:
                                                next_in.append((s, best_adj + pj[s_off] + pen))
                                else:  # op_n == "I"
                                    if ed_n:
                                        i_a = bisect.bisect_right(ed_n, u - l_n)
                                        i_b = bisect.bisect_left(ed_n, v)
                                        for idx_e in range(i_a, i_b):
                                            s = ed_n[idx_e]
                                            e = s + l_n
                                            if s < u:
                                                if e <= v:
                                                    next_in.append((e, 0.0))
                                                elif 0.0 > best_obj:
                                                    best_obj = 0.0
                                            elif n_in > 0:
                                                while ip < n_in and in_pts[ip][0] <= s:
                                                    p_i, v_i = in_pts[ip]
                                                    pi_off = p_i - lo_j
                                                    b_i = bp_j[pi_off]
                                                    adj_i = v_i - pj[pi_off]
                                                    if b_i != cur_brk:
                                                        cur_brk = b_i
                                                        best_adj = adj_i
                                                    elif adj_i > best_adj:
                                                        best_adj = adj_i
                                                    ip += 1
                                                s_off = s - lo_j
                                                if cur_brk == bp_j[s_off] and best_adj > half_neg:
                                                    sv = best_adj + pj[s_off]
                                                    if e <= v:
                                                        next_in.append((e, sv))
                                                    elif sv > best_obj:
                                                        best_obj = sv
                            if n_in > 0 and lo_j <= v <= hi_j and reach_s[j][v - lo_j]:
                                while ip < n_in and in_pts[ip][0] <= v:
                                    p_i, v_i = in_pts[ip]
                                    pi_off = p_i - lo_j
                                    b_i = bp_j[pi_off]
                                    adj_i = v_i - pj[pi_off]
                                    if b_i != cur_brk:
                                        cur_brk = b_i
                                        best_adj = adj_i
                                    elif adj_i > best_adj:
                                        best_adj = adj_i
                                    ip += 1
                                v_off = v - lo_j
                                if cur_brk == bp_j[v_off] and best_adj > half_neg:
                                    ev = best_adj + pj[v_off]
                                    if ev > best_obj:
                                        best_obj = ev
                            in_pts = next_in
                    else:
                        # Exact O(|active_edges| * n_i) two-pointer diagonal-jumping 2D DP
                        n_i = max_i + 1
                        in_pts2: list[tuple[int, list[float]]] = []
                        if q_lo_d[0] <= u <= q_hi_d[0] and reach_p[0][u - q_lo_d[0]]:
                            row0 = [neg_f] * n_i
                            row0[0] = 0.0
                            in_pts2.append((u, row0))
                        for j in range(k_cnt + 1):
                            lo_j = q_lo_d[j]
                            hi_j = q_hi_d[j]
                            pj = diag_pref[j]
                            bp_j = opt_break_pref[j]
                            n_in = len(in_pts2)
                            ip = 0
                            cur_brk = -1
                            best_adj2: list[float] | None = None
                            next_in2: list[tuple[int, list[float]]] = []
                            if j < k_cnt:
                                lo_j1 = q_lo_d[j + 1]
                                hi_j1 = q_hi_d[j + 1]
                                if lo_j1 <= u <= hi_j1 and reach_p[j + 1][u - lo_j1]:
                                    r0 = [neg_f] * n_i
                                    r0[0] = 0.0
                                    next_in2.append((u, r0))
                                op_n, l_n, _, _ = indels[j]
                                ed_n = opt_edges[j]
                                if op_n == "D":
                                    if not require_zero_indel and ed_n and n_in > 0:
                                        pen = -(min_identity * l_n)
                                        i_a = bisect.bisect_right(ed_n, u)
                                        i_b = bisect.bisect_left(ed_n, v)
                                        for idx_e in range(i_a, i_b):
                                            s = ed_n[idx_e]
                                            while ip < n_in and in_pts2[ip][0] <= s:
                                                p_i, r_i = in_pts2[ip]
                                                pi_off = p_i - lo_j
                                                b_i = bp_j[pi_off]
                                                p_val = pj[pi_off]
                                                if b_i != cur_brk or best_adj2 is None:
                                                    cur_brk = b_i
                                                    best_adj2 = [
                                                        (rv - p_val if rv > half_neg else neg_f)
                                                        for rv in r_i
                                                    ]
                                                else:
                                                    for ic in range(n_i):
                                                        rv = r_i[ic]
                                                        if rv > half_neg:
                                                            av = rv - p_val
                                                            if av > best_adj2[ic]:
                                                                best_adj2[ic] = av
                                                ip += 1
                                            s_off = s - lo_j
                                            if cur_brk == bp_j[s_off] and best_adj2 is not None:
                                                p_s_val = pj[s_off] + pen
                                                dst = [
                                                    (av + p_s_val if av > half_neg else neg_f)
                                                    for av in best_adj2
                                                ]
                                                next_in2.append((s, dst))
                                else:  # op_n == "I"
                                    if ed_n:
                                        i_a = bisect.bisect_right(ed_n, u - l_n)
                                        i_b = bisect.bisect_left(ed_n, v)
                                        for idx_e in range(i_a, i_b):
                                            s = ed_n[idx_e]
                                            e = s + l_n
                                            if s < u:
                                                ov_i = min(e, v) - u
                                                if ov_i > max_i:
                                                    continue
                                                if e <= v:
                                                    dst = [neg_f] * n_i
                                                    dst[ov_i] = 0.0
                                                    next_in2.append((e, dst))
                                                elif 0.0 > best_obj:
                                                    best_obj = 0.0
                                            elif n_in > 0:
                                                ov_i = min(e, v) - s
                                                if ov_i > max_i:
                                                    continue
                                                while ip < n_in and in_pts2[ip][0] <= s:
                                                    p_i, r_i = in_pts2[ip]
                                                    pi_off = p_i - lo_j
                                                    b_i = bp_j[pi_off]
                                                    p_val = pj[pi_off]
                                                    if b_i != cur_brk or best_adj2 is None:
                                                        cur_brk = b_i
                                                        best_adj2 = [
                                                            (rv - p_val if rv > half_neg else neg_f)
                                                            for rv in r_i
                                                        ]
                                                    else:
                                                        for ic in range(n_i):
                                                            rv = r_i[ic]
                                                            if rv > half_neg:
                                                                av = rv - p_val
                                                                if av > best_adj2[ic]:
                                                                    best_adj2[ic] = av
                                                    ip += 1
                                                s_off = s - lo_j
                                                if cur_brk == bp_j[s_off] and best_adj2 is not None:
                                                    p_s_val = pj[s_off]
                                                    lim_ic = n_i - ov_i
                                                    if e <= v:
                                                        dst = [neg_f] * n_i
                                                        any_ok = False
                                                        for ic in range(lim_ic):
                                                            av = best_adj2[ic]
                                                            if av > half_neg:
                                                                dst[ic + ov_i] = av + p_s_val
                                                                any_ok = True
                                                        if any_ok:
                                                            next_in2.append((e, dst))
                                                    else:
                                                        for ic in range(lim_ic):
                                                            av = best_adj2[ic]
                                                            if av > half_neg:
                                                                sv = av + p_s_val
                                                                if sv > best_obj:
                                                                    best_obj = sv
                            if n_in > 0 and lo_j <= v <= hi_j and reach_s[j][v - lo_j]:
                                while ip < n_in and in_pts2[ip][0] <= v:
                                    p_i, r_i = in_pts2[ip]
                                    pi_off = p_i - lo_j
                                    b_i = bp_j[pi_off]
                                    p_val = pj[pi_off]
                                    if b_i != cur_brk or best_adj2 is None:
                                        cur_brk = b_i
                                        best_adj2 = [
                                            (rv - p_val if rv > half_neg else neg_f)
                                            for rv in r_i
                                        ]
                                    else:
                                        for ic in range(n_i):
                                            rv = r_i[ic]
                                            if rv > half_neg:
                                                av = rv - p_val
                                                if av > best_adj2[ic]:
                                                    best_adj2[ic] = av
                                    ip += 1
                                v_off = v - lo_j
                                if cur_brk == bp_j[v_off] and best_adj2 is not None:
                                    mx_av = max(best_adj2)
                                    if mx_av > half_neg:
                                        ev = mx_av + pj[v_off]
                                        if ev > best_obj:
                                            best_obj = ev
                            in_pts2 = next_in2

                    if best_obj <= half_neg:
                        return False
                    if require_zero_indel:
                        return True
                    return (best_obj + 1e-12) >= min_identity * target_len

                def _search_windows_k(
                    w_size_cur: int, strip_int_d: bool
                ) -> bool:
                    if (seg_q_en - seg_q_st) < w_size_cur:
                        return False
                    w_lo = max(min(req_lo, seg_q_st), req_hi - w_size_cur)
                    w_hi = min(req_lo, max(req_hi, seg_q_en) - w_size_cur)
                    if w_lo > w_hi:
                        return False
                    eff_w_sets = w_sets
                    if strip_int_d:
                        eff_w_sets = [
                            [s for s in ws if not (r_pa < s < r_pb)]
                            if indels[j][0] == "D"
                            else ws
                            for j, ws in enumerate(w_sets)
                        ]
                    if eff_w_sets == p_sets:
                        rp_w, rs_w, ed_w, bp_w = rp_p, rs_p, ed_p, bp_p
                    else:
                        rp_w, rs_w, ed_w, bp_w = _prepare_reach(eff_w_sets)
                    # Mirror-symmetric alternating endpoint window search with
                    # strictly min_identity-independent (w_sets-only, window-scoped,
                    # orbit*leaf-scaled) work guard preserving Attack 12-3 exactness
                    # and < 50 ms SLO across K=2..12 (1D & 2D).
                    win_u = max(w_lo, seg_q_st) - scan_lo
                    win_v = min(w_hi + w_size_cur, seg_q_en) - scan_lo
                    w_max_i = w_size_cur - ((9 * w_size_cur + 9) // 10)
                    sum_i_win = 0
                    n_edges_win = 0
                    for j_w in range(k_cnt):
                        op_w, l_w, _, _ = indels[j_w]
                        ws_j = w_sets[j_w]
                        if not ws_j:
                            continue
                        if op_w == "D":
                            ia = bisect.bisect_right(ws_j, win_u)
                            ib = bisect.bisect_left(ws_j, win_v)
                            if ib > ia:
                                n_edges_win += ib - ia
                        else:
                            ia = bisect.bisect_right(ws_j, win_u - l_w)
                            ib = bisect.bisect_left(ws_j, win_v)
                            if ib > ia:
                                n_edges_win += ib - ia
                                sum_i_win += min(l_w, w_size_cur)
                    eff_ni = (w_max_i + 1) if sum_i_win > w_max_i else 1
                    est_work = (
                        (w_hi - w_lo + 1)
                        * n_edges_win
                        * eff_ni
                        * _n_orbit_perms
                        * _n_work_leaves
                    )
                    use_work_cap = est_work > (
                        100000
                        if w_size_cur >= 70
                        else (300000 if w_size_cur > 50 else 1500000)
                    )
                    lw = w_lo
                    rw = w_hi
                    while lw <= rw:
                        for w_s in ((lw,) if lw == rw else (lw, rw)):
                            w_e = w_s + w_size_cur
                            u = max(w_s, seg_q_st) - scan_lo
                            v = min(w_e, seg_q_en) - scan_lo
                            if _dag_best_on_uv(
                                u, v, w_size_cur, rp_w, rs_w, ed_w, bp_w, False
                            ):
                                return True
                        if use_work_cap:
                            return False
                        lw += 1
                        rw -= 1
                    return False

                r_pa = max(r_p_s, 0)
                r_pb = min(r_p_e, n_q)
                p_key = tuple(id(s) for s in p_sets)
                cached_p = _p_eval_cache.get(p_key)
                if cached_p is None:
                    rp_p, rs_p, ed_p, bp_p = _prepare_reach(p_sets)
                    raw_p_ok = _dag_best_on_uv(
                        r_pa, r_pb, p_len, rp_p, rs_p, ed_p, bp_p, False
                    )
                    zero_p_ok: bool | None = None
                    p_ref = tuple(p_sets)
                    _p_eval_cache[p_key] = (
                        p_ref,
                        rp_p,
                        rs_p,
                        ed_p,
                        bp_p,
                        raw_p_ok,
                        zero_p_ok,
                    )
                else:
                    p_ref, rp_p, rs_p, ed_p, bp_p, raw_p_ok, zero_p_ok = cached_p

                def _is_zero_p_ok() -> bool:
                    nonlocal zero_p_ok
                    if zero_p_ok is None:
                        zero_p_ok = _dag_best_on_uv(
                            r_pa, r_pb, p_len, rp_p, rs_p, ed_p, bp_p, True
                        )
                        _p_eval_cache[p_key] = (
                            p_ref,
                            rp_p,
                            rs_p,
                            ed_p,
                            bp_p,
                            raw_p_ok,
                            zero_p_ok,
                        )
                    return zero_p_ok

                if raw_p_ok:
                    if not force_window and (min_hit_len <= 0 or p_len >= min_hit_len):
                        return True
                    w_size = shift_w_size if force_window else min_hit_len
                    if _search_windows_k(w_size, False):
                        return True
                    if w_size < shift_w_size and _is_zero_p_ok():
                        return _search_windows_k(shift_w_size, True)
                    return False
                if not _is_zero_p_ok():
                    return False
                return _search_windows_k(shift_w_size, True)

            def _eval_config(
                p_sets: list[list[int]],
                w_sets: list[list[int]],
                req_lo: int,
                req_hi: int,
            ) -> bool:
                if k_cnt == 1:
                    return _eval_k1(p_sets[0], w_sets[0], req_lo, req_hi)
                return _eval_k_general(p_sets, w_sets, req_lo, req_hi)

            if not split_inesc_indices and not s_cover_indices:
                seg_ok = _eval_config(p_allowed, w_allowed, p_s, p_e)
            elif len(split_inesc_indices) > 6:
                seg_ok = False
            else:
                # Mirror-symmetric leaf bound (<= 64 leaves for arbitrary --min-sv-len)
                bounded_s_cover: list[tuple[int, int, list[int]]] = []
                max_cov_per = max(2, 64 >> len(split_inesc_indices))
                _n_work_leaves = 1 << len(split_inesc_indices)
                for j_ind, l_ind, cov_set in s_cover_indices:
                    if len(cov_set) > max_cov_per:
                        half_c = max_cov_per // 2
                        cov_use = cov_set[:half_c] + cov_set[-half_c:]
                    else:
                        cov_use = cov_set
                    _n_work_leaves *= len(cov_use)
                    bounded_s_cover.append((j_ind, l_ind, cov_use))

                def _recurse_cover(
                    idx_c: int,
                    p_cur: list[list[int]],
                    w_cur: list[list[int]],
                    req_lo: int,
                    req_hi: int,
                ) -> bool:
                    if idx_c == len(bounded_s_cover):
                        return _eval_config(p_cur, w_cur, req_lo, req_hi)
                    j_ind, l_ind, cov_set = bounded_s_cover[idx_c]
                    for s_c in cov_set:
                        w_cur[j_ind] = [s_c]
                        if not _recurse_cover(
                            idx_c + 1,
                            p_cur,
                            w_cur,
                            min(req_lo, s_c + scan_lo),
                            max(req_hi, s_c + l_ind + scan_lo),
                        ):
                            return False
                    return True

                def _recurse_split(
                    idx_s: int, p_cur: list[list[int]], w_cur: list[list[int]]
                ) -> bool:
                    if idx_s == len(split_inesc_indices):
                        return _recurse_cover(0, p_cur, w_cur, p_s, p_e)
                    j_ind, (ep0, ep1) = split_inesc_indices[idx_s]
                    p_cur[j_ind] = [ep0]
                    w_cur[j_ind] = [ep0]
                    if not _recurse_split(idx_s + 1, p_cur, w_cur):
                        return False
                    p_cur[j_ind] = [ep1]
                    w_cur[j_ind] = [ep1]
                    return _recurse_split(idx_s + 1, p_cur, w_cur)

                seg_ok = _recurse_split(0, list(p_allowed), list(w_allowed))

            return m_max, seg_ok, should_mark_ins_block

        m_max_0, seg_ok, mark_blk = _eval_indels_order(indels)
        if m_max_0 == -2:
            return _eval_raw_segment(
                seg_ops, op_lo, op_hi, seg_q_st, seg_q_en, m_p, x_p, i_p, d_p
            )

        # Component C: Complete mirror-invariant unpinned I/D interleaving orbit
        # closure for 2 <= K <= 4 (at most C(4,2) = 6 orderings total, skipping
        # suboptimal orderings right after F DP via min_m_max).
        if 2 <= k_cnt <= 4:
            if not any(x[3] for x in indels):
                i_ops = [x for x in indels if x[0] == "I"]
                d_ops = [x for x in indels if x[0] == "D"]
                if i_ops and d_ops:
                    n_i_ops = len(i_ops)
                    orig_mask = 0
                    for j_m, x_m in enumerate(indels):
                        if x_m[0] == "I":
                            orig_mask |= 1 << j_m
                    best_m = m_max_0
                    best_res: list[tuple[bool, bool]] = [(seg_ok, mark_blk)]
                    for mask in range(1, 1 << k_cnt):
                        if mask == orig_mask or mask.bit_count() != n_i_ops:
                            continue
                        cand_indels: list[tuple[str, int, int, bool]] = []
                        ii = 0
                        di = 0
                        for pos in range(k_cnt):
                            if (mask >> pos) & 1:
                                cand_indels.append(i_ops[ii])
                                ii += 1
                            else:
                                cand_indels.append(d_ops[di])
                                di += 1
                        m_c, ok_c, blk_c = _eval_indels_order(
                            cand_indels, min_m_max=best_m
                        )
                        if m_c == -2 or m_c < best_m:
                            continue
                        if m_c > best_m:
                            best_m = m_c
                            best_res = [(ok_c, blk_c)]
                        else:
                            best_res.append((ok_c, blk_c))
                    seg_ok = all(ok_r for ok_r, _ in best_res)
                    mark_blk = any(blk_r for ok_r, blk_r in best_res if not ok_r)
            else:
                adj_swaps = [
                    j_sw
                    for j_sw in range(k_cnt - 1)
                    if indels[j_sw][0] != indels[j_sw + 1][0]
                    and not indels[j_sw][3]
                    and not indels[j_sw + 1][3]
                ]
                if len(adj_swaps) == 1:
                    sw_i = adj_swaps[0]
                    sw_indels = list(indels)
                    sw_indels[sw_i], sw_indels[sw_i + 1] = (
                        sw_indels[sw_i + 1],
                        sw_indels[sw_i],
                    )
                    m_max_1, ok_1, blk_1 = _eval_indels_order(
                        sw_indels, min_m_max=m_max_0
                    )
                    if m_max_1 != -2 and m_max_1 >= m_max_0:
                        if m_max_1 > m_max_0:
                            seg_ok, mark_blk = ok_1, blk_1
                        else:
                            ok_0 = seg_ok
                            seg_ok = ok_0 and ok_1
                            mark_blk = (mark_blk if not ok_0 else False) or (
                                blk_1 if not ok_1 else False
                            )

        if not seg_ok and mark_blk:
            assert ins_block_t_pieces_out is not None
            ins_block_t_pieces_out.add(cur_t_piece)
        return seg_ok

    # ------------------------------------------------------------------
    # Piece-level structural D >= min_hit_len splitting & extension
    # ------------------------------------------------------------------
    def _piece_ops_cover_subinterval(
        piece_ops: Sequence[tuple[int, str, int]],
    ) -> bool:
        if min_hit_len <= 0:
            return _segment_covers_subinterval(piece_ops)
        cur_seg: list[tuple[int, str, int]] = []
        pending_left_ext: list[tuple[int, str, int]] | None = None
        prev_d_inside_p = False
        n_p = len(piece_ops)
        right_q_lim = max(p_e, p_s + shift_w_size)
        left_q_lim = min(p_s, p_e - shift_w_size)
        for idx_e, entry in enumerate(piece_ops):
            if entry[1] == "D" and entry[2] >= min_hit_len:
                q_p = entry[0]
                d_len = entry[2]
                cur_d_inside_p = p_s < q_p < p_e
                next_left_ext: list[tuple[int, str, int]] | None = None
                if cur_seg and not cur_d_inside_p:
                    if not prev_d_inside_p and _segment_covers_subinterval(cur_seg):
                        return True
                    if pending_left_ext is not None and _segment_covers_subinterval(
                        _extend_seg_left_ops(cur_seg, pending_left_ext)
                    ):
                        return True
                can_try_r = bool(cur_seg) and left_q_lim < q_p < right_q_lim
                can_try_l = left_q_lim < q_p < right_q_lim
                if (
                    (can_try_r or can_try_l)
                    and query_seq is not None
                    and q_seq_len >= max(p_e, q_en)
                ):
                    d_seqs, x_map = _get_dx_for(
                        max(q_st, left_q_lim - shift_w_size),
                        min(q_en, right_q_lim + shift_w_size),
                    )
                    d_seq = d_seqs.get((q_p, d_len))
                    if d_seq is not None:
                        if can_try_r:
                            base_r = right_q_lim - q_p
                            r_orig, r_ref, r_qsub = _collect_mx_right(
                                piece_ops,
                                idx_e + 1,
                                n_p,
                                q_p,
                                x_map,
                                base_r + shift_w_size,
                            )
                            walked_r = _scan_struct_d_ext(
                                r_orig, r_ref, d_seq, r_qsub, 1, base_r
                            )
                            while (
                                walked_r
                                and walked_r[-1] == "X"
                                and q_p + len(walked_r) - 1 >= p_e
                            ):
                                walked_r.pop()
                            if walked_r and not (p_s < q_p + len(walked_r) < p_e):
                                r_add_ops = _coalesce_base_ops(q_p, walked_r)
                                seg_r = _extend_seg_right_ops(cur_seg, r_add_ops)
                                if not prev_d_inside_p and _segment_covers_subinterval(
                                    seg_r
                                ):
                                    return True
                                if (
                                    pending_left_ext is not None
                                    and _segment_covers_subinterval(
                                        _extend_seg_left_ops(
                                            seg_r, pending_left_ext
                                        )
                                    )
                                ):
                                    return True
                        if can_try_l:
                            base_l = q_p - left_q_lim
                            l_orig, l_ref, l_qsub = _collect_mx_left(
                                piece_ops,
                                idx_e - 1,
                                0,
                                q_p,
                                x_map,
                                base_l + shift_w_size,
                            )
                            walked_l = _scan_struct_d_ext(
                                l_orig, l_ref, d_seq[::-1], l_qsub, -1, base_l
                            )
                            while (
                                walked_l
                                and walked_l[-1] == "X"
                                and q_p - len(walked_l) < p_s
                            ):
                                walked_l.pop()
                            if walked_l and not (p_s < q_p - len(walked_l) < p_e):
                                walked_l.reverse()
                                next_left_ext = _coalesce_base_ops(
                                    q_p - len(walked_l), walked_l
                                )
                cur_seg = []
                pending_left_ext = next_left_ext
                prev_d_inside_p = cur_d_inside_p
            else:
                cur_seg.append(entry)
        if cur_seg:
            if not prev_d_inside_p and _segment_covers_subinterval(cur_seg):
                return True
            if pending_left_ext is not None and _segment_covers_subinterval(
                _extend_seg_left_ops(cur_seg, pending_left_ext)
            ):
                return True
        return False

    if use_direct:
        direct_ops: list[tuple[int, str, int]] = []
        q_pos = q_st
        for op, length in ops:
            direct_ops.append((q_pos, op, length))
            if op in ("M", "X", "I"):
                q_pos += length
        return _piece_ops_cover_subinterval(direct_ops)

    assert t_starts is not None and t_ends is not None
    piece_runs = _partition_bridged_hit_ops(
        ops=ops,
        q_start=q_st,
        r_st=int(hit.r_st),
        r_en=int(hit.r_en),
        strand=strand,
        t_starts=t_starts,
        t_ends=t_ends,
    )
    for k_piece in sorted(piece_runs):
        if blocked_t_pieces and k_piece in blocked_t_pieces:
            continue
        cur_t_piece = k_piece
        piece_ops = [
            (q_p, op, sub_len)
            for q_p, _r_lo, _r_hi, op, sub_len in piece_runs[k_piece]
        ]
        if _piece_ops_cover_subinterval(piece_ops):
            return True
    return False


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


class _HitPieceUnit(NamedTuple):
    ctg: str
    strand: int
    piece_key: int
    r_st: int
    r_en: int
    q_st: int
    q_en: int
    protected_ins: tuple[tuple[int, int], ...]
    mx_intervals: tuple[tuple[int, int], ...]
    total_mx: int


class _HitPieceEvalUnit(NamedTuple):
    meta: _HitPieceUnit
    runs: tuple[tuple[int, str, int], ...]
    cov_spans: tuple[tuple[int, int], ...]


def _mx_overlap(
    mx_intervals: Sequence[tuple[int, int]],
    iv_s: int,
    iv_e: int,
) -> int:
    """Return total overlap bp between sorted disjoint `mx_intervals` and `[iv_s, iv_e)`."""
    if iv_e <= iv_s or not mx_intervals:
        return 0
    total = 0
    for ms, me in mx_intervals:
        if me <= iv_s:
            continue
        if ms >= iv_e:
            break
        ov = min(me, iv_e) - max(ms, iv_s)
        if ov > 0:
            total += ov
    return total


def _extract_hit_piece_eval_units(
    hit: Any,
    query_offset: int,
    min_sv_len: int,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
) -> list[_HitPieceEvalUnit]:
    """
    Decompose a single `mappy` hit into atomic per-target-piece evaluation units with
    query coordinates already shifted by `+ query_offset`, sharing `_partition_bridged_hit_ops`
    on bridged hits and skipping zero-query (`D`-only) units.
    """
    if not getattr(hit, "cs", None):
        return []
    ctg = str(getattr(hit, "ctg", ""))
    strand = int(hit.strand)
    ops = parse_cs_to_query_ops(hit.cs, strand)
    check_targets = t_starts is not None and t_ends is not None

    use_direct = True
    direct_piece_key = 0
    if check_targets:
        assert t_starts is not None and t_ends is not None
        pos_t = bisect.bisect_right(t_starts, int(hit.r_st)) - 1
        if pos_t < 0 or int(hit.r_en) > t_ends[pos_t]:
            use_direct = False
        else:
            direct_piece_key = pos_t

    if use_direct:
        q_st_abs = query_offset + int(hit.q_st)
        q_pos = q_st_abs
        prot: list[tuple[int, int]] = []
        mx_raw: list[tuple[int, int]] = []
        runs_direct: list[tuple[int, str, int]] = []
        for op, length in ops:
            runs_direct.append((q_pos, op, length))
            if op == "I":
                if length >= min_sv_len:
                    prot.append((q_pos, q_pos + length))
                q_pos += length
            elif op in ("M", "X"):
                mx_raw.append((q_pos, q_pos + length))
                q_pos += length
        if q_pos <= q_st_abs:
            return []
        mx_merged = tuple(merge_intervals(mx_raw))
        total_mx = sum(e - s for s, e in mx_merged)
        prot_t = tuple(prot)
        cov_t = ((q_st_abs, q_pos),) if prot_t else ()
        return [
            _HitPieceEvalUnit(
                meta=_HitPieceUnit(
                    ctg=ctg,
                    strand=strand,
                    piece_key=direct_piece_key,
                    r_st=int(hit.r_st),
                    r_en=int(hit.r_en),
                    q_st=q_st_abs,
                    q_en=q_pos,
                    protected_ins=prot_t,
                    mx_intervals=mx_merged,
                    total_mx=total_mx,
                ),
                runs=tuple(runs_direct),
                cov_spans=cov_t,
            )
        ]

    assert t_starts is not None and t_ends is not None
    piece_runs = _partition_bridged_hit_ops(
        ops=ops,
        q_start=query_offset + int(hit.q_st),
        r_st=int(hit.r_st),
        r_en=int(hit.r_en),
        strand=strand,
        t_starts=t_starts,
        t_ends=t_ends,
    )
    units: list[_HitPieceEvalUnit] = []
    for k_piece in sorted(piece_runs):
        runs_k = piece_runs[k_piece]
        q_min = -1
        q_max = -1
        r_min = -1
        r_max = -1
        prot_k: list[tuple[int, int]] = []
        mx_raw_k: list[tuple[int, int]] = []
        eval_runs_k: list[tuple[int, str, int]] = []
        for q_p, r_lo, r_hi, op, sub_len in runs_k:
            eval_runs_k.append((q_p, op, sub_len))
            if op in ("M", "X", "I"):
                q_e = q_p + sub_len
                if q_min < 0 or q_p < q_min:
                    q_min = q_p
                if q_e > q_max:
                    q_max = q_e
                if op == "I":
                    if sub_len >= min_sv_len:
                        prot_k.append((q_p, q_e))
                else:
                    mx_raw_k.append((q_p, q_e))
            if r_min < 0 or r_lo < r_min:
                r_min = r_lo
            if r_hi > r_max:
                r_max = r_hi
        if q_max <= q_min:
            # Skip D-only piece runs with zero query coverage
            continue
        mx_merged_k = tuple(merge_intervals(mx_raw_k))
        total_mx_k = sum(e - s for s, e in mx_merged_k)
        prot_t_k = tuple(prot_k)
        cov_t_k = (
            tuple(
                (q_p, q_p + sub_len)
                for q_p, op, sub_len in eval_runs_k
                if op in ("M", "X", "I")
            )
            if prot_t_k
            else ()
        )
        units.append(
            _HitPieceEvalUnit(
                meta=_HitPieceUnit(
                    ctg=ctg,
                    strand=strand,
                    piece_key=k_piece,
                    r_st=r_min,
                    r_en=r_max,
                    q_st=q_min,
                    q_en=q_max,
                    protected_ins=prot_t_k,
                    mx_intervals=mx_merged_k,
                    total_mx=total_mx_k,
                ),
                runs=tuple(eval_runs_k),
                cov_spans=cov_t_k,
            )
        )
    return units


def _extract_hit_piece_metadata(
    hit: Any,
    min_sv_len: int,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
) -> list[_HitPieceUnit]:
    """
    Extract per-target-piece coordinate, `protected_ins`, and `(M, X)` coverage metadata
    for a single `mappy` hit (sharing `_extract_hit_piece_eval_units`).
    """
    return [
        u.meta
        for u in _extract_hit_piece_eval_units(hit, 0, min_sv_len, t_starts, t_ends)
    ]


def _evaluate_piece_units_on_blocks(
    units: Sequence[_HitPieceEvalUnit],
    q_len: int,
    num_blocks: int,
    min_identity: float = REF_SUB_MIN_IDENTITY,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Evaluate pre-extracted atomic per-piece hit units on the global 50 bp block grid."""
    hit_masked_intervals: list[tuple[int, int]] = []
    hit_aligned_intervals: list[tuple[int, int]] = []
    for unit in units:
        block_m = [0] * num_blocks
        block_x = [0] * num_blocks
        block_i = [0] * num_blocks
        block_d = [0] * num_blocks
        for q_p, op, sub_len in unit.runs:
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
        _finalize_hit_block_counts(
            block_m=block_m,
            block_x=block_x,
            block_i=block_i,
            block_d=block_d,
            protected_ins=unit.meta.protected_ins,
            cov_spans=unit.cov_spans,
            q_len=q_len,
            num_blocks=num_blocks,
            min_identity=min_identity,
            hit_masked_intervals=hit_masked_intervals,
            hit_aligned_intervals=hit_aligned_intervals,
        )
    return hit_masked_intervals, hit_aligned_intervals


def _unit_span(u: _HitPieceUnit) -> int:
    return (u.q_en - u.q_st) + (u.r_en - u.r_st)


def _unit_subsumed_by(u_small: _HitPieceUnit, u_big: _HitPieceUnit) -> bool:
    if (
        u_small.ctg != u_big.ctg
        or u_small.strand != u_big.strand
        or u_small.piece_key != u_big.piece_key
    ):
        return False
    if not (
        u_small.r_st >= u_big.r_st
        and u_small.r_en <= u_big.r_en
        and u_small.q_st >= u_big.q_st
        and u_small.q_en <= u_big.q_en
    ):
        return False
    # Tandem-duplication guard: do NOT suppress a hit genuinely aligning M+X bases
    # inside any protected_ins interval of u_big.
    for ps, pe in u_big.protected_ins:
        mx_in = _mx_overlap(u_small.mx_intervals, ps, pe)
        if mx_in > 0:
            if mx_in * 10 >= 9 * min(REF_SUB_BLOCK_SIZE, pe - ps):
                return False
            if u_small.total_mx > 0 and mx_in * 2 >= u_small.total_mx:
                return False
    return True


def _preserves_protected_ins(u_big: _HitPieceUnit, u_small: _HitPieceUnit) -> bool:
    for ps, pe in u_small.protected_ins:
        ins_len = pe - ps
        if ins_len <= 0:
            continue
        min_ov = max(1, ins_len // 2)
        if any(
            min(bp_e, pe) - max(bp_s, ps) >= min_ov
            for bp_s, bp_e in u_big.protected_ins
        ):
            continue
        mx_in = _mx_overlap(u_big.mx_intervals, ps, pe)
        if mx_in * 10 >= 9 * min(REF_SUB_BLOCK_SIZE, ins_len):
            continue
        return False
    return True


def _filter_cross_aligner_shadow_units(
    main_units: Sequence[_HitPieceEvalUnit],
    short_units: Sequence[_HitPieceEvalUnit],
) -> list[_HitPieceEvalUnit]:
    """
    Deduplicate cross-aligner shadow alignments at atomic per-piece-unit granularity
    grouped by `(ctg, strand, piece_key)`:
      - Step 1a (`ds_1a`): Drop `short_units` (`s_span >= m_span`) subsuming a `main_unit`
        while shattering its `protected_ins`.
      - Step 1b (`dm_1`, tentative): Mark `main_units` subsumed by a surviving `short_unit`
        preserving `protected_ins` with `s_span > m_span` or converse exact-span tie
        (`s_span == m_span` and `mu` shattered `su.protected_ins`).
      - Step 2a (`ds_2a`): Drop remaining `short_units` subsumed (`m_span >= s_span`) by a
        surviving `main_unit` preserving `protected_ins` (canonical `asm20` wins ties).
      - Step 2b (`short -> short`, `ds_2b`): Among surviving `short_units` not in
        `ds_1a | ds_2a` (`len > 1`), drop any secondary arm subsumed by a larger (or
        non-shattering exact-tie) surviving `short_unit` preserving `protected_ins`.
      - Witness-Survival Re-Check (`dm_final`): If `dm_1` is non-empty and `ds_2a` or
        `ds_2b` is non-empty, keep in `dm_final` only `m_idx in dm_1` still matched by
        a surviving `s_idx` not in `ds_1a | ds_2a | ds_2b`.
    """
    if not short_units:
        return list(main_units)

    main_by_key: dict[tuple[str, int, int], list[int]] = collections.defaultdict(list)
    for m_idx, m_u in enumerate(main_units):
        um = m_u.meta
        main_by_key[(um.ctg, um.strand, um.piece_key)].append(m_idx)

    short_by_key: dict[tuple[str, int, int], list[int]] = collections.defaultdict(list)
    for s_idx, s_u in enumerate(short_units):
        us = s_u.meta
        short_by_key[(us.ctg, us.strand, us.piece_key)].append(s_idx)

    drop_main: set[int] = set()
    drop_short: set[int] = set()

    for key, s_indices in short_by_key.items():
        m_indices = main_by_key.get(key, ())
        ds_1a: set[int] = set()
        dm_1: set[int] = set()
        ds_2a: set[int] = set()

        if m_indices:
            # Step 1a: Drop short_units (>= span) that subsume a main_unit while
            # shattering its protected_ins.
            for s_idx in s_indices:
                su = short_units[s_idx].meta
                s_span = _unit_span(su)
                for m_idx in m_indices:
                    mu = main_units[m_idx].meta
                    if (
                        s_span >= _unit_span(mu)
                        and _unit_subsumed_by(mu, su)
                        and not _preserves_protected_ins(su, mu)
                    ):
                        ds_1a.add(s_idx)
                        break

            # Step 1b (tentative): Mark main_units subsumed by a surviving short_unit
            # that preserves protected_ins (strictly larger span, or exact-span tie
            # where mu shattered su.protected_ins).
            for m_idx in m_indices:
                mu = main_units[m_idx].meta
                m_span = _unit_span(mu)
                for s_idx in s_indices:
                    if s_idx in ds_1a:
                        continue
                    su = short_units[s_idx].meta
                    s_span = _unit_span(su)
                    if (
                        s_span >= m_span
                        and _unit_subsumed_by(mu, su)
                        and _preserves_protected_ins(su, mu)
                        and (
                            s_span > m_span
                            or not _preserves_protected_ins(mu, su)
                        )
                    ):
                        dm_1.add(m_idx)
                        break

            # Step 2a: Drop remaining short_units subsumed by a surviving main_unit
            # that preserves protected_ins (canonical asm20 wins ties).
            for s_idx in s_indices:
                if s_idx in ds_1a:
                    continue
                su = short_units[s_idx].meta
                s_span = _unit_span(su)
                for m_idx in m_indices:
                    if m_idx in dm_1:
                        continue
                    mu = main_units[m_idx].meta
                    if (
                        _unit_span(mu) >= s_span
                        and _unit_subsumed_by(su, mu)
                        and _preserves_protected_ins(mu, su)
                    ):
                        ds_2a.add(s_idx)
                        break

        # Step 2b: Cleanup surviving secondary short_unit arms subsumed by a larger
        # (or non-shattering exact-tie) surviving short_unit on the same piece.
        ds_ab = ds_1a | ds_2a
        surv_s = [s_idx for s_idx in s_indices if s_idx not in ds_ab]
        ds_2b: set[int] = set()
        if len(surv_s) > 1:
            for s_idx in surv_s:
                su = short_units[s_idx].meta
                s_span = _unit_span(su)
                for s2_idx in surv_s:
                    if s2_idx == s_idx:
                        continue
                    su2 = short_units[s2_idx].meta
                    s2_span = _unit_span(su2)
                    if (
                        s2_span >= s_span
                        and _unit_subsumed_by(su, su2)
                        and _preserves_protected_ins(su2, su)
                        and (
                            s2_span > s_span
                            or not _preserves_protected_ins(su, su2)
                        )
                    ):
                        ds_2b.add(s_idx)
                        break
        ds_all = ds_ab | ds_2b
        drop_short.update(ds_all)

        # Witness-Survival Re-Check: prevent dropped-witness cycles when Step 2a
        # or Step 2b eliminates a short_unit that tentatively dropped a main_unit.
        if m_indices:
            if dm_1 and (ds_2a or ds_2b):
                dm_final: set[int] = set()
                for m_idx in dm_1:
                    mu = main_units[m_idx].meta
                    m_span = _unit_span(mu)
                    for s_idx in s_indices:
                        if s_idx in ds_all:
                            continue
                        su = short_units[s_idx].meta
                        s_span = _unit_span(su)
                        if (
                            s_span >= m_span
                            and _unit_subsumed_by(mu, su)
                            and _preserves_protected_ins(su, mu)
                            and (
                                s_span > m_span
                                or not _preserves_protected_ins(mu, su)
                            )
                        ):
                            dm_final.add(m_idx)
                            break
            else:
                dm_final = dm_1
            drop_main.update(dm_final)

    kept_units: list[_HitPieceEvalUnit] = [
        u for idx, u in enumerate(main_units) if idx not in drop_main
    ]
    kept_units.extend(
        u for idx, u in enumerate(short_units) if idx not in drop_short
    )
    return kept_units


def _filter_cross_aligner_shadow_hits(
    main_hits: Sequence[Any],
    short_hits: Sequence[Any],
    min_sv_len: int,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
) -> list[Any]:
    """
    Deduplicate same-locus shadow hits when both `aligner` (`asm20`) and `short_aligner`
    map a `<= 500 bp` slice. Prevents an unbridged `asm20` arm from clobbering a bridged
    `short_aligner` `protected_ins` (or falsely masking `< min_identity` blocks), prevents
    identical-span `I`-shift/shattered-`I` disagreements, and preserves genuine tandem
    duplications aligning `M+X` inside `protected_ins`.
    """
    main_meta = [
        _extract_hit_piece_metadata(h, min_sv_len, t_starts, t_ends)
        for h in main_hits
    ]
    short_meta = [
        _extract_hit_piece_metadata(h, min_sv_len, t_starts, t_ends)
        for h in short_hits
    ]

    drop_main: set[int] = set()
    drop_short: set[int] = set()

    def _match_strictly_subsuming(
        m_units: Sequence[_HitPieceUnit],
        s_units: Sequence[_HitPieceUnit],
    ) -> list[tuple[_HitPieceUnit, _HitPieceUnit]] | None:
        matched_pairs: list[tuple[_HitPieceUnit, _HitPieceUnit]] = []
        has_strict_growth = len(s_units) > len(m_units)
        for mu in m_units:
            m_span = _unit_span(mu)
            best_su: _HitPieceUnit | None = None
            for su in s_units:
                s_span = _unit_span(su)
                if s_span >= m_span and _unit_subsumed_by(mu, su):
                    if s_span > m_span:
                        best_su = su
                        has_strict_growth = True
                        break
                    if best_su is None:
                        best_su = su
            if best_su is None:
                return None
            matched_pairs.append((mu, best_su))
        if not has_strict_growth:
            return None
        return matched_pairs

    # Precompute Step 1 candidate pairs (m_idx, s_idx, preserves_protected_ins).
    strict_matches: list[tuple[int, int, bool]] = []
    for m_idx, m_units in enumerate(main_meta):
        if not m_units:
            continue
        for s_idx, s_units in enumerate(short_meta):
            if not s_units:
                continue
            matched_pairs = _match_strictly_subsuming(m_units, s_units)
            if matched_pairs is None:
                continue
            preserves = all(
                _preserves_protected_ins(su, mu) for mu, su in matched_pairs
            )
            strict_matches.append((m_idx, s_idx, preserves))

    # Step 1a: Drop any short_hit that strictly subsumes a main_hit while shattering its
    # protected_ins (e.g., I(1)+M(1)+I(49) vs I(50)+X(1)) when all of short_hit's units
    # correspond to that main_hit. Resolve drop_short BEFORE drop_main so a doomed
    # short_hit never suppresses an unrelated main_hit.
    for m_idx, s_idx, preserves in strict_matches:
        if preserves or s_idx in drop_short:
            continue
        m_units = main_meta[m_idx]
        s_units = short_meta[s_idx]
        if len(s_units) == 1 or all(
            any(
                _unit_span(su) >= _unit_span(mu) and _unit_subsumed_by(mu, su)
                for mu in m_units
            )
            for su in s_units
        ):
            drop_short.add(s_idx)

    # Step 1b: Drop main_hits strictly subsumed by a surviving short_hit that preserves
    # all protected_ins.
    for m_idx, s_idx, preserves in strict_matches:
        if not preserves or s_idx in drop_short or m_idx in drop_main:
            continue
        drop_main.add(m_idx)

    # Step 2: Drop remaining short_hits subsumed (including identical-span ties) by a kept
    # main_hit that preserves protected_ins so canonical asm20 wins same-locus ties.
    for s_idx, s_units in enumerate(short_meta):
        if s_idx in drop_short or not s_units:
            continue
        for m_idx, m_units in enumerate(main_meta):
            if m_idx in drop_main or not m_units:
                continue
            if all(
                any(
                    _unit_span(mu) >= _unit_span(su)
                    and _unit_subsumed_by(su, mu)
                    and _preserves_protected_ins(mu, su)
                    for mu in m_units
                )
                for su in s_units
            ):
                drop_short.add(s_idx)
                break

    kept_hits: list[Any] = [
        h for idx, h in enumerate(main_hits) if idx not in drop_main
    ]
    kept_hits.extend(
        h for idx, h in enumerate(short_hits) if idx not in drop_short
    )
    return kept_hits


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
        if main_hits and short_hits:
            main_units: list[_HitPieceEvalUnit] = []
            for h in main_hits:
                main_units.extend(
                    _extract_hit_piece_eval_units(
                        h, query_offset, min_sv_len, t_starts, t_ends
                    )
                )
            short_units: list[_HitPieceEvalUnit] = []
            for h in short_hits:
                short_units.extend(
                    _extract_hit_piece_eval_units(
                        h, query_offset, min_sv_len, t_starts, t_ends
                    )
                )
            kept_units = _filter_cross_aligner_shadow_units(main_units, short_units)
            return _evaluate_piece_units_on_blocks(
                kept_units,
                q_len=q_len,
                num_blocks=num_blocks,
                min_identity=min_identity,
            )
        if len(short_hits) > 1:
            short_units = []
            for h in short_hits:
                short_units.extend(
                    _extract_hit_piece_eval_units(
                        h, query_offset, min_sv_len, t_starts, t_ends
                    )
                )
            kept_units = _filter_cross_aligner_shadow_units((), short_units)
            return _evaluate_piece_units_on_blocks(
                kept_units,
                q_len=q_len,
                num_blocks=num_blocks,
                min_identity=min_identity,
            )
        return _evaluate_mappy_hits_on_blocks(
            hits=main_hits if main_hits else short_hits,
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


def _cluster_edlib_raw_locations(
    locations: Sequence[tuple[int, int]],
    offset: int = 0,
    tol: int = 8,
) -> list[tuple[int, int]]:
    """
    Group adjacent inclusive raw `edlib HW` locations into half-open bounding
    clusters anchored within `tol` bp so homopolymer/indel endpoint ties invoke
    `_realign_and_trim_edlib_slice` O(1) times per locus without merging shifts
    >= `min_sv_len`.
    """
    clusters: list[tuple[int, int]] = []
    if not locations:
        return clusters
    anc_s = offset + locations[0][0]
    anc_e = offset + locations[0][1] + 1
    cur_s, cur_e = anc_s, anc_e
    for ls, le_incl in locations[1:]:
        hs = offset + ls
        he = offset + le_incl + 1
        if abs(hs - anc_s) <= tol and abs(he - anc_e) <= tol:
            if hs < cur_s:
                cur_s = hs
            if he > cur_e:
                cur_e = he
        else:
            clusters.append((cur_s, cur_e))
            anc_s, anc_e = hs, he
            cur_s, cur_e = hs, he
    clusters.append((cur_s, cur_e))
    return clusters


def _edlib_intervals_distinct(
    s1: int,
    e1: int,
    s2: int,
    e2: int,
    min_sv_len: int,
) -> bool:
    """
    Return True iff two half-open forward target intervals `[s1, e1)` and `[s2, e2)`
    represent distinct alignment loci rather than equivalent indel-boundary shifts.
    """
    ov = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union > 0 and ov * 2 < union:
        return True
    if min_sv_len > 0 and (
        abs(s1 - s2) >= min_sv_len or abs(e1 - e2) >= min_sv_len
    ):
        return True
    return False


def _edlib_rc_interval_distinct(
    fwd_s: int,
    fwd_e: int,
    rc_s: int,
    rc_e: int,
    min_sv_len: int,
) -> bool:
    """
    Return True iff a trimmed homologous RC interval `[rc_s, rc_e)` represents a
    distinct locus from `[fwd_s, fwd_e)` (extends outside `[fwd_s, fwd_e)` by
    `>= min_sv_len` on either side, or has `> 50%` of its span outside `[fwd_s, fwd_e)`),
    rather than a contained internal palindrome or `< min_sv_len` shifted same-locus hit.
    """
    rc_span = rc_e - rc_s
    if rc_span <= 0:
        return False
    ov = max(0, min(fwd_e, rc_e) - max(fwd_s, rc_s))
    if ov * 2 < rc_span:
        return True
    if min_sv_len > 0 and (
        (fwd_s - rc_s) >= min_sv_len or (rc_e - fwd_e) >= min_sv_len
    ):
        return True
    return False


def _trim_edlib_hit_for_c1(
    query: str,
    target: str,
    loc_s: int,
    loc_e: int,
    min_aligned_core: int,
) -> tuple[bool, list[tuple[str, int]], int, int, int, int, int]:
    """
    Validate and `asm20`-trim a raw `edlib HW` hit `[loc_s, loc_e)` via
    `_realign_and_trim_edlib_slice`. Returns
    `(is_valid, ops, q_start, q_end, t_start, t_end, asm20_score)`.
    """
    h_ops, h_qs, h_qe, h_ts, h_te = _realign_and_trim_edlib_slice(
        query, target[loc_s:loc_e], 0, loc_s
    )
    if not h_ops:
        return False, [], 0, 0, 0, 0, 0
    if (h_qe - h_qs) < min_aligned_core or (h_te - h_ts) < min_aligned_core:
        return False, h_ops, h_qs, h_qe, h_ts, h_te, 0
    h_score = _asm20_ops_score(h_ops)
    if h_score <= 0:
        return False, h_ops, h_qs, h_qe, h_ts, h_te, h_score
    return True, h_ops, h_qs, h_qe, h_ts, h_te, h_score


def _edlib_core_max_ed(span: int) -> int:
    """Secondary-hit Levenshtein threshold for an aligned core (`15%` if `<25 bp`, `20%` if `>=25 bp`)."""
    if span < 25:
        return int(0.15 * span)
    return int(0.20 * span)


def _has_distinct_shifted_edlib_hit(
    query: str,
    target: str,
    t_start: int,
    t_end: int,
    max_ed: int,
    min_sv_len: int,
    min_sec_len: int,
) -> bool:
    """
    Search shifted right (`target[right_cut:]`) and left (`target[:left_cut]`) slices
    for any credible forward alignment of `query` within `<= max_ed` whose interval
    is distinct from `[t_start, t_end)`.

    Uses `dist_shift = min(min_sv_len, max(1, t_span // 3 + 1))` so short tandem
    copies (`t_span < min_sv_len`, e.g. `u = 25..35 bp` or custom `--min-sv-len 200`)
    are never clamped to `t_start`/`t_end`, while `shift_safe >= max_ed + 1 + max(0, t_span - len_q)`
    preserves exact triangle-inequality self-hit exclusion.
    """
    len_r = len(target)
    len_q = len(query)
    if max_ed < 0 or len_q == 0 or len_r < min_sec_len:
        return False

    t_span = t_end - t_start
    dist_shift = (
        min(min_sv_len, max(1, t_span // 3 + 1))
        if min_sv_len > 0
        else max(1, t_span // 3 + 1)
    )
    shift_safe = max(dist_shift, max_ed + 1 + max(0, t_span - len_q))

    right_cut = min(t_end, t_start + shift_safe)
    if len_r - right_cut >= min_sec_len:
        res_r = edlib.align(
            query, target[right_cut:], mode="HW", task="locations", k=max_ed
        )
        if 0 <= res_r.get("editDistance", -1) <= max_ed:
            for ls, le_incl in res_r.get("locations") or []:
                if _edlib_intervals_distinct(
                    t_start,
                    t_end,
                    right_cut + ls,
                    right_cut + le_incl + 1,
                    min_sv_len,
                ):
                    return True

    left_cut = max(t_start, t_end - shift_safe)
    if left_cut >= min_sec_len:
        res_l = edlib.align(
            query, target[:left_cut], mode="HW", task="locations", k=max_ed
        )
        if 0 <= res_l.get("editDistance", -1) <= max_ed:
            for ls, le_incl in res_l.get("locations") or []:
                if _edlib_intervals_distinct(
                    t_start, t_end, ls, le_incl + 1, min_sv_len
                ):
                    return True

    return False


def _has_distinct_shifted_edlib_rc_hit(
    query: str,
    target: str,
    fwd_start: int,
    fwd_end: int,
    rc_spans: Sequence[tuple[int, int]],
    max_ed: int,
    min_sv_len: int,
    min_sec_len: int,
    min_aligned_core: int,
) -> bool:
    """
    Search shifted right and left slices outside the trimmed same-locus RC
    interval(s) `rc_spans` (half-open `(rs, re)`) for any credible RC alignment
    distinct from BOTH `[fwd_start, fwd_end)` AND every interval in `rc_spans`.
    Any overlapping candidate slice hit is `asm20`-trimmed before distinctness check.
    """
    len_r = len(target)
    len_q = len(query)
    if max_ed < 0 or len_q == 0 or len_r < min_sec_len or not rc_spans:
        return False

    rc_min_s = min(s_i for s_i, _ in rc_spans)
    rc_max_s = max(s_i for s_i, _ in rc_spans)
    rc_min_e = min(e_i for _, e_i in rc_spans)
    rc_max_e = max(e_i for _, e_i in rc_spans)
    rc_max_span = max(e_i - s_i for s_i, e_i in rc_spans)
    dist_shift = (
        min(min_sv_len, max(1, rc_max_span // 3 + 1))
        if min_sv_len > 0
        else max(1, rc_max_span // 3 + 1)
    )
    shift_rc = max(dist_shift, max_ed + 1 + max(0, rc_max_span - len_q))

    def _check_slice(sub_target: str, offset: int) -> bool:
        res = edlib.align(
            query, sub_target, mode="HW", task="locations", k=max_ed
        )
        if not (0 <= res.get("editDistance", -1) <= max_ed):
            return False
        for hs, he in _cluster_edlib_raw_locations(
            res.get("locations") or [], offset=offset
        ):
            overlaps_any = (min(fwd_end, he) > max(fwd_start, hs)) or any(
                min(re, he) > max(rs, hs) for rs, re in rc_spans
            )
            if not overlaps_any:
                return True
            valid, _, _, _, h_ts, h_te, _ = _trim_edlib_hit_for_c1(
                query, target, hs, he, min_aligned_core
            )
            if (
                valid
                and _edlib_rc_interval_distinct(
                    fwd_start, fwd_end, h_ts, h_te, min_sv_len
                )
                and all(
                    _edlib_rc_interval_distinct(rs, re, h_ts, h_te, min_sv_len)
                    for rs, re in rc_spans
                )
            ):
                return True
        return False

    right_cut = min(rc_max_e, rc_max_s + shift_rc)
    if len_r - right_cut >= min_sec_len and _check_slice(
        target[right_cut:], right_cut
    ):
        return True

    left_cut = max(rc_min_s, rc_min_e - shift_rc)
    if left_cut >= min_sec_len and _check_slice(target[:left_cut], 0):
        return True

    return False


def _has_disqualifying_edlib_rc_hit(
    rc_query: str,
    target: str,
    t_start: int,
    t_end: int,
    m_total: int,
    fwd_core_score: int,
    rc_max_ed_distinct: int,
    rc_search_k: int,
    min_sv_len: int,
    min_aligned_core: int,
    trimmed_right: bool,
    trimmed_left: bool,
    check_same_locus_tie: bool,
) -> bool:
    """
    Evaluate reverse-complement Criterion 1c for `rc_query` on `target`:
    - Disjoint raw hits at `rc_ed <= rc_max_ed_distinct` (or trimmed valid hits if
      `rc_ed > rc_max_ed_distinct`) reject immediately.
    - Overlapping raw hits are `asm20`-trimmed via `_trim_edlib_hit_for_c1`, then
      checked for: (i) distinct RC copy via `_edlib_rc_interval_distinct`, and
      (ii) same-locus orientation tie (including `<= _EDLIB_RC_BLEED_TOL` trimmed
      overhang flank bleed).
    - If primary `rc_ed <= rc_max_ed_distinct` landed on a same-locus hit, runs
      `_has_distinct_shifted_edlib_rc_hit` anchored to the trimmed same-locus RC
      span(s) to detect any slightly worse second distinct RC copy.
    """
    len_q = len(rc_query)
    if len_q == 0 or rc_search_k < 0:
        return False

    rc_res = edlib.align(
        rc_query, target, mode="HW", task="locations", k=rc_search_k
    )
    rc_ed = rc_res.get("editDistance", -1)
    if not (0 <= rc_ed <= rc_search_k):
        return False

    raw_clusters = _cluster_edlib_raw_locations(
        rc_res.get("locations") or [], offset=0
    )
    same_locus_rc_spans: list[tuple[int, int]] = []
    bleed_score_tol = (
        (_EDLIB_RC_BLEED_TOL if trimmed_right else 0)
        + (_EDLIB_RC_BLEED_TOL if trimmed_left else 0)
    )

    for ls, le in raw_clusters:
        overlaps_fwd = min(t_end, le) > max(t_start, ls)
        if not overlaps_fwd:
            if rc_ed <= rc_max_ed_distinct:
                return True
            valid, _, _, _, h_ts, h_te, _ = _trim_edlib_hit_for_c1(
                rc_query, target, ls, le, min_aligned_core
            )
            if valid and _edlib_rc_interval_distinct(
                t_start, t_end, h_ts, h_te, min_sv_len
            ):
                return True
            continue

        valid, rc_ops, rc_qs, rc_qe, rc_ts, rc_te, rc_score = (
            _trim_edlib_hit_for_c1(
                rc_query, target, ls, le, min_aligned_core
            )
        )
        if not valid:
            continue

        if rc_ed <= rc_max_ed_distinct and _edlib_rc_interval_distinct(
            t_start, t_end, rc_ts, rc_te, min_sv_len
        ):
            return True

        if check_same_locus_tie:
            rc_m = sum(ln for op, ln in rc_ops if op == "M")
            if (
                (trimmed_right or rc_qs == 0)
                and (trimmed_left or rc_qe == len_q)
                and (fwd_core_score - rc_score) <= bleed_score_tol
                and (m_total - rc_m) <= bleed_score_tol
            ):
                return True

        same_locus_rc_spans.append((rc_ts, rc_te))

    if rc_ed <= rc_max_ed_distinct:
        anchor_spans = (
            same_locus_rc_spans if same_locus_rc_spans else [(t_start, t_end)]
        )
        min_rc_len = max(1, len_q - rc_max_ed_distinct)
        if _has_distinct_shifted_edlib_rc_hit(
            rc_query,
            target,
            t_start,
            t_end,
            anchor_spans,
            rc_max_ed_distinct,
            min_sv_len,
            min_rc_len,
            min_aligned_core,
        ):
            return True

    return False


def _cluster_edlib_raw_locations_bounds(
    locations: Sequence[tuple[int, int]],
    offset: int = 0,
    tol: int = 8,
) -> list[tuple[int, int, int, int]]:
    """
    Cluster adjacent inclusive raw `edlib HW` locations within `tol` bp,
    returning `(min_s, max_s, min_e, max_e)` in half-open coordinates.
    Tracking exact `min_e`/`max_s` alongside the bounding box `(min_s, max_e)`
    prevents tight gap windows from clipping adjacent segments or spilling back
    into self-identical arms.
    """
    clusters: list[tuple[int, int, int, int]] = []
    if not locations:
        return clusters
    anc_s = offset + locations[0][0]
    anc_e = offset + locations[0][1] + 1
    min_s = max_s = anc_s
    min_e = max_e = anc_e
    for ls, le_incl in locations[1:]:
        hs = offset + ls
        he = offset + le_incl + 1
        if abs(hs - anc_s) <= tol and abs(he - anc_e) <= tol:
            if hs < min_s:
                min_s = hs
            if hs > max_s:
                max_s = hs
            if he < min_e:
                min_e = he
            if he > max_e:
                max_e = he
        else:
            clusters.append((min_s, max_s, min_e, max_e))
            anc_s, anc_e = hs, he
            min_s = max_s = hs
            min_e = max_e = he
    clusters.append((min_s, max_s, min_e, max_e))
    return clusters


def _is_edlib_same_locus_rc_tie(
    rc_q_core: str,
    target: str,
    rc_s: int,
    rc_e: int,
    m_total: int,
    fwd_core_score: int,
    min_aligned_core: int,
    trimmed_right: bool,
    trimmed_left: bool,
    bleed_k_pad: int,
) -> bool:
    """
    Evaluate the provably zero-FN Patch 2 same-locus RC orientation-tie check
    on `target[rc_s:rc_e]` (unshifted or `< min_sv_len`-shifted relative to
    `[t_start, t_end)`). Handles short `<= 0`-score palindromes and clean
    `< 15 bp` shifted arms across large internal indels without modifying
    `_has_disqualifying_edlib_rc_hit`.
    """
    q_span = len(rc_q_core)
    t_sub_len = rc_e - rc_s
    if q_span == 0 or t_sub_len <= 0:
        return False
    # Zero-FN length bound: rc_m <= t_sub_len, so m_total - rc_m <= bleed_k_pad
    # requires t_sub_len >= m_total - bleed_k_pad and k_sl_t >= 0.
    if t_sub_len < m_total - bleed_k_pad:
        return False
    k_sl_t = (q_span - m_total) + (t_sub_len - m_total) + 2 * bleed_k_pad
    if k_sl_t < 0:
        return False
    t_sub = target[rc_s:rc_e]
    sl_ed = edlib.align(
        rc_q_core, t_sub, mode="NW", task="distance", k=k_sl_t
    ).get("editDistance", -1)
    if not (0 <= sl_ed <= k_sl_t):
        return False
    nw_res = edlib.align(rc_q_core, t_sub, mode="NW", task="path")
    nw_cigar = nw_res.get("cigar")
    if nw_cigar is None or nw_res.get("editDistance", -1) < 0:
        return False
    raw_ops = parse_edlib_cigar_to_ops(nw_cigar, seq_s=rc_q_core, seq_r=t_sub)
    if not raw_ops:
        return False
    rc_ops_sl, rc_qs_sl, rc_qe_sl, rc_ts_sl, rc_te_sl = (
        _trim_edlib_terminal_overhangs(
            raw_ops, 0, q_span, rc_s, rc_e, seq_s=rc_q_core, seq_r=target
        )
    )
    if (
        rc_ops_sl
        and (rc_qe_sl - rc_qs_sl) >= min_aligned_core
        and (rc_te_sl - rc_ts_sl) >= min_aligned_core
    ):
        rc_sc_sl = _asm20_ops_score(rc_ops_sl)
        rc_m_sl = sum(ln for op, ln in rc_ops_sl if op == "M")
        if (
            (trimmed_right or rc_qs_sl == 0)
            and (trimmed_left or rc_qe_sl == q_span)
            and (fwd_core_score - rc_sc_sl) <= bleed_k_pad
            and (m_total - rc_m_sl) <= bleed_k_pad
        ):
            return True
    # Fallback when _trim_edlib_terminal_overhangs clips a clean < 15 bp outer
    # arm across a large internal deletion (e.g. pal50[10:]+45D -> 26M 45D 14M):
    # require exact 'M' at both ends of untrimmed NW ops and exact tie bounds.
    if (
        raw_ops[0][0] == "M"
        and raw_ops[-1][0] == "M"
        and q_span >= min_aligned_core
        and t_sub_len >= min_aligned_core
    ):
        raw_sc = _asm20_ops_score(raw_ops)
        raw_m = sum(ln for op, ln in raw_ops if op == "M")
        if (
            (fwd_core_score - raw_sc) <= bleed_k_pad
            and (m_total - raw_m) <= bleed_k_pad
        ):
            return True
    return False


def _has_distinct_edlib_indel_copy2(
    q_core: str,
    rc_q_core: str,
    target: str,
    t_start: int,
    t_end: int,
    min_sv_len: int,
    min_aligned_core: int,
) -> bool:
    """
    Unified bounded outer-arm & interior-arm Copy 2 finder and full-copy
    `asm20` validator (Stage 6E) for cases where Copy 1 has no split-worthy
    internal indels (or differs in indel polarity from Copy 2), while a
    distinct Copy 2 carries `< min_sv_len` indels/truncations (including
    canceling `I+D`, single/double-ended tip damage, or `ed=0` single-decoy
    tip masking) exceeding Stage 6B/6C's `20%` edit cap on `+` or `-` strand.

    Only activates when `2 * (min_sv_len - 1) > int(0.15 * q_span)` and
    `q_span >= 2 * min_aligned_core` (zero cost on `>= 654 bp` cores at
    default `min_sv_len=50`, e.g. `0 ms` on `950bp_1snp_nonpal_msl50`) and
    when enough target sequence lies outside `[t_start, t_end)` for a distinct
    Copy 2 to exist. Decouples cheap C-level `edlib NW distance` screening
    from gap-localized sorted-DP validation with proximity deduplication.
    """
    len_r = len(target)
    q_span = len(q_core)
    if (
        min_sv_len <= 1
        or q_span < 2 * min_aligned_core
        or 2 * (min_sv_len - 1) <= int(0.15 * q_span)
    ):
        return False

    arm_len = min_aligned_core
    arm_k = _edlib_core_max_ed(arm_len)
    min_c2_span = max(min_aligned_core, q_span - (min_sv_len - 1) - 8)
    max_c2_span = q_span + (min_sv_len - 1) + 8
    max_outside = t_start + (len_r - t_end)
    min_distinct_t_al = max(min_aligned_core, q_span - 2 * (min_sv_len - 1))
    if (
        t_start < min_sv_len
        and (len_r - t_end) < min_sv_len
        and max_outside * 2 <= min_distinct_t_al
    ):
        return False

    q_core_u = q_core.upper()
    rc_q_core_u = rc_q_core.upper()
    target_u = target.upper()

    max_probe_clusters = 48
    max_nw_outer = 64
    max_nw_fb = 32
    max_dp_outer = 2
    max_dp_fb = 2
    prox_r = min(max(64, min_sv_len), max(16, q_span // 2))
    max_i_valid = min(
        min_sv_len - 1, q_span - max(min_aligned_core, q_span // 2)
    )

    def _is_low_complexity_probe(probe: str) -> bool:
        p_len = len(probe)
        if p_len < 6:
            return False
        if p_len <= 24:
            return (
                len(
                    dict.fromkeys(
                        probe[idx : idx + 3] for idx in range(p_len - 2)
                    )
                )
                <= 4
            )
        # Fast 2-sample pre-check on long probes: if both the 1/4 and 3/4
        # 19-mer windows are complex (> 4 distinct 3-mers), the probe cannot
        # be >= 75% dominated by a single 1-4 bp repeat.
        w1 = probe[p_len // 4 : p_len // 4 + 19]
        w2 = probe[max(0, (3 * p_len) // 4 - 19) : (3 * p_len) // 4]
        if (
            len(w1) >= 19
            and len(w2) >= 19
            and len(dict.fromkeys(w1[i : i + 3] for i in range(17))) > 4
            and len(dict.fromkeys(w2[i : i + 3] for i in range(17))) > 4
        ):
            return False
        if (
            len(dict.fromkeys(probe[idx : idx + 3] for idx in range(p_len - 2)))
            <= 4
        ):
            return True
        counts: dict[str, int] = {}
        for idx in range(p_len - 2):
            kmer = probe[idx : idx + 3]
            counts[kmer] = counts.get(kmer, 0) + 1
        n_kmers = p_len - 2
        top4 = sum(sorted(counts.values(), reverse=True)[:4])
        return top4 * 4 >= 3 * n_kmers

    def _calc_k_c2(c2_span: int) -> int:
        max_d_valid = min(min_sv_len - 1, max(0, c2_span - min_aligned_core))
        indel_cap = max(
            abs(c2_span - q_span),
            min(
                2 * max_i_valid + (c2_span - q_span),
                2 * max_d_valid + (q_span - c2_span),
            )
            + 8,
        )
        return min(
            q_span + c2_span - 2 * min_aligned_core,
            indel_cap + 8 + _edlib_core_max_ed(min(q_span, c2_span)),
        )

    def _probe_slice_bounds(
        probe: str,
        sl_s: int,
        sl_e: int,
        k_lim: int,
        max_cl: int = max_probe_clusters,
    ) -> list[tuple[int, int, int, int]]:
        min_len = max(1, len(probe) - k_lim)
        if sl_e - sl_s < min_len:
            return []
        res = edlib.align(
            probe, target_u[sl_s:sl_e], mode="HW", task="locations", k=k_lim
        )
        if not (0 <= res.get("editDistance", -1) <= k_lim):
            return []
        return _cluster_edlib_raw_locations_bounds(
            res.get("locations") or [], offset=sl_s
        )[:max_cl]

    def _merge_unique_clusters(
        lists: Sequence[list[tuple[int, int, int, int]]],
    ) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        seen: dict[tuple[int, int, int, int], None] = {}
        for lst in lists:
            for cl in lst:
                if cl not in seen:
                    seen[cl] = None
                    merged.append(cl)
        merged.sort()
        return merged

    def _try_exact_single_gap_ops(
        s_cl: str,
        r_cl: str,
    ) -> list[tuple[str, int]] | None:
        qs_cl = len(s_cl)
        ts_cl = len(r_cl)
        if qs_cl == ts_cl or min(qs_cl, ts_cl) < 8:
            return None
        is_del = ts_cl > qs_cl
        short_s = s_cl if is_del else r_cl
        long_s = r_cl if is_del else s_cl
        m = len(short_s)
        g = len(long_s) - m
        pref = [0] * (m + 1)
        acc = 0
        for idx_s in range(m):
            if short_s[idx_s] == long_s[idx_s]:
                acc += 1
            pref[idx_s + 1] = acc
        suff = [0] * (m + 1)
        acc = 0
        for idx_s in range(m - 1, -1, -1):
            if short_s[idx_s] == long_s[idx_s + g]:
                acc += 1
            suff[idx_s] = acc
        best_eq = -1
        best_k = 0
        for k_pos in range(m + 1):
            tot = pref[k_pos] + suff[k_pos]
            if tot > best_eq:
                best_eq = tot
                best_k = k_pos
        n_mm = m - best_eq
        if n_mm * 20 > m * 3:
            return None
        if n_mm >= 3:
            mm_flags = [0] * m
            for idx_s in range(best_k):
                if short_s[idx_s] != long_s[idx_s]:
                    mm_flags[idx_s] = 1
            for idx_s in range(best_k, m):
                if short_s[idx_s] != long_s[idx_s + g]:
                    mm_flags[idx_s] = 1
            w_mm = sum(mm_flags[:8])
            if w_mm >= 3:
                return None
            for idx_s in range(8, m):
                w_mm += mm_flags[idx_s] - mm_flags[idx_s - 8]
                if w_mm >= 3:
                    return None
        sg_ops: list[tuple[str, int]] = []
        if best_k > 0:
            _emit_mx_runs(sg_ops, short_s[:best_k], long_s[:best_k])
        sg_ops.append(("D" if is_del else "I", g))
        if best_k < m:
            _emit_mx_runs(sg_ops, short_s[best_k:], long_s[best_k + g :])
        return _coalesce_ops(sg_ops)

    def _realign_stage6e_subcluster(
        seg_ops: list[tuple[str, int]],
        s_seg: str,
        r_seg: str,
        anchor_m: int,
        dp_cells_left: list[int],
        used_heavy_dp: list[bool],
    ) -> list[tuple[str, int]]:
        n_seg = len(seg_ops)
        out_sub: list[tuple[str, int]] = []
        qp = 0
        tp = 0
        si = 0
        while si < n_seg:
            op, ln = seg_ops[si]
            if op == "M" and ln >= anchor_m:
                out_sub.append((op, ln))
                qp += ln
                tp += ln
                si += 1
                continue
            sj = si
            qe = qp
            te = tp
            ng = 0
            nx = 0
            while sj < n_seg and not (
                seg_ops[sj][0] == "M" and seg_ops[sj][1] >= anchor_m
            ):
                cop, cln = seg_ops[sj]
                if cop == "M":
                    qe += cln
                    te += cln
                elif cop == "X":
                    qe += cln
                    te += cln
                    nx += cln
                elif cop == "I":
                    qe += cln
                    ng += 1
                elif cop == "D":
                    te += cln
                    ng += 1
                sj += 1
            qs_cl = qe - qp
            ts_cl = te - tp
            needs_dp = (ng >= 2) or (ng >= 1 and nx > 0)
            if needs_dp and anchor_m >= 8:
                sg_fast = _try_exact_single_gap_ops(
                    s_seg[qp:qe], r_seg[tp:te]
                )
                if sg_fast is not None:
                    out_sub.extend(sg_fast)
                    qp = qe
                    tp = te
                    si = sj
                    continue
            box_cells = qs_cl * ts_cl
            fits_fast = (
                box_cells <= dp_cells_left[0]
                and (
                    (qs_cl <= 40 and ts_cl <= 40)
                    or (
                        min(qs_cl, ts_cl) <= 24
                        and max(qs_cl, ts_cl) <= EDLIB_MAX_LEN
                        and box_cells <= 1600
                    )
                )
            )
            if needs_dp and fits_fast:
                dp_cells_left[0] -= box_cells
                used_heavy_dp[0] = True
                out_sub.extend(
                    _align_two_piece_affine_asm20(
                        s_seg[qp:qe], r_seg[tp:te]
                    )
                )
            elif (
                needs_dp
                and qs_cl >= min_aligned_core
                and ts_cl >= min_aligned_core
            ):
                # Recover off-diagonal exact/near-exact internal blocks
                # (e.g. canceling D+M+I shattered by Levenshtein NW) before
                # falling back to anchor_m sub-splitting.
                s_cl = s_seg[qp:qe]
                r_cl = r_seg[tp:te]
                best_anc: tuple[int, int, int] | None = None
                probe_starts = [
                    (qs_cl - min_aligned_core) // 2,
                    qs_cl // 4,
                    max(0, (3 * qs_cl) // 4 - min_aligned_core),
                    qs_cl // 8,
                    max(0, (7 * qs_cl) // 8 - min_aligned_core),
                ]
                for ps in dict.fromkeys(probe_starts):
                    if ps < 0 or ps + min_aligned_core > qs_cl:
                        continue
                    kmer = s_cl[ps : ps + min_aligned_core]
                    hit_pos = r_cl.find(kmer)
                    if hit_pos == -1:
                        continue
                    # Reject ambiguous repeats inside cluster
                    if r_cl.find(kmer, hit_pos + 1) != -1:
                        continue
                    al = ps
                    bl = hit_pos
                    while al > 0 and bl > 0 and s_cl[al - 1] == r_cl[bl - 1]:
                        al -= 1
                        bl -= 1
                    ar = ps + min_aligned_core
                    br = hit_pos + min_aligned_core
                    while (
                        ar < qs_cl and br < ts_cl and s_cl[ar] == r_cl[br]
                    ):
                        ar += 1
                        br += 1
                    m_len = ar - al
                    if best_anc is None or m_len > best_anc[0]:
                        best_anc = (m_len, al, bl)
                if best_anc is not None and best_anc[0] >= min_aligned_core:
                    m_len, al, bl = best_anc
                    ar = al + m_len
                    br = bl + m_len
                    for side_idx, (sub_s, sub_r) in enumerate(
                        ((s_cl[:al], r_cl[:bl]), (s_cl[ar:], r_cl[br:]))
                    ):
                        if not sub_s and not sub_r:
                            pass
                        elif not sub_s:
                            out_sub.append(("D", len(sub_r)))
                        elif not sub_r:
                            out_sub.append(("I", len(sub_s)))
                        else:
                            sub_nw = edlib.align(
                                sub_s, sub_r, mode="NW", task="path"
                            )
                            sub_cg = sub_nw.get("cigar")
                            if sub_cg is not None:
                                sub_raw: list[tuple[str, int]] = []
                                for sm in _EDLIB_CIGAR_RE.finditer(sub_cg):
                                    sln = int(sm.group(1))
                                    scd = sm.group(2)
                                    sub_raw.append(
                                        ("M" if scd == "=" else scd, sln)
                                    )
                                sub_ops = _collapse_edlib_gap_spans(
                                    _coalesce_ops(sub_raw), sub_s, sub_r
                                )
                                out_sub.extend(
                                    _realign_stage6e_subcluster(
                                        sub_ops,
                                        sub_s,
                                        sub_r,
                                        8,
                                        dp_cells_left,
                                        used_heavy_dp,
                                    )
                                )
                            else:
                                out_sub.append(("I", len(sub_s)))
                                out_sub.append(("D", len(sub_r)))
                        if side_idx == 0:
                            out_sub.append(("M", m_len))
                elif anchor_m > 4:
                    next_anc = 8 if anchor_m > 8 else 4
                    out_sub.extend(
                        _realign_stage6e_subcluster(
                            seg_ops[si:sj],
                            s_cl,
                            r_cl,
                            next_anc,
                            dp_cells_left,
                            used_heavy_dp,
                        )
                    )
                else:
                    out_sub.extend(seg_ops[si:sj])
            elif needs_dp and anchor_m > 4:
                next_anc = 8 if anchor_m > 8 else 4
                out_sub.extend(
                    _realign_stage6e_subcluster(
                        seg_ops[si:sj],
                        s_seg[qp:qe],
                        r_seg[tp:te],
                        next_anc,
                        dp_cells_left,
                        used_heavy_dp,
                    )
                )
            else:
                out_sub.extend(seg_ops[si:sj])
            qp = qe
            tp = te
            si = sj
        return _coalesce_ops(out_sub)

    def _realign_and_trim_stage6e_slice(
        q_str: str,
        t_sub: str,
        ls: int,
        k_c2: int,
        dp_cells_left: list[int],
        used_heavy_dp: list[bool],
    ) -> tuple[list[tuple[str, int]], int, int, int, int]:
        if not q_str or not t_sub:
            return [], 0, 0, ls, ls
        res = edlib.align(q_str, t_sub, mode="NW", task="path", k=k_c2)
        cigar = res.get("cigar")
        if cigar is None or res.get("editDistance", -1) < 0:
            return [], 0, 0, ls, ls
        raw_ops: list[tuple[str, int]] = []
        for m in _EDLIB_CIGAR_RE.finditer(cigar):
            ln = int(m.group(1))
            code = m.group(2)
            raw_ops.append(("M" if code == "=" else code, ln))
        ops = _coalesce_ops(raw_ops)
        q_len_cur = len(q_str)
        # Provably zero-FN exact length-conservation pre-DP exit: when both
        # ends start/end with ('M', >= 15) anchors, terminal trimming halts
        # immediately (qs=0, qe=q_len, ts=0, te=t_len), so any alignment has
        # h_d_bp >= t_len - q_len and h_i_bp >= q_len - t_len.
        if (
            ops
            and ops[0][0] == "M"
            and ops[0][1] >= _EDLIB_CLUSTER_ANCHOR_M
            and ops[-1][0] == "M"
            and ops[-1][1] >= _EDLIB_CLUSTER_ANCHOR_M
            and abs(len(t_sub) - q_len_cur) >= min_sv_len
        ):
            return [], 0, 0, ls, ls
        if len(ops) >= 2:
            # 1. Losslessly flatten Levenshtein-tied micro-substitutions of the
            # form `(I, 1) + <=3 bp M/X + (D, 1)` (or `D..I`) to diagonal M/X
            # whenever exact asm20 score is >= original (O(n), eliminates
            # spurious gap splits inside dense SNP stretches).
            if len(ops) >= 3:
                flat_ops: list[tuple[str, int]] = []
                fq = 0
                ft = 0
                fi = 0
                fn = len(ops)
                while fi < fn:
                    op0, ln0 = ops[fi]
                    if (
                        (op0 == "I" or op0 == "D")
                        and ln0 == 1
                        and fi + 2 < fn
                    ):
                        opp_g = "D" if op0 == "I" else "I"
                        fj = fi + 1
                        mid_len = 0
                        while (
                            fj < fn
                            and (ops[fj][0] == "M" or ops[fj][0] == "X")
                            and mid_len + ops[fj][1] <= 3
                        ):
                            mid_len += ops[fj][1]
                            fj += 1
                        if (
                            fj < fn
                            and ops[fj][0] == opp_g
                            and ops[fj][1] == 1
                        ):
                            span_m = mid_len + 1
                            diag_ops: list[tuple[str, int]] = []
                            _emit_mx_runs(
                                diag_ops,
                                q_str[fq : fq + span_m],
                                t_sub[ft : ft + span_m],
                            )
                            if _asm20_ops_score(diag_ops) >= _asm20_ops_score(
                                ops[fi : fj + 1]
                            ):
                                flat_ops.extend(diag_ops)
                                fq += span_m
                                ft += span_m
                                fi = fj + 1
                                continue
                    flat_ops.append((op0, ln0))
                    if op0 == "M" or op0 == "X":
                        fq += ln0
                        ft += ln0
                    elif op0 == "I":
                        fq += ln0
                    elif op0 == "D":
                        ft += ln0
                    fi += 1
                ops = _coalesce_ops(flat_ops)

            # 2. Fast pre-collapse Theorem 2 check on gap-free >= 22 bp M/X
            # stretches with >= 70% identity: `_collapse_edlib_gap_spans` never
            # alters gap-free M/X stretches, so if no valid terminal trim
            # (<= min_sv_len - 1) can even reach 85% match rate, exit before
            # gap-span collapsing and affine DP completely.
            def _mark_anchors(
                cur_ops: list[tuple[str, int]],
            ) -> tuple[list[bool], int]:
                n_cur = len(cur_ops)
                anc_flags = [False] * n_cur
                anc_x_cnt = 0
                ci = 0
                while ci < n_cur:
                    cop, cln = cur_ops[ci]
                    if cop == "M" and cln >= _EDLIB_CLUSTER_ANCHOR_M:
                        anc_flags[ci] = True
                        ci += 1
                        continue
                    if cop == "M" or cop == "X":
                        cj = ci
                        while cj < n_cur and (
                            cur_ops[cj][0] == "M" or cur_ops[cj][0] == "X"
                        ):
                            cj += 1
                        a_s = ci
                        a_e = cj
                        if ci > 0 and a_s < a_e and cur_ops[a_s][0] == "X":
                            a_s += 1
                        if (
                            cj < n_cur
                            and a_e > a_s
                            and cur_ops[a_e - 1][0] == "X"
                        ):
                            a_e -= 1
                        if a_e > a_s:
                            run_len = 0
                            run_m = 0
                            for k_op in range(a_s, a_e):
                                rop, rln = cur_ops[k_op]
                                run_len += rln
                                if rop == "M":
                                    run_m += rln
                            if (
                                run_len >= 25 and run_m * 5 >= run_len * 4
                            ) or (
                                run_len >= 40 and run_m * 10 >= run_len * 7
                            ):
                                for k_op in range(a_s, a_e):
                                    anc_flags[k_op] = True
                                    if cur_ops[k_op][0] == "X":
                                        anc_x_cnt += cur_ops[k_op][1]
                        ci = cj
                        continue
                    ci += 1
                return anc_flags, anc_x_cnt

            is_anchor, total_anc_x = _mark_anchors(ops)
            min_len_qt = min(q_len_cur, len(t_sub))
            min_gap_open = 1 if len(t_sub) != q_len_cur else 0
            if (
                total_anc_x > 0
                and ops[0][0] == "M"
                and ops[0][1] >= _EDLIB_CLUSTER_ANCHOR_M
                and ops[-1][0] == "M"
                and ops[-1][1] >= _EDLIB_CLUSTER_ANCHOR_M
                and (min_len_qt - total_anc_x) * 100
                < 85 * (min_len_qt + min_gap_open)
            ):
                return [], 0, 0, ls, ls
            if (
                total_anc_x > 0
                and (q_len_cur - total_anc_x) * 100 < 85 * q_len_cur
            ):
                max_tot = min(
                    min_sv_len - 1,
                    q_len_cur - max(min_aligned_core, q_len_cur // 2),
                )
                # `_trim_edlib_terminal_overhangs` halts at the first/last
                # `('M', >= 15)` anchor, so terminal trim cannot exceed the
                # query bases before the first / after the last such anchor.
                q_bef_m15 = 0
                for top, tln in ops:
                    if top == "M" and tln >= _EDLIB_CLUSTER_ANCHOR_M:
                        break
                    if top != "D":
                        q_bef_m15 += tln
                    if q_bef_m15 >= max_tot:
                        break
                q_aft_m15 = 0
                for top, tln in reversed(ops):
                    if top == "M" and tln >= _EDLIB_CLUSTER_ANCHOR_M:
                        break
                    if top != "D":
                        q_aft_m15 += tln
                    if q_aft_m15 >= max_tot:
                        break
                max_l_tr = min(max_tot, q_bef_m15)
                max_r_tr = min(max_tot, q_aft_m15)
                if max_l_tr == 0 and max_r_tr == 0:
                    return [], 0, 0, ls, ls
                if max_tot >= 0:
                    pref_x = [0] * (max_l_tr + 1)
                    suff_x = [0] * (max_r_tr + 1)
                    r_lo = q_len_cur - max_r_tr
                    q_cur = 0
                    for op_i, (aop, aln) in enumerate(ops):
                        if aop == "X" and is_anchor[op_i]:
                            if max_l_tr > 0 and q_cur < max_l_tr:
                                hi_p = min(q_cur + aln, max_l_tr)
                                for qk in range(q_cur, hi_p):
                                    pref_x[qk + 1] = 1
                            if max_r_tr > 0 and q_cur + aln > r_lo:
                                lo_s = max(q_cur, r_lo)
                                for qk in range(lo_s, q_cur + aln):
                                    suff_x[q_len_cur - qk] = 1
                        if aop != "D":
                            q_cur += aln
                    acc_p = 0
                    for tk in range(1, max_l_tr + 1):
                        acc_p += pref_x[tk]
                        pref_x[tk] = acc_p
                    acc_s = 0
                    best_g = [0] * (max_r_tr + 1)
                    cur_bg = 0
                    for tk in range(1, max_r_tr + 1):
                        acc_s += suff_x[tk]
                        val_g = 100 * acc_s - 15 * tk
                        if val_g > cur_bg:
                            cur_bg = val_g
                        best_g[tk] = cur_bg
                    need_gain = 100 * total_anc_x - 15 * q_len_cur
                    can_reach_85 = False
                    for l_tr in range(max_l_tr + 1):
                        r_lim = min(max_r_tr, max_tot - l_tr)
                        if (
                            100 * pref_x[l_tr] - 15 * l_tr + best_g[r_lim]
                            >= need_gain
                        ):
                            can_reach_85 = True
                            break
                    if not can_reach_85:
                        return [], 0, 0, ls, ls

            ops = _collapse_edlib_gap_spans(ops, q_str, t_sub)
            is_anchor, _ = _mark_anchors(ops)
            n_ops = len(ops)

            realigned: list[tuple[str, int]] = []
            qp = 0
            tp = 0
            idx = 0
            while idx < n_ops:
                if is_anchor[idx]:
                    aop, aln = ops[idx]
                    realigned.append((aop, aln))
                    qp += aln
                    tp += aln
                    idx += 1
                    continue
                j_cl = idx
                qe = qp
                te = tp
                ng = 0
                nx = 0
                while j_cl < n_ops and not is_anchor[j_cl]:
                    cop, cln = ops[j_cl]
                    if cop == "M":
                        qe += cln
                        te += cln
                    elif cop == "X":
                        qe += cln
                        te += cln
                        nx += cln
                    elif cop == "I":
                        qe += cln
                        ng += 1
                    elif cop == "D":
                        te += cln
                        ng += 1
                    j_cl += 1
                if (ng >= 2) or (ng >= 1 and nx > 0):
                    used_heavy_dp[0] = True
                    realigned.extend(
                        _realign_stage6e_subcluster(
                            ops[idx:j_cl],
                            q_str[qp:qe],
                            t_sub[tp:te],
                            _EDLIB_CLUSTER_ANCHOR_M,
                            dp_cells_left,
                            used_heavy_dp,
                        )
                    )
                else:
                    realigned.extend(ops[idx:j_cl])
                qp = qe
                tp = te
                idx = j_cl
            ops = _coalesce_ops(realigned)

        ops, qs, qe, ts, te = _trim_edlib_terminal_overhangs(
            ops, 0, len(q_str), 0, len(t_sub), seq_s=q_str, seq_r=t_sub
        )
        if not ops or qs >= qe or ts >= te:
            return [], 0, 0, ls, ls
        return ops, qs, qe, ls + ts, ls + te

    def _passes_ed100_prefilter(
        q_str: str,
        t_sub: str,
        ed_nw: int,
    ) -> bool:
        if ed_nw < 100:
            return True
        c2_span = len(t_sub)
        min_len_qt = min(q_span, c2_span)
        best_l = 0
        sc_l = 0
        max_sc_l = 0
        for di in range(min_len_qt):
            if q_str[di] == t_sub[di]:
                sc_l += 1
                if sc_l > max_sc_l:
                    max_sc_l = sc_l
                    best_l = di + 1
            else:
                sc_l -= 4
                if sc_l < max_sc_l - 64:
                    break
        best_r = 0
        sc_r = 0
        max_sc_r = 0
        rem_diag = min_len_qt - best_l
        for dj in range(rem_diag):
            if q_str[q_span - 1 - dj] == t_sub[c2_span - 1 - dj]:
                sc_r += 1
                if sc_r > max_sc_r:
                    max_sc_r = sc_r
                    best_r = dj + 1
            else:
                sc_r -= 4
                if sc_r < max_sc_r - 64:
                    break
        unm_q = q_span - best_l - best_r
        unm_t = c2_span - best_l - best_r
        if unm_q >= 2 * min_aligned_core and unm_t >= max(160, min_sv_len):
            t_mid = t_sub[best_l : c2_span - best_r]
            k_mid = 2
            ed_ml = edlib.align(
                q_str[best_l : best_l + min_aligned_core],
                t_mid,
                mode="HW",
                task="distance",
                k=k_mid,
            ).get("editDistance", -1)
            if not (0 <= ed_ml <= k_mid):
                ed_mr = edlib.align(
                    q_str[
                        q_span - best_r - min_aligned_core : q_span - best_r
                    ],
                    t_mid,
                    mode="HW",
                    task="distance",
                    k=k_mid,
                ).get("editDistance", -1)
                if not (0 <= ed_mr <= k_mid):
                    if (
                        best_l < min_aligned_core
                        and best_r < min_aligned_core
                        and unm_q >= 3 * min_aligned_core
                    ):
                        for q_ps in (
                            q_span // 4,
                            max(0, (3 * q_span) // 4 - min_aligned_core),
                        ):
                            ed_mc = edlib.align(
                                q_str[q_ps : q_ps + min_aligned_core],
                                t_mid,
                                mode="HW",
                                task="distance",
                                k=k_mid,
                            ).get("editDistance", -1)
                            if 0 <= ed_mc <= k_mid:
                                return True
                    return False
        return True

    def _screen_slice_copy2(
        q_str: str,
        ls: int,
        r_end: int,
        checked_nw: dict[tuple[int, int], tuple[int, int] | None],
        nw_used: list[int],
        cands: list[tuple[int, int, int, int]],
        nw_limit: int,
    ) -> None:
        c2_span = r_end - ls
        if not (min_c2_span <= c2_span <= max_c2_span):
            return
        if not _edlib_rc_interval_distinct(
            t_start, t_end, ls, r_end, min_sv_len
        ):
            return
        span_key = (ls, r_end)
        if span_key in checked_nw:
            cached = checked_nw[span_key]
            if cached is not None:
                cands.append((cached[0], ls, r_end, cached[1]))
            return
        if nw_used[0] >= nw_limit:
            return
        nw_used[0] += 1

        k_c2 = _calc_k_c2(c2_span)
        ed_nw = edlib.align(
            q_str, target_u[ls:r_end], mode="NW", task="distance", k=k_c2
        ).get("editDistance", -1)
        if not (0 <= ed_nw <= k_c2):
            checked_nw[span_key] = None
            return
        checked_nw[span_key] = (ed_nw, k_c2)
        cands.append((ed_nw, ls, r_end, k_c2))

    val_sub_cache: dict[tuple[str, str], bool] = {}

    def _validate_candidate_dp(
        q_str: str,
        ls: int,
        re: int,
        k_c2: int,
        dp_cells_left: list[int],
        used_heavy_dp: list[bool],
    ) -> bool:
        t_sub = target_u[ls:re]
        h_ops, h_qs, h_qe, h_ts, h_te = _realign_and_trim_stage6e_slice(
            q_str, t_sub, ls, k_c2, dp_cells_left, used_heavy_dp
        )
        if not h_ops:
            return False
        q_al = h_qe - h_qs
        t_al = h_te - h_ts
        if q_al < max(min_aligned_core, q_span // 2) or t_al < min_aligned_core:
            return False
        if not _edlib_rc_interval_distinct(
            t_start, t_end, h_ts, h_te, min_sv_len
        ):
            return False

        h_m = 0
        h_x = 0
        h_i_bp = 0
        h_d_bp = 0
        h_i_open = 0
        h_d_open = 0
        for op, ln in h_ops:
            if op == "M":
                h_m += ln
            elif op == "X":
                h_x += ln
            elif op == "I":
                h_i_bp += ln
                h_i_open += 1
            elif op == "D":
                if ln >= min_sv_len:
                    return False
                h_d_bp += ln
                h_d_open += 1
        if h_d_bp >= min_sv_len:
            return False
        if h_i_bp >= min_sv_len or (h_qs + (q_span - h_qe)) >= min_sv_len:
            return False
        denom = h_m + h_x + h_i_open + h_d_open
        if denom <= 0 or (h_m / denom) + 1e-12 < 0.85:
            return False
        if _asm20_ops_score(h_ops) <= 0 and (h_m / denom) + 1e-12 < 0.95:
            return False
        # Local window identity guard on non-'I' aligned columns (mirrors
        # Criterion 4 so a clean arm cannot carry a low-identity random bridge).
        w_len = min(50, min_sv_len)
        if w_len >= 10:
            col_m: list[int] = []
            col_d: list[int] = []
            for op, ln in h_ops:
                if op == "M":
                    col_m.extend([1] * ln)
                    col_d.extend([1] * ln)
                elif op == "X":
                    col_m.extend([0] * ln)
                    col_d.extend([1] * ln)
                elif op == "D":
                    if col_d:
                        col_d[-1] += 1
                    else:
                        col_m.append(0)
                        col_d.append(1)
            n_cols = len(col_m)
            if n_cols >= w_len:
                wm = sum(col_m[:w_len])
                wd = sum(col_d[:w_len])
                if wd > 0 and (wm / wd) + 1e-12 < 0.70:
                    return False
                for wi in range(w_len, n_cols):
                    wm += col_m[wi] - col_m[wi - w_len]
                    wd += col_d[wi] - col_d[wi - w_len]
                    if wd > 0 and (wm / wd) + 1e-12 < 0.70:
                        return False
        return True

    def _evaluate_candidates(
        q_str: str,
        cands: list[tuple[int, int, int, int]],
        dp_tested: list[tuple[int, int, int, int]],
        dp_cells_left: list[int],
        max_dp: int,
        dedup_r: int = prox_r,
    ) -> bool:
        if not cands:
            return False
        uniq_cands = list(dict.fromkeys(cands))
        uniq_cands.sort(
            key=lambda item: (
                1 if abs((item[2] - item[1]) - q_span) >= min_sv_len else 0,
                0 if (item[0] - abs((item[2] - item[1]) - q_span)) <= 12 else 1,
                item[0],
                item[1],
                item[2],
            )
        )
        dp_used = 0
        val_used = 0
        max_val = max(8, max_dp * 2)
        pf_dedup_r = min(24, dedup_r)
        pf_rejected: list[tuple[int, int, int, int]] = []
        for ed_nw, ls, r_end, k_c2 in uniq_cands:
            if dp_used >= max_dp or val_used >= max_val:
                break
            tier_ed = (
                0 if (ed_nw - abs((r_end - ls) - q_span)) <= 12 else 1
            )
            if any(
                abs(ls - p_ls) <= dedup_r
                and abs(r_end - p_re) <= dedup_r
                and (tier_ed > p_ex or (tier_ed == p_ex and ed_nw >= p_ed))
                for p_ex, p_ed, p_ls, p_re in dp_tested
            ):
                continue
            if ed_nw >= 100 and any(
                abs(ls - p_ls) <= pf_dedup_r
                and abs(r_end - p_re) <= pf_dedup_r
                and (tier_ed > p_ex or (tier_ed == p_ex and ed_nw >= p_ed))
                for p_ex, p_ed, p_ls, p_re in pf_rejected
            ):
                continue
            if not _passes_ed100_prefilter(q_str, target_u[ls:r_end], ed_nw):
                pf_rejected.append((tier_ed, ed_nw, ls, r_end))
                continue
            dp_tested.append((tier_ed, ed_nw, ls, r_end))
            val_used += 1
            used_heavy_dp = [False]
            vck = (q_str, target_u[ls:r_end])
            cached_v = val_sub_cache.get(vck)
            if cached_v is not None:
                if cached_v:
                    return True
                continue
            res_v = _validate_candidate_dp(
                q_str, ls, r_end, k_c2, dp_cells_left, used_heavy_dp
            )
            val_sub_cache[vck] = res_v
            if res_v:
                return True
            if used_heavy_dp[0]:
                dp_used += 1
        return False

    t_excess = max(0, (t_end - t_start) - q_span)

    def _collect_fwd_arm_clusters(
        probe: str,
        p_qs: int,
        p_qe: int,
        k_lim: int,
        max_cl: int = max_probe_clusters,
    ) -> list[tuple[int, int, int, int]]:
        collected: list[list[tuple[int, int, int, int]]] = []
        r_cut = min(t_end, t_start + p_qs + k_lim + 1 + t_excess)
        r_hits = _probe_slice_bounds(probe, r_cut, len_r, k_lim, max_cl)
        if r_hits:
            collected.append(r_hits)
            if (
                len(r_hits) <= 12
                and any(
                    min(t_end, cl[3]) > max(t_start, cl[0]) for cl in r_hits
                )
                and t_end < len_r
            ):
                collected.append(
                    _probe_slice_bounds(probe, t_end, len_r, k_lim, max_cl)
                )
        l_cut = max(0, min(t_end, t_start + p_qe) - (k_lim + 1 + t_excess))
        if p_qe == q_span:
            l_cut = max(t_start, t_end - (k_lim + 1 + t_excess))
        l_hits = _probe_slice_bounds(probe, 0, l_cut, k_lim, max_cl)
        if l_hits:
            collected.append(l_hits)
            if (
                len(l_hits) <= 12
                and any(
                    min(t_end, cl[3]) > max(t_start, cl[0]) for cl in l_hits
                )
                and t_start > 0
            ):
                collected.append(
                    _probe_slice_bounds(probe, 0, t_start, k_lim, max_cl)
                )
        return _merge_unique_clusters(collected)

    def _collect_rc_arm_clusters(
        probe: str,
        k_lim: int,
        max_cl: int = max_probe_clusters,
    ) -> list[tuple[int, int, int, int]]:
        p_len = len(probe)
        collected: list[list[tuple[int, int, int, int]]] = []
        hits = _probe_slice_bounds(probe, 0, len_r, k_lim, max_cl)
        if hits:
            collected.append(hits)
            if len(hits) <= 12:
                ov_hits = [
                    cl
                    for cl in hits
                    if min(t_end, cl[3]) > max(t_start, cl[0])
                ]
                if ov_hits:
                    ov_max_span = max(cl[3] - cl[0] for cl in ov_hits)
                    shift_rc = k_lim + 1 + max(0, ov_max_span - p_len)
                    rc_r_cut = min(
                        max(cl[3] for cl in ov_hits),
                        max(cl[0] for cl in ov_hits) + shift_rc,
                    )
                    collected.append(
                        _probe_slice_bounds(
                            probe, rc_r_cut, len_r, k_lim, max_cl
                        )
                    )
                    rc_l_cut = max(
                        min(cl[0] for cl in ov_hits),
                        min(cl[3] for cl in ov_hits) - shift_rc,
                    )
                    collected.append(
                        _probe_slice_bounds(probe, 0, rc_l_cut, k_lim, max_cl)
                    )
        return _merge_unique_clusters(collected)

    def _screen_local_from_left(
        q_str: str,
        r_probe: str,
        l_hits: Sequence[tuple[int, int, int, int]],
        checked_nw: dict[tuple[int, int], tuple[int, int] | None],
        nw_used: list[int],
        cands: list[tuple[int, int, int, int]],
        nw_limit: int,
    ) -> None:
        for lc_smin, lc_smax, _, _ in l_hits:
            if nw_used[0] >= nw_limit:
                break
            win_rs = max(
                lc_smin + 1,
                lc_smin + min_c2_span - arm_len - arm_k - 4,
            )
            win_re = min(len_r, lc_smax + max_c2_span + arm_k + 4)
            for _, _, rc_emin, rc_emax in _probe_slice_bounds(
                r_probe, win_rs, win_re, arm_k, 6
            ):
                _screen_slice_copy2(
                    q_str,
                    lc_smin,
                    rc_emax,
                    checked_nw,
                    nw_used,
                    cands,
                    nw_limit,
                )
                if lc_smax > lc_smin or rc_emin < rc_emax:
                    _screen_slice_copy2(
                        q_str,
                        lc_smax,
                        rc_emin,
                        checked_nw,
                        nw_used,
                        cands,
                        nw_limit,
                    )

    def _screen_local_from_right(
        q_str: str,
        l_probe: str,
        r_hits: Sequence[tuple[int, int, int, int]],
        checked_nw: dict[tuple[int, int], tuple[int, int] | None],
        nw_used: list[int],
        cands: list[tuple[int, int, int, int]],
        nw_limit: int,
    ) -> None:
        for _, _, rc_emin, rc_emax in r_hits:
            if nw_used[0] >= nw_limit:
                break
            win_ls = max(0, rc_emin - max_c2_span - arm_k - 4)
            win_le = min(
                rc_emax,
                rc_emax - min_c2_span + arm_len + arm_k + 4,
            )
            for lc_smin, lc_smax, _, _ in _probe_slice_bounds(
                l_probe, win_ls, win_le, arm_k, 6
            ):
                _screen_slice_copy2(
                    q_str,
                    lc_smin,
                    rc_emax,
                    checked_nw,
                    nw_used,
                    cands,
                    nw_limit,
                )
                if lc_smax > lc_smin or rc_emin < rc_emax:
                    _screen_slice_copy2(
                        q_str,
                        lc_smax,
                        rc_emin,
                        checked_nw,
                        nw_used,
                        cands,
                        nw_limit,
                    )

    def _probe_interior_arm_fallback(
        q_str: str,
        is_rc: bool,
        checked_nw: dict[tuple[int, int], tuple[int, int] | None],
        dp_tested: list[tuple[int, int, int, int]],
        dp_cells_left: list[int],
    ) -> bool:
        mid = q_span // 2
        k3_arm = min(q_span // 3, (q_span - (min_sv_len - 1)) // 2)
        inset = min(min_sv_len - 1, q_span // 5)
        raw_probes: list[tuple[int, int]] = []
        if mid > arm_len:
            raw_probes.append((0, mid))
            raw_probes.append((q_span - mid, q_span))
        if k3_arm >= min_aligned_core and k3_arm != mid:
            raw_probes.append((0, k3_arm))
            raw_probes.append((q_span - k3_arm, q_span))
        if (min_sv_len - 1) // 2 + 4 > int(0.10 * q_span):
            if inset > 0 and mid - inset >= min_aligned_core:
                raw_probes.append((inset, mid))
            if inset > 0 and (q_span - inset) - mid >= min_aligned_core:
                raw_probes.append((mid, q_span - inset))
        interior_probes = list(dict.fromkeys(raw_probes))
        if not interior_probes:
            return False

        tip_offsets = list(
            dict.fromkeys(
                off
                for off in (
                    0,
                    min(arm_len, max(1, (min_sv_len - 1) // 3)),
                    min(2 * arm_len, max(1, (2 * (min_sv_len - 1)) // 3)),
                    min(min_sv_len - 1, q_span - arm_len),
                )
                if 0 <= off and off + arm_len <= q_span
            )
        )

        nw_used_fb = [0]
        cands_comp: list[tuple[int, int, int, int]] = []
        probed_est_l: list[int] = []
        probed_est_r: list[int] = []
        min_shift = min(min_sv_len, max(1, (t_end - t_start) // 2 + 1))

        for p_qs, p_qe in interior_probes:
            if nw_used_fb[0] >= max_nw_fb:
                break
            sub = q_str[p_qs:p_qe]
            if _is_low_complexity_probe(sub):
                continue
            la = p_qe - p_qs
            la_k = _edlib_core_max_ed(la)
            clusters = (
                _collect_rc_arm_clusters(sub, la_k, 8)
                if is_rc
                else _collect_fwd_arm_clusters(sub, p_qs, p_qe, la_k, 8)
            )
            if not clusters:
                continue
            q_left = p_qs
            q_right = q_span - p_qe
            anchor_on_left = q_left <= q_right

            long_opp: list[tuple[int, int, int]] = []
            tip_opp: list[tuple[int, int, int]] = []
            if anchor_on_left:
                if q_right >= min_aligned_core:
                    long_opp.append(
                        (p_qe, q_span, _edlib_core_max_ed(q_right))
                    )
                if k3_arm >= min_aligned_core and q_span - k3_arm != p_qe:
                    long_opp.append(
                        (q_span - k3_arm, q_span, _edlib_core_max_ed(k3_arm))
                    )
                for off_r in tip_offsets:
                    op_s = q_span - off_r - arm_len
                    op_e = q_span - off_r
                    if op_s >= p_qs and op_e <= q_span:
                        tip_opp.append((op_s, op_e, arm_k))
            else:
                if q_left >= min_aligned_core:
                    long_opp.append((0, p_qs, _edlib_core_max_ed(q_left)))
                if k3_arm >= min_aligned_core and k3_arm != p_qs:
                    long_opp.append((0, k3_arm, _edlib_core_max_ed(k3_arm)))
                for off_l in tip_offsets:
                    op_s = off_l
                    op_e = off_l + arm_len
                    if op_e <= p_qe:
                        tip_opp.append((op_s, op_e, arm_k))
            long_opp = list(dict.fromkeys(long_opp))
            tip_opp = list(dict.fromkeys(tip_opp))

            for c_smin, c_smax, c_emin, c_emax in clusters[:8]:
                if nw_used_fb[0] >= max_nw_fb:
                    break
                w_s = max(
                    0,
                    c_smin
                    - (q_left + (min_sv_len - 1) if q_left > 0 else 0)
                    - 8,
                )
                w_e = min(
                    len_r,
                    c_emax
                    + (q_right + (min_sv_len - 1) if q_right > 0 else 0)
                    + 8,
                )
                if w_s <= t_start and w_e >= t_end:
                    if c_smin >= t_end:
                        w_s = max(w_s, t_start + min_shift)
                    elif c_emax <= t_start:
                        w_e = min(w_e, t_end - min_shift)
                    elif not is_rc:
                        continue
                if w_e - w_s < min_c2_span:
                    continue

                # 1. Complementary opposite-arm pairing: try long opposite
                # arms first; only probe inset 19-mers if no long opposite
                # arm matched (e.g. off-center short arm or >20% double trunc).
                matched_long = False
                if anchor_on_left:
                    ls0_est = max(0, c_smin - q_left)
                    ls1_est = max(0, c_smax - q_left)
                    if any(abs(ls0_est - pe) <= 8 for pe in probed_est_l):
                        continue
                    for probe_group_idx, p_group in enumerate(
                        (long_opp, tip_opp)
                    ):
                        if probe_group_idx == 1 and matched_long:
                            break
                        for op_s, op_e, op_k in p_group:
                            if nw_used_fb[0] >= max_nw_fb:
                                break
                            op_seq = q_str[op_s:op_e]
                            op_len = op_e - op_s
                            tail_r = q_span - op_e
                            ow_s = max(
                                w_s,
                                c_smin
                                + max(0, op_s - p_qs - (min_sv_len - 1))
                                - op_k
                                - 8,
                            )
                            ow_e = min(
                                w_e,
                                c_smax
                                + (op_e - p_qs)
                                + (min_sv_len - 1)
                                + op_k
                                + 8,
                            )
                            if ow_e - ow_s < max(1, op_len - op_k):
                                continue
                            opp_hits = _probe_slice_bounds(
                                op_seq, ow_s, ow_e, op_k, 4
                            )
                            if not opp_hits:
                                continue
                            if probe_group_idx == 0:
                                matched_long = True
                            for _, _, oce0, oce1 in opp_hits:
                                re_cand = min(len_r, oce1 + tail_r)
                                _screen_slice_copy2(
                                    q_str,
                                    ls0_est,
                                    re_cand,
                                    checked_nw,
                                    nw_used_fb,
                                    cands_comp,
                                    max_nw_fb,
                                )
                                if ls1_est > ls0_est or oce0 < oce1:
                                    _screen_slice_copy2(
                                        q_str,
                                        ls1_est,
                                        min(len_r, oce0 + tail_r),
                                        checked_nw,
                                        nw_used_fb,
                                        cands_comp,
                                        max_nw_fb,
                                    )
                            break
                    if matched_long:
                        probed_est_l.append(ls0_est)
                else:
                    re0_est = min(len_r, c_emax + q_right)
                    re1_est = min(len_r, c_emin + q_right)
                    if any(abs(re0_est - pe) <= 8 for pe in probed_est_r):
                        continue
                    for probe_group_idx, p_group in enumerate(
                        (long_opp, tip_opp)
                    ):
                        if probe_group_idx == 1 and matched_long:
                            break
                        for op_s, op_e, op_k in p_group:
                            if nw_used_fb[0] >= max_nw_fb:
                                break
                            op_seq = q_str[op_s:op_e]
                            op_len = op_e - op_s
                            lead_l = op_s
                            ow_s = max(
                                w_s,
                                c_emin
                                - (p_qe - op_s)
                                - (min_sv_len - 1)
                                - op_k
                                - 8,
                            )
                            ow_e = min(
                                w_e,
                                c_emax
                                - max(0, p_qe - op_e - (min_sv_len - 1))
                                + op_k
                                + 8,
                            )
                            if ow_e - ow_s < max(1, op_len - op_k):
                                continue
                            opp_hits = _probe_slice_bounds(
                                op_seq, ow_s, ow_e, op_k, 4
                            )
                            if not opp_hits:
                                continue
                            if probe_group_idx == 0:
                                matched_long = True
                            for ocs0, ocs1, _, _ in opp_hits:
                                ls_cand = max(0, ocs0 - lead_l)
                                _screen_slice_copy2(
                                    q_str,
                                    ls_cand,
                                    re0_est,
                                    checked_nw,
                                    nw_used_fb,
                                    cands_comp,
                                    max_nw_fb,
                                )
                                if ocs1 > ocs0 or re1_est < re0_est:
                                    _screen_slice_copy2(
                                        q_str,
                                        max(0, ocs1 - lead_l),
                                        re1_est,
                                        checked_nw,
                                        nw_used_fb,
                                        cands_comp,
                                        max_nw_fb,
                                    )
                            break
                    if matched_long:
                        probed_est_r.append(re0_est)

        return _evaluate_candidates(
            q_str,
            cands_comp,
            dp_tested,
            dp_cells_left,
            max_dp_fb,
            dedup_r=prox_r,
        )

    def _probe_strand_copy2(q_str: str, is_rc: bool) -> bool:
        l_probe = q_str[:arm_len]
        r_probe = q_str[-arm_len:]
        l_low = _is_low_complexity_probe(l_probe)
        r_low = _is_low_complexity_probe(r_probe)
        checked_nw: dict[tuple[int, int], tuple[int, int] | None] = {}
        nw_used_outer = [0]
        cands_outer: list[tuple[int, int, int, int]] = []
        dp_tested: list[tuple[int, int, int, int]] = []
        dp_cells_left = [800]

        # Case A: Both outer 19-mers are low-complexity -> go straight to
        # bounded interior-arm fallback without target-wide 19 bp scans.
        if l_low and r_low:
            return _probe_interior_arm_fallback(
                q_str, is_rc, checked_nw, dp_tested, dp_cells_left
            )

        # Case B: Exactly one outer 19-mer is low-complexity -> scan only the
        # complex outer arm target-wide and probe local windows; fall back
        # unconditionally to interior arms if unconfirmed.
        if l_low or r_low:
            if not l_low:
                l_hits = (
                    _collect_rc_arm_clusters(l_probe, arm_k)
                    if is_rc
                    else _collect_fwd_arm_clusters(l_probe, 0, arm_len, arm_k)
                )
                if l_hits and len(l_hits) <= 12:
                    _screen_local_from_left(
                        q_str,
                        r_probe,
                        l_hits,
                        checked_nw,
                        nw_used_outer,
                        cands_outer,
                        max_nw_outer,
                    )
                    if _evaluate_candidates(
                        q_str,
                        cands_outer,
                        dp_tested,
                        dp_cells_left,
                        max_dp_outer,
                    ):
                        return True
            else:
                r_hits = (
                    _collect_rc_arm_clusters(r_probe, arm_k)
                    if is_rc
                    else _collect_fwd_arm_clusters(
                        r_probe, q_span - arm_len, q_span, arm_k
                    )
                )
                if r_hits and len(r_hits) <= 12:
                    _screen_local_from_right(
                        q_str,
                        l_probe,
                        r_hits,
                        checked_nw,
                        nw_used_outer,
                        cands_outer,
                        max_nw_outer,
                    )
                    if _evaluate_candidates(
                        q_str,
                        cands_outer,
                        dp_tested,
                        dp_cells_left,
                        max_dp_outer,
                    ):
                        return True
            return _probe_interior_arm_fallback(
                q_str, is_rc, checked_nw, dp_tested, dp_cells_left
            )

        # Case C: Both outer 19-mers are complex. Collect both arms without
        # early return so single-ended tip damage or ed=0 decoy masking falls
        # through cleanly to local opposite-arm and interior-arm probing.
        if is_rc:
            l_hits = _collect_rc_arm_clusters(l_probe, arm_k)
            r_hits = _collect_rc_arm_clusters(r_probe, arm_k)
        else:
            l_hits = _collect_fwd_arm_clusters(l_probe, 0, arm_len, arm_k)
            r_hits = _collect_fwd_arm_clusters(
                r_probe, q_span - arm_len, q_span, arm_k
            )

        if not (len(l_hits) > 12 and len(r_hits) > 12):
            # 1. Fast O(|L| + |R|) two-pointer pair screen across L_hits & R_hits
            if l_hits and r_hits:
                r_idx = 0
                n_r = len(r_hits)
                for lc_smin, lc_smax, _, _ in l_hits:
                    if nw_used_outer[0] >= max_nw_outer:
                        break
                    while (
                        r_idx < n_r and r_hits[r_idx][3] < lc_smin + min_c2_span
                    ):
                        r_idx += 1
                    j = r_idx
                    rep_grid = len(l_hits) > 4 and n_r > 4
                    while j < n_r and r_hits[j][2] <= lc_smax + max_c2_span:
                        _, _, rc_emin, rc_emax = r_hits[j]
                        if rep_grid:
                            _screen_slice_copy2(
                                q_str,
                                lc_smax,
                                rc_emin,
                                checked_nw,
                                nw_used_outer,
                                cands_outer,
                                max_nw_outer,
                            )
                        else:
                            _screen_slice_copy2(
                                q_str,
                                lc_smin,
                                rc_emax,
                                checked_nw,
                                nw_used_outer,
                                cands_outer,
                                max_nw_outer,
                            )
                            if lc_smax > lc_smin or rc_emin < rc_emax:
                                _screen_slice_copy2(
                                    q_str,
                                    lc_smax,
                                    rc_emin,
                                    checked_nw,
                                    nw_used_outer,
                                    cands_outer,
                                    max_nw_outer,
                                )
                        j += 1

            # 2. Local opposite-arm screen for non-repetitive hits (defeats
            # single-arm decoys before entering interior-arm fallback)
            if l_hits and len(l_hits) <= 4:
                _screen_local_from_left(
                    q_str,
                    r_probe,
                    l_hits,
                    checked_nw,
                    nw_used_outer,
                    cands_outer,
                    max_nw_outer,
                )
            if r_hits and len(r_hits) <= 4:
                _screen_local_from_right(
                    q_str,
                    l_probe,
                    r_hits,
                    checked_nw,
                    nw_used_outer,
                    cands_outer,
                    max_nw_outer,
                )

            if _evaluate_candidates(
                q_str,
                cands_outer,
                dp_tested,
                dp_cells_left,
                max_dp_outer,
            ):
                return True

        return _probe_interior_arm_fallback(
            q_str, is_rc, checked_nw, dp_tested, dp_cells_left
        )

    if _probe_strand_copy2(q_core_u, is_rc=False):
        return True
    if _probe_strand_copy2(rc_q_core_u, is_rc=True):
        return True
    return False


def _has_distinct_edlib_del_split_copy(
    q_core: str,
    target: str,
    ops: Sequence[tuple[str, int]],
    t_start: int,
    t_end: int,
    min_sv_len: int,
    min_aligned_core: int,
    rc_q_core: str = "",
    m_total: int = 0,
    fwd_core_score: int = 0,
    trimmed_right: bool = False,
    trimmed_left: bool = False,
    bleed_k_pad: int = 0,
) -> bool:
    """
    Stage 6D internal-indel multi-arm Criterion 1 check (`C1-r4 NEW Bug 3`,
    `R1-r5 Major F1` Part 1, `C1-r5 NEW Bug 4`, `C1-r5 NEW Bug 5`,
    `R2-r5 F1/F2/F3`).

    Handles:
    1. Distinct forward/RC copies carrying divergent deleted sequences `D1 != D2`
       and/or compound `I + D` indels (splits `q_core` greedily on both internal
       `'D'` runs and internal `'I'` runs `>= 4 bp` into `2..4` reference-homologous
       arms, tracking expected query insertion gap `q_ins_gap` across each junction
       and keeping `total_net_del < min_sv_len` and `total_q_ins < min_sv_len`).
    2. Shifted (`1..min_sv_len-1 bp`) same-locus RC palindrome orientation ties
       across internal `'D'` / `I+D` indels (`is_rc=True` bypasses raw distinctness
       and evaluates `_is_edlib_same_locus_rc_tie` on valid shifted trimmed spans).
    3. Periodic/microsatellite arm `tol=8` cluster-bound re-evaluation in
       `_validate_chain` (`C1-r5 NEW Bug 5`), forward-strand `ov_hits` shifted-slice
       fallback (`R2-r5 F3`), reachable 1-level `allow_subsplit` fallback (`R2-r5 F2`),
       and strictly per-probe bounded latency (`R2-r5 F1`: chunked zero-FN segment
       reachability pre-filter + reverse-topological monotonic failed-node cache +
       deterministic `max_probe_steps` budget).
    """
    len_r = len(target)
    q_span = len(q_core)
    if min_sv_len <= 0 or q_span < 2 * min_aligned_core:
        return False

    # Collect internal 'D' and 'I' events: (ev_type, q_s, q_e, t_s, t_e, ln)
    indel_events: list[tuple[str, int, int, int, int, int]] = []
    q_pos = 0
    t_pos = t_start
    for op, ln in ops:
        if op == "M" or op == "X":
            q_pos += ln
            t_pos += ln
        elif op == "I":
            indel_events.append(("I", q_pos, q_pos + ln, t_pos, t_pos, ln))
            q_pos += ln
        elif op == "D":
            indel_events.append(("D", q_pos, q_pos, t_pos, t_pos + ln, ln))
            t_pos += ln
    if not indel_events:
        return False

    # Greedily split into 2..4 reference-homologous segments at largest valid
    # internal 'D' or >= 4 bp 'I' runs. Standard splits require >= min_aligned_core
    # on both sides; secondary short-arm splits allow one arm in
    # [min_sub_arm, min_aligned_core - 1] for substantial indels (>= 8 bp) while
    # preserving at least one >= min_aligned_core outer anchor arm.
    intervals: list[tuple[int, int, int, int]] = [(0, q_span, t_start, t_end)]
    max_segs = 4
    min_sub_arm = min(min_aligned_core, _EDLIB_MIN_CLEAN_INS_ARM)
    while len(intervals) < max_segs:
        best_split: tuple[int, int, int, int, int, int, int, int, int] | None = None
        for idx, (qi0, qi1, ti0, ti1) in enumerate(intervals):
            iv_d_bp = sum(
                eln
                for et, eq0, eq1, _, _, eln in indel_events
                if et == "D" and qi0 < eq0 and eq1 < qi1
            )
            iv_i_bp = sum(
                eln
                for et, eq0, eq1, _, _, eln in indel_events
                if et == "I" and qi0 < eq0 and eq1 < qi1
            )
            if iv_d_bp < 4 and iv_i_bp < 4:
                continue
            iv_d_thresh = max(3, _edlib_core_max_ed(qi1 - qi0) // 3)
            for et, eq0, eq1, et0, et1, eln in indel_events:
                if not (qi0 < eq0 and eq1 < qi1 and ti0 < et0 and et1 < ti1):
                    continue
                if et == "D":
                    if iv_d_bp < 4 or (eln < 4 and iv_d_bp <= iv_d_thresh):
                        continue
                else:
                    if eln < 4:
                        continue
                lq = eq0 - qi0
                rq = qi1 - eq1
                lt = et0 - ti0
                rt = ti1 - et1
                is_std = (
                    lq >= min_aligned_core
                    and rq >= min_aligned_core
                    and lt >= min_aligned_core
                    and rt >= min_aligned_core
                )
                if not is_std:
                    if eln < _EDLIB_MIN_CLEAN_INS_ARM:
                        continue
                    if (
                        lq < min_sub_arm
                        or rq < min_sub_arm
                        or lt < min_sub_arm
                        or rt < min_sub_arm
                    ):
                        continue
                    left_ok = lq >= min_aligned_core and lt >= min_aligned_core
                    right_ok = rq >= min_aligned_core and rt >= min_aligned_core
                    if not (left_ok or right_ok):
                        continue
                    new_first_q = lq if idx == 0 else (intervals[0][1] - intervals[0][0])
                    new_last_q = (
                        rq
                        if idx == len(intervals) - 1
                        else (intervals[-1][1] - intervals[-1][0])
                    )
                    if (
                        new_first_q < min_aligned_core
                        and new_last_q < min_aligned_core
                    ):
                        continue
                cand_key = (
                    1 if is_std else 0,
                    eln,
                    -idx,
                    -eq0,
                    idx,
                    eq0,
                    eq1,
                    et0,
                    et1,
                )
                if best_split is None or cand_key[:4] > best_split[:4]:
                    best_split = cand_key
        if best_split is None:
            break
        _, _, _, _, idx, eq0, eq1, et0, et1 = best_split
        qi0, qi1, ti0, ti1 = intervals[idx]
        intervals[idx : idx + 1] = [
            (qi0, eq0, ti0, et0),
            (eq1, qi1, et1, ti1),
        ]

    if len(intervals) < 2:
        return False

    def _seg_max_ed(slen: int) -> int:
        if slen < 15:
            return 0
        return _edlib_core_max_ed(slen)

    # (seg_seq, primary_ts, primary_te, seg_k)
    fwd_segs: list[tuple[str, int, int, int]] = [
        (q_core[qi0:qi1], ti0, ti1, _seg_max_ed(qi1 - qi0))
        for qi0, qi1, ti0, ti1 in intervals
    ]
    fwd_gaps: list[int] = [
        intervals[j + 1][0] - intervals[j][1]
        for j in range(len(intervals) - 1)
    ]
    rc_segs: list[tuple[str, int, int, int]] = [
        (revcomp(seq), ti0, ti1, seg_k)
        for seq, ti0, ti1, seg_k in reversed(fwd_segs)
    ]
    rc_gaps: list[int] = list(reversed(fwd_gaps))

    if not rc_q_core:
        rc_q_core = revcomp(q_core)

    max_local_clusters = 8
    max_anchor_clusters = 48
    max_probe_steps = 128
    max_validate_calls = 24

    # Strictly per-call local trim cache (zero cross-call state, 100% deterministic):
    # avoids recomputing identical NW/asm20 trims on repeated clusters across paths.
    seg_trim_cache: dict[
        tuple[str, int, int, bool], tuple[int, int, int, int, int] | None
    ] = {}

    def _eval_seg_slice(
        seg_seq: str, s_lo: int, s_hi: int, prefer_right: bool = False
    ) -> tuple[int, int, int, int, int] | None:
        if s_hi <= s_lo:
            return None
        ck = (seg_seq, s_lo, s_hi, prefer_right)
        if ck in seg_trim_cache:
            return seg_trim_cache[ck]
        seg_len = len(seg_seq)
        if s_hi - s_lo >= seg_len:
            ex_pos = (
                target.rfind(seg_seq, s_lo, s_hi)
                if prefer_right
                else target.find(seg_seq, s_lo, s_hi)
            )
            if ex_pos != -1:
                res_ev = (0, seg_len, ex_pos, ex_pos + seg_len, seg_len)
                seg_trim_cache[ck] = res_ev
                return res_ev
        h_ops, h_qs, h_qe, h_ts, h_te = _realign_and_trim_edlib_slice(
            seg_seq, target[s_lo:s_hi], 0, s_lo
        )
        if not h_ops:
            seg_trim_cache[ck] = None
            return None
        q_al = h_qe - h_qs
        t_al = h_te - h_ts
        if seg_len < min_aligned_core:
            seg_floor = seg_len if seg_len < 15 else max(12, seg_len - 2)
        else:
            seg_floor = min(min_aligned_core, max(12, seg_len // 2))
        if q_al < seg_floor or t_al < seg_floor:
            seg_trim_cache[ck] = None
            return None
        if _asm20_ops_score(h_ops) <= 0:
            seg_trim_cache[ck] = None
            return None
        res_ev = (h_qs, h_qe, h_ts, h_te, seg_len)
        seg_trim_cache[ck] = res_ev
        return res_ev

    def _validate_chain(
        chain_seqs: Sequence[str],
        chain_bounds: Sequence[tuple[int, int, int, int]],
        chain_gaps: Sequence[int],
        is_rc: bool,
    ) -> bool:
        # Bypass raw distinctness when is_rc=True so < min_sv_len shifted
        # same-locus palindrome ties reach trimmed tie evaluation (R1-r5 F1).
        if not is_rc and not _edlib_rc_interval_distinct(
            t_start, t_end, chain_bounds[0][0], chain_bounds[-1][3], min_sv_len
        ):
            return False

        total_q_ins = sum(chain_gaps)
        if total_q_ins >= min_sv_len:
            return False

        n_ch = len(chain_seqs)
        segs_q_total = sum(len(sq) for sq in chain_seqs)
        trimmed_opt: list[tuple[int, int, int, int, int] | None] = [
            None
        ] * n_ch
        # Evaluate longest segments first (e.g. unique u60/u150 arms before short
        # periodic arms) and reject any junction whose proven zero-FN lower bound
        # already exceeds its max_j_gap before trimming remaining segments.
        eval_order = sorted(range(n_ch), key=lambda i: -len(chain_seqs[i]))
        for s_i in eval_order:
            c_smin, _, _, c_emax = chain_bounds[s_i]
            ev = _eval_seg_slice(chain_seqs[s_i], c_smin, c_emax)
            if ev is None:
                return False
            trimmed_opt[s_i] = ev
            if s_i > 0:
                q_ig = chain_gaps[s_i - 1]
                mj_l = (
                    min(4, min_sv_len - 1) if q_ig > 0 else (min_sv_len - 1)
                )
                l_smin_nb, _, _, l_emax_nb = chain_bounds[s_i - 1]
                l_seq_nb = chain_seqs[s_i - 1]
                l_slack = (
                    0
                    if target.find(l_seq_nb, l_smin_nb, l_emax_nb) != -1
                    else _seg_max_ed(len(l_seq_nb)) + 2
                )
                r_start_lo = ev[2] - ev[0]
                if r_start_lo - (l_emax_nb + l_slack) > mj_l:
                    c_r_emin = chain_bounds[s_i][2]
                    if c_r_emin < c_emax and c_smin < c_r_emin:
                        alt_r0 = _eval_seg_slice(
                            chain_seqs[s_i], c_smin, c_r_emin
                        )
                        if (
                            alt_r0 is not None
                            and (alt_r0[2] - alt_r0[0]) < r_start_lo
                        ):
                            r_start_lo = alt_r0[2] - alt_r0[0]
                    if r_start_lo - (l_emax_nb + l_slack) > mj_l:
                        return False
            if s_i + 1 < n_ch:
                q_ig = chain_gaps[s_i]
                mj_r = (
                    min(4, min_sv_len - 1) if q_ig > 0 else (min_sv_len - 1)
                )
                r_smin_nb, _, _, r_emax_nb = chain_bounds[s_i + 1]
                r_seq_nb = chain_seqs[s_i + 1]
                r_slack = (
                    0
                    if target.find(r_seq_nb, r_smin_nb, r_emax_nb) != -1
                    else _seg_max_ed(len(r_seq_nb)) + 2
                )
                l_end_hi = ev[3] + (ev[4] - ev[1])
                if (r_smin_nb - r_slack) - l_end_hi > mj_r:
                    c_l_smax = chain_bounds[s_i][1]
                    if c_l_smax > c_smin and c_l_smax < c_emax:
                        alt_l0 = _eval_seg_slice(
                            chain_seqs[s_i],
                            c_l_smax,
                            c_emax,
                            prefer_right=True,
                        )
                        if alt_l0 is not None and (
                            alt_l0[3] + (alt_l0[4] - alt_l0[1]) > l_end_hi
                        ):
                            l_end_hi = alt_l0[3] + (alt_l0[4] - alt_l0[1])
                    if (r_smin_nb - r_slack) - l_end_hi > mj_r:
                        return False
        trimmed: list[tuple[int, int, int, int, int]] = [
            ev for ev in trimmed_opt if ev is not None
        ]

        def _calc_j_gap(j_idx: int) -> tuple[int, int]:
            _, prev_qe, _, prev_te, prev_len = trimmed[j_idx]
            next_qs, _, next_ts, _, _ = trimmed[j_idx + 1]
            q_ins_gap = chain_gaps[j_idx]
            ng = (next_ts - prev_te) - ((prev_len - prev_qe) + next_qs)
            mj = (
                min(4, min_sv_len - 1) if q_ins_gap > 0 else (min_sv_len - 1)
            )
            return ng, mj

        # Verify trimmed junction net gaps stay strictly within bounds:
        # - across 'D' junctions (q_ins_gap == 0): net_gap in [-4, min_sv_len - 1]
        # - across 'I' junctions (q_ins_gap > 0): reference arms abut tightly
        #   with net_gap in [-4, min(4, min_sv_len - 1)] without adding q_ins_gap
        #   into deletion gap.
        # If any junction exceeds bounds (or cumulative total_net_del >= min_sv_len)
        # due to NW left/right bias inside an overwide tol=8 periodic-arm cluster
        # [c_smin, c_emax), re-evaluate on rightmost [c_smax, c_emax) and/or
        # leftmost [c_smin, c_emin) before rejecting (C1-r5 NEW Bug 5).
        n_j = len(trimmed) - 1
        j_info = [_calc_j_gap(j_idx) for j_idx in range(n_j)]
        init_pos_del = sum(ng for ng, _ in j_info if ng > 0)
        needs_adjust = (
            init_pos_del >= min_sv_len
            or any(ng < -4 or ng > mj for ng, mj in j_info)
        )
        if needs_adjust:
            for idx_j in range(n_j):
                net_gap, max_j_gap = _calc_j_gap(idx_j)
                if net_gap > max_j_gap or (
                    init_pos_del >= min_sv_len and net_gap > 0
                ):
                    c_smin_l, c_smax_l, _, c_emax_l = chain_bounds[idx_j]
                    if c_smax_l > c_smin_l and c_smax_l < c_emax_l:
                        alt_l = _eval_seg_slice(
                            chain_seqs[idx_j],
                            c_smax_l,
                            c_emax_l,
                            prefer_right=True,
                        )
                        if alt_l is not None:
                            old_l = trimmed[idx_j]
                            trimmed[idx_j] = alt_l
                            new_ng, _ = _calc_j_gap(idx_j)
                            if new_ng < net_gap:
                                net_gap = new_ng
                            else:
                                trimmed[idx_j] = old_l
                    if net_gap > max_j_gap or (
                        init_pos_del >= min_sv_len and net_gap > 0
                    ):
                        c_smin_r, _, c_emin_r, c_emax_r = chain_bounds[
                            idx_j + 1
                        ]
                        if c_emin_r < c_emax_r and c_smin_r < c_emin_r:
                            alt_r = _eval_seg_slice(
                                chain_seqs[idx_j + 1], c_smin_r, c_emin_r
                            )
                            if alt_r is not None:
                                old_r = trimmed[idx_j + 1]
                                trimmed[idx_j + 1] = alt_r
                                new_ng, _ = _calc_j_gap(idx_j)
                                if new_ng < net_gap:
                                    net_gap = new_ng
                                else:
                                    trimmed[idx_j + 1] = old_r
                elif net_gap < -4:
                    c_smin_l, _, c_emin_l, c_emax_l = chain_bounds[idx_j]
                    if c_emin_l < c_emax_l and c_smin_l < c_emin_l:
                        alt_l = _eval_seg_slice(
                            chain_seqs[idx_j], c_smin_l, c_emin_l
                        )
                        if alt_l is not None:
                            old_l = trimmed[idx_j]
                            trimmed[idx_j] = alt_l
                            new_ng, _ = _calc_j_gap(idx_j)
                            if new_ng > net_gap:
                                net_gap = new_ng
                            else:
                                trimmed[idx_j] = old_l
                    if net_gap < -4:
                        c_smin_r, c_smax_r, _, c_emax_r = chain_bounds[
                            idx_j + 1
                        ]
                        if c_smax_r > c_smin_r and c_smax_r < c_emax_r:
                            alt_r = _eval_seg_slice(
                                chain_seqs[idx_j + 1],
                                c_smax_r,
                                c_emax_r,
                                prefer_right=True,
                            )
                            if alt_r is not None:
                                old_r = trimmed[idx_j + 1]
                                trimmed[idx_j + 1] = alt_r
                                new_ng, _ = _calc_j_gap(idx_j)
                                if new_ng > net_gap:
                                    net_gap = new_ng
                                else:
                                    trimmed[idx_j + 1] = old_r

            total_net_del = 0
            for idx_j in range(n_j):
                net_gap, max_j_gap = _calc_j_gap(idx_j)
                if net_gap < -4 or net_gap > max_j_gap:
                    return False
                if net_gap > 0:
                    total_net_del += net_gap
                    if total_net_del >= min_sv_len:
                        return False

        total_aligned_q = sum(item[1] - item[0] for item in trimmed)
        if total_aligned_q < max(min_aligned_core, segs_q_total // 2):
            return False

        rc_c_s = trimmed[0][2]
        rc_c_e = trimmed[-1][3]
        is_distinct = _edlib_rc_interval_distinct(
            t_start, t_end, rc_c_s, rc_c_e, min_sv_len
        )
        # Re-check distinctness on trimmed outer bounds (blocks C2-r3 B1 flank FPs);
        # for is_rc=True, allow shifted non-distinct spans to proceed to tie check.
        if not is_distinct and (
            not is_rc or (rc_c_s == t_start and rc_c_e == t_end)
        ):
            return False

        if is_distinct:
            return True

        # Shifted same-locus RC orientation tie check (R1-r5 Major F1 Part 1)
        return _is_edlib_same_locus_rc_tie(
            rc_q_core=rc_q_core,
            target=target,
            rc_s=rc_c_s,
            rc_e=rc_c_e,
            m_total=m_total,
            fwd_core_score=fwd_core_score,
            min_aligned_core=min_aligned_core,
            trimmed_right=trimmed_right,
            trimmed_left=trimmed_left,
            bleed_k_pad=bleed_k_pad,
        )

    seg_cl_cache: dict[
        tuple[str, int, int, int], list[tuple[int, int, int, int]]
    ] = {}

    def _find_seg_clusters(
        seg_seq: str, seg_k: int, s_lo: int, s_hi: int, max_cl: int
    ) -> list[tuple[int, int, int, int]]:
        fck = (seg_seq, s_lo, s_hi, max_cl)
        cached_cl = seg_cl_cache.get(fck)
        if cached_cl is not None:
            return cached_cl
        seg_len = len(seg_seq)
        if s_hi - s_lo >= seg_len:
            first_pos = target.find(seg_seq, s_lo, s_hi)
            if first_pos != -1:
                clusters_ex: list[tuple[int, int, int, int]] = []
                anc_s = max_s = first_pos
                pos = target.find(seg_seq, first_pos + 1, s_hi)
                while pos != -1:
                    if pos - anc_s <= 8:
                        max_s = pos
                    else:
                        clusters_ex.append(
                            (anc_s, max_s, anc_s + seg_len, max_s + seg_len)
                        )
                        if len(clusters_ex) >= max_cl:
                            seg_cl_cache[fck] = clusters_ex
                            return clusters_ex
                        anc_s = max_s = pos
                    pos = target.find(seg_seq, pos + 1, s_hi)
                clusters_ex.append(
                    (anc_s, max_s, anc_s + seg_len, max_s + seg_len)
                )
                seg_cl_cache[fck] = clusters_ex
                return clusters_ex
        res_fc = edlib.align(
            seg_seq, target[s_lo:s_hi], mode="HW", task="locations", k=seg_k
        )
        if not (0 <= res_fc.get("editDistance", -1) <= seg_k):
            seg_cl_cache[fck] = []
            return []
        out_cl = _cluster_edlib_raw_locations_bounds(
            res_fc.get("locations") or [], offset=s_lo
        )[:max_cl]
        seg_cl_cache[fck] = out_cl
        return out_cl

    def _extend_fwd(
        segs: Sequence[tuple[str, int, int, int]],
        seg_gaps: Sequence[int],
        idx: int,
        chain_seqs: list[str],
        chain_bounds: list[tuple[int, int, int, int]],
        acc_del: int,
        allow_subsplit: bool,
        is_rc: bool,
        split_tag: int,
        memo: dict[tuple[int, int, int, int, bool], int],
        budget: list[int],
        val_budget: list[int],
    ) -> tuple[bool, bool]:
        if idx == len(segs):
            if val_budget[0] <= 0 or budget[0] <= 0:
                return False, True
            val_budget[0] -= 1
            budget[0] -= 4
            return _validate_chain(chain_seqs, chain_bounds, seg_gaps, is_rc), True
        _, _, prev_emin, prev_emax = chain_bounds[-1]
        m_key = (split_tag, idx, prev_emin, prev_emax, allow_subsplit)
        prev_fail = memo.get(m_key)
        if prev_fail is not None and acc_del >= prev_fail:
            return False, False
        if budget[0] <= 0 or val_budget[0] <= 0:
            return False, True

        seg_seq, _, _, seg_k = segs[idx]
        seg_len = len(seg_seq)
        q_ins_gap = seg_gaps[idx - 1]
        prev_k = _seg_max_ed(len(chain_seqs[-1]))
        b_slack = prev_k + seg_k
        rem_del = (min_sv_len - 1) - acc_del
        rem_j_del = min(4, rem_del) if q_ins_gap > 0 else rem_del
        max_gap_lo = (4 if q_ins_gap > 0 else (min_sv_len - 1)) + b_slack
        win_s = max(0, prev_emin - 4)
        win_e = min(len_r, prev_emax + rem_j_del + b_slack + seg_len + seg_k + 4)
        reached_val = False
        if win_e - win_s >= max(1, seg_len - seg_k):
            budget[0] -= 1
            next_clusters = _find_seg_clusters(
                seg_seq, seg_k, win_s, win_e, max_local_clusters
            )
            if next_clusters:
                for nc in next_clusters:
                    ns_min, ns_max, _, _ = nc
                    if ns_max < prev_emin - 4:
                        continue
                    gap_lo = ns_min - prev_emax
                    if gap_lo > max_gap_lo:
                        continue
                    eff_gap = gap_lo - b_slack
                    new_acc = acc_del + (eff_gap if eff_gap > 0 else 0)
                    if new_acc >= min_sv_len:
                        continue
                    chain_seqs.append(seg_seq)
                    chain_bounds.append(nc)
                    ok, rv = _extend_fwd(
                        segs,
                        seg_gaps,
                        idx + 1,
                        chain_seqs,
                        chain_bounds,
                        new_acc,
                        allow_subsplit,
                        is_rc,
                        split_tag,
                        memo,
                        budget,
                        val_budget,
                    )
                    chain_seqs.pop()
                    chain_bounds.pop()
                    if ok:
                        return True, True
                    if rv:
                        reached_val = True
                    if budget[0] <= 0 or val_budget[0] <= 0:
                        return False, True

        if (
            allow_subsplit
            and len(chain_seqs) + (len(segs) - idx) < max_segs
            and seg_len >= 2 * min_aligned_core
            and rem_del >= 4
            and budget[0] > 0
            and val_budget[0] > 0
        ):
            mid = seg_len // 2
            left_sub = seg_seq[:mid]
            right_sub = seg_seq[mid:]
            sub_segs = (
                list(segs[:idx])
                + [
                    (left_sub, 0, 0, _seg_max_ed(len(left_sub))),
                    (right_sub, 0, 0, _seg_max_ed(len(right_sub))),
                ]
                + list(segs[idx + 1 :])
            )
            sub_gaps = list(seg_gaps[:idx]) + [0] + list(seg_gaps[idx:])
            ok, rv = _extend_fwd(
                sub_segs,
                sub_gaps,
                idx,
                chain_seqs,
                chain_bounds,
                acc_del,
                False,
                is_rc,
                idx,
                memo,
                budget,
                val_budget,
            )
            if ok:
                return True, True
            if rv:
                reached_val = True

        if not reached_val and budget[0] > 0 and val_budget[0] > 0:
            if prev_fail is None or acc_del < prev_fail:
                memo[m_key] = acc_del
        return False, reached_val

    def _extend_bwd(
        segs: Sequence[tuple[str, int, int, int]],
        seg_gaps: Sequence[int],
        idx: int,
        rev_seqs: list[str],
        rev_bounds: list[tuple[int, int, int, int]],
        acc_del: int,
        allow_subsplit: bool,
        is_rc: bool,
        split_tag: int,
        memo: dict[tuple[int, int, int, int, bool], int],
        budget: list[int],
        val_budget: list[int],
    ) -> tuple[bool, bool]:
        if idx < 0:
            if val_budget[0] <= 0 or budget[0] <= 0:
                return False, True
            val_budget[0] -= 1
            budget[0] -= 4
            return (
                _validate_chain(
                    list(reversed(rev_seqs)),
                    list(reversed(rev_bounds)),
                    seg_gaps,
                    is_rc,
                ),
                True,
            )
        next_smin, next_smax, _, _ = rev_bounds[-1]
        m_key = (split_tag, idx, next_smin, next_smax, allow_subsplit)
        prev_fail = memo.get(m_key)
        if prev_fail is not None and acc_del >= prev_fail:
            return False, False
        if budget[0] <= 0 or val_budget[0] <= 0:
            return False, True

        seg_seq, _, _, seg_k = segs[idx]
        seg_len = len(seg_seq)
        q_ins_gap = seg_gaps[idx]
        next_k = _seg_max_ed(len(rev_seqs[-1]))
        b_slack = next_k + seg_k
        rem_del = (min_sv_len - 1) - acc_del
        rem_j_del = min(4, rem_del) if q_ins_gap > 0 else rem_del
        max_gap_lo = (4 if q_ins_gap > 0 else (min_sv_len - 1)) + b_slack
        win_s = max(0, next_smin - rem_j_del - b_slack - seg_len - seg_k - 4)
        win_e = min(len_r, next_smax + 4)
        reached_val = False
        if win_e - win_s >= max(1, seg_len - seg_k):
            budget[0] -= 1
            prev_clusters = _find_seg_clusters(
                seg_seq, seg_k, win_s, win_e, max_local_clusters
            )
            if prev_clusters:
                for pc in prev_clusters:
                    _, _, pe_min, pe_max = pc
                    if pe_min > next_smax + 4:
                        continue
                    gap_lo = next_smin - pe_max
                    if gap_lo > max_gap_lo:
                        continue
                    eff_gap = gap_lo - b_slack
                    new_acc = acc_del + (eff_gap if eff_gap > 0 else 0)
                    if new_acc >= min_sv_len:
                        continue
                    rev_seqs.append(seg_seq)
                    rev_bounds.append(pc)
                    ok, rv = _extend_bwd(
                        segs,
                        seg_gaps,
                        idx - 1,
                        rev_seqs,
                        rev_bounds,
                        new_acc,
                        allow_subsplit,
                        is_rc,
                        split_tag,
                        memo,
                        budget,
                        val_budget,
                    )
                    rev_seqs.pop()
                    rev_bounds.pop()
                    if ok:
                        return True, True
                    if rv:
                        reached_val = True
                    if budget[0] <= 0 or val_budget[0] <= 0:
                        return False, True

        if (
            allow_subsplit
            and len(rev_seqs) + (idx + 1) < max_segs
            and seg_len >= 2 * min_aligned_core
            and rem_del >= 4
            and budget[0] > 0
            and val_budget[0] > 0
        ):
            mid = seg_len // 2
            left_sub = seg_seq[:mid]
            right_sub = seg_seq[mid:]
            sub_segs = (
                list(segs[:idx])
                + [
                    (left_sub, 0, 0, _seg_max_ed(len(left_sub))),
                    (right_sub, 0, 0, _seg_max_ed(len(right_sub))),
                ]
                + list(segs[idx + 1 :])
            )
            sub_gaps = list(seg_gaps[:idx]) + [0] + list(seg_gaps[idx:])
            ok, rv = _extend_bwd(
                sub_segs,
                sub_gaps,
                idx + 1,
                rev_seqs,
                rev_bounds,
                acc_del,
                False,
                is_rc,
                idx,
                memo,
                budget,
                val_budget,
            )
            if ok:
                return True, True
            if rv:
                reached_val = True

        if not reached_val and budget[0] > 0 and val_budget[0] > 0:
            if prev_fail is None or acc_del < prev_fail:
                memo[m_key] = acc_del
        return False, reached_val

    def _seg_exists_in_slice(
        seg_seq: str,
        seg_k: int,
        s_lo: int,
        s_hi: int,
        can_subsplit: bool,
    ) -> bool:
        sl_len = s_hi - s_lo
        if sl_len >= max(1, len(seg_seq) - seg_k):
            if sl_len >= len(seg_seq) and target.find(seg_seq, s_lo, s_hi) != -1:
                return True
            ed_val = edlib.align(
                seg_seq, target[s_lo:s_hi], mode="HW", task="distance", k=seg_k
            ).get("editDistance", -1)
            if 0 <= ed_val <= seg_k:
                return True
        if can_subsplit and len(seg_seq) >= 2 * min_aligned_core:
            mid = len(seg_seq) // 2
            l_sub = seg_seq[:mid]
            r_sub = seg_seq[mid:]
            lk = _seg_max_ed(len(l_sub))
            rk = _seg_max_ed(len(r_sub))
            if sl_len >= max(1, len(l_sub) - lk) and sl_len >= max(
                1, len(r_sub) - rk
            ):
                ed_l = edlib.align(
                    l_sub, target[s_lo:s_hi], mode="HW", task="distance", k=lk
                ).get("editDistance", -1)
                if 0 <= ed_l <= lk:
                    ed_r = edlib.align(
                        r_sub,
                        target[s_lo:s_hi],
                        mode="HW",
                        task="distance",
                        k=rk,
                    ).get("editDistance", -1)
                    if 0 <= ed_r <= rk:
                        return True
        return False

    low_comp_cache: dict[str, bool] = {}

    def _is_low_complexity_seg(seq: str) -> bool:
        cached_lc = low_comp_cache.get(seq)
        if cached_lc is not None:
            return cached_lc
        res_lc = (
            len(seq) >= 6
            and len(dict.fromkeys(seq[i : i + 3] for i in range(len(seq) - 2)))
            <= 4
        )
        low_comp_cache[seq] = res_lc
        return res_lc

    def _probe_anchor_slice(
        segs: Sequence[tuple[str, int, int, int]],
        seg_gaps: Sequence[int],
        from_left: bool,
        sl_s: int,
        sl_e: int,
        is_rc: bool,
    ) -> tuple[bool, list[tuple[int, int, int, int]]]:
        anc_idx = 0 if from_left else len(segs) - 1
        anc_seq, _, _, anc_k = segs[anc_idx]
        # Never perform a target-wide anchor search on a < min_aligned_core arm.
        if len(anc_seq) < min_aligned_core:
            return False, []
        min_anc_len = max(1, len(anc_seq) - anc_k)
        if sl_e - sl_s < min_anc_len:
            return False, []

        # R2-r5 F1 Defense 1a: Before running expensive `task="locations"` on a
        # low-complexity anchor arm (<= 4 distinct 3-mers, e.g. AT*12 / A*24),
        # check via O(1) `task="distance"` that every non-anchor segment exists
        # in the reachable portion of target (100% zero-FN).
        can_subsplit = len(segs) < max_segs
        if len(segs) >= 3 and _is_low_complexity_seg(anc_seq):
            if from_left:
                pre_lb = min(len_r, sl_s + max(0, min_anc_len - 4))
                pre_ub = len_r
                pre_indices = list(range(len(segs) - 1, 0, -1))
            else:
                pre_lb = 0
                pre_ub = max(0, sl_e - max(0, min_anc_len - 4))
                pre_indices = list(range(0, len(segs) - 1))
            for m_idx in pre_indices:
                m_seq, _, _, m_k = segs[m_idx]
                if not _seg_exists_in_slice(
                    m_seq, m_k, pre_lb, pre_ub, can_subsplit
                ):
                    return False, []

        clusters = _find_seg_clusters(
            anc_seq, anc_k, sl_s, sl_e, max_anchor_clusters
        )
        if not clusters:
            return False, []

        # R2-r5 F1 Defense 1b: Zero-FN chunked reachability pre-filter for
        # repetitive 3-4 seg probes. Partitions clusters relative to Copy 1
        # (preserving near_copy1 for is_rc=True shifted palindrome ties) and
        # prunes any chunk whose reachable target window lacks a required
        # non-anchor segment (including sub-split halves when len(segs) < max_segs).
        active_clusters = clusters
        if len(segs) >= 3 and len(clusters) >= 8:
            rem_segs = segs[1:] if from_left else segs[:-1]
            check_indices = (
                list(range(len(segs) - 1, 0, -1))
                if from_left
                else list(range(0, len(segs) - 1))
            )
            max_rem_span = (
                (min_sv_len - 1)
                + sum(4 for g in seg_gaps if g > 0)
                + sum(len(s) + sk + 8 for s, _, _, sk in rem_segs)
                + anc_k
                + 8
            )
            margin = (min_sv_len - 1) if is_rc else 0
            far_left = [cl for cl in clusters if cl[3] <= t_start - margin]
            far_right = [cl for cl in clusters if cl[0] >= t_end + margin]
            near_c1 = [
                cl
                for cl in clusters
                if cl[3] > t_start - margin and cl[0] < t_end + margin
            ]
            kept: list[tuple[int, int, int, int]] = []
            chunk_size = 8
            for part in (far_left, near_c1, far_right):
                if not part:
                    continue
                for c_off in range(0, len(part), chunk_size):
                    chunk = part[c_off : c_off + chunk_size]
                    if from_left:
                        ch_lb = max(0, min(cl[2] for cl in chunk) - 4)
                        ch_ub = min(
                            len_r, max(cl[3] for cl in chunk) + max_rem_span
                        )
                    else:
                        ch_lb = max(
                            0, min(cl[0] for cl in chunk) - max_rem_span
                        )
                        ch_ub = min(len_r, max(cl[1] for cl in chunk) + 4)
                    chunk_ok = True
                    for m_idx in check_indices:
                        m_seq, _, _, m_k = segs[m_idx]
                        if not _seg_exists_in_slice(
                            m_seq, m_k, ch_lb, ch_ub, can_subsplit
                        ):
                            chunk_ok = False
                            break
                    if chunk_ok:
                        kept.extend(chunk)
            active_clusters = kept
            if not active_clusters:
                return False, clusters

        # R2-r5 F1 Defenses 2 & 3 + R2-r6 Minor F3: Strictly per-probe
        # monotonic failed-node cache in reverse topological order +
        # deterministic step budget & per-cluster fair validation cap.
        memo: dict[tuple[int, int, int, int, bool], int] = {}
        budget = [
            min(64, max_probe_steps)
            if _is_low_complexity_seg(anc_seq)
            else max_probe_steps
        ]
        val_budget = [
            min(12, max_validate_calls)
            if _is_low_complexity_seg(anc_seq)
            else max_validate_calls
        ]
        iter_clusters = (
            reversed(active_clusters) if from_left else active_clusters
        )
        for cl in iter_clusters:
            if budget[0] <= 0 or val_budget[0] <= 0:
                break
            cl_cap = min(4, val_budget[0])
            cl_val = [cl_cap]
            if from_left:
                ok, _ = _extend_fwd(
                    segs,
                    seg_gaps,
                    1,
                    [anc_seq],
                    [cl],
                    0,
                    True,
                    is_rc,
                    -1,
                    memo,
                    budget,
                    cl_val,
                )
            else:
                ok, _ = _extend_bwd(
                    segs,
                    seg_gaps,
                    len(segs) - 2,
                    [anc_seq],
                    [cl],
                    0,
                    True,
                    is_rc,
                    -1,
                    memo,
                    budget,
                    cl_val,
                )
            val_budget[0] -= cl_cap - cl_val[0]
            if ok:
                return True, clusters
        return False, clusters

    # 1. Forward strand (+): self-excluding shifted slices for both right and
    # left copies, with ov_hits shifted-slice fallback (R2-r5 F3) mirroring RC
    # so repeated-arm queries (e.g. U1+D1+U1) cannot self-mask mutated Copy 2s.
    f0_seq, f0_ts, f0_te, f0_k = fwd_segs[0]
    fl_seq, fl_ts, fl_te, fl_k = fwd_segs[-1]
    shift_f0 = f0_k + 1 + max(0, (f0_te - f0_ts) - len(f0_seq))
    shift_fl = fl_k + 1 + max(0, (fl_te - fl_ts) - len(fl_seq))
    fwd_can_subsplit = len(fwd_segs) < max_segs
    f0_low = _is_low_complexity_seg(f0_seq)
    fl_low = _is_low_complexity_seg(fl_seq)

    def _probe_fwd_with_ov_fallback(
        from_left: bool,
        dir_right: bool,
        init_cut: int,
        sl_bound: int | None = None,
    ) -> tuple[bool, bool]:
        anc_idx = 0 if from_left else len(fwd_segs) - 1
        anc_seq, _, _, anc_k = fwd_segs[anc_idx]
        cur_cut = init_cut
        had_any_hit = False
        low_c = _is_low_complexity_seg(anc_seq)
        for _ in range(len(fwd_segs)):
            if dir_right:
                sl_s = cur_cut
                sl_e = len_r if sl_bound is None else min(len_r, sl_bound)
            else:
                sl_s = 0 if sl_bound is None else max(0, sl_bound)
                sl_e = cur_cut
            found, clusters = _probe_anchor_slice(
                fwd_segs, fwd_gaps, from_left, sl_s, sl_e, False
            )
            if found:
                return True, True
            if clusters:
                had_any_hit = True
            ov_hits = [
                cl
                for cl in clusters
                if min(t_end, cl[3]) > max(t_start, cl[0])
            ]
            if not ov_hits:
                break
            ov_max_span = max(cl[3] - cl[0] for cl in ov_hits)
            shift_ov = anc_k + 1 + max(0, ov_max_span - len(anc_seq))
            if dir_right:
                next_cut = min(
                    max(cl[3] for cl in ov_hits),
                    max(cl[0] for cl in ov_hits) + shift_ov,
                )
                if next_cut <= cur_cut:
                    break
                # If clusters already include non-overlapping hits past next_cut
                # at the same minimum editDistance, shifting past ov_hits cannot
                # unmask a worse hit outside [t_start, t_end).
                if (low_c or len(clusters) < max_anchor_clusters) and any(
                    cl[0] >= next_cut for cl in clusters
                ):
                    break
            else:
                next_cut = max(
                    min(cl[0] for cl in ov_hits),
                    min(cl[3] for cl in ov_hits) - shift_ov,
                )
                if next_cut >= cur_cut:
                    break
                if (low_c or len(clusters) < max_anchor_clusters) and any(
                    cl[3] <= next_cut for cl in clusters
                ):
                    break
            cur_cut = next_cut
        return False, had_any_hit

    r_cut_0 = min(f0_te, f0_ts + shift_f0)
    r_cut_l = min(fl_te, fl_ts + shift_fl)
    l_cut_0 = max(f0_ts, f0_te - shift_f0)
    l_cut_l = max(fl_ts, fl_te - shift_fl)

    # Probe the higher-complexity outer arm first on both right and left sides:
    # if it has zero hits in its shifted slice, tighten the opposite arm's slice
    # bound (100% zero-FN) so low-complexity arms skip repetitive tails.
    if fl_low and not f0_low:
        found_r0, hit_r0 = _probe_fwd_with_ov_fallback(True, True, r_cut_0)
        if found_r0:
            return True
        rl_bound: int | None = None
        if (
            len(f0_seq) >= min_aligned_core
            and not hit_r0
            and (
                not fwd_can_subsplit
                or not _seg_exists_in_slice(f0_seq, f0_k, r_cut_0, len_r, True)
            )
        ):
            max_post_span = (
                (min_sv_len - 1)
                + sum(4 for g in fwd_gaps if g > 0)
                + sum(len(s) + 3 * sk + 24 for s, _, _, sk in fwd_segs[1:])
                + f0_k
                + 24
            )
            rl_bound = min(len_r, r_cut_0 + len(f0_seq) + f0_k + max_post_span)
        if _probe_fwd_with_ov_fallback(
            False, True, r_cut_l, sl_bound=rl_bound
        )[0]:
            return True
    else:
        found_rl, hit_rl = _probe_fwd_with_ov_fallback(False, True, r_cut_l)
        if found_rl:
            return True
        r0_bound: int | None = None
        if (
            len(fl_seq) >= min_aligned_core
            and not hit_rl
            and (
                not fwd_can_subsplit
                or not _seg_exists_in_slice(fl_seq, fl_k, r_cut_l, len_r, True)
            )
        ):
            r0_bound = min(len_r, r_cut_l + len(f0_seq) + f0_k)
        if _probe_fwd_with_ov_fallback(
            True, True, r_cut_0, sl_bound=r0_bound
        )[0]:
            return True

    if f0_low and not fl_low:
        found_ll, hit_ll = _probe_fwd_with_ov_fallback(False, False, l_cut_l)
        if found_ll:
            return True
        l0_bound: int | None = None
        if (
            len(fl_seq) >= min_aligned_core
            and not hit_ll
            and (
                not fwd_can_subsplit
                or not _seg_exists_in_slice(fl_seq, fl_k, 0, l_cut_l, True)
            )
        ):
            max_pre_span = (
                (min_sv_len - 1)
                + sum(4 for g in fwd_gaps if g > 0)
                + sum(len(s) + 3 * sk + 24 for s, _, _, sk in fwd_segs[:-1])
                + fl_k
                + 24
            )
            l0_bound = max(0, l_cut_l - len(fl_seq) - fl_k - max_pre_span)
        if _probe_fwd_with_ov_fallback(
            True, False, l_cut_0, sl_bound=l0_bound
        )[0]:
            return True
    else:
        found_l0, hit_l0 = _probe_fwd_with_ov_fallback(True, False, l_cut_0)
        if found_l0:
            return True
        ll_bound: int | None = None
        if (
            len(f0_seq) >= min_aligned_core
            and not hit_l0
            and (
                not fwd_can_subsplit
                or not _seg_exists_in_slice(f0_seq, f0_k, 0, l_cut_0, True)
            )
        ):
            ll_bound = max(0, l_cut_0 - len(fl_seq) - fl_k)
        if _probe_fwd_with_ov_fallback(
            False, False, l_cut_l, sl_bound=ll_bound
        )[0]:
            return True

    # 2. Reverse-complement strand (-): full target first; if any anchor hit
    # overlapped [t_start, t_end) (e.g. internal palindrome masking a slightly
    # mutated distinct RC copy), also probe shifted slices outside those hits.
    rc_can_subsplit = len(rc_segs) < max_segs
    for from_left in (True, False):
        found_rc, rc_clusters = _probe_anchor_slice(
            rc_segs, rc_gaps, from_left, 0, len_r, True
        )
        if found_rc:
            return True
        if (
            from_left
            and not rc_clusters
            and len(rc_segs[0][0]) >= min_aligned_core
            and (
                not rc_can_subsplit
                or not _seg_exists_in_slice(
                    rc_segs[0][0], rc_segs[0][3], 0, len_r, True
                )
            )
        ):
            break
        ov_hits = [
            cl
            for cl in rc_clusters
            if min(t_end, cl[3]) > max(t_start, cl[0])
        ]
        if ov_hits:
            anc_idx = 0 if from_left else len(rc_segs) - 1
            anc_seq, _, _, anc_k = rc_segs[anc_idx]
            low_c_rc = _is_low_complexity_seg(anc_seq)
            ov_max_span = max(cl[3] - cl[0] for cl in ov_hits)
            shift_rc = anc_k + 1 + max(0, ov_max_span - len(anc_seq))
            rc_r_cut = min(
                max(cl[3] for cl in ov_hits),
                max(cl[0] for cl in ov_hits) + shift_rc,
            )
            if not (
                (low_c_rc or len(rc_clusters) < max_anchor_clusters)
                and any(cl[0] >= rc_r_cut for cl in rc_clusters)
            ):
                if _probe_anchor_slice(
                    rc_segs, rc_gaps, from_left, rc_r_cut, len_r, True
                )[0]:
                    return True
            rc_l_cut = max(
                min(cl[0] for cl in ov_hits),
                min(cl[3] for cl in ov_hits) - shift_rc,
            )
            if not (
                (low_c_rc or len(rc_clusters) < max_anchor_clusters)
                and any(cl[3] <= rc_l_cut for cl in rc_clusters)
            ):
                if _probe_anchor_slice(
                    rc_segs, rc_gaps, from_left, 0, rc_l_cut, True
                )[0]:
                    return True

    return False


def _check_edlib_criterion1_unique(
    s: str,
    r: str,
    min_sv_len: int,
    min_aligned_core: int,
    ops: Sequence[tuple[str, int]],
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    m_total: int,
    core_ed: int,
    raw_fwd_ed: int,
    raw_locations: Sequence[tuple[int, int]],
    span_extended: bool,
    span_trimmed: bool,
) -> bool:
    """
    Stage 6 Unified Criterion 1 for the `<= 1 kb` `edlib` branch.
    Consumes the Stage 5 canonical aligned-core contract and returns True iff
    `[t_start, t_end)` is a unique, unambiguous forward-strand match.
    """
    len_s = len(s)
    q_span = q_end - q_start
    t_span = t_end - t_start
    q_core = s[q_start:q_end]
    t_core = r[t_start:t_end]

    # Exact palindrome fast-path (O(q_span)): identical + and - alignments on [t_start, t_end)
    rc_q_core = revcomp(q_core)
    if q_core == rc_q_core:
        return False

    fwd_core_score = _asm20_ops_score(ops)

    # 6A. Criterion 1a: Tied optimal forward hits from Stage 1 raw_locations[1:]
    if not span_extended and len(raw_locations) > 1:
        need_trim_1a = span_trimmed or raw_fwd_ed > int(0.20 * len_s)
        for loc_s, loc_e in _cluster_edlib_raw_locations(
            raw_locations[1:], offset=0
        ):
            if not _edlib_intervals_distinct(
                t_start, t_end, loc_s, loc_e, min_sv_len
            ):
                continue
            if need_trim_1a:
                valid, _, _, _, h_ts, h_te, _ = _trim_edlib_hit_for_c1(
                    s, r, loc_s, loc_e, min_aligned_core
                )
                if valid and _edlib_intervals_distinct(
                    t_start, t_end, h_ts, h_te, min_sv_len
                ):
                    return False
            else:
                return False

    # 6B. Criterion 1b: Shifted / disjoint secondary forward hits on t_core (and q_core if distinct)
    t_core_max_ed = _edlib_core_max_ed(t_span)
    if _has_distinct_shifted_edlib_hit(
        t_core,
        r,
        t_start,
        t_end,
        t_core_max_ed,
        min_sv_len,
        max(1, t_span - t_core_max_ed),
    ):
        return False

    if q_core != t_core:
        q_core_max_ed = _edlib_core_max_ed(q_span)
        if _has_distinct_shifted_edlib_hit(
            q_core,
            r,
            t_start,
            t_end,
            q_core_max_ed,
            min_sv_len,
            max(1, q_span - q_core_max_ed),
        ):
            return False
    else:
        q_core_max_ed = t_core_max_ed

    # 6C. Criterion 1c: Unified RC distinct-copy & same-locus orientation-tie check
    trimmed_right = q_end < len_s
    trimmed_left = q_start > 0
    bleed_k_pad = (
        (_EDLIB_RC_BLEED_TOL if trimmed_right else 0)
        + (_EDLIB_RC_BLEED_TOL if trimmed_left else 0)
    )
    rc_q_tie_k = min(
        core_ed + bleed_k_pad,
        int(0.30 * q_span)
        if q_span >= _EDLIB_MIN_ALIGNED_CORE
        else q_core_max_ed,
    )
    rc_q_search_k = max(q_core_max_ed, rc_q_tie_k)

    if _has_disqualifying_edlib_rc_hit(
        rc_query=rc_q_core,
        target=r,
        t_start=t_start,
        t_end=t_end,
        m_total=m_total,
        fwd_core_score=fwd_core_score,
        rc_max_ed_distinct=q_core_max_ed,
        rc_search_k=rc_q_search_k,
        min_sv_len=min_sv_len,
        min_aligned_core=min_aligned_core,
        trimmed_right=trimmed_right,
        trimmed_left=trimmed_left,
        check_same_locus_tie=True,
    ):
        return False

    # Direct same-locus [t_start, t_end) tie check: global HW in
    # _has_disqualifying_edlib_rc_hit can abandon a short arm across an internal
    # indel (e.g. symmetric palindrome + 35-49D with 1 SNP or overhang bleed)
    # even when core_ed + bleed_k_pad <= rc_q_search_k, or miss when above 30%.
    # Pre-filter with provably zero-FN banded NW distance k_sl in C edlib, and
    # handle short <= 80 bp palindromes (where fwd_core_score <= 0).
    k_sl = (q_span - m_total) + (t_span - m_total) + 2 * bleed_k_pad
    if k_sl > 0 and _is_edlib_same_locus_rc_tie(
        rc_q_core=rc_q_core,
        target=r,
        rc_s=t_start,
        rc_e=t_end,
        m_total=m_total,
        fwd_core_score=fwd_core_score,
        min_aligned_core=min_aligned_core,
        trimmed_right=trimmed_right,
        trimmed_left=trimmed_left,
        bleed_k_pad=bleed_k_pad,
    ):
        return False

    # Shifted (< min_sv_len) local-HW same-locus tie check (R1-r5 Major F1 Part 2):
    # catches internal 'I' ties when core_ed + bleed_k_pad > rc_q_search_k (30% cap)
    # and short <= 0-score shifted palindrome ties skipped by _trim_edlib_hit_for_c1.
    if core_ed + bleed_k_pad > rc_q_search_k or fwd_core_score <= bleed_k_pad:
        k_local_hw = core_ed + bleed_k_pad
        if k_local_hw >= 0 and min_sv_len > 1:
            len_r = len(r)
            win_s = max(0, t_start - (min_sv_len - 1))
            win_e = min(len_r, t_end + (min_sv_len - 1))
            hw_sl = edlib.align(
                rc_q_core,
                r[win_s:win_e],
                mode="HW",
                task="locations",
                k=k_local_hw,
            )
            if 0 <= hw_sl.get("editDistance", -1) <= k_local_hw:
                for ls, le in _cluster_edlib_raw_locations(
                    hw_sl.get("locations") or [], offset=win_s
                )[:16]:
                    if ls == t_start and le == t_end:
                        continue
                    if _is_edlib_same_locus_rc_tie(
                        rc_q_core=rc_q_core,
                        target=r,
                        rc_s=ls,
                        rc_e=le,
                        m_total=m_total,
                        fwd_core_score=fwd_core_score,
                        min_aligned_core=min_aligned_core,
                        trimmed_right=trimmed_right,
                        trimmed_left=trimmed_left,
                        bleed_k_pad=bleed_k_pad,
                    ):
                        return False

    if t_core != q_core:
        if _has_disqualifying_edlib_rc_hit(
            rc_query=revcomp(t_core),
            target=r,
            t_start=t_start,
            t_end=t_end,
            m_total=m_total,
            fwd_core_score=fwd_core_score,
            rc_max_ed_distinct=t_core_max_ed,
            rc_search_k=t_core_max_ed,
            min_sv_len=min_sv_len,
            min_aligned_core=min_aligned_core,
            trimmed_right=False,
            trimmed_left=False,
            check_same_locus_tie=False,
        ):
            return False

    # 6D. Criterion 1 multi-arm internal-indel & shifted-palindrome check
    # (C1-r4 NEW Bug 3, R1-r5 Major F1 Part 1, C1-r5 NEW Bug 4):
    if _has_distinct_edlib_del_split_copy(
        q_core=q_core,
        target=r,
        ops=ops,
        t_start=t_start,
        t_end=t_end,
        min_sv_len=min_sv_len,
        min_aligned_core=min_aligned_core,
        rc_q_core=rc_q_core,
        m_total=m_total,
        fwd_core_score=fwd_core_score,
        trimmed_right=trimmed_right,
        trimmed_left=trimmed_left,
        bleed_k_pad=bleed_k_pad,
    ):
        return False

    # 6E. Exact/unsplit Copy 1 vs < min_sv_len-indel Copy 2 (C1-r5 NEW Bug 3):
    if _has_distinct_edlib_indel_copy2(
        q_core=q_core,
        rc_q_core=rc_q_core,
        target=r,
        t_start=t_start,
        t_end=t_end,
        min_sv_len=min_sv_len,
        min_aligned_core=min_aligned_core,
    ):
        return False

    return True


def _check_similarity_merge_with_span(
    seq_s: str,
    seq_r: str,
    min_sv_len: int = 50,
    cluster_id: float = 0.95,
    cluster_win_id: float = 0.70,
    aligner: Any | None = None,
    del_intervals_out: list[tuple[int, int]] | None = None,
) -> tuple[bool, int, int, int]:
    """
    Evaluate the 4-criterion similarity clustering check (§3.6.3) and return
    `(merged, t_start, t_end, del_bp_internal)` on `seq_r` so callers (e.g. Case A
    `rep0` backbone genotyping) can reuse exact target alignment coordinates without
    running a second alignment. When `del_intervals_out` is provided and the merge
    succeeds, appends half-open target deletion intervals `(t_d_start, t_d_end)` on `seq_r`.
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

    del_ivs: list[tuple[int, int]] = []
    is_edlib_branch = len_s <= EDLIB_MAX_LEN and len_r <= EDLIB_MAX_LEN
    if is_edlib_branch:
        # Stage 0: Exact zero-FN Levenshtein upper bound
        max_x_bound = int(
            math.floor(max(0.0, 1.0 - cluster_id) * len_s + 1e-9)
        )
        max_possible_fwd_ed = max_x_bound + 2 * (min_sv_len - 1)

        # Stage 1: Initial semi-global alignment (edlib mode='HW')
        res = edlib.align(
            s, r, mode="HW", task="path", k=max_possible_fwd_ed
        )
        cigar = res.get("cigar")
        raw_fwd_ed = res.get("editDistance", -1)
        raw_locations = res.get("locations") or []
        if (
            cigar is None
            or raw_fwd_ed < 0
            or raw_fwd_ed > max_possible_fwd_ed
            or not raw_locations
        ):
            return False, 0, 0, 0

        t_start_raw, t_end_incl = raw_locations[0]
        t_end_raw = t_end_incl + 1
        if t_start_raw < 0 or t_end_raw > len_r or t_start_raw >= t_end_raw:
            return False, 0, 0, 0

        # Stage 2: Pass 1 lossless coalescing + Pass 2 affine DP (15 -> 8 -> 4 + skinny)
        ops = parse_edlib_cigar_to_ops(
            cigar, seq_s=s, seq_r=r[t_start_raw:t_end_raw]
        )

        # Stage 3: Tightened asm20 terminal overhang trim + outward exact-match re-extension
        ops, q_start, q_end, t_start, t_end = _trim_edlib_terminal_overhangs(
            ops, 0, len_s, t_start_raw, t_end_raw, seq_s=s, seq_r=r
        )

        # Stage 4: Bounded short-arm recovery + Criterion 1 split-alignment guard
        ops, q_start, q_end, t_start, t_end, arm_recovered, split_detected = (
            _recover_edlib_abandoned_arms(
                s, r, ops, q_start, q_end, t_start, t_end, min_sv_len
            )
        )
        if split_detected:
            return False, 0, 0, 0

        # Stage 5: Post-trim/post-recovery min aligned-core floor + fast Criteria 1d / 2 / 3
        min_aligned_core = min(_EDLIB_MIN_ALIGNED_CORE, len_s)
        if (
            not ops
            or (q_end - q_start) < min_aligned_core
            or (t_end - t_start) < min_aligned_core
        ):
            return False, 0, 0, 0

        del_bp_internal = 0
        m_total = 0
        x_total = 0
        ins_bp_s = 0
        gap_open_i = 0
        gap_open_d = 0
        t_pos = t_start
        for op, length in ops:
            if op == "M":
                m_total += length
                t_pos += length
            elif op == "X":
                x_total += length
                t_pos += length
            elif op == "I":
                ins_bp_s += length
                gap_open_i += 1
            elif op == "D":
                if length >= min_sv_len:
                    return False, 0, 0, 0
                del_bp_internal += length
                gap_open_d += 1
                if del_intervals_out is not None:
                    del_ivs.append((t_pos, t_pos + length))
                t_pos += length
        if del_bp_internal >= min_sv_len:
            return False, 0, 0, 0

        denom_gc = m_total + x_total + gap_open_i + gap_open_d
        if denom_gc <= 0 or (m_total / denom_gc) + 1e-12 < cluster_id:
            return False, 0, 0, 0

        unaligned_ends_s = q_start + (len_s - q_end)
        if unaligned_ends_s + ins_bp_s >= min_sv_len:
            return False, 0, 0, 0

        # Stage 6: Unified Criterion 1 (1a / 1b / 1c anchored to trimmed/recovered core)
        core_ed = x_total + ins_bp_s + del_bp_internal
        span_extended = (
            arm_recovered or (t_start < t_start_raw) or (t_end > t_end_raw)
        )
        span_trimmed = (
            (t_start != t_start_raw)
            or (t_end != t_end_raw)
            or (q_start > 0)
            or (q_end < len_s)
        )
        if not _check_edlib_criterion1_unique(
            s=s,
            r=r,
            min_sv_len=min_sv_len,
            min_aligned_core=min_aligned_core,
            ops=ops,
            q_start=q_start,
            q_end=q_end,
            t_start=t_start,
            t_end=t_end,
            m_total=m_total,
            core_ed=core_ed,
            raw_fwd_ed=raw_fwd_ed,
            raw_locations=raw_locations,
            span_extended=span_extended,
            span_trimmed=span_trimmed,
        ):
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
        t_pos = t_start
        for op, length in ops:
            if op == "M" or op == "X":
                t_pos += length
            elif op == "D":
                if length >= min_sv_len:
                    return False, 0, 0, 0
                del_bp_internal += length
                if del_intervals_out is not None:
                    del_ivs.append((t_pos, t_pos + length))
                t_pos += length
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

    if del_intervals_out is not None and del_ivs:
        del_intervals_out.extend(del_ivs)
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
