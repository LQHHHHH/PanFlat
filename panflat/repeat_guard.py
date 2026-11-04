"""Shared repeat-guard, partial-node slicing, mappy 50bp reference subtraction, and 4-criterion similarity clustering primitives (§3.6.3, §3.6.6, §3.8.4-B, §3.8.4-C)."""

from __future__ import annotations

import collections
import re
from typing import Any, Literal, Mapping, Sequence

import edlib
import mappy
import numpy as np

from panflat.gfa_io import revcomp
from panflat.models import InputContractError, Step, StepProvenance

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

    if seq_len <= REPEAT_WINDOW_SIZE:
        window_bounds: list[tuple[int, int]] = [(0, seq_len)]
    else:
        starts = sorted(
            set(range(0, seq_len - REPEAT_WINDOW_SIZE + 1, REPEAT_WINDOW_STEP))
            | {seq_len - REPEAT_WINDOW_SIZE}
        )
        window_bounds = [(s, s + REPEAT_WINDOW_SIZE) for s in starts]

    # Vectorized 30-bit canonical 15-mer encoding
    arr = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    codes = _BASE_LUT[arr]

    num_kmers = seq_len - REPEAT_KMER_SIZE + 1
    invalid_prefix = np.zeros(seq_len + 1, dtype=np.int32)
    np.cumsum(codes > 3, out=invalid_prefix[1:])
    valid_mask = (
        invalid_prefix[REPEAT_KMER_SIZE:] - invalid_prefix[:-REPEAT_KMER_SIZE]
    ) == 0

    safe = np.where(codes <= 3, codes, 0).astype(np.uint64)
    comp = np.uint64(3) - safe
    fwd = np.zeros(num_kmers, dtype=np.uint64)
    rev = np.zeros(num_kmers, dtype=np.uint64)
    for j in range(REPEAT_KMER_SIZE):
        fwd = (fwd << np.uint64(2)) | safe[j : j + num_kmers]
        rev = rev | (comp[j : j + num_kmers] << np.uint64(2 * j))
    canon = np.minimum(fwd, rev)

    window_diagnostics: list[dict[str, Any]] = []
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
            counts = collections.Counter(w_canon.tolist())
            unique_keys = len(counts)
            singleton_count = sum(1 for c in counts.values() if c == 1)
            singleton_ratio = singleton_count / valid_pos
            # Strictly < 0.30 -> "low", >= 0.30 -> "non-low"
            if singleton_count * 10 < valid_pos * 3:
                state = "low"
            else:
                state = "non-low"

        window_diagnostics.append(
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
        )

    # Mask interval construction
    if len(window_diagnostics) == 1:
        # Short-block exception (500 <= L <= 1000)
        if window_diagnostics[0]["state"] == "low":
            masked_intervals = [(0, seq_len)]
        else:
            masked_intervals = []
    else:
        raw_masked: list[tuple[int, int]] = []
        current_run: list[dict[str, Any]] = []
        for w_diag in window_diagnostics:
            if w_diag["state"] == "low":
                current_run.append(w_diag)
            else:
                if len(current_run) >= 2:
                    raw_masked.append((current_run[0]["start"], current_run[-1]["end"]))
                current_run = []
        if len(current_run) >= 2:
            raw_masked.append((current_run[0]["start"], current_run[-1]["end"]))
        masked_intervals = merge_intervals(raw_masked)

    return masked_intervals, window_diagnostics


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


def parse_edlib_cigar_to_ops(
    cigar_str: str,
    seq_s: str | None = None,
    seq_r: str | None = None,
) -> list[tuple[str, int]]:
    """
    Parse an `edlib` extended CIGAR string (`=`, `X`, `I`, `D`) into coalesced
    `('M'|'X'|'I'|'D', length)` operations along `seq_s`.

    Because `edlib` uses unit Levenshtein edit distance without an affine gap-open
    penalty, a single contiguous insertion or deletion can be arbitrarily split by
    1-2 bp `'='` matches when the indel sequence shares bases with its flank.
    When `seq_s` and `seq_r` are provided, any pure `M+D` or `M+I` run that is
    algebraically equivalent (at identical edit distance and 0 mismatches) to a
    single contiguous `D` or `I` is normalized to its single-gap form.
    """
    raw_ops: list[tuple[str, int]] = []
    for m in _EDLIB_CIGAR_RE.finditer(cigar_str):
        length = int(m.group(1))
        code = m.group(2)
        op = "M" if code == "=" else code
        raw_ops.append((op, length))
    ops = _coalesce_ops(raw_ops)
    if seq_s is None or seq_r is None or len(ops) < 3:
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
            j = i
            q_end = q_idx
            t_end = t_idx
            gap_count = 0
            candidates: list[tuple[int, int, int]] = []
            while j < n_ops and ops[j][0] in ("M", gap_type):
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
                    d_total = t_span - q_span
                    for split in range(q_span + 1):
                        if (
                            seq_s[q_idx : q_idx + split] == seq_r[t_idx : t_idx + split]
                            and seq_s[q_idx + split : cand_q_end]
                            == seq_r[cand_t_end - (q_span - split) : cand_t_end]
                        ):
                            if split > 0:
                                normalized.append(("M", split))
                            normalized.append(("D", d_total))
                            if q_span - split > 0:
                                normalized.append(("M", q_span - split))
                            collapsed = True
                            break
                else:
                    i_total = q_span - t_span
                    for split in range(t_span + 1):
                        if (
                            seq_s[q_idx : q_idx + split] == seq_r[t_idx : t_idx + split]
                            and seq_s[cand_q_end - (t_span - split) : cand_q_end]
                            == seq_r[t_idx + split : cand_t_end]
                        ):
                            if split > 0:
                                normalized.append(("M", split))
                            normalized.append(("I", i_total))
                            if t_span - split > 0:
                                normalized.append(("M", t_span - split))
                            collapsed = True
                            break
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


def _evaluate_mappy_hits_on_blocks(
    hits: Sequence[Any],
    query_offset: int,
    q_len: int,
    num_blocks: int,
    min_sv_len: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Evaluate `mappy` hits on the global 50 bp block grid `[50*k, min(50*(k+1), q_len))`.
    Returns `(hit_masked_intervals, hit_aligned_intervals)`.
    """
    hit_masked_intervals: list[tuple[int, int]] = []
    hit_aligned_intervals: list[tuple[int, int]] = []
    for hit in hits:
        if not getattr(hit, "cs", None):
            continue
        ops = parse_cs_to_query_ops(hit.cs, hit.strand)

        block_m = [0] * num_blocks
        block_x = [0] * num_blocks
        block_i = [0] * num_blocks
        block_d = [0] * num_blocks
        protected_ins: list[tuple[int, int]] = []

        q_pos = query_offset + int(hit.q_st)
        for op, length in ops:
            if op in ("M", "X", "I"):
                q_next = q_pos + length
                if op == "I" and length >= min_sv_len:
                    protected_ins.append((q_pos, q_next))
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
                q_pos = q_next
            elif op == "D":
                # Assign D to the block of its right-hand query base, or last block if at query end
                k = min(max(0, q_pos // REF_SUB_BLOCK_SIZE), num_blocks - 1)
                block_d[k] += length

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

            # Condition 1: (M + X) / L_k >= 0.90
            if (m_cnt + x_cnt) * 10 >= 9 * l_k:
                hit_aligned_blocks.append((b_s, b_e))
                # Condition 2: M / (M + X + I + D) >= 0.85
                if denom > 0 and m_cnt * 100 >= 85 * denom:
                    hit_qualifying_blocks.append((b_s, b_e))

        if hit_qualifying_blocks:
            hit_masked = subtract_intervals(hit_qualifying_blocks, protected_ins)
            hit_masked_intervals.extend(hit_masked)
        if hit_aligned_blocks:
            hit_aligned = subtract_intervals(hit_aligned_blocks, protected_ins)
            hit_aligned_intervals.extend(hit_aligned)

    return hit_masked_intervals, hit_aligned_intervals


def subtract_reference_intervals_mappy(
    query_seq: str,
    ref_region_seq: str,
    min_sv_len: int = 50,
    aligner: Any | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """
    Reference interval subtraction using `mappy` (`preset='asm20'`, `cs=True`) (§3.8.4-B).

    - Evaluates fixed non-overlapping 50 bp query blocks (`[0, 50), [50, 100), ...`,
      final tail block evaluated at actual length `L_k`).
    - Both forward (`+`) and reverse (`-`) hits are mapped to forward query coordinates.
    - Deletions (`D`) are assigned to the 50 bp block of the query base immediately to
      their right, or to the last block if located at the end of the query.
    - Any contiguous query insertion (`I`) with `length >= min_sv_len` within a hit is
      protected from being masked by that hit.
    - A block `k` is masked by a hit iff:
      1. `(M + X) / L_k >= 0.90` (at least 90% aligned query bases), AND
      2. `M / (M + X + I + D) >= 0.85` (block identity >= 0.85, I/D counted in bases).
    - Returns `(masked_intervals, unmasked_intervals, reference_like_bp)`.
    """
    q_len = len(query_seq)
    if q_len == 0:
        return [], [], 0
    if not ref_region_seq:
        return [], [(0, q_len)], 0

    q_upper = query_seq.upper()
    if aligner is None:
        aligner = mappy.Aligner(seq=ref_region_seq.upper(), preset="asm20")

    num_blocks = (q_len + REF_SUB_BLOCK_SIZE - 1) // REF_SUB_BLOCK_SIZE
    primary_hits = list(aligner.map(q_upper, cs=True))
    all_hit_masked, all_hit_aligned = _evaluate_mappy_hits_on_blocks(
        hits=primary_hits,
        query_offset=0,
        q_len=q_len,
        num_blocks=num_blocks,
        min_sv_len=min_sv_len,
    )
    masked_intervals = merge_intervals(all_hit_masked)
    aligned_intervals = merge_intervals(all_hit_aligned)

    # When one or more large query insertions (e.g. >= 600 bp in C8) separate homologous
    # flanks, minimap2's RMQ chainer may chain flanks across the insertions and clip
    # downstream flanks at zdrop during DP. Iteratively re-mapping only the unaligned
    # sub-intervals on the same 50 bp block grid discovers all clipped flanks without
    # re-mapping blocks that were already aligned and rejected due to < 0.85 identity.
    seen_slices: set[tuple[int, int]] = {(0, q_len)}
    while aligned_intervals:
        unaligned = complement_intervals(q_len, aligned_intervals)
        for u_s, u_e in unaligned:
            blk_s = (u_s // REF_SUB_BLOCK_SIZE) * REF_SUB_BLOCK_SIZE
            blk_e = min(
                q_len,
                ((u_e + REF_SUB_BLOCK_SIZE - 1) // REF_SUB_BLOCK_SIZE)
                * REF_SUB_BLOCK_SIZE,
            )
            if (
                blk_e - blk_s >= REF_SUB_BLOCK_SIZE
                and (blk_s, blk_e) not in seen_slices
            ):
                seen_slices.add((blk_s, blk_e))
                sub_hits = list(aligner.map(q_upper[blk_s:blk_e], cs=True))
                if sub_hits:
                    extra_masked, extra_aligned = _evaluate_mappy_hits_on_blocks(
                        hits=sub_hits,
                        query_offset=blk_s,
                        q_len=q_len,
                        num_blocks=num_blocks,
                        min_sv_len=min_sv_len,
                    )
                    if extra_masked:
                        all_hit_masked.extend(extra_masked)
                    if extra_aligned:
                        all_hit_aligned.extend(extra_aligned)
        prev_aligned = aligned_intervals
        masked_intervals = merge_intervals(all_hit_masked)
        aligned_intervals = merge_intervals(all_hit_aligned)
        if aligned_intervals == prev_aligned:
            break

    unmasked_intervals = complement_intervals(q_len, masked_intervals)
    reference_like_bp = sum(e - s for s, e in masked_intervals)
    return masked_intervals, unmasked_intervals, reference_like_bp


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

    1. Single forward-strand alignment (`strand == +1`).
    2. Gap-compressed identity `ID_gc = M / (M + X + gap_open_I + gap_open_D) >= cluster_id`.
    3. `unaligned_ends(S) + internal_ins_bp(S) < min_sv_len`.
    4. Sliding window protection: for every window of length `min(len(S), min_sv_len)`
       sliding along `S`, `matched_columns / aligned_columns >= cluster_win_id`.

    Uses `edlib` (`mode='NW'`, `task='path'`) when both sequences are `<= 1000 bp`,
    and `mappy` (`preset='asm20'`, `cs=True`) when either sequence is `> 1000 bp`.
    """
    if not seq_s or not seq_r:
        return False

    s = seq_s.upper()
    r = seq_r.upper()
    len_s = len(s)
    len_r = len(r)

    if s == r:
        return True

    # Fast bound check for Criterion 3: len_s - len_r <= unaligned_ends(S) + internal_ins_bp(S)
    if len_s - len_r >= min_sv_len:
        return False

    if len_s <= EDLIB_MAX_LEN and len_r <= EDLIB_MAX_LEN:
        res = edlib.align(s, r, mode="NW", task="path")
        cigar = res.get("cigar")
        if cigar is None or res.get("editDistance", -1) < 0:
            return False
        q_start = 0
        q_end = len_s
        ops = parse_edlib_cigar_to_ops(cigar, seq_s=s, seq_r=r)
    else:
        if aligner is None:
            aligner = mappy.Aligner(seq=r, preset="asm20")
        hits = list(aligner.map(s, cs=True))
        # Criterion 1: must have a single forward-strand alignment
        if len(hits) != 1:
            return False
        hit = hits[0]
        if (
            hit.strand != 1
            or not getattr(hit, "is_primary", True)
            or not getattr(hit, "cs", None)
        ):
            return False
        q_start = int(hit.q_st)
        q_end = int(hit.q_en)
        ops = parse_cs_to_query_ops(hit.cs, strand=1)

    # Criterion 2: Gap-compressed identity ID_gc >= cluster_id
    m_total = sum(length for op, length in ops if op == "M")
    x_total = sum(length for op, length in ops if op == "X")
    gap_open_i = sum(1 for op, _ in ops if op == "I")
    gap_open_d = sum(1 for op, _ in ops if op == "D")
    denom_gc = m_total + x_total + gap_open_i + gap_open_d
    if denom_gc <= 0:
        return False
    id_gc = m_total / denom_gc
    if id_gc + 1e-12 < cluster_id:
        return False

    # Criterion 3: unaligned_ends(S) + internal_ins_bp(S) < min_sv_len
    unaligned_ends_s = q_start + (len_s - q_end)
    ins_bp_s = sum(length for op, length in ops if op == "I")
    if unaligned_ends_s + ins_bp_s >= min_sv_len:
        return False

    # Criterion 4: Local sliding window protection along S
    win_len = min(len_s, min_sv_len)
    if win_len <= 0:
        return False

    match_arr = np.zeros(len_s, dtype=np.int32)
    mis_arr = np.zeros(len_s, dtype=np.int32)
    ins_arr = np.zeros(len_s, dtype=np.int32)
    del_bp_arr = np.zeros(len_s, dtype=np.int32)
    del_open_arr = np.zeros(len_s, dtype=np.int32)

    if q_start > 0:
        ins_arr[:q_start] = 1
    if q_end < len_s:
        ins_arr[q_end:] = 1

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
            q_pos += length
        elif op == "D":
            target_q = min(max(0, q_pos), len_s - 1)
            del_bp_arr[target_q] += length
            del_open_arr[target_q] += 1

    match_csum = np.zeros(len_s + 1, dtype=np.int32)
    mis_csum = np.zeros(len_s + 1, dtype=np.int32)
    ins_csum = np.zeros(len_s + 1, dtype=np.int32)
    del_bp_csum = np.zeros(len_s + 1, dtype=np.int32)
    del_open_csum = np.zeros(len_s + 1, dtype=np.int32)

    np.cumsum(match_arr, out=match_csum[1:])
    np.cumsum(mis_arr, out=mis_csum[1:])
    np.cumsum(ins_arr, out=ins_csum[1:])
    np.cumsum(del_bp_arr, out=del_bp_csum[1:])
    np.cumsum(del_open_arr, out=del_open_csum[1:])

    win_m = match_csum[win_len:] - match_csum[:-win_len]
    win_x = mis_csum[win_len:] - mis_csum[:-win_len]
    win_i = ins_csum[win_len:] - ins_csum[:-win_len]
    win_d_bp = del_bp_csum[win_len:] - del_bp_csum[:-win_len]
    win_d_open = del_open_csum[win_len:] - del_open_csum[:-win_len]

    is_compensating_indel_sub = ((win_i > 0) & (win_d_bp > 0)) | (win_d_open >= 4)
    win_aligned_cols = np.where(
        is_compensating_indel_sub,
        win_len + win_d_bp,
        np.maximum(1, win_m + win_x + win_d_open),
    )

    if np.any(win_m + 1e-12 < cluster_win_id * win_aligned_cols):
        return False

    return True
