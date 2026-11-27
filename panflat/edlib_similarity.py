"""Short-sequence (<= 1 kb) edlib semi-global alignment, asm20 two-piece affine realignment, and Criterion 1 guards (§3.6.3)."""

from __future__ import annotations

import math
import re
from typing import Sequence

import edlib

from panflat.gfa_io import revcomp

EDLIB_MAX_LEN: int = 1000

_EDLIB_CIGAR_RE = re.compile(r"([0-9]+)([=XID])")


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


def _evaluate_edlib_similarity_branch(
    s: str,
    r: str,
    min_sv_len: int,
    cluster_id: float,
) -> tuple[bool, list[tuple[str, int]], int, int, int, int, int]:
    """
    Run Stages 0–6 of the `<= EDLIB_MAX_LEN` `edlib` similarity check (§3.6.3)
    and return `(ok, ops, q_start, q_end, t_start, t_end, del_bp_internal)`.
    """
    len_s = len(s)
    len_r = len(r)
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
        return False, [], 0, 0, 0, 0, 0

    t_start_raw, t_end_incl = raw_locations[0]
    t_end_raw = t_end_incl + 1
    if t_start_raw < 0 or t_end_raw > len_r or t_start_raw >= t_end_raw:
        return False, [], 0, 0, 0, 0, 0

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
        return False, [], 0, 0, 0, 0, 0

    # Stage 5: Post-trim/post-recovery min aligned-core floor + fast Criteria 1d / 2 / 3
    min_aligned_core = min(_EDLIB_MIN_ALIGNED_CORE, len_s)
    if (
        not ops
        or (q_end - q_start) < min_aligned_core
        or (t_end - t_start) < min_aligned_core
    ):
        return False, [], 0, 0, 0, 0, 0

    del_bp_internal = 0
    m_total = 0
    x_total = 0
    ins_bp_s = 0
    gap_open_i = 0
    gap_open_d = 0
    for op, length in ops:
        if op == "M":
            m_total += length
        elif op == "X":
            x_total += length
        elif op == "I":
            ins_bp_s += length
            gap_open_i += 1
        elif op == "D":
            if length >= min_sv_len:
                return False, [], 0, 0, 0, 0, 0
            del_bp_internal += length
            gap_open_d += 1
    if del_bp_internal >= min_sv_len:
        return False, [], 0, 0, 0, 0, 0

    denom_gc = m_total + x_total + gap_open_i + gap_open_d
    if denom_gc <= 0 or (m_total / denom_gc) + 1e-12 < cluster_id:
        return False, [], 0, 0, 0, 0, 0

    unaligned_ends_s = q_start + (len_s - q_end)
    if unaligned_ends_s + ins_bp_s >= min_sv_len:
        return False, [], 0, 0, 0, 0, 0

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
        return False, [], 0, 0, 0, 0, 0
    return True, ops, q_start, q_end, t_start, t_end, del_bp_internal
