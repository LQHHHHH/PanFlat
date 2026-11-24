"""Global Invariants I, II, III verifier (`panflat verify`) and `--qc` alignment redundancy analyzer (§1, §3.8.5, §5.5, §7)."""

from __future__ import annotations

from array import array
from bisect import bisect_right
import collections
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Mapping, Sequence

import mappy
import numpy as np

from panflat.gfa_io import parse_fasta, revcomp
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    MAX_COMPACT_NID,
    UNRESOLVED_REASONS,
    BuildConfig,
    ChromosomeBuildResult,
    CompactNodeStore,
    InvariantViolationError,
    LazyStepOffsetSequence,
    LazyStepTuple,
    VerificationReport,
    is_compact_nid_str,
)
from panflat.progress import log_progress

OFFSETS_HEADER: list[str] = [
    "chrom",
    "ref_step",
    "node_id",
    "ref_start",
    "ref_end",
    "pan_start",
    "pan_end",
    "slot_ins_len",
]

SV_HEADER: list[str] = [
    "chrom",
    "pan_start",
    "pan_end",
    "event_id",
    "type",
    "ref_start",
    "ref_end",
    "anchors",
    "alleles",
    "n_carrier",
    "n_called",
    "region_id",
]

INSERTS_HEADER: list[str] = [
    "chrom",
    "pan_start",
    "pan_end",
    "event_id",
    "allele",
    "piece",
    "src_hap",
    "src_path",
    "src_intervals",
    "src_strand",
    "members",
    "join_before",
]

UNRESOLVED_HEADER: list[str] = [
    "record_id",
    "region_id",
    "event_id",
    "src_path",
    "src_start",
    "src_end",
    "src_strand",
    "length",
    "reason",
    "exported_fasta_id",
]

_CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_VERIFY_FAST_P_RE = re.compile(
    r"^[1-9][0-9]{0,8}[+-](?:,[1-9][0-9]{0,8}[+-])*$", re.ASCII
)
_VERIFY_FAST_W_RE = re.compile(r"^(?:[><][1-9][0-9]{0,8})+$", re.ASCII)
_REVCOMP_BYTES_TABLE = bytes.maketrans(b"ACGTNRYSWKMBDHV", b"TGCANYRSWMKVHDB")


def _translate_cds_seq(cds_seq: str, phase: int = 0) -> str:
    seq = cds_seq[phase:].upper()
    aa: list[str] = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        aa.append(_CODON_TABLE.get(codon, "X"))
    return "".join(aa)


def _merge_intervals(intervals: Sequence[tuple[int, int]], max_len: int) -> int:
    """Return total union length of half-open intervals clipped to `[0, max_len)`."""
    clipped: list[tuple[int, int]] = []
    for s, e in intervals:
        cs = max(0, int(s))
        ce = min(max_len, int(e))
        if cs < ce:
            clipped.append((cs, ce))
    if not clipped:
        return 0
    clipped.sort()
    cur_s, cur_e = clipped[0]
    total = 0
    for s, e in clipped[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return min(max_len, max(0, total))


def _reconstruct_ref_seq_from_result(result: ChromosomeBuildResult, pan_up: str) -> str:
    """Reconstruct contiguous uppercase reference sequence from `(pan_up, result.step_offsets)`."""
    so = result.step_offsets
    if len(so) == 0 or not pan_up:
        return ""
    if isinstance(so, LazyStepOffsetSequence):
        n_st = len(so)
        slot_breaks = (np.flatnonzero(so.slot_ins_lens[1:] > 0) + 1).tolist()
        blk_starts = [0, *slot_breaks]
        blk_ends = [*slot_breaks, n_st]
        return "".join(
            pan_up[int(so.pan_starts[b_s]) : int(so.pan_ends[b_e - 1])]
            for b_s, b_e in zip(blk_starts, blk_ends)
        )
    parts: list[str] = []
    cur_ps = so[0].pan_start
    cur_pe = so[0].pan_end
    for row in so[1:]:
        if row.slot_ins_len > 0:
            parts.append(pan_up[cur_ps:cur_pe])
            cur_ps = row.pan_start
        cur_pe = row.pan_end
    parts.append(pan_up[cur_ps:cur_pe])
    return "".join(parts)


def _map_queries_parallel(
    aligner_main: mappy.Aligner | None,
    aligner_short: mappy.Aligner | None,
    query_indices: Sequence[int],
    piece_bytes: Sequence[bytes],
    piece_lens: Sequence[int],
    n_threads: int,
    t_starts: Sequence[int] | None = None,
    t_ends: Sequence[int] | None = None,
) -> list[tuple[int, list[tuple[int, int, float]]]]:
    """
    Map `query_indices` against read-only `aligner_main` (and optional `aligner_short` for
    pieces `<= 500 bp`) using C-level GIL-free `mappy.ThreadBuffer` multithreading,
    pre-encoded ASCII bytes, and adaptive LPT dynamic work stealing.
    """
    if (aligner_main is None and aligner_short is None) or not query_indices:
        return []

    ordered_q = sorted(query_indices, key=lambda idx: (-piece_lens[idx], idx))
    n_q = len(ordered_q)
    has_tbuf = hasattr(mappy, "ThreadBuffer")
    workers = min(max(1, int(n_threads)), n_q) if has_tbuf else 1
    check_sep = t_starts is not None and t_ends is not None

    def _map_one(
        idx_q: int,
        buf_main: Any,
        buf_short: Any,
    ) -> list[tuple[int, int, float]]:
        q_b = piece_bytes[idx_q]
        local_hits: list[tuple[int, int, float]] = []
        if aligner_main is not None:
            main_iter = (
                aligner_main.map(q_b, buf=buf_main)
                if buf_main is not None
                else aligner_main.map(q_b)
            )
            for hit in main_iter:
                if hit.blen < 150:
                    continue
                if check_sep:
                    assert t_starts is not None and t_ends is not None
                    pos_t = bisect_right(t_starts, hit.r_st) - 1
                    if pos_t < 0 or hit.r_en > t_ends[pos_t]:
                        continue
                local_hits.append((hit.q_st, hit.q_en, hit.mlen / hit.blen))
        if aligner_short is not None and piece_lens[idx_q] <= 500:
            short_iter = (
                aligner_short.map(q_b, buf=buf_short)
                if buf_short is not None
                else aligner_short.map(q_b)
            )
            for hit in short_iter:
                if hit.blen < 150:
                    continue
                if check_sep:
                    assert t_starts is not None and t_ends is not None
                    pos_t = bisect_right(t_starts, hit.r_st) - 1
                    if pos_t < 0 or hit.r_en > t_ends[pos_t]:
                        continue
                local_hits.append((hit.q_st, hit.q_en, hit.mlen / hit.blen))
        return local_hits

    if workers <= 1:
        buf_main = mappy.ThreadBuffer() if (has_tbuf and aligner_main is not None) else None
        buf_short = mappy.ThreadBuffer() if (has_tbuf and aligner_short is not None) else None
        return [(idx_q, _map_one(idx_q, buf_main, buf_short)) for idx_q in ordered_q]

    slot_hits: list[list[tuple[int, int, float]]] = [[] for _ in range(n_q)]
    cursor = 0
    cursor_lock = threading.Lock()

    def _worker() -> None:
        nonlocal cursor
        buf_main = mappy.ThreadBuffer() if aligner_main is not None else None
        buf_short = mappy.ThreadBuffer() if aligner_short is not None else None
        while True:
            with cursor_lock:
                start = cursor
                if start >= n_q:
                    return
                if piece_lens[ordered_q[start]] >= 2000:
                    step = 1
                else:
                    step = min(8, max(1, (n_q - start) // (workers * 4)))
                end = min(n_q, start + step)
                cursor = end
            for pos in range(start, end):
                slot_hits[pos] = _map_one(ordered_q[pos], buf_main, buf_short)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker) for _ in range(workers)]
        for fut in futs:
            fut.result()

    return list(zip(ordered_q, slot_hits))


def run_qc(
    result: ChromosomeBuildResult,
    config: BuildConfig | None = None,
) -> dict[str, Any]:
    """
    Compute `--qc` alignment redundancy metrics for inserted pieces (§7, F26).
    Uses 2-stage self-masking (`pan_no_long` + bitwise dyadic halving across `long_rows`)
    so 100% self-hits never suppress ~90–92% homologs via `minimap2` `pri_ratio = 0.80`,
    while keeping index builds `O(1)` in chromosome size (`O(|pan.fa| + log M * sum |piece|)`).
    """
    n_threads = max(1, int(config.threads)) if config is not None else 1
    short_lengths: list[int] = []
    long_rows: list[dict[str, Any]] = []

    for r in result.inserts_rows:
        p_s = int(r["pan_start"])
        p_e = int(r["pan_end"])
        length = p_e - p_s
        if 50 <= length < 150:
            short_lengths.append(length)
        elif length >= 150:
            long_rows.append(r)

    short_count = len(short_lengths)
    short_bp = sum(short_lengths)
    short_lengths_str = (
        ",".join(str(x) for x in short_lengths) if short_lengths else "0"
    )

    long_count = len(long_rows)
    long_bp = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in long_rows)

    metrics: dict[str, Any] = {
        "qc_enabled": "true",
        "qc_short_50_149_count": short_count,
        "qc_short_50_149_bp": short_bp,
        "qc_short_50_149_lengths": short_lengths_str,
        "qc_long_ge150_count": long_count,
        "qc_long_ge150_bp": long_bp,
        "qc_ge150_count": long_count,
        "qc_ge150_bp": long_bp,
    }

    pan_up = result.pan_seq.upper()
    piece_hits_by_idx: list[list[tuple[int, int, float]]] = [
        [] for _ in long_rows
    ]
    stage1_ref_hits_by_idx: list[list[tuple[int, int, float]]] = [
        [] for _ in long_rows
    ]
    has_step_offsets = len(result.step_offsets) > 0
    if long_rows and pan_up:
        piece_seqs: list[str] = []
        piece_lens: list[int] = []
        for r in long_rows:
            p_s = int(r["pan_start"])
            p_e = int(r["pan_end"])
            piece_lens.append(p_e - p_s)
            piece_seqs.append(str(r.get("seq") or pan_up[p_s:p_e]).upper())
        piece_bytes = [s.encode("ascii") for s in piece_seqs]

        if has_step_offsets:
            stage1_target_seq = _reconstruct_ref_seq_from_result(result, pan_up)
        else:
            pan_buf = bytearray(pan_up.encode("ascii"))
            for r in long_rows:
                p_s = int(r["pan_start"])
                p_e = int(r["pan_end"])
                pan_buf[p_s:p_e] = b"N" * (p_e - p_s)
            stage1_target_seq = pan_buf.decode("ascii")
            del pan_buf

        need_short = any(pl <= 500 for pl in piece_lens)

        # Stage 1: Query every long piece against contiguous reference sequence
        t_s1 = time.monotonic()
        aligner_asm20 = (
            mappy.Aligner(seq=stage1_target_seq, preset="asm20", best_n=50)
            if stage1_target_seq
            else None
        )
        aligner_short = (
            mappy.Aligner(
                seq=stage1_target_seq,
                k=13,
                w=5,
                min_cnt=2,
                min_chain_score=20,
                best_n=50,
            )
            if (stage1_target_seq and need_short)
            else None
        )
        stage1_results = _map_queries_parallel(
            aligner_main=aligner_asm20,
            aligner_short=aligner_short,
            query_indices=range(len(long_rows)),
            piece_bytes=piece_bytes,
            piece_lens=piece_lens,
            n_threads=n_threads,
        )
        for idx_r, q_hits in stage1_results:
            if q_hits:
                piece_hits_by_idx[idx_r].extend(q_hits)
                if has_step_offsets:
                    stage1_ref_hits_by_idx[idx_r].extend(q_hits)
        del aligner_asm20, aligner_short, stage1_target_seq, stage1_results
        log_progress(
            f"    --qc Stage 1 (ref alignment): {len(long_rows):,} pieces "
            f"({long_bp:,} bp, threads={min(n_threads, len(long_rows))}) "
            f"done in {time.monotonic() - t_s1:.2f}s"
        )

        # Stage 2: Bitwise dyadic halving across long pieces for exact piece-to-piece hits without self-hits
        n_long = len(long_rows)
        if n_long >= 2:
            sep = "N" * 200
            n_bits = (n_long - 1).bit_length()
            total_rounds = 2 * n_bits
            round_idx = 0
            for bit_idx in range(n_bits):
                for target_bit in (0, 1):
                    t_round = time.monotonic()
                    target_indices = [
                        j
                        for j in range(n_long)
                        if ((j >> bit_idx) & 1) == target_bit
                    ]
                    query_indices = [
                        j
                        for j in range(n_long)
                        if ((j >> bit_idx) & 1) != target_bit
                    ]
                    if not target_indices or not query_indices:
                        continue
                    t_starts: list[int] = []
                    t_ends: list[int] = []
                    t_chunks: list[str] = []
                    cur_off = 0
                    for idx_t, j in enumerate(target_indices):
                        t_seq = piece_seqs[j]
                        if idx_t > 0:
                            t_chunks.append(sep)
                            cur_off += len(sep)
                        t_starts.append(cur_off)
                        t_ends.append(cur_off + len(t_seq))
                        t_chunks.append(t_seq)
                        cur_off += len(t_seq)
                    subset_seq = "".join(t_chunks)
                    sub_asm20 = mappy.Aligner(
                        seq=subset_seq, preset="asm20", best_n=50
                    )
                    round_need_short = need_short and any(
                        piece_lens[q] <= 500 for q in query_indices
                    )
                    sub_short = (
                        mappy.Aligner(
                            seq=subset_seq,
                            k=13,
                            w=5,
                            min_cnt=2,
                            min_chain_score=20,
                            best_n=50,
                        )
                        if round_need_short
                        else None
                    )
                    round_results = _map_queries_parallel(
                        aligner_main=sub_asm20,
                        aligner_short=sub_short,
                        query_indices=query_indices,
                        piece_bytes=piece_bytes,
                        piece_lens=piece_lens,
                        n_threads=n_threads,
                        t_starts=t_starts,
                        t_ends=t_ends,
                    )
                    for idx_q, q_hits in round_results:
                        if q_hits:
                            piece_hits_by_idx[idx_q].extend(q_hits)
                    del sub_asm20, sub_short, subset_seq, round_results
                    round_idx += 1
                    log_progress(
                        f"    --qc Stage 2 round {round_idx}/{total_rounds}: "
                        f"targets={len(target_indices):,}, queries={len(query_indices):,} "
                        f"({time.monotonic() - t_round:.2f}s)"
                    )
        del piece_bytes

    for tau in (0.90, 0.95, 0.98):
        red_bp = 0
        for idx_r, r in enumerate(long_rows):
            piece_len = int(r["pan_end"]) - int(r["pan_start"])
            cov_intervals = [
                (q_s, q_e)
                for q_s, q_e, ident in piece_hits_by_idx[idx_r]
                if ident >= tau
            ]
            red_bp += _merge_intervals(cov_intervals, piece_len)

        pct = (100.0 * red_bp / long_bp) if long_bp > 0 else 0.0
        ratio = (red_bp / long_bp) if long_bp > 0 else 0.0
        metrics[f"qc_redundancy_{tau:.2f}_bp"] = red_bp
        metrics[f"qc_redundancy_{tau:.2f}_pct"] = f"{pct:.4f}"
        metrics[f"qc_redundancy_{tau:.2f}_ratio"] = f"{ratio:.6f}"

    # Spec §7 item 5: Separate CEN_ISLAND whole-reference hit statistics
    cen_event_ids = {
        ev.event_id
        for ev in result.sv_events
        if ev.event_type == "CEN_ISLAND"
    }
    cen_long_indices = [
        idx_r
        for idx_r, r in enumerate(long_rows)
        if str(r.get("event_id", "")) in cen_event_ids
    ]
    cen_long_rows = [long_rows[idx_r] for idx_r in cen_long_indices]
    cen_long_count = len(cen_long_rows)
    cen_long_bp = sum(
        int(r["pan_end"]) - int(r["pan_start"]) for r in cen_long_rows
    )
    metrics["qc_cen_island_ge150_count"] = cen_long_count
    metrics["qc_cen_island_ge150_bp"] = cen_long_bp
    metrics["qc_cen_island_long_ge150_count"] = cen_long_count
    metrics["qc_cen_island_long_ge150_bp"] = cen_long_bp

    cen_hits_by_idx: list[list[tuple[int, int, float]]] = [
        stage1_ref_hits_by_idx[idx_r] if has_step_offsets else []
        for idx_r in cen_long_indices
    ]

    for tau in (0.90, 0.95, 0.98):
        c_bp = 0
        for idx_c, r in enumerate(cen_long_rows):
            piece_len = int(r["pan_end"]) - int(r["pan_start"])
            cov_ivs = [
                (q_s, q_e)
                for q_s, q_e, ident in cen_hits_by_idx[idx_c]
                if ident >= tau
            ]
            c_bp += _merge_intervals(cov_ivs, piece_len)
        c_pct = (100.0 * c_bp / cen_long_bp) if cen_long_bp > 0 else 0.0
        c_ratio = (c_bp / cen_long_bp) if cen_long_bp > 0 else 0.0
        metrics[f"qc_cen_island_ref_hit_{tau:.2f}_bp"] = c_bp
        metrics[f"qc_cen_island_ref_hit_{tau:.2f}_pct"] = f"{c_pct:.4f}"
        metrics[f"qc_cen_island_ref_hit_{tau:.2f}_ratio"] = f"{c_ratio:.6f}"

    return metrics


compute_qc_metrics = run_qc


def _parse_tsv_file(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    lines = [
        l.rstrip("\r\n")
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("##")
    ]
    if not lines:
        return [], []
    header = lines[0].lstrip("#").split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(header):
            raise InvariantViolationError(
                f"Malformed TSV row in '{path}': expected {len(header)} columns, got {len(parts)}"
            )
        rows.append(dict(zip(header, parts)))
    return header, rows


def _parse_fasta_allow_empty(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    records: dict[str, str] = {}
    seen_names: set[str] = set()
    cur_name: str | None = None
    cur_has_seq = False
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    if not cur_has_seq:
                        raise InvariantViolationError(
                            f"Empty sequence for FASTA record '{cur_name}' in '{path}'"
                        )
                    records[cur_name] = "".join(chunks)
                hdr_body = line[1:].strip()
                if not hdr_body:
                    raise InvariantViolationError(
                        f"Empty FASTA header at line {line_no} in '{path}'"
                    )
                seq_name = hdr_body.split()[0]
                if seq_name in seen_names:
                    raise InvariantViolationError(
                        f"Duplicate FASTA sequence name '{seq_name}' in '{path}'"
                    )
                seen_names.add(seq_name)
                cur_name = seq_name
                cur_has_seq = False
                chunks = []
            else:
                if cur_name is None:
                    raise InvariantViolationError(
                        f"FASTA sequence data before header at line {line_no} in '{path}'"
                    )
                chunk = (
                    line
                    if (line.isascii() and line.isalpha())
                    else "".join(line.split())
                )
                if not chunk:
                    continue
                if not (chunk.isascii() and chunk.isalpha()):
                    raise InvariantViolationError(
                        f"Non-alphabetic characters in FASTA record '{cur_name}' in '{path}'"
                    )
                cur_has_seq = True
                chunks.append(chunk.upper())
    if cur_name is not None:
        if not cur_has_seq:
            raise InvariantViolationError(
                f"Empty sequence for FASTA record '{cur_name}' in '{path}'"
            )
        records[cur_name] = "".join(chunks)
    return records


def _parse_fasta_with_header_kv(
    path: Path,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Parse FASTA sequences and `key=value` header tokens for `{p}.unresolved.fa` (§5, §5.5)."""
    if not path.is_file():
        return {}, {}
    records: dict[str, str] = {}
    header_kv: dict[str, dict[str, str]] = {}
    seen_names: set[str] = set()
    cur_name: str | None = None
    cur_has_seq = False
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    if not cur_has_seq:
                        raise InvariantViolationError(
                            f"Empty sequence for FASTA record '{cur_name}' in '{path}'"
                        )
                    records[cur_name] = "".join(chunks)
                hdr_body = line[1:].strip()
                if not hdr_body:
                    raise InvariantViolationError(
                        f"Empty FASTA header at line {line_no} in '{path}'"
                    )
                tokens = hdr_body.split()
                seq_name = tokens[0]
                if seq_name in seen_names:
                    raise InvariantViolationError(
                        f"Duplicate FASTA sequence name '{seq_name}' in '{path}'"
                    )
                seen_names.add(seq_name)
                kv_map: dict[str, str] = {}
                for tok in tokens[1:]:
                    if "=" not in tok:
                        raise InvariantViolationError(
                            f"Malformed key=value token '{tok}' in FASTA header for '{seq_name}' in '{path}'"
                        )
                    k_tok, v_tok = tok.split("=", 1)
                    if not k_tok or not v_tok:
                        raise InvariantViolationError(
                            f"Empty key or value in token '{tok}' in FASTA header for '{seq_name}' in '{path}'"
                        )
                    kv_map[k_tok] = v_tok
                header_kv[seq_name] = kv_map
                cur_name = seq_name
                cur_has_seq = False
                chunks = []
            else:
                if cur_name is None:
                    raise InvariantViolationError(
                        f"FASTA sequence data before header at line {line_no} in '{path}'"
                    )
                chunk = (
                    line
                    if (line.isascii() and line.isalpha())
                    else "".join(line.split())
                )
                if not chunk:
                    continue
                if not (chunk.isascii() and chunk.isalpha()):
                    raise InvariantViolationError(
                        f"Non-alphabetic characters in FASTA record '{cur_name}' in '{path}'"
                    )
                cur_has_seq = True
                chunks.append(chunk.upper())
    if cur_name is not None:
        if not cur_has_seq:
            raise InvariantViolationError(
                f"Empty sequence for FASTA record '{cur_name}' in '{path}'"
            )
        records[cur_name] = "".join(chunks)
    return records, header_kv


@dataclass(frozen=True)
class _GFAPathMeta:
    """Lightweight metadata for a GFA `P` path or `W` walk used in `--backbone-prefix` verification."""

    src_path: str
    pan_name: str
    sample: str
    hap: str
    hap_id: str
    seqid: str
    is_w_line: bool


def _parse_pansn_meta(path_name: str) -> tuple[str, str, str, str]:
    if "#" in path_name:
        parts = path_name.split("#")
        if len(parts) >= 3:
            sample = parts[0].strip()
            hap = parts[1].strip()
            seqid = "#".join(parts[2:]).strip()
        else:
            sample = parts[0].strip()
            hap = parts[1].strip()
            seqid = parts[1].strip()
        return sample, hap, f"{sample}#{hap}", seqid
    return path_name, "1", path_name, path_name


@dataclass
class _SparsePathSlices:
    """Stores only requested source intervals for a chromosome-scale GFA path."""

    base_offset: int
    total_len: int
    slices: dict[tuple[int, int], str]


def _parse_gfa_paths_and_walks(
    gfa_path: Path,
    required_slices_by_path: Mapping[str, set[tuple[int, int]]] | None = None,
    out_path_metas: list[_GFAPathMeta] | None = None,
) -> tuple[Mapping[str, str], dict[str, Any]]:
    """Independently parse GFA S, P, W lines into `(nodes, path_seqs)`."""
    file_size = gfa_path.stat().st_size
    if file_size > 10_000_000 and required_slices_by_path is not None:
        from array import array as _c_array

        seq_buf = bytearray()
        node_lens_arr = _c_array("i", [0])
        expected_nid = 1
        all_seq_int = True
        max_nid = 0
        pw_offsets: list[int] = []

        with gfa_path.open("rb") as fh:
            while True:
                line_off = fh.tell()
                raw_b = fh.readline()
                if not raw_b:
                    break
                b0 = raw_b[0]
                if b0 == 76:  # b'L'
                    continue
                if b0 == 83 and raw_b[1:2] == b"\t":  # b'S\t'
                    t1 = raw_b.find(b"\t", 2)
                    if t1 <= 2:
                        raise InvariantViolationError(
                            "Malformed S line in GFA (missing segment ID or sequence)"
                        )
                    nid_b = raw_b[2:t1]
                    if not (
                        1 <= len(nid_b) <= 9
                        and nid_b.isdigit()
                        and nid_b[0] != 48
                    ):
                        all_seq_int = False
                        break
                    nid_int = int(nid_b)
                    if nid_int != expected_nid or nid_int > MAX_COMPACT_NID:
                        all_seq_int = False
                        break
                    t2 = raw_b.find(b"\t", t1 + 1)
                    seq_b = (
                        raw_b[t1 + 1 :].rstrip(b"\r\n")
                        if t2 < 0
                        else raw_b[t1 + 1 : t2]
                    ).upper()
                    if not seq_b or not seq_b.isalpha() or not seq_b.isascii():
                        raise InvariantViolationError(
                            f"Invalid or missing segment sequence for node '{nid_int}' in GFA"
                        )
                    seq_buf.extend(seq_b)
                    node_lens_arr.append(len(seq_b))
                    expected_nid += 1
                    max_nid = nid_int
                elif (b0 == 80 or b0 == 87) and raw_b[1:2] == b"\t":  # b'P\t' or b'W\t'
                    pw_offsets.append(line_off)

        if all_seq_int and max_nid > 0:
            node_lens = np.frombuffer(node_lens_arr, dtype=np.int32).copy()
            del node_lens_arr
            seq_offsets = np.zeros(max_nid + 2, dtype=np.int64)
            np.cumsum(node_lens[1:], dtype=np.int64, out=seq_offsets[2:])
            node_store = CompactNodeStore(
                seq_bytes=bytes(seq_buf),
                offsets=seq_offsets,
                node_lens=node_lens,
                max_nid=max_nid,
                extra={},
                count=max_nid,
            )
            del seq_buf
            path_seqs_sparse: dict[str, Any] = {}
            fast_metas: list[_GFAPathMeta] = []
            w_re = re.compile(r"([><])([1-9][0-9]{0,8})", re.ASCII)
            fast_ok = True

            def _matching_reqs(
                key1: str, key2: str | None = None
            ) -> set[tuple[int, int]]:
                reqs: set[tuple[int, int]] = set()
                for q_path, ivs in required_slices_by_path.items():
                    lookup = (
                        q_path.split("::", 1)[-1] if "::" in q_path else q_path
                    )
                    if (
                        lookup == key1
                        or key1.startswith(f"{lookup}:")
                        or (
                            key2 is not None
                            and (
                                lookup == key2 or key2.startswith(f"{lookup}:")
                            )
                        )
                    ):
                        reqs.update(ivs)
                return reqs

            def _extract_reqs_from_codes(
                codes: np.ndarray,
                reqs: set[tuple[int, int]],
                base_offset: int,
            ) -> tuple[int, dict[tuple[int, int], str]]:
                nids_abs = np.abs(codes)
                if int(nids_abs.shape[0]) > 0 and (
                    int(nids_abs.min()) <= 0 or int(nids_abs.max()) > max_nid
                ):
                    raise InvariantViolationError(
                        "GFA path/walk references unknown segment ID"
                    )
                step_lens = node_lens[nids_abs]
                n_c = int(codes.shape[0])
                cum_bp = np.empty(n_c + 1, dtype=np.int64)
                cum_bp[0] = 0
                np.cumsum(step_lens, dtype=np.int64, out=cum_bp[1:])
                total_len = int(cum_bp[-1])
                sl_map: dict[tuple[int, int], str] = {}
                for s_i, e_i in reqs:
                    rs = s_i - base_offset
                    re_ = e_i - base_offset
                    if 0 <= rs <= re_ <= total_len:
                        if rs == re_:
                            sl_map[(s_i, e_i)] = ""
                        else:
                            st_lo = (
                                int(np.searchsorted(cum_bp, rs, side="right"))
                                - 1
                            )
                            st_hi = int(
                                np.searchsorted(cum_bp, re_, side="left")
                            )
                            sub_seq = LazyStepTuple(codes[st_lo:st_hi]).spell(
                                node_store
                            )
                            base_lo = int(cum_bp[st_lo])
                            sl_map[(s_i, e_i)] = sub_seq[
                                rs - base_lo : re_ - base_lo
                            ]
                return total_len, sl_map

            with gfa_path.open("rb") as fh_pw:
                for off_pos in pw_offsets:
                    fh_pw.seek(off_pos)
                    raw = fh_pw.readline().decode("utf-8")
                    parts = raw.rstrip("\r\n").split("\t")
                    rec_type = parts[0]
                    if rec_type == "P" and len(parts) >= 3:
                        pname = parts[1]
                        step_str = parts[2].strip()
                        if not _VERIFY_FAST_P_RE.match(step_str):
                            fast_ok = False
                            break
                        s_p, h_p, hid_p, seq_p = _parse_pansn_meta(pname)
                        pan_n = f"{s_p}#{h_p}#{seq_p}" if "#" in pname else pname
                        fast_metas.append(
                            _GFAPathMeta(pname, pan_n, s_p, h_p, hid_p, seq_p, False)
                        )
                        reqs = _matching_reqs(pname)
                        if not reqs:
                            path_seqs_sparse[pname] = _SparsePathSlices(
                                0, 0, {}
                            )
                            continue
                        nids = np.fromstring(
                            step_str.translate({43: None, 45: None}),
                            dtype=np.int64,
                            sep=",",
                        )
                        if int(nids.shape[0]) > 0 and int(nids.max()) > max_nid:
                            fast_ok = False
                            break
                        nids32 = nids.astype(np.int32)
                        signs = np.frombuffer(
                            step_str.encode("ascii").translate(
                                None, b"0123456789, \t\r\n"
                            ),
                            dtype=np.uint8,
                        )
                        codes = np.where(signs == 45, -nids32, nids32).astype(
                            np.int32, copy=False
                        )
                        del step_str, nids, nids32, signs
                        total_len, sl_map = _extract_reqs_from_codes(
                            codes, reqs, 0
                        )
                        path_seqs_sparse[pname] = _SparsePathSlices(
                            0, total_len, sl_map
                        )
                    elif rec_type == "W" and len(parts) >= 7:
                        sample, hap, seqid = parts[1], parts[2], parts[3]
                        start = 0 if parts[4] == "*" else int(parts[4])
                        end = -1 if parts[5] == "*" else int(parts[5])
                        walk_str = parts[6].strip()
                        if not _VERIFY_FAST_W_RE.match(walk_str):
                            fast_ok = False
                            break
                        key_plain = f"{sample}#{hap}#{seqid}"
                        if end >= 0:
                            key_with_coords = (
                                f"{sample}#{hap}#{seqid}:{start}-{end}"
                            )
                            fast_metas.append(
                                _GFAPathMeta(
                                    key_with_coords,
                                    key_plain,
                                    sample,
                                    hap,
                                    f"{sample}#{hap}",
                                    seqid,
                                    True,
                                )
                            )
                            reqs = _matching_reqs(key_with_coords, key_plain)
                            if not reqs:
                                sp_empty = _SparsePathSlices(
                                    start, max(0, end - start), {}
                                )
                                path_seqs_sparse[key_with_coords] = sp_empty
                                if key_plain not in path_seqs_sparse:
                                    path_seqs_sparse[key_plain] = sp_empty
                                continue
                        codes64 = np.asarray(
                            [
                                int(m.group(2))
                                if m.group(1) == ">"
                                else -int(m.group(2))
                                for m in w_re.finditer(walk_str)
                            ],
                            dtype=np.int64,
                        )
                        if (
                            int(codes64.shape[0]) > 0
                            and int(np.abs(codes64).max()) > max_nid
                        ):
                            fast_ok = False
                            break
                        codes = codes64.astype(np.int32)
                        walk_len = int(node_lens[np.abs(codes)].sum())
                        computed_end = start + walk_len if end < 0 else end
                        key_with_coords = (
                            f"{sample}#{hap}#{seqid}:{start}-{computed_end}"
                        )
                        if end < 0:
                            fast_metas.append(
                                _GFAPathMeta(
                                    key_with_coords,
                                    key_plain,
                                    sample,
                                    hap,
                                    f"{sample}#{hap}",
                                    seqid,
                                    True,
                                )
                            )
                        reqs = _matching_reqs(key_with_coords, key_plain)
                        if not reqs:
                            sp_empty = _SparsePathSlices(start, walk_len, {})
                            path_seqs_sparse[key_with_coords] = sp_empty
                            if key_plain not in path_seqs_sparse:
                                path_seqs_sparse[key_plain] = sp_empty
                            continue
                        total_len, sl_map = _extract_reqs_from_codes(
                            codes, reqs, start
                        )
                        sp_obj = _SparsePathSlices(start, total_len, sl_map)
                        path_seqs_sparse[key_with_coords] = sp_obj
                        if key_plain not in path_seqs_sparse:
                            path_seqs_sparse[key_plain] = sp_obj

            if fast_ok:
                if out_path_metas is not None:
                    out_path_metas.extend(fast_metas)
                return node_store, path_seqs_sparse

    nodes: dict[str, str] = {}
    raw_p: list[tuple[str, str]] = []
    raw_w: list[tuple[str, str, str, int, int, str]] = []

    for raw in gfa_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.rstrip("\r\n").split("\t")
        rec_type = parts[0]
        if rec_type == "S" and len(parts) >= 3:
            nid_s = parts[1].strip()
            seq_s = parts[2].strip()
            if not seq_s or not (seq_s.isascii() and seq_s.isalpha()):
                raise InvariantViolationError(
                    f"Invalid or missing segment sequence for node '{nid_s}' in GFA"
                )
            if nid_s in nodes:
                raise InvariantViolationError(
                    f"Duplicate S-line segment ID '{nid_s}' in GFA"
                )
            nodes[nid_s] = seq_s.upper()
        elif rec_type == "P" and len(parts) >= 3:
            raw_p.append((parts[1], parts[2]))
        elif rec_type == "W" and len(parts) >= 7:
            s_val = 0 if parts[4] == "*" else int(parts[4])
            e_val = -1 if parts[5] == "*" else int(parts[5])
            raw_w.append(
                (parts[1], parts[2], parts[3], s_val, e_val, parts[6])
            )

    path_seqs: dict[str, tuple[int, str]] = {}
    for pname, step_str in raw_p:
        if out_path_metas is not None:
            s_p, h_p, hid_p, seq_p = _parse_pansn_meta(pname)
            pan_n = f"{s_p}#{h_p}#{seq_p}" if "#" in pname else pname
            out_path_metas.append(
                _GFAPathMeta(pname, pan_n, s_p, h_p, hid_p, seq_p, False)
            )
        chunks = []
        for token in step_str.split(","):
            tok = token.strip()
            if not tok:
                continue
            nid, orient = tok[:-1], tok[-1]
            if orient not in ("+", "-") or nid not in nodes:
                raise InvariantViolationError(
                    f"Malformed or unknown step '{tok}' in P line '{pname}'"
                )
            seq = nodes[nid]
            chunks.append(seq if orient == "+" else revcomp(seq))
        path_seqs[pname] = (0, "".join(chunks))

    w_full_re = re.compile(r"^(?:[><][^><\s]+)+$")
    w_step_re = re.compile(r"([><])([^><\s]+)")
    for sample, hap, seqid, start, end, walk_str in raw_w:
        walk_clean = walk_str.strip()
        if not walk_clean or not w_full_re.match(walk_clean):
            raise InvariantViolationError(
                f"Malformed W-line walk string '{walk_clean}' for '{sample}#{hap}#{seqid}'"
            )
        chunks = []
        for match in w_step_re.finditer(walk_clean):
            orient_sym, nid = match.group(1), match.group(2)
            if nid not in nodes:
                raise InvariantViolationError(
                    f"Unknown node '{nid}' in W line for '{sample}#{hap}#{seqid}'"
                )
            seq = nodes[nid]
            chunks.append(seq if orient_sym == ">" else revcomp(seq))
        full_seq = "".join(chunks)
        computed_end = start + len(full_seq) if end < 0 else end
        key_with_coords = f"{sample}#{hap}#{seqid}:{start}-{computed_end}"
        key_plain = f"{sample}#{hap}#{seqid}"
        if out_path_metas is not None:
            out_path_metas.append(
                _GFAPathMeta(
                    key_with_coords,
                    key_plain,
                    sample,
                    hap,
                    f"{sample}#{hap}",
                    seqid,
                    True,
                )
            )
        path_seqs[key_with_coords] = (start, full_seq)
        if key_plain not in path_seqs:
            path_seqs[key_plain] = (start, full_seq)

    return nodes, path_seqs


def _extract_source_slice(
    path_seqs: Mapping[str, Any],
    src_path: str,
    src_start: int,
    src_end: int,
    src_strand: str,
) -> str:
    lookup = src_path
    if "::" in lookup and lookup not in path_seqs:
        lookup = lookup.split("::", 1)[-1]

    if lookup in path_seqs:
        val = path_seqs[lookup]
        if isinstance(val, _SparsePathSlices):
            if (src_start, src_end) in val.slices:
                sub = val.slices[(src_start, src_end)]
                return sub if src_strand == "+" else revcomp(sub)
        else:
            base_offset, full_seq = val
            rs = src_start - base_offset
            re_ = src_end - base_offset
            if 0 <= rs <= re_ <= len(full_seq):
                sub = full_seq[rs:re_]
                return sub if src_strand == "+" else revcomp(sub)

    for k, val in path_seqs.items():
        if k == lookup or k.startswith(f"{lookup}:"):
            if isinstance(val, _SparsePathSlices):
                if (src_start, src_end) in val.slices:
                    sub = val.slices[(src_start, src_end)]
                    return sub if src_strand == "+" else revcomp(sub)
            else:
                cand_offset, cand_seq = val
                rs = src_start - cand_offset
                re_ = src_end - cand_offset
                if 0 <= rs <= re_ <= len(cand_seq):
                    sub = cand_seq[rs:re_]
                    return sub if src_strand == "+" else revcomp(sub)

    raise InvariantViolationError(
        f"Source slice {src_path}:{src_start}-{src_end}({src_strand}) out of bounds or missing in GFA"
    )


def _parse_src_interval(spec: str) -> tuple[int, int]:
    s = spec.strip()
    if ":" in s:
        s = s.rsplit(":", 1)[-1]
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    if ".." in s:
        a, b = s.split("..", 1)
        return int(a), int(b)
    raise InvariantViolationError(f"Invalid source interval '{spec}'")


def _reconstruct_gff3_transcripts_and_proteins(
    seqs: Mapping[str, str],
    gff_path: Path,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    exons_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    cds_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )

    for raw in gff_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("##FASTA"):
            break
        if line.startswith("#"):
            continue
        parts = raw.rstrip("\r\n").split("\t")
        if len(parts) != 9:
            raise InvariantViolationError(
                f"Malformed GFF3 row in '{gff_path}': expected 9 columns, got {len(parts)}"
            )
        attrs: dict[str, str] = {}
        for item in parts[8].split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                attrs[k.strip()] = v.strip()
        parent = attrs.get("Parent")
        if not parent:
            continue
        s1, e1 = int(parts[3]), int(parts[4])
        if s1 < 1 or s1 > e1:
            raise InvariantViolationError(
                f"Invalid GFF3 coordinates [{s1}, {e1}] in '{gff_path}'"
            )
        row = {
            "seqid": parts[0],
            "type": parts[2],
            "start": s1,
            "end": e1,
            "strand": parts[6],
            "phase": parts[7],
        }
        parent_ids = [x.strip() for x in parent.split(",") if x.strip()]
        if row["type"] == "exon":
            for p_id in parent_ids:
                exons_by_parent[(row["seqid"], p_id)].append(row)
        elif row["type"] == "CDS":
            for p_id in parent_ids:
                cds_by_parent[(row["seqid"], p_id)].append(row)

    transcripts: dict[tuple[str, str], str] = {}
    for (seqid, parent), ex_list in exons_by_parent.items():
        strand = ex_list[0]["strand"]
        if seqid not in seqs:
            raise InvariantViolationError(
                f"GFF3 seqid '{seqid}' missing from FASTA"
            )
        chrom_seq = seqs[seqid].upper()
        ordered = sorted(
            ex_list, key=lambda r: r["start"], reverse=(strand == "-")
        )
        parts_seq: list[str] = []
        for r in ordered:
            if r["end"] > len(chrom_seq):
                raise InvariantViolationError(
                    f"GFF3 exon end {r['end']} exceeds sequence length {len(chrom_seq)}"
                )
            seg = chrom_seq[r["start"] - 1 : r["end"]]
            parts_seq.append(seg if strand == "+" else revcomp(seg))
        transcripts[(seqid, parent)] = "".join(parts_seq)

    proteins: dict[tuple[str, str], str] = {}
    for (seqid, parent), cds_list in cds_by_parent.items():
        strand = cds_list[0]["strand"]
        if seqid not in seqs:
            raise InvariantViolationError(
                f"GFF3 seqid '{seqid}' missing from FASTA"
            )
        chrom_seq = seqs[seqid].upper()
        ordered = sorted(
            cds_list, key=lambda r: r["start"], reverse=(strand == "-")
        )
        parts_seq = []
        expected_phase: int | None = None
        for r in ordered:
            if r["end"] > len(chrom_seq):
                raise InvariantViolationError(
                    f"GFF3 CDS end {r['end']} exceeds sequence length {len(chrom_seq)}"
                )
            seg = chrom_seq[r["start"] - 1 : r["end"]]
            seg_oriented = seg if strand == "+" else revcomp(seg)
            phase = int(r["phase"]) if r["phase"] != "." else 0
            if expected_phase is not None and phase != expected_phase:
                raise InvariantViolationError(
                    f"CDS phase propagation mismatch on '{parent}' ({seqid}): expected {expected_phase}, got {phase}"
                )
            expected_phase = (3 - ((len(seg_oriented) - phase) % 3)) % 3
            parts_seq.append(seg_oriented)
        first_phase = (
            int(ordered[0]["phase"]) if ordered[0]["phase"] != "." else 0
        )
        proteins[(seqid, parent)] = _translate_cds_seq(
            "".join(parts_seq), phase=first_phase
        )

    return transcripts, proteins


_W_COORD_SUFFIX_RE = re.compile(r":[0-9]+-[0-9]+$", re.ASCII)


def _strip_w_coord_suffix(name: str) -> str:
    return _W_COORD_SUFFIX_RE.sub("", name)


def _match_gfa_paths_by_backbone_prefix(
    metas: Sequence[_GFAPathMeta],
    prefix: str,
) -> list[_GFAPathMeta]:
    """Match GFA `P`/`W` path metadata against `--backbone-prefix` using §2.1 PanSN rules."""
    if not prefix:
        return []
    matched: list[_GFAPathMeta] = []
    if prefix.endswith("#"):
        for m in metas:
            if m.src_path.startswith(prefix) or m.pan_name.startswith(prefix):
                matched.append(m)
        if len(matched) == 0 and (
            prefix.endswith("#0#") or prefix.endswith("#1#")
        ):
            base_sample = prefix[:-3]
            if base_sample and "#" not in base_sample:
                sample_pansn = [
                    m
                    for m in metas
                    if ("#" in m.src_path or m.is_w_line)
                    and m.sample == base_sample
                ]
                hap1 = [m for m in sample_pansn if m.hap == "1"]
                if len(hap1) == 1:
                    matched = hap1
                elif len(hap1) == 0:
                    hap0 = [m for m in sample_pansn if m.hap == "0"]
                    if len(hap0) == 1:
                        matched = hap0
    elif "#" in prefix:
        for m in metas:
            if (
                m.src_path == prefix
                or m.pan_name == prefix
                or m.src_path.startswith(f"{prefix}#")
                or m.pan_name.startswith(f"{prefix}#")
                or m.src_path.startswith(f"{prefix}:")
            ):
                matched.append(m)
    else:
        for m in metas:
            if "#" in m.src_path or m.is_w_line:
                if m.sample == prefix:
                    matched.append(m)
            else:
                if m.src_path.startswith(prefix):
                    matched.append(m)
        if len(matched) > 1:
            exact_matches = [m for m in matched if m.src_path == prefix]
            if len(exact_matches) == 1:
                matched = exact_matches
            else:
                pansn_matches = [
                    m for m in matched if ("#" in m.src_path or m.is_w_line)
                ]
                if len(pansn_matches) == len(matched):
                    hap1 = [m for m in pansn_matches if m.hap == "1"]
                    if len(hap1) == 1:
                        matched = hap1
                    elif len(hap1) == 0:
                        hap0 = [m for m in pansn_matches if m.hap == "0"]
                        if len(hap0) == 1:
                            matched = hap0
    return matched


def _match_fasta_candidates_by_bp(
    target_chrom: str,
    candidates: Mapping[str, str],
    bp: str,
) -> list[tuple[str, str]]:
    """Return candidate `(header, seq)` pairs matching `bp` for `target_chrom` with `#`-boundary and `#0#` fallback."""
    if not bp:
        return []

    def _matches_hdr(hdr: str, pref: str) -> bool:
        if not pref:
            return False
        if "#" in hdr:
            h_sample, _, _, h_seqid = _parse_pansn_meta(hdr)
            h_seqid_base = _strip_w_coord_suffix(h_seqid)
            hdr_base = _strip_w_coord_suffix(hdr)
            pref_base = _strip_w_coord_suffix(pref)
            chrom_ok = (
                h_seqid == target_chrom
                or h_seqid_base == target_chrom
                or hdr.endswith(f"#{target_chrom}")
                or hdr_base.endswith(f"#{target_chrom}")
            )
            if pref.endswith("#"):
                return hdr.startswith(pref) and chrom_ok
            if "#" in pref:
                if pref != pref_base:
                    if hdr != hdr_base:
                        return hdr == pref and chrom_ok
                    return hdr == pref_base and chrom_ok
                return (
                    hdr == pref
                    or hdr_base == pref
                    or hdr == f"{pref}#{target_chrom}"
                    or hdr_base == f"{pref}#{target_chrom}"
                ) and chrom_ok
            return h_sample == pref and chrom_ok
        if pref.endswith("#") or "#" in pref:
            return False
        return hdr == target_chrom and hdr.startswith(pref)

    pref_hits = [
        (h, seq) for h, seq in candidates.items() if _matches_hdr(h, bp)
    ]
    if not pref_hits and (bp.endswith("#0#") or bp.endswith("#1#")):
        base_sample = bp[:-3]
        if base_sample and "#" not in base_sample:
            pref_hits = [
                (h, seq)
                for h, seq in candidates.items()
                if "#" in h and _matches_hdr(h, base_sample)
            ]

    if len(pref_hits) > 1:
        base_s = (
            bp[:-3]
            if (bp.endswith("#0#") or bp.endswith("#1#"))
            else bp.rstrip("#")
        )
        if "#" not in base_s:

            def _is_hap_for_tc(hdr_val: str, want_hap: str) -> bool:
                if "#" not in hdr_val:
                    return False
                s_m, h_m, _, sq_m = _parse_pansn_meta(hdr_val)
                return (
                    s_m == base_s
                    and h_m == want_hap
                    and (
                        sq_m == target_chrom
                        or _strip_w_coord_suffix(sq_m) == target_chrom
                    )
                )

            hap1 = [
                (h, seq) for h, seq in pref_hits if _is_hap_for_tc(h, "1")
            ]
            if len(hap1) == 1:
                return hap1
            if len(hap1) == 0:
                hap0 = [
                    (h, seq) for h, seq in pref_hits if _is_hap_for_tc(h, "0")
                ]
                if len(hap0) == 1:
                    return hap0
    return pref_hits


def _resolve_chrom_by_explicit_bp(
    target_chrom: str,
    candidates: Mapping[str, str],
    bp: str,
    total_fasta_records: int,
    all_target_chroms: set[str],
) -> str | None:
    """Resolve reference sequence for `target_chrom` under an explicit `--backbone-prefix` override."""
    if not bp:
        return None
    hits = _match_fasta_candidates_by_bp(target_chrom, candidates, bp)
    if len(hits) == 1:
        return hits[0][1]
    if len(hits) > 1:
        return None

    # No PanSN header matched `bp`. If `--ref-fa` contains PanSN headers for `target_chrom`, fail strictly.
    has_pansn_for_chrom = any(
        "#" in h
        and h != target_chrom
        and (
            _parse_pansn_meta(h)[3] == target_chrom
            or _strip_w_coord_suffix(_parse_pansn_meta(h)[3]) == target_chrom
            or h.endswith(f"#{target_chrom}")
            or _strip_w_coord_suffix(h).endswith(f"#{target_chrom}")
        )
        for h in candidates
    )
    if has_pansn_for_chrom:
        return None

    # Plain chromosome-named `--ref-fa` (e.g. single-record `>Chr01` or plain multi-chrom `>Chr01`, `>Chr02`):
    # GFA path verification enforces `bp` separately.
    if len(all_target_chroms) == 1 and total_fasta_records == 1:
        return candidates.get(target_chrom)
    if (
        len(all_target_chroms) > 1
        and all("#" not in h for h in candidates)
        and all_target_chroms.issubset(candidates.keys())
    ):
        return candidates.get(target_chrom)
    return None


def _resolve_chrom_from_candidates(
    target_chrom: str,
    candidates: Mapping[str, str],
    bp: str,
    rp: str,
    rh: str,
) -> str | None:
    if rp:
        rp_clean = _strip_w_coord_suffix(rp)
        if rp in candidates:
            return candidates[rp]
        if rp_clean in candidates:
            return candidates[rp_clean]
    if rh and f"{rh}#{target_chrom}" in candidates:
        return candidates[f"{rh}#{target_chrom}"]
    if target_chrom in candidates:
        return candidates[target_chrom]

    if bp:
        pref_hits = _match_fasta_candidates_by_bp(target_chrom, candidates, bp)
        if len(pref_hits) == 1:
            return pref_hits[0][1]

    suf_hits = [
        (h, seq)
        for h, seq in candidates.items()
        if ("#" in target_chrom and target_chrom.endswith(f"#{h}"))
        or (
            "#" in h
            and (
                h.endswith(f"#{target_chrom}")
                or _strip_w_coord_suffix(h).endswith(f"#{target_chrom}")
            )
        )
    ]
    if len(suf_hits) == 1:
        return suf_hits[0][1]
    return None


def _load_ref_records_for_verify(
    ref_fa_file: Path,
    pan_records: Mapping[str, str],
    summary_chrom: str | None,
    backbone_prefix: str | None = None,
    ref_path: str | None = None,
    ref_hap: str | None = None,
    summary_kv: Mapping[str, str] | None = None,
    explicit_backbone: bool = False,
) -> dict[str, str]:
    """
    Load reference FASTA records for `verify_build_outputs`.
    Supports single-chromosome builds (streaming only candidate records for `target_chrom`)
    and multi-chromosome `panflat merge` bundles verified against PanSN multi-FASTAs.
    """
    if not pan_records:
        return parse_fasta(str(ref_fa_file))

    target_chroms = list(pan_records.keys())
    target_set = set(target_chroms)
    skv = summary_kv or {}
    global_bp = (backbone_prefix or "").strip()
    global_rp = "" if explicit_backbone else (ref_path or "").strip()
    global_rh = "" if explicit_backbone else (ref_hap or "").strip()

    if explicit_backbone and not global_bp:
        raise InvariantViolationError(
            "Explicit --backbone-prefix must not be empty"
        )

    wanted_exact: set[str] = set(target_set)
    if explicit_backbone and global_bp and "#" in global_bp and not global_bp.endswith("#"):
        wanted_exact.add(global_bp)
        wanted_exact.add(_strip_w_coord_suffix(global_bp))
    if global_rp:
        wanted_exact.add(global_rp)
        wanted_exact.add(_strip_w_coord_suffix(global_rp))
    if not explicit_backbone:
        for tc in target_chroms:
            c_rp = skv.get(f"{tc}:ref_path", "").strip()
            c_rh = skv.get(f"{tc}:ref_hap", "").strip()
            if c_rp:
                wanted_exact.add(c_rp)
                wanted_exact.add(_strip_w_coord_suffix(c_rp))
            if c_rh:
                wanted_exact.add(f"{c_rh}#{tc}")
            if global_rh:
                wanted_exact.add(f"{global_rh}#{tc}")

    def _is_relevant_hdr(hdr: str) -> bool:
        hdr_base = _strip_w_coord_suffix(hdr)
        if hdr in wanted_exact or hdr_base in wanted_exact:
            return True
        if "#" in hdr:
            _, _, _, h_seqid = _parse_pansn_meta(hdr)
            h_seqid_base = _strip_w_coord_suffix(h_seqid)
            suf = hdr.rsplit("#", 1)[-1]
            suf_base = _strip_w_coord_suffix(suf)
            if (
                h_seqid in target_set
                or h_seqid_base in target_set
                or suf in target_set
                or suf_base in target_set
            ):
                return True
        for tc in target_chroms:
            if "#" in tc and tc.endswith(f"#{hdr}"):
                return True
        return False

    if len(pan_records) == 1:
        # Stream only relevant candidate records so >1GB multi-chromosome FASTAs stay small in RAM,
        # while strictly validating FASTA syntax across all records (kept and skipped).
        candidates: dict[str, str] = {}
        first_hdr: str | None = None
        cur_name: str | None = None
        cur_has_seq = False
        keep = False
        chunks: list[str] = []
        seen_hdrs: set[str] = set()
        saw_any = False
        with ref_fa_file.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                saw_any = True
                if line.startswith(">"):
                    if cur_name is not None:
                        if not cur_has_seq:
                            raise InvariantViolationError(
                                f"Empty FASTA sequence for '{cur_name}' in '{ref_fa_file}'"
                            )
                        if keep:
                            candidates[cur_name] = "".join(chunks)
                    hdr_body = line[1:].strip()
                    cur_name = hdr_body.split()[0] if hdr_body else ""
                    if not cur_name:
                        raise InvariantViolationError(
                            f"Empty FASTA header at line {line_no} in '{ref_fa_file}'"
                        )
                    if cur_name in seen_hdrs:
                        raise InvariantViolationError(
                            f"Duplicate FASTA header '{cur_name}' in '{ref_fa_file}'"
                        )
                    seen_hdrs.add(cur_name)
                    if len(seen_hdrs) == 1:
                        first_hdr = cur_name
                        keep = True
                    else:
                        if (
                            len(seen_hdrs) == 2
                            and first_hdr is not None
                            and not _is_relevant_hdr(first_hdr)
                        ):
                            candidates.pop(first_hdr, None)
                        keep = _is_relevant_hdr(cur_name)
                    cur_has_seq = False
                    chunks = []
                else:
                    if cur_name is None:
                        raise InvariantViolationError(
                            f"FASTA sequence data before header at line {line_no} in '{ref_fa_file}'"
                        )
                    if line.isascii() and line.isalpha():
                        cur_has_seq = True
                        if keep:
                            chunks.append(line.upper())
                    else:
                        chunk = "".join(line.split())
                        if not chunk:
                            continue
                        if not (chunk.isascii() and chunk.isalpha()):
                            raise InvariantViolationError(
                                f"Non-alphabetic characters in FASTA record '{cur_name}' in '{ref_fa_file}'"
                            )
                        cur_has_seq = True
                        if keep:
                            chunks.append(chunk.upper())
        if not saw_any:
            raise InvariantViolationError(f"Empty FASTA file '{ref_fa_file}'")
        if cur_name is not None:
            if not cur_has_seq:
                raise InvariantViolationError(
                    f"Empty FASTA sequence for '{cur_name}' in '{ref_fa_file}'"
                )
            if keep:
                candidates[cur_name] = "".join(chunks)

        tc = target_chroms[0]
        if explicit_backbone:
            resolved_exp = _resolve_chrom_by_explicit_bp(
                tc, candidates, global_bp, len(seen_hdrs), target_set
            )
            if resolved_exp is None:
                raise InvariantViolationError(
                    f"No unique reference FASTA record matching --backbone-prefix '{global_bp}' for chromosome '{tc}'"
                )
            return {tc: resolved_exp}

        resolved = _resolve_chrom_from_candidates(
            tc,
            candidates,
            global_bp or skv.get(f"{tc}:backbone_prefix", "").strip(),
            global_rp or skv.get(f"{tc}:ref_path", "").strip(),
            global_rh or skv.get(f"{tc}:ref_hap", "").strip(),
        )
        if resolved is not None:
            return {tc: resolved}
        return parse_fasta(str(ref_fa_file))

    raw_records = parse_fasta(str(ref_fa_file))
    if explicit_backbone:
        resolved_exp_multi: dict[str, str] = {}
        for tc in target_chroms:
            seq_tc = _resolve_chrom_by_explicit_bp(
                tc, raw_records, global_bp, len(raw_records), target_set
            )
            if seq_tc is None:
                raise InvariantViolationError(
                    f"No unique reference FASTA record matching --backbone-prefix '{global_bp}' for chromosome '{tc}'"
                )
            resolved_exp_multi[tc] = seq_tc
        return resolved_exp_multi

    if set(raw_records.keys()) == target_set:
        return raw_records

    resolved_multi: dict[str, str] = {}
    all_resolved = True
    for tc in target_chroms:
        c_bp = global_bp or skv.get(f"{tc}:backbone_prefix", "").strip()
        c_rp = global_rp if len(target_chroms) == 1 else skv.get(f"{tc}:ref_path", "").strip()
        c_rh = global_rh if len(target_chroms) == 1 else skv.get(f"{tc}:ref_hap", "").strip()
        seq_tc = _resolve_chrom_from_candidates(
            tc, raw_records, c_bp, c_rp, c_rh
        )
        if seq_tc is None:
            all_resolved = False
            break
        resolved_multi[tc] = seq_tc

    if all_resolved and resolved_multi:
        return resolved_multi
    return raw_records


def _sha256_file(path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of `path` (supports Python >= 3.10)."""
    file_digest_fn = getattr(hashlib, "file_digest", None)
    if file_digest_fn is not None:
        with path.open("rb") as fh:
            return file_digest_fn(fh, "sha256").hexdigest().lower()

    hasher = hashlib.sha256()
    buf = bytearray(4 * 1024 * 1024)
    mv = memoryview(buf)
    with path.open("rb") as fh:
        while True:
            n_read = fh.readinto(buf)
            if not n_read:
                break
            hasher.update(mv[:n_read])
    return hasher.hexdigest().lower()


def verify_build_outputs(
    prefix: str | Path,
    gfa_path: str | Path,
    ref_fa_path: str | Path,
    ref_gff_path: str | Path | None = None,
    centromere_bed_path: str | Path | None = None,
    spacer_n: int | None = None,
    raise_on_error: bool = False,
    backbone_prefix: str | None = None,
) -> VerificationReport:
    """
    Verify Global Invariants I, II, III, GFA SHA-256, unresolved provenance,
    centromere 27-col accounting & mount offset delta, and GFF3 liftover integrity (§1, §3.8.5, §5.5).
    """
    p = Path(prefix)
    gfa_file = Path(gfa_path)
    ref_fa_file = Path(ref_fa_path)

    inv1_ok = True
    inv2_ok = True
    inv3_ok = True
    cen_ok = True
    unres_ok = True
    errors: list[str] = []

    checked_ref_steps = 0
    checked_pieces = 0
    checked_chain_blocks = 0
    checked_unresolved = 0

    def _err(category: str, msg: str) -> None:
        nonlocal inv1_ok, inv2_ok, inv3_ok, cen_ok, unres_ok
        errors.append(f"[{category}] {msg}")
        if category == "INV1":
            inv1_ok = False
        elif category == "INV2":
            inv2_ok = False
        elif category == "INV3":
            inv3_ok = False
        elif category == "CEN":
            cen_ok = False
        elif category == "UNRES":
            unres_ok = False
        else:
            inv1_ok = False

    if not gfa_file.is_file():
        _err("INV2", f"GFA file not found: '{gfa_file}'")
    if not ref_fa_file.is_file():
        _err("INV1", f"Reference FASTA file not found: '{ref_fa_file}'")
    if ref_gff_path is not None and not Path(ref_gff_path).is_file():
        _err("INV3", f"Reference GFF3 file not found: '{ref_gff_path}'")
    if centromere_bed_path is not None and not Path(centromere_bed_path).is_file():
        _err("CEN", f"Centromere BED file not found: '{centromere_bed_path}'")

    pan_fa_path = Path(f"{p}.pan.fa")
    offsets_path = Path(f"{p}.offsets.tsv")
    chain_path = Path(f"{p}.ref_to_pan.chain")
    sv_path = Path(f"{p}.sv.tsv")
    genotypes_path = Path(f"{p}.genotypes.tsv")
    inserts_path = Path(f"{p}.inserts.tsv")
    unresolved_path = Path(f"{p}.unresolved.tsv")
    unresolved_fa_path = Path(f"{p}.unresolved.fa")
    centromere_path = Path(f"{p}.centromere.tsv")
    lifted_gff_path = Path(f"{p}.ref.lifted.gff3")
    summary_path = Path(f"{p}.summary.tsv")
    log_path = Path(f"{p}.log")

    for req_path, cat in [
        (pan_fa_path, "INV1"),
        (offsets_path, "INV1"),
        (chain_path, "INV3"),
        (sv_path, "INV2"),
        (genotypes_path, "INV2"),
        (inserts_path, "INV2"),
        (unresolved_path, "UNRES"),
        (unresolved_fa_path, "UNRES"),
        (summary_path, "INV2"),
        (log_path, "INV2"),
    ]:
        if not req_path.is_file():
            _err(cat, f"Required output file missing: '{req_path}'")

    if errors:
        report = VerificationReport(
            passed=False,
            invariant_1_passed=inv1_ok,
            invariant_2_passed=inv2_ok,
            invariant_3_passed=inv3_ok,
            centromere_passed=cen_ok,
            unresolved_passed=unres_ok,
            errors=tuple(errors),
        )
        if raise_on_error:
            raise InvariantViolationError("; ".join(errors))
        return report

    # 0. Parse summary.tsv and verify GFA SHA-256
    summary_kv: dict[str, str] = {}
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            summary_kv[parts[0]] = parts[1]

    actual_sha = _sha256_file(gfa_file)
    recorded_sha = summary_kv.get("gfa_sha256")
    if recorded_sha and recorded_sha.lower() != actual_sha:
        _err(
            "INV2",
            f"GFA SHA-256 mismatch: summary.tsv has {recorded_sha}, actual is {actual_sha}",
        )

    if spacer_n is not None:
        eff_spacer_n = spacer_n
    elif "spacer_n" in summary_kv and summary_kv["spacer_n"].isdigit():
        eff_spacer_n = int(summary_kv["spacer_n"])
    else:
        eff_spacer_n = next(
            (
                int(v)
                for k, v in summary_kv.items()
                if k.endswith(":spacer_n") and v.isdigit()
            ),
            100,
        )

    is_large_offsets = offsets_path.stat().st_size > 2_000_000
    explicit_backbone = backbone_prefix is not None
    gfa_path_metas: list[_GFAPathMeta] = []

    try:
        pan_records = parse_fasta(str(pan_fa_path))
        eff_backbone_prefix = (
            backbone_prefix
            if explicit_backbone
            else summary_kv.get("backbone_prefix")
        )
        ref_records = _load_ref_records_for_verify(
            ref_fa_file,
            pan_records,
            summary_kv.get("chrom"),
            eff_backbone_prefix,
            ref_path=None if explicit_backbone else summary_kv.get("ref_path"),
            ref_hap=None if explicit_backbone else summary_kv.get("ref_hap"),
            summary_kv=summary_kv,
            explicit_backbone=explicit_backbone,
        )
        sv_hdr, sv_rows = _parse_tsv_file(sv_path)
        ins_hdr, ins_rows = _parse_tsv_file(inserts_path)
        unres_hdr, unres_rows = _parse_tsv_file(unresolved_path)

        req_slices: dict[str, set[tuple[int, int]]] = collections.defaultdict(set)
        for r in ins_rows:
            try:
                s_i, e_i = _parse_src_interval(r.get("src_intervals", ""))
                req_slices[r.get("src_path", "")].add((s_i, e_i))
            except Exception:
                pass
        for u in unres_rows:
            try:
                req_slices[u.get("src_path", "")].add(
                    (int(u.get("src_start", "0")), int(u.get("src_end", "0")))
                )
            except Exception:
                pass

        gfa_nodes, path_seqs = _parse_gfa_paths_and_walks(
            gfa_file, req_slices, out_path_metas=gfa_path_metas
        )
        if explicit_backbone:
            bp_clean = (backbone_prefix or "").strip()
            is_multi_chrom_bundle = len(pan_records) > 1 or (
                summary_kv.get("chrom_count", "").isdigit()
                and int(summary_kv["chrom_count"]) > 1
            )
            for tc in pan_records:
                if is_multi_chrom_bundle:
                    rec_rp_tc = summary_kv.get(f"{tc}:ref_path", "").strip()
                    cand_metas = [
                        m
                        for m in gfa_path_metas
                        if m.seqid == tc
                        or _strip_w_coord_suffix(m.seqid) == tc
                        or m.src_path == tc
                        or m.pan_name == tc
                        or (rec_rp_tc and m.src_path == rec_rp_tc)
                    ]
                    matched_m = _match_gfa_paths_by_backbone_prefix(
                        cand_metas, bp_clean
                    )
                else:
                    matched_m = _match_gfa_paths_by_backbone_prefix(
                        gfa_path_metas, bp_clean
                    )
                if len(matched_m) != 1:
                    _err(
                        "INV1",
                        f"Expected 1 GFA reference path matching --backbone-prefix '{bp_clean}' for '{tc}', found {len(matched_m)}",
                    )
                else:
                    m_ref = matched_m[0]
                    rec_rp = summary_kv.get(f"{tc}:ref_path", "").strip() or (
                        summary_kv.get("ref_path", "").strip()
                        if len(pan_records) == 1
                        else ""
                    )
                    rec_rh = summary_kv.get(f"{tc}:ref_hap", "").strip() or (
                        summary_kv.get("ref_hap", "").strip()
                        if len(pan_records) == 1
                        else ""
                    )
                    if rec_rp and m_ref.src_path != rec_rp:
                        _err(
                            "INV1",
                            f"GFA reference path '{m_ref.src_path}' matched by --backbone-prefix '{bp_clean}' does not match recorded ref_path '{rec_rp}' for '{tc}'",
                        )
                    elif rec_rh and m_ref.hap_id != rec_rh:
                        _err(
                            "INV1",
                            f"GFA reference haplotype '{m_ref.hap_id}' matched by --backbone-prefix '{bp_clean}' does not match recorded ref_hap '{rec_rh}' for '{tc}'",
                        )

        if not is_large_offsets:
            offsets_hdr, offsets_rows = _parse_tsv_file(offsets_path)
        else:
            offsets_rows = []
            with offsets_path.open("r", encoding="utf-8") as off_fh:
                first_line = off_fh.readline().rstrip("\r\n")
                offsets_hdr = (
                    first_line.lstrip("#").split("\t") if first_line else []
                )
    except Exception as exc:
        _err("INV1", f"Failed to parse build output or input files: {exc}")
        report = VerificationReport(
            passed=False,
            invariant_1_passed=inv1_ok,
            invariant_2_passed=inv2_ok,
            invariant_3_passed=inv3_ok,
            centromere_passed=cen_ok,
            unresolved_passed=unres_ok,
            errors=tuple(errors),
        )
        if raise_on_error:
            raise InvariantViolationError("; ".join(errors)) from exc
        return report

    if offsets_hdr != OFFSETS_HEADER:
        _err("INV1", f"offsets.tsv header mismatch: {offsets_hdr}")
    if sv_hdr != SV_HEADER:
        _err("INV2", f"sv.tsv header mismatch: {sv_hdr}")
    if ins_hdr != INSERTS_HEADER:
        _err("INV2", f"inserts.tsv header mismatch: {ins_hdr}")
    if unres_hdr != UNRESOLVED_HEADER:
        _err("UNRES", f"unresolved.tsv header mismatch: {unres_hdr}")

    # Compact array representation for chromosome-scale offsets.tsv
    @dataclass
    class _CompactChromOffsets:
        step_idx: np.ndarray
        nids: np.ndarray | list[str]
        r_s: np.ndarray
        r_e: np.ndarray
        p_s: np.ndarray
        p_e: np.ndarray
        slot_ins: np.ndarray

    compact_offsets_by_chrom: dict[str, _CompactChromOffsets] = {}
    offsets_by_chrom: dict[str, list[dict[str, str]]] = collections.defaultdict(
        list
    )

    if is_large_offsets:
        raw_cols_by_chrom: dict[
            str,
            tuple[
                array,
                array,
                list[str],
                array,
                array,
                array,
                array,
                array,
            ],
        ] = {}
        with offsets_path.open("r", encoding="utf-8") as off_fh:
            _ = off_fh.readline()
            for line_no, raw_line in enumerate(off_fh, start=1):
                line = raw_line.rstrip("\r\n")
                if not line or line.startswith("##"):
                    continue
                parts = line.split("\t")
                if len(parts) != 8:
                    _err(
                        "INV1",
                        f"Malformed TSV row in '{offsets_path}' at line {line_no}",
                    )
                    break
                c_name = parts[0]
                if c_name not in raw_cols_by_chrom:
                    raw_cols_by_chrom[c_name] = (
                        array("q"),
                        array("i"),
                        [],
                        array("q"),
                        array("q"),
                        array("q"),
                        array("q"),
                        array("q"),
                    )
                cols = raw_cols_by_chrom[c_name]
                try:
                    cols[0].append(int(parts[1]))
                    if not cols[2] and is_compact_nid_str(parts[2]):
                        cols[1].append(int(parts[2]))
                    else:
                        if not cols[2] and len(cols[1]) > 0:
                            cols[2].extend(str(x) for x in cols[1])
                            del cols[1][:]
                        cols[2].append(parts[2])
                    cols[3].append(int(parts[3]))
                    cols[4].append(int(parts[4]))
                    cols[5].append(int(parts[5]))
                    cols[6].append(int(parts[6]))
                    cols[7].append(int(parts[7]))
                except ValueError as exc:
                    _err(
                        "INV1",
                        f"Non-integer field in offsets.tsv at line {line_no}: {exc}",
                    )
                    break
        for c_name, cols in raw_cols_by_chrom.items():
            nids_val: np.ndarray | list[str] = (
                cols[2]
                if cols[2]
                else np.frombuffer(cols[1], dtype=np.int32).copy()
            )
            compact_offsets_by_chrom[c_name] = _CompactChromOffsets(
                step_idx=np.frombuffer(cols[0], dtype=np.int64).copy(),
                nids=nids_val,
                r_s=np.frombuffer(cols[3], dtype=np.int64).copy(),
                r_e=np.frombuffer(cols[4], dtype=np.int64).copy(),
                p_s=np.frombuffer(cols[5], dtype=np.int64).copy(),
                p_e=np.frombuffer(cols[6], dtype=np.int64).copy(),
                slot_ins=np.frombuffer(cols[7], dtype=np.int64).copy(),
            )
        raw_cols_by_chrom.clear()
    else:
        for r in offsets_rows:
            offsets_by_chrom[r["chrom"]].append(r)

    ins_by_chrom: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in ins_rows:
        ins_by_chrom[r["chrom"]].append(r)

    # 1. Global Invariant I: Reference Base Preservation
    for ref_chrom, ref_seq in ref_records.items():
        ref_up = ref_seq.upper()
        if ref_chrom not in pan_records:
            _err("INV1", f"Reference chromosome '{ref_chrom}' missing in pan.fa")
            continue
        pan_up = pan_records[ref_chrom].upper()

        if is_large_offsets:
            co = compact_offsets_by_chrom.get(ref_chrom)
            if co is None or int(co.step_idx.shape[0]) == 0:
                _err("INV1", f"No offsets.tsv rows for chromosome '{ref_chrom}'")
                continue
            n_st = int(co.step_idx.shape[0])
            expected_idx = np.arange(n_st, dtype=np.int64)
            if not bool(np.array_equal(co.step_idx, expected_idx)):
                bad_i = int(np.flatnonzero(co.step_idx != expected_idx)[0])
                _err("INV1", f"Expected ref_step {bad_i}, got {int(co.step_idx[bad_i])}")
            if co.r_s[0] != 0 or co.slot_ins[0] != 0 or co.p_s[0] != 0:
                _err(
                    "INV1",
                    f"First reference step must have ref_start=0, pan_start=0, slot_ins_len=0; got ({int(co.r_s[0])}, {int(co.p_s[0])}, {int(co.slot_ins[0])})",
                )
            if n_st > 1 and not bool(np.array_equal(co.r_s[1:], co.r_e[:-1])):
                bad_i = int(np.flatnonzero(co.r_s[1:] != co.r_e[:-1])[0]) + 1
                _err(
                    "INV1",
                    f"Ref coordinate gap/overlap at step {bad_i}: expected {int(co.r_e[bad_i - 1])}, got {int(co.r_s[bad_i])}",
                )
            if bool(np.any(co.r_e <= co.r_s)):
                bad_i = int(np.flatnonzero(co.r_e <= co.r_s)[0])
                _err("INV1", f"Invalid ref step range at step {bad_i}")
            if not bool(np.array_equal(co.p_e - co.p_s, co.r_e - co.r_s)):
                bad_i = int(np.flatnonzero((co.p_e - co.p_s) != (co.r_e - co.r_s))[0])
                _err("INV1", f"Step length mismatch at step {bad_i}")
            if bool(np.any(co.slot_ins < 0)) or (
                n_st > 1
                and not bool(
                    np.array_equal(co.p_s[1:], co.p_e[:-1] + co.slot_ins[1:])
                )
            ):
                _err("INV1", "pan_start mismatch in offsets.tsv")
            if int(co.p_s[0]) < 0 or int(co.p_e[-1]) > len(pan_up):
                _err("INV1", f"Pan step coordinates out of bounds for pan_len={len(pan_up)}")
            else:
                # Check contiguous blocks between insertion slots in O(blocks) + every node against gfa_nodes
                slot_breaks = (
                    np.flatnonzero(co.slot_ins[1:] > 0) + 1
                ).tolist()
                blk_starts = [0, *slot_breaks]
                blk_ends = [*slot_breaks, n_st]
                for b_s, b_e in zip(blk_starts, blk_ends):
                    rs_blk = int(co.r_s[b_s])
                    re_blk = int(co.r_e[b_e - 1])
                    ps_blk = int(co.p_s[b_s])
                    pe_blk = int(co.p_e[b_e - 1])
                    if pan_up[ps_blk:pe_blk] != ref_up[rs_blk:re_blk]:
                        _err(
                            "INV1",
                            f"Reference base mismatch in block steps [{b_s}, {b_e})",
                        )
                        break
                if isinstance(gfa_nodes, CompactNodeStore) and isinstance(
                    co.nids, np.ndarray
                ):
                    nids_int = co.nids
                    nl_arr = gfa_nodes._node_lens
                    if bool(
                        np.any(
                            (nids_int <= 0)
                            | (nids_int >= int(nl_arr.shape[0]))
                            | (nl_arr[np.clip(nids_int, 0, int(nl_arr.shape[0]) - 1)] == 0)
                        )
                    ):
                        _err("INV1", "Reference step node missing from GFA")
                    elif not bool(
                        np.array_equal(nl_arr[nids_int], co.r_e - co.r_s)
                    ):
                        _err(
                            "INV1",
                            "GFA node length mismatch with ref step range",
                        )
                    else:
                        seq_bytes = gfa_nodes._seq_bytes
                        off_starts = gfa_nodes._offsets[nids_int].tolist()
                        off_ends = gfa_nodes._offsets[nids_int + 1].tolist()
                        ref_bytes = ref_up.encode("ascii")
                        for nid_i, os_i, oe_i, rs_i, re_i in zip(
                            nids_int.tolist(),
                            off_starts,
                            off_ends,
                            co.r_s.tolist(),
                            co.r_e.tolist(),
                        ):
                            nb = seq_bytes[os_i:oe_i]
                            rb = ref_bytes[rs_i:re_i]
                            if (
                                nb != rb
                                and nb.translate(_REVCOMP_BYTES_TABLE)[::-1] != rb
                            ):
                                _err(
                                    "INV1",
                                    f"GFA node '{nid_i}' sequence does not match ref[{rs_i}:{re_i}]",
                                )
                                break
                else:
                    for idx_n, nid in enumerate(co.nids):
                        nid_s = str(nid)
                        if nid_s not in gfa_nodes:
                            _err(
                                "INV1",
                                f"Reference step node '{nid_s}' missing from GFA",
                            )
                            break
                        rs_i = int(co.r_s[idx_n])
                        re_i = int(co.r_e[idx_n])
                        node_seq = gfa_nodes[nid_s]
                        if len(node_seq) != re_i - rs_i:
                            _err(
                                "INV1",
                                "GFA node length mismatch with ref step range",
                            )
                            break
                        ref_slice = ref_up[rs_i:re_i]
                        if node_seq != ref_slice and revcomp(node_seq) != ref_slice:
                            _err(
                                "INV1",
                                f"GFA node '{nid_s}' sequence does not match ref[{rs_i}:{re_i}]",
                            )
                            break
            if int(co.r_e[-1]) != len(ref_up):
                _err(
                    "INV1",
                    f"offsets.tsv ends at ref {int(co.r_e[-1])}, expected {len(ref_up)}",
                )
            last_pe_fast = int(co.p_e[-1])
            if last_pe_fast != len(pan_up):
                has_trailing_ins = any(
                    int(r.get("pan_start", "-1")) >= last_pe_fast
                    for r in ins_by_chrom.get(ref_chrom, ())
                    if str(r.get("pan_start", "")).isdigit()
                )
                if last_pe_fast > len(pan_up) or not has_trailing_ins:
                    _err(
                        "INV1",
                        f"offsets.tsv ends at pan {last_pe_fast}, expected pan_len={len(pan_up)}",
                    )
            checked_ref_steps += n_st
            continue

        chrom_offsets = offsets_by_chrom.get(ref_chrom, [])
        if not chrom_offsets:
            _err("INV1", f"No offsets.tsv rows for chromosome '{ref_chrom}'")
            continue

        expected_ref_pos = 0
        prev_pan_end = 0
        for idx, row in enumerate(chrom_offsets):
            try:
                step_idx = int(row["ref_step"])
                r_s = int(row["ref_start"])
                r_e = int(row["ref_end"])
                p_s = int(row["pan_start"])
                p_e = int(row["pan_end"])
                slot_ins = int(row["slot_ins_len"])
            except ValueError as exc:
                _err("INV1", f"Non-integer field in offsets.tsv at row {idx}: {exc}")
                break

            if step_idx != idx:
                _err("INV1", f"Expected ref_step {idx}, got {step_idx}")
            if idx == 0 and (r_s != 0 or slot_ins != 0 or p_s != 0):
                _err(
                    "INV1",
                    f"First reference step must have ref_start=0, pan_start=0, slot_ins_len=0; got ({r_s}, {p_s}, {slot_ins})",
                )
            if r_s != expected_ref_pos:
                _err(
                    "INV1",
                    f"Ref coordinate gap/overlap at step {idx}: expected {expected_ref_pos}, got {r_s}",
                )
            if r_e <= r_s:
                _err("INV1", f"Invalid ref step range [{r_s}, {r_e}) at step {idx}")
            if p_e - p_s != r_e - r_s:
                _err(
                    "INV1",
                    f"Step length mismatch at step {idx}: pan {p_e - p_s} vs ref {r_e - r_s}",
                )
            if slot_ins < 0 or p_s != prev_pan_end + slot_ins:
                _err(
                    "INV1",
                    f"pan_start mismatch at step {idx}: {p_s} != {prev_pan_end} + {slot_ins}",
                )
            if p_s < 0 or p_e > len(pan_up):
                _err(
                    "INV1",
                    f"Pan step coordinates [{p_s}, {p_e}) out of bounds for pan_len={len(pan_up)}",
                )
            else:
                if pan_up[p_s:p_e] != ref_up[r_s:r_e]:
                    _err(
                        "INV1",
                        f"Reference base mismatch at step {idx} ({row['node_id']}): pan[{p_s}:{p_e}] != ref[{r_s}:{r_e}]",
                    )
            nid = row["node_id"]
            if nid not in gfa_nodes:
                _err("INV1", f"Reference step node '{nid}' missing from GFA")
            else:
                node_seq = gfa_nodes[nid]
                ref_slice = ref_up[r_s:r_e]
                if node_seq != ref_slice and revcomp(node_seq) != ref_slice:
                    _err(
                        "INV1",
                        f"GFA node '{nid}' sequence does not match ref[{r_s}:{r_e}]",
                    )

            expected_ref_pos = r_e
            prev_pan_end = p_e
            checked_ref_steps += 1

        if expected_ref_pos != len(ref_up):
            _err(
                "INV1",
                f"offsets.tsv ends at ref {expected_ref_pos}, expected {len(ref_up)}",
            )
        if prev_pan_end != len(pan_up):
            has_trailing_ins_s = any(
                int(r.get("pan_start", "-1")) >= prev_pan_end
                for r in ins_by_chrom.get(ref_chrom, ())
                if str(r.get("pan_start", "")).isdigit()
            )
            if prev_pan_end > len(pan_up) or not has_trailing_ins_s:
                _err(
                    "INV1",
                    f"offsets.tsv ends at pan {prev_pan_end}, expected pan_len={len(pan_up)}",
                )

    # 2. Global Invariant II: Piece Provenance, Spacers & Full Partition

    for ref_chrom, pan_seq in pan_records.items():
        pan_up = pan_seq.upper()
        n_pan = len(pan_up)
        covered_arr = np.zeros(n_pan, dtype=np.uint8)

        if is_large_offsets:
            co = compact_offsets_by_chrom.get(ref_chrom)
            if co is not None and int(co.step_idx.shape[0]) > 0:
                n_st = int(co.step_idx.shape[0])
                slot_breaks = (
                    np.flatnonzero(co.slot_ins[1:] > 0) + 1
                ).tolist()
                blk_starts = [0, *slot_breaks]
                blk_ends = [*slot_breaks, n_st]
                for b_s, b_e in zip(blk_starts, blk_ends):
                    ps_blk = int(co.p_s[b_s])
                    pe_blk = int(co.p_e[b_e - 1])
                    if 0 <= ps_blk <= pe_blk <= n_pan:
                        if bool(np.any(covered_arr[ps_blk:pe_blk] != 0)):
                            _err(
                                "INV1",
                                f"Overlapping reference step in [{ps_blk}, {pe_blk}) on {ref_chrom}",
                            )
                            break
                        covered_arr[ps_blk:pe_blk] = 1
        else:
            for srow in offsets_by_chrom.get(ref_chrom, ()):
                p_s = int(srow["pan_start"])
                p_e = int(srow["pan_end"])
                if 0 <= p_s <= p_e <= n_pan:
                    if bool(np.any(covered_arr[p_s:p_e] != 0)):
                        _err(
                            "INV1",
                            f"Overlapping reference step at pan [{p_s}, {p_e}) on {ref_chrom}",
                        )
                        break
                    covered_arr[p_s:p_e] = 1

        prev_p_e: int | None = None
        prev_ev_id: str | None = None
        prev_allele: str | None = None

        for row in ins_by_chrom.get(ref_chrom, ()):
            try:
                p_s = int(row["pan_start"])
                p_e = int(row["pan_end"])
                src_s, src_e = _parse_src_interval(row["src_intervals"])
            except Exception as exc:
                _err("INV2", f"Invalid coordinates in inserts.tsv: {exc}")
                continue

            if not (0 <= p_s < p_e <= n_pan):
                _err(
                    "INV2",
                    f"Insert piece [{p_s}, {p_e}) out of bounds for pan_len={n_pan}",
                )
                continue

            src_strand = row["src_strand"]
            if src_strand not in ("+", "-"):
                _err("INV2", f"Invalid src_strand '{src_strand}' in inserts.tsv")
                continue

            join_before = row["join_before"]
            if join_before not in ("source_contiguous", "spacer", "synthetic"):
                _err("INV2", f"Invalid join_before '{join_before}' in inserts.tsv")
                continue

            if src_e - src_s != p_e - p_s:
                _err(
                    "INV2",
                    f"Piece length mismatch for {row['event_id']} piece {row['piece']}: pan={p_e - p_s} vs src={src_e - src_s}",
                )
            else:
                try:
                    expected_piece_seq = _extract_source_slice(
                        path_seqs, row["src_path"], src_s, src_e, src_strand
                    ).upper()
                    if pan_up[p_s:p_e] != expected_piece_seq:
                        _err(
                            "INV2",
                            f"Piece sequence mismatch for {row['event_id']} piece {row['piece']} at pan[{p_s}:{p_e}]",
                        )
                except Exception as exc:
                    _err("INV2", str(exc))

            if join_before == "spacer":
                if eff_spacer_n > 0:
                    if p_s < eff_spacer_n:
                        _err(
                            "INV2",
                            f"Piece with join_before='spacer' at pan_start={p_s} < spacer_n={eff_spacer_n}",
                        )
                    elif pan_up[p_s - eff_spacer_n : p_s] != "N" * eff_spacer_n:
                        _err(
                            "INV2",
                            f"Expected {eff_spacer_n} 'N' spacer before pan_start={p_s}",
                        )
            elif (
                prev_p_e is not None
                and prev_ev_id == row["event_id"]
                and prev_allele == row["allele"]
            ):
                if p_s != prev_p_e:
                    _err(
                        "INV2",
                        f"Non-spacer piece gap/overlap in {row['event_id']}: expected pan_start={prev_p_e}, got {p_s}",
                    )

            if prev_p_e is not None and p_s < prev_p_e:
                _err(
                    "INV2",
                    f"Overlapping insert pieces in inserts.tsv: {p_s} < {prev_p_e}",
                )

            if bool(np.any(covered_arr[p_s:p_e] != 0)):
                _err(
                    "INV2",
                    f"Overlap between insert piece and existing interval at pan[{p_s}:{p_e}]",
                )
            covered_arr[p_s:p_e] = 2

            prev_p_e = p_e
            prev_ev_id = row["event_id"]
            prev_allele = row["allele"]
            checked_pieces += 1

        # Partition check: uncovered bases must be 'N' in exact multiples of eff_spacer_n
        zero_indices = np.flatnonzero(covered_arr == 0)
        if int(zero_indices.shape[0]) > 0:
            gaps = np.flatnonzero(np.diff(zero_indices) > 1)
            run_starts = np.empty(int(gaps.shape[0]) + 1, dtype=np.int64)
            run_ends = np.empty(int(gaps.shape[0]) + 1, dtype=np.int64)
            run_starts[0] = zero_indices[0]
            if int(gaps.shape[0]) > 0:
                run_starts[1:] = zero_indices[gaps + 1]
                run_ends[:-1] = zero_indices[gaps] + 1
            run_ends[-1] = zero_indices[-1] + 1
            for r_s_i, r_e_i in zip(run_starts.tolist(), run_ends.tolist()):
                run_len = r_e_i - r_s_i
                if pan_up[r_s_i:r_e_i] != "N" * run_len:
                    _err(
                        "INV2",
                        f"Uncovered non-spacer base in pan[{r_s_i}:{r_e_i}] on {ref_chrom}",
                    )
                if eff_spacer_n > 0 and (run_len % eff_spacer_n) != 0:
                    _err(
                        "INV2",
                        f"Spacer run [{r_s_i}, {r_e_i}) of length {run_len} is not a multiple of spacer_n={eff_spacer_n}",
                    )

    # 3. Global Invariant III: Chain, Offsets & Coordinate Consistency
    chain_text = chain_path.read_text(encoding="utf-8")
    chain_headers: list[dict[str, Any]] = []
    chain_blocks_list: list[list[tuple[int, int, int]]] = []
    chain_comment_blocks_list: list[list[tuple[int, int, int]]] = []
    chain_comment_steps_list: list[list[tuple[int, int, int, int, int, int]]] = []

    cur_blocks: list[tuple[int, int, int]] | None = None
    cur_c_blocks: list[tuple[int, int, int]] | None = None
    cur_c_steps: list[tuple[int, int, int, int, int, int]] | None = None
    for raw_line in chain_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            c_parts = line.split("\t")
            if c_parts[0] == "# block" and len(c_parts) >= 4:
                if cur_c_blocks is None:
                    _err(
                        "INV3",
                        f"Orphan # block comment before chain header: {line}",
                    )
                else:
                    try:
                        cur_c_blocks.append(
                            (int(c_parts[1]), int(c_parts[2]), int(c_parts[3]))
                        )
                    except ValueError:
                        _err("INV3", f"Malformed # block comment in chain: {line}")
            elif c_parts[0] == "# step" and len(c_parts) >= 7:
                if cur_c_steps is None:
                    _err(
                        "INV3",
                        f"Orphan # step comment before chain header: {line}",
                    )
                else:
                    try:
                        cur_c_steps.append(
                            (
                                int(c_parts[1]),
                                int(c_parts[2]),
                                int(c_parts[3]),
                                int(c_parts[4]),
                                int(c_parts[5]),
                                int(c_parts[6]),
                            )
                        )
                    except ValueError:
                        _err("INV3", f"Malformed # step comment in chain: {line}")
            continue

        parts = line.split()
        if parts[0] == "chain":
            if len(parts) < 12:
                _err("INV3", f"Malformed chain header: {line}")
                continue
            if cur_blocks is not None:
                chain_blocks_list.append(cur_blocks)
                chain_comment_blocks_list.append(cur_c_blocks or [])
                chain_comment_steps_list.append(cur_c_steps or [])
            try:
                chain_headers.append(
                    {
                        "score": int(parts[1]),
                        "tName": parts[2],
                        "tSize": int(parts[3]),
                        "tStrand": parts[4],
                        "tStart": int(parts[5]),
                        "tEnd": int(parts[6]),
                        "qName": parts[7],
                        "qSize": int(parts[8]),
                        "qStrand": parts[9],
                        "qStart": int(parts[10]),
                        "qEnd": int(parts[11]),
                        "id": parts[12] if len(parts) > 12 else "1",
                    }
                )
            except ValueError as exc:
                _err("INV3", f"Invalid integer in chain header: {exc}")
                cur_blocks = None
                cur_c_blocks = None
                cur_c_steps = None
                continue
            cur_blocks = []
            cur_c_blocks = []
            cur_c_steps = []
        elif cur_blocks is not None:
            try:
                if len(parts) == 3:
                    cur_blocks.append(
                        (int(parts[0]), int(parts[1]), int(parts[2]))
                    )
                elif len(parts) == 1:
                    cur_blocks.append((int(parts[0]), 0, 0))
                else:
                    _err("INV3", f"Malformed chain block line: {line}")
            except ValueError as exc:
                _err("INV3", f"Invalid integer in chain block line: {exc}")

    if cur_blocks is not None:
        chain_blocks_list.append(cur_blocks)
        chain_comment_blocks_list.append(cur_c_blocks or [])
        chain_comment_steps_list.append(cur_c_steps or [])

    if not chain_headers:
        _err("INV3", "ref_to_pan.chain contains no chain headers")

    chain_by_chrom: dict[
        str,
        tuple[
            dict[str, Any],
            list[tuple[int, int, int]],
            list[tuple[int, int, int]],
            list[tuple[int, int, int, int, int, int]],
        ],
    ] = {}
    for ch, blks, c_blks, c_steps in zip(
        chain_headers,
        chain_blocks_list,
        chain_comment_blocks_list,
        chain_comment_steps_list,
    ):
        t_name = ch["tName"]
        if t_name in chain_by_chrom:
            _err("INV3", f"Duplicate chain header for chromosome '{t_name}'")
        else:
            chain_by_chrom[t_name] = (ch, blks, c_blks, c_steps)

    for ref_chrom in ref_records:
        if ref_chrom not in chain_by_chrom or ref_chrom not in pan_records:
            _err("INV3", f"Missing chain for chromosome '{ref_chrom}'")

    for ref_chrom, pan_seq_raw in pan_records.items():
        if ref_chrom not in chain_by_chrom or ref_chrom not in ref_records:
            if ref_chrom not in chain_by_chrom:
                _err("INV3", f"Missing chain for chromosome '{ref_chrom}'")
            continue
        ch, blocks, c_blks, c_steps = chain_by_chrom[ref_chrom]
        ref_up = ref_records[ref_chrom].upper()
        pan_up = pan_seq_raw.upper()

        if ch["tName"] != ref_chrom or ch["qName"] != ref_chrom:
            _err(
                "INV3",
                f"Chain tName/qName ({ch['tName']}/{ch['qName']}) != {ref_chrom}",
            )
        if ch["tSize"] != len(ref_up) or ch["tStart"] != 0 or ch["tEnd"] != len(ref_up):
            _err(
                "INV3",
                f"Chain target bounds ({ch['tStart']}, {ch['tEnd']}, {ch['tSize']}) != (0, {len(ref_up)}, {len(ref_up)})",
            )
        if is_large_offsets:
            co = compact_offsets_by_chrom.get(ref_chrom)
            chrom_offsets = []
            offset_by_ref_start = {}
            expected_q_end = (
                int(co.p_e[-1])
                if co is not None and int(co.p_e.shape[0]) > 0
                else len(pan_up)
            )
        else:
            co = None
            chrom_offsets = offsets_by_chrom.get(ref_chrom, [])
            offset_by_ref_start = {
                int(r["ref_start"]): r for r in chrom_offsets if "ref_start" in r
            }
            expected_q_end = (
                int(chrom_offsets[-1]["pan_end"])
                if chrom_offsets and str(chrom_offsets[-1].get("pan_end", "")).isdigit()
                else len(pan_up)
            )

        if (
            ch["qSize"] != len(pan_up)
            or ch["qStart"] != 0
            or ch["qEnd"] != expected_q_end
        ):
            _err(
                "INV3",
                f"Chain query bounds ({ch['qStart']}, {ch['qEnd']}, {ch['qSize']}) != (0, {expected_q_end}, {len(pan_up)})",
            )

        t_cur = ch["tStart"]
        q_cur = ch["qStart"]
        seen_dq_boundaries: dict[int, int] = {}
        blk_t_starts: list[int] = []
        blk_t_ends: list[int] = []
        blk_q_starts: list[int] = []
        computed_blocks_for_chrom: list[tuple[int, int, int]] = []

        for b_idx, (size, dt, dq) in enumerate(blocks):
            if size <= 0:
                _err("INV3", f"Chain block {b_idx} has size={size} <= 0")
                break
            if dt != 0:
                _err("INV3", f"Chain block {b_idx} has non-zero dt={dt}")
            if dq < 0:
                _err("INV3", f"Chain block {b_idx} has negative dq={dq}")
            if t_cur + size > len(ref_up) or q_cur + size > len(pan_up):
                _err(
                    "INV3",
                    f"Chain block {b_idx} exceeds reference or pan sequence bounds",
                )
                break
            if pan_up[q_cur : q_cur + size] != ref_up[t_cur : t_cur + size]:
                _err(
                    "INV3",
                    f"Chain block {b_idx} sequence mismatch at ref[{t_cur}:{t_cur + size}] vs pan[{q_cur}:{q_cur + size}]",
                )

            computed_blocks_for_chrom.append((t_cur, t_cur + size, dq))
            blk_t_starts.append(t_cur)
            blk_t_ends.append(t_cur + size)
            blk_q_starts.append(q_cur)

            boundary_ref = t_cur + size
            if boundary_ref < len(ref_up):
                if co is not None:
                    pos_b = int(np.searchsorted(co.r_s, boundary_ref, side="left"))
                    if pos_b >= int(co.r_s.shape[0]) or int(co.r_s[pos_b]) != boundary_ref:
                        _err(
                            "INV3",
                            f"Chain block boundary at ref={boundary_ref} does not match any step ref_start in offsets.tsv",
                        )
                    else:
                        s_ins = int(co.slot_ins[pos_b])
                        if dq != s_ins or dq <= 0:
                            _err(
                                "INV3",
                                f"Chain dq={dq} at ref={boundary_ref} disagrees with slot_ins_len={s_ins}",
                            )
                else:
                    if boundary_ref not in offset_by_ref_start:
                        _err(
                            "INV3",
                            f"Chain block boundary at ref={boundary_ref} does not match any step ref_start in offsets.tsv",
                        )
                    else:
                        s_ins = int(offset_by_ref_start[boundary_ref]["slot_ins_len"])
                        if dq != s_ins or dq <= 0:
                            _err(
                                "INV3",
                                f"Chain dq={dq} at ref={boundary_ref} disagrees with slot_ins_len={s_ins}",
                            )
                seen_dq_boundaries[boundary_ref] = dq
            else:
                if dq != 0:
                    _err("INV3", f"Final chain block has non-zero dq={dq}")

            t_cur += size + dt
            q_cur += size + dq
            checked_chain_blocks += 1

        if t_cur != ch["tEnd"]:
            _err("INV3", f"Chain t_cur={t_cur} != tEnd={ch['tEnd']}")
        if q_cur != ch["qEnd"]:
            _err("INV3", f"Chain q_cur={q_cur} != qEnd={ch['qEnd']}")

        blk_ts_arr = np.asarray(blk_t_starts, dtype=np.int64)
        blk_te_arr = np.asarray(blk_t_ends, dtype=np.int64)
        blk_qs_arr = np.asarray(blk_q_starts, dtype=np.int64)

        def _map_ref_pos_via_chain(r_pos: int) -> int | None:
            if int(blk_te_arr.shape[0]) == 0:
                return None
            idx_b = int(np.searchsorted(blk_te_arr, r_pos, side="right"))
            if idx_b < int(blk_te_arr.shape[0]) and int(blk_ts_arr[idx_b]) <= r_pos < int(blk_te_arr[idx_b]):
                return int(blk_qs_arr[idx_b]) + (r_pos - int(blk_ts_arr[idx_b]))
            return None

        if co is not None:
            pos_slots = np.flatnonzero(co.slot_ins > 0).tolist()
            for s_i in pos_slots:
                r_s_val = int(co.r_s[s_i])
                s_ins_val = int(co.slot_ins[s_i])
                if seen_dq_boundaries.get(r_s_val) != s_ins_val:
                    _err(
                        "INV3",
                        f"Step at ref_start={r_s_val} has slot_ins_len={s_ins_val} missing or mismatched in chain blocks",
                    )
            if int(blk_te_arr.shape[0]) > 0 and t_cur == len(ref_up):
                idx_s = np.clip(
                    np.searchsorted(blk_te_arr, co.r_s, side="right"),
                    0,
                    int(blk_te_arr.shape[0]) - 1,
                )
                mapped_ps = blk_qs_arr[idx_s] + (co.r_s - blk_ts_arr[idx_s])
                idx_e = np.clip(
                    np.searchsorted(blk_te_arr, co.r_e - 1, side="right"),
                    0,
                    int(blk_te_arr.shape[0]) - 1,
                )
                mapped_pe = blk_qs_arr[idx_e] + ((co.r_e - 1) - blk_ts_arr[idx_e])
                if not (
                    bool(np.array_equal(mapped_ps, co.p_s))
                    and bool(np.array_equal(mapped_pe, co.p_e - 1))
                ):
                    _err("INV3", "Coordinate mismatch between chain and offsets")
        else:
            for srow in chrom_offsets:
                r_s = int(srow["ref_start"])
                r_e = int(srow["ref_end"])
                p_s = int(srow["pan_start"])
                p_e = int(srow["pan_end"])
                s_ins = int(srow["slot_ins_len"])
                if s_ins > 0 and seen_dq_boundaries.get(r_s) != s_ins:
                    _err(
                        "INV3",
                        f"Step at ref_start={r_s} has slot_ins_len={s_ins} missing or mismatched in chain blocks",
                    )
                if (
                    _map_ref_pos_via_chain(r_s) != p_s
                    or _map_ref_pos_via_chain(r_e - 1) != p_e - 1
                ):
                    _err(
                        "INV3",
                        f"Coordinate mismatch between chain and offsets at step ref[{r_s}:{r_e})",
                    )

        if t_cur == len(ref_up) and len(ref_up) > 0:
            rng = random.Random(12345)
            sample_positions = [0, len(ref_up) - 1] + [
                rng.randrange(len(ref_up))
                for _ in range(min(25, len(ref_up)))
            ]
            for r_pos in sample_positions:
                p_chain = _map_ref_pos_via_chain(r_pos)
                if p_chain is None or pan_up[p_chain] != ref_up[r_pos]:
                    _err(
                        "INV3",
                        f"Spot-check base mismatch at ref={r_pos}: pan[{p_chain}] != ref[{r_pos}]",
                    )

        if c_blks:
            if c_blks != computed_blocks_for_chrom:
                _err(
                    "INV3",
                    f"Chain # block footer metadata does not match chain blocks for '{ref_chrom}': {c_blks} != {computed_blocks_for_chrom}",
                )

        if c_steps:
            if co is not None:
                expected_steps = list(
                    zip(
                        co.step_idx.tolist(),
                        co.r_s.tolist(),
                        co.r_e.tolist(),
                        co.p_s.tolist(),
                        co.p_e.tolist(),
                        co.slot_ins.tolist(),
                    )
                )
            else:
                expected_steps = [
                    (
                        int(r["ref_step"]),
                        int(r["ref_start"]),
                        int(r["ref_end"]),
                        int(r["pan_start"]),
                        int(r["pan_end"]),
                        int(r["slot_ins_len"]),
                    )
                    for r in chrom_offsets
                ]
            if c_steps != expected_steps:
                _err(
                    "INV3",
                    f"Chain # step footer metadata does not match offsets.tsv for '{ref_chrom}'",
                )

    # 4. Unresolved Provenance & FASTA Check
    unres_fa: dict[str, str] = {}
    unres_hdr_kv: dict[str, dict[str, str]] = {}
    try:
        unres_fa, unres_hdr_kv = _parse_fasta_with_header_kv(unresolved_fa_path)
    except Exception as exc:
        _err("UNRES", f"Malformed unresolved.fa '{unresolved_fa_path}': {exc}")

    referenced_exp_ids: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for urow in unres_rows:
        try:
            u_s = int(urow["src_start"])
            u_e = int(urow["src_end"])
            u_len = int(urow["length"])
        except ValueError as exc:
            _err("UNRES", f"Non-integer in unresolved.tsv: {exc}")
            continue

        if u_len <= 0 or u_e - u_s != u_len:
            _err(
                "UNRES",
                f"Unresolved length mismatch for {urow['record_id']}: {u_e} - {u_s} != {u_len}",
            )
            continue

        if urow["reason"] not in UNRESOLVED_REASONS:
            _err(
                "UNRES",
                f"Unknown unresolved reason '{urow['reason']}' for {urow['record_id']}",
            )

        try:
            u_seq = _extract_source_slice(
                path_seqs, urow["src_path"], u_s, u_e, urow["src_strand"]
            ).upper()
        except Exception as exc:
            _err("UNRES", str(exc))
            continue

        exp_id = urow["exported_fasta_id"]
        if exp_id != ".":
            referenced_exp_ids[exp_id].append(urow)
            if exp_id not in unres_fa:
                _err(
                    "UNRES",
                    f"Exported FASTA ID '{exp_id}' missing from unresolved.fa",
                )
            else:
                if unres_fa[exp_id].upper() != u_seq:
                    _err(
                        "UNRES",
                        f"Unresolved FASTA sequence mismatch for '{exp_id}'",
                    )
                hdr_kv = unres_hdr_kv.get(exp_id, {})
                exp_src = f"{urow['src_path']}:{u_s}-{u_e}"
                exp_strand = urow["src_strand"]
                if hdr_kv.get("src") != exp_src:
                    _err(
                        "UNRES",
                        f"Unresolved FASTA header src mismatch for '{exp_id}': "
                        f"got '{hdr_kv.get('src')}', expected '{exp_src}'",
                    )
                if hdr_kv.get("strand") != exp_strand:
                    _err(
                        "UNRES",
                        f"Unresolved FASTA header strand mismatch for '{exp_id}': "
                        f"got '{hdr_kv.get('strand')}', expected '{exp_strand}'",
                    )
        checked_unresolved += 1

    for exp_id, urows_for_id in referenced_exp_ids.items():
        if exp_id in unres_hdr_kv:
            hdr_reason = unres_hdr_kv[exp_id].get("reason")
            valid_reasons = {r["reason"] for r in urows_for_id}
            if hdr_reason not in valid_reasons:
                _err(
                    "UNRES",
                    f"Unresolved FASTA header reason mismatch for '{exp_id}': "
                    f"got '{hdr_reason}', expected one of {sorted(valid_reasons)}",
                )

    for fa_id in unres_fa:
        if fa_id not in referenced_exp_ids:
            _err(
                "UNRES",
                f"Orphan FASTA record '{fa_id}' in unresolved.fa not referenced by unresolved.tsv",
            )

    # 5. Centromere Accounting & Local Offset Delta Check
    cen_expected = centromere_bed_path is not None or any(
        v not in ("off", "")
        for k, v in summary_kv.items()
        if k == "centromere_mode" or k.endswith(":centromere_mode")
    )
    if cen_expected and not centromere_path.is_file():
        _err(
            "CEN",
            f"Centromere processing active, but '{centromere_path}' is missing",
        )
    if centromere_path.is_file():
        cen_hdr, cen_rows = _parse_tsv_file(centromere_path)
        if cen_hdr != list(CENTROMERE_TSV_COLUMNS):
            _err(
                "CEN",
                f"centromere.tsv header mismatch: expected 27 columns, got {cen_hdr}",
            )
        else:
            regions_seen: dict[
                tuple[str, str], tuple[int, int, str, str, str]
            ] = {}
            accepted_bp_by_reg: dict[tuple[str, str], int] = (
                collections.defaultdict(int)
            )
            supported_evs_by_reg: dict[tuple[str, str], set[str]] = (
                collections.defaultdict(set)
            )

            for crow in cen_rows:
                chrom = crow["chrom"]
                rid = crow["region_id"]
                try:
                    bed_s = int(crow["bed_start"])
                    bed_e = int(crow["bed_end"])
                    r_s = int(crow["ref_start"])
                    r_e = int(crow["ref_end"])
                    l_ref = int(crow["L_ref"])
                except ValueError as exc:
                    _err("CEN", f"Invalid integer in centromere.tsv: {exc}")
                    continue

                if not (0 <= r_s <= bed_s < bed_e <= r_e):
                    _err(
                        "CEN",
                        f"Invalid centromere coordinates ref[{r_s},{r_e}) vs bed[{bed_s},{bed_e}) in {rid}",
                    )
                if l_ref != r_e - r_s:
                    _err(
                        "CEN",
                        f"Centromere L_ref={l_ref} != ref_end - ref_start ({r_e - r_s}) in {rid}",
                    )

                rkey = (chrom, rid)
                if rkey in regions_seen:
                    if regions_seen[rkey][:2] != (r_s, r_e):
                        _err(
                            "CEN",
                            f"Inconsistent ref_start/ref_end across rows of region {rid}",
                        )
                else:
                    regions_seen[rkey] = (
                        r_s,
                        r_e,
                        crow["mode"],
                        crow["anchor_u"],
                        crow["anchor_v"],
                    )

                nonref = crow["nonref_input_bp"]
                if nonref != ".":
                    try:
                        total_in = int(nonref)
                        ref_like = int(crow["reference_like_bp"])
                        rep_like = int(crow["repeat_like_bp"])
                        short_b = int(crow["short_bp"])
                        unres_b = int(crow["unresolved_bp"])
                        acc_b = int(crow["accepted_source_bp"])
                    except ValueError as exc:
                        _err(
                            "CEN",
                            f"Invalid 5-way accounting integer in {rid}: {exc}",
                        )
                        continue
                    if (
                        min(total_in, ref_like, rep_like, short_b, unres_b, acc_b)
                        < 0
                    ):
                        _err(
                            "CEN",
                            f"Negative base accounting value in {rid} hap {crow['hap']}",
                        )
                    if ref_like + rep_like + short_b + unres_b + acc_b != total_in:
                        _err(
                            "CEN",
                            f"5-way base accounting mismatch in {rid} hap {crow['hap']}: "
                            f"{ref_like}+{rep_like}+{short_b}+{unres_b}+{acc_b} != {total_in}",
                        )
                    accepted_bp_by_reg[rkey] += acc_b

                sup_ev = crow["supported_events"]
                if sup_ev and sup_ev != ".":
                    for eid in sup_ev.split(","):
                        if eid.strip():
                            supported_evs_by_reg[rkey].add(eid.strip())

            # Check non-overlapping centromere regions per chromosome
            regs_by_chrom: dict[str, list[tuple[int, int, str]]] = (
                collections.defaultdict(list)
            )
            for (chrom, rid), (r_s, r_e, _, _, _) in regions_seen.items():
                regs_by_chrom[chrom].append((r_s, r_e, rid))
            for chrom, ivs in regs_by_chrom.items():
                ivs.sort()
                for i in range(len(ivs) - 1):
                    if ivs[i][1] > ivs[i + 1][0]:
                        _err(
                            "CEN",
                            f"Overlapping centromere regions on {chrom}: {ivs[i]} vs {ivs[i + 1]}",
                        )

            # Check anchor_u / anchor_v step agreement and local mount offset delta at ref_end (ref_start(v))
            for (chrom, rid), (
                r_s,
                r_e,
                mode,
                anchor_u,
                anchor_v,
            ) in regions_seen.items():
                if is_large_offsets:
                    co = compact_offsets_by_chrom.get(chrom)
                    if co is None or int(co.step_idx.shape[0]) == 0:
                        continue
                    ref_total_len = int(co.r_e[-1])

                    def _find_by_ref_end(re_val: int) -> tuple[str, str, int] | None:
                        idx_f = int(np.searchsorted(co.r_e, re_val, side="left"))
                        if idx_f < int(co.r_e.shape[0]) and int(co.r_e[idx_f]) == re_val:
                            return (
                                str(int(co.step_idx[idx_f])),
                                str(co.nids[idx_f]),
                                int(co.slot_ins[idx_f]),
                            )
                        return None

                    def _find_by_ref_start(
                        rs_val: int,
                    ) -> tuple[str, str, int, int, int] | None:
                        idx_f = int(np.searchsorted(co.r_s, rs_val, side="left"))
                        if idx_f < int(co.r_s.shape[0]) and int(co.r_s[idx_f]) == rs_val:
                            prev_pe = int(co.p_e[idx_f - 1]) if idx_f > 0 else 0
                            return (
                                str(int(co.step_idx[idx_f])),
                                str(co.nids[idx_f]),
                                int(co.slot_ins[idx_f]),
                                prev_pe,
                                int(co.p_s[idx_f]),
                            )
                        return None
                else:
                    chrom_offsets = offsets_by_chrom.get(chrom, [])
                    if not chrom_offsets:
                        continue
                    offset_idx_by_ref_start_c = {
                        int(r["ref_start"]): idx_r
                        for idx_r, r in enumerate(chrom_offsets)
                    }
                    offset_by_ref_end_c = {
                        int(r["ref_end"]): r for r in chrom_offsets
                    }
                    ref_total_len = int(chrom_offsets[-1]["ref_end"])

                    def _find_by_ref_end(re_val: int) -> tuple[str, str, int] | None:
                        r_row = offset_by_ref_end_c.get(re_val)
                        if r_row is None:
                            return None
                        return (
                            r_row["ref_step"],
                            r_row["node_id"],
                            int(r_row["slot_ins_len"]),
                        )

                    def _find_by_ref_start(
                        rs_val: int,
                    ) -> tuple[str, str, int, int, int] | None:
                        idx_r = offset_idx_by_ref_start_c.get(rs_val)
                        if idx_r is None:
                            return None
                        r_row = chrom_offsets[idx_r]
                        prev_pe = (
                            int(chrom_offsets[idx_r - 1]["pan_end"])
                            if idx_r > 0
                            else 0
                        )
                        return (
                            r_row["ref_step"],
                            r_row["node_id"],
                            int(r_row["slot_ins_len"]),
                            prev_pe,
                            int(r_row["pan_start"]),
                        )

                if anchor_u in ("VIRTUAL_START", ".") or r_s == 0:
                    if r_s != 0:
                        _err(
                            "CEN",
                            f"Centromere {rid} virtual left boundary ref_start={r_s} != 0",
                        )
                else:
                    u_clean = anchor_u.rstrip("+-")
                    u_info = _find_by_ref_end(r_s)
                    if u_info is None or (
                        u_info[0] != anchor_u and u_info[1] != u_clean
                    ):
                        _err(
                            "CEN",
                            f"Centromere {rid} ref_start={r_s} does not match anchor_u='{anchor_u}' in offsets.tsv",
                        )

                if anchor_v not in ("L", "END", "VIRTUAL_END", "."):
                    v_clean = anchor_v.rstrip("+-")
                    v_info = _find_by_ref_start(r_e)
                    if v_info is None or (
                        v_info[0] != anchor_v and v_info[1] != v_clean
                    ):
                        _err(
                            "CEN",
                            f"Centromere {rid} ref_end={r_e} does not match anchor_v='{anchor_v}' in offsets.tsv",
                        )
                else:
                    if r_e != ref_total_len:
                        _err(
                            "CEN",
                            f"Centromere {rid} virtual right boundary ref_end={r_e} != {ref_total_len}",
                        )

                cen_events_for_reg = [
                    ev
                    for ev in sv_rows
                    if ev["chrom"] == chrom
                    and ev["type"] == "CEN_ISLAND"
                    and (
                        ev["region_id"] == rid
                        or int(ev["ref_start"]) == r_e
                        or ev["event_id"] in supported_evs_by_reg[(chrom, rid)]
                    )
                ]
                for cev in cen_events_for_reg:
                    if int(cev["ref_start"]) != r_e or int(cev["ref_end"]) != r_e:
                        _err(
                            "CEN",
                            f"CEN_ISLAND event {cev['event_id']} ref_start={cev['ref_start']} != region ref_end={r_e}",
                        )

                if accepted_bp_by_reg[(chrom, rid)] > 0 and not cen_events_for_reg:
                    _err(
                        "CEN",
                        f"Region {rid} has accepted_source_bp={accepted_bp_by_reg[(chrom, rid)]} but no CEN_ISLAND event at ref_end={r_e}",
                    )

                slot_ins_len: int | None = None
                slot_pan_start: int | None = None
                slot_pan_end: int | None = None
                if r_e == ref_total_len and chrom in pan_records:
                    if is_large_offsets:
                        co_c = compact_offsets_by_chrom.get(chrom)
                        last_pe_c = (
                            int(co_c.p_e[-1])
                            if co_c is not None and int(co_c.p_e.shape[0]) > 0
                            else 0
                        )
                    else:
                        c_offs = offsets_by_chrom.get(chrom, [])
                        last_pe_c = int(c_offs[-1]["pan_end"]) if c_offs else 0
                    slot_pan_start = last_pe_c
                    slot_pan_end = len(pan_records[chrom])
                    slot_ins_len = slot_pan_end - slot_pan_start
                else:
                    v_mount_info = _find_by_ref_start(r_e)
                    if v_mount_info is not None:
                        slot_ins_len = v_mount_info[2]
                        slot_pan_start = v_mount_info[3]
                        slot_pan_end = v_mount_info[4]

                if cen_events_for_reg:
                    if (
                        slot_ins_len is None
                        or slot_pan_start is None
                        or slot_pan_end is None
                        or chrom not in pan_records
                    ):
                        _err(
                            "CEN",
                            f"CEN_ISLAND mounted at ref_end={r_e} which is missing from offsets.tsv",
                        )
                    else:
                        local_offset_delta = slot_pan_end - slot_pan_start
                        if local_offset_delta != slot_ins_len:
                            _err(
                                "CEN",
                                f"Mount point at ref={r_e} local_offset_delta={local_offset_delta} "
                                f"!= slot_ins_len={slot_ins_len}",
                            )

                        cen_ev_ids = {ev["event_id"] for ev in cen_events_for_reg}
                        norm_ev_ids = {
                            ev["event_id"]
                            for ev in sv_rows
                            if ev["chrom"] == chrom
                            and ev["type"] == "INS"
                            and int(ev["ref_start"]) == r_e
                            and int(ev["ref_end"]) == r_e
                        }
                        allowed_ev_ids = cen_ev_ids | norm_ev_ids

                        slot_pieces: list[dict[str, str]] = []
                        for ins_row in ins_by_chrom.get(chrom, ()):
                            try:
                                p_s_i = int(ins_row["pan_start"])
                                p_e_i = int(ins_row["pan_end"])
                            except ValueError:
                                continue
                            overlaps_slot = (
                                p_s_i < slot_pan_end and p_e_i > slot_pan_start
                            )
                            in_allowed = ins_row["event_id"] in allowed_ev_ids
                            if overlaps_slot or in_allowed:
                                if not (
                                    slot_pan_start <= p_s_i < p_e_i <= slot_pan_end
                                ):
                                    _err(
                                        "CEN",
                                        f"Mount-slot piece {ins_row['event_id']}:{ins_row['piece']} "
                                        f"[{p_s_i},{p_e_i}) escapes slot [{slot_pan_start},{slot_pan_end})",
                                    )
                                    continue
                                if not in_allowed:
                                    _err(
                                        "CEN",
                                        f"Unexpected piece {ins_row['event_id']}:{ins_row['piece']} "
                                        f"inside centromere mount slot [{slot_pan_start},{slot_pan_end}) at ref={r_e}",
                                    )
                                    continue
                                slot_pieces.append(ins_row)

                        slot_pieces.sort(
                            key=lambda r_p: (
                                int(r_p["pan_start"]),
                                int(r_p["pan_end"]),
                            )
                        )
                        cen_bp = sum(
                            int(cp["pan_end"]) - int(cp["pan_start"])
                            for cp in slot_pieces
                            if cp["event_id"] in cen_ev_ids
                        )
                        normal_ins_bp = sum(
                            int(np_r["pan_end"]) - int(np_r["pan_start"])
                            for np_r in slot_pieces
                            if np_r["event_id"] in norm_ev_ids
                        )

                        if cen_bp <= 0 or not slot_pieces:
                            _err(
                                "CEN",
                                f"Mount point at ref={r_e} has no valid CEN_ISLAND bases (cen_bp={cen_bp})",
                            )
                        else:
                            pan_up_c = pan_records[chrom].upper()
                            seen_norm_piece = False
                            for idx_sp, sp in enumerate(slot_pieces):
                                ps_i = int(sp["pan_start"])
                                ev_i = sp["event_id"]
                                jb_i = sp["join_before"]
                                if ev_i in norm_ev_ids:
                                    seen_norm_piece = True
                                elif seen_norm_piece:
                                    _err(
                                        "CEN",
                                        f"Mount point at ref={r_e} places CEN_ISLAND piece {ev_i} "
                                        f"after co-mounted normal INS piece (§3.8.4-D)",
                                    )

                                if idx_sp == 0:
                                    lead_gap = ps_i - slot_pan_start
                                    if jb_i != "spacer" or lead_gap != eff_spacer_n:
                                        _err(
                                            "CEN",
                                            f"Mount point at ref={r_e} leading spacer length={lead_gap} "
                                            f"(join_before='{jb_i}') != spacer_n={eff_spacer_n}",
                                        )
                                    elif (
                                        pan_up_c[slot_pan_start:ps_i]
                                        != "N" * eff_spacer_n
                                    ):
                                        _err(
                                            "CEN",
                                            f"Mount point at ref={r_e} leading gap pan[{slot_pan_start}:{ps_i}] is not all 'N'",
                                        )
                                else:
                                    prev_sp = slot_pieces[idx_sp - 1]
                                    prev_pe_i = int(prev_sp["pan_end"])
                                    gap_i = ps_i - prev_pe_i
                                    if ev_i != prev_sp["event_id"] and jb_i != "spacer":
                                        _err(
                                            "CEN",
                                            f"Mount point at ref={r_e} boundary between {prev_sp['event_id']} "
                                            f"and {ev_i} must have join_before='spacer', got '{jb_i}'",
                                        )
                                    if jb_i == "spacer":
                                        if gap_i != eff_spacer_n:
                                            _err(
                                                "CEN",
                                                f"Mount point at ref={r_e} inter-piece spacer gap={gap_i} "
                                                f"before {ev_i}:{sp['piece']} != spacer_n={eff_spacer_n} "
                                                f"(stacked or missing spacer violates §3.8.4-D)",
                                            )
                                        elif (
                                            pan_up_c[prev_pe_i:ps_i]
                                            != "N" * eff_spacer_n
                                        ):
                                            _err(
                                                "CEN",
                                                f"Mount point at ref={r_e} inter-piece gap pan[{prev_pe_i}:{ps_i}] is not all 'N'",
                                            )
                                    else:
                                        if gap_i != 0:
                                            _err(
                                                "CEN",
                                                f"Mount point at ref={r_e} non-spacer join '{jb_i}' "
                                                f"before {ev_i}:{sp['piece']} has non-zero gap={gap_i}",
                                            )

                            last_pe_slot = int(slot_pieces[-1]["pan_end"])
                            trail_gap = slot_pan_end - last_pe_slot
                            expected_trail_gap = (
                                eff_spacer_n if normal_ins_bp == 0 else 0
                            )
                            if trail_gap != expected_trail_gap:
                                _err(
                                    "CEN",
                                    f"Mount point at ref={r_e} trailing gap={trail_gap} != "
                                    f"expected={expected_trail_gap} (§3.8.4-D / §3.6.4)",
                                )
                            elif (
                                expected_trail_gap > 0
                                and pan_up_c[last_pe_slot:slot_pan_end]
                                != "N" * expected_trail_gap
                            ):
                                _err(
                                    "CEN",
                                    f"Mount point at ref={r_e} trailing gap pan[{last_pe_slot}:{slot_pan_end}] is not all 'N'",
                                )

                            n_spacer_joins = sum(
                                1
                                for idx_sp, sp in enumerate(slot_pieces)
                                if idx_sp == 0 or sp["join_before"] == "spacer"
                            ) + (1 if normal_ins_bp == 0 else 0)
                            expected_spacer_bp = n_spacer_joins * eff_spacer_n
                            actual_spacer_bp = (
                                local_offset_delta - cen_bp - normal_ins_bp
                            )
                            expected_total_delta = (
                                cen_bp + normal_ins_bp + expected_spacer_bp
                            )
                            if (
                                actual_spacer_bp != expected_spacer_bp
                                or local_offset_delta != expected_total_delta
                            ):
                                _err(
                                    "CEN",
                                    f"Mount point at ref={r_e} local_offset_delta={local_offset_delta} "
                                    f"!= cen_bp({cen_bp}) + normal_ins_bp({normal_ins_bp}) + "
                                    f"spacer_bp({expected_spacer_bp}) (§3.8.5)",
                                )
                elif (
                    slot_ins_len is not None
                    and slot_pan_start is not None
                    and slot_pan_end is not None
                ):
                    norm_ev_ids_empty_cen = {
                        ev["event_id"]
                        for ev in sv_rows
                        if ev["chrom"] == chrom
                        and ev["type"] == "INS"
                        and int(ev["ref_start"]) == r_e
                        and int(ev["ref_end"]) == r_e
                    }
                    if not norm_ev_ids_empty_cen and (
                        slot_ins_len != 0 or slot_pan_end != slot_pan_start
                    ):
                        _err(
                            "CEN",
                            f"Region {rid} retained 0 CEN_ISLAND representatives and has no normal INS at "
                            f"ref={r_e}, but slot_ins_len={slot_ins_len} (delta={slot_pan_end - slot_pan_start}) != 0 (§3.8.4-D)",
                        )

    # 6. Lifted GFF3 Verification (when ref_gff_path and {p}.ref.lifted.gff3 exist)
    if ref_gff_path is not None and Path(ref_gff_path).is_file():
        if not lifted_gff_path.is_file():
            _err(
                "INV3",
                f"Expected lifted GFF3 file '{lifted_gff_path}' for --ref-gff '{ref_gff_path}'",
            )
        else:
            try:
                ref_tx, ref_prot = _reconstruct_gff3_transcripts_and_proteins(
                    ref_records, Path(ref_gff_path)
                )
                pan_tx, pan_prot = _reconstruct_gff3_transcripts_and_proteins(
                    pan_records, lifted_gff_path
                )
                if ref_tx != pan_tx:
                    _err(
                        "INV3",
                        "Spliced mRNA transcripts differ between reference GFF3 and lifted GFF3",
                    )
                if ref_prot != pan_prot:
                    _err(
                        "INV3",
                        "Translated CDS proteins differ between reference GFF3 and lifted GFF3",
                    )
            except Exception as exc:
                _err("INV3", f"GFF3 liftover verification failed: {exc}")

    passed = (
        inv1_ok
        and inv2_ok
        and inv3_ok
        and cen_ok
        and unres_ok
        and (len(errors) == 0)
    )
    report = VerificationReport(
        passed=passed,
        invariant_1_passed=inv1_ok,
        invariant_2_passed=inv2_ok,
        invariant_3_passed=inv3_ok,
        centromere_passed=cen_ok,
        unresolved_passed=unres_ok,
        errors=tuple(errors),
        checked_ref_steps=checked_ref_steps,
        checked_pieces=checked_pieces,
        checked_chain_blocks=checked_chain_blocks,
        checked_unresolved=checked_unresolved,
    )
    if raise_on_error and not passed:
        raise InvariantViolationError("; ".join(errors))
    return report


verify_prefix = verify_build_outputs
