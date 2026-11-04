"""Global Invariants I, II, III verifier (`panflat verify`) and `--qc` alignment redundancy analyzer (§1, §3.8.5, §5.5, §7)."""

from __future__ import annotations

from array import array
import collections
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import mappy
import numpy as np

from panflat.gfa_io import parse_fasta, revcomp
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    UNRESOLVED_REASONS,
    BuildConfig,
    ChromosomeBuildResult,
    CompactNodeStore,
    InvariantViolationError,
    LazyStepTuple,
    VerificationReport,
)

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


def run_qc(
    result: ChromosomeBuildResult,
    config: BuildConfig | None = None,
) -> dict[str, Any]:
    """
    Compute `--qc` alignment redundancy metrics for inserted pieces (§7, F26).
    """
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
    piece_hits_by_idx: list[list[tuple[int, int, float]]] = []
    if long_rows and pan_up:
        for r in long_rows:
            p_s = int(r["pan_start"])
            p_e = int(r["pan_end"])
            piece_len = p_e - p_s
            seq = str(r.get("seq") or pan_up[p_s:p_e]).upper()
            masked_pan = pan_up[:p_s] + ("N" * piece_len) + pan_up[p_e:]
            aligner_asm20 = mappy.Aligner(
                seq=masked_pan, preset="asm20", best_n=50
            )
            aligner_short = mappy.Aligner(
                seq=masked_pan,
                k=13,
                w=5,
                min_cnt=2,
                min_chain_score=20,
                best_n=50,
            )
            hits_info: list[tuple[int, int, float]] = []
            for hit in list(aligner_asm20.map(seq)) + list(
                aligner_short.map(seq)
            ):
                if hit.blen <= 0:
                    continue
                identity = hit.mlen / hit.blen
                hits_info.append((hit.q_st, hit.q_en, identity))
            piece_hits_by_idx.append(hits_info)

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
    cur_name: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    records[cur_name] = "".join(chunks).upper()
                cur_name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append("".join(line.split()))
    if cur_name is not None:
        records[cur_name] = "".join(chunks).upper()
    return records


@dataclass
class _SparsePathSlices:
    """Stores only requested source intervals for a chromosome-scale GFA path."""

    base_offset: int
    total_len: int
    slices: dict[tuple[int, int], str]


def _parse_gfa_paths_and_walks(
    gfa_path: Path,
    required_slices_by_path: Mapping[str, set[tuple[int, int]]] | None = None,
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
        raw_pw_lines: list[bytes] = []

        with gfa_path.open("rb") as fh:
            for raw_b in fh:
                if not raw_b:
                    continue
                b0 = raw_b[0]
                if b0 == 76:  # b'L'
                    continue
                if b0 == 83 and raw_b[1:2] == b"\t":  # b'S\t'
                    t1 = raw_b.find(b"\t", 2)
                    if t1 < 0:
                        continue
                    nid_b = raw_b[2:t1]
                    if not nid_b.isdigit():
                        all_seq_int = False
                        break
                    nid_int = int(nid_b)
                    if nid_int != expected_nid:
                        all_seq_int = False
                        break
                    t2 = raw_b.find(b"\t", t1 + 1)
                    seq_b = (
                        raw_b[t1 + 1 :].rstrip(b"\r\n")
                        if t2 < 0
                        else raw_b[t1 + 1 : t2]
                    ).upper()
                    seq_buf.extend(seq_b)
                    node_lens_arr.append(len(seq_b))
                    expected_nid += 1
                    max_nid = nid_int
                elif (b0 == 80 or b0 == 87) and raw_b[1:2] == b"\t":  # b'P\t' or b'W\t'
                    raw_pw_lines.append(raw_b)

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
            w_re = re.compile(r"([><])(\d+)")

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
                step_lens = node_lens[np.abs(codes)]
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

            for raw_b in raw_pw_lines:
                raw = raw_b.decode("utf-8")
                parts = raw.rstrip("\r\n").split("\t")
                rec_type = parts[0]
                if rec_type == "P" and len(parts) >= 3:
                    pname = parts[1]
                    reqs = _matching_reqs(pname)
                    if not reqs:
                        path_seqs_sparse[pname] = _SparsePathSlices(0, 0, {})
                        continue
                    step_str = parts[2].strip()
                    nids = np.fromstring(
                        step_str.translate({43: None, 45: None}),
                        dtype=np.int32,
                        sep=",",
                    )
                    signs = np.frombuffer(
                        step_str.encode("ascii").translate(
                            None, b"0123456789, \t\r\n"
                        ),
                        dtype=np.uint8,
                    )
                    codes = np.where(signs == 45, -nids, nids).astype(
                        np.int32, copy=False
                    )
                    del step_str, nids, signs
                    total_len, sl_map = _extract_reqs_from_codes(codes, reqs, 0)
                    path_seqs_sparse[pname] = _SparsePathSlices(
                        0, total_len, sl_map
                    )
                elif rec_type == "W" and len(parts) >= 7:
                    sample, hap, seqid = parts[1], parts[2], parts[3]
                    start = 0 if parts[4] == "*" else int(parts[4])
                    end = -1 if parts[5] == "*" else int(parts[5])
                    key_plain = f"{sample}#{hap}#{seqid}"
                    codes = np.asarray(
                        [
                            int(m.group(2))
                            if m.group(1) == ">"
                            else -int(m.group(2))
                            for m in w_re.finditer(parts[6])
                        ],
                        dtype=np.int32,
                    )
                    walk_len = int(node_lens[np.abs(codes)].sum())
                    computed_end = start + walk_len if end < 0 else end
                    key_with_coords = (
                        f"{sample}#{hap}#{seqid}:{start}-{computed_end}"
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
            nodes[parts[1]] = parts[2].upper()
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
        chunks: list[str] = []
        for token in step_str.split(","):
            tok = token.strip()
            if not tok:
                continue
            nid, orient = tok[:-1], tok[-1]
            seq = nodes[nid]
            chunks.append(seq if orient == "+" else revcomp(seq))
        path_seqs[pname] = (0, "".join(chunks))

    w_step_re = re.compile(r"([><])([^><\s]+)")
    for sample, hap, seqid, start, end, walk_str in raw_w:
        chunks = []
        for match in w_step_re.finditer(walk_str):
            orient_sym, nid = match.group(1), match.group(2)
            seq = nodes[nid]
            chunks.append(seq if orient_sym == ">" else revcomp(seq))
        full_seq = "".join(chunks)
        computed_end = start + len(full_seq) if end < 0 else end
        key_with_coords = f"{sample}#{hap}#{seqid}:{start}-{computed_end}"
        key_plain = f"{sample}#{hap}#{seqid}"
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
    if ":" in s and not s.split(":")[0].isdigit():
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
) -> tuple[dict[str, str], dict[str, str]]:
    exons_by_parent: dict[str, list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    cds_by_parent: dict[str, list[dict[str, Any]]] = collections.defaultdict(
        list
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
        if row["type"] == "exon":
            exons_by_parent[parent].append(row)
        elif row["type"] == "CDS":
            cds_by_parent[parent].append(row)

    transcripts: dict[str, str] = {}
    for parent, ex_list in exons_by_parent.items():
        strand = ex_list[0]["strand"]
        seqid = ex_list[0]["seqid"]
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
        transcripts[parent] = "".join(parts_seq)

    proteins: dict[str, str] = {}
    for parent, cds_list in cds_by_parent.items():
        strand = cds_list[0]["strand"]
        seqid = cds_list[0]["seqid"]
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
                    f"CDS phase propagation mismatch on '{parent}': expected {expected_phase}, got {phase}"
                )
            expected_phase = (3 - ((len(seg_oriented) - phase) % 3)) % 3
            parts_seq.append(seg_oriented)
        first_phase = (
            int(ordered[0]["phase"]) if ordered[0]["phase"] != "." else 0
        )
        proteins[parent] = _translate_cds_seq(
            "".join(parts_seq), phase=first_phase
        )

    return transcripts, proteins


def _load_ref_records_for_verify(
    ref_fa_file: Path,
    pan_records: Mapping[str, str],
    summary_chrom: str | None,
    backbone_prefix: str | None = None,
) -> dict[str, str]:
    """
    Load reference FASTA records for `verify_build_outputs`.
    When `summary_chrom` is a single chromosome present in `pan_records` and `ref_fa_file`
    is a multi-FASTA, streams only the matching reference chromosome (including PanSN `#`-suffix match).
    """
    if (
        summary_chrom
        and summary_chrom != "ALL"
        and summary_chrom in pan_records
        and len(pan_records) == 1
    ):
        exact_seq: str | None = None
        prefix_matches: list[tuple[str, str]] = []
        suffix_matches: list[tuple[str, str]] = []
        cur_name: str | None = None
        keep_mode: str | None = None
        chunks: list[str] = []
        bp = (backbone_prefix or "").strip()

        def _check_hdr(hdr: str) -> str | None:
            if hdr == summary_chrom:
                return "exact"
            if bp and hdr.startswith(bp) and hdr.endswith(f"#{summary_chrom}"):
                return "prefix"
            if "#" in summary_chrom and summary_chrom.endswith(f"#{hdr}"):
                return "suffix"
            if "#" in hdr and hdr.endswith(f"#{summary_chrom}"):
                return "suffix"
            return None

        with ref_fa_file.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_name is not None and keep_mode is not None:
                        s_val = "".join(chunks).upper()
                        if keep_mode == "exact":
                            exact_seq = s_val
                        elif keep_mode == "prefix":
                            prefix_matches.append((cur_name, s_val))
                        else:
                            suffix_matches.append((cur_name, s_val))
                    cur_name = line[1:].strip().split()[0]
                    if not cur_name:
                        raise InvariantViolationError("Empty FASTA header")
                    keep_mode = _check_hdr(cur_name)
                    chunks = []
                elif keep_mode is not None:
                    chunks.append("".join(line.split()))

        if cur_name is not None and keep_mode is not None:
            s_val = "".join(chunks).upper()
            if keep_mode == "exact":
                exact_seq = s_val
            elif keep_mode == "prefix":
                prefix_matches.append((cur_name, s_val))
            else:
                suffix_matches.append((cur_name, s_val))

        if exact_seq is not None:
            return {summary_chrom: exact_seq}
        if len(prefix_matches) == 1:
            return {summary_chrom: prefix_matches[0][1]}
        if len(prefix_matches) > 1 and bp:
            for h_name, h_seq in prefix_matches:
                if h_name in (f"{bp}#1#{summary_chrom}", f"{bp}#0#{summary_chrom}"):
                    return {summary_chrom: h_seq}
        if len(suffix_matches) == 1:
            return {summary_chrom: suffix_matches[0][1]}

    return parse_fasta(str(ref_fa_file))


def verify_build_outputs(
    prefix: str | Path,
    gfa_path: str | Path,
    ref_fa_path: str | Path,
    ref_gff_path: str | Path | None = None,
    centromere_bed_path: str | Path | None = None,
    spacer_n: int | None = None,
    raise_on_error: bool = False,
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

    with gfa_file.open("rb") as gfa_bin_fh:
        actual_sha = hashlib.file_digest(gfa_bin_fh, "sha256").hexdigest().lower()
    recorded_sha = summary_kv.get("gfa_sha256")
    if recorded_sha and recorded_sha.lower() != actual_sha:
        _err(
            "INV2",
            f"GFA SHA-256 mismatch: summary.tsv has {recorded_sha}, actual is {actual_sha}",
        )

    eff_spacer_n = (
        spacer_n
        if spacer_n is not None
        else int(summary_kv.get("spacer_n", "100"))
    )

    is_large_offsets = offsets_path.stat().st_size > 2_000_000

    try:
        pan_records = parse_fasta(str(pan_fa_path))
        ref_records = _load_ref_records_for_verify(
            ref_fa_file,
            pan_records,
            summary_kv.get("chrom"),
            summary_kv.get("backbone_prefix"),
        )
        sv_hdr, sv_rows = _parse_tsv_file(sv_path)
        ins_hdr, ins_rows = _parse_tsv_file(inserts_path)
        unres_hdr, unres_rows = _parse_tsv_file(unresolved_path)
        unres_fa = _parse_fasta_allow_empty(unresolved_fa_path)

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

        gfa_nodes, path_seqs = _parse_gfa_paths_and_walks(gfa_file, req_slices)
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
                    if not cols[2] and parts[2].isdigit():
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
                if isinstance(gfa_nodes, CompactNodeStore):
                    nids_int = (
                        co.nids
                        if isinstance(co.nids, np.ndarray)
                        else np.asarray(co.nids, dtype=np.int32)
                    )
                    if bool(np.any((nids_int <= 0) | (nids_int >= int(gfa_nodes.node_lens.shape[0])) | (gfa_nodes.node_lens[nids_int] == 0))):
                        _err("INV1", "Reference step node missing from GFA")
                    elif not bool(np.array_equal(gfa_nodes.node_lens[nids_int], co.r_e - co.r_s)):
                        _err("INV1", "GFA node length mismatch with ref step range")
                else:
                    for idx_n, nid in enumerate(co.nids):
                        if str(nid) not in gfa_nodes:
                            _err("INV1", f"Reference step node '{nid}' missing from GFA")
                            break
            if int(co.r_e[-1]) != len(ref_up):
                _err(
                    "INV1",
                    f"offsets.tsv ends at ref {int(co.r_e[-1])}, expected {len(ref_up)}",
                )
            if int(co.p_e[-1]) != len(pan_up):
                _err(
                    "INV1",
                    f"offsets.tsv ends at pan {int(co.p_e[-1])}, expected pan_len={len(pan_up)}",
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
            _err(
                "INV1",
                f"offsets.tsv ends at pan {prev_pan_end}, expected pan_len={len(pan_up)}",
            )

    # 2. Global Invariant II: Piece Provenance, Spacers & Full Partition
    ins_by_chrom: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in ins_rows:
        ins_by_chrom[r["chrom"]].append(r)

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
    comment_blocks: list[tuple[int, int, int]] = []
    comment_steps: list[tuple[int, int, int, int, int, int]] = []

    cur_blocks: list[tuple[int, int, int]] | None = None
    for raw_line in chain_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            c_parts = line.split("\t")
            if c_parts[0] == "# block" and len(c_parts) >= 4:
                try:
                    comment_blocks.append(
                        (int(c_parts[1]), int(c_parts[2]), int(c_parts[3]))
                    )
                except ValueError:
                    _err("INV3", f"Malformed # block comment in chain: {line}")
            elif c_parts[0] == "# step" and len(c_parts) >= 7:
                try:
                    comment_steps.append(
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
            cur_blocks = []
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

    if not chain_headers:
        _err("INV3", "ref_to_pan.chain contains no chain headers")

    all_computed_blocks: list[tuple[int, int, int]] = []
    chain_by_chrom = {
        ch["tName"]: (ch, blks)
        for ch, blks in zip(chain_headers, chain_blocks_list)
    }

    for ref_chrom, ref_seq in ref_records.items():
        if ref_chrom not in chain_by_chrom or ref_chrom not in pan_records:
            _err("INV3", f"Missing chain for chromosome '{ref_chrom}'")
            continue
        ch, blocks = chain_by_chrom[ref_chrom]
        ref_up = ref_seq.upper()
        pan_up = pan_records[ref_chrom].upper()

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
        if ch["qSize"] != len(pan_up) or ch["qStart"] != 0 or ch["qEnd"] != len(pan_up):
            _err(
                "INV3",
                f"Chain query bounds ({ch['qStart']}, {ch['qEnd']}, {ch['qSize']}) != (0, {len(pan_up)}, {len(pan_up)})",
            )

        if is_large_offsets:
            co = compact_offsets_by_chrom.get(ref_chrom)
            offset_by_ref_start = {}
        else:
            co = None
            chrom_offsets = offsets_by_chrom.get(ref_chrom, [])
            offset_by_ref_start = {
                int(r["ref_start"]): r for r in chrom_offsets if "ref_start" in r
            }

        t_cur = ch["tStart"]
        q_cur = ch["qStart"]
        seen_dq_boundaries: dict[int, int] = {}
        blk_t_starts: list[int] = []
        blk_t_ends: list[int] = []
        blk_q_starts: list[int] = []

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

            all_computed_blocks.append((t_cur, t_cur + size, dq))
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

    if comment_blocks:
        if comment_blocks != all_computed_blocks:
            _err(
                "INV3",
                f"Chain # block footer metadata does not match chain blocks: {comment_blocks} != {all_computed_blocks}",
            )

    if comment_steps and not is_large_offsets:
        expected_steps = [
            (
                int(r["ref_step"]),
                int(r["ref_start"]),
                int(r["ref_end"]),
                int(r["pan_start"]),
                int(r["pan_end"]),
                int(r["slot_ins_len"]),
            )
            for r in offsets_rows
        ]
        if comment_steps != expected_steps:
            _err(
                "INV3",
                "Chain # step footer metadata does not match offsets.tsv",
            )

    # 4. Unresolved Provenance & FASTA Check
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
            if exp_id not in unres_fa:
                _err(
                    "UNRES",
                    f"Exported FASTA ID '{exp_id}' missing from unresolved.fa",
                )
            elif unres_fa[exp_id].upper() != u_seq:
                _err(
                    "UNRES",
                    f"Unresolved FASTA sequence mismatch for '{exp_id}'",
                )
        checked_unresolved += 1

    # 5. Centromere Accounting & Local Offset Delta Check
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

                    def _find_by_ref_start(rs_val: int) -> tuple[str, str, int] | None:
                        idx_f = int(np.searchsorted(co.r_s, rs_val, side="left"))
                        if idx_f < int(co.r_s.shape[0]) and int(co.r_s[idx_f]) == rs_val:
                            return (
                                str(int(co.step_idx[idx_f])),
                                str(co.nids[idx_f]),
                                int(co.slot_ins[idx_f]),
                            )
                        return None
                else:
                    chrom_offsets = offsets_by_chrom.get(chrom, [])
                    if not chrom_offsets:
                        continue
                    offset_by_ref_start_c = {
                        int(r["ref_start"]): r for r in chrom_offsets
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

                    def _find_by_ref_start(rs_val: int) -> tuple[str, str, int] | None:
                        r_row = offset_by_ref_start_c.get(rs_val)
                        if r_row is None:
                            return None
                        return (
                            r_row["ref_step"],
                            r_row["node_id"],
                            int(r_row["slot_ins_len"]),
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

                if cen_events_for_reg:
                    v_mount_info = _find_by_ref_start(r_e)
                    if v_mount_info is None:
                        _err(
                            "CEN",
                            f"CEN_ISLAND mounted at ref_end={r_e} which is missing from offsets.tsv",
                        )
                    else:
                        slot_ins_len = v_mount_info[2]
                        cen_ev_ids = {ev["event_id"] for ev in cen_events_for_reg}
                        cen_pieces = [
                            ins
                            for ins in ins_by_chrom.get(chrom, ())
                            if ins["event_id"] in cen_ev_ids
                        ]
                        cen_bp = sum(
                            int(cp["pan_end"]) - int(cp["pan_start"])
                            for cp in cen_pieces
                        )
                        if cen_bp <= 0 or slot_ins_len < cen_bp + 2 * eff_spacer_n:
                            _err(
                                "CEN",
                                f"Mount point at ref={r_e} slot_ins_len={slot_ins_len} < cen_bp={cen_bp} + 2*spacer_n={2 * eff_spacer_n}",
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
