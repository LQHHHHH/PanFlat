"""Global Invariants I, II, III verifier (`panflat verify`) and `--qc` alignment redundancy analyzer (§1, §3.8.5, §5.5, §7)."""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import mappy

from panflat.gfa_io import parse_fasta, revcomp
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    UNRESOLVED_REASONS,
    BuildConfig,
    ChromosomeBuildResult,
    InvariantViolationError,
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

    - Short pieces (`50 <= L < 150`) are reported separately (`qc_short_50_149_*`).
    - Long pieces (`L >= 150`) are aligned against `result.pan_seq` with `mappy` (`preset="asm20"`)
      and per-piece union redundant bases are computed at `0.90`, `0.95`, and `0.98` identity thresholds.
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
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    records: dict[str, str] = {}
    cur_name: str | None = None
    chunks: list[str] = []
    for raw_line in text.splitlines():
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


def _parse_gfa_paths_and_walks(
    gfa_path: Path,
) -> tuple[dict[str, str], dict[str, tuple[int, str]]]:
    """Independently parse GFA S, P, W lines into `(nodes, path_seqs)`."""
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
    path_seqs: Mapping[str, tuple[int, str]],
    src_path: str,
    src_start: int,
    src_end: int,
    src_strand: str,
) -> str:
    lookup = src_path
    if "::" in lookup and lookup not in path_seqs:
        lookup = lookup.split("::", 1)[-1]

    if lookup in path_seqs:
        base_offset, full_seq = path_seqs[lookup]
        rs = src_start - base_offset
        re_ = src_end - base_offset
        if 0 <= rs <= re_ <= len(full_seq):
            sub = full_seq[rs:re_]
            return sub if src_strand == "+" else revcomp(sub)

    for k, (cand_offset, cand_seq) in path_seqs.items():
        if k == lookup or k.startswith(f"{lookup}:"):
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

    actual_sha = hashlib.sha256(gfa_file.read_bytes()).hexdigest().lower()
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

    try:
        ref_records = parse_fasta(str(ref_fa_file))
        pan_records = parse_fasta(str(pan_fa_path))
        offsets_hdr, offsets_rows = _parse_tsv_file(offsets_path)
        sv_hdr, sv_rows = _parse_tsv_file(sv_path)
        ins_hdr, ins_rows = _parse_tsv_file(inserts_path)
        unres_hdr, unres_rows = _parse_tsv_file(unresolved_path)
        unres_fa = _parse_fasta_allow_empty(unresolved_fa_path)
        gfa_nodes, path_seqs = _parse_gfa_paths_and_walks(gfa_file)
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

    # 1. Global Invariant I: Reference Base Preservation
    offsets_by_chrom: dict[str, list[dict[str, str]]] = collections.defaultdict(
        list
    )
    for r in offsets_rows:
        offsets_by_chrom[r["chrom"]].append(r)

    for ref_chrom, ref_seq in ref_records.items():
        ref_up = ref_seq.upper()
        if ref_chrom not in pan_records:
            _err("INV1", f"Reference chromosome '{ref_chrom}' missing in pan.fa")
            continue
        pan_up = pan_records[ref_chrom].upper()
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
        covered = bytearray(len(pan_up))

        for srow in offsets_by_chrom.get(ref_chrom, ()):
            p_s = int(srow["pan_start"])
            p_e = int(srow["pan_end"])
            if 0 <= p_s <= p_e <= len(pan_up):
                for pos in range(p_s, p_e):
                    if covered[pos] != 0:
                        _err(
                            "INV1",
                            f"Overlapping reference step at pan pos {pos} on {ref_chrom}",
                        )
                        break
                    covered[pos] = 1

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

            if not (0 <= p_s < p_e <= len(pan_up)):
                _err(
                    "INV2",
                    f"Insert piece [{p_s}, {p_e}) out of bounds for pan_len={len(pan_up)}",
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

            for pos in range(p_s, p_e):
                if covered[pos] != 0:
                    _err(
                        "INV2",
                        f"Overlap between insert piece and existing interval at pan pos {pos}",
                    )
                    break
                covered[pos] = 2

            prev_p_e = p_e
            prev_ev_id = row["event_id"]
            prev_allele = row["allele"]
            checked_pieces += 1

        # Partition check: uncovered bases must be 'N' in exact multiples of eff_spacer_n
        idx_pos = 0
        n_pan = len(pan_up)
        while idx_pos < n_pan:
            if covered[idx_pos] != 0:
                idx_pos += 1
                continue
            run_s = idx_pos
            while idx_pos < n_pan and covered[idx_pos] == 0:
                if pan_up[idx_pos] != "N":
                    _err(
                        "INV2",
                        f"Uncovered non-spacer base '{pan_up[idx_pos]}' at pan[{idx_pos}] on {ref_chrom}",
                    )
                    break
                idx_pos += 1
            while idx_pos < n_pan and covered[idx_pos] == 0:
                idx_pos += 1
            run_len = idx_pos - run_s
            if eff_spacer_n > 0 and (run_len % eff_spacer_n) != 0:
                _err(
                    "INV2",
                    f"Spacer run [{run_s}, {idx_pos}) of length {run_len} is not a multiple of spacer_n={eff_spacer_n}",
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

        chrom_offsets = offsets_by_chrom.get(ref_chrom, [])
        offset_by_ref_start = {
            int(r["ref_start"]): r for r in chrom_offsets if "ref_start" in r
        }

        t_cur = ch["tStart"]
        q_cur = ch["qStart"]
        ref_to_pan_map: dict[int, int] = {}
        seen_dq_boundaries: dict[int, int] = {}

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
            for k in range(size):
                ref_to_pan_map[t_cur + k] = q_cur + k

            boundary_ref = t_cur + size
            if boundary_ref < len(ref_up):
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
            if ref_to_pan_map.get(r_s) != p_s or ref_to_pan_map.get(r_e - 1) != p_e - 1:
                _err(
                    "INV3",
                    f"Coordinate mismatch between chain and offsets at step ref[{r_s}:{r_e})",
                )

        if len(ref_to_pan_map) == len(ref_up) and len(ref_up) > 0:
            rng = random.Random(12345)
            sample_positions = [0, len(ref_up) - 1] + [
                rng.randrange(len(ref_up))
                for _ in range(min(25, len(ref_up)))
            ]
            for r_pos in sample_positions:
                p_chain = ref_to_pan_map[r_pos]
                if pan_up[p_chain] != ref_up[r_pos]:
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

    if comment_steps:
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
                chrom_offsets = offsets_by_chrom.get(chrom, [])
                if not chrom_offsets:
                    continue
                offset_by_ref_start = {
                    int(r["ref_start"]): r for r in chrom_offsets
                }
                offset_by_ref_end = {
                    int(r["ref_end"]): r for r in chrom_offsets
                }
                ref_total_len = int(chrom_offsets[-1]["ref_end"])

                if anchor_u in ("VIRTUAL_START", ".") or r_s == 0:
                    if r_s != 0:
                        _err(
                            "CEN",
                            f"Centromere {rid} virtual left boundary ref_start={r_s} != 0",
                        )
                else:
                    u_clean = anchor_u.rstrip("+-")
                    if r_s not in offset_by_ref_end or (
                        offset_by_ref_end[r_s]["ref_step"] != anchor_u
                        and offset_by_ref_end[r_s]["node_id"] != u_clean
                    ):
                        _err(
                            "CEN",
                            f"Centromere {rid} ref_start={r_s} does not match anchor_u='{anchor_u}' in offsets.tsv",
                        )

                if anchor_v not in ("L", "END", "VIRTUAL_END", "."):
                    v_clean = anchor_v.rstrip("+-")
                    if r_e not in offset_by_ref_start or (
                        offset_by_ref_start[r_e]["ref_step"] != anchor_v
                        and offset_by_ref_start[r_e]["node_id"] != v_clean
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
                    if r_e not in offset_by_ref_start:
                        _err(
                            "CEN",
                            f"CEN_ISLAND mounted at ref_end={r_e} which is missing from offsets.tsv",
                        )
                    else:
                        slot_ins_len = int(
                            offset_by_ref_start[r_e]["slot_ins_len"]
                        )
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
