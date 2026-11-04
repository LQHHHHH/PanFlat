"""Shared opaque-box E2E test harness and independent verification oracles for PanFlat v1.

Rules enforced in this module:
1. ZERO imports from `panflat` (strictly opaque-box via subprocess CLI invocation).
2. Deterministic sequence/graph generation from fixed PRNG seeds.
3. Independent oracles for Global Invariants I, II, III, centromere 5-way accounting
   and local offset delta, and GFF3 liftover transcript/CDS/protein preservation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path("/home/antigravity/LQHHHHH/PanFlat")

_COMPLEMENT_TABLE = str.maketrans(
    "ACGTRYKMSWBDHVNacgtrykmswbdhvn",
    "TGCAYRMKSWVHDBNtgcayrmkswvhdbn",
)

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

_SENSE_CODONS: tuple[str, ...] = tuple(
    codon for codon, aa in _CODON_TABLE.items() if aa != "*"
)


# ---------------------------------------------------------------------------
# 2.1 Sequence & K-mer Helpers
# ---------------------------------------------------------------------------

def revcomp(seq: str) -> str:
    """Exact DNA reverse-complement (A<->T, C<->G, N<->N, case-preserving)."""
    return seq.translate(_COMPLEMENT_TABLE)[::-1]


def _canonical_kmer(kmer: str) -> str:
    rc = revcomp(kmer)
    return kmer if kmer <= rc else rc


def make_complex_seq(length: int, seed: int = 42) -> str:
    """Generate a deterministic uppercase ACGT sequence with high 15-mer complexity.

    Guaranteed to have canonical 15-mer singleton_ratio >= 0.90 in every 500-1000 bp
    window so it never triggers repeat_mask_intervals.
    """
    if length <= 0:
        return ""
    rng = random.Random(seed)
    bases = ("A", "C", "G", "T")
    if length < 15:
        return "".join(rng.choice(bases) for _ in range(length))

    seq_chars = [rng.choice(bases) for _ in range(14)]
    used_canonical: set[str] = set()
    for _ in range(14, length):
        prefix14 = "".join(seq_chars[-14:])
        start_idx = rng.randrange(4)
        chosen: str | None = None
        for offset in range(4):
            b = bases[(start_idx + offset) % 4]
            cand = prefix14 + b
            ck = _canonical_kmer(cand)
            if ck not in used_canonical:
                chosen = b
                used_canonical.add(ck)
                break
        if chosen is None:
            chosen = bases[start_idx]
            used_canonical.add(_canonical_kmer(prefix14 + chosen))
        seq_chars.append(chosen)
    return "".join(seq_chars)


def make_satellite_seq(
    unit_len: int = 20,
    total_len: int = 1500,
    seed: int = 100,
    divergence: float = 0.0,
) -> str:
    """Generate a tandem repeat array of `total_len` bp by repeating a `unit_len` bp motif."""
    if total_len <= 0:
        return ""
    unit_len = max(1, unit_len)
    rng = random.Random(seed)
    bases = ("A", "C", "G", "T")
    unit = "".join(rng.choice(bases) for _ in range(unit_len))
    repeats = (total_len // unit_len) + 2
    raw = (unit * repeats)[:total_len]
    if divergence <= 0.0:
        return raw

    chars = list(raw)
    num_mut = int(round(total_len * divergence))
    positions = rng.sample(range(total_len), min(total_len, num_mut))
    for pos in positions:
        orig = chars[pos]
        alts = [b for b in bases if b != orig]
        chars[pos] = rng.choice(alts)
    return "".join(chars)


def compute_canonical_15mer_singleton_ratio(window_seq: str) -> tuple[str, float | None]:
    """Implement §3.8.4-C window classification on a single window sequence."""
    length = len(window_seq)
    if length < 15:
        return ("unknown", None)
    total_pos = length - 14
    seq_up = window_seq.upper()
    valid_kmers: list[str] = []
    valid_bases = set("ACGT")
    for i in range(total_pos):
        kmer = seq_up[i : i + 15]
        if all(ch in valid_bases for ch in kmer):
            valid_kmers.append(_canonical_kmer(kmer))

    if len(valid_kmers) < 0.90 * total_pos:
        return ("unknown", None)

    counts = Counter(valid_kmers)
    singletons = sum(1 for kmer in valid_kmers if counts[kmer] == 1)
    ratio = singletons / len(valid_kmers)
    status = "low" if ratio < 0.30 else "non-low"
    return (status, ratio)


def make_window_with_singleton_ratio(
    length: int = 514,
    target_ratio: float = 0.30,
    seed: int = 200,
) -> str:
    """Construct a sequence of `length` bp with exact canonical 15-mer `singleton_ratio`."""
    if length < 15:
        return make_complex_seq(length, seed=seed)
    total_pos = length - 14
    target_singletons = int(round(target_ratio * total_pos))
    return make_seq_with_exact_singleton_ratio(
        window_len=length,
        target_singleton_count=target_singletons,
        seed=seed,
    )


def make_seq_with_exact_singleton_ratio(
    window_len: int = 514,
    target_singleton_count: int | float = 150,
    seed: int = 200,
) -> str:
    """Construct a window of length `window_len` having exact singleton canonical 15-mers."""
    if window_len < 15:
        return make_complex_seq(window_len, seed=seed)
    total_pos = window_len - 14
    if isinstance(target_singleton_count, float) and 0.0 <= target_singleton_count <= 1.0:
        k_single = int(round(target_singleton_count * total_pos))
    else:
        k_single = int(target_singleton_count)

    k_single = max(0, min(total_pos, k_single))
    if k_single == total_pos:
        return make_complex_seq(window_len, seed=seed)
    if k_single == 0:
        unit_len = min(20, max(1, total_pos // 3))
        return make_satellite_seq(unit_len=unit_len, total_len=window_len, seed=seed)

    non_singletons = total_pos - k_single
    if non_singletons == 1:
        non_singletons = 2
        k_single = total_pos - 2

    rng = random.Random(seed)
    bases = ("A", "C", "G", "T")
    unit_len = min(16, max(1, non_singletons // 2))
    motif = "".join(rng.choice(bases) for _ in range(unit_len))
    periodic_len = non_singletons + 14
    periodic_part = (motif * ((periodic_len // unit_len) + 3))[:periodic_len]

    seq_chars = list(periodic_part)
    used_canonical: set[str] = {
        _canonical_kmer(periodic_part[i : i + 15]) for i in range(non_singletons)
    }

    for step_idx in range(k_single):
        prefix14 = "".join(seq_chars[-14:])
        avoid_periodic_base = motif[(periodic_len + step_idx) % unit_len] if step_idx == 0 else None
        candidates = [b for b in bases if b != avoid_periodic_base]
        rng.shuffle(candidates)
        if avoid_periodic_base is not None:
            candidates.append(avoid_periodic_base)
        chosen: str | None = None
        for b in candidates:
            ck = _canonical_kmer(prefix14 + b)
            if ck not in used_canonical:
                chosen = b
                used_canonical.add(ck)
                break
        if chosen is None:
            chosen = candidates[0]
            used_canonical.add(_canonical_kmer(prefix14 + chosen))
        seq_chars.append(chosen)

    result = "".join(seq_chars)
    assert len(result) == window_len
    return result


def oracle_repeat_mask_intervals(seq: str) -> list[tuple[int, int]]:
    """Independent oracle for §3.8.4-C `repeat_mask_intervals(seq)`."""
    length = len(seq)
    if length < 500:
        return []
    if length <= 1000:
        status, _ = compute_canonical_15mer_singleton_ratio(seq)
        return [(0, length)] if status == "low" else []

    starts = list(range(0, length - 1000 + 1, 500))
    tail_start = length - 1000
    if tail_start not in starts:
        starts.append(tail_start)
    starts.sort()

    windows: list[tuple[int, int, str]] = []
    for s in starts:
        e = s + 1000
        status, _ = compute_canonical_15mer_singleton_ratio(seq[s:e])
        windows.append((s, e, status))

    raw_intervals: list[tuple[int, int]] = []
    i = 0
    n_win = len(windows)
    while i < n_win:
        if windows[i][2] != "low":
            i += 1
            continue
        j = i
        while j < n_win and windows[j][2] == "low":
            j += 1
        if j - i >= 2:
            raw_intervals.append((windows[i][0], windows[j - 1][1]))
        i = j

    if not raw_intervals:
        return []
    merged: list[tuple[int, int]] = [raw_intervals[0]]
    for s, e in raw_intervals[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def mutate_seq_snps(
    seq: str,
    num_snps: int | Sequence[int] = 0,
    positions: Sequence[int] | None = None,
    seed: int = 300,
    snp_positions: Sequence[int] | int | None = None,
) -> str:
    """Introduce deterministic point substitutions at specified or evenly spaced positions."""
    if not seq:
        return seq
    rng = random.Random(seed)
    bases = ("A", "C", "G", "T")
    chars = list(seq.upper())
    length = len(chars)

    target_positions: list[int] = []
    if snp_positions is not None:
        if isinstance(snp_positions, int):
            num_snps = snp_positions
        else:
            target_positions = [int(p) for p in snp_positions]
    elif isinstance(num_snps, Sequence) and not isinstance(num_snps, (str, bytes)):
        target_positions = [int(p) for p in num_snps]
    elif positions is not None:
        target_positions = [int(p) for p in positions]

    if not target_positions and isinstance(num_snps, int) and num_snps > 0:
        count = min(length, num_snps)
        step = length / (count + 1)
        used: set[int] = set()
        for i in range(1, count + 1):
            pos = min(length - 1, max(0, int(round(i * step))))
            while pos in used and pos + 1 < length:
                pos += 1
            used.add(pos)
            target_positions.append(pos)

    for pos in target_positions:
        if 0 <= pos < length:
            orig = chars[pos]
            alts = [b for b in bases if b != orig]
            chars[pos] = alts[rng.randrange(len(alts))] if alts else "A"
    return "".join(chars)


def make_coding_seq(num_codons: int = 60, seed: int = 500) -> str:
    """Generate ATG + (num_codons - 2) non-stop sense codons + TAA stop codon with high complexity."""
    num_codons = max(2, num_codons)
    rng = random.Random(seed)
    codons = ["ATG"]
    used_canonical: set[str] = set()

    for _ in range(num_codons - 2):
        candidates = list(_SENSE_CODONS)
        rng.shuffle(candidates)
        chosen = candidates[0]
        cur_seq = "".join(codons)
        if len(cur_seq) >= 12:
            for cand in candidates:
                test_tail = (cur_seq + cand)[-17:]
                ok = True
                for i in range(max(0, len(test_tail) - 15 + 1)):
                    kmer = test_tail[i : i + 15]
                    if len(kmer) == 15 and _canonical_kmer(kmer) in used_canonical:
                        ok = False
                        break
                if ok:
                    chosen = cand
                    break
        codons.append(chosen)
        cur_seq = "".join(codons)
        for i in range(max(0, len(cur_seq) - 17), len(cur_seq) - 14):
            if i >= 0:
                used_canonical.add(_canonical_kmer(cur_seq[i : i + 15]))

    codons.append("TAA")
    return "".join(codons)


def translate_cds(cds_seq: str, phase: int = 0) -> str:
    """Translate `cds_seq[phase:]` into amino acids using standard genetic code (`*` for stop)."""
    trimmed = cds_seq[phase:].upper()
    aa_list: list[str] = []
    for i in range(0, len(trimmed) - 2, 3):
        codon = trimmed[i : i + 3]
        aa_list.append(_CODON_TABLE.get(codon, "X"))
    return "".join(aa_list)


# ---------------------------------------------------------------------------
# 2.2 Synthetic Input Builder (`SyntheticInputBuilder` & `InputPaths`)
# ---------------------------------------------------------------------------

@dataclass
class InputPaths:
    gfa_path: Path
    ref_fa_path: Path
    bed_path: Path | None
    gff_path: Path | None
    out_prefix: Path
    chrom: str
    backbone_prefix: str


def _normalize_step(step: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(step, tuple):
        nid, orient = str(step[0]), str(step[1])
        assert orient in ("+", "-"), f"Invalid step orientation: {orient}"
        return (nid, orient)
    s = str(step).strip()
    if s.startswith(">") or s.startswith("<"):
        return (s[1:], "+" if s[0] == ">" else "-")
    if s.endswith("+") or s.endswith("-"):
        return (s[:-1], s[-1])
    return (s, "+")


class SyntheticInputBuilder:
    """Builder for synthetic GFA v1, reference FASTA, centromere BED, and reference GFF3 files."""

    def __init__(
        self,
        tmp_path: Path,
        chrom: str = "Chr01",
        backbone_prefix: str = "Ref#1#",
    ) -> None:
        self.tmp_path = Path(tmp_path)
        self.chrom = chrom
        self.backbone_prefix = backbone_prefix
        self.nodes: dict[str, str] = {}
        self.node_order: list[str] = []
        self.links: list[tuple[str, str, str, str, str]] = []
        self.link_set: set[tuple[str, str, str, str, str]] = set()
        self.paths: list[tuple[str, list[tuple[str, str]], str]] = []
        self.walks: list[tuple[str, str, str, str, str, list[tuple[str, str]]]] = []
        self.raw_lines: list[str] = []
        self.ref_fa_records: dict[str, str] = {}
        self.bed_rows: list[tuple[str, int, int, str | None]] = []
        self.gff_lines: list[str] = []

    def add_node(self, node_id: str, seq: str) -> SyntheticInputBuilder:
        nid = str(node_id)
        if nid not in self.nodes:
            self.node_order.append(nid)
        self.nodes[nid] = seq
        return self

    def add_link(
        self,
        from_id: str,
        from_orient: str,
        to_id: str,
        to_orient: str,
        overlap: str = "0M",
    ) -> SyntheticInputBuilder:
        entry = (str(from_id), str(from_orient), str(to_id), str(to_orient), str(overlap))
        if entry not in self.link_set:
            self.link_set.add(entry)
            self.links.append(entry)
        return self

    def _register_adj_links(self, norm_steps: Sequence[tuple[str, str]]) -> None:
        for i in range(len(norm_steps) - 1):
            u, uo = norm_steps[i]
            v, vo = norm_steps[i + 1]
            self.add_link(u, uo, v, vo, "0M")

    def add_path(
        self,
        path_name: str,
        steps: Sequence[str | tuple[str, str]],
        overlaps: str = "*",
        auto_links: bool = True,
    ) -> SyntheticInputBuilder:
        norm_steps = [_normalize_step(s) for s in steps]
        if auto_links:
            self._register_adj_links(norm_steps)
        self.paths.append((str(path_name), norm_steps, str(overlaps)))
        return self

    def add_walk(
        self,
        sample: str,
        hap: str | int,
        seqid: str,
        start: int | str,
        end: int | str | None,
        steps: Sequence[str | tuple[str, str]],
        auto_links: bool = True,
    ) -> SyntheticInputBuilder:
        norm_steps = [_normalize_step(s) for s in steps]
        if auto_links:
            self._register_adj_links(norm_steps)
        if end is None and isinstance(start, int):
            total_len = sum(len(self.nodes.get(nid, "")) for nid, _ in norm_steps)
            end_val = str(start + total_len)
        else:
            end_val = "*" if end is None else str(end)
        self.walks.append(
            (str(sample), str(hap), str(seqid), str(start), end_val, norm_steps)
        )
        return self

    def add_raw_line(self, line: str) -> SyntheticInputBuilder:
        self.raw_lines.append(line.rstrip("\r\n"))
        return self

    def spell_steps(self, steps: Sequence[str | tuple[str, str]]) -> str:
        norm_steps = [_normalize_step(s) for s in steps]
        parts: list[str] = []
        for nid, orient in norm_steps:
            seq = self.nodes[nid]
            parts.append(seq if orient == "+" else revcomp(seq))
        return "".join(parts)

    def set_ref_path(
        self,
        path_name: str,
        steps: Sequence[str | tuple[str, str]],
        ref_fa_seq: str | None = None,
        ref_fa_chrom: str | None = None,
        overlaps: str = "*",
        auto_links: bool = True,
        as_w_line: bool = False,
        w_sample: str = "Ref",
        w_hap: str = "1",
        w_seqid: str | None = None,
        w_start: int = 0,
    ) -> SyntheticInputBuilder:
        norm_steps = [_normalize_step(s) for s in steps]
        if as_w_line:
            self.add_walk(
                sample=w_sample,
                hap=w_hap,
                seqid=w_seqid or self.chrom,
                start=w_start,
                end=None,
                steps=norm_steps,
                auto_links=auto_links,
            )
        else:
            self.paths = [p for p in self.paths if p[0] != str(path_name)]
            self.add_path(
                path_name=path_name,
                steps=norm_steps,
                overlaps=overlaps,
                auto_links=auto_links,
            )
        chrom_name = ref_fa_chrom or self.chrom
        if ref_fa_seq is None:
            ref_fa_seq = self.spell_steps(norm_steps)
        self.ref_fa_records[chrom_name] = ref_fa_seq
        return self

    def add_ref_fasta_record(self, chrom: str, seq: str) -> SyntheticInputBuilder:
        self.ref_fa_records[str(chrom)] = seq
        return self

    def add_bed_region(
        self,
        start: int,
        end: int,
        region_id: str | None = None,
        chrom: str | None = None,
    ) -> SyntheticInputBuilder:
        self.bed_rows.append((chrom or self.chrom, int(start), int(end), region_id))
        return self

    def add_gff_line(
        self,
        seqid: str,
        source: str,
        ftype: str,
        start_1based: int,
        end_1based: int,
        score: str = ".",
        strand: str = "+",
        phase: str = ".",
        attributes: str = "",
    ) -> SyntheticInputBuilder:
        row = "\t".join(
            [
                str(seqid),
                str(source),
                str(ftype),
                str(start_1based),
                str(end_1based),
                str(score),
                str(strand),
                str(phase),
                str(attributes),
            ]
        )
        self.gff_lines.append(row)
        return self

    def add_gff_gene(
        self,
        gene_id: str,
        mrna_id: str,
        strand: str,
        exons: Sequence[tuple[int, int]],
        cds_list: Sequence[tuple[int, int, int]],
        chrom: str | None = None,
    ) -> SyntheticInputBuilder:
        seqid = chrom or self.chrom
        all_starts = [s for s, _ in exons] + [s for s, _, _ in cds_list]
        all_ends = [e for _, e in exons] + [e for _, e, _ in cds_list]
        gene_start = min(all_starts)
        gene_end = max(all_ends)
        self.add_gff_line(
            seqid, "panflat_test", "gene", gene_start, gene_end, ".", strand, ".", f"ID={gene_id}"
        )
        self.add_gff_line(
            seqid,
            "panflat_test",
            "mRNA",
            gene_start,
            gene_end,
            ".",
            strand,
            ".",
            f"ID={mrna_id};Parent={gene_id}",
        )
        for idx, (ex_s, ex_e) in enumerate(exons, start=1):
            self.add_gff_line(
                seqid,
                "panflat_test",
                "exon",
                ex_s,
                ex_e,
                ".",
                strand,
                ".",
                f"ID={mrna_id}.exon{idx};Parent={mrna_id}",
            )
        for idx, (cds_s, cds_e, cds_phase) in enumerate(cds_list, start=1):
            self.add_gff_line(
                seqid,
                "panflat_test",
                "CDS",
                cds_s,
                cds_e,
                ".",
                strand,
                str(cds_phase),
                f"ID={mrna_id}.cds{idx};Parent={mrna_id}",
            )
        return self

    def write(self, name: str = "test") -> InputPaths:
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        gfa_path = self.tmp_path / f"{name}.gfa"
        ref_fa_path = self.tmp_path / f"{name}.ref.fa"
        out_prefix = self.tmp_path / f"{name}_out"

        gfa_lines = ["H\tVN:Z:1.0"]
        for nid in self.node_order:
            gfa_lines.append(f"S\t{nid}\t{self.nodes[nid]}")
        for u, uo, v, vo, ov in self.links:
            gfa_lines.append(f"L\t{u}\t{uo}\t{v}\t{vo}\t{ov}")
        for pname, steps, ov in self.paths:
            step_str = ",".join(f"{nid}{orient}" for nid, orient in steps)
            gfa_lines.append(f"P\t{pname}\t{step_str}\t{ov}")
        for sample, hap, seqid, start, end, steps in self.walks:
            walk_str = "".join(f"{'>' if orient == '+' else '<'}{nid}" for nid, orient in steps)
            gfa_lines.append(f"W\t{sample}\t{hap}\t{seqid}\t{start}\t{end}\t{walk_str}")
        for rline in self.raw_lines:
            gfa_lines.append(rline)
        gfa_path.write_text("\n".join(gfa_lines) + "\n", encoding="utf-8")

        fa_lines: list[str] = []
        for cname, cseq in self.ref_fa_records.items():
            fa_lines.append(f">{cname}")
            fa_lines.append(cseq)
        ref_fa_path.write_text("\n".join(fa_lines) + ("\n" if fa_lines else ""), encoding="utf-8")

        bed_path: Path | None = None
        if self.bed_rows:
            bed_path = self.tmp_path / f"{name}.centromere.bed"
            blines: list[str] = []
            for cname, s, e, rid in self.bed_rows:
                if rid is None:
                    blines.append(f"{cname}\t{s}\t{e}")
                else:
                    blines.append(f"{cname}\t{s}\t{e}\t{rid}")
            bed_path.write_text("\n".join(blines) + "\n", encoding="utf-8")

        gff_path: Path | None = None
        if self.gff_lines:
            gff_path = self.tmp_path / f"{name}.ref.gff3"
            header = ["##gff-version 3"]
            gff_path.write_text("\n".join(header + self.gff_lines) + "\n", encoding="utf-8")

        return InputPaths(
            gfa_path=gfa_path,
            ref_fa_path=ref_fa_path,
            bed_path=bed_path,
            gff_path=gff_path,
            out_prefix=out_prefix,
            chrom=self.chrom,
            backbone_prefix=self.backbone_prefix,
        )


SyntheticGFABuilder = SyntheticInputBuilder


# ---------------------------------------------------------------------------
# 2.3 CLI Subprocess Runners & `OutputBundle`
# ---------------------------------------------------------------------------

def _parse_fasta_file(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    if not path.exists():
        return records
    cur_id: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records[cur_id] = "".join(chunks)
            cur_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if cur_id is not None:
        records[cur_id] = "".join(chunks)
    return records


def _parse_fasta_with_headers(path: Path) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return records
    cur_id: str | None = None
    cur_header: str = ""
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records[cur_id] = (cur_header, "".join(chunks))
            cur_header = line[1:]
            cur_id = cur_header.split()[0]
            chunks = []
        else:
            chunks.append(line)
    if cur_id is not None:
        records[cur_id] = (cur_header, "".join(chunks))
    return records


def _parse_tsv(
    path: Path,
    int_cols: Sequence[str] = (),
) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return ([], [])
    lines = [
        line.rstrip("\r\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return ([], [])
    header = lines[0].lstrip("#").split("\t")
    int_set = set(int_cols)
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        row: dict[str, Any] = {}
        for col, val in zip(header, parts):
            if col in int_set and val != "." and (val.isdigit() or (val.startswith("-") and val[1:].isdigit())):
                row[col] = int(val)
            else:
                row[col] = val
        rows.append(row)
    return (header, rows)


def _parse_gff_attributes(attr_str: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in attr_str.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            attrs[k.strip()] = v.strip()
    return attrs


class OutputBundle:
    """Parsed representation of all 11 output files produced by `panflat build` or `panflat merge`."""

    def __init__(self, prefix: Path | str) -> None:
        self.prefix = Path(prefix)
        self.pan_fa_path = Path(f"{self.prefix}.pan.fa")
        self.offsets_path = Path(f"{self.prefix}.offsets.tsv")
        self.chain_path = Path(f"{self.prefix}.ref_to_pan.chain")
        self.sv_path = Path(f"{self.prefix}.sv.tsv")
        self.genotypes_path = Path(f"{self.prefix}.genotypes.tsv")
        self.inserts_path = Path(f"{self.prefix}.inserts.tsv")
        self.unresolved_path = Path(f"{self.prefix}.unresolved.tsv")
        self.unresolved_fa_path = Path(f"{self.prefix}.unresolved.fa")
        self.centromere_path = Path(f"{self.prefix}.centromere.tsv")
        self.lifted_gff_path = Path(f"{self.prefix}.ref.lifted.gff3")
        self.summary_path = Path(f"{self.prefix}.summary.tsv")
        self.log_path = Path(f"{self.prefix}.log")

        self.pan_fa: dict[str, str] = _parse_fasta_file(self.pan_fa_path)
        self.pan_seq: str = next(iter(self.pan_fa.values()), "")

        self.offsets_header, self.offsets = _parse_tsv(
            self.offsets_path,
            int_cols=("ref_step", "ref_start", "ref_end", "pan_start", "pan_end", "slot_ins_len"),
        )

        self.chain_text: str = (
            self.chain_path.read_text(encoding="utf-8") if self.chain_path.exists() else ""
        )
        self.chain_headers: list[dict[str, Any]] = []
        self.chain_blocks: list[list[tuple[int, int, int]]] = []
        self._parse_chain()

        self.sv_header, self.sv_rows = _parse_tsv(
            self.sv_path,
            int_cols=("pan_start", "pan_end", "ref_start", "ref_end", "n_carrier", "n_called"),
        )

        self.genotype_haps: list[str] = []
        self.genotypes: dict[str, dict[str, str]] = {}
        self._parse_genotypes()

        self.inserts_header, self.inserts = _parse_tsv(
            self.inserts_path,
            int_cols=("pan_start", "pan_end", "piece"),
        )

        self.unresolved_header, self.unresolved = _parse_tsv(
            self.unresolved_path,
            int_cols=("src_start", "src_end", "length"),
        )
        self.unresolved_fa: dict[str, tuple[str, str]] = _parse_fasta_with_headers(
            self.unresolved_fa_path
        )

        self.centromere_header: list[str] | None = None
        self.centromere_rows: list[dict[str, Any]] | None = None
        if self.centromere_path.exists():
            self.centromere_header, self.centromere_rows = _parse_tsv(
                self.centromere_path,
                int_cols=("bed_start", "bed_end", "ref_start", "ref_end", "L_ref"),
            )

        self.lifted_gff_rows: list[dict[str, Any]] | None = None
        if self.lifted_gff_path.exists():
            self.lifted_gff_rows = self._parse_gff(self.lifted_gff_path)

        self.summary_text: str = (
            self.summary_path.read_text(encoding="utf-8") if self.summary_path.exists() else ""
        )
        self.summary_kv: dict[str, str] = self._parse_summary()

        self.log_text: str = (
            self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else ""
        )
        self.log_json_records: list[dict[str, Any]] = self._parse_log_json()

    def _parse_chain(self) -> None:
        if not self.chain_text.strip():
            return
        cur_blocks: list[tuple[int, int, int]] | None = None
        for raw in self.chain_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "chain":
                if cur_blocks is not None:
                    self.chain_blocks.append(cur_blocks)
                self.chain_headers.append(
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
                cur_blocks = []
            elif cur_blocks is not None:
                if len(parts) == 3:
                    cur_blocks.append((int(parts[0]), int(parts[1]), int(parts[2])))
                elif len(parts) == 1:
                    cur_blocks.append((int(parts[0]), 0, 0))
        if cur_blocks is not None:
            self.chain_blocks.append(cur_blocks)

    def _parse_genotypes(self) -> None:
        if not self.genotypes_path.exists():
            return
        lines = [
            l.rstrip("\r\n")
            for l in self.genotypes_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if not lines:
            return
        header = lines[0].lstrip("#").split("\t")
        if len(header) >= 2:
            self.genotype_haps = header[1:]
        for line in lines[1:]:
            parts = line.split("\t")
            if not parts:
                continue
            ev_id = parts[0]
            self.genotypes[ev_id] = {
                hap: val for hap, val in zip(self.genotype_haps, parts[1:])
            }

    @staticmethod
    def _parse_gff(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            attrs = _parse_gff_attributes(parts[8])
            rows.append(
                {
                    "seqid": parts[0],
                    "source": parts[1],
                    "type": parts[2],
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "score": parts[5],
                    "strand": parts[6],
                    "phase": parts[7],
                    "attributes_raw": parts[8],
                    "attributes": attrs,
                }
            )
        return rows

    def _parse_summary(self) -> dict[str, str]:
        kv: dict[str, str] = {}
        for raw in self.summary_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                kv[parts[0]] = parts[1]
            elif len(parts) >= 3:
                kv[f"{parts[0]}:{parts[1]}"] = "\t".join(parts[2:])
                kv[parts[0]] = parts[1]
        return kv

    def _parse_log_json(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for raw in self.log_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            idx = line.find("{")
            if idx != -1 and line.endswith("}"):
                candidate = line[idx:]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
        return records


def run_panflat_cli(
    args: Sequence[str | Path],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Invoke `python3 -m panflat.cli` in a subprocess with PYTHONPATH set to PROJECT_ROOT."""
    cmd = [sys.executable, "-m", "panflat.cli", *[str(a) for a in args]]
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"panflat CLI failed (exit {proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc


def run_build(
    inputs: InputPaths,
    extra_args: Sequence[str | Path] = (),
    check: bool = True,
    out_prefix: Path | str | None = None,
    include_bed: bool = True,
    include_gff: bool = True,
) -> tuple[subprocess.CompletedProcess[str], OutputBundle | None]:
    """Run `panflat build` on `inputs` and return `(CompletedProcess, OutputBundle | None)`."""
    prefix = Path(out_prefix) if out_prefix is not None else inputs.out_prefix
    extra_strs = [str(x) for x in extra_args]
    args: list[str | Path] = [
        "build",
        "--gfa",
        inputs.gfa_path,
        "--ref-fa",
        inputs.ref_fa_path,
        "--backbone-prefix",
        inputs.backbone_prefix,
        "--out-prefix",
        prefix,
    ]
    if include_gff and inputs.gff_path is not None and "--ref-gff" not in extra_strs:
        args.extend(["--ref-gff", inputs.gff_path])
    if include_bed and inputs.bed_path is not None and "--centromere-bed" not in extra_strs:
        args.extend(["--centromere-bed", inputs.bed_path])
    args.extend(extra_args)

    proc = run_panflat_cli(args, check=check)
    bundle = OutputBundle(prefix) if proc.returncode == 0 else None
    return (proc, bundle)


def run_verify(
    prefix: Path | str,
    gfa_path: Path | str,
    ref_fa_path: Path | str,
    extra_args: Sequence[str | Path] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run standalone `panflat verify`."""
    args: list[str | Path] = [
        "verify",
        "--prefix",
        prefix,
        "--gfa",
        gfa_path,
        "--ref-fa",
        ref_fa_path,
        *extra_args,
    ]
    return run_panflat_cli(args, check=check)


def run_merge(
    prefixes: Sequence[Path | str],
    out_prefix: Path | str,
    extra_args: Sequence[str | Path] = (),
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], OutputBundle | None]:
    """Write a prefix-list file, run `panflat merge`, and return `(CompletedProcess, OutputBundle)`."""
    out_p = Path(out_prefix)
    prefix_list_path = out_p.parent / f"{out_p.name}_prefixes.txt"
    prefix_list_path.write_text(
        "\n".join(str(p) for p in prefixes) + "\n", encoding="utf-8"
    )
    args: list[str | Path] = [
        "merge",
        "--prefix-list",
        prefix_list_path,
        "--out-prefix",
        out_p,
        *extra_args,
    ]
    proc = run_panflat_cli(args, check=check)
    bundle = OutputBundle(out_p) if proc.returncode == 0 else None
    return (proc, bundle)


# ---------------------------------------------------------------------------
# 2.4 Independent Oracles
# ---------------------------------------------------------------------------

def _parse_gfa_paths_and_walks_independently(
    gfa_path: Path,
) -> tuple[dict[str, str], dict[str, tuple[int, str]]]:
    """Independently parse GFA S, P, W lines into `(nodes, path_sequences)`.

    Returns:
      - `nodes`: `node_id -> uppercase sequence`
      - `path_sequences`: `src_path_key -> (base_offset, spelled_sequence)`
        Supports lookup by full `P` name, `Sample#Hap#Seq:start-end`, or `Sample#Hap#Seq`.
    """
    nodes: dict[str, str] = {}
    raw_p: list[tuple[str, str]] = []
    raw_w: list[tuple[str, str, str, int, int, str]] = []
    for raw in gfa_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        rec_type = parts[0]
        if rec_type == "S" and len(parts) >= 3:
            nodes[parts[1]] = parts[2].upper()
        elif rec_type == "P" and len(parts) >= 3:
            raw_p.append((parts[1], parts[2]))
        elif rec_type == "W" and len(parts) >= 7:
            s_val = 0 if parts[4] == "*" else int(parts[4])
            e_val = -1 if parts[5] == "*" else int(parts[5])
            raw_w.append((parts[1], parts[2], parts[3], s_val, e_val, parts[6]))

    path_seqs: dict[str, tuple[int, str]] = {}
    for pname, step_str in raw_p:
        chunks: list[str] = []
        for token in step_str.split(","):
            token = token.strip()
            if not token:
                continue
            nid, orient = token[:-1], token[-1]
            seq = nodes[nid]
            chunks.append(seq if orient == "+" else revcomp(seq))
        full_seq = "".join(chunks)
        path_seqs[pname] = (0, full_seq)

    w_step_re = re.compile(r"([><])([^><]+)")
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

    return (nodes, path_seqs)


def _extract_source_slice(
    path_seqs: Mapping[str, tuple[int, str]],
    src_path: str,
    src_start: int,
    src_end: int,
    src_strand: str,
) -> str:
    lookup_key = src_path
    if "::" in lookup_key and lookup_key not in path_seqs:
        lookup_key = lookup_key.split("::", 1)[-1]
    assert lookup_key in path_seqs or any(
        k.startswith(f"{lookup_key}:") for k in path_seqs
    ), (
        f"Source path {src_path!r} not found in GFA paths {list(path_seqs.keys())[:10]}"
    )
    base_offset, full_seq = path_seqs.get(lookup_key, (0, ""))
    rel_start = src_start - base_offset
    rel_end = src_end - base_offset
    if not (0 <= rel_start <= rel_end <= len(full_seq)):
        for k, (cand_offset, cand_seq) in path_seqs.items():
            if k == lookup_key or k.startswith(f"{lookup_key}:"):
                rs = src_start - cand_offset
                re_ = src_end - cand_offset
                if 0 <= rs <= re_ <= len(cand_seq):
                    base_offset, full_seq = cand_offset, cand_seq
                    rel_start, rel_end = rs, re_
                    break
    assert 0 <= rel_start <= rel_end <= len(full_seq), (
        f"Invalid source slice [{src_start}, {src_end}) (rel [{rel_start}, {rel_end})) "
        f"for {src_path} of length {len(full_seq)} (offset {base_offset})"
    )
    sub = full_seq[rel_start:rel_end]
    return sub if src_strand == "+" else revcomp(sub)


def _parse_interval_spec(spec: str | int, fallback_end: int | None = None) -> tuple[int, int]:
    if isinstance(spec, int) and fallback_end is not None:
        return (spec, fallback_end)
    s = str(spec).strip()
    if ":" in s and not s.split(":")[0].isdigit():
        s = s.rsplit(":", 1)[-1]
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    if ".." in s:
        a, b = s.split("..", 1)
        return (int(a), int(b))
    raise AssertionError(f"Cannot parse source interval specification: {spec!r}")


def assert_all_global_invariants(
    bundle: OutputBundle,
    gfa_path: Path,
    ref_fa_path: Path,
    spacer_n: int = 100,
) -> None:
    """Independently verify Global Invariants I, II, III and unresolved provenance."""
    ref_records = _parse_fasta_file(Path(ref_fa_path))
    assert len(ref_records) == 1, f"Expected 1 reference chromosome in {ref_fa_path}, got {len(ref_records)}"
    ref_chrom, ref_seq = next(iter(ref_records.items()))
    ref_seq_up = ref_seq.upper()

    assert ref_chrom in bundle.pan_fa, f"Missing {ref_chrom} in pan.fa"
    pan_seq = bundle.pan_fa[ref_chrom]
    pan_seq_up = pan_seq.upper()

    # --- Invariant I: Reference Base Preservation ---
    offsets = bundle.offsets
    assert len(offsets) > 0, "offsets.tsv must not be empty"
    reconstructed_ref_chunks: list[str] = []
    expected_ref_pos = 0
    prev_pan_end = 0
    for idx, row in enumerate(offsets):
        assert row["chrom"] == ref_chrom
        assert row["ref_step"] == idx, f"Expected ref_step {idx}, got {row['ref_step']}"
        r_s, r_e = int(row["ref_start"]), int(row["ref_end"])
        p_s, p_e = int(row["pan_start"]), int(row["pan_end"])
        slot_ins = int(row["slot_ins_len"])
        assert r_s == expected_ref_pos, f"Ref gap/overlap at step {idx}: expected {expected_ref_pos}, got {r_s}"
        assert r_e > r_s, f"Empty/negative ref step at {idx}: [{r_s}, {r_e})"
        assert p_e - p_s == r_e - r_s, f"Step length mismatch at {idx}"
        assert p_s == prev_pan_end + slot_ins, (
            f"pan_start mismatch at step {idx}: {p_s} != {prev_pan_end} + {slot_ins}"
        )
        pan_step_seq = pan_seq_up[p_s:p_e]
        ref_step_seq = ref_seq_up[r_s:r_e]
        assert pan_step_seq == ref_step_seq, (
            f"Invariant I violated at ref_step {idx} ({row['node_id']}): "
            f"pan[{p_s}:{p_e}] != ref[{r_s}:{r_e}]"
        )
        reconstructed_ref_chunks.append(pan_step_seq)
        expected_ref_pos = r_e
        prev_pan_end = p_e

    assert expected_ref_pos == len(ref_seq_up), (
        f"Offsets did not cover full reference: {expected_ref_pos} != {len(ref_seq_up)}"
    )
    assert "".join(reconstructed_ref_chunks) == ref_seq_up, "Invariant I full reconstruction mismatch"

    # --- Invariant II: Piece Provenance & Join Integrity ---
    _, path_seqs = _parse_gfa_paths_and_walks_independently(Path(gfa_path))
    prev_ins_end: int | None = None
    for row in bundle.inserts:
        p_s, p_e = int(row["pan_start"]), int(row["pan_end"])
        assert 0 <= p_s < p_e <= len(pan_seq_up), f"Insert pan coords out of bounds: [{p_s}, {p_e})"
        src_path = str(row["src_path"])
        src_strand = str(row["src_strand"])
        assert src_strand in ("+", "-"), f"Invalid src_strand: {src_strand}"
        src_s, src_e = _parse_interval_spec(row["src_intervals"])
        assert src_e - src_s == p_e - p_s, (
            f"Piece length mismatch for {row['event_id']}: pan {p_e - p_s} vs src {src_e - src_s}"
        )
        expected_piece_seq = _extract_source_slice(
            path_seqs, src_path, src_s, src_e, src_strand
        ).upper()
        actual_piece_seq = pan_seq_up[p_s:p_e]
        assert actual_piece_seq == expected_piece_seq, (
            f"Invariant II piece sequence mismatch for {row['event_id']} piece {row['piece']} "
            f"at pan[{p_s}:{p_e}] vs {src_path}:{src_s}-{src_e}({src_strand})"
        )
        join_before = str(row["join_before"])
        assert join_before in ("source_contiguous", "spacer", "synthetic"), (
            f"Invalid join_before value: {join_before}"
        )
        if join_before == "spacer" and p_s >= spacer_n:
            spacer_slice = pan_seq_up[p_s - spacer_n : p_s]
            assert spacer_slice == "N" * spacer_n, (
                f"Expected {spacer_n} 'N' spacer before pan_start={p_s}, got {spacer_slice[:20]!r}"
            )
        if prev_ins_end is not None:
            assert p_s >= prev_ins_end, f"Overlapping insert rows: {p_s} < {prev_ins_end}"
        prev_ins_end = p_e

    # Verify partition of pan_seq into reference steps, insert pieces, and N spacers
    covered = bytearray(len(pan_seq_up))
    for row in offsets:
        for pos in range(int(row["pan_start"]), int(row["pan_end"])):
            assert covered[pos] == 0, f"Overlap between ref step and existing interval at pan pos {pos}"
            covered[pos] = 1
    for row in bundle.inserts:
        for pos in range(int(row["pan_start"]), int(row["pan_end"])):
            assert covered[pos] == 0, f"Overlap between insert piece and existing interval at pan pos {pos}"
            covered[pos] = 2
    for pos, tag in enumerate(covered):
        if tag == 0:
            assert pan_seq_up[pos] == "N", (
                f"Invariant II violated: uncovered base at pan[{pos}] is {pan_seq_up[pos]!r}, not spacer 'N'"
            )

    # --- Invariant III: Chain, Offsets & Coordinate Consistency ---
    assert len(bundle.chain_headers) >= 1, "ref_to_pan.chain must have at least 1 chain header"
    ch = bundle.chain_headers[0]
    blocks = bundle.chain_blocks[0]
    assert ch["tName"] == ref_chrom and ch["qName"] == ref_chrom
    assert ch["tSize"] == len(ref_seq_up)
    assert ch["qSize"] == len(pan_seq_up)
    assert ch["tStart"] == 0 and ch["tEnd"] == len(ref_seq_up)

    t_cur = ch["tStart"]
    q_cur = ch["qStart"]
    ref_to_pan_map: dict[int, int] = {}
    for size, dt, dq in blocks:
        assert size > 0, f"Chain block size must be > 0, got {size}"
        assert dt == 0, f"PanFlat v1 chain must not delete reference bases (dt={dt})"
        assert dq >= 0, f"Chain dq must be >= 0 (dq={dq})"
        assert pan_seq_up[q_cur : q_cur + size] == ref_seq_up[t_cur : t_cur + size], (
            f"Invariant III chain block mismatch at ref[{t_cur}:{t_cur+size}] vs pan[{q_cur}:{q_cur+size}]"
        )
        for offset in range(size):
            ref_to_pan_map[t_cur + offset] = q_cur + offset
        t_cur += size + dt
        q_cur += size + dq

    assert t_cur == ch["tEnd"], f"Chain tEnd mismatch: {t_cur} != {ch['tEnd']}"
    assert q_cur == ch["qEnd"], f"Chain qEnd mismatch: {q_cur} != {ch['qEnd']}"

    # Fixed-seed random coordinate sampling check between offsets, chain, and pan.fa
    rng = random.Random(12345)
    sample_positions = [0, len(ref_seq_up) - 1] + [
        rng.randrange(len(ref_seq_up)) for _ in range(min(25, len(ref_seq_up)))
    ]
    for r_pos in sample_positions:
        pan_pos_chain = ref_to_pan_map[r_pos]
        assert pan_seq_up[pan_pos_chain] == ref_seq_up[r_pos]
        matched_step = [
            row for row in offsets if int(row["ref_start"]) <= r_pos < int(row["ref_end"])
        ]
        assert len(matched_step) == 1
        srow = matched_step[0]
        pan_pos_offset = int(srow["pan_start"]) + (r_pos - int(srow["ref_start"]))
        assert pan_pos_chain == pan_pos_offset, (
            f"Invariant III mismatch between chain ({pan_pos_chain}) and offsets ({pan_pos_offset}) at ref {r_pos}"
        )

    # --- Unresolved Provenance Verification ---
    for urow in bundle.unresolved:
        u_len = int(urow["length"])
        u_s, u_e = int(urow["src_start"]), int(urow["src_end"])
        assert u_e - u_s == u_len, f"Unresolved length mismatch: {u_e} - {u_s} != {u_len}"
        u_seq = _extract_source_slice(
            path_seqs, str(urow["src_path"]), u_s, u_e, str(urow["src_strand"])
        ).upper()
        assert len(u_seq) == u_len
        exp_id = str(urow["exported_fasta_id"])
        if exp_id != ".":
            assert exp_id in bundle.unresolved_fa, f"Exported ID {exp_id} missing from unresolved.fa"
            _, fa_seq = bundle.unresolved_fa[exp_id]
            assert fa_seq.upper() == u_seq, f"Unresolved FASTA sequence mismatch for {exp_id}"


def assert_centromere_accounting_and_offsets(
    bundle: OutputBundle,
    spacer_n: int = 100,
) -> None:
    """Verify 27-column centromere.tsv 5-way base accounting and local offset delta at `ref_start(v)`."""
    assert bundle.centromere_rows is not None, "Expected centromere.tsv to exist"
    expected_27_cols = [
        "chrom", "region_id", "input_bed_ids", "bed_start", "bed_end",
        "ref_start", "ref_end", "anchor_u", "anchor_v", "mode",
        "hap", "status", "ref_status", "src_path", "src_start",
        "src_end", "src_strand", "L_ref", "L_path", "length_ratio",
        "nonref_input_bp", "reference_like_bp", "repeat_like_bp", "short_bp",
        "unresolved_bp", "accepted_source_bp", "supported_events",
    ]
    assert bundle.centromere_header == expected_27_cols, (
        f"centromere.tsv columns mismatch:\nExpected: {expected_27_cols}\nActual:   {bundle.centromere_header}"
    )

    regions_seen: dict[str, tuple[int, int, str, str]] = {}
    for row in bundle.centromere_rows:
        rid = str(row["region_id"])
        r_s, r_e = int(row["ref_start"]), int(row["ref_end"])
        mode = str(row["mode"])
        anchor_v = str(row["anchor_v"])
        regions_seen[rid] = (r_s, r_e, mode, anchor_v)

        nonref = str(row["nonref_input_bp"])
        if nonref != ".":
            total_in = int(nonref)
            parts_sum = (
                int(row["reference_like_bp"])
                + int(row["repeat_like_bp"])
                + int(row["short_bp"])
                + int(row["unresolved_bp"])
                + int(row["accepted_source_bp"])
            )
            assert parts_sum == total_in, (
                f"5-way base accounting mismatch in region {rid} hap {row['hap']}: "
                f"{parts_sum} != {total_in} (row={row})"
            )

    # Check non-overlapping expanded regions per chromosome
    by_chrom: dict[str, list[tuple[int, int, str]]] = {}
    for rid, (r_s, r_e, _, _) in regions_seen.items():
        chrom = next(r["chrom"] for r in bundle.centromere_rows if r["region_id"] == rid)
        by_chrom.setdefault(str(chrom), []).append((r_s, r_e, rid))
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            assert intervals[i][1] <= intervals[i + 1][0], (
                f"Overlapping centromere regions on {chrom}: {intervals[i]} vs {intervals[i+1]}"
            )

    # Verify local offset delta at mount point ref_start(v) when offsets.tsv is present
    if bundle.offsets:
        offset_by_ref_start = {int(r["ref_start"]): r for r in bundle.offsets}
        for rid, (r_s, r_e, mode, anchor_v) in regions_seen.items():
            if r_e in offset_by_ref_start:
                step_row = offset_by_ref_start[r_e]
                slot_ins_len = int(step_row["slot_ins_len"])
                cen_events = [
                    ev for ev in bundle.sv_rows
                    if ev["type"] == "CEN_ISLAND" and int(ev["ref_start"]) == r_e
                ]
                if cen_events:
                    cen_pieces = [
                        ins for ins in bundle.inserts
                        if ins["event_id"] in {ev["event_id"] for ev in cen_events}
                    ]
                    cen_bp = sum(int(p["pan_end"]) - int(p["pan_start"]) for p in cen_pieces)
                    assert slot_ins_len >= cen_bp + 2 * spacer_n, (
                        f"Mount point at ref={r_e} slot_ins_len ({slot_ins_len}) < "
                        f"CEN_ISLAND bases ({cen_bp}) + 2*spacer ({2*spacer_n})"
                    )


def _reconstruct_gff_mrna_and_proteins(
    fa_path: Path,
    gff_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Reconstruct spliced mRNA transcripts and translated CDS proteins from FASTA + GFF3."""
    seqs = _parse_fasta_file(fa_path)
    rows = OutputBundle._parse_gff(gff_path)
    exons_by_parent: dict[str, list[dict[str, Any]]] = {}
    cds_by_parent: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        parent = row["attributes"].get("Parent")
        if not parent:
            continue
        if row["type"] == "exon":
            exons_by_parent.setdefault(parent, []).append(row)
        elif row["type"] == "CDS":
            cds_by_parent.setdefault(parent, []).append(row)

    transcripts: dict[str, str] = {}
    for parent, ex_list in exons_by_parent.items():
        strand = ex_list[0]["strand"]
        seqid = ex_list[0]["seqid"]
        chrom_seq = seqs[seqid].upper()
        ordered = sorted(ex_list, key=lambda r: r["start"], reverse=(strand == "-"))
        parts: list[str] = []
        for r in ordered:
            seg = chrom_seq[r["start"] - 1 : r["end"]]
            parts.append(seg if strand == "+" else revcomp(seg))
        transcripts[parent] = "".join(parts)

    proteins: dict[str, str] = {}
    for parent, cds_list in cds_by_parent.items():
        strand = cds_list[0]["strand"]
        seqid = cds_list[0]["seqid"]
        chrom_seq = seqs[seqid].upper()
        ordered = sorted(cds_list, key=lambda r: r["start"], reverse=(strand == "-"))
        parts = []
        expected_phase: int | None = None
        for r in ordered:
            seg = chrom_seq[r["start"] - 1 : r["end"]]
            seg_oriented = seg if strand == "+" else revcomp(seg)
            phase = int(r["phase"]) if r["phase"] != "." else 0
            if expected_phase is not None:
                assert phase == expected_phase, (
                    f"CDS phase propagation error on {parent}: expected {expected_phase}, got {phase} at {r}"
                )
            expected_phase = (3 - ((len(seg_oriented) - phase) % 3)) % 3
            parts.append(seg_oriented)
        first_phase = int(ordered[0]["phase"]) if ordered[0]["phase"] != "." else 0
        full_cds = "".join(parts)
        proteins[parent] = translate_cds(full_cds, phase=first_phase)

    return (transcripts, proteins)


def assert_gff3_liftover_integrity(
    ref_fa_path: Path,
    ref_gff_path: Path,
    pan_fa_path: Path,
    lifted_gff_path: Path,
) -> None:
    """Assert 100% nucleotide transcript and translated protein identity before and after GFF3 liftover."""
    ref_tx, ref_prot = _reconstruct_gff_mrna_and_proteins(Path(ref_fa_path), Path(ref_gff_path))
    pan_tx, pan_prot = _reconstruct_gff_mrna_and_proteins(Path(pan_fa_path), Path(lifted_gff_path))

    assert set(ref_tx.keys()) == set(pan_tx.keys()), (
        f"Transcript IDs differ: {set(ref_tx.keys())} vs {set(pan_tx.keys())}"
    )
    for tx_id, expected_seq in ref_tx.items():
        actual_seq = pan_tx[tx_id]
        assert actual_seq == expected_seq, (
            f"Spliced transcript mismatch for {tx_id}: len {len(actual_seq)} vs {len(expected_seq)}"
        )

    assert set(ref_prot.keys()) == set(pan_prot.keys()), (
        f"Protein IDs differ: {set(ref_prot.keys())} vs {set(pan_prot.keys())}"
    )
    for tx_id, expected_aa in ref_prot.items():
        actual_aa = pan_prot[tx_id]
        assert actual_aa == expected_aa, (
            f"Translated protein mismatch for {tx_id}: {actual_aa!r} != {expected_aa!r}"
        )

    gffread_bin = shutil.which("gffread")
    if gffread_bin is not None:
        ref_out = Path(lifted_gff_path).parent / "_oracle_ref_tx.fa"
        pan_out = Path(lifted_gff_path).parent / "_oracle_pan_tx.fa"
        r1 = subprocess.run(
            [gffread_bin, "-w", str(ref_out), "-g", str(ref_fa_path), str(ref_gff_path)],
            capture_output=True,
            text=True,
        )
        r2 = subprocess.run(
            [gffread_bin, "-w", str(pan_out), "-g", str(pan_fa_path), str(lifted_gff_path)],
            capture_output=True,
            text=True,
        )
        if r1.returncode == 0 and r2.returncode == 0 and ref_out.exists() and pan_out.exists():
            fa1 = {k: v.upper() for k, v in _parse_fasta_file(ref_out).items()}
            fa2 = {k: v.upper() for k, v in _parse_fasta_file(pan_out).items()}
            assert fa1 == fa2, "gffread extracted transcripts differ between ref and lifted GFF3"


assert_gff3_protein_and_transcript_identity = assert_gff3_liftover_integrity
