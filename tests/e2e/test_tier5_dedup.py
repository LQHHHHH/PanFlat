"""Tier 5 strictly opaque-box CLI E2E tests for PanFlat homologous redundancy elimination (DEDUP_PLAN.md §4.2).

Rules strictly enforced in this file:
- ZERO `panflat` imports (`! rg -n '^\\s*(import|from)\\s+panflat\\b' tests/e2e` MUST exit 0).
- All builds, verifications, and merges execute via subprocess CLI through `tests.e2e.helpers`.
- Covers all 4 mandatory DEDUP_PLAN.md §4.2 scenarios plus CLI argument validation & multi-chrom merge:
  1. Ectopic multi-copy TE subtraction + embedded novel gene retention + GFF3 liftover +
     per-allele `partial_content=1` isolation + `.` genotyping + `--dedup-id 0.95` retention.
  2. Pericentromeric shifted-anchor cross-group deduplication (`Chr06A` scenario: ~252 kb -> ~52 kb)
     with co-located `DEL` preservation and zero spacers on dropped groups.
  3. 22-haplotype `--qc` closed-loop redundancy reduction (>70% -> <2%) with 100% recall of
     all 6 planted unique single-copy non-reference novel sequences.
  4. T22 multicore (`--threads 1` vs `--threads 4` with `PANFLAT_FORCE_FORK=1`) 12-file SHA-256
     byte-identity and shuffled GFA `P`/`W` input-order invariance.
  5. Subprocess CLI argument rejection and associative multi-chromosome `panflat merge`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Sequence

import pytest

from tests.e2e.helpers import (
    InputPaths,
    SyntheticInputBuilder,
    assert_all_global_invariants,
    assert_centromere_accounting_and_offsets,
    assert_gff3_liftover_integrity,
    make_coding_seq,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
    revcomp,
    run_build,
    run_merge,
    run_verify,
)

_ALL_12_OUTPUT_SUFFIXES: tuple[str, ...] = (
    ".pan.fa",
    ".offsets.tsv",
    ".ref_to_pan.chain",
    ".sv.tsv",
    ".genotypes.tsv",
    ".inserts.tsv",
    ".ref.lifted.gff3",
    ".centromere.tsv",
    ".unresolved.tsv",
    ".unresolved.fa",
    ".summary.tsv",
    ".log",
)

_BIO_10_OUTPUT_SUFFIXES: tuple[str, ...] = (
    ".pan.fa",
    ".offsets.tsv",
    ".ref_to_pan.chain",
    ".sv.tsv",
    ".genotypes.tsv",
    ".inserts.tsv",
    ".ref.lifted.gff3",
    ".centromere.tsv",
    ".unresolved.tsv",
    ".unresolved.fa",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fast_complex_dna(length: int, seed: int) -> str:
    """Fast deterministic high-complexity DNA generator (>99.9% unique 15-mers) for large 50 kb tests."""
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=length))


def _mutate_per_50bp_block(seq: str, block_offsets: Sequence[int], seed: int = 700) -> str:
    """Deterministically mutate exact offsets within every complete 50 bp block."""
    positions: list[int] = []
    for blk_start in range(0, len(seq) - 49, 50):
        for off in block_offsets:
            positions.append(blk_start + off)
    return mutate_seq_snps(seq, positions=positions, seed=seed)


def _reverse_steps(steps: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(nid, "-" if orient == "+" else "+") for nid, orient in reversed(steps)]


def _assert_12_files_sha256_identical_and_clean_streams(
    prefix_a: Path,
    prefix_b: Path,
    proc_a: subprocess.CompletedProcess[str],
    proc_b: subprocess.CompletedProcess[str],
) -> None:
    """Assert all 12 outputs exist & match SHA-256 byte-for-byte, stdout=='', and zero fork warnings."""
    assert proc_a.stdout == "", f"Unexpected stdout in run A:\n{proc_a.stdout}"
    assert proc_b.stdout == "", f"Unexpected stdout in run B:\n{proc_b.stdout}"
    for label, stderr in (("run A", proc_a.stderr), ("run B", proc_b.stderr)):
        assert "DeprecationWarning" not in stderr, (
            f"DeprecationWarning on stderr in {label}:\n{stderr}"
        )
        assert "multi-threaded, use of fork()" not in stderr, (
            f"Multi-threaded fork warning on stderr in {label}:\n{stderr}"
        )
    for suffix in _ALL_12_OUTPUT_SUFFIXES:
        pa = Path(f"{prefix_a}{suffix}")
        pb = Path(f"{prefix_b}{suffix}")
        assert pa.is_file(), f"Missing output file in run A: {pa}"
        assert pb.is_file(), f"Missing output file in run B: {pb}"
        ha, hb = _sha256_file(pa), _sha256_file(pb)
        assert ha == hb, f"SHA-256 mismatch on '{suffix}': {ha} ({pa}) != {hb} ({pb})"


def _normalize_gfa_meta_text(text: str, gfa_path: Path, gfa_sha256: str) -> str:
    """Mask only `gfa_path` and `gfa_sha256` tokens so original vs shuffled GFA metadata can be compared."""
    norm = text.replace(str(gfa_path), "<GFA_PATH>")
    norm = norm.replace(gfa_sha256, "<GFA_SHA256>")
    return norm


# ============================================================================
# Test 1: Ectopic Multi-Copy TE Subtraction + Embedded Novel Gene + GFF3
# ============================================================================


def test_tier5_dedup_1_ectopic_te_subtraction_novel_gene_gff3_and_threshold(
    tmp_path: Path,
) -> None:
    """DEDUP_PLAN §4.2 Test 1: ectopic 8 kb TE subtraction, 1.2 kb novel island retention,
    per-allele `partial_content=1` isolation, `.` genotyping, GFF3 liftover, and `--dedup-id 0.95` retention.
    """
    b = SyntheticInputBuilder(tmp_path / "t1_inputs", chrom="Chr01", backbone_prefix="Ref#1#")

    te_left = _fast_complex_dna(4000, seed=1101)
    te_right = _fast_complex_dna(4000, seed=1102)
    # 2-exon coding genes on '+' and '-' strands (360 bp CDS = 180 + 180 bp)
    plus_cds = make_coding_seq(num_codons=120, seed=1103)
    minus_cds = make_coding_seq(num_codons=120, seed=1104)
    minus_rc = revcomp(minus_cds)

    r1 = make_complex_seq(600, seed=1110)
    r2_te_l = te_left
    r3_te_r = te_right
    r4_plus_ex1 = make_complex_seq(20, seed=1111) + plus_cds[:180]  # 200 bp
    r5_intron = make_complex_seq(500, seed=1112)
    r6_plus_ex2 = plus_cds[180:] + make_complex_seq(20, seed=1113)  # 200 bp
    r7_minus_ex2 = make_complex_seq(20, seed=1114) + minus_rc[:180]  # 200 bp
    r8_minus_int = make_complex_seq(400, seed=1115)
    r9_minus_ex1 = minus_rc[180:] + make_complex_seq(20, seed=1116)  # 200 bp

    # Exact 94.0% identity per 50 bp block (47/50 = 0.94, soft-clip-proof offsets (6, 7, 43)):
    # masked at --dedup-id 0.90, retained at --dedup-id 0.95!
    te_l_94 = _mutate_per_50bp_block(te_left, (6, 7, 43), seed=1120)
    te_r_94 = _mutate_per_50bp_block(te_right, (6, 7, 43), seed=1121)
    novel_gene_1200 = make_complex_seq(1200, seed=1122)
    clean_sibling_1000 = make_complex_seq(1000, seed=1123)

    donor1_ins = te_l_94 + novel_gene_1200 + te_r_94  # 9,200 bp
    donor2_pure_te = _mutate_per_50bp_block(te_left, (6, 7, 43), seed=1124) + _mutate_per_50bp_block(
        te_right, (6, 7, 43), seed=1125
    )  # 8,000 bp

    for nid, seq in [
        ("1", r1),
        ("2", r2_te_l),
        ("3", r3_te_r),
        ("4", r4_plus_ex1),
        ("5", r5_intron),
        ("6", r6_plus_ex2),
        ("7", r7_minus_ex2),
        ("8", r8_minus_int),
        ("9", r9_minus_ex1),
        ("10", donor1_ins),
        ("11", donor2_pure_te),
        ("12", clean_sibling_1000),
    ]:
        b.add_node(nid, seq)

    ref_steps = [f"{i}+" for i in range(1, 10)]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # Compute exact 0-based ref coordinates for GFF3 gene placement
    ref_pos: dict[str, tuple[int, int]] = {}
    cur = 0
    for i in range(1, 10):
        nid = str(i)
        ln = len(b.nodes[nid])
        ref_pos[nid] = (cur, cur + ln)
        cur += ln

    # '+' gene spans the insertion slot between node 5 and node 6!
    r4_s, r4_e = ref_pos["4"]
    r6_s, _ = ref_pos["6"]
    b.add_gff_gene(
        gene_id="gene_plus",
        mrna_id="mrna_plus",
        strand="+",
        exons=[(r4_s + 21, r4_e), (r6_s + 1, r6_s + 180)],
        cds_list=[(r4_s + 21, r4_e, 0), (r6_s + 1, r6_s + 180, 0)],
    )
    # '-' gene sits downstream of the insertion slot (coordinates shifted by inserted bases)
    r7_s, r7_e = ref_pos["7"]
    r9_s, _ = ref_pos["9"]
    b.add_gff_gene(
        gene_id="gene_minus",
        mrna_id="mrna_minus",
        strand="-",
        exons=[(r7_s + 21, r7_e), (r9_s + 1, r9_s + 180)],
        cds_list=[(r7_s + 21, r7_e, 0), (r9_s + 1, r9_s + 180, 0)],
    )

    # Donor 1: 4kb TE_left(94%) + 1.2kb novel + 4kb TE_right(94%) between 5+ and 6+
    b.add_path("D1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "10+", "6+", "7+", "8+", "9+"])
    # Donor 2: pure 8kb TE(94%) at the exact same locus (5+, 6+)
    b.add_path("D2#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "11+", "6+", "7+", "8+", "9+"])
    # Donor 3: clean unrelated 1.0kb novel allele at the same locus (5+, 6+)
    b.add_path("D3#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "12+", "6+", "7+", "8+", "9+"])
    # Donor 4: exact reference match
    b.add_path("D4#1#Chr01", ref_steps)

    inp = b.write("t1_ectopic_te")
    assert inp.gff_path is not None

    # --- Run A: --dedup-mode ref --dedup-id 0.90 ---
    out_90 = tmp_path / "t1_ref_90"
    proc_90, bun_90 = run_build(
        inp,
        extra_args=["--dedup-mode", "ref", "--dedup-id", "0.90"],
        out_prefix=out_90,
    )
    assert proc_90.returncode == 0 and bun_90 is not None

    run_verify(
        out_90,
        inp.gfa_path,
        inp.ref_fa_path,
        extra_args=["--ref-gff", inp.gff_path],
        check=True,
    )
    assert_all_global_invariants(bun_90, inp.gfa_path, inp.ref_fa_path)
    assert_gff3_liftover_integrity(
        inp.ref_fa_path, inp.gff_path, bun_90.pan_fa_path, bun_90.lifted_gff_path
    )

    # Confirm 0 REFERENCE_LIKE rows in unresolved.tsv (logged ONLY to .log JSON)
    assert all(row["reason"] != "REFERENCE_LIKE" for row in bun_90.unresolved)
    assert any(rec.get("final_reason") == "REFERENCE_LIKE" for rec in bun_90.log_json_records)
    assert bun_90.summary_kv["dedup_mode"] == "ref"
    assert bun_90.summary_kv["dedup_id"] == "0.9"
    assert int(bun_90.summary_kv["dedup_ref_masked_bp"]) == 16_000

    # Inspect INS event at locus (5+, 6+)
    ins_evs = [r for r in bun_90.sv_rows if r["type"] == "INS"]
    assert len(ins_evs) == 1
    ins_ev = ins_evs[0]
    gt = bun_90.genotypes[ins_ev["event_id"]]

    # Genotyping: D2#1 (pure TE) MUST be '.', D4#1 and Ref#1 MUST be '0'
    assert gt["Ref#1"] == "0"
    assert gt["D4#1"] == "0"
    assert gt["D2#1"] == ".", f"Pure-TE donor D2#1 must be '.', got {gt['D2#1']!r}"
    assert gt["D1#1"] in ("1", "2")
    assert gt["D3#1"] in ("1", "2") and gt["D3#1"] != gt["D1#1"]

    # Parse per-allele descriptors from sv.tsv `alleles` column (e.g. `1:len=1200,partial_content=1;2:len=1000`)
    allele_desc_by_id: dict[str, str] = {}
    for tok in str(ins_ev["alleles"]).split(";"):
        a_id = tok.split(":", 1)[0]
        allele_desc_by_id[a_id] = tok

    d1_allele_desc = allele_desc_by_id[gt["D1#1"]]
    d3_allele_desc = allele_desc_by_id[gt["D3#1"]]
    assert "partial_content=1" in d1_allele_desc, (
        f"Expected partial_content=1 on trimmed D1 allele: {d1_allele_desc}"
    )
    assert "partial_content=1" not in d3_allele_desc, (
        f"Clean sibling D3 allele must NOT inherit partial_content=1: {d3_allele_desc}"
    )

    # Confirm ONLY the ~1.2 kb novel island was mounted for D1 (and 1.0 kb for D3)
    d1_pieces = [r for r in bun_90.inserts if str(r["src_path"]) == "D1#1#Chr01"]
    assert len(d1_pieces) == 1
    d1_len = int(d1_pieces[0]["pan_end"]) - int(d1_pieces[0]["pan_start"])
    assert d1_len == 1200
    assert bun_90.pan_seq[int(d1_pieces[0]["pan_start"]) : int(d1_pieces[0]["pan_end"])].upper() == novel_gene_1200

    d3_pieces = [r for r in bun_90.inserts if str(r["src_path"]) == "D3#1#Chr01"]
    assert len(d3_pieces) == 1
    assert int(d3_pieces[0]["pan_end"]) - int(d3_pieces[0]["pan_start"]) == 1000

    # --- Run B: --dedup-mode ref --dedup-id 0.95 (94%-identical flanks MUST be retained!) ---
    out_95 = tmp_path / "t1_ref_95"
    proc_95, bun_95 = run_build(
        inp,
        extra_args=["--dedup-mode", "ref", "--dedup-id", "0.95"],
        out_prefix=out_95,
    )
    assert proc_95.returncode == 0 and bun_95 is not None

    run_verify(
        out_95,
        inp.gfa_path,
        inp.ref_fa_path,
        extra_args=["--ref-gff", inp.gff_path],
        check=True,
    )
    assert_all_global_invariants(bun_95, inp.gfa_path, inp.ref_fa_path)
    assert_gff3_liftover_integrity(
        inp.ref_fa_path, inp.gff_path, bun_95.pan_fa_path, bun_95.lifted_gff_path
    )

    assert int(bun_95.summary_kv["dedup_ref_masked_bp"]) == 0
    d1_pieces_95 = [r for r in bun_95.inserts if str(r["src_path"]) == "D1#1#Chr01"]
    assert len(d1_pieces_95) == 1
    assert int(d1_pieces_95[0]["pan_end"]) - int(d1_pieces_95[0]["pan_start"]) == 9200
    ins_ev_95 = next(r for r in bun_95.sv_rows if r["type"] == "INS")
    gt_95 = bun_95.genotypes[ins_ev_95["event_id"]]
    assert gt_95["D2#1"] not in ("0", ".")  # 8 kb TE retained and called as an allele


# ============================================================================
# Test 2: Pericentromeric Shifted-Anchor Cross-Group Dedup (`Chr06A` Scenario)
# ============================================================================


def test_tier5_dedup_2_pericentromeric_shifted_anchor_cross_group_dedup(
    tmp_path: Path,
) -> None:
    """DEDUP_PLAN §4.2 Test 2: 50 kb shared sequence across shifted anchor pairs drops from
    ~252 kb to ~52 kb non-N inserted bases under `--dedup-mode pan`, retaining the 50 kb body
    at the highest-priority compact slot and the 2 kb private block (`,partial_content=1`) while
    dropping pure-homolog shifted groups (0 spacers) and preserving co-located `DEL` events.
    """
    b = SyntheticInputBuilder(tmp_path / "t2_inputs", chrom="Chr01", backbone_prefix="Ref#1#")

    # Reference: 14 nodes x 1,000 bp = 14 kb
    for i in range(1, 15):
        b.add_node(str(i), _fast_complex_dna(1000, seed=2100 + i))

    body_50k = _fast_complex_dna(50_000, seed=2200)
    # 96% similar variants (2 SNPs / 50 bp = 48/50 = 0.96 >= 0.90)
    body_50k_v1 = _mutate_per_50bp_block(body_50k, (5, 35), seed=2201)
    body_50k_v2 = _mutate_per_50bp_block(body_50k, (6, 36), seed=2202)
    body_50k_v3 = _mutate_per_50bp_block(body_50k, (7, 37), seed=2203)
    priv_2k = _fast_complex_dna(2_000, seed=2204)
    body_52k_priv = (
        _mutate_per_50bp_block(body_50k[:25_000], (8, 38), seed=2205)
        + priv_2k
        + _mutate_per_50bp_block(body_50k[25_000:], (8, 38), seed=2206)
    )

    b.add_node("101", body_50k)
    b.add_node("102", body_50k_v1)
    b.add_node("103", body_50k_v2)
    b.add_node("104", body_50k_v3)
    b.add_node("105", body_52k_priv)

    ref_steps = [f"{i}+" for i in range(1, 15)]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # Group 1 (highest priority: support=2, compact span=0 between nodes 6 and 7 -> ref_pos=6000)
    p_compact = [f"{i}+" for i in range(1, 7)] + ["101+"] + [f"{i}+" for i in range(7, 15)]
    b.add_path("H1a#1#Chr01", p_compact)
    b.add_path("H1b#1#Chr01", p_compact)

    # Group 2 (shifted wider span 5..8 -> deletes 6, 7; pure 50 kb homolog)
    b.add_path(
        "H2#1#Chr01",
        [f"{i}+" for i in range(1, 6)] + ["102+"] + [f"{i}+" for i in range(8, 15)],
    )
    # Group 3 (shifted wider span 4..9 -> deletes 5..8; pure 50 kb homolog)
    b.add_path(
        "H3#1#Chr01",
        [f"{i}+" for i in range(1, 5)] + ["103+"] + [f"{i}+" for i in range(9, 15)],
    )
    # Group 4 (shifted wider span 3..10 -> deletes 4..9; pure 50 kb homolog)
    b.add_path(
        "H4#1#Chr01",
        [f"{i}+" for i in range(1, 4)] + ["104+"] + [f"{i}+" for i in range(10, 15)],
    )
    # Group 5 (shifted wider span 2..11 -> deletes 3..10; 25kb hom + 2kb private novel + 25kb hom)
    b.add_path(
        "H5#1#Chr01",
        [f"{i}+" for i in range(1, 3)] + ["105+"] + [f"{i}+" for i in range(11, 15)],
    )

    inp = b.write("t2_shifted_anchors")

    # --- Run A: --dedup-mode off (~252 kb non-N inserted bases) ---
    out_off = tmp_path / "t2_off"
    proc_off, bun_off = run_build(inp, extra_args=["--dedup-mode", "off"], out_prefix=out_off)
    assert proc_off.returncode == 0 and bun_off is not None
    run_verify(out_off, inp.gfa_path, inp.ref_fa_path, check=True)
    assert_all_global_invariants(bun_off, inp.gfa_path, inp.ref_fa_path)

    non_n_off = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_off.inserts)
    assert non_n_off == 252_000

    # --- Run B: --dedup-mode pan --dedup-id 0.90 (~52 kb non-N inserted bases) ---
    out_pan = tmp_path / "t2_pan"
    proc_pan, bun_pan = run_build(
        inp,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90"],
        out_prefix=out_pan,
    )
    assert proc_pan.returncode == 0 and bun_pan is not None
    run_verify(out_pan, inp.gfa_path, inp.ref_fa_path, check=True)
    assert_all_global_invariants(bun_pan, inp.gfa_path, inp.ref_fa_path)

    non_n_pan = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_pan.inserts)
    assert 51_000 <= non_n_pan <= 53_000, f"Expected ~52 kb non-N inserted bases, got {non_n_pan}"
    assert non_n_pan == 52_000

    # 3 pure-homolog shifted groups dropped with ZERO 'N' spacers emitted at their mount steps (steps 7, 8, 9)
    assert int(bun_pan.summary_kv["dedup_dropped_groups"]) == 3
    assert int(bun_pan.summary_kv["dedup_pan_masked_bp"]) == 200_000
    assert bun_pan.offsets[7]["slot_ins_len"] == 0  # node 8 (H2's right anchor)
    assert bun_pan.offsets[8]["slot_ins_len"] == 0  # node 9 (H3's right anchor)
    assert bun_pan.offsets[9]["slot_ins_len"] == 0  # node 10 (H4's right anchor)

    # Highest-priority compact slot (before node 7 = ref_step 6) mounts the 50 kb body
    assert bun_pan.offsets[6]["slot_ins_len"] == 50_000

    # Private haplotype's locus (before node 11 = ref_step 10) mounts the 2 kb private block
    # (span > 1 so leading + trailing 100 bp spacers around the 2,000 bp block -> 2,200 bp)
    ins_events = [r for r in bun_pan.sv_rows if r["type"] == "INS"]
    assert len(ins_events) == 2
    compact_ins = next(r for r in ins_events if int(r["ref_start"]) == 6000)
    priv_ins = next(r for r in ins_events if int(r["ref_start"]) == 10000)
    assert "partial_content=1" not in str(compact_ins["alleles"])
    assert "partial_content=1" in str(priv_ins["alleles"])

    priv_pieces = [r for r in bun_pan.inserts if str(r["event_id"]) == priv_ins["event_id"]]
    assert len(priv_pieces) == 1
    p_s, p_e = int(priv_pieces[0]["pan_start"]), int(priv_pieces[0]["pan_end"])
    assert p_e - p_s == 2_000
    assert bun_pan.pan_seq[p_s:p_e].upper() == priv_2k

    # Co-located DEL events on ALL 4 wider-span haplotypes (H2..H5) MUST be preserved!
    del_events = [r for r in bun_pan.sv_rows if r["type"] == "DEL"]
    del_carriers = {
        hap
        for ev in del_events
        for hap, val in bun_pan.genotypes[ev["event_id"]].items()
        if val == "1"
    }
    assert {"H2#1", "H3#1", "H4#1", "H5#1"}.issubset(del_carriers), (
        f"Expected co-located DEL events preserved for H2..H5, got carriers {del_carriers}"
    )


# ============================================================================
# Test 3: `--qc` Closed-Loop Redundancy Validation on 22 Haplotypes + 100% Recall
# ============================================================================


def test_tier5_dedup_3_qc_closed_loop_22_haplotypes_and_100pct_recall(
    tmp_path: Path,
) -> None:
    """DEDUP_PLAN §4.2 Test 3: 22-haplotype chromosome with heavy ectopic ref-TE + multi-copy
    non-ref TE redundancy reduces `qc_redundancy_0.90_pct` from >70% (`off`) to <2% (`pan`),
    while maintaining 100% recall of all 6 planted unique single-copy novel sequences.
    """
    b = SyntheticInputBuilder(tmp_path / "t3_inputs", chrom="Chr01", backbone_prefix="Ref#1#")

    # 30 reference nodes x 1,000 bp = 30 kb reference
    n_ref = 30
    for i in range(1, n_ref + 1):
        b.add_node(str(i), _fast_complex_dna(1000, seed=3100 + i))

    ref_steps = [f"{i}+" for i in range(1, n_ref + 1)]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # 3 reference-resident TE templates (3 kb each from nodes 2..4, 5..7, 8..10)
    ref_te_a = "".join(b.nodes[str(k)] for k in range(2, 5))
    ref_te_b = "".join(b.nodes[str(k)] for k in range(5, 8))
    ref_te_c = "".join(b.nodes[str(k)] for k in range(8, 11))
    ref_tes = [ref_te_a, ref_te_b, ref_te_c]

    # 2 multi-copy non-reference TE families (3.5 kb each)
    nonref_te_1 = _fast_complex_dna(3500, seed=3201)
    nonref_te_2 = _fast_complex_dna(3500, seed=3202)

    # 6 planted unique single-copy non-reference novel sequences (>= 500 bp each)
    planted_unique_seqs = [
        _fast_complex_dna(600, seed=3301),
        _fast_complex_dna(750, seed=3302),
        _fast_complex_dna(900, seed=3303),
        _fast_complex_dna(1050, seed=3304),
        _fast_complex_dna(1200, seed=3305),
        _fast_complex_dna(1400, seed=3306),
    ]

    next_nid = 100

    def _alloc(seq: str) -> str:
        nonlocal next_nid
        nid = str(next_nid)
        next_nid += 1
        b.add_node(nid, seq)
        return nid

    def _path_with_inserts(ins_after_node: dict[int, str]) -> list[str]:
        steps: list[str] = []
        for idx in range(1, n_ref + 1):
            steps.append(f"{idx}+")
            if idx in ins_after_node:
                steps.append(f"{ins_after_node[idx]}+")
        return steps

    nonref_te_1_nid = _alloc(nonref_te_1)
    nonref_te_2_nid = _alloc(nonref_te_2)

    # Build 21 alt haplotypes (total = 22 haplotypes including Ref#1):
    # - H01..H12: ectopic 96%-identical copies of reference TEs at 12 distinct loci (12..23)
    # - H13..H17: nonref_te_1 benchmark (support=2 at locus 24) + 3 shifted 96% copies (loci 25..27)
    # - H18..H21: nonref_te_2 benchmark (support=2 at locus 1) + 2 shifted 96% copies (loci 2..3)
    # - 6 planted unique single-copy novel sequences:
    #   4 standalone (on H01, H05, H10, H15) + 2 embedded inside homologous flanks (on H18 & H21)!
    for h_idx in range(1, 22):
        ins_map: dict[int, str] = {}
        if h_idx <= 12:
            # Ectopic ref TE (3 kb, 96% identity) at locus (11 + h_idx)
            tpl = ref_tes[(h_idx - 1) % 3]
            off_a = 10 + (h_idx % 5)
            mut_te = _mutate_per_50bp_block(tpl, (off_a, off_a + 20), seed=3400 + h_idx)
            if h_idx % 2 == 0:
                mut_te = revcomp(mut_te)
            ins_map[11 + h_idx] = _alloc(mut_te)
        elif h_idx <= 14:
            # Intact nonref_te_1 benchmark (support=2 at locus 24 -> Priority 1)
            ins_map[24] = nonref_te_1_nid
        elif h_idx <= 17:
            # 3 shifted 96%-identical copies of nonref_te_1 at loci 25..27
            off_a = 10 + (h_idx % 5)
            copy1 = _mutate_per_50bp_block(nonref_te_1, (off_a, off_a + 20), seed=3500 + h_idx)
            if h_idx % 2 == 1:
                copy1 = revcomp(copy1)
            ins_map[10 + h_idx] = _alloc(copy1)
        else:
            if h_idx in (18, 19):
                # Intact nonref_te_2 benchmark (support=2 at locus 1 -> Priority 1)
                ins_map[1] = nonref_te_2_nid
            else:
                # 2 shifted 96%-identical copies of nonref_te_2 at loci 2..3
                off_a = 10 + (h_idx % 5)
                copy2 = _mutate_per_50bp_block(nonref_te_2, (off_a, off_a + 20), seed=3600 + h_idx)
                ins_map[h_idx - 18] = _alloc(copy2)

            if h_idx == 18:
                # Planted unique #5 embedded inside ectopic ref TE flanks!
                flanked_ref = (
                    _mutate_per_50bp_block(ref_te_a[:1500], (10, 30), seed=3650)
                    + planted_unique_seqs[4]
                    + _mutate_per_50bp_block(ref_te_a[1500:], (10, 30), seed=3651)
                )
                ins_map[29] = _alloc(flanked_ref)
            elif h_idx == 21:
                # Planted unique #6 embedded inside non-ref TE flanks!
                flanked_nr = (
                    _mutate_per_50bp_block(nonref_te_1[:1500], (10, 30), seed=3652)
                    + planted_unique_seqs[5]
                    + _mutate_per_50bp_block(nonref_te_1[1500:], (10, 30), seed=3653)
                )
                ins_map[10] = _alloc(flanked_nr)

        # Plant standalone unique single-copy novel sequences #1..#4 at distinct clean loci
        if h_idx == 1:
            ins_map[6] = _alloc(planted_unique_seqs[0])
        elif h_idx == 5:
            ins_map[7] = _alloc(planted_unique_seqs[1])
        elif h_idx == 10:
            ins_map[8] = _alloc(planted_unique_seqs[2])
        elif h_idx == 15:
            ins_map[9] = _alloc(planted_unique_seqs[3])

        b.add_path(f"H{h_idx:02d}#1#Chr01", _path_with_inserts(ins_map))

    inp = b.write("t3_qc_22haps")

    # --- Run A: --dedup-mode off --qc ---
    out_off = tmp_path / "t3_off"
    proc_off, bun_off = run_build(
        inp,
        extra_args=["--dedup-mode", "off", "--qc", "--threads", "2"],
        out_prefix=out_off,
    )
    assert proc_off.returncode == 0 and bun_off is not None
    assert len(bun_off.genotype_haps) == 22
    red_off = float(bun_off.summary_kv["qc_redundancy_0.90_pct"])
    assert red_off > 70.0, f"Expected >70% redundancy under --dedup-mode off, got {red_off:.2f}%"

    # --- Run B: --dedup-mode pan --dedup-id 0.90 --qc ---
    out_pan = tmp_path / "t3_pan"
    proc_pan, bun_pan = run_build(
        inp,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--qc", "--threads", "2"],
        out_prefix=out_pan,
    )
    assert proc_pan.returncode == 0 and bun_pan is not None
    assert len(bun_pan.genotype_haps) == 22

    run_verify(out_pan, inp.gfa_path, inp.ref_fa_path, check=True)
    assert_all_global_invariants(bun_pan, inp.gfa_path, inp.ref_fa_path)

    red_pan = float(bun_pan.summary_kv["qc_redundancy_0.90_pct"])
    assert red_pan < 2.0, f"Expected <2.0% redundancy under --dedup-mode pan, got {red_pan:.4f}%"
    assert int(bun_pan.summary_kv["dedup_ref_masked_bp"]) > 30_000
    assert int(bun_pan.summary_kv["dedup_pan_masked_bp"]) >= 20_000
    assert int(bun_pan.summary_kv["dedup_dropped_groups"]) == 17

    # Verify 100% recall of ALL 6 planted unique single-copy novel sequences in mounted inserts!
    pan_up = bun_pan.pan_seq.upper()
    mounted_piece_seqs = [
        pan_up[int(r["pan_start"]) : int(r["pan_end"])] for r in bun_pan.inserts
    ]
    for idx_u, u_seq in enumerate(planted_unique_seqs, start=1):
        u_rc = revcomp(u_seq)
        found = any((u_seq in p_seq) or (u_rc in p_seq) for p_seq in mounted_piece_seqs)
        assert found, f"Planted unique novel sequence #{idx_u} ({len(u_seq)} bp) lost after pan dedup!"


# ============================================================================
# Test 4: T22 Multicore Byte-Identity & Shuffled GFA Input-Order Invariance
# ============================================================================


def _build_t22_dedup_compact_int_graph(tmp_path: Path) -> InputPaths:
    """Construct a compact-integer-ID (1..N) graph exercising Modules A, B, C, CEN_ISLAND,
    repeat-guard, GFF3 (+/- genes), W-lines (incl. whole-contig reversed), and --qc.
    """
    b = SyntheticInputBuilder(tmp_path / "t4_inputs", chrom="Chr01", backbone_prefix="Ref#1#")

    next_id = 1
    nid_of: dict[str, str] = {}

    def _alloc(tag: str, seq: str) -> str:
        nonlocal next_id
        nid = str(next_id)
        next_id += 1
        nid_of[tag] = nid
        b.add_node(nid, seq)
        return nid

    plus_cds = make_coding_seq(num_codons=120, seed=4101)  # 360 bp
    minus_cds = make_coding_seq(num_codons=120, seed=4102)  # 360 bp
    minus_rc = revcomp(minus_cds)

    # Reference nodes r1..r20 (compact integer IDs 1..20)
    # r2: 1200 bp ref TE; r5..r7: + gene; r10..r11: centromere (2000 bp); r14..r16: - gene
    for i in range(1, 21):
        if i == 2:
            _alloc(f"r{i}", make_complex_seq(1200, seed=4200 + i))
        elif i == 5:
            _alloc(f"r{i}", make_complex_seq(20, seed=4205) + plus_cds[:180])
        elif i == 6:
            _alloc(f"r{i}", make_complex_seq(300, seed=4206))
        elif i == 7:
            _alloc(f"r{i}", plus_cds[180:] + make_complex_seq(20, seed=4207))
        elif i in (10, 11):
            _alloc(f"r{i}", make_complex_seq(1000, seed=4200 + i))
        elif i == 14:
            _alloc(f"r{i}", make_complex_seq(20, seed=4214) + minus_rc[:180])
        elif i == 15:
            _alloc(f"r{i}", make_complex_seq(300, seed=4215))
        elif i == 16:
            _alloc(f"r{i}", minus_rc[180:] + make_complex_seq(20, seed=4216))
        else:
            _alloc(f"r{i}", make_complex_seq(300, seed=4200 + i))

    ref_te = b.nodes[nid_of["r2"]]
    cen_r1 = b.nodes[nid_of["r10"]]
    cen_r2 = b.nodes[nid_of["r11"]]

    # Module A nodes: pure ectopic TE + TE-flanked novel island
    mod_a_pure = _alloc("mod_a_pure", _mutate_per_50bp_block(ref_te, (6, 7, 43), seed=4301))
    mod_a_flanked = _alloc(
        "mod_a_flanked",
        _mutate_per_50bp_block(ref_te[:600], (6, 7, 43), seed=4302)
        + make_complex_seq(600, seed=4303)
        + _mutate_per_50bp_block(ref_te[600:], (6, 7, 43), seed=4304),
    )

    # Module B nodes: childless secondary backbone (20 bp bubble) + 99% similar real node
    mb_fl = _alloc("mb_fl", make_complex_seq(250, seed=4310))
    mb_b1 = _alloc("mb_b1", make_complex_seq(20, seed=4311))
    mb_b2 = _alloc("mb_b2", make_complex_seq(20, seed=4312))
    mb_fr = _alloc("mb_fr", make_complex_seq(250, seed=4313))
    mb_bsec_seq = b.nodes[mb_fl] + b.nodes[mb_b1] + b.nodes[mb_fr]
    mb_real99 = _alloc("mb_real99", mutate_seq_snps(mb_bsec_seq, positions=[40, 140, 320, 460], seed=4314))

    # Centromere nodes: CEN_ISLAND (700 bp), Step B ref-subtracted + island, and satellite
    cen_isl_seq = make_complex_seq(700, seed=4320)
    cen_isl = _alloc("cen_isl", cen_isl_seq)
    cen_step_b = _alloc(
        "cen_step_b",
        cen_r1[100:600] + make_complex_seq(650, seed=4321) + revcomp(cen_r2[100:600]),
    )
    cen_sat = _alloc("cen_sat", make_satellite_seq(unit_len=20, total_len=1500, seed=4322))

    # Module C nodes: non-ref sequence duplicated across shifted groups + normal-arm copy of CEN_ISLAND
    pan_base_seq = make_complex_seq(1200, seed=4330)
    pan_base = _alloc("pan_base", pan_base_seq)
    pan_dup_shift = _alloc("pan_dup_shift", _mutate_per_50bp_block(pan_base_seq, (5, 35), seed=4331))
    pan_dup_priv = _alloc(
        "pan_dup_priv",
        _mutate_per_50bp_block(pan_base_seq[:600], (6, 36), seed=4332)
        + make_complex_seq(350, seed=4333)
        + _mutate_per_50bp_block(pan_base_seq[600:], (6, 36), seed=4334),
    )
    pan_cen_copy = _alloc("pan_cen_copy", _mutate_per_50bp_block(cen_isl_seq, (5, 35), seed=4335))
    pure_sat = _alloc("pure_sat", make_satellite_seq(unit_len=20, total_len=1500, seed=4336))
    rev_nov = _alloc("rev_nov", make_complex_seq(450, seed=4337))

    def _s(tag: str, orient: str = "+") -> tuple[str, str]:
        return (nid_of[tag], orient)

    def _rng(lo: int, hi: int) -> list[tuple[str, str]]:
        return [_s(f"r{k}") for k in range(lo, hi + 1)]

    ref_steps = _rng(1, 20)
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    ref_offsets: dict[str, tuple[int, int]] = {}
    cur = 0
    for nid, _ in ref_steps:
        ln = len(b.nodes[nid])
        ref_offsets[nid] = (cur, cur + ln)
        cur += ln

    # Centromere BED: r10..r11 (anchors u=r9, v=r12)
    b.add_bed_region(ref_offsets[nid_of["r10"]][0], ref_offsets[nid_of["r11"]][1], region_id="CEN_T22")

    # GFF3 genes (+ on r5..r7, - on r14..r16)
    r5_s, r5_e = ref_offsets[nid_of["r5"]]
    r7_s, _ = ref_offsets[nid_of["r7"]]
    b.add_gff_gene(
        gene_id="t22_gene_plus",
        mrna_id="t22_mrna_plus",
        strand="+",
        exons=[(r5_s + 21, r5_e), (r7_s + 1, r7_s + 180)],
        cds_list=[(r5_s + 21, r5_e, 0), (r7_s + 1, r7_s + 180, 0)],
    )
    r14_s, r14_e = ref_offsets[nid_of["r14"]]
    r16_s, _ = ref_offsets[nid_of["r16"]]
    b.add_gff_gene(
        gene_id="t22_gene_minus",
        mrna_id="t22_mrna_minus",
        strand="-",
        exons=[(r14_s + 21, r14_e), (r16_s + 1, r16_s + 180)],
        cds_list=[(r14_s + 21, r14_e, 0), (r16_s + 1, r16_s + 180, 0)],
    )

    # H1: Mod A flanked (r5..r6), Mod B b1 (r7..r8), CEN_ISLAND, pan_base high-support (r13..r14)
    b.add_path(
        "H1#1#Chr01",
        [
            *_rng(1, 5),
            (mod_a_flanked, "+"),
            _s("r6"),
            _s("r7"),
            (mb_fl, "+"),
            (mb_b1, "+"),
            (mb_fr, "+"),
            _s("r8"),
            _s("r9"),
            (cen_isl, "+"),
            _s("r12"),
            _s("r13"),
            (pan_base, "+"),
            *_rng(14, 20),
        ],
    )
    # H2: Mod A pure TE (r5..r6), Mod B b2 (r7..r8), CEN Step B, pan_base high-support (r13..r14)
    b.add_path(
        "H2#1#Chr01",
        [
            *_rng(1, 5),
            (mod_a_pure, "+"),
            _s("r6"),
            _s("r7"),
            (mb_fl, "+"),
            (mb_b2, "+"),
            (mb_fr, "+"),
            _s("r8"),
            _s("r9"),
            (cen_step_b, "+"),
            _s("r12"),
            _s("r13"),
            (pan_base, "+"),
            *_rng(14, 20),
        ],
    )
    # H3: Mod B real99 (r7..r8), CEN satellite, pan_dup_shift (r15..r16), pan_cen_copy (r17..r18)
    b.add_path(
        "H3#1#Chr01",
        [
            *_rng(1, 7),
            (mb_real99, "+"),
            _s("r8"),
            _s("r9"),
            (cen_sat, "+"),
            *_rng(12, 15),
            (pan_dup_shift, "+"),
            _s("r16"),
            _s("r17"),
            (pan_cen_copy, "+"),
            *_rng(18, 20),
        ],
    )
    # H4 (W-line): repeat-guard pure satellite (r3..r4) + pan_dup_priv (r16..r17)
    h4_steps = [
        *_rng(1, 3),
        (pure_sat, "+"),
        *_rng(4, 16),
        (pan_dup_priv, "+"),
        *_rng(17, 20),
    ]
    b.add_walk("H4", "1", "ctgH4", start=200, end=None, steps=h4_steps)

    # H5 (WHOLE-CONTIG REVERSED W-line spanning centromere): cen_isl (r9..r12) + novel ins (r18..r19)
    h5_fwd = [
        *_rng(1, 9),
        (cen_isl, "+"),
        *_rng(12, 18),
        (rev_nov, "+"),
        _s("r19"),
        _s("r20"),
    ]
    b.add_walk("H5", "1", "ctgH5_rev", start=400, end=None, steps=_reverse_steps(h5_fwd))

    return b.write("t4_t22_dedup")


def _write_shuffled_pw_gfa(src_gfa: Path, dst_gfa: Path, seed: int = 20261125) -> None:
    """Create a GFA with identical H/S/L lines and deterministically shuffled P/W lines."""
    lines = src_gfa.read_text(encoding="utf-8").splitlines()
    head_sl: list[str] = []
    pw_lines: list[str] = []
    for line in lines:
        if line.startswith("P\t") or line.startswith("W\t"):
            pw_lines.append(line)
        else:
            head_sl.append(line)

    rng = random.Random(seed)
    shuffled = list(pw_lines)
    rng.shuffle(shuffled)
    # Guarantee the shuffled order genuinely differs from the original order
    if shuffled == pw_lines:
        shuffled = list(reversed(pw_lines))
    assert shuffled != pw_lines

    dst_gfa.write_text("\n".join(head_sl + shuffled) + "\n", encoding="utf-8")


def test_tier5_dedup_4_t22_multicore_and_shuffled_gfa_invariance(
    tmp_path: Path,
) -> None:
    """DEDUP_PLAN §4.2 Test 4: T22 12-file SHA-256 byte-identity (`--threads 1` vs `--threads 4`
    with `PANFLAT_FORCE_FORK=1`) and shuffled `P`/`W` GFA input-order invariance.
    """
    orig_inp = _build_t22_dedup_compact_int_graph(tmp_path)
    assert orig_inp.bed_path is not None and orig_inp.gff_path is not None

    shuf_gfa = tmp_path / "t4_inputs" / "t4_t22_dedup_shuffled.gfa"
    _write_shuffled_pw_gfa(orig_inp.gfa_path, shuf_gfa)
    shuf_inp = InputPaths(
        gfa_path=shuf_gfa,
        ref_fa_path=orig_inp.ref_fa_path,
        bed_path=orig_inp.bed_path,
        gff_path=orig_inp.gff_path,
        out_prefix=tmp_path / "t4_shuf_out",
        chrom=orig_inp.chrom,
        backbone_prefix=orig_inp.backbone_prefix,
    )

    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    common_args = [
        "--dedup-mode",
        "pan",
        "--dedup-id",
        "0.90",
        "--centromere-mode",
        "extract-islands",
        "--centromere-min-island",
        "500",
        "--dump-repeat-sequences",
        "--qc",
    ]

    out_orig_t1 = tmp_path / "t4_orig_t1"
    out_orig_t4 = tmp_path / "t4_orig_t4"
    out_shuf_t1 = tmp_path / "t4_shuf_t1"
    out_shuf_t4 = tmp_path / "t4_shuf_t4"

    proc_o1, bun_o1 = run_build(
        orig_inp, extra_args=[*common_args, "--threads", "1"], out_prefix=out_orig_t1, extra_env=env
    )
    proc_o4, bun_o4 = run_build(
        orig_inp, extra_args=[*common_args, "--threads", "4"], out_prefix=out_orig_t4, extra_env=env
    )
    proc_s1, bun_s1 = run_build(
        shuf_inp, extra_args=[*common_args, "--threads", "1"], out_prefix=out_shuf_t1, extra_env=env
    )
    proc_s4, bun_s4 = run_build(
        shuf_inp, extra_args=[*common_args, "--threads", "4"], out_prefix=out_shuf_t4, extra_env=env
    )
    assert bun_o1 is not None and bun_o4 is not None and bun_s1 is not None and bun_s4 is not None

    # Subcheck A: same GFA --threads 1 vs --threads 4 -> ALL 12 files SHA-256 byte-identical
    _assert_12_files_sha256_identical_and_clean_streams(out_orig_t1, out_orig_t4, proc_o1, proc_o4)
    # Subcheck B1: shuffled GFA --threads 1 vs --threads 4 -> ALL 12 files SHA-256 byte-identical
    _assert_12_files_sha256_identical_and_clean_streams(out_shuf_t1, out_shuf_t4, proc_s1, proc_s4)

    # Subcheck B2: original --threads 1 vs shuffled --threads 4 ->
    # ALL 10 biological output files SHA-256 byte-identical + .summary.tsv / .log identical after
    # normalizing ONLY gfa_path and gfa_sha256!
    for suffix in _BIO_10_OUTPUT_SUFFIXES:
        ha = _sha256_file(Path(f"{out_orig_t1}{suffix}"))
        hb = _sha256_file(Path(f"{out_shuf_t4}{suffix}"))
        assert ha == hb, f"Shuffled-GFA biological mismatch on '{suffix}': {ha} != {hb}"

    orig_sha = _sha256_file(orig_inp.gfa_path)
    shuf_sha = _sha256_file(shuf_inp.gfa_path)
    assert orig_sha != shuf_sha
    for meta_suffix in (".summary.tsv", ".log"):
        txt_o = _normalize_gfa_meta_text(
            Path(f"{out_orig_t1}{meta_suffix}").read_text(encoding="utf-8"),
            orig_inp.gfa_path,
            orig_sha,
        )
        txt_s = _normalize_gfa_meta_text(
            Path(f"{out_shuf_t4}{meta_suffix}").read_text(encoding="utf-8"),
            shuf_inp.gfa_path,
            shuf_sha,
        )
        assert txt_o == txt_s, f"Shuffled-GFA normalized metadata mismatch on '{meta_suffix}'"

    # Independent CLI verify + global/centromere/GFF3 oracles on original & shuffled builds
    for prefix, gfa_p in ((out_orig_t4, orig_inp.gfa_path), (out_shuf_t4, shuf_inp.gfa_path)):
        run_verify(
            prefix,
            gfa_p,
            orig_inp.ref_fa_path,
            extra_args=["--centromere-bed", orig_inp.bed_path, "--ref-gff", orig_inp.gff_path],
            check=True,
        )
    assert_all_global_invariants(bun_o4, orig_inp.gfa_path, orig_inp.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_o4)
    assert_gff3_liftover_integrity(
        orig_inp.ref_fa_path, orig_inp.gff_path, bun_o4.pan_fa_path, bun_o4.lifted_gff_path
    )

    # Non-vacuous check: Modules A, B, C, CEN_ISLAND, and --qc all genuinely fired
    assert 2300 <= int(bun_o4.summary_kv["dedup_ref_masked_bp"]) <= 2400
    assert int(bun_o4.summary_kv["dedup_pan_masked_bp"]) == 3100  # 1200 + 1200 + 700
    assert int(bun_o4.summary_kv["dedup_dropped_groups"]) == 2  # pan_dup_shift + pan_cen_copy
    assert float(bun_o4.summary_kv["qc_redundancy_0.90_pct"]) < 2.0
    assert any(r["type"] == "CEN_ISLAND" for r in bun_o4.sv_rows)


# ============================================================================
# Test 5: Subprocess CLI Validation & Multi-Chromosome Associative Merge E2E
# ============================================================================


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--dedup-mode", "bogus"],
        ["--dedup-id", "0.79"],
        ["--dedup-id", "0.995"],
        ["--dedup-id", "nan"],
        ["--dedup-id", "inf"],
        ["--dedup-id", "-inf"],
    ],
)
def test_tier5_dedup_5a_cli_validation_rejects_invalid_args(
    tmp_path: Path,
    bad_args: list[str],
) -> None:
    """CLI rejects invalid `--dedup-mode` / `--dedup-id` cleanly with non-zero exit & no traceback."""
    b = SyntheticInputBuilder(tmp_path / "cli_val", chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("1", make_complex_seq(200, seed=5101))
    b.add_node("2", make_complex_seq(200, seed=5102))
    b.set_ref_path("Ref#1#Chr01", ["1+", "2+"])
    inp = b.write("val")

    proc, bun = run_build(inp, extra_args=bad_args, check=False)
    assert proc.returncode != 0
    assert bun is None
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip() != ""


def test_tier5_dedup_5b_multi_chrom_merge_associative_cli(tmp_path: Path) -> None:
    """Subprocess `panflat merge` sums `dedup_*` counters accurately and associatively across 3 chroms."""
    chrom_prefixes: list[Path] = []
    expected_ref = 0
    expected_pan = 0
    expected_short = 0
    expected_drop = 0

    for idx, chrom in enumerate(("Chr01", "Chr02", "Chr03"), start=1):
        seed_base = 5200 + idx * 30
        b = SyntheticInputBuilder(tmp_path / f"m_{chrom}", chrom=chrom, backbone_prefix="Ref#1#")
        r1 = make_complex_seq(400, seed=seed_base + 1)
        r2_te = make_complex_seq(800, seed=seed_base + 2)
        r3 = make_complex_seq(400, seed=seed_base + 3)
        r4 = make_complex_seq(400, seed=seed_base + 4)
        r5 = make_complex_seq(400, seed=seed_base + 5)

        te_short = _mutate_per_50bp_block(r2_te, (6, 7, 43), seed=seed_base + 6) + make_complex_seq(
            20, seed=seed_base + 7
        )
        pan_nov = make_complex_seq(600, seed=seed_base + 8)
        pan_dup = _mutate_per_50bp_block(pan_nov, (5, 35), seed=seed_base + 9)

        for nid, seq in [
            ("1", r1),
            ("2", r2_te),
            ("3", r3),
            ("4", r4),
            ("5", r5),
            ("6", te_short),
            ("7", pan_nov),
            ("8", pan_dup),
        ]:
            b.add_node(nid, seq)

        b.set_ref_path(f"Ref#1#{chrom}", ["1+", "2+", "3+", "4+", "5+"])
        b.add_path(f"H1#1#{chrom}", ["1+", "6+", "2+", "3+", "7+", "4+", "5+"])
        b.add_path(f"H2#1#{chrom}", ["1+", "2+", "3+", "7+", "4+", "8+", "5+"])
        inp = b.write(chrom)

        proc, bun = run_build(
            inp, extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--qc"]
        )
        assert proc.returncode == 0 and bun is not None
        run_verify(inp.out_prefix, inp.gfa_path, inp.ref_fa_path, check=True)
        chrom_prefixes.append(inp.out_prefix)

        expected_ref += int(bun.summary_kv["dedup_ref_masked_bp"])
        expected_pan += int(bun.summary_kv["dedup_pan_masked_bp"])
        expected_short += int(bun.summary_kv["dedup_short_residual_bp"])
        expected_drop += int(bun.summary_kv["dedup_dropped_groups"])

    assert expected_ref == 2400
    assert expected_pan == 1800
    assert expected_short == 60
    assert expected_drop == 6

    # Associative merge via CLI: flat vs left-nested vs right-nested
    _, bun_flat = run_merge(chrom_prefixes, tmp_path / "merged_flat")
    _, bun_12 = run_merge(chrom_prefixes[:2], tmp_path / "merged_12")
    _, bun_left = run_merge([tmp_path / "merged_12", chrom_prefixes[2]], tmp_path / "merged_left")
    _, bun_23 = run_merge(chrom_prefixes[1:], tmp_path / "merged_23")
    _, bun_right = run_merge([chrom_prefixes[0], tmp_path / "merged_23"], tmp_path / "merged_right")
    assert bun_flat and bun_12 and bun_left and bun_23 and bun_right

    assert bun_flat.summary_text == bun_left.summary_text == bun_right.summary_text

    # Confirm 0 duplicate keys & exact aggregated values
    raw_keys = [
        line.split("\t", 1)[0]
        for line in bun_flat.summary_text.splitlines()
        if line.strip() and line.strip() != "metric\tvalue"
    ]
    assert len(raw_keys) == len(set(raw_keys))
    assert bun_flat.summary_kv["dedup_mode"] == "pan"
    assert bun_flat.summary_kv["dedup_id"] == "0.9"
    assert int(bun_flat.summary_kv["dedup_ref_masked_bp"]) == expected_ref
    assert int(bun_flat.summary_kv["dedup_pan_masked_bp"]) == expected_pan
    assert int(bun_flat.summary_kv["dedup_short_residual_bp"]) == expected_short
    assert int(bun_flat.summary_kv["dedup_dropped_groups"]) == expected_drop
    assert bun_flat.summary_kv["qc_enabled"] == "true"


def test_tier5_dedup_6_synthetic_rep_compaction_and_depth2_cli(tmp_path: Path) -> None:
    """Opaque-box CLI regression covering synthetic-rep Stage 5.5 compaction & depth-2 attribution:
    - Probe 1C: `b_sec` 100% dropped while child survives -> `Hap1="."`, 1 allele, `PAN_HOMOLOG_SUBTRACTED`.
    - Probe 2C & Scenario 5.3A: child dropped or trimmed `-300 bp` -> `Hap1` clean (no `partial_content=1`),
      `Hap2` exact surviving length `3300 <= 3600` (no inflation).
    - Scenario 5.3B: 100% of `Hap2` subtracted while `Hap1`'s replaced `b_sec` segment survives ->
      `Hap2="."`, 1 allele, `members=Hap1#1`.
    - Probe 5d (depth-2 nested): only `Hap3`'s depth-2 piece `Y` survives -> `Hap1a..c="."`, `Hap2="."`,
      `Hap3="1"`, `allele=1`, `members=Hap3#1`.
    """
    hom_l = make_complex_seq(1500, seed=5301)
    hom_m = make_complex_seq(1000, seed=5302)
    hom_r = make_complex_seq(1500, seed=5303)
    nov_13 = make_complex_seq(600, seed=5304)
    part_hom = make_complex_seq(300, seed=5305)
    part_nov = make_complex_seq(300, seed=5306)
    ref6 = [f"{i}+" for i in range(1, 7)]

    def _build_two_hap_synth(
        tag: str, sup_seq: str, node13_seq: str, seed_off: int
    ) -> tuple[InputPaths, object]:
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=seed_off + i))
        b.add_node("10", sup_seq)
        b.add_node("11", hom_l)
        b.add_node("14", hom_m)
        b.add_node("12", hom_r)
        b.add_node("13", node13_seq)
        b.set_ref_path("Ref#1#Chr01", ref6)
        for k in range(1, 5):
            b.add_path(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"])
        b.add_path("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "12+", "5+", "6+"])
        b.add_path("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "13+", "12+", "5+", "6+"])
        inp = b.write(tag)
        proc, bun = run_build(inp, extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90"])
        assert proc.returncode == 0 and bun is not None
        run_verify(inp.out_prefix, inp.gfa_path, inp.ref_fa_path, check=True)
        return inp, bun

    # (1) Probe 1C: b_sec (11+14+12) 100% dropped, child 13 survives
    _, bun1c = _build_two_hap_synth("cli_probe1c", hom_l + hom_m + hom_r, nov_13, 5310)
    ev1c = next(r for r in bun1c.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    gt1c = bun1c.genotypes[ev1c["event_id"]]
    ins1c = [r for r in bun1c.inserts if r["event_id"] == ev1c["event_id"]]
    assert ev1c["alleles"] == "1:len=600,src=Hap2#1,n=1,partial_content=1"
    assert gt1c["Ref#1"] == "0" and gt1c["Hap1#1"] == "." and gt1c["Hap2#1"] == "1"
    assert len(ins1c) == 1 and ins1c[0]["allele"] == "1" and ins1c[0]["members"] == "Hap2#1"
    h1_logs = [
        json.loads(line.split("\t", 1)[1])
        for line in bun1c.log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("CANDIDATE_JSON\t")
    ]
    h1_logs = [e for e in h1_logs if e.get("hap") == "Hap1#1"]
    assert h1_logs and all(e["final_reason"] == "PAN_HOMOLOG_SUBTRACTED" for e in h1_logs)

    # (2) Probe 2C: child 13 dropped, b_sec untouched -> Hap1 clean (no partial_content=1)
    _, bun2c = _build_two_hap_synth("cli_probe2c", nov_13, nov_13, 5320)
    ev2c = next(r for r in bun2c.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert ev2c["alleles"] == "1:len=4000,src=Hap1#1,n=1;2:len=3000,src=Hap2#1,n=1,partial_content=1"

    # (3) Scenario 5.3A: child 13 trimmed -300 bp -> Hap1 clean 4000 bp, Hap2 exact 3300 bp (<= 3600)
    inp3a, bun3a = _build_two_hap_synth("cli_scen53a", part_hom, part_hom + part_nov, 5330)
    _, bun3a_off = run_build(
        inp3a, extra_args=["--dedup-mode", "off"], out_prefix=tmp_path / "cli_scen53a_off"
    )
    assert bun3a_off is not None
    ev3a_off = next(
        r for r in bun3a_off.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"
    )
    ev3a_pan = next(
        r for r in bun3a.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"
    )
    assert ev3a_off["alleles"] == "1:len=4000,src=Hap1#1,n=1;2:len=3600,src=Hap2#1,n=1"
    assert ev3a_pan["alleles"] == "1:len=4000,src=Hap1#1,n=1;2:len=3300,src=Hap2#1,n=1,partial_content=1"

    # (4) Scenario 5.3B: 100% of Hap2 subtracted, only Hap1's replaced b_sec segment 14 survives
    _, bun3b = _build_two_hap_synth("cli_scen53b", hom_l + nov_13 + hom_r, nov_13, 5340)
    ev3b = next(r for r in bun3b.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    gt3b = bun3b.genotypes[ev3b["event_id"]]
    ins3b = [r for r in bun3b.inserts if r["event_id"] == ev3b["event_id"]]
    assert ev3b["alleles"] == "1:len=1000,src=Hap1#1,n=1,partial_content=1"
    assert gt3b["Ref#1"] == "0" and gt3b["Hap1#1"] == "1" and gt3b["Hap2#1"] == "."
    assert len(ins3b) == 1 and ins3b[0]["allele"] == "1" and ins3b[0]["members"] == "Hap1#1"

    # (5) Probe 5d: depth-2 nested synthetic rep, only Hap3's private depth-2 piece Y survives
    b5d = SyntheticInputBuilder(tmp_path / "cli_probe5d", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 5):
        b5d.add_node(str(i), make_complex_seq(400, seed=5350 + i))
    s_a = make_complex_seq(1000, seed=5360)
    s_d = make_complex_seq(3000, seed=5361)
    s_b = make_complex_seq(1000, seed=5362)
    s_x1 = make_complex_seq(1000, seed=5363)
    s_d2 = make_complex_seq(1000, seed=5364)
    s_x2 = make_complex_seq(500, seed=5365)
    s_y = make_complex_seq(600, seed=5366)
    for nid, sq in [
        ("10", s_a + s_d + s_b + s_x1 + s_d2 + s_x2),
        ("11", s_a),
        ("12", s_d),
        ("13", s_b),
        ("14", s_x1),
        ("15", s_d2),
        ("16", s_x2),
        ("17", s_y),
    ]:
        b5d.add_node(nid, sq)
    b5d.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "4+"])
    for k in range(1, 7):
        b5d.add_path(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+"])
    for tag_h in ("Hap1a", "Hap1b", "Hap1c"):
        b5d.add_path(f"{tag_h}#1#Chr01", ["1+", "2+", "3+", "11+", "12+", "13+", "4+"])
    b5d.add_path("Hap2#1#Chr01", ["1+", "2+", "3+", "11+", "14+", "15+", "16+", "13+", "4+"])
    b5d.add_path("Hap3#1#Chr01", ["1+", "2+", "3+", "11+", "14+", "17+", "16+", "13+", "4+"])
    inp5d = b5d.write("cli_probe5d")
    proc5d, bun5d = run_build(inp5d, extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90"])
    assert proc5d.returncode == 0 and bun5d is not None
    run_verify(inp5d.out_prefix, inp5d.gfa_path, inp5d.ref_fa_path, check=True)
    ev5d = next(r for r in bun5d.sv_rows if r["type"] == "INS" and r["anchors"] == "2..3")
    gt5d = bun5d.genotypes[ev5d["event_id"]]
    ins5d = [r for r in bun5d.inserts if r["event_id"] == ev5d["event_id"]]
    assert ev5d["alleles"] == "1:len=600,src=Hap3#1,n=1,partial_content=1"
    assert gt5d["Ref#1"] == "0"
    assert gt5d["Hap1a#1"] == "." and gt5d["Hap1b#1"] == "." and gt5d["Hap1c#1"] == "."
    assert gt5d["Hap2#1"] == "." and gt5d["Hap3#1"] == "1"
    assert len(ins5d) == 1 and ins5d[0]["allele"] == "1" and ins5d[0]["members"] == "Hap3#1"


def test_tier5_dedup_7_protected_ins_non50_min_sv_len_cli(tmp_path: Path) -> None:
    """Opaque-box CLI regression for Bug #1: 55 bp novel insertion at [495, 550) flanked by
    reference homology MUST survive intact at `--min-sv-len 55` under `--dedup-mode ref` and `pan`.
    """
    b = SyntheticInputBuilder(tmp_path / "cli_bug1", chrom="Chr01", backbone_prefix="Ref#1#")
    r1 = make_complex_seq(500, seed=5401)
    r2_te = make_complex_seq(1200, seed=5402)
    r3 = make_complex_seq(500, seed=5403)
    r4 = make_complex_seq(500, seed=5404)
    novel_55 = make_complex_seq(55, seed=5405)
    ectopic_node = r2_te[100:595] + novel_55 + r2_te[595:1095]  # ins at [495, 550)

    for nid, seq in [
        ("1", r1),
        ("2", r2_te),
        ("3", r3),
        ("4", r4),
        ("5", ectopic_node),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "4+"])
    b.add_path("H1#1#Chr01", ["1+", "2+", "3+", "5+", "4+"])
    inp = b.write("cli_bug1")

    for mode in ("ref", "pan"):
        out_p = tmp_path / f"cli_bug1_{mode}"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", mode, "--dedup-id", "0.90", "--min-sv-len", "55"],
            out_prefix=out_p,
        )
        assert proc.returncode == 0 and bun is not None
        run_verify(out_p, inp.gfa_path, inp.ref_fa_path, check=True)
        assert len(bun.inserts) == 1
        assert bun.inserts[0]["pan_end"] - bun.inserts[0]["pan_start"] == 55
        assert novel_55 in bun.pan_seq
        assert int(bun.summary_kv["dedup_dropped_groups"]) == 0


def test_tier5_dedup_8_iter3_repeat_guard_44m6i_and_l4_cli(tmp_path: Path) -> None:
    """Opaque-box CLI regression for Iter 3 RG-1..3:
    - `44M+6I` (`494 bp ref_L + 55 bp nov55 + 500 bp ref_R`, including 1-bp shift
      `nov55[0] == ref_R[0]`) on BOTH `+` and `-` strands at `--min-sv-len 50` and `55`.
    - `<= 500 bp` L4 (`245 bp ref_L + 55 bp nov55 + 76 bp ref_R`, 376 bp total) on BOTH
      `+` and `-` strands at `--min-sv-len 50` and `55`.
    - Verified under both `--dedup-mode ref` and `--dedup-mode pan` with `panflat verify`.
    """
    r1 = make_complex_seq(500, seed=5801)
    r2_te = make_complex_seq(1200, seed=5802)
    r3 = make_complex_seq(500, seed=5803)
    r4 = make_complex_seq(500, seed=5804)
    nov55_base = make_complex_seq(55, seed=5805)
    # Deliberately set nov55[0] == r2_te[594] (1-bp shift test on 44M+6I)
    nov55_shift = r2_te[594] + nov55_base[1:]
    node_44m6i_fwd = r2_te[100:594] + nov55_shift + r2_te[594:1094]
    node_44m6i_rc = revcomp(node_44m6i_fwd)

    nov55_l4 = make_complex_seq(55, seed=5806)
    node_l4_fwd = r2_te[100:345] + nov55_l4 + r2_te[345:421]  # 376 bp <= 500 bp
    node_l4_rc = revcomp(r2_te[100:176] + nov55_l4 + r2_te[176:421])  # 376 bp <= 500 bp

    cases = [
        ("rg_44m6i_fwd", node_44m6i_fwd, nov55_shift),
        ("rg_44m6i_rc", node_44m6i_rc, revcomp(nov55_shift)),
        ("rg_l4_fwd", node_l4_fwd, nov55_l4),
        ("rg_l4_rc", node_l4_rc, revcomp(nov55_l4)),
    ]
    for tag, alt_seq, expected_nov in cases:
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, sq in [("1", r1), ("2", r2_te), ("3", r3), ("4", r4), ("5", alt_seq)]:
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "4+"])
        b.add_path("H1#1#Chr01", ["1+", "2+", "3+", "5+", "4+"])
        inp = b.write(tag)

        for min_sv in ("50", "55"):
            for mode in ("ref", "pan"):
                out_p = tmp_path / f"{tag}_{mode}_sv{min_sv}"
                proc, bun = run_build(
                    inp,
                    extra_args=[
                        "--dedup-mode",
                        mode,
                        "--dedup-id",
                        "0.90",
                        "--min-sv-len",
                        min_sv,
                    ],
                    out_prefix=out_p,
                )
                assert proc.returncode == 0 and bun is not None
                run_verify(out_p, inp.gfa_path, inp.ref_fa_path, check=True)
                assert len(bun.inserts) == 1, f"{tag} {mode} sv={min_sv}: {bun.inserts}"
                assert bun.inserts[0]["pan_end"] - bun.inserts[0]["pan_start"] >= 55
                assert expected_nov in bun.pan_seq, f"{tag} {mode} sv={min_sv}: lost nov55"
                assert int(bun.summary_kv["dedup_dropped_groups"]) == 0


def test_tier5_dedup_9_iter3_bsec_pure_del_childless_and_hetero_merge_cli(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI regression for Iter 3 LN-1..5:
    - Probe A1 (only 14 + trimmed -300 bp no L2038 collision) + `--dedup-mode off` parity.
    - Probe F1 (untouched `ref` & trimmed `pan` -> Hap2 exact 3500 clean).
    - Probe A5 (childless prefix del -> Hap2=".", members=Hap1#1,Hap3#1).
    - Loophole #2A (1 bp SNP replaced by 600 bp ins -> zero 1 bp orphan, dropped_groups=1).
    - Probe 1 (Merge Case M2 `rc_long=4010` + `rc_hetero` in `ref` & `pan`).
    - Probe 3B (`L2639` exact-seq merge of `("START", 0)` & `(0, 1)` -> zero `TypeError`).
    """
    s11 = make_complex_seq(1500, seed=5901)
    s14 = make_complex_seq(1000, seed=5902)
    s15 = make_complex_seq(500, seed=5903)
    s12 = make_complex_seq(1500, seed=5904)
    s13 = make_complex_seq(600, seed=5905)
    ref6 = [f"{i}+" for i in range(1, 7)]

    def _run_cli(
        tag: str,
        nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        mode: str = "pan",
    ) -> tuple[InputPaths, object]:
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=5910 + i))
        for nid, sq in nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ref6)
        for pname, psteps in paths:
            b.add_path(pname, psteps)
        inp = b.write(tag)
        out_p = tmp_path / f"{tag}_{mode}_out"
        proc, bun = run_build(
            inp, extra_args=["--dedup-mode", mode, "--dedup-id", "0.90"], out_prefix=out_p
        )
        assert proc.returncode == 0 and bun is not None
        run_verify(out_p, inp.gfa_path, inp.ref_fa_path, check=True)
        return inp, bun

    sup4 = [
        (f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"])
        for k in range(1, 5)
    ]

    # (1) Probe A1 (only 14 + trimmed -300 + --dedup-mode off parity)
    a1_paths = sup4 + [
        ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "13+", "12+", "5+", "6+"]),
    ]
    _, bun_a1 = _run_cli(
        "e2e_a1_only14",
        {"10": s11 + s13 + s12, "11": s11, "14": s14, "12": s12, "13": s13},
        a1_paths,
    )
    ev = next(r for r in bun_a1.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    gt = bun_a1.genotypes[ev["event_id"]]
    ins = [r for r in bun_a1.inserts if r["event_id"] == ev["event_id"]]
    assert ev["alleles"] == "1:len=1000,src=Hap1#1,n=1,partial_content=1"
    assert gt["Hap1#1"] == "1" and gt["Hap2#1"] == "." and gt["Hap3#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1"
    h2_logs = [
        json.loads(line.split("\t", 1)[1])
        for line in bun_a1.log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("CANDIDATE_JSON\t")
    ]
    h2_logs = [e for e in h2_logs if e.get("hap") == "Hap2#1"]
    assert h2_logs and all(e["final_reason"] == "PAN_HOMOLOG_SUBTRACTED" for e in h2_logs)

    inp_a1_trim, bun_a1_trim = _run_cli(
        "e2e_a1_trim",
        {"10": s14[:300], "11": s11, "14": s14, "12": s12, "13": s13},
        a1_paths,
    )
    ev = next(r for r in bun_a1_trim.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
    assert "len=3700" in al["Hap1#1"] and "partial_content=1" in al["Hap1#1"]
    assert "len=3000" in al["Hap2#1"] and "partial_content=1" not in al["Hap2#1"]
    assert "len=3600" in al["Hap3#1"] and "partial_content=1" not in al["Hap3#1"]

    # --dedup-mode off parity check on Probe A1
    out_off = tmp_path / "e2e_a1_trim_off"
    proc_off, bun_off = run_build(inp_a1_trim, extra_args=["--dedup-mode", "off"], out_prefix=out_off)
    assert proc_off.returncode == 0 and bun_off is not None
    run_verify(out_off, inp_a1_trim.gfa_path, inp_a1_trim.ref_fa_path, check=True)
    ev_off = next(r for r in bun_off.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert "partial_content=1" not in ev_off["alleles"]

    # (2) Probe F1 (untouched ref & trimmed pan -> Hap2 exact 3500 clean)
    f1_paths = sup4 + [
        ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "15+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "13+", "12+", "5+", "6+"]),
    ]
    for mode in ("ref", "pan"):
        _, bun_f1 = _run_cli(
            f"e2e_f1_{mode}",
            {"10": s14[:300], "11": s11, "14": s14, "15": s15, "12": s12, "13": s13},
            f1_paths,
            mode=mode,
        )
        ev = next(r for r in bun_f1.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
        al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
        assert "len=3500" in al["Hap2#1"] and "partial_content=1" not in al["Hap2#1"]

    # (3) Probe A5 (childless prefix del)
    _, bun_a5 = _run_cli(
        "e2e_a5",
        {"10": s11 + s13 + s12, "11": s11, "14": s14, "12": s12, "13": s13},
        sup4 + [
            ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "14+", "11+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "14+", "11+", "13+", "12+", "5+", "6+"]),
        ],
    )
    ev = next(r for r in bun_a5.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    gt = bun_a5.genotypes[ev["event_id"]]
    ins = [r for r in bun_a5.inserts if r["event_id"] == ev["event_id"]]
    assert gt["Hap2#1"] == "." and gt["Hap1#1"] != "." and gt["Hap3#1"] != "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1,Hap3#1"

    # (4) Loophole #2A (1 bp SNP replaced by 600 bp ins, 100% homolog -> zero 1 bp orphan)
    _, bun_l2a = _run_cli(
        "e2e_l2a",
        {"10": s11 + "A" + s15 + s12 + s13, "11": s11, "14": "A", "15": s15, "12": s12, "13": s13},
        sup4 + [
            ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "13+", "15+", "12+", "5+", "6+"]),
        ],
    )
    assert not [r for r in bun_l2a.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
    assert int(bun_l2a.summary_kv["dedup_dropped_groups"]) == 1

    # (5) Probe 1 (Merge Case M2 rc_long=4010 + rc_hetero in ref & pan)
    s_long = s11 + s14 + s12 + make_complex_seq(10, seed=5920)
    m2_paths = sup4 + [
        ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "20+", "5+", "6+"]),
    ]
    _, bun_m2_ref = _run_cli(
        "e2e_m2_ref",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "20": s_long},
        m2_paths,
        mode="ref",
    )
    ev = next(r for r in bun_m2_ref.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
    assert "len=3000" in al["Hap2#1"]

    _, bun_m2_pan = _run_cli(
        "e2e_m2_pan",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "20": s_long},
        m2_paths,
        mode="pan",
    )
    ev = next(r for r in bun_m2_pan.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_m2_pan.genotypes[ev["event_id"]]["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"] for r in bun_m2_pan.inserts if r["event_id"] == ev["event_id"])

    # (6) Probe 3B (R3-5): genuine matching-partition synth_comp exact-seq merge of
    # Comp 1 ("START", 0) + ("__b_excl__",) and Comp 2 (0, 1) + ("__b_excl__",) sharing
    # ONE rep_idx (0 spacers, exact per-piece members, zero TypeError in Stage 6c)
    s_c = make_complex_seq(400, seed=5930)
    s_a = make_complex_seq(500, seed=5931)
    s_b = make_complex_seq(500, seed=5932)
    s_d = make_complex_seq(800, seed=5933)
    nodes_3b = {
        "10": s_c + s_b + s_d,
        "40": s_a, "41": s_b, "42": s_c, "43": s_d,
        "50": s_c, "51": s_b, "52": s_a, "53": s_d,
    }
    paths_3b = [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "41+", "43+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "41+", "43+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "42+", "40+", "41+", "5+", "6+"]),
        ("Hap2b#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "43+", "5+", "6+"]),
        ("Hap3a#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "51+", "53+", "5+", "6+"]),
        ("Hap3b#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "51+", "53+", "5+", "6+"]),
        ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "50+", "52+", "51+", "5+", "6+"]),
    ]
    _, bun_3b_ref = _run_cli("e2e_probe3b_ref", nodes_3b, paths_3b, mode="ref")
    ev = next(r for r in bun_3b_ref.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins = [r for r in bun_3b_ref.inserts if r["event_id"] == ev["event_id"]]
    assert len(ins) == 4 and all(r["join_before"] != "spacer" for r in ins)
    assert [int(r["pan_end"]) - int(r["pan_start"]) for r in ins] == [400, 500, 500, 800]
    assert ins[0]["members"] == "Hap2#1,Hap3a#1,Hap3b#1,Hap4#1"
    assert ins[1]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap2b#1,Hap4#1"
    assert ins[2]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap3a#1,Hap3b#1,Hap4#1"
    assert ins[3]["members"] == "Hap1a#1,Hap1b#1,Hap2b#1,Hap3a#1,Hap3b#1"
    assert len(ev["alleles"].split(";")) == 5

    sup8_3b = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    _, bun_3b_pan = _run_cli("e2e_probe3b_pan", nodes_3b, sup8_3b + paths_3b, mode="pan")
    ev_p = next(r for r in bun_3b_pan.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_p = [r for r in bun_3b_pan.inserts if r["event_id"] == ev_p["event_id"]]
    assert len(ins_p) == 1 and (int(ins_p[0]["pan_end"]) - int(ins_p[0]["pan_start"])) == 500
    assert ins_p[0]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap2b#1,Hap4#1"
    assert bun_3b_pan.genotypes[ev_p["event_id"]]["Hap3a#1"] == "."
    assert bun_3b_pan.genotypes[ev_p["event_id"]]["Hap3b#1"] == "."


def test_tier5_dedup_10_iter3_module_abc_protected_ins_and_t22_cli(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI regression for Iter 3 End-to-End Module A+B+C + T22 Determinism:
    - Probe 5 (`44M+6I` `nov55` inside `REF_TE` within a multi-member component, `--min-sv-len 55`)
      under `--dedup-mode ref` and `--dedup-mode pan` (`nov55 in pan_seq`, `14` `members=Hap1#1`).
    - Combined T22 multicore (`--threads 1` vs `--threads 4` with `PANFLAT_FORCE_FORK=1`) 12-file
      SHA-256 byte-identity + shuffled GFA input-order 10-file biological SHA-256 identity
      exercising both `repeat_guard.py` (RG-1..3) and `linearizer.py` (LN-1..5) fixes.
    """
    ref_te = make_complex_seq(1200, seed=6001)
    s11 = make_complex_seq(1500, seed=6002)
    s14 = make_complex_seq(1000, seed=6003)
    s12 = make_complex_seq(1500, seed=6004)
    nov55 = make_complex_seq(55, seed=6005)
    te_44m6i = ref_te[100:594] + nov55 + ref_te[594:1094]  # 44M+6I

    # Part 1: Probe 5 in ref and pan modes
    b5 = SyntheticInputBuilder(tmp_path / "e2e_p5", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 7):
        b5.add_node(str(i), ref_te if i == 2 else make_complex_seq(400, seed=6010 + i))
    for nid, sq in [("10", s11 + s12), ("11", s11), ("14", s14), ("12", s12), ("18", te_44m6i)]:
        b5.add_node(nid, sq)
    b5.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 7)])
    for k in range(1, 5):
        b5.add_path(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"])
    b5.add_path("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "12+", "5+", "6+"])
    b5.add_path("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "18+", "12+", "5+", "6+"])
    inp5 = b5.write("e2e_p5")

    out_p5_ref = tmp_path / "e2e_p5_ref"
    p_ref, bun_ref = run_build(
        inp5,
        extra_args=["--dedup-mode", "ref", "--dedup-id", "0.90", "--min-sv-len", "55"],
        out_prefix=out_p5_ref,
    )
    assert p_ref.returncode == 0 and bun_ref is not None
    run_verify(out_p5_ref, inp5.gfa_path, inp5.ref_fa_path, check=True)
    ev_r = next(r for r in bun_ref.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_r = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_r["alleles"].split(";")}
    assert "len=4000" in al_r["Hap1#1"] and "partial_content=1" not in al_r["Hap1#1"]
    assert "len=3099" in al_r["Hap2#1"] and "partial_content=1" in al_r["Hap2#1"]
    assert nov55 in bun_ref.pan_seq

    out_p5_pan = tmp_path / "e2e_p5_pan"
    p_pan, bun_pan = run_build(
        inp5,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "55"],
        out_prefix=out_p5_pan,
    )
    assert p_pan.returncode == 0 and bun_pan is not None
    run_verify(out_p5_pan, inp5.gfa_path, inp5.ref_fa_path, check=True)
    ev_p = next(r for r in bun_pan.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_p = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_p["alleles"].split(";")}
    ins_p = [r for r in bun_pan.inserts if r["event_id"] == ev_p["event_id"]]
    assert "len=1000" in al_p["Hap1#1"] and "partial_content=1" in al_p["Hap1#1"]
    assert "len=100" in al_p["Hap2#1"] and "partial_content=1" in al_p["Hap2#1"]
    assert nov55 in bun_pan.pan_seq
    by_len = {r["pan_end"] - r["pan_start"]: r for r in ins_p}
    assert by_len[1000]["members"] == "Hap1#1"
    assert by_len[100]["members"] == "Hap2#1"

    # Part 2: Multi-locus T22 multicore (t1 vs t4 with PANFLAT_FORCE_FORK=1) + shuffled GFA
    bt = SyntheticInputBuilder(tmp_path / "e2e_i3_t22", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 13):
        bt.add_node(str(i), ref_te if i == 2 else make_complex_seq(400, seed=6030 + i))
    nov55_l4 = make_complex_seq(55, seed=6050)
    node_l4 = ref_te[100:345] + nov55_l4 + ref_te[345:421]  # 376 bp <= 500 bp L4
    s_long = s11 + s14 + s12 + make_complex_seq(10, seed=6051)
    s13 = make_complex_seq(600, seed=6052)
    s_c = make_complex_seq(400, seed=6053)
    s_a = make_complex_seq(500, seed=6054)
    s_b = make_complex_seq(500, seed=6055)
    s_d = make_complex_seq(800, seed=6056)
    for nid, sq in [
        ("20", s11 + s12),
        ("21", s11),
        ("22", s14),
        ("23", s12),
        ("24", te_44m6i),
        ("25", node_l4),
        ("26", s_long),
        ("27", s13),
        ("30", s_a),
        ("31", s_b),
        ("32", s_c),
        ("33", s_c),
        ("34", s_b),
        ("35", s_a),
        ("36", s_d),
        ("37", s_d),
    ]:
        bt.add_node(nid, sq)
    ref12 = [f"{i}+" for i in range(1, 13)]
    bt.set_ref_path("Ref#1#Chr01", ref12)
    # Add GFF3 gene on node 1 (ref 1..400) & BED region on node 11 (ref 4800..5200) for all 12 outputs
    bt.add_gff_gene(
        gene_id="i3_t22_gene",
        mrna_id="i3_t22_mrna",
        strand="+",
        exons=[(21, 200)],
        cds_list=[(21, 200, 0)],
    )
    bt.add_bed_region(4800, 5200, region_id="CEN_I3_T22")
    # 5 Sup haplotypes at 1..2 carrying 20 (s11+s12)
    for k in range(1, 6):
        bt.add_path(f"Sup{k}#1#Chr01", ["1+", "20+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"])
    # Locus A (3..4): Probe 5 + M2 hetero merge; Locus B (5..6): L4 <=500bp; Locus C (7..8): Probe A5;
    # Locus D (9..10): genuine matching-partition synth_comp merge ("START", 0) + (0, 1) sharing 1 rep_idx
    bt.add_path("H1#1#Chr01",  ["1+", "2+", "3+", "21+", "22+", "23+", "4+", "5+", "25+", "6+", "7+", "22+", "21+", "23+", "8+", "9+", "30+", "31+", "36+", "10+", "11+", "12+"])
    bt.add_path("H1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "30+", "36+", "10+", "11+", "12+"])
    bt.add_path("H2#1#Chr01",  ["1+", "2+", "3+", "21+", "24+", "23+", "4+", "5+", "6+", "7+", "21+", "23+", "8+", "9+", "32+", "30+", "31+", "10+", "11+", "12+"])
    bt.add_path("H3#1#Chr01",  ["1+", "2+", "3+", "21+", "23+", "4+", "5+", "6+", "7+", "22+", "21+", "27+", "23+", "8+", "9+", "33+", "34+", "37+", "10+", "11+", "12+"])
    bt.add_path("H4#1#Chr01",  ["1+", "2+", "3+", "26+", "4+", "5+", "6+", "7+", "8+", "9+", "33+", "35+", "34+", "10+", "11+", "12+"])
    inp_t22 = bt.write("e2e_i3_t22")

    shuf_gfa = tmp_path / "e2e_i3_t22_shuf.gfa"
    _write_shuffled_pw_gfa(inp_t22.gfa_path, shuf_gfa, seed=20261125)
    inp_shuf = InputPaths(
        gfa_path=shuf_gfa,
        ref_fa_path=inp_t22.ref_fa_path,
        bed_path=inp_t22.bed_path,
        gff_path=inp_t22.gff_path,
        out_prefix=tmp_path / "e2e_i3_t22_shuf_out",
        chrom=inp_t22.chrom,
        backbone_prefix=inp_t22.backbone_prefix,
    )

    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    args = [
        "--dedup-mode",
        "pan",
        "--dedup-id",
        "0.90",
        "--min-sv-len",
        "55",
        "--centromere-mode",
        "extract-islands",
        "--qc",
    ]
    out_t1 = tmp_path / "i3_t22_t1"
    out_t4 = tmp_path / "i3_t22_t4"
    out_s4 = tmp_path / "i3_t22_s4"

    p_t1, b_t1 = run_build(inp_t22, extra_args=[*args, "--threads", "1"], out_prefix=out_t1, extra_env=env)
    p_t4, b_t4 = run_build(inp_t22, extra_args=[*args, "--threads", "4"], out_prefix=out_t4, extra_env=env)
    p_s4, b_s4 = run_build(inp_shuf, extra_args=[*args, "--threads", "4"], out_prefix=out_s4, extra_env=env)
    assert p_t1.returncode == 0 and b_t1 is not None
    assert p_t4.returncode == 0 and b_t4 is not None
    assert p_s4.returncode == 0 and b_s4 is not None

    run_verify(out_t1, inp_t22.gfa_path, inp_t22.ref_fa_path, check=True)
    run_verify(out_s4, inp_shuf.gfa_path, inp_shuf.ref_fa_path, check=True)
    _assert_12_files_sha256_identical_and_clean_streams(out_t1, out_t4, p_t1, p_t4)
    for sfx in _BIO_10_OUTPUT_SUFFIXES:
        assert _sha256_file(Path(f"{out_t1}{sfx}")) == _sha256_file(
            Path(f"{out_s4}{sfx}")
        ), f"Shuffled GFA biological SHA-256 mismatch on {sfx}"
    assert nov55 in b_t1.pan_seq and nov55_l4 in b_t1.pan_seq
    # Confirm Locus D (8..9) genuinely merged Comp 1 ("START",0) + Comp 2 (0,1) into 1 rep_idx (0 spacers)
    ev_d = next(r for r in b_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "8..9")
    ins_d = [r for r in b_t1.inserts if r["event_id"] == ev_d["event_id"]]
    assert len(ins_d) == 4 and all(r["join_before"] != "spacer" for r in ins_d)
    assert [int(r["pan_end"]) - int(r["pan_start"]) for r in ins_d] == [400, 500, 500, 800]


def test_tier5_dedup_11_iter4_surgical_closure_cli(tmp_path: Path) -> None:
    """Opaque-box CLI regression for Iteration 4 surgical closure (zero panflat imports):
    1. Probe 3F-CLI: `--dedup-mode pan --min-sv-len 55` multi-piece `'N'*200` Module C wave
       preserves `nov55` intact (`len=55`, `dedup_dropped_groups == 0`).
    2. ADV-1 (R3-1): partial substring `rep0` child merge in `ref` (`Hap2 len=4000`) & `pan`
       (`Hap2="."`, `members="Hap1a#1,Hap1b#1"`).
    3. ADV-2b & ADV-2c (R3-2): distinct `missing_steps` subsets preserve clean `Hap3 len=3000`
       without leaked `,partial_content=1` and remain invariant under `AHap3`/`ZHap2` rename.
    4. ADV-3c & ADV-16 (R3-3): mismatched `synth_comp` vs 1-piece `real_rep` keeps non-carriers
       `"."` and matching-partition `synth_comp` merge refreshes `members="Hap2#1,Hap4#1"`.
    5. ADV-7 (R3-4): `rep_is_hetero` full-span guard with `m_ts + m_del >= 50` rejects merge
       (`Hap3="."`, `members="Hap1#1"`).
    """
    # (1) Probe 3F-CLI: multi-piece 'N'*200 wave L4-1A 55 bp survival
    rng = random.Random(20261125)
    ref_pool = "".join(rng.choices("ACGT", k=30000))
    p0 = p1 = nov55 = ""
    for trial in range(9):
        rp0, rp1 = 500 + trial * 250, 15000 + trial * 250
        p0, p1 = ref_pool[rp0 : rp0 + 500], ref_pool[rp1 : rp1 + 600]
        nov55 = "".join(rng.choices("ACGT", k=55))
    q_500 = p0[-220:] + p1[:175] + nov55 + p1[175:225]
    assert len(q_500) == 500

    b3f = SyntheticInputBuilder(tmp_path / "cli_p3f", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 9):
        b3f.add_node(str(i), make_complex_seq(400, seed=7100 + i))
    b3f.add_node("10", p0)
    b3f.add_node("11", p1)
    b3f.add_node("12", q_500)
    b3f.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 9)])
    for k in range(1, 6):
        path = ["1+", "10+", "2+", "3+"]
        if k <= 4:
            path.append("11+")
        path.extend(["4+", "5+", "6+", "7+", "8+"])
        b3f.add_path(f"Sup{k}#1#Chr01", path)
    b3f.add_path("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "12+", "6+", "7+", "8+"])
    inp3f = b3f.write("cli_p3f")
    out3f = tmp_path / "cli_p3f_out"
    p3f, bun3f = run_build(
        inp3f,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "55"],
        out_prefix=out3f,
    )
    assert p3f.returncode == 0 and bun3f is not None
    run_verify(out3f, inp3f.gfa_path, inp3f.ref_fa_path, check=True)
    assert nov55 in bun3f.pan_seq
    ev3f = next(r for r in bun3f.sv_rows if r["type"] == "INS" and r["anchors"] == "4..5")
    assert "len=55" in ev3f["alleles"]
    assert bun3f.summary_kv["dedup_dropped_groups"] == "0"

    # Shared helper for CLI ADV-1 / ADV-2b / ADV-2c / ADV-3c / ADV-16 / ADV-7
    def _run_adv_cli(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        mode: str,
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=7200 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 7)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", mode, "--dedup-id", "0.90", "--min-sv-len", "50"],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        run_verify(out, inp.gfa_path, inp.ref_fa_path, check=True)
        return bun

    s11 = make_complex_seq(1500, seed=7301)
    s14 = make_complex_seq(1000, seed=7302)
    s15 = make_complex_seq(1000, seed=7303)
    s12 = make_complex_seq(1500, seed=7304)
    s13 = make_complex_seq(600, seed=7305)
    s15p = mutate_seq_snps(s15, [i * 50 + 10 for i in range(18)])
    sup6 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 7)]
    sup8 = sup6 + [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in (7, 8)]

    # (2) CLI ADV-1 (partial rep0 child in ref & pan)
    nodes_adv1 = {"10": s11 + s15 + s12, "11": s11, "14": s14, "15": s15, "15p": s15p, "12": s12}
    paths_adv1 = sup6[:5] + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "15p+", "12+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv1_ref", nodes_adv1, paths_adv1, "ref")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["Hap2#1"]

    bun = _run_adv_cli("cli_adv1_pan", nodes_adv1, paths_adv1, "pan")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins = [r for r in bun.inserts if r["event_id"] == ev["event_id"]]
    assert bun.genotypes[ev["event_id"]]["Hap1a#1"] == "1"
    assert bun.genotypes[ev["event_id"]]["Hap1b#1"] == "1"
    assert bun.genotypes[ev["event_id"]]["Hap2#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1a#1,Hap1b#1"

    # (3) CLI ADV-2b & ADV-2c (distinct missing_steps + name-order invariance)
    nodes_adv2 = {"10": s14[:300], "11": s11, "14": s14, "15": s15, "12": s12}
    paths_adv2b = sup6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "14-", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv2b_pan", nodes_adv2, paths_adv2b, "pan")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
    assert "len=4700" in al["Hap1a#1"] and "partial_content=1" in al["Hap1a#1"]
    assert "len=3700" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert "len=3000" in al["Hap3#1"] and "partial_content=1" not in al["Hap3#1"]

    paths_adv2c = sup6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15+", "12+", "5+", "6+"]),
        ("ZHap2#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14-", "12+", "5+", "6+"]),
        ("AHap3#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv2c_ref", nodes_adv2, paths_adv2c, "ref")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["ZHap2#1"] and "len=3000" in al["AHap3#1"]

    # (4) CLI ADV-3c & ADV-16 (synth_comp exact-seq merge & members refresh)
    nodes_adv3c = {
        "10": s11 + s12 + s15,
        "11": s11, "13": s13, "12": s12, "15": s15,
        "20": s11 + s13 + s12 + s15,
    }
    paths_adv3c = sup6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "15+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "15+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "13+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "20+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv3c_pan", nodes_adv3c, paths_adv3c, "pan")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins = [r for r in bun.inserts if r["event_id"] == ev["event_id"]]
    assert bun.genotypes[ev["event_id"]]["Hap1a#1"] == "."
    assert bun.genotypes[ev["event_id"]]["Hap1b#1"] == "."
    assert bun.genotypes[ev["event_id"]]["Hap2#1"] == "1"
    assert all("Hap1a#1" not in r["members"] and "Hap1b#1" not in r["members"] for r in ins)

    nodes_adv16 = {
        "10": s11 + s12 + s15,
        "11": s11, "13": s13, "12": s12, "15": s15,
        "21": s11, "23": s13, "22": s12, "25": s15,
    }
    paths_adv16 = sup8 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "15+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "12+", "15+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "13+", "12+", "5+", "6+"]),
        ("Hap3a#1#Chr01", ["1+", "2+", "3+", "4+", "21+", "22+", "25+", "5+", "6+"]),
        ("Hap3b#1#Chr01", ["1+", "2+", "3+", "4+", "21+", "22+", "25+", "5+", "6+"]),
        ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "21+", "23+", "22+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv16_pan", nodes_adv16, paths_adv16, "pan")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins = [r for r in bun.inserts if r["event_id"] == ev["event_id"]]
    assert bun.genotypes[ev["event_id"]]["Hap2#1"] == "1"
    assert bun.genotypes[ev["event_id"]]["Hap4#1"] == "1"
    assert bun.genotypes[ev["event_id"]]["Hap1a#1"] == "."
    assert bun.genotypes[ev["event_id"]]["Hap3a#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap2#1,Hap4#1"

    # (5) CLI ADV-7 (rep_is_hetero m_ts + m_del >= 50)
    p30 = make_complex_seq(30, seed=7401)
    sc1 = make_complex_seq(1500, seed=7402)
    m30 = make_complex_seq(30, seed=7403)
    sc2 = make_complex_seq(1500, seed=7404)
    ins25 = make_complex_seq(25, seed=7405)
    nodes_adv7 = {
        "10": sc1 + sc2,
        "p30": p30, "sc1": sc1, "m30": m30, "sc2": sc2,
        "20": sc1 + sc2 + ins25,
    }
    paths_adv7 = sup6 + [
        ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "p30+", "sc1+", "m30+", "sc2+", "5+", "6+"]),
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "sc1-", "sc2-", "5+", "6+"]),
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "20+", "5+", "6+"]),
    ]
    bun = _run_adv_cli("cli_adv7_pan", nodes_adv7, paths_adv7, "pan")
    ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins = [r for r in bun.inserts if r["event_id"] == ev["event_id"]]
    assert bun.genotypes[ev["event_id"]]["Hap1#1"] == "1"
    assert bun.genotypes[ev["event_id"]]["Hap2#1"] == "."
    assert bun.genotypes[ev["event_id"]]["Hap3#1"] == "."
    assert len(ins) == 2 and all(r["members"] == "Hap1#1" for r in ins)


def test_tier5_dedup_12_iter5_non_adjacent_3piece_wave_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 1 (Iter 5, ZERO panflat imports):
    Non-adjacent 3-piece Module C wave `p0(500) + N*200 + pm(150) + N*200 + p2(600)`
    under `panflat build --dedup-mode pan --dedup-id 0.90 --min-sv-len 55`:
    - Offset variant `p0[250:470] + p2[40:215] + nov55_off + p2[215:265]` (leading/trailing D strip + atomic dedup)
    - Zero-offset variant `p0[-220:] + p2[:175] + nov55_zero + p2[175:225]` (atomic per-piece-unit dedup)
    Asserts both `nov55_off` and `nov55_zero` survive in `pan.fa`, `dedup_dropped_groups == "0"`,
    and standalone `panflat verify` passes.
    """
    # Exact trial=12 sequence from explorer_iter5_1 (offset off0=30, off2=40)
    rng_off = random.Random(20261125)
    ref_off = "".join(rng_off.choices("ACGT", k=60000))
    p0_off = pm_off = p2_off = nov55_off = ""
    for trial in range(13):
        rp0, rpm, rp2 = 500 + trial * 150, 20000 + trial * 150, 40000 + trial * 150
        p0_off = ref_off[rp0 : rp0 + 500]
        pm_off = ref_off[rpm : rpm + 150]
        p2_off = ref_off[rp2 : rp2 + 600]
        nov55_off = "".join(rng_off.choices("ACGT", k=55))
    q_off = p0_off[250:470] + p2_off[40:215] + nov55_off + p2_off[215:265]
    assert len(q_off) == 500

    b_off = SyntheticInputBuilder(tmp_path / "cli_i5_3p_off", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 10):
        b_off.add_node(str(i), make_complex_seq(400, seed=8100 + i))
    b_off.add_node("10", p0_off)
    b_off.add_node("11", pm_off)
    b_off.add_node("12", p2_off)
    b_off.add_node("13", q_off)
    b_off.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 10)])
    for k in range(1, 7):
        path = ["1+", "10+", "2+", "3+"]
        if k <= 5:
            path.append("11+")
        path.extend(["4+", "5+"])
        if k <= 4:
            path.append("12+")
        path.extend(["6+", "7+", "8+", "9+"])
        b_off.add_path(f"Sup{k}#1#Chr01", path)
    b_off.add_path("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "13+", "8+", "9+"])
    inp_off = b_off.write("cli_i5_3p_off")
    out_off = tmp_path / "cli_i5_3p_off_out"
    p_o, bun_o = run_build(
        inp_off,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "55"],
        out_prefix=out_off,
    )
    assert p_o.returncode == 0 and bun_o is not None, p_o.stderr
    run_verify(out_off, inp_off.gfa_path, inp_off.ref_fa_path, check=True)
    assert nov55_off in bun_o.pan_seq
    assert bun_o.summary_kv["dedup_dropped_groups"] == "0"
    ev_o = next(r for r in bun_o.sv_rows if r["type"] == "INS" and r["anchors"] == "6..7")
    assert "len=55" in ev_o["alleles"]

    # Exact trial=3 zero-offset sequence from reviewer_dedup_r4_1 (len_mid=150)
    rng_z = random.Random(20261125)
    ref_z = "".join(rng_z.choices("ACGT", k=40000))
    p0_z = pm_z = p2_z = nov55_z = ""
    for trial in range(4):
        rp0, rpm, rp2 = 500 + trial * 250, 14000 + trial * 250, 27000 + trial * 250
        p0_z = ref_z[rp0 : rp0 + 500]
        pm_z = ref_z[rpm : rpm + 150]
        p2_z = ref_z[rp2 : rp2 + 600]
        nov55_z = "".join(rng_z.choices("ACGT", k=55))
    q_z = p0_z[-220:] + p2_z[:175] + nov55_z + p2_z[175:225]
    assert len(q_z) == 500

    b_z = SyntheticInputBuilder(tmp_path / "cli_i5_3p_zero", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 10):
        b_z.add_node(str(i), make_complex_seq(400, seed=8150 + i))
    b_z.add_node("10", p0_z)
    b_z.add_node("11", pm_z)
    b_z.add_node("12", p2_z)
    b_z.add_node("13", q_z)
    b_z.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 10)])
    for k in range(1, 7):
        path = ["1+", "10+", "2+", "3+"]
        if k <= 5:
            path.append("11+")
        path.extend(["4+", "5+"])
        if k <= 4:
            path.append("12+")
        path.extend(["6+", "7+", "8+", "9+"])
        b_z.add_path(f"Sup{k}#1#Chr01", path)
    b_z.add_path("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "13+", "8+", "9+"])
    inp_z = b_z.write("cli_i5_3p_zero")
    out_z = tmp_path / "cli_i5_3p_zero_out"
    p_z, bun_z = run_build(
        inp_z,
        extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "55"],
        out_prefix=out_z,
    )
    assert p_z.returncode == 0 and bun_z is not None, p_z.stderr
    run_verify(out_z, inp_z.gfa_path, inp_z.ref_fa_path, check=True)
    assert nov55_z in bun_z.pan_seq
    assert bun_z.summary_kv["dedup_dropped_groups"] == "0"
    ev_z = next(r for r in bun_z.sv_rows if r["type"] == "INS" and r["anchors"] == "6..7")
    assert "len=55" in ev_z["alleles"]


def test_tier5_dedup_13_iter5_r4_1_merges_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 2 (Iter 5, ZERO panflat imports):
    1. `R4_3B_ZERO_TANDEM_1000bp` in `ref` (both haplotype orders: exact 4500/3500, zero
       name-order flip) and `pan` (zero `partial_content=1` leak onto untouched `Hap3a-3d`).
    2. `M4_multi_occ_triangle_sup8_pan` (`Sup = sc1+sc2` and `Sup = sc1+m30+sc2`):
       `Hap3c` rejected from merging into `rep_is_hetero` and absent from `p30`/`m30` `members`.
    """
    def _run_cli(tag: str, extra_nodes: dict[str, str], paths: list[tuple[str, list[str]]], mode: str):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=8200 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 7)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", mode, "--dedup-id", "0.90", "--min-sv-len", "50"],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        run_verify(out, inp.gfa_path, inp.ref_fa_path, check=True)
        return bun

    # (1) R4_3B_ZERO_TANDEM_1000bp
    s_a = make_complex_seq(1000, seed=8211)
    s_b = make_complex_seq(1000, seed=8212)
    s_c = make_complex_seq(1000, seed=8213)
    s_e = make_complex_seq(1000, seed=8214)
    s_d = make_complex_seq(1500, seed=8215)
    nodes_r43b = {
        "10": s_c[:300],
        "40": s_a, "41": s_c, "42": s_b, "48": s_e, "49": s_d,
        "50": s_a, "52": s_b, "53": s_c, "58": s_e, "59": s_d,
    }
    sup14 = [(f"Sup{k:02d}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 15)]
    paths_r43b = sup14 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "41+", "48+", "49+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "41+", "48+", "49+", "5+", "6+"]),
        ("Hap1c#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "41+", "48+", "49+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "40+", "42+", "41+", "48+", "5+", "6+"]),
        ("Hap2b#1#Chr01", ["1+", "2+", "3+", "4+", "40+", "48+", "49+", "5+", "6+"]),
        ("Hap3a#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "58+", "59+", "5+", "6+"]),
        ("Hap3b#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "58+", "59+", "5+", "6+"]),
        ("Hap3c#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "58+", "59+", "5+", "6+"]),
        ("Hap3d#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "58+", "59+", "5+", "6+"]),
        ("Hap4a#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "52+", "53+", "58+", "5+", "6+"]),
        ("Hap4b#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "52+", "53+", "58+", "5+", "6+"]),
        ("Hap4c#1#Chr01", ["1+", "2+", "3+", "4+", "50+", "52+", "58+", "5+", "6+"]),
    ]
    bun_ref = _run_cli("cli_i5_r43b_ref", nodes_r43b, paths_r43b, "ref")
    ev_r = next(r for r in bun_ref.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_r = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_r["alleles"].split(";")}
    assert bun_ref.genotypes[ev_r["event_id"]]["Hap1a#1"] != bun_ref.genotypes[ev_r["event_id"]]["Hap3a#1"]
    assert "len=4500" in al_r["Hap1a#1"] and "len=3500" in al_r["Hap3a#1"]

    paths_flip = sup14 + [
        (p.replace("Hap1", "ZHap1").replace("Hap2", "ZHap2").replace("Hap3", "AHap3").replace("Hap4", "AHap4"), st)
        for p, st in paths_r43b[14:]
    ]
    bun_flip = _run_cli("cli_i5_r43b_flip_ref", nodes_r43b, paths_flip, "ref")
    ev_f = next(r for r in bun_flip.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_f = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_f["alleles"].split(";")}
    assert bun_flip.genotypes[ev_f["event_id"]]["ZHap1a#1"] != bun_flip.genotypes[ev_f["event_id"]]["AHap3a#1"]
    assert "len=4500" in al_f["ZHap1a#1"] and "len=3500" in al_f["AHap3a#1"]

    bun_pan = _run_cli("cli_i5_r43b_pan", nodes_r43b, paths_r43b, "pan")
    ev_p = next(r for r in bun_pan.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_p = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_p["alleles"].split(";")}
    assert bun_pan.genotypes[ev_p["event_id"]]["Hap3a#1"] == "1"
    assert bun_pan.genotypes[ev_p["event_id"]]["Hap1a#1"] == "."
    assert "len=3500" in al_p["Hap3a#1"] and "partial_content=1" not in al_p["Hap3a#1"]
    assert "len=3700" in al_p["Hap4a#1"] and "partial_content=1" in al_p["Hap4a#1"]

    # (2) M4_multi_occ_triangle_sup8_pan
    p30 = make_complex_seq(30, seed=8221)
    sc1 = make_complex_seq(1500, seed=8222)
    m30 = make_complex_seq(30, seed=8223)
    sc2 = make_complex_seq(1500, seed=8224)
    sc1p = mutate_seq_snps(sc1, positions=[i * 100 + 15 for i in range(12)], seed=8225)
    sc2p = mutate_seq_snps(sc2, positions=[i * 100 + 25 for i in range(12)], seed=8226)
    nodes_m4 = {
        "10": sc1 + sc2,
        "p30": p30, "sc1": sc1, "m30": m30, "sc2": sc2,
        "sc1p": sc1p, "m30p": m30, "sc2p": sc2p,
    }
    paths_m4 = sup14[:8] + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "p30+", "sc1+", "m30+", "sc2+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "p30+", "sc1+", "m30+", "sc2+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "sc1-", "sc2-", "5+", "6+"]),
        ("Hap3a#1#Chr01", ["1+", "2+", "3+", "4+", "sc1p+", "m30p+", "sc2p+", "5+", "6+"]),
        ("Hap3b#1#Chr01", ["1+", "2+", "3+", "4+", "sc1p+", "m30p+", "sc2p+", "5+", "6+"]),
        ("Hap3c#1#Chr01", ["1+", "2+", "3+", "4+", "sc1p+", "sc2p+", "5+", "6+"]),
    ]
    bun_m4 = _run_cli("cli_i5_m4_tri_pan", nodes_m4, paths_m4, "pan")
    ev_m4 = next(r for r in bun_m4.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_m4 = [r for r in bun_m4.inserts if r["event_id"] == ev_m4["event_id"]]
    assert bun_m4.genotypes[ev_m4["event_id"]]["Hap3c#1"] != bun_m4.genotypes[ev_m4["event_id"]]["Hap1a#1"]
    p30_m30 = [r for r in ins_m4 if (int(r["pan_end"]) - int(r["pan_start"])) == 30]
    assert len(p30_m30) == 2 and all(r["members"] == "Hap1a#1,Hap1b#1" for r in p30_m30)

    bun_m4s = _run_cli("cli_i5_m4_tri_supm_pan", {**nodes_m4, "10": sc1 + m30 + sc2}, paths_m4, "pan")
    ev_m4s = next(r for r in bun_m4s.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_m4s = [r for r in bun_m4s.inserts if r["event_id"] == ev_m4s["event_id"]]
    assert bun_m4s.genotypes[ev_m4s["event_id"]]["Hap3c#1"] == "."
    assert len(ins_m4s) == 1 and ins_m4s[0]["members"] == "Hap1a#1,Hap1b#1"


def test_tier5_dedup_14_iter5_r4_2_and_short_ctx_t22_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 3 (Iter 5, ZERO panflat imports):
    1. `R4_1C2` (`ov=1, 20, 25` on `15a(40)+15b(40)`): `pan` preserves full 80 bp intact
       (`Hap2 == "."`, `members == "Hap1a#1,Hap1b#1"`), `ref` emits distinct `len=4000`.
    2. `R4_1D`: partial-`Sup` excludes BOTH `m30` and `t30` for `Hap2` (`gt == "."`);
       full-`Sup` cleanly drops to `0` INS via §4.4.
    3. `T9_m15_t35_full_sup_pan` cleanly dropped to `0` INS; `ADV-7` preserved; `spacer_n=0` verified.
    4. T22 multicore (`--threads 1` vs `--threads 4` with `PANFLAT_FORCE_FORK=1`) 12-file SHA-256
       byte-identity + `panflat verify` PASS on a combined multi-locus Iteration 5 GFA.
    """
    def _run_cli(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        mode: str,
        spacer_n: int = 100,
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=8300 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 7)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=[
                "--dedup-mode",
                mode,
                "--dedup-id",
                "0.90",
                "--min-sv-len",
                "50",
                "--spacer-n",
                str(spacer_n),
            ],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        run_verify(
            out,
            inp.gfa_path,
            inp.ref_fa_path,
            extra_args=["--spacer-n", str(spacer_n)],
            check=True,
        )
        return bun

    s11 = make_complex_seq(1500, seed=8311)
    s14 = make_complex_seq(1000, seed=8312)
    s15a = make_complex_seq(40, seed=8313)
    s15b = make_complex_seq(40, seed=8314)
    s12 = make_complex_seq(1500, seed=8315)
    s14p = mutate_seq_snps(s14, positions=[i * 100 + 20 for i in range(10)], seed=8316)
    sup6 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 7)]

    # (1) R4_1C2 (ov=1, 20, 25)
    for ov in (1, 20, 25):
        nodes_ov = {
            "10": s11 + s14 + s12,
            "11": s11, "14": s14, "15a": s15a, "15b": s15b, "12": s12,
            "14ov": s14p + s15a[:ov],
        }
        paths_ov = sup6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15a+", "15b+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "15a+", "15b+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "14ov+", "12+", "5+", "6+"]),
        ]
        bun_r = _run_cli(f"cli_i5_r41c2_ov{ov}_ref", nodes_ov, paths_ov, "ref")
        ev_r = next(r for r in bun_r.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
        al_r = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_r["alleles"].split(";")}
        assert "len=4000" in al_r["Hap2#1"] and "len=4080" in al_r["Hap1a#1"]

        bun_p = _run_cli(f"cli_i5_r41c2_ov{ov}_pan", nodes_ov, paths_ov, "pan")
        ev_p = next(r for r in bun_p.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
        ins_p = [r for r in bun_p.inserts if r["event_id"] == ev_p["event_id"]]
        assert bun_p.genotypes[ev_p["event_id"]]["Hap1a#1"] == "1"
        assert bun_p.genotypes[ev_p["event_id"]]["Hap2#1"] == "."
        assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in ins_p) == 80
        assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in ins_p)

    # (2) R4_1D (partial-Sup and full-Sup)
    m30 = make_complex_seq(30, seed=8321)
    s15 = make_complex_seq(1000, seed=8322)
    t30 = make_complex_seq(30, seed=8323)
    s15p = mutate_seq_snps(s15, positions=[i * 100 + 30 for i in range(10)], seed=8324)
    nodes_1d = {
        "10": s11 + s14 + s15 + s12,
        "11": s11, "14": s14, "m30": m30, "15": s15, "t30": t30, "12": s12,
        "20": s14p + s15p,
    }
    paths_1d = sup6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m30+", "15+", "t30+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m30+", "15+", "t30+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "20+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "14-", "15-", "12+", "5+", "6+"]),
    ]
    bun_1d = _run_cli("cli_i5_r41d_pan", nodes_1d, paths_1d, "pan")
    ev_1d = next(r for r in bun_1d.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_1d = [r for r in bun_1d.inserts if r["event_id"] == ev_1d["event_id"]]
    assert bun_1d.genotypes[ev_1d["event_id"]]["Hap2#1"] == "."
    assert bun_1d.genotypes[ev_1d["event_id"]]["Hap3#1"] == "."
    assert len(ins_1d) == 2 and all(r["members"] == "Hap1a#1,Hap1b#1" for r in ins_1d)

    bun_1d_full = _run_cli("cli_i5_r41d_full_pan", {**nodes_1d, "10": s11 + s14 + m30 + s15 + t30 + s12}, paths_1d, "pan")
    assert not [r for r in bun_1d_full.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]

    # (3) T9_m15_t35_full_sup_pan, spacer_n=0, & ADV-7
    m15 = make_complex_seq(15, seed=8331)
    t35 = make_complex_seq(35, seed=8332)
    nov35 = make_complex_seq(35, seed=8333)
    nodes_t9 = {
        "10": s11 + s14 + m15 + s15 + t35 + s12,
        "11": s11, "14": s14, "m15": m15, "15": s15, "t35": t35, "12": s12,
    }
    paths_t9 = sup6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m15+", "15+", "t35+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m15+", "15+", "t35+", "12+", "5+", "6+"]),
        ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "14-", "15-", "12+", "5+", "6+"]),
    ]
    bun_t9 = _run_cli("cli_i5_t9_full_pan", nodes_t9, paths_t9, "pan")
    assert not [r for r in bun_t9.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]

    # spacer_n=0 with m15 dropped via §4.4 and nov35 preserved -> [INV2] PASS
    bun_sp0 = _run_cli(
        "cli_i5_sp0_pan",
        {**nodes_t9, "10": s11 + s14 + m15 + s15 + s12, "t35": nov35},
        paths_t9,
        "pan",
        spacer_n=0,
    )
    ev_sp0 = next(r for r in bun_sp0.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_sp0 = [r for r in bun_sp0.inserts if r["event_id"] == ev_sp0["event_id"]]
    assert len(ins_sp0) == 1 and (int(ins_sp0[0]["pan_end"]) - int(ins_sp0[0]["pan_start"])) == 35

    # ADV-7 preserved intact
    p30_a7 = make_complex_seq(30, seed=8334)
    nodes_a7 = {"10": s11 + s12, "p30": p30_a7, "11": s11, "m30": m30, "12": s12}
    paths_a7 = sup6 + [
        ("Hap1#1#Chr01", ["1+", "2+", "3+", "4+", "p30+", "11+", "m30+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "11-", "12-", "5+", "6+"]),
    ]
    bun_a7 = _run_cli("cli_i5_adv7_pan", nodes_a7, paths_a7, "pan")
    ev_a7 = next(r for r in bun_a7.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_a7 = [r for r in bun_a7.inserts if r["event_id"] == ev_a7["event_id"]]
    assert len(ins_a7) == 2 and all(r["members"] == "Hap1#1" for r in ins_a7)

    # (4) Combined multi-locus T22 determinism (--threads 1 vs --threads 4, PANFLAT_FORCE_FORK=1)
    rng_t = random.Random(20261125)
    ref_t = "".join(rng_t.choices("ACGT", k=60000))
    p0_t, pm_t, p2_t = ref_t[2300:2800], ref_t[21800:21950], ref_t[41800:42400]
    nov55_t = make_complex_seq(55, seed=8341)
    q_t = p0_t[250:470] + p2_t[40:215] + nov55_t + p2_t[215:265]

    bt22 = SyntheticInputBuilder(tmp_path / "cli_i5_t22", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 14):
        bt22.add_node(str(i), make_complex_seq(400, seed=8400 + i))
    bt22.add_node("w0", p0_t)
    bt22.add_node("wm", pm_t)
    bt22.add_node("w2", p2_t)
    bt22.add_node("wq", q_t)
    bt22.add_node("sup_full", s11 + s14 + m15 + s15 + t35 + s12)
    bt22.add_node("n11", s11)
    bt22.add_node("n14", s14)
    bt22.add_node("n15a", s15a)
    bt22.add_node("n15b", s15b)
    bt22.add_node("n14ov", s14p + s15a[:25])
    bt22.add_node("nm15", m15)
    bt22.add_node("n15", s15)
    bt22.add_node("nt35", t35)
    bt22.add_node("n12", s12)
    bt22.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 14)])
    bt22.add_gff_gene(
        gene_id="i5_t22_gene",
        mrna_id="i5_t22_mrna",
        strand="+",
        exons=[(21, 200)],
        cds_list=[(21, 200, 0)],
    )
    bt22.add_bed_region(4800, 5200, region_id="CEN_I5_T22")
    for k in range(1, 7):
        p_steps = ["1+", "w0+", "2+", "3+"]
        if k <= 5:
            p_steps.append("wm+")
        p_steps.extend(["4+", "5+"])
        if k <= 4:
            p_steps.append("w2+")
        p_steps.extend(["6+", "sup_full+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
        bt22.add_path(f"Sup{k}#1#Chr01", p_steps)
    # Locus 7..8: non-adjacent 3-piece offset wave nov55
    bt22.add_path("HapW#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "wq+", "9+", "10+", "11+", "12+", "13+"])
    # Locus 9..10: R4_1C2 ov=25
    bt22.add_path("HapC1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "n11+", "n14+", "n15a+", "n15b+", "n12+", "11+", "12+", "13+"])
    bt22.add_path("HapC2#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "n11+", "n14+", "n15a+", "n15b+", "n12+", "11+", "12+", "13+"])
    bt22.add_path("HapC3#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "n11+", "n14ov+", "n12+", "11+", "12+", "13+"])
    # Locus 11..12: T9_m15_t35 full-Sup drop via §4.4
    bt22.add_path("HapT1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "n11+", "n14+", "nm15+", "n15+", "nt35+", "n12+", "13+"])
    bt22.add_path("HapT2#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "n11+", "n14+", "nm15+", "n15+", "nt35+", "n12+", "13+"])
    bt22.add_path("HapT3#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "n11+", "n14-", "n15-", "n12+", "13+"])

    inp_t22 = bt22.write("cli_i5_t22")
    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    args = ["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "55"]
    out_t1 = tmp_path / "cli_i5_t22_t1"
    out_t4 = tmp_path / "cli_i5_t22_t4"
    p_t1, b_t1 = run_build(inp_t22, extra_args=[*args, "--threads", "1"], out_prefix=out_t1, extra_env=env)
    p_t4, b_t4 = run_build(inp_t22, extra_args=[*args, "--threads", "4"], out_prefix=out_t4, extra_env=env)
    assert p_t1.returncode == 0 and b_t1 is not None, p_t1.stderr
    assert p_t4.returncode == 0 and b_t4 is not None, p_t4.stderr
    run_verify(out_t1, inp_t22.gfa_path, inp_t22.ref_fa_path, check=True)
    _assert_12_files_sha256_identical_and_clean_streams(out_t1, out_t4, p_t1, p_t4)
    assert nov55_t in b_t1.pan_seq
    assert not [r for r in b_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "11..12"]
    ev_c = next(r for r in b_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "9..10")
    assert b_t1.genotypes[ev_c["event_id"]]["HapC3#1"] == "."


def test_tier5_dedup_15_iter6_closure_and_t22_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 4 (Iter 6 Final Closure, ZERO panflat imports):
    1. Bug 1: `BUG3_DEFAULT_CLI` (`ref` & `pan`), `H4c-nested-v2` (`ref`), `Attack 2B` (`ref` & `pan`).
    2. Bug 2: `H4` (`Sup=A+m1+B`, `pan`), `H4-PREFIX-80D` (`pan`), `ADV-7-HOM` (`pan`).
    3. Bug 3: `H1c` (`pan`), `ATTACK4_big` (`aL=1535`, `pan`), `Topo3a` (`pan`), `R4_1D` 3 bp microhomology (`pan`).
    4. Bug 4 & Refinement #2: `fuzz_1_fwd == fuzz_1_rev` (`ref` & `pan`), `PROBE_3C_TRUE_ZERO` (`ref` & `pan`, no `len=0,`).
    5. Combined multi-locus T22 `--threads 1` vs `--threads 4` (`PANFLAT_FORCE_FORK=1`) 12-file SHA-256 identity
       under BOTH `--dedup-mode pan` and `--dedup-mode off`, plus `panflat verify` PASS.
    """
    def _run_cli(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        mode: str,
        dedup_id: str = "0.90",
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 7):
            b.add_node(str(i), make_complex_seq(400, seed=8600 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 7)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", mode, "--dedup-id", dedup_id, "--min-sv-len", "50"],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        run_verify(out, inp.gfa_path, inp.ref_fa_path, check=True)
        return bun

    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    sup6 = sup8[:6]

    # (1) Bug 1: BUG3_DEFAULT_CLI (ref & pan), H4c-nested-v2 (ref), Attack 2B (ref & pan)
    la, ra = make_complex_seq(1000, seed=8611), make_complex_seq(1000, seed=8612)
    seq_a, seq_c = make_complex_seq(1500, seed=8613), make_complex_seq(1500, seed=8614)
    t45, seq_x, seq_y = make_complex_seq(45, seed=8615), make_complex_seq(40, seed=8616), make_complex_seq(40, seed=8617)
    a2 = mutate_seq_snps(seq_a, positions=[i * 150 + 20 for i in range(10)], seed=8618)
    c2 = mutate_seq_snps(seq_c, positions=[i * 150 + 25 for i in range(10)], seed=8619)
    t45_2 = mutate_seq_snps(t45, positions=[15], seed=8620)
    b3_nodes = {"La": la, "Ra": ra, "AC_T45": seq_a + seq_c + t45, "A2": a2, "X": seq_x, "Y": seq_y, "C2": c2, "T45_2": t45_2}
    b3_paths = [
        (f"Hap0{ch}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "AC_T45+", "Ra+", "5+", "6+"]) for ch in ("a", "b", "c", "d")
    ] + [
        (f"Hap1{ch}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "A2+", "X+", "C2+", "T45_2+", "Ra+", "5+", "6+"]) for ch in ("a", "b")
    ] + [
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "A2+", "X+", "Y+", "C2+", "Ra+", "5+", "6+"]),
    ]
    for mode in ("ref", "pan"):
        bun = _run_cli(f"cli_i6_bug3_{mode}", b3_nodes, b3_paths, mode)
        assert (seq_x + seq_y) in bun.pan_seq
        ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
        al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev["alleles"].split(";")}
        assert "len=5045" in al["Hap0a#1"] and "n=6" in al["Hap0a#1"]
        h2_bp = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun.inserts if r["event_id"] == ev["event_id"] and "Hap2#1" in r["members"].split(","))
        if mode == "ref":
            assert "len=5080" in al["Hap2#1"] and h2_bp == 5080
        else:
            assert "partial_content=1" in al["Hap2#1"] and f"len={h2_bp}" in al["Hap2#1"] and h2_bp >= 2080

    nl, nr = make_complex_seq(1500, seed=8621), make_complex_seq(1500, seed=8622)
    na1, na2, na3 = make_complex_seq(1000, seed=8623), make_complex_seq(1000, seed=8624), make_complex_seq(1000, seed=8625)
    nx1, nx2 = make_complex_seq(20, seed=8626), make_complex_seq(20, seed=8627)
    nins1, nins2 = make_complex_seq(35, seed=8628), make_complex_seq(35, seed=8629)
    h4cn_nodes = {
        "L": nl, "R": nr, "rmid": na1 + na2 + na3,
        "a1p": mutate_seq_snps(na1, positions=[i * 100 + 20 for i in range(10)], seed=8630),
        "a2p": mutate_seq_snps(na2, positions=[i * 100 + 20 for i in range(10)], seed=8631),
        "a3p": mutate_seq_snps(na3, positions=[i * 100 + 20 for i in range(10)], seed=8632),
        "x1": nx1, "x2": nx2, "ins1": nins1, "ins2": nins2,
    }
    h4cn_paths = [
        (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "L+", "rmid+", "R+", "5+", "6+"]) for k in range(1, 6)
    ] + [
        (f"{h}#1#Chr01", ["1+", "2+", "3+", "4+", "L+", "a1p+", "x1+", "a2p+", "x2+", "a3p+", "R+", "5+", "6+"]) for h in ("Hap2a", "Hap2b", "Hap2d")
    ] + [
        ("Hap2c#1#Chr01", ["1+", "2+", "3+", "4+", "L+", "a1p+", "ins1+", "a2p+", "ins2+", "a3p+", "R+", "5+", "6+"]),
    ]
    bun_cn = _run_cli("cli_i6_h4cn_ref", h4cn_nodes, h4cn_paths, "ref")
    assert nins1 in bun_cn.pan_seq and nins2 in bun_cn.pan_seq
    ev_cn = next(r for r in bun_cn.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    h2c_bp = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_cn.inserts if r["event_id"] == ev_cn["event_id"] and "Hap2c#1" in r["members"].split(","))
    assert h2c_bp == 6070

    d1, d2, d3 = make_complex_seq(400, seed=8633), make_complex_seq(200, seed=8634), make_complex_seq(400, seed=8635)
    ds_mut = _mutate_per_50bp_block(d1 + d2 + d3, (25,), seed=8636)
    a2b_nodes = {"10": nl + nr, "La": nl, "Ra": nr, "rmid": d1 + d2 + d3, "s1": ds_mut[:400], "s2": ds_mut[400:600], "s3": ds_mut[600:]}
    a2b_paths = sup8 + [
        (f"Hap0_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "rmid+", "Ra+", "5+", "6+"]) for k in range(1, 5)
    ] + [
        (f"Hap1{ch}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "s1+", "s2+", "s3+", "Ra+", "5+", "6+"]) for ch in ("a", "b")
    ] + [
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "s1+", "s3+", "Ra+", "5+", "6+"]),
    ]
    bun_2br = _run_cli("cli_i6_atk2b_ref", a2b_nodes, a2b_paths, "ref")
    ev_2br = next(r for r in bun_2br.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    h2_2br = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_2br.inserts if r["event_id"] == ev_2br["event_id"] and "Hap2#1" in r["members"].split(","))
    assert h2_2br == 3800
    a2b_nodes_full = {**a2b_nodes, "10": nl + ds_mut[:400] + ds_mut[600:] + nr}
    bun_2bp = _run_cli("cli_i6_atk2b_pan", a2b_nodes_full, a2b_paths, "pan")
    ev_2bp = next(r for r in bun_2bp.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_2bp.genotypes[ev_2bp["event_id"]]["Hap0_1#1"] == "1"
    assert bun_2bp.genotypes[ev_2bp["event_id"]]["Hap2#1"] == "."

    # (2) Bug 2: H4 (Sup=A+m1+B), H4-PREFIX-80D, ADV-7-HOM (pan)
    m1, m2 = make_complex_seq(40, seed=8641), make_complex_seq(40, seed=8642)
    h4_nodes = {"10": seq_a + m1 + seq_c, "rep_n": seq_a + m1 + m2 + seq_c, "ap": a2, "m1": m1, "bp": c2}
    h4_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "rep_n+", "5+", "6+"]) for k in range(1, 5)
    ] + [
        ("Hap2a#1#Chr01", ["1+", "2+", "3+", "4+", "ap+", "m1+", "bp+", "5+", "6+"]),
        ("Hap2b#1#Chr01", ["1+", "2+", "3+", "4+", "ap+", "m1+", "bp+", "5+", "6+"]),
        ("Hap2c#1#Chr01", ["1+", "2+", "3+", "4+", "ap+", "bp+", "5+", "6+"]),
    ]
    bun_h4 = _run_cli("cli_i6_h4_pan", h4_nodes, h4_paths, "pan")
    ev_h4 = next(r for r in bun_h4.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_h4.genotypes[ev_h4["event_id"]]["Hap2c#1"] == "."
    assert all("Hap2c#1" not in r["members"].split(",") for r in bun_h4.inserts if r["event_id"] == ev_h4["event_id"])

    p1, p2 = make_complex_seq(40, seed=8643), make_complex_seq(40, seed=8644)
    pref_nodes = {"10": seq_a + seq_c, "rep_n": p1 + p2 + seq_a + seq_c, "p2": p2, "ap": a2, "bp": c2}
    pref_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "rep_n+", "5+", "6+"]) for k in range(1, 5)
    ] + [
        ("Hap2a#1#Chr01", ["1+", "2+", "3+", "4+", "p2+", "ap+", "bp+", "5+", "6+"]),
        ("Hap2b#1#Chr01", ["1+", "2+", "3+", "4+", "p2+", "ap+", "bp+", "5+", "6+"]),
        ("Hap2c#1#Chr01", ["1+", "2+", "3+", "4+", "ap+", "bp+", "5+", "6+"]),
    ]
    bun_pr = _run_cli("cli_i6_h4pref_pan", pref_nodes, pref_paths, "pan")
    ev_pr = next(r for r in bun_pr.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_pr.genotypes[ev_pr["event_id"]]["Hap2c#1"] == "."

    p30, m30, ins25 = make_complex_seq(30, seed=8645), make_complex_seq(30, seed=8646), make_complex_seq(25, seed=8647)
    a7h_nodes = {"10": seq_a + seq_c, "rep_n": p30 + seq_a + m30 + seq_c, "h3": a2 + c2 + ins25}
    a7h_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "rep_n+", "5+", "6+"]) for k in range(1, 5)
    ] + [
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "h3+", "5+", "6+"]),
    ]
    bun_a7h = _run_cli("cli_i6_adv7hom_pan", a7h_nodes, a7h_paths, "pan")
    ev_a7h = next(r for r in bun_a7h.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_a7h.genotypes[ev_a7h["event_id"]]["Hap3#1"] == "."

    # (3) Bug 3: H1c, ATTACK4_big (aL=1535), Topo3a, R4_1D 3 bp microhomology (pan)
    s11, s12 = make_complex_seq(1500, seed=8651), make_complex_seq(1500, seed=8652)
    n14a, n14b = make_complex_seq(500, seed=8653), make_complex_seq(500, seed=8654)
    h1c_alt = n14a[49:] + n14b[:451]
    bun_h1c = _run_cli(
        "cli_i6_h1c_pan",
        {"10": s11 + h1c_alt + s12, "11": s11, "14a": n14a, "14b": n14b, "12": s12, "alt": h1c_alt},
        sup6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14a+", "14b+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14a+", "14b+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_h1c = next(r for r in bun_h1c.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_h1c.genotypes[ev_h1c["event_id"]]["Hap2#1"] == "."

    al1535 = make_complex_seq(1535, seed=8655)
    u0, u1 = make_complex_seq(100, seed=8656), make_complex_seq(55, seed=8657)
    nov90 = u0[65:] + u1
    bun_a4b = _run_cli(
        "cli_i6_atk4big_pan",
        {"10": al1535 + u0[:65] + s12, "11": al1535, "u0": u0, "u1": u1, "12": s12, "alt": u0[:65]},
        sup6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "u0+", "u1+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "u0+", "u1+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_a4b = next(r for r in bun_a4b.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_a4b.genotypes[ev_a4b["event_id"]]["Hap2#1"] == "." and nov90 in bun_a4b.pan_seq
    assert int(bun_a4b.summary_kv["dedup_short_residual_bp"]) == 0

    t3_s0, t3_s1, t3_mid, t3_big = make_complex_seq(100, seed=8661), make_complex_seq(100, seed=8662), make_complex_seq(500, seed=8663), make_complex_seq(100, seed=8664)
    t3_alt = t3_s0[45:] + t3_s1[:55] + t3_mid
    bun_t3a = _run_cli(
        "cli_i6_topo3a_pan",
        {"10": s11 + t3_alt + t3_big + s12, "11": s11, "s0": t3_s0, "s1": t3_s1, "mid": t3_mid, "big": t3_big, "12": s12, "alt": t3_alt},
        sup6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "s0+", "s1+", "mid+", "big+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "s0+", "s1+", "mid+", "big+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_t3a = next(r for r in bun_t3a.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_t3a.genotypes[ev_t3a["event_id"]]["Hap2#1"] == "."

    mh_14 = make_complex_seq(1000, seed=8665)
    mh_m30 = make_complex_seq(27, seed=8666) + mh_14[-3:]
    mh_15 = make_complex_seq(1000, seed=8667)
    mh_t30 = make_complex_seq(30, seed=8668)
    mh_14p = mutate_seq_snps(mh_14, positions=[i * 100 + 20 for i in range(10)], seed=8669)
    mh_15p = mutate_seq_snps(mh_15, positions=[i * 100 + 25 for i in range(10)], seed=8670)
    bun_mh = _run_cli(
        "cli_i6_r41d_mh3_pan",
        {"10": s11 + s12, "11": s11, "14": mh_14, "m30": mh_m30, "15": mh_15, "t30": mh_t30, "12": s12, "20": mh_14p + mh_15p},
        sup6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m30+", "15+", "t30+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "14+", "m30+", "15+", "t30+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "20+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_mh = next(r for r in bun_mh.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    al_mh = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_mh["alleles"].split(";")}
    assert "len=2060" in al_mh["Hap1a#1"] and "len=2000" in al_mh["Hap2#1"]

    # (4) Bug 4 & Refinement #2: fuzz_1_fwd == fuzz_1_rev (ref & pan) + PROBE_3C_TRUE_ZERO (ref & pan)
    fa1, fb1, fd1, fe1 = make_complex_seq(800, seed=8671), make_complex_seq(800, seed=8672), make_complex_seq(300, seed=8673), make_complex_seq(600, seed=8674)
    fs1, fs2 = make_complex_seq(25, seed=8675), make_complex_seq(30, seed=8676)
    f_nodes = {"a1": fa1, "b1": fb1, "d1": fd1, "e1": fe1, "s1": fs1, "s2": fs2}
    for mode in ("ref", "pan"):
        got = {}
        for otag, (ha, hb, hc, hd) in (("fwd", ("HapA", "HapB", "HapC", "HapD")), ("rev", ("HapZ", "HapY", "HapX", "HapW"))):
            bun_f = _run_cli(
                f"cli_i6_fuzz1_{otag}_{mode}",
                f_nodes,
                [
                    (f"{hd}#1#Chr01", ["1+", "2+", "3+", "4+", "a1+", "b1+", "d1+", "e1+", "5+", "6+"]),
                    (f"{ha}#1#Chr01", ["1+", "2+", "3+", "4+", "a1+", "s1+", "b1+", "s2+", "e1+", "5+", "6+"]),
                    (f"{hb}#1#Chr01", ["1+", "2+", "3+", "4+", "a1+", "s1+", "b1+", "s2+", "e1+", "5+", "6+"]),
                    (f"{hc}#1#Chr01", ["1+", "2+", "3+", "4+", "a1+", "b1+", "e1+", "5+", "6+"]),
                ],
                mode,
            )
            ev_f = next(r for r in bun_f.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
            got[otag] = ev_f["alleles"].split(";")[0].split(",")[0]
        assert got["fwd"] == got["rev"] == "1:len=2200"

    p3_13, p3_14 = make_complex_seq(500, seed=8677), make_complex_seq(1000, seed=8678)
    p3_nodes = {"13": p3_13, "14": p3_14, "15": p3_14[:200]}
    p3_paths = [
        (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "13+", "14+", "5+", "6+"]) for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "14+", "15+", "5+", "6+"]),
        ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "15+", "5+", "6+"]),
    ]
    for mode in ("ref", "pan"):
        bun_p3 = _run_cli(f"cli_i6_probe3c_{mode}", p3_nodes, p3_paths, mode)
        ev_p3 = next(r for r in bun_p3.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
        assert "len=0," not in ev_p3["alleles"]
        assert bun_p3.genotypes[ev_p3["event_id"]]["Hap3#1"] == "."

    # (5) Combined multi-locus T22 determinism (--threads 1 vs --threads 4, PANFLAT_FORCE_FORK=1) for pan & off
    bt = SyntheticInputBuilder(tmp_path / "cli_i6_t22", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 14):
        bt.add_node(str(i), make_complex_seq(400, seed=8700 + i))
    for nid, sq in {
        "La": la, "Ra": ra, "AC_T45": seq_a + seq_c + t45, "A2": a2, "X": seq_x, "Y": seq_y, "C2": c2, "T45_2": t45_2,
        "sup_u": al1535 + u0[:65] + s12, "al": al1535, "u0": u0, "u1": u1, "s12": s12, "ualt": u0[:65],
        "p13": p3_13, "p14": p3_14, "p15": p3_14[:200],
        "fa1": fa1, "fb1": fb1, "fd1": fd1, "fe1": fe1, "fs1": fs1, "fs2": fs2,
    }.items():
        bt.add_node(nid, sq)
    bt.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 14)])
    bt.add_gff_gene(gene_id="i6_t22_g", mrna_id="i6_t22_m", strand="+", exons=[(21, 200)], cds_list=[(21, 200, 0)])
    bt.add_bed_region(4800, 5200, region_id="CEN_I6_T22")
    for k in range(1, 7):
        bt.add_path(f"Sup{k}#1#Chr01", ["1+", "sup_u+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    # Locus 3..4: BUG3_DEFAULT_CLI
    for ch in ("a", "b", "c", "d"):
        bt.add_path(f"B3H0{ch}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "AC_T45+", "Ra+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    for ch in ("a", "b"):
        bt.add_path(f"B3H1{ch}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "A2+", "X+", "C2+", "T45_2+", "Ra+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    bt.add_path("B3H2#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "A2+", "X+", "Y+", "C2+", "Ra+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    # Locus 5..6: ATTACK4_big (aL=1535)
    bt.add_path("A4H1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "al+", "u0+", "u1+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    bt.add_path("A4H1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "al+", "u0+", "u1+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    bt.add_path("A4H2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "al+", "ualt+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+", "13+"])
    # Locus 7..8: PROBE_3C_TRUE_ZERO
    for k in range(1, 5):
        bt.add_path(f"Z3H1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "p13+", "p14+", "9+", "10+", "11+", "12+", "13+"])
    bt.add_path("Z3H2#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "p14+", "p15+", "9+", "10+", "11+", "12+", "13+"])
    bt.add_path("Z3H3#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "p15+", "9+", "10+", "11+", "12+", "13+"])
    # Locus 9..10: fuzz_1
    bt.add_path("FZ_D#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "fa1+", "fb1+", "fd1+", "fe1+", "11+", "12+", "13+"])
    bt.add_path("FZ_A#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "fa1+", "fs1+", "fb1+", "fs2+", "fe1+", "11+", "12+", "13+"])
    bt.add_path("FZ_B#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "fa1+", "fs1+", "fb1+", "fs2+", "fe1+", "11+", "12+", "13+"])
    bt.add_path("FZ_C#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "fa1+", "fb1+", "fe1+", "11+", "12+", "13+"])

    inp_t = bt.write("cli_i6_t22")
    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    for mode in ("pan", "off"):
        args = ["--dedup-mode", mode, "--dedup-id", "0.90", "--min-sv-len", "50"]
        o1 = tmp_path / f"cli_i6_t22_{mode}_t1"
        o4 = tmp_path / f"cli_i6_t22_{mode}_t4"
        p1_r, b1_r = run_build(inp_t, extra_args=[*args, "--threads", "1"], out_prefix=o1, extra_env=env)
        p4_r, b4_r = run_build(inp_t, extra_args=[*args, "--threads", "4"], out_prefix=o4, extra_env=env)
        assert p1_r.returncode == 0 and b1_r is not None, p1_r.stderr
        assert p4_r.returncode == 0 and b4_r is not None, p4_r.stderr
        run_verify(o1, inp_t.gfa_path, inp_t.ref_fa_path, check=True)
        _assert_12_files_sha256_identical_and_clean_streams(o1, o4, p1_r, p4_r)
        if mode == "pan":
            assert (seq_x + seq_y) in b1_r.pan_seq and nov90 in b1_r.pan_seq
            assert int(b1_r.summary_kv["dedup_short_residual_bp"]) == 0
            ev_z = next(r for r in b1_r.sv_rows if r["type"] == "INS" and r["anchors"] == "7..8")
            assert "len=0," not in ev_z["alleles"] and b1_r.genotypes[ev_z["event_id"]]["Z3H3#1"] == "."
            ev_fz = next(r for r in b1_r.sv_rows if r["type"] == "INS" and r["anchors"] == "9..10")
            assert "1:len=2200," in ev_fz["alleles"]


def test_tier5_dedup_16_iter7_r42_subslice_finding1_and_t22_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 5 (Iter 7 Closure, STRICTLY ZERO panflat imports):
    (a) Killshot A/D (10 kb gene 1 bp D + TE49 + Sup) & Killshot B/C (30 kb genes, 0 spacers, gt != '.').
    (b) PGGB_1BP_MH + r62_f2_gen (unconditional exact-both-endpoints Phase 0).
    (c) XT-70 + sc==rep0 default --spacer-n 100 (pan).
    (d) Finding 1 Probe 1D (Sup6 survives) + Probe B (L3457 order).
    (e) ATTACK_4A (dead-end mid5 bridge) + MULTI_SUBSLICE_TRAP (Stage 5.5 short-piece run).
    (f) T22 --threads 1 vs --threads 4 (PANFLAT_FORCE_FORK=1) + shuffled P-lines SHA-256 byte-identity
        in both `pan` and `off` modes, with `panflat verify` PASS on every build.
    """
    def _run_cli(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        mode: str,
        n_ref: int = 6,
        dedup_id: str = "0.90",
        spacer_n: int = 100,
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, n_ref + 1):
            b.add_node(str(i), make_complex_seq(400, seed=8800 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, n_ref + 1)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=[
                "--dedup-mode", mode,
                "--dedup-id", dedup_id,
                "--min-sv-len", "50",
                "--spacer-n", str(spacer_n),
            ],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        verify_extra = ["--spacer-n", str(spacer_n)] if spacer_n != 100 else []
        run_verify(out, inp.gfa_path, inp.ref_fa_path, extra_args=verify_extra, check=True)
        return bun

    s11 = make_complex_seq(1500, seed=8811)
    s12 = make_complex_seq(1500, seed=8812)
    sup6_6 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 7)]

    # (a) Killshot A/D (10 kb gene 1 bp D + TE49 + Sup) & Killshot B/C (30 kb genes, 0 spacers, gt != '.')
    ga10 = make_complex_seq(10000, seed=8821)
    te49 = make_complex_seq(49, seed=8822)
    gb5 = make_complex_seq(5000, seed=8823)
    ga10_1d = ga10[:5000] + ga10[5001:]
    ksa_paths_ref = [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "TE49+", "gB+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "TE49+", "gB+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "gB+", "12+", "5+", "6+"]),
    ]
    bun_ka = _run_cli(
        "cli_i7_ksA_ref",
        {"11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga10_1d},
        ksa_paths_ref,
        "ref",
    )
    ev_ka = next(r for r in bun_ka.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_ka = [r for r in bun_ka.inserts if r["event_id"] == ev_ka["event_id"]]
    assert all(r["join_before"] != "spacer" for r in ins_ka)
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in ins_ka if "Hap2#1" in r["members"].split(",")) == 18000

    bun_kd = _run_cli(
        "cli_i7_ksD_pan_095",
        {"10": s11 + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga10_1d},
        sup6_6 + ksa_paths_ref,
        "pan",
        dedup_id="0.95",
    )
    ev_kd = next(r for r in bun_kd.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_kd.genotypes[ev_kd["event_id"]]["Hap2#1"] != "."
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_kd.inserts if r["event_id"] == ev_kd["event_id"] and "Hap2#1" in r["members"].split(",")) == 10000

    g15a, g15b = make_complex_seq(15000, seed=8824), make_complex_seq(15000, seed=8825)
    ksbc_alt = (g15a[:7500] + g15a[7520:]) + g15b[:14970]
    ksbc_paths = [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "g1+", "g2+", "12+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "g1+", "g2+", "12+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
    ]
    bun_kbc_r = _run_cli("cli_i7_ksBC_ref", {"11": s11, "g1": g15a, "g2": g15b, "12": s12, "alt": ksbc_alt}, ksbc_paths, "ref")
    ev_kbcr = next(r for r in bun_kbc_r.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_kbcr = [r for r in bun_kbc_r.inserts if r["event_id"] == ev_kbcr["event_id"]]
    assert all(r["join_before"] != "spacer" for r in ins_kbcr)
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in ins_kbcr if "Hap2#1" in r["members"].split(",")) == 32950

    bun_kbc_p = _run_cli("cli_i7_ksBC_pan_nov", {"10": s11 + s12, "11": s11, "g1": g15a, "g2": g15b, "12": s12, "alt": ksbc_alt}, sup6_6 + ksbc_paths, "pan")
    ev_kbcp = next(r for r in bun_kbc_p.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_kbc_p.genotypes[ev_kbcp["event_id"]]["Hap2#1"] != "."
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_kbc_p.inserts if r["event_id"] == ev_kbcp["event_id"] and "Hap2#1" in r["members"].split(",")) == 29950

    # (b) PGGB_1BP_MH + r62_f2_gen
    ga_1bp = make_complex_seq(1998, seed=8831) + "TC"
    gb_1bp = "G" + make_complex_seq(1999, seed=8832)
    alt_1bp = mutate_seq_snps(ga_1bp, positions=[i * 200 + 30 for i in range(9)], seed=8833) + mutate_seq_snps(gb_1bp, positions=[i * 200 + 30 for i in range(9)], seed=8834)
    bun_1bp = _run_cli(
        "cli_i7_pggb_1bp_pan",
        {"10": s11 + s12, "11": s11, "gA": ga_1bp, "d1": "C", "gB": gb_1bp, "t49": te49, "12": s12, "alt": alt_1bp},
        sup6_6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "d1+", "gB+", "t49+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "d1+", "gB+", "t49+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_1bp = next(r for r in bun_1bp.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_1bp.inserts if r["event_id"] == ev_1bp["event_id"] and "Hap2#1" in r["members"].split(",")) == 4000

    ga_r62 = make_complex_seq(1980, seed=8835) + ("A" * 20)
    mid_r62 = "A" * 30
    d25_r62 = "A" * 25
    gb_r62 = "C" + make_complex_seq(1999, seed=8836)
    t30 = make_complex_seq(30, seed=8837)
    alt_r62 = mutate_seq_snps(ga_r62, positions=[i * 200 + 20 for i in range(9)], seed=8838) + mid_r62 + mutate_seq_snps(gb_r62, positions=[i * 200 + 20 for i in range(9)], seed=8839)
    bun_r62 = _run_cli(
        "cli_i7_r62_f2_gen_ref",
        {"11": s11, "gA": ga_r62, "mid": mid_r62, "d25": d25_r62, "gB": gb_r62, "t30": t30, "12": s12, "alt": alt_r62},
        [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "mid+", "d25+", "gB+", "t30+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "mid+", "d25+", "gB+", "t30+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "12+", "5+", "6+"]),
        ],
        "ref",
    )
    ev_r62 = next(r for r in bun_r62.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_r62.inserts if r["event_id"] == ev_r62["event_id"] and "Hap2#1" in r["members"].split(",")) == 7030

    # (c) XT-70 + sc==rep0 default --spacer-n 100 (pan)
    mid500 = make_complex_seq(500, seed=8841)
    d35a, d35b = make_complex_seq(35, seed=8842), make_complex_seq(35, seed=8843)
    bun_xt = _run_cli(
        "cli_i7_xt70_pan",
        {"10": s11 + mid500 + s12, "11": s11, "d1": d35a, "mid": mid500, "d2": d35b, "12": s12},
        sup6_6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "d1+", "mid+", "d2+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "d1+", "mid+", "d2+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "mid+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_xt = next(r for r in bun_xt.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_xt.genotypes[ev_xt["event_id"]]["Hap1a#1"] == "1"
    assert bun_xt.genotypes[ev_xt["event_id"]]["Hap2#1"] == "."

    sc_la, sc_ra = make_complex_seq(1500, seed=8844), make_complex_seq(1500, seed=8845)
    sc_x, sc_y, sc_z = make_complex_seq(1500, seed=8846), make_complex_seq(1000, seed=8847), make_complex_seq(1500, seed=8848)
    sc_b0, sc_i1 = make_complex_seq(500, seed=8849), make_complex_seq(500, seed=8850)
    sc_rep0 = sc_x + d35a + sc_b0 + sc_i1 + sc_y + d35b + sc_z
    sup10_6 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 11)]
    bun_sc = _run_cli(
        "cli_i7_sc_rep0_sp100_pan",
        {
            "10": sc_la + sc_x + sc_b0 + sc_y + sc_z + sc_ra,
            "La": sc_la, "Ra": sc_ra, "R0": sc_rep0,
            "X": sc_x, "nv1": d35a, "B0": sc_b0, "I1": sc_i1, "Y": sc_y, "nv2": d35b, "Z": sc_z,
        },
        sup10_6 + [
            (f"Hap0_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "R0+", "Ra+", "5+", "6+"]) for k in range(1, 5)
        ] + [
            ("Hap3a#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "X+", "nv1+", "B0+", "Y+", "nv2+", "Z+", "Ra+", "5+", "6+"]),
            ("Hap3b#1#Chr01", ["1+", "2+", "3+", "4+", "La+", "X+", "nv1+", "B0+", "Y+", "nv2+", "Z+", "Ra+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "La+", "X+", "B0+", "Y+", "Z+", "Ra+", "5+", "6+"]),
            ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "La+", "X+", "nv1+", "I1+", "Y+", "nv2+", "Z+", "Ra+", "5+", "6+"]),
        ],
        "pan",
        spacer_n=100,
    )
    ev_sc = next(r for r in bun_sc.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert bun_sc.genotypes[ev_sc["event_id"]]["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in bun_sc.inserts if r["event_id"] == ev_sc["event_id"])

    # (d) Finding 1 Probe 1D (Sup6 at 5..6 survives) + Probe B (L3457 order: 5-real repB before 4-real+2-excl repA)
    f1_13, f1_14 = make_complex_seq(500, seed=8851), make_complex_seq(1000, seed=8852)
    f1_14a, f1_15 = f1_14[:800], f1_14[800:]
    sup6_56_8 = [(f"Sup{k}#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "sup5+", "7+", "8+"]) for k in range(1, 7)]
    bun_1d = _run_cli(
        "cli_i7_probe1d_pan",
        {"13": f1_13, "14": f1_14, "14a": f1_14a, "15": f1_15, "sup5": f1_13 + f1_14},
        sup6_56_8 + [
            (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "13+", "14+", "5+", "6+", "7+", "8+"]) for k in range(1, 5)
        ] + [
            ("Hap2#1#Chr01", ["1+", "2+", "3+", "4+", "13+", "14a+", "15+", "5+", "6+", "7+", "8+"]),
            ("Hap3#1#Chr01", ["1+", "2+", "3+", "4+", "15+", "5+", "6+", "7+", "8+"]),
        ],
        "pan",
        n_ref=8,
    )
    ins_anc_1d = {r["anchors"] for r in bun_1d.sv_rows if r["type"] == "INS"}
    assert "5..6" in ins_anc_1d and "3..4" not in ins_anc_1d

    nov_b = make_complex_seq(1200, seed=8853)
    bun_pb = _run_cli(
        "cli_i7_probeB_pan",
        {"13": f1_13, "14": f1_14, "14a": f1_14a, "15": f1_15, "novB": nov_b},
        [
            (f"Hap1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "13+", "14+", "5+", "6+"]) for k in range(1, 4)
        ] + [
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "13+", "14a+", "15+", "5+", "6+"]),
            ("HapZ1#1#Chr01", ["1+", "2+", "3+", "4+", "15+", "5+", "6+"]),
            ("HapZ2#1#Chr01", ["1+", "2+", "3+", "4+", "15+", "5+", "6+"]),
        ] + [
            (f"HapB_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "novB+", "5+", "6+"]) for k in range(1, 6)
        ],
        "pan",
    )
    ev_pb = next(r for r in bun_pb.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_pb = [r for r in bun_pb.inserts if r["event_id"] == ev_pb["event_id"]]
    assert len(ins_pb) == 2
    assert int(ins_pb[0]["pan_end"]) - int(ins_pb[0]["pan_start"]) == 1200
    assert int(ins_pb[1]["pan_end"]) - int(ins_pb[1]["pan_start"]) == 1500
    assert bun_pb.genotypes[ev_pb["event_id"]]["HapZ1#1"] == "." and bun_pb.genotypes[ev_pb["event_id"]]["HapZ2#1"] == "."

    # (e) ATTACK_4A + MULTI_SUBSLICE_TRAP
    prev200, mid5, cur200 = make_complex_seq(200, seed=8861), make_complex_seq(5, seed=8862), make_complex_seq(200, seed=8863)
    p4a_alt, c4a_alt = prev200[:145] + prev200[180:200], cur200[:20] + cur200[55:200]
    bun_4a = _run_cli(
        "cli_i7_attack_4a_pan",
        {"10": s11 + p4a_alt + mid5 + c4a_alt + s12, "11": s11, "prev": prev200, "mid5": mid5, "cur": cur200, "12": s12, "p_alt": p4a_alt, "c_alt": c4a_alt},
        sup6_6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "prev+", "mid5+", "cur+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "prev+", "mid5+", "cur+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "p_alt+", "mid5+", "c_alt+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_4a = next(r for r in bun_4a.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_4a = [r for r in bun_4a.inserts if r["event_id"] == ev_4a["event_id"]]
    assert bun_4a.genotypes[ev_4a["event_id"]]["Hap2#1"] == "."
    assert len(ins_4a) == 1 and ins_4a[0]["members"] == "Hap1a#1,Hap1b#1"

    # Genuine rep0-merging Refinement C topology (TRUE_MST: both Hap2 & Hap3 merge into rep0 creating contiguous 30 + 29 short pieces)
    l0_m, n100_m, smid_m = make_complex_seq(1500, seed=12311), make_complex_seq(100, seed=12312), make_complex_seq(95, seed=12313)
    ancr_m, r0_m = make_complex_seq(700, seed=12314), make_complex_seq(1500, seed=12315)
    nov_a_m, nov_b_m = make_complex_seq(200, seed=12320), make_complex_seq(200, seed=12321)
    mst_paths_cli = sup6_6 + [
        ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "n100+", "smid+", "ancR+", "R0+", "5+", "6+"]),
        ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "n100+", "smid+", "ancR+", "R0+", "5+", "6+"]),
        ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "alt2+", "R0+", "5+", "6+"]),
        ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "alt3+", "R0+", "5+", "6+"]),
    ]
    mst_nodes_base = {
        "L0": l0_m, "n100": n100_m, "smid": smid_m, "ancR": ancr_m, "R0": r0_m,
        "alt2": smid_m[:60] + ancr_m[:685], "alt3": n100_m + smid_m[30:] + ancr_m[:680],
    }

    # Adjacent-Sup TRUE_MST
    bun_mst_adj = _run_cli(
        "cli_i7_mst_adj_pan",
        {**mst_nodes_base, "10": l0_m + smid_m[:60] + ancr_m[:685] + r0_m},
        mst_paths_cli,
        "pan",
    )
    ev_mst_adj = next(r for r in bun_mst_adj.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_mst_adj = [r for r in bun_mst_adj.inserts if r["event_id"] == ev_mst_adj["event_id"]]
    pan34_mst_adj = "".join(bun_mst_adj.pan_seq[int(r["pan_start"]):int(r["pan_end"])] for r in ins_mst_adj)
    assert bun_mst_adj.genotypes[ev_mst_adj["event_id"]]["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_mst_adj)
    assert smid_m[:60] not in pan34_mst_adj and smid_m[:30] not in pan34_mst_adj and smid_m[30:59] not in pan34_mst_adj
    assert bun_mst_adj.pan_seq.count(smid_m[:60]) == 1
    assert n100_m in pan34_mst_adj and smid_m[60:94] in pan34_mst_adj

    # Separated/ectopic-Sup TRUE_MST_SEPARATED_SUP_pan (unmasked-flank fallback prevents ancR/L0 hijacking)
    bun_mst_sep = _run_cli(
        "cli_i7_mst_sep_pan",
        {**mst_nodes_base, "10": l0_m + nov_a_m + smid_m[:60] + nov_b_m + ancr_m[:685] + r0_m},
        mst_paths_cli,
        "pan",
    )
    ev_mst_sep = next(r for r in bun_mst_sep.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_mst_sep = [r for r in bun_mst_sep.inserts if r["event_id"] == ev_mst_sep["event_id"]]
    pan34_mst_sep = "".join(bun_mst_sep.pan_seq[int(r["pan_start"]):int(r["pan_end"])] for r in ins_mst_sep)
    assert bun_mst_sep.genotypes[ev_mst_sep["event_id"]]["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_mst_sep)
    assert smid_m[:60] not in pan34_mst_sep and smid_m[:30] not in pan34_mst_sep and smid_m[30:59] not in pan34_mst_sep
    assert bun_mst_sep.pan_seq.count(smid_m[:60]) == 1
    assert n100_m in pan34_mst_sep and smid_m[60:94] in pan34_mst_sep

    # ref_c_no_false_drop: isolated 25 bp (< 50 bp) match inside >= 50 bp short-run MUST NOT be dropped
    bun_mst_nfd = _run_cli(
        "cli_i7_mst_nfd_pan",
        {**mst_nodes_base, "10": l0_m + nov_a_m + smid_m[:25] + nov_b_m + ancr_m[:685] + r0_m},
        mst_paths_cli,
        "pan",
    )
    ev_mst_nfd = next(r for r in bun_mst_nfd.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_mst_nfd = [r for r in bun_mst_nfd.inserts if r["event_id"] == ev_mst_nfd["event_id"]]
    pan34_mst_nfd = "".join(bun_mst_nfd.pan_seq[int(r["pan_start"]):int(r["pan_end"])] for r in ins_mst_nfd)
    assert bun_mst_nfd.genotypes[ev_mst_nfd["event_id"]]["Hap2#1"] != "."
    assert smid_m[:30] in pan34_mst_nfd and smid_m[30:59] in pan34_mst_nfd

    # P4_RIPPLE_REAL_pan: 60 bp exact match sm[25:85] across 30+30 short pieces separated from masked R0
    nov250_p4, sm120_p4, t25_p4 = make_complex_seq(250, seed=12401), make_complex_seq(120, seed=12402), make_complex_seq(25, seed=12403)
    bun_p4 = _run_cli(
        "cli_i8_p4_ripple_pan",
        {
            "10": l0_m + nov_a_m + sm120_p4[25:85] + nov_b_m + r0_m,
            "L0": l0_m, "nov250": nov250_p4, "sm": sm120_p4, "t25": t25_p4, "R0": r0_m,
            "a2": sm120_p4[:85], "a3": sm120_p4[25:], "a4": sm120_p4[:25] + sm120_p4[55:],
        },
        sup6_6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "nov250+", "sm+", "t25+", "R0+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "nov250+", "sm+", "t25+", "R0+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "nov250+", "a2+", "R0+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "nov250+", "a3+", "R0+", "5+", "6+"]),
            ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "nov250+", "a4+", "R0+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_p4 = next(r for r in bun_p4.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_p4 = [r for r in bun_p4.inserts if r["event_id"] == ev_p4["event_id"]]
    pan34_p4 = "".join(bun_p4.pan_seq[int(r["pan_start"]):int(r["pan_end"])] for r in ins_p4)
    assert sm120_p4[25:55] not in pan34_p4 and sm120_p4[55:85] not in pan34_p4
    assert sm120_p4[:25] in pan34_p4 and sm120_p4[85:] in pan34_p4 and t25_p4 in pan34_p4

    # Iteration 8 CLI check: TE500 (500 bp >= min_sv_len) + 25 bp internal gene del (Sup == Hap2 -> gt == ".")
    te500_cli = make_complex_seq(500, seed=12411)
    ga10_d25 = ga10[:5000] + ga10[5025:]
    bun_te500 = _run_cli(
        "cli_i8_te500_25d_pan",
        {"10": s11 + ga10_d25 + gb5 + s12, "11": s11, "gA": ga10, "TE500": te500_cli, "gB": gb5, "12": s12, "alt": ga10_d25},
        sup6_6 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "TE500+", "gB+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "TE500+", "gB+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt+", "gB+", "12+", "5+", "6+"]),
        ],
        "pan",
    )
    ev_te500 = next(r for r in bun_te500.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ins_te500 = [r for r in bun_te500.inserts if r["event_id"] == ev_te500["event_id"]]
    assert bun_te500.genotypes[ev_te500["event_id"]]["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_te500)

    # (f) Combined multi-locus T22 determinism (--threads 1 vs --threads 4 + PANFLAT_FORCE_FORK=1 + shuffled P-lines) in pan & off
    t22_nodes = {
        "sup_comb": s11 + gb5 + s12 + p4a_alt + mid5 + c4a_alt,
        "s11": s11, "s12": s12, "gA": ga10, "TE49": te49, "gB": gb5, "ga_1d": ga10_1d,
        "prev": prev200, "mid5": mid5, "cur": cur200, "p_alt": p4a_alt, "c_alt": c4a_alt,
        "f13": f1_13, "f14": f1_14, "f14a": f1_14a, "f15": f1_15, "novB": nov_b,
        "d1": d35a, "mid500": mid500, "d2": d35b,
    }
    t22_alt_paths: list[tuple[str, list[str]]] = []
    for k in range(1, 7):
        t22_alt_paths.append((f"Sup{k}#1#Chr01", ["1+", "sup_comb+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]))
    # Locus 3..4: Killshot A/D
    t22_alt_paths.extend([
        ("KSH1a#1#Chr01", ["1+", "2+", "3+", "4+", "s11+", "gA+", "TE49+", "gB+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("KSH1b#1#Chr01", ["1+", "2+", "3+", "4+", "s11+", "gA+", "TE49+", "gB+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("KSH2#1#Chr01",  ["1+", "2+", "3+", "4+", "s11+", "ga_1d+", "gB+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
    ])
    # Locus 5..6: ATTACK_4A
    t22_alt_paths.extend([
        ("A4H1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "prev+", "mid5+", "cur+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("A4H1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "prev+", "mid5+", "cur+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("A4H2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "p_alt+", "mid5+", "c_alt+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
    ])
    # Locus 7..8: Probe B (Finding 1 L3457 order)
    for k in range(1, 4):
        t22_alt_paths.append((f"PB1_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "f13+", "f14+", "9+", "10+", "11+", "12+"]))
    t22_alt_paths.extend([
        ("PB2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "f13+", "f14a+", "f15+", "9+", "10+", "11+", "12+"]),
        ("PBZ1#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "f15+", "9+", "10+", "11+", "12+"]),
        ("PBZ2#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "f15+", "9+", "10+", "11+", "12+"]),
    ])
    for k in range(1, 6):
        t22_alt_paths.append((f"PBB_{k}#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "novB+", "9+", "10+", "11+", "12+"]))
    # Locus 9..10: XT-70
    t22_alt_paths.extend([
        ("XT1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "d1+", "mid500+", "d2+", "s12+", "11+", "12+"]),
        ("XT1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "d1+", "mid500+", "d2+", "s12+", "11+", "12+"]),
        ("XT2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "mid500+", "s12+", "11+", "12+"]),
    ])

    def _build_t22_inp(tag: str, alt_paths: list[tuple[str, list[str]]]) -> InputPaths:
        bt = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 13):
            bt.add_node(str(i), make_complex_seq(400, seed=8900 + i))
        for nid, sq in t22_nodes.items():
            bt.add_node(nid, sq)
        bt.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 13)])
        bt.add_gff_gene(gene_id="i7_t22_g", mrna_id="i7_t22_m", strand="+", exons=[(21, 200)], cds_list=[(21, 200, 0)])
        bt.add_bed_region(4400, 4800, region_id="CEN_I7_T22")
        for pname, steps in alt_paths:
            bt.add_path(pname, steps)
        return bt.write(tag)

    inp_orig = _build_t22_inp("cli_i7_t22_orig", t22_alt_paths)
    shuf_paths = list(t22_alt_paths)
    random.Random(20261126).shuffle(shuf_paths)
    inp_shuf = _build_t22_inp("cli_i7_t22_shuf", shuf_paths)
    sha_orig_gfa = _sha256_file(inp_orig.gfa_path)
    sha_shuf_gfa = _sha256_file(inp_shuf.gfa_path)

    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    for mode in ("pan", "off"):
        args = ["--dedup-mode", mode, "--dedup-id", "0.90", "--min-sv-len", "50"]
        o1 = tmp_path / f"cli_i7_t22_{mode}_t1"
        o4 = tmp_path / f"cli_i7_t22_{mode}_t4"
        os4 = tmp_path / f"cli_i7_t22_{mode}_shuf_t4"
        p1_r, b1_r = run_build(inp_orig, extra_args=[*args, "--threads", "1"], out_prefix=o1, extra_env=env)
        p4_r, b4_r = run_build(inp_orig, extra_args=[*args, "--threads", "4"], out_prefix=o4, extra_env=env)
        ps_r, bs_r = run_build(inp_shuf, extra_args=[*args, "--threads", "4"], out_prefix=os4, extra_env=env)
        assert p1_r.returncode == 0 and b1_r is not None, p1_r.stderr
        assert p4_r.returncode == 0 and b4_r is not None, p4_r.stderr
        assert ps_r.returncode == 0 and bs_r is not None, ps_r.stderr
        run_verify(o1, inp_orig.gfa_path, inp_orig.ref_fa_path, check=True)
        run_verify(os4, inp_shuf.gfa_path, inp_shuf.ref_fa_path, check=True)
        # 12-file SHA-256 byte-identity (--threads 1 vs --threads 4)
        _assert_12_files_sha256_identical_and_clean_streams(o1, o4, p1_r, p4_r)
        # Shuffled P-lines invariance: 10 biological files byte-identical + normalized summary/log identical
        for suf in _BIO_10_OUTPUT_SUFFIXES:
            assert _sha256_file(Path(f"{o1}{suf}")) == _sha256_file(Path(f"{os4}{suf}")), (
                f"Shuffled GFA mismatch on {mode} {suf}"
            )
        for meta_suf in (".summary.tsv", ".log"):
            norm_o = _normalize_gfa_meta_text(Path(f"{o1}{meta_suf}").read_text(), inp_orig.gfa_path, sha_orig_gfa)
            norm_s = _normalize_gfa_meta_text(Path(f"{os4}{meta_suf}").read_text(), inp_shuf.gfa_path, sha_shuf_gfa)
            assert norm_o == norm_s, f"Shuffled GFA meta mismatch on {mode} {meta_suf}"
        if mode == "pan":
            ev_ks = next(r for r in b1_r.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
            assert b1_r.genotypes[ev_ks["event_id"]]["KSH2#1"] != "."
            ev_a4 = next(r for r in b1_r.sv_rows if r["type"] == "INS" and r["anchors"] == "5..6")
            assert b1_r.genotypes[ev_a4["event_id"]]["A4H2#1"] == "."
            ev_pbt = next(r for r in b1_r.sv_rows if r["type"] == "INS" and r["anchors"] == "7..8")
            assert b1_r.genotypes[ev_pbt["event_id"]]["PBZ1#1"] == "." and b1_r.genotypes[ev_pbt["event_id"]]["PBZ2#1"] == "."
            ins_pbt = [r for r in b1_r.inserts if r["event_id"] == ev_pbt["event_id"]]
            assert int(ins_pbt[0]["pan_end"]) - int(ins_pbt[0]["pan_start"]) == 1200


def test_tier5_dedup_17_iter9_unified_stage55_jacobi_and_t22_cli(tmp_path: Path) -> None:
    """Opaque-box CLI E2E 6 (Iter 9 Closure, STRICTLY ZERO panflat imports):
    1. CLI_BUG1 (PGGB 1 bp GFA bubble) + ZERO_EXCL_PURE_AC_SB4 (gt[Hap2] == '.', 0 Hap2 members).
    2. HITS_1_SELF_HIJACK_GHOST_SUPH2 (gt[Hap2] == '.', 0 ghost allele).
    3. CLI_BUG2 forward + mirror (gt[Hap4] == '.', pan_seq.count(s0) == 1).
    4. CLI_BUG3 (5 bp 1-SNP sliver, gt[Hap2] == '.', 0 Hap2 ghost allele).
    5. CROSS_WAVE_ISO_45BP_REQ0 (s45 preserved in pan.fa).
    6. Combined trial=22 + LIVE_C_TC_GG + DEFAULT090_2HOP fwd/mir + s20_pr fwd/mir CLI
       + T22 --threads 1 vs --threads 4 (PANFLAT_FORCE_FORK=1) + shuffled GFA SHA-256 identity.
    """
    def _run_i9_cli(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[str]]],
        n_ref: int = 6,
        dedup_id: str = "0.90",
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, n_ref + 1):
            b.add_node(str(i), make_complex_seq(400, seed=19800 + i))
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, n_ref + 1)])
        for pname, steps in paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", "pan", "--dedup-id", dedup_id, "--min-sv-len", "50"],
            out_prefix=out,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} failed: {proc.stderr}"
        run_verify(out, inp.gfa_path, inp.ref_fa_path, check=True)
        return bun

    def _assert_cli_clean(bun, anchors: str = "3..4", hap: str = "Hap2#1") -> None:
        evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == anchors]
        if not evs:
            return
        ev = evs[0]
        assert bun.genotypes[ev["event_id"]][hap] == ".", (
            f"Expected {hap} == '.', got {bun.genotypes[ev['event_id']][hap]} ({ev['alleles']})"
        )
        assert all(
            hap not in r["members"].split(",")
            for r in bun.inserts
            if r["event_id"] == ev["event_id"]
        )

    def _cli_pan_at(bun, anchors: str = "3..4") -> str:
        ev = next(r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == anchors)
        return "".join(
            bun.pan_seq[int(r["pan_start"]) : int(r["pan_end"])]
            for r in bun.inserts
            if r["event_id"] == ev["event_id"]
        )

    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    s11, s12 = make_complex_seq(1500, seed=19811), make_complex_seq(1500, seed=19812)
    ga5, ga10 = make_complex_seq(5000, seed=19813), make_complex_seq(10000, seed=19814)
    ta50, tb50 = make_complex_seq(50, seed=19815), make_complex_seq(50, seed=19816)
    nov_l, nov_r = make_complex_seq(250, seed=19817), make_complex_seq(250, seed=19818)
    nov_a, nov_b = make_complex_seq(200, seed=19819), make_complex_seq(200, seed=19820)
    s30a, s30b = make_complex_seq(30, seed=19821), make_complex_seq(30, seed=19822)
    ins_l200, ins_r200 = make_complex_seq(200, seed=19823), make_complex_seq(200, seed=19824)
    tail50 = make_complex_seq(50, seed=19825)

    # (1) CLI_BUG1 (PGGB 1 bp GFA bubble) + ZERO_EXCL_PURE_AC_SB4
    bun_b1 = _run_i9_cli(
        "cli_i9_bug1",
        {"10": s11 + ga5 + ta50 + s12, "11": s11, "sb": "A", "gA": ga5, "TE": ta50, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "sb+", "gA+", "TE+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "sb+", "gA+", "TE+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "gA+", "TE+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "gA+", "12+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_b1, "3..4", "Hap2#1")

    bun_ac = _run_i9_cli(
        "cli_i9_zero_excl_ac_sb4",
        {"10": s11 + s12, "11": s11, "sb": "ACGT", "insL": ins_l200, "insR": ins_r200, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "sb+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "sb+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "insL+", "sb+", "insR+", "12+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_ac, "3..4", "Hap2#1")

    # (2) HITS_1_SELF_HIJACK_GHOST_SUPH2
    bun_h1 = _run_i9_cli(
        "cli_i9_hits1",
        {"10": s11 + nov_a + s30a + s30b + nov_b + s12, "L0": s11, "novL": nov_l, "s30a": s30a, "s30b": s30b, "novR": nov_r, "R0": s12},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novL+", "s30a+", "s30b+", "novR+", "R0+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novL+", "s30a+", "s30b+", "novR+", "R0+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s30a+", "s30b+", "R0+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novL+", "s30a+", "R0+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_h1, "3..4", "Hap2#1")
    p34_h1 = _cli_pan_at(bun_h1, "3..4")
    assert s30a not in p34_h1 and s30b not in p34_h1 and nov_l in p34_h1 and nov_r in p34_h1

    # (3) CLI_BUG2 forward + mirror
    s0_b2, s1_b2 = make_complex_seq(20, seed=19831), make_complex_seq(20, seed=19832)
    r0_50 = make_complex_seq(50, seed=19833)
    ta_60, tb_60 = make_complex_seq(60, seed=19834), make_complex_seq(60, seed=19835)
    te_1500 = make_complex_seq(1500, seed=19836)
    bun_b2f = _run_i9_cli(
        "cli_i9_bug2_fwd",
        {"10": s11 + nov_a + s0_b2 + s1_b2 + r0_50 + nov_b + te_1500, "L0": s11, "s0": s0_b2, "s1": s1_b2, "R0": r0_50, "ta": ta_60, "tb": tb_60, "TE": te_1500},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s0+", "s1+", "R0+", "ta+", "tb+", "TE+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s0+", "s1+", "R0+", "ta+", "tb+", "TE+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s1+", "R0+", "TE+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s0+", "R0+", "TE+", "5+", "6+"]),
            ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s0+", "s1+", "R0+", "TE+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_b2f, "3..4", "Hap4#1")
    assert bun_b2f.pan_seq.count(s0_b2) == 1 and bun_b2f.pan_seq.count(s1_b2) == 1

    bun_b2m = _run_i9_cli(
        "cli_i9_bug2_mir",
        {"10": te_1500 + nov_b + r0_50 + s0_b2 + s1_b2 + nov_a + s11, "TE": te_1500, "tb": tb_60, "ta": ta_60, "R0": r0_50, "s0": s0_b2, "s1": s1_b2, "L0": s11},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "TE+", "tb+", "ta+", "R0+", "s0+", "s1+", "L0+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "TE+", "tb+", "ta+", "R0+", "s0+", "s1+", "L0+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "TE+", "R0+", "s1+", "L0+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "TE+", "R0+", "s0+", "L0+", "5+", "6+"]),
            ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "TE+", "R0+", "s0+", "s1+", "L0+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_b2m, "3..4", "Hap4#1")
    assert bun_b2m.pan_seq.count(s0_b2) == 1 and bun_b2m.pan_seq.count(s1_b2) == 1

    # (4) CLI_BUG3 (5 bp 1-SNP sliver)
    ga_h2_b3 = mutate_seq_snps(ga10, positions=[5012], seed=19840)
    ga_h2_b3 = ga_h2_b3[:5000] + ga_h2_b3[5010:]
    ga_h3_b3 = ga10[:5000] + ga10[5015:]
    bun_b3 = _run_i9_cli(
        "cli_i9_bug3",
        {"10": s11 + ga_h2_b3 + tb50 + s12, "11": s11, "gA": ga10, "ta": ta50, "tb": tb50, "12": s12, "alt2": ga_h2_b3, "alt3": ga_h3_b3},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "ta+", "tb+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "ta+", "tb+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt2+", "tb+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt3+", "ta+", "12+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_b3, "3..4", "Hap2#1")

    # (5) CROSS_WAVE_ISO_45BP_REQ0 (s45 preserved in pan.fa)
    s45 = make_complex_seq(45, seed=19841)
    bun_45 = _run_i9_cli(
        "cli_i9_iso45",
        {"10": s11 + s12, "10b": nov_a + s45 + nov_b, "L0": s11, "s45": s45, "t50": tail50, "R0": s12},
        sup8 + [
            (f"SupB{k}#1#Chr01", ["1+", "2+", "10b+", "3+", "4+", "5+", "6+"]) for k in range(1, 6)
        ] + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s45+", "t50+", "R0+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s45+", "t50+", "R0+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "R0+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s45+", "R0+", "5+", "6+"]),
        ],
    )
    p34_45_cli = _cli_pan_at(bun_45, "3..4")
    assert s45 in p34_45_cli and tail50 in p34_45_cli

    # (6) Combined trial=22 + LIVE_C_TC_GG + DEFAULT090_2HOP fwd/mir + s20_pr fwd/mir
    #     + T22 --threads 1 vs --threads 4 (PANFLAT_FORCE_FORK=1) + shuffled GFA SHA-256!
    pb_d90 = make_complex_seq(19, seed=19851)
    pr_nov50 = make_complex_seq(50, seed=19852)
    pr_msk250 = make_complex_seq(250, seed=19853)
    s20_ctg = make_complex_seq(17, seed=19854) + "CTG"
    nov_b_ctg = "CTG" + make_complex_seq(197, seed=19855)

    # Mirror s20_pr & DEFAULT090_2HOP standalone quick verify
    bun_sp_mir = _run_i9_cli(
        "cli_i9_s20_pr_mir",
        {"10": s11 + pr_msk250 + nov_a + pr_nov50 + s20_ctg + nov_b_ctg + s12, "L0": s11, "t50": tail50, "Pl": pr_msk250 + pr_nov50, "s20": s20_ctg, "R0": s12},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "t50+", "Pl+", "s20+", "R0+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "t50+", "Pl+", "s20+", "R0+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "Pl+", "R0+", "5+", "6+"]),
        ],
    )
    assert s20_ctg not in _cli_pan_at(bun_sp_mir, "3..4")

    m_s11, m_s12 = revcomp(s12), revcomp(s11)
    m_ta, m_tb = revcomp(tb50), revcomp(ta50)
    m_pa_d90, m_sb_d90, m_pb_d90 = revcomp(pb_d90), revcomp("TAAAAAA"), revcomp("GGGAAAAAA")
    bun_d90_mir = _run_i9_cli(
        "cli_i9_d90_2hop_mir",
        {"10": m_s11 + m_ta + m_pa_d90 + m_pb_d90 + m_tb + m_s12, "11": m_s11, "ta": m_ta, "pa": m_pa_d90, "sb": m_sb_d90, "pb": m_pb_d90, "tb": m_tb, "12": m_s12},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "ta+", "pa+", "sb+", "pb+", "tb+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "ta+", "pa+", "sb+", "pb+", "tb+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "ta+", "pa+", "pb+", "tb+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "pa+", "sb+", "pb+", "12+", "5+", "6+"]),
        ],
    )
    _assert_cli_clean(bun_d90_mir, "3..4", "Hap2#1")

    # Combined multi-locus T22 graph (12 ref nodes):
    # 3..4: trial=22 | 5..6: LIVE_C_TC_GG | 7..8: DEFAULT090_2HOP_FWD | 9..10: s20_pr_fwd
    comb_nodes = {
        "sup_t22": s11 + ta50 + "ACCAATTAC" + "AAAGGAACCTCATTCAGGT" + tb50 + s12,
        "sup_ctc": s11 + ta50 + "C" + "GG" + tb50 + s12,
        "sup_d90": s11 + ta50 + "GGGAAAAAA" + pb_d90 + tb50 + s12,
        "sup_spr": s11 + nov_a + s20_ctg + pr_nov50 + nov_b_ctg + pr_msk250 + s12,
        "s11": s11, "s12": s12, "ta": ta50, "tb": tb50, "t50": tail50,
        "t22_pa": "ACCAATTAC", "t22_sb": "TC", "t22_pb": "AAAGGAACCTCATTCAGGT",
        "ctc_pa": "C", "ctc_sb": "TC", "ctc_pb": "GG",
        "d90_pa": "GGGAAAAAA", "d90_sb": "TAAAAAA", "d90_pb": pb_d90,
        "s20": s20_ctg, "Pr": pr_nov50 + pr_msk250,
    }
    comb_paths: list[tuple[str, list[str]]] = []
    for k in range(1, 9):
        comb_paths.append((
            f"Sup{k}#1#Chr01",
            ["1+", "sup_t22+", "sup_ctc+", "sup_d90+", "sup_spr+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"],
        ))
    # 3..4: trial=22
    comb_paths.extend([
        ("T22H1a#1#Chr01", ["1+", "2+", "3+", "4+", "s11+", "ta+", "t22_pa+", "t22_sb+", "t22_pb+", "tb+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("T22H1b#1#Chr01", ["1+", "2+", "3+", "4+", "s11+", "ta+", "t22_pa+", "t22_sb+", "t22_pb+", "tb+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("T22H2#1#Chr01",  ["1+", "2+", "3+", "4+", "s11+", "ta+", "t22_pa+", "t22_pb+", "tb+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("T22H3#1#Chr01",  ["1+", "2+", "3+", "4+", "s11+", "t22_pa+", "t22_sb+", "t22_pb+", "s12+", "5+", "6+", "7+", "8+", "9+", "10+", "11+", "12+"]),
    ])
    # 5..6: LIVE_C_TC_GG
    comb_paths.extend([
        ("CTCH1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "ta+", "ctc_pa+", "ctc_sb+", "ctc_pb+", "tb+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("CTCH1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "ta+", "ctc_pa+", "ctc_sb+", "ctc_pb+", "tb+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("CTCH2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "ta+", "ctc_pa+", "ctc_pb+", "tb+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
        ("CTCH3#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "s11+", "ctc_pa+", "ctc_sb+", "ctc_pb+", "s12+", "7+", "8+", "9+", "10+", "11+", "12+"]),
    ])
    # 7..8: DEFAULT090_2HOP_FWD
    comb_paths.extend([
        ("D90H1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "s11+", "ta+", "d90_pa+", "d90_sb+", "d90_pb+", "tb+", "s12+", "9+", "10+", "11+", "12+"]),
        ("D90H1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "s11+", "ta+", "d90_pa+", "d90_sb+", "d90_pb+", "tb+", "s12+", "9+", "10+", "11+", "12+"]),
        ("D90H2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "s11+", "ta+", "d90_pa+", "d90_pb+", "tb+", "s12+", "9+", "10+", "11+", "12+"]),
        ("D90H3#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "s11+", "d90_pa+", "d90_sb+", "d90_pb+", "s12+", "9+", "10+", "11+", "12+"]),
    ])
    # 9..10: s20_pr_fwd
    comb_paths.extend([
        ("SPRH1a#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "s20+", "Pr+", "t50+", "s12+", "11+", "12+"]),
        ("SPRH1b#1#Chr01", ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "s20+", "Pr+", "t50+", "s12+", "11+", "12+"]),
        ("SPRH2#1#Chr01",  ["1+", "2+", "3+", "4+", "5+", "6+", "7+", "8+", "9+", "10+", "s11+", "Pr+", "s12+", "11+", "12+"]),
    ])

    def _build_i9_t22_inp(tag: str, alt_paths: list[tuple[str, list[str]]]) -> InputPaths:
        bt = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for i in range(1, 13):
            bt.add_node(str(i), make_complex_seq(400, seed=19900 + i))
        for nid, sq in comb_nodes.items():
            bt.add_node(nid, sq)
        bt.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 13)])
        bt.add_gff_gene(gene_id="i9_t22_g", mrna_id="i9_t22_m", strand="+", exons=[(21, 200)], cds_list=[(21, 200, 0)])
        bt.add_bed_region(4400, 4800, region_id="CEN_I9_T22")
        for pname, steps in alt_paths:
            bt.add_path(pname, steps)
        return bt.write(tag)

    inp_o = _build_i9_t22_inp("cli_i9_t22_orig", comb_paths)
    shuf_p = list(comb_paths)
    random.Random(202611269).shuffle(shuf_p)
    inp_s = _build_i9_t22_inp("cli_i9_t22_shuf", shuf_p)
    sha_o, sha_s = _sha256_file(inp_o.gfa_path), _sha256_file(inp_s.gfa_path)

    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    o1 = tmp_path / "cli_i9_t22_pan_t1"
    o4 = tmp_path / "cli_i9_t22_pan_t4"
    os4 = tmp_path / "cli_i9_t22_pan_shuf_t4"
    base_args = ["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "50"]
    p1, b1 = run_build(inp_o, extra_args=[*base_args, "--threads", "1"], out_prefix=o1, extra_env=env)
    p4, b4 = run_build(inp_o, extra_args=[*base_args, "--threads", "4"], out_prefix=o4, extra_env=env)
    ps, bs = run_build(inp_s, extra_args=[*base_args, "--threads", "4"], out_prefix=os4, extra_env=env)
    assert p1.returncode == 0 and b1 is not None, p1.stderr
    assert p4.returncode == 0 and b4 is not None, p4.stderr
    assert ps.returncode == 0 and bs is not None, ps.stderr
    run_verify(o1, inp_o.gfa_path, inp_o.ref_fa_path, check=True)
    run_verify(os4, inp_s.gfa_path, inp_s.ref_fa_path, check=True)

    _assert_12_files_sha256_identical_and_clean_streams(o1, o4, p1, p4)
    for suf in _BIO_10_OUTPUT_SUFFIXES:
        assert _sha256_file(Path(f"{o1}{suf}")) == _sha256_file(Path(f"{os4}{suf}")), (
            f"Shuffled GFA mismatch on {suf}"
        )
    for meta_suf in (".summary.tsv", ".log"):
        assert _normalize_gfa_meta_text(Path(f"{o1}{meta_suf}").read_text(), inp_o.gfa_path, sha_o) == (
            _normalize_gfa_meta_text(Path(f"{os4}{meta_suf}").read_text(), inp_s.gfa_path, sha_s)
        )
    _assert_cli_clean(b1, "3..4", "T22H2#1")
    _assert_cli_clean(b1, "5..6", "CTCH2#1")
    _assert_cli_clean(b1, "7..8", "D90H2#1")
    assert s20_ctg not in _cli_pan_at(b1, "9..10")


def test_e2e_iter10_zero_penalty_indel_x_window_canon_and_attack5l(tmp_path: Path) -> None:
    """Iter 10 Strictly Opaque-Box CLI E2E Regressions (ZERO `panflat` imports):
    1. `BUG-R9-1a` (`rev_0` inverted-`Sup` `13 kb` `I+X`), `BUG-R9-1b` (`fwd_short_4`
       100% forward-strand `280 bp` `I+M+X`), `BUG-R9-1c` (`D(4)+X(1)` `7 kb` `FWD` & `REV`).
    2. `BUG1_D10_0.90` (`FWD` & `MIR`), `BUG1b_D4_0.95` (`FWD` & `MIR`), `BUG2a_I7_0.90`
       (genuine `6 bp`-MH `FWD` & `MIR`), `BUG2b_I2_0.99` (`FWD` & `MIR`), and `5O-2a`
       (`s20|sb10` `9 bp`-MH novel ins conserved `FWD` & `MIR`).
    3. `BUG-R9-2` `Attack 5L` (`FWD` & `MIR`) + `Attack 5L_split` (`FWD` & `MIR`).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut_pos(seq: str, positions: Sequence[int]) -> str:
        tr = {"A": "C", "C": "G", "G": "T", "T": "A"}
        arr = list(seq)
        for p in positions:
            arr[p] = tr[arr[p]]
        return "".join(arr)

    base_ref6 = {str(i): _mseq(400, 88500 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    sup15 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 16)]

    def _run_i10_cli(
        tag: str,
        extra_nodes: dict[str, str],
        alt_paths: list[tuple[str, list[str]]],
        dedup_id: str = "0.90",
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, sq in base_ref6.items():
            b.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ref6_steps)
        for pname, steps in alt_paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out_pref = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", "pan", "--dedup-id", dedup_id, "--min-sv-len", "50"],
            out_prefix=out_pref,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
        run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
        return bun

    def _assert_clean_hap(bun, hap: str, anchors: str = "3..4") -> None:
        evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == anchors]
        assert evs, f"Missing expected INS event at {anchors}"
        ev = evs[0]
        assert bun.genotypes[ev["event_id"]][hap] == ".", (
            f"Expected {hap} == '.', got {bun.genotypes[ev['event_id']][hap]} ({ev['alleles']})"
        )
        assert all(
            hap not in r["members"].split(",")
            for r in bun.inserts
            if r["event_id"] == ev["event_id"]
        )

    s11, s12 = _mseq(1500, 88511), _mseq(1500, 88512)
    ta, tb = _mseq(50, 88513), _mseq(50, 88514)

    # (1a) BUG-R9-1a: rev_0 inverted-Sup (- strand) 13 kb I+X
    ga10 = _mseq(10000, 88600)
    alt2_1a = _mut_pos(ga10, [5010])
    alt2_1a = alt2_1a[:5000] + alt2_1a[5010:]
    alt3_1a = ga10[:5000] + ga10[5013:]
    bun_1a = _run_i10_cli(
        "i10_bug_r9_1a",
        {"10": revcomp(s11 + alt2_1a + tb + s12), "11": s11, "gA": ga10, "ta": ta, "tb": tb, "12": s12, "alt2": alt2_1a, "alt3": alt3_1a},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "ta+", "tb+", "12+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gA+", "ta+", "tb+", "12+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt2+", "tb+", "12+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt3+", "ta+", "12+", "5+", "6+"]),
        ],
    )
    _assert_clean_hap(bun_1a, "Hap2#1")

    # (1b) BUG-R9-1b: fwd_short_4 100% forward-strand (+) 280 bp I+M+X
    ga_s = _mseq(240, 88704)
    alt2_1b = _mut_pos(ga_s, [110])
    alt2_1b = alt2_1b[:100] + alt2_1b[110:]
    alt3_1b = ga_s[:100] + ga_s[113:]
    bun_1b = _run_i10_cli(
        "i10_bug_r9_1b",
        {"10": alt2_1b + tb, "gA": ga_s, "ta": ta, "tb": tb, "alt2": alt2_1b, "alt3": alt3_1b},
        sup8 + [
            ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "gA+", "ta+", "tb+", "5+", "6+"]),
            ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "gA+", "ta+", "tb+", "5+", "6+"]),
            ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "alt2+", "tb+", "5+", "6+"]),
            ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "alt3+", "ta+", "5+", "6+"]),
        ],
    )
    _assert_clean_hap(bun_1b, "Hap2#1")

    # (1c) BUG-R9-1c: D(4)+X(1) 7 kb FWD (+) and REV (-)
    g_l2, g_r2 = _mseq(2000, 99101), _mseq(2000, 99102)
    alt2_1c = g_l2 + "ACTCCAC" + g_r2
    for tag_1c, sup_1c in (
        ("i10_bug_r9_1c_fwd", s11 + alt2_1c + tb + s12),
        ("i10_bug_r9_1c_rev", revcomp(s11 + alt2_1c + tb + s12)),
    ):
        bun_1c = _run_i10_cli(
            tag_1c,
            {"10": sup_1c, "11": s11, "gL": g_l2, "sliv": "ACG", "gR": g_r2, "ta": ta, "tb": tb, "12": s12, "alt2": alt2_1c},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gL+", "sliv+", "gR+", "ta+", "tb+", "12+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "11+", "gL+", "sliv+", "gR+", "ta+", "tb+", "12+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "alt2+", "tb+", "12+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "11+", "gL+", "gR+", "ta+", "12+", "5+", "6+"]),
            ],
        )
        _assert_clean_hap(bun_1c, "Hap2#1")

    # (2) BUG1_D10_0.90, BUG1b_D4_0.95, genuine 6 bp-MH BUG2a_I7_0.90, BUG2b_I2_0.99, and 5O-2a (s20|sb10) FWD & MIR
    l0, r0 = _mseq(1500, 33101), _mseq(1500, 33102)
    nov_r, nov_sup = _mseq(200, 33103), _mseq(200, 33104)
    s20_d = _mseq(19, 33105) + "A"
    s40_d = "G" + _mseq(39, 33106)

    for tag_b1, d_seq, did in (
        ("i10_bug1_d10", "CGTACGTACA", "0.90"),
        ("i10_bug1b_d4", "CGTA", "0.95"),
    ):
        # FWD
        bun_f = _run_i10_cli(
            f"{tag_b1}_fwd",
            {"10": l0 + s20_d + d_seq + s40_d + nov_sup + r0, "L0": l0, "s20": s20_d, "s40": s40_d, "novR": nov_r, "R0": r0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "novR+", "R0+", "5+", "6+"]),
            ],
            dedup_id=did,
        )
        _assert_clean_hap(bun_f, "Hap3#1")
        assert bun_f.pan_seq.count(s20_d) == 1
        # MIR
        ml0, mr0, mnov_r, mnov_s = revcomp(r0), revcomp(l0), revcomp(nov_r), revcomp(nov_sup)
        ms20, md, ms40 = revcomp(s20_d), revcomp(d_seq), revcomp(s40_d)
        bun_m = _run_i10_cli(
            f"{tag_b1}_mir",
            {"10": ml0 + mnov_s + ms40 + md + ms20 + mr0, "L0": ml0, "novR": mnov_r, "s40": ms40, "s20": ms20, "R0": mr0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR+", "s40+", "s20+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR+", "s40+", "s20+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR+", "s40+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR+", "s20+", "R0+", "5+", "6+"]),
            ],
            dedup_id=did,
        )
        _assert_clean_hap(bun_m, "Hap3#1")
        assert bun_m.pan_seq.count(ms20) == 1

    # Genuine 6 bp-MH BUG2a_I7_0.90, BUG2b_I2_0.99, and 5O-2a (s20|sb10 9 bp-MH) FWD & MIR
    s20_2a = _mseq(13, 33201) + "G" + "ACGTTC"
    sb7_2a = "TACGTTC"
    s25_2a = _mseq(25, 33202)
    s20_2b = _mseq(19, 33301) + "A"
    sb2_2b = "CA"
    s40_2b = "G" + _mseq(39, 33302)
    mh9_5o2 = "ACGTTCGAC"
    s20_5o2 = _mseq(11, 33401) + mh9_5o2
    sb10_5o2 = "T" + mh9_5o2
    s25_5o2 = _mseq(25, 33402)

    for tag_b2, s20_v, sb_v, sr_v, did, check_sb_cnt in (
        ("i10_bug2a_6mh", s20_2a, sb7_2a, s25_2a, "0.90", False),
        ("i10_bug2b_i2", s20_2b, sb2_2b, s40_2b, "0.99", False),
        ("i10_5o2a_s20_sb10", s20_5o2, sb10_5o2, s25_5o2, "0.90", True),
    ):
        # FWD
        bun_2f = _run_i10_cli(
            f"{tag_b2}_fwd",
            {"10": l0 + s20_v + sr_v + nov_sup + r0, "L0": l0, "s20": s20_v, "sb": sb_v, "sR": sr_v, "novR": nov_r, "R0": r0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "sb+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "sb+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "sb+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "sb+", "novR+", "R0+", "5+", "6+"]),
            ],
            dedup_id=did,
        )
        _assert_clean_hap(bun_2f, "Hap2#1")
        assert bun_2f.pan_seq.count(s20_v) == 1
        if check_sb_cnt:
            assert bun_2f.pan_seq.count(sb_v) == 1
        # MIR
        ml0, mr0, mnov_r, mnov_s = revcomp(r0), revcomp(l0), revcomp(nov_r), revcomp(nov_sup)
        ms20, msb, msr = revcomp(s20_v), revcomp(sb_v), revcomp(sr_v)
        bun_2m = _run_i10_cli(
            f"{tag_b2}_mir",
            {"10": ml0 + mnov_s + msr + ms20 + mr0, "L0": ml0, "novR": mnov_r, "sR": msr, "sb": msb, "s20": ms20, "R0": mr0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR+", "sR+", "sb+", "s20+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR+", "sR+", "sb+", "s20+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR+", "sR+", "sb+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR+", "sb+", "s20+", "R0+", "5+", "6+"]),
            ],
            dedup_id=did,
        )
        _assert_clean_hap(bun_2m, "Hap2#1")
        assert bun_2m.pan_seq.count(ms20) == 1
        if check_sb_cnt:
            assert bun_2m.pan_seq.count(msb) == 1

    # (3) BUG-R9-2: Attack 5L (FWD & MIR) + Attack 5L_split (FWD & MIR)
    xs = [_mseq(40, 94000 + k) for k in range(4)]
    s2_35, s1_10, s0_15 = _mseq(35, 94010), _mseq(10, 94011), _mseq(15, 94012)
    p500a, p500b, p500c = _mseq(500, 94020), _mseq(500, 94021), _mseq(500, 94022)
    ta60, tb60 = _mseq(60, 94030), _mseq(60, 94031)
    sup_5l_seq = p500a + "".join(xs) + s2_35 + p500b + s2_35 + s1_10 + s0_15 + p500c

    def _wrap34(mid: list[str]) -> list[str]:
        return ["1+", "2+", "3+", "4+", *mid, "5+", "6+"]

    for variant_tag, split_s2 in (("attack5l", False), ("attack5l_split", True)):
        for is_mir in (False, True):
            ori_s = revcomp if is_mir else (lambda s: s)
            if split_s2:
                raw_items = [
                    ("x0", xs[0]), ("x1", xs[1]), ("x2", xs[2]), ("x3", xs[3]),
                    ("s2a", s2_35[:18]), ("s2b", s2_35[18:]), ("s1", s1_10), ("s0", s0_15),
                ]
            else:
                raw_items = [
                    ("x0", xs[0]), ("x1", xs[1]), ("x2", xs[2]), ("x3", xs[3]),
                    ("s2", s2_35), ("s1", s1_10), ("s0", s0_15),
                ]
            ordered = list(reversed(raw_items)) if is_mir else list(raw_items)
            nodes_5l: dict[str, str] = {"10": ori_s(sup_5l_seq), "ta": ori_s(ta60), "tb": ori_s(tb60)}
            for nm, sq in ordered:
                nodes_5l[nm] = ori_s(sq)
            base_in = [f"{nm}+" for nm, _ in ordered]
            tail_in = ["tb+", "ta+"] if is_mir else ["ta+", "tb+"]
            full_in = (tail_in + base_in) if is_mir else (base_in + tail_in)
            paths_5l = sup15 + [
                ("Hap1a#1#Chr01", _wrap34(full_in)),
                ("Hap1b#1#Chr01", _wrap34(full_in)),
                ("HapClean#1#Chr01", _wrap34(base_in)),
            ] + [
                (f"HapS{k}#1#Chr01", _wrap34([st for st in base_in if st != f"{nm}+"]))
                for k, (nm, _) in enumerate(ordered)
            ]
            bun_5l = _run_i10_cli(f"i10_{variant_tag}_{'mir' if is_mir else 'fwd'}", nodes_5l, paths_5l)
            _assert_clean_hap(bun_5l, "HapClean#1")
            assert bun_5l.pan_seq.count(ori_s(s1_10)) == 1
            assert bun_5l.pan_seq.count(ori_s(s0_15)) == 1


def test_e2e_iter11_design2pp_diagonal_dp_and_target_piece_guard(tmp_path: Path) -> None:
    """Iter 11 Strictly Opaque-Box CLI E2E Regressions (`Design 2++`, ZERO `panflat` imports):
    1. `Finding 1a` (`s40` & `s41`), `Finding 1b` (`u42+nov8` `pA(46)+pB(46)` + 8th-order
       co-drop attack `L0+pC(30)+pA+pB+R0`), `Killshot A2` (`L0+pC(30)+u1(12)+u2(12)+R0`),
       `A^10|A^10` (`L0+pC(30)+A^10+A^10+R0`), and `2-Target-Piece Control` (`SupA` blocked,
       `SupB` drops `HapPure` cleanly).
    2. `tail_snps = 1..4` before `nov20(20)+novR(200)` (`gt[Hap2] == '.'` & `s40_mut` absent
       from `pan.fa`, guarding against `Design 1` ghost-allele regression).
    3. `H2x` (`FWD == REV == '.'`), `BUG-R10-1` (`FWD` & `REV`), `BUG-R10-2` (`D1_0.99`,
       `D3_0.95`, `D6_0.90`), `BUG-R10-3` (`FWD` & `REV`), and `BUG-R10-4` (`FWD` & `REV`).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut_pos(seq: str, positions: Sequence[int]) -> str:
        tr = {"A": "C", "C": "G", "G": "T", "T": "A"}
        arr = list(seq)
        for p in positions:
            arr[p] = tr[arr[p]]
        return "".join(arr)

    base_ref6 = {str(i): _mseq(400, 69000 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]

    def _run_i11_cli(
        tag: str,
        extra_nodes: dict[str, str],
        alt_paths: list[tuple[str, list[str]]],
        dedup_id: str = "0.90",
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, sq in base_ref6.items():
            b.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ref6_steps)
        for pname, steps in alt_paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out_pref = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", "pan", "--dedup-id", dedup_id, "--min-sv-len", "50"],
            out_prefix=out_pref,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
        run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
        evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
        assert evs, f"{tag}: missing expected INS event at 3..4"
        ev = evs[0]
        return ev, bun.genotypes[ev["event_id"]], bun.pan_seq

    l0 = _mseq(1500, 33101)
    r0 = _mseq(1500, 33102)
    nov_r = _mseq(200, 33103)
    nov_sup = _mseq(200, 33104)

    # (1a) Finding 1a: s40 & s41 (11 bp, 2 bp novel) MUST BOTH preserve Hap2
    mh9 = "ACGTTCGAC"
    m10 = _mseq(8, 33401) + "GG"
    p1_atk = m10 + mh9 + "T"
    p2_atk = "A" + mh9
    for tag_1a, s_tail in (
        ("i11_f1a_s40", "C" + _mseq(39, 33402)),
        ("i11_f1a_s41", "C" + _mseq(40, 33402)),
    ):
        _, gt_1a, pan_1a = _run_i11_cli(
            tag_1a,
            {"10": l0 + m10 + mh9 + s_tail + nov_sup + r0, "L0": l0, "p1": p1_atk, "p2": p2_atk, "sR": s_tail, "novR": nov_r, "R0": r0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "p1+", "p2+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "p1+", "p2+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "p1+", "p2+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "p2+", "sR+", "novR+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "p1+", "sR+", "novR+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_1a["Hap2#1"] != "." and p1_atk in pan_1a and p2_atk in pan_1a, (
            f"{tag_1a} FAIL: gt={gt_1a['Hap2#1']}"
        )

    # (1b) Finding 1b (u42+nov8 pA(46)+pB(46)) + 8th-order co-drop attack (L0+pC(30)+pA+pB+R0)
    u42 = _mseq(42, 65201)
    nov8 = "GATCCATG"
    p_a, p_b = u42 + nov8[:4], nov8[4:] + u42
    t_a, t_b = _mseq(60, 65111), _mseq(60, 65112)
    _, gt_1b, pan_1b = _run_i11_cli(
        "i11_f1b_split",
        {"10": l0 + u42 + r0, "L0": l0, "pA": p_a, "pB": p_b, "R0": r0, "tA": t_a, "tB": t_b},
        sup8 + [
            ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pB+", "R0+", "tB+", "5+", "6+"]),
            ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pA+", "R0+", "tA+", "5+", "6+"]),
            ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "5+", "6+"]),
        ],
    )
    assert gt_1b["HapPure#1"] != "." and p_a in pan_1b and p_b in pan_1b and nov8 in pan_1b

    p_c = _mseq(30, 65301)
    t_c = _mseq(60, 65113)
    _, gt_1b_co, pan_1b_co = _run_i11_cli(
        "i11_f1b_codrop_8th",
        {"10": l0 + p_c + u42 + r0, "L0": l0, "pC": p_c, "pA": p_a, "pB": p_b, "R0": r0, "tA": t_a, "tB": t_b, "tC": t_c},
        sup8 + [
            ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "pA+", "pB+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
            ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "pA+", "pB+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
            ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "pB+", "R0+", "tB+", "tC+", "5+", "6+"]),
            ("Hap4#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "pA+", "R0+", "tA+", "tC+", "5+", "6+"]),
            ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pC+", "pA+", "pB+", "R0+", "5+", "6+"]),
        ],
    )
    assert gt_1b_co["HapPure#1"] != "." and nov8 in pan_1b_co

    # (1c) Killshot A2 (L0+pC(30)+u1(12)+u2(12)+R0) & A^10|A^10 (L0+pC(30)+A^10+A^10+R0)
    u0_ka2 = "GACGTCTAGCTT"
    u1_ka2 = _mut_pos(u0_ka2, [1, 3, 5])
    u2_ka2 = _mut_pos(u0_ka2, [6, 8, 10])
    for tag_td, s_u0, s_u1, s_u2 in (
        ("i11_killshot_a2", u0_ka2, u1_ka2, u2_ka2),
        ("i11_a10_a10_tandem", "A" * 10, "A" * 10, "A" * 10),
    ):
        _, gt_td, pan_td = _run_i11_cli(
            tag_td,
            {"10": l0 + p_c + s_u0 + r0, "L0": l0, "pC": p_c, "u1": s_u1, "u2": s_u2, "R0": r0, "tA": t_a, "tB": t_b, "tC": t_c},
            sup8 + [
                ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "u1+", "u2+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
                ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "u1+", "u2+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
                ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "u1+", "u2+", "R0+", "tA+", "tB+", "5+", "6+"]),
                ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "u2+", "R0+", "tB+", "tC+", "5+", "6+"]),
                ("Hap4#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "u1+", "R0+", "tA+", "tC+", "5+", "6+"]),
                ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pC+", "u1+", "u2+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_td["HapPure#1"] != "." and (s_u1 + s_u2) in pan_td, f"{tag_td} FAIL: gt={gt_td['HapPure#1']}"

    # (1d) 2-Target-Piece Control: SupA (1..2) is target-piece-blocked, SupB (2..3) drops HapPure cleanly
    sup_a10 = [(f"SupA{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 11)]
    sup_b8 = [(f"SupB{k}#1#Chr01", ["1+", "2+", "20+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    nov_b = _mseq(200, 65320)
    _, gt_2tp, _ = _run_i11_cli(
        "i11_2target_piece_ctrl",
        {
            "10": l0 + u42 + r0,
            "20": l0 + p_a + p_b + nov_b,
            "L0": l0, "pA": p_a, "pB": p_b, "R0": r0, "tA": t_a, "tB": t_b,
        },
        sup_a10 + sup_b8 + [
            ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pB+", "R0+", "tB+", "5+", "6+"]),
            ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pA+", "R0+", "tA+", "5+", "6+"]),
            ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "5+", "6+"]),
        ],
    )
    assert gt_2tp["HapPure#1"] == ".", f"2-Target-Piece Control FAIL: gt={gt_2tp['HapPure#1']}"

    # (2) tail_snps = 1..4 before nov20(20)+novR(200): MUST drop s40_mut cleanly (no Design 1 ghost allele!)
    s40_base = _mseq(40, 65400)
    nov20 = _mseq(20, 65401)
    for k_snps in (1, 2, 3, 4):
        s40_mut = _mut_pos(s40_base, list(range(40 - k_snps, 40)))
        _, gt_ts, pan_ts = _run_i11_cli(
            f"i11_tail_snps_{k_snps}",
            {"10": l0 + s40_base + nov_sup + r0, "L0": l0, "s40m": s40_mut, "nov20": nov20, "novR": nov_r, "R0": r0},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s40m+", "nov20+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s40m+", "nov20+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s40m+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "nov20+", "novR+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s40m+", "novR+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_ts["Hap2#1"] == "." and s40_mut not in pan_ts, (
            f"tail_snps={k_snps} ghost allele leak: gt={gt_ts['Hap2#1']} s40m_in={s40_mut in pan_ts}"
        )

    # (3a) Finding 2b: H2x FWD == REV exact parity (under Iter 12 s_cover dual,
    # u1(12) + pt(10) + tail2(2) is a 12 bp quasi-tandem, so pt(10) is conserved
    # symmetrically on both + and - strands with identical gt and alleles).
    u0_h2 = "GACGTCTAGCTT"
    u1_h2 = "GCCGTCTAGCTT"
    u2_h2 = "GACGTCTAGCTG"
    pt_h2, tail2_h2 = u2_h2[:10], u2_h2[10:]
    fl_h2, fr_h2 = _mseq(200, 65501), _mseq(200, 65502)
    h2_results: list[tuple[str, str]] = []
    for tag_h2, sup_h2 in (
        ("i11_h2x_fwd", fl_h2 + u0_h2 + fr_h2),
        ("i11_h2x_rev", revcomp(fl_h2 + u0_h2 + fr_h2)),
    ):
        ev_h2, gt_h2, _ = _run_i11_cli(
            tag_h2,
            {"10": sup_h2, "L0": fl_h2, "u1": u1_h2, "pt": pt_h2, "t2": tail2_h2, "R0": fr_h2, "tA": t_a, "tB": t_b},
            sup8 + [
                ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "u1+", "pt+", "t2+", "R0+", "tA+", "tB+", "5+", "6+"]),
                ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "u1+", "pt+", "t2+", "R0+", "tA+", "tB+", "5+", "6+"]),
                ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pt+", "t2+", "R0+", "tB+", "5+", "6+"]),
                ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "u1+", "t2+", "R0+", "tA+", "5+", "6+"]),
                ("Hap4#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "u1+", "pt+", "R0+", "tA+", "5+", "6+"]),
                ("HapTest#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pt+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_h2["HapTest#1"] != ".", f"{tag_h2} FAIL: expected conserved pt10, got gt={gt_h2['HapTest#1']}"
        h2_results.append((gt_h2["HapTest#1"], ev_h2["alleles"]))
    assert h2_results[0] == h2_results[1], f"H2x FWD/REV mismatch: {h2_results}"

    # (3b) BUG-R10-1 (DEFAULT pan/0.90/50, FWD & REV): D(7)+X(1) at q_p==p_s
    l0_c = _mseq(1500, 93101)[:-1] + "C"
    r0_c = "G" + _mseq(1500, 93102)[1:]
    nov_rc, nov_sc = _mseq(200, 93103), _mseq(200, 93104)
    l0_1 = _mseq(1500, 93201)[:-1] + "T"
    s25_1 = "T" + _mseq(24, 93202)
    sup_1_fwd = l0_1 + "GTGTTAA" + "G" + s25_1 + nov_sc + r0_c
    for tag_r1, sup_r1 in (("i11_r10_1_fwd", sup_1_fwd), ("i11_r10_1_rev", revcomp(sup_1_fwd))):
        _, gt_r1, _ = _run_i11_cli(
            tag_r1,
            {"10": sup_r1, "L0": l0_1, "sliv": "C", "s25": s25_1, "novR": nov_rc, "R0": r0_c},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "sliv+", "s25+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "sliv+", "s25+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s25+", "novR+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "sliv+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "sliv+", "novR+", "R0+", "5+", "6+"]),
            ],
            dedup_id="0.90",
        )
        assert gt_r1["Hap3#1"] == ".", f"{tag_r1} FAIL: gt={gt_r1['Hap3#1']}"

    # (3c) BUG-R10-2 (D1_0.99, D3_0.95, D6_0.90): non-tandem D(k) need==d_len
    for tag_r2, d_s, s20_s, s40_s, did in (
        ("D1_0.99", "A", "ACGTACGTAGTCAGTCAGTC", "G" + _mseq(39, 92101), "0.99"),
        ("D3_0.95", "ACG", "ACGTTGGCATGCTAGTCAGT", "G" + _mseq(39, 92102), "0.95"),
        ("D6_0.90", "ACGTAG", "ACGTAGTTGGCATGCTAGTC", "G" + _mseq(39, 92103), "0.90"),
    ):
        sup_m = revcomp(l0_c + nov_sc + s40_s + d_s + s20_s + r0_c)
        l0m, r0m = revcomp(r0_c), revcomp(l0_c)
        s20m, s40m, nov_rm = revcomp(s20_s), revcomp(s40_s), revcomp(nov_rc)
        _, gt_r2, pan_r2 = _run_i11_cli(
            f"i11_r10_2_{tag_r2}",
            {"10": sup_m, "L0": l0m, "s20": s20m, "s40": s40m, "novR": nov_rm, "R0": r0m},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s20+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s20+", "novR+", "R0+", "5+", "6+"]),
            ],
            dedup_id=did,
        )
        assert gt_r2["Hap3#1"] == "." and pan_r2.count(s20m) == 1, (
            f"BUG-R10-2[{tag_r2}] FAIL: gt={gt_r2['Hap3#1']} cnt={pan_r2.count(s20m)}"
        )

    # (3d) BUG-R10-3 (DEFAULT pan/0.90/50, FWD & REV): D(4)+M(5)+X(1)
    nov_r1, nov_r2 = _mseq(200, 95103), _mseq(200, 95104)
    nov_s1, nov_s2 = _mseq(200, 95105), _mseq(200, 95106)
    s15_3, s33_3, s40_3 = _mseq(15, 95200), _mseq(28, 95201) + "ACGAT", "C" + _mseq(39, 95202)
    sup_3_fwd = l0_c + nov_s1 + s15_3 + s33_3[:-5] + "ACGA" + "ACGAT" + "G" + s40_3[1:] + nov_s2 + r0_c
    for tag_r3, sup_r3 in (("i11_r10_3_fwd", sup_3_fwd), ("i11_r10_3_rev", revcomp(sup_3_fwd))):
        _, gt_r3, _ = _run_i11_cli(
            tag_r3,
            {"10": sup_r3, "L0": l0_c, "novR1": nov_r1, "s15": s15_3, "s33": s33_3, "s40": s40_3, "novR2": nov_r2, "R0": r0_c},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR1+", "s15+", "s33+", "s40+", "novR2+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "novR1+", "s15+", "s33+", "s40+", "novR2+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR1+", "s15+", "s40+", "novR2+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s33+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR1+", "s33+", "s40+", "novR2+", "R0+", "5+", "6+"]),
                ("Hap5#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "novR1+", "s15+", "s33+", "novR2+", "R0+", "5+", "6+"]),
            ],
            dedup_id="0.90",
        )
        assert gt_r3["Hap3#1"] == ".", f"{tag_r3} FAIL: gt={gt_r3['Hap3#1']}"

    # (3e) BUG-R10-4 (pan/0.99/50, FWD & REV): flush I(1)+M(1)+X(1) at p_e
    s34_4, s40_4 = _mseq(33, 92301) + "A", "AG" + _mseq(38, 92302)
    sup_4_fwd = l0_c + s34_4 + "T" + s40_4[2:] + nov_sc + r0_c
    for tag_r4, sup_r4, exp_cnt in (("i11_r10_4_fwd", sup_4_fwd, 1), ("i11_r10_4_rev", revcomp(sup_4_fwd), 0)):
        _, gt_r4, pan_r4 = _run_i11_cli(
            tag_r4,
            {"10": sup_r4, "L0": l0_c, "s34": s34_4, "s40": s40_4, "novR": nov_rc, "R0": r0_c},
            sup8 + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s34+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "s34+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap2#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s40+", "novR+", "R0+", "5+", "6+"]),
                ("Hap3#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s34+", "R0+", "5+", "6+"]),
                ("Hap4#1#Chr01",  ["1+", "2+", "3+", "4+", "L0+", "s34+", "novR+", "R0+", "5+", "6+"]),
            ],
            dedup_id="0.99",
        )
        assert gt_r4["Hap3#1"] == "." and pan_r4.count(s34_4) == exp_cnt, (
            f"{tag_r4} FAIL: gt={gt_r4['Hap3#1']} cnt={pan_r4.count(s34_4)}"
        )


def test_e2e_iter12_bug_a_bug_b_and_str_outer_i_closures(tmp_path: Path) -> None:
    """Iter 12 Strictly Opaque-Box CLI E2E Regressions (ZERO `panflat` imports):
    1. BUG A distant co-drop (`A^10|A^10`, `u49|u49`, `QT-12`, 1-sided `pC(30)` AND
       2-sided `pC(30)+pD(30)`): all 6 builds conserve exact full tandem/quasi-tandem
       allele lengths (`20`, `98`, `24`) and `gt[HapPure#1] != "."`.
    2. BUG B `< L`-split tandem SVs with ZERO `pC`/`pD` (`L60_30x4`, `L80_40x4`,
       `L20_10x4`, `u48+"GT"+u48` `49+49`): all 4 builds conserve `gt[HapPure#1] != "."`
       and preserve tandem copy sequence in `pan.fa`.
    3. `r11_1` Finding 1 (all 5 STR/poly-A CLI topologies x FWD/REV = 10 builds under
       DEFAULT `pan/0.90/50`): right/left `(AC)^100`, right `A^200`, `A^130 + 1 SNP`,
       strictly interior `(AG)^113 + 1 SNP` -> `FWD == REV` identical `gt` & `alleles`
       and `verify == PASS` on all 10 builds.
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut_pos(seq: str, positions: Sequence[int]) -> str:
        tr = {"A": "C", "C": "G", "G": "T", "T": "A"}
        arr = list(seq)
        for p in positions:
            arr[p] = tr[arr[p]]
        return "".join(arr)

    base_ref6 = {str(i): _mseq(400, 81000 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]

    def _run_i12_cli(
        tag: str,
        extra_nodes: dict[str, str],
        alt_paths: list[tuple[str, list[str]]],
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, sq in base_ref6.items():
            b.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ref6_steps)
        for pname, steps in alt_paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out_pref = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "50"],
            out_prefix=out_pref,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
        run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
        evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
        assert evs, f"{tag}: missing expected INS event at 3..4"
        ev = evs[0]
        return ev, bun.genotypes[ev["event_id"]], bun.pan_seq

    l0b, r0b = _mseq(1500, 33101), _mseq(1500, 33102)
    p_c, p_d = _mseq(30, 77101), _mseq(30, 77102)
    t_a, t_b, t_c, t_d = _mseq(60, 77111), _mseq(60, 77112), _mseq(60, 77113), _mseq(60, 77114)

    # (1) BUG A: distant co-drop 1-sided (pC) & 2-sided (pC + pD) on A^10|A^10, u49|u49, QT-12
    u0_qt = "GACGTCTAGCTT"
    u1_qt = _mut_pos(u0_qt, [2])
    u2_qt = _mut_pos(u0_qt, [9])
    for name_a, u0_a, u1_a, u2_a, ml_len in (
        ("a10_a10", "A" * 10, "A" * 10, "A" * 10, 190),
        ("u49_u49", _mseq(49, 77601), _mseq(49, 77601), _mseq(49, 77601), 151),
        ("qt12", u0_qt, u1_qt, u2_qt, 188),
    ):
        m_l, m_r = _mseq(ml_len, 77602), _mseq(ml_len, 77603)
        exp_len = len(u1_a) + len(u2_a)
        ev_2s, gt_2s, pan_2s = _run_i12_cli(
            f"i12_bug_a_2s_{name_a}",
            {
                "10": l0b + p_c + m_l + u0_a + m_r + p_d + r0b,
                "L0": l0b, "pC": p_c, "mL": m_l, "u1": u1_a, "u2": u2_a,
                "mR": m_r, "pD": p_d, "R0": r0b, "tA": t_a, "tB": t_b, "tC": t_c, "tD": t_d,
            },
            sup8 + [
                ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "pD+", "R0+", "tA+", "tB+", "tC+", "tD+", "5+", "6+"]),
                ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "pD+", "R0+", "tA+", "tB+", "tC+", "tD+", "5+", "6+"]),
                ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "mL+", "u1+", "u2+", "mR+", "pD+", "R0+", "tA+", "tB+", "tD+", "5+", "6+"]),
                ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u2+", "mR+", "pD+", "R0+", "tB+", "tC+", "tD+", "5+", "6+"]),
                ("Hap4#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "mR+", "pD+", "R0+", "tA+", "tC+", "tD+", "5+", "6+"]),
                ("Hap5#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
                ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "pD+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_2s["HapPure#1"] != "." and (u1_a + u2_a) in pan_2s and f"len={exp_len},src=HapPure#1" in ev_2s["alleles"], (
            f"BUG A 2s {name_a} FAIL: gt={gt_2s['HapPure#1']} al={ev_2s['alleles']}"
        )

        ev_1s, gt_1s, pan_1s = _run_i12_cli(
            f"i12_bug_a_1s_{name_a}",
            {
                "10": l0b + p_c + m_l + u0_a + m_r + r0b,
                "L0": l0b, "pC": p_c, "mL": m_l, "u1": u1_a, "u2": u2_a,
                "mR": m_r, "R0": r0b, "tA": t_a, "tB": t_b, "tC": t_c,
            },
            sup8 + [
                ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
                ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "R0+", "tA+", "tB+", "tC+", "5+", "6+"]),
                ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "mL+", "u1+", "u2+", "mR+", "R0+", "tA+", "tB+", "5+", "6+"]),
                ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u2+", "mR+", "R0+", "tB+", "tC+", "5+", "6+"]),
                ("Hap4#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "mR+", "R0+", "tA+", "tC+", "5+", "6+"]),
                ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pC+", "mL+", "u1+", "u2+", "mR+", "R0+", "5+", "6+"]),
            ],
        )
        assert gt_1s["HapPure#1"] != "." and (u1_a + u2_a) in pan_1s and f"len={exp_len},src=HapPure#1" in ev_1s["alleles"], (
            f"BUG A 1s {name_a} FAIL: gt={gt_1s['HapPure#1']} al={ev_1s['alleles']}"
        )

    # (2) BUG B: < L-split tandem SVs (ZERO pC/pD): L60_30x4, L80_40x4, L20_10x4, u48+"GT"+u48
    for tag_b, u_len, chunk_len in (("l60_30x4", 60, 30), ("l80_40x4", 80, 40), ("l20_10x4", 20, 10)):
        u_b = _mseq(u_len, 78000 + u_len)
        uu_b = u_b + u_b
        qs_b = [(f"q{i}", uu_b[i * chunk_len : (i + 1) * chunk_len]) for i in range(4)]
        qst_b = [f"{k}+" for k, _ in qs_b]
        tst_b = [f"t{i}+" for i in range(4)]
        nodes_b = {
            "10": l0b + u_b + r0b, "L0": l0b, "R0": r0b,
            **dict(qs_b), **{f"t{i}": _mseq(60, 79100 + i) for i in range(4)},
        }
        paths_b = (
            sup8
            + [
                ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + qst_b + ["R0+"] + tst_b + ["5+", "6+"]),
                ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + qst_b + ["R0+"] + tst_b + ["5+", "6+"]),
            ]
            + [
                (
                    f"HapDrop{i}#1#Chr01",
                    ["1+", "2+", "3+", "4+", "L0+"]
                    + [s for j, s in enumerate(qst_b) if j != i]
                    + ["R0+"]
                    + [s for j, s in enumerate(tst_b) if j != i]
                    + ["5+", "6+"],
                )
                for i in range(4)
            ]
            + [("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + qst_b + ["R0+", "5+", "6+"])]
        )
        _, gt_b, pan_b = _run_i12_cli(f"i12_bug_b_{tag_b}", nodes_b, paths_b)
        assert gt_b["HapPure#1"] != "." and u_b in pan_b, f"BUG B {tag_b} FAIL: gt={gt_b['HapPure#1']}"
        if u_len in (60, 20):
            assert uu_b in pan_b, f"BUG B {tag_b} missing full uu in pan"

    u48 = _mseq(48, 78300)
    p_a49, p_b49 = u48 + "G", "T" + u48
    # Exact 50 bp tandem boundary (l0[-1]=='T', r0[0]=='G'): min_ov==0 on both pA & pB, s_cover conserves full pA+pB
    l0_tg, r0_tg = l0b[:-1] + "T", "G" + r0b[1:]
    _, gt_b49_tg, pan_b49_tg = _run_i12_cli(
        "i12_bug_b_u48_gt_u48_exact_scover",
        {"10": l0_tg + u48 + r0_tg, "L0": l0_tg, "pA": p_a49, "pB": p_b49, "R0": r0_tg, "tA": t_a, "tB": t_b},
        sup8 + [
            ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pB+", "R0+", "tB+", "5+", "6+"]),
            ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pA+", "R0+", "tA+", "5+", "6+"]),
            ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "5+", "6+"]),
        ],
    )
    assert gt_b49_tg["HapPure#1"] != "." and (p_a49 + p_b49) in pan_b49_tg
    # Asymmetric boundary (l0[-1]=='T', r0[0]=='A'): pA(u48+'G') conserved (gt != '.'), never wiped out
    _, gt_b49, pan_b49 = _run_i12_cli(
        "i12_bug_b_u48_gt_u48_asym",
        {"10": l0b + u48 + r0b, "L0": l0b, "pA": p_a49, "pB": p_b49, "R0": r0b, "tA": t_a, "tB": t_b},
        sup8 + [
            ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "tA+", "tB+", "5+", "6+"]),
            ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pB+", "R0+", "tB+", "5+", "6+"]),
            ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pA+", "R0+", "tA+", "5+", "6+"]),
            ("HapPure#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pA+", "pB+", "R0+", "5+", "6+"]),
        ],
    )
    assert gt_b49["HapPure#1"] != "." and p_a49 in pan_b49

    # (3) r11_1 Finding 1: all 5 STR/poly-A CLI topologies (FWD == REV, 10 builds)
    l0_s = _mseq(1500, 81101)[:-1] + "T"
    r0_s = "G" + _mseq(1500, 81102)[1:]
    for tag_s, sup_fwd, node_l, node_p, node_r in (
        ("str_r",   l0_s + "AC" * 95 + r0_s,         l0_s + "AC" * 93,  "AC" * 7 + r0_s[:5], r0_s[5:]),
        ("str_l",   l0_s + "AC" * 95 + r0_s,         l0_s[:-5],         l0_s[-5:] + "AC" * 7, "AC" * 93 + r0_s),
        ("polya_r", l0_s + "A" * 190 + r0_s,         l0_s + "A" * 186,  "A" * 14 + r0_s[:5], r0_s[5:]),
        ("polyax",  l0_s + "A" * 120 + "G" + r0_s,   l0_s + "A" * 116,  "A" * 14 + "C",      r0_s),
        ("str_int", l0_s + "AG" * 108 + "C" + r0_s,  l0_s + "AG" * 37,  "AG" * 10,           "AG" * 66 + "A" + r0_s),
    ):
        str_res: list[tuple[str, str]] = []
        for sfx, sup_seq in (("fwd", sup_fwd), ("rev", revcomp(sup_fwd))):
            ev_s, gt_s, _ = _run_i12_cli(
                f"i12_{tag_s}_{sfx}",
                {"10": sup_seq, "L0": node_l, "pE": node_p, "R0": node_r, "tA": t_a, "tB": t_b},
                sup8 + [
                    ("Hap1a#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pE+", "R0+", "tA+", "tB+", "5+", "6+"]),
                    ("Hap1b#1#Chr01",   ["1+", "2+", "3+", "4+", "L0+", "pE+", "R0+", "tA+", "tB+", "5+", "6+"]),
                    ("Hap2#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "R0+", "tB+", "5+", "6+"]),
                    ("Hap3#1#Chr01",    ["1+", "2+", "3+", "4+", "L0+", "pE+", "tA+", "5+", "6+"]),
                    ("HapTest#1#Chr01", ["1+", "2+", "3+", "4+", "L0+", "pE+", "R0+", "5+", "6+"]),
                ],
            )
            str_res.append((gt_s["HapTest#1"], ev_s["alleles"]))
        assert str_res[0] == str_res[1] and str_res[0][0] != ".", (
            f"STR CLI {tag_s} FWD/REV mismatch: fwd={str_res[0]} rev={str_res[1]}"
        )


def test_e2e_iter13_cap_free_outer_i_and_unbounded_tandem_closures(tmp_path: Path) -> None:
    """Iter 13 Strictly Opaque-Box CLI E2E Regressions (ZERO `panflat` imports):
    Exercises all 7 live CLI topologies x FWD/REV (= 14 `run_build` + 14 `run_verify`
    cycles under DEFAULT `--dedup-mode pan --dedup-id 0.90 --min-sv-len 50`):
    1. `pure_a210`: `A^210` vs `A^105` (`6x35 bp`) — `FWD == REV` & all 6 pieces conserved.
    2. `str140_rel5`: `(AC)^70` vs `(AC)^65` + 1 SNP at offset 35 — `FWD == REV`.
    3. `polya140_rel5`: `A^140` vs `A^130` + 1 SNP at offset 35 — `FWD == REV`.
    4. `str140_left_rel5`: `(AC)^70` vs `(AC)^65` + 1 SNP at offset 95 (left mirror) — `FWD == REV`.
    5. `QT_100`: `U_1(100)[95] + U_2(100)[5]` quasi-tandem SV — `FWD == REV` & NOT vaporized.
    6. `u120_d15_11th`: `D(15) + I(120)` vs `I(120) + D(15)` (11th-order Break B1) — `FWD == REV`.
    7. `u100_3copy_snp_11th`: `U(100)*3 + 1 SNP` in copy 3 (11th-order Break B2) — `FWD == REV`.
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut_pos(seq: str, positions: Sequence[int]) -> str:
        tr = {"A": "C", "C": "G", "G": "T", "T": "A"}
        arr = list(seq)
        for p in positions:
            arr[p] = tr[arr[p]]
        return "".join(arr)

    base_ref6 = {str(i): _mseq(400, 88000 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup20 = [(f"Sup{k:02d}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 21)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]

    def _run_i13_cli(
        tag: str,
        extra_nodes: dict[str, str],
        alt_paths: list[tuple[str, list[str]]],
    ):
        b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, sq in base_ref6.items():
            b.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            b.add_node(nid, sq)
        b.set_ref_path("Ref#1#Chr01", ref6_steps)
        for pname, steps in alt_paths:
            b.add_path(pname, steps)
        inp = b.write(tag)
        out_pref = tmp_path / f"{tag}_out"
        proc, bun = run_build(
            inp,
            extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "50"],
            out_prefix=out_pref,
        )
        assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
        run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
        evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
        assert evs, f"{tag}: missing expected INS event at 3..4"
        ev = evs[0]
        return ev, bun.genotypes[ev["event_id"]], bun.inserts, bun.pan_seq

    def _run_fwd_rev_pieces(
        name: str,
        sup_fwd: str,
        l0_seq: str,
        r0_seq: str,
        p_list: list[tuple[str, str]],
        sup_paths: list[tuple[str, list[str]]],
        probe_hap: str = "HapTest#1",
        seed_base: int = 89000,
    ):
        t_nodes = {f"t{i}": _mseq(60, seed_base + i) for i in range(len(p_list))}
        p_steps = [f"{k}+" for k, _ in p_list]
        t_steps = [f"t{i}+" for i in range(len(p_list))]
        out_pair = []
        for sfx, sup_seq in (("fwd", sup_fwd), ("rev", revcomp(sup_fwd))):
            nodes = {"10": sup_seq, "L0": l0_seq, "R0": r0_seq, **dict(p_list), **t_nodes}
            paths = (
                sup_paths
                + [
                    ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                    ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                ]
                + [
                    (
                        f"HapDrop{i}#1#Chr01",
                        ["1+", "2+", "3+", "4+", "L0+"]
                        + [s for j, s in enumerate(p_steps) if j != i]
                        + ["R0+"]
                        + [s for j, s in enumerate(t_steps) if j != i]
                        + ["5+", "6+"],
                    )
                    for i in range(len(p_list))
                ]
                + [(f"{probe_hap}#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+", "5+", "6+"])]
            )
            out_pair.append(_run_i13_cli(f"i13_{name}_{sfx}", nodes, paths))
        return out_pair[0], out_pair[1]

    l0_a = _mseq(1500, 88101)[:-1] + "T"
    r0_a = "G" + _mseq(1500, 88102)[1:]

    # (1) pure_a210 (A^210 vs A^105, 6x35 bp): FWD == REV & all 6 35-bp pieces conserved
    qs_a = [(f"q{i}", "A" * 35) for i in range(6)]
    (ev_f1, gt_f1, ins_f1, _), (ev_r1, gt_r1, ins_r1, _) = _run_fwd_rev_pieces(
        "pure_a210", l0_a + ("A" * 105) + r0_a, l0_a, r0_a, qs_a, sup20, probe_hap="HapPure#1", seed_base=89000
    )
    ins_35_f = [
        (r["src_intervals"], r["allele"])
        for r in ins_f1
        if r["event_id"] == ev_f1["event_id"] and r["pan_end"] - r["pan_start"] == 35
    ]
    ins_35_r = [
        (r["src_intervals"], r["allele"])
        for r in ins_r1
        if r["event_id"] == ev_r1["event_id"] and r["pan_end"] - r["pan_start"] == 35
    ]
    assert ev_f1["alleles"] == ev_r1["alleles"], f"pure_a210 alleles mismatch: {ev_f1['alleles']} vs {ev_r1['alleles']}"
    assert ins_35_f == ins_35_r and len(ins_35_f) == 6, f"pure_a210 35bp inserts FAIL: {ins_35_f} vs {ins_35_r}"
    assert gt_f1["HapPure#1"] == gt_r1["HapPure#1"] and gt_f1["HapPure#1"] != "."

    # (2-4) str140_rel5, polya140_rel5, str140_left_rel5 (r12_2 3 isolated CLI topologies)
    l0_s = _mseq(1500, 81101)[:-1] + "T"
    r0_s = "G" + _mseq(1500, 81102)[1:]
    rt1 = list("AC" * 65)
    rt1[35] = "T"
    rt2 = list("A" * 130)
    rt2[35] = "C"
    rt3 = list("AC" * 65)
    rt3[95] = "T"
    for name_s, sup_s, p_list_s in (
        ("str140_rel5", l0_s + "".join(rt1) + r0_s, [("p1", "AC" * 20), ("p2", "AC" * 20), ("p3", "AC" * 20), ("pE", "AC" * 10)]),
        ("polya140_rel5", l0_s + "".join(rt2) + r0_s, [("p1", "A" * 40), ("p2", "A" * 40), ("p3", "A" * 40), ("pE", "A" * 20)]),
        ("str140_left_rel5", l0_s + "".join(rt3) + r0_s, [("pE", "AC" * 10), ("p1", "AC" * 20), ("p2", "AC" * 20), ("p3", "AC" * 20)]),
    ):
        (ev_fs, gt_fs, _, _), (ev_rs, gt_rs, _, _) = _run_fwd_rev_pieces(
            name_s, sup_s, l0_s, r0_s, p_list_s, sup8, probe_hap="HapTest#1", seed_base=89100
        )
        assert (gt_fs["HapTest#1"], ev_fs["alleles"]) == (gt_rs["HapTest#1"], ev_rs["alleles"]), (
            f"{name_s} FWD/REV mismatch: fwd={(gt_fs['HapTest#1'], ev_fs['alleles'])} rev={(gt_rs['HapTest#1'], ev_rs['alleles'])}"
        )
        assert gt_fs["HapTest#1"] != "."

    # (5) QT_100: U_1(100)[95] + U_2(100)[5] quasi-tandem SV (FWD == REV, >= 100 bp conserved)
    u0_qt = _mseq(100, 89201)
    u1_qt = _mut_pos(u0_qt, [95])
    u2_qt = _mut_pos(u0_qt, [5])
    uu_qt = u1_qt + u2_qt
    p_list_qt = [(f"q{i}", uu_qt[i * 30 : min((i + 1) * 30, len(uu_qt))]) for i in range(7)]
    (ev_fq, gt_fq, _, pan_fq), (ev_rq, gt_rq, _, pan_rq) = _run_fwd_rev_pieces(
        "QT_100", l0_a + u0_qt + r0_a, l0_a, r0_a, p_list_qt, sup8, probe_hap="HapPure#1", seed_base=89210
    )
    assert (gt_fq["HapPure#1"], ev_fq["alleles"]) == (gt_rq["HapPure#1"], ev_rq["alleles"]), (
        f"QT_100 FWD/REV mismatch: fwd={ev_fq['alleles']} rev={ev_rq['alleles']}"
    )
    assert gt_fq["HapPure#1"] != "." and u1_qt in pan_fq and u2_qt in pan_fq and u1_qt in pan_rq

    # (6) u120_d15_11th: D(15) + I(120) vs I(120) + D(15) (11th-order Break B1)
    u120 = _mseq(120, 89301)
    d15 = _mseq(15, 89302)
    uu120 = u120 + u120
    p_list_d15 = [(f"q{i}", uu120[i * 40 : (i + 1) * 40]) for i in range(6)]
    (ev_fd, gt_fd, _, pan_fd), (ev_rd, gt_rd, _, pan_rd) = _run_fwd_rev_pieces(
        "u120_d15_11th", l0_a + d15 + u120 + r0_a, l0_a, r0_a, p_list_d15, sup8, probe_hap="HapPure#1", seed_base=89310
    )
    assert (gt_fd["HapPure#1"], ev_fd["alleles"]) == (gt_rd["HapPure#1"], ev_rd["alleles"]), (
        f"u120_d15_11th FWD/REV mismatch: fwd={ev_fd['alleles']} rev={ev_rd['alleles']}"
    )
    assert gt_fd["HapPure#1"] != "." and u120 in pan_fd and u120 in pan_rd

    # (7) u100_3copy_snp_11th: U(100)*3 + 1 SNP in copy 3 (11th-order Break B2)
    u100 = _mseq(100, 89401)
    u100_mut = _mut_pos(u100, [50])
    q_3c = u100 + u100 + u100_mut
    p_list_3c = [(f"q{i}", q_3c[i * 30 : min((i + 1) * 30, len(q_3c))]) for i in range(10)]
    (ev_f3, gt_f3, _, pan_f3), (ev_r3, gt_r3, _, pan_r3) = _run_fwd_rev_pieces(
        "u100_3copy_snp_11th", l0_a + u100 + u100 + r0_a, l0_a, r0_a, p_list_3c, sup8, probe_hap="HapPure#1", seed_base=89410
    )
    assert (gt_f3["HapPure#1"], ev_f3["alleles"]) == (gt_r3["HapPure#1"], ev_r3["alleles"]), (
        f"u100_3copy_snp_11th FWD/REV mismatch: fwd={ev_f3['alleles']} rev={ev_r3['alleles']}"
    )
    assert gt_f3["HapPure#1"] != "." and u100_mut in pan_f3 and u100_mut in pan_r3


def test_e2e_iter14_flank_snp_str_closures(tmp_path: Path) -> None:
    """Iter 14 Strictly Opaque-Box CLI E2E Regressions (ZERO `panflat` imports):
    Exercises all 3 isolated live 1-flank-SNP STR topologies x FWD/REV (= 6 `run_build`
    + 6 `run_verify(..., check=True)` cycles under DEFAULT `--dedup-mode pan
    --dedup-id 0.90 --min-sv-len 50`, ZERO SNPs inside the STR and 1 SNP at ±100 in
    `1,500 bp` flank `R0`/`L0`):
    1. `str140_flank1snp_r`: `(AC)^70` vs `(AC)^65` + 1 SNP at `R0[100]`.
    2. `polya140_flank1snp_r`: `A^140` vs `A^130` + 1 SNP at `R0[100]`.
    3. `str140_flank1snp_l`: `(AC)^70` vs `(AC)^65` + 1 SNP at `L0[1400]`.
    Asserts exact `FWD == REV` parity on `gt`, `alleles` (`1:len=380`, `6:len=140`),
    and `inserts` (ZERO silent 60-bp allele drops).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut1(seq: str, pos: int) -> str:
        tr = {"A": "C", "C": "G", "G": "T", "T": "A"}
        arr = list(seq)
        arr[pos] = tr[arr[pos]]
        return "".join(arr)

    base_ref6 = {str(i): _mseq(400, 69000 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]
    l0 = _mseq(1500, 81101)[:-1] + "T"
    r0 = "G" + _mseq(1500, 81102)[1:]
    r0_1 = _mut1(r0, 100)
    l0_1 = _mut1(l0, 1400)

    topologies = (
        (
            "str140_flank1snp_r",
            l0 + ("AC" * 65) + r0_1,
            [("p1", "AC" * 20), ("p2", "AC" * 20), ("p3", "AC" * 20), ("pE", "AC" * 10)],
        ),
        (
            "polya140_flank1snp_r",
            l0 + ("A" * 130) + r0_1,
            [("p1", "A" * 40), ("p2", "A" * 40), ("p3", "A" * 40), ("pE", "A" * 20)],
        ),
        (
            "str140_flank1snp_l",
            l0_1 + ("AC" * 65) + r0,
            [("pE", "AC" * 10), ("p1", "AC" * 20), ("p2", "AC" * 20), ("p3", "AC" * 20)],
        ),
    )

    for name, sup_fwd, p_list in topologies:
        t_nodes = {f"t{i}": _mseq(60, 82100 + i) for i in range(len(p_list))}
        p_steps = [f"{k}+" for k, _ in p_list]
        t_steps = [f"t{i}+" for i in range(len(p_list))]
        pair_res = []
        for sfx, sup_seq in (("fwd", sup_fwd), ("rev", revcomp(sup_fwd))):
            tag = f"i14_{name}_{sfx}"
            b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
            for nid, sq in base_ref6.items():
                b.add_node(nid, sq)
            extra_nodes = {"10": sup_seq, "L0": l0, "R0": r0, **dict(p_list), **t_nodes}
            for nid, sq in extra_nodes.items():
                b.add_node(nid, sq)
            b.set_ref_path("Ref#1#Chr01", ref6_steps)
            alt_paths = (
                sup8
                + [
                    ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                    ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                ]
                + [
                    (
                        f"HapDrop{i}#1#Chr01",
                        ["1+", "2+", "3+", "4+", "L0+"]
                        + [s for j, s in enumerate(p_steps) if j != i]
                        + ["R0+"]
                        + [s for j, s in enumerate(t_steps) if j != i]
                        + ["5+", "6+"],
                    )
                    for i in range(len(p_list))
                ]
                + [("HapTest#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+", "5+", "6+"])]
            )
            for pname, steps in alt_paths:
                b.add_path(pname, steps)
            inp = b.write(tag)
            out_pref = tmp_path / f"{tag}_out"
            proc, bun = run_build(
                inp,
                extra_args=["--dedup-mode", "pan", "--dedup-id", "0.90", "--min-sv-len", "50"],
                out_prefix=out_pref,
            )
            assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
            run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
            evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
            assert evs, f"{tag}: missing expected INS event at 3..4"
            ev = evs[0]
            gt_row = bun.genotypes[ev["event_id"]]
            ev_ins = [
                (r["allele"], r["pan_end"] - r["pan_start"], r["src_intervals"])
                for r in bun.inserts
                if r["event_id"] == ev["event_id"]
            ]
            pair_res.append((gt_row, ev["alleles"], ev_ins))

        (gt_f, alleles_f, ins_f), (gt_r, alleles_r, ins_r) = pair_res
        assert gt_f == gt_r, f"{name} FWD/REV gt mismatch: {gt_f} vs {gt_r}"
        assert gt_f["HapTest#1"] == "6", f"{name} unexpected HapTest#1 gt: {gt_f['HapTest#1']}"
        assert alleles_f == alleles_r, f"{name} FWD/REV alleles mismatch:\n  FWD: {alleles_f}\n  REV: {alleles_r}"
        assert "1:len=380," in alleles_f and "6:len=140," in alleles_f, (
            f"{name} missing expected 1:len=380 / 6:len=140 in alleles: {alleles_f}"
        )
        assert ins_f == ins_r, f"{name} FWD/REV inserts mismatch:\n  FWD: {ins_f}\n  REV: {ins_r}"


def test_e2e_iter15_multi_snp_str_island_closures(tmp_path: Path) -> None:
    """Iter 15 Strictly Opaque-Box CLI E2E Regressions for BUG-R14-1 (ZERO `panflat` imports):
    Exercises BOTH live 2-interior-SNP Long-STR topologies x FWD/REV (= 4 `run_build`
    + 4 `run_verify(..., check=True)` cycles):
    1. `TRIAL129_LIVE_CLI_2INTSNP` (`GAACT`, 2 interior SNPs at 57 & 159, `p1=55, p2=45, p3=45, pE=20`,
       `--dedup-mode pan --dedup-id 0.90 --min-sv-len 50`): previously swapped kept/dropped alleles
       between FWD and REV.
    2. `gaact_085` (`GAACT`, 2 interior SNPs at 57 & 159, `p1=50, p2=50, p3=45, pE=20`,
       `--dedup-mode pan --dedup-id 0.85 --min-sv-len 50`): previously silently dropped `HapTest#1`'s
       20 bp `pE` allele on REV only.
    Asserts exact `FWD == REV` parity on `gt`, `alleles`, and `inserts`, plus `panflat verify == PASS`.
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    base_ref6 = {str(i): _mseq(400, 69500 + i) for i in range(1, 7)}
    ref6_steps = [f"{i}+" for i in range(1, 7)]
    sup8 = [(f"Sup{k}#1#Chr01", ["1+", "10+", "2+", "3+", "4+", "5+", "6+"]) for k in range(1, 9)]

    l0 = _mseq(1500, 99101)[:-1] + "G"
    r0 = "C" + _mseq(1500, 99102)[1:]
    str_r = "GAACT" * 33
    str_q_arr = list("GAACT" * 34)
    str_q_arr[57] = "G"
    str_q_arr[159] = "C"
    str_q = "".join(str_q_arr)
    sup_fwd = l0 + str_r + r0
    r0_tail = str_q[165:170] + r0

    cases = (
        (
            "trial129_2intsnp_090",
            "0.90",
            [
                ("p1", str_q[0:55]),
                ("p2", str_q[55:100]),
                ("p3", str_q[100:145]),
                ("pE", str_q[145:165]),
            ],
        ),
        (
            "gaact_085",
            "0.85",
            [
                ("p1", str_q[0:50]),
                ("p2", str_q[50:100]),
                ("p3", str_q[100:145]),
                ("pE", str_q[145:165]),
            ],
        ),
    )

    for name, dedup_id, p_list in cases:
        t_nodes = {f"t{i}": _mseq(60, 82500 + i) for i in range(len(p_list))}
        p_steps = [f"{k}+" for k, _ in p_list]
        t_steps = [f"t{i}+" for i in range(len(p_list))]
        pair_res = []
        for sfx, sup_seq in (("fwd", sup_fwd), ("rev", revcomp(sup_fwd))):
            tag = f"i15_{name}_{sfx}"
            b = SyntheticInputBuilder(tmp_path / tag, chrom="Chr01", backbone_prefix="Ref#1#")
            for nid, sq in base_ref6.items():
                b.add_node(nid, sq)
            extra_nodes = {"10": sup_seq, "L0": l0, "R0": r0_tail, **dict(p_list), **t_nodes}
            for nid, sq in extra_nodes.items():
                b.add_node(nid, sq)
            b.set_ref_path("Ref#1#Chr01", ref6_steps)
            alt_paths = (
                sup8
                + [
                    ("Hap1a#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                    ("Hap1b#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+"] + t_steps + ["5+", "6+"]),
                ]
                + [
                    (
                        f"HapDrop{i}#1#Chr01",
                        ["1+", "2+", "3+", "4+", "L0+"]
                        + [s for j, s in enumerate(p_steps) if j != i]
                        + ["R0+"]
                        + [s for j, s in enumerate(t_steps) if j != i]
                        + ["5+", "6+"],
                    )
                    for i in range(len(p_list))
                ]
                + [("HapTest#1#Chr01", ["1+", "2+", "3+", "4+", "L0+"] + p_steps + ["R0+", "5+", "6+"])]
            )
            for pname, steps in alt_paths:
                b.add_path(pname, steps)
            inp = b.write(tag)
            out_pref = tmp_path / f"{tag}_out"
            proc, bun = run_build(
                inp,
                extra_args=["--dedup-mode", "pan", "--dedup-id", dedup_id, "--min-sv-len", "50"],
                out_prefix=out_pref,
            )
            assert proc.returncode == 0 and bun is not None, f"{tag} build failed:\n{proc.stderr}"
            run_verify(out_pref, inp.gfa_path, inp.ref_fa_path, check=True)
            evs = [r for r in bun.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4"]
            assert evs, f"{tag}: missing expected INS event at 3..4"
            ev = evs[0]
            gt_row = bun.genotypes[ev["event_id"]]
            ev_ins = [
                (r["allele"], r["pan_end"] - r["pan_start"], r["src_intervals"])
                for r in bun.inserts
                if r["event_id"] == ev["event_id"]
            ]
            pair_res.append(
                (gt_row, ev["alleles"], ev_ins, bun.summary_kv.get("dedup_pan_masked_bp"))
            )

        (gt_f, alleles_f, ins_f, pm_f), (gt_r, alleles_r, ins_r, pm_r) = pair_res
        assert pm_f == pm_r == "3170", f"{name} unexpected dedup_pan_masked_bp: fwd={pm_f} rev={pm_r}"
        assert gt_f == gt_r, f"{name} FWD/REV gt mismatch: {gt_f} vs {gt_r}"
        assert gt_f["HapTest#1"] == ".", f"{name} expected HapTest#1 == '.', got {gt_f['HapTest#1']}"
        assert alleles_f == alleles_r, f"{name} FWD/REV alleles mismatch:\n  FWD: {alleles_f}\n  REV: {alleles_r}"
        assert "1:len=240," in alleles_f, f"{name} expected 1:len=240 in alleles: {alleles_f}"
        assert ins_f == ins_r and len(ins_f) == 4, f"{name} FWD/REV inserts mismatch:\n  FWD: {ins_f}\n  REV: {ins_r}"

