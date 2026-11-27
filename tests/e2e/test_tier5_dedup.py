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
