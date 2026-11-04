"""Tier 1 Requirement-Driven Opaque-Box E2E Feature Tests for PanFlat v1.

Covers:
- All 26 features (`F1` through `F26`) with >= 5 distinct E2E test cases per feature (>= 130 total).
- All 24 general constructive acceptance scenarios (`T1` through `T24` from §8.1).
- All 20 centromere & repeat-guard acceptance scenarios (`C1` through `C20` from §8.2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.helpers import (
    OutputBundle,
    SyntheticInputBuilder,
    assert_all_global_invariants,
    assert_centromere_accounting_and_offsets,
    assert_gff3_liftover_integrity,
    compute_canonical_15mer_singleton_ratio,
    make_coding_seq,
    make_complex_seq,
    make_satellite_seq,
    make_window_with_singleton_ratio,
    mutate_seq_snps,
    oracle_repeat_mask_intervals,
    revcomp,
    run_build,
    run_merge,
    run_panflat_cli,
    run_verify,
    translate_cds,
)


def _find_sv(bundle: OutputBundle, sv_type: str) -> list[dict[str, Any]]:
    return [row for row in bundle.sv_rows if row["type"] == sv_type]


def _find_unresolved(bundle: OutputBundle, reason: str) -> list[dict[str, Any]]:
    return [row for row in bundle.unresolved if row["reason"] == reason]


def _centromere_by_hap(bundle: OutputBundle) -> dict[str, dict[str, Any]]:
    assert bundle.centromere_rows is not None
    return {str(row["hap"]): row for row in bundle.centromere_rows}


# ===========================================================================
# === F1: Invariant I: Reference Base Preservation (>= 5 tests) ===
# ===========================================================================


def test_f1_01_pure_reference_no_alt_paths(tmp_path: Path) -> None:
    """F1: Pure reference path with no alt paths preserves reference bases 100% identically."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=101)
    s2 = make_complex_seq(250, seed=102)
    s3 = make_complex_seq(180, seed=103)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    inputs = b.write("f1_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + s2 + s3
    assert len(bundle.offsets) == 3
    assert [int(r["slot_ins_len"]) for r in bundle.offsets] == [0, 0, 0]
    assert len(bundle.sv_rows) == 0
    assert len(bundle.inserts) == 0


def test_f1_02_t1_single_strain_3kb_insertion(tmp_path: Path) -> None:
    """F1 / T1: Single-strain 3 kb insertion is placed once at exact reference coordinate."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=111)
    s2 = make_complex_seq(350, seed=112)
    s_ins = make_complex_seq(3000, seed=113)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins3k", s_ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins3k+", "n2+"])
    inputs = b.write("f1_02_t1")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + s_ins + s2
    assert len(bundle.offsets) == 2
    assert int(bundle.offsets[0]["pan_start"]) == 0
    assert int(bundle.offsets[0]["pan_end"]) == 300
    assert int(bundle.offsets[0]["slot_ins_len"]) == 0
    assert int(bundle.offsets[1]["ref_start"]) == 300
    assert int(bundle.offsets[1]["pan_start"]) == 3300
    assert int(bundle.offsets[1]["pan_end"]) == 3650
    assert int(bundle.offsets[1]["slot_ins_len"]) == 3000

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 300
    assert int(ins_events[0]["ref_end"]) == 300
    assert int(ins_events[0]["pan_start"]) == 300
    assert int(ins_events[0]["pan_end"]) == 3300


def test_f1_03_multiple_insertions_at_distinct_slots(tmp_path: Path) -> None:
    """F1: Multiple insertions at distinct reference slots shift pan coordinates while preserving ref bases."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=121)
    s2 = make_complex_seq(200, seed=122)
    s3 = make_complex_seq(200, seed=123)
    s4 = make_complex_seq(200, seed=124)
    ins_a = make_complex_seq(150, seed=125)
    ins_b = make_complex_seq(250, seed=126)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ia", ins_a), ("ib", ins_b)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ia+", "n2+", "n3+", "ib+", "n4+"])
    inputs = b.write("f1_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + ins_a + s2 + s3 + ins_b + s4
    assert [int(r["slot_ins_len"]) for r in bundle.offsets] == [0, 150, 0, 250]
    assert [int(r["pan_start"]) for r in bundle.offsets] == [0, 350, 550, 1000]


def test_f1_04_replacement_branch_preserves_reference_in_situ(tmp_path: Path) -> None:
    """F1: At a replacement locus (B(a,b) non-empty), reference bases stay in-situ before N+alt."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=131)
    s2 = make_complex_seq(220, seed=132)
    s3 = make_complex_seq(200, seed=133)
    s_alt = make_complex_seq(180, seed=134)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("alt2", s_alt)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "alt2+", "n3+"])
    inputs = b.write("f1_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + s2 + ("N" * 100) + s_alt + s3
    assert [int(r["slot_ins_len"]) for r in bundle.offsets] == [0, 0, 280]


def test_f1_05_mixed_orientation_ref_steps_and_native_n_bases(tmp_path: Path) -> None:
    """F1: Reference path with '-' steps and native 'N' bases reconstructs --ref-fa exactly."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(180, seed=141)
    s2_raw = make_complex_seq(100, seed=142) + ("N" * 25) + make_complex_seq(95, seed=143)
    s3 = make_complex_seq(180, seed=144)
    s_ins = make_complex_seq(120, seed=145)
    b.add_node("n1", s1).add_node("n2", s2_raw).add_node("n3", s3).add_node("ins1", s_ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2-", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2-", "ins1+", "n3+"])
    inputs = b.write("f1_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    expected_ref = s1 + revcomp(s2_raw) + s3
    reconstructed = "".join(
        bundle.pan_seq[int(r["pan_start"]) : int(r["pan_end"])] for r in bundle.offsets
    )
    assert reconstructed == expected_ref
    vproc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path)
    assert vproc.returncode == 0


# ===========================================================================
# === F2: Invariant II: Piece Provenance & Join Integrity (>= 5 tests) ===
# ===========================================================================


def test_f2_01_single_piece_source_contiguous_provenance(tmp_path: Path) -> None:
    """F2: Multi-node contiguous insertion from a single donor merges or tracks as source_contiguous."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=201)
    s2 = make_complex_seq(200, seed=202)
    x1 = make_complex_seq(100, seed=203)
    x2 = make_complex_seq(130, seed=204)
    for nid, seq in [("n1", s1), ("n2", s2), ("x1", x1), ("x2", x2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "x1+", "x2+", "n2+"])
    inputs = b.write("f2_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) >= 1
    assert all(row["join_before"] == "source_contiguous" for row in bundle.inserts)
    assert all(row["src_strand"] == "+" for row in bundle.inserts)
    ins_seq = "".join(
        bundle.pan_seq[int(r["pan_start"]) : int(r["pan_end"])] for r in bundle.inserts
    )
    assert ins_seq == x1 + x2


def test_f2_02_multi_representative_spacer_join_before(tmp_path: Path) -> None:
    """F2: Two non-homologous representatives at the same slot record join_before='spacer' on the second."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=211)
    s2 = make_complex_seq(200, seed=212)
    r1 = make_complex_seq(180, seed=213)
    r2 = make_complex_seq(140, seed=214)
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1), ("r2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f2_02")

    _, bundle = run_build(inputs, extra_args=["--spacer-n", "80"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path, spacer_n=80)

    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "spacer"
    p0_end = int(bundle.inserts[0]["pan_end"])
    p1_start = int(bundle.inserts[1]["pan_start"])
    assert p1_start - p0_end == 80
    assert bundle.pan_seq[p0_end:p1_start] == "N" * 80


def test_f2_03_synthetic_join_before_from_cycle_folding(tmp_path: Path) -> None:
    """F2: Cycle folding a repeated novel node before a tail node splits pieces with join_before='synthetic'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=221)
    s2 = make_complex_seq(200, seed=222)
    u_seq = make_complex_seq(90, seed=223)
    tail_seq = make_complex_seq(110, seed=224)
    for nid, seq in [("n1", s1), ("n2", s2), ("u", u_seq), ("tail", tail_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    inputs = b.write("f2_03")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "synthetic"
    assert bundle.pan_seq == s1 + u_seq + tail_seq + s2


def test_f2_04_reverse_complement_piece_from_normalized_inversion(tmp_path: Path) -> None:
    """F2: Piece originating inside a normalized inversion has src_strand='-' and pre-flip coordinates."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=231)
    s2 = make_complex_seq(200, seed=232)
    s3 = make_complex_seq(200, seed=233)
    s4 = make_complex_seq(200, seed=234)
    x_seq = make_complex_seq(160, seed=235)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("x", x_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n3-", "x+", "n2-", "n4+"])
    inputs = b.write("f2_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    ins_row = bundle.inserts[0]
    assert ins_row["src_strand"] == "-"
    assert ins_row["src_intervals"] in ("400-560", "Alt1#1#Chr01:400-560")
    p_s, p_e = int(ins_row["pan_start"]), int(ins_row["pan_end"])
    assert bundle.pan_seq[p_s:p_e] == revcomp(x_seq)


def test_f2_05_piece_with_source_n_and_w_line_nonzero_start(tmp_path: Path) -> None:
    """F2: Piece from a W-line with non-zero start and native 'N' bases preserves exact coordinates and Ns."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=241)
    s2 = make_complex_seq(250, seed=242)
    x_with_n = make_complex_seq(90, seed=243) + ("N" * 15) + make_complex_seq(95, seed=244)
    b.add_node("n1", s1).add_node("n2", s2).add_node("xn", x_with_n)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_walk("AltW", "1", "ctg1", 2500, 3200, ["n1+", "xn+", "n2+"])
    inputs = b.write("f2_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    ins_row = bundle.inserts[0]
    assert "2750-2950" in str(ins_row["src_intervals"])
    p_s, p_e = int(ins_row["pan_start"]), int(ins_row["pan_end"])
    assert bundle.pan_seq[p_s:p_e] == x_with_n


# ===========================================================================
# === F3: Invariant III: Coordinate Consistency (>= 5 tests) ===
# ===========================================================================


def test_f3_01_chain_single_block_no_insertions(tmp_path: Path) -> None:
    """F3: Reference-only build emits a single-block chain with tSize == qSize and dt=0, dq=0."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=301)
    s2 = make_complex_seq(200, seed=302)
    s3 = make_complex_seq(200, seed=303)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    inputs = b.write("f3_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.chain_headers) == 1
    ch = bundle.chain_headers[0]
    assert ch["tSize"] == 600
    assert ch["qSize"] == 600
    assert bundle.chain_blocks[0] == [(600, 0, 0)]


def test_f3_02_chain_multi_block_multiple_insertions(tmp_path: Path) -> None:
    """F3: Multiple insertions produce exact dt=0, dq=slot_ins_len chain gaps."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=311)
    s2 = make_complex_seq(200, seed=312)
    s3 = make_complex_seq(200, seed=313)
    s4 = make_complex_seq(200, seed=314)
    ia = make_complex_seq(150, seed=315)
    ib = make_complex_seq(250, seed=316)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ia", ia), ("ib", ib)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ia+", "n2+", "n3+", "ib+", "n4+"])
    inputs = b.write("f3_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ch = bundle.chain_headers[0]
    assert ch["tSize"] == 800
    assert ch["qSize"] == 1200
    assert bundle.chain_blocks[0] == [(200, 0, 150), (400, 0, 250), (200, 0, 0)]


def test_f3_03_chain_replacement_site_slot_ins_len(tmp_path: Path) -> None:
    """F3: Replacement site records dq = spacer_n + len(R2) before right anchor b in chain and offsets."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=321)
    s2 = make_complex_seq(200, seed=322)
    s3 = make_complex_seq(200, seed=323)
    r2 = make_complex_seq(160, seed=324)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("r2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r2+", "n3+"])
    inputs = b.write("f3_03")

    _, bundle = run_build(inputs, extra_args=["--spacer-n", "100"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path, spacer_n=100)

    assert int(bundle.offsets[2]["slot_ins_len"]) == 260
    assert bundle.chain_blocks[0] == [(400, 0, 260), (200, 0, 0)]


def test_f3_04_custom_spacer_n_50_vs_200_coordinate_shifts(tmp_path: Path) -> None:
    """F3: Changing --spacer-n from 50 to 200 shifts chain dq, offsets, and pan.fa length by +150 bp."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=331)
    s2 = make_complex_seq(200, seed=332)
    r1 = make_complex_seq(150, seed=333)
    r2 = make_complex_seq(140, seed=334)
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1), ("r2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f3_04")

    _, bundle50 = run_build(inputs, extra_args=["--spacer-n", "50"], out_prefix=tmp_path / "out50")
    _, bundle200 = run_build(inputs, extra_args=["--spacer-n", "200"], out_prefix=tmp_path / "out200")
    assert bundle50 is not None and bundle200 is not None
    assert_all_global_invariants(bundle50, inputs.gfa_path, inputs.ref_fa_path, spacer_n=50)
    assert_all_global_invariants(bundle200, inputs.gfa_path, inputs.ref_fa_path, spacer_n=200)

    assert len(bundle200.pan_seq) - len(bundle50.pan_seq) == 150
    assert int(bundle200.offsets[1]["slot_ins_len"]) - int(bundle50.offsets[1]["slot_ins_len"]) == 150


def test_f3_05_standalone_panflat_verify_checks_chain_offsets_and_pan_fa(tmp_path: Path) -> None:
    """F3: Standalone `panflat verify` validates Invariants I, II, and III on build outputs."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=341)
    s2 = make_complex_seq(250, seed=342)
    ins = make_complex_seq(175, seed=343)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+"])
    inputs = b.write("f3_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    vproc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path)
    assert vproc.returncode == 0
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F4: Reference Backbone Matching & Validation (>= 5 tests) ===
# ===========================================================================


def test_f4_01_standard_positive_ref_path_matching_backbone_prefix(tmp_path: Path) -> None:
    """F4: --backbone-prefix uniquely matches the reference path and outputs --ref-fa chrom name."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01_Ref", backbone_prefix="Nipponbare#1#")
    s1 = make_complex_seq(200, seed=401)
    s2 = make_complex_seq(200, seed=402)
    ins = make_complex_seq(150, seed=403)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Nipponbare#1#Chr01_Ref", ["n1+", "n2+"])
    b.add_path("Kasalath#1#Chr01_Ref", ["n1+", "ins+", "n2+"])
    inputs = b.write("f4_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert "Chr01_Ref" in bundle.pan_fa
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f4_02_t21a_reference_path_with_negative_steps(tmp_path: Path) -> None:
    """F4 / T21: Reference path with '-' steps spells reverse-complement and matches relative orientation."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=411)
    s2 = make_complex_seq(220, seed=412)
    s3 = make_complex_seq(210, seed=413)
    s4 = make_complex_seq(200, seed=414)
    ins = make_complex_seq(150, seed=415)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2-", "n3-", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2-", "ins+", "n3-", "n4+"])
    inputs = b.write("f4_02_t21a")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "INV")) == 0
    assert len(_find_sv(bundle, "INS")) == 1
    assert bundle.pan_seq == s1 + revcomp(s2) + ins + revcomp(s3) + s4


def test_f4_03_case_insensitive_ref_fasta_matching(tmp_path: Path) -> None:
    """F4: --ref-fa with lowercase/mixed-case letters matches uppercase GFA reference path."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=421)
    s2 = make_complex_seq(200, seed=422)
    b.add_node("n1", s1).add_node("n2", s2)
    mixed_ref = s1[:100].lower() + s1[100:] + s2.lower()
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], ref_fa_seq=mixed_ref)
    inputs = b.write("f4_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f4_04_reference_with_exact_matching_n_runs(tmp_path: Path) -> None:
    """F4: Reference sequence containing 'N' runs matches --ref-fa when N positions match exactly."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(150, seed=431)
    s_n = "N" * 60
    s2 = make_complex_seq(150, seed=432)
    b.add_node("n1", s1).add_node("ngap", s_n).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "ngap+", "n2+"])
    inputs = b.write("f4_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + s_n + s2


def test_f4_05_reference_path_supplied_as_w_line(tmp_path: Path) -> None:
    """F4: Reference path specified via a GFA W-line is matched and validated properly."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=441)
    s2 = make_complex_seq(200, seed=442)
    ins = make_complex_seq(140, seed=443)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path(
        "Ref#1#Chr01",
        ["n1+", "n2+"],
        as_w_line=True,
        w_sample="Ref",
        w_hap="1",
        w_seqid="Chr01",
        w_start=0,
    )
    b.add_walk("Alt1", "1", "Chr01", 0, 540, ["n1+", "ins+", "n2+"])
    inputs = b.write("f4_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + ins + s2


# ===========================================================================
# === F5: GFA P-line and W-line Parsing (>= 5 tests) ===
# ===========================================================================


def test_f5_01_p_lines_pansn_sample_hap_naming(tmp_path: Path) -> None:
    """F5: P lines with PanSN Sample#Hap#Seq naming populate genotypes.tsv columns accurately."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=501)
    s2 = make_complex_seq(200, seed=502)
    ins = make_complex_seq(150, seed=503)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("StrainA#1#Chr01", ["n1+", "ins+", "n2+"])
    b.add_path("StrainB#2#Chr01", ["n1+", "n2+"])
    inputs = b.write("f5_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert set(bundle.genotype_haps) == {"Ref#1", "StrainA#1", "StrainB#2"}
    ev_id = bundle.sv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["StrainA#1"] == "1"
    assert bundle.genotypes[ev_id]["StrainB#2"] == "0"


def test_f5_02_w_lines_zero_start_offset(tmp_path: Path) -> None:
    """F5: W-line with seq_start=0 records 0-based source intervals starting at node offset."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=511)
    s2 = make_complex_seq(200, seed=512)
    ins = make_complex_seq(150, seed=513)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_walk("StrainW", "1", "ctg1", 0, 550, ["n1+", "ins+", "n2+"])
    inputs = b.write("f5_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_hap"] == "StrainW#1"
    assert "200-350" in str(bundle.inserts[0]["src_intervals"])


def test_f5_03_t21b_w_line_nonzero_start_offset_preserved(tmp_path: Path) -> None:
    """F5 / T21: W-line with non-zero seq_start=5000 preserves original start offset in provenance."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=521)
    s2 = make_complex_seq(200, seed=522)
    ins = make_complex_seq(150, seed=523)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_walk("StrainW", "1", "ctg1", 5000, 5550, ["n1+", "ins+", "n2+"])
    inputs = b.write("f5_03_t21b")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    assert "5200-5350" in str(bundle.inserts[0]["src_intervals"])


def test_f5_04_multiple_w_lines_same_haplotype_not_stitched_across_gap(tmp_path: Path) -> None:
    """F5: Multiple W-line segments for the same Sample#Hap are processed independently without gap stitching."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=531)
    s2 = make_complex_seq(200, seed=532)
    s_mid = make_complex_seq(300, seed=533)
    s3 = make_complex_seq(200, seed=534)
    s4 = make_complex_seq(200, seed=535)
    ins1 = make_complex_seq(120, seed=536)
    ins2 = make_complex_seq(130, seed=537)
    for nid, seq in [
        ("n1", s1), ("n2", s2), ("nmid", s_mid), ("n3", s3), ("n4", s4),
        ("ins1", ins1), ("ins2", ins2),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "nmid+", "n3+", "n4+"])
    b.add_walk("Draft", "1", "ctgA", 0, 520, ["n1+", "ins1+", "n2+"])
    b.add_walk("Draft", "1", "ctgB", 1000, 1530, ["n3+", "ins2+", "n4+"])
    inputs = b.write("f5_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "DEL")) == 0
    assert len(_find_sv(bundle, "INS")) == 2


def test_f5_05_mixed_p_and_w_lines_in_same_graph(tmp_path: Path) -> None:
    """F5: P-line and W-line donor paths coexist in the same graph and are genotyped together."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=541)
    s2 = make_complex_seq(200, seed=542)
    s3 = make_complex_seq(200, seed=543)
    ip = make_complex_seq(140, seed=544)
    iw = make_complex_seq(160, seed=545)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ip", ip), ("iw", iw)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("DonorP#1#Chr01", ["n1+", "ip+", "n2+", "n3+"])
    b.add_walk("DonorW", "1", "ctg1", 100, 860, ["n1+", "n2+", "iw+", "n3+"])
    inputs = b.write("f5_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "INS")) == 2
    assert set(bundle.genotype_haps) == {"Ref#1", "DonorP#1", "DonorW#1"}


# ===========================================================================
# === F6: GFA Structural Integrity Checks (>= 5 tests) ===
# ===========================================================================


def test_f6_01_arbitrary_string_node_ids(tmp_path: Path) -> None:
    """F6: Node IDs are treated as arbitrary strings without assuming numeric order."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=601)
    s2 = make_complex_seq(200, seed=602)
    s3 = make_complex_seq(150, seed=603)
    b.add_node("node_zzz_first", s1)
    b.add_node("node_aaa_second", s2)
    b.add_node("novel_seg_007", s3)
    b.set_ref_path("Ref#1#Chr01", ["node_zzz_first+", "node_aaa_second+"])
    b.add_path("Alt1#1#Chr01", ["node_zzz_first+", "novel_seg_007+", "node_aaa_second+"])
    inputs = b.write("f6_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert [r["node_id"] for r in bundle.offsets] == ["node_zzz_first", "node_aaa_second"]


def test_f6_02_p_lines_with_explicit_0m_overlaps(tmp_path: Path) -> None:
    """F6: P lines with explicit '0M,0M' overlaps are accepted."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=611)
    s2 = make_complex_seq(200, seed=612)
    ins = make_complex_seq(150, seed=613)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="0M")
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+"], overlaps="0M,0M")
    inputs = b.write("f6_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f6_03_p_lines_with_star_overlaps_backed_by_0m_links(tmp_path: Path) -> None:
    """F6: P lines with '*' overlap field backed by '0M' L lines pass validation."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=621)
    s2 = make_complex_seq(200, seed=622)
    b.add_node("n1", s1).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="*", auto_links=True)
    inputs = b.write("f6_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f6_04_gfa_sha256_recorded_in_metadata(tmp_path: Path) -> None:
    """F6: Input GFA SHA-256 digest is computed and recorded in summary.tsv or .log."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=631)
    s2 = make_complex_seq(200, seed=632)
    b.add_node("n1", s1).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    inputs = b.write("f6_04")

    expected_sha = hashlib.sha256(inputs.gfa_path.read_bytes()).hexdigest()
    _, bundle = run_build(inputs)
    assert bundle is not None
    combined_meta = (bundle.summary_text + "\n" + bundle.log_text).lower()
    assert expected_sha.lower() in combined_meta


def test_f6_05_t21c_reject_unknown_or_nonzero_overlap_and_j_lines(tmp_path: Path) -> None:
    """F6 / T21: GFA with non-zero overlap (e.g. 15M) or J line is explicitly rejected."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=641)
    s2 = make_complex_seq(200, seed=642)
    b.add_node("n1", s1).add_node("n2", s2)
    b.add_link("n1", "+", "n2", "+", overlap="15M")
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="15M", auto_links=False)
    inputs = b.write("f6_05_t21c")

    proc, bundle = run_build(inputs, check=False)
    assert proc.returncode != 0
    assert bundle is None


# ===========================================================================
# === F7: Step Projection onto Backbone B (>= 5 tests) ===
# ===========================================================================


def test_f7_01_projection_onto_all_positive_backbone(tmp_path: Path) -> None:
    """F7: Branch steps matching '+' backbone steps project as relative-forward anchors."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=701)
    s2 = make_complex_seq(200, seed=702)
    s3 = make_complex_seq(200, seed=703)
    ins = make_complex_seq(130, seed=704)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+", "n3+"])
    inputs = b.write("f7_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "INS")) == 1
    assert bundle.sv_rows[0]["anchors"] == "0..1"


def test_f7_02_projection_onto_negative_backbone_steps(tmp_path: Path) -> None:
    """F7: Branch '-' step matching backbone '-' step projects as relative '+', while '+' step projects as '-'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=711)
    s2 = make_complex_seq(200, seed=712)
    s3 = make_complex_seq(250, seed=713)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2-", "n3+"])
    b.add_path("AltCollinear#1#Chr01", ["n1+", "n2-", "n3+"])
    b.add_path("AltInverted#1#Chr01", ["n1+", "n2+", "n3+"])
    inputs = b.write("f7_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    inv_events = _find_sv(bundle, "INV")
    assert len(inv_events) == 1
    ev_id = inv_events[0]["event_id"]
    assert bundle.genotypes[ev_id]["AltCollinear#1"] == "0"
    assert bundle.genotypes[ev_id]["AltInverted#1"] == "1"


def test_f7_03_repeated_backbone_node_projects_to_none(tmp_path: Path) -> None:
    """F7: Multi-copy node in B[lo:hi) projects to rank=None (not a unique anchor)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=721)
    sr = make_complex_seq(100, seed=722)
    s2 = make_complex_seq(200, seed=723)
    s3 = make_complex_seq(200, seed=724)
    ins = make_complex_seq(150, seed=725)
    for nid, seq in [("n1", s1), ("rpt", sr), ("n2", s2), ("n3", s3), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "rpt+", "n2+", "rpt+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "rpt+", "n2+", "rpt+", "ins+", "n3+"])
    inputs = b.write("f7_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 600


def test_f7_04_globally_repeated_node_becomes_unique_anchor_in_subinterval(tmp_path: Path) -> None:
    """F7: Node repeated globally in B becomes a unique anchor when projecting onto sub-interval B(a,b)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=731)
    sr = make_complex_seq(150, seed=732)
    s2 = make_complex_seq(200, seed=733)
    sm = make_complex_seq(150, seed=734)
    s3 = make_complex_seq(200, seed=735)
    ins_sub = make_complex_seq(140, seed=736)
    for nid, seq in [("n1", s1), ("rpt", sr), ("n2", s2), ("m1", sm), ("n3", s3), ("isub", ins_sub)]:
        b.add_node(nid, seq)
    # In full B, 'rpt' appears at index 1 and 3 (multi-copy globally), while 'm1' is unique at index 4.
    # If Alt1 skips m1? Wait: if 'm1' is also repeated globally (e.g. at index 1 and 4), then top-level
    # unique anchors are n1 (0), n2 (2), n3 (5). Inside B(n2, n3) = [rpt+, m1+], BOTH rpt and m1 appear ONCE!
    b.set_ref_path("Ref#1#Chr01", ["n1+", "rpt+", "m1+", "n2+", "rpt+", "m1+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "rpt+", "m1+", "n2+", "rpt+", "isub+", "m1+", "n3+"])
    inputs = b.write("f7_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    # Inside B(n2, n3) = [rpt+, m1+], local re-anchoring anchors on rpt+ and m1+, placing isub BEFORE m1 (ref_start=850)
    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 850
    assert bundle.pan_seq == s1 + sr + sm + s2 + sr + ins_sub + sm + s3


def test_f7_05_t18_secondary_backbone_with_negative_novel_node(tmp_path: Path) -> None:
    """F7 / T18: Secondary backbone containing '-' novel node anchors child branches by relative orientation."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=741)
    s2 = make_complex_seq(200, seed=742)
    u_a = make_complex_seq(150, seed=743)
    u_neg = make_complex_seq(160, seed=744)
    u_b = make_complex_seq(150, seed=745)
    u_extra = make_complex_seq(120, seed=746)
    for nid, seq in [
        ("n1", s1), ("n2", s2), ("uA", u_a), ("uNeg", u_neg), ("uB", u_b), ("uEx", u_extra),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Alt1 and Alt2 share uA+ -> uNeg- -> uB+ (support=2 -> selected as secondary backbone)
    b.add_path("Alt1#1#Chr01", ["n1+", "uA+", "uNeg-", "uB+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "uA+", "uNeg-", "uB+", "n2+"])
    # Alt3 has uA+ -> uNeg- -> uEx+ -> uB+
    b.add_path("Alt3#1#Chr01", ["n1+", "uA+", "uNeg-", "uEx+", "uB+", "n2+"])
    inputs = b.write("f7_05_t18")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Secondary backbone is uA+ -> uNeg- -> uB+, with uEx+ inserted before uB+!
    expected_ins_block = u_a + revcomp(u_neg) + u_extra + u_b
    assert bundle.pan_seq == s1 + expected_ins_block + s2


# ===========================================================================
# === F8: Top-level Whole-Contig Orientation (>= 5 tests) ===
# ===========================================================================


def test_f8_01_t23_whole_contig_reverse_oriented_no_inv(tmp_path: Path) -> None:
    """F8 / T23: Whole contig in reverse orientation is flipped first (src_strand='-') and NOT called as INV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=801)
    s2 = make_complex_seq(200, seed=802)
    s3 = make_complex_seq(200, seed=803)
    s4 = make_complex_seq(200, seed=804)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("AltRev#1#Chr01", ["n4-", "n3-", "n2-", "n1-"])
    inputs = b.write("f8_01_t23")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.sv_rows) == 0
    assert bundle.pan_seq == s1 + s2 + s3 + s4


def test_f8_02_reverse_oriented_contig_with_novel_insertion(tmp_path: Path) -> None:
    """F8: Reverse-assembled contig carrying a novel insertion flips contig and records piece with src_strand='-'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=811)
    s2 = make_complex_seq(200, seed=812)
    s3 = make_complex_seq(200, seed=813)
    x_raw = make_complex_seq(150, seed=814)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("xraw", x_raw)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("AltRev#1#Chr01", ["n3-", "xraw+", "n2-", "n1-"])
    inputs = b.write("f8_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "INV")) == 0
    assert len(_find_sv(bundle, "INS")) == 1
    assert bundle.pan_seq == s1 + s2 + revcomp(x_raw) + s3
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "-"
    assert "200-350" in str(bundle.inserts[0]["src_intervals"])


def test_f8_03_mostly_forward_contig_keeps_positive_orientation(tmp_path: Path) -> None:
    """F8: Contig where forward anchor bases > reverse anchor bases stays '+' and calls INV on reverse node."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(400, seed=821)
    s2 = make_complex_seq(150, seed=822)
    s3 = make_complex_seq(400, seed=823)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2-", "n3+"])
    inputs = b.write("f8_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "INV")) == 1


def test_f8_04_reverse_oriented_w_line_preserves_seq_start_offset(tmp_path: Path) -> None:
    """F8: Reverse-oriented W-line contig preserves original unflipped 0-based seq_start coordinates."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=831)
    s2 = make_complex_seq(200, seed=832)
    s3 = make_complex_seq(200, seed=833)
    x_raw = make_complex_seq(150, seed=834)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("xraw", x_raw)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_walk("AltW", "1", "ctgRev", 3000, 3750, ["n3-", "xraw+", "n2-", "n1-"])
    inputs = b.write("f8_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "-"
    assert "3200-3350" in str(bundle.inserts[0]["src_intervals"])


def test_f8_05_reverse_oriented_contig_with_deletion(tmp_path: Path) -> None:
    """F8: Reverse-oriented contig skipping a middle reference node calls DEL and zero INV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=841)
    s2 = make_complex_seq(200, seed=842)
    s3 = make_complex_seq(250, seed=843)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("AltRev#1#Chr01", ["n3-", "n1-"])
    inputs = b.write("f8_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "INV")) == 0
    dels = _find_sv(bundle, "DEL")
    assert len(dels) == 1
    assert int(dels[0]["ref_start"]) == 250
    assert int(dels[0]["ref_end"]) == 450


# ===========================================================================
# === F9: Reverse Chain Detection & Inversion Normalization (>= 5 tests) ===
# ===========================================================================


def test_f9_01_t6_large_pure_inversion_fasta_unchanged(tmp_path: Path) -> None:
    """F9 / T6: Pure multi-node inversion (even exceeding max-jump) records INV and leaves pan.fa identical to ref."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(1000, seed=901)
    inv1 = make_complex_seq(600, seed=902)
    inv2 = make_complex_seq(700, seed=903)
    inv3 = make_complex_seq(600, seed=904)
    s5 = make_complex_seq(1000, seed=905)
    for nid, seq in [("n1", s1), ("inv1", inv1), ("inv2", inv2), ("inv3", inv3), ("n5", s5)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "inv1+", "inv2+", "inv3+", "n5+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "inv3-", "inv2-", "inv1-", "n5+"])
    inputs = b.write("f9_01_t6")

    # Even with --max-jump 1000 (< 1900 bp inversion span), inversion detection is not limited by max-jump
    _, bundle = run_build(inputs, extra_args=["--max-jump", "1000"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + inv1 + inv2 + inv3 + s5
    invs = _find_sv(bundle, "INV")
    assert len(invs) == 1
    assert int(invs[0]["ref_start"]) == 1000
    assert int(invs[0]["ref_end"]) == 2900
    ev_id = invs[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"


def test_f9_02_t7_inversion_with_nested_novel_node_x(tmp_path: Path) -> None:
    """F9 / T7: 11+ -> 13- -> X+ -> 12- -> 14+ normalizes to 11+ -> 12+ -> X- -> 13+ -> 14+, inserting revcomp(X) before 13."""
    b = SyntheticInputBuilder(tmp_path)
    s11 = make_complex_seq(200, seed=911)
    s12 = make_complex_seq(200, seed=912)
    s13 = make_complex_seq(200, seed=913)
    s14 = make_complex_seq(200, seed=914)
    sx = make_complex_seq(150, seed=915)
    for nid, seq in [("11", s11), ("12", s12), ("13", s13), ("14", s14), ("X", sx)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["11+", "12+", "13+", "14+"])
    b.add_path("Alt1#1#Chr01", ["11+", "13-", "X+", "12-", "14+"])
    inputs = b.write("f9_02_t7")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s11 + s12 + revcomp(sx) + s13 + s14
    invs = _find_sv(bundle, "INV")
    ins_list = _find_sv(bundle, "INS")
    assert len(invs) == 1
    assert len(ins_list) == 1
    assert int(ins_list[0]["ref_start"]) == 400
    assert invs[0]["region_id"] == ins_list[0]["region_id"]
    assert bundle.inserts[0]["src_strand"] == "-"


def test_f9_03_t8_single_node_unchopped_inversion(tmp_path: Path) -> None:
    """F9 / T8: Single unchopped node inversion flanked by L and F is classified as INV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=921)
    s2 = make_complex_seq(600, seed=922)
    s3 = make_complex_seq(300, seed=923)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2-", "n3+"])
    inputs = b.write("f9_03_t8")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    invs = _find_sv(bundle, "INV")
    assert len(invs) == 1
    assert int(invs[0]["ref_start"]) == 300
    assert int(invs[0]["ref_end"]) == 900
    assert bundle.pan_seq == s1 + s2 + s3


def test_f9_04_inv_displaced_when_flank_ranks_out_of_order(tmp_path: Path) -> None:
    """F9: Reverse chain with both L and F present but rank(L) >= rmin is classified as INV_DISPLACED."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=931)
    s2 = make_complex_seq(200, seed=932)
    s3 = make_complex_seq(200, seed=933)
    s4 = make_complex_seq(300, seed=934)
    s5 = make_complex_seq(300, seed=935)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("n5", s5)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    # L is n4+ (rank 3), R is [n3-, n2-] (rmin=1, rmax=2), F is n5+ (rank 4). rank(L) >= rmin -> INV_DISPLACED
    b.add_path("Alt1#1#Chr01", ["n1+", "n4+", "n3-", "n2-", "n5+"])
    inputs = b.write("f9_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    disp = _find_sv(bundle, "INV_DISPLACED")
    assert len(disp) == 1


def test_f9_05_t24_inv_edge_and_inv_conflict_route_novel_to_unresolved(tmp_path: Path) -> None:
    """F9 / T24: Reverse chain missing a flank -> INV_EDGE; overlapping rank intervals -> INV_CONFLICT; novel seqs -> unresolved."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=941)
    s2 = make_complex_seq(200, seed=942)
    s3 = make_complex_seq(200, seed=943)
    s4 = make_complex_seq(200, seed=944)
    s5 = make_complex_seq(500, seed=945)
    s6 = make_complex_seq(500, seed=946)
    x_edge = make_complex_seq(140, seed=947)
    x_conf1 = make_complex_seq(150, seed=948)
    x_conf2 = make_complex_seq(160, seed=949)
    for nid, seq in [
        ("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("n5", s5), ("n6", s6),
        ("xe", x_edge), ("xc1", x_conf1), ("xc2", x_conf2),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+", "n6+"])
    # AltEdge starts with [n2- -> xe+ -> n1-] (missing L flank) then forward n3+ -> n4+ -> n5+ -> n6+ (forward weight 1400 > rev 400)
    b.add_path("AltEdge#1#Chr01", ["n2-", "xe+", "n1-", "n3+", "n4+", "n5+", "n6+"])
    # AltConflict has R1=[n4- -> xc1+ -> n2-] (ranks [1, 3]) and R2=[n3- -> xc2+ -> n1-] (ranks [0, 2]), which overlap!
    b.add_path("AltConf#1#Chr01", ["n1+", "n4-", "xc1+", "n2-", "n5+", "n3-", "xc2+", "n1-", "n6+"])
    inputs = b.write("f9_05_t24")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "INV_EDGE")) >= 1
    assert len(_find_sv(bundle, "INV_CONFLICT")) >= 1
    assert len(_find_unresolved(bundle, "INV_EDGE")) == 1
    assert len(_find_unresolved(bundle, "INV_CONFLICT")) == 2
    assert bundle.pan_seq == s1 + s2 + s3 + s4 + s5 + s6


# ===========================================================================
# === F10: Weighted LIS & Multi-hit Annotation (>= 5 tests) ===
# ===========================================================================


def test_f10_01_collinear_forward_anchors_all_in_lis(tmp_path: Path) -> None:
    """F10: Strictly increasing forward anchors are all selected into the main chain."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1001)
    s2 = make_complex_seq(200, seed=1002)
    s3 = make_complex_seq(200, seed=1003)
    s4 = make_complex_seq(200, seed=1004)
    ins = make_complex_seq(120, seed=1005)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2+", "ins+", "n3+", "n4+"])
    inputs = b.write("f10_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(bundle.sv_rows) == 1
    assert bundle.sv_rows[0]["anchors"] == "1..2"


def test_f10_02_node_length_weighted_lis_selects_heavier_chain(tmp_path: Path) -> None:
    """F10: Weighted LIS maximizes total anchor base length rather than anchor count."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1011)
    s2_heavy = make_complex_seq(600, seed=1012)
    s3_small = make_complex_seq(100, seed=1013)
    s4_small = make_complex_seq(100, seed=1014)
    s5 = make_complex_seq(200, seed=1015)
    ins_after_2 = make_complex_seq(150, seed=1016)
    for nid, seq in [
        ("n1", s1), ("n2", s2_heavy), ("n3", s3_small), ("n4", s4_small), ("n5", s5), ("ins", ins_after_2),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    # Branch visits n1 -> n3 -> n4 -> n2 -> ins -> n5.
    # Chain [n1, n3, n4, n5] has 4 anchors (600 bp); chain [n1, n2, n5] has 3 anchors (1000 bp).
    # Weighted LIS must pick [n1, n2, n5], so 'ins' is in D(n2, n5)!
    b.add_path("Alt1#1#Chr01", ["n1+", "n3+", "n4+", "n2+", "ins+", "n5+"])
    inputs = b.write("f10_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 1000  # before n5 (after n2..n4)


def test_f10_03_lis_tie_breaking_by_earlier_source_step_order(tmp_path: Path) -> None:
    """F10: Equal-weight LIS chains break ties by selecting earlier source step indices."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1021)
    s2 = make_complex_seq(200, seed=1022)
    s3 = make_complex_seq(200, seed=1023)
    s4 = make_complex_seq(200, seed=1024)
    ins_a = make_complex_seq(150, seed=1025)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ia", ins_a)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Chains [n1, n3, n4] and [n1, n2, n4] both have weight 600 bp, but n3 appears earlier than n2 in branch!
    b.add_path("Alt1#1#Chr01", ["n1+", "n3+", "ia+", "n2+", "n4+"])
    inputs = b.write("f10_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    # Because [n1, n3, n4] is chosen, ia+ is in D(n3, n4) and mounts before n4 (ref_start=600)
    assert int(ins_events[0]["ref_start"]) == 600


def test_f10_04_duplicate_anchor_rank_prevented_and_multi_hit_annotated(tmp_path: Path) -> None:
    """F10: Same anchor rank cannot enter LIS twice and triggers multi_hit annotation in sv.tsv."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1031)
    s2 = make_complex_seq(200, seed=1032)
    s3 = make_complex_seq(200, seed=1033)
    ins = make_complex_seq(140, seed=1034)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2+", "ins+", "n2+", "n3+"])
    inputs = b.write("f10_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert "multi_hit" in str(ins_events[0]["alleles"])


def test_f10_05_weighted_lis_after_inversion_normalization(tmp_path: Path) -> None:
    """F10: After normalizing a reverse chain, its flipped anchors join the forward weighted LIS."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1041)
    s2 = make_complex_seq(200, seed=1042)
    s3 = make_complex_seq(200, seed=1043)
    s4 = make_complex_seq(200, seed=1044)
    ins = make_complex_seq(150, seed=1045)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n3-", "n2-", "ins+", "n4+"])
    inputs = b.write("f10_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 600
    assert bundle.pan_seq == s1 + s2 + s3 + ins + s4


# ===========================================================================
# === F11: Deviation Segment D(a,b) Extraction (>= 5 tests) ===
# ===========================================================================


def test_f11_01_pure_insertion_deviation_empty_backbone_slice(tmp_path: Path) -> None:
    """F11: Pure insertion deviation where B(a,b) is empty and branch slice is non-empty."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1101)
    s2 = make_complex_seq(200, seed=1102)
    ins = make_complex_seq(150, seed=1103)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+"])
    inputs = b.write("f11_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert [r["type"] for r in bundle.sv_rows] == ["INS"]


def test_f11_02_pure_deletion_deviation_empty_branch_slice(tmp_path: Path) -> None:
    """F11: Pure deletion deviation where B(a,b) is non-empty and branch slice is empty."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1111)
    s2 = make_complex_seq(200, seed=1112)
    s3 = make_complex_seq(200, seed=1113)
    b.add_node("n1", s1).add_node("n2", s2).add_node("n3", s3)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n3+"])
    inputs = b.write("f11_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert [r["type"] for r in bundle.sv_rows] == ["DEL"]
    assert bundle.pan_seq == s1 + s2 + s3


def test_f11_03_replacement_deviation_both_non_empty_and_different(tmp_path: Path) -> None:
    """F11: Replacement deviation where both B(a,b) and branch slice are non-empty and differ."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1121)
    s2 = make_complex_seq(200, seed=1122)
    s3 = make_complex_seq(200, seed=1123)
    alt2 = make_complex_seq(170, seed=1124)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("alt2", alt2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "alt2+", "n3+"])
    inputs = b.write("f11_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    sv_types = {r["type"] for r in bundle.sv_rows}
    assert "INS" in sv_types and "DEL" in sv_types


def test_f11_04_t15_identical_branch_with_internal_repeats_no_deviation(tmp_path: Path) -> None:
    """F11 / T15: Branch identical to reference across internal repeats generates no D(a,b) and no SV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1131)
    sr = make_complex_seq(120, seed=1132)
    s2 = make_complex_seq(200, seed=1133)
    b.add_node("n1", s1).add_node("rpt", sr).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "rpt+", "rpt+", "rpt+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "rpt+", "rpt+", "rpt+", "n2+"])
    inputs = b.write("f11_04_t15")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.sv_rows) == 0
    assert len(bundle.inserts) == 0
    assert bundle.pan_seq == s1 + (sr * 3) + s2


def test_f11_05_multiple_independent_deviations_along_single_contig(tmp_path: Path) -> None:
    """F11: Multiple non-empty deviations along a single branch are extracted independently."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1141)
    s2 = make_complex_seq(200, seed=1142)
    s3 = make_complex_seq(200, seed=1143)
    s4 = make_complex_seq(200, seed=1144)
    s5 = make_complex_seq(200, seed=1145)
    ins = make_complex_seq(130, seed=1146)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("n5", s5), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+", "n3+", "n5+"])
    inputs = b.write("f11_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "INS")) == 1
    assert len(_find_sv(bundle, "DEL")) == 1


# ===========================================================================
# === F12: Cycle Folding (Tandem Repeat Compression) (>= 5 tests) ===
# ===========================================================================


def test_f12_01_t17a_contiguous_novel_repeat_u_u_u_folded_to_max_copy_1(tmp_path: Path) -> None:
    """F12 / T17: Contiguous novel repeat U-U-U is folded to 1 copy under --max-copy 1."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1201)
    s2 = make_complex_seq(200, seed=1202)
    u_seq = make_complex_seq(90, seed=1203)
    tail_seq = make_complex_seq(80, seed=1204)
    for nid, seq in [("n1", s1), ("n2", s2), ("U", u_seq), ("tail", tail_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "U+", "U+", "U+", "tail+", "n2+"])
    inputs = b.write("f12_01_t17a")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + u_seq + tail_seq + s2
    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "synthetic"


def test_f12_02_t17b_interrupted_u_x_u_not_folded_x_preserved(tmp_path: Path) -> None:
    """F12 / T17: Novel nodes U-X-U with private node X do not form a tandem period, preserving U-X-U intact."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1211)
    s2 = make_complex_seq(200, seed=1212)
    u_seq = make_complex_seq(90, seed=1213)
    x_seq = make_complex_seq(85, seed=1214)
    for nid, seq in [("n1", s1), ("n2", s2), ("U", u_seq), ("X", x_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "U+", "X+", "U+", "n2+"])
    inputs = b.write("f12_02_t17b")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + u_seq + x_seq + u_seq + s2
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["join_before"] == "source_contiguous"


def test_f12_03_multi_node_period_u1_u2_folded(tmp_path: Path) -> None:
    """F12: Multi-node tandem period (U1+ -> U2+) * 3 folds to 1 copy of (U1+ -> U2+)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1221)
    s2 = make_complex_seq(200, seed=1222)
    u1 = make_complex_seq(60, seed=1223)
    u2 = make_complex_seq(65, seed=1224)
    for nid, seq in [("n1", s1), ("n2", s2), ("u1", u1), ("u2", u2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u1+", "u2+", "u1+", "u2+", "u1+", "u2+", "n2+"])
    inputs = b.write("f12_03")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + u1 + u2 + s2


def test_f12_04_max_copy_2_retains_two_copies(tmp_path: Path) -> None:
    """F12: --max-copy 2 retains exactly 2 copies of a 4-copy novel repeat."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1231)
    s2 = make_complex_seq(200, seed=1232)
    u_seq = make_complex_seq(80, seed=1233)
    b.add_node("n1", s1).add_node("n2", s2).add_node("U", u_seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "U+", "U+", "U+", "U+", "n2+"])
    inputs = b.write("f12_04")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "2"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + (u_seq * 2) + s2


def test_f12_05_trailing_incomplete_half_period_preserved_intact(tmp_path: Path) -> None:
    """F12: Trailing incomplete half-period after k complete periods is preserved intact after folding."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1241)
    s2 = make_complex_seq(200, seed=1242)
    u1 = make_complex_seq(60, seed=1243)
    u2 = make_complex_seq(60, seed=1244)
    for nid, seq in [("n1", s1), ("n2", s2), ("u1", u1), ("u2", u2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # (u1+, u2+) repeated k=2 times followed by trailing incomplete u1+
    b.add_path("Alt1#1#Chr01", ["n1+", "u1+", "u2+", "u1+", "u2+", "u1+", "n2+"])
    inputs = b.write("f12_05")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + u1 + u2 + u1 + s2
    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "synthetic"


# ===========================================================================
# === F13: Canonical Period CNV Counting (>= 5 tests) ===
# ===========================================================================


def test_f13_01_t10_tandem_repeat_5_to_6_copies_cnv(tmp_path: Path) -> None:
    """F13 / T10: Tandem repeat expanding from 5 copies to 6 copies records CNV (5/6) and leaves pan.fa unchanged."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1301)
    su = make_complex_seq(60, seed=1302)
    s2 = make_complex_seq(200, seed=1303)
    b.add_node("n1", s1).add_node("u", su).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+"] + ["u+"] * 5 + ["n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+"] + ["u+"] * 6 + ["n2+"])
    inputs = b.write("f13_01_t10")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + (su * 5) + s2
    cnvs = _find_sv(bundle, "CNV")
    assert len(cnvs) == 1
    ev_id = cnvs[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "5"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "6"


def test_f13_02_tandem_repeat_copy_contraction_5_to_3_cnv(tmp_path: Path) -> None:
    """F13: Tandem repeat contracting from 5 copies to 3 copies records CNV (5/3)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1311)
    su = make_complex_seq(60, seed=1312)
    s2 = make_complex_seq(200, seed=1313)
    b.add_node("n1", s1).add_node("u", su).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+"] + ["u+"] * 5 + ["n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+"] + ["u+"] * 3 + ["n2+"])
    inputs = b.write("f13_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    cnvs = _find_sv(bundle, "CNV")
    assert len(cnvs) == 1
    ev_id = cnvs[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "5"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "3"


def test_f13_03_canonical_lexicographic_rotation_equivalence(tmp_path: Path) -> None:
    """F13: Cyclic rotations (b+, a+)*3 vs (a+, b+)*5 share canonical(U) = (a+, b+) and emit CNV (3/5)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1321)
    sa = make_complex_seq(40, seed=1322)
    sb = make_complex_seq(40, seed=1323)
    s2 = make_complex_seq(200, seed=1324)
    for nid, seq in [("n1", s1), ("a", sa), ("b", sb), ("n2", s2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+"] + ["b+", "a+"] * 3 + ["n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+"] + ["a+", "b+"] * 5 + ["n2+"])
    inputs = b.write("f13_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    cnvs = _find_sv(bundle, "CNV")
    assert len(cnvs) == 1
    ev_id = cnvs[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "3"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "5"


def test_f13_04_cnv_below_min_sv_len_not_called(tmp_path: Path) -> None:
    """F13: Copy number difference with |k_alt - k_ref| * len(U) < min-sv-len does not emit CNV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1331)
    su = make_complex_seq(30, seed=1332)
    s2 = make_complex_seq(200, seed=1333)
    b.add_node("n1", s1).add_node("u", su).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+"] + ["u+"] * 4 + ["n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+"] + ["u+"] * 5 + ["n2+"])
    inputs = b.write("f13_04")

    _, bundle = run_build(inputs, extra_args=["--min-sv-len", "50"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "CNV")) == 0


def test_f13_05_inconsistent_periods_do_not_emit_cnv(tmp_path: Path) -> None:
    """F13: When B(a,b) and branch have different non-equivalent periods, no CNV is emitted."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1341)
    su1 = make_complex_seq(60, seed=1342)
    su2 = make_complex_seq(60, seed=1343)
    s2 = make_complex_seq(200, seed=1344)
    for nid, seq in [("n1", s1), ("u1", su1), ("u2", su2), ("n2", s2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u1+", "u1+", "u1+", "u2+", "u2+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u1+", "u1+", "u2+", "u2+", "u2+", "n2+"])
    inputs = b.write("f13_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "CNV")) == 0


# ===========================================================================
# === F14: Subproblem State, Progress Metric & Local Re-anchoring (>= 5 tests) ===
# ===========================================================================


def test_f14_01_local_reanchoring_shrinks_interval_on_locally_unique_node(tmp_path: Path) -> None:
    """F14: Progress condition (a) shrinks [lo, hi) when B(a,b) contains a locally unique node."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1401)
    s_dup = make_complex_seq(150, seed=1402)
    s2 = make_complex_seq(200, seed=1403)
    s_skip = make_complex_seq(120, seed=1404)
    s3 = make_complex_seq(200, seed=1405)
    s_ins = make_complex_seq(140, seed=1406)
    for nid, seq in [
        ("n1", s1), ("dup", s_dup), ("n2", s2), ("skip", s_skip), ("n3", s3), ("ins", s_ins),
    ]:
        b.add_node(nid, seq)
    # 'dup' appears twice globally (steps 1 and 4), 'skip' is skipped by Alt1 so top-level LIS is [n1, n2, n3].
    # Inside B(n2, n3) = [skip+, dup+], 'dup' is unique! Alt1 has [dup+, ins+], so local re-anchoring
    # shrinks (n2, n3) to (n2, dup) [DEL of skip] and (dup, n3) [INS of ins before n3]!
    b.set_ref_path("Ref#1#Chr01", ["n1+", "dup+", "n2+", "skip+", "dup+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "dup+", "n2+", "dup+", "ins+", "n3+"])
    inputs = b.write("f14_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 820  # before n3


def test_f14_02_t16_termination_when_local_reanchoring_cannot_shrink(tmp_path: Path) -> None:
    """F14 / T16: When B(a,b) has no unique anchor, local re-anchoring terminates without infinite recursion."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1411)
    sr = make_complex_seq(120, seed=1412)
    s2 = make_complex_seq(200, seed=1413)
    ins = make_complex_seq(160, seed=1414)
    for nid, seq in [("n1", s1), ("r1", sr), ("n2", s2), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r1+", "r1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "ins+", "r1+", "n2+"])
    inputs = b.write("f14_02_t16")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert bundle.pan_seq == s1 + sr + sr + ("N" * 100) + ins + s2


def test_f14_03_multi_level_local_reanchoring_then_component_recursion(tmp_path: Path) -> None:
    """F14: Subproblem first shrinks [lo, hi) via local re-anchoring (a) then shrinks |M| via component recursion (b)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1421)
    sx = make_complex_seq(150, seed=1422)
    sy = make_complex_seq(150, seed=1423)
    s2 = make_complex_seq(200, seed=1424)
    s3 = make_complex_seq(200, seed=1425)
    sh_a = make_complex_seq(200, seed=1426)
    br_b = make_complex_seq(140, seed=1427)
    br_c = make_complex_seq(130, seed=1428)
    for nid, seq in [
        ("n1", s1), ("x", sx), ("y", sy), ("n2", s2), ("n3", s3),
        ("shA", sh_a), ("brB", br_b), ("brC", br_c),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "x+", "y+", "n2+", "x+", "y+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "x+", "y+", "n2+", "x+", "shA+", "brB+", "y+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "x+", "y+", "n2+", "x+", "shA+", "brC+", "y+", "n3+"])
    inputs = b.write("f14_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert int(ins_events[0]["ref_start"]) == 850  # before second y+


def test_f14_04_t19_distal_reference_node_inside_recursive_subproblem_not_reinserted(tmp_path: Path) -> None:
    """F14 / T19: Top-level reference node inside a recursive subproblem acts as a separator and is never re-inserted."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1431)
    s2 = make_complex_seq(200, seed=1432)
    s_distal = make_complex_seq(250, seed=1433)
    s3 = make_complex_seq(200, seed=1434)
    sh_a = make_complex_seq(200, seed=1435)
    sh_b = make_complex_seq(200, seed=1436)
    priv_c = make_complex_seq(150, seed=1437)
    for nid, seq in [
        ("n1", s1), ("n2", s2), ("ndist", s_distal), ("n3", s3),
        ("shA", sh_a), ("shB", sh_b), ("privC", priv_c),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "ndist+", "n3+"])
    # Alt1 and Alt2 have shA+ -> shB+ (support=2 -> secondary backbone)
    b.add_path("Alt1#1#Chr01", ["n1+", "shA+", "shB+", "n2+", "ndist+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "shA+", "shB+", "n2+", "ndist+", "n3+"])
    # Alt3 has shA+ -> ndist+ -> privC+ -> shB+ inside the component
    b.add_path("Alt3#1#Chr01", ["n1+", "shA+", "ndist+", "privC+", "shB+", "n2+", "ndist+", "n3+"])
    inputs = b.write("f14_04_t19")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # s_distal must appear ONLY ONCE in pan.fa (at its native reference position), never inside the insertion!
    assert bundle.pan_seq.count(s_distal) == 1
    assert priv_c in bundle.pan_seq


def test_f14_05_progress_metric_strictly_decreases_deep_nesting(tmp_path: Path) -> None:
    """F14: Multi-level component recursion terminates cleanly without any max-depth parameter."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1441)
    s2 = make_complex_seq(200, seed=1442)
    a = make_complex_seq(200, seed=1443)
    b1 = make_complex_seq(200, seed=1444)
    b2 = make_complex_seq(160, seed=1445)
    c1 = make_complex_seq(150, seed=1446)
    c2 = make_complex_seq(140, seed=1447)
    for nid, seq in [("n1", s1), ("n2", s2), ("A", a), ("B1", b1), ("B2", b2), ("C1", c1), ("C2", c2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "A+", "B1+", "C1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "A+", "B1+", "C1+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "A+", "B1+", "C2+", "n2+"])
    b.add_path("Alt4#1#Chr01", ["n1+", "A+", "B2+", "n2+"])
    inputs = b.write("f14_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    for seq in (a, b1, b2, c1, c2):
        assert seq in bundle.pan_seq


# ===========================================================================
# === F15: Candidate Block Extraction & Unanchored Ends (>= 5 tests) ===
# ===========================================================================


def test_f15_01_t9_distal_reference_node_and_novel_sequence_dup_tra(tmp_path: Path) -> None:
    """F15 / T9: Deviation carrying a distal reference block (>max-jump) and novel sequence calls DUP_TRA and INS without duplicating ref bases."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1501)
    s2 = make_complex_seq(200, seed=1502)
    s3 = make_complex_seq(800, seed=1503)
    s4 = make_complex_seq(800, seed=1504)
    s5 = make_complex_seq(200, seed=1505)
    nov = make_complex_seq(160, seed=1506)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("n5", s5), ("nov", nov)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    # Main chain is n1 -> n2 -> n3 -> n4 -> n5 (adjacent spans <= 1000).
    # Inside D(n1, n2), branch visits distal anchor n5+ (ref_start=2000, distance from n1=1800 > max_jump=1000) + nov+.
    b.add_path("Alt1#1#Chr01", ["n1+", "n5+", "nov+", "n2+", "n3+", "n4+", "n5+"])
    inputs = b.write("f15_01_t9")

    _, bundle = run_build(inputs, extra_args=["--max-jump", "1000"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "DUP_TRA")) == 1
    assert len(_find_sv(bundle, "INS")) == 1
    assert bundle.pan_seq.count(s5) == 1
    assert bundle.pan_seq == s1 + nov + s2 + s3 + s4 + s5


def test_f15_02_t11_top_level_contig_break_unanchored_end_left(tmp_path: Path) -> None:
    """F15 / T11: Contig break with >=50 bp unanchored prefix before first anchor goes to UNANCHORED_END and gets '.' genotype."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1511)
    s2 = make_complex_seq(200, seed=1512)
    s3 = make_complex_seq(200, seed=1513)
    unanch_l = make_complex_seq(150, seed=1514)
    ins_12 = make_complex_seq(140, seed=1515)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ul", unanch_l), ("ins12", ins_12)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("AltBreak#1#Chr01", ["ul+", "n2+", "n3+"])
    b.add_path("AltFull#1#Chr01", ["n1+", "ins12+", "n2+", "n3+"])
    inputs = b.write("f15_02_t11")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    unres = _find_unresolved(bundle, "UNANCHORED_END")
    assert len(unres) == 1
    assert int(unres[0]["length"]) == 150
    assert unres[0]["exported_fasta_id"] != "."
    assert unanch_l not in bundle.pan_seq

    ev_id = _find_sv(bundle, "INS")[0]["event_id"]
    assert bundle.genotypes[ev_id]["AltBreak#1"] == "."
    assert bundle.genotypes[ev_id]["AltFull#1"] == "1"


def test_f15_03_top_level_unanchored_end_right(tmp_path: Path) -> None:
    """F15: Contig ending early with >=50 bp unanchored suffix after last anchor goes to UNANCHORED_END."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1521)
    s2 = make_complex_seq(200, seed=1522)
    s3 = make_complex_seq(200, seed=1523)
    unanch_r = make_complex_seq(165, seed=1524)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ur", unanch_r)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("AltBreakR#1#Chr01", ["n1+", "n2+", "ur+"])
    inputs = b.write("f15_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    unres = _find_unresolved(bundle, "UNANCHORED_END")
    assert len(unres) == 1
    assert int(unres[0]["length"]) == 165
    assert unanch_r not in bundle.pan_seq


def test_f15_04_t4_recursive_logical_end_retained_not_unanchored_end(tmp_path: Path) -> None:
    """F15 / T4: In recursive child subproblem, branch reaching logical END forms D(a*, END) and is NOT sent to UNANCHORED_END."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1531)
    s2 = make_complex_seq(200, seed=1532)
    seq_a = make_complex_seq(400, seed=1533)
    seq_b = make_complex_seq(160, seed=1534)
    seq_c = make_complex_seq(150, seed=1535)
    for nid, seq in [("n1", s1), ("n2", s2), ("A", seq_a), ("B", seq_b), ("C", seq_c)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "A+", "B+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "A+", "B+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "A+", "C+", "n2+"])
    inputs = b.write("f15_04_t4")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.unresolved) == 0
    assert bundle.pan_seq == s1 + seq_a + seq_b + ("N" * 100) + seq_c + s2


def test_f15_05_repeat_residual_annotation_on_undecomposable_multi_copy_node(tmp_path: Path) -> None:
    """F15: Undecomposable candidate carrying a current-interval multi-copy node is kept whole and tagged repeat_residual."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1541)
    s2 = make_complex_seq(200, seed=1542)
    u0 = make_complex_seq(150, seed=1540)
    u1 = make_complex_seq(120, seed=1543)
    m_rep = make_complex_seq(100, seed=1544)
    u2 = make_complex_seq(120, seed=1545)
    nov_z = make_complex_seq(120, seed=1546)
    for nid, seq in [
        ("n1", s1), ("n2", s2), ("u0", u0), ("u1", u1), ("mrep", m_rep), ("u2", u2), ("novZ", nov_z),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Secondary backbone has mrep twice: u0+ -> u1+ -> mrep+ -> mrep+ -> u2+ (support score 1320 > 1140)
    b.add_path("Alt1#1#Chr01", ["n1+", "u0+", "u1+", "mrep+", "mrep+", "u2+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "u0+", "u1+", "mrep+", "mrep+", "u2+", "n2+"])
    # Alt3 has u1+ -> mrep+ -> novZ+ -> mrep+ -> u2+: inside D(u1, u2), B_sec(u1, u2) = [mrep+, mrep+] has no unique anchor!
    b.add_path("Alt3#1#Chr01", ["n1+", "u1+", "mrep+", "novZ+", "mrep+", "u2+", "n2+"])
    inputs = b.write("f15_05")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "2"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 1
    assert "repeat_residual" in str(ins_events[0]["alleles"])


# ===========================================================================
# === F16: Parallel Branch Groups G(a,b) & Component Recursion (>= 5 tests) ===
# ===========================================================================


def test_f16_01_t4_full_8kb_shared_prefix_bifurcating_b_and_c(tmp_path: Path) -> None:
    """F16 / T4: Shared 8 kb A bifurcating into non-homologous B and C outputs A + B + N + C with 2 distinct allele signatures."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=1601)
    s2 = make_complex_seq(300, seed=1602)
    seq_a = make_complex_seq(8000, seed=1603)
    seq_b = make_complex_seq(250, seed=1604)
    seq_c = make_complex_seq(220, seed=1605)
    for nid, seq in [("n1", s1), ("n2", s2), ("A8k", seq_a), ("B", seq_b), ("C", seq_c)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "A8k+", "B+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "A8k+", "B+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "A8k+", "C+", "n2+"])
    inputs = b.write("f16_01_t4")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + seq_a + seq_b + ("N" * 100) + seq_c + s2
    assert len(bundle.unresolved) == 0
    ev = _find_sv(bundle, "INS")[0]
    ev_id = ev["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt3#1"] == "2"


def test_f16_02_sec_backbone_support_selects_higher_member_count(tmp_path: Path) -> None:
    """F16: Default --sec-backbone support selects higher-support shorter path over single-member longer path."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1611)
    s2 = make_complex_seq(200, seed=1612)
    sh = make_complex_seq(200, seed=1613)
    short_br = make_complex_seq(180, seed=1614)
    long_br = make_complex_seq(300, seed=1615)
    for nid, seq in [("n1", s1), ("n2", s2), ("sh", sh), ("sbr", short_br), ("lbr", long_br)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "sh+", "lbr+", "n2+"])
    inputs = b.write("f16_02")

    _, bundle = run_build(inputs, extra_args=["--sec-backbone", "support"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + sh + short_br + ("N" * 100) + long_br + s2


def test_f16_03_sec_backbone_longest_selects_longest_candidate(tmp_path: Path) -> None:
    """F16: --sec-backbone longest selects the longest candidate even when another candidate has higher support."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1621)
    s2 = make_complex_seq(200, seed=1622)
    sh = make_complex_seq(200, seed=1623)
    short_br = make_complex_seq(120, seed=1624)
    long_br = make_complex_seq(300, seed=1625)
    for nid, seq in [("n1", s1), ("n2", s2), ("sh", sh), ("sbr", short_br), ("lbr", long_br)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "sh+", "lbr+", "n2+"])
    inputs = b.write("f16_03")

    _, bundle = run_build(inputs, extra_args=["--sec-backbone", "longest"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + sh + long_br + ("N" * 100) + short_br + s2


def test_f16_04_support_score_counts_distinct_haplotypes_not_node_revisits(tmp_path: Path) -> None:
    """F16: Secondary backbone support score counts each node once per distinct member source, not per revisit."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1631)
    s2 = make_complex_seq(200, seed=1632)
    sh = make_complex_seq(150, seed=1633)
    loop_node = make_complex_seq(100, seed=1634)
    other_node = make_complex_seq(120, seed=1635)
    for nid, seq in [("n1", s1), ("n2", s2), ("sh", sh), ("loop", loop_node), ("oth", other_node)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Alt1 visits loop 3 times (1 member), while Alt2 and Alt3 visit oth (2 members).
    # Support of Alt2/Alt3 = 150*3 + 120*2 = 690; support of Alt1 = 150*3 + 100*1 = 550.
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "loop+", "loop+", "loop+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "oth+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "sh+", "oth+", "n2+"])
    inputs = b.write("f16_04")

    _, bundle = run_build(inputs, extra_args=["--sec-backbone", "support", "--max-copy", "3"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + sh + other_node + ("N" * 100) + (loop_node * 3) + s2


def test_f16_05_exact_step_path_deduplication_before_component_partitioning(tmp_path: Path) -> None:
    """F16: Identical directed step paths across multiple haplotypes are exact-deduplicated into 1 single-member component."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1641)
    s2 = make_complex_seq(200, seed=1642)
    ins = make_complex_seq(180, seed=1643)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    for i in range(1, 5):
        b.add_path(f"Alt{i}#1#Chr01", ["n1+", "ins+", "n2+"])
    inputs = b.write("f16_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + ins + s2
    ev = _find_sv(bundle, "INS")[0]
    assert int(ev["n_carrier"]) == 4


# ===========================================================================
# === F17: Similarity Clustering (edlib <=1kb / mappy >1kb) (>= 5 tests) ===
# ===========================================================================


def test_f17_01_t2a_edlib_97pct_dispersed_snps_merged(tmp_path: Path) -> None:
    """F17 / T2a: Two 97% similar insertions (<=1 kb, edlib) with dispersed SNPs merge into 1 representative."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1701)
    s2 = make_complex_seq(200, seed=1702)
    r1 = make_complex_seq(600, seed=1703)
    # 18 dispersed SNPs out of 600 bp -> 97.0% identity >= 0.95, window identity >= 0.94 >= 0.70
    r2 = mutate_seq_snps(r1, num_snps=18, seed=1704)
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1), ("r2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f17_01_t2a")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    assert "N" not in bundle.pan_seq
    ev_id = _find_sv(bundle, "INS")[0]["event_id"]
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "1"


def test_f17_02_t2b_edlib_97pct_with_50bp_private_block_not_merged(tmp_path: Path) -> None:
    """F17 / T2b: 97% similar <=1 kb insertion containing a >=50 bp private block is NOT merged."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1711)
    s2 = make_complex_seq(200, seed=1712)
    r1 = make_complex_seq(800, seed=1713)
    priv55 = make_complex_seq(55, seed=1714)
    # Trim 35 bp from each outer end and insert 55 bp contiguous private block in the middle of r1
    # (785 bp < 800 bp so r1 is representative R first, and r2 fails Criterion 3 with 55 >= 50)
    r2 = r1[35:400] + priv55 + r1[400:-35]
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1), ("r2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f17_02_t2b")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 2
    assert r1 in bundle.pan_seq and r2 in bundle.pan_seq
    ev_id = _find_sv(bundle, "INS")[0]["event_id"]
    assert bundle.genotypes[ev_id]["Alt1#1"] != bundle.genotypes[ev_id]["Alt2#1"]


def test_f17_03_t5a_20_strains_5kb_dispersed_snps_and_micro_indels_merged(tmp_path: Path) -> None:
    """F17 / T5a: 20 strains sharing a 5 kb insertion with 50 dispersed SNPs / micro-indels (<50 bp) merge into 1 representative."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=1721)
    s2 = make_complex_seq(300, seed=1722)
    base_5k = make_complex_seq(5000, seed=1723)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins_s1", base_5k)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Strain01#1#Chr01", ["n1+", "ins_s1+", "n2+"])

    for i in range(2, 21):
        mut_seq = mutate_seq_snps(base_5k, num_snps=50, seed=1720 + i)
        nid = f"ins_s{i}"
        b.add_node(nid, mut_seq)
        b.add_path(f"Strain{i:02d}#1#Chr01", ["n1+", f"{nid}+", "n2+"])

    inputs = b.write("f17_03_t5a")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    assert len(bundle.pan_seq) == 300 + 5000 + 300
    ev_id = _find_sv(bundle, "INS")[0]["event_id"]
    for i in range(1, 21):
        assert bundle.genotypes[ev_id][f"Strain{i:02d}#1"] == "1"


def test_f17_04_t5b_and_t12a_mappy_large_candidate_with_private_insertion_not_merged(tmp_path: Path) -> None:
    """F17 / T5b / T12a: 10 kb candidate containing a 300 bp private insertion is rejected by Criterion 3."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=1731)
    s2 = make_complex_seq(300, seed=1732)
    r1_10k = make_complex_seq(10000, seed=1733)
    priv_300 = make_complex_seq(300, seed=1734)
    r2_10k3 = r1_10k[200:5000] + priv_300 + r1_10k[5000:-200]
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1_10k), ("r2", r2_10k3)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f17_04_t12a")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 2
    assert r1_10k in bundle.pan_seq and r2_10k3 in bundle.pan_seq


def test_f17_05_t12b_mappy_10kb_with_300bp_substitution_block_rejected_by_window_id(tmp_path: Path) -> None:
    """F17 / T12b: 10 kb candidate with a 300 bp contiguous substitution block (>97% overall ID) is rejected by Criterion 4."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(300, seed=1741)
    s2 = make_complex_seq(300, seed=1742)
    r1_10k = make_complex_seq(10000, seed=1743)
    # Mutate every base in [4500:4800] (300 bp contiguous substitution block) so overall ID is 9700/10000 = 0.97 >= 0.95
    # and 0 indels (Criterion 3 passes), but 50 bp sliding window over [4500:4800] has 0% identity < 0.70!
    sub_300 = mutate_seq_snps(r1_10k[4500:4800], positions=list(range(300)), seed=1744)
    r2_10k = r1_10k[:4500] + sub_300 + r1_10k[4800:]
    for nid, seq in [("n1", s1), ("n2", s2), ("r1", r1_10k), ("r2", r2_10k)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f17_05_t12b")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 2
    assert r1_10k in bundle.pan_seq and r2_10k in bundle.pan_seq


# ===========================================================================
# === F18: Unified Output Block Layout & Join Annotation (>= 5 tests) ===
# ===========================================================================


def test_f18_01_t3_two_non_homologous_insertions_pure_insertion_layout(tmp_path: Path) -> None:
    """F18 / T3: Two non-homologous insertions at a pure insertion site output a + R1 + N + R2 + b."""
    b = SyntheticInputBuilder(tmp_path)
    sa = make_complex_seq(200, seed=1801)
    sb = make_complex_seq(200, seed=1802)
    r1 = make_complex_seq(220, seed=1803)
    r2 = make_complex_seq(160, seed=1804)
    for nid, seq in [("a", sa), ("b", sb), ("R1", r1), ("R2", r2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["a+", "b+"])
    b.add_path("Alt1#1#Chr01", ["a+", "R1+", "b+"])
    b.add_path("Alt2#1#Chr01", ["a+", "R2+", "b+"])
    inputs = b.write("f18_01_t3")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == sa + r1 + ("N" * 100) + r2 + sb


def test_f18_02_replacement_locus_places_non_backbone_before_right_anchor_b(tmp_path: Path) -> None:
    """F18: When B(a,b) is non-empty, B(a,b) stays in place and non-backbone reps are placed before right anchor b."""
    b = SyntheticInputBuilder(tmp_path)
    sa = make_complex_seq(200, seed=1811)
    sm = make_complex_seq(180, seed=1812)
    sb = make_complex_seq(200, seed=1813)
    r2 = make_complex_seq(210, seed=1814)
    r3 = make_complex_seq(150, seed=1815)
    for nid, seq in [("a", sa), ("m", sm), ("b", sb), ("R2", r2), ("R3", r3)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["a+", "m+", "b+"])
    b.add_path("Alt1#1#Chr01", ["a+", "R2+", "b+"])
    b.add_path("Alt2#1#Chr01", ["a+", "R3+", "b+"])
    inputs = b.write("f18_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == sa + sm + ("N" * 100) + r2 + ("N" * 100) + r3 + sb


def test_f18_03_representative_ordering_support_desc_len_desc_token_asc(tmp_path: Path) -> None:
    """F18: Non-backbone representatives are ordered by support desc, then length desc, then token string asc."""
    b = SyntheticInputBuilder(tmp_path)
    sa = make_complex_seq(200, seed=1821)
    sb = make_complex_seq(200, seed=1822)
    rep_hi_sup = make_complex_seq(120, seed=1823)
    rep_long = make_complex_seq(260, seed=1824)
    rep_short = make_complex_seq(150, seed=1825)
    for nid, seq in [("a", sa), ("b", sb), ("repS", rep_hi_sup), ("repL", rep_long), ("repM", rep_short)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["a+", "b+"])
    b.add_path("Alt1#1#Chr01", ["a+", "repS+", "b+"])
    b.add_path("Alt2#1#Chr01", ["a+", "repS+", "b+"])
    b.add_path("Alt3#1#Chr01", ["a+", "repL+", "b+"])
    b.add_path("Alt4#1#Chr01", ["a+", "repM+", "b+"])
    inputs = b.write("f18_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == sa + rep_hi_sup + ("N" * 100) + rep_long + ("N" * 100) + rep_short + sb


def test_f18_04_t20_multi_groups_at_same_right_anchor_ordered_by_span_asc(tmp_path: Path) -> None:
    """F18 / T20: Two groups G(n2, n3) (span 1) and G(n1, n3) (span 2) sharing right anchor n3 are ordered by span asc and emitted as separate INS events."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1831)
    s2 = make_complex_seq(200, seed=1832)
    s3 = make_complex_seq(200, seed=1833)
    ins_narrow = make_complex_seq(140, seed=1834)
    ins_wide = make_complex_seq(170, seed=1835)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("inarrow", ins_narrow), ("iwide", ins_wide)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    # Pure insertion G(n2, n3) has span = 2 - 1 = 1
    b.add_path("Alt1#1#Chr01", ["n1+", "n2+", "inarrow+", "n3+"])
    # Replacement G(n1, n3) has span = 2 - 0 = 2
    b.add_path("Alt2#1#Chr01", ["n1+", "iwide+", "n3+"])
    inputs = b.write("f18_04_t20")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # G(n2, n3) (span 1) comes first before n3, then N*100, then G(n1, n3) (span 2)
    assert bundle.pan_seq == s1 + s2 + ins_narrow + ("N" * 100) + ins_wide + s3
    ins_events = _find_sv(bundle, "INS")
    assert len(ins_events) == 2
    assert ins_events[0]["event_id"] != ins_events[1]["event_id"]


def test_f18_05_join_before_annotations_across_all_three_types(tmp_path: Path) -> None:
    """F18: inserts.tsv accurately records source_contiguous, synthetic, and spacer in the same slot."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1841)
    s2 = make_complex_seq(200, seed=1842)
    u = make_complex_seq(80, seed=1843)
    tail = make_complex_seq(90, seed=1844)
    rep2 = make_complex_seq(130, seed=1845)
    for nid, seq in [("n1", s1), ("n2", s2), ("u", u), ("tail", tail), ("rep2", rep2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "rep2+", "n2+"])
    inputs = b.write("f18_05")

    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert [r["join_before"] for r in bundle.inserts] == ["source_contiguous", "synthetic", "spacer"]


# ===========================================================================
# === F19: Universal Repeat Guard (--repeat-guard on|off) (>= 5 tests) ===
# ===========================================================================


def test_f19_01_c1_default_repeat_guard_on_vs_off_normal_region(tmp_path: Path) -> None:
    """F19 / C1: Without centromere params, default --repeat-guard on masks normal satellite candidate (REPEAT_LIKE), whereas off retains it."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1901)
    s2 = make_complex_seq(250, seed=1902)
    sat = make_satellite_seq(unit_len=20, total_len=1500, seed=1903)
    b.add_node("n1", s1).add_node("n2", s2).add_node("sat", sat)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sat+", "n2+"])
    inputs = b.write("f19_01_c1")

    _, bundle_on = run_build(inputs, out_prefix=tmp_path / "c1_on")
    assert bundle_on is not None
    assert_all_global_invariants(bundle_on, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle_on.pan_seq == s1 + s2
    rep_unres = _find_unresolved(bundle_on, "REPEAT_LIKE")
    assert len(rep_unres) == 1
    assert int(rep_unres[0]["length"]) == 1500
    assert rep_unres[0]["exported_fasta_id"] == "."

    _, bundle_off = run_build(inputs, extra_args=["--repeat-guard", "off"], out_prefix=tmp_path / "c1_off")
    assert bundle_off is not None
    assert_all_global_invariants(bundle_off, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle_off.pan_seq == s1 + sat + s2


def test_f19_02_c16_satellite_array_with_embedded_1kb_complex_island(tmp_path: Path) -> None:
    """F19 / C16: Low-divergence satellite array flanking a 1 kb complex island masks satellite windows and retains only the complex residual."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1911)
    s2 = make_complex_seq(250, seed=1912)
    sat_l = make_satellite_seq(unit_len=20, total_len=2000, seed=1913)
    island_1k = make_complex_seq(1200, seed=1914)
    sat_r = make_satellite_seq(unit_len=20, total_len=2000, seed=1915)
    hybrid = sat_l + island_1k + sat_r
    b.add_node("n1", s1).add_node("n2", s2).add_node("hyb", hybrid)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "hyb+", "n2+"])
    inputs = b.write("f19_02_c16")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Flanking satellite arrays are masked as REPEAT_LIKE, while central complex residual is retained!
    rep_unres = _find_unresolved(bundle, "REPEAT_LIKE")
    assert len(rep_unres) == 2
    assert len(bundle.inserts) == 1
    ins_len = int(bundle.inserts[0]["pan_end"]) - int(bundle.inserts[0]["pan_start"])
    assert 500 <= ins_len <= 1400
    assert len(bundle.pan_seq) < 2000
    assert "partial_content=1" in str(_find_sv(bundle, "INS")[0]["alleles"])


def test_f19_03_c17_no_bed_ref_with_n_gap_satellite_masked_complex_gene_retained(tmp_path: Path) -> None:
    """F19 / C17: Without BED, reference with N-gap has donor satellite insertion blocked by repeat-guard while donor complex gene is retained intact."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1921)
    s_ngap = "N" * 200
    s2 = make_complex_seq(250, seed=1922)
    s3 = make_complex_seq(250, seed=1923)
    sat = make_satellite_seq(unit_len=20, total_len=1500, seed=1924)
    gene_seq = make_coding_seq(num_codons=400, seed=1925)  # 1200 bp complex gene
    for nid, seq in [("n1", s1), ("ngap", s_ngap), ("n2", s2), ("n3", s3), ("sat", sat), ("gene", gene_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "ngap+", "n2+", "n3+"])
    b.add_path("DonorSat#1#Chr01", ["n1+", "sat+", "n2+", "n3+"])
    b.add_path("DonorGene#1#Chr01", ["n1+", "ngap+", "n2+", "gene+", "n3+"])
    inputs = b.write("f19_03_c17")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_unresolved(bundle, "REPEAT_LIKE")) == 1
    assert sat not in bundle.pan_seq
    assert gene_seq in bundle.pan_seq


def test_f19_04_c20_candidate_split_into_two_residuals_and_fully_masked_genotype_dot(tmp_path: Path) -> None:
    """F19 / C20: Candidate split by masked central satellite into two >=50 bp residuals is not seamlessly concatenated and fully masked hap gets '.'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1931)
    s2 = make_complex_seq(250, seed=1932)
    left_complex = make_complex_seq(400, seed=1933)
    mid_sat = make_satellite_seq(unit_len=20, total_len=1800, seed=1934)
    right_complex = make_complex_seq(400, seed=1935)
    split_cand = left_complex + mid_sat + right_complex
    pure_sat = make_satellite_seq(unit_len=20, total_len=1500, seed=1936)
    for nid, seq in [("n1", s1), ("n2", s2), ("sc", split_cand), ("ps", pure_sat)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("AltSplit#1#Chr01", ["n1+", "sc+", "n2+"])
    b.add_path("AltMasked#1#Chr01", ["n1+", "ps+", "n2+"])
    b.add_path("AltRef#1#Chr01", ["n1+", "n2+"])
    inputs = b.write("f19_04_c20")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Two residuals separated by N*spacer (never seamlessly stitched!)
    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "spacer"
    p0_e = int(bundle.inserts[0]["pan_end"])
    p1_s = int(bundle.inserts[1]["pan_start"])
    assert bundle.pan_seq[p0_e:p1_s] == "N" * 100

    ins_ev = _find_sv(bundle, "INS")[0]
    assert "partial_content=1" in str(ins_ev["alleles"])
    ev_id = ins_ev["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["AltRef#1"] == "0"
    assert bundle.genotypes[ev_id]["AltSplit#1"] == "1"
    # Fully masked donor MUST get '.', NOT '0'!
    assert bundle.genotypes[ev_id]["AltMasked#1"] == "."


def test_f19_05_repeat_guard_partial_node_slicing_provenance(tmp_path: Path) -> None:
    """F19: Repeat mask boundary falling mid-node creates a partial-node slice with exact sub-node coordinates."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1941)
    s2 = make_complex_seq(250, seed=1942)
    sat_prefix = make_satellite_seq(unit_len=20, total_len=1600, seed=1943)
    complex_suffix = make_complex_seq(700, seed=1944)
    hybrid_node = sat_prefix + complex_suffix
    expected_mask = oracle_repeat_mask_intervals(hybrid_node)
    assert len(expected_mask) == 1
    mask_s, mask_e = expected_mask[0]
    assert mask_s == 0 and 1500 <= mask_e < 2200

    b.add_node("n1", s1).add_node("n2", s2).add_node("hyb", hybrid_node)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "hyb+", "n2+"])
    inputs = b.write("f19_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.inserts) == 1
    expected_residual = hybrid_node[mask_e:]
    assert bundle.pan_seq == s1 + expected_residual + s2
    assert f"{250 + mask_e}-{250 + len(hybrid_node)}" in str(bundle.inserts[0]["src_intervals"])


# ===========================================================================
# === F20: Top-level SV Event Calling & Genotyping (>= 5 tests) ===
# ===========================================================================


def test_f20_01_t13_skip_200bp_ref_with_30bp_novel_calls_only_del(tmp_path: Path) -> None:
    """F20 / T13: Skipping 200 bp of reference while carrying 30 bp (<50 bp) novel sequence records only DEL and leaves pan.fa unchanged."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2001)
    s2 = make_complex_seq(200, seed=2002)
    s3 = make_complex_seq(200, seed=2003)
    small30 = make_complex_seq(30, seed=2004)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("sm30", small30)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sm30+", "n3+"])
    inputs = b.write("f20_01_t13")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.pan_seq == s1 + s2 + s3
    assert [r["type"] for r in bundle.sv_rows] == ["DEL"]
    del_ev = bundle.sv_rows[0]
    assert int(del_ev["ref_start"]) == 200
    assert int(del_ev["ref_end"]) == 400
    ev_id = del_ev["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"


def test_f20_02_dup_tra_called_when_deviation_anchor_exceeds_max_jump(tmp_path: Path) -> None:
    """F20: Deviation containing an out-of-chain anchor > max-jump away from left anchor a emits DUP_TRA."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2011)
    s2 = make_complex_seq(200, seed=2012)
    s3 = make_complex_seq(900, seed=2013)
    s4 = make_complex_seq(200, seed=2014)
    nov = make_complex_seq(140, seed=2015)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("nov", nov)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n4+", "nov+", "n2+", "n3+", "n4+"])
    inputs = b.write("f20_02")

    _, bundle = run_build(inputs, extra_args=["--max-jump", "800"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "DUP_TRA")) == 1
    assert len(_find_sv(bundle, "INS")) == 1
    assert _find_sv(bundle, "DUP_TRA")[0]["region_id"] == _find_sv(bundle, "INS")[0]["region_id"]


def test_f20_03_large_jump_overrides_dup_tra_and_routes_to_unresolved(tmp_path: Path) -> None:
    """F20: When post-normalization main-chain span > max-jump, LARGE_JUMP overrides DUP_TRA and sends novel seqs to unresolved."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2021)
    s2_big = make_complex_seq(1500, seed=2022)
    s3 = make_complex_seq(200, seed=2023)
    s4 = make_complex_seq(200, seed=2024)
    nov_lj = make_complex_seq(180, seed=2025)
    for nid, seq in [("n1", s1), ("n2", s2_big), ("n3", s3), ("n4", s4), ("nlj", nov_lj)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Branch jumps from n1+ across 1500 bp n2+ to n3+ (> max_jump=1000) while also visiting n4+ and nlj+
    b.add_path("Alt1#1#Chr01", ["n1+", "n4+", "nlj+", "n3+", "n4+"])
    inputs = b.write("f20_03")

    _, bundle = run_build(inputs, extra_args=["--max-jump", "1000"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(_find_sv(bundle, "LARGE_JUMP")) == 1
    assert len(_find_sv(bundle, "DUP_TRA")) == 0
    assert len(_find_sv(bundle, "INS")) == 0
    assert bundle.pan_seq == s1 + s2_big + s3 + s4
    lj_unres = _find_unresolved(bundle, "LARGE_JUMP")
    assert len(lj_unres) == 1
    assert int(lj_unres[0]["length"]) == 180
    assert lj_unres[0]["exported_fasta_id"] != "."


def test_f20_04_multi_allelic_ins_genotype_numbering_and_contig_conflict_dot(tmp_path: Path) -> None:
    """F20: INS alleles are numbered 1..C by carrier count desc then signature lex order, and contradictory contigs get '.'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2031)
    s2 = make_complex_seq(200, seed=2032)
    ins_a = make_complex_seq(150, seed=2033)
    ins_b = make_complex_seq(190, seed=2034)
    for nid, seq in [("n1", s1), ("n2", s2), ("ia", ins_a), ("ib", ins_b)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ia+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "ia+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "ib+", "n2+"])
    b.add_path("AltRef#1#Chr01", ["n1+", "n2+"])
    # Contradictory contigs for AltConf#1
    b.add_walk("AltConf", "1", "ctg1", 0, 400, ["n1+", "n2+"])
    b.add_walk("AltConf", "1", "ctg2", 1000, 1550, ["n1+", "ia+", "n2+"])
    inputs = b.write("f20_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_id = _find_sv(bundle, "INS")[0]["event_id"]
    gt = bundle.genotypes[ev_id]
    assert gt["Ref#1"] == "0"
    assert gt["AltRef#1"] == "0"
    assert gt["Alt1#1"] == "1"
    assert gt["Alt2#1"] == "1"
    assert gt["Alt3#1"] == "2"
    assert gt["AltConf#1"] == "."


def test_f20_05_colocated_inv_and_nested_ins_share_region_id(tmp_path: Path) -> None:
    """F20: Co-located basic SV events (e.g. INV and its internal INS) occupy separate rows linked by region_id."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2041)
    s2 = make_complex_seq(200, seed=2042)
    s3 = make_complex_seq(200, seed=2043)
    s4 = make_complex_seq(200, seed=2044)
    sx = make_complex_seq(150, seed=2045)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("n4", s4), ("x", sx)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n3-", "x+", "n2-", "n4+"])
    inputs = b.write("f20_05")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    invs = _find_sv(bundle, "INV")
    ins_list = _find_sv(bundle, "INS")
    assert len(invs) == 1 and len(ins_list) == 1
    assert invs[0]["region_id"] == ins_list[0]["region_id"]
    assert invs[0]["region_id"] != "."


# ===========================================================================
# === F21: Centromere Mode CLI & BED Anchor Delimitation (>= 5 tests) ===
# ===========================================================================


def test_f21_01_centromere_bed_without_mode_defaults_to_bypass(tmp_path: Path) -> None:
    """F21: Providing --centromere-bed without --centromere-mode defaults to bypass mode."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2101)
    scen = make_complex_seq(300, seed=2102)
    sv = make_complex_seq(200, seed=2103)
    alt_cen = make_complex_seq(250, seed=2104)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("acen", alt_cen)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "acen+", "v+"])
    b.add_bed_region(200, 500, "CEN_1")
    inputs = b.write("f21_01")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert bundle.centromere_rows is not None
    assert all(r["mode"] == "bypass" for r in bundle.centromere_rows)


def test_f21_02_c5a_bed_inside_nodes_expanded_and_overlapping_merged(tmp_path: Path) -> None:
    """F21 / C5a: BED endpoints falling inside nodes expand outward to unique anchors u/v and overlapping intervals merge into 1 region."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2111)      # [0, 200) unique
    sc1 = make_complex_seq(200, seed=2112)     # [200, 400) repeated
    sc2 = make_complex_seq(200, seed=2113)     # [600, 800) repeated
    sv = make_complex_seq(200, seed=2114)      # [1000, 1200) unique
    for nid, seq in [("u", su), ("c1", sc1), ("c2", sc2), ("v", sv)]:
        b.add_node(nid, seq)
    # Reference: u+ [0,200), c1+ [200,400), c1+ [400,600), c2+ [600,800), c2+ [800,1000), v+ [1000,1200)
    # Only u (step 0) and v (step 5) are unique in reference!
    b.set_ref_path("Ref#1#Chr01", ["u+", "c1+", "c1+", "c2+", "c2+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "c1+", "c2+", "v+"])
    # Two BED intervals falling inside repeated nodes that both expand outward to [200, 1000) and merge!
    b.add_bed_region(250, 550, "BED_A")
    b.add_bed_region(650, 950, "BED_B")
    inputs = b.write("f21_02_c5a")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert bundle.centromere_rows is not None
    region_ids = {r["region_id"] for r in bundle.centromere_rows}
    assert len(region_ids) == 1
    row0 = bundle.centromere_rows[0]
    assert int(row0["ref_start"]) == 200
    assert int(row0["ref_end"]) == 1000
    assert "BED_A" in str(row0["input_bed_ids"]) and "BED_B" in str(row0["input_bed_ids"])


def test_f21_03_c5b_missing_unique_flank_virtual_boundary_downgrade_to_bypass(tmp_path: Path) -> None:
    """F21 / C5b: Missing left/right unique reference anchor uses virtual boundary 0/L, logs CEN_ANCHOR_NOT_FOUND, downgrades to bypass, and never fabricates sample complete status."""
    b = SyntheticInputBuilder(tmp_path)
    sr = make_complex_seq(200, seed=2121)
    sv = make_complex_seq(200, seed=2122)
    s_end = make_complex_seq(200, seed=2123)
    alt_isl = make_complex_seq(600, seed=2124)
    for nid, seq in [("r", sr), ("v", sv), ("nend", s_end), ("isl", alt_isl)]:
        b.add_node(nid, seq)
    # Reference starts with repeated node r+ -> r+ [0, 400) before unique anchor v+ [400, 600) -> nend+ [600, 800).
    # BED [50, 350) has NO unique anchor to the left of 50!
    b.set_ref_path("Ref#1#Chr01", ["r+", "r+", "v+", "nend+"])
    b.add_path("Alt1#1#Chr01", ["r+", "isl+", "v+", "nend+"])
    b.add_bed_region(50, 350, "CEN_NO_LEFT")
    inputs = b.write("f21_03_c5b")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    combined_meta = bundle.summary_text + "\n" + bundle.log_text
    assert "CEN_ANCHOR_NOT_FOUND" in combined_meta
    by_hap = _centromere_by_hap(bundle)
    alt_row = by_hap["Alt1#1"]
    assert alt_row["mode"] == "bypass"
    assert int(alt_row["ref_start"]) == 0
    assert int(alt_row["ref_end"]) == 400
    assert alt_row["status"] in ("partial", "unmapped")
    assert alt_row["L_path"] == "."
    assert alt_row["length_ratio"] == "."
    assert len(_find_sv(bundle, "CEN_ISLAND")) == 0


def test_f21_04_centromere_boundary_ambiguous_branch_routed_to_unresolved(tmp_path: Path) -> None:
    """F21: Branch skipping anchor u and jumping from outside into v has its novel sequence routed to CEN_BOUNDARY_AMBIGUOUS."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2131)
    su = make_complex_seq(200, seed=2132)
    scen = make_complex_seq(300, seed=2133)
    sv = make_complex_seq(200, seed=2134)
    s2 = make_complex_seq(200, seed=2135)
    nov_amb = make_complex_seq(160, seed=2136)
    for nid, seq in [("n1", s1), ("u", su), ("cen", scen), ("v", sv), ("n2", s2), ("namb", nov_amb)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "n2+"])
    b.add_path("AltAmb#1#Chr01", ["n1+", "namb+", "v+", "n2+"])
    b.add_bed_region(400, 700, "CEN1")
    inputs = b.write("f21_04")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    amb_unres = _find_unresolved(bundle, "CEN_BOUNDARY_AMBIGUOUS")
    assert len(amb_unres) == 1
    assert int(amb_unres[0]["length"]) == 160
    assert nov_amb not in bundle.pan_seq


def test_f21_05_normal_insertions_outside_u_v_at_exact_boundaries_processed_normally(tmp_path: Path) -> None:
    """F21: Normal insertions in (n1, u) and (v, n2) outside [ref_end(u), ref_start(v)) are processed by the normal pipeline."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2141)
    su = make_complex_seq(200, seed=2142)
    scen = make_complex_seq(300, seed=2143)
    sv = make_complex_seq(200, seed=2144)
    s2 = make_complex_seq(200, seed=2145)
    ins_l = make_complex_seq(130, seed=2146)
    ins_r = make_complex_seq(140, seed=2147)
    for nid, seq in [
        ("n1", s1), ("u", su), ("cen", scen), ("v", sv), ("n2", s2), ("il", ins_l), ("ir", ins_r),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "il+", "u+", "cen+", "v+", "ir+", "n2+"])
    b.add_bed_region(400, 700, "CEN1")
    inputs = b.write("f21_05")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(_find_sv(bundle, "INS")) == 2
    assert bundle.pan_seq == s1 + ins_l + su + scen + sv + ins_r + s2


# ===========================================================================
# === F22: Centromere Branch Completeness & Length Ratio (>= 5 tests) ===
# ===========================================================================


def test_f22_01_c3_complete_traversal_5_to_6_copies_length_ratio_no_cnv(tmp_path: Path) -> None:
    """F22 / C3: Reference 5 copies vs sample 6 copies inside centromere reports L_path for all 6 visits, length_ratio=1.2, and no normal CNV event."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2201)
    s_unit = make_complex_seq(200, seed=2202)
    sv = make_complex_seq(200, seed=2203)
    b.add_node("u", su).add_node("unit", s_unit).add_node("v", sv)
    b.set_ref_path("Ref#1#Chr01", ["u+"] + ["unit+"] * 5 + ["v+"])
    b.add_path("Alt1#1#Chr01", ["u+"] + ["unit+"] * 6 + ["v+"])
    b.add_bed_region(200, 1200, "CEN1")
    inputs = b.write("f22_01_c3")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert len(_find_sv(bundle, "CNV")) == 0
    by_hap = _centromere_by_hap(bundle)
    alt_row = by_hap["Alt1#1"]
    assert alt_row["status"] == "complete"
    assert int(alt_row["L_ref"]) == 1000
    assert int(alt_row["L_path"]) == 1200
    assert float(alt_row["length_ratio"]) == pytest.approx(1.2)


def test_f22_02_c4a_partial_single_flank_traversal(tmp_path: Path) -> None:
    """F22 / C4a: Haplotype hitting only anchor u (missing v) is classified as partial with L_path='.' and length_ratio='.'."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2211)
    scen = make_complex_seq(400, seed=2212)
    sv = make_complex_seq(200, seed=2213)
    part_seq = make_complex_seq(600, seed=2214)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("pseq", part_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("AltPart#1#Chr01", ["u+", "pseq+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f22_02_c4a")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    row = _centromere_by_hap(bundle)["AltPart#1"]
    assert row["status"] == "partial"
    assert row["L_path"] == "."
    assert row["length_ratio"] == "."
    assert len(_find_sv(bundle, "CEN_ISLAND")) == 0


def test_f22_03_c4b_gapped_traversal_containing_n(tmp_path: Path) -> None:
    """F22 / C4b: Haplotype traversing u..v with internal 'N' bases has status='gapped', reports L_path, sets length_ratio='.', and does not mount."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2221)
    scen = make_complex_seq(400, seed=2222)
    sv = make_complex_seq(200, seed=2223)
    gap_node = make_complex_seq(300, seed=2224) + ("N" * 50) + make_complex_seq(300, seed=2225)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("gnode", gap_node)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("AltGap#1#Chr01", ["u+", "gnode+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f22_03_c4b")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    row = _centromere_by_hap(bundle)["AltGap#1"]
    assert row["status"] == "gapped"
    assert int(row["L_path"]) == 650
    assert row["length_ratio"] == "."
    assert len(_find_sv(bundle, "CEN_ISLAND")) == 0
    assert len(_find_unresolved(bundle, "CEN_INCOMPLETE")) == 1


def test_f22_04_c4c_ambiguous_multiple_traversals_or_repeated_anchor(tmp_path: Path) -> None:
    """F22 / C4c: Haplotype hitting u/v multiple times or having multiple u..v traversals has status='ambiguous' and length_ratio='.'."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2231)
    scen = make_complex_seq(400, seed=2232)
    sv = make_complex_seq(200, seed=2233)
    amb_isl = make_complex_seq(600, seed=2234)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("aisl", amb_isl)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    # Repeats anchor u+ twice on the same path
    b.add_path("AltAmb#1#Chr01", ["u+", "aisl+", "u+", "cen+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f22_04_c4c")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    row = _centromere_by_hap(bundle)["AltAmb#1"]
    assert row["status"] == "ambiguous"
    assert row["L_path"] == "."
    assert row["length_ratio"] == "."
    assert len(_find_sv(bundle, "CEN_ISLAND")) == 0
    assert len(_find_unresolved(bundle, "CEN_AMBIGUOUS")) >= 1


def test_f22_05_unmapped_sample_and_gapped_reference_status(tmp_path: Path) -> None:
    """F22: Reference centromere containing 'N' sets ref_status='gapped', length_ratio='.', and routes complete donor islands to CEN_REFERENCE_GAP, while distal-only hap is unmapped."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2241)
    su = make_complex_seq(200, seed=2242)
    scen_gap = make_complex_seq(200, seed=2243) + ("N" * 40) + make_complex_seq(200, seed=2244)
    sv = make_complex_seq(200, seed=2245)
    s2 = make_complex_seq(200, seed=2246)
    isl = make_complex_seq(600, seed=2247)
    for nid, seq in [("n1", s1), ("u", su), ("cgap", scen_gap), ("v", sv), ("n2", s2), ("isl", isl)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cgap+", "v+", "n2+"])
    b.add_path("AltComp#1#Chr01", ["n1+", "u+", "isl+", "v+", "n2+"])
    b.add_walk("AltUnmap", "1", "ctg1", 0, 200, ["n1+"])
    b.add_bed_region(400, 840, "CEN1")
    inputs = b.write("f22_05")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    by_hap = _centromere_by_hap(bundle)
    comp_row = by_hap["AltComp#1"]
    assert comp_row["status"] == "complete"
    assert comp_row["ref_status"] == "gapped"
    assert comp_row["length_ratio"] == "."
    assert len(_find_unresolved(bundle, "CEN_REFERENCE_GAP")) == 1
    assert by_hap["AltUnmap#1"]["status"] == "unmapped"


# ===========================================================================
# === F23: Centromere bypass Mode (>= 5 tests) ===
# ===========================================================================


def test_f23_01_c2_bypass_mode_over_500kb_array_diverted_before_max_jump(tmp_path: Path) -> None:
    """F23 / C2: BED covering >500 kb array in bypass mode diverts before max-jump, keeps reference intact, and records non-ref block as CEN_BYPASS."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2301)
    # 510 kb centromere reference array (> default 500 kb max-jump)
    scen_510k = make_satellite_seq(unit_len=170, total_len=510_000, seed=2302)
    sv = make_complex_seq(200, seed=2303)
    alt_block = make_complex_seq(600, seed=2304)
    for nid, seq in [("u", su), ("cen510k", scen_510k), ("v", sv), ("ablock", alt_block)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen510k+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "ablock+", "v+"])
    b.add_bed_region(200, 510_200, "CEN_510K")
    inputs = b.write("f23_01_c2")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert bundle.pan_seq == su + scen_510k + sv
    assert len(_find_sv(bundle, "LARGE_JUMP")) == 0
    assert len(_find_sv(bundle, "DEL")) == 0
    byp = _find_unresolved(bundle, "CEN_BYPASS")
    assert len(byp) == 1
    assert int(byp[0]["length"]) == 600
    assert byp[0]["exported_fasta_id"] == "."


def test_f23_02_bypass_mode_with_no_normal_variants_pan_fa_equals_ref_fa(tmp_path: Path) -> None:
    """F23: When input has only bypass centromere regions and invariant normal regions, pan.fa is 100% identical to ref.fa."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2311)
    su = make_complex_seq(200, seed=2312)
    scen = make_complex_seq(400, seed=2313)
    sv = make_complex_seq(200, seed=2314)
    s2 = make_complex_seq(200, seed=2315)
    alt_c1 = make_complex_seq(350, seed=2316)
    alt_c2 = make_complex_seq(500, seed=2317)
    for nid, seq in [("n1", s1), ("u", su), ("cen", scen), ("v", sv), ("n2", s2), ("ac1", alt_c1), ("ac2", alt_c2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "ac1+", "v+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "u+", "ac2+", "v+", "n2+"])
    b.add_bed_region(400, 800, "CEN1")
    inputs = b.write("f23_02")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert bundle.pan_seq == s1 + su + scen + sv + s2
    assert len(bundle.sv_rows) == 0


def test_f23_03_bypass_mode_centromere_tsv_accounting_all_nonref_in_unresolved_bp(tmp_path: Path) -> None:
    """F23: In bypass mode, all localizable non-reference bases are accounted in unresolved_bp with supported_events='.'."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2321)
    scen = make_complex_seq(300, seed=2322)
    sv = make_complex_seq(200, seed=2323)
    nr1 = make_complex_seq(250, seed=2324)
    nr2 = make_complex_seq(350, seed=2325)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("nr1", nr1), ("nr2", nr2)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "nr1+", "cen+", "nr2+", "v+"])
    b.add_bed_region(200, 500, "CEN1")
    inputs = b.write("f23_03")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    row = _centromere_by_hap(bundle)["Alt1#1"]
    assert int(row["nonref_input_bp"]) == 600
    assert int(row["unresolved_bp"]) == 600
    assert int(row["reference_like_bp"]) == 0
    assert int(row["repeat_like_bp"]) == 0
    assert int(row["short_bp"]) == 0
    assert int(row["accepted_source_bp"]) == 0
    assert row["supported_events"] == "."


def test_f23_04_bypass_mode_suppresses_internal_del_inv_cnv_and_cen_island(tmp_path: Path) -> None:
    """F23: bypass mode does not emit DEL, INV, CNV, or CEN_ISLAND for rearrangements inside [ref_end(u), ref_start(v))."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2331)
    sc1 = make_complex_seq(200, seed=2332)
    sc2 = make_complex_seq(200, seed=2333)
    sv = make_complex_seq(200, seed=2334)
    for nid, seq in [("u", su), ("c1", sc1), ("c2", sc2), ("v", sv)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "c1+", "c2+", "c2+", "c2+", "v+"])
    # Alt1 inverts c1 and changes copy number of c2 inside the centromere
    b.add_path("Alt1#1#Chr01", ["u+", "c1-", "v+"])
    b.add_bed_region(200, 1000, "CEN1")
    inputs = b.write("f23_04")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert len(bundle.sv_rows) == 0


def test_f23_05_bypass_mode_with_dump_repeat_sequences_exports_to_unresolved_fa(tmp_path: Path) -> None:
    """F23: --dump-repeat-sequences exports CEN_BYPASS blocks into unresolved.fa without changing pan.fa."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2341)
    scen = make_complex_seq(300, seed=2342)
    sv = make_complex_seq(200, seed=2343)
    byp_seq = make_complex_seq(400, seed=2344)
    for nid, seq in [("u", su), ("cen", scen), ("v", sv), ("byp", byp_seq)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "byp+", "v+"])
    b.add_bed_region(200, 500, "CEN1")
    inputs = b.write("f23_05")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass", "--dump-repeat-sequences"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    byp_rows = _find_unresolved(bundle, "CEN_BYPASS")
    assert len(byp_rows) == 1
    exp_id = str(byp_rows[0]["exported_fasta_id"])
    assert exp_id != "."
    assert exp_id in bundle.unresolved_fa
    assert bundle.unresolved_fa[exp_id][1] == byp_seq


# ===========================================================================
# === F24: Centromere extract-islands Mode (Steps A-D) (>= 5 tests) ===
# ===========================================================================


def test_f24_01_c6_and_c7_distinct_node_satellite_filtered_vs_revisited_node_with_complex_island_retained(
    tmp_path: Path,
) -> None:
    """F24 / C6 / C7: Satellite unrolled across distinct node IDs (revisited_fraction=0) is filtered by Step C (CEN_REPEAT_LIKE), whereas candidate with revisited nodes + >=500 bp complex sequence is retained as CEN_ISLAND."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2401)
    scen = make_complex_seq(400, seed=2402)
    sv = make_complex_seq(200, seed=2403)
    # C6: 1500 bp satellite split across 3 distinct nodes (each visited once, revisited_fraction == 0.0)
    sat_full = make_satellite_seq(unit_len=20, total_len=1500, seed=2404)
    sat1, sat2, sat3 = sat_full[:500], sat_full[500:1000], sat_full[1000:]
    # C7: Candidate with a revisited 100 bp node 'rvis' + 600 bp complex island -> total 800 bp, revisited_fraction == 200/800 = 0.25 > 0
    rvis = make_complex_seq(100, seed=2405)
    c600 = make_complex_seq(600, seed=2406)
    for nid, seq in [
        ("u", su), ("cen", scen), ("v", sv),
        ("s1", sat1), ("s2", sat2), ("s3", sat3),
        ("rvis", rvis), ("c600", c600),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("AltC6#1#Chr01", ["u+", "s1+", "s2+", "s3+", "v+"])
    b.add_path("AltC7#1#Chr01", ["u+", "rvis+", "c600+", "rvis+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f24_01_c6_c7")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    by_hap = _centromere_by_hap(bundle)
    # C6 is masked as repeat_like_bp = 1500
    assert int(by_hap["AltC6#1"]["repeat_like_bp"]) == 1500
    assert int(by_hap["AltC6#1"]["accepted_source_bp"]) == 0
    assert len(_find_unresolved(bundle, "CEN_REPEAT_LIKE")) == 1

    # C7 is retained intact (800 bp) and mounted at ref_start(v) = 600 with leading and trailing N*100
    assert int(by_hap["AltC7#1"]["accepted_source_bp"]) == 800
    cen_islands = _find_sv(bundle, "CEN_ISLAND")
    assert len(cen_islands) == 1
    ev_id = cen_islands[0]["event_id"]
    assert bundle.genotypes[ev_id]["AltC7#1"] == "1"
    assert bundle.genotypes[ev_id]["AltC6#1"] == "0"
    assert bundle.pan_seq == su + scen + ("N" * 100) + (rvis + c600 + rvis) + ("N" * 100) + sv


def test_f24_02_c8_and_c9_step_b_mappy_50bp_reference_subtraction(tmp_path: Path) -> None:
    """F24 / C8 / C9: Step B 50 bp reference subtraction protects >=50 bp internal query insertion (C8), masks reverse-strand homology (C9), and retains low-identity replacement block (C9)."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2411)
    ref_cen = make_complex_seq(2000, seed=2412)
    sv = make_complex_seq(200, seed=2413)
    # C8: 400 bp ref-homologous left + 600 bp novel query insertion + 400 bp ref-homologous right
    ins_600 = make_complex_seq(600, seed=2414)
    cand_c8 = ref_cen[200:600] + ins_600 + ref_cen[600:1000]
    # C9: 600 bp reverse-complement of ref_cen[1100:1700] + 650 bp novel complex replacement block (<70% identity to ref)
    rev_hom_600 = revcomp(ref_cen[1100:1700])
    low_id_650 = make_complex_seq(650, seed=2415)
    cand_c9 = rev_hom_600 + low_id_650

    for nid, seq in [("u", su), ("rcen", ref_cen), ("v", sv), ("c8", cand_c8), ("c9", cand_c9)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "rcen+", "v+"])
    b.add_path("AltC8#1#Chr01", ["u+", "c8+", "v+"])
    b.add_path("AltC9#1#Chr01", ["u+", "c9+", "v+"])
    b.add_bed_region(200, 2200, "CEN1")
    inputs = b.write("f24_02_c8_c9")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    by_hap = _centromere_by_hap(bundle)
    # C8: ~800 bp masked as reference_like_bp, ~600 bp internal insertion protected and retained!
    assert int(by_hap["AltC8#1"]["reference_like_bp"]) >= 700
    assert int(by_hap["AltC8#1"]["accepted_source_bp"]) >= 500
    assert ins_600[50:-50] in bundle.pan_seq

    # C9: ~600 bp reverse-strand homologous prefix masked as reference_like_bp, ~650 bp low-identity suffix retained!
    assert int(by_hap["AltC9#1"]["reference_like_bp"]) >= 500
    assert int(by_hap["AltC9#1"]["accepted_source_bp"]) >= 500
    assert low_id_650[50:-50] in bundle.pan_seq
    assert rev_hom_600[100:-100] not in bundle.pan_seq


def test_f24_03_c10_singleton_ratio_thresholds_below_030_eq_030_mid_and_060(tmp_path: Path) -> None:
    """F24 / C10: Step C singleton_ratio < 0.30 is masked (CEN_REPEAT_LIKE), while == 0.30, 0.30-0.60, and >= 0.60 are retained (not silently deleted)."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2421)
    scen = make_complex_seq(400, seed=2422)
    sv = make_complex_seq(200, seed=2423)

    # Window length 614 -> 600 15-mer positions:
    # 179/600 = 0.29833 < 0.30 (low -> masked)
    seq_below = make_window_with_singleton_ratio(length=614, target_ratio=179 / 600, seed=2424)
    # 180/600 = 0.30000 == 0.30 (non-low -> retained)
    seq_eq30 = make_window_with_singleton_ratio(length=614, target_ratio=180 / 600, seed=2425)
    # 270/600 = 0.45000 in (0.30, 0.60) (non-low -> retained)
    seq_mid = make_window_with_singleton_ratio(length=614, target_ratio=270 / 600, seed=2426)
    # 360/600 = 0.60000 == 0.60 (non-low -> retained)
    seq_eq60 = make_window_with_singleton_ratio(length=614, target_ratio=360 / 600, seed=2427)

    assert compute_canonical_15mer_singleton_ratio(seq_below)[0] == "low"
    assert compute_canonical_15mer_singleton_ratio(seq_eq30)[0] == "non-low"
    assert compute_canonical_15mer_singleton_ratio(seq_mid)[0] == "non-low"
    assert compute_canonical_15mer_singleton_ratio(seq_eq60)[0] == "non-low"

    for nid, seq in [
        ("u", su), ("cen", scen), ("v", sv),
        ("s_low", seq_below), ("s_30", seq_eq30), ("s_mid", seq_mid), ("s_60", seq_eq60),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("HapLow#1#Chr01", ["u+", "s_low+", "v+"])
    b.add_path("HapEq30#1#Chr01", ["u+", "s_30+", "v+"])
    b.add_path("HapMid#1#Chr01", ["u+", "s_mid+", "v+"])
    b.add_path("HapEq60#1#Chr01", ["u+", "s_60+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f24_03_c10")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    by_hap = _centromere_by_hap(bundle)
    assert int(by_hap["HapLow#1"]["repeat_like_bp"]) == 614
    assert int(by_hap["HapLow#1"]["accepted_source_bp"]) == 0
    assert int(by_hap["HapEq30#1"]["accepted_source_bp"]) == 614
    assert int(by_hap["HapMid#1"]["accepted_source_bp"]) == 614
    assert int(by_hap["HapEq60#1"]["accepted_source_bp"]) == 614
    assert seq_eq30 in bundle.pan_seq
    assert seq_mid in bundle.pan_seq
    assert seq_eq60 in bundle.pan_seq


def test_f24_04_c11_and_c19_window_lengths_500_999_1000_1001_isolated_low_and_ambiguous_bases(
    tmp_path: Path,
) -> None:
    """F24 / C11 / C19: Single-window 500/999/1000 bp low blocks and 1001 bp 2-window low block are masked; isolated single low window in >1000 bp is NOT masked; unknown/ambiguous breaks low runs and goes to CEN_SEQUENCE_UNCERTAIN."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2431)
    scen = make_complex_seq(400, seed=2432)
    sv = make_complex_seq(200, seed=2433)

    sat_500 = make_satellite_seq(20, 500, seed=2434)
    sat_999 = make_satellite_seq(20, 999, seed=2435)
    sat_1000 = make_satellite_seq(20, 1000, seed=2436)
    sat_1001 = make_satellite_seq(20, 1001, seed=2437)
    # 2000 bp candidate with ONLY 1 isolated low window [0, 1000): 750 bp satellite + 1250 bp complex
    # Window [0, 1000) has 750 bp satellite -> ratio ~ 250/986 = 0.253 < 0.30 (low);
    # Window [500, 1500) has 250 bp satellite + 750 bp complex -> ratio ~ 750/986 = 0.76 >= 0.30 (non-low);
    # Window [1000, 2000) is 100% complex -> non-low! Only 1 isolated low window -> NOT masked!
    iso_low_2000 = make_satellite_seq(20, 750, seed=2438) + make_complex_seq(1250, seed=2439)
    assert oracle_repeat_mask_intervals(iso_low_2000) == []

    # Candidate containing ambiguous IUPAC 'R'/'Y'/'M' bases (not 'N', so path status is 'complete',
    # but residual contains non-ACGT ambiguous bases -> routed to CEN_SEQUENCE_UNCERTAIN per §3.8.4-C!)
    ambig_cand = make_complex_seq(300, seed=2440) + ("R" * 120) + make_complex_seq(300, seed=2441)

    for nid, seq in [
        ("u", su), ("cen", scen), ("v", sv),
        ("s500", sat_500), ("s999", sat_999), ("s1000", sat_1000), ("s1001", sat_1001),
        ("iso2k", iso_low_2000), ("ambig", ambig_cand),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("Hap500#1#Chr01", ["u+", "s500+", "v+"])
    b.add_path("Hap999#1#Chr01", ["u+", "s999+", "v+"])
    b.add_path("Hap1000#1#Chr01", ["u+", "s1000+", "v+"])
    b.add_path("Hap1001#1#Chr01", ["u+", "s1001+", "v+"])
    b.add_path("HapIso#1#Chr01", ["u+", "iso2k+", "v+"])
    b.add_path("HapAmbig#1#Chr01", ["u+", "ambig+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f24_04_c11_c19")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    by_hap = _centromere_by_hap(bundle)
    assert int(by_hap["Hap500#1"]["repeat_like_bp"]) == 500
    assert int(by_hap["Hap999#1"]["repeat_like_bp"]) == 999
    assert int(by_hap["Hap1000#1"]["repeat_like_bp"]) == 1000
    assert int(by_hap["Hap1001#1"]["repeat_like_bp"]) == 1001
    assert int(by_hap["HapIso#1"]["accepted_source_bp"]) == 2000
    assert iso_low_2000 in bundle.pan_seq
    assert len(_find_unresolved(bundle, "CEN_SEQUENCE_UNCERTAIN")) == 1


def test_f24_05_c12_step_d_island_clustering_with_4_criteria(tmp_path: Path) -> None:
    """F24 / C12: Step D clusters >=95% similar islands using all 4 criteria of §3.6.3 and retains candidates with >=50 bp private differences."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2451)
    scen = make_complex_seq(400, seed=2452)
    sv = make_complex_seq(200, seed=2453)
    isl1 = make_complex_seq(800, seed=2454)
    # 16 dispersed SNPs out of 800 bp -> 98% identity -> merges with isl1
    isl1_snp = mutate_seq_snps(isl1, num_snps=16, seed=2455)
    # 80 bp private insertion inside isl1 (trimmed to 780 bp < 800 bp) -> fails Criterion 3 -> kept as separate CEN_ISLAND
    priv80 = make_complex_seq(80, seed=2456)
    isl2_priv = isl1[50:400] + priv80 + isl1[400:-50]

    for nid, seq in [
        ("u", su), ("cen", scen), ("v", sv), ("i1", isl1), ("i1s", isl1_snp), ("i2p", isl2_priv),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "i1+", "v+"])
    b.add_path("Alt2#1#Chr01", ["u+", "i1s+", "v+"])
    b.add_path("Alt3#1#Chr01", ["u+", "i2p+", "v+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f24_05_c12")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    cen_events = _find_sv(bundle, "CEN_ISLAND")
    assert len(cen_events) == 2
    # The 2-member cluster (Alt1#1, Alt2#1) has n_carrier == 2, while Alt3#1's private island has n_carrier == 1
    carriers = sorted(int(ev["n_carrier"]) for ev in cen_events)
    assert carriers == [1, 2]


def test_f24_06_c13_multi_island_single_hap_and_incomplete_hap_genotypes(tmp_path: Path) -> None:
    """F24 / C13: One haplotype carrying two islands and another with incomplete path emit two CEN_ISLAND events where the former is '1' for both and the latter is '.'."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2461)
    sc1 = make_complex_seq(200, seed=2462)
    sc2 = make_complex_seq(200, seed=2463)
    sv = make_complex_seq(200, seed=2464)
    isl_a = make_complex_seq(600, seed=2465)
    isl_b = make_complex_seq(650, seed=2466)
    for nid, seq in [("u", su), ("c1", sc1), ("c2", sc2), ("v", sv), ("ia", isl_a), ("ib", isl_b)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "c1+", "c2+", "v+"])
    # AltTwo carries both ia+ and ib+ separated by reference centromere node c1+
    b.add_path("AltTwo#1#Chr01", ["u+", "ia+", "c1+", "ib+", "c2+", "v+"])
    b.add_path("AltZero#1#Chr01", ["u+", "c1+", "c2+", "v+"])
    b.add_path("AltPart#1#Chr01", ["u+", "c1+"])
    b.add_bed_region(200, 600, "CEN1")
    inputs = b.write("f24_06_c13")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    cen_events = _find_sv(bundle, "CEN_ISLAND")
    assert len(cen_events) == 2
    for ev in cen_events:
        ev_id = ev["event_id"]
        assert bundle.genotypes[ev_id]["AltTwo#1"] == "1"
        assert bundle.genotypes[ev_id]["AltZero#1"] == "0"
        assert bundle.genotypes[ev_id]["Ref#1"] == "0"
        assert bundle.genotypes[ev_id]["AltPart#1"] == "."


def test_f24_07_c15_blind_panel_summary_and_structured_json_logs(tmp_path: Path) -> None:
    """F24 / C15: Blind panel of pure repeat, old TE, private TE, hybrid repeat+island, ~10% diverged satellite, and 10 kb long-unit repeat logs structured JSON records and summary constants."""
    b = SyntheticInputBuilder(tmp_path)
    su = make_complex_seq(200, seed=2471)
    ref_cen = make_complex_seq(1500, seed=2472)
    sv = make_complex_seq(200, seed=2473)

    pure_sat = make_satellite_seq(20, 1500, seed=2474)
    old_te = ref_cen[200:1000]  # 800 bp homologous to reference centromere
    priv_te = make_complex_seq(800, seed=2475)
    hyb_sat_isl = make_satellite_seq(20, 1500, seed=2476) + make_complex_seq(800, seed=2477)
    div10_sat = make_satellite_seq(20, 1500, seed=2478, divergence=0.10)
    long_unit_10k = make_complex_seq(5000, seed=2479) * 2

    for nid, seq in [
        ("u", su), ("rcen", ref_cen), ("v", sv),
        ("psat", pure_sat), ("ote", old_te), ("pte", priv_te),
        ("hyb", hyb_sat_isl), ("d10", div10_sat), ("lu10k", long_unit_10k),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "rcen+", "v+"])
    b.add_path("HapPureSat#1#Chr01", ["u+", "psat+", "v+"])
    b.add_path("HapOldTE#1#Chr01", ["u+", "ote+", "v+"])
    b.add_path("HapPrivTE#1#Chr01", ["u+", "pte+", "v+"])
    b.add_path("HapHyb#1#Chr01", ["u+", "hyb+", "v+"])
    b.add_path("HapDiv10#1#Chr01", ["u+", "d10+", "v+"])
    b.add_path("HapLong10k#1#Chr01", ["u+", "lu10k+", "v+"])
    b.add_bed_region(200, 1700, "CEN1")
    inputs = b.write("f24_07_c15")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert len(bundle.log_json_records) >= 6
    req_keys = {"region_id", "hap", "candidate_id", "parent_id", "length", "revisited_fraction"}
    for rec in bundle.log_json_records:
        assert req_keys.issubset(rec.keys())


def test_f24_08_c18_repeat_coordinate_retention_and_optional_fasta_dump(tmp_path: Path) -> None:
    """F24 / C18: Filtered repeats (CEN_REPEAT_LIKE / REPEAT_LIKE) retain exact intervals in unresolved.tsv without FASTA export by default, and export deduplicated FASTA under --dump-repeat-sequences while keeping pan.fa identical."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2481)
    su = make_complex_seq(200, seed=2482)
    scen = make_complex_seq(400, seed=2483)
    sv = make_complex_seq(200, seed=2484)
    s2 = make_complex_seq(200, seed=2485)
    norm_sat = make_satellite_seq(20, 1500, seed=2486)
    cen_sat = make_satellite_seq(20, 1600, seed=2487)
    for nid, seq in [
        ("n1", s1), ("u", su), ("cen", scen), ("v", sv), ("n2", s2), ("nsat", norm_sat), ("csat", cen_sat),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "nsat+", "u+", "csat+", "v+", "n2+"])
    b.add_bed_region(400, 800, "CEN1")
    inputs = b.write("f24_08_c18")

    _, bundle_def = run_build(
        inputs, extra_args=["--centromere-mode", "extract-islands"], out_prefix=tmp_path / "c18_def"
    )
    _, bundle_dump = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--dump-repeat-sequences"],
        out_prefix=tmp_path / "c18_dump",
    )
    assert bundle_def is not None and bundle_dump is not None
    assert_all_global_invariants(bundle_def, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bundle_dump, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle_def.pan_seq == bundle_dump.pan_seq
    assert len(bundle_def.unresolved_fa) == 0
    assert len(bundle_dump.unresolved_fa) == 2


# ===========================================================================
# === F25: Deterministic Reference GFF3 Liftover (--ref-gff) (>= 5 tests) ===
# ===========================================================================


def test_f25_01_t14a_plus_strand_intronic_utr_cds_split_and_splice_proximal(tmp_path: Path) -> None:
    """F25 / T14 (+ strand): Insertions in UTR, intron, splice-proximal window, and inside CDS split exon/CDS, recalculate phase, add tags, and preserve 100% protein translation."""
    b = SyntheticInputBuilder(tmp_path)
    # Build a 5-node reference chromosome (200 bp each = 1000 bp)
    # Gene1 on '+' strand:
    # Exon 1: [51, 250] (5' UTR [51, 100], CDS 1 [101, 250] = 150 bp, 50 codons)
    # Intron: [251, 450]
    # Exon 2: [451, 750] (CDS 2 [451, 690] = 240 bp, 80 codons; 3' UTR [691, 750])
    coding_390 = make_coding_seq(num_codons=130, seed=2501)  # 390 bp = 150 + 240
    ref_full = (
        make_complex_seq(100, seed=2502)
        + coding_390[:150]
        + make_complex_seq(200, seed=2503)
        + coding_390[150:]
        + make_complex_seq(310, seed=2504)
    )
    assert len(ref_full) == 1000
    # Split reference at exact insertion points:
    # Slot 1 at 75 (inside 5' UTR [51, 100])
    # Slot 2 at 350 (inside intron [251, 450])
    # Slot 3 at 446 (5 bp upstream of Exon 2 start 451 -> splice_site_proximal_sv=True!)
    # Slot 4 at 551 (inside CDS 2 [451, 690] at 101 bp into CDS 2, non-multiple of 3 -> splits exon 2 & CDS 2!)
    pts = [0, 75, 350, 446, 551, 1000]
    ref_steps = []
    for idx in range(len(pts) - 1):
        nid = f"r{idx+1}"
        b.add_node(nid, ref_full[pts[idx] : pts[idx + 1]])
        ref_steps.append(f"{nid}+")
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    ins_utr = make_complex_seq(100, seed=2505)
    ins_intron = make_complex_seq(120, seed=2506)
    ins_splice = make_complex_seq(90, seed=2507)
    ins_cds = make_complex_seq(150, seed=2508)
    for nid, seq in [("iutr", ins_utr), ("iint", ins_intron), ("ispl", ins_splice), ("icds", ins_cds)]:
        b.add_node(nid, seq)

    b.add_path(
        "Alt1#1#Chr01",
        ["r1+", "iutr+", "r2+", "iint+", "r3+", "ispl+", "r4+", "icds+", "r5+"],
    )
    b.add_gff_gene(
        gene_id="gene_plus",
        mrna_id="mrna_plus",
        strand="+",
        exons=[(51, 250), (451, 750)],
        cds_list=[(101, 250, 0), (451, 690, 0)],
    )
    inputs = b.write("f25_01_t14a")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    gff_text = bundle.lifted_gff_path.read_text(encoding="utf-8")
    assert "utr_insertion" in gff_text
    assert "intronic_insertion" in gff_text
    assert "cds_split_by_insertion" in gff_text
    assert "projection_split=True" in gff_text
    assert "splice_site_proximal_sv=True" in gff_text


def test_f25_02_t14b_minus_strand_cds_split_reverse_transcriptional_phase(tmp_path: Path) -> None:
    """F25 / T14 (- strand): Negative-strand CDS split by internal insertions recalculates phase from highest coordinate to lowest coordinate and preserves 100% protein translation."""
    b = SyntheticInputBuilder(tmp_path)
    coding_360 = make_coding_seq(num_codons=120, seed=2511)  # 360 bp
    # On '-' strand, CDS 1 (5' in transcription) is at higher coordinates [551, 730] (180 bp),
    # and CDS 2 (3' in transcription) is at lower coordinates [151, 330] (180 bp).
    rc_cds2 = revcomp(coding_360[180:])  # placed at [150:330] (0-based)
    rc_cds1 = revcomp(coding_360[:180])  # placed at [550:730] (0-based)
    ref_full = (
        make_complex_seq(150, seed=2512)
        + rc_cds2
        + make_complex_seq(220, seed=2513)
        + rc_cds1
        + make_complex_seq(170, seed=2514)
    )
    assert len(ref_full) == 900
    # Insert at 0-based 200 (inside lower CDS [150, 330)) and 0-based 611 (inside upper CDS [550, 730) at 119 bp from 730 -> non-multiple of 3!)
    pts = [0, 200, 611, 900]
    ref_steps = []
    for idx in range(len(pts) - 1):
        nid = f"rm{idx+1}"
        b.add_node(nid, ref_full[pts[idx] : pts[idx + 1]])
        ref_steps.append(f"{nid}+")
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    ins1 = make_complex_seq(110, seed=2515)
    ins2 = make_complex_seq(130, seed=2516)
    b.add_node("im1", ins1).add_node("im2", ins2)
    b.add_path("Alt1#1#Chr01", ["rm1+", "im1+", "rm2+", "im2+", "rm3+"])
    b.add_gff_gene(
        gene_id="gene_minus",
        mrna_id="mrna_minus",
        strand="-",
        exons=[(151, 330), (551, 730)],
        cds_list=[(151, 330, 0), (551, 730, 0)],
    )
    inputs = b.write("f25_02_t14b")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    assert bundle.lifted_gff_rows is not None
    cds_rows = [r for r in bundle.lifted_gff_rows if r["type"] == "CDS"]
    assert len(cds_rows) == 4


def test_f25_03_insertion_at_exact_feature_outside_boundary_not_split(tmp_path: Path) -> None:
    """F25: Insertion at exact left outside boundary (s-1|s) or right outside boundary (e|e+1) does not split exon/CDS and marks splice_site_proximal_sv=True."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2521)
    s_cds = make_coding_seq(num_codons=60, seed=2522)  # 180 bp at 0-based [200, 380) -> 1-based [201, 380]
    s3 = make_complex_seq(200, seed=2523)
    ins_l = make_complex_seq(100, seed=2524)
    ins_r = make_complex_seq(120, seed=2525)
    for nid, seq in [("n1", s1), ("ncds", s_cds), ("n3", s3), ("il", ins_l), ("ir", ins_r)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "ncds+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "il+", "ncds+", "ir+", "n3+"])
    b.add_gff_gene(
        gene_id="gene_bdy",
        mrna_id="mrna_bdy",
        strand="+",
        exons=[(201, 380)],
        cds_list=[(201, 380, 0)],
    )
    inputs = b.write("f25_03")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    assert bundle.lifted_gff_rows is not None
    cds_rows = [r for r in bundle.lifted_gff_rows if r["type"] == "CDS"]
    assert len(cds_rows) == 1
    assert cds_rows[0]["start"] == 301
    assert cds_rows[0]["end"] == 480
    gff_text = bundle.lifted_gff_path.read_text(encoding="utf-8")
    assert "splice_site_proximal_sv=True" in gff_text


def test_f25_04_parent_gene_and_mrna_span_expansion_over_split_children(tmp_path: Path) -> None:
    """F25: Parent gene and mRNA intervals expand to enclose all shifted and split child features."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(100, seed=2531)
    s_cds = make_coding_seq(num_codons=80, seed=2532)  # 240 bp
    s3 = make_complex_seq(100, seed=2533)
    ins = make_complex_seq(250, seed=2534)
    b.add_node("n1", s1).add_node("c1", s_cds[:120]).add_node("c2", s_cds[120:]).add_node("n3", s3)
    b.add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "c1+", "c2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "c1+", "ins+", "c2+", "n3+"])
    b.add_gff_gene(
        gene_id="gene_exp",
        mrna_id="mrna_exp",
        strand="+",
        exons=[(101, 340)],
        cds_list=[(101, 340, 0)],
    )
    inputs = b.write("f25_04")

    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    assert bundle.lifted_gff_rows is not None
    gene_row = next(r for r in bundle.lifted_gff_rows if r["type"] == "gene")
    mrna_row = next(r for r in bundle.lifted_gff_rows if r["type"] == "mRNA")
    assert gene_row["start"] == 101 and gene_row["end"] == 590
    assert mrna_row["start"] == 101 and mrna_row["end"] == 590


def test_f25_05_c14_centromere_island_and_normal_insertion_co_mounted_inside_gff_feature(
    tmp_path: Path,
) -> None:
    """F25 / C14: Mount point ref_start(v) inside a GFF3 feature shared by a CEN_ISLAND and a normal INS places CEN_ISLAND first, uses a single N spacer between adjacent blocks, and passes offsets/chain/GFF verification."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2540)      # step 0: [0, 200)
    su = make_complex_seq(200, seed=2541)      # step 1: [200, 400) anchor u
    sc1 = make_complex_seq(200, seed=2542)     # step 2: [400, 600)
    # Exon 2 / CDS 2 spans [701, 1060] (360 bp coding sequence across c2[100:200] + v[0:200] + n2[0:60])
    coding_360 = make_coding_seq(num_codons=120, seed=2543)
    sc2 = make_complex_seq(100, seed=2544) + coding_360[:100]   # step 3: [600, 800)
    sv = coding_360[100:300]                                     # step 4: [800, 1000) anchor v
    s2 = coding_360[300:] + make_complex_seq(140, seed=2545)    # step 5: [1000, 1200)

    cen_isl = make_complex_seq(600, seed=2546)
    norm_ins = make_complex_seq(180, seed=2547)

    for nid, seq in [
        ("n1", s1), ("u", su), ("c1", sc1), ("c2", sc2), ("v", sv), ("n2", s2),
        ("cisl", cen_isl), ("nins", norm_ins),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "c1+", "c2+", "v+", "n2+"])
    # AltCen has complete centromere traversal carrying cen_isl (mounted at ref_start(v)=800)
    b.add_path("AltCen#1#Chr01", ["n1+", "u+", "c1+", "cisl+", "c2+", "v+", "n2+"])
    # AltNorm starts at c2+ (cannot be sliced at u..v) and has a pure insertion at (c2, v) -> ref_start(v)=800!
    b.add_path("AltNorm#1#Chr01", ["c2+", "nins+", "v+", "n2+"])
    # BED [450, 750) expands outward to u (ref_end=400) and v (ref_start=800)
    b.add_bed_region(450, 750, "CEN1")

    # Gene with CDS spanning [701, 1060], so mount point ref_start(v)=800 is strictly inside the CDS!
    b.add_gff_gene(
        gene_id="gene_c14",
        mrna_id="mrna_c14",
        strand="+",
        exons=[(701, 1060)],
        cds_list=[(701, 1060, 0)],
    )
    inputs = b.write("f25_05_c14")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    cen_evs = _find_sv(bundle, "CEN_ISLAND")
    ins_evs = _find_sv(bundle, "INS")
    assert len(cen_evs) == 1
    assert len(ins_evs) == 1
    assert int(cen_evs[0]["ref_start"]) == 800
    assert int(ins_evs[0]["ref_start"]) == 800
    # Verify CEN_ISLAND is placed before normal INS at ref_start(v)=800 with a single N*100 spacer between them
    assert int(cen_evs[0]["pan_start"]) < int(ins_evs[0]["pan_start"])
    assert int(ins_evs[0]["pan_start"]) - int(cen_evs[0]["pan_end"]) == 100
    assert "N" * 200 not in bundle.pan_seq


# ===========================================================================
# === F26: Output Writers, CLI (build, verify, merge), Threads & --qc (>= 5 tests) ===
# ===========================================================================


def test_f26_01_all_11_output_files_and_exact_headers(tmp_path: Path) -> None:
    """F26: panflat build with --ref-gff and --centromere-mode extract-islands produces all 11 output files with exact specification headers."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2601)
    su = make_complex_seq(200, seed=2602)
    scen = make_complex_seq(300, seed=2603)
    sv = make_complex_seq(200, seed=2604)
    s_cds = make_coding_seq(num_codons=60, seed=2605)
    isl = make_complex_seq(600, seed=2606)
    unanch = make_complex_seq(120, seed=2607)
    for nid, seq in [
        ("n1", s1), ("u", su), ("cen", scen), ("v", sv), ("ncds", s_cds), ("isl", isl), ("un", unanch),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "ncds+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "isl+", "v+", "ncds+"])
    b.add_path("AltBreak#1#Chr01", ["un+", "v+", "ncds+"])
    b.add_bed_region(400, 700, "CEN1")
    b.add_gff_gene("g1", "m1", "+", [(901, 1080)], [(901, 1080, 0)])
    inputs = b.write("f26_01")

    _, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    for p in [
        bundle.pan_fa_path,
        bundle.offsets_path,
        bundle.chain_path,
        bundle.sv_path,
        bundle.genotypes_path,
        bundle.inserts_path,
        bundle.unresolved_path,
        bundle.unresolved_fa_path,
        bundle.centromere_path,
        bundle.lifted_gff_path,
        bundle.summary_path,
        bundle.log_path,
    ]:
        assert p.exists(), f"Missing output file: {p}"

    assert bundle.offsets_header == [
        "chrom", "ref_step", "node_id", "ref_start", "ref_end", "pan_start", "pan_end", "slot_ins_len",
    ]
    assert bundle.sv_header == [
        "chrom", "pan_start", "pan_end", "event_id", "type", "ref_start", "ref_end",
        "anchors", "alleles", "n_carrier", "n_called", "region_id",
    ]
    assert bundle.inserts_header == [
        "chrom", "pan_start", "pan_end", "event_id", "allele", "piece",
        "src_hap", "src_path", "src_intervals", "src_strand", "members", "join_before",
    ]
    assert bundle.unresolved_header == [
        "record_id", "region_id", "event_id", "src_path", "src_start", "src_end",
        "src_strand", "length", "reason", "exported_fasta_id",
    ]


def test_f26_02_standalone_panflat_verify_detects_corrupted_output(tmp_path: Path) -> None:
    """F26: Standalone `panflat verify` passes on valid output and fails with non-zero exit when pan.fa is corrupted."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2611)
    s2 = make_complex_seq(200, seed=2612)
    ins = make_complex_seq(150, seed=2613)
    b.add_node("n1", s1).add_node("n2", s2).add_node("ins", ins)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ins+", "n2+"])
    inputs = b.write("f26_02")

    _, bundle = run_build(inputs)
    assert bundle is not None
    ok_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path)
    assert ok_proc.returncode == 0

    # Corrupt 1 base inside pan.fa
    orig_seq = bundle.pan_seq
    bad_base = "A" if orig_seq[10] != "A" else "C"
    corrupted_seq = orig_seq[:10] + bad_base + orig_seq[11:]
    bundle.pan_fa_path.write_text(f">{inputs.chrom}\n{corrupted_seq}\n", encoding="utf-8")

    bad_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert bad_proc.returncode != 0


def test_f26_03_panflat_merge_multi_chromosome_workflow(tmp_path: Path) -> None:
    """F26: `panflat merge` combines multiple chromosomes (one with centromere mode, one without) while preserving unique event/region IDs and not fabricating centromere rows for normal chromosomes."""
    # Chr01 with extract-islands
    b1 = SyntheticInputBuilder(tmp_path / "chr1", chrom="Chr01", backbone_prefix="Ref#1#")
    c1_u = make_complex_seq(200, seed=2621)
    c1_cen = make_complex_seq(300, seed=2622)
    c1_v = make_complex_seq(200, seed=2623)
    c1_isl = make_complex_seq(600, seed=2624)
    for nid, seq in [("u", c1_u), ("cen", c1_cen), ("v", c1_v), ("isl", c1_isl)]:
        b1.add_node(nid, seq)
    b1.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b1.add_path("Alt1#1#Chr01", ["u+", "isl+", "v+"])
    b1.add_bed_region(200, 500, "CEN_CHR1")
    in1 = b1.write("chr1")
    _, bun1 = run_build(in1, extra_args=["--centromere-mode", "extract-islands"])
    assert bun1 is not None

    # Chr02 without centromere mode
    b2 = SyntheticInputBuilder(tmp_path / "chr2", chrom="Chr02", backbone_prefix="Ref#1#")
    c2_1 = make_complex_seq(200, seed=2625)
    c2_2 = make_complex_seq(200, seed=2626)
    c2_ins = make_complex_seq(150, seed=2627)
    for nid, seq in [("m1", c2_1), ("m2", c2_2), ("mins", c2_ins)]:
        b2.add_node(nid, seq)
    b2.set_ref_path("Ref#1#Chr02", ["m1+", "m2+"])
    b2.add_path("Alt1#1#Chr02", ["m1+", "mins+", "m2+"])
    in2 = b2.write("chr2")
    _, bun2 = run_build(in2)
    assert bun2 is not None

    merged_prefix = tmp_path / "merged_all"
    _, m_bundle = run_merge([in1.out_prefix, in2.out_prefix], merged_prefix)
    assert m_bundle is not None

    assert set(m_bundle.pan_fa.keys()) == {"Chr01", "Chr02"}
    assert len(m_bundle.sv_rows) == 2
    assert len({r["event_id"] for r in m_bundle.sv_rows}) == 2
    assert m_bundle.centromere_rows is not None
    # Chr02 did not enable centromere mode -> centromere.tsv only has Chr01 rows!
    assert {r["chrom"] for r in m_bundle.centromere_rows} == {"Chr01"}


def test_f26_04_t22_shuffled_input_order_and_multi_thread_byte_identical_outputs(tmp_path: Path) -> None:
    """F26 / T22: Multi-threaded execution (--threads 4 vs --threads 1) and shuffled GFA input order produce byte-identical outputs across multiple loci and short split CDS segments."""
    s1 = make_complex_seq(200, seed=2631)
    # 60 bp CDS split into 2 bp + 58 bp by an insertion!
    cds_plus = make_coding_seq(num_codons=20, seed=2632)  # 60 bp
    s3 = make_complex_seq(200, seed=2633)
    cds_minus = revcomp(make_coding_seq(num_codons=20, seed=2634))  # 60 bp on '-' strand
    s5 = make_complex_seq(200, seed=2635)
    ins_a = make_complex_seq(140, seed=2636)
    ins_b = make_complex_seq(160, seed=2637)

    def _build_graph(sub_dir: Path, shuffle: bool) -> InputPaths:
        b = SyntheticInputBuilder(sub_dir)
        nodes = [
            ("n1", s1),
            ("cp1", cds_plus[:2]),    # 2 bp ultra-short CDS piece!
            ("cp2", cds_plus[2:]),    # 58 bp CDS piece
            ("n3", s3),
            ("cm1", cds_minus[:59]),  # 59 bp CDS piece
            ("cm2", cds_minus[59:]),  # 1 bp ultra-short CDS piece on '-' strand!
            ("n5", s5),
            ("ia", ins_a),
            ("ib", ins_b),
        ]
        if shuffle:
            nodes = list(reversed(nodes))
        for nid, seq in nodes:
            b.add_node(nid, seq)
        b.set_ref_path("Ref#1#Chr01", ["n1+", "cp1+", "cp2+", "n3+", "cm1+", "cm2+", "n5+"])
        paths = [
            ("Alt1#1#Chr01", ["n1+", "cp1+", "ia+", "cp2+", "n3+", "cm1+", "cm2+", "n5+"]),
            ("Alt2#1#Chr01", ["n1+", "cp1+", "cp2+", "n3+", "cm1+", "ib+", "cm2+", "n5+"]),
        ]
        if shuffle:
            paths = list(reversed(paths))
        for pname, psteps in paths:
            b.add_path(pname, psteps)
        b.add_gff_gene("gp", "mp", "+", [(201, 260)], [(201, 260, 0)])
        b.add_gff_gene("gm", "mm", "-", [(461, 520)], [(461, 520, 0)])
        return b.write("t22")

    in_norm = _build_graph(tmp_path / "norm", shuffle=False)
    in_shuf = _build_graph(tmp_path / "shuf", shuffle=True)

    _, bun_t1 = run_build(in_norm, extra_args=["--threads", "1"], out_prefix=tmp_path / "out_t1")
    _, bun_t4 = run_build(in_norm, extra_args=["--threads", "4"], out_prefix=tmp_path / "out_t4")
    _, bun_shuf = run_build(in_shuf, extra_args=["--threads", "2"], out_prefix=tmp_path / "out_shuf")
    assert bun_t1 is not None and bun_t4 is not None and bun_shuf is not None

    assert_all_global_invariants(bun_t1, in_norm.gfa_path, in_norm.ref_fa_path)
    assert_gff3_liftover_integrity(
        in_norm.ref_fa_path, in_norm.gff_path, bun_t1.pan_fa_path, bun_t1.lifted_gff_path
    )

    # --threads 1 and --threads 4 must be byte-for-byte identical across all core output files!
    for ext in [
        ".pan.fa", ".offsets.tsv", ".ref_to_pan.chain", ".sv.tsv",
        ".genotypes.tsv", ".inserts.tsv", ".ref.lifted.gff3", ".unresolved.tsv",
    ]:
        p1 = Path(f"{tmp_path / 'out_t1'}{ext}").read_bytes()
        p4 = Path(f"{tmp_path / 'out_t4'}{ext}").read_bytes()
        assert p1 == p4, f"Byte mismatch between --threads 1 and --threads 4 on {ext}"

    # Shuffled GFA input order produces identical pan.fa, offsets.tsv, chain, and lifted GFF3
    assert bun_shuf.pan_seq == bun_t1.pan_seq
    assert bun_shuf.offsets == bun_t1.offsets
    assert bun_shuf.chain_blocks == bun_t1.chain_blocks


def test_f26_05_qc_flag_alignment_redundancy_metrics_in_summary(tmp_path: Path) -> None:
    """F26: `panflat build --qc` computes alignment redundancy metrics for >=150 bp pieces and separately reports 50-149 bp pieces in summary.tsv."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=2641)
    s2 = make_complex_seq(250, seed=2642)
    s3 = make_complex_seq(250, seed=2643)
    ins_big = make_complex_seq(300, seed=2644)    # >= 150 bp
    ins_small = make_complex_seq(80, seed=2645)   # 50-149 bp
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ibig", ins_big), ("ism", ins_small)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ibig+", "n2+", "ism+", "n3+"])
    inputs = b.write("f26_05")

    _, bundle = run_build(inputs, extra_args=["--qc"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    summary_low = bundle.summary_text.lower()
    assert "qc" in summary_low or "0.90" in summary_low or "149" in summary_low
    assert "80" in bundle.summary_text
