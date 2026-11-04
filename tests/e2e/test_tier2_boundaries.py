"""Tier 2 Opaque-Box End-to-End (E2E) Boundary, Threshold & Negative Tests (F1-F26).

Strictly invokes PanFlat via its documented CLI (`panflat build`, `panflat verify`,
`panflat merge`) in a subprocess and validates exit codes, stdout/stderr, and the
11 output files on disk using independent oracles from `tests.e2e.helpers`.
Zero imports from `panflat`.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.helpers import (
    InputPaths,
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


def _build_two_anchor_with_ins(tmp_path: Path, name: str = "base") -> InputPaths:
    """Create a compact 2-node reference with a 60 bp insertion for verify tests."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=11))
    b.add_node("n2", make_complex_seq(100, seed=12))
    b.add_node("x1", make_complex_seq(60, seed=13))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x1+", "n2+"])
    return b.write(name)


# ===========================================================================
# === F1 Boundaries: Invariant I: Reference Base Preservation (5 tests) ===
# ===========================================================================


def test_f1_boundary_single_1bp_reference_nodes_with_insertion(tmp_path: Path) -> None:
    """F1 Boundary: Reference composed of minimal 1 bp nodes preserves every base."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", "A")
    b.add_node("n2", "T")
    b.add_node("x_ins", make_complex_seq(50, seed=101))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x_ins+", "n2+"])
    inputs = b.write("f1_1bp_nodes")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.offsets) == 2
    assert bundle.offsets[0]["ref_start"] == 0
    assert bundle.offsets[0]["ref_end"] == 1
    assert bundle.offsets[0]["pan_start"] == 0
    assert bundle.offsets[0]["pan_end"] == 1
    assert bundle.offsets[0]["slot_ins_len"] == 0
    assert bundle.offsets[1]["ref_start"] == 1
    assert bundle.offsets[1]["ref_end"] == 2
    assert bundle.offsets[1]["pan_start"] == 51
    assert bundle.offsets[1]["pan_end"] == 52
    assert bundle.offsets[1]["slot_ins_len"] == 50
    assert bundle.pan_seq[0] == "A"
    assert bundle.pan_seq[-1] == "T"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f1_negative_verify_fails_on_1bp_pan_fa_ref_base_mutation(tmp_path: Path) -> None:
    """F1 Negative: `panflat verify` exits non-zero if 1 reference base in pan.fa is mutated."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f1_mut_ref")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    orig_seq = bundle.pan_seq
    flipped0 = "T" if orig_seq[0] != "T" else "G"
    corrupted = flipped0 + orig_seq[1:]
    bundle.pan_fa_path.write_text(f">{inputs.chrom}\n{corrupted}\n", encoding="utf-8")

    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f1_negative_verify_fails_on_offsets_1bp_gap_or_overlap(tmp_path: Path) -> None:
    """F1 Negative: `panflat verify` fails when offsets.tsv has a 1 bp gap or 1 bp overlap."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f1_offsets_gap")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    orig_offsets = bundle.offsets_path.read_text(encoding="utf-8")
    lines = orig_offsets.splitlines()
    row1 = lines[2].split("\t")
    # Introduce 1 bp gap in ref_start of step 1 (100 -> 101)
    row1_gap = list(row1)
    row1_gap[3] = str(int(row1[3]) + 1)
    bundle.offsets_path.write_text(
        "\n".join([lines[0], lines[1], "\t".join(row1_gap)]) + "\n", encoding="utf-8"
    )
    v_gap = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_gap.returncode != 0

    # Introduce 1 bp overlap in ref_start of step 1 (100 -> 99)
    row1_ov = list(row1)
    row1_ov[3] = str(int(row1[3]) - 1)
    bundle.offsets_path.write_text(
        "\n".join([lines[0], lines[1], "\t".join(row1_ov)]) + "\n", encoding="utf-8"
    )
    v_ov = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_ov.returncode != 0


def test_f1_negative_verify_fails_on_pan_fa_1bp_truncation(tmp_path: Path) -> None:
    """F1 Negative: `panflat verify` exits non-zero when pan.fa is truncated by 1 bp."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f1_trunc")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    bundle.pan_fa_path.write_text(
        f">{inputs.chrom}\n{bundle.pan_seq[:-1]}\n", encoding="utf-8"
    )
    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f1_boundary_5prime_offset_strictly_zero_with_unanchored_5prime_tail(
    tmp_path: Path,
) -> None:
    """F1 Boundary (§3.7): 5' offset is always 0 even when donor has >=50 bp 5' tail."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=14))
    b.add_node("n2", make_complex_seq(100, seed=15))
    b.add_node("u5", make_complex_seq(75, seed=16))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["u5+", "n1+", "n2+"])
    inputs = b.write("f1_5prime_zero")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[0]["ref_step"] == 0
    assert bundle.offsets[0]["ref_start"] == 0
    assert bundle.offsets[0]["pan_start"] == 0
    assert bundle.offsets[0]["slot_ins_len"] == 0
    assert any(r["reason"] == "UNANCHORED_END" and r["length"] == 75 for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F2 Boundaries: Invariant II: Piece Provenance & Join Integrity (5 tests) ===
# ===========================================================================


def test_f2_negative_verify_fails_on_1bp_mutation_inside_inserted_piece(
    tmp_path: Path,
) -> None:
    """F2 Negative: `panflat verify` fails when 1 base inside an inserted piece is altered."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f2_mut_piece")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1

    ins_start = int(bundle.inserts[0]["pan_start"])
    seq_list = list(bundle.pan_seq)
    seq_list[ins_start + 5] = "A" if seq_list[ins_start + 5] != "A" else "C"
    bundle.pan_fa_path.write_text(
        f">{inputs.chrom}\n{''.join(seq_list)}\n", encoding="utf-8"
    )
    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f2_negative_verify_fails_on_flipped_src_strand_in_inserts_tsv(
    tmp_path: Path,
) -> None:
    """F2 Negative: `panflat verify` fails when src_strand in inserts.tsv is flipped."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f2_flip_strand")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    lines = bundle.inserts_path.read_text(encoding="utf-8").splitlines()
    cols = lines[1].split("\t")
    cols[9] = "-" if cols[9] == "+" else "+"
    bundle.inserts_path.write_text("\n".join([lines[0], "\t".join(cols)]) + "\n", encoding="utf-8")

    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f2_negative_verify_fails_on_non_spacer_piece_1bp_gap_or_overlap(
    tmp_path: Path,
) -> None:
    """F2 Negative: `panflat verify` fails when non-spacer pieces have a 1 bp gap or overlap."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=21))
    b.add_node("n2", make_complex_seq(100, seed=22))
    b.add_node("u1", make_complex_seq(60, seed=23))
    b.add_node("w1", make_complex_seq(60, seed=24))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "u1+", "u1+", "w1+", "n2+"])
    inputs = b.write("f2_piece_gap")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 2

    orig_text = bundle.inserts_path.read_text(encoding="utf-8")
    lines = orig_text.splitlines()
    p1 = lines[1].split("\t")
    # 1 bp gap: shrink pan_end of piece 1 by 1
    p1_gap = list(p1)
    p1_gap[2] = str(int(p1[2]) - 1)
    bundle.inserts_path.write_text(
        "\n".join([lines[0], "\t".join(p1_gap), lines[2]]) + "\n", encoding="utf-8"
    )
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 1 bp overlap: expand pan_end of piece 1 by 1
    p1_ov = list(p1)
    p1_ov[2] = str(int(p1[2]) + 1)
    bundle.inserts_path.write_text(
        "\n".join([lines[0], "\t".join(p1_ov), lines[2]]) + "\n", encoding="utf-8"
    )
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


def test_f2_negative_verify_fails_on_corrupted_spacer_non_n_base(tmp_path: Path) -> None:
    """F2 Negative: `panflat verify` fails when an artificial N spacer contains a non-N base."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=25))
    b.add_node("n2", make_complex_seq(100, seed=26))
    b.add_node("r1", make_complex_seq(60, seed=27))
    b.add_node("r2", make_complex_seq(70, seed=28))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f2_spacer_corrupt")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 2

    spacer_pos = int(bundle.inserts[0]["pan_end"]) + 10
    assert bundle.pan_seq[spacer_pos] == "N"
    mut_seq = bundle.pan_seq[:spacer_pos] + "G" + bundle.pan_seq[spacer_pos + 1 :]
    bundle.pan_fa_path.write_text(f">{inputs.chrom}\n{mut_seq}\n", encoding="utf-8")

    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f2_boundary_source_n_bases_inside_piece_verified_against_gfa(
    tmp_path: Path,
) -> None:
    """F2 Boundary: Piece containing native source N bases is verified against GFA."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=29))
    b.add_node("n2", make_complex_seq(100, seed=30))
    seq_with_n = make_complex_seq(35, seed=31) + "NNNNN" + make_complex_seq(40, seed=32)
    b.add_node("x_n", seq_with_n)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x_n+", "n2+"])
    inputs = b.write("f2_native_n")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    p_s = int(bundle.inserts[0]["pan_start"])
    p_e = int(bundle.inserts[0]["pan_end"])
    assert bundle.pan_seq[p_s:p_e] == seq_with_n
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Mutating a native source N to A in pan.fa must fail `panflat verify`
    n_idx = p_s + 36
    assert bundle.pan_seq[n_idx] == "N"
    bad_seq = bundle.pan_seq[:n_idx] + "A" + bundle.pan_seq[n_idx + 1 :]
    bundle.pan_fa_path.write_text(f">{inputs.chrom}\n{bad_seq}\n", encoding="utf-8")
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


# ===========================================================================
# === F3 Boundaries: Invariant III: Coordinate Consistency (5 tests) ===
# ===========================================================================


def test_f3_negative_verify_fails_when_chain_tsize_or_qsize_off_by_one(
    tmp_path: Path,
) -> None:
    """F3 Negative: `panflat verify` fails if chain tSize or qSize is off by 1."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f3_chain_size")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    orig_chain = bundle.chain_text
    lines = orig_chain.splitlines()
    hdr = lines[0].split()
    # Corrupt tSize (+1)
    hdr_t = list(hdr)
    hdr_t[3] = str(int(hdr[3]) + 1)
    bundle.chain_path.write_text("\n".join([" ".join(hdr_t), *lines[1:]]) + "\n", encoding="utf-8")
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # Corrupt qSize (-1)
    hdr_q = list(hdr)
    hdr_q[8] = str(int(hdr[8]) - 1)
    bundle.chain_path.write_text("\n".join([" ".join(hdr_q), *lines[1:]]) + "\n", encoding="utf-8")
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


def test_f3_negative_verify_fails_when_chain_dt_nonzero(tmp_path: Path) -> None:
    """F3 Negative: `panflat verify` fails if chain block dt != 0."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f3_chain_dt")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    lines = bundle.chain_text.splitlines()
    blk = lines[1].split()
    blk[1] = "1"  # dt = 1 instead of 0
    bundle.chain_path.write_text("\n".join([lines[0], "\t".join(blk), *lines[2:]]) + "\n", encoding="utf-8")
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


def test_f3_negative_verify_fails_when_chain_dq_mismatch_slot_ins_len(
    tmp_path: Path,
) -> None:
    """F3 Negative: `panflat verify` fails when chain dq disagrees with slot_ins_len."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f3_chain_dq")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    lines = bundle.chain_text.splitlines()
    blk = lines[1].split()
    blk[2] = str(int(blk[2]) - 1)  # dq off by 1
    bundle.chain_path.write_text("\n".join([lines[0], "\t".join(blk), *lines[2:]]) + "\n", encoding="utf-8")
    assert run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


def test_f3_boundary_insertion_at_first_internal_step_vs_last_internal_step(
    tmp_path: Path,
) -> None:
    """F3 Boundary: Insertions at step 1 (first internal) and step N_ref-1 (last internal)."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(4):
        b.add_node(f"n{i}", make_complex_seq(60, seed=40 + i))
    b.add_node("x_first", make_complex_seq(60, seed=45))
    b.add_node("x_last", make_complex_seq(70, seed=46))
    b.set_ref_path("Ref#1#Chr01", ["n0+", "n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n0+", "x_first+", "n1+", "n2+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n0+", "n1+", "n2+", "x_last+", "n3+"])
    inputs = b.write("f3_first_last")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert [r["slot_ins_len"] for r in bundle.offsets] == [0, 60, 0, 70]
    blocks = bundle.chain_blocks[0]
    assert blocks == [(60, 0, 60), (120, 0, 70), (60, 0, 0)]
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f3_boundary_consecutive_reference_steps_all_have_insertions(
    tmp_path: Path,
) -> None:
    """F3 Boundary: Every consecutive internal reference boundary has an insertion."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(4):
        b.add_node(f"n{i}", make_complex_seq(50, seed=50 + i))
    b.add_node("i1", make_complex_seq(50, seed=55))
    b.add_node("i2", make_complex_seq(60, seed=56))
    b.add_node("i3", make_complex_seq(70, seed=57))
    b.set_ref_path("Ref#1#Chr01", ["n0+", "n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n0+", "i1+", "n1+", "i2+", "n2+", "i3+", "n3+"])
    inputs = b.write("f3_consec_ins")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert [r["slot_ins_len"] for r in bundle.offsets] == [0, 50, 60, 70]
    assert bundle.chain_blocks[0] == [(50, 0, 50), (50, 0, 60), (50, 0, 70), (50, 0, 0)]
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F4 Boundaries: Reference Backbone Matching & Validation (5 tests) ===
# ===========================================================================


def test_f4_negative_zero_matching_backbone_paths_in_gfa(tmp_path: Path) -> None:
    """F4 Negative: Exit non-zero when --backbone-prefix matches 0 paths in GFA."""
    b = SyntheticInputBuilder(tmp_path, backbone_prefix="MissingRef#1#")
    b.add_node("n1", make_complex_seq(100, seed=61))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    inputs = b.write("f4_zero_backbone")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f4_negative_multiple_matching_backbone_paths_in_gfa(tmp_path: Path) -> None:
    """F4 Negative: Exit non-zero when --backbone-prefix matches >=2 paths in GFA."""
    b = SyntheticInputBuilder(tmp_path, backbone_prefix="Ref#1#")
    b.add_node("n1", make_complex_seq(100, seed=62))
    b.add_node("n2", make_complex_seq(100, seed=63))
    b.set_ref_path("Ref#1#Chr01_a", ["n1+"])
    b.add_path("Ref#1#Chr01_b", ["n2+"])
    inputs = b.write("f4_two_backbones")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f4_negative_multiple_sequences_in_ref_fa(tmp_path: Path) -> None:
    """F4 Negative: Exit non-zero when --ref-fa contains multiple sequences."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=64))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    b.add_ref_fasta_record("Chr02", make_complex_seq(100, seed=65))
    inputs = b.write("f4_multi_reffa")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f4_negative_1bp_snp_mismatch_between_gfa_ref_and_ref_fa(tmp_path: Path) -> None:
    """F4 Negative: Exit non-zero when assembled GFA reference path differs by 1 SNP from ref.fa."""
    b = SyntheticInputBuilder(tmp_path)
    seq = make_complex_seq(200, seed=66)
    b.add_node("n1", seq)
    mut_ref = mutate_seq_snps(seq, positions=[100], seed=67)
    b.set_ref_path("Ref#1#Chr01", ["n1+"], ref_fa_seq=mut_ref)
    inputs = b.write("f4_1bp_snp")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f4_negative_n_mismatch_vs_case_insensitive_n_match(tmp_path: Path) -> None:
    """F4 Negative/Boundary: N vs A mismatch fails; lowercase 'n' vs uppercase 'N' succeeds."""
    seq_with_n = make_complex_seq(50, seed=68) + "N" + make_complex_seq(50, seed=69)
    seq_no_n = seq_with_n[:50] + "A" + seq_with_n[51:]

    # 1) GFA has A where ref.fa has N -> must fail
    b1 = SyntheticInputBuilder(tmp_path / "case1")
    b1.add_node("n1", seq_no_n)
    b1.set_ref_path("Ref#1#Chr01", ["n1+"], ref_fa_seq=seq_with_n)
    p1, _ = run_build(b1.write("f4_n_mismatch"), check=False)
    assert p1.returncode != 0

    # 2) GFA has N where ref.fa has lowercase 'n' -> must succeed
    b2 = SyntheticInputBuilder(tmp_path / "case2")
    b2.add_node("n1", seq_with_n)
    b2.set_ref_path("Ref#1#Chr01", ["n1+"], ref_fa_seq=seq_with_n.lower())
    in2 = b2.write("f4_n_case_ok")
    p2, bundle2 = run_build(in2, check=True)
    assert p2.returncode == 0 and bundle2 is not None
    assert_all_global_invariants(bundle2, in2.gfa_path, in2.ref_fa_path)


# ===========================================================================
# === F5 Boundaries: GFA P-line and W-line Parsing (5 tests) ===
# ===========================================================================


def test_f5_negative_duplicate_source_interval_across_p_and_w_lines(tmp_path: Path) -> None:
    """F5 Negative (§2.1 / T21): Duplicate source interval in both P and W lines is rejected."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=71))
    b.add_node("n2", make_complex_seq(100, seed=72))
    b.add_node("x1", make_complex_seq(60, seed=73))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x1+", "n2+"])
    b.add_walk("Alt", "1", "Chr01", 0, 260, ["n1+", "x1+", "n2+"])
    inputs = b.write("f5_dup_pw")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f5_boundary_two_w_lines_with_gap_not_stitched_across_gap(tmp_path: Path) -> None:
    """F5 Boundary (§2.1): Two W segments for same Sample#Hap#Seq are NOT stitched across gap."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=74))
    b.add_node("n2", make_complex_seq(100, seed=75))
    b.add_node("n3", make_complex_seq(100, seed=76))
    b.add_node("u_end1", make_complex_seq(60, seed=77))
    b.add_node("u_start2", make_complex_seq(60, seed=78))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    # W1 covers [0, 160): n1 -> u_end1; W2 covers [300, 460): u_start2 -> n3
    b.add_walk("Alt", "1", "Chr01", 0, 160, ["n1+", "u_end1+"])
    b.add_walk("Alt", "1", "Chr01", 300, 460, ["u_start2+", "n3+"])
    inputs = b.write("f5_w_gap")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Because W1 and W2 are not stitched across the gap, n2 is NOT called as DEL
    # and u_end1 / u_start2 are NOT inserted into pan.fa; they are UNANCHORED_END
    assert len(bundle.sv_rows) == 0
    assert bundle.pan_seq == b.spell_steps(["n1+", "n2+", "n3+"])
    unres_coords = {(r["src_start"], r["src_end"], r["reason"]) for r in bundle.unresolved}
    assert (100, 160, "UNANCHORED_END") in unres_coords
    assert (300, 360, "UNANCHORED_END") in unres_coords
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f5_boundary_w_line_large_start_offset_preserved_in_inserts_and_unresolved(
    tmp_path: Path,
) -> None:
    """F5 Boundary: W-line with large seq_start=999999 preserves 0-based source offset."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=79))
    b.add_node("n2", make_complex_seq(100, seed=80))
    b.add_node("u0", make_complex_seq(60, seed=81))
    b.add_node("x1", make_complex_seq(70, seed=82))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Walk length = 60 + 100 + 70 + 100 = 330, starting at 999999 -> end = 1000329
    b.add_walk("Alt", "1", "ctg99", 999999, 1000329, ["u0+", "n1+", "x1+", "n2+"])
    inputs = b.write("f5_w_large_offset")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_intervals"] == "1000159-1000229"
    assert any(
        r["src_start"] == 999999 and r["src_end"] == 1000059 and r["reason"] == "UNANCHORED_END"
        for r in bundle.unresolved
    )
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f5_boundary_p_line_without_pansn_hash_separator(tmp_path: Path) -> None:
    """F5 Boundary: P-line path name without '#' separator is parsed deterministically."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=83))
    b.add_node("n2", make_complex_seq(100, seed=84))
    b.add_node("x1", make_complex_seq(60, seed=85))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("PlainDonorContig", ["n1+", "x1+", "n2+"])
    inputs = b.write("f5_plain_p")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 1
    assert bundle.sv_rows[0]["type"] == "INS"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f5_negative_p_or_w_line_referencing_undefined_node_id(tmp_path: Path) -> None:
    """F5 Negative: P or W line referencing a node missing from S lines exits non-zero."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=86))
    b.add_node("n2", make_complex_seq(100, seed=87))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "missing_node+", "n2+"])
    inputs = b.write("f5_undef_node")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


# ===========================================================================
# === F6 Boundaries: GFA Structural Integrity Checks (5 tests) ===
# ===========================================================================


def test_f6_negative_s_line_with_missing_star_sequence(tmp_path: Path) -> None:
    """F6 Negative (§2.1): Node S line with '*' missing sequence exits non-zero."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=91))
    b.add_node("n2", make_complex_seq(100, seed=92))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_raw_line("S\tn_star\t*")
    inputs = b.write("f6_star_seq")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f6_negative_l_line_with_nonzero_overlap(tmp_path: Path) -> None:
    """F6 Negative (§2.1): Link L line with non-zero overlap (e.g. 10M) exits non-zero."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=93))
    b.add_node("n2", make_complex_seq(100, seed=94))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], auto_links=False)
    b.add_link("n1", "+", "n2", "+", overlap="10M")
    inputs = b.write("f6_nonzero_l")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f6_negative_p_line_with_explicit_nonzero_or_unknown_overlap(tmp_path: Path) -> None:
    """F6 Negative (§2.1 / T21): P line with non-zero ('10M') or unknown ('?') overlap fails."""
    b1 = SyntheticInputBuilder(tmp_path / "p_10m")
    b1.add_node("n1", make_complex_seq(100, seed=95))
    b1.add_node("n2", make_complex_seq(100, seed=96))
    b1.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="10M")
    assert run_build(b1.write("f6_p_10m"), check=False)[0].returncode != 0

    b2 = SyntheticInputBuilder(tmp_path / "p_unk")
    b2.add_node("n1", make_complex_seq(100, seed=95))
    b2.add_node("n2", make_complex_seq(100, seed=96))
    b2.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="?")
    assert run_build(b2.write("f6_p_unk"), check=False)[0].returncode != 0


def test_f6_negative_p_line_with_star_overlap_and_missing_l_link(tmp_path: Path) -> None:
    """F6 Negative (§2.1): P line with '*' overlap when corresponding L link is missing fails."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=97))
    b.add_node("n2", make_complex_seq(100, seed=98))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"], overlaps="*", auto_links=False)
    inputs = b.write("f6_broken_link")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


def test_f6_negative_gfa_containing_j_jump_line(tmp_path: Path) -> None:
    """F6 Negative (§2.1 / T21): GFA containing a J jump line exits non-zero."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=99))
    b.add_node("n2", make_complex_seq(100, seed=100))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_raw_line("J\tn1\t+\tn2\t+\t*")
    inputs = b.write("f6_j_line")

    proc, _ = run_build(inputs, check=False)
    assert proc.returncode != 0


# ===========================================================================
# === F7 Boundaries: Step Projection onto Backbone B (5 tests) ===
# ===========================================================================


def test_f7_boundary_node_occurring_twice_in_backbone_not_top_anchor_vs_once_is_anchor(
    tmp_path: Path,
) -> None:
    """F7 Boundary (§3.1-3.2): Node occurring 2x in B is not a top anchor, 1x is an anchor."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=111))
    b.add_node("r_dup", make_complex_seq(80, seed=112))
    b.add_node("n2", make_complex_seq(100, seed=113))
    b.add_node("n3", make_complex_seq(100, seed=114))
    b.add_node("x_ins", make_complex_seq(60, seed=115))
    # r_dup appears twice globally in B (steps 1 and 3), while n1, n2, n3 appear once
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r_dup+", "n2+", "r_dup+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x_ins+", "r_dup+", "n2+", "r_dup+", "n3+"])
    inputs = b.write("f7_dup_vs_unique")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # In D(n1, n2), B(n1, n2) = [r_dup+] where r_dup is unique locally -> x_ins anchored before step 1
    assert bundle.offsets[1]["node_id"] == "r_dup"
    assert bundle.offsets[1]["slot_ins_len"] == 60
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f7_boundary_single_anchor_in_opposite_orientation_projects_rel_minus(
    tmp_path: Path,
) -> None:
    """F7 Boundary: Branch step with sign opposite to B step projects to rel_orient='-'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=116))
    b.add_node("n2", make_complex_seq(80, seed=117))
    b.add_node("n3", make_complex_seq(100, seed=118))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2-", "n3+"])
    inputs = b.write("f7_rel_minus")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 1
    assert bundle.sv_rows[0]["type"] == "INV"
    assert bundle.sv_rows[0]["ref_start"] == 100
    assert bundle.sv_rows[0]["ref_end"] == 180
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f7_boundary_branch_with_zero_projected_anchors_goes_to_unanchored_end(
    tmp_path: Path,
) -> None:
    """F7 Boundary: Branch with 0 anchors in B goes entirely to UNANCHORED_END."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=119))
    b.add_node("n2", make_complex_seq(100, seed=120))
    b.add_node("u1", make_complex_seq(60, seed=121))
    b.add_node("u2", make_complex_seq(60, seed=122))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["u1+", "u2+"])
    inputs = b.write("f7_zero_anchors")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 0
    assert any(r["reason"] == "UNANCHORED_END" and r["length"] == 120 for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f7_boundary_all_negative_backbone_steps_relative_orientation_projection(
    tmp_path: Path,
) -> None:
    """F7 Boundary (§2.1 / T21): Reference steps all '-' project '-' as '+' and '+' as '-'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=123))
    b.add_node("n2", make_complex_seq(90, seed=124))
    b.add_node("n3", make_complex_seq(100, seed=125))
    b.add_node("x1", make_complex_seq(60, seed=126))
    b.set_ref_path("Ref#1#Chr01", ["n1-", "n2-", "n3-"])
    # Alt1 matches '-' signs -> relative forward; Alt2 has n2+ -> relative reverse (INV)
    b.add_path("Alt1#1#Chr01", ["n1-", "x1+", "n2-", "n3-"])
    b.add_path("Alt2#1#Chr01", ["n1-", "n2+", "n3-"])
    inputs = b.write("f7_neg_backbone")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    sv_types = {r["type"] for r in bundle.sv_rows}
    assert "INS" in sv_types
    assert "INV" in sv_types
    assert bundle.inserts[0]["src_strand"] == "+"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f7_boundary_global_duplicate_ref_node_becomes_unique_anchor_in_local_subinterval(
    tmp_path: Path,
) -> None:
    """F7 Boundary: Node appearing 2x globally in B but 1x in B(a,b) is a valid local anchor."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=127))
    b.add_node("r_sh", make_complex_seq(80, seed=128))
    b.add_node("n2", make_complex_seq(100, seed=129))
    b.add_node("n3", make_complex_seq(100, seed=130))
    b.add_node("n4", make_complex_seq(100, seed=131))
    b.add_node("x_after_rsh", make_complex_seq(65, seed=132))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r_sh+", "n2+", "n3+", "r_sh+", "n4+"])
    b.add_path("Alt#1#Chr01", ["n1+", "r_sh+", "x_after_rsh+", "n2+", "n3+", "r_sh+", "n4+"])
    inputs = b.write("f7_local_anchor")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # x_after_rsh is placed in D(r_sh, n2), i.e., before n2 (ref_step 2, ref_start 180)
    assert bundle.offsets[2]["node_id"] == "n2"
    assert bundle.offsets[2]["slot_ins_len"] == 65
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F8 Boundaries: Top-level Whole-Contig Orientation (5 tests) ===
# ===========================================================================


def test_f8_boundary_exact_weight_tie_l_minus_equals_l_plus_does_not_flip(
    tmp_path: Path,
) -> None:
    """F8 Boundary (§3.3.0): Exact tie L_minus == L_plus does NOT flip whole contig."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=141))
    b.add_node("n2", make_complex_seq(100, seed=142))
    b.add_node("x_fwd", make_complex_seq(60, seed=144))
    b.add_node("x_edge", make_complex_seq(60, seed=145))
    # Forward anchors n1+ (100) + n2+ (100) = 200 bp; Reverse anchor n3- (200 bp)
    b.add_node("n3", make_complex_seq(200, seed=143))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x_fwd+", "n2+", "x_edge+", "n3-"])
    inputs = b.write("f8_tie_no_flip")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Contig stays forward -> x_fwd between n1 and n2 has src_strand == '+'
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "+"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f8_boundary_l_minus_exceeds_l_plus_by_1bp_flips_whole_contig(
    tmp_path: Path,
) -> None:
    """F8 Boundary (§3.3.0): L_minus == L_plus + 1 (1 bp heavier reverse) flips whole contig."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=146))
    b.add_node("n2", make_complex_seq(50, seed=147))
    b.add_node("n3", make_complex_seq(51, seed=148))
    b.add_node("x1", make_complex_seq(60, seed=149))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    # L_plus = 100 (n1+), L_minus = 51 + 50 = 101 (n3-, n2-) -> flips whole contig!
    b.add_path("Alt#1#Chr01", ["n3-", "x1+", "n2-", "n1+"])
    inputs = b.write("f8_plus1_flip")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "-"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f8_boundary_l_minus_less_than_l_plus_by_1bp_stays_forward(
    tmp_path: Path,
) -> None:
    """F8 Boundary (§3.3.0): L_minus == L_plus - 1 stays forward (src_strand='+')."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(51, seed=150))
    b.add_node("n2", make_complex_seq(50, seed=151))
    b.add_node("n3", make_complex_seq(100, seed=152))
    b.add_node("x1", make_complex_seq(60, seed=153))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    # L_plus = 51 + 50 = 101, L_minus = 100 (n3-) -> stays forward
    b.add_path("Alt#1#Chr01", ["n1+", "x1+", "n2+", "n3-"])
    inputs = b.write("f8_minus1_stay")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "+"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f8_boundary_whole_contig_orientation_not_applied_in_recursive_subproblems(
    tmp_path: Path,
) -> None:
    """F8 Boundary (§3.3 / §9): Whole-contig orientation only runs at top level, not recursively."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(150, seed=154))
    b.add_node("n2", make_complex_seq(150, seed=155))
    b.add_node("sh", make_complex_seq(120, seed=156))
    b.add_node("b1", make_complex_seq(80, seed=157))
    b.add_node("b2", make_complex_seq(70, seed=158))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Both contigs are top-level forward (n1+ and n2+ = 300 bp forward)
    # Inside D(n1, n2), Alt1 has sh+ -> b1+, Alt2 has b2+ -> sh-
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "b1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "b2+", "sh-", "n2+"])
    inputs = b.write("f8_no_rec_flip")

    proc, bundle = run_build(inputs, extra_args=["--sec-backbone", "longest"])
    assert proc.returncode == 0 and bundle is not None
    # Alt2 is not flipped inside recursion; all inserted pieces keep src_strand == '+'
    assert all(ins["src_strand"] == "+" for ins in bundle.inserts)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f8_boundary_base_weight_dominates_step_count(tmp_path: Path) -> None:
    """F8 Boundary (§3.3.0): 2 long reverse anchors (160 bp) beat 5 short forward anchors (100 bp)."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(1, 6):
        b.add_node(f"s{i}", make_complex_seq(20, seed=160 + i))
    b.add_node("b1", make_complex_seq(80, seed=166))
    b.add_node("b2", make_complex_seq(80, seed=167))
    b.add_node("x_ins", make_complex_seq(60, seed=168))
    b.set_ref_path("Ref#1#Chr01", ["s1+", "s2+", "s3+", "s4+", "s5+", "b1+", "b2+"])
    # 2 reverse steps = 160 bp > 5 forward steps = 100 bp -> whole contig flips
    b.add_path(
        "Alt#1#Chr01",
        ["b2-", "x_ins+", "b1-", "s1+", "s2+", "s3+", "s4+", "s5+"],
    )
    inputs = b.write("f8_weight_vs_count")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert any(ins["src_strand"] == "-" for ins in bundle.inserts)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F9 Boundaries: Reverse Chain Detection & Inversion Normalization (5 tests) ===
# ===========================================================================


def test_f9_boundary_inversion_span_49bp_vs_50bp_threshold(tmp_path: Path) -> None:
    """F9 Boundary (§3.3.2): 49 bp inversion is normalized silently; 50 bp is called as INV."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=171))
    b.add_node("inv49", make_complex_seq(49, seed=172))
    b.add_node("n2", make_complex_seq(100, seed=173))
    b.add_node("inv50", make_complex_seq(50, seed=174))
    b.add_node("n3", make_complex_seq(100, seed=175))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "inv49+", "n2+", "inv50+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "inv49-", "n2+", "inv50-", "n3+"])
    inputs = b.write("f9_49_vs_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    inv_events = [r for r in bundle.sv_rows if r["type"] == "INV"]
    assert len(inv_events) == 1
    assert inv_events[0]["ref_start"] == 249
    assert inv_events[0]["ref_end"] == 299
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f9_boundary_single_anchor_inversion_with_nested_insertion(tmp_path: Path) -> None:
    """F9 Boundary (§3.3.2): Single-anchor reverse chain flanked by L and F is normalized as INV."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=176))
    b.add_node("n2", make_complex_seq(90, seed=177))
    b.add_node("n3", make_complex_seq(100, seed=178))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2-", "n3+"])
    inputs = b.write("f9_single_anchor_inv")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 1
    assert bundle.sv_rows[0]["type"] == "INV"
    assert bundle.genotypes[bundle.sv_rows[0]["event_id"]]["Alt#1"] == "1"
    assert bundle.genotypes[bundle.sv_rows[0]["event_id"]]["Ref#1"] == "0"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f9_boundary_inv_edge_missing_left_flank_vs_missing_right_flank(
    tmp_path: Path,
) -> None:
    """F9 Boundary (§3.3.2 / T24): Reverse chain missing L or missing F is INV_EDGE."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(1, 5):
        b.add_node(f"n{i}", make_complex_seq(100, seed=180 + i))
    b.add_node("x_left", make_complex_seq(60, seed=185))
    b.add_node("x_right", make_complex_seq(65, seed=186))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Alt1: missing L before [n1-] (L_plus = 300 > L_minus = 100)
    b.add_path("Alt1#1#Chr01", ["n1-", "x_left+", "n2+", "n3+", "n4+"])
    # Alt2: missing F after [n4-] (L_plus = 300 > L_minus = 100)
    b.add_path("Alt2#1#Chr01", ["n1+", "n2+", "n3+", "x_right+", "n4-"])
    inputs = b.write("f9_inv_edge")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    edge_events = [r for r in bundle.sv_rows if r["type"] == "INV_EDGE"]
    assert len(edge_events) >= 2
    assert any(r["reason"] == "INV_EDGE" for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f9_boundary_inv_conflict_overlapping_rank_intervals(tmp_path: Path) -> None:
    """F9 Boundary (§3.3.2 / T24): Two reverse chains with overlapping rank intervals -> INV_CONFLICT."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(1, 8):
        b.add_node(f"n{i}", make_complex_seq(100, seed=190 + i))
    b.add_node("x1", make_complex_seq(60, seed=198))
    b.add_node("x2", make_complex_seq(65, seed=199))
    # Forward anchors n1, n4, n6, n7 = 450 bp > reverse anchors n3-, n2-, n5-, n2- = 400 bp
    b.add_node("n7", make_complex_seq(150, seed=197))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+", "n6+", "n7+"])
    # Chain 1: n3- -> x1+ -> n2- (ranks [1, 2]); Chain 2: n5- -> x2+ -> n2- (ranks [1, 4]) -> overlap!
    b.add_path(
        "Alt#1#Chr01",
        ["n1+", "n3-", "x1+", "n2-", "n4+", "n5-", "x2+", "n2-", "n6+", "n7+"],
    )
    inputs = b.write("f9_inv_conflict")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert any(r["type"] == "INV_CONFLICT" for r in bundle.sv_rows)
    conflict_unres = [r for r in bundle.unresolved if r["reason"] == "INV_CONFLICT"]
    assert len(conflict_unres) >= 2
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f9_boundary_inversion_exceeding_max_jump_is_not_blocked(tmp_path: Path) -> None:
    """F9 Boundary (§3.3.2 & §3.7): Inversion > --max-jump is still normalized and internal INS recovered."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(200, seed=201))
    b.add_node("n2", make_complex_seq(150, seed=202))
    b.add_node("n3", make_complex_seq(150, seed=203))
    b.add_node("n4", make_complex_seq(200, seed=204))
    b.add_node("x_in_inv", make_complex_seq(60, seed=205))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Inversion spans n2..n3 (300 bp > max-jump 200); L_plus = 400 > L_minus = 300
    b.add_path("Alt#1#Chr01", ["n1+", "n3-", "x_in_inv+", "n2-", "n4+"])
    inputs = b.write("f9_inv_gt_maxjump")

    proc, bundle = run_build(inputs, extra_args=["--max-jump", "200"])
    assert proc.returncode == 0 and bundle is not None
    types = {r["type"] for r in bundle.sv_rows}
    assert "INV" in types
    assert "INS" in types
    assert "LARGE_JUMP" not in types
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_strand"] == "-"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F10 Boundaries: Weighted LIS & Multi-hit Annotation (5 tests) ===
# ===========================================================================


def test_f10_boundary_equal_weight_lis_tie_broken_by_earlier_source_step(
    tmp_path: Path,
) -> None:
    """F10 Boundary (§3.4.1): Equal-weight LIS tie broken by earlier branch step order."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=211))
    b.add_node("n2", make_complex_seq(80, seed=212))
    b.add_node("n3", make_complex_seq(80, seed=213))
    b.add_node("n4", make_complex_seq(100, seed=214))
    b.add_node("x_ins", make_complex_seq(60, seed=215))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Branch visits n3+ before n2+: chains (n1,n3,n4) and (n1,n2,n4) both have weight 280.
    # Earlier source step n3+ wins main chain -> x_ins is in D(n3, n4) before n4 (ref_step 3).
    b.add_path("Alt#1#Chr01", ["n1+", "n3+", "x_ins+", "n2+", "n4+"])
    inputs = b.write("f10_lis_tie")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[3]["node_id"] == "n4"
    assert bundle.offsets[3]["slot_ins_len"] == 60
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f10_boundary_1bp_weight_difference_overrides_node_count(tmp_path: Path) -> None:
    """F10 Boundary (§3.4.1): 1 bp heavier anchor (101 bp) beats two 50 bp anchors (100 bp)."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=216))
    b.add_node("s1", make_complex_seq(50, seed=217))
    b.add_node("s2", make_complex_seq(50, seed=218))
    b.add_node("big", make_complex_seq(101, seed=219))
    b.add_node("n2", make_complex_seq(100, seed=220))
    b.add_node("x_after_big", make_complex_seq(60, seed=221))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "s1+", "s2+", "big+", "n2+"])
    # Branch visits big+ (101 bp) then s1+ -> s2+ (100 bp): (n1, big, n2) has weight 301 > 300
    b.add_path("Alt#1#Chr01", ["n1+", "big+", "x_after_big+", "s1+", "s2+", "n2+"])
    inputs = b.write("f10_1bp_heavier")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Main chain is n1 -> big -> n2; x_after_big is in D(big, n2) before n2 (ref_step 4)
    assert bundle.offsets[4]["node_id"] == "n2"
    assert bundle.offsets[4]["slot_ins_len"] == 60
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f10_boundary_duplicate_anchor_visit_keeps_first_and_marks_multi_hit(
    tmp_path: Path,
) -> None:
    """F10 Boundary (§3.4.1): Visiting same anchor twice keeps first in LIS and marks multi_hit."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=222))
    b.add_node("n2", make_complex_seq(100, seed=223))
    b.add_node("n3", make_complex_seq(100, seed=224))
    b.add_node("x1", make_complex_seq(60, seed=225))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+", "x1+", "n2+", "n3+"])
    inputs = b.write("f10_multi_hit")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) >= 1
    assert any("multi_hit" in r["alleles"] or "multi_hit" in bundle.log_text for r in bundle.sv_rows)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f10_boundary_strictly_decreasing_forward_anchors_selects_single_heaviest(
    tmp_path: Path,
) -> None:
    """F10 Boundary: Strictly decreasing forward ranks [2, 1, 0] yield 1-node LIS of heaviest anchor."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(80, seed=226))
    b.add_node("n2", make_complex_seq(150, seed=227))
    b.add_node("n3", make_complex_seq(90, seed=228))
    b.add_node("x_left", make_complex_seq(60, seed=229))
    b.add_node("x_right", make_complex_seq(65, seed=230))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n3+", "x_left+", "n2+", "x_right+", "n1+"])
    inputs = b.write("f10_dec_ranks")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Only n2 is in main chain; x_left (before n2) and x_right (after n2) are top-level UNANCHORED_END
    unres_lens = {r["length"] for r in bundle.unresolved if r["reason"] == "UNANCHORED_END"}
    assert 60 in unres_lens and 65 in unres_lens
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f10_boundary_single_anchor_contig_forms_one_element_lis(tmp_path: Path) -> None:
    """F10 Boundary: Contig hitting only 1 reference anchor forms a 1-element LIS chain."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=231))
    b.add_node("n2", make_complex_seq(100, seed=232))
    b.add_node("n3", make_complex_seq(100, seed=233))
    b.add_node("u_pre", make_complex_seq(55, seed=234))
    b.add_node("u_post", make_complex_seq(60, seed=235))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["u_pre+", "n2+", "u_post+"])
    inputs = b.write("f10_single_anchor")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 0
    unres = [r for r in bundle.unresolved if r["reason"] == "UNANCHORED_END"]
    assert len(unres) == 2
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F11 Boundaries: Deviation Segment D(a,b) Extraction (5 tests) ===
# ===========================================================================


def test_f11_boundary_same_node_id_opposite_orientation_creates_deviation(
    tmp_path: Path,
) -> None:
    """F11 Boundary (§3.4.2): Same node IDs as B(a,b) but flipped sign creates D(a,b)."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=241))
    b.add_node("r_dup", make_complex_seq(60, seed=242))
    b.add_node("n2", make_complex_seq(100, seed=243))
    b.add_node("x_new", make_complex_seq(60, seed=244))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r_dup+", "r_dup+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "r_dup+", "x_new+", "r_dup-", "n2+"])
    inputs = b.write("f11_orient_diff")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f11_boundary_empty_branch_and_empty_backbone_produces_no_deviation(
    tmp_path: Path,
) -> None:
    """F11 Boundary: Empty branch slice and empty B(a,b) produces zero deviations and zero SVs."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=245))
    b.add_node("n2", make_complex_seq(100, seed=246))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+"])
    inputs = b.write("f11_empty_both")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 0
    assert len(bundle.inserts) == 0
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f11_boundary_identical_multicopy_reference_traversal_produces_no_deviation(
    tmp_path: Path,
) -> None:
    """F11 Boundary (§3.4.2 / T15): Branch identical to B(a,b) across repeats creates no deviation."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=247))
    b.add_node("r1", make_complex_seq(70, seed=248))
    b.add_node("r2", make_complex_seq(70, seed=249))
    b.add_node("n2", make_complex_seq(100, seed=250))
    steps = ["n1+", "r1+", "r2+", "r1+", "r2+", "n2+"]
    b.set_ref_path("Ref#1#Chr01", steps)
    b.add_path("Alt#1#Chr01", steps)
    inputs = b.write("f11_t15_identical")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 0
    assert bundle.pan_seq == b.spell_steps(steps)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f11_boundary_recursive_deviation_at_logical_start_and_end(
    tmp_path: Path,
) -> None:
    """F11 Boundary (§3.5 / T4): D(START, a) and D(b, END) inside recursion are retained, not unresolved."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=251))
    b.add_node("n2", make_complex_seq(100, seed=252))
    b.add_node("m_shared", make_complex_seq(120, seed=253))
    b.add_node("pre_a", make_complex_seq(60, seed=254))
    b.add_node("post_a", make_complex_seq(65, seed=255))
    b.add_node("pre_b", make_complex_seq(70, seed=256))
    b.add_node("post_b", make_complex_seq(75, seed=257))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "pre_a+", "m_shared+", "post_a+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "pre_b+", "m_shared+", "post_b+", "n2+"])
    inputs = b.write("f11_start_end_dev")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.unresolved) == 0
    for node_id in ("m_shared", "pre_a", "post_a", "pre_b", "post_b"):
        assert b.nodes[node_id] in bundle.pan_seq
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f11_boundary_deviation_with_49bp_novel_node_gets_allele_zero(
    tmp_path: Path,
) -> None:
    """F11 Boundary (§3.6.1 & §5.2): Haplotype with 49 bp (<50) difference gets allele 0."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=258))
    b.add_node("n2", make_complex_seq(100, seed=259))
    b.add_node("x70", make_complex_seq(70, seed=260))
    b.add_node("x49", make_complex_seq(49, seed=261))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "x70+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "x49+", "n2+"])
    inputs = b.write("f11_49bp_allele0")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 1
    ev_id = bundle.sv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "0"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F12 Boundaries: Cycle Folding (Tandem Repeat Compression) (5 tests) ===
# ===========================================================================


def test_f12_boundary_tandem_count_equal_to_max_copy_not_folded(tmp_path: Path) -> None:
    """F12 Boundary (§3.4.3): Repeat count k == max_copy (k=2, max_copy=2) is NOT folded."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=271))
    b.add_node("n2", make_complex_seq(100, seed=272))
    b.add_node("u", make_complex_seq(60, seed=273))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "u+", "u+", "n2+"])
    inputs = b.write("f12_k_eq_maxcopy")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "2"])
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["pan_end"] - bundle.inserts[0]["pan_start"] == 120
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f12_boundary_tandem_count_max_copy_plus_one_folds_and_splits_piece(
    tmp_path: Path,
) -> None:
    """F12 Boundary (§3.4.3): k == max_copy + 1 (k=2, max_copy=1) deletes copy 2 and splits piece."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=274))
    b.add_node("n2", make_complex_seq(100, seed=275))
    b.add_node("u", make_complex_seq(60, seed=276))
    b.add_node("tail", make_complex_seq(60, seed=277))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    inputs = b.write("f12_k_gt_maxcopy")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["src_intervals"] == "100-160"
    assert bundle.inserts[1]["src_intervals"] == "220-280"
    assert bundle.inserts[1]["join_before"] == "synthetic"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f12_boundary_period_containing_global_ref_node_is_not_folded(
    tmp_path: Path,
) -> None:
    """F12 Boundary (§3.4.3): Period containing a global reference node is separated, not cycle-folded."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=278))
    b.add_node("n2", make_complex_seq(100, seed=279))
    b.add_node("r_far", make_complex_seq(30, seed=280))
    b.add_node("n3", make_complex_seq(100, seed=281))
    b.add_node("u", make_complex_seq(60, seed=282))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "r_far+", "n3+"])
    # In D(n1, n2), (u+ -> r_far+) repeated twice: r_far is a top-level reference separator
    b.add_path("Alt#1#Chr01", ["n1+", "u+", "r_far+", "u+", "n2+", "r_far+", "n3+"])
    inputs = b.write("f12_ref_in_period")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f12_boundary_partial_node_slice_not_folded_with_full_node(tmp_path: Path) -> None:
    """F12 Boundary (§3.6.6): Partial-node slice from repeat-guard does not fold with full node."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=283))
    b.add_node("n2", make_complex_seq(100, seed=284))
    u_seq = make_complex_seq(700, seed=285)
    sat_seq = make_satellite_seq(20, 1250, seed=286)
    b.add_node("sat_node", sat_seq)
    b.add_node("u", u_seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Candidate: sat_node (1250) + u (700) + u (700) = 2650 bp
    # Windows: [0, 1000) low, [500, 1500) low -> masks [0, 1500), slicing 250 bp into first u!
    # Unmasked residual is [1500, 2650) = 450 bp partial slice of first u + 700 bp full second u (1150 bp total)
    b.add_path("Alt#1#Chr01", ["n1+", "sat_node+", "u+", "u+", "n2+"])
    inputs = b.write("f12_partial_node")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on", "--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    assert any(r["reason"] == "REPEAT_LIKE" for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f12_boundary_trailing_incomplete_period_preserved_intact(tmp_path: Path) -> None:
    """F12 Boundary (§3.4.3): Trailing half-period after k full periods is preserved intact."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=287))
    b.add_node("n2", make_complex_seq(100, seed=288))
    b.add_node("u1", make_complex_seq(35, seed=289))
    b.add_node("u2", make_complex_seq(35, seed=290))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Period p=2: (u1+, u2+) repeated k=3 times + trailing u1+
    b.add_path(
        "Alt#1#Chr01",
        ["n1+", "u1+", "u2+", "u1+", "u2+", "u1+", "u2+", "u1+", "n2+"],
    )
    inputs = b.write("f12_trailing_half")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    # Kept: first (u1+, u2+) (70 bp) + trailing u1+ (35 bp) = 105 bp across 2 pieces
    assert len(bundle.inserts) == 2
    total_ins = sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bundle.inserts)
    assert total_ins == 105
    assert bundle.inserts[1]["join_before"] == "synthetic"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F13 Boundaries: Canonical Period CNV Counting (5 tests) ===
# ===========================================================================


def test_f13_boundary_cnv_copy_diff_49bp_not_called_vs_50bp_called(tmp_path: Path) -> None:
    """F13 Boundary (§3.4.4 & §3.7): |k_alt - k_ref|*len(U) == 49 bp not called; == 50 bp called."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=301))
    b.add_node("u49", make_complex_seq(49, seed=302))
    b.add_node("n2", make_complex_seq(100, seed=303))
    b.add_node("u50", make_complex_seq(50, seed=304))
    b.add_node("n3", make_complex_seq(100, seed=305))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u49+", "u49+", "n2+", "u50+", "u50+", "n3+"])
    b.add_path(
        "Alt#1#Chr01",
        ["n1+", "u49+", "u49+", "u49+", "n2+", "u50+", "u50+", "u50+", "n3+"],
    )
    inputs = b.write("f13_49_vs_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    cnv_rows = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    assert len(cnv_rows) == 1
    assert cnv_rows[0]["ref_start"] == 298
    assert cnv_rows[0]["ref_end"] == 398
    ev_id = cnv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "2"
    assert bundle.genotypes[ev_id]["Alt#1"] == "3"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f13_boundary_cnv_contraction_k_ref_3_to_k_alt_2_with_50bp_unit(
    tmp_path: Path,
) -> None:
    """F13 Boundary: Copy contraction from k_ref=3 to k_alt=2 with 50 bp unit emits CNV."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=306))
    b.add_node("u50", make_complex_seq(50, seed=307))
    b.add_node("n2", make_complex_seq(100, seed=308))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u50+", "u50+", "u50+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "u50+", "u50+", "n2+"])
    inputs = b.write("f13_contraction")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    cnv_rows = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    assert len(cnv_rows) == 1
    ev_id = cnv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "3"
    assert bundle.genotypes[ev_id]["Alt#1"] == "2"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f13_boundary_three_node_cyclic_rotations_canonicalized_to_same_unit(
    tmp_path: Path,
) -> None:
    """F13 Boundary (§3.4.4): Rotated 3-node periods (A,B,C), (B,C,A), (C,A,B) share canonical(U)."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=309))
    b.add_node("a", make_complex_seq(20, seed=310))
    b.add_node("b", make_complex_seq(20, seed=311))
    b.add_node("c", make_complex_seq(20, seed=312))
    b.add_node("n2", make_complex_seq(100, seed=313))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "a+", "b+", "c+", "a+", "b+", "c+", "n2+"])
    b.add_path(
        "Alt1#1#Chr01",
        ["n1+", "b+", "c+", "a+", "b+", "c+", "a+", "b+", "c+", "a+", "n2+"],
    )
    b.add_path(
        "Alt2#1#Chr01",
        ["n1+", "c+", "a+", "b+", "c+", "a+", "b+", "c+", "a+", "b+", "c+", "a+", "b+", "n2+"],
    )
    inputs = b.write("f13_cyclic_rot")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    cnv_rows = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    assert len(cnv_rows) == 1
    ev_id = cnv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "2"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "3"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "4"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f13_boundary_non_equivalent_reverse_order_units_do_not_call_cnv(
    tmp_path: Path,
) -> None:
    """F13 Boundary (§3.4.4): Non-rotational order (C,B,A) vs (A,B,C) has different canonical(U) -> no CNV."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=314))
    b.add_node("a", make_complex_seq(20, seed=315))
    b.add_node("b", make_complex_seq(20, seed=316))
    b.add_node("c", make_complex_seq(20, seed=317))
    b.add_node("n2", make_complex_seq(100, seed=318))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "a+", "b+", "c+", "a+", "b+", "c+", "n2+"])
    # (c+, b+, a+) is NOT a cyclic rotation of (a+, b+, c+)
    b.add_path(
        "Alt#1#Chr01",
        ["n1+", "c+", "b+", "a+", "c+", "b+", "a+", "c+", "b+", "a+", "n2+"],
    )
    inputs = b.write("f13_non_rot")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert not any(r["type"] == "CNV" for r in bundle.sv_rows)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f13_boundary_cnv_counted_before_cycle_folding_with_missing_hap_dot(
    tmp_path: Path,
) -> None:
    """F13 Boundary (§3.4.4 & §5.2): Haplotype without consistent period at CNV locus gets '.'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=319))
    b.add_node("u", make_complex_seq(30, seed=320))
    b.add_node("v", make_complex_seq(30, seed=321))
    b.add_node("n2", make_complex_seq(100, seed=322))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "v+", "u+", "v+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "v+", "u+", "v+", "u+", "v+", "n2+"])
    # Alt2 has non-periodic arrangement (u+, u+, v+) -> no consistent period with (u+, v+) -> '.'
    b.add_path("Alt2#1#Chr01", ["n1+", "u+", "u+", "v+", "n2+"])
    inputs = b.write("f13_cnv_dot")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    cnv_rows = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    assert len(cnv_rows) == 1
    ev_id = cnv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "2"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "3"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "."
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F14 Boundaries: Subproblem State, Progress Metric & Local Re-anchoring (5 tests) ===
# ===========================================================================


def test_f14_boundary_unique_node_in_b_ab_not_visited_forward_terminates_reanchoring(
    tmp_path: Path,
) -> None:
    """F14 Boundary (§3.5 / T16): Unique node in B(a,b) not visited forward by branch terminates cleanly."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=331))
    b.add_node("r_mid", make_complex_seq(70, seed=332))
    b.add_node("n2", make_complex_seq(100, seed=333))
    b.add_node("n3", make_complex_seq(100, seed=334))
    b.add_node("x_new", make_complex_seq(60, seed=335))
    # r_mid appears twice globally (steps 1 and 3), so it is unique inside B(n1, n2) = [r_mid+]
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r_mid+", "n2+", "r_mid+", "n3+"])
    # Branch does not visit r_mid at all between n1 and n2
    b.add_path("Alt#1#Chr01", ["n1+", "x_new+", "n2+", "r_mid+", "n3+"])
    inputs = b.write("f14_no_fwd_visit")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f14_boundary_reanchoring_interval_fails_to_shrink_terminates_immediately(
    tmp_path: Path,
) -> None:
    """F14 Boundary (§3.5 / T16): When B(a,b) has only multi-copy nodes, local re-anchoring terminates."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=336))
    b.add_node("r_dup", make_complex_seq(60, seed=337))
    b.add_node("n2", make_complex_seq(100, seed=338))
    b.add_node("x1", make_complex_seq(65, seed=339))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r_dup+", "r_dup+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "r_dup+", "x1+", "r_dup+", "n2+"])
    inputs = b.write("f14_t16_no_shrink")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert b.nodes["x1"] in bundle.pan_seq
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f14_boundary_component_recursion_reduces_member_count_from_2_to_1(
    tmp_path: Path,
) -> None:
    """F14 Boundary (§3.5): Component recursion reduces |M| from 2 to 1 and terminates at |M|=1."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=340))
    b.add_node("n2", make_complex_seq(100, seed=341))
    b.add_node("s_common", make_complex_seq(100, seed=342))
    b.add_node("u1", make_complex_seq(60, seed=343))
    b.add_node("u2", make_complex_seq(70, seed=344))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "s_common+", "u1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "s_common+", "u2+", "n2+"])
    inputs = b.write("f14_m2_to_m1")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Secondary backbone is s_common + u2 (170 bp), followed by N*100 + u1 (60 bp) -> 330 bp
    assert bundle.offsets[1]["slot_ins_len"] == 330
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f14_boundary_deep_4_level_local_reanchoring_terminates_without_max_depth(
    tmp_path: Path,
) -> None:
    """F14 Boundary (§3.5): 4-level nested local re-anchoring resolves without any max-depth cutoff."""
    b = SyntheticInputBuilder(tmp_path)
    for name, seed in [
        ("n0", 345), ("l1", 346), ("l2", 347), ("l3", 348), ("l4", 349), ("n1", 350),
        ("d1", 351), ("d2", 352), ("d3", 353), ("d4", 354), ("x_deep", 355),
    ]:
        b.add_node(name, make_complex_seq(60, seed=seed))
    # Construct reference so l1, l2, l3, l4 each appear twice globally (once after n1)
    ref_steps = [
        "n0+", "l1+", "d1+", "l2+", "d2+", "l3+", "d3+", "l4+", "d4+", "n1+",
        "l1+", "l2+", "l3+", "l4+",
    ]
    b.set_ref_path("Ref#1#Chr01", ref_steps)
    alt_steps = [
        "n0+", "l1+", "d1+", "l2+", "d2+", "l3+", "d3+", "l4+", "x_deep+", "d4+", "n1+",
        "l1+", "l2+", "l3+", "l4+",
    ]
    b.add_path("Alt#1#Chr01", alt_steps)
    inputs = b.write("f14_deep4")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[8]["node_id"] == "d4"
    assert bundle.offsets[8]["slot_ins_len"] == 60
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f14_boundary_single_member_component_uses_real_sequence_directly(
    tmp_path: Path,
) -> None:
    """F14 Boundary (§3.6.2): Single-member component (|M|=1) uses real sequence without recursion."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=356))
    b.add_node("n2", make_complex_seq(100, seed=357))
    b.add_node("c1", make_complex_seq(50, seed=358))
    b.add_node("c2", make_complex_seq(50, seed=359))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "c1+", "c2+", "n2+"])
    inputs = b.write("f14_single_member")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["pan_end"] - bundle.inserts[0]["pan_start"] == 100
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F15 Boundaries: Candidate Block Extraction & Unanchored Ends (5 tests) ===
# ===========================================================================


def test_f15_boundary_candidate_length_49bp_discarded_vs_50bp_retained(
    tmp_path: Path,
) -> None:
    """F15 Boundary (§3.5): Candidate of 49 bp (<50) discarded; 50 bp (==50) retained as INS."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=361))
    b.add_node("n2", make_complex_seq(100, seed=362))
    b.add_node("n3", make_complex_seq(100, seed=363))
    b.add_node("c49", make_complex_seq(49, seed=364))
    b.add_node("c50", make_complex_seq(50, seed=365))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "c49+", "n2+", "c50+", "n3+"])
    inputs = b.write("f15_49_vs_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[1]["slot_ins_len"] == 0
    assert bundle.offsets[2]["slot_ins_len"] == 50
    assert len(bundle.sv_rows) == 1
    assert bundle.sv_rows[0]["ref_start"] == 200
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f15_boundary_custom_min_sv_len_99bp_discarded_vs_100bp_retained(
    tmp_path: Path,
) -> None:
    """F15 Boundary: With --min-sv-len 100, 99 bp candidate is discarded and 100 bp is retained."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=366))
    b.add_node("n2", make_complex_seq(100, seed=367))
    b.add_node("n3", make_complex_seq(100, seed=368))
    b.add_node("c99", make_complex_seq(99, seed=369))
    b.add_node("c100", make_complex_seq(100, seed=370))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "c99+", "n2+", "c100+", "n3+"])
    inputs = b.write("f15_99_vs_100")

    proc, bundle = run_build(inputs, extra_args=["--min-sv-len", "100"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[1]["slot_ins_len"] == 0
    assert bundle.offsets[2]["slot_ins_len"] == 100
    assert len(bundle.sv_rows) == 1
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f15_boundary_top_level_unanchored_end_49bp_ignored_vs_50bp_recorded(
    tmp_path: Path,
) -> None:
    """F15 Boundary (§3.5 / T11): Top-level unanchored end 49 bp ignored; 50 bp recorded."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=371))
    b.add_node("n2", make_complex_seq(100, seed=372))
    b.add_node("u49", make_complex_seq(49, seed=373))
    b.add_node("u50", make_complex_seq(50, seed=374))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["u49+", "n1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "n2+", "u50+"])
    inputs = b.write("f15_unanch_49_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    unres = [r for r in bundle.unresolved if r["reason"] == "UNANCHORED_END"]
    assert len(unres) == 1
    assert unres[0]["length"] == 50
    assert unres[0]["exported_fasta_id"] != "."
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f15_boundary_top_level_ref_node_inside_deviation_splits_candidates(
    tmp_path: Path,
) -> None:
    """F15 Boundary (§3.5 / T9 / T19): Top-level reference node acts as strict separator in D(a,b)."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=375))
    b.add_node("n2", make_complex_seq(100, seed=376))
    b.add_node("r_ref", make_complex_seq(40, seed=377))
    b.add_node("n3", make_complex_seq(100, seed=378))
    b.add_node("x1", make_complex_seq(60, seed=379))
    b.add_node("x2", make_complex_seq(65, seed=380))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "r_ref+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x1+", "r_ref+", "x2+", "n2+", "r_ref+", "n3+"])
    inputs = b.write("f15_ref_sep")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # x1 (60) and x2 (65) are separated by r_ref -> never stitched; r_ref appears only once in pan.fa
    assert bundle.pan_seq.count(b.nodes["r_ref"]) == 1
    assert len(bundle.inserts) == 2
    assert {int(r["pan_end"]) - int(r["pan_start"]) for r in bundle.inserts} == {60, 65}
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f15_boundary_repeat_residual_annotated_when_multicopy_local_node_embedded(
    tmp_path: Path,
) -> None:
    """F15 Boundary (§3.5): Candidate containing multi-copy local backbone node marks repeat_residual."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=381))
    b.add_node("n2", make_complex_seq(100, seed=382))
    b.add_node("u0", make_complex_seq(100, seed=380))
    b.add_node("u1_a", make_complex_seq(60, seed=385))
    b.add_node("m_dup", make_complex_seq(40, seed=383))
    b.add_node("s_tail", make_complex_seq(80, seed=384))
    b.add_node("nov_z", make_complex_seq(60, seed=386))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Two carriers share u0+ -> u1_a+ -> m_dup+ -> m_dup+ -> s_tail+ so it becomes B_sec
    b.add_path("Alt1a#1#Chr01", ["n1+", "u0+", "u1_a+", "m_dup+", "m_dup+", "s_tail+", "n2+"])
    b.add_path("Alt1b#1#Chr01", ["n1+", "u0+", "u1_a+", "m_dup+", "m_dup+", "s_tail+", "n2+"])
    # Alt2 has u1_a+ -> m_dup+ -> nov_z+ -> m_dup+ -> s_tail+: inside D(u1_a, s_tail), m_dup+ is multi-copy -> repeat_residual
    b.add_path("Alt2#1#Chr01", ["n1+", "u1_a+", "m_dup+", "nov_z+", "m_dup+", "s_tail+", "n2+"])
    inputs = b.write("f15_rep_residual")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "2", "--sec-backbone", "support"])
    assert proc.returncode == 0 and bundle is not None
    assert (
        "repeat_residual" in bundle.sv_rows[0]["alleles"]
        or "repeat_residual" in bundle.log_text
    )
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F16 Boundaries: Parallel Branch Groups G(a,b) & Component Recursion (5 tests) ===
# ===========================================================================


def test_f16_boundary_sec_backbone_support_tie_broken_by_length_then_token_then_src(
    tmp_path: Path,
) -> None:
    """F16 Boundary (§3.6.2): Equal support and equal length tie broken deterministically by token ASC."""
    for order_idx, hap_order in enumerate([("Alt1#1#Chr01", "Alt2#1#Chr01"), ("Alt2#1#Chr01", "Alt1#1#Chr01")]):
        b = SyntheticInputBuilder(tmp_path / f"ord_{order_idx}")
        b.add_node("n1", make_complex_seq(100, seed=391))
        b.add_node("n2", make_complex_seq(100, seed=392))
        b.add_node("sh", make_complex_seq(100, seed=393))
        b.add_node("a_tok", make_complex_seq(60, seed=394))
        b.add_node("b_tok", make_complex_seq(60, seed=395))
        b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
        paths = {
            "Alt1#1#Chr01": ["n1+", "sh+", "a_tok+", "n2+"],
            "Alt2#1#Chr01": ["n1+", "sh+", "b_tok+", "n2+"],
        }
        for h in hap_order:
            b.add_path(h, paths[h])
        inputs = b.write(f"f16_tie_{order_idx}")
        proc, bundle = run_build(inputs, extra_args=["--sec-backbone", "support"])
        assert proc.returncode == 0 and bundle is not None
        # sh+ -> a_tok+ must be chosen as secondary backbone ahead of b_tok+ in both input orders
        pos_a = bundle.pan_seq.index(b.nodes["a_tok"])
        pos_b = bundle.pan_seq.index(b.nodes["b_tok"])
        assert pos_a < pos_b
        assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f16_boundary_sec_backbone_longest_selects_longer_over_higher_support(
    tmp_path: Path,
) -> None:
    """F16 Boundary (§3.6.2): --sec-backbone longest picks 1-carrier long path over 3-carrier short path."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=396))
    b.add_node("n2", make_complex_seq(100, seed=397))
    b.add_node("sh", make_complex_seq(100, seed=398))
    b.add_node("short_b", make_complex_seq(80, seed=399))
    b.add_node("long_b", make_complex_seq(200, seed=400))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "short_b+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "short_b+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "sh+", "short_b+", "n2+"])
    b.add_path("Alt4#1#Chr01", ["n1+", "sh+", "long_b+", "n2+"])

    in_sup = b.write("f16_sup")
    _, bun_sup = run_build(in_sup, extra_args=["--sec-backbone", "support"], out_prefix=tmp_path / "sup_out")
    _, bun_lng = run_build(in_sup, extra_args=["--sec-backbone", "longest"], out_prefix=tmp_path / "lng_out")
    assert bun_sup is not None and bun_lng is not None

    # In support mode, short_b comes immediately after sh; in longest mode, long_b comes immediately after sh
    assert bun_sup.pan_seq.index(b.nodes["short_b"]) < bun_sup.pan_seq.index(b.nodes["long_b"])
    assert bun_lng.pan_seq.index(b.nodes["long_b"]) < bun_lng.pan_seq.index(b.nodes["short_b"])
    assert_all_global_invariants(bun_sup, in_sup.gfa_path, in_sup.ref_fa_path)
    assert_all_global_invariants(bun_lng, in_sup.gfa_path, in_sup.ref_fa_path)


def test_f16_boundary_shared_node_in_opposite_orientation_unioned_into_same_component(
    tmp_path: Path,
) -> None:
    """F16 Boundary (§3.6.2 / T18): Shared node ID in opposite signs (sh+ vs sh-) unions into 1 component."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=401))
    b.add_node("n2", make_complex_seq(100, seed=402))
    b.add_node("sh", make_complex_seq(100, seed=403))
    b.add_node("tail1", make_complex_seq(70, seed=404))
    b.add_node("tail2", make_complex_seq(80, seed=405))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh-", "tail1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh-", "tail2+", "n2+"])
    inputs = b.write("f16_opp_orient_uf")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # sh- is shared in one component -> appears only once in pan.fa (not duplicated as two components)
    assert bundle.pan_seq.count(revcomp(b.nodes["sh"])) == 1
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f16_boundary_shared_partial_node_slice_does_not_union_components(
    tmp_path: Path,
) -> None:
    """F16 Boundary (§3.6.6): Partial-node slice from repeat-guard does NOT union candidates in UF."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=406))
    b.add_node("n2", make_complex_seq(100, seed=407))
    # Node sh_mixed: 1450 bp satellite + 50 bp complex tail = 1500 bp
    # Paired with unique 100 bp suffix -> total candidate = 1600 bp
    # Windows [0, 1000) and [500, 1500) are low (1450 bp satellite in [500, 1500) has 950 sat + 50 complex -> low!)
    # Window [600, 1600) has 850 sat + 150 complex -> wait: make sh_mixed = 1400 bp sat + 100 bp complex = 1500 bp,
    # and unique suffix = 400 bp complex -> total 1900 bp!
    # In 1900 bp: [0, 1000) is pure sat (low), [500, 1500) has 900 sat + 100 complex (100/986=0.101 < 0.30 -> low!),
    # [1000, 1900) tail window [900, 1900) has 500 sat + 500 complex (500/986=0.507 >= 0.30 -> non-low!).
    # Masked interval is [0, 1500) -> Wait: we want the cut at 1500 to fall INSIDE sh_mixed!
    # So make sh_mixed = 1400 bp sat + 160 bp complex = 1560 bp, and unique suffix = 340 bp complex (total 1900 bp).
    # Then [500, 1500) has 900 sat + 100 complex (low), [900, 1900) has 500 sat + 500 complex (non-low).
    # Masked interval is [0, 1500), which cuts 60 bp before the end of sh_mixed (1560 bp)!
    sh_mixed_seq = make_satellite_seq(20, 1400, seed=408) + make_complex_seq(160, seed=409)
    b.add_node("sh_mixed", sh_mixed_seq)
    b.add_node("uniq1", make_complex_seq(340, seed=410))
    b.add_node("uniq2", make_complex_seq(340, seed=411))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh_mixed+", "uniq1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh_mixed+", "uniq2+", "n2+"])
    inputs = b.write("f16_partial_no_uf")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    # Both residuals are 400 bp (60 bp partial slice of sh_mixed + 340 bp uniq1/uniq2)
    # Because sh_mixed is partial, they are NOT unioned into a single component -> both 400 bp residuals emitted!
    assert bundle.offsets[1]["slot_ins_len"] == 400 + 100 + 400
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f16_boundary_exact_step_path_deduplication_before_component_partitioning(
    tmp_path: Path,
) -> None:
    """F16 Boundary (§3.6.2): Identical directed step paths across 4 haplotypes deduplicate to 1 path."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=412))
    b.add_node("n2", make_complex_seq(100, seed=413))
    b.add_node("x1", make_complex_seq(60, seed=414))
    b.add_node("x2", make_complex_seq(60, seed=415))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    for i in range(1, 5):
        b.add_path(f"Alt{i}#1#Chr01", ["n1+", "x1+", "x2+", "n2+"])
    inputs = b.write("f16_exact_dedup")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.inserts) == 1
    assert bundle.sv_rows[0]["n_carrier"] == 4
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F17 Boundaries: Similarity Clustering (edlib <=1kb / mappy >1kb) (5 tests) ===
# ===========================================================================


def test_f17_boundary_engine_switch_1000bp_edlib_vs_1001bp_mappy(tmp_path: Path) -> None:
    """F17 Boundary (§3.6.3): <=1000 bp uses edlib and >1000 bp (1001 bp) uses mappy with same criteria."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=421))
    b.add_node("n2", make_complex_seq(100, seed=422))
    b.add_node("n3", make_complex_seq(100, seed=423))
    seq_1000 = make_complex_seq(1000, seed=424)
    seq_1000_snp = mutate_seq_snps(seq_1000, num_snps=10, seed=425)
    seq_1001 = make_complex_seq(1001, seed=426)
    seq_1001_snp = mutate_seq_snps(seq_1001, num_snps=10, seed=427)
    b.add_node("r1000", seq_1000)
    b.add_node("s1000", seq_1000_snp)
    b.add_node("r1001", seq_1001)
    b.add_node("s1001", seq_1001_snp)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1000+", "n2+", "r1001+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "s1000+", "n2+", "s1001+", "n3+"])
    inputs = b.write("f17_1000_vs_1001")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Both loci merge the 99% similar sequences into 1 representative each
    assert bundle.offsets[1]["slot_ins_len"] == 1000
    assert bundle.offsets[2]["slot_ins_len"] == 1001
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f17_boundary_criterion2_id_gc_below_vs_above_cluster_id_095(
    tmp_path: Path,
) -> None:
    """F17 Boundary (§3.6.3): ID_gc = 189/200 = 0.945 (<0.95) not merged; 191/200 = 0.955 (>=0.95) merged."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=428))
    b.add_node("n2", make_complex_seq(100, seed=429))
    b.add_node("n3", make_complex_seq(100, seed=430))
    base_a = make_complex_seq(200, seed=431)
    alt_a_low = mutate_seq_snps(base_a, num_snps=11, seed=432)  # 189/200 = 0.945 < 0.95
    base_b = make_complex_seq(200, seed=433)
    alt_b_high = mutate_seq_snps(base_b, num_snps=9, seed=434)  # 191/200 = 0.955 >= 0.95
    b.add_node("ra", base_a)
    b.add_node("sa", alt_a_low)
    b.add_node("rb", base_b)
    b.add_node("sb", alt_b_high)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ra+", "n2+", "rb+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sa+", "n2+", "sb+", "n3+"])
    inputs = b.write("f17_crit2_095")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Locus 1 (n1..n2): not merged -> 200 + 100 + 200 = 500 bp; Locus 2 (n2..n3): merged -> 200 bp
    assert bundle.offsets[1]["slot_ins_len"] == 500
    assert bundle.offsets[2]["slot_ins_len"] == 200
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f17_boundary_criterion3_unaligned_plus_internal_ins_49bp_merged_vs_50bp_unmerged(
    tmp_path: Path,
) -> None:
    """F17 Boundary (§3.6.3 / T12): S internal insertion of 49 bp (<50) merges; 50 bp (==50) does not merge."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=435))
    b.add_node("n2", make_complex_seq(100, seed=436))
    b.add_node("n3", make_complex_seq(100, seed=437))
    core1 = make_complex_seq(400, seed=438)
    s_ins49 = core1[:200] + make_complex_seq(49, seed=439) + core1[200:]  # 449 bp
    core2 = make_complex_seq(400, seed=440)
    s_ins50 = core2[:200] + make_complex_seq(50, seed=441) + core2[200:]  # 450 bp
    # Put longer sequence as R or test S against longer R:
    # Since real sequences are sorted by length DESC, make R longer (460 bp) so S (449/450 bp) aligns to R
    # Or make S shorter than R by adding a 49 bp / 50 bp insertion in S while R has a 60 bp tail elsewhere:
    # Better yet: R is 450 bp (`core1 + tail50`), S is 400 bp (`core1`) -> in S, unaligned_ends + internal_ins == 0!
    # Wait: Criterion 3 checks `unaligned_ends(S) + internal_ins_bp(S) < min-sv-len` where S is the shorter/subsequent candidate compared against existing representative R!
    # So make R = 500 bp, and S = R[:200] + novel_ins(49 or 50 bp) + R[250:] (so len(S) = 449 or 450 bp <= len(R) = 500 bp)!
    r1 = make_complex_seq(500, seed=438)
    s1_49 = r1[:200] + make_complex_seq(49, seed=439) + r1[260:]  # len = 200 + 49 + 240 = 489 < 500
    r2 = make_complex_seq(500, seed=440)
    s2_50 = r2[:200] + make_complex_seq(50, seed=441) + r2[260:]  # len = 200 + 50 + 240 = 490 < 500
    # Wait: if r1[200:260] is replaced by 49 bp, edlib might align the 49 bp as ~12 matches + 37 mismatches + 11 D!
    # To make S have a pure internal insertion of 49 bp vs 50 bp relative to R while len(S) <= len(R),
    # let R have higher support (2 haplotypes carry R of length 400 bp, 1 haplotype carries S of length 449/450 bp)
    # Wait: in §3.6.2 step 4, real sequences are compared in descending length order ("其余真实序列按长度降序，依次与已有代表比较")!
    # So if R has length 500 bp (`flank_L(250) + flank_R(250)`), and S has length 499 bp (`flank_L(225) + ins_49(49) + flank_R(225)` where S starts 25 bp inside R and ends 25 bp before R's end, so S has 0 unaligned ends and a pure 49 bp internal insertion against R!):
    flank_l1 = make_complex_seq(250, seed=438)
    flank_r1 = make_complex_seq(250, seed=439)
    r_seq1 = flank_l1 + flank_r1  # 500 bp
    s_seq1_49 = flank_l1[25:] + make_complex_seq(49, seed=440) + flank_r1[:-25]  # 225 + 49 + 225 = 499 bp < 500 bp!

    flank_l2 = make_complex_seq(250, seed=441)
    flank_r2 = make_complex_seq(250, seed=442)
    r_seq2 = flank_l2 + flank_r2  # 500 bp
    s_seq2_50 = flank_l2[26:] + make_complex_seq(50, seed=443) + flank_r2[:-25]  # 224 + 50 + 225 = 499 bp < 500 bp!

    b.add_node("r1", r_seq1)
    b.add_node("s1", s_seq1_49)
    b.add_node("r2", r_seq2)
    b.add_node("s2", s_seq2_50)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+", "r2+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "s1+", "n2+", "s2+", "n3+"])
    inputs = b.write("f17_crit3_49_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Locus 1 (49 bp internal ins in S): merged into r1 (500 bp)
    # Locus 2 (50 bp internal ins in S): NOT merged -> 500 + 100 + 499 = 1099 bp
    assert bundle.offsets[1]["slot_ins_len"] == 500
    assert bundle.offsets[2]["slot_ins_len"] == 1099
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f17_boundary_criterion4_sliding_50bp_window_34_matches_unmerged_vs_36_matches_merged(
    tmp_path: Path,
) -> None:
    """F17 Boundary (§3.6.3 / T12): 50 bp window with 34/50=0.68 (<0.70) fails; 36/50=0.72 (>=0.70) merges."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=444))
    b.add_node("n2", make_complex_seq(100, seed=445))
    b.add_node("n3", make_complex_seq(100, seed=446))

    r_a = make_complex_seq(400, seed=447)
    # 16 SNPs inside [150, 200) -> window [150, 200) has 34/50 = 0.68 < 0.70 (while ID_gc = 384/400 = 0.96 >= 0.95)
    pos_16 = [150 + i * 3 for i in range(16)]
    s_a_fail = mutate_seq_snps(r_a, positions=pos_16, seed=448)

    r_b = make_complex_seq(400, seed=449)
    # 14 SNPs inside [150, 200) -> worst 50 bp window has 36/50 = 0.72 >= 0.70 (and ID_gc = 386/400 = 0.965 >= 0.95)
    pos_14 = [150 + i * 3 for i in range(14)]
    s_b_pass = mutate_seq_snps(r_b, positions=pos_14, seed=450)

    b.add_node("ra", r_a)
    b.add_node("sa", s_a_fail)
    b.add_node("rb", r_b)
    b.add_node("sb", s_b_pass)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "ra+", "n2+", "rb+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sa+", "n2+", "sb+", "n3+"])
    inputs = b.write("f17_crit4_win")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Locus 1 fails Criterion 4 -> 400 + 100 + 400 = 900 bp; Locus 2 passes -> 400 bp
    assert bundle.offsets[1]["slot_ins_len"] == 900
    assert bundle.offsets[2]["slot_ins_len"] == 400
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f17_boundary_synthetic_sequence_never_similarity_clustered_with_real_sequence(
    tmp_path: Path,
) -> None:
    """F17 Boundary (§3.6.2): Synthetic sequence from multi-member component only exact-deduplicates."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=451))
    b.add_node("n2", make_complex_seq(100, seed=452))
    b.add_node("sh", make_complex_seq(100, seed=453))
    b.add_node("b1", make_complex_seq(100, seed=454))
    b.add_node("b2", make_complex_seq(60, seed=455))
    # Component 1 has 2 members (sh+ -> b1+ and sh+ -> b2+) -> produces a synthetic sequence.
    # Component 2 is a 1-member real sequence 99.5% identical to sh+ -> b1+ (1 SNP).
    real_99 = mutate_seq_snps(b.nodes["sh"] + b.nodes["b1"], positions=[50], seed=456)
    b.add_node("real_near", real_99)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "b1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "b2+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "real_near+", "n2+"])
    inputs = b.write("f17_synth_no_cluster")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Both the synthetic component (200 + 100 + 60 = 360 bp) and real_near (200 bp) are kept: 360 + 100 + 200 = 660 bp
    assert bundle.offsets[1]["slot_ins_len"] == 660
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F18 Boundaries: Unified Output Block Layout & Join Annotation (5 tests) ===
# ===========================================================================


def test_f18_boundary_representative_ordering_support_then_length_then_token(
    tmp_path: Path,
) -> None:
    """F18 Boundary (§3.6.4): Representatives ordered by support DESC, length DESC, token ASC."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=461))
    b.add_node("n2", make_complex_seq(100, seed=462))
    b.add_node("rep_sup2", make_complex_seq(60, seed=463))
    b.add_node("rep_long", make_complex_seq(90, seed=464))
    b.add_node("a_tok", make_complex_seq(70, seed=465))
    b.add_node("b_tok", make_complex_seq(70, seed=466))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "b_tok+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "a_tok+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "rep_long+", "n2+"])
    b.add_path("Alt4#1#Chr01", ["n1+", "rep_sup2+", "n2+"])
    b.add_path("Alt5#1#Chr01", ["n1+", "rep_sup2+", "n2+"])
    inputs = b.write("f18_rep_order")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    p_sup2 = bundle.pan_seq.index(b.nodes["rep_sup2"])
    p_long = bundle.pan_seq.index(b.nodes["rep_long"])
    p_atok = bundle.pan_seq.index(b.nodes["a_tok"])
    p_btok = bundle.pan_seq.index(b.nodes["b_tok"])
    assert p_sup2 < p_long < p_atok < p_btok
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f18_boundary_multiple_groups_at_same_right_anchor_ordered_by_span_asc_then_a_asc(
    tmp_path: Path,
) -> None:
    """F18 Boundary (§3.6.4 / T20): Multi-groups at right anchor b ordered by span (b-a) ASC, then a ASC."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(4):
        b.add_node(f"n{i}", make_complex_seq(100, seed=470 + i))
    b.add_node("g_span1", make_complex_seq(60, seed=475))
    b.add_node("g_span2", make_complex_seq(65, seed=476))
    b.add_node("g_span3", make_complex_seq(70, seed=477))
    b.set_ref_path("Ref#1#Chr01", ["n0+", "n1+", "n2+", "n3+"])
    # All 3 groups have right anchor n3 (step 3):
    b.add_path("Alt1#1#Chr01", ["n0+", "g_span3+", "n3+"])          # G(n0, n3): span = 3
    b.add_path("Alt2#1#Chr01", ["n0+", "n1+", "g_span2+", "n3+"])   # G(n1, n3): span = 2
    b.add_path("Alt3#1#Chr01", ["n0+", "n1+", "n2+", "g_span1+", "n3+"])  # G(n2, n3): span = 1
    inputs = b.write("f18_span_order")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    idx1 = bundle.pan_seq.index(b.nodes["g_span1"])
    idx2 = bundle.pan_seq.index(b.nodes["g_span2"])
    idx3 = bundle.pan_seq.index(b.nodes["g_span3"])
    assert idx1 < idx2 < idx3
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f18_boundary_replacement_group_keeps_b_ab_first_and_inserts_alt_before_b(
    tmp_path: Path,
) -> None:
    """F18 Boundary (§3.6.4): When B(a,b) is non-empty, B(a,b) stays in place and R2 inserts before b."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=478))
    b.add_node("n2", make_complex_seq(80, seed=479))
    b.add_node("n3", make_complex_seq(100, seed=480))
    b.add_node("alt_rep", make_complex_seq(70, seed=481))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "alt_rep+", "n3+"])
    inputs = b.write("f18_replacement_layout")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    # Layout must be: n1 (100) + n2 (80) + N*100 + alt_rep (70) + n3 (100)
    expected = (
        b.nodes["n1"]
        + b.nodes["n2"]
        + ("N" * 100)
        + b.nodes["alt_rep"]
        + b.nodes["n3"]
    )
    assert bundle.pan_seq == expected
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f18_boundary_custom_spacer_n_1_vs_250_exact_spacer_count(tmp_path: Path) -> None:
    """F18 Boundary: --spacer-n 1 and --spacer-n 250 insert exact N counts."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=482))
    b.add_node("n2", make_complex_seq(100, seed=483))
    b.add_node("r1", make_complex_seq(60, seed=484))
    b.add_node("r2", make_complex_seq(70, seed=485))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "r2+", "n2+"])
    inputs = b.write("f18_custom_spacer")

    for sp_len in (1, 250):
        out_p = tmp_path / f"sp_{sp_len}"
        proc, bundle = run_build(inputs, extra_args=["--spacer-n", str(sp_len)], out_prefix=out_p)
        assert proc.returncode == 0 and bundle is not None
        assert bundle.offsets[1]["slot_ins_len"] == 60 + sp_len + 70
        assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path, spacer_n=sp_len)


def test_f18_boundary_join_before_source_contiguous_vs_synthetic_vs_spacer(
    tmp_path: Path,
) -> None:
    """F18 Boundary (§3.6.5): Verify source_contiguous, synthetic, and spacer in inserts.tsv."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=486))
    b.add_node("n2", make_complex_seq(100, seed=487))
    b.add_node("u", make_complex_seq(60, seed=488))
    b.add_node("w", make_complex_seq(60, seed=489))
    b.add_node("other", make_complex_seq(70, seed=490))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Alt1 has u+ -> u+ -> w+ (120 bp after folding, 2 pieces: source_contiguous then synthetic)
    # Alt2 has other+ (70 bp, 1 piece after spacer: join_before='spacer')
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "u+", "w+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "other+", "n2+"])
    inputs = b.write("f18_all_joins")

    proc, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert proc.returncode == 0 and bundle is not None
    joins = [r["join_before"] for r in bundle.inserts]
    assert joins == ["source_contiguous", "synthetic", "spacer"]
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F19 Boundaries: Universal Repeat Guard (--repeat-guard on|off) (6 tests) ===
# ===========================================================================


def test_f19_boundary_499bp_satellite_unmasked_vs_500bp_satellite_masked(
    tmp_path: Path,
) -> None:
    """F19 Boundary (§3.6.6 & §3.8.4-C): 499 bp satellite skips repeat-guard; 500 bp is masked."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=501))
    b.add_node("n2", make_complex_seq(100, seed=502))
    b.add_node("n3", make_complex_seq(100, seed=503))
    b.add_node("sat499", make_satellite_seq(20, 499, seed=504))
    b.add_node("sat500", make_satellite_seq(20, 500, seed=505))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "sat499+", "n2+", "sat500+", "n3+"])
    inputs = b.write("f19_499_vs_500")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    # 499 bp is retained at step 1; 500 bp is masked as REPEAT_LIKE at step 2
    assert bundle.offsets[1]["slot_ins_len"] == 499
    assert bundle.offsets[2]["slot_ins_len"] == 0
    rep_unres = [r for r in bundle.unresolved if r["reason"] == "REPEAT_LIKE"]
    assert len(rep_unres) == 1
    assert rep_unres[0]["length"] == 500
    assert rep_unres[0]["exported_fasta_id"] == "."
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f19_boundary_999bp_and_1000bp_single_window_vs_1001bp_two_windows(
    tmp_path: Path,
) -> None:
    """F19 Boundary (C11 / C19): 999 bp and 1000 bp use 1 window; 1001 bp uses 2 windows ([0,1000) & [1,1001))."""
    b = SyntheticInputBuilder(tmp_path)
    for i in range(1, 5):
        b.add_node(f"n{i}", make_complex_seq(100, seed=510 + i))
    b.add_node("sat999", make_satellite_seq(20, 999, seed=515))
    b.add_node("sat1000", make_satellite_seq(20, 1000, seed=516))
    # Construct 1001 bp sequence where window [0, 1000) has 149/986 (<0.30, low) or wait:
    # In [0, 1000), total_pos = 986, 0.30 * 986 = 295.8!
    # So if [0, 1000) has 295 singletons (295/986 = 0.29918 < 0.30 -> low) and the 1001st base
    # adds 1 singleton while removing 1 non-singleton at pos 0, then window [1, 1001) has
    # 296/986 = 0.30020 >= 0.30 -> non-low!
    # Thus in 1001 bp, window 0 is low and window 1 is non-low -> only 1 low window -> NOT masked!
    w1000_295 = make_window_with_singleton_ratio(1000, target_ratio=295 / 986, seed=517)
    status0, ratio0 = compute_canonical_15mer_singleton_ratio(w1000_295)
    assert status0 == "low"
    # Append 1 base so that the new trailing 15-mer is unique, making [1, 1001) have 296 singletons >= 0.30
    chosen_tail = "A"
    for base in ("A", "C", "G", "T"):
        cand1001 = w1000_295 + base
        st1, _ = compute_canonical_15mer_singleton_ratio(cand1001[1:1001])
        if st1 == "non-low":
            chosen_tail = base
            break
    seq_1001_one_low = w1000_295 + chosen_tail
    assert oracle_repeat_mask_intervals(seq_1001_one_low) == []

    b.add_node("cand1001", seq_1001_one_low)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt#1#Chr01", ["n1+", "sat999+", "n2+", "sat1000+", "n3+", "cand1001+", "n4+"])
    inputs = b.write("f19_999_1000_1001")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    # sat999 and sat1000 are completely masked (1 window each, low); cand1001 is NOT masked (only 1 of 2 windows low)
    assert bundle.offsets[1]["slot_ins_len"] == 0
    assert bundle.offsets[2]["slot_ins_len"] == 0
    assert bundle.offsets[3]["slot_ins_len"] == 1001
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f19_boundary_singleton_ratio_strictly_below_030_vs_equal_030(
    tmp_path: Path,
) -> None:
    """F19 Boundary (C10): singleton_ratio = 149/500 = 0.298 (<0.30) masked; 150/500 = 0.300 (==0.30) kept."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=521))
    b.add_node("n2", make_complex_seq(100, seed=522))
    b.add_node("n3", make_complex_seq(100, seed=523))

    seq_below = make_window_with_singleton_ratio(514, target_ratio=149 / 500, seed=524)
    st_below, r_below = compute_canonical_15mer_singleton_ratio(seq_below)
    assert st_below == "low" and r_below is not None and r_below < 0.30

    seq_equal = make_window_with_singleton_ratio(514, target_ratio=150 / 500, seed=525)
    st_equal, r_equal = compute_canonical_15mer_singleton_ratio(seq_equal)
    assert st_equal == "non-low" and r_equal == 0.30

    b.add_node("w_low", seq_below)
    b.add_node("w_eq", seq_equal)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "w_low+", "n2+", "w_eq+", "n3+"])
    inputs = b.write("f19_c10_030")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[1]["slot_ins_len"] == 0
    assert bundle.offsets[2]["slot_ins_len"] == 514
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f19_boundary_long_candidate_single_isolated_low_window_not_masked_vs_two_adjacent_low_masked(
    tmp_path: Path,
) -> None:
    """F19 Boundary (C19): In >1000 bp, 1 isolated low window is NOT masked; 2 adjacent low windows ARE masked."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=531))
    b.add_node("n2", make_complex_seq(100, seed=532))
    b.add_node("n3", make_complex_seq(100, seed=533))

    # 2000 bp with 1 isolated low window at [500, 1500):
    # [0, 500) complex, [500, 1500) satellite (low), [1500, 2000) complex -> windows [0,1000) non-low, [500,1500) low, [1000,2000) non-low
    seq_iso = (
        make_complex_seq(500, seed=534)
        + make_satellite_seq(20, 1000, seed=535)
        + make_complex_seq(500, seed=536)
    )
    assert oracle_repeat_mask_intervals(seq_iso) == []

    # 2000 bp with 2 adjacent low windows [0, 1000) and [500, 1500):
    seq_two = make_satellite_seq(20, 1500, seed=537) + make_complex_seq(500, seed=538)
    assert oracle_repeat_mask_intervals(seq_two) == [(0, 1500)]

    b.add_node("c_iso", seq_iso)
    b.add_node("c_two", seq_two)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "c_iso+", "n2+", "c_two+", "n3+"])
    inputs = b.write("f19_c19_iso_vs_two")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[1]["slot_ins_len"] == 2000
    assert bundle.offsets[2]["slot_ins_len"] == 500
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f19_boundary_unknown_window_with_over_10pct_n_breaks_adjacent_low_run(
    tmp_path: Path,
) -> None:
    """F19 Boundary (C19): Window with <90% valid 15-mers is 'unknown' and breaks adjacent low windows."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=541))
    b.add_node("n2", make_complex_seq(100, seed=542))
    # Construct 2000 bp where:
    # [0, 900) is satellite, [900, 1000) is satellite (so [0, 1000) has 0 N -> low),
    # [1000, 1120) is 120 'N's (so [500, 1500) has 120 'N's -> >10% invalid -> unknown!),
    # [1120, 2000) is complex (so [1000, 2000) is unknown).
    # Wait: even better, construct 2500 bp where:
    # [0, 1000) is pure satellite (low),
    # [1000, 1500) has 120 'N's at [1000, 1120) + 380 complex bases (so [500, 1500) has 120 N -> unknown, and [1000, 2000) has 120 N -> unknown),
    # [1500, 2500) is pure satellite (so [1500, 2500) is low!).
    # Then the windows are: [0,1000)=low, [500,1500)=unknown, [1000,2000)=unknown, [1500,2500)=low -> 0 adjacent low pairs!
    seq_unk = (
        make_satellite_seq(20, 1000, seed=543)
        + ("N" * 120)
        + make_complex_seq(380, seed=544)
        + make_satellite_seq(20, 1000, seed=545)
    )
    assert len(seq_unk) == 2500
    assert oracle_repeat_mask_intervals(seq_unk) == []

    b.add_node("c_unk", seq_unk)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "c_unk+", "n2+"])
    inputs = b.write("f19_c19_unknown_break")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.offsets[1]["slot_ins_len"] == 2500
    assert not any(r["reason"] == "REPEAT_LIKE" for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f19_boundary_unmasked_residual_49bp_discarded_vs_50bp_retained(
    tmp_path: Path,
) -> None:
    """F19 Boundary (§3.6.6): Unmasked normal residual of 49 bp (<50) is discarded; 50 bp (==50) retained."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=546))
    b.add_node("n2", make_complex_seq(100, seed=547))
    b.add_node("n3", make_complex_seq(100, seed=548))

    # Construct 1549 bp and 1550 bp where [0, 1000) and [500, 1500) are low (masking [0, 1500)),
    # and the right-aligned tail window [549, 1549) / [550, 1550) is unknown (via 120 'N's in [1500, ...)? No,
    # if we put 275 complex bases at [1225, 1500) and 49/50 complex bases at [1500, 1549/1550),
    # then [500, 1500) has ~275/986 = 0.278 < 0.30 (low!), while [549, 1549) has ~(275+49)/986 = 0.328 >= 0.30 (non-low!)).
    sat_prefix = make_satellite_seq(20, 1225, seed=549)
    comp_mid = make_complex_seq(275, seed=550)
    tail_49 = make_complex_seq(49, seed=551)
    tail_50 = make_complex_seq(50, seed=552)

    cand_1549 = sat_prefix + comp_mid + tail_49
    cand_1550 = sat_prefix + comp_mid + tail_50
    assert oracle_repeat_mask_intervals(cand_1549) == [(0, 1500)]
    assert oracle_repeat_mask_intervals(cand_1550) == [(0, 1500)]

    b.add_node("c1549", cand_1549)
    b.add_node("c1550", cand_1550)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "c1549+", "n2+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "n2+", "c1550+", "n3+"])
    inputs = b.write("f19_res_49_vs_50")

    proc, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert proc.returncode == 0 and bundle is not None
    # At step 1 (n1..n2): 49 bp residual is discarded -> slot_ins_len == 0, Alt1#1 genotype is '.'
    # At step 2 (n2..n3): 50 bp residual is retained -> slot_ins_len == 50, sv.tsv has partial_content=1
    assert bundle.offsets[1]["slot_ins_len"] == 0
    assert bundle.offsets[2]["slot_ins_len"] == 50
    ins_events = [r for r in bundle.sv_rows if r["type"] == "INS"]
    assert len(ins_events) == 1
    assert "partial_content=1" in ins_events[0]["alleles"]
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F20 Boundaries: Top-level SV Event Calling & Genotyping (5 tests) ===
# ===========================================================================


def test_f20_boundary_max_jump_exact_threshold_equal_vs_plus_one(tmp_path: Path) -> None:
    """F20 Boundary (§3.7): Main-chain span == max_jump is DEL/INS; == max_jump + 1 is LARGE_JUMP."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=561))
    b.add_node("skip500", make_complex_seq(500, seed=562))
    b.add_node("n2", make_complex_seq(100, seed=563))
    b.add_node("skip501", make_complex_seq(501, seed=564))
    b.add_node("n3", make_complex_seq(100, seed=565))
    b.add_node("x_ok", make_complex_seq(60, seed=566))
    b.add_node("x_jump", make_complex_seq(65, seed=567))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "skip500+", "n2+", "skip501+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "x_ok+", "n2+", "x_jump+", "n3+"])
    inputs = b.write("f20_max_jump")

    proc, bundle = run_build(inputs, extra_args=["--max-jump", "500"])
    assert proc.returncode == 0 and bundle is not None
    # Locus 1 (span 500 == max_jump): DEL and INS called, x_ok inserted before n2
    # Locus 2 (span 501 > max_jump): LARGE_JUMP called, x_jump routed to unresolved.tsv
    assert bundle.offsets[2]["slot_ins_len"] == 160
    assert bundle.inserts[0]["pan_end"] - bundle.inserts[0]["pan_start"] == 60
    assert bundle.offsets[4]["slot_ins_len"] == 0
    types = {r["type"] for r in bundle.sv_rows}
    assert "DEL" in types and "INS" in types and "LARGE_JUMP" in types
    assert any(r["reason"] == "LARGE_JUMP" and r["length"] == 65 for r in bundle.unresolved)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f20_boundary_del_length_49bp_not_called_vs_50bp_called(tmp_path: Path) -> None:
    """F20 Boundary (§3.7 / T13): Skipped reference of 49 bp (<50) not called; 50 bp called as DEL."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=568))
    b.add_node("d49", make_complex_seq(49, seed=569))
    b.add_node("n2", make_complex_seq(100, seed=570))
    b.add_node("d50", make_complex_seq(50, seed=571))
    b.add_node("n3", make_complex_seq(100, seed=572))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "d49+", "n2+", "d50+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+", "n3+"])
    inputs = b.write("f20_del_49_50")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    del_rows = [r for r in bundle.sv_rows if r["type"] == "DEL"]
    assert len(del_rows) == 1
    assert del_rows[0]["ref_start"] == 249
    assert del_rows[0]["ref_end"] == 299
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f20_boundary_conflicting_w_line_contigs_for_same_haplotype_yield_dot_genotype(
    tmp_path: Path,
) -> None:
    """F20 Boundary (§5.2): Two contigs of same haplotype giving contradictory calls yield '.'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=573))
    b.add_node("n2", make_complex_seq(100, seed=574))
    b.add_node("x_ins", make_complex_seq(70, seed=575))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_walk("Alt1", "1", "ctgA", 0, 200, ["n1+", "n2+"])
    b.add_walk("Alt1", "1", "ctgB", 0, 270, ["n1+", "x_ins+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "x_ins+", "n2+"])
    inputs = b.write("f20_conflict_gt")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    ev_id = bundle.sv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "."
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f20_boundary_contig_ending_at_left_anchor_without_right_anchor_gets_dot_genotype(
    tmp_path: Path,
) -> None:
    """F20 Boundary (§5.2 / T11): Contig covering only left anchor a (not b) gets '.' instead of '0'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=576))
    b.add_node("n2", make_complex_seq(100, seed=577))
    b.add_node("x_ins", make_complex_seq(60, seed=578))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "x_ins+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+"])
    inputs = b.write("f20_partial_gt_dot")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None
    ev_id = bundle.sv_rows[0]["event_id"]
    assert bundle.genotypes[ev_id]["Ref#1"] == "0"
    assert bundle.genotypes[ev_id]["Alt1#1"] == "1"
    assert bundle.genotypes[ev_id]["Alt2#1"] == "."
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f20_boundary_dup_tra_distance_exceeding_max_jump_calls_dup_tra_and_ins(
    tmp_path: Path,
) -> None:
    """F20 Boundary (§3.7 / T9): Off-chain anchor > max-jump from a calls DUP_TRA and inserts novel seq."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=579))
    b.add_node("n2", make_complex_seq(100, seed=580))
    b.add_node("n3", make_complex_seq(300, seed=581))
    b.add_node("d_far", make_complex_seq(60, seed=582))
    b.add_node("n4", make_complex_seq(100, seed=583))
    b.add_node("x_new", make_complex_seq(65, seed=584))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "d_far+", "n4+"])
    # In D(n1, n2) (span 0 <= max-jump 200), branch visits d_far (distance > 200 from n1) + x_new
    b.add_path("Alt#1#Chr01", ["n1+", "d_far+", "x_new+", "n2+", "n3+", "d_far+", "n4+"])
    inputs = b.write("f20_dup_tra")

    proc, bundle = run_build(inputs, extra_args=["--max-jump", "200"])
    assert proc.returncode == 0 and bundle is not None
    types = {r["type"] for r in bundle.sv_rows}
    assert "DUP_TRA" in types
    assert "INS" in types
    assert bundle.pan_seq.count(b.nodes["d_far"]) == 1
    assert b.nodes["x_new"] in bundle.pan_seq
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F21 Boundaries: Centromere Mode CLI & BED Anchor Delimitation (5 tests) ===
# ===========================================================================


def test_f21_negative_centromere_bed_with_explicit_mode_off_fails(tmp_path: Path) -> None:
    """F21 Negative (§3.8.1): Passing --centromere-bed with explicit --centromere-mode off fails."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=591))
    b.add_node("c", make_complex_seq(200, seed=592))
    b.add_node("v", make_complex_seq(100, seed=593))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f21_bed_mode_off")

    proc, _ = run_build(inputs, extra_args=["--centromere-mode", "off"], check=False)
    assert proc.returncode != 0


def test_f21_negative_centromere_mode_without_bed_fails(tmp_path: Path) -> None:
    """F21 Negative (§3.8.1): Passing bypass or extract-islands without --centromere-bed fails."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=594))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    inputs = b.write("f21_mode_no_bed")

    p_byp, _ = run_build(inputs, extra_args=["--centromere-mode", "bypass"], include_bed=False, check=False)
    p_ext, _ = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"], include_bed=False, check=False)
    assert p_byp.returncode != 0
    assert p_ext.returncode != 0


def test_f21_negative_centromere_min_island_when_mode_off_or_below_min_sv_len_fails(
    tmp_path: Path,
) -> None:
    """F21 Negative (§6): Passing --centromere-min-island when mode=off or < --min-sv-len fails."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=595))
    b.add_node("c", make_complex_seq(200, seed=596))
    b.add_node("v", make_complex_seq(100, seed=597))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f21_min_island_err")

    # 1) Explicit --centromere-min-island without centromere mode / BED
    p1, _ = run_build(
        inputs,
        extra_args=["--centromere-min-island", "500"],
        include_bed=False,
        check=False,
    )
    assert p1.returncode != 0

    # 2) --centromere-min-island 40 < --min-sv-len 50
    p2, _ = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "40", "--min-sv-len", "50"],
        check=False,
    )
    assert p2.returncode != 0


def test_f21_negative_invalid_bed_chrom_or_coordinates_or_empty_fails(
    tmp_path: Path,
) -> None:
    """F21 Negative (§3.8.1): Invalid BED chrom, start >= end, end > ref_len, or empty BED fails."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=598))
    b.add_node("c", make_complex_seq(200, seed=599))
    b.add_node("v", make_complex_seq(100, seed=600))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    inputs = b.write("f21_bad_bed")

    bad_beds = [
        "WrongChrom\t120\t280\tcen1\n",
        "Chr01\t280\t120\tcen1\n",
        "Chr01\t120\t9999\tcen1\n",
        "",
    ]
    for idx, content in enumerate(bad_beds):
        bed_file = tmp_path / f"bad_{idx}.bed"
        bed_file.write_text(content, encoding="utf-8")
        proc, _ = run_build(
            inputs,
            extra_args=["--centromere-bed", bed_file, "--centromere-mode", "bypass"],
            include_bed=False,
            check=False,
        )
        assert proc.returncode != 0


def test_f21_boundary_cen_anchor_not_found_downgrades_to_bypass_and_whole_chrom(
    tmp_path: Path,
) -> None:
    """F21 Boundary (§3.8.1 / C5): Missing unique u and v uses virtual 0..L and downgrades to bypass."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("r1", make_complex_seq(100, seed=601))
    b.add_node("r2", make_complex_seq(100, seed=602))
    # r1 and r2 each appear twice -> 0 unique reference anchors on entire chromosome
    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r1+", "r2+"])
    b.add_path("Alt#1#Chr01", ["r1+", "r2+", "r1+", "r2+"])
    b.add_bed_region(120, 280, "cen_all_dup")
    inputs = b.write("f21_no_anchors")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    for row in bundle.centromere_rows:
        assert row["mode"] == "bypass"
        assert row["ref_start"] == 0 and row["ref_end"] == 400
        if row["hap"] == "Alt#1":
            assert row["status"] in ("partial", "unmapped")
            assert row["length_ratio"] == "."
    assert "CEN_ANCHOR_NOT_FOUND" in bundle.log_text or "CEN_ANCHOR_NOT_FOUND" in bundle.summary_text
    assert (
        "expanded_to_whole_chromosome" in bundle.log_text
        or "expanded_to_whole_chromosome" in bundle.summary_text
    )
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F22 Boundaries: Centromere Branch Completeness & Length Ratio (5 tests) ===
# ===========================================================================


def test_f22_boundary_l_path_zero_when_u_directly_followed_by_v(tmp_path: Path) -> None:
    """F22 Boundary (§3.8.2): Sample path u+ -> v+ with 0 internal steps has L_path=0 and ratio=0.0."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=611))
    b.add_node("c", make_complex_seq(200, seed=612))
    b.add_node("v", make_complex_seq(100, seed=613))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f22_lpath_zero")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    alt_row = next(r for r in bundle.centromere_rows if r["hap"] == "Alt#1")
    assert alt_row["status"] == "complete"
    assert int(alt_row["L_ref"]) == 200
    assert int(alt_row["L_path"]) == 0
    assert float(alt_row["length_ratio"]) == 0.0
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f22_boundary_ambiguous_priority_over_complete_for_same_haplotype(
    tmp_path: Path,
) -> None:
    """F22 Boundary (§3.8.2 / C4): Multiple traversals for same haplotype prioritize status='ambiguous'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=614))
    b.add_node("c", make_complex_seq(200, seed=615))
    b.add_node("v", make_complex_seq(100, seed=616))
    b.add_node("alt_c", make_complex_seq(250, seed=617))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_walk("Alt", "1", "ctg1", 0, 400, ["u+", "c+", "v+"])
    b.add_walk("Alt", "1", "ctg2", 0, 450, ["u+", "alt_c+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f22_ambig_prio")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    alt_row = next(r for r in bundle.centromere_rows if r["hap"] == "Alt#1")
    assert alt_row["status"] == "ambiguous"
    assert alt_row["L_path"] == "."
    assert alt_row["length_ratio"] == "."
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f22_boundary_gapped_sample_reports_l_path_but_dot_length_ratio(
    tmp_path: Path,
) -> None:
    """F22 Boundary (§3.8.2 / C4): Sample with N inside u..v is 'gapped', reports L_path, ratio='.'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=618))
    b.add_node("c", make_complex_seq(200, seed=619))
    b.add_node("v", make_complex_seq(100, seed=620))
    b.add_node("c_gap", make_complex_seq(100, seed=621) + "NNNNN" + make_complex_seq(195, seed=622))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "c_gap+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f22_sample_gap")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    alt_row = next(r for r in bundle.centromere_rows if r["hap"] == "Alt#1")
    assert alt_row["status"] == "gapped"
    assert alt_row["ref_status"] == "complete"
    assert int(alt_row["L_path"]) == 300
    assert alt_row["length_ratio"] == "."
    assert any(r["reason"] == "CEN_INCOMPLETE" for r in bundle.unresolved)
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f22_boundary_gapped_reference_reports_l_path_but_dot_length_ratio_and_cen_reference_gap(
    tmp_path: Path,
) -> None:
    """F22 Boundary (§3.8.2 & §3.8.4): Reference with N inside u..v sets ref_status='gapped' and ratio='.'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=623))
    b.add_node("c_ref_n", make_complex_seq(90, seed=624) + "NNNNNNNNNN" + make_complex_seq(100, seed=625))
    b.add_node("v", make_complex_seq(100, seed=626))
    b.add_node("island", make_complex_seq(600, seed=627))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c_ref_n+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "island+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f22_ref_gap")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    alt_row = next(r for r in bundle.centromere_rows if r["hap"] == "Alt#1")
    assert alt_row["status"] == "complete"
    assert alt_row["ref_status"] == "gapped"
    assert int(alt_row["L_path"]) == 600
    assert alt_row["length_ratio"] == "."
    assert any(r["reason"] == "CEN_REFERENCE_GAP" for r in bundle.unresolved)
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f22_boundary_touching_and_overlapping_bed_intervals_merged_into_single_region(
    tmp_path: Path,
) -> None:
    """F22 Boundary (§3.8.1 / C5): Touching BED intervals merge into a single region after u/v expansion."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=628))
    b.add_node("r1", make_complex_seq(100, seed=629))
    b.add_node("r2", make_complex_seq(100, seed=630))
    b.add_node("v", make_complex_seq(100, seed=631))
    b.set_ref_path("Ref#1#Chr01", ["u+", "r1+", "r1+", "r2+", "r2+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "r1+", "r2+", "v+"])
    # Touching BED intervals [120, 300) and [300, 480)
    b.add_bed_region(120, 300, "bed_part1")
    b.add_bed_region(300, 480, "bed_part2")
    inputs = b.write("f22_bed_merge")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    region_ids = {r["region_id"] for r in bundle.centromere_rows}
    assert len(region_ids) == 1
    row0 = bundle.centromere_rows[0]
    assert "bed_part1" in row0["input_bed_ids"] and "bed_part2" in row0["input_bed_ids"]
    assert int(row0["ref_start"]) == 100 and int(row0["ref_end"]) == 500
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F23 Boundaries: Centromere bypass Mode (5 tests) ===
# ===========================================================================


def test_f23_boundary_bypass_nonref_block_49bp_vs_50bp_unresolved_threshold(
    tmp_path: Path,
) -> None:
    """F23 Boundary (§3.8.3): In bypass, non-ref block >=50 bp is recorded as CEN_BYPASS; 49 bp is not."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=641))
    b.add_node("c", make_complex_seq(200, seed=642))
    b.add_node("v", make_complex_seq(100, seed=643))
    b.add_node("x49", make_complex_seq(49, seed=644))
    b.add_node("x50", make_complex_seq(50, seed=645))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "x49+", "c+", "v+"])
    b.add_path("Alt2#1#Chr01", ["u+", "x50+", "c+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f23_49_vs_50")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert proc.returncode == 0 and bundle is not None
    byp_rows = [r for r in bundle.unresolved if r["reason"] == "CEN_BYPASS"]
    assert len(byp_rows) == 1
    assert byp_rows[0]["length"] == 50
    assert byp_rows[0]["exported_fasta_id"] == "."
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f23_boundary_bypass_multiple_nonref_blocks_separated_by_ref_node_not_stitched(
    tmp_path: Path,
) -> None:
    """F23 Boundary (§3.8.3 & §5.5): Disjoint non-ref blocks in bypass are recorded as separate rows."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=646))
    b.add_node("c", make_complex_seq(200, seed=647))
    b.add_node("v", make_complex_seq(100, seed=648))
    b.add_node("x1", make_complex_seq(60, seed=649))
    b.add_node("x2", make_complex_seq(70, seed=650))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "x1+", "c+", "x2+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f23_two_blocks")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert proc.returncode == 0 and bundle is not None
    byp_rows = [r for r in bundle.unresolved if r["reason"] == "CEN_BYPASS"]
    assert len(byp_rows) == 2
    assert {(r["src_start"], r["src_end"]) for r in byp_rows} == {(100, 160), (360, 430)}
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f23_boundary_bypass_does_not_create_cen_island_or_internal_sv_events(
    tmp_path: Path,
) -> None:
    """F23 Boundary (§3.8.3 / C2): bypass mode emits 0 SV events even with >500 kb or complex blocks."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=651))
    b.add_node("c1", make_complex_seq(150, seed=652))
    b.add_node("c2", make_complex_seq(150, seed=653))
    b.add_node("v", make_complex_seq(100, seed=654))
    b.add_node("big_ins", make_complex_seq(800, seed=655))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c1+", "c2+", "v+"])
    # Alt has inversion of c1 and deletion of c2 plus 800 bp insertion inside centromere
    b.add_path("Alt#1#Chr01", ["u+", "c1-", "big_ins+", "v+"])
    b.add_bed_region(120, 380, "cen1")
    inputs = b.write("f23_no_sv")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass", "--max-jump", "200"])
    assert proc.returncode == 0 and bundle is not None
    assert len(bundle.sv_rows) == 0
    assert bundle.pan_seq == b.spell_steps(["u+", "c1+", "c2+", "v+"])
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f23_boundary_bypass_reverse_oriented_contig_preserves_negative_src_strand(
    tmp_path: Path,
) -> None:
    """F23 Boundary (§3.8.2-3.8.3): Reverse-assembled contig in bypass records src_strand='-'."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(150, seed=656))
    b.add_node("c", make_complex_seq(200, seed=657))
    b.add_node("v", make_complex_seq(150, seed=658))
    b.add_node("x_cen", make_complex_seq(80, seed=659))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    # Reverse contig: v- (150) -> x_cen+ (80) -> u- (150)
    b.add_path("Alt#1#Chr01", ["v-", "x_cen+", "u-"])
    b.add_bed_region(160, 340, "cen1")
    inputs = b.write("f23_rev_bypass")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "bypass"])
    assert proc.returncode == 0 and bundle is not None
    byp_rows = [r for r in bundle.unresolved if r["reason"] == "CEN_BYPASS"]
    assert len(byp_rows) == 1
    assert byp_rows[0]["src_strand"] == "-"
    assert byp_rows[0]["src_start"] == 150 and byp_rows[0]["src_end"] == 230
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f23_boundary_bypass_dump_repeat_sequences_exports_and_deduplicates_fasta(
    tmp_path: Path,
) -> None:
    """F23 Boundary (§5.5 / C18): --dump-repeat-sequences exports CEN_BYPASS to unresolved.fa without changing pan.fa."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=660))
    b.add_node("c", make_complex_seq(200, seed=661))
    b.add_node("v", make_complex_seq(100, seed=662))
    b.add_node("x_byp", make_complex_seq(120, seed=663))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "x_byp+", "c+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f23_dump_byp")

    _, bun_off = run_build(
        inputs, extra_args=["--centromere-mode", "bypass"], out_prefix=tmp_path / "byp_off"
    )
    _, bun_on = run_build(
        inputs,
        extra_args=["--centromere-mode", "bypass", "--dump-repeat-sequences"],
        out_prefix=tmp_path / "byp_on",
    )
    assert bun_off is not None and bun_on is not None
    assert bun_off.pan_seq == bun_on.pan_seq
    assert bun_off.unresolved[0]["exported_fasta_id"] == "."
    assert len(bun_off.unresolved_fa) == 0
    assert bun_on.unresolved[0]["exported_fasta_id"] != "."
    assert len(bun_on.unresolved_fa) == 1
    assert_all_global_invariants(bun_on, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F24 Boundaries: Centromere extract-islands Mode (Steps A-D) (6 tests) ===
# ===========================================================================


def test_f24_boundary_step_a_and_c_499bp_short_residual_vs_500bp_cen_island(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.4 / C11): 499 bp (<500) is CEN_SHORT_RESIDUAL; 500 bp (==500) is CEN_ISLAND."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=671))
    b.add_node("c", make_complex_seq(200, seed=672))
    b.add_node("v", make_complex_seq(100, seed=673))
    b.add_node("isl499", make_complex_seq(499, seed=674))
    b.add_node("isl500", make_complex_seq(500, seed=675))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "isl499+", "v+"])
    b.add_path("Alt2#1#Chr01", ["u+", "isl500+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f24_499_vs_500")

    proc, bundle = run_build(
        inputs, extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"]
    )
    assert proc.returncode == 0 and bundle is not None
    cen_events = [r for r in bundle.sv_rows if r["type"] == "CEN_ISLAND"]
    assert len(cen_events) == 1
    assert bundle.offsets[2]["slot_ins_len"] == 100 + 500 + 100
    assert bundle.centromere_rows is not None
    row1 = next(r for r in bundle.centromere_rows if r["hap"] == "Alt1#1")
    row2 = next(r for r in bundle.centromere_rows if r["hap"] == "Alt2#1")
    assert int(row1["short_bp"]) == 499 and int(row1["accepted_source_bp"]) == 0
    assert int(row2["accepted_source_bp"]) == 500 and int(row2["short_bp"]) == 0
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f24_boundary_residual_49bp_counted_in_short_bp_but_not_exported_to_fasta(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.5 & §5.4): 49 bp (<min-sv-len) residual after Step C is counted in short_bp, not FASTA."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=676))
    b.add_node("c", make_complex_seq(200, seed=677))
    b.add_node("v", make_complex_seq(100, seed=678))
    # Construct 1549 bp candidate where [0, 1500) is masked by Step C and [1500, 1549) is a 49 bp unmasked residual
    cand_1549 = (
        make_satellite_seq(20, 1225, seed=679)
        + make_complex_seq(275, seed=680)
        + make_complex_seq(49, seed=681)
    )
    assert oracle_repeat_mask_intervals(cand_1549) == [(0, 1500)]
    b.add_node("c1549", cand_1549)
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "c1549+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f24_49bp_short")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    alt_row = next(r for r in bundle.centromere_rows if r["hap"] == "Alt#1")
    assert int(alt_row["nonref_input_bp"]) == 1549
    assert int(alt_row["repeat_like_bp"]) == 1500
    assert int(alt_row["short_bp"]) == 49
    assert int(alt_row["accepted_source_bp"]) == 0
    assert len(bundle.unresolved_fa) == 0
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f24_boundary_step_b_mappy_50bp_block_identity_and_reverse_homology(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.4-B / C9): Reverse-complement reference-like block is subtracted by Step B."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=682))
    ref_cen_seq = make_complex_seq(600, seed=683)
    b.add_node("c_ref", ref_cen_seq)
    b.add_node("v", make_complex_seq(100, seed=684))
    # Alt1 has exact reverse-complement of c_ref (different node ID so Step A extracts it, Step B subtracts it)
    b.add_node("c_rev_hom", revcomp(ref_cen_seq))
    b.add_node("c_novel", make_complex_seq(600, seed=685))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c_ref+", "v+"])
    b.add_path("Alt1#1#Chr01", ["u+", "c_rev_hom+", "v+"])
    b.add_path("Alt2#1#Chr01", ["u+", "c_novel+", "v+"])
    b.add_bed_region(120, 680, "cen1")
    inputs = b.write("f24_rev_sub")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.centromere_rows is not None
    r1 = next(r for r in bundle.centromere_rows if r["hap"] == "Alt1#1")
    r2 = next(r for r in bundle.centromere_rows if r["hap"] == "Alt2#1")
    assert int(r1["reference_like_bp"]) == 600 and int(r1["accepted_source_bp"]) == 0
    assert int(r2["reference_like_bp"]) == 0 and int(r2["accepted_source_bp"]) == 600
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f24_boundary_step_b_internal_query_insertion_protected_from_masking(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.4-B / C8): >=50 bp (600 bp) query insertion flanked by ref homology is protected."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=686))
    flank_l = make_complex_seq(300, seed=687)
    flank_r = make_complex_seq(300, seed=688)
    novel_isl = make_complex_seq(600, seed=689)
    b.add_node("c_ref", flank_l + flank_r)
    b.add_node("v", make_complex_seq(100, seed=690))
    b.add_node("cand_c8", flank_l + novel_isl + flank_r)
    b.set_ref_path("Ref#1#Chr01", ["u+", "c_ref+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "cand_c8+", "v+"])
    b.add_bed_region(120, 680, "cen1")
    inputs = b.write("f24_c8_protect")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    cen_events = [r for r in bundle.sv_rows if r["type"] == "CEN_ISLAND"]
    assert len(cen_events) == 1
    assert novel_isl in bundle.pan_seq
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f24_boundary_ambiguous_base_in_candidate_routed_to_cen_sequence_uncertain(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.4-C / C11): Residual containing IUPAC ambiguous base 'R' goes to CEN_SEQUENCE_UNCERTAIN."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=691))
    b.add_node("c", make_complex_seq(200, seed=692))
    b.add_node("v", make_complex_seq(100, seed=693))
    ambig_seq = make_complex_seq(300, seed=694) + "R" + make_complex_seq(299, seed=695)
    b.add_node("c_ambig", ambig_seq)
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "c_ambig+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f24_c11_ambig")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert not any(r["type"] == "CEN_ISLAND" for r in bundle.sv_rows)
    assert any(r["reason"] == "CEN_SEQUENCE_UNCERTAIN" for r in bundle.unresolved)
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f24_boundary_zero_retained_islands_emits_no_leading_or_trailing_spacers(
    tmp_path: Path,
) -> None:
    """F24 Boundary (§3.8.4-D): When 0 CEN_ISLAND representatives remain, 0 spacers are added at ref_start(v)."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=696))
    b.add_node("c", make_complex_seq(200, seed=697))
    b.add_node("v", make_complex_seq(100, seed=698))
    b.add_node("sat_only", make_satellite_seq(20, 1000, seed=699))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "sat_only+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f24_zero_islands")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None
    assert bundle.pan_seq == b.spell_steps(["u+", "c+", "v+"])
    assert bundle.offsets[2]["slot_ins_len"] == 0
    assert_centromere_accounting_and_offsets(bundle)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# === F25 Boundaries: Deterministic Reference GFF3 Liftover (--ref-gff) (5 tests) ===
# ===========================================================================


def test_f25_boundary_insertion_at_exact_left_exterior_boundary_vs_1bp_inside_cds(
    tmp_path: Path,
) -> None:
    """F25 Boundary (§4): Insertion at s-1|s is exterior (not split); at s|s+1 is interior (1 bp split)."""
    # 1) Exterior left boundary at 0-based 100 (1-based feature [101, 280])
    b1 = SyntheticInputBuilder(tmp_path / "ext_left")
    b1.add_node("n1", make_complex_seq(100, seed=701))
    cds_seq = make_coding_seq(60, seed=702)  # 180 bp
    b1.add_node("n2", cds_seq)
    b1.add_node("n3", make_complex_seq(100, seed=703))
    b1.add_node("x_ins", make_complex_seq(60, seed=704))
    b1.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b1.add_path("Alt#1#Chr01", ["n1+", "x_ins+", "n2+", "n3+"])
    b1.add_gff_gene("g1", "t1", "+", [(101, 280)], [(101, 280, 0)])
    in1 = b1.write("f25_ext_left")

    p1, bun1 = run_build(in1)
    assert p1.returncode == 0 and bun1 is not None and bun1.lifted_gff_rows is not None
    cds_rows1 = [r for r in bun1.lifted_gff_rows if r["type"] == "CDS"]
    assert len(cds_rows1) == 1
    assert cds_rows1[0]["start"] == 161 and cds_rows1[0]["end"] == 340
    exon_rows1 = [r for r in bun1.lifted_gff_rows if r["type"] == "exon"]
    assert exon_rows1[0]["attributes"].get("splice_site_proximal_sv") == "True"
    assert_gff3_liftover_integrity(in1.ref_fa_path, in1.gff_path, bun1.pan_fa_path, bun1.lifted_gff_path)

    # 2) 1 bp inside left boundary (0-based 101, splitting 180 bp CDS into 1 bp + 179 bp)
    b2 = SyntheticInputBuilder(tmp_path / "int_1bp")
    b2.add_node("n1", make_complex_seq(100, seed=701))
    b2.add_node("n2_1bp", cds_seq[:1])
    b2.add_node("n2_rest", cds_seq[1:])
    b2.add_node("n3", make_complex_seq(100, seed=703))
    b2.add_node("x_ins", make_complex_seq(60, seed=704))
    b2.set_ref_path("Ref#1#Chr01", ["n1+", "n2_1bp+", "n2_rest+", "n3+"])
    b2.add_path("Alt#1#Chr01", ["n1+", "n2_1bp+", "x_ins+", "n2_rest+", "n3+"])
    b2.add_gff_gene("g1", "t1", "+", [(101, 280)], [(101, 280, 0)])
    in2 = b2.write("f25_int_1bp")

    p2, bun2 = run_build(in2)
    assert p2.returncode == 0 and bun2 is not None and bun2.lifted_gff_rows is not None
    cds_rows2 = [r for r in bun2.lifted_gff_rows if r["type"] == "CDS"]
    assert len(cds_rows2) == 2
    assert (cds_rows2[0]["start"], cds_rows2[0]["end"], cds_rows2[0]["phase"]) == (101, 101, "0")
    assert (cds_rows2[1]["start"], cds_rows2[1]["end"], cds_rows2[1]["phase"]) == (162, 340, "2")
    assert cds_rows2[0]["attributes"].get("projection_split") == "True"
    assert cds_rows2[0]["attributes"].get("sv_status") == "cds_split_by_insertion"
    assert_gff3_liftover_integrity(in2.ref_fa_path, in2.gff_path, bun2.pan_fa_path, bun2.lifted_gff_path)


def test_f25_boundary_insertion_at_exact_right_exterior_boundary_not_split(
    tmp_path: Path,
) -> None:
    """F25 Boundary (§4): Insertion at exact right exterior boundary e|e+1 does not split feature."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=705))
    cds_seq = make_coding_seq(60, seed=706)  # 180 bp -> 1-based [101, 280]
    b.add_node("n2", cds_seq)
    b.add_node("n3", make_complex_seq(100, seed=707))
    b.add_node("x_ins", make_complex_seq(60, seed=708))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+", "x_ins+", "n3+"])
    b.add_gff_gene("g1", "t1", "+", [(101, 280)], [(101, 280, 0)])
    inputs = b.write("f25_ext_right")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None and bundle.lifted_gff_rows is not None
    cds_rows = [r for r in bundle.lifted_gff_rows if r["type"] == "CDS"]
    assert len(cds_rows) == 1
    assert cds_rows[0]["start"] == 101 and cds_rows[0]["end"] == 280
    exon_rows = [r for r in bundle.lifted_gff_rows if r["type"] == "exon"]
    assert exon_rows[0]["attributes"].get("splice_site_proximal_sv") == "True"
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )


def test_f25_boundary_splice_window_8bp_inclusive_vs_9bp_exclusive(tmp_path: Path) -> None:
    """F25 Boundary (§4): Insertion at distance 8 bp from exon gets splice_site_proximal_sv=True; 9 bp does not."""
    # Exon starts at 1-based 101 (0-based 100).
    # In case 8bp, insertion is at 0-based 92 (distance = 100 - 92 = 8 <= 8).
    # In case 9bp, insertion is at 0-based 91 (distance = 100 - 91 = 9 > 8).
    for dist, ins_pos, expect_tag in [(8, 92, True), (9, 91, False)]:
        b = SyntheticInputBuilder(tmp_path / f"sw_{dist}")
        b.add_node("n_pre", make_complex_seq(ins_pos, seed=710))
        b.add_node("n_gap", make_complex_seq(100 - ins_pos, seed=711))
        cds_seq = make_coding_seq(60, seed=712)
        b.add_node("n_cds", cds_seq)
        b.add_node("n_post", make_complex_seq(100, seed=713))
        b.add_node("x_ins", make_complex_seq(60, seed=714))
        b.set_ref_path("Ref#1#Chr01", ["n_pre+", "n_gap+", "n_cds+", "n_post+"])
        b.add_path("Alt#1#Chr01", ["n_pre+", "x_ins+", "n_gap+", "n_cds+", "n_post+"])
        b.add_gff_gene("g1", "t1", "+", [(101, 280)], [(101, 280, 0)])
        inputs = b.write(f"f25_sw_{dist}")

        proc, bundle = run_build(inputs, extra_args=["--splice-window", "8"])
        assert proc.returncode == 0 and bundle is not None and bundle.lifted_gff_rows is not None
        exon_rows = [r for r in bundle.lifted_gff_rows if r["type"] == "exon"]
        has_tag = exon_rows[0]["attributes"].get("splice_site_proximal_sv") == "True"
        assert has_tag is expect_tag
        assert_gff3_liftover_integrity(
            inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
        )


def test_f25_boundary_1bp_and_2bp_split_cds_phase_propagation_plus_and_minus_strands(
    tmp_path: Path,
) -> None:
    """F25 Boundary (§4 / T22): 1 bp and 2 bp micro-CDS segments propagate phase accurately on + and - strands."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n0", make_complex_seq(100, seed=720))
    plus_cds = make_coding_seq(60, seed=721)  # 180 bp: split into 1 bp + 1 bp + 178 bp -> phases 0, 2, 1
    b.add_node("p1", plus_cds[:1])
    b.add_node("p2", plus_cds[1:2])
    b.add_node("p3", plus_cds[2:])
    b.add_node("n_mid", make_complex_seq(100, seed=722))
    minus_cds_rc = revcomp(make_coding_seq(60, seed=723))  # 180 bp on '-' strand: split into 177 bp + 2 bp + 1 bp (in desc order: 1 bp, 2 bp, 177 bp -> phases 0, 2, 0)
    b.add_node("m1", minus_cds_rc[:177])
    b.add_node("m2", minus_cds_rc[177:179])
    b.add_node("m3", minus_cds_rc[179:])
    b.add_node("n_end", make_complex_seq(100, seed=724))
    b.add_node("x1", make_complex_seq(55, seed=725))
    b.add_node("x2", make_complex_seq(60, seed=726))
    b.add_node("x3", make_complex_seq(65, seed=727))
    b.add_node("x4", make_complex_seq(70, seed=728))

    b.set_ref_path("Ref#1#Chr01", ["n0+", "p1+", "p2+", "p3+", "n_mid+", "m1+", "m2+", "m3+", "n_end+"])
    b.add_path(
        "Alt#1#Chr01",
        ["n0+", "p1+", "x1+", "p2+", "x2+", "p3+", "n_mid+", "m1+", "x3+", "m2+", "x4+", "m3+", "n_end+"],
    )
    b.add_gff_gene("g_plus", "t_plus", "+", [(101, 280)], [(101, 280, 0)])
    b.add_gff_gene("g_minus", "t_minus", "-", [(381, 560)], [(381, 560, 0)])
    inputs = b.write("f25_t22_micro_cds")

    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None and bundle.lifted_gff_rows is not None
    plus_cds_rows = [
        r for r in bundle.lifted_gff_rows
        if r["type"] == "CDS" and r["attributes"].get("Parent") == "t_plus"
    ]
    minus_cds_rows = [
        r for r in bundle.lifted_gff_rows
        if r["type"] == "CDS" and r["attributes"].get("Parent") == "t_minus"
    ]
    assert [r["phase"] for r in sorted(plus_cds_rows, key=lambda r: r["start"])] == ["0", "2", "1"]
    assert [r["phase"] for r in sorted(minus_cds_rows, key=lambda r: r["start"], reverse=True)] == ["0", "2", "0"]
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f25_negative_malformed_gff3_fewer_columns_or_start_gt_end_fails(
    tmp_path: Path,
) -> None:
    """F25 Negative: Malformed GFF3 file (<9 columns or start > end) exits non-zero."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(200, seed=729))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    inputs = b.write("f25_bad_gff")

    bad_gff1 = tmp_path / "bad_cols.gff3"
    bad_gff1.write_text("##gff-version 3\nChr01\tsrc\tgene\t10\t100\n", encoding="utf-8")
    p1, _ = run_build(inputs, extra_args=["--ref-gff", bad_gff1], include_gff=False, check=False)
    assert p1.returncode != 0

    bad_gff2 = tmp_path / "bad_coords.gff3"
    bad_gff2.write_text(
        "##gff-version 3\nChr01\tsrc\tgene\t150\t50\t.\t+\t.\tID=g1\n", encoding="utf-8"
    )
    p2, _ = run_build(inputs, extra_args=["--ref-gff", bad_gff2], include_gff=False, check=False)
    assert p2.returncode != 0


# ===========================================================================
# === F26 Boundaries: Output Writers, CLI (build, verify, merge), Threads & --qc (5 tests) ===
# ===========================================================================


def test_f26_boundary_qc_149bp_short_bucket_vs_150bp_alignment_bucket(
    tmp_path: Path,
) -> None:
    """F26 Boundary (§7): --qc reports 50-149 bp insertions (149 bp) separately from >=150 bp insertions."""
    b = SyntheticInputBuilder(tmp_path)
    ref_part = make_complex_seq(200, seed=731)
    b.add_node("n1", ref_part)
    b.add_node("n2", make_complex_seq(200, seed=732))
    b.add_node("n3", make_complex_seq(200, seed=733))
    b.add_node("ins149", make_complex_seq(149, seed=734))
    # 150 bp insertion homologous to ref_part[:150]
    b.add_node("ins150", ref_part[:150])
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "ins149+", "n2+", "ins150+", "n3+"])
    inputs = b.write("f26_qc_149_150")

    proc, bundle = run_build(inputs, extra_args=["--qc"])
    assert proc.returncode == 0 and bundle is not None
    assert "149" in bundle.summary_text
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)


def test_f26_negative_verify_fails_when_gfa_modified_after_build_sha256_mismatch(
    tmp_path: Path,
) -> None:
    """F26 Negative (§5.5): `panflat verify` fails if input GFA is modified after build (SHA-256 mismatch)."""
    inputs = _build_two_anchor_with_ins(tmp_path, "f26_sha256")
    proc, bundle = run_build(inputs)
    assert proc.returncode == 0 and bundle is not None

    # Modify GFA file after build (append a comment or extra space)
    gfa_content = inputs.gfa_path.read_text(encoding="utf-8")
    inputs.gfa_path.write_text(gfa_content + "# post-build modification\n", encoding="utf-8")

    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f26_negative_verify_fails_when_centromere_local_offset_delta_corrupted(
    tmp_path: Path,
) -> None:
    """F26 Negative (§3.8.5): `panflat verify` fails if centromere mount offset or island record is corrupted."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("u", make_complex_seq(100, seed=741))
    b.add_node("c", make_complex_seq(200, seed=742))
    b.add_node("v", make_complex_seq(100, seed=743))
    b.add_node("isl", make_complex_seq(600, seed=744))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "isl+", "v+"])
    b.add_bed_region(120, 280, "cen1")
    inputs = b.write("f26_cen_delta")

    proc, bundle = run_build(inputs, extra_args=["--centromere-mode", "extract-islands"])
    assert proc.returncode == 0 and bundle is not None

    # Corrupt accepted_source_bp in centromere.tsv so 5-way accounting / mount delta fails verification
    cen_lines = bundle.centromere_path.read_text(encoding="utf-8").splitlines()
    cols = cen_lines[-1].split("\t")
    cols[25] = "999"  # corrupt accepted_source_bp
    bundle.centromere_path.write_text("\n".join([*cen_lines[:-1], "\t".join(cols)]) + "\n", encoding="utf-8")

    v_proc = run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert v_proc.returncode != 0


def test_f26_negative_merge_fails_on_empty_or_nonexistent_prefix_list(
    tmp_path: Path,
) -> None:
    """F26 Negative (§6): `panflat merge` exits non-zero on missing or empty --prefix-list."""
    missing_list = tmp_path / "does_not_exist.txt"
    p1 = run_panflat_cli(
        ["merge", "--prefix-list", missing_list, "--out-prefix", tmp_path / "m1"],
        check=False,
    )
    assert p1.returncode != 0

    empty_list = tmp_path / "empty_prefixes.txt"
    empty_list.write_text("", encoding="utf-8")
    p2 = run_panflat_cli(
        ["merge", "--prefix-list", empty_list, "--out-prefix", tmp_path / "m2"],
        check=False,
    )
    assert p2.returncode != 0


def test_f26_negative_cli_rejects_invalid_option_values(tmp_path: Path) -> None:
    """F26 Negative (§6): CLI rejects invalid --repeat-guard, --sec-backbone, --cluster-id, or --min-sv-len."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n1", make_complex_seq(100, seed=751))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    inputs = b.write("f26_bad_cli")

    invalid_arg_sets = [
        ["--repeat-guard", "invalid_mode"],
        ["--sec-backbone", "invalid_choice"],
        ["--cluster-id", "1.5"],
        ["--min-sv-len", "0"],
    ]
    for bad_args in invalid_arg_sets:
        proc, _ = run_build(inputs, extra_args=bad_args, check=False)
        assert proc.returncode != 0
