"""Tier 5 White-Box Adversarial & Gap-Closure End-to-End (E2E) Tests for PanFlat v1.

Targets uncovered control-flow branches, complex boundary interactions, and invariant stress
probes identified via white-box line/branch audit of:
- `panflat/models.py` (Step/StepProvenance subslice, flipped orientation propagation, effective mode)
- `panflat/gfa_io.py` (S-line LN:i tags, W-line '*' bounds, 2-part/unqualified PanSN names, BED track/browser headers, exhaustive contract error rejections)
- `panflat/repeat_guard.py` (edlib split-indel normalization D-M-D / I-M-I, mappy >1kb unaligned ends q_start>0 / q_end<len_s + window protection, mappy iterative sub-interval re-mapping across multiple large insertions)
- `panflat/normalizer.py` (multi-node novel runs in INV_CONFLICT & INV_EDGE, multiple unnormalized chains in a single forward-anchor gap, weighted LIS tie-breakers on first_step_idx & full index tuple, local multi-hit node tracking, back-to-back tandem repeat cycle folding synthetic piece split, canonical CNV partial-shared & aperiodic edge cases)
- `panflat/centromere.py` (single-side u-only and v-only bracketed internal slices, multi-u/v ambiguous fragment internal slices in bypass & extract-islands, pure insertion strictly inside centromere -> CEN_BOUNDARY_AMBIGUOUS, ref_status='gapped' short vs long residuals, Step C split short residual CEN_SHORT_RESIDUAL, Step D >1kb mappy island clustering, CEN_SEQUENCE_UNCERTAIN)
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.helpers import (
    OutputBundle,
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
    run_panflat_cli,
    run_verify,
)


# ---------------------------------------------------------------------------
# 1. GFA/FASTA/BED Format Variations: LN:i tags, W-line '*' bounds, BED headers
# ---------------------------------------------------------------------------

def test_tier5_m1m2_01_gfa_ln_tag_w_wildcard_bounds_and_bed_headers(tmp_path: Path) -> None:
    """White-box target: `panflat/gfa_io.py` valid format branches:
    - GFA `#` comments, blank lines, `S` lines with matching `LN:i:<len>` tags
    - Single-step `P` line with explicit `0M` overlap
    - `W` line with wildcard `start='*'` and `end='*'` (`seq_start=0`, `seq_end=spelled_len`)
    - FASTA with blank lines and mixed case
    - BED file with `#` comment, `track ...`, `browser ...` lines, and space-delimited 3-col row
    """
    s1 = make_complex_seq(250, seed=5101)
    s2 = make_complex_seq(300, seed=5102)
    s3 = make_complex_seq(250, seed=5103)
    s4 = make_complex_seq(250, seed=5106)
    s_ins = make_complex_seq(180, seed=5104)
    s_single = make_complex_seq(250, seed=5105)

    gfa_path = tmp_path / "fmt.gfa"
    ref_fa_path = tmp_path / "fmt.ref.fa"
    bed_path = tmp_path / "fmt.cen.bed"
    out_prefix = tmp_path / "fmt_out"

    gfa_lines = [
        "H\tVN:Z:1.0",
        "# Comment line in GFA",
        "",
        f"S\tn1\t{s1}\tLN:i:{len(s1)}",
        f"S\tn2\t{s2}\tLN:i:{len(s2)}",
        f"S\tn3\t{s3}\tLN:i:{len(s3)}",
        f"S\tn4\t{s4}\tLN:i:{len(s4)}",
        f"S\tins1\t{s_ins}\tLN:i:{len(s_ins)}",
        f"S\tsingle\t{s_single}\tLN:i:{len(s_single)}",
        "L\tn1\t+\tn2\t+\t0M",
        "L\tn2\t+\tn3\t+\t0M",
        "L\tn3\t+\tn4\t+\t0M",
        "L\tn3\t+\tins1\t+\t0M",
        "L\tins1\t+\tn4\t+\t0M",
        "P\tRef#1#Chr01\tn1+,n2+,n3+,n4+\t0M,0M,0M",
        "P\tSingleStep#1#Chr01\tsingle+\t0M",
        "W\tWalkStar\t1\tChr01\t*\t*\t>n1>n2>n3>ins1>n4",
    ]
    gfa_path.write_text("\n".join(gfa_lines) + "\n", encoding="utf-8")

    ref_seq_mixed = s1.lower() + "\n\n" + s2 + "\n" + s3.lower() + "\n" + s4
    ref_fa_path.write_text(f">Chr01 description\n\n{ref_seq_mixed}\n\n", encoding="utf-8")

    bed_lines = [
        "# BED comment line",
        "track name=centromeres description=\"test\"",
        "browser position Chr01:1-800",
        "Chr01 250 550",
    ]
    bed_path.write_text("\n".join(bed_lines) + "\n", encoding="utf-8")

    proc = run_panflat_cli(
        [
            "build",
            "--gfa", gfa_path,
            "--ref-fa", ref_fa_path,
            "--backbone-prefix", "Ref#1#",
            "--centromere-bed", bed_path,
            "--centromere-mode", "bypass",
            "--out-prefix", out_prefix,
        ]
    )
    assert proc.returncode == 0
    from tests.e2e.helpers import OutputBundle
    bundle = OutputBundle(out_prefix)
    assert_all_global_invariants(bundle, gfa_path, ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert len(bundle.inserts) == 1
    assert bundle.inserts[0]["src_path"] == "WalkStar#1#Chr01:0-1230"


# ---------------------------------------------------------------------------
# 2. GFA/FASTA/BED Exhaustive Malformed Input Contract Rejections
# ---------------------------------------------------------------------------

def test_tier5_m1m2_02_exhaustive_input_contract_error_branches(tmp_path: Path) -> None:
    """White-box target: `panflat/gfa_io.py` error branches in `parse_fasta`, `parse_bed`,
    `load_and_validate_inputs`, and `_parse_pansn_path_name`.
    """
    s1 = make_complex_seq(200, seed=5201)
    s2 = make_complex_seq(200, seed=5202)
    valid_gfa = tmp_path / "valid.gfa"
    valid_fa = tmp_path / "valid.fa"
    valid_gfa.write_text(
        f"H\tVN:Z:1.0\nS\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\n",
        encoding="utf-8",
    )
    valid_fa.write_text(f">Chr01\n{s1}{s2}\n", encoding="utf-8")

    # 2a. Valid 2-part PanSN path name (Sample#Hap) and unqualified path name (NoHash)
    gfa_pansn = tmp_path / "pansn_2part.gfa"
    gfa_pansn.write_text(
        f"H\tVN:Z:1.0\nS\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\n"
        "P\tRef#1#Chr01\tn1+,n2+\t*\n"
        "P\tTwoPart#1\tn1+,n2+\t*\n"
        "P\tUnqualifiedName\tn1+,n2+\t*\n",
        encoding="utf-8",
    )
    proc_ok = run_panflat_cli(
        [
            "build",
            "--gfa", gfa_pansn,
            "--ref-fa", valid_fa,
            "--backbone-prefix", "Ref#1#",
            "--out-prefix", tmp_path / "pansn_ok",
        ]
    )
    assert proc_ok.returncode == 0

    # 2b. Malformed FASTA cases: missing file, data before header, empty header, empty record,
    # non-alphabetic sequence, duplicate header
    bad_fas = [
        (tmp_path / "no_such.fa", None),
        (tmp_path / "pre_hdr.fa", "ACGT\n>Chr01\nACGT\n"),
        (tmp_path / "empty_hdr.fa", ">   \nACGT\n"),
        (tmp_path / "empty_rec1.fa", ">Chr01\n>Chr02\nACGT\n"),
        (tmp_path / "empty_rec2.fa", ">Chr01\n"),
        (tmp_path / "non_alpha.fa", ">Chr01\nACGT123\n"),
        (tmp_path / "dup_hdr.fa", f">Chr01\n{s1}\n>Chr01\n{s2}\n"),
        (tmp_path / "no_rec.fa", "   \n\n"),
    ]
    for idx, (fa_p, content) in enumerate(bad_fas):
        if content is not None:
            fa_p.write_text(content, encoding="utf-8")
        pr = run_panflat_cli(
            [
                "build",
                "--gfa", valid_gfa,
                "--ref-fa", fa_p,
                "--backbone-prefix", "Ref#1#",
                "--out-prefix", tmp_path / f"bad_fa_{idx}",
            ],
            check=False,
        )
        assert pr.returncode != 0

    # 2c. Malformed BED cases: missing file, non-integer start/end
    for idx, bed_content in enumerate([None, "Chr01\tabc\t200\n"]):
        bed_p = tmp_path / f"bad_{idx}.bed"
        if bed_content is not None:
            bed_p.write_text(bed_content, encoding="utf-8")
        pr = run_panflat_cli(
            [
                "build",
                "--gfa", valid_gfa,
                "--ref-fa", valid_fa,
                "--backbone-prefix", "Ref#1#",
                "--centromere-bed", bed_p,
                "--centromere-mode", "bypass",
                "--out-prefix", tmp_path / f"bad_bed_{idx}",
            ],
            check=False,
        )
        assert pr.returncode != 0

    # 2d. Malformed GFA cases: C-line, bad LN:i tag, mismatched LN:i tag, no S records,
    # bad P overlaps, malformed PanSN, bad W-lines, and 2-part P/W overlap
    bad_gfas = [
        "",  # empty GFA
        f"S\tn1\t{s1}\nC\tn1\t+\tn1\t+\t0\t0M\n",  # C line
        "S\tn1\n",  # S < 3 fields
        f"S\t\t{s1}\n",  # empty node ID
        f"S\tn1\t{s1}\nS\tn1\t{s2}\n",  # duplicate S node ID
        "S\tn1\tACGT123\n",  # non-alphabetic S sequence
        f"S\tn1\t{s1}\tLN:i:xyz\n",  # non-int LN:i tag
        f"S\tn1\t{s1}\tLN:i:999\n",  # mismatched LN:i tag
        "H\tVN:Z:1.0\n",  # no S records
        f"S\tn1\t{s1}\nL\tn1\t+\tn1\n",  # L < 6 fields
        f"S\tn1\t{s1}\nL\tunknown\t+\tn1\t+\t0M\n",  # L unknown node
        f"S\tn1\t{s1}\nL\tn1\t?\tn1\t+\t0M\n",  # L bad orient
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\tn1+\n",  # P < 4 fields
        f"S\tn1\t{s1}\nP\t\tn1+\t*\n",  # P empty name
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\tn1+\t*\nP\tRef#1#Chr01\tn1+\t*\n",  # P duplicate name
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\t,\t*\n",  # P empty steps
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\tbadstep\t*\n",  # P bad step token
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\tunknown+\t*\n",  # P unknown node
        f"S\tn1\t{s1}\nP\tRef#1#Chr01\tn1+\t5M\n",  # single-step P non-0M overlap
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t0M,0M\n",  # wrong overlap count
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nP\tBad##Name\tn1+,n2+\t*\n",  # malformed PanSN
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\t0\n",  # W < 7 fields
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\t*\t1\tChr01\t0\t400\t>n1>n2\n",  # W '*' sample
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\tabc\t400\t>n1>n2\n",  # W bad start
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\t-5\t400\t>n1>n2\n",  # W negative start
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\t0\txyz\t>n1>n2\n",  # W bad end
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\t0\t400\tbadwalk\n",  # W bad walk
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nW\tS1\t1\tChr01\t0\t400\t>unknown\n",  # W unknown node
        f"S\tn1\t{s1}\nS\tn2\t{s2}\nL\tn1\t+\tn2\t+\t0M\nP\tRef#1#Chr01\tn1+,n2+\t*\nP\tS1#1\tn1+,n2+\t*\nW\tS1\t1\t1\t0\t400\t>n1>n2\n",  # 2-part P/W overlap
    ]
    for idx, g_txt in enumerate(bad_gfas):
        gp = tmp_path / f"bad_{idx}.gfa"
        gp.write_text(g_txt, encoding="utf-8")
        pr = run_panflat_cli(
            [
                "build",
                "--gfa", gp,
                "--ref-fa", valid_fa,
                "--backbone-prefix", "Ref#1#",
                "--out-prefix", tmp_path / f"bad_gfa_out_{idx}",
            ],
            check=False,
        )
        assert pr.returncode != 0


# ---------------------------------------------------------------------------
# 3. Flipped W-line Contig + Multi-Node Partial Repeat-Guard Slicing
# ---------------------------------------------------------------------------

def test_tier5_m1m2_03_flipped_w_line_contig_multi_node_repeat_guard_slicing(tmp_path: Path) -> None:
    """White-box target: `panflat/models.py` (`Step.subslice`, `StepProvenance.subslice` on `src_strand='-'`)
    and `panflat/repeat_guard.py` (`slice_candidate_by_intervals` & `merge_contiguous_provenance` on `'-'`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(300, seed=5301))
    b.add_node("r2", make_complex_seq(300, seed=5302))

    # Build a 3-node insertion where the middle node contains a 1600 bp satellite flanked by 250 bp complex DNA
    s_n1 = make_complex_seq(300, seed=5303)
    s_n2 = make_complex_seq(250, seed=5304) + make_satellite_seq(20, 1600, seed=5305) + make_complex_seq(250, seed=5306)
    s_n3 = make_complex_seq(300, seed=5307)
    b.add_node("n1", s_n1)
    b.add_node("n2", s_n2)
    b.add_node("n3", s_n3)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # Whole-contig reverse W-line starting at offset 8000:
    # r2- -> n3- -> n2- -> n1- -> r1-
    # After orient_contig_top_level: r1+ -> n1+ -> n2+ -> n3+ -> r2+ with src_strand='-'
    b.add_walk("DraftRev", "1", "ctgW", 8000, None, ["r2-", "n3-", "n2-", "n1-", "r1-"])

    inputs = b.write("flip_w_rg")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on", "--dump-repeat-sequences"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Both residuals and the masked REPEAT_LIKE interval must have src_strand == '-'
    assert len(bundle.inserts) == 2
    for ins in bundle.inserts:
        assert ins["src_strand"] == "-"
    rep_rows = [u for u in bundle.unresolved if u["reason"] == "REPEAT_LIKE"]
    assert len(rep_rows) == 1
    assert rep_rows[0]["src_strand"] == "-"


# ---------------------------------------------------------------------------
# 4. Repeat-Guard: Mappy >1kb Unaligned Ends (q_start > 0 & q_end < len_s) + Window Protection
# ---------------------------------------------------------------------------

def test_tier5_m1m2_04_repeat_guard_mappy_gt1kb_unaligned_ends_and_criterion3(tmp_path: Path) -> None:
    """White-box target: `panflat/repeat_guard.py` (`check_similarity_merge` >1kb `mappy` path:
    - Criterion 3 rejection when `unaligned_ends_s + ins_bp_s >= min_sv_len` (line 805)
    - Criterion 4 window evaluation when `q_start > 0` and `q_end < len_s` (lines 818-821))
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(300, seed=5401))
    b.add_node("r2", make_complex_seq(300, seed=5402))

    core = make_complex_seq(1360, seed=5403)
    # Representative R (1420 bp): 30bp poly-A + 1360bp core + 30bp poly-C
    seq_r = ("A" * 30) + core + ("C" * 30)
    # Candidate S_merge (1400 bp): 20bp poly-T + 1360bp core + 20bp poly-G
    # mappy aligns core [20..1380] -> q_start=20 > 0, q_end=1380 < 1400, unaligned_ends = 40 < 50 -> MERGES!
    seq_s_merge = ("T" * 20) + core + ("G" * 20)
    # Candidate S_reject (1410 bp): 25bp poly-T + 1360bp core + 25bp poly-G
    # mappy aligns core [25..1385] -> unaligned_ends = 25 + 25 = 50 >= min_sv_len(50) -> REJECTED by Criterion 3!
    seq_s_reject = ("T" * 25) + core + ("G" * 25)

    b.add_node("rep_R", seq_r)
    b.add_node("cand_S_ok", seq_s_merge)
    b.add_node("cand_S_bad", seq_s_reject)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("H1#1#Chr01", ["r1+", "rep_R+", "r2+"])
    b.add_path("H2#1#Chr01", ["r1+", "cand_S_ok+", "r2+"])
    b.add_path("H3#1#Chr01", ["r1+", "cand_S_bad+", "r2+"])

    inputs = b.write("mappy_ends")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "off"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # H1 and H2 merge into 1 representative (support=2), while H3 stays separate (2 pieces total)
    assert len(bundle.inserts) == 2
    members_sets = [set(str(r["members"]).split(",")) for r in bundle.inserts]
    assert {"H1#1", "H2#1"} in members_sets
    assert {"H3#1"} in members_sets


# ---------------------------------------------------------------------------
# 5. Repeat-Guard: Mappy Iterative Sub-Interval Re-Mapping in Centromere Step B
# ---------------------------------------------------------------------------

def test_tier5_m1m2_05_centromere_step_b_iterative_subinterval_remapping(tmp_path: Path) -> None:
    """White-box target: `panflat/repeat_guard.py` (`subtract_reference_intervals_mappy` lines 688-721)
    and `panflat/centromere.py` (`SPLIT_BY_REFERENCE_SUBTRACTION`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_n0 = make_complex_seq(250, seed=5501)
    s_u = make_complex_seq(250, seed=5502)
    s_cen_ref = make_complex_seq(1500, seed=5503)
    s_v = make_complex_seq(250, seed=5504)
    s_n4 = make_complex_seq(250, seed=5505)

    b.add_node("n0", s_n0)
    b.add_node("u", s_u)
    b.add_node("cen_ref", s_cen_ref)
    b.add_node("v", s_v)
    b.add_node("n4", s_n4)

    # Construct a 2700 bp candidate with 3 homologous blocks separated by two 900 bp novel islands:
    # 300 bp (fwd homologous to cen_ref[100:400]) + 900 bp novel_1 + 300 bp (revcomp homologous to cen_ref[550:850])
    # + 900 bp novel_2 + 300 bp (fwd homologous to cen_ref[1000:1300])
    hom1 = mutate_seq_snps(s_cen_ref[100:400], num_snps=6, seed=5506)
    nov1 = make_complex_seq(900, seed=5507)
    hom2_rc = revcomp(mutate_seq_snps(s_cen_ref[550:850], num_snps=4, seed=5508))
    nov2 = make_complex_seq(900, seed=5509)
    hom3 = mutate_seq_snps(s_cen_ref[1000:1300], num_snps=6, seed=5510)

    b.add_node("cand_multi", hom1 + nov1 + hom2_rc + nov2 + hom3)
    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen_ref+", "v+", "n4+"])
    b.add_path("Donor#1#Chr01", ["n0+", "u+", "cen_ref+", "cand_multi+", "v+", "n4+"])
    b.add_bed_region(500, 2000, "CEN_SUB")

    inputs = b.write("cen_iter_sub")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    donor_row = next(r for r in (bundle.centromere_rows or []) if r["hap"] == "Donor#1")
    assert int(donor_row["nonref_input_bp"]) == 2700
    assert int(donor_row["reference_like_bp"]) == 900
    assert int(donor_row["accepted_source_bp"]) == 1800
    assert any(j.get("final_reason") == "SPLIT_BY_REFERENCE_SUBTRACTION" for j in bundle.log_json_records)


# ---------------------------------------------------------------------------
# 6. Normalizer: Multi-Node Novel Runs in INV_CONFLICT and INV_EDGE + Multiple Chains in Gap
# ---------------------------------------------------------------------------

def test_tier5_m1m2_06_normalizer_multinode_novel_in_inv_conflict_and_inv_edge(tmp_path: Path) -> None:
    """White-box target: `panflat/normalizer.py` (`_extract_contiguous_nonref_slices` lines 282-322,
    `normalize_inversions_top_level` lines 488-560 with 3 unnormalized reverse chains in one gap).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    # Make r1 and r4 1000 bp each so forward weight (2250 bp) > reverse weight (1500 bp)
    b.add_node("r1", make_complex_seq(1000, seed=5601))
    b.add_node("r2", make_complex_seq(250, seed=5602))
    b.add_node("r3", make_complex_seq(250, seed=5603))
    b.add_node("r4", make_complex_seq(1000, seed=5604))
    b.add_node("r5", make_complex_seq(250, seed=5605))

    for tag, seed in [("a1", 5611), ("a2", 5612), ("b1", 5613), ("b2", 5614), ("e1", 5615), ("e2", 5616)]:
        b.add_node(f"nov_{tag}", make_complex_seq(90, seed=seed))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+"])
    # HapConflict has 3 overlapping reverse chains in the same (r1+, r4+) gap:
    # Chain 1: r3- -> nov_a1+ -> nov_a2+ -> r2- (2-node novel run = 180 bp)
    # Chain 2: r3- -> nov_b1+ -> nov_b2+ -> r2- (2-node novel run = 180 bp)
    # Chain 3: r3- -> r2- (adjacent to r4+, so trailing gap after Chain 3 is empty)
    b.add_path(
        "HapConflict#1#Chr01",
        [
            "r1+",
            "r3-", "nov_a1+", "nov_a2+", "r2-",
            "r3-", "nov_b1+", "nov_b2+", "r2-",
            "r3-", "r2-",
            "r4+", "r5+",
        ],
    )
    # HapEdge has an unnormalized INV_EDGE at the right contig end with a 2-node novel run nov_e1+ -> nov_e2+
    b.add_path(
        "HapEdge#1#Chr01",
        ["r1+", "r2+", "r3+", "r5-", "nov_e1+", "nov_e2+", "r4-"],
    )

    inputs = b.write("inv_multinode")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    unres_by_reason = {}
    for u in bundle.unresolved:
        unres_by_reason.setdefault(u["reason"], []).append(u)
    assert "INV_CONFLICT" in unres_by_reason
    assert "INV_EDGE" in unres_by_reason
    # Each 2-node 90bp+90bp run must be merged into a single 180bp contiguous source interval
    assert any(int(u["length"]) == 180 for u in unres_by_reason["INV_CONFLICT"])
    assert any(int(u["length"]) == 180 for u in unres_by_reason["INV_EDGE"])


# ---------------------------------------------------------------------------
# 7. Normalizer: Weighted LIS Tie-Breakers & Local Multi-Hit Node Tracking
# ---------------------------------------------------------------------------

def test_tier5_m1m2_07_normalizer_weighted_lis_tie_breakers_and_multi_hit(tmp_path: Path) -> None:
    """White-box target: `panflat/normalizer.py` (`compute_weighted_lis` lines 643-655 tie-breaking
    on `first_step_idx` `f_a < f_b` and full index tuple `_reconstruct_indices`, plus `_local_multi_hit_nodes`
    lines 749-770 when `right_anc` and `b_steps` contain multi-hit nodes).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    # r1, r2, r3, r4 all have identical 250 bp length to force exact LIS weight ties
    for idx in range(1, 6):
        b.add_node(f"r{idx}", make_complex_seq(250, seed=5700 + idx))
    b.add_node("ins_tie1", make_complex_seq(150, seed=5711))
    b.add_node("ins_tie2", make_complex_seq(160, seed=5712))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+"])
    # HapTieFirst: two competing 2-step increasing chains of weight 500 ending at r4+:
    # Chain A: r2+ (branch idx 0) -> r4+ (branch idx 3), first_step_idx = 0
    # Chain B: r1+ (branch idx 1) -> r4+ (branch idx 3), first_step_idx = 1
    # Tie-breaker f_a < f_b (line 653) must pick Chain A (r2+ -> r4+)!
    b.add_path("HapTieFirst#1#Chr01", ["r2+", "r1+", "ins_tie1+", "r4+", "r5+"])

    # HapTieTuple: two competing 3-step increasing chains of weight 750 sharing the same first step r1+ (idx 0)
    # and ending at r4+ (idx 4), plus revisiting r3+ and r4+ so right_anc and b_steps hit multi_hit_set:
    # r1+ (idx 0) -> r3+ (idx 1) -> r2+ (idx 2) -> ins_tie2+ (idx 3) -> r4+ (idx 4) -> r3+ (idx 5) -> r4+ (idx 6) -> r5+ (idx 7)
    b.add_path(
        "HapTieTuple#1#Chr01",
        ["r1+", "r3+", "r2+", "ins_tie2+", "r4+", "r3+", "r4+", "r5+"],
    )

    inputs = b.write("lis_ties")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert any("multi_hit=1" in str(ev["alleles"]) for ev in bundle.sv_rows)


# ---------------------------------------------------------------------------
# 8. Normalizer: Back-to-Back Tandem Repeats Cycle Folding Synthetic Piece Split
# ---------------------------------------------------------------------------

def test_tier5_m1m2_08_normalizer_adjacent_tandem_repeats_cycle_folding(tmp_path: Path) -> None:
    """White-box target: `panflat/normalizer.py` (`fold_tandem_cycles` lines 1071-1074
    `if pending_break and len(folded_steps) > 0 and keep_copies > 0: piece_boundaries.append(...)`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=5801))
    b.add_node("r2", make_complex_seq(250, seed=5802))
    b.add_node("U", make_complex_seq(70, seed=5803))
    b.add_node("V", make_complex_seq(80, seed=5804))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # Donor has U+ x 3 (210 bp) immediately followed by V+ x 3 (240 bp) = 450 bp (< 500 bp repeat-guard threshold)
    b.add_path("Donor#1#Chr01", ["r1+", "U+", "U+", "U+", "V+", "V+", "V+", "r2+"])

    inputs = b.write("adj_repeats")
    _, bundle = run_build(inputs, extra_args=["--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # U+ (70 bp, source interval [250, 320)) and V+ (80 bp, source interval [460, 540))
    # are separated by deleted copies 2..3 of U+, so they must form 2 pieces with join_before='synthetic'!
    assert len(bundle.inserts) == 2
    assert bundle.inserts[0]["join_before"] == "source_contiguous"
    assert bundle.inserts[1]["join_before"] == "synthetic"
    assert int(bundle.inserts[0]["pan_end"]) - int(bundle.inserts[0]["pan_start"]) == 70
    assert int(bundle.inserts[1]["pan_end"]) - int(bundle.inserts[1]["pan_start"]) == 80


# ---------------------------------------------------------------------------
# 9. Normalizer & Linearizer: Partial Shared CNV + DEL Co-Occurrence & Aperiodic Remainders
# ---------------------------------------------------------------------------

def test_tier5_m1m2_09_partial_shared_cnv_with_del_and_aperiodic_remainders(tmp_path: Path) -> None:
    """White-box target: `panflat/normalizer.py` (`_extract_full_period` lines 1138-1149)
    and `panflat/linearizer.py` (lines 3002-3151 partial shared CNV + DEL inside `B(a,b)`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=5901))
    b.add_node("del_part", make_complex_seq(120, seed=5902))
    b.add_node("tr_u", make_complex_seq(60, seed=5903))
    b.add_node("r2", make_complex_seq(250, seed=5904))
    b.add_node("tr_v", make_complex_seq(60, seed=5905))
    b.add_node("r3", make_complex_seq(250, seed=5906))
    b.add_node("nov_tail", make_complex_seq(80, seed=5907))

    # Between r1 and r2: reference has del_part+ followed by 2 copies of tr_u+
    # Between r2 and r3: reference has 2 copies of tr_v+
    b.set_ref_path(
        "Ref#1#Chr01",
        ["r1+", "del_part+", "tr_u+", "tr_u+", "r2+", "tr_v+", "tr_v+", "r3+"],
    )
    # Donor1:
    # - Between r1 and r2: deletes del_part+ (120 bp DEL) and expands tr_u+ from 2 to 4 copies (120 bp CNV)!
    # - Between r2 and r3: has tr_v+ -> tr_v+ -> nov_tail+ (trailing incomplete remainder -> NOT a pure CNV)
    b.add_path(
        "Donor1#1#Chr01",
        ["r1+", "tr_u+", "tr_u+", "tr_u+", "tr_u+", "r2+", "tr_v+", "tr_v+", "nov_tail+", "r3+"],
    )

    inputs = b.write("partial_cnv_del")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = [ev["type"] for ev in bundle.sv_rows]
    assert "DEL" in ev_types
    assert "CNV" in ev_types
    assert "INS" in ev_types


# ---------------------------------------------------------------------------
# 10. Centromere: Single-Side u-only / v-only & Multi-u/v Ambiguous Slices in Bypass & Extract
# ---------------------------------------------------------------------------

def test_tier5_m1m2_10_centromere_single_side_and_ambiguous_bracketed_slices(tmp_path: Path) -> None:
    """White-box target: `panflat/centromere.py` (`_collect_single_side_slice` lines 519-631 for
    both `u`-only and `v`-only fragments, multi-`u`/`v` ambiguous fragment slice loop lines 650-673,
    and `bypass` non-complete slice unresolved/log emission lines 1132-1170).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [
        ("n0", 6001), ("u", 6002), ("c_u1", 6003), ("c_rep", 6004), ("c_u2", 6005), ("v", 6006), ("n_out", 6007)
    ]:
        b.add_node(nid, make_complex_seq(200, seed=seed))
    for nid, seed in [
        ("cand_u_int", 6011), ("cand_v_int", 6012), ("cand_amb1", 6013), ("cand_amb2", 6014)
    ]:
        b.add_node(nid, make_complex_seq(150, seed=seed))

    # Reference:
    # n0 [0,200), u [200,400), c_u1 [400,600), c_rep [600,800), c_rep [800,1000), c_u2 [1000,1200), v [1200,1400), n_out [1400,1600)
    # BED [450, 1150) expands outward to u (step 1, ref_end=400) and v (step 6, ref_start=1200).
    # Note: c_u1 (step 2 = u+1) and c_u2 (step 5 = v-1) are unique ref steps inside the centromere,
    # while c_rep is an internal_only_ref_node!
    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "c_u1+", "c_rep+", "c_rep+", "c_u2+", "v+", "n_out+"])
    # 1) HapPartU: u-only partial hitting u+ -> c_u1+ (u+1) -> cand_u_int+ -> c_rep+ (internal ref node)
    b.add_path("HapPartU#1#Chr01", ["n0+", "u+", "c_u1+", "cand_u_int+", "c_rep+"])
    # 2) HapPartV: v-only partial hitting c_rep+ -> cand_v_int+ -> c_u2+ (v-1) -> v+ -> n_out+
    b.add_path("HapPartV#1#Chr01", ["c_rep+", "cand_v_int+", "c_u2+", "v+", "n_out+"])
    # 3) HapAmbUV: hits u+ twice on the same fragment with internal non-ref candidates
    b.add_path("HapAmbUV#1#Chr01", ["n0+", "u+", "cand_amb1+", "u+", "c_rep+", "cand_amb2+", "v+", "n_out+"])

    b.add_bed_region(450, 1150, "CEN_SLICE")
    inputs = b.write("cen_slices")

    # Run in bypass mode (with --dump-repeat-sequences)
    _, bun_byp = run_build(
        inputs,
        out_prefix=tmp_path / "byp",
        extra_args=["--centromere-mode", "bypass", "--dump-repeat-sequences"],
    )
    # Run in extract-islands mode
    _, bun_ext = run_build(
        inputs,
        out_prefix=tmp_path / "ext",
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bun_byp is not None and bun_ext is not None
    assert_all_global_invariants(bun_byp, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_ext, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_byp)
    assert_centromere_accounting_and_offsets(bun_ext)

    # Verify all 4 internal non-ref candidates were captured as CEN_BYPASS in bypass mode
    byp_unres = [u for u in bun_byp.unresolved if u["reason"] == "CEN_BYPASS"]
    assert len(byp_unres) == 4
    # And in extract-islands mode, captured as CEN_INCOMPLETE (2) and CEN_AMBIGUOUS (2)
    ext_reasons = [u["reason"] for u in bun_ext.unresolved]
    assert ext_reasons.count("CEN_INCOMPLETE") == 2
    assert ext_reasons.count("CEN_AMBIGUOUS") == 2


# ---------------------------------------------------------------------------
# 11. Centromere: CEN_BOUNDARY_AMBIGUOUS Pure Insertion Strictly Inside & Crossing Jump
# ---------------------------------------------------------------------------

def test_tier5_m1m2_11_centromere_boundary_ambiguous_pure_insertion_and_crossing_jump(tmp_path: Path) -> None:
    """White-box target: `panflat/centromere.py` (Step 4b lines 826-904:
    - Pure insertion at `span_start == span_end` strictly inside `(reg.ref_start, reg.ref_end)` (lines 854-856)
    - Crossing jump `span_start < span_end` across `[reg.ref_start, reg.ref_end)` without hitting `u` or `v` (lines 848-851)
    - `seen_exported_intervals` registration of `boundary_ambiguous_records` in `process_centromere_regions` (lines 964-969))
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 6101), ("u", 6102), ("c_u1", 6103), ("c_u2", 6104), ("v", 6105), ("n4", 6106)]:
        b.add_node(nid, make_complex_seq(200, seed=seed))
    b.add_node("pure_in_cen", make_complex_seq(160, seed=6107))
    b.add_node("cross_cen", make_complex_seq(180, seed=6108))

    # Ref: n0 [0,200), u [200,400), c_u1 [400,600), c_u2 [600,800), v [800,1000), n4 [1000,1200)
    # BED [450, 750) covers c_u1 and c_u2 -> centromere region is [400, 800) with u=step 1, v=step 4.
    # The junction between c_u1 and c_u2 is at ref=600, which is strictly inside (400, 800)!
    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "c_u1+", "c_u2+", "v+", "n4+"])
    # HapPureInside: partial fragment missing u and v, with pure insertion at ref=600 between c_u1+ and c_u2+
    b.add_path("HapPureInside#1#Chr01", ["c_u1+", "pure_in_cen+", "c_u2+"])
    # HapCrossJump: fragment jumping directly from n0+ to n4+ with cross_cen+
    b.add_path("HapCrossJump#1#Chr01", ["n0+", "cross_cen+", "n4+"])

    b.add_bed_region(450, 750, "CEN_AMB")
    inputs = b.write("cen_amb_pure")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    amb_unres = [u for u in bundle.unresolved if u["reason"] == "CEN_BOUNDARY_AMBIGUOUS"]
    assert len(amb_unres) == 2
    assert {int(u["length"]) for u in amb_unres} == {160, 180}
    # Neither ambiguous insertion may enter pan.fa
    assert len(bundle.inserts) == 0


# ---------------------------------------------------------------------------
# 12. Centromere: ref_status='gapped' Short vs Long Candidates & Step C Split Short Residual
# ---------------------------------------------------------------------------

def test_tier5_m1m2_12_centromere_ref_gapped_short_and_step_c_split_short_residual(tmp_path: Path) -> None:
    """White-box target: `panflat/centromere.py`:
    - `ref_status == "gapped"` with both a short candidate (`50 <= c_len < 500` -> `CEN_SHORT_RESIDUAL`, lines 1352-1380)
      and a long candidate (`>= 500` -> `CEN_REFERENCE_GAP`, lines 1381-1408)
    - `ref_status == "complete"` where Step C (`repeat_mask_intervals`) splits a candidate into a
      1500 bp `CEN_REPEAT_LIKE` block and a 220 bp complex tail (`50 <= c_len < 500` -> `CEN_SHORT_RESIDUAL`, lines 1653-1682)
    """
    # Part A: ref_status == "gapped" while Donor#1 is "complete" (Donor traverses cen_clean+, not cen_n+)
    b_gap = SyntheticInputBuilder(tmp_path / "gap_ref", chrom="Chr01", backbone_prefix="Ref#1#")
    b_gap.add_node("n0", make_complex_seq(200, seed=6201))
    b_gap.add_node("u", make_complex_seq(200, seed=6202))
    b_gap.add_node("cen_n", make_complex_seq(200, seed=6203) + ("N" * 20) + make_complex_seq(200, seed=6204))
    b_gap.add_node("cen_clean", make_complex_seq(200, seed=6208))
    b_gap.add_node("v", make_complex_seq(200, seed=6205))
    b_gap.add_node("short_c", make_complex_seq(180, seed=6206))
    b_gap.add_node("long_c", make_complex_seq(600, seed=6207))

    b_gap.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen_n+", "cen_clean+", "v+"])
    b_gap.add_path("Donor#1#Chr01", ["n0+", "u+", "short_c+", "cen_clean+", "long_c+", "v+"])
    b_gap.add_bed_region(400, 1020, "CEN_GAP")
    inp_gap = b_gap.write("gap")
    _, bun_gap = run_build(
        inp_gap,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bun_gap is not None
    assert_all_global_invariants(bun_gap, inp_gap.gfa_path, inp_gap.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_gap)
    d_row = next(r for r in (bun_gap.centromere_rows or []) if r["hap"] == "Donor#1")
    assert d_row["status"] == "complete"
    assert d_row["ref_status"] == "gapped"
    assert int(d_row["short_bp"]) == 180
    assert int(d_row["unresolved_bp"]) == 600
    assert {u["reason"] for u in bun_gap.unresolved} == {"CEN_SHORT_RESIDUAL", "CEN_REFERENCE_GAP"}

    # Part B: ref_status == "complete" with Step C splitting 1720 bp (1350 bp sat + 370 bp complex)
    # into 1500 bp masked ([0,1000) U [500,1500) = [0,1500)) + 220 bp short complex tail ([1500,1720))!
    b_sc = SyntheticInputBuilder(tmp_path / "step_c_short", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 6211), ("u", 6212), ("cen", 6213), ("v", 6214)]:
        b_sc.add_node(nid, make_complex_seq(250, seed=seed))
    sat_1350 = make_satellite_seq(unit_len=20, total_len=1350, seed=6215)
    tail_370 = make_complex_seq(370, seed=6216)
    b_sc.add_node("sat_tail", sat_1350 + tail_370)

    b_sc.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b_sc.add_path("Donor#1#Chr01", ["n0+", "u+", "cen+", "sat_tail+", "v+"])
    b_sc.add_bed_region(500, 750, "CEN_SC")
    inp_sc = b_sc.write("sc")
    _, bun_sc = run_build(
        inp_sc,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bun_sc is not None
    assert_all_global_invariants(bun_sc, inp_sc.gfa_path, inp_sc.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_sc)
    sc_row = next(r for r in (bun_sc.centromere_rows or []) if r["hap"] == "Donor#1")
    assert int(sc_row["repeat_like_bp"]) == 1500
    assert int(sc_row["short_bp"]) == 220
    assert {u["reason"] for u in bun_sc.unresolved} == {"CEN_REPEAT_LIKE", "CEN_SHORT_RESIDUAL"}


# ---------------------------------------------------------------------------
# 13. Centromere: Step D >1kb Mappy Island Clustering & CEN_SEQUENCE_UNCERTAIN
# ---------------------------------------------------------------------------

def test_tier5_m1m2_13_centromere_step_d_gt1kb_mappy_clustering_and_sequence_uncertain(tmp_path: Path) -> None:
    """White-box target: `panflat/centromere.py`:
    - Step D island clustering when candidate islands are `> 1000 bp` (lines 1798-1813 `cl.aligner = mappy.Aligner(...)`)
    - Step C residual `>= 500 bp` containing IUPAC ambiguity base `'R'` -> `CEN_SEQUENCE_UNCERTAIN` (lines 1684-1712)
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 6301), ("u", 6302), ("cen", 6303), ("v", 6304), ("n4", 6305)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    isl_1300_base = make_complex_seq(1300, seed=6306)
    isl_1300_snp = mutate_seq_snps(isl_1300_base, num_snps=10, seed=6307)  # >99% identical, >1000 bp -> mappy merge!
    isl_1200_other = make_complex_seq(1200, seed=6308)
    # 600 bp complex sequence with 1 IUPAC ambiguity code 'R' (keeps valid 15-mer ratio > 90% so Step C doesn't mask it,
    # then triggers line 1684 `if any(ch not in "ACGTacgt" for ch in c_seq)` -> CEN_SEQUENCE_UNCERTAIN!)
    seq_uncertain = make_complex_seq(300, seed=6309) + "R" + make_complex_seq(299, seed=6310)

    b.add_node("isl_1300_a", isl_1300_base)
    b.add_node("isl_1300_b", isl_1300_snp)
    b.add_node("isl_1200_c", isl_1200_other)
    b.add_node("isl_unc", seq_uncertain)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+", "n4+"])
    b.add_path("H1#1#Chr01", ["n0+", "u+", "cen+", "isl_1300_a+", "v+", "n4+"])
    b.add_path("H2#1#Chr01", ["n0+", "u+", "cen+", "isl_1300_b+", "v+", "n4+"])
    b.add_path("H3#1#Chr01", ["n0+", "u+", "cen+", "isl_1200_c+", "v+", "n4+"])
    b.add_path("H4#1#Chr01", ["n0+", "u+", "cen+", "isl_unc+", "v+", "n4+"])

    b.add_bed_region(500, 750, "CEN_GT1K")
    inputs = b.write("cen_gt1k")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    # H1 and H2 merge into Chr01_CEN_ISLAND_1 (1300 bp, n=2); H3 is Chr01_CEN_ISLAND_2 (1200 bp, n=1)
    cen_evs = [ev for ev in bundle.sv_rows if ev["type"] == "CEN_ISLAND"]
    assert len(cen_evs) == 2
    assert int(cen_evs[0]["n_carrier"]) == 2
    assert int(cen_evs[1]["n_carrier"]) == 1

    # H4's IUPAC 'R' candidate is routed to CEN_SEQUENCE_UNCERTAIN
    assert any(u["reason"] == "CEN_SEQUENCE_UNCERTAIN" for u in bundle.unresolved)
    h4_row = next(r for r in (bundle.centromere_rows or []) if r["hap"] == "H4#1")
    assert int(h4_row["unresolved_bp"]) == 600


# ---------------------------------------------------------------------------
# 14. Normalizer: Zero-Forward-Anchor Unnormalized Contig & Top-Level Empty Main Chain
# ---------------------------------------------------------------------------

def test_tier5_m1m2_14_zero_forward_anchor_unnormalized_contig(tmp_path: Path) -> None:
    """White-box target: `panflat/normalizer.py` (`compute_weighted_lis` line 623 `if n == 0: return [], multi_hit`
    and `extract_deviations` lines 776-796 `if is_top_level and not main_chain`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=6401))
    b.add_node("r2", make_complex_seq(250, seed=6402))
    b.add_node("nov_only", make_complex_seq(180, seed=6403))
    b.add_node("ins_norm", make_complex_seq(150, seed=6404))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # HapNoAnchor has zero reference nodes at all -> n == 0 in compute_weighted_lis,
    # main_chain == [] in extract_deviations -> emitted as UNANCHORED_END!
    b.add_path("HapNoAnchor#1#Chr01", ["nov_only+"])
    b.add_path("HapNormal#1#Chr01", ["r1+", "ins_norm+", "r2+"])

    inputs = b.write("no_fwd_anc")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    unanch = [u for u in bundle.unresolved if u["reason"] == "UNANCHORED_END"]
    assert len(unanch) == 1
    assert int(unanch[0]["length"]) == 180


# ---------------------------------------------------------------------------
# 15. Direct Unit/White-Box Exerciser for Defensive Invariant Guards in M1 & M2
# ---------------------------------------------------------------------------

def test_tier5_m1m2_15_cli_contract_parameter_and_missing_input_guards(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI verification of defensive parameter-range and missing-file contract
    guards in `panflat build` (corresponding internal Python API guards live in `tests/unit/`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("n1", make_complex_seq(120, seed=5901))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    b.add_bed_region(20, 80, "CEN1")
    inputs = b.write("cli_guards")

    # Missing GFA / FASTA files fail cleanly with non-zero exit code and zero Python tracebacks
    res_missing = run_panflat_cli(
        [
            "build",
            "--gfa",
            str(tmp_path / "nonexistent_panflat_test.gfa"),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(tmp_path / "bad_missing"),
        ],
        check=False,
    )
    assert res_missing.returncode != 0 and "Traceback" not in res_missing.stderr

    # Parameter range guards via CLI
    base = [
        "build",
        "--gfa",
        str(inputs.gfa_path),
        "--ref-fa",
        str(inputs.ref_fa_path),
        "--out-prefix",
        str(tmp_path / "bad_param"),
    ]
    for bad_flags in [
        ["--backbone-prefix", "   "],
        ["--backbone-prefix", "Ref#1#", "--max-jump", "0"],
        ["--backbone-prefix", "Ref#1#", "--max-copy", "0"],
        ["--backbone-prefix", "Ref#1#", "--spacer-n", "-1"],
        ["--backbone-prefix", "Ref#1#", "--threads", "0"],
    ]:
        proc = run_panflat_cli([*base, *bad_flags], check=False)
        assert proc.returncode != 0 and "Traceback" not in proc.stderr

    # Default centromere mode when --centromere-bed is provided without explicit --centromere-mode is 'bypass'
    _, bun_default_cen = run_build(inputs)
    assert bun_default_cen is not None
    assert bun_default_cen.summary_kv["centromere_mode"] == "bypass"


# ---------------------------------------------------------------------------
# 16. Full-Pipeline Multi-Subsystem Adversarial Stress Test
# ---------------------------------------------------------------------------

def test_tier5_m1m2_16_full_pipeline_adversarial_stress(tmp_path: Path) -> None:
    """End-to-end stress test combining:
    - Negative-strand reference steps (`r1-`, `u-`)
    - Whole-contig flipped W-line haplotype (`src_strand='-'`, `seq_start=12000`)
    - Centromere `extract-islands` co-mounting a `>1000 bp` clustered `CEN_ISLAND` with a normal `INS` at `ref_start(v)`
    - Top-level `INV` containing a nested hybrid repeat-guard candidate sliced mid-node (`partial_content=1`)
    - Back-to-back tandem repeat cycle folding (`join_before='synthetic'`)
    - Positive and negative strand multi-exon GFF3 genes split across the co-mounted and nested insertions
    - `--threads 2`, `--qc`, `--dump-repeat-sequences`, and standalone `panflat verify`
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    cds_plus = make_coding_seq(80, seed=6601)  # 240 bp
    cds_minus_on_plus = revcomp(make_coding_seq(80, seed=6602))  # 240 bp

    # Reference nodes:
    # r0 [0, 250): contains gene1 (strand '+') exon1 [51, 200]
    # u [250, 500): stored in '-' orientation in GFA so ref uses u-
    # cen [500, 1000): 500 bp centromere reference
    # v [1000, 1300): starts at ref=1000 where gene1 exon2 [951, 1150] crosses ref=1000!
    # r_inv [1300, 1600): inverted in Donor1 with nested hybrid repeat-guard candidate
    # r_end [1600, 2000): contains gene2 (strand '-') [1651, 1890]
    s_r0 = make_complex_seq(50, seed=6610) + cds_plus[:150] + make_complex_seq(50, seed=6611)
    s_u_plus = make_complex_seq(250, seed=6612)
    s_cen = make_complex_seq(450, seed=6613) + cds_plus[150:200]
    s_v = cds_plus[200:240] + make_complex_seq(260, seed=6614)
    s_r_inv = make_complex_seq(300, seed=6615)
    s_r_end = make_complex_seq(50, seed=6616) + cds_minus_on_plus + make_complex_seq(110, seed=6617)

    b.add_node("r0", s_r0)
    b.add_node("u", revcomp(s_u_plus))  # Ref uses u-
    b.add_node("cen", s_cen)
    b.add_node("v", s_v)
    b.add_node("r_inv", s_r_inv)
    b.add_node("r_end", s_r_end)

    # >1kb centromere island (1150 bp)
    b.add_node("isl_big", make_complex_seq(1150, seed=6620))
    # Normal insertion at v (160 bp)
    b.add_node("norm_at_v", make_complex_seq(160, seed=6621))
    # Hybrid repeat-guard candidate inside inversion (300bp complex + 1500bp sat + 300bp complex)
    b.add_node(
        "hyb_inv",
        make_complex_seq(300, seed=6622)
        + make_satellite_seq(20, 1500, seed=6623)
        + make_complex_seq(300, seed=6624),
    )
    # Back-to-back tandem repeats U x 2 + V x 2
    b.add_node("trU", make_complex_seq(80, seed=6625))
    b.add_node("trV", make_complex_seq(90, seed=6626))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u-", "cen+", "v+", "r_inv+", "r_end+"])

    # Donor1: whole-contig reverse W-line carrying isl_big inside centromere,
    # inversion of r_inv with nested hyb_inv, and trU x 2 + trV x 2 before r_end!
    # Forward logical walk: r0+ -> u- -> cen+ -> isl_big+ -> v+ -> [r_inv- -> hyb_inv+] -> trU+ -> trU+ -> trV+ -> trV+ -> r_end+
    # Wait: for [r_inv- -> hyb_inv+] to be a multi-node or single-anchor INV with nested insertion,
    # put hyb_inv between two inverted ref anchors, or put trU+ -> trU+ -> trV+ -> trV+ -> hyb_inv+ between v+ and r_inv-!
    # Let's make Donor1 forward logical walk:
    # r0+ -> u- -> cen+ -> isl_big+ -> v+ -> hyb_inv+ -> r_inv- -> trU+ -> trU+ -> trV+ -> trV+ -> r_end+
    # Whole-contig reverse W-line of that:
    b.add_walk(
        "Donor1",
        "1",
        "ctgStress",
        12000,
        None,
        ["r_end-", "trV-", "trV-", "trU-", "trU-", "r_inv+", "hyb_inv-", "v-", "isl_big-", "cen-", "u+", "r0-"],
    )
    # Donor2: partial contig starting at cen+ adding norm_at_v+ right before v+
    b.add_path("Donor2#1#Chr01", ["cen+", "norm_at_v+", "v+", "r_inv+", "r_end+"])

    b.add_bed_region(500, 950, "CEN_STRESS")
    b.add_gff_gene(
        gene_id="gene_pos",
        mrna_id="mrna_pos",
        strand="+",
        exons=[(51, 200), (951, 1040)],
        cds_list=[(51, 200, 0), (951, 1040, 0)],
    )
    b.add_gff_gene(
        gene_id="gene_neg",
        mrna_id="mrna_neg",
        strand="-",
        exons=[(1651, 1890)],
        cds_list=[(1651, 1890, 0)],
    )

    inputs = b.write("stress_all")
    _, bundle = run_build(
        inputs,
        extra_args=[
            "--centromere-mode", "extract-islands",
            "--centromere-min-island", "500",
            "--repeat-guard", "on",
            "--dump-repeat-sequences",
            "--threads", "2",
            "--qc",
        ],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )
    run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path)


# ===========================================================================
# M3 & M4 White-Box Adversarial Tests (test_tier5_m3m4_01 .. test_tier5_m3m4_14)
# ===========================================================================

# ---------------------------------------------------------------------------
# 17 (M3/M4 #1). Deep Recursive Local Re-Anchoring & Ancestor State Key Guard
# ---------------------------------------------------------------------------

def test_tier5_m3m4_01_recursive_local_reanchoring_and_ancestor_termination(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`reanchor_deviation` lines 265-411,
    `can_transition` lines 153-161, and `_solve_multi_member_component` fallback lines 1132-1161).
    Exercises a 2-level local re-anchoring cascade where internal reference nodes are multi-copy
    at top level `B[0:8)` (`count == 2`) but become locally unique (`count == 1`) inside nested
    sub-intervals `B[1:6)` and `B[1:4)`, plus direct verification of `can_transition` ancestor/metric
    termination guards.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [
        ("r0", 7101),
        ("m", 7102),
        ("loc_in", 7103),
        ("loc_out", 7104),
        ("r6", 7105),
        ("ins_pre", 7106),
        ("ins_post", 7107),
    ]:
        b.add_node(nid, make_complex_seq(220, seed=seed))

    # Top-level reference steps 0..7:
    # 0: r0+ (unique in 0..8)
    # 1: m+ (appears at 1, 3)
    # 2: loc_in+ (appears at 2, 5 -> count=2 in 0..8, count=2 in 1..6, count=1 in 1..4!)
    # 3: m+ (appears at 1, 3)
    # 4: loc_out+ (appears at 4, 7 -> count=2 in 0..8, count=1 in 1..6!)
    # 5: loc_in+ (appears at 2, 5)
    # 6: r6+ (unique in 0..8)
    # 7: loc_out+ (appears at 4, 7)
    b.set_ref_path(
        "Ref#1#Chr01",
        ["r0+", "m+", "loc_in+", "m+", "loc_out+", "loc_in+", "r6+", "loc_out+"],
    )
    # Donor traverses all reference nodes in order and inserts ins_pre+ before loc_in+ (step 2)
    # and ins_post+ before m+ (step 3):
    b.add_path(
        "Donor#1#Chr01",
        [
            "r0+",
            "m+",
            "ins_pre+",
            "loc_in+",
            "ins_post+",
            "m+",
            "loc_out+",
            "loc_in+",
            "r6+",
            "loc_out+",
        ],
    )

    inputs = b.write("reanchor_cascade")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # All intermediate reference steps 1..5 were covered via 2-level local re-anchoring, so NO false DEL!
    assert [ev["type"] for ev in bundle.sv_rows] == ["INS", "INS"]
    # ins_pre is anchored at 1..2 (ref_start of step 2 = 440); ins_post is anchored at 2..3 (ref_start of step 3 = 660)
    assert [int(ev["ref_start"]) for ev in bundle.sv_rows] == [440, 660]
    assert [ev["anchors"] for ev in bundle.sv_rows] == ["1..2", "2..3"]


# ---------------------------------------------------------------------------
# 18 (M3/M4 #2). Secondary Backbone: support vs longest & Tie-Breakers
# ---------------------------------------------------------------------------

def test_tier5_m3m4_02_sec_backbone_support_vs_longest_and_tiebreakers(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`_solve_multi_member_component` lines 1083-1117).
    Verifies:
    - `--sec-backbone support` selects candidate with highest node-weighted distinct-haplotype support
      (ignoring duplicate contigs from the same haplotype).
    - `--sec-backbone longest` selects candidate with largest base length.
    - Deterministic tie-breaking when support and length are equal (`token_str` ASC).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=7201))
    b.add_node("r2", make_complex_seq(250, seed=7202))
    b.add_node("hub", make_complex_seq(100, seed=7203))
    b.add_node("arm_sup", make_complex_seq(200, seed=7204))  # hub+arm_sup = 300 bp, carried by 3 distinct haps
    b.add_node("arm_lng", make_complex_seq(380, seed=7205))  # hub+arm_lng = 480 bp, carried by 1 hap

    # Second locus between r2 and r3 for exact tie-breaking on token_str ("hub2+,tie_A+" vs "hub2+,tie_B+")
    b.add_node("r3", make_complex_seq(250, seed=7206))
    b.add_node("hub2", make_complex_seq(100, seed=7207))
    b.add_node("tie_A", make_complex_seq(150, seed=7208))
    b.add_node("tie_B", make_complex_seq(150, seed=7209))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+"])
    # H1, H2, H3 carry hub+ -> arm_sup+ (plus H1 has a second W-line contig with the same walk)
    b.add_walk("H1", "1", "ctg1", 0, None, ["r1+", "hub+", "arm_sup+", "r2+", "hub2+", "tie_B+", "r3+"])
    b.add_walk("H1", "1", "ctg2", 5000, None, ["r1+", "hub+", "arm_sup+", "r2+"])
    b.add_path("H2#1#Chr01", ["r1+", "hub+", "arm_sup+", "r2+", "hub2+", "tie_A+", "r3+"])
    b.add_path("H3#1#Chr01", ["r1+", "hub+", "arm_sup+", "r2+"])
    # H4 carries the longer hub+ -> arm_lng+
    b.add_path("H4#1#Chr01", ["r1+", "hub+", "arm_lng+", "r2+"])

    inputs = b.write("sec_bb")
    _, bun_sup = run_build(inputs, out_prefix=tmp_path / "sup_out", extra_args=["--sec-backbone", "support"])
    _, bun_lng = run_build(inputs, out_prefix=tmp_path / "lng_out", extra_args=["--sec-backbone", "longest"])
    assert bun_sup is not None and bun_lng is not None
    assert_all_global_invariants(bun_sup, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_lng, inputs.gfa_path, inputs.ref_fa_path)

    # Under support: B_sec at locus 1 is hub+ -> arm_sup+ (from H1#1), and arm_lng+ (from H4#1) is mounted before END
    ev1_sup_pieces = [p for p in bun_sup.inserts if p["event_id"] == "Chr01_INS_1"]
    ev1_lng_pieces = [p for p in bun_lng.inserts if p["event_id"] == "Chr01_INS_1"]
    assert ev1_sup_pieces[0]["src_hap"] == "H1#1"
    assert ev1_lng_pieces[0]["src_hap"] == "H4#1"

    # At locus 2 (tie between tie_A+ on H2#1 and tie_B+ on H1#1 with identical length 250 bp and support 1):
    # token_str "hub2+,tie_A+" < "hub2+,tie_B+", so H2#1 is chosen as B_sec in both modes!
    ev2_sup_pieces = [p for p in bun_sup.inserts if p["event_id"] == "Chr01_INS_2"]
    assert ev2_sup_pieces[0]["src_hap"] == "H2#1"


# ---------------------------------------------------------------------------
# 19 (M3/M4 #3). Union-Find Partial-Node Isolation vs Opposite-Strand Full-Node Sharing
# ---------------------------------------------------------------------------

def test_tier5_m3m4_03_union_find_partial_node_slice_isolation_vs_full_node_sharing(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`solve_parallel_group` Step 2 lines 1667-1697).
    Verifies:
    - Two repeat-guard residuals sliced from the SAME GFA node (`is_partial=True`) do NOT merge
      into a Union-Find component (`is_synthetic=False`, separated by `N * spacer_n`).
    - Two candidates sharing a FULL non-reference node (`is_partial=False`) in opposite orientations
      (`full_shared+` vs `full_shared-`) DO merge into a single Union-Find component (`is_synthetic=True`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=7301))
    b.add_node("r2", make_complex_seq(250, seed=7302))
    b.add_node("r3", make_complex_seq(250, seed=7303))

    # Locus 1 (r1..r2): single hybrid node 500bp complex + 1500bp sat + 500bp complex -> 2 partial slices of same node
    s_hyb = (
        make_complex_seq(500, seed=7304)
        + make_satellite_seq(20, 1500, seed=7305)
        + make_complex_seq(500, seed=7306)
    )
    b.add_node("hyb_same_node", s_hyb)

    # Locus 2 (r2..r3): full_shared (120 bp) traversed in '+' by H2 and '-' by H3, plus common_fwd (100 bp)
    b.add_node("full_shared", make_complex_seq(120, seed=7307))
    b.add_node("common_fwd", make_complex_seq(100, seed=7310))
    b.add_node("arm_plus", make_complex_seq(150, seed=7308))
    b.add_node("arm_minus", make_complex_seq(160, seed=7309))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+"])
    b.add_path("H1#1#Chr01", ["r1+", "hyb_same_node+", "r2+", "r3+"])
    b.add_path("H2#1#Chr01", ["r1+", "r2+", "full_shared+", "common_fwd+", "arm_plus+", "r3+"])
    b.add_path("H3#1#Chr01", ["r1+", "r2+", "full_shared-", "arm_minus+", "common_fwd+", "r3+"])

    inputs = b.write("uf_partial_vs_full")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # At Locus 1 (Chr01_INS_1): the two 500bp partial-node slices from hyb_same_node do NOT form a synthetic
    # component; they become 2 separate representatives separated by a spacer!
    ins1_pieces = [p for p in bundle.inserts if p["event_id"] == "Chr01_INS_1"]
    assert len(ins1_pieces) == 2
    assert ins1_pieces[0]["join_before"] == "source_contiguous"
    assert ins1_pieces[1]["join_before"] == "spacer"

    # At Locus 2 (Chr01_INS_2): H2 and H3 are merged by Union-Find into 1 synthetic component!
    ins2_pieces = [p for p in bundle.inserts if p["event_id"] == "Chr01_INS_2"]
    assert any(p["join_before"] == "synthetic" for p in ins2_pieces)
    assert any(p["members"] == "H2#1,H3#1" for p in ins2_pieces)


# ---------------------------------------------------------------------------
# 20 (M3/M4 #4). Replacement Group R_0 Backbone Merge vs Synthetic Exact Dedup
# ---------------------------------------------------------------------------

def test_tier5_m3m4_04_replacement_group_r0_backbone_merge_vs_synthetic_exact_dedup(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`solve_parallel_group` Step 4 lines 1765-1895).
    Verifies:
    - In a replacement group `G(a, b)` where `R_0 = B(a, b)` is non-empty (`> 1000 bp` to also hit
      `rep0_aligner = mappy.Aligner(...)`), a real candidate with >99% similarity to `R_0` merges
      into `R_0` (`rep_idx = 0`) and is not emitted as a duplicate insertion.
    - Two multi-member synthetic components that are >99% similar (1 SNP difference) do NOT
      similarity-merge because synthetic sequences only exact-deduplicate (`is_synthetic == True`),
      while an exact-sequence synthetic match (`rep.seq == sc.seq`) DOES deduplicate.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=7401))
    s_r2_ref = make_complex_seq(1200, seed=7402)
    b.add_node("r2_ref", s_r2_ref)
    b.add_node("r3", make_complex_seq(250, seed=7403))

    # Real candidate with 5 SNPs vs r2_ref (99.6% identical, >1000 bp -> merges into R_0 via rep0_aligner!)
    b.add_node("r2_near_ref", mutate_seq_snps(s_r2_ref, num_snps=5, seed=7404))

    # Synthetic component A: hubA (100bp) -> armA1 (150bp) and hubA -> armA2 (80bp) -> armA1 (150bp)
    s_hub_a = make_complex_seq(100, seed=7405)
    s_arm_a1 = make_complex_seq(150, seed=7406)
    s_arm_a2 = make_complex_seq(80, seed=7407)
    b.add_node("hubA", s_hub_a)
    b.add_node("armA1", s_arm_a1)
    b.add_node("armA2", s_arm_a2)

    # Synthetic component B: 1 SNP in hubB vs hubA (99% similar synthetic sequence, must NOT merge with A!)
    b.add_node("hubB", mutate_seq_snps(s_hub_a, num_snps=1, seed=7408))
    b.add_node("armB1", s_arm_a1)
    b.add_node("armB2", s_arm_a2)

    # Synthetic component C: exact same sequences as Component A on distinct node IDs (hubC, armC1, armC2)
    # -> produces a synthetic sequence with `rep.seq == sc.seq` and hits exact synthetic dedup (lines 1850-1875)!
    b.add_node("hubC", s_hub_a)
    b.add_node("armC1", s_arm_a1)
    b.add_node("armC2", s_arm_a2)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2_ref+", "r3+"])
    b.add_path("H_r0#1#Chr01", ["r1+", "r2_near_ref+", "r3+"])
    b.add_path("HA1#1#Chr01", ["r1+", "hubA+", "armA1+", "r3+"])
    b.add_path("HA2#1#Chr01", ["r1+", "hubA+", "armA2+", "armA1+", "r3+"])
    b.add_path("HB1#1#Chr01", ["r1+", "hubB+", "armB1+", "r3+"])
    b.add_path("HB2#1#Chr01", ["r1+", "hubB+", "armB2+", "armB1+", "r3+"])
    b.add_path("HC1#1#Chr01", ["r1+", "hubC+", "armC1+", "r3+"])
    b.add_path("HC2#1#Chr01", ["r1+", "hubC+", "armC2+", "armC1+", "r3+"])

    inputs = b.write("r0_and_synth_dedup")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "off"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # H_r0#1 merged into R_0 (so its genotype on the INS event is "0"!),
    # while Component A + Component C exact-deduplicated into one representative (4 haps: HA1, HA2, HC1, HC2)
    # and Component B remained a separate representative (2 haps: HB1, HB2)!
    ins_gt = bundle.genotypes["Chr01_INS_1"]
    assert ins_gt["H_r0#1"] == "0"
    # Component A (HA1, HA2) and Component C (HC1, HC2) exact-deduplicated into the SAME synthetic representative:
    # HA1#1 and HC1#1 share the exact same allele call, and HA2#1 and HC2#1 share the exact same allele call,
    # whereas Component B (HB1, HB2) with 1 SNP did NOT merge with Component A/C and has distinct alleles!
    assert ins_gt["HA1#1"] == ins_gt["HC1#1"]
    assert ins_gt["HA2#1"] == ins_gt["HC2#1"]
    assert ins_gt["HB1#1"] not in (ins_gt["HA1#1"], ins_gt["HA2#1"], "0", ".")
    ins_pieces = [p for p in bundle.inserts if p["event_id"] == "Chr01_INS_1"]
    all_members = [set(str(p["members"]).split(",")) for p in ins_pieces]
    assert {"HA1#1", "HA2#1"} in all_members
    assert {"HB1#1", "HB2#1"} in all_members
    assert {"HC1#1", "HC2#1"} not in all_members


# ---------------------------------------------------------------------------
# 21 (M3/M4 #5). Multi-Group Same Right Anchor Span Ordering & CEN_ISLAND Co-Mount
# ---------------------------------------------------------------------------

def test_tier5_m3m4_05_multi_group_same_right_anchor_span_ordering_and_cen_comount(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`assemble_pan_and_offsets` lines 1987-2090).
    Verifies:
    - At right anchor `v` (step 3), `CEN_ISLAND` mounts first (`ref + N + CEN_ISLAND + N + G(2,3) + v`),
      sharing a single `N * spacer_n` between `CEN_ISLAND` and the following normal group `G(2, 3)`.
    - At right anchor `r6` (step 6), three parallel branch groups `G(5, 6)` (`span=1`), `G(4, 6)` (`span=2`),
      and `G(3, 6)` (`span=3`) are ordered strictly by `group_span = b - a` ASC (`G(5,6)`, then `G(4,6)`,
      then `G(3,6)`), separated by `N * spacer_n`.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx, nid in enumerate(["r0", "u", "cen", "v", "r4", "r5", "r6"]):
        b.add_node(nid, make_complex_seq(250, seed=7501 + idx))

    s_cen_isl = make_complex_seq(600, seed=7510)
    s_g23 = make_complex_seq(110, seed=7511)
    s_g56 = make_complex_seq(120, seed=7512)
    s_g46 = make_complex_seq(130, seed=7513)
    s_g36 = make_complex_seq(140, seed=7514)

    b.add_node("cen_isl", s_cen_isl)
    b.add_node("g23", s_g23)
    b.add_node("g56", s_g56)
    b.add_node("g46", s_g46)
    b.add_node("g36", s_g36)

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "r5+", "r6+"])
    # H_cen: complete centromere traversal carrying cen_isl+ AND g36+ (between v=3 and r6=6, span=3)
    b.add_path("H_cen#1#Chr01", ["r0+", "u+", "cen+", "cen_isl+", "v+", "g36+", "r6+"])
    # H_g23: partial contig starting at cen+ carrying g23+ right before v+ (group_span=1 at b=3),
    # and g46+ between r4=4 and r6=6 (group_span=2 at b=6)
    b.add_path("H_g23#1#Chr01", ["cen+", "g23+", "v+", "r4+", "g46+", "r6+"])
    # H_g56: pure insertion g56+ between r5=5 and r6=6 (group_span=1 at b=6)
    b.add_path("H_g56#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "r5+", "g56+", "r6+"])

    b.add_bed_region(500, 750, "CEN_M")
    inputs = b.write("multi_slot_order")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500", "--spacer-n", "100"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    # Check slot before step 3 (v, ref_start=750):
    # Must be: N*100 + s_cen_isl (600) + N*100 + s_g23 (110) => slot_ins_len = 100 + 600 + 100 + 110 = 910!
    step3_row = next(r for r in bundle.offsets if int(r["ref_step"]) == 3)
    assert int(step3_row["slot_ins_len"]) == 910
    step2_end = int(next(r for r in bundle.offsets if int(r["ref_step"]) == 2)["pan_end"])
    step3_start = int(step3_row["pan_start"])
    assert bundle.pan_seq[step2_end:step3_start] == ("N" * 100) + s_cen_isl + ("N" * 100) + s_g23

    # Check slot before step 6 (r6, ref_start=1500):
    # Ordered by span ASC: G(5,6) [span=1, pure ins -> no leading spacer], then N*100 + G(4,6) [span=2], then N*100 + G(3,6) [span=3]
    # Total slot_ins_len = 120 + 100 + 130 + 100 + 140 = 590!
    step6_row = next(r for r in bundle.offsets if int(r["ref_step"]) == 6)
    assert int(step6_row["slot_ins_len"]) == 590
    step5_end = int(next(r for r in bundle.offsets if int(r["ref_step"]) == 5)["pan_end"])
    step6_start = int(step6_row["pan_start"])
    assert bundle.pan_seq[step5_end:step6_start] == s_g56 + ("N" * 100) + s_g46 + ("N" * 100) + s_g36


# ---------------------------------------------------------------------------
# 22 (M3/M4 #6). Repeat-Guard Multi-Residual Signature, Short Residual & All-Masked '.'
# ---------------------------------------------------------------------------

def test_tier5_m3m4_06_repeat_guard_multi_residual_ordered_signature_and_all_masked_dot(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`screen_and_fold_candidate` lines 747-852,
    and multi-residual signature / multi-allele piece tagging lines 3180-3345).
    Constructs a 4035 bp normal candidate on `DonorSplit#1` that `repeat_mask_intervals` splits into:
    - `c1`: `[0, 500)` (`500 bp >= 50 bp` -> kept residual 1)
    - masked `[500, 2000)` (`1500 bp` -> `REPEAT_LIKE` in `unresolved.tsv`)
    - `c2`: `[2000, 2500)` (`500 bp >= 50 bp` -> kept residual 2)
    - masked `[2500, 4000)` (`1500 bp` -> `REPEAT_LIKE` in `unresolved.tsv`)
    - `c3`: `[4000, 4035)` (`35 bp < 50 bp` -> `SHORT_RESIDUAL` in JSON log)
    Meanwhile `DonorC1Only#1` carries only `c1` (`500 bp`), and `DonorAllMasked#1` carries 1500 bp pure
    satellite (`kept_slices == []` -> genotype `.`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=7601))
    b.add_node("r2", make_complex_seq(250, seed=7602))

    s_c1 = make_complex_seq(500, seed=7603)
    s_sat1 = make_satellite_seq(20, 1500, seed=7604)
    s_c2 = make_complex_seq(500, seed=7605)
    s_sat2 = make_satellite_seq(20, 1220, seed=7606)
    s_tail = make_complex_seq(315, seed=7607)
    s_4035 = s_c1 + s_sat1 + s_c2 + s_sat2 + s_tail
    assert len(s_4035) == 4035

    b.add_node("cand_4035", s_4035)
    b.add_node("cand_c1_only", s_c1)
    b.add_node("cand_all_sat", make_satellite_seq(20, 1500, seed=7608))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("DonorSplit#1#Chr01", ["r1+", "cand_4035+", "r2+"])
    b.add_path("DonorC1Only#1#Chr01", ["r1+", "cand_c1_only+", "r2+"])
    b.add_path("DonorAllMasked#1#Chr01", ["r1+", "cand_all_sat+", "r2+"])

    inputs = b.write("rg_multi_res")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on", "--dump-repeat-sequences"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Verify JSON logs include SPLIT_BY_REPEAT_MASK, SHORT_RESIDUAL (35 bp), REPEAT_LIKE, and ACCEPTED_CANDIDATE
    reasons = [j.get("final_reason") for j in bundle.log_json_records]
    assert "SPLIT_BY_REPEAT_MASK" in reasons
    assert "SHORT_RESIDUAL" in reasons
    assert "REPEAT_LIKE" in reasons
    assert reasons.count("ACCEPTED_CANDIDATE") >= 3

    # Verify DonorAllMasked#1 has genotype "." while DonorSplit#1 and DonorC1Only#1 have distinct called alleles "1" and "2"
    gt_row = bundle.genotypes["Chr01_INS_1"]
    assert gt_row["Ref#1"] == "0"
    assert gt_row["DonorAllMasked#1"] == "."
    assert {gt_row["DonorSplit#1"], gt_row["DonorC1Only#1"]} == {"1", "2"}

    # Verify Representative 1 (s_c1, 500 bp) is shared by both alleles ("1,2" in inserts.tsv allele column)
    # and partial_content=1 is annotated in sv.tsv!
    assert "partial_content=1" in str(bundle.sv_rows[0]["alleles"])
    assert any(str(p["allele"]) == "1,2" for p in bundle.inserts)


# ---------------------------------------------------------------------------
# 23 (M3/M4 #7). Multi-Contig Haplotype Genotype Conflicts Across SV Types
# ---------------------------------------------------------------------------

def test_tier5_m3m4_07_multi_contig_haplotype_genotype_conflicts_across_sv_types(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (`_resolve_haplotype_genotypes` lines 2223-2245
    and per-contig genotype aggregation for `INS`, `DEL`, `INV`, `CNV`, `DUP_TRA`).
    Verifies that when a haplotype (`HapConflict#1`) has two W-line contigs covering the same locus
    with contradictory non-`.` calls, its genotype resolves to `'.'`, whereas `HapConcord#1` with
    two concordant W-line contigs resolves to the called allele.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 12):
        b.add_node(f"r{i}", make_complex_seq(200, seed=7700 + i))
    b.add_node("tr", make_complex_seq(60, seed=7720))
    b.add_node("ins_x", make_complex_seq(120, seed=7721))

    # Reference path:
    # r1 (0), r2 (1) [INS between r1..r2],
    # r3 (2) [DEL of r3 between r2..r4], r4 (3),
    # r5 (4) [INV of r5 between r4..r6], r6 (5),
    # tr+ x 2 (6, 7) [CNV between r6..r7], r7 (8),
    # r8 (9), r9 (10), r10 (11), r11 (12) [DUP_TRA jumping to r1 from between r9..r10 with --max-jump 600]
    b.set_ref_path(
        "Ref#1#Chr01",
        ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+", "tr+", "tr+", "r7+", "r8+", "r9+", "r10+", "r11+"],
    )

    # Variant walk (carries INS ins_x+, DEL of r3+, INV r5-, CNV tr+ x 4, and DUP_TRA r2+ between r9+ and r10+)
    var_steps = [
        "r1+", "ins_x+", "r2+",
        "r4+",
        "r5-", "r6+",
        "tr+", "tr+", "tr+", "tr+", "r7+",
        "r8+", "r9+", "r2+", "r10+", "r11+",
    ]
    # Reference-like walk (0 for INS, DEL, INV, DUP_TRA; 3 copies for CNV to conflict with 4 copies)
    ref_like_steps = [
        "r1+", "r2+",
        "r3+", "r4+",
        "r5+", "r6+",
        "tr+", "tr+", "tr+", "r7+",
        "r8+", "r9+", "r10+", "r11+",
    ]

    # HapConcord#1 has 2 W-line contigs both carrying var_steps -> concordant!
    b.add_walk("HapConcord", "1", "ctg1", 0, None, var_steps)
    b.add_walk("HapConcord", "1", "ctg2", 10000, None, var_steps)

    # HapConflict#1 has 1 W-line contig with var_steps and 1 W-line contig with ref_like_steps -> conflicts at all 5 loci!
    b.add_walk("HapConflict", "1", "ctgA", 0, None, var_steps)
    b.add_walk("HapConflict", "1", "ctgB", 10000, None, ref_like_steps)

    inputs = b.write("multi_ctg_gt")
    _, bundle = run_build(inputs, extra_args=["--max-jump", "600"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert {"INS", "DEL", "INV", "CNV", "DUP_TRA"}.issubset(ev_types)

    for ev in bundle.sv_rows:
        eid = ev["event_id"]
        g = bundle.genotypes[eid]
        assert g["HapConflict#1"] == ".", f"Expected '.' for HapConflict#1 on {eid} ({ev['type']}), got {g['HapConflict#1']}"
        assert g["HapConcord#1"] not in (".", "0"), f"Expected non-ref call for HapConcord#1 on {eid}"


# ---------------------------------------------------------------------------
# 24 (M3/M4 #8). Co-Located SV Events Sharing region_id & Event Priority
# ---------------------------------------------------------------------------

def test_tier5_m3m4_08_colocated_sv_events_shared_region_id_and_priority(tmp_path: Path) -> None:
    """White-box target: `panflat/linearizer.py` (Step 7 region_id grouping lines 3382-3451
    and `LARGE_JUMP` priority over `DUP_TRA`/`DEL`/`INS` lines 2471-2505).
    Verifies:
    - Co-located `INV` on `[250, 500)` and `INS` at `250` are grouped into the same `region_id` (`Chr01_REG_1`).
    - Co-located `DEL`, partial shared `CNV`, and `INS` within `B(a,b)` share `Chr01_REG_2`.
    - A jump exceeding `--max-jump` (`> 600 bp`) is emitted as `LARGE_JUMP`, routing its novel sequence
      to `unresolved.tsv` with matching `region_id` and `event_id`.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 11):
        b.add_node(f"r{i}", make_complex_seq(250, seed=7800 + i))
    b.add_node("tr", make_complex_seq(60, seed=7820))
    b.add_node("ins_co_inv", make_complex_seq(130, seed=7821))
    b.add_node("ins_co_del", make_complex_seq(140, seed=7822))
    b.add_node("nov_lj", make_complex_seq(160, seed=7823))

    b.set_ref_path(
        "Ref#1#Chr01",
        ["r1+", "r2+", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+", "r7+", "r8+", "r9+", "r10+"],
    )
    # H1: inverts r2+ AND inserts ins_co_inv+ at ref=250 (co-located in Chr01_REG_1),
    # then between r3+ and r5+ deletes r4+, expands tr+ from 2 to 4 copies, and inserts ins_co_del+ (co-located in Chr01_REG_2)
    b.add_path(
        "H1#1#Chr01",
        [
            "r1+", "ins_co_inv+", "r2-", "r3+",
            "tr+", "tr+", "tr+", "tr+", "ins_co_del+", "r5+",
            "r6+", "r7+", "r8+", "r9+", "r10+",
        ],
    )
    # H2: jumps from r6+ to r10+ (> 600 bp ref jump across r7, r8, r9 = 750 bp) with nov_lj+
    b.add_path(
        "H2#1#Chr01",
        ["r1+", "r2+", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+", "nov_lj+", "r10+"],
    )

    inputs = b.write("colocated_reg")
    _, bundle = run_build(inputs, extra_args=["--max-jump", "600"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    reg1_types = {ev["type"] for ev in bundle.sv_rows if ev["region_id"] == "Chr01_REG_1"}
    assert reg1_types == {"INS", "INV"}
    reg2_types = {ev["type"] for ev in bundle.sv_rows if ev["region_id"] == "Chr01_REG_2"}
    assert reg2_types == {"DEL", "CNV", "INS"}

    lj_evs = [ev for ev in bundle.sv_rows if ev["type"] == "LARGE_JUMP"]
    assert len(lj_evs) == 1
    lj_unres = [u for u in bundle.unresolved if u["reason"] == "LARGE_JUMP"]
    assert len(lj_unres) == 1
    assert lj_unres[0]["event_id"] == lj_evs[0]["event_id"]
    assert lj_unres[0]["region_id"] == lj_evs[0]["region_id"]


# ---------------------------------------------------------------------------
# 25 (M3/M4 #9). GFF3 Liftover: 3-Level DAG, UTR/Intron, Splice Window w vs w+1 & Micro-CDS
# ---------------------------------------------------------------------------

def test_tier5_m3m4_09_gff3_liftover_multilevel_dag_utr_intron_splice_and_micro_cds(tmp_path: Path) -> None:
    """White-box target: `panflat/gff_liftover.py` (3-level `gene -> transcript -> mRNA -> UTR/exon/CDS`
    relaxation, `five_prime_UTR`/`three_prime_UTR` splitting, combined `utr_insertion,cds_split_by_insertion`
    and `intronic_insertion`, exact `--splice-window` `w=8` vs `w+1=9`, `1 bp` & `2 bp` micro-CDS splits
    on `+` and `-` strands, and `##FASTA` directive stop).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    cds_pos = make_coding_seq(30, seed=7901)  # 90 bp
    cds_neg_rc = revcomp(make_coding_seq(30, seed=7902))  # 90 bp

    # Build reference steps so insertion junctions fall at exact target coordinates:
    # Step 0 (n0, 60 bp): [0, 60) -> 5'UTR is [21, 100] (0-based [20, 100)), so insertion at ref=60 splits 5'UTR!
    # Step 1 (n1, 41 bp): [60, 101) -> CDS1 starts at 101 (0-based 100). Insertion at ref=101 splits CDS1 after 1 bp!
    # Step 2 (n2, 2 bp): [101, 103) -> Insertion at ref=103 splits CDS1 after 2 bp more (1 bp + 2 bp micro-CDS)!
    # Step 3 (n3, 87 bp): [103, 190) -> Finishes 90 bp CDS1 [101, 190] and exon1 [21, 190].
    # Step 4 (n4, 102 bp): [190, 292) -> Intron is (190, 300). Insertion at ref=292 is 8 bp before exon2 start 301 (0-based 300):
    #   |292 - 300| = 8 <= splice_window(8) -> triggers intronic_insertion AND splice_site_proximal_sv=True on exon2!
    # Step 5 (n5, 108 bp): [292, 400) -> Exon2 / 3'UTR is [301, 380].
    # Step 6 (n6, 101 bp): [400, 501) -> Minus-strand gene2 CDS is [500, 589] (0-based [499, 589)).
    # Step 7 (n7, 86 bp): [501, 587) -> Insertion at ref=587 splits minus-strand CDS [499, 589) into 2 bp (5' on '-') + 86 bp + 2 bp!
    # Step 8 (n8, 13 bp): [587, 600) -> Insertion at ref=600 is 9 bp after exon_neg [499, 591] (0-based e0=591):
    #   |600 - 591| = 9 > splice_window(8) -> does NOT trigger splice_site_proximal_sv on exon_neg!
    # Step 9 (n9, 100 bp): [600, 700)
    s_n0 = make_complex_seq(60, seed=7910)
    s_n1 = make_complex_seq(40, seed=7911) + cds_pos[0:1]
    s_n2 = cds_pos[1:3]
    s_n3 = cds_pos[3:90]
    s_n4 = make_complex_seq(102, seed=7912)
    s_n5 = make_complex_seq(108, seed=7913)
    s_n6 = make_complex_seq(99, seed=7914) + cds_neg_rc[0:2]
    s_n7 = cds_neg_rc[2:88]
    s_n8 = cds_neg_rc[88:90] + make_complex_seq(11, seed=7915)
    s_n9 = make_complex_seq(100, seed=7916)

    for idx, s in enumerate([s_n0, s_n1, s_n2, s_n3, s_n4, s_n5, s_n6, s_n7, s_n8, s_n9]):
        b.add_node(f"n{idx}", s)
    for idx in range(1, 7):
        b.add_node(f"ins{idx}", make_complex_seq(80, seed=7930 + idx))

    b.set_ref_path("Ref#1#Chr01", [f"n{i}+" for i in range(10)])
    # Donor inserts at ref=60 (after n0), ref=101 (after n1), ref=103 (after n2),
    # ref=292 (after n4), ref=587 (after n7), and ref=600 (after n8)
    b.add_path(
        "Donor#1#Chr01",
        [
            "n0+", "ins1+",
            "n1+", "ins2+",
            "n2+", "ins3+",
            "n3+",
            "n4+", "ins4+",
            "n5+", "n6+",
            "n7+", "ins5+",
            "n8+", "ins6+",
            "n9+",
        ],
    )

    # 3-level hierarchy: region1 -> gene1 -> mrna1 -> five_prime_UTR, exon1, CDS1, exon2, three_prime_UTR
    b.add_gff_line("Chr01", "src", "region", 21, 380, ".", "+", ".", "ID=reg1;Name=R1")
    b.add_gff_line("Chr01", "src", "gene", 21, 380, ".", "+", ".", "ID=gene1;Parent=reg1")
    b.add_gff_line("Chr01", "src", "mRNA", 21, 380, ".", "+", ".", "ID=mrna1;Parent=gene1;flag_only")
    b.add_gff_line("Chr01", "src", "five_prime_UTR", 21, 100, ".", "+", ".", "ID=utr5_1;Parent=mrna1")
    b.add_gff_line("Chr01", "src", "exon", 21, 190, ".", "+", ".", "ID=ex1_1;Parent=mrna1")
    b.add_gff_line("Chr01", "src", "CDS", 101, 190, ".", "+", "0", "ID=cds1_1;Parent=mrna1")
    b.add_gff_line("Chr01", "src", "exon", 301, 380, ".", "+", ".", "ID=ex1_2;Parent=mrna1")
    b.add_gff_line("Chr01", "src", "three_prime_UTR", 301, 380, ".", "+", ".", "ID=utr3_1;Parent=mrna1")

    # Minus-strand gene2 with 2-bp micro-CDS split at ref=587 (4 bp from 591 -> splice_site_proximal_sv=True on exon1)
    # and second exon [610, 680] (0-based s0=609, distance |600 - 609| = 9 > 8 -> no splice_site_proximal_sv on exon2!)
    b.add_gff_gene(
        gene_id="gene2",
        mrna_id="mrna2",
        strand="-",
        exons=[(500, 591), (610, 680)],
        cds_list=[(500, 589, 0)],
    )

    inputs = b.write("gff_deep")
    # Append a ##FASTA section to verify parser stops cleanly at ##FASTA
    assert inputs.gff_path is not None
    inputs.gff_path.write_text(
        inputs.gff_path.read_text(encoding="utf-8") + "##FASTA\n>Chr01\nACGTACGT\n",
        encoding="utf-8",
    )

    _, bundle = run_build(inputs, extra_args=["--splice-window", "8"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    lifted_text = bundle.lifted_gff_path.read_text(encoding="utf-8")
    assert "sv_status=intronic_insertion,utr_insertion,cds_split_by_insertion" in lifted_text
    assert "ID=ex1_2;Parent=mrna1;splice_site_proximal_sv=True" in lifted_text
    # mrna2.exon2 ([610, 680], 0-based s0=609) is 9 bp from insertion at 600 (> 8), so no splice_site_proximal_sv!
    ex2_2_lines = [l for l in lifted_text.splitlines() if "ID=mrna2.exon2" in l]
    assert len(ex2_2_lines) == 1
    assert "splice_site_proximal_sv=True" not in ex2_2_lines[0]


# ---------------------------------------------------------------------------
# 26 (M3/M4 #10). GFF3 Liftover Contract Error Rejections & Overloads
# ---------------------------------------------------------------------------

def test_tier5_m3m4_10_gff3_liftover_contract_error_rejections(tmp_path: Path) -> None:
    """Opaque-box CLI verification of `gff_liftover.py` validation error branches,
    missing `--ref-gff` file rejection, and comments-only empty GFF3 handling.
    """
    b = SyntheticInputBuilder(tmp_path / "gff_err", chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("n1", make_complex_seq(200, seed=8091))
    b.set_ref_path("Ref#1#Chr01", ["n1+"])
    inputs = b.write("gff_base")

    # 1. Missing GFF3 file rejected by CLI
    missing_proc = run_panflat_cli(
        [
            "build",
            "--gfa",
            str(inputs.gfa_path),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(tmp_path / "missing.gff3"),
            "--out-prefix",
            str(tmp_path / "bad_missing_gff"),
        ],
        check=False,
    )
    assert missing_proc.returncode != 0 and "Traceback" not in missing_proc.stderr

    # 2. Malformed GFF3 lines matrix rejected by CLI
    bad_gff_lines = [
        "Chr01\tsrc\tgene\t1\t100",  # < 9 cols
        "\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1",  # empty seqid
        "Chr02\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1",  # wrong seqid
        "Chr01\t\tgene\t1\t100\t.\t+\t.\tID=g1",  # empty source
        "Chr01\tsrc\t\t1\t100\t.\t+\t.\tID=g1",  # empty ftype
        "Chr01\tsrc\tgene\tabc\t100\t.\t+\t.\tID=g1",  # non-int start
        "Chr01\tsrc\tgene\t0\t100\t.\t+\t.\tID=g1",  # start < 1
        "Chr01\tsrc\tgene\t120\t100\t.\t+\t.\tID=g1",  # start > end
        "Chr01\tsrc\tgene\t1\t250\t.\t+\t.\tID=g1",  # end > ref_len (200)
        "Chr01\tsrc\tgene\t1\t100\t.\t*\t.\tID=g1",  # invalid strand
        "Chr01\tsrc\tCDS\t1\t100\t.\t+\t3\tID=c1",  # invalid phase
    ]
    for idx, bad_line in enumerate(bad_gff_lines):
        bad_gff = tmp_path / f"bad_{idx}.gff3"
        bad_gff.write_text(f"##gff-version 3\n{bad_line}\n", encoding="utf-8")
        proc = run_panflat_cli(
            [
                "build",
                "--gfa",
                str(inputs.gfa_path),
                "--ref-fa",
                str(inputs.ref_fa_path),
                "--backbone-prefix",
                "Ref#1#",
                "--ref-gff",
                str(bad_gff),
                "--out-prefix",
                str(tmp_path / f"bad_gff_out_{idx}"),
            ],
            check=False,
        )
        assert proc.returncode != 0 and "Traceback" not in proc.stderr

    # 3. Comments-only GFF3 succeeds and emits header-only lifted GFF3
    empty_gff = tmp_path / "comments_only.gff3"
    empty_gff.write_text("##gff-version 3\n# comment\n", encoding="utf-8")
    out_emp = tmp_path / "emp_gff_out"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(inputs.gfa_path),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(empty_gff),
            "--out-prefix",
            str(out_emp),
        ],
        check=True,
    )
    assert Path(f"{out_emp}.ref.lifted.gff3").read_text(encoding="utf-8").splitlines() == [
        "##gff-version 3"
    ]


# ---------------------------------------------------------------------------
# 27 (M3/M4 #11). Multi-Chromosome Merge: Heterogeneous Centromere Modes & Haplotype Union
# ---------------------------------------------------------------------------

def test_tier5_m3m4_11_multi_chromosome_merge_heterogeneous_centromere_and_hap_union(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI target: `panflat merge` across 3 chromosomes with heterogeneous
    configurations and haplotype subsets:
    - `Chr01`: `extract-islands` + `--ref-gff`, haplotypes `Ref#1`, `H1#1`, `H2#1`
    - `Chr02`: `bypass` + `--dump-repeat-sequences`, haplotypes `Ref#1`, `H2#1`, `H3#1`
    - `Chr03`: `centromere-mode=off`, haplotypes `Ref#1`, `H1#1`, `H4#1` + `UNANCHORED_END`
    Also tests CLI `panflat merge` error branches (missing prefix list, empty prefix list,
    missing per-chrom file, duplicate chromosome name).
    """
    # Chr01: extract-islands + GFF3
    b1 = SyntheticInputBuilder(tmp_path / "c1", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 8101), ("u", 8102), ("cen", 8103), ("v", 8104)]:
        b1.add_node(nid, make_complex_seq(250, seed=seed))
    b1.add_node("isl1", make_complex_seq(550, seed=8105))
    b1.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b1.add_path("H1#1#Chr01", ["n0+", "u+", "cen+", "isl1+", "v+"])
    b1.add_path("H2#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b1.add_bed_region(500, 750, "CEN1")
    b1.add_gff_gene("g1", "m1", "+", [(20, 199)], [(20, 199, 0)])
    inp1 = b1.write("chr01")
    run_build(inp1, extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"])

    # Chr02: bypass + dump-repeat-sequences
    b2 = SyntheticInputBuilder(tmp_path / "c2", chrom="Chr02", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 8111), ("u", 8112), ("cen", 8113), ("v", 8114)]:
        b2.add_node(nid, make_complex_seq(250, seed=seed))
    b2.add_node("byp_ins", make_complex_seq(180, seed=8115))
    b2.add_node("norm_ins2", make_complex_seq(150, seed=8116))
    b2.set_ref_path("Ref#1#Chr02", ["n0+", "u+", "cen+", "v+"])
    b2.add_path("H2#1#Chr02", ["n0+", "u+", "cen+", "byp_ins+", "v+"])
    b2.add_path("H3#1#Chr02", ["n0+", "norm_ins2+", "u+", "cen+", "v+"])
    b2.add_bed_region(500, 750, "CEN2")
    inp2 = b2.write("chr02")
    run_build(inp2, extra_args=["--centromere-mode", "bypass", "--dump-repeat-sequences"])

    # Chr03: centromere off + UNANCHORED_END
    b3 = SyntheticInputBuilder(tmp_path / "c3", chrom="Chr03", backbone_prefix="Ref#1#")
    b3.add_node("r1", make_complex_seq(250, seed=8121))
    b3.add_node("r2", make_complex_seq(250, seed=8122))
    b3.add_node("ins3", make_complex_seq(140, seed=8123))
    b3.add_node("unanch3", make_complex_seq(130, seed=8124))
    b3.set_ref_path("Ref#1#Chr03", ["r1+", "r2+"])
    b3.add_path("H1#1#Chr03", ["r1+", "ins3+", "r2+"])
    b3.add_path("H4#1#Chr03", ["unanch3+", "r1+", "r2+"])
    inp3 = b3.write("chr03")
    run_build(inp3)

    merged_prefix = tmp_path / "merged_all"
    _, mbun = run_merge([inp1.out_prefix, inp2.out_prefix, inp3.out_prefix], merged_prefix)
    assert mbun is not None
    assert list(mbun.pan_fa.keys()) == ["Chr01", "Chr02", "Chr03"]

    # Verify chain IDs 1, 2, 3
    chain_hdrs = [
        ln for ln in mbun.chain_path.read_text(encoding="utf-8").splitlines() if ln.startswith("chain ")
    ]
    assert [ln.split()[12] for ln in chain_hdrs] == ["1", "2", "3"]

    # Verify haplotype union in genotypes.tsv with '.' filled for missing chromosome haplotypes
    assert mbun.genotype_haps == ["Ref#1", "H1#1", "H2#1", "H3#1", "H4#1"]
    chr1_eid = next(k for k in mbun.genotypes if str(k).startswith("Chr01_"))
    assert mbun.genotypes[chr1_eid]["H3#1"] == "." and mbun.genotypes[chr1_eid]["H4#1"] == "."
    chr2_eid = next(k for k in mbun.genotypes if str(k).startswith("Chr02_"))
    assert mbun.genotypes[chr2_eid]["H1#1"] == "." and mbun.genotypes[chr2_eid]["H4#1"] == "."
    chr3_eid = next(k for k in mbun.genotypes if str(k).startswith("Chr03_"))
    assert mbun.genotypes[chr3_eid]["H2#1"] == "." and mbun.genotypes[chr3_eid]["H3#1"] == "."

    # Verify centromere.tsv has Chr01 and Chr02 rows only
    assert mbun.centromere_rows is not None
    assert {r["chrom"] for r in mbun.centromere_rows} == {"Chr01", "Chr02"}
    assert mbun.lifted_gff_path.is_file()
    assert int(mbun.summary_kv["chrom_count"]) == 3

    # CLI panflat merge error branches
    p_missing_list = run_panflat_cli(
        ["merge", "--prefix-list", str(tmp_path / "no_such_list.txt"), "--out-prefix", str(tmp_path / "bad_m1")],
        check=False,
    )
    assert p_missing_list.returncode != 0
    empty_list = tmp_path / "empty_list.txt"
    empty_list.write_text("# comment only\n\n", encoding="utf-8")
    p_empty_list = run_panflat_cli(
        ["merge", "--prefix-list", str(empty_list), "--out-prefix", str(tmp_path / "bad_m2")],
        check=False,
    )
    assert p_empty_list.returncode != 0
    pm3, _ = run_merge([tmp_path / "missing_prefix"], tmp_path / "bad_m3", check=False)
    assert pm3.returncode != 0
    pm4, _ = run_merge([inp1.out_prefix, inp1.out_prefix], tmp_path / "bad_m4", check=False)
    assert pm4.returncode != 0


# ---------------------------------------------------------------------------
# 28 (M3/M4 #12). Standalone `panflat verify` Exhaustive Tamper Detection Matrix
# ---------------------------------------------------------------------------

def test_tier5_m3m4_12_verify_exhaustive_tamper_detection_matrix(tmp_path: Path) -> None:
    """Opaque-box CLI target: standalone `panflat verify` tamper detection across all 10
    output surfaces (`pan.fa` ref & spacer bases, `offsets.tsv`, `ref_to_pan.chain`,
    `inserts.tsv`, `centromere.tsv`, `unresolved.fa`, `ref.lifted.gff3`, `summary.tsv`
    `gfa_sha256`, and missing output file).
    """
    import shutil

    b = SyntheticInputBuilder(tmp_path / "base", chrom="Chr01", backbone_prefix="Ref#1#")
    cds = make_coding_seq(40, seed=8201)  # 120 bp
    b.add_node("n0", make_complex_seq(50, seed=8202) + cds + make_complex_seq(80, seed=8203))
    b.add_node("u", make_complex_seq(250, seed=8204))
    b.add_node("cen", make_complex_seq(250, seed=8205))
    b.add_node("v", make_complex_seq(250, seed=8206))
    b.add_node("isl", make_complex_seq(550, seed=8207))
    b.add_node("unanch", make_complex_seq(120, seed=8208))

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b.add_path("H1#1#Chr01", ["n0+", "u+", "cen+", "isl+", "v+"])
    b.add_path("H2#1#Chr01", ["unanch+", "n0+", "u+", "cen+", "v+"])
    b.add_bed_region(500, 750, "CEN_V")
    b.add_gff_gene("g1", "m1", "+", [(51, 170)], [(51, 170, 0)])

    inputs = b.write("ver_base")
    run_build(inputs, extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"])
    assert inputs.gff_path is not None and inputs.bed_path is not None
    run_verify(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--ref-gff", inputs.gff_path, "--centromere-bed", inputs.bed_path],
        check=True,
    )

    def _clone_prefix(tag: str) -> Path:
        dst_dir = tmp_path / f"tamper_{tag}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in inputs.out_prefix.parent.glob(f"{inputs.out_prefix.name}.*"):
            shutil.copy2(f, dst_dir / f.name)
        return dst_dir / inputs.out_prefix.name

    # 1. Tamper pan.fa reference base at pos 0
    p1 = _clone_prefix("pan_ref")
    fa_txt = Path(f"{p1}.pan.fa").read_text(encoding="utf-8").splitlines()
    flipped_base = "T" if fa_txt[1][0] != "T" else "A"
    fa_txt[1] = flipped_base + fa_txt[1][1:]
    Path(f"{p1}.pan.fa").write_text("\n".join(fa_txt) + "\n", encoding="utf-8")
    assert run_verify(p1, inputs.gfa_path, inputs.ref_fa_path, extra_args=["--ref-gff", inputs.gff_path, "--centromere-bed", inputs.bed_path], check=False).returncode != 0

    # 2. Tamper pan.fa spacer base ('N' -> 'A')
    p2 = _clone_prefix("pan_spacer")
    fa_txt2 = Path(f"{p2}.pan.fa").read_text(encoding="utf-8").splitlines()
    seq2 = list(fa_txt2[1])
    seq2[fa_txt2[1].index("N")] = "A"
    fa_txt2[1] = "".join(seq2)
    Path(f"{p2}.pan.fa").write_text("\n".join(fa_txt2) + "\n", encoding="utf-8")
    assert run_verify(p2, inputs.gfa_path, inputs.ref_fa_path, extra_args=["--ref-gff", inputs.gff_path, "--centromere-bed", inputs.bed_path], check=False).returncode != 0

    # 3. Tamper offsets.tsv
    p3 = _clone_prefix("offsets")
    off_lines = Path(f"{p3}.offsets.tsv").read_text(encoding="utf-8").splitlines()
    cols = off_lines[1].split("\t")
    cols[3] = "5"
    off_lines[1] = "\t".join(cols)
    Path(f"{p3}.offsets.tsv").write_text("\n".join(off_lines) + "\n", encoding="utf-8")
    assert run_verify(p3, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 4. Tamper ref_to_pan.chain
    p4 = _clone_prefix("chain")
    ch_txt = Path(f"{p4}.ref_to_pan.chain").read_text(encoding="utf-8").replace("# block\t0\t", "# block\t10\t")
    Path(f"{p4}.ref_to_pan.chain").write_text(ch_txt, encoding="utf-8")
    assert run_verify(p4, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 5. Tamper inserts.tsv
    p5 = _clone_prefix("inserts")
    ins_lines = Path(f"{p5}.inserts.tsv").read_text(encoding="utf-8").splitlines()
    icols = ins_lines[1].split("\t")
    icols[9] = "-" if icols[9] == "+" else "+"
    ins_lines[1] = "\t".join(icols)
    Path(f"{p5}.inserts.tsv").write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    assert run_verify(p5, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 6. Tamper centromere.tsv
    p6 = _clone_prefix("cen")
    c_lines = Path(f"{p6}.centromere.tsv").read_text(encoding="utf-8").splitlines()
    for i in range(1, len(c_lines)):
        ccols = c_lines[i].split("\t")
        if ccols[20] != ".":
            ccols[25] = "99999"
            c_lines[i] = "\t".join(ccols)
    Path(f"{p6}.centromere.tsv").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    assert run_verify(p6, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 7. Tamper unresolved.fa
    p7 = _clone_prefix("unres_fa")
    Path(f"{p7}.unresolved.fa").write_text(">Chr01_UNRES_1\nTTTTTTTTTT\n", encoding="utf-8")
    assert run_verify(p7, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 8. Tamper ref.lifted.gff3
    p8 = _clone_prefix("gff")
    g_txt = Path(f"{p8}.ref.lifted.gff3").read_text(encoding="utf-8").replace("\t170\t", "\t167\t")
    Path(f"{p8}.ref.lifted.gff3").write_text(g_txt, encoding="utf-8")
    assert run_verify(p8, inputs.gfa_path, inputs.ref_fa_path, extra_args=["--ref-gff", inputs.gff_path], check=False).returncode != 0

    # 9. Tamper summary.tsv gfa_sha256
    p9 = _clone_prefix("sha")
    s_txt = Path(f"{p9}.summary.tsv").read_text(encoding="utf-8")
    Path(f"{p9}.summary.tsv").write_text(
        "\n".join(
            f"gfa_sha256\t{'f' * 64}" if ln.startswith("gfa_sha256\t") else ln
            for ln in s_txt.splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_verify(p9, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0

    # 10. Missing required output file
    p10 = _clone_prefix("missing")
    Path(f"{p10}.sv.tsv").unlink()
    assert run_verify(p10, inputs.gfa_path, inputs.ref_fa_path, check=False).returncode != 0


# ---------------------------------------------------------------------------
# 29 (M3/M4 #13). `--qc` Bucket Boundaries (50, 149, 150 bp) & Union Cap <= 1.0
# ---------------------------------------------------------------------------

def test_tier5_m3m4_13_qc_bucket_boundaries_50_149_150_and_union_cap(tmp_path: Path) -> None:
    """Opaque-box CLI target: `panflat build --qc` bucket boundaries (`50`, `149`, `150 bp`)
    and multi-hit union interval capping (`ratio <= 1.0`).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_hom_250 = make_complex_seq(250, seed=8301)
    b.add_node("r1", s_hom_250)
    b.add_node("r2", make_complex_seq(250, seed=8302))
    b.add_node("r3", s_hom_250)
    b.add_node("r4", make_complex_seq(250, seed=8303))
    b.add_node("r5", s_hom_250)

    b.add_node("ins_50", make_complex_seq(50, seed=8304))
    b.add_node("ins_149", make_complex_seq(149, seed=8305))
    b.add_node("ins_150", make_complex_seq(150, seed=8306))
    b.add_node("ins_250_hom", mutate_seq_snps(s_hom_250, num_snps=2, seed=8307))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+"])
    b.add_path(
        "Donor#1#Chr01",
        ["r1+", "ins_50+", "r2+", "ins_149+", "r3+", "ins_150+", "r4+", "ins_250_hom+", "r5+"],
    )

    inputs = b.write("qc_bounds")
    _, bundle = run_build(inputs, extra_args=["--qc"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert bundle.summary_kv["qc_enabled"] == "true"
    assert int(bundle.summary_kv["qc_short_50_149_count"]) == 2
    assert int(bundle.summary_kv["qc_short_50_149_bp"]) == 199
    assert bundle.summary_kv["qc_short_50_149_lengths"] == "50,149"
    assert int(bundle.summary_kv["qc_long_ge150_count"]) == 2
    assert int(bundle.summary_kv["qc_long_ge150_bp"]) == 400
    assert int(bundle.summary_kv["qc_redundancy_0.98_bp"]) == 250
    assert 0.0 < float(bundle.summary_kv["qc_redundancy_0.98_ratio"]) <= 1.0


# ---------------------------------------------------------------------------
# 30 (M3/M4 #14). CLI Flag Validation & Multithreaded Shuffled GFA Determinism
# ---------------------------------------------------------------------------

def test_tier5_m3m4_14_cli_entrypoints_flags_and_multithreaded_shuffled_determinism(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI target: invalid CLI flag rejections via `run_panflat_cli` and
    deterministic multithreaded (`--threads 4`) execution on a shuffled GFA input vs
    `--threads 1` on the unshuffled GFA input.
    """
    import random

    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx, nid in enumerate(["r0", "u", "cen", "v", "r4", "r5"]):
        b.add_node(nid, make_complex_seq(250, seed=8400 + idx))
    b.add_node("isl_a", make_complex_seq(550, seed=8410))
    b.add_node("ins_1", make_complex_seq(140, seed=8411))
    b.add_node("ins_2", make_complex_seq(160, seed=8412))
    b.add_node(
        "hyb_rg",
        make_complex_seq(250, seed=8413)
        + make_satellite_seq(20, 1500, seed=8414)
        + make_complex_seq(250, seed=8415),
    )

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "r5+"])
    b.add_path("H1#1#Chr01", ["r0+", "ins_1+", "u+", "cen+", "isl_a+", "v+", "r4-", "r5+"])
    b.add_path("H2#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "hyb_rg+", "r5+"])
    b.add_walk("H3", "1", "ctgW", 4000, None, ["r5-", "ins_2-", "r4-", "v-", "cen-", "u-", "r0-"])
    b.add_bed_region(500, 750, "CEN_DET")
    b.add_gff_gene("g1", "m1", "+", [(20, 199)], [(20, 199, 0)])

    inputs = b.write("det_orig")
    assert inputs.gff_path is not None and inputs.bed_path is not None

    base_cmd = [
        "build",
        "--gfa",
        str(inputs.gfa_path),
        "--ref-fa",
        str(inputs.ref_fa_path),
        "--backbone-prefix",
        "Ref#1#",
        "--out-prefix",
        str(tmp_path / "bad_cli"),
    ]
    bad_arg_sets = [
        [*base_cmd, "--cluster-cov", "0.0"],
        [*base_cmd, "--cluster-cov", "1.2"],
        [*base_cmd, "--cluster-id", "0.0"],
        [*base_cmd, "--cluster-win-id", "1.5"],
        [*base_cmd, "--sec-backbone", "invalid"],
        [*base_cmd, "--repeat-guard", "maybe"],
        [*base_cmd, "--centromere-mode", "invalid"],
        [*base_cmd, "--centromere-mode", "extract-islands"],
        [*base_cmd, "--centromere-bed", str(inputs.bed_path), "--centromere-mode", "off"],
        [*base_cmd, "--centromere-min-island", "600"],
        [*base_cmd, "--centromere-bed", str(inputs.bed_path), "--centromere-mode", "extract-islands", "--centromere-min-island", "20"],
        [*base_cmd, "--min-sv-len", "0"],
        [*base_cmd, "--splice-window", "-1"],
    ]
    for bad_argv in bad_arg_sets:
        proc = run_panflat_cli(bad_argv, check=False)
        assert proc.returncode != 0 and "Traceback" not in proc.stderr

    orig_lines = inputs.gfa_path.read_text(encoding="utf-8").splitlines()
    hdr_lines = [ln for ln in orig_lines if ln.startswith("H\t")]
    body_lines = [ln for ln in orig_lines if not ln.startswith("H\t") and ln.strip()]
    rng = random.Random(8499)
    rng.shuffle(body_lines)
    shuf_gfa = tmp_path / "det_shuf.gfa"
    shuf_gfa.write_text("\n".join(hdr_lines + body_lines) + "\n", encoding="utf-8")

    out_t1 = tmp_path / "out_t1"
    out_t4 = tmp_path / "out_t4"
    common_flags = [
        "--ref-fa",
        str(inputs.ref_fa_path),
        "--backbone-prefix",
        "Ref#1#",
        "--ref-gff",
        str(inputs.gff_path),
        "--centromere-bed",
        str(inputs.bed_path),
        "--centromere-mode",
        "extract-islands",
        "--centromere-min-island",
        "500",
        "--dump-repeat-sequences",
        "--qc",
    ]
    run_panflat_cli(
        ["build", "--gfa", str(inputs.gfa_path), "--out-prefix", str(out_t1), "--threads", "1", *common_flags],
        check=True,
    )
    run_panflat_cli(
        ["build", "--gfa", str(shuf_gfa), "--out-prefix", str(out_t4), "--threads", "4", *common_flags],
        check=True,
    )
    run_verify(
        out_t4,
        shuf_gfa,
        inputs.ref_fa_path,
        extra_args=["--ref-gff", inputs.gff_path, "--centromere-bed", inputs.bed_path],
        check=True,
    )

    for ext in [
        ".pan.fa",
        ".offsets.tsv",
        ".ref_to_pan.chain",
        ".sv.tsv",
        ".genotypes.tsv",
        ".inserts.tsv",
        ".centromere.tsv",
        ".ref.lifted.gff3",
        ".unresolved.tsv",
        ".unresolved.fa",
    ]:
        assert Path(f"{out_t1}{ext}").read_text(encoding="utf-8") == Path(f"{out_t4}{ext}").read_text(encoding="utf-8"), (
            f"Determinism mismatch in {ext} between threads=1 unshuffled and threads=4 shuffled GFA"
        )


# ===========================================================================
# M7 Deep Code-Review Hardening & Bug-Fix Verification CLI E2E Suite
# ===========================================================================


def test_tier5_m7_01_w_walk_int_gfa_full_sv_reanchor_and_centromere_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #1 (CLI E2E): Pure W-line integer GFA (`1..20`, activating compact
    `LazyStepTuple` fast paths) exercising `DEL` (empty `LazyStepTuple` branch), `CNV`,
    `DUP_TRA`, `LARGE_JUMP`, local re-anchoring across multi-copy nodes, and centromere
    `extract-islands` & `bypass` via `panflat build` + `panflat verify`.
    """
    b = SyntheticInputBuilder(tmp_path / "m7_w", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 20):
        b.add_node(str(i), make_complex_seq(200, seed=9800 + i))
    b.add_node("20", make_complex_seq(600, seed=9820))

    ref_toks = [
        "1+", "2+", "3+", "4+",
        "5+", "6+", "5+",
        "7+", "8+",
        "9+", "9+", "10+",
        "11+", "12+", "13+", "14+", "15+", "16+",
    ]
    alt1_toks = [
        "1+", "2+", "3+", "20+", "4+",
        "5+", "17+", "6+", "5+",
        "8+",
        "9+", "9+", "9+", "9+", "10+",
        "11+", "12+", "13+", "14+", "8+", "15+", "16+",
    ]
    alt2_toks = [
        "1+", "2+", "3+", "4+",
        "5+", "6+", "5+",
        "7+", "8+",
        "9+", "9+", "10+",
        "11+", "18+", "15+", "16+",
    ]

    # Write pure W-line integer GFA (no P-lines at all)
    links: set[tuple[str, str]] = set()
    for w in (ref_toks, alt1_toks, alt2_toks):
        for x, y in zip(w[:-1], w[1:]):
            links.add((x[:-1], y[:-1]))
    gfa_lines = ["H\tVN:Z:1.0"]
    for i in range(1, 21):
        gfa_lines.append(f"S\t{i}\t{b.nodes[str(i)]}")
    for u, v in sorted(links, key=lambda kv: (int(kv[0]), int(kv[1]))):
        gfa_lines.append(f"L\t{u}\t+\t{v}\t+\t0M")
    gfa_lines.append("W\tRef\t1\tChr01\t0\t*\t" + "".join(f">{t[:-1]}" for t in ref_toks))
    gfa_lines.append("W\tAlt1\t1\tctgA\t500\t*\t" + "".join(f">{t[:-1]}" for t in alt1_toks))
    gfa_lines.append("W\tAlt2\t1\tctgB\t900\t*\t" + "".join(f">{t[:-1]}" for t in alt2_toks))

    gfa_path = tmp_path / "pure_w.gfa"
    gfa_path.write_text("\n".join(gfa_lines) + "\n", encoding="utf-8")
    ref_fa_path = tmp_path / "pure_w.fa"
    ref_fa_path.write_text(
        ">Chr01\n" + "".join(b.nodes[t[:-1]] for t in ref_toks) + "\n",
        encoding="utf-8",
    )
    bed_path = tmp_path / "pure_w.bed"
    bed_path.write_text("Chr01\t400\t600\tCEN_W\n", encoding="utf-8")

    for mode in ("extract-islands", "bypass"):
        out_p = tmp_path / f"out_w_{mode}"
        extra = ["--centromere-mode", mode]
        if mode == "extract-islands":
            extra += ["--centromere-min-island", "500"]
        run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gfa_path),
                "--ref-fa",
                str(ref_fa_path),
                "--backbone-prefix",
                "Ref#1#",
                "--centromere-bed",
                str(bed_path),
                "--max-jump",
                "500",
                *extra,
                "--out-prefix",
                str(out_p),
            ],
            check=True,
        )
        run_verify(out_p, gfa_path, ref_fa_path, extra_args=["--centromere-bed", bed_path], check=True)
        bun = OutputBundle(out_p)
        assert_all_global_invariants(bun, gfa_path, ref_fa_path)
        assert_centromere_accounting_and_offsets(bun)
        sv_types = {r["type"] for r in bun.sv_rows}
        assert {"INS", "DEL", "CNV", "DUP_TRA", "LARGE_JUMP"}.issubset(sv_types)
        if mode == "extract-islands":
            assert "CEN_ISLAND" in sv_types
        else:
            assert "CEN_ISLAND" not in sv_types


def test_tier5_m7_02_pansn_backbone_prefix_disambiguation_and_verify_override_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #2 (CLI E2E):
    - `--backbone-prefix SampleA` vs `SampleA2#1#Chr10A` / `SampleAB#1#Chr10A`
    - `--backbone-prefix SampleA#1` vs `SampleA#10#Chr10A`
    - `--backbone-prefix SampleA#0#` fallback to `SampleA#1#Chr10A`
    - Multi-record PanSN `--ref-fa` selection and ambiguous/unmatched rejection
    - Standalone `panflat verify --backbone-prefix ...` positive AND negative (`WrongSample#`
      and `SampleB#1#` failing verification even when `summary.tsv` recorded valid `ref_path`).
    """
    s_a = make_complex_seq(180, seed=9831)
    s_b = make_complex_seq(180, seed=9832)
    s_ins = make_complex_seq(90, seed=9833)
    gfa_path = tmp_path / "pansn.gfa"
    gfa_path.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s_a}",
                f"S\t2\t{s_b}",
                f"S\t3\t{s_ins}",
                "L\t1\t+\t3\t+\t0M",
                "P\tSampleA#1#Chr10A\t1+\t*",
                "P\tSampleA2#1#Chr10A\t2+\t*",
                "P\tSampleAB#1#Chr10A\t2+\t*",
                "P\tSampleA#10#Chr10A\t2+\t*",
                "P\tSampleB#1#Chr10A\t1+,3+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    multi_fa = tmp_path / "pansn_multi.fa"
    multi_fa.write_text(
        f">SampleB#1#Chr10A\n{s_a + s_ins}\n"
        f">SampleA2#1#Chr10A\n{s_b}\n"
        f">SampleA#10#Chr10A\n{s_b}\n"
        f">SampleA#1#Chr10A\n{s_a}\n"
        f">SampleA#1#Chr10B\n{s_b}\n",
        encoding="utf-8",
    )

    for idx, ok_bp in enumerate(["SampleA", "SampleA#1", "SampleA#0#", "SampleA#1#"]):
        out_p = tmp_path / f"ok_bp_{idx}"
        run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gfa_path),
                "--ref-fa",
                str(multi_fa),
                "--backbone-prefix",
                ok_bp,
                "--out-prefix",
                str(out_p),
            ],
            check=True,
        )
        bun = OutputBundle(out_p)
        assert bun.summary_kv["chrom"] == "Chr10A"
        assert int(bun.summary_kv["ref_len"]) == len(s_a)

    # Ambiguous / unmatched multi-record plain FASTA (>Chr10A + >Chr10B without '#') fails CLI
    plain_multi = tmp_path / "plain_multi.fa"
    plain_multi.write_text(f">Chr10A\n{s_a}\n>Chr10B\n{s_b}\n", encoding="utf-8")
    p_plain = run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(plain_multi),
            "--backbone-prefix",
            "SampleA#1#",
            "--out-prefix",
            str(tmp_path / "bad_plain"),
        ],
        check=False,
    )
    assert p_plain.returncode != 0 and "Traceback" not in p_plain.stderr

    # Standalone panflat verify --backbone-prefix positive & negative override checks
    built_prefix = tmp_path / "ok_bp_3"
    run_verify(built_prefix, gfa_path, multi_fa, check=True)
    run_panflat_cli(
        [
            "verify",
            "--prefix",
            str(built_prefix),
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(multi_fa),
            "--backbone-prefix",
            "SampleA#0#",
        ],
        check=True,
    )
    p_wrong1 = run_panflat_cli(
        [
            "verify",
            "--prefix",
            str(built_prefix),
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(multi_fa),
            "--backbone-prefix",
            "WrongSample#",
        ],
        check=False,
    )
    assert p_wrong1.returncode != 0
    p_wrong2 = run_panflat_cli(
        [
            "verify",
            "--prefix",
            str(built_prefix),
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(multi_fa),
            "--backbone-prefix",
            "SampleB#1#",
        ],
        check=False,
    )
    assert p_wrong2.returncode != 0


def test_tier5_m7_03_noncanonical_zero_padded_and_sparse_50m_node_ids_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #3 (CLI E2E): Coexisting zero-padded `"01"`, `"001"`, canonical `"1"`,
    and huge integer `"999999999999999999999999999999"` node IDs, plus sparse integer
    `S 1` + `S 50000000` completing rapidly via CLI `panflat build` + `panflat verify`.
    """
    s01 = make_complex_seq(120, seed=9841)
    s001 = make_complex_seq(120, seed=9842)
    s1 = make_complex_seq(120, seed=9843)
    shuge = make_complex_seq(120, seed=9844)
    gfa_noncanon = tmp_path / "noncanon.gfa"
    huge_id = "999999999999999999999999999999"
    gfa_noncanon.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t01\t{s01}",
                f"S\t001\t{s001}",
                f"S\t1\t{s1}",
                f"S\t{huge_id}\t{shuge}",
                f"L\t01\t+\t{huge_id}\t+\t0M",
                "L\t01\t+\t001\t+\t0M",
                "L\t001\t+\t1\t+\t0M",
                f"L\t1\t+\t{huge_id}\t+\t0M",
                f"P\tRef#1#Chr01\t01+,{huge_id}+\t*",
                f"P\tAlt#1#Chr01\t01+,001+,1+,{huge_id}+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa_noncanon = tmp_path / "noncanon.fa"
    fa_noncanon.write_text(f">Chr01\n{s01 + shuge}\n", encoding="utf-8")
    out_nc = tmp_path / "out_nc"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_noncanon),
            "--ref-fa",
            str(fa_noncanon),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out_nc),
        ],
        check=True,
    )
    run_verify(out_nc, gfa_noncanon, fa_noncanon, check=True)
    bun_nc = OutputBundle(out_nc)
    assert len(bun_nc.sv_rows) == 1 and int(bun_nc.inserts[0]["pan_end"]) - int(bun_nc.inserts[0]["pan_start"]) == 240

    # Sparse canonical integer IDs: S 1 and S 50000000
    s_50m = make_complex_seq(150, seed=9845)
    gfa_sparse = tmp_path / "sparse50m.gfa"
    gfa_sparse.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s1}",
                f"S\t2\t{s01}",
                f"S\t50000000\t{s_50m}",
                "L\t1\t+\t50000000\t+\t0M",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t50000000\t+\t0M",
                "P\tRef#1#Chr01\t1+,50000000+\t*",
                "P\tAlt#1#Chr01\t1+,2+,50000000+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa_sparse = tmp_path / "sparse50m.fa"
    fa_sparse.write_text(f">Chr01\n{s1 + s_50m}\n", encoding="utf-8")
    out_sp = tmp_path / "out_sp50m"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_sparse),
            "--ref-fa",
            str(fa_sparse),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out_sp),
        ],
        check=True,
    )
    run_verify(out_sp, gfa_sparse, fa_sparse, check=True)


def test_tier5_m7_04_qc_self_hit_masking_91pct_homologs_and_cen_ref_hits_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #4 (CLI E2E): Two `1000 bp` inserted pieces ~91.5% identical (`85` SNPs)
    report `qc_redundancy_0.90_bp == 2000` (`100%`) and `qc_redundancy_0.95_bp == 0` via
    `panflat build --qc` (confirming 100% self-hit in `pan.fa` no longer suppresses ~91.5%
    homologs), plus a `CEN_ISLAND` with ~92% homology to a non-centromeric reference arm.
    """
    b = SyntheticInputBuilder(tmp_path / "m7_qc", chrom="Chr01", backbone_prefix="Ref#1#")
    s_arm_hom = make_complex_seq(1000, seed=9850)
    b.add_node("r1", s_arm_hom)
    b.add_node("u", make_complex_seq(300, seed=9851))
    b.add_node("cen", make_complex_seq(300, seed=9852))
    b.add_node("v", make_complex_seq(300, seed=9853))
    b.add_node("r2", make_complex_seq(300, seed=9854))
    b.add_node("r3", make_complex_seq(300, seed=9855))

    import random as _rnd

    p1 = make_complex_seq(1000, seed=9856)
    p2 = mutate_seq_snps(p1, positions=sorted(_rnd.Random(9857).sample(range(30, 970), 75)), seed=9857)
    cen_isl = mutate_seq_snps(s_arm_hom, positions=sorted(_rnd.Random(9858).sample(range(30, 970), 75)), seed=9858)
    b.add_node("p1", p1)
    b.add_node("p2", p2)
    b.add_node("cisl", cen_isl)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "u+", "cen+", "v+", "r2+", "r3+"])
    b.add_path("Alt#1#Chr01", ["r1+", "u+", "cen+", "cisl+", "v+", "p1+", "r2+", "p2+", "r3+"])
    b.add_bed_region(1300, 1600, "CEN_QC")
    inputs = b.write("m7_qc")

    # Test both without centromere (pure 2 pieces: 2000 bp) and with centromere island (3000 bp)
    out_pure2 = tmp_path / "out_pure2"
    gfa_pure2 = tmp_path / "pure2.gfa"
    fa_pure2 = tmp_path / "pure2.fa"
    r_a, r_b, r_c = make_complex_seq(300, seed=9861), make_complex_seq(300, seed=9862), make_complex_seq(300, seed=9863)
    gfa_pure2.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{r_a}",
                f"S\t2\t{r_b}",
                f"S\t3\t{r_c}",
                f"S\t4\t{p1}",
                f"S\t5\t{p2}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t1\t+\t4\t+\t0M",
                "L\t4\t+\t2\t+\t0M",
                "L\t2\t+\t5\t+\t0M",
                "L\t5\t+\t3\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+\t*",
                "P\tAlt#1#Chr01\t1+,4+,2+,5+,3+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa_pure2.write_text(f">Chr01\n{r_a + r_b + r_c}\n", encoding="utf-8")
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_pure2),
            "--ref-fa",
            str(fa_pure2),
            "--backbone-prefix",
            "Ref#1#",
            "--qc",
            "--out-prefix",
            str(out_pure2),
        ],
        check=True,
    )
    bun_p2 = OutputBundle(out_pure2)
    assert int(bun_p2.summary_kv["qc_long_ge150_bp"]) == 2000
    assert int(bun_p2.summary_kv["qc_redundancy_0.90_bp"]) == 2000
    assert int(bun_p2.summary_kv["qc_redundancy_0.95_bp"]) == 0

    _, bun_cen = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500", "--qc"],
    )
    assert bun_cen is not None
    assert int(bun_cen.summary_kv["qc_cen_island_ge150_bp"]) == 1000
    assert int(bun_cen.summary_kv["qc_cen_island_ref_hit_0.90_bp"]) == 1000
    assert int(bun_cen.summary_kv["qc_cen_island_ref_hit_0.95_bp"]) == 0


def test_tier5_m7_05_gff3_multi_parent_isoforms_liftover_and_verify_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #5 (CLI E2E): Shared `exon` and `CDS` rows carrying `Parent=tx1,tx2`
    (comma-separated multiple parent transcripts) lifted across internal CDS and intronic
    insertions and verified by `panflat verify --ref-gff` with 100% reconstructed mRNA &
    protein sequence identity for both `tx1` and `tx2`.
    """
    cds_up = make_coding_seq(31, seed=9871)[:-3]  # 90 bp: ATG + 29 sense codons (no stop)
    intron = make_complex_seq(90, seed=9872)
    shared_orf = make_coding_seq(60, seed=9873)  # 180 bp: ATG + 58 sense codons + TAA
    cds_sh1 = shared_orf[:90]
    cds_sh2 = shared_orf[90:]
    ins_intron = make_complex_seq(70, seed=9875)
    ins_cds = make_complex_seq(80, seed=9876)

    gfa_path = tmp_path / "mp_iso.gfa"
    gfa_path.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{cds_up}",
                f"S\t2\t{intron[:45]}",
                f"S\t3\t{intron[45:]}",
                f"S\t4\t{cds_sh1}",
                f"S\t5\t{cds_sh2}",
                f"S\t6\t{ins_intron}",
                f"S\t7\t{ins_cds}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t3\t+\t4\t+\t0M",
                "L\t4\t+\t5\t+\t0M",
                "L\t2\t+\t6\t+\t0M",
                "L\t6\t+\t3\t+\t0M",
                "L\t4\t+\t7\t+\t0M",
                "L\t7\t+\t5\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+,4+,5+\t*",
                "P\tAlt#1#Chr01\t1+,2+,6+,3+,4+,7+,5+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref_seq = cds_up + intron + cds_sh1 + cds_sh2
    fa_path = tmp_path / "mp_iso.fa"
    fa_path.write_text(f">Chr01\n{ref_seq}\n", encoding="utf-8")

    gff_path = tmp_path / "mp_iso.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "Chr01\tsrc\tgene\t1\t360\t.\t+\t.\tID=g1",
                "Chr01\tsrc\tmRNA\t1\t360\t.\t+\t.\tID=tx1;Parent=g1",
                "Chr01\tsrc\tmRNA\t181\t360\t.\t+\t.\tID=tx2;Parent=g1",
                "Chr01\tsrc\texon\t1\t90\t.\t+\t.\tID=ex1;Parent=tx1",
                "Chr01\tsrc\tCDS\t1\t90\t.\t+\t0\tID=cds1;Parent=tx1",
                "Chr01\tsrc\texon\t181\t360\t.\t+\t.\tID=ex_shared;Parent=tx1,tx2",
                "Chr01\tsrc\tCDS\t181\t360\t.\t+\t0\tID=cds_shared;Parent=tx1,tx2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_p = tmp_path / "out_mp_iso"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(gff_path),
            "--out-prefix",
            str(out_p),
        ],
        check=True,
    )
    run_verify(out_p, gfa_path, fa_path, extra_args=["--ref-gff", gff_path], check=True)

    bun = OutputBundle(out_p)
    pan_seq = bun.pan_fa["Chr01"]
    ex_by_tx: dict[str, list[tuple[int, int]]] = {"tx1": [], "tx2": []}
    tx_status: dict[str, str] = {}
    for row in bun.lifted_gff_rows:
        attrs = row["attributes"]
        if row["type"] == "mRNA":
            tx_status[attrs["ID"]] = attrs.get("sv_status", "")
        elif row["type"] == "exon":
            for pid in attrs.get("Parent", "").split(","):
                if pid in ex_by_tx:
                    ex_by_tx[pid].append((int(row["start"]) - 1, int(row["end"])))

    assert "".join(pan_seq[s:e] for s, e in sorted(ex_by_tx["tx1"])) == cds_up + cds_sh1 + cds_sh2
    assert "".join(pan_seq[s:e] for s, e in sorted(ex_by_tx["tx2"])) == cds_sh1 + cds_sh2
    assert "intronic_insertion" in tx_status["tx1"] and "cds_split_by_insertion" in tx_status["tx1"]
    assert tx_status["tx2"] == "cds_split_by_insertion"


def test_tier5_m7_06_hierarchical_merge_colliding_ids_custom_spacer_and_reversed_ref_verify_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #2 & #6 (CLI E2E):
    - Multi-chromosome `panflat merge` across 3 chromosomes built with `--spacer-n 50` and
      `--qc`, with cross-GFA colliding `src_path` (`scaf_shared`), `exported_fasta_id`,
      `region_id` (`CEN_SHARED`), and `event_id`.
    - Hierarchical `panflat merge` (`merge([chr1, chr2])` + `chr3`) preserving exact
      per-chromosome `Chr02::scaf_shared` / `Chr03::scaf_shared` attribution and unique
      per-chromosome `summary.tsv` keys.
    - Standalone `panflat verify` on the merged bundle against a multi-sequence PanSN
      `--ref-fa` whose record order (`Chr03`, `Chr02`, `Chr01`) is reversed relative to
      `prefix_list` order (`Chr01`, `Chr02`, `Chr03`).
    """
    chrom_specs = [("Chr01", 10), ("Chr02", 20), ("Chr03", 30)]
    per_chrom_prefixes: list[Path] = []
    comb_gfa_lines = ["H\tVN:Z:1.0"]
    ref_records_map: dict[str, str] = {}

    for chrom, nbase in chrom_specs:
        s1 = make_complex_seq(200, seed=9900 + nbase + 1)
        s_u = make_complex_seq(200, seed=9900 + nbase + 2)
        s_cen = make_complex_seq(200, seed=9900 + nbase + 3)
        s_v = make_complex_seq(200, seed=9900 + nbase + 4)
        ins_seq = make_complex_seq(180, seed=9900 + nbase + 5)
        unanch = make_complex_seq(120, seed=9900 + nbase + 6)
        cen_isl = make_complex_seq(550, seed=9900 + nbase + 7)

        alt_path = "scaf_c1" if chrom == "Chr01" else "scaf_shared"
        gfa_c_lines = [
            "H\tVN:Z:1.0",
            f"S\t{nbase+1}\t{s1}",
            f"S\t{nbase+2}\t{s_u}",
            f"S\t{nbase+3}\t{s_cen}",
            f"S\t{nbase+4}\t{s_v}",
            f"S\t{nbase+5}\t{ins_seq}",
            f"S\t{nbase+6}\t{unanch}",
            f"S\t{nbase+7}\t{cen_isl}",
            f"L\t{nbase+6}\t+\t{nbase+1}\t+\t0M",
            f"L\t{nbase+1}\t+\t{nbase+2}\t+\t0M",
            f"L\t{nbase+1}\t+\t{nbase+5}\t+\t0M",
            f"L\t{nbase+5}\t+\t{nbase+2}\t+\t0M",
            f"L\t{nbase+2}\t+\t{nbase+3}\t+\t0M",
            f"L\t{nbase+3}\t+\t{nbase+4}\t+\t0M",
            f"L\t{nbase+3}\t+\t{nbase+7}\t+\t0M",
            f"L\t{nbase+7}\t+\t{nbase+4}\t+\t0M",
            f"P\tSampleA#1#{chrom}\t{nbase+1}+,{nbase+2}+,{nbase+3}+,{nbase+4}+\t*",
            f"P\t{alt_path}\t{nbase+6}+,{nbase+1}+,{nbase+5}+,{nbase+2}+,{nbase+3}+,{nbase+7}+,{nbase+4}+\t*",
        ]
        gfa_c = tmp_path / f"{chrom}.gfa"
        gfa_c.write_text("\n".join(gfa_c_lines) + "\n", encoding="utf-8")
        for gl in gfa_c_lines[1:]:
            if not gl.startswith("P\tscaf_shared\t"):
                comb_gfa_lines.append(gl)
            else:
                comb_gfa_lines.append(gl.replace("P\tscaf_shared\t", f"P\t{chrom}::scaf_shared\t", 1))

        ref_seq_c = s1 + s_u + s_cen + s_v
        ref_records_map[chrom] = ref_seq_c
        fa_c = tmp_path / f"{chrom}.fa"
        fa_c.write_text(f">SampleA#1#{chrom}\n{ref_seq_c}\n", encoding="utf-8")
        bed_c = tmp_path / f"{chrom}.bed"
        bed_c.write_text(f"{chrom}\t400\t600\tCEN_SHARED\n", encoding="utf-8")

        out_c = tmp_path / f"built_{chrom}"
        run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gfa_c),
                "--ref-fa",
                str(fa_c),
                "--backbone-prefix",
                "SampleA#1#",
                "--centromere-bed",
                str(bed_c),
                "--centromere-mode",
                "extract-islands",
                "--centromere-min-island",
                "500",
                "--spacer-n",
                "50",
                "--qc",
                "--out-prefix",
                str(out_c),
            ],
            check=True,
        )
        per_chrom_prefixes.append(out_c)

    # Hierarchical merge via CLI: merge([Chr01, Chr02]) -> m12, then merge([m12, Chr03]) -> m123
    m12 = tmp_path / "m12"
    run_merge([per_chrom_prefixes[0], per_chrom_prefixes[1]], m12)
    m123 = tmp_path / "m123"
    _, bun_123 = run_merge([m12, per_chrom_prefixes[2]], m123)
    assert bun_123 is not None

    # Reversed multi-sequence PanSN FASTA (Chr03, Chr02, Chr01 + distractor SampleB records)
    rev_multi_fa = tmp_path / "rev_multi.fa"
    rev_multi_fa.write_text(
        f">SampleB#1#Chr03\n{make_complex_seq(800, seed=9991)}\n"
        f">SampleA#1#Chr03\n{ref_records_map['Chr03']}\n"
        f">SampleA#1#Chr02\n{ref_records_map['Chr02']}\n"
        f">SampleB#1#Chr01\n{make_complex_seq(800, seed=9992)}\n"
        f">SampleA#1#Chr01\n{ref_records_map['Chr01']}\n",
        encoding="utf-8",
    )
    comb_gfa = tmp_path / "comb_123.gfa"
    comb_gfa.write_text("\n".join(comb_gfa_lines) + "\n", encoding="utf-8")

    # Standalone panflat verify passes on reversed PanSN multi-FASTA
    run_verify(m123, comb_gfa, rev_multi_fa, check=True)

    # Colliding scaf_shared on Chr02 and Chr03 must be qualified as Chr02::scaf_shared and Chr03::scaf_shared
    ins_by_chrom = [(r["chrom"], r["src_path"]) for r in bun_123.inserts]
    assert ("Chr02", "Chr02::scaf_shared") in ins_by_chrom
    assert ("Chr03", "Chr03::scaf_shared") in ins_by_chrom
    assert ("Chr02", "Chr01::scaf_shared") not in ins_by_chrom

    # And wrong --backbone-prefix on merged verify fails
    p_m_wrong = run_panflat_cli(
        [
            "verify",
            "--prefix",
            str(m123),
            "--gfa",
            str(comb_gfa),
            "--ref-fa",
            str(rev_multi_fa),
            "--backbone-prefix",
            "WrongSample#",
        ],
        check=False,
    )
    assert p_m_wrong.returncode != 0


def test_tier5_m7_07_centromere_duplicate_flanking_anchor_and_repeat_guard_boundaries_cli(
    tmp_path: Path,
) -> None:
    """M7 Checklist #7 (CLI E2E):
    - Centromere flanking node `u` or `v` visited twice inside a donor contig (`ambiguous` /
      `partial` status, never false `complete`).
    - Repeat-guard window boundaries (`499`, `500`, `999`, `1000`, `1001 bp`), exact `0.30`
      vs `< 0.30` singleton_ratio boundary (`180/600` vs `179/600`), and `unknown` window
      breaking adjacent `low` runs via `panflat build` + `panflat verify`.
    """
    b = SyntheticInputBuilder(tmp_path / "m7_cen_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for idx, nid in enumerate(["r0", "u", "cen", "v", "r1", "r2", "r3", "r4", "r5", "r6"]):
        b.add_node(nid, make_complex_seq(200, seed=9950 + idx))
    b.add_node("cisl", make_complex_seq(600, seed=9965))

    sat_big = make_satellite_seq(20, 2600, seed=9601)
    sat_614 = make_satellite_seq(20, 614, seed=901)
    comp_614 = make_complex_seq(614, seed=902)
    b.add_node("sat499", sat_big[:499])  # < 500 bp -> NOT repeat-masked, kept!
    b.add_node("sat500", sat_big[:500])  # == 500 bp -> short-block masked!
    b.add_node("sat1001", sat_big[:1001])  # == 1001 bp -> 2-window masked!
    b.add_node("eq030", sat_614[:434] + comp_614[434:614])  # 180/600 == 0.30 -> kept!
    b.add_node("lt030", sat_614[:435] + comp_614[435:614])  # 179/600 < 0.30 -> masked!
    b.add_node("unk_brk", sat_big[:1000] + ("N" * 200) + sat_big[1200:2500])  # unknown breaks low runs -> kept!

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen+", "v+", "r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    b.add_path("DupU#1#Chr01", ["r0+", "u+", "cen+", "u+", "cisl+", "v+", "r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    b.add_path("DupV#1#Chr01", ["r0+", "u+", "cisl+", "v+", "cen+", "v+", "r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    b.add_path(
        "RG#1#Chr01",
        [
            "r0+", "u+", "cen+", "v+",
            "sat499+", "r1+",
            "sat500+", "r2+",
            "sat1001+", "r3+",
            "eq030+", "r4+",
            "lt030+", "r5+",
            "unk_brk+", "r6+",
        ],
    )
    b.add_bed_region(400, 600, "CEN_DUP")
    inputs = b.write("cen_rg")
    assert inputs.bed_path is not None

    _, bun = run_build(
        inputs,
        extra_args=[
            "--centromere-mode",
            "extract-islands",
            "--centromere-min-island",
            "500",
            "--repeat-guard",
            "on",
            "--dump-repeat-sequences",
        ],
    )
    assert bun is not None
    run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, extra_args=["--centromere-bed", inputs.bed_path], check=True)

    assert bun.centromere_rows is not None
    cen_by_hap = {r["hap"]: r for r in bun.centromere_rows}
    for h in ("DupU#1", "DupV#1"):
        assert cen_by_hap[h]["status"] in ("ambiguous", "partial")
        assert cen_by_hap[h]["length_ratio"] == "."

    # sat500 (500 bp), sat1001 (1001 bp), and lt030 (614 bp) are masked as REPEAT_LIKE;
    # sat499 (499 bp), eq030 (614 bp), and unk_brk (2500 bp total) are KEPT as INS!
    rep_lens = sorted(int(r["length"]) for r in bun.unresolved if r["reason"] == "REPEAT_LIKE")
    assert rep_lens == [500, 614, 1001]
    ins_lens = sorted(int(r["pan_end"]) - int(r["pan_start"]) for r in bun.inserts)
    assert [499, 614, 2500] == ins_lens


def test_tier5_m7_08_malformed_fasta_gfa_w_overlap_and_cli_bounds_rejections(
    tmp_path: Path,
) -> None:
    """M7 Checklist #8 (CLI E2E): Opaque-box CLI rejection matrix (non-zero exit code,
    clean error message, zero Python tracebacks) across malformed FASTA files, GFA `LN:i:`
    mismatch, non-`0M` L/P overlaps, overlapping W-line intervals on `(hap_id, seqid)`,
    duplicate P+W source intervals, and invalid `verify --spacer-n -1`.
    """
    valid_gfa = tmp_path / "ok.gfa"
    valid_gfa.write_text("H\tVN:Z:1.0\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\n", encoding="utf-8")
    valid_fa = tmp_path / "ok.fa"
    valid_fa.write_text(">Chr01\nACGTACGT\n", encoding="utf-8")

    # 1. Malformed FASTA files
    for idx, bad_fa in enumerate(
        [
            "",
            "ACGTACGT\n>Chr01\nACGT\n",
            ">\nACGTACGT\n",
            ">Chr01\n>Chr02\nACGT\n",
            ">Chr01\nACGTACGT\n>Chr01\nACGTACGT\n",
            ">Chr01\nACGT1234\n",
        ]
    ):
        fp = tmp_path / f"bad_fa_{idx}.fa"
        fp.write_text(bad_fa, encoding="utf-8")
        proc = run_panflat_cli(
            [
                "build",
                "--gfa",
                str(valid_gfa),
                "--ref-fa",
                str(fp),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(tmp_path / f"out_bfa_{idx}"),
            ],
            check=False,
        )
        assert proc.returncode != 0 and "Traceback" not in proc.stderr

    # 2. Malformed GFA files
    for idx, bad_gfa in enumerate(
        [
            "H\tVN:Z:1.0\nS\t1\tACGTACGT\tLN:i:99\nP\tRef#1#Chr01\t1+\t*\n",
            "H\tVN:Z:1.0\nS\t1\tACGT$CGT\nP\tRef#1#Chr01\t1+\t*\n",
            "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tACGT\nL\t1\t+\t2\t+\t4M\nP\tRef#1#Chr01\t1+,2+\t*\n",
            "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tACGT\nL\t1\t+\t2\t+\t0M\nP\tRef#1#Chr01\t1+,2+\t4M\n",
            "H\tVN:Z:1.0\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\nW\tAlt\t1\tctg1\t0\t8\t>1\nW\tAlt\t1\tctg1\t4\t12\t>1\n",
            "H\tVN:Z:1.0\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\nP\tAlt#1#Chr01\t1+\t*\nW\tAlt\t1\tChr01\t0\t8\t>1\n",
        ]
    ):
        gp = tmp_path / f"bad_gfa_{idx}.gfa"
        gp.write_text(bad_gfa, encoding="utf-8")
        proc = run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gp),
                "--ref-fa",
                str(valid_fa),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(tmp_path / f"out_bgfa_{idx}"),
            ],
            check=False,
        )
        assert proc.returncode != 0 and "Traceback" not in proc.stderr

    # 3. Invalid verify --spacer-n -1 rejected cleanly
    out_ok = tmp_path / "out_ok"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(valid_gfa),
            "--ref-fa",
            str(valid_fa),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out_ok),
        ],
        check=True,
    )
    proc_ver = run_panflat_cli(
        [
            "verify",
            "--prefix",
            str(out_ok),
            "--gfa",
            str(valid_gfa),
            "--ref-fa",
            str(valid_fa),
            "--spacer-n",
            "-1",
        ],
        check=False,
    )
    assert proc_ver.returncode != 0 and "Traceback" not in proc_ver.stderr


def test_tier5_m7_09_r2_d1_four_way_isomorphic_graph_multi_hit_isolation_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D1` (CLI E2E): 4-way isomorphic graph encoding parity across
    integer `P`, integer `W` (`seq_start > 0`), string `P`, and mixed `0`/`01`/`¹` node IDs.
    Verifies via `panflat build` + `panflat verify` that neither a `< min_sv_len` micro-SNP
    nor a co-occurring pure `DEL` on a `multi_hit` non-carrier haplotype leaks `multi_hit=1`
    onto `INS` alleles, and all 4 encodings produce identical `pan.fa`, `sv.tsv`, and
    `genotypes.tsv`.
    """
    s_a = make_complex_seq(140, seed=10101)
    s_b = make_complex_seq(140, seed=10102)
    s_del = make_complex_seq(80, seed=10103)
    s_c = make_complex_seq(140, seed=10104)
    s_ins1 = make_complex_seq(110, seed=10105)
    s_mic = make_complex_seq(20, seed=10106)
    s_ins2 = make_complex_seq(95, seed=10107)

    ref_fa = tmp_path / "r2_d1_ref.fa"
    ref_fa.write_text(f">Chr01\n{s_a + s_b + s_del + s_c}\n", encoding="utf-8")

    def _build_isomorphic_gfa(
        nids: tuple[str, str, str, str, str, str, str],
        mode: str,
    ) -> list[str]:
        na, nb, ndel, nc, nins1, nmic, nins2 = nids
        seqs = (s_a, s_b, s_del, s_c, s_ins1, s_mic, s_ins2)
        ref_w = [na, nb, ndel, nc]
        alt1_w = [na, nins1, nb, ndel, nc]
        alt2_w = [na, nmic, nb, nc, na]
        alt3_w = [na, nb, nins2, nc]

        edges: list[tuple[str, str]] = []
        seen_e: set[tuple[str, str]] = set()
        for w in (ref_w, alt1_w, alt2_w, alt3_w):
            for u, v in zip(w[:-1], w[1:]):
                if (u, v) not in seen_e:
                    seen_e.add((u, v))
                    edges.append((u, v))

        lines = ["H\tVN:Z:1.0"]
        for nid, seq in zip(nids, seqs):
            lines.append(f"S\t{nid}\t{seq}")
        for u, v in edges:
            lines.append(f"L\t{u}\t+\t{v}\t+\t0M")

        if mode == "p":
            lines.append("P\tRef#1#Chr01\t" + ",".join(f"{x}+" for x in ref_w) + "\t*")
            lines.append("P\tAlt1#1#Chr01\t" + ",".join(f"{x}+" for x in alt1_w) + "\t*")
            lines.append("P\tAlt2#1#Chr01\t" + ",".join(f"{x}+" for x in alt2_w) + "\t*")
            lines.append("P\tAlt3#1#Chr01\t" + ",".join(f"{x}+" for x in alt3_w) + "\t*")
        elif mode == "w":
            lines.append("W\tRef\t1\tChr01\t200\t*\t" + "".join(f">{x}" for x in ref_w))
            lines.append("W\tAlt1\t1\tctg1\t350\t*\t" + "".join(f">{x}" for x in alt1_w))
            lines.append("W\tAlt2\t1\tctg2\t500\t*\t" + "".join(f">{x}" for x in alt2_w))
            lines.append("W\tAlt3\t1\tctg3\t750\t*\t" + "".join(f">{x}" for x in alt3_w))
        else:
            lines.append("P\tRef#1#Chr01\t" + ",".join(f"{x}+" for x in ref_w) + "\t*")
            lines.append("P\tAlt1#1#Chr01\t" + ",".join(f"{x}+" for x in alt1_w) + "\t*")
            lines.append("W\tAlt2\t1\tctg2\t420\t*\t" + "".join(f">{x}" for x in alt2_w))
            lines.append("W\tAlt3\t1\tctg3\t880\t*\t" + "".join(f">{x}" for x in alt3_w))
        return lines

    encodings = {
        "int_p": _build_isomorphic_gfa(("1", "2", "3", "4", "5", "6", "7"), "p"),
        "int_w": _build_isomorphic_gfa(("1", "2", "3", "4", "5", "6", "7"), "w"),
        "str_p": _build_isomorphic_gfa(
            ("nd_a", "nd_b", "nd_del", "nd_c", "nd_ins1", "nd_mic", "nd_ins2"), "p"
        ),
        "mixed_0_01_sup1": _build_isomorphic_gfa(
            ("0", "01", "1", "¹", "001", "²", "nd_ins2"), "mixed"
        ),
    }

    pan_texts: dict[str, str] = {}
    sv_texts: dict[str, str] = {}
    gt_texts: dict[str, str] = {}
    for ename, gfa_lines in encodings.items():
        gfa_p = tmp_path / f"{ename}.gfa"
        gfa_p.write_text("\n".join(gfa_lines) + "\n", encoding="utf-8")
        out_p = tmp_path / f"out_{ename}"
        run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gfa_p),
                "--ref-fa",
                str(ref_fa),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(out_p),
            ],
            check=True,
        )
        run_verify(out_p, gfa_p, ref_fa, check=True)
        bun = OutputBundle(out_p)
        assert_all_global_invariants(bun, gfa_p, ref_fa)

        ins_rows = [r for r in bun.sv_rows if r["type"] == "INS"]
        del_rows = [r for r in bun.sv_rows if r["type"] == "DEL"]
        assert len(ins_rows) == 2 and len(del_rows) == 1
        for ir in ins_rows:
            assert "multi_hit=1" not in str(ir["alleles"])
        assert "multi_hit=1" in str(del_rows[0]["alleles"])

        ins_slot01 = next(r for r in ins_rows if int(r["ref_start"]) == len(s_a))
        assert bun.genotypes[ins_slot01["event_id"]]["Alt1#1"] == "1"
        assert bun.genotypes[ins_slot01["event_id"]]["Alt2#1"] == "0"

        pan_texts[ename] = Path(f"{out_p}.pan.fa").read_text(encoding="utf-8")
        sv_texts[ename] = Path(f"{out_p}.sv.tsv").read_text(encoding="utf-8")
        gt_texts[ename] = Path(f"{out_p}.genotypes.tsv").read_text(encoding="utf-8")

    assert pan_texts["int_p"] == pan_texts["int_w"] == pan_texts["str_p"] == pan_texts["mixed_0_01_sup1"]
    assert sv_texts["int_p"] == sv_texts["int_w"] == sv_texts["str_p"] == sv_texts["mixed_0_01_sup1"]
    assert gt_texts["int_p"] == gt_texts["int_w"] == gt_texts["str_p"] == gt_texts["mixed_0_01_sup1"]


def test_tier5_m7_10_r2_d2_exact_p_w_src_path_collision_large_w_offset_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D2` (CLI E2E): Exact `P` + `W` `src_path` collision when
    `W.seq_start >= len(P)` (`P\tAlt#1#Chr01:1000-1200` + `W\tAlt\t1\tChr01\t1000\t1200`
    or wildcard `*` computing `1200`) is cleanly rejected by `panflat build` across both
    integer and string GFA paths with non-zero exit code and zero Python tracebacks, while
    disjoint `P` (`Alt#1#Chr01`) + high-offset `W` (`1000..1360`) succeeds cleanly.
    """
    s1 = make_complex_seq(100, seed=10201)
    s2 = make_complex_seq(100, seed=10202)
    s3 = make_complex_seq(100, seed=10203)
    s4 = make_complex_seq(160, seed=10204)
    fa_path = tmp_path / "r2_d2.fa"
    fa_path.write_text(f">Chr01\n{s1 + s2 + s3}\n", encoding="utf-8")

    bad_variants = [
        # Integer fast GFA path, explicit seq_end=1200 where seq_start=1000 >= len(P)=260
        [
            "H\tVN:Z:1.0",
            f"S\t1\t{s1}",
            f"S\t2\t{s2}",
            f"S\t3\t{s3}",
            f"S\t4\t{s4}",
            "L\t1\t+\t2\t+\t0M",
            "L\t2\t+\t3\t+\t0M",
            "L\t4\t+\t2\t+\t0M",
            "P\tRef#1#Chr01\t1+,2+,3+\t*",
            "P\tAlt#1#Chr01:1000-1200\t4+,2+\t*",
            "W\tAlt\t1\tChr01\t1000\t1200\t>1>2",
        ],
        # String GFA path, wildcard '*' end computing 1000 + 200 = 1200 -> exact src_path collision
        [
            "H\tVN:Z:1.0",
            f"S\tn1\t{s1}",
            f"S\tn2\t{s2}",
            f"S\tn3\t{s3}",
            f"S\tn4\t{s4}",
            "L\tn1\t+\tn2\t+\t0M",
            "L\tn2\t+\tn3\t+\t0M",
            "L\tn4\t+\tn2\t+\t0M",
            "P\tRef#1#Chr01\tn1+,n2+,n3+\t*",
            "P\tAlt#1#Chr01:1000-1200\tn4+,n2+\t*",
            "W\tAlt\t1\tChr01\t1000\t*\t>n1>n2",
        ],
    ]
    for idx, lines in enumerate(bad_variants):
        gfa_bad = tmp_path / f"collide_{idx}.gfa"
        gfa_bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        proc = run_panflat_cli(
            [
                "build",
                "--gfa",
                str(gfa_bad),
                "--ref-fa",
                str(fa_path),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(tmp_path / f"out_collide_{idx}"),
            ],
            check=False,
        )
        assert proc.returncode != 0
        assert "Traceback" not in proc.stderr
        assert "Ambiguous duplicate source interval" in proc.stderr

    # Positive control: disjoint P [0, 200) and W [1000, 1460) on Alt#1#Chr01 passes CLI build + verify
    gfa_ok = tmp_path / "disjoint_ok.gfa"
    gfa_ok.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s1}",
                f"S\t2\t{s2}",
                f"S\t3\t{s3}",
                f"S\t4\t{s4}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t1\t+\t4\t+\t0M",
                "L\t4\t+\t2\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+\t*",
                "P\tAlt#1#Chr01\t1+,2+\t*",
                "W\tAlt\t1\tChr01\t1000\t*\t>1>4>2>3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_ok = tmp_path / "out_disjoint_ok"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_ok),
            "--ref-fa",
            str(fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out_ok),
        ],
        check=True,
    )
    run_verify(out_ok, gfa_ok, fa_path, check=True)


def test_tier5_m7_11_r2_d4_mixed_gt10k_and_le10k_steps_merge_and_verify_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D4` (CLI E2E): `10,050`-step `Chr01` (omitting `# step` lines via
    `LazyStepOffsetSequence`) merged with `20`-step `Chr02` (emitting `20` `# step` lines)
    passes `panflat build`, `panflat merge`, and standalone `panflat verify` (`rc == 0`),
    while tampering `# step`, duplicating `tName`, or adding orphan `# block` lines fails
    `panflat verify` cleanly.
    """
    import shutil

    # Chr01: 10,050 reference steps (steps 0..10047 repeat node 1, steps 10048..10049 unique 2+, 3+ with INS 4+)
    u20 = "ACGTACGTTGCATGCAACGT"  # 20 bp
    s_u2 = make_complex_seq(100, seed=10301)
    s_u3 = make_complex_seq(100, seed=10302)
    s_ins1 = make_complex_seq(90, seed=10303)

    c1_ref_steps = ["1+"] * 10_048 + ["2+", "3+"]
    assert len(c1_ref_steps) == 10_050
    c1_ref_seq = (u20 * 10_048) + s_u2 + s_u3
    gfa1 = tmp_path / "chr01_10k.gfa"
    gfa1.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{u20}",
                f"S\t2\t{s_u2}",
                f"S\t3\t{s_u3}",
                f"S\t4\t{s_ins1}",
                "L\t1\t+\t1\t+\t0M",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t2\t+\t4\t+\t0M",
                "L\t4\t+\t3\t+\t0M",
                f"P\tRef#1#Chr01\t{','.join(c1_ref_steps)}\t*",
                "P\tAlt#1#Chr01\t2+,4+,3+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa1 = tmp_path / "chr01_10k.fa"
    fa1.write_text(f">Chr01\n{c1_ref_seq}\n", encoding="utf-8")
    out1 = tmp_path / "b_chr01"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa1),
            "--ref-fa",
            str(fa1),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out1),
        ],
        check=True,
    )

    # Chr02: 20 reference steps (nodes 11..30) + INS node 31 between 20+ and 21+
    c2_nodes = {str(10 + i): make_complex_seq(60, seed=10320 + i) for i in range(1, 21)}
    s_ins2 = make_complex_seq(80, seed=10350)
    c2_ref_steps = [f"{10 + i}+" for i in range(1, 21)]
    assert len(c2_ref_steps) == 20
    c2_alt_steps = [f"{10 + i}+" for i in range(1, 11)] + ["31+"] + [f"{10 + i}+" for i in range(11, 21)]
    c2_lines = ["H\tVN:Z:1.0"]
    for nid, seq in c2_nodes.items():
        c2_lines.append(f"S\t{nid}\t{seq}")
    c2_lines.append(f"S\t31\t{s_ins2}")
    for i in range(1, 20):
        c2_lines.append(f"L\t{10 + i}\t+\t{11 + i}\t+\t0M")
    c2_lines.append("L\t20\t+\t31\t+\t0M")
    c2_lines.append("L\t31\t+\t21\t+\t0M")
    c2_lines.append(f"P\tRef#1#Chr02\t{','.join(c2_ref_steps)}\t*")
    c2_lines.append(f"P\tAlt#1#Chr02\t{','.join(c2_alt_steps)}\t*")

    gfa2 = tmp_path / "chr02_20.gfa"
    gfa2.write_text("\n".join(c2_lines) + "\n", encoding="utf-8")
    c2_ref_seq = "".join(c2_nodes[str(10 + i)] for i in range(1, 21))
    fa2 = tmp_path / "chr02_20.fa"
    fa2.write_text(f">Chr02\n{c2_ref_seq}\n", encoding="utf-8")
    out2 = tmp_path / "b_chr02"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa2),
            "--ref-fa",
            str(fa2),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out2),
        ],
        check=True,
    )

    merged = tmp_path / "merged_mixed_steps"
    _, mbun = run_merge([out1, out2], merged)
    assert mbun is not None
    step_comment_lines = [
        ln for ln in mbun.chain_text.splitlines() if ln.startswith("# step\t")
    ]
    assert len(step_comment_lines) == 20

    comb_gfa = tmp_path / "comb_mixed.gfa"
    comb_gfa.write_text(
        gfa1.read_text(encoding="utf-8")
        + "\n".join(c2_lines[1:])
        + "\n",
        encoding="utf-8",
    )
    comb_fa = tmp_path / "comb_mixed.fa"
    comb_fa.write_text(f">Chr01\n{c1_ref_seq}\n>Chr02\n{c2_ref_seq}\n", encoding="utf-8")

    # Standalone panflat verify passes (rc == 0)
    proc_ok = run_verify(merged, comb_gfa, comb_fa, check=True)
    assert proc_ok.returncode == 0

    # Negative CLI tamper checks on merged ref_to_pan.chain
    orig_chain = mbun.chain_text
    for idx_bad, mutated_chain in enumerate(
        [
            orig_chain.replace("# step\t0\t0\t60\t", "# step\t0\t0\t999\t", 1),
            orig_chain + f"\nchain {len(c2_ref_seq)} Chr02 {len(c2_ref_seq)} + 0 {len(c2_ref_seq)} Chr02 {len(c2_ref_seq)} + 0 {len(c2_ref_seq)} 3\n{len(c2_ref_seq)}\n",
            "# block\t0\t60\t0\n" + orig_chain,
        ]
    ):
        tdir = tmp_path / f"tamper_chain_{idx_bad}"
        tdir.mkdir()
        for f in merged.parent.glob(f"{merged.name}.*"):
            shutil.copy2(f, tdir / f.name)
        tpref = tdir / merged.name
        Path(f"{tpref}.ref_to_pan.chain").write_text(mutated_chain, encoding="utf-8")
        p_bad = run_verify(tpref, comb_gfa, comb_fa, check=False)
        assert p_bad.returncode != 0 and "Traceback" not in p_bad.stderr


def test_tier5_m7_12_r2_d5_qc_contiguous_ref_stage1_and_short_insert_chimeric_guard_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D5` (CLI E2E):
    1. Contiguous-reference Stage 1 sensitivity: a `220 bp` (`>= 150 bp`) homolog of
       `ref[0:220]` whose reference origin is split across two `110 bp` anchors (`< 150 bp`
       each) by an `80 bp` short slot insertion in `pan.fa` is accurately detected (`220 bp`
       at `>= 0.98` identity) by `panflat build --qc`.
    2. Chimeric Stage 1 & Stage 2 specificity: `0 bp` false positives across a `210 bp`
       `100 bp short_insert + 110 bp right_anchor` junction AND across adjacent Stage 2
       target pieces separated by `N * 200`.
    """
    r1 = make_complex_seq(110, seed=10401)
    r2 = make_complex_seq(110, seed=10402)
    r3 = make_complex_seq(200, seed=10403)
    r4 = make_complex_seq(200, seed=10404)
    r5 = make_complex_seq(200, seed=10405)

    # 1. Stage 1 contiguous-reference sensitivity across a split reference slot
    short80 = make_complex_seq(80, seed=10406)
    hom220 = mutate_seq_snps(r1 + r2, num_snps=2, seed=10407)
    gfa_sens = tmp_path / "qc_sens.gfa"
    gfa_sens.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{r1}",
                f"S\t2\t{r2}",
                f"S\t3\t{r3}",
                f"S\t4\t{short80}",
                f"S\t5\t{hom220}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t1\t+\t4\t+\t0M",
                "L\t4\t+\t2\t+\t0M",
                "L\t2\t+\t5\t+\t0M",
                "L\t5\t+\t3\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+\t*",
                "P\tAlt1#1#Chr01\t1+,4+,2+,3+\t*",
                "P\tAlt2#1#Chr01\t1+,2+,5+,3+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa_sens = tmp_path / "qc_sens.fa"
    fa_sens.write_text(f">Chr01\n{r1 + r2 + r3}\n", encoding="utf-8")
    out_sens = tmp_path / "out_qc_sens"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_sens),
            "--ref-fa",
            str(fa_sens),
            "--backbone-prefix",
            "Ref#1#",
            "--qc",
            "--out-prefix",
            str(out_sens),
        ],
        check=True,
    )
    bun_sens = OutputBundle(out_sens)
    assert_all_global_invariants(bun_sens, gfa_sens, fa_sens)
    assert int(bun_sens.summary_kv["qc_short_50_149_count"]) == 1
    assert int(bun_sens.summary_kv["qc_short_50_149_bp"]) == 80
    assert int(bun_sens.summary_kv["qc_long_ge150_count"]) == 1
    assert int(bun_sens.summary_kv["qc_long_ge150_bp"]) == 220
    assert int(bun_sens.summary_kv["qc_redundancy_0.98_bp"]) == 220

    # 2. Stage 1 short_insert+right_anchor chimera & Stage 2 cross-target sep bridging -> 0 bp false positives
    short100 = make_complex_seq(100, seed=10410)
    chim210 = short100 + r2  # 100 bp short insert + 110 bp right_anchor (< 150 bp each!)
    p_left = make_complex_seq(160, seed=10411)
    p_right = make_complex_seq(160, seed=10412)
    p_bridge = p_left[60:] + p_right[:100]  # 100 bp suffix of p_left + 100 bp prefix of p_right

    gfa_spec = tmp_path / "qc_spec.gfa"
    gfa_spec.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{r1}",
                f"S\t2\t{r2}",
                f"S\t3\t{r3}",
                f"S\t4\t{r4}",
                f"S\t5\t{r5}",
                f"S\t6\t{short100}",
                f"S\t7\t{chim210}",
                f"S\t8\t{p_left}",
                f"S\t9\t{p_bridge}",
                f"S\t10\t{p_right}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t3\t+\t4\t+\t0M",
                "L\t4\t+\t5\t+\t0M",
                "L\t1\t+\t6\t+\t0M",
                "L\t6\t+\t2\t+\t0M",
                "L\t2\t+\t7\t+\t0M",
                "L\t7\t+\t3\t+\t0M",
                "L\t3\t+\t8\t+\t0M",
                "L\t8\t+\t4\t+\t0M",
                "L\t3\t+\t9\t+\t0M",
                "L\t9\t+\t4\t+\t0M",
                "L\t4\t+\t10\t+\t0M",
                "L\t10\t+\t5\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+,4+,5+\t*",
                "P\tAlt1#1#Chr01\t1+,6+,2+,3+,8+,4+,5+\t*",
                "P\tAlt2#1#Chr01\t1+,2+,7+,3+,9+,4+,10+,5+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fa_spec = tmp_path / "qc_spec.fa"
    fa_spec.write_text(f">Chr01\n{r1 + r2 + r3 + r4 + r5}\n", encoding="utf-8")
    out_spec = tmp_path / "out_qc_spec"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa_spec),
            "--ref-fa",
            str(fa_spec),
            "--backbone-prefix",
            "Ref#1#",
            "--qc",
            "--out-prefix",
            str(out_spec),
        ],
        check=True,
    )
    bun_spec = OutputBundle(out_spec)
    assert_all_global_invariants(bun_spec, gfa_spec, fa_spec)
    assert int(bun_spec.summary_kv["qc_short_50_149_count"]) == 1
    assert int(bun_spec.summary_kv["qc_short_50_149_bp"]) == 100
    assert int(bun_spec.summary_kv["qc_long_ge150_count"]) == 4
    assert int(bun_spec.summary_kv["qc_long_ge150_bp"]) == 730
    assert int(bun_spec.summary_kv["qc_redundancy_0.90_bp"]) == 0
    assert int(bun_spec.summary_kv["qc_redundancy_0.95_bp"]) == 0
    assert int(bun_spec.summary_kv["qc_redundancy_0.98_bp"]) == 0


def test_tier5_m7_13_r2_d6_w_ref_coord_suffix_ambiguous_single_verify_and_merged_nonpansn_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D6` (CLI E2E):
    1. `W`-line reference (`W Ref 1 Chr01 100 900`) passes CLI `panflat build` + `panflat verify`
       with coordinate-suffixed FASTA (`>Ref#1#Chr01:100-900`) and PanSN FASTA (`>Ref#1#Chr01`)
       across `--backbone-prefix Ref#1#Chr01:100-900`, `Ref#1#`, `Ref#0#`, and `Ref`.
    2. Single-chromosome ambiguous `verify --backbone-prefix SampleA#1#` fails cleanly (`rc != 0`)
       when `SampleA#1#Chr10A` + `SampleA#1#Chr10A_unplaced` coexist, while `SampleA#1#Chr10A` passes.
    3. Merged non-PanSN `Ref_Chr01` + `Ref_Chr02` passes `panflat verify --backbone-prefix Ref_`
       (and fails `Wrong_`), plus non-PanSN exact path tie-breaking (`RefChr01` vs `RefChr01_alt`).
    """
    s1 = make_complex_seq(400, seed=10501)
    s2 = make_complex_seq(400, seed=10502)
    s_ins = make_complex_seq(100, seed=10503)
    s_c2 = make_complex_seq(300, seed=10504)
    ref_800 = s1 + s2

    # 1. W-line reference with :100-900 suffix in GFA & FASTA
    wref_gfa = tmp_path / "wref.gfa"
    wref_gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s1}",
                f"S\t2\t{s2}",
                f"S\t3\t{s_ins}",
                "L\t1\t+\t2\t+\t0M",
                "L\t1\t+\t3\t+\t0M",
                "L\t3\t+\t2\t+\t0M",
                "W\tRef\t1\tChr01\t100\t900\t>1>2",
                "W\tAlt\t1\tChr01\t50\t950\t>1>3>2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    w_cs_multi_fa = tmp_path / "w_cs_multi.fa"
    w_cs_multi_fa.write_text(
        f">Alt#1#Chr01:50-950\n{s1 + s_ins + s2}\n>Ref#1#Chr01:100-900\n{ref_800}\n",
        encoding="utf-8",
    )
    w_pansn_fa = tmp_path / "w_pansn.fa"
    w_pansn_fa.write_text(
        f">Alt#1#Chr01\n{s1 + s_ins + s2}\n>Ref#1#Chr01\n{ref_800}\n",
        encoding="utf-8",
    )

    out_w_cs = tmp_path / "out_w_cs"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(wref_gfa),
            "--ref-fa",
            str(w_cs_multi_fa),
            "--backbone-prefix",
            "Ref#1#Chr01:100-900",
            "--out-prefix",
            str(out_w_cs),
        ],
        check=True,
    )
    out_w_ps = tmp_path / "out_w_ps"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(wref_gfa),
            "--ref-fa",
            str(w_pansn_fa),
            "--backbone-prefix",
            "Ref#1#Chr01:100-900",
            "--out-prefix",
            str(out_w_ps),
        ],
        check=True,
    )
    for bp_ok in ("Ref#1#Chr01:100-900", "Ref#1#Chr01", "Ref#1#", "Ref#0#", "Ref"):
        run_verify(out_w_cs, wref_gfa, w_cs_multi_fa, extra_args=["--backbone-prefix", bp_ok], check=True)
        run_verify(out_w_ps, wref_gfa, w_pansn_fa, extra_args=["--backbone-prefix", bp_ok], check=True)
    p_w_bad = run_verify(
        out_w_cs, wref_gfa, w_cs_multi_fa, extra_args=["--backbone-prefix", "Wrong#1#"], check=False
    )
    assert p_w_bad.returncode != 0 and "Traceback" not in p_w_bad.stderr

    # 2. Single-chrom ambiguous `SampleA#1#` rejected on both `build` and `verify`
    amb_gfa = tmp_path / "amb_single.gfa"
    amb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\n"
        "P\tSampleA#1#Chr10A\t1+\t*\nP\tSampleA#1#Chr10A_unplaced\t2+\t*\n",
        encoding="utf-8",
    )
    amb_fa = tmp_path / "amb_single.fa"
    amb_fa.write_text(
        f">SampleA#1#Chr10A\n{s1}\n>SampleA#1#Chr10A_unplaced\n{s_c2}\n",
        encoding="utf-8",
    )
    p_amb_build = run_panflat_cli(
        [
            "build",
            "--gfa",
            str(amb_gfa),
            "--ref-fa",
            str(amb_fa),
            "--backbone-prefix",
            "SampleA#1#",
            "--out-prefix",
            str(tmp_path / "out_amb_fail"),
        ],
        check=False,
    )
    assert p_amb_build.returncode != 0 and "Traceback" not in p_amb_build.stderr

    out_amb_ok = tmp_path / "out_amb_ok"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(amb_gfa),
            "--ref-fa",
            str(amb_fa),
            "--backbone-prefix",
            "SampleA#1#Chr10A",
            "--out-prefix",
            str(out_amb_ok),
        ],
        check=True,
    )
    run_verify(out_amb_ok, amb_gfa, amb_fa, extra_args=["--backbone-prefix", "SampleA#1#Chr10A"], check=True)
    p_amb_ver = run_verify(
        out_amb_ok, amb_gfa, amb_fa, extra_args=["--backbone-prefix", "SampleA#1#"], check=False
    )
    assert p_amb_ver.returncode != 0 and "Traceback" not in p_amb_ver.stderr

    # 3. Merged non-PanSN `Ref_Chr01` + `Ref_Chr02` with `verify --backbone-prefix Ref_` + exact tie-break
    np1_gfa = tmp_path / "np1.gfa"
    np1_gfa.write_text(f"H\tVN:Z:1.0\nS\t1\t{s1}\nP\tRef_Chr01\t1+\t*\n", encoding="utf-8")
    np2_gfa = tmp_path / "np2.gfa"
    np2_gfa.write_text(f"H\tVN:Z:1.0\nS\t2\t{s_c2}\nP\tRef_Chr02\t2+\t*\n", encoding="utf-8")
    fa_c1 = tmp_path / "plain_c1.fa"
    fa_c1.write_text(f">Chr01\n{s1}\n", encoding="utf-8")
    fa_c2 = tmp_path / "plain_c2.fa"
    fa_c2.write_text(f">Chr02\n{s_c2}\n", encoding="utf-8")

    np_b1 = tmp_path / "np_b1"
    np_b2 = tmp_path / "np_b2"
    run_panflat_cli(
        ["build", "--gfa", str(np1_gfa), "--ref-fa", str(fa_c1), "--backbone-prefix", "Ref_", "--out-prefix", str(np_b1)],
        check=True,
    )
    run_panflat_cli(
        ["build", "--gfa", str(np2_gfa), "--ref-fa", str(fa_c2), "--backbone-prefix", "Ref_", "--out-prefix", str(np_b2)],
        check=True,
    )
    np_merged = tmp_path / "np_merged"
    run_merge([np_b1, np_b2], np_merged)

    np_comb_gfa = tmp_path / "np_comb.gfa"
    np_comb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\nP\tRef_Chr01\t1+\t*\nP\tRef_Chr02\t2+\t*\n",
        encoding="utf-8",
    )
    plain_multi_fa = tmp_path / "plain_multi.fa"
    plain_multi_fa.write_text(f">Chr01\n{s1}\n>Chr02\n{s_c2}\n", encoding="utf-8")

    run_verify(np_merged, np_comb_gfa, plain_multi_fa, extra_args=["--backbone-prefix", "Ref_"], check=True)
    p_np_bad = run_verify(
        np_merged, np_comb_gfa, plain_multi_fa, extra_args=["--backbone-prefix", "Wrong_"], check=False
    )
    assert p_np_bad.returncode != 0 and "Traceback" not in p_np_bad.stderr

    # Exact non-PanSN tie-break: RefChr01 vs RefChr01_alt
    tie_gfa = tmp_path / "tie.gfa"
    tie_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\nP\tRefChr01\t1+\t*\nP\tRefChr01_alt\t2+\t*\n",
        encoding="utf-8",
    )
    out_tie = tmp_path / "out_tie"
    run_panflat_cli(
        ["build", "--gfa", str(tie_gfa), "--ref-fa", str(fa_c1), "--backbone-prefix", "RefChr01", "--out-prefix", str(out_tie)],
        check=True,
    )
    run_verify(out_tie, tie_gfa, fa_c1, extra_args=["--backbone-prefix", "RefChr01"], check=True)
    assert run_verify(out_tie, tie_gfa, fa_c1, extra_args=["--backbone-prefix", "RefChr"], check=False).returncode != 0


def test_tier5_m7_14_r2_d7_to_d12_right_nested_merge_gff_remap_isoform_status_and_cen_tamper_cli(
    tmp_path: Path,
) -> None:
    """M7 Round 2 `R2-D7`, `R2-D8`, `R2-D9`, `R2-D10`, `R2-D11`, `R2-D12` (CLI E2E):
    - CLI `flat` (`merge([Chr01, Chr02, Chr03])`), `left-nested`, and `right-nested`
      (`merge([Chr01, merge([Chr02, Chr03])])`) produce byte-identical outputs when `Chr02`
      has 0 unresolved rows, `Chr01`+`Chr03` share `unplaced_scaf_A` and unprefixed
      `UNRES_000001`, and all 3 chromosomes have colliding `SV_000001`.
    - `{merged}.ref.lifted.gff3` remaps `sv_id=` per chromosome (`SV_000001`,
      `Chr02_SV_000001`, `Chr03_SV_000001`) and isolates isoform `sv_status` across shared
      multi-`Parent` exons (`tx1` -> `cds_split_by_insertion`, `tx2` -> `utr_insertion`).
    - Standalone `panflat verify --ref-gff` passes when `Parent=tx1` is reused across
      `Chr01`, `Chr02`, `Chr03`, fails on CDS tamper, fails on malformed skipped `--ref-fa`
      records, and fails when `{merged}.centromere.tsv` is deleted even without `--centromere-bed`.
    """
    import shutil

    # Chr01 (nodes 11..16): extract-islands centromere + shared exon Parent=tx1,tx2 + unplaced_scaf_A
    orf1_a = "ATG" + ("GCT" * 29)  # 90 bp
    orf1_b = "GCT" * 30  # 90 bp (reads through into orf2 in tx1)
    cen_intron = make_complex_seq(100, seed=10601)
    orf2 = "ATG" + ("CTG" * 58) + "TAA"  # 180 bp
    ins_c1 = "GGT" * 30  # 90 bp coding insertion inside exon 1
    un_c1 = make_complex_seq(110, seed=10602)

    gfa1_lines = [
        "H\tVN:Z:1.0",
        f"S\t11\t{orf1_a}",
        f"S\t12\t{orf1_b}",
        f"S\t13\t{cen_intron}",
        f"S\t14\t{orf2}",
        f"S\t15\t{ins_c1}",
        f"S\t16\t{un_c1}",
        "L\t16\t+\t11\t+\t0M",
        "L\t11\t+\t12\t+\t0M",
        "L\t11\t+\t15\t+\t0M",
        "L\t15\t+\t12\t+\t0M",
        "L\t12\t+\t13\t+\t0M",
        "L\t13\t+\t14\t+\t0M",
        "P\tRef#1#Chr01\t11+,12+,13+,14+\t*",
        "P\tAlt#1#Chr01\t11+,15+,12+,13+,14+\t*",
        "P\tunplaced_scaf_A\t16+,11+,12+,13+,14+\t*",
    ]
    gfa1 = tmp_path / "c1.gfa"
    gfa1.write_text("\n".join(gfa1_lines) + "\n", encoding="utf-8")
    ref_c1 = orf1_a + orf1_b + cen_intron + orf2
    fa1 = tmp_path / "c1.fa"
    fa1.write_text(f">Chr01\n{ref_c1}\n", encoding="utf-8")
    bed1 = tmp_path / "c1.bed"
    bed1.write_text("Chr01\t180\t280\tCEN1\n", encoding="utf-8")
    gff1 = tmp_path / "c1.gff3"
    gff1.write_text(
        "##gff-version 3\n"
        "Chr01\tsrc\tgene\t1\t460\t.\t+\t.\tID=g1\n"
        "Chr01\tsrc\tmRNA\t1\t460\t.\t+\t.\tID=tx1;Parent=g1\n"
        "Chr01\tsrc\tmRNA\t1\t460\t.\t+\t.\tID=tx2;Parent=g1\n"
        "Chr01\tsrc\texon\t1\t180\t.\t+\t.\tID=ex1_shared;Parent=tx1,tx2\n"
        "Chr01\tsrc\texon\t281\t460\t.\t+\t.\tID=ex2_shared;Parent=tx1,tx2\n"
        "Chr01\tsrc\tCDS\t1\t180\t.\t+\t0\tID=cds1_tx1;Parent=tx1\n"
        "Chr01\tsrc\tCDS\t281\t460\t.\t+\t0\tID=cds2_shared;Parent=tx1,tx2\n",
        encoding="utf-8",
    )
    out1 = tmp_path / "p_Chr01"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa1),
            "--ref-fa",
            str(fa1),
            "--backbone-prefix",
            "Ref#1#",
            "--centromere-bed",
            str(bed1),
            "--centromere-mode",
            "extract-islands",
            "--ref-gff",
            str(gff1),
            "--out-prefix",
            str(out1),
        ],
        check=True,
    )

    # Chr02 (nodes 21..25): 0 unresolved rows, centromere off, colliding ID=tx1
    ex_c2_1 = "ATG" + ("ACC" * 29)
    in_c2_a = make_complex_seq(60, seed=10611)
    in_c2_b = make_complex_seq(60, seed=10612)
    ex_c2_2 = ("ACC" * 29) + "TAA"
    ins_c2 = make_complex_seq(75, seed=10613)
    gfa2_lines = [
        "H\tVN:Z:1.0",
        f"S\t21\t{ex_c2_1}",
        f"S\t22\t{in_c2_a}",
        f"S\t23\t{in_c2_b}",
        f"S\t24\t{ex_c2_2}",
        f"S\t25\t{ins_c2}",
        "L\t21\t+\t22\t+\t0M",
        "L\t22\t+\t23\t+\t0M",
        "L\t22\t+\t25\t+\t0M",
        "L\t25\t+\t23\t+\t0M",
        "L\t23\t+\t24\t+\t0M",
        "P\tRef#1#Chr02\t21+,22+,23+,24+\t*",
        "P\tAlt#1#Chr02\t21+,22+,25+,23+,24+\t*",
    ]
    gfa2 = tmp_path / "c2.gfa"
    gfa2.write_text("\n".join(gfa2_lines) + "\n", encoding="utf-8")
    ref_c2 = ex_c2_1 + in_c2_a + in_c2_b + ex_c2_2
    fa2 = tmp_path / "c2.fa"
    fa2.write_text(f">Chr02\n{ref_c2}\n", encoding="utf-8")
    gff2 = tmp_path / "c2.gff3"
    gff2.write_text(
        "##gff-version 3\n"
        "Chr02\tsrc\tgene\t1\t300\t.\t+\t.\tID=g1\n"
        "Chr02\tsrc\tmRNA\t1\t300\t.\t+\t.\tID=tx1;Parent=g1\n"
        "Chr02\tsrc\texon\t1\t90\t.\t+\t.\tID=ex1;Parent=tx1\n"
        "Chr02\tsrc\texon\t211\t300\t.\t+\t.\tID=ex2;Parent=tx1\n"
        "Chr02\tsrc\tCDS\t1\t90\t.\t+\t0\tID=cds1;Parent=tx1\n"
        "Chr02\tsrc\tCDS\t211\t300\t.\t+\t0\tID=cds2;Parent=tx1\n",
        encoding="utf-8",
    )
    out2 = tmp_path / "p_Chr02"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa2),
            "--ref-fa",
            str(fa2),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(gff2),
            "--out-prefix",
            str(out2),
        ],
        check=True,
    )
    assert len(OutputBundle(out2).unresolved) == 0

    # Chr03 (nodes 31..36): colliding unplaced_scaf_A + colliding ID=tx1
    ex_c3_1 = "ATG" + ("GTC" * 29)
    in_c3_a = make_complex_seq(60, seed=10621)
    in_c3_b = make_complex_seq(60, seed=10622)
    ex_c3_2 = ("GTC" * 29) + "TAA"
    ins_c3 = make_complex_seq(84, seed=10623)
    un_c3 = make_complex_seq(130, seed=10624)
    gfa3_lines = [
        "H\tVN:Z:1.0",
        f"S\t31\t{ex_c3_1}",
        f"S\t32\t{in_c3_a}",
        f"S\t33\t{in_c3_b}",
        f"S\t34\t{ex_c3_2}",
        f"S\t35\t{ins_c3}",
        f"S\t36\t{un_c3}",
        "L\t36\t+\t31\t+\t0M",
        "L\t31\t+\t32\t+\t0M",
        "L\t32\t+\t33\t+\t0M",
        "L\t32\t+\t35\t+\t0M",
        "L\t35\t+\t33\t+\t0M",
        "L\t33\t+\t34\t+\t0M",
        "P\tRef#1#Chr03\t31+,32+,33+,34+\t*",
        "P\tAlt#1#Chr03\t31+,32+,35+,33+,34+\t*",
        "P\tunplaced_scaf_A\t36+,31+,32+,33+,34+\t*",
    ]
    gfa3 = tmp_path / "c3.gfa"
    gfa3.write_text("\n".join(gfa3_lines) + "\n", encoding="utf-8")
    ref_c3 = ex_c3_1 + in_c3_a + in_c3_b + ex_c3_2
    fa3 = tmp_path / "c3.fa"
    fa3.write_text(f">Chr03\n{ref_c3}\n", encoding="utf-8")
    gff3 = tmp_path / "c3.gff3"
    gff3.write_text(
        "##gff-version 3\n"
        "Chr03\tsrc\tgene\t1\t300\t.\t+\t.\tID=g1\n"
        "Chr03\tsrc\tmRNA\t1\t300\t.\t+\t.\tID=tx1;Parent=g1\n"
        "Chr03\tsrc\texon\t1\t90\t.\t+\t.\tID=ex1;Parent=tx1\n"
        "Chr03\tsrc\texon\t211\t300\t.\t+\t.\tID=ex2;Parent=tx1\n"
        "Chr03\tsrc\tCDS\t1\t90\t.\t+\t0\tID=cds1;Parent=tx1\n"
        "Chr03\tsrc\tCDS\t211\t300\t.\t+\t0\tID=cds2;Parent=tx1\n",
        encoding="utf-8",
    )
    out3 = tmp_path / "p_Chr03"
    run_panflat_cli(
        [
            "build",
            "--gfa",
            str(gfa3),
            "--ref-fa",
            str(fa3),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(gff3),
            "--out-prefix",
            str(out3),
        ],
        check=True,
    )

    # Rewrite SV IDs to colliding SV_000001 and unresolved IDs to unprefixed UNRES_000001
    for chrom, out_p in (("Chr01", out1), ("Chr02", out2), ("Chr03", out3)):
        for ext in ("sv.tsv", "genotypes.tsv", "inserts.tsv", "ref.lifted.gff3"):
            fp = Path(f"{out_p}.{ext}")
            fp.write_text(
                fp.read_text(encoding="utf-8").replace(f"{chrom}_INS_1", "SV_000001"),
                encoding="utf-8",
            )
        if chrom != "Chr02":
            for ext in ("unresolved.tsv", "unresolved.fa"):
                fp = Path(f"{out_p}.{ext}")
                fp.write_text(
                    fp.read_text(encoding="utf-8").replace(f"{chrom}_UNRES_1", "UNRES_000001"),
                    encoding="utf-8",
                )

    # CLI flat vs left-nested vs right-nested 3-chromosome panflat merge
    m_flat = tmp_path / "m_flat"
    run_merge([out1, out2, out3], m_flat)

    m12 = tmp_path / "m12"
    run_merge([out1, out2], m12)
    m_left = tmp_path / "m_left"
    run_merge([m12, out3], m_left)

    m23 = tmp_path / "m23"
    run_merge([out2, out3], m23)
    m_right = tmp_path / "m_right"
    _, bun_right = run_merge([out1, m23], m_right)
    assert bun_right is not None

    for ext in (
        ".pan.fa",
        ".offsets.tsv",
        ".ref_to_pan.chain",
        ".sv.tsv",
        ".genotypes.tsv",
        ".inserts.tsv",
        ".centromere.tsv",
        ".ref.lifted.gff3",
        ".unresolved.tsv",
        ".unresolved.fa",
    ):
        t_f = Path(f"{m_flat}{ext}").read_text(encoding="utf-8")
        t_l = Path(f"{m_left}{ext}").read_text(encoding="utf-8")
        t_r = Path(f"{m_right}{ext}").read_text(encoding="utf-8")
        assert t_f == t_l == t_r, f"Merge associativity mismatch on {ext}"

    # Verify R2-D7 unresolved IDs/src_path, R2-D8 lifted GFF3 sv_id= remapping, & R2-D12 isoform sv_status
    assert [(r["record_id"], r["src_path"]) for r in bun_right.unresolved] == [
        ("Chr01_UNRES_000001", "Chr01::unplaced_scaf_A"),
        ("Chr03_UNRES_000001", "Chr03::unplaced_scaf_A"),
    ]
    sv_ids = {r["event_id"] for r in bun_right.sv_rows}
    assert sv_ids == {"SV_000001", "Chr02_SV_000001", "Chr03_SV_000001"}

    assert bun_right.lifted_gff_rows is not None
    mrna_by_chrom_id: dict[tuple[str, str], dict[str, str]] = {}
    for row in bun_right.lifted_gff_rows:
        attrs = row["attributes"]
        if "sv_id" in attrs:
            for tok in attrs["sv_id"].split(","):
                assert tok in sv_ids
        if row["type"] == "mRNA":
            mrna_by_chrom_id[(row["seqid"], attrs["ID"])] = attrs

    assert mrna_by_chrom_id[("Chr01", "tx1")]["sv_status"] == "cds_split_by_insertion"
    assert mrna_by_chrom_id[("Chr01", "tx2")]["sv_status"] == "utr_insertion"
    assert mrna_by_chrom_id[("Chr01", "tx1")]["sv_id"] == "SV_000001"
    assert mrna_by_chrom_id[("Chr02", "tx1")]["sv_id"] == "Chr02_SV_000001"
    assert mrna_by_chrom_id[("Chr03", "tx1")]["sv_id"] == "Chr03_SV_000001"

    # Build combined GFA, FASTA, and GFF3 for standalone CLI panflat verify --ref-gff
    comb_gfa_lines = ["H\tVN:Z:1.0"]
    for chrom, glines in (("Chr01", gfa1_lines), ("Chr02", gfa2_lines), ("Chr03", gfa3_lines)):
        for gl in glines[1:]:
            if gl.startswith("P\tunplaced_scaf_A\t"):
                comb_gfa_lines.append(gl.replace("P\tunplaced_scaf_A\t", f"P\t{chrom}::unplaced_scaf_A\t", 1))
            else:
                comb_gfa_lines.append(gl)
    comb_gfa = tmp_path / "comb_all.gfa"
    comb_gfa.write_text("\n".join(comb_gfa_lines) + "\n", encoding="utf-8")
    comb_fa = tmp_path / "comb_all.fa"
    comb_fa.write_text(f">Chr01\n{ref_c1}\n>Chr02\n{ref_c2}\n>Chr03\n{ref_c3}\n", encoding="utf-8")
    comb_gff = tmp_path / "comb_all.gff3"
    comb_gff.write_text(
        gff1.read_text(encoding="utf-8")
        + "\n".join(gff2.read_text(encoding="utf-8").splitlines()[1:])
        + "\n"
        + "\n".join(gff3.read_text(encoding="utf-8").splitlines()[1:])
        + "\n",
        encoding="utf-8",
    )

    # Standalone panflat verify --ref-gff passes with colliding Parent=tx1 across all 3 chromosomes (R2-D9)
    run_verify(m_right, comb_gfa, comb_fa, extra_args=["--ref-gff", comb_gff], check=True)

    # Tamper 1: Corrupting Chr01 CDS coordinate fails verify --ref-gff despite colliding ID=tx1 on Chr02/Chr03
    t_gff_dir = tmp_path / "tamper_gff"
    t_gff_dir.mkdir()
    for f in m_right.parent.glob(f"{m_right.name}.*"):
        shutil.copy2(f, t_gff_dir / f.name)
    t_gff_pref = t_gff_dir / m_right.name
    Path(f"{t_gff_pref}.ref.lifted.gff3").write_text(
        Path(f"{t_gff_pref}.ref.lifted.gff3")
        .read_text(encoding="utf-8")
        .replace("Chr01\tsrc\tCDS\t1\t90", "Chr01\tsrc\tCDS\t4\t90", 1),
        encoding="utf-8",
    )
    p_bad_gff = run_verify(t_gff_pref, comb_gfa, comb_fa, extra_args=["--ref-gff", comb_gff], check=False)
    assert p_bad_gff.returncode != 0 and "Traceback" not in p_bad_gff.stderr

    # Tamper 2 (R2-D10): Malformed non-alphabetic skipped record in PanSN --ref-fa fails CLI verify
    bad_pansn_fa = tmp_path / "bad_skipped_pansn.fa"
    bad_pansn_fa.write_text(
        f">OtherSample#1#Chr99\nACGT12345\n"
        f">Ref#1#Chr01\n{ref_c1}\n>Ref#1#Chr02\n{ref_c2}\n>Ref#1#Chr03\n{ref_c3}\n",
        encoding="utf-8",
    )
    p_bad_fa = run_verify(m_right, comb_gfa, bad_pansn_fa, check=False)
    assert p_bad_fa.returncode != 0 and "Traceback" not in p_bad_fa.stderr

    # Tamper 3 (R2-D11): Deleting merged.centromere.tsv fails CLI verify even without --centromere-bed
    t_cen_dir = tmp_path / "tamper_cen"
    t_cen_dir.mkdir()
    for f in m_right.parent.glob(f"{m_right.name}.*"):
        shutil.copy2(f, t_cen_dir / f.name)
    t_cen_pref = t_cen_dir / m_right.name
    Path(f"{t_cen_pref}.centromere.tsv").unlink()
    p_no_cen = run_verify(t_cen_pref, comb_gfa, comb_fa, check=False)
    assert p_no_cen.returncode != 0 and "Traceback" not in p_no_cen.stderr


def test_tier5_m7_15_p1_1_and_p1_2_gff_gene_isoform_union_and_per_allele_ins_tags_cli(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI E2E regression for P1 Task 1.1 & Task 1.2:
    1. Multi-isoform `gene` container unions `sv_status` across child `mRNA`s (`utr_insertion,cds_split_by_insertion`
       on shared-exon gene; `intronic_insertion,cds_split_by_insertion` on cassette-exon-skipping gene)
       while keeping sibling `mRNA` `sv_status` isolated.
    2. Multi-allelic `INS` `alleles` field isolates `partial_content=1`, `repeat_residual=1`, and
       `multi_hit=1` strictly to affected alleles without polluting co-located clean alleles.
    """
    n1 = "ATG" + ("GCT" * 29)  # 90 bp
    n2 = ("GCT" * 29) + "TAA"  # 90 bp
    intr1 = make_complex_seq(60, seed=11101)
    n3 = "ATG" + ("CTG" * 28) + "TAA"  # 90 bp

    n4 = "ATG" + ("CAG" * 29)  # 90 bp
    intr2a = make_complex_seq(30, seed=11102)
    cas_a = "GAC" * 15  # 45 bp
    cas_b = "GAC" * 15  # 45 bp
    intr2b = make_complex_seq(30, seed=11103)
    n5 = ("CAG" * 29) + "TGA"  # 90 bp

    clean_ins = make_complex_seq(600, seed=11111)
    part_ins = make_complex_seq(640, seed=11112) + make_satellite_seq(
        unit_len=12, total_len=1550, seed=11113
    )
    mh_ins = make_complex_seq(520, seed=11114)
    full_sat = make_satellite_seq(unit_len=11, total_len=1600, seed=11115)

    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seq in [
        ("n1", n1),
        ("n2", n2),
        ("intr1", intr1),
        ("n3", n3),
        ("n4", n4),
        ("intr2a", intr2a),
        ("cas_a", cas_a),
        ("cas_b", cas_b),
        ("intr2b", intr2b),
        ("n5", n5),
        ("clean_ins", clean_ins),
        ("part_ins", part_ins),
        ("mh_ins", mh_ins),
        ("full_sat", full_sat),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path(
        "Ref#1#Chr01",
        ["n1+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )
    clean_ins2 = make_complex_seq(580, seed=11116)
    b.add_node("clean_ins2", clean_ins2)
    b.add_path(
        "Clean#1#Chr01",
        ["n1+", "clean_ins+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "clean_ins2+", "cas_b+", "intr2b+", "n5+"],
    )
    b.add_path(
        "Partial#1#Chr01",
        ["n1+", "part_ins+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )
    # Backbone anchor n2+ visited twice -> §3.4.1 multi_hit=1 on MultiHit#1 at (n1, n2)
    b.add_path(
        "MultiHit#1#Chr01",
        ["n1+", "mh_ins+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n2+", "n5+"],
    )
    b.add_path(
        "FullSat#1#Chr01",
        ["n1+", "full_sat+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )

    sec_m = make_complex_seq(100, seed=11117)
    sec_b_extra = make_complex_seq(100, seed=11120)
    sec_u = make_complex_seq(100, seed=11118)
    sec_novel = make_complex_seq(100, seed=11119)
    b.add_node("sec_m", sec_m)
    b.add_node("sec_b_extra", sec_b_extra)
    b.add_node("sec_u", sec_u)
    b.add_node("sec_novel", sec_novel)
    b.add_path(
        "SecBack1#1#Chr01",
        ["n1+", "sec_m+", "sec_b_extra+", "sec_m+", "sec_u+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )
    b.add_path(
        "SecBack2#1#Chr01",
        ["n1+", "sec_m+", "sec_b_extra+", "sec_m+", "sec_u+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )
    b.add_path(
        "SecRep#1#Chr01",
        ["n1+", "sec_m+", "sec_novel+", "sec_u+", "n2+", "intr1+", "n3+", "n4+", "intr2a+", "cas_a+", "cas_b+", "intr2b+", "n5+"],
    )

    b.gff_lines.extend(
        [
            "Chr01\tsrc\tgene\t1\t330\t.\t+\t.\tID=g_shared",
            "Chr01\tsrc\tmRNA\t1\t330\t.\t+\t.\tID=tx_s1;Parent=g_shared",
            "Chr01\tsrc\tmRNA\t1\t330\t.\t+\t.\tID=tx_s2;Parent=g_shared",
            "Chr01\tsrc\texon\t1\t180\t.\t+\t.\tID=ex_s1;Parent=tx_s1,tx_s2",
            "Chr01\tsrc\texon\t241\t330\t.\t+\t.\tID=ex_s2;Parent=tx_s1,tx_s2",
            "Chr01\tsrc\tCDS\t1\t180\t.\t+\t0\tID=cds_s1;Parent=tx_s1",
            "Chr01\tsrc\tthree_prime_UTR\t241\t330\t.\t+\t.\tID=utr_s1;Parent=tx_s1",
            "Chr01\tsrc\tfive_prime_UTR\t1\t180\t.\t+\t.\tID=utr_s2;Parent=tx_s2",
            "Chr01\tsrc\tCDS\t241\t330\t.\t+\t0\tID=cds_s2;Parent=tx_s2",
            "Chr01\tsrc\tgene\t331\t660\t.\t+\t.\tID=g_cassette",
            "Chr01\tsrc\tmRNA\t331\t660\t.\t+\t.\tID=tx_c1;Parent=g_cassette",
            "Chr01\tsrc\tmRNA\t331\t660\t.\t+\t.\tID=tx_c2;Parent=g_cassette",
            "Chr01\tsrc\texon\t331\t420\t.\t+\t.\tID=ex_c1;Parent=tx_c1,tx_c2",
            "Chr01\tsrc\texon\t451\t540\t.\t+\t.\tID=ex_cas;Parent=tx_c1",
            "Chr01\tsrc\texon\t571\t660\t.\t+\t.\tID=ex_c3;Parent=tx_c1,tx_c2",
            "Chr01\tsrc\tCDS\t331\t420\t.\t+\t0\tID=cds_c1a;Parent=tx_c1",
            "Chr01\tsrc\tCDS\t451\t540\t.\t+\t0\tID=cds_cas;Parent=tx_c1",
            "Chr01\tsrc\tCDS\t571\t660\t.\t+\t0\tID=cds_c1b;Parent=tx_c1",
            "Chr01\tsrc\tCDS\t331\t420\t.\t+\t0\tID=cds_c2a;Parent=tx_c2",
            "Chr01\tsrc\tCDS\t571\t660\t.\t+\t0\tID=cds_c2b;Parent=tx_c2",
        ]
    )

    inputs = b.write("cli_p1_1_2")
    _, bundle = run_build(inputs, extra_args=["--ref-gff", str(inputs.gff_path)])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )
    run_verify(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--ref-gff", str(inputs.gff_path)],
        check=True,
    )

    assert bundle.lifted_gff_rows is not None
    gff_by_id = {
        r["attributes"]["ID"]: r["attributes"]
        for r in bundle.lifted_gff_rows
        if r["type"] in ("gene", "mRNA")
    }
    assert gff_by_id["tx_s1"]["sv_status"] == "cds_split_by_insertion"
    assert gff_by_id["tx_s2"]["sv_status"] == "utr_insertion"
    assert gff_by_id["g_shared"]["sv_status"] == "utr_insertion,cds_split_by_insertion"

    assert gff_by_id["tx_c1"]["sv_status"] == "cds_split_by_insertion"
    assert gff_by_id["tx_c2"]["sv_status"] == "intronic_insertion"
    assert gff_by_id["g_cassette"]["sv_status"] == "intronic_insertion,cds_split_by_insertion"

    ins_by_ref = {int(r["ref_start"]): r for r in bundle.sv_rows if r["type"] == "INS"}
    ev90 = ins_by_ref[90]
    gt90 = bundle.genotypes[ev90["event_id"]]
    assert gt90["FullSat#1"] == "."
    assert gt90["SecBack1#1"] == gt90["SecBack2#1"]
    amap90 = dict(e.split(":", 1) for e in str(ev90["alleles"]).split(";"))
    for clean_gt in (gt90["Clean#1"], gt90["SecBack1#1"]):
        assert "partial_content=1" not in amap90[clean_gt]
        assert "repeat_residual=1" not in amap90[clean_gt]
        assert "multi_hit=1" not in amap90[clean_gt]
    assert "partial_content=1" in amap90[gt90["Partial#1"]]
    assert "repeat_residual=1" not in amap90[gt90["Partial#1"]]
    assert "multi_hit=1" not in amap90[gt90["Partial#1"]]
    assert "multi_hit=1" in amap90[gt90["MultiHit#1"]]
    assert "partial_content=1" not in amap90[gt90["MultiHit#1"]]
    assert "repeat_residual=1" not in amap90[gt90["MultiHit#1"]]
    assert "repeat_residual=1" in amap90[gt90["SecRep#1"]]
    assert "multi_hit=1" not in amap90[gt90["SecRep#1"]]
    assert "partial_content=1" not in amap90[gt90["SecRep#1"]]


def test_tier5_m7_16_p1_3_and_p1_4_cen_mount_stacked_spacer_and_unresolved_header_tamper_cli(
    tmp_path: Path,
) -> None:
    """Opaque-box CLI E2E regression for P1 Task 1.3 & Task 1.4:
    1. `unresolved.fa` FASTA headers include standalone `reason=<...> src=<path>:<start>-<end> strand=<±>`,
       and `panflat verify` exits non-zero (`1`) when `strand=` or `src=` is tampered.
    2. C14 co-mounted `CEN_ISLAND` + normal `INS` tampered with a stacked `2 * spacer_n` gap
       (with all downstream `pan.fa`, `offsets.tsv`, `ref_to_pan.chain`, `sv.tsv`, `inserts.tsv`
       shifted consistently) is rejected by `panflat verify` (`[CEN]`).
    """
    import shutil

    spacer_n = 100
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r0", make_complex_seq(250, seed=11201))
    b.add_node("u", make_complex_seq(250, seed=11202))
    b.add_node("cen_ref", make_complex_seq(400, seed=11203))
    b.add_node("v", make_complex_seq(250, seed=11204))
    b.add_node("r_end", make_complex_seq(250, seed=11205))
    b.add_node("up_ins", make_complex_seq(180, seed=11206))
    b.add_node("cen_isl", make_complex_seq(620, seed=11207))
    b.add_node("co_ins", make_complex_seq(210, seed=11208))
    b.add_node("un_tail", make_complex_seq(150, seed=11209))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen_ref+", "v+", "r_end+"])
    b.add_path("HapCen#1#Chr01", ["r0+", "up_ins+", "u+", "cen_isl+", "v+", "r_end+"])
    # Reversed partial contig (omitting u-) so UNANCHORED_END has src_strand='-' AND co-mounted normal INS at ref=900
    b.add_path("HapNormRev#1#Chr01", ["r_end-", "v-", "co_ins-", "cen_ref-", "un_tail-"])
    b.add_bed_region(520, 880, "CEN_MAIN")

    inputs = b.write("cli_p1_3_4")
    _, bundle = run_build(
        inputs,
        extra_args=[
            "--centromere-mode",
            "extract-islands",
            "--centromere-min-island",
            "500",
            "--spacer-n",
            str(spacer_n),
        ],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle, spacer_n=spacer_n)
    run_verify(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--centromere-bed", str(inputs.bed_path), "--spacer-n", str(spacer_n)],
        check=True,
    )

    # Check standalone `strand=-` in unresolved.fa header
    assert len(bundle.unresolved_fa) == 1
    exp_id, (full_hdr, _) = next(iter(bundle.unresolved_fa.items()))
    hdr_kv = dict(t.split("=", 1) for t in full_hdr.split()[1:])
    assert hdr_kv["reason"] == "UNANCHORED_END"
    assert hdr_kv["strand"] == "-"
    assert "(" not in hdr_kv["src"]

    # Tamper 1: Flip strand=- -> strand=+ in unresolved.fa -> CLI verify must fail with exit code 1
    t_ufa_dir = tmp_path / "tamper_ufa_hdr"
    t_ufa_dir.mkdir()
    for fp in inputs.out_prefix.parent.glob(f"{inputs.out_prefix.name}.*"):
        shutil.copy2(fp, t_ufa_dir / fp.name)
    t_ufa_pref = t_ufa_dir / inputs.out_prefix.name
    ufa_p = Path(f"{t_ufa_pref}.unresolved.fa")
    ufa_p.write_text(
        ufa_p.read_text(encoding="utf-8").replace("strand=-", "strand=+", 1),
        encoding="utf-8",
    )
    res_ufa = run_verify(t_ufa_pref, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert (
        res_ufa.returncode == 1
        and "[UNRES]" in res_ufa.stderr
        and "Traceback" not in res_ufa.stderr
    )

    # Tamper 2: Consistently inject stacked 2*spacer_n between CEN_ISLAND and co-mounted normal INS
    t_cen_dir = tmp_path / "tamper_cen_stacked"
    t_cen_dir.mkdir()
    for fp in inputs.out_prefix.parent.glob(f"{inputs.out_prefix.name}.*"):
        shutil.copy2(fp, t_cen_dir / fp.name)
    t_cen_pref = t_cen_dir / inputs.out_prefix.name

    norm_ev = next(
        ev for ev in bundle.sv_rows if ev["type"] == "INS" and int(ev["ref_start"]) == 900
    )
    norm_piece = next(
        ins for ins in bundle.inserts if ins["event_id"] == norm_ev["event_id"]
    )
    insert_pos = int(norm_piece["pan_start"])

    # Shift pan.fa, offsets.tsv, ref_to_pan.chain, inserts.tsv, sv.tsv by +spacer_n
    pan_lines = Path(f"{t_cen_pref}.pan.fa").read_text(encoding="utf-8").splitlines()
    pan_seq_new = (
        "".join(pan_lines[1:])[:insert_pos]
        + ("N" * spacer_n)
        + "".join(pan_lines[1:])[insert_pos:]
    )
    Path(f"{t_cen_pref}.pan.fa").write_text(
        f"{pan_lines[0]}\n{pan_seq_new}\n", encoding="utf-8"
    )

    off_lines = Path(f"{t_cen_pref}.offsets.tsv").read_text(encoding="utf-8").splitlines()
    oh = {k: i for i, k in enumerate(off_lines[0].lstrip("#").split("\t"))}
    new_off = [off_lines[0]]
    step_tups: list[tuple[int, int, int, int, int, int]] = []
    for line in off_lines[1:]:
        cols = line.split("\t")
        ps = int(cols[oh["pan_start"]])
        pe = int(cols[oh["pan_end"]])
        sins = int(cols[oh["slot_ins_len"]])
        prev_pe_b = ps - sins
        if ps >= insert_pos:
            ps += spacer_n
            pe += spacer_n
            if prev_pe_b <= insert_pos:
                sins += spacer_n
        cols[oh["pan_start"]] = str(ps)
        cols[oh["pan_end"]] = str(pe)
        cols[oh["slot_ins_len"]] = str(sins)
        new_off.append("\t".join(cols))
        step_tups.append(
            (
                int(cols[oh["ref_step"]]),
                int(cols[oh["ref_start"]]),
                int(cols[oh["ref_end"]]),
                ps,
                pe,
                sins,
            )
        )
    Path(f"{t_cen_pref}.offsets.tsv").write_text("\n".join(new_off) + "\n", encoding="utf-8")

    ch_hdr = next(
        ln.split()
        for ln in Path(f"{t_cen_pref}.ref_to_pan.chain").read_text(encoding="utf-8").splitlines()
        if ln.startswith("chain ")
    )
    ch_hdr[8] = str(len(pan_seq_new))
    ch_hdr[11] = str(step_tups[-1][4])
    blks: list[tuple[int, int, int]] = []
    c_rs, c_re = step_tups[0][1], step_tups[0][2]
    for st_i in range(1, len(step_tups)):
        _, rs_i, re_i, _, _, sins_i = step_tups[st_i]
        if sins_i > 0:
            blks.append((c_rs, c_re, sins_i))
            c_rs, c_re = rs_i, re_i
        else:
            c_re = re_i
    blks.append((c_rs, c_re, 0))
    ch_out = [" ".join(ch_hdr)]
    for b_rs, b_re, b_dq in blks:
        ch_out.append(f"# block\t{b_rs}\t{b_re}\t{b_dq}")
        ch_out.append(f"{b_re - b_rs}\t0\t{b_dq}" if b_dq > 0 else f"{b_re - b_rs}")
    for st_t in step_tups:
        ch_out.append(
            f"# step\t{st_t[0]}\t{st_t[1]}\t{st_t[2]}\t{st_t[3]}\t{st_t[4]}\t{st_t[5]}"
        )
    Path(f"{t_cen_pref}.ref_to_pan.chain").write_text(
        "\n".join(ch_out) + "\n\n", encoding="utf-8"
    )

    ins_lines = Path(f"{t_cen_pref}.inserts.tsv").read_text(encoding="utf-8").splitlines()
    new_ins = [ins_lines[0]]
    ev_spans: dict[str, list[int]] = {}
    for line in ins_lines[1:]:
        cols = line.split("\t")
        ps, pe = int(cols[1]), int(cols[2])
        if ps >= insert_pos:
            ps += spacer_n
            pe += spacer_n
        cols[1], cols[2] = str(ps), str(pe)
        ev_id = cols[3]
        if ev_id not in ev_spans:
            ev_spans[ev_id] = [ps, pe]
        else:
            ev_spans[ev_id][0] = min(ev_spans[ev_id][0], ps)
            ev_spans[ev_id][1] = max(ev_spans[ev_id][1], pe)
        new_ins.append("\t".join(cols))
    Path(f"{t_cen_pref}.inserts.tsv").write_text("\n".join(new_ins) + "\n", encoding="utf-8")

    sv_lines = Path(f"{t_cen_pref}.sv.tsv").read_text(encoding="utf-8").splitlines()
    new_sv = [sv_lines[0]]
    for line in sv_lines[1:]:
        cols = line.split("\t")
        if cols[3] in ev_spans:
            cols[1], cols[2] = str(ev_spans[cols[3]][0]), str(ev_spans[cols[3]][1])
        new_sv.append("\t".join(cols))
    Path(f"{t_cen_pref}.sv.tsv").write_text("\n".join(new_sv) + "\n", encoding="utf-8")

    res_cen = run_verify(
        t_cen_pref,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--centromere-bed", str(inputs.bed_path), "--spacer-n", str(spacer_n)],
        check=False,
    )
    assert (
        res_cen.returncode == 1
        and "[CEN]" in res_cen.stderr
        and "Traceback" not in res_cen.stderr
    )


def test_tier5_m7_17_preflight_validation_and_stderr_progress_cli(
    tmp_path: Path,
) -> None:
    """Black-box CLI test verifying:
    1. Missing or mismatched `--ref-gff` fails during pre-flight without emitting any output files.
    2. Successful `panflat build` emits real-time timestamped stage progress on `stderr`
       while keeping `stdout` empty and passing all invariants.
    3. `panflat verify` rejects nonexistent explicit `--ref-gff` / `--centromere-bed`.
    """
    b = SyntheticInputBuilder(tmp_path / "pf_cli", chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("1", make_complex_seq(400, seed=9201))
    b.add_node("2", make_complex_seq(120, seed=9202))
    b.add_node("3", make_complex_seq(400, seed=9203))
    b.set_ref_path("Ref#1#Chr01", ["1+", "3+"])
    b.add_path("Alt1#1#Chr01", ["1+", "2+", "3+"])
    b.add_gff_line("Chr01", "src", "gene", 20, 380, ".", "+", ".", "ID=g1")
    inputs = b.write("pf_case")

    # 1. Mismatched --ref-gff fails before writing any output files
    bad_gff = tmp_path / "bad_chrom.gff3"
    bad_gff.write_text(
        "##gff-version 3\nChr99\tsrc\tgene\t10\t100\t.\t+\t.\tID=g1\n",
        encoding="utf-8",
    )
    bad_out = tmp_path / "should_not_exist"
    p_bad = run_panflat_cli(
        [
            "build",
            "--gfa",
            str(inputs.gfa_path),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--backbone-prefix",
            "Ref#1#",
            "--ref-gff",
            str(bad_gff),
            "--out-prefix",
            str(bad_out),
        ],
        check=False,
    )
    assert p_bad.returncode != 0
    assert "Traceback" not in p_bad.stderr
    assert "does not match reference chromosome" in p_bad.stderr
    assert list(tmp_path.glob("should_not_exist*")) == []

    # 2. Successful build emits timestamped stage progress on stderr, keeps stdout clean
    ok_proc, bundle = run_build(inputs, extra_args=["--ref-gff", str(inputs.gff_path)])
    assert ok_proc.stdout == ""
    for marker in ("[panflat]", "[1/4]", "[2/4]", "[3/4]", "[4/4]", "Build complete"):
        assert marker in ok_proc.stderr
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # 3. panflat verify rejects nonexistent explicit --ref-gff and --centromere-bed
    v_missing_gff = run_verify(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--ref-gff", str(tmp_path / "missing.gff3")],
        check=False,
    )
    assert (
        v_missing_gff.returncode == 1
        and "Reference GFF3 file not found" in v_missing_gff.stderr
        and "Traceback" not in v_missing_gff.stderr
    )

    v_missing_bed = run_verify(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        extra_args=["--centromere-bed", str(tmp_path / "missing.bed")],
        check=False,
    )
    assert (
        v_missing_bed.returncode == 1
        and "Centromere BED file not found" in v_missing_bed.stderr
        and "Traceback" not in v_missing_bed.stderr
    )

