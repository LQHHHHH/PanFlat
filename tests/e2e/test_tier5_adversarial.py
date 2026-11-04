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

def test_tier5_m1m2_15_whitebox_defensive_contract_guards_in_m1_m2() -> None:
    """Directly exercise defensive contract guards in `models.py`, `gfa_io.py`,
    `repeat_guard.py`, `normalizer.py`, and `centromere.py` that cannot be reached via
    valid CLI inputs (e.g., programmatic misuse of internal APIs).
    """
    import pytest
    from panflat.gfa_io import spell_steps
    from panflat.models import BuildConfig, CLIArgumentError, InputContractError, Step, StepProvenance
    from panflat.normalizer import (
        canonical_rotation,
        detect_canonical_cnv,
        fold_tandem_cycles,
        project_steps_onto_backbone,
    )
    from panflat.repeat_guard import (
        check_similarity_merge,
        complement_intervals,
        merge_contiguous_provenance,
        merge_intervals,
        parse_cs_to_query_ops,
        parse_edlib_cigar_to_ops,
        slice_candidate_by_intervals,
        subtract_intervals,
        subtract_reference_intervals_mappy,
    )

    # 1. models.py: Step.flipped without full_node_len on partial step, invalid subslice bounds
    p_step = Step("n1", "+", slice_start=5, slice_end=20, is_partial=True)
    with pytest.raises(ValueError):
        p_step.flipped()
    flipped_p = p_step.flipped(full_node_len=50)
    assert flipped_p.slice_start == 30 and flipped_p.slice_end == 45
    with pytest.raises(ValueError):
        p_step.subslice(-1, 5, full_node_len=50)

    prov = StepProvenance("H#1", "H#1#Chr01", 10, 30, "+")
    with pytest.raises(ValueError):
        prov.subslice(5, 3)

    cfg = BuildConfig(
        gfa="a.gfa",
        ref_fa="a.fa",
        backbone_prefix="Ref#",
        out_prefix="out",
        centromere_bed="cen.bed",
        centromere_mode="off",
        centromere_mode_explicit=False,
    )
    assert cfg.effective_centromere_mode == "bypass"

    # 2. gfa_io.py: _flip_orient, spell_steps and validate_build_config error guards
    from panflat.gfa_io import _flip_orient
    with pytest.raises(InputContractError):
        _flip_orient("?")
    with pytest.raises(InputContractError):
        spell_steps([Step("missing", "+")], {"n1": "ACGT"})
    with pytest.raises(InputContractError):
        spell_steps([Step("n1", "+", slice_start=3, slice_end=2, is_partial=True)], {"n1": "ACGT"})

    from panflat.gfa_io import load_and_validate_inputs, validate_build_config
    for bad_cfg in [
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="  ", out_prefix="o"),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", max_jump=0),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", max_copy=0),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", spacer_n=-1),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", threads=0),
    ]:
        with pytest.raises(CLIArgumentError):
            validate_build_config(bad_cfg)
    with pytest.raises(InputContractError):
        load_and_validate_inputs(BuildConfig(gfa="/nonexistent_panflat_test.gfa", ref_fa="a", backbone_prefix="R#", out_prefix="o"))

    # 3. repeat_guard.py: interval helpers, edlib split-gap > 0, cs long tokens, slice_candidate guards
    assert merge_intervals([(5, 2), (10, 10)]) == []
    assert complement_intervals(0, [(0, 5)]) == []
    assert complement_intervals(10, [(0, 10), (12, 15)]) == []
    assert subtract_intervals([], [(0, 5)]) == []
    assert merge_contiguous_provenance([]) == []
    assert merge_contiguous_provenance([prov], "ACGT") == [(prov, "")]

    with pytest.raises(InputContractError):
        slice_candidate_by_intervals([Step("n1", "+")], [], {"n1": "ACGT"}, [(0, 4)], [])
    with pytest.raises(InputContractError):
        slice_candidate_by_intervals([Step("unknown", "+")], [prov], {"n1": "ACGT"}, [(0, 4)], [])
    with pytest.raises(InputContractError):
        slice_candidate_by_intervals(
            [Step("n1", "+", slice_start=3, slice_end=2, is_partial=True)],
            [prov],
            {"n1": "ACGT"},
            [(0, 4)],
            [],
        )
    with pytest.raises(InputContractError):
        slice_candidate_by_intervals([Step("n1", "+")], [prov], {"n1": "ACGT"}, [(0, 4)], [])

    # Long cs token '=ACGT' and splice '~gt10ag' and reverse strand
    assert parse_cs_to_query_ops("", 1) == []
    cs_ops = parse_cs_to_query_ops("=ACGT*at+cc-gg~gt10ag", strand=-1)
    assert any(op == "M" for op, _ in cs_ops) and any(op == "D" for op, _ in cs_ops)
    assert parse_edlib_cigar_to_ops("") == []
    r_dmd = "AAXXXXAATTYYYYTT"
    s_dmd = "AATT"
    assert parse_edlib_cigar_to_ops("6D4=6D", seq_s=s_dmd, seq_r=r_dmd) == [("M", 2), ("D", 12), ("M", 2)]
    assert parse_edlib_cigar_to_ops("6I4=6I", seq_s=r_dmd, seq_r=s_dmd) == [("M", 2), ("I", 12), ("M", 2)]

    assert subtract_reference_intervals_mappy("", "ACGT") == ([], [], 0)
    assert subtract_reference_intervals_mappy("ACGT", "NNNN") == ([], [(0, 4)], 0)
    assert check_similarity_merge("", "ACGT") is False
    assert check_similarity_merge("ACGT", "TGCA", min_sv_len=0) is False

    # 4. normalizer.py: flexible overload signatures & guards
    from panflat.models import PathFragment, ProjectedAnchor
    from panflat.normalizer import (
        _extract_contiguous_nonref_slices,
        compute_weighted_lis,
        extract_deviations,
    )
    from panflat.centromere import _extract_nonref_runs

    st_a = Step("n1", "+")
    st_b = Step("n2", "+")
    nodes_ab = {"n1": "ACGT", "n2": "TGCA"}
    # Calling project_steps_onto_backbone without explicit provenance (synthesizes provenance)
    ancs = project_steps_onto_backbone([st_a, st_b], [st_a, st_b], nodes_ab, 0, 2)
    assert len(ancs) == 2
    with pytest.raises(InputContractError):
        project_steps_onto_backbone([st_a], [prov, prov], [st_a], nodes_ab)

    # Discontiguous provenance break in _extract_contiguous_nonref_slices and _extract_nonref_runs
    prov1 = StepProvenance("H#1", "H#1#Chr01", 0, 4, "+")
    prov2_gap = StepProvenance("H#1", "H#1#Chr01", 10, 14, "+")
    slices_gap = _extract_contiguous_nonref_slices(
        [st_a, st_b], [prov1, prov2_gap], nodes_ab, set(), min_sv_len=1
    )
    assert len(slices_gap) == 2
    runs_gap = _extract_nonref_runs([st_a, st_b], [prov1, prov2_gap], nodes_ab, set())
    assert len(runs_gap) == 2

    # compute_weighted_lis with only reverse-oriented anchors (n == 0 forward anchors)
    assert compute_weighted_lis([ProjectedAnchor(0, 0, "-", "n1", 4, st_a, prov1)]) == ([], False)

    # extract_deviations in nested mode (is_top_level=False) with empty main_chain
    frag_ab = PathFragment("H", "1", "H#1", "Chr01", "H#1#Chr01", 0, (st_a,), (prov1,), False, 4)
    nested_devs = extract_deviations(
        frag_ab, [], [st_b], is_top_level=False, lo=0, hi=1
    )
    assert len(nested_devs) == 1 and nested_devs[0].left_anchor_rank == "START"

    with pytest.raises(InputContractError):
        fold_tandem_cycles([st_a], [], set(), max_copy=1)
    with pytest.raises(InputContractError):
        fold_tandem_cycles([st_a], [prov], set(), max_copy=0)
    empty_fold = fold_tandem_cycles([], [], set(), max_copy=1)
    assert empty_fold.folded_steps == ()
    assert canonical_rotation([]) == ()
    # detect_canonical_cnv where both sides have k=1 (line 1179)
    assert detect_canonical_cnv([st_a], [st_a], nodes_ab, min_sv_len=1) is None


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

    # White-box check of can_transition and _solve_multi_member_component fallback when blocked by ancestor_keys
    from panflat.linearizer import (
        DedupCandidate,
        ExtractedCandidate,
        SubproblemStateKey,
        _solve_multi_member_component,
        can_transition,
    )
    from panflat.models import BuildConfig, GFAGraph, PathFragment, Piece, Step, StepProvenance

    k_parent = SubproblemStateKey(b_id=(("B", 0, 10, "+"),), lo=0, hi=10, has_logical_start=False, has_logical_end=False, m_key=((("M1", 0, 5, "+"),), (("M2", 0, 5, "+"),)))
    k_same = SubproblemStateKey(b_id=(("B", 0, 10, "+"),), lo=0, hi=10, has_logical_start=False, has_logical_end=False, m_key=((("M1", 0, 5, "+"),), (("M2", 0, 5, "+"),)))
    k_child_ok = SubproblemStateKey(b_id=(("B", 0, 10, "+"),), lo=1, hi=5, has_logical_start=True, has_logical_end=True, m_key=((("M1", 0, 5, "+"),),))
    assert not can_transition(k_parent, k_same, frozenset())
    assert can_transition(k_parent, k_child_ok, frozenset())
    assert not can_transition(k_parent, k_child_ok, frozenset({k_child_ok}))

    # Exercise _solve_multi_member_component fallback when parent_state_key has smaller metric than child
    st1 = Step("x1", "+")
    st2 = Step("x2", "+")
    p1 = StepProvenance("H1#1", "H1#1#Chr01", 0, 60, "+")
    p2 = StepProvenance("H2#1", "H2#1#Chr01", 0, 60, "+")
    s_x1 = make_complex_seq(60, seed=7110)
    s_x2 = make_complex_seq(60, seed=7111)
    ec1 = ExtractedCandidate("c1", None, "H1#1", "H1#1#Chr01", 0, 1, 0, (st1, st2), (p1, p1), s_x1 + s_x2)
    ec2 = ExtractedCandidate("c2", None, "H2#1", "H2#1#Chr01", 0, 1, 1, (st1,), (p2,), s_x1)
    dc1 = DedupCandidate(((st1.node_id, "+", 0, None, False), (st2.node_id, "+", 0, None, False)), (st1, st2), s_x1 + s_x2, (Piece("H1#1", "H1#1#Chr01", 0, 120, "+", s_x1 + s_x2, "source_contiguous", ("H1#1",)),), frozenset({"H1#1"}), (ec1,), ec1, "x1+,x2+", "H1#1")
    dc2 = DedupCandidate(((st1.node_id, "+", 0, None, False),), (st1,), s_x1, (Piece("H2#1", "H2#1#Chr01", 0, 60, "+", s_x1, "source_contiguous", ("H2#1",)),), frozenset({"H2#1"}), (ec2,), ec2, "x1+", "H2#1")
    dummy_frag = PathFragment("Ref", "1", "Ref#1", "Chr01", "Ref#1#Chr01", 0, (st1,), (p1,), False, 60)
    dummy_graph = GFAGraph("g.gfa", "0" * 64, "Chr01", s_x1, {"x1": s_x1, "x2": s_x2}, set(), dummy_frag, (), ("Ref#1", "H1#1", "H2#1"), ((0, 60),), frozenset(), {})
    dummy_cfg = BuildConfig(gfa="g.gfa", ref_fa="r.fa", backbone_prefix="Ref#1#", out_prefix="o")
    tiny_parent_key = SubproblemStateKey(b_id=(), lo=0, hi=0, has_logical_start=True, has_logical_end=True, m_key=())
    fb_res = _solve_multi_member_component([dc1, dc2], dummy_graph, dummy_cfg, tiny_parent_key, frozenset())
    assert len(fb_res) == 2 and all(not r.is_synthetic for r in fb_res)


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
    """White-box target: `panflat/gff_liftover.py` (all validation error branches lines 169-260,
    3-arg overload lines 147-156, empty features line 293, and empty `step_offsets` fallback lines 333-354).
    """
    import pytest
    from panflat.gff_liftover import liftover_gff3, liftover_gff3_lines
    from panflat.models import InputContractError, StepOffsetRow, SVEvent

    step_offsets = [StepOffsetRow("Chr01", 0, "n1", 0, 200, 0, 200, 0)]

    # 1. Missing GFF3 file
    with pytest.raises(InputContractError):
        liftover_gff3(tmp_path / "missing.gff3", "Chr01", step_offsets, [])

    # 2. Malformed GFF3 lines matrix
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
    for bad_line in bad_gff_lines:
        with pytest.raises(InputContractError):
            liftover_gff3_lines([bad_line], "Chr01", step_offsets, [])

    # 3. Empty feature GFF3 (comments only)
    assert liftover_gff3_lines(["##gff-version 3", "# comment"], "Chr01", step_offsets, []) == ["##gff-version 3"]

    # 4. 3-arg overload `liftover_gff3(gff_path, step_offsets, insertion_events)` and empty `step_offsets` fallback
    gff_file = tmp_path / "valid_overload.gff3"
    gff_file.write_text(
        "##gff-version 3\nChr01\tsrc\texon\t10\t100\t.\t+\t.\tID=ex1;projection_split=Old\n",
        encoding="utf-8",
    )
    ins_ev = SVEvent(
        chrom="Chr01",
        pan_start=50,
        pan_end=130,
        event_id="Chr01_INS_1",
        event_type="INS",
        ref_start=50,
        ref_end=50,
        anchors="0..1",
        alleles_str="1:len=80,src=H1#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id="Chr01_REG_1",
        genotypes={"Ref#1": "0", "H1#1": "1"},
        pieces=(),
    )
    res_overload = liftover_gff3(gff_file, step_offsets, [ins_ev])
    assert len(res_overload) == 3  # header + 2 split exon segments
    res_empty_offsets = liftover_gff3_lines(
        ["Chr01\tsrc\texon\t10\t100\t.\t+\t.\t"],
        "Chr01",
        [],
        [ins_ev],
    )
    assert len(res_empty_offsets) == 3


# ---------------------------------------------------------------------------
# 27 (M3/M4 #11). Multi-Chromosome Merge: Heterogeneous Centromere Modes & Haplotype Union
# ---------------------------------------------------------------------------

def test_tier5_m3m4_11_multi_chromosome_merge_heterogeneous_centromere_and_hap_union(tmp_path: Path) -> None:
    """White-box target: `panflat/writers.py` (`merge_chromosomes` lines 268-575).
    Merges 3 chromosomes with heterogeneous configurations and haplotype subsets:
    - `Chr01`: `extract-islands` + `--ref-gff`, haplotypes `Ref#1`, `H1#1`, `H2#1`
    - `Chr02`: `bypass` + `--dump-repeat-sequences`, haplotypes `Ref#1`, `H2#1`, `H3#1`
    - `Chr03`: `centromere-mode=off`, haplotypes `Ref#1`, `H1#1`, `H4#1` + `UNANCHORED_END`
    Also tests `merge_chromosomes` error branches (missing prefix list, empty prefix list,
    missing per-chrom file, duplicate chromosome name).
    """
    import pytest
    from tests.e2e.helpers import OutputBundle, run_merge
    from panflat.models import InputContractError
    from panflat.writers import merge_chromosomes

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
    chain_hdrs = [l for l in mbun.chain_path.read_text(encoding="utf-8").splitlines() if l.startswith("chain ")]
    assert [l.split()[12] for l in chain_hdrs] == ["1", "2", "3"]

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

    # Verify merge_chromosomes error branches
    with pytest.raises(InputContractError):
        merge_chromosomes(tmp_path / "no_such_list.txt", tmp_path / "bad_m1")
    with pytest.raises(InputContractError):
        merge_chromosomes([], tmp_path / "bad_m2")
    with pytest.raises(InputContractError):
        merge_chromosomes([tmp_path / "missing_prefix"], tmp_path / "bad_m3")
    with pytest.raises(InputContractError):
        merge_chromosomes([inp1.out_prefix, inp1.out_prefix], tmp_path / "bad_m4")


# ---------------------------------------------------------------------------
# 28 (M3/M4 #12). Standalone `panflat verify` Exhaustive Tamper Detection Matrix
# ---------------------------------------------------------------------------

def test_tier5_m3m4_12_verify_exhaustive_tamper_detection_matrix(tmp_path: Path) -> None:
    """White-box target: `panflat/verify_qc.py` (`verify_build_outputs` lines 480-1403).
    Verifies that standalone `panflat verify` detects tampering across:
    1. `pan.fa` reference base mutation -> `invariant_1_passed == False`
    2. `pan.fa` spacer base mutation (`N -> A`) -> `invariant_2_passed == False`
    3. `offsets.tsv` step gap -> `invariant_1_passed == False`
    4. `ref_to_pan.chain` `dt != 0`, `dq` mismatch, `# block` / `# step` footer mismatch -> `invariant_3_passed == False`
    5. `inserts.tsv` `src_strand` flip (`+ -> -`) -> `invariant_2_passed == False`
    6. `centromere.tsv` 5-way accounting mismatch & mount point delta mismatch -> `centromere_passed == False`
    7. `unresolved.fa` sequence corruption -> `unresolved_passed == False`
    8. `ref.lifted.gff3` CDS coordinate shift -> `invariant_3_passed == False`
    9. `summary.tsv` `gfa_sha256` tamper -> `invariant_2_passed == False`
    10. Missing required output file -> `passed == False`
    """
    import shutil
    from panflat.verify_qc import verify_build_outputs

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

    # Clean build passes verify
    clean_rep = verify_build_outputs(
        inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, inputs.gff_path, inputs.bed_path
    )
    assert clean_rep.passed

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
    r1 = verify_build_outputs(p1, inputs.gfa_path, inputs.ref_fa_path, inputs.gff_path, inputs.bed_path)
    assert not r1.passed and not r1.invariant_1_passed

    # 2. Tamper pan.fa spacer base ('N' -> 'A' right before CEN_ISLAND at pan pos 750)
    p2 = _clone_prefix("pan_spacer")
    fa_txt2 = Path(f"{p2}.pan.fa").read_text(encoding="utf-8").splitlines()
    seq2 = list(fa_txt2[1])
    n_idx = fa_txt2[1].index("N")
    seq2[n_idx] = "A"
    fa_txt2[1] = "".join(seq2)
    Path(f"{p2}.pan.fa").write_text("\n".join(fa_txt2) + "\n", encoding="utf-8")
    r2 = verify_build_outputs(p2, inputs.gfa_path, inputs.ref_fa_path, inputs.gff_path, inputs.bed_path)
    assert not r2.passed and not r2.invariant_2_passed

    # 3. Tamper offsets.tsv (change ref_start of first step from 0 to 5)
    p3 = _clone_prefix("offsets")
    off_lines = Path(f"{p3}.offsets.tsv").read_text(encoding="utf-8").splitlines()
    cols = off_lines[1].split("\t")
    cols[3] = "5"
    off_lines[1] = "\t".join(cols)
    Path(f"{p3}.offsets.tsv").write_text("\n".join(off_lines) + "\n", encoding="utf-8")
    r3 = verify_build_outputs(p3, inputs.gfa_path, inputs.ref_fa_path)
    assert not r3.passed and not r3.invariant_1_passed

    # 4. Tamper ref_to_pan.chain (# block footer comment mismatch)
    p4 = _clone_prefix("chain")
    ch_txt = Path(f"{p4}.ref_to_pan.chain").read_text(encoding="utf-8").replace("# block\t0\t", "# block\t10\t")
    Path(f"{p4}.ref_to_pan.chain").write_text(ch_txt, encoding="utf-8")
    r4 = verify_build_outputs(p4, inputs.gfa_path, inputs.ref_fa_path)
    assert not r4.passed and not r4.invariant_3_passed

    # 5. Tamper inserts.tsv (flip src_strand '+' -> '-')
    p5 = _clone_prefix("inserts")
    ins_lines = Path(f"{p5}.inserts.tsv").read_text(encoding="utf-8").splitlines()
    icols = ins_lines[1].split("\t")
    icols[9] = "-" if icols[9] == "+" else "+"
    ins_lines[1] = "\t".join(icols)
    Path(f"{p5}.inserts.tsv").write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    r5 = verify_build_outputs(p5, inputs.gfa_path, inputs.ref_fa_path)
    assert not r5.passed and not r5.invariant_2_passed

    # 6. Tamper centromere.tsv (corrupt accepted_source_bp so 5-way sum fails)
    p6 = _clone_prefix("cen")
    c_lines = Path(f"{p6}.centromere.tsv").read_text(encoding="utf-8").splitlines()
    for i in range(1, len(c_lines)):
        ccols = c_lines[i].split("\t")
        if ccols[20] != ".":
            ccols[25] = "99999"
            c_lines[i] = "\t".join(ccols)
    Path(f"{p6}.centromere.tsv").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    r6 = verify_build_outputs(p6, inputs.gfa_path, inputs.ref_fa_path)
    assert not r6.passed and not r6.centromere_passed

    # 7. Tamper unresolved.fa (corrupt sequence)
    p7 = _clone_prefix("unres_fa")
    Path(f"{p7}.unresolved.fa").write_text(">Chr01_UNRES_1\nTTTTTTTTTT\n", encoding="utf-8")
    r7 = verify_build_outputs(p7, inputs.gfa_path, inputs.ref_fa_path)
    assert not r7.passed and not r7.unresolved_passed

    # 8. Tamper ref.lifted.gff3 (shift CDS end by -3 bp so translated protein differs)
    p8 = _clone_prefix("gff")
    g_txt = Path(f"{p8}.ref.lifted.gff3").read_text(encoding="utf-8").replace("\t170\t", "\t167\t")
    Path(f"{p8}.ref.lifted.gff3").write_text(g_txt, encoding="utf-8")
    r8 = verify_build_outputs(p8, inputs.gfa_path, inputs.ref_fa_path, ref_gff_path=inputs.gff_path)
    assert not r8.passed and not r8.invariant_3_passed

    # 9. Tamper summary.tsv gfa_sha256
    p9 = _clone_prefix("sha")
    s_txt = Path(f"{p9}.summary.tsv").read_text(encoding="utf-8")
    Path(f"{p9}.summary.tsv").write_text(
        "\n".join(
            f"gfa_sha256\t{'f' * 64}" if l.startswith("gfa_sha256\t") else l
            for l in s_txt.splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    r9 = verify_build_outputs(p9, inputs.gfa_path, inputs.ref_fa_path)
    assert not r9.passed and not r9.invariant_2_passed

    # 10. Missing required output file
    p10 = _clone_prefix("missing")
    Path(f"{p10}.sv.tsv").unlink()
    r10 = verify_build_outputs(p10, inputs.gfa_path, inputs.ref_fa_path)
    assert not r10.passed


# ---------------------------------------------------------------------------
# 29 (M3/M4 #13). `--qc` Bucket Boundaries (50, 149, 150 bp) & Union Cap <= 1.0
# ---------------------------------------------------------------------------

def test_tier5_m3m4_13_qc_bucket_boundaries_50_149_150_and_union_cap(tmp_path: Path) -> None:
    """White-box target: `panflat/verify_qc.py` (`run_qc` and `_merge_intervals` lines 107-220).
    Constructs inserted pieces of exact lengths:
    - `50 bp` (lower boundary of short bucket `[50, 149]`)
    - `149 bp` (upper boundary of short bucket `[50, 149]`)
    - `150 bp` (lower boundary of long bucket `>= 150`, unique complex sequence)
    - `250 bp` (long piece whose sequence also appears 3 times in the reference backbone of `pan.fa`,
      testing multi-hit union interval merging `_merge_intervals` so `ratio <= 1.0`).
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
    # 250 bp inserted node with 2 SNPs relative to s_hom_250 (>99% identical to 3 reference copies in pan.fa)
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
    # ins_250_hom (250 bp) aligns to the 3 copies of s_hom_250 in pan.fa at >=98% identity,
    # and _merge_intervals caps its redundant bases at 250 bp (not 3 * 250 = 750 bp)!
    assert int(bundle.summary_kv["qc_redundancy_0.98_bp"]) == 250
    assert 0.0 < float(bundle.summary_kv["qc_redundancy_0.98_ratio"]) <= 1.0


# ---------------------------------------------------------------------------
# 30 (M3/M4 #14). CLI Flag Validation & Multithreaded Shuffled GFA Determinism
# ---------------------------------------------------------------------------

def test_tier5_m3m4_14_cli_entrypoints_flags_and_multithreaded_shuffled_determinism(tmp_path: Path) -> None:
    """White-box target: `panflat/cli.py` (`main` lines 195-298) and `panflat/gfa_io.py`
    (`validate_build_config` lines 65-145), plus deterministic multithreaded (`--threads 4`)
    execution on a shuffled GFA input vs `--threads 1` on the unshuffled GFA input.
    """
    import random
    from panflat.cli import main as cli_main

    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx, nid in enumerate(["r0", "u", "cen", "v", "r4", "r5"]):
        b.add_node(nid, make_complex_seq(250, seed=8400 + idx))
    b.add_node("isl_a", make_complex_seq(550, seed=8410))
    b.add_node("ins_1", make_complex_seq(140, seed=8411))
    b.add_node("ins_2", make_complex_seq(160, seed=8412))
    b.add_node("hyb_rg", make_complex_seq(250, seed=8413) + make_satellite_seq(20, 1500, seed=8414) + make_complex_seq(250, seed=8415))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "r5+"])
    b.add_path("H1#1#Chr01", ["r0+", "ins_1+", "u+", "cen+", "isl_a+", "v+", "r4-", "r5+"])
    b.add_path("H2#1#Chr01", ["r0+", "u+", "cen+", "v+", "r4+", "hyb_rg+", "r5+"])
    b.add_walk("H3", "1", "ctgW", 4000, None, ["r5-", "ins_2-", "r4-", "v-", "cen-", "u-", "r0-"])
    b.add_bed_region(500, 750, "CEN_DET")
    b.add_gff_gene("g1", "m1", "+", [(20, 199)], [(20, 199, 0)])

    inputs = b.write("det_orig")

    # 1. Invalid CLI flag combinations via direct `cli_main` (in-process coverage)
    base_cmd = [
        "build",
        "--gfa", str(inputs.gfa_path),
        "--ref-fa", str(inputs.ref_fa_path),
        "--backbone-prefix", "Ref#1#",
        "--out-prefix", str(tmp_path / "bad_cli"),
    ]
    bad_arg_sets = [
        [*base_cmd, "--cluster-cov", "0.0"],
        [*base_cmd, "--cluster-cov", "1.2"],
        [*base_cmd, "--cluster-id", "0.0"],
        [*base_cmd, "--cluster-win-id", "1.5"],
        [*base_cmd, "--sec-backbone", "invalid"],
        [*base_cmd, "--repeat-guard", "maybe"],
        [*base_cmd, "--centromere-mode", "invalid"],
        [*base_cmd, "--centromere-mode", "extract-islands"],  # missing --centromere-bed
        [*base_cmd, "--centromere-bed", str(inputs.bed_path), "--centromere-mode", "off"],  # explicit off with bed
        [*base_cmd, "--centromere-min-island", "600"],  # explicit min-island when centromere mode is off
        [*base_cmd, "--centromere-bed", str(inputs.bed_path), "--centromere-mode", "extract-islands", "--centromere-min-island", "20"],
        [*base_cmd, "--min-sv-len", "0"],
        [*base_cmd, "--splice-window", "-1"],
    ]
    for bad_argv in bad_arg_sets:
        assert cli_main(bad_argv) != 0

    # 2. Create a shuffled copy of the GFA (permuting S, L, P, W lines deterministically)
    orig_lines = inputs.gfa_path.read_text(encoding="utf-8").splitlines()
    hdr_lines = [l for l in orig_lines if l.startswith("H\t")]
    body_lines = [l for l in orig_lines if not l.startswith("H\t") and l.strip()]
    rng = random.Random(8499)
    rng.shuffle(body_lines)
    shuf_gfa = tmp_path / "det_shuf.gfa"
    shuf_gfa.write_text("\n".join(hdr_lines + body_lines) + "\n", encoding="utf-8")

    out_t1 = tmp_path / "out_t1"
    out_t4 = tmp_path / "out_t4"
    common_flags = [
        "--ref-fa", str(inputs.ref_fa_path),
        "--backbone-prefix", "Ref#1#",
        "--ref-gff", str(inputs.gff_path),
        "--centromere-bed", str(inputs.bed_path),
        "--centromere-mode", "extract-islands",
        "--centromere-min-island", "500",
        "--dump-repeat-sequences",
        "--qc",
    ]
    assert cli_main(["build", "--gfa", str(inputs.gfa_path), "--out-prefix", str(out_t1), "--threads", "1", *common_flags]) == 0
    assert cli_main(["build", "--gfa", str(shuf_gfa), "--out-prefix", str(out_t4), "--threads", "4", *common_flags]) == 0
    assert cli_main(["verify", "--prefix", str(out_t4), "--gfa", str(shuf_gfa), "--ref-fa", str(inputs.ref_fa_path), "--ref-gff", str(inputs.gff_path)]) == 0

    # All primary output files must be byte-for-byte identical between unshuffled threads=1 and shuffled threads=4!
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

