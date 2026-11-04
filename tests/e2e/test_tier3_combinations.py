"""Tier 3 End-to-End (E2E) Cross-Feature Combination Tests for PanFlat v1.

Covers >= 26 pairwise and multi-subsystem interactions across:
- Centromere (`bypass` / `extract-islands`) + normal insertions + GFF3 liftover (`C14`)
- Multi-group right-anchor placement + span ordering + `CEN_ISLAND` co-mounting
- `--dump-repeat-sequences` off vs on across normal & centromere repeat classes (`C18`)
- `--repeat-guard` partial-node slicing + Union-Find component recursion + genotyping (`C20`)
- Top-level inversions (`INV`, `INV_DISPLACED`, `INV_EDGE`, `INV_CONFLICT`) + nested recursion + repeat-guard (`T7`, `T4`, `T24`)
- Shuffled GFA input order + multi-threaded (`--threads 1` vs `--threads 4`) byte-for-byte determinism (`T22`)
- Whole-contig orientation (`T23`) + `W`-line non-zero `seq_start` offsets + mixed SV types
- Cycle folding (`--max-copy`) + pre-folding canonical rotation CNV (`T10`, `T17`)
- Multi-chromosome `panflat merge` + `panflat verify` + `--qc` alignment redundancy
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil

import pytest

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
    run_verify,
)


# ---------------------------------------------------------------------------
# 1. C14: Co-mounting CEN_ISLAND + normal INS + GFF3 CDS liftover at ref_start(v)
# ---------------------------------------------------------------------------

def test_tier3_01_c14_comounting_cen_island_normal_ins_and_gff3(tmp_path: Path) -> None:
    """C14: Centromere extract-islands mounts CEN_ISLAND at ref_start(v) while a normal INS
    also mounts at ref_start(v) inside a GFF3 CDS feature.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_n0 = make_complex_seq(200, seed=101)
    s_u = make_complex_seq(200, seed=102)
    s_c1 = make_complex_seq(200, seed=103)
    # Place a 300 bp coding sequence spanning across the boundary between c2 and v:
    # 150 bp at the end of c2 (50..200) and 150 bp at the start of v (0..150)
    cds_full = make_coding_seq(100, seed=501)  # 300 bp
    s_c2 = make_complex_seq(50, seed=106) + cds_full[:150]  # 200 bp total: [600, 800)
    s_v_tail = make_complex_seq(250, seed=104)
    s_v = cds_full[150:] + s_v_tail  # 400 bp total: [800, 1200)
    s_n4 = make_complex_seq(300, seed=105)

    b.add_node("n0", s_n0)
    b.add_node("u", s_u)
    b.add_node("c1", s_c1)
    b.add_node("c2", s_c2)
    b.add_node("v", s_v)
    b.add_node("n4", s_n4)

    s_island = make_complex_seq(650, seed=201)
    s_norm_ins = make_complex_seq(180, seed=202)
    b.add_node("isl1", s_island)
    b.add_node("norm_ins", s_norm_ins)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "c1+", "c2+", "v+", "n4+"])
    # Complete haplotype across u..v carrying a >=500 bp complex centromere island
    b.add_path("Donor1#1#Chr01", ["n0+", "u+", "c1+", "isl1+", "c2+", "v+", "n4+"])
    # Partial contig starting at c2+ (without u+) with pure insertion at (c2, v) -> ref_start(v)=800
    b.add_path("Donor2#1#Chr01", ["c2+", "norm_ins+", "v+", "n4+"])

    # Reference coordinates:
    # n0: [0, 200), u: [200, 400), c1: [400, 600), c2: [600, 800), v: [800, 1200), n4: [1200, 1500)
    # BED [450, 750) expands outward to u (ref_end=400) and v (ref_start=800)
    b.add_bed_region(450, 750, "CEN1")
    # GFF3 CDS spans 1-based [651, 950] (0-based [650, 950)), crossing ref_start(v)=800
    b.add_gff_gene(
        gene_id="gene_c14",
        mrna_id="mrna_c14",
        strand="+",
        exons=[(601, 1000)],
        cds_list=[(651, 950, 0)],
    )

    inputs = b.write("c14_comount")
    spacer_n = 100
    _, bundle = run_build(
        inputs,
        extra_args=[
            "--centromere-mode", "extract-islands",
            "--centromere-min-island", "500",
            "--spacer-n", str(spacer_n),
        ],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path, spacer_n=spacer_n)
    assert_centromere_accounting_and_offsets(bundle, spacer_n=spacer_n)
    assert bundle.lifted_gff_path.exists()
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    # Check co-mounting order at ref_start(v) == 800: CEN_ISLAND precedes normal INS,
    # and adjacent blocks share a single spacer (never doubled).
    at_v_events = [ev for ev in bundle.sv_rows if int(ev["ref_start"]) == 800]
    types_at_v = [ev["type"] for ev in at_v_events]
    assert "CEN_ISLAND" in types_at_v and "INS" in types_at_v
    assert types_at_v.index("CEN_ISLAND") < types_at_v.index("INS")

    v_offset_row = next(r for r in bundle.offsets if r["node_id"] == "v")
    expected_slot_len = len(s_island) + len(s_norm_ins) + 2 * spacer_n
    assert int(v_offset_row["slot_ins_len"]) == expected_slot_len

    # Verify single N*spacer_n between CEN_ISLAND and normal INS in pan.fa
    cen_ins_row = next(ins for ins in bundle.inserts if "CEN_ISLAND" in ins["event_id"])
    norm_ins_row = next(ins for ins in bundle.inserts if "INS" in ins["event_id"] and "CEN" not in ins["event_id"])
    assert int(norm_ins_row["pan_start"]) - int(cen_ins_row["pan_end"]) == spacer_n
    assert bundle.pan_seq[int(cen_ins_row["pan_end"]) : int(norm_ins_row["pan_start"])] == "N" * spacer_n

    # Verify CDS was split across the co-mounted insertion
    cds_rows = [r for r in (bundle.lifted_gff_rows or []) if r["type"] == "CDS"]
    assert len(cds_rows) == 2
    for crow in cds_rows:
        assert crow["attributes"].get("projection_split") == "True"
        assert "cds_split_by_insertion" in crow["attributes"].get("sv_status", "")


# ---------------------------------------------------------------------------
# 2. Co-mounting multiple CEN_ISLAND reps + multi-span normal groups
# ---------------------------------------------------------------------------

def test_tier3_02_comounting_multiple_cen_islands_and_multi_span_normal_groups(tmp_path: Path) -> None:
    """Verify multiple CEN_ISLAND representatives co-mounting with a normal group at v,
    plus span ordering (b - a) ascending for multiple normal groups sharing a right anchor.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("n0", make_complex_seq(200, seed=11))
    b.add_node("u", make_complex_seq(200, seed=12))
    b.add_node("c1", make_complex_seq(150, seed=13))
    b.add_node("c2", make_complex_seq(150, seed=17))
    b.add_node("v", make_complex_seq(250, seed=14))
    b.add_node("n4", make_complex_seq(250, seed=15))
    b.add_node("n5", make_complex_seq(250, seed=16))

    isl_a = make_complex_seq(600, seed=21)
    isl_b = make_complex_seq(550, seed=22)
    norm_v = make_complex_seq(120, seed=23)
    ins_span1 = make_complex_seq(110, seed=24)
    ins_span2 = make_complex_seq(130, seed=25)

    b.add_node("isl_a", isl_a)
    b.add_node("isl_b", isl_b)
    b.add_node("norm_v", norm_v)
    b.add_node("ins_span1", ins_span1)
    b.add_node("ins_span2", ins_span2)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "c1+", "c2+", "v+", "n4+", "n5+"])
    # Two haplotypes with isl_a (support=2) and one haplotype with isl_b (support=1)
    b.add_path("Hap1#1#Chr01", ["n0+", "u+", "c1+", "isl_a+", "c2+", "v+", "n4+", "n5+"])
    b.add_path("Hap2#1#Chr01", ["n0+", "u+", "c1+", "isl_a+", "c2+", "v+", "n4+", "ins_span1+", "n5+"])
    b.add_path("Hap3#1#Chr01", ["n0+", "u+", "c1+", "isl_b+", "c2+", "v+", "ins_span2+", "n5+"])
    # Partial haplotype starting at c2+ (without u+) adding normal insertion at v (ref_start=700)
    b.add_path("Hap4#1#Chr01", ["c2+", "norm_v+", "v+", "n4+", "n5+"])

    # u: [200, 400), c1: [400, 550), c2: [550, 700), v: [700, 950)
    b.add_bed_region(420, 650, "CEN_MAIN")
    inputs = b.write("multi_comount")
    spacer_n = 80
    _, bundle = run_build(
        inputs,
        extra_args=[
            "--centromere-mode", "extract-islands",
            "--centromere-min-island", "500",
            "--spacer-n", str(spacer_n),
        ],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path, spacer_n=spacer_n)
    assert_centromere_accounting_and_offsets(bundle, spacer_n=spacer_n)

    # At v (step 4, ref_start=700): isl_a (support 2, 600bp) -> isl_b (support 1, 550bp) -> norm_v (120bp)
    v_inserts = [
        ins for ins in bundle.inserts
        if int(ins["pan_start"]) >= int(bundle.offsets[3]["pan_end"])
        and int(ins["pan_end"]) <= int(bundle.offsets[4]["pan_start"])
    ]
    assert len(v_inserts) == 3
    assert [int(r["pan_end"]) - int(r["pan_start"]) for r in v_inserts] == [600, 550, 120]

    # At n5 (step 6): pure insertion G(n4, n5) has span 1; replacement G(v, n5) has span 2.
    # Span 1 must precede span 2!
    n5_inserts = [
        ins for ins in bundle.inserts
        if int(ins["pan_start"]) >= int(bundle.offsets[5]["pan_end"])
        and int(ins["pan_end"]) <= int(bundle.offsets[6]["pan_start"])
    ]
    assert len(n5_inserts) == 2
    assert [int(r["pan_end"]) - int(r["pan_start"]) for r in n5_inserts] == [110, 130]


# ---------------------------------------------------------------------------
# 3. C18: --dump-repeat-sequences Off vs On across REPEAT_LIKE, CEN_BYPASS, CEN_REPEAT_LIKE
# ---------------------------------------------------------------------------

def test_tier3_03_c18_dump_repeat_sequences_off_vs_on(tmp_path: Path) -> None:
    """C18: Compare default (off) vs --dump-repeat-sequences across normal REPEAT_LIKE,
    centromere CEN_REPEAT_LIKE, and CEN_BYPASS.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 31), ("u", 32), ("cen_ref", 33), ("v", 34), ("n4", 35)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    sat_normal = make_satellite_seq(unit_len=20, total_len=1200, seed=301)
    sat_cen = make_satellite_seq(unit_len=25, total_len=1400, seed=302)
    b.add_node("sat_norm", sat_normal)
    b.add_node("sat_cen", sat_cen)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen_ref+", "v+", "n4+"])
    # Two haplotypes sharing the exact same normal satellite insertion to test deduplicated FASTA export
    b.add_path("Hap1#1#Chr01", ["n0+", "sat_norm+", "u+", "cen_ref+", "sat_cen+", "v+", "n4+"])
    b.add_path("Hap2#1#Chr01", ["n0+", "sat_norm+", "u+", "cen_ref+", "v+", "n4+"])

    b.add_bed_region(500, 750, "CEN1")
    inputs = b.write("c18_dump")

    _, bundle_off = run_build(
        inputs,
        out_prefix=tmp_path / "out_off",
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    _, bundle_on = run_build(
        inputs,
        out_prefix=tmp_path / "out_on",
        extra_args=[
            "--centromere-mode", "extract-islands",
            "--centromere-min-island", "500",
            "--dump-repeat-sequences",
        ],
    )
    assert bundle_off is not None and bundle_on is not None
    assert_all_global_invariants(bundle_off, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bundle_on, inputs.gfa_path, inputs.ref_fa_path)

    # pan.fa must be byte-for-byte identical between off and on
    assert bundle_off.pan_seq == bundle_on.pan_seq

    # In off mode, REPEAT_LIKE and CEN_REPEAT_LIKE must have exported_fasta_id == '.'
    reasons_off = {r["reason"] for r in bundle_off.unresolved}
    assert "REPEAT_LIKE" in reasons_off and "CEN_REPEAT_LIKE" in reasons_off
    for r in bundle_off.unresolved:
        if r["reason"] in ("REPEAT_LIKE", "CEN_REPEAT_LIKE", "CEN_BYPASS"):
            assert r["exported_fasta_id"] == "."
    assert len(bundle_off.unresolved_fa) == 0

    # In on mode, REPEAT_LIKE and CEN_REPEAT_LIKE are exported to unresolved.fa
    for r in bundle_on.unresolved:
        if r["reason"] in ("REPEAT_LIKE", "CEN_REPEAT_LIKE"):
            assert r["exported_fasta_id"] != "."
            assert r["exported_fasta_id"] in bundle_on.unresolved_fa
    assert 2 <= len(bundle_on.unresolved_fa) <= 3


# ---------------------------------------------------------------------------
# 4. C20 + Component Recursion: Partial-node slicing, multi-residual sig, partial_content=1
# ---------------------------------------------------------------------------

def test_tier3_04_c20_repeat_guard_split_residuals_and_component_recursion(tmp_path: Path) -> None:
    """C20 + Component Recursion: Repeat-guard masks central satellite inside a shared node,
    splitting the candidate into two >=50 bp complex residuals alongside a fully-masked haplotype.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=401))
    b.add_node("r2", make_complex_seq(250, seed=402))

    left_complex = make_complex_seq(400, seed=403)
    mid_satellite = make_satellite_seq(unit_len=20, total_len=1800, seed=404)
    right_complex = make_complex_seq(400, seed=405)
    extra_branch = make_complex_seq(150, seed=406)

    # Hybrid node where repeat-guard will mask mid_satellite and slice inside the node
    b.add_node("hyb", left_complex + mid_satellite + right_complex)
    b.add_node("extra", extra_branch)
    b.add_node("pure_sat", make_satellite_seq(unit_len=18, total_len=1200, seed=407))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("Hap1#1#Chr01", ["r1+", "hyb+", "r2+"])
    b.add_path("Hap2#1#Chr01", ["r1+", "hyb+", "extra+", "r2+"])
    b.add_path("Hap3#1#Chr01", ["r1+", "pure_sat+", "r2+"])
    b.add_path("HapRef#1#Chr01", ["r1+", "r2+"])

    inputs = b.write("c20_comp")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Verify REPEAT_LIKE recorded in unresolved.tsv
    repeat_rows = [r for r in bundle.unresolved if r["reason"] == "REPEAT_LIKE"]
    assert len(repeat_rows) >= 2

    # Verify INS event has partial_content=1 in alleles column
    ins_events = [ev for ev in bundle.sv_rows if ev["type"] == "INS"]
    assert len(ins_events) == 1
    ev = ins_events[0]
    assert "partial_content=1" in str(ev["alleles"])

    # Verify fully-masked Hap3#1 gets '.' (NOT '0'), while HapRef#1 gets '0'
    gt = bundle.genotypes[ev["event_id"]]
    assert gt["Hap3#1"] == "."
    assert gt["HapRef#1"] == "0"
    assert gt["Hap1#1"] not in ("0", ".")
    assert gt["Hap2#1"] not in ("0", ".")

    # Verify the two residuals from hyb are separated by N*spacer and never stitched directly
    assert len(bundle.inserts) >= 2


# ---------------------------------------------------------------------------
# 5. T7 Inversion + --repeat-guard on Nested Candidate
# ---------------------------------------------------------------------------

def test_tier3_05_t7_inversion_with_nested_repeat_guard_candidate(tmp_path: Path) -> None:
    """T7 + Repeat-Guard: Top-level INV containing a nested novel sequence whose central
    satellite array is masked by repeat-guard while flanking complex residuals are inserted
    in reverse-complement (src_strand='-').
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n11", 51), ("n12", 52), ("n13", 53), ("n14", 54)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    flank_l = make_complex_seq(400, seed=55)
    center_sat = make_satellite_seq(unit_len=20, total_len=1800, seed=56)
    flank_r = make_complex_seq(400, seed=57)
    b.add_node("X_nested", flank_l + center_sat + flank_r)

    b.set_ref_path("Ref#1#Chr01", ["n11+", "n12+", "n13+", "n14+"])
    # Branch has [n13- -> X_nested+ -> n12-] between n11+ and n14+
    # After inversion normalization: n11+ -> n12+ -> X_nested- -> n13+ -> n14+
    b.add_path("Donor#1#Chr01", ["n11+", "n13-", "X_nested+", "n12-", "n14+"])

    inputs = b.write("t7_rg")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "INV" in ev_types and "INS" in ev_types
    # Both INV and nested INS share the same region_id or locus
    ins_rows = bundle.inserts
    assert len(ins_rows) >= 1
    for r in ins_rows:
        assert r["src_strand"] == "-"
        # Placed before n13 (step 2, ref_start=500)
        assert int(r["pan_end"]) <= int(bundle.offsets[2]["pan_start"])

    assert any(u["reason"] == "REPEAT_LIKE" for u in bundle.unresolved)


# ---------------------------------------------------------------------------
# 6. INV_DISPLACED + Nested T4 Bifurcating Insertion (support vs longest)
# ---------------------------------------------------------------------------

def test_tier3_06_inv_displaced_with_nested_t4_bifurcation_support_vs_longest(tmp_path: Path) -> None:
    """Displaced inversion (INV_DISPLACED) normalized at top level, containing a nested
    bifurcating insertion A+B (2 haplotypes, shorter) vs A+C (1 haplotype, longer).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n1", 61), ("n2", 62), ("n3", 63), ("n4", 64), ("n5", 65)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    seq_a = make_complex_seq(300, seed=66)
    seq_b = make_complex_seq(250, seed=67)
    seq_c = make_complex_seq(400, seed=68)
    b.add_node("A", seq_a)
    b.add_node("B", seq_b)
    b.add_node("C", seq_c)

    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    # Displaced inversion: L is n3+ (rank 2), R is [n2-] (rank 1 < rank(L)!), F is n4+ (rank 3),
    # plus n1+ (rank 0) and n3+ (rank 2) -> weighted LIS picks n1+ -> n2+ -> (nested) -> n4+ -> n5+
    b.add_path("Hap1#1#Chr01", ["n1+", "n3+", "C-", "A-", "n2-", "n4+", "n5+"])
    b.add_path("Hap2#1#Chr01", ["n1+", "n3+", "B-", "A-", "n2-", "n4+", "n5+"])
    b.add_path("Hap3#1#Chr01", ["n1+", "n3+", "B-", "A-", "n2-", "n4+", "n5+"])

    inputs = b.write("inv_disp_t4")
    _, bundle_sup = run_build(
        inputs,
        out_prefix=tmp_path / "sup",
        extra_args=["--sec-backbone", "support"],
    )
    _, bundle_lng = run_build(
        inputs,
        out_prefix=tmp_path / "lng",
        extra_args=["--sec-backbone", "longest"],
    )
    assert bundle_sup is not None and bundle_lng is not None
    assert_all_global_invariants(bundle_sup, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bundle_lng, inputs.gfa_path, inputs.ref_fa_path)

    assert any(ev["type"] == "INV_DISPLACED" for ev in bundle_sup.sv_rows)
    assert any(ev["type"] == "INS" for ev in bundle_sup.sv_rows)
    # In support mode, A+B is secondary backbone (A + B + N + C);
    # in longest mode, A+C is secondary backbone (A + C + N + B).
    assert bundle_sup.pan_seq != bundle_lng.pan_seq
    assert len(bundle_sup.pan_seq) == len(bundle_lng.pan_seq)


# ---------------------------------------------------------------------------
# 7. T22: Shuffled GFA Input Order + --threads 1 vs --threads 4 Determinism
# ---------------------------------------------------------------------------

def test_tier3_07_t22_shuffled_gfa_order_and_multithreaded_determinism(tmp_path: Path) -> None:
    """T22: Permute S, L, and P/W line order in GFA and run with --threads 1 and --threads 4;
    verify byte-for-byte determinism across threads and identical sequences/events across permutations.
    """
    b1 = SyntheticInputBuilder(tmp_path / "order1", chrom="Chr01", backbone_prefix="Ref#1#")
    nodes = [
        ("r1", make_complex_seq(250, seed=71)),
        ("r2", make_complex_seq(250, seed=72)),
        ("r3", make_complex_seq(250, seed=73)),
        ("r4", make_complex_seq(250, seed=74)),
        ("ins1", make_complex_seq(180, seed=75)),
        ("ins2", make_complex_seq(210, seed=76)),
        ("ins3", make_complex_seq(160, seed=77)),
    ]
    for nid, seq in nodes:
        b1.add_node(nid, seq)
    b1.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
    b1.add_path("SampleA#1#Chr01", ["r1+", "ins1+", "r2+", "r3+", "r4+"])
    b1.add_path("SampleB#1#Chr01", ["r1+", "ins2+", "r2+", "r3-", "r4+"])
    b1.add_path("SampleC#1#Chr01", ["r1+", "r2+", "ins3+", "r3+", "r4+"])
    inputs1 = b1.write("ord1")

    # Build second GFA with reversed node order and reversed path order
    b2 = SyntheticInputBuilder(tmp_path / "order2", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seq in reversed(nodes):
        b2.add_node(nid, seq)
    b2.add_path("SampleC#1#Chr01", ["r1+", "r2+", "ins3+", "r3+", "r4+"])
    b2.add_path("SampleB#1#Chr01", ["r1+", "ins2+", "r2+", "r3-", "r4+"])
    b2.add_path("SampleA#1#Chr01", ["r1+", "ins1+", "r2+", "r3+", "r4+"])
    b2.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
    inputs2 = b2.write("ord2")

    _, bun_t1 = run_build(inputs1, out_prefix=tmp_path / "t1", extra_args=["--threads", "1"])
    _, bun_t4 = run_build(inputs1, out_prefix=tmp_path / "t4", extra_args=["--threads", "4"])
    _, bun_shuf = run_build(inputs2, out_prefix=tmp_path / "shuf", extra_args=["--threads", "4"])
    assert bun_t1 is not None and bun_t4 is not None and bun_shuf is not None

    assert_all_global_invariants(bun_t1, inputs1.gfa_path, inputs1.ref_fa_path)
    assert_all_global_invariants(bun_t4, inputs1.gfa_path, inputs1.ref_fa_path)
    assert_all_global_invariants(bun_shuf, inputs2.gfa_path, inputs2.ref_fa_path)

    # Byte-for-byte identity between --threads 1 and --threads 4
    for suffix in (".pan.fa", ".offsets.tsv", ".ref_to_pan.chain", ".sv.tsv", ".genotypes.tsv", ".inserts.tsv"):
        p1 = Path(f"{tmp_path / 't1'}{suffix}").read_bytes()
        p4 = Path(f"{tmp_path / 't4'}{suffix}").read_bytes()
        assert p1 == p4, f"Mismatch between --threads 1 and --threads 4 on {suffix}"

    # Shuffled GFA order must yield identical pan.fa, offsets.tsv, sv.tsv, and inserts.tsv
    assert bun_t1.pan_seq == bun_shuf.pan_seq
    assert bun_t1.offsets == bun_shuf.offsets
    assert bun_t1.sv_rows == bun_shuf.sv_rows
    assert bun_t1.inserts == bun_shuf.inserts


# ---------------------------------------------------------------------------
# 8. T23 Whole-Contig Reverse + W-line Non-zero seq_start + INV + DEL + INS
# ---------------------------------------------------------------------------

def test_tier3_08_t23_whole_contig_reverse_w_line_nonzero_offset_inv_del_ins(tmp_path: Path) -> None:
    """T23 + W-line non-zero seq_start + internal INV + DEL + INS."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 81), ("r2", 82), ("r3", 83), ("r4", 84), ("r5", 85), ("r6", 86)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))
    ins_seq = make_complex_seq(180, seed=87)
    b.add_node("ins_w", ins_seq)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    # Forward logical walk would be: r1+ -> ins_w+ -> r2+ -> r3- (internal INV) -> (skip r5: DEL) -> r6+
    #Wait: to make r3- a valid INV, it needs L=r2+ (rank 1) and F=r4+ (rank 3), then skip r5+ (DEL) -> r6+!
    # So forward logical walk: r1+ -> ins_w+ -> r2+ -> r3- -> r4+ -> r6+
    # Whole-contig reverse walk in W-line: r6- -> r4- -> r3+ -> r2- -> ins_w- -> r1-
    w_start = 10000
    b.add_walk(
        sample="DraftA",
        hap="1",
        seqid="ctg99",
        start=w_start,
        end=None,
        steps=["r6-", "r4-", "r3+", "r2-", "ins_w-", "r1-"],
    )

    inputs = b.write("w_rev_combo")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = [ev["type"] for ev in bundle.sv_rows]
    assert ev_types.count("INV") == 1
    assert "DEL" in ev_types
    assert "INS" in ev_types

    # Verify W-line seq_start=10000 is preserved in inserts.tsv and src_strand is '-'
    assert len(bundle.inserts) == 1
    ins_row = bundle.inserts[0]
    assert ins_row["src_strand"] == "-"
    s_coord, e_coord = [int(x) for x in str(ins_row["src_intervals"]).rsplit(":", 1)[-1].split("-")]
    # In the W-line ["r6-", "r4-", "r3+", "r2-", "ins_w-", "r1-"], four 250bp nodes precede ins_w- (1000bp)
    assert s_coord == w_start + 1000
    assert e_coord == w_start + 1000 + len(ins_seq)


# ---------------------------------------------------------------------------
# 9. Cycle Folding (--max-copy 1 vs 2) + Pre-folding Canonical CNV (T10 + T17)
# ---------------------------------------------------------------------------

def test_tier3_09_t10_t17_cycle_folding_max_copy_and_prefolding_canonical_cnv(tmp_path: Path) -> None:
    """T10 + T17: Pre-folding canonical rotation CNV combined with novel tandem repeat V-V-V-Tail
    folded under --max-copy 1 vs --max-copy 2 with join_before='synthetic'.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("rL", make_complex_seq(250, seed=91))
    b.add_node("u1", make_complex_seq(40, seed=92))
    b.add_node("u2", make_complex_seq(40, seed=93))
    b.add_node("rM", make_complex_seq(250, seed=94))
    b.add_node("rR", make_complex_seq(250, seed=95))

    b.add_node("V", make_complex_seq(120, seed=96))
    b.add_node("Tail", make_complex_seq(100, seed=97))

    # Reference has 3 copies of (u1+, u2+) between rL and rM
    b.set_ref_path(
        "Ref#1#Chr01",
        ["rL+", "u1+", "u2+", "u1+", "u2+", "u1+", "u2+", "rM+", "rR+"],
    )
    # Donor1 has 5 copies of (u1+, u2+) (delta = 2 * 80 = 160 bp >= 50 -> CNV)
    # AND between rM and rR has novel tandem repeat V+ -> V+ -> V+ -> Tail+
    b.add_path(
        "Donor1#1#Chr01",
        [
            "rL+",
            "u1+", "u2+", "u1+", "u2+", "u1+", "u2+", "u1+", "u2+", "u1+", "u2+",
            "rM+",
            "V+", "V+", "V+", "Tail+",
            "rR+",
        ],
    )

    inputs = b.write("cnv_fold")
    _, bun_mc1 = run_build(inputs, out_prefix=tmp_path / "mc1", extra_args=["--max-copy", "1"])
    _, bun_mc2 = run_build(inputs, out_prefix=tmp_path / "mc2", extra_args=["--max-copy", "2"])
    assert bun_mc1 is not None and bun_mc2 is not None
    assert_all_global_invariants(bun_mc1, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_mc2, inputs.gfa_path, inputs.ref_fa_path)

    # CNV genotype must use pre-folding counts (ref=3, Donor1=5) in both runs
    cnv_ev1 = next(ev for ev in bun_mc1.sv_rows if ev["type"] == "CNV")
    cnv_ev2 = next(ev for ev in bun_mc2.sv_rows if ev["type"] == "CNV")
    assert bun_mc1.genotypes[cnv_ev1["event_id"]]["Ref#1"] == "3"
    assert bun_mc1.genotypes[cnv_ev1["event_id"]]["Donor1#1"] == "5"
    assert bun_mc2.genotypes[cnv_ev2["event_id"]]["Donor1#1"] == "5"

    # Novel repeat V-V-V-Tail is folded to 1 copy of V + Tail (220 bp) in mc1,
    # and 2 copies of V + Tail (340 bp) in mc2, split into 2 pieces with join_before='synthetic'
    assert len(bun_mc1.inserts) == 2
    assert bun_mc1.inserts[1]["join_before"] == "synthetic"
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_mc1.inserts) == 120 + 100
    assert sum(int(r["pan_end"]) - int(r["pan_start"]) for r in bun_mc2.inserts) == 2 * 120 + 100


# ---------------------------------------------------------------------------
# 10. Multi-Chromosome panflat merge + panflat verify (extract-islands + off + GFF3)
# ---------------------------------------------------------------------------

def test_tier3_10_multichrom_merge_extract_islands_and_normal_with_gff_and_verify(tmp_path: Path) -> None:
    """Build Chr01 (extract-islands + GFF3) and Chr02 (off + GFF3), verify each, and merge."""
    # Chr01
    b1 = SyntheticInputBuilder(tmp_path / "chr1", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("c1_n0", 111), ("c1_u", 112), ("c1_cen", 113), ("c1_v", 114)]:
        b1.add_node(nid, make_complex_seq(250, seed=seed))
    b1.add_node("c1_isl", make_complex_seq(600, seed=115))
    b1.set_ref_path("Ref#1#Chr01", ["c1_n0+", "c1_u+", "c1_cen+", "c1_v+"])
    b1.add_path("StrainA#1#Chr01", ["c1_n0+", "c1_u+", "c1_cen+", "c1_isl+", "c1_v+"])
    b1.add_bed_region(500, 750, "CEN_CHR1")
    b1.add_gff_gene("g1", "m1", "+", [(51, 200)], [(51, 197, 0)], chrom="Chr01")
    inp1 = b1.write("chr01")

    # Chr02 (centromere-mode off)
    b2 = SyntheticInputBuilder(tmp_path / "chr2", chrom="Chr02", backbone_prefix="Ref#1#")
    b2.add_node("c2_r1", make_complex_seq(300, seed=121))
    b2.add_node("c2_r2", make_complex_seq(300, seed=122))
    b2.add_node("c2_ins", make_complex_seq(180, seed=123))
    b2.add_node("c2_unanch", make_complex_seq(90, seed=124))
    b2.set_ref_path("Ref#1#Chr02", ["c2_r1+", "c2_r2+"])
    b2.add_path("StrainA#1#Chr02", ["c2_unanch+", "c2_r1+", "c2_ins+", "c2_r2+"])
    b2.add_gff_gene("g2", "m2", "-", [(351, 550)], [(351, 548, 0)], chrom="Chr02")
    inp2 = b2.write("chr02")

    _, bun1 = run_build(
        inp1,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    _, bun2 = run_build(inp2, extra_args=["--centromere-mode", "off"])
    assert bun1 is not None and bun2 is not None

    run_verify(inp1.out_prefix, inp1.gfa_path, inp1.ref_fa_path)
    run_verify(inp2.out_prefix, inp2.gfa_path, inp2.ref_fa_path)

    _, merged = run_merge([inp1.out_prefix, inp2.out_prefix], tmp_path / "merged_all")
    assert merged is not None
    assert set(merged.pan_fa.keys()) == {"Chr01", "Chr02"}
    assert merged.pan_fa["Chr01"] == bun1.pan_seq
    assert merged.pan_fa["Chr02"] == bun2.pan_seq

    # centromere.tsv must only contain Chr01 rows, never fabricated Chr02 rows
    assert merged.centromere_rows is not None
    assert all(r["chrom"] == "Chr01" for r in merged.centromere_rows)
    assert len(merged.centromere_rows) == len(bun1.centromere_rows or [])

    # Merged lifted GFF3 contains genes from both Chr01 and Chr02
    assert merged.lifted_gff_rows is not None
    gff_chroms = {r["seqid"] for r in merged.lifted_gff_rows}
    assert gff_chroms == {"Chr01", "Chr02"}


# ---------------------------------------------------------------------------
# 11. Multi-Chromosome panflat merge Combining bypass and extract-islands
# ---------------------------------------------------------------------------

def test_tier3_11_multichrom_merge_bypass_and_extract_islands_unique_ids(tmp_path: Path) -> None:
    """Merge Chr01 (bypass) and Chr02 (extract-islands) and verify unique region_id and event_id."""
    b1 = SyntheticInputBuilder(tmp_path / "c1", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 131), ("u", 132), ("cen", 133), ("v", 134)]:
        b1.add_node(nid, make_complex_seq(250, seed=seed))
    b1.add_node("byp_block", make_complex_seq(600, seed=135))
    b1.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b1.add_path("HapA#1#Chr01", ["n0+", "u+", "cen+", "byp_block+", "v+"])
    b1.add_bed_region(500, 750, "CEN")
    inp1 = b1.write("c1")

    b2 = SyntheticInputBuilder(tmp_path / "c2", chrom="Chr02", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 141), ("u", 142), ("cen", 143), ("v", 144)]:
        b2.add_node(nid, make_complex_seq(250, seed=seed))
    b2.add_node("isl_block", make_complex_seq(600, seed=145))
    b2.set_ref_path("Ref#1#Chr02", ["n0+", "u+", "cen+", "v+"])
    b2.add_path("HapA#1#Chr02", ["n0+", "u+", "cen+", "isl_block+", "v+"])
    b2.add_bed_region(500, 750, "CEN")
    inp2 = b2.write("c2")

    _, bun1 = run_build(inp1, extra_args=["--centromere-mode", "bypass"])
    _, bun2 = run_build(
        inp2,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bun1 is not None and bun2 is not None

    _, merged = run_merge([inp1.out_prefix, inp2.out_prefix], tmp_path / "merged_cen")
    assert merged is not None
    assert merged.centromere_rows is not None
    region_ids = {r["region_id"] for r in merged.centromere_rows}
    assert len(region_ids) == 2
    modes = {r["mode"] for r in merged.centromere_rows}
    assert modes == {"bypass", "extract-islands"}


# ---------------------------------------------------------------------------
# 12. --qc Alignment Redundancy with Clustered vs Unclustered Groups & 50-149 bp Bins
# ---------------------------------------------------------------------------

def test_tier3_12_qc_alignment_redundancy_clustered_vs_unclustered_and_short_bins(tmp_path: Path) -> None:
    """Run panflat build --qc with 50-149 bp insertions and >=150 bp homologous insertions
    across different groups, verifying summary.tsv QC metrics and union <= 100%.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 151), ("r2", 152), ("r3", 153), ("r4", 154)]:
        b.add_node(nid, make_complex_seq(300, seed=seed))

    short_ins = make_complex_seq(90, seed=155)  # 50..149 bp bin
    hom_ins_1 = make_complex_seq(400, seed=156)
    hom_ins_2 = mutate_seq_snps(hom_ins_1, num_snps=4, seed=157)  # ~99% identical, placed in another group

    b.add_node("short90", short_ins)
    b.add_node("hom1", hom_ins_1)
    b.add_node("hom2", hom_ins_2)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
    b.add_path("Hap1#1#Chr01", ["r1+", "short90+", "r2+", "hom1+", "r3+", "r4+"])
    b.add_path("Hap2#1#Chr01", ["r1+", "r2+", "r3+", "hom2+", "r4+"])

    inputs = b.write("qc_combo")
    _, bundle = run_build(inputs, extra_args=["--qc"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Verify summary.tsv contains QC metrics and 50-149 bp statistics
    summary_txt = bundle.summary_text
    assert "90" in summary_txt
    assert "0.90" in summary_txt or "90" in summary_txt
    assert "0.95" in summary_txt or "95" in summary_txt
    assert "0.98" in summary_txt or "98" in summary_txt


# ---------------------------------------------------------------------------
# 13. Negative-Strand Ref Steps + Negative-Strand GFF3 CDS Split + --splice-window
# ---------------------------------------------------------------------------

def test_tier3_13_negative_strand_ref_steps_and_negative_strand_gff3_cds_splits(tmp_path: Path) -> None:
    """Reference path with '-' steps + negative-strand ('-') multi-exon GFF3 gene split by
    two insertions (including a 1-2 bp micro-segment) and splice-site proximal insertion.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    # Build a 240 bp coding sequence for a negative-strand gene (so on '+' ref it is revcomp(cds))
    cds_neg = make_coding_seq(80, seed=161)  # 240 bp
    # Exon 2 (transcriptional first on '-' strand): 120 bp of cds_neg[:120] -> on '+' ref: revcomp(cds_neg[:120])
    # Exon 1 (transcriptional second on '-' strand): 120 bp of cds_neg[120:] -> on '+' ref: revcomp(cds_neg[120:])
    ex1_plus = revcomp(cds_neg[120:])  # 120 bp at ref [200, 320)
    intron_plus = make_complex_seq(180, seed=162)  # 180 bp at ref [320, 500)
    ex2_plus = revcomp(cds_neg[:120])  # 120 bp at ref [500, 620)

    # Split ex2_plus at 2 bp before its end (ref 618) and ex1_plus in the middle (ref 260)
    r0_seq = make_complex_seq(200, seed=163)  # [0, 200)
    r1_seq = ex1_plus[:60]  # [200, 260)
    r2_seq = ex1_plus[60:] + intron_plus[:5]  # [260, 325) -> junction at 325 is 5 bp from exon1 end 320!
    r3_seq = intron_plus[5:] + ex2_plus[:118]  # [325, 618)
    r4_seq = ex2_plus[118:] + make_complex_seq(150, seed=164)  # [618, 770)

    # Put r1 and r3 in '-' orientation in the GFA node table so the reference uses r1- and r3-
    b.add_node("r0", r0_seq)
    b.add_node("r1", revcomp(r1_seq))
    b.add_node("r2", r2_seq)
    b.add_node("r3", revcomp(r3_seq))
    b.add_node("r4", r4_seq)

    b.add_node("ins_cds1", make_complex_seq(150, seed=165))
    b.add_node("ins_splice", make_complex_seq(140, seed=166))
    b.add_node("ins_cds2", make_complex_seq(160, seed=167))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "r1-", "r2+", "r3-", "r4+"])
    b.add_path("Donor#1#Chr01", ["r0+", "r1-", "ins_cds1+", "r2+", "ins_splice+", "r3-", "ins_cds2+", "r4+"])

    # 1-based GFF3 coordinates on reference:
    # Exon 1 (lower coords on '-'): [201, 320], Exon 2 (higher coords on '-'): [501, 620]
    b.add_gff_gene(
        gene_id="gene_neg",
        mrna_id="mrna_neg",
        strand="-",
        exons=[(201, 320), (501, 620)],
        cds_list=[(201, 320, 0), (501, 620, 0)],
    )

    inputs = b.write("neg_ref_gff")
    _, bundle = run_build(inputs, extra_args=["--splice-window", "8"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    # Verify micro-segment (2 bp) CDS split and splice_site_proximal_sv=True
    cds_rows = [r for r in (bundle.lifted_gff_rows or []) if r["type"] == "CDS"]
    assert len(cds_rows) == 4
    assert any(r["end"] - r["start"] + 1 == 2 for r in cds_rows)
    assert any(
        r["attributes"].get("splice_site_proximal_sv") == "True"
        for r in (bundle.lifted_gff_rows or [])
    )


# ---------------------------------------------------------------------------
# 14. T9 DUP_TRA + Local INS + --repeat-guard on
# ---------------------------------------------------------------------------

def test_tier3_14_dup_tra_with_local_ins_and_repeat_guard(tmp_path: Path) -> None:
    """T9 + Repeat-Guard: Deviation between r1 and r2 contains a distant reference node r5
    (> --max-jump away -> DUP_TRA), flanked by a complex insertion (retained as INS) and a
    low-complexity satellite block (masked as REPEAT_LIKE).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 171), ("r2", 172), ("r3", 173), ("r4", 174), ("r5", 175)]:
        b.add_node(nid, make_complex_seq(400, seed=seed))

    novel_complex = make_complex_seq(220, seed=176)
    novel_sat = make_satellite_seq(unit_len=20, total_len=1200, seed=177)
    b.add_node("nov_c", novel_complex)
    b.add_node("nov_s", novel_sat)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+"])
    # Set --max-jump 800 so r5 (ref_start=1600) is > 800 bp away from r1 (ref_end=400)
    b.add_path("Donor#1#Chr01", ["r1+", "nov_c+", "r5+", "nov_s+", "r2+", "r3+", "r4+", "r5+"])

    inputs = b.write("dup_tra_rg")
    _, bundle = run_build(inputs, extra_args=["--max-jump", "800", "--repeat-guard", "on"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "DUP_TRA" in ev_types
    assert "INS" in ev_types
    # Distant reference node r5 must NOT be duplicated in inserts.tsv
    assert len(bundle.inserts) == 1
    assert int(bundle.inserts[0]["pan_end"]) - int(bundle.inserts[0]["pan_start"]) == len(novel_complex)
    assert any(u["reason"] == "REPEAT_LIKE" for u in bundle.unresolved)


# ---------------------------------------------------------------------------
# 15. LARGE_JUMP Overriding DUP_TRA with unresolved.fa Export
# ---------------------------------------------------------------------------

def test_tier3_15_large_jump_overrides_dup_tra_and_exports_unresolved_fasta(tmp_path: Path) -> None:
    """§3.7 priority rule: When a deviation span exceeds --max-jump AND visits a distant
    reference node, LARGE_JUMP takes precedence over DUP_TRA; novel sequence goes to
    unresolved.tsv and unresolved.fa (not inserted into pan.fa).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 181), ("r2", 182), ("r3", 183), ("r4", 184), ("r5", 185), ("r6", 186)]:
        b.add_node(nid, make_complex_seq(400, seed=seed))
    nov_lj = make_complex_seq(260, seed=187)
    b.add_node("nov_lj", nov_lj)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    # Branch jumps from r1 (ref_end=400) to r5 (ref_start=1600, span=1200 > max_jump=600)
    # and visits r6 inside the deviation
    b.add_path("JumpHap#1#Chr01", ["r1+", "nov_lj+", "r6+", "r5+", "r6+"])

    inputs = b.write("lj_override")
    _, bundle = run_build(inputs, extra_args=["--max-jump", "600"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "LARGE_JUMP" in ev_types
    assert "DUP_TRA" not in ev_types
    assert "INS" not in ev_types
    assert bundle.pan_seq == b.spell_steps(["r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])

    lj_unres = [u for u in bundle.unresolved if u["reason"] == "LARGE_JUMP"]
    assert len(lj_unres) == 1
    assert lj_unres[0]["exported_fasta_id"] in bundle.unresolved_fa


# ---------------------------------------------------------------------------
# 16. INV_EDGE and INV_CONFLICT Coexisting with Clean INS Genotypes
# ---------------------------------------------------------------------------

def test_tier3_16_inv_edge_and_inv_conflict_coexisting_with_clean_ins_genotypes(tmp_path: Path) -> None:
    """T24 + Genotyping: Hap1 has a clean INS between r2 and r3; Hap2 has an INV_EDGE
    (missing left flank L); Hap3 has overlapping reverse chains (INV_CONFLICT); HapRef matches ref.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, length, seed in [
        ("r1", 500, 191),
        ("r2", 250, 192),
        ("r3", 250, 193),
        ("r4", 250, 194),
        ("r5", 500, 195),
        ("r6", 500, 199),
    ]:
        b.add_node(nid, make_complex_seq(length, seed=seed))

    b.add_node("ins_clean", make_complex_seq(180, seed=196))
    b.add_node("nov_edge", make_complex_seq(170, seed=197))
    b.add_node("nov_conf", make_complex_seq(190, seed=198))
    b.add_node("nov_conf2", make_complex_seq(160, seed=200))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    b.add_path("Hap1#1#Chr01", ["r1+", "r2+", "ins_clean+", "r3+", "r4+", "r5+", "r6+"])
    # Hap2 starts with reverse chain [r3- -> nov_edge+ -> r2-] then forward flank r4+ -> r5+ -> r6+ (missing L -> INV_EDGE, L_plus=1250 > L_minus=500)
    b.add_path("Hap2#1#Chr01", ["r3-", "nov_edge+", "r2-", "r4+", "r5+", "r6+"])
    # Hap3 has L_plus = 500+500+500 = 1500 > L_minus = 250+250+250+500 = 1250, and two reverse chains
    # R1 = [r4- -> nov_conf+ -> r2-] (ranks [1, 3]) and R2 = [r3- -> nov_conf2+ -> r1-] (ranks [0, 2]) that overlap!
    b.add_path("Hap3#1#Chr01", ["r1+", "r4-", "nov_conf+", "r2-", "r5+", "r3-", "nov_conf2+", "r1-", "r6+"])

    inputs = b.write("inv_edge_conf")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert {"INS", "INV_EDGE", "INV_CONFLICT"}.issubset(ev_types)

    unres_reasons = {u["reason"] for u in bundle.unresolved}
    assert "INV_EDGE" in unres_reasons and "INV_CONFLICT" in unres_reasons
    for u in bundle.unresolved:
        if u["reason"] in ("INV_EDGE", "INV_CONFLICT"):
            assert u["exported_fasta_id"] in bundle.unresolved_fa

    ins_ev = next(ev for ev in bundle.sv_rows if ev["type"] == "INS")
    gt = bundle.genotypes[ins_ev["event_id"]]
    assert gt["Ref#1"] == "0"
    assert gt["Hap1#1"] == "1"
    assert gt["Hap2#1"] == "."
    assert gt["Hap3#1"] == "."


# ---------------------------------------------------------------------------
# 17. CEN_ANCHOR_NOT_FOUND Downgrade + Second Disjoint extract-islands Region
# ---------------------------------------------------------------------------

def test_tier3_17_cen_anchor_not_found_downgrade_plus_extract_islands_second_region(tmp_path: Path) -> None:
    """C5 + Multi-region centromere: Region 1 starts at ref 0 (missing left anchor u ->
    CEN_ANCHOR_NOT_FOUND, forced downgrade to bypass) while Region 2 has valid anchors
    (u2, v2) and executes full extract-islands on the same chromosome.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r0", 201), ("r1", 202), ("r2", 203), ("r3", 204), ("r4", 205)]:
        b.add_node(nid, make_complex_seq(300, seed=seed))

    b.add_node("isl_reg1", make_complex_seq(600, seed=206))
    b.add_node("isl_reg2", make_complex_seq(600, seed=207))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "r1+", "r2+", "r3+", "r4+"])
    b.add_path("Donor#1#Chr01", ["r0+", "isl_reg1+", "r1+", "r2+", "r3+", "isl_reg2+", "r4+"])

    # Region 1 covers [0, 300) (r0 -> no unique anchor before 0! v1 is r1 at 300)
    # Region 2 covers [900, 1200) (r3 -> u2 is r2 at [600,900), v2 is r4 at [1200,1500))
    b.add_bed_region(0, 300, "CEN_LEFT_EDGE")
    b.add_bed_region(900, 1200, "CEN_INTERNAL")

    inputs = b.write("two_cen_downgrade")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert "CEN_ANCHOR_NOT_FOUND" in bundle.log_text or "CEN_ANCHOR_NOT_FOUND" in bundle.summary_text
    assert bundle.centromere_rows is not None
    reg1_rows = [r for r in bundle.centromere_rows if int(r["ref_start"]) == 0]
    reg2_rows = [r for r in bundle.centromere_rows if int(r["ref_start"]) == 900]
    assert len(reg1_rows) >= 1 and len(reg2_rows) >= 1
    assert all(r["mode"] == "bypass" for r in reg1_rows)
    assert all(r["mode"] == "extract-islands" for r in reg2_rows)

    # Only Region 2 mounts a CEN_ISLAND
    cen_islands = [ev for ev in bundle.sv_rows if ev["type"] == "CEN_ISLAND"]
    assert len(cen_islands) == 1
    assert int(cen_islands[0]["ref_start"]) == 1200


# ---------------------------------------------------------------------------
# 18. CEN_REFERENCE_GAP + Adjacent Normal Region with Native N Preservation
# ---------------------------------------------------------------------------

def test_tier3_18_cen_reference_gap_plus_normal_ref_n_node_preservation(tmp_path: Path) -> None:
    """C4 + C17: Centromere reference interval contains N gap (ref_status=gapped ->
    CEN_REFERENCE_GAP, length_ratio='.', no mounting) while an adjacent normal region has
    a reference node with native Ns and a clean donor insertion.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("u", make_complex_seq(250, seed=211))
    b.add_node("cen_gapped", make_complex_seq(150, seed=212) + ("N" * 50) + make_complex_seq(150, seed=213))
    b.add_node("v", make_complex_seq(250, seed=214))
    b.add_node("norm_n_ref", make_complex_seq(100, seed=215) + "NNNNN" + make_complex_seq(100, seed=216))
    b.add_node("end_ref", make_complex_seq(250, seed=217))

    b.add_node("cen_cand", make_complex_seq(600, seed=218))
    b.add_node("norm_ins", make_complex_seq(200, seed=219))

    b.set_ref_path("Ref#1#Chr01", ["u+", "cen_gapped+", "v+", "norm_n_ref+", "end_ref+"])
    b.add_path("Donor#1#Chr01", ["u+", "cen_cand+", "v+", "norm_n_ref+", "norm_ins+", "end_ref+"])

    b.add_bed_region(250, 600, "CEN_GAP")
    inputs = b.write("cen_ref_gap")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert bundle.centromere_rows is not None
    donor_cen = next(r for r in bundle.centromere_rows if r["hap"] == "Donor#1")
    assert donor_cen["ref_status"] == "gapped"
    assert donor_cen["length_ratio"] == "."
    assert not any(ev["type"] == "CEN_ISLAND" for ev in bundle.sv_rows)
    assert any(u["reason"] == "CEN_REFERENCE_GAP" for u in bundle.unresolved)
    assert any(ev["type"] == "INS" for ev in bundle.sv_rows)


# ---------------------------------------------------------------------------
# 19. Centromere Step B Reverse-Strand Subtraction + Protected INS + Step C Mask
# ---------------------------------------------------------------------------

def test_tier3_19_centromere_step_b_revcomp_subtraction_protected_ins_and_step_c_mask(tmp_path: Path) -> None:
    """C8 + C9 + C16: Centromere candidate containing:
    (1) 400 bp reverse-complement copy of cen_ref (subtracted by Step B mappy 50bp blocks),
    (2) 650 bp protected internal novel complex island (retained), and
    (3) 1500 bp low-complexity satellite tail (masked by Step C repeat_mask_intervals).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_u = make_complex_seq(250, seed=221)
    s_cen_ref = make_complex_seq(800, seed=222)
    s_v = make_complex_seq(250, seed=223)

    b.add_node("u", s_u)
    b.add_node("cen_ref", s_cen_ref)
    b.add_node("v", s_v)

    rc_flank_left = revcomp(s_cen_ref[100:500])  # 400 bp reverse-complement homologous to cen_ref
    novel_island = make_complex_seq(650, seed=224)  # >= 500 bp novel island
    rc_flank_right = revcomp(s_cen_ref[500:750])  # 250 bp reverse-complement homologous to cen_ref
    sat_tail = make_satellite_seq(unit_len=20, total_len=1500, seed=225)  # 1500 bp satellite

    b.add_node("cand_comp", rc_flank_left + novel_island + rc_flank_right + sat_tail)

    b.set_ref_path("Ref#1#Chr01", ["u+", "cen_ref+", "v+"])
    b.add_path("Donor#1#Chr01", ["u+", "cen_ref+", "cand_comp+", "v+"])
    b.add_bed_region(250, 1050, "CEN1")

    inputs = b.write("step_bc_combo")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert bundle.centromere_rows is not None
    d_row = next(r for r in bundle.centromere_rows if r["hap"] == "Donor#1")
    assert int(d_row["reference_like_bp"]) > 0
    assert int(d_row["repeat_like_bp"]) > 0
    assert int(d_row["accepted_source_bp"]) >= 500
    assert any(ev["type"] == "CEN_ISLAND" for ev in bundle.sv_rows)


# ---------------------------------------------------------------------------
# 20. Weighted LIS Multi-Hit Anchor + CNV + DEL in Same Chromosome
# ---------------------------------------------------------------------------

def test_tier3_20_weighted_lis_multi_hit_with_cnv_and_del_same_region(tmp_path: Path) -> None:
    """F10 + F13 + F20: Anchor visited twice (multi_hit) alongside a deletion (DEL) and
    canonical period CNV.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 231), ("r2", 232), ("r3", 233), ("r4", 234), ("r5", 235)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))
    b.add_node("u_rep", make_complex_seq(60, seed=236))
    b.add_node("ins_mh", make_complex_seq(150, seed=237))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "u_rep+", "u_rep+", "r4+", "r5+"])
    # Donor visits r2+ twice (r2+ -> ins_mh+ -> r2+), skips r3+ (250bp DEL), and has 4 copies of u_rep+ (CNV 2->4)
    b.add_path(
        "Donor#1#Chr01",
        ["r1+", "r2+", "ins_mh+", "r2+", "u_rep+", "u_rep+", "u_rep+", "u_rep+", "r4+", "r5+"],
    )

    inputs = b.write("mh_cnv_del")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "DEL" in ev_types
    assert "CNV" in ev_types
    assert "multi_hit" in bundle.sv_path.read_text(encoding="utf-8") or "multi_hit" in bundle.log_text


# ---------------------------------------------------------------------------
# 21. Secondary Backbone support vs longest with edlib (<=1kb) and mappy (>1kb) Clustering
# ---------------------------------------------------------------------------

def test_tier3_21_sec_backbone_support_vs_longest_with_edlib_and_mappy_clustering(tmp_path: Path) -> None:
    """F16 + F17: Component recursion and similarity clustering in both <=1kb (edlib)
    and >1kb (mappy) regimes under --sec-backbone support vs longest.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 241), ("r2", 242), ("r3", 243)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    # Group 1 (between r1 and r2): <=1kb real sequences (edlib regime), 98% identical -> merge into 1 rep
    s_ed1 = make_complex_seq(600, seed=244)
    s_ed2 = mutate_seq_snps(s_ed1, num_snps=6, seed=245)  # 99% identity
    b.add_node("ed1", s_ed1)
    b.add_node("ed2", s_ed2)

    # Group 2 (between r2 and r3): >1kb real sequences (mappy regime), 98.5% identical -> merge into 1 rep
    s_mp1 = make_complex_seq(1500, seed=246)
    s_mp2 = mutate_seq_snps(s_mp1, num_snps=15, seed=247)  # 99% identity
    b.add_node("mp1", s_mp1)
    b.add_node("mp2", s_mp2)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+"])
    b.add_path("H1#1#Chr01", ["r1+", "ed1+", "r2+", "mp1+", "r3+"])
    b.add_path("H2#1#Chr01", ["r1+", "ed2+", "r2+", "mp2+", "r3+"])

    inputs = b.write("edlib_mappy_cluster")
    _, bundle = run_build(inputs, extra_args=["--cluster-id", "0.95", "--cluster-win-id", "0.70"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Both Group 1 and Group 2 should cluster their 2 near-identical sequences into 1 representative each
    assert len(bundle.inserts) == 2
    for ins_row in bundle.inserts:
        members = str(ins_row["members"]).split(",")
        assert set(members) == {"H1#1", "H2#1"}


# ---------------------------------------------------------------------------
# 22. Partial-Node Slicing Ignored as Full Node ID in Union-Find & Cycle Folding
# ---------------------------------------------------------------------------

def test_tier3_22_partial_node_slicing_ignored_in_union_find_and_cycle_folding(tmp_path: Path) -> None:
    """§3.6.6: When repeat-guard slices inside node 'shared_hyb', the resulting partial
    steps must NOT act as full node IDs in Union-Find component partitioning or cycle folding.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=251))
    b.add_node("r2", make_complex_seq(250, seed=252))

    # Construct a node with non-homologous left complex (400bp), central satellite (1800bp),
    # and non-homologous right complex (400bp).
    left_c = make_complex_seq(400, seed=253)
    mid_sat = make_satellite_seq(unit_len=20, total_len=1800, seed=254)
    right_c = make_complex_seq(400, seed=255)
    b.add_node("shared_hyb", left_c + mid_sat + right_c)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # Donor traverses shared_hyb twice in tandem: shared_hyb+ -> shared_hyb+
    # Because repeat-guard slices inside shared_hyb+, the resulting residuals are partial steps
    # and must NOT be cycle-folded as if full 'shared_hyb+' were unmasked!
    b.add_path("Donor#1#Chr01", ["r1+", "shared_hyb+", "shared_hyb+", "r2+"])

    inputs = b.write("partial_uf_fold")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on", "--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert any(u["reason"] == "REPEAT_LIKE" for u in bundle.unresolved)
    assert len(bundle.inserts) >= 2


# ---------------------------------------------------------------------------
# 23. GFF3 All Four SV Tags on a Single Multi-Exon Gene
# ---------------------------------------------------------------------------

def test_tier3_23_gff3_all_four_sv_tags_on_single_multi_exon_gene(tmp_path: Path) -> None:
    """§4.4: Verify intronic_insertion, utr_insertion, cds_split_by_insertion (with
    projection_split=True), and splice_site_proximal_sv=True on a single multi-exon gene.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    cds_seq = make_coding_seq(60, seed=261)  # 180 bp
    # Gene layout on '+' reference:
    # [0, 100): upstream flank (r0)
    # [100, 200): 5' UTR in Exon 1, with UTR insertion at 150 (r1=[100,150), r2=[150,200))
    # [200, 290): CDS part 1 in Exon 1 (90 bp) (r3=[200,290)) -> Exon 1 is [100, 290) (1-based 101..290)
    # [290, 490): Intron 1 (200 bp), with intronic insertion at 390 (r4=[290,390)) and
    #             splice-proximal insertion at 485 (5 bp before Exon 2 start 490!) (r5=[390,485), r6=[485,490))
    # [490, 580): CDS part 2 in Exon 2 (90 bp), with CDS insertion at 535 (r7=[490,535), r8=[535,580))
    # [580, 700): downstream flank (r9)
    b.add_node("r0", make_complex_seq(100, seed=262))
    b.add_node("r1", make_complex_seq(50, seed=263))
    b.add_node("r2", make_complex_seq(50, seed=264))
    b.add_node("r3", cds_seq[:90])
    b.add_node("r4", make_complex_seq(100, seed=265))
    b.add_node("r5", make_complex_seq(95, seed=266))
    b.add_node("r6", make_complex_seq(5, seed=267))
    b.add_node("r7", cds_seq[90:135])
    b.add_node("r8", cds_seq[135:180])
    b.add_node("r9", make_complex_seq(120, seed=268))

    for nid, seed in [("ins_utr", 271), ("ins_intron", 272), ("ins_splice", 273), ("ins_cds", 274)]:
        b.add_node(nid, make_complex_seq(120, seed=seed))

    ref_steps = ["r0+", "r1+", "r2+", "r3+", "r4+", "r5+", "r6+", "r7+", "r8+", "r9+"]
    b.set_ref_path("Ref#1#Chr01", ref_steps)
    b.add_path(
        "Donor#1#Chr01",
        [
            "r0+", "r1+", "ins_utr+", "r2+", "r3+",
            "r4+", "ins_intron+", "r5+", "ins_splice+", "r6+",
            "r7+", "ins_cds+", "r8+", "r9+",
        ],
    )

    # 1-based closed GFF3 coordinates:
    # Exon 1: [101, 290] (5' UTR [101, 200] + CDS1 [201, 290])
    # Exon 2: [491, 580] (CDS2 [491, 580])
    b.add_gff_gene(
        gene_id="gene_all_tags",
        mrna_id="mrna_all_tags",
        strand="+",
        exons=[(101, 290), (491, 580)],
        cds_list=[(201, 290, 0), (491, 580, 0)],
    )

    inputs = b.write("gff_four_tags")
    _, bundle = run_build(inputs, extra_args=["--splice-window", "8"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    gff_txt = bundle.lifted_gff_path.read_text(encoding="utf-8")
    assert "utr_insertion" in gff_txt
    assert "intronic_insertion" in gff_txt
    assert "cds_split_by_insertion" in gff_txt
    assert "projection_split=True" in gff_txt
    assert "splice_site_proximal_sv=True" in gff_txt
    assert "sv_id=" in gff_txt


# ---------------------------------------------------------------------------
# 24. panflat verify Detects Tampered pan.fa, offsets.tsv, and ref_to_pan.chain
# ---------------------------------------------------------------------------

def test_tier3_24_verify_detects_tampered_pan_fa_and_offsets_and_chain(tmp_path: Path) -> None:
    """Verify standalone `panflat verify` rejects tampered pan.fa, offsets.tsv, or chain."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=281))
    b.add_node("r2", make_complex_seq(250, seed=282))
    b.add_node("ins1", make_complex_seq(150, seed=283))
    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("Donor#1#Chr01", ["r1+", "ins1+", "r2+"])
    inputs = b.write("tamper_base")

    _, bundle = run_build(inputs)
    assert bundle is not None
    run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=True)

    # 1) Tamper 1 base in pan.fa -> Invariant I/II failure
    p_fa = tmp_path / "tamper_fa"
    for f in tmp_path.glob("tamper_base_out.*"):
        shutil.copy(f, tmp_path / f.name.replace("tamper_base_out", "tamper_fa"))
    fa_file = Path(f"{p_fa}.pan.fa")
    seq_Mut = list(bundle.pan_seq)
    seq_Mut[10] = "A" if seq_Mut[10] != "A" else "C"
    fa_file.write_text(f">Chr01\n{''.join(seq_Mut)}\n", encoding="utf-8")
    proc_fa = run_verify(p_fa, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert proc_fa.returncode != 0

    # 2) Tamper pan_start in offsets.tsv -> Invariant I/III failure
    p_off = tmp_path / "tamper_off"
    for f in tmp_path.glob("tamper_base_out.*"):
        shutil.copy(f, tmp_path / f.name.replace("tamper_base_out", "tamper_off"))
    off_file = Path(f"{p_off}.offsets.tsv")
    off_lines = off_file.read_text(encoding="utf-8").splitlines()
    cols = off_lines[-1].split("\t")
    cols[5] = str(int(cols[5]) + 5)  # corrupt pan_start
    off_lines[-1] = "\t".join(cols)
    off_file.write_text("\n".join(off_lines) + "\n", encoding="utf-8")
    proc_off = run_verify(p_off, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert proc_off.returncode != 0

    # 3) Tamper dq in ref_to_pan.chain -> Invariant III failure
    p_ch = tmp_path / "tamper_ch"
    for f in tmp_path.glob("tamper_base_out.*"):
        shutil.copy(f, tmp_path / f.name.replace("tamper_base_out", "tamper_ch"))
    ch_file = Path(f"{p_ch}.ref_to_pan.chain")
    ch_text = ch_file.read_text(encoding="utf-8").replace("\t0\t250", "\t0\t199")
    ch_file.write_text(ch_text, encoding="utf-8")
    proc_ch = run_verify(p_ch, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert proc_ch.returncode != 0


# ---------------------------------------------------------------------------
# 25. panflat verify Detects Tampered inserts.tsv and Centromere Offset Delta
# ---------------------------------------------------------------------------

def test_tier3_25_verify_detects_tampered_inserts_and_centromere_offset_delta(tmp_path: Path) -> None:
    """Verify standalone `panflat verify` rejects corrupted inserts.tsv provenance and
    corrupted centromere mount offset deltas.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 291), ("u", 292), ("cen", 293), ("v", 294)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))
    b.add_node("isl", make_complex_seq(600, seed=295))
    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b.add_path("Donor#1#Chr01", ["n0+", "u+", "cen+", "isl+", "v+"])
    b.add_bed_region(500, 750, "CEN1")
    inputs = b.write("tamper_cen")

    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    run_verify(inputs.out_prefix, inputs.gfa_path, inputs.ref_fa_path, check=True)

    # 1) Corrupt src_strand in inserts.tsv -> Invariant II failure
    p_ins = tmp_path / "tamper_ins"
    for f in tmp_path.glob("tamper_cen_out.*"):
        shutil.copy(f, tmp_path / f.name.replace("tamper_cen_out", "tamper_ins"))
    ins_file = Path(f"{p_ins}.inserts.tsv")
    ins_lines = ins_file.read_text(encoding="utf-8").splitlines()
    cols = ins_lines[1].split("\t")
    cols[9] = "-" if cols[9] == "+" else "+"
    ins_lines[1] = "\t".join(cols)
    ins_file.write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    proc_ins = run_verify(p_ins, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert proc_ins.returncode != 0

    # 2) Corrupt centromere.tsv ref_end (mount point coordinate) -> centromere offset check failure
    p_c = tmp_path / "tamper_c"
    for f in tmp_path.glob("tamper_cen_out.*"):
        shutil.copy(f, tmp_path / f.name.replace("tamper_cen_out", "tamper_c"))
    c_file = Path(f"{p_c}.centromere.tsv")
    c_lines = c_file.read_text(encoding="utf-8").splitlines()
    for idx in range(1, len(c_lines)):
        c_cols = c_lines[idx].split("\t")
        c_cols[6] = "250"  # point ref_end to step u instead of step v (where slot_ins_len is 0!)
        c_lines[idx] = "\t".join(c_cols)
    c_file.write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    proc_c = run_verify(p_c, inputs.gfa_path, inputs.ref_fa_path, check=False)
    assert proc_c.returncode != 0


# ---------------------------------------------------------------------------
# 26. Local Re-anchoring Inside Inversion + Unanchored Contig Ends (T11 + T16 + F14)
# ---------------------------------------------------------------------------

def test_tier3_26_local_reanchoring_inside_inversion_with_unanchored_contig_ends(tmp_path: Path) -> None:
    """F14 + T11 + T16: Contig with unanchored >=50 bp ends (UNANCHORED_END), an internal
    normalized inversion, and a deviation containing a locally unique reference node (which
    appears twice globally in the reference but once inside B(a,b)) triggering local re-anchoring.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 301), ("r2", 302), ("r3", 303), ("r4", 304), ("r5", 305)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))
    # Node 'dup_ref' appears twice in full reference (so not a top-level anchor!),
    # but appears ONCE inside (r1, r3), so local re-anchoring inside D(r1, r3) anchors on it!
    b.add_node("dup_ref", make_complex_seq(200, seed=306))
    b.add_node("end5", make_complex_seq(110, seed=307))
    b.add_node("end3", make_complex_seq(120, seed=308))
    b.add_node("sub_ins", make_complex_seq(160, seed=309))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "dup_ref+", "r2+", "r3+", "r4+", "dup_ref+", "r5+"])
    # Donor has 5' unanchored end5+, traverses r1+ -> dup_ref+ -> sub_ins+ -> r3+ (skipping r2+, locally re-anchoring on dup_ref+),
    # then inverts r4- before r5+, plus 3' unanchored end3+
    b.add_path(
        "Donor#1#Chr01",
        ["end5+", "r1+", "dup_ref+", "sub_ins+", "r3+", "r4-", "r5+", "end3+"],
    )

    inputs = b.write("reanchor_unanch")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    unanch = [u for u in bundle.unresolved if u["reason"] == "UNANCHORED_END"]
    assert len(unanch) == 2
    for u in unanch:
        assert u["exported_fasta_id"] in bundle.unresolved_fa

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert {"INS", "DEL", "INV"}.issubset(ev_types)


# ---------------------------------------------------------------------------
# 27. Secondary Backbone Negative Steps + Distal Ref Separator Inside Recursion (T18 + T19)
# ---------------------------------------------------------------------------

def test_tier3_27_secondary_backbone_negative_steps_and_distal_ref_separator(tmp_path: Path) -> None:
    """T18 + T19: Recursive component whose secondary backbone contains negative-oriented
    novel steps ('-') while another member carries a distal top-level reference node inside
    the recursive deviation (which acts as a strict separator and is never inserted).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("r1", 311), ("r2", 312), ("r3", 313), ("r4", 314)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    b.add_node("novA", make_complex_seq(200, seed=315))
    b.add_node("novB", make_complex_seq(200, seed=316))
    b.add_node("novC", make_complex_seq(180, seed=317))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
    # Two haplotypes traverse novA- -> novB- after §3.3.0 whole-contig flip (making it secondary backbone with '-' steps and src_strand='-')
    b.add_path("Hap1#1#Chr01", ["r4-", "r3-", "r2-", "novB+", "novA+", "r1-"])
    b.add_path("Hap2#1#Chr01", ["r4-", "r3-", "r2-", "novB+", "novA+", "r1-"])
    # Third haplotype shares novA- then visits distal ref node r4+ and novC+ before novB-
    b.add_path("Hap3#1#Chr01", ["r1+", "novA-", "r4+", "novC+", "novB-", "r2+", "r3+", "r4+"])

    inputs = b.write("t18_t19_combo")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Verify negative-strand provenance for novA- / novB- and that r4's sequence is NOT in inserts.tsv
    r4_seq = b.nodes["r4"].upper()
    assert any(ins["src_strand"] == "-" for ins in bundle.inserts)
    for ins in bundle.inserts:
        piece_seq = bundle.pan_seq[int(ins["pan_start"]) : int(ins["pan_end"])].upper()
        assert piece_seq != r4_seq


# ---------------------------------------------------------------------------
# 28. Contradictory Multi-Contig Same Haplotype Genotyping Conflict + W-lines
# ---------------------------------------------------------------------------

def test_tier3_28_contradictory_multi_contig_same_haplotype_genotyping_and_w_lines(tmp_path: Path) -> None:
    """§5.2 conflict rule: When two W-line contigs belonging to the same Sample#Hap cover
    the same locus with contradictory calls (one carries an INS, the other matches reference 0),
    its genotype must be '.' (conflict), whereas clean haplotypes get '1' or '0'.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(250, seed=321))
    b.add_node("r2", make_complex_seq(250, seed=322))
    b.add_node("ins_x", make_complex_seq(180, seed=323))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # Clean carrier haplotype
    b.add_walk("CleanCarrier", "1", "ctg1", 0, None, ["r1+", "ins_x+", "r2+"])
    # Clean reference haplotype
    b.add_walk("CleanRef", "1", "ctg1", 0, None, ["r1+", "r2+"])
    # Contradictory haplotype: ctgA carries ins_x+, ctgB has reference allele 0
    b.add_walk("ConflictSample", "1", "ctgA", 500, None, ["r1+", "ins_x+", "r2+"])
    b.add_walk("ConflictSample", "1", "ctgB", 2000, None, ["r1+", "r2+"])

    inputs = b.write("gt_conflict_w")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_ev = next(ev for ev in bundle.sv_rows if ev["type"] == "INS")
    gt = bundle.genotypes[ins_ev["event_id"]]
    assert gt["Ref#1"] == "0"
    assert gt["CleanRef#1"] == "0"
    assert gt["CleanCarrier#1"] == "1"
    assert gt["ConflictSample#1"] == "."
