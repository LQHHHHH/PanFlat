"""M7 deep white-box code-review hardening tests across models, GFA I/O, normalizer, repeat_guard, centromere, linearizer, liftover, writers, and verify_qc."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from panflat.cli import main as cli_main
from panflat.gfa_io import load_and_validate_inputs, validate_build_config
from panflat.gff_liftover import liftover_gff3
from panflat.linearizer import assemble_pan_and_offsets, run_chromosome_pipeline
from panflat.models import (
    BuildConfig,
    ChromosomeBuildResult,
    CompactNodeStore,
    CompactRefNodeSet,
    CompactVisitCounts,
    InputContractError,
    InsertionSlotBlock,
    LazyProvenanceTuple,
    LazyStepOffsetSequence,
    LazyStepTuple,
    Piece,
    Step,
    SVEvent,
    is_compact_nid_str,
)
from panflat.normalizer import (
    compute_raw_visit_counts,
    normalize_inversions_top_level,
    orient_contig_top_level,
)
from panflat.repeat_guard import repeat_mask_intervals
from panflat.verify_qc import run_qc, verify_build_outputs
from panflat.writers import _atomic_open_text, merge_chromosomes, write_build_outputs
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    canonical_15mer_stats,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
)
from tests.e2e.helpers import (
    OutputBundle,
    SyntheticInputBuilder,
    assert_all_global_invariants,
    run_build,
)


def test_compact_models_reject_leading_zero_zero_overflow_and_negative_indexing() -> None:
    """Verify compact NID stores reject '0', leading-zero '01', and >MAX_COMPACT_NID strings without collision or OverflowError."""
    assert is_compact_nid_str("1")
    assert is_compact_nid_str("50000000")
    assert not is_compact_nid_str("")
    assert not is_compact_nid_str("0")
    assert not is_compact_nid_str("01")
    assert not is_compact_nid_str("50000001")
    assert not is_compact_nid_str("999999999999")
    assert not is_compact_nid_str("node1")

    store = CompactNodeStore(
        seq_bytes=b"ACGTGGGG",
        offsets=np.array([0, 0, 4, 8], dtype=np.int64),
        node_lens=np.array([0, 4, 4], dtype=np.int32),
        max_nid=2,
        extra={"01": "TTTT", "0": "CCCC"},
        count=4,
    )
    assert store["1"] == "ACGT"
    assert store["01"] == "TTTT"
    assert store["0"] == "CCCC"
    assert "999999999999" not in store
    assert "-1" not in store
    with pytest.raises(KeyError):
        _ = store["999999999999"]
    assert store.get("999999999999", "MISSING") == "MISSING"

    mask = np.array([0, 1, 0], dtype=np.uint8)
    ref_set = CompactRefNodeSet(mask, count=2, extra=frozenset({"02"}))
    assert "1" in ref_set
    assert "01" not in ref_set
    assert "02" in ref_set
    assert "2" not in ref_set
    assert "0" not in ref_set
    assert "999999999999" not in ref_set

    vc = CompactVisitCounts(
        multi_nids=np.array([2], dtype=np.int32),
        multi_counts=np.array([3], dtype=np.int32),
        visited_bits=np.packbits(np.array([0, 1, 1], dtype=np.uint8)),
        max_visited_nid=2,
        total_visited=2,
    )
    assert vc["1"] == 1
    assert vc["2"] == 3
    assert "01" not in vc
    assert "0" not in vc
    assert "999999999999" not in vc
    assert vc.get("01", 0) == 0
    assert vc.get("999999999999", 0) == 0

    lo = LazyStepOffsetSequence(
        chrom="Chr01",
        ref_nids=np.array([1, 2], dtype=np.int32),
        ref_starts=np.array([0, 10], dtype=np.int64),
        ref_ends=np.array([10, 25], dtype=np.int64),
        pan_starts=np.array([0, 10], dtype=np.int64),
        pan_ends=np.array([10, 25], dtype=np.int64),
        slot_ins_lens=np.array([0, 0], dtype=np.int64),
    )
    assert lo[-1].ref_step == 1
    assert lo[-2].ref_step == 0
    with pytest.raises(IndexError):
        _ = lo[2]
    with pytest.raises(IndexError):
        _ = lo[-3]


def test_gfa_io_backbone_prefix_pansn_boundary_and_leading_zero_segment_fallback(
    tmp_path: Path,
) -> None:
    """Verify --backbone-prefix respects PanSN '#' boundaries and leading-zero GFA segments do not collide."""
    gb = SyntheticGFAFixtureBuilder()
    s_ref = make_complex_seq(120, seed=7001)
    s_alt = make_complex_seq(120, seed=7002)
    s_01 = make_complex_seq(80, seed=7003)
    gb.add_node("1", s_ref)
    gb.add_node("01", s_01)
    gb.add_node("2", s_alt)
    gb.add_p_path("Ref#1#Chr01", [("1", "+"), ("01", "+")])
    gb.add_p_path("RefAlt#1#Chr01", [("2", "+"), ("01", "+")])
    gb.add_p_path("Ref#1#Chr010", [("2", "+")])

    gfa_p = gb.write_gfa(tmp_path / "pansn.gfa")
    ref_fa_p = gb.write_fasta(
        tmp_path / "pansn.ref.fa",
        {
            "Ref#1#Chr01": s_ref + s_01,
            "RefAlt#1#Chr01": s_alt + s_01,
            "Ref#1#Chr010": s_alt,
        },
    )

    cfg_exact = BuildConfig(
        gfa=str(gfa_p),
        ref_fa=str(ref_fa_p),
        backbone_prefix="Ref#1#Chr01",
        out_prefix=str(tmp_path / "out_exact"),
    )
    graph_exact = load_and_validate_inputs(cfg_exact)
    assert graph_exact.ref_chrom == "Chr01"
    assert graph_exact.nodes["1"] == s_ref
    assert graph_exact.nodes["01"] == s_01

    gb2 = SyntheticGFAFixtureBuilder()
    gb2.add_node("1", s_ref)
    gb2.add_node("2", s_alt)
    gb2.add_p_path("Ref#1#Chr01", [("1", "+")])
    gb2.add_p_path("RefAlt#1#Chr01", [("2", "+")])
    gfa_p2 = gb2.write_gfa(tmp_path / "pansn2.gfa")
    ref_fa_p2 = gb2.write_fasta(
        tmp_path / "pansn2.ref.fa",
        {"Ref#1#Chr01": s_ref, "RefAlt#1#Chr01": s_alt},
    )
    cfg_sample = BuildConfig(
        gfa=str(gfa_p2),
        ref_fa=str(ref_fa_p2),
        backbone_prefix="Ref",
        out_prefix=str(tmp_path / "out_sample"),
    )
    graph_sample = load_and_validate_inputs(cfg_sample)
    assert graph_sample.ref_fragment.src_path == "Ref#1#Chr01"


def test_normalizer_preserves_lazy_step_tuple_on_orient_and_inversion_normalization(
    tmp_path: Path,
) -> None:
    """Verify orient_contig_top_level, normalize_inversions_top_level, and compute_raw_visit_counts preserve compact representations."""
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 7):
        gb.add_node(str(i), make_complex_seq(120, seed=7010 + i))
    gb.add_p_path(
        "Ref#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")],
    )
    gb.add_p_path(
        "AltRev#1#Chr01",
        [("5", "-"), ("4", "-"), ("3", "-"), ("2", "-"), ("1", "-")],
    )
    gb.add_p_path(
        "AltInv#1#Chr01",
        [("1", "+"), ("6", "+"), ("3", "-"), ("2", "-"), ("6", "+"), ("4", "+"), ("5", "+")],
    )

    gfa_p = gb.write_gfa(tmp_path / "norm_lazy.gfa")
    ref_seq = gb.spell_steps([("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")])
    fa_p = gb.write_fasta(tmp_path / "norm_lazy.fa", {"Chr01": ref_seq})
    cfg = BuildConfig(
        gfa=str(gfa_p),
        ref_fa=str(fa_p),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out_norm_lazy"),
    )
    graph = load_and_validate_inputs(cfg)
    rev_frag = next(f for f in graph.alt_fragments if f.hap_id == "AltRev#1")
    inv_frag = next(f for f in graph.alt_fragments if f.hap_id == "AltInv#1")
    assert isinstance(rev_frag.steps, LazyStepTuple)

    oriented = orient_contig_top_level(rev_frag, graph)
    assert oriented.is_flipped_contig
    assert isinstance(oriented.steps, LazyStepTuple)
    assert isinstance(oriented.provenance, LazyProvenanceTuple)

    inv_res = normalize_inversions_top_level(inv_frag, graph, cfg)
    assert len(inv_res.chains) == 1
    assert isinstance(inv_res.normalized_fragment.steps, LazyStepTuple)
    assert isinstance(inv_res.normalized_fragment.provenance, LazyProvenanceTuple)

    # Exercise >50_000 LazyStepTuple branch for CompactVisitCounts
    big_codes = np.arange(1, 50_005, dtype=np.int32)
    big_codes[-1] = 6  # node 6 visited twice
    vc = compute_raw_visit_counts(LazyStepTuple(big_codes))
    assert isinstance(vc, CompactVisitCounts)
    assert vc["6"] == 2
    assert vc["1"] == 1
    assert 6 in vc._multi_set
    assert 1 not in vc._multi_set


def test_co_locus_canonical_cnv_and_complete_del_both_emitted_with_shared_region(
    tmp_path: Path,
) -> None:
    """Verify a canonical CNV locus where another haplotype deletes the entire repeat emits both CNV and DEL sharing region_id."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("a", make_complex_seq(80, seed=7101))
    b.add_node("u", make_complex_seq(60, seed=7102))
    b.add_node("z", make_complex_seq(80, seed=7103))

    b.set_ref_path("Ref#1#Chr01", ["a+", "u+", "u+", "z+"])
    b.add_path("HapCNV#1#Chr01", ["a+", "u+", "u+", "u+", "z+"])
    b.add_path("HapDEL#1#Chr01", ["a+", "z+"])

    inputs = b.write("co_locus_cnv_del")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    cnv_evs = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    del_evs = [r for r in bundle.sv_rows if r["type"] == "DEL"]
    assert len(cnv_evs) == 1
    assert len(del_evs) == 1
    assert cnv_evs[0]["region_id"] != "."
    assert cnv_evs[0]["region_id"] == del_evs[0]["region_id"]
    assert bundle.genotypes[cnv_evs[0]["event_id"]]["HapCNV#1"] == "3"
    assert bundle.genotypes[del_evs[0]["event_id"]]["HapDEL#1"] == "1"
    assert bundle.genotypes[del_evs[0]["event_id"]]["HapCNV#1"] == "0"


def test_partial_cnv_multiallelic_alleles_str_and_flanking_del(
    tmp_path: Path,
) -> None:
    """Verify partial CNV with two distinct copy-number expansions alongside flanking DEL formats multi-allelic alleles."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("a", make_complex_seq(80, seed=7201))
    b.add_node("d", make_complex_seq(70, seed=7202))
    b.add_node("u", make_complex_seq(60, seed=7203))
    b.add_node("z", make_complex_seq(80, seed=7204))

    b.set_ref_path("Ref#1#Chr01", ["a+", "d+", "u+", "u+", "z+"])
    b.add_path("Hap3#1#Chr01", ["a+", "u+", "u+", "u+", "z+"])
    b.add_path("Hap4#1#Chr01", ["a+", "u+", "u+", "u+", "u+", "z+"])

    inputs = b.write("partial_cnv_multi")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    cnv_evs = [r for r in bundle.sv_rows if r["type"] == "CNV"]
    del_evs = [r for r in bundle.sv_rows if r["type"] == "DEL"]
    assert len(cnv_evs) == 1
    assert len(del_evs) == 1
    alleles_parts = cnv_evs[0]["alleles"].split(";")
    assert len(alleles_parts) == 2
    assert "k_alt=3" in cnv_evs[0]["alleles"]
    assert "k_alt=4" in cnv_evs[0]["alleles"]


def test_vectorized_deviation_prefilter_emits_large_jump_when_max_jump_lt_min_sv_len(
    tmp_path: Path,
) -> None:
    """Verify LazyStepTuple cand_mask preserves LARGE_JUMP deviations when max_jump < r_bp < min_sv_len."""
    gb = SyntheticGFAFixtureBuilder()
    ref_steps: list[tuple[str, str]] = []
    alt_steps: list[tuple[str, str]] = []
    for i in range(1, 30):
        nid = str(i)
        seq = make_complex_seq(120 if i == 15 else 60, seed=7300 + i)
        gb.add_node(nid, seq)
        ref_steps.append((nid, "+"))
        if i != 15:
            alt_steps.append((nid, "+"))

    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path("Alt#1#Chr01", alt_steps)
    gfa_p = gb.write_gfa(tmp_path / "lj_mask.gfa")
    ref_seq = gb.spell_steps(ref_steps)
    fa_p = gb.write_fasta(tmp_path / "lj_mask.ref.fa", {"Chr01": ref_seq})
    out_p = tmp_path / "lj_mask_out"

    rc = cli_main(
        [
            "build",
            "--gfa",
            str(gfa_p),
            "--ref-fa",
            str(fa_p),
            "--backbone-prefix",
            "Ref#1#",
            "--out-prefix",
            str(out_p),
            "--min-sv-len",
            "200",
            "--max-jump",
            "80",
        ]
    )
    assert rc == 0
    bundle = OutputBundle(out_p)
    lj_evs = [r for r in bundle.sv_rows if r["type"] == "LARGE_JUMP"]
    assert len(lj_evs) == 1


def test_right_telomere_virtual_end_centromere_and_trailing_slot_assembly(
    tmp_path: Path,
) -> None:
    """Verify VIRTUAL_END centromere BED downgrades cleanly and trailing n_ref slot assembly passes INV1/INV2/INV3."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s1 = make_complex_seq(200, seed=7401)
    su = make_complex_seq(200, seed=7402)
    scen = make_complex_seq(300, seed=7403)
    isl = make_complex_seq(600, seed=7404)
    for nid, seq in [("n1", s1), ("u", su), ("cen", scen), ("isl", isl)]:
        b.add_node(nid, seq)

    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "isl+"])
    b.add_bed_region(400, 700, "CEN_TEL")
    inputs = b.write("cen_virtual_end")

    _, bundle = run_build(
        inputs,
        extra_args=[
            "--centromere-mode",
            "extract-islands",
            "--centromere-min-island",
            "500",
        ],
    )
    assert bundle is not None
    assert bundle.summary_kv.get("cen_anchor_not_found") == "CEN_ANCHOR_NOT_FOUND"

    # Directly test trailing slot assembly at r_idx == n_ref (right_anchor_ref_step = 3)
    trail_prefix = tmp_path / "trail_slot"
    cfg = BuildConfig(
        gfa=str(inputs.gfa_path),
        ref_fa=str(inputs.ref_fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(trail_prefix),
        spacer_n=100,
    )
    graph = load_and_validate_inputs(cfg)
    n_ref = len(graph.ref_fragment.steps)
    piece = Piece(
        src_hap="Alt1#1",
        src_path="Alt1#1#Chr01",
        src_start=400,
        src_end=1000,
        src_strand="+",
        seq=isl,
        join_before="spacer",
        members=("Alt1#1",),
        allele_idx=1,
        piece_idx=1,
    )
    trail_blk = InsertionSlotBlock(
        right_anchor_ref_step=n_ref,
        is_centromere_island=True,
        group_span=1,
        left_anchor_ref_step=1,
        event_id="Chr01_CEN_1",
        representatives=((piece,),),
        trailing_spacer=True,
    )
    trail_ev = SVEvent(
        chrom="Chr01",
        pan_start=0,
        pan_end=0,
        event_id="Chr01_CEN_1",
        event_type="CEN_ISLAND",
        ref_start=700,
        ref_end=700,
        anchors="1..END",
        alleles_str="1:len=600,src=Alt1#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id="CEN_1",
        genotypes={"Ref#1": "0", "Alt1#1": "1"},
        pieces=(piece,),
    )
    pan_seq, step_offsets, upd_evs, ins_rows = assemble_pan_and_offsets(
        graph=graph,
        slot_blocks_by_step={n_ref: [trail_blk]},
        sv_events=[trail_ev],
        spacer_n=100,
    )
    assert len(pan_seq) == 700 + 600 + 200
    assert upd_evs[0].pan_start == 800
    assert upd_evs[0].pan_end == 1400

    res = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq=pan_seq,
        step_offsets=step_offsets,
        sv_events=upd_evs,
        inserts_rows=ins_rows,
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        all_haplotypes=graph.all_haplotypes,
        summary_metrics={
            "chrom": "Chr01",
            "gfa_sha256": graph.gfa_sha256,
            "ref_len": 700,
            "pan_len": len(pan_seq),
            "sv_total_count": 1,
        },
        log_lines=(),
        json_log_entries=(),
        gfa_sha256=graph.gfa_sha256,
        gfa_path=str(inputs.gfa_path),
        ref_fa_path=str(inputs.ref_fa_path),
    )
    write_build_outputs(res, cfg)
    rep = verify_build_outputs(
        prefix=trail_prefix,
        gfa_path=inputs.gfa_path,
        ref_fa_path=inputs.ref_fa_path,
        spacer_n=100,
        raise_on_error=True,
    )
    assert rep.passed


def test_verify_secondary_haplotype_ref_path_and_large_offsets_compact_node_store(
    tmp_path: Path,
) -> None:
    """Verify panflat verify resolves secondary haplotype Ref#2#Chr01 via summary.tsv ref_path and handles >2MB offsets.tsv + >10MB GFA."""
    gb = SyntheticGFAFixtureBuilder()
    s_h1 = make_complex_seq(200, seed=7501)
    s_h2 = make_complex_seq(200, seed=7502)
    gb.add_node("1", s_h1)
    gb.add_node("2", s_h2)
    gb.add_p_path("Ref#1#Chr01", [("1", "+")])
    gb.add_p_path("Ref#2#Chr01", [("2", "+")])
    gfa_p = gb.write_gfa(tmp_path / "sec_hap.gfa")
    fa_p = gb.write_fasta(
        tmp_path / "sec_hap.fa",
        {"Ref#1#Chr01": s_h1, "Ref#2#Chr01": s_h2},
    )
    out_p = tmp_path / "sec_hap_out"

    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa_p),
                "--ref-fa",
                str(fa_p),
                "--backbone-prefix",
                "Ref#2#",
                "--out-prefix",
                str(out_p),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "verify",
                "--prefix",
                str(out_p),
                "--gfa",
                str(gfa_p),
                "--ref-fa",
                str(fa_p),
            ]
        )
        == 0
    )

    # Exercise >10MB GFA + >2MB offsets.tsv (CompactNodeStore._node_lens + _CompactChromOffsets)
    n_steps = 58_000
    unit_seq = "ACGTACGTACGTACGTACGT"
    pad_seq = "ACGT" * 48_000
    large_gfa = tmp_path / "large_compact.gfa"
    large_fa = tmp_path / "large_compact.fa"
    with large_gfa.open("w", encoding="utf-8") as fh:
        fh.write("H\tVN:Z:1.0\n")
        for idx in range(1, n_steps + 1):
            fh.write(f"S\t{idx}\t{unit_seq}\n")
        for idx in range(n_steps + 1, n_steps + 52):
            fh.write(f"S\t{idx}\t{pad_seq}\n")
        for idx in range(1, n_steps):
            fh.write(f"L\t{idx}\t+\t{idx + 1}\t+\t0M\n")
        p_steps = ",".join(f"{idx}+" for idx in range(1, n_steps + 1))
        fh.write(f"P\tRef#1#Chr01\t{p_steps}\t*\n")
        fh.write(f"P\tAlt#1#Chr01\t{p_steps}\t*\n")

    ref_full = unit_seq * n_steps
    with large_fa.open("w", encoding="utf-8") as fh:
        fh.write(f">Chr01\n{ref_full}\n")

    large_out = tmp_path / "large_out"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(large_gfa),
                "--ref-fa",
                str(large_fa),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(large_out),
            ]
        )
        == 0
    )
    assert Path(f"{large_out}.offsets.tsv").stat().st_size > 2_000_000
    assert large_gfa.stat().st_size > 10_000_000
    rep_large = verify_build_outputs(
        prefix=large_out,
        gfa_path=large_gfa,
        ref_fa_path=large_fa,
        raise_on_error=True,
    )
    assert rep_large.passed
    assert rep_large.checked_ref_steps == n_steps


def test_qc_cen_island_whole_ref_hit_stats_merge_aggregation_and_atomic_write_cleanup(
    tmp_path: Path,
) -> None:
    """Verify run_qc CEN_ISLAND whole-reference hit metrics, merge_chromosomes QC aggregation, and atomic temp cleanup."""
    b1 = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_arm = make_complex_seq(600, seed=7601)
    su = make_complex_seq(200, seed=7602)
    scen = make_complex_seq(300, seed=7603)
    sv = make_complex_seq(200, seed=7604)
    isl_hom_arm = mutate_seq_snps(s_arm, [50, 250, 450])
    for nid, seq in [
        ("arm", s_arm),
        ("u", su),
        ("cen", scen),
        ("v", sv),
        ("isl", isl_hom_arm),
    ]:
        b1.add_node(nid, seq)
    b1.set_ref_path("Ref#1#Chr01", ["arm+", "u+", "cen+", "v+"])
    b1.add_path("Alt1#1#Chr01", ["arm+", "u+", "isl+", "v+"])
    b1.add_bed_region(800, 1100, "CEN1")
    in1 = b1.write("qc_c1")

    _, bun1 = run_build(
        in1,
        extra_args=[
            "--centromere-mode",
            "extract-islands",
            "--centromere-min-island",
            "500",
            "--qc",
        ],
    )
    assert bun1 is not None
    assert bun1.summary_kv["qc_cen_island_ge150_count"] == "1"
    assert bun1.summary_kv["qc_cen_island_ge150_bp"] == "600"
    assert int(bun1.summary_kv["qc_cen_island_ref_hit_0.98_bp"]) == 600

    b2 = SyntheticInputBuilder(tmp_path, chrom="Chr02", backbone_prefix="Ref#1#")
    b2.add_node("r1", make_complex_seq(200, seed=7611))
    b2.add_node("i80", make_complex_seq(80, seed=7612))
    b2.add_node("r2", make_complex_seq(200, seed=7613))
    b2.set_ref_path("Ref#1#Chr02", ["r1+", "r2+"])
    b2.add_path("Alt1#1#Chr02", ["r1+", "i80+", "r2+"])
    in2 = b2.write("qc_c2")
    _, bun2 = run_build(in2, extra_args=["--qc"])
    assert bun2 is not None

    merged_prefix = tmp_path / "merged_qc"
    merge_chromosomes([in1.out_prefix, in2.out_prefix], merged_prefix)
    merged_bundle = OutputBundle(merged_prefix)
    assert merged_bundle.summary_kv["qc_enabled"] == "true"
    assert merged_bundle.summary_kv["qc_short_50_149_count"] == "1"
    assert merged_bundle.summary_kv["qc_short_50_149_bp"] == "80"
    assert merged_bundle.summary_kv["qc_cen_island_ge150_count"] == "1"
    assert merged_bundle.summary_kv["qc_cen_island_ref_hit_0.98_bp"] == "600"

    target_f = tmp_path / "atomic_target.tsv"
    target_f.write_text("ORIGINAL\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="simulated crash"):
        with _atomic_open_text(target_f) as fh:
            fh.write("PARTIAL_CORRUPT")
            raise RuntimeError("simulated crash")
    assert target_f.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert not list(tmp_path.glob(".atomic_target.tsv.tmp.*"))


def test_gff_liftover_lazy_step_offsets_splice_proximity_and_repeat_mask_fast_path(
    tmp_path: Path,
) -> None:
    """Verify liftover_gff3 with LazyStepOffsetSequence + bisect splice proximity and repeat_mask_intervals window accuracy."""
    lo = LazyStepOffsetSequence(
        chrom="Chr01",
        ref_nids=np.array([1, 2, 3], dtype=np.int32),
        ref_starts=np.array([0, 100, 200], dtype=np.int64),
        ref_ends=np.array([100, 200, 300], dtype=np.int64),
        pan_starts=np.array([0, 250, 350], dtype=np.int64),
        pan_ends=np.array([100, 350, 450], dtype=np.int64),
        slot_ins_lens=np.array([0, 150, 0], dtype=np.int64),
    )
    gff_in = tmp_path / "genes.gff3"
    gff_in.write_text(
        "##gff-version 3\n"
        "Chr01\tsrc\tmRNA\t1\t300\t.\t+\t.\tID=tx1\n"
        "Chr01\tsrc\texon\t10\t95\t.\t+\t.\tID=ex1;Parent=tx1\n"
        "Chr01\tsrc\texon\t120\t280\t.\t+\t.\tID=ex2;Parent=tx1\n",
        encoding="utf-8",
    )
    ins_ev = SVEvent(
        chrom="Chr01",
        pan_start=100,
        pan_end=250,
        event_id="Chr01_SV_1",
        event_type="INS",
        ref_start=100,
        ref_end=100,
        anchors="0..1",
        alleles_str="1:len=150,src=Alt#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id=".",
        genotypes={"Ref#1": "0", "Alt#1": "1"},
    )
    lifted = liftover_gff3(
        gff_path=gff_in,
        ref_chrom="Chr01",
        step_offsets=lo,
        insertion_events=[ins_ev],
        splice_window=8,
    )
    mrna_line = next(ln for ln in lifted if "\tmRNA\t" in ln)
    assert "sv_status=intronic_insertion" in mrna_line
    assert "splice_site_proximal_sv=True" in mrna_line

    # Verify vectorized rolling 15-mer window masking matches independent canonical_15mer_stats oracle
    mixed_seq = (
        make_complex_seq(700, seed=7701)
        + make_satellite_seq(unit_len=20, total_len=1600, seed=7702)
        + ("N" * 150)
        + make_complex_seq(750, seed=7703)
    )
    masked_ivs, diags = repeat_mask_intervals(mixed_seq)
    assert len(masked_ivs) >= 1
    for d in diags:
        ws, we = int(d["start"]), int(d["end"])
        _, _, _, oracle_sr = canonical_15mer_stats(mixed_seq[ws:we])
        if oracle_sr is None:
            assert d["state"] == "unknown"
            assert d["singleton_ratio"] is None
        else:
            assert d["singleton_ratio"] == pytest.approx(oracle_sr, abs=1e-6)
            assert d["state"] == ("low" if oracle_sr < 0.30 else "non-low")


def test_merge_cross_gfa_same_src_path_qualification_and_exported_fasta_id_remapping(
    tmp_path: Path,
) -> None:
    """Verify merge_chromosomes qualifies shared non-PanSN src_path as chrom::src_path and remaps colliding exported_fasta_id."""
    un_seqs: dict[str, str] = {}
    for idx, chrom in enumerate(["Chr01", "Chr02"], start=1):
        s1 = make_complex_seq(120, seed=7800 + idx)
        s2 = make_complex_seq(120, seed=7810 + idx)
        un = make_complex_seq(120, seed=7820 + idx)
        un_seqs[chrom] = un
        gfa = tmp_path / f"{chrom}.gfa"
        gfa.write_text(
            "\n".join(
                [
                    "H\tVN:Z:1.0",
                    f"S\t1\t{s1}",
                    f"S\t2\t{s2}",
                    f"S\t3\t{un}",
                    "L\t3\t+\t1\t+\t0M",
                    "L\t1\t+\t2\t+\t0M",
                    f"P\tRef#1#{chrom}\t1+,2+\t*",
                    "P\tscaf_unplaced_1\t3+,1+,2+\t*",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        fa = tmp_path / f"{chrom}.fa"
        fa.write_text(f">{chrom}\n{s1 + s2}\n", encoding="utf-8")
        assert (
            cli_main(
                [
                    "build",
                    "--gfa",
                    str(gfa),
                    "--ref-fa",
                    str(fa),
                    "--backbone-prefix",
                    "Ref#1#",
                    "--out-prefix",
                    str(tmp_path / chrom),
                ]
            )
            == 0
        )

    # Also rewrite exported_fasta_id to literal 'UNRES_1' in both prefixes to exercise collision remapping directly
    for chrom in ("Chr01", "Chr02"):
        ut = (tmp_path / f"{chrom}.unresolved.tsv").read_text(encoding="utf-8")
        (tmp_path / f"{chrom}.unresolved.tsv").write_text(
            ut.replace(f"{chrom}_UNRES_1", "UNRES_1"), encoding="utf-8"
        )
        ufa = (tmp_path / f"{chrom}.unresolved.fa").read_text(encoding="utf-8")
        (tmp_path / f"{chrom}.unresolved.fa").write_text(
            ufa.replace(f">{chrom}_UNRES_1", ">UNRES_1"), encoding="utf-8"
        )

    merged_p = tmp_path / "merged_same_src"
    merge_chromosomes([tmp_path / "Chr01", tmp_path / "Chr02"], merged_p)
    bundle = OutputBundle(merged_p)
    assert [r["src_path"] for r in bundle.unresolved] == [
        "Chr01::scaf_unplaced_1",
        "Chr02::scaf_unplaced_1",
    ]
    exp_ids = [r["exported_fasta_id"] for r in bundle.unresolved]
    assert exp_ids == ["Chr01_UNRES_1", "Chr02_UNRES_1"]
    assert bundle.unresolved_fa[exp_ids[0]][1] == un_seqs["Chr01"]
    assert bundle.unresolved_fa[exp_ids[1]][1] == un_seqs["Chr02"]


def test_qc_2stage_self_masking_detects_91pct_homolog_without_pri_ratio_suppression(
    tmp_path: Path,
) -> None:
    """Verify run_qc 2-stage self-masking detects ~91%-identity homologs (24 SNPs / 300 bp = 92.0%) despite minimap2 pri_ratio=0.80."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_ref1 = make_complex_seq(300, seed=7901)
    s_ref2 = make_complex_seq(300, seed=7902)
    s_ref3 = make_complex_seq(300, seed=7903)
    # 24 evenly spaced SNPs over 300 bp -> 276/300 = 92.0% identity (asm20 score = 276 - 96 = 180 < 0.80 * 300 = 240!)
    snp_positions = [10 + 11 * k for k in range(24)]
    ins_92_vs_ref = mutate_seq_snps(s_ref1, snp_positions)

    # Piece-to-piece 92.0% homolog pair mounted at two separate slots
    p_base = make_complex_seq(300, seed=7904)
    p_hom = mutate_seq_snps(p_base, snp_positions)

    for nid, seq in [
        ("r1", s_ref1),
        ("r2", s_ref2),
        ("r3", s_ref3),
        ("iref", ins_92_vs_ref),
        ("p1", p_base),
        ("p2", p_hom),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+"])
    b.add_path("Hap1#1#Chr01", ["r1+", "iref+", "r2+", "p1+", "r3+"])
    b.add_path("Hap2#1#Chr01", ["r1+", "p2+", "r2+", "r3+"])
    inputs = b.write("qc_pri_ratio")

    _, bundle = run_build(inputs, extra_args=["--qc"])
    assert bundle is not None
    assert bundle.summary_kv["qc_long_ge150_count"] == "3"
    assert bundle.summary_kv["qc_long_ge150_bp"] == "900"
    assert int(bundle.summary_kv["qc_redundancy_0.90_bp"]) == 900
    assert int(bundle.summary_kv["qc_redundancy_0.95_bp"]) == 0


def test_multiparent_gff3_isoforms_and_sample_hap0_fallback_and_merged_pansn_verify(
    tmp_path: Path,
) -> None:
    """Verify multi-Parent GFF3 isoform phase check, Sample#0# multi-FASTA fallback, and merged PanSN --ref-fa verify."""
    gb = SyntheticGFAFixtureBuilder()
    orf1 = "ATG" + ("GCT" * 29)  # 90 bp
    orf2 = ("TTC" * 29) + "TAA"  # 90 bp
    ins_seq = make_complex_seq(120, seed=7951)
    gb.add_node("1", orf1)
    gb.add_node("2", orf2)
    gb.add_node("3", ins_seq)
    gb.add_p_path("SampleA#1#Chr01", [("1", "+"), ("2", "+")])
    gb.add_p_path("Alt#1#Chr01", [("1", "+"), ("3", "+"), ("2", "+")])
    gfa1 = gb.write_gfa(tmp_path / "mp1.gfa")

    gb2 = SyntheticGFAFixtureBuilder()
    s_c2 = make_complex_seq(200, seed=7952)
    gb2.add_node("1", s_c2)
    gb2.add_p_path("SampleA#1#Chr02", [("1", "+")])
    gb2.add_p_path("Alt#1#Chr02", [("1", "+")])
    gfa2 = gb2.write_gfa(tmp_path / "mp2.gfa")

    multi_fa = gb.write_fasta(
        tmp_path / "pansn_all.fa",
        {
            "SampleA#1#Chr01": orf1 + orf2,
            "SampleA#1#Chr02": s_c2,
            "Other#1#Chr01": ins_seq + orf2,
        },
    )
    gff_mp = tmp_path / "mp.gff3"
    gff_mp.write_text(
        "##gff-version 3\n"
        "Chr01\tsrc\tgene\t1\t180\t.\t+\t.\tID=g1\n"
        "Chr01\tsrc\tmRNA\t1\t180\t.\t+\t.\tID=m1;Parent=g1\n"
        "Chr01\tsrc\tmRNA\t1\t180\t.\t+\t.\tID=m2;Parent=g1\n"
        "Chr01\tsrc\texon\t1\t180\t.\t+\t.\tID=e1;Parent=m1,m2\n"
        "Chr01\tsrc\tCDS\t1\t180\t.\t+\t0\tID=c1;Parent=m1,m2\n",
        encoding="utf-8",
    )

    out_c1 = tmp_path / "mp_chr01"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa1),
                "--ref-fa",
                str(multi_fa),
                "--backbone-prefix",
                "SampleA#0#",
                "--ref-gff",
                str(gff_mp),
                "--out-prefix",
                str(out_c1),
            ]
        )
        == 0
    )

    out_c2 = tmp_path / "mp_chr02"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa2),
                "--ref-fa",
                str(multi_fa),
                "--backbone-prefix",
                "SampleA#0#",
                "--out-prefix",
                str(out_c2),
            ]
        )
        == 0
    )

    merged_mp = tmp_path / "mp_merged"
    merge_chromosomes([out_c1, out_c2], merged_mp)
    comb_gfa = tmp_path / "comb.gfa"
    comb_gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{orf1}",
                f"S\t2\t{orf2}",
                f"S\t3\t{ins_seq}",
                f"S\t4\t{s_c2}",
                "L\t1\t+\t2\t+\t0M",
                "L\t1\t+\t3\t+\t0M",
                "L\t3\t+\t2\t+\t0M",
                "P\tSampleA#1#Chr01\t1+,2+\t*",
                "P\tAlt#1#Chr01\t1+,3+,2+\t*",
                "P\tSampleA#1#Chr02\t4+\t*",
                "P\tAlt#1#Chr02\t4+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    off_txt = Path(f"{merged_mp}.offsets.tsv").read_text(encoding="utf-8").splitlines()
    fixed_off = [
        "\t".join(["Chr02", "0", "4", *ln.split("\t")[3:]])
        if ln.startswith("Chr02\t")
        else ln
        for ln in off_txt
    ]
    Path(f"{merged_mp}.offsets.tsv").write_text("\n".join(fixed_off) + "\n", encoding="utf-8")

    rep = verify_build_outputs(
        prefix=merged_mp,
        gfa_path=comb_gfa,
        ref_fa_path=multi_fa,
        ref_gff_path=gff_mp,
        raise_on_error=True,
    )
    assert rep.passed


# ---------------------------------------------------------------------------
# 14. W-Walk Integer-GFA Full SV Spectrum, Re-Anchoring & Centromere Modes
# ---------------------------------------------------------------------------


def test_m7_14_w_walk_int_gfa_full_sv_spectrum_reanchor_and_centromere(
    tmp_path: Path,
) -> None:
    """Item 1: Pure W-line integer GFA (`use_fast_int_paths=True`) exercising `DEL`,
    `CNV`, `DUP_TRA`, `LARGE_JUMP`, local re-anchoring, and centromere `extract-islands`
    & `bypass` without dropping empty `LazyStepTuple` deviations.
    """
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 20):
        gb.add_node(str(i), make_complex_seq(200, seed=9100 + i))
    gb.add_node("20", make_complex_seq(600, seed=9120))  # centromere island >= 500 bp

    # Reference walk (all W-lines, integer node IDs 1..20):
    # 1(0), 2(1:cen_u), 3(2:cen_mid), 4(3:cen_v),
    # 5(4:multi-copy m), 6(5:ins target), 5(6:multi-copy m),
    # 7(7:DEL target), 8(8),
    # 9(9:tr), 9(10:tr), 10(11),
    # 11(12), 12(13), 13(14), 14(15), 15(16), 16(17)
    ref_steps = [
        ("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"),
        ("5", "+"), ("6", "+"), ("5", "+"),
        ("7", "+"), ("8", "+"),
        ("9", "+"), ("9", "+"), ("10", "+"),
        ("11", "+"), ("12", "+"), ("13", "+"), ("14", "+"), ("15", "+"), ("16", "+"),
    ]
    gb.add_w_walk("Ref", "1", "Chr01", 0, None, ref_steps)

    # Alt1 (W-line):
    # - centromere island 20+ between 2+ and 4+
    # - local re-anchoring across multi-copy 5+ with insertion 17+ before 6+
    # - pure W-walk DEL of 7+ (empty LazyStepTuple branch between 5+ and 8+!)
    # - CNV expansion of 9+ from 2 to 4 copies
    # - DUP_TRA of distant right-arm ref node 8+ between 14+ and 15+ (distance > max_jump=500)
    alt1_steps = [
        ("1", "+"), ("2", "+"), ("3", "+"), ("20", "+"), ("4", "+"),
        ("5", "+"), ("17", "+"), ("6", "+"), ("5", "+"),
        ("8", "+"),
        ("9", "+"), ("9", "+"), ("9", "+"), ("9", "+"), ("10", "+"),
        ("11", "+"), ("12", "+"), ("13", "+"), ("14", "+"), ("8", "+"), ("15", "+"), ("16", "+"),
    ]
    gb.add_w_walk("Alt1", "1", "ctgA", 1000, None, alt1_steps)

    # Alt2 (W-line): LARGE_JUMP skipping 12+, 13+, 14+ (600 bp > max_jump=500) carrying 18+
    alt2_steps = [
        ("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"),
        ("5", "+"), ("6", "+"), ("5", "+"),
        ("7", "+"), ("8", "+"),
        ("9", "+"), ("9", "+"), ("10", "+"),
        ("11", "+"), ("18", "+"), ("15", "+"), ("16", "+"),
    ]
    gb.add_w_walk("Alt2", "1", "ctgB", 2000, None, alt2_steps)

    gfa = gb.write_gfa(tmp_path / "w_all.gfa")
    ref_seq = "".join(gb.nodes[nid] for nid, _ in ref_steps)
    ref_fa = gb.write_fasta(tmp_path / "w_ref.fa", {"Chr01": ref_seq})
    bed = tmp_path / "w_cen.bed"
    bed.write_text("Chr01\t400\t600\tCEN_W\n", encoding="utf-8")

    for mode in ("extract-islands", "bypass"):
        out_p = tmp_path / f"w_{mode}"
        cfg = BuildConfig(
            gfa=str(gfa),
            ref_fa=str(ref_fa),
            backbone_prefix="Ref#1#",
            out_prefix=str(out_p),
            centromere_bed=str(bed),
            centromere_mode=mode,
            centromere_min_island=500,
            centromere_mode_explicit=True,
            centromere_min_island_explicit=(mode == "extract-islands"),
            max_jump=500,
        )
        validate_build_config(cfg)
        graph = load_and_validate_inputs(cfg)
        assert isinstance(graph.ref_fragment.steps, LazyStepTuple)
        res = run_chromosome_pipeline(graph, cfg)
        write_build_outputs(res, cfg)
        rep = verify_build_outputs(
            prefix=out_p,
            gfa_path=gfa,
            ref_fa_path=ref_fa,
            centromere_bed_path=bed,
            raise_on_error=True,
        )
        assert rep.passed

        ev_types = {ev.event_type for ev in res.sv_events}
        assert {"INS", "DEL", "CNV", "DUP_TRA", "LARGE_JUMP"}.issubset(ev_types)
        if mode == "extract-islands":
            assert "CEN_ISLAND" in ev_types
        else:
            assert "CEN_ISLAND" not in ev_types
            assert any(u.reason == "CEN_BYPASS" for u in res.unresolved_records)


# ---------------------------------------------------------------------------
# 15. PanSN Backbone Prefix Matrix & Explicit Verify Override Precedence
# ---------------------------------------------------------------------------


def test_m7_15_backbone_prefix_boundary_matrix_and_verify_explicit_override(
    tmp_path: Path,
) -> None:
    """Item 2 & Decision #3:
    - `--backbone-prefix SampleA` rejects `SampleA2#1#Chr01` / `SampleAB#1#Chr01` false matches.
    - `--backbone-prefix SampleA#1` rejects `SampleA#10#Chr01` false haplotype match.
    - `--backbone-prefix SampleA#0#` falls back to `SampleA#1#Chr01` when `#0#` is absent.
    - Standalone `verify_build_outputs(..., backbone_prefix=...)` honors explicit caller override:
      succeeding on matching prefix (`SampleA#1#` / `SampleA#0#`) and FAILING when caller passes
      a wrong `--backbone-prefix` (`WrongSample#` or `SampleB#1#`), even though `summary.tsv`
      recorded `ref_path=SampleA#1#Chr10A`.
    """
    gb = SyntheticGFAFixtureBuilder()
    s_a = make_complex_seq(180, seed=9201)
    s_b = make_complex_seq(180, seed=9202)
    s_a10 = make_complex_seq(180, seed=9203)
    gb.add_node("1", s_a)
    gb.add_node("2", s_b)
    gb.add_node("3", s_a10)
    gb.add_p_path("SampleA#1#Chr10A", [("1", "+")])
    gb.add_p_path("SampleA2#1#Chr10A", [("2", "+")])
    gb.add_p_path("SampleAB#1#Chr10A", [("2", "+")])
    gb.add_p_path("SampleA#10#Chr10A", [("3", "+")])
    gb.add_p_path("SampleB#1#Chr10A", [("2", "+")])
    gfa = gb.write_gfa(tmp_path / "bp.gfa")

    multi_fa = gb.write_fasta(
        tmp_path / "bp_multi.fa",
        {
            "SampleB#1#Chr10A": s_b,
            "SampleA2#1#Chr10A": s_b,
            "SampleA#10#Chr10A": s_a10,
            "SampleA#1#Chr10A": s_a,
            "SampleA#1#Chr10B": s_b,
        },
    )

    # SampleA, SampleA#1, and SampleA#0# all uniquely select SampleA#1#Chr10A
    for ok_bp in ("SampleA", "SampleA#1", "SampleA#0#", "SampleA#1#"):
        g = load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa),
                ref_fa=str(multi_fa),
                backbone_prefix=ok_bp,
                out_prefix=str(tmp_path / f"ok_{ok_bp.replace('#', '_')}"),
            )
        )
        assert g.ref_fragment.src_path == "SampleA#1#Chr10A"
        assert g.ref_chrom == "Chr10A"
        assert g.ref_seq == s_a

    # Prefix with no matching sample raises InputContractError
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa),
                ref_fa=str(multi_fa),
                backbone_prefix="Sample",
                out_prefix=str(tmp_path / "bad_sample"),
            )
        )

    # Unmatched multi-record plain FASTA (>Chr10A + >Chr10B without '#') raises InputContractError
    plain_multi_fa = gb.write_fasta(
        tmp_path / "plain_multi.fa",
        {"Chr10A": s_a, "Chr10B": s_b},
    )
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa),
                ref_fa=str(plain_multi_fa),
                backbone_prefix="SampleA#1#",
                out_prefix=str(tmp_path / "bad_plain_multi"),
            )
        )

    # Full build with SampleA#1#, then test verify_build_outputs with positive & negative --backbone-prefix
    out_ver = tmp_path / "ver_bp"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa),
                "--ref-fa",
                str(multi_fa),
                "--backbone-prefix",
                "SampleA#1#",
                "--out-prefix",
                str(out_ver),
            ]
        )
        == 0
    )

    # Positive verify: omitted backbone_prefix, exact SampleA#1#, and fallback SampleA#0#
    assert verify_build_outputs(out_ver, gfa, multi_fa).passed
    assert verify_build_outputs(out_ver, gfa, multi_fa, backbone_prefix="SampleA#1#").passed
    assert verify_build_outputs(out_ver, gfa, multi_fa, backbone_prefix="SampleA#0#").passed

    # Negative verify: explicit wrong --backbone-prefix MUST fail verification!
    rep_wrong_nonexistent = verify_build_outputs(
        out_ver, gfa, multi_fa, backbone_prefix="WrongSample#"
    )
    assert not rep_wrong_nonexistent.passed
    assert cli_main(
        [
            "verify",
            "--prefix",
            str(out_ver),
            "--gfa",
            str(gfa),
            "--ref-fa",
            str(multi_fa),
            "--backbone-prefix",
            "WrongSample#",
        ]
    ) != 0

    rep_wrong_other_hap = verify_build_outputs(
        out_ver, gfa, multi_fa, backbone_prefix="SampleB#1#"
    )
    assert not rep_wrong_other_hap.passed
    assert cli_main(
        [
            "verify",
            "--prefix",
            str(out_ver),
            "--gfa",
            str(gfa),
            "--ref-fa",
            str(multi_fa),
            "--backbone-prefix",
            "SampleB#1#",
        ]
    ) != 0


# ---------------------------------------------------------------------------
# 16. Compact Container & Sequence Protocols (Full Edge Cases)
# ---------------------------------------------------------------------------


def test_m7_16_compact_containers_full_protocol_and_noncanonical_nids() -> None:
    """Item 3: Full protocol verification for `CompactLinkSet` (self-complementing link
    & >100k fast index), `CompactVisitCounts` (dense bitset & sparse `visited_nids`),
    `CompactRefNodeSet`, `CompactUniqueRefMap`, `LazyStepOffsetSequence`,
    `LazyRefStepRanges`, `LazyProjectedAnchorSequence`, and `LazyProvenanceTuple`.
    """
    from panflat.models import (
        CompactLinkSet,
        CompactRefNodeSet,
        CompactUniqueRefMap,
        CompactVisitCounts,
        LazyProjectedAnchorSequence,
        LazyProvenanceTuple,
        LazyRefStepRanges,
        LazyStepOffsetSequence,
        ProjectedAnchor,
        StepOffsetRow,
        StepProvenance,
    )
    from panflat.normalizer import compute_raw_visit_counts

    # 1. CompactLinkSet: self-reversing link (1, '+', 1, '-') vs normal link (2, '+', 2, '+')
    c_self = CompactLinkSet.encode_canon(1, 0, 1, 1)
    c_pair = CompactLinkSet.encode_canon(2, 0, 2, 0)
    ls = CompactLinkSet(
        np.unique(np.array([c_self, c_pair], dtype=np.uint64)),
        extra={("01", "+", "1", "+")},
    )
    assert len(ls) == len(list(ls)) == 4
    assert ("1", "+", "1", "-") in ls
    assert ("01", "+", "1", "+") in ls
    assert ("01", "+", "2", "+") not in ls
    assert ("+1", "+", "1", "-") not in ls

    # CompactLinkSet >100k links fast index `_first_v` path in validate_step_codes
    n_big = 100_050
    u_arr = np.arange(1, n_big + 1, dtype=np.uint64)
    v_arr = u_arr + np.uint64(1)
    big_canon = np.sort(((u_arr << np.uint64(1)) << np.uint64(32)) | (v_arr << np.uint64(1)))
    ls_big = CompactLinkSet(big_canon)
    assert ls_big.validate_step_codes(np.array([1, 2, 3, 4], dtype=np.int32)) == -1
    assert ls_big.validate_step_codes(np.array([1, 2, 99], dtype=np.int32)) == 1

    # 2. CompactVisitCounts: both dense bitset mode and sparse visited_nids mode (>50k steps)
    codes_dense = np.ones(50_010, dtype=np.int32)
    codes_dense[0] = 2
    codes_dense[1] = -3
    vc_dense = compute_raw_visit_counts(LazyStepTuple(codes_dense))
    assert isinstance(vc_dense, CompactVisitCounts)
    assert vc_dense["1"] == 50_008 and vc_dense["2"] == 1 and vc_dense["3"] == 1
    assert vc_dense.get("999", 0) == 0 and vc_dense.get("01", 0) == 0
    with pytest.raises(KeyError):
        _ = vc_dense["999"]
    assert len(vc_dense) == 3 and set(vc_dense) == {"1", "2", "3"}

    codes_sparse = np.ones(50_010, dtype=np.int32)
    codes_sparse[0] = 500_000
    vc_sparse = compute_raw_visit_counts(LazyStepTuple(codes_sparse))
    assert isinstance(vc_sparse, CompactVisitCounts)
    assert vc_sparse["1"] == 50_009 and vc_sparse["500000"] == 1
    assert "500000" in vc_sparse and "2" not in vc_sparse and "01" not in vc_sparse
    with pytest.raises(KeyError):
        _ = vc_sparse["2"]
    assert len(vc_sparse) == 2 and set(vc_sparse) == {"1", "500000"}

    # 3. CompactRefNodeSet & CompactUniqueRefMap non-canonical key isolation
    mask = np.array([False, True, False, True], dtype=bool)
    rset = CompactRefNodeSet(mask, extra=frozenset({"01"}))
    assert "1" in rset and "3" in rset and "01" in rset
    assert "03" not in rset and "+1" not in rset and " 1" not in rset and 1 not in rset
    assert len(rset) == 3 and set(rset) == {"1", "3", "01"}

    umap = CompactUniqueRefMap(np.array([-1, 0, -1, 1], dtype=np.int32), extra={"01": 2})
    assert umap["1"] == 0 and umap["3"] == 1 and umap["01"] == 2
    assert "03" not in umap and "+1" not in umap and "999999999999999999999999" not in umap
    with pytest.raises(KeyError):
        _ = umap["03"]
    assert len(umap) == 3 and set(umap) == {"1", "3", "01"}

    # 4. LazyStepOffsetSequence, LazyRefStepRanges, LazyProjectedAnchorSequence, LazyProvenanceTuple
    so_seq = LazyStepOffsetSequence(
        chrom="Chr01",
        ref_nids=np.array([10, 20], dtype=np.int32),
        ref_starts=np.array([0, 100], dtype=np.int64),
        ref_ends=np.array([100, 250], dtype=np.int64),
        pan_starts=np.array([0, 150], dtype=np.int64),
        pan_ends=np.array([100, 300], dtype=np.int64),
        slot_ins_lens=np.array([0, 50], dtype=np.int64),
    )
    assert so_seq[-1] == StepOffsetRow("Chr01", 1, "20", 100, 250, 150, 300, 50)
    assert len(so_seq[0:2]) == 2
    with pytest.raises(IndexError):
        _ = so_seq[2]

    rr = LazyRefStepRanges(np.array([0, 100, 250], dtype=np.int64))
    assert rr[-1] == (100, 250) and rr == [(0, 100), (100, 250)]
    with pytest.raises(IndexError):
        _ = rr[-3]

    st_seq = LazyStepTuple(np.array([1, -2], dtype=np.int32))
    assert st_seq.spell({"1": "acgt", "2": "aaac"}) == "acgtgttt"
    pv_seq = LazyProvenanceTuple(
        src_hap="H#1",
        src_path="H#1#Chr01",
        base_offset=10,
        orig_codes=np.array([1, -2], dtype=np.int32),
        node_lens=np.array([0, 4, 4], dtype=np.int32),
    )
    assert pv_seq == list(pv_seq)
    assert pv_seq[-1] == StepProvenance("H#1", "H#1#Chr01", 14, 18, "+")

    pa_seq = LazyProjectedAnchorSequence(
        branch_step_idx=np.array([0, 1], dtype=np.int32),
        rank_in_b=np.array([0, 1], dtype=np.int32),
        rel_sign=np.array([1, -1], dtype=np.int8),
        weights=np.array([4, 4], dtype=np.int64),
        steps=st_seq,
        provenance=pv_seq,
    )
    assert pa_seq[-1] == ProjectedAnchor(1, 1, "-", "2", 4, Step("2", "-"), pv_seq[1])
    assert pa_seq == list(pa_seq)


# ---------------------------------------------------------------------------
# 17. `--qc` Multi-Piece Dyadic Halving & CEN_ISLAND Whole-Ref Homolog Hits
# ---------------------------------------------------------------------------


def test_m7_17_qc_self_hit_masking_dyadic_multi_piece_and_cen_island_ref_hits(
    tmp_path: Path,
) -> None:
    """Item 4:
    - Three ~91.5% identical `1000 bp` inserted pieces at distinct loci (`n_long = 3` + 1
      `CEN_ISLAND` -> `n_long = 4`, exercising multi-bit dyadic halving): 100% self-hits in
      `pan.fa` are masked so all ~91.5% piece-to-piece homologs are detected at `tau=0.90`.
    - `CEN_ISLAND` piece with ~92% homology to a distant non-centromeric reference arm is
      reported in `qc_cen_island_ref_hit_0.90_bp` (`== 1000`) while `0.95` is `0`.
    """
    gb = SyntheticGFAFixtureBuilder()
    s_arm_hom = make_complex_seq(1000, seed=9400)
    gb.add_node("1", s_arm_hom)  # Reference arm carries s_arm_hom outside centromere!
    gb.add_node("2", make_complex_seq(300, seed=9401))  # cen u
    gb.add_node("3", make_complex_seq(300, seed=9402))  # cen inner
    gb.add_node("4", make_complex_seq(300, seed=9403))  # cen v
    gb.add_node("5", make_complex_seq(300, seed=9404))
    gb.add_node("6", make_complex_seq(300, seed=9405))
    gb.add_node("7", make_complex_seq(300, seed=9406))

    import random as _rnd

    base_p = make_complex_seq(1000, seed=9410)
    p1 = base_p
    p2 = mutate_seq_snps(base_p, sorted(_rnd.Random(9411).sample(range(30, 970), 75)))  # 92.5% to p1
    p3 = mutate_seq_snps(base_p, sorted(_rnd.Random(9412).sample(range(30, 970), 75)))  # 92.5% to p1
    cen_isl = mutate_seq_snps(s_arm_hom, sorted(_rnd.Random(9413).sample(range(30, 970), 75)))  # 92.5% to ref node 1!

    gb.add_node("10", p1)
    gb.add_node("11", p2)
    gb.add_node("12", p3)
    gb.add_node("13", cen_isl)

    gb.add_p_path("Ref#1#Chr01", [(str(i), "+") for i in range(1, 8)])
    gb.add_p_path(
        "Alt#1#Chr01",
        [
            ("1", "+"),
            ("2", "+"), ("3", "+"), ("13", "+"), ("4", "+"),
            ("10", "+"), ("5", "+"),
            ("11", "+"), ("6", "+"),
            ("12", "+"), ("7", "+"),
        ],
    )
    gfa = gb.write_gfa(tmp_path / "qc4.gfa")
    ref_fa = gb.write_fasta(
        tmp_path / "qc4.fa",
        {"Chr01": "".join(gb.nodes[str(i)] for i in range(1, 8))},
    )
    bed = tmp_path / "qc4.bed"
    bed.write_text("Chr01\t1300\t1600\tCEN_QC\n", encoding="utf-8")

    cfg = BuildConfig(
        gfa=str(gfa),
        ref_fa=str(ref_fa),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "qc4_out"),
        centromere_bed=str(bed),
        centromere_mode="extract-islands",
        centromere_min_island=500,
        centromere_mode_explicit=True,
        centromere_min_island_explicit=True,
        qc=True,
    )
    validate_build_config(cfg)
    graph = load_and_validate_inputs(cfg)
    res = run_chromosome_pipeline(graph, cfg)
    qc_m = run_qc(res, cfg)

    # All 4 pieces (p1, p2, p3 via piece-to-piece dyadic halving, cen_isl via ref arm node 1) hit at >=0.90!
    assert qc_m["qc_long_ge150_count"] == 4
    assert qc_m["qc_long_ge150_bp"] == 4000
    assert qc_m["qc_redundancy_0.90_bp"] == 4000
    assert qc_m["qc_redundancy_0.98_bp"] == 0
    assert qc_m["qc_cen_island_ge150_count"] == 1
    assert qc_m["qc_cen_island_ge150_bp"] == 1000
    assert qc_m["qc_cen_island_ref_hit_0.90_bp"] == 1000
    assert qc_m["qc_cen_island_ref_hit_0.95_bp"] == 0


# ---------------------------------------------------------------------------
# 18. Hierarchical Merge Collisions, QC Aggregation & Atomic Crash Cleanup
# ---------------------------------------------------------------------------


def test_m7_18_hierarchical_merge_collisions_qc_aggregation_and_atomic_write_cleanup(
    tmp_path: Path,
) -> None:
    """Item 6:
    - Hierarchical `merge_chromosomes([merge([chr1, chr2]), chr3])` with cross-GFA colliding
      non-PanSN `src_path`, `exported_fasta_id`, `region_id`, and `event_id`, plus `--qc`
      metric aggregation without duplicating or misattributing `Chr01:*` / `Chr02:*` rows.
    - `_atomic_open_text` and `_atomic_write_text` clean up `.tmp.*` partial files on exception
      while leaving any pre-existing target file intact.
    """
    from panflat.writers import _atomic_open_text

    # 1. Atomic crash cleanup & pre-existing target preservation
    target_f = tmp_path / "atomic_target.tsv"
    target_f.write_text("original_content\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="simulated mid-stream crash"):
        with _atomic_open_text(target_f) as fh:
            fh.write("partial_corrupted_data")
            raise RuntimeError("simulated mid-stream crash")
    assert target_f.read_text(encoding="utf-8") == "original_content\n"
    assert list(tmp_path.glob("*atomic_target.tsv.tmp.*")) == []

    # 2. Build Chr01, Chr02, Chr03 where Chr02 and Chr03 share non-PanSN src_path "scaf_shared"
    # in BOTH inserts.tsv and unresolved.tsv, and all 3 enable --qc and --spacer-n 50.
    prefixes: list[Path] = []
    for idx, chrom in enumerate(["Chr01", "Chr02", "Chr03"], start=1):
        gb = SyntheticGFAFixtureBuilder()
        nbase = idx * 10
        s1 = make_complex_seq(200, seed=9500 + idx)
        s2 = make_complex_seq(200, seed=9510 + idx)
        ins_long = make_complex_seq(180, seed=9520 + idx)
        unanch = make_complex_seq(120, seed=9530 + idx)
        gb.add_node(str(nbase + 1), s1)
        gb.add_node(str(nbase + 2), s2)
        gb.add_node(str(nbase + 3), ins_long)
        gb.add_node(str(nbase + 4), unanch)
        gb.add_p_path(f"SampleA#1#{chrom}", [(str(nbase + 1), "+"), (str(nbase + 2), "+")])
        alt_name = "scaf_c1_only" if chrom == "Chr01" else "scaf_shared"
        gb.add_p_path(
            alt_name,
            [
                (str(nbase + 4), "+"),
                (str(nbase + 1), "+"),
                (str(nbase + 3), "+"),
                (str(nbase + 2), "+"),
            ],
        )
        gfa = gb.write_gfa(tmp_path / f"{chrom}.gfa")
        fa = gb.write_fasta(tmp_path / f"{chrom}.fa", {f"SampleA#1#{chrom}": s1 + s2})
        out_p = tmp_path / f"out_{chrom}"
        assert (
            cli_main(
                [
                    "build",
                    "--gfa",
                    str(gfa),
                    "--ref-fa",
                    str(fa),
                    "--backbone-prefix",
                    "SampleA#1#",
                    "--spacer-n",
                    "50",
                    "--qc",
                    "--out-prefix",
                    str(out_p),
                ]
            )
            == 0
        )
        prefixes.append(out_p)

    # First merge [Chr01, Chr02] -> m12, then hierarchical merge [m12, Chr03] -> m123
    m12 = tmp_path / "m12"
    merge_chromosomes([prefixes[0], prefixes[1]], m12)
    m123 = tmp_path / "m123"
    merge_chromosomes([m12, prefixes[2]], m123)

    ins_lines = Path(f"{m123}.inserts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    ins_chrom_src = [(ln.split("\t")[0], ln.split("\t")[7]) for ln in ins_lines]
    assert ins_chrom_src == [
        ("Chr01", "scaf_c1_only"),
        ("Chr02", "Chr02::scaf_shared"),
        ("Chr03", "Chr03::scaf_shared"),
    ]

    unres_lines = Path(f"{m123}.unresolved.tsv").read_text(encoding="utf-8").splitlines()[1:]
    unres_rec_src = [(ln.split("\t")[0], ln.split("\t")[3]) for ln in unres_lines]
    assert unres_rec_src == [
        ("Chr01_UNRES_1", "scaf_c1_only"),
        ("Chr02_UNRES_1", "Chr02::scaf_shared"),
        ("Chr03_UNRES_1", "Chr03::scaf_shared"),
    ]

    # Summary metrics must have unique keys (no duplicate/corrupted Chr01:qc_* from m12's top-level aggregates)
    sum_rows = [
        ln.split("\t")
        for ln in Path(f"{m123}.summary.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if ln.strip()
    ]
    sum_keys = [r[0] for r in sum_rows]
    assert len(sum_keys) == len(set(sum_keys)), f"Duplicate keys in hierarchical summary.tsv: {sum_keys}"
    sum_dict = {r[0]: r[1] for r in sum_rows}
    assert sum_dict["chrom_count"] == "3"
    assert sum_dict["spacer_n"] == "50"
    assert sum_dict["qc_long_ge150_count"] == "3"
    assert sum_dict["Chr01:qc_long_ge150_count"] == "1"
    assert sum_dict["Chr02:qc_long_ge150_count"] == "1"
    assert sum_dict["Chr03:qc_long_ge150_count"] == "1"


# ---------------------------------------------------------------------------
# 19. Centromere Duplicate Anchor & Exact Repeat-Guard Boundaries
# ---------------------------------------------------------------------------


def test_m7_19_centromere_duplicate_flanking_anchor_and_repeat_guard_exact_boundaries(
    tmp_path: Path,
) -> None:
    """Item 7:
    - Centromere flanking anchor `u` or `v` visited twice inside a donor contig is marked
      `ambiguous`/`partial` (never false `complete` span, `length_ratio == '.'`).
    - Repeat-guard window boundaries (`499`, `500`, `999`, `1000`, `1001 bp`), exact
      `singleton_ratio == 0.30` (`180/600` -> `non-low`, unmasked) vs `179/600 < 0.30`
      (`low`, masked), and `unknown` window (`< 90%` valid 15-mers) breaking adjacent `low` runs.
    """
    from panflat.repeat_guard import repeat_mask_intervals

    # 1. Repeat-guard length boundaries: 499, 500, 999, 1000, 1001 bp
    sat_big = make_satellite_seq(20, 2600, seed=9601)
    assert repeat_mask_intervals(sat_big[:499]) == ([], [])
    assert repeat_mask_intervals(sat_big[:500])[0] == [(0, 500)]
    assert repeat_mask_intervals(sat_big[:999])[0] == [(0, 999)]
    assert repeat_mask_intervals(sat_big[:1000])[0] == [(0, 1000)]
    m_1001, d_1001 = repeat_mask_intervals(sat_big[:1001])
    assert [(w["start"], w["end"]) for w in d_1001] == [(0, 1000), (1, 1001)]
    assert m_1001 == [(0, 1001)]

    # 2. Exact singleton_ratio == 0.30 (180/600) vs 179/600 (< 0.30)
    sat_614 = make_satellite_seq(20, 614, seed=901)
    comp_614 = make_complex_seq(614, seed=902)
    seq_exact_030 = sat_614[:434] + comp_614[434:614]
    seq_below_030 = sat_614[:435] + comp_614[435:614]

    m_eq, d_eq = repeat_mask_intervals(seq_exact_030)
    assert d_eq[0]["valid_positions"] == 600 and d_eq[0]["singleton_count"] == 180
    assert d_eq[0]["singleton_ratio"] == 0.30 and d_eq[0]["state"] == "non-low"
    assert m_eq == []

    m_lt, d_lt = repeat_mask_intervals(seq_below_030)
    assert d_lt[0]["valid_positions"] == 600 and d_lt[0]["singleton_count"] == 179
    assert d_lt[0]["state"] == "low" and m_lt == [(0, 614)]

    # 3. Unknown window (< 90% valid 15-mers) breaks adjacent low runs on 2500 bp sequence
    # Windows: [0,1000) low, [500,1500) unknown, [1000,2000) unknown, [1500,2500) low -> 0 masked intervals!
    seq_unk_break = sat_big[:1000] + ("N" * 200) + sat_big[1200:2500]
    m_unk, d_unk = repeat_mask_intervals(seq_unk_break)
    assert [w["state"] for w in d_unk] == ["low", "unknown", "unknown", "low"]
    assert m_unk == []

    # 4. Centromere flanking node u visited twice vs v visited twice inside donor contigs
    gb = SyntheticGFAFixtureBuilder()
    for nid, seed in [("1", 9610), ("2", 9611), ("3", 9612), ("4", 9613), ("5", 9614)]:
        gb.add_node(nid, make_complex_seq(200, seed=seed))
    gb.add_node("6", make_complex_seq(600, seed=9615))
    gb.add_p_path("Ref#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")])
    # DupU visits anchor u ("2") twice; DupV visits anchor v ("4") twice
    gb.add_p_path(
        "DupU#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("2", "+"), ("6", "+"), ("4", "+"), ("5", "+")],
    )
    gb.add_p_path(
        "DupV#1#Chr01",
        [("1", "+"), ("2", "+"), ("6", "+"), ("4", "+"), ("3", "+"), ("4", "+"), ("5", "+")],
    )
    gfa = gb.write_gfa(tmp_path / "cen_dup_uv.gfa")
    ref_fa = gb.write_fasta(
        tmp_path / "cen_dup_uv.fa",
        {"Chr01": "".join(gb.nodes[str(i)] for i in range(1, 6))},
    )
    bed = tmp_path / "cen_dup_uv.bed"
    bed.write_text("Chr01\t400\t600\tCEN_UV\n", encoding="utf-8")

    cfg = BuildConfig(
        gfa=str(gfa),
        ref_fa=str(ref_fa),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "cen_uv_out"),
        centromere_bed=str(bed),
        centromere_mode="extract-islands",
        centromere_min_island=500,
        centromere_mode_explicit=True,
        centromere_min_island_explicit=True,
    )
    validate_build_config(cfg)
    graph = load_and_validate_inputs(cfg)
    res = run_chromosome_pipeline(graph, cfg)
    write_build_outputs(res, cfg)
    rep = verify_build_outputs(
        cfg.out_prefix, gfa, ref_fa, centromere_bed_path=bed, raise_on_error=True
    )
    assert rep.passed

    cen_by_hap = {r.hap: r for r in res.centromere_rows}
    for h in ("DupU#1", "DupV#1"):
        assert cen_by_hap[h].status in ("ambiguous", "partial")
        assert cen_by_hap[h].length_ratio == "."


# ---------------------------------------------------------------------------
# 20. Malformed FASTA / GFA / Overlapping W-Walk Rejections
# ---------------------------------------------------------------------------


def test_m7_20_malformed_fasta_gfa_and_w_interval_overlap_rejections(
    tmp_path: Path,
) -> None:
    """Item 8: Exhaustive `InputContractError` rejections for malformed FASTA records,
    GFA `LN:i:` mismatch, non-`0M` L/P overlaps, out-of-order integer `S` nodes with
    invalid deferred `L` link, overlapping W-line intervals on `(hap_id, seqid)`, and
    duplicate P+W source intervals.
    """
    from panflat.gfa_io import parse_fasta

    # 1. Malformed FASTA cases
    for idx, bad_fa_txt in enumerate(
        [
            "",  # empty file
            "ACGTACGT\n>Chr01\nACGT\n",  # sequence before '>'
            ">\nACGTACGT\n",  # empty header
            ">Chr01\n>Chr02\nACGT\n",  # empty sequence for Chr01
            ">Chr01\nACGT\n>Chr01\nTGCA\n",  # duplicate header
            ">Chr01\nACGT1234\n",  # non-alphabetic sequence
        ]
    ):
        p = tmp_path / f"bad_{idx}.fa"
        p.write_text(bad_fa_txt, encoding="utf-8")
        with pytest.raises(InputContractError):
            parse_fasta(str(p))

    valid_fa = tmp_path / "valid.fa"
    valid_fa.write_text(">Chr01\nACGTACGT\n", encoding="utf-8")

    # 2. Malformed GFA cases
    bad_gfa_cases = [
        # Mismatched LN:i tag
        "H\tVN:Z:1.0\nS\t1\tACGTACGT\tLN:i:99\nP\tRef#1#Chr01\t1+\t*\n",
        # Non-alphabetic GFA S sequence
        "H\tVN:Z:1.0\nS\t1\tACGT$CGT\nP\tRef#1#Chr01\t1+\t*\n",
        # Non-0M L overlap
        "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tACGT\nL\t1\t+\t2\t+\t4M\nP\tRef#1#Chr01\t1+,2+\t*\n",
        # Non-0M P overlap
        "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tACGT\nL\t1\t+\t2\t+\t0M\nP\tRef#1#Chr01\t1+,2+\t4M\n",
        # Out-of-order integer S nodes (S 3 before S 1) + deferred L link referencing missing node 99
        "H\tVN:Z:1.0\nL\t1\t+\t99\t+\t0M\nS\t3\tACGT\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\n",
        # Overlapping W-line intervals on same (hap_id, seqid): [0, 8) overlaps [4, 12)
        "H\tVN:Z:1.0\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\n"
        "W\tAlt\t1\tctg1\t0\t8\t>1\nW\tAlt\t1\tctg1\t4\t12\t>1\n",
        # Duplicate P and W source interval on Alt#1#Chr01
        "H\tVN:Z:1.0\nS\t1\tACGTACGT\nP\tRef#1#Chr01\t1+\t*\n"
        "P\tAlt#1#Chr01\t1+\t*\nW\tAlt\t1\tChr01\t0\t8\t>1\n",
    ]
    for idx, g_txt in enumerate(bad_gfa_cases):
        gp = tmp_path / f"bad_gfa_{idx}.gfa"
        gp.write_text(g_txt, encoding="utf-8")
        with pytest.raises(InputContractError):
            load_and_validate_inputs(
                BuildConfig(
                    gfa=str(gp),
                    ref_fa=str(valid_fa),
                    backbone_prefix="Ref#1#",
                    out_prefix=str(tmp_path / f"bo_{idx}"),
                )
            )


# ---------------------------------------------------------------------------
# 21. Migrated White-Box Internal API Defensive Contract Guards (M1–M4)
# ---------------------------------------------------------------------------


def test_m7_21_migrated_whitebox_defensive_guards_m1_to_m4(tmp_path: Path) -> None:
    """Preserves all white-box internal Python API contract checks migrated out of
    `tests/e2e/test_tier5_adversarial.py` so `tests/e2e/` maintains 100% zero-`panflat`
    import purity per `TEST_INFRA.md`.
    """
    from panflat.centromere import _extract_nonref_runs
    from panflat.gfa_io import _flip_orient, spell_steps
    from panflat.gff_liftover import liftover_gff3, liftover_gff3_lines
    from panflat.linearizer import (
        DedupCandidate,
        ExtractedCandidate,
        SubproblemStateKey,
        _solve_multi_member_component,
        can_transition,
    )
    from panflat.models import (
        CLIArgumentError,
        GFAGraph,
        PathFragment,
        Piece,
        ProjectedAnchor,
        StepOffsetRow,
        StepProvenance,
        SVEvent,
    )
    from panflat.normalizer import (
        _extract_contiguous_nonref_slices,
        canonical_rotation,
        compute_weighted_lis,
        detect_canonical_cnv,
        extract_deviations,
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

    # 1. Step / StepProvenance / BuildConfig guards
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

    # 2. gfa_io guards
    with pytest.raises(InputContractError):
        _flip_orient("?")
    with pytest.raises(InputContractError):
        spell_steps([Step("missing", "+")], {"n1": "ACGT"})
    with pytest.raises(InputContractError):
        spell_steps([Step("n1", "+", slice_start=3, slice_end=2, is_partial=True)], {"n1": "ACGT"})
    for bad_cfg in [
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="  ", out_prefix="o"),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", max_jump=0),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", max_copy=0),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", spacer_n=-1),
        BuildConfig(gfa="a", ref_fa="a", backbone_prefix="R#", out_prefix="o", threads=0),
    ]:
        with pytest.raises(CLIArgumentError):
            validate_build_config(bad_cfg)

    # 3. repeat_guard internal helpers
    assert merge_intervals([(5, 2), (10, 10)]) == []
    assert complement_intervals(0, [(0, 5)]) == []
    assert complement_intervals(10, [(0, 10), (12, 15)]) == []
    assert subtract_intervals([], [(0, 5)]) == []
    assert merge_contiguous_provenance([]) == []
    assert merge_contiguous_provenance([prov], "ACGT") == [(prov, "")]
    for bad_slice_args in [
        ([Step("n1", "+")], [], {"n1": "ACGT"}, [(0, 4)], []),
        ([Step("unknown", "+")], [prov], {"n1": "ACGT"}, [(0, 4)], []),
        ([Step("n1", "+", slice_start=3, slice_end=2, is_partial=True)], [prov], {"n1": "ACGT"}, [(0, 4)], []),
        ([Step("n1", "+")], [prov], {"n1": "ACGT"}, [(0, 4)], []),
    ]:
        with pytest.raises(InputContractError):
            slice_candidate_by_intervals(*bad_slice_args)

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

    # 4. normalizer & linearizer internal overloads & termination guards
    st_a = Step("n1", "+")
    st_b = Step("n2", "+")
    nodes_ab = {"n1": "ACGT", "n2": "TGCA"}
    assert len(project_steps_onto_backbone([st_a, st_b], [st_a, st_b], nodes_ab, 0, 2)) == 2
    with pytest.raises(InputContractError):
        project_steps_onto_backbone([st_a], [prov, prov], [st_a], nodes_ab)

    prov1 = StepProvenance("H#1", "H#1#Chr01", 0, 4, "+")
    prov2_gap = StepProvenance("H#1", "H#1#Chr01", 10, 14, "+")
    assert len(_extract_contiguous_nonref_slices([st_a, st_b], [prov1, prov2_gap], nodes_ab, set(), min_sv_len=1)) == 2
    assert len(_extract_nonref_runs([st_a, st_b], [prov1, prov2_gap], nodes_ab, set())) == 2
    assert compute_weighted_lis([ProjectedAnchor(0, 0, "-", "n1", 4, st_a, prov1)]) == ([], False)

    frag_ab = PathFragment("H", "1", "H#1", "Chr01", "H#1#Chr01", 0, (st_a,), (prov1,), False, 4)
    nested_devs = extract_deviations(frag_ab, [], [st_b], is_top_level=False, lo=0, hi=1)
    assert len(nested_devs) == 1 and nested_devs[0].left_anchor_rank == "START"
    with pytest.raises(InputContractError):
        fold_tandem_cycles([st_a], [], set(), max_copy=1)
    with pytest.raises(InputContractError):
        fold_tandem_cycles([st_a], [prov], set(), max_copy=0)
    assert fold_tandem_cycles([], [], set(), max_copy=1).folded_steps == ()
    assert canonical_rotation([]) == ()
    assert detect_canonical_cnv([st_a], [st_a], nodes_ab, min_sv_len=1) is None

    k_parent = SubproblemStateKey(b_id=(("B", 0, 10, "+"),), lo=0, hi=10, has_logical_start=False, has_logical_end=False, m_key=((("M1", 0, 5, "+"),), (("M2", 0, 5, "+"),)))
    k_child_ok = SubproblemStateKey(b_id=(("B", 0, 10, "+"),), lo=1, hi=5, has_logical_start=True, has_logical_end=True, m_key=((("M1", 0, 5, "+"),),))
    assert not can_transition(k_parent, k_parent, frozenset())
    assert can_transition(k_parent, k_child_ok, frozenset())
    assert not can_transition(k_parent, k_child_ok, frozenset({k_child_ok}))

    st1, st2 = Step("x1", "+"), Step("x2", "+")
    p1 = StepProvenance("H1#1", "H1#1#Chr01", 0, 60, "+")
    p2 = StepProvenance("H2#1", "H2#1#Chr01", 0, 60, "+")
    s_x1, s_x2 = make_complex_seq(60, seed=9710), make_complex_seq(60, seed=9711)
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

    # 5. gff_liftover 3-arg overload & empty step_offsets fallback + merge_chromosomes API guards
    gff_file = tmp_path / "overload.gff3"
    gff_file.write_text("##gff-version 3\nChr01\tsrc\texon\t10\t100\t.\t+\t.\tID=ex1\n", encoding="utf-8")
    ins_ev = SVEvent("Chr01", 50, 130, "Chr01_INS_1", "INS", 50, 50, "0..1", "1:len=80,src=H1#1,n=1", 1, 2, "Chr01_REG_1", {"Ref#1": "0", "H1#1": "1"}, ())
    assert len(liftover_gff3(gff_file, [StepOffsetRow("Chr01", 0, "n1", 0, 200, 0, 200, 0)], [ins_ev])) == 3
    assert len(liftover_gff3_lines(["Chr01\tsrc\texon\t10\t100\t.\t+\t.\t"], "Chr01", [], [ins_ev])) == 3

    with pytest.raises(InputContractError):
        merge_chromosomes(tmp_path / "no_such_list.txt", tmp_path / "bad_m1")
    with pytest.raises(InputContractError):
        merge_chromosomes([], tmp_path / "bad_m2")
    with pytest.raises(InputContractError):
        merge_chromosomes([tmp_path / "missing_prefix"], tmp_path / "bad_m3")

# ---------------------------------------------------------------------------
# Round 2 Remediation Regression Suite (R2-D1 .. R2-D13)
# ---------------------------------------------------------------------------


def test_m7_22_r2_d1_d13_encoding_parity_multi_hit_and_compact_ref_extra_bounds(
    tmp_path: Path,
) -> None:
    """R2-D1 & R2-D13:
    - 4-way GFA encoding parity (`int+P`, `int+W`, `str+P`, `mixed+P/W`) when a non-carrier
      haplotype at `(ta, tb)` has a `< min_sv_len` micro-SNP or a `>= min_sv_len` pure DEL
      with `multi_hit` elsewhere on its contig: `multi_hit=1` must NOT leak onto `Alt1#1`'s INS!
    - Direct Python API `CompactRefNodeSet` with non-empty `_extra` and undersized `_mask` /
      `_nid_to_ref_rank` arrays never raises `IndexError` or misclassifies `_extra` ref nodes.
    """
    s1 = make_complex_seq(120, seed=9801)
    s2 = make_complex_seq(20, seed=9802)
    s3 = make_complex_seq(120, seed=9803)
    s4 = make_complex_seq(90, seed=9804)
    s5 = make_complex_seq(20, seed=9805)
    ref_seq = s1 + s2 + s3
    ref_fa = tmp_path / "r2_d1.fa"
    ref_fa.write_text(f">Chr01\n{ref_seq}\n", encoding="utf-8")

    variants = {
        "int_p": [
            "H\tVN:Z:1.0",
            f"S\t1\t{s1}",
            f"S\t2\t{s2}",
            f"S\t3\t{s3}",
            f"S\t4\t{s4}",
            f"S\t5\t{s5}",
            "L\t1\t+\t2\t+\t0M",
            "L\t2\t+\t3\t+\t0M",
            "L\t1\t+\t4\t+\t0M",
            "L\t4\t+\t2\t+\t0M",
            "L\t1\t+\t5\t+\t0M",
            "L\t5\t+\t3\t+\t0M",
            "L\t3\t+\t1\t+\t0M",
            "P\tRef#1#Chr01\t1+,2+,3+\t*",
            "P\tAlt1#1#Chr01\t1+,4+,2+,3+\t*",
            "P\tAlt2#1#Chr01\t1+,5+,3+,1+\t*",
        ],
        "int_w": [
            "H\tVN:Z:1.0",
            f"S\t1\t{s1}",
            f"S\t2\t{s2}",
            f"S\t3\t{s3}",
            f"S\t4\t{s4}",
            f"S\t5\t{s5}",
            "L\t1\t+\t2\t+\t0M",
            "L\t2\t+\t3\t+\t0M",
            "L\t1\t+\t4\t+\t0M",
            "L\t4\t+\t2\t+\t0M",
            "L\t1\t+\t5\t+\t0M",
            "L\t5\t+\t3\t+\t0M",
            "L\t3\t+\t1\t+\t0M",
            "W\tRef\t1\tChr01\t0\t*\t>1>2>3",
            "W\tAlt1\t1\tChr01\t0\t*\t>1>4>2>3",
            "W\tAlt2\t1\tChr01\t0\t*\t>1>5>3>1",
        ],
        "str_p": [
            "H\tVN:Z:1.0",
            f"S\tn1\t{s1}",
            f"S\tn2\t{s2}",
            f"S\tn3\t{s3}",
            f"S\tn4\t{s4}",
            f"S\tn5\t{s5}",
            "L\tn1\t+\tn2\t+\t0M",
            "L\tn2\t+\tn3\t+\t0M",
            "L\tn1\t+\tn4\t+\t0M",
            "L\tn4\t+\tn2\t+\t0M",
            "L\tn1\t+\tn5\t+\t0M",
            "L\tn5\t+\tn3\t+\t0M",
            "L\tn3\t+\tn1\t+\t0M",
            "P\tRef#1#Chr01\tn1+,n2+,n3+\t*",
            "P\tAlt1#1#Chr01\tn1+,n4+,n2+,n3+\t*",
            "P\tAlt2#1#Chr01\tn1+,n5+,n3+,n1+\t*",
        ],
        "mixed_pw": [
            "H\tVN:Z:1.0",
            f"S\t1\t{s1}",
            f"S\t2\t{s2}",
            f"S\t3\t{s3}",
            f"S\t4\t{s4}",
            f"S\tn5\t{s5}",
            "L\t1\t+\t2\t+\t0M",
            "L\t2\t+\t3\t+\t0M",
            "L\t1\t+\t4\t+\t0M",
            "L\t4\t+\t2\t+\t0M",
            "L\t1\t+\tn5\t+\t0M",
            "L\tn5\t+\t3\t+\t0M",
            "L\t3\t+\t1\t+\t0M",
            "P\tRef#1#Chr01\t1+,2+,3+\t*",
            "W\tAlt1\t1\tChr01\t0\t*\t>1>4>2>3",
            "W\tAlt2\t1\tChr01\t0\t*\t>1>n5>3>1",
        ],
    }

    sv_sigs = {}
    gt_texts: dict[str, str] = {}
    for vname, lines in variants.items():
        gfa_p = tmp_path / f"{vname}.gfa"
        gfa_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_p = tmp_path / f"out_{vname}"
        assert (
            cli_main(
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
                ]
            )
            == 0
        )
        bun = OutputBundle(out_p)
        assert len(bun.sv_rows) == 1
        assert bun.sv_rows[0]["type"] == "INS"
        assert "multi_hit=1" not in bun.sv_rows[0]["alleles"]
        sv_sigs[vname] = [
            (r["type"], r["ref_start"], r["ref_end"], r["pan_start"], r["alleles"])
            for r in bun.sv_rows
        ]
        gt_texts[vname] = Path(f"{out_p}.genotypes.tsv").read_text(encoding="utf-8")

    assert sv_sigs["int_p"] == sv_sigs["int_w"] == sv_sigs["str_p"] == sv_sigs["mixed_pw"]
    assert gt_texts["int_p"] == gt_texts["int_w"] == gt_texts["str_p"] == gt_texts["mixed_pw"]

    # Also test co-occurring >= min_sv_len pure DEL with multi_hit at the exact same (ta, tb)
    s2_del = make_complex_seq(80, seed=9806)
    del_fa = tmp_path / "r2_d1_del.fa"
    del_fa.write_text(f">Chr01\n{s1 + s2_del + s3}\n", encoding="utf-8")
    del_gfa = tmp_path / "r2_d1_del.gfa"
    del_gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s1}",
                f"S\t2\t{s2_del}",
                f"S\t3\t{s3}",
                f"S\t4\t{s4}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t1\t+\t4\t+\t0M",
                "L\t4\t+\t2\t+\t0M",
                "L\t1\t+\t3\t+\t0M",
                "L\t3\t+\t1\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+\t*",
                "P\tAlt1#1#Chr01\t1+,4+,2+,3+\t*",
                "P\tAlt2#1#Chr01\t1+,3+,1+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_del = tmp_path / "out_del_mh"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(del_gfa),
                "--ref-fa",
                str(del_fa),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(out_del),
            ]
        )
        == 0
    )
    bun_del = OutputBundle(out_del)
    by_type = {r["type"]: r for r in bun_del.sv_rows}
    assert "multi_hit=1" not in by_type["INS"]["alleles"]
    assert "multi_hit=1" in by_type["DEL"]["alleles"]

    # R2-D13: Direct API CompactRefNodeSet with non-empty _extra and short _mask/_nid_to_ref_rank
    cfg_api = BuildConfig(
        gfa=str(tmp_path / "int_p.gfa"),
        ref_fa=str(ref_fa),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "api_out"),
    )
    gr = load_and_validate_inputs(cfg_api)
    from dataclasses import replace as dc_replace

    short_ref_set = CompactRefNodeSet(
        mask=np.array([False, True, True], dtype=np.bool_),
        extra=frozenset({"3"}),
    )
    gr_custom = dc_replace(
        gr,
        global_ref_nodes=short_ref_set,
        _nid_to_ref_rank=np.array([-1, 0, 1], dtype=np.int32),
    )
    res_custom = run_chromosome_pipeline(gr_custom, cfg_api)
    assert len(res_custom.sv_events) == 1
    assert "multi_hit=1" not in res_custom.sv_events[0].alleles_str


def test_m7_23_r2_d2_exact_p_w_src_path_collision_rejected_at_large_offset(
    tmp_path: Path,
) -> None:
    """R2-D2: Exact P/W src_path collision (`Alt#1#Chr01:1000-1200`) raises InputContractError even when seq_start >= len(P)."""
    s1 = make_complex_seq(100, seed=9811)
    s2 = make_complex_seq(100, seed=9812)
    s3 = make_complex_seq(100, seed=9813)
    s4 = make_complex_seq(260, seed=9814)
    fa = tmp_path / "r2_d2.fa"
    fa.write_text(f">Chr01\n{s1 + s2 + s3}\n", encoding="utf-8")

    for tag, n4 in (("int", "4"), ("str", "node4")):
        gfa = tmp_path / f"r2_d2_{tag}.gfa"
        gfa.write_text(
            "\n".join(
                [
                    "H\tVN:Z:1.0",
                    f"S\t1\t{s1}",
                    f"S\t2\t{s2}",
                    f"S\t3\t{s3}",
                    f"S\t{n4}\t{s4}",
                    "L\t1\t+\t2\t+\t0M",
                    "L\t2\t+\t3\t+\t0M",
                    f"L\t{n4}\t+\t2\t+\t0M",
                    "P\tRef#1#Chr01\t1+,2+,3+\t*",
                    f"P\tAlt#1#Chr01:1000-1200\t{n4}+,2+\t*",
                    "W\tAlt\t1\tChr01\t1000\t1200\t>1>2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(InputContractError, match="Ambiguous duplicate source interval"):
            load_and_validate_inputs(
                BuildConfig(
                    gfa=str(gfa),
                    ref_fa=str(fa),
                    backbone_prefix="Ref#1#",
                    out_prefix=str(tmp_path / f"out_{tag}"),
                )
            )


def test_m7_24_r2_d3_large_offsets_verify_checks_gfa_node_sequence_plus_and_minus(
    tmp_path: Path,
) -> None:
    """R2-D3: When `{p}.offsets.tsv > 2 MB`, `verify_build_outputs` verifies every step's GFA node sequence (+ or -) against `ref_fa`."""
    from panflat.gfa_io import revcomp as gfa_revcomp

    n_steps = 55_000
    step_len = 200
    unit_a = make_complex_seq(step_len, seed=9821)
    unit_b_ref = make_complex_seq(step_len, seed=9822)
    unit_b_rc = gfa_revcomp(unit_b_ref)  # stored in '-' orientation on node 2

    ref_seq = "".join(unit_a if i % 2 == 0 else unit_b_ref for i in range(n_steps))
    fa = tmp_path / "large_ref.fa"
    fa.write_text(f">Chr01\n{ref_seq}\n", encoding="utf-8")

    pref = tmp_path / "large_b"
    Path(f"{pref}.pan.fa").write_text(f">Chr01\n{ref_seq}\n", encoding="utf-8")
    with Path(f"{pref}.offsets.tsv").open("w", encoding="utf-8") as fh:
        fh.write("chrom\tref_step\tnode_id\tref_start\tref_end\tpan_start\tpan_end\tslot_ins_len\n")
        for i in range(n_steps):
            nid = 1 if i % 2 == 0 else 2
            s = i * step_len
            e = s + step_len
            fh.write(f"Chr01\t{i}\t{nid}\t{s}\t{e}\t{s}\t{e}\t0\n")
    assert Path(f"{pref}.offsets.tsv").stat().st_size > 2_000_000

    total_len = n_steps * step_len
    Path(f"{pref}.ref_to_pan.chain").write_text(
        f"chain {total_len} Chr01 {total_len} + 0 {total_len} Chr01 {total_len} + 0 {total_len} 1\n"
        f"{total_len}\n\n# block\t0\t{total_len}\t0\n",
        encoding="utf-8",
    )
    Path(f"{pref}.sv.tsv").write_text(
        "chrom\tpan_start\tpan_end\tevent_id\ttype\tref_start\tref_end\tanchors\talleles\tn_carrier\tn_called\tregion_id\n",
        encoding="utf-8",
    )
    Path(f"{pref}.genotypes.tsv").write_text("event_id\tRef#1\n", encoding="utf-8")
    Path(f"{pref}.inserts.tsv").write_text(
        "chrom\tpan_start\tpan_end\tevent_id\tallele\tpiece\tsrc_hap\tsrc_path\tsrc_intervals\tsrc_strand\tmembers\tjoin_before\n",
        encoding="utf-8",
    )
    Path(f"{pref}.unresolved.tsv").write_text(
        "record_id\tregion_id\tevent_id\tsrc_path\tsrc_start\tsrc_end\tsrc_strand\tlength\treason\texported_fasta_id\n",
        encoding="utf-8",
    )
    Path(f"{pref}.unresolved.fa").write_text("", encoding="utf-8")
    Path(f"{pref}.summary.tsv").write_text(
        "metric\tvalue\nchrom\tChr01\nref_path\tRef#1#Chr01\nref_hap\tRef#1\nspacer_n\t100\ncentromere_mode\toff\n",
        encoding="utf-8",
    )
    Path(f"{pref}.log").write_text("INFO build complete\n", encoding="utf-8")

    steps_str = ",".join("1+" if i % 2 == 0 else "2-" for i in range(n_steps))
    # 1) <10 MB dict GFA path: valid passes, mutated same-length S sequence fails INV1
    gfa_small_ok = tmp_path / "small_ok.gfa"
    gfa_small_ok.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{unit_a}\nS\t2\t{unit_b_rc}\nP\tRef#1#Chr01\t{steps_str}\t*\n",
        encoding="utf-8",
    )
    assert verify_build_outputs(pref, gfa_small_ok, fa).passed

    gfa_small_bad = tmp_path / "small_bad.gfa"
    gfa_small_bad.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{'T' * step_len}\nS\t2\t{unit_b_rc}\nP\tRef#1#Chr01\t{steps_str}\t*\n",
        encoding="utf-8",
    )
    rep_sb = verify_build_outputs(pref, gfa_small_bad, fa)
    assert not rep_sb.passed and not rep_sb.invariant_1_passed

    # 2) >10 MB CompactNodeStore GFA path: valid passes, mutated same-length S sequence fails INV1
    pad_seq = "A" * 10_500_000
    gfa_big_ok = tmp_path / "big_ok.gfa"
    gfa_big_ok.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{unit_a}\nS\t2\t{unit_b_rc}\nS\t3\t{pad_seq}\nP\tRef#1#Chr01\t{steps_str}\t*\n",
        encoding="utf-8",
    )
    assert verify_build_outputs(pref, gfa_big_ok, fa).passed

    gfa_big_bad = tmp_path / "big_bad.gfa"
    gfa_big_bad.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{unit_a}\nS\t2\t{'G' * step_len}\nS\t3\t{pad_seq}\nP\tRef#1#Chr01\t{steps_str}\t*\n",
        encoding="utf-8",
    )
    rep_bb = verify_build_outputs(pref, gfa_big_bad, fa)
    assert not rep_bb.passed and not rep_bb.invariant_1_passed


def test_m7_25_r2_d4_merged_chain_step_and_block_footer_tamper_and_duplicate_tname(
    tmp_path: Path,
) -> None:
    """R2-D4: Merged `ref_to_pan.chain` mixing `>10,000`-step Chr01 (no `# step`) and `<10,000`-step Chr02 (`# step` present) passes verify, and rejects tampered `# step`/`# block`, duplicate `tName`, and orphan comments."""
    # Chr01: 10,005 steps (10 bp each) -> LazyStepOffsetSequence omits `# step`
    n_c1 = 10_005
    u10 = "ACGTACGTAA"
    ref_c1 = u10 * n_c1
    gfa1 = tmp_path / "c1.gfa"
    p1_steps = ",".join("1+" for _ in range(n_c1))
    gfa1.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{u10}\nL\t1\t+\t1\t+\t0M\nP\tRef#1#Chr01\t{p1_steps}\t*\n",
        encoding="utf-8",
    )
    fa1 = tmp_path / "c1.fa"
    fa1.write_text(f">Chr01\n{ref_c1}\n", encoding="utf-8")
    out1 = tmp_path / "b_c1"
    assert (
        cli_main(
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
            ]
        )
        == 0
    )

    # Chr02: 2 steps -> emits `# step` lines
    s2_a = make_complex_seq(120, seed=9831)
    s2_b = make_complex_seq(120, seed=9832)
    gfa2 = tmp_path / "c2.gfa"
    gfa2.write_text(
        f"H\tVN:Z:1.0\nS\t2\t{s2_a}\nS\t3\t{s2_b}\nL\t2\t+\t3\t+\t0M\nP\tRef#1#Chr02\t2+,3+\t*\n",
        encoding="utf-8",
    )
    fa2 = tmp_path / "c2.fa"
    fa2.write_text(f">Chr02\n{s2_a + s2_b}\n", encoding="utf-8")
    out2 = tmp_path / "b_c2"
    assert (
        cli_main(
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
            ]
        )
        == 0
    )

    merged = tmp_path / "merged_chain"
    merge_chromosomes([out1, out2], merged)
    comb_gfa = tmp_path / "comb.gfa"
    comb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{u10}\nS\t2\t{s2_a}\nS\t3\t{s2_b}\n"
        "L\t1\t+\t1\t+\t0M\nL\t2\t+\t3\t+\t0M\n"
        f"P\tRef#1#Chr01\t{p1_steps}\t*\nP\tRef#1#Chr02\t2+,3+\t*\n",
        encoding="utf-8",
    )
    comb_fa = tmp_path / "comb.fa"
    comb_fa.write_text(f">Chr01\n{ref_c1}\n>Chr02\n{s2_a + s2_b}\n", encoding="utf-8")

    rep_ok = verify_build_outputs(merged, comb_gfa, comb_fa)
    assert rep_ok.passed and rep_ok.invariant_3_passed

    orig_chain = Path(f"{merged}.ref_to_pan.chain").read_text(encoding="utf-8")
    # 1) Tamper Chr02 `# step`
    Path(f"{merged}.ref_to_pan.chain").write_text(
        orig_chain.replace("# step\t0\t0\t120", "# step\t0\t0\t999", 1),
        encoding="utf-8",
    )
    rep_bad_step = verify_build_outputs(merged, comb_gfa, comb_fa)
    assert not rep_bad_step.passed and not rep_bad_step.invariant_3_passed

    # 2) Duplicate Chr02 chain header
    Path(f"{merged}.ref_to_pan.chain").write_text(
        orig_chain + "\nchain 240 Chr02 240 + 0 240 Chr02 240 + 0 240 3\n240\n",
        encoding="utf-8",
    )
    rep_dup = verify_build_outputs(merged, comb_gfa, comb_fa)
    assert not rep_dup.passed and not rep_dup.invariant_3_passed

    # 3) Orphan `# block` before first chain header
    Path(f"{merged}.ref_to_pan.chain").write_text(
        "# block\t0\t100\t0\n" + orig_chain, encoding="utf-8"
    )
    rep_orph = verify_build_outputs(merged, comb_gfa, comb_fa)
    assert not rep_orph.passed and not rep_orph.invariant_3_passed
    Path(f"{merged}.ref_to_pan.chain").write_text(orig_chain, encoding="utf-8")


def test_m7_26_r2_d5_qc_stage1_contiguous_ref_and_stage2_no_cross_sep_bridging(
    tmp_path: Path,
) -> None:
    """R2-D5:
    - Stage 1 aligns against contiguous `ref_only_seq` (not spacer-broken / short-insert-chimeric `pan_no_long`):
      a 200 bp 100% copy of `ref[0:200]` split by a slot insertion in `pan.fa` gets `qc_redundancy_0.98_bp == 200`,
      while a novel 200 bp piece matching `short_ins(100) + ref_b(100)` gets `qc_redundancy_0.90_bp == 0`.
    - Stage 2 rejects alignments bridging across `sep = 'N' * 200` between distinct target pieces.
    """
    a100 = make_complex_seq(100, seed=9841)
    b100 = make_complex_seq(100, seed=9842)
    c100 = make_complex_seq(100, seed=9843)
    x_short = make_complex_seq(100, seed=9844)

    # Direction 1: 200 bp piece = a100 + b100 (100% match to contiguous ref[0:200])
    gfa_fn = tmp_path / "qc_fn.gfa"
    gfa_fn.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{a100}",
                f"S\t2\t{b100}",
                f"S\t3\t{c100}",
                f"S\t4\t{x_short}",
                f"S\t5\t{a100 + b100}",
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
    fa_qc = tmp_path / "qc.fa"
    fa_qc.write_text(f">Chr01\n{a100 + b100 + c100}\n", encoding="utf-8")
    out_fn = tmp_path / "out_qc_fn"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa_fn),
                "--ref-fa",
                str(fa_qc),
                "--backbone-prefix",
                "Ref#1#",
                "--qc",
                "--out-prefix",
                str(out_fn),
            ]
        )
        == 0
    )
    bun_fn = OutputBundle(out_fn)
    assert bun_fn.summary_kv["qc_redundancy_0.98_bp"] == "200"

    # Direction 2: 200 bp piece = x_short + b100 (only 100 bp matches ref, <150 bp threshold!)
    gfa_fp = tmp_path / "qc_fp.gfa"
    gfa_fp.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{a100}",
                f"S\t2\t{b100}",
                f"S\t3\t{c100}",
                f"S\t4\t{x_short}",
                f"S\t5\t{x_short + b100}",
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
    out_fp = tmp_path / "out_qc_fp"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa_fp),
                "--ref-fa",
                str(fa_qc),
                "--backbone-prefix",
                "Ref#1#",
                "--qc",
                "--out-prefix",
                str(out_fp),
            ]
        )
        == 0
    )
    bun_fp = OutputBundle(out_fp)
    assert bun_fp.summary_kv["qc_redundancy_0.90_bp"] == "0"

    # Stage 2 guard: 3 long pieces where p0 (150 bp) and p2 (150 bp) sit adjacent in target_bit==0,
    # and p1 (100 bp suffix of p0 + 100 bp prefix of p2 = 200 bp) is in query_bit==1 -> must NOT bridge sep!
    p0_seq = make_complex_seq(150, seed=9845)
    p2_seq = make_complex_seq(150, seed=9846)
    p1_seq = p0_seq[50:] + p2_seq[:100]  # 100 bp match to p0, 100 bp match to p2 (<150 bp each!)
    ref_mock = make_complex_seq(200, seed=9847)
    pan_mock = ref_mock[:100] + p0_seq + p1_seq + p2_seq + ref_mock[100:]
    from panflat.models import StepOffsetRow

    mock_res = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq=pan_mock,
        step_offsets=(
            StepOffsetRow("Chr01", 0, "1", 0, 100, 0, 100, 0),
            StepOffsetRow("Chr01", 1, "2", 100, 200, 600, 700, 500),
        ),
        sv_events=(),
        all_haplotypes=("Ref#1", "Alt#1"),
        inserts_rows=(
            {"pan_start": 100, "pan_end": 250, "event_id": "E1"},
            {"pan_start": 250, "pan_end": 450, "event_id": "E2"},
            {"pan_start": 450, "pan_end": 600, "event_id": "E3"},
        ),
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        summary_metrics={},
        log_lines=(),
        json_log_entries=(),
        gfa_path="mock.gfa",
        gfa_sha256="0" * 64,
        ref_fa_path="mock.fa",
    )
    qc_m = run_qc(mock_res)
    assert qc_m["qc_redundancy_0.90_bp"] == 0


def test_m7_27_r2_d6_w_line_ref_fasta_merged_nonpansn_ambiguous_override_and_exact_tiebreak(
    tmp_path: Path,
) -> None:
    """R2-D6 (a),(b),(c),(d):
    - (a) W-line reference (`W Ref 1 Chr01 100 900`) with PanSN FASTA (`>Ref#1#Chr01`) and
      coordinate-suffixed FASTA (`>Ref#1#Chr01:100-900`) across all 6 `--backbone-prefix` forms.
    - (b) Merged non-PanSN `Ref_Chr01`/`Ref_Chr02` (`--backbone-prefix Ref_`) and W-walk
      `ctg1`/`ctg2` (`--backbone-prefix SampleA#1#`) with plain multi-chrom FASTA.
    - (c) Single-chrom ambiguous `verify --backbone-prefix SampleA#1#` rejected when GFA has
      `SampleA#1#Chr10A` + `SampleA#1#Chr10A_unplaced`.
    - (d) Non-PanSN exact path tie-breaking (`RefChr01` vs `RefChr01_alt`).
    """
    s1 = make_complex_seq(400, seed=9851)
    s2 = make_complex_seq(400, seed=9852)
    s_ins = make_complex_seq(100, seed=9853)
    ref_800 = s1 + s2

    # (a) W-line reference with coordinate-suffixed & PanSN multi-FASTA
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
    w_pansn_fa = tmp_path / "w_pansn.fa"
    w_pansn_fa.write_text(
        f">Alt#1#Chr01\n{s1 + s_ins + s2}\n>Ref#1#Chr01\n{ref_800}\n", encoding="utf-8"
    )
    w_cs_multi_fa = tmp_path / "w_cs_multi.fa"
    w_cs_multi_fa.write_text(
        f">Alt#1#Chr01:50-950\n{s1 + s_ins + s2}\n>Ref#1#Chr01:100-900\n{ref_800}\n",
        encoding="utf-8",
    )
    w_cs_single_fa = tmp_path / "w_cs_single.fa"
    w_cs_single_fa.write_text(f">Ref#1#Chr01:100-900\n{ref_800}\n", encoding="utf-8")

    out_w1 = tmp_path / "out_w1"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(wref_gfa),
                "--ref-fa",
                str(w_pansn_fa),
                "--backbone-prefix",
                "Ref#1#Chr01:100-900",
                "--out-prefix",
                str(out_w1),
            ]
        )
        == 0
    )
    out_w2 = tmp_path / "out_w2"
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(wref_gfa),
                "--ref-fa",
                str(w_cs_single_fa),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(out_w2),
            ]
        )
        == 0
    )
    for bp_ok in ("Ref#1#", "Ref#0#", "Ref", "Ref#1", "Ref#1#Chr01", "Ref#1#Chr01:100-900"):
        assert verify_build_outputs(out_w1, wref_gfa, w_pansn_fa, backbone_prefix=bp_ok).passed
        assert verify_build_outputs(out_w1, wref_gfa, w_cs_multi_fa, backbone_prefix=bp_ok).passed
        assert verify_build_outputs(out_w2, wref_gfa, w_cs_single_fa, backbone_prefix=bp_ok).passed
    assert not verify_build_outputs(
        out_w1, wref_gfa, w_cs_multi_fa, backbone_prefix="WrongSample#1#"
    ).passed

    # (b) Merged non-PanSN Ref_Chr01/Ref_Chr02 + plain multi-FASTA + --backbone-prefix Ref_
    s_c2 = make_complex_seq(300, seed=9854)
    np1_gfa = tmp_path / "np1.gfa"
    np1_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nP\tRef_Chr01\t1+\t*\n", encoding="utf-8"
    )
    np2_gfa = tmp_path / "np2.gfa"
    np2_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t2\t{s_c2}\nP\tRef_Chr02\t2+\t*\n", encoding="utf-8"
    )
    fa_c1 = tmp_path / "plain_c1.fa"
    fa_c1.write_text(f">Chr01\n{s1}\n", encoding="utf-8")
    fa_c2 = tmp_path / "plain_c2.fa"
    fa_c2.write_text(f">Chr02\n{s_c2}\n", encoding="utf-8")
    assert cli_main(["build", "--gfa", str(np1_gfa), "--ref-fa", str(fa_c1), "--backbone-prefix", "Ref_", "--out-prefix", str(tmp_path / "np_b1")]) == 0
    assert cli_main(["build", "--gfa", str(np2_gfa), "--ref-fa", str(fa_c2), "--backbone-prefix", "Ref_", "--out-prefix", str(tmp_path / "np_b2")]) == 0
    np_merged = tmp_path / "np_merged"
    merge_chromosomes([tmp_path / "np_b1", tmp_path / "np_b2"], np_merged)
    np_comb_gfa = tmp_path / "np_comb.gfa"
    np_comb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\nP\tRef_Chr01\t1+\t*\nP\tRef_Chr02\t2+\t*\n",
        encoding="utf-8",
    )
    plain_multi_fa = tmp_path / "plain_multi.fa"
    plain_multi_fa.write_text(f">Chr01\n{s1}\n>Chr02\n{s_c2}\n", encoding="utf-8")
    assert verify_build_outputs(np_merged, np_comb_gfa, plain_multi_fa, backbone_prefix="Ref_").passed
    assert not verify_build_outputs(np_merged, np_comb_gfa, plain_multi_fa, backbone_prefix="Wrong_").passed

    # Also W-walk SampleA#1#ctg1 + SampleA#1#ctg2 with plain multi-chrom FASTA
    w1_gfa = tmp_path / "w_ctg1.gfa"
    w1_gfa.write_text(f"H\tVN:Z:1.0\nS\t1\t{s1}\nW\tSampleA\t1\tctg1\t0\t400\t>1\n", encoding="utf-8")
    w2_gfa = tmp_path / "w_ctg2.gfa"
    w2_gfa.write_text(f"H\tVN:Z:1.0\nS\t2\t{s_c2}\nW\tSampleA\t1\tctg2\t0\t300\t>2\n", encoding="utf-8")
    assert cli_main(["build", "--gfa", str(w1_gfa), "--ref-fa", str(fa_c1), "--backbone-prefix", "SampleA#1#", "--out-prefix", str(tmp_path / "w_b1")]) == 0
    assert cli_main(["build", "--gfa", str(w2_gfa), "--ref-fa", str(fa_c2), "--backbone-prefix", "SampleA#1#", "--out-prefix", str(tmp_path / "w_b2")]) == 0
    w_merged = tmp_path / "w_merged"
    merge_chromosomes([tmp_path / "w_b1", tmp_path / "w_b2"], w_merged)
    w_comb_gfa = tmp_path / "w_comb.gfa"
    w_comb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\nW\tSampleA\t1\tctg1\t0\t400\t>1\nW\tSampleA\t1\tctg2\t0\t300\t>2\n",
        encoding="utf-8",
    )
    assert verify_build_outputs(w_merged, w_comb_gfa, plain_multi_fa, backbone_prefix="SampleA#1#").passed
    assert not verify_build_outputs(w_merged, w_comb_gfa, plain_multi_fa, backbone_prefix="SampleB#1#").passed

    # (c) Single-chrom ambiguous `SampleA#1#` rejected on verify when GFA has Chr10A + Chr10A_unplaced
    amb_gfa = tmp_path / "amb_single.gfa"
    amb_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\n"
        "P\tSampleA#1#Chr10A\t1+\t*\nP\tSampleA#1#Chr10A_unplaced\t2+\t*\n",
        encoding="utf-8",
    )
    amb_fa = tmp_path / "amb_single.fa"
    amb_fa.write_text(f">SampleA#1#Chr10A\n{s1}\n>SampleA#1#Chr10A_unplaced\n{s_c2}\n", encoding="utf-8")
    out_amb = tmp_path / "out_amb_ok"
    assert cli_main(["build", "--gfa", str(amb_gfa), "--ref-fa", str(amb_fa), "--backbone-prefix", "SampleA#1#Chr10A", "--out-prefix", str(out_amb)]) == 0
    assert verify_build_outputs(out_amb, amb_gfa, amb_fa, backbone_prefix="SampleA#1#Chr10A").passed
    rep_amb_bad = verify_build_outputs(out_amb, amb_gfa, amb_fa, backbone_prefix="SampleA#1#")
    assert not rep_amb_bad.passed and not rep_amb_bad.invariant_1_passed

    # (d) Non-PanSN exact path tie-breaking (`RefChr01` vs `RefChr01_alt`)
    tie_gfa = tmp_path / "tie.gfa"
    tie_gfa.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{s1}\nS\t2\t{s_c2}\nP\tRefChr01\t1+\t*\nP\tRefChr01_alt\t2+\t*\n",
        encoding="utf-8",
    )
    out_tie = tmp_path / "out_tie"
    assert cli_main(["build", "--gfa", str(tie_gfa), "--ref-fa", str(fa_c1), "--backbone-prefix", "RefChr01", "--out-prefix", str(out_tie)]) == 0
    assert verify_build_outputs(out_tie, tie_gfa, fa_c1, backbone_prefix="RefChr01").passed
    assert not verify_build_outputs(out_tie, tie_gfa, fa_c1, backbone_prefix="RefChr").passed


def test_m7_28_r2_d7_d8_right_nested_merge_associativity_and_lifted_gff3_sv_id_remap(
    tmp_path: Path,
) -> None:
    """R2-D7 & R2-D8:
    - Right-nested `merge([Chr01, merge([Chr02, Chr03])])`, left-nested `merge([merge([Chr01, Chr02]), Chr03])`,
      and flat `merge([Chr01, Chr02, Chr03])` produce 100% identical outputs even when `Chr02` has 0 unresolved
      rows and `Chr01`/`Chr03` have colliding unprefixed `UNRES_000001` on `unplaced_scaf_A`.
    - Colliding `SV_000001` IDs renamed in `sv.tsv` are remapped identically in `{merged}.ref.lifted.gff3` `sv_id=`.
    """
    prefs: dict[str, Path] = {}
    for idx, chrom in enumerate(("Chr01", "Chr02", "Chr03"), start=1):
        s1 = make_complex_seq(150, seed=9860 + idx * 10 + 1)
        s2 = make_complex_seq(150, seed=9860 + idx * 10 + 2)
        ins_s = make_complex_seq(90, seed=9860 + idx * 10 + 3)
        un_s = make_complex_seq(120 + idx * 10, seed=9860 + idx * 10 + 4)

        gfa_lines = [
            "H\tVN:Z:1.0",
            f"S\t1\t{s1}",
            f"S\t2\t{s2}",
            f"S\t3\t{ins_s}",
            "L\t1\t+\t2\t+\t0M",
            "L\t1\t+\t3\t+\t0M",
            "L\t3\t+\t2\t+\t0M",
            f"P\tRef#1#{chrom}\t1+,2+\t*",
            f"P\tAlt#1#{chrom}\t1+,3+,2+\t*",
        ]
        if chrom != "Chr02":
            gfa_lines.insert(4, f"S\t4\t{un_s}")
            gfa_lines.insert(8, "L\t4\t+\t1\t+\t0M")
            gfa_lines.append("P\tunplaced_scaf_A\t4+,1+,2+\t*")

        gfa_p = tmp_path / f"{chrom}.gfa"
        gfa_p.write_text("\n".join(gfa_lines) + "\n", encoding="utf-8")
        fa_p = tmp_path / f"{chrom}.fa"
        fa_p.write_text(f">{chrom}\n{s1 + s2}\n", encoding="utf-8")
        gff_p = tmp_path / f"{chrom}.gff3"
        gff_p.write_text(
            f"##gff-version 3\n{chrom}\tsrc\tgene\t1\t300\t.\t+\t.\tID=g_{chrom}\n"
            f"{chrom}\tsrc\tmRNA\t1\t300\t.\t+\t.\tID=m_{chrom};Parent=g_{chrom}\n"
            f"{chrom}\tsrc\texon\t1\t300\t.\t+\t.\tID=e_{chrom};Parent=m_{chrom}\n",
            encoding="utf-8",
        )
        out_p = tmp_path / f"p_{chrom}"
        assert (
            cli_main(
                [
                    "build",
                    "--gfa",
                    str(gfa_p),
                    "--ref-fa",
                    str(fa_p),
                    "--backbone-prefix",
                    "Ref#1#",
                    "--ref-gff",
                    str(gff_p),
                    "--out-prefix",
                    str(out_p),
                ]
            )
            == 0
        )
        # Rewrite SV ID to colliding 'SV_000001' and UNRES ID to unprefixed 'UNRES_000001'
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
                    fp.read_text(encoding="utf-8").replace(
                        f"{chrom}_UNRES_1", "UNRES_000001"
                    ),
                    encoding="utf-8",
                )
        prefs[chrom] = out_p

    m_flat = tmp_path / "m_flat"
    merge_chromosomes([prefs["Chr01"], prefs["Chr02"], prefs["Chr03"]], m_flat)

    m12 = tmp_path / "m12"
    merge_chromosomes([prefs["Chr01"], prefs["Chr02"]], m12)
    m_left = tmp_path / "m_left"
    merge_chromosomes([m12, prefs["Chr03"]], m_left)

    m23 = tmp_path / "m23"
    merge_chromosomes([prefs["Chr02"], prefs["Chr03"]], m23)
    m_right = tmp_path / "m_right"
    merge_chromosomes([prefs["Chr01"], m23], m_right)

    for ext in ("unresolved.tsv", "unresolved.fa", "sv.tsv", "inserts.tsv", "ref.lifted.gff3"):
        t_flat = Path(f"{m_flat}.{ext}").read_text(encoding="utf-8")
        t_left = Path(f"{m_left}.{ext}").read_text(encoding="utf-8")
        t_right = Path(f"{m_right}.{ext}").read_text(encoding="utf-8")
        assert t_flat == t_left == t_right, f"Associativity mismatch in {ext}"

    bun_r = OutputBundle(m_right)
    assert [r["src_path"] for r in bun_r.unresolved] == [
        "Chr01::unplaced_scaf_A",
        "Chr03::unplaced_scaf_A",
    ]
    sv_ids_set = {r["event_id"] for r in bun_r.sv_rows}
    assert sv_ids_set == {"SV_000001", "Chr02_SV_000001", "Chr03_SV_000001"}

    gff_txt = Path(f"{m_right}.ref.lifted.gff3").read_text(encoding="utf-8")
    assert "Chr02\tsrc\tgene" in gff_txt and "sv_id=Chr02_SV_000001" in gff_txt
    assert "sv_id=Chr03_SV_000001" in gff_txt
    for line in gff_txt.splitlines():
        if line and not line.startswith("#") and "sv_id=" in line:
            attrs = dict(
                kv.split("=", 1) for kv in line.split("\t")[8].split(";") if "=" in kv
            )
            for eid in attrs["sv_id"].split(","):
                assert eid in sv_ids_set


def test_m7_29_r2_d9_d10_d11_d12_fasta_gfa_strictness_merged_cen_gff_keying_and_isoform_status(
    tmp_path: Path,
) -> None:
    """R2-D9, R2-D10, R2-D11, R2-D12:
    - R2-D9: `verify_build_outputs` rejects pre-header garbage, empty skipped records, and
      non-alphabetic characters in `--ref-fa` and duplicate/corrupt records in `unresolved.fa`.
    - R2-D10: `>10 MB` GFA fast path rejects Unicode digits in W walks (`>1\uff12`) and invalid S sequences.
    - R2-D11: Merged heterogeneous centromere bundle fails `[CEN]` when `centromere.tsv` is missing
      even without `--centromere-bed`; merged GFF3 with colliding `ID=tx1` across chromosomes keys
      transcripts/proteins by `(seqid, Parent)`.
    - R2-D12: Shared multi-Parent exon (`Parent=tx1,tx2`) does NOT bleed `cds_split_by_insertion`
      onto isoform `tx2` whose shared exon is 5'-UTR only.
    """
    orf1 = "ATG" + ("GCT" * 29)  # 90 bp
    orf2 = "GCT" * 29 + "TAA"    # 90 bp
    orf3 = "ATG" + ("CTG" * 58) + "TAA"  # 180 bp
    intron = make_complex_seq(100, seed=9881)
    ins90 = "GGT" * 30
    un100 = make_complex_seq(100, seed=9882)

    # Build Chr01 with multi-isoform GFF3 (R2-D12) + centromere + unresolved
    gfa1 = tmp_path / "d9_c1.gfa"
    gfa1.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{orf1}",
                f"S\t2\t{orf2}",
                f"S\t3\t{intron}",
                f"S\t4\t{orf3}",
                f"S\t5\t{ins90}",
                f"S\t6\t{un100}",
                "L\t6\t+\t1\t+\t0M",
                "L\t1\t+\t2\t+\t0M",
                "L\t1\t+\t5\t+\t0M",
                "L\t5\t+\t2\t+\t0M",
                "L\t2\t+\t3\t+\t0M",
                "L\t3\t+\t4\t+\t0M",
                "P\tRef#1#Chr01\t1+,2+,3+,4+\t*",
                "P\tAlt#1#Chr01\t6+,1+,5+,2+,3+,4+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref_c1 = orf1 + orf2 + intron + orf3
    fa1 = tmp_path / "d9_c1.fa"
    fa1.write_text(f">Chr01\n{ref_c1}\n", encoding="utf-8")
    bed1 = tmp_path / "d9_c1.bed"
    bed1.write_text("Chr01\t180\t280\tCEN1\n", encoding="utf-8")

    # tx1 has CDS in exon 1 (1..180) and exon 2 (281..460); tx2 shares exon 1 as 5'-UTR and only has CDS in exon 2!
    gff1 = tmp_path / "d9_c1.gff3"
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
    out1 = tmp_path / "out_d9_c1"
    assert (
        cli_main(
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
            ]
        )
        == 0
    )

    # Check R2-D12: tx1 gets cds_split_by_insertion, tx2 gets ONLY utr_insertion!
    lifted_lines = Path(f"{out1}.ref.lifted.gff3").read_text(encoding="utf-8").splitlines()
    attrs_by_id: dict[str, dict[str, str]] = {}
    for ln in lifted_lines:
        if ln and not ln.startswith("#"):
            at = dict(kv.split("=", 1) for kv in ln.split("\t")[8].split(";") if "=" in kv)
            if "ID" in at:
                attrs_by_id[at["ID"]] = at
    assert attrs_by_id["tx1"]["sv_status"] == "cds_split_by_insertion"
    assert attrs_by_id["tx2"]["sv_status"] == "utr_insertion"

    # Check R2-D9: malformed --ref-fa variants & duplicate/corrupt unresolved.fa
    for idx_bad, bad_fa_txt in enumerate(
        [
            f"GARBAGE_PRE_HEADER\n>Chr01\n{ref_c1}\n",
            f">EmptySkipped#1#Chr02\n>Ref#1#Chr01\n{ref_c1}\n",
            f">Skipped#1#Chr02\nACGT12345\n>Ref#1#Chr01\n{ref_c1}\n",
            f">Ref#1#Chr01\n{ref_c1[:-5]}12345\n",
        ]
    ):
        bad_fa = tmp_path / f"bad_fa_{idx_bad}.fa"
        bad_fa.write_text(bad_fa_txt, encoding="utf-8")
        rep_bf = verify_build_outputs(out1, gfa1, bad_fa)
        assert not rep_bf.passed and not rep_bf.invariant_1_passed

    orig_ufa = Path(f"{out1}.unresolved.fa").read_text(encoding="utf-8")
    Path(f"{out1}.unresolved.fa").write_text(
        ">Chr01_UNRES_1\nGARBAGE_12345\n" + orig_ufa, encoding="utf-8"
    )
    rep_ufa = verify_build_outputs(out1, gfa1, fa1)
    assert not rep_ufa.passed and not rep_ufa.unresolved_passed
    Path(f"{out1}.unresolved.fa").write_text(orig_ufa, encoding="utf-8")

    # Check R2-D10: >10.4 MB GFA with Unicode digit '\uff12' in W walk & invalid '*' in S line
    pad10m = "A" * 10_500_000
    gfa_uni = tmp_path / "uni_10m.gfa"
    gfa_uni.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{orf1}",
                f"S\t2\t{orf2}",
                f"S\t3\t{intron}",
                f"S\t4\t{orf3}",
                f"S\t5\t{ins90}",
                f"S\t6\t{un100}",
                f"S\t7\t{pad10m}",
                "P\tRef#1#Chr01\t1+,2+,3+,4+\t*",
                "P\tAlt#1#Chr01\t6+,1+,5+,2+,3+,4+\t*",
                "W\tExtra\t1\tChr01\t0\t90\t>1\uff12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert not verify_build_outputs(out1, gfa_uni, fa1).passed

    gfa_star = tmp_path / "star_10m.gfa"
    gfa_star.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{orf1}\nS\t2\t*\nS\t3\t{pad10m}\nP\tRef#1#Chr01\t1+\t*\n",
        encoding="utf-8",
    )
    assert not verify_build_outputs(out1, gfa_star, fa1).passed

    # Check R2-D11: Merge Chr01 (centromere active + ID=tx1) with Chr02 (centromere off + colliding ID=tx1)
    gfa2 = tmp_path / "d11_c2.gfa"
    gfa2.write_text(f"H\tVN:Z:1.0\nS\t10\t{orf3}\nP\tRef#1#Chr02\t10+\t*\n", encoding="utf-8")
    fa2 = tmp_path / "d11_c2.fa"
    fa2.write_text(f">Chr02\n{orf3}\n", encoding="utf-8")
    gff2 = tmp_path / "d11_c2.gff3"
    gff2.write_text(
        "##gff-version 3\n"
        "Chr02\tsrc\tgene\t1\t180\t.\t+\t.\tID=g1\n"
        "Chr02\tsrc\tmRNA\t1\t180\t.\t+\t.\tID=tx1;Parent=g1\n"
        "Chr02\tsrc\texon\t1\t180\t.\t+\t.\tID=ex_c2;Parent=tx1\n"
        "Chr02\tsrc\tCDS\t1\t180\t.\t+\t0\tID=cds_c2;Parent=tx1\n",
        encoding="utf-8",
    )
    out2 = tmp_path / "out_d11_c2"
    assert (
        cli_main(
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
            ]
        )
        == 0
    )
    merged_d11 = tmp_path / "merged_d11"
    merge_chromosomes([out1, out2], merged_d11)
    comb_gfa = tmp_path / "comb_d11.gfa"
    comb_gfa.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{orf1}",
                f"S\t2\t{orf2}",
                f"S\t3\t{intron}",
                f"S\t4\t{orf3}",
                f"S\t5\t{ins90}",
                f"S\t6\t{un100}",
                f"S\t10\t{orf3}",
                "P\tRef#1#Chr01\t1+,2+,3+,4+\t*",
                "P\tAlt#1#Chr01\t6+,1+,5+,2+,3+,4+\t*",
                "P\tRef#1#Chr02\t10+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    comb_fa = tmp_path / "comb_d11.fa"
    comb_fa.write_text(f">Chr01\n{ref_c1}\n>Chr02\n{orf3}\n", encoding="utf-8")
    comb_gff = tmp_path / "comb_d11.gff3"
    comb_gff.write_text(
        gff1.read_text(encoding="utf-8")
        + "\n".join(gff2.read_text(encoding="utf-8").splitlines()[1:])
        + "\n",
        encoding="utf-8",
    )

    rep_d11_ok = verify_build_outputs(
        merged_d11, comb_gfa, comb_fa, ref_gff_path=comb_gff
    )
    assert rep_d11_ok.passed and rep_d11_ok.centromere_passed and rep_d11_ok.invariant_3_passed

    # Corrupting Chr01's lifted CDS coordinate (despite colliding ID=tx1 on Chr02!) must fail INV3!
    orig_lgff = Path(f"{merged_d11}.ref.lifted.gff3").read_text(encoding="utf-8")
    Path(f"{merged_d11}.ref.lifted.gff3").write_text(
        orig_lgff.replace("Chr01\tsrc\tCDS\t1\t90", "Chr01\tsrc\tCDS\t4\t90", 1),
        encoding="utf-8",
    )
    rep_gff_bad = verify_build_outputs(
        merged_d11, comb_gfa, comb_fa, ref_gff_path=comb_gff
    )
    assert not rep_gff_bad.passed and not rep_gff_bad.invariant_3_passed
    Path(f"{merged_d11}.ref.lifted.gff3").write_text(orig_lgff, encoding="utf-8")

    # Deleting merged centromere.tsv without passing --centromere-bed must fail CEN!
    Path(f"{merged_d11}.centromere.tsv").unlink()
    rep_cen_missing = verify_build_outputs(merged_d11, comb_gfa, comb_fa)
    assert not rep_cen_missing.passed and not rep_cen_missing.centromere_passed


def test_m7_30_p1_1_gff_gene_multi_isoform_sv_status_union_without_sibling_bleed(
    tmp_path: Path,
) -> None:
    """Task 1.1 (§4 rule 4): Parent `gene` container unions `sv_status` across alternatively spliced
    child isoforms (`mRNA`/`transcript`) in canonical order (`intronic_insertion`, `utr_insertion`,
    `cds_split_by_insertion`), while leaf isoforms remain isolated against shared-exon bleed.
    Covers:
    1. Shared-exon gene (`g_shared`): `tx_s1` CDS split + `tx_s2` 5'-UTR insertion -> `g_shared` has
       `utr_insertion,cds_split_by_insertion`.
    2. Cassette-exon skipping gene (`g_cassette`): `tx_c1` CDS split + `tx_c2` intronic insertion ->
       `g_cassette` has `intronic_insertion,cds_split_by_insertion`.
    3. Triple-role gene (`g_triple`): `tx_t1` intronic + `tx_t2` UTR + `tx_t3` CDS at the same insertion
       point -> `g_triple` has `intronic_insertion,utr_insertion,cds_split_by_insertion`.
    """
    # Reference layout (990 bp total across 9 nodes):
    # Gene 1 (g_shared): [0, 330) -> n1 [0, 90), n2 [90, 180), intr1 [180, 240), n3 [240, 330); INS1 at 90
    # Gene 2 (g_cassette): [330, 660) -> n4 [330, 420), intr2a [420, 450), cas_a [450, 495), cas_b [495, 540),
    #                      intr2b [540, 570), n5 [570, 660); INS2 at 495
    # Gene 3 (g_triple): [660, 990) -> n6 [660, 750), intr3a [750, 780), tri_a [780, 825), tri_b [825, 870),
    #                    intr3b [870, 900), n7 [900, 990); INS3 at 825
    n1 = "ATG" + ("GCT" * 29)  # 90 bp
    n2 = ("GCT" * 29) + "TAA"  # 90 bp
    intr1 = make_complex_seq(60, seed=9901)
    n3 = "ATG" + ("CTG" * 28) + "TAA"  # 90 bp

    n4 = "ATG" + ("CAG" * 29)  # 90 bp
    intr2a = make_complex_seq(30, seed=9902)
    cas_a = "GAC" * 15  # 45 bp
    cas_b = "GAC" * 15  # 45 bp
    intr2b = make_complex_seq(30, seed=9903)
    n5 = ("CAG" * 29) + "TGA"  # 90 bp

    n6 = "ATG" + ("TCC" * 29)  # 90 bp
    intr3a = make_complex_seq(30, seed=9904)
    tri_a = "AAG" * 15  # 45 bp
    tri_b = "AAG" * 15  # 45 bp
    intr3b = make_complex_seq(30, seed=9905)
    n7 = ("TCC" * 29) + "TAG"  # 90 bp

    ins1 = make_complex_seq(90, seed=9911)
    ins2 = make_complex_seq(90, seed=9912)
    ins3 = make_complex_seq(90, seed=9913)

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
        ("n6", n6),
        ("intr3a", intr3a),
        ("tri_a", tri_a),
        ("tri_b", tri_b),
        ("intr3b", intr3b),
        ("n7", n7),
        ("ins1", ins1),
        ("ins2", ins2),
        ("ins3", ins3),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path(
        "Ref#1#Chr01",
        [
            "n1+",
            "n2+",
            "intr1+",
            "n3+",
            "n4+",
            "intr2a+",
            "cas_a+",
            "cas_b+",
            "intr2b+",
            "n5+",
            "n6+",
            "intr3a+",
            "tri_a+",
            "tri_b+",
            "intr3b+",
            "n7+",
        ],
    )
    b.add_path(
        "Alt#1#Chr01",
        [
            "n1+",
            "ins1+",
            "n2+",
            "intr1+",
            "n3+",
            "n4+",
            "intr2a+",
            "cas_a+",
            "ins2+",
            "cas_b+",
            "intr2b+",
            "n5+",
            "n6+",
            "intr3a+",
            "tri_a+",
            "ins3+",
            "tri_b+",
            "intr3b+",
            "n7+",
        ],
    )

    gff_lines = [
        # Gene 1: shared exon 1..180 (CDS in tx_s1, 5'-UTR in tx_s2)
        "Chr01\tsrc\tgene\t1\t330\t.\t+\t.\tID=g_shared",
        "Chr01\tsrc\tmRNA\t1\t330\t.\t+\t.\tID=tx_s1;Parent=g_shared",
        "Chr01\tsrc\tmRNA\t1\t330\t.\t+\t.\tID=tx_s2;Parent=g_shared",
        "Chr01\tsrc\texon\t1\t180\t.\t+\t.\tID=ex_s1;Parent=tx_s1,tx_s2",
        "Chr01\tsrc\texon\t241\t330\t.\t+\t.\tID=ex_s2;Parent=tx_s1,tx_s2",
        "Chr01\tsrc\tCDS\t1\t180\t.\t+\t0\tID=cds_s1;Parent=tx_s1",
        "Chr01\tsrc\tthree_prime_UTR\t241\t330\t.\t+\t.\tID=utr_s1;Parent=tx_s1",
        "Chr01\tsrc\tfive_prime_UTR\t1\t180\t.\t+\t.\tID=utr_s2;Parent=tx_s2",
        "Chr01\tsrc\tCDS\t241\t330\t.\t+\t0\tID=cds_s2;Parent=tx_s2",
        # Gene 2: cassette exon 451..540 included in tx_c1 (CDS) and skipped in tx_c2 (intronic)
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
        # Gene 3: 3 isoforms (tx_t1 skips 781..870 -> intronic; tx_t2 uses 781..870 as UTR; tx_t3 uses 781..870 as CDS)
        "Chr01\tsrc\tgene\t661\t990\t.\t+\t.\tID=g_triple",
        "Chr01\tsrc\tmRNA\t661\t990\t.\t+\t.\tID=tx_t1;Parent=g_triple",
        "Chr01\tsrc\tmRNA\t661\t990\t.\t+\t.\tID=tx_t2;Parent=g_triple",
        "Chr01\tsrc\tmRNA\t661\t990\t.\t+\t.\tID=tx_t3;Parent=g_triple",
        "Chr01\tsrc\texon\t661\t750\t.\t+\t.\tID=ex_t1;Parent=tx_t1,tx_t2,tx_t3",
        "Chr01\tsrc\texon\t781\t870\t.\t+\t.\tID=ex_tri;Parent=tx_t2,tx_t3",
        "Chr01\tsrc\texon\t901\t990\t.\t+\t.\tID=ex_t3;Parent=tx_t1,tx_t2,tx_t3",
        "Chr01\tsrc\tCDS\t661\t750\t.\t+\t0\tID=cds_t1a;Parent=tx_t1",
        "Chr01\tsrc\tCDS\t901\t990\t.\t+\t0\tID=cds_t1b;Parent=tx_t1",
        "Chr01\tsrc\tfive_prime_UTR\t661\t750\t.\t+\t.\tID=utr_t2a;Parent=tx_t2",
        "Chr01\tsrc\tfive_prime_UTR\t781\t870\t.\t+\t.\tID=utr_t2b;Parent=tx_t2",
        "Chr01\tsrc\tCDS\t901\t990\t.\t+\t0\tID=cds_t2c;Parent=tx_t2",
        "Chr01\tsrc\tCDS\t661\t750\t.\t+\t0\tID=cds_t3a;Parent=tx_t3",
        "Chr01\tsrc\tCDS\t781\t870\t.\t+\t0\tID=cds_t3b;Parent=tx_t3",
        "Chr01\tsrc\tCDS\t901\t990\t.\t+\t0\tID=cds_t3c;Parent=tx_t3",
    ]
    b.gff_lines.extend(gff_lines)

    inputs = b.write("p1_1_gff_union")
    _, bundle = run_build(inputs, extra_args=["--ref-gff", str(inputs.gff_path)])
    assert bundle is not None
    rep = verify_build_outputs(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        ref_gff_path=inputs.gff_path,
    )
    assert rep.passed and rep.invariant_3_passed

    assert bundle.lifted_gff_rows is not None
    by_id = {
        r["attributes"]["ID"]: r["attributes"]
        for r in bundle.lifted_gff_rows
        if r["type"] in ("gene", "mRNA")
    }

    # Leaf isoforms stay strictly isolated (no shared-exon sibling bleed!)
    assert by_id["tx_s1"]["sv_status"] == "cds_split_by_insertion"
    assert by_id["tx_s2"]["sv_status"] == "utr_insertion"
    assert by_id["tx_c1"]["sv_status"] == "cds_split_by_insertion"
    assert by_id["tx_c2"]["sv_status"] == "intronic_insertion"
    assert by_id["tx_t1"]["sv_status"] == "intronic_insertion"
    assert by_id["tx_t2"]["sv_status"] == "utr_insertion"
    assert by_id["tx_t3"]["sv_status"] == "cds_split_by_insertion"

    # Parent genes union all child isoforms' sv_status in canonical §4 order!
    assert by_id["g_shared"]["sv_status"] == "utr_insertion,cds_split_by_insertion"
    assert by_id["g_shared"]["sv_id"] == "Chr01_INS_1"
    assert by_id["g_cassette"]["sv_status"] == "intronic_insertion,cds_split_by_insertion"
    assert by_id["g_cassette"]["sv_id"] == "Chr01_INS_2"
    assert (
        by_id["g_triple"]["sv_status"]
        == "intronic_insertion,utr_insertion,cds_split_by_insertion"
    )
    assert by_id["g_triple"]["sv_id"] == "Chr01_INS_3"


def test_m7_31_p1_2_multiallelic_ins_per_allele_tag_isolation(tmp_path: Path) -> None:
    """Task 1.2 (§3.4.1, §3.5, §3.6.6, §5.1–5.2): Multi-allelic `INS` `alleles` tags
    (`partial_content=1`, `repeat_residual=1`, `multi_hit=1`) are strictly isolated per allele
    instead of broadcast across all alleles in `G(a, b)`.
    Covers:
    1. Independent heterologous alleles at `(r1, r2)`:
       - `CleanA#1`: clean 600 bp insertion (zero extra tags)
       - `PartialB#1`: 650 bp unique + 1550 bp low-complexity satellite tail -> `partial_content=1` only
       - `MultiHitC#1`: 520 bp node visited at two distant loci -> `multi_hit=1` only
       - `CycleRepE#1`: 310 bp node traversed twice in tandem (`n_cyc+ -> n_cyc+`) -> `repeat_residual=1,multi_hit=1`
       - `FullMaskD#1`: 1600 bp pure satellite completely masked -> genotype `.` (does not pollute `CleanA#1`)
    2. Recursive multi-member component (`A+B` vs `A+C_mh`) at `(r2, r3)`:
       - `CompClean1#1` & `CompClean2#1`: `compA + compB` (secondary backbone, clean) -> zero extra tags
       - `CompMH3#1`: `compA + compC_mh` (`compC_mh` visited twice globally) -> `multi_hit=1` only on `A+C_mh`!
    """
    r1 = make_complex_seq(250, seed=10101)
    r2 = make_complex_seq(250, seed=10102)
    r3 = make_complex_seq(250, seed=10103)
    r4 = make_complex_seq(250, seed=10104)

    seq_clean = make_complex_seq(600, seed=10111)
    seq_part_head = make_complex_seq(650, seed=10112)
    seq_part_tail = make_satellite_seq(unit_len=12, total_len=1550, seed=10113)
    seq_part_combined = seq_part_head + seq_part_tail
    seq_mh = make_complex_seq(520, seed=10114)
    sec_m = make_complex_seq(100, seed=10116)
    sec_b_extra = make_complex_seq(100, seed=10119)
    sec_u = make_complex_seq(100, seed=10117)
    sec_novel = make_complex_seq(100, seed=10118)
    seq_full_mask = make_satellite_seq(unit_len=11, total_len=1600, seed=10115)

    comp_a = make_complex_seq(320, seed=10121)
    comp_b = make_complex_seq(340, seed=10122)
    comp_c_mh = make_complex_seq(300, seed=10123)

    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seq in [
        ("r1", r1),
        ("r2", r2),
        ("r3", r3),
        ("r4", r4),
        ("n_clean", seq_clean),
        ("n_partial", seq_part_combined),
        ("n_mh", seq_mh),
        ("sec_m", sec_m),
        ("sec_b_extra", sec_b_extra),
        ("sec_u", sec_u),
        ("sec_novel", sec_novel),
        ("n_full_mask", seq_full_mask),
        ("compA", comp_a),
        ("compB", comp_b),
        ("compC_mh", comp_c_mh),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
    b.add_path("CleanA#1#Chr01", ["r1+", "n_clean+", "r2+", "r3+", "r4+"])
    b.add_path("PartialB#1#Chr01", ["r1+", "n_partial+", "r2+", "r3+", "r4+"])
    # r2+ is visited twice on MultiHitC#1 so anchor_counts['r2'] == 2 -> §3.4.1 multi_hit=1
    b.add_path("MultiHitC#1#Chr01", ["r1+", "n_mh+", "r2+", "r3+", "r2+", "r4+"])
    # B_sec = sec_m+(0) -> sec_b_extra+(1) -> sec_m+(2) -> sec_u+(3) has two copies of sec_m inside [0, 3),
    # so SecRepE#1's [sec_m+, sec_novel+] before sec_u+ is locally undecomposable -> §3.5 repeat_residual=1
    b.add_path(
        "SecBack1#1#Chr01",
        ["r1+", "sec_m+", "sec_b_extra+", "sec_m+", "sec_u+", "r2+", "r3+", "r4+"],
    )
    b.add_path(
        "SecBack2#1#Chr01",
        ["r1+", "sec_m+", "sec_b_extra+", "sec_m+", "sec_u+", "r2+", "r3+", "r4+"],
    )
    b.add_path(
        "SecRepE#1#Chr01",
        ["r1+", "sec_m+", "sec_novel+", "sec_u+", "r2+", "r3+", "r4+"],
    )
    b.add_path("FullMaskD#1#Chr01", ["r1+", "n_full_mask+", "r2+", "r3+", "r4+"])

    # Recursive multi-member component at (r2, r3): CompClean1 & CompClean2 carry compA+compB;
    # CompMH3 carries compA+compC_mh and visits backbone anchor r2+ twice -> multi_hit=1 on CompMH3 only
    b.add_path("CompClean1#1#Chr01", ["r1+", "r2+", "compA+", "compB+", "r3+", "r4+"])
    b.add_path("CompClean2#1#Chr01", ["r1+", "r2+", "compA+", "compB+", "r3+", "r4+"])
    b.add_path(
        "CompMH3#1#Chr01",
        ["r1+", "r2+", "compA+", "compC_mh+", "r3+", "r2+", "r4+"],
    )

    inputs = b.write("p1_2_allele_tags")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ins_by_ref = {
        int(ev["ref_start"]): ev for ev in bundle.sv_rows if ev["type"] == "INS"
    }
    # Locus 1 at ref_start = 250
    ev_loc1 = ins_by_ref[250]
    gt_loc1 = bundle.genotypes[ev_loc1["event_id"]]
    assert gt_loc1["FullMaskD#1"] == "."
    assert gt_loc1["SecBack1#1"] == gt_loc1["SecBack2#1"]
    carrier_gts = {
        gt_loc1["CleanA#1"],
        gt_loc1["PartialB#1"],
        gt_loc1["MultiHitC#1"],
        gt_loc1["SecBack1#1"],
        gt_loc1["SecRepE#1"],
    }
    assert len(carrier_gts) == 5 and not (carrier_gts & {"0", "."})

    allele_map_1 = dict(
        entry.split(":", 1) for entry in str(ev_loc1["alleles"]).split(";")
    )
    clean_meta = allele_map_1[gt_loc1["CleanA#1"]]
    sec_back_meta = allele_map_1[gt_loc1["SecBack1#1"]]
    partial_meta = allele_map_1[gt_loc1["PartialB#1"]]
    mh_meta = allele_map_1[gt_loc1["MultiHitC#1"]]
    rep_res_meta = allele_map_1[gt_loc1["SecRepE#1"]]

    for clean_m in (clean_meta, sec_back_meta):
        assert "partial_content=1" not in clean_m
        assert "repeat_residual=1" not in clean_m
        assert "multi_hit=1" not in clean_m

    assert "partial_content=1" in partial_meta
    assert "repeat_residual=1" not in partial_meta
    assert "multi_hit=1" not in partial_meta

    assert "multi_hit=1" in mh_meta
    assert "partial_content=1" not in mh_meta
    assert "repeat_residual=1" not in mh_meta

    assert "repeat_residual=1" in rep_res_meta
    assert "multi_hit=1" not in rep_res_meta
    assert "partial_content=1" not in rep_res_meta

    # Locus 2 (recursive component A+B vs A+C_mh) at ref_start = 500
    ev_loc2 = ins_by_ref[500]
    gt_loc2 = bundle.genotypes[ev_loc2["event_id"]]
    assert gt_loc2["CompClean1#1"] == gt_loc2["CompClean2#1"] == "1"
    assert gt_loc2["CompMH3#1"] == "2"

    allele_map_2 = dict(
        entry.split(":", 1) for entry in str(ev_loc2["alleles"]).split(";")
    )
    assert "multi_hit=1" not in allele_map_2["1"]
    assert "partial_content=1" not in allele_map_2["1"]
    assert "repeat_residual=1" not in allele_map_2["1"]
    assert "multi_hit=1" in allele_map_2["2"]


def _shift_bundle_pan_after(prefix: Path, insert_pan_pos: int, extra_n: int) -> None:
    """Insert `'N' * extra_n` into `{prefix}.pan.fa` at `insert_pan_pos` and shift all downstream
    coordinates in `offsets.tsv`, `ref_to_pan.chain`, `sv.tsv`, and `inserts.tsv` by `+extra_n`
    so Global Invariants I, II, and III remain satisfied while mount-point spacer layout is inflated.
    """
    # 1. pan.fa
    pan_lines = Path(f"{prefix}.pan.fa").read_text(encoding="utf-8").splitlines()
    hdr = pan_lines[0]
    seq = "".join(pan_lines[1:])
    new_seq = seq[:insert_pan_pos] + ("N" * extra_n) + seq[insert_pan_pos:]
    Path(f"{prefix}.pan.fa").write_text(f"{hdr}\n{new_seq}\n", encoding="utf-8")

    # 2. offsets.tsv
    off_lines = Path(f"{prefix}.offsets.tsv").read_text(encoding="utf-8").splitlines()
    oh = {k: i for i, k in enumerate(off_lines[0].lstrip("#").split("\t"))}
    new_off = [off_lines[0]]
    step_rows_for_chain: list[tuple[int, int, int, int, int, int]] = []
    for line in off_lines[1:]:
        cols = line.split("\t")
        p_s = int(cols[oh["pan_start"]])
        p_e = int(cols[oh["pan_end"]])
        s_ins = int(cols[oh["slot_ins_len"]])
        prev_pe_before = p_s - s_ins
        if p_s >= insert_pan_pos:
            p_s += extra_n
            p_e += extra_n
            if prev_pe_before <= insert_pan_pos:
                s_ins += extra_n
        cols[oh["pan_start"]] = str(p_s)
        cols[oh["pan_end"]] = str(p_e)
        cols[oh["slot_ins_len"]] = str(s_ins)
        new_off.append("\t".join(cols))
        step_rows_for_chain.append(
            (
                int(cols[oh["ref_step"]]),
                int(cols[oh["ref_start"]]),
                int(cols[oh["ref_end"]]),
                p_s,
                p_e,
                s_ins,
            )
        )
    Path(f"{prefix}.offsets.tsv").write_text("\n".join(new_off) + "\n", encoding="utf-8")

    # 3. Rebuild consistent ref_to_pan.chain from updated offsets + new_seq len
    ch_lines = Path(f"{prefix}.ref_to_pan.chain").read_text(encoding="utf-8").splitlines()
    ch_hdr_parts: list[str] = []
    for line in ch_lines:
        if line.startswith("chain "):
            ch_hdr_parts = line.split()
            break
    ch_hdr_parts[8] = str(len(new_seq))
    ch_hdr_parts[11] = str(step_rows_for_chain[-1][4])

    # Coalesce contiguous zero-slot steps into chain blocks
    blocks: list[tuple[int, int, int]] = []  # (r_s, r_e, dq_after)
    cur_rs = step_rows_for_chain[0][1]
    cur_re = step_rows_for_chain[0][2]
    for st_idx in range(1, len(step_rows_for_chain)):
        _, r_s, r_e, _, _, s_ins = step_rows_for_chain[st_idx]
        if s_ins > 0:
            blocks.append((cur_rs, cur_re, s_ins))
            cur_rs = r_s
            cur_re = r_e
        else:
            cur_re = r_e
    blocks.append((cur_rs, cur_re, 0))

    rebuilt_chain = [" ".join(ch_hdr_parts)]
    for b_rs, b_re, b_dq in blocks:
        rebuilt_chain.append(f"# block\t{b_rs}\t{b_re}\t{b_dq}")
        if b_dq > 0:
            rebuilt_chain.append(f"{b_re - b_rs}\t0\t{b_dq}")
        else:
            rebuilt_chain.append(f"{b_re - b_rs}")
    for st_tup in step_rows_for_chain:
        rebuilt_chain.append(
            f"# step\t{st_tup[0]}\t{st_tup[1]}\t{st_tup[2]}\t{st_tup[3]}\t{st_tup[4]}\t{st_tup[5]}"
        )
    Path(f"{prefix}.ref_to_pan.chain").write_text(
        "\n".join(rebuilt_chain) + "\n\n", encoding="utf-8"
    )

    # 4. inserts.tsv
    ins_lines = Path(f"{prefix}.inserts.tsv").read_text(encoding="utf-8").splitlines()
    new_ins = [ins_lines[0]]
    ev_spans: dict[str, list[int]] = {}
    for line in ins_lines[1:]:
        cols = line.split("\t")
        ps = int(cols[1])
        pe = int(cols[2])
        if ps >= insert_pan_pos:
            ps += extra_n
            pe += extra_n
        cols[1] = str(ps)
        cols[2] = str(pe)
        ev_id = cols[3]
        if ev_id not in ev_spans:
            ev_spans[ev_id] = [ps, pe]
        else:
            ev_spans[ev_id][0] = min(ev_spans[ev_id][0], ps)
            ev_spans[ev_id][1] = max(ev_spans[ev_id][1], pe)
        new_ins.append("\t".join(cols))
    Path(f"{prefix}.inserts.tsv").write_text("\n".join(new_ins) + "\n", encoding="utf-8")

    # 5. sv.tsv
    sv_lines = Path(f"{prefix}.sv.tsv").read_text(encoding="utf-8").splitlines()
    new_sv = [sv_lines[0]]
    for line in sv_lines[1:]:
        cols = line.split("\t")
        ev_id = cols[3]
        if ev_id in ev_spans:
            cols[1] = str(ev_spans[ev_id][0])
            cols[2] = str(ev_spans[ev_id][1])
        else:
            ps = int(cols[1])
            pe = int(cols[2])
            if ps >= insert_pan_pos:
                cols[1] = str(ps + extra_n)
                cols[2] = str(pe + extra_n)
        new_sv.append("\t".join(cols))
    Path(f"{prefix}.sv.tsv").write_text("\n".join(new_sv) + "\n", encoding="utf-8")


def test_m7_32_p1_3_verify_centromere_mount_exact_local_offset_and_stacked_spacer_rejection(
    tmp_path: Path,
) -> None:
    """Task 1.3 (§3.8.4-D & §3.8.5): `verify_build_outputs` enforces exact local offset delta
    `local_offset_delta == slot_ins_len == cen_bp + normal_ins_bp + expected_spacer_bp` and single
    non-stacked spacers at centromere mount points, even when all global Invariants I–III are kept
    consistent and an upstream insertion creates non-zero cumulative offset.
    """
    spacer_n = 100
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r0", make_complex_seq(250, seed=10201))
    b.add_node("u", make_complex_seq(250, seed=10202))
    b.add_node("cen_ref", make_complex_seq(400, seed=10203))
    b.add_node("v", make_complex_seq(250, seed=10204))
    b.add_node("r_end", make_complex_seq(250, seed=10205))
    b.add_node("up_ins", make_complex_seq(180, seed=10206))
    b.add_node("cen_isl", make_complex_seq(620, seed=10207))
    b.add_node("co_ins", make_complex_seq(210, seed=10208))

    # Ref: r0 [0,250), u [250,500), cen_ref [500,900), v [900,1150), r_end [1150,1400)
    b.set_ref_path("Ref#1#Chr01", ["r0+", "u+", "cen_ref+", "v+", "r_end+"])
    # Hap1 carries upstream INS at 250 AND centromere island [500, 900) mounted at 900
    b.add_path("HapCen#1#Chr01", ["r0+", "up_ins+", "u+", "cen_isl+", "v+", "r_end+"])
    # Hap2 partial contig starting at cen_ref+ (without u+) carries co-mounted normal INS at (cen_ref, v) -> ref_start(v) = 900
    b.add_path("HapNorm#1#Chr01", ["cen_ref+", "co_ins+", "v+", "r_end+"])
    b.add_bed_region(520, 880, "CEN_MAIN")

    inputs = b.write("p1_3_cen_mount")
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

    rep_ok = verify_build_outputs(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        centromere_bed_path=inputs.bed_path,
        spacer_n=spacer_n,
    )
    assert rep_ok.passed and rep_ok.centromere_passed

    # Locate the co-mounted normal INS piece at ref=900
    norm_ev = next(
        ev for ev in bundle.sv_rows if ev["type"] == "INS" and int(ev["ref_start"]) == 900
    )
    norm_piece = next(
        ins for ins in bundle.inserts if ins["event_id"] == norm_ev["event_id"]
    )
    norm_ps = int(norm_piece["pan_start"])
    norm_pe = int(norm_piece["pan_end"])

    # Tamper A: Inject stacked 2*spacer_n between CEN_ISLAND and co-mounted normal INS
    import shutil

    t1_dir = tmp_path / "tamper_stacked_between"
    t1_dir.mkdir()
    for fp in inputs.out_prefix.parent.glob(f"{inputs.out_prefix.name}.*"):
        shutil.copy2(fp, t1_dir / fp.name)
    t1_pref = t1_dir / inputs.out_prefix.name
    _shift_bundle_pan_after(t1_pref, norm_ps, spacer_n)

    rep_t1 = verify_build_outputs(
        t1_pref,
        inputs.gfa_path,
        inputs.ref_fa_path,
        centromere_bed_path=inputs.bed_path,
        spacer_n=spacer_n,
    )
    # Global Invariants I, II, III all pass, ONLY Section 5 exact mount-point check catches it!
    assert rep_t1.invariant_1_passed
    assert rep_t1.invariant_2_passed
    assert rep_t1.invariant_3_passed
    assert not rep_t1.centromere_passed
    assert not rep_t1.passed

    # Tamper B: Inject unexpected trailing spacer_n after co-mounted normal INS before anchor v
    t2_dir = tmp_path / "tamper_trailing_after_norm"
    t2_dir.mkdir()
    for fp in inputs.out_prefix.parent.glob(f"{inputs.out_prefix.name}.*"):
        shutil.copy2(fp, t2_dir / fp.name)
    t2_pref = t2_dir / inputs.out_prefix.name
    _shift_bundle_pan_after(t2_pref, norm_pe, spacer_n)

    rep_t2 = verify_build_outputs(
        t2_pref,
        inputs.gfa_path,
        inputs.ref_fa_path,
        centromere_bed_path=inputs.bed_path,
        spacer_n=spacer_n,
    )
    assert rep_t2.invariant_1_passed
    assert rep_t2.invariant_2_passed
    assert rep_t2.invariant_3_passed
    assert not rep_t2.centromere_passed
    assert not rep_t2.passed


def test_m7_33_p1_4_unresolved_fa_header_strand_and_verify_tamper_rejection(
    tmp_path: Path,
) -> None:
    """Task 1.4 (§5 & §5.5): `write_build_outputs` and `merge_chromosomes` emit standalone
    `reason=<...> src=<path>:<start>-<end> strand=<±>` tokens in `{p}.unresolved.fa`, and
    `verify_build_outputs` rejects tampered `strand=`, `src=`, `reason=`, or orphan FASTA records.
    """
    r1 = make_complex_seq(250, seed=10301)
    r2 = make_complex_seq(250, seed=10302)
    un_end_seq = make_complex_seq(160, seed=10303)
    sat_seq = make_satellite_seq(unit_len=12, total_len=1600, seed=10304)

    # Chr01: reversed Alt path so UNANCHORED_END has src_strand='-' after top-level orientation,
    # plus --dump-repeat-sequences exporting REPEAT_LIKE!
    b1 = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b1.add_node("r1", r1)
    b1.add_node("r2", r2)
    b1.add_node("un1", un_end_seq)
    b1.add_node("sat1", sat_seq)
    b1.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    # Reversed contig: r2- -> sat1- -> r1- -> un1- (oriented to +: un1+ -> r1+ -> sat1+ -> r2+ with src_strand='-')
    b1.add_path("RevAlt#1#Chr01", ["r2-", "sat1-", "r1-", "un1-"])
    in1 = b1.write("p1_4_c1")
    _, bun1 = run_build(in1, extra_args=["--dump-repeat-sequences"])
    assert bun1 is not None
    assert len(bun1.unresolved_fa) == 2

    for exp_id, (hdr_str, _) in bun1.unresolved_fa.items():
        toks = hdr_str.split()
        assert toks[0] == exp_id
        kv = dict(t.split("=", 1) for t in toks[1:])
        assert kv["strand"] == "-"
        assert kv["reason"] in ("UNANCHORED_END", "REPEAT_LIKE")
        assert "(" not in kv["src"] and ":" in kv["src"]

    # Chr02: forward Alt path with '+' strand UNANCHORED_END, then merge Chr01 + Chr02
    b2 = SyntheticInputBuilder(tmp_path, chrom="Chr02", backbone_prefix="Ref#1#")
    b2.add_node("c2_r1", make_complex_seq(250, seed=10311))
    b2.add_node("c2_r2", make_complex_seq(250, seed=10312))
    b2.add_node("c2_un", make_complex_seq(140, seed=10313))
    b2.set_ref_path("Ref#1#Chr02", ["c2_r1+", "c2_r2+"])
    b2.add_path("FwdAlt#1#Chr02", ["c2_un+", "c2_r1+", "c2_r2+"])
    in2 = b2.write("p1_4_c2")
    _, bun2 = run_build(in2)
    assert bun2 is not None

    merged_pref = tmp_path / "p1_4_merged"
    merge_chromosomes([in1.out_prefix, in2.out_prefix], merged_pref)
    bun_m = OutputBundle(merged_pref)
    assert len(bun_m.unresolved_fa) == 3
    for exp_id, (hdr_str, _) in bun_m.unresolved_fa.items():
        kv = dict(t.split("=", 1) for t in hdr_str.split()[1:])
        expected_strand = "-" if exp_id.startswith("Chr01_") else "+"
        assert kv["strand"] == expected_strand
        assert "(" not in kv["src"]

    # Verify clean pass on Chr01, then test 4 distinct header/orphan tampers
    rep_clean = verify_build_outputs(in1.out_prefix, in1.gfa_path, in1.ref_fa_path)
    assert rep_clean.passed and rep_clean.unresolved_passed

    orig_ufa = Path(f"{in1.out_prefix}.unresolved.fa").read_text(encoding="utf-8")

    # 1. Tamper strand=- -> strand=+
    Path(f"{in1.out_prefix}.unresolved.fa").write_text(
        orig_ufa.replace("strand=-", "strand=+", 1), encoding="utf-8"
    )
    rep_bad_strand = verify_build_outputs(in1.out_prefix, in1.gfa_path, in1.ref_fa_path)
    assert not rep_bad_strand.passed and not rep_bad_strand.unresolved_passed

    # 2. Tamper src=... path/range
    Path(f"{in1.out_prefix}.unresolved.fa").write_text(
        orig_ufa.replace("src=RevAlt#1#Chr01:", "src=Tampered#1#Chr01:", 1),
        encoding="utf-8",
    )
    rep_bad_src = verify_build_outputs(in1.out_prefix, in1.gfa_path, in1.ref_fa_path)
    assert not rep_bad_src.passed and not rep_bad_src.unresolved_passed

    # 3. Tamper reason=...
    Path(f"{in1.out_prefix}.unresolved.fa").write_text(
        orig_ufa.replace("reason=UNANCHORED_END", "reason=INV_CONFLICT", 1),
        encoding="utf-8",
    )
    rep_bad_reason = verify_build_outputs(in1.out_prefix, in1.gfa_path, in1.ref_fa_path)
    assert not rep_bad_reason.passed and not rep_bad_reason.unresolved_passed

    # 4. Append orphan FASTA record not in unresolved.tsv
    Path(f"{in1.out_prefix}.unresolved.fa").write_text(
        orig_ufa
        + ">Chr01_UNRES_999 reason=UNANCHORED_END src=RevAlt#1#Chr01:0-10 strand=+\nACGTACGTAC\n",
        encoding="utf-8",
    )
    rep_orphan = verify_build_outputs(in1.out_prefix, in1.gfa_path, in1.ref_fa_path)
    assert not rep_orphan.passed and not rep_orphan.unresolved_passed

    # Restore clean file
    Path(f"{in1.out_prefix}.unresolved.fa").write_text(orig_ufa, encoding="utf-8")


def test_m7_34_fail_fast_preflight_gff_bed_and_verify_missing_files(
    tmp_path: Path,
) -> None:
    """Fail-fast pre-flight validation:
    1. `load_and_validate_inputs` rejects missing `--ref-gff` and `--centromere-bed`
       BEFORE scanning the GFA (even if the GFA itself is invalid/empty).
    2. `load_and_validate_inputs` validates `--ref-gff` seqid/coordinates/format
       against `(ref_chrom, len(ref_seq))` BEFORE `run_chromosome_pipeline` runs.
    3. `verify_build_outputs` rejects nonexistent explicit `ref_gff_path` and
       `centromere_bed_path` rather than silently skipping verification.
    """
    b = SyntheticInputBuilder(tmp_path / "preflight", chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("1", make_complex_seq(300, seed=9101))
    b.set_ref_path("Ref#1#Chr01", ["1+"])
    b.add_gff_line("Chr01", "src", "gene", 10, 200, ".", "+", ".", "ID=g1")
    b.add_bed_region(50, 150, "CEN1")
    inputs = b.write("pf_ok")

    # 1. Missing --ref-gff and --centromere-bed fail BEFORE scanning an empty/corrupt GFA
    empty_gfa = tmp_path / "empty.gfa"
    empty_gfa.write_text("", encoding="utf-8")

    with pytest.raises(InputContractError, match="Reference GFF3 file not found"):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(empty_gfa),
                ref_fa=str(inputs.ref_fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "o1"),
                ref_gff=str(tmp_path / "missing.gff3"),
            )
        )

    with pytest.raises(InputContractError, match="Centromere BED file not found"):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(empty_gfa),
                ref_fa=str(inputs.ref_fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "o2"),
                centromere_bed=str(tmp_path / "missing.bed"),
                centromere_mode="bypass",
            )
        )

    # 2. Mismatched seqid or out-of-range coordinate in --ref-gff fails inside load_and_validate_inputs
    wrong_seqid_gff = tmp_path / "wrong_seqid.gff3"
    wrong_seqid_gff.write_text(
        "##gff-version 3\nChr02\tsrc\tgene\t10\t100\t.\t+\t.\tID=g1\n",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError, match="does not match reference chromosome"):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(inputs.gfa_path),
                ref_fa=str(inputs.ref_fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "o3"),
                ref_gff=str(wrong_seqid_gff),
            )
        )

    oob_gff = tmp_path / "oob.gff3"
    oob_gff.write_text(
        "##gff-version 3\nChr01\tsrc\tgene\t10\t500\t.\t+\t.\tID=g1\n",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError, match="exceeds reference length"):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(inputs.gfa_path),
                ref_fa=str(inputs.ref_fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "o4"),
                ref_gff=str(oob_gff),
            )
        )

    # 3. Valid build + verify rejects nonexistent explicit ref_gff_path / centromere_bed_path
    cfg_ok = BuildConfig(
        gfa=str(inputs.gfa_path),
        ref_fa=str(inputs.ref_fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(inputs.out_prefix),
        ref_gff=str(inputs.gff_path),
        centromere_bed=str(inputs.bed_path),
        centromere_mode="bypass",
    )
    graph = load_and_validate_inputs(cfg_ok)
    res = run_chromosome_pipeline(graph, cfg_ok)
    write_build_outputs(res, cfg_ok)

    rep_missing_gff = verify_build_outputs(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        ref_gff_path=tmp_path / "no_such.gff3",
    )
    assert not rep_missing_gff.passed
    assert any("Reference GFF3 file not found" in e for e in rep_missing_gff.errors)

    rep_missing_bed = verify_build_outputs(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        centromere_bed_path=tmp_path / "no_such.bed",
    )
    assert not rep_missing_bed.passed
    assert any("Centromere BED file not found" in e for e in rep_missing_bed.errors)


def test_m7_35_centromere_step4b_coalesces_consecutive_cuts_no_shard_explosion(
    tmp_path: Path,
) -> None:
    """Regression test for chromosome-scale Step 4b fragment explosion:
    When a haplotype misses boundary anchors `u`/`v` (e.g. SNP-split PGGB nodes)
    and traverses hundreds of unique reference nodes inside `[ref_start, ref_end)`
    (including repeated internal nodes and an internal novel sequence), Step 4b MUST
    coalesce consecutive/overlapping `amb_cuts` into ONE cut producing only the 2
    outer chromosome arms (NOT hundreds of 1-node internal shards), while still
    recording the internal novel run as `CEN_BOUNDARY_AMBIGUOUS` and mounting outer
    arm insertions normally.
    """
    from panflat.centromere import (
        delimit_centromere_regions,
        partition_fragments_by_centromere,
    )

    n_cen_nodes = 200
    b = SyntheticInputBuilder(
        tmp_path / "cen_coalesce", chrom="Chr06A", backbone_prefix="Ref#1#"
    )
    # Integer node IDs trigger the LazyStepTuple / NumPy fast path:
    # 1, 2 = left arm; 3 = u; 4..203 = 200 internal cen nodes; 204 = v; 205, 206 = right arm
    # 300 = left arm insertion (120 bp), 301 = internal cen novel (140 bp), 302 = right arm insertion (130 bp)
    for nid in range(1, 207):
        b.add_node(str(nid), make_complex_seq(100, seed=10000 + nid))
    b.add_node("300", make_complex_seq(120, seed=20001))
    b.add_node("301", make_complex_seq(140, seed=20002))
    b.add_node("302", make_complex_seq(130, seed=20003))

    ref_steps = [f"{nid}+" for nid in range(1, 207)]
    b.set_ref_path("Ref#1#Chr06A", ref_steps)

    # Alt1 skips u (3) and v (204), lands on first/last internal nodes 4 & 203 (touching [300,20300) boundary)
    alt_steps = (
        ["1+", "300+", "2+"]
        + [f"{nid}+" for nid in range(4, 51)]
        + ["50+"]
        + [f"{nid}+" for nid in range(51, 101)]
        + ["301+"]
        + [f"{nid}+" for nid in range(101, 204)]
        + ["205+", "302+", "206+"]
    )
    b.add_path("AltNoUV#1#Chr06A", alt_steps)

    # Alt2 jumps strictly across [300, 20300) boundaries (2 -> 5 and 202 -> 205)
    alt2_steps = (
        ["1+", "300+", "2+"]
        + [f"{nid}+" for nid in range(5, 203)]
        + ["205+", "302+", "206+"]
    )
    b.add_path("AltCross#1#Chr06A", alt2_steps)

    # BED covers internal nodes 4..203 => [300, 20300); u=node 3 [200,300), v=node 204 [20300,20400)
    b.add_bed_region(350, 20250, "CEN06A")
    inputs = b.write("cen_coalesce_case")

    cfg = BuildConfig(
        gfa=str(inputs.gfa_path),
        ref_fa=str(inputs.ref_fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(inputs.out_prefix),
        centromere_bed=str(inputs.bed_path),
        centromere_mode="bypass",
    )
    graph = load_and_validate_inputs(cfg)
    regions, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regions, graph, cfg)

    # Each haplotype must produce ONLY 2 outer arms (total 4), NOT ~400 single-node shards!
    assert len(part.normal_fragments) == 4
    by_hap: dict[str, list] = {}
    for nf in part.normal_fragments:
        by_hap.setdefault(nf.hap_id, []).append(nf)
    assert [len(f.steps) for f in by_hap["AltNoUV#1"]] == [4, 4]
    assert [len(f.steps) for f in by_hap["AltCross#1"]] == [3, 3]

    # Internal novel sequence still routed to CEN_BOUNDARY_AMBIGUOUS
    assert len(part.boundary_ambiguous_records) == 1
    assert part.boundary_ambiguous_records[0].reason == "CEN_BOUNDARY_AMBIGUOUS"
    assert part.boundary_ambiguous_records[0].length == 140

    # End-to-end build & verify passes all Global Invariants and mounts both outer arm insertions
    res = run_chromosome_pipeline(graph, cfg)
    write_build_outputs(res, cfg)
    rep = verify_build_outputs(
        inputs.out_prefix,
        inputs.gfa_path,
        inputs.ref_fa_path,
        centromere_bed_path=inputs.bed_path,
    )
    assert rep.passed, f"verify failed: {rep.errors}"
    ins_events = [ev for ev in res.sv_events if ev.event_type == "INS"]
    assert len(ins_events) == 2
    assert n_cen_nodes == 200


