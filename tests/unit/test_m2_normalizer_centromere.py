"""Unit tests for M2 (`panflat/normalizer.py` and `panflat/centromere.py`).

Covers:
- Features: F7, F8, F9, F10, F11, F12, F13, F21, F22, F23, F24
- Scenarios: T6, T7, T8, T10, T15, T17, T23, T24, C2–C16, C18, C19
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from panflat.centromere import (
    delimit_centromere_regions,
    partition_fragments_by_centromere,
    process_centromere_regions,
)
from panflat.gfa_io import load_and_validate_inputs, spell_steps
from panflat.models import (
    BuildConfig,
    CNVCall,
    GFAGraph,
    InputContractError,
    PathFragment,
    ProjectedAnchor,
    Step,
    StepProvenance,
)
from panflat.normalizer import (
    build_backbone_anchor_map,
    canonical_rotation,
    compute_raw_visit_counts,
    compute_weighted_lis,
    detect_canonical_cnv,
    extract_deviations,
    fold_tandem_cycles,
    normalize_inversions_top_level,
    orient_contig_top_level,
    project_step_tuples_onto_backbone,
    project_steps_onto_backbone,
)
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    make_complex_seq,
    make_satellite_seq,
    make_window_with_exact_singleton_ratio,
    mutate_seq_snps,
    revcomp_dna,
)


def _make_prov(
    steps: Sequence[Step],
    nodes: Mapping[str, str],
    src_hap: str = "Alt#1",
    src_path: str = "Alt#1#ctg1",
    base_offset: int = 0,
    src_strand: str = "+",
) -> tuple[StepProvenance, ...]:
    out: list[StepProvenance] = []
    cur = base_offset
    for st in steps:
        slen = st.step_len(nodes)
        out.append(
            StepProvenance(
                src_hap=src_hap,
                src_path=src_path,
                src_start=cur,
                src_end=cur + slen,
                src_strand=src_strand,  # type: ignore[arg-type]
            )
        )
        cur += slen
    return tuple(out)


def _make_fragment(
    steps: Sequence[Step],
    nodes: Mapping[str, str],
    hap_id: str = "Alt#1",
    seqid: str = "ctg1",
    base_offset: int = 0,
    is_w_line: bool = False,
) -> PathFragment:
    sample, hap = hap_id.split("#", 1) if "#" in hap_id else (hap_id, "1")
    total_len = sum(st.step_len(nodes) for st in steps)
    if is_w_line:
        src_path = f"{sample}#{hap}#{seqid}:{base_offset}-{base_offset + total_len}"
    else:
        src_path = f"{sample}#{hap}#{seqid}"
    prov = _make_prov(
        steps, nodes, src_hap=hap_id, src_path=src_path, base_offset=base_offset
    )
    return PathFragment(
        sample=sample,
        hap=hap,
        hap_id=hap_id,
        seqid=seqid,
        src_path=src_path,
        base_offset=base_offset,
        steps=tuple(steps),
        provenance=prov,
        is_w_line=is_w_line,
        total_len=total_len,
    )


def _build_loaded_graph(
    tmp_path: Path,
    builder: SyntheticGFAFixtureBuilder,
    ref_path_name: str = "Ref#1#Chr01",
    ref_chrom: str = "Chr01",
    backbone_prefix: str = "Ref#1#",
    bed_rows: Sequence[tuple[str, int, int] | tuple[str, int, int, str]] | None = None,
    centromere_mode: str = "off",
    centromere_mode_explicit: bool = False,
    centromere_min_island: int = 500,
    min_sv_len: int = 50,
    max_jump: int = 500000,
    dump_repeat_sequences: bool = False,
) -> tuple[GFAGraph, BuildConfig]:
    gfa_file = builder.write_gfa(tmp_path / "test.gfa")
    # Find ref steps from builder
    ref_steps_str = ""
    for p_line in builder.p_lines:
        parts = p_line.split("\t")
        if parts[1] == ref_path_name:
            ref_steps_str = parts[2]
            break
    assert ref_steps_str, f"Reference path {ref_path_name} not found in builder"
    ref_tuples = [(tok[:-1], tok[-1]) for tok in ref_steps_str.split(",")]
    ref_seq = builder.spell_steps(ref_tuples)
    fa_file = builder.write_fasta(tmp_path / "ref.fa", {ref_chrom: ref_seq})

    bed_path_str: str | None = None
    if bed_rows is not None:
        bed_file = builder.write_bed(tmp_path / "cen.bed", bed_rows)
        bed_path_str = str(bed_file)

    config = BuildConfig(
        gfa=str(gfa_file),
        ref_fa=str(fa_file),
        backbone_prefix=backbone_prefix,
        out_prefix=str(tmp_path / "out"),
        min_sv_len=min_sv_len,
        max_jump=max_jump,
        centromere_bed=bed_path_str,
        centromere_mode=centromere_mode,  # type: ignore[arg-type]
        centromere_mode_explicit=centromere_mode_explicit,
        centromere_min_island=centromere_min_island,
        dump_repeat_sequences=dump_repeat_sequences,
    )
    graph = load_and_validate_inputs(config)
    return graph, config


# ============================================================================
# Group 1: F7 — Step Projection (`project_steps_onto_backbone`)
# ============================================================================


def test_project_steps_unique_anchors_and_relative_orientations() -> None:
    nodes = {
        "n1": "A" * 100,
        "n2": "C" * 150,
        "n3": "G" * 200,
        "n4": "T" * 250,
        "X": "A" * 80,
    }
    backbone_steps = (
        Step("n1", "+"),
        Step("n2", "-"),
        Step("n3", "+"),
        Step("n4", "-"),
    )
    fragment_steps = (
        Step("n1", "+"),
        Step("X", "+"),
        Step("n2", "-"),
        Step("n3", "-"),
        Step("n4", "+"),
    )
    prov = _make_prov(fragment_steps, nodes)

    anchors = project_steps_onto_backbone(fragment_steps, prov, backbone_steps, nodes)
    assert len(anchors) == 4
    assert [
        (a.branch_step_idx, a.rank_in_b, a.rel_orient, a.weight, a.node_id)
        for a in anchors
    ] == [
        (0, 0, "+", 100, "n1"),
        (2, 1, "+", 150, "n2"),
        (3, 2, "-", 200, "n3"),
        (4, 3, "-", 250, "n4"),
    ]

    tuples = project_step_tuples_onto_backbone(fragment_steps, backbone_steps)
    assert tuples == [(0, "+"), (None, None), (1, "+"), (2, "-"), (3, "-")]


def test_project_steps_repeated_backbone_nodes_and_subinterval_lo_hi() -> None:
    nodes = {
        "n1": "A" * 100,
        "R": "C" * 120,
        "n2": "G" * 100,
        "n3": "T" * 100,
    }
    backbone_steps = (
        Step("n1", "+"),
        Step("R", "+"),
        Step("n2", "+"),
        Step("R", "-"),
        Step("n3", "+"),
    )
    fragment_steps = backbone_steps
    prov = _make_prov(fragment_steps, nodes)

    # Full interval [0, 5): R appears twice, so R is NOT an anchor
    full_anchors = project_steps_onto_backbone(
        fragment_steps, prov, backbone_steps, nodes, lo=0, hi=5
    )
    assert [(a.node_id, a.rank_in_b) for a in full_anchors] == [
        ("n1", 0),
        ("n2", 2),
        ("n3", 4),
    ]

    # Sub-interval [0, 3): R appears once at rank 1 ('+')
    sub1_anchors = project_steps_onto_backbone(
        fragment_steps, prov, backbone_steps, nodes, lo=0, hi=3
    )
    assert [(a.branch_step_idx, a.node_id, a.rank_in_b, a.rel_orient) for a in sub1_anchors] == [
        (0, "n1", 0, "+"),
        (1, "R", 1, "+"),
        (2, "n2", 2, "+"),
        (3, "R", 1, "-"),
    ]

    # Sub-interval [2, 5): R appears once at rank 3 ('-')
    sub2_anchors = project_steps_onto_backbone(
        fragment_steps, prov, backbone_steps, nodes, lo=2, hi=5
    )
    assert [(a.branch_step_idx, a.node_id, a.rank_in_b, a.rel_orient) for a in sub2_anchors] == [
        (1, "R", 3, "-"),
        (2, "n2", 2, "+"),
        (3, "R", 3, "+"),
        (4, "n3", 4, "+"),
    ]

    with pytest.raises(InputContractError):
        build_backbone_anchor_map(backbone_steps, lo=4, hi=2)


def test_project_steps_partial_steps_ignored_as_anchors() -> None:
    nodes = {"n1": "A" * 100, "n2": "C" * 100, "n3": "G" * 100}
    backbone_steps = (
        Step("n1", "+"),
        Step("n2", "+"),
        Step("n3", "+", slice_start=0, slice_end=50, is_partial=True),
    )
    fragment_steps = (
        Step("n1", "+"),
        Step("n2", "+", slice_start=10, slice_end=90, is_partial=True),
        Step("n3", "+"),
    )
    prov = _make_prov(fragment_steps, nodes)

    anchors = project_steps_onto_backbone(fragment_steps, prov, backbone_steps, nodes)
    assert len(anchors) == 1
    assert anchors[0].node_id == "n1"
    assert anchors[0].rank_in_b == 0


# ============================================================================
# Group 2: F8 (T23) — Whole-Contig Orientation (`orient_contig_top_level`)
# ============================================================================


def test_orient_contig_top_level_flips_when_reverse_bp_exceeds_forward_bp_t23(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("1", make_complex_seq(100, seed=1))
    b.add_node("2", make_complex_seq(150, seed=2))
    b.add_node("3", make_complex_seq(200, seed=3))
    b.add_node("X", make_complex_seq(80, seed=4))
    b.add_p_path("Ref#1#Chr01", [("1", "+"), ("2", "-"), ("3", "+")])
    # Reverse-assembled P contig and W walk (with base_offset=1000)
    rev_walk = [("3", "-"), ("X", "+"), ("2", "+"), ("1", "-")]
    b.add_p_path("AltP#1#ctg1", rev_walk)
    b.add_w_walk("AltW", 1, "ctgW", 1000, 1530, rev_walk)

    graph, config = _build_loaded_graph(tmp_path, b)
    alt_p = next(f for f in graph.alt_fragments if f.hap_id == "AltP#1")
    alt_w = next(f for f in graph.alt_fragments if f.hap_id == "AltW#1")

    orig_p_seq = spell_steps(alt_p.steps, graph.nodes)
    oriented_p = orient_contig_top_level(alt_p, graph)
    assert oriented_p.is_flipped_contig is True
    assert oriented_p.steps == (
        Step("1", "+"),
        Step("2", "-"),
        Step("X", "-"),
        Step("3", "+"),
    )
    assert spell_steps(oriented_p.steps, graph.nodes) == revcomp_dna(orig_p_seq)
    assert [
        (p.src_start, p.src_end, p.src_strand) for p in oriented_p.provenance
    ] == [
        (430, 530, "-"),
        (280, 430, "-"),
        (200, 280, "-"),
        (0, 200, "-"),
    ]

    oriented_w = orient_contig_top_level(alt_w, graph)
    assert oriented_w.is_flipped_contig is True
    assert [
        (p.src_start, p.src_end, p.src_strand) for p in oriented_w.provenance
    ] == [
        (1430, 1530, "-"),
        (1280, 1430, "-"),
        (1200, 1280, "-"),
        (1000, 1200, "-"),
    ]

    # T23: Whole-contig orientation must NOT emit any INV event
    inv_res = normalize_inversions_top_level(oriented_p, graph, config)
    assert inv_res.chains == ()
    assert inv_res.unresolved_records == ()


def test_orient_contig_top_level_ties_and_forward_dominant_not_flipped(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("n1", make_complex_seq(200, seed=10))
    b.add_node("n2", make_complex_seq(100, seed=11))
    b.add_node("n3", make_complex_seq(100, seed=12))
    b.add_node("X", make_complex_seq(150, seed=13))
    b.add_p_path("Ref#1#Chr01", [("n1", "+"), ("n2", "+"), ("n3", "+")])
    # Tie in bp: forward=200 (1 step), reverse=100+100=200 (2 steps)
    b.add_p_path("AltTie#1#ctg1", [("n1", "+"), ("n2", "-"), ("n3", "-")])
    # Zero anchors: 0 == 0
    b.add_p_path("AltZero#1#ctg1", [("X", "+")])

    graph, _ = _build_loaded_graph(tmp_path, b)
    for frag in graph.alt_fragments:
        oriented = orient_contig_top_level(frag, graph)
        assert oriented.is_flipped_contig is False
        assert oriented.steps == frag.steps
        assert all(p.src_strand == "+" for p in oriented.provenance)


def test_orient_contig_top_level_ignores_repeated_ref_nodes_and_partial_steps(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=21))
    b.add_node("R_dup", make_complex_seq(300, seed=22))
    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("R_dup", "+"), ("R_dup", "+")])
    b.add_p_path("Alt#1#ctg1", [("u1", "+"), ("R_dup", "-")])

    graph, _ = _build_loaded_graph(tmp_path, b)
    alt = graph.alt_fragments[0]
    oriented = orient_contig_top_level(alt, graph)
    assert oriented.is_flipped_contig is False


# ============================================================================
# Group 3: F9 (T6, T7, T8, T24) — Reverse Chain Detection & Inversion Normalization
# ============================================================================


def test_normalize_inversions_pure_multi_anchor_and_single_anchor_inv_t6_t8(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("11", make_complex_seq(100, seed=31))
    b.add_node("12", make_complex_seq(150, seed=32))
    b.add_node("13", make_complex_seq(150, seed=33))
    b.add_node("14", make_complex_seq(150, seed=34))
    b.add_node("15", make_complex_seq(100, seed=35))
    b.add_p_path(
        "Ref#1#Chr01",
        [("11", "+"), ("12", "+"), ("13", "+"), ("14", "+"), ("15", "+")],
    )
    # Multi-anchor pure inversion: 11+ -> 14- -> 13- -> 12- -> 15+
    b.add_p_path(
        "AltMulti#1#ctg1",
        [("11", "+"), ("14", "-"), ("13", "-"), ("12", "-"), ("15", "+")],
    )
    # Single-anchor unchopped inversion (T8): 11+ -> 12- -> 13+ -> 14+ -> 15+
    b.add_p_path(
        "AltSingle#1#ctg1",
        [("11", "+"), ("12", "-"), ("13", "+"), ("14", "+"), ("15", "+")],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    alt_multi = next(f for f in graph.alt_fragments if f.hap_id == "AltMulti#1")
    alt_single = next(f for f in graph.alt_fragments if f.hap_id == "AltSingle#1")

    res_multi = normalize_inversions_top_level(alt_multi, graph, config)
    assert len(res_multi.chains) == 1
    cm = res_multi.chains[0]
    assert cm.classification == "INV"
    assert cm.normalized is True
    assert (cm.rmin, cm.rmax) == (1, 3)
    assert (cm.left_flank_rank, cm.right_flank_rank) == (0, 4)
    assert (cm.ref_start, cm.ref_end) == (100, 550)
    assert res_multi.normalized_fragment.steps == graph.ref_fragment.steps

    # Verify normalized pure inversion produces 0 deviations
    anc_m = project_steps_onto_backbone(res_multi.normalized_fragment, graph)
    lis_m, _ = compute_weighted_lis(anc_m)
    devs_m = extract_deviations(
        res_multi.normalized_fragment, lis_m, graph.ref_fragment.steps
    )
    assert devs_m == []

    res_single = normalize_inversions_top_level(alt_single, graph, config)
    assert len(res_single.chains) == 1
    cs = res_single.chains[0]
    assert cs.classification == "INV"
    assert cs.normalized is True
    assert cs.rmin == 1 == cs.rmax
    assert (cs.ref_start, cs.ref_end) == (100, 250)
    assert res_single.normalized_fragment.steps == graph.ref_fragment.steps


def test_normalize_inversions_nested_novel_node_x_preserved_with_preflip_coords_t7(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for nid, s in [("11", 41), ("12", 42), ("13", 43), ("14", 44)]:
        b.add_node(nid, make_complex_seq(100, seed=s))
    b.add_node("X", make_complex_seq(80, seed=45))
    b.add_p_path("Ref#1#Chr01", [("11", "+"), ("12", "+"), ("13", "+"), ("14", "+")])
    # W line with base_offset=500: 11+ -> 13- -> X+ -> 12- -> 14+
    b.add_w_walk(
        "Alt",
        1,
        "ctg1",
        500,
        980,
        [("11", "+"), ("13", "-"), ("X", "+"), ("12", "-"), ("14", "+")],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    alt = graph.alt_fragments[0]
    res = normalize_inversions_top_level(alt, graph, config)

    assert len(res.chains) == 1
    assert res.chains[0].classification == "INV"
    assert res.chains[0].normalized is True
    assert res.normalized_fragment.steps == (
        Step("11", "+"),
        Step("12", "+"),
        Step("X", "-"),
        Step("13", "+"),
        Step("14", "+"),
    )
    # X- is at index 2; its pre-flip W-line coordinates [700, 780) must be preserved with src_strand="-"
    x_prov = res.normalized_fragment.provenance[2]
    assert x_prov == StepProvenance(
        src_hap="Alt#1",
        src_path="Alt#1#ctg1:500-980",
        src_start=700,
        src_end=780,
        src_strand="-",
    )

    # LIS + Deviation extraction places X- in D(12, 13) (immediately before node 13)
    anchors = project_steps_onto_backbone(res.normalized_fragment, graph)
    main_chain, multi_hit = compute_weighted_lis(anchors)
    assert [a.node_id for a in main_chain] == ["11", "12", "13", "14"]
    assert multi_hit is False
    devs = extract_deviations(
        res.normalized_fragment, main_chain, graph.ref_fragment.steps
    )
    assert len(devs) == 1
    assert (devs[0].left_anchor_rank, devs[0].right_anchor_rank) == (1, 2)
    assert devs[0].branch_steps == (Step("X", "-"),)
    assert devs[0].branch_provenance == (x_prov,)


def test_normalize_inversions_inv_displaced_classification_and_normalization(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for i in range(6):
        b.add_node(f"n{i}", make_complex_seq(100, seed=50 + i))
    b.add_p_path("Ref#1#Chr01", [(f"n{i}", "+") for i in range(6)])
    # Branch A: L has rank 3 > rmin=1
    b.add_p_path(
        "AltA#1#ctg1",
        [("n0", "+"), ("n3", "+"), ("n2", "-"), ("n1", "-"), ("n4", "+"), ("n5", "+")],
    )
    # Branch B: F has rank 1 < rmax=4
    b.add_p_path(
        "AltB#1#ctg1",
        [("n0", "+"), ("n4", "-"), ("n3", "-"), ("n1", "+"), ("n2", "+"), ("n5", "+")],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    for frag in graph.alt_fragments:
        res = normalize_inversions_top_level(frag, graph, config)
        assert len(res.chains) == 1
        assert res.chains[0].classification == "INV_DISPLACED"
        assert res.chains[0].normalized is True
        assert all(st.orient == "+" for st in res.normalized_fragment.steps)


def test_normalize_inversions_inv_edge_missing_left_or_right_flank_routes_novel_to_unresolved_t24(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for i in range(5):
        b.add_node(f"n{i}", make_complex_seq(100, seed=60 + i))
    b.add_node("X_left", make_complex_seq(80, seed=66))
    b.add_node("X_right", make_complex_seq(90, seed=67))
    b.add_p_path("Ref#1#Chr01", [(f"n{i}", "+") for i in range(5)])
    # Missing L at contig start
    b.add_p_path(
        "AltLeft#1#ctg1",
        [
            ("n1", "-"),
            ("X_left", "+"),
            ("n0", "-"),
            ("n2", "+"),
            ("n3", "+"),
            ("n4", "+"),
        ],
    )
    # Missing F at contig end
    b.add_p_path(
        "AltRight#1#ctg1",
        [
            ("n0", "+"),
            ("n1", "+"),
            ("n2", "+"),
            ("n4", "-"),
            ("X_right", "+"),
            ("n3", "-"),
        ],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    alt_left = next(f for f in graph.alt_fragments if f.hap_id == "AltLeft#1")
    alt_right = next(f for f in graph.alt_fragments if f.hap_id == "AltRight#1")

    res_l = normalize_inversions_top_level(alt_left, graph, config)
    assert len(res_l.chains) == 1
    assert res_l.chains[0].classification == "INV_EDGE"
    assert res_l.chains[0].normalized is False
    assert res_l.normalized_fragment.steps == alt_left.steps
    assert len(res_l.unresolved_records) == 1
    rec_l = res_l.unresolved_records[0]
    assert rec_l.reason == "INV_EDGE"
    assert rec_l.length == 80
    assert rec_l.src_start == 100 and rec_l.src_end == 180
    assert rec_l.export_to_fasta is True
    assert rec_l.seq == graph.nodes["X_left"]

    # Verify unnormalized_branch_ranges suppresses duplicate UNANCHORED_END in extract_deviations
    anc_l = project_steps_onto_backbone(res_l.normalized_fragment, graph)
    lis_l, _ = compute_weighted_lis(anc_l)
    devs_l = extract_deviations(
        res_l.normalized_fragment,
        lis_l,
        graph.ref_fragment.steps,
        unnormalized_branch_ranges=res_l.unnormalized_branch_ranges,
    )
    assert devs_l == []

    res_r = normalize_inversions_top_level(alt_right, graph, config)
    assert len(res_r.chains) == 1
    assert res_r.chains[0].classification == "INV_EDGE"
    assert res_r.chains[0].normalized is False
    assert len(res_r.unresolved_records) == 1
    rec_r = res_r.unresolved_records[0]
    assert rec_r.reason == "INV_EDGE"
    assert rec_r.length == 90
    assert rec_r.export_to_fasta is True


def test_normalize_inversions_inv_conflict_overlapping_rank_intervals_neither_normalized_t24(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for i in range(7):
        b.add_node(f"n{i}", make_complex_seq(100, seed=70 + i))
    b.add_node("X1", make_complex_seq(70, seed=80))
    b.add_node("X2", make_complex_seq(85, seed=81))
    b.add_node("X3", make_complex_seq(60, seed=82))
    b.add_p_path("Ref#1#Chr01", [(f"n{i}", "+") for i in range(7)])

    # Branch 1: overlapping rank intervals [2, 4] and [1, 3]
    b.add_p_path(
        "AltConf1#1#ctg1",
        [
            ("n0", "+"),
            ("n4", "-"),
            ("X1", "+"),
            ("n2", "-"),
            ("n5", "+"),
            ("n3", "-"),
            ("X2", "+"),
            ("n1", "-"),
            ("n6", "+"),
        ],
    )
    # Branch 2: boundary rank touch [1, 2] and [2, 3], with second chain also missing F
    # Row 4 (INV_CONFLICT) must override Row 3 (INV_EDGE)
    b.add_p_path(
        "AltConf2#1#ctg1",
        [
            ("n0", "+"),
            ("n1", "+"),
            ("n5", "+"),
            ("n6", "+"),
            ("n2", "-"),
            ("n1", "-"),
            ("n4", "+"),
            ("n3", "-"),
            ("X3", "+"),
            ("n2", "-"),
        ],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    c1 = next(f for f in graph.alt_fragments if f.hap_id == "AltConf1#1")
    c2 = next(f for f in graph.alt_fragments if f.hap_id == "AltConf2#1")

    res1 = normalize_inversions_top_level(c1, graph, config)
    assert len(res1.chains) == 2
    assert all(ch.classification == "INV_CONFLICT" for ch in res1.chains)
    assert all(ch.normalized is False for ch in res1.chains)
    assert res1.normalized_fragment.steps == c1.steps
    assert len(res1.unresolved_records) == 2
    assert [r.length for r in res1.unresolved_records] == [70, 85]
    assert all(
        r.reason == "INV_CONFLICT" and r.export_to_fasta is True
        for r in res1.unresolved_records
    )

    res2 = normalize_inversions_top_level(c2, graph, config)
    assert len(res2.chains) == 2
    assert [ch.classification for ch in res2.chains] == [
        "INV_CONFLICT",
        "INV_CONFLICT",
    ]
    assert len(res2.unresolved_records) == 1
    assert res2.unresolved_records[0].reason == "INV_CONFLICT"


def test_normalize_inversions_adjacent_non_decreasing_reverse_runs_split_into_separate_chains(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for i in range(6):
        b.add_node(f"n{i}", make_complex_seq(100, seed=90 + i))
    b.add_p_path("Ref#1#Chr01", [(f"n{i}", "+") for i in range(6)])
    # n2-(r=2) -> n1-(r=1) followed immediately by n4-(r=4) -> n3-(r=3)
    b.add_p_path(
        "Alt#1#ctg1",
        [("n0", "+"), ("n2", "-"), ("n1", "-"), ("n4", "-"), ("n3", "-"), ("n5", "+")],
    )

    graph, config = _build_loaded_graph(tmp_path, b)
    res = normalize_inversions_top_level(graph.alt_fragments[0], graph, config)
    assert len(res.chains) == 2
    assert (res.chains[0].rmin, res.chains[0].rmax) == (1, 2)
    assert (res.chains[1].rmin, res.chains[1].rmax) == (3, 4)
    assert all(ch.classification == "INV" and ch.normalized for ch in res.chains)
    assert res.normalized_fragment.steps == graph.ref_fragment.steps


def test_normalize_inversions_max_jump_immunity_and_min_sv_len_span(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("n0", make_complex_seq(100, seed=101))
    b.add_node("n1", "A" * 500_000)
    b.add_node("n2", "C" * 500_000)
    b.add_node("n3", make_complex_seq(100, seed=102))
    b.add_p_path("Ref#1#Chr01", [("n0", "+"), ("n1", "+"), ("n2", "+"), ("n3", "+")])
    b.add_p_path("Alt#1#ctg1", [("n0", "+"), ("n2", "-"), ("n1", "-"), ("n3", "+")])

    graph, config = _build_loaded_graph(tmp_path, b, max_jump=1000)
    res = normalize_inversions_top_level(graph.alt_fragments[0], graph, config)
    assert len(res.chains) == 1
    assert res.chains[0].classification == "INV"
    assert res.chains[0].normalized is True
    assert res.chains[0].ref_end - res.chains[0].ref_start == 1_000_000


# ============================================================================
# Group 4: F10 — Weighted LIS & `multi_hit` (`compute_weighted_lis`)
# ============================================================================


def test_compute_weighted_lis_prefers_heavier_node_weights_over_step_count() -> None:
    dummy_prov = StepProvenance("Alt#1", "Alt#1#ctg1", 0, 10, "+")
    anchors = [
        ProjectedAnchor(0, 0, "+", "n0", 100, Step("n0", "+"), dummy_prov),
        ProjectedAnchor(1, 2, "+", "n2", 40, Step("n2", "+"), dummy_prov),
        ProjectedAnchor(2, 3, "+", "n3", 40, Step("n3", "+"), dummy_prov),
        ProjectedAnchor(3, 1, "+", "n1", 200, Step("n1", "+"), dummy_prov),
        ProjectedAnchor(4, 4, "+", "n4", 100, Step("n4", "+"), dummy_prov),
    ]
    chain, multi_hit = compute_weighted_lis(anchors)
    assert [a.rank_in_b for a in chain] == [0, 1, 4]
    assert sum(a.weight for a in chain) == 400
    assert multi_hit is False


def test_compute_weighted_lis_duplicate_rank_exclusion_and_earlier_source_step_tie_breaking() -> None:
    dummy_prov = StepProvenance("Alt#1", "Alt#1#ctg1", 0, 10, "+")
    # Case A: rank 1 repeated at branch_step_idx 1 and 2
    anchors_dup = [
        ProjectedAnchor(0, 0, "+", "n0", 100, Step("n0", "+"), dummy_prov),
        ProjectedAnchor(1, 1, "+", "n1", 100, Step("n1", "+"), dummy_prov),
        ProjectedAnchor(2, 1, "+", "n1", 100, Step("n1", "+"), dummy_prov),
        ProjectedAnchor(3, 2, "+", "n2", 100, Step("n2", "+"), dummy_prov),
    ]
    chain_a, multi_a = compute_weighted_lis(anchors_dup)
    assert [a.branch_step_idx for a in chain_a] == [0, 1, 3]
    assert [a.rank_in_b for a in chain_a] == [0, 1, 2]
    assert multi_a is True

    # Case B: Equal-weight chains [0, 3, 4] at indices (0, 1, 3) vs [0, 2, 4] at indices (0, 2, 3)
    anchors_tie = [
        ProjectedAnchor(0, 0, "+", "n0", 100, Step("n0", "+"), dummy_prov),
        ProjectedAnchor(1, 3, "+", "n3", 150, Step("n3", "+"), dummy_prov),
        ProjectedAnchor(2, 2, "+", "n2", 150, Step("n2", "+"), dummy_prov),
        ProjectedAnchor(3, 4, "+", "n4", 100, Step("n4", "+"), dummy_prov),
    ]
    chain_b, multi_b = compute_weighted_lis(anchors_tie)
    assert [a.branch_step_idx for a in chain_b] == [0, 1, 3]
    assert [a.rank_in_b for a in chain_b] == [0, 3, 4]
    assert multi_b is False


def test_compute_weighted_lis_multi_hit_detection_and_empty_or_reverse_only_inputs() -> None:
    assert compute_weighted_lis([]) == ([], False)

    dummy_prov = StepProvenance("Alt#1", "Alt#1#ctg1", 0, 10, "+")
    rev_and_fwd = [
        ProjectedAnchor(0, 1, "-", "n1", 100, Step("n1", "-"), dummy_prov),
        ProjectedAnchor(1, 1, "+", "n1", 100, Step("n1", "+"), dummy_prov),
    ]
    chain, multi_hit = compute_weighted_lis(rev_and_fwd)
    assert len(chain) == 1 and chain[0].branch_step_idx == 1
    assert multi_hit is True


# ============================================================================
# Group 5: F11 (T15) — Deviation Segment Extraction (`extract_deviations`)
# ============================================================================


def test_extract_deviations_identical_repeat_traversal_emits_no_deviation_t15() -> None:
    nodes = {
        "u1": "A" * 100,
        "R1": "C" * 80,
        "R2": "G" * 80,
        "u2": "T" * 100,
    }
    backbone_steps = (
        Step("u1", "+"),
        Step("R1", "+"),
        Step("R2", "-"),
        Step("R1", "+"),
        Step("R2", "-"),
        Step("u2", "+"),
    )
    frag = _make_fragment(backbone_steps, nodes)
    anchors = project_steps_onto_backbone(frag, backbone_steps, nodes)
    main_chain, _ = compute_weighted_lis(anchors)

    devs = extract_deviations(frag, main_chain, backbone_steps)
    assert devs == []


def test_extract_deviations_pure_insertion_pure_deletion_and_replacement() -> None:
    nodes = {f"n{i}": "A" * 100 for i in range(6)}
    nodes["X"] = "C" * 70
    nodes["Y"] = "G" * 90
    backbone_steps = tuple(Step(f"n{i}", "+") for i in range(6))
    branch_steps = (
        Step("n0", "+"),
        Step("X", "+"),
        Step("n1", "+"),
        Step("n3", "+"),
        Step("Y", "-"),
        Step("n5", "+"),
    )
    frag = _make_fragment(branch_steps, nodes)
    anchors = project_steps_onto_backbone(frag, backbone_steps, nodes)
    main_chain, _ = compute_weighted_lis(anchors)

    devs = extract_deviations(frag, main_chain, backbone_steps)
    assert len(devs) == 3
    # Pure insertion D(0, 1)
    assert (devs[0].left_anchor_rank, devs[0].right_anchor_rank) == (0, 1)
    assert devs[0].branch_steps == (Step("X", "+"),)
    assert devs[0].backbone_steps == ()
    # Pure deletion D(1, 3)
    assert (devs[1].left_anchor_rank, devs[1].right_anchor_rank) == (1, 3)
    assert devs[1].branch_steps == ()
    assert devs[1].backbone_steps == (Step("n2", "+"),)
    # Replacement D(3, 5)
    assert (devs[2].left_anchor_rank, devs[2].right_anchor_rank) == (3, 5)
    assert devs[2].branch_steps == (Step("Y", "-"),)
    assert devs[2].backbone_steps == (Step("n4", "+"),)


def test_extract_deviations_top_level_unanchored_ends_vs_subproblem_logical_start_end() -> None:
    nodes = {
        "b0": "A" * 100,
        "b1": "C" * 100,
        "b2": "G" * 100,
        "b3": "T" * 100,
        "pre": "A" * 60,
        "post": "C" * 60,
    }
    backbone_steps = (
        Step("b0", "+"),
        Step("b1", "+"),
        Step("b2", "+"),
        Step("b3", "+"),
    )
    frag = _make_fragment(
        (Step("pre", "+"), Step("b1", "+"), Step("b2", "+"), Step("post", "+")),
        nodes,
    )
    anchors = project_steps_onto_backbone(frag, backbone_steps, nodes)
    main_chain, _ = compute_weighted_lis(anchors)

    top_devs = extract_deviations(
        frag, main_chain, backbone_steps, is_top_level=True
    )
    assert len(top_devs) == 2
    assert top_devs[0].left_anchor_rank == "START" and top_devs[0].right_anchor_rank == 1
    assert top_devs[0].is_unanchored_start is True
    assert top_devs[0].backbone_steps == ()
    assert top_devs[1].left_anchor_rank == 2 and top_devs[1].right_anchor_rank == "END"
    assert top_devs[1].is_unanchored_end is True
    assert top_devs[1].backbone_steps == ()

    sub_devs = extract_deviations(
        frag, main_chain, backbone_steps, is_top_level=False, lo=0, hi=4
    )
    assert len(sub_devs) == 2
    assert sub_devs[0].is_unanchored_start is False
    assert sub_devs[0].backbone_steps == (Step("b0", "+"),)
    assert sub_devs[1].is_unanchored_end is False
    assert sub_devs[1].backbone_steps == (Step("b3", "+"),)


# ============================================================================
# Group 6: F12 (T17) — Greedy Cycle Folding (`fold_tandem_cycles`)
# ============================================================================


def test_fold_tandem_cycles_preserves_u_x_u_and_folds_u_u_u_t17() -> None:
    nodes = {
        "A": "A" * 50,
        "B": "C" * 50,
        "U": "G" * 80,
        "X": "T" * 60,
    }
    global_ref_nodes: set[str] = set()

    # Case A: U -> X -> U has no consecutive repeat period -> preserved intact
    steps_a = (Step("U", "+"), Step("X", "+"), Step("U", "+"))
    prov_a = _make_prov(steps_a, nodes)
    res_a = fold_tandem_cycles(steps_a, prov_a, global_ref_nodes, max_copy=1)
    assert res_a.folded_steps == steps_a
    assert res_a.piece_boundaries == ()
    assert res_a.deleted_intervals == ()

    # Case B: A -> U -> U -> U -> B folds U*3 to U*1 and records piece boundary before B
    steps_b = (
        Step("A", "+"),
        Step("U", "+"),
        Step("U", "+"),
        Step("U", "+"),
        Step("B", "+"),
    )
    prov_b = _make_prov(steps_b, nodes)
    res_b = fold_tandem_cycles(steps_b, prov_b, global_ref_nodes, max_copy=1)
    assert res_b.folded_steps == (Step("A", "+"), Step("U", "+"), Step("B", "+"))
    assert res_b.piece_boundaries == (2,)
    assert res_b.deleted_intervals == (prov_b[2:4],)


def test_fold_tandem_cycles_multi_node_period_max_copy_2_and_trailing_incomplete_period() -> None:
    nodes = {"U1": "A" * 40, "U2": "C" * 50}
    steps = (Step("U1", "+"), Step("U2", "-")) * 4 + (Step("U1", "+"),)
    prov = _make_prov(steps, nodes)

    res_1 = fold_tandem_cycles(steps, prov, set(), max_copy=1)
    assert res_1.folded_steps == (
        Step("U1", "+"),
        Step("U2", "-"),
        Step("U1", "+"),
    )
    assert res_1.piece_boundaries == (2,)
    assert res_1.deleted_intervals == (prov[2:8],)

    res_2 = fold_tandem_cycles(steps, prov, set(), max_copy=2)
    assert res_2.folded_steps == (Step("U1", "+"), Step("U2", "-")) * 2 + (
        Step("U1", "+"),
    )
    assert res_2.piece_boundaries == (4,)
    assert res_2.deleted_intervals == (prov[4:8],)

    res_4 = fold_tandem_cycles(steps, prov, set(), max_copy=4)
    assert res_4.folded_steps == steps
    assert res_4.piece_boundaries == ()
    assert res_4.deleted_intervals == ()


def test_fold_tandem_cycles_skips_global_ref_nodes_and_partial_steps_and_respects_orientation() -> None:
    nodes = {"R_ref": "A" * 50, "U": "C" * 50}
    # Global reference nodes are never folded
    steps_ref = (Step("R_ref", "+"),) * 3
    prov_ref = _make_prov(steps_ref, nodes)
    res_ref = fold_tandem_cycles(steps_ref, prov_ref, {"R_ref"}, max_copy=1)
    assert res_ref.folded_steps == steps_ref

    # Alternating orientations U+ -> U- -> U+ -> U- fold at p=2, not p=1
    steps_alt = (Step("U", "+"), Step("U", "-"), Step("U", "+"), Step("U", "-"))
    prov_alt = _make_prov(steps_alt, nodes)
    res_alt = fold_tandem_cycles(steps_alt, prov_alt, set(), max_copy=1)
    assert res_alt.folded_steps == (Step("U", "+"), Step("U", "-"))

    # Partial steps are never folded
    steps_part = (
        Step("U", "+", slice_start=0, slice_end=20, is_partial=True),
        Step("U", "+"),
        Step("U", "+"),
    )
    prov_part = _make_prov(steps_part, nodes)
    res_part = fold_tandem_cycles(steps_part, prov_part, set(), max_copy=1)
    assert res_part.folded_steps == (steps_part[0], Step("U", "+"))


def test_fold_tandem_cycles_multiple_disjoint_tandem_arrays_and_end_deletion() -> None:
    nodes = {"U": "A" * 50, "M": "C" * 50, "V": "G" * 50}
    steps = (Step("U", "+"),) * 3 + (Step("M", "+"),) + (Step("V", "+"),) * 3
    prov = _make_prov(steps, nodes)

    res = fold_tandem_cycles(steps, prov, set(), max_copy=1)
    assert res.folded_steps == (Step("U", "+"), Step("M", "+"), Step("V", "+"))
    # Split before M (index 1); end deletion after V produces no out-of-bounds split index
    assert res.piece_boundaries == (1,)
    assert res.deleted_intervals == (prov[1:3], prov[5:7])


# ============================================================================
# Group 7: F13 (T10) — Canonical Period CNV Counting (`detect_canonical_cnv`)
# ============================================================================


def test_detect_canonical_cnv_5_to_6_copies_and_min_sv_len_threshold_t10() -> None:
    nodes = {"R1": "A" * 35, "R2": "C" * 25}
    ref_steps = (Step("R1", "+"), Step("R2", "+")) * 5
    alt_steps_6 = (Step("R1", "+"), Step("R2", "+")) * 6
    alt_steps_3 = (Step("R1", "+"), Step("R2", "+")) * 3

    cnv_6 = detect_canonical_cnv(ref_steps, alt_steps_6, nodes, min_sv_len=50)
    assert cnv_6 == CNVCall(
        unit_tokens=(("R1", "+"), ("R2", "+")),
        unit_bp=60,
        k_ref=5,
        k_alt=6,
        delta_bp=60,
    )

    cnv_3 = detect_canonical_cnv(ref_steps, alt_steps_3, nodes, min_sv_len=50)
    assert cnv_3 == CNVCall(
        unit_tokens=(("R1", "+"), ("R2", "+")),
        unit_bp=60,
        k_ref=5,
        k_alt=3,
        delta_bp=120,
    )

    # Below min_sv_len threshold
    assert detect_canonical_cnv(ref_steps, alt_steps_6, nodes, min_sv_len=100) is None
    # Equal copy count
    assert detect_canonical_cnv(ref_steps, ref_steps, nodes, min_sv_len=50) is None
    assert compute_raw_visit_counts(ref_steps) == {"R1": 5, "R2": 5}


def test_detect_canonical_cnv_rotated_periods_match_via_lexicographical_canonical_rotation() -> None:
    nodes = {"nA": "A" * 20, "nB": "C" * 20, "nC": "G" * 20}
    ref_steps = (Step("nB", "+"), Step("nC", "-"), Step("nA", "+")) * 3
    alt_steps = (Step("nC", "-"), Step("nA", "+"), Step("nB", "+")) * 5

    cnv = detect_canonical_cnv(ref_steps, alt_steps, nodes, min_sv_len=50)
    assert cnv == CNVCall(
        unit_tokens=(("nA", "+"), ("nB", "+"), ("nC", "-")),
        unit_bp=60,
        k_ref=3,
        k_alt=5,
        delta_bp=120,
    )

    # Also test k=1 vs k=3 with cyclic rotation
    one_copy = (Step("nC", "-"), Step("nA", "+"), Step("nB", "+"))
    cnv_1_to_3 = detect_canonical_cnv(one_copy, ref_steps, nodes, min_sv_len=50)
    assert cnv_1_to_3 == CNVCall(
        unit_tokens=(("nA", "+"), ("nB", "+"), ("nC", "-")),
        unit_bp=60,
        k_ref=1,
        k_alt=3,
        delta_bp=120,
    )


def test_detect_canonical_cnv_mismatched_canonical_units_or_aperiodic_returns_none() -> None:
    nodes = {"nA": "A" * 30, "nB": "C" * 30, "nC": "G" * 30}
    ref_steps = (Step("nA", "+"), Step("nB", "+")) * 3
    alt_diff_unit = (Step("nA", "+"), Step("nC", "+")) * 4
    alt_aperiodic = (Step("nA", "+"), Step("nB", "+"), Step("nC", "+"))

    assert detect_canonical_cnv(ref_steps, alt_diff_unit, nodes, min_sv_len=50) is None
    assert detect_canonical_cnv(ref_steps, alt_aperiodic, nodes, min_sv_len=50) is None
    assert canonical_rotation([]) == ()


# ============================================================================
# Group 8–11: F21–F24 (C2–C16, C18, C19) — Two-Tier Centromere Engine
# ============================================================================


def test_delimit_centromere_regions_outward_expansion_and_overlap_merging_c5(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u0", make_complex_seq(100, seed=110))
    b.add_node("u1", make_complex_seq(100, seed=111))
    b.add_node("sat1", make_complex_seq(200, seed=112))
    b.add_node("sat2", make_complex_seq(200, seed=113))
    b.add_node("u2", make_complex_seq(100, seed=114))
    b.add_node("u3", make_complex_seq(100, seed=115))
    b.add_p_path(
        "Ref#1#Chr01",
        [
            ("u0", "+"),
            ("u1", "+"),
            ("sat1", "+"),
            ("sat2", "+"),
            ("sat1", "+"),
            ("sat2", "+"),
            ("u2", "+"),
            ("u3", "+"),
        ],
    )

    graph, config = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 250, 500, "BED_1"), ("Chr01", 700, 950, "BED_2")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regions, logs = delimit_centromere_regions(graph, config)
    assert len(regions) == 1
    reg = regions[0]
    assert reg.region_id == "CEN_1"
    assert reg.input_bed_ids == ("BED_1", "BED_2")
    assert (reg.bed_start, reg.bed_end) == (250, 950)
    assert (reg.ref_start, reg.ref_end) == (200, 1000)
    assert (reg.anchor_u_step, reg.anchor_v_step) == (1, 6)
    assert (reg.anchor_u_name, reg.anchor_v_name) == ("1", "6")
    assert reg.anchor_not_found is False
    assert reg.expanded_to_whole_chromosome is False
    assert reg.effective_mode == "extract-islands"
    assert len(logs) == 1


def test_delimit_centromere_regions_missing_flank_virtual_boundary_and_whole_chrom_c5(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    # Sub-case A: Missing left anchor u
    (tmp_path / "a").mkdir(exist_ok=True)
    b_a = gfa_builder()
    b_a.add_node("sat1", make_complex_seq(100, seed=121))
    b_a.add_node("u1", make_complex_seq(100, seed=122))
    b_a.add_node("u2", make_complex_seq(100, seed=123))
    b_a.add_p_path(
        "Ref#1#Chr01", [("sat1", "+"), ("sat1", "+"), ("u1", "+"), ("u2", "+")]
    )
    g_a, cfg_a = _build_loaded_graph(
        tmp_path / "a",
        b_a,
        bed_rows=[("Chr01", 50, 150, "BED_L")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs_a, logs_a = delimit_centromere_regions(g_a, cfg_a)
    assert regs_a[0].anchor_u_step is None
    assert regs_a[0].anchor_u_name == "VIRTUAL_START"
    assert regs_a[0].ref_start == 0
    assert regs_a[0].anchor_not_found is True
    assert regs_a[0].expanded_to_whole_chromosome is False
    assert regs_a[0].effective_mode == "bypass"
    assert any("CEN_ANCHOR_NOT_FOUND" in m for m in logs_a)

    # Sub-case B: Missing right anchor v
    (tmp_path / "b").mkdir(exist_ok=True)
    b_b = gfa_builder()
    b_b.add_node("u0", make_complex_seq(100, seed=124))
    b_b.add_node("u1", make_complex_seq(100, seed=125))
    b_b.add_node("sat1", make_complex_seq(100, seed=126))
    b_b.add_p_path(
        "Ref#1#Chr01", [("u0", "+"), ("u1", "+"), ("sat1", "+"), ("sat1", "+")]
    )
    g_b, cfg_b = _build_loaded_graph(
        tmp_path / "b",
        b_b,
        bed_rows=[("Chr01", 250, 350, "BED_R")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs_b, logs_b = delimit_centromere_regions(g_b, cfg_b)
    assert regs_b[0].anchor_v_step is None
    assert regs_b[0].anchor_v_name == "VIRTUAL_END"
    assert regs_b[0].ref_end == len(g_b.ref_seq)
    assert regs_b[0].effective_mode == "bypass"
    assert any("CEN_ANCHOR_NOT_FOUND" in m for m in logs_b)

    # Sub-case C: Missing both anchors -> expanded_to_whole_chromosome
    (tmp_path / "c").mkdir(exist_ok=True)
    b_c = gfa_builder()
    b_c.add_node("sat1", make_complex_seq(100, seed=127))
    b_c.add_node("sat2", make_complex_seq(100, seed=128))
    b_c.add_p_path(
        "Ref#1#Chr01", [("sat1", "+"), ("sat2", "+"), ("sat1", "+"), ("sat2", "+")]
    )
    g_c, cfg_c = _build_loaded_graph(
        tmp_path / "c",
        b_c,
        bed_rows=[("Chr01", 100, 300, "BED_BOTH")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs_c, logs_c = delimit_centromere_regions(g_c, cfg_c)
    assert regs_c[0].anchor_u_step is None and regs_c[0].anchor_v_step is None
    assert regs_c[0].expanded_to_whole_chromosome is True
    assert regs_c[0].effective_mode == "bypass"
    assert any("expanded_to_whole_chromosome=True" in m for m in logs_c)


def test_partition_fragments_by_centromere_clean_slicing_and_boundary_ambiguous(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u0", make_complex_seq(100, seed=130))
    b.add_node("u1", make_complex_seq(100, seed=131))
    b.add_node("cen_ref", make_complex_seq(200, seed=132))
    b.add_node("u2", make_complex_seq(100, seed=133))
    b.add_node("u3", make_complex_seq(100, seed=134))
    b.add_node("ins_left", make_complex_seq(60, seed=135))
    b.add_node("cen_alt", make_complex_seq(600, seed=136))
    b.add_node("ins_right", make_complex_seq(70, seed=137))
    b.add_node("cross_novel", make_complex_seq(80, seed=138))

    b.add_p_path(
        "Ref#1#Chr01",
        [("u0", "+"), ("u1", "+"), ("cen_ref", "+"), ("u2", "+"), ("u3", "+")],
    )
    b.add_p_path(
        "HapClean#1#ctg1",
        [
            ("u0", "+"),
            ("ins_left", "+"),
            ("u1", "+"),
            ("cen_alt", "+"),
            ("u2", "+"),
            ("ins_right", "+"),
            ("u3", "+"),
        ],
    )
    b.add_p_path(
        "HapCross#1#ctg1",
        [("u0", "+"), ("cross_novel", "+"), ("u2", "+"), ("u3", "+")],
    )

    graph, config = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 200, 400, "CEN_BED")],
        centromere_mode="bypass",
        centromere_mode_explicit=True,
    )
    regions, _ = delimit_centromere_regions(graph, config)
    part = partition_fragments_by_centromere(
        graph.alt_fragments, regions, graph, config
    )

    clean_slice = part.hap_slices[("CEN_1", "HapClean#1")]
    assert clean_slice.status == "complete"
    assert clean_slice.inner_steps == (Step("cen_alt", "+"),)
    assert ("HapClean#1#ctg1", 260, 860) in part.claimed_source_intervals

    clean_norm = [f for f in part.normal_fragments if f.hap_id == "HapClean#1"]
    assert len(clean_norm) == 2
    assert clean_norm[0].steps == (
        Step("u0", "+"),
        Step("ins_left", "+"),
        Step("u1", "+"),
    )
    assert clean_norm[1].steps == (
        Step("u2", "+"),
        Step("ins_right", "+"),
        Step("u3", "+"),
    )

    cross_slice = part.hap_slices[("CEN_1", "HapCross#1")]
    assert cross_slice.status == "partial"
    assert len(part.boundary_ambiguous_records) == 1
    amb_rec = part.boundary_ambiguous_records[0]
    assert amb_rec.reason == "CEN_BOUNDARY_AMBIGUOUS"
    assert amb_rec.length == 80
    assert amb_rec.export_to_fasta is True


def test_centromere_completeness_complete_and_revisited_step_length_ratio_c3(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=140))
    b.add_node("sat", make_complex_seq(200, seed=141))
    b.add_node("v1", make_complex_seq(100, seed=142))
    b.add_p_path(
        "Ref#1#Chr01",
        [("u1", "+")] + [("sat", "+")] * 5 + [("v1", "+")],
    )
    b.add_p_path(
        "Hap6#1#ctg1",
        [("u1", "+")] + [("sat", "+")] * 6 + [("v1", "+")],
    )
    b.add_p_path("Hap0#1#ctg1", [("u1", "+"), ("v1", "+")])

    graph, config = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 1100, "CEN_C3")],
        centromere_mode="bypass",
        centromere_mode_explicit=True,
    )
    regions, _ = delimit_centromere_regions(graph, config)
    part = partition_fragments_by_centromere(
        graph.alt_fragments, regions, graph, config
    )
    out = process_centromere_regions(part, graph, config)

    s6 = part.hap_slices[("CEN_1", "Hap6#1")]
    assert s6.status == "complete"
    assert s6.l_ref == 1000
    assert s6.l_path == 1200
    assert s6.length_ratio == pytest.approx(1.2)

    rows_by_hap = {r.hap: r for r in out.centromere_rows}
    assert rows_by_hap["Hap6#1"].L_path == 1200
    assert rows_by_hap["Hap6#1"].length_ratio == "1.200000"
    assert rows_by_hap["Hap0#1"].L_path == 0
    assert rows_by_hap["Hap0#1"].length_ratio == "0.000000"


def test_centromere_completeness_gapped_partial_ambiguous_unmapped_and_ref_gapped_c4(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    (tmp_path / "g1").mkdir(exist_ok=True)
    b = gfa_builder()
    b.add_node("u0", make_complex_seq(100, seed=150))
    b.add_node("u1", make_complex_seq(100, seed=151))
    b.add_node("cen_ref", make_complex_seq(500, seed=152))
    b.add_node("v1", make_complex_seq(100, seed=153))
    b.add_node("gap_node", make_complex_seq(200, seed=154) + "N" * 50 + make_complex_seq(200, seed=155))
    b.add_node("cand_node", make_complex_seq(600, seed=156))

    b.add_p_path("Ref#1#Chr01", [("u0", "+"), ("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapGapped#1#ctg1", [("u1", "+"), ("gap_node", "+"), ("v1", "+")])
    b.add_p_path("HapPartial#1#ctg1", [("u0", "+"), ("u1", "+")])
    b.add_p_path(
        "HapAmbigRepeatU#1#ctg1",
        [("u1", "+"), ("cand_node", "+"), ("u1", "+"), ("v1", "+")],
    )
    b.add_w_walk("HapAmbigMultiFrag", 1, "ctgA", 0, 700, [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_w_walk("HapAmbigMultiFrag", 1, "ctgB", 0, 100, [("u1", "+")])
    b.add_p_path("HapUnmapped#1#ctg1", [("u0", "+")])

    graph, config = _build_loaded_graph(
        tmp_path / "g1",
        b,
        bed_rows=[("Chr01", 200, 700, "CEN_C4")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, config)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, config)
    out = process_centromere_regions(part, graph, config)

    rows = {r.hap: r for r in out.centromere_rows}
    assert rows["HapGapped#1"].status == "gapped"
    assert rows["HapGapped#1"].L_path == 450
    assert rows["HapGapped#1"].length_ratio == "."
    assert rows["HapPartial#1"].status == "partial"
    assert rows["HapAmbigRepeatU#1"].status == "ambiguous"
    assert rows["HapAmbigMultiFrag#1"].status == "ambiguous"
    assert rows["HapUnmapped#1"].status == "unmapped"

    for h in ("HapPartial#1", "HapAmbigRepeatU#1", "HapAmbigMultiFrag#1", "HapUnmapped#1"):
        r = rows[h]
        assert r.L_path == "."
        assert r.length_ratio == "."
        assert r.src_path == "."
        assert r.nonref_input_bp == "."
        assert r.supported_events == "."

    # Second graph: reference centromere contains 'N' -> ref_status == "gapped" -> CEN_REFERENCE_GAP
    (tmp_path / "g2").mkdir(exist_ok=True)
    b2 = gfa_builder()
    b2.add_node("u1", make_complex_seq(100, seed=160))
    b2.add_node("cen_n", make_complex_seq(200, seed=161) + "N" * 50 + make_complex_seq(200, seed=162))
    b2.add_node("v1", make_complex_seq(100, seed=163))
    b2.add_node("alt_island", make_complex_seq(600, seed=164))
    b2.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_n", "+"), ("v1", "+")])
    b2.add_p_path("HapComp#1#ctg1", [("u1", "+"), ("alt_island", "+"), ("v1", "+")])

    g2, cfg2 = _build_loaded_graph(
        tmp_path / "g2",
        b2,
        bed_rows=[("Chr01", 100, 550, "CEN_REFGAP")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs2, _ = delimit_centromere_regions(g2, cfg2)
    part2 = partition_fragments_by_centromere(g2.alt_fragments, regs2, g2, cfg2)
    out2 = process_centromere_regions(part2, g2, cfg2)
    row_comp = next(r for r in out2.centromere_rows if r.hap == "HapComp#1")
    assert row_comp.status == "complete"
    assert row_comp.ref_status == "gapped"
    assert row_comp.length_ratio == "."
    assert row_comp.unresolved_bp == 600
    assert len(out2.unresolved_records) == 1
    assert out2.unresolved_records[0].reason == "CEN_REFERENCE_GAP"
    assert out2.unresolved_records[0].export_to_fasta is True


def test_centromere_bypass_mode_preserves_ref_and_records_cen_bypass_c2(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(150, seed=170))
    b.add_node("cen_ref", make_complex_seq(2000, seed=171))
    b.add_node("v1", make_complex_seq(150, seed=172))
    b.add_node("cen_alt", make_complex_seq(700, seed=173))
    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    # Reverse-assembled contig so orient_contig_top_level flips it to src_strand="-"
    b.add_p_path("HapAlt#1#ctg1", [("v1", "-"), ("cen_alt", "-"), ("u1", "-")])

    graph, config = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 150, 2150, "CEN_BYP")],
        centromere_mode="bypass",
        centromere_mode_explicit=True,
        max_jump=500,
    )
    oriented = [orient_contig_top_level(f, graph) for f in graph.alt_fragments]
    regs, _ = delimit_centromere_regions(graph, config)
    part = partition_fragments_by_centromere(oriented, regs, graph, config)
    out = process_centromere_regions(part, graph, config)

    assert out.island_slot_blocks == ()
    assert out.island_sv_events == ()
    assert len(out.unresolved_records) == 1
    rec = out.unresolved_records[0]
    assert rec.reason == "CEN_BYPASS"
    assert rec.length == 700
    assert rec.src_strand == "-"
    assert rec.export_to_fasta is False
    assert rec.exported_fasta_id == "."

    row = next(r for r in out.centromere_rows if r.hap == "HapAlt#1")
    assert row.nonref_input_bp == 700
    assert row.unresolved_bp == 700
    assert row.reference_like_bp == 0
    assert row.repeat_like_bp == 0
    assert row.short_bp == 0
    assert row.accepted_source_bp == 0
    assert row.supported_events == "."


def test_centromere_bypass_and_repeat_like_dump_repeat_sequences_toggle_c18(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=180))
    b.add_node("cen_ref", make_complex_seq(600, seed=181))
    b.add_node("v1", make_complex_seq(100, seed=182))
    b.add_node("sat_block", make_satellite_seq(unit_len=20, total_len=1500, seed=183))
    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapSat#1#ctg1", [("u1", "+"), ("sat_block", "+"), ("v1", "+")])

    for dump_flag in (False, True):
        sub_dir = tmp_path / f"dump_{dump_flag}"
        sub_dir.mkdir(exist_ok=True)
        graph, cfg = _build_loaded_graph(
            sub_dir,
            b,
            bed_rows=[("Chr01", 100, 700, "CEN_C18")],
            centromere_mode="extract-islands",
            centromere_mode_explicit=True,
            dump_repeat_sequences=dump_flag,
        )
        regs, _ = delimit_centromere_regions(graph, cfg)
        part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
        out = process_centromere_regions(part, graph, cfg)

        assert len(out.unresolved_records) == 1
        rec = out.unresolved_records[0]
        assert rec.reason == "CEN_REPEAT_LIKE"
        assert rec.export_to_fasta is dump_flag
        if dump_flag:
            assert rec.exported_fasta_id != "."
        else:
            assert rec.exported_fasta_id == "."


def test_extract_islands_step_a_revisited_fraction_and_distinct_node_satellite_filtered_c6_c7(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=190))
    b.add_node("cen_ref", make_complex_seq(600, seed=191))
    b.add_node("v1", make_complex_seq(100, seed=192))

    # C6: 15 distinct 100 bp nodes spelling a 1500 bp satellite (revisited_fraction == 0.0)
    sat_1500 = make_satellite_seq(unit_len=20, total_len=1500, seed=193)
    sat_steps: list[tuple[str, str]] = []
    for idx in range(15):
        nid = f"sat_n{idx}"
        b.add_node(nid, sat_1500[idx * 100 : (idx + 1) * 100])
        sat_steps.append((nid, "+"))

    # C7: Revisited complex node (250bp) + unique complex node (500bp) + revisited complex node (250bp)
    b.add_node("rev_node", make_complex_seq(250, seed=194))
    b.add_node("uniq_node", make_complex_seq(500, seed=195))

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapSat#1#ctg1", [("u1", "+"), *sat_steps, ("v1", "+")])
    b.add_p_path(
        "HapRev#1#ctg1",
        [("u1", "+"), ("rev_node", "+"), ("uniq_node", "+"), ("rev_node", "+"), ("v1", "+")],
    )

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 700, "CEN_C6_C7")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    rows = {r.hap: r for r in out.centromere_rows}
    # C6: HapSat#1 has revisited_fraction == 0.0 but is masked by Step C as CEN_REPEAT_LIKE
    assert rows["HapSat#1"].repeat_like_bp == 1500
    assert rows["HapSat#1"].accepted_source_bp == 0
    sat_log = next(e for e in out.json_log_entries if e["hap"] == "HapSat#1")
    assert sat_log["revisited_fraction"] == pytest.approx(0.0)
    assert sat_log["final_reason"] == "CEN_REPEAT_LIKE"

    # C7: HapRev#1 has revisited_fraction == 0.5, is NOT deleted by Step A, and survives as CEN_ISLAND
    assert rows["HapRev#1"].accepted_source_bp == 1000
    rev_log = next(e for e in out.json_log_entries if e["hap"] == "HapRev#1")
    assert rev_log["revisited_fraction"] == pytest.approx(0.5)
    assert rev_log["final_reason"] == "ACCEPTED_ISLAND"
    assert len(out.island_sv_events) == 1
    assert out.island_sv_events[0].pan_end == 1000


def test_extract_islands_step_b_reference_subtraction_protects_internal_ins_and_revcomp_c8_c9(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    ref_cen = make_complex_seq(3000, seed=801)
    ins_600 = make_complex_seq(600, seed=802)
    flank_600 = make_complex_seq(600, seed=803)
    flank_300 = make_complex_seq(300, seed=804)

    b.add_node("u1", make_complex_seq(100, seed=800))
    b.add_node("cen_ref", ref_cen)
    b.add_node("v1", make_complex_seq(100, seed=805))
    # C8: 500bp ref homology + 600bp novel insertion + 500bp ref homology
    b.add_node("c8_node", ref_cen[200:700] + ins_600 + ref_cen[700:1200])
    # C9: 600bp novel + 500bp reverse-complement ref homology + 300bp novel (<500bp short residual)
    b.add_node("c9_node", flank_600 + revcomp_dna(ref_cen[1500:2000]) + flank_300)

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapC8#1#ctg1", [("u1", "+"), ("c8_node", "+"), ("v1", "+")])
    b.add_p_path("HapC9#1#ctg1", [("u1", "+"), ("c9_node", "+"), ("v1", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 3100, "CEN_C8_C9")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    rows = {r.hap: r for r in out.centromere_rows}
    assert rows["HapC8#1"].nonref_input_bp == 1600
    assert rows["HapC8#1"].reference_like_bp == 1000
    assert rows["HapC8#1"].accepted_source_bp == 600

    assert rows["HapC9#1"].nonref_input_bp == 1400
    assert rows["HapC9#1"].reference_like_bp == 500
    assert rows["HapC9#1"].accepted_source_bp == 600
    assert rows["HapC9#1"].short_bp == 300
    assert any(
        r.reason == "CEN_SHORT_RESIDUAL" and r.length == 300 and r.export_to_fasta
        for r in out.unresolved_records
    )


def test_extract_islands_step_c_satellite_array_with_embedded_1kb_island_c16(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=810))
    b.add_node("cen_ref", make_complex_seq(1000, seed=811))
    b.add_node("v1", make_complex_seq(100, seed=812))

    sat_left = make_satellite_seq(unit_len=25, total_len=2000, seed=813)
    island_1500 = make_complex_seq(1500, seed=814)
    sat_right = make_satellite_seq(unit_len=25, total_len=2500, seed=815)
    composite_seq = sat_left + island_1500 + sat_right
    # Split composite_seq across three 2000 bp GFA nodes so interval boundaries fall inside nodes
    b.add_node("n_part1", composite_seq[0:2000])
    b.add_node("n_part2", composite_seq[2000:4000])
    b.add_node("n_part3", composite_seq[4000:6000])

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path(
        "HapC16#1#ctg1",
        [("u1", "+"), ("n_part1", "+"), ("n_part2", "+"), ("n_part3", "+"), ("v1", "+")],
    )

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 1100, "CEN_C16")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    row = next(r for r in out.centromere_rows if r.hap == "HapC16#1")
    assert row.nonref_input_bp == 6000
    assert row.repeat_like_bp == 4500
    assert row.accepted_source_bp == 1500
    assert len(out.island_sv_events) == 1
    ev = out.island_sv_events[0]
    assert ev.pieces[0].seq == island_1500
    assert (ev.pieces[0].src_start, ev.pieces[0].src_end) == (2100, 3600)


def test_extract_islands_step_c_short_blocks_and_sequence_uncertain_c10_c11_c19(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=820))
    b.add_node("cen_ref", make_complex_seq(800, seed=821))
    b.add_node("v1", make_complex_seq(100, seed=822))

    # 500 bp low-singleton (<0.30) vs 500 bp exact 0.30 singleton_ratio (500 - 14 = 486 valid; 0.30 -> 146 singletons >= 0.30)
    b.add_node("low_500", make_satellite_seq(20, 500, seed=823))
    b.add_node("low_999", make_satellite_seq(20, 999, seed=824))
    b.add_node("low_1000", make_satellite_seq(20, 1000, seed=825))
    b.add_node("low_1001", make_satellite_seq(20, 1001, seed=826))
    # 1000 bp window with exact 0.30 ratio: N_pos = 986; 296 / 986 = 0.3002 >= 0.30
    b.add_node(
        "pass_1000",
        make_window_with_exact_singleton_ratio(1000, target_singletons=296, seed=827),
    )
    # 600 bp block containing ambiguous IUPAC character 'R' -> CEN_SEQUENCE_UNCERTAIN
    b.add_node(
        "ambig_600",
        make_complex_seq(300, seed=828) + "R" + make_complex_seq(299, seed=829),
    )

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapLow500#1#ctg1", [("u1", "+"), ("low_500", "+"), ("v1", "+")])
    b.add_p_path("HapLow999#1#ctg1", [("u1", "+"), ("low_999", "+"), ("v1", "+")])
    b.add_p_path("HapLow1000#1#ctg1", [("u1", "+"), ("low_1000", "+"), ("v1", "+")])
    b.add_p_path("HapLow1001#1#ctg1", [("u1", "+"), ("low_1001", "+"), ("v1", "+")])
    b.add_p_path("HapPass1000#1#ctg1", [("u1", "+"), ("pass_1000", "+"), ("v1", "+")])
    b.add_p_path("HapAmbig#1#ctg1", [("u1", "+"), ("ambig_600", "+"), ("v1", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 900, "CEN_C10_C11")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    rows = {r.hap: r for r in out.centromere_rows}
    assert rows["HapLow500#1"].repeat_like_bp == 500
    assert rows["HapLow999#1"].repeat_like_bp == 999
    assert rows["HapLow1000#1"].repeat_like_bp == 1000
    assert rows["HapLow1001#1"].repeat_like_bp == 1001
    assert rows["HapPass1000#1"].accepted_source_bp == 1000
    assert rows["HapAmbig#1"].unresolved_bp == 600
    assert any(
        r.reason == "CEN_SEQUENCE_UNCERTAIN" and r.length == 600 and r.export_to_fasta
        for r in out.unresolved_records
    )


def test_extract_islands_step_d_clustering_ordering_mounting_and_genotypes_c12_c13_c14(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("u1", make_complex_seq(100, seed=900))
    b.add_node("cen_ref1", make_complex_seq(400, seed=901))
    b.add_node("cen_ref2", make_complex_seq(400, seed=902))
    b.add_node("v1", make_complex_seq(100, seed=903))

    island_a = make_complex_seq(800, seed=910)
    island_a_snp = mutate_seq_snps(island_a, [i * 50 + 10 for i in range(15)])
    island_b = island_a[:300] + make_complex_seq(100, seed=911) + island_a[400:]

    b.add_node("isl_a", island_a)
    b.add_node("isl_a_snp", island_a_snp)
    b.add_node("isl_b", island_b)

    b.add_p_path(
        "Ref#1#Chr01",
        [("u1", "+"), ("cen_ref1", "+"), ("cen_ref2", "+"), ("v1", "+")],
    )
    # Hap1#1 carries both island_a and island_b (separated by cen_ref1)
    b.add_p_path(
        "Hap1#1#ctg1",
        [("u1", "+"), ("isl_a", "+"), ("cen_ref1", "+"), ("isl_b", "+"), ("v1", "+")],
    )
    # Hap2#1 carries island_a_snp (merges with island_a)
    b.add_p_path("Hap2#1#ctg1", [("u1", "+"), ("isl_a_snp", "+"), ("v1", "+")])
    # Hap3#1 is complete with 0 islands
    b.add_p_path("Hap3#1#ctg1", [("u1", "+"), ("cen_ref1", "+"), ("v1", "+")])
    # Hap4#1 is partial (only hits u1)
    b.add_p_path("Hap4#1#ctg1", [("u1", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 900, "CEN_D")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    assert len(out.island_sv_events) == 2
    ev1, ev2 = out.island_sv_events
    assert ev1.event_id == "Chr01_CEN_ISLAND_1"
    assert ev1.n_carrier == 2
    assert ev1.n_called == 4
    assert ev1.genotypes == {
        "Ref#1": "0",
        "Hap1#1": "1",
        "Hap2#1": "1",
        "Hap3#1": "0",
        "Hap4#1": ".",
    }
    assert ev2.event_id == "Chr01_CEN_ISLAND_2"
    assert ev2.n_carrier == 1
    assert ev2.n_called == 4
    assert ev2.genotypes == {
        "Ref#1": "0",
        "Hap1#1": "1",
        "Hap2#1": "0",
        "Hap3#1": "0",
        "Hap4#1": ".",
    }

    assert len(out.island_slot_blocks) == 1
    slot = out.island_slot_blocks[0]
    assert slot.right_anchor_ref_step == regs[0].anchor_v_step
    assert slot.is_centromere_island is True
    assert slot.trailing_spacer is True
    assert len(slot.representatives) == 2
    for rep in slot.representatives:
        assert rep[0].join_before == "spacer"
        assert rep[0].allele_idx == 1

    rows = {r.hap: r for r in out.centromere_rows}
    assert rows["Hap1#1"].supported_events == "Chr01_CEN_ISLAND_1,Chr01_CEN_ISLAND_2"
    assert rows["Hap2#1"].supported_events == "Chr01_CEN_ISLAND_1"
    assert rows["Hap3#1"].supported_events == ""
    assert rows["Ref#1"].supported_events == ""
    assert rows["Hap4#1"].supported_events == "."


def test_extract_islands_five_way_accounting_invariant_and_27_col_centromere_row(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    ref_cen1 = make_complex_seq(600, seed=920)
    ref_cen2 = make_complex_seq(600, seed=921)
    b.add_node("u1", make_complex_seq(100, seed=922))
    b.add_node("cen_ref1", ref_cen1)
    b.add_node("cen_ref2", ref_cen2)
    b.add_node("v1", make_complex_seq(100, seed=923))

    # Block 1: 300 bp complex (< 500 -> short_bp += 300)
    b.add_node("blk1_short", make_complex_seq(300, seed=924))
    # Block 2: 600 bp with IUPAC 'R' at index 300 (-> unresolved_bp += 600, CEN_SEQUENCE_UNCERTAIN)
    b.add_node(
        "blk2_unc",
        make_complex_seq(300, seed=925) + "R" + make_complex_seq(299, seed=926),
    )
    # Block 3 (2600 bp):
    # [0, 1000) is 1000 bp satellite -> Step B residual 1 -> Step C masks 1000 bp (repeat_like_bp += 1000)
    # [1000, 1250) matches ref_cen1[0:250] -> Step B masks 250 bp (reference_like_bp += 250)
    # [1250, 1550) is 300 bp complex -> Step B residual 2 (< 500 bp) -> short_bp += 300
    # [1550, 1800) matches ref_cen2[0:250] -> Step B masks 250 bp (reference_like_bp += 250)
    # [1800, 2600) is 800 bp complex -> Step B residual 3 -> Step C passes -> accepted_source_bp += 800
    blk3_seq = (
        ("A" * 1000)
        + ref_cen1[0:250]
        + make_complex_seq(300, seed=927)
        + ref_cen2[0:250]
        + make_complex_seq(800, seed=928)
    )
    b.add_node("blk3_mix", blk3_seq)

    b.add_p_path(
        "Ref#1#Chr01",
        [("u1", "+"), ("cen_ref1", "+"), ("cen_ref2", "+"), ("v1", "+")],
    )
    b.add_p_path(
        "HapMix#1#ctg1",
        [
            ("u1", "+"),
            ("blk1_short", "+"),
            ("cen_ref1", "+"),
            ("blk2_unc", "+"),
            ("cen_ref2", "+"),
            ("blk3_mix", "+"),
            ("v1", "+"),
        ],
    )

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 1300, "CEN_5WAY")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    row = next(r for r in out.centromere_rows if r.hap == "HapMix#1")
    assert row.nonref_input_bp == 3500
    assert row.reference_like_bp == 500
    assert row.repeat_like_bp == 1000
    assert row.short_bp == 600
    assert row.unresolved_bp == 600
    assert row.accepted_source_bp == 800
    assert (
        row.reference_like_bp
        + row.repeat_like_bp
        + row.short_bp
        + row.unresolved_bp
        + row.accepted_source_bp
        == row.nonref_input_bp
    )
    assert len(row.to_tsv_fields()) == 27


def test_extract_islands_blind_test_categories_and_json_log_schema_c15(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    ref_cen = make_complex_seq(2000, seed=940)
    b.add_node("u1", make_complex_seq(100, seed=941))
    b.add_node("cen_ref", ref_cen)
    b.add_node("v1", make_complex_seq(100, seed=942))

    # 1. Pure satellite (1500 bp) -> CEN_REPEAT_LIKE
    b.add_node("c15_pure_sat", make_satellite_seq(25, 1500, seed=943))
    # 2. Old TE matching reference centromere (800 bp) -> CEN_REFERENCE_LIKE
    b.add_node("c15_old_te", ref_cen[400:1200])
    # 3. Private TE (900 bp complex) -> ACCEPTED_ISLAND
    b.add_node("c15_priv_te", make_complex_seq(900, seed=944))
    # 4. Hybrid repeat (1500 bp = 2 adjacent 1kb windows) + island (700 bp) -> split into CEN_REPEAT_LIKE + ACCEPTED_ISLAND
    b.add_node(
        "c15_hybrid",
        make_satellite_seq(20, 1500, seed=945) + make_complex_seq(700, seed=946),
    )
    # 5. Short block (250 bp) -> CEN_SHORT_RESIDUAL
    b.add_node("c15_short", make_complex_seq(250, seed=947))

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("Hap1#1#ctg1", [("u1", "+"), ("c15_pure_sat", "+"), ("v1", "+")])
    b.add_p_path("Hap2#1#ctg1", [("u1", "+"), ("c15_old_te", "+"), ("v1", "+")])
    b.add_p_path("Hap3#1#ctg1", [("u1", "+"), ("c15_priv_te", "+"), ("v1", "+")])
    b.add_p_path("Hap4#1#ctg1", [("u1", "+"), ("c15_hybrid", "+"), ("v1", "+")])
    b.add_p_path("Hap5#1#ctg1", [("u1", "+"), ("c15_short", "+"), ("v1", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 2100, "CEN_C15")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    expected_keys = {
        "region_id",
        "hap",
        "candidate_id",
        "parent_id",
        "src_path",
        "src_start",
        "src_end",
        "src_strand",
        "length",
        "revisited_fraction",
        "masked_intervals",
        "window_singleton_ratios",
        "final_reason",
        "representative_event_id",
    }
    assert len(out.json_log_entries) >= 5
    for entry in out.json_log_entries:
        assert set(entry.keys()) == expected_keys
        json.dumps(entry)
        if entry["final_reason"] in (
            "CEN_SHORT_RESIDUAL",
            "CEN_SEQUENCE_UNCERTAIN",
            "CEN_REFERENCE_LIKE",
        ):
            assert entry["window_singleton_ratios"] is None

    # Child residual from Hap4#1 hybrid must reference raw candidate_id as parent_id
    hap4_entries = [e for e in out.json_log_entries if e["hap"] == "Hap4#1"]
    parent_entry = next(
        e for e in hap4_entries if e["final_reason"] == "SPLIT_BY_REPEAT_MASK"
    )
    child_entry = next(
        e for e in hap4_entries if e["final_reason"] == "ACCEPTED_ISLAND"
    )
    assert child_entry["parent_id"] == parent_entry["candidate_id"]


def test_extract_islands_zero_retained_candidates_adds_no_slot_block_or_spacers(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    ref_cen = make_complex_seq(1000, seed=960)
    b.add_node("u1", make_complex_seq(100, seed=961))
    b.add_node("cen_ref", ref_cen)
    b.add_node("v1", make_complex_seq(100, seed=962))
    b.add_node("sat_only", make_satellite_seq(20, 1200, seed=963))
    b.add_node("ref_only", ref_cen[100:800])

    b.add_p_path("Ref#1#Chr01", [("u1", "+"), ("cen_ref", "+"), ("v1", "+")])
    b.add_p_path("HapA#1#ctg1", [("u1", "+"), ("sat_only", "+"), ("v1", "+")])
    b.add_p_path("HapB#1#ctg1", [("u1", "+"), ("ref_only", "+"), ("v1", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 100, 1100, "CEN_ZERO")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    assert out.island_slot_blocks == ()
    assert out.island_sv_events == ()
    for r in out.centromere_rows:
        assert r.accepted_source_bp == 0
        assert r.supported_events == ""


def test_partition_fragments_boundary_pure_insertions_at_ref_end_u_and_ref_start_v_c14(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for nid, length, seed in [
        ("n0", 200, 101),
        ("u", 200, 102),
        ("c1", 200, 103),
        ("c2", 200, 106),
        ("v", 400, 104),
        ("n4", 300, 105),
        ("isl1", 650, 201),
        ("norm_ins_v", 180, 202),
        ("norm_ins_u", 160, 203),
        ("cen_part", 550, 204),
    ]:
        b.add_node(nid, make_complex_seq(length, seed=seed))

    b.add_p_path(
        "Ref#1#Chr01",
        [("n0", "+"), ("u", "+"), ("c1", "+"), ("c2", "+"), ("v", "+"), ("n4", "+")],
    )
    b.add_p_path(
        "Donor1#1#Chr01",
        [
            ("n0", "+"),
            ("u", "+"),
            ("c1", "+"),
            ("isl1", "+"),
            ("c2", "+"),
            ("v", "+"),
            ("n4", "+"),
        ],
    )
    # Partial contig starting at c2+ with a pure insertion at ref_start(v) = 800
    b.add_p_path(
        "Donor2#1#Chr01",
        [("c2", "+"), ("norm_ins_v", "+"), ("v", "+"), ("n4", "+")],
    )
    # Partial contig ending at c1+ with a pure insertion at ref_end(u) = 400
    b.add_p_path(
        "Donor3#1#Chr01",
        [("n0", "+"), ("u", "+"), ("norm_ins_u", "+"), ("c1", "+")],
    )
    # Partial contig with both an internal centromere slice (c1+ -> cen_part+ -> c2+) and a boundary pure insertion at v
    b.add_p_path(
        "Donor4#1#Chr01",
        [
            ("c1", "+"),
            ("cen_part", "+"),
            ("c2", "+"),
            ("norm_ins_v", "+"),
            ("v", "+"),
            ("n4", "+"),
        ],
    )

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 450, 750, "CEN1")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    norm_by_hap: dict[str, list[list[str]]] = {}
    for f in part.normal_fragments:
        norm_by_hap.setdefault(f.hap_id, []).append([s.node_id for s in f.steps])

    assert norm_by_hap["Donor1#1"] == [["n0", "u"], ["v", "n4"]]
    assert norm_by_hap["Donor2#1"] == [["c2", "norm_ins_v", "v", "n4"]]
    assert norm_by_hap["Donor3#1"] == [["n0", "u", "norm_ins_u", "c1"]]
    assert norm_by_hap["Donor4#1"] == [["c1"], ["c2", "norm_ins_v", "v", "n4"]]

    # Donor2#1 and Donor3#1 have no internal centromere unresolved records; Donor4#1 emits CEN_INCOMPLETE for cen_part
    assert [(r.reason, r.src_path, r.length) for r in out.unresolved_records] == [
        ("CEN_INCOMPLETE", "Donor4#1#Chr01", 550)
    ]


def test_partition_fragments_single_sided_unanchored_ends_not_dropped_dicentric(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    for nid, length, seed in [
        ("r0", 250, 901),
        ("u1", 250, 902),
        ("cen_rep", 300, 903),
        ("v1", 250, 904),
        ("r_mid", 250, 905),
        ("u2", 250, 906),
        ("cen_end", 300, 907),
        ("isl_r1", 600, 908),
        ("isl_r2", 600, 909),
        ("lead_nov", 500, 910),
    ]:
        b.add_node(nid, make_complex_seq(length, seed=seed))

    b.add_p_path(
        "Ref#1#Chr01",
        [
            ("r0", "+"),
            ("u1", "+"),
            ("cen_rep", "+"),
            ("cen_rep", "+"),
            ("v1", "+"),
            ("r_mid", "+"),
            ("u2", "+"),
            ("cen_end", "+"),
        ],
    )
    b.add_p_path(
        "Donor#1#Chr01",
        [
            ("r0", "+"),
            ("u1", "+"),
            ("cen_rep", "+"),
            ("isl_r1", "+"),
            ("cen_rep", "+"),
            ("v1", "+"),
            ("r_mid", "+"),
            ("u2", "+"),
            ("cen_end", "+"),
            ("isl_r2", "+"),
        ],
    )
    b.add_p_path(
        "DonorLead#1#Chr01",
        [("lead_nov", "+"), ("cen_rep", "+"), ("v1", "+"), ("r_mid", "+")],
    )

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[
            ("Chr01", 520, 750, "BED_1A"),
            ("Chr01", 820, 1080, "BED_1B"),
            ("Chr01", 1850, 2150, "BED_2_END"),
        ],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)
    out = process_centromere_regions(part, graph, cfg)

    donor_frags = [
        [s.node_id for s in f.steps]
        for f in part.normal_fragments
        if f.hap_id == "Donor#1"
    ]
    assert donor_frags == [["r0", "u1"], ["v1", "r_mid", "u2"], ["cen_end", "isl_r2"]]
    assert part.hap_slices[("CEN_2", "Donor#1")].extra_inner_slices == ()
    assert out.unresolved_records == ()

    lead_frags = [
        [s.node_id for s in f.steps]
        for f in part.normal_fragments
        if f.hap_id == "DonorLead#1"
    ]
    assert lead_frags == [["lead_nov", "cen_rep"], ["v1", "r_mid"]]


def test_partition_fragments_missing_flank_status_never_ambiguous_c5(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    b = gfa_builder()
    b.add_node("r", make_complex_seq(200, seed=1))
    b.add_node("v", make_complex_seq(200, seed=2))
    b.add_node("other", make_complex_seq(200, seed=3))
    b.add_p_path("Ref#1#Chr01", [("r", "+"), ("r", "+"), ("v", "+")])
    # Visits v twice when u is VIRTUAL_START -> must still be partial, never ambiguous
    b.add_p_path("AltDupV#1#Chr01", [("r", "+"), ("v", "+"), ("v", "+")])
    b.add_p_path("AltUnmap#1#Chr01", [("other", "+")])

    graph, cfg = _build_loaded_graph(
        tmp_path,
        b,
        bed_rows=[("Chr01", 50, 350, "CEN_NO_LEFT")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
    )
    regs, _ = delimit_centromere_regions(graph, cfg)
    part = partition_fragments_by_centromere(graph.alt_fragments, regs, graph, cfg)

    assert regs[0].anchor_u_step is None
    assert regs[0].effective_mode == "bypass"
    assert part.hap_slices[("CEN_1", "AltDupV#1")].status == "partial"
    assert part.hap_slices[("CEN_1", "AltUnmap#1")].status == "unmapped"

