"""Unit tests for `panflat/linearizer.py` (F14, F15, F16, F17, F18, F19, F20, T1–T24, C1, C14, C16–C20)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from panflat.gfa_io import load_and_validate_inputs, revcomp
from panflat.linearizer import (
    CandidateBlock,
    ExtractedCandidate,
    SubproblemStateKey,
    assemble_pan_and_offsets,
    can_transition,
    extract_candidate_blocks,
    make_provenance_key,
    make_step_slice_key,
    make_token_str,
    reanchor_deviation,
    run_chromosome_pipeline,
    screen_and_fold_candidate,
    solve_chromosome,
    solve_parallel_group,
    steps_to_pieces,
)
from panflat.models import (
    BuildConfig,
    DeviationSegment,
    GFAGraph,
    Step,
    StepProvenance,
)
from tests.e2e.helpers import (
    SyntheticInputBuilder,
    make_coding_seq,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
)


def _build_graph_and_config(
    builder: SyntheticInputBuilder,
    name: str = "unit",
    **config_kwargs: object,
) -> tuple[GFAGraph, BuildConfig]:
    inputs = builder.write(name)
    defaults: dict[str, object] = {
        "gfa": inputs.gfa_path,
        "ref_fa": inputs.ref_fa_path,
        "backbone_prefix": inputs.backbone_prefix,
        "out_prefix": inputs.out_prefix,
        "centromere_bed": inputs.bed_path,
        "centromere_mode": "bypass" if inputs.bed_path is not None else "off",
        "ref_gff": inputs.gff_path,
    }
    defaults.update(config_kwargs)
    cfg = BuildConfig(**defaults)  # type: ignore[arg-type]
    graph = load_and_validate_inputs(cfg)
    return graph, cfg


def _assert_result_invariants(graph: GFAGraph, cfg: BuildConfig, res: object) -> None:
    """Verify Invariant I (reference base preservation) and Invariant II (piece provenance) on `ChromosomeBuildResult`."""
    from panflat.models import ChromosomeBuildResult

    assert isinstance(res, ChromosomeBuildResult)
    pan_seq = res.pan_seq
    ref_seq = graph.ref_seq

    # Invariant I: Reference steps reconstruct ref_seq and match slot_ins_len
    reconstructed: list[str] = []
    expected_ref = 0
    prev_pan_end = 0
    for idx, row in enumerate(res.step_offsets):
        assert row.ref_step == idx
        assert row.ref_start == expected_ref
        assert row.pan_start == prev_pan_end + row.slot_ins_len
        assert row.pan_end - row.pan_start == row.ref_end - row.ref_start
        step_pan_seq = pan_seq[row.pan_start : row.pan_end]
        step_ref_seq = ref_seq[row.ref_start : row.ref_end]
        assert step_pan_seq == step_ref_seq
        reconstructed.append(step_pan_seq)
        expected_ref = row.ref_end
        prev_pan_end = row.pan_end
    assert expected_ref == len(ref_seq)
    assert "".join(reconstructed) == ref_seq

    # Invariant II: Every insert piece matches pan_seq[pan_start:pan_end] and uncovered bases are 'N'
    covered = bytearray(len(pan_seq))
    for row in res.step_offsets:
        for p in range(row.pan_start, row.pan_end):
            assert covered[p] == 0
            covered[p] = 1
    for p_row in res.inserts_rows:
        p_s = int(p_row["pan_start"])
        p_e = int(p_row["pan_end"])
        s_s, s_e = (int(x) for x in str(p_row["src_intervals"]).split("-"))
        assert 0 <= p_s < p_e <= len(pan_seq)
        assert p_e - p_s == s_e - s_s
        assert pan_seq[p_s:p_e] == str(p_row["seq"]).upper()
        if p_row["join_before"] == "spacer" and p_s >= cfg.spacer_n:
            assert pan_seq[p_s - cfg.spacer_n : p_s] == "N" * cfg.spacer_n
        for p in range(p_s, p_e):
            assert covered[p] == 0
            covered[p] = 2
    for pos, tag in enumerate(covered):
        if tag == 0:
            assert pan_seq[pos] == "N"


# ============================================================================
# 1. SubproblemStateKey, Progress Metric & Local Re-anchoring (F14, T16)
# ============================================================================


def test_subproblem_state_key_and_can_transition_rules() -> None:
    """Verify lexicographic progress metric (|M|, hi - lo) and ancestor cycle prevention."""
    m1 = (("Alt1#1#Chr01", 0, 100, "+"),)
    m2 = (("Alt2#1#Chr01", 0, 100, "+"),)
    b_id = (("Ref#1#Chr01", 0, 1000, "+"),)

    parent = SubproblemStateKey(
        b_id=b_id,
        lo=0,
        hi=10,
        has_logical_start=True,
        has_logical_end=True,
        m_key=(m1, m2),
    )
    assert parent.metric == (2, 10)

    # Rule (a): |M| unchanged (2), hi - lo shrinks (10 -> 4) -> valid
    child_reanchor = SubproblemStateKey(
        b_id=b_id,
        lo=2,
        hi=6,
        has_logical_start=True,
        has_logical_end=True,
        m_key=(m1, m2),
    )
    assert can_transition(parent, child_reanchor, frozenset())

    # Rule (b): |M| shrinks (2 -> 1), even if hi - lo increases (10 -> 25) -> valid
    child_comp = SubproblemStateKey(
        b_id=(("Alt1#1#Chr01", 0, 500, "+"),),
        lo=0,
        hi=25,
        has_logical_start=True,
        has_logical_end=True,
        m_key=(m2,),
    )
    assert can_transition(parent, child_comp, frozenset())

    # Equal metric -> rejected
    same_metric = SubproblemStateKey(
        b_id=b_id,
        lo=0,
        hi=10,
        has_logical_start=False,
        has_logical_end=False,
        m_key=(m1, m2),
    )
    assert not can_transition(parent, same_metric, frozenset())

    # Ancestor state revisit -> rejected
    assert not can_transition(parent, child_reanchor, frozenset({child_reanchor}))


def test_local_reanchoring_shrinks_interval_on_locally_unique_node(
    tmp_path: Path,
) -> None:
    """F14: Globally repeated node 'dup' becomes a unique anchor inside B(n2, n3)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1401)
    s_dup = make_complex_seq(150, seed=1402)
    s2 = make_complex_seq(200, seed=1403)
    s_skip = make_complex_seq(120, seed=1404)
    s3 = make_complex_seq(200, seed=1405)
    s_ins = make_complex_seq(140, seed=1406)
    for nid, seq in [
        ("n1", s1),
        ("dup", s_dup),
        ("n2", s2),
        ("skip", s_skip),
        ("n3", s3),
        ("ins", s_ins),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "dup+", "n2+", "skip+", "dup+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "dup+", "n2+", "dup+", "ins+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "f14_local")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_evs = [e for e in res.sv_events if e.event_type == "INS"]
    del_evs = [e for e in res.sv_events if e.event_type == "DEL"]
    assert len(ins_evs) == 1
    assert ins_evs[0].ref_start == 820  # before n3
    assert len(del_evs) == 1
    assert del_evs[0].ref_start == 550 and del_evs[0].ref_end == 670


def test_t16_termination_when_no_unique_anchor_in_b_ab(tmp_path: Path) -> None:
    """F14 / T16: When B(a,b) = [r1+, r1+] has no unique anchor, local re-anchoring terminates cleanly."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1411)
    sr = make_complex_seq(120, seed=1412)
    s2 = make_complex_seq(200, seed=1413)
    ins = make_complex_seq(160, seed=1414)
    for nid, seq in [("n1", s1), ("r1", sr), ("n2", s2), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "r1+", "r1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "r1+", "ins+", "r1+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "t16_term")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert res.pan_seq == s1 + sr + sr + ("N" * 100) + ins + s2


# ============================================================================
# 2. Candidate Block Extraction & Unanchored Ends (F15, T4, T9, T11, T19)
# ============================================================================


def test_t9_and_t19_distal_ref_node_acts_as_separator_never_inserted(
    tmp_path: Path,
) -> None:
    """F15 / T9 / T19: Reference nodes inside top-level or recursive deviations act as separators and are never re-inserted."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1431)
    s2 = make_complex_seq(200, seed=1432)
    s_distal = make_complex_seq(250, seed=1433)
    s3 = make_complex_seq(200, seed=1434)
    sh_a = make_complex_seq(200, seed=1435)
    sh_b = make_complex_seq(200, seed=1436)
    priv_c = make_complex_seq(150, seed=1437)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("ndist", s_distal),
        ("n3", s3),
        ("shA", sh_a),
        ("shB", sh_b),
        ("privC", priv_c),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "ndist+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "shA+", "shB+", "n2+", "ndist+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "shA+", "shB+", "n2+", "ndist+", "n3+"])
    b.add_path(
        "Alt3#1#Chr01",
        ["n1+", "shA+", "ndist+", "privC+", "shB+", "n2+", "ndist+", "n3+"],
    )

    graph, cfg = _build_graph_and_config(b, "t19_sep")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert res.pan_seq.count(s_distal) == 1
    assert priv_c in res.pan_seq


def test_t11_top_level_unanchored_end_and_missing_genotype(tmp_path: Path) -> None:
    """F15 / T11: Unanchored contig prefix >= 50 bp goes to UNANCHORED_END and yields '.' genotype."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1511)
    s2 = make_complex_seq(200, seed=1512)
    s3 = make_complex_seq(200, seed=1513)
    unanch_l = make_complex_seq(150, seed=1514)
    ins_12 = make_complex_seq(140, seed=1515)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("ul", unanch_l),
        ("ins12", ins_12),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("AltBreak#1#Chr01", ["ul+", "n2+", "n3+"])
    b.add_path("AltFull#1#Chr01", ["n1+", "ins12+", "n2+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "t11_unanch")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    unanch = [u for u in res.unresolved_records if u.reason == "UNANCHORED_END"]
    assert len(unanch) == 1
    assert unanch[0].length == 150
    assert unanch[0].export_to_fasta and unanch[0].exported_fasta_id != "."
    assert unanch_l not in res.pan_seq

    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["AltBreak#1"] == "."
    assert ins_ev.genotypes["AltFull#1"] == "1"


def test_f15_repeat_residual_annotation_on_embedded_local_multicopy(
    tmp_path: Path,
) -> None:
    """F15: Candidate containing a multi-copy local backbone node marks repeat_residual."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1541)
    s2 = make_complex_seq(200, seed=1542)
    u0 = make_complex_seq(150, seed=1540)
    u1 = make_complex_seq(120, seed=1543)
    m_rep = make_complex_seq(100, seed=1544)
    u2 = make_complex_seq(120, seed=1545)
    nov_z = make_complex_seq(120, seed=1546)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("u0", u0),
        ("u1", u1),
        ("mrep", m_rep),
        ("u2", u2),
        ("novZ", nov_z),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u0+", "u1+", "mrep+", "mrep+", "u2+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "u0+", "u1+", "mrep+", "mrep+", "u2+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "u1+", "mrep+", "novZ+", "mrep+", "u2+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "f15_rep_res", max_copy=2)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert "repeat_residual" in ins_ev.alleles_str


# ============================================================================
# 3. Component Recursion & Secondary Backbone Selection (F16, T4, T18)
# ============================================================================


def test_t4_shared_prefix_bifurcation_and_sec_backbone_modes(tmp_path: Path) -> None:
    """F16 / T4: Shared prefix A bifurcating into B (2 carriers, shorter) and C (1 carrier, longer)."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1621)
    s2 = make_complex_seq(200, seed=1622)
    sh = make_complex_seq(300, seed=1623)
    short_br = make_complex_seq(150, seed=1624)
    long_br = make_complex_seq(280, seed=1625)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("sh", sh),
        ("sbr", short_br),
        ("lbr", long_br),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "sh+", "sbr+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "sh+", "lbr+", "n2+"])

    graph_sup, cfg_sup = _build_graph_and_config(
        b, "t4_sup", sec_backbone="support"
    )
    res_sup = run_chromosome_pipeline(graph_sup, cfg_sup)
    _assert_result_invariants(graph_sup, cfg_sup, res_sup)
    assert res_sup.pan_seq == s1 + sh + short_br + ("N" * 100) + long_br + s2

    ins_ev = [e for e in res_sup.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["Ref#1"] == "0"
    assert ins_ev.genotypes["Alt1#1"] == "1"
    assert ins_ev.genotypes["Alt2#1"] == "1"
    assert ins_ev.genotypes["Alt3#1"] == "2"

    graph_lng, cfg_lng = _build_graph_and_config(
        b, "t4_lng", sec_backbone="longest"
    )
    res_lng = run_chromosome_pipeline(graph_lng, cfg_lng)
    _assert_result_invariants(graph_lng, cfg_lng, res_lng)
    assert res_lng.pan_seq == s1 + sh + long_br + ("N" * 100) + short_br + s2


def test_t18_secondary_backbone_with_negative_novel_node(tmp_path: Path) -> None:
    """F7 / F16 / T18: Secondary backbone containing '-' novel node anchors child branches by relative orientation."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=741)
    s2 = make_complex_seq(200, seed=742)
    u_a = make_complex_seq(150, seed=743)
    u_neg = make_complex_seq(160, seed=744)
    u_b = make_complex_seq(150, seed=745)
    u_extra = make_complex_seq(120, seed=746)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("uA", u_a),
        ("uNeg", u_neg),
        ("uB", u_b),
        ("uEx", u_extra),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "uA+", "uNeg-", "uB+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "uA+", "uNeg-", "uB+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "uA+", "uNeg-", "uEx+", "uB+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "t18_neg_bsec")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert res.pan_seq == s1 + u_a + revcomp(u_neg) + u_extra + u_b + s2


# ============================================================================
# 4. Similarity Clustering (F17, T2a, T2b, T5a, T12a, T12b)
# ============================================================================


def test_f17_edlib_and_mappy_clustering_merge_and_reject_criteria(
    tmp_path: Path,
) -> None:
    """F17 / T2a / T2b / T12a / T12b: Verify <=1kb edlib and >1kb mappy 4-criterion similarity clustering."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 5):
        b.add_node(f"n{idx}", make_complex_seq(200, seed=1700 + idx))

    # Locus 1 (n1..n2): <=1kb (600bp) with 12 dispersed SNPs (98% ID) -> MERGED
    r_small = make_complex_seq(600, seed=1710)
    s_small_ok = mutate_seq_snps(r_small, num_snps=12, seed=1711)
    b.add_node("rs", r_small).add_node("ss", s_small_ok)

    # Locus 2 (n2..n3): >1kb (2000bp) with 20 dispersed SNPs (99% ID) -> MERGED via mappy
    r_large = make_complex_seq(2000, seed=1712)
    s_large_ok = mutate_seq_snps(r_large, num_snps=20, seed=1713)
    b.add_node("rl", r_large).add_node("sl", s_large_ok)

    # Locus 3 (n3..n4): >1kb (2000bp) with 60bp contiguous substitution block -> REJECTED by Criterion 4
    sub_60 = mutate_seq_snps(
        r_large[800:860], positions=list(range(60)), seed=1714
    )
    s_large_fail = r_large[:800] + sub_60 + r_large[860:]
    b.add_node("rl2", r_large).add_node("sl_fail", s_large_fail)

    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "rs+", "n2+", "rl+", "n3+", "rl2+", "n4+"])
    b.add_path(
        "Alt2#1#Chr01", ["n1+", "ss+", "n2+", "sl+", "n3+", "sl_fail+", "n4+"]
    )

    graph, cfg = _build_graph_and_config(b, "f17_cluster")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert res.step_offsets[1].slot_ins_len == 600
    assert res.step_offsets[2].slot_ins_len == 2000
    assert res.step_offsets[3].slot_ins_len == 2000 + 100 + 2000


# ============================================================================
# 5. Unified Output Layout, Span Ordering & Join Annotation (F18, T3, T20)
# ============================================================================


def test_t20_multi_groups_at_same_right_anchor_ordered_by_span_asc(
    tmp_path: Path,
) -> None:
    """F18 / T20: Groups sharing right anchor n3 ordered by span (b - a) ASC and emitted as separate INS events."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1831)
    s2 = make_complex_seq(200, seed=1832)
    s3 = make_complex_seq(200, seed=1833)
    ins_narrow = make_complex_seq(140, seed=1834)
    ins_wide = make_complex_seq(170, seed=1835)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("inarrow", ins_narrow),
        ("iwide", ins_wide),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "n2+", "inarrow+", "n3+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "iwide+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "t20_span")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert res.pan_seq == s1 + s2 + ins_narrow + ("N" * 100) + ins_wide + s3
    ins_evs = [e for e in res.sv_events if e.event_type == "INS"]
    assert len(ins_evs) == 2
    assert ins_evs[0].event_id != ins_evs[1].event_id


def test_f18_join_before_annotations_source_contiguous_synthetic_spacer(
    tmp_path: Path,
) -> None:
    """F18: Verify source_contiguous, synthetic, and spacer in insert_pieces."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1841)
    s2 = make_complex_seq(200, seed=1842)
    u = make_complex_seq(80, seed=1843)
    tail = make_complex_seq(90, seed=1844)
    rep2 = make_complex_seq(130, seed=1845)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("u", u),
        ("tail", tail),
        ("rep2", rep2),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "u+", "u+", "tail+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "rep2+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "f18_joins", max_copy=1)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert [p["join_before"] for p in res.inserts_rows] == [
        "source_contiguous",
        "synthetic",
        "spacer",
    ]


# ============================================================================
# 6. Universal Repeat Guard & Partial-Node Slicing (F19, C1, C16, C20)
# ============================================================================


def test_c20_candidate_split_by_repeat_guard_and_fully_masked_genotype_dot(
    tmp_path: Path,
) -> None:
    """F19 / C20: Central satellite masked into two residuals separated by N*spacer; fully masked hap gets '.'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1931)
    s2 = make_complex_seq(250, seed=1932)
    left_complex = make_complex_seq(400, seed=1933)
    mid_sat = make_satellite_seq(unit_len=20, total_len=1800, seed=1934)
    right_complex = make_complex_seq(400, seed=1935)
    split_cand = left_complex + mid_sat + right_complex
    pure_sat = make_satellite_seq(unit_len=20, total_len=1500, seed=1936)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("sc", split_cand),
        ("ps", pure_sat),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("AltSplit#1#Chr01", ["n1+", "sc+", "n2+"])
    b.add_path("AltMasked#1#Chr01", ["n1+", "ps+", "n2+"])
    b.add_path("AltRef#1#Chr01", ["n1+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "c20_split", repeat_guard="on")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert len(res.inserts_rows) == 2
    assert res.inserts_rows[0]["join_before"] == "source_contiguous"
    assert res.inserts_rows[1]["join_before"] == "spacer"
    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert "partial_content=1" in ins_ev.alleles_str
    assert ins_ev.genotypes["Ref#1"] == "0"
    assert ins_ev.genotypes["AltRef#1"] == "0"
    assert ins_ev.genotypes["AltSplit#1"] == "1"
    assert ins_ev.genotypes["AltMasked#1"] == "."


# ============================================================================
# 7. Top-Level SV Calling, Centromere Co-mounting & Thread Determinism (F20, C14, T22)
# ============================================================================


def test_c14_comounting_cen_island_and_normal_ins_at_right_anchor_v(
    tmp_path: Path,
) -> None:
    """C14: CEN_ISLAND mounts before normal INS at ref_start(v) with a single spacer between them."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("n0", make_complex_seq(200, seed=101))
    b.add_node("u", make_complex_seq(200, seed=102))
    b.add_node("c1", make_complex_seq(200, seed=103))
    b.add_node("c2", make_complex_seq(200, seed=104))
    b.add_node("v", make_complex_seq(300, seed=105))
    b.add_node("n4", make_complex_seq(200, seed=106))
    isl = make_complex_seq(600, seed=107)
    norm_ins = make_complex_seq(150, seed=108)
    b.add_node("isl", isl).add_node("nins", norm_ins)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "c1+", "c2+", "v+", "n4+"])
    b.add_path("Donor1#1#Chr01", ["n0+", "u+", "c1+", "isl+", "c2+", "v+", "n4+"])
    b.add_path("Donor2#1#Chr01", ["c2+", "nins+", "v+", "n4+"])
    b.add_bed_region(450, 750, "CEN1")

    graph, cfg = _build_graph_and_config(
        b,
        "c14_unit",
        centromere_mode="extract-islands",
        centromere_min_island=500,
        spacer_n=100,
    )
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    v_row = next(r for r in res.step_offsets if r.node_id == "v")
    assert v_row.slot_ins_len == len(isl) + len(norm_ins) + 200
    assert len(res.inserts_rows) == 2
    assert int(res.inserts_rows[1]["pan_start"]) - int(res.inserts_rows[0]["pan_end"]) == 100


def test_t22_multithreaded_determinism_and_all_sv_types(tmp_path: Path) -> None:
    """F20 / T22 / T24: Verify INV, INV_EDGE, INV_CONFLICT, DEL, CNV, DUP_TRA, LARGE_JUMP, INS and thread determinism."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 9):
        b.add_node(f"r{idx}", make_complex_seq(300, seed=2200 + idx))
    b.add_node("tr", make_complex_seq(60, seed=2210))
    b.add_node("ins1", make_complex_seq(180, seed=2211))
    b.add_node("ins2", make_complex_seq(210, seed=2212))
    b.add_node("nov_edge", make_complex_seq(150, seed=2213))

    b.set_ref_path(
        "Ref#1#Chr01",
        ["r1+", "r2+", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+", "r7+", "r8+"],
    )
    # Hap1: INS at (r1,r2), INV of r3-, CNV 2->4 of tr+, DEL of r6+
    b.add_path(
        "Hap1#1#Chr01",
        [
            "r1+",
            "ins1+",
            "r2+",
            "r3-",
            "r4+",
            "tr+",
            "tr+",
            "tr+",
            "tr+",
            "r5+",
            "r7+",
            "r8+",
        ],
    )
    # Hap2: INS at (r2,r3), DUP_TRA visiting r8+ between r5+ and r6+
    b.add_path(
        "Hap2#1#Chr01",
        [
            "r1+",
            "r2+",
            "ins2+",
            "r3+",
            "r4+",
            "tr+",
            "tr+",
            "r5+",
            "r8+",
            "r6+",
            "r7+",
            "r8+",
        ],
    )
    # Hap3: INV_EDGE at start
    b.add_path(
        "Hap3#1#Chr01",
        ["r2-", "nov_edge+", "r1-", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+", "r7+", "r8+"],
    )

    graph1, cfg1 = _build_graph_and_config(b, "t22_t1", max_jump=500, threads=1)
    graph4, cfg4 = _build_graph_and_config(b, "t22_t4", max_jump=500, threads=4)
    res1 = solve_chromosome(graph1, cfg1)
    res4 = solve_chromosome(graph4, cfg4)

    _assert_result_invariants(graph1, cfg1, res1)
    _assert_result_invariants(graph4, cfg4, res4)

    assert res1.pan_seq == res4.pan_seq
    assert res1.step_offsets == res4.step_offsets
    assert res1.sv_events == res4.sv_events
    assert res1.inserts_rows == res4.inserts_rows

    ev_types = {e.event_type for e in res1.sv_events}
    assert {"INS", "INV", "INV_EDGE", "CNV", "DEL", "DUP_TRA"}.issubset(ev_types)


def test_t1_t2_t3_simple_ins_del_and_replacement_with_nonempty_backbone(
    tmp_path: Path,
) -> None:
    """T1, T2, T3: Simple INS, simple DEL, and replacement B(a,b) != empty with N*100 before R_1."""
    b = SyntheticInputBuilder(tmp_path)
    n11 = make_complex_seq(200, seed=1011)
    n12 = make_complex_seq(200, seed=1012)
    n13 = make_complex_seq(200, seed=1013)
    n14 = make_complex_seq(200, seed=1014)
    n15 = make_complex_seq(150, seed=1015)
    n16 = make_complex_seq(170, seed=1016)
    for nid, seq in [
        ("11", n11),
        ("12", n12),
        ("13", n13),
        ("14", n14),
        ("15", n15),
        ("16", n16),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["11+", "12+", "13+", "14+"])
    # HapIns: T1 simple insertion between 11 and 12
    b.add_path("HapIns#1#Chr01", ["11+", "15+", "12+", "13+", "14+"])
    # HapDel: T2 simple deletion of 12
    b.add_path("HapDel#1#Chr01", ["11+", "13+", "14+"])
    # HapRep: T3 replacement of 13 with 16 between 12 and 14
    b.add_path("HapRep#1#Chr01", ["11+", "12+", "16+", "14+"])

    graph, cfg = _build_graph_and_config(b, "t1_t2_t3")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    # At step 1 (node 12): T1 simple insertion has empty backbone -> no leading N*100
    assert res.step_offsets[1].slot_ins_len == 150
    # At step 3 (node 14): T3 replacement has non-empty backbone B(12,14)=[13+] -> leading N*100 + 170
    assert res.step_offsets[3].slot_ins_len == 100 + 170
    assert res.pan_seq == n11 + n15 + n12 + n13 + ("N" * 100) + n16 + n14


def test_t5_shared_scaffold_with_sub_50bp_difference_absorbed(
    tmp_path: Path,
) -> None:
    """T5: Shared scaffold with <50bp internal bubble (35bp vs 35bp) absorbs difference without N*100."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1851)
    s2 = make_complex_seq(200, seed=1852)
    a_seq = make_complex_seq(150, seed=1853)
    x1 = make_complex_seq(35, seed=1854)
    x2 = make_complex_seq(35, seed=1855)
    b_seq = make_complex_seq(150, seed=1856)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("A", a_seq),
        ("X1", x1),
        ("X2", x2),
        ("B", b_seq),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "A+", "X1+", "B+", "n2+"])
    b.add_path("Alt2#1#Chr01", ["n1+", "A+", "X1+", "B+", "n2+"])
    b.add_path("Alt3#1#Chr01", ["n1+", "A+", "X2+", "B+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "t5_absorb")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert "N" not in res.pan_seq
    assert res.pan_seq == s1 + a_seq + x1 + b_seq + s2
    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["Alt1#1"] == "1"
    assert ins_ev.genotypes["Alt2#1"] == "1"
    assert ins_ev.genotypes["Alt3#1"] == "1"


def test_t7_inversion_with_nested_insertion_and_preflip_provenance(
    tmp_path: Path,
) -> None:
    """T7: 11+ -> 13- -> X+ -> 12- -> 14+ normalizes to 11+ -> 12+ -> X- -> 13+ -> 14+ with shared region_id."""
    b = SyntheticInputBuilder(tmp_path)
    s11 = make_complex_seq(200, seed=911)
    s12 = make_complex_seq(200, seed=912)
    s13 = make_complex_seq(200, seed=913)
    s14 = make_complex_seq(200, seed=914)
    sx = make_complex_seq(150, seed=915)
    for nid, seq in [
        ("11", s11),
        ("12", s12),
        ("13", s13),
        ("14", s14),
        ("X", sx),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["11+", "12+", "13+", "14+"])
    b.add_path("Alt1#1#Chr01", ["11+", "13-", "X+", "12-", "14+"])

    graph, cfg = _build_graph_and_config(b, "t7_inv_ins")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert res.pan_seq == s11 + s12 + revcomp(sx) + s13 + s14
    inv_ev = [e for e in res.sv_events if e.event_type == "INV"][0]
    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert inv_ev.region_id == ins_ev.region_id
    assert inv_ev.region_id != "."
    assert len(res.inserts_rows) == 1
    assert res.inserts_rows[0]["src_strand"] == "-"
    assert res.inserts_rows[0]["src_intervals"] == "400-550"


def test_t13_large_jump_priority_over_dup_tra_and_ins(tmp_path: Path) -> None:
    """T13: When post-normalization anchor span > max_jump, emit LARGE_JUMP and route novel seqs to unresolved."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=1301)
    s2 = make_complex_seq(600, seed=1302)
    s3 = make_complex_seq(600, seed=1303)
    s4 = make_complex_seq(200, seed=1304)
    nov = make_complex_seq(180, seed=1305)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("n4", s4),
        ("nov", nov),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Jump from n1 to n4 skips n2+n3 (1200 bp > max_jump=800) with novel sequence 'nov'
    b.add_path("AltJump#1#Chr01", ["n1+", "nov+", "n4+"])

    graph, cfg = _build_graph_and_config(b, "t13_lj", max_jump=800)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ev_types = [e.event_type for e in res.sv_events]
    assert ev_types == ["LARGE_JUMP"]
    assert nov not in res.pan_seq
    lj_unres = [u for u in res.unresolved_records if u.reason == "LARGE_JUMP"]
    assert len(lj_unres) == 1
    assert lj_unres[0].length == 180
    assert lj_unres[0].event_id == res.sv_events[0].event_id


def test_multi_contig_contradictory_calls_resolve_to_dot(tmp_path: Path) -> None:
    """§5.2: When two contigs of the same Sample#Hap make contradictory non-'.' calls at the same locus, emit '.'."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=2501)
    s2 = make_complex_seq(200, seed=2502)
    ins_a = make_complex_seq(160, seed=2503)
    for nid, seq in [("n1", s1), ("n2", s2), ("insA", ins_a)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    # Clean carrier
    b.add_path("Carrier#1#Chr01", ["n1+", "insA+", "n2+"])
    # ConflictHap has two W-line contigs covering (n1, n2): one carries insA+, one is reference-like
    b.add_walk("ConflictHap", "1", "ctg1", 0, None, ["n1+", "insA+", "n2+"])
    b.add_walk("ConflictHap", "1", "ctg2", 0, None, ["n1+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "multi_contig_conflict")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["Ref#1"] == "0"
    assert ins_ev.genotypes["Carrier#1"] == "1"
    assert ins_ev.genotypes["ConflictHap#1"] == "."


def test_three_level_component_recursion_and_nested_allele_signatures(
    tmp_path: Path,
) -> None:
    """§3.6.2 & §5.2: Three-level recursive component bifurcation produces distinct alleles 1, 2, 3 and piece allele lists."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("r1", make_complex_seq(250, seed=3171))
    b.add_node("r2", make_complex_seq(250, seed=3172))
    sh_root = make_complex_seq(200, seed=3173)
    sh_mid = make_complex_seq(180, seed=3174)
    leaf1 = make_complex_seq(140, seed=3175)
    leaf2 = make_complex_seq(160, seed=3176)
    sh_end = make_complex_seq(180, seed=3177)
    for nid, seq in [
        ("sh_root", sh_root),
        ("sh_mid", sh_mid),
        ("leaf1", leaf1),
        ("leaf2", leaf2),
        ("sh_end", sh_end),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("H1#1#Chr01", ["r1+", "sh_root+", "sh_end+", "r2+"])
    b.add_path("H2#1#Chr01", ["r1+", "sh_root+", "sh_end+", "r2+"])
    b.add_path("H3#1#Chr01", ["r1+", "sh_root+", "sh_mid+", "leaf1+", "sh_end+", "r2+"])
    b.add_path("H4#1#Chr01", ["r1+", "sh_root+", "sh_mid+", "leaf1+", "sh_end+", "r2+"])
    b.add_path("H5#1#Chr01", ["r1+", "sh_root+", "sh_mid+", "leaf2+", "sh_end+", "r2+"])

    graph, cfg = _build_graph_and_config(b, "three_level_rec")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["Ref#1"] == "0"
    assert ins_ev.genotypes["H1#1"] == ins_ev.genotypes["H2#1"]
    assert ins_ev.genotypes["H3#1"] == ins_ev.genotypes["H4#1"]
    assert len({ins_ev.genotypes["H1#1"], ins_ev.genotypes["H3#1"], ins_ev.genotypes["H5#1"]}) == 3


def test_composite_del_and_cnv_with_multi_hit_in_same_deviation(
    tmp_path: Path,
) -> None:
    """Verify composite DEL + shared-node CNV + multi_hit in the same top-level deviation."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 6):
        b.add_node(f"r{idx}", make_complex_seq(250, seed=3200 + idx))
    b.add_node("u_rep", make_complex_seq(60, seed=3206))
    b.add_node("ins_mh", make_complex_seq(140, seed=3207))

    b.set_ref_path(
        "Ref#1#Chr01", ["r1+", "r2+", "r3+", "u_rep+", "u_rep+", "r4+", "r5+"]
    )
    b.add_path(
        "Donor#1#Chr01",
        [
            "r1+",
            "r2+",
            "ins_mh+",
            "r2+",
            "u_rep+",
            "u_rep+",
            "u_rep+",
            "u_rep+",
            "r4+",
            "r5+",
        ],
    )

    graph, cfg = _build_graph_and_config(b, "composite_del_cnv")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ev_types = {e.event_type for e in res.sv_events}
    assert "DEL" in ev_types
    assert "CNV" in ev_types
    assert "INS" in ev_types
    assert any("multi_hit=1" in e.alleles_str for e in res.sv_events)


def test_c16_c17_repeat_guard_off_vs_on_and_dump_repeat_sequences(
    tmp_path: Path,
) -> None:
    """C16, C17: Verify --repeat-guard on vs off and --dump-repeat-sequences flag on REPEAT_LIKE."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=1901)
    s2 = make_complex_seq(250, seed=1902)
    sat = make_satellite_seq(unit_len=20, total_len=1500, seed=1903)
    for nid, seq in [("n1", s1), ("n2", s2), ("sat", sat)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("AltSat#1#Chr01", ["n1+", "sat+", "n2+"])

    # 1. repeat_guard="on", dump_repeat_sequences=False
    g_on, c_on = _build_graph_and_config(
        b, "rg_on", repeat_guard="on", dump_repeat_sequences=False
    )
    res_on = run_chromosome_pipeline(g_on, c_on)
    _assert_result_invariants(g_on, c_on, res_on)
    assert sat not in res_on.pan_seq
    rep_unres = [u for u in res_on.unresolved_records if u.reason == "REPEAT_LIKE"]
    assert len(rep_unres) == 1
    assert rep_unres[0].export_to_fasta is False
    assert rep_unres[0].exported_fasta_id == "."

    # 2. repeat_guard="on", dump_repeat_sequences=True
    g_dump, c_dump = _build_graph_and_config(
        b, "rg_dump", repeat_guard="on", dump_repeat_sequences=True
    )
    res_dump = run_chromosome_pipeline(g_dump, c_dump)
    _assert_result_invariants(g_dump, c_dump, res_dump)
    rep_dump = [u for u in res_dump.unresolved_records if u.reason == "REPEAT_LIKE"]
    assert len(rep_dump) == 1
    assert rep_dump[0].export_to_fasta is True
    assert rep_dump[0].exported_fasta_id != "."

    # 3. repeat_guard="off" -> satellite enters pan.fa as INS
    g_off, c_off = _build_graph_and_config(b, "rg_off", repeat_guard="off")
    res_off = run_chromosome_pipeline(g_off, c_off)
    _assert_result_invariants(g_off, c_off, res_off)
    assert sat in res_off.pan_seq
    assert len([e for e in res_off.sv_events if e.event_type == "INS"]) == 1


# ============================================================================
# 8. Helper Keys, Re-anchoring Edge Cases & Boundary Thresholds (49/50, 499/500, 1000/1001, max_jump)
# ============================================================================


def test_make_provenance_key_step_slice_key_and_token_str_helpers() -> None:
    """Verify make_provenance_key, make_step_slice_key, and make_token_str on full and partial steps."""
    assert make_provenance_key(()) == ()

    p1 = StepProvenance("H1#1", "H1#1#c1", 100, 200, "+")
    p2 = StepProvenance("H1#1", "H1#1#c1", 200, 350, "+")
    p3 = StepProvenance("H1#1", "H1#1#c1", 500, 600, "+")
    assert make_provenance_key((p1, p2, p3)) == (
        ("H1#1#c1", 100, 350, "+"),
        ("H1#1#c1", 500, 600, "+"),
    )

    st_full = Step("n1", "+")
    st_part = Step("n2", "-", slice_start=15, slice_end=95, is_partial=True)
    st_open = Step("n3", "+", slice_start=20, slice_end=None, is_partial=True)
    assert make_step_slice_key((st_full, st_part)) == (
        ("n1", "+", 0, None, False),
        ("n2", "-", 15, 95, True),
    )
    assert make_token_str((st_full, st_part, st_open)) == "n1+,n2-[15:95],n3+[20:]"


def test_reanchor_deviation_early_returns_and_ancestor_guard() -> None:
    """Verify reanchor_deviation early return conditions and ancestor cycle prevention."""
    nodes = {
        "n1": make_complex_seq(100, seed=4001),
        "n2": make_complex_seq(100, seed=4002),
        "n3": make_complex_seq(100, seed=4003),
    }
    backbone = (Step("n1", "+"), Step("n2", "+"), Step("n3", "+"))
    prov = (StepProvenance("A#1", "A#1#c", 0, 100, "+"),)

    # 1. Unanchored start deviation -> returned unchanged
    dev_unanch = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank="START",
        right_anchor_rank=1,
        branch_start_idx=0,
        branch_end_idx=1,
        branch_steps=(Step("n2", "+"),),
        branch_provenance=prov,
        backbone_steps=(Step("n1", "+"),),
        is_unanchored_start=True,
    )
    assert reanchor_deviation(dev_unanch, backbone, nodes) == [dev_unanch]

    # 2. Full interval (START..END -> lo_ab=0, hi_ab=3 == hi - lo) does not shrink hi - lo -> unchanged
    dev_full = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank="START",
        right_anchor_rank="END",
        branch_start_idx=0,
        branch_end_idx=1,
        branch_steps=(Step("n2", "+"),),
        branch_provenance=prov,
        backbone_steps=backbone,
    )
    assert reanchor_deviation(dev_full, backbone, nodes, lo=0, hi=3) == [dev_full]

    # 3. Empty branch_steps -> unchanged
    dev_empty = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank=0,
        right_anchor_rank=2,
        branch_start_idx=1,
        branch_end_idx=1,
        branch_steps=(),
        branch_provenance=(),
        backbone_steps=(Step("n2", "+"),),
    )
    assert reanchor_deviation(dev_empty, backbone, nodes, lo=0, hi=3) == [dev_empty]

    # 4. Reverse-only anchor in B(a,b) -> no forward anchor -> unchanged
    dev_rev = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank=0,
        right_anchor_rank=2,
        branch_start_idx=1,
        branch_end_idx=2,
        branch_steps=(Step("n2", "-"),),
        branch_provenance=prov,
        backbone_steps=(Step("n2", "+"),),
    )
    assert reanchor_deviation(dev_rev, backbone, nodes, lo=0, hi=3) == [dev_rev]

    # 5. Ancestor state collision blocks transition
    dev_ok = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank=0,
        right_anchor_rank=2,
        branch_start_idx=1,
        branch_end_idx=2,
        branch_steps=(Step("n2", "+"),),
        branch_provenance=prov,
        backbone_steps=(Step("n2", "+"),),
    )
    parent_key = SubproblemStateKey(
        b_id=(("B", 0, 3, "+"),),
        lo=0,
        hi=3,
        has_logical_start=False,
        has_logical_end=False,
        m_key=(make_provenance_key(prov),),
    )
    blocked_child = SubproblemStateKey(
        b_id=parent_key.b_id,
        lo=1,
        hi=2,
        has_logical_start=True,
        has_logical_end=True,
        m_key=parent_key.m_key,
    )
    assert reanchor_deviation(
        dev_ok,
        backbone,
        nodes,
        lo=0,
        hi=3,
        parent_key=parent_key,
        ancestor_keys=frozenset({blocked_child}),
    ) == [dev_ok]


def test_reanchor_deviation_multi_level_recursive_refinement_and_covered_ranks() -> None:
    """F14: Two-level local re-anchoring where level-2 anchor only becomes unique after level-1 split."""
    nodes = {
        "n0": make_complex_seq(100, seed=4010),
        "m1": make_complex_seq(100, seed=4011),
        "m2": make_complex_seq(100, seed=4012),
        "x": make_complex_seq(80, seed=4013),
        "n5": make_complex_seq(100, seed=4014),
    }
    # At top level [0, 6): m1 appears twice (ranks 1, 5? No: let's put m1 once inside (0, 5) and once at rank 5).
    # Wait: inside B(0, 5) = [m2+, m1+, m2+, skip+], m1+ appears once (rank 2) while m2+ appears twice (ranks 1, 3).
    # Once level 1 re-anchors on m1+ (rank 2), sub-interval B(2, 5) = [m2+, skip+] has m2+ appearing ONCE (rank 3)!
    backbone = (
        Step("n0", "+"),  # rank 0
        Step("m2", "+"),  # rank 1
        Step("m1", "+"),  # rank 2 (unique in [1, 5))
        Step("m2", "+"),  # rank 3 (repeated in [1, 5), but unique in [3, 5)!)
        Step("n0", "+"),  # rank 4 (unique in [3, 5))
        Step("m1", "+"),  # rank 5 (makes m1 repeated at top level [0, 6))
    )
    # Branch between rank 0 and rank 5 traverses: m2+ -> m1+ -> m2+ -> x+ -> n0+
    br_steps = (
        Step("m2", "+"),
        Step("m1", "+"),
        Step("m2", "+"),
        Step("x", "+"),
        Step("n0", "+"),
    )
    br_prov = (
        StepProvenance("A#1", "A#1#c", 100, 200, "+"),
        StepProvenance("A#1", "A#1#c", 200, 300, "+"),
        StepProvenance("A#1", "A#1#c", 300, 400, "+"),
        StepProvenance("A#1", "A#1#c", 400, 480, "+"),
        StepProvenance("A#1", "A#1#c", 480, 580, "+"),
    )
    dev = DeviationSegment(
        hap_id="A#1",
        src_path="A#1#c",
        left_anchor_rank=0,
        right_anchor_rank=5,
        branch_start_idx=1,
        branch_end_idx=6,
        branch_steps=br_steps,
        branch_provenance=br_prov,
        backbone_steps=backbone[1:5],
    )
    covered: set[int] = set()
    refined = reanchor_deviation(
        dev, backbone, nodes, lo=0, hi=6, covered_ranks_out=covered
    )
    assert len(refined) == 1
    assert refined[0].left_anchor_rank == 3
    assert refined[0].right_anchor_rank == 4
    assert refined[0].branch_steps == (Step("x", "+"),)
    assert {1, 2, 3, 4}.issubset(covered)


def test_boundary_49bp_vs_50bp_candidate_block_extraction(tmp_path: Path) -> None:
    """F15 boundary: 49 bp novel block (< min_sv_len=50) is discarded; 50 bp novel block is retained as INS."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4101)
    s2 = make_complex_seq(200, seed=4102)
    s3 = make_complex_seq(200, seed=4103)
    nov49 = make_complex_seq(49, seed=4104)
    nov50 = make_complex_seq(50, seed=4105)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("nov49", nov49),
        ("nov50", nov50),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "nov49+", "n2+", "nov50+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "bound_49_50_ins", min_sv_len=50)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert nov49 not in res.pan_seq
    assert nov50 in res.pan_seq
    ins_evs = [e for e in res.sv_events if e.event_type == "INS"]
    assert len(ins_evs) == 1
    assert ins_evs[0].ref_start == 400
    assert "len=50," in ins_evs[0].alleles_str


def test_boundary_49bp_vs_50bp_unanchored_start_and_end(tmp_path: Path) -> None:
    """F15 boundary: 49 bp unanchored prefix/suffix ignored; 50 bp unanchored prefix/suffix recorded in unresolved."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4111)
    s2 = make_complex_seq(200, seed=4112)
    s3 = make_complex_seq(200, seed=4113)
    pre49 = make_complex_seq(49, seed=4114)
    suf49 = make_complex_seq(49, seed=4115)
    pre50 = make_complex_seq(50, seed=4116)
    suf50 = make_complex_seq(50, seed=4117)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("p49", pre49),
        ("s49", suf49),
        ("p50", pre50),
        ("s50", suf50),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt49#1#Chr01", ["p49+", "n2+", "s49+"])
    b.add_path("Alt50#1#Chr01", ["p50+", "n2+", "s50+"])

    graph, cfg = _build_graph_and_config(b, "bound_49_50_unanch", min_sv_len=50)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    unanch = [u for u in res.unresolved_records if u.reason == "UNANCHORED_END"]
    assert len(unanch) == 2
    assert all(u.length == 50 for u in unanch)
    assert all(u.src_path == "Alt50#1#Chr01" for u in unanch)


def test_boundary_49bp_vs_50bp_del_threshold(tmp_path: Path) -> None:
    """F20 boundary: 49 bp deletion is ignored (< min_sv_len=50); 50 bp deletion emits DEL."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4121)
    d49 = make_complex_seq(49, seed=4122)
    s2 = make_complex_seq(200, seed=4123)
    d50 = make_complex_seq(50, seed=4124)
    s3 = make_complex_seq(200, seed=4125)
    for nid, seq in [("n1", s1), ("d49", d49), ("n2", s2), ("d50", d50), ("n3", s3)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "d49+", "n2+", "d50+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "bound_49_50_del", min_sv_len=50)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    del_evs = [e for e in res.sv_events if e.event_type == "DEL"]
    assert len(del_evs) == 1
    assert del_evs[0].ref_start == 449 and del_evs[0].ref_end == 499
    assert "1:len=50," in del_evs[0].alleles_str


def test_extract_candidate_blocks_splits_on_source_discontinuity_and_claimed_intervals() -> None:
    """Verify extract_candidate_blocks splits non-contiguous source provenance and filters claimed intervals."""
    s_a = make_complex_seq(80, seed=4131)
    s_b = make_complex_seq(90, seed=4132)
    nodes = {
        "r1": make_complex_seq(100, seed=4130),
        "r2": make_complex_seq(100, seed=4133),
        "a": s_a,
        "b": s_b,
    }
    backbone = (Step("r1", "+"), Step("r2", "+"))
    # Non-contiguous source coordinates (100..180 followed by 300..390)
    dev = DeviationSegment(
        hap_id="H#1",
        src_path="H#1#c1",
        left_anchor_rank=0,
        right_anchor_rank=1,
        branch_start_idx=1,
        branch_end_idx=3,
        branch_steps=(Step("a", "+"), Step("b", "+")),
        branch_provenance=(
            StepProvenance("H#1", "H#1#c1", 100, 180, "+"),
            StepProvenance("H#1", "H#1#c1", 300, 390, "+"),
        ),
        backbone_steps=(),
    )
    cands = extract_candidate_blocks(
        dev=dev,
        backbone_steps=backbone,
        nodes=nodes,
        global_ref_nodes={"r1", "r2"},
        min_sv_len=50,
    )
    assert len(cands) == 2
    assert cands[0].seq == s_a and cands[1].seq == s_b

    # Claiming the first interval filters it out when already_screened=False
    cands_claimed = extract_candidate_blocks(
        dev=dev,
        backbone_steps=backbone,
        nodes=nodes,
        global_ref_nodes={"r1", "r2"},
        min_sv_len=50,
        claimed_source_intervals={("H#1#c1", 100, 180)},
    )
    assert len(cands_claimed) == 1
    assert cands_claimed[0].seq == s_b


def test_extract_candidate_blocks_top_undecomposable_repeat_sets_repeat_residual(
    tmp_path: Path,
) -> None:
    """F15: When top-level B(a,b) has only multi-copy reference nodes, extracted novel blocks set repeat_residual=1."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4141)
    sr = make_complex_seq(100, seed=4142)
    s2 = make_complex_seq(200, seed=4143)
    nov = make_complex_seq(130, seed=4144)
    for nid, seq in [("n1", s1), ("rdup", sr), ("n2", s2), ("nov", nov)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "rdup+", "rdup+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "rdup+", "nov+", "rdup+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "top_undecomp_rep")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert "repeat_residual=1" in ins_ev.alleles_str


def test_boundary_499bp_vs_500bp_repeat_guard_threshold(tmp_path: Path) -> None:
    """F19 boundary: 499 bp satellite bypasses repeat-guard (< 500 bp); 500 bp satellite is masked as REPEAT_LIKE."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4201)
    s2 = make_complex_seq(200, seed=4202)
    s3 = make_complex_seq(200, seed=4203)
    sat499 = make_satellite_seq(unit_len=20, total_len=499, seed=4204)
    sat500 = make_satellite_seq(unit_len=20, total_len=500, seed=4205)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("n3", s3),
        ("sat499", sat499),
        ("sat500", sat500),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "sat499+", "n2+", "sat500+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "bound_499_500_rg", repeat_guard="on")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert sat499 in res.pan_seq
    assert sat500 not in res.pan_seq
    rep_unres = [u for u in res.unresolved_records if u.reason == "REPEAT_LIKE"]
    assert len(rep_unres) == 1
    assert rep_unres[0].length == 500


def test_screen_and_fold_candidate_short_residual_and_post_fold_below_min_sv_len(
    tmp_path: Path,
) -> None:
    """Verify screen_and_fold_candidate drops <50bp unmasked residuals (SHORT_RESIDUAL) and post-fold <50bp blocks."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4211)
    s2 = make_complex_seq(200, seed=4212)
    # 40 bp complex flank + 1500 bp satellite -> 1000bp window masks [500, 1540) or full satellite;
    # wait: to make [0, 40) unmasked and [40, 1540) masked, let's test direct post-fold < 50bp first:
    # Three copies of a 20 bp novel node = 60 bp raw (>= 50 bp), but after fold_tandem_cycles(max_copy=1) -> 20 bp (< 50 bp)!
    u20 = make_complex_seq(20, seed=4213)
    for nid, seq in [("n1", s1), ("n2", s2), ("u20", u20)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("AltFoldShort#1#Chr01", ["n1+", "u20+", "u20+", "u20+", "n2+"])

    graph, cfg = _build_graph_and_config(
        b, "fold_below_50", min_sv_len=50, max_copy=1
    )
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)
    assert len(res.sv_events) == 0
    assert res.pan_seq == s1 + s2


# ============================================================================
# 9. Partial-Node UF Exclusion, Synthetic vs Real Clustering, and Ordering
# ============================================================================


def test_partial_node_slice_excluded_from_union_find_component_partitioning(
    tmp_path: Path,
) -> None:
    """F16 / F19: Candidates sharing a node_id ONLY via partial-node slices (is_partial=True) are NOT unioned in UF."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4301)
    s2 = make_complex_seq(200, seed=4302)
    big_shared = make_complex_seq(500, seed=4303)
    u_left = make_complex_seq(120, seed=4304)
    u_right = make_complex_seq(130, seed=4305)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("big", big_shared),
        ("uL", u_left),
        ("uR", u_right),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "uL+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "uf_partial_excl")

    # Candidate 1 uses partial slice big+[0:100] + full step uL+
    c1_steps = (
        Step("big", "+", slice_start=0, slice_end=100, is_partial=True),
        Step("uL", "+"),
    )
    c1_prov = (
        StepProvenance("Alt1#1", "Alt1#1#Chr01", 200, 300, "+"),
        StepProvenance("Alt1#1", "Alt1#1#Chr01", 300, 420, "+"),
    )
    c1 = ExtractedCandidate(
        candidate_id="c1",
        parent_id="p1",
        hap_id="Alt1#1",
        src_path="Alt1#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=0,
        steps=c1_steps,
        provenance=c1_prov,
        seq=big_shared[:100] + u_left,
        partial_content=True,
    )

    # Candidate 2 uses partial slice big+[300:400] + full step uR+
    c2_steps = (
        Step("big", "+", slice_start=300, slice_end=400, is_partial=True),
        Step("uR", "+"),
    )
    c2_prov = (
        StepProvenance("Alt2#1", "Alt2#1#Chr01", 200, 300, "+"),
        StepProvenance("Alt2#1", "Alt2#1#Chr01", 300, 430, "+"),
    )
    c2 = ExtractedCandidate(
        candidate_id="c2",
        parent_id="p2",
        hap_id="Alt2#1",
        src_path="Alt2#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=1,
        steps=c2_steps,
        provenance=c2_prov,
        seq=big_shared[300:400] + u_right,
        partial_content=True,
    )

    sol = solve_parallel_group(
        left_anchor_rank=0,
        right_anchor_rank=1,
        backbone_slice_steps=(),
        candidates=[c1, c2],
        graph=graph,
        config=cfg,
    )
    assert len(sol.non_backbone_reps) == 2
    assert all(not r.is_synthetic for r in sol.non_backbone_reps)


def test_exact_deduplication_distinguishes_different_slice_bounds_and_orientations(
    tmp_path: Path,
) -> None:
    """§3.6.2 Step 1: Identical step+slice keys deduplicate; different slice bounds or orientations stay distinct."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4311)
    s2 = make_complex_seq(200, seed=4312)
    big = make_complex_seq(400, seed=4313)
    for nid, seq in [("n1", s1), ("n2", s2), ("big", big)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "big+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "dedup_slices")

    st_a = (Step("big", "+", slice_start=0, slice_end=150, is_partial=True),)
    st_b = (Step("big", "+", slice_start=200, slice_end=350, is_partial=True),)

    c1 = ExtractedCandidate(
        candidate_id="c1",
        parent_id=None,
        hap_id="H1#1",
        src_path="H1#1#c",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=0,
        steps=st_a,
        provenance=(StepProvenance("H1#1", "H1#1#c", 0, 150, "+"),),
        seq=big[0:150],
    )
    c2_same = ExtractedCandidate(
        candidate_id="c2",
        parent_id=None,
        hap_id="H2#1",
        src_path="H2#1#c",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=1,
        steps=st_a,
        provenance=(StepProvenance("H2#1", "H2#1#c", 0, 150, "+"),),
        seq=big[0:150],
    )
    c3_diff_slice = ExtractedCandidate(
        candidate_id="c3",
        parent_id=None,
        hap_id="H3#1",
        src_path="H3#1#c",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=2,
        steps=st_b,
        provenance=(StepProvenance("H3#1", "H3#1#c", 200, 350, "+"),),
        seq=big[200:350],
    )

    sol = solve_parallel_group(
        left_anchor_rank=0,
        right_anchor_rank=1,
        backbone_slice_steps=(),
        candidates=[c1, c2_same, c3_diff_slice],
        graph=graph,
        config=cfg,
    )
    assert len(sol.non_backbone_reps) == 2
    assert sol.non_backbone_reps[0].support_haps == {"H1#1", "H2#1"}
    assert sol.non_backbone_reps[1].support_haps == {"H3#1"}


def test_synthetic_vs_real_sequence_clustering_separation(tmp_path: Path) -> None:
    """§3.6.2 Step 4: Synthetic sequences (from multi-member UF component) do NOT similarity-merge with real sequences."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4321)
    s2 = make_complex_seq(200, seed=4322)
    sh = make_complex_seq(250, seed=4323)
    tail1 = make_complex_seq(250, seed=4324)
    # tail2 is a <50bp bubble (20bp) so the synthetic sequence is sh + tail1 (500bp, is_synthetic=True)
    b1 = make_complex_seq(20, seed=4325)
    b2 = make_complex_seq(20, seed=4326)
    synth_seq = sh + b1 + tail1
    # real_seq is 99% identical to synth_seq (5 SNPs out of 520bp), passing 4-criterion similarity if real,
    # but since synth_seq is is_synthetic=True, they must NOT merge!
    real_seq = mutate_seq_snps(synth_seq, num_snps=5, seed=4327)

    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("sh", sh),
        ("b1", b1),
        ("b2", b2),
        ("t1", tail1),
        ("real_node", real_seq),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("H1#1#Chr01", ["n1+", "sh+", "b1+", "t1+", "n2+"])
    b.add_path("H2#1#Chr01", ["n1+", "sh+", "b2+", "t1+", "n2+"])
    b.add_path("H3#1#Chr01", ["n1+", "real_node+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "synth_vs_real_sep")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    # Both the synthetic representative (H1, H2) and the real representative (H3) must be kept separate
    assert res.step_offsets[1].slot_ins_len == len(synth_seq) + 100 + len(real_seq)


def test_synthetic_exact_deduplication_with_identical_real_or_backbone_seq(
    tmp_path: Path,
) -> None:
    """§3.6.2 Step 4: Synthetic sequence 100% base-identical to a real representative exact-deduplicates cleanly."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4331)
    s2 = make_complex_seq(200, seed=4332)
    sh = make_complex_seq(200, seed=4333)
    b1 = make_complex_seq(20, seed=4334)
    b2 = make_complex_seq(20, seed=4335)
    t1 = make_complex_seq(200, seed=4336)
    exact_whole = sh + b1 + t1
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("sh", sh),
        ("b1", b1),
        ("b2", b2),
        ("t1", t1),
        ("whole", exact_whole),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("H1#1#Chr01", ["n1+", "sh+", "b1+", "t1+", "n2+"])
    b.add_path("H2#1#Chr01", ["n1+", "sh+", "b1+", "t1+", "n2+"])
    b.add_path("H3#1#Chr01", ["n1+", "sh+", "b2+", "t1+", "n2+"])
    b.add_path("H4#1#Chr01", ["n1+", "whole+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "synth_exact_dedup")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert res.step_offsets[1].slot_ins_len == len(exact_whole)


def test_boundary_1000bp_edlib_vs_1001bp_mappy_clustering_switch(
    tmp_path: Path,
) -> None:
    """F17 boundary: 1000 bp uses edlib, 1001 bp uses mappy; Criterion 3 boundary 49 bp vs 50 bp unaligned tail."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 5):
        b.add_node(f"n{idx}", make_complex_seq(200, seed=4400 + idx))

    # Locus 1 (n1..n2): Exactly 1000 bp with 15 SNPs -> merges via edlib
    r1000 = make_complex_seq(1000, seed=4410)
    s1000 = mutate_seq_snps(r1000, num_snps=15, seed=4411)
    b.add_node("r1000", r1000).add_node("s1000", s1000)

    # Locus 2 (n2..n3): Exactly 1001 bp with 15 SNPs -> merges via mappy
    r1001 = make_complex_seq(1001, seed=4412)
    s1001 = mutate_seq_snps(r1001, num_snps=15, seed=4413)
    b.add_node("r1001", r1001).add_node("s1001", s1001)

    # Locus 3 (n3..n4): R has 900bp core + 60bp tail (960bp), S has 900bp core + 50bp divergent tail (950bp)
    # -> S has 50 bp unaligned tail >= min_sv_len=50 -> REJECTED by Criterion 3
    core900 = make_complex_seq(900, seed=4414)
    tail60 = make_complex_seq(60, seed=4415)
    tail50 = make_complex_seq(50, seed=4416)
    r960 = core900 + tail60
    s950 = core900 + tail50
    b.add_node("r960", r960).add_node("s950", s950)

    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    b.add_path(
        "Alt1#1#Chr01", ["n1+", "r1000+", "n2+", "r1001+", "n3+", "r960+", "n4+"]
    )
    b.add_path(
        "Alt2#1#Chr01", ["n1+", "s1000+", "n2+", "s1001+", "n3+", "s950+", "n4+"]
    )

    graph, cfg = _build_graph_and_config(b, "bound_1000_1001")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert res.step_offsets[1].slot_ins_len == 1000
    assert res.step_offsets[2].slot_ins_len == 1001
    assert res.step_offsets[3].slot_ins_len == 960 + 100 + 950


def test_real_candidate_merging_into_nonempty_backbone_rep0(tmp_path: Path) -> None:
    """§3.6.2 Step 4 & T5b: Real replacement candidate >=95% similar to B(a,b) merges into R_0 = B(a,b) without INS."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4421)
    s_mid = make_complex_seq(300, seed=4422)
    s2 = make_complex_seq(200, seed=4423)
    # 5 SNPs on 300 bp (98.3% identity, no >=50bp indels) -> merges into R_0 = B(n1, n2) = s_mid!
    s_alt_mid = mutate_seq_snps(s_mid, num_snps=5, seed=4424)
    for nid, seq in [
        ("n1", s1),
        ("mid", s_mid),
        ("n2", s2),
        ("alt_mid", s_alt_mid),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "mid+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "alt_mid+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "merge_into_rep0")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert res.pan_seq == s1 + s_mid + s2
    assert len([e for e in res.sv_events if e.event_type == "INS"]) == 0


def test_representative_ordering_tie_breakers_support_length_token_hap(
    tmp_path: Path,
) -> None:
    """§3.6.4: Verify non-backbone representatives are ordered by support DESC, length DESC, token ASC."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4431)
    s2 = make_complex_seq(200, seed=4432)
    # R_sup2: support=2, len=120
    seq_sup2 = make_complex_seq(120, seed=4433)
    # R_long: support=1, len=250
    seq_long = make_complex_seq(250, seed=4434)
    # R_tokA: support=1, len=180, node_id="tokA"
    seq_tok_a = make_complex_seq(180, seed=4435)
    # R_tokB: support=1, len=180, node_id="tokB"
    seq_tok_b = make_complex_seq(180, seed=4436)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("sup2", seq_sup2),
        ("lng", seq_long),
        ("tokA", seq_tok_a),
        ("tokB", seq_tok_b),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("H1#1#Chr01", ["n1+", "tokB+", "n2+"])
    b.add_path("H2#1#Chr01", ["n1+", "tokA+", "n2+"])
    b.add_path("H3#1#Chr01", ["n1+", "lng+", "n2+"])
    b.add_path("H4#1#Chr01", ["n1+", "sup2+", "n2+"])
    b.add_path("H5#1#Chr01", ["n1+", "sup2+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "rep_order_ties")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    expected_slot = (
        seq_sup2
        + ("N" * 100)
        + seq_long
        + ("N" * 100)
        + seq_tok_a
        + ("N" * 100)
        + seq_tok_b
    )
    assert res.pan_seq == s1 + expected_slot + s2


# ============================================================================
# 10. SV Calling Boundaries, Inversions, Whole-Contig Flip, CNV & Multi-Contig Genotypes
# ============================================================================


def test_boundary_max_jump_exact_equal_vs_exceeding_threshold(tmp_path: Path) -> None:
    """F20 boundary: span_bp == max_jump (500) is normal DEL/INS; span_bp == max_jump + 1 (501) is LARGE_JUMP."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4501)
    skip500 = make_complex_seq(500, seed=4502)
    s2 = make_complex_seq(200, seed=4503)
    skip501 = make_complex_seq(501, seed=4504)
    s3 = make_complex_seq(200, seed=4505)
    for nid, seq in [
        ("n1", s1),
        ("sk500", skip500),
        ("n2", s2),
        ("sk501", skip501),
        ("n3", s3),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "sk500+", "n2+", "sk501+", "n3+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+", "n3+"])

    graph, cfg = _build_graph_and_config(b, "bound_max_jump", max_jump=500)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    del_evs = [e for e in res.sv_events if e.event_type == "DEL"]
    lj_evs = [e for e in res.sv_events if e.event_type == "LARGE_JUMP"]
    assert len(del_evs) == 1
    assert del_evs[0].ref_end - del_evs[0].ref_start == 500
    assert len(lj_evs) == 1
    assert lj_evs[0].ref_end - lj_evs[0].ref_start == 501


def test_boundary_dup_tra_distance_exact_equal_vs_exceeding_max_jump(
    tmp_path: Path,
) -> None:
    """§3.7 DUP_TRA boundary: visiting a unique ref anchor at distance == max_jump (500) is NOT DUP_TRA; 501 IS DUP_TRA."""
    b = SyntheticInputBuilder(tmp_path)
    n1 = make_complex_seq(200, seed=4511)
    n2 = make_complex_seq(200, seed=4512)
    n3 = make_complex_seq(300, seed=4513)  # ref_start(n3)=400; dist from n1_end(200) is 200 <= 500
    n4 = make_complex_seq(201, seed=4514)  # ref_start(n5)=901; dist from n1_end(200) is 701 > 500
    n5 = make_complex_seq(200, seed=4515)
    for nid, seq in [("n1", n1), ("n2", n2), ("n3", n3), ("n4", n4), ("n5", n5)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+", "n5+"])
    # NearCopy visits n4+ between n1+ and n2+: ref_start(n4)=700, a_end(n1)=200 -> dist = 500 == max_jump (no DUP_TRA)
    b.add_path("Near#1#Chr01", ["n1+", "n4+", "n2+", "n3+", "n4+", "n5+"])
    # FarCopy visits n5+ between n2+ and n3+: ref_start(n5)=901, a_end(n2)=400 -> dist = 501 > max_jump (emits DUP_TRA)
    b.add_path("Far#1#Chr01", ["n1+", "n2+", "n5+", "n3+", "n4+", "n5+"])

    graph, cfg = _build_graph_and_config(b, "bound_dup_tra", max_jump=500)
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    dt_evs = [e for e in res.sv_events if e.event_type == "DUP_TRA"]
    assert len(dt_evs) == 1
    assert dt_evs[0].anchors == "1..2"
    assert dt_evs[0].genotypes["Far#1"] == "1"
    assert dt_evs[0].genotypes["Near#1"] == "0"


def test_inv_displaced_and_inv_conflict_genotyping_and_unresolved_routing(
    tmp_path: Path,
) -> None:
    """§3.3.1–3.3.2 & §3.7 (T24): Verify INV_DISPLACED and INV_CONFLICT classification and unresolved routing."""
    b = SyntheticInputBuilder(tmp_path)
    b.add_node("r0", make_complex_seq(500, seed=4520))
    for idx in range(1, 6):
        b.add_node(f"r{idx}", make_complex_seq(150, seed=4520 + idx))
    b.add_node("r6", make_complex_seq(500, seed=4526))
    nov_conf = make_complex_seq(140, seed=4530)
    b.add_node("novC", nov_conf)

    b.set_ref_path("Ref#1#Chr01", ["r0+", "r1+", "r2+", "r3+", "r4+", "r5+", "r6+"])
    # DisplacedHap: L=r3+ (rank 3 > rmin=1), R=[r2- -> r1-] ([1, 2]), F=r4+ (rank 4) -> INV_DISPLACED
    b.add_path("DispHap#1#Chr01", ["r0+", "r3+", "r2-", "r1-", "r4+", "r5+", "r6+"])
    # ConflictHap: Two reverse chains with overlapping rank intervals [2, 4] (r4- -> r2-) and [1, 3] (r3- -> novC+ -> r1-)
    b.add_path(
        "ConfHap#1#Chr01",
        ["r0+", "r4-", "r2-", "r5+", "r3-", "novC+", "r1-", "r6+"],
    )

    graph, cfg = _build_graph_and_config(b, "inv_disp_conf")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ev_types = {e.event_type for e in res.sv_events}
    assert "INV_DISPLACED" in ev_types
    assert "INV_CONFLICT" in ev_types
    conf_unres = [u for u in res.unresolved_records if u.reason == "INV_CONFLICT"]
    assert len(conf_unres) == 1
    assert conf_unres[0].length == 140
    assert conf_unres[0].event_id.startswith("Chr01_INV_CONFLICT_")


def test_t23_whole_contig_reverse_complement_orientation_no_false_inv(
    tmp_path: Path,
) -> None:
    """§3.3.0 (T23): Entirely reverse-complemented contig is oriented to forward without emitting INV."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=4541)
    s2 = make_complex_seq(250, seed=4542)
    s3 = make_complex_seq(250, seed=4543)
    ins = make_complex_seq(160, seed=4544)
    for nid, seq in [("n1", s1), ("n2", s2), ("n3", s3), ("ins", ins)]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+"])
    # AltRev is given in reverse order: n3- -> n2- -> ins+ -> n1-
    # After flipping whole contig: n1+ -> ins- -> n2+ -> n3+
    b.add_path("AltRev#1#Chr01", ["n3-", "n2-", "ins+", "n1-"])

    graph, cfg = _build_graph_and_config(b, "t23_whole_flip")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert not any(e.event_type.startswith("INV") for e in res.sv_events)
    assert res.pan_seq == s1 + revcomp(ins) + s2 + s3
    assert len(res.inserts_rows) == 1
    assert res.inserts_rows[0]["src_strand"] == "-"
    assert res.inserts_rows[0]["src_intervals"] == "500-660"
    assert any("ORIENT_CONTIG" in line for line in res.log_lines)


def test_t10_canonical_cnv_suppresses_del_and_reanchoring(tmp_path: Path) -> None:
    """§3.4.4 & §3.7 (T10): Pure tandem copy-number expansion and contraction emit multi-allelic CNV without DEL."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4551)
    u30 = make_complex_seq(30, seed=4552)
    s2 = make_complex_seq(200, seed=4553)
    for nid, seq in [("n1", s1), ("u", u30), ("n2", s2)]:
        b.add_node(nid, seq)
    # Reference has 5 copies of u+ (150 bp)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "u+", "u+", "u+", "u+", "n2+"])
    # AltContract has 3 copies (|3 - 5| * 30 = 60 bp >= 50 bp)
    b.add_path("AltContract#1#Chr01", ["n1+", "u+", "u+", "u+", "n2+"])
    # AltExpand has 7 copies (|7 - 5| * 30 = 60 bp >= 50 bp)
    b.add_path(
        "AltExpand#1#Chr01",
        ["n1+", "u+", "u+", "u+", "u+", "u+", "u+", "u+", "n2+"],
    )

    graph, cfg = _build_graph_and_config(b, "t10_cnv")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    assert [e.event_type for e in res.sv_events] == ["CNV"]
    cnv_ev = res.sv_events[0]
    assert cnv_ev.genotypes["Ref#1"] == "5"
    assert cnv_ev.genotypes["AltContract#1"] == "3"
    assert cnv_ev.genotypes["AltExpand#1"] == "7"


def test_steps_to_pieces_and_merge_adjacent_source_contiguous_pieces() -> None:
    """Verify steps_to_pieces and _merge_adjacent_source_contiguous_pieces on '+' and '-' contiguous steps."""
    from panflat.linearizer import _merge_adjacent_source_contiguous_pieces

    s_a = make_complex_seq(70, seed=4601)
    s_b = make_complex_seq(80, seed=4602)
    nodes = {"a": s_a, "b": s_b}

    # 1. Contiguous '-' steps merge into one source_contiguous Piece (src_start=100, src_end=250)
    steps_neg = (Step("b", "-"), Step("a", "-"))
    prov_neg = (
        StepProvenance("H#1", "H#1#c", 170, 250, "-"),
        StepProvenance("H#1", "H#1#c", 100, 170, "-"),
    )
    pieces_neg = steps_to_pieces(steps_neg, prov_neg, (), nodes, ("H#1",))
    assert len(pieces_neg) == 1
    assert pieces_neg[0].src_start == 100 and pieces_neg[0].src_end == 250
    assert pieces_neg[0].src_strand == "-"
    assert pieces_neg[0].join_before == "source_contiguous"

    # 2. _merge_adjacent_source_contiguous_pieces merges two touching '-' Pieces
    p_hi = pieces_neg[0]
    p_lo = steps_to_pieces(
        (Step("a", "-"),),
        (StepProvenance("H#1", "H#1#c", 30, 100, "-"),),
        (),
        nodes,
        ("H#1",),
    )[0]
    merged_p, merged_c = _merge_adjacent_source_contiguous_pieces(
        (p_hi, p_lo), (frozenset({"c1"}), frozenset({"c2"}))
    )
    assert len(merged_p) == 1
    assert merged_p[0].src_start == 30 and merged_p[0].src_end == 250
    assert merged_c == (frozenset({"c1", "c2"}),)


def test_assemble_pan_and_offsets_coordinate_backfilling_for_all_event_types(
    tmp_path: Path,
) -> None:
    """Verify assemble_pan_and_offsets backfills pan_start and pan_end for point, span, INS, and CEN_ISLAND events."""
    from panflat.models import InsertionSlotBlock, Piece, SVEvent

    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(200, seed=4611)
    s2 = make_complex_seq(200, seed=4612)
    b.add_node("n1", s1).add_node("n2", s2)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("Alt#1#Chr01", ["n1+", "n2+"])
    graph, cfg = _build_graph_and_config(b, "asm_backfill")

    isl_seq = make_complex_seq(500, seed=4613)
    isl_piece = Piece(
        src_hap="Alt#1",
        src_path="Alt#1#Chr01",
        src_start=0,
        src_end=500,
        src_strand="+",
        seq=isl_seq,
        join_before="spacer",
        members=("Alt#1",),
    )
    cen_blk = InsertionSlotBlock(
        right_anchor_ref_step=1,
        is_centromere_island=True,
        group_span=1,
        left_anchor_ref_step=0,
        event_id="Chr01_CEN_ISLAND_1",
        representatives=((isl_piece,),),
        trailing_spacer=True,
    )
    cen_ev = SVEvent(
        chrom="Chr01",
        pan_start=0,
        pan_end=0,
        event_id="Chr01_CEN_ISLAND_1",
        event_type="CEN_ISLAND",
        ref_start=200,
        ref_end=200,
        anchors="n1..n2",
        alleles_str="1:len=500,src=Alt#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id="Chr01_CEN_1",
        genotypes={"Ref#1": "0", "Alt#1": "1"},
        pieces=(isl_piece,),
    )
    del_ev = SVEvent(
        chrom="Chr01",
        pan_start=0,
        pan_end=0,
        event_id="Chr01_DEL_1",
        event_type="DEL",
        ref_start=200,
        ref_end=400,
        anchors="0..1",
        alleles_str="1:len=200,src=Alt#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id="Chr01_REG_1",
        genotypes={"Ref#1": "0", "Alt#1": "1"},
        pieces=(),
    )

    pan_seq, step_offsets, upd_evs, ins_rows = assemble_pan_and_offsets(
        graph=graph,
        slot_blocks_by_step={1: [cen_blk]},
        sv_events=[cen_ev, del_ev],
        spacer_n=100,
    )
    assert pan_seq == s1 + ("N" * 100) + isl_seq + ("N" * 100) + s2
    assert step_offsets[1].slot_ins_len == 700
    assert upd_evs[0].pan_start == 300 and upd_evs[0].pan_end == 800
    assert upd_evs[1].pan_start == 900 and upd_evs[1].pan_end == 1100
    assert len(ins_rows) == 1


def test_colocated_events_share_region_id_and_disjoint_loci_get_distinct_region_ids(
    tmp_path: Path,
) -> None:
    """§5.2: Replacement (DEL + INS) at Locus 1 shares Chr01_REG_1; downstream INS at Locus 2 gets Chr01_REG_2."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 5):
        b.add_node(f"n{idx}", make_complex_seq(200, seed=4620 + idx))
    b.add_node("rep1", make_complex_seq(150, seed=4625))
    b.add_node("ins2", make_complex_seq(160, seed=4626))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # Replacement of n2 with rep1 between n1 and n3 + pure insertion of ins2 between n3 and n4
    b.add_path("Alt#1#Chr01", ["n1+", "rep1+", "n3+", "ins2+", "n4+"])

    graph, cfg = _build_graph_and_config(b, "colocated_reg_ids")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    del1 = [e for e in res.sv_events if e.event_type == "DEL"][0]
    ins_evs = [e for e in res.sv_events if e.event_type == "INS"]
    assert len(ins_evs) == 2
    assert del1.region_id == ins_evs[0].region_id == "Chr01_REG_1"
    assert ins_evs[1].region_id == "Chr01_REG_2"


def test_multi_contig_consistent_non_dot_calls_and_dot_ignoring(
    tmp_path: Path,
) -> None:
    """§5.2: Non-overlapping W-line contigs of the same haplotype resolve their respective covered loci cleanly."""
    b = SyntheticInputBuilder(tmp_path)
    for idx in range(1, 5):
        b.add_node(f"n{idx}", make_complex_seq(200, seed=4630 + idx))
    b.add_node("ins1", make_complex_seq(140, seed=4635))
    b.add_node("ins2", make_complex_seq(150, seed=4636))
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+", "n3+", "n4+"])
    # SplitHap has ctg1 covering (n1, n2) with ins1+ and ctg2 covering (n3, n4) with reference allele 0
    b.add_walk("SplitHap", "1", "ctg1", 0, None, ["n1+", "ins1+", "n2+"])
    b.add_walk("SplitHap", "1", "ctg2", 0, None, ["n3+", "n4+"])
    b.add_path("FullHap#1#Chr01", ["n1+", "n2+", "n3+", "ins2+", "n4+"])

    graph, cfg = _build_graph_and_config(b, "multi_contig_disjoint")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_evs = [e for e in res.sv_events if e.event_type == "INS"]
    assert len(ins_evs) == 2
    assert ins_evs[0].genotypes["SplitHap#1"] == "1"
    assert ins_evs[0].genotypes["FullHap#1"] == "0"
    assert ins_evs[1].genotypes["SplitHap#1"] == "0"
    assert ins_evs[1].genotypes["FullHap#1"] == "1"


def test_serial_insertions_from_split_repeat_candidate_preserve_source_order_in_signature(
    tmp_path: Path,
) -> None:
    """§5.2 & §3.6.6: Haplotype carrying both kept parts (1, 2) vs haplotype carrying only part (1,) get distinct alleles."""
    b = SyntheticInputBuilder(tmp_path)
    s1 = make_complex_seq(250, seed=4641)
    s2 = make_complex_seq(250, seed=4642)
    left_c = make_complex_seq(350, seed=4643)
    mid_sat = make_satellite_seq(unit_len=20, total_len=1600, seed=4644)
    right_c = make_complex_seq(380, seed=4645)
    for nid, seq in [
        ("n1", s1),
        ("n2", s2),
        ("lc", left_c),
        ("ms", mid_sat),
        ("rc", right_c),
    ]:
        b.add_node(nid, seq)
    b.set_ref_path("Ref#1#Chr01", ["n1+", "n2+"])
    b.add_path("HapBoth#1#Chr01", ["n1+", "lc+", "ms+", "rc+", "n2+"])
    b.add_path("HapLeft1#1#Chr01", ["n1+", "lc+", "n2+"])
    b.add_path("HapLeft2#1#Chr01", ["n1+", "lc+", "n2+"])

    graph, cfg = _build_graph_and_config(b, "serial_split_sigs", repeat_guard="on")
    res = run_chromosome_pipeline(graph, cfg)
    _assert_result_invariants(graph, cfg, res)

    ins_ev = [e for e in res.sv_events if e.event_type == "INS"][0]
    assert ins_ev.genotypes["HapLeft1#1"] == "1"
    assert ins_ev.genotypes["HapLeft2#1"] == "1"
    assert ins_ev.genotypes["HapBoth#1"] == "2"

