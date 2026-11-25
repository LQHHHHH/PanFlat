"""Unit and white-box regression tests for all 5 Review-Fix items (REVIEW_FIX_PLAN.md §4.1)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import pickle
import shutil
import sys

import pytest

from panflat.cli import main as cli_main
from panflat.gfa_io import load_and_validate_inputs
from panflat.gff_liftover import liftover_gff3_lines
from panflat.linearizer import (
    _ContigAnalysis,
    _ContigFragMeta,
    _contig_covers_anchor_pair,
    _contig_covers_rank_interval,
    reanchor_deviation,
    run_chromosome_pipeline,
)
from panflat.models import (
    BuildConfig,
    DeviationSegment,
    StepOffsetRow,
)
from panflat.repeat_guard import (
    _check_similarity_merge_with_span,
    check_similarity_merge,
)
from panflat.verify_qc import verify_build_outputs
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
    revcomp_dna,
)


def _run_pipeline_from_builder(
    tmp_path: Path,
    tag: str,
    gb: SyntheticGFAFixtureBuilder,
    ref_path_name: str,
    ref_steps: list[tuple[str, str]],
    backbone_prefix: str = "Ref#1#",
    max_jump: int = 1_000_000,
    max_copy: int = 2,
) -> tuple[Path, Path, object]:
    gfa_path = gb.write_gfa(tmp_path / f"{tag}.gfa")
    ref_seq = gb.spell_steps(ref_steps)
    ref_fa = gb.write_fasta(tmp_path / f"{tag}.ref.fa", {"Chr01": ref_seq})
    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(ref_fa),
        backbone_prefix=backbone_prefix,
        out_prefix=str(tmp_path / f"{tag}_out"),
        max_jump=max_jump,
        max_copy=max_copy,
    )
    graph = load_and_validate_inputs(cfg)
    res = run_chromosome_pipeline(graph, cfg)
    return gfa_path, ref_fa, res


def _build_via_cli(
    tmp_path: Path,
    tag: str,
    gb: SyntheticGFAFixtureBuilder,
    ref_steps: list[tuple[str, str]],
    backbone_prefix: str = "Ref#1#",
    extra_args: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    gfa_path = gb.write_gfa(tmp_path / f"{tag}.gfa")
    ref_seq = gb.spell_steps(ref_steps)
    ref_fa = gb.write_fasta(tmp_path / f"{tag}.ref.fa", {"Chr01": ref_seq})
    out_prefix = tmp_path / f"{tag}_out"
    argv = [
        "build",
        "--gfa",
        str(gfa_path),
        "--ref-fa",
        str(ref_fa),
        "--backbone-prefix",
        backbone_prefix,
        "--out-prefix",
        str(out_prefix),
        *(extra_args or []),
    ]
    assert cli_main(argv) == 0
    return gfa_path, ref_fa, out_prefix


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _rewrite_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as fh:
        fieldnames = csv.DictReader(fh, delimiter="\t").fieldnames
    assert fieldnames is not None
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Item 1: CNV branch-side shared-step contiguity + flanking DEL/INS coexistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf1_cnv_scattered_nodes_rejected_and_flanking_coexistence_preserved(
    tmp_path: Path,
    use_compact_ids: bool,
) -> None:
    """Item 1: scattered U-X-U-X-U must not call CNV; flanking DEL/INS + tandem U-U-U must preserve CNV."""
    ids = (
        {"A": "1", "U": "2", "D": "3", "B": "4", "X": "5"}
        if use_compact_ids
        else {"A": "nA", "U": "nU", "D": "nD", "B": "nB", "X": "nX"}
    )

    # Case A: Counterexample 1 — Ref = A-U-U-D-B, Alt = A-U-X-U-X-U-B -> 0 CNV, DEL(D) + INS(X)
    gb_a = SyntheticGFAFixtureBuilder()
    for idx, k in enumerate(["A", "U", "D", "B", "X"]):
        gb_a.add_node(ids[k], make_complex_seq(100, seed=1010 + idx))
    ref_a = [(ids[k], "+") for k in ["A", "U", "U", "D", "B"]]
    alt_a = [(ids[k], "+") for k in ["A", "U", "X", "U", "X", "U", "B"]]
    gb_a.add_p_path("Ref#1#Chr01", ref_a)
    gb_a.add_p_path("Alt1#1#Chr01", alt_a)

    _, _, res_a = _run_pipeline_from_builder(
        tmp_path, f"rf1_scattered_{use_compact_ids}", gb_a, "Ref#1#Chr01", ref_a
    )
    ev_types_a = [ev.event_type for ev in res_a.sv_events]
    assert "CNV" not in ev_types_a, f"Unexpected CNV on scattered branch: {res_a.sv_events}"
    assert "DEL" in ev_types_a
    assert "INS" in ev_types_a

    # Case B: Red Line 3 — Ref = A-D-U-U-B, Alt1 = A-U-U-U-B, Alt2 = A-U-U-U-X-B
    gb_b = SyntheticGFAFixtureBuilder()
    for idx, k in enumerate(["A", "D", "U", "B", "X"]):
        gb_b.add_node(ids[k], make_complex_seq(100, seed=1020 + idx))
    ref_b = [(ids[k], "+") for k in ["A", "D", "U", "U", "B"]]
    alt_b1 = [(ids[k], "+") for k in ["A", "U", "U", "U", "B"]]
    alt_b2 = [(ids[k], "+") for k in ["A", "U", "U", "U", "X", "B"]]
    gb_b.add_p_path("Ref#1#Chr01", ref_b)
    gb_b.add_p_path("Alt1#1#Chr01", alt_b1)
    gb_b.add_p_path("Alt2#1#Chr01", alt_b2)

    _, _, res_b = _run_pipeline_from_builder(
        tmp_path, f"rf1_flanking_{use_compact_ids}", gb_b, "Ref#1#Chr01", ref_b
    )
    cnv_evs = [ev for ev in res_b.sv_events if ev.event_type == "CNV"]
    del_evs = [ev for ev in res_b.sv_events if ev.event_type == "DEL"]
    ins_evs = [ev for ev in res_b.sv_events if ev.event_type == "INS"]
    assert len(cnv_evs) == 1
    assert len(del_evs) == 1
    assert len(ins_evs) == 1
    assert "k_ref=2,k_alt=3" in cnv_evs[0].alleles_str
    assert cnv_evs[0].genotypes["Alt1#1"] == "3"
    assert cnv_evs[0].genotypes["Alt2#1"] == "3"
    assert cnv_evs[0].region_id == del_evs[0].region_id == ins_evs[0].region_id


# ---------------------------------------------------------------------------
# Item 2.1: Strict §5.2 INS genotyping (missing evidence -> ".", ref-like -> "0")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf2_ins_genotyping_missing_evidence_dot_and_ref_like_zero(
    tmp_path: Path,
    use_compact_ids: bool,
) -> None:
    """Item 2.1: pure DEL, distant ref-block replacement, flanking DEL+rep0, and CNV expansion call '.'; exact & <50bp SNP call '0'."""
    ids = (
        {"A": "1", "D": "2", "B": "3", "RF": "4", "Z": "5", "X": "6", "DS": "7"}
        if use_compact_ids
        else {
            "A": "nA",
            "D": "nD",
            "B": "nB",
            "RF": "nRF",
            "Z": "nZ",
            "X": "nX",
            "DS": "nDS",
        }
    )
    seq_d = make_complex_seq(100, seed=2011)
    seq_d_snp = mutate_seq_snps(seq_d, [50])

    gb = SyntheticGFAFixtureBuilder()
    gb.add_node(ids["A"], make_complex_seq(100, seed=2010))
    gb.add_node(ids["D"], seq_d)
    gb.add_node(ids["B"], make_complex_seq(100, seed=2012))
    gb.add_node(ids["RF"], make_complex_seq(100, seed=2013))
    gb.add_node(ids["Z"], make_complex_seq(100, seed=2014))
    gb.add_node(ids["X"], make_complex_seq(100, seed=2015))
    gb.add_node(ids["DS"], seq_d_snp)

    ref_s = [(ids[k], "+") for k in ["A", "D", "B", "RF", "Z"]]
    gb.add_p_path("Ref#1#Chr01", ref_s)
    gb.add_p_path("Alt1#1#Chr01", [(ids[k], "+") for k in ["A", "X", "B", "RF", "Z"]])
    gb.add_p_path("Alt2#1#Chr01", [(ids[k], "+") for k in ["A", "B", "RF", "Z"]])
    gb.add_p_path("Alt3#1#Chr01", [(ids[k], "+") for k in ["A", "DS", "B", "RF", "Z"]])
    gb.add_p_path("Alt4#1#Chr01", [(ids[k], "+") for k in ["A", "D", "B", "RF", "Z"]])
    gb.add_p_path("Alt5#1#Chr01", [(ids[k], "+") for k in ["A", "RF", "B", "RF", "Z"]])

    _, _, res = _run_pipeline_from_builder(
        tmp_path, f"rf2_ins_main_{use_compact_ids}", gb, "Ref#1#Chr01", ref_s
    )
    ins_evs = [ev for ev in res.sv_events if ev.event_type == "INS"]
    assert len(ins_evs) == 1
    gt = ins_evs[0].genotypes
    assert gt["Ref#1"] == "0"
    assert gt["Alt1#1"] == "1"
    assert gt["Alt2#1"] == "."  # pure 100 bp DEL of D
    assert gt["Alt3#1"] == "0"  # 1-SNP bubble merging into rep0
    assert gt["Alt4#1"] == "0"  # exact match
    assert gt["Alt5#1"] == "."  # distant ref-block replacement
    assert ins_evs[0].n_called == 4  # Ref#1, Alt1#1, Alt3#1, Alt4#1

    # Case A edge: flanking 100 bp DEL + D2_snp merging into rep0=D1+D2 under HW -> must be "."
    gb2 = SyntheticGFAFixtureBuilder()
    d1_seq = make_complex_seq(100, seed=2021)
    d2_seq = make_complex_seq(100, seed=2022)
    gb2.add_node("1", make_complex_seq(100, seed=2020))
    gb2.add_node("2", d1_seq)
    gb2.add_node("3", d2_seq)
    gb2.add_node("4", make_complex_seq(100, seed=2023))
    gb2.add_node("5", make_complex_seq(120, seed=2024))
    gb2.add_node("6", mutate_seq_snps(d2_seq, [40]))
    ref_s2 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb2.add_p_path("Ref#1#Chr01", ref_s2)
    gb2.add_p_path("Alt1#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb2.add_p_path("Alt2#1#Chr01", [("1", "+"), ("6", "+"), ("4", "+")])
    gb2.add_p_path("Alt3#1#Chr01", ref_s2)
    _, _, res2 = _run_pipeline_from_builder(
        tmp_path, f"rf2_ins_case_a_flankdel_{use_compact_ids}", gb2, "Ref#1#Chr01", ref_s2
    )
    ins_evs2 = [ev for ev in res2.sv_events if ev.event_type == "INS"]
    assert len(ins_evs2) == 1
    assert ins_evs2[0].genotypes["Alt1#1"] == "1"
    assert ins_evs2[0].genotypes["Alt2#1"] == "."
    assert ins_evs2[0].genotypes["Alt3#1"] == "0"

    # Case B edge: tandem CNV expansion inside G(a, b) -> "." on INS; split-node 1-bp SNP -> "0"
    gb3 = SyntheticGFAFixtureBuilder()
    gb3.add_node("1", make_complex_seq(100, seed=2030))
    gb3.add_node("2", make_complex_seq(80, seed=2031))
    gb3.add_node("3", make_complex_seq(100, seed=2032))
    gb3.add_node("4", make_complex_seq(110, seed=2033))
    ref_s3 = [("1", "+"), ("2", "+"), ("2", "+"), ("3", "+")]
    gb3.add_p_path("Ref#1#Chr01", ref_s3)
    gb3.add_p_path("Alt1#1#Chr01", [("1", "+"), ("4", "+"), ("3", "+")])
    gb3.add_p_path("Alt2#1#Chr01", [("1", "+"), ("2", "+"), ("2", "+"), ("2", "+"), ("3", "+")])
    gb3.add_p_path("Alt3#1#Chr01", ref_s3)
    _, _, res3 = _run_pipeline_from_builder(
        tmp_path, f"rf2_ins_cnv_{use_compact_ids}", gb3, "Ref#1#Chr01", ref_s3
    )
    ins_evs3 = [ev for ev in res3.sv_events if ev.event_type == "INS"]
    assert len(ins_evs3) == 1
    assert ins_evs3[0].genotypes["Alt2#1"] == "."
    assert ins_evs3[0].genotypes["Alt3#1"] == "0"

    # Split-node 1-bp SNP bubble inside B(a, b)
    gb4 = SyntheticGFAFixtureBuilder()
    gb4.add_node("1", make_complex_seq(100, seed=2040))
    gb4.add_node("2", make_complex_seq(50, seed=2041))
    gb4.add_node("3", "A")
    gb4.add_node("4", make_complex_seq(49, seed=2042))
    gb4.add_node("5", make_complex_seq(100, seed=2043))
    gb4.add_node("6", make_complex_seq(100, seed=2044))
    gb4.add_node("7", "C")
    ref_s4 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")]
    gb4.add_p_path("Ref#1#Chr01", ref_s4)
    gb4.add_p_path("Alt1#1#Chr01", [("1", "+"), ("6", "+"), ("5", "+")])
    gb4.add_p_path("Alt2#1#Chr01", [("1", "+"), ("2", "+"), ("7", "+"), ("4", "+"), ("5", "+")])
    gb4.add_p_path("Alt3#1#Chr01", [("1", "+"), ("5", "+")])
    _, _, res4 = _run_pipeline_from_builder(
        tmp_path, f"rf2_ins_splitsnp_{use_compact_ids}", gb4, "Ref#1#Chr01", ref_s4
    )
    ins_evs4 = [ev for ev in res4.sv_events if ev.event_type == "INS"]
    assert len(ins_evs4) == 1
    assert ins_evs4[0].genotypes["Alt1#1"] == "1"
    assert ins_evs4[0].genotypes["Alt2#1"] == "0"
    assert ins_evs4[0].genotypes["Alt3#1"] == "."


# ---------------------------------------------------------------------------
# Item 2.2: Strict §5.2 INV genotyping (flanking anchors, single-node LARGE_JUMP)
# ---------------------------------------------------------------------------


def test_rf2_inv_genotyping_internal_short_contigs_and_single_node_large_jump(
    tmp_path: Path,
) -> None:
    """Item 2.2: short internal B+ / flipped B- contigs call '.'; single-node LARGE_JUMP & partial INV call '.'."""
    # Direct unit tests on _contig_covers_rank_interval
    def _mk_ca(
        cov: set[int],
        unnorm: list[tuple[int, int]] | None = None,
        lj: list[tuple[int, int]] | None = None,
        norm_inv: list[tuple[int, int]] | None = None,
    ) -> _ContigAnalysis:
        return _ContigAnalysis(
            raw_fragment=_ContigFragMeta("H#1", "H#1#Chr01"),
            lis_multi_hit=False,
            covered_ranks=cov,
            unnormalized_rank_spans=unnorm or [],
            large_jump_rank_spans=lj or [],
            node_visit_counts={},
            has_main_chain=bool(cov),
            contig_idx=0,
            normalized_inv_rank_spans=norm_inv or [],
        )

    # Single-node inversion rmin==rmax==2 crossed by LARGE_JUMP (1, 3) -> False!
    assert _contig_covers_rank_interval(_mk_ca({0, 1, 3, 4}, lj=[(1, 3)]), 2, 2, 5) is False
    # Overlapping normalized inversion -> False!
    assert _contig_covers_rank_interval(_mk_ca({0, 1, 2, 3, 4}, norm_inv=[(1, 2)]), 2, 3, 5) is False
    # Internal short contig covering only [1, 2] without external flanks -> False!
    assert _contig_covers_rank_interval(_mk_ca({1, 2}), 1, 2, 5) is False
    assert _contig_covers_rank_interval(_mk_ca({0, 1, 2, 3}), 1, 2, 5) is True
    # Left INV_EDGE (rmin == 0): needs 0 and right flank > rmax
    assert _contig_covers_rank_interval(_mk_ca({0, 1}), 0, 1, 5) is False
    assert _contig_covers_rank_interval(_mk_ca({1, 2}), 0, 1, 5) is False
    assert _contig_covers_rank_interval(_mk_ca({0, 1, 2}), 0, 1, 5) is True
    # Right INV_EDGE (rmax == n_ref - 1): needs n_ref-1 and left flank < rmin
    assert _contig_covers_rank_interval(_mk_ca({3, 4}), 3, 4, 5) is False
    assert _contig_covers_rank_interval(_mk_ca({2, 3, 4}), 3, 4, 5) is True
    # Whole-chromosome span -> False
    assert _contig_covers_rank_interval(_mk_ca({0, 1, 2, 3, 4}), 0, 4, 5) is False

    # End-to-end Counterexample 2b: Ref = A-B-C, Alt1 = A-B--C, Alt2 = B+, Alt3 = B-, Alt4 = A-B-C
    gb = SyntheticGFAFixtureBuilder()
    gb.add_node("1", make_complex_seq(150, seed=2201))
    gb.add_node("2", make_complex_seq(200, seed=2202))
    gb.add_node("3", make_complex_seq(150, seed=2203))
    ref_s = [("1", "+"), ("2", "+"), ("3", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_s)
    gb.add_p_path("Alt1#1#Chr01", [("1", "+"), ("2", "-"), ("3", "+")])
    gb.add_p_path("Alt2#1#Chr01", [("2", "+")])
    gb.add_p_path("Alt3#1#Chr01", [("2", "-")])
    gb.add_p_path("Alt4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+")])

    _, _, res = _run_pipeline_from_builder(
        tmp_path, "rf2_inv_counterexample", gb, "Ref#1#Chr01", ref_s
    )
    inv_evs = [ev for ev in res.sv_events if ev.event_type == "INV"]
    assert len(inv_evs) == 1
    gt = inv_evs[0].genotypes
    assert gt["Ref#1"] == "0"
    assert gt["Alt1#1"] == "1"
    assert gt["Alt2#1"] == "."
    assert gt["Alt3#1"] == "."
    assert gt["Alt4#1"] == "0"
    assert inv_evs[0].n_called == 3  # Ref#1, Alt1#1, Alt4#1

    # Single-node INV crossed by LARGE_JUMP in full pipeline
    gb_lj = SyntheticGFAFixtureBuilder()
    gb_lj.add_node("1", make_complex_seq(150, seed=2211))
    gb_lj.add_node("2", make_complex_seq(200, seed=2212))
    gb_lj.add_node("3", make_complex_seq(200, seed=2213))
    gb_lj.add_node("4", make_complex_seq(150, seed=2214))
    ref_lj = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb_lj.add_p_path("Ref#1#Chr01", ref_lj)
    gb_lj.add_p_path("Alt1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "-"), ("4", "+")])
    gb_lj.add_p_path("Alt2#1#Chr01", [("1", "+"), ("4", "+")])  # skips 400 bp > max_jump=150
    _, _, res_lj = _run_pipeline_from_builder(
        tmp_path,
        "rf2_inv_single_lj",
        gb_lj,
        "Ref#1#Chr01",
        ref_lj,
        max_jump=150,
    )
    inv_lj = [ev for ev in res_lj.sv_events if ev.event_type == "INV"]
    assert len(inv_lj) == 1
    assert inv_lj[0].genotypes["Alt1#1"] == "1"
    assert inv_lj[0].genotypes["Alt2#1"] == "."

    # INV_DISPLACED: short contig covering displaced flanks (3, 4) without left flank < rmin=1 -> "."
    gb_d = SyntheticGFAFixtureBuilder()
    for i in range(1, 6):
        gb_d.add_node(str(i), make_complex_seq(150, seed=2220 + i))
    ref_d = [(str(i), "+") for i in range(1, 6)]
    gb_d.add_p_path("Ref#1#Chr01", ref_d)
    gb_d.add_p_path("Alt1#1#Chr01", [("1", "+"), ("4", "+"), ("3", "-"), ("2", "-"), ("5", "+")])
    gb_d.add_p_path("Alt2#1#Chr01", [("4", "+"), ("5", "+")])
    gb_d.add_p_path("Alt3#1#Chr01", ref_d)
    _, _, res_d = _run_pipeline_from_builder(
        tmp_path, "rf2_inv_displaced", gb_d, "Ref#1#Chr01", ref_d
    )
    inv_disp = [ev for ev in res_d.sv_events if ev.event_type == "INV_DISPLACED"]
    assert len(inv_disp) == 1
    assert inv_disp[0].genotypes["Alt1#1"] == "1"
    assert inv_disp[0].genotypes["Alt2#1"] == "."
    assert inv_disp[0].genotypes["Alt3#1"] == "0"


# ---------------------------------------------------------------------------
# Item 3.1 & 3.2: Short-sequence Criterion 1 protection + Criterion 3/4 boundary
# ---------------------------------------------------------------------------


def test_rf3_similarity_merge_short_multihit_revcomp_split_and_small_ins_boundary() -> None:
    """Item 3: short edlib HW multi-hit/revcomp/split-D rejection + 1-49 bp pure ins preservation."""
    u200 = make_complex_seq(200, seed=3001)
    u200_2snp = mutate_seq_snps(u200, [40, 150])

    # Criterion 1a: tied exact multi-hit U in U+U -> False
    assert check_similarity_merge(u200, u200 + u200) is False

    # Criterion 1b: near-optimal secondary forward hit (right & left) -> False
    assert check_similarity_merge(u200, u200 + u200_2snp) is False
    assert check_similarity_merge(u200, u200_2snp + u200) is False

    # Criterion 1c: reverse-complement hit U in U + revcomp(U) -> False
    assert check_similarity_merge(u200, u200 + revcomp_dna(u200)) is False

    # Criterion 1a: overlapping shifted tandem loci U+U (200) in U+U+U (300) -> False
    u100 = make_complex_seq(100, seed=3002)
    assert check_similarity_merge(u100 + u100, u100 + u100 + u100) is False

    # Criterion 1d (edlib): single 50D or cumulative 25D+25D=50D -> False; 30D and 25D+10D -> True
    r500 = make_complex_seq(500, seed=3003)
    assert check_similarity_merge(r500[:200] + r500[250:], r500) is False
    r600 = make_complex_seq(600, seed=3004)
    assert check_similarity_merge(r600[:200] + r600[225:400] + r600[425:], r600) is False
    assert check_similarity_merge(r600[:250] + r600[280:], r600) is True
    assert check_similarity_merge(r600[:200] + r600[225:400] + r600[410:], r600) is True

    # Criterion 1d (mappy > 1000 bp): single 100D chained deletion -> False
    r2000 = make_complex_seq(2000, seed=3005)
    assert check_similarity_merge(r2000[:950] + r2000[1050:], r2000) is False

    # Red Line 2: 200=30I200= and 225=49I225= (both full & embedded t_start>0) -> True; 50I -> False
    p1 = make_complex_seq(200, seed=3010)
    p2 = make_complex_seq(200, seed=3011)
    assert check_similarity_merge(p1 + make_complex_seq(30, seed=3012) + p2, p1 + p2) is True

    m1 = make_complex_seq(225, seed=3013)
    m2 = make_complex_seq(225, seed=3014)
    ins49 = make_complex_seq(49, seed=3015)
    ins50 = make_complex_seq(50, seed=3016)
    assert check_similarity_merge(m1 + ins49 + m2, m1 + m2) is True
    fl = make_complex_seq(25, seed=3017)
    fr = make_complex_seq(25, seed=3018)
    assert check_similarity_merge(m1 + ins49 + m2, fl + m1 + m2 + fr) is True
    assert check_similarity_merge(m1[1:] + ins50 + m2, fl + m1 + m2 + fr) is False

    # Asymmetric substitution block: 15= + 5X + 30I inside a 50 bp window -> False
    r400 = make_complex_seq(400, seed=3020)
    mut5 = mutate_seq_snps(r400, [202, 205, 208, 211, 214])
    s_asym = mut5[:216] + make_complex_seq(30, seed=3021) + mut5[216:]
    assert check_similarity_merge(s_asym, r400) is False

    # Unique prefix substring U(500) in U(500) + unique(200) -> True (matches mappy)
    u500 = make_complex_seq(500, seed=3022)
    uniq200 = make_complex_seq(200, seed=3023)
    assert check_similarity_merge(u500, u500 + uniq200) is True

    # Float precision check at cluster_win_id=0.80: (1.0 - 0.80) * 50 = 10.0
    # 4 SNPs + 6 bp insertion inside the same 50 bp window (win_i + win_x == 10): must NOT fail due to IEEE-754 truncation
    r400_f = make_complex_seq(400, seed=3025)
    mut4_f = mutate_seq_snps(r400_f, [202, 206, 210, 214])
    s_eq10 = mut4_f[:228] + make_complex_seq(6, seed=3026) + mut4_f[228:]
    assert check_similarity_merge(s_eq10, r400_f, cluster_win_id=0.80) is True
    # 4 SNPs + 7 bp insertion inside the same 50 bp window (win_i + win_x == 11 > 10): 39/50 = 0.78 < 0.80 -> False
    s_gt10 = mut4_f[:228] + make_complex_seq(7, seed=3027) + mut4_f[228:]
    assert check_similarity_merge(s_gt10, r400_f, cluster_win_id=0.80) is False


# ---------------------------------------------------------------------------
# Item 4.1 & 4.2: Precise join_before + verify_qc Global Invariant II rejection
# ---------------------------------------------------------------------------


def test_rf4_source_contiguous_exact_and_verify_inv2_tamper_rejection(
    tmp_path: Path,
) -> None:
    """Item 4: exact synthetic vs source_contiguous marking + verify_qc INV2 tamper rejection (small & >10MB streaming)."""

    def _tamper_join_before_and_assert_inv2(
        out_prefix: Path,
        gfa_path: Path,
        ref_fa: Path,
        row_idx: int,
        new_join: str,
    ) -> None:
        clone_p = tmp_path / f"{out_prefix.name}_tamper_{row_idx}_{new_join}"
        for f in out_prefix.parent.glob(f"{out_prefix.name}.*"):
            shutil.copy(f, f.parent / f.name.replace(out_prefix.name, clone_p.name))
        ins_file = Path(f"{clone_p}.inserts.tsv")
        rows = _read_tsv(ins_file)
        rows[row_idx]["join_before"] = new_join
        _rewrite_tsv(ins_file, rows)
        rep = verify_build_outputs(clone_p, gfa_path, ref_fa)
        assert not rep.passed
        assert not rep.invariant_2_passed
        assert any("[INV2]" in e for e in rep.errors), f"Expected [INV2], got {rep.errors}"

    # A. Counterexample 4a: Ref = A-B-D, Alt = A-D-X-B-D -> X starts after stripped D -> join_before == "synthetic"
    gb_a = SyntheticGFAFixtureBuilder()
    for nid, seed in [("A", 4001), ("B", 4002), ("D", 4003), ("X", 4004)]:
        gb_a.add_node(nid, make_complex_seq(150, seed=seed))
    ref_a = [("A", "+"), ("B", "+"), ("D", "+")]
    gb_a.add_p_path("Ref#1#Chr01", ref_a)
    gb_a.add_p_path("Alt1#1#Chr01", [("A", "+"), ("D", "+"), ("X", "+"), ("B", "+"), ("D", "+")])
    gfa_a, fa_a, out_a = _build_via_cli(tmp_path, "rf4_stripped", gb_a, ref_a)
    rows_a = _read_tsv(Path(f"{out_a}.inserts.tsv"))
    assert len(rows_a) == 1
    assert rows_a[0]["join_before"] == "synthetic"
    assert verify_build_outputs(out_a, gfa_a, fa_a).passed
    _tamper_join_before_and_assert_inv2(out_a, gfa_a, fa_a, 0, "source_contiguous")

    # B. Repeat-guard left-prefix-trimmed (u_start > 0 -> synthetic) vs right-trimmed (u_start == 0 -> source_contiguous)
    gb_rg = SyntheticGFAFixtureBuilder()
    gb_rg.add_node("1", make_complex_seq(200, seed=4010))
    gb_rg.add_node("2", make_complex_seq(200, seed=4011))
    gb_rg.add_node("3", make_complex_seq(200, seed=4012))
    sat = make_satellite_seq(unit_len=20, total_len=1500, seed=4013)
    gb_rg.add_node("4", sat + make_complex_seq(400, seed=4014))  # left-trimmed
    gb_rg.add_node("5", make_complex_seq(400, seed=4015) + sat)  # right-trimmed
    ref_rg = [("1", "+"), ("2", "+"), ("3", "+")]
    gb_rg.add_p_path("Ref#1#Chr01", ref_rg)
    gb_rg.add_p_path("Alt1#1#Chr01", [("1", "+"), ("4", "+"), ("2", "+"), ("5", "+"), ("3", "+")])
    gfa_rg, fa_rg, out_rg = _build_via_cli(tmp_path, "rf4_rg", gb_rg, ref_rg)
    rows_rg = _read_tsv(Path(f"{out_rg}.inserts.tsv"))
    assert len(rows_rg) == 2
    assert rows_rg[0]["join_before"] == "synthetic"  # left-trimmed u_start > 0
    assert rows_rg[1]["join_before"] == "source_contiguous"  # right-trimmed u_start == 0 (Red Line 4)
    assert verify_build_outputs(out_rg, gfa_rg, fa_rg).passed
    _tamper_join_before_and_assert_inv2(out_rg, gfa_rg, fa_rg, 0, "source_contiguous")

    # C. Inversion-boundary (synthetic) vs T7 inverted interior (source_contiguous) vs T23 W-line revcomp (source_contiguous)
    gb_inv = SyntheticGFAFixtureBuilder()
    for nid, seed in [("11", 4020), ("12", 4021), ("13", 4022), ("14", 4023), ("15", 4024), ("16", 4025)]:
        gb_inv.add_node(nid, make_complex_seq(200, seed=seed))
    ref_inv = [("11", "+"), ("12", "+"), ("13", "+"), ("14", "+")]
    gb_inv.add_p_path("Ref#1#Chr01", ref_inv)
    # Boundary insertion after inverted 13-12: normalized 13+ is followed by 15+ (source jump -> synthetic)
    gb_inv.add_p_path("AltBnd#1#Chr01", [("11", "+"), ("13", "-"), ("12", "-"), ("15", "+"), ("14", "+")])
    # T7 inverted interior insertion: 11+ 13- 16+ 12- 14+ -> normalized 12+ 16- 13+ (source_contiguous on '-')
    gb_inv.add_p_path("AltT7#1#Chr01", [("11", "+"), ("13", "-"), ("16", "+"), ("12", "-"), ("14", "+")])
    gfa_inv, fa_inv, out_inv = _build_via_cli(tmp_path, "rf4_inv_bnd", gb_inv, ref_inv)
    rows_inv = _read_tsv(Path(f"{out_inv}.inserts.tsv"))
    by_hap = {r["src_hap"]: r for r in rows_inv}
    assert by_hap["AltBnd#1"]["join_before"] == "synthetic"
    assert by_hap["AltT7#1"]["join_before"] == "source_contiguous"
    assert verify_build_outputs(out_inv, gfa_inv, fa_inv).passed
    bnd_idx = next(i for i, r in enumerate(rows_inv) if r["src_hap"] == "AltBnd#1")
    _tamper_join_before_and_assert_inv2(out_inv, gfa_inv, fa_inv, bnd_idx, "source_contiguous")

    # T23 W-line whole-contig revcomp with non-zero start
    gb_t23 = SyntheticGFAFixtureBuilder()
    for nid, seed in [("1", 4030), ("2", 4031), ("3", 4032)]:
        gb_t23.add_node(nid, make_complex_seq(200, seed=seed))
    ref_t23 = [("1", "+"), ("2", "+")]
    gb_t23.add_p_path("Ref#1#Chr01", ref_t23)
    gb_t23.add_w_walk("AltW", 1, "ctgRev", 10000, None, [("2", "-"), ("3", "-"), ("1", "-")])
    gfa_t23, fa_t23, out_t23 = _build_via_cli(tmp_path, "rf4_t23", gb_t23, ref_t23)
    rows_t23 = _read_tsv(Path(f"{out_t23}.inserts.tsv"))
    assert len(rows_t23) == 1
    assert rows_t23[0]["src_strand"] == "-"
    assert rows_t23[0]["join_before"] == "source_contiguous"
    assert verify_build_outputs(out_t23, gfa_t23, fa_t23).passed

    # D. Cycle folding U-U-Tail (--max-copy 1) -> piece 0 source_contiguous, piece 1 synthetic; tamper piece 1 -> INV2 (Situation I)
    gb_cf = SyntheticGFAFixtureBuilder()
    for nid, seed in [("1", 4040), ("2", 4041), ("3", 4042), ("4", 4043)]:
        gb_cf.add_node(nid, make_complex_seq(120, seed=seed))
    ref_cf = [("1", "+"), ("2", "+")]
    gb_cf.add_p_path("Ref#1#Chr01", ref_cf)
    gb_cf.add_p_path("Alt1#1#Chr01", [("1", "+"), ("3", "+"), ("3", "+"), ("4", "+"), ("2", "+")])
    gfa_cf, fa_cf, out_cf = _build_via_cli(
        tmp_path, "rf4_cycle", gb_cf, ref_cf, extra_args=["--max-copy", "1"]
    )
    rows_cf = _read_tsv(Path(f"{out_cf}.inserts.tsv"))
    assert len(rows_cf) == 2
    assert rows_cf[0]["join_before"] == "source_contiguous"
    assert rows_cf[1]["join_before"] == "synthetic"
    assert verify_build_outputs(out_cf, gfa_cf, fa_cf).passed
    _tamper_join_before_and_assert_inv2(out_cf, gfa_cf, fa_cf, 1, "source_contiguous")

    # E. Situation III: tampering a "spacer" piece to "source_contiguous" or "synthetic" -> INV2
    gb_sp = SyntheticGFAFixtureBuilder()
    for nid, seed in [("1", 4050), ("2", 4051), ("3", 4052), ("4", 4053)]:
        gb_sp.add_node(nid, make_complex_seq(150, seed=seed))
    ref_sp = [("1", "+"), ("2", "+")]
    gb_sp.add_p_path("Ref#1#Chr01", ref_sp)
    gb_sp.add_p_path("Alt1#1#Chr01", [("1", "+"), ("3", "+"), ("2", "+")])
    gb_sp.add_p_path("Alt2#1#Chr01", [("1", "+"), ("4", "+"), ("2", "+")])
    gfa_sp, fa_sp, out_sp = _build_via_cli(tmp_path, "rf4_spacer", gb_sp, ref_sp)
    rows_sp = _read_tsv(Path(f"{out_sp}.inserts.tsv"))
    assert len(rows_sp) == 2
    assert rows_sp[0]["join_before"] == "source_contiguous"
    assert rows_sp[1]["join_before"] == "spacer"
    assert verify_build_outputs(out_sp, gfa_sp, fa_sp).passed
    _tamper_join_before_and_assert_inv2(out_sp, gfa_sp, fa_sp, 1, "source_contiguous")
    _tamper_join_before_and_assert_inv2(out_sp, gfa_sp, fa_sp, 1, "synthetic")

    # F. >10 MB streaming GFA verify parity (Situation II: genuine source_contiguous passes; tampered synthetic->source_contiguous fails INV2)
    large_gfa = tmp_path / "rf4_large_stream.gfa"
    large_fa = tmp_path / "rf4_large_stream.fa"
    s1 = make_complex_seq(200, seed=4060)
    s2 = make_complex_seq(200, seed=4061)
    s3 = make_complex_seq(200, seed=4062)
    s4 = make_complex_seq(200, seed=4063)  # pure ins between 1 and 2
    s5 = make_complex_seq(200, seed=4064)  # stripped-ref ins between 2 and 3 (after stripped 1+)
    pad_seq = "ACGT" * 55_000  # 220,000 bp x 50 nodes = 11 MB (>10 MB threshold)
    with large_gfa.open("w", encoding="utf-8") as fh:
        fh.write("H\tVN:Z:1.0\n")
        fh.write(f"S\t1\t{s1}\n")
        fh.write(f"S\t2\t{s2}\n")
        fh.write(f"S\t3\t{s3}\n")
        fh.write(f"S\t4\t{s4}\n")
        fh.write(f"S\t5\t{s5}\n")
        for pid in range(6, 56):
            fh.write(f"S\t{pid}\t{pad_seq}\n")
        fh.write("L\t1\t+\t2\t+\t0M\n")
        fh.write("L\t2\t+\t3\t+\t0M\n")
        fh.write("L\t1\t+\t4\t+\t0M\n")
        fh.write("L\t4\t+\t2\t+\t0M\n")
        fh.write("L\t2\t+\t1\t+\t0M\n")
        fh.write("L\t1\t+\t5\t+\t0M\n")
        fh.write("L\t5\t+\t3\t+\t0M\n")
        fh.write("P\tRef#1#Chr01\t1+,2+,3+\t*\n")
        fh.write("P\tAltPure#1#Chr01\t1+,4+,2+,3+\t*\n")
        fh.write("P\tAltStrip#1#Chr01\t1+,2+,1+,5+,3+\t*\n")
    assert large_gfa.stat().st_size > 10_000_000
    SyntheticGFAFixtureBuilder.write_fasta(large_fa, {"Chr01": s1 + s2 + s3})
    out_lg = tmp_path / "rf4_large_out"
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
                str(out_lg),
            ]
        )
        == 0
    )
    rows_lg = _read_tsv(Path(f"{out_lg}.inserts.tsv"))
    assert len(rows_lg) == 2
    pure_idx = next(i for i, r in enumerate(rows_lg) if r["src_hap"] == "AltPure#1")
    strip_idx = next(i for i, r in enumerate(rows_lg) if r["src_hap"] == "AltStrip#1")
    assert rows_lg[pure_idx]["join_before"] == "source_contiguous"
    assert rows_lg[strip_idx]["join_before"] == "synthetic"
    assert verify_build_outputs(out_lg, large_gfa, large_fa).passed
    # Tamper AltStrip piece (directly following ref step 2+, Situation II fast_step_bounds) to source_contiguous -> INV2!
    _tamper_join_before_and_assert_inv2(out_lg, large_gfa, large_fa, strip_idx, "source_contiguous")


# ---------------------------------------------------------------------------
# Item 5: 1,200-layer nested local re-anchoring + 1,200-deep GFF3 container chain + recursion limit restoration
# ---------------------------------------------------------------------------


def _build_nested_reanchor_ladder(
    depth: int = 1200,
) -> tuple[SyntheticGFAFixtureBuilder, list[tuple[str, str]], list[tuple[str, str]], str]:
    """
    Construct a valid GFA forcing EXACTLY `depth` strictly nested local re-anchoring layers:
      Ref = [u1, S, u2, u1, u3, u2, ..., u_D, u_{D-1}, u_D, E]
      Alt = [u1, S, u1, u2, ..., u_D, X(60bp), E]
    At layer k (1..D), u_k is the unique anchor in B[lo_k:hi_k), shrinking the interval step by step
    until X(60bp) is terminals-mounted between u_D and E.
    """
    gb = SyntheticGFAFixtureBuilder()
    # Node IDs: 1 = S (100bp), 2 = E (100bp), 3 = X (60bp), 4..4+depth-1 = u_1..u_D (10bp each)
    gb.add_node("1", make_complex_seq(100, seed=5001))
    gb.add_node("2", make_complex_seq(100, seed=5002))
    x_seq = make_complex_seq(60, seed=5003)
    gb.add_node("3", x_seq)
    for k in range(1, depth + 1):
        gb.add_node(str(3 + k), make_complex_seq(10, seed=5100 + k))

    def _u(k: int) -> tuple[str, str]:
        return (str(3 + k), "+")

    ref_steps: list[tuple[str, str]] = [_u(1), ("1", "+")]
    for k in range(1, depth):
        ref_steps.append(_u(k + 1))
        ref_steps.append(_u(k))
    ref_steps.append(_u(depth))
    ref_steps.append(("2", "+"))

    alt_steps: list[tuple[str, str]] = [_u(1), ("1", "+")]
    for k in range(1, depth + 1):
        alt_steps.append(_u(k))
    alt_steps.append(("3", "+"))
    alt_steps.append(("2", "+"))

    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path("Alt1#1#Chr01", alt_steps)
    return gb, ref_steps, alt_steps, x_seq


def test_rf5_deep_nested_reanchor_1200_layers_and_gff_container_chain(
    tmp_path: Path,
) -> None:
    """Item 5: 1,200-layer nested re-anchor + 1,200-deep GFF3 container chain without RecursionError."""
    depth = 1200
    gb, ref_steps, _, x_seq = _build_nested_reanchor_ladder(depth=depth)
    gfa_path, ref_fa, out_prefix = _build_via_cli(tmp_path, "rf5_ladder_1200", gb, ref_steps)

    # Verify independent QC passes
    rep = verify_build_outputs(out_prefix, gfa_path, ref_fa)
    assert rep.passed, f"verify failed: {rep.errors}"

    # Verify the 60 bp insertion X was accurately recovered and mounted at the terminal locus
    sv_rows = _read_tsv(Path(f"{out_prefix}.sv.tsv"))
    ins_rows = [r for r in sv_rows if r["type"] == "INS"]
    assert len(ins_rows) == 1
    assert "len=60" in ins_rows[0]["alleles"]
    inserts_rows = _read_tsv(Path(f"{out_prefix}.inserts.tsv"))
    assert len(inserts_rows) == 1
    assert int(inserts_rows[0]["pan_end"]) - int(inserts_rows[0]["pan_start"]) == 60
    assert inserts_rows[0]["join_before"] == "source_contiguous"
    pan_seq = "".join(
        line.strip()
        for line in Path(f"{out_prefix}.pan.fa").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    )
    ps, pe = int(inserts_rows[0]["pan_start"]), int(inserts_rows[0]["pan_end"])
    assert pan_seq[ps:pe] == x_seq

    # Direct white-box call to reanchor_deviation at 1,200 depth (explicitly below current recursion limit)
    graph = load_and_validate_inputs(
        BuildConfig(
            gfa=str(gfa_path),
            ref_fa=str(ref_fa),
            backbone_prefix="Ref#1#",
            out_prefix=str(tmp_path / "rf5_direct_out"),
        )
    )
    alt_frag = graph.alt_fragments[0]
    init_dev = DeviationSegment(
        hap_id=alt_frag.hap_id,
        src_path=alt_frag.src_path,
        left_anchor_rank=1,
        right_anchor_rank=len(ref_steps) - 1,
        branch_start_idx=2,
        branch_end_idx=len(alt_frag.steps) - 1,
        branch_steps=alt_frag.steps[2 : len(alt_frag.steps) - 1],
        branch_provenance=alt_frag.provenance[2 : len(alt_frag.steps) - 1],
        backbone_steps=graph.ref_fragment.steps[2 : len(ref_steps) - 1],
    )
    cov_ranks: set[int] = {1, len(ref_steps) - 1}
    refined = reanchor_deviation(
        dev=init_dev,
        backbone_steps=graph.ref_fragment.steps,
        nodes=graph.nodes,
        lo=0,
        hi=len(ref_steps),
        covered_ranks_out=cov_ranks,
    )
    assert any(
        len(d.branch_steps) == 1 and d.branch_steps[0].node_id == "3"
        for d in refined
    )
    assert (len(ref_steps) - 2) in cov_ranks

    # 1,200-deep GFF3 container chain -> iterative post-order traversal without RecursionError
    gff_lines = ["##gff-version 3"]
    gff_lines.append("Chr01\tsim\tgene\t1\t500\t.\t+\t.\tID=c0")
    for k in range(1, depth + 1):
        gff_lines.append(f"Chr01\tsim\tmRNA\t1\t500\t.\t+\t.\tID=c{k};Parent=c{k - 1}")
    gff_lines.append(f"Chr01\tsim\texon\t10\t200\t.\t+\t.\tID=ex1;Parent=c{depth}")
    gff_lines.append(f"Chr01\tsim\tCDS\t10\t200\t.\t+\t0\tID=cds1;Parent=c{depth}")
    offsets = [
        StepOffsetRow(
            chrom="Chr01",
            ref_step=0,
            node_id="1",
            ref_start=0,
            ref_end=500,
            pan_start=0,
            pan_end=500,
            slot_ins_len=0,
        )
    ]
    lifted = liftover_gff3_lines(gff_lines, "Chr01", offsets, [])
    assert sum(1 for ln in lifted if ln and not ln.startswith("#")) == depth + 3

    # Verify sys.getrecursionlimit() is restored even after solving a component whose size triggers headroom guard
    before_limit = sys.getrecursionlimit()
    gb_rec = SyntheticGFAFixtureBuilder()
    gb_rec.add_node("1", make_complex_seq(100, seed=5201))
    gb_rec.add_node("2", make_complex_seq(100, seed=5202))
    gb_rec.add_node("3", make_complex_seq(100, seed=5203))
    gb_rec.add_node("4", make_complex_seq(100, seed=5204))
    gb_rec.add_node("5", make_complex_seq(100, seed=5205))
    ref_rec = [("1", "+"), ("2", "+")]
    gb_rec.add_p_path("Ref#1#Chr01", ref_rec)
    gb_rec.add_p_path("Alt1#1#Chr01", [("1", "+"), ("3", "+"), ("4", "+"), ("2", "+")])
    gb_rec.add_p_path("Alt2#1#Chr01", [("1", "+"), ("3", "+"), ("5", "+"), ("2", "+")])
    # Lower recursion limit briefly? No: safely verify headroom guard restores original limit cleanly
    _, _, _ = _run_pipeline_from_builder(tmp_path, "rf5_reclimit", gb_rec, "Ref#1#Chr01", ref_rec)
    assert sys.getrecursionlimit() == before_limit


# ---------------------------------------------------------------------------
# Wave 1 Consolidated Regression Tests (Items 2.1, 2.2, 3.1, 3.2, 4.2)
# ---------------------------------------------------------------------------


def test_rf2_wave1_case_a_monotone_coverage_and_case_b_dropped_small_ins(
    tmp_path: Path,
) -> None:
    """Wave 1 Item 2.1: all 5 Case A counterexamples call '.', positive controls call '0', and Stage 4 dropped <50bp novel insertions accumulate cleanly in Case B (compact & string)."""
    # 1. Probe A-7b (50bp DEL D1 + 1bp INS in D2 -> '.') + positive controls (D_snp, D+30I, flanking D_del30 -> '0')
    d1 = make_complex_seq(50, seed=9401)
    d2 = make_complex_seq(100, seed=9402)
    gb_a7b = SyntheticGFAFixtureBuilder()
    gb_a7b.add_node("1", make_complex_seq(150, seed=9400))
    gb_a7b.add_node("2", d1)
    gb_a7b.add_node("3", d2)
    gb_a7b.add_node("4", make_complex_seq(150, seed=9403))
    gb_a7b.add_node("5", make_complex_seq(150, seed=9404))
    gb_a7b.add_node("6", d2[:50] + "A" + d2[50:])
    gb_a7b.add_node("7", mutate_seq_snps(d1 + d2, [75]))
    gb_a7b.add_node("8", d1 + d2[:50] + make_complex_seq(30, seed=9405) + d2[50:])
    gb_a7b.add_node("9", (d1 + d2)[30:])
    ref_4 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb_a7b.add_p_path("Ref#1#Chr01", ref_4)
    gb_a7b.add_p_path("Alt1#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb_a7b.add_p_path("Alt50Del1Ins#1#Chr01", [("1", "+"), ("6", "+"), ("4", "+")])
    gb_a7b.add_p_path("AltDSnp#1#Chr01", [("1", "+"), ("7", "+"), ("4", "+")])
    gb_a7b.add_p_path("AltD30I#1#Chr01", [("1", "+"), ("8", "+"), ("4", "+")])
    gb_a7b.add_p_path("AltD30DFlank#1#Chr01", [("1", "+"), ("9", "+"), ("4", "+")])
    _, _, res_a7b = _run_pipeline_from_builder(tmp_path, "w1_a7b", gb_a7b, "Ref#1#Chr01", ref_4)
    ins_a7b = [e for e in res_a7b.sv_events if e.event_type == "INS"][0]
    del_a7b = [e for e in res_a7b.sv_events if e.event_type == "DEL"][0]
    assert del_a7b.genotypes["Alt50Del1Ins#1"] == "1"
    assert ins_a7b.genotypes["Alt50Del1Ins#1"] == "."
    assert ins_a7b.genotypes["AltDSnp#1"] == "0"
    assert ins_a7b.genotypes["AltD30I#1"] == "0"
    assert ins_a7b.genotypes["AltD30DFlank#1"] == "0"

    # Positive control: internal 30bp DEL inside D (230bp -> 200bp) -> '0'
    d230 = make_complex_seq(230, seed=9409)
    gb_d30 = SyntheticGFAFixtureBuilder()
    gb_d30.add_node("1", make_complex_seq(150, seed=9406))
    gb_d30.add_node("2", d230)
    gb_d30.add_node("3", make_complex_seq(150, seed=9407))
    gb_d30.add_node("4", make_complex_seq(150, seed=9408))
    gb_d30.add_node("5", d230[:100] + d230[130:])
    ref_d30 = [("1", "+"), ("2", "+"), ("3", "+")]
    gb_d30.add_p_path("Ref#1#Chr01", ref_d30)
    gb_d30.add_p_path("Alt1#1#Chr01", [("1", "+"), ("4", "+"), ("3", "+")])
    gb_d30.add_p_path("AltD30DInt#1#Chr01", [("1", "+"), ("5", "+"), ("3", "+")])
    _, _, res_d30 = _run_pipeline_from_builder(tmp_path, "w1_d30int", gb_d30, "Ref#1#Chr01", ref_d30)
    ins_d30 = [e for e in res_d30.sv_events if e.event_type == "INS"][0]
    assert ins_d30.genotypes["AltD30DInt#1"] == "0"

    # 2. Probe A-7: 80bp DEL D1 + 40bp novel tail on D2 -> '.'
    gb_a7 = SyntheticGFAFixtureBuilder()
    d2_a7 = make_complex_seq(100, seed=9412)
    gb_a7.add_node("1", make_complex_seq(150, seed=9410))
    gb_a7.add_node("2", make_complex_seq(80, seed=9411))
    gb_a7.add_node("3", d2_a7)
    gb_a7.add_node("4", make_complex_seq(150, seed=9413))
    gb_a7.add_node("5", make_complex_seq(150, seed=9414))
    gb_a7.add_node("6", d2_a7 + make_complex_seq(40, seed=9415))
    gb_a7.add_p_path("Ref#1#Chr01", ref_4)
    gb_a7.add_p_path("Alt1#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb_a7.add_p_path("AltEvil7#1#Chr01", [("1", "+"), ("6", "+"), ("4", "+")])
    _, _, res_a7 = _run_pipeline_from_builder(tmp_path, "w1_a7", gb_a7, "Ref#1#Chr01", ref_4)
    assert [e for e in res_a7.sv_events if e.event_type == "INS"][0].genotypes["AltEvil7#1"] == "."

    # 3. Probe A-1: 100bp DEL D1 + cycle-folded 2x D2_snp (max_copy=1) -> '.'
    gb_a1 = SyntheticGFAFixtureBuilder()
    d2_a1 = make_complex_seq(100, seed=9422)
    gb_a1.add_node("1", make_complex_seq(150, seed=9420))
    gb_a1.add_node("2", make_complex_seq(100, seed=9421))
    gb_a1.add_node("3", d2_a1)
    gb_a1.add_node("4", make_complex_seq(150, seed=9423))
    gb_a1.add_node("5", make_complex_seq(150, seed=9424))
    gb_a1.add_node("6", mutate_seq_snps(d2_a1, [50]))
    gb_a1.add_p_path("Ref#1#Chr01", ref_4)
    gb_a1.add_p_path("Alt1#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb_a1.add_p_path("AltEvil1#1#Chr01", [("1", "+"), ("6", "+"), ("6", "+"), ("4", "+")])
    _, _, res_a1 = _run_pipeline_from_builder(
        tmp_path, "w1_a1", gb_a1, "Ref#1#Chr01", ref_4, max_copy=1
    )
    assert [e for e in res_a1.sv_events if e.event_type == "INS"][0].genotypes["AltEvil1#1"] == "."

    # 4. Probe A-3: 100bp DEL D1 + R_dup + D2_snp + R_dup + D2_snp -> '.'
    gb_a3 = SyntheticGFAFixtureBuilder()
    d2_a3 = make_complex_seq(100, seed=9433)
    gb_a3.add_node("1", make_complex_seq(150, seed=9430))
    gb_a3.add_node("2", make_complex_seq(100, seed=9431))
    gb_a3.add_node("3", make_complex_seq(100, seed=9432))
    gb_a3.add_node("4", d2_a3)
    gb_a3.add_node("5", make_complex_seq(150, seed=9434))
    gb_a3.add_node("6", make_complex_seq(150, seed=9435))
    gb_a3.add_node("7", mutate_seq_snps(d2_a3, [50]))
    ref_a3 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("3", "+"), ("5", "+")]
    gb_a3.add_p_path("Ref#1#Chr01", ref_a3)
    gb_a3.add_p_path("Alt1#1#Chr01", [("1", "+"), ("6", "+"), ("5", "+")])
    gb_a3.add_p_path(
        "AltEvil3#1#Chr01",
        [("1", "+"), ("3", "+"), ("7", "+"), ("3", "+"), ("7", "+"), ("5", "+")],
    )
    _, _, res_a3 = _run_pipeline_from_builder(tmp_path, "w1_a3", gb_a3, "Ref#1#Chr01", ref_a3)
    assert [e for e in res_a3.sv_events if e.event_type == "INS"][0].genotypes["AltEvil3#1"] == "."

    # 5. Challenger 1 P2.7b: pure unordered swap Ref=A-D1-D2-D2-B, AltSwapRep=A-D2-D2-D1_snp-B -> '.'
    gb_sw = SyntheticGFAFixtureBuilder()
    d1_sw = make_complex_seq(100, seed=9441)
    gb_sw.add_node("1", make_complex_seq(150, seed=9440))
    gb_sw.add_node("2", d1_sw)
    gb_sw.add_node("3", make_complex_seq(100, seed=9442))
    gb_sw.add_node("4", make_complex_seq(150, seed=9443))
    gb_sw.add_node("5", make_complex_seq(150, seed=9444))
    gb_sw.add_node("6", mutate_seq_snps(d1_sw, [50]))
    ref_sw = [("1", "+"), ("2", "+"), ("3", "+"), ("3", "+"), ("4", "+")]
    gb_sw.add_p_path("Ref#1#Chr01", ref_sw)
    gb_sw.add_p_path("AltIns#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb_sw.add_p_path(
        "AltSwapRep#1#Chr01",
        [("1", "+"), ("3", "+"), ("3", "+"), ("6", "+"), ("4", "+")],
    )
    _, _, res_sw = _run_pipeline_from_builder(tmp_path, "w1_p27b", gb_sw, "Ref#1#Chr01", ref_sw)
    assert [e for e in res_sw.sv_events if e.event_type == "INS"][0].genotypes["AltSwapRep#1"] == "."

    # 6. Challenger 1 NEW Bug 4: Stage 4 dropped <50bp novel insertions (compact & string IDs)
    for use_int in (True, False):
        gb_b4 = SyntheticGFAFixtureBuilder()
        id_fn = (lambda i: str(i)) if use_int else (lambda i: f"n{i}")
        for idx_n, nlen in [
            (1, 150),
            (2, 100),
            (3, 100),
            (4, 150),
            (5, 150),
            (6, 25),
            (7, 25),
            (8, 49),
            (9, 49),
            (10, 24),
            (11, 24),
        ]:
            gb_b4.add_node(id_fn(idx_n), make_complex_seq(nlen, seed=9800 + idx_n))
        ref_b4 = [(id_fn(i), "+") for i in (1, 2, 3, 4)]
        gb_b4.add_p_path("Ref#1#Chr01", ref_b4)
        gb_b4.add_p_path("AltIns#1#Chr01", [(id_fn(i), "+") for i in (1, 5, 4)])
        gb_b4.add_p_path("Alt2x25Ins#1#Chr01", [(id_fn(i), "+") for i in (1, 6, 2, 7, 3, 4)])
        gb_b4.add_p_path("Alt2x49Ins#1#Chr01", [(id_fn(i), "+") for i in (1, 8, 2, 9, 3, 4)])
        gb_b4.add_p_path("Alt2x24Ins#1#Chr01", [(id_fn(i), "+") for i in (1, 10, 2, 11, 3, 4)])
        _, _, res_b4 = _run_pipeline_from_builder(
            tmp_path, f"w1_b4_{use_int}", gb_b4, "Ref#1#Chr01", ref_b4
        )
        gt_b4 = [e for e in res_b4.sv_events if e.event_type == "INS"][0].genotypes
        assert gt_b4["Alt2x25Ins#1"] == "."
        assert gt_b4["Alt2x49Ins#1"] == "."
        assert gt_b4["Alt2x24Ins#1"] == "0"


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf2_wave1_inv_conflict_gap_skipping_node_calls_dot(
    tmp_path: Path,
    use_compact_ids: bool,
) -> None:
    """Wave 1 Item 2.2 (Challenger 1 NEW Bug 5): contig skipping B(1) inside an unnormalized INV_CONFLICT gap (0, 5) calls '.' on INV[1, 1]."""
    id_fn = (lambda i: str(i)) if use_compact_ids else (lambda i: f"n{i}")
    gb = SyntheticGFAFixtureBuilder()
    for i, length in [(1, 500), (2, 200), (3, 150), (4, 150), (5, 150), (6, 500)]:
        gb.add_node(id_fn(i), make_complex_seq(length, seed=9200 + i))
    ref_s = [(id_fn(i), "+") for i in range(1, 7)]
    gb.add_p_path("Ref#1#Chr01", ref_s)
    gb.add_p_path(
        "AltInvB#1#Chr01",
        [(id_fn(1), "+"), (id_fn(2), "-"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+"), (id_fn(6), "+")],
    )
    gb.add_p_path(
        "AltConfSkipB#1#Chr01",
        [(id_fn(1), "+"), (id_fn(4), "-"), (id_fn(3), "-"), (id_fn(5), "-"), (id_fn(4), "-"), (id_fn(6), "+")],
    )
    gb.add_p_path("AltRef#1#Chr01", ref_s)
    _, _, res = _run_pipeline_from_builder(
        tmp_path, f"w1_inv_conf_gap_{use_compact_ids}", gb, "Ref#1#Chr01", ref_s
    )
    inv_evs = [e for e in res.sv_events if e.event_type == "INV"]
    assert len(inv_evs) == 1
    assert inv_evs[0].genotypes["AltInvB#1"] == "1"
    assert inv_evs[0].genotypes["AltConfSkipB#1"] == "."
    assert inv_evs[0].genotypes["AltRef#1"] == "0"


def test_rf3_wave1_symmetric_affine_collapse_and_anchored_shifted_c1() -> None:
    """Wave 1 Items 3.1 & 3.2: symmetric affine I/D single-gap collapse + anchored sec_max_ed + shifted forward/RC hit detection."""
    # Item 3.1 (R2 F1 + C1 NEW Bug 3): 30I+1SNP, 20I+1SNP, 25/30/40/49D+1SNP, 4x4bp pure ins -> True
    for dist in (3, 5, 10, 15):
        for sd in range(10):
            r = make_complex_seq(400, seed=6000 + sd)
            r_mut = mutate_seq_snps(r, [200 + dist])
            s = r_mut[:200] + make_complex_seq(30, seed=7000 + sd) + r_mut[200:]
            assert check_similarity_merge(s, r) is True
    for sd in range(10):
        r = make_complex_seq(400, seed=6100 + sd)
        r_mut = mutate_seq_snps(r, [205])
        s = r_mut[:200] + make_complex_seq(20, seed=7100 + sd) + r_mut[200:]
        assert check_similarity_merge(s, r) is True
    for dl in (25, 30, 40, 49):
        for dist in (3, 5, 10, 15):
            for sd in range(10):
                r500 = make_complex_seq(500, seed=6200 + sd)
                s_del = r500[:200] + mutate_seq_snps(r500[200 + dl :], [dist])
                assert check_similarity_merge(s_del, r500) is True
    for sd in range(10):
        r4 = make_complex_seq(400, seed=6300 + sd)
        s_4x4 = (
            r4[:200]
            + make_complex_seq(4, seed=7300 + sd)
            + r4[200:210]
            + make_complex_seq(4, seed=7400 + sd)
            + r4[210:220]
            + make_complex_seq(4, seed=7500 + sd)
            + r4[220:230]
            + make_complex_seq(4, seed=7600 + sd)
            + r4[230:]
        )
        assert check_similarity_merge(s_4x4, r4) is True

    # Asymmetric replacement blocks must remain False
    for sd in range(15):
        r_a = make_complex_seq(400, seed=6500 + sd)
        mut5 = mutate_seq_snps(r_a, [202, 205, 208, 211, 214])
        assert (
            check_similarity_merge(
                mut5[:216] + make_complex_seq(30, seed=7800 + sd) + mut5[216:], r_a
            )
            is False
        )
        assert (
            check_similarity_merge(
                r_a[:190] + make_complex_seq(50, seed=7900 + sd) + r_a[210:], r_a
            )
            is False
        )
        assert (
            check_similarity_merge(
                r_a[:192] + make_complex_seq(45, seed=7925 + sd) + r_a[207:], r_a
            )
            is False
        )
        r_b = make_complex_seq(400, seed=6700 + sd)
        assert (
            check_similarity_merge(
                r_b[:190] + make_complex_seq(45, seed=7950 + sd) + r_b[215:], r_b
            )
            is False
        )
        assert (
            check_similarity_merge(
                r_b[:185] + make_complex_seq(50, seed=7975 + sd) + r_b[215:], r_b
            )
            is False
        )

    # Item 3.2 (R2 F3): overlapping shifted forward secondary hits -> False
    u100 = make_complex_seq(100, seed=3002)
    assert check_similarity_merge(u100 + u100, u100 + u100 + mutate_seq_snps(u100, [50])) is False
    assert check_similarity_merge(u100 + u100, mutate_seq_snps(u100, [20, 70]) + u100 + u100) is False

    # Item 3.2 (C1 NEW Bug 1): short query with fwd_ed > int(0.20 * len_s) -> 1b, 1c-tie, 1c-better-RC all False
    for u_len, ov_len in [(75, 25), (100, 30), (150, 45), (170, 48)]:
        u = make_complex_seq(u_len, seed=5100 + u_len)
        u_1snp = mutate_seq_snps(u, [10])
        ov = make_complex_seq(ov_len, seed=5200 + ov_len)
        for s_cand in (u[: u_len // 2] + ov + u[u_len // 2 :], ov + u):
            assert check_similarity_merge(s_cand, u + u_1snp) is False
            assert check_similarity_merge(s_cand, u + revcomp_dna(u)) is False
            assert check_similarity_merge(s_cand, revcomp_dna(u) + u_1snp) is False

    # Item 3.2 (C1 NEW Bug 2): pure Criterion 1c overlapping shifted RC hit (zero 1b hit) -> False
    half30 = make_complex_seq(30, seed=5501)
    p60 = half30 + revcomp_dna(half30)
    half60 = make_complex_seq(60, seed=5502)
    q120 = mutate_seq_snps(half60 + revcomp_dna(half60), [i * 5 for i in range(12)])
    rc_q120 = revcomp_dna(q120)
    same_pos = [i for i in range(120) if q120[i] == rc_q120[i]]
    q_rc_mut = mutate_seq_snps(rc_q120, same_pos[:50])
    assert check_similarity_merge(p60 + q120 + p60, p60 + q120 + p60 + q_rc_mut + p60) is False

    # Short-query unique random-flank controls must NOT suffer false rejections
    for u_len, ov_len in [(60, 15), (80, 20), (100, 30), (130, 48), (170, 48), (200, 49)]:
        for sd in range(10):
            u = make_complex_seq(u_len, seed=7000 + sd)
            ov = make_complex_seq(ov_len, seed=8000 + sd)
            fl = make_complex_seq(400, seed=9000 + sd)
            s_ctrl = u[: u_len // 2] + ov + u[u_len // 2 :]
            assert check_similarity_merge(s_ctrl, u + fl) is True


def test_rf4_wave1_verify_spacer_covered_n_and_streaming_backbone_parity(
    tmp_path: Path,
) -> None:
    """Wave 1 Item 4.2 (R2 F2, C2 NEW Bugs 1 & 2): covered-'N' spacer tamper rejection (small & >10MB), streaming backbone_prefix palindromic parity, merged >10MB ndarray fallback, and >10MB W-line end='*'."""
    pad_10mb_seq = "ACGT" * 55_000  # 220,000 bp x 50 nodes = 11 MB

    # 1. C2 NEW Bug 1: spacer tamper after Ref N*100, Piece1 N*100, and mixed Ref N*50 + Piece1 N*50 (small & >10MB)
    for is_large in (False, True):
        # 1a. Ref ending in N*100 -> tamper source_contiguous -> spacer
        sub_a = tmp_path / f"w1_sp_ref_{is_large}"
        sub_a.mkdir()
        s1 = make_complex_seq(100, seed=8101) + "N" * 100
        s2 = make_complex_seq(200, seed=8102)
        s3 = make_complex_seq(180, seed=8103)
        gfa_a = sub_a / "t.gfa"
        fa_a = sub_a / "ref.fa"
        with gfa_a.open("w", encoding="utf-8") as fh:
            fh.write("H\tVN:Z:1.0\n")
            fh.write(f"S\t1\t{s1}\nS\t2\t{s2}\nS\t3\t{s3}\n")
            if is_large:
                for pid in range(4, 54):
                    fh.write(f"S\t{pid}\t{pad_10mb_seq}\n")
            fh.write("L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nL\t3\t+\t2\t+\t0M\n")
            fh.write("P\tRef#1#Chr01\t1+,2+\t*\nP\tAlt1#1#Chr01\t1+,3+,2+\t*\n")
        fa_a.write_text(f">Chr01\n{s1}{s2}\n", encoding="utf-8")
        out_a = sub_a / "out"
        assert cli_main(["build", "--gfa", str(gfa_a), "--ref-fa", str(fa_a), "--backbone-prefix", "Ref#1#Chr01", "--out-prefix", str(out_a)]) == 0
        rows_a = _read_tsv(Path(f"{out_a}.inserts.tsv"))
        assert len(rows_a) == 1 and rows_a[0]["join_before"] == "source_contiguous"
        rows_a[0]["join_before"] = "spacer"
        _rewrite_tsv(Path(f"{out_a}.inserts.tsv"), rows_a)
        rep_a = verify_build_outputs(out_a, gfa_a, fa_a)
        assert not rep_a.passed and any("[INV2]" in e for e in rep_a.errors)

        # 1b. Cycle-folded Piece1 ending in N*100 (or mixed Ref N*50 + Piece1 N*50) -> tamper Piece2 synthetic -> spacer
        for mix_tag, r1_seq, p1_seq in [
            ("p1n100", make_complex_seq(200, seed=8110), make_complex_seq(60, seed=8111) + "N" * 100),
            ("mix50", make_complex_seq(150, seed=8112) + "N" * 50, "N" * 50),
        ]:
            sub_b = tmp_path / f"w1_sp_{mix_tag}_{is_large}"
            sub_b.mkdir()
            r2_seq = make_complex_seq(200, seed=8113)
            p2_seq = make_complex_seq(160, seed=8114)
            gfa_b = sub_b / "t.gfa"
            fa_b = sub_b / "ref.fa"
            with gfa_b.open("w", encoding="utf-8") as fh:
                fh.write("H\tVN:Z:1.0\n")
                fh.write(f"S\t1\t{r1_seq}\nS\t2\t{r2_seq}\nS\t3\t{p1_seq}\nS\t4\t{p2_seq}\n")
                if is_large:
                    for pid in range(5, 55):
                        fh.write(f"S\t{pid}\t{pad_10mb_seq}\n")
                fh.write("L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nL\t3\t+\t3\t+\t0M\nL\t3\t+\t4\t+\t0M\nL\t4\t+\t2\t+\t0M\n")
                fh.write("P\tRef#1#Chr01\t1+,2+\t*\nP\tAlt1#1#Chr01\t1+,3+,3+,4+,2+\t*\n")
            fa_b.write_text(f">Chr01\n{r1_seq}{r2_seq}\n", encoding="utf-8")
            out_b = sub_b / "out"
            assert cli_main(["build", "--gfa", str(gfa_b), "--ref-fa", str(fa_b), "--backbone-prefix", "Ref#1#Chr01", "--out-prefix", str(out_b), "--max-copy", "1"]) == 0
            rows_b = _read_tsv(Path(f"{out_b}.inserts.tsv"))
            assert len(rows_b) == 2 and rows_b[1]["join_before"] == "synthetic"
            rows_b[1]["join_before"] = "spacer"
            _rewrite_tsv(Path(f"{out_b}.inserts.tsv"), rows_b)
            rep_b = verify_build_outputs(out_b, gfa_b, fa_b)
            assert not rep_b.passed and any("[INV2]" in e for e in rep_b.errors)

    # 2. C2 NEW Bug 2 + R2 Finding 2: small & >10MB palindromic ref step without ref_path in summary.tsv -> both reject [INV2] (zero ValueError!)
    half_p = make_complex_seq(100, seed=8201)
    pal_seq = half_p + revcomp_dna(half_p)
    for is_large in (False, True):
        sub_p = tmp_path / f"w1_pal_{is_large}"
        sub_p.mkdir()
        n1 = make_complex_seq(200, seed=8200)
        n3 = make_complex_seq(200, seed=8202)
        n4 = make_complex_seq(180, seed=8203)
        gfa_p = sub_p / "t.gfa"
        fa_p = sub_p / "ref.fa"
        with gfa_p.open("w", encoding="utf-8") as fh:
            fh.write("H\tVN:Z:1.0\n")
            fh.write(f"S\t1\t{n1}\nS\t2\t{pal_seq}\nS\t3\t{n3}\nS\t4\t{n4}\n")
            if is_large:
                for pid in range(5, 55):
                    fh.write(f"S\t{pid}\t{pad_10mb_seq}\n")
            fh.write("L\t1\t+\t2\t+\t0M\nL\t2\t+\t3\t+\t0M\nL\t1\t+\t2\t-\t0M\nL\t2\t-\t4\t+\t0M\nL\t4\t+\t3\t+\t0M\n")
            fh.write("P\tRef#1#Chr01\t1+,2+,3+\t*\nP\tAlt1#1#Chr01\t1+,2-,4+,3+\t*\n")
        fa_p.write_text(f">Chr01\n{n1}{pal_seq}{n3}\n", encoding="utf-8")
        out_p = sub_p / "out"
        assert cli_main(["build", "--gfa", str(gfa_p), "--ref-fa", str(fa_p), "--backbone-prefix", "Ref#1#Chr01", "--out-prefix", str(out_p)]) == 0
        rows_p = _read_tsv(Path(f"{out_p}.inserts.tsv"))
        assert len(rows_p) == 1 and rows_p[0]["join_before"] == "synthetic"
        rows_p[0]["join_before"] = "source_contiguous"
        _rewrite_tsv(Path(f"{out_p}.inserts.tsv"), rows_p)
        sum_p = Path(f"{out_p}.summary.tsv")
        sum_p.write_text(
            "\n".join(ln for ln in sum_p.read_text(encoding="utf-8").splitlines() if not ln.startswith("ref_path\t")) + "\n",
            encoding="utf-8",
        )
        rep_p = verify_build_outputs(out_p, gfa_p, fa_p, backbone_prefix="Ref#1#Chr01")
        assert not rep_p.passed and any("[INV2]" in e for e in rep_p.errors)

    # 3. R2 Finding 2 + >10MB W-line end == '*': 2-chrom merged >10MB verify reaching eff_backbone_prefix fallback & W-line end='*'
    sub_m = tmp_path / "w1_merge_10mb"
    sub_m.mkdir()
    c1_1, c1_2, c1_x = make_complex_seq(200, seed=8301), make_complex_seq(200, seed=8302), make_complex_seq(180, seed=8303)
    c2_1, c2_2, c2_x = make_complex_seq(200, seed=8304), make_complex_seq(200, seed=8305), make_complex_seq(180, seed=8306)
    gfa_c1 = sub_m / "c1.gfa"
    gfa_c2 = sub_m / "c2.gfa"
    gfa_all = sub_m / "all.gfa"
    fa_c1 = sub_m / "c1.fa"
    fa_c2 = sub_m / "c2.fa"
    fa_all = sub_m / "all.fa"
    gfa_c1.write_text(
        f"H\tVN:Z:1.0\nS\t1\t{c1_1}\nS\t2\t{c1_2}\nS\t3\t{c1_x}\n"
        f"L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nL\t3\t+\t2\t+\t0M\n"
        f"P\tRef#1#Chr01\t1+,2+\t*\nW\tAltW\t1\tctgW\t10000\t*\t<2<3<1\n",
        encoding="utf-8",
    )
    gfa_c2.write_text(
        f"H\tVN:Z:1.0\nS\t4\t{c2_1}\nS\t5\t{c2_2}\nS\t6\t{c2_x}\n"
        f"L\t4\t+\t5\t+\t0M\nL\t4\t+\t6\t+\t0M\nL\t6\t+\t5\t+\t0M\n"
        f"P\tRef#1#Chr02\t4+,5+\t*\nP\tAlt1#1#Chr02\t4+,6+,5+\t*\n",
        encoding="utf-8",
    )
    with gfa_all.open("w", encoding="utf-8") as fh:
        fh.write("H\tVN:Z:1.0\n")
        fh.write(f"S\t1\t{c1_1}\nS\t2\t{c1_2}\nS\t3\t{c1_x}\nS\t4\t{c2_1}\nS\t5\t{c2_2}\nS\t6\t{c2_x}\n")
        for pid in range(7, 57):
            fh.write(f"S\t{pid}\t{pad_10mb_seq}\n")
        fh.write("L\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nL\t3\t+\t2\t+\t0M\n")
        fh.write("L\t4\t+\t5\t+\t0M\nL\t4\t+\t6\t+\t0M\nL\t6\t+\t5\t+\t0M\n")
        fh.write("P\tRef#1#Chr01\t1+,2+\t*\nW\tAltW\t1\tctgW\t10000\t*\t<2<3<1\n")
        fh.write("P\tRef#1#Chr02\t4+,5+\t*\nP\tAlt1#1#Chr02\t4+,6+,5+\t*\n")
    assert gfa_all.stat().st_size > 10_000_000
    fa_c1.write_text(f">Chr01\n{c1_1}{c1_2}\n", encoding="utf-8")
    fa_c2.write_text(f">Chr02\n{c2_1}{c2_2}\n", encoding="utf-8")
    fa_all.write_text(f">Chr01\n{c1_1}{c1_2}\n>Chr02\n{c2_1}{c2_2}\n", encoding="utf-8")
    out_c1, out_c2, out_m = sub_m / "out_c1", sub_m / "out_c2", sub_m / "out_merged"
    assert cli_main(["build", "--gfa", str(gfa_c1), "--ref-fa", str(fa_c1), "--backbone-prefix", "Ref#1#", "--out-prefix", str(out_c1)]) == 0
    assert cli_main(["build", "--gfa", str(gfa_c2), "--ref-fa", str(fa_c2), "--backbone-prefix", "Ref#1#", "--out-prefix", str(out_c2)]) == 0
    plist = sub_m / "prefixes.txt"
    plist.write_text(f"{out_c1}\n{out_c2}\n", encoding="utf-8")
    assert cli_main(["merge", "--prefix-list", str(plist), "--out-prefix", str(out_m)]) == 0
    # Strip per-chrom ref_path lines from merged summary.tsv to force the eff_backbone_prefix fallback on np.ndarray!
    sum_m = Path(f"{out_m}.summary.tsv")
    sum_m.write_text(
        "\n".join(ln for ln in sum_m.read_text(encoding="utf-8").splitlines() if ":ref_path\t" not in ln and not ln.startswith("ref_path\t")) + "\n",
        encoding="utf-8",
    )
    rep_m = verify_build_outputs(out_m, gfa_all, fa_all, backbone_prefix="Ref#1#")
    assert rep_m.passed, f"Expected merged >10MB verify (W-line end=* + eff_backbone_prefix fallback) to pass, got {rep_m.errors}"


# ---------------------------------------------------------------------------
# Wave 2 Consolidated Regression Tests (All 8 Surgical Fixes)
# ---------------------------------------------------------------------------


def test_rf2_wave2_case_a_synth_rep0_spans_and_ins_excess_accounting(
    tmp_path: Path,
) -> None:
    """Wave 2 R1-r2 Findings 1 & 2: synthetic sc.seq==rep0_seq aligns per occurrence (G2.1/G2.1d/G2.1c) and internal_del_bp never cancels ins_excess_bp or duplicated overlap (G2.2/G2.2c/G2.2d)."""
    # 1. R1-r2 Finding 1 (G2.1 compact & G2.1d string): synth comp sc.seq == rep0_seq with 100 bp DEL member
    for compact in (True, False):
        id_fn = (lambda i: str(i)) if compact else (lambda i: f"n{i}")
        d1, d2 = make_complex_seq(100, seed=1101), make_complex_seq(100, seed=1102)
        gb = SyntheticGFAFixtureBuilder()
        gb.add_node(id_fn(1), make_complex_seq(150, seed=1100))
        gb.add_node(id_fn(2), d1)
        gb.add_node(id_fn(3), d2)
        gb.add_node(id_fn(4), make_complex_seq(150, seed=1103))
        gb.add_node(id_fn(5), make_complex_seq(150, seed=1104))
        gb.add_node(id_fn(6), d1)
        gb.add_node(id_fn(7), d2)
        ref = [(id_fn(i), "+") for i in (1, 2, 3, 4)]
        gb.add_p_path("Ref#1#Chr01", ref)
        gb.add_p_path("AltIns#1#Chr01", [(id_fn(1), "+"), (id_fn(5), "+"), (id_fn(4), "+")])
        gb.add_p_path(
            "AltBsec#1#Chr01",
            [(id_fn(1), "+"), (id_fn(6), "+"), (id_fn(7), "+"), (id_fn(4), "+")],
        )
        gb.add_p_path("AltDelN1#1#Chr01", [(id_fn(1), "+"), (id_fn(7), "+"), (id_fn(4), "+")])
        _, _, res = _run_pipeline_from_builder(
            tmp_path, f"w2_g21_{compact}", gb, "Ref#1#Chr01", ref
        )
        gt = [e for e in res.sv_events if e.event_type == "INS"][0].genotypes
        assert gt["AltDelN1#1"] == "."
        assert gt["AltBsec#1"] == "0"
        assert gt["AltIns#1"] == "1"

    # 2. R1-r2 Finding 1c (G2.1c): active-child synth comp sc.seq == rep0_seq where b_sec carries 100 bp DEL
    d1c, d2c = make_complex_seq(100, seed=1111), make_complex_seq(100, seed=1112)
    gb1c = SyntheticGFAFixtureBuilder()
    gb1c.add_node("1", make_complex_seq(150, seed=1110))
    gb1c.add_node("2", d1c)
    gb1c.add_node("3", d2c)
    gb1c.add_node("4", make_complex_seq(150, seed=1113))
    gb1c.add_node("5", make_complex_seq(150, seed=1114))
    gb1c.add_node("6", d1c)
    gb1c.add_node("7", d2c)
    ref1c = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb1c.add_p_path("Ref#1#Chr01", ref1c)
    gb1c.add_p_path("AltIns#1#Chr01", [("1", "+"), ("5", "+"), ("4", "+")])
    gb1c.add_p_path("AltDelD2_a#1#Chr01", [("1", "+"), ("6", "+"), ("4", "+")])
    gb1c.add_p_path("AltDelD2_b#1#Chr01", [("1", "+"), ("6", "+"), ("4", "+")])
    gb1c.add_p_path("AltFull#1#Chr01", [("1", "+"), ("6", "+"), ("7", "+"), ("4", "+")])
    _, _, res1c = _run_pipeline_from_builder(
        tmp_path, "w2_g21c", gb1c, "Ref#1#Chr01", ref1c
    )
    gt1c = [e for e in res1c.sv_events if e.event_type == "INS"][0].genotypes
    assert gt1c["AltDelD2_a#1"] == "."
    assert gt1c["AltDelD2_b#1"] == "."
    assert gt1c["AltFull#1"] == "0"

    # 3. R1-r2 Finding 2 (G2.2 & G2.2c): 2x(20D+40I)=80I and dropped 25bp + (20D+35I)=60I -> "."; single (20D+35I) -> "0"
    b1, rd, b2 = (
        make_complex_seq(250, seed=1201),
        make_complex_seq(40, seed=1202),
        make_complex_seq(250, seed=1203),
    )
    n1 = b1[:60] + b1[80:190] + make_complex_seq(40, seed=1204) + b1[190:]
    n2 = b2[:60] + b2[80:190] + make_complex_seq(40, seed=1205) + b2[190:]
    n2_35 = b2[:60] + b2[80:190] + make_complex_seq(35, seed=1209) + b2[190:]
    gb2 = SyntheticGFAFixtureBuilder()
    gb2.add_node("1", make_complex_seq(150, seed=1200))
    gb2.add_node("2", b1)
    gb2.add_node("3", rd)
    gb2.add_node("4", b2)
    gb2.add_node("5", make_complex_seq(150, seed=1206))
    gb2.add_node("6", make_complex_seq(150, seed=1207))
    gb2.add_node("7", n1)
    gb2.add_node("8", n2)
    gb2.add_node("9", make_complex_seq(25, seed=1208))
    gb2.add_node("10", b1)
    gb2.add_node("11", n2_35)
    ref2 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("3", "+"), ("5", "+")]
    gb2.add_p_path("Ref#1#Chr01", ref2)
    gb2.add_p_path("AltIns#1#Chr01", [("1", "+"), ("6", "+"), ("5", "+")])
    gb2.add_p_path(
        "AltCanc80I40D#1#Chr01",
        [("1", "+"), ("7", "+"), ("3", "+"), ("8", "+"), ("3", "+"), ("5", "+")],
    )
    gb2.add_p_path(
        "AltDrop25And20D35I#1#Chr01",
        [("1", "+"), ("10", "+"), ("3", "+"), ("11", "+"), ("3", "+"), ("9", "+"), ("5", "+")],
    )
    gb2.add_p_path(
        "AltSingle20D35I#1#Chr01",
        [("1", "+"), ("10", "+"), ("3", "+"), ("11", "+"), ("3", "+"), ("5", "+")],
    )
    _, _, res2 = _run_pipeline_from_builder(
        tmp_path, "w2_g22_abc", gb2, "Ref#1#Chr01", ref2
    )
    gt2 = [e for e in res2.sv_events if e.event_type == "INS"][0].genotypes
    assert gt2["AltCanc80I40D#1"] == "."
    assert gt2["AltDrop25And20D35I#1"] == "."
    assert gt2["AltSingle20D35I#1"] == "0"

    # 4. R1-r2 Finding 2d (G2.2d): 70 bp duplicated backbone overlap + 30 bp internal DEL -> "."
    sa, sb, sc = (
        make_complex_seq(200, seed=1230),
        make_complex_seq(20, seed=1231),
        make_complex_seq(200, seed=1232),
    )
    full_b = sa + sb + sc
    gb2d = SyntheticGFAFixtureBuilder()
    gb2d.add_node("1", make_complex_seq(150, seed=1233))
    gb2d.add_node("2", sa)
    gb2d.add_node("3", sb)
    gb2d.add_node("4", sc)
    gb2d.add_node("5", make_complex_seq(150, seed=1234))
    gb2d.add_node("6", make_complex_seq(150, seed=1235))
    gb2d.add_node("7", full_b[:220])
    gb2d.add_node("8", full_b[150:280] + full_b[310:420])
    ref2d = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("3", "+"), ("5", "+")]
    gb2d.add_p_path("Ref#1#Chr01", ref2d)
    gb2d.add_p_path("AltIns#1#Chr01", [("1", "+"), ("6", "+"), ("5", "+")])
    gb2d.add_p_path(
        "AltOv70Del30#1#Chr01",
        [("1", "+"), ("7", "+"), ("3", "+"), ("8", "+"), ("3", "+"), ("5", "+")],
    )
    _, _, res2d = _run_pipeline_from_builder(
        tmp_path, "w2_g22d", gb2d, "Ref#1#Chr01", ref2d
    )
    assert [e for e in res2d.sv_events if e.event_type == "INS"][0].genotypes[
        "AltOv70Del30#1"
    ] == "."


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf2_wave2_inv_unnorm_displaced_anchor_preserves_forward_calls(
    tmp_path: Path,
    use_compact_ids: bool,
) -> None:
    """Wave 2 C1-r2 Bug 1: displaced non-main_chain '+' anchor near INV_EDGE/INV_CONFLICT must NOT poison unbroken forward regions across INV/INS/DEL/CNV."""
    id_fn = (lambda i: str(i)) if use_compact_ids else (lambda i: f"n{i}")
    gb = SyntheticGFAFixtureBuilder()
    for i, length in enumerate([500, 60, 200, 200, 200, 200, 200, 200, 200, 200], start=1):
        gb.add_node(id_fn(i), make_complex_seq(length, seed=8200 + i))
    gb.add_node(id_fn(11), make_complex_seq(150, seed=8211))
    ref_s = [(id_fn(i), "+") for i in range(1, 11)]
    gb.add_p_path("Ref#1#Chr01", ref_s)
    gb.add_p_path("AltInv4#1#Chr01", [(id_fn(i), "-" if i == 5 else "+") for i in range(1, 11)])
    gb.add_p_path("AltInv7#1#Chr01", [(id_fn(i), "-" if i == 8 else "+") for i in range(1, 11)])
    gb.add_p_path(
        "AltIns45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)]
        + [(id_fn(11), "+")]
        + [(id_fn(i), "+") for i in range(6, 11)],
    )
    gb.add_p_path("AltDel4#1#Chr01", [(id_fn(i), "+") for i in range(1, 11) if i != 5])
    gb.add_p_path(
        "AltEdgeDispLeft#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 10)] + [(id_fn(2), "+"), (id_fn(10), "-")],
    )
    gb.add_p_path(
        "AltLeftEdgeDisp#1#Chr01",
        [(id_fn(1), "-"), (id_fn(9), "+")] + [(id_fn(i), "+") for i in range(2, 11)],
    )
    gb.add_p_path(
        "AltConfDispRight#1#Chr01",
        [
            (id_fn(1), "+"),
            (id_fn(4), "-"),
            (id_fn(3), "-"),
            (id_fn(5), "-"),
            (id_fn(4), "-"),
            (id_fn(10), "+"),
        ]
        + [(id_fn(i), "+") for i in range(6, 11)],
    )
    _, _, res = _run_pipeline_from_builder(
        tmp_path, f"w2_b1_{use_compact_ids}", gb, "Ref#1#Chr01", ref_s
    )
    invs = {e.anchors: e.genotypes for e in res.sv_events if e.event_type == "INV"}
    ins45 = [e for e in res.sv_events if e.event_type == "INS"][0]
    del4 = [e for e in res.sv_events if e.event_type == "DEL" and e.anchors == "3..5"][0]
    for hap in ("AltEdgeDispLeft#1", "AltLeftEdgeDisp#1"):
        assert invs["3..5"][hap] == "0"
        assert ins45.genotypes[hap] == "0"
        assert del4.genotypes[hap] == "0"
    assert invs["6..8"]["AltConfDispRight#1"] == "0"
    assert invs["3..5"]["AltConfDispRight#1"] == "."

    # CNV preserved as "2" on unbroken region before right INV_EDGE + displaced 2+
    gb_cnv = SyntheticGFAFixtureBuilder()
    for i, length in enumerate([500, 60, 200, 200, 200, 200, 200, 200], start=1):
        gb_cnv.add_node(id_fn(i), make_complex_seq(length, seed=8300 + i))
    ref_cnv = [(id_fn(i), "+") for i in (1, 2, 3, 4, 5, 5, 6, 7, 8)]
    gb_cnv.add_p_path("Ref#1#Chr01", ref_cnv)
    gb_cnv.add_p_path("AltCnv3#1#Chr01", [(id_fn(i), "+") for i in (1, 2, 3, 4, 5, 5, 5, 6, 7, 8)])
    gb_cnv.add_p_path(
        "AltEdgeDispCnv#1#Chr01",
        [(id_fn(i), "+") for i in (1, 2, 3, 4, 5, 5, 6, 7, 2)] + [(id_fn(8), "-")],
    )
    _, _, res_cnv = _run_pipeline_from_builder(
        tmp_path, f"w2_b1_cnv_{use_compact_ids}", gb_cnv, "Ref#1#Chr01", ref_cnv
    )
    assert [e for e in res_cnv.sv_events if e.event_type == "CNV"][0].genotypes[
        "AltEdgeDispCnv#1"
    ] == "2"


def test_rf3_wave2_core_fallback_shifted_palindrome_affine_dp_and_overhang_trim() -> None:
    """Wave 2 repeat_guard.py fixes (R2-r2 F1 + len_s=125..159, C1-r2 Bugs 2, 3, 4, 5)."""
    # 1. R2-r2 F1 + C1-r2 ext (len_s = 60..159): aligned-core fallback catches 1b/1c duplicates without FPs on unique flanks
    for u_len, ov_len in [(40, 20), (50, 29), (60, 40), (70, 49), (75, 49)]:
        for sd in range(10):
            u = make_complex_seq(u_len, seed=21000 + u_len * 100 + sd)
            u1 = mutate_seq_snps(u, [u_len // 3])
            ov = make_complex_seq(ov_len, seed=22000 + ov_len * 100 + sd)
            assert check_similarity_merge(ov + u, u + revcomp_dna(u1)) is False
            assert check_similarity_merge(u[:8] + ov + u[8:], u + revcomp_dna(u1)) is False
            fl = make_complex_seq(120, seed=23000 + sd)
            assert check_similarity_merge(ov + u, u + fl) is True
            assert check_similarity_merge(u[:8] + ov + u[8:], u + fl) is True
    u50 = make_complex_seq(50, seed=20003)
    u50_1 = mutate_seq_snps(u50, [12])
    ov25 = make_complex_seq(25, seed=18503)
    sep = make_complex_seq(120, seed=17003)
    assert check_similarity_merge(u50[:25] + ov25 + u50[25:], u50 + sep + u50_1) is False
    assert check_similarity_merge(u50[:25] + ov25 + u50[25:], u50 + sep) is True
    for u_len, ov_len, k_snps in [(76, 49, 2), (76, 49, 3), (80, 49, 5), (90, 49, 8)]:
        for sd in range(5):
            u = make_complex_seq(u_len, seed=31000 + u_len * 100 + sd)
            uk = mutate_seq_snps(u, [i * (u_len // (k_snps + 1)) + 3 for i in range(k_snps)])
            ov = make_complex_seq(ov_len, seed=32000 + u_len * 100 + sd)
            assert check_similarity_merge(ov + u, u + revcomp_dna(uk)) is False
            assert check_similarity_merge(ov + u, u + make_complex_seq(200, seed=33000 + sd)) is True

    # 2. C1-r2 Bug 2: single unique locus with shifted near-palindromic RC hit (d=10..49 left & right) -> 0 FPs
    for d in (10, 15, 25, 30, 40, 49):
        for sd in range(5):
            fl1 = make_complex_seq(100, seed=41000 + d * 100 + sd)
            fl2 = make_complex_seq(100, seed=42000 + d * 100 + sd)
            ad = make_complex_seq(d, seed=43000 + d * 100 + sd)
            p_len = 200 - d
            h = make_complex_seq(p_len // 2, seed=44000 + d * 100 + sd)
            p_near = mutate_seq_snps(h + ("A" if p_len % 2 else "") + revcomp_dna(h), [5, 15, 25, 35])
            assert check_similarity_merge(ad + p_near, fl1 + ad + p_near + revcomp_dna(ad) + fl2) is True
            assert check_similarity_merge(p_near + ad, fl1 + revcomp_dna(ad) + p_near + ad + fl2) is True

    # 3. C1-r2 Bug 3: 0-SNP pure small multi-insertions (4x4, 4x5, 5x3, 5x4 bp) must merge (True)
    base_new3 = make_complex_seq(500, seed=11001)
    ins_new3 = [make_complex_seq(4, seed=99010 + j) for j in range(4)]
    s_new3 = (
        base_new3[:200]
        + ins_new3[0]
        + base_new3[200:203]
        + ins_new3[1]
        + base_new3[203:206]
        + ins_new3[2]
        + base_new3[206:209]
        + ins_new3[3]
        + base_new3[209:]
    )
    assert check_similarity_merge(s_new3, base_new3) is True
    for n_ins, ilen, sp in [(4, 4, 2), (4, 4, 3), (4, 4, 5), (4, 5, 3), (5, 3, 3)]:
        for sd in range(10):
            base = make_complex_seq(500, seed=11000 + sd)
            s_mi = "".join(
                [base[:200]]
                + [
                    make_complex_seq(ilen, seed=99000 + sd * 10 + j)
                    + (base[200 + j * sp : 200 + (j + 1) * sp] if j < n_ins - 1 else "")
                    for j in range(n_ins)
                ]
                + [base[200 + (n_ins - 1) * sp :]]
            )
            assert check_similarity_merge(s_mi, base) is True

    # 4. C1-r2 Bug 4: deletion-direction asymmetric replacement blocks (45->15..65->20) -> False in both edlib and mappy
    for sd in range(20):
        r500 = make_complex_seq(500, seed=26000 + sd)
        repl15 = make_complex_seq(15, seed=27000 + sd)
        assert check_similarity_merge(r500[:200] + repl15 + r500[260:], r500) is False
        assert check_similarity_merge(r500[:200] + repl15 + r500[245:], r500) is False
    for rlen, slen in [(50, 15), (55, 15), (60, 20), (65, 20)]:
        for sd in range(10):
            r500 = make_complex_seq(500, seed=26000 + sd)
            repl = make_complex_seq(slen, seed=27000 + sd)
            assert check_similarity_merge(r500[:200] + repl + r500[200 + rlen :], r500) is False
            r1200 = make_complex_seq(1200, seed=26000 + sd)
            assert check_similarity_merge(r1200[:500] + repl + r1200[500 + rlen :], r1200) is False

    # 5. C1-r2 Bug 5: pure terminal overhangs on flanked R + near-end pure deletions -> True; 50 bp overhang -> False
    for ov in (20, 30, 40, 49):
        for sd in range(15):
            r600 = make_complex_seq(600, seed=88000 + sd)
            assert check_similarity_merge(r600[100:500] + make_complex_seq(ov, seed=89000 + sd), r600) is True
    for sd in range(15):
        r600 = make_complex_seq(600, seed=88000 + sd)
        ov50 = make_complex_seq(50, seed=89000 + sd)
        if ov50[0] == r600[500]:
            ov50 = ("A" if r600[500] != "A" else "C") + ov50[1:]
        assert check_similarity_merge(r600[100:500] + ov50, r600) is False
    for arm, dl, body in [(15, 30, 80), (20, 35, 100)]:
        for sd in range(15):
            r_nd = make_complex_seq(arm + dl + body, seed=77000 + sd)
            assert check_similarity_merge(r_nd[:arm] + r_nd[arm + dl :], r_nd) is True
    # Terminal 20->50 (diff0) rejected; terminal 50->20 merges with exact trimmed target span [0, 400)
    u400 = make_complex_seq(400, seed=50000)
    v20 = make_complex_seq(20, seed=51000)
    w50 = make_complex_seq(50, seed=52000)
    if w50[0] == v20[0]:
        w50 = ("A" if v20[0] != "A" else "C") + w50[1:]
    assert check_similarity_merge(u400 + w50, u400 + v20) is False
    v50 = make_complex_seq(50, seed=53000)
    w20 = make_complex_seq(20, seed=54000)
    if w20[0] == v50[0]:
        w20 = ("A" if v50[0] != "A" else "C") + w20[1:]
    assert _check_similarity_merge_with_span(u400 + w20, u400 + v50) == (True, 0, 400, 0)


# ---------------------------------------------------------------------------
# Wave 3 / Iteration 4 Consolidated Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf2_wave3_adjacent_anchor_unnormalized_conflict_gap_genotyping(
    tmp_path: Path,
    use_compact_ids: bool,
) -> None:
    """R1-r3 F1: adjacent-anchor unnormalized INV_CONFLICT gaps must record unnormalized_gap_spans and emit '.'."""
    # 1. Direct _ContigAnalysis backward-compat, pickle round-trip, and exact containment boundary checks
    ca_default = _ContigAnalysis(
        raw_fragment=_ContigFragMeta(hap_id="H#1", src_path="H#1#Chr01"),
        lis_multi_hit=False,
        covered_ranks=set(range(11)),
        unnormalized_rank_spans=[],
        large_jump_rank_spans=[],
        node_visit_counts={},
        has_main_chain=True,
        contig_idx=0,
    )
    assert ca_default.unnormalized_gap_spans == []

    ca_gap = _ContigAnalysis(
        raw_fragment=_ContigFragMeta(hap_id="H#1", src_path="H#1#Chr01"),
        lis_multi_hit=False,
        covered_ranks=set(range(11)),
        unnormalized_rank_spans=[],
        large_jump_rank_spans=[],
        node_visit_counts={},
        has_main_chain=True,
        contig_idx=0,
        unnormalized_gap_spans=[(4, 5)],
    )
    ca_unpickled = pickle.loads(pickle.dumps(ca_gap))
    assert ca_unpickled.unnormalized_gap_spans == [(4, 5)]
    for pair in ((4, 5), (3, 6), (4, 6), (3, 5)):
        assert _contig_covers_anchor_pair(ca_unpickled, pair[0], pair[1]) is False
    for pair in ((3, 4), (5, 6)):
        assert _contig_covers_anchor_pair(ca_unpickled, pair[0], pair[1]) is True
    for iv in ((4, 5), (3, 5), (4, 6)):
        assert _contig_covers_rank_interval(ca_unpickled, iv[0], iv[1], 11) is False
    for iv in ((4, 4), (5, 5)):
        assert _contig_covers_rank_interval(ca_unpickled, iv[0], iv[1], 11) is True

    ca_single = _ContigAnalysis(
        raw_fragment=_ContigFragMeta(hap_id="H#1", src_path="H#1#Chr01"),
        lis_multi_hit=False,
        covered_ranks={4},
        unnormalized_rank_spans=[],
        large_jump_rank_spans=[],
        node_visit_counts={},
        has_main_chain=True,
        contig_idx=1,
    )
    assert _contig_covers_anchor_pair(ca_single, 4, 5) is False
    assert _contig_covers_rank_interval(ca_single, 4, 4, 11) is False

    # 2. Attack 3c graph (11 ref nodes x 200 bp, max_jump=500, compact & string IDs)
    id_fn = (lambda i: str(i)) if use_compact_ids else (lambda i: f"n{i}")
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 12):
        gb.add_node(id_fn(i), make_complex_seq(200, seed=9100 + i))
    gb.add_node(id_fn(12), make_complex_seq(150, seed=9112))  # INS(4..5)
    gb.add_node(id_fn(13), make_complex_seq(150, seed=9113))  # INS(3..6)
    gb.add_node(id_fn(14), make_complex_seq(200, seed=9114))  # co-located novel in AltConfPlusNov
    gb.add_node(id_fn(15), make_complex_seq(150, seed=9115))  # INS(3..4)
    gb.add_node(id_fn(16), make_complex_seq(150, seed=9116))  # INS(5..6)
    ref_steps = [(id_fn(i), "+") for i in range(1, 12)]
    conflict_blk = [(id_fn(9), "-"), (id_fn(8), "-"), (id_fn(10), "-"), (id_fn(9), "-")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path(
        "AltIns45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)] + [(id_fn(12), "+")] + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb.add_p_path(
        "AltIns36#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)] + [(id_fn(13), "+")] + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb.add_p_path(
        "AltDel36#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)] + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb.add_p_path(
        "AltInv45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)]
        + [(id_fn(6), "-"), (id_fn(5), "-")]
        + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb.add_p_path(
        "AltDup45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)] + [(id_fn(11), "+")] + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb.add_p_path(
        "AltIns34#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)] + [(id_fn(15), "+")] + [(id_fn(i), "+") for i in range(5, 12)],
    )
    gb.add_p_path(
        "AltIns56#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 7)] + [(id_fn(16), "+")] + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb.add_p_path(
        "AltInv4#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)] + [(id_fn(5), "-")] + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb.add_p_path(
        "AltInv5#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)] + [(id_fn(6), "-")] + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb.add_p_path(
        "AltConfInIns45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)] + conflict_blk + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb.add_p_path(
        "AltConfPlusNov#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)]
        + [(id_fn(14), "+")]
        + conflict_blk
        + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb.add_p_path("AltClean#1#Chr01", ref_steps)
    gb.add_p_path("AltLeadConf#1#Chr01", conflict_blk + ref_steps)
    gb.add_p_path("AltTrailConf#1#Chr01", ref_steps + conflict_blk)
    gb.add_p_path("AltSingleAnc#1#Chr01", [(id_fn(5), "+")] + conflict_blk)

    gfa_p, fa_p, res = _run_pipeline_from_builder(
        tmp_path,
        f"w3_r1f1_{use_compact_ids}",
        gb,
        "Ref#1#Chr01",
        ref_steps,
        max_jump=500,
    )
    _, _, out_p = _build_via_cli(
        tmp_path,
        f"w3_r1f1_cli_{use_compact_ids}",
        gb,
        ref_steps,
        extra_args=["--max-jump", "500"],
    )
    assert verify_build_outputs(out_p, gfa_p, fa_p).passed

    def _find_ev(etype: str, anchors: str) -> object:
        return next(
            ev
            for ev in res.sv_events
            if ev.event_type == etype and ev.anchors == anchors
        )

    for etype, anchors in [
        ("INS", "4..5"),
        ("INS", "3..6"),
        ("DEL", "3..6"),
        ("INV", "3..6"),
        ("DUP_TRA", "4..5"),
    ]:
        ev = _find_ev(etype, anchors)
        assert ev.genotypes["AltConfInIns45#1"] == ".", f"{etype} {anchors} leaked on AltConfInIns45"
        assert ev.genotypes["AltConfPlusNov#1"] == ".", f"{etype} {anchors} leaked on AltConfPlusNov"
        assert ev.genotypes["AltClean#1"] == "0"
        assert ev.genotypes["AltLeadConf#1"] == "0"
        assert ev.genotypes["AltTrailConf#1"] == "0"
        assert ev.genotypes["AltSingleAnc#1"] == "."

    for etype, anchors in [
        ("INS", "3..4"),
        ("INS", "5..6"),
        ("INV", "3..5"),
        ("INV", "4..6"),
    ]:
        ev = _find_ev(etype, anchors)
        assert ev.genotypes["AltConfInIns45#1"] == "0"
        assert ev.genotypes["AltConfPlusNov#1"] == "0"
        assert ev.genotypes["AltClean#1"] == "0"

    # 3. LARGE_JUMP(3..6) and partial CNV spanning adjacent-anchor conflict gap
    gb_lj = SyntheticGFAFixtureBuilder()
    for i in range(1, 12):
        gb_lj.add_node(id_fn(i), make_complex_seq(200, seed=9200 + i))
    gb_lj.add_p_path("Ref#1#Chr01", ref_steps)
    gb_lj.add_p_path(
        "AltLj#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 5)] + [(id_fn(i), "+") for i in range(7, 12)],
    )
    gb_lj.add_p_path(
        "AltConfIn45#1#Chr01",
        [(id_fn(i), "+") for i in range(1, 6)] + conflict_blk + [(id_fn(i), "+") for i in range(6, 12)],
    )
    gb_lj.add_p_path("AltClean#1#Chr01", ref_steps)
    _, _, res_lj = _run_pipeline_from_builder(
        tmp_path,
        f"w3_r1f1_lj_{use_compact_ids}",
        gb_lj,
        "Ref#1#Chr01",
        ref_steps,
        max_jump=300,
    )
    lj_ev = next(ev for ev in res_lj.sv_events if ev.event_type == "LARGE_JUMP" and ev.anchors == "3..6")
    assert lj_ev.genotypes["AltConfIn45#1"] == "."
    assert lj_ev.genotypes["AltClean#1"] == "0"

    gb_cnv = SyntheticGFAFixtureBuilder()
    for i in range(1, 11):
        gb_cnv.add_node(id_fn(i), make_complex_seq(200, seed=9300 + i))
    ref_cnv = [(id_fn(i), "+") for i in (1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10)]
    conflict_cnv = [(id_fn(9), "-"), (id_fn(8), "-"), (id_fn(10), "-"), (id_fn(9), "-")]
    gb_cnv.add_p_path("Ref#1#Chr01", ref_cnv)
    gb_cnv.add_p_path(
        "AltDelCnv#1#Chr01",
        [(id_fn(i), "+") for i in (1, 2, 3, 5, 5, 5, 6, 7, 8, 9, 10)],
    )
    gb_cnv.add_p_path(
        "AltConfInG26#1#Chr01",
        [(id_fn(i), "+") for i in (1, 2, 3, 4, 5, 5)]
        + conflict_cnv
        + [(id_fn(i), "+") for i in (6, 7, 8, 9, 10)],
    )
    gb_cnv.add_p_path("AltClean#1#Chr01", ref_cnv)
    _, _, res_cnv = _run_pipeline_from_builder(
        tmp_path,
        f"w3_r1f1_pcnv_{use_compact_ids}",
        gb_cnv,
        "Ref#1#Chr01",
        ref_cnv,
        max_jump=1000,
    )
    pcnv_ev = next(ev for ev in res_cnv.sv_events if ev.event_type == "CNV")
    assert pcnv_ev.genotypes["AltDelCnv#1"] == "3"
    assert pcnv_ev.genotypes["AltConfInG26#1"] == "."
    assert pcnv_ev.genotypes["AltClean#1"] == "2"


@pytest.mark.parametrize("use_compact_ids", [True, False])
def test_rf2_wave3_centromere_split_ins_contig_idx_keying_and_fork_ipc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_compact_ids: bool,
) -> None:
    """C1-r3 NEW Bug 2: centromere-split multi-fragment contigs sharing (hap_id, src_path) must key INS by contig_idx."""
    id_fn = (lambda i: str(i)) if use_compact_ids else (lambda i: f"n{i}")

    # 1. Main centromere bypass graph with real 600 bp REPEAT_LIKE + flipped + T22 fork determinism
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 8):
        gb.add_node(id_fn(i), make_complex_seq(200, seed=9400 + i))
    gb.add_node(id_fn(11), make_complex_seq(150, seed=9411))  # i1
    gb.add_node(id_fn(12), make_complex_seq(160, seed=9412))  # i2
    gb.add_node(id_fn(13), "ACGT" * 150)  # 600 bp >= 500 bp -> guaranteed REPEAT_LIKE
    ref_steps = [(id_fn(i), "+") for i in range(1, 8)]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path(
        "AltIns1#1#Chr01",
        [(id_fn(1), "+"), (id_fn(11), "+")] + [(id_fn(i), "+") for i in range(2, 8)],
    )
    gb.add_p_path(
        "AltIns2#1#Chr01",
        [(id_fn(1), "+"), (id_fn(12), "+")] + [(id_fn(i), "+") for i in range(2, 8)],
    )
    loop_conflict = (
        [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")]
        + [(id_fn(5), "+"), (id_fn(6), "+")]
        + [(id_fn(1), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")]
    )
    gb.add_p_path("AltLoopConflict#1#Chr01", loop_conflict)
    gb.add_p_path(
        "AltLoopTwoAlleles#1#Chr01",
        [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")]
        + [(id_fn(5), "+"), (id_fn(6), "+")]
        + [(id_fn(1), "+"), (id_fn(12), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )
    gb.add_p_path(
        "AltLoopSameAllele#1#Chr01",
        [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")]
        + [(id_fn(5), "+"), (id_fn(6), "+")]
        + [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )
    gb.add_p_path(
        "AltLoopMaskRef#1#Chr01",
        [(id_fn(1), "+"), (id_fn(13), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")]
        + [(id_fn(5), "+"), (id_fn(6), "+")]
        + [(id_fn(1), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )
    gb.add_p_path(
        "AltLoopMaskAndA2#1#Chr01",
        [(id_fn(1), "+"), (id_fn(13), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")]
        + [(id_fn(5), "+"), (id_fn(6), "+")]
        + [(id_fn(1), "+"), (id_fn(12), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )
    gb.add_p_path(
        "AltLoopFlipped#1#Chr01",
        [(nid, "-" if o == "+" else "+") for nid, o in reversed(loop_conflict)],
    )
    gb.add_p_path(
        "AltTwoContigs#1#c1",
        [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")],
    )
    gb.add_p_path(
        "AltTwoContigs#1#c2",
        [(id_fn(1), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )
    gb.add_p_path(
        "AltTwoContigs12#1#c1",
        [(id_fn(1), "+"), (id_fn(11), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+")],
    )
    gb.add_p_path(
        "AltTwoContigs12#1#c2",
        [(id_fn(1), "+"), (id_fn(12), "+"), (id_fn(2), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(7), "+")],
    )

    bed_path = tmp_path / f"w3_b2_{use_compact_ids}.centromere.bed"
    bed_path.write_text("Chr01\t800\t1200\tCEN1\n", encoding="utf-8")

    gfa_t1, fa_t1, out_t1 = _build_via_cli(
        tmp_path,
        f"w3_b2_t1_{use_compact_ids}",
        gb,
        ref_steps,
        extra_args=[
            "--centromere-bed",
            str(bed_path),
            "--centromere-mode",
            "bypass",
            "--threads",
            "1",
        ],
    )
    out_t4 = tmp_path / f"w3_b2_t4_{use_compact_ids}_out"
    monkeypatch.setenv("PANFLAT_FORCE_FORK", "1")
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(gfa_t1),
                "--ref-fa",
                str(fa_t1),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(out_t4),
                "--centromere-bed",
                str(bed_path),
                "--centromere-mode",
                "bypass",
                "--threads",
                "4",
            ]
        )
        == 0
    )
    monkeypatch.delenv("PANFLAT_FORCE_FORK", raising=False)

    assert verify_build_outputs(out_t1, gfa_t1, fa_t1, centromere_bed_path=bed_path).passed
    assert verify_build_outputs(out_t4, gfa_t1, fa_t1, centromere_bed_path=bed_path).passed

    sv_rows = _read_tsv(Path(f"{out_t1}.sv.tsv"))
    ins_row = next(r for r in sv_rows if r["type"] == "INS" and r["anchors"] == "0..1")
    assert "len=310" not in ins_row["alleles"]
    assert "partial_content=1" not in ins_row["alleles"]

    gt_rows = _read_tsv(Path(f"{out_t1}.genotypes.tsv"))
    gt_map = next(r for r in gt_rows if r["event_id"] == ins_row["event_id"])
    assert gt_map["AltLoopConflict#1"] == "."
    assert gt_map["AltTwoContigs#1"] == "."
    assert gt_map["AltLoopTwoAlleles#1"] == "."
    assert gt_map["AltTwoContigs12#1"] == "."
    assert gt_map["AltLoopSameAllele#1"] == gt_map["AltIns1#1"]
    assert gt_map["AltLoopMaskRef#1"] == "0"
    assert gt_map["AltLoopMaskAndA2#1"] == gt_map["AltIns2#1"]
    assert gt_map["AltLoopFlipped#1"] == "."

    unres_rows = _read_tsv(Path(f"{out_t1}.unresolved.tsv"))
    assert sum(1 for r in unres_rows if r["reason"] == "REPEAT_LIKE") >= 2

    # Byte-identical SHA-256 across all emitted files between --threads 1 and forced fork --threads 4
    t1_files = sorted(out_t1.parent.glob(f"{out_t1.name}.*"))
    assert len(t1_files) >= 11
    for f1 in t1_files:
        suffix = f1.name[len(out_t1.name) :]
        f4 = out_t4.parent / f"{out_t4.name}{suffix}"
        assert f4.exists(), f"Missing fork output {f4}"
        assert hashlib.sha256(f1.read_bytes()).hexdigest() == hashlib.sha256(
            f4.read_bytes()
        ).hexdigest(), f"Fork mismatch on {suffix}"

    # 2. Centromere-split Case A rep0 interactions
    gb_ca = SyntheticGFAFixtureBuilder()
    d_ref = make_complex_seq(200, seed=9500)
    gb_ca.add_node(id_fn(1), make_complex_seq(200, seed=9501))
    gb_ca.add_node(id_fn(2), d_ref)  # backbone node inside G(0, 2)
    for i in range(3, 9):
        gb_ca.add_node(id_fn(i), make_complex_seq(200, seed=9500 + i))
    gb_ca.add_node(id_fn(21), make_complex_seq(180, seed=9521))  # true novel INS
    gb_ca.add_node(id_fn(22), mutate_seq_snps(d_ref, [50, 100]))  # Case A rep0 -> '0'
    gb_ca.add_node(id_fn(23), d_ref[:120])  # Case A rep0 uncov=80 >= 50 -> '.'
    ref_ca = [(id_fn(i), "+") for i in range(1, 9)]  # nodes 6,7 at [1000, 1400)
    gb_ca.add_p_path("Ref#1#Chr01", ref_ca)
    gb_ca.add_p_path(
        "AltTrueIns#1#Chr01",
        [(id_fn(1), "+"), (id_fn(21), "+")] + [(id_fn(i), "+") for i in range(3, 9)],
    )
    gb_ca.add_p_path(
        "AltLoopCaseAConf#1#Chr01",
        [(id_fn(1), "+"), (id_fn(21), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+")]
        + [(id_fn(6), "+"), (id_fn(7), "+")]
        + [(id_fn(1), "+"), (id_fn(22), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+"), (id_fn(8), "+")],
    )
    gb_ca.add_p_path(
        "AltLoopCaseAAndSub#1#Chr01",
        [(id_fn(1), "+"), (id_fn(22), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+")]
        + [(id_fn(6), "+"), (id_fn(7), "+")]
        + [(id_fn(1), "+"), (id_fn(23), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+"), (id_fn(8), "+")],
    )
    gb_ca.add_p_path(
        "AltTwoCaseAAndSub#1#c1",
        [(id_fn(1), "+"), (id_fn(22), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+")],
    )
    gb_ca.add_p_path(
        "AltTwoCaseAAndSub#1#c2",
        [(id_fn(1), "+"), (id_fn(23), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+"), (id_fn(8), "+")],
    )
    gb_ca.add_p_path(
        "AltLoopCaseABothDot#1#Chr01",
        [(id_fn(1), "+"), (id_fn(23), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+")]
        + [(id_fn(6), "+"), (id_fn(7), "+")]
        + [(id_fn(1), "+"), (id_fn(23), "+"), (id_fn(3), "+"), (id_fn(4), "+"), (id_fn(5), "+"), (id_fn(8), "+")],
    )
    bed_ca = tmp_path / f"w3_b2_ca_{use_compact_ids}.centromere.bed"
    bed_ca.write_text("Chr01\t1000\t1400\tCEN1\n", encoding="utf-8")
    gfa_ca, fa_ca, out_ca = _build_via_cli(
        tmp_path,
        f"w3_b2_ca_{use_compact_ids}",
        gb_ca,
        ref_ca,
        extra_args=["--centromere-bed", str(bed_ca), "--centromere-mode", "bypass"],
    )
    assert verify_build_outputs(out_ca, gfa_ca, fa_ca, centromere_bed_path=bed_ca).passed
    sv_ca = _read_tsv(Path(f"{out_ca}.sv.tsv"))
    ins_ca = next(r for r in sv_ca if r["type"] == "INS" and r["anchors"] == "0..2")
    gt_ca = next(r for r in _read_tsv(Path(f"{out_ca}.genotypes.tsv")) if r["event_id"] == ins_ca["event_id"])
    assert gt_ca["AltLoopCaseAConf#1"] == "."
    assert gt_ca["AltLoopCaseAAndSub#1"] == "0"
    assert gt_ca["AltTwoCaseAAndSub#1"] == "0"
    assert gt_ca["AltLoopCaseABothDot#1"] == "."


def test_rf3_wave3_edlib_min_core_overhang_arm_recovery_and_dispersed_dp() -> None:
    """R2-r3 F1/F2/F3 + C1-r3 B1: min aligned core, tightened overhang trim, 15->8->4/skinny DP, and arm recovery."""
    # 1. R2-r3 F1: unrelated >= min_sv_len sequences and short-target len_r < 19 rejected; short exact substrings preserved
    assert check_similarity_merge(make_complex_seq(50, seed=605001), make_complex_seq(500, seed=550001)) is False
    assert check_similarity_merge(make_complex_seq(50, seed=605008), make_complex_seq(50, seed=505008)) is False
    assert (
        check_similarity_merge(
            make_complex_seq(100, seed=710000),
            make_complex_seq(500, seed=720000),
            min_sv_len=100,
        )
        is False
    )
    for ls in (50, 51, 52, 55, 58):
        for lr in (50, 200, 500, 1000):
            for sd in range(10):
                assert (
                    check_similarity_merge(
                        make_complex_seq(ls, seed=605000 + ls * 100 + sd),
                        make_complex_seq(lr, seed=550000 + lr * 10 + sd),
                    )
                    is False
                )
    for lr in (1, 5, 10, 18):
        for sd in range(10):
            assert (
                check_similarity_merge(
                    make_complex_seq(50, seed=610000 + lr * 100 + sd),
                    make_complex_seq(lr, seed=620000 + lr * 100 + sd),
                )
                is False
            )
    for sl in (10, 15, 18):
        r100 = make_complex_seq(100, seed=630000 + sl)
        assert check_similarity_merge(r100[30 : 30 + sl], r100) is True

    # 2. R2-r3 F2: genuine >= 50 bp overhangs on short flanks rejected; clean >= 8 bp arms & cand_24_25 preserved
    for fl_k in (1, 2, 5, 10):
        for ov_l in (50, 52, 55):
            for sd in range(10):
                tail = make_complex_seq(100, seed=701000 + sd)
                u400 = make_complex_seq(400, seed=702000 + sd)
                fl = make_complex_seq(fl_k, seed=703000 + fl_k * 100 + sd)
                ov = make_complex_seq(ov_l, seed=704000 + ov_l * 100 + sd)
                if ov[0] == fl[0]:
                    ov = ("A" if fl[0] != "A" else "C") + ov[1:]
                if fl_k >= 2 and ov[1] == fl[1]:
                    ov = ov[:1] + ("A" if fl[1] != "A" else "C") + ov[2:]
                assert check_similarity_merge(u400 + ov, tail + u400 + fl) is False
                if ov[-1] == fl[-1]:
                    ov = ov[:-1] + ("A" if fl[-1] != "A" else "C")
                if fl_k >= 2 and ov[-2] == fl[-2]:
                    ov = ov[:-2] + ("A" if fl[-2] != "A" else "C") + ov[-1:]
                assert check_similarity_merge(ov + u400, fl + u400 + tail) is False
    for sd in range(15):
        u400 = make_complex_seq(400, seed=80000 + sd)
        fl1 = make_complex_seq(1, seed=82100 + sd)
        ov50 = make_complex_seq(50, seed=89000 + sd)
        if ov50[0] == fl1[0]:
            ov50 = ("A" if fl1[0] != "A" else "C") + ov50[1:]
        assert check_similarity_merge(u400 + ov50, u400 + fl1) is False
    for sd in range(10):
        u120 = make_complex_seq(120, seed=705000 + sd)
        ov30 = make_complex_seq(30, seed=706000 + sd)
        fl100 = make_complex_seq(100, seed=707000 + sd)
        s_arm = u120[:8] + ov30 + u120[8:]
        for r_ctrl in (u120 + fl100, fl100 + u120, fl100 + u120 + fl100):
            assert check_similarity_merge(s_arm, r_ctrl) is True
    # Exact outward re-extension control (matches test_m1_foundation.py:1247 cand_24_25)
    rep710 = make_complex_seq(600, seed=710)
    cand_24_25 = (
        make_complex_seq(24, seed=713)
        + rep710[:300]
        + make_complex_seq(25, seed=714)
        + rep710[300:]
    )
    assert check_similarity_merge(cand_24_25, rep710, min_sv_len=50) is True

    # 3. R2-r3 F3: dispersed SNPs (sp=13, 14, 15) + 30I / 30D and skinny min_sv_len=200 clusters
    for sp in (13, 14, 15):
        for sd in range(10):
            r800 = make_complex_seq(800, seed=64000 + sp * 100 + sd)
            pos = list(range(230, 345, sp)) + list(range(355, 470, sp))
            rm = mutate_seq_snps(r800, pos)
            ins30 = make_complex_seq(30, seed=65000 + sp * 100 + sd)
            assert check_similarity_merge(rm[:350] + ins30 + rm[350:], r800) is True
    for sd in range(10):
        r800 = make_complex_seq(800, seed=64000 + sd)
        rm = mutate_seq_snps(r800, list(range(230, 340, 15)) + list(range(390, 500, 15)))
        assert check_similarity_merge(rm[:350] + rm[380:], r800) is True
    for sd in range(5):
        r700 = make_complex_seq(700, seed=66000 + sd)
        assert check_similarity_merge(mutate_seq_snps(r700[:200] + r700[399:], [197, 203]), r700, min_sv_len=200) is True
        ins150 = make_complex_seq(150, seed=66500 + sd)
        assert (
            check_similarity_merge(
                mutate_seq_snps(r700[:250] + ins150 + r700[250:550], [247, 403]),
                r700[:550],
                min_sv_len=200,
            )
            is True
        )

    # 4. C1-r3 B1: pure < min_sv_len DEL & flanked INS arm recovery + Case A exact spans + split-alignment guard
    b205 = make_complex_seq(205, seed=82600)
    assert _check_similarity_merge_with_span(b205[:80] + b205[125:], b205) == (True, 0, 205, 45)
    fl100 = make_complex_seq(100, seed=82605)
    assert _check_similarity_merge_with_span(b205[:80] + b205[125:], fl100 + b205 + fl100) == (True, 100, 305, 45)
    u61 = make_complex_seq(61, seed=82610)
    ins49 = make_complex_seq(49, seed=82611)
    assert _check_similarity_merge_with_span(u61[:30] + ins49 + u61[30:], fl100 + u61 + fl100) == (True, 100, 161, 0)
    for ls in (80, 100, 120, 160, 200):
        for dl in (30, 45, 49):
            for sd in range(5):
                rb = make_complex_seq(ls + dl, seed=82000 + ls * 10 + dl * 100 + sd)
                assert check_similarity_merge(rb[: ls // 2] + rb[ls // 2 + dl :], rb) is True
    for sd in range(10):
        ra = make_complex_seq(90 + 49 + 400, seed=83900 + sd)
        sa = ra[:90] + ra[139:]
        fl = make_complex_seq(100, seed=83950 + sd)
        assert check_similarity_merge(sa, ra) is True
        assert check_similarity_merge(sa, fl + ra + fl) is True
    for ls, il in [(80, 40), (100, 49), (140, 49)]:
        for sd in range(5):
            fl = make_complex_seq(100, seed=84100 + ls * 10 + sd)
            core = make_complex_seq(ls - il, seed=84200 + ls * 10 + sd)
            ins = make_complex_seq(il, seed=84300 + ls * 10 + sd)
            mid = len(core) // 2
            assert check_similarity_merge(core[:mid] + ins + core[mid:], fl + core + fl) is True
    for sd in range(5):
        r600 = make_complex_seq(600, seed=68000 + sd)
        assert check_similarity_merge(r600[:150] + r600[349:], r600, min_sv_len=200) is True
    # Dual deletions 24D+24D & 45D+ov30
    fl50 = make_complex_seq(50, seed=85000)
    u40a, u40b, u40c = [make_complex_seq(40, seed=85010 + j) for j in range(3)]
    d24a, d24b = make_complex_seq(24, seed=85020), make_complex_seq(24, seed=85021)
    assert _check_similarity_merge_with_span(
        u40a + u40b + u40c, fl50 + u40a + d24a + u40b + d24b + u40c + fl50
    ) == (True, 50, 218, 48)
    ov30 = make_complex_seq(30, seed=85030)
    u55a, u55b = make_complex_seq(55, seed=85031), make_complex_seq(55, seed=85032)
    d45 = make_complex_seq(45, seed=85033)
    assert check_similarity_merge(ov30 + u55a + u55b, fl100 + u55a + d45 + u55b + fl100) is True
    # Negative controls: 50D, flanked 50I, split alignments 50M+50D+50M / 30M+100D+300M / msl=200 200M+200D+200M
    for sd in range(10):
        r150 = make_complex_seq(150, seed=86000 + sd)
        assert check_similarity_merge(r150[:50] + r150[100:], r150) is False
        r430 = make_complex_seq(430, seed=86100 + sd)
        assert check_similarity_merge(r430[:30] + r430[130:], r430) is False
        r600 = make_complex_seq(600, seed=86200 + sd)
        assert check_similarity_merge(r600[:200] + r600[400:], r600, min_sv_len=200) is False
        c60 = make_complex_seq(60, seed=86300 + sd)
        ins50 = make_complex_seq(50, seed=86400 + sd)
        assert check_similarity_merge(c60[:30] + ins50 + c60[30:], fl100 + c60 + fl100) is False


def test_rf3_wave3_edlib_unified_c1_palindrome_overhang_rc_trim_and_short_shift() -> None:
    """C1-r3 B3 + C2-r3 B1 + Caveat 3 + u=25..35 shift_safe + Fix 5 boundary."""

    def _mk_pal(p_len: int, seed: int) -> str:
        h = make_complex_seq(p_len // 2, seed=seed)
        return h + ("A" if p_len % 2 else "") + revcomp_dna(h)

    # 1. C1-r3 B3: exact palindrome + overhang (WITHOUT forcing ov[0] != fl_r[0]), two-end, P_1snp+ov, P100+49I
    for p_len in (80, 120, 200, 300):
        for ov_len in (0, 10, 20, 30, 45):
            for sd in range(8):
                fl_l = make_complex_seq(150, seed=51000 + p_len + sd)
                fl_r = make_complex_seq(150, seed=52000 + p_len + sd)
                pal = _mk_pal(p_len, seed=53000 + p_len + sd)
                ov = make_complex_seq(ov_len, seed=54000 + ov_len * 100 + sd)
                assert check_similarity_merge(pal + ov, fl_l + pal + fl_r) is False
                assert check_similarity_merge(ov + pal, fl_l + pal + fl_r) is False
    # Exact C1-r3 seed 53002 + two-end + P_1snp+ov + P(100)+49I direct tie check
    pal200_s2 = _mk_pal(200, seed=53002)
    fl_l2 = make_complex_seq(150, seed=51002)
    fl_r2 = make_complex_seq(150, seed=52002)
    assert check_similarity_merge(pal200_s2 + make_complex_seq(30, seed=54002), fl_l2 + pal200_s2 + fl_r2) is False
    assert (
        check_similarity_merge(
            make_complex_seq(15, seed=54102) + pal200_s2 + make_complex_seq(20, seed=54202),
            fl_l2 + pal200_s2 + fl_r2,
        )
        is False
    )
    assert (
        check_similarity_merge(
            mutate_seq_snps(pal200_s2, [100]) + make_complex_seq(25, seed=54302),
            fl_l2 + pal200_s2 + fl_r2,
        )
        is False
    )
    for sd in range(10):
        p100 = _mk_pal(100, seed=57000 + sd)
        ins49 = make_complex_seq(49, seed=58000 + sd)
        fl_a = make_complex_seq(150, seed=55000 + sd)
        fl_b = make_complex_seq(150, seed=56000 + sd)
        assert check_similarity_merge(p100[:50] + ins49 + p100[50:], fl_a + p100 + fl_b) is False
    # Positive controls: bridged 45D in flank + pal240 (span_extended=True) & single-copy hairpin + ov30
    for sd in range(10):
        fl_a = make_complex_seq(80, seed=59000 + sd)
        pal240 = _mk_pal(240, seed=59100 + sd)
        fl_c = make_complex_seq(80, seed=59200 + sd)
        assert check_similarity_merge(fl_a[:35] + pal240, fl_a + pal240 + fl_c) is True
        stem60 = make_complex_seq(60, seed=59300 + sd)
        hp140 = stem60 + make_complex_seq(20, seed=59400 + sd) + revcomp_dna(stem60)
        assert check_similarity_merge(hp140 + make_complex_seq(30, seed=59500 + sd), fl_a + hp140 + fl_c) is True

    # 2. C2-r3 B1: asymmetric-flank single-copy palindrome / hairpin / near-palindrome & < min_sv_len DEL -> True
    for sd in range(10):
        fl_a = make_complex_seq(120, seed=91000 + sd)
        fl_b = make_complex_seq(120, seed=92000 + sd)
        pal200 = _mk_pal(200, seed=93000 + sd)
        pal300 = _mk_pal(300, seed=93500 + sd)
        near200 = mutate_seq_snps(pal200, [20, 60, 140, 180])
        stem90 = make_complex_seq(90, seed=94000 + sd)
        hp200 = stem90 + make_complex_seq(20, seed=94500 + sd) + revcomp_dna(stem90)
        assert check_similarity_merge(fl_a[-20:] + pal200 + fl_b[:100], fl_a + pal200 + fl_b) is True
        assert check_similarity_merge(fl_a[-20:] + pal300 + fl_b[:100], fl_a + pal300 + fl_b) is True
        assert check_similarity_merge(hp200 + fl_b[:80], fl_a + hp200 + fl_b) is True
        assert check_similarity_merge(near200 + fl_b[:80], fl_a + near200 + fl_b) is True
        r_del = fl_a[:80] + _mk_pal(240, seed=95000 + sd) + fl_b[:100]
        assert check_similarity_merge(r_del[40:], r_del) is True
    # True RC duplicates rejected (P60+pal120+rc(P60) with 0/2/5 SNPs & contained pal200 + 2nd RC copy)
    for k_snps in (0, 2, 5):
        for sd in range(5):
            p60 = make_complex_seq(60, seed=97000 + k_snps * 100 + sd)
            p60_m = mutate_seq_snps(p60, [5 + i * 10 for i in range(k_snps)]) if k_snps else p60
            p120 = _mk_pal(120, seed=97100 + k_snps * 100 + sd)
            fl = make_complex_seq(80, seed=97200 + sd)
            assert check_similarity_merge(p60 + p120, fl + p60 + p120 + revcomp_dna(p60_m) + fl) is False
    for sd in range(5):
        fl1 = make_complex_seq(80, seed=97300 + sd)
        fl2 = make_complex_seq(120, seed=97400 + sd)
        p200 = _mk_pal(200, seed=97500 + sd)
        s_q = p200 + fl1
        r_t = s_q + fl2 + revcomp_dna(mutate_seq_snps(s_q, [10 * i + 5 for i in range(15)]))
        assert check_similarity_merge(s_q, r_t) is False

    # 3. C2-r3 Caveat 3 (locked spec-faithful rule)
    for sd in range(10):
        fl_a = make_complex_seq(150, seed=96000 + sd)
        fl_b = make_complex_seq(150, seed=96100 + sd)
        rc_a_last = revcomp_dna(fl_a[-1])
        if fl_b[0] == rc_a_last:
            fl_b = ("A" if rc_a_last != "A" else "C") + fl_b[1:]
        p300 = _mk_pal(300, seed=96200 + sd)
        ov25 = make_complex_seq(25, seed=96300 + sd)
        assert check_similarity_merge(p300 + fl_b[:50], fl_a + p300 + fl_b) is True
        r_d40 = fl_a[:50] + p300 + fl_b[:50]
        assert check_similarity_merge(r_d40[40:], r_d40) is True
        assert check_similarity_merge(p300 + fl_b[:10] + ov25, fl_a + p300 + fl_b) is True
        assert check_similarity_merge(p300 + fl_b[:5] + ov25, fl_a + p300 + fl_b) is False

    # 4. Short-core shift_safe (u = 25..35 bp tandem & inverted dups + msl=200 u=100 + unique 950 bp + Fix 5 boundary)
    for u_len in (25, 28, 30, 35):
        for ov_len in (20, 25):
            for sd in range(10):
                u = make_complex_seq(u_len, seed=100000 + u_len * 1000 + ov_len * 100 + sd)
                u1 = mutate_seq_snps(u, [u_len // 3])
                ov = make_complex_seq(ov_len, seed=500000 + u_len * 1000 + ov_len * 100 + sd)
                fl = make_complex_seq(80, seed=600000 + u_len * 100 + sd)
                assert check_similarity_merge(ov + u, u1 + u) is False
                assert check_similarity_merge(ov + u, fl + u1 + u + fl) is False
                assert check_similarity_merge(ov + u, u + revcomp_dna(u1)) is False
                assert check_similarity_merge(ov + u, fl + u + revcomp_dna(u1) + fl) is False
                # Forced 2 bp seam bleed on inverted duplicate
                ov_bleed = revcomp_dna(u1)[:2] + ov[2:]
                assert check_similarity_merge(u + ov_bleed, fl + u + revcomp_dna(u1) + fl) is False
    for sd in range(5):
        u100 = make_complex_seq(100, seed=700000 + sd)
        u100_1 = mutate_seq_snps(u100, [30])
        ov80 = make_complex_seq(80, seed=710000 + sd)
        assert check_similarity_merge(ov80 + u100, u100_1 + u100, min_sv_len=200) is False
    for u_len in (25, 30, 35, 50):
        for sd in range(10):
            r950 = make_complex_seq(950, seed=720000 + u_len * 100 + sd)
            assert check_similarity_merge(r950[400 : 400 + u_len], r950) is True
    # Fix 5 d=10, 30, 49 with non-bleeding ov30 -> True; exact d=50 distinct RC -> False
    for d in (10, 30, 49):
        for sd in range(5):
            fl1 = make_complex_seq(100, seed=730000 + d * 100 + sd)
            fl2 = make_complex_seq(100, seed=740000 + d * 100 + sd)
            ad = make_complex_seq(d, seed=750000 + d * 100 + sd)
            p_near = mutate_seq_snps(_mk_pal(200 - d, seed=760000 + d * 100 + sd), [5, 15, 25, 35])
            ov30 = make_complex_seq(30, seed=770000 + d * 100 + sd)
            if ov30[0] == fl2[0]:
                ov30 = ("A" if fl2[0] != "A" else "C") + ov30[1:]
            if ov30[-1] == fl1[-1]:
                ov30 = ov30[:-1] + ("A" if fl1[-1] != "A" else "C")
            assert check_similarity_merge(ad + p_near + ov30, fl1 + ad + p_near + revcomp_dna(ad) + fl2) is True
            assert check_similarity_merge(ov30 + p_near + ad, fl1 + revcomp_dna(ad) + p_near + ad + fl2) is True
    ad50 = make_complex_seq(50, seed=780000)
    p150_near = mutate_seq_snps(_mk_pal(150, seed=780001), [5, 15, 25, 35])
    fl1_50 = make_complex_seq(100, seed=780002)
    fl2_50 = make_complex_seq(100, seed=780003)
    assert check_similarity_merge(ad50 + p150_near, fl1_50 + ad50 + p150_near + revcomp_dna(ad50) + fl2_50) is False


def test_rf3_wave4_edlib_dual_del_split_pal_del_tie_and_dup_d1_d2() -> None:
    """Wave 4 / Iteration 5 regression guard: C1-r4 Bugs 1/2/3 + C2 V10 + bleed/FP controls."""
    import random

    def _mk_pal(p_len: int, seed: int) -> str:
        h = make_complex_seq(p_len // 2, seed=seed)
        return h + ("A" if p_len % 2 else "") + revcomp_dna(h)

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    def _flip1(ch: str) -> str:
        return "A" if ch != "A" else "C"

    # --- 1. C1-r4 Bug 1 & C2 V10: cumulative Stage 4 split-alignment guard ---
    # Dual DEL >= min_sv_len (98D, 50D, msl=100 198D, V10 598 vs 500 bp 98D) -> False
    r98 = make_complex_seq(45 + 49 + 100 + 49 + 45, seed=324945)
    fl100 = make_complex_seq(100, seed=324946)
    s98 = r98[:45] + r98[94:194] + r98[243:]
    assert check_similarity_merge(s98, r98) is False
    assert check_similarity_merge(s98, fl100 + r98 + fl100) is False
    r50 = make_complex_seq(40 + 25 + 100 + 25 + 40, seed=240400)
    assert check_similarity_merge(r50[:40] + r50[65:165] + r50[190:], r50) is False
    r198 = make_complex_seq(90 + 99 + 200 + 99 + 90, seed=324990)
    assert check_similarity_merge(r198[:90] + r198[189:389] + r198[488:], r198, min_sv_len=100) is False
    r598 = make_complex_seq(45 + 49 + 410 + 49 + 45, seed=60101)
    assert check_similarity_merge(r598[:45] + r598[94:504] + r598[553:], r598) is False

    # Cumulative split_del_bp across both unrecovered arms + triple DEL -> False
    for sd in range(5):
        rb = make_complex_seq(19 + 25 + 100 + 25 + 19, seed=71000 + sd)
        sb = mutate_seq_snps(rb[:19], [9]) + rb[44:144] + mutate_seq_snps(rb[169:], [9])
        assert check_similarity_merge(sb, rb) is False
        rt = make_complex_seq(30 + 20 + 60 + 20 + 60 + 20 + 30, seed=71100 + sd)
        assert check_similarity_merge(rt[:30] + rt[50:110] + rt[130:190] + rt[210:], rt) is False

    # Exact 1-3 bp boundary-collision overlap on armR=19 -> False
    for col in (1, 2, 3):
        for sd in range(4):
            rc = make_complex_seq(45 + 49 + 100 + 49 + 19, seed=72000 + col * 100 + sd)
            a_l, d1, mid, d2, a_r = rc[:45], rc[45:94], rc[94:194], rc[194:243], rc[243:]
            d2_col = a_r[:col] + _flip1(a_r[col]) + d2[col + 1 :]
            assert check_similarity_merge(a_l + mid + a_r, a_l + d1 + mid + d2_col + a_r) is False

    # Positive controls: 48D exact span, min_gap edge-SNP guard (38D+10D=48D), novel ov + 49D FP guard
    r48 = make_complex_seq(40 + 24 + 100 + 24 + 40, seed=142424)
    assert _check_similarity_merge_with_span(r48[:40] + r48[64:164] + r48[188:], r48) == (True, 0, 228, 48)
    for sd in range(8):
        rmg = make_complex_seq(40 + 20 + 10 + 100 + 38 + 45, seed=73000 + sd)
        a_l_m = mutate_seq_snps(rmg[40:60], [10, 18, 19])
        assert check_similarity_merge(a_l_m + rmg[70:170] + rmg[208:], rmg) is True
    for ov_len in (19, 30, 49):
        for sd in range(4):
            fl_l = make_complex_seq(300, seed=73100 + ov_len * 10 + sd)
            fl_r = make_complex_seq(300, seed=73200 + ov_len * 10 + sd)
            body = make_complex_seq(300 + 49, seed=73300 + ov_len * 10 + sd)
            ov = make_complex_seq(ov_len, seed=73400 + ov_len * 10 + sd)
            assert check_similarity_merge(ov + body[:150] + body[199:], fl_l + body + fl_r) is True

    # --- 2. C1-r4 Bug 2: exact zero-FN k_sl same-locus tie & short <=0-score pal ---
    # Zero-overhang 1-SNP symmetric palindrome + internal DEL -> False
    for p_len, d in ((140, 40), (160, 45), (180, 49)):
        for sd in range(4):
            pal = _mk_pal(p_len, seed=220000 + p_len + sd)
            fl1 = make_complex_seq(150, seed=222000 + sd)
            fl2 = make_complex_seq(150, seed=223000 + sd)
            dd = make_complex_seq(d, seed=221000 + d + sd)
            r_pal = fl1 + pal[: p_len // 2] + dd + pal[p_len // 2 :] + fl2
            assert check_similarity_merge(mutate_seq_snps(pal, [20]), r_pal) is False

    # Exact b=5, 7 bleed + ov20 + 45D (locks exact k_sl against naive_k FN)
    for b in (5, 7):
        for sd in range(5):
            pal = _mk_pal(160, seed=75000 + b * 100 + sd)
            fl1 = make_complex_seq(150, seed=75100 + b * 100 + sd)
            fl2 = make_complex_seq(150, seed=75200 + b * 100 + sd)
            dd = make_complex_seq(45, seed=75300 + b * 100 + sd)
            ov = make_complex_seq(20, seed=75400 + b * 100 + sd)
            ov_b = fl2[:b] + _flip1(fl2[b]) + ov[b + 1 :]
            assert check_similarity_merge(pal + ov_b, fl1 + pal[:80] + dd + pal[80:] + fl2) is False

    # Short <= 0 score symmetric palindromes vs non-palindromic controls
    for sd in range(4):
        fl1, fl2 = make_complex_seq(120, seed=76000 + sd), make_complex_seq(120, seed=76100 + sd)
        p60, p80 = _mk_pal(60, seed=76200 + sd), _mk_pal(80, seed=76300 + sd)
        i49 = make_complex_seq(49, seed=76400 + sd)
        d45, d49 = make_complex_seq(45, seed=76500 + sd), make_complex_seq(49, seed=76600 + sd)
        assert check_similarity_merge(p60[:30] + i49 + p60[30:], fl1 + p60 + fl2) is False
        assert check_similarity_merge(mutate_seq_snps(p60, [10]), fl1 + p60[:30] + d45 + p60[30:] + fl2) is False
        assert check_similarity_merge(mutate_seq_snps(p80, [10]), fl1 + p80[:40] + d49 + p80[40:] + fl2) is False
        u60, u80 = make_complex_seq(60, seed=76700 + sd), make_complex_seq(80, seed=76800 + sd)
        assert check_similarity_merge(u60[:30] + i49 + u60[30:], fl1 + u60 + fl2) is True
        assert check_similarity_merge(mutate_seq_snps(u80, [10]), fl1 + u80[:40] + d49 + u80[40:] + fl2) is True

    # C1-r3 B3 bleed guards (single-end 2M 1X 5M & two-end 5M 1X) -> False
    pal160 = _mk_pal(160, seed=77001)
    fl1_b, fl2_b = make_complex_seq(150, seed=77002), make_complex_seq(150, seed=77003)
    ov_2m1x5m = fl2_b[:2] + _flip1(fl2_b[2]) + fl2_b[3:8] + _flip1(fl2_b[8]) + make_complex_seq(12, seed=77004)
    assert check_similarity_merge(pal160 + ov_2m1x5m, fl1_b + pal160 + fl2_b) is False
    b_r6 = fl2_b[:6]
    rc_br6 = revcomp_dna(b_r6)
    b_l6 = rc_br6[:5] + _flip1(rc_br6[5])
    fl1_mod = fl1_b[:-6] + b_l6
    ov_l = make_complex_seq(14, seed=77005)
    ov_r = make_complex_seq(14, seed=77006)
    if ov_l[-1] == fl1_mod[-7]:
        ov_l = ov_l[:-1] + _flip1(fl1_mod[-7])
    if ov_r[0] == fl2_b[6]:
        ov_r = _flip1(fl2_b[6]) + ov_r[1:]
    assert check_similarity_merge(ov_l + b_l6 + pal160 + b_r6 + ov_r, fl1_mod + pal160 + fl2_b) is False

    # Positive controls: bridged 45D + pal240 (L & R) & 950 bp 1-SNP non-palindromic
    for sd in range(5):
        fl_a, fl_c = make_complex_seq(80, seed=400000 + sd), make_complex_seq(80, seed=402000 + sd)
        pal240 = _mk_pal(240, seed=401000 + sd)
        assert check_similarity_merge(fl_a[:35] + pal240, fl_a + pal240 + fl_c) is True
        assert check_similarity_merge(pal240 + fl_c[45:], fl_a + pal240 + fl_c) is True
    r950_np = make_complex_seq(950, seed=500001)
    assert check_similarity_merge(mutate_seq_snps(r950_np, [475]), r950_np) is True

    # --- 3. C1-r4 Bug 3: Stage 6D internal-deletion multi-arm uniqueness ---
    # Exact random 0-SNP D1!=D2, +5 SNPs, AT/GC (fwd & RC), msl=200 random 0-SNP
    u1 = make_complex_seq(50, seed=120504)
    u2 = make_complex_seq(50, seed=121504)
    sep60 = make_complex_seq(60, seed=122004)
    d1_r, d2_r = make_complex_seq(49, seed=123004), make_complex_seq(49, seed=124004)
    assert check_similarity_merge(u1 + u2, u1 + d1_r + u2 + sep60 + u1 + d2_r + u2) is False
    u1_m, u2_m = mutate_seq_snps(u1, [5, 20, 35]), mutate_seq_snps(u2, [10, 30])
    da49, dg49 = _mk_biased(49, "AT", 125004), _mk_biased(49, "GC", 126004)
    assert check_similarity_merge(u1 + u2, u1 + d1_r + u2 + sep60 + u1_m + d2_r + u2_m) is False
    assert check_similarity_merge(u1 + u2, u1 + da49 + u2 + sep60 + u1 + dg49 + u2) is False
    assert check_similarity_merge(u1 + u2, u1 + da49 + u2 + sep60 + revcomp_dna(u1 + dg49 + u2)) is False
    assert check_similarity_merge(u1 + u2, u1 + d1_r + u2 + sep60 + revcomp_dna(u1_m + d2_r + u2_m)) is False

    for sd in range(4):
        a100, b100 = make_complex_seq(100, seed=81000 + sd), make_complex_seq(100, seed=82000 + sd)
        d180a, d180b = make_complex_seq(180, seed=83000 + sd), make_complex_seq(180, seed=84000 + sd)
        s60 = make_complex_seq(60, seed=85000 + sd)
        assert check_similarity_merge(a100 + b100, a100 + d180a + b100 + s60 + a100 + d180b + b100, min_sv_len=200) is False
        assert check_similarity_merge(a100 + b100, a100 + d180a + b100 + s60 + revcomp_dna(a100 + d180b + b100), min_sv_len=200) is False

    # Tandem sep=0 (with & without 3 bp seam bleed), overlapping U+D1+U+D2+U, and multi-DEL (24D+24D & asym 24D+24D vs 45D)
    u25a, u25b = make_complex_seq(25, seed=135001), make_complex_seq(25, seed=136001)
    u25a_m, u25b_m = mutate_seq_snps(u25a, [4, 12, 20]), mutate_seq_snps(u25b, [4, 12, 20])
    assert check_similarity_merge(u25a + u25b, u25a + da49 + u25b + u25a_m + dg49 + u25b_m) is False
    ov20 = make_complex_seq(20, seed=137001)
    assert check_similarity_merge(u1 + u2 + u1_m[:3] + ov20[3:], u1 + da49 + u2 + u1_m + dg49 + u2_m) is False
    assert check_similarity_merge(u1 + u1, u1 + da49 + u1 + dg49 + u1) is False
    c1, c2, c3 = [make_complex_seq(40, seed=230000 + j) for j in range(3)]
    d1a, d1b = _mk_biased(24, "AT", 233000), _mk_biased(24, "AT", 234000)
    d2a, d2b = _mk_biased(24, "GC", 235000), _mk_biased(24, "GC", 236000)
    dg45 = _mk_biased(45, "GC", 238000)
    assert check_similarity_merge(c1 + c2 + c3, c1 + d1a + c2 + d1b + c3 + sep60 + c1 + d2a + c2 + d2b + c3) is False
    assert check_similarity_merge(c1 + c2 + c3, c1 + d1a + c2 + d1b + c3 + sep60 + c1 + dg45 + c2 + c3) is False

    # Bug 3 FP controls: U1-only repeat, unique 950 bp DEL, C2-r3 B1 internal 40D + fl55
    for sd in range(5):
        ua, ub = make_complex_seq(25, seed=350000 + sd), make_complex_seq(120, seed=351000 + sd)
        unrel = make_complex_seq(180, seed=354000 + sd)
        assert check_similarity_merge(ua + ub, ua + da49 + ub + sep60 + ua + unrel) is True
        r950_d = make_complex_seq(950, seed=340000 + sd)
        assert check_similarity_merge(r950_d[300:350] + r950_d[399:449], r950_d) is True
        fl_a, fl_b = make_complex_seq(100, seed=310000 + sd), make_complex_seq(100, seed=311000 + sd)
        p240 = _mk_pal(240, seed=312000 + sd)
        assert check_similarity_merge(p240[:100] + p240[140:] + fl_b[:55], fl_a + p240 + fl_b) is True
        assert check_similarity_merge(fl_a[-55:] + p240[:100] + p240[140:], fl_a + p240 + fl_b) is True


def test_rf3_wave5_stage4_overhang_split_p1c_and_mappy_parity() -> None:
    """Wave 5 / Iteration 6 Group A regression guard: C1-r5 Bugs 1 & 2, R1-r5 Minor F2, FP & Red Line 3 controls."""
    import random

    def _mk_at_rich(n: int, seed: int, at_prob: float = 0.60) -> str:
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            if rng.random() < at_prob:
                out.append(rng.choice("AT"))
            else:
                out.append(rng.choice("CG"))
        return "".join(out)

    def _flip1(ch: str) -> str:
        return "A" if ch != "A" else "C"

    # --- 1. C1-r5 NEW Bug 1: Stage 4 split-alignment guard under outer novel overhangs ---
    # Verbatim Challenger 1 (r5) §5 Bug 1R & 1L (470 vs 275 bp, arm=30, 45D+45D=90D, ov=15)
    arm_l = make_complex_seq(30, seed=15000)
    d1 = make_complex_seq(45, seed=15100)
    mid = make_complex_seq(200, seed=15200)
    d2 = make_complex_seq(45, seed=15300)
    arm_r = make_complex_seq(30, seed=15400)
    ov15 = make_complex_seq(15, seed=15500)
    fl60 = make_complex_seq(60, seed=15600)
    r_b1 = fl60 + arm_l + d1 + mid + d2 + arm_r + fl60
    assert check_similarity_merge(arm_l + mid + arm_r + ov15, r_b1) is False
    assert check_similarity_merge(ov15 + arm_l + mid + arm_r, r_b1) is False

    # Sweep ov in (8, 10, 12, 18, 25) on both L & R ends -> False
    for ov_len in (8, 10, 12, 18, 25):
        for sd in range(3):
            al = make_complex_seq(30, seed=810000 + ov_len * 10 + sd)
            dd1 = make_complex_seq(45, seed=811000 + ov_len * 10 + sd)
            md = make_complex_seq(200, seed=812000 + ov_len * 10 + sd)
            dd2 = make_complex_seq(45, seed=813000 + ov_len * 10 + sd)
            ar = make_complex_seq(30, seed=814000 + ov_len * 10 + sd)
            ov = make_complex_seq(ov_len, seed=815000 + ov_len * 10 + sd)
            fl = make_complex_seq(60, seed=816000 + ov_len * 10 + sd)
            rt = fl + al + dd1 + md + dd2 + ar + fl
            assert check_similarity_merge(al + md + ar + ov, rt) is False
            assert check_similarity_merge(ov + al + md + ar, rt) is False

    # Boundary-collision col in (1, 2, 3) on arm=19 (0 & 1 SNP) + ov in (3, 15) L & R -> False
    for col in (1, 2, 3):
        for ov_len in (3, 15):
            for n_snp in (0, 1):
                for sd in range(3):
                    al = make_complex_seq(45, seed=820000 + col * 100 + ov_len * 10 + sd)
                    dd1 = make_complex_seq(45, seed=821000 + col * 100 + ov_len * 10 + sd)
                    md = make_complex_seq(120, seed=822000 + col * 100 + ov_len * 10 + sd)
                    dd2 = make_complex_seq(45, seed=823000 + col * 100 + ov_len * 10 + sd)
                    ar = make_complex_seq(19, seed=824000 + col * 100 + ov_len * 10 + sd)
                    ov = make_complex_seq(ov_len, seed=825000 + col * 100 + ov_len * 10 + sd)
                    ar_q = mutate_seq_snps(ar, [9]) if n_snp else ar
                    # Right-arm collision + right overhang
                    dd2_col = ar[:col] + _flip1(ar[col]) + dd2[col + 1 :]
                    assert (
                        check_similarity_merge(
                            al + md + ar_q + ov,
                            fl60 + al + dd1 + md + dd2_col + ar + fl60,
                        )
                        is False
                    )
                    # Left-arm collision + left overhang
                    dd1_col = dd1[: -(col + 1)] + _flip1(ar[-(col + 1)]) + ar[-col:]
                    assert (
                        check_similarity_merge(
                            ov + ar_q + md + al,
                            fl60 + ar + dd1_col + md + dd2 + al + fl60,
                        )
                        is False
                    )

    # Both-end overhang + dual 25D (ovL=4 + armL19(1SNP) + 25D + mid + 25D + armR19(1SNP) + ovR=4) -> False
    for sd in range(4):
        al19 = make_complex_seq(19, seed=830000 + sd)
        ar19 = make_complex_seq(19, seed=831000 + sd)
        md120 = make_complex_seq(120, seed=832000 + sd)
        d25a = make_complex_seq(25, seed=833000 + sd)
        d25b = make_complex_seq(25, seed=834000 + sd)
        ov4l = make_complex_seq(4, seed=835000 + sd)
        ov4r = make_complex_seq(4, seed=836000 + sd)
        r_both = fl60 + al19 + d25a + md120 + d25b + ar19 + fl60
        s_both = ov4l + mutate_seq_snps(al19, [9]) + md120 + mutate_seq_snps(ar19, [9]) + ov4r
        assert check_similarity_merge(s_both, r_both) is False

    # Single 60D + arm25 + ov15 -> False (L & R)
    for sd in range(3):
        a25 = make_complex_seq(25, seed=837000 + sd)
        d60 = make_complex_seq(60, seed=838000 + sd)
        md180 = make_complex_seq(180, seed=839000 + sd)
        ov15_s = make_complex_seq(15, seed=839500 + sd)
        assert check_similarity_merge(md180 + a25 + ov15_s, fl60 + md180 + d60 + a25 + fl60) is False
        assert check_similarity_merge(ov15_s + a25 + md180, fl60 + a25 + d60 + md180 + fl60) is False

    # Bug 1 FP controls: 60% AT-rich novel ov(16, 19, 30, 49) + 45D/49D in fl700 (L & R) -> True
    for ov_len in (16, 19, 30, 49):
        for d_len in (45, 49):
            for sd in range(3):
                fl_at = _mk_at_rich(350, seed=840000 + ov_len * 100 + d_len * 10 + sd)
                body = make_complex_seq(220 + d_len, seed=841000 + ov_len * 100 + d_len * 10 + sd)
                ov_at = _mk_at_rich(ov_len, seed=842000 + ov_len * 100 + d_len * 10 + sd)
                s_body = body[:110] + body[110 + d_len :]
                assert check_similarity_merge(ov_at + s_body, fl_at + body + fl_at) is True
                assert check_similarity_merge(s_body + ov_at, fl_at + body + fl_at) is True

    # P1-G FP control: 20 bp left tandem dup + 45D right (with & without ov10) -> True
    for sd in range(4):
        u20 = make_complex_seq(20, seed=845000 + sd)
        md150 = make_complex_seq(150, seed=846000 + sd)
        d45_g = make_complex_seq(45, seed=847000 + sd)
        tail50 = make_complex_seq(50, seed=848000 + sd)
        ov10 = make_complex_seq(10, seed=849000 + sd)
        r_p1g = fl60 + u20 + md150 + d45_g + tail50 + fl60
        assert check_similarity_merge(u20 + u20 + md150 + tail50, r_p1g) is True
        assert check_similarity_merge(ov10 + u20 + u20 + md150 + tail50, r_p1g) is True

    # --- 2. R1-r5 Minor F2: Length-conservation target-gap lower bound (P1-C vs P1-B & internal 1I) ---
    for sd in range(5):
        fl40 = make_complex_seq(40, seed=850000 + sd)
        arm20 = make_complex_seq(20, seed=851000 + sd)
        d12 = make_complex_seq(12, seed=852000 + sd)
        md100 = make_complex_seq(100, seed=853000 + sd)
        d38 = make_complex_seq(38, seed=854000 + sd)
        a45 = make_complex_seq(45, seed=855000 + sd)
        ov15_c = make_complex_seq(15, seed=856000 + sd)
        # Left P1-C: clean 3-SNP (1 mid + 2 edge == d12[:2]) & ov15 2-edge-SNP -> False
        arm20_l = arm20[:10] + _flip1(arm20[10]) + arm20[11:18] + d12[:2]
        arm20_l_ov = arm20[:18] + d12[:2]
        if arm20_l[0] == fl40[-1]:
            fl40_l = fl40[:-1] + _flip1(arm20_l[0])
        else:
            fl40_l = fl40
        r_p1c_l = fl40_l + arm20 + d12 + md100 + d38 + a45
        assert check_similarity_merge(arm20_l + md100 + a45, r_p1c_l) is False
        assert check_similarity_merge(ov15_c + arm20_l_ov + md100 + a45, r_p1c_l) is False
        # Right P1-C: clean 3-SNP (1 mid + 2 edge == d12[-2:]) & ov15 2-edge-SNP -> False
        arm20_r = d12[-2:] + arm20[2:10] + _flip1(arm20[10]) + arm20[11:]
        arm20_r_ov = d12[-2:] + arm20[2:]
        if arm20_r[-1] == fl40[0]:
            fl40_r = _flip1(arm20_r[-1]) + fl40[1:]
        else:
            fl40_r = fl40
        r_p1c_r = a45 + d38 + md100 + d12 + arm20 + fl40_r
        assert check_similarity_merge(a45 + md100 + arm20_r, r_p1c_r) is False
        assert check_similarity_merge(a45 + md100 + arm20_r_ov + ov15_c, r_p1c_r) is False
        # P1-B positive control (38D+10D=48D edge SNPs, L & R) -> True
        d10 = d12[:10]
        arm20_bl = arm20[:10] + _flip1(arm20[10]) + arm20[11:18] + d10[:2]
        arm20_br = d10[-2:] + arm20[2:10] + _flip1(arm20[10]) + arm20[11:]
        assert check_similarity_merge(arm20_bl + md100 + a45, fl40_l + arm20 + d10 + md100 + d38 + a45) is True
        assert check_similarity_merge(a45 + md100 + arm20_br, a45 + d38 + md100 + d10 + arm20 + fl40_r) is True
        # Internal 1I inside arm20 across 38D+10D=48D (t_span_hit < q_arm_len) -> True
        arm20_1i = arm20[:10] + _flip1(arm20[10]) + arm20[10:]
        assert check_similarity_merge(arm20_1i + md100 + a45, fl40_l + arm20 + d10 + md100 + d38 + a45) is True
        assert check_similarity_merge(a45 + md100 + arm20_1i, a45 + d38 + md100 + d10 + arm20 + fl40_r) is True

    # --- 3. C1-r5 NEW Bug 2: 999/1000/1001/1200 bp edlib vs mappy cumulative del_bp_internal parity ---
    # Verbatim Challenger 1 (r5) §5 Bug 2 (1001 bp 49D+49D=98D)
    p1 = make_complex_seq(250, seed=96300)
    p2 = make_complex_seq(250, seed=96301)
    p3 = make_complex_seq(403, seed=96302)
    dd1_98 = make_complex_seq(49, seed=96303)
    dd2_98 = make_complex_seq(49, seed=96304)
    assert check_similarity_merge(p1 + p2 + p3, p1 + dd1_98 + p2 + dd2_98 + p3) is False

    for target_len in (999, 1000, 1001, 1200):
        seq_r = make_complex_seq(target_len, seed=860000 + target_len)
        # Dual & quad DEL >= 50 bp -> False across both engines
        for d_a, d_b in ((25, 25), (30, 30), (49, 49)):
            s_2d = seq_r[:250] + seq_r[250 + d_a : 500 + d_a] + seq_r[500 + d_a + d_b :]
            assert check_similarity_merge(s_2d, seq_r) is False
        s_4d = (
            seq_r[:150]
            + seq_r[199:349]
            + seq_r[398:548]
            + seq_r[597:747]
            + seq_r[796:]
        )
        assert check_similarity_merge(s_4d, seq_r) is False
        # Dual DEL < 50 bp (48D, 49D) -> True across both engines
        for d_a, d_b in ((24, 24), (24, 25)):
            s_ok_d = seq_r[:250] + seq_r[250 + d_a : 500 + d_a] + seq_r[500 + d_a + d_b :]
            assert check_similarity_merge(s_ok_d, seq_r) is True
        # Small INS & combined I+D (< 50 bp) -> True; cumulative 25I+25I=50I -> False
        ins30 = make_complex_seq(30, seed=861000 + target_len)
        ins49 = make_complex_seq(49, seed=862000 + target_len)
        ins24 = make_complex_seq(24, seed=863000 + target_len)
        ins25a = make_complex_seq(25, seed=864000 + target_len)
        ins25b = make_complex_seq(25, seed=865000 + target_len)
        ins20 = make_complex_seq(20, seed=866000 + target_len)
        assert check_similarity_merge(seq_r[:400] + ins30 + seq_r[400:], seq_r) is True
        assert check_similarity_merge(seq_r[:400] + ins49 + seq_r[400:], seq_r) is True
        assert check_similarity_merge(seq_r[:300] + ins24 + seq_r[300:600] + ins25a + seq_r[600:], seq_r) is True
        assert check_similarity_merge(seq_r[:300] + ins25a + seq_r[300:600] + ins25b + seq_r[600:], seq_r) is False
        assert check_similarity_merge(seq_r[:300] + ins20 + seq_r[300:600] + seq_r[620:], seq_r) is True
        # Clean 400 bp embedded subsequence -> True across both engines
        assert check_similarity_merge(seq_r[250:650], seq_r) is True

    # Red Line 3 explicit guards: 200=30I200= & 225=49I225= MUST remain True
    r400 = make_complex_seq(400, seed=867001)
    assert check_similarity_merge(r400[:200] + make_complex_seq(30, seed=867002) + r400[200:], r400) is True
    r450 = make_complex_seq(450, seed=867003)
    assert check_similarity_merge(r450[:225] + make_complex_seq(49, seed=867004) + r450[225:], r450) is True


def test_rf3_wave5_shifted_pal_ties_exact_vs_indel_copy2_and_compound_id() -> None:
    """Wave 5 / Iteration 6 Group B + Stage 6E hardening guard: R1-r5 F1, C1-r5 Bugs 3 & 4, and FP controls."""
    import random

    def _mk_pal(p_len: int, seed: int) -> str:
        h = make_complex_seq(p_len // 2, seed=seed)
        return h + ("A" if p_len % 2 else "") + revcomp_dna(h)

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    def _flip1(ch: str) -> str:
        return "A" if ch != "A" else "C"

    # --- 1. R1-r5 Major F1: Shifted same-locus palindrome orientation ties ---
    # Verbatim Reviewer 1 (r5) §5.1 reproducer
    h80 = make_complex_seq(80, seed=220160)
    pal160 = h80 + revcomp_dna(h80)
    fl1 = make_complex_seq(150, seed=222000)
    fl2 = make_complex_seq(150, seed=223000)
    d45 = make_complex_seq(45, seed=221045)
    r_pal = fl1 + pal160[:80] + d45 + pal160[80:] + fl2
    for pos in (0, 1, 2, 155, 159):
        assert check_similarity_merge(mutate_seq_snps(pal160, [pos]), r_pal) is False
    assert check_similarity_merge(mutate_seq_snps(pal160, [0]), pal160[:80] + d45 + pal160[80:]) is False
    for k in (1, 2, 5, 10, 20, 40):
        assert check_similarity_merge(pal160[k:], r_pal) is False
        assert check_similarity_merge(pal160[:-k], r_pal) is False
    pal240 = _mk_pal(240, seed=910240)
    r_pal240 = fl1 + pal240[:120] + d45 + pal240[120:] + fl2
    for k in (40, 49):
        assert check_similarity_merge(pal240[k:], r_pal240) is False
        assert check_similarity_merge(pal240[:-k], r_pal240) is False

    # Short-arm shifted palindrome ties (pal50[10:]+45D, pal60[15:]+45D) -> False
    for p_len, k in ((50, 10), (60, 15)):
        for sd in range(3):
            p_sh = _mk_pal(p_len, seed=911000 + p_len * 10 + sd)
            dd = make_complex_seq(45, seed=912000 + p_len * 10 + sd)
            r_sh = fl1 + p_sh[: p_len // 2] + dd + p_sh[p_len // 2 :] + fl2
            assert check_similarity_merge(p_sh[k:], r_sh) is False
            assert check_similarity_merge(p_sh[:-k], r_sh) is False

    # Shifted internal 'I' ties (pal60/80/100[1,5,10:]+45I, +SNP@0, pal140[30:]+49I, pal38[2:]+15I) -> False
    i45 = make_complex_seq(45, seed=994000)
    for p_len in (60, 80, 100):
        p_i = _mk_pal(p_len, seed=991801 + p_len)
        r_pi = fl1 + p_i + fl2
        for k in (1, 5, 10):
            assert check_similarity_merge(p_i[k : p_len // 2] + i45 + p_i[p_len // 2 :], r_pi) is False
        s_snp0 = mutate_seq_snps(p_i[: p_len // 2], [0]) + i45 + p_i[p_len // 2 :]
        assert check_similarity_merge(s_snp0, r_pi) is False
    pal140 = _mk_pal(140, seed=913140)
    i49 = make_complex_seq(49, seed=913149)
    assert check_similarity_merge(pal140[30:70] + i49 + pal140[70:], fl1 + pal140 + fl2) is False
    # <= 0-score <= 30% shifted internal 'I' tie (pal38[2:]+15I) -> False
    pal38 = _mk_pal(38, seed=913038)
    i15 = make_complex_seq(15, seed=913015)
    assert check_similarity_merge(pal38[2:19] + i15 + pal38[19:], fl1 + pal38 + fl2) is False

    # Shifted bleed (pal160[0,1,5,10,20:]+45D+b=5+ov20) -> False
    ov20_b = make_complex_seq(20, seed=914000)
    ov_b5 = fl2[:5] + _flip1(fl2[5]) + ov20_b[6:]
    for k in (0, 1, 5, 10, 20):
        assert check_similarity_merge(pal160[k:] + ov_b5, r_pal) is False

    # Shifted compound 2x20I+45D palindrome -> False
    pal200 = _mk_pal(200, seed=915200)
    ins20_a = make_complex_seq(20, seed=915201)
    ins20_b = make_complex_seq(20, seed=915202)
    r_cmp_pal = fl1 + pal200[:100] + d45 + pal200[100:] + fl2
    s_cmp_pal = pal200[5:50] + ins20_a + pal200[50:150] + ins20_b + pal200[150:]
    assert check_similarity_merge(s_cmp_pal, r_cmp_pal) is False

    # Group B R1-r5 F1 FP controls (MUST remain True) & C1-r3 B3 bleed guards (MUST remain False)
    for fl_len in (10, 20, 35, 55):
        fl_a, fl_b = make_complex_seq(100, seed=916000 + fl_len), make_complex_seq(100, seed=916100 + fl_len)
        p240_c = _mk_pal(240, seed=916200 + fl_len)
        assert check_similarity_merge(p240_c[:100] + p240_c[140:] + fl_b[:fl_len], fl_a + p240_c + fl_b) is True
        assert check_similarity_merge(fl_a[-fl_len:] + p240_c[:100] + p240_c[140:], fl_a + p240_c + fl_b) is True
    for d_fix5 in (10, 30, 49):
        fl_a, fl_b = make_complex_seq(100, seed=917000 + d_fix5), make_complex_seq(100, seed=917100 + d_fix5)
        ad = make_complex_seq(d_fix5, seed=917200 + d_fix5)
        p_near = mutate_seq_snps(_mk_pal(200 - d_fix5, seed=917300 + d_fix5), [5, 15, 25, 35])
        d30_f5 = make_complex_seq(30, seed=917400 + d_fix5)
        ov25 = make_complex_seq(25, seed=917500 + d_fix5)
        if ov25[0] == fl_b[0]:
            ov25 = _flip1(fl_b[0]) + ov25[1:]
        assert (
            check_similarity_merge(
                ad + p_near + ov25,
                fl_a + ad + p_near[:80] + d30_f5 + p_near[80:] + revcomp_dna(ad) + fl_b,
            )
            is True
        )
    # Bridged 45D + pal240, P70_asym + 45D, P80 + 45I + fl(8, 10, 15, 20) -> True
    fl_br_a, fl_br_c = make_complex_seq(80, seed=400001), make_complex_seq(80, seed=402001)
    assert check_similarity_merge(fl_br_a[:35] + pal240, fl_br_a + pal240 + fl_br_c) is True
    p70 = _mk_pal(70, seed=918070)
    u25_as = make_complex_seq(25, seed=918025)
    assert check_similarity_merge(u25_as + p70, fl1 + u25_as + p70[:35] + d45 + p70[35:] + fl2) is True
    p80_c = _mk_pal(80, seed=918080)
    for fl_ex in (8, 10, 15, 20):
        assert check_similarity_merge(p80_c[:40] + i45 + p80_c[40:] + fl2[:fl_ex], fl1 + p80_c + fl2) is True
    # C1-r3 B3 bleed guard -> False
    ov_b3 = fl2[:2] + _flip1(fl2[2]) + fl2[3:8] + _flip1(fl2[8]) + make_complex_seq(12, seed=918999)
    assert check_similarity_merge(pal160 + ov_b3, fl1 + pal160 + fl2) is False

    # --- 2. C1-r5 NEW Bug 3: Exact Copy 1 vs < min_sv_len-indel Copy 2 (K2/K3 fwd & RC) ---
    # Verbatim Challenger 1 (r5) §5 Bug 3a/3b
    u1, u2 = make_complex_seq(90, seed=141001), make_complex_seq(90, seed=141002)
    d_c2, sep40 = make_complex_seq(40, seed=141003), make_complex_seq(40, seed=141004)
    assert check_similarity_merge(u1 + u2, u1 + u2 + sep40 + u1 + d_c2 + u2) is False
    assert check_similarity_merge(u1 + u2, u1 + u2 + sep40 + revcomp_dna(u1 + d_c2 + u2)) is False
    u1i, u2i = make_complex_seq(80, seed=170080), make_complex_seq(80, seed=171080)
    i45_b3, fli = make_complex_seq(45, seed=172045), make_complex_seq(60, seed=174000)
    si = u1i + i45_b3 + u2i
    assert check_similarity_merge(si, fli + si + sep40 + u1i + u2i + fli) is False
    assert check_similarity_merge(si, fli + si + sep40 + revcomp_dna(u1i + u2i) + fli) is False

    # K2 centered (30+45D+30, 120+49D+120, msl=200 180+180D+180) & off-center (19+45D+130, 130+45D+19, 30+45D+130, 130+45D+30)
    for la, ra, dl, msl in (
        (30, 30, 45, 50),
        (120, 120, 49, 50),
        (180, 180, 180, 200),
        (19, 130, 45, 50),
        (130, 19, 45, 50),
        (30, 130, 45, 50),
        (130, 30, 45, 50),
    ):
        a_s = make_complex_seq(la, seed=920000 + la * 10 + ra)
        b_s = make_complex_seq(ra, seed=921000 + la * 10 + ra)
        d_s = make_complex_seq(dl, seed=922000 + la * 10 + ra)
        sp_s = make_complex_seq(40, seed=923000 + la * 10 + ra)
        assert check_similarity_merge(a_s + b_s, a_s + b_s + sp_s + a_s + d_s + b_s, min_sv_len=msl) is False
        assert (
            check_similarity_merge(a_s + b_s, a_s + b_s + sp_s + revcomp_dna(a_s + d_s + b_s), min_sv_len=msl)
            is False
        )

    # K2 with 1-2 SNPs, overhangs, U1==U2, and repetitive AT flanks -> False
    u1_m = mutate_seq_snps(u1, [20, 60])
    assert check_similarity_merge(u1 + u2, u1 + u2 + sep40 + u1_m + d_c2 + u2) is False
    ov12 = make_complex_seq(12, seed=924012)
    assert check_similarity_merge(ov12 + u1 + u2, fli + u1 + u2 + sep40 + u1 + d_c2 + u2 + fli) is False
    assert check_similarity_merge(u1 + u1, u1 + u1 + sep40 + u1 + d_c2 + u1) is False
    u_at_mid = make_complex_seq(150, seed=924150)
    q_at_k2 = "AT" * 12 + u_at_mid + "AT" * 12
    r_at_k2 = q_at_k2 + "AT" * 180 + "AT" * 12 + u_at_mid[:75] + d_c2 + u_at_mid[75:] + "AT" * 12
    assert check_similarity_merge(q_at_k2, r_at_k2) is False

    # K3 centered (60+40I+60, msl=200 120+150I+120) & off-center (35+45I+95, 95+45I+35), fwd & RC -> False
    for la, ra, il, msl in (
        (60, 60, 40, 50),
        (120, 120, 150, 200),
        (35, 95, 45, 50),
        (95, 35, 45, 50),
    ):
        a_s = make_complex_seq(la, seed=925000 + la * 10 + ra)
        b_s = make_complex_seq(ra, seed=926000 + la * 10 + ra)
        i_s = make_complex_seq(il, seed=927000 + la * 10 + ra)
        sp_s = make_complex_seq(40, seed=928000 + la * 10 + ra)
        q_k3 = a_s + i_s + b_s
        assert check_similarity_merge(q_k3, fli + q_k3 + sp_s + a_s + b_s + fli, min_sv_len=msl) is False
        assert check_similarity_merge(q_k3, fli + q_k3 + sp_s + revcomp_dna(a_s + b_s) + fli, min_sv_len=msl) is False

    # R2-r5 F4 cross-polarity: Copy 1 (20D) vs Copy 2 (25I) -> False
    for sd in range(3):
        xa = make_complex_seq(40, seed=929000 + sd)
        xm = make_complex_seq(25, seed=929100 + sd)
        xb = make_complex_seq(40, seed=929200 + sd)
        xd20 = make_complex_seq(20, seed=929300 + sd)
        xsep = make_complex_seq(60, seed=929400 + sd)
        assert check_similarity_merge(xa + xm + xb, xa + xd20 + xm + xb + xsep + xa + xb) is False

    # Stage 6E FP controls: U1-only, tandem/homo flanks, ADV_MS200_SELF_RC_PAL150_AT200 -> True
    unrel_fp = make_complex_seq(180, seed=929500)
    assert check_similarity_merge(u1 + u2, u1 + u2 + sep40 + u1 + unrel_fp) is True
    assert check_similarity_merge(u1 + u2, "A" * 200 + u1 + u2 + "A" * 200) is True
    assert check_similarity_merge(u1 + u2, ("AT" * 100) + u1 + u2 + ("AT" * 100)) is True
    pal150_adv = _mk_pal(150, seed=929600)
    sep250_adv = make_complex_seq(250, seed=929601)
    q_adv200 = "AT" * 36 + pal150_adv
    r_adv200 = (
        "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + pal150_adv + sep250_adv + "AT" * 200
    )
    assert check_similarity_merge(q_adv200, r_adv200, min_sv_len=200) is True

    # --- 3. C1-r5 NEW Bug 4: Stage 6D compound I+D duplicated targets ---
    # Verbatim Challenger 1 (r5) §5 Bug 4
    u1a, u1b, u2d = make_complex_seq(35, seed=40000), make_complex_seq(60, seed=40100), make_complex_seq(60, seed=40200)
    ins30, sepd = make_complex_seq(30, seed=40300), make_complex_seq(60, seed=40400)
    da45 = _mk_biased(45, "AT", 40500)
    dg45 = _mk_biased(45, "GC", 40600)
    assert check_similarity_merge(u1a + ins30 + u1b + u2d, u1a + u1b + da45 + u2d) is True
    assert (
        check_similarity_merge(
            u1a + ins30 + u1b + u2d,
            u1a + u1b + da45 + u2d + sepd + u1a + u1b + dg45 + u2d,
        )
        is False
    )

    # I in (25, 35, 45) fwd & RC -> single-copy True, dup False
    for i_len in (25, 35, 45):
        ins_v = make_complex_seq(i_len, seed=930000 + i_len)
        q_v = u1a + ins_v + u1b + u2d
        c1_v = u1a + u1b + da45 + u2d
        c2_v = u1a + u1b + dg45 + u2d
        assert check_similarity_merge(q_v, c1_v) is True
        assert check_similarity_merge(q_v, c1_v + sepd + c2_v) is False
        assert check_similarity_merge(q_v, c1_v + sepd + revcomp_dna(c2_v)) is False

    # Reversed D + I, I + 2xD (4 segs), and short u1a=15 -> single-copy True, dup False
    q_rev = u2d + u1b + ins30 + u1a
    c1_rev = u2d + da45 + u1b + u1a
    c2_rev = u2d + dg45 + u1b + u1a
    assert check_similarity_merge(q_rev, c1_rev) is True
    assert check_similarity_merge(q_rev, c1_rev + sepd + c2_rev) is False

    u3d = make_complex_seq(50, seed=931000)
    da22a, da22b = _mk_biased(22, "AT", 931100), _mk_biased(22, "AT", 931101)
    dg22a, dg22b = _mk_biased(22, "GC", 931200), _mk_biased(22, "GC", 931201)
    q_4seg = u1a + ins30 + u1b + u2d + u3d
    c1_4seg = u1a + u1b + da22a + u2d + da22b + u3d
    c2_4seg = u1a + u1b + dg22a + u2d + dg22b + u3d
    assert check_similarity_merge(q_4seg, c1_4seg) is True
    assert check_similarity_merge(q_4seg, c1_4seg + sepd + c2_4seg) is False

    u1a_15 = make_complex_seq(15, seed=932000)
    ins35 = make_complex_seq(35, seed=932001)
    q_sh15 = u1a_15 + ins35 + u1b + u2d
    c1_sh15 = u1a_15 + u1b + da45 + u2d
    c2_sh15 = u1a_15 + u1b + dg45 + u2d
    assert check_similarity_merge(q_sh15, c1_sh15) is True
    assert check_similarity_merge(q_sh15, c1_sh15 + sepd + c2_sh15) is False


def test_rf3_wave5_stage6d_subsplit_fwd_ov_periodic_and_latency_bounds() -> None:
    """Wave 5 / Iteration 6 Group C regression guard: R2-r5 F1/F2/F3, C1-r5 Bug 5, determinism, latency & bonus F4."""
    import random
    import statistics
    import time

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    # --- 1. R2-r5 F2: Reachable 1-level allow_subsplit (msl=200 & msl=50, fwd & bwd) ---
    # Verbatim Reviewer 2 (r5) §5.1 (msl=200 tandem 25D+25D vs 45D, fwd & bwd)
    c1 = make_complex_seq(50, seed=310000)
    c2 = make_complex_seq(50, seed=311000)
    c3 = make_complex_seq(50, seed=312000)
    d2a, d2b = _mk_biased(25, "AT", 313000), _mk_biased(25, "AT", 314000)
    d1 = _mk_biased(45, "GC", 315000)
    assert check_similarity_merge(c1 + c2 + c3, c1 + d2a + c2 + d2b + c3 + c1 + d1 + c2 + c3, min_sv_len=200) is False
    assert check_similarity_merge(c2 + c3 + c1, c2 + c3 + d1 + c1 + c2 + d2a + c3 + d2b + c1, min_sv_len=200) is False
    # Default msl=50 (18D+18D vs 34D, fwd & bwd)
    for sd in range(4):
        a1 = make_complex_seq(40, seed=940000 + sd)
        a2 = make_complex_seq(40, seed=941000 + sd)
        a3 = make_complex_seq(40, seed=942000 + sd)
        da18a, da18b = _mk_biased(18, "AT", 943000 + sd), _mk_biased(18, "AT", 944000 + sd)
        dg34 = _mk_biased(34, "GC", 945000 + sd)
        assert check_similarity_merge(a1 + a2 + a3, a1 + da18a + a2 + da18b + a3 + a1 + dg34 + a2 + a3) is False
        assert check_similarity_merge(a2 + a3 + a1, a2 + a3 + dg34 + a1 + a2 + da18a + a3 + da18b + a1) is False

    # --- 2. R2-r5 F3: Forward-strand ov_hits shifted-slice fallback (U1+D1+U1 L/R/swapped/3-arm) ---
    for sd in range(6):
        u1 = make_complex_seq(50, seed=215000 + 50 * 100 + sd)
        ua = mutate_seq_snps(u1, [10])
        ub = mutate_seq_snps(u1, [10, 30])
        sep60 = make_complex_seq(60, seed=225000 + sd)
        d1_r = make_complex_seq(49, seed=275000 + sd)
        d2_gc = _mk_biased(49, "GGCC", 285000 + sd)
        d2_rand = make_complex_seq(49, seed=295000 + sd)
        # Verbatim Reviewer 2 (r5) §5.1 right copy (rand/GC & pure rand)
        assert check_similarity_merge(u1 + u1, u1 + d1_r + u1 + sep60 + ua + d2_gc + ub) is False
        assert check_similarity_merge(u1 + u1, u1 + d1_r + u1 + sep60 + ua + d2_rand + ub) is False
        # Mirror: Copy 2 on left
        assert check_similarity_merge(u1 + u1, ua + d2_gc + ub + sep60 + u1 + d1_r + u1) is False
        # Swapped 2-SNP / 1-SNP arms
        assert check_similarity_merge(u1 + u1, u1 + d1_r + u1 + sep60 + ub + d2_gc + ua) is False
    # 3-arm U1+24D+U1+24D+U1
    u1_3 = make_complex_seq(45, seed=946000)
    u1_3a, u1_3b = mutate_seq_snps(u1_3, [10]), mutate_seq_snps(u1_3, [10, 25])
    sep60_3 = make_complex_seq(60, seed=946001)
    r_3arm = (
        u1_3
        + make_complex_seq(24, seed=946002)
        + u1_3
        + make_complex_seq(24, seed=946003)
        + u1_3
        + sep60_3
        + u1_3a
        + _mk_biased(24, "GC", 946004)
        + u1_3b
        + _mk_biased(24, "GC", 946005)
        + u1_3a
    )
    assert check_similarity_merge(u1_3 + u1_3 + u1_3, r_3arm) is False

    # --- 3. C1-r5 NEW Bug 5: Periodic/microsat arm tol=8 cluster-bound re-evaluation ---
    # Verbatim Challenger 1 (r5) §5 Bug 5 (motif_len=4, D=49)
    motif4 = make_complex_seq(4, seed=50024)
    u1_per4 = (motif4 * 13)[:50]
    u2_e1 = make_complex_seq(50, seed=52002)
    r_e1 = (
        u1_per4
        + _mk_biased(49, "AT", 53002)
        + u2_e1
        + make_complex_seq(60, seed=55002)
        + motif4
        + u1_per4
        + _mk_biased(49, "GC", 54002)
        + u2_e1
    )
    assert check_similarity_merge(u1_per4 + u2_e1, r_e1) is False

    # motif_len=2..8 x D in (44, 48, 49) across L, R (0 & 1 SNP), both, 3-seg, I-junction + single-copy FP controls
    motifs_l = {2: "GC", 3: "GCT", 4: "GCAT", 5: "GCCGT", 6: "GCTCGA", 7: "GCCGTCG", 8: "GCTCGACG"}
    motifs_r = {2: "CT", 3: "CGT", 4: "CCGT", 5: "CCGCT", 6: "CCGCGT", 7: "CCCTACG", 8: "CCCTACGT"}
    for m_len in range(2, 9):
        motif = motifs_l[m_len]
        motif_b = motifs_r[m_len]
        u_per = (motif * (50 // m_len + 2))[:50]
        u_per_b = (motif_b * (50 // m_len + 2))[:50]
        u_non = make_complex_seq(50, seed=951000 + m_len * 100)
        sep_p = make_complex_seq(60, seed=952000 + m_len * 100)
        for d_len in (44, 48, 49):
            da = _mk_biased(d_len, "AT", 953000 + m_len * 100 + d_len)
            dg = _mk_biased(d_len, "GC", 954000 + m_len * 100 + d_len)
            # Single-copy controls (proving non-vacuity) -> True
            assert check_similarity_merge(u_per + u_non, u_per + da + u_non) is True
            assert check_similarity_merge(u_non + u_per, u_non + da + u_per) is True
            assert check_similarity_merge(u_per + u_per_b, u_per + da + u_per_b) is True
            # Left periodic -> False
            assert check_similarity_merge(u_per + u_non, u_per + da + u_non + sep_p + motif + u_per + dg + u_non) is False
            # Right periodic mirror (0 SNP & 1 SNP) -> False
            assert check_similarity_merge(u_non + u_per, u_non + da + u_per + sep_p + u_non + dg + u_per + motif) is False
            u_per_1 = mutate_seq_snps(u_per, [25])
            assert (
                check_similarity_merge(u_non + u_per, u_non + da + u_per + sep_p + u_non + dg + u_per_1 + motif)
                is False
            )
            # Both periodic -> False
            assert (
                check_similarity_merge(
                    u_per + u_per_b,
                    u_per + da + u_per_b + sep_p + motif + u_per + dg + u_per_b + motif_b,
                )
                is False
            )
        # 3-seg cumulative 24D+24D with periodic middle arm -> False
        u_mid_c = make_complex_seq(50, seed=956000 + m_len)
        da24a, da24b = _mk_biased(24, "AT", 956100 + m_len), _mk_biased(24, "AT", 956200 + m_len)
        dg24a, dg24b = _mk_biased(24, "GC", 956300 + m_len), _mk_biased(24, "GC", 956400 + m_len)
        assert (
            check_similarity_merge(
                u_per + u_non + u_mid_c,
                u_per + da24a + u_non + da24b + u_mid_c + sep_p + motif + u_per + dg24a + u_non + dg24b + u_mid_c,
            )
            is False
        )
        # Periodic arm next to 'I' junction -> False
        ins25_p = make_complex_seq(25, seed=956500 + m_len)
        assert (
            check_similarity_merge(
                u_per + ins25_p + u_non + u_mid_c,
                u_per + u_non + da + u_mid_c + sep_p + motif + u_per + u_non + dg + u_mid_c,
            )
            is False
        )

    # --- 4. R2-r5 F1: Recall with AT*200 distractors + determinism + latency bounds ---
    u_r1 = make_complex_seq(50, seed=960001)
    u_r2 = make_complex_seq(50, seed=960002)
    u_r3 = make_complex_seq(50, seed=960003)
    sep_r = make_complex_seq(40, seed=960004)
    at_dist = "AT" * 100
    # 4-seg recall with AT*200 distractor before & after Copy 2
    q_dist4 = "AT" * 12 + "AT" * 12 + "AT" * 12 + u_r1
    c1_d4 = "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u_r1
    c2_d4 = "AT" * 12 + "T" * 8 + "AT" * 12 + "A" * 8 + "AT" * 12 + "T" * 8 + u_r1
    assert check_similarity_merge(q_dist4, c1_d4 + sep_r + c2_d4 + sep_r + at_dist) is False
    assert check_similarity_merge(q_dist4, c1_d4 + sep_r + at_dist + sep_r + c2_d4) is False
    # 3-seg allow_subsplit recall after AT*200 distractor
    c1_d3 = "AT" * 12 + "C" * 15 + u_r2 + u_r3
    c2_d3 = "AT" * 12 + "G" * 15 + u_r2 + "T" * 15 + u_r3
    assert check_similarity_merge("AT" * 12 + u_r2 + u_r3, c1_d3 + sep_r + at_dist + sep_r + c2_d3) is False

    # Determinism & CI-safe (< 15 ms) latency checks across all key 1 kb worst-case bombs
    u150 = make_complex_seq(150, seed=930001)
    sep80 = make_complex_seq(80, seed=930002)
    u250 = make_complex_seq(250, seed=930003)
    u150_b = make_complex_seq(150, seed=930004)
    r950_np = make_complex_seq(950, seed=930007)
    u200_at = "AT" * 10 + make_complex_seq(160, seed=930008) + "AT" * 10
    r950_at = "AT" * 180 + u200_at + "AT" * 195

    latency_cases: list[tuple[str, str, str, int]] = [
        (
            "REAL_BOMB_AT24_3x8D_u150",
            "AT" * 36 + u150,
            "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u150 + sep80 + "AT" * 300,
            50,
        ),
        (
            "REAL_BOMB_A24_3x8D_u150",
            "A" * 72 + u150,
            "A" * 24 + "C" * 8 + "A" * 24 + "G" * 8 + "A" * 24 + "C" * 8 + u150 + sep80 + "A" * 600,
            50,
        ),
        (
            "BOMB2_AT24_3x8D_u250_msl200",
            "AT" * 36 + u250,
            "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u250 + sep80 + "AT" * 250,
            200,
        ),
        (
            "REAL_BOMB_3SEG_AT25_2x15D",
            "AT" * 12 + "A" + "AT" * 12 + "A" + u150_b,
            "AT" * 12 + "A" + "C" * 15 + "AT" * 12 + "A" + "G" * 15 + u150_b + sep80 + "AT" * 300,
            50,
        ),
        (
            "ADV_ZEROSEP_AT24_LEFT_TSTART",
            "AT" * 36 + u150,
            "AT" * 300 + "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u150,
            50,
        ),
        (
            "ADV_COMPOUND_ID_4SEG_AT300",
            "AT" * 36 + u150[:75] + make_complex_seq(8, seed=930009) + u150[75:],
            "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + u150 + sep80 + "AT" * 300,
            50,
        ),
        (
            "950bp_1snp_nonpal",
            mutate_seq_snps(r950_np, [475]),
            r950_np,
            50,
        ),
        (
            "200bp_ATflank_in_950bp_AT",
            u200_at,
            r950_at,
            50,
        ),
    ]
    for label, q_lat, r_lat, msl in latency_cases:
        results = []
        times_ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = check_similarity_merge(q_lat, r_lat, min_sv_len=msl)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append(res)
        assert results == [True, True, True], f"{label} expected deterministic True, got {results}"
        assert statistics.median(times_ms) < 15.0, f"{label} median {statistics.median(times_ms):.2f} ms >= 15 ms"

    # Bonus closed R2-r5 F4 checks: TinyMid_14 (0-SNP), short_outer_arm17, 4x12D (5-arm AT/GC) -> False
    tm_l, tm_m, tm_r = make_complex_seq(35, seed=970001), make_complex_seq(14, seed=970002), make_complex_seq(35, seed=970003)
    assert (
        check_similarity_merge(
            tm_l + tm_m + tm_r,
            tm_l
            + _mk_biased(24, "AT", 970004)
            + tm_m
            + _mk_biased(24, "AT", 970005)
            + tm_r
            + sep_r
            + tm_l
            + _mk_biased(24, "GC", 970006)
            + tm_m
            + _mk_biased(24, "GC", 970007)
            + tm_r,
        )
        is False
    )
    sh17, sh50 = make_complex_seq(17, seed=970101), make_complex_seq(50, seed=970102)
    assert (
        check_similarity_merge(
            sh17 + sh50,
            sh17
            + _mk_biased(45, "AT", 970103)
            + sh50
            + sep_r
            + sh17
            + _mk_biased(45, "GC", 970104)
            + sh50,
        )
        is False
    )
    arms5 = [make_complex_seq(25, seed=970200 + i) for i in range(5)]
    c1_5 = arms5[0] + _mk_biased(12, "AT", 970210) + arms5[1] + _mk_biased(12, "AT", 970211) + arms5[2] + _mk_biased(12, "AT", 970212) + arms5[3] + _mk_biased(12, "AT", 970213) + arms5[4]
    c2_5 = arms5[0] + _mk_biased(12, "GC", 970220) + arms5[1] + _mk_biased(12, "GC", 970221) + arms5[2] + _mk_biased(12, "GC", 970222) + arms5[3] + _mk_biased(12, "GC", 970223) + arms5[4]
    assert check_similarity_merge("".join(arms5), c1_5 + sep_r + c2_5) is False


def test_rf3_wave6_stage4_outer_mid_pigeonhole_and_stage6d_fairness() -> None:
    """Wave 6 / Iteration 7 Group 1 + SA-4 + SA-5 regression guard: C1-r6 Bug 1, R2-r6 Minor F3, FP/Red Line 3 & latency."""
    import random
    import statistics
    import time

    from panflat.repeat_guard import _has_distinct_edlib_del_split_copy

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    def _mk_at_rich(n: int, seed: int, at_prob: float) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice("AT") if rng.random() < at_prob else rng.choice("CG") for _ in range(n))

    def _flip1(ch: str) -> str:
        return "A" if ch != "A" else "C"

    # --- 1. Verbatim C1-r6 NEW Bug 1 (Group A Stage 4 outer-19bp split guard) ---
    fl3 = make_complex_seq(80, seed=771000)
    aL3, aR3 = make_complex_seq(30, seed=772030), make_complex_seq(30, seed=773030)
    d1_3, d2_3 = make_complex_seq(45, seed=774045), make_complex_seq(45, seed=775045)
    d1_100 = make_complex_seq(100, seed=794100)
    core3 = make_complex_seq(240, seed=776000)
    ins15 = make_complex_seq(15, seed=778015)
    assert check_similarity_merge(aL3 + core3 + ins15 + aR3, fl3 + aL3 + d1_3 + core3 + d2_3 + aR3 + fl3) is False
    assert check_similarity_merge(aL3 + ins15 + core3 + aR3, fl3 + aL3 + d1_100 + core3 + d2_3 + aR3 + fl3) is False

    # Sweep ins in (5, 10, 15, 25) and arm in (19, 25, 30) on both R (90D) and L (145D/105D) -> False
    for ins_l in (5, 10, 15, 25):
        for sd in range(3):
            fl = make_complex_seq(80, seed=981000 + ins_l * 10 + sd)
            al = make_complex_seq(30, seed=981100 + ins_l * 10 + sd)
            ar = make_complex_seq(30, seed=981200 + ins_l * 10 + sd)
            dd1 = make_complex_seq(45, seed=981300 + ins_l * 10 + sd)
            dd2 = make_complex_seq(45, seed=981400 + ins_l * 10 + sd)
            dd1_l = make_complex_seq(100, seed=981500 + ins_l * 10 + sd)
            cr = make_complex_seq(240, seed=981600 + ins_l * 10 + sd)
            ins_s = make_complex_seq(ins_l, seed=981700 + ins_l * 10 + sd)
            assert check_similarity_merge(al + cr + ins_s + ar, fl + al + dd1 + cr + dd2 + ar + fl) is False
            assert check_similarity_merge(al + ins_s + cr + ar, fl + al + dd1_l + cr + dd2 + ar + fl) is False

    for arm_l in (19, 25, 30):
        for sd in range(3):
            fl = make_complex_seq(80, seed=982000 + arm_l * 10 + sd)
            al = make_complex_seq(arm_l, seed=982100 + arm_l * 10 + sd)
            ar = make_complex_seq(arm_l, seed=982200 + arm_l * 10 + sd)
            dd1 = make_complex_seq(45, seed=982300 + arm_l * 10 + sd)
            dd2 = make_complex_seq(45, seed=982400 + arm_l * 10 + sd)
            dd1_l = make_complex_seq(60, seed=982500 + arm_l * 10 + sd)
            cr = make_complex_seq(240, seed=982600 + arm_l * 10 + sd)
            ins_s = make_complex_seq(15, seed=982700 + arm_l * 10 + sd)
            assert check_similarity_merge(al + cr + ins_s + ar, fl + al + dd1 + cr + dd2 + ar + fl) is False
            assert check_similarity_merge(al + ins_s + cr + ar, fl + al + dd1_l + cr + dd2 + ar + fl) is False

    # 1SNP_inner on outer arm (arm=30, ins=15, 90D R) -> False
    for sd in range(4):
        fl = make_complex_seq(80, seed=983000 + sd)
        al = make_complex_seq(30, seed=983100 + sd)
        ar = make_complex_seq(30, seed=983200 + sd)
        dd1 = make_complex_seq(45, seed=983300 + sd)
        dd2 = make_complex_seq(45, seed=983400 + sd)
        cr = make_complex_seq(240, seed=983500 + sd)
        ins_s = make_complex_seq(15, seed=983600 + sd)
        ar_inner_1snp = mutate_seq_snps(ar, [4])
        assert check_similarity_merge(al + cr + ins_s + ar_inner_1snp, fl + al + dd1 + cr + dd2 + ar + fl) is False

    # Custom min_sv_len in (20, 100, 200) -> False on >= msl dual DEL + ins, True on < msl control
    for msl, dl, ins_l, arm_l in ((20, 15, 8, 19), (100, 80, 25, 30), (200, 150, 35, 35)):
        fl = make_complex_seq(60, seed=984000 + msl)
        al = make_complex_seq(arm_l, seed=984100 + msl)
        ar = make_complex_seq(arm_l, seed=984200 + msl)
        dd1 = make_complex_seq(dl, seed=984300 + msl)
        dd2 = make_complex_seq(dl, seed=984400 + msl)
        cr = make_complex_seq(220, seed=984500 + msl)
        ins_s = make_complex_seq(ins_l, seed=984600 + msl)
        assert (
            check_similarity_merge(al + cr + ins_s + ar, fl + al + dd1 + cr + dd2 + ar + fl, min_sv_len=msl)
            is False
        )
        ok_d = (msl - 2) // 2
        assert (
            check_similarity_merge(
                al + cr + ar,
                fl + al + dd1[:ok_d] + cr + dd2[:ok_d] + ar + fl,
                min_sv_len=msl,
            )
            is True
        )

    # SA-4: Symmetric & asymmetric ov + arm19 + ins (L & R) -> all False
    for ins_l, ov_l in ((10, 10), (18, 6), (6, 18), (15, 8), (20, 5)):
        for sd in range(3):
            fl = make_complex_seq(80, seed=985000 + ins_l * 100 + ov_l * 10 + sd)
            al = make_complex_seq(30, seed=985100 + ins_l * 100 + ov_l * 10 + sd)
            ar19 = make_complex_seq(19, seed=985200 + ins_l * 100 + ov_l * 10 + sd)
            dd1 = make_complex_seq(45, seed=985300 + ins_l * 100 + ov_l * 10 + sd)
            dd2 = make_complex_seq(45, seed=985400 + ins_l * 100 + ov_l * 10 + sd)
            dd1_l = make_complex_seq(100, seed=985500 + ins_l * 100 + ov_l * 10 + sd)
            cr = make_complex_seq(240, seed=985600 + ins_l * 100 + ov_l * 10 + sd)
            ins_s = make_complex_seq(ins_l, seed=985700 + ins_l * 100 + ov_l * 10 + sd)
            ov_s = make_complex_seq(ov_l, seed=985800 + ins_l * 100 + ov_l * 10 + sd)
            assert check_similarity_merge(al + cr + ins_s + ar19 + ov_s, fl + al + dd1 + cr + dd2 + ar19 + fl) is False
            assert check_similarity_merge(ov_s + ar19 + ins_s + cr + al, fl + ar19 + dd1_l + cr + dd2 + al + fl) is False

    # --- 2. Stage 4 FP & Red Line 3 controls (MUST remain True) ---
    # 0D+48D+ins15+arm19 & 0D+48D+ov10+arm19+ins10 -> True
    for sd in (0, 2, 3, 4, 7):
        fl = make_complex_seq(80, seed=986000 + sd)
        cr = make_complex_seq(240, seed=986100 + sd)
        d48 = make_complex_seq(48, seed=986200 + sd)
        ar19 = make_complex_seq(19, seed=986300 + sd)
        ins15_c = make_complex_seq(15, seed=986400 + sd)
        ins10_c = make_complex_seq(10, seed=986500 + sd)
        ov10_c = make_complex_seq(10, seed=986600 + sd)
        assert check_similarity_merge(cr + ins15_c + ar19, fl + cr + d48 + ar19 + fl) is True
        assert check_similarity_merge(cr + ins10_c + ar19 + ov10_c, fl + cr + d48 + ar19 + fl) is True

    # SA-4 inward micro-indel bridge FP control: 24D+23D+ins15+arm30_1bpDel@5 (total DEL = 48 < 50) -> True
    for sd in (0, 1, 2, 3, 5):
        fl = make_complex_seq(80, seed=986700 + sd)
        al = make_complex_seq(30, seed=986710 + sd)
        d24 = make_complex_seq(24, seed=986720 + sd)
        cr = make_complex_seq(240, seed=986730 + sd)
        d23 = make_complex_seq(23, seed=986740 + sd)
        ar30 = make_complex_seq(30, seed=986750 + sd)
        ins15_m = make_complex_seq(15, seed=986760 + sd)
        ar30_1del = ar30[:5] + ar30[6:]
        assert check_similarity_merge(al + cr + ins15_m + ar30_1del, fl + al + d24 + cr + d23 + ar30 + fl) is True

    # P1-G tandem flanks (L & R) and P1-B 38D+10D=48D (L & R) -> True
    for sd in range(4):
        fl60 = make_complex_seq(60, seed=986800 + sd)
        u20 = make_complex_seq(20, seed=986810 + sd)
        md150 = make_complex_seq(150, seed=986820 + sd)
        d45_g = make_complex_seq(45, seed=986830 + sd)
        tail50 = make_complex_seq(50, seed=986840 + sd)
        ov10_g = make_complex_seq(10, seed=986850 + sd)
        r_p1g_l = fl60 + u20 + md150 + d45_g + tail50 + fl60
        r_p1g_r = fl60 + tail50 + d45_g + md150 + u20 + fl60
        assert check_similarity_merge(u20 + u20 + md150 + tail50, r_p1g_l) is True
        assert check_similarity_merge(ov10_g + u20 + u20 + md150 + tail50, r_p1g_l) is True
        assert check_similarity_merge(tail50 + md150 + u20 + u20, r_p1g_r) is True
        assert check_similarity_merge(tail50 + md150 + u20 + u20 + ov10_g, r_p1g_r) is True

        arm20 = make_complex_seq(20, seed=986860 + sd)
        d10 = make_complex_seq(10, seed=986870 + sd)
        md100 = make_complex_seq(100, seed=986880 + sd)
        d38 = make_complex_seq(38, seed=986890 + sd)
        a45 = make_complex_seq(45, seed=986900 + sd)
        arm20_bl = arm20[:10] + _flip1(arm20[10]) + arm20[11:18] + d10[:2]
        arm20_br = d10[-2:] + arm20[2:10] + _flip1(arm20[10]) + arm20[11:]
        assert check_similarity_merge(arm20_bl + md100 + a45, fl60 + arm20 + d10 + md100 + d38 + a45) is True
        assert check_similarity_merge(a45 + md100 + arm20_br, a45 + d38 + md100 + d10 + arm20 + fl60) is True

    # 70% and 83% AT novel overhangs (ov=40 + 45D, fl700, L & R) -> True
    for at_prob, tag_off in ((0.70, 0), (0.83, 100)):
        for sd in range(5):
            fl_at_l = _mk_at_rich(350, seed=987000 + tag_off + sd, at_prob=at_prob)
            fl_at_r = _mk_at_rich(350, seed=987050 + tag_off + sd, at_prob=at_prob)
            body = make_complex_seq(220 + 45, seed=987200 + tag_off + sd)
            ov40 = _mk_at_rich(40, seed=987300 + tag_off + sd, at_prob=at_prob)
            s_body = body[:110] + body[155:]
            assert check_similarity_merge(ov40 + s_body, fl_at_l + body + fl_at_r) is True
            assert check_similarity_merge(s_body + ov40, fl_at_l + body + fl_at_r) is True

    # Red Line 3 explicit guards: 200=30I200=, 225=49I225=, 48D, 49D -> True; 50I/50D -> False
    r400 = make_complex_seq(400, seed=987401)
    assert check_similarity_merge(r400[:200] + make_complex_seq(30, seed=987402) + r400[200:], r400) is True
    r450 = make_complex_seq(450, seed=987403)
    assert check_similarity_merge(r450[:225] + make_complex_seq(49, seed=987404) + r450[225:], r450) is True
    assert check_similarity_merge(r450[:225] + make_complex_seq(50, seed=987405) + r450[225:], r450) is False
    assert check_similarity_merge(r450[:200] + r450[248:], r450) is True
    assert check_similarity_merge(r450[:200] + r450[249:], r450) is True
    assert check_similarity_merge(r450[:200] + r450[250:], r450) is False

    # --- 3. Verbatim R2-r6 Minor F3: 10/10 direct Stage 6D per-cluster val_budget fairness ---
    for sd in range(10):
        m1 = make_complex_seq(10, seed=620000 + sd)
        m2 = make_complex_seq(10, seed=621000 + sd)
        m3 = make_complex_seq(10, seed=622000 + sd)
        b1, b2, b3 = m1 * 3, m2 * 3, m3 * 3
        s_vb = b1 + b2 + b3
        c1_vb = b1 + _mk_biased(24, "AT", 623000 + sd) + b2 + _mk_biased(24, "AT", 624000 + sd) + b3
        c2_vb = b1 + _mk_biased(24, "GC", 625000 + sd) + b2 + _mk_biased(24, "GC", 626000 + sd) + b3
        dist_vb = m1 * 6 + make_complex_seq(12, seed=627000 + sd) + m2 * 6 + make_complex_seq(12, seed=628000 + sd) + m3 * 6
        sep60_vb = make_complex_seq(60, seed=629000 + sd)
        r_vb = c1_vb + sep60_vb + dist_vb + sep60_vb + c2_vb + sep60_vb + dist_vb
        ops_vb = [("M", 30), ("D", 24), ("M", 30), ("D", 24), ("M", 30)]
        assert _has_distinct_edlib_del_split_copy(s_vb, r_vb, ops_vb, 0, len(c1_vb), 50, 19) is True

    # --- 4. SA-5: CI-safe deterministic latency on Stage 6D worst cases (< 15 ms CI assert) ---
    u60_d = make_complex_seq(60, seed=988001)
    sep80_d = make_complex_seq(80, seed=988002)
    u60_dec8 = mutate_seq_snps(u60_d, list(range(3, 56, 7)))
    q_def1ab = "AT" * 36 + u60_d
    r_def1ab = (
        "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u60_d
        + sep80_d
        + "AT" * 110
        + ("G" * 55 + u60_dec8) * 3
    )
    assert len(q_def1ab) == 132 and len(r_def1ab) == 801
    u150_l = make_complex_seq(150, seed=930001)
    sep80_l = make_complex_seq(80, seed=930002)
    u250_l = make_complex_seq(250, seed=930003)

    s6d_lat_cases: list[tuple[str, str, str, int]] = [
        ("S6D_4seg_defeat_Def1ab_AT", q_def1ab, r_def1ab, 50),
        (
            "REAL_BOMB_AT24_3x8D_u150",
            "AT" * 36 + u150_l,
            "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u150_l + sep80_l + "AT" * 300,
            50,
        ),
        (
            "BOMB2_AT24_3x8D_u250_msl200",
            "AT" * 36 + u250_l,
            "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u250_l + sep80_l + "AT" * 250,
            200,
        ),
        (
            "ADV_ZEROSEP_AT24_LEFT_TSTART",
            "AT" * 36 + u150_l,
            "AT" * 300 + "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + u150_l,
            50,
        ),
    ]
    for label, q_lat, r_lat, msl in s6d_lat_cases:
        results = []
        times_ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = check_similarity_merge(q_lat, r_lat, min_sv_len=msl)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append(res)
        assert results == [True, True, True], f"{label} expected deterministic True, got {results}"
        assert statistics.median(times_ms) < 15.0, f"{label} median {statistics.median(times_ms):.2f} ms >= 15 ms"


def test_rf3_wave6_stage6e_unified_recall_fairness_and_latency() -> None:
    """Wave 6 / Iteration 7 Group 2 + SA-1..SA-3 regression guard: R1-r6 F1/F2, R2-r6 F1/F2, C1-r6 Bugs 2 & 3, FP & latency."""
    import statistics
    import time

    def _mk_pal(p_len: int, seed: int) -> str:
        h = make_complex_seq(p_len // 2, seed=seed)
        return h + ("A" if p_len % 2 else "") + revcomp_dna(h)

    def _flip1(ch: str) -> str:
        return "A" if ch != "A" else "C"

    # --- 1. Verbatim R1-r6 Major F1 §5.1 (8/8 fwd & RC) + double-ended + off-center + msl=100 ---
    u1_r1 = make_complex_seq(120, seed=771000)
    u2_r1 = make_complex_seq(120, seed=772000)
    d49_r1 = make_complex_seq(49, seed=773000)
    sep_r1 = make_complex_seq(40, seed=774000)
    s_r1 = u1_r1 + u2_r1
    k2_cases = [
        ("K2_trunc10_L", u1_r1[10:] + d49_r1 + u2_r1),
        ("K2_trunc15_R", u1_r1 + d49_r1 + u2_r1[:-15]),
        ("K2_1x3bp_del_tip19", u1_r1[:8] + u1_r1[11:] + d49_r1 + u2_r1),
        ("K2_1x3bp_ins_tip19", u1_r1[:8] + "ACG" + u1_r1[8:] + d49_r1 + u2_r1),
        ("K2_3snp_tip19_L", mutate_seq_snps(u1_r1, [2, 8, 14]) + d49_r1 + u2_r1),
    ]
    for name, c2 in k2_cases:
        assert check_similarity_merge(s_r1, c2) is True, f"Invalid control {name}"
        assert check_similarity_merge(s_r1, s_r1 + sep_r1 + c2) is False, f"Bypass fwd {name}"
        assert check_similarity_merge(s_r1, s_r1 + sep_r1 + revcomp_dna(c2)) is False, f"Bypass RC {name}"

    u1k = make_complex_seq(70, seed=881000)
    u2k = make_complex_seq(70, seed=882000)
    i35k = make_complex_seq(35, seed=883000)
    i40k = make_complex_seq(40, seed=883000)
    sepk = make_complex_seq(40, seed=884000)
    flk = make_complex_seq(50, seed=885000)
    for name, sk, c2k in (
        ("K3_I35_trunc10_L", u1k + i35k + u2k, u1k[10:] + u2k),
        ("K3_I40_3bp_del_tip", u1k + i40k + u2k, u1k[:8] + u1k[11:] + u2k),
        ("K3_I40_3snp_tip", u1k + i40k + u2k, mutate_seq_snps(u1k, [2, 8, 14]) + u2k),
    ):
        assert check_similarity_merge(sk, flk + c2k + flk) is True, f"Invalid control {name}"
        assert check_similarity_merge(sk, flk + sk + sepk + c2k + flk) is False, f"Bypass fwd {name}"
        assert check_similarity_merge(sk, flk + sk + sepk + revcomp_dna(c2k) + flk) is False, f"Bypass RC {name}"

    # Double-ended 24L+24R+49D (q=240 & 340), off-center K2 (85/155, trunc15_L), and msl=100 (300/400 bp, D=95)
    for q_len in (240, 340):
        half = q_len // 2
        ua = make_complex_seq(half, seed=990000 + q_len)
        ub = make_complex_seq(half, seed=990100 + q_len)
        dd49 = make_complex_seq(49, seed=990200 + q_len)
        sp40 = make_complex_seq(40, seed=990300 + q_len)
        sq = ua + ub
        c2_de = ua[24:] + dd49 + ub[:-24]
        assert check_similarity_merge(sq, c2_de) is True
        assert check_similarity_merge(sq, sq + sp40 + c2_de) is False
        assert check_similarity_merge(sq, sq + sp40 + revcomp_dna(c2_de)) is False

    u85 = make_complex_seq(85, seed=990400)
    u155 = make_complex_seq(155, seed=990401)
    d49_oc = make_complex_seq(49, seed=990402)
    s_oc = u85 + u155
    c2_oc = u85[15:] + d49_oc + u155
    assert check_similarity_merge(s_oc, c2_oc) is True
    assert check_similarity_merge(s_oc, s_oc + sep_r1 + c2_oc) is False
    assert check_similarity_merge(s_oc, s_oc + sep_r1 + revcomp_dna(c2_oc)) is False

    for q_m100, tr in ((300, 10), (400, 15)):
        half = q_m100 // 2
        ua = make_complex_seq(half, seed=990500 + q_m100)
        ub = make_complex_seq(half, seed=990600 + q_m100)
        d95 = make_complex_seq(95, seed=990700 + q_m100)
        sq = ua + ub
        c2_m = ua[tr:] + d95 + ub
        assert check_similarity_merge(sq, c2_m, min_sv_len=100) is True
        assert check_similarity_merge(sq, sq + sep_r1 + c2_m, min_sv_len=100) is False
        assert check_similarity_merge(sq, sq + sep_r1 + revcomp_dna(c2_m), min_sv_len=100) is False

    # --- 2. Verbatim C1-r6 NEW Bug 2: Case B-L/R, Case C, LTR (msl=50/100) + 120 bp U1 interior-half decoy ---
    at20 = "AT" * 10
    u1_2, u2_2 = make_complex_seq(120, seed=614120), make_complex_seq(120, seed=615120)
    d49_2 = make_complex_seq(49, seed=611049)
    sep_2 = make_complex_seq(35, seed=612000)
    fl_2 = make_complex_seq(35, seed=613000)
    # Case B-L (msl=50)
    s_cb_l = u1_2 + u2_2[:100] + at20
    c2_cb_l = mutate_seq_snps(u1_2, [8]) + d49_2 + u2_2[:100] + at20
    assert check_similarity_merge(s_cb_l, fl_2 + c2_cb_l + fl_2) is True
    assert check_similarity_merge(s_cb_l, fl_2 + s_cb_l + u1_2[:20] + sep_2 + c2_cb_l + fl_2) is False
    assert check_similarity_merge(s_cb_l, fl_2 + s_cb_l + revcomp_dna(u1_2[:20]) + sep_2 + revcomp_dna(c2_cb_l) + fl_2) is False
    # Case B-R (msl=50)
    s_cb_r = at20 + u1_2[:100] + u2_2
    c2_cb_r = at20 + u1_2[:100] + d49_2 + mutate_seq_snps(u2_2, [112])
    assert check_similarity_merge(s_cb_r, fl_2 + c2_cb_r + fl_2) is True
    assert check_similarity_merge(s_cb_r, fl_2 + s_cb_r + sep_2 + c2_cb_r + u2_2[-20:] + fl_2) is False
    assert check_similarity_merge(s_cb_r, fl_2 + s_cb_r + sep_2 + revcomp_dna(c2_cb_r) + revcomp_dna(u2_2[-20:]) + fl_2) is False
    # Case C (msl=50)
    s_cc = u1_2 + u2_2
    c2_cc = mutate_seq_snps(u1_2, [8]) + d49_2 + mutate_seq_snps(u2_2, [112])
    assert check_similarity_merge(s_cc, fl_2 + c2_cc + fl_2) is True
    assert check_similarity_merge(s_cc, fl_2 + s_cc + u1_2[:20] + sep_2 + c2_cc + u2_2[-20:] + fl_2) is False
    assert check_similarity_merge(s_cc, fl_2 + s_cc + revcomp_dna(u1_2[:20]) + sep_2 + revcomp_dna(c2_cc) + revcomp_dna(u2_2[-20:]) + fl_2) is False
    # LTR P(25)+U1+U2+P (msl=50 & msl=100)
    for msl, u_sz, d_sz in ((50, 95, 49), (100, 155, 95)):
        p25 = make_complex_seq(25, seed=991000 + msl)
        lu1 = make_complex_seq(u_sz, seed=991100 + msl)
        lu2 = make_complex_seq(u_sz, seed=991200 + msl)
        ld = p25 + make_complex_seq(d_sz - 25, seed=991300 + msl)
        s_ltr = p25 + lu1 + lu2 + p25
        c2_ltr = mutate_seq_snps(p25, [8]) + lu1 + ld + lu2 + mutate_seq_snps(p25, [16])
        assert check_similarity_merge(s_ltr, fl_2 + c2_ltr + fl_2, min_sv_len=msl) is True
        assert check_similarity_merge(s_ltr, fl_2 + s_ltr + sep_2 + c2_ltr + fl_2, min_sv_len=msl) is False
        assert check_similarity_merge(s_ltr, fl_2 + s_ltr + sep_2 + revcomp_dna(c2_ltr) + fl_2, min_sv_len=msl) is False
    # Single 120 bp U1 interior-half ed=0 decoy -> False (fwd & RC)
    c2_ih = mutate_seq_snps(u1_2, [8, 60]) + d49_2 + mutate_seq_snps(u2_2, [112])
    assert check_similarity_merge(s_cc, fl_2 + c2_ih + fl_2) is True
    assert check_similarity_merge(s_cc, fl_2 + s_cc + sep_2 + u1_2 + sep_2 + c2_ih + fl_2) is False
    assert check_similarity_merge(s_cc, fl_2 + s_cc + sep_2 + revcomp_dna(u1_2) + sep_2 + revcomp_dna(c2_ih) + fl_2) is False

    # --- 3. Verbatim C1-r6 NEW Bug 3 & R1-r6 Minor F2 + SA-3 ---
    a3, b3, c3 = make_complex_seq(110, seed=631330), make_complex_seq(110, seed=632330), make_complex_seq(85, seed=633330)
    i25_3, d49_3, sep_3 = make_complex_seq(25, seed=634000), make_complex_seq(49, seed=635000), make_complex_seq(30, seed=636000)
    s_l2507 = a3 + b3 + i25_3 + c3
    c2_l2507 = a3 + d49_3 + b3 + c3
    assert check_similarity_merge(s_l2507, c2_l2507) is True
    assert check_similarity_merge(s_l2507, s_l2507 + sep_3 + c2_l2507) is False
    assert check_similarity_merge(s_l2507, s_l2507 + sep_3 + revcomp_dna(c2_l2507)) is False

    # q=330/350/380 (49D+25..35I), including SA-3 sd=7 49D+30I RC, and q=345 49D+45I
    for q_sz, al, bl, il, sd_list in (
        (330, 100, 100, 30, [0, 3, 7]),
        (350, 110, 110, 30, [0, 2]),
        (380, 120, 120, 35, [0, 2]),
        (345, 100, 100, 45, [0, 5]),
    ):
        cl = q_sz - al - bl - il
        for sd in sd_list:
            aa = make_complex_seq(al, seed=992000 + q_sz * 10 + sd)
            bb = make_complex_seq(bl, seed=992100 + q_sz * 10 + sd)
            cc = make_complex_seq(cl, seed=992200 + q_sz * 10 + sd)
            ii = make_complex_seq(il, seed=992300 + q_sz * 10 + sd)
            dd = make_complex_seq(49, seed=992400 + q_sz * 10 + sd)
            sp = make_complex_seq(30, seed=992500 + q_sz * 10 + sd)
            sq = aa + bb + ii + cc
            c2q = aa + dd + bb + cc
            assert check_similarity_merge(sq, c2q) is True
            assert check_similarity_merge(sq, sq + sp + c2q) is False
            assert check_similarity_merge(sq, sq + sp + revcomp_dna(c2q)) is False

    # Explicit SA-3 seed=7 exact construction (q=330, 49D+30I, RC)
    sa3_a = make_complex_seq(100, seed=710007)
    sa3_b = make_complex_seq(100, seed=711007)
    sa3_c = make_complex_seq(100, seed=712007)
    sa3_i30 = make_complex_seq(30, seed=713007)
    sa3_d49 = make_complex_seq(49, seed=714007)
    sa3_sep = make_complex_seq(30, seed=715007)
    sa3_q = sa3_a + sa3_i30 + sa3_b + sa3_c
    sa3_c2 = sa3_a + sa3_b + sa3_d49 + sa3_c
    assert check_similarity_merge(sa3_q, sa3_c2) is True
    assert check_similarity_merge(sa3_q, sa3_q + sa3_sep + sa3_c2) is False
    assert check_similarity_merge(sa3_q, sa3_q + sa3_sep + revcomp_dna(sa3_c2)) is False

    # Pure 0-I q=330 49D + 40/45/48trunc
    a3t, b3t = make_complex_seq(165, seed=711330), make_complex_seq(165, seed=712330)
    s_t = a3t + b3t
    for tr_l in (40, 45, 48):
        anti = mutate_seq_snps(a3t[:tr_l], list(range(tr_l)))
        c2_t = a3t[tr_l:] + make_complex_seq(49, seed=713000 + tr_l) + b3t
        assert check_similarity_merge(s_t, anti + c2_t) is True
        assert check_similarity_merge(s_t, s_t + make_complex_seq(15, seed=714000 + tr_l) + anti + c2_t) is False

    # R1-r6 Minor F2: q=170/190/270 (30..40I + 45D) and msl=100/200 canceling I+D
    for al, il, bl, cl, dl, msl in (
        (40, 30, 60, 40, 45, 50),
        (45, 35, 65, 45, 45, 50),
        (75, 40, 80, 75, 45, 50),
    ):
        aa = make_complex_seq(al, seed=993000 + msl * 10 + al)
        ii = make_complex_seq(il, seed=993100 + msl * 10 + al)
        bb = make_complex_seq(bl, seed=993200 + msl * 10 + al)
        cc = make_complex_seq(cl, seed=993300 + msl * 10 + al)
        dd = make_complex_seq(dl, seed=993400 + msl * 10 + al)
        sp = make_complex_seq(30, seed=993500 + msl * 10 + al)
        sq = aa + ii + bb + cc
        c2q = aa + bb + dd + cc
        assert check_similarity_merge(sq, c2q, min_sv_len=msl) is True
        assert check_similarity_merge(sq, sq + sp + c2q, min_sv_len=msl) is False
        assert check_similarity_merge(sq, sq + sp + revcomp_dna(c2q), min_sv_len=msl) is False
    for al, bl, cl, il, dl, msl in (
        (100, 180, 100, 50, 75, 100),
        (80, 215, 80, 55, 85, 200),
    ):
        aa = make_complex_seq(al, seed=993000 + msl * 10)
        ii = make_complex_seq(il, seed=993100 + msl * 10)
        bb = make_complex_seq(bl, seed=993200 + msl * 10)
        cc = make_complex_seq(cl, seed=993300 + msl * 10)
        dd = make_complex_seq(dl, seed=993400 + msl * 10)
        sp = make_complex_seq(20, seed=993500 + msl * 10)
        sq = aa + bb + ii + cc
        c2q = aa + dd + bb + cc
        assert check_similarity_merge(sq, c2q, min_sv_len=msl) is True
        assert check_similarity_merge(sq, sq + sp + c2q, min_sv_len=msl) is False
        assert check_similarity_merge(sq, sq + sp + revcomp_dna(c2q), min_sv_len=msl) is False

    # --- 4. Verbatim R2-r6 Major F1 correctness (3A-4 + 225-pair giant) + SA-2 distractors ---
    lp5 = "ACGTA" * 3 + "ACGT"
    rp5 = "TCGAG" * 3 + "TCGA"
    for sd in range(4):
        u1 = lp5 + make_complex_seq(71, seed=550000 + sd)
        u2 = make_complex_seq(71, seed=551000 + sd) + rp5
        s_3a = u1 + u2
        c2_3a = u1 + make_complex_seq(45, seed=552000 + sd) + u2
        sep_3a = make_complex_seq(40, seed=553000 + sd)
        dist_3a = "ACGTA" * 12 + make_complex_seq(130, seed=554000 + sd) + "TCGAG" * 12
        assert check_similarity_merge(s_3a, c2_3a) is True
        assert check_similarity_merge(s_3a, s_3a + sep_3a + dist_3a + sep_3a + c2_3a) is False
        assert check_similarity_merge(s_3a, s_3a + sep_3a + revcomp_dna(dist_3a) + sep_3a + revcomp_dna(c2_3a)) is False

    # 225-pair giant distractor (15x15 pairs >> max_nw_outer=64) -> False (fwd & RC)
    u1_g = lp5 + make_complex_seq(71, seed=559001)
    u2_g = make_complex_seq(71, seed=559002) + rp5
    s_g = u1_g + u2_g
    c2_g = u1_g + make_complex_seq(45, seed=559003) + u2_g
    dist_g = "ACGTA" * 24 + make_complex_seq(100, seed=559004) + "TCGAG" * 24
    sep_g = make_complex_seq(30, seed=559005)
    assert check_similarity_merge(s_g, c2_g) is True
    assert check_similarity_merge(s_g, s_g + sep_g + dist_g + sep_g + c2_g) is False
    assert check_similarity_merge(s_g, s_g + sep_g + revcomp_dna(dist_g) + sep_g + revcomp_dna(c2_g)) is False

    # SA-2: n_dist=4 high-ed (45X), n_dist=8 low-ed (18X), and n_dist=3 50D distractors before 45D/48D Copy 2 -> False
    u1_sa2 = make_complex_seq(50, seed=994001)
    u2_sa2 = make_complex_seq(50, seed=994002)
    s_sa2 = u1_sa2 + u2_sa2
    c2_sa2 = u1_sa2 + make_complex_seq(45, seed=994003) + u2_sa2
    assert check_similarity_merge(s_sa2, c2_sa2) is True
    dist_hi = u1_sa2[:20] + mutate_seq_snps(s_sa2[20:80], list(range(0, 60, 2)) + list(range(1, 30, 2))) + u2_sa2[-20:]
    dist_lo = u1_sa2[:20] + mutate_seq_snps(s_sa2[20:80], list(range(1, 55, 3))) + u2_sa2[-20:]
    dist_50d = u1_sa2 + make_complex_seq(50, seed=994004) + u2_sa2
    sp15 = make_complex_seq(15, seed=994005)
    assert check_similarity_merge(s_sa2, s_sa2 + sp15 + (dist_hi + sp15) * 4 + c2_sa2) is False
    assert check_similarity_merge(s_sa2, s_sa2 + sp15 + (dist_lo + sp15) * 8 + c2_sa2) is False
    assert check_similarity_merge(s_sa2, s_sa2 + sp15 + (dist_50d + sp15) * 3 + c2_sa2) is False

    # --- 5. Verbatim R2-r6 Major F2 (msl=160..200 dual_1bp_ins & dual_3bp_ins) ---
    for msl, arm_l, d_l in ((160, 175, 145), (170, 185, 155), (180, 185, 165), (200, 190, 180)):
        for ins_l, ins_txt_l, ins_txt_r in ((1, "A", "T"), (3, "ACG", "TGA")):
            for sd in range(3):
                u1 = make_complex_seq(arm_l, seed=860000 + msl * 10 + sd + ins_l * 1000)
                u2 = make_complex_seq(arm_l, seed=861000 + msl * 10 + sd + ins_l * 1000)
                d_seq = make_complex_seq(d_l, seed=862000 + msl * 10 + sd + ins_l * 1000)
                sep20 = make_complex_seq(20, seed=863000 + msl * 10 + sd + ins_l * 1000)
                sq = u1 + u2
                c2_f2 = u1[: arm_l - 12] + ins_txt_l + u1[arm_l - 12 :] + d_seq + u2[:12] + ins_txt_r + u2[12:]
                assert check_similarity_merge(sq, c2_f2, min_sv_len=msl) is True
                assert check_similarity_merge(sq, sq + sep20 + c2_f2, min_sv_len=msl) is False
                assert check_similarity_merge(sq, sq + sep20 + revcomp_dna(c2_f2), min_sv_len=msl) is False

    # --- 6. Stage 6E Latency & Determinism checks (< 15 ms CI assert; < 25 ms on ADV_MS200_PAL150) ---
    q_mid_6e = make_complex_seq(282, seed=941010)
    q_6e_1s = lp5 + q_mid_6e + rp5
    q_mid_dense = mutate_seq_snps(q_mid_6e, list(range(2, 277, 5)))
    r_6e_1s = (
        q_6e_1s
        + make_complex_seq(40, seed=941001)
        + ("ACGTA" * 14)
        + q_mid_dense[:140]
        + make_complex_seq(12, seed=941030)
        + q_mid_dense[140:]
        + ("TCGAG" * 14)
    )
    rp5_pal = revcomp_dna(lp5)
    h141 = make_complex_seq(141, seed=942010)
    q_mid_pal = h141 + mutate_seq_snps(revcomp_dna(h141), list(range(5, 140, 20)))
    q_mega = lp5 + q_mid_pal + rp5_pal
    q_mid_pal_dense = mutate_seq_snps(q_mid_pal, list(range(2, 277, 5)))
    r_mega_10i = (
        q_mega
        + make_complex_seq(40, seed=942001)
        + ("ACGTA" * 14)
        + q_mid_pal_dense[:141]
        + make_complex_seq(10, seed=942030)
        + q_mid_pal_dense[141:]
        + (revcomp_dna("ACGTA") * 14)
    )
    r_mega_16i = (
        q_mega
        + make_complex_seq(40, seed=942001)
        + ("ACGTA" * 14)
        + q_mid_pal_dense[:141]
        + make_complex_seq(16, seed=942031)
        + q_mid_pal_dense[141:]
        + (revcomp_dna("ACGTA") * 14)
    )
    r_mega_bothstr = (
        q_mega
        + make_complex_seq(40, seed=942002)
        + ("ACGTA" * 14)
        + q_mid_pal_dense[:141]
        + make_complex_seq(12, seed=942032)
        + q_mid_pal_dense[141:]
        + (revcomp_dna("ACGTA") * 14)
    )
    q_mid_400 = make_complex_seq(362, seed=943010)
    q_400_200 = lp5 + q_mid_400 + rp5
    q_mid_400_d = mutate_seq_snps(q_mid_400, list(range(2, 357, 5)))
    r_400_200 = (
        q_400_200
        + make_complex_seq(40, seed=943001)
        + ("ACGTA" * 14)
        + q_mid_400_d[:181]
        + make_complex_seq(15, seed=943030)
        + q_mid_400_d[181:]
        + ("TCGAG" * 14)
    )
    q_mid_200 = make_complex_seq(162, seed=944010)
    q_2loci = lp5 + q_mid_200 + rp5
    q_mid_200_d = mutate_seq_snps(q_mid_200, list(range(2, 157, 4)))
    dec_2l = ("ACGTA" * 5) + q_mid_200_d[:81] + make_complex_seq(10, seed=944030) + q_mid_200_d[81:] + ("TCGAG" * 5)
    sep40_2l = make_complex_seq(40, seed=944001)
    r_2loci = q_2loci + sep40_2l + dec_2l + sep40_2l + revcomp_dna(dec_2l)

    q_mid_450 = make_complex_seq(412, seed=945010)
    q_w450 = lp5 + q_mid_450 + rp5_pal
    q_mid_450_d = mutate_seq_snps(q_mid_450, list(range(2, 407, 5)))
    r_w450 = (
        q_w450
        + make_complex_seq(20, seed=945001)
        + ("ACGTA" * 9)
        + q_mid_450_d[:206]
        + make_complex_seq(12, seed=945030)
        + q_mid_450_d[206:]
        + (revcomp_dna("ACGTA") * 9)
    )
    q_mid_650 = make_complex_seq(612, seed=946010)
    q_w650 = lp5 + q_mid_650 + rp5
    r_w650 = make_complex_seq(175, seed=946001) + q_w650 + make_complex_seq(175, seed=946002)

    r950_np = make_complex_seq(950, seed=930007)
    u200_at = "AT" * 10 + make_complex_seq(160, seed=930008) + "AT" * 10
    r950_at = "AT" * 180 + u200_at + "AT" * 195
    pal150_adv = _mk_pal(150, seed=929600)
    sep250_adv = make_complex_seq(250, seed=929601)
    q_adv200 = "AT" * 36 + pal150_adv
    r_adv200 = "AT" * 12 + "C" * 8 + "AT" * 12 + "G" * 8 + "AT" * 12 + "C" * 8 + pal150_adv + sep250_adv + "AT" * 200

    s6e_lat_cases: list[tuple[str, str, str, int, float]] = [
        ("BOMB_6E_msl50_1s_5bp_indel", q_6e_1s, r_6e_1s, 50, 15.0),
        ("MEGA_6E_msl50_2s_48DP_10I", q_mega, r_mega_10i, 50, 15.0),
        ("MEGA_6E_msl50_2s_48DP_16I", q_mega, r_mega_16i, 50, 15.0),
        ("S6E_msl50_q320_48x_DP_bothstr", q_mega, r_mega_bothstr, 50, 15.0),
        ("S6E_msl200_q400_24x_DP_edlt100", q_400_200, r_400_200, 200, 15.0),
        ("2LOCI_PLUS_MINUS_q200", q_2loci, r_2loci, 50, 15.0),
        ("W2507_2s_q450_bomb", q_w450, r_w450, 50, 15.0),
        ("W2507_ov_q650_bomb", q_w650, r_w650, 50, 15.0),
        ("950bp_1snp_nonpal_msl50", mutate_seq_snps(r950_np, [475]), r950_np, 50, 15.0),
        ("950bp_1snp_nonpal_msl200", mutate_seq_snps(r950_np, [475]), r950_np, 200, 15.0),
        ("200bp_ATflank_in_950bp_AT", u200_at, r950_at, 50, 15.0),
        ("ADV_MS200_SELF_RC_PAL150_AT200", q_adv200, r_adv200, 200, 25.0),
    ]
    for label, q_lat, r_lat, msl, lim_ms in s6e_lat_cases:
        results = []
        times_ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = check_similarity_merge(q_lat, r_lat, min_sv_len=msl)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append(res)
        assert results == [True, True, True], f"{label} expected deterministic True, got {results}"
        assert statistics.median(times_ms) < lim_ms, f"{label} median {statistics.median(times_ms):.2f} ms >= {lim_ms} ms"

    # --- 7. Stage 6E FP controls (MUST remain True) ---
    # C2-r3 B1 (incl. q=330..450)
    for p_len in (240, 340, 440):
        fl_a, fl_b = make_complex_seq(100, seed=995000 + p_len), make_complex_seq(100, seed=995100 + p_len)
        p_b1 = _mk_pal(p_len, seed=995200 + p_len)
        mid_b1 = p_len // 2
        assert check_similarity_merge(p_b1[: mid_b1 - 20] + p_b1[mid_b1 + 20 :] + fl_b[:35], fl_a + p_b1 + fl_b) is True
    # Fix 5
    for d_fix5 in (10, 30, 49):
        fl_a, fl_b = make_complex_seq(100, seed=995300 + d_fix5), make_complex_seq(100, seed=995400 + d_fix5)
        ad = make_complex_seq(d_fix5, seed=995500 + d_fix5)
        p_near = mutate_seq_snps(_mk_pal(200 - d_fix5, seed=995600 + d_fix5), [5, 15, 25, 35])
        d30_f5 = make_complex_seq(30, seed=995700 + d_fix5)
        ov25 = make_complex_seq(25, seed=995800 + d_fix5)
        if ov25[0] == fl_b[0]:
            ov25 = _flip1(fl_b[0]) + ov25[1:]
        assert (
            check_similarity_merge(
                ad + p_near + ov25,
                fl_a + ad + p_near[:80] + d30_f5 + p_near[80:] + revcomp_dna(ad) + fl_b,
            )
            is True
        )
    # U1-only (q=240..500), 50D/60D Copy 2, and tandem/homo flanks -> True
    for q_fp in (240, 350, 500):
        h_fp = q_fp // 2
        ua = make_complex_seq(h_fp, seed=996000 + q_fp)
        ub = make_complex_seq(h_fp, seed=996100 + q_fp)
        unrel = make_complex_seq(h_fp, seed=996200 + q_fp)
        sp_fp = make_complex_seq(30, seed=996300 + q_fp)
        assert check_similarity_merge(ua + ub, ua + ub + sp_fp + ua + unrel) is True
    for d_big in (50, 60):
        ua = make_complex_seq(120, seed=996400 + d_big)
        ub = make_complex_seq(120, seed=996500 + d_big)
        db = make_complex_seq(d_big, seed=996600 + d_big)
        assert check_similarity_merge(ua + ub, ua + ub + sep_r1 + ua + db + ub) is True
    assert check_similarity_merge(u1_r1 + u2_r1, "A" * 200 + u1_r1 + u2_r1 + "A" * 200) is True
    assert check_similarity_merge(u1_r1 + u2_r1, ("AT" * 100) + u1_r1 + u2_r1 + ("AT" * 100)) is True


