"""White-box unit tests for PanFlat homologous redundancy elimination (DEDUP_PLAN.md).

Covers:
1. §2 CLI/Config defaults & `validate_build_config` validation + centromere Step B 0.85 isolation.
2. Module A.1 (`repeat_guard.py`): exact 50 bp `min_identity` boundary, `short_aligner` + `short_buf`
   on +/- strands and iterative zdrop <=500 bp flanks, `protected_ins`, empty `ref_region_seq`
   with prebuilt aligners, and `t_starts`/`t_ends` target-piece-scoped op splitting.
3. Module A.2 (`linearizer.py` Stage 2.5a): Step 1 -> Step 2 -> Step 3 double slicing,
   `partial_content` preservation, `left_anchor_contiguous` transitions, sibling preservation
   on all-repeat residual, JSON-only `REFERENCE_LIKE` / `SHORT_RESIDUAL` logging, and
   `contig_had_subtracted_homolog: set[int]` genotyping protection.
4. Module B (`linearizer.py` Stage 2.5b): childless secondary backbone unlocking
   (`is_synthetic=False`) and `m_child` deletion-vs-`rep0` genotyping guard.
5. Module C (`linearizer.py` Stage 5.5): `_subslice_piece` (+/- strands), all 7 `join_before`
   state transitions, lockstep `piece_cand_ids`, untouched `< min_sv_len` piece preservation,
   13-mer seed gate (stride 1 and stride 7), small-budget multi-wave + mini-subwave + lazy-dirty
   `tail_wave` + mid-batch wave-seal carryover + Priority-0 `CEN_ISLAND` seeding, and
   `threads=1` vs `threads=4` + `ThreadBuffer`-absent exact parity.
6. Module D (`writers.py::merge_chromosomes`): associative hierarchical 3-chromosome merge
   (`flat == left-nested == right-nested`), exact summed `dedup_*` counters, 0 duplicate keys,
   and heterogeneous `dedup_mode`/`dedup_id` handling.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
from typing import Sequence

import mappy
import pytest

import panflat.linearizer as linearizer_mod
from panflat.gfa_io import load_and_validate_inputs, validate_build_config
from panflat.linearizer import (
    CandidateBlock,
    ClusterRepresentative,
    ExtractedCandidate,
    GroupRepeatGuardResult,
    ParallelGroupSolution,
    _CandidateDedupStats,
    _deduplicate_representatives_across_chromosome,
    _nested_key_sort_key,
    _screen_and_solve_one_pair,
    _Stage5ForkContext,
    _subslice_piece,
    _TailSeedGate,
    run_chromosome_pipeline,
    screen_and_fold_candidate,
    solve_parallel_group,
)
from panflat.models import (
    BuildConfig,
    CentromereOutput,
    CLIArgumentError,
    InsertionSlotBlock,
    Piece,
    Step,
    StepProvenance,
)
from panflat.repeat_guard import (
    REF_SUB_MIN_IDENTITY,
    _check_similarity_merge_with_span,
    _evaluate_mappy_hits_on_blocks,
    _evaluate_piece_units_on_blocks,
    _extract_hit_piece_eval_units,
    _filter_cross_aligner_shadow_hits,
    _filter_cross_aligner_shadow_units,
    _HitPieceEvalUnit,
    _HitPieceUnit,
    _partition_bridged_hit_ops,
    hit_covers_subinterval_mappy,
    subtract_reference_intervals_mappy,
)
from panflat.verify_qc import verify_build_outputs
from panflat.writers import merge_chromosomes, write_build_outputs
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
    revcomp_dna,
)


def _mutate_per_50bp_block(seq: str, block_offsets: Sequence[int]) -> str:
    """Mutate deterministic offsets inside every complete 50 bp block of `seq`."""
    positions: list[int] = []
    for blk_start in range(0, len(seq) - 49, 50):
        for off in block_offsets:
            positions.append(blk_start + off)
    return mutate_seq_snps(seq, positions)


def _build_graph_and_config(
    tmp_path: Path,
    tag: str,
    gb: SyntheticGFAFixtureBuilder,
    ref_steps: Sequence[tuple[str, str]],
    chrom: str = "Chr01",
    backbone_prefix: str = "Ref#1#",
    bed_rows: Sequence[tuple[str, int, int] | tuple[str, int, int, str]] | None = None,
    **cfg_kwargs: object,
) -> tuple[object, BuildConfig]:
    gfa_path = gb.write_gfa(tmp_path / f"{tag}.gfa")
    ref_seq = gb.spell_steps(ref_steps)
    ref_fa = gb.write_fasta(tmp_path / f"{tag}.ref.fa", {chrom: ref_seq})
    bed_path: str | None = None
    if bed_rows is not None:
        bed_path = str(gb.write_bed(tmp_path / f"{tag}.cen.bed", bed_rows))
    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(ref_fa),
        backbone_prefix=backbone_prefix,
        out_prefix=str(tmp_path / f"{tag}_out"),
        centromere_bed=bed_path,
        **cfg_kwargs,  # type: ignore[arg-type]
    )
    graph = load_and_validate_inputs(cfg)
    return graph, cfg


# ============================================================================
# 1. §2 CLI, Config & Validation + Centromere Step B 0.85 Isolation
# ============================================================================


def test_build_config_dedup_defaults_and_validation(tmp_path: Path) -> None:
    """§2: `BuildConfig` defaults are `off`/0.90; `validate_build_config` enforces exact domain."""
    cfg_default = BuildConfig(
        gfa=str(tmp_path / "a.gfa"),
        ref_fa=str(tmp_path / "a.fa"),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
    )
    assert cfg_default.dedup_mode == "off"
    assert cfg_default.dedup_id == 0.90
    validate_build_config(cfg_default)

    for mode, did in (("off", 0.90), ("ref", 0.80), ("pan", 0.99), ("pan", 0.90)):
        cfg = BuildConfig(
            gfa=str(tmp_path / "a.gfa"),
            ref_fa=str(tmp_path / "a.fa"),
            backbone_prefix="Ref#1#",
            out_prefix=str(tmp_path / "out"),
            dedup_mode=mode,  # type: ignore[arg-type]
            dedup_id=did,
        )
        validate_build_config(cfg)

    for bad_mode in ("invalid", "auto", "", "PAN"):
        cfg_bad = BuildConfig(
            gfa=str(tmp_path / "a.gfa"),
            ref_fa=str(tmp_path / "a.fa"),
            backbone_prefix="Ref#1#",
            out_prefix=str(tmp_path / "out"),
            dedup_mode=bad_mode,  # type: ignore[arg-type]
        )
        with pytest.raises(CLIArgumentError, match="--dedup-mode"):
            validate_build_config(cfg_bad)

    for bad_id in (0.799, 0.991, -0.5, 1.0, float("nan"), float("inf"), float("-inf")):
        cfg_bad = BuildConfig(
            gfa=str(tmp_path / "a.gfa"),
            ref_fa=str(tmp_path / "a.fa"),
            backbone_prefix="Ref#1#",
            out_prefix=str(tmp_path / "out"),
            dedup_mode="ref",
            dedup_id=bad_id,
        )
        with pytest.raises(CLIArgumentError, match="--dedup-id"):
            validate_build_config(cfg_bad)


def test_centromere_step_b_isolated_at_085_under_high_dedup_id(tmp_path: Path) -> None:
    """§2: Centromere Step B stays locked at 0.85 even when `--dedup-mode ref --dedup-id 0.98`."""
    gb = SyntheticGFAFixtureBuilder()
    r1 = make_complex_seq(1200, seed=101)
    r2 = make_complex_seq(500, seed=102)
    cen_r = make_complex_seq(1500, seed=103)
    r4 = make_complex_seq(500, seed=104)
    r5 = make_complex_seq(500, seed=105)

    # 6 SNPs per 50 bp block -> 44/50 = 0.88 identity (above 0.85, strictly below 0.98)
    # Offsets [2, 3, 4, 45, 46, 47] leave a clean 40 bp exact middle run for asm20 seeding.
    cen_hom_88 = _mutate_per_50bp_block(cen_r[200:1200], (2, 3, 4, 45, 46, 47))
    cen_novel = make_complex_seq(700, seed=106)
    arm_hom_88 = _mutate_per_50bp_block(r1[100:1100], (2, 3, 4, 45, 46, 47))

    for nid, seq in [
        ("1", r1),
        ("2", r2),
        ("3", cen_r),
        ("4", r4),
        ("5", r5),
        ("6", cen_hom_88 + cen_novel),
        ("7", arm_hom_88),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path(
        "Alt1#1#Chr01",
        [("1", "+"), ("2", "+"), ("6", "+"), ("4", "+"), ("7", "+"), ("5", "+")],
    )

    cen_s = len(r1) + len(r2)
    cen_e = cen_s + len(cen_r)
    graph, cfg = _build_graph_and_config(
        tmp_path,
        "cen_085_iso",
        gb,
        ref_steps,
        bed_rows=[("Chr01", cen_s, cen_e, "CEN_ISO")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
        centromere_min_island=500,
        dedup_mode="ref",
        dedup_id=0.98,
    )
    res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]

    assert res.summary_metrics["ref_sub_min_identity"] == REF_SUB_MIN_IDENTITY == 0.85
    assert res.summary_metrics["dedup_mode"] == "ref"
    assert res.summary_metrics["dedup_id"] == 0.98

    # Centromere Step B MUST still subtract the 1000 bp 88%-identical sequence (0.88 >= 0.85)
    alt_cen = next(r for r in res.centromere_rows if r.hap == "Alt1#1")
    assert int(alt_cen.reference_like_bp) == 1000
    assert int(alt_cen.accepted_source_bp) == 700

    # Normal-arm Module A MUST NOT subtract the 88%-identical ectopic copy under dedup_id=0.98
    assert int(res.summary_metrics["dedup_ref_masked_bp"]) == 0
    arm_ins = [ev for ev in res.sv_events if ev.event_type == "INS"]
    assert len(arm_ins) == 1
    assert any(
        int(row["pan_end"]) - int(row["pan_start"]) == 1000
        and str(row["event_id"]) == arm_ins[0].event_id
        for row in res.inserts_rows
    )


# ============================================================================
# 2. Module A.1 (`panflat/repeat_guard.py`)
# ============================================================================


def test_subtract_reference_min_identity_exact_50bp_boundary() -> None:
    """Module A.1: 46/50 = 0.92 blocks are masked at 0.90 & 0.92, and unmasked at 0.95."""
    ref_seq = make_complex_seq(3000, seed=201)
    base = ref_seq[1000:1500]  # 10 blocks of 50 bp
    # Exact flanks [0:150] & [350:500]; middle 4 blocks [150:350] have 4 SNPs / 50 bp = 46/50 = 0.92
    mid_92 = _mutate_per_50bp_block(base[150:350], (2, 3, 46, 47))
    query = base[:150] + mid_92 + base[350:]

    aln = mappy.Aligner(seq=ref_seq, preset="asm20", best_n=50)
    buf = mappy.ThreadBuffer()

    m90, u90, bp90 = subtract_reference_intervals_mappy(
        query_seq=query, ref_region_seq="", aligner=aln, buf=buf, min_identity=0.90
    )
    assert m90 == [(0, 500)] and u90 == [] and bp90 == 500

    m92, u92, bp92 = subtract_reference_intervals_mappy(
        query_seq=query, ref_region_seq="", aligner=aln, buf=buf, min_identity=0.92
    )
    assert m92 == [(0, 500)] and u92 == [] and bp92 == 500

    m95, u95, bp95 = subtract_reference_intervals_mappy(
        query_seq=query, ref_region_seq="", aligner=aln, buf=buf, min_identity=0.95
    )
    assert m95 == [(0, 150), (350, 500)]
    assert u95 == [(150, 350)]
    assert bp95 == 300


def test_subtract_reference_short_aligner_strands_zdrop_and_protected_ins() -> None:
    """Module A.1: `short_aligner` catches 50-250 bp +/- & <=500 bp zdrop flanks while preserving `protected_ins`."""
    ref_seq = make_complex_seq(5000, seed=202)
    main_aln = mappy.Aligner(seq=ref_seq, preset="asm20", best_n=50)
    short_aln = mappy.Aligner(
        seq=ref_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    buf_m = mappy.ThreadBuffer()
    buf_s = mappy.ThreadBuffer()

    # (a) 50-250 bp homologous queries on BOTH '+' and '-' strands
    for q_len in (50, 75, 100, 150, 250):
        fwd = ref_seq[1200 : 1200 + q_len]
        for q_seq in (fwd, revcomp_dna(fwd)):
            m_ivs, u_ivs, m_bp = subtract_reference_intervals_mappy(
                query_seq=q_seq,
                ref_region_seq="",
                aligner=main_aln,
                buf=buf_m,
                min_identity=0.90,
                short_aligner=short_aln,
                short_buf=buf_s,
            )
            assert m_ivs == [(0, q_len)] and u_ivs == [] and m_bp == q_len

    # Verify a 50 bp homologous query is missed by asm20 alone but caught by short_aligner on +/- strands
    q50 = ref_seq[2100:2150]
    for q_seq in (q50, revcomp_dna(q50)):
        m_main, _, bp_main = subtract_reference_intervals_mappy(
            query_seq=q_seq,
            ref_region_seq="",
            aligner=main_aln,
            buf=buf_m,
            min_identity=0.90,
        )
        m_both, _, bp_both = subtract_reference_intervals_mappy(
            query_seq=q_seq,
            ref_region_seq="",
            aligner=main_aln,
            buf=buf_m,
            min_identity=0.90,
            short_aligner=short_aln,
            short_buf=buf_s,
        )
        assert m_main == [] and bp_main == 0
        assert m_both == [(0, 50)] and bp_both == 50

    # (b) Internal I >= min_sv_len (`protected_ins`) preserved while both flanks are masked
    novel_ins = make_complex_seq(150, seed=203)
    q_ins = ref_seq[500:1300] + novel_ins + ref_seq[1300:2100]
    m_ins, u_ins, bp_ins = subtract_reference_intervals_mappy(
        query_seq=q_ins,
        ref_region_seq="",
        min_sv_len=50,
        aligner=main_aln,
        buf=buf_m,
        min_identity=0.90,
        short_aligner=short_aln,
        short_buf=buf_s,
    )
    assert u_ins == [(800, 950)]
    assert m_ins == [(0, 800), (950, 1750)]
    assert bp_ins == 1600

    # (c) Iterative zdrop re-mapping: >500 bp query with <=500 bp unaligned region containing
    #     a 150 bp novel gap + 50 bp '-' strand homologous tail (200 bp <= 500 bp triggers short_aligner!)
    novel_gap = make_complex_seq(150, seed=204)
    tail_50_rc = revcomp_dna(ref_seq[3800:3850])
    q_zdrop = ref_seq[200:1400] + novel_gap + tail_50_rc  # 1400 bp (> 500 bp)
    m_no_short, u_no_short, bp_no_short = subtract_reference_intervals_mappy(
        query_seq=q_zdrop,
        ref_region_seq="",
        min_sv_len=50,
        aligner=main_aln,
        buf=buf_m,
        min_identity=0.90,
    )
    assert m_no_short == [(0, 1200)] and u_no_short == [(1200, 1400)] and bp_no_short == 1200

    m_with_short, u_with_short, bp_with_short = subtract_reference_intervals_mappy(
        query_seq=q_zdrop,
        ref_region_seq="",
        min_sv_len=50,
        aligner=main_aln,
        buf=buf_m,
        min_identity=0.90,
        short_aligner=short_aln,
        short_buf=buf_s,
    )
    assert m_with_short == [(0, 1200), (1350, 1400)]
    assert u_with_short == [(1200, 1350)]
    assert bp_with_short == 1250


def test_subtract_reference_prebuilt_aligner_empty_ref_region_seq() -> None:
    """Module A.1: prebuilt `aligner` / `short_aligner` work with `ref_region_seq=''`."""
    ref_seq = make_complex_seq(1200, seed=205)
    q_long = ref_seq[200:500]
    q_short = ref_seq[200:300]
    aln = mappy.Aligner(seq=ref_seq, preset="asm20", best_n=50)
    saln = mappy.Aligner(seq=ref_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)

    m1, u1, bp1 = subtract_reference_intervals_mappy(q_long, ref_region_seq="", aligner=aln)
    assert m1 == [(0, 300)] and u1 == [] and bp1 == 300

    m2, u2, bp2 = subtract_reference_intervals_mappy(
        q_short, ref_region_seq="", aligner=None, short_aligner=saln
    )
    assert m2 == [(0, 100)] and u2 == [] and bp2 == 100

    m3, u3, bp3 = subtract_reference_intervals_mappy(
        q_short, ref_region_seq="", aligner=None, short_aligner=None
    )
    assert m3 == [] and u3 == [(0, 100)] and bp3 == 0


def test_subtract_reference_target_piece_scoped_op_splitting() -> None:
    """Module A.1 + C: `t_starts`/`t_ends` blocks cross-piece chimeric combining while keeping intact PieceA+PieceB."""
    piece_a = make_complex_seq(600, seed=210)
    piece_b = make_complex_seq(600, seed=211)
    sep = "N" * 200
    wave_seq = piece_a + sep + piece_b
    t_starts = (0, 800)
    t_ends = (600, 1400)

    aln_main = mappy.Aligner(seq=wave_seq, preset="asm20", best_n=50)
    aln_short = mappy.Aligner(
        seq=wave_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    buf_m = mappy.ThreadBuffer()
    buf_s = mappy.ThreadBuffer()

    # 1. Chimeric 25 bp PieceA + 25 bp PieceB query MUST yield 0 masked bases
    chim_50 = piece_a[-25:] + piece_b[:25]
    for q_chim in (chim_50, revcomp_dna(chim_50)):
        m_c, u_c, bp_c = subtract_reference_intervals_mappy(
            query_seq=q_chim,
            ref_region_seq="",
            aligner=aln_main,
            buf=buf_m,
            min_identity=0.90,
            short_aligner=aln_short,
            short_buf=buf_s,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        assert m_c == [] and u_c == [(0, 50)] and bp_c == 0

    # Also test synthetic bridged hit on BOTH '+' and '-' strands directly on _evaluate_mappy_hits_on_blocks:
    # 25M from PieceA[575:600] + 200D across sep + 25M from PieceB[800:825] inside one 50 bp block
    fake_hit_plus = SimpleNamespace(
        q_st=0, q_en=50, r_st=575, r_en=825, strand=1, cs=":25-" + ("n" * 200) + ":25"
    )
    fake_hit_minus = SimpleNamespace(
        q_st=0, q_en=50, r_st=575, r_en=825, strand=-1, cs=":25-" + ("n" * 200) + ":25"
    )
    for fhit in (fake_hit_plus, fake_hit_minus):
        hm, ha = _evaluate_mappy_hits_on_blocks(
            hits=[fhit],
            query_offset=0,
            q_len=50,
            num_blocks=1,
            min_sv_len=50,
            min_identity=0.90,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        assert hm == [] and ha == []

    # 2. Intact PieceA + PieceB on BOTH '+' and '-' strands MUST mask 100% (1200 bp)
    intact_ab = piece_a + piece_b
    for q_ab in (intact_ab, revcomp_dna(intact_ab)):
        m_ab, u_ab, bp_ab = subtract_reference_intervals_mappy(
            query_seq=q_ab,
            ref_region_seq="",
            aligner=aln_main,
            buf=buf_m,
            min_identity=0.90,
            short_aligner=aln_short,
            short_buf=buf_s,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        assert m_ab == [(0, 1200)] and u_ab == [] and bp_ab == 1200

    # 3. PieceA + 200 bp novel + PieceB on BOTH '+' and '-' strands masks both flanks
    #    and preserves [600, 800) with zero 'N'*200 D-penalty leakage
    novel_200 = make_complex_seq(200, seed=212)
    q_flanked = piece_a + novel_200 + piece_b
    for q_f in (q_flanked, revcomp_dna(q_flanked)):
        m_f, u_f, bp_f = subtract_reference_intervals_mappy(
            query_seq=q_f,
            ref_region_seq="",
            aligner=aln_main,
            buf=buf_m,
            min_identity=0.90,
            short_aligner=aln_short,
            short_buf=buf_s,
            t_starts=t_starts,
            t_ends=t_ends,
        )
        assert m_f == [(0, 600), (800, 1400)]
        assert u_f == [(600, 800)]
        assert bp_f == 1200


# ============================================================================
# 3. Module A.2 (`panflat/linearizer.py` Stage 2.5a)
# ============================================================================


def test_screen_and_fold_candidate_step1_to_step2_double_slicing_and_logging(
    tmp_path: Path,
) -> None:
    """Module A.2: Step 1 -> Step 2 -> Step 3 double slicing preserves flags, siblings, and JSON-only logs."""
    gb = SyntheticGFAFixtureBuilder()
    ref_left = make_complex_seq(500, seed=301)
    ref_te = make_complex_seq(1000, seed=302)
    ref_right = make_complex_seq(500, seed=303)

    novel_800 = make_complex_seq(800, seed=304)
    sat_1500 = make_satellite_seq(unit_len=20, total_len=1500, seed=305)
    short_25 = make_complex_seq(25, seed=306)
    novel_600 = make_complex_seq(600, seed=307)

    # Nodes for the 5 test candidates
    c_left_cut_seq = ref_te + novel_800
    c_sibling_seq = sat_1500 + ref_te + novel_800
    c_right_cut_seq = novel_800 + ref_te
    c_ref_only_seq = ref_te + short_25
    c_short_plus_kept_seq = novel_600 + ref_te + short_25

    for nid, seq in [
        ("r1", ref_left),
        ("r2", ref_te),
        ("r3", ref_right),
        ("n_lcut", c_left_cut_seq),
        ("n_sib", c_sibling_seq),
        ("n_rcut", c_right_cut_seq),
        ("n_ronly", c_ref_only_seq),
        ("n_skeep", c_short_plus_kept_seq),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [("r1", "+"), ("r2", "+"), ("r3", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path("H1#1#Chr01", [("r1", "+"), ("n_lcut", "+"), ("r2", "+"), ("r3", "+")])

    graph, cfg = _build_graph_and_config(
        tmp_path, "mod_a2_slicing", gb, ref_steps, dedup_mode="ref", dedup_id=0.90
    )

    def _run_one(
        cid: str, nid: str, seq: str
    ) -> tuple[
        list[ExtractedCandidate],
        list[object],
        list[dict[str, object]],
        bool,
        _CandidateDedupStats,
    ]:
        raw = CandidateBlock(
            candidate_id=cid,
            hap_id="H1#1",
            src_path="H1#1#Chr01",
            left_anchor_rank=0,
            right_anchor_rank=1,
            source_order=0,
            steps=(Step(nid, "+"),),
            provenance=(StepProvenance("H1#1", "H1#1#Chr01", 500, 500 + len(seq), "+"),),
            seq=seq,
            left_anchor_contiguous=True,
        )
        stats = _CandidateDedupStats()
        kept, unres, logs, had_m, _ = screen_and_fold_candidate(
            raw_cand=raw,
            graph=graph,  # type: ignore[arg-type]
            config=cfg,
            region_id="REG_1",
            next_source_order=0,
            unres_counter_ref=[1],
            seen_exported_intervals={},
            dedup_stats_out=stats,
        )
        return kept, unres, logs, had_m, stats

    # (a) Left flank cut by Step 1, Step 2 finds 0 repeats -> partial_content MUST remain True,
    #     left_anchor_contiguous MUST become False
    kept_a, unres_a, logs_a, had_m_a, st_a = _run_one("cA", "n_lcut", c_left_cut_seq)
    assert len(kept_a) == 1
    assert kept_a[0].seq == novel_800
    assert kept_a[0].partial_content is True
    assert kept_a[0].left_anchor_contiguous is False
    assert unres_a == [] and had_m_a is False
    assert st_a.had_subtracted_homolog is True and st_a.ref_masked_bp == 1000
    assert any(entry["final_reason"] == "SPLIT_BY_REFERENCE_SUBTRACTION" for entry in logs_a)

    # (b) All-repeat Step 2 residual (1500 bp sat) MUST NOT discard sibling Step 1 residual (800 bp novel)
    kept_b, unres_b, logs_b, had_m_b, st_b = _run_one("cB", "n_sib", c_sibling_seq)
    assert len(kept_b) == 1
    assert kept_b[0].seq == novel_800
    assert kept_b[0].partial_content is True
    assert kept_b[0].left_anchor_contiguous is False
    assert had_m_b is True
    assert len(unres_b) == 1 and getattr(unres_b[0], "reason") == "REPEAT_LIKE"
    assert st_b.had_subtracted_homolog is True and st_b.ref_masked_bp == 1000
    assert any(entry["final_reason"] == "REPEAT_LIKE" for entry in logs_b)

    # (c) Right flank cut only -> left_anchor_contiguous stays True while partial_content is True
    kept_c, unres_c, _, _, st_c = _run_one("cC", "n_rcut", c_right_cut_seq)
    assert len(kept_c) == 1
    assert kept_c[0].seq == novel_800
    assert kept_c[0].partial_content is True
    assert kept_c[0].left_anchor_contiguous is True
    assert unres_c == [] and st_c.ref_masked_bp == 1000

    # (d) Pure REFERENCE_LIKE (+ 25 bp short residual): 0 unresolved_records, JSON log only
    kept_d, unres_d, logs_d, _, st_d = _run_one("cD", "n_ronly", c_ref_only_seq)
    assert kept_d == []
    assert unres_d == []
    assert [entry["final_reason"] for entry in logs_d] == ["REFERENCE_LIKE"]
    assert st_d.had_subtracted_homolog is True
    assert st_d.ref_masked_bp == 1000
    assert st_d.short_residual_bp == 25

    # (e) Short residual (25 bp) + kept novel (600 bp): SHORT_RESIDUAL in JSON logs only (never unres_out)
    kept_e, unres_e, logs_e, _, st_e = _run_one("cE", "n_skeep", c_short_plus_kept_seq)
    assert len(kept_e) == 1 and kept_e[0].seq == novel_600
    assert unres_e == []
    assert any(entry["final_reason"] == "SHORT_RESIDUAL" for entry in logs_e)
    assert st_e.ref_masked_bp == 1000 and st_e.short_residual_bp == 25


def test_module_a2_contig_had_subtracted_homolog_stores_int_and_genotypes_dot(
    tmp_path: Path,
) -> None:
    """Module A.2: `contig_had_subtracted_homolog` stores `int` (`contig_idx`) and emits '.' on INS."""
    gb = SyntheticGFAFixtureBuilder()
    r1 = make_complex_seq(500, seed=310)
    r2_te = make_complex_seq(1200, seed=311)
    r3 = make_complex_seq(500, seed=312)
    r4 = make_complex_seq(500, seed=313)
    nov = make_complex_seq(700, seed=314)
    te_94 = _mutate_per_50bp_block(r2_te, (3, 4, 46))  # 47/50 = 94%

    for nid, seq in [
        ("1", r1),
        ("2", r2_te),
        ("3", r3),
        ("4", r4),
        ("5", nov),
        ("6", te_94),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("5", "+"), ("4", "+")])
    gb.add_p_path("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("6", "+"), ("4", "+")])
    gb.add_p_path("Hap3#1#Chr01", ref_steps)

    graph, cfg = _build_graph_and_config(
        tmp_path, "mod_a2_int_key", gb, ref_steps, dedup_mode="ref", dedup_id=0.90
    )

    # Directly inspect `_screen_and_solve_one_pair` return via spy to verify `set[int]`
    captured_guards: list[GroupRepeatGuardResult] = []
    orig_fn = _screen_and_solve_one_pair

    def _spy(
        pair: tuple[int, int],
        ctx: _Stage5ForkContext,
        unres_counter_ref: list[int],
        seen_exported_intervals: dict[tuple[str, int, int, str], str],
        ref_dedup_buf: object = None,
        ref_dedup_short_buf: object = None,
    ) -> tuple[GroupRepeatGuardResult, ParallelGroupSolution | None]:
        g_res, sol = orig_fn(
            pair,
            ctx,
            unres_counter_ref=unres_counter_ref,
            seen_exported_intervals=seen_exported_intervals,
            ref_dedup_buf=ref_dedup_buf,
            ref_dedup_short_buf=ref_dedup_short_buf,
        )
        captured_guards.append(g_res)
        return g_res, sol

    mp = pytest.MonkeyPatch()
    mp.setattr(linearizer_mod, "_screen_and_solve_one_pair", _spy)
    try:
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
    finally:
        mp.undo()

    sub_sets = [g.contig_had_subtracted_homolog for g in captured_guards if g.contig_had_subtracted_homolog]
    assert len(sub_sets) == 1
    assert all(isinstance(k, int) for k in sub_sets[0])

    ins_evs = [ev for ev in res.sv_events if ev.event_type == "INS"]
    assert len(ins_evs) == 1
    gt = ins_evs[0].genotypes
    assert gt["Ref#1"] == "0"
    assert gt["Hap1#1"] == "1"
    assert gt["Hap2#1"] == "."
    assert gt["Hap3#1"] == "0"
    assert all(u.reason != "REFERENCE_LIKE" for u in res.unresolved_records)
    assert int(res.summary_metrics["dedup_ref_masked_bp"]) == 1200


# ============================================================================
# 4. Module B (`panflat/linearizer.py` Stage 2.5b)
# ============================================================================


def test_module_b_childless_secondary_backbone_unlocks_clustering(tmp_path: Path) -> None:
    """Module B: childless multi-member secondary backbone unlocks similarity merge under ref/pan."""
    gb = SyntheticGFAFixtureBuilder()
    r1 = make_complex_seq(400, seed=401)
    r2 = make_complex_seq(400, seed=402)
    fl = make_complex_seq(250, seed=403)
    b1 = make_complex_seq(20, seed=404)
    b2 = make_complex_seq(20, seed=405)
    fr = make_complex_seq(250, seed=406)
    bsec_seq = fl + b1 + fr
    real_99 = mutate_seq_snps(bsec_seq, [40, 140, 320, 460])  # 4/520 = 99.2%

    for nid, seq in [
        ("1", r1),
        ("2", r2),
        ("3", fl),
        ("4", b1),
        ("5", b2),
        ("6", fr),
        ("7", real_99),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    gb.add_p_path("H1#1#Chr01", [("1", "+"), ("3", "+"), ("4", "+"), ("6", "+"), ("2", "+")])
    gb.add_p_path("H2#1#Chr01", [("1", "+"), ("3", "+"), ("5", "+"), ("6", "+"), ("2", "+")])
    gb.add_p_path("H3#1#Chr01", [("1", "+"), ("7", "+"), ("2", "+")])

    g_off, c_off = _build_graph_and_config(tmp_path, "modb_off", gb, ref_steps, dedup_mode="off")
    res_off = run_chromosome_pipeline(g_off, c_off)  # type: ignore[arg-type]
    # Under off: multi-member component stays is_synthetic=True -> 2 reps (520 + 100 + 520 = 1140)
    assert res_off.step_offsets[1].slot_ins_len == len(bsec_seq) + 100 + len(real_99)

    for mode in ("ref", "pan"):
        g_on, c_on = _build_graph_and_config(
            tmp_path, f"modb_{mode}", gb, ref_steps, dedup_mode=mode
        )
        res_on = run_chromosome_pipeline(g_on, c_on)  # type: ignore[arg-type]
        # Under ref/pan: childless secondary backbone unlocks is_synthetic=False -> merges into 1 rep!
        assert res_on.step_offsets[1].slot_ins_len == len(bsec_seq)
        assert len(res_on.inserts_rows) == 1
        assert set(str(res_on.inserts_rows[0]["members"]).split(",")) == {
            "H1#1",
            "H2#1",
            "H3#1",
        }


def test_module_b_m_child_deletion_vs_rep0_genotyping_guard(tmp_path: Path) -> None:
    """Module B: when `b_sec` merges into `rep0`, sibling `m_child` carrying >=50bp DEL calls '.' on INS and '1' on DEL."""
    gb = SyntheticGFAFixtureBuilder()
    r1 = make_complex_seq(400, seed=410)
    d1 = make_complex_seq(500, seed=411)
    d2 = make_complex_seq(500, seed=412)
    r4 = make_complex_seq(400, seed=413)
    rep0_seq = d1 + d2  # 1000 bp

    # 1 SNP per 50 bp block = 49/50 = 0.98:
    # - Untouched by Module A at dedup_id=0.99 (0.98 < 0.99)
    # - Passes Module B similarity merge into rep0 (0.98 >= cluster_id=0.95)
    bsec_seq = _mutate_per_50bp_block(rep0_seq, (25,))
    s1 = bsec_seq[:400]
    s2 = bsec_seq[400:600]  # 200 bp internal segment deleted in m_child!
    s3 = bsec_seq[600:]
    novel_seq = make_complex_seq(600, seed=414)

    for nid, seq in [
        ("1", r1),
        ("2", d1),
        ("3", d2),
        ("4", r4),
        ("5", s1),
        ("6", s2),
        ("7", s3),
        ("8", novel_seq),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    # Two b_sec haps -> support=2 wins secondary backbone
    gb.add_p_path("Bsec1#1#Chr01", [("1", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("4", "+")])
    gb.add_p_path("Bsec2#1#Chr01", [("1", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("4", "+")])
    # m_child shares nodes 5 & 7 with b_sec, deleting 200 bp node 6 -> active_child_solutions == []
    gb.add_p_path("Mchild#1#Chr01", [("1", "+"), ("5", "+"), ("7", "+"), ("4", "+")])
    # Unrelated novel insertion so an INS event is emitted at G(0, 3)
    gb.add_p_path("Novel#1#Chr01", [("1", "+"), ("8", "+"), ("4", "+")])
    gb.add_p_path("RefCopy#1#Chr01", ref_steps)

    graph, cfg = _build_graph_and_config(
        tmp_path,
        "modb_rep0_guard",
        gb,
        ref_steps,
        dedup_mode="ref",
        dedup_id=0.99,
    )
    res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]

    # Confirm Module A did NOT subtract b_sec or m_child
    assert int(res.summary_metrics["dedup_ref_masked_bp"]) == 0

    ins_evs = [ev for ev in res.sv_events if ev.event_type == "INS"]
    del_evs = [ev for ev in res.sv_events if ev.event_type == "DEL"]
    assert len(ins_evs) == 1
    assert len(del_evs) >= 1

    ins_gt = ins_evs[0].genotypes
    assert ins_gt["Ref#1"] == "0"
    assert ins_gt["RefCopy#1"] == "0"
    assert ins_gt["Bsec1#1"] == "0"
    assert ins_gt["Bsec2#1"] == "0"
    assert ins_gt["Novel#1"] == "1"
    assert ins_gt["Mchild#1"] == ".", f"m_child deletion carrier falsely genotyped as {ins_gt['Mchild#1']!r}"

    assert any(ev.genotypes.get("Mchild#1") == "1" for ev in del_evs)

    # Also verify directly in solve_parallel_group that b_sec is in rep0_cand_spans and m_child is NOT
    cand_bsec1 = ExtractedCandidate(
        candidate_id="c_b1",
        parent_id=None,
        hap_id="Bsec1#1",
        src_path="Bsec1#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=3,
        source_order=0,
        steps=(Step("5", "+"), Step("6", "+"), Step("7", "+")),
        provenance=(
            StepProvenance("Bsec1#1", "Bsec1#1#Chr01", 400, 800, "+"),
            StepProvenance("Bsec1#1", "Bsec1#1#Chr01", 800, 1000, "+"),
            StepProvenance("Bsec1#1", "Bsec1#1#Chr01", 1000, 1400, "+"),
        ),
        seq=bsec_seq,
    )
    cand_bsec2 = ExtractedCandidate(
        candidate_id="c_b2",
        parent_id=None,
        hap_id="Bsec2#1",
        src_path="Bsec2#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=3,
        source_order=1,
        steps=(Step("5", "+"), Step("6", "+"), Step("7", "+")),
        provenance=(
            StepProvenance("Bsec2#1", "Bsec2#1#Chr01", 400, 800, "+"),
            StepProvenance("Bsec2#1", "Bsec2#1#Chr01", 800, 1000, "+"),
            StepProvenance("Bsec2#1", "Bsec2#1#Chr01", 1000, 1400, "+"),
        ),
        seq=bsec_seq,
    )
    cand_mchild = ExtractedCandidate(
        candidate_id="c_mc",
        parent_id=None,
        hap_id="Mchild#1",
        src_path="Mchild#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=3,
        source_order=2,
        steps=(Step("5", "+"), Step("7", "+")),
        provenance=(
            StepProvenance("Mchild#1", "Mchild#1#Chr01", 400, 800, "+"),
            StepProvenance("Mchild#1", "Mchild#1#Chr01", 800, 1200, "+"),
        ),
        seq=s1 + s3,
    )
    sol = solve_parallel_group(
        left_anchor_rank=0,
        right_anchor_rank=3,
        backbone_slice_steps=(Step("2", "+"), Step("3", "+")),
        candidates=[cand_bsec1, cand_bsec2, cand_mchild],
        graph=graph,  # type: ignore[arg-type]
        config=cfg,
    )
    assert sol.non_backbone_reps == ()
    assert "c_b1" in sol.rep0_cand_spans and "c_b2" in sol.rep0_cand_spans
    assert "c_mc" not in sol.rep0_cand_spans


# ============================================================================
# 5. Module C (`panflat/linearizer.py` Stage 5.5)
# ============================================================================


def test_subslice_piece_plus_and_minus_strands() -> None:
    """Module C: `_subslice_piece` maps half-open intervals accurately on '+' and '-' strands."""
    full_src = make_complex_seq(2000, seed=501)

    p_plus = Piece(
        src_hap="H1#1",
        src_path="H1#1#Chr01",
        src_start=1000,
        src_end=1500,
        src_strand="+",
        seq=full_src[1000:1500],
        join_before="source_contiguous",
        members=("H1#1",),
    )
    sub_p = _subslice_piece(p_plus, 120, 370, "synthetic")
    assert sub_p.src_start == 1120 and sub_p.src_end == 1370
    assert sub_p.seq == full_src[1120:1370] == p_plus.seq[120:370]
    assert sub_p.join_before == "synthetic"

    p_minus = Piece(
        src_hap="H2#1",
        src_path="H2#1#Chr01",
        src_start=1000,
        src_end=1500,
        src_strand="-",
        seq=revcomp_dna(full_src[1000:1500]),
        join_before="source_contiguous",
        members=("H2#1",),
    )
    sub_m = _subslice_piece(p_minus, 120, 370, "spacer")
    assert sub_m.src_start == 1130 and sub_m.src_end == 1380
    assert sub_m.seq == revcomp_dna(full_src[1130:1380]) == p_minus.seq[120:370]
    assert sub_m.join_before == "spacer"


def test_module_c_all_7_join_before_transitions_and_piece_cand_ids_lockstep(
    tmp_path: Path,
) -> None:
    """Module C: verify all 7 `join_before` transitions, untouched `<min_sv_len` preservation, and lockstep `piece_cand_ids`."""
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 12):
        gb.add_node(f"r{i}", make_complex_seq(200, seed=510 + i))
    ref_steps = [(f"r{i}", "+") for i in range(1, 12)]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    graph, cfg = _build_graph_and_config(
        tmp_path, "modc_7trans", gb, ref_steps, dedup_mode="pan", dedup_id=0.90
    )

    # Seed homologous sequences via Priority-0 CEN_ISLANDs
    h0 = make_complex_seq(400, seed=530)
    h1 = make_complex_seq(400, seed=531)
    h2 = make_complex_seq(400, seed=532)
    nov = [make_complex_seq(250, seed=540 + i) for i in range(16)]
    short_30 = make_complex_seq(30, seed=560)  # untouched < min_sv_len=50

    def _mk_piece(
        seq: str,
        jb: str = "source_contiguous",
        strand: str = "+",
        hap: str = "H1#1",
    ) -> Piece:
        return Piece(
            src_hap=hap,
            src_path=f"{hap}#Chr01",
            src_start=100,
            src_end=100 + len(seq),
            src_strand=strand,  # type: ignore[arg-type]
            seq=seq,
            join_before=jb,  # type: ignore[arg-type]
            members=(hap,),
        )

    cen_blk = InsertionSlotBlock(
        right_anchor_ref_step=1,
        is_centromere_island=True,
        group_span=1,
        left_anchor_ref_step=0,
        event_id="Chr01_CEN_ISLAND_1",
        representatives=(
            (_mk_piece(h0, "spacer", hap="Cen#1"),),
            (_mk_piece(h1, "spacer", hap="Cen#1"),),
            (_mk_piece(h2, "spacer", hap="Cen#1"),),
        ),
        trailing_spacer=True,
    )
    cen_out = CentromereOutput(
        centromere_rows=(),
        island_slot_blocks=(cen_blk,),
        island_sv_events=(),
        unresolved_records=(),
        json_log_entries=(),
    )

    dummy_ex = ExtractedCandidate(
        candidate_id="ex",
        parent_id=None,
        hap_id="H1#1",
        src_path="H1#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=1,
        source_order=0,
        steps=(),
        provenance=(),
        seq="",
    )

    def _mk_rep(
        r_idx: int,
        pieces: Sequence[Piece],
        pc_ids: Sequence[frozenset[str]],
        tok: str,
        sup: int = 1,
    ) -> ClusterRepresentative:
        return ClusterRepresentative(
            rep_idx=r_idx,
            is_synthetic=True,
            seq="",
            length_bp_no_n=sum(len(p.seq) for p in pieces),
            pieces=tuple(pieces),
            support_haps=set(),
            token_str=tok,
            min_hap="H1#1",
            exemplar=dummy_ex,
            member_occurrences=[],
            nested_choices_by_cand_id={},
            reconstructed_len_by_cand_id={},
            piece_cand_ids=tuple(pc_ids),
            support_hap_count=sup,
        )

    # Construct 8 reps across distinct groups testing all 7 transitions + left-trimmed split
    specs: list[tuple[tuple[int, int], ClusterRepresentative, list[str]]] = [
        # (1) Dropped P0 -> P1 becomes p_i=0 with join_before="synthetic" (no orphan "spacer")
        (
            (0, 1),
            _mk_rep(
                1,
                [_mk_piece(h0, "source_contiguous"), _mk_piece(nov[0], "spacer")],
                [frozenset({"c1a"}), frozenset({"c1a", "c1b"})],
                "t1",
            ),
            ["c1a", "c1b"],
        ),
        # (2) Tail-trimmed P0 -> P1 becomes "spacer"
        (
            (1, 2),
            _mk_rep(
                1,
                [_mk_piece(nov[1] + h0, "source_contiguous"), _mk_piece(nov[2], "synthetic")],
                [frozenset({"c2a"}), frozenset({"c2b"})],
                "t2",
            ),
            ["c2a", "c2b"],
        ),
        # (3) Left-trimmed P1 -> P1 becomes "spacer"
        (
            (2, 3),
            _mk_rep(
                1,
                [_mk_piece(nov[3], "source_contiguous"), _mk_piece(h0 + nov[4], "synthetic")],
                [frozenset({"c3a"}), frozenset({"c3b"})],
                "t3",
            ),
            ["c3a", "c3b"],
        ),
        # (4) Dropped middle P1 -> P2 becomes "spacer", piece_cand_ids aligned
        (
            (3, 4),
            _mk_rep(
                1,
                [
                    _mk_piece(nov[5], "source_contiguous"),
                    _mk_piece(h1, "synthetic"),
                    _mk_piece(nov[6], "synthetic"),
                ],
                [frozenset({"c4a"}), frozenset({"c4b"}), frozenset({"c4c"})],
                "t4",
            ),
            ["c4a", "c4b", "c4c"],
        ),
        # (5a) Single piece split into 2 subpieces -> 1st keeps "source_contiguous", 2nd gets "spacer"
        (
            (4, 5),
            _mk_rep(
                1,
                [_mk_piece(nov[7] + h2 + nov[8], "source_contiguous")],
                [frozenset({"c5a", "c5b"})],
                "t5a",
            ),
            ["c5a", "c5b"],
        ),
        # (5b) Left-trimmed + split single piece -> 1st gets "synthetic", 2nd gets "spacer"
        (
            (5, 6),
            _mk_rep(
                1,
                [_mk_piece(h0 + nov[9] + h1 + nov[10], "source_contiguous")],
                [frozenset({"c5c"})],
                "t5b",
            ),
            ["c5c"],
        ),
        # (6) Untouched multi-piece preserved identically (partial_content remains False)
        (
            (6, 7),
            _mk_rep(
                1,
                [_mk_piece(nov[11], "source_contiguous"), _mk_piece(nov[12], "synthetic")],
                [frozenset({"c6a"}), frozenset({"c6b"})],
                "t6",
            ),
            ["c6a", "c6b"],
        ),
        # (7) Untouched < min_sv_len (30 bp) sub-piece preserved intact
        (
            (7, 8),
            _mk_rep(
                1,
                [_mk_piece(short_30, "source_contiguous"), _mk_piece(nov[13], "synthetic")],
                [frozenset({"c7a"}), frozenset({"c7b"})],
                "t7",
            ),
            ["c7a", "c7b"],
        ),
    ]

    sorted_pairs = [pair for pair, _, _ in specs]
    guards = {pair: GroupRepeatGuardResult() for pair in sorted_pairs}
    sols: dict[tuple[int, int], ParallelGroupSolution] = {}
    for pair, rep, cids in specs:
        sols[pair] = ParallelGroupSolution(
            left_anchor_rank=pair[0],
            right_anchor_rank=pair[1],
            has_nonempty_backbone=False,
            backbone_seq="",
            non_backbone_reps=(rep,),
            cand_to_rep_idx={c: 1 for c in cids},
            cand_to_nested_sig={c: () for c in cids},
            cand_to_reconstructed_len={c: rep.length_bp_no_n for c in cids},
            cand_to_flags={c: (False, False, False) for c in cids},
        )

    pan_m_bp, pan_sr_bp, dropped_g = _deduplicate_representatives_across_chromosome(
        graph=graph,  # type: ignore[arg-type]
        config=cfg,
        sorted_group_pairs=sorted_pairs,
        group_guard_by_pair=guards,
        group_solutions=sols,
        cen_out=cen_out,
    )

    assert pan_m_bp == 7 * 400  # h0 x4 + h1 x2 + h2 x1
    assert pan_sr_bp == 0
    assert dropped_g == 0

    # Check (1) Dropped P0 -> synthetic, lockstep piece_cand_ids, per-allele partial isolation
    r1 = sols[(0, 1)].non_backbone_reps[0]
    assert len(r1.pieces) == 1
    assert r1.pieces[0].seq == nov[0]
    assert r1.pieces[0].join_before == "synthetic"
    assert r1.piece_cand_ids == (frozenset({"c1a", "c1b"}),)
    assert sols[(0, 1)].cand_to_flags["c1a"][0] is True
    assert sols[(0, 1)].cand_to_flags["c1b"][0] is False

    # Check (2) Tail-trimmed P0 -> P1 spacer
    r2 = sols[(1, 2)].non_backbone_reps[0]
    assert [p.seq for p in r2.pieces] == [nov[1], nov[2]]
    assert [p.join_before for p in r2.pieces] == ["source_contiguous", "spacer"]
    assert r2.piece_cand_ids == (frozenset({"c2a"}), frozenset({"c2b"}))

    # Check (3) Left-trimmed P1 -> P1 spacer
    r3 = sols[(2, 3)].non_backbone_reps[0]
    assert [p.seq for p in r3.pieces] == [nov[3], nov[4]]
    assert [p.join_before for p in r3.pieces] == ["source_contiguous", "spacer"]
    assert r3.piece_cand_ids == (frozenset({"c3a"}), frozenset({"c3b"}))

    # Check (4) Dropped middle P1 -> P2 spacer, lockstep piece_cand_ids, c4b remapped to -1
    r4 = sols[(3, 4)].non_backbone_reps[0]
    assert [p.seq for p in r4.pieces] == [nov[5], nov[6]]
    assert [p.join_before for p in r4.pieces] == ["source_contiguous", "spacer"]
    assert r4.piece_cand_ids == (frozenset({"c4a"}), frozenset({"c4c"}))
    assert sols[(3, 4)].cand_to_rep_idx["c4b"] == -1
    assert "c4b" not in sols[(3, 4)].cand_to_reconstructed_len
    assert "c4b" not in sols[(3, 4)].cand_to_nested_sig
    assert sols[(3, 4)].cand_to_flags["c4b"][0] is True
    assert sols[(3, 4)].cand_to_flags["c4a"][0] is False
    assert sols[(3, 4)].cand_to_flags["c4c"][0] is False

    # Check (5a) Single piece split into 2 -> source_contiguous, spacer, duplicated cand_ids
    r5a = sols[(4, 5)].non_backbone_reps[0]
    assert [p.seq for p in r5a.pieces] == [nov[7], nov[8]]
    assert [p.join_before for p in r5a.pieces] == ["source_contiguous", "spacer"]
    assert r5a.piece_cand_ids == (frozenset({"c5a", "c5b"}), frozenset({"c5a", "c5b"}))

    # Check (5b) Left-trimmed + split -> synthetic, spacer
    r5b = sols[(5, 6)].non_backbone_reps[0]
    assert [p.seq for p in r5b.pieces] == [nov[9], nov[10]]
    assert [p.join_before for p in r5b.pieces] == ["synthetic", "spacer"]
    assert r5b.piece_cand_ids == (frozenset({"c5c"}), frozenset({"c5c"}))

    # Check (6) Untouched multi-piece preserved identically (partial_content stays False)
    r6 = sols[(6, 7)].non_backbone_reps[0]
    assert [p.seq for p in r6.pieces] == [nov[11], nov[12]]
    assert [p.join_before for p in r6.pieces] == ["source_contiguous", "synthetic"]
    assert r6.partial_content is False
    assert sols[(6, 7)].cand_to_flags["c6a"][0] is False

    # Check (7) Untouched < min_sv_len sub-piece preserved intact
    r7 = sols[(7, 8)].non_backbone_reps[0]
    assert [p.seq for p in r7.pieces] == [short_30, nov[13]]
    assert [p.join_before for p in r7.pieces] == ["source_contiguous", "synthetic"]
    assert r7.partial_content is False


def test_module_c_seed_gate_small_budget_waves_carryover_cen_and_thread_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module C: dual-regime seed gate (`<=500` 13-mer + `>500` 19-mer), small-budget waves/subwaves/carryover, CEN_ISLAND Priority-0, and t1==t4."""
    # (a) Direct verification of dual-regime `_TailSeedGate` on <=500 bp and >500 bp queries
    target = make_complex_seq(400, seed=570)
    gate = _TailSeedGate()
    tail_list = [target]

    disjoint_short = make_complex_seq(300, seed=571)
    assert gate.has_seed_hit(disjoint_short, tail_list) is False

    # 20 bp match (span >= 7 on 13-mers -> min_chain_score=20) on '+' and '-' strands in <=500 bp query
    for hit_seq in (target[100:120], revcomp_dna(target[100:120])):
        q_s = disjoint_short[:130] + hit_seq + disjoint_short[150:]
        assert len(q_s) <= 500
        assert gate.has_seed_hit(q_s, tail_list) is True

    # Two separated 13 bp matches on same strand (with small indel) also hit <=500 bp gate
    two_13_fwd = target[50:63] + "ACGTACGTAC" + target[73:86]
    for hit_pair in (two_13_fwd, revcomp_dna(two_13_fwd)):
        q_pair = disjoint_short[:100] + hit_pair + disjoint_short[100 + len(hit_pair) :]
        assert len(q_pair) <= 500
        assert gate.has_seed_hit(q_pair, tail_list) is True

    # >500 bp query: disjoint misses; 40 bp exact match (min_chain_score=40) hits on '+' and '-' strands
    disjoint_long = make_complex_seq(800, seed=572)
    assert gate.has_seed_hit(disjoint_long, tail_list) is False
    for pos in (0, 210, 213, 216, 760):
        for hit40 in (target[50:90], revcomp_dna(target[50:90])):
            q_l = disjoint_long[:pos] + hit40 + disjoint_long[pos + 40 :]
            assert len(q_l) > 500
            assert gate.has_seed_hit(q_l, tail_list) is True
    gate.release()

    # (b) Small-budget multi-wave + mini-subwave + lazy-dirty tail_wave + mid-batch seal carryover
    #     + Priority-0 CEN_ISLAND seeding with threads=1 vs threads=4 exact parity
    monkeypatch.setattr(linearizer_mod, "_PAN_DEDUP_WAVE_BUDGET_BP", 2_400)
    monkeypatch.setattr(linearizer_mod, "_PAN_DEDUP_SUBWAVE_BP", 800)
    monkeypatch.setattr(linearizer_mod, "_PAN_DEDUP_QUERY_BATCH_BP", 3_200)

    gb = SyntheticGFAFixtureBuilder()
    # 20 ref nodes (nodes 1..20), centromere at node 10 (1000 bp)
    for i in range(1, 21):
        rlen = 1000 if i == 10 else 250
        gb.add_node(str(i), make_complex_seq(rlen, seed=600 + i))

    cen_island_seq = make_complex_seq(900, seed=650)
    cen_copy_96 = _mutate_per_50bp_block(cen_island_seq, (5, 30))  # 48/50 = 96%

    # 12 novel 550 bp seeds (6,600 bp total -> >=2 sealed 2,400 bp waves + multiple 800 bp subwaves)
    nov_blocks = [make_complex_seq(550, seed=660 + k) for k in range(12)]
    dup_block_0_exact = nov_blocks[0]
    dup_block_0_96_rc = revcomp_dna(_mutate_per_50bp_block(nov_blocks[0], (6, 32)))
    dup_block_4_96 = _mutate_per_50bp_block(nov_blocks[4], (7, 33))
    flanked_priv = (
        _mutate_per_50bp_block(nov_blocks[2], (5, 30))
        + make_complex_seq(250, seed=699)
        + _mutate_per_50bp_block(nov_blocks[3], (5, 30))
    )

    nid_counter = 21

    def _add_alt_node(seq: str) -> str:
        nonlocal nid_counter
        nid = str(nid_counter)
        nid_counter += 1
        gb.add_node(nid, seq)
        return nid

    cen_nid = _add_alt_node(cen_island_seq)
    cen_copy_nid = _add_alt_node(cen_copy_96)
    nov_nids = [_add_alt_node(s) for s in nov_blocks]
    dup0_ex_nid = _add_alt_node(dup_block_0_exact)
    dup0_rc_nid = _add_alt_node(dup_block_0_96_rc)
    dup4_nid = _add_alt_node(dup_block_4_96)
    flanked_nid = _add_alt_node(flanked_priv)

    ref_steps = [(str(i), "+") for i in range(1, 21)]
    gb.add_p_path("Ref#1#Chr01", ref_steps)

    # Helper to insert nodes between ref nodes (pos -> pos+1) and optionally replace centromere node 10
    def _make_path(ins_after: dict[int, str], replace_cen: str | None = None) -> list[tuple[str, str]]:
        steps: list[tuple[str, str]] = []
        for idx in range(1, 21):
            if idx == 10 and replace_cen is not None:
                steps.append((replace_cen, "+"))
            else:
                steps.append((str(idx), "+"))
            if idx in ins_after:
                steps.append((ins_after[idx], "+"))
        return steps

    # H1 & H2 carry high-priority copies of nov_blocks[0..5] + CEN_ISLAND
    gb.add_p_path(
        "H1#1#Chr01",
        _make_path(
            {1: nov_nids[0], 2: nov_nids[1], 3: nov_nids[2], 4: nov_nids[3], 5: nov_nids[4], 6: nov_nids[5]},
            replace_cen=cen_nid,
        ),
    )
    gb.add_p_path(
        "H2#1#Chr01",
        _make_path(
            {1: nov_nids[0], 2: nov_nids[1], 3: nov_nids[2], 4: nov_nids[3], 5: nov_nids[4], 6: nov_nids[5]},
            replace_cen=cen_nid,
        ),
    )
    # H3 carries nov_blocks[6..11] at loci 7..8 and 11..14
    gb.add_p_path(
        "H3#1#Chr01",
        _make_path(
            {7: nov_nids[6], 8: nov_nids[7], 11: nov_nids[8], 12: nov_nids[9], 13: nov_nids[10], 14: nov_nids[11]}
        ),
    )
    # H4 carries lower-priority duplicates across shifted loci (exact, revcomp 96%, tail_wave 96%,
    # flanked private novel, AND a normal-arm 96% copy of CEN_ISLAND!)
    gb.add_p_path(
        "H4#1#Chr01",
        _make_path(
            {
                15: dup0_ex_nid,
                16: dup0_rc_nid,
                17: dup4_nid,
                18: flanked_nid,
                19: cen_copy_nid,
            }
        ),
    )

    cen_s = 9 * 250
    cen_e = cen_s + 1000
    g1, c1 = _build_graph_and_config(
        tmp_path,
        "modc_waves",
        gb,
        ref_steps,
        bed_rows=[("Chr01", cen_s, cen_e, "CEN_W")],
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
        centromere_min_island=500,
        dedup_mode="pan",
        dedup_id=0.90,
        threads=1,
    )
    c4 = BuildConfig(
        gfa=c1.gfa,
        ref_fa=c1.ref_fa,
        backbone_prefix=c1.backbone_prefix,
        out_prefix=str(tmp_path / "modc_waves_t4_out"),
        centromere_bed=c1.centromere_bed,
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
        centromere_min_island=500,
        dedup_mode="pan",
        dedup_id=0.90,
        threads=4,
    )

    res_t1 = run_chromosome_pipeline(g1, c1)  # type: ignore[arg-type]
    res_t4 = run_chromosome_pipeline(g1, c4)  # type: ignore[arg-type]

    assert res_t1.pan_seq == res_t4.pan_seq
    assert res_t1.summary_metrics == res_t4.summary_metrics
    assert res_t1.inserts_rows == res_t4.inserts_rows
    assert res_t1.sv_events == res_t4.sv_events

    # Verify Priority-0 CEN_ISLAND suppressed the normal-arm 900 bp copy at locus 19..20,
    # pure duplicate groups at 15..16, 16..17, 17..18 were dropped, and locus 18..19 retained ~250 bp novel!
    assert int(res_t1.summary_metrics["dedup_pan_masked_bp"]) >= 900 + 550 * 5 - 100
    assert int(res_t1.summary_metrics["dedup_dropped_groups"]) == 4
    # Locus 19..20 (step 19) must have 0 slot_ins_len because CEN_ISLAND Priority-0 subtracted it
    assert res_t1.step_offsets[19].slot_ins_len == 0
    # Locus 18..19 (step 18) retained ONLY the ~250 bp private novel middle (not 1,350 bp!)
    assert res_t1.step_offsets[18].slot_ins_len == 250

    # Also verify ThreadBuffer-absent fallback produces identical result
    monkeypatch.delattr(mappy, "ThreadBuffer", raising=False)
    res_fb = run_chromosome_pipeline(g1, c4)  # type: ignore[arg-type]
    assert res_fb.pan_seq == res_t1.pan_seq
    assert res_fb.summary_metrics == res_t1.summary_metrics


def test_module_c_shared_locus_dot_genotyping_intra_piece_dup_and_lazy_tail_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module C & D: shared-locus rep compaction + '.' genotyping, intra-piece split dup, and lazy-dirty tail_wave."""
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 10):
        gb.add_node(str(i), make_complex_seq(250, seed=680 + i))

    # Base 400 bp sequence mounted at locus (1, 2) with high support (2 haps -> Priority 1)
    base_hom = make_complex_seq(400, seed=690)
    # 3 consecutive 96%-similar copies at loci (2, 3), (3, 4), (4, 5) (< SUBWAVE_BP=128kb)
    # -> lazy-dirty tail_wave MUST build ONLY ONCE on the first duplicate and stay clean for the rest!
    copy_a = _mutate_per_50bp_block(base_hom, (5, 35))
    copy_b = _mutate_per_50bp_block(base_hom, (6, 36))
    copy_c = _mutate_per_50bp_block(base_hom, (7, 37))

    # Shared locus (5, 6):
    # - HapA carries copy_d (96% similar to base_hom -> 100% subtracted in Module C -> r_idx=-1 -> '.')
    # - HapB carries clean_nov (600 bp novel -> survives, reindexed to 1, NO partial_content=1)
    copy_d = _mutate_per_50bp_block(base_hom, (8, 38))
    clean_nov = make_complex_seq(600, seed=691)

    # Intra-piece split duplicate at locus (6, 7): X (300 bp) + copy_e (400 bp hom) + X_96 (300 bp)
    # -> copy_e subtracted, splitting piece into [X, X_96]; immediate acceptance of X subtracts X_96!
    intra_x = make_complex_seq(300, seed=692)
    intra_x_96 = _mutate_per_50bp_block(intra_x, (6, 36))
    copy_e = _mutate_per_50bp_block(base_hom, (9, 39))
    intra_dup_seq = intra_x + copy_e + intra_x_96

    for nid, seq in [
        ("10", base_hom),
        ("11", copy_a),
        ("12", copy_b),
        ("13", copy_c),
        ("14", copy_d),
        ("15", clean_nov),
        ("16", intra_dup_seq),
    ]:
        gb.add_node(nid, seq)

    ref_steps = [(str(i), "+") for i in range(1, 10)]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    # High-support carriers of base_hom at (1, 2)
    gb.add_p_path(
        "Sup1#1#Chr01",
        [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+"), ("9", "+")],
    )
    gb.add_p_path(
        "Sup2#1#Chr01",
        [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+"), ("9", "+")],
    )
    # HapA carries 3 consecutive copies + shared-locus copy_d at (5, 6)
    gb.add_p_path(
        "HapA#1#Chr01",
        [
            ("1", "+"),
            ("2", "+"),
            ("11", "+"),
            ("3", "+"),
            ("12", "+"),
            ("4", "+"),
            ("13", "+"),
            ("5", "+"),
            ("14", "+"),
            ("6", "+"),
            ("7", "+"),
            ("8", "+"),
            ("9", "+"),
        ],
    )
    # HapB carries clean_nov at shared locus (5, 6) and intra_dup_seq at (6, 7)
    gb.add_p_path(
        "HapB#1#Chr01",
        [
            ("1", "+"),
            ("2", "+"),
            ("3", "+"),
            ("4", "+"),
            ("5", "+"),
            ("15", "+"),
            ("6", "+"),
            ("16", "+"),
            ("7", "+"),
            ("8", "+"),
            ("9", "+"),
        ],
    )

    graph, cfg = _build_graph_and_config(
        tmp_path, "modc_shared_lazy", gb, ref_steps, dedup_mode="pan", dedup_id=0.90, threads=1
    )

    wave_build_calls: list[list[str]] = []
    orig_build = linearizer_mod._build_pan_dedup_wave_index

    def _spy_build(piece_seqs: Sequence[str]) -> object:
        wave_build_calls.append(list(piece_seqs))
        return orig_build(piece_seqs)

    monkeypatch.setattr(linearizer_mod, "_build_pan_dedup_wave_index", _spy_build)
    res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]

    # 1. Shared locus (5, 6) -> ref_pos = 5*250 = 1250: HapB='1', HapA='.', Ref#1='0', Sup1#1='0'
    shared_ins = [
        ev for ev in res.sv_events if ev.event_type == "INS" and ev.ref_start == 1250
    ]
    assert len(shared_ins) == 1
    assert shared_ins[0].genotypes["HapB#1"] == "1"
    assert shared_ins[0].genotypes["HapA#1"] == "."
    assert shared_ins[0].genotypes["Ref#1"] == "0"
    assert shared_ins[0].genotypes["Sup1#1"] == "0"
    assert "partial_content=1" not in shared_ins[0].alleles_str

    # 2. Intra-piece split duplicate at (6, 7) -> step 6: ONLY one 300 bp copy of intra_x survives!
    assert res.step_offsets[6].slot_ins_len == 300
    intra_ins = [
        ev for ev in res.sv_events if ev.event_type == "INS" and ev.ref_start == 1500
    ]
    assert len(intra_ins) == 1
    assert "partial_content=1" in intra_ins[0].alleles_str

    # 3. Lazy-dirty tail_wave: copies a, b, c, d all hit the SAME clean tail_wave without rebuilding!
    #    Total tail_wave builds across the entire chromosome should be <= 3
    #    (once for base_hom/copy_a..d, once when intra_dup_seq hits base_hom after clean_nov added,
    #     and once when intra_x_96 hits intra_x — NEVER 1 per copy!).
    assert len(wave_build_calls) <= 3, f"Expected lazy tail_wave reuse, got {len(wave_build_calls)} builds"
    assert int(res.summary_metrics["dedup_dropped_groups"]) == 3


# ============================================================================
# 6. Module D (`panflat/writers.py::merge_chromosomes`)
# ============================================================================


def _build_and_write_chrom_bundle(
    tmp_path: Path,
    chrom: str,
    seed_base: int,
    dedup_mode: str,
    dedup_id: float,
) -> tuple[Path, dict[str, object]]:
    """Build and write a single chromosome with non-zero ref/pan/short/dropped dedup counters."""
    gb = SyntheticGFAFixtureBuilder()
    r1 = make_complex_seq(400, seed=seed_base + 1)
    r2_te = make_complex_seq(800, seed=seed_base + 2)
    r3 = make_complex_seq(400, seed=seed_base + 3)
    r4 = make_complex_seq(400, seed=seed_base + 4)
    r5 = make_complex_seq(400, seed=seed_base + 5)

    # Pure ectopic ref TE + 20 bp short residual -> ref_masked_bp=800, short_residual_bp=20, dropped_groups+=1
    te_short = _mutate_per_50bp_block(r2_te, (6, 7, 43)) + make_complex_seq(20, seed=seed_base + 6)
    # Cross-group non-ref sequence mounted at r3..r4 and duplicated at r4..r5 (dropped in pan mode)
    pan_nov = make_complex_seq(600, seed=seed_base + 7)
    pan_dup = _mutate_per_50bp_block(pan_nov, (3, 46))

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
        gb.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+")]
    gb.add_p_path(f"Ref#1#{chrom}", ref_steps)
    gb.add_p_path(
        f"H1#1#{chrom}",
        [("1", "+"), ("6", "+"), ("2", "+"), ("3", "+"), ("7", "+"), ("4", "+"), ("5", "+")],
    )
    gb.add_p_path(
        f"H2#1#{chrom}",
        [("1", "+"), ("2", "+"), ("3", "+"), ("7", "+"), ("4", "+"), ("8", "+"), ("5", "+")],
    )

    graph, cfg = _build_graph_and_config(
        tmp_path,
        f"m_{chrom}_{dedup_mode}",
        gb,
        ref_steps,
        chrom=chrom,
        dedup_mode=dedup_mode,
        dedup_id=dedup_id,
    )
    res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
    write_build_outputs(res, cfg)
    return Path(cfg.out_prefix), dict(res.summary_metrics)


def _parse_summary_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line == "metric\tvalue":
            continue
        k, v = line.split("\t", 1)
        rows.append((k, v))
    return rows


def test_merge_chromosomes_dedup_associative_and_heterogeneous(tmp_path: Path) -> None:
    """Module D: associative 3-chrom merge (`flat == left == right`), 0 dup keys, and heterogeneous dedup_mode."""
    p1, m1 = _build_and_write_chrom_bundle(tmp_path, "Chr01", 700, "pan", 0.90)
    p2, m2 = _build_and_write_chrom_bundle(tmp_path, "Chr02", 720, "pan", 0.90)
    p3, m3 = _build_and_write_chrom_bundle(tmp_path, "Chr03", 740, "pan", 0.90)

    # Confirm non-vacuous dedup metrics on each single chromosome
    for m in (m1, m2, m3):
        assert int(m["dedup_ref_masked_bp"]) == 800
        assert int(m["dedup_pan_masked_bp"]) == 600
        assert int(m["dedup_short_residual_bp"]) == 20
        assert int(m["dedup_dropped_groups"]) == 2

    # Associative 3-chromosome merge: flat vs left-nested vs right-nested
    flat_p = tmp_path / "merged_flat"
    m12_p = tmp_path / "merged_12"
    left_p = tmp_path / "merged_left"
    m23_p = tmp_path / "merged_23"
    right_p = tmp_path / "merged_right"

    merge_chromosomes([p1, p2, p3], flat_p)
    merge_chromosomes([p1, p2], m12_p)
    merge_chromosomes([m12_p, p3], left_p)
    merge_chromosomes([p2, p3], m23_p)
    merge_chromosomes([p1, m23_p], right_p)

    flat_txt = Path(f"{flat_p}.summary.tsv").read_text(encoding="utf-8")
    left_txt = Path(f"{left_p}.summary.tsv").read_text(encoding="utf-8")
    right_txt = Path(f"{right_p}.summary.tsv").read_text(encoding="utf-8")
    assert flat_txt == left_txt == right_txt

    flat_rows = _parse_summary_rows(Path(f"{flat_p}.summary.tsv"))
    keys = [k for k, _ in flat_rows]
    assert len(keys) == len(set(keys)), "Duplicate metric keys found in merged summary.tsv"
    kv = dict(flat_rows)
    assert kv["dedup_mode"] == "pan"
    assert kv["dedup_id"] == "0.9"
    assert int(kv["dedup_ref_masked_bp"]) == 2400
    assert int(kv["dedup_pan_masked_bp"]) == 1800
    assert int(kv["dedup_short_residual_bp"]) == 60
    assert int(kv["dedup_dropped_groups"]) == 6

    # Heterogeneous merge: Chr01 (pan, 0.90) + Chr04 (ref, 0.95)
    p4, m4 = _build_and_write_chrom_bundle(tmp_path, "Chr04", 760, "ref", 0.95)
    assert int(m4["dedup_ref_masked_bp"]) == 0  # 94% TE retained under dedup_id=0.95
    assert int(m4["dedup_pan_masked_bp"]) == 0

    het_p = tmp_path / "merged_het"
    merge_chromosomes([p1, p4], het_p)
    het_rows = _parse_summary_rows(Path(f"{het_p}.summary.tsv"))
    het_keys = [k for k, _ in het_rows]
    assert len(het_keys) == len(set(het_keys))
    het_kv = dict(het_rows)
    # Conflicting mode/id omitted from top-level, preserved per-chrom, counters still summed
    assert "dedup_mode" not in het_kv
    assert "dedup_id" not in het_kv
    assert het_kv["Chr01:dedup_mode"] == "pan"
    assert het_kv["Chr04:dedup_mode"] == "ref"
    assert het_kv["Chr01:dedup_id"] == "0.9"
    assert het_kv["Chr04:dedup_id"] == "0.95"
    assert int(het_kv["dedup_ref_masked_bp"]) == 800
    assert int(het_kv["dedup_pan_masked_bp"]) == 600
    assert int(het_kv["dedup_short_residual_bp"]) == 20
    assert int(het_kv["dedup_dropped_groups"]) == 2


# ============================================================================
# 7. Iteration 2 Regression Tests (Bugs #1, C1, C2, C3 & Findings 1-4)
# ============================================================================


def test_regression_bug1_protected_ins_and_rejected_block_with_short_aligner() -> None:
    """Bug #1 regression: iterative zdrop re-mapping must not round outward across
    already-aligned flank bases adjacent to `protected_ins` (Variant 1A on +/- strands
    and Variant 1B rejected < min_identity block), while still recovering zdrop flanks.
    """
    # Variant 1A: 55 bp and 105 bp protected internal insertions at non-50-aligned
    # offsets [495, 550), [500, 555), [495, 600), [500, 605) on both '+' and '-' strands.
    rng = random.Random(42)
    ref = "".join(rng.choices("ACGT", k=20000))
    aln = mappy.Aligner(seq=ref, preset="asm20", best_n=50)
    saln = mappy.Aligner(seq=ref, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)

    for ins_len, fl_l_len, min_sv in (
        (55, 495, 50),
        (55, 500, 50),
        (55, 495, 55),
        (55, 500, 55),
        (105, 495, 105),
        (105, 500, 105),
    ):
        novel = "".join(rng.choices("ACGT", k=ins_len))
        q_fwd = ref[5000 : 5000 + fl_l_len] + novel + ref[5000 + fl_l_len : 5000 + fl_l_len + 500]
        exp_fwd = [(fl_l_len, fl_l_len + ins_len)]
        _, u_no_short, _ = subtract_reference_intervals_mappy(
            q_fwd, min_sv_len=min_sv, aligner=aln, short_aligner=None, min_identity=0.90
        )
        _, u_short, _ = subtract_reference_intervals_mappy(
            q_fwd, min_sv_len=min_sv, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert u_no_short == exp_fwd
        assert u_short == exp_fwd

        # Reverse-complement strand check
        q_rc = revcomp_dna(q_fwd)
        rc_s = len(q_fwd) - (fl_l_len + ins_len)
        rc_e = len(q_fwd) - fl_l_len
        exp_rc = [(rc_s, rc_e)]
        _, u_rc_no_short, _ = subtract_reference_intervals_mappy(
            q_rc, min_sv_len=min_sv, aligner=aln, short_aligner=None, min_identity=0.90
        )
        _, u_rc_short, _ = subtract_reference_intervals_mappy(
            q_rc, min_sv_len=min_sv, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert u_rc_no_short == exp_rc
        assert u_rc_short == exp_rc

    # Variant 1B: block 9 [450, 500) has 43M + 2X + 5I (86% < 90%); must NOT be re-mapped & masked.
    rng2 = random.Random(123)
    ref2 = "".join(rng2.choices("ACGT", k=10000))
    aln2 = mappy.Aligner(seq=ref2, preset="asm20", best_n=50)
    saln2 = mappy.Aligner(seq=ref2, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
    fl_l = list(ref2[4000:4495])
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    for p in (468, 472):
        fl_l[p] = swap[fl_l[p]]
    q2 = "".join(fl_l) + "".join(rng2.choices("ACGT", k=55)) + ref2[4495:4995]
    m_no_short, _, _ = subtract_reference_intervals_mappy(
        q2, min_sv_len=50, aligner=aln2, short_aligner=None, min_identity=0.90
    )
    m_short, _, _ = subtract_reference_intervals_mappy(
        q2, min_sv_len=50, aligner=aln2, short_aligner=saln2, min_identity=0.90
    )
    assert m_no_short == [(0, 450), (550, 1050)]
    assert m_short == [(0, 450), (550, 1050)]

    # Zdrop flank recovery still intact for >500 bp query with <=500 bp clipped homologous tail
    z_gap = "".join(rng2.choices("ACGT", k=150))
    z_q = ref2[1000:2200] + z_gap + revcomp_dna(ref2[7000:7050])
    m_z, u_z, bp_z = subtract_reference_intervals_mappy(
        z_q, min_sv_len=50, aligner=aln2, short_aligner=saln2, min_identity=0.90
    )
    assert m_z == [(0, 1200), (1350, 1400)] and u_z == [(1200, 1350)] and bp_z == 1250


def test_regression_synthetic_rep_compaction_and_depth2_unit(tmp_path: Path) -> None:
    """Finding 1 & 2 + Bug C1 (Paths 1 & 2) + Bug C2 regression:
    Verifies Probe 1C, Probe 2C, Scenario 5.3A (plus `--dedup-mode off` compatibility),
    Scenario 5.3B, and Probe 5d depth-2 nested synthetic rep with `verify_build_outputs`.
    """
    hom_l = make_complex_seq(1500, seed=7101)
    hom_m = make_complex_seq(1000, seed=7102)
    hom_r = make_complex_seq(1500, seed=7103)
    nov_13 = make_complex_seq(600, seed=7104)
    part_hom = make_complex_seq(300, seed=7105)
    part_nov = make_complex_seq(300, seed=7106)
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_two_hap(
        tag: str, sup_seq: str, node13_seq: str, seed_off: int, dedup_mode: str = "pan"
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=seed_off + i))
        gb.add_node("10", sup_seq)
        gb.add_node("11", hom_l)
        gb.add_node("14", hom_m)
        gb.add_node("12", hom_r)
        gb.add_node("13", node13_seq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for k in range(1, 5):
            gb.add_p_path(
                f"Sup{k}#1#Chr01",
                [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
            )
        gb.add_p_path(
            "Hap1#1#Chr01",
            [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")],
        )
        gb.add_p_path(
            "Hap2#1#Chr01",
            [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")],
        )
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=dedup_mode, dedup_id=0.90
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    # (1) Probe 1C: b_sec 100% dropped, child 13 survives
    res1c = _run_two_hap("u_probe1c", hom_l + hom_m + hom_r, nov_13, 7110)
    ev1c = next(e for e in res1c.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins1c = [r for r in res1c.inserts_rows if r["event_id"] == ev1c.event_id]
    assert ev1c.alleles_str == "1:len=600,src=Hap2#1,n=1,partial_content=1"
    assert ev1c.genotypes["Hap1#1"] == "." and ev1c.genotypes["Hap2#1"] == "1"
    assert len(ins1c) == 1 and ins1c[0]["allele"] == "1" and ins1c[0]["members"] == "Hap2#1"
    h1_logs = [e for e in res1c.json_log_entries if e.get("hap") == "Hap1#1"]
    assert h1_logs and all(e["final_reason"] == "PAN_HOMOLOG_SUBTRACTED" for e in h1_logs)

    # (2) Probe 2C: child 13 dropped, b_sec untouched -> Hap1 clean
    res2c = _run_two_hap("u_probe2c", nov_13, nov_13, 7120)
    ev2c = next(e for e in res2c.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev2c.alleles_str == "1:len=4000,src=Hap1#1,n=1;2:len=3000,src=Hap2#1,n=1,partial_content=1"

    # (3) Scenario 5.3A: child trimmed -300 bp -> Hap1 clean 4000, Hap2 exact 3300 (<= 3600)
    res3a_off = _run_two_hap("u_scen53a_off", part_hom, part_hom + part_nov, 7130, dedup_mode="off")
    res3a_pan = _run_two_hap("u_scen53a_pan", part_hom, part_hom + part_nov, 7130, dedup_mode="pan")
    ev3a_off = next(e for e in res3a_off.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ev3a_pan = next(e for e in res3a_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev3a_off.alleles_str == "1:len=4000,src=Hap1#1,n=1;2:len=3600,src=Hap2#1,n=1"
    assert ev3a_pan.alleles_str == "1:len=4000,src=Hap1#1,n=1;2:len=3300,src=Hap2#1,n=1,partial_content=1"

    # (4) Scenario 5.3B: 100% of Hap2 subtracted, only Hap1's replaced b_sec 14 survives
    res3b = _run_two_hap("u_scen53b", hom_l + nov_13 + hom_r, nov_13, 7140)
    ev3b = next(e for e in res3b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins3b = [r for r in res3b.inserts_rows if r["event_id"] == ev3b.event_id]
    assert ev3b.alleles_str == "1:len=1000,src=Hap1#1,n=1,partial_content=1"
    assert ev3b.genotypes["Hap1#1"] == "1" and ev3b.genotypes["Hap2#1"] == "."
    assert len(ins3b) == 1 and ins3b[0]["allele"] == "1" and ins3b[0]["members"] == "Hap1#1"

    # (5) Probe 5d: depth-2 nested synthetic rep, only Hap3's depth-2 piece Y survives
    gb5d = SyntheticGFAFixtureBuilder()
    for i in range(1, 5):
        gb5d.add_node(str(i), make_complex_seq(400, seed=7150 + i))
    s_a = make_complex_seq(1000, seed=7160)
    s_d = make_complex_seq(3000, seed=7161)
    s_b = make_complex_seq(1000, seed=7162)
    s_x1 = make_complex_seq(1000, seed=7163)
    s_d2 = make_complex_seq(1000, seed=7164)
    s_x2 = make_complex_seq(500, seed=7165)
    s_y = make_complex_seq(600, seed=7166)
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
        gb5d.add_node(nid, sq)
    ref4 = [(str(i), "+") for i in range(1, 5)]
    gb5d.add_p_path("Ref#1#Chr01", ref4)
    for k in range(1, 7):
        gb5d.add_p_path(f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+")])
    for tag_h in ("Hap1a", "Hap1b", "Hap1c"):
        gb5d.add_p_path(
            f"{tag_h}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("12", "+"), ("13", "+"), ("4", "+")]
        )
    gb5d.add_p_path(
        "Hap2#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("16", "+"), ("13", "+"), ("4", "+")],
    )
    gb5d.add_p_path(
        "Hap3#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("14", "+"), ("17", "+"), ("16", "+"), ("13", "+"), ("4", "+")],
    )
    g5d, c5d = _build_graph_and_config(tmp_path, "u_probe5d", gb5d, ref4, dedup_mode="pan", dedup_id=0.90)
    res5d = run_chromosome_pipeline(g5d, c5d)  # type: ignore[arg-type]
    write_build_outputs(res5d, c5d)
    v5d = verify_build_outputs(
        c5d.out_prefix,
        gfa_path=c5d.gfa,
        ref_fa_path=c5d.ref_fa,
        backbone_prefix=c5d.backbone_prefix,
        spacer_n=c5d.spacer_n,
    )
    assert v5d.passed, f"u_probe5d verify failed: {v5d.errors}"
    ev5d = next(e for e in res5d.sv_events if e.event_type == "INS" and e.anchors == "2..3")
    ins5d = [r for r in res5d.inserts_rows if r["event_id"] == ev5d.event_id]
    assert ev5d.alleles_str == "1:len=600,src=Hap3#1,n=1,partial_content=1"
    assert ev5d.genotypes["Hap1a#1"] == "." and ev5d.genotypes["Hap2#1"] == "."
    assert ev5d.genotypes["Hap3#1"] == "1"
    assert len(ins5d) == 1 and ins5d[0]["allele"] == "1" and ins5d[0]["members"] == "Hap3#1"


def test_regression_bug_c3_dual_regime_seed_gate_zero_fn_and_no_thrash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug C3 regression: dual-regime `_TailSeedGate` has 0 false negatives against live `mappy`
    across <=500/>500/+/-/tandem/'N' cases, lazy `aligner_short` is not eagerly built on >500bp
    waves, and 1,500 x 250 bp unique pieces runs with <1% FP, <3x amplification, and <=25 builds.
    """
    # Part 1: Live mappy zero-FN audit across +/- strands, boundary 500/501, tandem repeats, 'N's
    rng = random.Random(20261125)
    tail_pieces = [
        "".join(rng.choices("ACGT", k=1200)),
        "".join(rng.choices("ACGT", k=500)) + "N" * 15 + "".join(rng.choices("ACGT", k=400)),
        "ACGTACGTACGTTAG" * 30,  # tandem repeat
        "".join(rng.choices("ACGT", k=250)),
    ]
    wave = linearizer_mod._build_pan_dedup_wave_index(tail_pieces)
    # Lazy aligner_short must NOT be built until first <=500 bp query
    assert wave._aligner_short is None
    saln = wave.get_short_aligner()
    assert wave._aligner_short is saln and wave.wave_seq == ""

    gate = _TailSeedGate()
    swap = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}

    def _mut(seq: str, rate: float) -> str:
        arr = list(seq)
        for i, ch in enumerate(arr):
            if ch in swap and rng.random() < rate:
                arr[i] = rng.choice(swap[ch])
        return "".join(arr)

    skips = 0
    for trial in range(400):
        tp = rng.choice(tail_pieces)
        qlen = rng.choice([50, 75, 150, 250, 499, 500, 501, 550, 800])
        mlen = min(qlen, len(tp), rng.choice([25, 50, 100, 220]))
        st = rng.randint(0, len(tp) - mlen)
        seg = _mut(tp[st : st + mlen], rng.choice([0.0, 0.04, 0.10, 0.20]))
        if trial % 2 == 1:
            seg = revcomp_dna(seg)
        pad_l = rng.randint(0, qlen - len(seg))
        pad_r = qlen - len(seg) - pad_l
        q = "".join(rng.choices("ACGT", k=pad_l)) + seg + "".join(rng.choices("ACGT", k=pad_r))
        if not gate.has_seed_hit(q, tail_pieces):
            skips += 1
            m_ivs, _, b_m = subtract_reference_intervals_mappy(
                q,
                "",
                min_sv_len=50,
                aligner=wave.aligner_main,
                min_identity=0.80,
                short_aligner=saln,
                t_starts=wave.t_starts,
                t_ends=wave.t_ends,
            )
            assert not m_ivs and b_m == 0, f"Seed-gate false negative on qlen={qlen}, m_ivs={m_ivs}"
    assert skips > 0
    gate.release()

    # Part 2: 1,500 x 250 bp unique pieces (375 kb) -> <1% FP, <3x amplification, <=25 builds
    n_reps = 1500
    piece_len = 250
    rng_u = random.Random(20261126)
    seqs = ["".join(rng_u.choices("ACGT", k=piece_len)) for _ in range(n_reps)]
    ref_ranges = tuple((i * 100, (i + 1) * 100) for i in range(n_reps + 2))
    graph = SimpleNamespace(ref_step_ranges=ref_ranges)
    cfg = BuildConfig(
        gfa="x",
        ref_fa="y",
        backbone_prefix="Ref#1#",
        out_prefix="z",
        dedup_mode="pan",
        dedup_id=0.90,
        threads=1,
    )
    cen_out = CentromereOutput((), (), (), (), ())
    dummy_ex = ExtractedCandidate("ex", None, "H1#1", "H1#1#Chr01", 0, 1, 0, (), (), "")
    pairs: list[tuple[int, int]] = []
    guards: dict[tuple[int, int], GroupRepeatGuardResult] = {}
    sols: dict[tuple[int, int], ParallelGroupSolution] = {}
    for i, s in enumerate(seqs):
        pair = (i, i + 1)
        pairs.append(pair)
        guards[pair] = GroupRepeatGuardResult()
        p = Piece("H1#1", "H1#1#Chr01", 0, len(s), "+", s, "source_contiguous", ("H1#1",))
        rep = ClusterRepresentative(
            rep_idx=1,
            is_synthetic=False,
            seq="",
            length_bp_no_n=len(s),
            pieces=(p,),
            support_haps=set(),
            token_str=f"t{i:05d}",
            min_hap="H1#1",
            exemplar=dummy_ex,
            member_occurrences=[],
            nested_choices_by_cand_id={},
            reconstructed_len_by_cand_id={},
            piece_cand_ids=(frozenset({f"c{i}"}),),
            support_hap_count=1,
        )
        sols[pair] = ParallelGroupSolution(
            left_anchor_rank=i,
            right_anchor_rank=i + 1,
            has_nonempty_backbone=False,
            backbone_seq="",
            non_backbone_reps=(rep,),
            cand_to_rep_idx={f"c{i}": 1},
            cand_to_nested_sig={f"c{i}": ()},
            cand_to_reconstructed_len={f"c{i}": len(s)},
            cand_to_flags={f"c{i}": (False, False, False)},
        )

    orig_build = linearizer_mod._build_pan_dedup_wave_index
    orig_gate = linearizer_mod._TailSeedGate.has_seed_hit
    builds = 0
    indexed_bp = 0
    gate_calls = 0
    gate_true = 0

    def _spy_build(pseqs: Sequence[str]) -> object:
        nonlocal builds, indexed_bp
        builds += 1
        indexed_bp += sum(len(x) for x in pseqs)
        return orig_build(pseqs)

    def _spy_gate(self: _TailSeedGate, sub_seq: str, tp: Sequence[str]) -> bool:
        nonlocal gate_calls, gate_true
        gate_calls += 1
        hit = orig_gate(self, sub_seq, tp)
        if hit:
            gate_true += 1
        return hit

    monkeypatch.setattr(linearizer_mod, "_build_pan_dedup_wave_index", _spy_build)
    monkeypatch.setattr(linearizer_mod._TailSeedGate, "has_seed_hit", _spy_gate)
    pm, psr, dg = _deduplicate_representatives_across_chromosome(
        graph=graph,  # type: ignore[arg-type]
        config=cfg,
        sorted_group_pairs=pairs,
        group_guard_by_pair=guards,
        group_solutions=sols,
        cen_out=cen_out,
    )
    assert pm == 0 and psr == 0 and dg == 0
    assert gate_calls > 1000
    fp_rate = gate_true / gate_calls
    amp = indexed_bp / (n_reps * piece_len)
    assert fp_rate < 0.01, f"Seed-gate FP rate too high: {fp_rate:.4f}"
    assert builds <= 25, f"Too many wave index builds: {builds}"
    assert amp < 3.0, f"Indexing amplification too high: {amp:.2f}x"


def test_regression_finding3_nested_intra_piece_dup_and_finding4_jb(tmp_path: Path) -> None:
    """Finding 3 & Finding 4 regression:
    - Nested intra-piece split duplicate `A + T_existing + X + A + X` mounts only 2 pieces (`A`, `X`)
      and passes `verify_build_outputs`.
    - Untouched `orig_p_idx == 0` piece with `join_before='spacer'` is normalized to `'synthetic'`
      without falsely setting `partial_content=True`.
    """
    # Finding 3: A + T_existing + X + A + X
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 5):
        gb.add_node(str(i), make_complex_seq(400, seed=7200 + i))
    p_a = make_complex_seq(600, seed=7210)
    p_t = make_complex_seq(600, seed=7211)
    p_x = make_complex_seq(600, seed=7212)
    gb.add_node("10", p_t)
    gb.add_node("11", p_a + p_t + p_x + p_a + p_x)
    ref4 = [(str(i), "+") for i in range(1, 5)]
    gb.add_p_path("Ref#1#Chr01", ref4)
    for k in range(1, 4):
        gb.add_p_path(f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+")])
    gb.add_p_path("Hdup#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("4", "+")])

    graph, cfg = _build_graph_and_config(tmp_path, "u_find3", gb, ref4, dedup_mode="pan", dedup_id=0.90)
    res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
    write_build_outputs(res, cfg)
    v = verify_build_outputs(
        cfg.out_prefix,
        gfa_path=cfg.gfa,
        ref_fa_path=cfg.ref_fa,
        backbone_prefix=cfg.backbone_prefix,
        spacer_n=cfg.spacer_n,
    )
    assert v.passed, f"u_find3 verify failed: {v.errors}"
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "2..3")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert len(ins) == 2
    assert [r["pan_end"] - r["pan_start"] for r in ins] == [600, 600]
    assert [r["join_before"] for r in ins] == ["source_contiguous", "spacer"]

    # Finding 4: untouched orig_p_idx==0 piece with join_before="spacer" persisted as "synthetic"
    # without falsely setting partial_content=True
    p0 = Piece("H1#1", "H1#1#Chr01", 0, 300, "+", make_complex_seq(300, seed=7220), "spacer", ("H1#1",))
    dummy_ex = ExtractedCandidate("ex", None, "H1#1", "H1#1#Chr01", 0, 1, 0, (), (), "")
    rep0 = ClusterRepresentative(
        rep_idx=1,
        is_synthetic=False,
        seq="",
        length_bp_no_n=300,
        pieces=(p0,),
        support_haps=set(),
        token_str="t0",
        min_hap="H1#1",
        exemplar=dummy_ex,
        member_occurrences=[],
        nested_choices_by_cand_id={},
        reconstructed_len_by_cand_id={},
        piece_cand_ids=(frozenset({"c0"}),),
        support_hap_count=1,
    )
    sol0 = ParallelGroupSolution(
        left_anchor_rank=0,
        right_anchor_rank=1,
        has_nonempty_backbone=False,
        backbone_seq="",
        non_backbone_reps=(rep0,),
        cand_to_rep_idx={"c0": 1},
        cand_to_nested_sig={"c0": ()},
        cand_to_reconstructed_len={"c0": 300},
        cand_to_flags={"c0": (False, False, False)},
    )
    _deduplicate_representatives_across_chromosome(
        graph=SimpleNamespace(ref_step_ranges=((0, 100), (100, 200))),  # type: ignore[arg-type]
        config=cfg,
        sorted_group_pairs=[(0, 1)],
        group_guard_by_pair={(0, 1): GroupRepeatGuardResult()},
        group_solutions={(0, 1): sol0},
        cen_out=CentromereOutput((), (), (), (), ()),
    )
    assert sol0.non_backbone_reps[0].pieces[0].join_before == "synthetic"
    assert sol0.non_backbone_reps[0].partial_content is False
    assert sol0.cand_to_flags["c0"][0] is False


# ============================================================================
# 8. Iteration 3 Proof-Grade Unified Remediation Tests (RG-1..3, LN-1..5)
# ============================================================================


def test_iter3_repeat_guard_offset_sweep_and_variant1b() -> None:
    """Iter 3 RG-1..3: full 50-modulo offset sweep (450..499 x {51,55,105} x +/- = 300 cases)
    asserting exact parity `u_with_short == u_no_short`, plus Residual Variant 1B
    (`42M+2X+6I = 84% < 90%`) asserting Block 9 is NOT falsely masked.
    """
    rng = random.Random(42)
    ref = "".join(rng.choices("ACGT", k=25000))
    aln = mappy.Aligner(seq=ref, preset="asm20", best_n=50)
    saln = mappy.Aligner(
        seq=ref, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )

    mismatches = 0
    for fl_l in range(450, 500):
        for ins_len in (51, 55, 105):
            nov = "".join(rng.choices("ACGT", k=ins_len))
            q_fwd = (
                ref[5000 : 5000 + fl_l]
                + nov
                + ref[5000 + fl_l : 5000 + fl_l + 500]
            )
            for qs in (q_fwd, revcomp_dna(q_fwd)):
                _, u_no_short, _ = subtract_reference_intervals_mappy(
                    qs,
                    min_sv_len=ins_len,
                    aligner=aln,
                    short_aligner=None,
                    min_identity=0.90,
                )
                _, u_with_short, _ = subtract_reference_intervals_mappy(
                    qs,
                    min_sv_len=ins_len,
                    aligner=aln,
                    short_aligner=saln,
                    min_identity=0.90,
                )
                if u_no_short != u_with_short:
                    mismatches += 1
                assert max((e - s for s, e in u_with_short), default=0) >= ins_len
    assert mismatches == 0, f"Offset sweep failed: {mismatches}/300"

    # Residual Variant 1B: 42M + 2X + 6I = 84% < 90% in Block 9 [450, 500)
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    rng_1b = random.Random(123)
    ref_1b = "".join(rng_1b.choices("ACGT", k=10000))
    aln_1b = mappy.Aligner(seq=ref_1b, preset="asm20", best_n=50)
    saln_1b = mappy.Aligner(
        seq=ref_1b, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    fl_l_1b = list(ref_1b[4000:4494])  # 44 bp in Block 9 -> 42M+2X+6I
    fl_l_1b[468] = swap[fl_l_1b[468]]
    fl_l_1b[472] = swap[fl_l_1b[472]]
    q_1b = "".join(fl_l_1b) + "".join(rng_1b.choices("ACGT", k=55)) + ref_1b[4494:4994]
    m0, _, _ = subtract_reference_intervals_mappy(
        q_1b, min_sv_len=50, aligner=aln_1b, short_aligner=None, min_identity=0.90
    )
    m1, _, _ = subtract_reference_intervals_mappy(
        q_1b, min_sv_len=50, aligner=aln_1b, short_aligner=saln_1b, min_identity=0.90
    )
    assert m0 == [(0, 450), (550, 1049)]
    assert m1 == m0


def test_iter3_repeat_guard_3rd_order_l4_l1_and_positive_invariants() -> None:
    """Iter 3 RG-1..3: 3rd-order loopholes L4-1A/1B/short-I, L4 multi-piece `t_starts`/`t_ends`
    on piece k>=1 (+/-), L1 mid-block clip + synthetic Subcase 3b, SimpleNamespace without `.ctg`
    & shattered-I tie-break, plus positive §3.8.4-B invariants (ectopic TE, tandem dup, full
    second copy, dual insertions with 30 bp mid-flank, and adjacent 500+50 microhomology).
    """
    rng = random.Random(42)
    ref = "".join(rng.choices("ACGT", k=25000))
    aln = mappy.Aligner(seq=ref, preset="asm20", best_n=50)
    saln = mappy.Aligner(
        seq=ref, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}

    # (1) L4-1A: <=500 bp (245 + 55 + 76 = 376 bp) on both '+' and mirrored '-' strands
    for trial in range(25):
        rp = 2000 + trial * 150
        nov55 = "".join(rng.choices("ACGT", k=55))
        qf = ref[rp : rp + 245] + nov55 + ref[rp + 245 : rp + 245 + 76]
        _, uf, _ = subtract_reference_intervals_mappy(
            qf, min_sv_len=55, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert max((e - s for s, e in uf), default=0) >= 55, f"L4-1A fwd trial={trial}: {uf}"
        qr = revcomp_dna(ref[rp : rp + 76] + nov55 + ref[rp + 76 : rp + 76 + 245])
        _, ur, _ = subtract_reference_intervals_mappy(
            qr, min_sv_len=55, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert max((e - s for s, e in ur), default=0) >= 55, f"L4-1A rc trial={trial}: {ur}"

    # (2) L4-1B: <=500 bp with 43M + 2X + 5I = 86% < 90% in Block 4 [200, 250)
    for trial in range(15):
        rp = 2000 + trial * 150
        fl = list(ref[rp : rp + 245])
        fl[218] = swap[fl[218]]
        fl[222] = swap[fl[222]]
        q1b = "".join(fl) + "".join(rng.choices("ACGT", k=55)) + ref[rp + 245 : rp + 245 + 76]
        m1b, _, _ = subtract_reference_intervals_mappy(
            q1b, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert not any(s <= 200 and e >= 250 for s, e in m1b), f"L4-1B trial={trial}: {m1b}"

    # (3) L4-short-I: 30 bp I (< min_sv_len=50) with 43M + 2X + 5I = 86% < 90%
    fl_si = list(ref[6000:6245])
    fl_si[218] = swap[fl_si[218]]
    fl_si[222] = swap[fl_si[222]]
    q_si = "".join(fl_si) + "".join(rng.choices("ACGT", k=30)) + ref[6245:6295]
    m_si, _, _ = subtract_reference_intervals_mappy(
        q_si, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    assert not any(s <= 200 and e >= 250 for s, e in m_si), f"L4-short-I failed: {m_si}"

    # (4) L4 multi-piece t_starts/t_ends on piece k >= 1 (+/- strands; exercises piece_key=pos_t)
    sep = "N" * 200
    p0 = ref[1000:1800]
    p1 = ref[3000:3800]
    p2 = ref[5000:5800]
    cat_ref = p0 + sep + p1 + sep + p2
    t_starts = [0, len(p0) + 200, len(p0) + 200 + len(p1) + 200]
    t_ends = [len(p0), t_starts[1] + len(p1), t_starts[2] + len(p2)]
    aln_cat = mappy.Aligner(seq=cat_ref, preset="asm20", best_n=50)
    saln_cat = mappy.Aligner(
        seq=cat_ref, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    for trial in range(10):
        nov55 = "".join(rng.choices("ACGT", k=55))
        q_mp = p1[100:345] + nov55 + p1[345:421] + p2[0:60]  # 436 bp <= 500 bp on k=1 & k=2
        for qs in (q_mp, revcomp_dna(q_mp)):
            _, u_mp, _ = subtract_reference_intervals_mappy(
                qs,
                min_sv_len=55,
                aligner=aln_cat,
                short_aligner=saln_cat,
                min_identity=0.90,
                t_starts=t_starts,
                t_ends=t_ends,
            )
            assert max((e - s for s, e in u_mp), default=0) >= 55, f"Multi-piece L4 failed: {u_mp}"

    # (5) L1 mid-block soft-clip + protected_ins in same block (255 + 50 + 39 + 106 = 450 bp)
    nov_l1_ins = "".join(rng.choices("ACGT", k=50))
    nov_l1_suf = "".join(rng.choices("ACGT", k=106))
    q_l1 = ref[8000:8255] + nov_l1_ins + ref[8255:8294] + nov_l1_suf
    m_l1, u_l1, _ = subtract_reference_intervals_mappy(
        q_l1, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    # Entire [255, 305) protected_ins must survive unmasked (zero backward flank leakage)
    assert all(not (ms < 305 and me > 255) for ms, me in m_l1), f"L1 clobbered ins: m={m_l1}, u={u_l1}"
    assert any(us <= 255 and ue >= 305 for us, ue in u_l1), f"L1 lost ins: m={m_l1}, u={u_l1}"

    # (6) Synthetic Subcase 3b (ins_s % 50 == 0, 44M left arm [256, 300) touching ins [300, 350))
    hit_3b = SimpleNamespace(
        q_st=256,
        q_en=400,
        r_st=100,
        r_en=194,
        strand=1,
        cs=":44+" + "a" * 50 + ":50",
    )
    hm_3b, ha_3b = _evaluate_mappy_hits_on_blocks(
        [hit_3b], query_offset=0, q_len=400, num_blocks=8, min_sv_len=50, min_identity=0.90
    )
    assert (256, 300) in ha_3b
    assert all(not (s < 350 and e > 300) for s, e in ha_3b)
    assert hm_3b == [(350, 400)]

    # (7) Synthetic SimpleNamespace without `.ctg` + shattered-I tie-break
    m_arm = SimpleNamespace(q_st=0, q_en=245, r_st=1000, r_en=1245, strand=1, cs=":245")
    s_bridge = SimpleNamespace(
        q_st=0, q_en=376, r_st=1000, r_en=1321, strand=1, cs=":245+" + "a" * 55 + ":76"
    )
    kept_l4 = _filter_cross_aligner_shadow_hits([m_arm], [s_bridge], min_sv_len=55)
    assert kept_l4 == [s_bridge]

    m_canon = SimpleNamespace(
        q_st=0, q_en=300, r_st=1000, r_en=1250, strand=1, cs=":100+" + "a" * 50 + "*ac:149"
    )
    s_shat = SimpleNamespace(
        q_st=0, q_en=301, r_st=1000, r_en=1251, strand=1, cs=":100+a:1+" + "a" * 49 + ":150"
    )
    kept_shat = _filter_cross_aligner_shadow_hits([m_canon], [s_shat], min_sv_len=50)
    assert kept_shat == [m_canon]

    # (8) Positive §3.8.4-B invariants:
    # (8a) Ectopic 100 bp TE embedded inside 150 bp protected_ins (>500 bp AND <=500 bp)
    q_ect_big = (
        ref[5000:5494]
        + "".join(rng.choices("ACGT", k=6))
        + ref[15000:15100]
        + "".join(rng.choices("ACGT", k=44))
        + ref[5494:6000]
    )
    m_eb, u_eb, _ = subtract_reference_intervals_mappy(
        q_ect_big, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    assert (500, 600) in m_eb and u_eb

    q_ect_sm = (
        ref[5000:5150]
        + "".join(rng.choices("ACGT", k=50))
        + ref[15000:15100]
        + "".join(rng.choices("ACGT", k=50))
        + ref[5150:5250]
    )  # 450 bp <= 500 bp
    m_es, u_es, _ = subtract_reference_intervals_mappy(
        q_ect_sm, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    assert any(s <= 200 and e >= 300 for s, e in m_es)
    assert any(e - s >= 50 for s, e in u_es)

    # (8b) Tandem 100 bp dup inside protected_ins (<=500 bp)
    q_td_exact = ref[2000:2200] + ref[2050:2150] + ref[2200:2350]
    m_td, u_td, _ = subtract_reference_intervals_mappy(
        q_td_exact, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    assert m_td == [(0, 450)] and u_td == []

    # (8c) Full second copy containing ins_55 masks 100% (§3.8.4-B L282)
    nov55_full = "".join(rng.choices("ACGT", k=55))
    q_full_copy = ref[1000:1494] + nov55_full + ref[1494:2000]
    ref_with_copy = ref + "N" * 200 + q_full_copy
    aln_fc = mappy.Aligner(seq=ref_with_copy, preset="asm20", best_n=50)
    saln_fc = mappy.Aligner(
        seq=ref_with_copy, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    m_fc, u_fc, _ = subtract_reference_intervals_mappy(
        q_full_copy, min_sv_len=50, aligner=aln_fc, short_aligner=saln_fc, min_identity=0.90
    )
    assert m_fc == [(0, len(q_full_copy))] and u_fc == []

    # (8d) Dual insertions separated by 30 bp mid-flank + ectopic TE inside ins_2
    ins1_dual = "".join(rng.choices("ACGT", k=60))
    q_dual = (
        ref[7000:7494]
        + ins1_dual
        + ref[7494:7524]  # 30 bp mid-flank -> ends at 584
        + "".join(rng.choices("ACGT", k=16))
        + ref[18000:18100]  # ectopic TE inside ins_2 at [600, 700)
        + "".join(rng.choices("ACGT", k=30))
        + ref[7524:8000]
    )
    m_du, u_du, _ = subtract_reference_intervals_mappy(
        q_dual, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
    )
    # ins1_dual at [494, 554) stays unmasked; ectopic TE at [600, 700) is masked
    assert all(not (ms < 554 and me > 494) for ms, me in m_du)
    assert any(s <= 600 and e >= 700 for s, e in m_du)

    # (8e) Adjacent 500 bp ref_A + 50 bp ref_B (including microhomology overhang)
    for k in range(10):
        q_adj = ref[1000 + k * 200 : 1500 + k * 200] + ref[12000 + k * 200 : 12050 + k * 200]
        m_adj, u_adj, _ = subtract_reference_intervals_mappy(
            q_adj, min_sv_len=50, aligner=aln, short_aligner=saln, min_identity=0.90
        )
        assert m_adj == [(0, 550)] and u_adj == [], f"Adjacent 500+50 failed k={k}: {m_adj}"


def test_iter3_bsec_ownership_pure_deletions_and_childless_components(
    tmp_path: Path,
) -> None:
    """Iter 3 LN-1..5: Probes A1 (only 14 + trimmed -300 no L2038 collision), A2,
    F1 (untouched & trimmed -> exact 3500), A4 (30 bp L848 + tandem-folded L1859 + 3030
    non-interference), A5 (childless prefix del), 2-hap pure del, suffix del, depth-2
    childless pure del, and Case (h) multi-dev (2 dels, del+ins, 2x20 bp sub-SV micro-dels vs 2x30 bp cross-tdev >= min_sv_len).
    """
    s11 = make_complex_seq(1500, seed=8101)
    s14 = make_complex_seq(1000, seed=8102)
    s15 = make_complex_seq(500, seed=8103)
    s12 = make_complex_seq(1500, seed=8104)
    s13 = make_complex_seq(600, seed=8105)
    s16 = make_complex_seq(700, seed=8106)
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_case(
        tag: str,
        nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str = "pan",
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=8110 + i))
        for nid, sq in nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, psteps in paths:
            gb.add_p_path(pname, psteps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=dedup_mode, dedup_id=0.90
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _sup_paths(n: int = 4) -> list[tuple[str, list[tuple[str, str]]]]:
        return [
            (
                f"Sup{k}#1#Chr01",
                [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
            )
            for k in range(1, n + 1)
        ]

    # (1) Probe A1: Hap1=11+14+12, Hap2=11+12 (pure del), Hap3=11+13+12
    a1_paths = _sup_paths() + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_a1_only14 = _run_case(
        "i3_a1_only14",
        {"10": s11 + s13 + s12, "11": s11, "14": s14, "12": s12, "13": s13},
        a1_paths,
    )
    ev = next(e for e in res_a1_only14.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_a1_only14.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.alleles_str == "1:len=1000,src=Hap1#1,n=1,partial_content=1"
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap3#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1"
    h2_logs = [e for e in res_a1_only14.json_log_entries if e.get("hap") == "Hap2#1"]
    assert h2_logs and all(e["final_reason"] == "PAN_HOMOLOG_SUBTRACTED" for e in h2_logs)

    # Probe A1 trimmed -300 bp (no L2038 collision between Hap1 and Hap2)
    res_a1_trim = _run_case(
        "i3_a1_trim",
        {"10": s14[:300], "11": s11, "14": s14, "12": s12, "13": s13},
        a1_paths,
    )
    ev = next(e for e in res_a1_trim.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_map = {
        tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")
    }
    assert "len=3700" in al_map["Hap1#1"] and "partial_content=1" in al_map["Hap1#1"]
    assert "len=3000" in al_map["Hap2#1"] and "partial_content=1" not in al_map["Hap2#1"]
    assert "len=3600" in al_map["Hap3#1"] and "partial_content=1" not in al_map["Hap3#1"]
    assert len({ev.genotypes["Hap1#1"], ev.genotypes["Hap2#1"], ev.genotypes["Hap3#1"]}) == 3

    # (2) Probe A2: Hap1=11+14+15+12, Hap2=11+15+12, Hap3=11+14+13+12; only 14 survives
    a2_paths = _sup_paths() + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_a2 = _run_case(
        "i3_a2",
        {"10": s11 + s15 + s13 + s12, "11": s11, "14": s14, "15": s15, "12": s12, "13": s13},
        a2_paths,
    )
    ev = next(e for e in res_a2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_a2.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap2#1"] == "."
    assert ev.genotypes["Hap1#1"] != "." and ev.genotypes["Hap3#1"] != "."
    assert "Hap2#1" not in ev.alleles_str
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1,Hap3#1"

    # (3) Probe F1: untouched (ref) AND trimmed -300 bp (pan) -> Hap2 exact 3500 clean
    res_f1_ref = _run_case(
        "i3_f1_ref",
        {"10": s14[:300], "11": s11, "14": s14, "15": s15, "12": s12, "13": s13},
        a2_paths,
        dedup_mode="ref",
    )
    ev = next(e for e in res_f1_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_f1_ref = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=3500" in al_f1_ref["Hap2#1"] and "partial_content=1" not in al_f1_ref["Hap2#1"]

    res_f1_pan = _run_case(
        "i3_f1_pan",
        {"10": s14[:300], "11": s11, "14": s14, "15": s15, "12": s12, "13": s13},
        a2_paths,
        dedup_mode="pan",
    )
    ev = next(e for e in res_f1_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_f1_pan = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=3500" in al_f1_pan["Hap2#1"] and "partial_content=1" not in al_f1_pan["Hap2#1"]
    assert "len=4200" in al_f1_pan["Hap1#1"] and "partial_content=1" in al_f1_pan["Hap1#1"]
    assert "len=4300" in al_f1_pan["Hap3#1"] and "partial_content=1" in al_f1_pan["Hap3#1"]

    # (4) Probe A4: 30 bp short replacement (L848) + tandem-folded 4x15->30 bp (L1859) + 3030 non-interference
    short30 = make_complex_seq(30, seed=8120)
    unit15 = make_complex_seq(15, seed=8121)
    a4_paths = _sup_paths() + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("17", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("18", "+"), ("18", "+"), ("18", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_a4_only14 = _run_case(
        "i3_a4_only14",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "17": short30, "18": unit15},
        a4_paths,
    )
    ev = next(e for e in res_a4_only14.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_a4_only14.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap3#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1"

    res_a4_trim = _run_case(
        "i3_a4_trim",
        {"10": s14[:300], "11": s11, "14": s14, "12": s12, "17": short30, "18": unit15},
        a4_paths,
    )
    ev = next(e for e in res_a4_trim.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_a4 = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=3000" in al_a4["Hap2#1"] and "partial_content=1" not in al_a4["Hap2#1"]

    # (5) Probe A5: childless prefix pure deletion (b_sec=Hap3=14+11+13+12, Hap1=14+11+12, Hap2=11+12)
    a5_paths = _sup_paths() + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("14", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("14", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_a5 = _run_case(
        "i3_a5",
        {"10": s11 + s13 + s12, "11": s11, "14": s14, "12": s12, "13": s13},
        a5_paths,
    )
    ev = next(e for e in res_a5.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_a5.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap2#1"] == "."
    assert ev.genotypes["Hap1#1"] != "." and ev.genotypes["Hap3#1"] != "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1,Hap3#1"

    # (6) 2-hap pure deletion (Loophole #1B) + Suffix pure deletion (l, "END")
    res_2hap = _run_case(
        "i3_2hap_del",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12},
        _sup_paths() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_2hap.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "."
    assert [r["members"] for r in res_2hap.inserts_rows if r["event_id"] == ev.event_id] == ["Hap1#1"]

    res_suf = _run_case(
        "i3_suf_del",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12},
        _sup_paths() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("14", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_suf.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "."
    assert [r["members"] for r in res_suf.inserts_rows if r["event_id"] == ev.event_id] == ["Hap1#1"]

    # (7) Depth-2 childless pure deletion (Loophole #1C)
    s_a = make_complex_seq(1000, seed=8130)
    s_d = make_complex_seq(2500, seed=8131)
    s_b = make_complex_seq(1000, seed=8132)
    s_x1 = make_complex_seq(900, seed=8133)
    s_d2 = make_complex_seq(1000, seed=8134)
    s_x2 = make_complex_seq(600, seed=8135)
    res_d2 = _run_case(
        "i3_depth2_childless",
        {"10": s_a + s_d + s_b + s_x1 + s_x2, "11": s_a, "12": s_d, "13": s_b, "14": s_x1, "15": s_d2, "16": s_x2},
        _sup_paths(6) + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("13", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("13", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("13", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("16", "+"), ("13", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("16", "+"), ("13", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_d2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_d2.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == "." and ev.genotypes["Hap3#1"] == "." and ev.genotypes["Hap2#1"] == "1"
    assert len(ins) == 1 and ins[0]["members"] == "Hap2#1"

    # (8) Case (h): multi-deviation on same m_child (2 dels, del+ins, 2x20bp sub-SV vs 2x30bp cross-tdev)
    res_2del = _run_case(
        "i3_2del",
        {"10": s11 + s15 + s12, "11": s11, "14": s14, "15": s15, "16": s16, "12": s12},
        _sup_paths() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("16", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_2del.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "."

    res_delins = _run_case(
        "i3_delins",
        {"10": s11 + s15 + s13 + s12, "11": s11, "14": s14, "15": s15, "16": s16, "13": s13, "12": s12},
        _sup_paths() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("16", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_delins.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "."

    # Sub-SV cross-tdev micro-dels (2x20 = 40 bp < min_sv_len=50) stay unsplit
    m1_20 = make_complex_seq(20, seed=8140)
    m2_20 = make_complex_seq(20, seed=8141)
    res_micro_40 = _run_case(
        "i3_micro_dels_40",
        {"10": s13, "11": s11, "17": m1_20, "15": s15, "18": m2_20, "12": s12},
        [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("17", "+"), ("15", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="ref",
    )
    ev_40 = next(e for e in res_micro_40.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_40 = [r for r in res_micro_40.inserts_rows if r["event_id"] == ev_40.event_id]
    assert len(ins_40) == 1 and ins_40[0]["pan_end"] - ins_40[0]["pan_start"] == 3540

    # Cross-tdev aggregate >= min_sv_len (2x30 = 60 bp >= 50) cleanly excludes 17/18 in ref & pan
    m1_30 = make_complex_seq(30, seed=8140)
    m2_30 = make_complex_seq(30, seed=8141)
    res_micro_60_ref = _run_case(
        "i3_micro_dels_60_ref",
        {"10": s13, "11": s11, "17": m1_30, "15": s15, "18": m2_30, "12": s12},
        [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("17", "+"), ("15", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="ref",
    )
    ev_60r = next(e for e in res_micro_60_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_60r = [r for r in res_micro_60_ref.inserts_rows if r["event_id"] == ev_60r.event_id]
    assert ev_60r.genotypes["Hap1#1"] != ev_60r.genotypes["Hap2#1"]
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_60r if "Hap1#1" in r["members"].split(",")) == 3560
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_60r if "Hap2#1" in r["members"].split(",")) == 3500
    assert all(r["join_before"] != "spacer" for r in ins_60r)

    res_micro_60_pan = _run_case(
        "i3_micro_dels_60_pan",
        {"10": s11 + s15 + s12, "11": s11, "17": m1_30, "15": s15, "18": m2_30, "12": s12},
        _sup_paths() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("17", "+"), ("15", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="pan",
    )
    ev_60p = next(e for e in res_micro_60_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_60p.genotypes["Hap1#1"] == "1" and ev_60p.genotypes["Hap2#1"] == "."
    ins_60p = [r for r in res_micro_60_pan.inserts_rows if r["event_id"] == ev_60p.event_id]
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_60p)


def test_iter3_bsec_edge_cases_1bp_orphan_inv_mc_partial_rep0_uf(
    tmp_path: Path,
) -> None:
    """Iter 3 LN-1..5 edge cases: Loophole #2A/2B (1 bp SNP replaced by 600 bp ins — zero
    1 bp orphan), Case (d) active & inactive all-rep0, Loophole #3A (inverted sec_sc_nonref),
    Loophole #3B/3C (sec_mc_nonref copy-number del & swapped X/Y), Probe 4 (two-run tdev
    dropped sec_mc_nonref run + kept novel 13), Loophole #3D (direct partial step), and
    Case (g) transitive UF 0-anchor member.
    """
    s11 = make_complex_seq(1500, seed=8201)
    s14 = make_complex_seq(1000, seed=8202)
    s15 = make_complex_seq(1000, seed=8203)
    s12 = make_complex_seq(1500, seed=8204)
    s13 = make_complex_seq(600, seed=8205)
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_case(
        tag: str,
        nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str = "pan",
        max_copy: int = 1,
        sec_backbone: str = "support",
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=8210 + i))
        for nid, sq in nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, psteps in paths:
            gb.add_p_path(pname, psteps)
        graph, cfg = _build_graph_and_config(
            tmp_path,
            tag,
            gb,
            ref6,
            dedup_mode=dedup_mode,
            dedup_id=0.90,
            max_copy=max_copy,
            sec_backbone=sec_backbone,
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _sup(n: int = 4) -> list[tuple[str, list[tuple[str, str]]]]:
        return [
            (
                f"Sup{k}#1#Chr01",
                [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
            )
            for k in range(1, n + 1)
        ]

    # (1) Loophole #2A & #2B: 1 bp SNP node in b_sec replaced by 600 bp ins -> zero 1 bp orphan
    l2_paths = _sup() + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_2a = _run_case(
        "i3_l2a",
        {"10": s11 + "A" + s15 + s12 + s13, "11": s11, "14": "A", "15": s15, "12": s12, "13": s13},
        l2_paths,
    )
    assert not [e for e in res_2a.sv_events if e.event_type == "INS" and e.anchors == "3..4"]
    assert res_2a.summary_metrics["dedup_dropped_groups"] == 1

    res_2b = _run_case(
        "i3_l2b",
        {"10": s11 + "A" + s15 + s12, "11": s11, "14": "A", "15": s15, "12": s12, "13": s13},
        l2_paths,
    )
    ev = next(e for e in res_2b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_2b.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.alleles_str == "1:len=600,src=Hap2#1,n=1,partial_content=1"
    assert ev.genotypes["Hap1#1"] == "." and ev.genotypes["Hap2#1"] == "1"
    assert len(ins) == 1 and ins[0]["pan_end"] - ins[0]["pan_start"] == 600

    # (2) Case (d): active AND inactive all-rep0 child groups (14' 97.5% sim to 14 -> retained!)
    s14_sim = mutate_seq_snps(s14, list(range(20, 1000, 40)))  # 25 SNPs / 1000 bp = 97.5%
    for tag_d, include_h3 in (("i3_cased_inactive", False), ("i3_cased_active", True)):
        paths_d = _sup() + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("16", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        if include_h3:
            paths_d.append(
                ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")])
            )
        res_d = _run_case(
            tag_d,
            {"10": s11 + s13 + s12, "11": s11, "14": s14, "16": s14_sim, "13": s13, "12": s12},
            paths_d,
        )
        ev = next(e for e in res_d.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins = [r for r in res_d.inserts_rows if r["event_id"] == ev.event_id]
        assert ev.genotypes["Hap2#1"] != "."
        assert "Hap2#1" in ins[0]["members"]

    # (3) Loophole #3A: inverted sec_sc_nonref (14-) + adjacent insertion 13
    res_3a = _run_case(
        "i3_l3a_inv",
        {"10": s11 + s15 + s13 + s12, "11": s11, "14": s14, "15": s15, "13": s13, "12": s12},
        _sup() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_3a.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_3a.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1#1"] != "." and ev.genotypes["Hap2#1"] != "."
    assert "Hap2#1" in ins[0]["members"]

    # (4) Loophole #3B (multi-copy copy-number del, max_copy=2, sec_backbone=longest) & #3C (swapped [X, Y] / [Y, X])
    res_3b = _run_case(
        "i3_l3b_mc",
        {"10": s11 + s13 + s12, "11": s11, "14": s14, "13": s13, "12": s12},
        [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="ref",
        max_copy=2,
        sec_backbone="longest",
    )
    ev = next(e for e in res_3b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_3b = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=5000" in al_3b["Hap1a#1"]
    assert "len=4000" in al_3b["Hap2#1"]
    assert "len=4600" in al_3b["Hap3#1"]

    s_x = make_complex_seq(800, seed=8220)
    s_y = make_complex_seq(800, seed=8221)
    res_3c = _run_case(
        "i3_l3c_swap",
        {"10": s11 + s_y + s12, "11": s11, "14": s_x, "15": s_y, "12": s12},
        _sup() + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_3c.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_3c.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1#1"] != "." and ev.genotypes["Hap2#1"] != "."
    assert "Hap2#1" in ins[0]["members"]

    # (5) Probe 4 (Gotcha #3): two-run tdev dropped sec_mc_nonref run + kept novel 13
    s_mc = make_complex_seq(700, seed=8230)
    s_sc = make_complex_seq(600, seed=8231)
    res_p4 = _run_case(
        "i3_probe4_2run",
        {"10": s11 + s_sc + s15 + s13 + s12, "11": s11, "14": s_mc, "16": s_sc, "15": s15, "13": s13, "12": s12},
        _sup() + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("16", "+"), ("15", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("16", "+"), ("15", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("16", "-"), ("13", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_p4.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_p4.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap2#1"] != "."
    assert len(ins) == 1 and "Hap2#1" in ins[0]["members"] and "Hap1a#1" in ins[0]["members"]

    # (6) Loophole #3D: direct _solve_multi_member_component partial step test
    nodes_3d = {"11": s11, "14": s14, "12": s12}
    g_3d = SimpleNamespace(nodes=nodes_3d, global_ref_nodes=frozenset())
    cfg_3d = BuildConfig(
        gfa="x", ref_fa="y", backbone_prefix="Ref#1#", out_prefix="z",
        dedup_mode="pan", dedup_id=0.90, min_sv_len=50,
    )

    def _mk_dc(cid: str, hap: str, steps: tuple[Step, ...]) -> object:
        provs: list[StepProvenance] = []
        off = 0
        for st in steps:
            ln = st.step_len(nodes_3d)
            provs.append(StepProvenance(hap, f"{hap}#Chr01", off, off + ln, "+"))
            off += ln
        seq = "".join(
            (nodes_3d[st.node_id] if st.orient == "+" else revcomp_dna(nodes_3d[st.node_id]))[
                st.slice_start : st.slice_end
            ]
            for st in steps
        )
        ec = ExtractedCandidate(
            candidate_id=cid, parent_id=None, hap_id=hap, src_path=f"{hap}#Chr01",
            left_anchor_rank=0, right_anchor_rank=1, source_order=0,
            steps=steps, provenance=tuple(provs), seq=seq,
        )
        pcs = linearizer_mod.steps_to_pieces(
            steps=steps, provenance=tuple(provs), piece_boundaries=(),
            nodes=nodes_3d, members=(hap,),
        )
        return linearizer_mod.DedupCandidate(
            step_key=linearizer_mod.make_step_slice_key(steps),
            steps=steps, seq=seq, pieces=pcs, support_haps=frozenset({hap}),
            occurrences=(ec,), exemplar=ec,
            token_str=linearizer_mod.make_token_str(steps), min_hap=hap,
        )

    # Case 3D-1: partial 14[0:30] vs full 14(1000) -> m_child excluded from 14(1000)
    dc_b1 = _mk_dc("cb1", "Hap1#1", (Step("11", "+"), Step("14", "+"), Step("12", "+")))
    dc_m1 = _mk_dc("cm1", "Hap2#1", (Step("11", "+"), Step("14", "+", 0, 30, True), Step("12", "+")))
    out_3d1 = linearizer_mod._solve_multi_member_component(
        [dc_b1, dc_m1], g_3d, cfg_3d, None, frozenset()  # type: ignore[arg-type]
    )
    assert len(out_3d1) == 1
    mid_pcs = [
        (p, pc) for p, pc in zip(out_3d1[0].pieces, out_3d1[0].piece_cand_ids) if len(p.seq) == 1000
    ]
    assert len(mid_pcs) == 1 and mid_pcs[0][1] == frozenset({"cb1"})

    # Case 3D-2: orientation-normalized matching partial 14+[200:230] vs 14-[770:800] -> retained!
    dc_b2 = _mk_dc("cb2", "Hap1#1", (Step("11", "+"), Step("14", "+", 200, 230, True), Step("12", "+")))
    dc_m2 = _mk_dc("cm2", "Hap2#1", (Step("11", "+"), Step("14", "-", 770, 800, True), Step("12", "+")))
    out_3d2 = linearizer_mod._solve_multi_member_component(
        [dc_b2, dc_m2], g_3d, cfg_3d, None, frozenset()  # type: ignore[arg-type]
    )
    assert len(out_3d2) == 1
    assert all(pc == frozenset({"cb2", "cm2"}) for pc in out_3d2[0].piece_cand_ids)

    # (7) Case (g): transitive UF 0-anchor member (Hap3=99+100 -> '.')
    s99 = make_complex_seq(600, seed=8240)
    s100 = make_complex_seq(600, seed=8241)
    res_uf = _run_case(
        "i3_caseg_uf",
        {"10": s11 + s99 + s100, "11": s11, "14": s14, "99": s99, "100": s100},
        _sup(5) + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("99", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("99", "+"), ("100", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev = next(e for e in res_uf.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_uf.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == "1" and ev.genotypes["Hap1b#1"] == "1"
    assert ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap3#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1a#1,Hap1b#1"

    # Case (g) middle-only overlap: Hap2 carries ONLY middle node 14 (deletes prefix 11 & suffix 12)
    res_mid = _run_case(
        "i3_caseg_mid",
        {"10": s14, "11": s11, "14": s14, "12": s12},
        _sup(5) + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("14", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_mid = next(e for e in res_mid.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_mid = [r for r in res_mid.inserts_rows if r["event_id"] == ev_mid.event_id]
    assert ev_mid.genotypes["Hap1#1"] == "1" and ev_mid.genotypes["Hap2#1"] == "."
    assert all(r["members"] == "Hap1#1" for r in ins_mid)


def test_iter3_cross_module_hetero_merge_typeerror_and_caveat1(tmp_path: Path) -> None:
    """Iter 3 Gotchas #1, #2, #4:
    - Probe 1: Merge Cases M2 (`rc_long=4010` + `rc_hetero`), M3 (`rc_hetero_B` + `rep_hetero_A`),
      and M4 (`rc_single` -> `rep_hetero` harmonization & per-piece `members`) in `ref` & `pan`.
    - Probe 3B: `L2639` exact-seq merge of `("START", 0)` and `(0, 1)` -> zero `TypeError` in Stage 6c.
    - Probe 2B: Caveat #1 Module A split `Hap2` (`occ_idx == 0`) does NOT leak `,partial_content=1`
      onto untouched `Hap3` (`occ_idx == 1`).
    """
    s11 = make_complex_seq(1500, seed=8301)
    s14 = make_complex_seq(1000, seed=8302)
    s12 = make_complex_seq(1500, seed=8303)
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_case(
        tag: str,
        nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str = "pan",
        ref_override: list[tuple[str, str]] | None = None,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        r_steps = ref_override if ref_override is not None else ref6
        for i in range(1, 7):
            if str(i) not in nodes:
                gb.add_node(str(i), make_complex_seq(400, seed=8310 + i))
        for nid, sq in nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", r_steps)
        for pname, psteps in paths:
            gb.add_p_path(pname, psteps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, r_steps, dedup_mode=dedup_mode, dedup_id=0.90
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    # (1) Probe 1 — Merge Case M2 (rc_long=4010 single-member Hap3 + rc_hetero={Hap1=4000, Hap2=3000})
    s_long = s11 + s14 + s12 + make_complex_seq(10, seed=8320)  # 4010 bp, >=99% sim to 11+14+12
    m2_paths = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("20", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_m2_ref = _run_case(
        "i3_m2_ref",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "20": s_long},
        m2_paths,
        dedup_mode="ref",
    )
    ev = next(e for e in res_m2_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_m2 = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=3000" in al_m2["Hap2#1"]

    res_m2_pan = _run_case(
        "i3_m2_pan",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "20": s_long},
        m2_paths,
        dedup_mode="pan",
    )
    ev = next(e for e in res_m2_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_m2_pan.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"] for r in ins)

    # (2) Merge Case M3 (two hetero childless comps with 99% similar b_sec)
    s11_b = mutate_seq_snps(s11, [100, 500, 900])
    s14_b = mutate_seq_snps(s14, [100, 500])
    s12_b = mutate_seq_snps(s12, [100, 500, 900])
    res_m3 = _run_case(
        "i3_m3_pan",
        {
            "10": s11 + s12,
            "11": s11, "14": s14, "12": s12,
            "21": s11_b, "24": s14_b, "22": s12_b,
        },
        [
            (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
            for k in range(1, 5)
        ] + [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("21", "+"), ("24", "+"), ("22", "+"), ("5", "+"), ("6", "+")]),
            ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("21", "+"), ("22", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="pan",
    )
    ev = next(e for e in res_m3.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap4#1"] == "."

    # (3) Merge Case M4 (rc_single -> rep_hetero: L2543 harmonization & L2711 per-piece members)
    s_short_sim = (s11 + s14 + s12)[:3990]  # 3990 bp < 4000 bp, 99.7% sim -> merges into rep_hetero!
    res_m4 = _run_case(
        "i3_m4_pan",
        {"10": s11 + s12, "11": s11, "14": s14, "12": s12, "30": s_short_sim},
        [
            (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
            for k in range(1, 6)
        ] + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("30", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="pan",
    )
    ev = next(e for e in res_m4.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_m4.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == ev.genotypes["Hap1b#1"] == ev.genotypes["Hap3#1"] == "1"
    assert ev.genotypes["Hap2#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1a#1,Hap1b#1,Hap3#1"

    # (4) Probe 3B (R3-5): direct _nested_key_sort_key total-order test + GENUINE
    # matching-partition synth_comp exact-seq merge of Comp 1 ("START", 0) + ("__b_excl__",)
    # and Comp 2 (0, 1) + ("__b_excl__",) sharing the SAME rep_idx (0 spacers, zero TypeError)
    mixed_keys = [
        ("START", "END"),
        ("START", 0),
        (0, 1),
        (1, "END"),
        ("__b_excl__",),
        ((0, 1), ("__b_excl__",)),
        (("START", 0), (0, 1)),
        0,
        "START",
    ]
    sorted_mixed = sorted(mixed_keys, key=_nested_key_sort_key)
    assert len(sorted_mixed) == len(mixed_keys)
    # Comparable int-tuple keys preserve native Python order
    int_tuples = [(0, 2), (0, 1), (1, 2), (0, 3)]
    assert sorted(int_tuples, key=_nested_key_sort_key) == sorted(int_tuples)

    s_c = make_complex_seq(400, seed=8330)
    s_a = make_complex_seq(500, seed=8331)
    s_b = make_complex_seq(500, seed=8332)
    s_d = make_complex_seq(800, seed=8333)
    # Comp 1: b_sec = 40(500)+41(500)+43(800) [1800], Hap2 = 42(400)+40+41 [g_key=("START",0)],
    #         Hap2b = 40+43 [deletes 41 -> splits 40|41] -> pieces (400, 500, 500, 800), 0 spacers
    # Comp 2: b_sec = 50(400)+51(500)+53(800) [1700], Hap4 = 50+52(500)+51 [g_key=(0,1)]
    #         -> pieces (400, 500, 500, 800), 0 spacers, EXACT seq == Comp 1 -> merges into SAME rep_idx!
    nodes_3b = {
        "10": s_c + s_b + s_d,
        "40": s_a, "41": s_b, "42": s_c, "43": s_d,
        "50": s_c, "51": s_b, "52": s_a, "53": s_d,
    }
    paths_3b = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("42", "+"), ("40", "+"), ("41", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("51", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_3b_ref = _run_case("i3_probe3b_ref", nodes_3b, paths_3b, dedup_mode="ref")
    ev3b = next(e for e in res_3b_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins3b = [r for r in res_3b_ref.inserts_rows if r["event_id"] == ev3b.event_id]
    # All 4 pieces share ONE merged rep_idx (zero spacer split between Comp 1 and Comp 2)
    assert len(ins3b) == 4 and all(r["join_before"] != "spacer" for r in ins3b)
    assert [r["pan_end"] - r["pan_start"] for r in ins3b] == [400, 500, 500, 800]
    assert ins3b[0]["members"] == "Hap2#1,Hap3a#1,Hap3b#1,Hap4#1"
    assert ins3b[1]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap2b#1,Hap4#1"
    assert ins3b[2]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap3a#1,Hap3b#1,Hap4#1"
    assert ins3b[3]["members"] == "Hap1a#1,Hap1b#1,Hap2b#1,Hap3a#1,Hap3b#1"
    assert len(ev3b.alleles_str.split(";")) == 5

    sup8_3b = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 9)
    ]
    res_3b_pan = _run_case("i3_probe3b_pan", nodes_3b, sup8_3b + paths_3b, dedup_mode="pan")
    ev3b_p = next(e for e in res_3b_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins3b_p = [r for r in res_3b_pan.inserts_rows if r["event_id"] == ev3b_p.event_id]
    assert len(ins3b_p) == 1 and (ins3b_p[0]["pan_end"] - ins3b_p[0]["pan_start"]) == 500
    assert ins3b_p[0]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap2b#1,Hap4#1"
    assert ev3b_p.genotypes["Hap3a#1"] == "." and ev3b_p.genotypes["Hap3b#1"] == "."

    # (5) Probe 2B (Gotcha #4 / Caveat #1): Module A split Hap2 (occ_idx==0) must NOT leak
    # `,partial_content=1` onto untouched sibling Hap3 (occ_idx==1) sharing identical child steps
    ref_te = make_complex_seq(1200, seed=8340)
    s13 = make_complex_seq(600, seed=8341)
    s19 = make_complex_seq(600, seed=8342)
    ref_p2b = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")]
    res_p2b = _run_case(
        "i3_probe2b",
        {
            "2": ref_te,
            "11": s11, "14": s14, "12": s12, "13": s13, "18": ref_te, "19": s19,
        },
        [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            # Hap2 < Hap3 lexicographically -> occ_idx == 0; Module A splits off node 18 (ref_te),
            # leaving kc1 = [11, 13, 12] (partial_content=True) + kc2 = [19] (partial_content=True)
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("18", "+"), ("19", "+"), ("5", "+"), ("6", "+")]),
            # Hap3 untouched with exact same steps [11, 13, 12] (occ_idx == 1)
            ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        dedup_mode="ref",
        ref_override=ref_p2b,
    )
    ev2b = next(e for e in res_p2b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_2b = {tok.split("src=")[1].split(",")[0]: tok for tok in ev2b.alleles_str.split(";")}
    assert "partial_content=1" in al_2b["Hap2#1"]
    assert "partial_content=1" not in al_2b["Hap3#1"]
    assert "len=3600" in al_2b["Hap3#1"]


def test_iter3_module_abc_end_to_end_protected_ins_split(tmp_path: Path) -> None:
    """Iter 3 End-to-End Module A + B + C:
    - Probe 5: `44M+6I` `nov55` inside `REF_TE` within a multi-member component (`min_sv_len=55`)
      in `ref`, `pan`, and `off` modes (`nov55` survives intact, exact lengths, zero false members).
    - Loophole #4B/#4C: plain `REF_TE` Module A split (`Hap1=11+14+12`, `Hap2=11+REF_TE+12`)
      in `ref` (`Hap2 len=3000`, not `8000`) and `pan` (`Hap2="."`, no phantom `2:len=2000`).
    """
    ref_te = make_complex_seq(1200, seed=8401)
    s11 = make_complex_seq(1500, seed=8402)
    s14 = make_complex_seq(1000, seed=8403)
    s12 = make_complex_seq(1500, seed=8404)
    nov55 = make_complex_seq(55, seed=8405)
    te_with_44m6i = ref_te[100:594] + nov55 + ref_te[594:1094]  # ins at [494, 549) -> 44M+6I
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_probe(
        tag: str,
        mid_hap2: str,
        dedup_mode: str,
        min_sv_len: int = 55,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), ref_te if i == 2 else make_complex_seq(400, seed=8410 + i))
        gb.add_node("10", s11 + s12)
        gb.add_node("11", s11)
        gb.add_node("14", s14)
        gb.add_node("12", s12)
        gb.add_node("18", mid_hap2)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for k in range(1, 5):
            gb.add_p_path(
                f"Sup{k}#1#Chr01",
                [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
            )
        gb.add_p_path(
            "Hap1#1#Chr01",
            [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")],
        )
        gb.add_p_path(
            "Hap2#1#Chr01",
            [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")],
        )
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=dedup_mode, dedup_id=0.90, min_sv_len=min_sv_len
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    # (1) Probe 5 (44M+6I at [494, 549) -> Block 39 44/50 < 90% + Block 40 1/50 < 90% => 100 bp
    # unmasked residual containing nov55 intact, plus non-50-aligned partial r3=1499 bp guarded
    # from merging into rep_hetero): ref, pan, off
    res_p5_ref = _run_probe("i3_p5_ref", te_with_44m6i, "ref", min_sv_len=55)
    ev = next(e for e in res_p5_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=4000" in al["Hap1#1"] and "partial_content=1" not in al["Hap1#1"]
    assert "len=3099" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert nov55 in res_p5_ref.pan_seq

    res_p5_pan = _run_probe("i3_p5_pan", te_with_44m6i, "pan", min_sv_len=55)
    ev = next(e for e in res_p5_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    ins = [r for r in res_p5_pan.inserts_rows if r["event_id"] == ev.event_id]
    assert "len=1000" in al["Hap1#1"] and "partial_content=1" in al["Hap1#1"]
    assert "len=100" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert nov55 in res_p5_pan.pan_seq
    by_len = {r["pan_end"] - r["pan_start"]: r for r in ins}
    assert by_len[1000]["members"] == "Hap1#1"
    assert by_len[100]["members"] == "Hap2#1"

    # Also verify exact 55 bp residual (45M+5I at [495, 550) + 500 bp right TE -> full step 12+)
    swap_b = {"A": "C", "C": "G", "G": "T", "T": "A"}
    nov55_45 = (
        (swap_b[ref_te[595]] if nov55[0] == ref_te[595] else nov55[0])
        + nov55[1:-1]
        + (swap_b[ref_te[594]] if nov55[-1] == ref_te[594] else nov55[-1])
    )
    te_with_45m5i = ref_te[100:595] + nov55_45 + ref_te[595:1095]
    res_p5b_ref = _run_probe("i3_p5b_ref", te_with_45m5i, "ref", min_sv_len=55)
    ev = next(e for e in res_p5b_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=4000" in al["Hap1#1"] and "partial_content=1" not in al["Hap1#1"]
    assert "len=3055" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert nov55_45 in res_p5b_ref.pan_seq

    res_p5b_pan = _run_probe("i3_p5b_pan", te_with_45m5i, "pan", min_sv_len=55)
    ev = next(e for e in res_p5b_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    ins = [r for r in res_p5b_pan.inserts_rows if r["event_id"] == ev.event_id]
    assert "len=1000" in al["Hap1#1"] and "partial_content=1" in al["Hap1#1"]
    assert "len=55" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert nov55_45 in res_p5b_pan.pan_seq
    by_len_b = {r["pan_end"] - r["pan_start"]: r for r in ins}
    assert by_len_b[1000]["members"] == "Hap1#1"
    assert by_len_b[55]["members"] == "Hap2#1"

    res_p5_off = _run_probe("i3_p5_off", te_with_44m6i, "off", min_sv_len=55)
    assert nov55 in res_p5_off.pan_seq

    # (2) Loophole #4B/#4C: plain REF_TE split
    res_4b = _run_probe("i3_l4b_ref", ref_te, "ref", min_sv_len=50)
    ev = next(e for e in res_4b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=4000" in al["Hap1#1"]
    assert "len=3000" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]

    res_4c = _run_probe("i3_l4c_pan", ref_te, "pan", min_sv_len=50)
    ev = next(e for e in res_4c.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res_4c.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.alleles_str == "1:len=1000,src=Hap1#1,n=1,partial_content=1"
    assert ev.genotypes["Hap1#1"] == "1" and ev.genotypes["Hap2#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1#1"


def test_iter4_repeat_guard_multi_unit_subsumption_and_two_phase_step1() -> None:
    """Iter 4 RG-3 regressions (Findings 1 & 2):
    1. Multi-unit `_match_strictly_subsuming`: `mu0 == su0` on piece 0 and `mu1 < su1`
       on piece 1 drops the unbridged `asm20` shadow hit across Module C `'N'*200` waves
       (while identical-span multi-unit ties still let canonical `asm20` win Step 2).
    2. Live `mappy` `'N'*200` wave stress: `L4-1A` `55 bp` novel insertion (50 trials x 3
       configs -> 0/150 destroyed) and `L4-short-I` Variant 1B `43M+2X+5I = 86% < 90%`
       (40 trials x 3 configs -> 0/120 false masked).
    3. Two-phase Step 1a (`drop_short`) -> Step 1b (`drop_main`): a doomed shattered-`I`
       `short_hit` never drops an unrelated `main_hit`, invariant to hit order.
    """
    # (1) Finding 1 synthetic: mu0 == su0 on piece 0, mu1 < su1 on piece 1
    ts, te = [0, 500], [300, 1200]
    hm = SimpleNamespace(
        q_st=0, q_en=305, r_st=240, r_en=745, strand=1, cs=":60-" + "n" * 200 + ":245"
    )
    hs = SimpleNamespace(
        q_st=0,
        q_en=436,
        r_st=240,
        r_en=821,
        strand=1,
        cs=":60-" + "n" * 200 + ":245+" + "a" * 55 + ":76",
    )
    assert _filter_cross_aligner_shadow_hits([hm], [hs], 55, ts, te) == [hs]

    # Exact identical-span multi-unit tie must still let canonical asm20 win Step 2
    hs_tie = SimpleNamespace(
        q_st=0, q_en=305, r_st=240, r_en=745, strand=1, cs=":60-" + "n" * 200 + ":245"
    )
    assert _filter_cross_aligner_shadow_hits([hm], [hs_tie], 55, ts, te) == [hm]

    # (2) Finding 2 synthetic dropped-witness cycle (both [hm0, hm1] and [hm1, hm0])
    hm0 = SimpleNamespace(q_st=0, q_en=100, r_st=1000, r_en=1100, strand=1, cs=":100")
    hm1 = SimpleNamespace(
        q_st=100,
        q_en=399,
        r_st=1100,
        r_en=1349,
        strand=1,
        cs=":100+" + "a" * 50 + "*ac:148",
    )
    hs0 = SimpleNamespace(
        q_st=0,
        q_en=400,
        r_st=1000,
        r_en=1350,
        strand=1,
        cs=":200+a:1+" + "a" * 49 + ":149",
    )
    assert _filter_cross_aligner_shadow_hits([hm0, hm1], [hs0], 50) == [hm0, hm1]
    assert _filter_cross_aligner_shadow_hits([hm1, hm0], [hs0], 50) == [hm1, hm0]

    # (3) Finding 1 live mappy: L4-1A 55 bp novel insertion across 'N'*200 wave (50 x 3)
    rng = random.Random(20261125)
    ref = "".join(rng.choices("ACGT", k=30000))
    for trial in range(50):
        rp0, rp1 = 500 + trial * 250, 15000 + trial * 250
        p0, p1 = ref[rp0 : rp0 + 500], ref[rp1 : rp1 + 600]
        cat_seq = p0 + ("N" * 200) + p1
        a_cat = mappy.Aligner(seq=cat_seq, preset="asm20", best_n=50)
        s_cat = mappy.Aligner(
            seq=cat_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
        )
        nov55 = "".join(rng.choices("ACGT", k=55))
        for fl0, fl1_l, fl1_r in [(227, 168, 50), (230, 165, 50), (220, 175, 50)]:
            q = p0[-fl0:] + p1[:fl1_l] + nov55 + p1[fl1_l : fl1_l + fl1_r]
            _, u_res, _ = subtract_reference_intervals_mappy(
                q,
                min_sv_len=55,
                aligner=a_cat,
                short_aligner=s_cat,
                min_identity=0.90,
                t_starts=[0, 700],
                t_ends=[500, 1300],
            )
            assert max((e - s for s, e in u_res), default=0) >= 55, (
                f"Destroyed 55 bp insertion on trial={trial}, cfg={(fl0, fl1_l, fl1_r)}: {u_res}"
            )

    # (4) Finding 1 live mappy: L4-short-I Variant 1B 43M+2X+5I (86% < 90%) across 'N'*200
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    false_masked = 0
    for trial in range(40):
        rp0, rp1 = 600 + trial * 250, 16000 + trial * 250
        p0, p1 = ref[rp0 : rp0 + 500], ref[rp1 : rp1 + 600]
        cat_seq = p0 + ("N" * 200) + p1
        a_cat = mappy.Aligner(seq=cat_seq, preset="asm20", best_n=50)
        s_cat = mappy.Aligner(
            seq=cat_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
        )
        for fl0 in (220, 225, 230):
            fl1_l = 445 - fl0
            p1_left = list(p1[:fl1_l])
            p1_left[410 - fl0] = swap[p1_left[410 - fl0]]
            p1_left[425 - fl0] = swap[p1_left[425 - fl0]]
            ins5 = "".join(swap[c] for c in p1[fl1_l : fl1_l + 5])
            q = p0[-fl0:] + "".join(p1_left) + ins5 + p1[fl1_l : fl1_l + 50]
            m_res, _, _ = subtract_reference_intervals_mappy(
                q,
                min_sv_len=50,
                aligner=a_cat,
                short_aligner=s_cat,
                min_identity=0.90,
                t_starts=[0, 700],
                t_ends=[500, 1300],
            )
            if any(s <= 400 and e >= 450 for s, e in m_res):
                false_masked += 1
    assert false_masked == 0, f"Variant 1B false masked {false_masked}/120 trials"


def test_iter4_linearizer_r3_1_to_r3_4_adversarial_regressions(tmp_path: Path) -> None:
    """Iter 4 linearizer regressions (`R3-1` .. `R3-4`):
    - R3-1 (`ADV-1` inactive & `ADV-8` active): partial substring `rep0` child merge
      tracks exact `(t_start, t_end)` per `b_steps[s_idx]`, preventing `+1000 bp`
      inflation in `ref` and false-positive `"1"` / `members` in `pan`.
    - R3-2 (`ADV-2`, `ADV-2b`, `ADV-2c`, `ADV-2d`): exact `(("__b_excl__",), excl_steps)`
      separates distinct `missing_steps` subsets across childless and active child
      groups, preventing leaked `,partial_content=1` and haplotype-name-order flips.
    - R3-3 (`ADV-3c`, `ADV-6c_sup8`, `ADV-16`): heterogeneous `synth_comp` exact-seq
      merge requires matching piece partition under `dedup_mode != "off"`, and refreshes
      `Piece.members` on merged `synth_reps`.
    - R3-4 (`ADV-7`): `rep_is_hetero` full-span guard includes `+ m_del`, rejecting
      `m_ts + m_del = 60 >= 50` merges.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_adv(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=9100 + i))
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=dedup_mode, dedup_id=0.90, min_sv_len=50
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    s11 = make_complex_seq(1500, seed=9201)
    s14 = make_complex_seq(1000, seed=9202)
    s15 = make_complex_seq(1000, seed=9203)
    s12 = make_complex_seq(1500, seed=9204)
    s13 = make_complex_seq(600, seed=9205)
    s15p = mutate_seq_snps(s15, [i * 50 + 10 for i in range(18)])  # 98.2% sim to s15

    sup5 = [
        (
            f"Sup{k}#1#Chr01",
            [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
        )
        for k in range(1, 6)
    ]
    sup6 = sup5 + [
        (
            "Sup6#1#Chr01",
            [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
        )
    ]
    sup8 = sup6 + [
        (
            f"Sup{k}#1#Chr01",
            [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")],
        )
        for k in (7, 8)
    ]

    # (1) R3-1: ADV-1 (inactive partial rep0 child) & ADV-8 (active partial rep0 child)
    nodes_adv1 = {"10": s11 + s15 + s12, "11": s11, "14": s14, "15": s15, "15p": s15p, "12": s12}
    paths_adv1 = sup5 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15p", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv1_ref", nodes_adv1, paths_adv1, "ref")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["Hap2#1"]
    assert ev.genotypes["Hap1a#1"] == ev.genotypes["Hap1b#1"] != ev.genotypes["Hap2#1"]

    res = _run_adv("i4_adv1_pan", nodes_adv1, paths_adv1, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == ev.genotypes["Hap1b#1"] == "1"
    assert ev.genotypes["Hap2#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1a#1,Hap1b#1"

    nodes_adv8 = {**nodes_adv1, "10": s11 + s15 + s12 + s13, "13": s13}
    paths_adv8 = paths_adv1 + [
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv8_ref", nodes_adv8, paths_adv8, "ref")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["Hap2#1"] and "len=3600" in al["Hap3#1"]

    res = _run_adv("i4_adv8_pan", nodes_adv8, paths_adv8, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == ev.genotypes["Hap1b#1"] == "1"
    assert ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap3#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap1a#1,Hap1b#1"

    # (2) R3-2: ADV-2 / ADV-2b / ADV-2c (name-order invariance) / ADV-2d (active g_key)
    nodes_adv2 = {"10": s14[:300], "11": s11, "14": s14, "15": s15, "12": s12, "13": s13}
    paths_adv2 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv2_ref", nodes_adv2, paths_adv2, "ref")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["Hap2#1"] and "len=3000" in al["Hap3#1"]

    res = _run_adv("i4_adv2b_pan", nodes_adv2, paths_adv2, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=4700" in al["Hap1a#1"] and "partial_content=1" in al["Hap1a#1"]
    assert "len=3700" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert "len=3000" in al["Hap3#1"] and "partial_content=1" not in al["Hap3#1"]
    assert ev.genotypes["Hap2#1"] != ev.genotypes["Hap3#1"]

    paths_adv2c = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("ZHap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("AHap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv2c_ref", nodes_adv2, paths_adv2c, "ref")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=5000" in al["Hap1a#1"] and "len=4000" in al["ZHap2#1"] and "len=3000" in al["AHap3#1"]

    paths_adv2d = paths_adv2 + [
        ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv2d_pan", nodes_adv2, paths_adv2d, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert "len=4700" in al["Hap1a#1"] and "partial_content=1" in al["Hap1a#1"]
    assert "len=3700" in al["Hap2#1"] and "partial_content=1" in al["Hap2#1"]
    assert "len=3000" in al["Hap3#1"] and "partial_content=1" not in al["Hap3#1"]
    assert "len=3600" in al["Hap4#1"] and "partial_content=1" not in al["Hap4#1"]
    assert len({ev.genotypes[h] for h in ("Hap1a#1", "Hap2#1", "Hap3#1", "Hap4#1")}) == 4

    # (3) R3-3: ADV-3c, ADV-6c_sup8, and ADV-16
    nodes_adv3c = {
        "10": s11 + s12 + s15,
        "11": s11, "13": s13, "12": s12, "15": s15,
        "20": s11 + s13 + s12 + s15,
    }
    paths_adv3c = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("20", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv3c_pan", nodes_adv3c, paths_adv3c, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1a#1"] == "." and ev.genotypes["Hap1b#1"] == "."
    assert ev.genotypes["Hap2#1"] == "1"
    assert all("Hap1a#1" not in r["members"] and "Hap1b#1" not in r["members"] for r in ins)

    s_c = make_complex_seq(400, seed=9301)
    s_a = make_complex_seq(500, seed=9302)
    s_b = make_complex_seq(500, seed=9303)
    s_d = make_complex_seq(800, seed=9304)
    nodes_adv6c = {
        "10": s_c + s_b + s_d,
        "40": s_a, "41": s_b, "42": s_c, "43": s_d,
        "50": s_c, "51": s_b, "52": s_a, "53": s_d,
    }
    paths_adv6c = sup8 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("42", "+"), ("40", "+"), ("41", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("51", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv6c_sup8_pan", nodes_adv6c, paths_adv6c, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap3a#1"] == "." and ev.genotypes["Hap3b#1"] == "."
    assert all("Hap3a#1" not in r["members"] and "Hap3b#1" not in r["members"] for r in ins)

    nodes_adv16 = {
        "10": s11 + s12 + s15,
        "11": s11, "13": s13, "12": s12, "15": s15,
        "21": s11, "23": s13, "22": s12, "25": s15,
    }
    paths_adv16 = sup8 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("13", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("21", "+"), ("22", "+"), ("25", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("21", "+"), ("22", "+"), ("25", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("21", "+"), ("23", "+"), ("22", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv16_ref", nodes_adv16, paths_adv16, "ref")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert len(ins) == 4 and all(r["join_before"] != "spacer" for r in ins)
    assert ins[1]["members"] == "Hap2#1,Hap4#1"

    res = _run_adv("i4_adv16_pan", nodes_adv16, paths_adv16, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap2#1"] == "1" and ev.genotypes["Hap4#1"] == "1"
    assert ev.genotypes["Hap1a#1"] == "." and ev.genotypes["Hap3a#1"] == "."
    assert len(ins) == 1 and ins[0]["members"] == "Hap2#1,Hap4#1"

    # (4) R3-4: ADV-7 (rep_is_hetero m_ts=30 + m_del=30 = 60 >= 50 rejects merge)
    p30 = make_complex_seq(30, seed=9401)
    sc1 = make_complex_seq(1500, seed=9402)
    m30 = make_complex_seq(30, seed=9403)
    sc2 = make_complex_seq(1500, seed=9404)
    ins25 = make_complex_seq(25, seed=9405)
    nodes_adv7 = {
        "10": sc1 + sc2,
        "p30": p30, "sc1": sc1, "m30": m30, "sc2": sc2,
        "20": sc1 + sc2 + ins25,
    }
    paths_adv7 = sup6 + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p30", "+"), ("sc1", "+"), ("m30", "+"), ("sc2", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("sc1", "-"), ("sc2", "-"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("20", "+"), ("5", "+"), ("6", "+")]),
    ]
    res = _run_adv("i4_adv7_pan", nodes_adv7, paths_adv7, "pan")
    ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
    assert ev.genotypes["Hap1#1"] == "1"
    assert ev.genotypes["Hap2#1"] == "." and ev.genotypes["Hap3#1"] == "."
    assert len(ins) == 2 and all(r["members"] == "Hap1#1" for r in ins)


def test_iter5_repeat_guard_atomic_unit_dedup_and_d_strip() -> None:
    """Iter 5 repeat_guard.py proof-grade regressions (`explorer_iter5_1` + Region B):
    1. Non-adjacent 3-piece wave `p0 + N*200 + pm(len_mid) + N*200 + p2` with both
       `off=(0, 0)` and `off=(30, 40)` (`+/-` strands, `len_mid in (150, 500, 750)`):
       0 `nov55` clobbers and 0 Variant 1B `ins=20` false masks.
    2. Offset-bridged 100%-homologous `q_800 = p0[70:470] + p1[40:440]` (`+/-` strands):
       `m == [(0, 800)], u == [], bp == 800`, plus genuine internal `D` preservation.
    3. `query_offset > 0` zdrop pass preserving offset-bridged `nov55` at `[1395, 1450)`.
    4. `_filter_cross_aligner_shadow_units` permutation-invariant invariants (Attack B,
       converse exact-span tie, forward exact-span tie + tandem-dup `mu0`, Witness-Survival
       Re-Check, Step 2b orphaned `su_arm` including `main_units == ()` & disjoint `piece_key`
       Guardrail 1, identical `short_units`, `D`-only equivalence, and block-eval parity).
    5. `_check_similarity_merge_with_span(..., del_intervals_out=...)` on `edlib` and `mappy`.
    """
    rng = random.Random(20261125)
    ref = "".join(rng.choices("ACGT", k=60000))

    # (1) Offset-bridged 100%-homologous q_800 (+/- strands) + internal D preservation
    p0_q800, p1_q800 = ref[500:1000], ref[15000:15600]
    cat2 = p0_q800 + ("N" * 200) + p1_q800
    a2 = mappy.Aligner(seq=cat2, preset="asm20", best_n=50)
    s2 = mappy.Aligner(seq=cat2, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
    q800 = p0_q800[70:470] + p1_q800[40:440]
    for q in (q800, revcomp_dna(q800)):
        m, u, bp = subtract_reference_intervals_mappy(
            q,
            min_sv_len=50,
            aligner=a2,
            short_aligner=s2,
            min_identity=0.90,
            t_starts=[0, 700],
            t_ends=[500, 1300],
        )
        assert m == [(0, 800)] and u == [] and bp == 800

    # Synthetic bridged hit with leading/trailing separator-bleed D AND genuine internal 10D
    ops_int_d = [("M", 30), ("D", 10), ("M", 20), ("D", 270), ("M", 50)]
    part = _partition_bridged_hit_ops(
        ops=ops_int_d,
        q_start=0,
        r_st=410,
        r_en=790,
        strand=1,
        t_starts=[0, 700],
        t_ends=[500, 1300],
    )
    assert set(part.keys()) == {0, 1}
    # Piece 0 keeps internal 10D, strips trailing 30D bleed into [470, 500)
    assert [op for _, _, _, op, _ in part[0]] == ["M", "D", "M"]
    assert part[0][1][4] == 10 and part[0][0][1] == 410 and part[0][-1][2] == 470
    # Piece 1 strips leading 40D bleed from [700, 740)
    assert [op for _, _, _, op, _ in part[1]] == ["M"]
    assert part[1][0][1] == 740 and part[1][0][2] == 790
    # Internal 10D drops block 0 identity to 50/60 = 0.833 < 0.90 (unmasked) while block 1 = 50/50 masked
    h_int_d = SimpleNamespace(
        q_st=0,
        q_en=100,
        r_st=410,
        r_en=790,
        strand=1,
        cs=":30-" + "a" * 10 + ":20-" + "a" * 270 + ":50",
    )
    hm_d, _ = _evaluate_mappy_hits_on_blocks(
        [h_int_d], 0, 100, 2, min_sv_len=50, min_identity=0.90, t_starts=[0, 700], t_ends=[500, 1300]
    )
    assert hm_d == [(50, 100)]

    # (2) Non-adjacent 3-piece wave (off=(0,0) and off=(30,40), +/- strands, len_mid in {150, 500, 750})
    bridged_checked = 0
    for len_mid in (150, 500, 750):
        for off0, off2 in ((0, 0), (30, 40)):
            for trial in range(15):
                rp0 = 500 + trial * 250
                rpm = 18000 + trial * 250
                rp2 = 36000 + trial * 250
                p0 = ref[rp0 : rp0 + 500]
                pm = ref[rpm : rpm + len_mid]
                p2 = ref[rp2 : rp2 + 600]
                cat3 = p0 + ("N" * 200) + pm + ("N" * 200) + p2
                ts = [0, 700, 700 + len_mid + 200]
                te = [500, 700 + len_mid, 700 + len_mid + 800]
                a3 = mappy.Aligner(seq=cat3, preset="asm20", best_n=50)
                s3 = mappy.Aligner(
                    seq=cat3, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
                )
                nov55 = "".join(rng.choices("ACGT", k=55))
                qf = (
                    p0[280 - off0 : 500 - off0]
                    + p2[off2 : off2 + 175]
                    + nov55
                    + p2[off2 + 175 : off2 + 225]
                )
                for q in (qf, revcomp_dna(qf)):
                    sh = list(s3.map(q, cs=True))
                    if any(
                        u.meta.protected_ins
                        for h in sh
                        for u in _extract_hit_piece_eval_units(h, 0, 55, ts, te)
                    ):
                        bridged_checked += 1
                        _, u_res, _ = subtract_reference_intervals_mappy(
                            q,
                            min_sv_len=55,
                            aligner=a3,
                            short_aligner=s3,
                            min_identity=0.90,
                            t_starts=ts,
                            t_ends=te,
                        )
                        assert max((e - s for s, e in u_res), default=0) >= 55
    assert bridged_checked >= 50

    # Variant 1B (ins=20, 43M+2X+5I = 86% < 90%) on 3-piece wave for off=(0,0) and off=(30,40)
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    v1b_bridged = 0
    for off0, off2 in ((0, 0), (30, 40)):
        for trial in range(15):
            rp0 = 600 + trial * 250
            rpm = 19000 + trial * 250
            rp2 = 38000 + trial * 250
            p0, pm, p2 = ref[rp0 : rp0 + 500], ref[rpm : rpm + 150], ref[rp2 : rp2 + 600]
            cat3 = p0 + ("N" * 200) + pm + ("N" * 200) + p2
            ts, te = [0, 700, 1050], [500, 850, 1650]
            a3 = mappy.Aligner(seq=cat3, preset="asm20", best_n=50)
            s3 = mappy.Aligner(
                seq=cat3, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
            )
            for fl0 in (220, 225, 230):
                fl1_l = 445 - fl0
                p2_left = list(p2[off2 : off2 + fl1_l])
                p2_left[410 - fl0] = swap[p2_left[410 - fl0]]
                p2_left[425 - fl0] = swap[p2_left[425 - fl0]]
                ins20 = "".join(rng.choices("ACGT", k=20))
                q = (
                    p0[500 - off0 - fl0 : 500 - off0]
                    + "".join(p2_left)
                    + ins20
                    + p2[off2 + fl1_l : off2 + fl1_l + 50]
                )[:500]
                sh_v = list(s3.map(q, cs=True))
                if not any(
                    u.meta.piece_key == 2 and u.meta.q_en > 450
                    for h in sh_v
                    for u in _extract_hit_piece_eval_units(h, 0, 50, ts, te)
                ):
                    continue
                v1b_bridged += 1
                m_res, _, _ = subtract_reference_intervals_mappy(
                    q,
                    min_sv_len=50,
                    aligner=a3,
                    short_aligner=s3,
                    min_identity=0.90,
                    t_starts=ts,
                    t_ends=te,
                )
                assert not any(s <= 400 and e >= 450 for s, e in m_res)
    assert v1b_bridged >= 40

    # (3) query_offset > 0 zdrop pass preserving offset-bridged nov55 at [1395, 1450)
    zdrop_checked = 0
    p3_flank = ref[50000:52000]
    for trial in range(15):
        rp0 = 500 + trial * 150
        rpm = 20000 + trial * 150
        rp2 = 40000 + trial * 150
        p0, pm, p2 = ref[rp0 : rp0 + 500], ref[rpm : rpm + 150], ref[rp2 : rp2 + 600]
        cat4 = p3_flank + ("N" * 200) + p0 + ("N" * 200) + pm + ("N" * 200) + p2
        ts4 = [0, 2200, 2900, 3250]
        te4 = [2000, 2700, 3050, 3850]
        a4 = mappy.Aligner(seq=cat4, preset="asm20", best_n=50)
        s4 = mappy.Aligner(seq=cat4, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
        nov55 = "".join(rng.choices("ACGT", k=55))
        q_mid = p0[250:470] + p2[40:215] + nov55 + p2[215:265]
        sh_mid = list(s4.map(q_mid, cs=True))
        if not any(
            u.meta.protected_ins
            for h in sh_mid
            for u in _extract_hit_piece_eval_units(h, 1000, 55, ts4, te4)
        ):
            continue
        zdrop_checked += 1
        q_full = p3_flank[:1000] + q_mid + p3_flank[1000:2000]
        _, u_full, _ = subtract_reference_intervals_mappy(
            q_full,
            min_sv_len=55,
            aligner=a4,
            short_aligner=s4,
            min_identity=0.90,
            t_starts=ts4,
            t_ends=te4,
        )
        assert any(s <= 1395 and e >= 1450 for s, e in u_full), (
            f"query_offset=1000 zdrop clobbered [1395,1450): {u_full}"
        )
    assert zdrop_checked >= 3

    # (4) _filter_cross_aligner_shadow_units permutation-invariant invariants
    ts_ab, te_ab = [0, 500], [300, 1200]
    # 4a: Multi-unit shattered-I Attack B (hs0 bridges p0+p1 with shattered I on p1 vs separate [hm0, hm1])
    hm0 = SimpleNamespace(q_st=0, q_en=100, r_st=200, r_en=300, strand=1, cs=":100")
    hm1 = SimpleNamespace(
        q_st=100,
        q_en=399,
        r_st=500,
        r_en=749,
        strand=1,
        cs=":145+" + "a" * 50 + "*ac:103",
    )
    hs0 = SimpleNamespace(
        q_st=0,
        q_en=400,
        r_st=200,
        r_en=750,
        strand=1,
        cs=":100-" + "n" * 200 + ":145+a:1+" + "a" * 49 + ":104",
    )
    u_hm0 = _extract_hit_piece_eval_units(hm0, 0, 50, ts_ab, te_ab)
    u_hm1 = _extract_hit_piece_eval_units(hm1, 0, 50, ts_ab, te_ab)
    u_hs0 = _extract_hit_piece_eval_units(hs0, 0, 50, ts_ab, te_ab)
    for m_order in (u_hm0 + u_hm1, u_hm1 + u_hm0):
        for s_order in (u_hs0, list(reversed(u_hs0))):
            kept = _filter_cross_aligner_shadow_units(m_order, s_order)
            assert len(kept) == 2
            assert {u.meta.piece_key for u in kept} == {0, 1}
            p1_u = next(u for u in kept if u.meta.piece_key == 1)
            assert p1_u.meta.protected_ins == ((245, 295),)
            km, _ = _evaluate_piece_units_on_blocks(kept, 400, 8, min_identity=0.90)
            assert all(e <= 245 or s >= 295 for s, e in km)

    # 4b: Converse exact-span tie (su has protected_ins, mu shattered it -> keep [su])
    h_mu_shat = SimpleNamespace(
        q_st=0, q_en=295, r_st=1000, r_en=1245, strand=1, cs=":95+a:1+" + "a" * 49 + ":149"
    )
    h_su_prot = SimpleNamespace(
        q_st=0, q_en=295, r_st=1000, r_en=1245, strand=1, cs=":95+" + "a" * 50 + "*ac:149"
    )
    u_mu_shat = _extract_hit_piece_eval_units(h_mu_shat, 0, 50)
    u_su_prot = _extract_hit_piece_eval_units(h_su_prot, 0, 50)
    kept_conv = _filter_cross_aligner_shadow_units(u_mu_shat, u_su_prot)
    assert kept_conv == u_su_prot

    # 4c: Forward exact-span tie + tandem-dup mu0 inside mu1.protected_ins -> Step 1a drops su, keeps [mu0, mu1]
    h_m_dup = SimpleNamespace(q_st=100, q_en=150, r_st=1100, r_en=1150, strand=1, cs=":50")
    h_m_prot = SimpleNamespace(
        q_st=0, q_en=300, r_st=1000, r_en=1250, strand=1, cs=":100+" + "a" * 50 + ":150"
    )
    h_s_shat = SimpleNamespace(
        q_st=0, q_en=300, r_st=1000, r_en=1250, strand=1, cs=":100+a:1+" + "a" * 49 + ":149"
    )
    u_m_dup = _extract_hit_piece_eval_units(h_m_dup, 0, 50)
    u_m_prot = _extract_hit_piece_eval_units(h_m_prot, 0, 50)
    u_s_shat = _extract_hit_piece_eval_units(h_s_shat, 0, 50)
    for m_ord in (u_m_dup + u_m_prot, u_m_prot + u_m_dup):
        assert len(_filter_cross_aligner_shadow_units(m_ord, u_s_shat)) == 2
        assert {u.meta.q_en for u in _filter_cross_aligner_shadow_units(m_ord, u_s_shat)} == {
            150,
            300,
        }

    # 4d: Witness-Survival Re-Check (smaller shattered su [0,300) + larger mu2 [0,350) + tandem-dup mu1 [100,150))
    h_m_big = SimpleNamespace(
        q_st=0, q_en=350, r_st=1000, r_en=1300, strand=1, cs=":100+" + "a" * 50 + ":200"
    )
    u_m_big = _extract_hit_piece_eval_units(h_m_big, 0, 50)
    for m_ord in (u_m_dup + u_m_big, u_m_big + u_m_dup):
        kept_ws = _filter_cross_aligner_shadow_units(m_ord, u_s_shat)
        assert len(kept_ws) == 2
        assert {u.meta.q_en for u in kept_ws} == {150, 350}

    # 4e: Step 2b orphaned su_arm alongside su_bridge (with mu_arm, with main_units==(), and Guardrail 1 disjoint piece_key)
    h_arm = SimpleNamespace(q_st=0, q_en=245, r_st=500, r_en=745, strand=1, cs=":245")
    h_bridge = SimpleNamespace(
        q_st=0,
        q_en=376,
        r_st=500,
        r_en=821,
        strand=1,
        cs=":245+" + "a" * 55 + ":76",
    )
    h_su_dup = SimpleNamespace(q_st=245, q_en=300, r_st=700, r_en=755, strand=1, cs=":55")
    u_arm = _extract_hit_piece_eval_units(h_arm, 0, 55, ts_ab, te_ab)
    u_bridge = _extract_hit_piece_eval_units(h_bridge, 0, 55, ts_ab, te_ab)
    u_su_dup = _extract_hit_piece_eval_units(h_su_dup, 0, 55, ts_ab, te_ab)
    assert _filter_cross_aligner_shadow_units(u_arm, u_arm + u_bridge) == u_bridge
    # Guardrail 1: main_units == () still runs Step 2b!
    assert _filter_cross_aligner_shadow_units((), u_arm + u_bridge) == u_bridge
    assert _filter_cross_aligner_shadow_units((), u_bridge + u_arm) == u_bridge
    # Guardrail 1: main_units only on piece 0, short_units [su_arm, su_bridge] on piece 1
    kept_g1 = _filter_cross_aligner_shadow_units(u_hm0, u_arm + u_bridge)
    assert kept_g1 == u_hm0 + u_bridge
    # Tandem-dup short unit inside su_bridge.protected_ins survives Step 2b
    kept_sdup = _filter_cross_aligner_shadow_units((), u_bridge + u_su_dup + u_arm)
    assert len(kept_sdup) == 2 and u_bridge[0] in kept_sdup and u_su_dup[0] in kept_sdup
    # Identical short_units preserved
    assert len(_filter_cross_aligner_shadow_units((), u_bridge + u_bridge)) == 2

    # 4f: D-only equivalence & _evaluate_piece_units_on_blocks == _evaluate_mappy_hits_on_blocks
    h_donly_dir = SimpleNamespace(q_st=0, q_en=0, r_st=0, r_en=40, strand=1, cs="-" + "a" * 40)
    h_donly_br = SimpleNamespace(q_st=0, q_en=0, r_st=250, r_en=550, strand=1, cs="-" + "a" * 300)
    assert _extract_hit_piece_eval_units(h_donly_dir, 0, 50) == []
    assert _extract_hit_piece_eval_units(h_donly_br, 0, 50, ts_ab, te_ab) == []
    assert _evaluate_mappy_hits_on_blocks([h_donly_dir], 0, 100, 2, 50) == ([], [])
    assert _evaluate_mappy_hits_on_blocks([h_donly_br], 0, 100, 2, 50, t_starts=ts_ab, t_ends=te_ab) == ([], [])

    u_off = _extract_hit_piece_eval_units(h_bridge, 100, 55, ts_ab, te_ab)
    assert _evaluate_piece_units_on_blocks(u_off, 500, 10, min_identity=0.90) == (
        _evaluate_mappy_hits_on_blocks(
            [h_bridge], 100, 500, 10, 55, min_identity=0.90, t_starts=ts_ab, t_ends=te_ab
        )
    )

    # (5) _check_similarity_merge_with_span(..., del_intervals_out=...) on edlib & mappy
    r_ed = make_complex_seq(600, seed=5501)
    s_ed1 = r_ed[:200] + r_ed[230:]  # single 30 bp D at [200, 230)
    s_ed2 = r_ed[:100] + r_ed[120:300] + r_ed[325:]  # double D [100,120) & [300,325) = 45 bp
    s_ed_rej = r_ed[:200] + r_ed[260:]  # 60 bp D >= 50 -> rejected
    for sq, exp_d in ((s_ed1, 30), (s_ed2, 45)):
        res_plain = _check_similarity_merge_with_span(sq, r_ed, min_sv_len=50)
        d_out: list[tuple[int, int]] = []
        res_with = _check_similarity_merge_with_span(
            sq, r_ed, min_sv_len=50, del_intervals_out=d_out
        )
        assert res_plain == res_with and len(res_with) == 4
        assert res_with[0] is True and res_with[3] == exp_d
        assert sum(e - s for s, e in d_out) == exp_d
        assert all(res_with[1] <= s < e <= res_with[2] for s, e in d_out)
    d_rej: list[tuple[int, int]] = []
    assert _check_similarity_merge_with_span(
        s_ed_rej, r_ed, min_sv_len=50, del_intervals_out=d_rej
    ) == (False, 0, 0, 0)
    assert d_rej == []

    r_mp = make_complex_seq(2500, seed=5502)
    s_mp1 = r_mp[:1000] + r_mp[1030:]  # single 30 bp D
    s_mp2 = r_mp[:500] + r_mp[520:1500] + r_mp[1524:]  # double D = 44 bp
    s_mp_rej = r_mp[:1000] + r_mp[1060:]  # 60 bp D >= 50 -> rejected
    aln_mp = mappy.Aligner(seq=r_mp, preset="asm20")
    for sq, exp_d in ((s_mp1, 30), (s_mp2, 44)):
        res_plain = _check_similarity_merge_with_span(sq, r_mp, min_sv_len=50, aligner=aln_mp)
        d_out = []
        res_with = _check_similarity_merge_with_span(
            sq, r_mp, min_sv_len=50, aligner=aln_mp, del_intervals_out=d_out
        )
        assert res_plain == res_with and len(res_with) == 4
        assert res_with[0] is True and res_with[3] == exp_d
        assert sum(e - s for s, e in d_out) == exp_d
        assert all(res_with[1] <= s < e <= res_with[2] for s, e in d_out)
    d_rej_mp: list[tuple[int, int]] = []
    assert _check_similarity_merge_with_span(
        s_mp_rej, r_mp, min_sv_len=50, aligner=aln_mp, del_intervals_out=d_rej_mp
    ) == (False, 0, 0, 0)
    assert d_rej_mp == []


def test_iter5_linearizer_r4_1a_and_r4_1b_merges(tmp_path: Path) -> None:
    """Iter 5 linearizer `R4-1a` & `R4-1b` regressions (`explorer_iter5_2` §4.1–§4.2, Guardrail 2):
    1. `R4_3B_ZERO_TANDEM_1000bp` (`ref`, `name_flip_ref`, `pan` with `Sup01..14` trimming `s_c`).
    2. `R4_3B_DEFAULT_MAXCOPY1_COLLISION` (`max_copy=1` tandem folding differing piece masks).
    3. Transitive 3-component chain (`Comp 1 + Comp 2` valid merge, `Comp 3` collides with `Comp 2`).
    4. Guardrail 2: multi-`occ_s` `[3970, 4030]` spread around `rep=4000` rejected (`60 >= 50`),
       while `[3980, 4020]` (`40 < 50`) merges cleanly.
    5. Positive re-checks (`R4_3A1`, `R4_3A2`): matching-partition `real_rep` + `synth_comp`
       still merge into 1 rep / 0 spacers.
    6. `R4-1b` (`M4_multi_occ_triangle_sup8_pan`): `Hap3c` 60 bp short of `rep_is_hetero`
       is rejected (`gt == "."` and absent from `p30`/`m30` `members`).
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_case(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str,
        **extra_cfg: object,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=9500 + i))
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path,
            tag,
            gb,
            ref6,
            dedup_mode=dedup_mode,
            dedup_id=0.90,
            min_sv_len=50,
            **extra_cfg,
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    # (1) R4_3B_ZERO_TANDEM_1000bp (ref, name_flip_ref, pan)
    s_a = make_complex_seq(1000, seed=9601)
    s_b = make_complex_seq(1000, seed=9602)
    s_c = make_complex_seq(1000, seed=9603)
    s_e = make_complex_seq(1000, seed=9604)
    s_d = make_complex_seq(1500, seed=9605)
    nodes_r43b = {
        "10": s_c[:300],
        "40": s_a, "41": s_c, "42": s_b, "48": s_e, "49": s_d,
        "50": s_a, "52": s_b, "53": s_c, "58": s_e, "59": s_d,
    }
    sup14 = [
        (f"Sup{k:02d}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 15)
    ]
    paths_r43b = sup14 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("42", "+"), ("41", "+"), ("48", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3d#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_ref = _run_case("i5_r43b_ref", nodes_r43b, paths_r43b, "ref")
    ev = next(e for e in res_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al = {tok.split("src=")[1].split(",")[0]: tok for tok in ev.alleles_str.split(";")}
    assert ev.genotypes["Hap1a#1"] != ev.genotypes["Hap3a#1"]
    assert "len=4500" in al["Hap1a#1"] and "n=3" in al["Hap1a#1"]
    assert "len=3500" in al["Hap3a#1"] and "n=4" in al["Hap3a#1"]

    # Haplotype name flip must produce identical biological allele lengths (no 4500 <-> 3500 flip)
    paths_flip = sup14 + [
        (p.replace("Hap1", "ZHap1").replace("Hap2", "ZHap2").replace("Hap3", "AHap3").replace("Hap4", "AHap4"), st)
        for p, st in paths_r43b[14:]
    ]
    res_flip = _run_case("i5_r43b_flip_ref", nodes_r43b, paths_flip, "ref")
    ev_f = next(e for e in res_flip.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_f = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_f.alleles_str.split(";")}
    assert ev_f.genotypes["ZHap1a#1"] != ev_f.genotypes["AHap3a#1"]
    assert "len=4500" in al_f["ZHap1a#1"] and "len=3500" in al_f["AHap3a#1"]

    res_pan = _run_case("i5_r43b_pan", nodes_r43b, paths_r43b, "pan")
    ev_p = next(e for e in res_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_p = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_p.alleles_str.split(";")}
    assert ev_p.genotypes["Hap1a#1"] == "." and ev_p.genotypes["Hap3a#1"] == "1"
    assert "len=3500" in al_p["Hap3a#1"] and "partial_content=1" not in al_p["Hap3a#1"]
    assert "len=3700" in al_p["Hap4a#1"] and "partial_content=1" in al_p["Hap4a#1"]
    assert "len=3000" in al_p["Hap4c#1"] and "partial_content=1" not in al_p["Hap4c#1"]

    # (2) R4_3B_DEFAULT_MAXCOPY1_COLLISION (default max_copy=1 tandem fold differing piece masks)
    ts_a = make_complex_seq(80, seed=9611)
    ts_b = make_complex_seq(80, seed=9612)
    ts_c = make_complex_seq(70, seed=9613)
    ts_e = make_complex_seq(80, seed=9614)
    ts_d = make_complex_seq(80, seed=9615)
    nodes_mc1 = {
        "10": ts_c,
        "40": ts_a, "41": ts_c, "42": ts_b, "48": ts_e, "49": ts_d,
        "50": ts_a, "52": ts_b, "53": ts_c, "58": ts_e, "59": ts_d,
    }
    paths_mc1 = sup14 + [
        # Tandem-folded prefix 40+40 -> 1 copy under default max_copy=1 with identical partition
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("41", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("41", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("42", "+"), ("41", "+"), ("48", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("48", "+"), ("49", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("50", "+"), ("58", "+"), ("59", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("50", "+"), ("52", "+"), ("58", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_mc1_ref = _run_case("i5_mc1_ref", nodes_mc1, paths_mc1, "ref")
    ev_mc = next(e for e in res_mc1_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_mc = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_mc.alleles_str.split(";")}
    assert ev_mc.genotypes["Hap1a#1"] != ev_mc.genotypes["Hap3a#1"]
    assert "len=310" in al_mc["Hap1a#1"] and "len=240" in al_mc["Hap3a#1"]

    res_mc1_pan = _run_case("i5_mc1_pan", nodes_mc1, paths_mc1, "pan")
    ev_mcp = next(e for e in res_mc1_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_mcp = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_mcp.alleles_str.split(";")}
    assert "partial_content=1" not in al_mcp["Hap3a#1"] and "len=240" in al_mcp["Hap3a#1"]

    # (3) Transitive 3-component chain & (4) Guardrail 2 multi-occ_s [3970, 4030] spread
    # Directly verify solve_parallel_group exact_target logic via controlled _solve_multi_member_component hook
    p_seq_1 = make_complex_seq(200, seed=9621)
    p_seq_2 = make_complex_seq(200, seed=9622)
    p_seq_3 = make_complex_seq(200, seed=9623)
    full_3p = p_seq_1 + p_seq_2 + p_seq_3

    def _mk_occ(cid: str, hap: str, nid: str, seq: str, off: int = 0) -> ExtractedCandidate:
        return ExtractedCandidate(
            candidate_id=cid,
            parent_id=None,
            hap_id=hap,
            src_path=f"{hap}#Chr01",
            left_anchor_rank=0,
            right_anchor_rank=1,
            source_order=off,
            steps=(Step(nid, "+"),),
            provenance=(StepProvenance(hap, f"{hap}#Chr01", 0, len(seq), "+"),),
            seq=seq,
            piece_boundaries=(),
        )

    def _mk_comp(
        occs: list[ExtractedCandidate],
        piece_cand_ids: tuple[frozenset[str], ...],
        nested_by_cid: dict[str, tuple[tuple[object, tuple[int, ...]], ...]],
        rlen_by_cid: dict[str, int],
        token: str,
    ) -> linearizer_mod.ComponentResult:
        pieces = (
            Piece(occs[0].hap_id, occs[0].src_path, 0, 200, "+", p_seq_1, "source_contiguous", tuple(sorted({o.hap_id for o in occs}))),
            Piece(occs[0].hap_id, occs[0].src_path, 200, 400, "+", p_seq_2, "source_contiguous", tuple(sorted({o.hap_id for o in occs}))),
            Piece(occs[0].hap_id, occs[0].src_path, 400, 600, "+", p_seq_3, "source_contiguous", tuple(sorted({o.hap_id for o in occs}))),
        )
        sup = {o.hap_id for o in occs}
        return linearizer_mod.ComponentResult(
            is_synthetic=True,
            seq=full_3p,
            length_bp_no_n=600,
            pieces=pieces,
            support_haps=sup,
            token_str=token,
            min_hap=min(sup),
            exemplar=occs[0],
            member_occurrences=occs,
            nested_choices_by_cand_id=nested_by_cid,
            reconstructed_len_by_cand_id=rlen_by_cid,
            piece_cand_ids=piece_cand_ids,
        )

    # 3-component transitive chain: Comp 1 (sig_A) + Comp 2 (sig_A compat + sig_B mask (0,2)) -> MERGE;
    # Comp 3 (sig_B mask (0,1) != (0,2)) -> REJECTED against merged Comp 1+2!
    gb_min = SyntheticGFAFixtureBuilder()
    gb_min.add_node("1", make_complex_seq(100, seed=9624))
    gb_min.add_node("2", make_complex_seq(100, seed=9625))
    gb_min.add_node("nA", full_3p)
    gb_min.add_node("nB", full_3p)
    gb_min.add_node("nC", full_3p)
    gb_min.add_p_path("Ref#1#Chr01", [("1", "+"), ("2", "+")])
    g_min, cfg_min = _build_graph_and_config(
        tmp_path, "i5_trans_g2", gb_min, [("1", "+"), ("2", "+")], dedup_mode="pan", min_sv_len=50
    )
    sig_a = (((0, 1), (0,)),)
    sig_b = (((0, 1), (1,)),)

    c1_o = [_mk_occ(f"c1_{k}", f"H1_{k}#1", "nA", full_3p, k) for k in range(4)]
    c2_o = [_mk_occ(f"c2_{k}", f"H2_{k}#1", "nB", full_3p, k + 10) for k in range(3)]
    c3_o = [_mk_occ(f"c3_{k}", f"H3_{k}#1", "nC", full_3p, k + 20) for k in range(2)]

    comp1 = _mk_comp(
        c1_o,
        (frozenset(o.candidate_id for o in c1_o), frozenset(o.candidate_id for o in c1_o), frozenset(o.candidate_id for o in c1_o)),
        {o.candidate_id: sig_a for o in c1_o},
        {o.candidate_id: 600 for o in c1_o},
        "tok_1",
    )
    comp2 = _mk_comp(
        c2_o,
        (
            frozenset(o.candidate_id for o in c2_o),
            frozenset({c2_o[0].candidate_id, c2_o[1].candidate_id}),
            frozenset(o.candidate_id for o in c2_o),
        ),
        {c2_o[0].candidate_id: sig_a, c2_o[1].candidate_id: sig_a, c2_o[2].candidate_id: sig_b},
        {c2_o[0].candidate_id: 600, c2_o[1].candidate_id: 600, c2_o[2].candidate_id: 400},
        "tok_2",
    )
    comp3 = _mk_comp(
        c3_o,
        (
            frozenset(o.candidate_id for o in c3_o),
            frozenset(o.candidate_id for o in c3_o),
            frozenset({c3_o[0].candidate_id}),
        ),
        {c3_o[0].candidate_id: sig_a, c3_o[1].candidate_id: sig_b},
        {c3_o[0].candidate_id: 600, c3_o[1].candidate_id: 400},
        "tok_3",
    )

    # Guardrail 2: rep rlen=4000, sc has two occs with same sig_a at 3970 & 4030 (spread 60 >= 50 -> reject)
    g2_r_o = [_mk_occ(f"gr_{k}", f"GR_{k}#1", "nA", full_3p, k) for k in range(3)]
    g2_s_bad = [_mk_occ(f"gsb_{k}", f"GSB_{k}#1", "nB", full_3p, k + 10) for k in range(2)]
    g2_s_ok = [_mk_occ(f"gso_{k}", f"GSO_{k}#1", "nB", full_3p, k + 10) for k in range(2)]

    rep_g2 = _mk_comp(
        g2_r_o,
        (frozenset(), frozenset(), frozenset()),
        {o.candidate_id: sig_a for o in g2_r_o},
        {o.candidate_id: 4000 for o in g2_r_o},
        "tok_r",
    )
    sc_g2_bad = _mk_comp(
        g2_s_bad,
        (frozenset(), frozenset(), frozenset()),
        {o.candidate_id: sig_a for o in g2_s_bad},
        {g2_s_bad[0].candidate_id: 3970, g2_s_bad[1].candidate_id: 4030},
        "tok_sb",
    )
    sc_g2_ok = _mk_comp(
        g2_s_ok,
        (frozenset(), frozenset(), frozenset()),
        {o.candidate_id: sig_a for o in g2_s_ok},
        {g2_s_ok[0].candidate_id: 3980, g2_s_ok[1].candidate_id: 4020},
        "tok_so",
    )

    orig_smmc = linearizer_mod._solve_multi_member_component
    try:
        comp_map: dict[str, list[linearizer_mod.ComponentResult]] = {
            "nA": [comp1],
            "nB": [comp2],
            "nC": [comp3],
        }
        linearizer_mod._solve_multi_member_component = (  # type: ignore[assignment]
            lambda comp, *a, **kw: comp_map[comp[0].steps[0].node_id]
        )
        # Need >=2 distinct dedup candidates per UF component to enter _solve_multi_member_component;
        # use dummy 2-step distinct seqs mapped via comp_map.
        gb_min.add_node("nA2", full_3p + "A")
        gb_min.add_node("nB2", full_3p + "C")
        gb_min.add_node("nC2", full_3p + "G")
        g_min2, cfg_min2 = _build_graph_and_config(
            tmp_path, "i5_trans_g2b", gb_min, [("1", "+"), ("2", "+")], dedup_mode="pan", min_sv_len=50
        )
        cands_t3 = (
            c1_o
            + [_mk_occ("c1_x", "H1_x#1", "nA2", full_3p + "A", 9)]
            + c2_o
            + [_mk_occ("c2_x", "H2_x#1", "nB2", full_3p + "C", 19)]
            + c3_o
            + [_mk_occ("c3_x", "H3_x#1", "nC2", full_3p + "G", 29)]
        )
        # Share node nA/nB/nC within each pair so UF forms 3 multi-member components
        cands_t3 = [
            ExtractedCandidate(**{**o.__dict__, "steps": (Step("nA", "+"), Step("nA2", "+"))})
            if o.candidate_id == "c1_x"
            else ExtractedCandidate(**{**o.__dict__, "steps": (Step("nB", "+"), Step("nB2", "+"))})
            if o.candidate_id == "c2_x"
            else ExtractedCandidate(**{**o.__dict__, "steps": (Step("nC", "+"), Step("nC2", "+"))})
            if o.candidate_id == "c3_x"
            else o
            for o in cands_t3
        ]
        sol_t3 = solve_parallel_group(
            left_anchor_rank=0,
            right_anchor_rank=1,
            backbone_slice_steps=(),
            candidates=cands_t3,
            graph=g_min2,  # type: ignore[arg-type]
            config=cfg_min2,
        )
        assert len(sol_t3.non_backbone_reps) == 2
        # Comp 1 and Comp 2 merged into rep 1; Comp 3 stayed separate in rep 2!
        assert sol_t3.cand_to_rep_idx["c1_0"] == sol_t3.cand_to_rep_idx["c2_0"]
        assert sol_t3.cand_to_rep_idx["c3_0"] != sol_t3.cand_to_rep_idx["c1_0"]

        # Guardrail 2 check: [3970, 4030] rejected (2 reps), [3980, 4020] accepted (1 rep)
        comp_map = {"nA": [rep_g2], "nB": [sc_g2_bad]}
        cands_g2_bad = (
            g2_r_o
            + [ExtractedCandidate(**{**_mk_occ("gr_x", "GR_x#1", "nA2", full_3p + "A", 9).__dict__, "steps": (Step("nA", "+"), Step("nA2", "+"))})]
            + g2_s_bad
            + [ExtractedCandidate(**{**_mk_occ("gsb_x", "GSB_x#1", "nB2", full_3p + "C", 19).__dict__, "steps": (Step("nB", "+"), Step("nB2", "+"))})]
        )
        sol_g2_bad = solve_parallel_group(0, 1, (), cands_g2_bad, g_min2, cfg_min2)  # type: ignore[arg-type]
        assert len(sol_g2_bad.non_backbone_reps) == 2

        comp_map = {"nA": [rep_g2], "nB": [sc_g2_ok]}
        cands_g2_ok = (
            g2_r_o
            + [ExtractedCandidate(**{**_mk_occ("gr_x", "GR_x#1", "nA2", full_3p + "A", 9).__dict__, "steps": (Step("nA", "+"), Step("nA2", "+"))})]
            + g2_s_ok
            + [ExtractedCandidate(**{**_mk_occ("gso_x", "GSO_x#1", "nB2", full_3p + "C", 19).__dict__, "steps": (Step("nB", "+"), Step("nB2", "+"))})]
        )
        sol_g2_ok = solve_parallel_group(0, 1, (), cands_g2_ok, g_min2, cfg_min2)  # type: ignore[arg-type]
        assert len(sol_g2_ok.non_backbone_reps) == 1
    finally:
        linearizer_mod._solve_multi_member_component = orig_smmc

    # (5) Positive re-checks: R4_3A1 (hetero real_rep + matching synth_comp) & R4_3A2 (homo real_rep + matching synth_comp)
    ra = make_complex_seq(500, seed=9631)
    rb = make_complex_seq(500, seed=9632)
    rc_s = make_complex_seq(600, seed=9633)
    rd_s = make_complex_seq(800, seed=9634)
    nodes_3a1 = {
        "10": ra + rc_s + rd_s,
        "40": ra, "41": rb, "42": rc_s, "43": rd_s,
        "50": ra, "51": rb, "52": rc_s, "53": rd_s,
    }
    paths_3a1 = sup14[:6] + [
        # Comp 1: childless hetero real_rep (exact 4-piece partition [40, 41, 42, 43], is_synthetic=False)
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("42", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("42", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("42", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("41", "+"), ("42", "+"), ("5", "+"), ("6", "+")]),
        # Comp 2: matching-partition synth_comp (b_sec=50+52+53=1900 > Hap4=50+51+52=1600 -> is_synthetic=True)
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("52", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_3a1 = _run_case("i5_r4_3a1_ref", nodes_3a1, paths_3a1, "ref")
    ev_3a1 = next(e for e in res_3a1.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_3a1 = [r for r in res_3a1.inserts_rows if r["event_id"] == ev_3a1.event_id]
    assert len(ins_3a1) == 4 and all(r["join_before"] != "spacer" for r in ins_3a1)
    assert ins_3a1[1]["members"] == "Hap1a#1,Hap1b#1,Hap2b#1,Hap4#1"

    # R4_3A2: homogeneous 4-piece tandem-folded real_rep (is_synthetic=False) absorbing 4-piece synth_comp
    nodes_3a2 = {
        "10": ra + rc_s + rd_s,
        "40": ra, "41": rb, "42": rc_s, "43": rd_s,
        "50": ra, "51": rb, "52": rc_s, "53": rd_s,
    }
    paths_3a2 = sup14[:6] + [
        # Comp 1: tandem-folded 40+40, 41+41, 42+42, 43 -> 4-piece homogeneous real_rep [ra, rb, rc_s, rd_s]
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("41", "+"), ("41", "+"), ("42", "+"), ("42", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("40", "+"), ("40", "+"), ("41", "+"), ("41", "+"), ("42", "+"), ("42", "+"), ("43", "+"), ("5", "+"), ("6", "+")]),
        # Comp 2: matching 4-piece synth_comp (b_sec=50+52+53=1900 > Hap3=50+51+52=1600 -> is_synthetic=True)
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("52", "+"), ("53", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("50", "+"), ("51", "+"), ("52", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_3a2 = _run_case("i5_r4_3a2_ref", nodes_3a2, paths_3a2, "ref")
    ev_3a2 = next(e for e in res_3a2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_3a2 = [r for r in res_3a2.inserts_rows if r["event_id"] == ev_3a2.event_id]
    assert len(ins_3a2) == 4 and all(r["join_before"] != "spacer" for r in ins_3a2)
    assert ins_3a2[1]["members"] == "Hap1a#1,Hap1b#1,Hap3#1"

    # (6) R4-1b: M4_multi_occ_triangle_sup8_pan
    p30 = make_complex_seq(30, seed=9641)
    sc1 = make_complex_seq(1500, seed=9642)
    m30 = make_complex_seq(30, seed=9643)
    sc2 = make_complex_seq(1500, seed=9644)
    sc1p = mutate_seq_snps(sc1, [i * 100 + 15 for i in range(12)])
    sc2p = mutate_seq_snps(sc2, [i * 100 + 25 for i in range(12)])
    nodes_m4 = {
        "10": sc1 + sc2,
        "p30": p30, "sc1": sc1, "m30": m30, "sc2": sc2,
        "sc1p": sc1p, "m30p": m30, "sc2p": sc2p,
    }
    paths_m4 = sup14[:8] + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p30", "+"), ("sc1", "+"), ("m30", "+"), ("sc2", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p30", "+"), ("sc1", "+"), ("m30", "+"), ("sc2", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("sc1", "-"), ("sc2", "-"), ("5", "+"), ("6", "+")]),
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("sc1p", "+"), ("m30p", "+"), ("sc2p", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("sc1p", "+"), ("m30p", "+"), ("sc2p", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("sc1p", "+"), ("sc2p", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_m4 = _run_case("i5_m4_tri_pan", nodes_m4, paths_m4, "pan")
    ev_m4 = next(e for e in res_m4.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_m4 = [r for r in res_m4.inserts_rows if r["event_id"] == ev_m4.event_id]
    assert ev_m4.genotypes["Hap3c#1"] != ev_m4.genotypes["Hap1a#1"]
    assert ev_m4.genotypes["Hap2#1"] == "."
    p30_m30_rows = [r for r in ins_m4 if (r["pan_end"] - r["pan_start"]) == 30]
    assert len(p30_m30_rows) == 2
    assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in p30_m30_rows)

    # With Sup carrying sc1+m30+sc2, rejected rc (Hap3a/3b/3c) drops cleanly to ".", leaving only novel p30
    res_m4_supm = _run_case(
        "i5_m4_tri_supm_pan", {**nodes_m4, "10": sc1 + m30 + sc2}, paths_m4, "pan"
    )
    ev_m4s = next(e for e in res_m4_supm.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_m4s = [r for r in res_m4_supm.inserts_rows if r["event_id"] == ev_m4s.event_id]
    assert ev_m4s.genotypes["Hap1a#1"] == ev_m4s.genotypes["Hap1b#1"] == "1"
    assert ev_m4s.genotypes["Hap2#1"] == "." and ev_m4s.genotypes["Hap3c#1"] == "."
    assert len(ins_m4s) == 1 and ins_m4s[0]["members"] == "Hap1a#1,Hap1b#1"
    assert ins_m4s[0]["pan_end"] - ins_m4s[0]["pan_start"] == 30


def test_iter5_linearizer_r4_2_exact_rep0_coverage_and_healing(tmp_path: Path) -> None:
    """Iter 5 linearizer `R4-2` regressions (`explorer_iter5_2` §4.3, Guardrail 4):
    1. `R4_1C2` (`ov=1`) & `CHECK #3` (`ov=20, 25`) on `15a(40)+15b(40)`:
       distinct allele in `ref`; full `80 bp` intact in `pan` (`Hap2 == "."`,
       `members == "Hap1a#1,Hap1b#1"`); cleanly dropped with full `Sup`.
    2. `R4_1D` (`m30(30)` internal `30D` + `t30(30)` tail): BOTH `m30` & `t30` excluded
       for `Hap2` (`Hap2 == "."`, absent from BOTH `members`); cleanly dropped with full `Sup`.
    3. `CHECK #1` (`3x` dup `20D` + intact/`30D` union): zero double-counting.
    4. `CHECK #2` (`T2` & `T2b` `940/1000` on `14(1000)`): distinct allele in `ref`,
       `Hap2 == "."` in `pan`.
    5. Benign non-SV trims (`29/60` and `19+19/40+40`): stay `"1"` (zero false exclusions).
    6. Isolated `1 bp` replacement (Guardrail 4): exact exclusion without false healing.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_r42(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_mode: str,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=9700 + i))
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=dedup_mode, dedup_id=0.90, min_sv_len=50
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    s11 = make_complex_seq(1500, seed=9711)
    s14 = make_complex_seq(1000, seed=9712)
    s15a = make_complex_seq(40, seed=9713)
    s15b = make_complex_seq(40, seed=9714)
    s12 = make_complex_seq(1500, seed=9715)
    s14p = mutate_seq_snps(s14, [i * 100 + 20 for i in range(10)])
    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]

    # (1) R4_1C2 (ov=1) + CHECK #3 (ov=20, ov=25) on 15a(40)+15b(40)
    for ov in (1, 20, 25):
        nodes_ov = {
            "10": s11 + s14 + s12,
            "11": s11, "14": s14, "15a": s15a, "15b": s15b, "12": s12,
            "14ov": s14p + s15a[:ov],
        }
        paths_ov = sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15a", "+"), ("15b", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15a", "+"), ("15b", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14ov", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        res_ref = _run_r42(f"i5_r41c2_ov{ov}_ref", nodes_ov, paths_ov, "ref")
        ev_r = next(e for e in res_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        al_r = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_r.alleles_str.split(";")}
        assert ev_r.genotypes["Hap1a#1"] != ev_r.genotypes["Hap2#1"]
        assert "len=4080" in al_r["Hap1a#1"] and "len=4000" in al_r["Hap2#1"]

        res_pan = _run_r42(f"i5_r41c2_ov{ov}_pan", nodes_ov, paths_ov, "pan")
        ev_p = next(e for e in res_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_p = [r for r in res_pan.inserts_rows if r["event_id"] == ev_p.event_id]
        assert ev_p.genotypes["Hap1a#1"] == ev_p.genotypes["Hap1b#1"] == "1"
        assert ev_p.genotypes["Hap2#1"] == "."
        assert sum(r["pan_end"] - r["pan_start"] for r in ins_p) == 80
        assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in ins_p)

        # Full Sup drops completely to 0 INS at 3..4
        nodes_ov_full = {**nodes_ov, "10": s11 + s14 + s15a + s15b + s12}
        res_full = _run_r42(f"i5_r41c2_ov{ov}_full_pan", nodes_ov_full, paths_ov, "pan")
        assert not [e for e in res_full.sv_events if e.event_type == "INS" and e.anchors == "3..4"]

    # (2) R4_1D: m30(30) internal 30D + t30(30) tail
    m30 = make_complex_seq(30, seed=9721)
    s15 = make_complex_seq(1000, seed=9722)
    t30 = make_complex_seq(30, seed=9723)
    s15p = mutate_seq_snps(s15, [i * 100 + 30 for i in range(10)])
    nodes_r41d = {
        "10": s11 + s14 + s15 + s12,
        "11": s11, "14": s14, "m30": m30, "15": s15, "t30": t30, "12": s12,
        "20": s14p + s15p,
    }
    paths_r41d = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m30", "+"), ("15", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m30", "+"), ("15", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("20", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("15", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_1d = _run_r42("i5_r41d_pan", nodes_r41d, paths_r41d, "pan")
    ev_1d = next(e for e in res_1d.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_1d = [r for r in res_1d.inserts_rows if r["event_id"] == ev_1d.event_id]
    assert ev_1d.genotypes["Hap1a#1"] == ev_1d.genotypes["Hap1b#1"] == "1"
    assert ev_1d.genotypes["Hap2#1"] == "." and ev_1d.genotypes["Hap3#1"] == "."
    assert len(ins_1d) == 2 and all(r["members"] == "Hap1a#1,Hap1b#1" for r in ins_1d)

    # R4_1D with full Sup cleanly drops both m30 & t30 via §4.4
    nodes_r41d_full = {**nodes_r41d, "10": s11 + s14 + m30 + s15 + t30 + s12}
    res_1d_full = _run_r42("i5_r41d_full_pan", nodes_r41d_full, paths_r41d, "pan")
    assert not [e for e in res_1d_full.sv_events if e.event_type == "INS" and e.anchors == "3..4"]

    # (3) CHECK #1: 3x duplicate 14d(20D) split at global ref node "2" each misses 1500 bp
    # flank (member_excluded_uncov_bp >= 50) + non-benign 20D/30D -> atomically sub-sliced with
    # 0 spacers (1000 bp total, 980 bp for Hap2, 1000 bp for Hap3), AND under Sup == 11+14d20+12
    # Hap2 drops cleanly to gt == "." (no 50 bp ghost allele). Also verify true sub-SV (< min_sv_len)
    # 20D / 30D with intact 11/12 flanks stays unsplit (len == 1).
    s14_d20 = s14[:400] + s14[420:]
    s14_d30 = s14[:400] + s14[430:]
    nodes_chk1 = {
        "10": s11 + s12,
        "11": s11, "14": s14, "14d20": s14_d20, "14d30": s14_d30, "12": s12,
    }
    paths_chk1 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14d20", "+"), ("2", "+"), ("14d20", "+"), ("2", "+"), ("14d20", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("2", "+"), ("14d30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_chk1 = _run_r42("i5_chk1_pan", nodes_chk1, paths_chk1, "pan")
    ev_c1 = next(e for e in res_chk1.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_c1 = [r for r in res_chk1.inserts_rows if r["event_id"] == ev_c1.event_id]
    assert ev_c1.genotypes["Hap2#1"] not in ("0", ".")
    assert ev_c1.genotypes["Hap3#1"] not in ("0", ".")
    assert all(r["join_before"] != "spacer" for r in ins_c1)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_c1) == 1000
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_c1 if "Hap2#1" in r["members"].split(",")) == 980
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_c1 if "Hap3#1" in r["members"].split(",")) == 1000

    # CHECK #1 with Sup == 11 + 14d20 + 12 (P1I): Hap2 must drop cleanly to "." with 0 ghost pieces
    res_chk1_sup = _run_r42("i5_chk1_sup_h2_pan", {**nodes_chk1, "10": s11 + s14_d20 + s12}, paths_chk1, "pan")
    ev_c1s = next(e for e in res_chk1_sup.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_c1s = [r for r in res_chk1_sup.inserts_rows if r["event_id"] == ev_c1s.event_id]
    assert ev_c1s.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_c1s)

    # True sub-SV (< min_sv_len) same-tdev 20D/30D with intact flanks stays unsplit (len == 1)
    paths_chk1_sub = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14d20", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14d30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_chk1_sub = _run_r42("i5_chk1_true_subsv_pan", nodes_chk1, paths_chk1_sub, "pan")
    ev_c1sub = next(e for e in res_chk1_sub.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_c1sub = [r for r in res_chk1_sub.inserts_rows if r["event_id"] == ev_c1sub.event_id]
    assert len(ins_c1sub) == 1
    assert ins_c1sub[0]["members"] == "Hap1a#1,Hap1b#1,Hap2#1,Hap3#1"

    # (4) CHECK #2: T2 & T2b (940/1000 on 14(1000), 60 bp deleted)
    nodes_t2 = {
        "10": s11 + s12,
        "11": s11, "14": s14, "14_940": s14[:940], "12": s12,
    }
    paths_t2 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14_940", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_t2_ref = _run_r42("i5_t2_ref", nodes_t2, paths_t2, "ref")
    ev_t2r = next(e for e in res_t2_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_t2r = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_t2r.alleles_str.split(";")}
    assert "len=4000" in al_t2r["Hap1a#1"] and "len=3000" in al_t2r["Hap2#1"]
    res_t2_pan = _run_r42("i5_t2_pan", nodes_t2, paths_t2, "pan")
    ev_t2p = next(e for e in res_t2_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_t2p.genotypes["Hap1a#1"] == "1" and ev_t2p.genotypes["Hap2#1"] == "."
    res_t2b_pan = _run_r42("i5_t2b_pan", {**nodes_t2, "10": s11 + s14[:940] + s12}, paths_t2, "pan")
    ev_t2bp = next(e for e in res_t2b_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_t2bp.genotypes["Hap1a#1"] == "1" and ev_t2bp.genotypes["Hap2#1"] == "."

    # (5) Benign non-SV trims: 29/60 (31 bp del) and 19+19/40+40 (42 bp del) must stay "1"
    s60 = make_complex_seq(60, seed=9731)
    nodes_b31 = {
        "10": s11 + s12,
        "11": s11, "14": s14, "60n": s60, "12": s12,
        "20": s14p + s60[:29],
    }
    paths_b31 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("60n", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("60n", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("20", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_b31 = _run_r42("i5_b31_pan", nodes_b31, paths_b31, "pan")
    ev_b31 = next(e for e in res_b31.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_b31.genotypes["Hap1a#1"] == ev_b31.genotypes["Hap2#1"] == "1"

    # 19+19 on 15a(40)+15b(40) = 42 bp < 50 deleted (via 14'+15a[:19] and 15b[21:]+12' inside tdev)
    s15_2 = make_complex_seq(1000, seed=9732)
    nodes_b42 = {
        "10": s11 + s12,
        "11": s11, "14": s14, "15a": s15a, "15b": s15b, "15": s15_2, "12": s12,
        "20": s14p + s15a[:19] + s15b[21:] + s15_2,
    }
    paths_b42 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15a", "+"), ("15b", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("15a", "+"), ("15b", "+"), ("15", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("20", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_b42 = _run_r42("i5_b42_pan", nodes_b42, paths_b42, "pan")
    ev_b42 = next(e for e in res_b42.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_b42.genotypes["Hap1a#1"] == ev_b42.genotypes["Hap2#1"] == "1"

    # (6) Guardrail 4: isolated 1 bp step replaced by active >= min_sv_len child insertion
    s_ins200 = make_complex_seq(200, seed=9741)
    nodes_iso1 = {
        "10": s11 + s12,
        "11": s11, "bp1": "A", "12": s12, "ins200": s_ins200,
    }
    paths_iso1 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("bp1", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("bp1", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ins200", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_iso1 = _run_r42("i5_iso1_pan", nodes_iso1, paths_iso1, "pan")
    ev_iso = next(e for e in res_iso1.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_iso.genotypes["Hap2#1"] != ev_iso.genotypes["Hap1a#1"]
    ins_iso = [r for r in res_iso1.inserts_rows if r["event_id"] == ev_iso.event_id]
    assert any(r["pan_end"] - r["pan_start"] == 200 and r["members"] == "Hap2#1" for r in ins_iso)


def test_iter5_stage55_short_piece_context_check(tmp_path: Path) -> None:
    """Iter 5 §4.4 Stage 5.5 `< min_sv_len` `source_contiguous` context check (Guardrails 5–7):
    1. Direct `hit_covers_subinterval_mappy` verification across all 7 edge-case families.
    2. `T9_m15_t35_full_sup_pan` cleanly dropped (0 INS).
    3. Reverse-strand `-` consecutive `< min_sv_len` pieces (`[A_masked, B15, C15]` &
       `[B15, C15, D_masked]`, Guardrail 5) cleanly dropped; partial novel survivor repairs
       `join_before` (`"synthetic"` at idx 0 / `"spacer"` after dropped piece).
    4. `spacer_n=0` dropped short piece passes `[INV2]` cleanly.
    5. Divergent `24/30` (`44/50 = 88.0% < 90%`) and internal-`5D` (`25/30`) short pieces preserved (while `26/30` = `46/50 = 92.0% >= 90%` under 50 bp aligned-window semantics).
    6. `ADV-7` (`p30+m30`) preserved intact under §4.4.
    """
    # (1) Direct hit_covers_subinterval_mappy edge-case families
    h_exact = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=230, strand=1, cs=":230")
    assert hit_covers_subinterval_mappy(h_exact, 100, 130, min_identity=0.90) is True
    # Reverse strand exact
    h_rev = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=230, strand=-1, cs=":230")
    assert hit_covers_subinterval_mappy(h_rev, 100, 130, min_identity=0.90) is True
    # ADV-7: 30 bp novel insertion inside [100, 130) -> I(30) -> False
    h_adv7_ins = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=200, strand=1, cs=":100+" + "a" * 30 + ":100")
    assert hit_covers_subinterval_mappy(h_adv7_ins, 100, 130, min_identity=0.90) is False
    # ADV-7 prefix p30: hit starts at q_st=30 == p_e -> fast-skip False
    h_adv7_pref = SimpleNamespace(q_st=30, q_en=230, r_st=0, r_en=200, strand=1, cs=":200")
    assert hit_covers_subinterval_mappy(h_adv7_pref, 0, 30, min_identity=0.90) is False
    # Under 50 bp aligned-window semantics: 6 SNPs in 30 bp (44/50 = 88.0% < 90%) -> False;
    # 4 SNPs in 30 bp inside exact flanks (46/50 = 92.0% >= 90%) -> True; 3 SNPs (27/30 = 90.0%) -> True
    h_24 = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=230, strand=1, cs=":102*ac:4*ac:4*ac:4*ac:4*ac:4*ac:102")
    assert hit_covers_subinterval_mappy(h_24, 100, 130, min_identity=0.90) is False
    h_26 = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=230, strand=1, cs=":105*ac:5*ac:5*ac:5*ac:106")
    assert hit_covers_subinterval_mappy(h_26, 100, 130, min_identity=0.90) is True
    h_27 = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=230, strand=1, cs=":105*ac:5*ac:5*ac:112")
    assert hit_covers_subinterval_mappy(h_27, 100, 130, min_identity=0.90) is True
    # Internal 5D inside (100, 130) -> 30/35 = 85.7% < 90% -> False
    h_int5d = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=235, strand=1, cs=":115-" + "a" * 5 + ":115")
    assert hit_covers_subinterval_mappy(h_int5d, 100, 130, min_identity=0.90) is False
    # Boundary 5D at q_p == p_s (outside (p_s, p_e)) -> NOT penalized -> True
    h_bnd5d = SimpleNamespace(q_st=0, q_en=230, r_st=0, r_en=235, strand=1, cs=":100-" + "a" * 5 + ":130")
    assert hit_covers_subinterval_mappy(h_bnd5d, 100, 130, min_identity=0.90) is True
    # Chimeric 2-piece wave split: piece 0 covers [100, 115) (15 bp), piece 1 covers [115, 130) (15 bp) -> False
    h_chim = SimpleNamespace(q_st=0, q_en=230, r_st=385, r_en=815, strand=1, cs=":115-" + "n" * 200 + ":115")
    assert (
        hit_covers_subinterval_mappy(
            h_chim, 100, 130, min_identity=0.90, t_starts=[0, 700], t_ends=[500, 1300]
        )
        is False
    )

    # Shared full-pipeline runner for §4.4 cases
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_s44(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        spacer_n: int = 100,
    ) -> object:
        gb = SyntheticGFAFixtureBuilder()
        for i in range(1, 7):
            gb.add_node(str(i), make_complex_seq(400, seed=9800 + i))
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path,
            tag,
            gb,
            ref6,
            dedup_mode="pan",
            dedup_id=0.90,
            min_sv_len=50,
            spacer_n=spacer_n,
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    s11 = make_complex_seq(1500, seed=9811)
    s14 = make_complex_seq(1000, seed=9812)
    m15 = make_complex_seq(15, seed=9813)
    s15 = make_complex_seq(1000, seed=9814)
    t35 = make_complex_seq(35, seed=9815)
    s12 = make_complex_seq(1500, seed=9816)
    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]

    # (2) T9_m15_t35_full_sup_pan -> cleanly dropped (0 INS at 3..4)
    nodes_t9 = {
        "10": s11 + s14 + m15 + s15 + t35 + s12,
        "11": s11, "14": s14, "m15": m15, "15": s15, "t35": t35, "12": s12,
    }
    paths_t9 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m15", "+"), ("15", "+"), ("t35", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m15", "+"), ("15", "+"), ("t35", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("15", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_t9 = _run_s44("i5_t9_full_pan", nodes_t9, paths_t9)
    assert not [e for e in res_t9.sv_events if e.event_type == "INS" and e.anchors == "3..4"]

    # (3) Reverse-strand '-' consecutive < min_sv_len pieces ([A_masked, B15, C15] & [B15, C15, D_masked])
    # plus partial drop join_before repair & Guardrail 5 fixed-point propagation
    seq_a = make_complex_seq(500, seed=9821)
    seq_b = make_complex_seq(15, seed=9822)
    seq_c = make_complex_seq(15, seed=9823)
    seq_d = make_complex_seq(500, seed=9824)
    seq_nov = make_complex_seq(15, seed=9825)
    seq_k = make_complex_seq(300, seed=9826)

    gb_rev = SyntheticGFAFixtureBuilder()
    for i in range(1, 5):
        gb_rev.add_node(str(i), make_complex_seq(200, seed=9830 + i))
    gb_rev.add_p_path("Ref#1#Chr01", [(str(i), "+") for i in range(1, 5)])
    g_rev, cfg_rev = _build_graph_and_config(
        tmp_path, "i5_rev_s44", gb_rev, [(str(i), "+") for i in range(1, 5)], dedup_mode="pan", min_sv_len=50
    )

    def _mk_simple_rep(rep_idx: int, pieces: tuple[Piece, ...], sup_cnt: int, cid: str) -> tuple[ClusterRepresentative, ParallelGroupSolution]:
        tot_len = sum(len(p.seq) for p in pieces)
        ex = ExtractedCandidate(
            candidate_id=cid,
            parent_id=None,
            hap_id="H#1",
            src_path="H#1#Chr01",
            left_anchor_rank=0,
            right_anchor_rank=1,
            source_order=rep_idx,
            steps=(),
            provenance=(),
            seq="",
            piece_boundaries=(),
        )
        rep = ClusterRepresentative(
            rep_idx=1,
            is_synthetic=True,
            seq="",
            length_bp_no_n=tot_len,
            pieces=pieces,
            support_haps={f"S{k}#1" for k in range(sup_cnt)},
            token_str=f"t{rep_idx}",
            min_hap="S0#1",
            exemplar=ex,
            member_occurrences=[],
            nested_choices_by_cand_id={},
            reconstructed_len_by_cand_id={},
            piece_cand_ids=tuple(frozenset({cid}) for _ in pieces),
            support_hap_count=sup_cnt,
        )
        sol = ParallelGroupSolution(
            left_anchor_rank=0,
            right_anchor_rank=1,
            has_nonempty_backbone=False,
            backbone_seq="",
            non_backbone_reps=(rep,),
            cand_to_rep_idx={cid: 1},
            cand_to_nested_sig={cid: ()},
            cand_to_reconstructed_len={cid: tot_len},
            cand_to_flags={cid: (False, False, False)},
        )
        return rep, sol

    empty_cen = CentromereOutput((), (), (), (), ())

    # 3a: Reverse-strand [A_masked(500), B(15), C(15)] -> both B & C drop (left-to-right chain on '-' strand)
    sup_p_abc = Piece("Sup#1", "Sup#1#Chr01", 0, 530, "+", seq_a + seq_b + seq_c, "source_contiguous", ("Sup#1",))
    _, sol_sup_abc = _mk_simple_rep(0, (sup_p_abc,), 10, "sup_abc")
    # On '-' strand, piece order goes from higher src coords down to lower src coords:
    # A=[30, 530), B=[15, 30), C=[0, 15)
    p_a_r = Piece("H#1", "H#1#Chr01", 30, 530, "-", seq_a, "source_contiguous", ("H#1",))
    p_b_r = Piece("H#1", "H#1#Chr01", 15, 30, "-", seq_b, "source_contiguous", ("H#1",))
    p_c_r = Piece("H#1", "H#1#Chr01", 0, 15, "-", seq_c, "source_contiguous", ("H#1",))
    _, sol_q_abc = _mk_simple_rep(1, (p_a_r, p_b_r, p_c_r), 2, "q_abc")
    sols_3a = {(0, 1): sol_sup_abc, (1, 2): sol_q_abc}
    guards_3a = {p: GroupRepeatGuardResult() for p in sols_3a}
    _deduplicate_representatives_across_chromosome(g_rev, cfg_rev, [(0, 1), (1, 2)], guards_3a, sols_3a, empty_cen)  # type: ignore[arg-type]
    assert sol_q_abc.non_backbone_reps == ()

    # 3b: Reverse-strand [B(15), C(15), D_masked(500)] -> right-to-left fixed-point chain drops C then B
    sup_p_bcd = Piece("Sup#1", "Sup#1#Chr01", 0, 530, "+", seq_b + seq_c + seq_d, "source_contiguous", ("Sup#1",))
    _, sol_sup_bcd = _mk_simple_rep(0, (sup_p_bcd,), 10, "sup_bcd")
    p_b2 = Piece("H#1", "H#1#Chr01", 515, 530, "-", seq_b, "source_contiguous", ("H#1",))
    p_c2 = Piece("H#1", "H#1#Chr01", 500, 515, "-", seq_c, "source_contiguous", ("H#1",))
    p_d2 = Piece("H#1", "H#1#Chr01", 0, 500, "-", seq_d, "source_contiguous", ("H#1",))
    _, sol_q_bcd = _mk_simple_rep(1, (p_b2, p_c2, p_d2), 2, "q_bcd")
    sols_3b = {(0, 1): sol_sup_bcd, (1, 2): sol_q_bcd}
    guards_3b = {p: GroupRepeatGuardResult() for p in sols_3b}
    _deduplicate_representatives_across_chromosome(g_rev, cfg_rev, [(0, 1), (1, 2)], guards_3b, sols_3b, empty_cen)  # type: ignore[arg-type]
    assert sol_q_bcd.non_backbone_reps == ()

    # 3c: Partial drop join_before repair: [K(300, kept) + A_right_masked(500), B(15, hom), C(15, novel)] -> C gets "spacer"
    sup_p_ka = Piece("Sup#1", "Sup#1#Chr01", 0, 515, "+", seq_a + seq_b, "source_contiguous", ("Sup#1",))
    _, sol_sup_ka = _mk_simple_rep(0, (sup_p_ka,), 10, "sup_ka")
    p_ka = Piece("H#1", "H#1#Chr01", 0, 800, "+", seq_k + seq_a, "source_contiguous", ("H#1",))
    p_b_p = Piece("H#1", "H#1#Chr01", 800, 815, "+", seq_b, "source_contiguous", ("H#1",))
    p_nov_p = Piece("H#1", "H#1#Chr01", 815, 830, "+", seq_nov, "source_contiguous", ("H#1",))
    _, sol_q_ka = _mk_simple_rep(1, (p_ka, p_b_p, p_nov_p), 2, "q_ka")
    sols_3c = {(0, 1): sol_sup_ka, (1, 2): sol_q_ka}
    guards_3c = {p: GroupRepeatGuardResult() for p in sols_3c}
    _deduplicate_representatives_across_chromosome(g_rev, cfg_rev, [(0, 1), (1, 2)], guards_3c, sols_3c, empty_cen)  # type: ignore[arg-type]
    kept_ps = sol_q_ka.non_backbone_reps[0].pieces
    assert len(kept_ps) == 2 and kept_ps[1].seq == seq_nov and kept_ps[1].join_before == "spacer"

    # (4) spacer_n=0 dropped short piece passes [INV2] cleanly (m15 dropped, t35 kept as novel!)
    nov35 = make_complex_seq(35, seed=9841)
    nodes_sp0 = {
        "10": s11 + s14 + m15 + s15 + s12,
        "11": s11, "14": s14, "m15": m15, "15": s15, "t35": nov35, "12": s12,
    }
    res_sp0 = _run_s44("i5_sp0_s44", nodes_sp0, paths_t9, spacer_n=0)
    ev_sp0 = next(e for e in res_sp0.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_sp0 = [r for r in res_sp0.inserts_rows if r["event_id"] == ev_sp0.event_id]
    assert len(ins_sp0) == 1 and ins_sp0[0]["pan_end"] - ins_sp0[0]["pan_start"] == 35

    # (5) Divergent 24/30 (6 SNPs, 44/50 = 88.0% < 90%) & internal-5D (25 bp vs 30 bp in Sup) short pieces preserved
    m30_base = make_complex_seq(30, seed=9851)
    m30_div6 = mutate_seq_snps(m30_base, [2, 7, 12, 17, 22, 27])  # 44/50 = 88.0% < 90%
    m25_d5 = m30_base[:12] + m30_base[17:]  # 5 bp internal deletion relative to Sup's m30_base
    nodes_div = {
        "10": s11 + s14 + m30_base + s15 + m30_base + s12,
        "11": s11, "14": s14, "m30d": m30_div6, "15": s15, "t25": m25_d5, "12": s12,
    }
    paths_div = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m30d", "+"), ("15", "+"), ("t25", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("m30d", "+"), ("15", "+"), ("t25", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "-"), ("15", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_div = _run_s44("i5_div_int5d_s44", nodes_div, paths_div)
    ev_div = next(e for e in res_div.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_div = [r for r in res_div.inserts_rows if r["event_id"] == ev_div.event_id]
    assert sorted(r["pan_end"] - r["pan_start"] for r in ins_div) == [25, 30]
    assert m30_div6 in res_div.pan_seq and m25_d5 in res_div.pan_seq

    # (6) ADV-7 preservation: novel p30 + m30 survive intact under §4.4
    p30_a7 = make_complex_seq(30, seed=9861)
    m30_a7 = make_complex_seq(30, seed=9862)
    nodes_a7 = {
        "10": s11 + s12,
        "p30": p30_a7, "11": s11, "m30": m30_a7, "12": s12,
    }
    paths_a7 = sup6 + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p30", "+"), ("11", "+"), ("m30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "-"), ("12", "-"), ("5", "+"), ("6", "+")]),
    ]
    res_a7 = _run_s44("i5_adv7_s44", nodes_a7, paths_a7)
    ev_a7 = next(e for e in res_a7.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_a7 = [r for r in res_a7.inserts_rows if r["event_id"] == ev_a7.event_id]
    assert len(ins_a7) == 2 and all(r["members"] == "Hap1#1" for r in ins_a7)
    assert p30_a7 in res_a7.pan_seq and m30_a7 in res_a7.pan_seq


def test_iter6_bug1_rep0_split_and_subseq_cursor(tmp_path: Path) -> None:
    """Iter 6 Bug 1 + Refinement #1 (`solve_parallel_group` Phase 4a/4b `rep0` split & `_is_step_subsequence`):
    1. `BUG3_DEFAULT_CLI` (`ref` & `pan`, default `--dedup-id 0.90`): `XY80` preserved, `Hap2` owns exact pieces.
    2. `H4c` (top-level `--dedup-id 0.98`, `ref` & `pan`): `nov1+nov2` (`65 bp`) intact, `INS` emitted.
    3. `H4c-nested-v2` (default `--dedup-id 0.90`, `ref` & `pan`): `ins1`/`ins2` preserved, `Hap2c` owns full `6070 bp`.
    4. `Attack 2A-short` (top-level `len(occ) < len(rep0)` carrying `XY80`, `--dedup-id 0.99`, `ref`): `XY80` preserved.
    5. `Attack 2B` (nested pure 200 bp deletion vs child `rep0`, default `--dedup-id 0.90`, `ref` & `pan`): `Hap2` owns `3800 bp` in `ref` and `"."` in `pan`.
    6. `sc.seq == rep0_seq` (`spacer_n=0`, nested `synth_comp`): failing `occ` with `+70 bp` split cleanly, `nov1`/`nov2` intact.
    7. Refinement #1 `_is_step_subsequence`: top-level partial-step pure deletion absorbed into `rep0`, while overlapping tandem sub-slices and inversions are rejected and split out.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]

    def _run_b1(
        tag: str,
        ref_nodes: dict[str, str],
        extra_nodes: dict[str, str],
        ref_steps: list[tuple[str, str]],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str,
        dedup_id: float = 0.90,
        spacer_n: int = 100,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in ref_nodes.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref_steps)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path,
            tag,
            gb,
            ref_steps,
            dedup_mode=mode,
            dedup_id=dedup_id,
            min_sv_len=50,
            spacer_n=spacer_n,
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    base_ref6 = {str(i): make_complex_seq(400, seed=10100 + i) for i in range(1, 7)}

    # (1) BUG3_DEFAULT_CLI (ref & pan, default dedup_id=0.90)
    la = make_complex_seq(1000, seed=10111)
    ra = make_complex_seq(1000, seed=10112)
    seq_a = make_complex_seq(1500, seed=10113)
    seq_c = make_complex_seq(1500, seed=10114)
    t45 = make_complex_seq(45, seed=10115)
    seq_x = make_complex_seq(40, seed=10116)
    seq_y = make_complex_seq(40, seed=10117)
    a2 = mutate_seq_snps(seq_a, [i * 150 + 20 for i in range(10)])
    c2 = mutate_seq_snps(seq_c, [i * 150 + 25 for i in range(10)])
    t45_2 = mutate_seq_snps(t45, [15])
    bug3_nodes = {
        "La": la, "Ra": ra, "AC_T45": seq_a + seq_c + t45,
        "A2": a2, "X": seq_x, "Y": seq_y, "C2": c2, "T45_2": t45_2,
    }
    bug3_paths = [
        (f"Hap0{ch}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("AC_T45", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for ch in ("a", "b", "c", "d")
    ] + [
        (f"Hap1{ch}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("A2", "+"), ("X", "+"), ("C2", "+"), ("T45_2", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for ch in ("a", "b")
    ] + [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("A2", "+"), ("X", "+"), ("Y", "+"), ("C2", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_b3 = _run_b1(f"i6_bug3_{mode}", base_ref6, bug3_nodes, ref6, bug3_paths, mode, dedup_id=0.90)
        assert (seq_x + seq_y) in res_b3.pan_seq, f"BUG3 ({mode}): XY80 erased from pan_seq"
        ev_b3 = next(e for e in res_b3.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        al_b3 = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_b3.alleles_str.split(";")}
        assert "len=5045" in al_b3["Hap0a#1"] and "n=6" in al_b3["Hap0a#1"]
        assert "partial_content=1" not in al_b3["Hap0a#1"]
        hap2_bp = sum(
            r["pan_end"] - r["pan_start"]
            for r in res_b3.inserts_rows
            if r["event_id"] == ev_b3.event_id and "Hap2#1" in r["members"].split(",")
        )
        if mode == "ref":
            assert "len=5080" in al_b3["Hap2#1"] and hap2_bp == 5080
        else:
            assert "partial_content=1" in al_b3["Hap2#1"]
            assert f"len={hap2_bp}" in al_b3["Hap2#1"] and hap2_bp >= 2080

    # (2) H4c (top-level dedup_id=0.98, ref & pan)
    h4c_a = make_complex_seq(1500, seed=10121)
    h4c_b = make_complex_seq(1500, seed=10122)
    h4c_ap = _mutate_per_50bp_block(h4c_a, (10, 35))  # 96% id < 0.98 dedup_id, >= 0.95 cluster_id
    h4c_bp = _mutate_per_50bp_block(h4c_b, (10, 35))
    nov1 = make_complex_seq(25, seed=10123)
    x15 = make_complex_seq(15, seed=10124)
    nov2 = make_complex_seq(40, seed=10125)
    h4c_ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("rmid", "+"), ("4", "+"), ("5", "+"), ("6", "+")]
    h4c_ref_nodes = {**base_ref6, "rmid": h4c_a + h4c_b}
    h4c_extra = {"ap": h4c_ap, "bp": h4c_bp, "nov1": nov1, "x15": x15, "nov2": nov2}
    h4c_paths = [
        (f"{h}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("ap", "+"), ("nov1", "+"), ("x15", "+"), ("bp", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for h in ("Hap2a", "Hap2b", "Hap2d")
    ] + [
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("ap", "+"), ("nov1", "+"), ("nov2", "+"), ("bp", "+"), ("4", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_h4c = _run_b1(f"i6_h4c_{mode}", h4c_ref_nodes, h4c_extra, h4c_ref_steps, h4c_paths, mode, dedup_id=0.98)
        ins_evs = [e for e in res_h4c.sv_events if e.event_type == "INS"]
        assert len(ins_evs) == 1
        assert "1:len=3065,src=Hap2c#1,n=1" in ins_evs[0].alleles_str
        assert (nov1 + nov2) in res_h4c.pan_seq

    # (3) H4c-nested-v2 (default dedup_id=0.90, ref & pan)
    nl = make_complex_seq(1500, seed=10131)
    nr = make_complex_seq(1500, seed=10132)
    na1 = make_complex_seq(1000, seed=10133)
    na2 = make_complex_seq(1000, seed=10134)
    na3 = make_complex_seq(1000, seed=10135)
    nx1 = make_complex_seq(20, seed=10136)
    nx2 = make_complex_seq(20, seed=10137)
    nins1 = make_complex_seq(35, seed=10138)
    nins2 = make_complex_seq(35, seed=10139)
    h4cn_nodes = {
        "L": nl, "R": nr, "rmid": na1 + na2 + na3,
        "a1p": mutate_seq_snps(na1, [i * 100 + 20 for i in range(10)]),
        "a2p": mutate_seq_snps(na2, [i * 100 + 20 for i in range(10)]),
        "a3p": mutate_seq_snps(na3, [i * 100 + 20 for i in range(10)]),
        "x1": nx1, "x2": nx2, "ins1": nins1, "ins2": nins2,
    }
    h4cn_paths = [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L", "+"), ("rmid", "+"), ("R", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 6)
    ] + [
        (f"{h}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L", "+"), ("a1p", "+"), ("x1", "+"), ("a2p", "+"), ("x2", "+"), ("a3p", "+"), ("R", "+"), ("5", "+"), ("6", "+")])
        for h in ("Hap2a", "Hap2b", "Hap2d")
    ] + [
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L", "+"), ("a1p", "+"), ("ins1", "+"), ("a2p", "+"), ("ins2", "+"), ("a3p", "+"), ("R", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_h4cn = _run_b1(f"i6_h4cn_{mode}", base_ref6, h4cn_nodes, ref6, h4cn_paths, mode, dedup_id=0.90)
        assert nins1 in res_h4cn.pan_seq and nins2 in res_h4cn.pan_seq
        ev_cn = next(e for e in res_h4cn.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        al_cn = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_cn.alleles_str.split(";")}
        assert "len=6000" in al_cn["Hap1_1#1"] and "n=8" in al_cn["Hap1_1#1"]
        hap2c_bp = sum(
            r["pan_end"] - r["pan_start"]
            for r in res_h4cn.inserts_rows
            if r["event_id"] == ev_cn.event_id and "Hap2c#1" in r["members"].split(",")
        )
        if mode == "ref":
            assert hap2c_bp == 6070 and "len=6070" in al_cn["Hap2c#1"]
        else:
            assert "partial_content=1" in al_cn["Hap2c#1"]
            assert f"len={hap2c_bp}" in al_cn["Hap2c#1"] and hap2c_bp >= 3070

    # (4) Attack 2A-short (top-level len(occ) < len(rep0) carrying XY80, dedup_id=0.99, ref)
    a2a_a = make_complex_seq(1500, seed=10141)
    a2a_d = make_complex_seq(200, seed=10142)
    a2a_c = make_complex_seq(1500, seed=10143)
    a2a_x = make_complex_seq(40, seed=10144)
    a2a_y = make_complex_seq(40, seed=10145)
    a2a_ref_nodes = {**base_ref6, "rmid": a2a_a + a2a_d + a2a_c}
    a2a_extra = {
        "A2": _mutate_per_50bp_block(a2a_a, (20,)),
        "D2": _mutate_per_50bp_block(a2a_d, (20,)),
        "C2": _mutate_per_50bp_block(a2a_c, (20,)),
        "X": a2a_x, "Y": a2a_y,
    }
    a2a_paths = [
        (f"Hap1{ch}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("A2", "+"), ("X", "+"), ("D2", "+"), ("C2", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for ch in ("a", "b")
    ] + [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("A2", "+"), ("X", "+"), ("Y", "+"), ("C2", "+"), ("4", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_2a = _run_b1("i6_atk2a_ref", a2a_ref_nodes, a2a_extra, h4c_ref_steps, a2a_paths, "ref", dedup_id=0.99)
    ins_2a = [e for e in res_2a.sv_events if e.event_type == "INS"]
    assert len(ins_2a) == 1 and "1:len=3080,src=Hap2#1,n=1" in ins_2a[0].alleles_str
    assert (a2a_x + a2a_y) in res_2a.pan_seq

    # (5) Attack 2B (nested pure 200 bp deletion vs child rep0, default dedup_id=0.90, ref & pan)
    b2_l = make_complex_seq(1500, seed=10151)
    b2_r = make_complex_seq(1500, seed=10152)
    b2_d1 = make_complex_seq(400, seed=10153)
    b2_d2 = make_complex_seq(200, seed=10154)
    b2_d3 = make_complex_seq(400, seed=10155)
    b2_s = _mutate_per_50bp_block(b2_d1 + b2_d2 + b2_d3, (25,))
    b2_nodes = {
        "10": b2_l + b2_r,
        "La": b2_l, "Ra": b2_r, "rmid": b2_d1 + b2_d2 + b2_d3,
        "s1": b2_s[:400], "s2": b2_s[400:600], "s3": b2_s[600:],
    }
    b2_paths = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 9)
    ] + [
        (f"Hap0_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("rmid", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        (f"Hap1{ch}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("s1", "+"), ("s2", "+"), ("s3", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for ch in ("a", "b")
    ] + [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("s1", "+"), ("s3", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_2b_ref = _run_b1("i6_atk2b_ref", base_ref6, b2_nodes, ref6, b2_paths, "ref", dedup_id=0.90)
    ev_2br = next(e for e in res_2b_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_2br = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_2br.alleles_str.split(";")}
    hap2_2br_bp = sum(
        r["pan_end"] - r["pan_start"]
        for r in res_2b_ref.inserts_rows
        if r["event_id"] == ev_2br.event_id and "Hap2#1" in r["members"].split(",")
    )
    assert "len=4000" in al_2br["Hap0_1#1"] and "n=6" in al_2br["Hap0_1#1"]
    assert "len=3800" in al_2br["Hap2#1"] and hap2_2br_bp == 3800

    res_2b_pan_lr = _run_b1("i6_atk2b_pan_lr", base_ref6, b2_nodes, ref6, b2_paths, "pan", dedup_id=0.90)
    ev_2bp_lr = next(e for e in res_2b_pan_lr.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2bp_lr.genotypes["Hap0_1#1"] == "1" and ev_2bp_lr.genotypes["Hap2#1"] != "1"
    rmid_rows = [r for r in res_2b_pan_lr.inserts_rows if r["event_id"] == ev_2bp_lr.event_id and "Hap0_1#1" in r["members"].split(",")]
    assert rmid_rows and all("Hap2#1" not in r["members"].split(",") for r in rmid_rows)

    b2_nodes_full = {**b2_nodes, "10": b2_l + b2_s[:400] + b2_s[600:] + b2_r}
    res_2b_pan = _run_b1("i6_atk2b_pan_full", base_ref6, b2_nodes_full, ref6, b2_paths, "pan", dedup_id=0.90)
    ev_2bp = next(e for e in res_2b_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2bp.genotypes["Hap0_1#1"] == "1" and ev_2bp.genotypes["Hap2#1"] == "."
    assert all(
        "Hap2#1" not in r["members"].split(",")
        for r in res_2b_pan.inserts_rows
        if r["event_id"] == ev_2bp.event_id
    )

    # (6) synth_comp sc.seq == rep0_seq (spacer_n=0) split
    sc_l = make_complex_seq(1500, seed=10161)
    sc_r = make_complex_seq(1500, seed=10162)
    sc_c1 = make_complex_seq(1000, seed=10163)
    sc_sub = make_complex_seq(300, seed=10164)
    sc_c2 = make_complex_seq(1000, seed=10165)
    sc_nov1 = make_complex_seq(35, seed=10166)
    sc_nov2 = make_complex_seq(35, seed=10167)
    sc_nodes = {
        "La": sc_l, "Ra": sc_r, "rmid": sc_c1 + sc_sub + sc_c2,
        "c1": sc_c1, "sub": sc_sub, "c2": sc_c2, "nv1": sc_nov1, "nv2": sc_nov2,
    }
    # Inner b_sec = (c1, c2) [Hap1a..1c, support=3], Hap2 = (c1, sub(300), c2) [active child rep -> synth_comp with
    # spacer_n=0 -> sc.seq == c1 + sub + c2 == rmid == rep0_seq], Hap3 = (c1, nv1(35), c2, nv2(35)) [fails merged_occ].
    sc_paths = [
        (f"Hap0_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("rmid", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 6)
    ] + [
        (f"Hap1{ch}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("c1", "+"), ("c2", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for ch in ("a", "b", "c")
    ] + [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("c1", "+"), ("sub", "+"), ("c2", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("c1", "+"), ("nv1", "+"), ("c2", "+"), ("nv2", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_sc = _run_b1("i6_sc_eq_rep0_sp0", base_ref6, sc_nodes, ref6, sc_paths, "ref", dedup_id=0.90, spacer_n=0)
    assert sc_nov1 in res_sc.pan_seq and sc_nov2 in res_sc.pan_seq
    ev_sc = next(e for e in res_sc.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    hap3_sc_bp = sum(
        r["pan_end"] - r["pan_start"]
        for r in res_sc.inserts_rows
        if r["event_id"] == ev_sc.event_id and "Hap3#1" in r["members"].split(",")
    )
    assert hap3_sc_bp == 5070

    # (7) Refinement #1: sub-slice-aware monotonic-cursor _is_step_subsequence
    s5_ref = make_complex_seq(500, seed=10171)
    s6_ref = make_complex_seq(500, seed=10172)
    bsec_sub = _mutate_per_50bp_block(s5_ref + s6_ref, (25,))
    s5_mut, s6_mut = bsec_sub[:500], bsec_sub[500:]
    gb_sub = SyntheticGFAFixtureBuilder()
    for nid, sq in [("1", make_complex_seq(200, seed=10173)), ("2", s5_ref), ("3", s6_ref), ("4", make_complex_seq(200, seed=10174)), ("5", s5_mut), ("6", s6_mut)]:
        gb_sub.add_node(nid, sq)
    ref4 = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    gb_sub.add_p_path("Ref#1#Chr01", ref4)
    g_sub, cfg_sub = _build_graph_and_config(tmp_path, "i6_subseq_cursor", gb_sub, ref4, dedup_mode="ref", dedup_id=0.99)

    def _mk_c(cid: str, hap: str, order: int, steps: tuple[Step, ...], seq: str) -> ExtractedCandidate:
        pos = 200
        provs: list[StepProvenance] = []
        for st in steps:
            sl = st.step_len(g_sub.nodes)  # type: ignore[attr-defined]
            provs.append(StepProvenance(hap, f"{hap}#Chr01", pos, pos + sl, st.orient))
            pos += sl
        return ExtractedCandidate(
            candidate_id=cid, parent_id=None, hap_id=hap, src_path=f"{hap}#Chr01",
            left_anchor_rank=0, right_anchor_rank=3, source_order=order,
            steps=steps, provenance=tuple(provs), seq=seq,
        )

    cb1 = _mk_c("cb1", "B1#1", 0, (Step("5", "+"), Step("6", "+")), bsec_sub)
    cb2 = _mk_c("cb2", "B2#1", 1, (Step("5", "+"), Step("6", "+")), bsec_sub)
    # 7a: Top-level partial-step pure deletion (5+[0:300), 6+) -> absorbed into rep0 without rep0_cand_spans
    cm_part = _mk_c("cm_p", "Mp#1", 2, (Step("5", "+", 0, 300, True), Step("6", "+")), s5_mut[:300] + s6_mut)
    sol_part = solve_parallel_group(0, 3, (Step("2", "+"), Step("3", "+")), [cb1, cb2, cm_part], g_sub, cfg_sub, is_top_level=True)  # type: ignore[arg-type]
    assert sol_part.non_backbone_reps == ()
    assert sol_part.cand_to_rep_idx["cm_p"] == 0 and "cm_p" not in sol_part.rep0_cand_spans
    # 7b: Negative check — overlapping tandem sub-slices (5+[0:300), 5+[200:400), 6+) -> rejected & split out!
    cm_ov = _mk_c(
        "cm_ov", "Mov#1", 2,
        (Step("5", "+", 0, 300, True), Step("5", "+", 200, 400, True), Step("6", "+")),
        s5_mut[:300] + s5_mut[200:400] + s6_mut,
    )
    sol_ov = solve_parallel_group(0, 3, (Step("2", "+"), Step("3", "+")), [cb1, cb2, cm_ov], g_sub, cfg_sub, is_top_level=True)  # type: ignore[arg-type]
    assert len(sol_ov.non_backbone_reps) == 1 and sol_ov.cand_to_rep_idx["cm_ov"] == 1
    # 7c: Negative check — inverted sub-step (5-[0:300), 6+) -> rejected & split out!
    cm_inv = _mk_c(
        "cm_inv", "Minv#1", 2,
        (Step("5", "-", 0, 300, True), Step("6", "+")),
        revcomp_dna(s5_mut)[:300] + s6_mut,
    )
    sol_inv = solve_parallel_group(0, 3, (Step("2", "+"), Step("3", "+")), [cb1, cb2, cm_inv], g_sub, cfg_sub, is_top_level=True)  # type: ignore[arg-type]
    assert len(sol_inv.non_backbone_reps) == 1 and sol_inv.cand_to_rep_idx["cm_inv"] == 1


def test_iter6_bug2_require_full_span_hom_rep(tmp_path: Path) -> None:
    """Iter 6 Bug 2 (`require_full_span` for homogeneous `rep` in `solve_parallel_group`):
    1. `H4` (`ref` & `pan` under both `Sup = A+B` and `Sup = A+m1+B`).
    2. `BUG2` (`ref` & `pan`) and `BUG2b` (`ref` & `pan`).
    3. `H4-PREFIX-80D` (`ref` & `pan`) and `H4-MIXED-60D` (`ref` & `pan`).
    4. `ADV-7-HOM` (single-member `Hap3` vs homogeneous `rep`, `pan`).
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=10200 + i) for i in range(1, 7)}

    def _run_b2(tag: str, extra_nodes: dict[str, str], paths: list[tuple[str, list[tuple[str, str]]]], mode: str):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=0.90, min_sv_len=50)
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(cfg.out_prefix, gfa_path=cfg.gfa, ref_fa_path=cfg.ref_fa, backbone_prefix=cfg.backbone_prefix, spacer_n=cfg.spacer_n)
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    sup8 = [(f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")]) for k in range(1, 9)]

    # (1) H4 (ref, pan Sup=A+B, pan Sup=A+m1+B)
    a = make_complex_seq(1500, seed=10211)
    m1 = make_complex_seq(40, seed=10212)
    m2 = make_complex_seq(40, seed=10213)
    b = make_complex_seq(1500, seed=10214)
    ap = mutate_seq_snps(a, [i * 150 + 20 for i in range(10)])
    bp = mutate_seq_snps(b, [i * 150 + 25 for i in range(10)])
    h4_nodes = {"10": a + b, "rep_n": a + m1 + m2 + b, "ap": ap, "m1": m1, "bp": bp}
    h4_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m1", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m1", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_h4_ref = _run_b2("i6_h4_ref", h4_nodes, h4_paths, "ref")
    ev_h4r = next(e for e in res_h4_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_h4r.genotypes["Hap1_1#1"] != ev_h4r.genotypes["Hap2c#1"]

    res_h4_pan_ab = _run_b2("i6_h4_pan_ab", h4_nodes, h4_paths, "pan")
    ev_h4p = next(e for e in res_h4_pan_ab.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_h4p.genotypes["Hap1_1#1"] == "1" and ev_h4p.genotypes["Hap2c#1"] == "2"
    ins_h4p_rep1 = [r for r in res_h4_pan_ab.inserts_rows if r["event_id"] == ev_h4p.event_id and "Hap1_1#1" in r["members"].split(",")]
    assert ins_h4p_rep1 and all("Hap2c#1" not in r["members"].split(",") for r in ins_h4p_rep1)

    res_h4_pan_am1b = _run_b2("i6_h4_pan_am1b", {**h4_nodes, "10": a + m1 + b}, h4_paths, "pan")
    ev_h4pm = next(e for e in res_h4_pan_am1b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_h4pm.genotypes["Hap1_1#1"] == "1" and ev_h4pm.genotypes["Hap2c#1"] == "."
    assert all("Hap2c#1" not in r["members"].split(",") for r in res_h4_pan_am1b.inserts_rows if r["event_id"] == ev_h4pm.event_id)

    # (2) BUG2 & BUG2b (ref & pan)
    b2_nodes = {"10": a + b, "rep_n": a + m1 + m2 + b, "ap": ap, "m1": mutate_seq_snps(m1, [15]), "bp": bp}
    b2_paths = sup8 + [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m1", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m1", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_b2 = _run_b2(f"i6_bug2_{mode}", b2_nodes, b2_paths, mode)
        ev_b2 = next(e for e in res_b2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert ev_b2.genotypes["Hap1#1"] != ev_b2.genotypes["Hap2c#1"]
        hap1_rows = [r for r in res_b2.inserts_rows if r["event_id"] == ev_b2.event_id and "Hap1#1" in r["members"].split(",")]
        assert hap1_rows and all("Hap2c#1" not in r["members"].split(",") for r in hap1_rows)

    t45 = make_complex_seq(45, seed=10221)
    u45 = make_complex_seq(45, seed=10222)
    b2b_x = make_complex_seq(40, seed=10223)
    b2b_y = make_complex_seq(40, seed=10224)
    b2b_nodes = {
        "rep_n": a + b + t45 + u45,
        "ap": ap, "X": b2b_x, "Y": b2b_y, "bp": bp, "T45_2": mutate_seq_snps(t45, [15]),
    }
    b2b_paths = [
        ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("X", "+"), ("bp", "+"), ("T45_2", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("X", "+"), ("bp", "+"), ("T45_2", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("X", "+"), ("Y", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_b2b = _run_b2(f"i6_bug2b_{mode}", b2b_nodes, b2b_paths, mode)
        ev_b2b = next(e for e in res_b2b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert ev_b2b.genotypes["Hap1#1"] != ev_b2b.genotypes["Hap2c#1"]
        assert b2b_x in res_b2b.pan_seq

    # (3) H4-PREFIX-80D & H4-MIXED-60D (ref & pan)
    p1 = make_complex_seq(40, seed=10231)
    p2 = make_complex_seq(40, seed=10232)
    pref_nodes = {"10": a + b, "rep_n": p1 + p2 + a + b, "p2": p2, "ap": ap, "bp": bp}
    pref_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p2", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("p2", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_pref_ref = _run_b2("i6_h4pref_ref", pref_nodes, pref_paths, "ref")
    ev_pr = next(e for e in res_pref_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_pr.genotypes["Hap1_1#1"] != ev_pr.genotypes["Hap2c#1"]
    res_pref_pan = _run_b2("i6_h4pref_pan", pref_nodes, pref_paths, "pan")
    ev_pp = next(e for e in res_pref_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_pp.genotypes["Hap1_1#1"] == "1" and ev_pp.genotypes["Hap2c#1"] == "."
    assert all("Hap2c#1" not in r["members"].split(",") for r in res_pref_pan.inserts_rows if r["event_id"] == ev_pp.event_id)

    p30 = make_complex_seq(30, seed=10233)
    m30 = make_complex_seq(30, seed=10234)
    mix_nodes = {"10": a + b, "rep_n": p30 + a + m30 + b, "ap": ap, "m30": m30, "bp": bp}
    mix_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap2a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m30", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("m30", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2c#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ap", "+"), ("bp", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode in ("ref", "pan"):
        res_mix = _run_b2(f"i6_h4mix_{mode}", mix_nodes, mix_paths, mode)
        ev_mx = next(e for e in res_mix.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert ev_mx.genotypes["Hap1_1#1"] != ev_mx.genotypes["Hap2c#1"]
        h1_rows = [r for r in res_mix.inserts_rows if r["event_id"] == ev_mx.event_id and "Hap1_1#1" in r["members"].split(",")]
        assert h1_rows and all("Hap2c#1" not in r["members"].split(",") for r in h1_rows)

    # (4) ADV-7-HOM (single-member Hap3 vs homogeneous rep, pan)
    ins25 = make_complex_seq(25, seed=10241)
    adv7h_nodes = {"10": a + b, "rep_n": p30 + a + m30 + b, "h3": ap + bp + ins25}
    adv7h_paths = sup8 + [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("rep_n", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("h3", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_a7h = _run_b2("i6_adv7_hom_pan", adv7h_nodes, adv7h_paths, "pan")
    ev_a7h = next(e for e in res_a7h.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_a7h.genotypes["Hap1_1#1"] == "1" and ev_a7h.genotypes["Hap3#1"] == "."
    assert all("Hap3#1" not in r["members"].split(",") for r in res_a7h.inserts_rows if r["event_id"] == ev_a7h.event_id)


def test_iter6_bug3_r4_2_phases_and_microhomology(tmp_path: Path) -> None:
    """Iter 6 Bug 3 (`R4-2` Phases 0, 1, 2a, 2b in `_solve_multi_member_component`):
    1. `H1c` (`49+49=98` zero-`D`), `H1a` (`147`), `H1b` (`57`), `ATTACK2` (`90`), `ATTACK3` (`72`): distinct allele in `ref`, `Hap2 == "."` in `pan`.
    2. `H2` (`15a(50)+15b(40)`), `ATTACK4` (`u0(100)+u1(40)` at `aL=1535`/`1500`), & `ATTACK4_big` (`u0(100)+u1(55)` at `aL=1535`/`1500`): full novel intact, `dedup_short_residual_bp == 0`, `Hap2 == "."`.
    3. `Topo3a` & `Topo3b`: `Hap2 == "."` & absent from `members` in `pan`.
    4. Real-genome microhomology protection (Phase 0 `+`/`-` strand): `R4_1D` 3 bp microhomology & `gene_A(5000)+TE(500)+gene_B(5000)` 5 bp microhomology + `gene_A(5000, 2bp int D)+TE(500)` Phase 2b non-over-exclusion.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=10300 + i) for i in range(1, 7)}

    def _run_b3(tag: str, extra_nodes: dict[str, str], paths: list[tuple[str, list[tuple[str, str]]]], mode: str):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=0.90, min_sv_len=50)
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(cfg.out_prefix, gfa_path=cfg.gfa, ref_fa_path=cfg.ref_fa, backbone_prefix=cfg.backbone_prefix, spacer_n=cfg.spacer_n)
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    s11 = make_complex_seq(1500, seed=10311)
    s12 = make_complex_seq(1500, seed=10312)
    sup6 = [(f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")]) for k in range(1, 7)]

    def _mk_34_paths(b_mid: list[tuple[str, str]], h2_mid: list[tuple[str, str]]) -> list[tuple[str, list[tuple[str, str]]]]:
        pre = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+")]
        suf = [("12", "+"), ("5", "+"), ("6", "+")]
        return sup6 + [
            ("Hap1a#1#Chr01", pre + b_mid + suf),
            ("Hap1b#1#Chr01", pre + b_mid + suf),
            ("Hap2#1#Chr01", pre + h2_mid + suf),
        ]

    # (1) H1c (49+49=98 zero-D), H1a (147), H1b (57), ATTACK2 (90), ATTACK3 (72)
    n14a = make_complex_seq(500, seed=10321)
    n14b = make_complex_seq(500, seed=10322)
    h1c_alt = n14a[49:] + n14b[:451]
    h1c_nodes = {"10": s11 + h1c_alt + s12, "11": s11, "14a": n14a, "14b": n14b, "12": s12, "alt": h1c_alt}
    h1c_paths = _mk_34_paths([("14a", "+"), ("14b", "+")], [("alt", "+")])

    h1a_s1, h1a_mid, h1a_s2 = make_complex_seq(100, seed=10323), make_complex_seq(500, seed=10324), make_complex_seq(100, seed=10325)
    h1a_alt = h1a_s1[49:] + h1a_mid[:225] + h1a_mid[274:] + h1a_s2[:51]
    h1a_nodes = {"10": s11 + h1a_alt + s12, "11": s11, "s1": h1a_s1, "sm": h1a_mid, "s2": h1a_s2, "12": s12, "alt": h1a_alt}
    h1a_paths = _mk_34_paths([("s1", "+"), ("sm", "+"), ("s2", "+")], [("alt", "+")])

    h1b_s1, h1b_m1, h1b_s2, h1b_m2, h1b_s3 = (
        make_complex_seq(40, seed=10326), make_complex_seq(350, seed=10327),
        make_complex_seq(40, seed=10328), make_complex_seq(350, seed=10329), make_complex_seq(40, seed=10330),
    )
    h1b_alt = h1b_s1[19:] + h1b_m1 + h1b_s2[:10] + h1b_s2[29:] + h1b_m2 + h1b_s3[:21]
    h1b_nodes = {"10": s11 + h1b_alt + s12, "11": s11, "s1": h1b_s1, "m1": h1b_m1, "s2": h1b_s2, "m2": h1b_m2, "s3": h1b_s3, "12": s12, "alt": h1b_alt}
    h1b_paths = _mk_34_paths([("s1", "+"), ("m1", "+"), ("s2", "+"), ("m2", "+"), ("s3", "+")], [("alt", "+")])

    a2_s0, a2_s1 = make_complex_seq(100, seed=10331), make_complex_seq(100, seed=10332)
    a2_alt = a2_s0[45:] + a2_s1[:55]
    a2_nodes = {"10": s11 + a2_alt + s12, "11": s11, "s0": a2_s0, "s1": a2_s1, "12": s12, "alt": a2_alt}
    a2_paths = _mk_34_paths([("s0", "+"), ("s1", "+")], [("alt", "+")])

    a3_t0, a3_t1, a3_t2 = make_complex_seq(49, seed=10333), make_complex_seq(49, seed=10334), make_complex_seq(49, seed=10335)
    a3_alt = a3_t0[24:] + a3_t1[:12] + a3_t1[36:] + a3_t2[:25]
    a3_nodes = {"10": s11 + a3_alt + s12, "11": s11, "t0": a3_t0, "t1": a3_t1, "t2": a3_t2, "12": s12, "alt": a3_alt}
    a3_paths = _mk_34_paths([("t0", "+"), ("t1", "+"), ("t2", "+")], [("alt", "+")])

    for name, nds, pths in [
        ("h1c", h1c_nodes, h1c_paths),
        ("h1a", h1a_nodes, h1a_paths),
        ("h1b", h1b_nodes, h1b_paths),
        ("atk2", a2_nodes, a2_paths),
        ("atk3", a3_nodes, a3_paths),
    ]:
        res_r = _run_b3(f"i6_{name}_ref", nds, pths, "ref")
        ev_r = next(e for e in res_r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert ev_r.genotypes["Hap1a#1"] != ev_r.genotypes["Hap2#1"], f"{name} ref collapsed Hap2"
        res_p = _run_b3(f"i6_{name}_pan", nds, pths, "pan")
        ev_p = next(e for e in res_p.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert ev_p.genotypes["Hap1a#1"] == "1" and ev_p.genotypes["Hap2#1"] == "."
        assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in res_p.inserts_rows if r["event_id"] == ev_p.event_id)

    # (2) H2, ATTACK4 (aL=1535 & 1500), ATTACK4_big (u0(100,ov=65)+u1(55,ov=0) at aL=1535 & 1500)
    h2_14 = make_complex_seq(1000, seed=10341)
    h2_15a = make_complex_seq(50, seed=10342)
    h2_15b = make_complex_seq(40, seed=10343)
    h2_alt = mutate_seq_snps(h2_14, [i * 100 + 20 for i in range(10)]) + h2_15a[:26]
    h2_nodes = {"10": s11 + h2_14 + h2_15a[:26] + s12, "11": s11, "14": h2_14, "15a": h2_15a, "15b": h2_15b, "12": s12, "alt": h2_alt}
    res_h2 = _run_b3("i6_h2_pan", h2_nodes, _mk_34_paths([("14", "+"), ("15a", "+"), ("15b", "+")], [("alt", "+")]), "pan")
    ev_h2 = next(e for e in res_h2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_h2.genotypes["Hap2#1"] == "." and (h2_15a[26:] + h2_15b) in res_h2.pan_seq
    assert int(res_h2.summary_metrics["dedup_short_residual_bp"]) == 0

    u0 = make_complex_seq(100, seed=10344)
    for u1_len, tag_u in ((40, "atk4"), (55, "atk4big")):
        u1 = make_complex_seq(u1_len, seed=10345 + u1_len)
        nov_u = u0[65:] + u1
        for al_len in (1535, 1500):
            al_seq = make_complex_seq(al_len, seed=10346 + al_len)
            alt_u = u0[:65]
            u_nodes = {"10": al_seq + alt_u + s12, "11": al_seq, "u0": u0, "u1": u1, "12": s12, "alt": alt_u}
            res_u = _run_b3(f"i6_{tag_u}_al{al_len}_pan", u_nodes, _mk_34_paths([("u0", "+"), ("u1", "+")], [("alt", "+")]), "pan")
            ev_u = next(e for e in res_u.sv_events if e.event_type == "INS" and e.anchors == "3..4")
            assert ev_u.genotypes["Hap1a#1"] == "1" and ev_u.genotypes["Hap2#1"] == "."
            assert nov_u in res_u.pan_seq
            assert int(res_u.summary_metrics["dedup_short_residual_bp"]) == 0
            assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in res_u.inserts_rows if r["event_id"] == ev_u.event_id)

    # (3) Topo3a (s0(45)+s1(45)+mid(500)+big_del(100)) & Topo3b (s0(35)+mid(500)+big_del(30))
    t3_s0, t3_s1, t3_mid, t3_big100, t3_big30 = (
        make_complex_seq(100, seed=10351), make_complex_seq(100, seed=10352),
        make_complex_seq(500, seed=10353), make_complex_seq(100, seed=10354), make_complex_seq(30, seed=10355),
    )
    t3a_alt = t3_s0[45:] + t3_s1[:55] + t3_mid
    t3a_nodes = {"10": s11 + t3a_alt + t3_big100 + s12, "11": s11, "s0": t3_s0, "s1": t3_s1, "mid": t3_mid, "big": t3_big100, "12": s12, "alt": t3a_alt}
    res_t3a = _run_b3("i6_topo3a_pan", t3a_nodes, _mk_34_paths([("s0", "+"), ("s1", "+"), ("mid", "+"), ("big", "+")], [("alt", "+")]), "pan")
    ev_t3a = next(e for e in res_t3a.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_t3a.genotypes["Hap1a#1"] == "1" and ev_t3a.genotypes["Hap2#1"] == "."
    assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in res_t3a.inserts_rows if r["event_id"] == ev_t3a.event_id)

    t3b_alt = t3_s0[35:] + t3_mid
    t3b_nodes = {"10": s11 + t3b_alt + s12, "11": s11, "s0": t3_s0, "mid": t3_mid, "big": t3_big30, "12": s12, "alt": t3b_alt}
    res_t3b = _run_b3("i6_topo3b_pan", t3b_nodes, _mk_34_paths([("s0", "+"), ("mid", "+"), ("big", "+")], [("alt", "+")]), "pan")
    ev_t3b = next(e for e in res_t3b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_t3b.genotypes["Hap1a#1"] == "1" and ev_t3b.genotypes["Hap2#1"] == "."
    assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in res_t3b.inserts_rows if r["event_id"] == ev_t3b.event_id)

    # (4) Real-genome microhomology protection (Phase 0 '+' and '-' strand + gene_A+TE+gene_B + Phase 2b 2bp int D)
    mh_14 = make_complex_seq(1000, seed=10361)
    mh_m30 = make_complex_seq(27, seed=10362) + mh_14[-3:]  # Forced 3 bp microhomology s14[-3:] == m30[-3:]!
    mh_15 = make_complex_seq(1000, seed=10363)
    mh_t30 = make_complex_seq(30, seed=10364)
    mh_14p = mutate_seq_snps(mh_14, [i * 100 + 20 for i in range(10)])
    mh_15p = mutate_seq_snps(mh_15, [i * 100 + 25 for i in range(10)])
    # 4a: '+' strand R4_1D with 3 bp microhomology (Sup = 11+12) -> Hap2 keeps 2000 bp (14+15), excludes ONLY m30+t30
    mh_plus_nodes = {"10": s11 + s12, "11": s11, "14": mh_14, "m30": mh_m30, "15": mh_15, "t30": mh_t30, "12": s12, "20": mh_14p + mh_15p}
    res_mhp = _run_b3("i6_mh3_plus_pan", mh_plus_nodes, _mk_34_paths([("14", "+"), ("m30", "+"), ("15", "+"), ("t30", "+")], [("20", "+")]), "pan")
    ev_mhp = next(e for e in res_mhp.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_mhp = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_mhp.alleles_str.split(";")}
    assert "len=2060" in al_mhp["Hap1a#1"] and "len=2000" in al_mhp["Hap2#1"]

    # 4b: '-' strand R4_1D with 3 bp microhomology (steps oriented '-') -> Hap2 still keeps 2000 bp (14-+15-)
    mh_minus_nodes = {
        "10": s11 + s12, "11": s11, "12": s12,
        "14r": revcomp_dna(mh_14), "m30r": revcomp_dna(mh_m30),
        "15r": revcomp_dna(mh_15), "t30r": revcomp_dna(mh_t30),
        "20": mh_14p + mh_15p,
    }
    res_mhm = _run_b3("i6_mh3_minus_pan", mh_minus_nodes, _mk_34_paths([("14r", "-"), ("m30r", "-"), ("15r", "-"), ("t30r", "-")], [("20", "+")]), "pan")
    ev_mhm = next(e for e in res_mhm.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    al_mhm = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_mhm.alleles_str.split(";")}
    assert "len=2060" in al_mhm["Hap1a#1"] and "len=2000" in al_mhm["Hap2#1"]

    # 4c: gene_A(5000) + TE40(40, 5 bp microhomology) + gene_B(5000) + t30(30) -> Phase 0 snaps 5 bp left-shift,
    # keeping all 10,000 bp (gA+gB) for Hap2; AND gene_A(5000, 2bp int D) + TE(500) + shared gB(5000) -> Phase 2b
    # keeps gene_A(5000) + gB(5000) = 10,000 bp for Hap3 (never over-excludes 5 kb gene!).
    gene_a = make_complex_seq(5000, seed=10371)
    te40_mh5 = make_complex_seq(35, seed=10372) + gene_a[-5:]  # Forced 5 bp microhomology!
    te500 = make_complex_seq(500, seed=10374)
    gene_b = make_complex_seq(5000, seed=10373)
    ga_mut = mutate_seq_snps(gene_a, [i * 250 + 30 for i in range(18)])
    gb_mut = mutate_seq_snps(gene_b, [i * 250 + 30 for i in range(18)])
    ga_d2 = ga_mut[:2500] + ga_mut[2502:]  # Benign 2 bp internal D alongside separate 500 bp TE deletion
    gtb_mh_nodes = {
        "10": s11 + s12, "11": s11, "gA": gene_a, "TE40": te40_mh5, "gB": gene_b, "t30": mh_t30,
        "12": s12, "alt_mh": ga_mut + gb_mut,
    }
    gtb_mh_paths = _mk_34_paths([("gA", "+"), ("TE40", "+"), ("gB", "+"), ("t30", "+")], [("alt_mh", "+")])
    res_gtb_mh = _run_b3("i6_geneA_te40_geneB_mh5", gtb_mh_nodes, gtb_mh_paths, "pan")
    ev_gtb_mh = next(e for e in res_gtb_mh.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    h2_mh_bp = sum(r["pan_end"] - r["pan_start"] for r in res_gtb_mh.inserts_rows if r["event_id"] == ev_gtb_mh.event_id and "Hap2#1" in r["members"].split(","))
    assert h2_mh_bp == 10000, f"Hap2#1 falsely lost gene_A under 5 bp microhomology: got {h2_mh_bp}"

    gtb_d2_nodes = {
        "10": s11 + s12, "11": s11, "gA": gene_a, "TE500": te500, "gB": gene_b, "12": s12, "alt_ga_d2": ga_d2,
    }
    gtb_d2_paths = _mk_34_paths([("gA", "+"), ("TE500", "+"), ("gB", "+")], [("alt_ga_d2", "+"), ("gB", "+")])
    res_gtb_d2 = _run_b3("i6_geneA_2bpD_te500", gtb_d2_nodes, gtb_d2_paths, "pan")
    ev_gtb_d2 = next(e for e in res_gtb_d2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    h2_d2_bp = sum(r["pan_end"] - r["pan_start"] for r in res_gtb_d2.inserts_rows if r["event_id"] == ev_gtb_d2.event_id and "Hap2#1" in r["members"].split(","))
    assert h2_d2_bp == 10000, f"Hap2#1 falsely lost gene_A under 2 bp internal D + 500 bp TE del: got {h2_d2_bp}"


def test_iter6_bug4_option4a_and_zero_rec_len_guard(tmp_path: Path) -> None:
    """Iter 6 Bug 4 (Option 4A deterministic exact `reconstructed_len`) + Refinement #2 (`kc_rec_len <= 0` Stage 6c guard):
    1. `fuzz_1_fwd == fuzz_1_rev` (`ref` & `pan`, plus without `HapC`): exact `1:len=2200` matching `inserts.tsv`.
    2. `T2_with_Hap3` rename-invariance (`AHap2`/`ZHap3` vs `ZHap2`/`AHap3`, `ref`): exact `2:len=3000`.
    3. Probe A4 rename-invariance (`Hap2`/`Hap3` swapped, `ref`): exact `1:len=3000`.
    4. Transitive-UF `kc_rec_len == 0` (`PROBE_3C_TRUE_ZERO` in `ref/0.90`, `pan/0.90`, `pan/0.99`): NO `len=0,` allele, `gt[Hap3] == "."`.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=10400 + i) for i in range(1, 7)}

    def _run_b4(tag: str, extra_nodes: dict[str, str], paths: list[tuple[str, list[tuple[str, str]]]], mode: str, dedup_id: float = 0.90):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=dedup_id, min_sv_len=50)
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(cfg.out_prefix, gfa_path=cfg.gfa, ref_fa_path=cfg.ref_fa, backbone_prefix=cfg.backbone_prefix, spacer_n=cfg.spacer_n)
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    # (1) fuzz_1_fwd == fuzz_1_rev (ref & pan) + without HapC
    fa1 = make_complex_seq(800, seed=10411)
    fb1 = make_complex_seq(800, seed=10412)
    fd1 = make_complex_seq(300, seed=10413)
    fe1 = make_complex_seq(600, seed=10414)
    fs1 = make_complex_seq(25, seed=10415)
    fs2 = make_complex_seq(30, seed=10416)
    f_nodes = {"a1": fa1, "b1": fb1, "d1": fd1, "e1": fe1, "s1": fs1, "s2": fs2}
    for mode in ("ref", "pan"):
        lens_by_order: dict[str, str] = {}
        for ord_tag, names in (("fwd", ("HapA", "HapB", "HapC", "HapD")), ("rev", ("HapZ", "HapY", "HapX", "HapW"))):
            h_a, h_b, h_c, h_d = names
            pths = [
                (f"{h_d}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("b1", "+"), ("d1", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
                (f"{h_a}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("s1", "+"), ("b1", "+"), ("s2", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
                (f"{h_b}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("s1", "+"), ("b1", "+"), ("s2", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
                (f"{h_c}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("b1", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
            ]
            res_f = _run_b4(f"i6_fuzz1_{ord_tag}_{mode}", f_nodes, pths, mode)
            ev_f = next(e for e in res_f.sv_events if e.event_type == "INS" and e.anchors == "3..4")
            al_f = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_f.alleles_str.split(";")}
            first_tok = ev_f.alleles_str.split(";")[0]
            assert "len=2200" in first_tok and "n=3" in first_tok
            assert "len=2500" in al_f[f"{h_d}#1"]
            ins_a = sum(r["pan_end"] - r["pan_start"] for r in res_f.inserts_rows if r["event_id"] == ev_f.event_id and f"{h_a}#1" in r["members"].split(","))
            assert ins_a == 2200
            lens_by_order[ord_tag] = first_tok.split(",")[0]
        assert lens_by_order["fwd"] == lens_by_order["rev"] == "1:len=2200"

    # fuzz_1 without HapC -> still exact len=2200 (0 bp gap vs inserts.tsv)
    pths_noc = [
        ("HapD#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("b1", "+"), ("d1", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
        ("HapA#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("s1", "+"), ("b1", "+"), ("s2", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
        ("HapB#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("a1", "+"), ("s1", "+"), ("b1", "+"), ("s2", "+"), ("e1", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_noc = _run_b4("i6_fuzz1_noc_ref", f_nodes, pths_noc, "ref")
    ev_noc = next(e for e in res_noc.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert "1:len=2200,src=HapA#1,n=2" in ev_noc.alleles_str

    # (2) T2_with_Hap3 rename-invariance (AHap2/ZHap3 vs ZHap2/AHap3)
    s11 = make_complex_seq(1500, seed=10421)
    s14 = make_complex_seq(1000, seed=10422)
    s12 = make_complex_seq(1500, seed=10423)
    t2h3_nodes = {"11": s11, "14": s14, "14_940": s14[:940], "12": s12}
    for ord_tag, (h2_n, h3_n) in (("fwd", ("AHap2", "ZHap3")), ("rev", ("ZHap2", "AHap3"))):
        t2h3_paths = [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            (f"{h2_n}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14_940", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            (f"{h3_n}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        res_t2 = _run_b4(f"i6_t2h3_{ord_tag}", t2h3_nodes, t2h3_paths, "ref")
        ev_t2 = next(e for e in res_t2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        al_toks = ev_t2.alleles_str.split(";")
        assert "len=4000" in al_toks[0] and "len=3000" in al_toks[1] and "n=2" in al_toks[1]

    # (3) Probe A4 rename-invariance (Hap2/Hap3 swapped -> still len=3000)
    short30 = make_complex_seq(30, seed=10431)
    unit15 = make_complex_seq(15, seed=10432)
    a4_nodes = {"11": s11, "14": s14, "12": s12, "17": short30, "18": unit15}
    for ord_tag, (h2_n, h3_n) in (("fwd", ("AHap2", "ZHap3")), ("rev", ("ZHap2", "AHap3"))):
        a4_paths = [
            ("Hap1#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("14", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            (f"{h2_n}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("17", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            (f"{h3_n}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("18", "+"), ("18", "+"), ("18", "+"), ("18", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        res_a4 = _run_b4(f"i6_a4_inv_{ord_tag}", a4_nodes, a4_paths, "ref")
        ev_a4 = next(e for e in res_a4.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert "1:len=3000," in ev_a4.alleles_str and "n=2" in ev_a4.alleles_str.split(";")[0]

    # (4) Transitive-UF kc_rec_len == 0 (PROBE_3C_TRUE_ZERO, Refinement #2)
    p3_13 = make_complex_seq(500, seed=10441)
    p3_14 = make_complex_seq(1000, seed=10442)
    p3_nodes = {"13": p3_13, "14": p3_14, "15": p3_14[:200]}
    p3_paths = [
        (f"Hap1_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("13", "+"), ("14", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("14", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("15", "+"), ("5", "+"), ("6", "+")]),
    ]
    for mode, did in (("ref", 0.90), ("pan", 0.90), ("pan", 0.99)):
        res_p3 = _run_b4(f"i6_probe3c_{mode}_{int(did * 100)}", p3_nodes, p3_paths, mode, dedup_id=did)
        ev_p3 = next(e for e in res_p3.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        assert "len=0," not in ev_p3.alleles_str
        assert ev_p3.genotypes["Hap3#1"] == "."
        assert all("Hap3#1" not in r["members"].split(",") for r in res_p3.inserts_rows if r["event_id"] == ev_p3.event_id)
        if mode == "ref":
            assert "1:len=1500,src=Hap1_1#1,n=4;2:len=1200,src=Hap2#1,n=1" == ev_p3.alleles_str
        else:
            assert "1:len=1500,src=Hap1_1#1,n=4;2:len=1000,src=Hap2#1,n=1,partial_content=1" == ev_p3.alleles_str


def test_iter6_repeat_guard_step2b_reorder_and_perm_invariance() -> None:
    """Iter 6 Edit 6 (`panflat/repeat_guard.py` Step 2b before Witness-Survival Re-Check):
    1. `Attack A` (`m1=[80,140), s1=[40,140), s2=[0,260)` with `protected_ins=((100,200),)`): keeps `[m1, s2]`.
    2. `STEP2B_WITNESS_CHECK_TIE` & `STEP2B_WITNESS_CHECK_BIG`: keeps `[mu_td, su2_clean]` / `[mu_td, su2_clean_big]`.
    3. 3-step Step 2b chain (`s1 ⊂ s2 ⊂ s3`): rescues `m1`, keeps `[m1, s3]`.
    4. Exhaustive `4! x 2!` permutation invariance across main & short unit orderings.
    """
    def _mk_unit(
        q_st: int,
        q_en: int,
        r_st: int,
        r_en: int,
        pins: tuple[tuple[int, int], ...] = (),
        mx_ivs: tuple[tuple[int, int], ...] | None = None,
        tag_piece: int = 0,
    ) -> _HitPieceEvalUnit:
        if mx_ivs is None:
            mx_ivs = ((q_st, q_en),)
        tot_mx = sum(e - s for s, e in mx_ivs)
        meta = _HitPieceUnit(
            ctg="ctg0",
            strand=1,
            piece_key=tag_piece,
            r_st=r_st,
            r_en=r_en,
            q_st=q_st,
            q_en=q_en,
            protected_ins=pins,
            mx_intervals=mx_ivs,
            total_mx=tot_mx,
        )
        return _HitPieceEvalUnit(meta=meta, runs=(), cov_spans=((q_st, q_en),))

    # (1) Attack A: s1 drops m1 in Step 1b, s2 drops s1 in Step 2b, Witness-Survival rescues m1 -> [m1, s2]
    m1_a = _mk_unit(80, 140, 1080, 1140)
    s1_a = _mk_unit(40, 140, 1040, 1140)
    s2_a = _mk_unit(0, 260, 1000, 1160, pins=((100, 200),), mx_ivs=((0, 100), (200, 260)))
    kept_a = _filter_cross_aligner_shadow_units([m1_a], [s1_a, s2_a])
    assert {(u.meta.q_st, u.meta.q_en, u.meta.protected_ins) for u in kept_a} == {
        (80, 140, ()),
        (0, 260, ((100, 200),)),
    }

    # (2) STEP2B_WITNESS_CHECK_TIE & STEP2B_WITNESS_CHECK_BIG
    mu_td = _mk_unit(100, 150, 1100, 1150)
    su1_shat = _mk_unit(0, 200, 1000, 1150, pins=(), mx_ivs=((0, 100), (150, 200)))
    su2_tie = _mk_unit(0, 200, 1000, 1150, pins=((100, 150),), mx_ivs=((0, 100), (150, 200)))
    su2_big = _mk_unit(0, 250, 1000, 1200, pins=((100, 150),), mx_ivs=((0, 100), (150, 250)))
    kept_tie = _filter_cross_aligner_shadow_units([mu_td], [su1_shat, su2_tie])
    assert {(u.meta.q_st, u.meta.q_en, u.meta.protected_ins) for u in kept_tie} == {
        (100, 150, ()),
        (0, 200, ((100, 150),)),
    }
    kept_big = _filter_cross_aligner_shadow_units([mu_td], [su1_shat, su2_big])
    assert {(u.meta.q_st, u.meta.q_en, u.meta.protected_ins) for u in kept_big} == {
        (100, 150, ()),
        (0, 250, ((100, 150),)),
    }

    # (3) 3-step Step 2b chain (s1 ⊂ s2 ⊂ s3) -> m1 rescued, s1 & s2 dropped -> [m1, s3]
    m1_c = _mk_unit(80, 140, 1080, 1140)
    s1_c = _mk_unit(60, 140, 1060, 1140)
    s2_c = _mk_unit(40, 140, 1040, 1140)
    s3_c = _mk_unit(0, 260, 1000, 1160, pins=((100, 200),), mx_ivs=((0, 100), (200, 260)))
    kept_c = _filter_cross_aligner_shadow_units([m1_c], [s1_c, s2_c, s3_c])
    assert {(u.meta.q_st, u.meta.q_en, u.meta.protected_ins) for u in kept_c} == {
        (80, 140, ()),
        (0, 260, ((100, 200),)),
    }

    # (4) 4! x 2! = 48 exhaustive permutation invariance
    m2_c = _mk_unit(20, 90, 1020, 1090)
    s_extra = _mk_unit(210, 255, 1110, 1155)
    mains_perm = [m1_c, m2_c]
    shorts_perm = [s1_c, s2_c, s3_c, s_extra]
    ref_out: frozenset[tuple[int, int, tuple[tuple[int, int], ...]]] | None = None
    for m_ord in itertools.permutations(mains_perm):
        for s_ord in itertools.permutations(shorts_perm):
            res_p = _filter_cross_aligner_shadow_units(list(m_ord), list(s_ord))
            sig = frozenset((u.meta.q_st, u.meta.q_en, u.meta.protected_ins) for u in res_p)
            if ref_out is None:
                ref_out = sig
            else:
                assert sig == ref_out
    assert ref_out == frozenset({(80, 140, ()), (0, 260, ((100, 200),))})


def test_iter7_finding1_zero_rec_len_members_priority_sort_and_nested(
    tmp_path: Path,
) -> None:
    """Iter 7 Finding 1 (Edits 3/5/6/8):
    1. Probe 1 (ref & pan): childless real_rep with len(set(piece_cand_ids)) == 1 excludes 0-piece Hap3.
    2. Probe 1C (pan): Stage 5.5 trimmed Probe 1 still excludes Hap3 from surviving piece members.
    3. Probe 1D & PROBE_3C_PRIO (pan): active support_hap_count preserves Sup6 priority for real & synth reps.
    4. Probe 5 & Probe C (ref & pan): multi-cand contig across 2 reps & multi-occ same-hap inside same rep.
    5. Probe A (3-level nested, ref & pan): exact 5400 bp in ref and 0 bp / gt='.' in pan.
    6. Probe B (ref & pan): intra-group L3457 sort mounts 5-real repB before 4-real+2-excluded repA.
    7. Probe D (pan) & Probe E (ref & pan): Stage 5.5 drop La -> HapX='.'; Step 4b merge into Probe 1.
    8. Probe H (pan): 2-level synth rep with 9 real + 2 excluded yields priority to 10-real Sup10.
    9. Sub-Slicing x Finding 1 (24x2 bp shred collapsing HapZ to kc_rec_len==0 top-level + nested).
    """
    ref8 = [(str(i), "+") for i in range(1, 9)]
    base_ref8 = {str(i): make_complex_seq(400, seed=11000 + i) for i in range(1, 9)}

    def _run_f1(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str,
        dedup_id: float = 0.90,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref8.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref8)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path,
            tag,
            gb,
            ref8,
            dedup_mode=mode,
            dedup_id=dedup_id,
            min_sv_len=50,
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _ins34(mid: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        return [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), *mid, ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")]

    s13 = make_complex_seq(500, seed=11011)
    s14 = make_complex_seq(1000, seed=11012)
    s14a = s14[:800]
    s15 = s14[800:]
    p1_nodes = {"13": s13, "14": s14, "14a": s14a, "15": s15}
    p1_paths = [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("13", "+"), ("14a", "+"), ("15", "+")])),
        ("Hap3#1#Chr01", _ins34([("15", "+")])),
    ]

    # (1) Probe 1 (ref & pan)
    for mode in ("ref", "pan"):
        res = _run_f1(f"i7_probe1_{mode}", p1_nodes, p1_paths, mode)
        ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins = [r for r in res.inserts_rows if r["event_id"] == ev.event_id]
        assert ev.genotypes["Hap3#1"] == "."
        assert "1:len=1500,src=Hap1_1#1,n=5" == ev.alleles_str
        assert all("Hap3#1" not in r["members"].split(",") for r in ins)
        assert all(len(r["members"].split(",")) == 5 for r in ins)

    # (2) Probe 1C (Stage 5.5 trimmed Probe 1 -> Sup at 0..1 trims s13, leaves s14)
    sup6_01 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("sup0", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")])
        for k in range(1, 7)
    ]
    res_1c = _run_f1("i7_probe1c_pan", {**p1_nodes, "sup0": s13}, sup6_01 + p1_paths, "pan")
    ev_1c = next(e for e in res_1c.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_1c = [r for r in res_1c.inserts_rows if r["event_id"] == ev_1c.event_id]
    assert ev_1c.genotypes["Hap3#1"] == "."
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_1c) == 1000
    assert all("Hap3#1" not in r["members"].split(",") for r in ins_1c)

    # (3) Probe 1D (real_rep 5 real + 1 excluded at 3..4 vs 6 real Sup at 5..6 -> Sup6 at 5..6 survives!)
    sup6_56 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("sup5", "+"), ("7", "+"), ("8", "+")])
        for k in range(1, 7)
    ]
    res_1d = _run_f1("i7_probe1d_pan", {**p1_nodes, "sup5": s13 + s14}, sup6_56 + p1_paths, "pan")
    evs_1d = {e.anchors: e for e in res_1d.sv_events if e.event_type == "INS"}
    assert "5..6" in evs_1d and "3..4" not in evs_1d

    # PROBE_3C_PRIO (synth_rep 5 real + 1 excluded at 3..4 vs 6 real Sup at 5..6 -> Sup6 at 5..6 survives!)
    p3c_nodes = {"13": s13, "14": s14, "15": s14[:200], "sup5": s13 + s14}
    p3c_paths = sup6_56 + [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("14", "+"), ("15", "+")])),
        ("Hap3#1#Chr01", _ins34([("15", "+")])),
    ]
    res_3cp = _run_f1("i7_probe3c_prio_pan", p3c_nodes, p3c_paths, "pan")
    evs_3cp = {e.anchors: e for e in res_3cp.sv_events if e.event_type == "INS"}
    assert "5..6" in evs_3cp and "3..4" not in evs_3cp

    # (4) Probe 5 (multi-cand contig: Hap3 has 15 [rec_len=0 in rep1] + ref(2+) + nov600 [rep2])
    nov600 = make_complex_seq(600, seed=11015)
    p5_nodes = {**p1_nodes, "nov": nov600}
    p5_paths = [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("13", "+"), ("14a", "+"), ("15", "+")])),
        ("Hap3#1#Chr01", _ins34([("15", "+"), ("2", "+"), ("nov", "+")])),
    ]
    for mode in ("ref", "pan"):
        res_p5 = _run_f1(f"i7_probe5_{mode}", p5_nodes, p5_paths, mode)
        ev_p5 = next(e for e in res_p5.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_p5 = [r for r in res_p5.inserts_rows if r["event_id"] == ev_p5.event_id]
        assert ev_p5.genotypes["Hap3#1"] == "2"
        rep1_rows = [r for r in ins_p5 if r["pan_end"] - r["pan_start"] == 1500]
        rep2_rows = [r for r in ins_p5 if r["pan_end"] - r["pan_start"] == 600]
        assert rep1_rows and all("Hap3#1" not in r["members"].split(",") for r in rep1_rows)
        assert rep2_rows and all(r["members"] == "Hap3#1" for r in rep2_rows)

    # Probe C (multi-occ same hap in SAME rep: Hap3 = 13+14 [1500] + 2+ + 15 [0])
    pc_paths = [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("13", "+"), ("14a", "+"), ("15", "+")])),
        ("Hap3#1#Chr01", _ins34([("13", "+"), ("14", "+"), ("2", "+"), ("15", "+")])),
    ]
    for mode in ("ref", "pan"):
        res_pc = _run_f1(f"i7_probec_{mode}", p1_nodes, pc_paths, mode)
        ev_pc = next(e for e in res_pc.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_pc = [r for r in res_pc.inserts_rows if r["event_id"] == ev_pc.event_id]
        assert ev_pc.genotypes["Hap3#1"] == "1"
        assert "n=6" in ev_pc.alleles_str and "partial_content=1" in ev_pc.alleles_str
        assert all(r["members"].split(",").count("Hap3#1") == 1 for r in ins_pc)

    # (5) Probe A (3-level nested: Outer[Oa+Ox+Ob] -> Mid[Ma+Mx+Mb] -> Inner[Probe 1])
    oa = make_complex_seq(1500, seed=11021)
    ox = make_complex_seq(2000, seed=11022)
    ob = make_complex_seq(1500, seed=11023)
    ma = make_complex_seq(1200, seed=11024)
    mx = make_complex_seq(1800, seed=11025)
    mb = make_complex_seq(1200, seed=11026)
    pa_nodes = {
        "Oa": oa, "Ox": ox, "Ob": ob, "Ma": ma, "Mx": mx, "Mb": mb,
        "13": s13, "14": s14, "14a": s14a, "15": s15,
        "sup_ab": oa + ob + ma + mb,
    }
    pa_paths_base = [
        (f"HapO{k}#1#Chr01", _ins34([("Oa", "+"), ("Ox", "+"), ("Ob", "+")]))
        for k in range(1, 7)
    ] + [
        (f"HapM{k}#1#Chr01", _ins34([("Oa", "+"), ("Ma", "+"), ("Mx", "+"), ("Mb", "+"), ("Ob", "+")]))
        for k in range(1, 6)
    ] + [
        (f"HapI{k}#1#Chr01", _ins34([("Oa", "+"), ("Ma", "+"), ("13", "+"), ("14", "+"), ("Mb", "+"), ("Ob", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("Oa", "+"), ("Ma", "+"), ("13", "+"), ("14a", "+"), ("15", "+"), ("Mb", "+"), ("Ob", "+")])),
        ("Hap3#1#Chr01", _ins34([("Oa", "+"), ("Ma", "+"), ("15", "+"), ("Mb", "+"), ("Ob", "+")])),
    ]
    res_pa_ref = _run_f1("i7_probea_ref", pa_nodes, pa_paths_base, "ref")
    ev_par = next(e for e in res_pa_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_par = [r for r in res_pa_ref.inserts_rows if r["event_id"] == ev_par.event_id]
    al_par = {tok.split("src=")[1].split(",")[0]: tok for tok in ev_par.alleles_str.split(";")}
    assert "len=5400" in al_par["Hap3#1"]
    h3_par_bp = sum(r["pan_end"] - r["pan_start"] for r in ins_par if "Hap3#1" in r["members"].split(","))
    assert h3_par_bp == 5400

    sup19_01 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("sup_ab", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")])
        for k in range(1, 20)
    ]
    res_pa_pan = _run_f1("i7_probea_pan", pa_nodes, sup19_01 + pa_paths_base, "pan")
    ev_pap = next(e for e in res_pa_pan.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_pap = [r for r in res_pa_pan.inserts_rows if r["event_id"] == ev_pap.event_id]
    assert ev_pap.genotypes["Hap3#1"] == "."
    assert all("Hap3#1" not in r["members"].split(",") for r in ins_pap)

    # (6) Probe B (intra-group L3457 sort: 5-real repB mounted before 4-real+2-excluded repA)
    nov_b = make_complex_seq(1200, seed=11031)
    pb_nodes = {**p1_nodes, "novB": nov_b}
    pb_paths = [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 4)
    ] + [
        ("Hap2#1#Chr01", _ins34([("13", "+"), ("14a", "+"), ("15", "+")])),
        ("HapZ1#1#Chr01", _ins34([("15", "+")])),
        ("HapZ2#1#Chr01", _ins34([("15", "+")])),
    ] + [
        (f"HapB_{k}#1#Chr01", _ins34([("novB", "+")]))
        for k in range(1, 6)
    ]
    for mode in ("ref", "pan"):
        res_pb = _run_f1(f"i7_probeb_{mode}", pb_nodes, pb_paths, mode)
        ev_pb = next(e for e in res_pb.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_pb = [r for r in res_pb.inserts_rows if r["event_id"] == ev_pb.event_id]
        assert len(ins_pb) == 2
        assert ins_pb[0]["pan_end"] - ins_pb[0]["pan_start"] == 1200
        assert ins_pb[1]["pan_end"] - ins_pb[1]["pan_start"] == 1500
        assert ev_pb.genotypes["HapZ1#1"] == "." and ev_pb.genotypes["HapZ2#1"] == "."
        assert all(
            "HapZ1#1" not in r["members"].split(",") and "HapZ2#1" not in r["members"].split(",")
            for r in ins_pb
        )

    # (7) Probe D (Stage 5.5 drops La owned by HapX/HapY1/HapY2, keeps NovY owned by HapY1/HapY2)
    la_d = make_complex_seq(1500, seed=11041)
    nov_y = make_complex_seq(1000, seed=11042)
    pd_nodes = {"sup0": la_d, "La": la_d, "NovY": nov_y}
    pd_paths = sup6_01 + [
        ("HapY1#1#Chr01", _ins34([("La", "+"), ("NovY", "+")])),
        ("HapY2#1#Chr01", _ins34([("La", "+"), ("NovY", "+")])),
        ("HapX#1#Chr01", _ins34([("La", "+")])),
    ]
    res_pd = _run_f1("i7_probed_pan", pd_nodes, pd_paths, "pan")
    ev_pd = next(e for e in res_pd.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_pd = [r for r in res_pd.inserts_rows if r["event_id"] == ev_pd.event_id]
    assert ev_pd.genotypes["HapX#1"] == "." and ev_pd.genotypes["HapY1#1"] == "1" and ev_pd.genotypes["HapY2#1"] == "1"
    assert len(ins_pd) == 1 and ins_pd[0]["members"] == "HapY1#1,HapY2#1"

    # Probe E (Step 4b merge into Probe 1: Hap4=mut1314 merges into Probe 1, Hap3 still excluded)
    mut1314 = mutate_seq_snps(s13 + s14, [i * 150 + 25 for i in range(10)])
    pe_paths = p1_paths + [("Hap4#1#Chr01", _ins34([("mut1314", "+")]))]
    for mode in ("ref", "pan"):
        res_pe = _run_f1(f"i7_probee_{mode}", {**p1_nodes, "mut1314": mut1314}, pe_paths, mode)
        ev_pe = next(e for e in res_pe.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_pe = [r for r in res_pe.inserts_rows if r["event_id"] == ev_pe.event_id]
        assert ev_pe.genotypes["Hap4#1"] == "1" and ev_pe.genotypes["Hap3#1"] == "."
        assert "n=6" in ev_pe.alleles_str
        assert all("Hap3#1" not in r["members"].split(",") and len(r["members"].split(",")) == 6 for r in ins_pe)

    # (8) Probe H (synth rep with 9 real + 2 excluded at 3..4 vs 10 real Sup10 at 5..6 -> Sup10 survives!)
    nov_ch = make_complex_seq(100, seed=11051)
    sup10_56 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("sup5", "+"), ("7", "+"), ("8", "+")])
        for k in range(1, 11)
    ]
    ph_nodes = {"13": s13, "14": s14, "15": s14[:200], "nc": nov_ch, "sup5": s13 + s14}
    ph_paths = sup10_56 + [
        (f"Hap1_{k}#1#Chr01", _ins34([("13", "+"), ("14", "+")]))
        for k in range(1, 8)
    ] + [
        ("Hap2a#1#Chr01", _ins34([("14", "+"), ("15", "+"), ("nc", "+")])),
        ("Hap2b#1#Chr01", _ins34([("14", "+"), ("15", "+"), ("nc", "+")])),
        ("HapZ1#1#Chr01", _ins34([("15", "+")])),
        ("HapZ2#1#Chr01", _ins34([("15", "+")])),
    ]
    res_ph = _run_f1("i7_probeh_pan", ph_nodes, ph_paths, "pan")
    evs_ph = {e.anchors: e for e in res_ph.sv_events if e.event_type == "INS"}
    assert "5..6" in evs_ph
    ev_ph56 = evs_ph["5..6"]
    assert sum(r["pan_end"] - r["pan_start"] for r in res_ph.inserts_rows if r["event_id"] == ev_ph56.event_id) == 1500
    if "3..4" in evs_ph:
        assert evs_ph["3..4"].genotypes["HapZ1#1"] == "." and evs_ph["3..4"].genotypes["HapZ2#1"] == "."

    # (9) Sub-Slicing x Finding 1 (24x2 bp shred + 4 bp trim collapsing HapZ to kc_rec_len == 0 top-level + Sup6 + nested)
    sh_s1 = make_complex_seq(500, seed=11061)
    sh_s2 = make_complex_seq(500, seed=11062)
    sh_full = sh_s1 + sh_s2
    sh_del_starts = [41 * i + 20 for i in range(24)]  # 24 x 2 bp = 48 bp internal (< 50) + 4 bp trim = 52 bp >= 50; every covered island <= 39 bp < 50
    sh_parts: list[str] = []
    cur_p = 0
    for ds in sh_del_starts:
        sh_parts.append(sh_full[cur_p:ds])
        cur_p = ds + 2
    sh_parts.append(sh_full[cur_p:965])
    shred_prefix = "".join(sh_parts)
    br40 = sh_s2[465:496]  # 31 bp bridge (< min_sv_len=50); 4 bp tail sh_s2[496:500] trimmed in HapZ
    sh_nodes = {
        "S1": sh_s1, "S2": sh_s2, "S2a": sh_s2[:465], "B40": br40, "S2t": sh_s2[496:],
        "Shred": shred_prefix, "sup5": sh_full,
        "Oa": oa, "Ox": ox, "Ob": ob,
    }
    sh_top_paths = [
        (f"Hap1_{k}#1#Chr01", _ins34([("S1", "+"), ("S2", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("S1", "+"), ("S2a", "+"), ("B40", "+"), ("S2t", "+")])),
        ("HapZ#1#Chr01", _ins34([("Shred", "+"), ("B40", "+")])),
    ]
    res_sh_ref = _run_f1("i7_shred_top_ref", sh_nodes, sh_top_paths, "ref")
    ev_shr = next(e for e in res_sh_ref.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_shr = [r for r in res_sh_ref.inserts_rows if r["event_id"] == ev_shr.event_id]
    assert ev_shr.genotypes["HapZ#1"] == "."
    assert "len=0," not in ev_shr.alleles_str and "n=5" in ev_shr.alleles_str
    assert all("HapZ#1" not in r["members"].split(",") for r in ins_shr)

    res_sh_prio = _run_f1("i7_shred_prio_pan", sh_nodes, sup6_56 + sh_top_paths, "pan")
    evs_shp = {e.anchors: e for e in res_sh_prio.sv_events if e.event_type == "INS"}
    assert "5..6" in evs_shp and "3..4" not in evs_shp

    sh_nested_paths = [
        (f"HapO{k}#1#Chr01", _ins34([("Oa", "+"), ("Ox", "+"), ("Ob", "+")]))
        for k in range(1, 7)
    ] + [
        (f"Hap1_{k}#1#Chr01", _ins34([("Oa", "+"), ("S1", "+"), ("S2", "+"), ("Ob", "+")]))
        for k in range(1, 5)
    ] + [
        ("Hap2#1#Chr01", _ins34([("Oa", "+"), ("S1", "+"), ("S2a", "+"), ("B40", "+"), ("S2t", "+"), ("Ob", "+")])),
        ("HapZ#1#Chr01", _ins34([("Oa", "+"), ("Shred", "+"), ("B40", "+"), ("Ob", "+")])),
    ]
    res_sh_n = _run_f1("i7_shred_nested_ref", sh_nodes, sh_nested_paths, "ref")
    ev_shn = next(e for e in res_sh_n.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_shn = [r for r in res_sh_n.inserts_rows if r["event_id"] == ev_shn.event_id]
    hz_n_bp = sum(r["pan_end"] - r["pan_start"] for r in ins_shn if "HapZ#1" in r["members"].split(","))
    assert hz_n_bp == 3000  # Oa(1500) + Ob(1500), zero S1/S2 leakage!


def test_iter7_phase0_unconditional_exact_both_endpoints_and_strand_sweep(
    tmp_path: Path,
) -> None:
    """Iter 7 Defect A (Unconditional Incremental Exact-Both-Endpoints Phase 0):
    1. PGGB_1BP_MH (ref & pan): 1 bp step deleted with mh_len == del_len == 1.
    2. Homopolymer AAAA+A, TANDEM_U40, PROBE_2B: mh_len >= del_len outside (u_s, u_e).
    3. r62_f2_gen: straddling homopolymer where range(b_lo_i, b_hi_i) is non-empty yet exact snap lies outside.
    4. Tandem >= 3 (u30x3, u1x3) & multi-deletion collision (st1 + mid30 + st2).
    5. 20x randomized +/- strand microhomology sweep (3.3b, 20/20 pass).
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=11100 + i) for i in range(1, 7)}
    s11 = make_complex_seq(1500, seed=11111)
    s12 = make_complex_seq(1500, seed=11112)

    def _run_p0(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=0.90, min_sv_len=50
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]

    def _p34(h1_mid: list[tuple[str, str]], h2_mid: list[tuple[str, str]], with_sup: bool) -> list[tuple[str, list[tuple[str, str]]]]:
        base = [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h1_mid, ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h1_mid, ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h2_mid, ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        return (sup6 + base) if with_sup else base

    def _h2_bp(res) -> int:
        ev = next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        return sum(
            r["pan_end"] - r["pan_start"]
            for r in res.inserts_rows
            if r["event_id"] == ev.event_id and "Hap2#1" in r["members"].split(",")
        )

    # (1) PGGB_1BP_MH (ref=7000, pan=4000)
    ga_1 = make_complex_seq(1998, seed=11121) + "TC"
    gb_1 = "G" + make_complex_seq(1999, seed=11122)
    t49 = make_complex_seq(49, seed=11123)
    ga_1p = mutate_seq_snps(ga_1, [i * 200 + 30 for i in range(9)])
    gb_1p = mutate_seq_snps(gb_1, [i * 200 + 30 for i in range(9)])
    nodes_1bp = {"10": s11 + s12, "11": s11, "gA": ga_1, "d1": "C", "gB": gb_1, "t49": t49, "12": s12, "alt": ga_1p + gb_1p}
    res_1bp_r = _run_p0("i7_pggb_1bp_ref", nodes_1bp, _p34([("gA", "+"), ("d1", "+"), ("gB", "+"), ("t49", "+")], [("alt", "+")], False), "ref")
    assert _h2_bp(res_1bp_r) == 7000
    res_1bp_p = _run_p0("i7_pggb_1bp_pan", nodes_1bp, _p34([("gA", "+"), ("d1", "+"), ("gB", "+"), ("t49", "+")], [("alt", "+")], True), "pan")
    assert _h2_bp(res_1bp_p) == 4000

    # (2) Homopolymer AAAA+A (7000 ref), TANDEM_U40 (4000 pan), PROBE_2B (2000 pan)
    ga_a4 = make_complex_seq(1995, seed=11124) + "GAAAA"
    gb_a4 = "T" + make_complex_seq(1999, seed=11125)
    nodes_a4 = {
        "10": s11 + s12, "11": s11, "gA": ga_a4, "d1": "A", "gB": gb_a4, "t49": t49, "12": s12,
        "alt": mutate_seq_snps(ga_a4, [i * 200 + 30 for i in range(9)]) + mutate_seq_snps(gb_a4, [i * 200 + 30 for i in range(9)]),
    }
    res_a4 = _run_p0("i7_aaaa_plus_a_ref", nodes_a4, _p34([("gA", "+"), ("d1", "+"), ("gB", "+"), ("t49", "+")], [("alt", "+")], False), "ref")
    assert _h2_bp(res_a4) == 7000

    u40 = make_complex_seq(40, seed=11126)
    ga_u40 = make_complex_seq(1960, seed=11127) + u40
    gb_u40 = make_complex_seq(2000, seed=11128)
    t30 = make_complex_seq(30, seed=11129)
    nodes_u40 = {
        "10": s11 + s12, "11": s11, "gA": ga_u40, "d40": u40, "gB": gb_u40, "t30": t30, "12": s12,
        "alt": mutate_seq_snps(ga_u40, [i * 200 + 20 for i in range(9)]) + mutate_seq_snps(gb_u40, [i * 200 + 20 for i in range(9)]),
    }
    res_u40 = _run_p0("i7_tandem_u40_pan", nodes_u40, _p34([("gA", "+"), ("d40", "+"), ("gB", "+"), ("t30", "+")], [("alt", "+")], True), "pan")
    assert _h2_bp(res_u40) == 4000

    p2b_m30 = make_complex_seq(30, seed=11131)
    p2b_14 = make_complex_seq(970, seed=11132) + p2b_m30
    p2b_15 = make_complex_seq(1000, seed=11133)
    nodes_2b = {
        "10": s11 + s12, "11": s11, "14": p2b_14, "m30": p2b_m30, "15": p2b_15, "t30": t30, "12": s12,
        "alt": mutate_seq_snps(p2b_14, [i * 100 + 20 for i in range(9)]) + mutate_seq_snps(p2b_15, [i * 100 + 20 for i in range(9)]),
    }
    res_2b = _run_p0("i7_probe_2b_pan", nodes_2b, _p34([("14", "+"), ("m30", "+"), ("15", "+"), ("t30", "+")], [("alt", "+")], True), "pan")
    assert _h2_bp(res_2b) == 2000

    # (3) r62_f2_gen: straddling homopolymer with non-empty range(b_lo_i, b_hi_i) -> 7030 ref
    ga_r62 = make_complex_seq(1980, seed=11141) + ("A" * 20)
    mid_r62 = "A" * 30
    d25_r62 = "A" * 25
    gb_r62 = "C" + make_complex_seq(1999, seed=11142)
    nodes_r62 = {
        "10": s11 + s12, "11": s11, "gA": ga_r62, "mid": mid_r62, "d25": d25_r62, "gB": gb_r62, "t30": t30, "12": s12,
        "alt": mutate_seq_snps(ga_r62, [i * 200 + 20 for i in range(9)]) + mid_r62 + mutate_seq_snps(gb_r62, [i * 200 + 20 for i in range(9)]),
    }
    res_r62 = _run_p0("i7_r62_f2_gen_ref", nodes_r62, _p34([("gA", "+"), ("mid", "+"), ("d25", "+"), ("gB", "+"), ("t30", "+")], [("alt", "+")], False), "ref")
    assert _h2_bp(res_r62) == 7030

    # (4) Tandem >= 3 (u30x3, u1x3) & multi-deletion collision (st1 + mid30 + st2)
    u30 = make_complex_seq(30, seed=11151)
    ga_3x = make_complex_seq(1910, seed=11152) + (u30 * 3)
    gb_3x = make_complex_seq(2000, seed=11153)
    nodes_3x = {
        "10": s11 + s12, "11": s11, "gA": ga_3x, "d30": u30, "gB": gb_3x, "t30": t30, "12": s12,
        "alt": mutate_seq_snps(ga_3x, [i * 200 + 20 for i in range(9)]) + mutate_seq_snps(gb_3x, [i * 200 + 20 for i in range(9)]),
    }
    res_3x = _run_p0("i7_u30x3_pan", nodes_3x, _p34([("gA", "+"), ("d30", "+"), ("gB", "+"), ("t30", "+")], [("alt", "+")], True), "pan")
    assert _h2_bp(res_3x) == 4000

    ga_g6 = make_complex_seq(1993, seed=11154) + "TGGGGGG"
    gb_g6 = "C" + make_complex_seq(1999, seed=11155)
    nodes_g6 = {
        "10": s11 + s12, "11": s11, "gA": ga_g6, "d1": "G", "gB": gb_g6, "t49": t49, "12": s12,
        "alt": mutate_seq_snps(ga_g6, [i * 200 + 20 for i in range(9)]) + mutate_seq_snps(gb_g6, [i * 200 + 20 for i in range(9)]),
    }
    res_g6 = _run_p0("i7_u1x3_pan", nodes_g6, _p34([("gA", "+"), ("d1", "+"), ("gB", "+"), ("t49", "+")], [("alt", "+")], True), "pan")
    assert _h2_bp(res_g6) == 4000

    # Multi-deletion collision (st1 + mid30 + st2)
    u20 = make_complex_seq(20, seed=11161)
    v20 = make_complex_seq(20, seed=11162)
    ga_mc = make_complex_seq(1980, seed=11163) + u20
    mid_mc = u20[:10] + make_complex_seq(10, seed=11164) + v20[:10]
    gb_mc = v20 + make_complex_seq(1980, seed=11165)
    t20 = make_complex_seq(20, seed=11166)
    nodes_mc = {
        "10": s11 + s12, "11": s11, "gA": ga_mc, "st1": u20, "mid": mid_mc, "st2": v20, "gB": gb_mc, "t20": t20, "12": s12,
        "alt": mutate_seq_snps(ga_mc, [i * 200 + 20 for i in range(9)]) + mid_mc + mutate_seq_snps(gb_mc, [i * 200 + 20 for i in range(9)]),
    }
    res_mc = _run_p0("i7_multi_del_coll_ref", nodes_mc, _p34([("gA", "+"), ("st1", "+"), ("mid", "+"), ("st2", "+"), ("gB", "+"), ("t20", "+")], [("alt", "+")], False), "ref")
    assert _h2_bp(res_mc) == 7030

    # (5) 20x randomized +/- strand microhomology sweep (3.3b, 20/20 pass)
    rng_sw = random.Random(20260733)
    for trial in range(20):
        strand = "+" if trial < 10 else "-"
        del_len = rng_sw.randint(1, 35)
        mh_len = rng_sw.randint(1, 45)
        t_len = max(20, 55 - del_len)
        unit = make_complex_seq(del_len, seed=11200 + trial * 10)
        mh_tail = (unit * ((mh_len // del_len) + 2))[-mh_len:]
        ga_sw = make_complex_seq(1500 - mh_len, seed=11201 + trial * 10) + mh_tail
        gb_sw = make_complex_seq(1500, seed=11202 + trial * 10)
        td_sw = make_complex_seq(t_len, seed=11203 + trial * 10)
        alt_sw = mutate_seq_snps(ga_sw, [i * 200 + 20 for i in range(7)]) + mutate_seq_snps(gb_sw, [i * 200 + 20 for i in range(7)])
        if strand == "+":
            sw_nodes = {"10": s11 + s12, "11": s11, "gA": ga_sw, "d": unit, "gB": gb_sw, "td": td_sw, "12": s12, "alt": alt_sw}
            h1_m = [("gA", "+"), ("d", "+"), ("gB", "+"), ("td", "+")]
        else:
            sw_nodes = {
                "10": s11 + s12, "11": s11, "12": s12, "alt": alt_sw,
                "gA": revcomp_dna(ga_sw), "d": revcomp_dna(unit), "gB": revcomp_dna(gb_sw), "td": revcomp_dna(td_sw),
            }
            h1_m = [("gA", "-"), ("d", "-"), ("gB", "-"), ("td", "-")]
        res_sw = _run_p0(f"i7_sweep_{trial}_{'p' if strand == '+' else 'm'}", sw_nodes, _p34(h1_m, [("alt", "+")], False), "ref")
        assert _h2_bp(res_sw) == 6000, f"Phase 0 sweep trial={trial} ({strand}, del={del_len}, mh={mh_len}) got {_h2_bp(res_sw)}"


def test_iter7_subslice_killshots_benign_exemption_and_stage55_short_runs(
    tmp_path: Path,
) -> None:
    """Iter 7 Defect B + Refinements A/B/C + Coordinate/Merge Probes:
    1. Killshots A, B, C, D, D2, D3 across dedup_id in {0.80, 0.90, 0.95, 0.99} (0 spacers in ref).
    2. Killshot E (Phase 2a 10 kb gene 35R + u1(40)): nov75 in pan_seq, dedup_short_residual_bp == 0.
    3. Refinement A: ATTACK_2A (3 bp del + 3 SNPs same 50 bp win at 0.90), ATTACK_2E (6 bp del at 0.80),
       ATTACK_2F (Killshot D h2_bp==10000 at 0.95), ATTACK_2H/2I, ATTACK_2C (multi-hap gA).
    4. Refinement B: SPLIT_COV_35_30, SPLIT_COV_20_80, ATTACK_4A (mid5 dead-end bridge), EXISTING_CUTS_ISLAND.
    5. Refinement C: MULTI_SUBSLICE_TRAP & ref_c_no_false_drop (isolated 25 bp match inside >=50 bp run).
    6. Stage 2.5a already-sliced partial Step (orient='-' & src_strand='-') + R4-2 sub-slicing,
       Step 4b merge & exact_target isolation on sub-sliced real_rep.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=11300 + i) for i in range(1, 7)}
    s11 = make_complex_seq(1500, seed=11311)
    s12 = make_complex_seq(1500, seed=11312)

    def _run_b(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str,
        dedup_id: float = 0.90,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=dedup_id, min_sv_len=50
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]

    def _mk34(h1_m: list[tuple[str, str]], h2_m: list[tuple[str, str]], with_sup: bool) -> list[tuple[str, list[tuple[str, str]]]]:
        base = [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h1_m, ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h1_m, ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), *h2_m, ("12", "+"), ("5", "+"), ("6", "+")]),
        ]
        return (sup6 + base) if with_sup else base

    # (1) Killshots A, D, D2, D3 across dedup_id in {0.80, 0.90, 0.95, 0.99} (includes ATTACK_2F at 0.95)
    # Note: gB is a shared anchor step in Hap2 so tdev is [gA(10000), TE49(49)]: internal_del=1 (< 50), TE49 ov=0 -> total=50 >= 50!
    ga10 = make_complex_seq(10000, seed=11321)
    te49 = make_complex_seq(49, seed=11322)
    gb5 = make_complex_seq(5000, seed=11323)
    ga10_1d = ga10[:5000] + ga10[5001:]
    ks_a_nodes_base = {"11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga10_1d}
    ks_a_paths_ref = _mk34([("gA", "+"), ("TE49", "+"), ("gB", "+")], [("alt", "+"), ("gB", "+")], False)
    ks_a_paths_pan = _mk34([("gA", "+"), ("TE49", "+"), ("gB", "+")], [("alt", "+"), ("gB", "+")], True)
    for did in (0.80, 0.90, 0.95, 0.99):
        dtag = int(round(did * 100))
        # Killshot A (ref, 0 spacers)
        res_ka = _run_b(f"i7_ksA_ref_{dtag}", {**ks_a_nodes_base, "10": s11 + s12}, ks_a_paths_ref, "ref", dedup_id=did)
        ev_ka = next(e for e in res_ka.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_ka = [r for r in res_ka.inserts_rows if r["event_id"] == ev_ka.event_id]
        assert all(r["join_before"] != "spacer" for r in ins_ka)
        h2_ka = sum(r["pan_end"] - r["pan_start"] for r in ins_ka if "Hap2#1" in r["members"].split(","))
        assert h2_ka == (18000 if did <= 0.95 else 17999)

        # Killshot D (pan, Sup = 11+gB+12 -> gA novel; ATTACK_2F explicitly checks exact 10000 at 0.95)
        res_kd = _run_b(f"i7_ksD_pan_{dtag}", {**ks_a_nodes_base, "10": s11 + gb5 + s12}, ks_a_paths_pan, "pan", dedup_id=did)
        ev_kd = next(e for e in res_kd.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_kd = [r for r in res_kd.inserts_rows if r["event_id"] == ev_kd.event_id]
        assert ev_kd.genotypes["Hap2#1"] != "."
        h2_kd = sum(r["pan_end"] - r["pan_start"] for r in ins_kd if "Hap2#1" in r["members"].split(","))
        if did <= 0.95:
            assert h2_kd == 10000, f"Killshot D / ATTACK_2F failed at dedup_id={did}: h2_bp={h2_kd}"
        else:
            assert h2_kd >= 9995, f"Killshot D failed at dedup_id=0.99: h2_bp={h2_kd}"

        # Killshot D2 (pan, Sup = 100% Hap2 -> gt[Hap2] == '.')
        res_kd2 = _run_b(f"i7_ksD2_pan_{dtag}", {**ks_a_nodes_base, "10": s11 + ga10_1d + gb5 + s12}, ks_a_paths_pan, "pan", dedup_id=did)
        ev_kd2 = next(e for e in res_kd2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
        ins_kd2 = [r for r in res_kd2.inserts_rows if r["event_id"] == ev_kd2.event_id]
        assert ev_kd2.genotypes["Hap2#1"] == "."
        assert all("Hap2#1" not in r["members"].split(",") for r in ins_kd2)

        # Killshot D3 (pan, Sup = 100% b_sec -> 0 INS at 3..4)
        res_kd3 = _run_b(f"i7_ksD3_pan_{dtag}", {**ks_a_nodes_base, "10": s11 + ga10 + te49 + gb5 + s12}, ks_a_paths_pan, "pan", dedup_id=did)
        assert not [e for e in res_kd3.sv_events if e.event_type == "INS" and e.anchors == "3..4"]

    # Killshot B (2 x 15 kb genes, 25L + 25R = 50 bp) & Killshot C (single 30 kb gene, 20 internal D + 30 right trim = 50 bp)
    g15a = make_complex_seq(15000, seed=11331)
    g15b = make_complex_seq(15000, seed=11332)
    ksb_alt = g15a[25:] + g15b[:14975]
    ksb_nodes = {"11": s11, "g1": g15a, "g2": g15b, "12": s12, "alt": ksb_alt}
    res_ksb_r = _run_b("i7_ksB_ref", {**ksb_nodes, "10": s11 + s12}, _mk34([("g1", "+"), ("g2", "+")], [("alt", "+")], False), "ref")
    ev_ksb_r = next(e for e in res_ksb_r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_ksb_r = [r for r in res_ksb_r.inserts_rows if r["event_id"] == ev_ksb_r.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_ksb_r)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ksb_r if "Hap2#1" in r["members"].split(",")) == 32950
    res_ksb_pn = _run_b("i7_ksB_pan_nov", {**ksb_nodes, "10": s11 + s12}, _mk34([("g1", "+"), ("g2", "+")], [("alt", "+")], True), "pan")
    ev_ksb_pn = next(e for e in res_ksb_pn.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_ksb_pn.genotypes["Hap2#1"] != "."
    assert sum(r["pan_end"] - r["pan_start"] for r in res_ksb_pn.inserts_rows if r["event_id"] == ev_ksb_pn.event_id and "Hap2#1" in r["members"].split(",")) == 29950
    res_ksb_ps = _run_b("i7_ksB_pan_sup", {**ksb_nodes, "10": s11 + ksb_alt + s12}, _mk34([("g1", "+"), ("g2", "+")], [("alt", "+")], True), "pan")
    ev_ksb_ps = next(e for e in res_ksb_ps.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_ksb_ps.genotypes["Hap2#1"] == "."

    g30a = make_complex_seq(15000, seed=11333)
    g30b = make_complex_seq(15000, seed=11334)
    ksc_alt = (g30a[:7500] + g30a[7520:]) + g30b[:14970]  # 20 bp internal D on g30a (< 50) + 30 bp right trim on g30b (< 50) = 50 bp >= 50
    ksc_nodes = {"11": s11, "g30a": g30a, "g30b": g30b, "12": s12, "alt": ksc_alt}
    res_ksc_r = _run_b("i7_ksC_ref", {**ksc_nodes, "10": s11 + s12}, _mk34([("g30a", "+"), ("g30b", "+")], [("alt", "+")], False), "ref")
    ev_ksc_r = next(e for e in res_ksc_r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_ksc_r = [r for r in res_ksc_r.inserts_rows if r["event_id"] == ev_ksc_r.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_ksc_r)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ksc_r if "Hap2#1" in r["members"].split(",")) == 32950
    res_ksc_pn = _run_b("i7_ksC_pan_nov", {**ksc_nodes, "10": s11 + s12}, _mk34([("g30a", "+"), ("g30b", "+")], [("alt", "+")], True), "pan")
    ev_ksc_pn = next(e for e in res_ksc_pn.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_ksc_pn.genotypes["Hap2#1"] != "."
    assert sum(r["pan_end"] - r["pan_start"] for r in res_ksc_pn.inserts_rows if r["event_id"] == ev_ksc_pn.event_id and "Hap2#1" in r["members"].split(",")) == 29950
    res_ksc_ps = _run_b("i7_ksC_pan_sup", {**ksc_nodes, "10": s11 + ksc_alt + s12}, _mk34([("g30a", "+"), ("g30b", "+")], [("alt", "+")], True), "pan")
    ev_ksc_ps = next(e for e in res_ksc_ps.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_ksc_ps.genotypes["Hap2#1"] == "."

    # (2) Killshot E (Phase 2a 10 kb gene 35R + u1(40))
    u1_40 = make_complex_seq(40, seed=11335)
    nov75_e = ga10[9965:] + u1_40
    kse_nodes = {"11": s11, "gA": ga10, "u1": u1_40, "12": s12, "alt": ga10[:9965]}
    res_kse_r = _run_b("i7_ksE_ref", {**kse_nodes, "10": s11 + s12}, _mk34([("gA", "+"), ("u1", "+")], [("alt", "+")], False), "ref")
    ev_kse_r = next(e for e in res_kse_r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert sum(r["pan_end"] - r["pan_start"] for r in res_kse_r.inserts_rows if r["event_id"] == ev_kse_r.event_id and "Hap2#1" in r["members"].split(",")) == 12965
    res_kse_ps = _run_b("i7_ksE_pan_sup", {**kse_nodes, "10": s11 + ga10[:9965] + s12}, _mk34([("gA", "+"), ("u1", "+")], [("alt", "+")], True), "pan")
    ev_kse_ps = next(e for e in res_kse_ps.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_kse_ps.genotypes["Hap2#1"] == "."
    assert nov75_e in res_kse_ps.pan_seq
    assert int(res_kse_ps.summary_metrics["dedup_short_residual_bp"]) == 0

    # (3) Refinement A probes: ATTACK_2A, ATTACK_2E, ATTACK_2H, ATTACK_2I, ATTACK_2C
    # ATTACK_2A: 3 bp internal del + 3 SNPs in SAME 50 bp window at dedup_id=0.90 + TE49
    ga_2a_pre = mutate_seq_snps(ga10, [4970, 4980, 4990])
    ga_2a_alt = ga_2a_pre[:5000] + ga_2a_pre[5003:]
    res_2a = _run_b(
        "i7_attack_2a_pan",
        {"10": s11 + ga_2a_alt + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga_2a_alt},
        ks_a_paths_pan,
        "pan",
        dedup_id=0.90,
    )
    ev_2a = next(e for e in res_2a.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2a.genotypes["Hap2#1"] == "."

    # ATTACK_2E: 6 bp internal del at dedup_id=0.80 (Condition 1 caps I <= 5) + TE49
    ga_2e_alt = ga10[:5000] + ga10[5006:]
    res_2e = _run_b(
        "i7_attack_2e_pan",
        {"10": s11 + ga_2e_alt + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga_2e_alt},
        ks_a_paths_pan,
        "pan",
        dedup_id=0.80,
    )
    ev_2e = next(e for e in res_2e.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2e.genotypes["Hap2#1"] == "."

    # ATTACK_2H (deletion 10 bp from step edge < REF_SUB_BLOCK_SIZE=50) & ATTACK_2I (80 bp step with 1 bp del at 40)
    ga_2h_alt = ga10[:10] + ga10[11:]
    res_2h = _run_b(
        "i7_attack_2h_pan",
        {"10": s11 + ga_2h_alt + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt": ga_2h_alt},
        ks_a_paths_pan,
        "pan",
        dedup_id=0.90,
    )
    ev_2h = next(e for e in res_2h.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2h.genotypes["Hap2#1"] == "."

    s80 = make_complex_seq(80, seed=11341)
    s80_d1 = s80[:40] + s80[41:]
    res_2i = _run_b(
        "i7_attack_2i_pan",
        {"10": s11 + s80_d1 + gb5 + s12, "11": s11, "s80": s80, "TE49": te49, "gB": gb5, "12": s12, "alt": s80_d1},
        _mk34([("s80", "+"), ("TE49", "+"), ("gB", "+")], [("alt", "+"), ("gB", "+")], True),
        "pan",
        dedup_id=0.90,
    )
    ev_2i = next(e for e in res_2i.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2i.genotypes["Hap2#1"] == "."

    # ATTACK_2C (multi-hap gA: Hap2 benign 1 bp D + TE49, Hap3 3 bp D + 3 SNPs + TE49)
    paths_2c = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_2c_h3 = _run_b(
        "i7_attack_2c_sup_h3_pan",
        {"10": s11 + ga_2a_alt + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt2": ga10_1d, "alt3": ga_2a_alt},
        paths_2c,
        "pan",
        dedup_id=0.90,
    )
    ev_2c_h3 = next(e for e in res_2c_h3.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2c_h3.genotypes["Hap3#1"] == "."
    assert all("Hap3#1" not in r["members"].split(",") for r in res_2c_h3.inserts_rows if r["event_id"] == ev_2c_h3.event_id)

    res_2c_h2 = _run_b(
        "i7_attack_2c_sup_h2_pan",
        {"10": s11 + ga10_1d + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt2": ga10_1d, "alt3": ga_2a_alt},
        paths_2c,
        "pan",
        dedup_id=0.90,
    )
    ev_2c_h2 = next(e for e in res_2c_h2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_2c_h2.genotypes["Hap2#1"] == "." and ev_2c_h2.genotypes["Hap3#1"] == "."
    assert all(
        "Hap2#1" not in r["members"].split(",") and "Hap3#1" not in r["members"].split(",")
        for r in res_2c_h2.inserts_rows
        if r["event_id"] == ev_2c_h2.event_id
    )

    # (4) Refinement B probes: SPLIT_COV_35_30, SPLIT_COV_20_80, ATTACK_4A, EXISTING_CUTS_ISLAND
    s100_a = make_complex_seq(100, seed=11351)
    t30_b = make_complex_seq(30, seed=11352)
    alt_35_30 = s100_a[:35] + s100_a[70:]
    res_sc1 = _run_b(
        "i7_split_cov_35_30_pan",
        {"10": s11 + alt_35_30 + s12, "11": s11, "s0": s100_a, "t30": t30_b, "12": s12, "alt": alt_35_30},
        _mk34([("s0", "+"), ("t30", "+")], [("alt", "+")], True),
        "pan",
    )
    ev_sc1 = next(e for e in res_sc1.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_sc1.genotypes["Hap2#1"] == "."

    alt_20_80 = s100_a[:20] + s100_a[60:]
    res_sc2 = _run_b(
        "i7_split_cov_20_80_pan",
        {"10": s11 + alt_20_80 + s12, "11": s11, "p30": t30_b, "s0": s100_a, "12": s12, "alt": alt_20_80},
        _mk34([("p30", "+"), ("s0", "+")], [("alt", "+")], True),
        "pan",
    )
    ev_sc2 = next(e for e in res_sc2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_sc2.genotypes["Hap2#1"] == "."

    # ATTACK_4A: mid(5) 100%-covered dead-end bridge trapped between sub-sliced prev(200) and cur(200)
    prev200 = make_complex_seq(200, seed=11353)
    mid5 = make_complex_seq(5, seed=11354)
    cur200 = make_complex_seq(200, seed=11355)
    prev_alt = prev200[:145] + prev200[180:200]
    cur_alt = cur200[0:20] + cur200[55:200]
    res_4a = _run_b(
        "i7_attack_4a_pan",
        {"10": s11 + prev_alt + mid5 + cur_alt + s12, "11": s11, "prev": prev200, "mid5": mid5, "cur": cur200, "12": s12, "p_alt": prev_alt, "c_alt": cur_alt},
        _mk34([("prev", "+"), ("mid5", "+"), ("cur", "+")], [("p_alt", "+"), ("mid5", "+"), ("c_alt", "+")], True),
        "pan",
    )
    ev_4a = next(e for e in res_4a.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_4a = [r for r in res_4a.inserts_rows if r["event_id"] == ev_4a.event_id]
    assert ev_4a.genotypes["Hap2#1"] == "."
    assert len(ins_4a) == 1
    p4a_len = ins_4a[0]["pan_end"] - ins_4a[0]["pan_start"]
    assert 110 <= p4a_len <= 120
    assert mid5 in res_4a.pan_seq[ins_4a[0]["pan_start"] : ins_4a[0]["pan_end"]]
    assert ins_4a[0]["members"] == "Hap1a#1,Hap1b#1"

    # EXISTING_CUTS_ISLAND: active child insertion at s0|12 creates existing_cut trapping s0[775:800] (25 bp)
    s0_ec = make_complex_seq(800, seed=11358)
    ins_ec = make_complex_seq(200, seed=11356)
    alt_ec = s0_ec[:740] + s0_ec[775:800]
    t20_ec = make_complex_seq(20, seed=11357)
    paths_ec2 = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("t20", "+"), ("s0", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("t20", "+"), ("s0", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("t20", "+"), ("s0", "+"), ("ins_c", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_ec2 = _run_b(
        "i7_existing_cuts_island_pan",
        {"10": s11 + alt_ec + s12, "11": s11, "t20": t20_ec, "s0": s0_ec, "ins_c": ins_ec, "12": s12, "alt": alt_ec},
        paths_ec2,
        "pan",
    )
    ev_ec2 = next(e for e in res_ec2.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_ec2.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_ec2.inserts_rows if r["event_id"] == ev_ec2.event_id)

    # (5) Refinement C probes: genuine rep0-merging TRUE_MST (adjacent & separated/ectopic Sup),
    # ref_c_no_false_drop (isolated 25 bp match inside >= 50 bp short-run), & P4_RIPPLE_REAL_pan
    l0_m = make_complex_seq(1500, seed=12311)
    n100_m = make_complex_seq(100, seed=12312)
    smid_m = make_complex_seq(95, seed=12313)
    ancr_m = make_complex_seq(700, seed=12314)
    r0_m = make_complex_seq(1500, seed=12315)
    nov_a_m = make_complex_seq(200, seed=12320)
    nov_b_m = make_complex_seq(200, seed=12321)
    mst_paths = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("n100", "+"), ("smid", "+"), ("ancR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("n100", "+"), ("smid", "+"), ("ancR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("alt2", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("alt3", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
    ]
    mst_base = {
        "L0": l0_m, "n100": n100_m, "smid": smid_m, "ancR": ancr_m, "R0": r0_m,
        "alt2": smid_m[:60] + ancr_m[:685],
        "alt3": n100_m + smid_m[30:] + ancr_m[:680],
    }

    # (5a) Adjacent-Sup TRUE_MST: both Hap2 & Hap3 merge into rep0 creating contiguous 30 + 29 short pieces
    res_mst = _run_b(
        "i7_true_mst_adj_sup_pan",
        {**mst_base, "10": l0_m + smid_m[:60] + ancr_m[:685] + r0_m},
        mst_paths,
        "pan",
    )
    ev_mst = next(e for e in res_mst.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_mst = [r for r in res_mst.inserts_rows if r["event_id"] == ev_mst.event_id]
    pan34_mst = "".join(res_mst.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_mst)
    assert ev_mst.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_mst)
    assert smid_m[:60] not in pan34_mst and smid_m[:30] not in pan34_mst and smid_m[30:59] not in pan34_mst
    assert res_mst.pan_seq.count(smid_m[:60]) == 1
    assert n100_m in pan34_mst and smid_m[60:94] in pan34_mst

    # (5b) Separated/ectopic-Sup TRUE_MST_SEPARATED_SUP_pan: unmasked-flank fallback prevents ancR/L0 hijacking
    res_mst_sep = _run_b(
        "i7_true_mst_sep_sup_pan",
        {**mst_base, "10": l0_m + nov_a_m + smid_m[:60] + nov_b_m + ancr_m[:685] + r0_m},
        mst_paths,
        "pan",
    )
    ev_mst_sep = next(e for e in res_mst_sep.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_mst_sep = [r for r in res_mst_sep.inserts_rows if r["event_id"] == ev_mst_sep.event_id]
    pan34_sep = "".join(res_mst_sep.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_mst_sep)
    assert ev_mst_sep.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_mst_sep)
    assert smid_m[:60] not in pan34_sep and smid_m[:30] not in pan34_sep and smid_m[30:59] not in pan34_sep
    assert res_mst_sep.pan_seq.count(smid_m[:60]) == 1
    assert n100_m in pan34_sep and smid_m[60:94] in pan34_sep

    # (5c) ref_c_no_false_drop: isolated 25 bp match inside >= 50 bp short-run MUST NOT be dropped!
    res_nfd = _run_b(
        "i7_ref_c_no_false_drop_pan",
        {**mst_base, "10": l0_m + nov_a_m + smid_m[:25] + nov_b_m + ancr_m[:685] + r0_m},
        mst_paths,
        "pan",
    )
    ev_nfd = next(e for e in res_nfd.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_nfd = [r for r in res_nfd.inserts_rows if r["event_id"] == ev_nfd.event_id]
    pan34_nfd = "".join(res_nfd.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_nfd)
    assert ev_nfd.genotypes["Hap2#1"] != "."
    assert smid_m[:30] in pan34_nfd and smid_m[30:59] in pan34_nfd

    # (5d) P4_RIPPLE_REAL_pan: 60 bp exact match sm[25:85] across 30+30 short pieces separated from masked R0
    nov250_p4 = make_complex_seq(250, seed=12401)
    sm120_p4 = make_complex_seq(120, seed=12402)
    t25_p4 = make_complex_seq(25, seed=12403)
    p4_paths = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("sm", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("sm", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a2", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a3", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a4", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_p4 = _run_b(
        "i7_p4_ripple_real_pan",
        {
            "10": l0_m + nov_a_m + sm120_p4[25:85] + nov_b_m + r0_m,
            "L0": l0_m, "nov250": nov250_p4, "sm": sm120_p4, "t25": t25_p4, "R0": r0_m,
            "a2": sm120_p4[:85], "a3": sm120_p4[25:], "a4": sm120_p4[:25] + sm120_p4[55:],
        },
        p4_paths,
        "pan",
    )
    ev_p4 = next(e for e in res_p4.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_p4 = [r for r in res_p4.inserts_rows if r["event_id"] == ev_p4.event_id]
    pan34_p4 = "".join(res_p4.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_p4)
    assert sm120_p4[25:55] not in pan34_p4 and sm120_p4[55:85] not in pan34_p4
    assert sm120_p4[:25] in pan34_p4 and sm120_p4[85:] in pan34_p4 and t25_p4 in pan34_p4

    # (6) Coordinate & Merge Probes:
    # (6a) Stage 2.5a already-sliced partial Step (orient='-') + R4-2 sub-slicing + [INV2] PASS + 0 spacers
    ref_pfx = base_ref6["1"]  # 400 bp exact reference homolog stripped by Stage 2.5a
    nov_1200 = make_complex_seq(1200, seed=11371)
    nov_1200_alt = nov_1200[:600] + nov_1200[620:]  # 20 bp internal D (< 50) + t30(30) ov=0 -> total=50 >= 50
    t30_6 = make_complex_seq(30, seed=11373)
    ps_nodes = {
        "ns_rc": revcomp_dna(ref_pfx + nov_1200),
        "t30": t30_6,
        "12": s12,
        "alt_rc": revcomp_dna(ref_pfx + nov_1200_alt),
    }
    ps_paths = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ns_rc", "-"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("ns_rc", "-"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("alt_rc", "-"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_ps = _run_b("i7_already_sliced_minus_ref", ps_nodes, ps_paths, "ref")
    ev_ps = next(e for e in res_ps.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_ps = [r for r in res_ps.inserts_rows if r["event_id"] == ev_ps.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_ps)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ps if "Hap1a#1" in r["members"].split(",")) == 2730
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ps if "Hap2#1" in r["members"].split(",")) == 2680

    # (6b) Inverted contig (src_strand == '-') + R4-2 sub-slicing -> [INV2] PASS + 0 spacers
    inv_nodes = {"11": s11, "g4": make_complex_seq(4000, seed=11372), "t30": t30_6, "12": s12}
    inv_g4 = inv_nodes["g4"]
    inv_nodes["alt"] = inv_g4[:2000] + inv_g4[2020:]  # 20 bp internal D (< 50) + t30(30) = 50 bp

    def _rev(steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(n, "-" if o == "+" else "+") for n, o in reversed(steps)]

    inv_paths = [
        ("Hap1a#1#Chr01", _rev([("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")])),
        ("Hap1b#1#Chr01", _rev([("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")])),
        ("Hap2#1#Chr01", _rev([("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt", "+"), ("12", "+"), ("5", "+"), ("6", "+")])),
    ]
    res_inv = _run_b("i7_inv_contig_subslice_ref", inv_nodes, inv_paths, "ref")
    ev_inv = next(e for e in res_inv.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_inv = [r for r in res_inv.inserts_rows if r["event_id"] == ev_inv.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_inv)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_inv if "Hap2#1" in r["members"].split(",")) == 6980

    # (6c) Step 4b merge into sub-sliced real_rep + exact_target isolation
    mut_full = mutate_seq_snps(s11 + inv_g4 + t30_6 + s12, [i * 350 + 30 for i in range(18)])
    s4b_paths = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("mut_full", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_s4b = _run_b("i7_step4b_subslice_ref", {**inv_nodes, "mut_full": mut_full}, s4b_paths, "ref")
    ev_s4b = next(e for e in res_s4b.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_s4b = [r for r in res_s4b.inserts_rows if r["event_id"] == ev_s4b.event_id]
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_s4b if "Hap4#1" in r["members"].split(",")) == 7030
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_s4b if "Hap2#1" in r["members"].split(",")) == 6980

    # exact_target isolation with sub-sliced real_rep vs second component of identical sequence -> 0 cross-contamination
    ex_nodes = {
        "11": s11, "g4": inv_g4, "t30": t30_6, "12": s12, "alt": inv_nodes["alt"],
        "11b": s11, "g4b": inv_g4, "t30b": t30_6, "12b": s12, "altb": inv_g4[:3940],  # 60 bp right trim -> whole-step excluded g4b+t30b (len=3000)
    }
    ex_paths = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g4", "+"), ("t30", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap5a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11b", "+"), ("g4b", "+"), ("t30b", "+"), ("12b", "+"), ("5", "+"), ("6", "+")]),
        ("Hap5b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11b", "+"), ("g4b", "+"), ("t30b", "+"), ("12b", "+"), ("5", "+"), ("6", "+")]),
        ("Hap6#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11b", "+"), ("altb", "+"), ("12b", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_ex = _run_b("i7_exact_target_subslice_ref", ex_nodes, ex_paths, "ref")
    ev_ex = next(e for e in res_ex.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_ex = [r for r in res_ex.inserts_rows if r["event_id"] == ev_ex.event_id]
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ex if "Hap2#1" in r["members"].split(",")) == 6980
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_ex if "Hap6#1" in r["members"].split(",")) == 3000


def test_iter7_cross_tdev_aggregation(tmp_path: Path) -> None:
    """Iter 7 Defect C (Unified Member-Wide Cross-tdev Aggregation):
    1. XT-70 (ref & pan): 2 x 35 bp ov==0 deletions across 2 tdevs.
    2. XT-200 (pan): 5 x 40 bp ov==0 deletions across 5 tdevs.
    3. 4.1 (ref & pan): 2 x 30 bp ov==0 cross-tdev inside synthetic component.
    4. sc==rep0 (sp=0 ref & default sp=100 pan): synthetic component matching rep0 at sp=0,
       and clean nv1/nv2 exclusion at default sp=100 pan.
    5. XT-MIX (ref & pan): ov==0 d35 + partial 20 bp del on 1 kb step across 2 tdevs.
    6. Case C3a (pan): d100 (>=50) + nv2(35) (ov==0) across 2 tdevs.
    7. Killshot-XT (ref & pan): g1(10 kb, 25D) + M + g2(10 kb, 25D) across 2 tdevs.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=11400 + i) for i in range(1, 7)}
    s11 = make_complex_seq(1500, seed=11411)
    s12 = make_complex_seq(1500, seed=11412)
    mid500 = make_complex_seq(500, seed=11413)

    def _run_xt(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str,
        spacer_n: int = 100,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=0.90, min_sv_len=50, spacer_n=spacer_n
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]

    # (1) XT-70 (ref & pan)
    d35a = make_complex_seq(35, seed=11421)
    d35b = make_complex_seq(35, seed=11422)
    xt70_nodes = {"10": s11 + mid500 + s12, "11": s11, "d1": d35a, "mid": mid500, "d2": d35b, "12": s12}
    xt70_base = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d1", "+"), ("mid", "+"), ("d2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d1", "+"), ("mid", "+"), ("d2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("mid", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_70r = _run_xt("i7_xt70_ref", xt70_nodes, xt70_base, "ref")
    ev_70r = next(e for e in res_70r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_70r = [r for r in res_70r.inserts_rows if r["event_id"] == ev_70r.event_id]
    assert ev_70r.genotypes["Hap1a#1"] != ev_70r.genotypes["Hap2#1"]
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_70r if "Hap2#1" in r["members"].split(",")) == 3500
    assert all(r["join_before"] != "spacer" for r in ins_70r)
    res_70p = _run_xt("i7_xt70_pan", xt70_nodes, sup6 + xt70_base, "pan")
    ev_70p = next(e for e in res_70p.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_70p.genotypes["Hap1a#1"] == "1" and ev_70p.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_70p.inserts_rows if r["event_id"] == ev_70p.event_id)

    # (2) XT-200 (5 x 40 bp ov==0 deletions across 5 tdevs, pan)
    xt200_nodes: dict[str, str] = {"11": s11, "12": s12}
    h1_200: list[tuple[str, str]] = [("11", "+")]
    h2_200: list[tuple[str, str]] = [("11", "+")]
    sup_200_parts: list[str] = [s11]
    for k in range(5):
        dk = make_complex_seq(40, seed=11430 + k * 2)
        xt200_nodes[f"d{k}"] = dk
        h1_200.append((f"d{k}", "+"))
        if k < 4:
            mk = make_complex_seq(200, seed=11431 + k * 2)
            xt200_nodes[f"m{k}"] = mk
            h1_200.append((f"m{k}", "+"))
            h2_200.append((f"m{k}", "+"))
            sup_200_parts.append(mk)
    h1_200.append(("12", "+"))
    h2_200.append(("12", "+"))
    sup_200_parts.append(s12)
    xt200_nodes["10"] = "".join(sup_200_parts)
    xt200_paths = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), *h1_200, ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), *h1_200, ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), *h2_200, ("5", "+"), ("6", "+")]),
    ]
    res_200p = _run_xt("i7_xt200_pan", xt200_nodes, xt200_paths, "pan")
    ev_200p = next(e for e in res_200p.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_200p.genotypes["Hap1a#1"] == "1" and ev_200p.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_200p.inserts_rows if r["event_id"] == ev_200p.event_id)

    # (3) 4.1 (2 x 30 bp ov==0 cross-tdev inside synthetic component, ref & pan)
    d30a = make_complex_seq(30, seed=11441)
    d30b = make_complex_seq(30, seed=11442)
    ins200 = make_complex_seq(200, seed=11443)
    n41_nodes = {"10": s11 + mid500 + s12, "11": s11, "d1": d30a, "mid": mid500, "d2": d30b, "ins2": ins200, "12": s12}
    n41_base = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d1", "+"), ("mid", "+"), ("d2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d1", "+"), ("mid", "+"), ("d2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("mid", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ins2", "+"), ("mid", "+"), ("d2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_41r = _run_xt("i7_41_ref", n41_nodes, n41_base, "ref")
    ev_41r = next(e for e in res_41r.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert sum(r["pan_end"] - r["pan_start"] for r in res_41r.inserts_rows if r["event_id"] == ev_41r.event_id and "Hap2#1" in r["members"].split(",")) == 3500
    res_41p = _run_xt("i7_41_pan", n41_nodes, sup6 + n41_base, "pan")
    ev_41p = next(e for e in res_41p.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_41p.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_41p.inserts_rows if r["event_id"] == ev_41p.event_id)

    # (4) sc==rep0 (sp=0 ref & default sp=100 pan)
    sc_la = make_complex_seq(1500, seed=11451)
    sc_x = make_complex_seq(1500, seed=11452)
    sc_nv1 = make_complex_seq(35, seed=11453)
    sc_b0 = make_complex_seq(500, seed=11454)
    sc_i1 = make_complex_seq(500, seed=11455)
    sc_y = make_complex_seq(1000, seed=11456)
    sc_nv2 = make_complex_seq(35, seed=11457)
    sc_z = make_complex_seq(1500, seed=11458)
    sc_ra = make_complex_seq(1500, seed=11459)
    sc_rep0 = sc_x + sc_nv1 + sc_b0 + sc_i1 + sc_y + sc_nv2 + sc_z  # 5070 bp
    sc_nodes = {
        "10": sc_la + sc_x + sc_b0 + sc_y + sc_z + sc_ra,
        "La": sc_la, "Ra": sc_ra, "R0": sc_rep0,
        "X": sc_x, "nv1": sc_nv1, "B0": sc_b0, "I1": sc_i1, "Y": sc_y, "nv2": sc_nv2, "Z": sc_z,
    }
    sc_base = [
        (f"Hap0_{k}#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("R0", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 5)
    ] + [
        ("Hap3a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("X", "+"), ("nv1", "+"), ("B0", "+"), ("Y", "+"), ("nv2", "+"), ("Z", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("X", "+"), ("nv1", "+"), ("B0", "+"), ("Y", "+"), ("nv2", "+"), ("Z", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("X", "+"), ("B0", "+"), ("Y", "+"), ("Z", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("La", "+"), ("X", "+"), ("nv1", "+"), ("I1", "+"), ("Y", "+"), ("nv2", "+"), ("Z", "+"), ("Ra", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_scr = _run_xt("i7_sc_rep0_sp0_ref", sc_nodes, sc_base, "ref", spacer_n=0)
    ev_scr = next(e for e in res_scr.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_scr = [r for r in res_scr.inserts_rows if r["event_id"] == ev_scr.event_id]
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_scr if "Hap0_1#1" in r["members"].split(",")) == 8070
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_scr if "Hap2#1" in r["members"].split(",")) == 7500

    sup10_sc = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 11)
    ]
    res_scp = _run_xt("i7_sc_rep0_sp100_pan", sc_nodes, sup10_sc + sc_base, "pan", spacer_n=100)
    ev_scp = next(e for e in res_scp.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_scp.genotypes["Hap2#1"] == "."
    ins_scp = [r for r in res_scp.inserts_rows if r["event_id"] == ev_scp.event_id]
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_scp)

    # (5) XT-MIX (ov==0 d35 + partial 20 bp del on 1 kb step across 2 tdevs, ref & pan)
    sbig = make_complex_seq(1000, seed=11461)
    sbig_d20 = sbig[:500] + sbig[520:]
    xtm_nodes = {"10": s11 + mid500 + sbig_d20 + s12, "11": s11, "d35": d35a, "mid": mid500, "sbig": sbig, "12": s12, "sbig_d": sbig_d20}
    xtm_base = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d35", "+"), ("mid", "+"), ("sbig", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d35", "+"), ("mid", "+"), ("sbig", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("mid", "+"), ("sbig_d", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_mr = _run_xt("i7_xt_mix_ref", xtm_nodes, xtm_base, "ref")
    ev_mr = next(e for e in res_mr.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_mr = [r for r in res_mr.inserts_rows if r["event_id"] == ev_mr.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_mr)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_mr if "Hap2#1" in r["members"].split(",")) == 4480
    res_mp = _run_xt("i7_xt_mix_pan", xtm_nodes, sup6 + xtm_base, "pan")
    ev_mp = next(e for e in res_mp.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_mp.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_mp.inserts_rows if r["event_id"] == ev_mp.event_id)

    # (6) Case C3a (d100 >= 50 + nv2(35) ov==0 across 2 tdevs, pan)
    d100 = make_complex_seq(100, seed=11471)
    c3a_nodes = {"10": s11 + d100 + mid500 + s12, "11": s11, "d100": d100, "mid": mid500, "nv2": d35b, "12": s12}
    c3a_paths = sup6 + [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d100", "+"), ("mid", "+"), ("nv2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("d100", "+"), ("mid", "+"), ("nv2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("mid", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    res_c3a = _run_xt("i7_case_c3a_pan", c3a_nodes, c3a_paths, "pan")
    ev_c3a = next(e for e in res_c3a.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_c3a = [r for r in res_c3a.inserts_rows if r["event_id"] == ev_c3a.event_id]
    assert ev_c3a.genotypes["Hap1a#1"] == "1" and ev_c3a.genotypes["Hap2#1"] == "."
    assert all(r["members"] == "Hap1a#1,Hap1b#1" for r in ins_c3a)

    # (7) Killshot-XT (g1(10 kb, 25D) + M(1 kb) + g2(10 kb, 25D) across 2 tdevs)
    g1_10k = make_complex_seq(10000, seed=11481)
    m_1k = make_complex_seq(1000, seed=11482)
    g2_10k = make_complex_seq(10000, seed=11483)
    g1_d25 = g1_10k[:5000] + g1_10k[5025:]
    g2_d25 = g2_10k[:5000] + g2_10k[5025:]
    kxt_base = [
        ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g1", "+"), ("M", "+"), ("g2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g1", "+"), ("M", "+"), ("g2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g1d", "+"), ("M", "+"), ("g2d", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    kxt_nodes = {"11": s11, "g1": g1_10k, "M": m_1k, "g2": g2_10k, "12": s12, "g1d": g1_d25, "g2d": g2_d25}
    res_kxtr = _run_xt("i7_ks_xt_ref", {**kxt_nodes, "10": s11 + s12}, kxt_base, "ref")
    ev_kxtr = next(e for e in res_kxtr.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    ins_kxtr = [r for r in res_kxtr.inserts_rows if r["event_id"] == ev_kxtr.event_id]
    assert all(r["join_before"] != "spacer" for r in ins_kxtr)
    assert sum(r["pan_end"] - r["pan_start"] for r in ins_kxtr if "Hap2#1" in r["members"].split(",")) == 23950

    res_kxtpn = _run_xt("i7_ks_xt_pan_nov", {**kxt_nodes, "10": s11 + m_1k + s12}, sup6 + kxt_base, "pan")
    ev_kxtpn = next(e for e in res_kxtpn.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_kxtpn.genotypes["Hap2#1"] != "."
    h2_kxtpn = sum(r["pan_end"] - r["pan_start"] for r in res_kxtpn.inserts_rows if r["event_id"] == ev_kxtpn.event_id and "Hap2#1" in r["members"].split(","))
    assert 19950 <= h2_kxtpn <= 20100

    res_kxtps = _run_xt("i7_ks_xt_pan_sup", {**kxt_nodes, "10": s11 + g1_d25 + m_1k + g2_d25 + s12}, sup6 + kxt_base, "pan")
    ev_kxtps = next(e for e in res_kxtps.sv_events if e.event_type == "INS" and e.anchors == "3..4")
    assert ev_kxtps.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in res_kxtps.inserts_rows if r["event_id"] == ev_kxtps.event_id)


def test_iter8_phase2b_te50_deferred_revocation_local_window_and_refinement_c(
    tmp_path: Path,
) -> None:
    """Iter 8 Surgical Closure Regressions (all r7_1 & r7_2 adversarial probes):
    1. Phase 2b bypass removal (member_excluded_uncov_bp >= min_sv_len):
       - TE500_25D (500 bp TE + 25 bp internal gene del, default pan 0.90)
       - PROBE9 (TE50 + 20D @ 0.90), P1E (cross-tdev TE50 + M + 20D), P1F (TE50 + L30)
       - PROBE2 / P1B2 (TE50 + ATTACK_2A 3 bp del + 3 SNPs at 4960..4990 @ 0.90)
       - PROBE3 / P1H2 (ATTACK_2C with TE50 vs TE49 -> deferred revocation closes 3 bp leak)
       - PROBE4 / P1C2 (TE50 + 6D @ 0.80)
       - PROBE8 / P1D (TE50 + 1D @ 0.99)
       - Benign TE50 + 1 bp del stays 100% unsplit (h2_bp == 10000) when alone
    2. total < min_sv_len deferred revocation & rev_ov_bp < min_sv_len promotion:
       - PROBE6 (Hap2 1 bp del with TE49 intact vs Hap3 sub-slice -> Hap2/Hap3 gt == '.', 0 ghost alleles)
       - PROBE6b (with novel TE100 -> 3..4 survives, gt[Hap2] == '.')
       - REV_OV_LT50 (s60: Hap2 25D ov=35 < 50 revoked into b_whole by Hap3 10D sub-slice)
    3. _is_sub_block_benign_in_rep0 tight local window (P2D_ECTOPIC_FOOL_BENIGN_SupH2)
       & Knaster-Tarski benign_edge_margin = max(50, min_sv_len) at min_sv_len=100.
    4. Refinement C unmasked-flank fallback on TRUE_MST_SEPARATED_SUP_pan & P4_RIPPLE_REAL_pan.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=12800 + i) for i in range(1, 7)}
    sup6 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 7)
    ]
    s11 = make_complex_seq(1500, seed=12811)
    s12 = make_complex_seq(1500, seed=12812)
    ga10 = make_complex_seq(10000, seed=12830)
    gb5 = make_complex_seq(5000, seed=12831)
    te500 = make_complex_seq(500, seed=12832)
    te50 = make_complex_seq(50, seed=12833)
    te49 = make_complex_seq(49, seed=12834)
    m500 = make_complex_seq(500, seed=12835)

    def _run_i8(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str = "pan",
        dedup_id: float = 0.90,
        min_sv_len: int = 50,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=dedup_id, min_sv_len=min_sv_len
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix,
            gfa_path=cfg.gfa,
            ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix,
            spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _ev34(res):
        return next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")

    def _ins34(res, ev):
        return [r for r in res.inserts_rows if r["event_id"] == ev.event_id]

    def _mk_te_paths(te_node: str, hap_name: str = "Hap2#1#Chr01") -> list[tuple[str, list[tuple[str, str]]]]:
        return sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), (te_node, "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), (te_node, "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            (hap_name,        [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ]

    # (1a) TE500_25D: 500 bp TE del + 25 bp internal gene del (default pan 0.90)
    ga_d25 = ga10[:5000] + ga10[5025:]
    res_500 = _run_i8(
        "i8_te500_25d_pan",
        {"10": s11 + ga_d25 + gb5 + s12, "11": s11, "gA": ga10, "TE": te500, "gB": gb5, "12": s12, "alt": ga_d25},
        _mk_te_paths("TE"),
    )
    ev_500 = _ev34(res_500)
    assert ev_500.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_500, ev_500))

    # (1b) PROBE9: TE50 + 20 bp internal del @ 0.90, plus TE50 + L30 trim & cross-tdev TE50 + M + 20D
    ga_d20 = ga10[:5000] + ga10[5020:]
    res_p9 = _run_i8(
        "i8_probe9_te50_20d_pan",
        {"10": s11 + ga_d20 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_d20},
        _mk_te_paths("TE"),
    )
    ev_p9 = _ev34(res_p9)
    assert ev_p9.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p9, ev_p9))

    ga_l30 = ga10[30:]
    res_p1f = _run_i8(
        "i8_p1f_te50_l30_pan",
        {"10": s11 + ga_l30 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_l30},
        _mk_te_paths("TE"),
    )
    ev_p1f = _ev34(res_p1f)
    assert ev_p1f.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p1f, ev_p1f))

    res_p1e = _run_i8(
        "i8_p1e_te50_xt_20d_pan",
        {"10": s11 + m500 + ga_d20 + gb5 + s12, "11": s11, "TE": te50, "M": m500, "gA": ga10, "gB": gb5, "12": s12, "alt": ga_d20},
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("TE", "+"), ("M", "+"), ("gA", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("TE", "+"), ("M", "+"), ("gA", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("M", "+"), ("alt", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_p1e = _ev34(res_p1e)
    assert ev_p1e.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p1e, ev_p1e))

    # (1c) PROBE2 / P1B2: TE50 + ATTACK_2A (3 bp del at 4960..4963 + 3 SNPs at 4970/4980/4990 @ 0.90)
    ga_2a_4960 = mutate_seq_snps(ga10, [4970, 4980, 4990])
    ga_2a_4960 = ga_2a_4960[:4960] + ga_2a_4960[4963:]
    res_p2 = _run_i8(
        "i8_probe2_te50_attack2a_pan",
        {"10": s11 + ga_2a_4960 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_2a_4960},
        _mk_te_paths("TE", hap_name="Hap3#1#Chr01"),
    )
    ev_p2 = _ev34(res_p2)
    assert ev_p2.genotypes["Hap3#1"] == "."
    assert all("Hap3#1" not in r["members"].split(",") for r in _ins34(res_p2, ev_p2))

    # (1d) PROBE3 / P1H2: ATTACK_2C with TE50 (Hap2) vs TE49 (Hap3) -> deferred revocation eliminates 3 bp ghost allele
    ga_d1 = ga10[:5000] + ga10[5001:]
    ga_2a_5000 = mutate_seq_snps(ga10, [5010, 5020, 5030])
    ga_2a_5000 = ga_2a_5000[:5000] + ga_2a_5000[5003:]
    res_p3 = _run_i8(
        "i8_probe3_attack2c_te50_vs_te49_pan",
        {
            "10": s11 + ga_d1 + te49 + m500 + gb5 + s12,
            "11": s11, "gA": ga10, "TE49": te49, "M": m500, "TE50": te50, "gB": gb5, "12": s12,
            "alt2": ga_d1, "alt3": ga_2a_5000,
        },
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("M", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("M", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("TE49", "+"), ("M", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("M", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_p3 = _ev34(res_p3)
    assert ev_p3.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p3, ev_p3))

    # (1e) PROBE4 / P1C2: TE50 + 6D @ 0.80
    ga_d6 = ga10[:5000] + ga10[5006:]
    res_p4d = _run_i8(
        "i8_probe4_te50_6d_080_pan",
        {"10": s11 + ga_d6 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_d6},
        _mk_te_paths("TE"),
        dedup_id=0.80,
    )
    ev_p4d = _ev34(res_p4d)
    assert ev_p4d.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p4d, ev_p4d))

    # (1f) PROBE8 / P1D: TE50 + 1D @ 0.99 (sub_block_cap == 0)
    res_p8 = _run_i8(
        "i8_probe8_te50_1d_099_pan",
        {"10": s11 + ga_d1 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_d1},
        _mk_te_paths("TE"),
        dedup_id=0.99,
    )
    ev_p8 = _ev34(res_p8)
    assert ev_p8.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p8, ev_p8))

    # (1g) Benign TE50 + 1 bp del stays unsplit when alone (novel gA -> h2_bp == 10000)
    res_b50 = _run_i8(
        "i8_benign_te50_1d_alone_pan",
        {"10": s11 + gb5 + s12, "11": s11, "gA": ga10, "TE": te50, "gB": gb5, "12": s12, "alt": ga_d1},
        _mk_te_paths("TE"),
    )
    ev_b50 = _ev34(res_b50)
    h2_b50 = sum(r["pan_end"] - r["pan_start"] for r in _ins34(res_b50, ev_b50) if "Hap2#1" in r["members"].split(","))
    assert ev_b50.genotypes["Hap2#1"] != "." and h2_b50 == 10000

    # (2a) PROBE6: member_total_uncov_bp < min_sv_len (Hap2 1 bp del, TE49 intact) vs Hap3 sub-slice
    res_p6 = _run_i8(
        "i8_probe6_total_lt50_drop_pan",
        {"10": s11 + ga_d1 + te49 + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "gB": gb5, "12": s12, "alt2": ga_d1, "alt3": ga_2a_5000},
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    evs_p6 = [e for e in res_p6.sv_events if e.event_type == "INS" and e.anchors == "3..4"]
    if evs_p6:
        assert evs_p6[0].genotypes["Hap2#1"] == "." and evs_p6[0].genotypes["Hap3#1"] == "."
        assert all(
            "Hap2#1" not in r["members"].split(",") and "Hap3#1" not in r["members"].split(",")
            for r in _ins34(res_p6, evs_p6[0])
        )

    # (2b) PROBE6b: same topology with novel TE100 in Hap1/Hap3 -> 3..4 survives, Hap2 has gt == "."
    te100 = make_complex_seq(100, seed=12836)
    res_p6b = _run_i8(
        "i8_probe6b_total_lt50_surv_pan",
        {"10": s11 + ga_d1 + te49 + gb5 + s12, "11": s11, "gA": ga10, "TE49": te49, "TE100": te100, "gB": gb5, "12": s12, "alt2": ga_d1, "alt3": ga_2a_5000},
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("TE100", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE49", "+"), ("TE100", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("TE49", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("TE100", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_p6b = _ev34(res_p6b)
    assert ev_p6b.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p6b, ev_p6b))

    # (2c) REV_OV_LT50: revoked member with rev_ov_bp < min_sv_len promotes to b_whole (0 ghost alleles)
    s60 = make_complex_seq(60, seed=12837)
    a2_60 = s60[:20] + s60[45:]
    a3_60 = s60[:20] + s60[30:]
    res_rev = _run_i8(
        "i8_rev_ov_lt50_pan",
        {"10": s11 + a2_60 + te50 + gb5 + s12, "11": s11, "s60": s60, "TE50": te50, "gB": gb5, "12": s12, "a2": a2_60, "a3": a3_60},
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("s60", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("s60", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("a2", "+"), ("TE50", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("a3", "+"), ("gB", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_rev = _ev34(res_rev)
    assert ev_rev.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_rev, ev_rev))

    # (3a) P2D_ECTOPIC_FOOL_BENIGN_SupH2: ectopic 100 bp repeat 500 bp away cannot fool local-window benign check
    nb_100 = ga10[4950:5000] + ga10[5001:5051]
    nb_ect = mutate_seq_snps(nb_100, [10, 20, 30, 40])
    ga_p2d_base = ga10[:5500] + nb_ect + ga10[5600:]
    ga_p2d_mut = mutate_seq_snps(ga_p2d_base, [4955, 4965, 4975, 4985, 4995, 5015])
    ga_p2d_alt = ga_p2d_mut[:5000] + ga_p2d_mut[5001:]
    res_p2d = _run_i8(
        "i8_p2d_ectopic_fool_pan",
        {"10": s11 + ga_p2d_alt + gb5 + s12, "11": s11, "gA": ga_p2d_base, "TE49": te49, "gB": gb5, "12": s12, "alt": ga_p2d_alt},
        _mk_te_paths("TE49"),
    )
    ev_p2d = _ev34(res_p2d)
    assert ev_p2d.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_p2d, ev_p2d))

    # (3b) Knaster-Tarski monotonicity edge margin max(50, min_sv_len) at min_sv_len=100
    te99 = make_complex_seq(99, seed=12838)
    ga_edge70 = ga10[:70] + ga10[71:]
    res_m100 = _run_i8(
        "i8_min_sv_100_edge_margin_pan",
        {"10": s11 + ga_edge70 + gb5 + s12, "11": s11, "gA": ga10, "TE99": te99, "gB": gb5, "12": s12, "alt": ga_edge70},
        _mk_te_paths("TE99"),
        min_sv_len=100,
    )
    ev_m100 = _ev34(res_m100)
    assert ev_m100.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_m100, ev_m100))

    # (4) Refinement C unmasked-flank fallback: TRUE_MST_SEPARATED_SUP_pan & P4_RIPPLE_REAL_pan
    l0 = make_complex_seq(1500, seed=12311)
    n100 = make_complex_seq(100, seed=12312)
    smid = make_complex_seq(95, seed=12313)
    ancr = make_complex_seq(700, seed=12314)
    r0 = make_complex_seq(1500, seed=12315)
    nov_a = make_complex_seq(200, seed=12320)
    nov_b = make_complex_seq(200, seed=12321)
    res_sep = _run_i8(
        "i8_true_mst_sep_sup_pan",
        {
            "10": l0 + nov_a + smid[:60] + nov_b + ancr[:685] + r0,
            "L0": l0, "n100": n100, "smid": smid, "ancR": ancr, "R0": r0,
            "alt2": smid[:60] + ancr[:685], "alt3": n100 + smid[30:] + ancr[:680],
        },
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("n100", "+"), ("smid", "+"), ("ancR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("n100", "+"), ("smid", "+"), ("ancR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("alt2", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("alt3", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_sep = _ev34(res_sep)
    ins_sep = _ins34(res_sep, ev_sep)
    pan34_sep = "".join(res_sep.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_sep)
    assert ev_sep.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in ins_sep)
    assert smid[:60] not in pan34_sep and smid[:30] not in pan34_sep and smid[30:59] not in pan34_sep
    assert res_sep.pan_seq.count(smid[:60]) == 1

    nov250 = make_complex_seq(250, seed=12401)
    sm120 = make_complex_seq(120, seed=12402)
    t25 = make_complex_seq(25, seed=12403)
    res_rip = _run_i8(
        "i8_p4_ripple_real_pan",
        {
            "10": l0 + nov_a + sm120[25:85] + nov_b + r0,
            "L0": l0, "nov250": nov250, "sm": sm120, "t25": t25, "R0": r0,
            "a2": sm120[:85], "a3": sm120[25:], "a4": sm120[:25] + sm120[55:],
        },
        sup6 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("sm", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("sm", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a2", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a3", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("nov250", "+"), ("a4", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_rip = _ev34(res_rip)
    ins_rip = _ins34(res_rip, ev_rip)
    pan34_rip = "".join(res_rip.pan_seq[r["pan_start"] : r["pan_end"]] for r in ins_rip)
    assert sm120[25:55] not in pan34_rip and sm120[55:85] not in pan34_rip
    assert sm120[:25] in pan34_rip and sm120[85:] in pan34_rip and t25 in pan34_rip


def test_iter9_hit_covers_subinterval_mappy_edge_matrix() -> None:
    """Iter 9 Part A: unified `hit_covers_subinterval_mappy` edge matrix:
    1. 1..49 bp SNP slivers (CLI_BUG3: 1 bp, 5 bp, 10 bp 2-SNP, right-del mirror,
       p_len=49 vs 50 at 5 vs 6 SNPs, and strand == -1 sliding window).
    2. Exact zero-penalty I-microhomology shifting (trial=22 fwd/rev/mirror,
       LIVE_C_TC_GG m_p==0, pa=AAA|sb=AAA 0.90 vs 0.95, mixed-case query_seq,
       I flanked by X blocked, tandem conservation GGT+A^7|A^7+CCG, A^10|A^10,
       PROBE B1d p_len=50 44bp+A^6|A^6+44bp at min_hit_len=50 & 0, internal 1bp I).
    3. Exact zero-penalty D-microhomology shifting (large-D 200 bp s20_pr_mir
       + and - strands & left-shift vs non-MH, small-D boundary MH vs internal 5D
       & tandem 7D conservation), D at window edge vs interior, soft-clipped edges,
       and bridged 'N'*200 30+30 on + and - strands.
    """
    # (1) 1..49 bp SNP slivers & sliding 50 bp window
    # 1 bp 1-SNP with adjacent 10 bp I on left (CLI_BUG3): [200, 250) has 49/50 = 98% >= 90%
    h_1bp_l = SimpleNamespace(q_st=0, q_en=410, r_st=0, r_en=400, strand=1, cs=":190+" + "a" * 10 + "*ac:209")
    assert hit_covers_subinterval_mappy(h_1bp_l, 200, 201, min_identity=0.90, min_hit_len=50) is True
    # Right-del mirror: 10 bp I on right, [151, 201) has 49/50 = 98%
    h_1bp_r = SimpleNamespace(q_st=0, q_en=410, r_st=0, r_en=400, strand=1, cs=":200*ac+" + "a" * 10 + ":199")
    assert hit_covers_subinterval_mappy(h_1bp_r, 200, 201, min_identity=0.90, min_hit_len=50) is True
    # 5 bp 1-SNP center sliver with 10 bp I immediately left (CLI_BUG3)
    h_5bp = SimpleNamespace(q_st=0, q_en=410, r_st=0, r_en=400, strand=1, cs=":190+" + "a" * 10 + ":2*ac:207")
    assert hit_covers_subinterval_mappy(h_5bp, 200, 205, min_identity=0.90, min_hit_len=50) is True
    # 10 bp 2-SNP sliver -> 48/50 = 96% >= 90%
    h_10bp = SimpleNamespace(q_st=0, q_en=410, r_st=0, r_en=410, strand=1, cs=":202*ac:4*ac:202")
    assert hit_covers_subinterval_mappy(h_10bp, 200, 210, min_identity=0.90, min_hit_len=50) is True
    # p_len=49 vs p_len=50 boundary at 5 SNPs (45/50=90% True) vs 6 SNPs (44/50=88% False)
    h_5s = SimpleNamespace(q_st=0, q_en=400, r_st=0, r_en=400, strand=1, cs=":205*ac:7*ac:7*ac:7*ac:7*ac:162")
    h_6s = SimpleNamespace(q_st=0, q_en=400, r_st=0, r_en=400, strand=1, cs=":205*ac:6*ac:6*ac:6*ac:6*ac:6*ac:159")
    assert hit_covers_subinterval_mappy(h_5s, 200, 249, min_identity=0.90, min_hit_len=50) is True
    assert hit_covers_subinterval_mappy(h_6s, 200, 249, min_identity=0.90, min_hit_len=50) is False
    assert hit_covers_subinterval_mappy(h_5s, 200, 250, min_identity=0.90, min_hit_len=50) is True
    assert hit_covers_subinterval_mappy(h_6s, 200, 250, min_identity=0.90, min_hit_len=50) is False
    # strand == -1 sliding window (cs tokens reverse into ascending query coords)
    h_5bp_rev = SimpleNamespace(q_st=0, q_en=410, r_st=0, r_en=400, strand=-1, cs=":207*ac:2+" + "a" * 10 + ":190")
    assert hit_covers_subinterval_mappy(h_5bp_rev, 200, 205, min_identity=0.90, min_hit_len=50) is True
    h_6s_rev = SimpleNamespace(q_st=0, q_en=400, r_st=0, r_en=400, strand=-1, cs=":159*ac:6*ac:6*ac:6*ac:6*ac:6*ac:205")
    assert hit_covers_subinterval_mappy(h_6s_rev, 200, 249, min_identity=0.90, min_hit_len=50) is False

    # (2) I-microhomology shifting & tandem conservation
    flank_l = make_complex_seq(200, seed=19001)
    flank_r = make_complex_seq(219, seed=19002)
    # trial=22: pa="ACCAATTAC" [200,209), sb="TC" [209,211), cs=":208+ct:220"
    q_t22 = flank_l + "ACCAATTAC" + "TC" + flank_r
    h_t22_fwd = SimpleNamespace(q_st=0, q_en=len(q_t22), r_st=0, r_en=len(q_t22) - 2, strand=1, cs=":208+ct:220")
    assert hit_covers_subinterval_mappy(h_t22_fwd, 200, 209, 0.90, min_hit_len=50, query_seq=q_t22) is True
    assert hit_covers_subinterval_mappy(h_t22_fwd, 209, 211, 0.90, min_hit_len=50, query_seq=q_t22) is False
    # strand == -1 trial=22
    h_t22_rev = SimpleNamespace(q_st=0, q_en=len(q_t22), r_st=0, r_en=len(q_t22) - 2, strand=-1, cs=":220+ag:208")
    assert hit_covers_subinterval_mappy(h_t22_rev, 200, 209, 0.90, min_hit_len=50, query_seq=q_t22) is True
    assert hit_covers_subinterval_mappy(h_t22_rev, 209, 211, 0.90, min_hit_len=50, query_seq=q_t22) is False
    # Mirror trial=22: sb="CT" [200,202), pb="CATTAACCA" [202,211), cs=":201+tc:227"
    q_t22_mir = flank_l + "CT" + "CATTAACCA" + flank_r
    h_t22_mir = SimpleNamespace(q_st=0, q_en=len(q_t22_mir), r_st=0, r_en=len(q_t22_mir) - 2, strand=1, cs=":201+tc:227")
    assert hit_covers_subinterval_mappy(h_t22_mir, 202, 211, 0.90, min_hit_len=50, query_seq=q_t22_mir) is True
    assert hit_covers_subinterval_mappy(h_t22_mir, 200, 202, 0.90, min_hit_len=50, query_seq=q_t22_mir) is False
    # LIVE_C_TC_GG (m_p == 0 on pa="C", sb="TC", mixed-case query_seq; avoid accidental TCTC at flank_l[-1])
    flank_l_g = flank_l[:-2] + "AG"
    flank_r_g = "GG" + flank_r[2:]
    q_c_tc = flank_l_g + "C" + "TC" + flank_r_g
    h_c_tc = SimpleNamespace(q_st=0, q_en=len(q_c_tc), r_st=0, r_en=len(q_c_tc) - 2, strand=1, cs=":200+ct:220")
    assert hit_covers_subinterval_mappy(h_c_tc, 200, 201, 0.90, min_hit_len=50, query_seq=q_c_tc) is True
    assert hit_covers_subinterval_mappy(h_c_tc, 201, 203, 0.90, min_hit_len=50, query_seq=q_c_tc) is False
    assert hit_covers_subinterval_mappy(h_c_tc, 200, 201, 0.90, min_hit_len=50, query_seq=q_c_tc.lower()) is True
    # pa="AAA" | sb="AAA": 47/50 = 94% -> True @ 0.90, False @ 0.95
    q_aaa = flank_l + "AAA" + "AAA" + flank_r
    h_aaa = SimpleNamespace(q_st=0, q_en=len(q_aaa), r_st=0, r_en=len(q_aaa) - 3, strand=1, cs=":200+aaa:222")
    assert hit_covers_subinterval_mappy(h_aaa, 200, 203, 0.90, min_hit_len=50, query_seq=q_aaa) is True
    assert hit_covers_subinterval_mappy(h_aaa, 200, 203, 0.95, min_hit_len=50, query_seq=q_aaa) is False
    # I flanked by X -> zero-penalty I+X <-> X+I swap credits X (not M): 48/50=96% True@0.90/0.96, False@0.97;
    # novel I(6)+X(1) & non-MH I flanked by M still remain False
    h_ix = SimpleNamespace(q_st=0, q_en=len(q_t22), r_st=0, r_en=len(q_t22) - 2, strand=1, cs=":207*ac+ct*ac:219")
    assert hit_covers_subinterval_mappy(h_ix, 200, 209, 0.90, min_hit_len=50, query_seq=q_t22) is True
    assert hit_covers_subinterval_mappy(h_ix, 200, 209, 0.96, min_hit_len=50, query_seq=q_t22) is True
    assert hit_covers_subinterval_mappy(h_ix, 200, 209, 0.97, min_hit_len=50, query_seq=q_t22) is False
    q_i6x = flank_l + "GCTAGC" + "T" + flank_r
    h_i6x = SimpleNamespace(q_st=0, q_en=len(q_i6x), r_st=0, r_en=len(q_i6x) - 6, strand=1, cs=":200+gctagc*at:219")
    assert hit_covers_subinterval_mappy(h_i6x, 200, 206, 0.90, min_hit_len=50, query_seq=q_i6x) is False
    # Tandem conservation controls (BOTH min_hit_len=50 AND min_hit_len=0!)
    # GGT+A^7 | A^7+CCG (43/50 = 86% < 90%) -> pa MUST return False
    q_tand7 = flank_l + "GGT" + ("A" * 7) + ("A" * 7) + "CCG" + flank_r
    h_tand7 = SimpleNamespace(q_st=0, q_en=len(q_tand7), r_st=0, r_en=len(q_tand7) - 7, strand=1, cs=":203+" + "a" * 7 + ":229")
    for mhl in (50, 0):
        assert hit_covers_subinterval_mappy(h_tand7, 200, 210, 0.90, min_hit_len=mhl, query_seq=q_tand7) is False
    # A^10 | A^10 (40/50 = 80% < 90%) -> pa MUST return False
    q_tand10 = flank_l + ("A" * 10) + ("A" * 10) + flank_r
    h_tand10 = SimpleNamespace(q_st=0, q_en=len(q_tand10), r_st=0, r_en=len(q_tand10) - 10, strand=1, cs=":200+" + "a" * 10 + ":229")
    for mhl in (50, 0):
        assert hit_covers_subinterval_mappy(h_tand10, 200, 210, 0.90, min_hit_len=mhl, query_seq=q_tand10) is False
    # PROBE B1d: p_len == 50 (44bp+A^6 | A^6+44bp, 44/50 = 88% < 90%) -> MUST return False at mhl=50 & 0!
    b44_l = make_complex_seq(44, seed=19003)
    b44_r = make_complex_seq(44, seed=19004)
    q_b1d = flank_l + b44_l + ("A" * 6) + ("A" * 6) + b44_r + flank_r
    h_b1d = SimpleNamespace(q_st=0, q_en=len(q_b1d), r_st=0, r_en=len(q_b1d) - 6, strand=1, cs=":244+" + "a" * 6 + ":269")
    for mhl in (50, 0):
        assert hit_covers_subinterval_mappy(h_b1d, 200, 250, 0.90, min_hit_len=mhl, query_seq=q_b1d) is False
    # Internal 1 bp I (non-homopolymer & homopolymer trapped inside [200, 206)) -> False
    q_int_i = flank_l + "CTGATC" + flank_r
    h_int_i = SimpleNamespace(q_st=0, q_en=len(q_int_i), r_st=0, r_en=len(q_int_i) - 1, strand=1, cs=":203+a:221")
    assert hit_covers_subinterval_mappy(h_int_i, 200, 206, 0.90, min_hit_len=50, query_seq=q_int_i) is False
    q_int_hp = flank_l + "CAAAAC" + flank_r
    h_int_hp = SimpleNamespace(q_st=0, q_en=len(q_int_hp), r_st=0, r_en=len(q_int_hp) - 1, strand=1, cs=":202+a:222")
    assert hit_covers_subinterval_mappy(h_int_hp, 200, 206, 0.90, min_hit_len=50, query_seq=q_int_hp) is False

    # (3) D-microhomology shifting (+ and - strands, large-D & small-D) & bridged guards
    d200_core = make_complex_seq(197, seed=19010)
    d200_mh = "CTG" + d200_core
    d200_nomh = "AAA" + d200_core
    s20_ctg = make_complex_seq(17, seed=19011) + "CTG"
    q_d_mir = make_complex_seq(200, seed=19012) + s20_ctg + make_complex_seq(200, seed=19013)
    # Right-shift large-D (200 bp) on strand == +1 (s20_pr_mir: ':150-<200D>:67-<200D>:203')
    h_d_mh_fwd = SimpleNamespace(
        q_st=0, q_en=420, r_st=0, r_en=820, strand=1,
        cs=":150-" + d200_core.lower() + "gat:67-" + d200_mh.lower() + ":203",
    )
    h_d_nomh_fwd = SimpleNamespace(
        q_st=0, q_en=420, r_st=0, r_en=820, strand=1,
        cs=":150-" + d200_core.lower() + "gat:67-" + d200_nomh.lower() + ":203",
    )
    assert hit_covers_subinterval_mappy(h_d_mh_fwd, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is True
    assert hit_covers_subinterval_mappy(h_d_nomh_fwd, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is False
    # Right-shift large-D on strand == -1 (deleted seq in cs is revcomp of forward query-oriented d_seq)
    h_d_mh_rev = SimpleNamespace(
        q_st=0, q_en=420, r_st=0, r_en=820, strand=-1,
        cs=":203-" + revcomp_dna(d200_mh).lower() + ":67-" + revcomp_dna(d200_core + "GAT").lower() + ":150",
    )
    h_d_nomh_rev = SimpleNamespace(
        q_st=0, q_en=420, r_st=0, r_en=820, strand=-1,
        cs=":203-" + revcomp_dna(d200_nomh).lower() + ":67-" + revcomp_dna(d200_core + "GAT").lower() + ":150",
    )
    assert hit_covers_subinterval_mappy(h_d_mh_rev, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is True
    assert hit_covers_subinterval_mappy(h_d_nomh_rev, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is False
    # Left-shift large-D (200 bp) at q_p == 203 (3 bp MH at start of [200, 220))
    s20_l_ctg = "CTG" + make_complex_seq(17, seed=19014)
    q_d_left = make_complex_seq(200, seed=19015) + s20_l_ctg + make_complex_seq(200, seed=19016)
    h_dl_mh = SimpleNamespace(q_st=0, q_en=420, r_st=0, r_en=620, strand=1, cs=":203-" + (d200_core + "CTG").lower() + ":217")
    h_dl_nomh = SimpleNamespace(q_st=0, q_en=420, r_st=0, r_en=620, strand=1, cs=":203-" + (d200_core + "AAA").lower() + ":217")
    assert hit_covers_subinterval_mappy(h_dl_mh, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_left) is True
    assert hit_covers_subinterval_mappy(h_dl_nomh, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_left) is False
    # Small-D (< min_hit_len) boundary MH vs internal 5D & tandem 7D conservation
    h_sd_mh = SimpleNamespace(q_st=0, q_en=420, r_st=0, r_en=423, strand=1, cs=":217-ctg:203")
    assert hit_covers_subinterval_mappy(h_sd_mh, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is True
    h_sd_int5 = SimpleNamespace(q_st=0, q_en=420, r_st=0, r_en=425, strand=1, cs=":210-acgta:210")
    assert hit_covers_subinterval_mappy(h_sd_int5, 200, 220, 0.90, min_hit_len=50, query_seq=q_d_mir) is False
    # Tandem 7D: zero-penalty D(7) shifts cleanly out of [200, 217) -> FWD == REV == True at mhl=50 & 0!
    q_tand7d = flank_l + "GGT" + ("A" * 14) + "CCG" + flank_r
    h_tand7d = SimpleNamespace(q_st=0, q_en=len(q_tand7d), r_st=0, r_en=len(q_tand7d) + 7, strand=1, cs=":210-aaaaaaa:229")
    h_tand7d_rev = SimpleNamespace(q_st=0, q_en=len(q_tand7d), r_st=0, r_en=len(q_tand7d) + 7, strand=-1, cs=":229-ttttttt:210")
    for mhl in (50, 0):
        assert hit_covers_subinterval_mappy(h_tand7d, 200, 217, 0.90, min_hit_len=mhl, query_seq=q_tand7d) is True
        assert hit_covers_subinterval_mappy(h_tand7d_rev, 200, 217, 0.90, min_hit_len=mhl, query_seq=q_tand7d) is True
    # D right at window boundary w_s / w_e (NOT penalized) vs strictly inside (w_s+1 / w_e-1, penalized)
    h_d_edge = SimpleNamespace(q_st=200, q_en=250, r_st=0, r_en=60, strand=1, cs=":50-aaaaaaaaaa")
    assert hit_covers_subinterval_mappy(h_d_edge, 200, 220, 0.90, min_hit_len=50) is True
    h_d_inside = SimpleNamespace(q_st=200, q_en=250, r_st=0, r_en=60, strand=1, cs=":49-aaaaaaaaaa:1")
    assert hit_covers_subinterval_mappy(h_d_inside, 200, 220, 0.90, min_hit_len=50) is False
    # Soft-clipped seg_q_st > p_s / seg_q_en < p_e
    h_sc_l = SimpleNamespace(q_st=204, q_en=260, r_st=0, r_en=56, strand=1, cs=":56")
    assert hit_covers_subinterval_mappy(h_sc_l, 200, 220, 0.90, min_hit_len=50) is False
    h_sc_r = SimpleNamespace(q_st=140, q_en=216, r_st=0, r_en=76, strand=1, cs=":76")
    assert hit_covers_subinterval_mappy(h_sc_r, 200, 220, 0.90, min_hit_len=50) is False
    # Bridged 'N'*200 30+30 (piece 0 has 30 bp, piece 1 has 30 bp, total 60 >= 50) -> MUST return False on + & -!
    h_br_pos = SimpleNamespace(q_st=0, q_en=60, r_st=270, r_en=530, strand=1, cs=":30-" + "n" * 200 + ":30")
    h_br_neg = SimpleNamespace(q_st=0, q_en=60, r_st=270, r_en=530, strand=-1, cs=":30-" + "n" * 200 + ":30")
    assert hit_covers_subinterval_mappy(h_br_pos, 0, 25, 0.90, t_starts=[0, 500], t_ends=[300, 800], min_hit_len=50) is False
    assert hit_covers_subinterval_mappy(h_br_neg, 0, 25, 0.90, t_starts=[0, 500], t_ends=[300, 800], min_hit_len=50) is False


def test_iter9_stage55_self_pool_symmetric_grid_and_codrop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iter 9 Part B.2: Stage 5.5 rep-local `self_*` pool + 4-cutpoint symmetric grid
    + `checked_unmsk_by_orig`/`tested_pairs_by_orig` + hit-verified co-dropping:
    1. HITS_1_SELF_HIJACK_GHOST_SUPH2, DIAG_F2_ADV1, LIVE_F2_ADV3C, HITS_4_FALSE_DROP_25BP_ADJ_MASKED.
    2. LIVE_PARTIAL_MASKED_HIJACKS_CTX_U, CROSS_WAVE_ISO_45BP_REQ0.
    3. CLI_BUG2 fwd+mirror, s20 + Pright(50 nov + 250 msk) fwd+mirror (incl. CTG D-MH),
       3-block 60+120+60 sandwich fwd+mirror, ADV8b, ADV9 (ctx_lu), ADV9b (ctx_ru),
       ADV10 (isolated 30 bp), ADV11 (3-piece 20+20+20 RC), Attack S3 (44+6 unmasked neighbor).
    4. join_before='spacer' boundary, iv_queue need_rescreen intra-piece nested self-dedup,
       mid-batch wave-seal carryover with untouched_short_entries, and 25/25 asm20 large-D sweep.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=19100 + i) for i in range(1, 7)}
    sup8 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 9)
    ]

    def _run_p(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        dedup_id: float = 0.90,
        min_sv_len: int = 50,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode="pan", dedup_id=dedup_id, min_sv_len=min_sv_len
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix, gfa_path=cfg.gfa, ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix, spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _ev34(res):
        return next(e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4")

    def _ins34(res, ev):
        return [r for r in res.inserts_rows if r["event_id"] == ev.event_id]

    def _pan34(res, ev) -> str:
        return "".join(res.pan_seq[r["pan_start"] : r["pan_end"]] for r in _ins34(res, ev))

    l0 = make_complex_seq(1500, seed=19111)
    r0 = make_complex_seq(1500, seed=19112)
    nov_l = make_complex_seq(250, seed=19113)
    nov_r = make_complex_seq(250, seed=19114)
    nov_a = make_complex_seq(200, seed=19115)
    nov_b = make_complex_seq(200, seed=19116)
    s30a = make_complex_seq(30, seed=19117)
    s30b = make_complex_seq(30, seed=19118)
    tail50 = make_complex_seq(50, seed=19119)

    # (1a) HITS_1_SELF_HIJACK_GHOST_SUPH2: own novL/novR must never self-hijack s30a+s30b
    res_h1 = _run_p(
        "i9_hits1_self_hijack",
        {
            "10": l0 + nov_a + s30a + s30b + nov_b + r0,
            "L0": l0, "novL": nov_l, "s30a": s30a, "s30b": s30b, "novR": nov_r, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("s30a", "+"), ("s30b", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("s30a", "+"), ("s30b", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s30a", "+"), ("s30b", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("s30a", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_h1 = _ev34(res_h1)
    assert ev_h1.genotypes["Hap2#1"] == "."
    assert all("Hap2#1" not in r["members"].split(",") for r in _ins34(res_h1, ev_h1))
    p34_h1 = _pan34(res_h1, ev_h1)
    assert s30a not in p34_h1 and s30b not in p34_h1 and nov_l in p34_h1 and nov_r in p34_h1

    # (1b) DIAG_F2_ADV1: P4_RIPPLE with novel 250 bp flanks on BOTH sides
    sm120 = make_complex_seq(120, seed=19121)
    t25_d1 = make_complex_seq(25, seed=19120)
    res_d1 = _run_p(
        "i9_diag_f2_adv1",
        {
            "10": l0 + nov_a + sm120[25:85] + nov_b + r0,
            "L0": l0, "novL": nov_l, "sm": sm120, "novR": nov_r, "t25": t25_d1, "R0": r0,
            "a2": sm120[:85], "a3": sm120[25:], "a4": sm120[:25] + sm120[55:],
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("sm", "+"), ("novR", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("sm", "+"), ("novR", "+"), ("t25", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("a2", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("a3", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("a4", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    p34_d1 = _pan34(res_d1, _ev34(res_d1))
    assert sm120[25:55] not in p34_d1 and sm120[55:85] not in p34_d1
    assert sm120[:25] in p34_d1 and sm120[85:] in p34_d1 and nov_l in p34_d1 and nov_r in p34_d1

    # (1c) LIVE_F2_ADV3C: co-linear Lsm(100)+s20a(20) adjacent to unmasked novR(250)
    lsm100 = make_complex_seq(100, seed=19122)
    s20a = make_complex_seq(20, seed=19123)
    res_3c = _run_p(
        "i9_live_f2_adv3c",
        {
            "10": l0 + nov_a + lsm100 + s20a + nov_b + r0,
            "L0": l0, "novL": nov_l, "Lsm": lsm100, "s20a": s20a, "novR": nov_r, "t50": tail50, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("Lsm", "+"), ("s20a", "+"), ("novR", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("Lsm", "+"), ("s20a", "+"), ("novR", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novL", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    p34_3c = _pan34(res_3c, _ev34(res_3c))
    assert s20a not in p34_3c and nov_l in p34_3c and nov_r in p34_3c and tail50 in p34_3c

    # (1d) HITS_4_FALSE_DROP_25BP_ADJ_MASKED: isolated 25 bp novel s25a MUST survive
    s25a = make_complex_seq(25, seed=19124)
    res_h4 = _run_p(
        "i9_hits4_no_false_drop_25bp",
        {
            "10": l0 + nov_a + s25a + nov_b + r0,
            "L0": l0, "s25a": s25a, "novR": nov_r, "t50": tail50, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s25a", "+"), ("novR", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s25a", "+"), ("novR", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("novR", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    assert s25a in _pan34(res_h4, _ev34(res_h4))

    # (2a) LIVE_PARTIAL_MASKED_HIJACKS_CTX_U: Pright = 50 nov + 250 msk (piece_end_masked == (False, True))
    pr_nov50 = make_complex_seq(50, seed=19125)
    pr_msk250 = make_complex_seq(250, seed=19126)
    pr_300 = pr_nov50 + pr_msk250
    res_pm = _run_p(
        "i9_live_partial_masked",
        {
            "10": l0 + nov_a + s30a + s30b + nov_b + pr_msk250 + r0,
            "L0": l0, "s30a": s30a, "s30b": s30b, "Pr": pr_300, "t50": tail50, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s30a", "+"), ("s30b", "+"), ("Pr", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s30a", "+"), ("s30b", "+"), ("Pr", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s30a", "+"), ("Pr", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("Pr", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    p34_pm = _pan34(res_pm, _ev34(res_pm))
    assert s30a not in p34_pm and s30b not in p34_pm and pr_nov50 in p34_pm and tail50 in p34_pm

    # (2b) CROSS_WAVE_ISO_45BP_REQ0: isolated 45 bp < min_sv_len MUST NOT drop across waves
    s45 = make_complex_seq(45, seed=19127)
    res_45 = _run_p(
        "i9_cross_wave_iso_45bp",
        {
            "10": l0 + r0,
            "10b": nov_a + s45 + nov_b,
            "L0": l0, "s45": s45, "t50": tail50, "R0": r0,
        },
        sup8 + [
            (f"SupB{k}#1#Chr01", [("1", "+"), ("2", "+"), ("10b", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
            for k in range(1, 6)
        ] + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s45", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s45", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s45", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    p34_45 = _pan34(res_45, _ev34(res_45))
    assert s45 in p34_45 and tail50 in p34_45

    # (3a) CLI_BUG2 forward + exact mirror symmetry
    s0_b2 = make_complex_seq(20, seed=19131)
    s1_b2 = make_complex_seq(20, seed=19132)
    r0_50 = make_complex_seq(50, seed=19133)
    ta_60 = make_complex_seq(60, seed=19134)
    tb_60 = make_complex_seq(60, seed=19135)
    te_1500 = make_complex_seq(1500, seed=19136)
    res_b2_fwd = _run_p(
        "i9_cli_bug2_fwd",
        {"10": l0 + nov_a + s0_b2 + s1_b2 + r0_50 + nov_b + te_1500, "L0": l0, "s0": s0_b2, "s1": s1_b2, "R0": r0_50, "ta": ta_60, "tb": tb_60, "TE": te_1500},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s0", "+"), ("s1", "+"), ("R0", "+"), ("ta", "+"), ("tb", "+"), ("TE", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s0", "+"), ("s1", "+"), ("R0", "+"), ("ta", "+"), ("tb", "+"), ("TE", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s1", "+"), ("R0", "+"), ("TE", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s0", "+"), ("R0", "+"), ("TE", "+"), ("5", "+"), ("6", "+")]),
            ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s0", "+"), ("s1", "+"), ("R0", "+"), ("TE", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_b2f = _ev34(res_b2_fwd)
    assert ev_b2f.genotypes["Hap4#1"] == "." and res_b2_fwd.pan_seq.count(s0_b2) == 1 and res_b2_fwd.pan_seq.count(s1_b2) == 1

    res_b2_mir = _run_p(
        "i9_cli_bug2_mir",
        {"10": te_1500 + nov_b + r0_50 + s0_b2 + s1_b2 + nov_a + l0, "TE": te_1500, "tb": tb_60, "ta": ta_60, "R0": r0_50, "s0": s0_b2, "s1": s1_b2, "L0": l0},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("TE", "+"), ("tb", "+"), ("ta", "+"), ("R0", "+"), ("s0", "+"), ("s1", "+"), ("L0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("TE", "+"), ("tb", "+"), ("ta", "+"), ("R0", "+"), ("s0", "+"), ("s1", "+"), ("L0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("TE", "+"), ("R0", "+"), ("s1", "+"), ("L0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("TE", "+"), ("R0", "+"), ("s0", "+"), ("L0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap4#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("TE", "+"), ("R0", "+"), ("s0", "+"), ("s1", "+"), ("L0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    ev_b2m = _ev34(res_b2_mir)
    assert ev_b2m.genotypes["Hap4#1"] == "." and res_b2_mir.pan_seq.count(s0_b2) == 1 and res_b2_mir.pan_seq.count(s1_b2) == 1

    # (3b) s20 + Pright(50 nov + 250 msk) forward + mirror (with explicit CTG D-microhomology!)
    s20_ctg = make_complex_seq(17, seed=19141) + "CTG"
    nov_b_ctg = "CTG" + make_complex_seq(197, seed=19142)
    res_sp_fwd = _run_p(
        "i9_s20_pr_fwd",
        {
            "10": l0 + nov_a + s20_ctg + pr_nov50 + nov_b_ctg + pr_msk250 + r0,
            "L0": l0, "s20": s20_ctg, "Pr": pr_300, "t50": tail50, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s20", "+"), ("Pr", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("s20", "+"), ("Pr", "+"), ("t50", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("Pr", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    assert s20_ctg not in _pan34(res_sp_fwd, _ev34(res_sp_fwd))

    # Mirror: Pleft(250 msk + 50 nov) + s20_ctg + R0(1500), with nov_b_ctg immediately right of s20_ctg in Sup
    pl_300 = pr_msk250 + pr_nov50
    res_sp_mir = _run_p(
        "i9_s20_pr_mir",
        {
            "10": l0 + pr_msk250 + nov_a + pr_nov50 + s20_ctg + nov_b_ctg + r0,
            "L0": l0, "t50": tail50, "Pl": pl_300, "s20": s20_ctg, "R0": r0,
        },
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("t50", "+"), ("Pl", "+"), ("s20", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("t50", "+"), ("Pl", "+"), ("s20", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("L0", "+"), ("Pl", "+"), ("R0", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    assert s20_ctg not in _pan34(res_sp_mir, _ev34(res_sp_mir))

    # (3c) 3-block 60+120+60 ectopic sandwich forward + mirror (co-dropping peels all 3 blocks cleanly)
    gb_dir = SyntheticGFAFixtureBuilder()
    for i in range(1, 5):
        gb_dir.add_node(str(i), make_complex_seq(200, seed=19150 + i))
    gb_dir.add_p_path("Ref#1#Chr01", [(str(i), "+") for i in range(1, 5)])
    g_dir, cfg_dir = _build_graph_and_config(
        tmp_path, "i9_direct_s55", gb_dir, [(str(i), "+") for i in range(1, 5)], dedup_mode="pan", min_sv_len=50
    )
    empty_cen = CentromereOutput((), (), (), (), ())

    def _mk_rep(idx: int, pieces: tuple[Piece, ...], sup_cnt: int, cid: str) -> ParallelGroupSolution:
        tot = sum(len(p.seq) for p in pieces)
        ex = ExtractedCandidate(cid, None, "H#1", "H#1#Chr01", 0, 1, idx, (), (), "", ())
        rep = ClusterRepresentative(
            rep_idx=1, is_synthetic=True, seq="", length_bp_no_n=tot, pieces=pieces,
            support_haps={f"S{k}#1" for k in range(sup_cnt)}, token_str=f"t{idx}", min_hap="S0#1",
            exemplar=ex, member_occurrences=[], nested_choices_by_cand_id={},
            reconstructed_len_by_cand_id={}, piece_cand_ids=tuple(frozenset({cid}) for _ in pieces),
            support_hap_count=sup_cnt,
        )
        return ParallelGroupSolution(
            0, 1, False, "", (rep,), {cid: 1}, {cid: ()}, {cid: tot}, {cid: (False, False, False)}
        )

    def _mk_contig_pieces(seqs: Sequence[str], strand: str = "+") -> tuple[Piece, ...]:
        out: list[Piece] = []
        if strand == "+":
            cur = 0
            for s in seqs:
                out.append(Piece("H#1", "H#1#Chr01", cur, cur + len(s), "+", s, "source_contiguous", ("H#1",)))
                cur += len(s)
        else:
            cur = sum(len(s) for s in seqs)
            for s in seqs:
                out.append(Piece("H#1", "H#1#Chr01", cur - len(s), cur, "-", s, "source_contiguous", ("H#1",)))
                cur -= len(s)
        return tuple(out)

    blk1_ps = [make_complex_seq(20, seed=19160 + i) for i in range(3)]
    blk2_ps = [make_complex_seq(40, seed=19165 + i) for i in range(3)]
    blk3_ps = [make_complex_seq(20, seed=19170 + i) for i in range(3)]
    pad_s = [make_complex_seq(200, seed=19175 + i) for i in range(4)]
    sup_sand = l0 + pad_s[0] + "".join(blk1_ps) + pad_s[1] + "".join(blk2_ps) + pad_s[2] + "".join(blk3_ps) + pad_s[3] + r0
    q_sand_seqs = [l0, *blk1_ps, *blk2_ps, *blk3_ps, r0]
    for is_mir, q_list in ((False, q_sand_seqs), (True, [revcomp_dna(x) for x in reversed(q_sand_seqs)])):
        s_sup = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_sand), "+", sup_sand, "source_contiguous", ("Sup#1",)),), 10, f"sup_sw_{is_mir}")
        s_q = _mk_rep(1, _mk_contig_pieces(q_list, strand="-" if is_mir else "+"), 2, f"q_sw_{is_mir}")
        sols = {(0, 1): s_sup, (1, 2): s_q}
        _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols}, sols, empty_cen)  # type: ignore[arg-type]
        assert s_q.non_backbone_reps == (), f"3-block sandwich failed (mir={is_mir})"

    # (3d) ADV8b (partially-masked left (T,F)), ADV9 (ctx_lu), ADV9b (ctx_ru), ADV10 (isolated 30bp),
    # ADV11 (3-piece 20+20+20 RC), Attack S3 (unmasked 44+6 piece adjacent to s20@SiteB)
    # ADV8b: Pleft(250 msk + 50 nov, (True, False)) + s30a + s30b + R0(1500)
    pl_msk_nov = pr_msk250 + pr_nov50
    sup_8b = l0 + pr_msk250 + nov_a + s30a + s30b + nov_b + r0
    s_sup_8b = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_8b), "+", sup_8b, "source_contiguous", ("Sup#1",)),), 10, "sup8b")
    s_q_8b = _mk_rep(1, _mk_contig_pieces([pl_msk_nov, s30a, s30b, r0]), 2, "q8b")
    sols_8b = {(0, 1): s_sup_8b, (1, 2): s_q_8b}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_8b}, sols_8b, empty_cen)  # type: ignore[arg-type]
    assert [p.seq for p in s_q_8b.non_backbone_reps[0].pieces] == [pr_nov50]

    # ADV9 (ctx_lu) & ADV9b (ctx_ru)
    sup_9 = l0 + s20a + nov_a + r0
    s_sup_9 = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_9), "+", sup_9, "source_contiguous", ("Sup#1",)),), 10, "sup9")
    s_q_9 = _mk_rep(1, _mk_contig_pieces([l0, s20a, r0]), 2, "q9")
    sols_9 = {(0, 1): s_sup_9, (1, 2): s_q_9}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_9}, sols_9, empty_cen)  # type: ignore[arg-type]
    assert s_q_9.non_backbone_reps == ()

    sup_9b = l0 + nov_a + s20a + r0
    s_sup_9b = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_9b), "+", sup_9b, "source_contiguous", ("Sup#1",)),), 10, "sup9b")
    s_q_9b = _mk_rep(1, _mk_contig_pieces([l0, s20a, r0]), 2, "q9b")
    sols_9b = {(0, 1): s_sup_9b, (1, 2): s_q_9b}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_9b}, sols_9b, empty_cen)  # type: ignore[arg-type]
    assert s_q_9b.non_backbone_reps == ()

    # ADV10: isolated 30 bp ectopic match MUST be preserved
    sup_10 = l0 + nov_a + s30a + nov_b + r0
    s_sup_10 = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_10), "+", sup_10, "source_contiguous", ("Sup#1",)),), 10, "sup10")
    s_q_10 = _mk_rep(1, _mk_contig_pieces([l0, s30a, r0]), 2, "q10")
    sols_10 = {(0, 1): s_sup_10, (1, 2): s_q_10}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_10}, sols_10, empty_cen)  # type: ignore[arg-type]
    assert [p.seq for p in s_q_10.non_backbone_reps[0].pieces] == [s30a]

    # ADV11: 3-piece 20+20+20 RC separated Sup
    rc20_a, rc20_b, rc20_c = make_complex_seq(20, seed=19181), make_complex_seq(20, seed=19182), make_complex_seq(20, seed=19183)
    sup_11 = l0 + nov_a + revcomp_dna(rc20_a + rc20_b + rc20_c) + nov_b + r0
    s_sup_11 = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_11), "+", sup_11, "source_contiguous", ("Sup#1",)),), 10, "sup11")
    s_q_11 = _mk_rep(1, _mk_contig_pieces([l0, rc20_a, rc20_b, rc20_c, r0]), 2, "q11")
    sols_11 = {(0, 1): s_sup_11, (1, 2): s_q_11}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_11}, sols_11, empty_cen)  # type: ignore[arg-type]
    assert s_q_11.non_backbone_reps == ()

    # Attack S3: unmasked 50 bp piece (44 bp novel + 6 bp matching SiteB) adjacent to s20@SiteB
    u44 = make_complex_seq(44, seed=19184)
    site_b_6 = make_complex_seq(6, seed=19185)
    site_b_tail = make_complex_seq(100, seed=19186)
    sup_s3 = l0 + nov_a + site_b_6 + s20a + site_b_tail + nov_b + r0
    s_sup_s3 = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_s3), "+", sup_s3, "source_contiguous", ("Sup#1",)),), 10, "sups3")
    s_q_s3 = _mk_rep(1, _mk_contig_pieces([u44 + site_b_6, s20a, site_b_tail]), 2, "qs3")
    sols_s3 = {(0, 1): s_sup_s3, (1, 2): s_q_s3}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_s3}, sols_s3, empty_cen)  # type: ignore[arg-type]
    assert [p.seq for p in s_q_s3.non_backbone_reps[0].pieces] == [u44 + site_b_6]

    # (4a) join_before="spacer" strictly blocks cross-spacer flank & short-run leak
    sp_p1 = Piece("H#1", "H#1#Chr01", 0, 30, "+", s30a, "source_contiguous", ("H#1",))
    sp_p2 = Piece("H#1", "H#1#Chr01", 30, 60, "+", s30b, "spacer", ("H#1",))
    s_sup_sp = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, 60, "+", s30a + s30b, "source_contiguous", ("Sup#1",)),), 10, "supsp")
    s_q_sp = _mk_rep(1, (sp_p1, sp_p2), 2, "qsp")
    sols_sp = {(0, 1): s_sup_sp, (1, 2): s_q_sp}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_sp}, sols_sp, empty_cen)  # type: ignore[arg-type]
    assert [p.seq for p in s_q_sp.non_backbone_reps[0].pieces] == [s30a, s30b]

    # (4b) iv_queue need_rescreen intra-piece nested self-dedup: A + T + X + A + X -> [A, X]
    seq_a_n = make_complex_seq(400, seed=19191)
    seq_t_n = make_complex_seq(400, seed=19192)
    seq_x_n = make_complex_seq(400, seed=19193)
    s_sup_iv = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, 400, "+", seq_t_n, "source_contiguous", ("Sup#1",)),), 10, "supiv")
    q_iv_seq = seq_a_n + seq_t_n + seq_x_n + seq_a_n + seq_x_n
    s_q_iv = _mk_rep(1, (Piece("H#1", "H#1#Chr01", 0, len(q_iv_seq), "+", q_iv_seq, "source_contiguous", ("H#1",)),), 2, "qiv")
    sols_iv = {(0, 1): s_sup_iv, (1, 2): s_q_iv}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_iv}, sols_iv, empty_cen)  # type: ignore[arg-type]
    assert [p.seq for p in s_q_iv.non_backbone_reps[0].pieces] == [seq_a_n, seq_x_n]

    # (4c) Mid-batch wave-seal carryover with untouched_short_entries
    monkeypatch.setattr(linearizer_mod, "_PAN_DEDUP_WAVE_BUDGET_BP", 600)
    monkeypatch.setattr(linearizer_mod, "_PAN_DEDUP_QUERY_BATCH_BP", 10000)
    c_l = make_complex_seq(350, seed=19194)
    c_r = make_complex_seq(350, seed=19195)
    s_sup_co = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(c_l) + 60 + len(c_r), "+", c_l + s30a + s30b + c_r, "source_contiguous", ("Sup#1",)),), 10, "supco")
    s_q_co = _mk_rep(1, _mk_contig_pieces([c_l, s30a, s30b, c_r]), 2, "qco")
    sols_co = {(0, 1): s_sup_co, (1, 2): s_q_co}
    _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_co}, sols_co, empty_cen)  # type: ignore[arg-type]
    assert s_q_co.non_backbone_reps == ()
    monkeypatch.undo()

    # (4d) 25/25 asm20 large-D bridging false-drop sweep (s_len=25..45 x gap=50..200)
    for si, s_len in enumerate((25, 30, 35, 40, 45)):
        for gi, gap_len in enumerate((50, 75, 100, 150, 200)):
            s_iso = make_complex_seq(s_len, seed=19200 + si * 10 + gi)
            g_a = make_complex_seq(gap_len, seed=19300 + si * 10 + gi)
            g_b = make_complex_seq(gap_len, seed=19400 + si * 10 + gi)
            sup_sw = l0 + g_a + s_iso + g_b + r0
            s_sup_sw = _mk_rep(0, (Piece("Sup#1", "Sup#1#Chr01", 0, len(sup_sw), "+", sup_sw, "source_contiguous", ("Sup#1",)),), 10, f"sw_{si}_{gi}")
            s_q_sw = _mk_rep(1, _mk_contig_pieces([l0, s_iso, r0]), 2, f"qw_{si}_{gi}")
            sols_sw = {(0, 1): s_sup_sw, (1, 2): s_q_sw}
            _deduplicate_representatives_across_chromosome(g_dir, cfg_dir, [(0, 1), (1, 2)], {p: GroupRepeatGuardResult() for p in sols_sw}, sols_sw, empty_cen)  # type: ignore[arg-type]
            assert [p.seq for p in s_q_sw.non_backbone_reps[0].pieces] == [s_iso], (
                f"asm20 large-D false-drop at s_len={s_len}, gap={gap_len}"
            )


def test_iter9_jacobi_deferred_benign_revocation_and_cross_interaction(
    tmp_path: Path,
) -> None:
    """Iter 9 Part B.1 + Combined Cross-Interaction Regressions:
    1. CLI_BUG1 (1 bp), ATK1_s4 (4 bp), ATK4_WITH_Hap3 (orig & non-MH), ATK2c (--min-sv-len 150).
    2. FLANKED_SB_1HOP, FLANKED_SB_2HOP, 1SIDE_SOFTCLIP_4BP_PARTIAL/MISSING,
       ZERO_EXCL_PURE_AC_SB4, ZERO_EXCL_PURE_AC_G40, P2A_WHOLE_PLUS_BENIGN_NOVEL (10 kb unsplit!).
    3. Cross-module combined proofs: LIVE_C_TC_GG (pa='C', sb='TC'), trial=22 (0.90/0.95/0.99
       + fwd==mir), DEFAULT090_2HOP fwd==mir, CLI_BUG3 (1 bp, 5 bp, 10 bp 2-SNP, right-del mirror).
    4. 6/6 cascade permutation invariance + compact deterministic PGGB/SNP-sliver/2-hop fuzz.
    """
    ref6 = [(str(i), "+") for i in range(1, 7)]
    base_ref6 = {str(i): make_complex_seq(400, seed=19500 + i) for i in range(1, 7)}
    sup8 = [
        (f"Sup{k}#1#Chr01", [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+")])
        for k in range(1, 9)
    ]

    def _run_j(
        tag: str,
        extra_nodes: dict[str, str],
        paths: list[tuple[str, list[tuple[str, str]]]],
        mode: str = "pan",
        dedup_id: float = 0.90,
        min_sv_len: int = 50,
    ):
        gb = SyntheticGFAFixtureBuilder()
        for nid, sq in base_ref6.items():
            gb.add_node(nid, sq)
        for nid, sq in extra_nodes.items():
            gb.add_node(nid, sq)
        gb.add_p_path("Ref#1#Chr01", ref6)
        for pname, steps in paths:
            gb.add_p_path(pname, steps)
        graph, cfg = _build_graph_and_config(
            tmp_path, tag, gb, ref6, dedup_mode=mode, dedup_id=dedup_id, min_sv_len=min_sv_len
        )
        res = run_chromosome_pipeline(graph, cfg)  # type: ignore[arg-type]
        write_build_outputs(res, cfg)
        v = verify_build_outputs(
            cfg.out_prefix, gfa_path=cfg.gfa, ref_fa_path=cfg.ref_fa,
            backbone_prefix=cfg.backbone_prefix, spacer_n=cfg.spacer_n,
        )
        assert v.passed, f"{tag} verify failed: {v.errors}"
        return res

    def _evs34(res):
        return [e for e in res.sv_events if e.event_type == "INS" and e.anchors == "3..4"]

    def _ins_sig(res, ev):
        return [
            (r["pan_end"] - r["pan_start"], r["members"], r["join_before"])
            for r in res.inserts_rows
            if r["event_id"] == ev.event_id
        ]

    def _assert_hap_clean(res, hap: str = "Hap2#1") -> None:
        evs = _evs34(res)
        if not evs:
            return
        ev = evs[0]
        assert ev.genotypes[hap] == ".", f"Expected {hap} == '.', got {ev.genotypes[hap]} ({ev.alleles})"
        assert all(
            hap not in r["members"].split(",")
            for r in res.inserts_rows
            if r["event_id"] == ev.event_id
        )

    s11 = make_complex_seq(1500, seed=19511)
    s12 = make_complex_seq(1500, seed=19512)
    ga5 = make_complex_seq(5000, seed=19513)
    ga10 = make_complex_seq(10000, seed=19514)
    ta50 = make_complex_seq(50, seed=19515)
    tb50 = make_complex_seq(50, seed=19516)

    # (1a) CLI_BUG1 (1 bp) & ATK1_s4 (4 bp)
    for tag_b1, sb_seq in (("i9_cli_bug1_s1", "A"), ("i9_atk1_s4", "ACGT")):
        res_b1 = _run_j(
            tag_b1,
            {"10": s11 + ga5 + ta50 + s12, "11": s11, "sb": sb_seq, "gA": ga5, "TE": ta50, "12": s12},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
        )
        _assert_hap_clean(res_b1, "Hap2#1")

    # (1b) ATK4_WITH_Hap3 (orig & non-MH)
    for tag_a4, g40 in (
        ("i9_atk4_orig", make_complex_seq(40, seed=19520)),
        ("i9_atk4_nomh", "G" + make_complex_seq(38, seed=19521) + "C"),
    ):
        g40_d5 = g40[5:]
        res_a4 = _run_j(
            tag_a4,
            {"10": s11 + ta50 + g40_d5 + tb50 + s12, "11": s11, "ta": ta50, "g40": g40, "tb": tb50, "12": s12, "alt2": g40_d5},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("g40", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("g40", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("alt2", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g40", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
        )
        _assert_hap_clean(res_a4, "Hap2#1")

    # (1c) ATK2c (--min-sv-len 150, ref mode): enclosing [4940, 5060) revokes Hap2 on [5000, 5001)
    te150 = make_complex_seq(150, seed=19522)
    ga10_d1 = ga10[:5000] + ga10[5001:]
    ga10_enc = ga10[:4940] + ga10[5060:]
    res_2c = _run_j(
        "i9_atk2c_min150_ref",
        {"11": s11, "gA": ga10, "TE150": te150, "12": s12, "alt2": ga10_d1, "alt3": ga10_enc},
        [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE150", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE150", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("TE150", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
        mode="ref",
        min_sv_len=150,
    )
    sig_2c = _ins_sig(res_2c, _evs34(res_2c)[0])
    assert (1, "Hap1a#1,Hap1b#1", "source_contiguous") in sig_2c

    # (2a) FLANKED_SB_1HOP & FLANKED_SB_2HOP
    p10a, p10b = make_complex_seq(10, seed=19523), make_complex_seq(10, seed=19524)
    res_1hop = _run_j(
        "i9_flanked_sb_1hop",
        {"10": s11 + ta50 + tb50 + s12, "11": s11, "ta": ta50, "sb": "ACGT", "tb": tb50, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("sb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("sb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_1hop, "Hap2#1")

    res_2hop = _run_j(
        "i9_flanked_sb_2hop",
        {"10": s11 + ta50 + p10a + p10b + tb50 + s12, "11": s11, "ta": ta50, "p10a": p10a, "sb": "ACGT", "p10b": p10b, "tb": tb50, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p10a", "+"), ("sb", "+"), ("p10b", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p10a", "+"), ("sb", "+"), ("p10b", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p10a", "+"), ("p10b", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("p10a", "+"), ("sb", "+"), ("p10b", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_2hop, "Hap2#1")

    # (2b) 1SIDE_SOFTCLIP_4BP_PARTIAL & 1SIDE_SOFTCLIP_4BP_MISSING
    ga5_sc = ga5[:5] + ga5[9:]
    res_sc_p = _run_j(
        "i9_1side_softclip_partial",
        {"10": s11 + ta50 + ga5_sc + s12, "11": s11, "ta": ta50, "gA": ga5, "12": s12, "alt2": ga5_sc},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("alt2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_sc_p, "Hap2#1")

    p5_sc = make_complex_seq(5, seed=19525)
    res_sc_m = _run_j(
        "i9_1side_softclip_missing",
        {"10": s11 + ta50 + p5_sc + ga5 + s12, "11": s11, "ta": ta50, "p5": p5_sc, "sb": "ACGT", "gA": ga5, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p5", "+"), ("sb", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p5", "+"), ("sb", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p5", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("p5", "+"), ("sb", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_sc_m, "Hap2#1")

    # (2c) ZERO_EXCL_PURE_AC_SB4 & ZERO_EXCL_PURE_AC_G40 (zero exclusions, pure active_child_solutions!)
    ins_l200, ins_r200 = make_complex_seq(200, seed=19526), make_complex_seq(200, seed=19527)
    res_ac_sb4 = _run_j(
        "i9_zero_excl_pure_ac_sb4",
        {"10": s11 + s12, "11": s11, "sb": "ACGT", "insL": ins_l200, "insR": ins_r200, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("insL", "+"), ("sb", "+"), ("insR", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_ac_sb4, "Hap2#1")

    g40_ac = make_complex_seq(40, seed=19528)
    g40_ac_d5 = g40_ac[5:]
    res_ac_g40 = _run_j(
        "i9_zero_excl_pure_ac_g40",
        {"10": s11 + g40_ac_d5 + s12, "11": s11, "g40": g40_ac, "alt2": g40_ac_d5, "insL": ins_l200, "insR": ins_r200, "12": s12},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g40", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("g40", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("insL", "+"), ("g40", "+"), ("insR", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    _assert_hap_clean(res_ac_g40, "Hap2#1")

    # (2d) P2A_WHOLE_PLUS_BENIGN_NOVEL: 10,000 bp novel gene stays 100% unsplit!
    res_p2a = _run_j(
        "i9_p2a_whole_plus_benign_novel",
        {"10": s11 + s12, "11": s11, "gA": ga10, "TE50": ta50, "12": s12, "alt2": ga10_d1},
        sup8 + [
            ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE50", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE50", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("TE50", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ],
    )
    assert _ins_sig(res_p2a, _evs34(res_p2a)[0]) == [
        (10000, "Hap1a#1,Hap1b#1,Hap2#1", "synthetic"),
        (50, "Hap1a#1,Hap1b#1,Hap3#1", "source_contiguous"),
    ]

    # (3a) LIVE_C_TC_GG (m_p == 0 on pa="C", sb="TC", pb="GG")
    def _run_2hop_pair(tag_p: str, pa: str, sb: str, pb: str, dedup_id: float = 0.90):
        res_f = _run_j(
            f"{tag_p}_fwd",
            {"10": s11 + ta50 + pa + pb + tb50 + s12, "11": s11, "ta": ta50, "pa": pa, "sb": sb, "pb": pb, "tb": tb50, "12": s12},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
            dedup_id=dedup_id,
        )
        m_11, m_12 = revcomp_dna(s12), revcomp_dna(s11)
        m_ta, m_tb = revcomp_dna(tb50), revcomp_dna(ta50)
        m_pa, m_sb, m_pb = revcomp_dna(pb), revcomp_dna(sb), revcomp_dna(pa)
        res_m = _run_j(
            f"{tag_p}_mir",
            {"10": m_11 + m_ta + m_pa + m_pb + m_tb + m_12, "11": m_11, "ta": m_ta, "pa": m_pa, "sb": m_sb, "pb": m_pb, "tb": m_tb, "12": m_12},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("pa", "+"), ("pb", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("pa", "+"), ("sb", "+"), ("pb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
            dedup_id=dedup_id,
        )
        return res_f, res_m

    rf_ctc, rm_ctc = _run_2hop_pair("i9_live_c_tc_gg", "C", "TC", "GG", dedup_id=0.90)
    for r_x in (rf_ctc, rm_ctc):
        _assert_hap_clean(r_x, "Hap2#1")
        assert _ins_sig(r_x, _evs34(r_x)[0]) == [(2, "Hap1a#1,Hap1b#1,Hap3#1", "synthetic")]

    # (3b) trial=22 across 0.90 / 0.95 / 0.99 (fwd == mirror!)
    for did in (0.90, 0.95, 0.99):
        rf_22, rm_22 = _run_2hop_pair(
            f"i9_t22_{int(did * 100)}", "ACCAATTAC", "TC", "AAAGGAACCTCATTCAGGT", dedup_id=did
        )
        _assert_hap_clean(rf_22, "Hap2#1")
        _assert_hap_clean(rm_22, "Hap2#1")
        assert _ins_sig(rf_22, _evs34(rf_22)[0]) == _ins_sig(rm_22, _evs34(rm_22)[0]) == [
            (2, "Hap1a#1,Hap1b#1,Hap3#1", "synthetic")
        ]

    # (3c) DEFAULT090_2HOP fwd == mirror (checked_unmsk_by_orig 2-hop symmetry!)
    rf_d90, rm_d90 = _run_2hop_pair(
        "i9_default090_2hop", "GGGAAAAAA", "TAAAAAA", make_complex_seq(19, seed=19530), dedup_id=0.90
    )
    _assert_hap_clean(rf_d90, "Hap2#1")
    _assert_hap_clean(rm_d90, "Hap2#1")
    assert _ins_sig(rf_d90, _evs34(rf_d90)[0]) == _ins_sig(rm_d90, _evs34(rm_d90)[0]) == [
        (7, "Hap1a#1,Hap1b#1,Hap3#1", "synthetic")
    ]

    # (3d) CLI_BUG3 live pipeline: 1 bp, 5 bp, 10 bp 2-SNP, and right-del mirror
    for tag_b3, h2_del, h3_del, snp_pos in (
        ("i9_bug3_1bp", (5000, 5010), (5000, 5011), [5010]),
        ("i9_bug3_5bp", (5000, 5010), (5000, 5015), [5012]),
        ("i9_bug3_10bp_2s", (5000, 5010), (5000, 5020), [5013, 5017]),
        ("i9_bug3_rdel_5bp", (5015, 5025), (5010, 5025), [5012]),
    ):
        alt2_b3 = mutate_seq_snps(ga10, snp_pos)
        alt2_b3 = alt2_b3[: h2_del[0]] + alt2_b3[h2_del[1] :]
        alt3_b3 = ga10[: h3_del[0]] + ga10[h3_del[1] :]
        res_b3 = _run_j(
            tag_b3,
            {"10": s11 + alt2_b3 + tb50 + s12, "11": s11, "gA": ga10, "ta": ta50, "tb": tb50, "12": s12, "alt2": alt2_b3, "alt3": alt3_b3},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("ta", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("ta", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("ta", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
        )
        _assert_hap_clean(res_b3, "Hap2#1")

    # (4a) 6/6 cascade permutation invariance
    perm_members = [
        ("Hap2#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("p10", "+"), ("sb2", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap3#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("sb1", "+"), ("p10", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
        ("Hap4#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb1", "+"), ("p10", "+"), ("sb2", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
    ]
    perm_sigs: set[tuple[tuple[int, str, str], ...]] = set()
    for p_i, perm in enumerate(itertools.permutations(perm_members)):
        res_pm = _run_j(
            f"i9_perm_{p_i}",
            {"11": s11, "ta": ta50, "sb1": "ACGT", "p10": p10a, "sb2": "TGCA", "gA": ga5, "12": s12},
            [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("sb1", "+"), ("p10", "+"), ("sb2", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("ta", "+"), ("sb1", "+"), ("p10", "+"), ("sb2", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                *perm,
            ],
            mode="ref",
        )
        perm_sigs.add(tuple(_ins_sig(res_pm, _evs34(res_pm)[0])))
    assert len(perm_sigs) == 1

    # (4b) Compact deterministic fuzz: 10x PGGB 1-5 bp + 10x 1-9 bp SNP-sliver + 10x 2-hop MH (0 leaks)
    rng = random.Random(20261126)
    for t in range(10):
        # PGGB 1-5 bp bubble
        sb_f = "".join(rng.choices("ACGT", k=rng.randint(1, 5)))
        res_fp = _run_j(
            f"i9_fz_pggb_{t}",
            {"10": s11 + ga5 + ta50 + s12, "11": s11, "sb": sb_f, "gA": ga5, "TE": ta50, "12": s12},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("sb", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("TE", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
        )
        _assert_hap_clean(res_fp, "Hap2#1")

        # 1-9 bp 1-SNP sliver
        sliv = rng.randint(1, 9)
        pos_s = 5010 + rng.randrange(sliv)
        a2_f = mutate_seq_snps(ga10, [pos_s])
        a2_f = a2_f[:5000] + a2_f[5010:]
        a3_f = ga10[:5000] + ga10[5010 + sliv :]
        res_fs = _run_j(
            f"i9_fz_sliv_{t}",
            {"10": s11 + a2_f + tb50 + s12, "11": s11, "gA": ga10, "ta": ta50, "tb": tb50, "12": s12, "alt2": a2_f, "alt3": a3_f},
            sup8 + [
                ("Hap1a#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("ta", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap1b#1#Chr01", [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("gA", "+"), ("ta", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap2#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt2", "+"), ("tb", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
                ("Hap3#1#Chr01",  [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("alt3", "+"), ("ta", "+"), ("12", "+"), ("5", "+"), ("6", "+")]),
            ],
        )
        _assert_hap_clean(res_fs, "Hap2#1")

        # 2-hop flanked microhomology bubble (incl. 1 bp pa/pb)
        pa_f = "".join(rng.choices("ACGT", k=rng.randint(1, 15)))
        sb_2 = "".join(rng.choices("ACGT", k=rng.randint(1, 4)))
        pb_f = "".join(rng.choices("ACGT", k=rng.randint(1, 20)))
        rf_2, rm_2 = _run_2hop_pair(f"i9_fz_2h_{t}", pa_f, sb_2, pb_f, dedup_id=0.90)
        _assert_hap_clean(rf_2, "Hap2#1")
        _assert_hap_clean(rm_2, "Hap2#1")


def test_iter10_repeat_guard_canonical_zero_penalty_and_5th_order_refinements() -> None:
    """Iter 10 Part A Regressions (`panflat/repeat_guard.py`):
    1. `BUG-R9-1a/1b/1c` (`I+X`, `I+M+X`, `D(4)+X(1)` `+`/`-` parity), `SLIV k=6/27`
       multi-base `X+M+I`/`M+X+M+I`, live periodic `D(1)+X(*at)+M(4)`, `I(3)+X+X`,
       and true bridged `'N'*200` `I+X`/`D+X` (`+1`/`-1`).
    2. `BUG1_D10_0.90`, `BUG1b_D4_0.95`, `BUG2a_I7_0.90` (genuine 6 bp MH vs accidental
       7 bp exact tandem `33201` conservation), and `BUG2b_I2_0.99`.
    3. 5th-order refinements: `5O-1a/1b` (`d_win_dual_alt` exact `X`-rewrite at `0.99`),
       `5O-2a` (`s20|sb10` `9 bp`-MH novel ins `FWD==MIR` at `mhl=50 & 0`), `5O-2b`
       (1-bp-shifted `h_tand7` `GGT+A^6|A^8+CCG`), `5O-2c` (`A^20` vs `A^10` & `A^98` vs
       `A^49`), and `5O-2d` (stranded-straddling `I` `FWD==REV==MIR`).
    4. Structural `D >= 50` (`trial=22` `49+2=51 bp`, dual `D(200)` at `p_s` & `p_e`,
       `45+D(200)+5X` X-pad immunity, `D(200)+X(1)`), `trial=251` short-flank outer-`D`,
       and all tandem conservation controls (`mhl=50 & 0`).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut1(ch: str) -> str:
        return {"A": "C", "C": "G", "G": "T", "T": "A"}[ch]

    def _map_hits(t_seq: str, q_seq: str, short: bool = False) -> list[object]:
        aln = (
            mappy.Aligner(seq=t_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
            if short
            else mappy.Aligner(seq=t_seq, preset="asm20", best_n=50)
        )
        return list(aln.map(q_seq, cs=True))

    def _any_cov(
        t_seq: str,
        q_seq: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
        short: bool = False,
        t_starts: tuple[int, ...] | None = None,
        t_ends: tuple[int, ...] | None = None,
    ) -> bool:
        hits = _map_hits(t_seq, q_seq, short=short)
        return any(
            hit_covers_subinterval_mappy(
                h,
                p_s,
                p_e,
                min_id,
                t_starts=t_starts,
                t_ends=t_ends,
                min_hit_len=mhl,
                query_seq=q_seq,
            )
            for h in hits
        )

    flank_l = make_complex_seq(200, seed=21001)
    flank_r = make_complex_seq(220, seed=21002)

    # (1a) BUG-R9-1a (inverted Sup - strand I+X <-> X+I), BUG-R9-1b (fwd-short I+M+X), BUG-R9-1c (D(4)+X(1))
    ga10 = _mseq(10000, 88600)
    a2_1a = ga10[:5000] + _mut1(ga10[5010]) + ga10[5011:]
    q_1a = ga10[4800:5220]
    for t_1a in (a2_1a, revcomp_dna(a2_1a)):
        assert _any_cov(t_1a, q_1a, 210, 213, 0.90, 50) is True

    ga_s = _mseq(240, 88704)
    a2_1b = ga_s[:100] + _mut1(ga_s[110]) + ga_s[111:]
    for t_1b in (a2_1b, revcomp_dna(a2_1b)):
        assert _any_cov(t_1b, ga_s, 100, 101, 0.90, 50) is True

    g_l2, g_r2 = _mseq(2000, 99101), _mseq(2000, 99102)
    q_1c = g_l2[-200:] + "ACG" + g_r2[:200]
    t_1c_fwd = g_l2 + "ACTCCAC" + g_r2
    assert _any_cov(t_1c_fwd, q_1c, 200, 203, 0.90, 50) is True
    assert _any_cov(revcomp_dna(t_1c_fwd), q_1c, 200, 203, 0.90, 50) is True

    # (1b) SLIV k=6 (X+M+I <-> I+M+X) and SLIV k=27 (X+M+M+I <-> I+M+M+X) multi-base zero-penalty swaps
    # SLIV k=6: Q[200..206] = "C" (X: T='A') + "G" (M: T='G') + I("TTCAC", [202, 207))
    # Shift I left by dl=1: Q[206]='C' != T[201]='G' (orig_m=1, new_m=0 -> fails greedy 1-bp!)
    # Shift I left by dl=2: Q[205]='A' == T[200]='A' (orig_m=1, new_m=1 -> exact zero-penalty!)
    q_k6 = flank_l + "CG" + "TTCAC" + flank_r
    h_k6_fwd = SimpleNamespace(
        q_st=0, q_en=len(q_k6), r_st=0, r_en=len(q_k6) - 5, strand=1, cs=":200*ac:1+ttcac:220"
    )
    h_k6_rev = SimpleNamespace(
        q_st=0, q_en=len(q_k6), r_st=0, r_en=len(q_k6) - 5, strand=-1, cs=":220+gtgaa:1*tg:200"
    )
    assert hit_covers_subinterval_mappy(h_k6_fwd, 205, 207, 0.90, min_hit_len=50, query_seq=q_k6) is True
    assert hit_covers_subinterval_mappy(h_k6_rev, 205, 207, 0.90, min_hit_len=50, query_seq=q_k6) is True

    # SLIV k=27: Q[200..211] = "C" (X: T='A') + "T" (M: T='T') + "G" (M: T='G') + I("CCCCCCATC", [203, 212))
    # Shift I left by dl=1 (211:'C'!=202:'G' -> 0/1), dl=2 (210:'T'==201:'T' -> 1/2), dl=3 (209:'A'==200:'A' -> 2/2 exact!)
    q_k27 = flank_l + "CTG" + "CCCCCCATC" + flank_r
    h_k27_fwd = SimpleNamespace(
        q_st=0, q_en=len(q_k27), r_st=0, r_en=len(q_k27) - 9, strand=1, cs=":200*ac:2+ccccccatc:220"
    )
    h_k27_rev = SimpleNamespace(
        q_st=0, q_en=len(q_k27), r_st=0, r_en=len(q_k27) - 9, strand=-1, cs=":220+gatgggggg:2*tg:200"
    )
    assert hit_covers_subinterval_mappy(h_k27_fwd, 209, 212, 0.90, min_hit_len=50, query_seq=q_k27) is True
    assert hit_covers_subinterval_mappy(h_k27_rev, 209, 212, 0.90, min_hit_len=50, query_seq=q_k27) is True

    # (1c) Live mappy periodic D(1)+X(*at)+M(4) strand parity (cs=":101-a*at:104")
    pl_100 = make_complex_seq(100, seed=21010)
    pr_100 = make_complex_seq(100, seed=21011)
    q_per = pl_100 + "GTAAAA" + pr_100
    t_per = pl_100 + "GAAAAAA" + pr_100
    assert _any_cov(t_per, q_per, 100, 106, 0.90, 50, short=True) is True
    assert _any_cov(revcomp_dna(t_per), q_per, 100, 106, 0.90, 50, short=True) is True

    # (1d) I(3)+X+X credits X (not M): 48/50=96% True@0.90/0.95/0.96, False@0.97
    q_i3xx = flank_l + "ACG" + "CT" + flank_r
    h_i3xx = SimpleNamespace(
        q_st=0, q_en=len(q_i3xx), r_st=0, r_en=len(q_i3xx) - 3, strand=1, cs=":200+acg*tc*gt:220"
    )
    assert hit_covers_subinterval_mappy(h_i3xx, 200, 202, 0.90, min_hit_len=50, query_seq=q_i3xx) is True
    assert hit_covers_subinterval_mappy(h_i3xx, 200, 202, 0.95, min_hit_len=50, query_seq=q_i3xx) is True
    assert hit_covers_subinterval_mappy(h_i3xx, 200, 202, 0.96, min_hit_len=50, query_seq=q_i3xx) is True
    assert hit_covers_subinterval_mappy(h_i3xx, 200, 202, 0.97, min_hit_len=50, query_seq=q_i3xx) is False

    # (1e) True bridged 'N'*200 I+X and D+X on +1 and -1 strands (5O-6)
    br_a = make_complex_seq(100, seed=21021)
    br_b = make_complex_seq(100, seed=21022)
    br_c = make_complex_seq(100, seed=21023)
    # Piece 0 = br_a + br_b (200 bp), spacer = 'N'*200, Piece 1 = br_c (100 bp)
    t_br_starts = (0, 400)
    t_br_ends = (200, 500)
    # Bridged I+X in Piece 0: I(2) straddling p=[80, 99) followed by X at 100
    q_brix = br_a[:98] + "CT" + "G" + br_b[1:] + ("N" * 200) + br_c
    h_brix_f = SimpleNamespace(
        q_st=0, q_en=len(q_brix), r_st=0, r_en=500, strand=1,
        cs=f":98+ct*ag:99:200:{len(br_c)}",
    )
    h_brix_r = SimpleNamespace(
        q_st=0, q_en=len(q_brix), r_st=0, r_en=500, strand=-1,
        cs=f":{len(br_c)}:200:99*tc+ag:98",
    )
    assert hit_covers_subinterval_mappy(h_brix_f, 80, 99, 0.90, t_br_starts, t_br_ends, 50, q_brix) is True
    assert hit_covers_subinterval_mappy(h_brix_r, 80, 99, 0.90, t_br_starts, t_br_ends, 50, q_brix) is True
    # Bridged D+X in Piece 0: D(4) at q=99 followed by X at q=99 inside p=[90, 100)
    q_brdx = br_a[:99] + "G" + br_b[4:] + ("N" * 200) + br_c
    h_brdx_f = SimpleNamespace(
        q_st=0, q_en=len(q_brdx), r_st=0, r_en=500, strand=1,
        cs=f":99-tcca*ag:96:200:{len(br_c)}",
    )
    h_brdx_r = SimpleNamespace(
        q_st=0, q_en=len(q_brdx), r_st=0, r_en=500, strand=-1,
        cs=f":{len(br_c)}:200:96*tc-tgga:99",
    )
    assert hit_covers_subinterval_mappy(h_brdx_f, 90, 100, 0.90, t_br_starts, t_br_ends, 50, q_brdx) is True
    assert hit_covers_subinterval_mappy(h_brdx_r, 90, 100, 0.90, t_br_starts, t_br_ends, 50, q_brdx) is True

    # (2) BUG1_D10_0.90, BUG1b_D4_0.95, BUG2a_I7_0.90 (genuine 6 bp MH vs accidental 7 bp tandem 33201), BUG2b_I2_0.99
    l0 = _mseq(400, 33101)
    r0 = _mseq(400, 33102)
    s20_b1 = _mseq(19, 33105) + "A"
    s40_b1 = "G" + _mseq(39, 33106)
    q_b1 = l0[-200:] + s20_b1 + s40_b1 + r0[:200]
    t_b1_d10 = l0 + s20_b1 + "CGTACGTACA" + s40_b1 + r0
    assert _any_cov(t_b1_d10, q_b1, 200, 220, 0.90, 50) is True
    assert _any_cov(revcomp_dna(t_b1_d10), revcomp_dna(q_b1), len(q_b1) - 220, len(q_b1) - 200, 0.90, 50) is True
    t_b1_d4 = l0 + s20_b1 + "CGTA" + s40_b1 + r0
    assert _any_cov(t_b1_d4, q_b1, 200, 220, 0.95, 50) is True
    assert _any_cov(revcomp_dna(t_b1_d4), revcomp_dna(q_b1), len(q_b1) - 220, len(q_b1) - 200, 0.95, 50) is True

    # Genuine 6 bp MH BUG2a_I7_0.90 -> True (and sb7 -> False), while accidental 7 bp tandem 33201 -> False!
    sb7 = "TACGTTC"
    s25_2 = _mseq(25, 33202)
    s20_6mh = _mseq(13, 33201) + "G" + "ACGTTC"
    assert not s20_6mh.endswith(sb7) and s20_6mh.endswith(sb7[1:])
    q_b2a = l0[-200:] + s20_6mh + sb7 + s25_2 + r0[:200]
    t_b2a = l0 + s20_6mh + s25_2 + r0
    assert _any_cov(t_b2a, q_b2a, 200, 220, 0.90, 50) is True
    assert _any_cov(t_b2a, q_b2a, 220, 227, 0.90, 50) is False
    assert _any_cov(revcomp_dna(t_b2a), revcomp_dna(q_b2a), len(q_b2a) - 220, len(q_b2a) - 200, 0.90, 50) is True
    assert _any_cov(revcomp_dna(t_b2a), revcomp_dna(q_b2a), len(q_b2a) - 227, len(q_b2a) - 220, 0.90, 50) is False

    s20_tand7 = _mseq(14, 33201) + "ACGTTC"
    assert s20_tand7.endswith(sb7)  # exact 7 bp tandem TACGTTC|TACGTTC
    q_b2a_tand = l0[-200:] + s20_tand7 + sb7 + s25_2 + r0[:200]
    t_b2a_tand = l0 + s20_tand7 + s25_2 + r0
    assert _any_cov(t_b2a_tand, q_b2a_tand, 200, 220, 0.90, 50) is False
    assert _any_cov(t_b2a_tand, q_b2a_tand, 220, 227, 0.90, 50) is False
    assert _any_cov(revcomp_dna(t_b2a_tand), q_b2a_tand, 200, 220, 0.90, 50) is False
    assert _any_cov(revcomp_dna(t_b2a_tand), q_b2a_tand, 220, 227, 0.90, 50) is False
    assert _any_cov(revcomp_dna(t_b2a_tand), revcomp_dna(q_b2a_tand), len(q_b2a_tand) - 220, len(q_b2a_tand) - 200, 0.90, 50) is False
    assert _any_cov(revcomp_dna(t_b2a_tand), revcomp_dna(q_b2a_tand), len(q_b2a_tand) - 227, len(q_b2a_tand) - 220, 0.90, 50) is False

    # BUG2b_I2_0.99 (1 bp MH sb2="CA")
    s20_3 = _mseq(19, 33301) + "A"
    s40_3 = "G" + _mseq(39, 33302)
    q_b2b = l0[-200:] + s20_3 + "CA" + s40_3 + r0[:200]
    t_b2b = l0 + s20_3 + s40_3 + r0
    assert _any_cov(t_b2b, q_b2b, 200, 220, 0.99, 50) is True
    assert _any_cov(t_b2b, q_b2b, 220, 222, 0.99, 50) is False
    assert _any_cov(revcomp_dna(t_b2b), revcomp_dna(q_b2b), len(q_b2b) - 220, len(q_b2b) - 200, 0.99, 50) is True

    # (3a) 5O-1a & 5O-1b: d_win_dual_alt exact X-rewrite at 0.99
    # 5O-1b: p=[100, 106), D(5) at q=103 with can_r_win (X moves 106->103) AND can_l_win (X moves 55->102).
    # Unshifted has X only at 55 & 106 (both outside [56, 106)), but EVERY zero-penalty shift puts X in [56, 106) -> False!
    # Construct periodic D(5)="ACGTG" across q=56..102 and q=104..105:
    q_5o1b_l = make_complex_seq(55, seed=21031)
    # 47 bases q=56..102: q=102 mismatches d_seq[4], q=55 matches d_seq[(103-1-55)%5]? Solve directly via t_stream_left:
    # Let d_seq = "ACGTG" (len=5). Leftward from q=102..55 (48 steps):
    # orig_left[0..46] = M, orig_left[47] = X (at q=55).
    # Choose q[56..102] so q[102] != 'G' (M->X at k=0), q[56..101] matches t_stream_left[1..46] (M->M),
    # and q[55] == t_stream_left[47] (X->M at k=47, where T[55] != q[55])!
    d5_1b = "ACGTG"
    # Build Q[102..55] in reverse index k=0..47:
    # t_stream_left = list(reversed(d5_1b)) + [T[102], T[101], ..., T[55]]
    # For k=0 (q=102): orig=M => T[102]=Q[102]; want Q[102] != d5_1b[4]='G', pick 'A'.
    # For k=1..46 (q=101..56): orig=M => T[q]=Q[q]; set Q[102-k] = t_stream_left[k] (guaranteed defined since k-5 < k!).
    # For k=47 (q=55): orig=X => Q[55] = t_stream_left[47], T[55] = _mut1(Q[55]).
    rev_q_5o1b: list[str] = ["A"]
    t_str_l: list[str] = list(reversed(d5_1b)) + ["A"]
    for k_l in range(1, 47):
        ch = t_str_l[k_l]
        rev_q_5o1b.append(ch)
        t_str_l.append(ch)
    q55_b = t_str_l[47]
    t55_b = _mut1(q55_b)
    rev_q_5o1b.append(q55_b)
    mid_l_5o1b = "".join(reversed(rev_q_5o1b))  # q=55..102 (48 bp)
    # Rightward from q=103..106 (4 steps): orig=[M, M, M, X(at 106)]; new=[X(at 103), M, M, M]
    # t_stream_right = list(d5_1b) + [T[103], ...]
    # k=0 (q=103): Q[103] != d5_1b[0]='A' -> 'C' (M->X)
    # k=1 (q=104): Q[104] == d5_1b[1]='C' -> 'C' (M->M)
    # k=2 (q=105): Q[105] == d5_1b[2]='G' -> 'G' (M->M)
    # k=3 (q=106): Q[106] == d5_1b[3]='T' -> 'T' (X->M with T[106]='A')
    q_5o1b = q_5o1b_l + mid_l_5o1b + "CCGT" + make_complex_seq(60, seed=21032)
    h_5o1b_unsh = SimpleNamespace(
        q_st=0, q_en=len(q_5o1b), r_st=0, r_en=len(q_5o1b) + 5, strand=1,
        cs=f":55*{t55_b.lower()}{q55_b.lower()}:47-{d5_1b.lower()}:3*at:60",
    )
    h_5o1b_sh_r = SimpleNamespace(
        q_st=0, q_en=len(q_5o1b), r_st=0, r_en=len(q_5o1b) + 5, strand=1,
        cs=f":55*{t55_b.lower()}{q55_b.lower()}:47*ac:3-gtacc:60",
    )
    assert hit_covers_subinterval_mappy(h_5o1b_unsh, 100, 106, 0.99, min_hit_len=50, query_seq=q_5o1b) is False
    assert hit_covers_subinterval_mappy(h_5o1b_sh_r, 100, 106, 0.99, min_hit_len=50, query_seq=q_5o1b) is False

    # 5O-1a: p=[100, 106), D(5)="ACGTG" at q=103, X at 103 (orig_r=[X, M, M, M]) -> right shift moves X from 103 to 106
    # (OUTSIDE [56, 106) -> 50/50 = 100% True@0.99!), while left shift is pure M to q=100 (leaving X at 103 inside [56, 106)).
    # Left flank [56, 100) is clean; right flank has X at 120 so ONLY [56, 106) can reach 50/50!
    q_5o1a = make_complex_seq(100, seed=21033) + "GTG" + "ACGA" + make_complex_seq(60, seed=21034)
    # Check: q=100..102 = "GTG" == d5_1b[2:5] -> can_l_win (pure M, l_exit_d=3, X stays at 103).
    # Rightward q=103..106 = "ACGA", orig=[X(*ca), M, M, M], d5_1b="ACGTG":
    # k=0: Q[103]='A'==d5_1b[0] (M), k=1: 'C'=='C' (M), k=2: 'G'=='G' (M), k=3: 'A'!='T' (X at 106).
    # Make q=103..106 pure zero-penalty at d=4 by having X at 103 (orig_m=0,1,2,3; new_m=1,2,3,3):
    # Shift right by 3 already gives [M,M,M] at 103..105 -> [56, 106) is 50/50 = 100% True@0.99!
    h_5o1a_unsh = SimpleNamespace(
        q_st=0, q_en=len(q_5o1a), r_st=0, r_en=len(q_5o1a) + 5, strand=1,
        cs=f":103-{d5_1b.lower()}*ca:16*ac:46",
    )
    h_5o1a_sh_r = SimpleNamespace(
        q_st=0, q_en=len(q_5o1a), r_st=0, r_en=len(q_5o1a) + 5, strand=1,
        cs=":106*ta-gtgac:13*ac:46",
    )
    assert hit_covers_subinterval_mappy(h_5o1a_unsh, 100, 106, 0.99, min_hit_len=50, query_seq=q_5o1a) is True
    assert hit_covers_subinterval_mappy(h_5o1a_sh_r, 100, 106, 0.99, min_hit_len=50, query_seq=q_5o1a) is True

    # (3b) 5O-2a (s20|sb10 9bp-MH novel ins conserved FWD==MIR at mhl=50 & 0)
    mh9 = "ACGTTCGAC"
    s20_5o2 = _mseq(10, 21041) + "G" + mh9
    sb10_5o2 = "T" + mh9
    assert not s20_5o2.endswith(sb10_5o2) and s20_5o2.endswith(mh9)
    s25_5o2 = _mseq(25, 21042)
    q_5o2a = l0[-200:] + s20_5o2 + sb10_5o2 + s25_5o2 + r0[:200]
    t_5o2a = l0 + s20_5o2 + s25_5o2 + r0
    for mhl in (50, 0):
        fwd_pair = (
            _any_cov(t_5o2a, q_5o2a, 200, 220, 0.90, mhl),
            _any_cov(t_5o2a, q_5o2a, 220, 230, 0.90, mhl),
        )
        mir_q = revcomp_dna(q_5o2a)
        mir_t = revcomp_dna(t_5o2a)
        mir_pair = (
            _any_cov(mir_t, mir_q, len(q_5o2a) - 220, len(q_5o2a) - 200, 0.90, mhl),
            _any_cov(mir_t, mir_q, len(q_5o2a) - 230, len(q_5o2a) - 220, 0.90, mhl),
        )
        assert fwd_pair == mir_pair == (True, False), f"5O-2a mhl={mhl}: fwd={fwd_pair} mir={mir_pair}"

    # (3c) 5O-2b (1-bp-shifted h_tand7 GGT+A^6 | A^8+CCG conserved FWD==MIR at mhl=50 & 0)
    q_5o2b = flank_l + "GGT" + ("A" * 6) + ("A" * 8) + "CCG" + flank_r
    h_5o2b_f = SimpleNamespace(
        q_st=0, q_en=len(q_5o2b), r_st=0, r_en=len(q_5o2b) - 7, strand=1,
        cs=":203+" + ("a" * 7) + ":230",
    )
    q_5o2b_m = revcomp_dna(q_5o2b)
    h_5o2b_m = SimpleNamespace(
        q_st=0, q_en=len(q_5o2b_m), r_st=0, r_en=len(q_5o2b_m) - 7, strand=1,
        cs=":223+" + ("t" * 7) + ":210",
    )
    for mhl in (50, 0):
        pf = (
            hit_covers_subinterval_mappy(h_5o2b_f, 200, 209, 0.90, min_hit_len=mhl, query_seq=q_5o2b),
            hit_covers_subinterval_mappy(h_5o2b_f, 209, 220, 0.90, min_hit_len=mhl, query_seq=q_5o2b),
        )
        pm = (
            hit_covers_subinterval_mappy(h_5o2b_m, len(q_5o2b) - 209, len(q_5o2b) - 200, 0.90, min_hit_len=mhl, query_seq=q_5o2b_m),
            hit_covers_subinterval_mappy(h_5o2b_m, len(q_5o2b) - 220, len(q_5o2b) - 209, 0.90, min_hit_len=mhl, query_seq=q_5o2b_m),
        )
        assert pf == pm and pf != (True, True), f"5O-2b mhl={mhl}: pf={pf} pm={pm}"

    # (3d) 5O-2c (4-piece A^20 vs A^10 FWD==MIR==[F,F,F,T] via Iter 12 s_cover dual & 49 bp A^98 vs A^49 conserved)
    q_a20 = flank_l + "GGT" + ("A" * 20) + "CCG" + flank_r
    h_a20_f = SimpleNamespace(
        q_st=0, q_en=len(q_a20), r_st=0, r_en=len(q_a20) - 10, strand=1,
        cs=":203+" + ("a" * 10) + ":233",
    )
    q_a20_m = revcomp_dna(q_a20)
    h_a20_m = SimpleNamespace(
        q_st=0, q_en=len(q_a20_m), r_st=0, r_en=len(q_a20_m) - 10, strand=1,
        cs=":223+" + ("t" * 10) + ":213",
    )
    # 4 pieces covering A^20: [203, 207) A^4, [207, 212) A^5, [212, 222) A^10, [222, 226) ACCG
    spans_a20 = [(203, 207), (207, 212), (212, 222), (222, 226)]
    res_a20_f = [
        hit_covers_subinterval_mappy(h_a20_f, s, e, 0.90, min_hit_len=50, query_seq=q_a20)
        for s, e in spans_a20
    ]
    res_a20_m = [
        hit_covers_subinterval_mappy(h_a20_m, len(q_a20) - e, len(q_a20) - s, 0.90, min_hit_len=50, query_seq=q_a20_m)
        for s, e in spans_a20
    ]
    assert res_a20_f == res_a20_m == [False, False, False, True]
    q_a98 = flank_l + "GGT" + ("A" * 98) + "CCG" + flank_r
    h_a98 = SimpleNamespace(
        q_st=0, q_en=len(q_a98), r_st=0, r_en=len(q_a98) - 49, strand=1,
        cs=":203+" + ("a" * 49) + ":272",
    )
    assert hit_covers_subinterval_mappy(h_a98, 203, 252, 0.90, min_hit_len=50, query_seq=q_a98) is False

    # (3e) 5O-2d (stranded-straddling I: best_ov_final > 0, r_pull_dr == 0, l_pull_dl == 0 -> FWD==REV==MIR)
    # I(6)="TCAGTC" straddling p_s=200 at [198, 204), flanked by non-repeating bases -> can shift 1 bp left to [197, 203)
    # (still straddling p_s=200!) and cannot pull across p_s=200 right -> canonicalized in b_i on all strands/orientations.
    fl_2d = flank_l[:-3] + "GGC"  # q[197..200) = "GGC"
    fr_2d = "GGT" + flank_r[3:]
    # I(6) at [198, 204) = "CTCAGT", q[197]='G', q[203]='T' -> make q[197]='T' == q[203]='T', q[196]='A' != q[202]='G',
    # q[198]='C' != q[204]='G' so L_min=197, L_max=198 (permanently straddling p_s=200!)
    fl_2d = flank_l[:-4] + "AT" + "CT"  # q[196]='A', q[197]='T', q[198]='C', q[199]='T'
    q_5o2d = fl_2d + "CAGT" + _mseq(36, 21045) + fr_2d  # I(6)="CTCAGT" at [198, 204), p=[200, 240)
    q_5o2d = q_5o2d[:204] + "G" + q_5o2d[205:]  # ensure q[204]='G' != q[198]='C'
    h_2d_f = SimpleNamespace(
        q_st=0, q_en=len(q_5o2d), r_st=0, r_en=len(q_5o2d) - 6, strand=1,
        cs=f":198+ctcagt:{len(q_5o2d) - 204}",
    )
    h_2d_r = SimpleNamespace(
        q_st=0, q_en=len(q_5o2d), r_st=0, r_en=len(q_5o2d) - 6, strand=-1,
        cs=f":{len(q_5o2d) - 204}+actgag:198",
    )
    q_5o2d_m = revcomp_dna(q_5o2d)
    h_2d_m = SimpleNamespace(
        q_st=0, q_en=len(q_5o2d_m), r_st=0, r_en=len(q_5o2d_m) - 6, strand=1,
        cs=f":{len(q_5o2d) - 204}+actgag:198",
    )
    vf_2d = hit_covers_subinterval_mappy(h_2d_f, 200, 240, 0.92, min_hit_len=50, query_seq=q_5o2d)
    vr_2d = hit_covers_subinterval_mappy(h_2d_r, 200, 240, 0.92, min_hit_len=50, query_seq=q_5o2d)
    vm_2d = hit_covers_subinterval_mappy(h_2d_m, len(q_5o2d) - 240, len(q_5o2d) - 200, 0.92, min_hit_len=50, query_seq=q_5o2d_m)
    assert vf_2d == vr_2d == vm_2d, f"5O-2d involution failed: vf={vf_2d} vr={vr_2d} vm={vm_2d}"

    # (4) Structural D >= 50 (trial=22, dual D(200), 45+D(200)+5X X-pad immunity, D(200)+X(1)) & trial=251 outer-D
    d200_ta = "TA" + make_complex_seq(198, seed=21051)
    q_t22d = make_complex_seq(49, seed=21052) + "TA" + make_complex_seq(198, seed=21053)
    h_t22d_r = SimpleNamespace(
        q_st=0, q_en=len(q_t22d), r_st=0, r_en=len(q_t22d) + 200, strand=1,
        cs=f":49-{d200_ta.lower()}:200",
    )
    assert hit_covers_subinterval_mappy(h_t22d_r, 0, 49, 0.90, min_hit_len=50, query_seq=q_t22d) is True
    q_t22d_m = revcomp_dna(q_t22d)
    h_t22d_l = SimpleNamespace(
        q_st=0, q_en=len(q_t22d_m), r_st=0, r_en=len(q_t22d_m) + 200, strand=1,
        cs=f":200-{revcomp_dna(d200_ta).lower()}:49",
    )
    assert hit_covers_subinterval_mappy(h_t22d_l, 200, 249, 0.90, min_hit_len=50, query_seq=q_t22d_m) is True

    # Dual D(200) at both p_s and p_e: 46 bp core + 2 bp MH left + 2 bp MH right = 50 bp -> True
    d200_l = make_complex_seq(198, seed=21054) + "CG"
    d200_r = "AT" + make_complex_seq(198, seed=21055)
    q_dual_d = make_complex_seq(98, seed=21056) + "CG" + make_complex_seq(46, seed=21057) + "AT" + make_complex_seq(98, seed=21058)
    h_dual_d = SimpleNamespace(
        q_st=0, q_en=len(q_dual_d), r_st=0, r_en=len(q_dual_d) + 400, strand=1,
        cs=f":100-{d200_l.lower()}:46-{d200_r.lower()}:100",
    )
    assert hit_covers_subinterval_mappy(h_dual_d, 100, 146, 0.90, min_hit_len=50, query_seq=q_dual_d) is True

    # 45+D(200)+5X X-pad immunity (MUST stay False on BOTH right and left!)
    d200_np = "CGTGC" + make_complex_seq(190, seed=21059) + "CGTGC"
    q_xpad_r = make_complex_seq(45, seed=21060) + "AAAAA" + make_complex_seq(100, seed=21061)
    h_xpad_r = SimpleNamespace(
        q_st=0, q_en=len(q_xpad_r), r_st=0, r_en=len(q_xpad_r) + 200, strand=1,
        cs=f":45-{d200_np.lower()}*ca*ca*ca*ca*ca:100",
    )
    assert hit_covers_subinterval_mappy(h_xpad_r, 0, 45, 0.90, min_hit_len=50, query_seq=q_xpad_r) is False
    q_xpad_l = make_complex_seq(100, seed=21061) + "AAAAA" + make_complex_seq(45, seed=21060)
    h_xpad_l = SimpleNamespace(
        q_st=0, q_en=len(q_xpad_l), r_st=0, r_en=len(q_xpad_l) + 200, strand=1,
        cs=f":100*ca*ca*ca*ca*ca-{d200_np.lower()}:45",
    )
    assert hit_covers_subinterval_mappy(h_xpad_l, 105, 150, 0.90, min_hit_len=50, query_seq=q_xpad_l) is False

    # D(200)+X(1) inside [p_s, p_e): 49 bp M + D(200) + X(1) at q=49 inside p=[0, 50) + 1 bp MH at q=50 -> True!
    d200_dx = "AG" + make_complex_seq(198, seed=21062)
    q_dx1 = make_complex_seq(49, seed=21063) + "CG" + make_complex_seq(100, seed=21064)
    h_dx1 = SimpleNamespace(
        q_st=0, q_en=len(q_dx1), r_st=0, r_en=len(q_dx1) + 200, strand=1,
        cs=f":49-{d200_dx.lower()}*tc:101",
    )
    assert hit_covers_subinterval_mappy(h_dx1, 0, 50, 0.90, min_hit_len=50, query_seq=q_dx1) is True

    # trial=251: short flank L_len=40 < 50, p=[40, 42), outer D(13) at q=43 with pure-M MH < 13 -> FWD==MIR==True
    d13_251 = "ACGTACGTACGTA"
    q_251 = make_complex_seq(40, seed=21065) + "GT" + "A" + d13_251[:7] + make_complex_seq(60, seed=21066)
    h_251_f = SimpleNamespace(
        q_st=0, q_en=len(q_251), r_st=0, r_en=len(q_251) + 13, strand=1,
        cs=f":43-{d13_251.lower()}:67",
    )
    h_251_r = SimpleNamespace(
        q_st=0, q_en=len(q_251), r_st=0, r_en=len(q_251) + 13, strand=-1,
        cs=f":67-{revcomp_dna(d13_251).lower()}:43",
    )
    assert hit_covers_subinterval_mappy(h_251_f, 40, 42, 0.90, min_hit_len=50, query_seq=q_251) is True
    assert hit_covers_subinterval_mappy(h_251_r, 40, 42, 0.90, min_hit_len=50, query_seq=q_251) is True

    # (5) Tandem conservation controls (BOTH mhl=50 AND mhl=0)
    q_t7d = flank_l + "GGT" + ("A" * 14) + "CCG" + flank_r
    h_t7d = SimpleNamespace(q_st=0, q_en=len(q_t7d), r_st=0, r_en=len(q_t7d) + 7, strand=1, cs=":210-aaaaaaa:230")
    q_xa5 = flank_l + "C" + ("A" * 5) + "C" + ("A" * 5) + flank_r
    h_xa5 = SimpleNamespace(q_st=0, q_en=len(q_xa5), r_st=0, r_en=len(q_xa5) - 6, strand=1, cs=":200+caaaaa:226")
    q_int5d = flank_l + "GCTAGCTGAT" + flank_r
    h_int5d = SimpleNamespace(q_st=0, q_en=len(q_int5d), r_st=0, r_en=len(q_int5d) + 5, strand=1, cs=":205-tcgta:225")
    for mhl in (50, 0):
        assert hit_covers_subinterval_mappy(h_t7d, 200, 217, 0.90, min_hit_len=mhl, query_seq=q_t7d) is True
        assert hit_covers_subinterval_mappy(h_xa5, 200, 206, 0.90, min_hit_len=mhl, query_seq=q_xa5) is False
        assert hit_covers_subinterval_mappy(h_int5d, 200, 210, 0.90, min_hit_len=mhl, query_seq=q_int5d) is False


def test_iter10_stage55_attack5l_finding4_and_cross_interaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iter 10 Part B Regressions (`panflat/linearizer.py` Stage 5.5):
    1. `BUG-R9-2` + `5O-4`: `Attack 5L`, `5L_split`, `5L_3split`,
       `5L_3split_right_novel_short` (`5O-4`), `5L_two_sided` (`100%`, `92%`, `10+10`),
       `FWD == MIR`, and `0` false drops on isolated `10..49 bp` pieces.
    2. `Finding #4` + `5O-3` + `5O-3b`: devilish `1a/1b/1c/1d`, `5O-3` (`protected_ins`
       inside `88%` rejected flank), `5O-3b` (bridged `'N'*200` `88%` flanks), and
       `interleaved_96pct_te` (`N=400`, `wave_builds <= 3` + exact parity).
    3. Cross-interaction `5O-7`: `Attack 5L` + `1 bp`-MH `D(10)` + `I+X` SNP sliver +
       `tail_dirty` (`FWD == MIR`).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mk_contig_pieces(seqs: list[str], hap: str = "HapQ#1") -> tuple[Piece, ...]:
        out: list[Piece] = []
        off = 0
        for sq in seqs:
            out.append(Piece(hap, f"{hap}#Chr01", off, off + len(sq), "+", sq, "source_contiguous", (hap,)))
            off += len(sq)
        return tuple(out)

    def _mk_sol(idx: int, pieces: tuple[Piece, ...], sup_n: int, tok: str) -> ParallelGroupSolution:
        cid = f"c{idx}"
        tot = sum(len(p.seq) for p in pieces)
        hap0 = pieces[0].src_hap
        ex = ExtractedCandidate(cid, None, hap0, f"{hap0}#Chr01", 0, 1, idx, (), (), "", ())
        rep = ClusterRepresentative(
            rep_idx=1,
            is_synthetic=len(pieces) > 1,
            seq="",
            length_bp_no_n=tot,
            pieces=pieces,
            support_haps={f"S{k}#1" for k in range(sup_n)},
            token_str=tok,
            min_hap="S0#1",
            exemplar=ex,
            member_occurrences=[],
            nested_choices_by_cand_id={},
            reconstructed_len_by_cand_id={},
            piece_cand_ids=tuple(frozenset({cid}) for _ in pieces),
            support_hap_count=sup_n,
        )
        return ParallelGroupSolution(
            0, 1, False, "", (rep,), {cid: 1}, {cid: ()}, {cid: tot}, {cid: (False, False, False)}
        )

    def _run_s55(
        sols: dict[tuple[int, int], ParallelGroupSolution],
        min_sv_len: int = 50,
        dedup_id: float = 0.90,
        n_nodes: int = 12,
    ) -> tuple[int, int, int]:
        g = SimpleNamespace(ref_step_ranges=[(i * 200, (i + 1) * 200) for i in range(n_nodes)])
        cfg = BuildConfig(
            gfa=str(tmp_path / "x.gfa"),
            ref_fa=str(tmp_path / "x.fa"),
            backbone_prefix="Ref#1#",
            out_prefix=str(tmp_path / "out"),
            dedup_mode="pan",
            dedup_id=dedup_id,
            min_sv_len=min_sv_len,
            threads=1,
        )
        empty_cen = CentromereOutput([], [], [], [], [])
        pairs = sorted(sols.keys())
        return _deduplicate_representatives_across_chromosome(
            g, cfg, pairs, {p: GroupRepeatGuardResult() for p in pairs}, sols, empty_cen  # type: ignore[arg-type]
        )

    def _kept_lens(sol: ParallelGroupSolution) -> list[int]:
        if not sol.non_backbone_reps:
            return []
        return [len(p.seq) for p in sol.non_backbone_reps[0].pieces]

    def _run_attack5l_pair(
        sup_seqs: list[str],
        q_pieces: list[str],
    ) -> tuple[list[int], list[int]]:
        sols_f: dict[tuple[int, int], ParallelGroupSolution] = {}
        for i, ss in enumerate(sup_seqs):
            sols_f[(i, i + 1)] = _mk_sol(i, (Piece("Sup#1", "Sup#1#Chr01", 0, len(ss), "+", ss, "source_contiguous", ("Sup#1",)),), 20 - i, f"sf_{i}")
        q_key = (len(sup_seqs), len(sup_seqs) + 1)
        sols_f[q_key] = _mk_sol(99, _mk_contig_pieces(q_pieces), 2, "qf")
        _run_s55(sols_f, n_nodes=len(sup_seqs) + 4)

        sols_m: dict[tuple[int, int], ParallelGroupSolution] = {}
        for i, ss in enumerate(sup_seqs):
            ss_m = revcomp_dna(ss)
            sols_m[(i, i + 1)] = _mk_sol(i, (Piece("Sup#1", "Sup#1#Chr01", 0, len(ss_m), "+", ss_m, "source_contiguous", ("Sup#1",)),), 20 - i, f"sm_{i}")
        q_pieces_m = [revcomp_dna(s) for s in reversed(q_pieces)]
        sols_m[q_key] = _mk_sol(99, _mk_contig_pieces(q_pieces_m), 2, "qm")
        _run_s55(sols_m, n_nodes=len(sup_seqs) + 4)
        return _kept_lens(sols_f[q_key]), _kept_lens(sols_m[q_key])

    # (1) BUG-R9-2 + 5O-4: Attack 5L family + isolated 10..49 bp zero-false-drop
    xs = [_mseq(40, 21100 + k) for k in range(4)]
    s2 = _mseq(35, 21110)
    s1 = _mseq(10, 21111)
    s0 = _mseq(15, 21112)
    ta60, tb60 = _mseq(60, 21113), _mseq(60, 21114)
    p500a, p500b, p500c = _mseq(500, 21120), _mseq(500, 21121), _mseq(500, 21122)
    sup_5l = [p500a + "".join(xs) + s2 + p500b + s2 + s1 + s0 + p500c]

    # Attack 5L (single s2=35)
    assert _run_attack5l_pair(sup_5l, [*xs, s2, s1, s0, ta60, tb60]) == ([60, 60], [60, 60])
    # Attack 5L_split (s2 = 18 + 17)
    assert _run_attack5l_pair(sup_5l, [*xs, s2[:18], s2[18:], s1, s0, ta60, tb60]) == ([60, 60], [60, 60])
    # Attack 5L_3split (s2 = 12 + 12 + 11)
    assert _run_attack5l_pair(sup_5l, [*xs, s2[:12], s2[12:24], s2[24:], s1, s0, ta60, tb60]) == ([60, 60], [60, 60])
    # 5O-4: Attack 5L_3split_right_novel_short (n1=30 after s0)
    n1_30 = _mseq(30, 21130)
    assert _run_attack5l_pair(sup_5l, [*xs, s2[:12], s2[12:24], s2[24:], s1, s0, n1_30, ta60]) == ([30, 60], [60, 30])

    # Attack 5L_two_sided (100%, 92%, and 10+10 split)
    sl_20, sm_20, sr_20 = _mseq(20, 21140), _mseq(20, 21141), _mseq(20, 21142)
    yr = [_mseq(40, 21150 + k) for k in range(4)]
    sup_2s_100 = [
        p500a + "".join(xs) + sl_20 + p500b + sl_20 + sm_20 + sr_20 + p500c + sr_20 + "".join(yr) + p500a
    ]
    assert _run_attack5l_pair(sup_2s_100, [*xs, sl_20, sm_20, sr_20, *yr]) == ([], [])
    assert _run_attack5l_pair(sup_2s_100, [*xs, sl_20[:10], sl_20[10:], sm_20, sr_20[:10], sr_20[10:], *yr]) == ([], [])
    site_b_92 = mutate_seq_snps(sl_20 + sm_20 + sr_20, [5, 15, 30, 45])
    sup_2s_92 = [
        p500a + "".join(xs) + sl_20 + p500b + site_b_92 + p500c + sr_20 + "".join(yr) + p500a
    ]
    assert _run_attack5l_pair(sup_2s_92, [*xs, sl_20, sm_20, sr_20, *yr]) == ([], [])

    # Isolated novel short pieces 10..49 bp -> 0 false drops (FWD == MIR)
    for iso_len in (10, 15, 25, 35, 45, 49):
        iso_s = _mseq(iso_len, 21160 + iso_len)
        assert _run_attack5l_pair([p500a + p500b], [p500a, iso_s, p500b]) == ([iso_len], [iso_len])

    # (2) Finding #4 zero-FN (devilish 1a-1d, 5O-3, 5O-3b) & interleaved_96pct_te no-thrash
    def _no19_93pct(seq: str) -> str:
        # 1 SNP every 18 bp -> 94.4% identity (>= 0.90) with ZERO exact 19-mers!
        return mutate_seq_snps(seq, list(range(17, len(seq), 18)))

    def _mutate_88pct_l(seq: str) -> str:
        # 6 clustered SNPs per 50 bp block -> 88% (< 0.90) with 38 bp exact runs so asm20 aligns & rejects!
        return _mutate_per_50bp_block(seq, (1, 3, 5, 7, 9, 11))

    def _mutate_88pct_r(seq: str) -> str:
        return _mutate_per_50bp_block(seq, (38, 40, 42, 44, 46, 48))

    def _mutate_84pct(seq: str) -> str:
        return _mutate_per_50bp_block(seq, (1, 3, 5, 7, 9, 11, 13, 15))

    # 1a: a800 in tail_wave (P_old), b400 in tail_delta (P_delta); query = a800 + b400_no19 -> 100% masked!
    a800 = _mseq(800, 21201)
    b400 = _mseq(400, 21202)
    trig_a = mutate_seq_snps(a800, [50])  # non-exact -> builds tail_wave containing a800
    q_1a = a800 + _no19_93pct(b400)
    sols_1a = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 800, "+", a800, "source_contiguous", ("H0#1",)),), 10, "a0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 800, "+", trig_a, "source_contiguous", ("H1#1",)),), 9, "a1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 400, "+", b400, "source_contiguous", ("H2#1",)),), 8, "a2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, len(q_1a), "+", q_1a, "source_contiguous", ("H3#1",)),), 7, "a3"),
    }
    _run_s55(sols_1a)
    assert sols_1a[(3, 4)].non_backbone_reps == ()

    # 1b: a200 + a600_84pct + b400_no19 -> rejected 84% alignment leaves 400 bp short_slice hitting P_delta!
    a200 = _mseq(200, 21210)
    a600 = _mseq(600, 21211)
    p_old_1b = a200 + a600
    trig_1b = mutate_seq_snps(p_old_1b, [50])
    q_1b = a200 + _mutate_84pct(a600) + _no19_93pct(b400)
    sols_1b = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 800, "+", p_old_1b, "source_contiguous", ("H0#1",)),), 10, "b0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 800, "+", trig_1b, "source_contiguous", ("H1#1",)),), 9, "b1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 400, "+", b400, "source_contiguous", ("H2#1",)),), 8, "b2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, len(q_1b), "+", q_1b, "source_contiguous", ("H3#1",)),), 7, "b3"),
    }
    _run_s55(sols_1b)
    assert _kept_lens(sols_1b[(3, 4)]) == [600]

    # 1c: min_sv_len=100, dsr=80 residual hits P_delta (0 short_res_bp!)
    b80 = _mseq(80, 21220)
    b80_pad = _mseq(120, 21221)
    q_1c = a800 + _no19_93pct(b80)
    sols_1c = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 800, "+", a800, "source_contiguous", ("H0#1",)),), 10, "c0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 800, "+", trig_a, "source_contiguous", ("H1#1",)),), 9, "c1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 200, "+", b80 + b80_pad, "source_contiguous", ("H2#1",)),), 8, "c2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, len(q_1c), "+", q_1c, "source_contiguous", ("H3#1",)),), 7, "c3"),
    }
    _, dsr_1c, _ = _run_s55(sols_1c, min_sv_len=100)
    assert sols_1c[(3, 4)].non_backbone_reps == () and dsr_1c == 0

    # 1d: _find_subinterval_homologous_hit_in_waves tail_dirty fast-path (short piece in P_delta drops!)
    d_l = _mseq(200, 21230)
    d_s30 = _mseq(30, 21231)
    d_r = _mseq(200, 21232)
    d_l_trig = mutate_seq_snps(d_l, [10])
    sols_1d = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 200, "+", d_l, "source_contiguous", ("H0#1",)),), 10, "d0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 200, "+", d_r, "source_contiguous", ("H1#1",)),), 9, "d1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 200, "+", d_l_trig, "source_contiguous", ("H2#1",)),), 8, "d2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, 190, "+", d_l[-80:] + d_s30 + d_r[:80], "source_contiguous", ("H3#1",)),), 7, "d3"),
        (4, 5): _mk_sol(4, _mk_contig_pieces([d_l, d_s30, d_r]), 6, "d4"),
    }
    _run_s55(sols_1d)
    assert sols_1d[(4, 5)].non_backbone_reps == ()

    # 5O-3: protected_ins inside 88% rejected flank (fl600_88 + b200_no19 + fr600_88) -> b200 masked!
    fl600, fr600 = _mseq(600, 21240), _mseq(600, 21241)
    b200 = _mseq(200, 21242)
    p_old_5o3 = fl600 + fr600
    trig_5o3 = mutate_seq_snps(p_old_5o3, [50])
    q_5o3 = _mutate_88pct_l(fl600) + _no19_93pct(b200) + _mutate_88pct_r(fr600)
    sols_5o3 = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 1200, "+", p_old_5o3, "source_contiguous", ("H0#1",)),), 10, "o3_0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 1200, "+", trig_5o3, "source_contiguous", ("H1#1",)),), 9, "o3_1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 200, "+", b200, "source_contiguous", ("H2#1",)),), 8, "o3_2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, len(q_5o3), "+", q_5o3, "source_contiguous", ("H3#1",)),), 7, "o3_3"),
    }
    _run_s55(sols_5o3)
    kept_5o3 = _kept_lens(sols_5o3[(3, 4)])
    assert len(kept_5o3) == 2 and kept_5o3[0] == 600 and 599 <= kept_5o3[1] <= 601

    # 5O-3b: single bridged 'N'*200 88% flanks (separate fl600 & fr600 in P_old, b200 in P_delta) -> b200 masked!
    fl600b, fr600b = _mseq(600, 21250), _mseq(600, 21251)
    q_5o3b = _mutate_88pct_l(fl600b) + _no19_93pct(b200) + _mutate_88pct_r(fr600b)
    sols_5o3b = {
        (0, 1): _mk_sol(0, (Piece("H0#1", "H0#1#Chr01", 0, 600, "+", fl600b, "source_contiguous", ("H0#1",)),), 10, "o3b_0"),
        (1, 2): _mk_sol(1, (Piece("H1#1", "H1#1#Chr01", 0, 600, "+", fr600b, "source_contiguous", ("H1#1",)),), 9, "o3b_1"),
        (2, 3): _mk_sol(2, (Piece("H2#1", "H2#1#Chr01", 0, 600, "+", mutate_seq_snps(fl600b, [50]), "source_contiguous", ("H2#1",)),), 8, "o3b_2"),
        (3, 4): _mk_sol(3, (Piece("H3#1", "H3#1#Chr01", 0, 200, "+", b200, "source_contiguous", ("H3#1",)),), 7, "o3b_3"),
        (4, 5): _mk_sol(4, (Piece("H4#1", "H4#1#Chr01", 0, len(q_5o3b), "+", q_5o3b, "source_contiguous", ("H4#1",)),), 6, "o3b_4"),
    }
    _run_s55(sols_5o3b)
    assert _kept_lens(sols_5o3b[(4, 5)]) == [600, 600]

    # interleaved_96pct_te (N=400): wave_builds <= 3 & 100% parity
    base_te = _mseq(1500, 21260)
    n_te = 400
    sols_te: dict[tuple[int, int], ParallelGroupSolution] = {}
    sols_te[(0, 1)] = _mk_sol(0, (Piece("TE0#1", "TE0#1#Chr01", 0, 1500, "+", base_te, "source_contiguous", ("TE0#1",)),), n_te + 5, "te_base")
    for k in range(1, n_te):
        if k % 2 == 1:
            # 96% TE copy (2 SNPs per 50 bp block)
            off_k = (k * 3) % 40
            sq = _mutate_per_50bp_block(base_te, (off_k, off_k + 7))
        else:
            sq = _mseq(1500, 22000 + k)
        sols_te[(k, k + 1)] = _mk_sol(k, (Piece(f"H{k}#1", f"H{k}#1#Chr01", 0, 1500, "+", sq, "source_contiguous", (f"H{k}#1",)),), n_te + 5 - k, f"te_{k}")

    orig_build = linearizer_mod._build_pan_dedup_wave_index
    build_calls = 0

    def _spy_build(seqs: Sequence[str]) -> object:
        nonlocal build_calls
        build_calls += 1
        return orig_build(seqs)

    monkeypatch.setattr(linearizer_mod, "_build_pan_dedup_wave_index", _spy_build)
    pm_te, psr_te, drop_te = _run_s55(sols_te, n_nodes=n_te + 2)
    monkeypatch.undo()
    assert build_calls <= 3, f"Expected <= 3 wave builds on interleaved_96pct_te N=400, got {build_calls}"
    assert drop_te == n_te // 2 and psr_te == 0 and pm_te == (n_te // 2) * 1500

    # (3) Cross-interaction 5O-7: Attack 5L + 1 bp-MH D(10) + I+X SNP sliver + tail_dirty (FWD == MIR)
    s20_ci = _mseq(19, 21301) + "A"
    s25_ci = "G" + _mseq(24, 21302)
    fl_ci = _mseq(250, 21303)
    fr_ci = _mseq(250, 21304)
    # Query A combines Attack 5L left hijack + 1bp-MH D(10) between s20_ci and s25_ci + novel ta60
    sup_ci_a = p500a + "".join(xs) + s2 + p500b + s2 + s1 + s0 + p500c + fl_ci + s20_ci + "CGTACGTACA" + s25_ci + fr_ci
    # Query B has I+X SNP sliver covered on - strand in tail_dirty state
    ix_l, ix_r = _mseq(250, 21310), _mseq(250, 21311)
    ix_ins = _mseq(10, 21312)
    ix_sliv = _mseq(3, 21313)
    ix_sup_seq = revcomp_dna(ix_l + mutate_seq_snps(ix_sliv, [0]) + ix_r)

    for is_mir in (False, True):
        def _ori(s: str) -> str:
            return revcomp_dna(s) if is_mir else s

        def _ori_list(lst: list[str]) -> list[str]:
            return [revcomp_dna(s) for s in reversed(lst)] if is_mir else list(lst)

        q_a_pieces = _ori_list([*xs, s2, s1, s0, fl_ci, s20_ci, s25_ci, ta60])
        q_ix_pieces = _ori_list([ix_l, ix_ins, ix_sliv, ix_r])
        sols_ci = {
            (0, 1): _mk_sol(0, (Piece("S0#1", "S0#1#Chr01", 0, len(sup_ci_a), "+", _ori(sup_ci_a), "source_contiguous", ("S0#1",)),), 20, "ci0"),
            (1, 2): _mk_sol(1, (Piece("S1#1", "S1#1#Chr01", 0, len(sup_ci_a), "+", mutate_seq_snps(_ori(sup_ci_a), [100]), "source_contiguous", ("S1#1",)),), 19, "ci1"),
            (2, 3): _mk_sol(2, (Piece("S2#1", "S2#1#Chr01", 0, len(ix_sup_seq), "+", _ori(ix_sup_seq), "source_contiguous", ("S2#1",)),), 18, "ci2"),
            (3, 4): _mk_sol(3, _mk_contig_pieces(q_a_pieces, "QA#1"), 5, "ci3"),
            (4, 5): _mk_sol(4, _mk_contig_pieces(q_ix_pieces, "QIX#1"), 4, "ci4"),
        }
        _run_s55(sols_ci)
        exp_a = [60, 25] if is_mir else [25, 60]
        assert _kept_lens(sols_ci[(3, 4)]) == exp_a, f"5O-7 mir={is_mir} QA: {_kept_lens(sols_ci[(3, 4)])}"
        assert _kept_lens(sols_ci[(4, 5)]) == [10], f"5O-7 mir={is_mir} QIX: {_kept_lens(sols_ci[(4, 5)])}"


def test_iter11_diagonal_dp_equivalence_and_redteam_closures() -> None:
    """Iter 11 Design 2++ Regressions (`panflat/repeat_guard.py` Diagonal-DP):
    1. `Finding 1a` (`11 bp` `2 bp`-novel hot-potato), `Finding 1b` (`50 bp` `84%`-id SV
       spanning hit `mhl=50 & 0`, `0.90 & 0.80` + `ins_block_t_pieces_out`), and
       `s20 + nov50_mh5` non-false-rejection (`protected_ins` requires `min_ov > 0`).
    2. `H1` (`0.80..0.99`), 2-sided SNP interior-midpoint plateau attack, `H2x` short &
       long (`pt10=True, pL12=False, pR12=False`), and `H3` (`D@49 == D@51` across `0.80..0.99`).
    3. `BUG-R10-1..4` (`outer D(7)+X(1)`, `outer D(1)@0.99`, `30x` unbiased `1 bp` `D@0.99`
       `0/30` asym, `I(5)+D(1)` & `I(4)+D(2)` @ `0.90`, `D(4)+M(5)+X(1)`, flush `I(1)+M(1)+X(1)`,
       `5`-SNP @ `0.90`, `s34|s40@0.98` `(True, False)`, `CAG|G@0.99` `(False, False)`).
    4. Red-team closures: `Killshot A1/A2`, `C6` (`q_p=219`, `q_p=199`), `C7` `K=2` mirror
       parity, `A^200` live `mappy`, and `100 kb + I(7)` latency guard (`< 10 ms`).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _map_hits(t_seq: str, q_seq: str, short: bool = False) -> list[object]:
        aln = (
            mappy.Aligner(seq=t_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
            if short
            else mappy.Aligner(seq=t_seq, preset="asm20", best_n=50)
        )
        return list(aln.map(q_seq, cs=True))

    def _cov_triplet(
        t_seq: str,
        q_seq: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
        short: bool = False,
    ) -> tuple[bool, bool, bool]:
        q_len = len(q_seq)
        vf = any(
            hit_covers_subinterval_mappy(h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq)
            for h in _map_hits(t_seq, q_seq, short=short)
        )
        vr = any(
            hit_covers_subinterval_mappy(h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq)
            for h in _map_hits(revcomp_dna(t_seq), q_seq, short=short)
        )
        q_rc = revcomp_dna(q_seq)
        vm = any(
            hit_covers_subinterval_mappy(
                h, q_len - p_e, q_len - p_s, min_id, min_hit_len=mhl, query_seq=q_rc
            )
            for h in _map_hits(revcomp_dna(t_seq), q_rc, short=short)
        )
        return vf, vr, vm

    l0 = _mseq(1500, 33101)
    r0 = _mseq(1500, 33102)
    nov_r = _mseq(200, 33103)
    nov_sup = _mseq(200, 33104)

    # (1a) Finding 1a: 11 bp (2 bp novel) hot-potato split p1(20) + p2(10)
    mh9 = "ACGTTCGAC"
    m10 = _mseq(8, 33401) + "GG"
    s40 = "C" + _mseq(39, 33402)
    p1_atk = m10 + mh9 + "T"
    p2_atk = "A" + mh9
    q_f1a = l0[-200:] + p1_atk + p2_atk + s40 + nov_r + r0[:200]
    t_f1a = l0 + m10 + mh9 + s40 + nov_sup + r0
    for mhl in (50, 0):
        assert _cov_triplet(t_f1a, q_f1a, 200, 220, 0.90, mhl) == (False, False, False)
        assert _cov_triplet(t_f1a, q_f1a, 220, 230, 0.90, mhl) == (False, False, False)

    # (1b) Finding 1b: 50 bp 84%-id SV spanning hit (mhl=50 & 0, 0.90 & 0.80) + ins_block_t_pieces_out
    u42 = _mseq(42, 65201)
    nov8 = "GATCCATG"
    p_a = u42 + nov8[:4]
    p_b = nov8[4:] + u42
    q_f1b = l0[-200:] + p_a + p_b + r0[:200]
    t_f1b = l0 + u42 + r0
    for mid in (0.90, 0.80):
        for mhl in (50, 0):
            assert _cov_triplet(t_f1b, q_f1b, 200, 246, mid, mhl) == (False, False, False)
            assert _cov_triplet(t_f1b, q_f1b, 246, 292, mid, mhl) == (False, False, False)
    f1b_hits = _map_hits(t_f1b, q_f1b)
    assert f1b_hits
    blk_a: set[int] = set()
    blk_b: set[int] = set()
    assert hit_covers_subinterval_mappy(
        f1b_hits[0], 200, 246, 0.90, min_hit_len=50, query_seq=q_f1b, ins_block_t_pieces_out=blk_a
    ) is False
    assert hit_covers_subinterval_mappy(
        f1b_hits[0], 246, 292, 0.90, min_hit_len=50, query_seq=q_f1b, ins_block_t_pieces_out=blk_b
    ) is False
    assert blk_a == {0} and blk_b == {0}

    # (1c) s20 + nov50_mh5 non-false-rejection: protected_ins MUST NOT fire when min_ov == 0
    mh5 = "ACGTT"
    s20_nf = _mseq(15, 51001) + mh5
    nov50_mh5 = _mseq(45, 51002) + mh5
    s40_nf = _mseq(40, 51003)
    q_nf = l0[-200:] + s20_nf + nov50_mh5 + s40_nf + r0[:200]
    t_nf = l0 + s20_nf + s40_nf + r0
    assert _cov_triplet(t_nf, q_nf, 200, 220, 0.90, 50) == (True, True, True)
    assert _cov_triplet(t_nf, q_nf, 220, 270, 0.90, 50) == (False, False, False)

    # (2a) H1 (I_len > p_len, non-tandem 12 bp ins) across 0.80..0.99
    fl_h1 = _mseq(198, 5001) + "C"
    fr_h1 = "C" + _mseq(199, 5002)
    unit10 = "ACGTTCGAGC"
    q_h1 = fl_h1 + unit10 + "TA" + unit10 + fr_h1
    t_h1 = fl_h1 + unit10 + fr_h1
    for mid in (0.80, 0.90, 0.95, 0.99):
        for mhl in (50, 0):
            assert _cov_triplet(t_h1, q_h1, 200, 210, mid, mhl) == (False, False, False)

    # (2b) 2-sided SNP interior-midpoint plateau attack (s=196, 198, 200, REV -> ALL False)
    fl_p = _mseq(180, 51011) + "A" + _mseq(9, 51012) + "A" + _mseq(5, 51013) + "C"
    fr_p = "C" + _mseq(5, 51014) + "A" + _mseq(9, 51015) + "A" + _mseq(180, 51016)
    q_plat = fl_p + ("G" * 10) + fr_p
    for cs_plat, st_plat in (
        (":180*ca:9*ca:6+gggggg:10*ca:9*ca:180", 1),
        (":180*ca:9*ca:8+gggggg:8*ca:9*ca:180", 1),
        (":180*ca:9*ca:10+gggggg:6*ca:9*ca:180", 1),
        (":180*gt:9*gt:6+cccccc:10*gt:9*gt:180", -1),
    ):
        h_plat = SimpleNamespace(
            q_st=0, q_en=len(q_plat), r_st=0, r_en=len(q_plat) - 6, strand=st_plat, cs=cs_plat
        )
        assert hit_covers_subinterval_mappy(
            h_plat, 200, 202, 0.90, min_hit_len=50, query_seq=q_plat
        ) is False

    # (2c) H2x short & long (under Iter 12 s_cover dual, u1(12)+pt(10)+tail2(2) is a
    # 12 bp quasi-tandem so pt10=False, pL12=False, pR12=False; FWD == REV == MIR == False)
    u0_h2 = "GACGTCTAGCTT"
    u1_h2 = "GCCGTCTAGCTT"
    u2_h2 = "GACGTCTAGCTG"
    fl24_h2 = _mseq(24, 51101)
    fr50_h2 = _mseq(50, 51102)
    t_h2s = fl24_h2 + u0_h2 + fr50_h2
    q_h2s = fl24_h2 + u1_h2 + u2_h2 + fr50_h2
    assert _cov_triplet(t_h2s, q_h2s, 36, 46, 0.90, 50, short=True) == (False, False, False)
    assert _cov_triplet(t_h2s, q_h2s, 24, 36, 0.90, 50, short=True) == (False, False, False)
    assert _cov_triplet(t_h2s, q_h2s, 36, 48, 0.90, 50, short=True) == (False, False, False)
    h2s_f = SimpleNamespace(q_st=0, q_en=98, r_st=0, r_en=86, strand=1, cs=":24+gccgtctagctt:11*tg:50")
    h2s_r = SimpleNamespace(q_st=0, q_en=98, r_st=0, r_en=86, strand=-1, cs=":50+cagctagacgtc:10*tg:25")
    assert hit_covers_subinterval_mappy(h2s_f, 36, 46, 0.90, min_hit_len=50, query_seq=q_h2s) is False
    assert hit_covers_subinterval_mappy(h2s_r, 36, 46, 0.90, min_hit_len=50, query_seq=q_h2s) is False
    assert hit_covers_subinterval_mappy(h2s_f, 24, 36, 0.90, min_hit_len=50, query_seq=q_h2s) is False
    assert hit_covers_subinterval_mappy(h2s_r, 24, 36, 0.90, min_hit_len=50, query_seq=q_h2s) is False
    assert hit_covers_subinterval_mappy(h2s_f, 36, 48, 0.90, min_hit_len=50, query_seq=q_h2s) is False
    assert hit_covers_subinterval_mappy(h2s_r, 36, 48, 0.90, min_hit_len=50, query_seq=q_h2s) is False

    fl200_h2 = _mseq(200, 51103)
    fr200_h2 = _mseq(200, 51104)
    t_h2l = fl200_h2 + u0_h2 + fr200_h2
    q_h2l = fl200_h2 + u1_h2 + u2_h2 + fr200_h2
    assert _cov_triplet(t_h2l, q_h2l, 212, 222, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_h2l, q_h2l, 200, 212, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_h2l, q_h2l, 212, 224, 0.90, 50) == (False, False, False)

    # (2d) H3: structural D(200) at q_p=49 vs q_p=51 (2-bp D+M+X swap) across 0.80..0.99
    fl49_h3 = _mseq(49, 51201)
    d_core_h3 = _mseq(198, 51202)
    tail_h3 = _mseq(198, 51203)
    d200_49 = "AG" + d_core_h3
    d200_51 = d_core_h3 + "CA"
    q_h3 = fl49_h3 + "CG" + tail_h3
    h3_49 = SimpleNamespace(q_st=0, q_en=len(q_h3), r_st=0, r_en=len(q_h3) + 200, strand=1, cs=f":49-{d200_49.lower()}:1*ag:198")
    h3_51 = SimpleNamespace(q_st=0, q_en=len(q_h3), r_st=0, r_en=len(q_h3) + 200, strand=1, cs=f":49*ac:1-{d200_51.lower()}:198")
    for mid, exp_h3 in ((0.80, True), (0.90, True), (0.95, True), (0.99, False)):
        r49 = hit_covers_subinterval_mappy(h3_49, 0, 49, mid, min_hit_len=50, query_seq=q_h3)
        r51 = hit_covers_subinterval_mappy(h3_51, 0, 49, mid, min_hit_len=50, query_seq=q_h3)
        assert r49 == r51 is exp_h3

    # (3a) BUG-R10-1: outer D(7)+X(1) at q_p==p_s + outer D(1)@0.99 short flank
    l0_1 = _mseq(1500, 93201)[:-1] + "T"
    s25_1 = "T" + _mseq(24, 93202)
    q_r10_1 = l0_1[-200:] + "C" + s25_1
    t_r10_1 = l0_1 + "GTGTTAA" + "G" + s25_1 + nov_sup + r0
    assert _cov_triplet(t_r10_1, q_r10_1, 200, 201, 0.90, 50) == (True, True, True)
    q_od1 = _mseq(49, 51301) + "A" + _mseq(195, 51302)
    h_od1_f = SimpleNamespace(q_st=0, q_en=len(q_od1), r_st=0, r_en=len(q_od1) + 1, strand=1, cs=":49-a:196")
    h_od1_r = SimpleNamespace(q_st=0, q_en=len(q_od1), r_st=0, r_en=len(q_od1) + 1, strand=-1, cs=":195-t:50")
    assert hit_covers_subinterval_mappy(h_od1_f, 49, 50, 0.99, min_hit_len=50, query_seq=q_od1) is True
    assert hit_covers_subinterval_mappy(h_od1_r, 49, 50, 0.99, min_hit_len=50, query_seq=q_od1) is True

    # (3b) BUG-R10-2: 30x unbiased random 1 bp D@0.99 (0/30 asym) + I(5)+D(1) & I(4)+D(2) @ 0.90
    rng_d1 = random.Random(51400)
    d1_asym = 0
    for _ in range(30):
        fl_d = "".join(rng_d1.choices("ACGT", k=200))
        s20_d = "".join(rng_d1.choices("ACGT", k=20))
        d1_b = rng_d1.choice("ACGT")
        s40_d = "".join(rng_d1.choices("ACGT", k=40))
        fr_d = "".join(rng_d1.choices("ACGT", k=200))
        q_d1 = fl_d + s20_d + s40_d + fr_d
        t_d1 = fl_d + s20_d + d1_b + s40_d + fr_d
        vf, vr, vm = _cov_triplet(t_d1, q_d1, 200, 220, 0.99, 50)
        if not (vf == vr == vm):
            d1_asym += 1
    assert d1_asym == 0, f"Unbiased 1 bp D@0.99 asymmetries: {d1_asym}/30"

    # Multi-indel I(5)+D(1) and I(4)+D(2) @ 0.90
    fl_md = _mseq(200, 51410)
    fr_md = _mseq(200, 51411)
    q_i5d1 = fl_md + "ACGTACGTACGTA" + "GGTAC" + "CTGACTG" + fr_md
    t_i5d1 = fl_md + "ACGTACGTACGTA" + "CTG" + "A" + "ACTG" + fr_md
    vf1, vr1, vm1 = _cov_triplet(t_i5d1, q_i5d1, 200, 213, 0.90, 50)
    assert vf1 == vr1 == vm1 is True
    q_i4d2 = fl_md + "TGCAACGTTAGCA" + "CCGT" + "ATCGTAG" + fr_md
    t_i4d2 = fl_md + "TGCAACGTTAGCA" + "ATC" + "GA" + "GTAG" + fr_md
    vf2, vr2, vm2 = _cov_triplet(t_i4d2, q_i4d2, 200, 213, 0.90, 50)
    assert vf2 == vr2 == vm2 is True

    # (3c) BUG-R10-3: D(4)+M(5)+X(1)
    s15_3 = _mseq(15, 95200)
    s33_3 = _mseq(28, 95201) + "ACGAT"
    s40_3 = "C" + _mseq(39, 95202)
    q_r10_3 = s15_3 + s33_3 + s40_3
    t_r10_3 = l0 + nov_sup + s15_3 + s33_3[:-5] + "ACGA" + "ACGAT" + "G" + s40_3[1:] + nov_sup + r0
    assert _cov_triplet(t_r10_3, q_r10_3, 15, 48, 0.90, 50, short=True) == (True, True, True)

    # (3d) BUG-R10-4: flush I(1)+M(1)+X(1) @ 0.99, 5-SNP @ 0.90, s34|s40 @ 0.98 (Killshot E), CAG|G @ 0.99
    s34_4 = _mseq(33, 92301) + "A"
    s40_4 = "AG" + _mseq(38, 92302)
    q_r10_4 = l0[-200:] + s34_4 + s40_4 + nov_r + r0[:200]
    t_r10_4 = l0 + s34_4 + "T" + s40_4[2:] + nov_sup + r0
    assert _cov_triplet(t_r10_4, q_r10_4, 200, 234, 0.99, 50) == (True, True, True)
    # Killshot E: s34|s40 @ 0.98 MUST return (True, False) — NEVER double-drop s40!
    assert _cov_triplet(t_r10_4, q_r10_4, 200, 234, 0.98, 50) == (True, True, True)
    assert _cov_triplet(t_r10_4, q_r10_4, 234, 274, 0.98, 50) == (False, False, False)
    # 5-SNP s34 @ 0.90
    s34_5snp = mutate_seq_snps(s34_4, [2, 7, 13, 19, 25])
    q_5snp = l0[-200:] + s34_5snp + s40_4 + nov_r + r0[:200]
    assert _cov_triplet(t_r10_4, q_5snp, 200, 234, 0.90, 50) == (True, True, True)
    # CAG|G @ 0.99 pure-M tandem conserved (False, False) across FWD == REV == MIR
    fl_cg = _mseq(199, 51501) + "T"
    fr_cg = "T" + _mseq(199, 51502)
    q_cagg = fl_cg + "CAG" + "G" + fr_cg
    t_cagg = fl_cg + "CAG" + fr_cg
    assert _cov_triplet(t_cagg, q_cagg, 200, 203, 0.99, 50) == (False, False, False)
    assert _cov_triplet(t_cagg, q_cagg, 203, 204, 0.99, 50) == (False, False, False)

    # (4a) Killshot A1 (L=5, 1 SNP -> 44/50=88%) & Killshot A2 (L=12, 3 SNPs -> 35/50=70%)
    fl_k = _mseq(200, 51601)
    fr_k = _mseq(200, 51602)
    t_ka1 = fl_k + "ACGTT" + fr_k
    q_ka1 = fl_k + "ACATT" + "ACGTG" + fr_k
    for mid in (0.91, 0.95):
        assert _cov_triplet(t_ka1, q_ka1, 200, 205, mid, 50) == (False, False, False)
        assert _cov_triplet(t_ka1, q_ka1, 205, 210, mid, 50) == (False, False, False)

    u0_ka2 = "GACGTCTAGCTT"
    u1_ka2 = mutate_seq_snps(u0_ka2, [1, 3, 5])
    u2_ka2 = mutate_seq_snps(u0_ka2, [6, 8, 10])
    t_ka2 = fl_k + u0_ka2 + fr_k
    q_ka2 = fl_k + u1_ka2 + u2_ka2 + fr_k
    assert _cov_triplet(t_ka2, q_ka2, 200, 212, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_ka2, q_ka2, 212, 224, 0.90, 50) == (False, False, False)

    # (4b) C6 structural D(200) strictly inside (p_s, p_e) (q_p=219 -> False) & q_p=199 < p_s=200 (True == True)
    q_c6 = _mseq(200, 51610) + _mseq(20, 51611) + _mseq(200, 51612)
    nomh_ch = next(ch for ch in "ACGT" if ch not in (q_c6[218], q_c6[219]))
    d200_c6_nomh = nomh_ch * 200
    h_c6_219 = SimpleNamespace(
        q_st=0, q_en=len(q_c6), r_st=0, r_en=len(q_c6) + 200, strand=1, cs=f":219-{d200_c6_nomh.lower()}:201"
    )
    assert hit_covers_subinterval_mappy(h_c6_219, 200, 220, 0.90, min_hit_len=50, query_seq=q_c6) is False
    # q_p=199 < p_s=200 vs q_p=205: D(200) starts with q_c6[199:225] (26 bp MH across entire p=[200, 220)!)
    d200_c6_mh199 = q_c6[199:225] + _mseq(174, 51613)
    d200_c6_mh205 = q_c6[205:225] + _mseq(174, 51613) + q_c6[199:205]
    h_c6_199 = SimpleNamespace(
        q_st=0, q_en=len(q_c6), r_st=0, r_en=len(q_c6) + 200, strand=1, cs=f":199-{d200_c6_mh199.lower()}:221"
    )
    h_c6_205 = SimpleNamespace(
        q_st=0, q_en=len(q_c6), r_st=0, r_en=len(q_c6) + 200, strand=1, cs=f":205-{d200_c6_mh205.lower()}:215"
    )
    r199 = hit_covers_subinterval_mappy(h_c6_199, 200, 220, 0.90, min_hit_len=50, query_seq=q_c6)
    r205 = hit_covers_subinterval_mappy(h_c6_205, 200, 220, 0.90, min_hit_len=50, query_seq=q_c6)
    assert r199 == r205 is True

    # (4c) C7 K=2 mirror parity (:72+aaa:4-aa:73 vs :73-tt:4+ttt:72)
    q_c7 = _mseq(72, 51701) + "AAA" + "ACGT" + _mseq(73, 51702)
    q_c7_rc = revcomp_dna(q_c7)
    h_c7_f = SimpleNamespace(q_st=0, q_en=152, r_st=0, r_en=151, strand=1, cs=":72+aaa:4-aa:73")
    h_c7_r = SimpleNamespace(q_st=0, q_en=152, r_st=0, r_en=151, strand=-1, cs=":73-tt:4+ttt:72")
    h_c7_m = SimpleNamespace(q_st=0, q_en=152, r_st=0, r_en=151, strand=1, cs=":73-tt:4+ttt:72")
    for p_s7, p_e7 in ((72, 79), (75, 79), (70, 80)):
        for mid in (0.80, 0.90, 0.95, 0.99):
            for mhl in (50, 0):
                vf7 = hit_covers_subinterval_mappy(h_c7_f, p_s7, p_e7, mid, min_hit_len=mhl, query_seq=q_c7)
                vr7 = hit_covers_subinterval_mappy(h_c7_r, p_s7, p_e7, mid, min_hit_len=mhl, query_seq=q_c7)
                vm7 = hit_covers_subinterval_mappy(h_c7_m, 152 - p_e7, 152 - p_s7, mid, min_hit_len=mhl, query_seq=q_c7_rc)
                assert vf7 == vr7 == vm7, f"C7 K=2 mismatch at {(p_s7, p_e7, mid, mhl)}: {(vf7, vr7, vm7)}"

    # (4d) A^200 asymmetric homopolymer live mappy (FWD == REV == MIR == False)
    fl_a = _mseq(100, 51801)[:-1] + "C"
    fr_a = "G" + _mseq(100, 51802)[1:]
    t_a200 = fl_a + ("A" * 190) + fr_a
    q_a200 = fl_a + ("A" * 200) + fr_a
    assert _cov_triplet(t_a200, q_a200, 100, 110, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_a200, q_a200, 290, 300, 0.90, 50) == (False, False, False)

    # (4e) 100 kb + I(7) latency guard (< 10 ms/call)
    q_100k = "C" + ("A" * 100_007) + "G"
    h_100k = SimpleNamespace(
        q_st=0, q_en=len(q_100k), r_st=0, r_en=len(q_100k) - 7, strand=1, cs=":50001+aaaaaaa:50001"
    )
    assert hit_covers_subinterval_mappy(h_100k, 50001, 50008, 0.90, min_hit_len=50, query_seq=q_100k) is False
    t0 = time.perf_counter()
    for _ in range(5):
        assert hit_covers_subinterval_mappy(h_100k, 50001, 50008, 0.90, min_hit_len=50, query_seq=q_100k) is False
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
    assert avg_ms < 10.0, f"100 kb + I(7) too slow: {avg_ms:.3f} ms/call"


def test_iter12_surgical_closures_bug_b_outer_i_and_latency() -> None:
    """Iter 12 Surgical Closure Unit Regressions:
    1. BUG B (`s_cover` dual when `l_j > p_len`): `L60_30x4` (with `ins_block_t_pieces_out`),
       `L80_40x4`, `L20_10x4`, `L12_6x4`, `L6_3x4`, `L4_2x4` (`True@0.90, False@0.95`),
       `QT24_12x4`, `u48+"GT"+u48` (`49+49`), `U(60)*4/10` outer-`I` `s_cover`, and `100x`
       `K=2` `s_cover + D` live-`mappy` fuzz (`0/100` asym).
    2. `r11_1` Finding 1 (outer-`I` reachability + `M,X,I` + `M,X,M,I` + boundary-`X`):
       right/left ZERO-SNP `(AC)^100` & `A^200`, `A^130 + X`, `t=15/15b/15c` & `t=195`,
       `100x` ZERO-SNP STR right-boundary @ default `0.90` (`0/100` asym).
    3. `r11_1` Minor Finding 2 (dense poly-A `K=20, 40, 60` latency `< 10 ms/call`) &
       Minor Finding 3 (large `p_len = 500` structural `D` `_get_dx_for` parity).
    """
    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _cov_triplet(
        t_seq: str, q_seq: str, p_s: int, p_e: int, min_id: float = 0.90, mhl: int = 50
    ) -> tuple[bool, bool, bool]:
        q_len = len(q_seq)
        vf = any(
            hit_covers_subinterval_mappy(h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq)
            for h in mappy.Aligner(seq=t_seq, preset="asm20", best_n=50).map(q_seq, cs=True)
        )
        aln_rc = mappy.Aligner(seq=revcomp_dna(t_seq), preset="asm20", best_n=50)
        vr = any(
            hit_covers_subinterval_mappy(h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq)
            for h in aln_rc.map(q_seq, cs=True)
        )
        q_rc = revcomp_dna(q_seq)
        vm = any(
            hit_covers_subinterval_mappy(
                h, q_len - p_e, q_len - p_s, min_id, min_hit_len=mhl, query_seq=q_rc
            )
            for h in aln_rc.map(q_rc, cs=True)
        )
        return vf, vr, vm

    # (1a) BUG B: L60_30x4, L80_40x4, L20_10x4, L12_6x4, L6_3x4, L4_2x4, QT24_12x4, u48+"GT"+u48
    fl200, fr200 = _mseq(200, 78001), _mseq(200, 78002)
    for u_len, piece_len, mid, mhl, exp in (
        (60, 30, 0.90, 50, False),
        (60, 30, 0.80, 50, False),
        (60, 30, 0.90, 0, False),
        (80, 40, 0.90, 50, False),
        (20, 10, 0.90, 50, False),
        (12, 6, 0.90, 50, False),
        (6, 3, 0.90, 50, False),
        (4, 2, 0.90, 50, True),
        (4, 2, 0.95, 50, False),
    ):
        u_seq = _mseq(u_len, 78100 + u_len)
        q_b = fl200 + u_seq + u_seq + fr200
        t_b = fl200 + u_seq + fr200
        for i in range(4):
            ps, pe = 200 + i * piece_len, 200 + (i + 1) * piece_len
            assert _cov_triplet(t_b, q_b, ps, pe, mid, mhl) == (exp, exp, exp)

    # Verify ins_block_t_pieces_out records target piece on L60_30x4
    u60 = _mseq(60, 78160)
    q_l60 = fl200 + u60 + u60 + fr200
    t_l60 = fl200 + u60 + fr200
    l60_hits = list(mappy.Aligner(seq=t_l60, preset="asm20", best_n=50).map(q_l60, cs=True))
    assert l60_hits
    for i in range(4):
        blk_s: set[int] = set()
        assert hit_covers_subinterval_mappy(
            l60_hits[0], 200 + i * 30, 200 + (i + 1) * 30, 0.90,
            min_hit_len=50, query_seq=q_l60, ins_block_t_pieces_out=blk_s,
        ) is False
        assert blk_s == {0}

    # QT24_12x4 (83% quasi-tandem split 12x4)
    u24 = _mseq(24, 78200)
    u24_1 = mutate_seq_snps(u24, [2, 7])
    u24_2 = mutate_seq_snps(u24, [14, 19])
    q_qt24 = fl200 + u24_1 + u24_2 + fr200
    t_qt24 = fl200 + u24 + fr200
    for i in range(4):
        assert _cov_triplet(t_qt24, q_qt24, 200 + i * 12, 200 + (i + 1) * 12, 0.90, 50) == (False, False, False)

    # u48 + "G" + "T" + u48 (50 bp SV split 49+49)
    u48 = _mseq(48, 78300)
    q_49 = fl200 + u48 + "GT" + u48 + fr200
    t_49 = fl200 + u48 + fr200
    assert _cov_triplet(t_49, q_49, 200, 249, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_49, q_49, 249, 298, 0.90, 50) == (False, False, False)

    # U(60)*4 & U(60)*10 outer-I s_cover parity
    for reps in (4, 10):
        q_ur = fl200 + u60 * reps + fr200
        t_ur = fl200 + u60 * (reps - 1) + fr200
        assert _cov_triplet(t_ur, q_ur, 200, 230, 0.90, 50) == (False, False, False)
        assert _cov_triplet(t_ur, q_ur, 200 + reps * 60 - 30, 200 + reps * 60, 0.90, 50) == (False, False, False)

    # (1b) 100x K=2 s_cover + D live mappy fuzz (0/100 asym)
    rng_k2 = random.Random(271828)
    k2_asym = 0
    for t_i in range(100):
        fl_k2 = "".join(rng_k2.choices("ACGT", k=120))
        fr_k2 = "".join(rng_k2.choices("ACGT", k=120))
        ul_k2 = rng_k2.choice([3, 4, 5, 8, 12])
        u_k2 = "".join(rng_k2.choices("ACGT", k=ul_k2))
        gap_k2 = "".join(rng_k2.choices("ACGT", k=rng_k2.randint(3, 15)))
        d_k2 = "".join(rng_k2.choices("ACGT", k=rng_k2.randint(1, 4)))
        if t_i % 2 == 0:
            q_k2 = fl_k2 + u_k2 + u_k2 + gap_k2 + fr_k2
            t_k2 = fl_k2 + u_k2 + gap_k2 + d_k2 + fr_k2
            ps_k2 = 120 + rng_k2.randint(0, ul_k2)
        else:
            q_k2 = fl_k2 + gap_k2 + u_k2 + u_k2 + fr_k2
            t_k2 = fl_k2 + d_k2 + gap_k2 + u_k2 + fr_k2
            ps_k2 = 120 + len(gap_k2) + rng_k2.randint(0, ul_k2)
        pe_k2 = ps_k2 + rng_k2.randint(1, ul_k2 - 1)
        mid_k2 = rng_k2.choice([0.80, 0.90, 0.95, 0.99])
        vf, vr, vm = _cov_triplet(t_k2, q_k2, ps_k2, pe_k2, mid_k2, 50)
        if not (vf == vr == vm):
            k2_asym += 1
    assert k2_asym == 0, f"K=2 s_cover+D asymmetries: {k2_asym}/100"

    # (2a) r11_1 Finding 1: right/left ZERO-SNP (AC)^100 & A^200, A^130 + X, t=15/15b/15c & t=195
    fl_s = _mseq(100, 82001)[:-1] + "T"
    fr_s = "G" + _mseq(100, 82002)[1:]
    for rep_q, rep_t in (("A" * 200, "A" * 190), ("AC" * 100, "AC" * 95)):
        q_sr = fl_s + rep_q + fr_s
        t_sr = fl_s + rep_t + fr_s
        for mhl in (50, 0):
            assert _cov_triplet(t_sr, q_sr, 285, 305, 0.90, mhl) == (False, False, False)
            assert _cov_triplet(t_sr, q_sr, 290, 305, 0.90, mhl) == (False, False, False)
            assert _cov_triplet(t_sr, q_sr, 95, 115, 0.90, mhl) == (False, False, False)

    # A^130 + X at p_e - 1
    q_ax = fl_s + ("A" * 130) + "C" + fr_s
    t_ax = fl_s + ("A" * 120) + "G" + fr_s
    assert _cov_triplet(t_ax, q_ax, 211, 231, 0.90, 50) == (False, False, False)

    # t=15/15b/15c (GAACT + edge SNP) & t=195 ((AG)^108 + A vs (AG)^103 + C @ 0.80)
    fl15 = _mseq(119, 15) + "T"
    fr15 = _mseq(120, 16)
    for qb in ("A", "G", "T"):
        q15 = fl15[:119] + "T" + ("GAACT" * 32) + qb + fr15
        t15 = fl15[:119] + "T" + ("GAACT" * 30) + "C" + fr15
        assert _cov_triplet(t15, q15, 139, 168, 0.99, 50) == (False, False, False)
    fl195 = _mseq(119, 195) + "T"
    fr195 = "T" + _mseq(119, 196)
    q195 = fl195 + ("AG" * 108) + "A" + fr195
    t195 = fl195 + ("AG" * 103) + "C" + fr195
    assert _cov_triplet(t195, q195, 194, 214, 0.80, 50) == (False, False, False)

    # (2b) 100x ZERO-SNP STR right-boundary @ DEFAULT 0.90 (0/100 asym)
    rng_str = random.Random(99999)
    str_asym = 0
    for _ in range(100):
        ul = rng_str.choice([1, 2, 3, 4])
        u = "".join(rng_str.choices("ACGT", k=ul))
        nr = rng_str.randint(140 // ul, 220 // ul)
        ic = rng_str.randint(max(1, 6 // ul), max(1, 14 // ul))
        il = ic * ul
        fl_r = "".join(rng_str.choices("ACGT", k=120))
        fr_r = "".join(rng_str.choices("ACGT", k=120))
        q_r = fl_r + u * (nr + ic) + fr_r
        t_r = fl_r + u * nr + fr_r
        ps_r = 120 + (nr + ic) * ul - il
        pe_r = ps_r + il + rng_str.randint(1, 10)
        vf, vr, vm = _cov_triplet(t_r, q_r, ps_r, pe_r, 0.90, 50)
        if not (vf == vr == vm):
            str_asym += 1
    assert str_asym == 0, f"100x ZERO-SNP STR right-boundary @ 0.90 asym: {str_asym}/100"

    # (3a) r11_1 Minor Finding 2: dense poly-A K = 20, 40, 60 latency guard (< 10 ms/call)
    for k_val in (20, 40, 60):
        q_pa = ("C" * 100) + ("A" * (k_val * 3 + 100)) + ("G" * 100)
        rem_m = len(q_pa) - 100 - (k_val // 2) * 5
        cs_pa = ":100" + ("+a:2-a:2" * (k_val // 2)) + f":{rem_m}"
        r_len = 100 + (k_val // 2) * 5 + rem_m
        h_pa = SimpleNamespace(q_st=0, q_en=len(q_pa), r_st=0, r_en=r_len, strand=1, cs=cs_pa)
        hit_covers_subinterval_mappy(h_pa, 110, 120, 0.90, min_hit_len=50, query_seq=q_pa)
        t0 = time.perf_counter()
        for _ in range(5):
            hit_covers_subinterval_mappy(h_pa, 110, 120, 0.90, min_hit_len=50, query_seq=q_pa)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
        assert dt_ms < 10.0, f"Dense poly-A K={k_val} too slow: {dt_ms:.3f} ms/call"

    # (3b) r11_1 Minor Finding 3: large p_len = 500 structural D _get_dx_for parity
    q_big = _mseq(100, 82101) + _mseq(500, 82102) + _mseq(100, 82103)
    d60_big = _mseq(60, 82104)
    h_big_l = SimpleNamespace(q_st=0, q_en=len(q_big), r_st=0, r_en=len(q_big) + 60, strand=1, cs=f":99-{d60_big.lower()}:601")
    h_big_r = SimpleNamespace(q_st=0, q_en=len(q_big), r_st=0, r_en=len(q_big) + 60, strand=1, cs=f":601-{d60_big.lower()}:99")
    assert hit_covers_subinterval_mappy(h_big_l, 100, 600, 0.90, min_hit_len=50, query_seq=q_big) is True
    assert hit_covers_subinterval_mappy(h_big_r, 100, 600, 0.90, min_hit_len=50, query_seq=q_big) is True


def test_iter13_cap_free_outer_i_unbounded_tandem_and_11th_order_closures() -> None:
    """Iter 13 11th-Order-Hardened Cap-Free `repeat_guard.py` Unit Regressions:
    1. `r12_1` Findings 1a/1b/2: `U(101)+U(101)`, `U(120)+U(120)`, `U(200)+U(200)`
       (`p=[500,540)` 100% inside `I(200)` + `ins_block_t_pieces_out == {0}`),
       `U(5000)+U(5000)`, `A^(2*105)`/`A^(2*150)`, and `K=4 U(80)+U(80)` / `U(95)+U(95)`
       + 3x `2 bp` D's.
    2. `r12_2` Finding 1: `9/9` `rel=+1..+9` interior-SNP blind zone, `Probe B` boundary `X`
       at `scan_lo` (`> 128 bp` distance), and `200x` random Long-STR + 1 interior SNP @ `0.90`
       (`0/200` asym).
    3. `explorer_iter13_2` `BS-3`–`BS-10`: quasi-tandem `U_1[L-5]+U_2[5]` (`L=80..250`),
       asymmetric `U_0+U_2(3 SNPs)` (`L=150, 200`), close 2-SNP `[-3, +4]` & multi-left
       2/3-SNP STRs, live `mappy` `D(40)+I(12)` vs `I(12)+D(40)`, two-sided
       `I(4)+M(1140)+I(670)` tie-break, `--min-sv-len 250 K=2`, `mhl=250 10x I(2)` `< 20 ms`,
       and non-tandem microhomology negative controls (`nov50_mh5/10/15`, `nov120_mh10`,
       `nov200_mh10` == `True`).
    4. `explorer_iter13_3` 11th-order Breaks B1–B5: `D(2..40)+I(120,200)` vs `I+D`, 3-copy
       `U(60/100/120)*3 + 1 SNP` & `A^300`/`A^600 + 1..3 SNPs`, `Step B` suboptimal
       `q_il=135` vs `q_il=115` (+ past-`s_hi` SNP), `K=3` `IDD/DID/DDI` & `IID/IDI/DII`
       orbit invariance, `K=5 total_cells > 2048`, live `K=4 mhl=250 < 25 ms`, and
       `100 kb + I(7) < 5 ms`.
    """
    _comp_lc = str.maketrans("acgtn", "tgcan")
    _cs_tok_re = re.compile(r":[0-9]+|\*[a-z]{2}|\+[a-z]+|-[a-z]+")

    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _rc_cs(cs: str) -> str:
        toks = _cs_tok_re.findall(cs)
        assert "".join(toks) == cs, f"Malformed cs: {cs}"
        out: list[str] = []
        for tok in reversed(toks):
            k = tok[0]
            if k == ":":
                out.append(tok)
            elif k == "*":
                out.append("*" + tok[1:].translate(_comp_lc))
            else:
                out.append(k + tok[1:][::-1].translate(_comp_lc))
        return "".join(out)

    def _cs_lens(cs: str) -> tuple[int, int]:
        q_l = 0
        r_l = 0
        for tok in _cs_tok_re.findall(cs):
            k = tok[0]
            if k == ":":
                v = int(tok[1:])
                q_l += v
                r_l += v
            elif k == "*":
                q_l += 1
                r_l += 1
            elif k == "+":
                q_l += len(tok) - 1
            else:
                r_l += len(tok) - 1
        return q_l, r_l

    def _eval_cs_triplet(
        q_seq: str,
        cs_fwd: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
    ) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
        q_l, r_l = _cs_lens(cs_fwd)
        assert q_l == len(q_seq), f"cs q_len {q_l} != len(q_seq) {len(q_seq)}"
        cs_rev = _rc_cs(cs_fwd)
        q_rc = revcomp_dna(q_seq)
        hf = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=1, cs=cs_fwd)
        hr = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=-1, cs=cs_rev)
        hm = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=1, cs=cs_rev)
        bf: set[int] = set()
        br: set[int] = set()
        bm: set[int] = set()
        vf = hit_covers_subinterval_mappy(
            hf, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=bf
        )
        vr = hit_covers_subinterval_mappy(
            hr, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=br
        )
        vm = hit_covers_subinterval_mappy(
            hm,
            q_l - p_e,
            q_l - p_s,
            min_id,
            min_hit_len=mhl,
            query_seq=q_rc,
            ins_block_t_pieces_out=bm,
        )
        return (vf, vr, vm), (bool(bf), bool(br), bool(bm))

    _hit_cache: dict[tuple[str, str], tuple[list[object], list[object], list[object]]] = {}

    def _cov_triplet_blk(
        t_seq: str,
        q_seq: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
    ) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
        q_len = len(q_seq)
        q_rc = revcomp_dna(q_seq)
        cached = _hit_cache.get((t_seq, q_seq))
        if cached is None:
            hf_list = list(mappy.Aligner(seq=t_seq, preset="asm20", best_n=50).map(q_seq, cs=True))
            aln_rc = mappy.Aligner(seq=revcomp_dna(t_seq), preset="asm20", best_n=50)
            hr_list = list(aln_rc.map(q_seq, cs=True))
            hm_list = list(aln_rc.map(q_rc, cs=True))
            cached = (hf_list, hr_list, hm_list)
            _hit_cache[(t_seq, q_seq)] = cached
        hf_list, hr_list, hm_list = cached
        bf: set[int] = set()
        vf = any(
            hit_covers_subinterval_mappy(
                h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=bf
            )
            for h in hf_list
        )
        br: set[int] = set()
        vr = any(
            hit_covers_subinterval_mappy(
                h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=br
            )
            for h in hr_list
        )
        bm: set[int] = set()
        vm = any(
            hit_covers_subinterval_mappy(
                h,
                q_len - p_e,
                q_len - p_s,
                min_id,
                min_hit_len=mhl,
                query_seq=q_rc,
                ins_block_t_pieces_out=bm,
            )
            for h in hm_list
        )
        return (vf, vr, vm), (bool(bf), bool(br), bool(bm))

    def _cov_triplet(
        t_seq: str, q_seq: str, p_s: int, p_e: int, min_id: float = 0.90, mhl: int = 50
    ) -> tuple[bool, bool, bool]:
        return _cov_triplet_blk(t_seq, q_seq, p_s, p_e, min_id=min_id, mhl=mhl)[0]

    # ------------------------------------------------------------------
    # (1) r12_1 Findings 1a / 1b / 2
    # ------------------------------------------------------------------
    fl500, fr500 = _mseq(500, 91001), _mseq(500, 91002)

    # 1a & 1b: U(101)+U(101), U(120)+U(120), U(200)+U(200) (p=[500,540) 100% inside I(200))
    for u_len in (101, 120, 200):
        u_seq = _mseq(u_len, 91100 + u_len)
        q_u = fl500 + u_seq + u_seq + fr500
        t_u = fl500 + u_seq + fr500
        v_l, b_l = _cov_triplet_blk(t_u, q_u, 500, 540, 0.90, 50)
        v_r, b_r = _cov_triplet_blk(t_u, q_u, 500 + 2 * u_len - 40, 500 + 2 * u_len, 0.90, 50)
        assert v_l == (False, False, False) and b_l == (True, True, True), (
            f"U({u_len}) left FAIL: v={v_l} b={b_l}"
        )
        assert v_r == (False, False, False) and b_r == (True, True, True), (
            f"U({u_len}) right FAIL: v={v_r} b={b_r}"
        )

    # Direct single-hit ins_block_t_pieces_out == {0} check on U(200)+U(200) p=[500, 540)
    u200 = _mseq(200, 91300)
    q200, t200 = fl500 + u200 + u200 + fr500, fl500 + u200 + fr500
    h200 = list(mappy.Aligner(seq=t200, preset="asm20", best_n=50).map(q200, cs=True))[0]
    blk200: set[int] = set()
    assert (
        hit_covers_subinterval_mappy(
            h200, 500, 540, 0.90, min_hit_len=50, query_seq=q200, ins_block_t_pieces_out=blk200
        )
        is False
    )
    assert blk200 == {0}

    # Unbounded U(5000)+U(5000) exact cap-free check (left/mid/right pieces + blk == {0})
    u5000 = _mseq(5000, 91500)
    q5000 = fl500 + u5000 + u5000 + fr500
    cs5000 = f":500+{u5000.lower()}:5500"
    for ps5, pe5 in ((500, 540), (5460, 5500), (5500, 5540), (10460, 10500)):
        v5, b5 = _eval_cs_triplet(q5000, cs5000, ps5, pe5, 0.90, 50)
        assert v5 == (False, False, False) and b5 == (True, True, True)

    # A^(2*105) & A^(2*150) exact cap-free check
    fl_pa = fl500[:-1] + "C"
    fr_pa = "G" + fr500[1:]
    for half_a in (105, 150):
        q_a = fl_pa + ("A" * (2 * half_a)) + fr_pa
        cs_a = f":500+{'a' * half_a}:{half_a + 500}"
        for ps_a, pe_a in ((500, 535), (500 + 2 * half_a - 35, 500 + 2 * half_a)):
            va, ba = _eval_cs_triplet(q_a, cs_a, ps_a, pe_a, 0.90, 50)
            assert va == (False, False, False) and ba == (True, True, True)
    # Also live mappy on A^(2*105) vs A^105
    q_a210 = fl_pa + ("A" * 210) + fr_pa
    t_a105 = fl_pa + ("A" * 105) + fr_pa
    assert _cov_triplet(t_a105, q_a210, 500, 535, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_a105, q_a210, 675, 710, 0.90, 50) == (False, False, False)

    # r12_1 Finding 2: K=4 U(80)+U(80) & U(95)+U(95) + 3x 2bp D's (p_len=20)
    mid50 = _mseq(50, 95601)
    sq_k4 = "".join(_mseq(15, 95700 + k) for k in range(3))
    st_k4 = "".join(_mseq(15, 95700 + k) + _mseq(2, 95800 + k) for k in range(3))
    s1_k4 = 500 + len(sq_k4) + 50
    for u_len_k4 in (80, 95):
        u_k4 = _mseq(u_len_k4, 95500 + u_len_k4)
        qk4 = fl500 + sq_k4 + mid50 + u_k4 + u_k4 + fr500
        tk4 = fl500 + st_k4 + mid50 + u_k4 + fr500
        assert _cov_triplet(tk4, qk4, s1_k4, s1_k4 + 20, 0.90, 50) == (False, False, False)
        assert _cov_triplet(
            tk4, qk4, s1_k4 + 2 * u_len_k4 - 20, s1_k4 + 2 * u_len_k4, 0.90, 50
        ) == (False, False, False)

    # ------------------------------------------------------------------
    # (2) r12_2 Finding 1: 9/9 rel=+1..+9, Probe B, 200x Long-STR+1-SNP
    # ------------------------------------------------------------------
    fl100 = _mseq(100, 93001)[:-1] + "T"
    fr100 = "G" + _mseq(100, 93002)[1:]
    q_bz = fl100 + ("AC" * 80) + fr100
    for rel in range(0, 11):
        rt_bz = list("AC" * 75)
        idx_snp = 50 + rel  # aligns to q_snp = 160 + rel = scan_lo + rel on FWD
        rt_bz[idx_snp] = next(c for c in "ACGT" if c != rt_bz[idx_snp])
        t_bz = fl100 + "".join(rt_bz) + fr100
        assert _cov_triplet(t_bz, q_bz, 240, 260, 0.90, 50) == (False, False, False), (
            f"Interior-SNP blind zone FAIL at rel={rel}"
        )

    # Probe B: boundary X at scan_lo with scan_lo - q_il > 128
    rt_pb = list("AC" * 155)
    rt_pb[160] = "T"
    t_pb = fl100 + "".join(rt_pb) + fr100
    q_pb = fl100 + ("AC" * 160) + fr100
    assert _cov_triplet(t_pb, q_pb, 350, 370, 0.90, 50) == (False, False, False)
    assert _cov_triplet(t_pb, q_pb, 355, 370, 0.90, 50) == (False, False, False)

    # 200x Random Long-STR + 1 Interior SNP @ DEFAULT 0.90 (seed=314159) -> 0/200 asym
    rng_s1 = random.Random(314159)
    asym_s1 = 0
    for _ in range(200):
        ul = rng_s1.choice([1, 2, 3, 4])
        u = "".join(rng_s1.choices("ACGT", k=ul))
        nr = rng_s1.randint(150 // ul, 220 // ul)
        ic = rng_s1.randint(max(1, 6 // ul), max(1, 16 // ul))
        fl_r = "".join(rng_s1.choices("ACGT", k=120))
        fr_r = "".join(rng_s1.choices("ACGT", k=120))
        rt_r = list(u * nr)
        sp = rng_s1.randint(20, len(rt_r) - 21)
        rt_r[sp] = next(c for c in "ACGT" if c != rt_r[sp])
        pe_r = 120 + (nr + ic) * ul
        ps_r = pe_r - rng_s1.randint(15, 25)
        r_tri = _cov_triplet(fl_r + "".join(rt_r) + fr_r, fl_r + u * (nr + ic) + fr_r, ps_r, pe_r)
        if not (r_tri[0] == r_tri[1] == r_tri[2]):
            asym_s1 += 1
    assert asym_s1 == 0, f"200x Long-STR + 1 interior SNP @ 0.90 asym: {asym_s1}/200"

    # ------------------------------------------------------------------
    # (3) explorer_iter13_2 BS-3..BS-10 + Negative Controls
    # ------------------------------------------------------------------
    fl1200, fr1200 = _mseq(1200, 94001), _mseq(1200, 94002)

    # BS-3: quasi-tandem U_1[L-5] + U_2[5] for L = 80..250 (tiling 30 bp pieces, kept_bp >= L)
    for l_qt in (80, 100, 150, 200, 250):
        u0_q = _mseq(l_qt, 94100 + l_qt)
        u1_q = mutate_seq_snps(u0_q, [l_qt - 5])
        u2_q = mutate_seq_snps(u0_q, [5])
        q_qt = fl1200 + u1_q + u2_q + fr1200
        t_qt = fl1200 + u0_q + fr1200
        kept_bp = 0
        for off in range(0, 2 * l_qt, 30):
            ps_q = 1200 + off
            pe_q = 1200 + min(off + 30, 2 * l_qt)
            tri_q = _cov_triplet(t_qt, q_qt, ps_q, pe_q, 0.90, 50)
            assert tri_q[0] == tri_q[1] == tri_q[2], f"BS-3 L={l_qt} off={off} asym: {tri_q}"
            if not tri_q[0]:
                kept_bp += pe_q - ps_q
        assert kept_bp >= l_qt, f"BS-3 L={l_qt} vaporized: kept_bp={kept_bp} < {l_qt}"

    # BS-3/4: asymmetric U_0 + U_2(3 SNPs at [20, 60, 90]) for L = 150, 200
    for l_as in (150, 200):
        u0_a = _mseq(l_as, 94300 + l_as)
        u2_a = mutate_seq_snps(u0_a, [20, 60, 90])
        q_as = fl1200 + u0_a + u2_a + fr1200
        t_as = fl1200 + u0_a + fr1200
        vl_a, bl_a = _cov_triplet_blk(t_as, q_as, 1200 + 30, 1200 + 60, 0.90, 50)
        vr_a, br_a = _cov_triplet_blk(t_as, q_as, 1200 + l_as + 30, 1200 + l_as + 60, 0.90, 50)
        assert vl_a == (True, True, True) and bl_a == (False, False, False)
        assert vr_a == (False, False, False) and br_a == (True, True, True)

    # BS-5: close 2-SNP [-3, +4] & multi-left 2/3-SNP STRs
    q_bs5 = fl100 + ("AC" * 100) + fr100
    for rel_offs in ((-70, -25), (-80, -45, -15), (-40, -20, -5), (-40, -20, 5), (-3, 4)):
        rt_5 = list("AC" * 95)
        # p=(280, 300), scan_lo=200; target offset inside STR corresponding to q_pos is (q_pos - 110)
        for ro in rel_offs:
            t_idx = 200 + ro - 110
            rt_5[t_idx] = next(c for c in "ACGT" if c != rt_5[t_idx])
        t_bs5 = fl100 + "".join(rt_5) + fr100
        assert _cov_triplet(t_bs5, q_bs5, 280, 300, 0.90, 50) == (False, False, False), (
            f"BS-5 multi-SNP FAIL at {rel_offs}"
        )

    # BS-6: live mappy D(40)+I(12) vs I(12)+D(40) across (AC)^150
    fl250 = _mseq(250, 94501)[:-1] + "T"
    fr250 = "G" + _mseq(250, 94502)[1:]
    d40_nr = _mseq(40, 94503)
    q_bs6 = fl250 + ("AC" * 150) + fr250
    for d_pos in (0, 20, 60, 100, 140, 200):
        t_rep = "AC" * 144
        t_bs6 = fl250 + t_rep[:d_pos] + d40_nr + t_rep[d_pos:] + fr250
        for ps6, pe6 in ((250, 270), (530, 550)):
            tri6 = _cov_triplet(t_bs6, q_bs6, ps6, pe6, 0.90, 50)
            assert tri6[0] == tri6[1] == tri6[2], f"BS-6 d_pos={d_pos} p=({ps6},{pe6}) asym: {tri6}"

    # BS-7: two-sided I(4) + M(1140) + I(670) symmetric tie-break
    q_bs7 = ("C" * 50) + ("A" * 1814) + ("G" * 50)
    cs_bs7 = ":50+aaaa:1140+" + ("a" * 670) + ":50"
    v_bs7, b_bs7 = _eval_cs_triplet(q_bs7, cs_bs7, 830, 870, 0.90, 50)
    assert v_bs7 == (False, False, False) and b_bs7 == (True, True, True)

    # BS-8: --min-sv-len 250 K=2 (U(120)+U(120)+D(2))
    u120_8 = _mseq(120, 94600)
    q_bs8 = fl500 + u120_8 + u120_8 + _mseq(30, 94601) + fr500
    t_bs8 = fl500 + u120_8 + _mseq(30, 94601) + "GT" + fr500
    assert _cov_triplet(t_bs8, q_bs8, 500, 540, 0.90, 250) == (False, False, False)
    assert _cov_triplet(t_bs8, q_bs8, 700, 740, 0.90, 250) == (False, False, False)

    # BS-10: mhl=250 10x I(2) s_cover (< 20 ms & returns False)
    q_bs10 = ("C" * 50) + ("A" * 300) + ("G" * 50)
    cs_bs10 = ":50" + ("+aa:20" * 10) + ":130"
    h_bs10 = SimpleNamespace(
        q_st=0, q_en=len(q_bs10), r_st=0, r_en=len(q_bs10) - 20, strand=1, cs=cs_bs10
    )
    t0 = time.perf_counter()
    res_bs10 = hit_covers_subinterval_mappy(h_bs10, 150, 151, 0.90, min_hit_len=250, query_seq=q_bs10)
    dt_bs10_ms = (time.perf_counter() - t0) * 1000.0
    assert res_bs10 is False and dt_bs10_ms < 20.0, f"BS-10 FAIL: res={res_bs10} dt={dt_bs10_ms:.2f}ms"

    # Non-tandem microhomology negative controls (MUST return True — zero false rejections)
    s40_nc = _mseq(40, 94703)
    for idx_nc, (nov_len, mh_len) in enumerate(((50, 5), (50, 10), (50, 15), (120, 10), (200, 10))):
        mh_seq = _mseq(mh_len, 94710 + idx_nc)
        s20_nc = _mseq(20 - mh_len, 94720 + idx_nc) + mh_seq
        nov_seq = _mseq(nov_len - mh_len, 94730 + idx_nc) + mh_seq
        q_nc = fl1200 + s20_nc + nov_seq + s40_nc + fr1200
        t_nc = fl1200 + s20_nc + s40_nc + fr1200
        assert _cov_triplet(t_nc, q_nc, 1200, 1220, 0.90, 50) == (True, True, True), (
            f"Negative control nov{nov_len}_mh{mh_len} false-rejected!"
        )

    # ------------------------------------------------------------------
    # (4) explorer_iter13_3 11th-Order Breaks B1–B5
    # ------------------------------------------------------------------
    fl200, fr200 = _mseq(200, 96001), _mseq(200, 96002)

    # B1: D(2..40)+I(120, 200) vs I(120, 200)+D(2..40) on right piece
    for l_b1 in (120, 200):
        u_b1 = _mseq(l_b1, 96100 + l_b1)
        q_b1 = fl200 + u_b1 + u_b1 + fr200
        for d_b1 in (2, 5, 15, 40):
            d_seq_b1 = _mseq(d_b1, 96200 + d_b1)
            cs_di = f":200-{d_seq_b1.lower()}+{u_b1.lower()}:{l_b1 + 200}"
            cs_id = f":200+{u_b1.lower()}-{d_seq_b1.lower()}:{l_b1 + 200}"
            v_di, b_di = _eval_cs_triplet(q_b1, cs_di, 200 + l_b1, 200 + l_b1 + 30, 0.90, 50)
            v_id, b_id = _eval_cs_triplet(q_b1, cs_id, 200 + l_b1, 200 + l_b1 + 30, 0.90, 50)
            assert v_di == v_id == (False, False, False) and b_di == b_id == (True, True, True), (
                f"B1 L={l_b1} d={d_b1} FAIL: di={(v_di, b_di)} id={(v_id, b_id)}"
            )

    # B2: 3-copy U(60/100/120)*3 + 1 SNP in copy 3 & A^300/A^600 + 1..3 SNPs
    for l_b2 in (60, 100, 120):
        u_b2 = _mseq(l_b2, 96300 + l_b2)
        snp_off = l_b2 // 2
        u_b2_mut = mutate_seq_snps(u_b2, [snp_off])
        t_ch = u_b2[snp_off].lower()
        q_ch = u_b2_mut[snp_off].lower()
        q_3c = fl200 + u_b2 + u_b2 + u_b2_mut + fr200
        # Suboptimal copy-1 I vs optimal copy-3 I
        cs_c1 = f":200+{u_b2.lower()}:{l_b2 + snp_off}*{t_ch}{q_ch}:{l_b2 - snp_off - 1 + 200}"
        cs_c3 = f":{200 + 2 * l_b2}+{u_b2_mut.lower()}:200"
        for ps_3, pe_3, exp_v, exp_b in (
            (200, 230, True, False),
            (200 + l_b2, 200 + l_b2 + 30, True, False),
            (200 + 2 * l_b2, 200 + 2 * l_b2 + 30, False, True),
        ):
            vc1, bc1 = _eval_cs_triplet(q_3c, cs_c1, ps_3, pe_3, 0.90, 50)
            vc3, bc3 = _eval_cs_triplet(q_3c, cs_c3, ps_3, pe_3, 0.90, 50)
            assert vc1 == vc3 == (exp_v, exp_v, exp_v) and bc1 == bc3 == (exp_b, exp_b, exp_b), (
                f"B2 3-copy L={l_b2} p=({ps_3},{pe_3}) FAIL: c1={(vc1, bc1)} c3={(vc3, bc3)}"
            )

    for a_tot in (300, 600):
        i_len_a = 100
        for n_snps in (1, 2, 3):
            snp_ps = [a_tot - 20 - 15 * k for k in range(n_snps)]
            snp_ps.sort()
            arr_a = ["A"] * a_tot
            for sp in snp_ps:
                arr_a[sp] = "C"
            rep_a = "".join(arr_a)
            q_an = fl200[:-1] + "T" + rep_a + "G" + fr200[1:]
            # Suboptimal I(100) at left (leaves all SNPs as *ac), vs optimal I(100) at right covering all SNPs
            parts_l = [f":200+{'a' * i_len_a}"]
            cur = i_len_a
            for sp in snp_ps:
                parts_l.append(f":{sp - cur}*ac")
                cur = sp + 1
            parts_l.append(f":{a_tot - cur + 200}")
            cs_al = "".join(parts_l)
            cs_ar = f":{200 + a_tot - i_len_a}+{rep_a[-i_len_a:].lower()}:200"
            for ps_an, pe_an in ((200, 230), (200 + a_tot - 30, 200 + a_tot)):
                vl_an, bl_an = _eval_cs_triplet(q_an, cs_al, ps_an, pe_an, 0.90, 50)
                vr_an, br_an = _eval_cs_triplet(q_an, cs_ar, ps_an, pe_an, 0.90, 50)
                assert vl_an == vr_an and bl_an == br_an, (
                    f"B2 A^{a_tot} n_snps={n_snps} p=({ps_an},{pe_an}) FAIL: l={(vl_an, bl_an)} r={(vr_an, br_an)}"
                )

    # B3: Step B suboptimal q_il=135 (right of SNP C@120) vs q_il=115 (+ past-s_hi SNP)
    fl_b3 = fl100[:-1] + "T"
    fr_b3 = "G" + fr100[1:]
    q_b3_l = fl_b3 + ("A" * 20) + "C" + ("A" * 189) + fr_b3
    cs_b3_opt = ":115+aaaaacaaaa:285"
    cs_b3_sub = ":120*ac:14+aaaaaaaaaa:265"
    vo_b3, bo_b3 = _eval_cs_triplet(q_b3_l, cs_b3_opt, 290, 310, 0.90, 50)
    vs_b3, bs_b3 = _eval_cs_triplet(q_b3_l, cs_b3_sub, 290, 310, 0.90, 50)
    assert vo_b3 == vs_b3 == (True, True, True) and bo_b3 == bs_b3 == (False, False, False)

    # Past-s_hi SNP: SNP C@289 lies strictly past s_hi=280 for p=(190, 210)
    q_b3_r = fl_b3 + ("A" * 189) + "C" + ("A" * 20) + fr_b3
    cs_b3_p_sub = ":100+aaaaaaaaaa:179*ac:120"
    cs_b3_p_opt = ":285+aaaacaaaaa:115"
    vp1, bp1 = _eval_cs_triplet(q_b3_r, cs_b3_p_sub, 190, 210, 0.90, 50)
    vp2, bp2 = _eval_cs_triplet(q_b3_r, cs_b3_p_opt, 190, 210, 0.90, 50)
    assert vp1 == vp2 == (True, True, True) and bp1 == bp2 == (False, False, False)

    # B4: Component C K=3 IDD/DID/DDI & IID/IDI/DII complete orbit invariance
    fl_b4 = fl200[:-1] + "C"
    fr_b4 = "G" + fr200[1:]
    q_idd = fl_b4 + ("A" * 120) + fr_b4
    cs_idd_all = (
        ":220+aaaaaaaa:10-ttt:10-ggg:272",
        ":220-ttt:10+aaaaaaaa:10-ggg:272",
        ":220-ttt:10-ggg:10+aaaaaaaa:272",
    )
    for p_off in (0, 20, 40):
        res_idd = [_eval_cs_triplet(q_idd, c_s, 200 + p_off, 200 + p_off + 20, 0.90, 50) for c_s in cs_idd_all]
        assert res_idd[0] == res_idd[1] == res_idd[2], f"B4 IDD/DID/DDI p_off={p_off} FAIL: {res_idd}"

    cs_iid_all = (
        ":220+aaaa:10+aaaa:10-cccccc:272",
        ":220+aaaa:10-cccccc:10+aaaa:272",
        ":220-cccccc:10+aaaa:10+aaaa:272",
    )
    for p_off in (0, 30, 60):
        res_iid = [_eval_cs_triplet(q_idd, c_s, 200 + p_off, 200 + p_off + 20, 0.90, 50) for c_s in cs_iid_all]
        assert res_iid[0] == res_iid[1] == res_iid[2], f"B4 IID/IDI/DII p_off={p_off} FAIL: {res_iid}"

    # B5: K=5 total_cells > 2048 (mhl=50 @0.95, mhl=250 @0.95, K=5 s_cover L=4),
    # live K=4 mhl=250 (< 25 ms), and 100 kb + I(7) (< 5 ms)
    fl_k5, fr_k5 = _mseq(250, 96601), _mseq(250, 96602)
    nov80_k5 = _mseq(80, 96603)
    # K=5 mhl=50 @0.95: I(80) expands scan to ~500 bp -> 6 * 500 = 3000 > 2048 cells
    q_k5_50 = fl_k5 + _mseq(100, 96604) + nov80_k5 + ("A" * 80) + fr_k5
    cs_k5_50_l = f":300-c:10-g:10-t:30+{nov80_k5.lower()}-aaa:330"
    cs_k5_50_r = f":300-c:10-g:10-t:30+{nov80_k5.lower()}:80-aaa:250"
    v5_l, b5_l = _eval_cs_triplet(q_k5_50, cs_k5_50_l, 460, 480, 0.95, 50)
    v5_r, b5_r = _eval_cs_triplet(q_k5_50, cs_k5_50_r, 460, 480, 0.95, 50)
    assert v5_l == v5_r == (True, True, True) and b5_l == b5_r == (False, False, False)

    # K=5 mhl=250 @0.95: n_q = 980 -> 6 * 980 = 5880 > 2048 cells
    q_k5_250 = _mseq(400, 96610) + ("A" * 200) + _mseq(400, 96611)
    cs_k5_250_l = ":300-c:10-g:10-t:10-c:70-" + ("a" * 20) + ":600"
    cs_k5_250_r = ":300-c:10-g:10-t:10-c:270-" + ("a" * 20) + ":400"
    v5_250_l, b5_250_l = _eval_cs_triplet(q_k5_250, cs_k5_250_l, 490, 510, 0.95, 250)
    v5_250_r, b5_250_r = _eval_cs_triplet(q_k5_250, cs_k5_250_r, 490, 510, 0.95, 250)
    assert v5_250_l == v5_250_r == (True, True, True) and b5_250_l == b5_250_r == (False, False, False)

    # K=5 s_cover L=4 (p_len=2, total_cells ~3000 > 2048)
    q_k5_sc = fl_k5 + _mseq(100, 96604) + nov80_k5 + _mseq(10, 96605) + ("A" * 84) + fr_k5
    cs_k5_sc_l = f":300-c:10-g:10-t:30+{nov80_k5.lower()}:10+aaaa:330"
    cs_k5_sc_r = f":300-c:10-g:10-t:30+{nov80_k5.lower()}:90+aaaa:250"
    v5_sc_l, b5_sc_l = _eval_cs_triplet(q_k5_sc, cs_k5_sc_l, 470, 472, 0.95, 50)
    v5_sc_r, b5_sc_r = _eval_cs_triplet(q_k5_sc, cs_k5_sc_r, 470, 472, 0.95, 50)
    assert v5_sc_l == v5_sc_r == (False, False, False) and b5_sc_l == b5_sc_r == (True, True, True)

    # Live mappy K=4 mhl=250 (3x D(2) + I(13)) < 25 ms/call & symmetric
    fl300 = _mseq(300, 96501)[:-1] + "T"
    fr300 = "G" + _mseq(300, 96502)[1:]
    d3_q = "".join(_mseq(20, 96510 + k) for k in range(3)) + _mseq(20, 96530)
    d3_t = "".join(_mseq(20, 96510 + k) + _mseq(2, 96520 + k) for k in range(3)) + _mseq(20, 96530)
    q_k4_250 = fl300 + d3_q + ("A" * 113) + fr300
    t_k4_250 = fl300 + d3_t + ("A" * 100) + fr300
    h_k4_f = list(mappy.Aligner(seq=t_k4_250, preset="asm20", best_n=50).map(q_k4_250, cs=True))
    h_k4_r = list(mappy.Aligner(seq=revcomp_dna(t_k4_250), preset="asm20", best_n=50).map(q_k4_250, cs=True))
    assert h_k4_f and h_k4_r
    p_k4_s, p_k4_e = 300 + len(d3_q) + 50, 300 + len(d3_q) + 70
    t0 = time.perf_counter()
    vf_250 = any(
        hit_covers_subinterval_mappy(h, p_k4_s, p_k4_e, 0.90, min_hit_len=250, query_seq=q_k4_250)
        for h in h_k4_f
    )
    dt_f_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    vr_250 = any(
        hit_covers_subinterval_mappy(h, p_k4_s, p_k4_e, 0.90, min_hit_len=250, query_seq=q_k4_250)
        for h in h_k4_r
    )
    dt_r_ms = (time.perf_counter() - t0) * 1000.0
    assert vf_250 == vr_250
    assert dt_f_ms < 25.0 and dt_r_ms < 25.0, (
        f"Live K=4 mhl=250 too slow: fwd={dt_f_ms:.2f}ms rev={dt_r_ms:.2f}ms"
    )

    # 100 kb + I(7) latency guard (< 5 ms/call)
    q_100k = "C" + ("A" * 100_007) + "G"
    h_100k = SimpleNamespace(
        q_st=0, q_en=len(q_100k), r_st=0, r_en=len(q_100k) - 7, strand=1, cs=":50001+aaaaaaa:50001"
    )
    assert hit_covers_subinterval_mappy(h_100k, 50001, 50008, 0.90, min_hit_len=50, query_seq=q_100k) is False
    t0 = time.perf_counter()
    for _ in range(5):
        assert (
            hit_covers_subinterval_mappy(h_100k, 50001, 50008, 0.90, min_hit_len=50, query_seq=q_100k)
            is False
        )
    avg_100k_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
    assert avg_100k_ms < 5.0, f"100 kb + I(7) too slow: {avg_100k_ms:.3f} ms/call"


def test_iter14_flank_snp_monotonicity_d_in_p_and_slo_closures() -> None:
    """Iter 14 Unified `repeat_guard.py` (Hunks 0–9) Unit Regressions:
    1. Fix 14-1 (Step B Phase-2 flank-SNP closure): direct M/X flank-SNP + Scenario S3
       strictly-better past-s_hi SNP + realistic 96–99%-flank Long-STR sweep across all
       5 motifs (`A`/`AC`/`ACG`/`ACGT`/`GAACT`) & `0–2` interior SNPs -> `0` asym.
    2. Fix 14-2 (`Attack 12-3`): exact `min_identity` sweep `[0.80..0.99]` strictly
       monotonic + `True` on `0.90..0.98`, `False` on `0.99`, `FWD == MIR`.
    3. Fix 14-3a (`p_has_int_d` removal): synthetic `U(105/120/200/250)` `I`-then-`D`
       vs `D`-then-`I`, `Attack 12-2`, and live `mappy` `A^320 dlen=8 @ 0.85`.
    4. Fix 14-3b (`> 8000` mirror-symmetric tie-break): `Attack 12-4` & exact-tie
       `d_l == d_r` (`FWD == MIR`).
    5. Bonus Catch E1: `2-SNP U(400)*2` adjacent (`[490,500)`, `[1300,1310)`) & interior
       (`[500,510)`, `[1290,1300)`) to `P` -> all 4 orientations equal.
    6. Bonus Catch E2 (`0 < min_hit_len < 50` monotonicity): `raw`, `k1`, `k2` at
       `mhl=20, 30` + randomized sweep across `mhl=(0, 15, 20, 30, 45, 50, 100, 250)`.
    7. Fix #6 & Bonus Catch E3 (`< 50 ms` SLO guards): `2D-WC-FAIL K=2..12 mhl=250 < 25 ms`,
       `K=4 (3I+1D sumI=46) < 25 ms`, `cat1_mhl250_k4_PASS < 50 ms`,
       `cat1_mhl100_k4_FAIL < 25 ms`, `poly-A K=6..60 < 10 ms`, `100k+I(7) < 5 ms`.
    """
    _comp_lc = str.maketrans("acgtn", "tgcan")
    _cs_tok_re = re.compile(r":[0-9]+|\*[a-z]{2}|\+[a-z]+|-[a-z]+")
    _tr_mut = {"A": "C", "C": "G", "G": "T", "T": "A"}

    def _mseq(n: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n))

    def _mut1(seq: str, pos: int) -> str:
        arr = list(seq)
        arr[pos] = _tr_mut[arr[pos]]
        return "".join(arr)

    def _rc_cs(cs: str) -> str:
        toks = _cs_tok_re.findall(cs)
        assert "".join(toks) == cs, f"Malformed cs: {cs}"
        out: list[str] = []
        for tok in reversed(toks):
            k = tok[0]
            if k == ":":
                out.append(tok)
            elif k == "*":
                out.append("*" + tok[1:].translate(_comp_lc))
            else:
                out.append(k + tok[1:][::-1].translate(_comp_lc))
        return "".join(out)

    def _cs_lens(cs: str) -> tuple[int, int]:
        q_l = 0
        r_l = 0
        for tok in _cs_tok_re.findall(cs):
            k = tok[0]
            if k == ":":
                v = int(tok[1:])
                q_l += v
                r_l += v
            elif k == "*":
                q_l += 1
                r_l += 1
            elif k == "+":
                q_l += len(tok) - 1
            else:
                r_l += len(tok) - 1
        return q_l, r_l

    def _make_match_cs(ref_s: str, qry_s: str) -> str:
        assert len(ref_s) == len(qry_s)
        parts: list[str] = []
        run = 0
        for rc, qc in zip(ref_s, qry_s):
            if rc == qc:
                run += 1
            else:
                if run:
                    parts.append(f":{run}")
                    run = 0
                parts.append(f"*{rc.lower()}{qc.lower()}")
        if run:
            parts.append(f":{run}")
        return "".join(parts)

    def _eval_cs_triplet(
        q_seq: str,
        cs_fwd: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
    ) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
        q_l, r_l = _cs_lens(cs_fwd)
        assert q_l == len(q_seq), f"cs q_len {q_l} != len(q_seq) {len(q_seq)}"
        cs_rev = _rc_cs(cs_fwd)
        q_rc = revcomp_dna(q_seq)
        hf = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=1, cs=cs_fwd)
        hr = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=-1, cs=cs_rev)
        hm = SimpleNamespace(q_st=0, q_en=q_l, r_st=0, r_en=r_l, strand=1, cs=cs_rev)
        bf: set[int] = set()
        br: set[int] = set()
        bm: set[int] = set()
        vf = hit_covers_subinterval_mappy(
            hf, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=bf
        )
        vr = hit_covers_subinterval_mappy(
            hr, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=br
        )
        vm = hit_covers_subinterval_mappy(
            hm,
            q_l - p_e,
            q_l - p_s,
            min_id,
            min_hit_len=mhl,
            query_seq=q_rc,
            ins_block_t_pieces_out=bm,
        )
        return (vf, vr, vm), (bool(bf), bool(br), bool(bm))

    _hit_cache: dict[tuple[str, str], tuple[list[object], list[object], list[object]]] = {}

    def _cov_triplet_blk(
        t_seq: str,
        q_seq: str,
        p_s: int,
        p_e: int,
        min_id: float = 0.90,
        mhl: int = 50,
    ) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
        q_len = len(q_seq)
        q_rc = revcomp_dna(q_seq)
        cached = _hit_cache.get((t_seq, q_seq))
        if cached is None:
            hf_list = list(mappy.Aligner(seq=t_seq, preset="asm20", best_n=50).map(q_seq, cs=True))
            aln_rc = mappy.Aligner(seq=revcomp_dna(t_seq), preset="asm20", best_n=50)
            hr_list = list(aln_rc.map(q_seq, cs=True))
            hm_list = list(aln_rc.map(q_rc, cs=True))
            cached = (hf_list, hr_list, hm_list)
            _hit_cache[(t_seq, q_seq)] = cached
        hf_list, hr_list, hm_list = cached
        bf: set[int] = set()
        vf = any(
            hit_covers_subinterval_mappy(
                h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=bf
            )
            for h in hf_list
        )
        br: set[int] = set()
        vr = any(
            hit_covers_subinterval_mappy(
                h, p_s, p_e, min_id, min_hit_len=mhl, query_seq=q_seq, ins_block_t_pieces_out=br
            )
            for h in hr_list
        )
        bm: set[int] = set()
        vm = any(
            hit_covers_subinterval_mappy(
                h,
                q_len - p_e,
                q_len - p_s,
                min_id,
                min_hit_len=mhl,
                query_seq=q_rc,
                ins_block_t_pieces_out=bm,
            )
            for h in hm_list
        )
        return (vf, vr, vm), (bool(bf), bool(br), bool(bm))

    # ------------------------------------------------------------------
    # (1) Fix 14-1: Step B Phase-2 Flank-SNP Closure + Scenario S3 +
    #     Realistic 96-99%-Flank Long-STR Sweep (All 5 Motifs, 0-2 Int SNPs)
    # ------------------------------------------------------------------
    fl500 = _mseq(500, 77001)[:-1] + "T"
    fr500 = "G" + _mseq(500, 77002)[1:]
    fr500_1 = _mut1(fr500, 160)
    fl500_1 = _mut1(fl500, 340)
    for tag_f1, t_f1, q_f1, ps_f1, pe_f1 in (
        ("ac70_r160", fl500 + ("AC" * 65) + fr500_1, fl500 + ("AC" * 70) + fr500, 620, 640),
        ("ac70_l160", fl500_1 + ("AC" * 65) + fr500, fl500 + ("AC" * 70) + fr500, 500, 520),
        ("a140_r160", fl500 + ("A" * 130) + fr500_1, fl500 + ("A" * 140) + fr500, 620, 640),
    ):
        v_f1, b_f1 = _cov_triplet_blk(t_f1, q_f1, ps_f1, pe_f1, 0.90, 50)
        assert v_f1 == (False, False, False) and b_f1 == (True, True, True), (
            f"Fix 14-1 {tag_f1} FAIL: v={v_f1} b={b_f1}"
        )

    fl150 = _mseq(150, 97001)[:-1] + "T"
    fr150 = "G" + _mseq(150, 97002)[1:]
    q_ac80 = fl150 + ("AC" * 80) + fr150
    v_r80, b_r80 = _cov_triplet_blk(fl150 + ("AC" * 75) + _mut1(fr150, 100), q_ac80, 290, 310, 0.90, 50)
    v_l80, b_l80 = _cov_triplet_blk(_mut1(fl150, 40) + ("AC" * 75) + fr150, q_ac80, 150, 170, 0.90, 50)
    assert v_r80 == (False, False, False) and b_r80 == (True, True, True)
    assert v_l80 == (False, False, False) and b_l80 == (True, True, True)

    # Direct "X"-branch flank-SNP closure (L3711 / L3987)
    q_x = ("C" * 100) + ("A" * 160) + ("G" * 5) + "T" + ("G" * 194)
    v_xsub, b_xsub = _eval_cs_triplet(q_x, ":100+aaaaaaaaaa:155*ct:84*cg:109", 240, 260, 0.90, 50)
    v_xopt, b_xopt = _eval_cs_triplet(q_x, ":250+aaaaaaaaaa:5*ct:84*cg:109", 240, 260, 0.90, 50)
    assert v_xsub == v_xopt == (False, False, False) and b_xsub == b_xopt == (True, True, True)

    # Scenario S3: strictly-better past-s_hi SNP MUST still suppress outer-I -> True == True
    fl100 = _mseq(100, 97101)[:-1] + "T"
    fr100 = "G" + _mseq(100, 97102)[1:]
    q_s3 = fl100 + ("A" * 189) + "C" + ("A" * 20) + fr100
    # Add a second ordinary flank SNP past the fixing SNP to exercise rem_x_all > 0
    r_b_fl = _tr_mut[q_s3[360]].lower()
    q_b_fl = q_s3[360].lower()
    cs_s3_sub = f":100+aaaaaaaaaa:179*ac:70*{r_b_fl}{q_b_fl}:49"
    cs_s3_opt = f":285+aaaacaaaaa:65*{r_b_fl}{q_b_fl}:49"
    v_s3_sub, b_s3_sub = _eval_cs_triplet(q_s3, cs_s3_sub, 190, 210, 0.90, 50)
    v_s3_opt, b_s3_opt = _eval_cs_triplet(q_s3, cs_s3_opt, 190, 210, 0.90, 50)
    assert v_s3_sub == v_s3_opt == (True, True, True) and b_s3_sub == b_s3_opt == (False, False, False), (
        f"Scenario S3 FAIL: sub={(v_s3_sub, b_s3_sub)} opt={(v_s3_opt, b_s3_opt)}"
    )

    # Realistic 96–99%-flank Long-STR sweep across all 5 motifs & 0-2 interior SNPs
    # (50 live mappy + 60 score-equivalent synthetic 140–300 bp pairs -> 0 asym)
    motifs_5 = ("A", "AC", "ACG", "ACGT", "GAACT")
    rng_f1 = random.Random(20261201)
    asym_live = 0
    for ti in range(50):
        u = motifs_5[ti % 5]
        ul = len(u)
        max_bp = 165 if ul == 1 else (185 if ul == 2 else 235)
        nr = rng_f1.randint(140 // ul, max_bp // ul)
        ic = rng_f1.randint(max(1, 6 // ul), max(1, 16 // ul))
        fl_r = "".join(rng_f1.choices("ACGT", k=240))
        fr_r = "".join(rng_f1.choices("ACGT", k=240))
        n_fl_snps = rng_f1.choice([2, 3, 6, 9])  # 96% to 99.2% flank homology
        fl_t, fr_t = fl_r, fr_r
        for pos in rng_f1.sample(range(10, 200), n_fl_snps):
            fl_t = _mut1(fl_t, pos)
        for pos in rng_f1.sample(range(40, 230), n_fl_snps):
            fr_t = _mut1(fr_t, pos)
        rt_r = list(u * nr)
        n_int_snps = rng_f1.choice([0, 1, 2])
        for sp in rng_f1.sample(range(20, len(rt_r) - 21), n_int_snps):
            rt_r[sp] = _tr_mut[rt_r[sp]]
        q_r = fl_r + (u * (nr + ic)) + fr_r
        t_r = fl_t + "".join(rt_r) + fr_t
        if ti % 2 == 0:
            pe_r = 240 + (nr + ic) * ul
            ps_r = pe_r - rng_f1.randint(15, 25)
        else:
            ps_r = 240
            pe_r = 240 + rng_f1.randint(15, 25)
        v_r, b_r = _cov_triplet_blk(t_r, q_r, ps_r, pe_r, 0.90, 50)
        if not (v_r[0] == v_r[1] == v_r[2] and b_r[0] == b_r[1] == b_r[2]):
            asym_live += 1
    assert asym_live == 0, f"Fix 14-1 realistic 96-99%-flank live STR sweep asym: {asym_live}/50"

    asym_syn = 0
    for ti in range(60):
        u = motifs_5[ti % 5]
        ul = len(u)
        nr = rng_f1.randint(140 // ul, 280 // ul)
        min_ic = max(1, (6 + ul - 1) // ul)
        ic = rng_f1.randint(min_ic, max(min_ic + 1, 24 // ul))
        fl_r = "".join(rng_f1.choices("ACGT", k=240))
        fr_r = "".join(rng_f1.choices("ACGT", k=240))
        fl_r = fl_r[:-1] + _tr_mut[u[0]]
        fr_r = _tr_mut[u[-1]] + fr_r[1:]
        n_fl_snps = rng_f1.choice([2, 5, 9])
        fl_t, fr_t = fl_r, fr_r
        for pos in rng_f1.sample(range(10, 210), n_fl_snps):
            fl_t = _mut1(fl_t, pos)
        for pos in rng_f1.sample(range(30, 230), n_fl_snps):
            fr_t = _mut1(fr_t, pos)
        q_str = u * (nr + ic)
        i_len = ic * ul
        t_arr = list(u * nr)
        n_int_snps = rng_f1.choice([0, 1, 2])
        for sp in rng_f1.sample(range(i_len + 10, len(t_arr) - i_len - 10), n_int_snps):
            t_arr[sp] = _tr_mut[t_arr[sp]]
        t_str = "".join(t_arr)
        q_r = fl_r + q_str + fr_r
        cs_l = _make_match_cs(fl_t, fl_r) + f"+{q_str[:i_len].lower()}" + _make_match_cs(t_str, q_str[i_len:]) + _make_match_cs(fr_t, fr_r)
        cs_r = _make_match_cs(fl_t, fl_r) + _make_match_cs(t_str, q_str[:-i_len]) + f"+{q_str[-i_len:].lower()}" + _make_match_cs(fr_t, fr_r)
        if ti % 2 == 0:
            pe_r = 240 + len(q_str)
            ps_r = pe_r - rng_f1.randint(15, 25)
        else:
            ps_r = 240
            pe_r = 240 + rng_f1.randint(15, 25)
        vl, bl = _eval_cs_triplet(q_r, cs_l, ps_r, pe_r, 0.90, 50)
        vr, br = _eval_cs_triplet(q_r, cs_r, ps_r, pe_r, 0.90, 50)
        if not (vl == vr == (False, False, False) and bl == br == (True, True, True)):
            asym_syn += 1
    assert asym_syn == 0, f"Fix 14-1 realistic 96-99%-flank syn STR sweep asym: {asym_syn}/60"

    # ------------------------------------------------------------------
    # (2) Fix 14-2: Attack 12-3 Exact Monotonicity Sweep [0.80..0.99]
    # ------------------------------------------------------------------
    q_m3 = (
        _mseq(10, 83001)
        + ("G" * 26)
        + _mseq(94, 83002)
        + ("C" * 28)
        + _mseq(192, 83003)
        + ("T" * 10)
        + ("A" * 22)
        + _mseq(190, 83004)
        + ("C" * 28)
        + _mseq(130, 83005)
    )
    cs_m3 = ":10+" + ("g" * 26) + ":94" + ("*ac" * 28) + ":192:2*at:1*at:5:20-aa:2:190" + ("*ac" * 28) + ":130"
    taus_a3 = (0.80, 0.85, 0.90, 0.92, 0.93, 0.94, 0.95, 0.98, 0.99)
    res_a3_f: list[bool] = []
    res_a3_m: list[bool] = []
    for tau in taus_a3:
        (vf3, vr3, vm3), _ = _eval_cs_triplet(q_m3, cs_m3, 350, 380, tau, 250)
        assert vf3 == vr3 == vm3, f"Attack 12-3 tau={tau} FWD/REV/MIR mismatch: {(vf3, vr3, vm3)}"
        res_a3_f.append(vf3)
        res_a3_m.append(vm3)
    expected_a3 = [True, True, True, True, True, True, True, True, False]
    assert res_a3_f == expected_a3 and res_a3_m == expected_a3, (
        f"Attack 12-3 FAIL: got {res_a3_f}, expected {expected_a3}"
    )

    # ------------------------------------------------------------------
    # (3) Fix 14-3a: Step A p_has_int_d Removal (Synthetic U(105..250),
    #     Attack 12-2, Live mappy A^320 dlen=8 @ 0.85)
    # ------------------------------------------------------------------
    for u_len_d in (105, 120, 200, 250):
        u_d = _mseq(u_len_d, 98100 + u_len_d)
        q_ud = fl500 + u_d + u_d + fr500
        cs_id = f":500+{u_d.lower()}:20-gg:{u_len_d - 20 + 500}"
        cs_di = f":520-gg:{u_len_d - 20}+{u_d.lower()}:500"
        v_id, b_id = _eval_cs_triplet(q_ud, cs_id, 500, 540, 0.90, 50)
        v_di, b_di = _eval_cs_triplet(q_ud, cs_di, 500, 540, 0.90, 50)
        assert v_id == v_di == (False, False, False) and b_id == b_di == (True, True, True), (
            f"Fix 14-3a U({u_len_d}) FAIL: id={(v_id, b_id)} di={(v_di, b_di)}"
        )

    # Attack 12-2: U(250)*2 + D(2) at center, P=[865, 885)
    u250_a2 = _mseq(250, 77250)
    q_a2 = fl500 + u250_a2 + u250_a2 + fr500
    v_a2_f, b_a2_f = _eval_cs_triplet(q_a2, f":500+{u250_a2.lower()}:125-gt:625", 865, 885, 0.90, 50)
    v_a2_r, b_a2_r = _eval_cs_triplet(q_a2, f":625-gt:125+{u250_a2.lower()}:500", 865, 885, 0.90, 50)
    assert v_a2_f == v_a2_r == (False, False, False) and b_a2_f == b_a2_r == (True, True, True)

    # Live mappy A^320 dlen=8 @ 0.85 on both [500, 549) and [771, 820)
    t_a320 = fl500 + ("A" * 20) + ("C" * 8) + ("A" * 120) + ("G" * 8) + ("A" * 20) + fr500
    q_a320 = fl500 + ("A" * 320) + fr500
    for ps_320, pe_320 in ((500, 549), (771, 820)):
        v_320, b_320 = _cov_triplet_blk(t_a320, q_a320, ps_320, pe_320, 0.85, 50)
        assert v_320 == (False, False, False) and b_320 == (True, True, True), (
            f"Fix 14-3a live A^320 dlen=8 p=({ps_320},{pe_320}) FAIL: v={v_320} b={b_320}"
        )

    # ------------------------------------------------------------------
    # (4) Fix 14-3b: Step A > 8000 Mirror-Symmetric Tie-Break
    # ------------------------------------------------------------------
    # Attack 12-4
    q_8k_arr = list(("C" * 50) + ("A" * 9200) + ("G" * 50))
    q_8k_arr[180] = "C"
    q_8k_arr[9080] = "C"
    q_8k = "".join(q_8k_arr)
    cs_8k = ":180*ac:320" + ("-t:20" * 13) + ":3739+" + ("a" * 60) + ":4520*ac:219"
    v_8k, _ = _eval_cs_triplet(q_8k, cs_8k, 4500, 4520, 0.90, 50)
    assert v_8k == (False, False, False), f"Attack 12-4 FAIL: {v_8k}"

    # Exact symmetric tie (d_l == d_r with cand_hi - cand_lo > 8000, both ok_l & ok_r True)
    q_tie_arr = list(("C" * 50) + ("A" * 9200) + ("G" * 50))
    q_tie_arr[500] = "C"
    q_tie_arr[8799] = "C"
    q_tie = "".join(q_tie_arr)
    cs_tie = ":500*ac:4119+" + ("a" * 60) + ":4119*ac:500"
    v_tie, _ = _eval_cs_triplet(q_tie, cs_tie, 4640, 4660, 0.90, 50)
    assert v_tie[0] == v_tie[1] == v_tie[2], f"Fix 14-3b exact tie FAIL: {v_tie}"

    # ------------------------------------------------------------------
    # (5) Bonus Catch E1: 2-SNP U(400)*2 Adjacent & Interior to P (4 Orientations)
    # ------------------------------------------------------------------
    u400_0 = _mseq(400, 99401)
    u400_1 = mutate_seq_snps(u400_0, [20])
    u400_2 = mutate_seq_snps(u400_0, [380])
    q_e1 = fl500 + u400_1 + u400_2 + fr500
    cs_e1_left = f":500+{u400_1.lower()}:380*{u400_0[380].lower()}{u400_2[380].lower()}:519"
    cs_e1_right = f":520*{u400_0[20].lower()}{u400_1[20].lower()}:379+{u400_2.lower()}:500"
    for ps_e1, pe_e1, exp_e1 in (
        (490, 500, True),
        (1300, 1310, True),
        (500, 510, False),
        (1290, 1300, False),
    ):
        vl_e1, bl_e1 = _eval_cs_triplet(q_e1, cs_e1_left, ps_e1, pe_e1, 0.90, 50)
        vr_e1, br_e1 = _eval_cs_triplet(q_e1, cs_e1_right, ps_e1, pe_e1, 0.90, 50)
        assert vl_e1 == vr_e1 == (exp_e1, exp_e1, exp_e1) and bl_e1 == br_e1, (
            f"Bonus Catch E1 p=({ps_e1},{pe_e1}) FAIL: l={(vl_e1, bl_e1)} r={(vr_e1, br_e1)}"
        )

    # ------------------------------------------------------------------
    # (6) Bonus Catch E2: 0 < min_hit_len < 50 Monotonicity Closure
    #     (raw, k1, k2 at mhl=20, 30 + randomized sweep over all mhl regimes)
    # ------------------------------------------------------------------
    rng_e2 = random.Random(42)
    q_e2 = "".join(rng_e2.choices("ACGT", k=200))

    def _x2(pos: int) -> str:
        return (
            f"*{_tr_mut[q_e2[pos]].lower()}{q_e2[pos].lower()}"
            f"*{_tr_mut[q_e2[pos + 1]].lower()}{q_e2[pos + 1].lower()}"
        )

    d45 = next(c for c in "ACGT" if c not in (q_e2[44], q_e2[45]))
    d155 = next(c for c in "ACGT" if c not in (q_e2[154], q_e2[155]))
    # mhl = 30 (P=[90, 110))
    mid_x30 = f"{_x2(80)}:8{_x2(90)}:18{_x2(110)}:8"
    for label_e2, cs_e2 in (
        ("raw", f":80{mid_x30}:80"),
        ("k1", f":45-{d45.lower()}:35{mid_x30}:80"),
        ("k2", f":45-{d45.lower()}:35{mid_x30}:35-{d155.lower()}:45"),
    ):
        taus_30 = (0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.99)
        vals_30: list[bool] = []
        for tau in taus_30:
            (vf2, vr2, vm2), _ = _eval_cs_triplet(q_e2, cs_e2, 90, 110, tau, 30)
            assert vf2 == vr2 == vm2, f"E2 mhl=30 {label_e2} tau={tau} mirror FAIL"
            vals_30.append(vf2)
        assert not any(
            vals_30[i] is False and vals_30[i + 1] is True for i in range(len(vals_30) - 1)
        ), f"E2 mhl=30 {label_e2} monotonicity FAIL: {list(zip(taus_30, vals_30))}"

    # mhl = 20 (P=[90, 100): 9/10=90% on P, <=16/20=80% on every 20-bp window, 46/50=92% on 50-bp window)
    def _x1(pos: int) -> str:
        return f"*{_tr_mut[q_e2[pos]].lower()}{q_e2[pos].lower()}"

    mid_x20 = (
        f"{_x1(87)}{_x1(88)}{_x1(89)}:5"
        f"{_x1(95)}:4"
        f"{_x1(100)}{_x1(101)}{_x1(102)}:17"
    )
    for label_e2, cs_e2 in (
        ("raw", f":87{mid_x20}:80"),
        ("k1", f":45-{d45.lower()}:42{mid_x20}:80"),
        ("k2", f":45-{d45.lower()}:42{mid_x20}:35-{d155.lower()}:45"),
    ):
        taus_20 = (0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.99)
        vals_20: list[bool] = []
        for tau in taus_20:
            (vf2, vr2, vm2), _ = _eval_cs_triplet(q_e2, cs_e2, 90, 100, tau, 20)
            assert vf2 == vr2 == vm2, f"E2 mhl=20 {label_e2} tau={tau} mirror FAIL"
            vals_20.append(vf2)
        assert vals_20 == [True, True, True, True, True, True, False, False, False], (
            f"E2 mhl=20 {label_e2} FAIL: {list(zip(taus_20, vals_20))}"
        )

    # Randomized 80x monotonicity & FWD==MIR sweep across ALL mhl regimes
    rng_sw = random.Random(20261402)
    mhl_grid = (0, 15, 20, 30, 45, 50, 100, 250)
    taus_sw = (0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.95, 0.97, 0.99)
    for ti in range(80):
        mhl_c = mhl_grid[ti % len(mhl_grid)]
        k_ind = rng_sw.randint(0, 4)
        q_len_sw = max(320, mhl_c + 160)
        q_sw = "".join(rng_sw.choices("ACGT", k=q_len_sw))
        # Build synthetic cs with random SNPs + k_ind small I/D ops
        p_len_sw = rng_sw.randint(5, 25)
        p_s_sw = (q_len_sw - p_len_sw) // 2
        p_e_sw = p_s_sw + p_len_sw
        # Choose indel cut points avoiding overlap
        cuts = sorted(rng_sw.sample(range(30, q_len_sw - 30, 8), k_ind)) if k_ind else []
        snp_set = set(rng_sw.sample(range(20, q_len_sw - 20), rng_sw.randint(6, 18)))
        cs_parts: list[str] = []
        cur = 0
        for cp in cuts:
            if cp < cur:
                continue
            for pos in range(cur, cp):
                if pos in snp_set:
                    cs_parts.append(f"*{_tr_mut[q_sw[pos]].lower()}{q_sw[pos].lower()}")
                else:
                    cs_parts.append(":1")
            cur = cp
            if rng_sw.random() < 0.5 and cur + 3 < q_len_sw - 20:
                il_sw = rng_sw.randint(1, 3)
                cs_parts.append("+" + q_sw[cur : cur + il_sw].lower())
                cur += il_sw
            else:
                dl_sw = rng_sw.randint(1, 3)
                cs_parts.append("-" + "".join(rng_sw.choices("acgt", k=dl_sw)))
        for pos in range(cur, q_len_sw):
            if pos in snp_set:
                cs_parts.append(f"*{_tr_mut[q_sw[pos]].lower()}{q_sw[pos].lower()}")
            else:
                cs_parts.append(":1")
        # Merge adjacent :1 runs
        merged: list[str] = []
        m_run = 0
        for tok in cs_parts:
            if tok == ":1":
                m_run += 1
            else:
                if m_run:
                    merged.append(f":{m_run}")
                    m_run = 0
                merged.append(tok)
        if m_run:
            merged.append(f":{m_run}")
        cs_sw = "".join(merged)
        vals_sw: list[bool] = []
        for tau in taus_sw:
            (vf_s, vr_s, vm_s), _ = _eval_cs_triplet(q_sw, cs_sw, p_s_sw, p_e_sw, tau, mhl_c)
            assert vf_s == vr_s == vm_s, f"E2 sweep ti={ti} mhl={mhl_c} tau={tau} mirror FAIL"
            vals_sw.append(vf_s)
        assert not any(
            vals_sw[i] is False and vals_sw[i + 1] is True for i in range(len(vals_sw) - 1)
        ), f"E2 sweep ti={ti} mhl={mhl_c} mono FAIL: {list(zip(taus_sw, vals_sw))}"

    # ------------------------------------------------------------------
    # (7) Fix #6 & Bonus Catch E3: < 50 ms Latency SLO Guards
    # ------------------------------------------------------------------
    # 7a. 2D-WC-FAIL K=2..12 mhl=250 (< 25 ms each, FWD == MIR)
    for k_wc, i_l_wc in ((2, 14), (3, 10), (4, 7), (5, 6), (6, 5), (8, 4), (10, 4), (12, 4)):
        tot_i_wc = k_wc * i_l_wc
        gap_wc = 20
        body_m = 460
        rem_m = body_m - (k_wc - 1) * gap_wc
        q_wc = ("C" * 40) + ("A" * (body_m + tot_i_wc)) + ("G" * 40)
        cs_wc_parts = [f":5{'*tc' * 35}:20"]
        for _ in range(k_wc - 1):
            cs_wc_parts.append(f"+{'a' * i_l_wc}:{gap_wc}")
        cs_wc_parts.append(f"+{'a' * i_l_wc}:{rem_m - 20}{'*tg' * 35}:5")
        cs_wc = "".join(cs_wc_parts)
        q_l_wc, r_l_wc = _cs_lens(cs_wc)
        assert q_l_wc == len(q_wc)
        h_wc = SimpleNamespace(q_st=0, q_en=q_l_wc, r_st=0, r_en=r_l_wc, strand=1, cs=cs_wc)
        h_wc_m = SimpleNamespace(q_st=0, q_en=q_l_wc, r_st=0, r_en=r_l_wc, strand=1, cs=_rc_cs(cs_wc))
        p_mid = q_l_wc // 2
        t0 = time.perf_counter()
        v_wc = hit_covers_subinterval_mappy(h_wc, p_mid - 10, p_mid + 10, 0.95, min_hit_len=250, query_seq=q_wc)
        dt_wc_ms = (time.perf_counter() - t0) * 1000.0
        v_wc_m = hit_covers_subinterval_mappy(
            h_wc_m, q_l_wc - (p_mid + 10), q_l_wc - (p_mid - 10), 0.95, min_hit_len=250, query_seq=revcomp_dna(q_wc)
        )
        assert v_wc is False and v_wc == v_wc_m and dt_wc_ms < 25.0, (
            f"2D-WC-FAIL K={k_wc} mhl=250 FAIL: v={(v_wc, v_wc_m)} dt={dt_wc_ms:.2f}ms"
        )

    # 7b. K=4 (3I+1D sumI=46) mhl=250 (< 25 ms, R4_235 worst-case shape)
    q_r4 = ("C" * 40) + ("A" * 520) + ("G" * 40)
    cs_r4 = f":5{'*tc' * 35}:60+{'a' * 16}:40+{'a' * 15}:40+{'a' * 15}:40-tt:294{'*tg' * 35}:5"
    q_l_r4, r_l_r4 = _cs_lens(cs_r4)
    assert q_l_r4 == len(q_r4)
    h_r4 = SimpleNamespace(q_st=0, q_en=q_l_r4, r_st=0, r_en=r_l_r4, strand=1, cs=cs_r4)
    t0 = time.perf_counter()
    v_r4 = hit_covers_subinterval_mappy(h_r4, 290, 310, 0.95, min_hit_len=250, query_seq=q_r4)
    dt_r4_ms = (time.perf_counter() - t0) * 1000.0
    assert v_r4 is False and dt_r4_ms < 25.0, f"K=4 (3I+1D) FAIL: v={v_r4} dt={dt_r4_ms:.2f}ms"

    # 7c. Bonus Catch E3: cat1_mhl250_k4_PASS (< 50 ms) & cat1_mhl100_k4_FAIL (< 25 ms)
    # s_cover (I(24) covering P=[240, 250)) + 3 D(2) on A^450 -> 4 Component C perms x 15 leaves
    q_c1 = ("C" * 50) + ("A" * 450) + ("G" * 50)
    cs_c1_250 = f":50:120+{'a' * 24}:40-tt:40-tt:40-tt:186:50"
    q_l_c1, r_l_c1 = _cs_lens(cs_c1_250)
    assert q_l_c1 == len(q_c1)
    h_c1_250 = SimpleNamespace(q_st=0, q_en=q_l_c1, r_st=0, r_en=r_l_c1, strand=1, cs=cs_c1_250)
    t0 = time.perf_counter()
    v_c1_pass = hit_covers_subinterval_mappy(h_c1_250, 240, 250, 0.80, min_hit_len=250, query_seq=q_c1)
    dt_c1_pass_ms = (time.perf_counter() - t0) * 1000.0
    assert v_c1_pass is True and dt_c1_pass_ms < 50.0, (
        f"cat1_mhl250_k4_PASS FAIL: v={v_c1_pass} dt={dt_c1_pass_ms:.2f}ms"
    )

    cs_c1_100 = f":50:120+{'a' * 10}:40-tt:40-tt:40-tt:200:50"
    q_l_c100, r_l_c100 = _cs_lens(cs_c1_100)
    assert q_l_c100 == len(q_c1)
    h_c1_100 = SimpleNamespace(q_st=0, q_en=q_l_c100, r_st=0, r_en=r_l_c100, strand=1, cs=cs_c1_100)
    t0 = time.perf_counter()
    v_c1_fail = hit_covers_subinterval_mappy(h_c1_100, 240, 245, 0.99, min_hit_len=100, query_seq=q_c1)
    dt_c1_fail_ms = (time.perf_counter() - t0) * 1000.0
    assert v_c1_fail is False and dt_c1_fail_ms < 25.0, (
        f"cat1_mhl100_k4_FAIL FAIL: v={v_c1_fail} dt={dt_c1_fail_ms:.2f}ms"
    )

    # 7d. poly-A K=6..60 mhl=50 (< 10 ms/call) & 100k+I(7) (< 5 ms/call)
    for k_pa in (6, 10, 20, 40, 60):
        cs_pa = ":50" + ("+a:5" * k_pa) + ":100"
        q_l_pa, r_l_pa = _cs_lens(cs_pa)
        q_pa = ("C" * 50) + ("A" * (q_l_pa - 100)) + ("G" * 50)
        h_pa = SimpleNamespace(q_st=0, q_en=q_l_pa, r_st=0, r_en=r_l_pa, strand=1, cs=cs_pa)
        t0 = time.perf_counter()
        hit_covers_subinterval_mappy(h_pa, q_l_pa // 2 - 5, q_l_pa // 2 + 5, 0.90, min_hit_len=50, query_seq=q_pa)
        dt_pa_ms = (time.perf_counter() - t0) * 1000.0
        assert dt_pa_ms < 10.0, f"poly-A K={k_pa} too slow: {dt_pa_ms:.2f}ms"

    q_100k_14 = "C" + ("A" * 100_007) + "G"
    h_100k_14 = SimpleNamespace(
        q_st=0, q_en=len(q_100k_14), r_st=0, r_en=len(q_100k_14) - 7, strand=1, cs=":50001+aaaaaaa:50001"
    )
    t0 = time.perf_counter()
    for _ in range(5):
        assert (
            hit_covers_subinterval_mappy(h_100k_14, 50001, 50008, 0.90, min_hit_len=50, query_seq=q_100k_14)
            is False
        )
    avg_100k_14_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
    assert avg_100k_14_ms < 5.0, f"100k+I(7) too slow: {avg_100k_14_ms:.3f}ms"


def test_iter15_multi_snp_str_outer_island_and_mhl80_99_slo_closures() -> None:
    """Iter 15 Permanent Regressions for BUG-R14-1 & BUG-R14-2 (+ Scenario S3 pos=295):
    1. Exact Trial 129 (`GAACT`, 2 interior SNPs at 57 & 159, `tau=0.80..0.95` -> FWD==REV==MIR).
    2. Exact 1B Trial 4 (`ACG`, `l_i=12`, 3 optimal islands `s=180, 306, 419` across `0.80..0.95`),
       plus large-separation (`150..8500 bp`), `K=2` multi-`I` outer-island, and suboptimal native `I`.
    3. Probe 5A sweep: `100x` live `mappy` 2-interior-SNP Long-STRs across all 5 motifs
       (`A`, `AC`, `ACG`, `ACGT`, `GAACT`) -> `0` same-topology asymmetries.
    4. Exact BUG-R14-2 latency sweep (`2D-WC-FAIL`, `K=4, 6, 8, 10, 12` across legal
       `mhl=50, 60, 70, 75, 80, 85, 90, 95, 98, 99, 100, 250`) -> all strictly `< 50 ms`.
    5. True past-`scan_hi` Scenario S3 (`pos=295 > scan_hi=290`, `rem_x_all > 0`) -> `sub == opt`.
    """
    _tr_mut = {"A": "C", "C": "G", "G": "T", "T": "A"}
    _tok_re = re.compile(r":[0-9]+|\*[a-z]{2}|\+[a-z]+|-[a-z]+")

    def _make_cs_single_i(r_seq: str, q_seq: str, s: int, l_i: int) -> str:
        ops: list[str] = []

        def _emit(r_seg: str, q_seg: str) -> None:
            m_run = 0
            for rb, qb in zip(r_seg, q_seg):
                if rb == qb:
                    m_run += 1
                else:
                    if m_run:
                        ops.append(f":{m_run}")
                        m_run = 0
                    ops.append(f"*{rb.lower()}{qb.lower()}")
            if m_run:
                ops.append(f":{m_run}")

        _emit(r_seq[:s], q_seq[:s])
        ops.append(f"+{q_seq[s : s + l_i].lower()}")
        _emit(r_seq[s:], q_seq[s + l_i :])
        return "".join(ops)

    def _mir_cs(cs: str) -> str:
        out: list[str] = []
        for tok in reversed(_tok_re.findall(cs)):
            if tok[0] in (":", "*"):
                out.append(tok)
            else:
                out.append(tok[0] + tok[1:][::-1])
        return "".join(out)

    def _rc_cs_i15(cs: str) -> str:
        comp = str.maketrans("acgtn", "tgcan")
        out: list[str] = []
        for tok in reversed(_tok_re.findall(cs)):
            if tok[0] == ":":
                out.append(tok)
            elif tok[0] == "*":
                out.append("*" + tok[1:].translate(comp))
            else:
                out.append(tok[0] + tok[1:][::-1].translate(comp))
        return "".join(out)

    def _topo(cs: str) -> tuple[tuple[str, int], ...]:
        return tuple(
            (tok[0], len(tok) - 1)
            for tok in _tok_re.findall(cs)
            if tok[0] in ("+", "-")
        )

    # 1. Exact Trial 129 (GAACT, 2 interior SNPs at 57 & 159) -> FWD == REV == MIR
    rng_t129 = random.Random(99)
    fl_129 = "".join(rng_t129.choices("ACGT", k=500))[:-1] + "G"
    fr_129 = "C" + "".join(rng_t129.choices("ACGT", k=500))[1:]
    str_r_129 = "GAACT" * 33
    str_q_129 = list("GAACT" * 34)
    str_q_129[57] = "G"
    str_q_129[159] = "C"
    r_129 = fl_129 + str_r_129 + fr_129
    q_129 = fl_129 + "".join(str_q_129) + fr_129
    hf_129 = list(mappy.Aligner(seq=r_129, preset="asm20", best_n=50).map(q_129, cs=True))[0]
    hr_129 = list(
        mappy.Aligner(seq=revcomp_dna(r_129), preset="asm20", best_n=50).map(
            revcomp_dna(q_129), cs=True
        )
    )[0]
    assert _topo(hf_129.cs) == _topo(hr_129.cs)[::-1] and hf_129.mlen == hr_129.mlen
    ps_129, pe_129 = 500 + 145, 500 + 165
    hm_129 = SimpleNamespace(
        q_st=len(q_129) - hf_129.q_en,
        q_en=len(q_129) - hf_129.q_st,
        r_st=len(r_129) - hf_129.r_en,
        r_en=len(r_129) - hf_129.r_st,
        strand=1,
        cs=_mir_cs(hf_129.cs),
    )
    for tau, exp_v in ((0.80, True), (0.85, True), (0.90, True), (0.95, False)):
        vf = hit_covers_subinterval_mappy(hf_129, ps_129, pe_129, tau, query_seq=q_129, min_hit_len=50)
        vr = hit_covers_subinterval_mappy(
            hr_129, len(q_129) - pe_129, len(q_129) - ps_129, tau, query_seq=revcomp_dna(q_129), min_hit_len=50
        )
        vm = hit_covers_subinterval_mappy(
            hm_129, len(q_129) - pe_129, len(q_129) - ps_129, tau, query_seq=q_129[::-1], min_hit_len=50
        )
        assert vf == vr == vm == exp_v, (
            f"Trial 129 FAIL tau={tau}: FWD={vf} REV={vr} MIR={vm} exp={exp_v}"
        )

    # Large-separation 2-int-SNP STR (150, 400, 1500, and 8500 > 8000 cap) -> FWD == REV == MIR
    for sep in (150, 400, 1500, 8500):
        fl_s = "ACGT" * 50
        fr_s = "TGCA" * 50
        n_c = (sep + 200) // 5
        sr_s = "GAACT" * n_c
        sq_s = list("GAACT" * (n_c + 1))
        sq_s[50] = "C"
        sq_s[50 + sep] = "C"
        r_sep = fl_s + sr_s + fr_s
        q_sep = fl_s + "".join(sq_s) + fr_s
        cs_l = _make_cs_single_i(r_sep, q_sep, 200 + 46, 5)
        cs_r = _make_cs_single_i(r_sep, q_sep, 200 + 50 + sep, 5)
        h_l = SimpleNamespace(q_st=0, q_en=len(q_sep), r_st=0, r_en=len(r_sep), strand=1, cs=cs_l)
        h_r = SimpleNamespace(q_st=0, q_en=len(q_sep), r_st=0, r_en=len(r_sep), strand=1, cs=cs_r)
        h_lm = SimpleNamespace(q_st=0, q_en=len(q_sep), r_st=0, r_en=len(r_sep), strand=1, cs=_mir_cs(cs_l))
        ps_s, pe_s = 200 + 50 + sep - 10, 200 + 50 + sep + 10
        for tau in (0.80, 0.90, 0.95):
            vl = hit_covers_subinterval_mappy(h_l, ps_s, pe_s, tau, query_seq=q_sep, min_hit_len=50)
            vr = hit_covers_subinterval_mappy(h_r, ps_s, pe_s, tau, query_seq=q_sep, min_hit_len=50)
            vlm = hit_covers_subinterval_mappy(
                h_lm, len(q_sep) - pe_s, len(q_sep) - ps_s, tau, query_seq=q_sep[::-1], min_hit_len=50
            )
            assert vl == vr == vlm, f"sep={sep} tau={tau} FAIL: {(vl, vr, vlm)}"

    # 2. Exact 1B Trial 4 (ACG, l_i=12, 3 optimal islands s=180, 306, 419 across 0.80..0.95)
    rng_1b = random.Random(12345)
    fl_1b = "".join(rng_1b.choices("ACGT", k=120))
    fr_1b = "".join(rng_1b.choices("ACGT", k=120))
    sr_1b = "ACG" * 100
    sq_1b = list("ACG" * 104)
    for p_snp in (63, 187, 299):
        sq_1b[p_snp] = "T"
    r_1b = fl_1b + sr_1b + fr_1b
    q_1b = fl_1b + "".join(sq_1b) + fr_1b
    scores_1b = [
        sum(1 for a, b in zip(r_1b[:s], q_1b[:s]) if a == b)
        + sum(1 for a, b in zip(r_1b[s:], q_1b[s + 12 :]) if a == b)
        for s in (180, 306, 419)
    ]
    assert len(set(scores_1b)) == 1
    for tau in (0.80, 0.85, 0.90, 0.95):
        vs_1b = []
        for s_opt in (180, 306, 419):
            cs_s = _make_cs_single_i(r_1b, q_1b, s_opt, 12)
            h_s = SimpleNamespace(q_st=0, q_en=len(q_1b), r_st=0, r_en=len(r_1b), strand=1, cs=cs_s)
            h_sm = SimpleNamespace(q_st=0, q_en=len(q_1b), r_st=0, r_en=len(r_1b), strand=1, cs=_mir_cs(cs_s))
            v_s = hit_covers_subinterval_mappy(h_s, 185, 205, tau, query_seq=q_1b, min_hit_len=50)
            v_sm = hit_covers_subinterval_mappy(
                h_sm, len(q_1b) - 205, len(q_1b) - 185, tau, query_seq=q_1b[::-1], min_hit_len=50
            )
            assert v_s == v_sm
            vs_1b.append(v_s)
        assert vs_1b == [True, True, True], f"1B Trial 4 FAIL at tau={tau}: {vs_1b}"

    # 2b. K=2 multi-I outer-island parity (I_2@603 vs I_2@698 alongside I_1@712)
    fl_k2 = "ACGT" * 125
    fr_k2 = "TGCA" * 125
    q_k2_arr = list(fl_k2 + "GAACT" * 41)
    q_k2_arr[607] = "C"
    q_k2_arr[699] = "C"
    r_k2 = fl_k2 + "GAACT" * 40 + "ACACACACAC" + fr_k2
    q_k2 = "".join(q_k2_arr) + "ACACACA" + "GG" + "CAC" + fr_k2

    def _cs_two_i(r_s: str, q_s: str, s2: int, l2: int, s1_q: int, l1: int) -> str:
        ops: list[str] = []

        def _emit(rs: str, qs: str) -> None:
            m_run = 0
            for rb, qb in zip(rs, qs):
                if rb == qb:
                    m_run += 1
                else:
                    if m_run:
                        ops.append(f":{m_run}")
                        m_run = 0
                    ops.append(f"*{rb.lower()}{qb.lower()}")
            if m_run:
                ops.append(f":{m_run}")

        r_mid = s1_q - l2
        _emit(r_s[:s2], q_s[:s2])
        ops.append(f"+{q_s[s2 : s2 + l2].lower()}")
        _emit(r_s[s2:r_mid], q_s[s2 + l2 : s1_q])
        ops.append(f"+{q_s[s1_q : s1_q + l1].lower()}")
        _emit(r_s[r_mid:], q_s[s1_q + l1 :])
        return "".join(ops)

    h_k2_o = SimpleNamespace(
        q_st=0, q_en=len(q_k2), r_st=0, r_en=len(r_k2), strand=1, cs=_cs_two_i(r_k2, q_k2, 603, 5, 712, 2)
    )
    h_k2_i = SimpleNamespace(
        q_st=0, q_en=len(q_k2), r_st=0, r_en=len(r_k2), strand=1, cs=_cs_two_i(r_k2, q_k2, 698, 5, 712, 2)
    )
    for tau in (0.80, 0.85, 0.90, 0.95):
        assert hit_covers_subinterval_mappy(
            h_k2_o, 695, 715, tau, query_seq=q_k2, min_hit_len=50
        ) == hit_covers_subinterval_mappy(
            h_k2_i, 695, 715, tau, query_seq=q_k2, min_hit_len=50
        )

    # 3. Probe 5A: 100x live mappy 2-interior-SNP Long-STRs across all 5 motifs -> 0 asym
    rng_5a = random.Random(5551401)
    motifs_5a = ("A", "AC", "ACG", "ACGT", "GAACT")
    checks_5a = 0
    asym_5a = 0
    for ti in range(100):
        m = motifs_5a[ti % 5]
        ml = len(m)
        copies_r = rng_5a.randint(150 // ml, 220 // ml)
        ins_copies = rng_5a.randint(1, max(1, min(6, 12 // ml)))
        fl_5 = "".join(rng_5a.choices("ACGT", k=500))
        fr_5 = "".join(rng_5a.choices("ACGT", k=500))
        sr_5 = m * copies_r
        sq_5 = list(m * (copies_r + ins_copies))
        l_sq = len(sq_5)
        p1 = rng_5a.randint(25, l_sq // 2 - 20)
        p2 = rng_5a.randint(p1 + 70, l_sq - 25)
        for p_pos in (p1, p2):
            sq_5[p_pos] = rng_5a.choice([b for b in "ACGT" if b != sq_5[p_pos]])
        r_5 = fl_5 + sr_5 + fr_5
        q_5 = fl_5 + "".join(sq_5) + fr_5
        hfs = list(mappy.Aligner(seq=r_5, preset="asm20", best_n=50).map(q_5, cs=True))
        hrs = list(
            mappy.Aligner(seq=revcomp_dna(r_5), preset="asm20", best_n=50).map(
                revcomp_dna(q_5), cs=True
            )
        )
        if not hfs or not hrs:
            continue
        hf, hr = hfs[0], hrs[0]
        if _topo(hf.cs) != _topo(hr.cs)[::-1] or hf.mlen != hr.mlen:
            continue
        q_5_rc = revcomp_dna(q_5)
        for p_pos in (p1, p2):
            ps_5 = 500 + p_pos - 10
            pe_5 = 500 + p_pos + 10
            for tau in (0.85, 0.90):
                checks_5a += 1
                vf = hit_covers_subinterval_mappy(hf, ps_5, pe_5, tau, query_seq=q_5, min_hit_len=50)
                vr = hit_covers_subinterval_mappy(
                    hr, len(q_5) - pe_5, len(q_5) - ps_5, tau, query_seq=q_5_rc, min_hit_len=50
                )
                if vf != vr:
                    asym_5a += 1
    assert checks_5a >= 200 and asym_5a == 0, f"Probe 5A FAIL: asym={asym_5a}/{checks_5a}"

    # 4. Exact BUG-R14-2 sweep: 2D-WC-FAIL K=4..12 across mhl=50..250 -> ALL < 50 ms
    for mhl_wc in (50, 60, 70, 75, 80, 85, 90, 95, 98, 99, 100, 250):
        for k_wc in (4, 6, 8, 10, 12):
            n_i = k_wc - 2
            l_i = 4
            half_a = mhl_wc + 40
            step = half_a // (n_i + 1)
            cs_wc = (
                ":300"
                + "".join(f":{step}+{'a' * l_i}" for _ in range(n_i))
                + f":{half_a - step * n_i - 1}-tttt:2-tttt:{half_a}:300"
            )
            q_p = 300 + half_a + n_i * l_i
            q_wc = "C" * 300 + "A" * (2 * half_a + 1 + n_i * l_i) + "G" * 300
            h_wc = SimpleNamespace(
                q_st=0, q_en=len(q_wc), r_st=0, r_en=600 + 2 * half_a + 9, strand=1, cs=cs_wc
            )
            t0 = time.perf_counter()
            res_wc = hit_covers_subinterval_mappy(h_wc, q_p, q_p + 1, 0.99, query_seq=q_wc, min_hit_len=mhl_wc)
            dt_wc_ms = (time.perf_counter() - t0) * 1000.0
            assert res_wc is False and dt_wc_ms < 50.0, (
                f"BUG-R14-2 SLO breach at mhl={mhl_wc} K={k_wc}: res={res_wc} dt={dt_wc_ms:.2f}ms"
            )
            if mhl_wc in (69, 70, 99) and k_wc == 12:
                h_wc_m = SimpleNamespace(
                    q_st=0, q_en=len(q_wc), r_st=0, r_en=600 + 2 * half_a + 9, strand=1, cs=_rc_cs_i15(cs_wc)
                )
                res_wc_m = hit_covers_subinterval_mappy(
                    h_wc_m,
                    len(q_wc) - (q_p + 1),
                    len(q_wc) - q_p,
                    0.99,
                    query_seq=revcomp_dna(q_wc),
                    min_hit_len=mhl_wc,
                )
                assert res_wc_m is False

    # 5. True past-scan_hi Scenario S3 (pos=295 > scan_hi=290, rem_x_all > 0)
    fl_s3 = "".join(random.Random(97101).choices("ACGT", k=100))[:-1] + "T"
    fr_s3 = "G" + "".join(random.Random(97102).choices("ACGT", k=100))[1:]
    q_s3_295 = fl_s3 + ("A" * 195) + "C" + ("A" * 20) + fr_s3
    r_b_fl = _tr_mut[q_s3_295[365]].lower()
    q_b_fl = q_s3_295[365].lower()
    cs_295_sub = f":100+aaaaaaaaaa:185*ac:69*{r_b_fl}{q_b_fl}:50"
    cs_295_opt = f":290+aaaaacaaaa:65*{r_b_fl}{q_b_fl}:50"
    h_295_sub = SimpleNamespace(
        q_st=0, q_en=len(q_s3_295), r_st=0, r_en=len(q_s3_295) - 10, strand=1, cs=cs_295_sub
    )
    h_295_opt = SimpleNamespace(
        q_st=0, q_en=len(q_s3_295), r_st=0, r_en=len(q_s3_295) - 10, strand=1, cs=cs_295_opt
    )
    h_295_sub_m = SimpleNamespace(
        q_st=0, q_en=len(q_s3_295), r_st=0, r_en=len(q_s3_295) - 10, strand=1, cs=_mir_cs(cs_295_sub)
    )
    for tau in (0.90, 0.95):
        v_sub = hit_covers_subinterval_mappy(h_295_sub, 190, 210, tau, query_seq=q_s3_295, min_hit_len=50)
        v_opt = hit_covers_subinterval_mappy(h_295_opt, 190, 210, tau, query_seq=q_s3_295, min_hit_len=50)
        v_mir = hit_covers_subinterval_mappy(
            h_295_sub_m, len(q_s3_295) - 210, len(q_s3_295) - 190, tau, query_seq=q_s3_295[::-1], min_hit_len=50
        )
        assert v_sub == v_opt == v_mir, f"S3 pos=295 FAIL tau={tau}: {(v_sub, v_opt, v_mir)}"
