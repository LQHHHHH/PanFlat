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

from pathlib import Path
import random
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
    _evaluate_mappy_hits_on_blocks,
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


def _mutate_poisson_with_micro_indels(
    seq: str,
    snp_rate: float,
    micro_ins_specs: Sequence[tuple[int, int]],
    micro_del_specs: Sequence[tuple[int, int]],
    seed: int,
) -> str:
    """Apply realistic random Poisson SNPs plus deterministic micro-insertions and deletions."""
    rng = random.Random(seed)
    swap = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    ins_map = {pos: "".join(rng.choices("ACGT", k=ln)) for pos, ln in micro_ins_specs}
    del_ranges = sorted(micro_del_specs)
    out: list[str] = []
    i = 0
    n = len(seq)
    del_idx = 0
    while i < n:
        if i in ins_map:
            out.append(ins_map[i])
        if del_idx < len(del_ranges) and i == del_ranges[del_idx][0]:
            i += del_ranges[del_idx][1]
            del_idx += 1
            continue
        ch = seq[i]
        if ch in swap and rng.random() < snp_rate:
            out.append(rng.choice(swap[ch]))
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def test_smooth_homolog_blocks_poisson_micro_indels_sv_del_and_large_window() -> None:
    """Chr06 regression: `smooth_homolog_blocks=True` eliminates 50 bp Swiss-cheese holes from
    Poisson SNP clusters, 6-25 bp micro-indels, and >=50 bp target deletions while preserving
    `protected_ins`, >=200 bp sub-threshold tracts, and >32 kb window-boundary insertions.
    """
    rng = random.Random(8101)
    ref_seq = "".join(rng.choices("ACGT", k=90_000))
    aln = mappy.Aligner(seq=ref_seq, preset="asm20", best_n=50)
    saln = mappy.Aligner(seq=ref_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50)
    buf_m = mappy.ThreadBuffer()
    buf_s = mappy.ThreadBuffer()

    # (a) 15 kb homologous TE (~94.5% identity) with random Poisson SNPs (4.8%), six 8-18 bp
    #     micro-insertions, four 8-15 bp micro-deletions, and one 160 bp structural deletion:
    base_15k = ref_seq[10_000:25_000]
    mut_15k = _mutate_poisson_with_micro_indels(
        base_15k,
        snp_rate=0.048,
        micro_ins_specs=[(1210, 10), (3420, 14), (5815, 8), (8120, 18), (10910, 12), (13205, 9)],
        micro_del_specs=[(2100, 10), (4900, 12), (7500, 160), (12100, 9)],
        seed=8102,
    )
    # Without smoothing (strict v1 50 bp block grid), Poisson dips + micro-indels + 160D punch many holes
    _, u_raw, _ = subtract_reference_intervals_mappy(
        mut_15k, aligner=aln, buf=buf_m, min_identity=0.90, smooth_homolog_blocks=False
    )
    assert len(u_raw) >= 10

    # With smooth_homolog_blocks=True, 100% of the homologous query is masked on '+' and '-' strands
    for q_hom in (mut_15k, revcomp_dna(mut_15k)):
        m_sm, u_sm, bp_sm = subtract_reference_intervals_mappy(
            q_hom,
            aligner=aln,
            buf=buf_m,
            min_identity=0.90,
            short_aligner=saln,
            short_buf=buf_s,
            smooth_homolog_blocks=True,
        )
        assert u_sm == [] and m_sm == [(0, len(q_hom))] and bp_sm == len(q_hom)

    # (b) Guards: >=200 bp sub-threshold tract NOT masked at 0.95; 150 bp 60%-divergent replacement NOT masked
    flank_l = ref_seq[30_000:30_400]
    mid_250_92 = _mutate_per_50bp_block(ref_seq[30_400:30_650], (4, 14, 28, 42))  # 46/50 = 92%
    flank_r = ref_seq[30_650:31_050]
    _, u_guard95, _ = subtract_reference_intervals_mappy(
        flank_l + mid_250_92 + flank_r,
        aligner=aln,
        buf=buf_m,
        min_identity=0.95,
        smooth_homolog_blocks=True,
    )
    assert u_guard95 == [(400, 650)]

    div_150 = _mutate_poisson_with_micro_indels(
        ref_seq[30_400:30_550], snp_rate=0.45, micro_ins_specs=[], micro_del_specs=[], seed=8103
    )
    _, u_div, _ = subtract_reference_intervals_mappy(
        flank_l + div_150 + flank_r[100:],
        aligner=aln,
        buf=buf_m,
        min_identity=0.90,
        smooth_homolog_blocks=True,
    )
    assert any(350 <= us <= 410 and 540 <= ue <= 600 for us, ue in u_div)

    # (c) >32 kb large-query windowing (`70 kb` + `2,000 bp` novel insertion straddling 24,000 boundary at [23996, 25996))
    big_base = _mutate_poisson_with_micro_indels(
        ref_seq[15_000:85_000],
        snp_rate=0.04,
        micro_ins_specs=[(5120, 9), (23500, 11), (45200, 12)],
        micro_del_specs=[(12300, 8), (52000, 120)],
        seed=8104,
    )
    novel_2k = make_complex_seq(2000, seed=8105)
    ins_pos = 23_996  # 4 bp before first 24,000 bp window boundary!
    big_q = big_base[:ins_pos] + novel_2k + big_base[ins_pos:]
    prot_out: list[tuple[int, int]] = []
    _, u_big, _ = subtract_reference_intervals_mappy(
        big_q,
        aligner=aln,
        buf=buf_m,
        min_identity=0.90,
        short_aligner=saln,
        short_buf=buf_s,
        smooth_homolog_blocks=True,
        protected_ins_out=prot_out,
    )
    assert len(u_big) == 1
    assert u_big[0][1] - u_big[0][0] == 2000
    assert abs(u_big[0][0] - ins_pos) <= 5
    assert any(ps <= ins_pos + 50 and pe >= ins_pos + 1950 for ps, pe in prot_out)


def test_stage55_contiguous_chain_whole_sv_drop_and_zero_lt50_islands(tmp_path: Path) -> None:
    """Chr06 regression:
    1. Multi-piece N-free synthetic SV `[1000 bp, 30 bp, 1000 bp]` duplicating an accepted
       `2,030 bp` locus is subtracted as ONE unified sequence and 100% dropped as a whole SV
       (zero `< 50 bp` zombie pieces).
    2. Isolated `< min_sv_len` N-bounded chains and `< 150 bp` non-protected split shards are
       discarded into `dedup_short_residual_bp` across both `ref` and `pan` modes.
    """
    gb = SyntheticGFAFixtureBuilder()
    for i in range(1, 9):
        gb.add_node(str(i), make_complex_seq(300, seed=8200 + i))

    seg_l = make_complex_seq(1000, seed=8220)
    seg_m30 = make_complex_seq(30, seed=8221)
    seg_m30_alt = make_complex_seq(80, seed=8222)
    seg_r = make_complex_seq(1000, seed=8223)
    full_2030 = seg_l + seg_m30 + seg_r
    unrelated_28 = make_complex_seq(28, seed=8224)
    tip_70 = make_complex_seq(70, seed=8225)
    clean_65 = make_complex_seq(65, seed=8226)

    for nid, sq in [
        ("10", full_2030),
        ("11", seg_l),
        ("12", seg_m30),
        ("13", seg_r),
        ("14", seg_m30_alt),
        ("15", unrelated_28),
        ("16", tip_70 + _mutate_per_50bp_block(seg_l + seg_r, (5, 35))),
        ("17", clean_65),
    ]:
        gb.add_node(nid, sq)

    ref_steps = [(str(i), "+") for i in range(1, 9)]
    gb.add_p_path("Ref#1#Chr01", ref_steps)
    # High-support locus (1, 2) carries full_2030 + alt_80 so both full_2030 and alt_80 are accepted
    for k in range(1, 5):
        gb.add_p_path(
            f"Sup{k}#1#Chr01",
            [("1", "+"), ("10", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
        )
    gb.add_p_path(
        "SupAlt#1#Chr01",
        [("1", "+"), ("11", "+"), ("14", "+"), ("13", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
    )
    # Locus (3, 4): multi-member synthetic rep `[11(1000) + 12(30) + 13(1000)]` + child `14(80)` -> 100% redundant!
    gb.add_p_path(
        "DupA#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("12", "+"), ("13", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
    )
    gb.add_p_path(
        "DupB#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("11", "+"), ("14", "+"), ("13", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
    )
    # Locus (4, 5): `[11(1000) + 15(unrelated 28bp) + 13(1000)]` -> flanks subtracted, 28bp < 50bp dropped -> whole SV dropped!
    gb.add_p_path(
        "Orph#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("11", "+"), ("15", "+"), ("13", "+"), ("5", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
    )
    # Locus (5, 6): 70 bp unaligned boundary tip + 2000 bp homolog -> 70 bp < 150 bp shard dropped -> whole SV dropped!
    gb.add_p_path(
        "Tip#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("16", "+"), ("6", "+"), ("7", "+"), ("8", "+")],
    )
    # Locus (6, 7): untouched standalone 65 bp insertion -> MUST be retained!
    gb.add_p_path(
        "Clean#1#Chr01",
        [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+"), ("5", "+"), ("6", "+"), ("17", "+"), ("7", "+"), ("8", "+")],
    )

    graph, cfg = _build_graph_and_config(
        tmp_path, "u_chr06_chain", gb, ref_steps, dedup_mode="pan", dedup_id=0.90
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
    assert v.passed, f"verify failed: {v.errors}"

    # Loci (3,4) [step 3], (4,5) [step 4], and (5,6) [step 5] MUST all be 100% dropped (0 slot_ins_len)!
    assert res.step_offsets[3].slot_ins_len == 0
    assert res.step_offsets[4].slot_ins_len == 0
    assert res.step_offsets[5].slot_ins_len == 0
    # Untouched standalone 65 bp insertion at (6,7) [step 6] is preserved with 0 'N' spacers!
    assert res.step_offsets[6].slot_ins_len == 65
    assert int(res.summary_metrics["dedup_dropped_groups"]) == 3

    # Strictly zero <50 bp non-N islands in pan.fa
    non_n_runs = [seg for seg in res.pan_seq.split("N") if seg]
    assert all(len(seg) >= 50 for seg in non_n_runs)
    assert all(int(r["pan_end"]) - int(r["pan_start"]) >= 50 for r in res.inserts_rows)

