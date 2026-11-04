"""Milestone 1 Foundation Unit Tests covering models.py, gfa_io.py, and repeat_guard.py."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Callable

import pytest

from panflat.gfa_io import (
    load_and_validate_inputs,
    parse_bed,
    parse_fasta,
    revcomp,
    spell_steps,
)
from panflat.models import (
    BuildConfig,
    CentromereRegion,
    CentromereRow,
    CLIArgumentError,
    InputContractError,
    InsertionSlotBlock,
    PathFragment,
    Piece,
    Step,
    StepProvenance,
    SVEvent,
    UnresolvedRecord,
)
from panflat.repeat_guard import (
    check_similarity_merge,
    complement_intervals,
    merge_contiguous_provenance,
    merge_intervals,
    repeat_mask_intervals,
    slice_candidate_by_intervals,
    subtract_intervals,
    subtract_reference_intervals_mappy,
)
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    make_complex_seq,
    make_satellite_seq,
    make_window_with_exact_singleton_ratio,
    mutate_seq_snps,
    revcomp_dna,
)


# ============================================================================
# Group 1: models.py & Sequence Helpers (Tests 1-2)
# ============================================================================


def test_models_immutability_and_defaults() -> None:
    step = Step(node_id="10", orient="+")
    assert step.slice_start == 0
    assert step.slice_end is None
    assert step.is_partial is False
    assert step.token == ("10", "+")
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.node_id = "20"  # type: ignore[misc]

    flipped_full = step.flipped()
    assert flipped_full == Step(node_id="10", orient="-")

    partial_step = step.subslice(2, 8, full_node_len=10)
    assert partial_step.is_partial is True
    assert partial_step.slice_start == 2
    assert partial_step.slice_end == 8
    flipped_partial = partial_step.flipped(full_node_len=10)
    assert flipped_partial == Step(
        node_id="10", orient="-", slice_start=2, slice_end=8, is_partial=True
    )

    prov = StepProvenance(
        src_hap="SampleA#1",
        src_path="SampleA#1#Chr01",
        src_start=100,
        src_end=150,
        src_strand="+",
    )
    assert prov.length == 50
    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.src_start = 0  # type: ignore[misc]

    cfg = BuildConfig(
        gfa="in.gfa",
        ref_fa="ref.fa",
        backbone_prefix="Nipponbare#1#",
        out_prefix="out",
    )
    assert cfg.ref_gff is None
    assert cfg.min_sv_len == 50
    assert cfg.max_jump == 500000
    assert cfg.cluster_id == 0.95
    assert cfg.cluster_win_id == 0.70
    assert cfg.sec_backbone == "support"
    assert cfg.max_copy == 1
    assert cfg.spacer_n == 100
    assert cfg.splice_window == 8
    assert cfg.threads == 1
    assert cfg.repeat_guard == "on"
    assert cfg.centromere_bed is None
    assert cfg.centromere_mode == "off"
    assert cfg.centromere_min_island == 500
    assert cfg.dump_repeat_sequences is False
    assert cfg.qc is False
    assert cfg.effective_centromere_mode == "off"

    piece = Piece(
        src_hap="SampleA#1",
        src_path="SampleA#1#Chr01",
        src_start=10,
        src_end=60,
        src_strand="+",
        seq="A" * 50,
        join_before="source_contiguous",
        members=("SampleA#1",),
    )
    assert piece.length == 50
    assert piece.src_intervals_str == "10-60"
    assert piece.allele_str == "1"

    unres = UnresolvedRecord(
        record_id="u1",
        region_id=".",
        event_id=".",
        src_path="SampleA#1#Chr01",
        src_start=0,
        src_end=60,
        src_strand="+",
        length=60,
        reason="UNANCHORED_END",
        seq="A" * 60,
        export_to_fasta=True,
    )
    assert unres.exported_fasta_id == "."

    sv = SVEvent(
        chrom="Chr01",
        pan_start=100,
        pan_end=150,
        event_id="Chr01_INS_1",
        event_type="INS",
        ref_start=100,
        ref_end=100,
        anchors="0..1",
        alleles_str="1:len=50,src=SampleA#1,n=1",
        n_carrier=1,
        n_called=2,
        region_id="REG_1",
        genotypes={"Nipponbare#1": "0", "SampleA#1": "1"},
    )
    assert sv.pieces == ()

    slot = InsertionSlotBlock(
        right_anchor_ref_step=1,
        is_centromere_island=False,
        group_span=1,
        left_anchor_ref_step=0,
        event_id="Chr01_INS_1",
        representatives=((piece,),),
    )
    assert slot.trailing_spacer is False

    cen_reg = CentromereRegion(
        region_id="CEN_1",
        chrom="Chr01",
        input_bed_ids=("Chr01:100-200",),
        bed_start=100,
        bed_end=200,
        ref_start=80,
        ref_end=220,
        anchor_u_step=0,
        anchor_v_step=3,
        anchor_u_name="0",
        anchor_v_name="3",
        effective_mode="bypass",
        anchor_not_found=False,
        expanded_to_whole_chromosome=False,
    )
    assert cen_reg.effective_mode == "bypass"

    cen_row = CentromereRow(
        chrom="Chr01",
        region_id="CEN_1",
        input_bed_ids="Chr01:100-200",
        bed_start=100,
        bed_end=200,
        ref_start=80,
        ref_end=220,
        anchor_u="0",
        anchor_v="3",
        mode="bypass",
        hap="Nipponbare#1",
        status="complete",
        ref_status="complete",
        src_path="Nipponbare#1#Chr01",
        src_start=80,
        src_end=220,
        src_strand="+",
        L_ref=140,
        L_path=140,
        length_ratio="1.000000",
        nonref_input_bp=0,
        reference_like_bp=0,
        repeat_like_bp=0,
        short_bp=0,
        unresolved_bp=0,
        accepted_source_bp=0,
        supported_events=".",
    )
    assert len(cen_row.to_tsv_fields()) == 27


def test_revcomp_and_spell_steps_full_and_partial() -> None:
    assert revcomp("AAACCCGGGTTTNNN") == "NNNAAACCCGGGTTT"
    assert revcomp("acgtNn") == "nNacgt"

    nodes = {"n1": "AAACCC", "n2": "GGGTTT"}
    full_steps = (Step("n1", "+"), Step("n2", "-"))
    assert spell_steps(full_steps, nodes) == "AAACCCAAACCC"

    # Partial steps: n1+ [2:5) -> "ACC", n2- ("AAACCC") [1:4) -> "AAC"
    partial_steps = (
        Step("n1", "+", slice_start=2, slice_end=5, is_partial=True),
        Step("n2", "-", slice_start=1, slice_end=4, is_partial=True),
    )
    assert spell_steps(partial_steps, nodes) == "ACCAAC"


# ============================================================================
# Group 2: gfa_io.py — Input Contracts & BED Validation (Tests 3-17)
# ============================================================================


def test_valid_p_and_w_lines_negative_ref_steps_and_sha256(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("n1", "ACTGACTGAA")  # 10 bp
    builder.add_node("n2", "CCNGGTTAA")  # 9 bp, contains N
    builder.add_node("n3", "GGGCCCAAA")  # 9 bp
    builder.add_node("n4", "TTTTAAAA")  # 8 bp

    ref_steps = [("n1", "+"), ("n2", "-"), ("n3", "+")]
    builder.add_p_path("Nipponbare#1#Chr01", ref_steps, overlaps="*")
    builder.add_p_path("AltA#1#Chr01", [("n1", "+"), ("n4", "+"), ("n3", "+")], overlaps="*")
    builder.add_w_walk("AltB", 1, "ctg1", 500, 518, [("n1", "+"), ("n4", "-")])
    builder.add_w_walk("AltB", 1, "ctg1", 1000, 1018, [("n2", "+"), ("n3", "+")])

    gfa_path = builder.write_gfa(tmp_path / "test.gfa")
    ref_spelled_mixed = ("ACTGACTGAA" + revcomp_dna("CCNGGTTAA") + "GGGCCCAAA").lower()
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": ref_spelled_mixed})

    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Nipponbare#1#",
        out_prefix=str(tmp_path / "out"),
    )
    graph = load_and_validate_inputs(cfg)

    assert graph.gfa_sha256 == hashlib.sha256(gfa_path.read_bytes()).hexdigest()
    assert graph.ref_chrom == "Chr01"
    assert graph.ref_seq == "ACTGACTGAATTAACCNGGGGGCCCAAA"
    assert graph.ref_step_ranges == ((0, 10), (10, 19), (19, 28))
    assert graph.global_ref_nodes == frozenset({"n1", "n2", "n3"})
    assert graph.unique_ref_step_by_node == {"n1": 0, "n2": 1, "n3": 2}
    assert graph.all_haplotypes == ("Nipponbare#1", "AltA#1", "AltB#1")
    assert len(graph.alt_fragments) == 3

    w1 = graph.alt_fragments[1]
    assert w1.is_w_line is True
    assert w1.base_offset == 500
    assert w1.src_path == "AltB#1#ctg1:500-518"
    assert w1.provenance[0].src_start == 500
    assert w1.provenance[0].src_end == 510
    assert w1.provenance[1].src_start == 510
    assert w1.provenance[1].src_end == 518

    w2 = graph.alt_fragments[2]
    assert w2.is_w_line is True
    assert w2.base_offset == 1000
    assert w2.src_path == "AltB#1#ctg1:1000-1018"
    assert w2.provenance[0].src_start == 1000
    assert w2.provenance[0].src_end == 1009


def test_ref_repeated_node_excluded_from_unique_ref_step_by_node(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("R1", "AAAAACCCCC")
    builder.add_node("R_dup", "GGGGGTTTTT")
    builder.add_node("R2", "ATCGATCGAT")
    steps = [("R1", "+"), ("R_dup", "+"), ("R_dup", "+"), ("R2", "+")]
    builder.add_p_path("Ref#1#Chr01", steps)

    gfa_path = builder.write_gfa(tmp_path / "dup.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": builder.spell_steps(steps)})

    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
    )
    graph = load_and_validate_inputs(cfg)
    assert graph.global_ref_nodes == frozenset({"R1", "R_dup", "R2"})
    assert graph.unique_ref_step_by_node == {"R1": 0, "R2": 3}
    assert "R_dup" not in graph.unique_ref_step_by_node


def test_string_node_ids_and_bidirectional_l_links(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("node_A", "ACGTACGTAA")
    builder.add_node("tig_01", "TTGCATGCAA")
    # Add only reverse-complement L link: tig_01- -> node_A- is equivalent to node_A+ -> tig_01+
    builder.add_link("tig_01", "-", "node_A", "-", overlap="0M")
    builder.add_p_path(
        "Ref#1#Chr01",
        [("node_A", "+"), ("tig_01", "+")],
        overlaps="*",
        auto_links=False,
    )

    gfa_path = builder.write_gfa(tmp_path / "bidir.gfa")
    fa_path = builder.write_fasta(
        tmp_path / "ref.fa",
        {"Chr01": "ACGTACGTAATTGCATGCAA"},
    )
    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
    )
    graph = load_and_validate_inputs(cfg)
    assert ("node_A", "+", "tig_01", "+") in graph.links
    assert ("tig_01", "-", "node_A", "-") in graph.links


def test_reject_non_unique_backbone_prefix_zero_or_multiple(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTACGTAA")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    builder.add_p_path("Ref#1#Chr02", [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "multi.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "ACGTACGTAA"})

    # 0 matches
    cfg_zero = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Missing#",
        out_prefix=str(tmp_path / "out"),
    )
    with pytest.raises(InputContractError):
        load_and_validate_inputs(cfg_zero)

    # 2 matches
    cfg_multi = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
    )
    with pytest.raises(InputContractError):
        load_and_validate_inputs(cfg_multi)


def test_reject_non_unique_or_empty_ref_fasta(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTACGTAA")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "ok.gfa")

    empty_fa = tmp_path / "empty.fa"
    empty_fa.write_text("", encoding="utf-8")
    with pytest.raises(InputContractError):
        parse_fasta(str(empty_fa))

    multi_fa = builder.write_fasta(
        tmp_path / "multi.fa",
        {"Chr01": "ACGTACGTAA", "Chr02": "ACGTACGTAA"},
    )
    cfg = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(multi_fa),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
    )
    with pytest.raises(InputContractError):
        load_and_validate_inputs(cfg)


def test_reject_ref_base_mismatch_and_n_mismatch(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTNACGTA")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "ref_n.gfa")

    # 1. N in GFA vs A in FASTA
    fa_no_n = builder.write_fasta(tmp_path / "ref1.fa", {"Chr01": "ACGTAACGTA"})
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_no_n),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )

    # 2. A in GFA vs N in FASTA
    fa_extra_n = builder.write_fasta(tmp_path / "ref2.fa", {"Chr01": "ACGTNNCGTA"})
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_extra_n),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )

    # 3. Length mismatch
    fa_short = builder.write_fasta(tmp_path / "ref3.fa", {"Chr01": "ACGTNACGT"})
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_short),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_missing_node_sequence_star(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "*")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "star_node.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "ACGT"})

    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_nonzero_or_unknown_l_overlap(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    for bad_ov in ("10M", "*", "5M1I"):
        builder = gfa_builder()
        builder.add_node("1", "ACGTACGTAA")
        builder.add_node("2", "TTGCATGCAA")
        builder.add_link("1", "+", "2", "+", overlap=bad_ov)
        builder.add_p_path(
            "Ref#1#Chr01",
            [("1", "+"), ("2", "+")],
            overlaps="*",
            auto_links=False,
        )
        gfa_path = builder.write_gfa(tmp_path / f"bad_l_{bad_ov}.gfa")
        fa_path = builder.write_fasta(
            tmp_path / "ref.fa", {"Chr01": "ACGTACGTAATTGCATGCAA"}
        )
        with pytest.raises(InputContractError):
            load_and_validate_inputs(
                BuildConfig(
                    gfa=str(gfa_path),
                    ref_fa=str(fa_path),
                    backbone_prefix="Ref#1#",
                    out_prefix=str(tmp_path / "out"),
                )
            )


def test_reject_nonzero_p_overlap_and_star_without_0m_link(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    fa_path = SyntheticGFAFixtureBuilder.write_fasta(
        tmp_path / "ref.fa", {"Chr01": "ACGTACGTAATTGCATGCAA"}
    )

    # 1. Explicit non-zero P overlap "5M"
    b1 = gfa_builder()
    b1.add_node("1", "ACGTACGTAA")
    b1.add_node("2", "TTGCATGCAA")
    b1.add_p_path("Ref#1#Chr01", [("1", "+"), ("2", "+")], overlaps="5M", auto_links=True)
    gfa1 = b1.write_gfa(tmp_path / "bad_p_ov.gfa")
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa1),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )

    # 2. P overlap "*" when 0M L-link is missing
    b2 = gfa_builder()
    b2.add_node("1", "ACGTACGTAA")
    b2.add_node("2", "TTGCATGCAA")
    b2.add_p_path("Ref#1#Chr01", [("1", "+"), ("2", "+")], overlaps="*", auto_links=False)
    gfa2 = b2.write_gfa(tmp_path / "missing_l_for_star.gfa")
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa2),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_broken_link_in_p_or_w_walk(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTACGTAA")
    builder.add_node("2", "TTGCATGCAA")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    # W walk transitions 1+ -> 2+ without an L link
    builder.add_w_walk(
        "Alt", 1, "ctg1", 0, 20, [("1", "+"), ("2", "+")], auto_links=False
    )
    gfa_path = builder.write_gfa(tmp_path / "broken_w.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "ACGTACGTAA"})

    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_j_line_presence(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTACGTAA")
    builder.add_node("2", "TTGCATGCAA")
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    builder.add_j_line("1", "+", "2", "+")
    gfa_path = builder.write_gfa(tmp_path / "j_line.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "ACGTACGTAA"})

    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_duplicate_p_and_w_interval_ambiguity(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    fa_path = SyntheticGFAFixtureBuilder.write_fasta(
        tmp_path / "ref.fa", {"Chr01": "ACGTACGTAA"}
    )

    # 1. Same haplotype/seqid in both P and W lines
    b1 = gfa_builder()
    b1.add_node("1", "ACGTACGTAA")
    b1.add_p_path("Ref#1#Chr01", [("1", "+")])
    b1.add_p_path("Alt#1#ctg1", [("1", "+")])
    b1.add_w_walk("Alt", 1, "ctg1", 0, 10, [("1", "+")])
    gfa1 = b1.write_gfa(tmp_path / "dup_pw.gfa")
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa1),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )

    # 2. Overlapping W lines on the same Sample#Hap#seqid
    b2 = gfa_builder()
    b2.add_node("1", "ACGTACGTAA")
    b2.add_p_path("Ref#1#Chr01", [("1", "+")])
    b2.add_w_walk("Alt", 1, "ctg1", 100, 110, [("1", "+")])
    b2.add_w_walk("Alt", 1, "ctg1", 105, 115, [("1", "+")])
    gfa2 = b2.write_gfa(tmp_path / "ov_ww.gfa")
    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa2),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_reject_w_line_length_mismatch(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "ACGTACGTAA")  # 10 bp
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    # W line declares span [100, 125) = 25 bp for a 10 bp step
    builder.add_w_walk("Alt", 1, "ctg1", 100, 125, [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "w_len_mismatch.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "ACGTACGTAA"})

    with pytest.raises(InputContractError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
            )
        )


def test_parse_bed_valid_3col_and_4col_and_validation_errors(
    tmp_path: Path,
) -> None:
    bed_ok = SyntheticGFAFixtureBuilder.write_bed(
        tmp_path / "ok.bed",
        [
            ("Chr01", 100, 300),
            ("Chr01", 500, 800, "CEN_CUSTOM"),
        ],
    )
    parsed = parse_bed(str(bed_ok), ref_chrom="Chr01", ref_len=1000)
    assert parsed == [
        ("Chr01", 100, 300, "Chr01:100-300"),
        ("Chr01", 500, 800, "CEN_CUSTOM"),
    ]

    # Wrong chrom
    bed_wrong_chrom = SyntheticGFAFixtureBuilder.write_bed(
        tmp_path / "wrong_chrom.bed", [("Chr02", 100, 200)]
    )
    with pytest.raises(InputContractError):
        parse_bed(str(bed_wrong_chrom), ref_chrom="Chr01", ref_len=1000)

    # start < 0
    bed_neg = SyntheticGFAFixtureBuilder.write_bed(
        tmp_path / "neg.bed", [("Chr01", -5, 200)]
    )
    with pytest.raises(InputContractError):
        parse_bed(str(bed_neg), ref_chrom="Chr01", ref_len=1000)

    # start >= end
    bed_inv = SyntheticGFAFixtureBuilder.write_bed(
        tmp_path / "inv.bed", [("Chr01", 200, 200)]
    )
    with pytest.raises(InputContractError):
        parse_bed(str(bed_inv), ref_chrom="Chr01", ref_len=1000)

    # end > ref_len
    bed_oob = SyntheticGFAFixtureBuilder.write_bed(
        tmp_path / "oob.bed", [("Chr01", 100, 1001)]
    )
    with pytest.raises(InputContractError):
        parse_bed(str(bed_oob), ref_chrom="Chr01", ref_len=1000)

    # < 3 columns
    bed_2col = tmp_path / "2col.bed"
    bed_2col.write_text("Chr01\t100\n", encoding="utf-8")
    with pytest.raises(InputContractError):
        parse_bed(str(bed_2col), ref_chrom="Chr01", ref_len=1000)

    # Empty BED
    bed_empty = tmp_path / "empty.bed"
    bed_empty.write_text("# comment only\n", encoding="utf-8")
    with pytest.raises(InputContractError):
        parse_bed(str(bed_empty), ref_chrom="Chr01", ref_len=1000)


def test_centromere_cli_mode_and_bed_conflict_validation(
    tmp_path: Path,
    gfa_builder: Callable[[], SyntheticGFAFixtureBuilder],
) -> None:
    builder = gfa_builder()
    builder.add_node("1", "A" * 1000)
    builder.add_p_path("Ref#1#Chr01", [("1", "+")])
    gfa_path = builder.write_gfa(tmp_path / "cen.gfa")
    fa_path = builder.write_fasta(tmp_path / "ref.fa", {"Chr01": "A" * 1000})
    bed_path = builder.write_bed(tmp_path / "cen.bed", [("Chr01", 200, 600)])

    # 1. BED provided without explicit mode defaults to 'bypass'
    cfg_auto_bypass = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
        centromere_bed=str(bed_path),
        centromere_mode="off",
        centromere_mode_explicit=False,
    )
    assert cfg_auto_bypass.effective_centromere_mode == "bypass"
    load_and_validate_inputs(cfg_auto_bypass)

    # 2. Explicit --centromere-mode off + --centromere-bed raises CLIArgumentError
    cfg_conflict = BuildConfig(
        gfa=str(gfa_path),
        ref_fa=str(fa_path),
        backbone_prefix="Ref#1#",
        out_prefix=str(tmp_path / "out"),
        centromere_bed=str(bed_path),
        centromere_mode="off",
        centromere_mode_explicit=True,
    )
    with pytest.raises(CLIArgumentError):
        load_and_validate_inputs(cfg_conflict)

    # 3. --centromere-mode bypass or extract-islands without --centromere-bed raises error
    for mode in ("bypass", "extract-islands"):
        with pytest.raises(CLIArgumentError):
            load_and_validate_inputs(
                BuildConfig(
                    gfa=str(gfa_path),
                    ref_fa=str(fa_path),
                    backbone_prefix="Ref#1#",
                    out_prefix=str(tmp_path / "out"),
                    centromere_bed=None,
                    centromere_mode=mode,  # type: ignore[arg-type]
                    centromere_mode_explicit=True,
                )
            )

    # 4. --centromere-min-island < --min-sv-len raises error
    with pytest.raises(CLIArgumentError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
                centromere_bed=str(bed_path),
                centromere_mode="extract-islands",
                centromere_mode_explicit=True,
                min_sv_len=100,
                centromere_min_island=80,
                centromere_min_island_explicit=True,
            )
        )

    # 5. Explicit --centromere-min-island when centromere mode is off raises error
    with pytest.raises(CLIArgumentError):
        load_and_validate_inputs(
            BuildConfig(
                gfa=str(gfa_path),
                ref_fa=str(fa_path),
                backbone_prefix="Ref#1#",
                out_prefix=str(tmp_path / "out"),
                centromere_bed=None,
                centromere_mode="off",
                centromere_min_island=500,
                centromere_min_island_explicit=True,
            )
        )


# ============================================================================
# Group 3: repeat_guard.py — repeat_mask_intervals & slice_candidate_by_intervals (Tests 18-23)
# ============================================================================


def test_repeat_mask_intervals_length_boundaries_499_to_2500() -> None:
    # L = 499: < 500 bp, not evaluated
    masked_499, diag_499 = repeat_mask_intervals(make_satellite_seq(20, 499))
    assert masked_499 == []
    assert diag_499 == []

    # L = 500, 999, 1000: single window [0, L)
    for length in (500, 999, 1000):
        masked, diag = repeat_mask_intervals(make_satellite_seq(20, length))
        assert [(w["start"], w["end"]) for w in diag] == [(0, length)]
        assert masked == [(0, length)]

    # L = 1001: 2 windows [0, 1000) and [1, 1001)
    masked_1001, diag_1001 = repeat_mask_intervals(make_satellite_seq(20, 1001))
    assert [(w["start"], w["end"]) for w in diag_1001] == [(0, 1000), (1, 1001)]
    assert masked_1001 == [(0, 1001)]

    # L = 1500: 2 deduplicated windows [0, 1000) and [500, 1500)
    masked_1500, diag_1500 = repeat_mask_intervals(make_satellite_seq(20, 1500))
    assert [(w["start"], w["end"]) for w in diag_1500] == [(0, 1000), (500, 1500)]
    assert masked_1500 == [(0, 1500)]

    # L = 2500: 4 windows [0, 1000), [500, 1500), [1000, 2000), [1500, 2500)
    # Complex flanks [0, 500) and [2000, 2500) + satellite middle [500, 2000)
    hybrid_2500 = (
        make_complex_seq(500, seed=11)
        + ("A" * 1500)
        + make_complex_seq(500, seed=12)
    )
    masked_2500, diag_2500 = repeat_mask_intervals(hybrid_2500)
    assert [(w["start"], w["end"]) for w in diag_2500] == [
        (0, 1000),
        (500, 1500),
        (1000, 2000),
        (1500, 2500),
    ]
    assert [w["state"] for w in diag_2500] == ["non-low", "low", "low", "non-low"]
    assert masked_2500 == [(500, 2000)]


def test_repeat_mask_intervals_singleton_ratio_exact_boundaries() -> None:
    # 514 bp window -> N_pos = 500
    # 1. 149/500 = 0.298 < 0.30 -> low -> masked
    seq_low = make_window_with_exact_singleton_ratio(514, target_singletons=149, seed=201)
    masked_low, diag_low = repeat_mask_intervals(seq_low)
    assert diag_low[0]["singleton_count"] == 149
    assert diag_low[0]["valid_positions"] == 500
    assert diag_low[0]["singleton_ratio"] == pytest.approx(0.298)
    assert diag_low[0]["state"] == "low"
    assert masked_low == [(0, 514)]

    # 2. 150/500 = 0.300 == 0.30 -> non-low -> unmasked (C10)
    seq_eq30 = make_window_with_exact_singleton_ratio(514, target_singletons=150, seed=202)
    masked_eq30, diag_eq30 = repeat_mask_intervals(seq_eq30)
    assert diag_eq30[0]["singleton_count"] == 150
    assert diag_eq30[0]["singleton_ratio"] == pytest.approx(0.300)
    assert diag_eq30[0]["state"] == "non-low"
    assert masked_eq30 == []

    # 3. 225/500 = 0.450 -> non-low -> unmasked (C10)
    seq_mid = make_window_with_exact_singleton_ratio(514, target_singletons=225, seed=203)
    masked_mid, diag_mid = repeat_mask_intervals(seq_mid)
    assert diag_mid[0]["singleton_ratio"] == pytest.approx(0.450)
    assert diag_mid[0]["state"] == "non-low"
    assert masked_mid == []

    # 4. 300/500 = 0.600 -> non-low -> unmasked (C10)
    seq_eq60 = make_window_with_exact_singleton_ratio(514, target_singletons=300, seed=204)
    masked_eq60, diag_eq60 = repeat_mask_intervals(seq_eq60)
    assert diag_eq60[0]["singleton_ratio"] == pytest.approx(0.600)
    assert diag_eq60[0]["state"] == "non-low"
    assert masked_eq60 == []

    # 5. 1000 bp window (N_pos = 986): 295/986 < 0.30 vs 296/986 > 0.30, and 270/900 == 0.30
    seq_1k_low = make_window_with_exact_singleton_ratio(1000, target_singletons=295, seed=205)
    m_1k_low, d_1k_low = repeat_mask_intervals(seq_1k_low)
    assert d_1k_low[0]["state"] == "low"
    assert m_1k_low == [(0, 1000)]

    seq_1k_nonlow = make_window_with_exact_singleton_ratio(1000, target_singletons=296, seed=206)
    m_1k_nonlow, d_1k_nonlow = repeat_mask_intervals(seq_1k_nonlow)
    assert d_1k_nonlow[0]["state"] == "non-low"
    assert m_1k_nonlow == []

    seq_1k_exact30 = make_window_with_exact_singleton_ratio(
        1000, target_singletons=270, valid_positions=900, seed=207
    )
    m_1k_ex30, d_1k_ex30 = repeat_mask_intervals(seq_1k_exact30)
    assert d_1k_ex30[0]["singleton_ratio"] == pytest.approx(0.30)
    assert d_1k_ex30[0]["state"] == "non-low"
    assert m_1k_ex30 == []


def test_repeat_mask_intervals_canonical_revcomp_kmer_equivalence() -> None:
    half = make_complex_seq(300, seed=301)
    # Inverted repeat: forward 15-mers are nearly all unique, but canonical 15-mers
    # each appear twice (once in `half`, once in `revcomp_dna(half)`).
    inv_repeat = half + revcomp_dna(half)
    masked, diag = repeat_mask_intervals(inv_repeat)
    assert len(diag) == 1
    assert diag[0]["singleton_ratio"] < 0.05
    assert diag[0]["state"] == "low"
    assert masked == [(0, 600)]


def test_repeat_mask_intervals_unknown_window_90pct_and_adjacency_break() -> None:
    # 1. 1000 bp window (N_pos = 986): 888 valid (90.06% >= 90%) vs 887 valid (89.96% < 90%)
    seq_888 = ("A" * 902) + ("N" * 98)  # 902 - 14 = 888 valid 15-mers, all 'A'*15
    m_888, d_888 = repeat_mask_intervals(seq_888)
    assert d_888[0]["valid_positions"] == 888
    assert d_888[0]["state"] == "low"
    assert m_888 == [(0, 1000)]

    seq_887 = ("A" * 901) + ("N" * 99)  # 901 - 14 = 887 valid 15-mers (< 90%)
    m_887, d_887 = repeat_mask_intervals(seq_887)
    assert d_887[0]["valid_positions"] == 887
    assert d_887[0]["state"] == "unknown"
    assert d_887[0]["singleton_ratio"] is None
    assert m_887 == []

    # 2. 2000 bp sequence with ["low", "unknown", "low"] windows (C19):
    # W0=[0, 1000), W1=[500, 1500), W2=[1000, 2000)
    # Place 7 'N' bases at [950, 965, 980, 1005, 1020, 1035, 1050] inside a satellite
    arr = list("A" * 2000)
    for pos in (950, 965, 980, 1005, 1020, 1035, 1050):
        arr[pos] = "N"
    seq_2000 = "".join(arr)
    m_2000, d_2000 = repeat_mask_intervals(seq_2000)
    assert [w["state"] for w in d_2000] == ["low", "unknown", "low"]
    # Because "unknown" interrupts the "low" run, neither isolated "low" window is masked!
    assert m_2000 == []


def test_repeat_mask_intervals_short_block_exception_vs_long_isolated_low() -> None:
    # 1. Short-block exception (500 <= L <= 1000): 1 low window masks [0, L)
    m_short, d_short = repeat_mask_intervals("A" * 750)
    assert len(d_short) == 1 and d_short[0]["state"] == "low"
    assert m_short == [(0, 750)]

    # 2. L = 1500 with 1 low window W0=[0, 1000) and 1 non-low window W1=[500, 1500):
    # First 500 bp complex, middle 500 bp 'A'*500, last 500 bp complex ->
    # Make [0, 800) 'A'*800 and [800, 1500) complex (700 bp complex in W1 -> 700 singletons / 986 = 0.71 non-low)
    seq_1500_isolated = ("A" * 800) + make_complex_seq(700, seed=401)
    m_1500, d_1500 = repeat_mask_intervals(seq_1500_isolated)
    assert [w["state"] for w in d_1500] == ["low", "non-low"]
    assert m_1500 == []


def test_slice_candidate_by_intervals_partial_node_plus_and_minus_steps_and_strands() -> None:
    # Build an original 1000 bp contig
    orig_contig = make_complex_seq(1000, seed=450)
    # Candidate occupies [100, 500) on orig_contig (400 bp across 3 steps: 100 bp, 200 bp, 100 bp)
    # Case A: src_strand == '+' -> candidate sequence is orig_contig[100:500]
    cand_plus_seq = orig_contig[100:500]
    nodes_plus = {
        "n1": cand_plus_seq[0:100],
        "n2": revcomp_dna(cand_plus_seq[100:300]),  # traversed as n2-
        "n3": cand_plus_seq[300:400],
    }
    steps = (Step("n1", "+"), Step("n2", "-"), Step("n3", "+"))
    prov_plus = (
        StepProvenance("Hap#1", "Hap#1#ctg1", 100, 200, "+"),
        StepProvenance("Hap#1", "Hap#1#ctg1", 200, 400, "+"),
        StepProvenance("Hap#1", "Hap#1#ctg1", 400, 500, "+"),
    )

    kept_plus, masked_plus = slice_candidate_by_intervals(
        steps=steps,
        provenance=prov_plus,
        nodes=nodes_plus,
        keep_intervals=[(0, 160), (240, 400)],
        mask_intervals=[(160, 240)],
    )
    assert len(kept_plus) == 2
    assert len(masked_plus) == 1

    # First residual [0, 160): n1+ (full) + n2- [0:60) (partial)
    s0, e0, st0, pr0, seq0 = kept_plus[0]
    assert (s0, e0) == (0, 160)
    assert seq0 == cand_plus_seq[0:160]
    assert spell_steps(st0, nodes_plus) == seq0
    assert st0[0].is_partial is False
    assert st0[1] == Step("n2", "-", slice_start=0, slice_end=60, is_partial=True)
    assert pr0[0] == StepProvenance("Hap#1", "Hap#1#ctg1", 100, 200, "+")
    assert pr0[1] == StepProvenance("Hap#1", "Hap#1#ctg1", 200, 260, "+")
    assert "".join(orig_contig[p.src_start : p.src_end] for p in pr0) == seq0

    # Masked [160, 240): n2- [60:140)
    ms0, me0, mpr0, mseq0 = masked_plus[0]
    assert (ms0, me0) == (160, 240)
    assert mseq0 == cand_plus_seq[160:240]
    assert mpr0 == (StepProvenance("Hap#1", "Hap#1#ctg1", 260, 340, "+"),)
    assert orig_contig[260:340] == mseq0

    # Second residual [240, 400): n2- [140:200) (partial) + n3+ (full)
    s1, e1, st1, pr1, seq1 = kept_plus[1]
    assert (s1, e1) == (240, 400)
    assert seq1 == cand_plus_seq[240:400]
    assert spell_steps(st1, nodes_plus) == seq1
    assert st1[0] == Step("n2", "-", slice_start=140, slice_end=200, is_partial=True)
    assert st1[1].is_partial is False
    assert pr1[0] == StepProvenance("Hap#1", "Hap#1#ctg1", 340, 400, "+")
    assert pr1[1] == StepProvenance("Hap#1", "Hap#1#ctg1", 400, 500, "+")
    assert "".join(orig_contig[p.src_start : p.src_end] for p in pr1) == seq1

    # Case B: src_strand == '-' -> candidate sequence is revcomp_dna(orig_contig[100:500])
    cand_minus_seq = revcomp_dna(orig_contig[100:500])
    nodes_minus = {
        "n1": cand_minus_seq[0:100],
        "n2": revcomp_dna(cand_minus_seq[100:300]),  # traversed as n2-
        "n3": cand_minus_seq[300:400],
    }
    prov_minus = (
        StepProvenance("Hap#1", "Hap#1#ctg1", 400, 500, "-"),
        StepProvenance("Hap#1", "Hap#1#ctg1", 200, 400, "-"),
        StepProvenance("Hap#1", "Hap#1#ctg1", 100, 200, "-"),
    )
    kept_minus, masked_minus = slice_candidate_by_intervals(
        steps=steps,
        provenance=prov_minus,
        nodes=nodes_minus,
        keep_intervals=[(0, 160), (240, 400)],
        mask_intervals=[(160, 240)],
    )
    for _, _, st_tup, pr_tup, sub_seq in kept_minus:
        assert spell_steps(st_tup, nodes_minus) == sub_seq
        reconstructed = "".join(
            revcomp_dna(orig_contig[p.src_start : p.src_end]) for p in pr_tup
        )
        assert reconstructed == sub_seq
        merged_provs = merge_contiguous_provenance(pr_tup, sub_seq)
        assert len(merged_provs) == 1
        mp, m_seq = merged_provs[0]
        assert m_seq == sub_seq
        assert revcomp_dna(orig_contig[mp.src_start : mp.src_end]) == sub_seq

    _, _, mpr_minus, mseq_minus = masked_minus[0]
    assert mpr_minus == (StepProvenance("Hap#1", "Hap#1#ctg1", 260, 340, "-"),)
    assert revcomp_dna(orig_contig[260:340]) == mseq_minus

    # Also test re-slicing an already partial step
    resliced_kept, _ = slice_candidate_by_intervals(
        steps=kept_minus[0][2],
        provenance=kept_minus[0][3],
        nodes=nodes_minus,
        keep_intervals=[(110, 150)],
        mask_intervals=[],
    )
    _, _, r_steps, r_provs, r_seq = resliced_kept[0]
    assert r_steps == (Step("n2", "-", slice_start=10, slice_end=50, is_partial=True),)
    assert spell_steps(r_steps, nodes_minus) == r_seq
    assert revcomp_dna(orig_contig[r_provs[0].src_start : r_provs[0].src_end]) == r_seq


# ============================================================================
# Group 4: repeat_guard.py — subtract_reference_intervals_mappy (Tests 24-27)
# ============================================================================


def test_subtract_ref_mappy_forward_and_reverse_strand_hits() -> None:
    ref_seq = make_complex_seq(2000, seed=501)

    # 1. Forward strand exact match (600 bp)
    q_fwd = ref_seq[300:900]
    masked_f, unmasked_f, ref_bp_f = subtract_reference_intervals_mappy(q_fwd, ref_seq)
    assert masked_f == [(0, 600)]
    assert unmasked_f == []
    assert ref_bp_f == 600

    # 2. Reverse-complement match embedded between two novel 250 bp flanks (C9)
    flank_l = make_complex_seq(250, seed=502)
    flank_r = make_complex_seq(250, seed=503)
    q_rev = flank_l + revcomp_dna(ref_seq[500:1000]) + flank_r
    masked_r, unmasked_r, ref_bp_r = subtract_reference_intervals_mappy(q_rev, ref_seq)
    assert masked_r == [(250, 750)]
    assert unmasked_r == [(0, 250), (750, 1000)]
    assert ref_bp_r == 500


def test_subtract_ref_mappy_protects_internal_insertion_ge_min_sv_len() -> None:
    ref_seq = make_complex_seq(4000, seed=601)

    # 1. C8: 600 bp novel insertion flanked by 500 bp homologous arms
    ins_600 = make_complex_seq(600, seed=602)
    q_c8 = ref_seq[200:700] + ins_600 + ref_seq[700:1200]
    masked_c8, unmasked_c8, ref_bp_c8 = subtract_reference_intervals_mappy(
        q_c8, ref_seq, min_sv_len=50
    )
    assert masked_c8 == [(0, 500), (1100, 1600)]
    assert unmasked_c8 == [(500, 1100)]
    assert ref_bp_c8 == 1000

    # 2. 60 bp novel insertion (>= min_sv_len=50) chained inside a single mappy hit
    ins_60 = make_complex_seq(60, seed=603)
    q_60 = ref_seq[200:700] + ins_60 + ref_seq[700:1200]
    masked_60, unmasked_60, _ = subtract_reference_intervals_mappy(
        q_60, ref_seq, min_sv_len=50
    )
    # The 60 bp insertion [500, 560) must be protected and present in unmasked_60
    unmasked_set = set()
    for s, e in unmasked_60:
        unmasked_set.update(range(s, e))
    assert set(range(500, 560)).issubset(unmasked_set)

    # 3. 4 homologous 500 bp flanks separated by three 800 bp novel insertions
    ins1 = make_complex_seq(800, seed=602)
    ins2 = make_complex_seq(800, seed=603)
    ins3 = make_complex_seq(800, seed=604)
    q_4f = (
        ref_seq[200:700]
        + ins1
        + ref_seq[700:1200]
        + ins2
        + ref_seq[1200:1700]
        + ins3
        + ref_seq[1700:2200]
    )
    masked_4f, unmasked_4f, ref_bp_4f = subtract_reference_intervals_mappy(
        q_4f, ref_seq, min_sv_len=50
    )
    assert masked_4f == [(0, 500), (1300, 1800), (2600, 3100), (3900, 4400)]
    assert unmasked_4f == [(500, 1300), (1800, 2600), (3100, 3900)]
    assert ref_bp_4f == 2000


def test_subtract_ref_mappy_50bp_block_coverage_90pct_and_identity_85pct_and_tail() -> None:
    ref_seq = make_complex_seq(2000, seed=604)

    # 530 bp segment from ref_seq (10 full 50 bp blocks + 30 bp tail block)
    # Mutate block 4 ([200, 250)) with 9 mismatches -> 41/50 = 0.82 < 0.85 (unmasked)
    # Mutate block 2 ([100, 150)) with 7 mismatches -> 43/50 = 0.86 >= 0.85 (masked)
    base_q = ref_seq[100:630]
    snp_pos = [
        105, 110, 115, 120, 125, 130, 135,  # 7 mismatches in [100, 150) -> 86% identity
        204, 208, 212, 216, 220, 224, 228, 232, 236,  # 9 mismatches in [200, 250) -> 82% identity
    ]
    q_mut = mutate_seq_snps(base_q, snp_pos)
    masked, unmasked, ref_bp = subtract_reference_intervals_mappy(q_mut, ref_seq)
    assert masked == [(0, 200), (250, 530)]
    assert unmasked == [(200, 250)]
    assert ref_bp == 480


def test_subtract_ref_mappy_deletion_assigned_to_right_block() -> None:
    ref_seq = make_complex_seq(1000, seed=605)
    # 1. Single 10 bp deletion of reference (ref_seq[300:310]) at query coordinate 200
    # Left block [150, 200) has M=50, D=0 -> identity 50/50 = 1.00 >= 0.85 (masked)
    # Right block [200, 250) receives D=10 -> M=50, D=10 -> identity 50/60 = 0.833 < 0.85 (unmasked)
    q_del = ref_seq[100:300] + ref_seq[310:510]
    masked, unmasked, ref_bp = subtract_reference_intervals_mappy(q_del, ref_seq)
    assert masked == [(0, 200), (250, 400)]
    assert unmasked == [(200, 250)]
    assert ref_bp == 350

    # 2. Multi-block 10 bp deletions at exact 50 bp boundaries q = 200, 250, ..., 650
    # Already-aligned blocks [200, 700) rejected due to < 0.85 identity must NOT be re-mapped
    # on a sliced sub-interval that severs the boundary deletion at q = 200
    blocks = ["A" + make_complex_seq(50, seed=1000 + i)[1:-1] + "A" for i in range(25)]
    del_seq = "CCCCCCCCCC"
    ref_custom = (
        "".join(blocks[0:4])
        + "".join(del_seq + blocks[i] for i in range(4, 14))
        + "".join(blocks[14:20])
    )
    q_multi_del = "".join(blocks[0:20])
    masked_md, unmasked_md, ref_bp_md = subtract_reference_intervals_mappy(
        q_multi_del, ref_custom
    )
    assert masked_md == [(0, 200), (700, 1000)]
    assert unmasked_md == [(200, 700)]
    assert ref_bp_md == 500


# ============================================================================
# Group 5: repeat_guard.py — check_similarity_merge (Tests 28-32)
# ============================================================================


def test_check_similarity_merge_edlib_dispersed_snps_and_gap_compressed_id() -> None:
    rep = make_complex_seq(600, seed=701)

    # 1. 97% identity (18 dispersed SNPs across 600 bp, <= 2 per 50 bp window) -> merges (True)
    snps_97 = [i * 32 + 10 for i in range(18)]
    cand_97 = mutate_seq_snps(rep, snps_97)
    assert check_similarity_merge(cand_97, rep) is True

    # 2. 94% identity (36 dispersed SNPs across 600 bp -> 564/600 = 0.94 < 0.95) -> fails Criterion 2 (False)
    snps_94 = [i * 16 + 5 for i in range(36)]
    cand_94 = mutate_seq_snps(rep, snps_94)
    assert check_similarity_merge(cand_94, rep) is False

    # 3. Single 30 bp deletion in S: M=570, X=0, gap_open_D=1 -> ID_gc = 570/571 = 0.998 >= 0.95 -> True
    cand_del30 = rep[:250] + rep[280:]
    assert check_similarity_merge(cand_del30, rep) is True

    # 4. Single 30 bp insertion in S: M=600, X=0, gap_open_I=1 -> ID_gc = 600/601 = 0.998 >= 0.95 -> True
    cand_ins30 = rep[:250] + make_complex_seq(30, seed=702) + rep[250:]
    assert check_similarity_merge(cand_ins30, rep) is True

    # 5. Two separate small deletions (25 bp at 250..275 and 10 bp at 550..560) with only 'M' between them:
    # ID_gc = 765 / 767 = 0.9974 >= 0.95 -> merges (True)
    rep_800 = make_complex_seq(800, seed=701)
    cand_two_del = rep_800[:250] + rep_800[275:550] + rep_800[560:]
    assert check_similarity_merge(cand_two_del, rep_800) is True

    # 6. Two separate small insertions (20 bp at 250 and 15 bp at 550, total 35 bp < 50) with only 'M' between them:
    # ID_gc = 800 / 802 = 0.9975 >= 0.95 -> merges (True)
    cand_two_ins = (
        rep_800[:250]
        + make_complex_seq(20, seed=703)
        + rep_800[250:550]
        + make_complex_seq(15, seed=704)
        + rep_800[550:]
    )
    assert check_similarity_merge(cand_two_ins, rep_800) is True


def test_check_similarity_merge_edlib_criterion3_unaligned_ends_plus_internal_ins() -> None:
    rep = make_complex_seq(600, seed=710)

    # 1. 49 bp internal insertion (< 50) -> merges (True)
    cand_ins49 = rep[:300] + make_complex_seq(49, seed=711) + rep[300:]
    assert check_similarity_merge(cand_ins49, rep, min_sv_len=50) is True

    # 2. 50 bp internal insertion (== 50) -> fails Criterion 3 (False)
    cand_ins50 = rep[:300] + make_complex_seq(50, seed=712) + rep[300:]
    assert check_similarity_merge(cand_ins50, rep, min_sv_len=50) is False

    # 3. 24 bp left overhang + 25 bp internal insertion (total 49 < 50) -> merges (True)
    cand_24_25 = (
        make_complex_seq(24, seed=713)
        + rep[:300]
        + make_complex_seq(25, seed=714)
        + rep[300:]
    )
    assert check_similarity_merge(cand_24_25, rep, min_sv_len=50) is True

    # 4. 25 bp left overhang + 25 bp internal insertion (total 50 >= 50) -> fails Criterion 3 (False)
    cand_25_25 = (
        make_complex_seq(25, seed=715)
        + rep[:300]
        + make_complex_seq(25, seed=716)
        + rep[300:]
    )
    assert check_similarity_merge(cand_25_25, rep, min_sv_len=50) is False


def test_check_similarity_merge_edlib_criterion4_sliding_window_private_block() -> None:
    rep = make_complex_seq(800, seed=720)

    # 1. 16 mismatches inside a single 50 bp window [300, 350):
    # Overall ID_gc = 784/800 = 0.98 >= 0.95, but window [300, 350) has 34/50 = 0.68 < 0.70 -> False
    pos_16 = [300 + i * 3 for i in range(16)]
    cand_16 = mutate_seq_snps(rep, pos_16)
    assert check_similarity_merge(cand_16, rep, min_sv_len=50, cluster_win_id=0.70) is False

    # 2. 15 mismatches inside [300, 350):
    # Window [300, 350) has 35/50 = 0.70 >= 0.70 -> True
    pos_15 = [300 + i * 3 for i in range(15)]
    cand_15 = mutate_seq_snps(rep, pos_15)
    assert check_similarity_merge(cand_15, rep, min_sv_len=50, cluster_win_id=0.70) is True

    # 3. 50 bp random substitution block replacing rep[300:350] (T2 / C12) -> False
    cand_sub50 = rep[:300] + make_complex_seq(50, seed=721) + rep[350:]
    assert check_similarity_merge(cand_sub50, rep, min_sv_len=50, cluster_win_id=0.70) is False

    # 4. Length-mismatched 45 bp random substitution block replacing rep[300:350] -> False
    cand_sub45_50 = rep[:300] + make_complex_seq(45, seed=722) + rep[350:]
    assert check_similarity_merge(cand_sub45_50, rep, min_sv_len=50, cluster_win_id=0.70) is False

    # 5. Dispersed SNPs (1 per 50 bp) combined with a 15 bp small insertion and a 15 bp small deletion (T5) -> True
    rep_t5 = make_complex_seq(800, seed=901)
    ins15 = make_complex_seq(15, seed=902)
    snps_t5 = [i * 50 + 10 for i in range(15)]
    rep_snp = mutate_seq_snps(rep_t5, snps_t5)
    cand_t5 = rep_snp[:200] + ins15 + rep_snp[200:500] + rep_snp[515:]
    assert check_similarity_merge(cand_t5, rep_t5, min_sv_len=50, cluster_win_id=0.70) is True


def test_check_similarity_merge_mappy_10kb_dispersed_snps_vs_300bp_ins_and_sub() -> None:
    rep_10k = make_complex_seq(10000, seed=730)

    # 1. 98% dispersed SNPs (200 SNPs spaced every 50 bp -> 1 SNP per 50 bp window) -> merges (True)
    snps_200 = [i * 50 + 25 for i in range(200)]
    cand_98 = mutate_seq_snps(rep_10k, snps_200)
    assert check_similarity_merge(cand_98, rep_10k) is True

    # 2. T12 Part 1: 10 kb candidate with 300 bp private insertion -> fails Criterion 3 (False)
    cand_ins300 = rep_10k[:5000] + make_complex_seq(300, seed=731) + rep_10k[5000:]
    assert check_similarity_merge(cand_ins300, rep_10k) is False

    # 3. T12 Part 2: 10 kb candidate with 300 bp substitution block (chained as 1 hit or split) -> fails (False)
    # Sub-case A: 300 bp random substitution block
    cand_sub300_rand = rep_10k[:5000] + make_complex_seq(300, seed=732) + rep_10k[5300:]
    assert check_similarity_merge(cand_sub300_rand, rep_10k) is False

    # Sub-case B: 300 bp substitution block with ~40% mutations that mappy chains as a single alignment
    # (overall ID_gc = 9880/10000 = 0.988 >= 0.95, 0 insertions, caught specifically by Criterion 4!)
    mut_positions = [5000 + i for i in range(300) if (i % 5 in (0, 2))]
    cand_sub300_chained = mutate_seq_snps(rep_10k, mut_positions)
    assert check_similarity_merge(cand_sub300_chained, rep_10k) is False


def test_check_similarity_merge_rejects_reverse_complement_unaligned_and_engine_boundary() -> None:
    # 1. Reverse-complement rejected in both edlib (600 bp) and mappy (2000 bp)
    seq_600 = make_complex_seq(600, seed=740)
    assert check_similarity_merge(revcomp_dna(seq_600), seq_600) is False

    seq_2000 = make_complex_seq(2000, seed=741)
    assert check_similarity_merge(revcomp_dna(seq_2000), seq_2000) is False

    # 2. Unrelated sequences rejected in both engines
    other_600 = make_complex_seq(600, seed=742)
    assert check_similarity_merge(other_600, seq_600) is False

    other_2000 = make_complex_seq(2000, seed=743)
    assert check_similarity_merge(other_2000, seq_2000) is False

    # 3. Exact 1000 bp (edlib) vs 1001 bp (mappy) boundary
    seq_1000 = make_complex_seq(1000, seed=744)
    cand_1000 = mutate_seq_snps(seq_1000, [100, 300, 500, 700, 900])
    assert check_similarity_merge(cand_1000, seq_1000) is True

    seq_1001 = make_complex_seq(1001, seed=745)
    cand_1001 = mutate_seq_snps(seq_1001, [100, 300, 500, 700, 900])
    assert check_similarity_merge(cand_1001, seq_1001) is True

    # 4. Interval helper sanity check
    assert merge_intervals([(10, 20), (5, 10), (30, 30), (25, 35)]) == [
        (5, 20),
        (25, 35),
    ]
    assert complement_intervals(40, [(5, 20), (25, 35)]) == [
        (0, 5),
        (20, 25),
        (35, 40),
    ]
    assert subtract_intervals([(0, 100)], [(20, 40), (60, 80)]) == [
        (0, 20),
        (40, 60),
        (80, 100),
    ]


def test_pansn_multifasta_ref_and_bare_sample_backbone_prefix(tmp_path: Path) -> None:
    """Verify multi-FASTA --ref-fa and bare sample-name --backbone-prefix resolution."""
    from panflat.cli import main as cli_main

    s1 = make_complex_seq(120, seed=801)
    s2 = make_complex_seq(120, seed=802)
    s3 = make_complex_seq(120, seed=803)
    s4 = make_complex_seq(120, seed=804)

    gfa_path = tmp_path / "pansn_multi.gfa"
    gfa_path.write_text(
        "\n".join(
            [
                "H\tVN:Z:1.0",
                f"S\t1\t{s1}",
                f"S\t2\t{s2}",
                f"S\t3\t{s3}",
                f"S\t4\t{s4}",
                "L\t1\t+\t2\t+\t0M",
                "L\t2\t+\t4\t+\t0M",
                "L\t1\t+\t3\t+\t0M",
                "L\t3\t+\t4\t+\t0M",
                "P\tPyrar_xinyeli#1#Chr10A\t1+,2+,4+\t*",
                "P\tPyrar_xinyeli#2#Chr10B\t1+,3+,4+\t*",
                "P\tPyrbe_duli#1#Chr10A\t1+,3+,4+\t*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ref_fa_path = tmp_path / "pangenome_multi.ref.fa"
    ref_fa_path.write_text(
        "\n".join(
            [
                ">Pyrar_xinyeli#1#Chr10A",
                s1 + s2 + s4,
                ">Pyrar_xinyeli#2#Chr10B",
                s1 + s3 + s4,
                ">Pyrbe_duli#1#Chr10A",
                s1 + s3 + s4,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_prefix = tmp_path / "out_pansn"
    rc_build = cli_main(
        [
            "build",
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(ref_fa_path),
            "--backbone-prefix",
            "Pyrar_xinyeli",
            "--out-prefix",
            str(out_prefix),
        ]
    )
    assert rc_build == 0

    rc_verify = cli_main(
        [
            "verify",
            "--prefix",
            str(out_prefix),
            "--gfa",
            str(gfa_path),
            "--ref-fa",
            str(ref_fa_path),
        ]
    )
    assert rc_verify == 0


def test_lazy_step_and_provenance_detach_unpins_parent_buffers() -> None:
    """Verify detach()/detached() copy only sliced sub-arrays and preserve exact provenance coordinates."""
    import numpy as np
    from panflat.models import (
        CompactVisitCounts,
        DeviationSegment,
        LazyProvenanceTuple,
        LazyStepTuple,
    )

    codes = np.array([1, 2, -3, 4, 5], dtype=np.int32)
    node_lens = np.array([0, 10, 20, 30, 40, 50], dtype=np.int32)
    parent_steps = LazyStepTuple(codes)
    parent_prov = LazyProvenanceTuple(
        src_hap="Alt#1",
        src_path="Alt#1#Chr01",
        base_offset=100,
        orig_codes=codes,
        node_lens=node_lens,
        total_bp=150,
    )

    # Forward contiguous slice [1:4]
    sl_steps = parent_steps[1:4]
    sl_prov = parent_prov[1:4]
    assert sl_steps._codes.base is not None

    dev = DeviationSegment(
        hap_id="Alt#1",
        src_path="Alt#1#Chr01",
        left_anchor_rank=0,
        right_anchor_rank=4,
        branch_start_idx=1,
        branch_end_idx=4,
        branch_steps=sl_steps,
        branch_provenance=sl_prov,
        backbone_steps=parent_steps[1:4],
    )
    det_dev = dev.detached(keep_provenance=True)
    assert isinstance(det_dev.branch_steps, LazyStepTuple)
    assert det_dev.branch_steps._codes.base is None
    assert list(det_dev.branch_steps) == list(sl_steps)
    assert list(det_dev.branch_provenance) == list(sl_prov)
    assert det_dev.backbone_steps == ()

    # Detach with keep_provenance=False drops provenance
    det_no_prov = dev.detached(keep_provenance=False)
    assert det_no_prov.branch_provenance == ()

    # Empty slice detach returns shared empty LazyStepTuple singleton
    empty_sl = parent_steps[2:2].detach()
    assert len(empty_sl) == 0
    assert empty_sl._codes.base is None

    # Flipped contig slice detach preserves exact reversed provenance
    flipped_prov = parent_prov.flipped_contig()
    flipped_sl = flipped_prov[1:4]
    det_flipped = flipped_sl.detach()
    assert list(det_flipped) == list(flipped_sl)

    # CompactVisitCounts O(1) membership and get()
    cvc = CompactVisitCounts(
        multi_nids=np.array([2, 5], dtype=np.int32),
        multi_counts=np.array([3, 7], dtype=np.int32),
    )
    assert cvc.get("1", 0) == 0
    assert cvc.get("2", 0) == 3
    assert cvc.get("5", 0) == 7
    assert cvc.get("non_int", 0) == 0

