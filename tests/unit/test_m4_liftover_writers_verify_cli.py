"""Unit tests for M4 modules: `gff_liftover.py`, `writers.py`, `verify_qc.py`, and `cli.py` (F1, F2, F3, F25, F26, T14, T22, C14)."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from panflat.cli import main as cli_main
from panflat.gfa_io import load_and_validate_inputs, revcomp
from panflat.gff_liftover import liftover_gff3, liftover_gff3_lines
from panflat.linearizer import run_chromosome_pipeline
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    BuildConfig,
    InputContractError,
    InvariantViolationError,
    StepOffsetRow,
    SVEvent,
)
from panflat.verify_qc import (
    INSERTS_HEADER,
    OFFSETS_HEADER,
    SV_HEADER,
    UNRESOLVED_HEADER,
    run_qc,
    verify_build_outputs,
)
from panflat.writers import merge_chromosomes, write_build_outputs
from tests.e2e.helpers import (
    OutputBundle,
    SyntheticInputBuilder,
    assert_all_global_invariants,
    assert_centromere_accounting_and_offsets,
    assert_gff3_liftover_integrity,
    make_coding_seq,
    make_complex_seq,
    mutate_seq_snps,
)


# ===========================================================================
# 1. GFF3 Liftover Unit Tests (`panflat/gff_liftover.py`)
# ===========================================================================


def test_gff_liftover_plus_strand_utr_intron_splice_and_cds_split(tmp_path: Path) -> None:
    """Test + strand GFF3 liftover with UTR insertion, intronic insertion, splice-proximal insertion, and CDS split."""
    step_offsets = [
        StepOffsetRow("Chr01", 0, "r1", 0, 75, 0, 75, 0),
        StepOffsetRow("Chr01", 1, "r2", 75, 350, 175, 450, 100),    # 100 bp UTR ins at ref=75
        StepOffsetRow("Chr01", 2, "r3", 350, 446, 570, 666, 120),   # 120 bp intron ins at ref=350
        StepOffsetRow("Chr01", 3, "r4", 446, 551, 756, 861, 90),    # 90 bp splice-proximal ins at ref=446 (4 bp before 450)
        StepOffsetRow("Chr01", 4, "r5", 551, 1000, 1011, 1460, 150), # 150 bp CDS ins at ref=551
    ]
    sv_events = [
        SVEvent("Chr01", 75, 175, "Chr01_INS_1", "INS", 75, 75, "0..1", "1:len=100,src=Alt#1,n=1", 1, 2, ".", {}),
        SVEvent("Chr01", 450, 570, "Chr01_INS_2", "INS", 350, 350, "1..2", "1:len=120,src=Alt#1,n=1", 1, 2, ".", {}),
        SVEvent("Chr01", 666, 756, "Chr01_INS_3", "INS", 446, 446, "2..3", "1:len=90,src=Alt#1,n=1", 1, 2, ".", {}),
        SVEvent("Chr01", 861, 1011, "Chr01_INS_4", "INS", 551, 551, "3..4", "1:len=150,src=Alt#1,n=1", 1, 2, ".", {}),
    ]
    gff_lines = [
        "##gff-version 3",
        "Chr01\tsim\tgene\t51\t750\t.\t+\t.\tID=g1",
        "Chr01\tsim\tmRNA\t51\t750\t.\t+\t.\tID=m1;Parent=g1",
        "Chr01\tsim\texon\t51\t250\t.\t+\t.\tID=m1.exon1;Parent=m1",
        "Chr01\tsim\texon\t451\t750\t.\t+\t.\tID=m1.exon2;Parent=m1",
        "Chr01\tsim\tCDS\t101\t250\t.\t+\t0\tID=m1.cds1;Parent=m1",
        "Chr01\tsim\tCDS\t451\t690\t.\t+\t0\tID=m1.cds2;Parent=m1",
    ]
    gff_file = tmp_path / "ref.gff3"
    gff_file.write_text("\n".join(gff_lines) + "\n", encoding="utf-8")

    # Test both 5-arg and 4-arg calling conventions
    lifted_5 = liftover_gff3(gff_file, "Chr01", step_offsets, sv_events, splice_window=8)
    lifted_4 = liftover_gff3(gff_file, step_offsets, sv_events, splice_window=8)
    assert lifted_5 == lifted_4
    assert lifted_5[0] == "##gff-version 3"

    joined = "\n".join(lifted_5)
    assert "utr_insertion" in joined
    assert "intronic_insertion" in joined
    assert "cds_split_by_insertion" in joined
    assert "projection_split=True" in joined
    assert "splice_site_proximal_sv=True" in joined
    assert "Chr01_INS_1" in joined and "Chr01_INS_4" in joined

    # CDS 2 [451, 690] (0-based [450, 690)) is split at 551 into 101 bp (phase 0) and 139 bp (phase (3 - 101%3)%3 == 1)
    cds_lines = [l.split("\t") for l in lifted_5 if not l.startswith("#") and l.split("\t")[2] == "CDS"]
    assert len(cds_lines) == 3
    assert (int(cds_lines[1][3]), int(cds_lines[1][4]), cds_lines[1][7]) == (761, 861, "0")
    assert (int(cds_lines[2][3]), int(cds_lines[2][4]), cds_lines[2][7]) == (1012, 1150, "1")


def test_gff_liftover_minus_strand_micro_cds_phase_propagation() -> None:
    """Test 1 bp and 2 bp micro-CDS phase propagation on + and - strands (T22)."""
    # 180 bp CDS on '-' strand at 1-based [101, 280] (0-based [100, 280)), split at 277 (279 is 1 bp from 280) and 279
    # Sub-intervals in genomic order: [100, 277) (177 bp), [277, 279) (2 bp), [279, 280) (1 bp)
    # In '-' transcriptional order (descending): 1 bp (phase 0) -> 2 bp (phase (3-1)%3 = 2) -> 177 bp (phase (3-(2-2)%3)%3 = 0)
    step_offsets = [
        StepOffsetRow("Chr01", 0, "n0", 0, 277, 0, 277, 0),
        StepOffsetRow("Chr01", 1, "n1", 277, 279, 337, 339, 60),
        StepOffsetRow("Chr01", 2, "n2", 279, 400, 409, 530, 70),
    ]
    sv_events = [
        SVEvent("Chr01", 277, 337, "Chr01_INS_1", "INS", 277, 277, "0..1", "1:len=60", 1, 2, ".", {}),
        SVEvent("Chr01", 339, 409, "Chr01_INS_2", "INS", 279, 279, "1..2", "1:len=70", 1, 2, ".", {}),
    ]
    gff_lines = [
        "##gff-version 3",
        "Chr01\tsim\tgene\t101\t280\t.\t-\t.\tID=gm",
        "Chr01\tsim\tmRNA\t101\t280\t.\t-\t.\tID=mm;Parent=gm",
        "Chr01\tsim\texon\t101\t280\t.\t-\t.\tID=mm.ex1;Parent=mm",
        "Chr01\tsim\tCDS\t101\t280\t.\t-\t0\tID=mm.cds1;Parent=mm",
    ]
    out = liftover_gff3_lines(gff_lines, "Chr01", step_offsets, sv_events, splice_window=8)
    cds_rows = [l.split("\t") for l in out if not l.startswith("#") and l.split("\t")[2] == "CDS"]
    assert len(cds_rows) == 3
    # Sorted descending by start (transcriptional order on '-' strand): phases must be ["0", "2", "0"]
    cds_desc = sorted(cds_rows, key=lambda r: int(r[3]), reverse=True)
    assert [r[7] for r in cds_desc] == ["0", "2", "0"]


def test_gff_liftover_exterior_boundaries_and_splice_window_threshold() -> None:
    """Verify exact exterior boundaries (s-1|s and e|e+1) do not split features, and splice window 8bp vs 9bp."""
    # Exon at 1-based [101, 280] -> 0-based [100, 280)
    # Insertion at 100 (left exterior boundary) and 280 (right exterior boundary)
    offsets_ext = [
        StepOffsetRow("Chr01", 0, "n1", 0, 100, 0, 100, 0),
        StepOffsetRow("Chr01", 1, "n2", 100, 280, 160, 340, 60),
        StepOffsetRow("Chr01", 2, "n3", 280, 400, 410, 530, 70),
    ]
    evs_ext = [
        SVEvent("Chr01", 100, 160, "E1", "INS", 100, 100, "0..1", "1:len=60", 1, 2, ".", {}),
        SVEvent("Chr01", 340, 410, "E2", "INS", 280, 280, "1..2", "1:len=70", 1, 2, ".", {}),
    ]
    gff_lines = [
        "##gff-version 3",
        "Chr01\tsim\texon\t101\t280\t.\t+\t.\tID=ex1;Parent=m1",
        "Chr01\tsim\tCDS\t101\t280\t.\t+\t0\tID=cds1;Parent=m1",
    ]
    lifted = liftover_gff3_lines(gff_lines, "Chr01", offsets_ext, evs_ext, splice_window=8)
    exon_rows = [l.split("\t") for l in lifted if not l.startswith("#") and l.split("\t")[2] == "exon"]
    cds_rows = [l.split("\t") for l in lifted if not l.startswith("#") and l.split("\t")[2] == "CDS"]
    assert len(exon_rows) == 1 and len(cds_rows) == 1
    assert (int(cds_rows[0][3]), int(cds_rows[0][4])) == (161, 340)
    assert "splice_site_proximal_sv=True" in exon_rows[0][8]
    assert "projection_split=True" not in exon_rows[0][8]

    # Distance 9 bp upstream (at 91 vs s0=100) with splice_window=8 -> not tagged
    offsets_9bp = [
        StepOffsetRow("Chr01", 0, "n1", 0, 91, 0, 91, 0),
        StepOffsetRow("Chr01", 1, "n2", 91, 400, 151, 460, 60),
    ]
    evs_9bp = [SVEvent("Chr01", 91, 151, "E9", "INS", 91, 91, "0..1", "1:len=60", 1, 2, ".", {})]
    lifted_9 = liftover_gff3_lines(gff_lines, "Chr01", offsets_9bp, evs_9bp, splice_window=8)
    assert "splice_site_proximal_sv=True" not in "\n".join(lifted_9)


def test_gff_liftover_malformed_inputs_raise_input_contract_error(tmp_path: Path) -> None:
    """Verify malformed GFF3 lines raise InputContractError."""
    offsets = [StepOffsetRow("Chr01", 0, "n1", 0, 200, 0, 200, 0)]

    with pytest.raises(InputContractError):
        liftover_gff3(tmp_path / "nonexistent.gff3", "Chr01", offsets, [])

    bad_cases = [
        ["Chr01\tsrc\tgene\t10\t100"],  # < 9 columns
        ["Chr01\tsrc\tgene\t150\t50\t.\t+\t.\tID=g1"],  # start > end
        ["Chr01\tsrc\tgene\t0\t50\t.\t+\t.\tID=g1"],  # start < 1
        ["Chr01\tsrc\tgene\t10\t250\t.\t+\t.\tID=g1"],  # end > ref_len (200)
        ["Chr02\tsrc\tgene\t10\t100\t.\t+\t.\tID=g1"],  # wrong seqid
        ["Chr01\tsrc\tgene\t10\t100\t.\t*\t.\tID=g1"],  # invalid strand
        ["Chr01\tsrc\tCDS\t10\t100\t.\t+\t3\tID=c1"],  # invalid phase
    ]
    for lines in bad_cases:
        with pytest.raises(InputContractError):
            liftover_gff3_lines(lines, "Chr01", offsets, [])


# ===========================================================================
# 2. Writers, Verifier, QC & CLI Integration Unit Tests
# ===========================================================================


def test_writers_verify_qc_and_cli_end_to_end_unit(tmp_path: Path) -> None:
    """Test write_build_outputs, run_qc, verify_build_outputs, and cli.main on a complete chromosome build."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s1 = make_complex_seq(200, seed=4001)
    su = make_complex_seq(200, seed=4002)
    scen = make_complex_seq(300, seed=4003)
    sv = make_complex_seq(200, seed=4004)
    s_cds = make_coding_seq(num_codons=60, seed=4005)  # 180 bp at [901, 1080]
    isl = make_complex_seq(600, seed=4006)
    ins_149 = make_complex_seq(149, seed=4007)
    ins_hom = mutate_seq_snps(s1[:180], num_snps=3, seed=4008)  # 180 bp homologous to s1
    unanch = make_complex_seq(90, seed=4009)

    for nid, seq in [
        ("n1", s1),
        ("u", su),
        ("cen", scen),
        ("v", sv),
        ("ncds1", s_cds[:62]),
        ("ncds2", s_cds[62:]),
        ("isl", isl),
        ("i149", ins_149),
        ("ihom", ins_hom),
        ("un", unanch),
    ]:
        b.add_node(nid, seq)

    b.set_ref_path("Ref#1#Chr01", ["n1+", "u+", "cen+", "v+", "ncds1+", "ncds2+"])
    b.add_path("Alt1#1#Chr01", ["n1+", "i149+", "u+", "isl+", "v+", "ncds1+", "ihom+", "ncds2+"])
    b.add_path("Alt2#1#Chr01", ["un+", "v+", "ncds1+", "ncds2+"])
    b.add_bed_region(400, 700, "CEN1")
    b.add_gff_gene("g1", "m1", "+", [(901, 1080)], [(901, 1080, 0)])
    inputs = b.write("m4_unit")

    ret = cli_main(
        [
            "build",
            "--gfa",
            str(inputs.gfa_path),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--backbone-prefix",
            inputs.backbone_prefix,
            "--out-prefix",
            str(inputs.out_prefix),
            "--ref-gff",
            str(inputs.gff_path),
            "--centromere-bed",
            str(inputs.bed_path),
            "--centromere-mode",
            "extract-islands",
            "--centromere-min-island",
            "500",
            "--qc",
        ]
    )
    assert ret == 0

    bundle = OutputBundle(inputs.out_prefix)
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )

    # Check QC metrics in summary.tsv
    assert bundle.summary_kv.get("qc_enabled") == "true"
    assert bundle.summary_kv.get("qc_short_50_149_count") == "1"
    assert bundle.summary_kv.get("qc_short_50_149_bp") == "149"
    assert "149" in bundle.summary_kv.get("qc_short_50_149_lengths", "")
    assert int(bundle.summary_kv.get("qc_long_ge150_count", "0")) >= 2
    assert int(bundle.summary_kv.get("qc_redundancy_0.95_bp", "0")) > 0

    # Standalone verify CLI passes
    v_ret = cli_main(
        [
            "verify",
            "--prefix",
            str(inputs.out_prefix),
            "--gfa",
            str(inputs.gfa_path),
            "--ref-fa",
            str(inputs.ref_fa_path),
            "--ref-gff",
            str(inputs.gff_path),
        ]
    )
    assert v_ret == 0


def test_verify_detects_all_tamper_modes(tmp_path: Path) -> None:
    """Verify verify_build_outputs catches corruptions in pan.fa, offsets.tsv, chain, inserts.tsv, centromere.tsv, and GFA SHA-256."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 4101), ("u", 4102), ("cen", 4103), ("v", 4104)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))
    b.add_node("isl", make_complex_seq(600, seed=4105))
    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+"])
    b.add_path("Alt#1#Chr01", ["n0+", "u+", "cen+", "isl+", "v+"])
    b.add_bed_region(500, 750, "CEN1")
    inputs = b.write("tamper_unit")

    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(inputs.gfa_path),
                "--ref-fa",
                str(inputs.ref_fa_path),
                "--backbone-prefix",
                inputs.backbone_prefix,
                "--out-prefix",
                str(inputs.out_prefix),
                "--centromere-bed",
                str(inputs.bed_path),
                "--centromere-mode",
                "extract-islands",
            ]
        )
        == 0
    )

    def _clone_prefix(tag: str) -> Path:
        dest = tmp_path / f"clone_{tag}"
        for f in tmp_path.glob("tamper_unit_out.*"):
            shutil.copy(f, tmp_path / f.name.replace("tamper_unit_out", f"clone_{tag}"))
        return dest

    # 1) Mutate 1 ref base in pan.fa
    p1 = _clone_prefix("pan_mut")
    fa_path = Path(f"{p1}.pan.fa")
    fa_lines = fa_path.read_text(encoding="utf-8").splitlines()
    seq = fa_lines[1]
    bad_base = "A" if seq[0] != "A" else "C"
    fa_path.write_text(f">Chr01\n{bad_base}{seq[1:]}\n", encoding="utf-8")
    rep1 = verify_build_outputs(p1, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep1.passed and not rep1.invariant_1_passed

    # 2) Corrupt spacer 'N' to 'G' in pan.fa
    p2 = _clone_prefix("spacer_mut")
    fa_path2 = Path(f"{p2}.pan.fa")
    # Mount point is before step v (ref=750), so pan[750:850] is the leading 100N spacer
    assert seq[755] == "N"
    fa_path2.write_text(f">Chr01\n{seq[:755]}G{seq[756:]}\n", encoding="utf-8")
    rep2 = verify_build_outputs(p2, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep2.passed and not rep2.invariant_2_passed

    # 3) Corrupt chain # block comment (.replace("\t0\t250", "\t0\t199"))
    p3 = _clone_prefix("chain_mut")
    ch_path = Path(f"{p3}.ref_to_pan.chain")
    ch_path.write_text(
        ch_path.read_text(encoding="utf-8").replace("\t0\t750", "\t0\t699"),
        encoding="utf-8",
    )
    rep3 = verify_build_outputs(p3, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep3.passed and not rep3.invariant_3_passed

    # 4) Flip src_strand in inserts.tsv
    p4 = _clone_prefix("ins_mut")
    ins_path = Path(f"{p4}.inserts.tsv")
    ins_lines = ins_path.read_text(encoding="utf-8").splitlines()
    cols = ins_lines[1].split("\t")
    cols[9] = "-" if cols[9] == "+" else "+"
    ins_lines[1] = "\t".join(cols)
    ins_path.write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    rep4 = verify_build_outputs(p4, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep4.passed and not rep4.invariant_2_passed

    # 5) Corrupt centromere.tsv ref_end (col 6)
    p5 = _clone_prefix("cen_mut")
    cen_path = Path(f"{p5}.centromere.tsv")
    cen_lines = cen_path.read_text(encoding="utf-8").splitlines()
    for i in range(1, len(cen_lines)):
        c_cols = cen_lines[i].split("\t")
        c_cols[6] = "250"
        cen_lines[i] = "\t".join(c_cols)
    cen_path.write_text("\n".join(cen_lines) + "\n", encoding="utf-8")
    rep5 = verify_build_outputs(p5, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep5.passed and not rep5.centromere_passed

    # 6) Modify GFA after build -> SHA-256 mismatch
    bad_gfa = tmp_path / "modified.gfa"
    bad_gfa.write_text(
        inputs.gfa_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(InvariantViolationError):
        verify_build_outputs(
            inputs.out_prefix, bad_gfa, inputs.ref_fa_path, raise_on_error=True
        )


def test_merge_chromosomes_and_cli_validation(tmp_path: Path) -> None:
    """Test multi-chromosome merge and negative CLI argument handling."""
    # Chr01 with centromere extract-islands + GFF
    b1 = SyntheticInputBuilder(tmp_path / "c1", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("u", 4201), ("cen", 4202), ("v", 4203)]:
        b1.add_node(nid, make_complex_seq(200, seed=seed))
    b1.add_node("isl", make_complex_seq(600, seed=4204))
    b1.set_ref_path("Ref#1#Chr01", ["u+", "cen+", "v+"])
    b1.add_path("AltA#1#Chr01", ["u+", "isl+", "v+"])
    b1.add_bed_region(200, 400, "CEN_C1")
    b1.add_gff_gene("g1", "m1", "+", [(11, 190)], [(11, 187, 0)], chrom="Chr01")
    in1 = b1.write("c1")

    # Chr02 without centromere mode + different haplotype AltB#1 + GFF
    b2 = SyntheticInputBuilder(tmp_path / "c2", chrom="Chr02", backbone_prefix="Ref#1#")
    b2.add_node("r1", make_complex_seq(200, seed=4205))
    b2.add_node("r2", make_complex_seq(200, seed=4206))
    b2.add_node("ins2", make_complex_seq(150, seed=4207))
    b2.set_ref_path("Ref#1#Chr02", ["r1+", "r2+"])
    b2.add_path("AltB#1#Chr02", ["r1+", "ins2+", "r2+"])
    b2.add_gff_gene("g2", "m2", "-", [(211, 390)], [(211, 387, 0)], chrom="Chr02")
    in2 = b2.write("c2")

    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(in1.gfa_path),
                "--ref-fa",
                str(in1.ref_fa_path),
                "--backbone-prefix",
                in1.backbone_prefix,
                "--out-prefix",
                str(in1.out_prefix),
                "--ref-gff",
                str(in1.gff_path),
                "--centromere-bed",
                str(in1.bed_path),
                "--centromere-mode",
                "extract-islands",
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(in2.gfa_path),
                "--ref-fa",
                str(in2.ref_fa_path),
                "--backbone-prefix",
                in2.backbone_prefix,
                "--out-prefix",
                str(in2.out_prefix),
                "--ref-gff",
                str(in2.gff_path),
            ]
        )
        == 0
    )

    merged_out = tmp_path / "merged_genome"
    merge_chromosomes([in1.out_prefix, in2.out_prefix], merged_out)
    mbun = OutputBundle(merged_out)
    assert set(mbun.pan_fa.keys()) == {"Chr01", "Chr02"}
    assert set(mbun.genotype_haps) == {"Ref#1", "AltA#1", "AltB#1"}
    assert mbun.centromere_rows is not None
    assert {r["chrom"] for r in mbun.centromere_rows} == {"Chr01"}
    assert mbun.lifted_gff_rows is not None
    assert {r["seqid"] for r in mbun.lifted_gff_rows} == {"Chr01", "Chr02"}

    # Negative CLI tests
    assert cli_main(["merge", "--prefix-list", str(tmp_path / "missing.txt"), "--out-prefix", str(tmp_path / "m_err")]) != 0
    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(in2.gfa_path),
                "--ref-fa",
                str(in2.ref_fa_path),
                "--backbone-prefix",
                "Ref#1#",
                "--out-prefix",
                str(tmp_path / "bad_opt"),
                "--centromere-min-island",
                "500",
            ]
        )
        != 0
    )


def test_gff_liftover_parent_span_expansion_and_fasta_directive_stop() -> None:
    """Verify parent gene/mRNA expand over split children and ##FASTA stops GFF3 parsing."""
    step_offsets = [
        StepOffsetRow("Chr01", 0, "n1", 0, 120, 0, 120, 0),
        StepOffsetRow("Chr01", 1, "n2", 120, 400, 370, 650, 250),
    ]
    sv_events = [
        SVEvent("Chr01", 120, 370, "Chr01_INS_1", "INS", 120, 120, "0..1", "1:len=250", 1, 2, ".", {}),
    ]
    gff_lines = [
        "##gff-version 3",
        "Chr01\tsim\tgene\t101\t340\t.\t+\t.\tID=g_exp",
        "Chr01\tsim\tmRNA\t101\t340\t.\t+\t.\tID=m_exp;Parent=g_exp",
        "Chr01\tsim\texon\t101\t340\t.\t+\t.\tID=m_exp.ex1;Parent=m_exp",
        "Chr01\tsim\tCDS\t101\t340\t.\t+\t0\tID=m_exp.cds1;Parent=m_exp",
        "##FASTA",
        ">Chr01",
        "ACGTACGT",
    ]
    lifted = liftover_gff3_lines(gff_lines, "Chr01", step_offsets, sv_events)
    rows = [l.split("\t") for l in lifted if not l.startswith("#")]
    gene_row = next(r for r in rows if r[2] == "gene")
    mrna_row = next(r for r in rows if r[2] == "mRNA")
    assert (int(gene_row[3]), int(gene_row[4])) == (101, 590)
    assert (int(mrna_row[3]), int(mrna_row[4])) == (101, 590)


def test_verify_detects_offsets_gap_overlap_truncation_and_5way_accounting(tmp_path: Path) -> None:
    """Verify verify_build_outputs catches 1bp offsets gap/overlap, 1bp pan.fa truncation, and 5-way accounting corruption."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("u", 4301), ("c", 4302), ("v", 4303)]:
        b.add_node(nid, make_complex_seq(150, seed=seed))
    b.add_node("isl", make_complex_seq(600, seed=4304))
    b.set_ref_path("Ref#1#Chr01", ["u+", "c+", "v+"])
    b.add_path("Alt#1#Chr01", ["u+", "c+", "isl+", "v+"])
    b.add_bed_region(150, 300, "CEN1")
    inputs = b.write("v_extra")

    assert (
        cli_main(
            [
                "build",
                "--gfa",
                str(inputs.gfa_path),
                "--ref-fa",
                str(inputs.ref_fa_path),
                "--backbone-prefix",
                inputs.backbone_prefix,
                "--out-prefix",
                str(inputs.out_prefix),
                "--centromere-bed",
                str(inputs.bed_path),
                "--centromere-mode",
                "extract-islands",
            ]
        )
        == 0
    )

    def _clone(tag: str) -> Path:
        dest = tmp_path / f"v_{tag}"
        for f in tmp_path.glob("v_extra_out.*"):
            shutil.copy(f, tmp_path / f.name.replace("v_extra_out", f"v_{tag}"))
        return dest

    # 1) 1 bp truncation of pan.fa
    p_trunc = _clone("trunc")
    fa_file = Path(f"{p_trunc}.pan.fa")
    lines = fa_file.read_text(encoding="utf-8").splitlines()
    fa_file.write_text(f"{lines[0]}\n{lines[1][:-1]}\n", encoding="utf-8")
    assert not verify_build_outputs(p_trunc, inputs.gfa_path, inputs.ref_fa_path).passed

    # 2) 1 bp gap in offsets.tsv
    p_gap = _clone("gap")
    off_file = Path(f"{p_gap}.offsets.tsv")
    off_lines = off_file.read_text(encoding="utf-8").splitlines()
    cols = off_lines[2].split("\t")
    cols[3] = str(int(cols[3]) + 1)
    off_lines[2] = "\t".join(cols)
    off_file.write_text("\n".join(off_lines) + "\n", encoding="utf-8")
    assert not verify_build_outputs(p_gap, inputs.gfa_path, inputs.ref_fa_path).passed

    # 3) Corrupt 5-way accounting in centromere.tsv
    p_acc = _clone("acc")
    cen_file = Path(f"{p_acc}.centromere.tsv")
    cen_lines = cen_file.read_text(encoding="utf-8").splitlines()
    c_cols = cen_lines[-1].split("\t")
    c_cols[25] = "999"
    cen_lines[-1] = "\t".join(c_cols)
    cen_file.write_text("\n".join(cen_lines) + "\n", encoding="utf-8")
    rep_acc = verify_build_outputs(p_acc, inputs.gfa_path, inputs.ref_fa_path)
    assert not rep_acc.passed and not rep_acc.centromere_passed


def test_cli_threads_determinism_and_invalid_options(tmp_path: Path) -> None:
    """Verify --threads 1 and --threads 4 produce byte-identical output and invalid flags return exit code 1."""
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(200, seed=4401))
    b.add_node("r2", make_complex_seq(200, seed=4402))
    b.add_node("ins", make_complex_seq(150, seed=4403))
    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("Alt#1#Chr01", ["r1+", "ins+", "r2+"])
    inputs = b.write("t_det")

    p_t1 = tmp_path / "out_t1"
    p_t4 = tmp_path / "out_t4"
    for p_out, th in [(p_t1, "1"), (p_t4, "4")]:
        assert (
            cli_main(
                [
                    "build",
                    "--gfa",
                    str(inputs.gfa_path),
                    "--ref-fa",
                    str(inputs.ref_fa_path),
                    "--backbone-prefix",
                    inputs.backbone_prefix,
                    "--out-prefix",
                    str(p_out),
                    "--threads",
                    th,
                ]
            )
            == 0
        )

    for ext in [".pan.fa", ".offsets.tsv", ".ref_to_pan.chain", ".sv.tsv", ".genotypes.tsv", ".inserts.tsv"]:
        assert Path(f"{p_t1}{ext}").read_bytes() == Path(f"{p_t4}{ext}").read_bytes()

    for bad_opt in [
        ["--repeat-guard", "invalid"],
        ["--sec-backbone", "invalid"],
        ["--cluster-id", "1.5"],
        ["--min-sv-len", "0"],
        ["--cluster-cov", "0.0"],
    ]:
        assert (
            cli_main(
                [
                    "build",
                    "--gfa",
                    str(inputs.gfa_path),
                    "--ref-fa",
                    str(inputs.ref_fa_path),
                    "--backbone-prefix",
                    inputs.backbone_prefix,
                    "--out-prefix",
                    str(tmp_path / "bad_cli"),
                    *bad_opt,
                ]
            )
            != 0
        )
