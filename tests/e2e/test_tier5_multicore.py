"""Tier 5 strictly opaque-box CLI E2E tests for true multicore execution and T22 byte-identity.

Enforces:
- ZERO internal library imports (strictly subprocess CLI execution via `helpers.py`).
- Pure compact integer node IDs (`1..N`) to activate `LazyStepTuple`, `LazyProvenanceTuple`,
  shared split-arm `_orig_codes`, `>64`-step `INV_EDGE` novel slices, and `CompactVisitCounts`.
- Exact SHA-256 byte-identity across all 12 build output files between `--threads 1` and `--threads 4`.
- Zero Python 3.12 multi-threaded `fork` `DeprecationWarning` on `stderr` under `PYTHONWARNINGS=default`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import subprocess

from tests.e2e.helpers import (
    InputPaths,
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
    run_verify,
)

_ALL_12_OUTPUT_SUFFIXES: tuple[str, ...] = (
    ".pan.fa",
    ".offsets.tsv",
    ".ref_to_pan.chain",
    ".sv.tsv",
    ".genotypes.tsv",
    ".inserts.tsv",
    ".ref.lifted.gff3",
    ".centromere.tsv",
    ".unresolved.tsv",
    ".unresolved.fa",
    ".summary.tsv",
    ".log",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_12_files_sha256_identical_and_clean_streams(
    prefix_a: Path,
    prefix_b: Path,
    proc_a: subprocess.CompletedProcess[str],
    proc_b: subprocess.CompletedProcess[str],
) -> None:
    """Assert all 12 build outputs exist and are SHA-256 identical, stdout=='', and zero fork warnings."""
    assert proc_a.stdout == "", f"Unexpected stdout in run A:\n{proc_a.stdout}"
    assert proc_b.stdout == "", f"Unexpected stdout in run B:\n{proc_b.stdout}"
    for label, stderr in (("run A", proc_a.stderr), ("run B", proc_b.stderr)):
        assert "DeprecationWarning" not in stderr, (
            f"DeprecationWarning detected on stderr in {label}:\n{stderr}"
        )
        assert "multi-threaded, use of fork()" not in stderr, (
            f"Multi-threaded fork warning detected on stderr in {label}:\n{stderr}"
        )

    for suffix in _ALL_12_OUTPUT_SUFFIXES:
        path_a = Path(f"{prefix_a}{suffix}")
        path_b = Path(f"{prefix_b}{suffix}")
        assert path_a.is_file(), f"Missing expected output file in run A: {path_a}"
        assert path_b.is_file(), f"Missing expected output file in run B: {path_b}"
        hash_a = _sha256_file(path_a)
        hash_b = _sha256_file(path_b)
        assert hash_a == hash_b, (
            f"T22 SHA-256 mismatch on '{suffix}': {hash_a} ({path_a}) != {hash_b} ({path_b})"
        )


def _fast_random_dna(length: int, rng: random.Random) -> str:
    """Fast high-complexity ACGT sequence generator (>99.9% unique 15-mers) for large graphs."""
    return "".join(rng.choices("ACGT", k=length))


def _reverse_steps(steps: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return exact reverse-complement step sequence for a whole-contig reversed walk."""
    return [(nid, "-" if orient == "+" else "+") for nid, orient in reversed(steps)]


# ---------------------------------------------------------------------------
# Test 1a & 1b: Adversarial Forced-Fork Tests (`PANFLAT_FORCE_FORK=1`)
# ---------------------------------------------------------------------------


def _build_adversarial_compact_int_graph(tmp_path: Path) -> InputPaths:
    """Construct an adversarial pure-compact-integer-ID graph covering all SV/unresolved classes."""
    b = SyntheticInputBuilder(
        tmp_path / "adv_inputs",
        chrom="Chr01",
        backbone_prefix="Ref#1#",
    )

    next_id = 1
    nid_of: dict[str, str] = {}

    def _alloc(tag: str, seq: str) -> str:
        nonlocal next_id
        assert tag not in nid_of
        nid = str(next_id)
        next_id += 1
        nid_of[tag] = nid
        b.add_node(nid, seq)
        return nid

    # Reference unique nodes 1..38 (all pure compact integers!), with valid coding sequences
    # placed in nodes 21..23 (+ strand gene) and 28..30 (- strand gene) for --ref-gff liftover.
    plus_cds = make_coding_seq(num_codons=120, seed=9101)  # 360 bp: exon1=180 + exon2=180
    minus_cds = make_coding_seq(num_codons=120, seed=9102)  # 360 bp on '-' strand
    minus_rc = revcomp(minus_cds)  # 360 bp on genomic '+' strand

    for i in range(1, 39):
        if i == 19:
            # Tandem repeat unit (80 bp, appears x2 in ref -> x4 in H1 for CNV)
            _alloc(f"r{i}", make_complex_seq(80, seed=9200 + i))
        elif i == 21:
            # 20 bp UTR/flank + first 180 bp of plus_cds = 200 bp
            _alloc(f"r{i}", make_complex_seq(20, seed=9221) + plus_cds[:180])
        elif i == 22:
            # Intron / spacer node (200 bp)
            _alloc(f"r{i}", make_complex_seq(200, seed=9222))
        elif i == 23:
            # Second 180 bp of plus_cds + 20 bp flank = 200 bp
            _alloc(f"r{i}", plus_cds[180:] + make_complex_seq(20, seed=9223))
        elif i == 25 or i == 26:
            # Centromere reference nodes (1,000 bp each = 2,000 bp centromere)
            _alloc(f"r{i}", make_complex_seq(1000, seed=9200 + i))
        elif i == 28:
            # 20 bp flank + first 180 bp of minus_rc (genomic coords of exon2 on '-' strand)
            _alloc(f"r{i}", make_complex_seq(20, seed=9228) + minus_rc[:180])
        elif i == 29:
            # Intron node between '-' strand exons (200 bp; hosts DUP_TRA + INS slot)
            _alloc(f"r{i}", make_complex_seq(200, seed=9229))
        elif i == 30:
            # Second 180 bp of minus_rc + 20 bp flank = 200 bp
            _alloc(f"r{i}", minus_rc[180:] + make_complex_seq(20, seed=9230))
        else:
            _alloc(f"r{i}", make_complex_seq(200, seed=9200 + i))

    # 70-node (>64 steps!) contiguous novel chain inside H2's INV_EDGE to force
    # LazyProvenanceTuple inside ReverseChainRecord.novel_unresolved_slices across Stage 2.4 IPC.
    inv_edge_chain_ids = [
        _alloc(f"ie_{k}", make_complex_seq(10, seed=9300 + k)) for k in range(70)
    ]

    # INV_CONFLICT novel nodes
    xc1_id = _alloc("xc1", make_complex_seq(160, seed=9401))
    xc2_id = _alloc("xc2", make_complex_seq(170, seed=9402))

    # Multi-branch recursive clustering locus nodes between r21 and r22
    hub_a = _alloc("hub_a", make_complex_seq(180, seed=9410))
    hub_b = _alloc("hub_b", make_complex_seq(210, seed=9411))
    hub_shared = _alloc("hub_shared", make_complex_seq(260, seed=9412))
    arm1_seq = make_complex_seq(260, seed=9413)
    arm1 = _alloc("arm1", arm1_seq)
    # 5 SNPs / 260 bp (~1.9% divergence) -> merges with arm1 in clustering
    arm2 = _alloc("arm2", mutate_seq_snps(arm1_seq, num_snps=5, seed=9414))
    arm3 = _alloc("arm3", make_complex_seq(280, seed=9415))

    # Repeat-guard hybrid split remnant (500 complex + 1500 sat + 500 complex = 2500 bp)
    # plus pure satellite block (1500 bp -> REPEAT_LIKE) between r22 and r23
    hybrid_rep_seq = (
        make_complex_seq(500, seed=9420)
        + make_satellite_seq(unit_len=20, total_len=1500, seed=9421)
        + make_complex_seq(500, seed=9422)
    )
    hybrid_rep = _alloc("hybrid_rep", hybrid_rep_seq)
    pure_sat = _alloc(
        "pure_sat", make_satellite_seq(unit_len=20, total_len=1500, seed=9423)
    )

    # Centromere alt alleles between u=r24 and v=r27
    cen_isl_1_seq = make_complex_seq(1200, seed=9430)
    cen_isl_1 = _alloc("cen_isl_1", cen_isl_1_seq)
    cen_isl_1_var = _alloc(
        "cen_isl_1_var", mutate_seq_snps(cen_isl_1_seq, num_snps=18, seed=9431)
    )
    cen_sat = _alloc(
        "cen_sat", make_satellite_seq(unit_len=25, total_len=1500, seed=9432)
    )
    cen_ref1_seq = b.nodes[nid_of["r25"]]
    cen_ref2_seq = b.nodes[nid_of["r26"]]
    cen_step_b = _alloc(
        "cen_step_b",
        cen_ref1_seq[100:600]
        + make_complex_seq(700, seed=9433)
        + revcomp(cen_ref2_seq[100:600]),
    )
    cen_isl_flip = _alloc("cen_isl_flip", make_complex_seq(650, seed=9434))

    # Right-arm SV novel nodes
    dup_nov = _alloc("dup_nov", make_complex_seq(220, seed=9440))
    lj_nov = _alloc("lj_nov", make_complex_seq(240, seed=9441))
    unanch_l = _alloc("unanch_l", make_complex_seq(160, seed=9442))
    unanch_r = _alloc("unanch_r", make_complex_seq(160, seed=9443))
    flip_right_ins = _alloc("flip_right_ins", make_complex_seq(260, seed=9444))
    # Homologous ~94%-identity copy of contiguous ref nodes r16 + r17 (400 bp) for --qc Stage 1
    qc_hom_ins = _alloc(
        "qc_hom_ins",
        mutate_seq_snps(
            b.nodes[nid_of["r16"]] + b.nodes[nid_of["r17"]], num_snps=24, seed=9445
        ),
    )

    def _s(tag: str, orient: str = "+") -> tuple[str, str]:
        return (nid_of[tag], orient)

    def _range(lo: int, hi: int) -> list[tuple[str, str]]:
        return [_s(f"r{k}") for k in range(lo, hi + 1)]

    # Reference path: r1..r18, r19, r19, r20..r24, r25, r26, r27..r38
    ref_steps = [
        *_range(1, 18),
        _s("r19"),
        _s("r19"),
        *_range(20, 38),
    ]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # Compute exact 0-based ref coordinates per node occurrence for BED and GFF3
    ref_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for nid, _ in ref_steps:
        seq_len = len(b.nodes[nid])
        if nid not in ref_offsets:
            ref_offsets[nid] = (cursor, cursor + seq_len)
        cursor += seq_len

    # Centromere BED: exact span of r25..r26 (anchors u=r24, v=r27)
    cen_start = ref_offsets[nid_of["r25"]][0]
    cen_end = ref_offsets[nid_of["r26"]][1]
    b.add_bed_region(cen_start, cen_end, region_id="CEN_ADV")

    # GFF3 Gene 1 (+ strand): exon1/CDS1 in r21 [start+20 .. end], exon2/CDS2 in r23 [start .. start+180]
    r21_s, r21_e = ref_offsets[nid_of["r21"]]
    r23_s, _ = ref_offsets[nid_of["r23"]]
    b.add_gff_gene(
        gene_id="gene_plus",
        mrna_id="mrna_plus",
        strand="+",
        exons=[(r21_s + 21, r21_e), (r23_s + 1, r23_s + 180)],
        cds_list=[(r21_s + 21, r21_e, 0), (r23_s + 1, r23_s + 180, 0)],
    )

    # GFF3 Gene 2 (- strand): exon1/CDS1 in r30 [start .. start+180], exon2/CDS2 in r28 [start+20 .. end]
    r28_s, r28_e = ref_offsets[nid_of["r28"]]
    r30_s, _ = ref_offsets[nid_of["r30"]]
    b.add_gff_gene(
        gene_id="gene_minus",
        mrna_id="mrna_minus",
        strand="-",
        exons=[(r28_s + 21, r28_e), (r30_s + 1, r30_s + 180)],
        cds_list=[(r28_s + 21, r28_e, 0), (r30_s + 1, r30_s + 180, 0)],
    )

    # H1: pure INV (r11, r12), DEL (r15), CNV (r19 x 4), repeat-guard hybrid remnant (r22..r23),
    #     CEN_ISLAND (>1 kb), and homologous insert at (r33..r34)
    b.add_path(
        "H1#1#Chr01",
        [
            *_range(1, 10),
            _s("r12", "-"),
            _s("r11", "-"),
            _s("r13"),
            _s("r14"),
            # r15 deleted
            _s("r16"),
            _s("r17"),
            _s("r18"),
            _s("r19"),
            _s("r19"),
            _s("r19"),
            _s("r19"),
            _s("r20"),
            _s("r21"),
            _s("r22"),
            (hybrid_rep, "+"),
            _s("r23"),
            _s("r24"),
            (cen_isl_1, "+"),
            *_range(27, 33),
            (qc_hom_ins, "+"),
            *_range(34, 38),
        ],
    )

    # H2: INV_EDGE at left end with 70-step (>64) novel chain, clustered CEN_ISLAND variant,
    #     and DUP_TRA (visits r37+ between r28+ and r29+, > max_jump=500) + novel ins
    b.add_path(
        "H2#1#Chr01",
        [
            _s("r2", "-"),
            *[(nid, "+") for nid in inv_edge_chain_ids],
            _s("r1", "-"),
            *_range(3, 18),
            _s("r19"),
            _s("r19"),
            *_range(20, 24),
            (cen_isl_1_var, "+"),
            _s("r27"),
            _s("r28"),
            (dup_nov, "+"),
            _s("r37"),
            *_range(29, 38),
        ],
    )

    # H3: INV_CONFLICT (overlapping R1=[r7-, xc1+, r5-] and R2=[r6-, xc2+, r4-]),
    #     CEN_REPEAT_LIKE (1500 bp satellite), and LARGE_JUMP (r30 -> lj_nov -> r35, skipping 800 bp > 500)
    b.add_path(
        "H3#1#Chr01",
        [
            *_range(1, 4),
            _s("r7", "-"),
            (xc1_id, "+"),
            _s("r5", "-"),
            _s("r8"),
            _s("r6", "-"),
            (xc2_id, "+"),
            _s("r4", "-"),
            *_range(9, 18),
            _s("r19"),
            _s("r19"),
            *_range(20, 24),
            (cen_sat, "+"),
            *_range(27, 30),
            (lj_nov, "+"),
            *_range(35, 38),
        ],
    )

    # H4: UNANCHORED_END (leading + trailing overhang) + multi-branch recursive locus (arm1)
    b.add_path(
        "H4#1#Chr01",
        [
            (unanch_l, "+"),
            *_range(1, 18),
            _s("r19"),
            _s("r19"),
            _s("r20"),
            _s("r21"),
            (hub_a, "+"),
            (hub_shared, "+"),
            (arm1, "+"),
            *_range(22, 38),
            (unanch_r, "+"),
        ],
    )

    # H5: multi-branch recursive locus (arm2, merges with arm1), pure satellite REPEAT_LIKE,
    #     and Step B centromere reference subtraction + island
    b.add_path(
        "H5#1#Chr01",
        [
            *_range(1, 18),
            _s("r19"),
            _s("r19"),
            _s("r20"),
            _s("r21"),
            (hub_a, "+"),
            (hub_shared, "+"),
            (arm2, "+"),
            _s("r22"),
            (pure_sat, "+"),
            _s("r23"),
            _s("r24"),
            (cen_step_b, "+"),
            *_range(27, 38),
        ],
    )

    # H6: multi-branch recursive locus (divergent hub_b + arm3) + W-line with non-zero start
    h6_steps = [
        *_range(1, 18),
        _s("r19"),
        _s("r19"),
        _s("r20"),
        _s("r21"),
        (hub_b, "+"),
        (hub_shared, "+"),
        (arm3, "+"),
        *_range(22, 38),
    ]
    b.add_walk("H6", "1", "ctgH6", start=300, end=None, steps=h6_steps)

    # H7: WHOLE-CONTIG REVERSED W-line with non-zero start spanning the centromere
    #     (exercises flipped split arms sharing `_orig_codes` with `is_contig_flipped=True`)
    h7_fwd_steps = [
        *_range(1, 18),
        _s("r19"),
        _s("r19"),
        *_range(20, 24),
        (cen_isl_flip, "+"),
        *_range(27, 35),
        (flip_right_ins, "+"),
        *_range(36, 38),
    ]
    b.add_walk(
        "H7", "1", "ctgH7_rev", start=500, end=None, steps=_reverse_steps(h7_fwd_steps)
    )

    return b.write("adv_multicore")


def test_tier5_multicore_1a_adversarial_forced_fork_extract_islands(
    tmp_path: Path,
) -> None:
    """Case 1a: PANFLAT_FORCE_FORK=1, extract-islands, --dump-repeat-sequences, --ref-gff, --qc."""
    inputs = _build_adversarial_compact_int_graph(tmp_path)
    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    common_args = [
        "--centromere-mode",
        "extract-islands",
        "--centromere-min-island",
        "500",
        "--dump-repeat-sequences",
        "--qc",
        "--max-jump",
        "500",
    ]
    out_t1 = tmp_path / "adv_1a_t1"
    out_t4 = tmp_path / "adv_1a_t4"

    proc_t1, bun_t1 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "1"],
        out_prefix=out_t1,
        extra_env=env,
    )
    proc_t4, bun_t4 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "4"],
        out_prefix=out_t4,
        extra_env=env,
    )
    assert bun_t1 is not None and bun_t4 is not None

    _assert_12_files_sha256_identical_and_clean_streams(out_t1, out_t4, proc_t1, proc_t4)

    # Independent CLI `panflat verify` on both outputs
    for prefix in (out_t1, out_t4):
        run_verify(
            prefix,
            inputs.gfa_path,
            inputs.ref_fa_path,
            extra_args=["--centromere-bed", inputs.bed_path],
            check=True,
        )

    # Independent global, centromere, and GFF3 oracles
    assert_all_global_invariants(bun_t4, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_t4)
    assert inputs.gff_path is not None
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bun_t4.pan_fa_path, bun_t4.lifted_gff_path
    )

    # Non-vacuous check: ALL required SV classes and unresolved reasons are genuinely present
    sv_types = {r["type"] for r in bun_t4.sv_rows}
    expected_sv_types = {
        "INV",
        "INV_EDGE",
        "INV_CONFLICT",
        "CNV",
        "DEL",
        "DUP_TRA",
        "LARGE_JUMP",
        "INS",
        "CEN_ISLAND",
    }
    assert expected_sv_types.issubset(sv_types), (
        f"Missing expected SV types: {expected_sv_types - sv_types}; got {sv_types}"
    )

    unres_reasons = {r["reason"] for r in bun_t4.unresolved}
    expected_unres_reasons = {
        "UNANCHORED_END",
        "REPEAT_LIKE",
        "INV_EDGE",
        "INV_CONFLICT",
        "LARGE_JUMP",
        "CEN_REPEAT_LIKE",
    }
    assert expected_unres_reasons.issubset(unres_reasons), (
        f"Missing expected unresolved reasons: {expected_unres_reasons - unres_reasons}; "
        f"got {unres_reasons}"
    )

    # Confirm the 70-step (>64) INV_EDGE novel slice (700 bp) actually reached unresolved.tsv
    inv_edge_unres = [r for r in bun_t4.unresolved if r["reason"] == "INV_EDGE"]
    assert any(int(r["length"]) == 700 for r in inv_edge_unres), (
        f"Expected 700 bp (70-step) INV_EDGE unresolved slice, got {inv_edge_unres}"
    )

    # Confirm whole-contig reversed W-line H7 split arms produced valid '-' strand inserts
    # (both centromere island 650 bp and right-arm insertion 260 bp)
    h7_inserts = [
        r for r in bun_t4.inserts if str(r["src_path"]).startswith("H7#1#ctgH7_rev")
    ]
    h7_lens = sorted(int(r["pan_end"]) - int(r["pan_start"]) for r in h7_inserts)
    assert h7_lens == [260, 650], f"Expected H7 flipped split-arm inserts [260, 650], got {h7_lens}"
    assert all(str(r["src_strand"]) == "-" for r in h7_inserts)

    # Confirm repeat-guard hybrid split remnants (two 500 bp pieces from H1) & --dump-repeat-sequences export
    h1_remnants = [
        r
        for r in bun_t4.inserts
        if str(r["src_path"]) == "H1#1#Chr01"
        and (int(r["pan_end"]) - int(r["pan_start"])) == 500
    ]
    assert len(h1_remnants) == 2, f"Expected 2 x 500 bp hybrid split remnants from H1, got {len(h1_remnants)}"
    for row in bun_t4.unresolved:
        if row["reason"] in ("REPEAT_LIKE", "CEN_REPEAT_LIKE"):
            assert row["exported_fasta_id"] != "."

    # Confirm Step B centromere reference subtraction occurred on H5
    assert bun_t4.centromere_rows is not None
    h5_cen = next(r for r in bun_t4.centromere_rows if r["hap"] == "H5#1")
    assert int(h5_cen["reference_like_bp"]) == 1000
    assert int(h5_cen["accepted_source_bp"]) == 700

    # Confirm multi-branch recursive locus called H4/H5/H6 as INS carriers and --qc detected redundancy
    rec_ins_events = [
        ev
        for ev in bun_t4.sv_rows
        if ev["type"] == "INS"
        and bun_t4.genotypes[ev["event_id"]].get("H4#1") not in ("0", ".")
        and bun_t4.genotypes[ev["event_id"]].get("H5#1") not in ("0", ".")
        and bun_t4.genotypes[ev["event_id"]].get("H6#1") not in ("0", ".")
    ]
    assert len(rec_ins_events) >= 1
    assert int(bun_t4.summary_kv["qc_long_ge150_count"]) > 0
    assert int(bun_t4.summary_kv["qc_redundancy_0.90_bp"]) > 0

    # Confirm gfa_io, fork stages, and --qc stderr progress milestones appeared
    for needle in (
        "GFA S/L parsed",
        "GFA P/W parsed",
        "Reference FASTA validated",
        "extract-islands (fork, workers=",
        "Normalizing & LIS contigs (fork, workers=",
        "--qc Stage 1 (ref alignment):",
        "--qc Stage 2 round ",
    ):
        assert needle in proc_t4.stderr, f"Missing expected progress log {needle!r} in stderr"
    assert "(fork, workers=" not in proc_t1.stderr


def test_tier5_multicore_1b_adversarial_forced_fork_bypass(
    tmp_path: Path,
) -> None:
    """Case 1b: PANFLAT_FORCE_FORK=1, bypass mode (no --dump-repeat-sequences), --ref-gff, --qc."""
    inputs = _build_adversarial_compact_int_graph(tmp_path)
    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}
    common_args = [
        "--centromere-mode",
        "bypass",
        "--qc",
        "--max-jump",
        "500",
    ]
    out_t1 = tmp_path / "adv_1b_t1"
    out_t4 = tmp_path / "adv_1b_t4"

    proc_t1, bun_t1 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "1"],
        out_prefix=out_t1,
        extra_env=env,
    )
    proc_t4, bun_t4 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "4"],
        out_prefix=out_t4,
        extra_env=env,
    )
    assert bun_t1 is not None and bun_t4 is not None

    _assert_12_files_sha256_identical_and_clean_streams(out_t1, out_t4, proc_t1, proc_t4)

    for prefix in (out_t1, out_t4):
        run_verify(
            prefix,
            inputs.gfa_path,
            inputs.ref_fa_path,
            extra_args=["--centromere-bed", inputs.bed_path],
            check=True,
        )

    assert_all_global_invariants(bun_t4, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_t4)
    assert inputs.gff_path is not None
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bun_t4.pan_fa_path, bun_t4.lifted_gff_path
    )

    sv_types = {r["type"] for r in bun_t4.sv_rows}
    expected_sv_types = {
        "INV",
        "INV_EDGE",
        "INV_CONFLICT",
        "CNV",
        "DEL",
        "DUP_TRA",
        "LARGE_JUMP",
        "INS",
    }
    assert expected_sv_types.issubset(sv_types), (
        f"Missing expected SV types: {expected_sv_types - sv_types}; got {sv_types}"
    )
    assert "CEN_ISLAND" not in sv_types

    unres_reasons = {r["reason"] for r in bun_t4.unresolved}
    expected_unres_reasons = {
        "UNANCHORED_END",
        "REPEAT_LIKE",
        "INV_EDGE",
        "INV_CONFLICT",
        "LARGE_JUMP",
        "CEN_BYPASS",
    }
    assert expected_unres_reasons.issubset(unres_reasons), (
        f"Missing expected unresolved reasons: {expected_unres_reasons - unres_reasons}; "
        f"got {unres_reasons}"
    )

    # Confirm H7 right split-arm insertion still has exact '-' strand provenance under bypass mode
    h7_inserts = [
        r for r in bun_t4.inserts if str(r["src_path"]).startswith("H7#1#ctgH7_rev")
    ]
    assert [int(r["pan_end"]) - int(r["pan_start"]) for r in h7_inserts] == [260]
    assert h7_inserts[0]["src_strand"] == "-"

    # Without --dump-repeat-sequences, REPEAT_LIKE and CEN_BYPASS must NOT be exported to unresolved.fa
    for row in bun_t4.unresolved:
        if row["reason"] in ("REPEAT_LIKE", "CEN_BYPASS"):
            assert row["exported_fasta_id"] == "."

    assert int(bun_t4.summary_kv["qc_long_ge150_count"]) > 0
    assert int(bun_t4.summary_kv["qc_redundancy_0.90_bp"]) > 0

    for needle in (
        "GFA S/L parsed",
        "GFA P/W parsed",
        "Reference FASTA validated",
        "Normalizing & LIS contigs (fork, workers=",
        "--qc Stage 1 (ref alignment):",
        "--qc Stage 2 round ",
    ):
        assert needle in proc_t4.stderr, f"Missing expected progress log {needle!r} in stderr"
    assert "(fork, workers=" not in proc_t1.stderr


# ---------------------------------------------------------------------------
# Test 2: Natural Real-Threshold Test (NO `PANFLAT_FORCE_FORK`, explicitly unset)
# ---------------------------------------------------------------------------


def _build_natural_threshold_graph(tmp_path: Path) -> InputPaths:
    """Build a pure compact-int graph exceeding all natural parallel thresholds:
    - `ref_steps = 5,100 >= 5,000` (naturally triggers Stage 2.4 fork across 8 split-arm fragments)
    - `40 >= 32` non-empty `terminal_groups` (naturally triggers Stage 2.5 fork)
    - `11,000 bp >= 10,000 bp` centromere with 4 complete alt haps (naturally triggers Stage 2.3b fork)
    - `--qc` with >=150 bp and >1,100 bp homologous pieces + `--ref-gff`
    """
    rng = random.Random(20261123)
    b = SyntheticInputBuilder(
        tmp_path / "nat_inputs",
        chrom="Chr01",
        backbone_prefix="Ref#1#",
    )

    n_ref = 5100
    node_len = 50  # 5,100 * 50 = 255,000 bp reference

    # Build valid coding sequences for two multi-exon genes (+ on left arm, - on right arm)
    # 60 codons = 180 bp split into two 90 bp exons spanning two 50 bp nodes each
    plus_cds = make_coding_seq(num_codons=60, seed=9601)  # 180 bp
    minus_cds = make_coding_seq(num_codons=60, seed=9602)  # 180 bp
    minus_rc = revcomp(minus_cds)  # 180 bp

    # Allocate reference nodes 1..5100
    for i in range(1, n_ref + 1):
        if i == 11:
            seq = _fast_random_dna(5, rng) + plus_cds[:45]
        elif i == 12:
            seq = plus_cds[45:90] + _fast_random_dna(5, rng)
        elif i == 15:
            seq = _fast_random_dna(5, rng) + plus_cds[90:135]
        elif i == 16:
            seq = plus_cds[135:180] + _fast_random_dna(5, rng)
        elif i == 2711:
            seq = _fast_random_dna(5, rng) + minus_rc[:45]
        elif i == 2712:
            seq = minus_rc[45:90] + _fast_random_dna(5, rng)
        elif i == 2715:
            seq = _fast_random_dna(5, rng) + minus_rc[90:135]
        elif i == 2716:
            seq = minus_rc[135:180] + _fast_random_dna(5, rng)
        else:
            seq = _fast_random_dna(node_len, rng)
        b.add_node(str(i), seq)

    next_nid = n_ref + 1

    def _new_node(seq: str) -> str:
        nonlocal next_nid
        nid = str(next_nid)
        next_nid += 1
        b.add_node(nid, seq)
        return nid

    # Reference path: 1+, 2+, ..., 5100+
    ref_steps = [(str(i), "+") for i in range(1, n_ref + 1)]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # Centromere BED: ref steps 2400..2619 (nodes 2401..2620) -> 220 * 50 = 11,000 bp (>= 10 kb)
    cen_first_idx = 2400  # 0-based ref step index (anchor u = step 2399 = node 2400)
    cen_last_idx = 2619   # 0-based ref step index (anchor v = step 2620 = node 2621)
    b.add_bed_region(
        cen_first_idx * node_len,
        (cen_last_idx + 1) * node_len,
        region_id="CEN_NAT",
    )

    # GFF3 genes spanning insertion loci (locus 0 sits between step 12 and 13 inside plus gene intron;
    # locus 20 sits between step 2712 and 2713 inside minus gene intron!)
    # Step i (0-based) covers 0-based [i*50, (i+1)*50), i.e. 1-based [i*50 + 1, (i+1)*50].
    # Node 11 is step 10 (500..550), Node 12 is step 11 (550..600) -> exon1 = 1-based [506, 595] (90 bp)
    # Node 15 is step 14 (700..750), Node 16 is step 15 (750..800) -> exon2 = 1-based [706, 795] (90 bp)
    b.add_gff_gene(
        gene_id="nat_gene_plus",
        mrna_id="nat_mrna_plus",
        strand="+",
        exons=[(506, 595), (706, 795)],
        cds_list=[(506, 595, 0), (706, 795, 0)],
    )
    # Node 2711 is step 2710 (135500..135550), Node 2712 is step 2711 (135550..135600) -> [135506, 135595]
    # Node 2715 is step 2714 (135700..135750), Node 2716 is step 2715 (135750..135800) -> [135706, 135795]
    b.add_gff_gene(
        gene_id="nat_gene_minus",
        mrna_id="nat_mrna_minus",
        strand="-",
        exons=[(135506, 135595), (135706, 135795)],
        cds_list=[(135506, 135595, 0), (135706, 135795, 0)],
    )

    # Plan 40 distinct non-empty terminal groups across left arm (20) and right arm (20),
    # well clear of centromere [2400..2619] and inversion/deletion loci:
    # Left loci anchor steps: 12 (intron of plus gene), then 100, 200, ..., 1900 (20 loci)
    # Right loci anchor steps: 2712 (intron of minus gene), then 2800, 2900, ..., 4600 (20 loci)
    left_ins_steps = [12] + [100 * k for k in range(1, 20)]
    right_ins_steps = [2712] + [2800 + 100 * k for k in range(19)]
    all_ins_steps = left_ins_steps + right_ins_steps
    assert len(all_ins_steps) == 40

    # Per-hap modifications indexed by ref step:
    # inserts_by_hap[h][step_idx] = list of (nid, "+") inserted right AFTER step_idx
    inserts_by_hap: list[dict[int, list[tuple[str, str]]]] = [{}, {}, {}, {}]

    # Shared homologous sequence pool so --qc Stage 2 dyadic halving finds real >=90% hits
    hom_base_300 = _fast_random_dna(300, rng)
    hom_base_1200 = _fast_random_dna(1200, rng)

    for loc_idx, st_idx in enumerate(all_ins_steps):
        if loc_idx == 5:
            # Repeat-guard pure satellite (1500 bp)
            sat_nid = _new_node(make_satellite_seq(unit_len=20, total_len=1500, seed=9701))
            inserts_by_hap[0][st_idx] = [(sat_nid, "+")]
            inserts_by_hap[1][st_idx] = [(sat_nid, "+")]
        elif loc_idx == 25:
            # Repeat-guard hybrid block (500 complex + 1500 sat + 500 complex = 2500 bp)
            hyb_seq = (
                _fast_random_dna(500, rng)
                + make_satellite_seq(unit_len=20, total_len=1500, seed=9702)
                + _fast_random_dna(500, rng)
            )
            hyb_nid = _new_node(hyb_seq)
            inserts_by_hap[2][st_idx] = [(hyb_nid, "+")]
            inserts_by_hap[3][st_idx] = [(hyb_nid, "+")]
        elif loc_idx % 8 == 0:
            # >1,100 bp homologous inserts across haplotypes (triggers >1 kb clustering + --qc)
            seq_a = mutate_seq_snps(hom_base_1200, num_snps=12 + loc_idx, seed=9710 + loc_idx)
            seq_b = mutate_seq_snps(seq_a, num_snps=20, seed=9750 + loc_idx)
            nid_a = _new_node(seq_a)
            nid_b = _new_node(seq_b)
            inserts_by_hap[loc_idx % 4][st_idx] = [(nid_a, "+")]
            inserts_by_hap[(loc_idx + 1) % 4][st_idx] = [(nid_b, "+")]
        elif loc_idx % 4 == 1:
            # ~92%-identity medium inserts (300 bp -> aligner_short + dyadic halving)
            seq_m = mutate_seq_snps(hom_base_300, num_snps=20 + (loc_idx % 5), seed=9800 + loc_idx)
            nid_m = _new_node(seq_m)
            inserts_by_hap[loc_idx % 4][st_idx] = [(nid_m, "+")]
            inserts_by_hap[(loc_idx + 2) % 4][st_idx] = [(nid_m, "+")]
        else:
            # Novel 180-360 bp complex insertions
            ins_len = 180 + (loc_idx % 7) * 30
            nid_n = _new_node(_fast_random_dna(ins_len, rng))
            inserts_by_hap[loc_idx % 4][st_idx] = [(nid_n, "+")]
            if loc_idx % 3 == 0:
                nid_n2 = _new_node(_fast_random_dna(ins_len + 40, rng))
                inserts_by_hap[(loc_idx + 1) % 4][st_idx] = [(nid_n2, "+")]

    # Centromere alt alleles for the 4 haplotypes (H1/H2 >1kb clustered island, H3 sat, H4 Step B)
    cen_isl_base = _fast_random_dna(1200, rng)
    cen_h1 = _new_node(cen_isl_base)
    cen_h2 = _new_node(mutate_seq_snps(cen_isl_base, num_snps=18, seed=9891))
    cen_h3 = _new_node(make_satellite_seq(unit_len=25, total_len=1500, seed=9892))
    ref_cen_sample = "".join(b.nodes[str(k)] for k in range(2411, 2423))  # 600 bp ref homology
    cen_h4 = _new_node(ref_cen_sample + _fast_random_dna(700, rng))
    cen_node_by_hap = [cen_h1, cen_h2, cen_h3, cen_h4]

    # 2 inversions (H1 at steps 2150..2155, H2 at steps 4850..4855)
    # 2 deletions (H3 deletes steps 2250..2252 = 150 bp, H4 deletes steps 4950..4952 = 150 bp)
    inv_spans = {0: (2150, 2156), 1: (4850, 4856)}
    del_spans = {2: (2250, 2253), 3: (4950, 4953)}

    for h_idx in range(4):
        steps: list[tuple[str, str]] = []
        i = 0
        while i < n_ref:
            if i == cen_first_idx:
                steps.append((cen_node_by_hap[h_idx], "+"))
                i = cen_last_idx + 1
                continue
            if h_idx in inv_spans and i == inv_spans[h_idx][0]:
                lo, hi = inv_spans[h_idx]
                for k in range(hi - 1, lo - 1, -1):
                    steps.append((str(k + 1), "-"))
                i = hi
                continue
            if h_idx in del_spans and i == del_spans[h_idx][0]:
                i = del_spans[h_idx][1]
                continue
            steps.append((str(i + 1), "+"))
            if i in inserts_by_hap[h_idx]:
                steps.extend(inserts_by_hap[h_idx][i])
            i += 1
        b.add_path(f"H{h_idx + 1}#1#Chr01", steps)

    return b.write("nat_threshold")


def test_tier5_multicore_2_natural_real_threshold_no_force_fork(
    tmp_path: Path,
) -> None:
    """Test 2: Natural real-threshold triggers (PANFLAT_FORCE_FORK explicitly unset)."""
    inputs = _build_natural_threshold_graph(tmp_path)
    # Explicitly unset PANFLAT_FORCE_FORK and PANFLAT_MAX_FORK_WORKERS to verify natural triggers
    env: dict[str, str | None] = {
        "PANFLAT_FORCE_FORK": None,
        "PANFLAT_MAX_FORK_WORKERS": None,
        "PYTHONWARNINGS": "default",
    }
    common_args = [
        "--centromere-mode",
        "extract-islands",
        "--centromere-min-island",
        "500",
        "--qc",
    ]
    out_t1 = tmp_path / "nat_t1"
    out_t4 = tmp_path / "nat_t4"

    proc_t1, bun_t1 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "1"],
        out_prefix=out_t1,
        extra_env=env,
    )
    proc_t4, bun_t4 = run_build(
        inputs,
        extra_args=[*common_args, "--threads", "4"],
        out_prefix=out_t4,
        extra_env=env,
    )
    assert bun_t1 is not None and bun_t4 is not None

    _assert_12_files_sha256_identical_and_clean_streams(out_t1, out_t4, proc_t1, proc_t4)

    for prefix in (out_t1, out_t4):
        run_verify(
            prefix,
            inputs.gfa_path,
            inputs.ref_fa_path,
            extra_args=["--centromere-bed", inputs.bed_path],
            check=True,
        )

    assert_all_global_invariants(bun_t4, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_t4)
    assert inputs.gff_path is not None
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bun_t4.pan_fa_path, bun_t4.lifted_gff_path
    )

    # Confirm all 4 parallel stages naturally engaged under --threads 4 WITHOUT PANFLAT_FORCE_FORK!
    for needle in (
        "GFA S/L parsed",
        "GFA P/W parsed",
        "Reference FASTA validated",
        "extract-islands (fork, workers=4)",
        "Normalizing & LIS contigs (fork, workers=4)",
        "Screening & solving parallel groups (fork, workers=4)",
        "--qc Stage 1 (ref alignment):",
        "threads=4)",
        "--qc Stage 2 round ",
    ):
        assert needle in proc_t4.stderr, (
            f"Missing natural parallel execution log {needle!r} in --threads 4 stderr:\n"
            f"{proc_t4.stderr}"
        )

    # Confirm serial run (--threads 1) did NOT fork
    assert "(fork, workers=" not in proc_t1.stderr

    # Confirm non-vacuous SVs and QC metrics
    sv_types = {r["type"] for r in bun_t4.sv_rows}
    assert {"INS", "INV", "DEL", "CEN_ISLAND"}.issubset(sv_types)
    assert int(bun_t4.summary_kv["qc_long_ge150_count"]) >= 35
    assert int(bun_t4.summary_kv["qc_redundancy_0.90_bp"]) > 0
