"""Tier 5 strictly opaque-box CLI E2E regression tests for all 5 Review-Fix items.

Enforces:
- ZERO internal `panflat` imports (strictly subprocess CLI execution via `tests.e2e.helpers`).
- All 5 counterexamples from REVIEW_FIX_PLAN.md §4.1.
- Negative tamper tests verifying `panflat verify` exits non-zero with `[INV2]` and zero tracebacks.
- 1,200-layer nested re-anchoring CLI build+verify and T22 `--threads 1` vs `--threads 4` SHA-256 byte-identity.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil

from tests.e2e.helpers import (
    SyntheticInputBuilder,
    assert_all_global_invariants,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
    revcomp,
    run_build,
    run_verify,
)

_CORE_OUTPUT_SUFFIXES: tuple[str, ...] = (
    ".pan.fa",
    ".offsets.tsv",
    ".ref_to_pan.chain",
    ".sv.tsv",
    ".genotypes.tsv",
    ".inserts.tsv",
    ".unresolved.tsv",
    ".unresolved.fa",
    ".summary.tsv",
    ".log",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tamper_inserts_join_before(
    src_prefix: Path,
    dst_prefix: Path,
    row_idx: int,
    new_join_before: str,
) -> None:
    for f in src_prefix.parent.glob(f"{src_prefix.name}.*"):
        shutil.copy(f, dst_prefix.parent / f.name.replace(src_prefix.name, dst_prefix.name))
    ins_path = Path(f"{dst_prefix}.inserts.tsv")
    with ins_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[row_idx]["join_before"] = new_join_before
    with ins_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_tier5_rf01_cnv_scattered_nodes_and_flanking_del_coexistence_cli(
    tmp_path: Path,
) -> None:
    """Counterexample 1 (CLI): A-U-X-U-X-U-B emits 0 CNV (only DEL+INS); A-U-U-U-B preserves CNV (k_ref=2, k_alt=3) + DEL."""
    # Graph 1: Ref = A-U-U-D-B, Alt = A-U-X-U-X-U-B (no CNV)
    b1 = SyntheticInputBuilder(tmp_path / "rf01_a", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("1", 6101), ("2", 6102), ("3", 6103), ("4", 6104), ("5", 6105)]:
        b1.add_node(nid, make_complex_seq(100, seed=seed))
    b1.set_ref_path("Ref#1#Chr01", ["1+", "2+", "2+", "3+", "4+"])
    b1.add_path("Alt1#1#Chr01", ["1+", "2+", "5+", "2+", "5+", "2+", "4+"])
    inp1 = b1.write("rf01_scattered")

    proc1, bun1 = run_build(inp1)
    assert proc1.returncode == 0 and bun1 is not None
    run_verify(inp1.out_prefix, inp1.gfa_path, inp1.ref_fa_path, check=True)
    assert_all_global_invariants(bun1, inp1.gfa_path, inp1.ref_fa_path)
    types1 = [r["type"] for r in bun1.sv_rows]
    assert "CNV" not in types1
    assert "DEL" in types1
    assert "INS" in types1

    # Graph 2: Ref = A-D-U-U-B, Alt2 = A-U-U-U-B (CNV k_ref=2,k_alt=3 + DEL) & Alt3 = A-U-U-U-X-B (CNV + DEL + INS)
    b2 = SyntheticInputBuilder(tmp_path / "rf01_b", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("1", 6111), ("2", 6112), ("3", 6113), ("4", 6114), ("5", 6115)]:
        b2.add_node(nid, make_complex_seq(100, seed=seed))
    b2.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "3+", "4+"])
    b2.add_path("Alt2#1#Chr01", ["1+", "3+", "3+", "3+", "4+"])
    b2.add_path("Alt3#1#Chr01", ["1+", "3+", "3+", "3+", "5+", "4+"])
    inp2 = b2.write("rf01_coexist")

    proc2, bun2 = run_build(inp2)
    assert proc2.returncode == 0 and bun2 is not None
    run_verify(inp2.out_prefix, inp2.gfa_path, inp2.ref_fa_path, check=True)
    assert_all_global_invariants(bun2, inp2.gfa_path, inp2.ref_fa_path)
    cnv_rows = [r for r in bun2.sv_rows if r["type"] == "CNV"]
    del_rows = [r for r in bun2.sv_rows if r["type"] == "DEL"]
    ins_rows = [r for r in bun2.sv_rows if r["type"] == "INS"]
    assert len(cnv_rows) == 1 and "k_ref=2,k_alt=3" in cnv_rows[0]["alleles"]
    assert len(del_rows) == 1 and len(ins_rows) == 1
    assert cnv_rows[0]["region_id"] == del_rows[0]["region_id"] == ins_rows[0]["region_id"]


def test_tier5_rf02_ins_and_inv_missing_evidence_genotyping_cli(
    tmp_path: Path,
) -> None:
    """Counterexample 2 (CLI): INS deletion-of-D calls '.' while D_snp/D call '0'; INV short B+/B- call '.' while A-B-C calls '0'."""
    # Part A: INS genotyping
    b_ins = SyntheticInputBuilder(tmp_path / "rf02_ins", chrom="Chr01", backbone_prefix="Ref#1#")
    seq_d = make_complex_seq(100, seed=6202)
    b_ins.add_node("1", make_complex_seq(100, seed=6201))
    b_ins.add_node("2", seq_d)
    b_ins.add_node("3", make_complex_seq(100, seed=6203))
    b_ins.add_node("4", make_complex_seq(100, seed=6204))  # X
    b_ins.add_node("5", mutate_seq_snps(seq_d, positions=[50]))  # D_snp
    b_ins.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+"])
    b_ins.add_path("Alt1#1#Chr01", ["1+", "4+", "3+"])
    b_ins.add_path("Alt2#1#Chr01", ["1+", "3+"])
    b_ins.add_path("Alt3#1#Chr01", ["1+", "5+", "3+"])
    b_ins.add_path("Alt4#1#Chr01", ["1+", "2+", "3+"])
    inp_ins = b_ins.write("rf02_ins")

    proc_ins, bun_ins = run_build(inp_ins)
    assert proc_ins.returncode == 0 and bun_ins is not None
    run_verify(inp_ins.out_prefix, inp_ins.gfa_path, inp_ins.ref_fa_path, check=True)
    assert_all_global_invariants(bun_ins, inp_ins.gfa_path, inp_ins.ref_fa_path)
    ins_ev = next(r for r in bun_ins.sv_rows if r["type"] == "INS")
    ins_gt = bun_ins.genotypes[ins_ev["event_id"]]
    assert ins_gt["Ref#1"] == "0"
    assert ins_gt["Alt1#1"] == "1"
    assert ins_gt["Alt2#1"] == "."
    assert ins_gt["Alt3#1"] == "0"
    assert ins_gt["Alt4#1"] == "0"
    assert int(ins_ev["n_called"]) == 4  # Alt2 excluded

    # Part B: INV genotyping
    b_inv = SyntheticInputBuilder(tmp_path / "rf02_inv", chrom="Chr01", backbone_prefix="Ref#1#")
    b_inv.add_node("1", make_complex_seq(150, seed=6211))
    b_inv.add_node("2", make_complex_seq(200, seed=6212))
    b_inv.add_node("3", make_complex_seq(150, seed=6213))
    b_inv.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+"])
    b_inv.add_path("Alt1#1#Chr01", ["1+", "2-", "3+"])
    b_inv.add_path("Alt2#1#Chr01", ["2+"])
    b_inv.add_path("Alt3#1#Chr01", ["2-"])
    b_inv.add_path("Alt4#1#Chr01", ["1+", "2+", "3+"])
    inp_inv = b_inv.write("rf02_inv")

    proc_inv, bun_inv = run_build(inp_inv)
    assert proc_inv.returncode == 0 and bun_inv is not None
    run_verify(inp_inv.out_prefix, inp_inv.gfa_path, inp_inv.ref_fa_path, check=True)
    assert_all_global_invariants(bun_inv, inp_inv.gfa_path, inp_inv.ref_fa_path)
    inv_ev = next(r for r in bun_inv.sv_rows if r["type"] == "INV")
    inv_gt = bun_inv.genotypes[inv_ev["event_id"]]
    assert inv_gt["Ref#1"] == "0"
    assert inv_gt["Alt1#1"] == "1"
    assert inv_gt["Alt2#1"] == "."
    assert inv_gt["Alt3#1"] == "."
    assert inv_gt["Alt4#1"] == "0"
    assert int(inv_ev["n_called"]) == 3


def test_tier5_rf03_short_multihit_kept_separate_and_small_ins_merged_cli(
    tmp_path: Path,
) -> None:
    """Counterexample 3 (CLI): U(200) vs U+U / U+U_2snp / U+rc(U) remain 2 alleles; 200=30I200= & 225=49I225= merge; 50I stays separate."""
    b = SyntheticInputBuilder(tmp_path / "rf03", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 8):
        b.add_node(f"r{i}", make_complex_seq(100, seed=6300 + i))

    # Locus 1 (r1..r2): U(200) vs U+U(400) -> 2 separate alleles (400 + 100 + 200 = 700)
    u1 = make_complex_seq(200, seed=6310)
    b.add_node("l1_s", u1)
    b.add_node("l1_r", u1 + u1)

    # Locus 2 (r2..r3): U(200) vs U+U_2snp(400) -> 2 separate alleles (700)
    u2 = make_complex_seq(200, seed=6311)
    b.add_node("l2_s", u2)
    b.add_node("l2_r", u2 + mutate_seq_snps(u2, positions=[40, 150]))

    # Locus 3 (r3..r4): U(200) vs U+revcomp(U)(400) -> 2 separate alleles (700)
    u3 = make_complex_seq(200, seed=6312)
    b.add_node("l3_s", u3)
    b.add_node("l3_r", u3 + revcomp(u3))

    # Locus 4 (r4..r5): 200=30I200= -> 1 merged representative (430 bp, since 430 > 400 sorted first)
    # Wait: put R=400 with 25bp flanks (450 bp) and S=200+30I+200 (430 bp) so S aligns into longer R!
    fl4 = make_complex_seq(225, seed=6320)
    fr4 = make_complex_seq(225, seed=6321)
    b.add_node("l4_r", fl4 + fr4)  # 450 bp
    b.add_node("l4_s", fl4[25:] + make_complex_seq(30, seed=6322) + fr4[:-25])  # 430 bp

    # Locus 5 (r5..r6): 225=49I225= embedded in 500 bp R -> 1 merged representative (500 bp)
    fl5 = make_complex_seq(250, seed=6323)
    fr5 = make_complex_seq(250, seed=6324)
    b.add_node("l5_r", fl5 + fr5)  # 500 bp
    b.add_node("l5_s", fl5[25:] + make_complex_seq(49, seed=6325) + fr5[:-25])  # 499 bp

    # Locus 6 (r6..r7): 50I embedded in 500 bp R -> NOT merged (500 + 100 + 499 = 1099 bp)
    fl6 = make_complex_seq(250, seed=6326)
    fr6 = make_complex_seq(250, seed=6327)
    b.add_node("l6_r", fl6 + fr6)  # 500 bp
    b.add_node("l6_s", fl6[26:] + make_complex_seq(50, seed=6328) + fr6[:-25])  # 499 bp

    b.set_ref_path("Ref#1#Chr01", [f"r{i}+" for i in range(1, 8)])
    b.add_path(
        "Alt1#1#Chr01",
        ["r1+", "l1_r+", "r2+", "l2_r+", "r3+", "l3_r+", "r4+", "l4_r+", "r5+", "l5_r+", "r6+", "l6_r+", "r7+"],
    )
    b.add_path(
        "Alt2#1#Chr01",
        ["r1+", "l1_s+", "r2+", "l2_s+", "r3+", "l3_s+", "r4+", "l4_s+", "r5+", "l5_s+", "r6+", "l6_s+", "r7+"],
    )
    inp = b.write("rf03_cli")

    proc, bun = run_build(inp)
    assert proc.returncode == 0 and bun is not None
    run_verify(inp.out_prefix, inp.gfa_path, inp.ref_fa_path, check=True)
    assert_all_global_invariants(bun, inp.gfa_path, inp.ref_fa_path)

    # Check exact slot_ins_len for each of the 6 loci (mounted before r2..r7 = steps 1..6)
    assert bun.offsets[1]["slot_ins_len"] == 700  # U vs U+U kept separate
    assert bun.offsets[2]["slot_ins_len"] == 700  # U vs U+U_2snp kept separate
    assert bun.offsets[3]["slot_ins_len"] == 700  # U vs U+rc(U) kept separate
    assert bun.offsets[4]["slot_ins_len"] == 450  # 200=30I200= merged
    assert bun.offsets[5]["slot_ins_len"] == 500  # 225=49I225= merged
    assert bun.offsets[6]["slot_ins_len"] == 1099  # 50I kept separate


def test_tier5_rf04_join_before_synthetic_and_verify_tamper_exit_nonzero_cli(
    tmp_path: Path,
) -> None:
    """Counterexample 4 (CLI): A-D-X-B-D, repeat-guard left-trim, and cycle-fold U-U-Tail emit 'synthetic' and reject tampering with exit 1 & [INV2]."""
    # 1. Stripped ref node: Ref = A-B-D, Alt = A-D-X-B-D
    b1 = SyntheticInputBuilder(tmp_path / "rf04_strip", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("A", 6401), ("B", 6402), ("D", 6403), ("X", 6404)]:
        b1.add_node(nid, make_complex_seq(150, seed=seed))
    b1.set_ref_path("Ref#1#Chr01", ["A+", "B+", "D+"])
    b1.add_path("Alt1#1#Chr01", ["A+", "D+", "X+", "B+", "D+"])
    inp1 = b1.write("rf04_strip")
    proc1, bun1 = run_build(inp1)
    assert proc1.returncode == 0 and bun1 is not None
    assert len(bun1.inserts) == 1 and bun1.inserts[0]["join_before"] == "synthetic"
    run_verify(inp1.out_prefix, inp1.gfa_path, inp1.ref_fa_path, check=True)
    assert_all_global_invariants(bun1, inp1.gfa_path, inp1.ref_fa_path)

    tamper1 = tmp_path / "rf04_strip" / "tamper1_out"
    _tamper_inserts_join_before(inp1.out_prefix, tamper1, 0, "source_contiguous")
    v1 = run_verify(tamper1, inp1.gfa_path, inp1.ref_fa_path, check=False)
    assert v1.returncode == 1
    assert "[INV2]" in v1.stderr
    assert "Traceback" not in v1.stderr

    # 2. Repeat-guard left-trimmed (u_start > 0 -> synthetic)
    b2 = SyntheticInputBuilder(tmp_path / "rf04_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    b2.add_node("1", make_complex_seq(200, seed=6410))
    b2.add_node("2", make_complex_seq(200, seed=6411))
    b2.add_node(
        "3",
        make_satellite_seq(unit_len=20, total_len=1500, seed=6412)
        + make_complex_seq(400, seed=6413),
    )
    b2.set_ref_path("Ref#1#Chr01", ["1+", "2+"])
    b2.add_path("Alt1#1#Chr01", ["1+", "3+", "2+"])
    inp2 = b2.write("rf04_rg")
    proc2, bun2 = run_build(inp2)
    assert proc2.returncode == 0 and bun2 is not None
    assert len(bun2.inserts) == 1 and bun2.inserts[0]["join_before"] == "synthetic"
    run_verify(inp2.out_prefix, inp2.gfa_path, inp2.ref_fa_path, check=True)

    tamper2 = tmp_path / "rf04_rg" / "tamper2_out"
    _tamper_inserts_join_before(inp2.out_prefix, tamper2, 0, "source_contiguous")
    v2 = run_verify(tamper2, inp2.gfa_path, inp2.ref_fa_path, check=False)
    assert v2.returncode == 1
    assert "[INV2]" in v2.stderr
    assert "Traceback" not in v2.stderr

    # 3. Cycle-folded U-U-Tail (--max-copy 1) -> piece 0 source_contiguous, piece 1 synthetic; tamper piece 1 -> exit 1 & [INV2]
    b3 = SyntheticInputBuilder(tmp_path / "rf04_cf", chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("1", 6420), ("2", 6421), ("3", 6422), ("4", 6423)]:
        b3.add_node(nid, make_complex_seq(120, seed=seed))
    b3.set_ref_path("Ref#1#Chr01", ["1+", "2+"])
    b3.add_path("Alt1#1#Chr01", ["1+", "3+", "3+", "4+", "2+"])
    inp3 = b3.write("rf04_cf")
    proc3, bun3 = run_build(inp3, extra_args=["--max-copy", "1"])
    assert proc3.returncode == 0 and bun3 is not None
    assert len(bun3.inserts) == 2
    assert bun3.inserts[0]["join_before"] == "source_contiguous"
    assert bun3.inserts[1]["join_before"] == "synthetic"
    run_verify(inp3.out_prefix, inp3.gfa_path, inp3.ref_fa_path, check=True)

    tamper3 = tmp_path / "rf04_cf" / "tamper3_out"
    _tamper_inserts_join_before(inp3.out_prefix, tamper3, 1, "source_contiguous")
    v3 = run_verify(tamper3, inp3.gfa_path, inp3.ref_fa_path, check=False)
    assert v3.returncode == 1
    assert "[INV2]" in v3.stderr
    assert "Traceback" not in v3.stderr


def test_tier5_rf05_1200_layer_nested_reanchor_and_t22_multithread_determinism_cli(
    tmp_path: Path,
) -> None:
    """Counterexample 5 + T22 (CLI): 1,200-layer nested re-anchor + combined RF1-5 graph with --threads 1 vs --threads 4 SHA-256 byte-identity."""
    depth = 1200
    b = SyntheticInputBuilder(tmp_path / "rf05", chrom="Chr01", backbone_prefix="Ref#1#")

    # Nodes 1..3: S(100), E(100), X(60); Nodes 4..1203: u_1..u_1200 (10 bp each)
    b.add_node("1", make_complex_seq(100, seed=6501))
    b.add_node("2", make_complex_seq(100, seed=6502))
    x_seq = make_complex_seq(60, seed=6503)
    b.add_node("3", x_seq)
    for k in range(1, depth + 1):
        b.add_node(str(3 + k), make_complex_seq(10, seed=6600 + k))

    # Extra tail region exercising Items 1-4 in the same T22 determinism graph:
    # tail ref: E(2) -> 1204(U) -> 1204(U) -> 1205(D) -> 1206(B) -> 1207(InvNode) -> 1208(C)
    base_extra = 4 + depth
    u_nid = str(base_extra)
    d_nid = str(base_extra + 1)
    b_nid = str(base_extra + 2)
    inv_nid = str(base_extra + 3)
    c_nid = str(base_extra + 4)
    x_scat_nid = str(base_extra + 5)
    u200 = make_complex_seq(200, seed=6520)
    mh_s_nid = str(base_extra + 6)
    mh_r_nid = str(base_extra + 7)
    strip_x_nid = str(base_extra + 8)
    z_nid = str(base_extra + 9)
    b.add_node(u_nid, make_complex_seq(100, seed=6510))
    b.add_node(d_nid, make_complex_seq(100, seed=6511))
    b.add_node(b_nid, make_complex_seq(100, seed=6512))
    b.add_node(inv_nid, make_complex_seq(200, seed=6513))
    b.add_node(c_nid, make_complex_seq(100, seed=6514))
    b.add_node(x_scat_nid, make_complex_seq(100, seed=6515))
    b.add_node(mh_s_nid, u200)
    b.add_node(mh_r_nid, u200 + u200)
    b.add_node(strip_x_nid, make_complex_seq(120, seed=6516))
    b.add_node(z_nid, make_complex_seq(100, seed=6517))

    def _u(k: int) -> tuple[str, str]:
        return (str(3 + k), "+")

    ladder_ref: list[tuple[str, str]] = [_u(1), ("1", "+")]
    for k in range(1, depth):
        ladder_ref.append(_u(k + 1))
        ladder_ref.append(_u(k))
    ladder_ref.append(_u(depth))
    ladder_ref.append(("2", "+"))

    tail_ref: list[tuple[str, str]] = [
        (u_nid, "+"),
        (u_nid, "+"),
        (d_nid, "+"),
        (b_nid, "+"),
        (inv_nid, "+"),
        (c_nid, "+"),
        (z_nid, "+"),
    ]
    full_ref = ladder_ref + tail_ref
    b.set_ref_path("Ref#1#Chr01", full_ref)

    # Alt1: 1,200-layer nested ladder + 60 bp X + scattered U-X-U-X-U + short multi-hit R(400)
    ladder_alt: list[tuple[str, str]] = [_u(1), ("1", "+")]
    for k in range(1, depth + 1):
        ladder_alt.append(_u(k))
    ladder_alt.append(("3", "+"))
    ladder_alt.append(("2", "+"))
    b.add_path(
        "Alt1#1#Chr01",
        ladder_alt
        + [
            (u_nid, "+"),
            (x_scat_nid, "+"),
            (u_nid, "+"),
            (x_scat_nid, "+"),
            (u_nid, "+"),
            (b_nid, "+"),
            (mh_r_nid, "+"),
            (inv_nid, "+"),
            (c_nid, "+"),
            (z_nid, "+"),
        ],
    )

    # Alt2: INV carrier (cleanly flanked by b_nid..c_nid) + short multi-hit S(200) + stripped ref node insertion (c_nid..z_nid)
    b.add_path(
        "Alt2#1#Chr01",
        ladder_ref
        + [
            (u_nid, "+"),
            (u_nid, "+"),
            (d_nid, "+"),
            (b_nid, "+"),
            (mh_s_nid, "+"),
            (inv_nid, "-"),
            (c_nid, "+"),
            (d_nid, "+"),
            (strip_x_nid, "+"),
            (z_nid, "+"),
        ],
    )

    # Alt3: short internal INV contig (should call '.' on INV)
    b.add_path("Alt3#1#Chr01", [(inv_nid, "-")])

    inp = b.write("rf05_combined_1200")
    out_t1 = tmp_path / "rf05_t1"
    out_t4 = tmp_path / "rf05_t4"
    env = {"PANFLAT_FORCE_FORK": "1", "PYTHONWARNINGS": "default"}

    p1, bun1 = run_build(inp, extra_args=["--threads", "1"], out_prefix=out_t1, extra_env=env)
    p4, bun4 = run_build(inp, extra_args=["--threads", "4"], out_prefix=out_t4, extra_env=env)
    assert p1.returncode == 0 and bun1 is not None
    assert p4.returncode == 0 and bun4 is not None

    # T22 strict byte-identity across all output files
    for suffix in _CORE_OUTPUT_SUFFIXES:
        f1 = Path(f"{out_t1}{suffix}")
        f4 = Path(f"{out_t4}{suffix}")
        assert f1.is_file() and f4.is_file()
        assert _sha256_file(f1) == _sha256_file(f4), f"SHA-256 mismatch on {suffix}"

    # Independent CLI verify + global invariant oracle
    for pref in (out_t1, out_t4):
        run_verify(pref, inp.gfa_path, inp.ref_fa_path, check=True)
    assert_all_global_invariants(bun4, inp.gfa_path, inp.ref_fa_path)

    # Verify the 60 bp insertion X from depth 1,200 was accurately recovered in pan.fa
    pan_seq = bun4.pan_seq.upper()
    assert x_seq in pan_seq
    assert any(
        r["type"] == "INS" and "len=60" in r["alleles"]
        for r in bun4.sv_rows
    )
    # Verify scattered branch did not produce false CNV and short INV contig called '.'
    assert not any(r["type"] == "CNV" for r in bun4.sv_rows)
    inv_ev = next(r for r in bun4.sv_rows if r["type"] == "INV")
    assert bun4.genotypes[inv_ev["event_id"]]["Alt3#1"] == "."


def test_tier5_rf06_wave1_case_a_b_affine_c1_and_spacer_n_tamper_cli(
    tmp_path: Path,
) -> None:
    """Wave 1 opaque-box CLI E2E checks: Case A 50bp DEL+1bp INS -> '.', Case B 2x25bp INS -> '.', 30I/30D+1SNP merge, short-query C1b/C1c separation, and spacer tamper after Ref N*100 -> exit 1 + [INV2]."""
    # 1. Case A (50 bp DEL D1 + 1 bp INS in D2 -> INS='.' alongside DEL='1') + Case B (2x25 bp novel INS -> '.', 2x24 bp -> '0')
    b_ins = SyntheticInputBuilder(tmp_path / "w1_ins", chrom="Chr01", backbone_prefix="Ref#1#")
    d1 = make_complex_seq(50, seed=6701)
    d2 = make_complex_seq(100, seed=6702)
    b_ins.add_node("1", make_complex_seq(150, seed=6700))
    b_ins.add_node("2", d1)
    b_ins.add_node("3", d2)
    b_ins.add_node("4", make_complex_seq(150, seed=6703))
    b_ins.add_node("5", make_complex_seq(150, seed=6704))  # X replacement
    b_ins.add_node("6", d2[:50] + "A" + d2[50:])  # 50bp DEL D1 + 1bp INS in D2
    b_ins.add_node("7", mutate_seq_snps(d1 + d2, positions=[75]))  # D_snp
    b_ins.add_node("8", make_complex_seq(25, seed=6705))
    b_ins.add_node("9", make_complex_seq(25, seed=6706))
    b_ins.add_node("10", make_complex_seq(24, seed=6707))
    b_ins.add_node("11", make_complex_seq(24, seed=6708))
    b_ins.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "4+"])
    b_ins.add_path("AltX#1#Chr01", ["1+", "5+", "4+"])
    b_ins.add_path("Alt50Del1Ins#1#Chr01", ["1+", "6+", "4+"])
    b_ins.add_path("AltDSnp#1#Chr01", ["1+", "7+", "4+"])
    b_ins.add_path("Alt2x25Ins#1#Chr01", ["1+", "8+", "2+", "9+", "3+", "4+"])
    b_ins.add_path("Alt2x24Ins#1#Chr01", ["1+", "10+", "2+", "11+", "3+", "4+"])
    inp_ins = b_ins.write("w1_ins")

    proc_ins, bun_ins = run_build(inp_ins)
    assert proc_ins.returncode == 0 and bun_ins is not None
    run_verify(inp_ins.out_prefix, inp_ins.gfa_path, inp_ins.ref_fa_path, check=True)
    assert_all_global_invariants(bun_ins, inp_ins.gfa_path, inp_ins.ref_fa_path)
    ins_ev = next(r for r in bun_ins.sv_rows if r["type"] == "INS")
    ins_gt = bun_ins.genotypes[ins_ev["event_id"]]
    assert ins_gt["AltX#1"] == "1"
    assert ins_gt["Alt50Del1Ins#1"] == "."
    assert ins_gt["AltDSnp#1"] == "0"
    assert ins_gt["Alt2x25Ins#1"] == "."
    assert ins_gt["Alt2x24Ins#1"] == "0"

    # 2. Item 3.1/3.2 CLI: 30I+1SNP & 30D+1SNP merge into 1 allele; s218 in U170+U170_1snp & rc(U170)+U170_1snp stays 2 alleles
    b_rg = SyntheticInputBuilder(tmp_path / "w1_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 6):
        b_rg.add_node(f"r{i}", make_complex_seq(100, seed=6800 + i))
    # Locus 1 (r1..r2): 30I + 1 SNP at +10bp embedded in 450bp R -> merges (slot_ins_len == 450)
    r_30i = make_complex_seq(450, seed=6810)
    r_30i_mut = mutate_seq_snps(r_30i, positions=[235])
    s_30i = r_30i_mut[25:225] + make_complex_seq(30, seed=6811) + r_30i_mut[225:-25]
    b_rg.add_node("l1_r", r_30i)
    b_rg.add_node("l1_s", s_30i)
    # Locus 2 (r2..r3): 30D + 1 SNP at +10bp in 500bp R -> merges (slot_ins_len == 500)
    r_30d = make_complex_seq(500, seed=6812)
    s_30d = r_30d[:200] + mutate_seq_snps(r_30d[230:], positions=[10])
    b_rg.add_node("l2_r", r_30d)
    b_rg.add_node("l2_s", s_30d)
    # Locus 3 (r3..r4): s218 vs U170 + U170_1snp -> stays 2 alleles (340 + 100 + 218 = 658)
    u170 = make_complex_seq(170, seed=6813)
    u170_1snp = mutate_seq_snps(u170, positions=[10])
    s218 = u170[:85] + make_complex_seq(48, seed=6814) + u170[85:]
    b_rg.add_node("l3_r", u170 + u170_1snp)
    b_rg.add_node("l3_s", s218)
    # Locus 4 (r4..r5): s218 vs rc(U170) + U170_1snp (strictly better RC) -> stays 2 alleles (658)
    b_rg.add_node("l4_r", revcomp(u170) + u170_1snp)
    b_rg.add_node("l4_s", s218)

    b_rg.set_ref_path("Ref#1#Chr01", [f"r{i}+" for i in range(1, 6)])
    b_rg.add_path("Alt1#1#Chr01", ["r1+", "l1_r+", "r2+", "l2_r+", "r3+", "l3_r+", "r4+", "l4_r+", "r5+"])
    b_rg.add_path("Alt2#1#Chr01", ["r1+", "l1_s+", "r2+", "l2_s+", "r3+", "l3_s+", "r4+", "l4_s+", "r5+"])
    inp_rg = b_rg.write("w1_rg")
    proc_rg, bun_rg = run_build(inp_rg)
    assert proc_rg.returncode == 0 and bun_rg is not None
    run_verify(inp_rg.out_prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    assert bun_rg.offsets[1]["slot_ins_len"] == 450
    assert bun_rg.offsets[2]["slot_ins_len"] == 500
    assert bun_rg.offsets[3]["slot_ins_len"] == 658
    assert bun_rg.offsets[4]["slot_ins_len"] == 658

    # 3. Item 4.2 CLI: tampering join_before='source_contiguous' -> 'spacer' after Ref ending in 'N'*100 -> exit 1 + [INV2]
    b_sp = SyntheticInputBuilder(tmp_path / "w1_spn", chrom="Chr01", backbone_prefix="Ref#1#")
    b_sp.add_node("1", make_complex_seq(100, seed=6901) + "N" * 100)
    b_sp.add_node("2", make_complex_seq(200, seed=6902))
    b_sp.add_node("3", make_complex_seq(180, seed=6903))
    b_sp.set_ref_path("Ref#1#Chr01", ["1+", "2+"])
    b_sp.add_path("Alt1#1#Chr01", ["1+", "3+", "2+"])
    inp_sp = b_sp.write("w1_spn")
    proc_sp, bun_sp = run_build(inp_sp)
    assert proc_sp.returncode == 0 and bun_sp is not None
    assert len(bun_sp.inserts) == 1 and bun_sp.inserts[0]["join_before"] == "source_contiguous"
    run_verify(inp_sp.out_prefix, inp_sp.gfa_path, inp_sp.ref_fa_path, check=True)

    tamper_sp = tmp_path / "w1_spn" / "tamper_sp_out"
    _tamper_inserts_join_before(inp_sp.out_prefix, tamper_sp, 0, "spacer")
    v_sp = run_verify(tamper_sp, inp_sp.gfa_path, inp_sp.ref_fa_path, check=False)
    assert v_sp.returncode == 1
    assert "[INV2]" in v_sp.stderr
    assert "Traceback" not in v_sp.stderr


def test_tier5_rf07_wave2_final_polish_cli(
    tmp_path: Path,
) -> None:
    """Wave 2 opaque-box CLI E2E checks: synthetic sc.seq==rep0_seq & ins_excess accounting, displaced '+' anchor near INV_EDGE preserving INV/INS/DEL='0', and short-query core fallback / shifted near-palindrome / 0-SNP 4x4bp / 60->15 / flanked 30bp overhang clustering."""
    # 1. R1-r2 Finding 1 (synth sc.seq == rep0_seq: AltDelN1='.', AltBsec='0')
    b_f1 = SyntheticInputBuilder(tmp_path / "w2_f1", chrom="Chr01", backbone_prefix="Ref#1#")
    d1, d2 = make_complex_seq(100, seed=7101), make_complex_seq(100, seed=7102)
    b_f1.add_node("1", make_complex_seq(150, seed=7100))
    b_f1.add_node("2", d1)
    b_f1.add_node("3", d2)
    b_f1.add_node("4", make_complex_seq(150, seed=7103))
    b_f1.add_node("5", make_complex_seq(150, seed=7104))
    b_f1.add_node("6", d1)
    b_f1.add_node("7", d2)
    b_f1.set_ref_path("Ref#1#Chr01", ["1+", "2+", "3+", "4+"])
    b_f1.add_path("AltIns#1#Chr01", ["1+", "5+", "4+"])
    b_f1.add_path("AltBsec#1#Chr01", ["1+", "6+", "7+", "4+"])
    b_f1.add_path("AltDelN1#1#Chr01", ["1+", "7+", "4+"])
    inp_f1 = b_f1.write("w2_f1")
    p_f1, bun_f1 = run_build(inp_f1)
    assert p_f1.returncode == 0 and bun_f1 is not None
    run_verify(inp_f1.out_prefix, inp_f1.gfa_path, inp_f1.ref_fa_path, check=True)
    assert_all_global_invariants(bun_f1, inp_f1.gfa_path, inp_f1.ref_fa_path)
    ins_f1 = next(r for r in bun_f1.sv_rows if r["type"] == "INS")
    gt_f1 = bun_f1.genotypes[ins_f1["event_id"]]
    assert gt_f1["AltDelN1#1"] == "." and gt_f1["AltBsec#1"] == "0" and gt_f1["AltIns#1"] == "1"

    # 2. C1-r2 Bug 1: right INV_EDGE + displaced 2+ keeps INV/INS/DEL='0' on unbroken forward region
    b_b1 = SyntheticInputBuilder(tmp_path / "w2_b1", chrom="Chr01", backbone_prefix="Ref#1#")
    for i, length in enumerate([500, 60, 200, 200, 200, 200, 200, 200, 200, 200], start=1):
        b_b1.add_node(str(i), make_complex_seq(length, seed=7200 + i))
    b_b1.add_node("11", make_complex_seq(150, seed=7211))
    b_b1.set_ref_path("Ref#1#Chr01", [f"{i}+" for i in range(1, 11)])
    b_b1.add_path("AltInv4#1#Chr01", [f"{i}-" if i == 5 else f"{i}+" for i in range(1, 11)])
    b_b1.add_path(
        "AltIns45#1#Chr01",
        [f"{i}+" for i in range(1, 6)] + ["11+"] + [f"{i}+" for i in range(6, 11)],
    )
    b_b1.add_path("AltDel4#1#Chr01", [f"{i}+" for i in range(1, 11) if i != 5])
    b_b1.add_path(
        "AltEdgeDispLeft#1#Chr01",
        [f"{i}+" for i in range(1, 10)] + ["2+", "10-"],
    )
    inp_b1 = b_b1.write("w2_b1")
    p_b1, bun_b1 = run_build(inp_b1)
    assert p_b1.returncode == 0 and bun_b1 is not None
    run_verify(inp_b1.out_prefix, inp_b1.gfa_path, inp_b1.ref_fa_path, check=True)
    assert_all_global_invariants(bun_b1, inp_b1.gfa_path, inp_b1.ref_fa_path)
    inv_b1 = next(r for r in bun_b1.sv_rows if r["type"] == "INV")
    ins_b1 = next(r for r in bun_b1.sv_rows if r["type"] == "INS")
    del_b1 = next(r for r in bun_b1.sv_rows if r["type"] == "DEL" and r["anchors"] == "3..5")
    assert bun_b1.genotypes[inv_b1["event_id"]]["AltEdgeDispLeft#1"] == "0"
    assert bun_b1.genotypes[ins_b1["event_id"]]["AltEdgeDispLeft#1"] == "0"
    assert bun_b1.genotypes[del_b1["event_id"]]["AltEdgeDispLeft#1"] == "0"

    # 3. Wave 2 repeat_guard.py CLI checks (5 loci in one graph)
    b_rg = SyntheticInputBuilder(tmp_path / "w2_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 7):
        b_rg.add_node(f"r{i}", make_complex_seq(100, seed=7300 + i))
    # Locus 1 (r1..r2): R2-r2 F1 u75+ov49 vs u75+rc(u75_1snp) -> 2 alleles (150 + 100 + 124 = 374)
    u75 = make_complex_seq(75, seed=7310)
    b_rg.add_node("l1_r", u75 + revcomp(mutate_seq_snps(u75, positions=[25])))
    b_rg.add_node("l1_s", make_complex_seq(49, seed=7311) + u75)
    # Locus 2 (r2..r3): C1-r2 Bug 2 shifted near-palindrome (d=30) -> 1 allele (430)
    fl1, fl2 = make_complex_seq(100, seed=7320), make_complex_seq(100, seed=7321)
    a30 = make_complex_seq(30, seed=7322)
    h85 = make_complex_seq(85, seed=7323)
    p170 = mutate_seq_snps(h85 + revcomp(h85), positions=[5, 15, 25, 35])
    b_rg.add_node("l2_r", fl1 + a30 + p170 + revcomp(a30) + fl2)
    b_rg.add_node("l2_s", a30 + p170)
    # Locus 3 (r3..r4): C1-r2 Bug 3 4x4bp 0-SNP pure insertions (sp=3) embedded in 560 bp R -> 1 allele (560)
    r560 = make_complex_seq(560, seed=7330)
    base500 = r560[30:530]
    ins4 = [make_complex_seq(4, seed=7331 + j) for j in range(4)]
    s_4x4 = (
        base500[:200]
        + ins4[0]
        + base500[200:203]
        + ins4[1]
        + base500[203:206]
        + ins4[2]
        + base500[206:209]
        + ins4[3]
        + base500[209:]
    )
    b_rg.add_node("l3_r", r560)
    b_rg.add_node("l3_s", s_4x4)
    # Locus 4 (r4..r5): C1-r2 Bug 4 60->15 deletion-direction replacement -> 2 alleles (500 + 100 + 455 = 1055)
    r500 = make_complex_seq(500, seed=7340)
    b_rg.add_node("l4_r", r500)
    b_rg.add_node("l4_s", r500[:200] + make_complex_seq(15, seed=7341) + r500[260:])
    # Locus 5 (r5..r6): C1-r2 Bug 5 pure 30 bp right overhang on flanked 600 bp R -> 1 allele (600)
    r600 = make_complex_seq(600, seed=7350)
    b_rg.add_node("l5_r", r600)
    b_rg.add_node("l5_s", r600[100:500] + make_complex_seq(30, seed=7351))

    b_rg.set_ref_path("Ref#1#Chr01", [f"r{i}+" for i in range(1, 7)])
    b_rg.add_path("Alt1#1#Chr01", ["r1+", "l1_r+", "r2+", "l2_r+", "r3+", "l3_r+", "r4+", "l4_r+", "r5+", "l5_r+", "r6+"])
    b_rg.add_path("Alt2#1#Chr01", ["r1+", "l1_s+", "r2+", "l2_s+", "r3+", "l3_s+", "r4+", "l4_s+", "r5+", "l5_s+", "r6+"])
    inp_rg = b_rg.write("w2_rg")
    p_rg, bun_rg = run_build(inp_rg)
    assert p_rg.returncode == 0 and bun_rg is not None
    run_verify(inp_rg.out_prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    assert_all_global_invariants(bun_rg, inp_rg.gfa_path, inp_rg.ref_fa_path)
    assert bun_rg.offsets[1]["slot_ins_len"] == 374
    assert bun_rg.offsets[2]["slot_ins_len"] == 430
    assert bun_rg.offsets[3]["slot_ins_len"] == 560
    assert bun_rg.offsets[4]["slot_ins_len"] == 1055
    assert bun_rg.offsets[5]["slot_ins_len"] == 600


def test_tier5_rf08_wave3_holistic_regression_guard_cli(tmp_path: Path) -> None:
    """Wave 3 / Iteration 4 strictly opaque-box CLI E2E regression guard covering all 8 findings."""
    # --- Subcase 1: R1-r3 F1 CLI (adjacent-anchor INV_CONFLICT + novel insertion) ---
    b1 = SyntheticInputBuilder(tmp_path / "w3_f1", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 12):
        b1.add_node(str(i), make_complex_seq(200, seed=8100 + i))
    b1.add_node("12", make_complex_seq(150, seed=8112))  # INS(4..5)
    b1.add_node("13", make_complex_seq(150, seed=8113))  # INS(3..6)
    b1.add_node("14", make_complex_seq(200, seed=8114))  # co-located novel in AltConfPlusNov
    b1.add_node("15", make_complex_seq(150, seed=8115))  # adjacent INS(3..4)
    ref1 = [f"{i}+" for i in range(1, 12)]
    b1.set_ref_path("Ref#1#Chr01", ref1)
    b1.add_path("AltIns45#1#Chr01", [f"{i}+" for i in range(1, 6)] + ["12+"] + [f"{i}+" for i in range(6, 12)])
    b1.add_path("AltIns36#1#Chr01", [f"{i}+" for i in range(1, 5)] + ["13+"] + [f"{i}+" for i in range(7, 12)])
    b1.add_path("AltDel36#1#Chr01", [f"{i}+" for i in range(1, 5)] + [f"{i}+" for i in range(7, 12)])
    b1.add_path("AltInv45#1#Chr01", [f"{i}+" for i in range(1, 5)] + ["6-", "5-"] + [f"{i}+" for i in range(7, 12)])
    b1.add_path("AltInv4#1#Chr01", [f"{i}+" for i in range(1, 5)] + ["5-"] + [f"{i}+" for i in range(6, 12)])
    b1.add_path("AltIns34#1#Chr01", [f"{i}+" for i in range(1, 5)] + ["15+"] + [f"{i}+" for i in range(5, 12)])
    b1.add_path("AltClean#1#Chr01", ref1)
    conflict_blk = ["9-", "8-", "10-", "9-"]
    b1.add_path(
        "AltConfInIns45#1#Chr01",
        [f"{i}+" for i in range(1, 6)] + conflict_blk + [f"{i}+" for i in range(6, 12)],
    )
    b1.add_path(
        "AltConfPlusNov#1#Chr01",
        [f"{i}+" for i in range(1, 6)] + ["14+"] + conflict_blk + [f"{i}+" for i in range(6, 12)],
    )
    inp1 = b1.write("w3_f1")
    p1, bun1 = run_build(inp1, extra_args=["--max-jump", "500"])
    assert p1.returncode == 0 and bun1 is not None
    run_verify(inp1.out_prefix, inp1.gfa_path, inp1.ref_fa_path, check=True)
    assert_all_global_invariants(bun1, inp1.gfa_path, inp1.ref_fa_path)

    ev_ins45 = next(r for r in bun1.sv_rows if r["type"] == "INS" and r["anchors"] == "4..5")
    ev_ins36 = next(r for r in bun1.sv_rows if r["type"] == "INS" and r["anchors"] == "3..6")
    ev_del36 = next(r for r in bun1.sv_rows if r["type"] == "DEL" and r["anchors"] == "3..6")
    ev_inv45 = next(r for r in bun1.sv_rows if r["type"] == "INV" and r["anchors"] == "3..6")
    ev_ins34 = next(r for r in bun1.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    ev_inv4 = next(r for r in bun1.sv_rows if r["type"] == "INV" and r["anchors"] == "3..5")
    for ev in (ev_ins45, ev_ins36, ev_del36, ev_inv45):
        gt = bun1.genotypes[ev["event_id"]]
        assert gt["AltConfInIns45#1"] == ".", f"{ev['type']} {ev['anchors']} leaked on AltConfInIns45"
        assert gt["AltConfPlusNov#1"] == ".", f"{ev['type']} {ev['anchors']} leaked on AltConfPlusNov"
        assert gt["AltClean#1"] == "0"
    for ev in (ev_ins34, ev_inv4):
        gt = bun1.genotypes[ev["event_id"]]
        assert gt["AltConfInIns45#1"] == "0"
        assert gt["AltConfPlusNov#1"] == "0"
        assert gt["AltClean#1"] == "0"

    # --- Subcase 2: C1-r3 B2 CLI (centromere-split multi-fragment INS + REPEAT_LIKE + T22 fork determinism) ---
    b2 = SyntheticInputBuilder(tmp_path / "w3_b2", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 8):
        b2.add_node(str(i), make_complex_seq(200, seed=8200 + i))
    b2.add_node("i1", make_complex_seq(150, seed=8211))
    b2.add_node("i2", make_complex_seq(160, seed=8212))
    b2.add_node("rep600", "ACGT" * 150)  # 600 bp >= 500 bp -> guaranteed REPEAT_LIKE mask
    ref2 = [f"{i}+" for i in range(1, 8)]
    b2.set_ref_path("Ref#1#Chr01", ref2)
    b2.add_bed_region(800, 1200, "CEN1")  # spans nodes 5 and 6
    b2.add_path("AltIns1#1#Chr01", ["1+", "i1+", "2+", "3+", "4+", "5+", "6+", "7+"])
    b2.add_path("AltIns2#1#Chr01", ["1+", "i2+", "2+", "3+", "4+", "5+", "6+", "7+"])
    b2.add_path(
        "AltLoopConflict#1#Chr01",
        ["1+", "i1+", "2+", "3+", "4+", "5+", "6+", "1+", "2+", "3+", "4+", "7+"],
    )
    b2.add_path(
        "AltLoopTwoAlleles#1#Chr01",
        ["1+", "i1+", "2+", "3+", "4+", "5+", "6+", "1+", "i2+", "2+", "3+", "4+", "7+"],
    )
    b2.add_path(
        "AltLoopSameAllele#1#Chr01",
        ["1+", "i1+", "2+", "3+", "4+", "5+", "6+", "1+", "i1+", "2+", "3+", "4+", "7+"],
    )
    b2.add_path(
        "AltLoopMaskRef#1#Chr01",
        ["1+", "rep600+", "2+", "3+", "4+", "5+", "6+", "1+", "2+", "3+", "4+", "7+"],
    )
    b2.add_path(
        "AltLoopMaskAndA2#1#Chr01",
        ["1+", "rep600+", "2+", "3+", "4+", "5+", "6+", "1+", "i2+", "2+", "3+", "4+", "7+"],
    )
    b2.add_path("AltTwoContigs#1#c1", ["1+", "i1+", "2+", "3+", "4+"])
    b2.add_path("AltTwoContigs#1#c2", ["1+", "2+", "3+", "4+", "7+"])
    b2.add_path("AltTwoContigs12#1#c1", ["1+", "i1+", "2+", "3+", "4+"])
    b2.add_path("AltTwoContigs12#1#c2", ["1+", "i2+", "2+", "3+", "4+", "7+"])
    inp2 = b2.write("w3_b2")

    p2_t1, bun2_t1 = run_build(
        inp2,
        out_prefix=tmp_path / "w3_b2" / "out_t1",
        extra_args=["--centromere-mode", "bypass", "--threads", "1"],
    )
    p2_t4, bun2_t4 = run_build(
        inp2,
        out_prefix=tmp_path / "w3_b2" / "out_t4",
        extra_args=["--centromere-mode", "bypass", "--threads", "4"],
        extra_env={"PANFLAT_FORCE_FORK": "1"},
    )
    assert p2_t1.returncode == 0 and bun2_t1 is not None
    assert p2_t4.returncode == 0 and bun2_t4 is not None
    assert inp2.bed_path is not None
    run_verify(
        bun2_t1.prefix,
        inp2.gfa_path,
        inp2.ref_fa_path,
        extra_args=["--centromere-bed", inp2.bed_path],
        check=True,
    )
    run_verify(
        bun2_t4.prefix,
        inp2.gfa_path,
        inp2.ref_fa_path,
        extra_args=["--centromere-bed", inp2.bed_path],
        check=True,
    )
    assert_all_global_invariants(bun2_t1, inp2.gfa_path, inp2.ref_fa_path)

    ins2 = next(r for r in bun2_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "0..1")
    gt2 = bun2_t1.genotypes[ins2["event_id"]]
    assert "len=310" not in ins2["alleles"]
    assert "partial_content=1" not in ins2["alleles"]
    assert gt2["AltLoopConflict#1"] == "."
    assert gt2["AltTwoContigs#1"] == "."
    assert gt2["AltLoopTwoAlleles#1"] == "."
    assert gt2["AltTwoContigs12#1"] == "."
    assert gt2["AltLoopSameAllele#1"] == gt2["AltIns1#1"]
    assert gt2["AltLoopMaskRef#1"] == "0"
    assert gt2["AltLoopMaskAndA2#1"] == gt2["AltIns2#1"]
    rep_like_rows = [r for r in bun2_t1.unresolved if r["reason"] == "REPEAT_LIKE"]
    assert len(rep_like_rows) >= 2

    for suffix in (*_CORE_OUTPUT_SUFFIXES, ".centromere.tsv"):
        f1 = Path(f"{bun2_t1.prefix}{suffix}")
        f4 = Path(f"{bun2_t4.prefix}{suffix}")
        assert f1.exists() and f4.exists(), f"Missing output {suffix}"
        assert _sha256_file(f1) == _sha256_file(f4), f"T22 mismatch on {suffix}"

    # --- Subcase 3a: Repeat-Guard Wave 3 CLI (5 pure-INS slots) ---
    b3a = SyntheticInputBuilder(tmp_path / "w3_rg_a", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 7):
        b3a.add_node(f"r{i}", make_complex_seq(100, seed=8300 + i))

    # Slot 1 (R2-r3 F1): two completely unrelated 50 bp sequences -> 2 alleles (50 + 100 + 50 = 200)
    b3a.add_node("s1_a", make_complex_seq(50, seed=605008))
    b3a.add_node("s1_b", make_complex_seq(50, seed=505008))

    # Slot 2 (R2-r3 F2): tail(100)+U(400)+fl(1) vs U(400)+ov(50) (ov[0]!=fl[0]) -> 2 alleles (501+100+450=1051)
    u400 = make_complex_seq(400, seed=80000)
    tail100 = make_complex_seq(100, seed=81000)
    fl1 = make_complex_seq(1, seed=82100)
    ov50 = make_complex_seq(50, seed=89000)
    if ov50[0] == fl1[0]:
        ov50 = ("A" if fl1[0] != "A" else "C") + ov50[1:]
    b3a.add_node("s2_r", tail100 + u400 + fl1)
    b3a.add_node("s2_s", u400 + ov50)

    # Slot 3 (R2-r3 F3): 850 bp flanked rep vs 830 bp dispersed sp=13 SNPs + 30I -> 1 merged allele (850)
    r800 = make_complex_seq(800, seed=641300)
    rm800 = mutate_seq_snps(r800, positions=list(range(230, 345, 13)) + list(range(355, 470, 13)))
    b3a.add_node("s3_r", make_complex_seq(25, seed=641301) + r800 + make_complex_seq(25, seed=641302))
    b3a.add_node("s3_s", rm800[:350] + make_complex_seq(30, seed=651300) + rm800[350:])

    # Slot 4 (C1-r3 B1): 305 bp base vs 260 bp pure 45 bp DEL -> 1 merged allele (305)
    b305 = make_complex_seq(305, seed=83400)
    b3a.add_node("s4_r", b305)
    b3a.add_node("s4_s", b305[:130] + b305[175:])

    # Slot 5 (C1-r3 B3): fl(150)+pal(200)+fl(150) vs pal(200)+ov(30) vs pal(200) -> 3 distinct alleles (1130)
    h100 = make_complex_seq(100, seed=53002)
    pal200 = h100 + revcomp(h100)
    b3a.add_node("s5_rep", make_complex_seq(150, seed=51002) + pal200 + make_complex_seq(150, seed=52002))
    b3a.add_node("s5_ov30", pal200 + make_complex_seq(30, seed=54002))
    b3a.add_node("s5_pal", pal200)

    b3a.set_ref_path("Ref#1#Chr01", [f"r{i}+" for i in range(1, 7)])
    b3a.add_path(
        "AltA#1#Chr01",
        ["r1+", "s1_a+", "r2+", "s2_r+", "r3+", "s3_r+", "r4+", "s4_r+", "r5+", "s5_rep+", "r6+"],
    )
    b3a.add_path(
        "AltB#1#Chr01",
        ["r1+", "s1_b+", "r2+", "s2_s+", "r3+", "s3_s+", "r4+", "s4_s+", "r5+", "s5_ov30+", "r6+"],
    )
    b3a.add_path(
        "AltC#1#Chr01",
        ["r1+", "r2+", "r3+", "r4+", "r5+", "s5_pal+", "r6+"],
    )
    inp3a = b3a.write("w3_rg_a")
    p3a, bun3a = run_build(inp3a)
    assert p3a.returncode == 0 and bun3a is not None
    run_verify(inp3a.out_prefix, inp3a.gfa_path, inp3a.ref_fa_path, check=True)
    assert_all_global_invariants(bun3a, inp3a.gfa_path, inp3a.ref_fa_path)
    assert bun3a.offsets[1]["slot_ins_len"] == 200
    assert bun3a.offsets[2]["slot_ins_len"] == 1051
    assert bun3a.offsets[3]["slot_ins_len"] == 850
    assert bun3a.offsets[4]["slot_ins_len"] == 305
    assert bun3a.offsets[5]["slot_ins_len"] == 1130
    ins_s5 = next(r for r in bun3a.sv_rows if r["type"] == "INS" and r["anchors"] == "4..5")
    gt_s5 = bun3a.genotypes[ins_s5["event_id"]]
    assert len({gt_s5["AltA#1"], gt_s5["AltB#1"], gt_s5["AltC#1"]}) == 3

    # --- Subcase 3b: Repeat-Guard Wave 3 CLI (Case A / T5 backbone: C2-r3 B1 & C1-r3 B1) ---
    b3b = SyntheticInputBuilder(tmp_path / "w3_rg_b", chrom="Chr01", backbone_prefix="Ref#1#")
    b3b.add_node("a1", make_complex_seq(150, seed=8401))
    h120 = make_complex_seq(120, seed=91000)
    pal240 = h120 + revcomp(h120)
    bb_pal = make_complex_seq(80, seed=91001) + pal240 + make_complex_seq(100, seed=91002)  # 420 bp
    b3b.add_node("bb_a", bb_pal)
    b3b.add_node("del40_a", bb_pal[40:])  # 380 bp: pure 40 bp < 50 bp DEL near internal palindrome
    b3b.add_node("ins_a", make_complex_seq(180, seed=91003))
    b3b.add_node("a2", make_complex_seq(150, seed=8402))
    bb_205 = make_complex_seq(205, seed=82600)
    b3b.add_node("bb_b", bb_205)
    b3b.add_node("del45_b", bb_205[:80] + bb_205[125:])  # 160 bp: pure 45 bp < 50 bp DEL
    b3b.add_node("ins_b", make_complex_seq(180, seed=82601))
    b3b.add_node("a3", make_complex_seq(150, seed=8403))

    b3b.set_ref_path("Ref#1#Chr01", ["a1+", "bb_a+", "a2+", "bb_b+", "a3+"])
    b3b.add_path("AltDel#1#Chr01", ["a1+", "del40_a+", "a2+", "del45_b+", "a3+"])
    b3b.add_path("AltTrueIns#1#Chr01", ["a1+", "ins_a+", "a2+", "ins_b+", "a3+"])
    inp3b = b3b.write("w3_rg_b")
    p3b, bun3b = run_build(inp3b)
    assert p3b.returncode == 0 and bun3b is not None
    run_verify(inp3b.out_prefix, inp3b.gfa_path, inp3b.ref_fa_path, check=True)
    assert_all_global_invariants(bun3b, inp3b.gfa_path, inp3b.ref_fa_path)
    ins_a = next(r for r in bun3b.sv_rows if r["type"] == "INS" and r["anchors"] == "0..2")
    ins_b = next(r for r in bun3b.sv_rows if r["type"] == "INS" and r["anchors"] == "2..4")
    assert bun3b.genotypes[ins_a["event_id"]]["AltDel#1"] == "0"
    assert bun3b.genotypes[ins_b["event_id"]]["AltDel#1"] == "0"
    assert bun3b.offsets[2]["slot_ins_len"] == 280  # 100 bp replacement spacer + 180 bp novel ins (0 bp bloat)
    assert bun3b.offsets[4]["slot_ins_len"] == 280  # 100 bp replacement spacer + 180 bp novel ins (0 bp bloat)


def test_tier5_rf09_wave4_repeat_guard_and_extract_islands_cli(tmp_path: Path) -> None:
    """Wave 4 / Iteration 5 strictly opaque-box CLI E2E guard: V10 extract-islands + Bugs 1/2/3 + T22 SHA-256."""
    import random

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    # --- Subcase 1: Challenger 2 V10 extract-islands (598 vs 500 bp 98 bp dual DEL -> 2 CEN_ISLANDs + T22 fork) ---
    b_v10 = SyntheticInputBuilder(tmp_path / "w4_v10", chrom="ChrX", backbone_prefix="Ref#1#")
    r598 = make_complex_seq(45 + 49 + 410 + 49 + 45, seed=60101)
    s500 = r598[:45] + r598[94:504] + r598[553:]
    b_v10.add_node("1", make_complex_seq(150, seed=60201))
    b_v10.add_node("2", make_complex_seq(400, seed=60202))
    b_v10.add_node("3", make_complex_seq(400, seed=60203))
    b_v10.add_node("4", make_complex_seq(150, seed=60204))
    b_v10.add_node("5", make_complex_seq(150, seed=60205))
    b_v10.add_node("6", r598)
    b_v10.add_node("7", s500)
    b_v10.set_ref_path("Ref#1#ChrX", ["1+", "2+", "3+", "4+", "5+"])
    b_v10.add_bed_region(160, 940, "CENX")
    b_v10.add_path("Alt598#1#ChrX", ["1+", "6+", "4+", "5+"])
    b_v10.add_path("Alt500#1#ChrX", ["1+", "7+", "4+", "5+"])
    inp_v10 = b_v10.write("w4_v10")

    pv_t1, bun_v10_t1 = run_build(
        inp_v10,
        out_prefix=tmp_path / "w4_v10" / "out_t1",
        extra_args=["--centromere-mode", "extract-islands", "--threads", "1"],
    )
    pv_t4, bun_v10_t4 = run_build(
        inp_v10,
        out_prefix=tmp_path / "w4_v10" / "out_t4",
        extra_args=["--centromere-mode", "extract-islands", "--threads", "4"],
        extra_env={"PANFLAT_FORCE_FORK": "1"},
    )
    assert pv_t1.returncode == 0 and bun_v10_t1 is not None
    assert pv_t4.returncode == 0 and bun_v10_t4 is not None
    assert inp_v10.bed_path is not None
    run_verify(
        bun_v10_t1.prefix,
        inp_v10.gfa_path,
        inp_v10.ref_fa_path,
        extra_args=["--centromere-bed", inp_v10.bed_path],
        check=True,
    )
    run_verify(
        bun_v10_t4.prefix,
        inp_v10.gfa_path,
        inp_v10.ref_fa_path,
        extra_args=["--centromere-bed", inp_v10.bed_path],
        check=True,
    )
    assert_all_global_invariants(bun_v10_t1, inp_v10.gfa_path, inp_v10.ref_fa_path)

    cen_evs = [r for r in bun_v10_t1.sv_rows if r["type"] == "CEN_ISLAND"]
    assert len(cen_evs) == 2, f"Expected 2 CEN_ISLAND events (98 bp dual DEL), got {len(cen_evs)}: {cen_evs}"
    assert any("len=598" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
    assert any("len=500" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
    for suffix in (*_CORE_OUTPUT_SUFFIXES, ".centromere.tsv"):
        f1 = Path(f"{bun_v10_t1.prefix}{suffix}")
        f4 = Path(f"{bun_v10_t4.prefix}{suffix}")
        assert f1.exists() and f4.exists(), f"Missing output {suffix}"
        assert _sha256_file(f1) == _sha256_file(f4), f"T22 mismatch on V10 {suffix}"

    # --- Subcase 2: Wave 4 Repeat-Guard CLI (Bugs 1, 2, 3 fwd/RC + 48D control + T22 fork) ---
    b_rg = SyntheticInputBuilder(tmp_path / "w4_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 7):
        b_rg.add_node(f"a{i}", make_complex_seq(150, seed=9100 + i))

    # Slot 1 (Bug 1): 288 vs 190 (98 bp dual DEL) -> 2 distinct alleles (288 + 100 + 190 = 578)
    r288 = make_complex_seq(45 + 49 + 100 + 49 + 45, seed=324945)
    b_rg.add_node("s1_r", r288)
    b_rg.add_node("s1_s", r288[:45] + r288[94:194] + r288[243:])

    # Slot 2 (Bug 2): pal160 + 45D + 1 SNP vs exact pal160 vs flanked rep -> 3 alleles (505 + 100 + 160 + 100 + 160 = 1025)
    h80 = make_complex_seq(80, seed=220160)
    pal160 = h80 + revcomp(h80)
    b_rg.add_node(
        "s2_r",
        make_complex_seq(150, seed=222000)
        + pal160[:80]
        + make_complex_seq(45, seed=221045)
        + pal160[80:]
        + make_complex_seq(150, seed=223000),
    )
    b_rg.add_node("s2_1snp", mutate_seq_snps(pal160, positions=[20]))
    b_rg.add_node("s2_exact", pal160)

    # Slot 3 (Bug 3 fwd): random 0-SNP dup D1!=D2 (358 bp) vs U1+U2 (100 bp) -> 2 alleles (358 + 100 + 100 = 558)
    u1 = make_complex_seq(50, seed=120504)
    u2 = make_complex_seq(50, seed=121504)
    sep60 = make_complex_seq(60, seed=122004)
    b_rg.add_node(
        "s3_r",
        u1 + make_complex_seq(49, seed=123004) + u2 + sep60 + u1 + make_complex_seq(49, seed=124004) + u2,
    )
    b_rg.add_node("s3_s", u1 + u2)

    # Slot 4 (Bug 3 RC): RC dup D1!=D2 (358 bp) vs U1+U2 (100 bp) -> 2 alleles (358 + 100 + 100 = 558)
    b_rg.add_node(
        "s4_r",
        u1 + _mk_biased(49, "AT", 125004) + u2 + sep60 + revcomp(u1 + _mk_biased(49, "GC", 126004) + u2),
    )
    b_rg.add_node("s4_s", u1 + u2)

    # Slot 5 (Positive Control): 48 bp dual DEL (228 vs 180 bp) -> 1 merged allele (228)
    r228 = make_complex_seq(40 + 24 + 100 + 24 + 40, seed=142424)
    b_rg.add_node("s5_r", r228)
    b_rg.add_node("s5_s", r228[:40] + r228[64:164] + r228[188:])

    b_rg.set_ref_path("Ref#1#Chr01", [f"a{i}+" for i in range(1, 7)])
    b_rg.add_path(
        "AltRep#1#Chr01",
        ["a1+", "s1_r+", "a2+", "s2_r+", "a3+", "s3_r+", "a4+", "s4_r+", "a5+", "s5_r+", "a6+"],
    )
    b_rg.add_path(
        "AltCand#1#Chr01",
        ["a1+", "s1_s+", "a2+", "s2_1snp+", "a3+", "s3_s+", "a4+", "s4_s+", "a5+", "s5_s+", "a6+"],
    )
    b_rg.add_path(
        "AltPalEx#1#Chr01",
        ["a1+", "a2+", "s2_exact+", "a3+", "a4+", "a5+", "a6+"],
    )
    inp_rg = b_rg.write("w4_rg")

    prg_t1, bun_rg_t1 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w4_rg" / "out_t1",
        extra_args=["--threads", "1"],
    )
    prg_t4, bun_rg_t4 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w4_rg" / "out_t4",
        extra_args=["--threads", "4"],
        extra_env={"PANFLAT_FORCE_FORK": "1"},
    )
    assert prg_t1.returncode == 0 and bun_rg_t1 is not None
    assert prg_t4.returncode == 0 and bun_rg_t4 is not None
    run_verify(bun_rg_t1.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    run_verify(bun_rg_t4.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    assert_all_global_invariants(bun_rg_t1, inp_rg.gfa_path, inp_rg.ref_fa_path)

    assert bun_rg_t1.offsets[1]["slot_ins_len"] == 578
    assert bun_rg_t1.offsets[2]["slot_ins_len"] == 1025
    assert bun_rg_t1.offsets[3]["slot_ins_len"] == 558
    assert bun_rg_t1.offsets[4]["slot_ins_len"] == 558
    assert bun_rg_t1.offsets[5]["slot_ins_len"] == 228

    ins_s2 = next(r for r in bun_rg_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "1..2")
    gt_s2 = bun_rg_t1.genotypes[ins_s2["event_id"]]
    assert len({gt_s2["AltRep#1"], gt_s2["AltCand#1"], gt_s2["AltPalEx#1"]}) == 3
    ins_s5 = next(r for r in bun_rg_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "4..5")
    gt_s5 = bun_rg_t1.genotypes[ins_s5["event_id"]]
    assert gt_s5["AltRep#1"] == gt_s5["AltCand#1"] == "1"

    for suffix in _CORE_OUTPUT_SUFFIXES:
        f1 = Path(f"{bun_rg_t1.prefix}{suffix}")
        f4 = Path(f"{bun_rg_t4.prefix}{suffix}")
        assert f1.exists() and f4.exists(), f"Missing output {suffix}"
        assert _sha256_file(f1) == _sha256_file(f4), f"T22 mismatch on w4_rg {suffix}"


def test_tier5_rf10_wave5_repeat_guard_parity_and_stage6d_cli(tmp_path: Path) -> None:
    """Wave 5 / Iteration 6 strictly opaque-box CLI E2E guard: extract-islands + Case A rep0 1001 bp + Slots 1-9 + T22."""
    import random

    def _mk_biased(n: int, alphabet: str, seed: int) -> str:
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(n))

    def _run_extract_islands_subcase(
        tag: str,
        seq_r: str,
        seq_s: str,
        extra_cli: list[str] | None = None,
    ) -> None:
        b = SyntheticInputBuilder(tmp_path / tag, chrom="ChrX", backbone_prefix="Ref#1#")
        b.add_node("1", make_complex_seq(150, seed=70101))
        b.add_node("2", make_complex_seq(400, seed=70102))
        b.add_node("3", make_complex_seq(400, seed=70103))
        b.add_node("4", make_complex_seq(150, seed=70104))
        b.add_node("5", make_complex_seq(150, seed=70105))
        b.add_node("6", seq_r)
        b.add_node("7", seq_s)
        b.set_ref_path("Ref#1#ChrX", ["1+", "2+", "3+", "4+", "5+"])
        b.add_bed_region(160, 940, "CENX")
        b.add_path("AltR#1#ChrX", ["1+", "6+", "4+", "5+"])
        b.add_path("AltS#1#ChrX", ["1+", "7+", "4+", "5+"])
        inp = b.write(tag)
        base_args = ["--centromere-mode", "extract-islands"] + (extra_cli or [])

        p1, bun1 = run_build(
            inp,
            out_prefix=tmp_path / tag / "out_t1",
            extra_args=base_args + ["--threads", "1"],
        )
        p4, bun4 = run_build(
            inp,
            out_prefix=tmp_path / tag / "out_t4",
            extra_args=base_args + ["--threads", "4"],
            extra_env={"PANFLAT_FORCE_FORK": "1"},
        )
        assert p1.returncode == 0 and bun1 is not None, f"{tag} t1 failed: {p1.stderr}"
        assert p4.returncode == 0 and bun4 is not None, f"{tag} t4 failed: {p4.stderr}"
        assert inp.bed_path is not None
        run_verify(
            bun1.prefix,
            inp.gfa_path,
            inp.ref_fa_path,
            extra_args=["--centromere-bed", inp.bed_path],
            check=True,
        )
        run_verify(
            bun4.prefix,
            inp.gfa_path,
            inp.ref_fa_path,
            extra_args=["--centromere-bed", inp.bed_path],
            check=True,
        )
        assert_all_global_invariants(bun1, inp.gfa_path, inp.ref_fa_path)

        cen_evs = [r for r in bun1.sv_rows if r["type"] == "CEN_ISLAND"]
        assert len(cen_evs) == 2, f"{tag}: expected 2 CEN_ISLANDs, got {len(cen_evs)}: {cen_evs}"
        assert any(f"len={len(seq_r)}" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
        assert any(f"len={len(seq_s)}" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
        for suffix in (*_CORE_OUTPUT_SUFFIXES, ".centromere.tsv"):
            f1 = Path(f"{bun1.prefix}{suffix}")
            f4 = Path(f"{bun4.prefix}{suffix}")
            assert f1.exists() and f4.exists(), f"{tag}: missing output {suffix}"
            assert _sha256_file(f1) == _sha256_file(f4), f"{tag}: T22 SHA-256 mismatch on {suffix}"

    # --- Subcase 1: extract-islands Wave 5 CLI (Bug 1, Bug 2, Bug 4) ---
    # 1a. Bug 1 (default min-island 500): 710 bp vs 515 bp (45D+45D=90D + 15 bp outer overhang)
    b1_al = make_complex_seq(30, seed=71001)
    b1_d1 = make_complex_seq(45, seed=71002)
    b1_mid = make_complex_seq(440, seed=71003)
    b1_d2 = make_complex_seq(45, seed=71004)
    b1_ar = make_complex_seq(30, seed=71005)
    b1_ov = make_complex_seq(15, seed=71006)
    b1_fl = make_complex_seq(60, seed=71007)
    _run_extract_islands_subcase(
        "w5_cen_bug1",
        b1_fl + b1_al + b1_d1 + b1_mid + b1_d2 + b1_ar + b1_fl,
        b1_al + b1_mid + b1_ar + b1_ov,
    )

    # 1b. Bug 2 (default min-island 500): 1001 bp vs 903 bp (49D+49D=98D in mappy)
    p1_m = make_complex_seq(250, seed=96300)
    p2_m = make_complex_seq(250, seed=96301)
    p3_m = make_complex_seq(403, seed=96302)
    dd1_m = make_complex_seq(49, seed=96303)
    dd2_m = make_complex_seq(49, seed=96304)
    _run_extract_islands_subcase(
        "w5_cen_bug2",
        p1_m + dd1_m + p2_m + dd2_m + p3_m,
        p1_m + p2_m + p3_m,
    )

    # 1c. Bug 4 (--centromere-min-island 150): 460 bp vs 185 bp (I30 + D1!=D2)
    u1a_c = make_complex_seq(35, seed=40000)
    u1b_c = make_complex_seq(60, seed=40100)
    u2_c = make_complex_seq(60, seed=40200)
    ins30_c = make_complex_seq(30, seed=40300)
    sep_c = make_complex_seq(60, seed=40400)
    da45_c = _mk_biased(45, "AT", 40500)
    dg45_c = _mk_biased(45, "GC", 40600)
    _run_extract_islands_subcase(
        "w5_cen_bug4",
        u1a_c + u1b_c + da45_c + u2_c + sep_c + u1a_c + u1b_c + dg45_c + u2_c,
        u1a_c + ins30_c + u1b_c + u2_c,
        extra_cli=["--centromere-min-island", "150"],
    )

    # --- Subcase 2: Normal INS + Case A rep0 1001 bp across Slots 1-9 + T22 fork ---
    b_rg = SyntheticInputBuilder(tmp_path / "w5_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 11):
        b_rg.add_node(f"a{i}", make_complex_seq(150, seed=9800 + i))

    # Slot 1 (Case A rep0 1001 bp between a1 & a2): AltDel98 (903 bp -> allele 1, gt='1'), AltDel48 (953 bp -> gt='0'), AltIns (200 bp -> allele 2, gt='2')
    bb1001 = p1_m + dd1_m + p2_m + dd2_m + p3_m
    assert len(bb1001) == 1001
    b_rg.add_node("bb1001", bb1001)
    b_rg.add_node("s1_del98", p1_m + p2_m + p3_m)
    b_rg.add_node("s1_del48", bb1001[:250] + bb1001[274:524] + bb1001[548:])
    b_rg.add_node("s1_ins200", make_complex_seq(200, seed=96310))

    # Slot 2 (Bug 1, 470 vs 275 bp ov15): 2 distinct alleles (470 + 100 + 275 = 845)
    s2_al = make_complex_seq(30, seed=15000)
    s2_d1 = make_complex_seq(45, seed=15100)
    s2_mid = make_complex_seq(200, seed=15200)
    s2_d2 = make_complex_seq(45, seed=15300)
    s2_ar = make_complex_seq(30, seed=15400)
    s2_ov = make_complex_seq(15, seed=15500)
    s2_fl = make_complex_seq(60, seed=15600)
    b_rg.add_node("s2_r", s2_fl + s2_al + s2_d1 + s2_mid + s2_d2 + s2_ar + s2_fl)
    b_rg.add_node("s2_s", s2_al + s2_mid + s2_ar + s2_ov)

    # Slot 3 (R1-r5 Major F1, AltPalSnp0 + AltPalSnp20 vs 505 bp AltRep): AltRep stays separate (len=505, n=1)
    h80 = make_complex_seq(80, seed=220160)
    pal160 = h80 + revcomp(h80)
    b_rg.add_node(
        "s3_r",
        make_complex_seq(150, seed=222000)
        + pal160[:80]
        + make_complex_seq(45, seed=221045)
        + pal160[80:]
        + make_complex_seq(150, seed=223000),
    )
    b_rg.add_node("s3_snp0", mutate_seq_snps(pal160, positions=[0]))
    b_rg.add_node("s3_snp20", mutate_seq_snps(pal160, positions=[20]))

    # Slot 4 & 5 (Bug 3 K2 fwd & RC, 440 vs 180 bp): 2 distinct alleles each (440 + 100 + 180 = 720)
    u1_k2, u2_k2 = make_complex_seq(90, seed=141001), make_complex_seq(90, seed=141002)
    d40_k2, sep40_k2 = make_complex_seq(40, seed=141003), make_complex_seq(40, seed=141004)
    b_rg.add_node("s4_r", u1_k2 + u2_k2 + sep40_k2 + u1_k2 + d40_k2 + u2_k2)
    b_rg.add_node("s4_s", u1_k2 + u2_k2)
    b_rg.add_node("s5_r", u1_k2 + u2_k2 + sep40_k2 + revcomp(u1_k2 + d40_k2 + u2_k2))
    b_rg.add_node("s5_s", u1_k2 + u2_k2)

    # Slot 6 (Bug 4 I30 + D1!=D2, 460 vs 185 bp): 2 distinct alleles (460 + 100 + 185 = 745)
    b_rg.add_node("s6_r", u1a_c + u1b_c + da45_c + u2_c + sep_c + u1a_c + u1b_c + dg45_c + u2_c)
    b_rg.add_node("s6_s", u1a_c + ins30_c + u1b_c + u2_c)

    # Slot 7 (Bug 5 motif_len=4, 362 vs 100 bp): 2 distinct alleles (362 + 100 + 100 = 562)
    motif4 = make_complex_seq(4, seed=50024)
    u1_per4 = (motif4 * 13)[:50]
    u2_b5 = make_complex_seq(50, seed=52002)
    b_rg.add_node(
        "s7_r",
        u1_per4
        + _mk_biased(49, "AT", 53002)
        + u2_b5
        + make_complex_seq(60, seed=55002)
        + motif4
        + u1_per4
        + _mk_biased(49, "GC", 54002)
        + u2_b5,
    )
    b_rg.add_node("s7_s", u1_per4 + u2_b5)

    # Slot 8 (R2-r5 F3 U1+D1+U1, 358 vs 100 bp): 2 distinct alleles (358 + 100 + 100 = 558)
    u1_f3 = make_complex_seq(50, seed=215000 + 50 * 100)
    u1_f3a = mutate_seq_snps(u1_f3, positions=[10])
    u1_f3b = mutate_seq_snps(u1_f3, positions=[10, 30])
    b_rg.add_node(
        "s8_r",
        u1_f3
        + make_complex_seq(49, seed=275000)
        + u1_f3
        + make_complex_seq(60, seed=225000)
        + u1_f3a
        + _mk_biased(49, "GGCC", 285000)
        + u1_f3b,
    )
    b_rg.add_node("s8_s", u1_f3 + u1_f3)

    # Slot 9 (Positive Red-Line Control 200=30I200=, 430 bp vs 400 bp): 1 merged allele (len=430, n=2, slot_ins_len == 430)
    r400_rl = make_complex_seq(400, seed=969001)
    b_rg.add_node("s9_r400", r400_rl)
    b_rg.add_node("s9_s430", r400_rl[:200] + make_complex_seq(30, seed=969002) + r400_rl[200:])

    ref_tail = [f"a{i}+" for i in range(2, 11)]
    b_rg.set_ref_path("Ref#1#Chr01", ["a1+", "bb1001+"] + ref_tail)
    b_rg.add_path("AltDel98#1#Chr01", ["a1+", "s1_del98+"] + ref_tail)
    b_rg.add_path("AltDel48#1#Chr01", ["a1+", "s1_del48+"] + ref_tail)
    b_rg.add_path("AltIns#1#Chr01", ["a1+", "s1_ins200+"] + ref_tail)
    b_rg.add_path(
        "AltRep#1#Chr01",
        [
            "a1+", "bb1001+", "a2+", "s2_r+", "a3+", "s3_r+", "a4+", "s4_r+",
            "a5+", "s5_r+", "a6+", "s6_r+", "a7+", "s7_r+", "a8+", "s8_r+",
            "a9+", "s9_s430+", "a10+",
        ],
    )
    b_rg.add_path(
        "AltCand#1#Chr01",
        [
            "a1+", "bb1001+", "a2+", "s2_s+", "a3+", "s3_snp0+", "a4+", "s4_s+",
            "a5+", "s5_s+", "a6+", "s6_s+", "a7+", "s7_s+", "a8+", "s8_s+",
            "a9+", "s9_r400+", "a10+",
        ],
    )
    b_rg.add_path(
        "AltPal20#1#Chr01",
        [
            "a1+", "bb1001+", "a2+", "a3+", "s3_snp20+", "a4+", "a5+",
            "a6+", "a7+", "a8+", "a9+", "a10+",
        ],
    )
    inp_rg = b_rg.write("w5_rg")

    prg_t1, bun_t1 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w5_rg" / "out_t1",
        extra_args=["--threads", "1"],
    )
    prg_t4, bun_t4 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w5_rg" / "out_t4",
        extra_args=["--threads", "4"],
        extra_env={"PANFLAT_FORCE_FORK": "1"},
    )
    assert prg_t1.returncode == 0 and bun_t1 is not None, f"w5_rg t1 failed: {prg_t1.stderr}"
    assert prg_t4.returncode == 0 and bun_t4 is not None, f"w5_rg t4 failed: {prg_t4.stderr}"
    run_verify(bun_t1.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    run_verify(bun_t4.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    assert_all_global_invariants(bun_t1, inp_rg.gfa_path, inp_rg.ref_fa_path)

    # Slot 1 (Bug 2 Case A rep0 1001 bp): AltDel98 preserved as allele 1 (len=903, gt='1'), AltIns='2', AltDel48='0', Ref='0'
    ins_s1 = next(r for r in bun_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "0..2")
    assert "len=903" in ins_s1["alleles"] and "n=1" in ins_s1["alleles"]
    gt_s1 = bun_t1.genotypes[ins_s1["event_id"]]
    assert gt_s1["Ref#1"] == "0"
    assert gt_s1["AltDel48#1"] == "0"
    assert gt_s1["AltDel98#1"] == "1"
    assert gt_s1["AltIns#1"] == "2"

    # Slots 2, 4, 5, 6, 7, 8: exact 2-allele slot lengths (rep + 100 bp spacer + cand)
    assert bun_t1.offsets[3]["slot_ins_len"] == 845   # Slot 2 (Bug 1: 470 + 100 + 275)
    assert bun_t1.offsets[5]["slot_ins_len"] == 720   # Slot 4 (Bug 3 K2 fwd: 440 + 100 + 180)
    assert bun_t1.offsets[6]["slot_ins_len"] == 720   # Slot 5 (Bug 3 K2 RC: 440 + 100 + 180)
    assert bun_t1.offsets[7]["slot_ins_len"] == 745   # Slot 6 (Bug 4 I30+D1!=D2: 460 + 100 + 185)
    assert bun_t1.offsets[8]["slot_ins_len"] == 562   # Slot 7 (Bug 5 motif_len=4: 362 + 100 + 100)
    assert bun_t1.offsets[9]["slot_ins_len"] == 558   # Slot 8 (R2-r5 F3 U1+D1+U1: 358 + 100 + 100)

    # Slot 3 (R1-r5 Major F1): AltRep#1 stays separate (len=505, n=1) and gt[AltPalSnp0] != gt[AltRep]
    ins_s3 = next(r for r in bun_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "3..4")
    assert "len=505" in ins_s3["alleles"] and "src=AltRep#1,n=1" in ins_s3["alleles"]
    gt_s3 = bun_t1.genotypes[ins_s3["event_id"]]
    assert gt_s3["AltCand#1"] != gt_s3["AltRep#1"]
    assert gt_s3["AltPal20#1"] != gt_s3["AltRep#1"]

    # Slot 9 (Positive Red-Line Control 200=30I200=): single merged 430 bp allele (n=2)
    assert bun_t1.offsets[10]["slot_ins_len"] == 430
    ins_s9 = next(r for r in bun_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "9..10")
    assert "len=430" in ins_s9["alleles"] and "n=2" in ins_s9["alleles"]
    gt_s9 = bun_t1.genotypes[ins_s9["event_id"]]
    assert gt_s9["AltRep#1"] == gt_s9["AltCand#1"] == "1"
    assert gt_s9["AltPal20#1"] == "0"

    # T22 strict determinism across all 10 core output files
    for suffix in _CORE_OUTPUT_SUFFIXES:
        f1 = Path(f"{bun_t1.prefix}{suffix}")
        f4 = Path(f"{bun_t4.prefix}{suffix}")
        assert f1.exists() and f4.exists(), f"Missing output {suffix}"
        assert _sha256_file(f1) == _sha256_file(f4), f"T22 mismatch on w5_rg {suffix}"


def test_tier5_rf11_wave6_stage4_stage6e_and_extract_islands_cli(tmp_path: Path) -> None:
    """Wave 6 / Iteration 7 strictly opaque-box CLI E2E guard: extract-islands (Bugs 1-3) + Slots 1-6 + T22 fork."""

    def _run_extract_islands_subcase(
        tag: str,
        seq_r: str,
        seq_s: str,
        extra_cli: list[str] | None = None,
    ) -> None:
        b = SyntheticInputBuilder(tmp_path / tag, chrom="ChrX", backbone_prefix="Ref#1#")
        b.add_node("1", make_complex_seq(150, seed=76101))
        b.add_node("2", make_complex_seq(400, seed=76102))
        b.add_node("3", make_complex_seq(400, seed=76103))
        b.add_node("4", make_complex_seq(150, seed=76104))
        b.add_node("5", make_complex_seq(150, seed=76105))
        b.add_node("6", seq_r)
        b.add_node("7", seq_s)
        b.set_ref_path("Ref#1#ChrX", ["1+", "2+", "3+", "4+", "5+"])
        b.add_bed_region(160, 940, "CENX")
        b.add_path("AltR#1#ChrX", ["1+", "6+", "4+", "5+"])
        b.add_path("AltS#1#ChrX", ["1+", "7+", "4+", "5+"])
        inp = b.write(tag)
        base_args = ["--centromere-mode", "extract-islands"] + (extra_cli or [])

        p1, bun1 = run_build(
            inp,
            out_prefix=tmp_path / tag / "out_t1",
            extra_args=base_args + ["--threads", "1"],
        )
        p4, bun4 = run_build(
            inp,
            out_prefix=tmp_path / tag / "out_t4",
            extra_args=base_args + ["--threads", "4"],
            extra_env={"PANFLAT_FORCE_FORK": "1"},
        )
        assert p1.returncode == 0 and bun1 is not None, f"{tag} t1 failed: {p1.stderr}"
        assert p4.returncode == 0 and bun4 is not None, f"{tag} t4 failed: {p4.stderr}"
        assert inp.bed_path is not None
        run_verify(
            bun1.prefix,
            inp.gfa_path,
            inp.ref_fa_path,
            extra_args=["--centromere-bed", inp.bed_path],
            check=True,
        )
        run_verify(
            bun4.prefix,
            inp.gfa_path,
            inp.ref_fa_path,
            extra_args=["--centromere-bed", inp.bed_path],
            check=True,
        )
        assert_all_global_invariants(bun1, inp.gfa_path, inp.ref_fa_path)

        assert len(bun1.unresolved) == 0, f"{tag}: expected 0 unresolved, got {bun1.unresolved}"
        cen_evs = [r for r in bun1.sv_rows if r["type"] == "CEN_ISLAND"]
        assert len(cen_evs) == 2, f"{tag}: expected 2 CEN_ISLANDs, got {len(cen_evs)}: {cen_evs}"
        assert any(f"len={len(seq_r)}" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
        assert any(f"len={len(seq_s)}" in r["alleles"] and "n=1" in r["alleles"] for r in cen_evs)
        for suffix in (*_CORE_OUTPUT_SUFFIXES, ".centromere.tsv"):
            f1 = Path(f"{bun1.prefix}{suffix}")
            f4 = Path(f"{bun4.prefix}{suffix}")
            assert f1.exists() and f4.exists(), f"{tag}: missing output {suffix}"
            assert _sha256_file(f1) == _sha256_file(f4), f"{tag}: T22 SHA-256 mismatch on {suffix}"

    # --- Shared Wave 6 scenario sequences (ALL len_r <= 1000 bp) ---
    # Scenario 1: C1-r6 NEW Bug 1 (750 vs 515 bp, default 500 bp extract-islands: 45D+45D=90D + 15 bp inner junc-ins)
    b1_fl_l = make_complex_seq(80, seed=771001)
    b1_fl_r = make_complex_seq(80, seed=771002)
    b1_al = make_complex_seq(30, seed=772030)
    b1_d1 = make_complex_seq(45, seed=774045)
    b1_core = make_complex_seq(440, seed=776440)
    b1_d2 = make_complex_seq(45, seed=775045)
    b1_ar = make_complex_seq(30, seed=773030)
    b1_ins15 = make_complex_seq(15, seed=778015)
    s1_r = b1_fl_l + b1_al + b1_d1 + b1_core + b1_d2 + b1_ar + b1_fl_r
    s1_s = b1_al + b1_core + b1_ins15 + b1_ar
    assert len(s1_r) == 750 and len(s1_s) == 515

    # Scenario 2: R1-r6 Major F1 (859 vs 240 bp, K2_trunc10_L with distinct 150 bp flanks)
    r1_fl_l = make_complex_seq(150, seed=771151)
    r1_fl_r = make_complex_seq(150, seed=771152)
    r1_u1 = make_complex_seq(120, seed=771000)
    r1_u2 = make_complex_seq(120, seed=772000)
    r1_d49 = make_complex_seq(49, seed=773000)
    r1_sep = make_complex_seq(40, seed=774000)
    s2_s = r1_u1 + r1_u2
    s2_r = r1_fl_l + s2_s + r1_sep + r1_u1[10:] + r1_d49 + r1_u2 + r1_fl_r
    assert len(s2_r) == 859 and len(s2_s) == 240

    # Scenario 3: R2-r6 Major F1 (735 vs 180 bp, 3A-4 exact-tip starvation)
    lp5 = "ACGTA" * 3 + "ACGT"
    rp5 = "TCGAG" * 3 + "TCGA"
    r2_u1 = lp5 + make_complex_seq(71, seed=550000)
    r2_u2 = make_complex_seq(71, seed=551000) + rp5
    r2_d45 = make_complex_seq(45, seed=552000)
    r2_sep = make_complex_seq(40, seed=553000)
    r2_dist = "ACGTA" * 12 + make_complex_seq(130, seed=554000) + "TCGAG" * 12
    s3_s = r2_u1 + r2_u2
    s3_r = s3_s + r2_sep + r2_dist + r2_sep + r2_u1 + r2_d45 + r2_u2
    assert len(s3_r) == 735 and len(s3_s) == 180

    # Scenario 4: C1-r6 NEW Bug 2 (884 vs 240 bp, Case B 1-SNP tip + 20 bp ed=0 decoy, distinct 150 bp flanks)
    at20 = "AT" * 10
    b2_fl_l = make_complex_seq(150, seed=613151)
    b2_fl_r = make_complex_seq(150, seed=613152)
    b2_u1 = make_complex_seq(120, seed=614120)
    b2_u2 = make_complex_seq(100, seed=615100)
    b2_d49 = make_complex_seq(49, seed=611049)
    b2_sep = make_complex_seq(35, seed=612000)
    s4_s = b2_u1 + b2_u2 + at20
    b2_c2 = mutate_seq_snps(b2_u1, positions=[8]) + b2_d49 + b2_u2 + at20
    s4_r = b2_fl_l + s4_s + b2_u1[:20] + b2_sep + b2_c2 + b2_fl_r
    assert len(s4_r) == 884 and len(s4_s) == 240

    # Scenario 5: C1-r6 NEW Bug 3 (994 vs 330 bp, q=330 49D+25I, distinct 140 bp flanks -> 994 <= 1000)
    b3_fl_l = make_complex_seq(140, seed=637141)
    b3_fl_r = make_complex_seq(140, seed=637142)
    b3_a = make_complex_seq(110, seed=631330)
    b3_b = make_complex_seq(110, seed=632330)
    b3_c = make_complex_seq(85, seed=633330)
    b3_i25 = make_complex_seq(25, seed=634000)
    b3_d49 = make_complex_seq(49, seed=635000)
    b3_sep = make_complex_seq(30, seed=636000)
    s5_s = b3_a + b3_b + b3_i25 + b3_c
    s5_r = b3_fl_l + s5_s + b3_sep + b3_a + b3_d49 + b3_b + b3_c + b3_fl_r
    assert len(s5_r) == 994 and len(s5_s) == 330

    # --- Subcase 1: extract-islands CLI (C1-r6 Bugs 1, 2, 3) ---
    _run_extract_islands_subcase("w6_cen_bug1", s1_r, s1_s)
    _run_extract_islands_subcase("w6_cen_bug2", s4_r, s4_s, extra_cli=["--centromere-min-island", "200"])
    _run_extract_islands_subcase("w6_cen_bug3", s5_r, s5_s, extra_cli=["--centromere-min-island", "300"])

    # --- Subcase 2: Multi-slot normal INS CLI (Slots 1-6 + Positive Red-Line 3 control + T22 fork) ---
    b_rg = SyntheticInputBuilder(tmp_path / "w6_rg", chrom="Chr01", backbone_prefix="Ref#1#")
    for i in range(1, 8):
        b_rg.add_node(f"a{i}", make_complex_seq(150, seed=78800 + i))

    b_rg.add_node("s1_r", s1_r)
    b_rg.add_node("s1_s", s1_s)
    b_rg.add_node("s2_r", s2_r)
    b_rg.add_node("s2_s", s2_s)
    b_rg.add_node("s3_r", s3_r)
    b_rg.add_node("s3_s", s3_s)
    b_rg.add_node("s4_r", s4_r)
    b_rg.add_node("s4_s", s4_s)
    b_rg.add_node("s5_r", s5_r)
    b_rg.add_node("s5_s", s5_s)

    # Slot 6: Positive Red-Line 3 control 225=49I225= (499 vs 450 bp -> 1 merged allele, n=2)
    r450_rl = make_complex_seq(450, seed=789001)
    s499_rl = r450_rl[:225] + make_complex_seq(49, seed=789002) + r450_rl[225:]
    assert len(s499_rl) == 499 and len(r450_rl) == 450
    b_rg.add_node("s6_s499", s499_rl)
    b_rg.add_node("s6_r450", r450_rl)

    b_rg.set_ref_path("Ref#1#Chr01", [f"a{i}+" for i in range(1, 8)])
    b_rg.add_path(
        "AltRep#1#Chr01",
        [
            "a1+", "s1_r+", "a2+", "s2_r+", "a3+", "s3_r+", "a4+", "s4_r+",
            "a5+", "s5_r+", "a6+", "s6_s499+", "a7+",
        ],
    )
    b_rg.add_path(
        "AltCand#1#Chr01",
        [
            "a1+", "s1_s+", "a2+", "s2_s+", "a3+", "s3_s+", "a4+", "s4_s+",
            "a5+", "s5_s+", "a6+", "s6_r450+", "a7+",
        ],
    )
    inp_rg = b_rg.write("w6_rg")

    prg_t1, bun_t1 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w6_rg" / "out_t1",
        extra_args=["--threads", "1"],
    )
    prg_t4, bun_t4 = run_build(
        inp_rg,
        out_prefix=tmp_path / "w6_rg" / "out_t4",
        extra_args=["--threads", "4"],
        extra_env={"PANFLAT_FORCE_FORK": "1"},
    )
    assert prg_t1.returncode == 0 and bun_t1 is not None, f"w6_rg t1 failed: {prg_t1.stderr}"
    assert prg_t4.returncode == 0 and bun_t4 is not None, f"w6_rg t4 failed: {prg_t4.stderr}"
    run_verify(bun_t1.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    run_verify(bun_t4.prefix, inp_rg.gfa_path, inp_rg.ref_fa_path, check=True)
    assert_all_global_invariants(bun_t1, inp_rg.gfa_path, inp_rg.ref_fa_path)
    assert len(bun_t1.unresolved) == 0, f"w6_rg expected 0 unresolved, got {bun_t1.unresolved}"

    # Slots 1-5: exact 2-allele slot lengths (rep + 100 bp spacer + cand) & genotypes Ref=0, AltRep=1, AltCand=2
    expected_2allele_slots = [
        (1, "0..1", 1365, 750, 515),  # Slot 1: C1-r6 Bug 1 (750 + 100 + 515)
        (2, "1..2", 1199, 859, 240),  # Slot 2: R1-r6 Major F1 (859 + 100 + 240)
        (3, "2..3", 1015, 735, 180),  # Slot 3: R2-r6 Major F1 3A-4 (735 + 100 + 180)
        (4, "3..4", 1224, 884, 240),  # Slot 4: C1-r6 Bug 2 (884 + 100 + 240)
        (5, "4..5", 1424, 994, 330),  # Slot 5: C1-r6 Bug 3 (994 + 100 + 330)
    ]
    for off_idx, anchors, exp_slot_len, exp_r_len, exp_s_len in expected_2allele_slots:
        assert bun_t1.offsets[off_idx]["slot_ins_len"] == exp_slot_len
        ins_ev = next(r for r in bun_t1.sv_rows if r["type"] == "INS" and r["anchors"] == anchors)
        assert f"len={exp_r_len}" in ins_ev["alleles"] and "src=AltRep#1,n=1" in ins_ev["alleles"]
        assert f"len={exp_s_len}" in ins_ev["alleles"] and "src=AltCand#1,n=1" in ins_ev["alleles"]
        gt = bun_t1.genotypes[ins_ev["event_id"]]
        assert gt["Ref#1"] == "0"
        assert gt["AltRep#1"] == "1"
        assert gt["AltCand#1"] == "2"

    # Slot 6 (Positive Red-Line 3 control 225=49I225=): 1 merged 499 bp allele (n=2), Ref=0, AltRep=1, AltCand=1
    assert bun_t1.offsets[6]["slot_ins_len"] == 499
    ins_s6 = next(r for r in bun_t1.sv_rows if r["type"] == "INS" and r["anchors"] == "5..6")
    assert "len=499" in ins_s6["alleles"] and "n=2" in ins_s6["alleles"]
    gt_s6 = bun_t1.genotypes[ins_s6["event_id"]]
    assert gt_s6["Ref#1"] == "0"
    assert gt_s6["AltRep#1"] == gt_s6["AltCand#1"] == "1"

    # T22 strict determinism across all 10 core output files
    for suffix in _CORE_OUTPUT_SUFFIXES:
        f1 = Path(f"{bun_t1.prefix}{suffix}")
        f4 = Path(f"{bun_t4.prefix}{suffix}")
        assert f1.exists() and f4.exists(), f"Missing output {suffix}"
        assert _sha256_file(f1) == _sha256_file(f4), f"T22 mismatch on w6_rg {suffix}"


