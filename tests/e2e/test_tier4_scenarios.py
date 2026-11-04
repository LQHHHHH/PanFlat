"""Tier 4 End-to-End (E2E) Real-World Pangenome Application Scenario Tests for PanFlat v1.

Covers >= 13 realistic multi-haplotype chromosome pangenome workflows modeling §8.3 and §10 (E1-E5):
1. `test_tier4_e1_simulated_30hap_pangenome_truth_set`
2. `test_tier4_e2_leave_one_out_pangenome_consistency`
3. `test_tier4_e3_multi_gene_gff3_liftover_and_protein_proteome`
4. `test_tier4_e4_parameter_sweep_cluster_id_and_sec_backbone`
5. `test_tier4_e5_centromere_blind_panel_and_mode_comparison`
6. `test_tier4_fragmented_w_line_draft_assemblies`
7. `test_tier4_nested_transposon_in_inversion_hotspot`
8. `test_tier4_tandem_vn_tr_and_satellite_filtering_pipeline`
9. `test_tier4_multi_centromere_dicentric_chromosome`
10. `test_tier4_three_chromosome_full_pangenome_build_verify_merge_qc`
11. `test_tier4_high_diversity_population_genotyping_matrix`
12. `test_tier4_repeat_guard_partial_node_erosion_and_provenance`
13. `test_tier4_reproducibility_and_audit_trail_completeness`
14. `test_tier4_hybrid_p_and_w_line_population_with_centromere_and_gff`
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.helpers import (
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
    run_merge,
    run_verify,
)


# ---------------------------------------------------------------------------
# 1. E1 / §8.3: Simulated 30-Haplotype Pangenome Truth Set
# ---------------------------------------------------------------------------

def test_tier4_e1_simulated_30hap_pangenome_truth_set(tmp_path: Path) -> None:
    """Simulate §8.3 / E1 with 30 haplotypes (Ref#1 + 29 donor strains) across a chromosome
    containing:
      - Locus 1: Pure inversion (INV)
      - Locus 2: Inversion with nested insertion (INV + INS)
      - Locus 3: 5 -> 6 tandem repeat expansion (CNV)
      - Locus 4: Diverged haplotype insertion cluster (20 strains with <50bp SNP divergence
        merged into 1 rep + 2 strains with a >=50bp private divergent block kept as 2nd rep)
      - Locus 5: 300 bp reference deletion (DEL)
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx in range(1, 12):
        b.add_node(f"r{idx}", make_complex_seq(300, seed=1000 + idx))

    # Tandem repeat unit (60 bp) repeated 5 times in reference between r6 and r7
    b.add_node("tr_u", make_complex_seq(60, seed=1050))
    # Nested insertion inside Locus 2 inversion (between r4 and r5)
    b.add_node("inv_ins", make_complex_seq(220, seed=1060))

    # Locus 4 (between r8 and r9): shared insertion across 20 strains with dispersed SNPs
    base_cluster_seq = make_complex_seq(500, seed=1070)
    b.add_node("clu_base", base_cluster_seq)
    for k in range(1, 20):
        b.add_node(f"clu_snp_{k}", mutate_seq_snps(base_cluster_seq, num_snps=5, seed=1100 + k))
    # Divergent allele at Locus 4 with a 120 bp private insertion block (trimmed to 480 bp < 500 bp)
    priv_block = make_complex_seq(120, seed=1080)
    b.add_node("clu_priv", base_cluster_seq[70:250] + priv_block + base_cluster_seq[250:-70])

    ref_steps = [
        "r1+", "r2+", "r3+", "r4+", "r5+", "r6+",
        "tr_u+", "tr_u+", "tr_u+", "tr_u+", "tr_u+",
        "r7+", "r8+", "r9+", "r10+", "r11+",
    ]
    b.set_ref_path("Ref#1#Chr01", ref_steps)

    # Build 29 donor haplotypes (total 30 haplotypes including Ref#1)
    for strain_idx in range(1, 30):
        hap_name = f"Strain{strain_idx:02d}#1#Chr01"
        steps: list[str] = ["r1+"]
        # Locus 1: Strains 1..5 carry pure inversion of r2
        if 1 <= strain_idx <= 5:
            steps.append("r2-")
        else:
            steps.append("r2+")
        steps.append("r3+")

        # Locus 2: Strains 6..10 carry inversion [r5- -> inv_ins+ -> r4-]
        if 6 <= strain_idx <= 10:
            steps.extend(["r5-", "inv_ins+", "r4-"])
        else:
            steps.extend(["r4+", "r5+"])
        steps.append("r6+")

        # Locus 3: Strains 11..16 carry 6 copies of tr_u+ (5 -> 6 CNV)
        n_tr = 6 if 11 <= strain_idx <= 16 else 5
        steps.extend(["tr_u+"] * n_tr)
        steps.append("r7+")
        steps.append("r8+")

        # Locus 4: Strains 1..20 carry the near-identical 500bp insertion cluster;
        # Strains 21..22 carry the divergent 620bp allele with a 120bp private block
        if strain_idx == 1:
            steps.append("clu_base+")
        elif 2 <= strain_idx <= 20:
            steps.append(f"clu_snp_{strain_idx - 1}+")
        elif strain_idx in (21, 22):
            steps.append("clu_priv+")
        steps.append("r9+")

        # Locus 5: Strains 23..27 skip r10+ (300bp DEL)
        if not (23 <= strain_idx <= 27):
            steps.append("r10+")
        steps.append("r11+")

        b.add_path(hap_name, steps)

    inputs = b.write("e1_30hap")
    _, bundle = run_build(inputs, extra_args=["--qc", "--threads", "2"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # All 30 haplotypes must appear in genotypes.tsv
    assert len(bundle.genotype_haps) == 30
    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert {"INV", "INS", "CNV", "DEL"}.issubset(ev_types)

    # Locus 4 cluster: 20 SNP-diverged strains merge into 1 rep, plus 1 rep for clu_priv (2 reps total at r9)
    r9_ref_start = int(next(r for r in bundle.offsets if r["node_id"] == "r9")["ref_start"])
    r9_ins_events = [ev for ev in bundle.sv_rows if ev["type"] == "INS" and int(ev["ref_start"]) == r9_ref_start]
    assert len(r9_ins_events) == 1
    r9_pieces = [ins for ins in bundle.inserts if ins["event_id"] == r9_ins_events[0]["event_id"]]
    assert len(r9_pieces) == 2


# ---------------------------------------------------------------------------
# 2. E2: Leave-One-Out Pangenome Consistency
# ---------------------------------------------------------------------------

def test_tier4_e2_leave_one_out_pangenome_consistency(tmp_path: Path) -> None:
    """Model E2 leave-one-out construction:
    - Leaving out a strain whose insertion is shared by other strains leaves pan.fa unchanged.
    - Leaving out a strain carrying a singleton private insertion reduces pan.fa by exactly
      len(singleton) + spacer_n while preserving all other coordinates and invariants.
    """
    def _build_panel(sub_dir: Path, exclude_strain: str | None) -> tuple[Path, Path, object]:
        b = SyntheticInputBuilder(sub_dir, chrom="Chr01", backbone_prefix="Ref#1#")
        for nid, seed in [("r1", 201), ("r2", 202), ("r3", 203), ("r4", 204)]:
            b.add_node(nid, make_complex_seq(250, seed=seed))
        b.add_node("shared_ins", make_complex_seq(300, seed=205))
        b.add_node("private_ins", make_complex_seq(240, seed=206))

        b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+"])
        strains = [
            ("S1", ["r1+", "shared_ins+", "r2+", "r3+", "r4+"]),
            ("S2", ["r1+", "shared_ins+", "r2+", "r3+", "r4+"]),
            ("S3", ["r1+", "shared_ins+", "r2+", "r3+", "r4+"]),
            ("S4_priv", ["r1+", "private_ins+", "r2+", "r3+", "r4+"]),
        ]
        for s_name, steps in strains:
            if s_name != exclude_strain:
                b.add_path(f"{s_name}#1#Chr01", steps)
        inp = b.write("panel")
        _, bun = run_build(inp, extra_args=["--spacer-n", "100"])
        assert bun is not None
        assert_all_global_invariants(bun, inp.gfa_path, inp.ref_fa_path, spacer_n=100)
        return (inp.gfa_path, inp.ref_fa_path, bun)

    _, _, bun_full = _build_panel(tmp_path / "full", exclude_strain=None)
    _, _, bun_loo_shared = _build_panel(tmp_path / "loo_s1", exclude_strain="S1")
    _, _, bun_loo_priv = _build_panel(tmp_path / "loo_s4", exclude_strain="S4_priv")

    assert bun_full.pan_seq == bun_loo_shared.pan_seq
    assert len(bun_full.pan_seq) - len(bun_loo_priv.pan_seq) == 240 + 100


# ---------------------------------------------------------------------------
# 3. E3: Multi-Gene GFF3 Liftover & Proteome Recovery (6 Genes on + and - Strands)
# ---------------------------------------------------------------------------

def test_tier4_e3_multi_gene_gff3_liftover_and_protein_proteome(tmp_path: Path) -> None:
    """Model E3 across a chromosome with 6 multi-exon genes (3 on '+' strand, 3 on '-' strand)
    experiencing intronic TE insertions, UTR insertions, splice-proximal insertions, and
    frame-splitting CDS insertions (phases 0, 1, 2 and 1-2 bp micro-segments).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    ref_steps: list[str] = []
    donor_steps: list[str] = []
    cur_pos = 0

    for g_idx in range(1, 7):
        strand = "+" if g_idx % 2 == 1 else "-"
        cds_raw = make_coding_seq(60, seed=3000 + g_idx)  # 180 bp
        cds_plus = cds_raw if strand == "+" else revcomp(cds_raw)

        # Intergenic upstream buffer (150 bp)
        ig_id = f"g{g_idx}_ig"
        b.add_node(ig_id, make_complex_seq(150, seed=3100 + g_idx))
        ref_steps.append(f"{ig_id}+")
        donor_steps.append(f"{ig_id}+")
        cur_pos += 150

        # Exon 1: 5' UTR (60 bp, split at 30 bp for UTR insertion) + CDS part 1 (73 bp -> phase 1!)
        utr1_a = f"g{g_idx}_u1a"
        utr1_b = f"g{g_idx}_u1b"
        cds1_a = f"g{g_idx}_c1a"
        cds1_b = f"g{g_idx}_c1b"
        b.add_node(utr1_a, make_complex_seq(30, seed=3200 + g_idx))
        b.add_node(utr1_b, make_complex_seq(30, seed=3300 + g_idx))
        # Split 73 bp CDS1 into 2 bp micro-segment + 71 bp segment!
        b.add_node(cds1_a, cds_plus[:2])
        b.add_node(cds1_b, cds_plus[2:73])

        ex1_start_1based = cur_pos + 1
        cds1_start_1based = cur_pos + 60 + 1
        cds1_end_1based = cur_pos + 60 + 73
        ex1_end_1based = cds1_end_1based
        cur_pos += 133

        # Intron 1 (160 bp, split into 80 bp + 75 bp + 5 bp for intronic and splice-proximal insertions)
        int_a = f"g{g_idx}_ia"
        int_b = f"g{g_idx}_ib"
        int_c = f"g{g_idx}_ic"
        b.add_node(int_a, make_complex_seq(80, seed=3400 + g_idx))
        b.add_node(int_b, make_complex_seq(75, seed=3500 + g_idx))
        b.add_node(int_c, make_complex_seq(5, seed=3600 + g_idx))
        cur_pos += 160

        # Exon 2: CDS part 2 (107 bp, split at 50 bp + 57 bp) + 3' UTR (40 bp)
        cds2_a = f"g{g_idx}_c2a"
        cds2_b = f"g{g_idx}_c2b"
        utr2 = f"g{g_idx}_u2"
        b.add_node(cds2_a, cds_plus[73:123])
        b.add_node(cds2_b, cds_plus[123:180])
        b.add_node(utr2, make_complex_seq(40, seed=3700 + g_idx))

        ex2_start_1based = cur_pos + 1
        cds2_start_1based = cur_pos + 1
        cds2_end_1based = cur_pos + 107
        ex2_end_1based = cur_pos + 147
        cur_pos += 147

        # Insertions for this gene
        ins_u = f"g{g_idx}_ins_utr"
        ins_c1 = f"g{g_idx}_ins_cds1"
        ins_i = f"g{g_idx}_ins_int"
        ins_sp = f"g{g_idx}_ins_sp"
        ins_c2 = f"g{g_idx}_ins_cds2"
        b.add_node(ins_u, make_complex_seq(120, seed=3800 + g_idx))
        b.add_node(ins_c1, make_complex_seq(150, seed=3810 + g_idx))
        b.add_node(ins_i, make_complex_seq(350, seed=3820 + g_idx))
        b.add_node(ins_sp, make_complex_seq(110, seed=3830 + g_idx))
        b.add_node(ins_c2, make_complex_seq(180, seed=3840 + g_idx))

        ref_steps.extend(
            [f"{utr1_a}+", f"{utr1_b}+", f"{cds1_a}+", f"{cds1_b}+", f"{int_a}+", f"{int_b}+", f"{int_c}+", f"{cds2_a}+", f"{cds2_b}+", f"{utr2}+"]
        )
        donor_steps.extend(
            [
                f"{utr1_a}+", f"{ins_u}+", f"{utr1_b}+",
                f"{cds1_a}+", f"{ins_c1}+", f"{cds1_b}+",
                f"{int_a}+", f"{ins_i}+", f"{int_b}+", f"{ins_sp}+", f"{int_c}+",
                f"{cds2_a}+", f"{ins_c2}+", f"{cds2_b}+", f"{utr2}+",
            ]
        )

        if strand == "+":
            # Exon 1 CDS has length 73 (phase 0); next CDS has phase (3 - (73 % 3)) % 3 == 2
            cds_entries = [(cds1_start_1based, cds1_end_1based, 0), (cds2_start_1based, cds2_end_1based, 2)]
        else:
            # On '-' strand, transcriptional order is Exon 2 (length 107, phase 0) then Exon 1 (phase (3 - 107%3)%3 == 1)
            cds_entries = [(cds1_start_1based, cds1_end_1based, 1), (cds2_start_1based, cds2_end_1based, 0)]

        b.add_gff_gene(
            gene_id=f"gene_{g_idx}",
            mrna_id=f"mrna_{g_idx}",
            strand=strand,
            exons=[(ex1_start_1based, ex1_end_1based), (ex2_start_1based, ex2_end_1based)],
            cds_list=cds_entries,
        )

    b.add_node("tail_end", make_complex_seq(200, seed=3999))
    ref_steps.append("tail_end+")
    donor_steps.append("tail_end+")

    b.set_ref_path("Ref#1#Chr01", ref_steps)
    b.add_path("DonorTE#1#Chr01", donor_steps)

    inputs = b.write("e3_proteome")
    _, bundle = run_build(inputs, extra_args=["--splice-window", "8"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )


# ---------------------------------------------------------------------------
# 4. E4: Parameter Sweep (--cluster-id, --cluster-win-id, --sec-backbone)
# ---------------------------------------------------------------------------

def test_tier4_e4_parameter_sweep_cluster_id_and_sec_backbone(tmp_path: Path) -> None:
    """Model E4 parameter sweep across --cluster-id {0.85, 0.95, 0.99}, --cluster-win-id,
    and --sec-backbone {support, longest}.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(300, seed=401))
    b.add_node("r2", make_complex_seq(300, seed=402))
    b.add_node("r3", make_complex_seq(300, seed=403))

    base_seq = make_complex_seq(500, seed=404)
    # 98% identical (10 SNPs in 500 bp)
    seq_98 = mutate_seq_snps(base_seq, num_snps=10, seed=405)
    # 90% identical (50 SNPs in 500 bp)
    seq_90 = mutate_seq_snps(base_seq, num_snps=50, seed=406)

    b.add_node("s_base", base_seq)
    b.add_node("s_98", seq_98)
    b.add_node("s_90", seq_90)

    # Secondary backbone locus between r2 and r3
    b.add_node("sharedA", make_complex_seq(250, seed=407))
    b.add_node("shortB", make_complex_seq(200, seed=408))
    b.add_node("longC", make_complex_seq(320, seed=409))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+"])
    b.add_path("H1#1#Chr01", ["r1+", "s_base+", "r2+", "sharedA+", "shortB+", "r3+"])
    b.add_path("H2#1#Chr01", ["r1+", "s_98+", "r2+", "sharedA+", "shortB+", "r3+"])
    b.add_path("H3#1#Chr01", ["r1+", "s_90+", "r2+", "sharedA+", "longC+", "r3+"])

    inputs = b.write("e4_sweep")

    _, bun_85 = run_build(
        inputs,
        out_prefix=tmp_path / "cid_85",
        extra_args=["--cluster-id", "0.85", "--cluster-win-id", "0.60", "--sec-backbone", "support"],
    )
    _, bun_95 = run_build(
        inputs,
        out_prefix=tmp_path / "cid_95",
        extra_args=["--cluster-id", "0.95", "--cluster-win-id", "0.70", "--sec-backbone", "support"],
    )
    _, bun_99 = run_build(
        inputs,
        out_prefix=tmp_path / "cid_99",
        extra_args=["--cluster-id", "0.99", "--cluster-win-id", "0.90", "--sec-backbone", "longest"],
    )
    assert bun_85 is not None and bun_95 is not None and bun_99 is not None
    assert_all_global_invariants(bun_85, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_95, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_99, inputs.gfa_path, inputs.ref_fa_path)

    # Monotonic pangenome size as clustering stringency increases:
    # At 0.85: s_base, s_98, s_90 merge into 1 rep
    # At 0.95: s_base and s_98 merge, s_90 stays separate (2 reps)
    # At 0.99: all 3 stay separate (3 reps)
    assert len(bun_85.pan_seq) < len(bun_95.pan_seq) < len(bun_99.pan_seq)


# ---------------------------------------------------------------------------
# 5. E5 / C15: Centromere Blind Panel & Mode Comparison (off vs bypass vs extract-islands)
# ---------------------------------------------------------------------------

def test_tier4_e5_centromere_blind_panel_and_mode_comparison(tmp_path: Path) -> None:
    """Model E5 / C15 centromere evaluation panel comparing --centromere-mode off, bypass,
    and extract-islands (with --repeat-guard on vs off) across:
      - True complex island (700 bp)
      - Old reference-like TE homologous to cen_ref (600 bp -> CEN_REFERENCE_LIKE)
      - Pure low-divergence satellite array (1500 bp -> CEN_REPEAT_LIKE)
      - Hybrid satellite + complex island block (1500 bp sat + 650 bp island)
      - Short residual (<500 bp -> CEN_SHORT_RESIDUAL)
      - Gapped haplotype path (status=gapped) and partial haplotype path (status=partial)
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    s_n0 = make_complex_seq(250, seed=501)
    s_u = make_complex_seq(250, seed=502)
    s_cen_ref = make_complex_seq(1000, seed=503)
    s_v = make_complex_seq(250, seed=504)
    s_n4 = make_complex_seq(250, seed=505)

    b.add_node("n0", s_n0)
    b.add_node("u", s_u)
    b.add_node("cen_ref", s_cen_ref)
    b.add_node("v", s_v)
    b.add_node("n4", s_n4)

    true_island = make_complex_seq(700, seed=506)
    old_ref_te = mutate_seq_snps(s_cen_ref[100:700], num_snps=12, seed=507)  # 98% identical to cen_ref
    pure_sat = make_satellite_seq(unit_len=20, total_len=1500, seed=508)
    hyb_block = make_satellite_seq(unit_len=20, total_len=1500, seed=509) + make_complex_seq(650, seed=510)
    short_cand = make_complex_seq(250, seed=511)  # < 500 bp min-island
    gapped_node = make_complex_seq(300, seed=512) + ("N" * 20) + make_complex_seq(300, seed=513)

    b.add_node("true_isl", true_island)
    b.add_node("old_te", old_ref_te)
    b.add_node("pure_sat", pure_sat)
    b.add_node("hyb_blk", hyb_block)
    b.add_node("short_c", short_cand)
    b.add_node("gap_n", gapped_node)

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen_ref+", "v+", "n4+"])
    b.add_path("H_island#1#Chr01", ["n0+", "u+", "cen_ref+", "true_isl+", "v+", "n4+"])
    b.add_path("H_oldte#1#Chr01", ["n0+", "u+", "cen_ref+", "old_te+", "v+", "n4+"])
    b.add_path("H_sat#1#Chr01", ["n0+", "u+", "cen_ref+", "pure_sat+", "v+", "n4+"])
    b.add_path("H_hyb#1#Chr01", ["n0+", "u+", "cen_ref+", "hyb_blk+", "v+", "n4+"])
    b.add_path("H_short#1#Chr01", ["n0+", "u+", "cen_ref+", "short_c+", "v+", "n4+"])
    b.add_path("H_gapped#1#Chr01", ["n0+", "u+", "cen_ref+", "gap_n+", "v+", "n4+"])
    b.add_path("H_partial#1#Chr01", ["n0+", "u+", "cen_ref+"])

    b.add_bed_region(500, 1500, "CEN_E5")
    inputs = b.write("e5_blind")

    # 1) bypass mode
    _, bun_byp = run_build(
        inputs,
        out_prefix=tmp_path / "e5_bypass",
        extra_args=["--centromere-mode", "bypass"],
    )
    # 2) extract-islands mode
    _, bun_ext = run_build(
        inputs,
        out_prefix=tmp_path / "e5_extract",
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    # 3) off mode (without passing BED)
    _, bun_off = run_build(
        inputs,
        out_prefix=tmp_path / "e5_off",
        include_bed=False,
        extra_args=["--centromere-mode", "off", "--repeat-guard", "on"],
    )
    assert bun_byp is not None and bun_ext is not None and bun_off is not None
    assert_all_global_invariants(bun_byp, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_ext, inputs.gfa_path, inputs.ref_fa_path)
    assert_all_global_invariants(bun_off, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun_byp)
    assert_centromere_accounting_and_offsets(bun_ext)

    # In bypass mode, pan.fa is identical to reference (no SVs outside centromere)
    ref_full_seq = b.spell_steps(["n0+", "u+", "cen_ref+", "v+", "n4+"])
    assert bun_byp.pan_seq == ref_full_seq
    assert any(u["reason"] == "CEN_BYPASS" for u in bun_byp.unresolved)

    # In extract-islands mode: true_isl and the 650bp island from hyb_blk are mounted as CEN_ISLAND;
    # old_te is subtracted as reference_like_bp; pure_sat is masked as repeat_like_bp; short_c is short_bp
    cen_by_hap = {r["hap"]: r for r in (bun_ext.centromere_rows or [])}
    assert int(cen_by_hap["H_island#1"]["accepted_source_bp"]) == 700
    assert int(cen_by_hap["H_oldte#1"]["reference_like_bp"]) == 600
    assert int(cen_by_hap["H_sat#1"]["repeat_like_bp"]) == 1500
    assert int(cen_by_hap["H_short#1"]["short_bp"]) == 250
    assert cen_by_hap["H_gapped#1"]["status"] == "gapped"
    assert cen_by_hap["H_partial#1"]["status"] == "partial"


# ---------------------------------------------------------------------------
# 6. Fragmented W-Line Draft Assemblies with Non-Zero Offsets & Reverse Contigs
# ---------------------------------------------------------------------------

def test_tier4_fragmented_w_line_draft_assemblies(tmp_path: Path) -> None:
    """Multi-sample pangenome where donor haplotypes are fragmented into multiple W-line
    contigs with non-zero start offsets, contig breaks (UNANCHORED_END), and reverse-oriented contigs.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx in range(1, 8):
        b.add_node(f"r{idx}", make_complex_seq(250, seed=600 + idx))

    b.add_node("ins_ctg1", make_complex_seq(180, seed=610))
    b.add_node("unanch_ctg1", make_complex_seq(95, seed=611))
    b.add_node("ins_ctg2", make_complex_seq(210, seed=612))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+", "r7+"])
    # DraftSample1 is split into two W-line contigs (ctgA covering r1..r3, ctgB in reverse covering r7..r5)
    # Must NOT stitch across the r3..r5 contig gap (no fake DEL of r4!)
    b.add_walk("Draft1", "1", "ctgA", 5000, None, ["r1+", "ins_ctg1+", "r2+", "r3+", "unanch_ctg1+"])
    b.add_walk("Draft1", "1", "ctgB", 25000, None, ["r7-", "r6-", "ins_ctg2-", "r5-"])

    inputs = b.write("w_drafts")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "INS" in ev_types
    assert "DEL" not in ev_types
    assert "INV" not in ev_types
    assert any(u["reason"] == "UNANCHORED_END" for u in bundle.unresolved)


# ---------------------------------------------------------------------------
# 7. Nested Transposon in Inversion Hotspot (INV + INV_DISPLACED + DUP_TRA + T4)
# ---------------------------------------------------------------------------

def test_tier4_nested_transposon_in_inversion_hotspot(tmp_path: Path) -> None:
    """Complex structural rearrangement hotspot combining top-level INV, INV_DISPLACED,
    DUP_TRA, and nested bifurcating TE insertions (A+B vs A+C).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx in range(1, 9):
        b.add_node(f"r{idx}", make_complex_seq(300, seed=700 + idx))

    b.add_node("te_A", make_complex_seq(350, seed=710))
    b.add_node("te_B", make_complex_seq(180, seed=711))
    b.add_node("te_C", make_complex_seq(220, seed=712))
    b.add_node("tra_ins", make_complex_seq(190, seed=713))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "r5+", "r6+", "r7+", "r8+"])
    # Hap1 & Hap2 carry inversion of [r2..r3] with nested TE A+B (in reverse inside raw inversion)
    b.add_path("Hap1#1#Chr01", ["r1+", "r3-", "te_B-", "te_A-", "r2-", "r4+", "r5+", "r6+", "r7+", "r8+"])
    b.add_path("Hap2#1#Chr01", ["r1+", "r3-", "te_B-", "te_A-", "r2-", "r4+", "r5+", "r6+", "r7+", "r8+"])
    # Hap3 carries the same inversion but with nested TE A+C, plus a DUP_TRA visiting r8+ between r5+ and r6+
    b.add_path("Hap3#1#Chr01", ["r1+", "r3-", "te_C-", "te_A-", "r2-", "r4+", "r5+", "r8+", "tra_ins+", "r6+", "r7+", "r8+"])

    inputs = b.write("inv_hotspot")
    _, bundle = run_build(inputs, extra_args=["--max-jump", "500"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert {"INV", "INS", "DUP_TRA"}.issubset(ev_types)


# ---------------------------------------------------------------------------
# 8. Tandem VNTR and Satellite Filtering Pipeline
# ---------------------------------------------------------------------------

def test_tier4_tandem_vn_tr_and_satellite_filtering_pipeline(tmp_path: Path) -> None:
    """Chromosome with reference-shared canonical CNV, novel tandem repeat folded via
    --max-copy 1, interrupted novel repeat U-X-U retained intact, and low-complexity
    satellite filtered by --repeat-guard on.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx in range(1, 7):
        b.add_node(f"r{idx}", make_complex_seq(250, seed=800 + idx))

    b.add_node("vntr_a", make_complex_seq(45, seed=810))
    b.add_node("vntr_b", make_complex_seq(45, seed=811))
    b.add_node("nov_U", make_complex_seq(120, seed=812))
    b.add_node("nov_X", make_complex_seq(100, seed=813))
    b.add_node("sat_arr", make_satellite_seq(unit_len=18, total_len=1400, seed=814))

    # Reference has 2 copies of (vntr_a+, vntr_b+) between r1 and r2
    b.set_ref_path(
        "Ref#1#Chr01",
        ["r1+", "vntr_a+", "vntr_b+", "vntr_a+", "vntr_b+", "r2+", "r3+", "r4+", "r5+", "r6+"],
    )
    # Donor has:
    # - 4 copies of rotated (vntr_b+, vntr_a+) between r1 and r2 (canonical rotation matches -> CNV 2->4)
    # - Novel tandem repeat nov_U+ x 4 between r2 and r3 (folded to 1 copy)
    # - Interrupted repeat nov_U+ -> nov_X+ -> nov_U+ between r3 and r4 (no tandem period >=2 -> X retained!)
    # - Satellite array sat_arr+ between r4 and r5 (masked by repeat-guard -> REPEAT_LIKE)
    b.add_path(
        "Donor#1#Chr01",
        [
            "r1+",
            "vntr_b+", "vntr_a+", "vntr_b+", "vntr_a+", "vntr_b+", "vntr_a+", "vntr_b+", "vntr_a+",
            "r2+",
            "nov_U+", "nov_U+", "nov_U+", "nov_U+",
            "r3+",
            "nov_U+", "nov_X+", "nov_U+",
            "r4+",
            "sat_arr+",
            "r5+", "r6+",
        ],
    )

    inputs = b.write("vntr_sat_pipe")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on", "--max-copy", "1"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    ev_types = {ev["type"] for ev in bundle.sv_rows}
    assert "CNV" in ev_types and "INS" in ev_types
    assert any(u["reason"] == "REPEAT_LIKE" for u in bundle.unresolved)
    # Ensure nov_X sequence is preserved in pan.fa
    assert b.nodes["nov_X"].upper() in bundle.pan_seq.upper()


# ---------------------------------------------------------------------------
# 9. Multi-Centromere Dicentric Chromosome (Overlapping BED Merge + Edge Downgrade)
# ---------------------------------------------------------------------------

def test_tier4_multi_centromere_dicentric_chromosome(tmp_path: Path) -> None:
    """Chromosome with 3 BED entries:
    - Two entries that overlap after outward expansion to unique anchors (merged into Region 1,
      running extract-islands and recording both input_bed_ids).
    - One entry at the right chromosome end lacking a right anchor v (downgraded to bypass
      with CEN_ANCHOR_NOT_FOUND as Region 2).
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r0", make_complex_seq(250, seed=901))
    b.add_node("u1", make_complex_seq(250, seed=902))
    # Non-unique repeat node appearing twice in Region 1 so outward expansion merges BED1A and BED1B
    b.add_node("cen_rep", make_complex_seq(300, seed=903))
    b.add_node("v1", make_complex_seq(250, seed=904))
    b.add_node("r_mid", make_complex_seq(250, seed=905))
    b.add_node("u2", make_complex_seq(250, seed=906))
    b.add_node("cen_end", make_complex_seq(300, seed=907))

    b.add_node("isl_r1", make_complex_seq(600, seed=908))
    b.add_node("isl_r2", make_complex_seq(600, seed=909))

    # Reference coordinates:
    # r0: [0,250), u1: [250,500), cen_rep: [500,800), cen_rep: [800,1100), v1: [1100,1350),
    # r_mid: [1350,1600), u2: [1600,1850), cen_end: [1850,2150)
    b.set_ref_path(
        "Ref#1#Chr01",
        ["r0+", "u1+", "cen_rep+", "cen_rep+", "v1+", "r_mid+", "u2+", "cen_end+"],
    )
    b.add_path(
        "Donor#1#Chr01",
        ["r0+", "u1+", "cen_rep+", "isl_r1+", "cen_rep+", "v1+", "r_mid+", "u2+", "cen_end+", "isl_r2+"],
    )

    b.add_bed_region(520, 750, "BED_1A")
    b.add_bed_region(820, 1080, "BED_1B")
    b.add_bed_region(1850, 2150, "BED_2_END")

    inputs = b.write("dicentric")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)

    assert bundle.centromere_rows is not None
    reg1_rows = [r for r in bundle.centromere_rows if int(r["ref_start"]) == 500]
    reg2_rows = [r for r in bundle.centromere_rows if int(r["ref_start"]) == 1850]
    assert len(reg1_rows) >= 1 and len(reg2_rows) >= 1
    assert "BED_1A" in str(reg1_rows[0]["input_bed_ids"]) and "BED_1B" in str(reg1_rows[0]["input_bed_ids"])
    assert reg1_rows[0]["mode"] == "extract-islands"
    assert reg2_rows[0]["mode"] == "bypass"


# ---------------------------------------------------------------------------
# 10. Three-Chromosome Full Pangenome Build -> Verify -> Merge -> QC Workflow
# ---------------------------------------------------------------------------

def test_tier4_three_chromosome_full_pangenome_build_verify_merge_qc(tmp_path: Path) -> None:
    """Complete 3-chromosome pangenome workflow:
      - Chr01: extract-islands + GFF3 + QC
      - Chr02: bypass + GFF3 + QC
      - Chr03: centromere off + GFF3 + QC
      -> standalone panflat verify on each -> panflat merge into genome-wide pangenome.
    """
    prefixes: list[Path] = []
    for idx, (chrom, mode) in enumerate(
        [("Chr01", "extract-islands"), ("Chr02", "bypass"), ("Chr03", "off")], start=1
    ):
        b = SyntheticInputBuilder(tmp_path / chrom, chrom=chrom, backbone_prefix="Ref#1#")
        for nid, seed in [("n0", 1010 + idx), ("u", 1020 + idx), ("cen", 1030 + idx), ("v", 1040 + idx)]:
            b.add_node(nid, make_complex_seq(250, seed=seed))
        b.add_node("cand", make_complex_seq(600, seed=1050 + idx))
        b.add_node("norm_ins", make_complex_seq(180, seed=1060 + idx))

        b.set_ref_path(f"Ref#1#{chrom}", ["n0+", "u+", "cen+", "v+"])
        b.add_path(f"Strain1#1#{chrom}", ["n0+", "norm_ins+", "u+", "cen+", "cand+", "v+"])
        if mode != "off":
            b.add_bed_region(500, 750, f"CEN_{chrom}")
        b.add_gff_gene(f"gene_{chrom}", f"mrna_{chrom}", "+", [(21, 200)], [(21, 197, 0)], chrom=chrom)

        inp = b.write(chrom)
        extra = ["--centromere-mode", mode, "--qc"]
        if mode == "extract-islands":
            extra.extend(["--centromere-min-island", "500"])
        _, bun = run_build(inp, include_bed=(mode != "off"), extra_args=extra)
        assert bun is not None
        assert_all_global_invariants(bun, inp.gfa_path, inp.ref_fa_path)
        assert_gff3_liftover_integrity(
            inp.ref_fa_path, inp.gff_path, bun.pan_fa_path, bun.lifted_gff_path
        )
        run_verify(inp.out_prefix, inp.gfa_path, inp.ref_fa_path)
        prefixes.append(inp.out_prefix)

    _, merged = run_merge(prefixes, tmp_path / "genome_merged")
    assert merged is not None
    assert set(merged.pan_fa.keys()) == {"Chr01", "Chr02", "Chr03"}
    assert merged.centromere_rows is not None
    assert {r["chrom"] for r in merged.centromere_rows} == {"Chr01", "Chr02"}
    assert merged.lifted_gff_rows is not None
    assert {r["seqid"] for r in merged.lifted_gff_rows} == {"Chr01", "Chr02", "Chr03"}


# ---------------------------------------------------------------------------
# 11. High-Diversity 16-Haplotype Population Genotyping Matrix
# ---------------------------------------------------------------------------

def test_tier4_high_diversity_population_genotyping_matrix(tmp_path: Path) -> None:
    """16-haplotype population with multi-allelic nested insertions (producing alleles 1..3
    ordered by carrier frequency descending), deletions, CNVs, inversions, and missing '.' calls.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for idx in range(1, 7):
        b.add_node(f"r{idx}", make_complex_seq(250, seed=1100 + idx))
    b.add_node("tr", make_complex_seq(60, seed=1110))

    # Multi-allelic insertion locus between r1 and r2:
    # Allele A+B (carried by 6 haplotypes -> allele 1)
    # Allele A+C (carried by 3 haplotypes -> allele 2)
    # Allele D (unshared, carried by 2 haplotypes -> allele 3)
    b.add_node("A", make_complex_seq(200, seed=1111))
    b.add_node("B", make_complex_seq(150, seed=1112))
    b.add_node("C", make_complex_seq(160, seed=1113))
    b.add_node("D", make_complex_seq(220, seed=1114))

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+"])

    # H01..H06: carry A+B at (r1,r2), INV of r3-
    for i in range(1, 7):
        b.add_path(f"H{i:02d}#1#Chr01", ["r1+", "A+", "B+", "r2+", "r3-", "r4+", "tr+", "tr+", "r5+", "r6+"])
    # H07..H09: carry A+C at (r1,r2), DEL of r3
    for i in range(7, 10):
        b.add_path(f"H{i:02d}#1#Chr01", ["r1+", "A+", "C+", "r2+", "r4+", "tr+", "tr+", "r5+", "r6+"])
    # H10..H11: carry D at (r1,r2), 4 copies of tr+ (CNV)
    for i in range(10, 12):
        b.add_path(f"H{i:02d}#1#Chr01", ["r1+", "D+", "r2+", "r3+", "r4+", "tr+", "tr+", "tr+", "tr+", "r5+", "r6+"])
    # H12..H13: match reference at (r1,r2) -> genotype '0'
    for i in range(12, 14):
        b.add_path(f"H{i:02d}#1#Chr01", ["r1+", "r2+", "r3+", "r4+", "tr+", "tr+", "r5+", "r6+"])
    # H14..H15: partial contigs covering only r4..r6 -> genotype '.' at (r1,r2)
    for i in range(14, 16):
        b.add_path(f"H{i:02d}#1#Chr01", ["r4+", "tr+", "tr+", "r5+", "r6+"])

    inputs = b.write("pop16_gt")
    _, bundle = run_build(inputs)
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    assert len(bundle.genotype_haps) == 16
    ins_ev = next(ev for ev in bundle.sv_rows if ev["type"] == "INS")
    gt = bundle.genotypes[ins_ev["event_id"]]
    # Carrier counts: H01..H06 (6) -> allele '1', H07..H09 (3) -> allele '2', H10..H11 (2) -> allele '3'
    assert all(gt[f"H{i:02d}#1"] == "1" for i in range(1, 7))
    assert all(gt[f"H{i:02d}#1"] == "2" for i in range(7, 10))
    assert all(gt[f"H{i:02d}#1"] == "3" for i in range(10, 12))
    assert all(gt[f"H{i:02d}#1"] == "0" for i in range(12, 14))
    assert all(gt[f"H{i:02d}#1"] == "." for i in range(14, 16))


# ---------------------------------------------------------------------------
# 12. Repeat-Guard Partial-Node Erosion & Multi-Node Provenance
# ---------------------------------------------------------------------------

def test_tier4_repeat_guard_partial_node_erosion_and_provenance(tmp_path: Path) -> None:
    """Multi-kb donor insertion spanning 3 GFA nodes (n_left + n_mid_hybrid + n_right) where
    repeat-guard masks an internal satellite window cutting mid-node inside n_mid_hybrid.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    b.add_node("r1", make_complex_seq(300, seed=1201))
    b.add_node("r2", make_complex_seq(300, seed=1202))

    s_left = make_complex_seq(350, seed=1203)
    s_mid_c1 = make_complex_seq(250, seed=1204)
    s_mid_sat = make_satellite_seq(unit_len=20, total_len=1600, seed=1205)
    s_mid_c2 = make_complex_seq(250, seed=1206)
    s_right = make_complex_seq(350, seed=1207)

    b.add_node("n_left", s_left)
    b.add_node("n_mid_hyb", s_mid_c1 + s_mid_sat + s_mid_c2)
    b.add_node("n_right", s_right)

    b.set_ref_path("Ref#1#Chr01", ["r1+", "r2+"])
    b.add_path("Donor#1#Chr01", ["r1+", "n_left+", "n_mid_hyb+", "n_right+", "r2+"])

    inputs = b.write("rg_partial_erosion")
    _, bundle = run_build(inputs, extra_args=["--repeat-guard", "on"])
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)

    # Verify repeat-masked region recorded in unresolved.tsv and two flanking complex residuals inserted
    rep_unres = [u for u in bundle.unresolved if u["reason"] == "REPEAT_LIKE"]
    assert len(rep_unres) == 1
    assert len(bundle.inserts) >= 2


# ---------------------------------------------------------------------------
# 13. Reproducibility & Audit Trail Completeness (SHA-256, JSON Logs, 5-Way Accounting)
# ---------------------------------------------------------------------------

def test_tier4_reproducibility_and_audit_trail_completeness(tmp_path: Path) -> None:
    """Verify SHA-256 input tracking, structured JSON candidate logs in {p}.log with all
    mandatory §3.8.5 fields, 5-way centromere accounting, and deterministic thread execution.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    for nid, seed in [("n0", 1301), ("u", 1302), ("cen", 1303), ("v", 1304), ("n4", 1305)]:
        b.add_node(nid, make_complex_seq(250, seed=seed))

    # Candidate visiting a complex node twice (so revisited_fraction > 0)
    b.add_node("rev_node", make_complex_seq(350, seed=1306))
    b.add_node("uniq_node", make_complex_seq(350, seed=1307))

    b.set_ref_path("Ref#1#Chr01", ["n0+", "u+", "cen+", "v+", "n4+"])
    b.add_path("Donor#1#Chr01", ["n0+", "u+", "cen+", "rev_node+", "uniq_node+", "rev_node+", "v+", "n4+"])
    b.add_bed_region(500, 750, "CEN_AUDIT")

    inputs = b.write("audit_trail")
    _, bun1 = run_build(
        inputs,
        out_prefix=tmp_path / "audit_t1",
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500", "--threads", "1"],
    )
    _, bun2 = run_build(
        inputs,
        out_prefix=tmp_path / "audit_t2",
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500", "--threads", "2"],
    )
    assert bun1 is not None and bun2 is not None
    assert_all_global_invariants(bun1, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bun1)
    assert bun1.pan_seq == bun2.pan_seq

    # Verify structured JSON candidate logs in {p}.log (§3.8.5)
    assert len(bun1.log_json_records) >= 1
    cand_logs = [rec for rec in bun1.log_json_records if "candidate_id" in rec]
    assert len(cand_logs) >= 1
    required_keys = {"region_id", "hap", "candidate_id", "parent_id", "length", "revisited_fraction"}
    assert any(required_keys.issubset(rec.keys()) for rec in cand_logs)
    assert any(
        rec.get("revisited_fraction") is not None and float(rec["revisited_fraction"]) > 0.0
        for rec in cand_logs
    )


# ---------------------------------------------------------------------------
# 14. Hybrid P-Line and W-Line Population with Centromere and GFF3 Liftover
# ---------------------------------------------------------------------------

def test_tier4_hybrid_p_and_w_line_population_with_centromere_and_gff(tmp_path: Path) -> None:
    """Population mixing P-line chromosome assemblies and W-line draft segments across
    a centromere extract-islands region, a normal SV locus, and a multi-exon GFF3 gene.
    """
    b = SyntheticInputBuilder(tmp_path, chrom="Chr01", backbone_prefix="Ref#1#")
    cds = make_coding_seq(60, seed=1401)  # 180 bp
    b.add_node("r0", make_complex_seq(100, seed=1402) + cds[:90])  # [0, 190)
    b.add_node("r1", cds[90:] + make_complex_seq(110, seed=1403))  # [190, 390)
    b.add_node("u", make_complex_seq(250, seed=1404))             # [390, 640)
    b.add_node("cen", make_complex_seq(400, seed=1405))           # [640, 1040)
    b.add_node("v", make_complex_seq(250, seed=1406))             # [1040, 1290)

    b.add_node("p_ins", make_complex_seq(150, seed=1407))
    b.add_node("w_isl", make_complex_seq(600, seed=1408))

    b.set_ref_path("Ref#1#Chr01", ["r0+", "r1+", "u+", "cen+", "v+"])
    # P-line haplotype carrying CDS insertion between r0 and r1
    b.add_path("PDonor#1#Chr01", ["r0+", "p_ins+", "r1+", "u+", "cen+", "v+"])
    # W-line haplotype with start offset 4000 carrying centromere island
    b.add_walk("WDonor", "1", "scf1", 4000, None, ["r0+", "r1+", "u+", "cen+", "w_isl+", "v+"])

    b.add_bed_region(640, 1040, "CEN_HYB")
    b.add_gff_gene("g_hyb", "m_hyb", "+", [(101, 280)], [(101, 280, 0)])

    inputs = b.write("hybrid_pw")
    _, bundle = run_build(
        inputs,
        extra_args=["--centromere-mode", "extract-islands", "--centromere-min-island", "500"],
    )
    assert bundle is not None
    assert_all_global_invariants(bundle, inputs.gfa_path, inputs.ref_fa_path)
    assert_centromere_accounting_and_offsets(bundle)
    assert_gff3_liftover_integrity(
        inputs.ref_fa_path, inputs.gff_path, bundle.pan_fa_path, bundle.lifted_gff_path
    )
