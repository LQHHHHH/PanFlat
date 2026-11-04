# PanFlat v1

**PanFlat v1** is a chromosome-by-chromosome PGGB pangenome graph linearizer (`panflat build`, `panflat verify`, `panflat merge`) written in Python ≥ 3.10. Using a reference path as its backbone, PanFlat merges ≥ 50 bp non-reference sequences in-situ via **Path-Relative Recursive Linearization**, augmented with:

- **Universal Repeat-Guard (`--repeat-guard on|off`)**: Screens raw normal-region source candidates ≥ 500 bp using canonical 15-mer singleton ratios in 1000 bp / 500 bp sliding windows (`singleton_ratio < 0.30`), masking repetitive intervals to `unresolved.tsv` (`REPEAT_LIKE`) while preserving unmasked residuals ≥ `min-sv-len` with exact partial-node source provenance.
- **Two-Tier Centromere Processing (`--centromere-mode off|bypass|extract-islands`)**: Expands user-supplied `--centromere-bed` intervals outward to nearest unique reference anchors `(u, v)`, computes per-haplotype traversal status and assembly length ratios (`L_path / L_ref`), and either preserves the reference intact (`bypass`) or extracts, reference-subtracts (`mappy` `asm20` 50 bp blocks), repeat-masks, clusters, and mounts novel candidate islands (`extract-islands`, `CEN_ISLAND`).
- **Deterministic Reference GFF3 Liftover (`--ref-gff`)**: Shifts 1-based closed GFF3 coordinates across upstream insertions, splits `exon` and `CDS` features interrupted by internal insertions with strand-aware transcriptional phase propagation, expands parent spans, and annotates `sv_status`, `projection_split`, `splice_site_proximal_sv`, and `sv_id`.
- **Automatic Verification of Global Invariants I, II, and III (`panflat verify`)**:
  - **Invariant I**: Every reference step's bases are preserved in exact order without loss or overlap (`offsets.tsv` vs `--ref-fa`, case-insensitive).
  - **Invariant II**: Every non-`N` output piece in `pan.fa` is an exact forward or reverse-complement substring of an original GFA path/walk interval (`inserts.tsv` `src_path`, `src_intervals`, `src_strand`), including native source `N`s, with all synthetic and spacer joins explicitly recorded.
  - **Invariant III**: `ref_to_pan.chain`, `offsets.tsv`, and `pan.fa` coordinates agree across all mapping block boundaries, insertion flanks, total lengths, and fixed-seed random coordinate samples.

---

## Dependencies & Installation

- **Python**: ≥ 3.10
- **Python Libraries**: `numpy`, `mappy`, `edlib` (`pytest` for test suite)
- **Optional External CLI Tools**: `minimap2` (used by `--qc`), `gffread` (for downstream transcript/protein verification)

```bash
pip install -e .
```

---

## Recommended ODGI Preprocessing

PanFlat v1 takes a chromosome-level GFA v1 graph satisfying the §2.1 Input Contract (explicit `S` node sequences, `0M` link/path overlaps, no `J` lines, unique reference path matching `--backbone-prefix`). If starting from an ODGI graph:

```bash
# ref_paths.txt: one reference path name per line, e.g. Nipponbare#1#Chr01
odgi groom  -i raw.og -o g.og  -R ref_paths.txt -t 16
odgi unchop -i g.og   -o u.og  -t 16
odgi sort   -i u.og   -o s.og  -H ref_paths.txt -p Ygs -t 16
odgi view   -i s.og -g > chr01.clean.gfa
```

---

## Command-Line Usage

### 1. Build a Chromosome Linear Pangenome (`panflat build`)

```bash
panflat build \
  --gfa chr01.clean.gfa \
  --ref-fa nipponbare.fa \
  --backbone-prefix "Nipponbare#1#" \
  --ref-gff nipponbare.gff3 \
  --min-sv-len 50 \
  --max-jump 500000 \
  --cluster-id 0.95 \
  --cluster-win-id 0.70 \
  --sec-backbone support \
  --max-copy 1 \
  --spacer-n 100 \
  --splice-window 8 \
  --threads 16 \
  --out-prefix rice_chr01
```

#### Centromere Processing Modes (`--centromere-mode`)

```bash
# Bypass mode: preserve reference centromere interval and log non-ref blocks as CEN_BYPASS
panflat build \
  --gfa chr01.clean.gfa \
  --ref-fa nipponbare.fa \
  --backbone-prefix "Nipponbare#1#" \
  --centromere-bed centromeres.bed \
  --centromere-mode bypass \
  --out-prefix rice_chr01_bypass

# Extract-islands mode: extract, subtract reference, repeat-mask, cluster, and mount CEN_ISLANDs
panflat build \
  --gfa chr01.clean.gfa \
  --ref-fa nipponbare.fa \
  --backbone-prefix "Nipponbare#1#" \
  --centromere-bed centromeres.bed \
  --centromere-mode extract-islands \
  --centromere-min-island 500 \
  --out-prefix rice_chr01_islands
```

### 2. Verify Output Invariants (`panflat verify`)

`panflat build` runs verification automatically upon completion. You can also run standalone verification:

```bash
panflat verify \
  --prefix rice_chr01 \
  --gfa chr01.clean.gfa \
  --ref-fa nipponbare.fa
```

### 3. Merge Per-Chromosome Outputs (`panflat merge`)

```bash
panflat merge \
  --prefix-list chr_prefixes.txt \
  --out-prefix rice_all
```

---

## Output Files

All coordinates (except GFF3) use 0-based half-open `[start, end)` intervals:

| File | Description |
|---|---|
| `{p}.pan.fa` | Linearized pangenome FASTA (sequence name matches `--ref-fa`) |
| `{p}.offsets.tsv` | Reference step-to-pan coordinate mapping (`chrom`, `ref_step`, `node_id`, `ref_start`, `ref_end`, `pan_start`, `pan_end`, `slot_ins_len`) |
| `{p}.ref_to_pan.chain` | UCSC chain file mapping reference (`target`) to pan (`query`) |
| `{p}.sv.tsv` | 12-column structural variant & `CEN_ISLAND` call table |
| `{p}.genotypes.tsv` | Event × haplotype (`Sample#Hap`) genotype matrix |
| `{p}.inserts.tsv` | Piece-level provenance table (`src_hap`, `src_path`, `src_intervals`, `src_strand`, `members`, `join_before`) used to verify Invariant II |
| `{p}.ref.lifted.gff3` | Deterministically lifted reference gene annotation (emitted when `--ref-gff` is supplied) |
| `{p}.centromere.tsv` | 27-column per-region × per-haplotype centromere traversal & accounting table (emitted when centromere mode is enabled) |
| `{p}.unresolved.tsv` | Source provenance and reason for all unplaced/masked segments |
| `{p}.unresolved.fa` | FASTA sequences of exported unresolved segments (respects `--dump-repeat-sequences`) |
| `{p}.summary.tsv` | Run statistics, invariant verification status, repeat-guard/centromere parameters, and optional `--qc` metrics |
| `{p}.log` | Detailed execution log including per-candidate JSON records for centromere/repeat filtering |
