# Project: PanFlat v1

## Architecture
PanFlat v1 is a chromosome-by-chromosome PGGB graph linearizer (`panflat build`, `panflat verify`, `panflat merge`) written in Python ≥ 3.10 using `numpy`, `mappy`, and `edlib` (with optional `minimap2` and `gffread`). Using a reference path as backbone, it merges ≥50 bp non-reference sequences in-situ via **Path-Relative Recursive Linearization**, augmented with **Universal Repeat-Guard (`--repeat-guard on|off`)**, **Two-Tier Centromere Processing (`--centromere-mode off|bypass|extract-islands`)**, **Deterministic Reference GFF3 Liftover (`--ref-gff`)**, and automatic verification of **Global Invariants I, II, and III**.

### Pipeline Data Flow
```
Input GFA (S/L/P/W) + Ref FASTA + Optional Centromere BED + Optional Ref GFF3
  │
  ▼
[panflat/gfa_io.py] (§2.1 Input Contract Validation, 0-based Source Coordinate Indexing)
  │
  ├──► [panflat/centromere.py] (§3.8.1 Outward Expansion to Unique Ref Anchors u/v, Overlap Merging)
  │
  ▼
[panflat/normalizer.py] (§3.2 Projection, §3.3.0 Whole-Contig Orientation)
  │
  ├──► Centromere Intervals (u..v) ──► [panflat/centromere.py] (§3.8.2–3.8.4 Status, bypass, extract-islands Steps A–D)
  │                                          │ (mounts CEN_ISLAND at ref_start(v))
  ▼                                          ▼
Normal Intervals ──► [panflat/normalizer.py] (§3.3.1–3.3.2 INV/INV_DISPLACED/INV_EDGE/INV_CONFLICT,
  │                                           §3.4.1 Weighted LIS, §3.4.2 Deviations D(a,b),
  │                                           §3.4.4 Pre-folding Canonical CNV, §3.4.3 Cycle Folding)
  ▼
[panflat/linearizer.py] (§3.5 Recursive Subproblems S=(B_id,[lo,hi),M) & Local Re-anchoring,
  │                      §3.6.6 Repeat-Guard on >=500bp Candidates with Partial-Node Slicing,
  │                      §3.6.1–3.6.5 Parallel Branch Groups G(a,b), Union-Find Components,
  │                      Secondary Backbone support|longest, 4-Criterion edlib/mappy Clustering,
  │                      Right-Anchor b Placement & join_before, §3.7 SV Events & §5.2 Genotypes)
  ▼
[panflat/gff_liftover.py] (§4 Deterministic GFF3 Liftover, Exon/CDS Split, Strand-Aware Phase)
  │
  ▼
[panflat/writers.py] (§5 11 Output Files: pan.fa, offsets.tsv, ref_to_pan.chain, sv.tsv,
  │                   genotypes.tsv, inserts.tsv, ref.lifted.gff3, centromere.tsv,
  │                   unresolved.tsv, unresolved.fa, summary.tsv, .log + panflat merge)
  ▼
[panflat/verify_qc.py] (§1 Invariants I, II, III & Centromere Offset Verifier + §7 --qc)
```

## Code Layout
```
/home/antigravity/LQHHHHH/PanFlat/
├── pyproject.toml                 # Package metadata & CLI entry point `panflat = panflat.cli:main`
├── .gitignore                     # Ignores .agents/, __pycache__/, *.pyc, .pytest_cache/, .venv/, build/, dist/, *.egg-info/
├── README.md                      # User documentation and usage guide
├── panflat_v1_final_revised.md    # Authoritative specification (read-only)
├── panflat/
│   ├── __init__.py
│   ├── models.py                  # Core immutable dataclasses & shared type contracts
│   ├── gfa_io.py                  # §2.1: GFA (S/L/P/W), FASTA, BED parsing & input contract validation
│   ├── repeat_guard.py            # §3.8.4-C & §3.6.6: 15-mer repeat_mask_intervals, partial-node slicing, §3.8.4-B mappy 50bp subtraction, §3.6.3 4-criterion aligner
│   ├── normalizer.py              # §3.2–§3.4: Step projection, contig orientation, INV table, weighted LIS, deviations, cycle folding, canonical CNV
│   ├── centromere.py              # §3.8: BED u/v anchor expansion, completeness/length_ratio, bypass, extract-islands Steps A–D
│   ├── linearizer.py              # §3.5–§3.7 & §5.2: Recursive subproblems, local re-anchoring, repeat-guard integration, UF components, clustering, placement, SV events & genotypes
│   ├── gff_liftover.py            # §4: Deterministic 0-based half-open GFF3 liftover, exon/CDS splitting, strand-aware phase propagation & tags
│   ├── writers.py                 # §5 & §6: Writers for all 11 output files + multi-chromosome `panflat merge`
│   ├── verify_qc.py               # §1 & §7: Global Invariants I/II/III verifier (`panflat verify`) & `--qc` alignment redundancy analyzer
│   └── cli.py                     # §6: CLI entry point (`build`, `verify`, `merge`) & deterministic multiprocessing (`--threads`)
└── tests/
    ├── conftest.py                # Shared synthetic GFA/FASTA/GFF3/BED builder helpers
    ├── unit/                      # Module-level unit tests (owned by M1–M4 workers)
    └── e2e/                       # Requirement-driven opaque-box E2E tests (Tiers 1–4 + T1–T24 + C1–C20, owned by E2E Testing Track) & Tier 5 adversarial tests
```

## Feature Inventory
Every feature from the Survey phase appears below with its assigned implementation milestone and E2E test coverage. No feature is left unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Invariant I: Reference Base Preservation | Reference step bases preserved in exact order without loss/overlap; `offsets.tsv` reconstructs `--ref-fa` base-for-base (case-insensitive). | M4 (verified in M5) | `panflat_v1_final_revised.md` §1 |
| F2 | Invariant II: Piece Provenance & Join Integrity | Every non-N output piece in `pan.fa` matches original GFA path interval (`src_path`, `src_intervals`, `src_strand`) including source `N`s; non-spacer intervals have no gaps/overlaps; all joins recorded. | M4 (verified in M5) | `panflat_v1_final_revised.md` §1 |
| F3 | Invariant III: Coordinate Consistency | `ref_to_pan.chain`, `offsets.tsv`, and `pan.fa` coordinates agree across all mapping block boundaries, flanking bases, total lengths, and fixed-seed random coordinate samples. | M4 (verified in M5) | `panflat_v1_final_revised.md` §1 |
| F4 | Reference Backbone Matching & Validation | Unique reference path matching `--backbone-prefix` in GFA and unique sequence in `--ref-fa`; assembled sequence matches `--ref-fa` (case-insensitive, exact `N` match, supports `-` ref steps). | M1 | `panflat_v1_final_revised.md` §2.1 |
| F5 | GFA P-line and W-line Parsing | Parse `P` lines (`Sample#Hap#Seq`) and `W` lines (`sample`, `haplotype`, `seqid`, `start`, `end`, `walk`) as independent contiguous segments without cross-gap stitching; preserve 0-based source coordinates and W-line `start` offset; reject P/W duplicate ambiguity. | M1 | `panflat_v1_final_revised.md` §2.1 |
| F6 | GFA Structural Integrity Checks | String node IDs; reject `*` node sequences, non-zero overlaps (`L` or `P`), `P` `*` overlap when `L` is not `0M`, broken links, or `J` lines; compute GFA SHA-256. | M1 | `panflat_v1_final_revised.md` §2.1 |
| F7 | Step Projection onto Backbone `B` | Project branch steps onto `B[lo:hi)` as `(rank_in_B | None, rel_orient)` where anchors appear once in `B[lo:hi)` and `rel_orient` is `+` if step sign matches `B` step sign else `-`. | M2 | `panflat_v1_final_revised.md` §3.1–3.2 |
| F8 | Top-level Whole-Contig Orientation | Flip entire contig (`src_strand='-'`) at top level if total base length of relative-reverse anchors exceeds relative-forward anchors; not counted as `INV`. | M2 | `panflat_v1_final_revised.md` §3.3.0 |
| F9 | Reverse Chain Detection & Inversion Normalization | Maximal decreasing relative-reverse anchor runs `R` (`[rmin, rmax]`) classified via `L`/`F` flanks and rank-interval overlap into `INV`, `INV_DISPLACED`, `INV_EDGE`, `INV_CONFLICT`; normalize `INV`/`INV_DISPLACED` while preserving pre-flip source coords; route `INV_EDGE`/`INV_CONFLICT` novel seqs to `unresolved`. | M2 | `panflat_v1_final_revised.md` §3.3.1–3.3.2 |
| F10 | Weighted LIS & Multi-hit Annotation | Node-length-weighted strictly increasing LIS on relative-forward anchors (unique ranks, source-order tie-breaking) + `multi_hit` annotation when an anchor is visited multiple times. | M2 | `panflat_v1_final_revised.md` §3.4.1 |
| F11 | Deviation Segment `D(a,b)` Extraction | Extract open branch step slice vs `B(a,b)` between adjacent LIS anchors `(a, b)` when they differ; identical repeat traversals produce no deviation. | M2 | `panflat_v1_final_revised.md` §3.4.2 |
| F12 | Cycle Folding (Tandem Repeat Compression) | Greedy left-to-right minimal period `p >= 1` detector on global non-reference tokens (`k >= 2`); keep `max_copy` copies, delete `max_copy+1..k`, record deleted intervals, split pieces at deletion boundary (`join_before='synthetic'`). | M2 | `panflat_v1_final_revised.md` §3.4.3 |
| F13 | Canonical Period CNV Counting | Pre-folding greedy period detection on `B(a,b)` and branch deviation using lexicographically smallest cyclic rotation `canonical(U)`; emit `CNV` when `|k_alt - k_ref| * len(U) >= min-sv-len`. | M2 | `panflat_v1_final_revised.md` §3.4.4 |
| F14 | Subproblem State, Progress Metric & Local Re-anchoring | Recursive state `S = (B_id, [lo, hi), M)` with strictly decreasing metric `(|M|, hi - lo)` via (a) local re-anchoring on unique nodes in `B(a,b)` or (b) component recursion; ancestor key guard guarantees termination without `max-depth`. | M3 | `panflat_v1_final_revised.md` §3.5 |
| F15 | Candidate Block Extraction & Unanchored Ends | Extract maximal contiguous runs of current-level new + global non-reference nodes (top-level reference nodes act as strict separators); `repeat_residual` flag; top-level contig ends `>= min-sv-len` go to `UNANCHORED_END`, recursive subproblems use logical `START`/`END`. | M3 | `panflat_v1_final_revised.md` §3.5 |
| F16 | Parallel Branch Groups `G(a,b)` & Component Recursion | Group deviations by `(a, b)`, exact-deduplicate step paths, partition non-backbone candidates via Union-Find on shared full node IDs, select secondary backbone (`--sec-backbone support|longest`) with deterministic tie-breaking, and recurse to build synthetic sequences. | M3 | `panflat_v1_final_revised.md` §3.6.1–3.6.2 |
| F17 | Similarity Clustering (`edlib` ≤1kb / `mappy` >1kb) | 4 mandatory merge criteria on real sequences (synthetic sequences only exact-deduplicate): (1) single forward alignment, (2) `ID_gc >= --cluster-id` (0.95), (3) `unaligned_ends(S) + internal_ins_bp(S) < --min-sv-len`, (4) sliding window of length `min-sv-len` has `matches / aligned_cols >= --cluster-win-id` (0.70). | M1 (primitive) & M3 (group clustering) | `panflat_v1_final_revised.md` §3.6.2–3.6.3 |
| F18 | Unified Output Block Layout & Join Annotation | Order representatives (`B(a,b)` first if non-empty, then support desc, length desc, token asc); insert non-backbone reps before right anchor `b` with `N*spacer`; order multi-groups at `b` by span `(b - a)` asc then `a` asc; annotate `join_before` (`source_contiguous`, `spacer`, `synthetic`). | M3 | `panflat_v1_final_revised.md` §3.6.4–3.6.5 |
| F19 | Universal Repeat Guard (`--repeat-guard on\|off`) | Screen raw normal-region candidates `>= 500 bp` via `repeat_mask_intervals` (1000bp window, 500bp step + right tail window, canonical 15-mer `singleton_ratio < 0.30` in ≥2 adjacent low windows or 500–1000bp single window); record `REPEAT_LIKE` in `unresolved.tsv`; retain separate unmasked residuals `>= min-sv-len` with partial-node slicing, `.` genotype if all masked, `partial_content=1` if partial. | M1 (primitive) & M3 (pipeline integration) | `panflat_v1_final_revised.md` §3.6.6, §3.8.4-C |
| F20 | Top-level SV Event Calling & Genotyping | Call `INS`, `DEL`, `CNV`, `INV`, `INV_DISPLACED`, `INV_EDGE`, `INV_CONFLICT`, `DUP_TRA`, `LARGE_JUMP` with §3.7 priority rules and §5.2 allele signature numbering (`0`, `1..C`, `.`). | M3 | `panflat_v1_final_revised.md` §3.7, §5.2 |
| F21 | Centromere Mode CLI & BED Anchor Delimitation | Validate `--centromere-mode off|bypass|extract-islands`, `--centromere-bed`, `--centromere-min-island`; expand BED intervals outward to nearest unique reference anchors `u, v`, merge overlapping/touching expanded intervals, fallback to virtual boundary `0`/`L` (`CEN_ANCHOR_NOT_FOUND`, downgrade to `bypass`, `expanded_to_whole_chromosome`), slice branches at `(u, v)` (`CEN_BOUNDARY_AMBIGUOUS`). | M2 | `panflat_v1_final_revised.md` §3.8.1 |
| F22 | Centromere Branch Completeness & Length Ratio | Classify per-haplotype traversal as `complete`, `gapped`, `ambiguous`, `partial`, or `unmapped` (`ambiguous` priority); compute `L_ref`, `L_path` (counting revisits), and `length_ratio = L_path / L_ref` (only when `status==complete`, `ref_status!=gapped`, `L_ref>0`). | M2 | `panflat_v1_final_revised.md` §3.8.2 |
| F23 | Centromere `bypass` Mode | Keep reference bases in `[ref_end(u), ref_start(v))` intact without internal SV calling; record localizable non-ref blocks `>= min-sv-len` in `unresolved.tsv` as `CEN_BYPASS`. | M2 | `panflat_v1_final_revised.md` §3.8.3 |
| F24 | Centromere `extract-islands` Mode (Steps A–D) | Handle `CEN_REFERENCE_GAP`, `CEN_INCOMPLETE`, `CEN_AMBIGUOUS`; on `complete` paths extract non-ref blocks `>= min-island` (`revisited_fraction`, `CEN_SEQUENCE_UNCERTAIN`), subtract ref via `mappy` `asm20` 50bp blocks (`>=90%` aligned, `>=0.85` identity, protecting `>= min-sv-len` insertions -> `CEN_REFERENCE_LIKE`, `CEN_SHORT_RESIDUAL`), run `repeat_mask_intervals` (`CEN_REPEAT_LIKE`), cluster & mount at `ref_start(v)` as `ref + N + R1 + ... + N + v` (`CEN_ISLAND`), emit 27-col `centromere.tsv` & JSON logs. | M1 (Step B/C primitives) & M2 (centromere engine) | `panflat_v1_final_revised.md` §3.8.4–3.8.5, §5.4 |
| F25 | Deterministic Reference GFF3 Liftover (`--ref-gff`) | Convert 1-based closed `[s, e]` to 0-based half-open `[s-1, e)`, shift by preceding insertions, split `exon`/`CDS` across internal insertions sharing `ID`, recalculate `CDS` phase in transcriptional order (`+` asc, `-` desc), expand parents, annotate `sv_status`, `projection_split`, `splice_site_proximal_sv`, `sv_id`. | M4 | `panflat_v1_final_revised.md` §4 |
| F26 | Output Writers, CLI (`build`, `verify`, `merge`), Threads & `--qc` | Generate all 11 output files, auto-invoke & standalone `panflat verify`, multi-chromosome `panflat merge`, deterministic `--threads` execution, and `--qc` `minimap2`/`mappy` `asm20` alignment redundancy metrics. | M4 | `panflat_v1_final_revised.md` §5, §6, §7 |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| 1 | `M1_foundation` | Environment provisioning (`numpy`, `mappy`, `edlib`, `pytest`, `minimap2`, `gffread`), package scaffolding (`pyproject.toml`, `.gitignore`, `README.md`, `panflat/__init__.py`), core data structures (`panflat/models.py`), GFA/FASTA/BED parser & strict §2.1 validator (`panflat/gfa_io.py`), and shared repeat-guard / `mappy` 50bp subtraction / `edlib`+`mappy` 4-criterion similarity alignment primitives (`panflat/repeat_guard.py`) + unit tests (`tests/unit/test_m1_foundation.py`). Covers **F4, F5, F6** and core primitives of **F17, F19, F24**. | none | DONE (`sub_orch_m1_1`: `pyproject.toml`, `panflat/models.py`, `panflat/gfa_io.py`, `panflat/repeat_guard.py`, `32/32` unit tests + Gate PASS) |
| 2 | `M2_normalizer_centromere` | `panflat/normalizer.py` (§3.2 step projection, §3.3.0 whole-contig orientation, §3.3.1–3.3.2 reverse chains & `INV`/`INV_DISPLACED`/`INV_EDGE`/`INV_CONFLICT` normalization, §3.4.1 weighted LIS & `multi_hit`, §3.4.2 deviations `D(a,b)`, §3.4.3 greedy cycle folding, §3.4.4 canonical rotation CNV counting) and `panflat/centromere.py` (§3.8.1 BED `u, v` anchor expansion/merging/fallback & branch slicing, §3.8.2 status & `length_ratio`, §3.8.3 `bypass`, §3.8.4 `extract-islands` Steps A–D, `CEN_ISLAND` mounting & JSON logs) + unit tests (`tests/unit/test_m2_normalizer_centromere.py`). Covers **F7, F8, F9, F10, F11, F12, F13, F21, F22, F23, F24**. | M1 | DONE (`sub_orch_m2_1`: `panflat/normalizer.py`, `panflat/centromere.py`, `76/76` M1+M2 unit tests + Gate PASS) |
| 3 | `M3_linearizer_events` | `panflat/linearizer.py` (§3.5 recursive subproblem solver `S=(B_id,[lo,hi),M)`, progress metric `(|M|, hi-lo)`, local re-anchoring, candidate extraction, `repeat_residual`, `UNANCHORED_END`; §3.6.6 `--repeat-guard` integration with partial-node slicing & `partial_content=1`; §3.6.1–3.6.5 parallel branch groups `G(a,b)`, Union-Find components, `--sec-backbone support|longest`, 4-criterion clustering, right-anchor `b` placement & co-mounting with `CEN_ISLAND`, `join_before` tagging; §3.7 top-level SV calling & §5.2 allele signature genotyping) + unit tests (`tests/unit/test_m3_linearizer.py`). Covers **F14, F15, F16, F17, F18, F19, F20**. | M1, M2 | DONE (`sub_orch_m3_1`: `panflat/linearizer.py`, `125/125` M1+M2+M3 unit tests + Gate PASS) |
| 4 | `M4_liftover_writers_verify_cli` | `panflat/gff_liftover.py` (§4 deterministic GFF3 liftover, exon/CDS splitting, strand-aware phase propagation & SV tags), `panflat/writers.py` (§5 all 11 output files + §6 `panflat merge`), `panflat/verify_qc.py` (§1 Invariants I, II, III, unresolved provenance & centromere offset delta verifier + §7 `--qc`), and `panflat/cli.py` (§6 `build`, `verify`, `merge` CLI & deterministic `--threads`) + unit tests (`tests/unit/test_m4_liftover_writers_verify_cli.py`). Covers **F1, F2, F3, F25, F26**. | M1, M2, M3 | DONE (`sub_orch_m4_1`: `gff_liftover.py`, `writers.py`, `verify_qc.py`, `cli.py`, `135/135` unit tests + Gate PASS) |
| E2E | `E2E_test_track` | Parallel requirement-driven opaque-box E2E Testing Track (`TEST_INFRA.md`, `TEST_READY.md`, `tests/e2e/`). Covers all 26 features (`F1`–`F26`), all 44 mandatory specification scenarios (`T1`–`T24`, `C1`–`C20`), and Tiers 1–4 test cases. | none (parallel track) | DONE (`sub_orch_e2e_1`: `TEST_INFRA.md`, `TEST_READY.md`, `307` E2E tests across Tiers 1–4 + Gate PASS) |
| 5 | `M5_final_verification_and_hardening` | Final Milestone: **Phase 1**: Pass 100% of E2E test suite (`tests/e2e/` Tiers 1–4 and `tests/unit/`). **Phase 2**: Tier 5 white-box adversarial coverage hardening (`Challenger -> Worker -> Reviewer + Challenger + Auditor` loop). | M1, M2, M3, M4, E2E | DONE (`sub_orch_m5_1`: Phase 1 `442/442` PASS + Gate PASS; Phase 2 `30` Tier 5 white-box adversarial tests integrated, `472/472` total unit + Tiers 1–5 E2E tests PASS + Gate PASS) |
| 6 | `M6_real_data_scaling_and_verification` | PanSN multi-FASTA reference selection, bare sample-name `--backbone-prefix` resolution, and chromosome-scale Compact/Lazy NumPy memory & runtime optimizations validated on the 35-haplotype Pyrar_xinyeli Chr10A dataset (2.15 GB GFA, 1.17 GB FASTA). | M1–M5 | DONE (`worker_real_r1_3`: `BUILD_RC=0`, `VERIFY_RC=0`, `474/474` tests PASS + Gate PASS) |

## Interface Contracts

### 1. `panflat/models.py` (Shared Immutable Data Structures)
- `Step`:
  - `node_id: str`
  - `orient: Literal['+', '-']`
  - `slice_start: int = 0` (0-based start offset within the oriented node sequence; `0` for full step)
  - `slice_end: int | None = None` (`None` or `len(node)` for full step; set when repeat-guard or mappy subtraction slices inside a node)
  - `is_partial: bool = False` (if `True`, step contributes sliced bases and source coordinates, but is ignored as a full `node_id` in anchor indexing, Union-Find component partitioning, and cycle folding)
- `StepProvenance`:
  - `src_hap: str` (`Sample#Hap`)
  - `src_path: str` (original `P` path name or `W` segment identifier `Sample#Hap#Seq:start-end`)
  - `src_start: int` (0-based start on original path/contig, including `W`-line `seq_start` offset)
  - `src_end: int` (0-based end on original path/contig, `src_end - src_start == step_len`)
  - `src_strand: Literal['+', '-']` (orientation of this step in the output piece relative to the original unflipped input path/contig)
- `PathFragment`:
  - `sample: str`
  - `hap: str`
  - `hap_id: str` (`Sample#Hap`)
  - `seqid: str`
  - `src_path: str`
  - `base_offset: int` (`0` for `P` lines; `seq_start` for `W` lines)
  - `steps: tuple[Step, ...]`
  - `provenance: tuple[StepProvenance, ...]`
  - `is_w_line: bool`
  - `total_len: int`
- `GFAGraph`:
  - `gfa_path: str`
  - `gfa_sha256: str`
  - `ref_chrom: str` (sequence name from `--ref-fa`)
  - `ref_seq: str` (full uppercase sequence from `--ref-fa`)
  - `nodes: dict[str, str]` (node_id -> uppercase sequence)
  - `links: set[tuple[str, str, str, str]]` (`(from_id, from_orient, to_id, to_orient)` with `0M`)
  - `ref_fragment: PathFragment`
  - `alt_fragments: tuple[PathFragment, ...]`
  - `all_haplotypes: tuple[str, ...]` (deterministic ordered list of all `Sample#Hap` IDs, reference haplotype first)
  - `ref_step_ranges: tuple[tuple[int, int], ...]` (`(ref_start, ref_end)` for each step index `0..N_ref-1` on `ref_fragment`)
  - `global_ref_nodes: frozenset[str]` (all `node_id`s appearing anywhere in `ref_fragment`)
  - `unique_ref_step_by_node: dict[str, int]` (`node_id -> ref_step_idx` for nodes appearing **exactly once** in `ref_fragment`)
- `Piece`:
  - `src_hap: str`
  - `src_path: str`
  - `src_start: int`
  - `src_end: int`
  - `src_strand: Literal['+', '-']`
  - `seq: str`
  - `join_before: Literal['source_contiguous', 'spacer', 'synthetic']`
  - `members: tuple[str, ...]`
  - `allele_idx: int = 1`
  - `piece_idx: int = 1`
- `UnresolvedRecord`:
  - `record_id: str`
  - `region_id: str`
  - `event_id: str`
  - `src_path: str`
  - `src_start: int`
  - `src_end: int`
  - `src_strand: Literal['+', '-']`
  - `length: int`
  - `reason: str` (`UNANCHORED_END`, `LARGE_JUMP`, `INV_EDGE`, `INV_CONFLICT`, `REPEAT_LIKE`, `CEN_BYPASS`, `CEN_REPEAT_LIKE`, `CEN_SHORT_RESIDUAL`, `CEN_REFERENCE_GAP`, `CEN_INCOMPLETE`, `CEN_AMBIGUOUS`, `CEN_BOUNDARY_AMBIGUOUS`, `CEN_SEQUENCE_UNCERTAIN`)
  - `seq: str`
  - `export_to_fasta: bool`
  - `exported_fasta_id: str` (`.` if not exported)
- `SVEvent`:
  - `chrom: str`
  - `pan_start: int`
  - `pan_end: int`
  - `event_id: str`
  - `event_type: str` (`INS`, `DEL`, `CNV`, `INV`, `INV_DISPLACED`, `INV_EDGE`, `INV_CONFLICT`, `DUP_TRA`, `LARGE_JUMP`, `CEN_ISLAND`)
  - `ref_start: int`
  - `ref_end: int`
  - `anchors: str`
  - `alleles_str: str`
  - `n_carrier: int`
  - `n_called: int`
  - `region_id: str`
  - `genotypes: dict[str, str]` (`hap_id -> genotype string`)
  - `pieces: tuple[Piece, ...]`
- `InsertionSlotBlock`:
  - `right_anchor_ref_step: int` (`0 <= step <= len(ref_steps)`, where `len(ref_steps)` means after final reference step if ever applicable, otherwise `0..len(ref_steps)-1` before step `b`)
  - `is_centromere_island: bool` (`True` for `CEN_ISLAND` blocks, which mount before normal groups at the same `ref_start(v)`)
  - `group_span: int` (`b - a` in reference step indices)
  - `left_anchor_ref_step: int` (`a`)
  - `event_id: str`
  - `representatives: tuple[tuple[Piece, ...], ...]` (each representative is a sequence of `Piece`s; representatives are separated by `N * spacer_n`)
  - `trailing_spacer: bool` (`True` for `CEN_ISLAND` when mounted directly before `v` without a following normal group, so `ref + N + R1 + ... + N + v` holds)
- `CentromereRegion`:
  - `region_id: str`
  - `chrom: str`
  - `input_bed_ids: tuple[str, ...]`
  - `bed_start: int`
  - `bed_end: int`
  - `ref_start: int`
  - `ref_end: int`
  - `anchor_u_step: int | None` (`None` if `VIRTUAL_START`)
  - `anchor_v_step: int | None` (`None` if `VIRTUAL_END`)
  - `anchor_u_name: str`
  - `anchor_v_name: str`
  - `effective_mode: Literal['bypass', 'extract-islands']`
  - `anchor_not_found: bool`
  - `expanded_to_whole_chromosome: bool`
- `CentromereRow`:
  - Exact 27 fields matching `{p}.centromere.tsv` header:
    `chrom`, `region_id`, `input_bed_ids`, `bed_start`, `bed_end`, `ref_start`, `ref_end`, `anchor_u`, `anchor_v`, `mode`, `hap`, `status`, `ref_status`, `src_path`, `src_start`, `src_end`, `src_strand`, `L_ref`, `L_path`, `length_ratio`, `nonref_input_bp`, `reference_like_bp`, `repeat_like_bp`, `short_bp`, `unresolved_bp`, `accepted_source_bp`, `supported_events`
- `BuildConfig`:
  - All CLI parameters from §6 (`gfa`, `ref_fa`, `backbone_prefix`, `out_prefix`, `ref_gff`, `min_sv_len=50`, `max_jump=500000`, `cluster_id=0.95`, `cluster_win_id=0.70`, `sec_backbone='support'`, `max_copy=1`, `spacer_n=100`, `splice_window=8`, `threads=1`, `repeat_guard='on'`, `centromere_bed=None`, `centromere_mode='off'`, `centromere_min_island=500`, `dump_repeat_sequences=False`, `qc=False`)

### 2. `panflat/gfa_io.py` ↔ Downstream Modules
- `load_and_validate_inputs(config: BuildConfig) -> GFAGraph`
- `parse_fasta(path: str) -> dict[str, str]`
- `parse_bed(path: str, ref_chrom: str, ref_len: int) -> list[tuple[str, int, int, str]]`
- `revcomp(seq: str) -> str`
- `spell_steps(steps: Sequence[Step], nodes: Mapping[str, str]) -> str`

### 3. `panflat/repeat_guard.py` ↔ `normalizer.py`, `centromere.py`, `linearizer.py`
- `repeat_mask_intervals(seq: str) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]`:
  - Implements §3.8.4-C (1000 bp window, 500 bp step + right-aligned tail window, 15-mer canonical `singleton_ratio < 0.30`, `<90%` valid ACGT -> `unknown`, ≥2 adjacent `low` windows or 500–1000 bp single `low` window). Returns `(masked_intervals, window_diagnostics)`.
- `slice_candidate_by_intervals(steps: Sequence[Step], provenance: Sequence[StepProvenance], nodes: Mapping[str, str], keep_intervals: Sequence[tuple[int, int]], mask_intervals: Sequence[tuple[int, int]]) -> tuple[list[tuple[int, int, tuple[Step, ...], tuple[StepProvenance, ...], str]], list[tuple[int, int, tuple[StepProvenance, ...], str]]]`:
  - Slices a step+provenance walk at base-coordinate intervals `[s, e)`, creating partial-node `Step` slices when boundaries fall inside a GFA node (§3.6.6).
- `subtract_reference_intervals_mappy(query_seq: str, ref_region_seq: str, min_sv_len: int = 50) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]`:
  - Implements §3.8.4-B (`mappy.Aligner(seq=ref_region_seq, preset="asm20")`, `cs=True`, both strands, 50 bp non-overlapping query blocks, `>=90%` aligned query bases and `M/(M+X+I+D) >= 0.85`, protecting contiguous query insertions `>= min_sv_len`). Returns `(masked_intervals, unmasked_intervals, reference_like_bp)`.
- `check_similarity_merge(seq_s: str, seq_r: str, min_sv_len: int = 50, cluster_id: float = 0.95, cluster_win_id: float = 0.70) -> bool`:
  - Implements §3.6.3 4-criterion similarity check (`edlib` when both `<= 1000 bp`, `mappy` when either `> 1000 bp`, forward strand only).

### 4. `panflat/normalizer.py` & `panflat/centromere.py` ↔ `panflat/linearizer.py`
- `orient_contig_top_level(fragment: PathFragment, graph: GFAGraph) -> PathFragment` (§3.3.0)
- `delimit_centromere_regions(graph: GFAGraph, config: BuildConfig) -> tuple[list[CentromereRegion], list[str]]` (§3.8.1)
- `partition_fragments_by_centromere(oriented_fragments: Sequence[PathFragment], regions: Sequence[CentromereRegion], graph: GFAGraph) -> CentromerePartitionResult` (§3.8.1)
- `process_centromere_regions(partition: CentromerePartitionResult, graph: GFAGraph, config: BuildConfig) -> CentromereOutput` (§3.8.2–3.8.5; returns `centromere_rows`, `island_slot_blocks`, `island_sv_events`, `unresolved_records`, `json_log_entries`)
- `normalize_inversions_top_level(fragment: PathFragment, graph: GFAGraph, config: BuildConfig) -> InversionNormalizationResult` (§3.3.1–3.3.2)
- `compute_weighted_lis(anchors: Sequence[ProjectedAnchor]) -> tuple[list[ProjectedAnchor], bool]` (§3.4.1)
- `extract_deviations(fragment: PathFragment, main_chain: Sequence[ProjectedAnchor], backbone_steps: Sequence[Step]) -> list[DeviationSegment]` (§3.4.2)
- `fold_tandem_cycles(steps: Sequence[Step], provenance: Sequence[StepProvenance], global_ref_nodes: AbstractSet[str], max_copy: int) -> CycleFoldResult` (§3.4.3)
- `detect_canonical_cnv(ref_steps: Sequence[Step], alt_steps: Sequence[Step], nodes: Mapping[str, str], min_sv_len: int) -> CNVCall | None` (§3.4.4)

### 5. `panflat/linearizer.py` ↔ `panflat/writers.py`, `panflat/gff_liftover.py`, `panflat/verify_qc.py`
- `run_chromosome_pipeline(graph: GFAGraph, config: BuildConfig) -> ChromosomeBuildResult`:
  - Coordinates centromere delimitation, whole-contig orientation, centromere vs normal partitioning, inversion normalization, weighted LIS, recursive linearization, repeat-guard, clustering, SV & genotype calling, coordinate offset calculation, and GFF3 liftover.
- `liftover_gff3(gff_path: str, ref_chrom: str, step_offsets: Sequence[StepOffsetRow], insertion_events: Sequence[SVEvent], splice_window: int = 8) -> list[str]` (§4)
- `write_build_outputs(result: ChromosomeBuildResult, config: BuildConfig) -> dict[str, str]` (§5)
- `verify_build_outputs(prefix: str, gfa_path: str, ref_fa_path: str, ref_gff_path: str | None = None) -> VerificationReport` (§1)
- `merge_chromosomes(prefix_list_path: str, out_prefix: str) -> None` (§6)
- `run_qc(result: ChromosomeBuildResult, config: BuildConfig) -> dict[str, Any]` (§7)
