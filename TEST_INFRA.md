# PanFlat v1 — End-to-End (E2E) Test Infrastructure (`TEST_INFRA.md`)

## 1. Overview & Opaque-Box Testing Philosophy
The PanFlat v1 E2E test suite (`/home/antigravity/LQHHHHH/PanFlat/tests/e2e/`) is a **requirement-driven, opaque-box test suite** designed directly from the authoritative specification (`/home/antigravity/LQHHHHH/PanFlat/panflat_v1_final_revised.md`) and `/home/antigravity/LQHHHHH/PanFlat/PROJECT.md`.

### Core Rules
1. **Strict CLI Boundary**: Every E2E test invokes PanFlat via its documented CLI (`python3 -m panflat.cli build ...`, `python3 -m panflat.cli verify ...`, `python3 -m panflat.cli merge ...`) in a subprocess and validates exit codes, stdout/stderr, and the 11 output files on disk.
2. **Zero Internal Dependencies**: `tests/e2e/` does NOT import any internal module from `panflat/` to build fixtures or compute expected answers. All synthetic inputs (GFA v1 with `S`, `L`, `P`, `W`, `J` lines, reference FASTA, GFF3, and centromere BED) and all output verifications are implemented independently in `tests/e2e/helpers.py`.
3. **Deterministic & Compact Fixtures**: All synthetic nucleotide sequences are generated from fixed PRNG seeds (`random.Random(seed)`) so that k-mer compositions, alignment identities, and coordinates are 100% reproducible across runs and fast enough to execute on 2 vCPUs.

---

## 2. Shared E2E Harness & Independent Oracles (`tests/e2e/helpers.py`)

### 2.1 CLI Subprocess Runner
- `run_panflat_cli(args: Sequence[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]`
  - Executes `[sys.executable, "-m", "panflat.cli", *args]` with `PYTHONPATH=/home/antigravity/LQHHHHH/PanFlat`.
  - Captures `returncode`, `stdout`, and `stderr`.
- `run_build(...)`, `run_verify(...)`, `run_merge(...)`: Convenience wrappers around `run_panflat_cli` that return parsed output directories/prefixes and `CompletedProcess`.

### 2.2 Synthetic Sequence & Graph Builders
- `make_complex_seq(length: int, seed: int = 42) -> str`:
  - Generates a deterministic high-complexity ACGT sequence whose canonical 15-mer `singleton_ratio` is guaranteed `>= 0.90` in every window (never triggers `repeat_mask_intervals`).
- `make_satellite_seq(unit_len: int, total_len: int, seed: int = 100, divergence: float = 0.0) -> str`:
  - Generates a tandem repeat array from a periodic unit of length `unit_len` with optional controlled point divergence, allowing exact construction of windows with `singleton_ratio < 0.30` (e.g. `0.05`), `== 0.30`, `0.45`, or `>= 0.60`.
- `make_seq_with_exact_singleton_ratio(window_len: int, target_singleton_count: int, seed: int = 200) -> str`:
  - Constructs a window of length `window_len` (where `N_pos = window_len - 14`) having an exact number of singleton canonical 15-mers (`target_singleton_count / N_pos`), enabling exact boundary tests around `0.30` (`C10`, `C19`).
- `mutate_seq_snps(seq: str, snp_positions: Sequence[int] | int, seed: int = 300) -> str`:
  - Introduces deterministic point substitutions at specified or evenly spaced positions (for `T2`, `T5`, `C12`).
- `revcomp(seq: str) -> str`:
  - Exact DNA reverse-complement (preserving `N`/`n`).
- `SyntheticGFABuilder`:
  - Builds GFA v1 files with `S` (nodes), `L` (links with configurable overlap, default `0M`), `P` (paths with PanSN names and configurable overlaps), `W` (walk lines with `sample`, `hap`, `seqid`, `start`, `end`, `walk`), and optional `J` lines (for negative contract tests).
  - Writes matching `--ref-fa`, `--centromere-bed`, and `--ref-gff` files to `tmp_path`.

### 2.3 Independent Output Parsers & Oracles
- `OutputBundle`: Parses all output files for prefix `{p}`:
  1. `{p}.pan.fa` -> `dict[str, str]`
  2. `{p}.offsets.tsv` -> list of 8-column dicts (`chrom`, `ref_step`, `node_id`, `ref_start`, `ref_end`, `pan_start`, `pan_end`, `slot_ins_len`)
  3. `{p}.ref_to_pan.chain` -> parsed chain headers and alignment blocks `(size, dt, dq)`
  4. `{p}.sv.tsv` -> list of 12-column dicts (`chrom`, `pan_start`, `pan_end`, `event_id`, `type`, `ref_start`, `ref_end`, `anchors`, `alleles`, `n_carrier`, `n_called`, `region_id`)
  5. `{p}.genotypes.tsv` -> dict mapping `event_id -> dict[hap_id, str]`
  6. `{p}.inserts.tsv` -> list of 12-column dicts (`chrom`, `pan_start`, `pan_end`, `event_id`, `allele`, `piece`, `src_hap`, `src_path`, `src_intervals`, `src_strand`, `members`, `join_before`)
  7. `{p}.unresolved.tsv` -> list of 10-column dicts (`record_id`, `region_id`, `event_id`, `src_path`, `src_start`, `src_end`, `src_strand`, `length`, `reason`, `exported_fasta_id`)
  8. `{p}.unresolved.fa` -> `dict[str, tuple[str, str]]` (`id -> (header, seq)`)
  9. `{p}.centromere.tsv` -> list of 27-column dicts (when centromere mode active)
  10. `{p}.ref.lifted.gff3` -> parsed GFF3 feature rows (when `--ref-gff` supplied)
  11. `{p}.summary.tsv` & `{p}.log` -> parsed key-value summary table, raw log text, and parsed JSON candidate log objects.
- `assert_all_global_invariants(bundle: OutputBundle, gfa_path: Path, ref_fa_path: Path, spacer_n: int = 100)`:
  - **Oracle Invariant I**: Concatenates `pan_seq[row.pan_start:row.pan_end]` for all rows in `offsets.tsv`, asserts exact base-for-base equality (case-insensitive) with `ref_fa`, asserts `ref_start..ref_end` forms a contiguous partition `[0, len(ref_seq))`, and asserts `pan_start == prev_pan_end + slot_ins_len`.
  - **Oracle Invariant II**: Parses original GFA paths/walks (`P` and `W` lines with 0-based `seq_start` offsets) and reconstructs each piece in `inserts.tsv` from `(src_path, src_intervals, src_strand)`. Asserts exact match with `pan_seq[pan_start:pan_end]` (including native `N`s), checks that non-spacer intervals have no gaps or overlaps, and verifies `N * spacer_n` blocks at `join_before == 'spacer'`.
  - **Oracle Invariant III**: Verifies `ref_to_pan.chain` target/query lengths, `dt == 0`, `dq == slot_ins_len`, block boundary coordinates, flanking bases around every insertion, and fixed-seed random coordinate mappings between `ref_fa` and `pan.fa`.
- `assert_gff3_protein_and_transcript_identity(ref_fa_path: Path, ref_gff_path: Path, pan_fa_path: Path, lifted_gff_path: Path)`:
  - Independently splices exons (for transcript) and CDS segments (in transcriptional order: `+` ascending, `-` descending, applying `phase`) from reference and lifted GFF3, translates CDS to protein sequence, and asserts 100% nucleotide and amino acid identity (and runs `gffread` if available on `PATH`).

---

## 3. 4-Tier E2E Test Suite Layout

| File | Tier | Minimum Tests | Description |
|---|---|---|---|
| `tests/e2e/test_tier1_features.py` | Tier 1 | `>= 130` (`>= 5` per feature `F1`–`F26`) | Requirement-driven feature tests covering `F1`–`F26` (`>= 5` tests each) and all 44 mandatory specification scenarios (`T1`–`T24` and `C1`–`C20`). |
| `tests/e2e/test_tier2_boundaries.py` | Tier 2 | `>= 130` (`>= 5` per feature `F1`–`F26`) | Boundary conditions, exact numeric thresholds, degenerate topologies, and negative/error contract tests (`>= 5` tests per feature `F1`–`F26`). |
| `tests/e2e/test_tier3_combinations.py` | Tier 3 | `>= 26` | Cross-feature pairwise and multi-subsystem combinations (`C14`, `C18`, `C20`, `T22`, multi-chromosome `merge` + `verify` + `--qc`, inversion + repeat-guard + recursion). |
| `tests/e2e/test_tier4_scenarios.py` | Tier 4 | `>= 13` | Realistic multi-haplotype chromosome pangenome application scenarios modeling §8.3 & §10 (`E1`–`E5`). |

---

## 4. How to Run the E2E Suite
- **Verify test collection and counts**:
  ```bash
  pytest --collect-only -q tests/e2e
  ```
- **Run the full E2E test suite**:
  ```bash
  pytest -v tests/e2e
  ```
