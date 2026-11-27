# Project: PanFlat Homologous Redundancy Elimination (`DEDUP_PLAN.md`)

## Architecture
Extends PanFlat v1's multicore build pipeline (`panflat/cli.py::main` → `linearizer.py::run_chromosome_pipeline` → `writers.py::write_build_outputs` / `merge_chromosomes`) with a four-layer algorithmic homologous redundancy elimination pipeline controlled by `--dedup-mode {off,ref,pan}` (default `off`) and `--dedup-id <float>` (default `0.90`, valid range `[0.80, 0.99]`):

1. **CLI & Config Layer (`models.py`, `cli.py`, `gfa_io.py`)**:
   - Extends `BuildConfig` with `dedup_mode: Literal["off", "ref", "pan"] = "off"` and `dedup_id: float = 0.90`.
   - Validates both arguments in `validate_build_config` (raising `CLIArgumentError`, catching out-of-range, `NaN`, and `±inf`).
   - Preserves 100% backward compatibility under `--dedup-mode off` (`585 / 585` baseline tests) and keeps centromere `extract-islands` Step B strictly fixed at `REF_SUB_MIN_IDENTITY = 0.85`.
2. **Module A — Stage 2.5a Full-Reference Homology Subtraction (`repeat_guard.py`, `linearizer.py`)**:
   - Generalizes `_evaluate_mappy_hits_on_blocks` and `subtract_reference_intervals_mappy` with `min_identity`, `short_aligner` (`k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50` for queries/sub-intervals `<= 500 bp`), `short_buf`, relaxed empty-`ref_region_seq` guard for prebuilt aligners, and `t_starts`/`t_ends` target-piece-scoped op splitting.
   - Parent constructs read-only `(ref_dedup_aligner, ref_dedup_short_aligner)` once prior to `_fork_gc_frozen()`, sharing via Linux `fork` COW across `_Stage5ForkContext` (and reusing directly in serial path) with per-worker/serial `mappy.ThreadBuffer` pairs; immediately freed (`del` + `gc.collect()`) after Stage 5 exits.
   - Executes Step 1 (reference subtraction) → Step 2 (15-mer repeat guard on surviving `>= 500 bp` residuals) → Step 3 (`fold_tandem_cycles` on `>= min_sv_len` residuals) inside `screen_and_fold_candidate`, propagating `r_partial` (OR'd) and `r_left_contig`, never stitching across masked intervals (`mask_intervals=()`), and logging `REFERENCE_LIKE` / `SPLIT_BY_REFERENCE_SUBTRACTION` / `SHORT_RESIDUAL` only to `.log` JSON (never `unresolved.tsv`/`unresolved.fa`).
3. **Module B — Stage 2.5b Childless Secondary-Backbone Unlocking (`linearizer.py`)**:
   - In `_solve_multi_member_component` (`not active_child_solutions`), sets `is_synthetic=(config.dedup_mode == "off")` so pure single-donor secondary backbones participate in Step 4 similarity clustering (`solve_parallel_group`).
   - Hardens `solve_parallel_group` `merged_rep0` handling to evaluate `occ.seq != rc.seq` individually via `_check_similarity_merge_with_span` (cached by `occ.seq`), preventing false `"0"` genotypes on `m_child` deletion carriers.
4. **Module C — Stage 5.5 Cross-Group/Cross-Rep Deterministic Priority Homologous Subtraction (`linearizer.py`, `--dedup-mode pan`)**:
   - Executes strictly AFTER all `fork` pools (Stages 2.3b, 2.4, 2.5) have exited and `ref_dedup_*` aligners are freed (`MULTICORE_PLAN.md` §2.3 fork-then-thread redline).
   - Priority-0 seeds accepted pool from `cen_out.island_slot_blocks`.
   - Orders all normal reps by total T22-deterministic `priority_key = (-rep.support_hap_count, ref_e_open - ref_s_open, -rep.length_bp_no_n, rank_a, rank_b, rep.rep_idx, rep.token_str)`.
   - Uses O(N), byte-identical wave buffering (`1.5 Mb` sealed waves, `~2 Mb` query batches, `384 kb` mini-subwaves + zero-false-negative dual-regime `_TailSeedGate` + lazy-dirty `tail_wave` + mid-batch wave-seal carryover) with `ThreadPoolExecutor` + per-thread `(buf_main, buf_short)` pairs and target-piece-scoped `'N'*200` hit splitting.
   - Slices surviving pieces via `_subslice_piece` with strict Global Invariant II `join_before` state transitions (no orphan `"spacer"` at `p_i == 0`; forced `"spacer"` at `p_i > 0` across any split/trim/drop gap), lockstep `rep.piece_cand_ids` updates, and untouched `< min_sv_len` sub-piece preservation.
5. **Module D — Stage 6–7 Compaction, Safe Genotyping, Summary & Associative Merge (`linearizer.py`, `writers.py`)**:
   - Compacts surviving reps densely (`1..K'`), maps dropped reps to `-1` in `cand_to_rep_idx`, isolates `,partial_content=1` per allele via `sol.cand_to_flags[cid]`, updates `cand_to_reconstructed_len`, and drops empty groups cleanly (zero `INS`, zero `InsertionSlotBlock`, zero `'N'` spacer; preserves `DEL`/`CNV`/`INV`/`DUP_TRA` and co-located `CEN_ISLAND` invariants).
   - Protects `genotypes.tsv` against false-negative `"0"` in Stage 6c `(iii)` for both Module A (`ckey in guard_res.contig_had_subtracted_homolog`, `set[int]`) and Module C (`r_idx == -1`), emitting `"."`.
   - Adds 6 `dedup_*` metrics to `{prefix}.summary.tsv` (keeping `ref_sub_min_identity = 0.85` untouched) with associative hierarchical aggregation in `merge_chromosomes`. `panflat/verify_qc.py` requires zero changes.

## Feature Inventory
Every Survey & `DEDUP_PLAN.md` feature is assigned a milestone (0 unassigned):

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Environment Provisioning & Live `mappy` Short-Aligner Check | Install `python3 -m pip install -e ".[dev]" && pyenv rehash` into pyenv Python `3.12.16`, verify `numpy`/`mappy`/`edlib`/`pytest`, confirm `short_aligner` (`k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50`) emits `hit.cs` on `+/-` strands | M1 | survey_1 |
| 2 | §2 CLI, Config & Validation | Add `dedup_mode`/`dedup_id` to `BuildConfig` (`models.py`), `--dedup-mode`/`--dedup-id` to `cli.py`, validation (`off|ref|pan`, `[0.80, 0.99]`, `NaN`/`inf` rejection) in `gfa_io.py::validate_build_config` | M1 | DEDUP_PLAN §2, survey_1 |
| 3 | §3 Module A.1: Generalized `repeat_guard.py` | Extend `_evaluate_mappy_hits_on_blocks` & `subtract_reference_intervals_mappy` with `min_identity`, `short_aligner`, `short_buf`, empty-`ref_region_seq` guard, and `t_starts`/`t_ends` target-piece-scoped op splitting; keep centromere Step B fixed at `0.85` | M1 | DEDUP_PLAN §3A.1, survey_1, survey_3 |
| 4 | §3 Module A.2: Stage 2.5a Ref Subtraction + Fork COW + IPC | Parent prebuilt `(ref_dedup_aligner, ref_dedup_short_aligner)` in `_Stage5ForkContext` before `_fork_gc_frozen()`, worker/serial `ThreadBuffer` pairs, post-Stage 5 `del` + `gc.collect()`, `screen_and_fold_candidate` Step 1→Step 2→Step 3 with `r_partial`/`r_left_contig` propagation, `REFERENCE_LIKE` JSON-only logging, `contig_had_subtracted_homolog: set[int]`, `support_hap_count` & conditional `token_str` IPC retention | M1 | DEDUP_PLAN §3A.2, survey_2 |
| 5 | §3 Module B: Stage 2.5b Childless Component Unlocking + `rep0` Guard | Set `is_synthetic=(config.dedup_mode == "off")` in `_solve_multi_member_component` (`not active_child_solutions`) and guard `merged_rep0` in `solve_parallel_group` for `occ.seq != rc.seq` via `_check_similarity_merge_with_span` | M1 | DEDUP_PLAN §3B, survey_2 |
| 6 | §3 Module C: Stage 5.5 Cross-Group Pan Dedup | Post-all-`fork` `_deduplicate_representatives_across_chromosome` with Priority-0 `CEN_ISLAND` seeding, T22 priority key, O(N) `1.5 Mb` wave / `128 kb` subwave / 13-mer seed gate / lazy-dirty `tail_wave` / mid-batch seal carryover, `ThreadPoolExecutor` + per-thread `ThreadBuffer` pairs, `_subslice_piece` `join_before` state machine, lockstep `piece_cand_ids`, untouched `< min_sv_len` preservation | M1 | DEDUP_PLAN §3C, survey_3 |
| 7 | §3 Module D: Stage 6–7 Compaction, Genotyping, Summary & Merge | Dense rep reindexing, `-1` mapping, per-allele `cand_to_flags` `partial_content=True` & `cand_to_reconstructed_len`, clean empty-group omission, Stage 6c `(iii)` `"."` genotype backfill (Module A + Module C), 6 `dedup_*` `summary.tsv` metrics (preserving `ref_sub_min_identity = 0.85`), associative `merge_chromosomes` aggregation | M1 | DEDUP_PLAN §3D, survey_1, survey_3 |
| 8 | §4.2 Unit & Opaque-Box CLI E2E Dedup Test Suite | Add `tests/unit/test_dedup.py` (CLI/config, `min_identity`/`short_aligner`/`t_starts`/`t_ends`, Module B + `m_child` `rep0`, `_subslice_piece` 7 transitions, wave/subwave/carryover determinism, associative merge) and `tests/e2e/test_tier5_dedup.py` (ZERO `panflat` imports; Tests 1–4: ectopic TE + GFF3, pericentromeric anchor drift, 20+ hap `--qc` closed-loop <2% with 100% recall, T22 multicore + shuffled GFA 12-file SHA-256 identity) | M2 | DEDUP_PLAN §4.2, survey_1–3 |
| 9 | Independent Whole-Project Verification Gate | Batched (`<= 2` concurrent) 2 Reviewers + 2 Challengers + 1 Whole-Project Forensic Auditor verifying all static gates, `panflat_v1_final_revised.md` 0-byte diff, full `585 + new` regression, adversarial probes, and T22 determinism (noting real `Chr06A` data absence on VM) | M3 | DEDUP_PLAN §4.1–4.3, survey_1–3 |

## Milestones
*(Executed sequentially by Project Orchestrator without sub-orchestrators to enforce strict user VM safety cap: at most 2 concurrent child subagents / ≤ 4 total active agents.)*

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Survey | 3 read-only Survey Explorers (`explorer_survey_1..3`) mapping CLI/Config, `repeat_guard.py`, `linearizer.py` Modules A–D, `writers.py`, `verify_qc.py`, tests, and `Chr06A` availability | none | DONE (`explorer_survey_1..3` retired & killed) |
| 1 | M1: Env Provisioning & Core Implementation | Features #1–#7 across `panflat/{models,cli,gfa_io,repeat_guard,linearizer,writers}.py`; verify static gates & `585 / 585` baseline regression under `--dedup-mode off` | M0 | DONE (`worker_core_impl`, 585/585 pass, static gates pass) |
| 2 | M2: Dedup Unit & Opaque-Box CLI E2E Tests | Feature #8: `tests/unit/test_dedup.py` + `tests/e2e/test_tier5_dedup.py` (Tests 1–4 + edge cases), resolve any uncovered edge-case bugs, verify full `585 + new` suite & static gates | M1 | DONE (`worker_dedup_tests`, 611/611 pass, 26 new tests, 0 core bugs) |
| 3 | M3: Verification & Cleanup | Streamlined implementation matching `DEDUP_PLAN.md` (Modules A–D + Stage 5.5 `_TailSeedGate` wave buffering); full regression & static gates verified | M2 | DONE (617/617 pass) |

## Interface Contracts

### `panflat/models.py` ↔ `panflat/cli.py` ↔ `panflat/gfa_io.py`
- `BuildConfig`: adds `dedup_mode: Literal["off", "ref", "pan"] = "off"` and `dedup_id: float = 0.90`.
- `validate_build_config(config)`: enforces `config.dedup_mode in ("off", "ref", "pan")` and `0.80 <= config.dedup_id <= 0.99`, raising `CLIArgumentError`.

### `panflat/repeat_guard.py` ↔ `panflat/linearizer.py` & `panflat/centromere.py`
- `_evaluate_mappy_hits_on_blocks(hits, query_offset, q_len, num_blocks, min_sv_len, min_identity=REF_SUB_MIN_IDENTITY, t_starts=None, t_ends=None)`
- `subtract_reference_intervals_mappy(query_seq, ref_region_seq="", min_sv_len=50, aligner=None, buf=None, min_identity=REF_SUB_MIN_IDENTITY, short_aligner=None, short_buf=None, t_starts=None, t_ends=None, collect_short_slices=None)`
- Centromere Step B (`panflat/centromere.py:1340`) passes defaults only → locked at `0.85` (`asm20` only).

### `panflat/linearizer.py` Internal Stage 2.5 → Stage 5.5 → Stage 6–7
- `GroupRepeatGuardResult`: adds `contig_had_subtracted_homolog: set[int]`, `ref_masked_bp: int`, `short_residual_bp: int` (keyed by `analysis.contig_idx: int` matching Stage 6c `ckey`).
- `ClusterRepresentative`: adds `support_hap_count: int = 0` (`len(rep.support_haps)` set in `solve_parallel_group`); `rep.token_str` preserved across Stage 5 IPC iff `config.dedup_mode == "pan"` and cleared after Stage 5.5.
- `ParallelGroupSolution.cand_to_rep_idx`: `-1` sentinel marks candidates whose rep was 100% subtracted in Stage 5.5; Stage 6c `(iii)` emits `"."` (never `"0"`).
- Fork-then-Thread Redline: Stage 5.5 `ThreadPoolExecutor` runs strictly after all `fork` pools (Stages 2.3b, 2.4, 2.5) have exited and `ref_dedup_*` aligners have been deleted + `gc.collect()`.

### `panflat/linearizer.py` ↔ `panflat/writers.py`
- `summary.tsv` adds `dedup_mode`, `dedup_id`, `dedup_ref_masked_bp`, `dedup_pan_masked_bp`, `dedup_short_residual_bp`, `dedup_dropped_groups` while leaving `ref_sub_min_identity = 0.85` unchanged.
- `merge_chromosomes`: isolates top-level `dedup_*` keys from already-merged inputs (`top_level_fallbacks` only copies `("spacer_n", "backbone_prefix")`) and computes associative top-level aggregates purely from `per_chrom_kv`.

## Code Layout
- `panflat/` (modified in M1): `models.py`, `cli.py`, `gfa_io.py`, `repeat_guard.py`, `linearizer.py`, `writers.py` (`verify_qc.py` and `centromere.py` require zero changes).
- `tests/unit/` (added in M2): `test_dedup.py` (alongside existing 7 unit test files, `217` baseline tests).
- `tests/e2e/` (**ZERO `panflat` imports permitted!**; added in M2): `test_tier5_dedup.py` (alongside existing 7 E2E test files, `368` baseline tests).
- Immutable spec (0-byte diff required): `panflat_v1_final_revised.md`.
- Agent metadata ONLY (never source/tests/data): `.agents/teamwork/`.

## Acceptance Gates (`DEDUP_PLAN.md` §4)
1. `git diff --exit-code panflat_v1_final_revised.md` → 0 bytes changed
2. `ruff check --select F,E9,W panflat tests` → 0 errors
3. `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e` → 0 violations
4. Full regression (`585` baseline tests under `--dedup-mode off` + all new dedup unit & E2E tests) → 100% pass
5. §4.2 Tests 1–4 verified via independent `panflat verify` (`[INV1]`/`[INV2]`/`[INV3]`/`[CEN]`/`[UNRES]`), pure-Python GFF3 protein/transcript oracle, `--qc` closed-loop `< 2%` redundancy with `100%` novel recall, and T22 12-file SHA-256 byte-identity (`--threads 1` vs `--threads 4` with `PANFLAT_FORCE_FORK=1` + shuffled GFA)
6. Note: Real pear `Chr06A` dataset (§4.3) is not present on this VM (verified read-only); clearly disclosed in final report.
