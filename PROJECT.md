# Project: PanFlat v1 True-Multicore Build (`MULTICORE_PLAN.md`)

## Architecture
PanFlat v1's build pipeline executes four strictly ordered top-level stages orchestrated by `panflat/cli.py::main`:
1. **Stage 1 (`gfa_io.py::load_and_validate_inputs`)**: Single-threaded streaming 2-pass GFA + SHA-256 + FASTA/BED/GFF3 validation with fine-grained `stderr` progress logs.
2. **Stage 2 (`linearizer.py::run_chromosome_pipeline`)**:
   - **Stage 2.1–2.3a**: Single-threaded centromere delimitation, top-level contig orientation, and centromere fragment partitioning (`centromere.py`).
   - **Stage 2.3b (`centromere.py::process_centromere_regions`, Module D)**: `bypass` single-threaded; `extract-islands` Linux `fork` `ProcessPoolExecutor` across complete non-ref haplotypes per region sharing read-only `mappy.Aligner(seq=...)` via COW, merging strictly in `graph.all_haplotypes` order.
   - **Stage 2.4 (`linearizer.py`, Module C)**: Linux `fork` `ProcessPoolExecutor` across `normal_fragments_sorted` (LPT-ordered), with pre-warmed `ref_steps._anchor_cache`, `gc.collect()`+`gc.freeze()`, IPC stripping/O(1) rebinding of `LazyProvenanceTuple._node_lens`, slim `CompactVisitCounts` (`_multi_nids` only), explicit `_ContigAnalysis.contig_idx` (eliminating all `id(ca)` checks), read-only discipline on shared split-arm `_orig_codes`, and strict ascending `frag_idx` deterministic merge (`UNANCHORED_END` numbering).
   - **Stage 2.5 (`linearizer.py`, Module B)**: Replaces the sole existing GIL-bound `ThreadPoolExecutor` with a fused Stage 2.5a (`repeat-guard` + cycle folding) + Stage 2.5b (`solve_parallel_group`) Linux `fork` `ProcessPoolExecutor`. Fast-filters empty groups in parent, packs active groups via LPT base-budget batching, strips `aligner`/unused fields/`sol.backbone_seq`/unexported `REPEAT_LIKE.seq`, merges strictly in `sorted_group_pairs` order (assigning `REPEAT_LIKE` `UNRES_*` IDs and `seen_exported_intervals` before Stage 6a), and performs all container cleanup strictly after pool exit to prevent COW dirtying.
   - **Stage 2.6–2.7 (`linearizer.py`)**: Single-threaded event collection (`6a INV*`, `6b LARGE_JUMP`, `6c DUP_TRA/CNV/DEL/INS + REPEAT_LIKE`), Stage 7 `region_id`/`event_id` backfill, and pan-genome assembly.
3. **Stage 3 (`writers.py::write_build_outputs` → `verify_qc.py::run_qc`, Module A)**: After all `fork` pools have exited (`threading.active_count() == 1`), `--qc` runs true C-level GIL-free multithreading via `ThreadPoolExecutor` + per-thread persistent `mappy.ThreadBuffer`, pre-encoded ASCII `bytes`, adaptive LPT chunking, verbatim `mappy.Aligner(seq=...)` (preserving `mid_occ=1000`), and Stage 2 separator-bridging filter (`t_starts`/`t_ends`). Writes all 12 deterministic output files atomically.
4. **Stage 4 (`verify_qc.py::verify_build_outputs`)**: Unchanged single-threaded streaming verification of Invariants I/II/III and GFA SHA-256.

**Fork & Thread Safety Invariant**: BLAS/OpenMP thread-count env vars (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) default to `"1"` at the top of `panflat/__init__.py` and `panflat/cli.py` prior to `import numpy`; `panflat/progress.py` registers `os.register_at_fork` wrappers for `_PROGRESS_LOCK`; all `fork` pools finish and join `_ExecutorManagerThread` (never setting `max_tasks_per_child`) before Stage 3 enters `ThreadPoolExecutor`. Non-Linux platforms gracefully fall back when `"fork"` is unavailable.

## Feature Inventory
Every Survey feature from `MULTICORE_PLAN.md` and Steps 0 Explorers is assigned a milestone (0 unassigned):

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Environment Provisioning & Live `mappy` Parity Check | Install `pip install -e ".[dev]"` into pyenv Python `3.12.16` (+ `pyenv rehash`), verify `numpy`/`mappy`/`edlib`/`pytest`, confirm `mappy.ThreadBuffer` bytes parity & `mappy.Aligner` fork safety, verify baseline 525/525 tests pass | M1 | survey_1, survey_3 |
| 2 | BLAS / OpenMP Implicit Thread Guard | Set `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_NUM_THREADS` to `"1"` via `os.environ.setdefault` at top of `panflat/__init__.py` and `panflat/cli.py` before `import numpy` | M1 | MULTICORE_PLAN §2.3, survey_1 |
| 3 | Fork-Safe Progress Lock Reset | Register `os.register_at_fork(before, after_in_parent, after_in_child)` wrappers in `panflat/progress.py` guarded by `hasattr(os, "register_at_fork")` | M1 | MULTICORE_PLAN §2.3, survey_1 |
| 4 | GFA I/O Fine-Grained `stderr` Progress Logs | Add 3 `log_progress` milestones in `panflat/gfa_io.py::load_and_validate_inputs`: S/L complete (O(1) link count), P/W complete, FASTA validated (never touching `.log`) | M1 | MULTICORE_PLAN §3 & §6, survey_1 |
| 5 | Module A: `--qc` True Multithreading (`verify_qc.py`) | Add `_map_queries_parallel` (`mappy.ThreadBuffer`, pre-encoded ASCII bytes, adaptive LPT chunking, optional `t_starts`/`t_ends` filter), preserve all 4 `Aligner(seq=...)` calls verbatim (`mid_occ=1000`), `config is None` safe, per-round index `del`, `round_need_short`, `stderr` progress logs | M1 | MULTICORE_PLAN §3A, survey_1 |
| 6 | IPC & COW Model Primitives (`models.py`) | Add `_EMPTY_NODE_LENS`, `LazyProvenanceTuple.without_node_lens()` & `.rebind_node_lens()`, optional `multi_counts=None` on `CompactVisitCounts` | M1 | MULTICORE_PLAN §2.2 & §3C, survey_2 |
| 7 | Explicit `contig_idx` Replacing `id(ca)` (`linearizer.py`) | Add `contig_idx: int` to `_ContigAnalysis`, replace all 6 Stage 6 `id(ca)` blocks (`INV`, `LARGE_JUMP`, `DUP_TRA`, full `CNV`, `DEL`, partial `CNV`) and clean up in-process `id(dev)` | M1 | MULTICORE_PLAN §2.1 & §3C, survey_2 |
| 8 | Module C: Stage 2.4 Contig `fork` Parallelism (`linearizer.py`) | Pure `_analyze_single_normal_fragment`, LPT `fork` `ProcessPoolExecutor` (`n_ref >= 5_000` or `PANFLAT_FORCE_FORK=1`, capped by `PANFLAT_MAX_FORK_WORKERS`), `_get_lazy_backbone_anchor_arrays` warmup, `gc.collect()`+`cache_clear()`+`gc.freeze()`, strip/rebind `_node_lens` on `DeviationSegment` & `>64`-step `ReverseChainRecord.novel_unresolved_slices`, slim `node_visit_counts`, read-only split-arm `_orig_codes` discipline, strict ascending `frag_idx` merge & `UNANCHORED_END` emission | M1 | MULTICORE_PLAN §2.1–2.3 & §3C, survey_2 |
| 9 | Module B: Stage 2.5 Fused `fork` Parallelism (`linearizer.py`) | Fuse Stage 2.5a (`repeat-guard` + cycle folding) & 2.5b (`solve_parallel_group`), fast-filter empty groups, LPT base-budget batching (`>=10_000 bp` singletons, `<=20_000 bp` / `eff_max_groups` batches), `fork` `ProcessPoolExecutor` (`n_active >= 32` or `PANFLAT_FORCE_FORK=1`), strip `sol.backbone_seq`, `rep.aligner`/unused fields, `kept_candidates`, unexported `REPEAT_LIKE.seq`, strict `sorted_group_pairs` `UNRES_*` & `seen_exported_intervals` replay, post-pool COW-safe cleanup, remove unused `threading`/`ThreadPoolExecutor` | M1 | MULTICORE_PLAN §2.1–2.3 & §3B, survey_2 |
| 10 | Module D: Stage 2.3b `extract-islands` `fork` Parallelism (`centromere.py`) | Pure per-haplotype Steps A–C worker sharing read-only `region_aligner` via COW, LPT `fork` `ProcessPoolExecutor` (`len(work_haps) >= 2` and scale threshold or `PANFLAT_FORCE_FORK=1`), IPC-strip `steps=()` and unexported `CEN_REPEAT_LIKE.seq=""`, optional `buf` in `repeat_guard.py::subtract_reference_intervals_mappy`, deterministic `graph.all_haplotypes`-ordered parent merge preserving `{region_id}_UNRES_*` and `.log` `CANDIDATE_JSON` order | M1 | MULTICORE_PLAN §3D & §6, survey_3 |
| 11 | New Unit & Opaque-Box E2E Multicore/T22 Tests | Add `tests/unit/test_multicore.py` (white-box IPC, pickle size, `contig_idx`, `CompactVisitCounts`, `run_qc`, `progress` fork lock, BLAS env, Module D parity) and `tests/e2e/test_tier5_multicore.py` (ZERO `panflat` imports; forced-fork adversarial compact-int graph with all SV/unresolved types + `>64`-step `INV_EDGE` + split-arm `_orig_codes` + `bypass`/`extract-islands` + `--dump-repeat-sequences` + `--ref-gff` + `--qc`; natural real-threshold `ref_steps>=5000` & `>=32` active groups; 12-file SHA-256 parity + independent `panflat verify` + zero fork `DeprecationWarning`) | M2 | MULTICORE_PLAN §5.3, survey_2, survey_3 |
| 12 | 30+ Haplotype Chromosome-Scale Benchmark & Final Acceptance Verification | Benchmark 33-hap chromosome-scale synthetic graph (`--threads 1` vs `--threads 2`/`4`, with `--qc` & `extract-islands`) measuring per-stage wall time, CPU%, peak RSS, and 12-file SHA-256 parity; pass static gates (`ruff`, E2E import check, `panflat_v1_final_revised.md` diff = 0), full regression (`525 + new` tests), batched 2 Reviewers + 2 Challengers + 1 Forensic Auditor gate | M2 / Verify | MULTICORE_PLAN §5.1–5.4, survey_1–3 |

## Milestones
*(Note: Executed directly by Project Orchestrator without nested sub-orchestrators to enforce user's strict VM safety cap of ≤ 3 concurrent subagents / ≤ 5 total active agents.)*

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Survey | 3 parallel Read-only Survey Explorers covering env, Modules A/B/C/D, tests, gates | none | DONE (`explorer_survey_1..3` killed) |
| 1 | M1: Env Provisioning & Core Multicore Implementation (Foundation + Modules A, B, C, D) | Features #1–#10 across `panflat/{__init__,cli,progress,gfa_io,verify_qc,models,linearizer,centromere,repeat_guard}.py`; verify static gates & 525/525 baseline regression | M0 | DONE (`worker_core_impl`, 525/525 pass) |
| 2 | M2: Multicore/T22 Unit + E2E Test Suite & 33-Hap Benchmark | Features #11–#12: `tests/unit/test_multicore.py`, `tests/e2e/test_tier5_multicore.py`, optional `extra_env` in `tests/e2e/helpers.py`, run full suite + 33-hap benchmark in `/tmp` | M1 | DONE (`worker_tests_bench`, 543/543 pass, 1.69× 33-hap speedup) |
| 3 | M3: Independent Batched Verification Wave & Final Gate | Batch 1: `reviewer_1` (full suite + static gates) + `reviewer_2` (code/COW/IPC audit) + `auditor_1` (whole-project Forensic Audit); Batch 2a: `challenger_1` (adversarial probes); Batch 2b: `challenger_2` (solo 33-hap multicore scaling & RSS benchmark); Gate pass → Report to Sentinel for Victory Audit | M2 | DONE (Gate PASS: `auditor_1` CLEAN, `reviewer_1` APPROVE 543/543, `reviewer_2` APPROVE, `challenger_1` APPROVE, `challenger_2` APPROVE) |

## Interface Contracts

### Environment Controls (Consistent Across Modules B, C, D)
- `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMEXPR_NUM_THREADS`: defaulted to `"1"` via `os.environ.setdefault` before `numpy` import.
- `PANFLAT_FORCE_FORK`: `"1"` / `"true"` / `"yes"` / `"on"` (case-insensitive) forces `fork` execution when `config.threads > 1` and `"fork"` is supported, even below production scale thresholds. When `config.threads == 1`, execution remains strictly serial regardless of `PANFLAT_FORCE_FORK`.
- `PANFLAT_MAX_FORK_WORKERS`: optional positive integer capping `max_workers` for `fork` `ProcessPoolExecutor` pools in Stages 2.3b, 2.4, and 2.5 (does NOT cap Stage 3 `--qc` threads).
- Non-Linux fallback: `"fork" in multiprocessing.get_all_start_methods()`; if absent, cleanly falls back to serial execution in Stages 2.3b/2.4/2.5.

### IPC / Model Contracts (`panflat/models.py` ↔ `panflat/linearizer.py`)
- `LazyProvenanceTuple.without_node_lens() -> LazyProvenanceTuple`: non-mutating shallow view with `_node_lens = _EMPTY_NODE_LENS` (`np.empty(0, dtype=np.int32)`) and `_cum_cache = [None]`.
- `LazyProvenanceTuple.rebind_node_lens(node_lens: np.ndarray) -> LazyProvenanceTuple`: O(1) restores `self._node_lens = node_lens` and returns `self`.
- `CompactVisitCounts(multi_nids: np.ndarray, multi_counts: np.ndarray | None = None, ...)`: when `multi_counts is None`, `get(nid, default)` returns `2` for `nid in self._multi_set`, preserving Stage 2.5 `> 1` semantics.
- `_ContigAnalysis.contig_idx: int = -1`: deterministic integer identity replacing all `id(ca)` checks in Stage 6.

### Global `UNRES_*` Numbering & Output Order Contract (T22)
1. Stage 2.3 (`centromere.py`): `{region_id}_UNRES_*` and `CANDIDATE_JSON` strictly in `graph.all_haplotypes` order.
2. Stage 2.4 (`linearizer.py`): `UNANCHORED_END` strictly in ascending `frag_idx = 0 .. K-1` order, immediately appended to `unresolved_records`.
3. Stage 2.5 (`linearizer.py`): `REPEAT_LIKE` numbered strictly in `sorted_group_pairs` order (updating `seen_exported_intervals` iff `config.dump_repeat_sequences`), deferred in `guard_res.unresolved_records`.
4. Stage 6a (`linearizer.py`): `INV_EDGE` / `INV_CONFLICT` in `(rmin, rmax, cls_name)` order.
5. Stage 6b (`linearizer.py`): `LARGE_JUMP` in `(rank_a, rank_b)` order.
6. Stage 6c & Stage 7 (`linearizer.py`): `REPEAT_LIKE` appended in `sorted_group_pairs` order; Stage 7 backfills `region_id` & `event_id`.

### Module A (`panflat/writers.py` ↔ `panflat/verify_qc.py`)
- `run_qc(result: ChromosomeBuildResult, config: BuildConfig | None = None) -> dict[str, Any]` (unchanged public signature; alias `compute_qc_metrics = run_qc` preserved; `config is None` defaults `n_threads = 1`).
- `_map_queries_parallel(aligner_main, aligner_short, query_indices, piece_bytes, piece_lens, n_threads, t_starts=None, t_ends=None) -> list[tuple[int, list[tuple[int, int, float]]]]`.

## Code Layout
- `panflat/` (13 Python modules; no `pipeline.py`):
  - `__init__.py`, `__main__.py`, `cli.py`, `progress.py`, `models.py`, `gfa_io.py`, `normalizer.py`, `repeat_guard.py`, `centromere.py`, `linearizer.py`, `gff_liftover.py`, `writers.py`, `verify_qc.py`
- `tests/`:
  - `conftest.py`
  - `tests/unit/`: `test_m1_foundation.py`, `test_m2_normalizer_centromere.py`, `test_m3_linearizer.py`, `test_m4_liftover_writers_verify_cli.py`, `test_m7_deep_review_hardening.py`, **`test_multicore.py` (new)**
  - `tests/e2e/` (**ZERO `panflat` imports permitted!**): `helpers.py`, `test_tier1_features.py`, `test_tier2_boundaries.py`, `test_tier3_combinations.py`, `test_tier4_scenarios.py`, `test_tier5_adversarial.py`, **`test_tier5_multicore.py` (new)**
- Immutable spec file (0-byte diff required): `panflat_v1_final_revised.md`
- Agent metadata ONLY (never source/tests/data): `.agents/teamwork/`

## Acceptance Gates (`MULTICORE_PLAN.md §5`)
1. `ruff check --select F,E9,W panflat tests` → 0 errors
2. `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e` → 0 violations
3. `git diff --exit-code panflat_v1_final_revised.md` → 0 bytes changed
4. Full regression suite (`525` existing + all new unit & E2E tests) → 100% pass
5. Byte-for-byte SHA-256 match across all 12 output files (`--threads 1` vs `--threads 4`) + independent `panflat verify` + zero Python 3.12 fork `DeprecationWarning` on `stderr` for both forced-fork adversarial and natural real-threshold graphs
6. 33-haplotype chromosome-scale synthetic benchmark (`threads=1` vs `threads=2`/`4`) confirming real multicore speedup, CPU utilization, bounded RSS, and 12-file SHA-256 identity
