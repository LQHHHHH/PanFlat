"""White-box unit tests for PanFlat v1 true-multicore execution and T22 determinism."""

from __future__ import annotations

import gc
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import mappy
import numpy as np
import pytest

import panflat.centromere as centromere_mod
import panflat.linearizer as linearizer_mod
import panflat.progress as progress_mod
from panflat.centromere import (
    delimit_centromere_regions,
    partition_fragments_by_centromere,
    process_centromere_regions,
)
from panflat.gfa_io import load_and_validate_inputs
from panflat.linearizer import (
    _ContigAnalysis,
    _ContigFragMeta,
    _fork_gc_frozen,
    _resolve_max_fork_workers,
    run_chromosome_pipeline,
)
from panflat.models import (
    BuildConfig,
    ChromosomeBuildResult,
    CompactVisitCounts,
    LazyProvenanceTuple,
    LazyStepTuple,
    StepOffsetRow,
    SVEvent,
)
from panflat.normalizer import compute_raw_visit_counts, orient_contig_top_level
from panflat.verify_qc import _map_queries_parallel, run_qc
from tests.conftest import (
    SyntheticGFAFixtureBuilder,
    make_complex_seq,
    make_satellite_seq,
    mutate_seq_snps,
    revcomp_dna,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_BLAS_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


# ---------------------------------------------------------------------------
# 1. BLAS / OpenMP Implicit Thread Guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_module", ["panflat", "panflat.cli"])
def test_blas_omp_thread_guard_defaults_to_1_and_keeps_single_os_thread(
    target_module: str,
) -> None:
    """Importing `panflat` or `panflat.cli` sets BLAS/OMP vars to '1' and prevents hidden pthreads."""
    env = dict(os.environ)
    for key in _BLAS_ENV_VARS:
        env.pop(key, None)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
    )

    probe_code = (
        f"import json, os, {target_module}\n"
        "import numpy as np\n"
        "a = np.ones((128, 128), dtype=np.float64)\n"
        "val = float((a @ a).sum())\n"
        "task_dir = '/proc/self/task'\n"
        "n_threads = len(os.listdir(task_dir)) if os.path.isdir(task_dir) else 1\n"
        "out = {\n"
        "    'env': {k: os.environ.get(k) for k in ("
        "'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'"
        ")},\n"
        "    'threads': n_threads,\n"
        "    'val': val,\n"
        "}\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in _BLAS_ENV_VARS:
        assert payload["env"][key] == "1", f"{target_module} failed to default {key} to '1'"
    assert payload["threads"] == 1, (
        f"Expected single OS thread after NumPy matmul via {target_module}, got {payload['threads']}"
    )
    assert payload["val"] == pytest.approx(128.0 * 128.0 * 128.0)


def test_blas_omp_thread_guard_respects_explicit_user_override() -> None:
    """`os.environ.setdefault` must not overwrite an explicitly pre-set BLAS/OMP thread variable."""
    env = dict(os.environ)
    for key in _BLAS_ENV_VARS:
        env.pop(key, None)
    env["OMP_NUM_THREADS"] = "2"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(REPO_ROOT)
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, panflat; print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    omp_val, oblas_val = proc.stdout.strip().split()
    assert omp_val == "2"
    assert oblas_val == "1"


# ---------------------------------------------------------------------------
# 2. Fork-Safe Progress Lock (`panflat/progress.py`)
# ---------------------------------------------------------------------------


def _fork_progress_child(
    conn: object,
    parent_lock: object,
) -> None:
    """Child target run under `fork` context to verify `_PROGRESS_LOCK` was reset."""
    buf = io.StringIO()
    old_stderr = sys.stderr
    try:
        is_new_lock = progress_mod._PROGRESS_LOCK is not parent_lock
        is_unlocked = not progress_mod._PROGRESS_LOCK.locked()
        sys.stderr = buf
        progress_mod.log_progress("fork-child-ok")
        conn.send((is_new_lock, is_unlocked, buf.getvalue()))  # type: ignore[attr-defined]
    finally:
        sys.stderr = old_stderr
        conn.close()  # type: ignore[attr-defined]


def test_progress_lock_register_at_fork_resets_cleanly_in_child() -> None:
    """Forked child receives a fresh unlocked `_PROGRESS_LOCK` and logs without deadlocking."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method not available on this platform")

    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    parent_lock = progress_mod._PROGRESS_LOCK

    proc = ctx.Process(target=_fork_progress_child, args=(child_conn, parent_lock))
    proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(timeout=10.0), "Forked child deadlocked calling log_progress"
        is_new_lock, is_unlocked, child_stderr = parent_conn.recv()
    finally:
        parent_conn.close()
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.kill()
            proc.join()

    assert proc.exitcode == 0
    assert is_new_lock is True
    assert is_unlocked is True
    assert "[panflat] fork-child-ok" in child_stderr
    assert progress_mod._PROGRESS_LOCK is parent_lock
    assert not progress_mod._PROGRESS_LOCK.locked()


def test_progress_reset_callback_replaces_held_lock() -> None:
    """Direct callback check: `_reset_progress_lock_in_child` replaces a locked lock cleanly."""
    orig_lock = progress_mod._PROGRESS_LOCK
    held = threading.Lock()
    held.acquire()
    progress_mod._PROGRESS_LOCK = held
    try:
        progress_mod._reset_progress_lock_in_child()
        assert progress_mod._PROGRESS_LOCK is not held
        assert not progress_mod._PROGRESS_LOCK.locked()
        buf = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = buf
            progress_mod.log_progress("after-reset-ok")
        finally:
            sys.stderr = old_stderr
        assert "[panflat] after-reset-ok" in buf.getvalue()
    finally:
        held.release()
        progress_mod._PROGRESS_LOCK = orig_lock


# ---------------------------------------------------------------------------
# 3. Module A (`verify_qc.py::run_qc`)
# ---------------------------------------------------------------------------


def _make_dummy_qc_config(tmp_path: Path, threads: int) -> BuildConfig:
    return BuildConfig(
        gfa=tmp_path / "dummy.gfa",
        ref_fa=tmp_path / "dummy.ref.fa",
        backbone_prefix="Ref#1#",
        out_prefix=tmp_path / "dummy_out",
        threads=threads,
        qc=True,
    )


def _build_synthetic_qc_result() -> tuple[ChromosomeBuildResult, dict[str, str]]:
    """Build a synthetic `ChromosomeBuildResult` exercising all `--qc` buckets and paths."""
    ref_steps_seqs = [
        make_complex_seq(1000, seed=6101),
        make_complex_seq(800, seed=6102),
        make_complex_seq(1000, seed=6103),
        make_complex_seq(800, seed=6104),
        make_complex_seq(1000, seed=6105),
        make_complex_seq(800, seed=6106),
    ]
    # Short pieces (50-149 bp)
    short_80 = make_complex_seq(80, seed=6110)
    short_149 = make_complex_seq(149, seed=6111)

    # Medium pieces (150-500 bp -> aligner_short)
    # 24 SNPs / 300 bp = 92.0% identity against ref_steps_seqs[0][100:400]
    snps_24 = [10 + 11 * k for k in range(24)]
    med_ref92 = mutate_seq_snps(ref_steps_seqs[0][100:400], snps_24)
    med_pair_a = make_complex_seq(300, seed=6112)
    med_pair_b = mutate_seq_snps(med_pair_a, snps_24)
    med_novel = make_complex_seq(220, seed=6113)

    # Separator-bridging trio: two novel 300 bp pieces + a 440 bp chimeric query
    # containing sep_left[-120:] + 'N'*200 + sep_right[:120] so only a separator-bridging
    # hit could ever pass blen >= 150.
    sep_left = make_complex_seq(300, seed=6114)
    sep_right = make_complex_seq(300, seed=6115)
    sep_bridge = sep_left[-120:] + ("N" * 200) + sep_right[:120]

    # Long pieces (>2000 bp -> LPT singleton chunk step=1 in _map_queries_parallel)
    long_a = make_complex_seq(2400, seed=6116)
    snps_long_44 = [25 + 48 * k for k in range(44)]  # 44 / 2200 = 2.0% divergence -> 98%
    long_b = mutate_seq_snps(long_a[100:2300], snps_long_44)

    # CEN_ISLAND pieces: one ~96% ref hit + one completely novel
    # 24 SNPs / 600 bp = 96.0% identity against ref_steps_seqs[2][100:700]
    snps_cen_24 = [12 + 24 * k for k in range(24)]
    cen_refhit = mutate_seq_snps(ref_steps_seqs[2][100:700], snps_cen_24)
    cen_novel = make_complex_seq(550, seed=6117)

    # Order pieces into slots with 100 bp 'N' spacers.
    # Place sep_left (long idx 0) and sep_right (long idx 2) at even long indices
    # and sep_bridge (long idx 1) at odd index so bit 0 pairs (sep_left, sep_right) as adjacent targets!
    slot_pieces: list[list[tuple[str, str]]] = [
        [("Chr01_INS_1", short_80), ("Chr01_INS_2", short_149)],
        [("Chr01_INS_3", sep_left), ("Chr01_INS_4", sep_bridge), ("Chr01_INS_5", sep_right)],
        [("Chr01_INS_6", med_ref92), ("Chr01_INS_7", med_pair_a), ("Chr01_INS_8", med_pair_b)],
        [("Chr01_INS_9", med_novel), ("Chr01_INS_10", long_a), ("Chr01_INS_11", long_b)],
        [("Chr01_CEN_ISLAND_1", cen_refhit), ("Chr01_CEN_ISLAND_2", cen_novel)],
    ]

    spacer = "N" * 100
    pan_parts: list[str] = []
    step_offsets: list[StepOffsetRow] = []
    inserts_rows: list[dict[str, object]] = []
    ref_pos = 0
    pan_pos = 0

    for step_idx, rseq in enumerate(ref_steps_seqs):
        slot_ins_len = 0
        if step_idx > 0:
            pieces_here = slot_pieces[step_idx - 1]
            slot_start_pan = pan_pos
            for p_idx, (ev_id, p_seq) in enumerate(pieces_here, start=1):
                pan_parts.append(spacer)
                pan_pos += len(spacer)
                p_start = pan_pos
                pan_parts.append(p_seq)
                pan_pos += len(p_seq)
                inserts_rows.append(
                    {
                        "chrom": "Chr01",
                        "event_id": ev_id,
                        "piece": p_idx,
                        "pan_start": p_start,
                        "pan_end": pan_pos,
                        "seq": p_seq,
                    }
                )
            pan_parts.append(spacer)
            pan_pos += len(spacer)
            slot_ins_len = pan_pos - slot_start_pan

        r_start = ref_pos
        r_end = ref_pos + len(rseq)
        p_start = pan_pos
        pan_parts.append(rseq)
        pan_pos += len(rseq)
        step_offsets.append(
            StepOffsetRow(
                chrom="Chr01",
                ref_step=step_idx,
                node_id=str(step_idx + 1),
                ref_start=r_start,
                ref_end=r_end,
                pan_start=p_start,
                pan_end=pan_pos,
                slot_ins_len=slot_ins_len,
            )
        )
        ref_pos = r_end

    sv_events = (
        SVEvent(
            event_id="Chr01_CEN_ISLAND_1",
            region_id="CEN_1",
            chrom="Chr01",
            event_type="CEN_ISLAND",
            pan_start=0,
            pan_end=0,
            ref_start=2800,
            ref_end=2800,
            anchors="3,4",
            alleles_str="1",
            n_carrier=1,
            n_called=2,
            genotypes={"Ref#1": "0", "Alt1#1": "1"},
        ),
        SVEvent(
            event_id="Chr01_CEN_ISLAND_2",
            region_id="CEN_1",
            chrom="Chr01",
            event_type="CEN_ISLAND",
            pan_start=0,
            pan_end=0,
            ref_start=2800,
            ref_end=2800,
            anchors="3,4",
            alleles_str="1",
            n_carrier=1,
            n_called=2,
            genotypes={"Ref#1": "0", "Alt1#1": "1"},
        ),
    )

    result = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq="".join(pan_parts),
        step_offsets=tuple(step_offsets),
        sv_events=sv_events,
        all_haplotypes=("Ref#1", "Alt1#1"),
        inserts_rows=tuple(inserts_rows),
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        summary_metrics={},
        log_lines=(),
        json_log_entries=(),
        gfa_path="dummy.gfa",
        gfa_sha256="0" * 64,
        ref_fa_path="dummy.ref.fa",
    )
    seqs_map = {
        "sep_left": sep_left,
        "sep_right": sep_right,
        "sep_bridge": sep_bridge,
    }
    return result, seqs_map


def test_module_a_run_qc_parallel_parity_config_none_and_separator_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module A: `run_qc` threads=1 vs threads=4 vs config=None parity, separator filter, and ThreadBuffer fallback."""
    res, seqs_map = _build_synthetic_qc_result()
    cfg_t1 = _make_dummy_qc_config(tmp_path, threads=1)
    cfg_t4 = _make_dummy_qc_config(tmp_path, threads=4)

    m_t1 = run_qc(res, cfg_t1)
    m_t4 = run_qc(res, cfg_t4)
    m_none = run_qc(res, config=None)

    assert m_t1 == m_t4
    assert m_t1 == m_none

    # Non-vacuous bucket and homology checks
    assert m_t1["qc_short_50_149_count"] == 2
    assert m_t1["qc_short_50_149_bp"] == 80 + 149
    assert m_t1["qc_long_ge150_count"] == 11
    assert m_t1["qc_redundancy_0.90_bp"] > m_t1["qc_redundancy_0.95_bp"] > 0
    assert m_t1["qc_redundancy_0.98_bp"] > 0
    assert m_t1["qc_cen_island_ge150_count"] == 2
    assert m_t1["qc_cen_island_ge150_bp"] == 600 + 550
    assert m_t1["qc_cen_island_ref_hit_0.90_bp"] >= 500
    assert m_t1["qc_cen_island_ref_hit_0.95_bp"] >= 500
    assert m_t1["qc_cen_island_ref_hit_0.98_bp"] == 0

    # Verify Stage 2 separator-bridging filter actively rejects cross-separator alignments
    sep = "N" * 200
    t_seq = seqs_map["sep_left"] + sep + seqs_map["sep_right"]
    t_starts = [0, len(seqs_map["sep_left"]) + len(sep)]
    t_ends = [len(seqs_map["sep_left"]), len(t_seq)]
    q_bytes = [seqs_map["sep_bridge"].encode("ascii")]
    q_lens = [len(seqs_map["sep_bridge"])]
    aln_short = mappy.Aligner(
        seq=t_seq, k=13, w=5, min_cnt=2, min_chain_score=20, best_n=50
    )
    unfiltered_hits = _map_queries_parallel(
        aligner_main=None,
        aligner_short=aln_short,
        query_indices=[0],
        piece_bytes=q_bytes,
        piece_lens=q_lens,
        n_threads=4,
    )
    filtered_hits = _map_queries_parallel(
        aligner_main=None,
        aligner_short=aln_short,
        query_indices=[0],
        piece_bytes=q_bytes,
        piece_lens=q_lens,
        n_threads=4,
        t_starts=t_starts,
        t_ends=t_ends,
    )
    assert len(unfiltered_hits[0][1]) > 0, "Expected unfiltered query to bridge across N*200 separator"
    assert filtered_hits[0][1] == [], "Expected t_starts/t_ends filter to reject separator-bridging hit"

    # Fallback when mappy.ThreadBuffer is absent
    monkeypatch.delattr(mappy, "ThreadBuffer", raising=False)
    assert not hasattr(mappy, "ThreadBuffer")
    m_fallback = run_qc(res, cfg_t4)
    assert m_fallback == m_t1


def test_module_a_run_qc_edge_cases_zero_one_piece_and_empty_pan_seq(
    tmp_path: Path,
) -> None:
    """Module A edge cases: 0-piece, 1-piece (Stage 2 skipped), and empty `pan_seq`."""
    cfg_t1 = _make_dummy_qc_config(tmp_path, threads=1)
    cfg_t4 = _make_dummy_qc_config(tmp_path, threads=4)

    ref_s = make_complex_seq(600, seed=6201)
    piece_s = make_complex_seq(250, seed=6202)
    spacer = "N" * 100

    # 0-piece
    res_zero = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq=ref_s,
        step_offsets=(StepOffsetRow("Chr01", 0, "1", 0, len(ref_s), 0, len(ref_s), 0),),
        sv_events=(),
        all_haplotypes=("Ref#1",),
        inserts_rows=(),
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        summary_metrics={},
        log_lines=(),
        json_log_entries=(),
        gfa_path="d.gfa",
        gfa_sha256="0" * 64,
        ref_fa_path="d.fa",
    )
    z1 = run_qc(res_zero, cfg_t1)
    z4 = run_qc(res_zero, cfg_t4)
    assert z1 == z4
    assert z1["qc_short_50_149_count"] == 0
    assert z1["qc_long_ge150_count"] == 0
    assert z1["qc_redundancy_0.90_bp"] == 0

    # 1-piece (n_long == 1 -> Stage 2 skipped)
    pan_one = ref_s[:300] + spacer + piece_s + spacer + ref_s[300:]
    p_st = 300 + len(spacer)
    p_en = p_st + len(piece_s)
    res_one = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq=pan_one,
        step_offsets=(
            StepOffsetRow("Chr01", 0, "1", 0, 300, 0, 300, 0),
            StepOffsetRow("Chr01", 1, "2", 300, 600, p_en + len(spacer), len(pan_one), len(piece_s) + 2 * len(spacer)),
        ),
        sv_events=(),
        all_haplotypes=("Ref#1", "Alt#1"),
        inserts_rows=(
            {
                "chrom": "Chr01",
                "event_id": "Chr01_INS_1",
                "piece": 1,
                "pan_start": p_st,
                "pan_end": p_en,
                "seq": piece_s,
            },
        ),
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        summary_metrics={},
        log_lines=(),
        json_log_entries=(),
        gfa_path="d.gfa",
        gfa_sha256="0" * 64,
        ref_fa_path="d.fa",
    )
    o1 = run_qc(res_one, cfg_t1)
    o4 = run_qc(res_one, cfg_t4)
    assert o1 == o4
    assert o1["qc_long_ge150_count"] == 1
    assert o1["qc_long_ge150_bp"] == 250

    # Empty pan_seq
    res_empty_pan = ChromosomeBuildResult(
        chrom="Chr01",
        pan_seq="",
        step_offsets=(),
        sv_events=(),
        all_haplotypes=("Ref#1",),
        inserts_rows=(
            {
                "chrom": "Chr01",
                "event_id": "Chr01_INS_1",
                "piece": 1,
                "pan_start": 0,
                "pan_end": 250,
                "seq": piece_s,
            },
        ),
        unresolved_records=(),
        centromere_rows=(),
        lifted_gff_lines=(),
        summary_metrics={},
        log_lines=(),
        json_log_entries=(),
        gfa_path="d.gfa",
        gfa_sha256="0" * 64,
        ref_fa_path="d.fa",
    )
    e1 = run_qc(res_empty_pan, cfg_t1)
    e4 = run_qc(res_empty_pan, cfg_t4)
    assert e1 == e4
    assert e1["qc_long_ge150_count"] == 1
    assert e1["qc_redundancy_0.90_bp"] == 0


# ---------------------------------------------------------------------------
# 4. `models.py` IPC Primitives
# ---------------------------------------------------------------------------


def test_lazy_provenance_without_and_rebind_node_lens_pickle_size_and_roundtrip() -> None:
    """`without_node_lens` drops a >4 MiB 1M-entry array to <8 KiB without mutating original."""
    node_lens = np.full(1_100_001, 50, dtype=np.int32)
    node_lens[0] = 0
    for i in range(1, 21):
        node_lens[i] = 30 + (i * 7)

    codes = np.array(
        [i if i % 2 else -i for i in range(1, 13)],
        dtype=np.int32,
    )
    prov_fwd = LazyProvenanceTuple(
        src_hap="Alt1#1",
        src_path="Alt1#1#ctg1",
        base_offset=100,
        orig_codes=codes,
        node_lens=node_lens,
        cum_cache=[None],
        slice_start=1,
        slice_end=10,
        is_contig_flipped=False,
        inv_intervals=((3, 7),),
        inv_starts=(3,),
    )
    prov_flip = LazyProvenanceTuple(
        src_hap="Alt2#1",
        src_path="Alt2#1#ctg2:500-2000",
        base_offset=500,
        orig_codes=codes,
        node_lens=node_lens,
        cum_cache=[None],
        slice_start=0,
        slice_end=12,
        is_contig_flipped=True,
        inv_intervals=((2, 5),),
        inv_starts=(2,),
    )

    for orig in (prov_fwd, prov_flip):
        orig_elems = list(orig)
        orig_runs = orig.merged_runs()
        orig_total_bp = orig.total_bp()

        full_pickle = pickle.dumps(orig)
        assert len(full_pickle) > 4 * 1024 * 1024, (
            f"Expected unstripped pickle > 4 MiB, got {len(full_pickle)} B"
        )

        stripped = orig.without_node_lens()
        # Original must not be mutated
        assert orig._node_lens is node_lens
        assert orig._node_lens.shape == (1_100_001,)
        assert stripped._node_lens.shape == (0,)

        slim_pickle = pickle.dumps(stripped)
        assert len(slim_pickle) < 8 * 1024, (
            f"Expected stripped pickle < 8 KiB, got {len(slim_pickle)} B"
        )

        restored = pickle.loads(slim_pickle)
        rebound = restored.rebind_node_lens(node_lens)
        assert rebound is restored
        assert restored._node_lens is node_lens
        assert list(restored) == orig_elems
        assert restored.merged_runs() == orig_runs
        assert restored.total_bp() == orig_total_bp


def test_compact_visit_counts_slim_multi_nids_matches_compute_raw_visit_counts() -> None:
    """`CompactVisitCounts(multi_nids=arr)` (`multi_counts=None`) preserves `> 1` visit semantics."""
    n_unique = 51_000
    base_codes = np.arange(1, n_unique + 1, dtype=np.int32)
    base_codes[1::2] = -base_codes[1::2]
    # Add extra visits: node 10 -> 2 visits, node 500 -> 3 visits, node 12345 -> 5 visits
    extras = np.array([10, -500, 500, 12345, -12345, 12345, -12345], dtype=np.int32)
    all_codes = np.concatenate([base_codes, extras])
    lazy_steps = LazyStepTuple(all_codes)

    full_vc = compute_raw_visit_counts(lazy_steps)
    assert isinstance(full_vc, CompactVisitCounts)
    assert full_vc._multi_set == frozenset({10, 500, 12345})
    assert full_vc.get("10", 0) == 2
    assert full_vc.get("500", 0) == 3
    assert full_vc.get("12345", 0) == 5
    assert full_vc.get("1", 0) == 1
    assert full_vc.get("99999", 0) == 0

    # Round-trip compact IPC payload (`_multi_nids` only)
    ipc_payload = pickle.dumps(full_vc._multi_nids)
    assert len(ipc_payload) < 1024
    slim_vc = CompactVisitCounts(multi_nids=pickle.loads(ipc_payload))
    assert slim_vc._multi_counts is None
    assert slim_vc._visited_bits is None
    assert slim_vc._multi_set == full_vc._multi_set

    probe_nids = ("1", "2", "10", "500", "12345", "51000", "60000", "node_str")
    for nid in probe_nids:
        assert (slim_vc.get(nid, 0) > 1) == (full_vc.get(nid, 0) > 1), (
            f"Mismatch for nid={nid}: slim={slim_vc.get(nid, 0)} vs full={full_vc.get(nid, 0)}"
        )
    assert slim_vc.get("10", 0) == 2
    assert slim_vc.get("500", 0) == 2
    assert slim_vc.get("12345", 0) == 2
    assert slim_vc["12345"] == 2
    assert slim_vc.get("1", 0) == 0
    assert "500" in slim_vc
    assert "1" not in slim_vc


# ---------------------------------------------------------------------------
# 5. Modules B & C (`linearizer.py`)
# ---------------------------------------------------------------------------


def _build_all_sv_unit_graph(
    tmp_path: Path,
    compact_int: bool,
    with_centromere: bool = False,
) -> tuple[Path, Path, Path | None]:
    """Build a multi-haplotype graph covering INV, LARGE_JUMP, DUP_TRA, CNV, DEL, INS."""
    b = SyntheticGFAFixtureBuilder()
    next_id = 1
    id_map: dict[str, str] = {}

    def _nid(logical: str) -> str:
        nonlocal next_id
        if logical not in id_map:
            if compact_int:
                id_map[logical] = str(next_id)
                next_id += 1
            else:
                id_map[logical] = logical
        return id_map[logical]

    # Reference backbone:
    # r1..r16 unique (200 bp each), with tandem repeat tr x 2 between r6 and r7,
    # and optional centromere nodes cen1+cen2 between r9 and r10.
    for i in range(1, 17):
        b.add_node(_nid(f"r{i}"), make_complex_seq(200, seed=7000 + i))
    b.add_node(_nid("tr"), make_complex_seq(80, seed=7050))
    if with_centromere:
        b.add_node(_nid("cen1"), make_complex_seq(800, seed=7061))
        b.add_node(_nid("cen2"), make_complex_seq(800, seed=7062))

    # Alt novel nodes
    b.add_node(_nid("ins_a"), make_complex_seq(260, seed=7071))
    b.add_node(_nid("ins_b"), make_complex_seq(240, seed=7072))
    b.add_node(_nid("lj_nov"), make_complex_seq(180, seed=7073))
    b.add_node(_nid("dup_nov"), make_complex_seq(160, seed=7074))
    b.add_node(_nid("sat_blk"), make_satellite_seq(20, 1200, seed=7075))
    b.add_node(_nid("unanch"), make_complex_seq(150, seed=7076))
    if with_centromere:
        b.add_node(_nid("cen_isl"), make_complex_seq(700, seed=7077))

    def _s(logical: str, orient: str = "+") -> tuple[str, str]:
        return (_nid(logical), orient)

    ref_mid = (
        [_s("cen1"), _s("cen2")] if with_centromere else []
    )
    ref_steps = [
        _s("r1"), _s("r2"), _s("r3"), _s("r4"), _s("r5"), _s("r6"),
        _s("tr"), _s("tr"),
        _s("r7"), _s("r8"), _s("r9"),
        *ref_mid,
        _s("r10"), _s("r11"), _s("r12"), _s("r13"), _s("r14"), _s("r15"), _s("r16"),
    ]
    b.add_p_path("Ref#1#Chr01", ref_steps)

    # Hap1: INV of (r2, r3), CNV (tr x 4), DEL of r8, and INS at (r10, r11)
    b.add_p_path(
        "Hap1#1#Chr01",
        [
            _s("r1"), _s("r3", "-"), _s("r2", "-"), _s("r4"), _s("r5"), _s("r6"),
            _s("tr"), _s("tr"), _s("tr"), _s("tr"),
            _s("r7"), _s("r9"),
            *([_s("cen_isl")] if with_centromere else []),
            _s("r10"), _s("ins_a"), _s("r11"), _s("r12"), _s("r13"), _s("r14"), _s("r15"), _s("r16"),
        ],
    )

    # Hap2: DUP_TRA (visits r15+ between r4+ and r5+; distance > max_jump=500),
    # repeat-guard satellite at (r10, r11), and UNANCHORED_END overhang
    b.add_p_path(
        "Hap2#1#Chr01",
        [
            _s("unanch"),
            _s("r1"), _s("r2"), _s("r3"), _s("r4"), _s("dup_nov"), _s("r15"), _s("r5"), _s("r6"),
            _s("tr"), _s("tr"),
            _s("r7"), _s("r8"), _s("r9"),
            *ref_mid,
            _s("r10"), _s("sat_blk"), _s("r11"), _s("r12"), _s("r13"), _s("r14"), _s("r15"), _s("r16"),
        ],
    )

    # Hap3: LARGE_JUMP skipping r11..r14 (800 bp > max_jump=500) carrying lj_nov
    b.add_p_path(
        "Hap3#1#Chr01",
        [
            _s("r1"), _s("r2"), _s("r3"), _s("r4"), _s("r5"), _s("r6"),
            _s("tr"), _s("tr"),
            _s("r7"), _s("r8"), _s("r9"),
            *ref_mid,
            _s("r10"), _s("lj_nov"), _s("r15"), _s("r16"),
        ],
    )

    # Hap4: co-carrier of INS (distinct allele ins_b) and DEL of r8
    b.add_p_path(
        "Hap4#1#Chr01",
        [
            _s("r1"), _s("r2"), _s("r3"), _s("r4"), _s("r5"), _s("r6"),
            _s("tr"), _s("tr"),
            _s("r7"), _s("r9"),
            *([_s("cen_isl")] if with_centromere else []),
            _s("r10"), _s("ins_b"), _s("r11"), _s("r12"), _s("r13"), _s("r14"), _s("r15"), _s("r16"),
        ],
    )

    tag = f"{'int' if compact_int else 'str'}_{'cen' if with_centromere else 'norm'}"
    gfa_path = b.write_gfa(tmp_path / f"{tag}.gfa")
    ref_seq = b.spell_steps(ref_steps)
    ref_fa = b.write_fasta(tmp_path / f"{tag}.ref.fa", {"Chr01": ref_seq})
    bed_path: Path | None = None
    if with_centromere:
        # r1..r6 (1200) + 2*tr (160) + r7..r9 (600) = 1960; cen1+cen2 = 1600 -> [1960, 3560)
        bed_path = b.write_bed(tmp_path / f"{tag}.cen.bed", [("Chr01", 1960, 3560, "CEN_U")])
    return gfa_path, ref_fa, bed_path


def _assert_build_results_equal(
    res_a: ChromosomeBuildResult,
    res_b: ChromosomeBuildResult,
) -> None:
    assert res_a.chrom == res_b.chrom
    assert res_a.pan_seq == res_b.pan_seq
    assert res_a.step_offsets == res_b.step_offsets
    assert res_a.sv_events == res_b.sv_events
    assert res_a.all_haplotypes == res_b.all_haplotypes
    assert res_a.inserts_rows == res_b.inserts_rows
    assert res_a.unresolved_records == res_b.unresolved_records
    assert res_a.centromere_rows == res_b.centromere_rows
    assert res_a.summary_metrics == res_b.summary_metrics
    assert res_a.log_lines == res_b.log_lines
    assert res_a.json_log_entries == res_b.json_log_entries


@pytest.mark.parametrize("compact_int", [True, False])
def test_modules_b_c_forced_fork_parity_and_contig_idx_genotyping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact_int: bool,
) -> None:
    """Modules B & C: forced-fork parity + explicit `contig_idx` genotypes on compact-int & string graphs."""
    gfa_path, ref_fa, _ = _build_all_sv_unit_graph(tmp_path, compact_int=compact_int)

    def _run(threads: int, force_fork: str, max_workers: str | None) -> ChromosomeBuildResult:
        monkeypatch.setenv("PANFLAT_FORCE_FORK", force_fork)
        if max_workers is None:
            monkeypatch.delenv("PANFLAT_MAX_FORK_WORKERS", raising=False)
        else:
            monkeypatch.setenv("PANFLAT_MAX_FORK_WORKERS", max_workers)
        cfg = BuildConfig(
            gfa=gfa_path,
            ref_fa=ref_fa,
            backbone_prefix="Ref#1#",
            out_prefix=tmp_path / f"out_{compact_int}_{threads}_{max_workers}",
            threads=threads,
            max_jump=500,
        )
        graph = load_and_validate_inputs(cfg)
        if compact_int:
            assert isinstance(graph.ref_fragment.steps, LazyStepTuple)
            assert len(graph.nodes._extra) == 0
        else:
            assert not isinstance(graph.ref_fragment.steps, LazyStepTuple)
        return run_chromosome_pipeline(graph, cfg)

    res_t1 = _run(threads=1, force_fork="1", max_workers=None)
    res_t4 = _run(threads=4, force_fork="1", max_workers=None)
    res_cap2 = _run(threads=4, force_fork="1", max_workers="2")
    res_cap1 = _run(threads=4, force_fork="1", max_workers="1")

    _assert_build_results_equal(res_t1, res_t4)
    _assert_build_results_equal(res_t1, res_cap2)
    _assert_build_results_equal(res_t1, res_cap1)

    ev_by_type: dict[str, list[SVEvent]] = {}
    for ev in res_t4.sv_events:
        ev_by_type.setdefault(ev.event_type, []).append(ev)

    required_types = {"INV", "LARGE_JUMP", "DUP_TRA", "CNV", "DEL", "INS"}
    assert required_types.issubset(ev_by_type.keys()), (
        f"Missing required SV types: {required_types - set(ev_by_type.keys())}"
    )

    # Verify explicit contig_idx preserved exact carrier genotypes (never degraded to '0' or '.')
    assert ev_by_type["INV"][0].genotypes["Hap1#1"] == "1"
    assert ev_by_type["INV"][0].genotypes["Ref#1"] == "0"
    assert ev_by_type["DUP_TRA"][0].genotypes["Hap2#1"] == "1"
    assert ev_by_type["LARGE_JUMP"][0].genotypes["Hap3#1"] == "1"
    assert ev_by_type["DEL"][0].genotypes["Hap1#1"] == "1"
    assert ev_by_type["DEL"][0].genotypes["Hap4#1"] == "1"
    assert ev_by_type["CNV"][0].genotypes["Hap1#1"] not in ("0", ".")
    ins_haps_called = {
        h
        for ev in ev_by_type["INS"]
        for h, gt in ev.genotypes.items()
        if gt not in ("0", ".")
    }
    assert {"Hap1#1", "Hap4#1"}.issubset(ins_haps_called)

    # Also verify _ContigAnalysis.contig_idx survives pickle round-trip despite new object id()
    ca = _ContigAnalysis(
        raw_fragment=_ContigFragMeta(hap_id="Hap1#1", src_path="Hap1#1#Chr01"),
        lis_multi_hit=False,
        covered_ranks={0, 1, 2},
        unnormalized_rank_spans=[],
        large_jump_rank_spans=[],
        node_visit_counts={},
        has_main_chain=True,
        contig_idx=7,
    )
    ca_rt = pickle.loads(pickle.dumps(ca))
    assert id(ca_rt) != id(ca)
    assert ca_rt.contig_idx == 7

    # Verify _resolve_max_fork_workers honors cap and ignores invalid strings
    monkeypatch.setenv("PANFLAT_MAX_FORK_WORKERS", "2")
    assert _resolve_max_fork_workers(8) == 2
    monkeypatch.setenv("PANFLAT_MAX_FORK_WORKERS", "1")
    assert _resolve_max_fork_workers(4) == 1
    monkeypatch.setenv("PANFLAT_MAX_FORK_WORKERS", "not-an-int")
    assert _resolve_max_fork_workers(4) == 4


def test_non_linux_fallback_when_fork_start_method_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `"fork"` is absent from `get_all_start_methods()`, Stage 2.3b/2.4/2.5 fall back safely."""
    gfa_path, ref_fa, bed_path = _build_all_sv_unit_graph(
        tmp_path, compact_int=True, with_centromere=True
    )
    assert bed_path is not None

    cfg_t1 = BuildConfig(
        gfa=gfa_path,
        ref_fa=ref_fa,
        backbone_prefix="Ref#1#",
        out_prefix=tmp_path / "fallback_t1",
        threads=1,
        max_jump=500,
        centromere_bed=bed_path,
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
        centromere_min_island=500,
    )
    cfg_t4 = BuildConfig(
        gfa=gfa_path,
        ref_fa=ref_fa,
        backbone_prefix="Ref#1#",
        out_prefix=tmp_path / "fallback_t4",
        threads=4,
        max_jump=500,
        centromere_bed=bed_path,
        centromere_mode="extract-islands",
        centromere_mode_explicit=True,
        centromere_min_island=500,
    )

    graph_t1 = load_and_validate_inputs(cfg_t1)
    res_t1 = run_chromosome_pipeline(graph_t1, cfg_t1)

    graph_cen_t1 = load_and_validate_inputs(cfg_t1)
    regs_t1, _ = delimit_centromere_regions(graph_cen_t1, cfg_t1)
    oriented_t1 = tuple(
        orient_contig_top_level(f, graph_cen_t1) for f in graph_cen_t1.alt_fragments
    )
    part_t1 = partition_fragments_by_centromere(oriented_t1, regs_t1, graph_cen_t1, cfg_t1)
    cen_t1 = process_centromere_regions(part_t1, graph_cen_t1, cfg_t1)

    # Simulate non-Linux platform where only "spawn" exists, even with PANFLAT_FORCE_FORK=1
    monkeypatch.setenv("PANFLAT_FORCE_FORK", "1")
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    orig_get_context = multiprocessing.get_context

    def _forbid_fork_ctx(method: str | None = None) -> object:
        if method == "fork":
            raise AssertionError("get_context('fork') must not be called when fork is unavailable")
        return orig_get_context(method)

    monkeypatch.setattr(multiprocessing, "get_context", _forbid_fork_ctx)
    assert linearizer_mod._supports_fork() is False
    assert centromere_mod._supports_fork() is False

    graph_t4 = load_and_validate_inputs(cfg_t4)
    res_t4 = run_chromosome_pipeline(graph_t4, cfg_t4)
    _assert_build_results_equal(res_t1, res_t4)

    graph_cen_t4 = load_and_validate_inputs(cfg_t4)
    regs_t4, _ = delimit_centromere_regions(graph_cen_t4, cfg_t4)
    oriented_t4 = tuple(
        orient_contig_top_level(f, graph_cen_t4) for f in graph_cen_t4.alt_fragments
    )
    part_t4 = partition_fragments_by_centromere(oriented_t4, regs_t4, graph_cen_t4, cfg_t4)
    cen_t4 = process_centromere_regions(part_t4, graph_cen_t4, cfg_t4)
    assert cen_t1 == cen_t4


def test_fork_gc_frozen_restores_gc_unfreeze_on_exception() -> None:
    """`_fork_gc_frozen` in both `linearizer` and `centromere` unfreezes GC even when body raises."""
    for ctx_fn in (_fork_gc_frozen, centromere_mod._fork_gc_frozen):
        gc.unfreeze()
        before = gc.get_freeze_count()
        assert before == 0
        with pytest.raises(RuntimeError, match="forced-gc-error"):
            with ctx_fn():
                assert gc.get_freeze_count() > before
                raise RuntimeError("forced-gc-error")
        assert gc.get_freeze_count() == before


# ---------------------------------------------------------------------------
# 6. Module D (`centromere.py::process_centromere_regions`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dump_repeat_sequences", [False, True])
def test_module_d_centromere_extract_islands_forced_fork_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dump_repeat_sequences: bool,
) -> None:
    """Module D: `extract-islands` fork vs serial parity across Steps B/C/D and all hap statuses."""
    b = SyntheticGFAFixtureBuilder()
    # Compact integer node IDs so fast path is active too
    u1 = make_complex_seq(100, seed=8100)
    cen_ref1 = make_complex_seq(1500, seed=8101)
    cen_ref2 = make_complex_seq(1500, seed=8102)
    v1 = make_complex_seq(100, seed=8103)

    # Island >1 kb (triggers Step D mappy clustering across HapStepB and HapCluster)
    island_1200 = make_complex_seq(1200, seed=8110)
    # 20 SNPs / 1200 bp = 1.67% divergence -> clusters with island_1200 via mappy (>1 kb)
    island_1200_var = mutate_seq_snps(island_1200, [30 + 55 * k for k in range(20)])

    # HapStepB: 500 bp ref homology + 1200 bp island + 500 bp revcomp ref homology + 300 bp short residual
    short_300 = make_complex_seq(300, seed=8111)
    step_b_seq = (
        cen_ref1[100:600]
        + island_1200
        + revcomp_dna(cen_ref2[200:700])
        + short_300
    )

    # HapStepC: satellite (2000 bp) + embedded island (1500 bp) + satellite (2500 bp),
    # split across 3 x 2000 bp GFA nodes so Step C repeat-mask intervals split partially inside nodes
    sat_left = make_satellite_seq(25, 2000, seed=8112)
    island_1500 = make_complex_seq(1500, seed=8113)
    sat_right = make_satellite_seq(25, 2500, seed=8114)
    composite_c = sat_left + island_1500 + sat_right

    # HapUncertain: 600 bp block containing IUPAC 'R' + 1200 bp pure satellite
    uncertain_600 = make_complex_seq(300, seed=8115) + "R" + make_complex_seq(299, seed=8116)
    pure_sat = make_satellite_seq(20, 1200, seed=8117)

    # HapGapped: sequence with N gap inside centromere
    gapped_seq = make_complex_seq(250, seed=8118) + ("N" * 100) + make_complex_seq(250, seed=8119)

    # HapPartial / HapAmbig nonref segments
    partial_nov = make_complex_seq(200, seed=8120)
    ambig_nov = make_complex_seq(200, seed=8121)

    nodes_in_order = [
        ("1", u1),
        ("2", cen_ref1),
        ("3", cen_ref2),
        ("4", v1),
        ("5", step_b_seq),
        ("6", composite_c[0:2000]),
        ("7", composite_c[2000:4000]),
        ("8", composite_c[4000:6000]),
        ("9", island_1200_var),
        ("10", uncertain_600),
        ("11", pure_sat),
        ("12", gapped_seq),
        ("13", partial_nov),
        ("14", ambig_nov),
    ]
    for nid, seq in nodes_in_order:
        b.add_node(nid, seq)

    ref_steps = [("1", "+"), ("2", "+"), ("3", "+"), ("4", "+")]
    b.add_p_path("Ref#1#Chr01", ref_steps)
    b.add_p_path("HapStepB#1#ctg1", [("1", "+"), ("5", "+"), ("4", "+")])
    b.add_p_path(
        "HapStepC#1#ctg1",
        [("1", "+"), ("6", "+"), ("7", "+"), ("8", "+"), ("4", "+")],
    )
    b.add_p_path("HapCluster#1#ctg1", [("1", "+"), ("9", "+"), ("4", "+")])
    b.add_p_path(
        "HapUncertain#1#ctg1",
        [("1", "+"), ("10", "+"), ("2", "+"), ("11", "+"), ("4", "+")],
    )
    b.add_p_path("HapGapped#1#ctg1", [("1", "+"), ("12", "+"), ("4", "+")])
    b.add_p_path("HapPartial#1#ctg1", [("1", "+"), ("13", "+")])
    b.add_p_path(
        "HapAmbig#1#ctg1",
        [("1", "+"), ("14", "+"), ("1", "+"), ("2", "+"), ("4", "+")],
    )

    gfa_path = b.write_gfa(tmp_path / f"cen_d_{dump_repeat_sequences}.gfa")
    ref_fa = b.write_fasta(
        tmp_path / f"cen_d_{dump_repeat_sequences}.fa",
        {"Chr01": b.spell_steps(ref_steps)},
    )
    bed_path = b.write_bed(
        tmp_path / f"cen_d_{dump_repeat_sequences}.bed",
        [("Chr01", 100, 3100, "CEN_MOD_D")],
    )

    monkeypatch.setenv("PANFLAT_FORCE_FORK", "1")
    monkeypatch.delenv("PANFLAT_MAX_FORK_WORKERS", raising=False)

    def _run_cen(threads: int):
        cfg = BuildConfig(
            gfa=gfa_path,
            ref_fa=ref_fa,
            backbone_prefix="Ref#1#",
            out_prefix=tmp_path / f"cen_out_{dump_repeat_sequences}_{threads}",
            threads=threads,
            centromere_bed=bed_path,
            centromere_mode="extract-islands",
            centromere_mode_explicit=True,
            centromere_min_island=500,
            dump_repeat_sequences=dump_repeat_sequences,
        )
        graph = load_and_validate_inputs(cfg)
        regs, _ = delimit_centromere_regions(graph, cfg)
        oriented = tuple(orient_contig_top_level(f, graph) for f in graph.alt_fragments)
        part = partition_fragments_by_centromere(oriented, regs, graph, cfg)
        return process_centromere_regions(part, graph, cfg)

    out_t1 = _run_cen(threads=1)
    out_t4 = _run_cen(threads=4)

    assert out_t1 == out_t4
    assert out_t1.centromere_rows == out_t4.centromere_rows
    assert out_t1.island_slot_blocks == out_t4.island_slot_blocks
    assert out_t1.island_sv_events == out_t4.island_sv_events
    assert out_t1.unresolved_records == out_t4.unresolved_records
    assert out_t1.json_log_entries == out_t4.json_log_entries
    assert out_t1.summary_metrics == out_t4.summary_metrics

    # Non-vacuous verification of every exercised branch
    rows_by_hap = {r.hap: r for r in out_t4.centromere_rows}
    statuses = {r.status for r in out_t4.centromere_rows}
    assert {"complete", "gapped", "partial", "ambiguous"}.issubset(statuses)

    assert int(rows_by_hap["HapStepB#1"].reference_like_bp) == 1000
    assert int(rows_by_hap["HapStepB#1"].accepted_source_bp) == 1200
    assert int(rows_by_hap["HapStepB#1"].short_bp) == 300
    assert int(rows_by_hap["HapStepC#1"].repeat_like_bp) == 4500
    assert int(rows_by_hap["HapStepC#1"].accepted_source_bp) == 1500
    assert int(rows_by_hap["HapUncertain#1"].unresolved_bp) == 600
    assert int(rows_by_hap["HapUncertain#1"].repeat_like_bp) == 1200

    unres_reasons = {u.reason for u in out_t4.unresolved_records}
    assert {
        "CEN_SHORT_RESIDUAL",
        "CEN_SEQUENCE_UNCERTAIN",
        "CEN_REPEAT_LIKE",
        "CEN_INCOMPLETE",
        "CEN_AMBIGUOUS",
    }.issubset(unres_reasons)

    rep_recs = [u for u in out_t4.unresolved_records if u.reason == "CEN_REPEAT_LIKE"]
    assert len(rep_recs) >= 2
    for rec in rep_recs:
        assert rec.export_to_fasta is dump_repeat_sequences
        if dump_repeat_sequences:
            assert len(rec.seq) == rec.length > 0
            assert rec.exported_fasta_id != "."
        else:
            assert rec.seq == ""
            assert rec.exported_fasta_id == "."

    # Step D >1 kb clustering merged HapStepB#1 and HapCluster#1 into a single event
    clustered = [ev for ev in out_t4.island_sv_events if ev.n_carrier >= 2]
    assert len(clustered) == 1
    assert clustered[0].genotypes["HapStepB#1"] == "1"
    assert clustered[0].genotypes["HapCluster#1"] == "1"
    assert clustered[0].genotypes["HapGapped#1"] == "."
    assert clustered[0].genotypes["HapPartial#1"] == "."
    assert clustered[0].genotypes["HapAmbig#1"] == "."
