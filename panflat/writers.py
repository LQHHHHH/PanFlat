"""Writers for all 11 PanFlat v1 output files and multi-chromosome `panflat merge` (§5, §6, F26)."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Mapping, Sequence, TextIO

import numpy as np

from panflat.gff_liftover import liftover_gff3
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    BuildConfig,
    ChromosomeBuildResult,
    InputContractError,
    LazyStepOffsetSequence,
    StepOffsetRow,
)
from panflat.progress import log_progress
from panflat.verify_qc import (
    INSERTS_HEADER,
    OFFSETS_HEADER,
    SV_HEADER,
    UNRESOLVED_HEADER,
    _parse_fasta_allow_empty,
    _parse_tsv_file,
    run_qc,
)


@contextlib.contextmanager
def _atomic_open_text(final_path: Path) -> Iterator[TextIO]:
    """Write text to a PID/thread-scoped sibling temp file and atomically replace `final_path`."""
    tmp_path = final_path.parent / (
        f".{final_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            yield fh
        os.replace(tmp_path, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _atomic_write_text(final_path: Path, content: str) -> None:
    with _atomic_open_text(final_path) as fh:
        fh.write(content)


def _build_chain_text(
    chrom: str,
    step_offsets: Sequence[StepOffsetRow],
    pan_len: int,
    chain_id: int = 1,
) -> str:
    """Format `{p}.ref_to_pan.chain` with header, blocks, blank line, and `# block` / `# step` comments."""
    if isinstance(step_offsets, LazyStepOffsetSequence):
        n_steps = len(step_offsets)
        ref_len = int(step_offsets.ref_ends[-1]) if n_steps > 0 else 0
        q_end = int(step_offsets.pan_ends[-1]) if n_steps > 0 else 0
        lines: list[str] = [
            f"chain {ref_len} {chrom} {ref_len} + 0 {ref_len} "
            f"{chrom} {pan_len} + 0 {q_end} {chain_id}"
        ]
        block_meta: list[tuple[int, int, int]] = []
        if n_steps > 0:
            dq_steps = (
                np.flatnonzero(step_offsets.slot_ins_lens[1:] > 0) + 1
            ).tolist()
            cur_t_start = 0
            for s_idx in dq_steps:
                t_end = int(step_offsets.ref_ends[s_idx - 1])
                next_dq = int(step_offsets.slot_ins_lens[s_idx])
                cur_size = t_end - cur_t_start
                lines.append(f"{cur_size}\t0\t{next_dq}")
                block_meta.append((cur_t_start, t_end, next_dq))
                cur_t_start = t_end
            last_size = ref_len - cur_t_start
            lines.append(f"{last_size}")
            block_meta.append((cur_t_start, ref_len, 0))
        lines.append("")
        for b_ts, b_te, b_dq in block_meta:
            lines.append(f"# block\t{b_ts}\t{b_te}\t{b_dq}")
        return "\n".join(lines) + "\n"

    ref_len = step_offsets[-1].ref_end if step_offsets else 0
    q_end = step_offsets[-1].pan_end if step_offsets else 0
    lines = [
        f"chain {ref_len} {chrom} {ref_len} + 0 {ref_len} "
        f"{chrom} {pan_len} + 0 {q_end} {chain_id}"
    ]

    block_meta = []
    cur_t_start = 0
    cur_size = 0

    for i, row in enumerate(step_offsets):
        cur_size += row.ref_end - row.ref_start
        if i + 1 < len(step_offsets):
            next_dq = step_offsets[i + 1].slot_ins_len
            if next_dq > 0:
                lines.append(f"{cur_size}\t0\t{next_dq}")
                block_meta.append((cur_t_start, row.ref_end, next_dq))
                cur_t_start = row.ref_end
                cur_size = 0
        else:
            lines.append(f"{cur_size}")
            block_meta.append((cur_t_start, row.ref_end, 0))

    lines.append("")
    for b_ts, b_te, b_dq in block_meta:
        lines.append(f"# block\t{b_ts}\t{b_te}\t{b_dq}")
    for row in step_offsets:
        lines.append(
            f"# step\t{row.ref_step}\t{row.ref_start}\t{row.ref_end}\t"
            f"{row.pan_start}\t{row.pan_end}\t{row.slot_ins_len}"
        )

    return "\n".join(lines) + "\n"


def write_build_outputs(
    result: ChromosomeBuildResult,
    config: BuildConfig,
) -> dict[str, str]:
    """
    Write all PanFlat v1 output files for a single chromosome build (§5, F26).

    Returns a dictionary mapping file key to written file path.
    """
    out_prefix = Path(config.out_prefix)
    if out_prefix.parent and str(out_prefix.parent) != "":
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
    p = str(out_prefix)

    # Compute lifted GFF3 lines if --ref-gff was provided and not yet populated
    if config.ref_gff is not None and not result.lifted_gff_lines:
        t_gff = time.monotonic()
        lifted_gff_lines: tuple[str, ...] = tuple(
            liftover_gff3(
                gff_path=config.ref_gff,
                ref_chrom=result.chrom,
                step_offsets=result.step_offsets,
                insertion_events=result.sv_events,
                splice_window=config.splice_window,
            )
        )
        log_progress(
            f"  GFF3 liftover complete ({time.monotonic() - t_gff:.2f}s): "
            f"{len(lifted_gff_lines):,} lines"
        )
    else:
        lifted_gff_lines = result.lifted_gff_lines

    # Prepare summary metrics (including centromere warnings and --qc metrics)
    summary_metrics: dict[str, Any] = dict(result.summary_metrics)
    summary_metrics.setdefault("backbone_prefix", config.backbone_prefix)
    summary_metrics.setdefault("spacer_n", config.spacer_n)
    summary_metrics.setdefault("min_sv_len", config.min_sv_len)
    summary_metrics.setdefault("max_jump", config.max_jump)
    summary_metrics.setdefault("max_copy", config.max_copy)
    summary_metrics.setdefault("cluster_id", config.cluster_id)
    summary_metrics.setdefault("cluster_win_id", config.cluster_win_id)
    summary_metrics.setdefault("sec_backbone", config.sec_backbone)
    summary_metrics.setdefault("repeat_guard", config.repeat_guard)
    summary_metrics.setdefault(
        "centromere_mode", config.effective_centromere_mode
    )

    for log_line in result.log_lines:
        if "CEN_ANCHOR_NOT_FOUND" in log_line:
            summary_metrics["cen_anchor_not_found"] = "CEN_ANCHOR_NOT_FOUND"
        if "expanded_to_whole_chromosome" in log_line:
            summary_metrics["expanded_to_whole_chromosome"] = (
                "expanded_to_whole_chromosome"
            )

    if config.qc and "qc_enabled" not in summary_metrics:
        t_qc = time.monotonic()
        log_progress("  Running --qc alignment redundancy analysis...")
        qc_metrics = run_qc(result, config)
        summary_metrics.update(qc_metrics)
        log_progress(
            f"  Alignment redundancy --qc complete ({time.monotonic() - t_qc:.2f}s)"
        )

    written: dict[str, str] = {}

    # 1. {p}.pan.fa
    pan_fa_path = Path(f"{p}.pan.fa")
    _atomic_write_text(pan_fa_path, f">{result.chrom}\n{result.pan_seq.upper()}\n")
    written["pan_fa"] = str(pan_fa_path)

    # 2. {p}.offsets.tsv
    offsets_path = Path(f"{p}.offsets.tsv")
    if isinstance(result.step_offsets, LazyStepOffsetSequence):
        lo = result.step_offsets
        n_rows = len(lo)
        chrom_s = lo.chrom
        nids_l = lo.ref_nids.tolist()
        rs_l = lo.ref_starts.tolist()
        re_l = lo.ref_ends.tolist()
        ps_l = lo.pan_starts.tolist()
        pe_l = lo.pan_ends.tolist()
        si_l = lo.slot_ins_lens.tolist()
        with _atomic_open_text(offsets_path) as fh:
            fh.write("\t".join(OFFSETS_HEADER) + "\n")
            chunk_size = 100_000
            for c_start in range(0, n_rows, chunk_size):
                c_end = min(n_rows, c_start + chunk_size)
                fh.write(
                    "\n".join(
                        f"{chrom_s}\t{i}\t{nids_l[i]}\t{rs_l[i]}\t{re_l[i]}\t{ps_l[i]}\t{pe_l[i]}\t{si_l[i]}"
                        for i in range(c_start, c_end)
                    )
                    + "\n"
                )
    else:
        with _atomic_open_text(offsets_path) as fh:
            fh.write("\t".join(OFFSETS_HEADER) + "\n")
            for row in result.step_offsets:
                fh.write(
                    f"{row.chrom}\t{row.ref_step}\t{row.node_id}\t"
                    f"{row.ref_start}\t{row.ref_end}\t{row.pan_start}\t"
                    f"{row.pan_end}\t{row.slot_ins_len}\n"
                )
    written["offsets_tsv"] = str(offsets_path)

    # 3. {p}.ref_to_pan.chain
    chain_path = Path(f"{p}.ref_to_pan.chain")
    chain_text = _build_chain_text(
        chrom=result.chrom,
        step_offsets=result.step_offsets,
        pan_len=len(result.pan_seq),
        chain_id=1,
    )
    _atomic_write_text(chain_path, chain_text)
    written["chain"] = str(chain_path)

    # 4. {p}.sv.tsv
    sv_path = Path(f"{p}.sv.tsv")
    with _atomic_open_text(sv_path) as fh:
        fh.write("\t".join(SV_HEADER) + "\n")
        for ev in result.sv_events:
            fh.write(
                f"{ev.chrom}\t{ev.pan_start}\t{ev.pan_end}\t{ev.event_id}\t"
                f"{ev.event_type}\t{ev.ref_start}\t{ev.ref_end}\t{ev.anchors}\t"
                f"{ev.alleles_str}\t{ev.n_carrier}\t{ev.n_called}\t{ev.region_id}\n"
            )
    written["sv_tsv"] = str(sv_path)

    # 5. {p}.genotypes.tsv
    genotypes_path = Path(f"{p}.genotypes.tsv")
    gt_header = (
        "event_id\t" + "\t".join(result.all_haplotypes)
        if result.all_haplotypes
        else "event_id"
    )
    with _atomic_open_text(genotypes_path) as fh:
        fh.write(gt_header + "\n")
        for ev in result.sv_events:
            vals = [ev.genotypes.get(h, ".") for h in result.all_haplotypes]
            fh.write("\t".join([ev.event_id, *vals]) + "\n")
    written["genotypes_tsv"] = str(genotypes_path)

    # 6. {p}.inserts.tsv
    inserts_path = Path(f"{p}.inserts.tsv")
    with _atomic_open_text(inserts_path) as fh:
        fh.write("\t".join(INSERTS_HEADER) + "\n")
        for r in result.inserts_rows:
            fh.write(
                f"{r['chrom']}\t{r['pan_start']}\t{r['pan_end']}\t{r['event_id']}\t"
                f"{r['allele']}\t{r['piece']}\t{r['src_hap']}\t{r['src_path']}\t"
                f"{r['src_intervals']}\t{r['src_strand']}\t{r['members']}\t{r['join_before']}\n"
            )
    written["inserts_tsv"] = str(inserts_path)

    # 7. {p}.unresolved.tsv
    unresolved_path = Path(f"{p}.unresolved.tsv")
    with _atomic_open_text(unresolved_path) as fh:
        fh.write("\t".join(UNRESOLVED_HEADER) + "\n")
        for u in result.unresolved_records:
            fh.write(
                f"{u.record_id}\t{u.region_id}\t{u.event_id}\t{u.src_path}\t"
                f"{u.src_start}\t{u.src_end}\t{u.src_strand}\t{u.length}\t"
                f"{u.reason}\t{u.exported_fasta_id}\n"
            )
    written["unresolved_tsv"] = str(unresolved_path)

    # 8. {p}.unresolved.fa (always created; deduplicated by exported_fasta_id)
    unresolved_fa_path = Path(f"{p}.unresolved.fa")
    seen_fasta_ids: set[str] = set()
    with _atomic_open_text(unresolved_fa_path) as fh:
        for u in result.unresolved_records:
            if (
                u.export_to_fasta
                and u.exported_fasta_id != "."
                and u.exported_fasta_id not in seen_fasta_ids
            ):
                seen_fasta_ids.add(u.exported_fasta_id)
                fh.write(
                    f">{u.exported_fasta_id} reason={u.reason} "
                    f"src={u.src_path}:{u.src_start}-{u.src_end} strand={u.src_strand} "
                    f"region_id={u.region_id} event_id={u.event_id} len={u.length}\n"
                    f"{u.seq.upper()}\n"
                )
    written["unresolved_fa"] = str(unresolved_fa_path)

    # 9. {p}.centromere.tsv (written when centromere mode != 'off' or centromere_rows non-empty)
    if config.effective_centromere_mode != "off" or result.centromere_rows:
        centromere_path = Path(f"{p}.centromere.tsv")
        with _atomic_open_text(centromere_path) as fh:
            fh.write("\t".join(CENTROMERE_TSV_COLUMNS) + "\n")
            for crow in result.centromere_rows:
                fh.write("\t".join(crow.to_tsv_fields()) + "\n")
        written["centromere_tsv"] = str(centromere_path)

    # 10. {p}.ref.lifted.gff3 (written when --ref-gff is given or lifted_gff_lines non-empty)
    if config.ref_gff is not None or lifted_gff_lines:
        lifted_gff_path = Path(f"{p}.ref.lifted.gff3")
        _atomic_write_text(lifted_gff_path, "\n".join(lifted_gff_lines) + "\n")
        written["lifted_gff"] = str(lifted_gff_path)

    # 11. {p}.summary.tsv and {p}.log
    summary_path = Path(f"{p}.summary.tsv")
    sum_lines = ["metric\tvalue"]
    for k, v in summary_metrics.items():
        sum_lines.append(f"{k}\t{v}")
    _atomic_write_text(summary_path, "\n".join(sum_lines) + "\n")
    written["summary_tsv"] = str(summary_path)

    log_path = Path(f"{p}.log")
    with _atomic_open_text(log_path) as fh:
        for log_l in result.log_lines:
            fh.write(log_l + "\n")
        fh.write(
            f"INFO\tBUILD_COMPLETE\tchrom={result.chrom}\tgfa_sha256={result.gfa_sha256}\t"
            f"pan_len={len(result.pan_seq)}\n"
        )
        for j_entry in result.json_log_entries:
            fh.write(f"CANDIDATE_JSON\t{json.dumps(j_entry, sort_keys=True)}\n")
    written["log"] = str(log_path)

    return written


def merge_chromosomes(
    prefix_list_path: str | Path | Sequence[str | Path],
    out_prefix: str | Path,
) -> dict[str, str]:
    """
    Merge per-chromosome PanFlat output sets into a multi-chromosome output bundle (§6, F26).

    `prefix_list_path` may be a path to a newline-delimited prefix list file or a sequence of prefixes.
    """
    if isinstance(prefix_list_path, (str, Path)):
        list_file = Path(prefix_list_path)
        if not list_file.is_file():
            raise InputContractError(
                f"Prefix list file not found: '{prefix_list_path}'"
            )
        raw_text = list_file.read_text(encoding="utf-8")
        prefixes = [
            Path(line.strip())
            for line in raw_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        prefixes = [Path(p) for p in prefix_list_path if str(p).strip()]

    if not prefixes:
        raise InputContractError("Prefix list is empty")

    # Validate required files for every input prefix
    required_exts = (
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
    for pref in prefixes:
        for ext in required_exts:
            req_file = Path(f"{pref}{ext}")
            if not req_file.is_file():
                raise InputContractError(
                    f"Missing required per-chromosome file '{req_file}' in merge"
                )

    out_p = Path(out_prefix)
    if out_p.parent and str(out_p.parent) != "":
        out_p.parent.mkdir(parents=True, exist_ok=True)
    p_str = str(out_p)
    written: dict[str, str] = {}

    # 1. Stream-merge {out}.pan.fa without buffering all chromosomes in memory
    all_chroms: list[str] = []
    seen_chroms: set[str] = set()
    chroms_for_prefix: list[list[str]] = []
    pan_fa_out = Path(f"{p_str}.pan.fa")
    with _atomic_open_text(pan_fa_out) as out_fa_fh:
        for pref in prefixes:
            fa_path = Path(f"{pref}.pan.fa")
            chroms_here: list[str] = []
            cur_name: str | None = None
            cur_has_seq = False
            with fa_path.open("r", encoding="utf-8") as in_fa_fh:
                for line_no, raw_line in enumerate(in_fa_fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if cur_name is not None:
                            if not cur_has_seq:
                                raise InputContractError(
                                    f"Empty sequence for FASTA record '{cur_name}' in '{fa_path}'"
                                )
                            out_fa_fh.write("\n")
                        hdr_body = line[1:].strip()
                        if not hdr_body:
                            raise InputContractError(
                                f"Empty FASTA header at line {line_no} in '{fa_path}'"
                            )
                        seq_name = hdr_body.split()[0]
                        if seq_name in seen_chroms:
                            raise InputContractError(
                                f"Duplicate chromosome '{seq_name}' across prefixes in merge"
                            )
                        seen_chroms.add(seq_name)
                        all_chroms.append(seq_name)
                        chroms_here.append(seq_name)
                        cur_name = seq_name
                        cur_has_seq = False
                        out_fa_fh.write(f">{seq_name}\n")
                    else:
                        if cur_name is None:
                            raise InputContractError(
                                f"FASTA sequence data before header at line {line_no} in '{fa_path}'"
                            )
                        chunk = (
                            line if (line.isascii() and line.isalpha()) else "".join(line.split())
                        )
                        if not chunk:
                            continue
                        if not (chunk.isascii() and chunk.isalpha()):
                            raise InputContractError(
                                f"Non-alphabetic characters in FASTA record '{cur_name}' in '{fa_path}'"
                            )
                        cur_has_seq = True
                        out_fa_fh.write(chunk.upper())
            if cur_name is None or not chroms_here:
                raise InputContractError(f"No valid FASTA records found in '{fa_path}'")
            if not cur_has_seq:
                raise InputContractError(
                    f"Empty sequence for FASTA record '{cur_name}' in '{fa_path}'"
                )
            out_fa_fh.write("\n")
            chroms_for_prefix.append(chroms_here)
    written["pan_fa"] = str(pan_fa_out)

    def _unique_prefixed_id(
        row_chrom: str, orig_id: str, seen: set[str]
    ) -> str:
        if orig_id not in seen:
            seen.add(orig_id)
            return orig_id
        cand = (
            orig_id
            if orig_id.startswith(f"{row_chrom}_") and orig_id not in seen
            else f"{row_chrom}_{orig_id}"
        )
        idx_u = 2
        while cand in seen:
            cand = f"{row_chrom}_{idx_u}_{orig_id}"
            idx_u += 1
        seen.add(cand)
        return cand

    def _unique_chrom_prefixed_unres_id(
        row_chrom: str, orig_id: str, seen: set[str]
    ) -> str:
        if len(all_chroms) <= 1:
            return _unique_prefixed_id(row_chrom, orig_id, seen)
        base_id = (
            orig_id
            if orig_id.startswith(f"{row_chrom}_")
            else f"{row_chrom}_{orig_id}"
        )
        if base_id not in seen:
            seen.add(base_id)
            return base_id
        idx_u = 2
        cand = f"{row_chrom}_{idx_u}_{orig_id}"
        while cand in seen:
            idx_u += 1
            cand = f"{row_chrom}_{idx_u}_{orig_id}"
        seen.add(cand)
        return cand

    def _split_qualified_src_path(raw_sp: str) -> tuple[str | None, str]:
        if "::" in raw_sp:
            cand_c, rest = raw_sp.split("::", 1)
            if cand_c in seen_chroms:
                return cand_c, rest
        return None, raw_sp

    # 2. Stream-merge {out}.offsets.tsv without materializing millions of dicts
    offsets_out = Path(f"{p_str}.offsets.tsv")
    n_off_cols = len(OFFSETS_HEADER)
    with _atomic_open_text(offsets_out) as out_off_fh:
        out_off_fh.write("\t".join(OFFSETS_HEADER) + "\n")
        for pref in prefixes:
            off_in = Path(f"{pref}.offsets.tsv")
            saw_hdr = False
            col_perm: list[int] | None = None
            with off_in.open("r", encoding="utf-8") as in_off_fh:
                for raw_line in in_off_fh:
                    stripped = raw_line.rstrip("\r\n")
                    if not stripped.strip() or stripped.lstrip().startswith("##"):
                        continue
                    if not saw_hdr:
                        hdr_cols = stripped.lstrip("#").split("\t")
                        if set(hdr_cols) != set(OFFSETS_HEADER) or len(hdr_cols) != n_off_cols:
                            raise InputContractError(
                                f"Invalid offsets.tsv header in '{off_in}': {hdr_cols}"
                            )
                        if hdr_cols != OFFSETS_HEADER:
                            col_perm = [hdr_cols.index(c) for c in OFFSETS_HEADER]
                        saw_hdr = True
                        continue
                    parts = stripped.split("\t")
                    if len(parts) != n_off_cols:
                        raise InputContractError(
                            f"Malformed offsets.tsv row in '{off_in}': expected {n_off_cols} columns"
                        )
                    if col_perm is None:
                        out_off_fh.write(stripped + "\n")
                    else:
                        out_off_fh.write("\t".join(parts[i] for i in col_perm) + "\n")
    written["offsets_tsv"] = str(offsets_out)

    # 3. Stream-merge {out}.ref_to_pan.chain (renumbering chain IDs 1, 2, ...)
    chain_out = Path(f"{p_str}.ref_to_pan.chain")
    chain_counter = 1
    with _atomic_open_text(chain_out) as out_ch_fh:
        for pref in prefixes:
            ch_in = Path(f"{pref}.ref_to_pan.chain")
            with ch_in.open("r", encoding="utf-8") as in_ch_fh:
                for raw_line in in_ch_fh:
                    stripped = raw_line.strip()
                    if stripped.startswith("chain "):
                        parts = stripped.split()
                        if len(parts) >= 13:
                            parts[12] = str(chain_counter)
                        else:
                            parts.append(str(chain_counter))
                        chain_counter += 1
                        out_ch_fh.write(" ".join(parts) + "\n")
                    else:
                        out_ch_fh.write(raw_line.rstrip("\r\n") + "\n")
    written["chain"] = str(chain_out)

    # 4 & 5. Merge {out}.sv.tsv and {out}.genotypes.tsv (attributing each row to its own `chrom`)
    all_haps: list[str] = []
    seen_haps: set[str] = set()
    per_prefix_gt: list[dict[str, dict[str, str]]] = []
    per_prefix_ev_id_map: list[dict[str, str]] = []
    per_prefix_ev_id_by_chrom_map: list[dict[tuple[str, str], str]] = []
    per_prefix_reg_id_map: list[dict[str, str]] = []
    per_prefix_ev_chrom_map: list[dict[str, str]] = []
    per_prefix_reg_chrom_map: list[dict[str, str]] = []
    seen_event_ids: set[str] = set()
    seen_region_ids: set[str] = set()
    merged_sv_count = 0

    sv_out = Path(f"{p_str}.sv.tsv")
    with _atomic_open_text(sv_out) as out_sv_fh:
        out_sv_fh.write("\t".join(SV_HEADER) + "\n")
        for idx_p, pref in enumerate(prefixes):
            default_chrom = chroms_for_prefix[idx_p][0]
            gt_hdr, gt_rows = _parse_tsv_file(Path(f"{pref}.genotypes.tsv"))
            haps_here = gt_hdr[1:] if len(gt_hdr) >= 2 else []
            for h in haps_here:
                if h not in seen_haps:
                    seen_haps.add(h)
                    all_haps.append(h)
            gt_map: dict[str, dict[str, str]] = {}
            for gr in gt_rows:
                eid = gr["event_id"]
                gt_map[eid] = {h: gr.get(h, ".") for h in haps_here}
            per_prefix_gt.append(gt_map)

            _, sv_rows = _parse_tsv_file(Path(f"{pref}.sv.tsv"))
            ev_id_remap: dict[str, str] = {}
            ev_id_by_chrom_remap: dict[tuple[str, str], str] = {}
            reg_id_remap: dict[str, str] = {}
            ev_chrom_map: dict[str, str] = {}
            reg_chrom_map: dict[str, str] = {}
            for r in sv_rows:
                row_chrom = r.get("chrom", "") or default_chrom
                orig_eid = r["event_id"]
                new_eid = _unique_prefixed_id(
                    row_chrom, orig_eid, seen_event_ids
                )
                ev_id_remap[orig_eid] = new_eid
                ev_id_by_chrom_remap[(row_chrom, orig_eid)] = new_eid
                ev_chrom_map[orig_eid] = row_chrom
                ev_chrom_map[new_eid] = row_chrom

                orig_rid = r["region_id"]
                if orig_rid != ".":
                    if orig_rid not in reg_id_remap:
                        new_rid = _unique_prefixed_id(
                            row_chrom, orig_rid, seen_region_ids
                        )
                        reg_id_remap[orig_rid] = new_rid
                    mapped_rid = reg_id_remap[orig_rid]
                    reg_chrom_map[orig_rid] = row_chrom
                    reg_chrom_map[mapped_rid] = row_chrom
                    r["region_id"] = mapped_rid
                r["event_id"] = new_eid
                merged_sv_count += 1
                out_sv_fh.write("\t".join(r[c] for c in SV_HEADER) + "\n")
            per_prefix_ev_id_map.append(ev_id_remap)
            per_prefix_ev_id_by_chrom_map.append(ev_id_by_chrom_remap)
            per_prefix_reg_id_map.append(reg_id_remap)
            per_prefix_ev_chrom_map.append(ev_chrom_map)
            per_prefix_reg_chrom_map.append(reg_chrom_map)
    written["sv_tsv"] = str(sv_out)

    gt_header_str = "event_id\t" + "\t".join(all_haps) if all_haps else "event_id"
    gt_out = Path(f"{p_str}.genotypes.tsv")
    with _atomic_open_text(gt_out) as out_gt_fh:
        out_gt_fh.write(gt_header_str + "\n")
        for idx_p, gt_map in enumerate(per_prefix_gt):
            ev_remap = per_prefix_ev_id_map[idx_p]
            for orig_eid, hap_vals in gt_map.items():
                mapped_eid = ev_remap.get(orig_eid, orig_eid)
                row_vals = [hap_vals.get(h, ".") for h in all_haps]
                out_gt_fh.write("\t".join([mapped_eid, *row_vals]) + "\n")
    written["genotypes_tsv"] = str(gt_out)

    # Pre-load centromere.tsv rows so bypass/extract-islands `region_id -> chrom` is registered before unresolved.tsv
    cen_rows_by_prefix: dict[int, list[dict[str, str]]] = {}
    for idx_p, pref in enumerate(prefixes):
        cen_file = Path(f"{pref}.centromere.tsv")
        if not cen_file.is_file():
            continue
        default_chrom = chroms_for_prefix[idx_p][0]
        reg_remap = per_prefix_reg_id_map[idx_p]
        reg_chrom_map = per_prefix_reg_chrom_map[idx_p]
        _, c_rows = _parse_tsv_file(cen_file)
        for r in c_rows:
            row_chrom = r.get("chrom", "") or default_chrom
            orig_rid = r["region_id"]
            if orig_rid != ".":
                if orig_rid not in reg_remap:
                    reg_remap[orig_rid] = _unique_prefixed_id(
                        row_chrom, orig_rid, seen_region_ids
                    )
                mapped_rid = reg_remap[orig_rid]
                reg_chrom_map[orig_rid] = row_chrom
                reg_chrom_map[mapped_rid] = row_chrom
        cen_rows_by_prefix[idx_p] = c_rows

    # Pre-scan inserts.tsv, centromere.tsv, and unresolved.tsv to attribute chromosomes and detect cross-chrom src_path collisions (§5.5)
    ins_rows_by_prefix: list[list[dict[str, str]]] = []
    unres_rows_by_prefix: list[list[dict[str, str]]] = []
    unres_chroms_by_prefix: list[list[str]] = []
    src_path_chrom_sets: dict[str, set[str]] = {}
    already_qualified_base_paths: set[str] = set()

    def _infer_unresolved_row_chrom(
        u_row: Mapping[str, str],
        p_chroms: Sequence[str],
        p_chroms_set: set[str],
        p_chroms_desc: Sequence[str],
        ev_chrom_map: Mapping[str, str],
        reg_chrom_map: Mapping[str, str],
        prefix_src_path_chroms: Mapping[str, set[str]],
    ) -> str:
        if len(p_chroms) == 1:
            return p_chroms[0]
        raw_sp = u_row.get("src_path", "")
        qual_c, base_sp = _split_qualified_src_path(raw_sp)
        if qual_c is not None:
            return qual_c
        eid = u_row.get("event_id", ".")
        if eid != "." and eid in ev_chrom_map:
            return ev_chrom_map[eid]
        rid = u_row.get("region_id", ".")
        if rid != "." and rid in reg_chrom_map:
            return reg_chrom_map[rid]
        if base_sp:
            sp_no_coords = base_sp.split(":", 1)[0]
            if "#" in sp_no_coords:
                sp_parts = sp_no_coords.split("#")
                sp_seqid = (
                    "#".join(sp_parts[2:])
                    if len(sp_parts) >= 3
                    else sp_parts[1]
                )
                if sp_seqid in p_chroms_set:
                    return sp_seqid
            elif sp_no_coords in p_chroms_set:
                return sp_no_coords
            assoc_chroms = prefix_src_path_chroms.get(base_sp)
            if assoc_chroms and len(assoc_chroms) == 1:
                return next(iter(assoc_chroms))
        for id_field in ("record_id", "exported_fasta_id", "region_id", "event_id"):
            f_val = u_row.get(id_field, ".")
            if f_val and f_val != ".":
                for c_cand in p_chroms_desc:
                    if f_val.startswith(f"{c_cand}_"):
                        return c_cand
        return p_chroms[0]

    for idx_p, pref in enumerate(prefixes):
        p_chroms = chroms_for_prefix[idx_p]
        p_chroms_set = set(p_chroms)
        p_chroms_desc = sorted(p_chroms, key=len, reverse=True)
        _, i_rows = _parse_tsv_file(Path(f"{pref}.inserts.tsv"))
        _, u_rows = _parse_tsv_file(Path(f"{pref}.unresolved.tsv"))
        ins_rows_by_prefix.append(i_rows)
        unres_rows_by_prefix.append(u_rows)

        prefix_src_path_chroms: dict[str, set[str]] = {}
        for r in i_rows:
            row_chrom = r.get("chrom", "") or p_chroms[0]
            raw_sp = r.get("src_path", "")
            if not raw_sp or raw_sp == ".":
                continue
            qual_c, base_sp = _split_qualified_src_path(raw_sp)
            eff_c = qual_c or row_chrom
            if qual_c is not None:
                already_qualified_base_paths.add(base_sp)
            else:
                prefix_src_path_chroms.setdefault(base_sp, set()).add(eff_c)
            src_path_chrom_sets.setdefault(base_sp, set()).add(eff_c)

        for r in cen_rows_by_prefix.get(idx_p, ()):
            row_chrom = r.get("chrom", "") or p_chroms[0]
            raw_sp = r.get("src_path", "")
            if not raw_sp or raw_sp == ".":
                continue
            qual_c, base_sp = _split_qualified_src_path(raw_sp)
            eff_c = qual_c or row_chrom
            if qual_c is not None:
                already_qualified_base_paths.add(base_sp)
            else:
                prefix_src_path_chroms.setdefault(base_sp, set()).add(eff_c)
            src_path_chrom_sets.setdefault(base_sp, set()).add(eff_c)

        u_chroms_here: list[str] = []
        ev_chrom_map = per_prefix_ev_chrom_map[idx_p]
        reg_chrom_map = per_prefix_reg_chrom_map[idx_p]
        for u in u_rows:
            u_chrom = _infer_unresolved_row_chrom(
                u,
                p_chroms,
                p_chroms_set,
                p_chroms_desc,
                ev_chrom_map,
                reg_chrom_map,
                prefix_src_path_chroms,
            )
            u_chroms_here.append(u_chrom)
            raw_sp = u.get("src_path", "")
            if raw_sp and raw_sp != ".":
                qual_c, base_sp = _split_qualified_src_path(raw_sp)
                eff_c = qual_c or u_chrom
                if qual_c is not None:
                    already_qualified_base_paths.add(base_sp)
                src_path_chrom_sets.setdefault(base_sp, set()).add(eff_c)
        unres_chroms_by_prefix.append(u_chroms_here)

    shared_base_src_paths = (
        {bsp for bsp, cset in src_path_chrom_sets.items() if len(cset) > 1}
        | already_qualified_base_paths
    )

    # 6. Merge {out}.inserts.tsv
    ins_out = Path(f"{p_str}.inserts.tsv")
    with _atomic_open_text(ins_out) as out_ins_fh:
        out_ins_fh.write("\t".join(INSERTS_HEADER) + "\n")
        for idx_p, rows in enumerate(ins_rows_by_prefix):
            default_chrom = chroms_for_prefix[idx_p][0]
            ev_remap = per_prefix_ev_id_map[idx_p]
            ev_by_c = per_prefix_ev_id_by_chrom_map[idx_p]
            for r in rows:
                row_chrom = r.get("chrom", "") or default_chrom
                orig_eid = r["event_id"]
                r["event_id"] = ev_by_c.get(
                    (row_chrom, orig_eid), ev_remap.get(orig_eid, orig_eid)
                )
                qual_c, base_sp = _split_qualified_src_path(r["src_path"])
                eff_c = qual_c or row_chrom
                if base_sp in shared_base_src_paths:
                    r["src_path"] = f"{eff_c}::{base_sp}"
                out_ins_fh.write("\t".join(r[c] for c in INSERTS_HEADER) + "\n")
    written["inserts_tsv"] = str(ins_out)

    # 7 & 8. Merge {out}.unresolved.tsv and {out}.unresolved.fa
    seen_rec_ids: set[str] = set()
    global_seen_fa_ids: set[str] = set()
    unres_out = Path(f"{p_str}.unresolved.tsv")
    unres_fa_out = Path(f"{p_str}.unresolved.fa")
    with (
        _atomic_open_text(unres_out) as out_u_fh,
        _atomic_open_text(unres_fa_out) as out_ufa_fh,
    ):
        out_u_fh.write("\t".join(UNRESOLVED_HEADER) + "\n")
        for idx_p, pref in enumerate(prefixes):
            ev_remap = per_prefix_ev_id_map[idx_p]
            ev_by_c = per_prefix_ev_id_by_chrom_map[idx_p]
            reg_remap = per_prefix_reg_id_map[idx_p]
            u_rows = unres_rows_by_prefix[idx_p]
            u_chroms = unres_chroms_by_prefix[idx_p]
            u_fa = _parse_fasta_allow_empty(Path(f"{pref}.unresolved.fa"))
            local_fa_remap: dict[tuple[str, str], str] = {}
            for idx_u, r in enumerate(u_rows):
                row_chrom = u_chroms[idx_u]
                orig_rec = r["record_id"]
                r["record_id"] = _unique_chrom_prefixed_unres_id(
                    row_chrom, orig_rec, seen_rec_ids
                )
                if r["region_id"] != ".":
                    if r["region_id"] not in reg_remap:
                        reg_remap[r["region_id"]] = _unique_prefixed_id(
                            row_chrom, r["region_id"], seen_region_ids
                        )
                    r["region_id"] = reg_remap[r["region_id"]]
                if r["event_id"] != ".":
                    orig_eid = r["event_id"]
                    r["event_id"] = ev_by_c.get(
                        (row_chrom, orig_eid), ev_remap.get(orig_eid, orig_eid)
                    )
                qual_c, base_sp = _split_qualified_src_path(r["src_path"])
                eff_c = qual_c or row_chrom
                if base_sp in shared_base_src_paths:
                    r["src_path"] = f"{eff_c}::{base_sp}"

                orig_exp_id = r["exported_fasta_id"]
                if orig_exp_id != ".":
                    fa_key = (row_chrom, orig_exp_id)
                    if fa_key not in local_fa_remap:
                        mapped_fa_id = _unique_chrom_prefixed_unres_id(
                            row_chrom, orig_exp_id, global_seen_fa_ids
                        )
                        local_fa_remap[fa_key] = mapped_fa_id
                        if orig_exp_id in u_fa:
                            out_ufa_fh.write(
                                f">{mapped_fa_id} reason={r['reason']} "
                                f"src={r['src_path']}:{r['src_start']}-{r['src_end']} strand={r['src_strand']} "
                                f"region_id={r['region_id']} event_id={r['event_id']} len={r['length']}\n"
                                f"{u_fa[orig_exp_id].upper()}\n"
                            )
                    r["exported_fasta_id"] = local_fa_remap[fa_key]
                out_u_fh.write("\t".join(r[c] for c in UNRESOLVED_HEADER) + "\n")
    written["unresolved_tsv"] = str(unres_out)
    written["unresolved_fa"] = str(unres_fa_out)

    # 9. Merge {out}.centromere.tsv (only rows from prefixes that produced centromere.tsv)
    if cen_rows_by_prefix:
        cen_out = Path(f"{p_str}.centromere.tsv")
        with _atomic_open_text(cen_out) as out_cen_fh:
            out_cen_fh.write("\t".join(CENTROMERE_TSV_COLUMNS) + "\n")
            for idx_p in sorted(cen_rows_by_prefix.keys()):
                default_chrom = chroms_for_prefix[idx_p][0]
                ev_remap = per_prefix_ev_id_map[idx_p]
                ev_by_c = per_prefix_ev_id_by_chrom_map[idx_p]
                reg_remap = per_prefix_reg_id_map[idx_p]
                for r in cen_rows_by_prefix[idx_p]:
                    row_chrom = r.get("chrom", "") or default_chrom
                    orig_rid = r["region_id"]
                    if orig_rid != ".":
                        if orig_rid not in reg_remap:
                            reg_remap[orig_rid] = _unique_prefixed_id(
                                row_chrom, orig_rid, seen_region_ids
                            )
                        r["region_id"] = reg_remap[orig_rid]
                    raw_sp = r.get("src_path", "")
                    if raw_sp and raw_sp != ".":
                        qual_c, base_sp = _split_qualified_src_path(raw_sp)
                        eff_c = qual_c or row_chrom
                        if base_sp in shared_base_src_paths:
                            r["src_path"] = f"{eff_c}::{base_sp}"
                    sup_ev = r["supported_events"]
                    if sup_ev and sup_ev != ".":
                        r["supported_events"] = ",".join(
                            ev_by_c.get(
                                (row_chrom, eid.strip()),
                                ev_remap.get(eid.strip(), eid.strip()),
                            )
                            for eid in sup_ev.split(",")
                            if eid.strip()
                        )
                    out_cen_fh.write("\t".join(r[c] for c in CENTROMERE_TSV_COLUMNS) + "\n")
        written["centromere_tsv"] = str(cen_out)

    # 10. Merge {out}.ref.lifted.gff3 (if any input prefix has .ref.lifted.gff3, remapping sv_id tokens)
    gff_entries = [
        (idx_p, Path(f"{pref}.ref.lifted.gff3"))
        for idx_p, pref in enumerate(prefixes)
        if Path(f"{pref}.ref.lifted.gff3").is_file()
    ]
    if gff_entries:
        gff_out = Path(f"{p_str}.ref.lifted.gff3")
        with _atomic_open_text(gff_out) as out_gff_fh:
            out_gff_fh.write("##gff-version 3\n")
            for idx_p, gff_file in gff_entries:
                ev_remap = per_prefix_ev_id_map[idx_p]
                ev_by_c = per_prefix_ev_id_by_chrom_map[idx_p]
                with gff_file.open("r", encoding="utf-8") as in_gff_fh:
                    for raw_line in in_gff_fh:
                        stripped = raw_line.strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        line_clean = raw_line.rstrip("\r\n")
                        if "sv_id=" in line_clean:
                            g_cols = line_clean.split("\t")
                            if len(g_cols) == 9:
                                g_seqid = g_cols[0]
                                new_attrs: list[str] = []
                                for attr_tok in g_cols[8].split(";"):
                                    if attr_tok.startswith("sv_id="):
                                        mapped_ids = [
                                            ev_by_c.get(
                                                (g_seqid, eid.strip()),
                                                ev_remap.get(
                                                    eid.strip(), eid.strip()
                                                ),
                                            )
                                            for eid in attr_tok[6:].split(",")
                                            if eid.strip()
                                        ]
                                        new_attrs.append(
                                            f"sv_id={','.join(mapped_ids)}"
                                        )
                                    else:
                                        new_attrs.append(attr_tok)
                                g_cols[8] = ";".join(new_attrs)
                                line_clean = "\t".join(g_cols)
                        out_gff_fh.write(line_clean + "\n")
        written["lifted_gff"] = str(gff_out)

    # 11. Merge {out}.summary.tsv and {out}.log (per-chromosome deduplicated keys & --qc aggregation)
    chrom_kv_by_chrom: dict[str, dict[str, str]] = {c: {} for c in all_chroms}
    skip_agg_keys = {
        "chrom_count",
        "total_ref_len",
        "total_pan_len",
        "total_sv_count",
    }

    log_out = Path(f"{p_str}.log")
    with _atomic_open_text(log_out) as out_log_fh:
        for idx_p, pref in enumerate(prefixes):
            p_chroms = chroms_for_prefix[idx_p]
            p_chroms_set = set(p_chroms)
            raw_kv_rows: list[tuple[str, str]] = []
            s_text = Path(f"{pref}.summary.tsv").read_text(encoding="utf-8")
            for raw_line in s_text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or line == "metric\tvalue":
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    raw_kv_rows.append((parts[0], parts[1]))

            is_already_merged = len(p_chroms) > 1 or any(
                k == "chrom_count"
                or (":" in k and k.split(":", 1)[0] in p_chroms_set)
                for k, _ in raw_kv_rows
            )

            top_level_fallbacks: dict[str, str] = {}
            for k, v in raw_kv_rows:
                if k in skip_agg_keys:
                    continue
                if ":" in k and k.split(":", 1)[0] in p_chroms_set:
                    c_part, sub_k = k.split(":", 1)
                    chrom_kv_by_chrom[c_part][sub_k] = v
                elif not is_already_merged:
                    chrom_kv_by_chrom[p_chroms[0]][k] = v
                else:
                    top_level_fallbacks[k] = v

            if is_already_merged and top_level_fallbacks:
                for c_part in p_chroms:
                    for fk in ("spacer_n", "backbone_prefix"):
                        if (
                            fk in top_level_fallbacks
                            and fk not in chrom_kv_by_chrom[c_part]
                        ):
                            chrom_kv_by_chrom[c_part][fk] = top_level_fallbacks[
                                fk
                            ]

            with Path(f"{pref}.log").open("r", encoding="utf-8") as in_log_fh:
                for log_line in in_log_fh:
                    out_log_fh.write(log_line)
    written["log"] = str(log_out)

    per_chrom_kv = [chrom_kv_by_chrom[c] for c in all_chroms]
    total_ref_len = sum(
        int(kv["ref_len"])
        for kv in per_chrom_kv
        if kv.get("ref_len", "").isdigit()
    )
    total_pan_len = sum(
        int(kv["pan_len"])
        for kv in per_chrom_kv
        if kv.get("pan_len", "").isdigit()
    )
    spacer_vals = {
        kv["spacer_n"] for kv in per_chrom_kv if "spacer_n" in kv
    }
    bp_vals = {
        kv["backbone_prefix"] for kv in per_chrom_kv if "backbone_prefix" in kv
    }

    if len(spacer_vals) > 1:
        raise InputContractError(
            f"Conflicting spacer_n values across prefixes in merge: {sorted(spacer_vals)}"
        )

    sum_lines = [
        "metric\tvalue",
        f"chrom_count\t{len(all_chroms)}",
        f"total_ref_len\t{total_ref_len}",
        f"total_pan_len\t{total_pan_len}",
        f"total_sv_count\t{merged_sv_count}",
    ]
    if len(spacer_vals) == 1:
        sum_lines.append(f"spacer_n\t{next(iter(spacer_vals))}")
    if len(bp_vals) == 1:
        sum_lines.append(f"backbone_prefix\t{next(iter(bp_vals))}")

    if per_chrom_kv and all(kv.get("qc_enabled") == "true" for kv in per_chrom_kv):
        tot_short_cnt = sum(int(kv.get("qc_short_50_149_count", "0")) for kv in per_chrom_kv)
        tot_short_bp = sum(int(kv.get("qc_short_50_149_bp", "0")) for kv in per_chrom_kv)
        short_len_parts = [
            kv["qc_short_50_149_lengths"]
            for kv in per_chrom_kv
            if kv.get("qc_short_50_149_lengths", "0") not in ("", "0")
        ]
        tot_short_lens = ",".join(short_len_parts) if short_len_parts else "0"
        tot_long_cnt = sum(int(kv.get("qc_long_ge150_count", "0")) for kv in per_chrom_kv)
        tot_long_bp = sum(int(kv.get("qc_long_ge150_bp", "0")) for kv in per_chrom_kv)
        sum_lines.extend(
            [
                "qc_enabled\ttrue",
                f"qc_short_50_149_count\t{tot_short_cnt}",
                f"qc_short_50_149_bp\t{tot_short_bp}",
                f"qc_short_50_149_lengths\t{tot_short_lens}",
                f"qc_long_ge150_count\t{tot_long_cnt}",
                f"qc_long_ge150_bp\t{tot_long_bp}",
                f"qc_ge150_count\t{tot_long_cnt}",
                f"qc_ge150_bp\t{tot_long_bp}",
            ]
        )
        for tau in ("0.90", "0.95", "0.98"):
            red_bp = sum(
                int(kv.get(f"qc_redundancy_{tau}_bp", "0")) for kv in per_chrom_kv
            )
            pct = (100.0 * red_bp / tot_long_bp) if tot_long_bp > 0 else 0.0
            ratio = (red_bp / tot_long_bp) if tot_long_bp > 0 else 0.0
            sum_lines.extend(
                [
                    f"qc_redundancy_{tau}_bp\t{red_bp}",
                    f"qc_redundancy_{tau}_pct\t{pct:.4f}",
                    f"qc_redundancy_{tau}_ratio\t{ratio:.6f}",
                ]
            )
        tot_cen_cnt = sum(
            int(kv.get("qc_cen_island_ge150_count", "0")) for kv in per_chrom_kv
        )
        tot_cen_bp = sum(
            int(kv.get("qc_cen_island_ge150_bp", "0")) for kv in per_chrom_kv
        )
        sum_lines.extend(
            [
                f"qc_cen_island_ge150_count\t{tot_cen_cnt}",
                f"qc_cen_island_ge150_bp\t{tot_cen_bp}",
                f"qc_cen_island_long_ge150_count\t{tot_cen_cnt}",
                f"qc_cen_island_long_ge150_bp\t{tot_cen_bp}",
            ]
        )
        for tau in ("0.90", "0.95", "0.98"):
            c_hit_bp = sum(
                int(kv.get(f"qc_cen_island_ref_hit_{tau}_bp", "0"))
                for kv in per_chrom_kv
            )
            c_pct = (100.0 * c_hit_bp / tot_cen_bp) if tot_cen_bp > 0 else 0.0
            c_ratio = (c_hit_bp / tot_cen_bp) if tot_cen_bp > 0 else 0.0
            sum_lines.extend(
                [
                    f"qc_cen_island_ref_hit_{tau}_bp\t{c_hit_bp}",
                    f"qc_cen_island_ref_hit_{tau}_pct\t{c_pct:.4f}",
                    f"qc_cen_island_ref_hit_{tau}_ratio\t{c_ratio:.6f}",
                ]
            )

    for c_name in all_chroms:
        for sub_k, v in chrom_kv_by_chrom[c_name].items():
            sum_lines.append(f"{c_name}:{sub_k}\t{v}")

    summary_out = Path(f"{p_str}.summary.tsv")
    _atomic_write_text(summary_out, "\n".join(sum_lines) + "\n")
    written["summary_tsv"] = str(summary_out)

    return written


merge_chromosome_outputs = merge_chromosomes
