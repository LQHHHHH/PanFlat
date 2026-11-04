"""Writers for all 11 PanFlat v1 output files and multi-chromosome `panflat merge` (§5, §6, F26)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from panflat.gfa_io import parse_fasta
from panflat.gff_liftover import liftover_gff3
from panflat.models import (
    CENTROMERE_TSV_COLUMNS,
    BuildConfig,
    ChromosomeBuildResult,
    InputContractError,
    StepOffsetRow,
)
from panflat.verify_qc import (
    INSERTS_HEADER,
    OFFSETS_HEADER,
    SV_HEADER,
    UNRESOLVED_HEADER,
    _parse_fasta_allow_empty,
    _parse_tsv_file,
    run_qc,
)


def _build_chain_text(
    chrom: str,
    step_offsets: Sequence[StepOffsetRow],
    pan_len: int,
    chain_id: int = 1,
) -> str:
    """Format `{p}.ref_to_pan.chain` with header, blocks, blank line, and `# block` / `# step` comments."""
    ref_len = step_offsets[-1].ref_end if step_offsets else 0
    lines: list[str] = [
        f"chain {ref_len} {chrom} {ref_len} + 0 {ref_len} "
        f"{chrom} {pan_len} + 0 {pan_len} {chain_id}"
    ]

    block_meta: list[tuple[int, int, int]] = []
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
        lifted_gff_lines: tuple[str, ...] = tuple(
            liftover_gff3(
                gff_path=config.ref_gff,
                ref_chrom=result.chrom,
                step_offsets=result.step_offsets,
                insertion_events=result.sv_events,
                splice_window=config.splice_window,
            )
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
        qc_metrics = run_qc(result, config)
        summary_metrics.update(qc_metrics)

    written: dict[str, str] = {}

    # 1. {p}.pan.fa
    pan_fa_path = Path(f"{p}.pan.fa")
    pan_fa_path.write_text(
        f">{result.chrom}\n{result.pan_seq.upper()}\n", encoding="utf-8"
    )
    written["pan_fa"] = str(pan_fa_path)

    # 2. {p}.offsets.tsv
    offsets_path = Path(f"{p}.offsets.tsv")
    offset_lines = ["\t".join(OFFSETS_HEADER)]
    for row in result.step_offsets:
        offset_lines.append(
            f"{row.chrom}\t{row.ref_step}\t{row.node_id}\t"
            f"{row.ref_start}\t{row.ref_end}\t{row.pan_start}\t"
            f"{row.pan_end}\t{row.slot_ins_len}"
        )
    offsets_path.write_text("\n".join(offset_lines) + "\n", encoding="utf-8")
    written["offsets_tsv"] = str(offsets_path)

    # 3. {p}.ref_to_pan.chain
    chain_path = Path(f"{p}.ref_to_pan.chain")
    chain_text = _build_chain_text(
        chrom=result.chrom,
        step_offsets=result.step_offsets,
        pan_len=len(result.pan_seq),
        chain_id=1,
    )
    chain_path.write_text(chain_text, encoding="utf-8")
    written["chain"] = str(chain_path)

    # 4. {p}.sv.tsv
    sv_path = Path(f"{p}.sv.tsv")
    sv_lines = ["\t".join(SV_HEADER)]
    for ev in result.sv_events:
        sv_lines.append(
            f"{ev.chrom}\t{ev.pan_start}\t{ev.pan_end}\t{ev.event_id}\t"
            f"{ev.event_type}\t{ev.ref_start}\t{ev.ref_end}\t{ev.anchors}\t"
            f"{ev.alleles_str}\t{ev.n_carrier}\t{ev.n_called}\t{ev.region_id}"
        )
    sv_path.write_text("\n".join(sv_lines) + "\n", encoding="utf-8")
    written["sv_tsv"] = str(sv_path)

    # 5. {p}.genotypes.tsv
    genotypes_path = Path(f"{p}.genotypes.tsv")
    gt_header = (
        "event_id\t" + "\t".join(result.all_haplotypes)
        if result.all_haplotypes
        else "event_id"
    )
    gt_lines = [gt_header]
    for ev in result.sv_events:
        vals = [ev.genotypes.get(h, ".") for h in result.all_haplotypes]
        gt_lines.append("\t".join([ev.event_id, *vals]))
    genotypes_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    written["genotypes_tsv"] = str(genotypes_path)

    # 6. {p}.inserts.tsv
    inserts_path = Path(f"{p}.inserts.tsv")
    ins_lines = ["\t".join(INSERTS_HEADER)]
    for r in result.inserts_rows:
        ins_lines.append(
            f"{r['chrom']}\t{r['pan_start']}\t{r['pan_end']}\t{r['event_id']}\t"
            f"{r['allele']}\t{r['piece']}\t{r['src_hap']}\t{r['src_path']}\t"
            f"{r['src_intervals']}\t{r['src_strand']}\t{r['members']}\t{r['join_before']}"
        )
    inserts_path.write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    written["inserts_tsv"] = str(inserts_path)

    # 7. {p}.unresolved.tsv
    unresolved_path = Path(f"{p}.unresolved.tsv")
    unres_lines = ["\t".join(UNRESOLVED_HEADER)]
    for u in result.unresolved_records:
        unres_lines.append(
            f"{u.record_id}\t{u.region_id}\t{u.event_id}\t{u.src_path}\t"
            f"{u.src_start}\t{u.src_end}\t{u.src_strand}\t{u.length}\t"
            f"{u.reason}\t{u.exported_fasta_id}"
        )
    unresolved_path.write_text("\n".join(unres_lines) + "\n", encoding="utf-8")
    written["unresolved_tsv"] = str(unresolved_path)

    # 8. {p}.unresolved.fa (always created; deduplicated by exported_fasta_id)
    unresolved_fa_path = Path(f"{p}.unresolved.fa")
    seen_fasta_ids: set[str] = set()
    fa_chunks: list[str] = []
    for u in result.unresolved_records:
        if (
            u.export_to_fasta
            and u.exported_fasta_id != "."
            and u.exported_fasta_id not in seen_fasta_ids
        ):
            seen_fasta_ids.add(u.exported_fasta_id)
            fa_chunks.append(
                f">{u.exported_fasta_id} region_id={u.region_id} event_id={u.event_id} "
                f"src={u.src_path}:{u.src_start}-{u.src_end}({u.src_strand}) "
                f"len={u.length} reason={u.reason}\n{u.seq.upper()}\n"
            )
    unresolved_fa_path.write_text("".join(fa_chunks), encoding="utf-8")
    written["unresolved_fa"] = str(unresolved_fa_path)

    # 9. {p}.centromere.tsv (written when centromere mode != 'off' or centromere_rows non-empty)
    if config.effective_centromere_mode != "off" or result.centromere_rows:
        centromere_path = Path(f"{p}.centromere.tsv")
        cen_lines = ["\t".join(CENTROMERE_TSV_COLUMNS)]
        for crow in result.centromere_rows:
            cen_lines.append("\t".join(crow.to_tsv_fields()))
        centromere_path.write_text(
            "\n".join(cen_lines) + "\n", encoding="utf-8"
        )
        written["centromere_tsv"] = str(centromere_path)

    # 10. {p}.ref.lifted.gff3 (written when --ref-gff is given or lifted_gff_lines non-empty)
    if config.ref_gff is not None or lifted_gff_lines:
        lifted_gff_path = Path(f"{p}.ref.lifted.gff3")
        lifted_gff_path.write_text(
            "\n".join(lifted_gff_lines) + "\n", encoding="utf-8"
        )
        written["lifted_gff"] = str(lifted_gff_path)

    # 11. {p}.summary.tsv and {p}.log
    summary_path = Path(f"{p}.summary.tsv")
    sum_lines = ["metric\tvalue"]
    for k, v in summary_metrics.items():
        sum_lines.append(f"{k}\t{v}")
    summary_path.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    written["summary_tsv"] = str(summary_path)

    log_path = Path(f"{p}.log")
    log_out: list[str] = list(result.log_lines)
    log_out.append(
        f"INFO\tBUILD_COMPLETE\tchrom={result.chrom}\tgfa_sha256={result.gfa_sha256}\t"
        f"pan_len={len(result.pan_seq)}"
    )
    for j_entry in result.json_log_entries:
        log_out.append(f"CANDIDATE_JSON\t{json.dumps(j_entry, sort_keys=True)}")
    log_path.write_text("\n".join(log_out) + "\n", encoding="utf-8")
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

    # 1. Merge {out}.pan.fa
    seen_chroms: set[str] = set()
    pan_fa_chunks: list[str] = []
    chrom_for_prefix: list[str] = []
    for pref in prefixes:
        fa_recs = parse_fasta(f"{pref}.pan.fa")
        first_chrom = next(iter(fa_recs.keys()))
        chrom_for_prefix.append(first_chrom)
        for ch_name, ch_seq in fa_recs.items():
            if ch_name in seen_chroms:
                raise InputContractError(
                    f"Duplicate chromosome '{ch_name}' across prefixes in merge"
                )
            seen_chroms.add(ch_name)
            pan_fa_chunks.append(f">{ch_name}\n{ch_seq.upper()}\n")

    pan_fa_out = Path(f"{p_str}.pan.fa")
    pan_fa_out.write_text("".join(pan_fa_chunks), encoding="utf-8")
    written["pan_fa"] = str(pan_fa_out)

    # 2. Merge {out}.offsets.tsv
    offsets_lines = ["\t".join(OFFSETS_HEADER)]
    for pref in prefixes:
        _, rows = _parse_tsv_file(Path(f"{pref}.offsets.tsv"))
        for r in rows:
            offsets_lines.append("\t".join(r[c] for c in OFFSETS_HEADER))
    offsets_out = Path(f"{p_str}.offsets.tsv")
    offsets_out.write_text("\n".join(offsets_lines) + "\n", encoding="utf-8")
    written["offsets_tsv"] = str(offsets_out)

    # 3. Merge {out}.ref_to_pan.chain (renumbering chain IDs 1, 2, ...)
    chain_lines_out: list[str] = []
    chain_counter = 1
    for pref in prefixes:
        ch_text = Path(f"{pref}.ref_to_pan.chain").read_text(encoding="utf-8")
        for raw_line in ch_text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("chain "):
                parts = stripped.split()
                if len(parts) >= 13:
                    parts[12] = str(chain_counter)
                else:
                    parts.append(str(chain_counter))
                chain_counter += 1
                chain_lines_out.append(" ".join(parts))
            else:
                chain_lines_out.append(raw_line.rstrip("\r\n"))
    chain_out = Path(f"{p_str}.ref_to_pan.chain")
    chain_out.write_text("\n".join(chain_lines_out) + "\n", encoding="utf-8")
    written["chain"] = str(chain_out)

    # 4 & 5. Merge {out}.sv.tsv and {out}.genotypes.tsv
    sv_lines = ["\t".join(SV_HEADER)]
    all_haps: list[str] = []
    seen_haps: set[str] = set()
    per_prefix_gt: list[tuple[str, dict[str, dict[str, str]]]] = []
    per_prefix_ev_id_map: list[dict[str, str]] = []
    seen_event_ids: set[str] = set()
    seen_region_ids: set[str] = set()

    for idx_p, pref in enumerate(prefixes):
        chrom = chrom_for_prefix[idx_p]
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
        per_prefix_gt.append((chrom, gt_map))

        _, sv_rows = _parse_tsv_file(Path(f"{pref}.sv.tsv"))
        ev_id_remap: dict[str, str] = {}
        reg_id_remap: dict[str, str] = {}
        for r in sv_rows:
            orig_eid = r["event_id"]
            new_eid = (
                f"{chrom}_{orig_eid}"
                if orig_eid in seen_event_ids
                else orig_eid
            )
            seen_event_ids.add(new_eid)
            ev_id_remap[orig_eid] = new_eid

            orig_rid = r["region_id"]
            if orig_rid != ".":
                if orig_rid not in reg_id_remap:
                    new_rid = (
                        f"{chrom}_{orig_rid}"
                        if orig_rid in seen_region_ids
                        else orig_rid
                    )
                    seen_region_ids.add(new_rid)
                    reg_id_remap[orig_rid] = new_rid
                r["region_id"] = reg_id_remap[orig_rid]
            r["event_id"] = new_eid
            sv_lines.append("\t".join(r[c] for c in SV_HEADER))
        per_prefix_ev_id_map.append(ev_id_remap)

    sv_out = Path(f"{p_str}.sv.tsv")
    sv_out.write_text("\n".join(sv_lines) + "\n", encoding="utf-8")
    written["sv_tsv"] = str(sv_out)

    gt_header_str = "event_id\t" + "\t".join(all_haps) if all_haps else "event_id"
    gt_lines_out = [gt_header_str]
    for idx_p, (_, gt_map) in enumerate(per_prefix_gt):
        ev_remap = per_prefix_ev_id_map[idx_p]
        for orig_eid, hap_vals in gt_map.items():
            mapped_eid = ev_remap.get(orig_eid, orig_eid)
            row_vals = [hap_vals.get(h, ".") for h in all_haps]
            gt_lines_out.append("\t".join([mapped_eid, *row_vals]))

    gt_out = Path(f"{p_str}.genotypes.tsv")
    gt_out.write_text("\n".join(gt_lines_out) + "\n", encoding="utf-8")
    written["genotypes_tsv"] = str(gt_out)

    # 6. Merge {out}.inserts.tsv
    ins_lines = ["\t".join(INSERTS_HEADER)]
    for idx_p, pref in enumerate(prefixes):
        ev_remap = per_prefix_ev_id_map[idx_p]
        _, rows = _parse_tsv_file(Path(f"{pref}.inserts.tsv"))
        for r in rows:
            r["event_id"] = ev_remap.get(r["event_id"], r["event_id"])
            ins_lines.append("\t".join(r[c] for c in INSERTS_HEADER))
    ins_out = Path(f"{p_str}.inserts.tsv")
    ins_out.write_text("\n".join(ins_lines) + "\n", encoding="utf-8")
    written["inserts_tsv"] = str(ins_out)

    # 7 & 8. Merge {out}.unresolved.tsv and {out}.unresolved.fa
    unres_lines = ["\t".join(UNRESOLVED_HEADER)]
    seen_rec_ids: set[str] = set()
    seen_fa_ids: set[str] = set()
    unres_fa_chunks: list[str] = []

    for idx_p, pref in enumerate(prefixes):
        chrom = chrom_for_prefix[idx_p]
        ev_remap = per_prefix_ev_id_map[idx_p]
        _, u_rows = _parse_tsv_file(Path(f"{pref}.unresolved.tsv"))
        u_fa = _parse_fasta_allow_empty(Path(f"{pref}.unresolved.fa"))
        for r in u_rows:
            orig_rec = r["record_id"]
            new_rec = (
                f"{chrom}_{orig_rec}"
                if orig_rec in seen_rec_ids
                else orig_rec
            )
            seen_rec_ids.add(new_rec)
            r["record_id"] = new_rec
            if r["event_id"] != ".":
                r["event_id"] = ev_remap.get(r["event_id"], r["event_id"])
            exp_id = r["exported_fasta_id"]
            if exp_id != "." and exp_id not in seen_fa_ids and exp_id in u_fa:
                seen_fa_ids.add(exp_id)
                unres_fa_chunks.append(
                    f">{exp_id} region_id={r['region_id']} event_id={r['event_id']} "
                    f"src={r['src_path']}:{r['src_start']}-{r['src_end']}({r['src_strand']}) "
                    f"len={r['length']} reason={r['reason']}\n{u_fa[exp_id].upper()}\n"
                )
            unres_lines.append("\t".join(r[c] for c in UNRESOLVED_HEADER))

    unres_out = Path(f"{p_str}.unresolved.tsv")
    unres_out.write_text("\n".join(unres_lines) + "\n", encoding="utf-8")
    written["unresolved_tsv"] = str(unres_out)

    unres_fa_out = Path(f"{p_str}.unresolved.fa")
    unres_fa_out.write_text("".join(unres_fa_chunks), encoding="utf-8")
    written["unresolved_fa"] = str(unres_fa_out)

    # 9. Merge {out}.centromere.tsv (only rows from prefixes that produced centromere.tsv)
    cen_data_lines: list[str] = []
    any_cen_file = False
    for idx_p, pref in enumerate(prefixes):
        cen_file = Path(f"{pref}.centromere.tsv")
        if cen_file.is_file():
            any_cen_file = True
            _, c_rows = _parse_tsv_file(cen_file)
            for r in c_rows:
                cen_data_lines.append(
                    "\t".join(r[c] for c in CENTROMERE_TSV_COLUMNS)
                )

    if any_cen_file:
        cen_out = Path(f"{p_str}.centromere.tsv")
        cen_out.write_text(
            "\n".join(["\t".join(CENTROMERE_TSV_COLUMNS), *cen_data_lines])
            + "\n",
            encoding="utf-8",
        )
        written["centromere_tsv"] = str(cen_out)

    # 10. Merge {out}.ref.lifted.gff3 (if any input prefix has .ref.lifted.gff3)
    gff_feature_lines: list[str] = []
    any_gff_file = False
    for pref in prefixes:
        gff_file = Path(f"{pref}.ref.lifted.gff3")
        if gff_file.is_file():
            any_gff_file = True
            for raw_line in gff_file.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                gff_feature_lines.append(raw_line.rstrip("\r\n"))

    if any_gff_file:
        gff_out = Path(f"{p_str}.ref.lifted.gff3")
        gff_out.write_text(
            "\n".join(["##gff-version 3", *gff_feature_lines]) + "\n",
            encoding="utf-8",
        )
        written["lifted_gff"] = str(gff_out)

    # 11. Merge {out}.summary.tsv and {out}.log
    sum_lines = ["metric\tvalue", f"chrom_count\t{len(prefixes)}"]
    total_ref_len = 0
    total_pan_len = 0
    total_sv_count = 0
    log_chunks: list[str] = []

    for idx_p, pref in enumerate(prefixes):
        chrom = chrom_for_prefix[idx_p]
        s_text = Path(f"{pref}.summary.tsv").read_text(encoding="utf-8")
        for raw_line in s_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "metric\tvalue":
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                k, v = parts[0], parts[1]
                if k == "ref_len" and v.isdigit():
                    total_ref_len += int(v)
                elif k == "pan_len" and v.isdigit():
                    total_pan_len += int(v)
                elif k == "sv_total_count" and v.isdigit():
                    total_sv_count += int(v)
                sum_lines.append(f"{chrom}:{k}\t{v}")
        log_chunks.append(Path(f"{pref}.log").read_text(encoding="utf-8"))

    sum_lines.insert(2, f"total_ref_len\t{total_ref_len}")
    sum_lines.insert(3, f"total_pan_len\t{total_pan_len}")
    sum_lines.insert(4, f"total_sv_count\t{total_sv_count}")

    summary_out = Path(f"{p_str}.summary.tsv")
    summary_out.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    written["summary_tsv"] = str(summary_out)

    log_out = Path(f"{p_str}.log")
    log_out.write_text("".join(log_chunks), encoding="utf-8")
    written["log"] = str(log_out)

    return written


merge_chromosome_outputs = merge_chromosomes
