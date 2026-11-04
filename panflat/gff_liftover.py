"""Deterministic Reference GFF3 Liftover for PanFlat v1 (§4, F25, T14, T22, C14)."""

from __future__ import annotations

from bisect import bisect_right
import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from panflat.models import (
    InputContractError,
    StepOffsetRow,
    SVEvent,
)

SPLITTABLE_FEATURE_TYPES: frozenset[str] = frozenset(
    {
        "exon",
        "CDS",
        "five_prime_UTR",
        "three_prime_UTR",
        "UTR",
    }
)

VALID_STRANDS: frozenset[str] = frozenset({"+", "-", ".", "?"})
VALID_PHASES: frozenset[str] = frozenset({"0", "1", "2", "."})


@dataclass
class _ParsedGFFFeature:
    """Internal representation of a validated input GFF3 feature row."""

    row_idx: int
    seqid: str
    source: str
    ftype: str
    start_1based: int
    end_1based: int
    s0: int
    e0: int
    score: str
    strand: str
    phase: str
    attr_items: list[tuple[str, str]]
    attr_dict: dict[str, str]
    feat_id: str | None
    parents: list[str]
    segments: list[tuple[int, int, str]] = field(default_factory=list)
    pan_s0: int = 0
    pan_e0: int = 0
    projection_split: bool = False
    sv_statuses: list[str] = field(default_factory=list)
    splice_site_proximal: bool = False
    sv_ids: list[str] = field(default_factory=list)


def _parse_gff3_attributes(
    attr_str: str,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Parse 9th-column GFF3 attributes while preserving original key order."""
    items: list[tuple[str, str]] = []
    mapping: dict[str, str] = {}
    for raw_token in attr_str.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            k_clean = k.strip()
            v_clean = v.strip()
            items.append((k_clean, v_clean))
            mapping[k_clean] = v_clean
        else:
            items.append((token, ""))
            mapping[token] = ""
    return items, mapping


def _format_gff3_attributes(
    orig_items: Sequence[tuple[str, str]],
    projection_split: bool,
    sv_statuses: Sequence[str],
    splice_site_proximal: bool,
    sv_ids: Sequence[str],
) -> str:
    """Format 9th-column attributes with generated PanFlat SV tags in canonical order."""
    reserved_keys = {
        "projection_split",
        "sv_status",
        "splice_site_proximal_sv",
        "sv_id",
    }
    parts: list[str] = []
    for k, v in orig_items:
        if k in reserved_keys:
            continue
        if v != "":
            parts.append(f"{k}={v}")
        else:
            parts.append(k)

    has_any_sv_tag = False
    if projection_split:
        parts.append("projection_split=True")
        has_any_sv_tag = True

    if sv_statuses:
        parts.append(f"sv_status={','.join(sv_statuses)}")
        has_any_sv_tag = True

    if splice_site_proximal:
        parts.append("splice_site_proximal_sv=True")
        has_any_sv_tag = True

    if has_any_sv_tag and sv_ids:
        parts.append(f"sv_id={','.join(sv_ids)}")

    return ";".join(parts) if parts else "."


def _dedup_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def liftover_gff3(
    gff_path: str | Path,
    ref_chrom: str | Sequence[StepOffsetRow],
    step_offsets: Sequence[StepOffsetRow] | Sequence[SVEvent] | None = None,
    insertion_events: Sequence[SVEvent] | None = None,
    splice_window: int = 8,
) -> list[str]:
    """
    Lift over reference GFF3 annotations onto `{p}.ref.lifted.gff3` coordinates (§4, F25).

    Supports both:
    - `liftover_gff3(gff_path, ref_chrom, step_offsets, insertion_events, splice_window=8)`
    - `liftover_gff3(gff_path, step_offsets, insertion_events, splice_window=8)`
    """
    if not isinstance(ref_chrom, str):
        actual_step_offsets: Sequence[StepOffsetRow] = ref_chrom
        actual_ins_events: Sequence[SVEvent] = (
            step_offsets  # type: ignore[assignment]
            if step_offsets is not None
            else ()
        )
        actual_ref_chrom = (
            actual_step_offsets[0].chrom if actual_step_offsets else ""
        )
    else:
        actual_ref_chrom = ref_chrom
        actual_step_offsets = (
            step_offsets  # type: ignore[assignment]
            if step_offsets is not None
            else ()
        )
        actual_ins_events = (
            insertion_events if insertion_events is not None else ()
        )

    path_obj = Path(gff_path)
    if not path_obj.is_file():
        raise InputContractError(f"Reference GFF3 file not found: '{gff_path}'")

    raw_text = path_obj.read_text(encoding="utf-8")
    return liftover_gff3_lines(
        gff_lines=raw_text.splitlines(),
        ref_chrom=actual_ref_chrom,
        step_offsets=actual_step_offsets,
        insertion_events=actual_ins_events,
        splice_window=splice_window,
    )


def liftover_gff3_lines(
    gff_lines: Sequence[str],
    ref_chrom: str,
    step_offsets: Sequence[StepOffsetRow],
    insertion_events: Sequence[SVEvent],
    splice_window: int = 8,
) -> list[str]:
    """Lift over in-memory GFF3 lines onto pan-genome coordinates (§4)."""
    ref_len: int | None = (
        step_offsets[-1].ref_end if step_offsets else None
    )

    # 1. Parse and validate all GFF3 feature lines
    features: list[_ParsedGFFFeature] = []
    for line_no, raw_line in enumerate(gff_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("##FASTA"):
            break
        if stripped.startswith("#"):
            continue

        cols = raw_line.rstrip("\r\n").split("\t")
        if len(cols) != 9:
            raise InputContractError(
                f"Malformed GFF3 at line {line_no}: expected 9 tab-separated columns, "
                f"got {len(cols)}"
            )

        seqid = cols[0].strip()
        if not seqid:
            raise InputContractError(f"Empty seqid in GFF3 at line {line_no}")
        if ref_chrom and seqid != ref_chrom:
            raise InputContractError(
                f"GFF3 seqid '{seqid}' at line {line_no} does not match reference "
                f"chromosome '{ref_chrom}'"
            )

        source = cols[1].strip()
        ftype = cols[2].strip()
        if not source or not ftype:
            raise InputContractError(
                f"Empty source or feature type in GFF3 at line {line_no}"
            )

        try:
            start_1based = int(cols[3].strip())
            end_1based = int(cols[4].strip())
        except ValueError as exc:
            raise InputContractError(
                f"Non-integer GFF3 coordinates at line {line_no}: "
                f"start='{cols[3]}', end='{cols[4]}'"
            ) from exc

        if start_1based < 1 or start_1based > end_1based:
            raise InputContractError(
                f"Invalid GFF3 coordinate range [{start_1based}, {end_1based}] "
                f"at line {line_no} (must satisfy 1 <= start <= end)"
            )
        if ref_len is not None and end_1based > ref_len:
            raise InputContractError(
                f"GFF3 feature end {end_1based} at line {line_no} exceeds "
                f"reference length {ref_len}"
            )

        score = cols[5].strip()
        strand = cols[6].strip()
        if strand not in VALID_STRANDS:
            raise InputContractError(
                f"Invalid GFF3 strand '{strand}' at line {line_no}"
            )

        phase = cols[7].strip()
        if phase not in VALID_PHASES:
            raise InputContractError(
                f"Invalid GFF3 phase '{phase}' at line {line_no}"
            )

        attr_str = cols[8].strip()
        attr_items, attr_dict = _parse_gff3_attributes(attr_str)
        feat_id = attr_dict.get("ID")
        parents = [
            p.strip()
            for p in attr_dict.get("Parent", "").split(",")
            if p.strip()
        ]

        s0 = start_1based - 1
        e0 = end_1based
        features.append(
            _ParsedGFFFeature(
                row_idx=len(features),
                seqid=seqid,
                source=source,
                ftype=ftype,
                start_1based=start_1based,
                end_1based=end_1based,
                s0=s0,
                e0=e0,
                score=score,
                strand=strand,
                phase=phase,
                attr_items=attr_items,
                attr_dict=attr_dict,
                feat_id=feat_id,
                parents=parents,
            )
        )

    if not features:
        return ["##gff-version 3"]

    # 2. Index insertion slots and event IDs by 0-based reference position r_ins
    seq_ins_events = sorted(
        [
            ev
            for ev in insertion_events
            if ev.event_type in ("INS", "CEN_ISLAND")
        ],
        key=lambda ev: (
            ev.ref_start,
            0 if ev.event_type == "CEN_ISLAND" else 1,
            ev.pan_start,
            ev.pan_end,
            ev.event_id,
        ),
    )

    ins_event_ids_by_pos: dict[int, list[str]] = collections.defaultdict(list)
    for ev in seq_ins_events:
        if ev.event_id and ev.event_id not in ins_event_ids_by_pos[ev.ref_start]:
            ins_event_ids_by_pos[ev.ref_start].append(ev.event_id)

    active_pos_set: set[int] = set()
    for row in step_offsets:
        if row.slot_ins_len > 0:
            active_pos_set.add(row.ref_start)
            if row.ref_start not in ins_event_ids_by_pos:
                ins_event_ids_by_pos[row.ref_start] = []

    for ev in seq_ins_events:
        if ev.pan_end > ev.pan_start or not step_offsets:
            active_pos_set.add(ev.ref_start)

    ins_positions: list[int] = sorted(active_pos_set)

    # 3. Build 0-based reference base -> 0-based pan base coordinate mapper
    ref_starts = [row.ref_start for row in step_offsets]

    # Fallback cumulative shifts if step_offsets is empty
    fallback_shifts: dict[int, int] = collections.defaultdict(int)
    if not step_offsets:
        events_by_pos: dict[int, list[SVEvent]] = collections.defaultdict(list)
        for ev in seq_ins_events:
            events_by_pos[ev.ref_start].append(ev)
        for pos, ev_list in events_by_pos.items():
            min_ps = min(e.pan_start for e in ev_list)
            max_pe = max(e.pan_end for e in ev_list)
            fallback_shifts[pos] = max(0, max_pe - min_ps)

    def pan_base_pos(r: int) -> int:
        if step_offsets:
            idx = bisect_right(ref_starts, r) - 1
            if idx < 0:
                idx = 0
            row = step_offsets[idx]
            return row.pan_start + (r - row.ref_start)
        shift = sum(
            delta for pos, delta in fallback_shifts.items() if pos <= r
        )
        return r + shift

    def map_start(a: int) -> int:
        return pan_base_pos(a)

    def map_end(b: int) -> int:
        return pan_base_pos(b - 1) + 1

    # 4. Split splittable features across strictly internal insertions & recalculate CDS phase
    for feat in features:
        internal_pts = [p for p in ins_positions if feat.s0 < p < feat.e0]
        if feat.ftype in SPLITTABLE_FEATURE_TYPES and internal_pts:
            feat.projection_split = True
            bounds = [feat.s0, *internal_pts, feat.e0]
            ref_sub_intervals = [
                (bounds[i], bounds[i + 1])
                for i in range(len(bounds) - 1)
                if bounds[i + 1] > bounds[i]
            ]
            m = len(ref_sub_intervals)
            seg_phases = [feat.phase] * m
            if feat.ftype == "CDS":
                cur_phase = (
                    int(feat.phase) if feat.phase in ("0", "1", "2") else 0
                )
                tx_order = (
                    range(m - 1, -1, -1)
                    if feat.strand == "-"
                    else range(0, m)
                )
                for idx in tx_order:
                    seg_phases[idx] = str(cur_phase)
                    seg_len = (
                        ref_sub_intervals[idx][1] - ref_sub_intervals[idx][0]
                    )
                    cur_phase = (3 - ((seg_len - cur_phase) % 3)) % 3

            for idx, (a_i, b_i) in enumerate(ref_sub_intervals):
                p_s0 = map_start(a_i)
                p_e0 = map_end(b_i)
                feat.segments.append((p_s0, p_e0, seg_phases[idx]))
        else:
            p_s0 = map_start(feat.s0)
            p_e0 = map_end(feat.e0)
            feat.segments.append((p_s0, p_e0, feat.phase))

        feat.pan_s0 = min(seg[0] for seg in feat.segments)
        feat.pan_e0 = max(seg[1] for seg in feat.segments)

    # 5. Build Parent -> children hierarchy for bottom-up span expansion & SV status annotation
    children_by_parent: dict[str, list[_ParsedGFFFeature]] = (
        collections.defaultdict(list)
    )
    feats_by_id: dict[str, list[_ParsedGFFFeature]] = collections.defaultdict(
        list
    )
    for feat in features:
        if feat.feat_id:
            feats_by_id[feat.feat_id].append(feat)
        for pid in feat.parents:
            children_by_parent[pid].append(feat)

    # Iterative bottom-up relaxation of container coordinates
    changed = True
    max_iters = len(features) + 2
    iter_count = 0
    while changed and iter_count < max_iters:
        changed = False
        iter_count += 1
        for feat in features:
            if not feat.feat_id or feat.feat_id not in children_by_parent:
                continue
            ch_list = children_by_parent[feat.feat_id]
            if not ch_list:
                continue
            new_s0 = min(
                feat.pan_s0, min(c.pan_s0 for c in ch_list)
            )
            new_e0 = max(
                feat.pan_e0, max(c.pan_e0 for c in ch_list)
            )
            if new_s0 != feat.pan_s0 or new_e0 != feat.pan_e0:
                feat.pan_s0 = new_s0
                feat.pan_e0 = new_e0
                if len(feat.segments) == 1:
                    feat.segments[0] = (new_s0, new_e0, feat.segments[0][2])
                changed = True

    def _get_descendants(root_feat: _ParsedGFFFeature) -> list[_ParsedGFFFeature]:
        if not root_feat.feat_id:
            return []
        desc: list[_ParsedGFFFeature] = []
        seen_rows: set[int] = set()
        queue = list(children_by_parent.get(root_feat.feat_id, ()))
        while queue:
            cur = queue.pop(0)
            if cur.row_idx in seen_rows:
                continue
            seen_rows.add(cur.row_idx)
            desc.append(cur)
            if cur.feat_id and cur.feat_id in children_by_parent:
                queue.extend(children_by_parent[cur.feat_id])
        return desc

    # 6. Annotate SV statuses, splice_site_proximal_sv, and sv_id on each feature
    for feat in features:
        internal_pts = [p for p in ins_positions if feat.s0 < p < feat.e0]
        if feat.ftype == "CDS":
            if internal_pts:
                feat.sv_statuses = ["cds_split_by_insertion"]
                ev_ids: list[str] = []
                for p in internal_pts:
                    ev_ids.extend(ins_event_ids_by_pos.get(p, ()))
                feat.sv_ids = _dedup_preserve_order(ev_ids)

        elif feat.ftype in ("five_prime_UTR", "three_prime_UTR", "UTR"):
            if internal_pts:
                feat.sv_statuses = ["utr_insertion"]
                ev_ids = []
                for p in internal_pts:
                    ev_ids.extend(ins_event_ids_by_pos.get(p, ()))
                feat.sv_ids = _dedup_preserve_order(ev_ids)

        elif feat.ftype == "exon":
            sibling_cds: list[_ParsedGFFFeature] = []
            for pid in feat.parents:
                for sib in children_by_parent.get(pid, ()):
                    if sib.ftype == "CDS":
                        sibling_cds.append(sib)
            if not sibling_cds and not feat.parents:
                sibling_cds = [f for f in features if f.ftype == "CDS"]

            status_set: set[str] = set()
            ev_ids = []
            for p in internal_pts:
                in_cds = any(c.s0 < p < c.e0 for c in sibling_cds)
                if in_cds:
                    status_set.add("cds_split_by_insertion")
                else:
                    status_set.add("utr_insertion")
                ev_ids.extend(ins_event_ids_by_pos.get(p, ()))

            feat.sv_statuses = [
                s
                for s in ("utr_insertion", "cds_split_by_insertion")
                if s in status_set
            ]

            proximal_pts = [
                p
                for p in ins_positions
                if min(abs(p - feat.s0), abs(p - feat.e0)) <= splice_window
            ]
            if proximal_pts:
                feat.splice_site_proximal = True
                for p in proximal_pts:
                    ev_ids.extend(ins_event_ids_by_pos.get(p, ()))

            feat.sv_ids = _dedup_preserve_order(ev_ids)

    # Annotate container features (mRNA, transcript, gene, etc.)
    for feat in features:
        if feat.ftype in SPLITTABLE_FEATURE_TYPES:
            continue
        desc = _get_descendants(feat)
        desc_exons = [d for d in desc if d.ftype == "exon"]
        desc_cds = [d for d in desc if d.ftype == "CDS"]
        desc_utrs = [
            d
            for d in desc
            if d.ftype in ("five_prime_UTR", "three_prime_UTR", "UTR")
        ]

        status_set = set()
        ev_ids = []

        internal_pts = [p for p in ins_positions if feat.s0 < p < feat.e0]
        for p in internal_pts:
            if any(c.s0 < p < c.e0 for c in desc_cds):
                status_set.add("cds_split_by_insertion")
                ev_ids.extend(ins_event_ids_by_pos.get(p, ()))
            elif any(u.s0 < p < u.e0 for u in desc_utrs) or any(
                e.s0 < p < e.e0 for e in desc_exons
            ):
                status_set.add("utr_insertion")
                ev_ids.extend(ins_event_ids_by_pos.get(p, ()))
            elif not any(e.s0 <= p <= e.e0 for e in desc_exons):
                status_set.add("intronic_insertion")
                ev_ids.extend(ins_event_ids_by_pos.get(p, ()))

        for d in desc:
            for st in d.sv_statuses:
                status_set.add(st)
            if d.splice_site_proximal:
                feat.splice_site_proximal = True
            ev_ids.extend(d.sv_ids)

        feat.sv_statuses = [
            s
            for s in (
                "intronic_insertion",
                "utr_insertion",
                "cds_split_by_insertion",
            )
            if s in status_set
        ]
        feat.sv_ids = _dedup_preserve_order(ev_ids)

    # 7. Emit formatted 1-based closed GFF3 lines
    out_lines: list[str] = ["##gff-version 3"]
    for feat in features:
        attr_formatted = _format_gff3_attributes(
            orig_items=feat.attr_items,
            projection_split=feat.projection_split,
            sv_statuses=feat.sv_statuses,
            splice_site_proximal=feat.splice_site_proximal,
            sv_ids=feat.sv_ids,
        )
        for seg_s0, seg_e0, seg_phase in feat.segments:
            start_1 = seg_s0 + 1
            end_1 = seg_e0
            out_lines.append(
                f"{feat.seqid}\t{feat.source}\t{feat.ftype}\t"
                f"{start_1}\t{end_1}\t{feat.score}\t{feat.strand}\t"
                f"{seg_phase}\t{attr_formatted}"
            )

    return out_lines
