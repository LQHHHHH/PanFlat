"""GFA v1 (S/L/P/W), FASTA, and BED parsing and §2.1 / §3.8.1 input contract validation."""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence

from panflat.models import (
    BuildConfig,
    CLIArgumentError,
    GFAGraph,
    InputContractError,
    PathFragment,
    Step,
    StepProvenance,
)

_REVCOMP_TABLE = str.maketrans(
    "ACGTNacgtnRYSWKMBDHVryswkmbdhv",
    "TGCANtgcanYRSWMKVHDByrswmkvhdb",
)

_P_STEP_RE = re.compile(r"^([^\s,+\-]+)([+\-])$")
_W_WALK_RE = re.compile(r"^([><][^><\s]+)+$")
_W_STEP_RE = re.compile(r"([><])([^><\s]+)")


def _flip_orient(orient: str) -> Literal["+", "-"]:
    if orient == "+":
        return "-"
    if orient == "-":
        return "+"
    raise InputContractError(f"Invalid step orientation '{orient}'")


def revcomp(seq: str) -> str:
    """Return the reverse-complement of a DNA sequence (preserving case and N)."""
    return seq.translate(_REVCOMP_TABLE)[::-1]


def spell_steps(steps: Sequence[Step], nodes: Mapping[str, str]) -> str:
    """Spell the concatenated nucleotide sequence of a sequence of Steps (full or partial)."""
    parts: list[str] = []
    for step in steps:
        if step.node_id not in nodes:
            raise InputContractError(f"Unknown node ID '{step.node_id}' in step walk")
        raw = nodes[step.node_id]
        oriented = raw if step.orient == "+" else revcomp(raw)
        if step.slice_start != 0 or step.slice_end is not None:
            end = step.slice_end if step.slice_end is not None else len(oriented)
            if step.slice_start < 0 or end > len(oriented) or step.slice_start >= end:
                raise InputContractError(
                    f"Invalid slice [{step.slice_start}, {end}) on node '{step.node_id}' "
                    f"of length {len(oriented)}"
                )
            oriented = oriented[step.slice_start:end]
        parts.append(oriented)
    return "".join(parts)


def parse_fasta(path: str) -> dict[str, str]:
    """Parse a FASTA file into `{seq_name: uppercase_sequence}` with strict validation."""
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise InputContractError(f"FASTA file not found: '{path}'")

    raw_text = fasta_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise InputContractError(f"FASTA file is empty: '{path}'")

    records: dict[str, str] = {}
    current_name: str | None = None
    current_chunks: list[str] = []

    def _flush_current() -> None:
        if current_name is None:
            return
        seq = "".join(current_chunks).upper()
        if not seq:
            raise InputContractError(
                f"Empty sequence for FASTA record '{current_name}' in '{path}'"
            )
        if not seq.isalpha():
            raise InputContractError(
                f"Non-alphabetic characters in FASTA record '{current_name}' in '{path}'"
            )
        records[current_name] = seq

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush_current()
            header_body = line[1:].strip()
            if not header_body:
                raise InputContractError(
                    f"Empty FASTA header at line {line_no} in '{path}'"
                )
            seq_name = header_body.split()[0]
            if seq_name in records:
                raise InputContractError(
                    f"Duplicate FASTA sequence name '{seq_name}' in '{path}'"
                )
            current_name = seq_name
            current_chunks = []
        else:
            if current_name is None:
                raise InputContractError(
                    f"FASTA sequence data before header at line {line_no} in '{path}'"
                )
            chunk = "".join(line.split())
            current_chunks.append(chunk)

    _flush_current()
    if not records:
        raise InputContractError(f"No valid FASTA records found in '{path}'")
    return records


def parse_bed(
    path: str,
    ref_chrom: str,
    ref_len: int,
) -> list[tuple[str, int, int, str]]:
    """
    Parse and validate a centromere BED file (§3.8.1).

    Returns a list of `(chrom, start, end, bed_id)` tuples using 0-based half-open
    coordinates `[start, end)` within `[0, ref_len]`.
    """
    bed_path = Path(path)
    if not bed_path.is_file():
        raise InputContractError(f"Centromere BED file not found: '{path}'")

    raw_text = bed_path.read_text(encoding="utf-8")
    intervals: list[tuple[str, int, int, str]] = []

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("track ") or stripped.startswith("browser "):
            continue

        cols = raw_line.rstrip("\r\n").split("\t")
        if len(cols) < 3:
            cols = stripped.split()
        if len(cols) < 3:
            raise InputContractError(
                f"BED line {line_no} in '{path}' has fewer than 3 columns: '{raw_line}'"
            )

        chrom = cols[0].strip()
        if chrom != ref_chrom:
            raise InputContractError(
                f"BED chromosome '{chrom}' at line {line_no} does not match "
                f"reference chromosome '{ref_chrom}'"
            )
        try:
            start = int(cols[1].strip())
            end = int(cols[2].strip())
        except ValueError as exc:
            raise InputContractError(
                f"Invalid integer coordinates in BED line {line_no} in '{path}'"
            ) from exc

        if start < 0 or start >= end or end > ref_len:
            raise InputContractError(
                f"Invalid BED interval [{start}, {end}) at line {line_no} in '{path}' "
                f"(must satisfy 0 <= start < end <= {ref_len})"
            )

        if len(cols) >= 4 and cols[3].strip() and cols[3].strip() != ".":
            bed_id = cols[3].strip()
        else:
            bed_id = f"{chrom}:{start}-{end}"

        intervals.append((chrom, start, end, bed_id))

    if not intervals:
        raise InputContractError(
            f"Centromere BED file '{path}' contains no valid intervals"
        )
    return intervals


def validate_build_config(config: BuildConfig) -> None:
    """Validate CLI and BuildConfig parameter bounds and centromere mode rules (§3.8.1, §6)."""
    if not config.backbone_prefix or not config.backbone_prefix.strip():
        raise CLIArgumentError("--backbone-prefix must be a non-empty string")
    if config.min_sv_len < 1:
        raise CLIArgumentError(f"--min-sv-len must be >= 1, got {config.min_sv_len}")
    if config.max_jump < 1:
        raise CLIArgumentError(f"--max-jump must be >= 1, got {config.max_jump}")
    if not (0.0 < config.cluster_id <= 1.0):
        raise CLIArgumentError(
            f"--cluster-id must be in (0.0, 1.0], got {config.cluster_id}"
        )
    if not (0.0 < config.cluster_win_id <= 1.0):
        raise CLIArgumentError(
            f"--cluster-win-id must be in (0.0, 1.0], got {config.cluster_win_id}"
        )
    if config.sec_backbone not in ("support", "longest"):
        raise CLIArgumentError(
            f"--sec-backbone must be 'support' or 'longest', got '{config.sec_backbone}'"
        )
    if config.max_copy < 1:
        raise CLIArgumentError(f"--max-copy must be >= 1, got {config.max_copy}")
    if config.spacer_n < 0:
        raise CLIArgumentError(f"--spacer-n must be >= 0, got {config.spacer_n}")
    if config.splice_window < 0:
        raise CLIArgumentError(
            f"--splice-window must be >= 0, got {config.splice_window}"
        )
    if config.threads < 1:
        raise CLIArgumentError(f"--threads must be >= 1, got {config.threads}")
    if config.repeat_guard not in ("on", "off"):
        raise CLIArgumentError(
            f"--repeat-guard must be 'on' or 'off', got '{config.repeat_guard}'"
        )
    if config.centromere_mode not in ("off", "bypass", "extract-islands"):
        raise CLIArgumentError(
            f"--centromere-mode must be 'off', 'bypass', or 'extract-islands', "
            f"got '{config.centromere_mode}'"
        )

    # §3.8.1 & §6 centromere flag interaction checks
    if (
        config.centromere_bed is not None
        and config.centromere_mode == "off"
        and config.centromere_mode_explicit
    ):
        raise CLIArgumentError(
            "Explicit --centromere-mode off conflicts with --centromere-bed (§3.8.1)"
        )

    eff_mode = config.effective_centromere_mode
    if eff_mode in ("bypass", "extract-islands") and not config.centromere_bed:
        raise CLIArgumentError(
            f"--centromere-mode {eff_mode} requires --centromere-bed (§3.8.1)"
        )

    if eff_mode == "off" and config.centromere_min_island_explicit:
        raise CLIArgumentError(
            "--centromere-min-island cannot be specified when centromere processing is off (§6)"
        )

    if (
        eff_mode in ("bypass", "extract-islands")
        and config.centromere_min_island < config.min_sv_len
    ):
        raise CLIArgumentError(
            f"--centromere-min-island ({config.centromere_min_island}) must be >= "
            f"--min-sv-len ({config.min_sv_len}) (§6)"
        )


def _parse_pansn_path_name(path_name: str) -> tuple[str, str, str, str]:
    """
    Parse PanSN path name `Sample#Hap#Seq` (or fallback when `#` is absent).

    Returns `(sample, hap, hap_id, seqid)`.
    """
    if "#" in path_name:
        parts = path_name.split("#")
        if len(parts) >= 3:
            sample = parts[0].strip()
            hap = parts[1].strip()
            seqid = "#".join(parts[2:]).strip()
        else:
            sample = parts[0].strip()
            hap = parts[1].strip()
            seqid = parts[1].strip()
        if not sample or not hap or not seqid:
            raise InputContractError(
                f"Malformed PanSN path name '{path_name}' in GFA P line"
            )
        return sample, hap, f"{sample}#{hap}", seqid
    return path_name, "1", path_name, path_name


def _build_step_provenance(
    steps: Sequence[Step],
    nodes: Mapping[str, str],
    src_hap: str,
    src_path: str,
    base_offset: int,
) -> tuple[tuple[StepProvenance, ...], int]:
    """Build 0-based StepProvenance tuples starting at `base_offset` on `src_path`."""
    prov_list: list[StepProvenance] = []
    cur = base_offset
    for step in steps:
        slen = len(nodes[step.node_id])
        prov_list.append(
            StepProvenance(
                src_hap=src_hap,
                src_path=src_path,
                src_start=cur,
                src_end=cur + slen,
                src_strand="+",
            )
        )
        cur += slen
    return tuple(prov_list), cur - base_offset


def load_and_validate_inputs(config: BuildConfig) -> GFAGraph:
    """
    Load and validate the input GFA, reference FASTA, and optional centromere BED (§2.1, §3.8.1).
    """
    validate_build_config(config)

    # 1. Load and validate unique reference FASTA record
    fasta_records = parse_fasta(config.ref_fa)
    if len(fasta_records) != 1:
        raise InputContractError(
            f"Reference FASTA '{config.ref_fa}' must contain exactly 1 sequence, "
            f"found {len(fasta_records)}"
        )
    ref_chrom, ref_seq = next(iter(fasta_records.items()))

    # 2. Read GFA bytes and compute SHA-256
    gfa_file = Path(config.gfa)
    if not gfa_file.is_file():
        raise InputContractError(f"GFA file not found: '{config.gfa}'")
    raw_bytes = gfa_file.read_bytes()
    if not raw_bytes.strip():
        raise InputContractError(f"GFA file is empty: '{config.gfa}'")
    gfa_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    gfa_text = raw_bytes.decode("utf-8")

    s_records: list[tuple[int, list[str]]] = []
    l_records: list[tuple[int, list[str]]] = []
    p_records: list[tuple[int, list[str]]] = []
    w_records: list[tuple[int, list[str]]] = []

    for line_no, raw_line in enumerate(gfa_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = raw_line.rstrip("\r\n").split("\t")
        rec_type = fields[0]
        if rec_type == "H":
            continue
        if rec_type == "J":
            raise InputContractError(
                f"J (jump) lines are forbidden by §2.1 input contract (line {line_no})"
            )
        if rec_type == "S":
            s_records.append((line_no, fields))
        elif rec_type == "L":
            l_records.append((line_no, fields))
        elif rec_type == "P":
            p_records.append((line_no, fields))
        elif rec_type == "W":
            w_records.append((line_no, fields))
        elif rec_type in ("C",):
            raise InputContractError(
                f"Unsupported GFA record type '{rec_type}' at line {line_no}"
            )

    # 3. Parse S lines (string node IDs, explicit sequence required)
    nodes: dict[str, str] = {}
    for line_no, fields in s_records:
        if len(fields) < 3:
            raise InputContractError(
                f"Malformed S line at line {line_no}: expected >= 3 fields"
            )
        node_id = fields[1].strip()
        seq = fields[2].strip()
        if not node_id:
            raise InputContractError(f"Empty node ID in S line at line {line_no}")
        if node_id in nodes:
            raise InputContractError(
                f"Duplicate S node ID '{node_id}' at line {line_no}"
            )
        if seq == "*" or not seq:
            raise InputContractError(
                f"Missing node sequence ('{seq}') for node '{node_id}' at line {line_no}"
            )
        if not seq.isalpha():
            raise InputContractError(
                f"Invalid characters in sequence for node '{node_id}' at line {line_no}"
            )
        for tag in fields[3:]:
            if tag.startswith("LN:i:"):
                try:
                    tag_len = int(tag.split(":", 2)[2])
                except ValueError as exc:
                    raise InputContractError(
                        f"Invalid LN tag '{tag}' on node '{node_id}' at line {line_no}"
                    ) from exc
                if tag_len != len(seq):
                    raise InputContractError(
                        f"LN:i:{tag_len} does not match sequence length {len(seq)} "
                        f"for node '{node_id}' at line {line_no}"
                    )
        nodes[node_id] = seq.upper()

    if not nodes:
        raise InputContractError(f"GFA file '{config.gfa}' contains no S node records")

    # 4. Parse L lines (must be 0M, index both forward and Watson-Crick reverse-complement)
    links: set[tuple[str, str, str, str]] = set()
    for line_no, fields in l_records:
        if len(fields) < 6:
            raise InputContractError(
                f"Malformed L line at line {line_no}: expected >= 6 fields"
            )
        from_id = fields[1].strip()
        from_orient = fields[2].strip()
        to_id = fields[3].strip()
        to_orient = fields[4].strip()
        overlap = fields[5].strip()

        if from_id not in nodes:
            raise InputContractError(
                f"L line at line {line_no} references unknown node '{from_id}'"
            )
        if to_id not in nodes:
            raise InputContractError(
                f"L line at line {line_no} references unknown node '{to_id}'"
            )
        if from_orient not in ("+", "-") or to_orient not in ("+", "-"):
            raise InputContractError(
                f"Invalid orientation in L line at line {line_no}: "
                f"'{from_orient}' -> '{to_orient}'"
            )
        if overlap != "0M":
            raise InputContractError(
                f"Non-zero or unknown L overlap '{overlap}' at line {line_no} "
                f"(only '0M' is allowed by §2.1)"
            )

        links.add((from_id, from_orient, to_id, to_orient))
        links.add(
            (
                to_id,
                _flip_orient(to_orient),
                from_id,
                _flip_orient(from_orient),
            )
        )

    # 5. Parse P and W lines
    all_fragments: list[PathFragment] = []
    seen_p_names: set[str] = set()
    # Track intervals per (hap_id, seqid) for P/W duplicate and W-overlap detection
    p_contig_intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = (
        collections.defaultdict(list)
    )
    p_by_path_name: dict[str, tuple[int, int]] = {}
    w_contig_intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = (
        collections.defaultdict(list)
    )

    for line_no, fields in p_records:
        if len(fields) < 4:
            raise InputContractError(
                f"Malformed P line at line {line_no}: expected >= 4 fields"
            )
        path_name = fields[1].strip()
        seg_str = fields[2].strip()
        overlaps_str = fields[3].strip()

        if not path_name:
            raise InputContractError(f"Empty path name in P line at line {line_no}")
        if path_name in seen_p_names:
            raise InputContractError(
                f"Duplicate P line path name '{path_name}' at line {line_no}"
            )
        seen_p_names.add(path_name)

        if not seg_str:
            raise InputContractError(
                f"Empty step list in P line '{path_name}' at line {line_no}"
            )
        raw_steps = seg_str.split(",")
        steps: list[Step] = []
        for token in raw_steps:
            m = _P_STEP_RE.match(token.strip())
            if not m:
                raise InputContractError(
                    f"Malformed step token '{token}' in P line '{path_name}' at line {line_no}"
                )
            nid, orient = m.group(1), m.group(2)
            if nid not in nodes:
                raise InputContractError(
                    f"P line '{path_name}' at line {line_no} references unknown node '{nid}'"
                )
            steps.append(Step(node_id=nid, orient=orient))  # type: ignore[arg-type]

        # Validate P overlaps and link continuity
        if len(steps) <= 1:
            if overlaps_str not in ("*", "0M", ""):
                raise InputContractError(
                    f"Invalid overlap '{overlaps_str}' on single-step P line "
                    f"'{path_name}' at line {line_no}"
                )
        else:
            if overlaps_str != "*":
                ov_tokens = [tok.strip() for tok in overlaps_str.split(",")]
                if len(ov_tokens) != len(steps) - 1:
                    raise InputContractError(
                        f"P line '{path_name}' at line {line_no} has {len(ov_tokens)} "
                        f"overlap fields for {len(steps)} steps"
                    )
                for ov in ov_tokens:
                    if ov not in ("0M", "*"):
                        raise InputContractError(
                            f"Non-zero or unknown P overlap '{ov}' in path "
                            f"'{path_name}' at line {line_no}"
                        )
            for idx in range(len(steps) - 1):
                u, v = steps[idx], steps[idx + 1]
                if (u.node_id, u.orient, v.node_id, v.orient) not in links:
                    raise InputContractError(
                        f"Missing 0M L-link for transition {u.node_id}{u.orient} -> "
                        f"{v.node_id}{v.orient} in P line '{path_name}' at line {line_no}"
                    )

        sample, hap, hap_id, seqid = _parse_pansn_path_name(path_name)
        provenance, total_len = _build_step_provenance(
            steps=steps,
            nodes=nodes,
            src_hap=hap_id,
            src_path=path_name,
            base_offset=0,
        )
        frag = PathFragment(
            sample=sample,
            hap=hap,
            hap_id=hap_id,
            seqid=seqid,
            src_path=path_name,
            base_offset=0,
            steps=tuple(steps),
            provenance=provenance,
            is_w_line=False,
            total_len=total_len,
        )
        all_fragments.append(frag)
        p_contig_intervals[(hap_id, seqid)].append((0, total_len, path_name))
        p_by_path_name[path_name] = (0, total_len)

    for line_no, fields in w_records:
        if len(fields) < 7:
            raise InputContractError(
                f"Malformed W line at line {line_no}: expected >= 7 fields"
            )
        sample = fields[1].strip()
        hap = fields[2].strip()
        seqid = fields[3].strip()
        start_str = fields[4].strip()
        end_str = fields[5].strip()
        walk_str = fields[6].strip()

        if not sample or sample == "*" or not hap or hap == "*" or not seqid or seqid == "*":
            raise InputContractError(
                f"Missing sample/haplotype/seqid in W line at line {line_no}"
            )
        try:
            seq_start = 0 if start_str == "*" else int(start_str)
        except ValueError as exc:
            raise InputContractError(
                f"Invalid seq_start '{start_str}' in W line at line {line_no}"
            ) from exc
        if seq_start < 0:
            raise InputContractError(
                f"Negative seq_start {seq_start} in W line at line {line_no}"
            )

        if not walk_str or not _W_WALK_RE.match(walk_str):
            raise InputContractError(
                f"Malformed walk string '{walk_str}' in W line at line {line_no}"
            )

        steps = []
        for m in _W_STEP_RE.finditer(walk_str):
            sym, nid = m.group(1), m.group(2)
            if nid not in nodes:
                raise InputContractError(
                    f"W line at line {line_no} references unknown node '{nid}'"
                )
            orient: Literal["+", "-"] = "+" if sym == ">" else "-"
            steps.append(Step(node_id=nid, orient=orient))

        if not steps:
            raise InputContractError(f"Empty walk in W line at line {line_no}")

        for idx in range(len(steps) - 1):
            u, v = steps[idx], steps[idx + 1]
            if (u.node_id, u.orient, v.node_id, v.orient) not in links:
                raise InputContractError(
                    f"Broken link {u.node_id}{u.orient} -> {v.node_id}{v.orient} "
                    f"in W line at line {line_no}"
                )

        spelled_len = sum(len(nodes[s.node_id]) for s in steps)
        if end_str != "*":
            try:
                seq_end = int(end_str)
            except ValueError as exc:
                raise InputContractError(
                    f"Invalid seq_end '{end_str}' in W line at line {line_no}"
                ) from exc
            if seq_end - seq_start != spelled_len:
                raise InputContractError(
                    f"W line coordinate span [{seq_start}, {seq_end}) (length "
                    f"{seq_end - seq_start}) does not match spelled walk length "
                    f"{spelled_len} at line {line_no}"
                )
        else:
            seq_end = seq_start + spelled_len

        hap_id = f"{sample}#{hap}"
        pan_name = f"{sample}#{hap}#{seqid}"
        src_path = f"{pan_name}:{seq_start}-{seq_end}"
        provenance, total_len = _build_step_provenance(
            steps=steps,
            nodes=nodes,
            src_hap=hap_id,
            src_path=src_path,
            base_offset=seq_start,
        )
        frag = PathFragment(
            sample=sample,
            hap=hap,
            hap_id=hap_id,
            seqid=seqid,
            src_path=src_path,
            base_offset=seq_start,
            steps=tuple(steps),
            provenance=provenance,
            is_w_line=True,
            total_len=total_len,
        )
        all_fragments.append(frag)
        w_contig_intervals[(hap_id, seqid)].append((seq_start, seq_end, src_path))

        # Check P/W duplicate interval ambiguity
        if pan_name in p_by_path_name:
            p_start, p_end = p_by_path_name[pan_name]
            if max(p_start, seq_start) < min(p_end, seq_end):
                raise InputContractError(
                    f"Ambiguous duplicate source interval in both P line '{pan_name}' "
                    f"and W line '{src_path}' (§2.1)"
                )
        for p_start, p_end, p_name in p_contig_intervals.get((hap_id, seqid), []):
            if max(p_start, seq_start) < min(p_end, seq_end):
                raise InputContractError(
                    f"Ambiguous duplicate source interval in both P line '{p_name}' "
                    f"and W line '{src_path}' (§2.1)"
                )

    # Check overlapping W-line intervals on the same (hap_id, seqid)
    for (hap_id, seqid), iv_list in w_contig_intervals.items():
        sorted_ivs = sorted(iv_list, key=lambda x: (x[0], x[1]))
        for i in range(len(sorted_ivs) - 1):
            s1, e1, name1 = sorted_ivs[i]
            s2, e2, name2 = sorted_ivs[i + 1]
            if e1 > s2:
                raise InputContractError(
                    f"Overlapping W-line intervals for '{hap_id}#{seqid}': "
                    f"'{name1}' [{s1}, {e1}) overlaps '{name2}' [{s2}, {e2}) (§2.1)"
                )

    # 6. Match unique reference path by --backbone-prefix
    prefix = config.backbone_prefix
    matched_indices: list[int] = []
    for idx, frag in enumerate(all_fragments):
        pan_name = f"{frag.sample}#{frag.hap}#{frag.seqid}"
        if frag.src_path.startswith(prefix) or pan_name.startswith(prefix):
            matched_indices.append(idx)

    if len(matched_indices) != 1:
        raise InputContractError(
            f"Expected exactly 1 reference path matching --backbone-prefix "
            f"'{prefix}', found {len(matched_indices)}"
        )

    ref_idx = matched_indices[0]
    ref_fragment = all_fragments[ref_idx]
    alt_fragments = tuple(
        frag for idx, frag in enumerate(all_fragments) if idx != ref_idx
    )

    # Verify spelled reference path matches --ref-fa base-for-base (case-insensitive, exact N match)
    assembled_ref = spell_steps(ref_fragment.steps, nodes).upper()
    if len(assembled_ref) != len(ref_seq):
        raise InputContractError(
            f"Assembled reference path '{ref_fragment.src_path}' length ({len(assembled_ref)}) "
            f"does not match --ref-fa '{ref_chrom}' length ({len(ref_seq)})"
        )
    if assembled_ref != ref_seq:
        for pos, (b_gfa, b_fa) in enumerate(zip(assembled_ref, ref_seq)):
            if b_gfa != b_fa:
                raise InputContractError(
                    f"Assembled reference path '{ref_fragment.src_path}' mismatches "
                    f"--ref-fa '{ref_chrom}' at 0-based position {pos}: "
                    f"GFA='{b_gfa}' vs FASTA='{b_fa}'"
                )

    # Validate optional centromere BED against reference chromosome and length
    if config.centromere_bed is not None:
        parse_bed(config.centromere_bed, ref_chrom, len(ref_seq))

    # 7. Index reference steps and haplotypes
    ref_step_ranges_list: list[tuple[int, int]] = []
    cur_ref_pos = 0
    node_counts: collections.Counter[str] = collections.Counter()
    for step in ref_fragment.steps:
        slen = len(nodes[step.node_id])
        ref_step_ranges_list.append((cur_ref_pos, cur_ref_pos + slen))
        cur_ref_pos += slen
        node_counts[step.node_id] += 1

    global_ref_nodes = frozenset(node_counts.keys())
    unique_ref_step_by_node: dict[str, int] = {
        step.node_id: step_idx
        for step_idx, step in enumerate(ref_fragment.steps)
        if node_counts[step.node_id] == 1
    }

    alt_haps = sorted(
        {frag.hap_id for frag in alt_fragments if frag.hap_id != ref_fragment.hap_id}
    )
    all_haplotypes = (ref_fragment.hap_id, *alt_haps)

    return GFAGraph(
        gfa_path=str(gfa_file),
        gfa_sha256=gfa_sha256,
        ref_chrom=ref_chrom,
        ref_seq=ref_seq,
        nodes=nodes,
        links=links,
        ref_fragment=ref_fragment,
        alt_fragments=alt_fragments,
        all_haplotypes=all_haplotypes,
        ref_step_ranges=tuple(ref_step_ranges_list),
        global_ref_nodes=global_ref_nodes,
        unique_ref_step_by_node=unique_ref_step_by_node,
    )
