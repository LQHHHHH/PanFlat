from __future__ import annotations

from array import array
import collections
import hashlib
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence

import numpy as np

from panflat.models import (
    BuildConfig,
    CLIArgumentError,
    CompactLinkSet,
    CompactNodeStore,
    CompactRefNodeSet,
    CompactUniqueRefMap,
    GFAGraph,
    InputContractError,
    LazyProvenanceTuple,
    LazyRefStepRanges,
    LazyStepTuple,
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

_FAST_P_SEG_RE = re.compile(r"^[0-9]+[+\-](?:,[0-9]+[+\-])*$")
_FAST_P_OV_RE = re.compile(r"^(?:0M|\*)(?:,(?:0M|\*))*$")
_DROP_SIGN_TABLE = str.maketrans("", "", "+-")
_DIGITS_COMMA_BYTES = b"0123456789,"


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
    if isinstance(steps, LazyStepTuple) and isinstance(nodes, CompactNodeStore):
        seq_bytes = nodes._seq_bytes
        offsets = nodes._offsets
        node_lens = nodes._node_lens
        max_nid = nodes._max_nid
        parts_fast: list[str] = []
        for c in steps._codes.tolist():
            nid = c if c > 0 else -c
            if nid < 1 or nid > max_nid or int(node_lens[nid]) <= 0:
                raise InputContractError(f"Unknown node ID '{nid}' in step walk")
            s = int(offsets[nid])
            e = int(offsets[nid + 1])
            raw = seq_bytes[s:e].decode("ascii")
            parts_fast.append(raw if c > 0 else revcomp(raw))
        return "".join(parts_fast)

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


def parse_fasta(
    path: str,
    keep_names: set[str] | None = None,
) -> dict[str, str]:
    """
    Stream and parse a FASTA file into `{seq_name: uppercase_sequence}` with strict validation.
    When `keep_names` is provided, all records are still validated for syntax and uniqueness,
    but only sequence records whose name is in `keep_names` are retained in memory.
    """
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise InputContractError(f"FASTA file not found: '{path}'")

    records: dict[str, str] = {}
    seen_names: set[str] = set()
    current_name: str | None = None
    current_chunks: list[str] = []
    current_has_seq = False
    current_keep = True
    saw_nonempty_line = False

    def _flush_current() -> None:
        if current_name is None:
            return
        if not current_has_seq:
            raise InputContractError(
                f"Empty sequence for FASTA record '{current_name}' in '{path}'"
            )
        if current_keep:
            records[current_name] = "".join(current_chunks)

    is_large_fa = fasta_path.stat().st_size > 10_000_000
    with fasta_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            if (
                is_large_fa
                and not current_keep
                and current_name is not None
                and raw_line[:1] != ">"
            ):
                if raw_line.strip():
                    current_has_seq = True
                continue
            line = raw_line.strip()
            if not line:
                continue
            saw_nonempty_line = True
            if line.startswith(">"):
                _flush_current()
                header_body = line[1:].strip()
                if not header_body:
                    raise InputContractError(
                        f"Empty FASTA header at line {line_no} in '{path}'"
                    )
                seq_name = header_body.split()[0]
                if seq_name in seen_names:
                    raise InputContractError(
                        f"Duplicate FASTA sequence name '{seq_name}' in '{path}'"
                    )
                seen_names.add(seq_name)
                current_name = seq_name
                current_chunks = []
                current_has_seq = False
                current_keep = keep_names is None or seq_name in keep_names
            else:
                if current_name is None:
                    raise InputContractError(
                        f"FASTA sequence data before header at line {line_no} in '{path}'"
                    )
                chunk = "".join(line.split()).upper()
                if not chunk:
                    continue
                if not chunk.isalpha():
                    raise InputContractError(
                        f"Non-alphabetic characters in FASTA record '{current_name}' in '{path}'"
                    )
                current_has_seq = True
                if current_keep:
                    current_chunks.append(chunk)

    if not saw_nonempty_line:
        raise InputContractError(f"FASTA file is empty: '{path}'")
    _flush_current()
    if not seen_names:
        raise InputContractError(f"No valid FASTA records found in '{path}'")
    return records


def _load_reference_fasta_for_fragment(
    fasta_file_path: str,
    ref_fragment: PathFragment,
) -> tuple[str, str]:
    """
    Stream `--ref-fa` and extract `(ref_chrom, ref_seq)` matching `ref_fragment`.
    - Single-record FASTA: uses that record (normalizing `Sample#Hap#Seq` header to `Seq`
      when it matches `ref_fragment`).
    - Multi-record FASTA: requires a unique PanSN record matching `ref_fragment.src_path`
      or `f"{ref_fragment.sample}#{ref_fragment.hap}#{ref_fragment.seqid}"` (containing `#`);
      only that single record is retained in memory.
    """
    fasta_path = Path(fasta_file_path)
    if not fasta_path.is_file():
        raise InputContractError(f"FASTA file not found: '{fasta_file_path}'")

    ref_pan_name = f"{ref_fragment.sample}#{ref_fragment.hap}#{ref_fragment.seqid}"
    target_pansn_names = {ref_fragment.src_path, ref_pan_name}

    seen_names: set[str] = set()
    first_name: str | None = None
    first_seq: str | None = None
    matched_pansn_seq: str | None = None

    current_name: str | None = None
    current_chunks: list[str] = []
    current_has_seq = False
    current_keep = False
    saw_nonempty_line = False

    def _flush_current() -> None:
        nonlocal first_name, first_seq, matched_pansn_seq
        if current_name is None:
            return
        if not current_has_seq:
            raise InputContractError(
                f"Empty sequence for FASTA record '{current_name}' in '{fasta_file_path}'"
            )
        if current_keep:
            seq_val = "".join(current_chunks)
            if len(seen_names) == 1:
                first_name = current_name
                first_seq = seq_val
            if "#" in current_name and current_name in target_pansn_names:
                matched_pansn_seq = seq_val

    is_large_fa = fasta_path.stat().st_size > 10_000_000
    with fasta_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            if (
                is_large_fa
                and not current_keep
                and current_name is not None
                and raw_line[:1] != ">"
            ):
                if raw_line.strip():
                    current_has_seq = True
                continue
            line = raw_line.strip()
            if not line:
                continue
            saw_nonempty_line = True
            if line.startswith(">"):
                _flush_current()
                header_body = line[1:].strip()
                if not header_body:
                    raise InputContractError(
                        f"Empty FASTA header at line {line_no} in '{fasta_file_path}'"
                    )
                seq_name = header_body.split()[0]
                if seq_name in seen_names:
                    raise InputContractError(
                        f"Duplicate FASTA sequence name '{seq_name}' in '{fasta_file_path}'"
                    )
                seen_names.add(seq_name)
                if len(seen_names) > 1:
                    # Drop first_seq immediately to bound memory on multi-FASTA
                    first_seq = None
                current_name = seq_name
                current_chunks = []
                current_has_seq = False
                is_target_pansn = ("#" in seq_name) and (seq_name in target_pansn_names)
                current_keep = (len(seen_names) == 1) or is_target_pansn
            else:
                if current_name is None:
                    raise InputContractError(
                        f"FASTA sequence data before header at line {line_no} in '{fasta_file_path}'"
                    )
                chunk = "".join(line.split()).upper()
                if not chunk:
                    continue
                if not chunk.isalpha():
                    raise InputContractError(
                        f"Non-alphabetic characters in FASTA record '{current_name}' in '{fasta_file_path}'"
                    )
                current_has_seq = True
                if current_keep:
                    current_chunks.append(chunk)

    if not saw_nonempty_line:
        raise InputContractError(f"FASTA file is empty: '{fasta_file_path}'")
    _flush_current()
    num_records = len(seen_names)
    if num_records == 0:
        raise InputContractError(f"No valid FASTA records found in '{fasta_file_path}'")

    if num_records == 1:
        assert first_name is not None and first_seq is not None
        if "#" in first_name and first_name in target_pansn_names:
            return ref_fragment.seqid, first_seq
        return first_name, first_seq

    if matched_pansn_seq is not None:
        return ref_fragment.seqid, matched_pansn_seq

    raise InputContractError(
        f"Reference FASTA '{fasta_file_path}' must contain exactly 1 sequence, "
        f"found {num_records}"
    )


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
    Uses streaming GFA/FASTA parsing and compact NumPy-backed containers for chromosome-scale graphs.
    """
    validate_build_config(config)

    # Check FASTA exists early before scanning a large GFA
    if not Path(config.ref_fa).is_file():
        raise InputContractError(f"FASTA file not found: '{config.ref_fa}'")

    gfa_file = Path(config.gfa)
    if not gfa_file.is_file():
        raise InputContractError(f"GFA file not found: '{config.gfa}'")

    # 1. Stream GFA: compute SHA-256, parse S and L records into compact buffers, and record P/W byte offsets
    hasher = hashlib.sha256()
    saw_nonempty_gfa = False

    seq_buf = bytearray()
    node_lens_arr = array("i", [0])  # 1-based index: node_lens_arr[nid]
    expected_next_nid = 1
    out_of_order_int_nodes: dict[int, bytes] = {}
    extra_nodes: dict[str, str] = {}
    max_int_nid = 0

    canon_links_arr = array("Q")
    deferred_int_links: list[tuple[int, int, int]] = []
    extra_raw_links: list[tuple[int, str, str, str, str]] = []

    p_line_offsets: list[tuple[int, int]] = []
    w_line_offsets: list[tuple[int, int]] = []

    with gfa_file.open("rb") as fh:
        line_no = 0
        cur_byte_offset = 0
        while True:
            raw_bytes_line = fh.readline()
            if not raw_bytes_line:
                break
            byte_offset = cur_byte_offset
            cur_byte_offset += len(raw_bytes_line)
            line_no += 1
            hasher.update(raw_bytes_line)

            first_byte = raw_bytes_line[0:2]
            if first_byte == b"S\t":
                saw_nonempty_gfa = True
                b_parts = raw_bytes_line.rstrip(b"\r\n").split(b"\t")
                if (
                    len(b_parts) == 3
                    and b_parts[1].isdigit()
                    and b_parts[1][0] != 48
                    and b_parts[2].isalpha()
                ):
                    nid_int = int(b_parts[1])
                    if (
                        (
                            nid_int < len(node_lens_arr)
                            and node_lens_arr[nid_int] > 0
                        )
                        or nid_int in out_of_order_int_nodes
                    ):
                        raise InputContractError(
                            f"Duplicate S node ID '{nid_int}' at line {line_no}"
                        )
                    seq_b = b_parts[2].upper()
                    if (
                        nid_int == expected_next_nid
                        and not out_of_order_int_nodes
                    ):
                        seq_buf.extend(seq_b)
                        node_lens_arr.append(len(seq_b))
                        expected_next_nid += 1
                    else:
                        out_of_order_int_nodes[nid_int] = seq_b
                    if nid_int > max_int_nid:
                        max_int_nid = nid_int
                    continue
            elif first_byte == b"L\t":
                saw_nonempty_gfa = True
                b_parts = raw_bytes_line.rstrip(b"\r\n").split(b"\t")
                if (
                    len(b_parts) == 6
                    and b_parts[5] == b"0M"
                    and b_parts[2] in (b"+", b"-")
                    and b_parts[4] in (b"+", b"-")
                    and b_parts[1].isdigit()
                    and b_parts[3].isdigit()
                    and b_parts[1][0] != 48
                    and b_parts[3][0] != 48
                ):
                    u_int = int(b_parts[1])
                    v_int = int(b_parts[3])
                    n_l_arr = len(node_lens_arr)
                    if (
                        u_int >= n_l_arr
                        or v_int >= n_l_arr
                        or node_lens_arr[u_int] <= 0
                        or node_lens_arr[v_int] <= 0
                    ):
                        deferred_int_links.append((line_no, u_int, v_int))
                    uc = (u_int << 1) | (0 if b_parts[2] == b"+" else 1)
                    vc = (v_int << 1) | (0 if b_parts[4] == b"+" else 1)
                    fwd = (uc << 32) | vc
                    rev = ((vc ^ 1) << 32) | (uc ^ 1)
                    canon_links_arr.append(fwd if fwd <= rev else rev)
                    continue

            stripped_b = raw_bytes_line.strip()
            if not stripped_b:
                continue
            saw_nonempty_gfa = True
            if stripped_b.startswith(b"#"):
                continue

            if first_byte == b"H\t" or stripped_b == b"H":
                continue

            raw_line = raw_bytes_line.decode("utf-8")
            fields = raw_line.rstrip("\r\n").split("\t")
            rec_type = fields[0]
            if rec_type == "H":
                continue
            if rec_type == "J":
                raise InputContractError(
                    f"J (jump) lines are forbidden by §2.1 input contract (line {line_no})"
                )
            if rec_type in ("C",):
                raise InputContractError(
                    f"Unsupported GFA record type '{rec_type}' at line {line_no}"
                )

            if rec_type == "S":
                if len(fields) < 3:
                    raise InputContractError(
                        f"Malformed S line at line {line_no}: expected >= 3 fields"
                    )
                node_id = fields[1].strip()
                seq = fields[2].strip()
                if not node_id:
                    raise InputContractError(f"Empty node ID in S line at line {line_no}")
                if seq == "*" or not seq:
                    raise InputContractError(
                        f"Missing node sequence ('{seq}') for node '{node_id}' at line {line_no}"
                    )
                if not seq.isalpha():
                    raise InputContractError(
                        f"Invalid characters in sequence for node '{node_id}' at line {line_no}"
                    )
                if len(fields) > 3:
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
                seq_up = seq.upper()
                if node_id.isdigit() and int(node_id) >= 1 and str(int(node_id)) == node_id:
                    nid_int = int(node_id)
                    if (
                        (nid_int < len(node_lens_arr) and node_lens_arr[nid_int] > 0)
                        or nid_int in out_of_order_int_nodes
                    ):
                        raise InputContractError(
                            f"Duplicate S node ID '{node_id}' at line {line_no}"
                        )
                    seq_b = seq_up.encode("ascii")
                    if nid_int == expected_next_nid and not out_of_order_int_nodes:
                        seq_buf.extend(seq_b)
                        node_lens_arr.append(len(seq_b))
                        expected_next_nid += 1
                    else:
                        out_of_order_int_nodes[nid_int] = seq_b
                    if nid_int > max_int_nid:
                        max_int_nid = nid_int
                else:
                    if node_id in extra_nodes:
                        raise InputContractError(
                            f"Duplicate S node ID '{node_id}' at line {line_no}"
                        )
                    extra_nodes[node_id] = seq_up

            elif rec_type == "L":
                if len(fields) < 6:
                    raise InputContractError(
                        f"Malformed L line at line {line_no}: expected >= 6 fields"
                    )
                from_id = fields[1].strip()
                from_orient = fields[2].strip()
                to_id = fields[3].strip()
                to_orient = fields[4].strip()
                overlap = fields[5].strip()
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
                if (
                    from_id.isdigit()
                    and to_id.isdigit()
                    and str(int(from_id)) == from_id
                    and str(int(to_id)) == to_id
                ):
                    u_int = int(from_id)
                    v_int = int(to_id)
                    if (
                        u_int < 1
                        or v_int < 1
                        or u_int >= len(node_lens_arr)
                        or v_int >= len(node_lens_arr)
                        or node_lens_arr[u_int] <= 0
                        or node_lens_arr[v_int] <= 0
                    ):
                        deferred_int_links.append((line_no, u_int, v_int))
                    canon_links_arr.append(
                        CompactLinkSet.encode_canon(
                            u_int,
                            0 if from_orient == "+" else 1,
                            v_int,
                            0 if to_orient == "+" else 1,
                        )
                    )
                else:
                    extra_raw_links.append(
                        (line_no, from_id, from_orient, to_id, to_orient)
                    )

            elif rec_type == "P":
                p_line_offsets.append((line_no, byte_offset))
            elif rec_type == "W":
                w_line_offsets.append((line_no, byte_offset))

    if not saw_nonempty_gfa:
        raise InputContractError(f"GFA file is empty: '{config.gfa}'")
    gfa_sha256 = hasher.hexdigest()

    # Finalize node storage
    if out_of_order_int_nodes:
        # Rebuild sequential buffer up to max_int_nid
        new_seq_buf = bytearray()
        new_lens = np.zeros(max_int_nid + 1, dtype=np.int32)
        cur_off = 0
        for nid in range(1, expected_next_nid):
            slen = node_lens_arr[nid]
            new_seq_buf.extend(seq_buf[cur_off : cur_off + slen])
            new_lens[nid] = slen
            cur_off += slen
        for nid in sorted(out_of_order_int_nodes.keys()):
            sb = out_of_order_int_nodes[nid]
            # Place in sequential order by rebuilding full array
        # Full rebuild in exact 1..max_int_nid order
        new_seq_buf = bytearray()
        new_lens = np.zeros(max_int_nid + 1, dtype=np.int32)
        in_order_slices: dict[int, bytes] = {}
        cur_off = 0
        for nid in range(1, expected_next_nid):
            slen = node_lens_arr[nid]
            in_order_slices[nid] = bytes(seq_buf[cur_off : cur_off + slen])
            cur_off += slen
        in_order_slices.update(out_of_order_int_nodes)
        out_of_order_int_nodes.clear()
        for nid in range(1, max_int_nid + 1):
            sb = in_order_slices.get(nid)
            if sb is not None:
                new_seq_buf.extend(sb)
                new_lens[nid] = len(sb)
        in_order_slices.clear()
        seq_bytes = bytes(new_seq_buf)
        del new_seq_buf, seq_buf, node_lens_arr
        node_lens_np = new_lens
    else:
        seq_bytes = bytes(seq_buf)
        del seq_buf
        node_lens_np = np.frombuffer(node_lens_arr, dtype=np.int32).copy()
        del node_lens_arr

    total_int_nodes = int(np.count_nonzero(node_lens_np[1:] > 0)) if max_int_nid > 0 else 0
    if total_int_nodes + len(extra_nodes) == 0:
        raise InputContractError(f"GFA file '{config.gfa}' contains no S node records")

    offsets_dtype = np.int64 if len(seq_bytes) > 2_000_000_000 else np.int32
    node_offsets_np = np.zeros(max_int_nid + 2, dtype=offsets_dtype)
    if max_int_nid > 0:
        np.cumsum(node_lens_np[1:], dtype=offsets_dtype, out=node_offsets_np[2:])

    nodes_store = CompactNodeStore(
        seq_bytes=seq_bytes,
        offsets=node_offsets_np,
        node_lens=node_lens_np,
        max_nid=max_int_nid,
        extra=extra_nodes,
        count=total_int_nodes + len(extra_nodes),
    )

    # Validate any deferred integer links and extra string links
    for l_no, u_int, v_int in deferred_int_links:
        if u_int < 1 or u_int > max_int_nid or int(node_lens_np[u_int]) <= 0:
            raise InputContractError(
                f"L line at line {l_no} references unknown node '{u_int}'"
            )
        if v_int < 1 or v_int > max_int_nid or int(node_lens_np[v_int]) <= 0:
            raise InputContractError(
                f"L line at line {l_no} references unknown node '{v_int}'"
            )
    deferred_int_links.clear()

    extra_links_set: set[tuple[str, str, str, str]] = set()
    for l_no, from_id, from_orient, to_id, to_orient in extra_raw_links:
        if from_id not in nodes_store:
            raise InputContractError(
                f"L line at line {l_no} references unknown node '{from_id}'"
            )
        if to_id not in nodes_store:
            raise InputContractError(
                f"L line at line {l_no} references unknown node '{to_id}'"
            )
        extra_links_set.add((from_id, from_orient, to_id, to_orient))
        extra_links_set.add(
            (to_id, _flip_orient(to_orient), from_id, _flip_orient(from_orient))
        )
    extra_raw_links.clear()

    if len(canon_links_arr) > 0:
        canon_links_np = np.unique(np.frombuffer(canon_links_arr, dtype=np.uint64))
    else:
        canon_links_np = np.empty(0, dtype=np.uint64)
    del canon_links_arr

    links_store = CompactLinkSet(canon_links=canon_links_np, extra=extra_links_set)
    use_fast_int_paths = (len(extra_nodes) == 0) and (len(extra_links_set) == 0)

    # 2. Stream P and W lines by seeking to their byte offsets (one line in memory at a time)
    all_fragments: list[PathFragment] = []
    seen_p_names: set[str] = set()
    p_contig_intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = (
        collections.defaultdict(list)
    )
    p_by_path_name: dict[str, tuple[int, int]] = {}
    w_contig_intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = (
        collections.defaultdict(list)
    )

    with gfa_file.open("rb") as fh:
        for line_no, byte_offset in p_line_offsets:
            fh.seek(byte_offset)
            raw_line = fh.readline().decode("utf-8")
            fields = raw_line.rstrip("\r\n").split("\t")
            del raw_line

            if len(fields) < 4:
                raise InputContractError(
                    f"Malformed P line at line {line_no}: expected >= 4 fields"
                )
            path_name = fields[1].strip()
            seg_str = fields[2].strip()
            overlaps_str = fields[3].strip()
            del fields

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

            if use_fast_int_paths and (
                (len(seg_str) > 100_000 and seg_str[-1] in "+-")
                or bool(_FAST_P_SEG_RE.match(seg_str))
            ):
                nids = np.fromstring(
                    seg_str.translate(_DROP_SIGN_TABLE), dtype=np.int32, sep=","
                )
                signs = np.frombuffer(
                    seg_str.encode("ascii").translate(None, _DIGITS_COMMA_BYTES),
                    dtype=np.uint8,
                )
                del seg_str
                n_steps = int(nids.shape[0])
                if int(nids.min()) < 1 or int(nids.max()) > max_int_nid:
                    for nid_val in nids.tolist():
                        if nid_val < 1 or nid_val > max_int_nid or int(node_lens_np[nid_val]) <= 0:
                            raise InputContractError(
                                f"P line '{path_name}' at line {line_no} references unknown node '{nid_val}'"
                            )
                step_lens = node_lens_np[nids]
                if not bool(np.all(step_lens > 0)):
                    bad_pos = int(np.flatnonzero(step_lens <= 0)[0])
                    raise InputContractError(
                        f"P line '{path_name}' at line {line_no} references unknown node '{int(nids[bad_pos])}'"
                    )
                codes = np.where(signs == 45, -nids, nids).astype(np.int32, copy=False)
                del nids, signs

                if n_steps <= 1:
                    if overlaps_str not in ("*", "0M", ""):
                        raise InputContractError(
                            f"Invalid overlap '{overlaps_str}' on single-step P line "
                            f"'{path_name}' at line {line_no}"
                        )
                else:
                    if overlaps_str != "*":
                        if (
                            _FAST_P_OV_RE.match(overlaps_str)
                            and overlaps_str.count(",") == n_steps - 2
                        ):
                            pass
                        else:
                            ov_tokens = [tok.strip() for tok in overlaps_str.split(",")]
                            if len(ov_tokens) != n_steps - 1:
                                raise InputContractError(
                                    f"P line '{path_name}' at line {line_no} has {len(ov_tokens)} "
                                    f"overlap fields for {n_steps} steps"
                                )
                            for ov in ov_tokens:
                                if ov not in ("0M", "*"):
                                    raise InputContractError(
                                        f"Non-zero or unknown P overlap '{ov}' in path "
                                        f"'{path_name}' at line {line_no}"
                                    )
                    bad_idx = links_store.validate_step_codes(codes)
                    if bad_idx >= 0:
                        u_c = int(codes[bad_idx])
                        v_c = int(codes[bad_idx + 1])
                        u_tok = f"{abs(u_c)}{'+' if u_c > 0 else '-'}"
                        v_tok = f"{abs(v_c)}{'+' if v_c > 0 else '-'}"
                        raise InputContractError(
                            f"Missing 0M L-link for transition {u_tok} -> "
                            f"{v_tok} in P line '{path_name}' at line {line_no}"
                        )

                sample, hap, hap_id, seqid = _parse_pansn_path_name(path_name)
                total_len = int(step_lens.sum(dtype=np.int64))
                del step_lens
                steps_seq: Any = LazyStepTuple(codes)
                prov_seq: Any = LazyProvenanceTuple(
                    src_hap=hap_id,
                    src_path=path_name,
                    base_offset=0,
                    orig_codes=codes,
                    node_lens=node_lens_np,
                    total_bp=total_len,
                )
            else:
                raw_steps = seg_str.split(",")
                del seg_str
                steps_list: list[Step] = []
                for token in raw_steps:
                    m = _P_STEP_RE.match(token.strip())
                    if not m:
                        raise InputContractError(
                            f"Malformed step token '{token}' in P line '{path_name}' at line {line_no}"
                        )
                    nid, orient = m.group(1), m.group(2)
                    if nid not in nodes_store:
                        raise InputContractError(
                            f"P line '{path_name}' at line {line_no} references unknown node '{nid}'"
                        )
                    steps_list.append(Step(node_id=nid, orient=orient))  # type: ignore[arg-type]

                if len(steps_list) <= 1:
                    if overlaps_str not in ("*", "0M", ""):
                        raise InputContractError(
                            f"Invalid overlap '{overlaps_str}' on single-step P line "
                            f"'{path_name}' at line {line_no}"
                        )
                else:
                    if overlaps_str != "*":
                        ov_tokens = [tok.strip() for tok in overlaps_str.split(",")]
                        if len(ov_tokens) != len(steps_list) - 1:
                            raise InputContractError(
                                f"P line '{path_name}' at line {line_no} has {len(ov_tokens)} "
                                f"overlap fields for {len(steps_list)} steps"
                            )
                        for ov in ov_tokens:
                            if ov not in ("0M", "*"):
                                raise InputContractError(
                                    f"Non-zero or unknown P overlap '{ov}' in path "
                                    f"'{path_name}' at line {line_no}"
                                )
                    for idx in range(len(steps_list) - 1):
                        u, v = steps_list[idx], steps_list[idx + 1]
                        if (u.node_id, u.orient, v.node_id, v.orient) not in links_store:
                            raise InputContractError(
                                f"Missing 0M L-link for transition {u.node_id}{u.orient} -> "
                                f"{v.node_id}{v.orient} in P line '{path_name}' at line {line_no}"
                            )

                sample, hap, hap_id, seqid = _parse_pansn_path_name(path_name)
                prov_seq, total_len = _build_step_provenance(
                    steps=steps_list,
                    nodes=nodes_store,
                    src_hap=hap_id,
                    src_path=path_name,
                    base_offset=0,
                )
                steps_seq = tuple(steps_list)

            frag = PathFragment(
                sample=sample,
                hap=hap,
                hap_id=hap_id,
                seqid=seqid,
                src_path=path_name,
                base_offset=0,
                steps=steps_seq,
                provenance=prov_seq,
                is_w_line=False,
                total_len=total_len,
            )
            all_fragments.append(frag)
            p_contig_intervals[(hap_id, seqid)].append((0, total_len, path_name))
            p_by_path_name[path_name] = (0, total_len)

        for line_no, byte_offset in w_line_offsets:
            fh.seek(byte_offset)
            raw_line = fh.readline().decode("utf-8")
            fields = raw_line.rstrip("\r\n").split("\t")
            del raw_line

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
            del fields

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

            w_steps: list[Step] = []
            for m in _W_STEP_RE.finditer(walk_str):
                sym, nid = m.group(1), m.group(2)
                if nid not in nodes_store:
                    raise InputContractError(
                        f"W line at line {line_no} references unknown node '{nid}'"
                    )
                orient: Literal["+", "-"] = "+" if sym == ">" else "-"
                w_steps.append(Step(node_id=nid, orient=orient))

            if not w_steps:
                raise InputContractError(f"Empty walk in W line at line {line_no}")

            for idx in range(len(w_steps) - 1):
                u, v = w_steps[idx], w_steps[idx + 1]
                if (u.node_id, u.orient, v.node_id, v.orient) not in links_store:
                    raise InputContractError(
                        f"Broken link {u.node_id}{u.orient} -> {v.node_id}{v.orient} "
                        f"in W line at line {line_no}"
                    )

            spelled_len = sum(nodes_store.node_len(s.node_id) for s in w_steps)
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
                steps=w_steps,
                nodes=nodes_store,
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
                steps=tuple(w_steps),
                provenance=provenance,
                is_w_line=True,
                total_len=total_len,
            )
            all_fragments.append(frag)
            w_contig_intervals[(hap_id, seqid)].append((seq_start, seq_end, src_path))

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

    links_store._first_v = None

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

    # 3. Match unique reference path by --backbone-prefix (with PanSN sample-prefix fallback)
    prefix = config.backbone_prefix
    matched_indices: list[int] = []
    for idx, frag in enumerate(all_fragments):
        pan_name = f"{frag.sample}#{frag.hap}#{frag.seqid}"
        if frag.src_path.startswith(prefix) or pan_name.startswith(prefix):
            matched_indices.append(idx)

    if len(matched_indices) != 1:
        base_sample = (
            prefix[:-3]
            if (
                len(matched_indices) == 0
                and (prefix.endswith("#0#") or prefix.endswith("#1#"))
            )
            else (prefix if "#" not in prefix else None)
        )
        if base_sample is not None:
            # Sample-name prefix (e.g. 'Pyrar_xinyeli' or 'Pyrar_xinyeli#0#') in a multi-haplotype PanSN GFA:
            # check if primary haplotype '#1#' (or '#0#' if '#1#' is absent) uniquely identifies 1 path
            hap1_indices = [
                idx
                for idx in range(len(all_fragments))
                if (
                    all_fragments[idx].sample == base_sample
                    or all_fragments[idx].src_path.startswith(
                        f"{base_sample}#1#"
                    )
                )
                and all_fragments[idx].hap == "1"
                and (
                    "#" in all_fragments[idx].src_path
                    or all_fragments[idx].is_w_line
                )
            ]
            if len(hap1_indices) == 1:
                matched_indices = hap1_indices
            elif len(hap1_indices) == 0:
                hap0_indices = [
                    idx
                    for idx in range(len(all_fragments))
                    if (
                        all_fragments[idx].sample == base_sample
                        or all_fragments[idx].src_path.startswith(
                            f"{base_sample}#0#"
                        )
                    )
                    and all_fragments[idx].hap == "0"
                    and (
                        "#" in all_fragments[idx].src_path
                        or all_fragments[idx].is_w_line
                    )
                ]
                if len(hap0_indices) == 1:
                    matched_indices = hap0_indices

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

    # 4. Load and validate reference FASTA record (supports single-FASTA and PanSN multi-FASTA)
    ref_chrom, ref_seq = _load_reference_fasta_for_fragment(config.ref_fa, ref_fragment)

    # Verify spelled reference path matches --ref-fa base-for-base (case-insensitive, exact N match)
    assembled_ref = spell_steps(ref_fragment.steps, nodes_store).upper()
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
    del assembled_ref

    # Validate optional centromere BED against reference chromosome and length
    if config.centromere_bed is not None:
        parse_bed(config.centromere_bed, ref_chrom, len(ref_seq))

    # 5. Index reference steps and haplotypes using compact NumPy arrays when integer-backed
    if isinstance(ref_fragment.steps, LazyStepTuple) and use_fast_int_paths:
        ref_codes = ref_fragment.steps._codes
        n_ref_steps = int(ref_codes.shape[0])
        ref_nids = np.abs(ref_codes)
        ref_lens = node_lens_np[ref_nids]

        ref_cum_offsets = np.zeros(n_ref_steps + 1, dtype=np.int64)
        np.cumsum(ref_lens, dtype=np.int64, out=ref_cum_offsets[1:])
        ref_step_ranges: Any = LazyRefStepRanges(ref_cum_offsets)

        ref_node_counts = np.bincount(ref_nids, minlength=max_int_nid + 1).astype(
            np.int32, copy=False
        )
        global_ref_mask = ref_node_counts > 0
        global_ref_nodes: Any = CompactRefNodeSet(global_ref_mask)

        nid_to_ref_rank = np.full(max_int_nid + 1, -1, dtype=np.int32)
        nid_to_ref_sign = np.zeros(max_int_nid + 1, dtype=np.int8)
        unique_step_mask = ref_node_counts[ref_nids] == 1
        unique_step_indices = np.flatnonzero(unique_step_mask).astype(np.int32)
        unique_nids = ref_nids[unique_step_indices]
        nid_to_ref_rank[unique_nids] = unique_step_indices
        nid_to_ref_sign[unique_nids] = np.where(
            ref_codes[unique_step_indices] > 0, 1, -1
        ).astype(np.int8)
        unique_ref_step_by_node: Any = CompactUniqueRefMap(
            nid_to_ref_rank, count=int(unique_step_indices.shape[0])
        )
    else:
        ref_step_ranges_list: list[tuple[int, int]] = []
        cur_ref_pos = 0
        node_counts: collections.Counter[str] = collections.Counter()
        for step in ref_fragment.steps:
            slen = nodes_store.node_len(step.node_id)
            ref_step_ranges_list.append((cur_ref_pos, cur_ref_pos + slen))
            cur_ref_pos += slen
            node_counts[step.node_id] += 1

        ref_step_ranges = tuple(ref_step_ranges_list)
        global_ref_nodes = frozenset(node_counts.keys())
        unique_ref_step_by_node = {
            step.node_id: step_idx
            for step_idx, step in enumerate(ref_fragment.steps)
            for _ in (0,)
            if node_counts[step.node_id] == 1
        }
        ref_cum_offsets = None
        nid_to_ref_rank = None
        nid_to_ref_sign = None
        global_ref_mask = None
        ref_node_counts = None

    alt_haps = sorted(
        {frag.hap_id for frag in alt_fragments if frag.hap_id != ref_fragment.hap_id}
    )
    all_haplotypes = (ref_fragment.hap_id, *alt_haps)

    return GFAGraph(
        gfa_path=str(gfa_file),
        gfa_sha256=gfa_sha256,
        ref_chrom=ref_chrom,
        ref_seq=ref_seq,
        nodes=nodes_store,
        links=links_store,
        ref_fragment=ref_fragment,
        alt_fragments=alt_fragments,
        all_haplotypes=all_haplotypes,
        ref_step_ranges=ref_step_ranges,
        global_ref_nodes=global_ref_nodes,
        unique_ref_step_by_node=unique_ref_step_by_node,
        _node_lens=node_lens_np,
        _ref_cum_offsets=ref_cum_offsets,
        _nid_to_ref_rank=nid_to_ref_rank,
        _nid_to_ref_sign=nid_to_ref_sign,
        _global_ref_mask=global_ref_mask,
        _ref_node_counts=ref_node_counts,
    )

