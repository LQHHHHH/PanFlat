"""Core immutable dataclasses and shared type contracts for PanFlat v1."""

from __future__ import annotations

import bisect
import collections.abc
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np


class PanFlatError(ValueError):
    """Base exception for all PanFlat contract, argument, and invariant errors."""


class InputContractError(PanFlatError):
    """Raised when input GFA, FASTA, BED, or GFF3 violates §2.1, §3.8.1, or §4 contracts."""


class CLIArgumentError(InputContractError):
    """Raised when CLI or BuildConfig arguments conflict or are out of valid bounds."""


class InvariantViolationError(PanFlatError):
    """Raised when Global Invariant I, II, III or centromere verification fails (§1, §3.8.5)."""


UNRESOLVED_REASONS: frozenset[str] = frozenset(
    {
        "UNANCHORED_END",
        "LARGE_JUMP",
        "INV_EDGE",
        "INV_CONFLICT",
        "REPEAT_LIKE",
        "CEN_BYPASS",
        "CEN_REPEAT_LIKE",
        "CEN_SHORT_RESIDUAL",
        "CEN_REFERENCE_GAP",
        "CEN_INCOMPLETE",
        "CEN_AMBIGUOUS",
        "CEN_BOUNDARY_AMBIGUOUS",
        "CEN_SEQUENCE_UNCERTAIN",
    }
)

DEFAULT_NO_FASTA_REASONS: frozenset[str] = frozenset(
    {
        "REPEAT_LIKE",
        "CEN_BYPASS",
        "CEN_REPEAT_LIKE",
    }
)

CENTROMERE_TSV_COLUMNS: tuple[str, ...] = (
    "chrom",
    "region_id",
    "input_bed_ids",
    "bed_start",
    "bed_end",
    "ref_start",
    "ref_end",
    "anchor_u",
    "anchor_v",
    "mode",
    "hap",
    "status",
    "ref_status",
    "src_path",
    "src_start",
    "src_end",
    "src_strand",
    "L_ref",
    "L_path",
    "length_ratio",
    "nonref_input_bp",
    "reference_like_bp",
    "repeat_like_bp",
    "short_bp",
    "unresolved_bp",
    "accepted_source_bp",
    "supported_events",
)


@dataclass(frozen=True, slots=True)
class Step:
    """An oriented GFA node traversal step, optionally sliced at partial-node boundaries."""

    node_id: str
    orient: Literal["+", "-"]
    slice_start: int = 0
    slice_end: int | None = None
    is_partial: bool = False

    @property
    def token(self) -> tuple[str, Literal["+", "-"]]:
        """Return (node_id, orient) tuple for full-step token comparison."""
        return (self.node_id, self.orient)

    def step_len(self, nodes: Mapping[str, str]) -> int:
        """Return the base length of this step (accounting for partial-node slicing)."""
        if hasattr(nodes, "node_len"):
            full_len = int(nodes.node_len(self.node_id))  # type: ignore[attr-defined]
        else:
            full_len = len(nodes[self.node_id])
        end = self.slice_end if self.slice_end is not None else full_len
        return end - self.slice_start

    def flipped(self, full_node_len: int | None = None) -> Step:
        """Return the reverse-complemented step, adjusting partial slice coordinates if needed."""
        new_orient: Literal["+", "-"] = "-" if self.orient == "+" else "+"
        if not self.is_partial and self.slice_start == 0 and self.slice_end is None:
            return Step(node_id=self.node_id, orient=new_orient)
        if full_node_len is None:
            raise ValueError("full_node_len is required to flip a sliced/partial Step")
        end = self.slice_end if self.slice_end is not None else full_node_len
        return Step(
            node_id=self.node_id,
            orient=new_orient,
            slice_start=full_node_len - end,
            slice_end=full_node_len - self.slice_start,
            is_partial=self.is_partial,
        )

    def subslice(self, rel_start: int, rel_end: int, full_node_len: int) -> Step:
        """Return a sub-slice [rel_start, rel_end) relative to this step's oriented bases."""
        cur_end = self.slice_end if self.slice_end is not None else full_node_len
        cur_len = cur_end - self.slice_start
        if rel_start < 0 or rel_end > cur_len or rel_start >= rel_end:
            raise ValueError(
                f"Invalid step subslice [{rel_start}, {rel_end}) for step length {cur_len}"
            )
        new_start = self.slice_start + rel_start
        new_end = self.slice_start + rel_end
        if new_start == 0 and new_end == full_node_len and not self.is_partial:
            return Step(node_id=self.node_id, orient=self.orient)
        return Step(
            node_id=self.node_id,
            orient=self.orient,
            slice_start=new_start,
            slice_end=new_end,
            is_partial=True,
        )


@dataclass(frozen=True, slots=True)
class StepProvenance:
    """0-based source coordinate interval on the original unflipped GFA P/W path."""

    src_hap: str
    src_path: str
    src_start: int
    src_end: int
    src_strand: Literal["+", "-"]

    @property
    def length(self) -> int:
        return self.src_end - self.src_start

    def flipped(self) -> StepProvenance:
        """Flip the source strand when a contig or inversion segment is reverse-complemented."""
        new_strand: Literal["+", "-"] = "-" if self.src_strand == "+" else "+"
        return StepProvenance(
            src_hap=self.src_hap,
            src_path=self.src_path,
            src_start=self.src_start,
            src_end=self.src_end,
            src_strand=new_strand,
        )

    def subslice(self, rel_start: int, rel_end: int) -> StepProvenance:
        """Return the 0-based source interval corresponding to [rel_start, rel_end) of this step."""
        cur_len = self.src_end - self.src_start
        if rel_start < 0 or rel_end > cur_len or rel_start >= rel_end:
            raise ValueError(
                f"Invalid provenance subslice [{rel_start}, {rel_end}) for length {cur_len}"
            )
        if self.src_strand == "+":
            new_start = self.src_start + rel_start
            new_end = self.src_start + rel_end
        else:
            new_start = self.src_end - rel_end
            new_end = self.src_end - rel_start
        return StepProvenance(
            src_hap=self.src_hap,
            src_path=self.src_path,
            src_start=new_start,
            src_end=new_end,
            src_strand=self.src_strand,
        )


@dataclass(frozen=True)
class PathFragment:
    """An independent contiguous GFA P path or W walk segment."""

    sample: str
    hap: str
    hap_id: str
    seqid: str
    src_path: str
    base_offset: int
    steps: tuple[Step, ...] | Any
    provenance: tuple[StepProvenance, ...] | Any
    is_w_line: bool
    total_len: int
    is_flipped_contig: bool = False


@dataclass(frozen=True)
class GFAGraph:
    """Validated in-memory GFA v1 graph and indexed reference backbone."""

    gfa_path: str
    gfa_sha256: str
    ref_chrom: str
    ref_seq: str
    nodes: dict[str, str] | Any
    links: set[tuple[str, str, str, str]] | Any
    ref_fragment: PathFragment
    alt_fragments: tuple[PathFragment, ...]
    all_haplotypes: tuple[str, ...]
    ref_step_ranges: tuple[tuple[int, int], ...] | Any
    global_ref_nodes: frozenset[str] | Any
    unique_ref_step_by_node: dict[str, int] | Any
    _node_lens: Any = None
    _ref_cum_offsets: Any = None
    _nid_to_ref_rank: Any = None
    _nid_to_ref_sign: Any = None
    _global_ref_mask: Any = None
    _ref_node_counts: Any = None


@dataclass(frozen=True, slots=True)
class Piece:
    """A contiguous source interval emitted into pan.fa and recorded in inserts.tsv."""

    src_hap: str
    src_path: str
    src_start: int
    src_end: int
    src_strand: Literal["+", "-"]
    seq: str
    join_before: Literal["source_contiguous", "spacer", "synthetic"]
    members: tuple[str, ...]
    allele_idx: int = 1
    piece_idx: int = 1
    alleles: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        return self.src_end - self.src_start

    @property
    def src_intervals_str(self) -> str:
        return f"{self.src_start}-{self.src_end}"

    @property
    def allele_str(self) -> str:
        if self.alleles:
            return ",".join(str(a) for a in self.alleles)
        return str(self.allele_idx)


@dataclass(frozen=True, slots=True)
class UnresolvedRecord:
    """A contiguous source interval not placed into pan.fa, recorded in unresolved.tsv."""

    record_id: str
    region_id: str
    event_id: str
    src_path: str
    src_start: int
    src_end: int
    src_strand: Literal["+", "-"]
    length: int
    reason: str
    seq: str
    export_to_fasta: bool
    exported_fasta_id: str = "."


@dataclass(frozen=True, slots=True)
class SVEvent:
    """A top-level structural variant or CEN_ISLAND event recorded in sv.tsv and genotypes.tsv."""

    chrom: str
    pan_start: int
    pan_end: int
    event_id: str
    event_type: str
    ref_start: int
    ref_end: int
    anchors: str
    alleles_str: str
    n_carrier: int
    n_called: int
    region_id: str
    genotypes: dict[str, str]
    pieces: tuple[Piece, ...] = ()


@dataclass(frozen=True, slots=True)
class InsertionSlotBlock:
    """An insertion block mounted before right anchor reference step `right_anchor_ref_step`."""

    right_anchor_ref_step: int
    is_centromere_island: bool
    group_span: int
    left_anchor_ref_step: int
    event_id: str
    representatives: tuple[tuple[Piece, ...], ...]
    trailing_spacer: bool = False


@dataclass(frozen=True, slots=True)
class CentromereRegion:
    """A reference-delimited centromere region expanded to unique flanking anchors (u, v)."""

    region_id: str
    chrom: str
    input_bed_ids: tuple[str, ...]
    bed_start: int
    bed_end: int
    ref_start: int
    ref_end: int
    anchor_u_step: int | None
    anchor_v_step: int | None
    anchor_u_name: str
    anchor_v_name: str
    effective_mode: Literal["bypass", "extract-islands"]
    anchor_not_found: bool
    expanded_to_whole_chromosome: bool


@dataclass(frozen=True, slots=True)
class CentromereRow:
    """A 27-column row in `{p}.centromere.tsv` (§5.4)."""

    chrom: str
    region_id: str
    input_bed_ids: str
    bed_start: int
    bed_end: int
    ref_start: int
    ref_end: int
    anchor_u: str
    anchor_v: str
    mode: str
    hap: str
    status: str
    ref_status: str
    src_path: str
    src_start: int | str
    src_end: int | str
    src_strand: str
    L_ref: int
    L_path: int | str
    length_ratio: str
    nonref_input_bp: int | str
    reference_like_bp: int | str
    repeat_like_bp: int | str
    short_bp: int | str
    unresolved_bp: int | str
    accepted_source_bp: int | str
    supported_events: str

    def to_tsv_fields(self) -> list[str]:
        return [str(getattr(self, col)) for col in CENTROMERE_TSV_COLUMNS]


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for `panflat build` (§6)."""

    gfa: str
    ref_fa: str
    backbone_prefix: str
    out_prefix: str
    ref_gff: str | None = None
    min_sv_len: int = 50
    max_jump: int = 500000
    cluster_id: float = 0.95
    cluster_win_id: float = 0.70
    sec_backbone: Literal["support", "longest"] = "support"
    max_copy: int = 1
    spacer_n: int = 100
    splice_window: int = 8
    threads: int = 1
    repeat_guard: Literal["on", "off"] = "on"
    centromere_bed: str | None = None
    centromere_mode: Literal["off", "bypass", "extract-islands"] = "off"
    centromere_min_island: int = 500
    dump_repeat_sequences: bool = False
    qc: bool = False
    centromere_mode_explicit: bool = False
    centromere_min_island_explicit: bool = False

    @property
    def effective_centromere_mode(self) -> Literal["off", "bypass", "extract-islands"]:
        """Resolve effective centromere mode (BED without explicit mode defaults to 'bypass')."""
        if (
            self.centromere_bed is not None
            and self.centromere_mode == "off"
            and not self.centromere_mode_explicit
        ):
            return "bypass"
        return self.centromere_mode


# Downstream M2-M4 helper dataclasses


@dataclass(frozen=True, slots=True)
class ProjectedAnchor:
    """A branch step projected onto a unique anchor in backbone B[lo:hi) (§3.2)."""

    branch_step_idx: int
    rank_in_b: int
    rel_orient: Literal["+", "-"]
    node_id: str
    weight: int
    step: Step
    provenance: StepProvenance


@dataclass(frozen=True, slots=True)
class ReverseChainRecord:
    """A maximal decreasing relative-reverse anchor run R (§3.3.1-3.3.2)."""

    chain_idx: int
    branch_start_idx: int
    branch_end_idx: int
    rmin: int
    rmax: int
    left_flank_rank: int | None
    right_flank_rank: int | None
    classification: Literal["INV", "INV_DISPLACED", "INV_EDGE", "INV_CONFLICT"]
    ref_start: int
    ref_end: int
    normalized: bool
    novel_unresolved_slices: tuple[tuple[tuple[Step, ...], tuple[StepProvenance, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class InversionNormalizationResult:
    """Result of top-level inversion detection and normalization on a PathFragment (§3.3)."""

    normalized_fragment: PathFragment
    chains: tuple[ReverseChainRecord, ...]
    unresolved_records: tuple[UnresolvedRecord, ...]
    unnormalized_branch_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class DeviationSegment:
    """A branch deviation D(a, b) between adjacent main-chain anchors (a, b) (§3.4.2)."""

    hap_id: str
    src_path: str
    left_anchor_rank: int | Literal["START"]
    right_anchor_rank: int | Literal["END"]
    branch_start_idx: int
    branch_end_idx: int
    branch_steps: tuple[Step, ...] | Any
    branch_provenance: tuple[StepProvenance, ...] | Any
    backbone_steps: tuple[Step, ...] | Any
    multi_hit_nodes: tuple[str, ...] = ()
    is_unanchored_start: bool = False
    is_unanchored_end: bool = False

    def detached(self, keep_provenance: bool = True) -> DeviationSegment:
        """Return a copy whose step and provenance slices do not pin full-contig parent arrays."""
        bs = (
            self.branch_steps.detach()
            if isinstance(self.branch_steps, LazyStepTuple)
            else self.branch_steps
        )
        if not keep_provenance:
            bp: Any = ()
        elif isinstance(self.branch_provenance, LazyProvenanceTuple):
            bp = self.branch_provenance.detach()
        else:
            bp = self.branch_provenance
        return DeviationSegment(
            hap_id=self.hap_id,
            src_path=self.src_path,
            left_anchor_rank=self.left_anchor_rank,
            right_anchor_rank=self.right_anchor_rank,
            branch_start_idx=self.branch_start_idx,
            branch_end_idx=self.branch_end_idx,
            branch_steps=bs,
            branch_provenance=bp,
            backbone_steps=(),
            multi_hit_nodes=self.multi_hit_nodes,
            is_unanchored_start=self.is_unanchored_start,
            is_unanchored_end=self.is_unanchored_end,
        )


@dataclass(frozen=True, slots=True)
class CycleFoldResult:
    """Result of greedy tandem repeat cycle folding on non-reference steps (§3.4.3)."""

    folded_steps: tuple[Step, ...]
    folded_provenance: tuple[StepProvenance, ...]
    piece_boundaries: tuple[int, ...]
    deleted_intervals: tuple[tuple[StepProvenance, ...], ...]


@dataclass(frozen=True, slots=True)
class CNVCall:
    """Pre-folding canonical period CNV call between B(a, b) and branch deviation (§3.4.4)."""

    unit_tokens: tuple[tuple[str, str], ...]
    unit_bp: int
    k_ref: int
    k_alt: int
    delta_bp: int


@dataclass(frozen=True)
class CentromereHapSlice:
    """Per-region, per-haplotype centromere traversal status and inner slice (§3.8.2)."""

    region_id: str
    hap_id: str
    status: Literal["complete", "gapped", "ambiguous", "partial", "unmapped"]
    ref_status: Literal["complete", "gapped"]
    fragment: PathFragment | None
    u_step_idx: int | None
    v_step_idx: int | None
    inner_steps: tuple[Step, ...] | Any
    inner_provenance: tuple[StepProvenance, ...] | Any
    src_path: str
    src_start: int | None
    src_end: int | None
    src_strand: Literal["+", "-"] | None
    l_ref: int
    l_path: int | None
    length_ratio: float | None
    extra_inner_slices: tuple[tuple[tuple[Step, ...], tuple[StepProvenance, ...]], ...] = ()


@dataclass(frozen=True)
class CentromerePartitionResult:
    """Result of partitioning oriented fragments at centromere region anchors (u, v) (§3.8.1)."""

    regions: tuple[CentromereRegion, ...]
    hap_slices: dict[tuple[str, str], CentromereHapSlice]
    normal_fragments: tuple[PathFragment, ...]
    claimed_source_intervals: set[tuple[str, int, int]]
    boundary_ambiguous_records: tuple[UnresolvedRecord, ...] = ()


@dataclass(frozen=True)
class CentromereOutput:
    """Outputs produced by centromere processing across all regions (§3.8.2-3.8.5)."""

    centromere_rows: tuple[CentromereRow, ...]
    island_slot_blocks: tuple[InsertionSlotBlock, ...]
    island_sv_events: tuple[SVEvent, ...]
    unresolved_records: tuple[UnresolvedRecord, ...]
    json_log_entries: tuple[dict[str, Any], ...]
    summary_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepOffsetRow:
    """A row in `{p}.offsets.tsv` mapping reference step coordinates to pan.fa (§5)."""

    chrom: str
    ref_step: int
    node_id: str
    ref_start: int
    ref_end: int
    pan_start: int
    pan_end: int
    slot_ins_len: int


@dataclass(frozen=True)
class ChromosomeBuildResult:
    """Complete in-memory output of `run_chromosome_pipeline` for a single chromosome."""

    chrom: str
    pan_seq: str
    step_offsets: tuple[StepOffsetRow, ...]
    sv_events: tuple[SVEvent, ...]
    all_haplotypes: tuple[str, ...]
    inserts_rows: tuple[dict[str, Any], ...]
    unresolved_records: tuple[UnresolvedRecord, ...]
    centromere_rows: tuple[CentromereRow, ...]
    lifted_gff_lines: tuple[str, ...]
    summary_metrics: dict[str, Any]
    log_lines: tuple[str, ...]
    json_log_entries: tuple[dict[str, Any], ...]
    gfa_path: str
    gfa_sha256: str
    ref_fa_path: str


@dataclass(frozen=True)
class VerificationReport:
    """Verification report from `verify_build_outputs` (§1 & §3.8.5)."""

    passed: bool
    invariant_1_passed: bool
    invariant_2_passed: bool
    invariant_3_passed: bool
    centromere_passed: bool = True
    unresolved_passed: bool = True
    errors: tuple[str, ...] = ()
    checked_ref_steps: int = 0
    checked_pieces: int = 0
    checked_chain_blocks: int = 0
    checked_unresolved: int = 0


# ============================================================================
# Compact Zero-Copy NumPy-Backed Containers for Chromosome-Scale PGGB Graphs
# ============================================================================


class CompactNodeStore(collections.abc.Mapping[str, str]):
    """
    Compact read-only Mapping[str, str] for GFA S-line node sequences.
    Stores sequential integer node IDs `1.._max_nid` in a flat ASCII byte buffer
    plus an offset array, with an `_extra` dict fallback for non-integer node IDs.
    """

    __slots__ = ("_seq_bytes", "_offsets", "_node_lens", "_max_nid", "_extra", "_count")

    def __init__(
        self,
        seq_bytes: bytes,
        offsets: np.ndarray,
        node_lens: np.ndarray,
        max_nid: int,
        extra: dict[str, str] | None = None,
        count: int | None = None,
    ) -> None:
        self._seq_bytes = seq_bytes
        self._offsets = offsets
        self._node_lens = node_lens
        self._max_nid = max_nid
        self._extra: dict[str, str] = extra or {}
        if count is not None:
            self._count = count
        else:
            int_cnt = int(np.count_nonzero(node_lens[1 : max_nid + 1] > 0)) if max_nid > 0 else 0
            self._count = int_cnt + len(self._extra)

    def __getitem__(self, key: str) -> str:
        if isinstance(key, str) and key.isdigit():
            nid = int(key)
            if 1 <= nid <= self._max_nid:
                s = int(self._offsets[nid])
                e = int(self._offsets[nid + 1])
                if e > s:
                    return self._seq_bytes[s:e].decode("ascii")
        return self._extra[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key.isdigit():
            nid = int(key)
            if 1 <= nid <= self._max_nid and int(self._node_lens[nid]) > 0:
                return True
        return key in self._extra

    @property
    def node_lens(self) -> np.ndarray:
        return self._node_lens

    def node_len(self, key: str) -> int:
        if key.isdigit():
            nid = int(key)
            if 1 <= nid <= self._max_nid:
                length = int(self._node_lens[nid])
                if length > 0:
                    return length
        return len(self._extra[key])

    def node_len_int(self, nid: int) -> int:
        return int(self._node_lens[nid])

    def get_seq_int(self, nid: int) -> str:
        s = int(self._offsets[nid])
        e = int(self._offsets[nid + 1])
        return self._seq_bytes[s:e].decode("ascii")

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[str]:
        for nid in range(1, self._max_nid + 1):
            if int(self._node_lens[nid]) > 0:
                yield str(nid)
        yield from self._extra


class CompactLinkSet(collections.abc.Set[tuple[str, str, str, str]]):
    """
    Compact read-only Set of bidirectional 0M GFA L-links `(u, uo, v, vo)`.
    Stores canonical undirected 64-bit integer edge codes in a sorted `np.uint64` array.
    """

    __slots__ = ("_canon_links", "_extra", "_first_v")

    def __init__(
        self,
        canon_links: np.ndarray,
        extra: set[tuple[str, str, str, str]] | None = None,
    ) -> None:
        self._canon_links = canon_links
        self._extra: set[tuple[str, str, str, str]] = extra or set()
        self._first_v: np.ndarray | None = None

    @staticmethod
    def encode_canon(u_nid: int, u_minus: int, v_nid: int, v_minus: int) -> int:
        uc = (u_nid << 1) | u_minus
        vc = (v_nid << 1) | v_minus
        fwd = (uc << 32) | vc
        rev = ((vc ^ 1) << 32) | (uc ^ 1)
        return fwd if fwd <= rev else rev

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, tuple) or len(item) != 4:
            return False
        u, uo, v, vo = item
        if (
            isinstance(u, str)
            and isinstance(v, str)
            and u.isdigit()
            and v.isdigit()
            and uo in ("+", "-")
            and vo in ("+", "-")
        ):
            key = np.uint64(
                self.encode_canon(
                    int(u),
                    0 if uo == "+" else 1,
                    int(v),
                    0 if vo == "+" else 1,
                )
            )
            arr = self._canon_links
            if arr.shape[0] > 0:
                idx = int(np.searchsorted(arr, key))
                if idx < arr.shape[0] and arr[idx] == key:
                    return True
        return item in self._extra

    def validate_step_codes(self, codes: np.ndarray) -> int:
        """
        Validate all adjacent transitions in a 1D signed `int32` step code array.
        Returns `-1` if all transitions exist in `_canon_links`, else the first failing index `>= 0`.
        """
        n = int(codes.shape[0])
        if n <= 1:
            return -1
        arr = self._canon_links
        n_links = int(arr.shape[0])
        if n_links == 0:
            return 0
        first_v = self._first_v
        if first_v is None and n_links > 100_000:
            u_keys = (arr >> np.uint64(32)).astype(np.int32)
            v_keys = (arr & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            max_uk = int(u_keys[-1])
            first_v = np.zeros(max_uk + 1, dtype=np.uint32)
            first_v[u_keys] = v_keys
            self._first_v = first_v

        chunk_size = 524_288
        for start in range(0, n - 1, chunk_size):
            end = min(n - 1, start + chunk_size)
            u_raw = codes[start:end]
            v_raw = codes[start + 1 : end + 1]
            u_nid = np.abs(u_raw).astype(np.uint64)
            v_nid = np.abs(v_raw).astype(np.uint64)
            u_minus = (u_raw < 0).astype(np.uint64)
            v_minus = (v_raw < 0).astype(np.uint64)
            uc = (u_nid << np.uint64(1)) | u_minus
            vc = (v_nid << np.uint64(1)) | v_minus
            fwd = (uc << np.uint64(32)) | vc
            rev = ((vc ^ np.uint64(1)) << np.uint64(32)) | (uc ^ np.uint64(1))
            canon = np.minimum(fwd, rev)
            if first_v is not None:
                max_uk = int(first_v.shape[0]) - 1
                uk = (canon >> np.uint64(32)).astype(np.int32)
                vk = (canon & np.uint64(0xFFFFFFFF)).astype(np.uint32)
                fast_ok = (uk <= max_uk) & (
                    first_v[np.minimum(uk, max_uk)] == vk
                )
                if bool(np.all(fast_ok)):
                    continue
                miss_idx = np.flatnonzero(~fast_ok)
                miss_canon = canon[miss_idx]
                pos = np.searchsorted(arr, miss_canon)
                valid = (pos < n_links) & (
                    arr[np.minimum(pos, n_links - 1)] == miss_canon
                )
                if not bool(np.all(valid)):
                    bad_rel = int(miss_idx[int(np.flatnonzero(~valid)[0])])
                    return start + bad_rel
            else:
                pos = np.searchsorted(arr, canon)
                valid = (pos < n_links) & (
                    arr[np.minimum(pos, n_links - 1)] == canon
                )
                if not bool(np.all(valid)):
                    bad_rel = int(np.flatnonzero(~valid)[0])
                    return start + bad_rel
        return -1

    def __len__(self) -> int:
        return int(self._canon_links.shape[0]) * 2 + len(self._extra)

    def __iter__(self) -> Iterator[tuple[str, str, str, str]]:
        for val in self._canon_links.tolist():
            uc = (val >> 32) & 0xFFFFFFFF
            vc = val & 0xFFFFFFFF
            u_id = str(uc >> 1)
            uo = "-" if (uc & 1) else "+"
            v_id = str(vc >> 1)
            vo = "-" if (vc & 1) else "+"
            yield (u_id, uo, v_id, vo)
            rev_vo = "+" if vo == "-" else "-"
            rev_uo = "+" if uo == "-" else "-"
            if (v_id, rev_vo, u_id, rev_uo) != (u_id, uo, v_id, vo):
                yield (v_id, rev_vo, u_id, rev_uo)
        yield from self._extra


_REVCOMP_ASCII_TABLE = str.maketrans(
    "ACGTNacgtnRYSWKMBDHVryswkmbdhv",
    "TGCANtgcanYRSWMKVHDByrswmkvhdb",
)


class LazyStepTuple(collections.abc.Sequence[Step]):
    """
    Zero-copy Sequence[Step] backed by a 1D signed `np.int32` array `_codes`
    (`+nid` -> `Step(str(nid), "+")`, `-nid` -> `Step(str(nid), "-")`).
    """

    __slots__ = ("_codes", "_anchor_cache")

    def __init__(self, codes: np.ndarray) -> None:
        self._codes = codes
        self._anchor_cache: Any = None

    def __len__(self) -> int:
        return int(self._codes.shape[0])

    def __bool__(self) -> bool:
        return self._codes.shape[0] > 0

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return LazyStepTuple(self._codes[idx])
        c = int(self._codes[idx])
        if c > 0:
            return Step(node_id=str(c), orient="+")
        return Step(node_id=str(-c), orient="-")

    def __iter__(self) -> Iterator[Step]:
        for c in self._codes.tolist():
            if c > 0:
                yield Step(node_id=str(c), orient="+")
            else:
                yield Step(node_id=str(-c), orient="-")

    def __reversed__(self) -> Iterator[Step]:
        for c in self._codes[::-1].tolist():
            if c > 0:
                yield Step(node_id=str(c), orient="+")
            else:
                yield Step(node_id=str(-c), orient="-")

    def flipped_all(self) -> LazyStepTuple:
        return LazyStepTuple(np.ascontiguousarray(-self._codes[::-1]))

    def detach(self) -> LazyStepTuple:
        """Return a compact LazyStepTuple whose `_codes` does not pin a larger parent buffer."""
        if int(self._codes.shape[0]) == 0:
            return _EMPTY_LAZY_STEPS
        if self._codes.base is not None:
            return LazyStepTuple(self._codes.copy())
        return self

    def spell(self, nodes: Mapping[str, str]) -> str:
        """Spell the nucleotide sequence of this step walk."""
        if isinstance(nodes, CompactNodeStore):
            seq_bytes = nodes._seq_bytes
            offsets = nodes._offsets
            node_lens = nodes._node_lens
            max_nid = nodes._max_nid
            parts_fast: list[str] = []
            for c in self._codes.tolist():
                nid = c if c > 0 else -c
                if nid < 1 or nid > max_nid or int(node_lens[nid]) <= 0:
                    raise InputContractError(f"Unknown node ID '{nid}' in step walk")
                s = int(offsets[nid])
                e = int(offsets[nid + 1])
                raw = seq_bytes[s:e].decode("ascii")
                parts_fast.append(
                    raw if c > 0 else raw.translate(_REVCOMP_ASCII_TABLE)[::-1]
                )
            return "".join(parts_fast)
        parts: list[str] = []
        for c in self._codes.tolist():
            nid_s = str(c if c > 0 else -c)
            raw = nodes[nid_s]
            parts.append(
                raw if c > 0 else raw.translate(_REVCOMP_ASCII_TABLE)[::-1]
            )
        return "".join(parts)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, LazyStepTuple):
            return bool(np.array_equal(self._codes, other._codes))
        if isinstance(other, (tuple, list)):
            if int(self._codes.shape[0]) != len(other):
                return False
            for c, st in zip(self._codes.tolist(), other):
                if (
                    not isinstance(st, Step)
                    or st.is_partial
                    or st.slice_start != 0
                    or st.slice_end is not None
                ):
                    return False
                nid = c if c > 0 else -c
                orient = "+" if c > 0 else "-"
                if st.orient != orient or st.node_id != str(nid):
                    return False
            return True
        return False

    def __hash__(self) -> int:
        return hash(self._codes.tobytes())


_EMPTY_LAZY_STEPS = LazyStepTuple(np.empty(0, dtype=np.int32))


class LazyProvenanceTuple(collections.abc.Sequence[StepProvenance]):
    """
    Zero-copy Sequence[StepProvenance] backed by shared cumulative node offsets,
    supporting O(1) slicing, whole-contig flipping, inversion interval flipping,
    and O(1) contiguous provenance merging.
    """

    __slots__ = (
        "src_hap",
        "src_path",
        "base_offset",
        "_orig_codes",
        "_node_lens",
        "_cum_cache",
        "_total_steps",
        "_slice_start",
        "_slice_end",
        "_is_contig_flipped",
        "_inv_intervals",
        "_inv_starts",
        "_total_bp",
    )

    def __init__(
        self,
        src_hap: str,
        src_path: str,
        base_offset: int,
        orig_codes: np.ndarray,
        node_lens: np.ndarray,
        cum_cache: list[np.ndarray | None] | None = None,
        slice_start: int = 0,
        slice_end: int | None = None,
        is_contig_flipped: bool = False,
        inv_intervals: tuple[tuple[int, int], ...] = (),
        inv_starts: tuple[int, ...] | None = None,
        total_bp: int | None = None,
    ) -> None:
        self.src_hap = src_hap
        self.src_path = src_path
        self.base_offset = base_offset
        self._orig_codes = orig_codes
        self._node_lens = node_lens
        self._cum_cache: list[np.ndarray | None] = (
            cum_cache if cum_cache is not None else [None]
        )
        self._total_steps = int(orig_codes.shape[0])
        self._slice_start = slice_start
        self._slice_end = self._total_steps if slice_end is None else slice_end
        self._is_contig_flipped = is_contig_flipped
        self._inv_intervals = inv_intervals
        self._inv_starts = (
            inv_starts
            if inv_starts is not None
            else tuple(iv[0] for iv in inv_intervals)
        )
        self._total_bp = total_bp

    def _get_cum(self) -> np.ndarray:
        cum = self._cum_cache[0]
        if cum is None:
            n = self._total_steps
            lens = self._node_lens[np.abs(self._orig_codes)]
            cum = np.empty(n + 1, dtype=np.int64)
            cum[0] = self.base_offset
            np.cumsum(lens, dtype=np.int64, out=cum[1:])
            if self.base_offset != 0:
                cum[1:] += self.base_offset
            self._cum_cache[0] = cum
        return cum

    def _map_abs_index(self, abs_idx: int) -> tuple[int, Literal["+", "-"]]:
        idx = abs_idx
        is_plus = True
        if self._inv_intervals:
            pos = bisect.bisect_right(self._inv_starts, idx) - 1
            if pos >= 0:
                iv_s, iv_e = self._inv_intervals[pos]
                if idx < iv_e:
                    idx = iv_s + iv_e - 1 - idx
                    is_plus = False
        if self._is_contig_flipped:
            idx = self._total_steps - 1 - idx
            is_plus = not is_plus
        return idx, ("+" if is_plus else "-")

    def __len__(self) -> int:
        return max(0, self._slice_end - self._slice_start)

    def __bool__(self) -> bool:
        return self._slice_end > self._slice_start

    def __getitem__(self, idx: int | slice) -> Any:
        n = self._slice_end - self._slice_start
        if isinstance(idx, slice):
            start, stop, step = idx.indices(n)
            if step == 1:
                new_s = self._slice_start + start
                new_e = self._slice_start + stop
                return LazyProvenanceTuple(
                    src_hap=self.src_hap,
                    src_path=self.src_path,
                    base_offset=self.base_offset,
                    orig_codes=self._orig_codes,
                    node_lens=self._node_lens,
                    cum_cache=self._cum_cache,
                    slice_start=new_s,
                    slice_end=new_e,
                    is_contig_flipped=self._is_contig_flipped,
                    inv_intervals=self._inv_intervals,
                    inv_starts=self._inv_starts,
                    total_bp=(
                        self._total_bp
                        if (new_s == 0 and new_e == self._total_steps)
                        else None
                    ),
                )
            return tuple(self[i] for i in range(start, stop, step))

        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("LazyProvenanceTuple index out of range")
        orig_i, strand = self._map_abs_index(self._slice_start + idx)
        if self._cum_cache[0] is None and self._total_steps > 0:
            if orig_i == 0:
                slen = int(self._node_lens[abs(int(self._orig_codes[0]))])
                return StepProvenance(
                    src_hap=self.src_hap,
                    src_path=self.src_path,
                    src_start=self.base_offset,
                    src_end=self.base_offset + slen,
                    src_strand=strand,
                )
            if orig_i == self._total_steps - 1 and self._total_bp is not None:
                slen = int(self._node_lens[abs(int(self._orig_codes[-1]))])
                end_pos = self.base_offset + self._total_bp
                return StepProvenance(
                    src_hap=self.src_hap,
                    src_path=self.src_path,
                    src_start=end_pos - slen,
                    src_end=end_pos,
                    src_strand=strand,
                )
        cum = self._get_cum()
        return StepProvenance(
            src_hap=self.src_hap,
            src_path=self.src_path,
            src_start=int(cum[orig_i]),
            src_end=int(cum[orig_i + 1]),
            src_strand=strand,
        )

    def __iter__(self) -> Iterator[StepProvenance]:
        n = len(self)
        for i in range(n):
            yield self[i]

    def __reversed__(self) -> Iterator[StepProvenance]:
        n = len(self)
        for i in range(n - 1, -1, -1):
            yield self[i]

    def flipped_contig(self) -> LazyProvenanceTuple:
        """Flip the entire slice as a whole-contig orientation flip."""
        new_s = self._total_steps - self._slice_end
        new_e = self._total_steps - self._slice_start
        return LazyProvenanceTuple(
            src_hap=self.src_hap,
            src_path=self.src_path,
            base_offset=self.base_offset,
            orig_codes=self._orig_codes,
            node_lens=self._node_lens,
            cum_cache=self._cum_cache,
            slice_start=new_s,
            slice_end=new_e,
            is_contig_flipped=not self._is_contig_flipped,
            inv_intervals=(),
            inv_starts=(),
            total_bp=(
                self._total_bp
                if (new_s == 0 and new_e == self._total_steps)
                else None
            ),
        )

    def with_flipped_intervals(
        self,
        rel_intervals: Sequence[tuple[int, int]],
    ) -> LazyProvenanceTuple:
        """Return a view with disjoint relative sub-intervals `[s, e)` flipped by inversion normalization."""
        if not rel_intervals:
            return self
        abs_ivs = tuple(
            sorted((self._slice_start + s, self._slice_start + e) for s, e in rel_intervals if s < e)
        )
        return LazyProvenanceTuple(
            src_hap=self.src_hap,
            src_path=self.src_path,
            base_offset=self.base_offset,
            orig_codes=self._orig_codes,
            node_lens=self._node_lens,
            cum_cache=self._cum_cache,
            slice_start=self._slice_start,
            slice_end=self._slice_end,
            is_contig_flipped=self._is_contig_flipped,
            inv_intervals=abs_ivs,
            inv_starts=tuple(iv[0] for iv in abs_ivs),
            total_bp=self._total_bp,
        )

    def detach(self) -> Any:
        """Return a compact provenance sequence detached from any full-contig parent arrays."""
        s_lo = self._slice_start
        s_hi = self._slice_end
        m = s_hi - s_lo
        if m <= 0:
            return ()
        if (
            m == self._total_steps
            and self._orig_codes.base is None
            and self._cum_cache[0] is None
        ):
            return self
        if not self.has_internal_discontinuity():
            cum = self._get_cum()
            orig_first, strand = self._map_abs_index(s_lo)
            orig_last, _ = self._map_abs_index(s_hi - 1)
            if strand == "+":
                new_base = int(cum[orig_first])
                bp_len = int(cum[orig_last + 1] - cum[orig_first])
                sub_codes = self._orig_codes[orig_first : orig_last + 1].copy()
                return LazyProvenanceTuple(
                    src_hap=self.src_hap,
                    src_path=self.src_path,
                    base_offset=new_base,
                    orig_codes=sub_codes,
                    node_lens=self._node_lens,
                    cum_cache=[None],
                    slice_start=0,
                    slice_end=m,
                    is_contig_flipped=False,
                    inv_intervals=(),
                    inv_starts=(),
                    total_bp=bp_len,
                )
            new_base = int(cum[orig_last])
            bp_len = int(cum[orig_first + 1] - cum[orig_last])
            sub_codes = self._orig_codes[orig_last : orig_first + 1].copy()
            return LazyProvenanceTuple(
                src_hap=self.src_hap,
                src_path=self.src_path,
                base_offset=new_base,
                orig_codes=sub_codes,
                node_lens=self._node_lens,
                cum_cache=[None],
                slice_start=0,
                slice_end=m,
                is_contig_flipped=True,
                inv_intervals=(),
                inv_starts=(),
                total_bp=bp_len,
            )
        return tuple(self[i] for i in range(m))

    def merged_runs(self) -> list[tuple[StepProvenance, int, int]]:
        """
        Return maximal contiguous `(StepProvenance, rel_bp_start, rel_bp_end)` runs across `self`
        in O(|_inv_intervals|) time without iterating over individual steps.
        """
        s_lo = self._slice_start
        s_hi = self._slice_end
        if s_lo >= s_hi:
            return []

        if (
            s_lo == 0
            and s_hi == self._total_steps
            and not self._inv_intervals
            and self._total_bp is not None
        ):
            strand_full: Literal["+", "-"] = (
                "-" if self._is_contig_flipped else "+"
            )
            prov_full = StepProvenance(
                src_hap=self.src_hap,
                src_path=self.src_path,
                src_start=self.base_offset,
                src_end=self.base_offset + self._total_bp,
                src_strand=strand_full,
            )
            return [(prov_full, 0, self._total_bp)]

        split_pts: list[int] = [s_lo]
        if self._inv_intervals:
            for iv_s, iv_e in self._inv_intervals:
                if s_lo < iv_s < s_hi:
                    split_pts.append(iv_s)
                if s_lo < iv_e < s_hi:
                    split_pts.append(iv_e)
        split_pts.append(s_hi)
        if len(split_pts) > 2:
            split_pts = sorted(set(split_pts))

        cum = self._get_cum()
        runs: list[tuple[StepProvenance, int, int]] = []
        bp_cursor = 0
        for idx in range(len(split_pts) - 1):
            a = split_pts[idx]
            b = split_pts[idx + 1]
            if a >= b:
                continue
            orig_first, strand = self._map_abs_index(a)
            orig_last, _ = self._map_abs_index(b - 1)
            if strand == "+":
                src_s = int(cum[orig_first])
                src_e = int(cum[orig_last + 1])
            else:
                src_s = int(cum[orig_last])
                src_e = int(cum[orig_first + 1])
            run_len = src_e - src_s
            prov = StepProvenance(
                src_hap=self.src_hap,
                src_path=self.src_path,
                src_start=src_s,
                src_end=src_e,
                src_strand=strand,
            )
            runs.append((prov, bp_cursor, bp_cursor + run_len))
            bp_cursor += run_len
        return runs

    def total_bp(self) -> int:
        s_lo = self._slice_start
        s_hi = self._slice_end
        if s_lo >= s_hi:
            return 0
        if s_lo == 0 and s_hi == self._total_steps and self._total_bp is not None:
            return self._total_bp
        if self._cum_cache[0] is None and not self.has_internal_discontinuity():
            orig_first, strand = self._map_abs_index(s_lo)
            orig_last, _ = self._map_abs_index(s_hi - 1)
            lo_i = orig_first if strand == "+" else orig_last
            hi_i = (orig_last if strand == "+" else orig_first) + 1
            return int(self._node_lens[np.abs(self._orig_codes[lo_i:hi_i])].sum())
        runs = self.merged_runs()
        return runs[-1][2] if runs else 0

    def min_src_start(self) -> int:
        if (
            self._slice_start == 0
            and self._slice_end == self._total_steps
            and self._total_bp is not None
        ):
            return self.base_offset
        runs = self.merged_runs()
        if not runs:
            return self.base_offset
        return min(r[0].src_start for r in runs)

    def max_src_end(self) -> int:
        if (
            self._slice_start == 0
            and self._slice_end == self._total_steps
            and self._total_bp is not None
        ):
            return self.base_offset + self._total_bp
        runs = self.merged_runs()
        if not runs:
            return self.base_offset
        return max(r[0].src_end for r in runs)

    def has_internal_discontinuity(self) -> bool:
        if not self._inv_intervals:
            return False
        s_lo = self._slice_start
        s_hi = self._slice_end
        for iv_s, iv_e in self._inv_intervals:
            if s_lo < iv_s < s_hi or s_lo < iv_e < s_hi:
                return True
        return False

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, (Sequence, tuple, list)):
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        return False


class LazyRefStepRanges(collections.abc.Sequence[tuple[int, int]]):
    """Compact Sequence[tuple[int, int]] backed by a 1D `int64` cumulative offset array."""

    __slots__ = ("_cum_offsets",)

    def __init__(self, cum_offsets: np.ndarray) -> None:
        self._cum_offsets = cum_offsets

    def __len__(self) -> int:
        return max(0, int(self._cum_offsets.shape[0]) - 1)

    def __getitem__(self, idx: int | slice) -> Any:
        n = int(self._cum_offsets.shape[0]) - 1
        if isinstance(idx, slice):
            return tuple(
                (int(self._cum_offsets[i]), int(self._cum_offsets[i + 1]))
                for i in range(*idx.indices(n))
            )
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("LazyRefStepRanges index out of range")
        return (int(self._cum_offsets[idx]), int(self._cum_offsets[idx + 1]))

    def __iter__(self) -> Iterator[tuple[int, int]]:
        starts = self._cum_offsets[:-1].tolist()
        ends = self._cum_offsets[1:].tolist()
        return iter(zip(starts, ends))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, LazyRefStepRanges):
            return bool(np.array_equal(self._cum_offsets, other._cum_offsets))
        if isinstance(other, (tuple, list)):
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        return False


class CompactRefNodeSet(collections.abc.Set[str]):
    """Compact Set[str] of reference node IDs backed by a boolean mask array."""

    __slots__ = ("_mask", "_extra", "_count")

    def __init__(
        self,
        mask: np.ndarray,
        extra: frozenset[str] = frozenset(),
        count: int | None = None,
    ) -> None:
        self._mask = mask
        self._extra = extra
        self._count = (
            count
            if count is not None
            else int(np.count_nonzero(mask)) + len(extra)
        )

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key.isdigit():
            nid = int(key)
            if 1 <= nid < int(self._mask.shape[0]):
                return bool(self._mask[nid])
        return key in self._extra

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[str]:
        for nid in np.flatnonzero(self._mask).tolist():
            yield str(nid)
        yield from self._extra


class CompactUniqueRefMap(collections.abc.Mapping[str, int]):
    """Compact Mapping[str, int] of unique reference node ID -> ref_step_idx."""

    __slots__ = ("_nid_to_rank", "_extra", "_count")

    def __init__(
        self,
        nid_to_rank: np.ndarray,
        extra: dict[str, int] | None = None,
        count: int | None = None,
    ) -> None:
        self._nid_to_rank = nid_to_rank
        self._extra: dict[str, int] = extra or {}
        self._count = (
            count
            if count is not None
            else int(np.count_nonzero(nid_to_rank >= 0)) + len(self._extra)
        )

    def __getitem__(self, key: str) -> int:
        if isinstance(key, str) and key.isdigit():
            nid = int(key)
            if 1 <= nid < int(self._nid_to_rank.shape[0]):
                r = int(self._nid_to_rank[nid])
                if r >= 0:
                    return r
        return self._extra[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key.isdigit():
            nid = int(key)
            if 1 <= nid < int(self._nid_to_rank.shape[0]):
                return int(self._nid_to_rank[nid]) >= 0
        return key in self._extra

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[str]:
        for nid in np.flatnonzero(self._nid_to_rank >= 0).tolist():
            yield str(nid)
        yield from self._extra


class LazyProjectedAnchorSequence(collections.abc.Sequence[ProjectedAnchor]):
    """
    Compact Sequence[ProjectedAnchor] backed by 1D NumPy arrays (`branch_step_idx`,
    `rank_in_b`, `rel_sign`, `weights`), constructing `ProjectedAnchor` objects on demand.
    """

    __slots__ = (
        "branch_step_idx",
        "rank_in_b",
        "rel_sign",
        "weights",
        "steps",
        "provenance",
        "multi_hit_nids",
        "has_multi_hit",
    )

    def __init__(
        self,
        branch_step_idx: np.ndarray,
        rank_in_b: np.ndarray,
        rel_sign: np.ndarray,
        weights: np.ndarray,
        steps: Sequence[Step],
        provenance: Sequence[StepProvenance],
        multi_hit_nids: tuple[str, ...] = (),
        has_multi_hit: bool = False,
    ) -> None:
        self.branch_step_idx = branch_step_idx
        self.rank_in_b = rank_in_b
        self.rel_sign = rel_sign
        self.weights = weights
        self.steps = steps
        self.provenance = provenance
        self.multi_hit_nids = multi_hit_nids
        self.has_multi_hit = has_multi_hit

    def __len__(self) -> int:
        return int(self.branch_step_idx.shape[0])

    def __bool__(self) -> bool:
        return self.branch_step_idx.shape[0] > 0

    def __getitem__(self, idx: int | slice) -> Any:
        n = int(self.branch_step_idx.shape[0])
        if isinstance(idx, slice):
            return LazyProjectedAnchorSequence(
                branch_step_idx=self.branch_step_idx[idx],
                rank_in_b=self.rank_in_b[idx],
                rel_sign=self.rel_sign[idx],
                weights=self.weights[idx],
                steps=self.steps,
                provenance=self.provenance,
                multi_hit_nids=self.multi_hit_nids,
                has_multi_hit=self.has_multi_hit,
            )
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("LazyProjectedAnchorSequence index out of range")
        b_idx = int(self.branch_step_idx[idx])
        r_idx = int(self.rank_in_b[idx])
        rsign = int(self.rel_sign[idx])
        w = int(self.weights[idx])
        st = self.steps[b_idx]
        pv = self.provenance[b_idx]
        return ProjectedAnchor(
            branch_step_idx=b_idx,
            rank_in_b=r_idx,
            rel_orient="+" if rsign > 0 else "-",
            node_id=st.node_id,
            weight=w,
            step=st,
            provenance=pv,
        )

    def subseq_by_indices(self, indices: np.ndarray) -> LazyProjectedAnchorSequence:
        return LazyProjectedAnchorSequence(
            branch_step_idx=self.branch_step_idx[indices],
            rank_in_b=self.rank_in_b[indices],
            rel_sign=self.rel_sign[indices],
            weights=self.weights[indices],
            steps=self.steps,
            provenance=self.provenance,
            multi_hit_nids=self.multi_hit_nids,
            has_multi_hit=self.has_multi_hit,
        )

    def __iter__(self) -> Iterator[ProjectedAnchor]:
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, (Sequence, list, tuple)):
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        return False


class CompactVisitCounts(collections.abc.Mapping[str, int]):
    """
    Compact Mapping[str, int] for per-contig node visit counts on chromosome-scale paths.
    Stores sorted `int32` arrays of multi-visited node IDs (`count >= 2`) and their counts.
    """

    __slots__ = ("_multi_nids", "_multi_counts", "_multi_set")

    def __init__(self, multi_nids: np.ndarray, multi_counts: np.ndarray) -> None:
        self._multi_nids = multi_nids
        self._multi_counts = multi_counts
        self._multi_set: frozenset[int] = frozenset(multi_nids.tolist())

    def get(self, key: str, default: Any = 0) -> Any:
        if isinstance(key, str) and key.isdigit():
            nid = int(key)
            if nid not in self._multi_set:
                return default
            arr = self._multi_nids
            idx = int(np.searchsorted(arr, nid))
            return int(self._multi_counts[idx])
        return default

    def __getitem__(self, key: str) -> int:
        val = self.get(key, -1)
        if val < 0:
            raise KeyError(key)
        return int(val)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return self.get(key, -1) >= 0
        return False

    def __len__(self) -> int:
        return int(self._multi_nids.shape[0])

    def __iter__(self) -> Iterator[str]:
        for nid in self._multi_nids.tolist():
            yield str(nid)


class CompactRankSet:
    """Compact boolean-mask set of reference step ranks with O(1) min/max tracking."""

    __slots__ = ("_mask", "_count", "_min_val", "_max_val")

    def __init__(self, n_ref: int) -> None:
        self._mask = np.zeros(n_ref, dtype=bool)
        self._count = 0
        self._min_val = 10**9
        self._max_val = -1

    def add(self, r: int) -> None:
        ri = int(r)
        if 0 <= ri < int(self._mask.shape[0]):
            if not bool(self._mask[ri]):
                self._mask[ri] = True
                self._count += 1
            if ri < self._min_val:
                self._min_val = ri
            if ri > self._max_val:
                self._max_val = ri

    def add_range(self, r_start: int, r_end: int) -> None:
        s = max(0, int(r_start))
        e = min(int(self._mask.shape[0]), int(r_end))
        if s < e:
            self._mask[s:e] = True
            self._count = max(self._count, e - s)
            if s < self._min_val:
                self._min_val = s
            if e - 1 > self._max_val:
                self._max_val = e - 1

    def add_array(self, arr: np.ndarray) -> None:
        if int(arr.shape[0]) > 0:
            self._mask[arr] = True
            self._count = max(self._count, int(arr.shape[0]))
            lo = int(arr[0])
            hi = int(arr[-1])
            if lo < self._min_val:
                self._min_val = lo
            if hi > self._max_val:
                self._max_val = hi

    def __contains__(self, r: object) -> bool:
        if isinstance(r, (int, np.integer)):
            ri = int(r)
            if 0 <= ri < int(self._mask.shape[0]):
                return bool(self._mask[ri])
        return False

    def __bool__(self) -> bool:
        return self._count > 0

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[int]:
        for idx in np.flatnonzero(self._mask).tolist():
            yield int(idx)


class LazyStepOffsetSequence(collections.abc.Sequence[StepOffsetRow]):
    """
    Zero-copy Sequence[StepOffsetRow] backed by NumPy arrays for chromosome-scale
    `step_offsets.tsv.gz` emission and `verify` checks.
    """

    __slots__ = (
        "chrom",
        "ref_nids",
        "ref_starts",
        "ref_ends",
        "pan_starts",
        "pan_ends",
        "slot_ins_lens",
    )

    def __init__(
        self,
        chrom: str,
        ref_nids: np.ndarray,
        ref_starts: np.ndarray,
        ref_ends: np.ndarray,
        pan_starts: np.ndarray,
        pan_ends: np.ndarray,
        slot_ins_lens: np.ndarray,
    ) -> None:
        self.chrom = chrom
        self.ref_nids = ref_nids
        self.ref_starts = ref_starts
        self.ref_ends = ref_ends
        self.pan_starts = pan_starts
        self.pan_ends = pan_ends
        self.slot_ins_lens = slot_ins_lens

    def __len__(self) -> int:
        return int(self.ref_nids.shape[0])

    def __getitem__(self, idx: int | slice) -> Any:
        n = int(self.ref_nids.shape[0])
        if isinstance(idx, slice):
            s, e, step = idx.indices(n)
            return tuple(self[i] for i in range(s, e, step))
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("LazyStepOffsetSequence index out of range")
        return StepOffsetRow(
            chrom=self.chrom,
            ref_step=idx,
            node_id=str(int(self.ref_nids[idx])),
            ref_start=int(self.ref_starts[idx]),
            ref_end=int(self.ref_ends[idx]),
            pan_start=int(self.pan_starts[idx]),
            pan_end=int(self.pan_ends[idx]),
            slot_ins_len=int(self.slot_ins_lens[idx]),
        )

    def __iter__(self) -> Iterator[StepOffsetRow]:
        for i in range(len(self)):
            yield self[i]



