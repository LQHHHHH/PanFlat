"""Core immutable dataclasses and shared type contracts for PanFlat v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    steps: tuple[Step, ...]
    provenance: tuple[StepProvenance, ...]
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
    nodes: dict[str, str]
    links: set[tuple[str, str, str, str]]
    ref_fragment: PathFragment
    alt_fragments: tuple[PathFragment, ...]
    all_haplotypes: tuple[str, ...]
    ref_step_ranges: tuple[tuple[int, int], ...]
    global_ref_nodes: frozenset[str]
    unique_ref_step_by_node: dict[str, int]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class InsertionSlotBlock:
    """An insertion block mounted before right anchor reference step `right_anchor_ref_step`."""

    right_anchor_ref_step: int
    is_centromere_island: bool
    group_span: int
    left_anchor_ref_step: int
    event_id: str
    representatives: tuple[tuple[Piece, ...], ...]
    trailing_spacer: bool = False


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ProjectedAnchor:
    """A branch step projected onto a unique anchor in backbone B[lo:hi) (§3.2)."""

    branch_step_idx: int
    rank_in_b: int
    rel_orient: Literal["+", "-"]
    node_id: str
    weight: int
    step: Step
    provenance: StepProvenance


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class InversionNormalizationResult:
    """Result of top-level inversion detection and normalization on a PathFragment (§3.3)."""

    normalized_fragment: PathFragment
    chains: tuple[ReverseChainRecord, ...]
    unresolved_records: tuple[UnresolvedRecord, ...]
    unnormalized_branch_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class DeviationSegment:
    """A branch deviation D(a, b) between adjacent main-chain anchors (a, b) (§3.4.2)."""

    hap_id: str
    src_path: str
    left_anchor_rank: int | Literal["START"]
    right_anchor_rank: int | Literal["END"]
    branch_start_idx: int
    branch_end_idx: int
    branch_steps: tuple[Step, ...]
    branch_provenance: tuple[StepProvenance, ...]
    backbone_steps: tuple[Step, ...]
    multi_hit_nodes: tuple[str, ...] = ()
    is_unanchored_start: bool = False
    is_unanchored_end: bool = False


@dataclass(frozen=True)
class CycleFoldResult:
    """Result of greedy tandem repeat cycle folding on non-reference steps (§3.4.3)."""

    folded_steps: tuple[Step, ...]
    folded_provenance: tuple[StepProvenance, ...]
    piece_boundaries: tuple[int, ...]
    deleted_intervals: tuple[tuple[StepProvenance, ...], ...]


@dataclass(frozen=True)
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
    inner_steps: tuple[Step, ...]
    inner_provenance: tuple[StepProvenance, ...]
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


@dataclass(frozen=True)
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
