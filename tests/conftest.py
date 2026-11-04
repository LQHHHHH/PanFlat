"""Shared synthetic GFA v1, FASTA, BED, and nucleotide sequence builders for unit tests."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Callable, Mapping, Sequence

import pytest

_RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp_dna(seq: str) -> str:
    """Test-side DNA reverse complement preserving case and N."""
    return seq.translate(_RC_TABLE)[::-1]


def canonical_15mer_stats(seq: str) -> tuple[int, int, int, float | None]:
    """Return `(singletons, valid_positions, total_positions, singleton_ratio)` for a window."""
    s = seq.upper()
    total_pos = max(0, len(s) - 14)
    if total_pos == 0:
        return 0, 0, 0, None
    counts: dict[str, int] = {}
    valid_pos = 0
    for i in range(total_pos):
        kmer = s[i : i + 15]
        if all(c in "ACGT" for c in kmer):
            valid_pos += 1
            ck = min(kmer, revcomp_dna(kmer))
            counts[ck] = counts.get(ck, 0) + 1
    if valid_pos * 10 < total_pos * 9:
        return 0, valid_pos, total_pos, None
    singletons = sum(1 for v in counts.values() if v == 1)
    return singletons, valid_pos, total_pos, singletons / valid_pos


def make_complex_seq(length: int, seed: int = 42) -> str:
    """
    Generate a deterministic high-complexity ACGT sequence of `length` bp
    without homopolymers >= 3, guaranteed to have high canonical 15-mer singleton_ratio.
    """
    rng = random.Random(seed)
    chars: list[str] = []
    alphabet = "ACGT"
    while len(chars) < length:
        c = rng.choice(alphabet)
        if len(chars) >= 2 and chars[-1] == c and chars[-2] == c:
            continue
        chars.append(c)
    return "".join(chars)


def make_satellite_seq(
    unit_len: int = 20,
    total_len: int = 1500,
    seed: int = 100,
    divergence: float = 0.0,
) -> str:
    """
    Generate a tandem repeat sequence by repeating a `unit_len` monomer to `total_len` bp,
    with optional point mutation fraction `divergence`.
    """
    rng = random.Random(seed)
    unit = make_complex_seq(unit_len, seed=seed)
    reps = (total_len // unit_len) + 2
    raw = list((unit * reps)[:total_len])
    if divergence > 0.0:
        num_mut = int(round(total_len * divergence))
        positions = rng.sample(range(total_len), k=min(num_mut, total_len))
        for pos in positions:
            orig = raw[pos]
            alts = [b for b in "ACGT" if b != orig]
            raw[pos] = rng.choice(alts)
    return "".join(raw)


def make_window_with_exact_singleton_ratio(
    window_len: int,
    target_singletons: int,
    valid_positions: int | None = None,
    seed: int = 200,
) -> str:
    """
    Construct a sequence of exact length `window_len` (where `N_pos = window_len - 14`)
    having `valid_positions` (default `N_pos`) valid ACGT 15-mer positions and
    EXACTLY `target_singletons` singleton canonical 15-mers.
    """
    n_pos = window_len - 14
    if valid_positions is None:
        valid_positions = n_pos
    assert 0 < valid_positions <= n_pos
    assert 0 <= target_singletons <= valid_positions
    acgt_len = valid_positions + 14
    n_tail_len = window_len - acgt_len

    rng = random.Random(seed)
    if target_singletons == valid_positions:
        attempt_seed = seed
        while True:
            acgt_seq = make_complex_seq(acgt_len, seed=attempt_seed)
            seq = acgt_seq + ("N" * n_tail_len)
            sing, val, _, _ = canonical_15mer_stats(seq)
            if sing == target_singletons and val == valid_positions:
                return seq
            attempt_seed += 1

    assert valid_positions - target_singletons >= 2, (
        "Cannot have exactly 1 non-singleton 15-mer position; "
        "valid_positions - target_singletons must be 0 or >= 2"
    )
    while True:
        prefix: list[str] = []
        while len(prefix) < target_singletons:
            c = rng.choice("ACGT")
            if len(prefix) >= 2 and prefix[-1] == c and prefix[-2] == c:
                continue
            prefix.append(c)
        if target_singletons > 0 and prefix[-1] in ("A", "T"):
            prefix[-1] = "C"
        acgt_seq = "".join(prefix) + ("A" * (acgt_len - target_singletons))
        seq = acgt_seq + ("N" * n_tail_len)

        s_up = seq.upper()
        counts: dict[str, int] = {}
        val = 0
        for i in range(n_pos):
            kmer = s_up[i : i + 15]
            if all(c in "ACGT" for c in kmer):
                val += 1
                ck = min(kmer, revcomp_dna(kmer))
                counts[ck] = counts.get(ck, 0) + 1
        sing = sum(1 for v in counts.values() if v == 1)
        if val == valid_positions and sing == target_singletons:
            return seq


def mutate_seq_snps(seq: str, snp_positions: Sequence[int]) -> str:
    """Deterministically substitute bases at `snp_positions` (A->C, C->G, G->T, T->A)."""
    sub_map = {
        "A": "C",
        "C": "G",
        "G": "T",
        "T": "A",
        "a": "c",
        "c": "g",
        "g": "t",
        "t": "a",
    }
    arr = list(seq)
    for pos in snp_positions:
        arr[pos] = sub_map.get(arr[pos], "C")
    return "".join(arr)


class SyntheticGFAFixtureBuilder:
    """Fluent builder for synthetic GFA v1, FASTA, and BED test files."""

    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}
        self.s_lines: list[str] = []
        self.l_lines: list[str] = []
        self.p_lines: list[str] = []
        self.w_lines: list[str] = []
        self.j_lines: list[str] = []
        self.raw_lines: list[str] = []
        self._recorded_links: set[tuple[str, str, str, str, str]] = set()

    def add_node(self, node_id: str, seq: str) -> SyntheticGFAFixtureBuilder:
        self.nodes[str(node_id)] = seq
        self.s_lines.append(f"S\t{node_id}\t{seq}")
        return self

    def add_link(
        self,
        from_id: str,
        from_orient: str,
        to_id: str,
        to_orient: str,
        overlap: str = "0M",
    ) -> SyntheticGFAFixtureBuilder:
        key = (str(from_id), from_orient, str(to_id), to_orient, overlap)
        if key not in self._recorded_links:
            self._recorded_links.add(key)
            self.l_lines.append(
                f"L\t{from_id}\t{from_orient}\t{to_id}\t{to_orient}\t{overlap}"
            )
        return self

    def add_auto_links_for_steps(
        self,
        steps: Sequence[tuple[str, str]],
        overlap: str = "0M",
    ) -> SyntheticGFAFixtureBuilder:
        for (u, uo), (v, vo) in zip(steps[:-1], steps[1:]):
            self.add_link(u, uo, v, vo, overlap=overlap)
        return self

    def add_p_path(
        self,
        path_name: str,
        steps: Sequence[tuple[str, str]],
        overlaps: str | Sequence[str] = "*",
        auto_links: bool = True,
    ) -> SyntheticGFAFixtureBuilder:
        if auto_links:
            self.add_auto_links_for_steps(steps, overlap="0M")
        step_str = ",".join(f"{nid}{orient}" for nid, orient in steps)
        if isinstance(overlaps, str):
            ov_str = overlaps
        else:
            ov_str = ",".join(overlaps)
        self.p_lines.append(f"P\t{path_name}\t{step_str}\t{ov_str}")
        return self

    def add_w_walk(
        self,
        sample: str,
        hap: int | str,
        seqid: str,
        start: int | str,
        end: int | str | None,
        steps: Sequence[tuple[str, str]],
        auto_links: bool = True,
    ) -> SyntheticGFAFixtureBuilder:
        if auto_links:
            self.add_auto_links_for_steps(steps, overlap="0M")
        if end is None and isinstance(start, int):
            end = start + len(self.spell_steps(steps))
        walk_str = "".join(f"{'>' if o == '+' else '<'}{nid}" for nid, o in steps)
        self.w_lines.append(f"W\t{sample}\t{hap}\t{seqid}\t{start}\t{end}\t{walk_str}")
        return self

    def add_j_line(
        self,
        from_id: str,
        from_orient: str,
        to_id: str,
        to_orient: str,
        distance: str = "*",
    ) -> SyntheticGFAFixtureBuilder:
        self.j_lines.append(
            f"J\t{from_id}\t{from_orient}\t{to_id}\t{to_orient}\t{distance}"
        )
        return self

    def spell_steps(self, steps: Sequence[tuple[str, str]]) -> str:
        out: list[str] = []
        for nid, orient in steps:
            seq = self.nodes[str(nid)]
            out.append(seq if orient == "+" else revcomp_dna(seq))
        return "".join(out)

    def build_gfa_text(self) -> str:
        lines = ["H\tVN:Z:1.0"]
        lines.extend(self.s_lines)
        lines.extend(self.l_lines)
        lines.extend(self.j_lines)
        lines.extend(self.p_lines)
        lines.extend(self.w_lines)
        lines.extend(self.raw_lines)
        return "\n".join(lines) + "\n"

    def write_gfa(self, path: Path) -> Path:
        path.write_text(self.build_gfa_text(), encoding="utf-8")
        return path

    @staticmethod
    def write_fasta(path: Path, records: Mapping[str, str]) -> Path:
        lines: list[str] = []
        for name, seq in records.items():
            lines.append(f">{name}")
            for i in range(0, len(seq), 80):
                lines.append(seq[i : i + 80])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def write_bed(
        path: Path,
        rows: Sequence[tuple[str, int, int] | tuple[str, int, int, str]],
    ) -> Path:
        lines = ["\t".join(str(x) for x in row) for row in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path


@pytest.fixture
def gfa_builder() -> Callable[[], SyntheticGFAFixtureBuilder]:
    return SyntheticGFAFixtureBuilder
