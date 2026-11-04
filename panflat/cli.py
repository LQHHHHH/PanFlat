"""Command-line interface (`panflat build`, `panflat verify`, `panflat merge`) (§6, F26, T22)."""

from __future__ import annotations

import argparse
import gc
import sys
from typing import Sequence

from panflat.gfa_io import load_and_validate_inputs, validate_build_config
from panflat.linearizer import run_chromosome_pipeline
from panflat.models import BuildConfig, CLIArgumentError, PanFlatError
from panflat.verify_qc import verify_build_outputs
from panflat.writers import merge_chromosomes, write_build_outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panflat",
        description="PanFlat v1: Path-Relative Recursive Linearizer for PGGB Pangenome Graphs",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. panflat build
    p_build = subparsers.add_parser(
        "build",
        help="Linearize a single-chromosome PGGB graph against a reference backbone",
    )
    p_build.add_argument("--gfa", required=True, help="Input GFA v1 file path")
    p_build.add_argument(
        "--ref-fa", required=True, help="Input reference FASTA file path"
    )
    p_build.add_argument(
        "--backbone-prefix",
        required=True,
        help="Reference path prefix in GFA (e.g. 'Ref#1#')",
    )
    p_build.add_argument(
        "--out-prefix",
        required=True,
        help="Output file prefix for the 11 output files",
    )
    p_build.add_argument(
        "--ref-gff",
        default=None,
        help="Optional reference GFF3 file for coordinate liftover",
    )
    p_build.add_argument(
        "--min-sv-len",
        type=int,
        default=50,
        help="Minimum SV / candidate block length in bp (default: 50)",
    )
    p_build.add_argument(
        "--max-jump",
        type=int,
        default=500000,
        help="Maximum backbone anchor jump before LARGE_JUMP / DUP_TRA (default: 500000)",
    )
    p_build.add_argument(
        "--cluster-id",
        type=float,
        default=0.95,
        help="Gap-compressed sequence identity threshold for clustering (default: 0.95)",
    )
    p_build.add_argument(
        "--cluster-cov",
        type=float,
        default=0.90,
        help="Optional cluster coverage parameter (default: 0.90)",
    )
    p_build.add_argument(
        "--cluster-win-id",
        type=float,
        default=0.70,
        help="Minimum sliding-window identity threshold for clustering (default: 0.70)",
    )
    p_build.add_argument(
        "--sec-backbone",
        default="support",
        help="Secondary backbone selection strategy ('support' or 'longest')",
    )
    p_build.add_argument(
        "--max-copy",
        type=int,
        default=1,
        help="Maximum copies retained during novel tandem repeat cycle folding (default: 1)",
    )
    p_build.add_argument(
        "--spacer-n",
        type=int,
        default=100,
        help="Number of 'N' bases in artificial spacers (default: 100)",
    )
    p_build.add_argument(
        "--splice-window",
        type=int,
        default=8,
        help="Splice-site proximity window in bp for GFF3 liftover (default: 8)",
    )
    p_build.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads (default: 1)",
    )
    p_build.add_argument(
        "--repeat-guard",
        default="on",
        help="Universal repeat-guard mode ('on' or 'off', default: 'on')",
    )
    p_build.add_argument(
        "--centromere-bed",
        default=None,
        help="Optional BED file delimiting reference centromere intervals",
    )
    p_build.add_argument(
        "--centromere-mode",
        default="off",
        help="Centromere processing mode ('off', 'bypass', or 'extract-islands')",
    )
    p_build.add_argument(
        "--centromere-min-island",
        type=int,
        default=500,
        help="Minimum island length in bp for extract-islands mode (default: 500)",
    )
    p_build.add_argument(
        "--dump-repeat-sequences",
        action="store_true",
        default=False,
        help="Export repeat-filtered sequences (REPEAT_LIKE, CEN_REPEAT_LIKE) to unresolved.fa",
    )
    p_build.add_argument(
        "--qc",
        action="store_true",
        default=False,
        help="Compute alignment redundancy QC metrics in summary.tsv",
    )

    # 2. panflat verify
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify Global Invariants I, II, III and output consistency on an existing build",
    )
    p_verify.add_argument(
        "--prefix", required=True, help="Output prefix to verify"
    )
    p_verify.add_argument(
        "--gfa", required=True, help="Original input GFA v1 file path"
    )
    p_verify.add_argument(
        "--ref-fa", required=True, help="Original reference FASTA file path"
    )
    p_verify.add_argument(
        "--ref-gff",
        default=None,
        help="Optional reference GFF3 file path",
    )
    p_verify.add_argument(
        "--centromere-bed",
        default=None,
        help="Optional centromere BED file path",
    )
    p_verify.add_argument(
        "--backbone-prefix",
        default=None,
        help="Optional reference backbone prefix",
    )
    p_verify.add_argument(
        "--spacer-n",
        type=int,
        default=None,
        help="Optional spacer length override",
    )

    # 3. panflat merge
    p_merge = subparsers.add_parser(
        "merge",
        help="Merge per-chromosome PanFlat outputs into a multi-chromosome bundle",
    )
    p_merge.add_argument(
        "--prefix-list",
        required=True,
        help="File containing one per-chromosome output prefix per line",
    )
    p_merge.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix for merged genome-wide files",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `panflat` (`build`, `verify`, `merge`)."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code

    try:
        if args.subcommand == "build":
            if not (0.0 < args.cluster_cov <= 1.0):
                raise CLIArgumentError(
                    f"--cluster-cov must be in (0.0, 1.0], got {args.cluster_cov}"
                )

            cen_mode_explicit = any(
                tok == "--centromere-mode"
                or tok.startswith("--centromere-mode=")
                for tok in raw_argv
            )
            cen_min_island_explicit = any(
                tok == "--centromere-min-island"
                or tok.startswith("--centromere-min-island=")
                for tok in raw_argv
            )

            config = BuildConfig(
                gfa=str(args.gfa),
                ref_fa=str(args.ref_fa),
                backbone_prefix=str(args.backbone_prefix),
                out_prefix=str(args.out_prefix),
                ref_gff=str(args.ref_gff) if args.ref_gff is not None else None,
                min_sv_len=int(args.min_sv_len),
                max_jump=int(args.max_jump),
                cluster_id=float(args.cluster_id),
                cluster_win_id=float(args.cluster_win_id),
                sec_backbone=args.sec_backbone,
                max_copy=int(args.max_copy),
                spacer_n=int(args.spacer_n),
                splice_window=int(args.splice_window),
                threads=int(args.threads),
                repeat_guard=args.repeat_guard,
                centromere_bed=(
                    str(args.centromere_bed)
                    if args.centromere_bed is not None
                    else None
                ),
                centromere_mode=args.centromere_mode,
                centromere_min_island=int(args.centromere_min_island),
                dump_repeat_sequences=bool(args.dump_repeat_sequences),
                qc=bool(args.qc),
                centromere_mode_explicit=cen_mode_explicit,
                centromere_min_island_explicit=cen_min_island_explicit,
            )

            validate_build_config(config)
            graph = load_and_validate_inputs(config)
            result = run_chromosome_pipeline(graph, config)
            del graph
            gc.collect()
            write_build_outputs(result, config)
            del result
            gc.collect()

            verify_build_outputs(
                prefix=config.out_prefix,
                gfa_path=config.gfa,
                ref_fa_path=config.ref_fa,
                ref_gff_path=config.ref_gff,
                centromere_bed_path=config.centromere_bed,
                spacer_n=config.spacer_n,
                raise_on_error=True,
            )
            return 0

        if args.subcommand == "verify":
            report = verify_build_outputs(
                prefix=args.prefix,
                gfa_path=args.gfa,
                ref_fa_path=args.ref_fa,
                ref_gff_path=args.ref_gff,
                centromere_bed_path=args.centromere_bed,
                spacer_n=args.spacer_n,
                raise_on_error=False,
            )
            if not report.passed:
                for err in report.errors:
                    print(err, file=sys.stderr)
                return 1
            return 0

        if args.subcommand == "merge":
            merge_chromosomes(
                prefix_list_path=args.prefix_list,
                out_prefix=args.out_prefix,
            )
            return 0

        print(f"Unknown subcommand: {args.subcommand}", file=sys.stderr)
        return 1

    except (PanFlatError, ValueError, OSError) as exc:
        print(f"panflat error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
