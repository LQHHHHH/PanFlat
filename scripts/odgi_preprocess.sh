#!/usr/bin/env bash
# PanFlat v1 — ODGI Reference-Guided Preprocessing Wrapper
#
# Runs the recommended 4-step ODGI preprocessing pipeline on a PGGB output
# directory (containing *.smooth.final.og) or a direct .og graph file:
#   1. odgi groom  (reference-guided strand orientation)
#   2. odgi unchop (unitig compaction)
#   3. odgi sort   (reference-guided Ygs sort, or fast -O node ID compaction)
#   4. odgi view   (export GFA v1 ready for `panflat build`)
#
# Usage examples:
#   # Using exact reference path name:
#   scripts/odgi_preprocess.sh \
#     -i community9_pggb_p96_s10k_asm5_g7910_f0_k47 \
#     -r "Pyrbr_ds#1#Chr15A" \
#     -o c9 \
#     -t 16
#
#   # Using PanSN prefix (auto-detects the unique reference path in the graph):
#   scripts/odgi_preprocess.sh \
#     -i community9_pggb_p96_s10k_asm5_g7910_f0_k47 \
#     -p "Pyrbr_ds#1#" \
#     -o c9 \
#     -t 16

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

INPUT=""
REF_PATH=""
REF_PREFIX=""
OUT_PREFIX=""
THREADS=16
SORT_MODE="ref-ygs"   # ref-ygs | optimize
KEEP_INTERMEDIATES=0
OVERWRITE=0

log() {
    printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$SCRIPT_NAME" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME -i <pggb_dir_or_og> (-r <ref_path> | -p <ref_prefix>) -o <out_prefix> [options]

Required arguments:
  -i, --input PATH          PGGB output directory (containing *.smooth.final.og)
                            or direct path to an input .og graph file.
  -r, --ref-path NAME       Exact full reference path name in the graph
                            (e.g. "Pyrbr_ds#1#Chr15A"). Mutually exclusive with -p.
  -p, --ref-prefix PREFIX   PanSN prefix matching a unique reference path in the graph
                            (e.g. "Pyrbr_ds#1#"). Mutually exclusive with -r.
  -o, --out-prefix PREFIX   Output prefix; produces <PREFIX>.clean.gfa and
                            <PREFIX>.ref_path.txt (parent directories created if needed).

Optional arguments:
  -t, --threads INT         Threads for odgi groom/unchop/sort [default: 16]
  --sort-mode MODE          Sorting strategy:
                              ref-ygs  : odgi sort -H <ref_path_file> -p Ygs (default)
                              optimize : odgi sort -O (fast 1..N node ID compaction)
  --keep-intermediates      Keep <PREFIX>.groom.og, <PREFIX>.unchop.og, <PREFIX>.sort.og
  -f, --force               Overwrite existing <PREFIX>.clean.gfa if present
  -h, --help                Show this help message and exit

Outputs:
  <out_prefix>.clean.gfa      Preprocessed GFA v1 graph for \`panflat build --gfa\`
  <out_prefix>.ref_path.txt   Resolved unique reference path name (1 line)
EOF
}

# ---- Parse CLI arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            INPUT="$2"
            shift 2
            ;;
        -r|--ref-path)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            REF_PATH="$2"
            shift 2
            ;;
        -p|--ref-prefix)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            REF_PREFIX="$2"
            shift 2
            ;;
        -o|--out-prefix)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            OUT_PREFIX="$2"
            shift 2
            ;;
        -t|--threads)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            THREADS="$2"
            shift 2
            ;;
        --sort-mode)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            SORT_MODE="$2"
            shift 2
            ;;
        --keep-intermediates)
            KEEP_INTERMEDIATES=1
            shift
            ;;
        -f|--force)
            OVERWRITE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (use -h for help)"
            ;;
    esac
done

# ---- Pre-flight validation ----
[[ -n "$INPUT" ]]      || die "Missing required argument: -i / --input"
[[ -n "$OUT_PREFIX" ]] || die "Missing required argument: -o / --out-prefix"
if [[ -n "$REF_PATH" && -n "$REF_PREFIX" ]]; then
    die "Specify either -r/--ref-path OR -p/--ref-prefix, not both"
fi
if [[ -z "$REF_PATH" && -z "$REF_PREFIX" ]]; then
    die "Must specify one of -r/--ref-path or -p/--ref-prefix"
fi
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    die "--threads must be a positive integer, got '$THREADS'"
fi
if [[ "$SORT_MODE" != "ref-ygs" && "$SORT_MODE" != "optimize" ]]; then
    die "--sort-mode must be 'ref-ygs' or 'optimize', got '$SORT_MODE'"
fi

command -v odgi >/dev/null 2>&1 || die "'odgi' not found in PATH"

# Resolve input .og graph
RAW_OG=""
if [[ -f "$INPUT" ]]; then
    RAW_OG="$INPUT"
elif [[ -d "$INPUT" ]]; then
    shopt -s nullglob
    candidates=( "$INPUT"/*.smooth.final.og )
    if [[ ${#candidates[@]} -eq 0 ]]; then
        candidates=( "$INPUT"/*.final.og )
    fi
    shopt -u nullglob
    if [[ ${#candidates[@]} -eq 0 ]]; then
        die "No *.smooth.final.og or *.final.og found in directory: '$INPUT'"
    elif [[ ${#candidates[@]} -gt 1 ]]; then
        printf 'Found multiple candidate .og files in %s:\n' "$INPUT" >&2
        printf '  %s\n' "${candidates[@]}" >&2
        die "Ambiguous input directory; pass the exact .og file to -i"
    fi
    RAW_OG="${candidates[0]}"
else
    die "Input path does not exist: '$INPUT'"
fi

OUT_DIR="$(dirname "$OUT_PREFIX")"
if [[ -n "$OUT_DIR" && "$OUT_DIR" != "." ]]; then
    mkdir -p "$OUT_DIR"
fi

CLEAN_GFA="${OUT_PREFIX}.clean.gfa"
REF_PATH_FILE="${OUT_PREFIX}.ref_path.txt"
if [[ -e "$CLEAN_GFA" && "$OVERWRITE" -ne 1 ]]; then
    die "Output already exists: '$CLEAN_GFA' (pass -f/--force to overwrite)"
fi

# ---- Validate & resolve unique reference path in RAW_OG ----
log "Inspecting paths in '$RAW_OG'..."
PATHS_OUT="$(odgi paths -i "$RAW_OG" -L)" || die "'odgi paths' failed on '$RAW_OG'"
[[ -n "$PATHS_OUT" ]] || die "No paths found in '$RAW_OG'"
mapfile -t ALL_PATHS <<< "$PATHS_OUT"

RESOLVED_REF=""
if [[ -n "$REF_PATH" ]]; then
    found=0
    for p in "${ALL_PATHS[@]}"; do
        if [[ "$p" == "$REF_PATH" ]]; then
            found=1
            RESOLVED_REF="$p"
            break
        fi
    done
    if [[ "$found" -ne 1 ]]; then
        log "Available paths in '$RAW_OG' (${#ALL_PATHS[@]} total):"
        printf '  %s\n' "${ALL_PATHS[@]}" >&2
        die "Exact reference path '$REF_PATH' not found in '$RAW_OG'"
    fi
else
    matches=()
    for p in "${ALL_PATHS[@]}"; do
        if [[ "$p" == "${REF_PREFIX}"* ]]; then
            matches+=( "$p" )
        fi
    done
    if [[ ${#matches[@]} -eq 0 ]]; then
        log "Available paths in '$RAW_OG' (${#ALL_PATHS[@]} total):"
        printf '  %s\n' "${ALL_PATHS[@]}" >&2
        die "No path matches --ref-prefix '$REF_PREFIX' in '$RAW_OG'"
    elif [[ ${#matches[@]} -gt 1 ]]; then
        log "Matched multiple paths for --ref-prefix '$REF_PREFIX':"
        printf '  %s\n' "${matches[@]}" >&2
        die "Expected exactly 1 reference path, found ${#matches[@]}; pass exact name via -r/--ref-path"
    fi
    RESOLVED_REF="${matches[0]}"
fi

printf '%s\n' "$RESOLVED_REF" > "$REF_PATH_FILE"
log "Resolved reference path: '$RESOLVED_REF' (total paths=${#ALL_PATHS[@]}, threads=${THREADS}, sort_mode=${SORT_MODE})"

# ---- Set up intermediate files & cleanup trap ----
GROOM_OG="${OUT_PREFIX}.groom.og"
UNCHOP_OG="${OUT_PREFIX}.unchop.og"
SORT_OG="${OUT_PREFIX}.sort.og"
TMP_GFA="${CLEAN_GFA}.tmp.$$"

cleanup() {
    rm -f "$TMP_GFA"
    if [[ "$KEEP_INTERMEDIATES" -ne 1 ]]; then
        rm -f "$GROOM_OG" "$UNCHOP_OG" "$SORT_OG"
    fi
}
trap cleanup EXIT

t_start=$SECONDS

# ---- Step 1: odgi groom ----
t0=$SECONDS
log "[1/4] odgi groom -R '$REF_PATH_FILE' ..."
odgi groom -i "$RAW_OG" -o "$GROOM_OG" -R "$REF_PATH_FILE" -t "$THREADS"
log "  groom done ($((SECONDS - t0))s)"

# ---- Step 2: odgi unchop ----
t0=$SECONDS
log "[2/4] odgi unchop ..."
odgi unchop -i "$GROOM_OG" -o "$UNCHOP_OG" -t "$THREADS"
log "  unchop done ($((SECONDS - t0))s)"

# ---- Step 3: odgi sort ----
t0=$SECONDS
if [[ "$SORT_MODE" == "ref-ygs" ]]; then
    if odgi sort --help 2>&1 | grep -q -- '-H'; then
        log "[3/4] odgi sort -H '$REF_PATH_FILE' -p Ygs ..."
        odgi sort -i "$UNCHOP_OG" -o "$SORT_OG" -H "$REF_PATH_FILE" -p Ygs -t "$THREADS"
    else
        log "WARNING: installed 'odgi sort' does not advertise -H; falling back to 'odgi sort -p Ygs'"
        log "[3/4] odgi sort -p Ygs ..."
        odgi sort -i "$UNCHOP_OG" -o "$SORT_OG" -p Ygs -t "$THREADS"
    fi
else
    log "[3/4] odgi sort -O (fast node ID optimization) ..."
    odgi sort -i "$UNCHOP_OG" -o "$SORT_OG" -O -t "$THREADS"
fi
log "  sort done ($((SECONDS - t0))s)"

# ---- Step 4: odgi view -g ----
t0=$SECONDS
log "[4/4] odgi view -g -> '$CLEAN_GFA' ..."
odgi view -i "$SORT_OG" -g > "$TMP_GFA"
[[ -s "$TMP_GFA" ]] || die "Exported GFA is empty: '$TMP_GFA'"
mv -f "$TMP_GFA" "$CLEAN_GFA"
log "  view done ($((SECONDS - t0))s)"

GFA_SIZE="$(du -h "$CLEAN_GFA" | awk '{print $1}')"
REF_CHROM="${RESOLVED_REF##*#}"
log "Preprocessing complete in $((SECONDS - t_start))s -> '$CLEAN_GFA' (${GFA_SIZE})"
log "Next step: run \`panflat build --gfa '$CLEAN_GFA' --ref-fa <ref_or_community.fa> --backbone-prefix '$RESOLVED_REF' --ref-gff <${REF_CHROM}.gff3> --out-prefix <prefix>\`"
