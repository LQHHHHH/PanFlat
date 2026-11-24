#!/usr/bin/env bash
# PanFlat v1 — ODGI Reference-Guided Preprocessing Wrapper (with Step Checkpoints)
#
# Runs the recommended 4-step ODGI preprocessing pipeline on a PGGB output
# directory (containing *.smooth.final.og) or a direct .og graph file:
#   1. odgi groom  (reference-guided strand orientation)
#   2. odgi unchop (unitig compaction)
#   3. odgi sort   (reference-guided Ygs sort, or fast -O node ID compaction)
#   4. odgi view   (export GFA v1 ready for `panflat build`)
#
# Output location rule:
#   - Default (no -o): writes <REF_CHROM>.clean.gfa, <REF_CHROM>.ref_path.txt,
#     and step checkpoint *.done files directly into the -i directory.
#   - Bare -o <name> (no '/'): writes <in_dir>/<name>.* inside the -i directory.
#   - Path -o <dir>/<name> (contains '/'): writes to the explicit <dir>/<name>.*.
#
# Checkpoints & resume:
#   Each step writes atomically to a temp file, moves it into place, and runs
#   `touch <prefix>.step{1..4}_*.done`. If a job is interrupted mid-way:
#     - Partial temp files are removed; completed step outputs & .done files are kept.
#     - Re-running the exact same command automatically skips completed steps and
#       resumes from the failed step.
#     - If all 4 steps are already done, re-running exits 0 immediately (batch-friendly).
#     - Pass -f/--force to clear checkpoints and rerun all steps from scratch.

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
Usage: $SCRIPT_NAME -i <pggb_dir_or_og> (-r <ref_path> | -p <ref_prefix>) [options]

Required arguments:
  -i, --input PATH          PGGB output directory (containing *.smooth.final.og)
                            or direct path to an input .og graph file.
  -r, --ref-path NAME       Exact full reference path name in the graph
                            (e.g. "Pyrbr_ds#1#Chr15A"). Mutually exclusive with -p.
  -p, --ref-prefix PREFIX   PanSN prefix matching a unique reference path in the graph
                            (e.g. "Pyrbr_ds#1#"). Mutually exclusive with -r.

Optional arguments:
  -o, --out-prefix PREFIX   Output prefix:
                              (omitted)   : <in_dir>/<REF_CHROM> (writes back into -i dir)
                              <name>      : <in_dir>/<name>      (writes back into -i dir)
                              <dir>/<name>: <dir>/<name>         (writes to explicit path)
  -t, --threads INT         Threads for odgi groom/unchop/sort [default: 16]
  --sort-mode MODE          Sorting strategy:
                              ref-ygs  : odgi sort -H <ref_path_file> -p Ygs (default)
                              optimize : odgi sort -O (fast 1..N node ID compaction)
  --keep-intermediates      Keep <prefix>.groom.og, <prefix>.unchop.og, <prefix>.sort.og
                            after all 4 steps succeed (on failure they are always kept for resume)
  -f, --force               Clear existing checkpoints and rerun all 4 steps from scratch
  -h, --help                Show this help message and exit

Outputs & checkpoints (inside -i directory by default):
  <prefix>.clean.gfa            Preprocessed GFA v1 graph for \`panflat build --gfa\`
  <prefix>.ref_path.txt         Resolved unique reference path name (1 line)
  <prefix>.step1_groom.done     Checkpoint marker after Step 1 (odgi groom)
  <prefix>.step2_unchop.done    Checkpoint marker after Step 2 (odgi unchop)
  <prefix>.step3_sort.done      Checkpoint marker after Step 3 (odgi sort)
  <prefix>.step4_view.done      Checkpoint marker after Step 4 (odgi view -g)
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
[[ -n "$INPUT" ]] || die "Missing required argument: -i / --input"
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
    INPUT_CLEAN="${INPUT%/}"
    [[ -n "$INPUT_CLEAN" ]] || INPUT_CLEAN="/"
    shopt -s nullglob
    candidates=( "${INPUT_CLEAN}"/*.smooth.final.og )
    if [[ ${#candidates[@]} -eq 0 ]]; then
        candidates=( "${INPUT_CLEAN}"/*.final.og )
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

IN_DIR="$(dirname "$RAW_OG")"

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

REF_CHROM="${RESOLVED_REF##*#}"
SAFE_REF_CHROM="${REF_CHROM//\//_}"

# Resolve output prefix: default & bare names write back into -i directory
if [[ -z "$OUT_PREFIX" ]]; then
    FINAL_PREFIX="${IN_DIR}/${SAFE_REF_CHROM}"
elif [[ "$OUT_PREFIX" != */* ]]; then
    FINAL_PREFIX="${IN_DIR}/${OUT_PREFIX}"
else
    FINAL_PREFIX="$OUT_PREFIX"
fi

OUT_DIR="$(dirname "$FINAL_PREFIX")"
if [[ -n "$OUT_DIR" && "$OUT_DIR" != "." ]]; then
    mkdir -p "$OUT_DIR"
fi

CLEAN_GFA="${FINAL_PREFIX}.clean.gfa"
REF_PATH_FILE="${FINAL_PREFIX}.ref_path.txt"
GROOM_OG="${FINAL_PREFIX}.groom.og"
UNCHOP_OG="${FINAL_PREFIX}.unchop.og"
SORT_OG="${FINAL_PREFIX}.sort.og"

DONE_GROOM="${FINAL_PREFIX}.step1_groom.done"
DONE_UNCHOP="${FINAL_PREFIX}.step2_unchop.done"
DONE_SORT="${FINAL_PREFIX}.step3_sort.done"
DONE_VIEW="${FINAL_PREFIX}.step4_view.done"

TMP_GROOM="${FINAL_PREFIX}.tmp.$$.groom.og"
TMP_UNCHOP="${FINAL_PREFIX}.tmp.$$.unchop.og"
TMP_SORT="${FINAL_PREFIX}.tmp.$$.sort.og"
TMP_GFA="${FINAL_PREFIX}.tmp.$$.clean.gfa"

# ---- Checkpoint & force handling ----
if [[ "$OVERWRITE" -eq 1 ]]; then
    log "Force rerun requested (-f/--force): clearing previous checkpoints for '${FINAL_PREFIX}'"
    rm -f "$DONE_GROOM" "$DONE_UNCHOP" "$DONE_SORT" "$DONE_VIEW"
elif [[ -f "$DONE_VIEW" && -s "$CLEAN_GFA" ]]; then
    log "Already complete ('${DONE_VIEW}' & '${CLEAN_GFA}' exist); skipping. Pass -f/--force to rerun."
    exit 0
elif [[ -e "$CLEAN_GFA" && ! -f "$DONE_VIEW" ]]; then
    die "Found '${CLEAN_GFA}' without checkpoint '${DONE_VIEW}' (possibly incomplete or external); pass -f/--force to overwrite"
fi

# Determine minimal set of steps required (backward dependency check)
NEED_VIEW=1
if [[ -f "$DONE_SORT" && -s "$SORT_OG" ]]; then
    NEED_SORT=0
    NEED_UNCHOP=0
    NEED_GROOM=0
else
    NEED_SORT=1
    rm -f "$DONE_SORT"
    if [[ -f "$DONE_UNCHOP" && -s "$UNCHOP_OG" ]]; then
        NEED_UNCHOP=0
        NEED_GROOM=0
    else
        NEED_UNCHOP=1
        rm -f "$DONE_UNCHOP"
        if [[ -f "$DONE_GROOM" && -s "$GROOM_OG" ]]; then
            NEED_GROOM=0
        else
            NEED_GROOM=1
            rm -f "$DONE_GROOM"
        fi
    fi
fi

printf '%s\n' "$RESOLVED_REF" > "$REF_PATH_FILE"
log "Resolved reference path: '$RESOLVED_REF' -> output prefix '${FINAL_PREFIX}' (total paths=${#ALL_PATHS[@]}, threads=${THREADS}, sort_mode=${SORT_MODE})"

# ---- Cleanup trap: preserve completed intermediates on failure for resume ----
PIPELINE_SUCCEEDED=0
cleanup() {
    rm -f "$TMP_GROOM" "$TMP_UNCHOP" "$TMP_SORT" "$TMP_GFA"
    if [[ "$PIPELINE_SUCCEEDED" -eq 1 && "$KEEP_INTERMEDIATES" -ne 1 ]]; then
        rm -f "$GROOM_OG" "$UNCHOP_OG" "$SORT_OG"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

t_start=$SECONDS

# ---- Step 1: odgi groom ----
if [[ "$NEED_GROOM" -eq 0 ]]; then
    log "[1/4] SKIP odgi groom (checkpoint '${DONE_GROOM}' ready)"
else
    t0=$SECONDS
    log "[1/4] odgi groom -R '$REF_PATH_FILE' ..."
    odgi groom -i "$RAW_OG" -o "$TMP_GROOM" -R "$REF_PATH_FILE" -t "$THREADS"
    [[ -s "$TMP_GROOM" ]] || die "odgi groom produced empty output: '$TMP_GROOM'"
    mv -f "$TMP_GROOM" "$GROOM_OG"
    touch "$DONE_GROOM"
    log "  groom done ($((SECONDS - t0))s) -> touched '${DONE_GROOM}'"
fi

# ---- Step 2: odgi unchop ----
if [[ "$NEED_UNCHOP" -eq 0 ]]; then
    log "[2/4] SKIP odgi unchop (checkpoint '${DONE_UNCHOP}' ready)"
else
    t0=$SECONDS
    log "[2/4] odgi unchop ..."
    odgi unchop -i "$GROOM_OG" -o "$TMP_UNCHOP" -t "$THREADS"
    [[ -s "$TMP_UNCHOP" ]] || die "odgi unchop produced empty output: '$TMP_UNCHOP'"
    mv -f "$TMP_UNCHOP" "$UNCHOP_OG"
    touch "$DONE_UNCHOP"
    log "  unchop done ($((SECONDS - t0))s) -> touched '${DONE_UNCHOP}'"
fi

# ---- Step 3: odgi sort ----
if [[ "$NEED_SORT" -eq 0 ]]; then
    log "[3/4] SKIP odgi sort (checkpoint '${DONE_SORT}' ready)"
else
    t0=$SECONDS
    if [[ "$SORT_MODE" == "ref-ygs" ]]; then
        if odgi sort --help 2>&1 | grep -q -- '-H'; then
            log "[3/4] odgi sort -H '$REF_PATH_FILE' -p Ygs ..."
            odgi sort -i "$UNCHOP_OG" -o "$TMP_SORT" -H "$REF_PATH_FILE" -p Ygs -t "$THREADS"
        else
            log "WARNING: installed 'odgi sort' does not advertise -H; falling back to 'odgi sort -p Ygs'"
            log "[3/4] odgi sort -p Ygs ..."
            odgi sort -i "$UNCHOP_OG" -o "$TMP_SORT" -p Ygs -t "$THREADS"
        fi
    else
        log "[3/4] odgi sort -O (fast node ID optimization) ..."
        odgi sort -i "$UNCHOP_OG" -o "$TMP_SORT" -O -t "$THREADS"
    fi
    [[ -s "$TMP_SORT" ]] || die "odgi sort produced empty output: '$TMP_SORT'"
    mv -f "$TMP_SORT" "$SORT_OG"
    touch "$DONE_SORT"
    log "  sort done ($((SECONDS - t0))s) -> touched '${DONE_SORT}'"
fi

# ---- Step 4: odgi view -g ----
if [[ "$NEED_VIEW" -eq 1 ]]; then
    t0=$SECONDS
    log "[4/4] odgi view -g -> '$CLEAN_GFA' ..."
    odgi view -i "$SORT_OG" -g > "$TMP_GFA"
    [[ -s "$TMP_GFA" ]] || die "Exported GFA is empty: '$TMP_GFA'"
    mv -f "$TMP_GFA" "$CLEAN_GFA"
    touch "$DONE_VIEW"
    log "  view done ($((SECONDS - t0))s) -> touched '${DONE_VIEW}'"
fi

PIPELINE_SUCCEEDED=1

GFA_SIZE="$(du -h "$CLEAN_GFA" | awk '{print $1}')"
log "Preprocessing complete in $((SECONDS - t_start))s -> '$CLEAN_GFA' (${GFA_SIZE})"
log "Next step: run \`panflat build --gfa '$CLEAN_GFA' --ref-fa <ref_or_community.fa> --backbone-prefix '$RESOLVED_REF' --ref-gff <${REF_CHROM}.gff3> --out-prefix <prefix>\`"
