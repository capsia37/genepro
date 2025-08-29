#!/usr/bin/env bash
# Run run_regr.py under py-spy for multiple parameter combinations and collect speedscope outputs
# Produces a CSV file with averages for each run (only averages are included)

set -euo pipefail

# Configurable lists (edit if needed)
SIZES_DEFAULT="100,1000,10000,100000,1000000"
NJOBS_LIST=(1 2 4 8 16)
# opencl modes: "opencl" -> use OpenCL, "cpu" -> CPU only
OPENCL_MODES=(opencl cpu)

# Fixed number of generations to run (passed to run_regr.py)
MAX_GENS=40

# Seconds to wait between runs for cooldown
COOLDOWN=30

OUTDIR="profiles"
CSV_OUT="profiles_summary.csv"
PARSER="$(dirname "$0")/speedscope_parser.py"
RUN_SCRIPT="$(dirname "$0")/run_regr.py"
PYSPY="py-spy"

mkdir -p "$OUTDIR"

# Command-line flags
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force)
      FORCE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      # ignore unknown args
      shift
      ;;
  esac
done

# Emit CSV header
# We'll include columns: size,n_jobs,mode,elapsed_s,test_acc,<target labels...>
# Ask parser to print header using a representative param list (we'll pass the keys as params to ensure header includes them)
# The parser expects --csv-header and optional --param keys
python3 "$PARSER" --csv-header -P size= -P n_jobs= -P mode= -P max_gens= -P elapsed_s= -P test_acc= > "$CSV_OUT"

# parse sizes into array
IFS=',' read -r -a SIZES <<< "${SIZES_DEFAULT}"

# compute last values for skipping cooldown after final run
LAST_N_IDX=$(( ${#SIZES[@]} - 1 ))
LAST_N="${SIZES[$LAST_N_IDX]}"
LAST_NJ_IDX=$(( ${#NJOBS_LIST[@]} - 1 ))
LAST_NJ="${NJOBS_LIST[$LAST_NJ_IDX]}"
LAST_MODE_IDX=$(( ${#OPENCL_MODES[@]} - 1 ))
LAST_MODE="${OPENCL_MODES[$LAST_MODE_IDX]}"

for n in "${SIZES[@]}"; do
  for nj in "${NJOBS_LIST[@]}"; do
    for mode in "${OPENCL_MODES[@]}"; do
      use_opencl=true
      extra_args=()
      if [ "$mode" = "cpu" ]; then
        extra_args+=(--cpu-only)
        use_opencl=false
      else
        extra_args+=(--opencl-only)
        use_opencl=true
      fi

      prof_file="$OUTDIR/profile_n${n}_j${nj}_mode${mode}.speedscope.json"
      echo "Running n=$n, n_jobs=$nj, mode=$mode -> output: $prof_file"

      # Run under py-spy; record speedscope format. Use --output to write the file.
      # We forward the sizes via --sizes and n-jobs via --n-jobs
      # If the profile already exists and we're not forcing, skip creation
      PROFILE_REUSED=0
      if [ -f "$prof_file" ] && [ "$FORCE" -eq 0 ]; then
        echo "Profile $prof_file already exists; skipping py-spy run (use --force to overwrite)."
        PROFILE_REUSED=1
      else
        # Run py-spy once and ignore the known transient "No child process" error
        PROFILE_REUSED=0
        set +e
  # request a metadata sidecar file written by run_regr
  meta_file="${prof_file}.meta.json"
  pyspy_out=$($PYSPY record --idle --native --subprocesses --format speedscope --output "$prof_file" -- python3 "$RUN_SCRIPT" --sizes="$n" --n-jobs="$nj" --max-gens="$MAX_GENS" --meta-out="$meta_file" "${extra_args[@]}" 2>&1)
        rc=$?
        set -e

        if [ $rc -ne 0 ]; then
          if echo "$pyspy_out" | grep -qi "No child process"; then
            echo "py-spy reported 'No child process' but profile created; ignoring error."
            rc=0
          else
            echo "py-spy failed (rc=$rc):" >&2
            echo "$pyspy_out" >&2
          fi
        fi
      fi

      # Try to obtain runtime and accuracy from metadata sidecar if present, else fall back to parsing run output
      elapsed_s=""
      test_acc=""
      meta_file="${prof_file}.meta.json"
      if [ -f "$meta_file" ]; then
        # read from JSON meta file
        elapsed_s=$(python3 - <<PYCODE
import json,sys
try:
  data=json.load(open('$meta_file'))
  v=data.get('elapsed_s','')
  print(v if v is not None else '')
except Exception:
  sys.exit(0)
PYCODE
)
        test_acc=$(python3 - <<PYCODE
import json,sys
try:
  data=json.load(open('$meta_file'))
  v=data.get('test_acc','')
  print(v if v is not None else '')
except Exception:
  sys.exit(0)
PYCODE
)
      else
        if [ "$PROFILE_REUSED" -eq 0 ]; then
          # Look for the "Completed run" summary line which includes time and acc.
          elapsed_s=$(echo "$pyspy_out" | sed -n 's/.*Completed run:.*time=\([0-9.]*\)s.*/\1/p' | head -n1 || true)
          test_acc=$(echo "$pyspy_out" | sed -n 's/.*Completed run:.*acc=\([0-9.]*\).*/\1/p' | head -n1 || true)
          # Fallback: run_regr also prints "The balanced accuracy on the test set is X.XXX"
          if [ -z "$test_acc" ]; then
            test_acc=$(echo "$pyspy_out" | sed -n 's/.*The balanced accuracy on the test set is \([0-9.]*\).*/\1/p' | head -n1 || true)
          fi
        fi
      fi

      # Feed the speedscope json into parser to get CSV line (averages only). Pass params so CSV contains them.
      csv_line=$(python3 "$PARSER" "$prof_file" --csv -P "size=$n" -P "n_jobs=$nj" -P "mode=$mode" -P "max_gens=$MAX_GENS" -P "elapsed_s=$elapsed_s" -P "test_acc=$test_acc") || true
      if [ -n "$csv_line" ]; then
        echo "$csv_line" >> "$CSV_OUT"
      else
        echo "No data produced for $prof_file; skipping CSV append." >&2
      fi

      # cooldown between runs to reduce temperature, but only for large datasets, not after the very last run,
      # and skip cooldown when the profile was reused
      if [ "$n" = "$LAST_N" ] && [ "$nj" = "$LAST_NJ" ] && [ "$mode" = "$LAST_MODE" ]; then
        :
      else
        # only apply cooldown when dataset size > 10000 and profile was not reused
        if [ "$PROFILE_REUSED" -eq 0 ] && [ "$n" -gt 10000 ]; then
          echo "Cooling down for ${COOLDOWN}s..."
          sleep "$COOLDOWN"
        fi
      fi

    done
  done
done

echo "All runs done. CSV summary written to $CSV_OUT"
