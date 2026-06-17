#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 '<cmd with {}>' [TOTAL] [CONCURRENCY] [LOGDIR] [START]

Example: $0 './train_dreambooth_cara.sh {}' 750 4 logs
Example (start at task 134): $0 './train_dreambooth_cara.sh {}' 749 2 logs 134

The string '{}' in the command will be replaced with the task id (1..TOTAL).
Each task is started with nohup and its stdout/stderr goes to LOGDIR/task-<id>.out
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

CMD_TEMPLATE="$1"
TOTAL=${2:-749}
CONCURRENCY=${3:-4}
LOGDIR=${4:-logs}
START=${5:-0}

if [ "$START" -gt "$TOTAL" ]; then
  echo "START ($START) cannot be greater than TOTAL ($TOTAL)"
  exit 2
fi

mkdir -p "$LOGDIR"

pids=()

for i in $(seq "$START" "$TOTAL"); do
  CMD=${CMD_TEMPLATE//\{\}/$i}
  echo "Launching task $i: $CMD"
  nohup bash -c "$CMD" > "$LOGDIR/task-$i.out" 2>&1 &
  pid=$!
  pids+=("$pid")

  # If we've reached concurrency, wait for at least one to finish
  while [ "${#pids[@]}" -ge "$CONCURRENCY" ]; do
    # prune finished pids
    alive=()
    for p in "${pids[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        alive+=("$p")
      fi
    done
    pids=("${alive[@]}")
    if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
      sleep 1
    fi
  done
done

# wait for remaining pids
for p in "${pids[@]}"; do
  wait "$p" || true
done

echo "All tasks from $START to $TOTAL completed. Logs in $LOGDIR"
