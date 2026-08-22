#!/usr/bin/env bash
set -u

ROOT=/workspace/project-expertise
PYTHON="$ROOT/.venv/bin/python"
MANIFEST="$ROOT/results/new763/matrix_manifest.json"
LOG="$ROOT/results/new763/finalizer.log"
RUNNER_PID="${1:-}"

mkdir -p "$ROOT/results/new763"
exec >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finalizer started runner_pid=$RUNNER_PID"

while true; do
  state="$($PYTHON - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print('waiting')
    raise SystemExit
d = json.loads(p.read_text())
jobs = d.get('jobs', {})
states = [v.get('status') for v in jobs.values()]
print('done' if len(jobs) >= 9 and states and all(s in {'done', 'failed'} for s in states) else 'waiting')
PY
  )"
  if [[ "$state" == done ]]; then
    break
  fi
  if [[ -n "$RUNNER_PID" ]] && ! ps -p "$RUNNER_PID" >/dev/null 2>&1; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] runner exited before terminal manifest; waiting 120s"
    sleep 120
  else
    sleep 60
  fi
done

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] running summary"
"$PYTHON" "$ROOT/scripts/summarize_new763.py" \
  --results-dir "$ROOT/results/new763" \
  --output "$ROOT/results/new763_summary.json"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] running campaign evaluation"
"$PYTHON" "$ROOT/scripts/eval_new763_campaigns.py" \
  --results-dir "$ROOT/results/new763" \
  --output "$ROOT/results/new763_campaigns.json"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finalizer complete"
