#!/usr/bin/env bash
# Run the Dynamo test suite on this machine.
#
#   bash scripts/run_tests.sh                 # everything
#   bash scripts/run_tests.sh -k token_reducer
#
# On WSL it repairs the /tmp/.X11-unix mount first, in the same shell session —
# WSLg re-creates that read-only mount on every new session, so doing it in a
# separate command does not survive to the test run.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

if uname -r 2>/dev/null | grep -qi microsoft && [ -f "$HERE/wsl_setup.sh" ]; then
  bash "$HERE/wsl_setup.sh" >/dev/null 2>&1 || {
    echo "run_tests: wsl_setup.sh could not repair /tmp/.X11-unix" >&2
    bash "$HERE/wsl_setup.sh" || true
    exit 1
  }
fi

# Stale servers from an interrupted run would make "display already in use" the
# wrong explanation for whatever went wrong.
pkill -x Xvfb 2>/dev/null
sleep 0.5

cd "$ROOT"
exec python3 -m pytest "$@"
