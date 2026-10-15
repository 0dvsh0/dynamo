#!/usr/bin/env bash
# Pre-flight check for a host that will run Dynamo.
#
#   bash scripts/preflight.sh
#
# Exits non-zero if anything Dynamo depends on is missing or misconfigured.
set -u

fail=0
ok()   { printf '  \033[92mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[91mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }

echo "== required binaries =="
for b in Xvfb ffmpeg ffprobe python3; do
  if command -v "$b" >/dev/null 2>&1; then ok "$b -> $(command -v "$b")"; else bad "$b missing"; fi
done

echo
echo "== /tmp/.X11-unix =="
# X refuses to bind its unix listener unless this directory is 1777, and when it
# refuses it keeps running while accepting nothing. WSLg in particular mounts the
# directory read-only, which is the same failure with an even better disguise.
X11DIR=/tmp/.X11-unix
if mount | grep -q " on $X11DIR "; then
  warn "$X11DIR is a mount point; X cannot bind there"
  echo "        fix with: umount $X11DIR && mkdir -p $X11DIR && chmod 1777 $X11DIR"
  fail=1
elif [ -d "$X11DIR" ]; then
  mode=$(stat -c '%a' "$X11DIR")
  if [ "$mode" = "1777" ]; then ok "$X11DIR mode $mode"; else bad "$X11DIR mode $mode, X needs 1777"; fi
else
  warn "$X11DIR does not exist (Dynamo will create it)"
fi

echo
echo "== optional =="
command -v xdpyinfo  >/dev/null 2>&1 && ok "xdpyinfo  (stronger readiness probe)"  || warn "xdpyinfo missing (socket probe only)"
command -v xsetroot  >/dev/null 2>&1 && ok "xsetroot  (repaint the root window in tests)" || warn "xsetroot missing (apt-get install x11-xserver-utils)"

echo
echo "== versions =="
ffmpeg -version 2>/dev/null | head -1 || true
python3 --version 2>/dev/null || true

echo
echo "== dynamo =="
if command -v dynamo >/dev/null 2>&1; then
  ok "$(dynamo --version)"
  dynamo info | sed 's/^/  /'
else
  warn "dynamo not on PATH (pip install -e .)"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "preflight: OK"
else
  echo "preflight: problems found (see above)"
fi
exit "$fail"
