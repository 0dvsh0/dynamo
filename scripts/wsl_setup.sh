#!/usr/bin/env bash
# Prepare a WSL distro to run Dynamo.
#
#   bash scripts/wsl_setup.sh
#
# WSLg mounts /tmp/.X11-unix as a read-only tmpfs so that X clients can reach the
# Windows-side X server. That mount is fatal for our own Xvfb: X cannot bind its
# unix listener there, keeps running, and accepts nothing — every readiness probe
# then times out with no visible cause.
#
# Disabling GUI applications in /etc/wsl.conf stops WSLg remounting it (takes
# effect on the next `wsl --shutdown`); this script also repairs it immediately
# so the current session is usable right now.
set -u

X11DIR=/tmp/.X11-unix

echo "== current state =="
mount | grep " on $X11DIR " || echo "  $X11DIR is not a mount point"
stat -c '  mode=%a %F' "$X11DIR" 2>/dev/null || echo "  $X11DIR does not exist"

echo
echo "== disabling WSLg GUI applications (persistent) =="
mkdir -p /etc/wsl.conf
if grep -q 'guiApplications' /etc/wsl.conf 2>/dev/null; then
  echo "  already configured"
else
  printf '[wsl2]\nguiApplications=false\n' >> /etc/wsl.conf
  echo "  wrote /etc/wsl.conf (run 'wsl --shutdown' from Windows to apply)"
fi

echo
echo "== repairing now =="
if mount | grep -q " on $X11DIR "; then
  umount "$X11DIR" 2>&1 && echo "  unmounted the WSLg tmpfs" \
    || echo "  WARNING: could not unmount $X11DIR"
fi
if mount | grep -q " on $X11DIR "; then
  echo "  ERROR: $X11DIR is still a read-only mount; Xvfb cannot bind there."
  echo "         Run 'wsl --shutdown' from Windows, then re-run this script."
  exit 1
fi
rmdir "$X11DIR" 2>/dev/null
mkdir -p "$X11DIR"
chmod 1777 "$X11DIR"
stat -c '  mode=%a %F' "$X11DIR"

echo
echo "== verifying =="
if bash "$(dirname "$0")/preflight.sh"; then
  echo "wsl_setup: ready"
else
  echo "wsl_setup: display directory fixed, but preflight still reports problems"
  exit 1
fi
