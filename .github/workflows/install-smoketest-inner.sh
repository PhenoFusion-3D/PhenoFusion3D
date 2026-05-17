#!/usr/bin/env bash
# Inner test script for .github/workflows/install-smoketest.yml.
#
# This runs INSIDE a fresh ubuntu:20.04 container with /repo mounted
# read-only from the runner's checkout. It simulates a brand-new ANU
# lab box: install only git/curl/sudo, clone the repo, run
# `bash launch.sh --help`, then run `bash launch.sh` with
# QT_QPA_PLATFORM=offscreen and confirm the GUI mainloop actually
# starts. Failing any stage exits non-zero and fails the workflow.

set -e

echo "================ STAGE 1: bare $(. /etc/os-release && echo "$PRETTY_NAME") ================"
echo "python3:   $(command -v python3 || echo NONE)"
echo "uv:        $(command -v uv || echo NONE)"
echo "git:       $(command -v git || echo NONE)"
echo

echo "================ STAGE 2: install only git / curl / sudo ================"
apt-get update -qq
apt-get install -qq -y git curl sudo ca-certificates
useradd -m -s /bin/bash labuser
echo "labuser ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/labuser

# Modern git refuses to operate on repos owned by a different user
# ("dubious ownership"). The GitHub runner checks out as the runner
# UID; inside the container that UID maps to nobody. Make every git
# in the container trust the mounted /repo regardless of its owner --
# this is container-local, doesn't leak to the host.
git config --system --add safe.directory /repo
sudo -u labuser git config --global --add safe.directory /repo
echo

echo "================ STAGE 3: clone repo from mounted workspace ================"
sudo -u labuser git clone -q /repo /home/labuser/PhenoFusion3D
sudo -u labuser bash -c 'cd /home/labuser/PhenoFusion3D && git log -1 --format="    HEAD: %h %s"'
echo

echo "================ STAGE 4: launcher CLI mode (must reach main.py --help) ================"
set +e
sudo -u labuser -H bash -c 'cd /home/labuser/PhenoFusion3D && bash launch.sh --help' > /tmp/cli.log 2>&1
cli_ec=$?
set -e
echo "    launcher exit: $cli_ec"
if [ "$cli_ec" -ne 0 ]; then
    echo "FAIL: launcher exited non-zero in CLI mode"
    echo "--- launcher output (last 60 lines) ---"
    tail -60 /tmp/cli.log
    exit 1
fi
if ! grep -q "usage: main.py" /tmp/cli.log; then
    echo "FAIL: launcher did not reach main.py --help"
    echo "--- launcher output (last 60 lines) ---"
    tail -60 /tmp/cli.log
    exit 1
fi
echo "PASS: main.py --help reached stdout."
echo

echo "================ STAGE 5: launcher GUI mode (offscreen Qt, 15s) ================"
set +e
sudo -u labuser -H bash -c '
    cd /home/labuser/PhenoFusion3D
    export QT_QPA_PLATFORM=offscreen
    timeout 15 bash launch.sh > /tmp/gui_stdout.log 2> /tmp/gui_stderr.log
'
gui_ec=$?
set -e
echo "    launcher exit: $gui_ec"
case "$gui_ec" in
    124)
        echo "    -> 124 = SIGTERM from timeout = GUI mainloop was running (PASS)"
        ;;
    0)
        echo "    -> 0 = main.py exited cleanly within 15s (also PASS)"
        ;;
    *)
        echo "    -> exit $gui_ec = FAIL"
        echo "--- launcher stderr (last 60 lines) ---"
        tail -60 /tmp/gui_stderr.log
        echo "--- launcher stdout (last 20 lines) ---"
        tail -20 /tmp/gui_stdout.log
        exit "$gui_ec"
        ;;
esac
echo

echo "================ ALL STAGES PASSED ================"
