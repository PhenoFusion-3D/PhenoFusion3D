#!/usr/bin/env bash
# Robust Linux/WSL launcher for PhenoFusion3D.
#
# Designed so a freshly-imaged lab Linux machine can `git clone` + `bash
# launch.sh` and reach the GUI in one shot. Specifically the launcher:
#   - Picks a Python >= 3.10 interpreter, trying common names in order
#     and apt-installing python3.10 if no usable interpreter is found.
#   - Creates / refreshes a dedicated Linux venv (default: .venv-linux)
#     with --system-site-packages so the ROS-installed rospy is visible.
#   - Installs the project + Python deps in editable mode.
#   - Detects missing native runtime libs by ldd'ing the Qt xcb platform
#     plugin (the usual culprit for "Could not load the Qt platform
#     plugin xcb" crashes on a fresh box) and apt-installs them in a
#     single sudo call.
#   - Pins QT_PLUGIN_PATH to PyQt's plugins so cv2's bundled Qt plugins
#     do not get loaded instead.
#   - Launches main.py and forwards CLI args.
#
# The Windows venv (venv/Scripts) is never touched; the Linux venv lives
# in .venv-linux/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() { printf '[launch] %s\n' "$*"; }
warn() { printf '[launch] WARNING: %s\n' "$*" >&2; }
err() { printf '[launch] ERROR: %s\n' "$*" >&2; }

###############################################################################
# 1. Pick a Python >= 3.10 interpreter.
###############################################################################

python_version_ok() {
    # $1 = interpreter; returns 0 if it reports Python >= 3.10.
    local interp="$1"
    "$interp" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)
PY
}

apt_install() {
    # Idempotent helper: install a list of apt packages with sudo. Returns
    # non-zero if sudo / apt is unavailable so callers can fall back.
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not available; cannot auto-install: $*"
        return 1
    fi
    log "Installing system packages (may prompt for sudo): $*"
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    else
        apt-get update -y
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    fi
}

pick_python() {
    local candidates=(python3.12 python3.11 python3.10 python3)
    for cand in "${candidates[@]}"; do
        if command -v "$cand" >/dev/null 2>&1 && python_version_ok "$cand"; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN="${PHENOFUSION_PYTHON:-}"
if [ -n "$PYTHON_BIN" ]; then
    if ! python_version_ok "$PYTHON_BIN"; then
        err "PHENOFUSION_PYTHON=$PYTHON_BIN is not Python >= 3.10."
        exit 1
    fi
else
    if ! PYTHON_BIN="$(pick_python)"; then
        warn "No Python >= 3.10 found on PATH. Attempting to install python3.10..."
        # python3.10 is in the default Ubuntu 22.04 repos; on Ubuntu 20.04
        # it requires the deadsnakes PPA. Try the plain package first.
        if apt_install python3.10 python3.10-venv python3.10-distutils; then
            :
        else
            if command -v apt-get >/dev/null 2>&1; then
                warn "Adding deadsnakes PPA for python3.10 (Ubuntu 20.04 fallback)..."
                apt_install software-properties-common || true
                if command -v sudo >/dev/null 2>&1; then
                    sudo add-apt-repository -y ppa:deadsnakes/ppa || true
                else
                    add-apt-repository -y ppa:deadsnakes/ppa || true
                fi
                apt_install python3.10 python3.10-venv python3.10-distutils || true
            fi
        fi
        if ! PYTHON_BIN="$(pick_python)"; then
            err "No Python >= 3.10 interpreter could be found or installed."
            err "Install one manually (e.g. 'sudo apt install python3.10 python3.10-venv')"
            err "or set PHENOFUSION_PYTHON=/path/to/python3.10 and re-run."
            exit 1
        fi
    fi
fi

log "Using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])'))"

###############################################################################
# 2. Ensure python3-venv is installable (Debian/Ubuntu split python out).
###############################################################################

if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
    pyver="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
    warn "'$PYTHON_BIN -m venv' is unavailable; installing ${pyver}-venv..."
    apt_install "${pyver}-venv" || true
    if ! "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
        err "Cannot create venvs with $PYTHON_BIN. Install ${pyver}-venv and re-run."
        exit 1
    fi
fi

###############################################################################
# 3. Create / refresh the Linux venv.
###############################################################################

VENV_DIR="${PHENOFUSION_LINUX_VENV:-.venv-linux}"

venv_python_ok() {
    # Existing venv may have been built against a different (now-missing)
    # interpreter or a stale conda env. Reject it if its python is broken
    # or its symlink is dangling.
    [ -x "$VENV_DIR/bin/python" ] || return 1
    "$VENV_DIR/bin/python" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1
}

if [ ! -f "$VENV_DIR/.has_system_site" ] || [ ! -f "$VENV_DIR/bin/activate" ] || ! venv_python_ok; then
    log "Creating Linux venv at $VENV_DIR (Python $($PYTHON_BIN -c 'import sys; print(sys.version.split()[0])'), --system-site-packages)..."
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    touch "$VENV_DIR/.has_system_site"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Active venv: $VIRTUAL_ENV"

###############################################################################
# 4. Install Python dependencies if any are missing.
###############################################################################

REQUIRED_PY_MODULES=(PyQt5 open3d cv2 numpy natsort pyqtgraph matplotlib scipy tqdm)

missing_modules() {
    python - <<'PY'
import importlib.util
required = ["PyQt5", "open3d", "cv2", "numpy", "natsort", "pyqtgraph", "matplotlib", "scipy", "tqdm"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(",".join(missing))
PY
}

missing="$(missing_modules)"
if [ -n "$missing" ]; then
    log "Installing missing Python dependencies (${missing})..."
    python -m pip install --upgrade pip
    if ! python -m pip install -e ".[ros]"; then
        # The [ros] extra is identical to [windows] today (just pyrealsense2);
        # if it fails, retry with the windows extra so the GUI still launches.
        warn "pip install -e '.[ros]' failed; retrying without extras..."
        python -m pip install -e .
    fi
    missing="$(missing_modules)"
    if [ -n "$missing" ]; then
        err "Missing required packages after install: $missing"
        exit 1
    fi
fi
log "Python dependencies look good."

###############################################################################
# 5. Detect & install missing native runtime libs.
#
# PyQt5 wheels do NOT bundle xcb / xkb / EGL / GL client libs; those must
# come from the host. On a freshly-imaged Ubuntu the most common failure
# is the Qt xcb plugin aborting on libxcb-icccm.so.4 / libxcb-keysyms.so.1
# being absent, which produces an opaque "Could not load the Qt platform
# plugin xcb" crash. We discover the actual missing sonames by ldd'ing
# the plugin and map them to apt package names below.
###############################################################################

# Map .so name -> Debian/Ubuntu package providing it. Keep this list aligned
# with the actual NEEDED entries of libqxcb.so / libQt5XcbQpa.so. Packages
# absent from a fresh Ubuntu Desktop install commonly include:
declare -A SO_TO_PKG=(
    [libxcb-icccm.so.4]="libxcb-icccm4"
    [libxcb-keysyms.so.1]="libxcb-keysyms1"
    [libxcb-image.so.0]="libxcb-image0"
    [libxcb-render-util.so.0]="libxcb-render-util0"
    [libxcb-render.so.0]="libxcb-render0"
    [libxcb-shape.so.0]="libxcb-shape0"
    [libxcb-shm.so.0]="libxcb-shm0"
    [libxcb-sync.so.1]="libxcb-sync1"
    [libxcb-xfixes.so.0]="libxcb-xfixes0"
    [libxcb-xinerama.so.0]="libxcb-xinerama0"
    [libxcb-xkb.so.1]="libxcb-xkb1"
    [libxcb-randr.so.0]="libxcb-randr0"
    [libxcb-cursor.so.0]="libxcb-cursor0"
    [libxkbcommon.so.0]="libxkbcommon0"
    [libxkbcommon-x11.so.0]="libxkbcommon-x11-0"
    [libxcb.so.1]="libxcb1"
    [libxcb-util.so.1]="libxcb-util1"
    [libX11.so.6]="libx11-6"
    [libX11-xcb.so.1]="libx11-xcb1"
    [libXext.so.6]="libxext6"
    [libEGL.so.1]="libegl1"
    [libGL.so.1]="libgl1"
    [libGLX.so.0]="libglx0"
    [libGLdispatch.so.0]="libglvnd0"
    [libfontconfig.so.1]="libfontconfig1"
    [libdbus-1.so.3]="libdbus-1-3"
    [libxkbcommon-x11.so.0]="libxkbcommon-x11-0"
)

PLUGIN_DIRS=()
for cand in "$VIRTUAL_ENV/lib/"python*"/site-packages/PyQt5/Qt5/plugins/platforms"; do
    [ -d "$cand" ] && PLUGIN_DIRS+=("$cand")
done

needs_pkgs=()
seen_pkgs=()
add_pkg() {
    local pkg="$1"
    for existing in "${seen_pkgs[@]:-}"; do
        [ "$existing" = "$pkg" ] && return 0
    done
    seen_pkgs+=("$pkg")
    needs_pkgs+=("$pkg")
}

for plugins_dir in "${PLUGIN_DIRS[@]:-}"; do
    for sofile in "$plugins_dir/libqxcb.so" "$plugins_dir/../../lib/libQt5XcbQpa.so.5"; do
        [ -f "$sofile" ] || continue
        while IFS= read -r soname; do
            [ -n "$soname" ] || continue
            pkg="${SO_TO_PKG[$soname]:-}"
            if [ -n "$pkg" ]; then
                add_pkg "$pkg"
            else
                warn "Unknown apt package for missing library: $soname"
            fi
        done < <(ldd "$sofile" 2>/dev/null | awk '/=> not found/ {print $1}')
    done
done

if [ "${#needs_pkgs[@]}" -gt 0 ]; then
    if apt_install "${needs_pkgs[@]}"; then
        log "Native runtime libraries installed."
    else
        warn "Could not auto-install: ${needs_pkgs[*]}"
        warn "Install manually: sudo apt install ${needs_pkgs[*]}"
    fi
fi

###############################################################################
# 6. Force Qt to load PyQt's platform plugins (not cv2's bundled plugins).
###############################################################################

PYQT_PLUGINS_DIR="$(
python - <<'PY'
import os
from PyQt5.QtCore import QLibraryInfo
plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
print(plugins if os.path.isdir(plugins) else "")
PY
)"
if [ -n "$PYQT_PLUGINS_DIR" ]; then
    export QT_PLUGIN_PATH="$PYQT_PLUGINS_DIR"
    export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT_PLUGINS_DIR/platforms"
fi

# Let callers override (e.g. QT_QPA_PLATFORM=wayland bash launch.sh).
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Matplotlib config dir under $HOME may be unwritable in sandboxed setups
# (the workspace is shared on the lab box). Park it in /tmp if needed so
# the GUI does not spam warnings on every launch.
if [ -z "${MPLCONFIGDIR:-}" ]; then
    if ! mkdir -p "$HOME/.config/matplotlib" >/dev/null 2>&1; then
        export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$USER"
        mkdir -p "$MPLCONFIGDIR"
    fi
fi

exec python main.py "$@"
