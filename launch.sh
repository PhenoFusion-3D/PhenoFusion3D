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

# All progress/diagnostic logging goes to stderr so that functions which
# return data via stdout (e.g. `PYTHON_BIN="$(pick_python)"` or
# `uv_py="$(install_python_via_uv)"`) are not polluted with log lines.
log() { printf '[launch] %s\n' "$*" >&2; }
warn() { printf '[launch] WARNING: %s\n' "$*" >&2; }
err() { printf '[launch] ERROR: %s\n' "$*" >&2; }

###############################################################################
# 1. Pick a Python >= 3.10 interpreter.
###############################################################################

python_version_ok() {
    # $1 = interpreter; returns 0 if it reports Python 3.10 <= ver <= 3.12.
    # 3.13+ is excluded because Open3D / pyrealsense2 wheels don't exist
    # for it yet; picking a 3.13 interpreter would cause `pip install -e .`
    # to fail with "no matching distribution found for open3d".
    local interp="$1"
    "$interp" - <<'PY' >/dev/null 2>&1
import sys
ver = sys.version_info[:2]
sys.exit(0 if (3, 10) <= ver <= (3, 12) else 1)
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
    # Order of preference:
    #   1. An existing valid venv's python (so we don't reinstall on a
    #      machine where someone already provisioned the interpreter).
    #   2. python3.12 / 3.11 / 3.10 / python3 on PATH.
    #   3. uv-installed Pythons (from a previous fallback run).
    #   4. Miniforge-managed Pythons under $HOME.
    local venv_default="${PHENOFUSION_LINUX_VENV:-.venv-linux}"
    if [ -x "$venv_default/bin/python" ] && python_version_ok "$venv_default/bin/python"; then
        echo "$venv_default/bin/python"
        return 0
    fi
    local candidates=(python3.12 python3.11 python3.10 python3)
    local extra_dirs=(
        "$HOME/.local/share/uv/python"
        "$HOME/.cache/uv/python"
        "$HOME/.miniforge3/bin"
        "$HOME/miniforge3/bin"
    )
    for cand in "${candidates[@]}"; do
        if command -v "$cand" >/dev/null 2>&1 && python_version_ok "$cand"; then
            echo "$cand"
            return 0
        fi
    done
    if command -v uv >/dev/null 2>&1; then
        local uv_py
        uv_py="$(uv python find '>=3.10' 2>/dev/null || true)"
        if [ -n "$uv_py" ] && [ -x "$uv_py" ] && python_version_ok "$uv_py"; then
            echo "$uv_py"
            return 0
        fi
    fi
    for dir in "${extra_dirs[@]}"; do
        [ -d "$dir" ] || continue
        for cand in "$dir"/cpython-*/bin/python3 "$dir"/python3.1[0-9] "$dir"/python3; do
            if [ -x "$cand" ] && python_version_ok "$cand"; then
                echo "$cand"
                return 0
            fi
        done
    done
    return 1
}

apt_python_install_attempt() {
    # Try a single apt-install of python$1, python$1-venv, python$1-distutils.
    # Returns 0 if any usable Python >= 3.10 ends up on PATH afterwards.
    local ver="$1"
    apt_install "python${ver}" "python${ver}-venv" "python${ver}-distutils" 2>/dev/null || true
    command -v "python${ver}" >/dev/null 2>&1 && python_version_ok "python${ver}"
}

install_uv() {
    # Install Astral's uv (single static binary; no sudo, no python needed).
    # Idempotent: returns 0 if uv is already on PATH.
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    log "Installing 'uv' (standalone Python downloader; no apt/sudo required)..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    else
        warn "Need curl or wget to install uv; please install one (e.g. 'sudo apt install curl')."
        return 1
    fi
    # The installer drops uv into $HOME/.local/bin or $HOME/.cargo/bin.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1
}

install_python_via_uv() {
    # Last-resort Python installer that bypasses apt entirely. Asks uv to
    # download CPython 3.11 (chosen for broad wheel availability) into a
    # user-local cache and returns the path to the interpreter on stdout.
    # NOTE: all progress output must go to stderr -- this function's stdout
    # is captured by the caller via `$(install_python_via_uv)`.
    install_uv || return 1
    log "Downloading CPython 3.11 via uv (this stays under $HOME, no sudo)..."
    if ! uv python install 3.11 >&2; then
        return 1
    fi
    local uv_py
    uv_py="$(uv python find 3.11 2>/dev/null || true)"
    if [ -n "$uv_py" ] && [ -x "$uv_py" ] && python_version_ok "$uv_py"; then
        echo "$uv_py"
        return 0
    fi
    return 1
}

PYTHON_BIN="${PHENOFUSION_PYTHON:-}"
if [ -n "$PYTHON_BIN" ]; then
    if ! python_version_ok "$PYTHON_BIN"; then
        err "PHENOFUSION_PYTHON=$PYTHON_BIN must be Python 3.10, 3.11, or 3.12."
        exit 1
    fi
else
    if ! PYTHON_BIN="$(pick_python)"; then
        warn "No Python >= 3.10 found on PATH. Trying apt first..."
        # Try the easy path: distro-packaged python3.10 (works out of the
        # box on Ubuntu 22.04+; needs the deadsnakes PPA on Ubuntu 20.04).
        # If 3.10 isn't available, fall through to 3.11, then 3.12.
        installed_via_apt=false
        for ver in 3.10 3.11 3.12; do
            if apt_python_install_attempt "$ver"; then
                installed_via_apt=true
                break
            fi
        done
        if [ "$installed_via_apt" != true ] && command -v apt-get >/dev/null 2>&1; then
            warn "Plain apt could not provide Python >= 3.10; adding deadsnakes PPA..."
            apt_install software-properties-common || true
            if command -v sudo >/dev/null 2>&1; then
                sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                sudo apt-get update -y >/dev/null 2>&1 || true
            else
                add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                apt-get update -y >/dev/null 2>&1 || true
            fi
            for ver in 3.10 3.11 3.12; do
                if apt_python_install_attempt "$ver"; then
                    installed_via_apt=true
                    break
                fi
            done
        fi
        # Final fallback: install uv and use it to download CPython into
        # the user's home directory. This works even when apt is broken,
        # the deadsnakes PPA has stale signing keys, or there's no sudo.
        if ! PYTHON_BIN="$(pick_python)"; then
            warn "apt path failed. Falling back to uv (no apt, no sudo needed)..."
            if uv_py="$(install_python_via_uv)"; then
                PYTHON_BIN="$uv_py"
            fi
        fi
        if [ -z "${PYTHON_BIN:-}" ] || ! python_version_ok "$PYTHON_BIN"; then
            err "Could not find or install a Python >= 3.10 interpreter."
            err "Tried: existing PATH, apt python3.{10,11,12} (with deadsnakes PPA), uv."
            err "Install one manually and set PHENOFUSION_PYTHON=/path/to/python3.x"
            err "Examples:"
            err "  curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.11"
            err "  sudo apt install python3.10 python3.10-venv python3.10-distutils"
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
# 4b. Self-heal a broken opencv-python install.
#
# pyproject.toml now requires opencv-python-headless, but a venv built
# against an earlier version of the project (or any environment where a
# user did `pip install opencv-python` by hand) ships its OWN Qt plugins
# under cv2/qt/plugins/ and silently overwrites QT_QPA_PLATFORM_PLUGIN_PATH
# from inside `import cv2`. That makes the GUI abort with:
#     qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
#         "/.../site-packages/cv2/qt/plugins"
# Detect that situation and swap in the -headless wheel before launching.
###############################################################################

if python - <<'PY' 2>/dev/null
import importlib.util, os, sys
spec = importlib.util.find_spec("cv2")
if spec is None or spec.submodule_search_locations is None:
    sys.exit(1)
for loc in spec.submodule_search_locations:
    if os.path.isdir(os.path.join(loc, "qt", "plugins")):
        sys.exit(0)
sys.exit(1)
PY
then
    log "Replacing opencv-python (bundles Qt plugins) with opencv-python-headless..."
    python -m pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
    python -m pip install --upgrade "opencv-python-headless>=4.8.0"
fi

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
    [libfreetype.so.6]="libfreetype6"
    [libdbus-1.so.3]="libdbus-1-3"
    [libxkbcommon-x11.so.0]="libxkbcommon-x11-0"
    # Open3D's pybind*.so links OpenMP, which is not installed by default
    # on Ubuntu Server / WSL minimal. Without it `import open3d` aborts
    # with: OSError: libgomp.so.1: cannot open shared object file.
    [libgomp.so.1]="libgomp1"
    # Defensive entries for other libs the Open3D / PyQt stacks pull in
    # that occasionally go missing on stripped-down images.
    [libstdc++.so.6]="libstdc++6"
    [libgcc_s.so.1]="libgcc-s1"
    [libGLU.so.1]="libglu1-mesa"
    [libusb-1.0.so.0]="libusb-1.0-0"
    [libpng16.so.16]="libpng16-16"
    [libz.so.1]="zlib1g"
    # libQt5Core links these; libglib2.0-0 provides libgthread/libglib/libgobject.
    [libgthread-2.0.so.0]="libglib2.0-0"
    [libglib-2.0.so.0]="libglib2.0-0"
    [libgobject-2.0.so.0]="libglib2.0-0"
    [libgio-2.0.so.0]="libglib2.0-0"
)

PLUGIN_DIRS=()
for cand in "$VIRTUAL_ENV/lib/"python*"/site-packages/PyQt5/Qt5/plugins/platforms"; do
    [ -d "$cand" ] && PLUGIN_DIRS+=("$cand")
done

# Native libs we want to scan with ldd. We deliberately cast a wide net
# here -- it costs nothing to ldd a few extra .so files, and missing any
# one of them means a confusing crash on the first GUI launch.
SCAN_SOFILES=()
# Qt xcb platform plugin (the canonical "Could not load xcb" trigger).
for plugins_dir in "${PLUGIN_DIRS[@]:-}"; do
    for sofile in "$plugins_dir/libqxcb.so" "$plugins_dir/../../lib/libQt5XcbQpa.so.5"; do
        [ -f "$sofile" ] && SCAN_SOFILES+=("$sofile")
    done
done
# Every other PyQt5 Qt5 .so in the wheel -- libQt5Core needs libgthread,
# libQt5Widgets needs libQt5Gui needs libfontconfig, etc. Scanning the
# whole bundle picks up libgthread / libglib / libgobject that the xcb
# plugin alone does not pull in.
for cand in "$VIRTUAL_ENV/lib/"python*"/site-packages/PyQt5/Qt5/lib/"libQt5*.so.5; do
    [ -f "$cand" ] && SCAN_SOFILES+=("$cand")
done
# Open3D's native module (cpu/pybind*.so) pulls in libgomp.
for cand in "$VIRTUAL_ENV/lib/"python*"/site-packages/open3d/cpu/"pybind*.so; do
    [ -f "$cand" ] && SCAN_SOFILES+=("$cand")
done
# pyrealsense2's native module pulls in libusb on stripped-down hosts.
for cand in "$VIRTUAL_ENV/lib/"python*"/site-packages/pyrealsense2/"pyrealsense2*.so; do
    [ -f "$cand" ] && SCAN_SOFILES+=("$cand")
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

for sofile in "${SCAN_SOFILES[@]:-}"; do
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
