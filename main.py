"""
main.py
-------
PhenoFusion3D entry point.

On launch, runs a small self-check (active Python, pyrealsense2 version,
visible RealSense devices) and prints the result to stdout. If something
is clearly misconfigured -- e.g. the user launched the app from the wrong
venv, or pip pulled a pyrealsense2 build that dropped L515 support -- a
modal dialog explains what to do, *before* the user clicks Capture.

The self-check is read-only and never aborts startup; it just provides
faster, more honest diagnostics than waiting for the first capture click
to fail with "No Intel RealSense camera was found".
"""
import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from app.main_window import MainWindow


def _detect_wsl() -> bool:
    """Return True if we're running inside WSL.

    WSL is its own kind of trap for this project: the RealSense camera is
    a Windows USB device, and WSL2 does not pass USB through to Linux
    without an explicit usbipd-win bridge. Even with the right SDK
    version, an L515 plugged into the Windows host is invisible to a
    Python interpreter running under WSL.
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def _startup_self_check() -> dict:
    """Probe the active env for things that commonly break capture.

    Returns a dict with keys: python, cwd, is_wsl, rs_version,
    rs_l515_compatible, devices (list of dicts with name/fw/usb/serial),
    error.
    """
    info = {
        "python": sys.executable,
        "cwd": os.getcwd(),
        "is_wsl": _detect_wsl(),
        "rs_version": None,
        "rs_l515_compatible": None,
        "devices": [],
        "error": None,
    }
    print(f"[startup] Python:       {info['python']}", flush=True)
    print(f"[startup] Working dir:  {info['cwd']}", flush=True)
    if info["is_wsl"]:
        print("[startup] Environment:  WSL (Linux running on Windows host)", flush=True)

    try:
        import pyrealsense2 as rs
    except ImportError as e:
        info["error"] = f"pyrealsense2 not installed: {e}"
        print("[startup] pyrealsense2: NOT INSTALLED -- camera capture disabled", flush=True)
        return info

    try:
        from importlib.metadata import version
        info["rs_version"] = version("pyrealsense2")
    except Exception:
        info["rs_version"] = "unknown"

    # Releases >= 2.55 dropped L515 enumeration after Intel EOL'd the camera.
    try:
        major, minor = (int(p) for p in info["rs_version"].split(".")[:2])
        info["rs_l515_compatible"] = (major, minor) < (2, 55)
    except Exception:
        info["rs_l515_compatible"] = None

    print(f"[startup] pyrealsense2: {info['rs_version']}", flush=True)

    try:
        ds = list(rs.context().query_devices())
    except Exception as e:
        info["error"] = f"query_devices failed: {e}"
        print(f"[startup] WARNING: query_devices() failed: {e}", flush=True)
        return info

    for d in ds:
        def get(kind, default="?"):
            try:
                if d.supports(kind):
                    return d.get_info(kind)
            except Exception:
                pass
            return default
        info["devices"].append({
            "name":   get(rs.camera_info.name),
            "fw":     get(rs.camera_info.firmware_version),
            "usb":    get(rs.camera_info.usb_type_descriptor),
            "serial": get(rs.camera_info.serial_number),
        })

    print(f"[startup] RealSense devices visible: {len(info['devices'])}", flush=True)
    for e in info["devices"]:
        print(
            f"[startup]   - {e['name']} (fw {e['fw']}, USB {e['usb']}, sn {e['serial']})",
            flush=True,
        )

    return info


def _show_startup_warning_if_needed(info: dict) -> None:
    """Surface clear startup-time problems as a modal QMessageBox."""
    if info.get("is_wsl") and not info["devices"]:
        QMessageBox.warning(
            None,
            "PhenoFusion3D -- launched from WSL, camera will not work",
            "PhenoFusion3D is running inside WSL (Linux on the Windows host).\n\n"
            f"Active Python:   {info['python']}\n"
            f"pyrealsense2:    {info.get('rs_version') or 'not installed'}\n"
            f"RealSense devices visible: 0\n\n"
            "WSL2 does not pass USB devices through to Linux by default, so "
            "any RealSense camera plugged into the Windows host is invisible "
            "from inside WSL -- regardless of which pyrealsense2 version is "
            "installed.\n\n"
            "Launch PhenoFusion3D from Windows PowerShell instead:\n\n"
            "    cd C:\\COMP3500\\PhenoFusion3DFork\\Howard-sPhenoFusion3D\n"
            "    .\\venv\\Scripts\\Activate.ps1\n"
            "    python main.py\n\n"
            "Your prompt should change to '(venv) PS C:\\...>' before you "
            "launch. If it stays as 'user@HOST:/mnt/c/...$' you're still in "
            "WSL and the camera will not be detected."
        )
        return

    if info.get("error") and "pyrealsense2 not installed" in info["error"]:
        QMessageBox.warning(
            None,
            "PhenoFusion3D -- camera capture disabled",
            "pyrealsense2 is not installed in this Python environment.\n\n"
            f"Active Python:\n  {info['python']}\n\n"
            "You can still load existing RGB-D folders, but in-app capture "
            "from a RealSense camera will not work until you install "
            "pyrealsense2:\n\n"
            '    pip install -e ".[windows]"        (D400 / D500 series)\n'
            '    pip install -e ".[windows,l515]"   (Intel RealSense L515)'
        )
        return

    if info.get("rs_version") and not info["devices"]:
        l515_hint = ""
        if info.get("rs_l515_compatible") is False:
            l515_hint = (
                "\n\nDetected pyrealsense2 >= 2.55. This release dropped support "
                "for the Intel RealSense L515 (Intel EOL'd the camera in 2021). "
                "If you have an L515, install with the [l515] extras on Python "
                "3.10 or 3.11:\n\n"
                '    pip install -e ".[windows,l515]"'
            )
        QMessageBox.warning(
            None,
            "PhenoFusion3D -- no RealSense camera detected",
            "The RealSense SDK reports 0 devices on this machine.\n\n"
            f"Active Python:   {info['python']}\n"
            f"pyrealsense2:    {info['rs_version']}\n\n"
            "If you intend to capture, verify that the camera is plugged into "
            "a USB 3 port directly on the motherboard (no hubs), and that no "
            "other app (Intel RealSense Viewer, Windows Camera, Teams, etc.) "
            "is currently using it." + l515_hint
        )


def main():
    # IMPORTANT: create QApplication *before* importing pyrealsense2 (which
    # _startup_self_check() does). Qt initialises COM in STA on the GUI
    # thread; librealsense's Media Foundation backend initialises COM in
    # MTA. Whichever runs second loses with RPC_E_CHANGED_MODE
    # (0x80010106). Doing Qt first means the self-check piggybacks on the
    # already-initialised STA, which Media Foundation handles fine.
    app = QApplication(sys.argv)
    app.setApplicationName("PhenoFusion3D")
    app.setStyle("Fusion")

    info = _startup_self_check()
    _show_startup_warning_if_needed(info)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
