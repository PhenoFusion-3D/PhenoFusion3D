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

from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from pathlib import Path

from app.main_window import MainWindow

from processing.canopy import CanopyReconstructionConfig, reconstruct_canopy
from processing.reconstructor import ReconstructionConfig, reconstruct_sequence


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




def _has_rgbd_frames(path: Path) -> bool:
    return any(path.glob("rgb_*.png")) and any(path.glob("depth_*.png"))


def _discover_records(input_path: str) -> list[Path]:
    root = Path(input_path)
    if _has_rgbd_frames(root):
        return [root]
    if not root.exists():
        return [root]
    records = sorted(path for path in root.iterdir() if path.is_dir() and _has_rgbd_frames(path))
    return records or [root]


def _batch_output_dir(base_output: str | None, input_path: str, record: Path, total_records: int, suffix: str) -> str | None:
    if total_records == 1:
        return base_output
    base = Path(base_output) if base_output else Path(input_path) / suffix
    return str(base / record.name)


def _batch_mask_dir(mask_dir: str | None, record: Path, total_records: int) -> str | None:
    if mask_dir is None or total_records == 1:
        return mask_dir
    base = Path(mask_dir)
    candidate = base / record.name
    return str(candidate if candidate.exists() else base)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run local RGB-D 3D reconstruction without the plant-scanning hardware stack."
    )
    parser.add_argument("--mode", choices=("rgbd", "canopy"), default="canopy", help="Pick the standard RGB-D stitcher or the plant-focused canopy reconstructor.")
    parser.add_argument("--input", required=True, help="Folder containing rgb_*.png, depth_*.png and intrinsics.")
    parser.add_argument("--camera-id", default="", help="Optional camera suffix if data is inside camera_<id>.")
    parser.add_argument("--step-size", type=int, default=8, help="Use every Nth frame to keep local runs practical.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on sampled frames.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this sampled frame index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this sampled frame index.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth divisor that converts raw depth to metres.")
    parser.add_argument("--depth-trunc", type=float, default=None, help="Discard points farther than this many metres. Defaults to auto-estimated.")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size used for cleanup and registration.")
    parser.add_argument("--min-fitness", type=float, default=0.01, help="Reject weak registrations after the warm-up frames.")
    parser.add_argument("--output-dir", default=None, help="Output folder. Defaults to <input>/reconstruction_local.")
    parser.add_argument("--green-filter", action="store_true", help="Keep only green-ish points in the final merged cloud.")
    parser.add_argument("--mask-dir", default=None, help="Optional directory containing mask_*.png for canopy mode.")
    parser.add_argument("--reference-token", type=int, default=None, help="Optional rgb/depth token to anchor canopy fusion.")
    parser.add_argument("--min-mask-area", type=int, default=180000, help="Ignore tiny masks when canopy mode selects frames.")
    parser.add_argument("--coverage-threshold", type=int, default=1, help="Canopy pixels must be supported by at least this many aligned frames.")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Vertical exaggeration for canopy mode. Keep 1.0 for metric output.")
    parser.add_argument("--no-auto-mask", action="store_true", help="Disable automatic green-leaf masks when no mask directory is present.")
    parser.add_argument("--canvas-padding", type=int, default=48, help="Extra pixels around the aligned canopy fusion canvas.")
    return parser



def _gui_main() -> None:
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



def _cli_main() -> None:
    args = build_parser().parse_args()
    records = _discover_records(args.input)
    if args.mode == "canopy":
        results = []
        for record in records:
            config = CanopyReconstructionConfig(
                mask_dir=_batch_mask_dir(args.mask_dir, record, len(records)),
                max_frames=args.max_frames or 9,
                min_mask_area=args.min_mask_area,
                reference_token=args.reference_token,
                coverage_threshold=args.coverage_threshold,
                z_scale=args.z_scale,
                output_dir=_batch_output_dir(args.output_dir, args.input, record, len(records), "canopy_batch"),
                auto_mask=not args.no_auto_mask,
                canvas_padding=args.canvas_padding,
            )
            results.append(reconstruct_canopy(record, config=config))

        print("Canopy reconstruction finished.")
        if len(results) > 1:
            print(f"Processed records: {len(results)}")
        for result in results:
            print(f"Input: {result.record_path}")
            print(f"Output directory: {result.output_dir}")
            print(f"Point cloud: {result.point_cloud_path}")
            print(f"Mesh: {result.mesh_path}")
            print(f"Interactive viewer: {result.viewer_path}")
            print(f"Masked RGB: {result.masked_rgb_path}")
            print(f"Summary: {result.summary_path}")
            print(f"Frames: {result.frames_used}/{result.frames_available} mask-backed frames")
            print(f"Reference frame token: {result.reference_token}")
            print(f"Final point count: {result.final_point_count}")
            print(f"Final triangle count: {result.final_triangle_count}")
        return

    results = []
    for record in records:
        config = ReconstructionConfig(
            camera_id=args.camera_id,
            step_size=args.step_size,
            max_frames=args.max_frames,
            start_index=args.start_index,
            end_index=args.end_index,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
            voxel_size=args.voxel_size,
            min_fitness=args.min_fitness,
            output_dir=_batch_output_dir(args.output_dir, args.input, record, len(records), "reconstruction_batch"),
            green_only=args.green_filter,
        )
        results.append(reconstruct_sequence(record, config=config))

    print("Reconstruction finished.")
    if len(results) > 1:
        print(f"Processed records: {len(results)}")
    for result in results:
        print(f"Input: {result.record_path}")
        print(f"Output directory: {result.output_dir}")
        print(f"Merged point cloud: {result.merged_point_cloud_path}")
        print(f"Poses: {result.pose_dir}")
        print(f"Summary: {result.summary_path}")
        print(
            "Frames: "
            f"{result.frames_registered}/{result.frames_selected} sampled "
            f"({result.frames_total} total in dataset)"
        )
        print(f"Final point count: {result.final_point_count}")



def main() -> None:
    if len(sys.argv) <= 1:
        _gui_main()
    else:
        _cli_main()


if __name__ == "__main__":
    main()
