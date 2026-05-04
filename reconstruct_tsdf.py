#!/usr/bin/env python3
"""
TSDF volumetric reconstruction for the gantry plant dataset.

Replaces the ICP point-cloud merging approach with Open3D ScalableTSDFVolume
integration (KinectFusion-style).  Key differences vs ICP merging:

  ICP merging          TSDF integration
  ───────────────      ────────────────────────────────────────────────
  Points stack up      Every frame votes on shared voxels
  Noise accumulates    Weighted running average cancels noise
  No empty-space carve Free-space rays prune floating outliers
  Sparse point cloud   Dense triangulated mesh via marching cubes
  Gantry dominates     depth_min/depth_trunc can cut it at source

Coordinate-frame convention (read this before touching the extrinsic):
  Open3D native camera frame  : +x right,  +y DOWN, +z FORWARD  (depth direction)
  "Flipped" world frame used
   by ICP accumulation        : +x right,  +y UP,   +z BACKWARD
  Flip matrix F = diag(1,-1,-1,1) — note F = F⁻¹

  ICP saved poses P_N are camera→world in the FLIPPED frame.
  TSDF expects extrinsic = world→camera in the NATIVE camera frame.
  Therefore:
      extrinsic_N  =  F  @  inv(P_N)  @  F

  For the pure-translation case (all rotations stripped by ICP):
      P_N = [[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]]
      →  extrinsic_N = [[1,0,0,-tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]]
  Only the X-translation sign flips; Y and Z keep the same sign.

Prerequisite:
  Run reconstruct_icp_sequence.py first to generate output/poses/frame_N.txt files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from utils_rgbd import load_intrinsics, pair_frames, plant_mask_bgr

# Flip matrix  F = F⁻¹  —  converts between native and flipped camera frames
_FLIP = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


def _parse_frame_names(text: str) -> set[str]:
    """Parse comma-separated frame image stems/names used for debug mask dumps."""
    out: set[str] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(Path(part).stem)
    return out


def _crop_point_cloud(
    pcd: o3d.geometry.PointCloud,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> o3d.geometry.PointCloud:
    """Crop a point cloud using an axis-aligned box in TSDF/native world coordinates."""
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([x_min, y_min, z_min], dtype=np.float64),
        max_bound=np.array([x_max, y_max, z_max], dtype=np.float64),
    )
    return pcd.crop(bbox)


def _plant_color_mask_rgb_float(colors: np.ndarray) -> np.ndarray:
    """Return mask for plant-coloured RGB float colours in [0, 1]."""
    if len(colors) == 0:
        return np.zeros((0,), dtype=bool)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    leaf = (
        (hsv[:, 0] >= 30)
        & (hsv[:, 0] <= 95)
        & (hsv[:, 1] >= 35)
        & (hsv[:, 2] >= 25)
    )
    stem = (
        (hsv[:, 0] >= 8)
        & (hsv[:, 0] <= 32)
        & (hsv[:, 1] >= 45)
        & (hsv[:, 2] >= 25)
        & (hsv[:, 2] <= 210)
    )
    return leaf | stem


def _filter_point_cloud_by_plant_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Keep only plant-coloured points from a cropped TSDF point cloud."""
    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        return pcd
    mask = _plant_color_mask_rgb_float(colors)
    return pcd.select_by_index(np.flatnonzero(mask).tolist())


def _crop_mesh_by_vertices(
    mesh: o3d.geometry.TriangleMesh,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> o3d.geometry.TriangleMesh:
    """Remove mesh vertices outside the plant crop box."""
    out = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(out.vertices)
    if len(vertices) == 0:
        return out
    keep = (
        (vertices[:, 0] >= x_min)
        & (vertices[:, 0] <= x_max)
        & (vertices[:, 1] >= y_min)
        & (vertices[:, 1] <= y_max)
        & (vertices[:, 2] >= z_min)
        & (vertices[:, 2] <= z_max)
    )
    vertex_colors = np.asarray(out.vertex_colors)
    if len(vertex_colors) == len(vertices):
        keep &= _plant_color_mask_rgb_float(vertex_colors)
    out.remove_vertices_by_mask(~keep)
    out.remove_degenerate_triangles()
    out.remove_duplicated_triangles()
    out.remove_duplicated_vertices()
    out.remove_non_manifold_edges()
    out.compute_vertex_normals()
    return out


def _pose_to_extrinsic(pose_cam2world_flipped: np.ndarray) -> np.ndarray:
    """
    Convert ICP camera-to-world pose (in flipped frame) to the world-to-camera
    extrinsic expected by Open3D TSDF integration (in native camera frame).

        extrinsic  =  F  @  inv(pose)  @  F
    """
    inv_pose = np.linalg.inv(pose_cam2world_flipped)
    return _FLIP @ inv_pose @ _FLIP


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="TSDF volumetric reconstruction using saved ICP poses."
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "main" / "test_plant_rs13_1",
        help="Dataset root (same as used for ICP).",
    )
    ap.add_argument(
        "--voxel",
        type=float,
        default=0.003,
        help="TSDF voxel side length in metres (default 3 mm).  "
             "Smaller -> more detail, more RAM.  Try 0.002 for 2 mm.",
    )
    ap.add_argument(
        "--sdf-trunc",
        type=float,
        default=None,
        help="TSDF truncation distance in metres (default 5 x --voxel).  "
             "Smaller -> sharper surfaces; too small -> incomplete fusion.",
    )
    ap.add_argument(
        "--depth-trunc",
        type=float,
        default=2.3,
        help="Maximum depth to integrate in metres (default 2.3 m).  "
             "Plant is ~1.7-2.1 m; keeping this tight excludes far background.",
    )
    ap.add_argument(
        "--depth-min",
        type=float,
        default=0.8,
        help="Minimum depth to integrate in metres (default 0.8 m).  "
             "Zeros out depth pixels closer than this before integration, "
             "removing the close gantry boom and camera mount.",
    )
    ap.add_argument(
        "--no-erode",
        action="store_true",
        help="Skip depth-edge erosion (flying-pixel suppression is on by default).",
    )
    ap.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip triangle-mesh extraction; save only the TSDF point cloud.",
    )
    ap.add_argument(
        "--poisson",
        action="store_true",
        help="Run Poisson surface reconstruction on the TSDF point cloud after "
             "mesh extraction (smoother but slower; writes tsdf_poisson.ply).",
    )
    ap.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Octree depth for Poisson reconstruction (default 9).",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="Subsampling step - must match the step used when poses were saved.",
    )
    ap.add_argument(
        "--color-mask",
        action="store_true",
        help=(
            "Zero out depth for non-plant pixels before integration using an HSV "
            "leaf/stem mask.  This removes most gray metal/table pixels without "
            "relying on depth-range thresholds."
        ),
    )
    ap.add_argument(
        "--write-debug-masks",
        action="store_true",
        help="Save HSV plant masks for representative frames under output/debug_masks/.",
    )
    ap.add_argument(
        "--debug-mask-frames",
        type=str,
        default="200,240,290,320,360",
        help=(
            "Comma-separated RGB image stems/names for mask dumps when "
            "--write-debug-masks is set. Default includes rgb/290.png."
        ),
    )
    ap.add_argument(
        "--plant-only-crop",
        action="store_true",
        help=(
            "Also save plant-only cropped outputs using the world-space crop box "
            "defined by --crop-* arguments."
        ),
    )
    ap.add_argument("--crop-x-min", type=float, default=-1.30)
    ap.add_argument("--crop-x-max", type=float, default=0.25)
    ap.add_argument("--crop-y-min", type=float, default=0.55)
    ap.add_argument("--crop-y-max", type=float, default=1.40)
    ap.add_argument("--crop-z-min", type=float, default=1.75)
    ap.add_argument("--crop-z-max", type=float, default=2.35)
    args = ap.parse_args()

    sdf_trunc = args.sdf_trunc if args.sdf_trunc is not None else args.voxel * 5.0

    data_dir = args.data.expanduser().resolve()
    output_dir = data_dir / "output"
    pose_dir = output_dir / "poses"
    intrinsic_path = data_dir / "kdc_intrinsics.txt"

    if not pose_dir.exists():
        raise SystemExit(
            f"Pose directory not found: {pose_dir}\n"
            "Run reconstruct_icp_sequence.py first to generate poses."
        )

    intr = load_intrinsics(intrinsic_path)
    K = intr["K"]
    W, H = intr["width"], intr["height"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    pairs_all = pair_frames(data_dir)

    # Collect (pair_index, frame_idx, pose_path) for all saved poses
    frame_entries: list[tuple[int, int, Path]] = []
    for pose_path in sorted(pose_dir.glob("frame_*.txt")):
        frame_idx = int(pose_path.stem.split("_")[1])
        pair_idx = frame_idx  # with step=1 they coincide; for step>1 frame_idx = pair_idx*step
        if pair_idx >= len(pairs_all):
            continue
        frame_entries.append((pair_idx, frame_idx, pose_path))
    frame_entries.sort(key=lambda x: x[1])

    if not frame_entries:
        raise SystemExit("No pose files found — cannot run TSDF integration.")

    debug_mask_names = _parse_frame_names(args.debug_mask_frames)
    debug_mask_dir = output_dir / "debug_masks"
    if args.write_debug_masks:
        debug_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tsdf] data_dir      = {data_dir}")
    print(f"[tsdf] frames        = {len(frame_entries)}")
    print(f"[tsdf] voxel_length  = {args.voxel*1000:.1f} mm")
    print(f"[tsdf] sdf_trunc     = {sdf_trunc*1000:.1f} mm")
    print(f"[tsdf] depth range   = [{args.depth_min:.2f}, {args.depth_trunc:.2f}] m")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    depth_min_mm = int(args.depth_min * 1000)

    for pair_idx, frame_idx, pose_path in tqdm(frame_entries, desc="TSDF"):
        rgb_path, depth_path = pairs_all[pair_idx]

        color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_u16 is None:
            tqdm.write(f"[tsdf] WARNING: cannot read frame {frame_idx}, skipping")
            continue

        # Suppress depth edge flying-pixels (same as ICP pipeline)
        if not args.no_erode:
            depth_f32 = depth_u16.astype(np.float32)
            sx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.abs(sx) + np.abs(sy)
            disc_mask = (edge_mag > 200).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            disc_dilated = cv2.dilate(disc_mask, kernel, iterations=1)
            depth_u16 = depth_u16.copy()
            depth_u16[disc_dilated > 0] = 0

        # Depth-min mask: zero out pixels closer than depth_min to exclude gantry
        if depth_min_mm > 0:
            close_mask = (depth_u16 > 0) & (depth_u16 < depth_min_mm)
            depth_u16 = depth_u16.copy()
            depth_u16[close_mask] = 0

        # Color mask: keep depth only where the pixel is plant-like.
        # This includes green leaves and brown stems while rejecting low-saturation
        # gray gantry/table pixels.
        if args.color_mask:
            plant_mask = plant_mask_bgr(color_bgr)
            if args.write_debug_masks and rgb_path.stem in debug_mask_names:
                overlay = color_bgr.copy()
                overlay[plant_mask == 0] = (overlay[plant_mask == 0] * 0.25).astype(np.uint8)
                cv2.imwrite(str(debug_mask_dir / f"{rgb_path.stem}_mask.png"), plant_mask)
                cv2.imwrite(str(debug_mask_dir / f"{rgb_path.stem}_overlay.png"), overlay)
            depth_u16 = depth_u16.copy()
            depth_u16[plant_mask == 0] = 0

        # Open3D uses RGB, OpenCV loads BGR
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_u16.astype(np.uint16))

        # depth_scale=1000 converts mm → metres;  depth_trunc clips far background
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Load saved ICP pose (camera→world in flipped frame) and convert to
        # world→camera in native Open3D camera frame for TSDF integration.
        pose = np.loadtxt(str(pose_path))
        extrinsic = _pose_to_extrinsic(pose)

        volume.integrate(rgbd, o3d_intrinsic, extrinsic)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract and save point cloud (always) ---
    pcd_path = output_dir / "tsdf_pcd.ply"
    print("[tsdf] Extracting point cloud ...")
    tsdf_pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(str(pcd_path), tsdf_pcd)
    print(f"[tsdf] Point cloud ({len(tsdf_pcd.points)} pts) -> {pcd_path}")

    if args.plant_only_crop:
        plant_pcd_path = output_dir / "tsdf_plant_only.ply"
        plant_pcd = _crop_point_cloud(
            tsdf_pcd,
            x_min=args.crop_x_min,
            x_max=args.crop_x_max,
            y_min=args.crop_y_min,
            y_max=args.crop_y_max,
            z_min=args.crop_z_min,
            z_max=args.crop_z_max,
        )
        plant_pcd = _filter_point_cloud_by_plant_color(plant_pcd)
        o3d.io.write_point_cloud(str(plant_pcd_path), plant_pcd)
        print(f"[tsdf] Plant-only point cloud ({len(plant_pcd.points)} pts) -> {plant_pcd_path}")

    # --- Extract triangle mesh ---
    if not args.no_mesh:
        mesh_path = output_dir / "tsdf_mesh.ply"
        print("[tsdf] Extracting triangle mesh ...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(
            f"[tsdf] Mesh ({len(mesh.vertices)} vertices, "
            f"{len(mesh.triangles)} triangles) -> {mesh_path}"
        )

        if args.plant_only_crop:
            plant_mesh_path = output_dir / "tsdf_plant_only_mesh.ply"
            plant_mesh = _crop_mesh_by_vertices(
                mesh,
                x_min=args.crop_x_min,
                x_max=args.crop_x_max,
                y_min=args.crop_y_min,
                y_max=args.crop_y_max,
                z_min=args.crop_z_min,
                z_max=args.crop_z_max,
            )
            o3d.io.write_triangle_mesh(str(plant_mesh_path), plant_mesh)
            print(
                f"[tsdf] Plant-only mesh ({len(plant_mesh.vertices)} vertices, "
                f"{len(plant_mesh.triangles)} triangles) -> {plant_mesh_path}"
            )

    # --- Optional Poisson pass ---
    if args.poisson:
        print("[tsdf] Running Poisson surface reconstruction ...")
        # TSDF point cloud already carries gradient-derived normals — much better
        # than estimating normals on a sparse ICP cloud.
        poisson_pcd = tsdf_pcd
        if not poisson_pcd.has_normals():
            poisson_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=args.voxel * 4, max_nn=30
                )
            )
            poisson_pcd.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 0.0])
            )

        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            poisson_pcd, depth=args.poisson_depth, width=0, scale=1.1, linear_fit=False
        )
        # Trim low-density vertices that Poisson hallucinated outside observed volume
        density_threshold = np.quantile(np.asarray(densities), 0.05)
        vertices_to_remove = np.asarray(densities) < density_threshold
        poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
        poisson_mesh.compute_vertex_normals()

        poisson_path = output_dir / "tsdf_poisson.ply"
        o3d.io.write_triangle_mesh(str(poisson_path), poisson_mesh)
        print(
            f"[tsdf] Poisson mesh ({len(poisson_mesh.vertices)} vertices, "
            f"{len(poisson_mesh.triangles)} triangles) -> {poisson_path}"
        )


if __name__ == "__main__":
    main()
