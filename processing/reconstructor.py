from __future__ import annotations

import os
import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d

from file_io.loader import (
    get_default_intrinsics,
    load_image_pairs,
    load_intrinsics,
    load_session_json,
)
from processing.rgbd import rgbd2pcd, plant_mask_bgr
from processing.icp import color_icp, point_to_plane_icp
from processing.utils import clean_pcd, clean_pcd_for_registration, green_filter_pcd
from processing.registration_agent import (
    RegistrationAgent,
    AgentConfig,
    apply_strategy,
)


class Reconstructor:
    """
    Sequential RGB-D point cloud reconstruction.

    Two operating modes selected by `use_known_poses`:

    ICP mode (use_known_poses=False, default):
        Registers each frame against the previous with colour-assisted ICP,
        accumulating a merged point cloud. Works for any camera motion.

    Known-pose / TSDF mode (use_known_poses=True):
        Skips ICP entirely. Computes camera poses from gantry kinematics
        (constant-velocity linear translation) and integrates all frames
        into an Open3D ScalableTSDFVolume. Produces clean, hole-filled
        surfaces and is much more robust when ICP is degenerate (e.g. flat
        scenes viewed from directly above).

    In both modes the class is designed to run inside a QThread worker --
    all UI interaction happens via the on_frame and on_complete callbacks.
    """

    def __init__(
        self,
        pairs,
        K,
        dist=None,
        step_size=1,
        depth_scale=1000.0,
        depth_trunc=3.5,
        voxel_size=0.005,
        max_iter=50,
        gantry_step_m=0.0,
        gantry_axis=0,
        depth_min_mm=0,
        erode=False,
        inpaint=False,
        use_known_poses=False,
        tsdf_voxel_m=0.003,
        min_fitness=0.3,
        max_rmse=0.015,
        save_path=None,
        on_frame=None,
        on_complete=None,
        bbox=None,
        mask_background=False,
        bg_sat_thresh=40,
        agent_config=None,
        allow_rotation=False,
        plant_icp=False,
    ):
        """
        Args:
            pairs           : list of (rgb_path, depth_path) tuples (already stepped)
            K               : 3x3 intrinsic matrix (np.ndarray)
            dist            : distortion coefficients list, or None
            step_size       : kept for metadata; loader handles stepping
            depth_scale     : mm->metres divisor (1000 for RealSense, 1.0 for ICL-NUIM)
            depth_trunc     : discard depth beyond this many metres
            voxel_size      : voxel size for ICP radius and final-output downsampling
            max_iter        : max ICP iterations per frame pair (ICP mode only)
            gantry_step_m   : camera translation per PAIR in metres (pre-multiplied by
                              sampling step). Used as ICP init seed (ICP mode) or as
                              kinematic pose step (known-pose mode).
            gantry_axis     : camera-space axis the gantry moves along: 0=X, 1=Y
                              (determined by calibrate_gantry.py)
            depth_min_mm    : near clip for raw depth (mm); 0 disables
            erode           : shrink valid depth mask (flying pixels) -- ICP mode
            inpaint         : fill interior holes in depth -- ICP mode
            use_known_poses : if True, use TSDF + kinematic poses instead of ICP
            tsdf_voxel_m    : TSDF voxel size in metres (known-pose mode only)
            save_path       : directory to write intermediate PLY files, or None
            on_frame        : callback(frame_idx, total, merged_pcd, fitness, rmse, status)
            on_complete     : callback(final_pcd, succeed_list, fail_list)
            bbox            : optional [x1, y1, x2, y2] crop passed to rgbd2pcd
            mask_background : zero low-saturation white/grey background depth
            bg_sat_thresh   : HSV saturation threshold for background masking
            allow_rotation  : if True, keep the full ICP rotation; if False (default),
                              strip the rotation component and keep only translation.
                              False is correct for a linear-translation gantry and
                              prevents systematic rotation drift from smearing the cloud.
            plant_icp       : if True, apply plant_mask_bgr to depth before building the
                              ICP source cloud so registration is driven by leaf/stem
                              geometry instead of gantry metal.
        """
        self.pairs           = pairs
        self.K               = K
        self.dist            = dist
        self.step_size       = step_size
        self.depth_scale     = depth_scale
        self.depth_trunc     = depth_trunc
        self.voxel_size      = voxel_size
        self.max_iter        = max_iter
        self.gantry_step_m   = gantry_step_m
        self.gantry_axis     = gantry_axis
        self.depth_min_mm    = depth_min_mm
        self.erode           = erode
        self.inpaint         = inpaint
        self.use_known_poses = use_known_poses
        self.tsdf_voxel_m    = tsdf_voxel_m
        self.min_fitness     = min_fitness
        self.max_rmse        = max_rmse
        self.save_path       = save_path
        self.on_frame        = on_frame
        self.on_complete     = on_complete
        self.bbox            = bbox
        self.mask_background = mask_background
        self.bg_sat_thresh   = bg_sat_thresh
        self.allow_rotation  = allow_rotation
        self.plant_icp       = plant_icp

        # Registration agent (ICP mode only). Uses the existing min_fitness /
        # max_rmse as absolute floors when no explicit config is supplied.
        if agent_config is None:
            agent_config = AgentConfig(
                floor_min_fitness=min_fitness,
                floor_max_rmse=max_rmse,
            )
        self.agent_config = agent_config
        self.agent        = RegistrationAgent(agent_config)

        self._stop_flag    = False
        self.reference_pcd = None
        self.succeed_list  = []
        self.fail_list     = []

        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

    def stop(self):
        """Signal the run loop to halt cleanly after the current frame."""
        self._stop_flag = True
        print('[reconstructor] Stop requested.')

    def run(self):
        """
        Main entry point. Dispatches to the appropriate mode.
        Call this from QThread.run() or directly from a test script.
        """
        self._stop_flag   = False
        self.succeed_list = []
        self.fail_list    = []
        self.reference_pcd = o3d.geometry.PointCloud()
        # Fresh agent state per run so successive runs don't share history.
        self.agent = RegistrationAgent(self.agent_config)

        if self.use_known_poses:
            return self._run_known_pose_tsdf()
        else:
            return self._run_icp()

    # ------------------------------------------------------------------
    # Mode A: Known-pose TSDF integration
    # ------------------------------------------------------------------

    def _apply_background_mask(self, color_rgb, depth_img):
        """Zero low-saturation white/grey background pixels in a depth image."""
        if not self.mask_background:
            return depth_img
        hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
        bg = (hsv[:, :, 1] < self.bg_sat_thresh).astype(np.uint8)
        bg = cv2.erode(bg, np.ones((3, 3), np.uint8), iterations=1)
        depth_img[bg > 0] = 0
        return depth_img

    def _run_known_pose_tsdf(self):
        """
        Integrate all frames into an Open3D ScalableTSDFVolume using
        camera poses derived from gantry kinematics.

        gantry_step_m is the 3D translation per consecutive PAIR (already
        multiplied by the sampling step by the caller).
        """
        total = len(self.pairs)
        print(f'[reconstructor] Known-pose TSDF: {total} frames, '
              f'step={self.gantry_step_m*1000:.2f}mm, '
              f'axis={self.gantry_axis}, '
              f'voxel={self.tsdf_voxel_m*1000:.1f}mm')

        # sdf_trunc = 4 x voxel_length. At ~2.8 m the D405 depth noise is
        # roughly centimetre-scale, so TSDF voxels must be sized to the
        # sensor noise floor rather than sub-millimetre close-range specs.
        sdf_trunc = self.tsdf_voxel_m * 4

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.tsdf_voxel_m,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        # Build PinholeCameraIntrinsic from first image shape + K matrix
        K_mat = np.array(self.K, dtype=np.float64)
        first_bgr = cv2.imread(self.pairs[0][0])
        if first_bgr is None:
            raise RuntimeError(f'Cannot read first frame: {self.pairs[0][0]}')
        img_h, img_w = first_bgr.shape[:2]

        # Adjust intrinsics for bbox crop if used
        cx = float(K_mat[0, 2])
        cy = float(K_mat[1, 2])
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            cx     -= x1
            cy     -= y1
            img_w   = x2 - x1
            img_h   = y2 - y1

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=img_w, height=img_h,
            fx=float(K_mat[0, 0]),
            fy=float(abs(K_mat[1, 1])),
            cx=cx, cy=cy,
        )

        # Pre-compute undistortion maps (if needed)
        map1, map2 = None, None
        if self.dist is not None and any(d != 0.0 for d in self.dist):
            dist_arr = np.array(self.dist, dtype=np.float64)
            raw_h, raw_w = first_bgr.shape[:2]
            map1, map2 = cv2.initUndistortRectifyMap(
                K_mat, dist_arr, None, K_mat, (raw_w, raw_h), cv2.CV_32FC1
            )

        depth_max_mm = int(self.depth_trunc * self.depth_scale)

        session = load_session_json(os.path.dirname(self.pairs[0][0]))
        frame_positions = (session or {}).get('frame_positions', {}) or {}
        pos_0 = None
        if frame_positions:
            print(
                f'[reconstructor] Using session.json gantry positions '
                f'for {len(frame_positions)} frames.'
            )

        def _session_position_for_frame(rgb_path):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            candidates = [stem]
            for prefix in ('rgb_', 'depth_'):
                if stem.startswith(prefix):
                    candidates.append(stem[len(prefix):])
            for key in candidates:
                if key in frame_positions:
                    return float(frame_positions[key])
            return None

        for i, (rgb_path, depth_path) in enumerate(self.pairs):
            if self._stop_flag:
                print(f'[reconstructor] Stopped at frame {i}.')
                self._emergency_save()
                break

            # Load
            color_bgr = cv2.imread(rgb_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if color_bgr is None or depth_raw is None:
                print(f'[reconstructor] WARNING: Cannot read frame {i}, skipping.')
                self.fail_list.append({'frame': i, 'reason': 'imread failed'})
                continue

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            # Undistort (depth: nearest-neighbour to avoid value corruption)
            if map1 is not None:
                color_rgb = cv2.undistort(
                    color_rgb, K_mat, np.array(self.dist, dtype=np.float64)
                )
                depth_raw = cv2.remap(depth_raw, map1, map2, cv2.INTER_NEAREST)

            # Optional bbox crop
            if self.bbox is not None:
                x1, y1, x2, y2 = self.bbox
                color_rgb = color_rgb[y1:y2, x1:x2]
                depth_raw = depth_raw[y1:y2, x1:x2]

            # Depth range masking
            depth_masked = depth_raw.astype(np.uint16).copy()
            depth_masked[depth_masked > depth_max_mm] = 0
            if self.depth_min_mm > 0:
                depth_masked[(depth_masked > 0) & (depth_masked < self.depth_min_mm)] = 0
            depth_masked = self._apply_background_mask(color_rgb, depth_masked)

            valid_depth_count = int(np.count_nonzero(depth_masked))
            depth_validity = float(valid_depth_count) / float(depth_masked.size)
            decision = self.agent.judge_tsdf_frame(depth_validity, valid_depth_count)
            if decision.action == 'reject':
                print(f'[reconstructor] WARNING: Frame {i} skipped: {decision.reason}')
                self.fail_list.append({
                    'frame': i,
                    'reason': decision.reason,
                    'fitness': depth_validity,
                    'rmse': None,
                    'note': 'known-pose TSDF depth gate',
                })
                self._fire_on_frame(i, total, self.reference_pcd,
                                    depth_validity, float('nan'), 'SKIPPED')
                continue

            # Build RGBD image (Open3D needs C-contiguous arrays)
            o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
            o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_masked))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_scale=self.depth_scale,
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            # Kinematic camera pose: camera-to-world for frame i.
            # Prefer encoder positions saved by ROS capture; fall back to the
            # configured constant step when old datasets have no session.json.
            T_c2w = np.eye(4)
            pos_i = _session_position_for_frame(rgb_path) if frame_positions else None
            if pos_i is not None:
                if pos_0 is None:
                    pos_0 = pos_i
                T_c2w[self.gantry_axis, 3] = pos_i - pos_0
            else:
                T_c2w[self.gantry_axis, 3] = i * self.gantry_step_m

            # TSDF integrate() expects world-to-camera (inverse of camera pose)
            extrinsic = np.linalg.inv(T_c2w)

            volume.integrate(rgbd, intrinsic, extrinsic)

            status = 'WARN' if decision.action == 'warn' else 'INTEGRATED'
            self.succeed_list.append({
                'frame': i,
                'fitness': depth_validity,
                'rmse': None,
                'status': status,
                'note': f'known-pose TSDF; {decision.reason}',
            })
            # Pass empty cloud during integration (full cloud only at the end).
            # TSDF mode has no ICP fitness/RMSE, so the fitness slot carries
            # per-frame depth validity instead of fake registration metrics.
            self._fire_on_frame(i, total, self.reference_pcd,
                                depth_validity, float('nan'), status)

            if i % 10 == 0 or i == total - 1:
                print(f'[reconstructor] TSDF {i + 1:4d}/{total}')

        print('[reconstructor] Extracting point cloud from TSDF volume...')
        self.reference_pcd = volume.extract_point_cloud()
        pts_before = len(self.reference_pcd.points)

        # TSDF already fused depth noise; use the lenient final-output cleaner.
        # The registration cleaner is too aggressive for thin plant geometry.
        self.reference_pcd = clean_pcd(
            self.reference_pcd,
            nb_neighbors=30,
            std_ratio=2.0,
            voxel_size=self.tsdf_voxel_m,
        )
        pts_after = len(self.reference_pcd.points)
        print(f'[reconstructor] Outlier removal: {pts_before:,} -> {pts_after:,} pts')

        print(f'[reconstructor] TSDF complete. '
              f'Points: {pts_after:,}  '
              f'success={len(self.succeed_list)}  fail={len(self.fail_list)}')

        self._save_intermediate()
        if self.on_complete:
            self.on_complete(self.reference_pcd, self.succeed_list, self.fail_list)

        return self.reference_pcd, self.succeed_list, self.fail_list

    # ------------------------------------------------------------------
    # Mode B: ICP-based registration (fixed)
    # ------------------------------------------------------------------

    def _run_icp(self):
        """
        Sequential frame-to-frame colour ICP.

        Enhancements over the original pipeline:
        - allow_rotation=False strips the rotation component from each ICP
          result, keeping only translation (correct for linear gantry, prevents
          systematic rotation drift from smearing the plant cloud).
        - plant_icp=True applies plant_mask_bgr to the depth image before
          building the source cloud so ICP is driven by plant geometry rather
          than the gantry metal structure.
        - Pose saving: each accepted frame writes its cameraÔåÆworld transform to
          output/poses/frame_{i}.txt for downstream TSDF / NeRF pipelines.
        - CSV log: fitness, rmse, status written to output/icp_log.csv.
        - Stable-reference fallback: when ICP fails, the last known-good
          transform is extrapolated using recent velocity to fill the gap.
        """
        total          = len(self.pairs)
        last_transform = np.eye(4)
        stable_history: list[np.ndarray] = [last_transform.copy()]
        target         = None
        csv_rows: list[list[object]] = []

        # Set up pose and log directories under save_path (if provided)
        pose_dir = None
        log_csv_path = None
        if self.save_path:
            pose_dir = os.path.join(self.save_path, 'poses')
            os.makedirs(pose_dir, exist_ok=True)
            log_csv_path = os.path.join(self.save_path, 'icp_log.csv')

        print(f'[reconstructor] ICP mode: {total} frames'
              f'  allow_rotation={self.allow_rotation}'
              f'  plant_icp={self.plant_icp}')

        for i, (rgb_path, depth_path) in enumerate(self.pairs):

            if self._stop_flag:
                print(f'[reconstructor] Stopped at frame {i}.')
                self._emergency_save()
                break

            # Load images
            color = cv2.imread(rgb_path)
            if color is None:
                print(f'[reconstructor] WARNING: Could not read {rgb_path}, skipping.')
                self.fail_list.append({'frame': i, 'reason': 'imread failed'})
                continue
            color_bgr = color  # keep BGR for plant_mask_bgr
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f'[reconstructor] WARNING: Could not read {depth_path}, skipping.')
                self.fail_list.append({'frame': i, 'reason': 'imread failed'})
                continue

            # Plant-ICP: zero non-plant depth before building source cloud so
            # ICP is anchored on leaf/stem geometry, not the gantry structure.
            depth_for_icp = depth
            if self.plant_icp:
                plant_mask = plant_mask_bgr(color_bgr)
                depth_for_icp = depth.copy()
                depth_for_icp[plant_mask == 0] = 0

            # Convert to point cloud
            try:
                source = rgbd2pcd(
                    color, depth_for_icp, self.K,
                    dist=self.dist,
                    bbox=self.bbox,
                    depth_scale=self.depth_scale,
                    depth_trunc=self.depth_trunc,
                    depth_min_mm=self.depth_min_mm,
                    erode=self.erode,
                    inpaint=self.inpaint,
                    mask_background=self.mask_background,
                    bg_sat_thresh=self.bg_sat_thresh,
                )
                # Outlier removal WITHOUT voxel downsampling before ICP.
                # Voxel downsampling before ICP kills sub-voxel displacement
                # signal (gantry moves ~1-7mm but voxels are 8mm).
                source = clean_pcd_for_registration(source)
            except Exception as e:
                print(f'[reconstructor] Frame {i} rgbd2pcd failed: {e}')
                self.fail_list.append({'frame': i, 'reason': str(e)})
                continue

            if source.is_empty():
                print(f'[reconstructor] Frame {i} produced empty cloud, skipping.')
                self.fail_list.append({'frame': i, 'reason': 'empty cloud'})
                continue

            # First frame: set as reference and target
            if i == 0:
                target = source
                self.reference_pcd = copy.deepcopy(source)
                if pose_dir:
                    np.savetxt(os.path.join(pose_dir, f'frame_{i}.txt'), last_transform)
                csv_rows.append([i, '', '', 'initial', '', 'OK', 'initial'])
                self.succeed_list.append({'frame': i, 'fitness': 1.0, 'rmse': 0.0,
                                          'recovered_via': None,
                                          'recovery_attempts': 0})
                self._fire_on_frame(i, total, self.reference_pcd, 1.0, 0.0, 'OK')
                self._save_intermediate()
                continue

            try:
                _, transformation, fitness, rmse = color_icp(
                    source, target,
                    max_iter=self.max_iter,
                    voxel_size=self.voxel_size,
                )
            except Exception as e:
                print(f'[reconstructor] Frame {i} ICP failed: {e}')
                self.fail_list.append({'frame': i, 'reason': f'ICP error: {e}',
                                       'recovery_attempts': 0,
                                       'last_strategy': None})
                self._fire_on_frame(i, total, self.reference_pcd, 0.0, 0.0, 'FAILED')
                csv_rows.append([i, 0.0, 0.0, 0.0, 0.0, 'FAILED', 'icp_exception'])
                continue

            # Strip rotation component when allow_rotation is False.
            # For a linear-translation gantry the camera only translates; ICP
            # spuriously estimates small rotations from depth noise that, when
            # accumulated, progressively smear the plant cloud.
            if not self.allow_rotation:
                transformation = transformation.copy()
                transformation[:3, :3] = np.eye(3, dtype=np.float64)

            # Compute per-step translation and rotation magnitudes for the log
            delta = np.linalg.inv(last_transform) @ transformation
            t_delta = float(np.linalg.norm(delta[:3, 3]))
            cos_t = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
            import math
            r_delta = math.degrees(math.acos(cos_t))

            if fitness > 0.0 or i < 3:
                last_transform = np.dot(last_transform, transformation)
                stable_history.append(last_transform.copy())
                stable_history = stable_history[-5:]
                frame_pcd = copy.deepcopy(source)
                frame_pcd.transform(last_transform)
                self.reference_pcd += frame_pcd
                target = source

                if pose_dir:
                    np.savetxt(os.path.join(pose_dir, f'frame_{i}.txt'), last_transform)

                status = 'OK'
                csv_rows.append([i, f'{fitness:.8g}', f'{rmse:.8g}',
                                  f'{t_delta:.8g}', f'{r_delta:.8g}', status, 'accepted'])
                self.succeed_list.append({
                    'frame': i, 'fitness': fitness, 'rmse': rmse,
                    'recovered_via': None,
                    'recovery_attempts': 0,
                })
                self._fire_on_frame(i, total, self.reference_pcd,
                                    fitness, rmse, status)
                self._save_intermediate()

                print(f'[reconstructor] Frame {i:4d}/{total} | '
                      f'fitness={fitness:.4f} | rmse={rmse:.4f} | '
                      f't={t_delta*1000:.1f}mm | {status}')
            else:
                # ICP rejected ÔÇö try a stable-reference fallback: extrapolate
                # from the last known velocity rather than accumulating bad pose.
                fallback = last_transform.copy()
                if len(stable_history) >= 2:
                    velocity = stable_history[-1][:3, 3] - stable_history[-2][:3, 3]
                    fallback[:3, 3] = stable_history[-1][:3, 3] + velocity
                    if not self.allow_rotation:
                        fallback[:3, :3] = np.eye(3, dtype=np.float64)
                    last_transform = fallback
                    stable_history.append(last_transform.copy())
                    stable_history = stable_history[-5:]
                    if pose_dir:
                        np.savetxt(os.path.join(pose_dir, f'frame_{i}.txt'), last_transform)
                    status = 'INTERPOLATED'
                else:
                    status = 'REJECTED'

                self.fail_list.append({
                    'frame': i, 'reason': 'fitness=0',
                    'fitness': fitness, 'rmse': rmse,
                    'recovery_attempts': 0,
                    'last_strategy': None,
                })
                csv_rows.append([i, f'{fitness:.8g}', f'{rmse:.8g}',
                                  f'{t_delta:.8g}', f'{r_delta:.8g}', status, 'rejected'])
                self._fire_on_frame(i, total, self.reference_pcd,
                                    fitness, rmse, status)
                print(f'[reconstructor] Frame {i:4d}/{total} | '
                      f'{status} (fitness=0)')

        print(f'[reconstructor] ICP complete. '
              f'Success={len(self.succeed_list)} Fail={len(self.fail_list)}')

        # Write per-frame CSV log
        if log_csv_path and csv_rows:
            with open(log_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_idx', 'fitness', 'rmse',
                    'translation_delta_m', 'rotation_delta_deg',
                    'status', 'reason',
                ])
                writer.writerows(csv_rows)
            print(f'[reconstructor] ICP log -> {log_csv_path}')

        if self.on_complete:
            self.on_complete(self.reference_pcd, self.succeed_list, self.fail_list)

        return self.reference_pcd, self.succeed_list, self.fail_list

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire_on_frame(self, frame_idx, total, pcd, fitness, rmse, status):
        if self.on_frame:
            self.on_frame(frame_idx, total, pcd, fitness, rmse, status)

    def _save_intermediate(self):
        if self.save_path and self.reference_pcd and not self.reference_pcd.is_empty():
            out = os.path.join(self.save_path, 'merge_pcd_live.ply')
            o3d.io.write_point_cloud(out, self.reference_pcd)

    def _emergency_save(self):
        if self.reference_pcd and not self.reference_pcd.is_empty():
            out = os.path.join(self.save_path or '.', 'emergency_save.ply')
            o3d.io.write_point_cloud(out, self.reference_pcd)
            print(f'[reconstructor] Emergency save written to {out}')


# --- CLI batch reconstruction (local RGB-D, no PyQt) ---

@dataclass
class ReconstructionConfig:
    camera_id: str = ""
    step_size: int = 8
    max_frames: int | None = None
    start_index: int = 0
    end_index: int | None = None
    depth_scale: float = 1000.0
    depth_trunc: float | None = None
    voxel_size: float = 0.005
    min_fitness: float = 0.01
    output_dir: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    green_only: bool = False


@dataclass
class ReconstructionResult:
    record_path: str
    output_dir: str
    merged_point_cloud_path: str
    pose_dir: str
    summary_path: str
    frames_total: int
    frames_selected: int
    frames_registered: int
    frames_failed: int
    final_point_count: int


def _resolve_record_path(record_path: Path, camera_id: str) -> Path:
    if camera_id:
        candidate = record_path / f"camera_{camera_id}"
        if candidate.exists():
            return candidate
    if any(record_path.glob("rgb_*.png")) and any(record_path.glob("depth_*.png")):
        return record_path
    camera_dirs = sorted(path for path in record_path.glob("camera_*") if path.is_dir())
    if len(camera_dirs) == 1:
        return camera_dirs[0]
    return record_path


def _frame_token(rgb_path: Path) -> str:
    return rgb_path.stem.replace("rgb_", "", 1)


def _load_frame(rgb_path: str, depth_path: str):
    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if color_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB frame: {rgb_path}")
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth frame: {depth_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb, depth


def _load_intrinsics_for_record(record_path: Path, first_rgb_path: str):
    for name in ("kdc_intrinsics.txt", "kd_intrinsics.txt"):
        intrinsics = load_intrinsics(str(record_path / name))
        if intrinsics is not None:
            return intrinsics

    sample = cv2.imread(first_rgb_path, cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"Failed to infer image size from {first_rgb_path}")
    height, width = sample.shape[:2]
    K, dist = get_default_intrinsics(width=width, height=height)
    return K, dist, width, height


def _estimate_depth_trunc(record_path: Path, pairs, depth_scale: float) -> float:
    sample_pairs = pairs[: min(12, len(pairs))]
    valid_depths = []
    for _, depth_path in sample_pairs:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        nz = depth[depth > 0]
        if nz.size:
            valid_depths.append(nz.astype(np.float32))
    if not valid_depths:
        return 3.0
    merged = np.concatenate(valid_depths)
    trunc = float(np.percentile(merged, 99.0) / depth_scale)
    trunc = max(1.5, min(trunc, 8.0))
    print(f"[reconstructor] Auto-selected depth_trunc={trunc:.3f} m")
    return trunc


def reconstruct_sequence(record_path: str | Path, config: ReconstructionConfig | None = None) -> ReconstructionResult:
    cfg = config or ReconstructionConfig()
    root = _resolve_record_path(Path(record_path), cfg.camera_id)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    step_size = max(1, int(cfg.step_size))
    all_pairs = load_image_pairs(str(root), str(root), step=1)
    frames_total = len(all_pairs)
    start_index = max(0, int(cfg.start_index))
    end_index = cfg.end_index if cfg.end_index is None else max(start_index, int(cfg.end_index))
    pairs = all_pairs[start_index:end_index:step_size]
    if cfg.max_frames is not None:
        pairs = pairs[: cfg.max_frames]
    if not pairs:
        raise ValueError(f"No RGB-D frame pairs found under {root}")

    K, dist, _, _ = _load_intrinsics_for_record(root, pairs[0][0])
    depth_trunc = cfg.depth_trunc if cfg.depth_trunc is not None else _estimate_depth_trunc(root, pairs, cfg.depth_scale)

    output_dir = Path(cfg.output_dir) if cfg.output_dir else root / "reconstruction_local"
    pose_dir = output_dir / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    merged = None
    target = None
    global_transform = np.eye(4)
    registered = 0
    failures = []

    for index, (rgb_path, depth_path) in enumerate(pairs):
        color_rgb, depth = _load_frame(rgb_path, depth_path)
        source = rgbd2pcd(
            color_rgb,
            depth,
            K,
            dist=dist,
            bbox=list(cfg.bbox) if cfg.bbox else None,
            depth_scale=cfg.depth_scale,
            depth_trunc=depth_trunc,
        )
        source = clean_pcd(source, voxel_size=cfg.voxel_size)
        frame_token = _frame_token(Path(rgb_path))

        if source.is_empty():
            failures.append({"frame": frame_token, "reason": "empty_point_cloud"})
            continue

        if merged is None:
            merged = copy.deepcopy(source)
            target = copy.deepcopy(source)
            registered = 1
            np.savetxt(pose_dir / f"{frame_token}_{cfg.camera_id}_pose.txt", global_transform)
            continue

        _, transform, fitness, inlier_rmse = color_icp(
            source,
            target,
            voxel_size=cfg.voxel_size,
        )
        if fitness < cfg.min_fitness and index > 2:
            failures.append(
                {
                    "frame": frame_token,
                    "reason": "low_fitness",
                    "fitness": float(fitness),
                    "inlier_rmse": float(inlier_rmse),
                }
            )
            continue

        global_transform = global_transform @ transform
        aligned = copy.deepcopy(source)
        aligned.transform(global_transform)
        merged += aligned
        target = source
        registered += 1
        np.savetxt(pose_dir / f"{frame_token}_{cfg.camera_id}_pose.txt", global_transform)

    if merged is None:
        raise RuntimeError("Reconstruction failed: every sampled frame produced an empty point cloud.")

    if cfg.green_only:
        merged = green_filter_pcd(merged)
        merged = clean_pcd(merged, voxel_size=cfg.voxel_size)

    if cfg.voxel_size > 0:
        merged = merged.voxel_down_sample(cfg.voxel_size)

    merged_path = output_dir / f"merge_pcd_cam{cfg.camera_id}.ply"
    summary_path = output_dir / "reconstruction_summary.json"
    o3d.io.write_point_cloud(str(merged_path), merged)

    summary = {
        "record_path": str(root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "merged_point_cloud_path": str(merged_path.resolve()),
        "pose_dir": str(pose_dir.resolve()),
        "frames_total": frames_total,
        "frames_selected": len(pairs),
        "frames_registered": registered,
        "frames_failed": len(failures),
        "final_point_count": len(merged.points),
        "depth_trunc_used": depth_trunc,
        "config": asdict(cfg),
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ReconstructionResult(
        record_path=str(root.resolve()),
        output_dir=str(output_dir.resolve()),
        merged_point_cloud_path=str(merged_path.resolve()),
        pose_dir=str(pose_dir.resolve()),
        summary_path=str(summary_path.resolve()),
        frames_total=frames_total,
        frames_selected=len(pairs),
        frames_registered=registered,
        frames_failed=len(failures),
        final_point_count=len(merged.points),
    )
