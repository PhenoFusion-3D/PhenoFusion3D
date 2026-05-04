"""Robust RGB-D sequence reconstruction with registration recovery.

Standalone version for use with reconstruct_icp_sequence.py.
Imports from utils_rgbd (root-level helper) rather than from the
PhenoFusion3D processing package so it can run independently.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pipeline_registration_agent import AgentDecision, RegistrationAgent, RegistrationResult
from utils_rgbd import clean_pcd, colored_icp, load_intrinsics, pair_frames, plant_mask_bgr, rgbd_to_pcd


@dataclass(frozen=True)
class ReconstructorConfig:
    data: Path
    step: int = 1
    max_frames: int | None = None
    undistort: bool = False
    erode_depth_edges: bool = True
    allow_rotation: bool = False
    icp_voxel: float = 0.01
    depth_trunc: float = 2.5
    plant_icp: bool = False


class Reconstructor:
    """Sequential reconstructor that gates and recovers registration failures."""

    def __init__(self, config: ReconstructorConfig, agent: RegistrationAgent | None = None) -> None:
        self.config = config
        self.agent = agent or RegistrationAgent()

    def run(self) -> None:
        data_dir = self.config.data.expanduser().resolve()
        intrinsic_path = data_dir / "kdc_intrinsics.txt"
        output_dir = data_dir / "output"
        pose_dir = output_dir / "poses"
        output_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        pairs_all = pair_frames(data_dir)
        ds_imgs = len(pairs_all) // self.config.step
        if ds_imgs < 1:
            raise SystemExit(f"Need at least {self.config.step} pairs; have {len(pairs_all)}.")
        if self.config.max_frames is not None:
            ds_imgs = min(ds_imgs, max(1, self.config.max_frames))

        intr = load_intrinsics(intrinsic_path)
        k_orig = intr["K"]
        dist = intr["dist"]
        intrinsic_wh = (intr["width"], intr["height"])

        reference_pcd: o3d.geometry.PointCloud | None = None
        target_model = o3d.geometry.PointCloud()
        last_stable_transform = np.eye(4, dtype=np.float64)
        stable_history: list[np.ndarray] = [last_stable_transform.copy()]
        csv_rows: list[list[object]] = []

        merge_path = output_dir / "merge_pcd.ply"
        log_csv = output_dir / "icp_log.csv"

        print(f"[icp] data_dir={data_dir}")
        print(
            f"[icp] pairs total={len(pairs_all)}, step={self.config.step}, "
            f"sampled iterations={ds_imgs}"
        )
        if self.config.plant_icp:
            print("[icp] plant_icp=True: masking non-plant depth before ICP")

        for i in tqdm(range(ds_imgs), desc="ICP"):
            idx = i * self.config.step
            rgb_p, depth_p = pairs_all[idx]
            color_bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
            depth_u16 = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
            if color_bgr is None or depth_u16 is None:
                raise RuntimeError(f"Cannot read idx {idx}: {rgb_p} / {depth_p}")

            k_use = np.array(k_orig, copy=True)
            if self.config.undistort:
                color_bgr = cv2.undistort(color_bgr, k_use, dist, None, k_use)
                depth_f32 = depth_u16.astype(np.float32)
                depth_remapped = cv2.undistort(depth_f32, k_use, dist, None, k_use)
                depth_u16 = np.round(depth_remapped).astype(depth_u16.dtype)

            depth_for_source = depth_u16
            if self.config.plant_icp:
                mask = plant_mask_bgr(color_bgr)
                depth_for_source = depth_u16.copy()
                depth_for_source[mask == 0] = 0

            source = rgbd_to_pcd(
                color_bgr,
                depth_for_source,
                k_use,
                depth_scale=1000.0,
                depth_trunc=self.config.depth_trunc,
                erode_depth_edges=self.config.erode_depth_edges,
                intrinsic_width_height=intrinsic_wh,
            )
            source_for_icp = clean_pcd(source)

            if i == 0:
                reference_pcd = copy.deepcopy(source)
                target_model = self._build_model_target(reference_pcd)
                np.savetxt(str(pose_dir / f"frame_{idx}.txt"), last_stable_transform)
                csv_rows.append([idx, "", "", "", "", "accepted", "initial", "initial"])
                o3d.io.write_point_cloud(str(merge_path), reference_pcd)
                continue

            result, decision = self._run_icp(
                source_for_icp,
                target_model,
                frame_idx=idx,
                init=last_stable_transform,
                last_stable_transform=last_stable_transform,
                stable_history=stable_history,
            )
            transform = result.transformation
            status = "accepted" if decision.accept else "rejected"
            if decision.reason == "stable_reference_fallback":
                status = "interpolated"
            csv_rows.append(
                [
                    idx,
                    f"{result.fitness:.8g}",
                    f"{result.inlier_rmse:.8g}",
                    f"{result.metrics.translation_delta:.8g}",
                    f"{result.metrics.rotation_delta:.8g}",
                    status,
                    result.method,
                    decision.reason,
                ]
            )

            if decision.accept:
                last_stable_transform = transform
                stable_history.append(last_stable_transform.copy())
                stable_history = stable_history[-5:]
                np.savetxt(str(pose_dir / f"frame_{idx}.txt"), last_stable_transform)
                assert reference_pcd is not None
                reference_pcd += copy.deepcopy(source).transform(last_stable_transform)
                target_model = self._build_model_target(reference_pcd)
                o3d.io.write_point_cloud(str(merge_path), reference_pcd)
                tqdm.write(
                    "[icp] frame_idx="
                    f"{idx} fitness={result.fitness:.6g} rmse={result.inlier_rmse:.6g} "
                    f"jump={result.metrics.translation_delta:.4g}m status={status} "
                    f"method={result.method}"
                )
            else:
                tqdm.write(
                    "[icp] frame_idx="
                    f"{idx} fitness={result.fitness:.6g} rmse={result.inlier_rmse:.6g} "
                    f"rejected method={result.method}"
                )

        if reference_pcd is not None:
            o3d.io.write_point_cloud(str(merge_path), reference_pcd)

        with log_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_idx",
                    "fitness",
                    "rmse",
                    "translation_delta_m",
                    "rotation_delta_deg",
                    "accepted",
                    "method",
                    "reason",
                ]
            )
            writer.writerows(csv_rows)

        print(f"[icp] Wrote merged cloud -> {merge_path}")
        if reference_pcd is not None and not reference_pcd.is_empty():
            clean_path = output_dir / "merge_pcd_clean.ply"
            final = reference_pcd.voxel_down_sample(voxel_size=self.config.icp_voxel)
            final, _ = final.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
            o3d.io.write_point_cloud(str(clean_path), final)
            print(f"[icp] Wrote cleaned cloud ({len(final.points)} pts) -> {clean_path}")

        print(f"[icp] CSV log -> {log_csv}")
        print(f"[icp] Poses saved under -> {pose_dir}")

    def _run_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        *,
        frame_idx: int,
        init: np.ndarray,
        last_stable_transform: np.ndarray,
        stable_history: list[np.ndarray],
    ) -> tuple[RegistrationResult, AgentDecision]:
        try:
            _res, transform, fitness, rmse = colored_icp(
                source,
                target,
                voxel_size=self.config.icp_voxel,
                init=init,
            )
            transform = self._constrain_rotation(transform)
            result = self.agent.make_result(
                frame_idx=frame_idx,
                transformation=transform,
                last_stable_transform=last_stable_transform,
                fitness=fitness,
                inlier_rmse=rmse,
                method="colored_icp",
            )
            if self.agent.evaluate(result.metrics):
                return result, AgentDecision(True, None, result, "accepted")
        except (RuntimeError, ValueError):
            result = self.agent.make_result(
                frame_idx=frame_idx,
                transformation=last_stable_transform.copy(),
                last_stable_transform=last_stable_transform,
                fitness=0.0,
                inlier_rmse=float("inf"),
                method="colored_icp_failed",
            )

        decision = self.agent.recover(
            source,
            target,
            frame_idx=frame_idx,
            voxel_size=self.config.icp_voxel,
            init=init,
            last_stable_transform=last_stable_transform,
        )
        if decision.accept and decision.retry_result is not None:
            recovered = self._with_rotation_constraint(
                decision.retry_result,
                frame_idx=frame_idx,
                last_stable_transform=last_stable_transform,
            )
            return recovered, AgentDecision(True, decision.strategy_applied, recovered, decision.reason)

        fallback = self._stable_reference_fallback(
            frame_idx=frame_idx,
            stable_history=stable_history,
            last_stable_transform=last_stable_transform,
            base_result=decision.retry_result or result,
        )
        return fallback, AgentDecision(True, "stable_reference_fallback", fallback, "stable_reference_fallback")

    def _stable_reference_fallback(
        self,
        *,
        frame_idx: int,
        stable_history: list[np.ndarray],
        last_stable_transform: np.ndarray,
        base_result: RegistrationResult,
    ) -> RegistrationResult:
        fallback = last_stable_transform.copy()
        if len(stable_history) >= 2:
            velocity = stable_history[-1][:3, 3] - stable_history[-2][:3, 3]
            fallback[:3, 3] = stable_history[-1][:3, 3] + velocity
        fallback = self._constrain_rotation(fallback)
        return self.agent.make_result(
            frame_idx=frame_idx,
            transformation=fallback,
            last_stable_transform=last_stable_transform,
            fitness=base_result.fitness,
            inlier_rmse=base_result.inlier_rmse,
            method="stable_reference_fallback",
        )

    def _with_rotation_constraint(
        self,
        result: RegistrationResult,
        *,
        frame_idx: int,
        last_stable_transform: np.ndarray,
    ) -> RegistrationResult:
        transform = self._constrain_rotation(result.transformation)
        return self.agent.make_result(
            frame_idx=frame_idx,
            transformation=transform,
            last_stable_transform=last_stable_transform,
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            method=result.method,
        )

    def _constrain_rotation(self, transform: np.ndarray) -> np.ndarray:
        out = np.array(transform, copy=True, dtype=np.float64)
        if not self.config.allow_rotation:
            out[:3, :3] = np.eye(3, dtype=np.float64)
        return out

    def _build_model_target(self, reference_pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        target = reference_pcd.voxel_down_sample(voxel_size=self.config.icp_voxel)
        return clean_pcd(target)
