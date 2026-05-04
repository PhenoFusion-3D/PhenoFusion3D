"""Registration quality gate and recovery strategies for RGB-D alignment.

Standalone version for use with reconstruct_icp_sequence.py.
Imports from utils_rgbd (root-level helper) rather than from the
PhenoFusion3D processing package so it can run independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d

from utils_rgbd import colored_icp, fpfh_ransac_initial_transform, point_to_plane_icp


@dataclass(frozen=True)
class RegistrationMetrics:
    """Metrics used to decide whether an alignment is trustworthy."""

    fitness: float
    inlier_rmse: float
    translation_delta: float
    rotation_delta: float
    frame_idx: int


@dataclass(frozen=True)
class RegistrationResult:
    """A candidate registration result."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    metrics: RegistrationMetrics
    method: str


@dataclass(frozen=True)
class AgentDecision:
    """Final accept/reject decision for a frame."""

    accept: bool
    strategy_applied: str | None
    retry_result: RegistrationResult | None
    reason: str


class RegistrationAgent:
    """Evaluate registration quality and try cheap recovery strategies."""

    FITNESS_MIN = 0.30
    RMSE_MAX = 0.025
    TRANSLATION_MAX = 0.06
    ROTATION_MAX = 5.0

    def __init__(
        self,
        *,
        fitness_min: float = FITNESS_MIN,
        rmse_max: float = RMSE_MAX,
        translation_max: float = TRANSLATION_MAX,
        rotation_max: float = ROTATION_MAX,
    ) -> None:
        self.fitness_min = fitness_min
        self.rmse_max = rmse_max
        self.translation_max = translation_max
        self.rotation_max = rotation_max

    def evaluate(self, metrics: RegistrationMetrics) -> bool:
        return (
            metrics.fitness >= self.fitness_min
            and metrics.inlier_rmse <= self.rmse_max
            and metrics.translation_delta <= self.translation_max
            and metrics.rotation_delta <= self.rotation_max
        )

    def metrics_from_transform(
        self,
        *,
        frame_idx: int,
        transformation: np.ndarray,
        last_stable_transform: np.ndarray,
        fitness: float,
        inlier_rmse: float,
    ) -> RegistrationMetrics:
        delta = np.linalg.inv(last_stable_transform) @ transformation
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        rot = delta[:3, :3]
        trace = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
        rotation_delta = float(np.degrees(np.arccos(trace)))
        return RegistrationMetrics(
            fitness=float(fitness),
            inlier_rmse=float(inlier_rmse),
            translation_delta=translation_delta,
            rotation_delta=rotation_delta,
            frame_idx=frame_idx,
        )

    def make_result(
        self,
        *,
        frame_idx: int,
        transformation: np.ndarray,
        last_stable_transform: np.ndarray,
        fitness: float,
        inlier_rmse: float,
        method: str,
    ) -> RegistrationResult:
        metrics = self.metrics_from_transform(
            frame_idx=frame_idx,
            transformation=transformation,
            last_stable_transform=last_stable_transform,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
        )
        return RegistrationResult(
            transformation=transformation,
            fitness=float(fitness),
            inlier_rmse=float(inlier_rmse),
            metrics=metrics,
            method=method,
        )

    def apply_strategy(
        self,
        strategy: int,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        *,
        frame_idx: int,
        voxel_size: float,
        init: np.ndarray,
        last_stable_transform: np.ndarray,
    ) -> RegistrationResult:
        strategies: dict[int, Callable[[], tuple[np.ndarray, float, float, str]]] = {
            1: lambda: self._stronger_downsample(source, target, voxel_size, init),
            2: lambda: self._tight_crop(source, target, voxel_size, init),
            3: lambda: self._more_iterations(source, target, voxel_size, init),
            4: lambda: self._point_to_plane(source, target, voxel_size, init),
            5: lambda: self._looser_correspondence(source, target, voxel_size, init),
            6: lambda: self._fpfh_ransac_then_colored(source, target, voxel_size),
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown registration recovery strategy {strategy}")
        transformation, fitness, rmse, method = strategies[strategy]()
        return self.make_result(
            frame_idx=frame_idx,
            transformation=transformation,
            last_stable_transform=last_stable_transform,
            fitness=fitness,
            inlier_rmse=rmse,
            method=method,
        )

    def recover(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        *,
        frame_idx: int,
        voxel_size: float,
        init: np.ndarray,
        last_stable_transform: np.ndarray,
        strategies: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    ) -> AgentDecision:
        best: RegistrationResult | None = None
        for strategy in strategies:
            try:
                result = self.apply_strategy(
                    strategy,
                    source,
                    target,
                    frame_idx=frame_idx,
                    voxel_size=voxel_size,
                    init=init,
                    last_stable_transform=last_stable_transform,
                )
            except (RuntimeError, ValueError):
                continue
            if best is None or (result.fitness, -result.inlier_rmse) > (best.fitness, -best.inlier_rmse):
                best = result
            if self.evaluate(result.metrics):
                return AgentDecision(
                    accept=True,
                    strategy_applied=f"strategy_{strategy}:{result.method}",
                    retry_result=result,
                    reason="recovered",
                )
        return AgentDecision(
            accept=False,
            strategy_applied=None,
            retry_result=best,
            reason="all_strategies_failed",
        )

    def _stronger_downsample(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, str]:
        _res, transform, fitness, rmse = colored_icp(
            source,
            target,
            voxel_size=voxel_size * 1.5,
            max_iteration=150,
            init=init,
        )
        return transform, fitness, rmse, "stronger_downsample_colored_icp"

    def _tight_crop(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, str]:
        src = _crop_percentile(source)
        tgt = _crop_percentile(target)
        _res, transform, fitness, rmse = colored_icp(
            src,
            tgt,
            voxel_size=voxel_size,
            max_iteration=150,
            init=init,
        )
        return transform, fitness, rmse, "tight_crop_colored_icp"

    def _more_iterations(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, str]:
        _res, transform, fitness, rmse = colored_icp(
            source,
            target,
            voxel_size=voxel_size,
            max_iteration=200,
            init=init,
        )
        return transform, fitness, rmse, "more_iterations_colored_icp"

    def _point_to_plane(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, str]:
        _res, transform, fitness, rmse = point_to_plane_icp(
            source,
            target,
            voxel_size=voxel_size,
            max_iteration=150,
            init=init,
        )
        return transform, fitness, rmse, "point_to_plane_icp"

    def _looser_correspondence(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
        init: np.ndarray,
    ) -> tuple[np.ndarray, float, float, str]:
        _res, transform, fitness, rmse = colored_icp(
            source,
            target,
            voxel_size=voxel_size,
            max_correspondence_distance=voxel_size * 8.0,
            max_iteration=150,
            init=init,
        )
        return transform, fitness, rmse, "looser_correspondence_colored_icp"

    def _fpfh_ransac_then_colored(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> tuple[np.ndarray, float, float, str]:
        _ransac, init, _fitness, _rmse = fpfh_ransac_initial_transform(
            source,
            target,
            voxel_size=voxel_size * 2.0,
        )
        _res, transform, fitness, rmse = colored_icp(
            source,
            target,
            voxel_size=voxel_size,
            max_iteration=200,
            init=init,
        )
        return transform, fitness, rmse, "fpfh_ransac_colored_icp"


def _crop_percentile(
    pcd: o3d.geometry.PointCloud,
    *,
    lower: float = 5.0,
    upper: float = 95.0,
) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    if len(points) < 100:
        return pcd
    min_bound = np.percentile(points, lower, axis=0)
    max_bound = np.percentile(points, upper, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)
