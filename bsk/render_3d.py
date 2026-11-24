"""3D scene renderer: court + players in true 3D space with temporal smoothing.

Uses the calibrated camera H matrix to lift 2D detections to 3D court coordinates,
then renders from arbitrary viewpoints using perspective projection.

Architecture:
1. CourtMesh3D - pre-built 3D court geometry (from bsk.court_3d)
2. PlayerSkeleton3D - lifts 2D pose keypoints to 3D using court plane assumption
3. TemporalSmoother - motion prediction for occlusion handling
4. SceneRenderer - projects everything to camera view with depth sorting

Profile requirement:
- 4GB mode: outputs only 2D coordinates (no 3D rendering)
- 8GB mode: enables full 3D skeleton + streaming to frontend
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from bsk.court_3d import CourtMesh3D, Court3DRenderer
from bsk.court import FIBA, Court
from bsk.schema import Pose2D, Tracked


class TemporalSmoother:
    """Temporal state estimation for skeleton smoothing and occlusion handling.
    
    Uses Kalman filter-like approach with constant bone length constraint.
    Predicts missing keypoints during occlusion using motion model.
    """
    
    def __init__(self, max_age: int = 10, process_noise: float = 0.1, court: Court = FIBA):
        self.max_age = max_age  # Max frames to keep tracking without detection
        self.process_noise = process_noise
        self.court = court
        # Use deque for automatic bounded history (more efficient than list slicing)
        self.history: dict[int, deque] = {}  # track_id -> recent skeletons
    
    def smooth(self, current: list[PlayerSkeleton3D], frame_n: int) -> list[PlayerSkeleton3D]:
        """Apply temporal smoothing and predict missing players.
        
        Args:
            current: Current frame detections
            frame_n: Frame number
        
        Returns:
            Smoothed skeletons with predicted positions for occluded players
        """
        result = []
        
        # Update history for detected players
        for skel in current:
            if skel.track_id not in self.history:
                self.history[skel.track_id] = deque(maxlen=5)  # Auto-trim to last 5
            self.history[skel.track_id].append(skel)
            
            # Apply smoothing if we have history
            if len(self.history[skel.track_id]) >= 2:
                smoothed = self._smooth_skeleton(skel.track_id)
                result.append(smoothed)
            else:
                result.append(skel)
        
        # Predict positions for lost tracks (occlusion)
        for track_id, hist in self.history.items():
            if track_id not in [s.track_id for s in current]:
                # Track is missing - predict position
                if len(hist) >= 2:
                    predicted = self._predict_skeleton(track_id, frame_n)
                    if predicted:
                        result.append(predicted)
        
        return result
    
    def _smooth_skeleton(self, track_id: int) -> PlayerSkeleton3D:
        """Exponential moving average smoothing for skeleton."""
        hist = self.history[track_id]
        if len(hist) < 2:
            return hist[-1]
        
        # Weighted average of recent skeletons
        hist_list = list(hist)  # Convert deque to list for slicing
        weights = np.array([1.0, 0.7, 0.5, 0.3, 0.2][:len(hist_list)])
        weights = weights / weights.sum()
        
        latest = hist_list[-1]
        kpts_smoothed = np.zeros_like(latest.kpts_3d)
        
        for i, skel in enumerate(hist_list[-len(weights):]):
            kpts_smoothed += weights[i] * skel.kpts_3d
        
        # Apply bone length constraints
        kpts_smoothed = self._enforce_bone_constraints(kpts_smoothed)
        
        return PlayerSkeleton3D(
            track_id=track_id,
            bbox_3d=latest.bbox_3d,
            kpts_3d=kpts_smoothed,
            height_m=latest.height_m,
            conf=latest.conf
        )
    
    def _predict_skeleton(self, track_id: int, frame_n: int) -> Optional[PlayerSkeleton3D]:
        """Predict skeleton position using linear motion model."""
        hist = self.history[track_id]
        if len(hist) < 2:
            return None
        
        # Check if prediction is too old
        age = frame_n - len(hist)  # Approximate
        if age > self.max_age:
            del self.history[track_id]
            return None
        
        # Linear extrapolation from last two frames
        prev = hist[-2]
        curr = hist[-1]
        
        velocity = curr.kpts_3d - prev.kpts_3d
        kpts_predicted = curr.kpts_3d + velocity * 0.5  # Half-step prediction
        
        # Clamp to court bounds
        kpts_predicted[:, 0] = np.clip(kpts_predicted[:, 0], -self.court.half_length, self.court.half_length)
        kpts_predicted[:, 1] = np.clip(kpts_predicted[:, 1], -self.court.half_width, self.court.half_width)
        kpts_predicted[:, 2] = np.maximum(kpts_predicted[:, 2], 0.0)  # z >= 0
        
        return PlayerSkeleton3D(
            track_id=track_id,
            bbox_3d=curr.bbox_3d,  # Keep last known bbox
            kpts_3d=kpts_predicted,
            height_m=curr.height_m,
            conf=curr.conf * 0.8  # Reduce confidence for predictions
        )
    
    @staticmethod
    def _enforce_bone_constraints(kpts: np.ndarray) -> np.ndarray:
        """Enforce constant bone length constraints.
        
        Typical human bone lengths (normalized):
        - Upper arm: ~0.3m
        - Forearm: ~0.25m
        - Thigh: ~0.45m
        - Shin: ~0.45m
        """
        # This is a simplified version - full implementation would use
        # inverse kinematics with proper bone length database
        return kpts


@dataclass(slots=True)
class PlayerSkeleton3D:
    """Player skeleton in 3D world coordinates.
    
    Uses the assumption that players stand on the court plane (z=0).
    Height is estimated from the bounding box or standard anthropometry.
    """
    track_id: int
    bbox_3d: tuple[float, float, float, float]  # (x0, y0, x1, y1) in court coords
    kpts_3d: np.ndarray  # (17, 3) x, y, z in meters
    height_m: float = 1.85  # average player height
    conf: float = 0.0
    
    @classmethod
    def from_2d(cls, tracked: Tracked, pose: Pose2D, H_inv: np.ndarray, 
                court: Court = FIBA) -> Optional["PlayerSkeleton3D"]:
        """Lift 2D pixel detections to 3D court coordinates.
        
        Args:
            tracked: 2D bounding box detection
            pose: 2D pose keypoints
            H_inv: Homography matrix (pixel -> court) - ALREADY INVERTED by caller
            court: Court dimensions for validation
        
        Returns:
            PlayerSkeleton3D or None if projection fails
        """
        if H_inv is None or pose.kpts is None:
            return None
        
        # Use H_inv directly (caller already inverted it)
        H_pixel_to_court = H_inv
        
        # Lift bbox corners to court plane
        x0, y0, x1, y1 = tracked.bbox_px
        pts_pixel = np.array([
            [x0, y0, 1],
            [x1, y0, 1],
            [x1, y1, 1],
            [x0, y1, 1],
        ], dtype=np.float64)
        
        # Project to court: p_court = H_pixel_to_court @ p_pixel
        pts_court = (H_pixel_to_court @ pts_pixel.T).T
        pts_court = pts_court[:, :2] / pts_court[:, 2:3]  # Normalize homogeneous coords
        
        bbox_3d = (
            float(pts_court[0, 0]), float(pts_court[0, 1]),
            float(pts_court[2, 0]), float(pts_court[2, 1])
        )
        
        # Lift keypoints to 3D (vectorized for performance)
        kpts_2d = np.array(pose.kpts)  # (17, 3) u, v, conf
        valid_mask = kpts_2d[:, 2] > 0.3  # Filter low-confidence keypoints
        
        if not valid_mask.any():
            return None
        
        # Vectorized projection: batch all keypoints at once
        # Create homogeneous coordinates (17, 3)
        pts_homogeneous = np.column_stack([kpts_2d[:, 0], kpts_2d[:, 1], np.ones(17)])
        # Project: (17, 3) @ (3, 3).T -> (17, 3)
        pts_projected = (H_pixel_to_court @ pts_homogeneous.T).T
        # Normalize homogeneous coordinates
        pts_normalized = pts_projected[:, :2] / pts_projected[:, 2:3]
        
        kpts_3d = np.zeros((17, 3), dtype=np.float32)
        kpts_3d[:, 0] = pts_normalized[:, 0].astype(np.float32)
        kpts_3d[:, 1] = pts_normalized[:, 1].astype(np.float32)
        
        # Vectorized height estimation
        bbox_height_px = tracked.bbox_px[3] - tracked.bbox_px[1]
        height_m = (bbox_height_px / 640) * 2.0  # Rough estimate: 640px ~ 2m
        
        # Estimate z-height for each keypoint based on body part
        heights = np.array([
            cls._estimate_height_vectorized(i, kpts_2d[i, 2], bbox_height_px)
            for i in range(17)
        ], dtype=np.float32)
        
        # Only set heights for valid keypoints
        kpts_3d[:, 2] = np.where(valid_mask, heights, 0.0)
        
        return cls(
            track_id=tracked.track_id,
            bbox_3d=bbox_3d,
            kpts_3d=kpts_3d,
            height_m=max(1.5, min(2.3, height_m)),  # Clamp to reasonable range
            conf=float(tracked.conf)
        )
    
    @staticmethod
    def _estimate_height(kpt_idx: int, conf: float, bbox_px: tuple) -> float:
        """Estimate z-height for a keypoint based on body part (legacy scalar version)."""
        bbox_height = bbox_px[3] - bbox_px[1]
        
        # Feet/ankles on ground
        if kpt_idx in [15, 16]:  # ankles
            return 0.0
        if kpt_idx in [13, 14]:  # knees
            return 0.5 * (bbox_height / 640) * 2.0
        if kpt_idx in [11, 12]:  # hips
            return 0.9 * (bbox_height / 640) * 2.0
        if kpt_idx in [5, 6]:  # shoulders
            return 1.4 * (bbox_height / 640) * 2.0
        if kpt_idx in [7, 8]:  # elbows
            return 1.1 * (bbox_height / 640) * 2.0
        if kpt_idx in [9, 10]:  # wrists
            return 0.8 * (bbox_height / 640) * 2.0
        if kpt_idx == 0:  # nose
            return 1.7 * (bbox_height / 640) * 2.0
        if kpt_idx in [1, 2, 3, 4]:  # head
            return 1.8 * (bbox_height / 640) * 2.0
        
        return 0.0
    
    @staticmethod
    def _estimate_height_vectorized(kpt_idx: int, conf: float, bbox_height_px: float) -> float:
        """Vectorized height estimation for batch processing.
        
        Same logic as _estimate_height but operates on pre-computed bbox height.
        """
        scale_factor = bbox_height_px / 640 * 2.0
        
        # Height multipliers for each COCO keypoint
        height_multipliers = np.array([
            1.7,  # 0: nose
            1.8,  # 1: left_eye
            1.8,  # 2: right_eye
            1.8,  # 3: left_ear
            1.8,  # 4: right_ear
            1.4,  # 5: left_shoulder
            1.4,  # 6: right_shoulder
            1.1,  # 7: left_elbow
            1.1,  # 8: right_elbow
            0.8,  # 9: left_wrist
            0.8,  # 10: right_wrist
            0.9,  # 11: left_hip
            0.9,  # 12: right_hip
            0.5,  # 13: left_knee
            0.5,  # 14: right_knee
            0.0,  # 15: left_ankle
            0.0,  # 16: right_ankle
        ], dtype=np.float32)
        
        return height_multipliers[kpt_idx] * scale_factor
        
        return 0.0  # Default to ground


@dataclass
class SceneObject:
    """Renderable object in 3D scene."""
    vertices: np.ndarray  # (N, 3) x, y, z in world coords
    faces: list[tuple[int, ...]]  # Vertex indices for each face
    color: tuple[int, int, int]  # RGB color
    alpha: float = 1.0  # Transparency


class SceneRenderer:
    """Renders court + players from camera viewpoint.
    
    Uses painter's algorithm for depth sorting (no GPU needed).
    Supports FIBA, NBA, and 3x3 court models.
    """
    
    def __init__(self, court: Court = FIBA, output_size: tuple[int, int] = (1280, 720)):
        self.court = court
        self.output_size = output_size
        self.mesh = CourtMesh3D.build(court)
        self.renderer = Court3DRenderer(self.mesh)
        
        # Camera parameters (will be set from calibration)
        self.camera_pos = np.array([0, -5, 3], dtype=np.float64)  # Broadcast view
        self.camera_target = np.array([0, 0, 0], dtype=np.float64)
        self.up = np.array([0, 0, 1], dtype=np.float64)
        self.fov_deg = 60
    
    def set_court_model(self, court: Court) -> None:
        """Switch court model (FIBA/NBA/3x3)."""
        self.court = court
        self.mesh = CourtMesh3D.build(court)
        self.renderer = Court3DRenderer(self.mesh)
        print(f"[3D] Court model switched to {court.model}")
    
    def set_camera_from_homography(self, H: np.ndarray) -> None:
        """Extract camera position from homography matrix.
        
        For a planar homography H = K[R|t], we can decompose to get camera pose.
        This is approximate but sufficient for visualization.
        """
        # Simple heuristic: use H to find camera center
        # In practice, this should come from proper camera calibration
        if H is not None and H.shape == (3, 3):
            # Camera center in world coords: C = -R^T @ t
            # From H decomposition (simplified)
            try:
                K, R, t = self._decompose_H(H)
                self.camera_pos = (-R.T @ t).flatten()
            except Exception:
                # Fallback to default broadcast view
                pass
    
    def _decompose_H(self, H):
        """Decompose homography into K, R, t (simplified)."""
        # This is a placeholder - proper decomposition needs more work
        K = np.eye(3)
        R = np.eye(3)
        t = np.zeros(3)
        return K, R, t
    
    def render_scene(self, players: list[PlayerSkeleton3D], 
                     show_skeletons: bool = True,
                     show_bboxes: bool = True) -> np.ndarray:
        """Render full 3D scene with court and players.
        
        Args:
            players: List of 3D player skeletons
            show_skeletons: Draw stick figures
            show_bboxes: Draw 3D bounding boxes
        
        Returns:
            RGB image [H, W, 3]
        """
        # Set camera view
        self.renderer.set_view(
            eye=self.camera_pos.tolist(),
            target=self.camera_target.tolist(),
            fov_deg=self.fov_deg
        )
        
        # Render court first
        img = self.renderer.render()
        
        # Overlay players
        if show_skeletons:
            img = self._draw_players(img, players, draw_bboxes=show_bboxes)
        
        return img
    
    def _draw_players(self, img: np.ndarray, players: list[PlayerSkeleton3D],
                      draw_bboxes: bool = True) -> np.ndarray:
        """Draw player skeletons and bboxes on rendered image.
        
        Projects 3D points back to 2D screen space and draws lines/circles.
        """
        from PIL import Image, ImageDraw
        
        h, w = img.shape[:2]
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Skeleton connections (COCO format)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        ]
        
        colors = [
            (255, 100, 100, 200),  # Red
            (100, 255, 100, 200),  # Green
            (100, 100, 255, 200),  # Blue
            (255, 255, 100, 200),  # Yellow
            (255, 100, 255, 200),  # Magenta
            (100, 255, 255, 200),  # Cyan
        ]
        
        for player in players:
            color = colors[player.track_id % len(colors)]
            
            # Project 3D keypoints to 2D screen
            kpts_2d = []
            for kpt in player.kpts_3d:
                screen_pt = self._project_3d_to_2d(kpt, w, h)
                if screen_pt is not None:
                    kpts_2d.append(screen_pt)
                else:
                    kpts_2d.append(None)
            
            # Draw skeleton lines
            if all(pt is not None for pt in kpts_2d):
                for i, j in connections:
                    if kpts_2d[i] and kpts_2d[j]:
                        draw.line(
                            [kpts_2d[i], kpts_2d[j]],
                            fill=color,
                            width=3
                        )
                
                # Draw joints
                for pt in kpts_2d:
                    if pt:
                        x, y = pt
                        r = 4
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            
            # Draw 3D bbox
            if draw_bboxes:
                self._draw_bbox_3d(draw, player, w, h, color)
        
        return np.array(pil_img.convert("RGB"))
    
    def _project_3d_to_2d(self, pt_3d: np.ndarray, 
                          screen_w: int, screen_h: int) -> Optional[tuple[int, int]]:
        """Project 3D world point to 2D screen coordinates.
        
        Uses simple perspective projection.
        """
        # Translate to camera space
        pt_cam = pt_3d - self.camera_pos
        
        # Check if point is behind camera
        if pt_cam[2] <= 0:
            return None
        
        # Perspective projection
        focal = screen_w / (2 * np.tan(np.radians(self.fov_deg / 2)))
        x_screen = (pt_cam[0] * focal / pt_cam[2]) + screen_w / 2
        y_screen = -(pt_cam[1] * focal / pt_cam[2]) + screen_h / 2  # Flip Y
        
        # Check bounds
        if x_screen < 0 or x_screen >= screen_w or y_screen < 0 or y_screen >= screen_h:
            return None
        
        return (int(x_screen), int(y_screen))
    
    def _draw_bbox_3d(self, draw, player: PlayerSkeleton3D, 
                      screen_w: int, screen_h: int, color: tuple):
        """Draw 3D bounding box projected to screen."""
        x0, y0, x1, y1 = player.bbox_3d
        z_bottom = 0.0
        z_top = player.height_m
        
        # 8 corners of bbox
        corners_3d = [
            np.array([x0, y0, z_bottom]),
            np.array([x1, y0, z_bottom]),
            np.array([x1, y1, z_bottom]),
            np.array([x0, y1, z_bottom]),
            np.array([x0, y0, z_top]),
            np.array([x1, y0, z_top]),
            np.array([x1, y1, z_top]),
            np.array([x0, y1, z_top]),
        ]
        
        # Project to 2D
        corners_2d = [self._project_3d_to_2d(c, screen_w, screen_h) for c in corners_3d]
        
        # Draw bottom face
        for i in range(4):
            j = (i + 1) % 4
            if corners_2d[i] and corners_2d[j]:
                draw.line([corners_2d[i], corners_2d[j]], fill=color, width=2)
        
        # Draw top face
        for i in range(4, 8):
            j = 4 + (i + 1 - 4) % 4
            if corners_2d[i] and corners_2d[j]:
                draw.line([corners_2d[i], corners_2d[j]], fill=color, width=2)
        
        # Draw vertical edges
        for i in range(4):
            if corners_2d[i] and corners_2d[i + 4]:
                draw.line([corners_2d[i], corners_2d[i + 4]], fill=color, width=2)


# Convenience function for server integration
def render_3d_scene(players_2d: list[tuple[Tracked, Pose2D]], 
                    H_inv: Optional[np.ndarray],
                    output_size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """High-level API: render 3D scene from 2D detections.
    
    Args:
        players_2d: List of (tracked_bbox, pose) pairs
        H_inv: Inverse homography (pixel -> court)
        output_size: Output image size (width, height)
    
    Returns:
        RGB image [H, W, 3]
    """
    renderer = SceneRenderer(output_size=output_size)
    
    # Lift 2D to 3D
    players_3d = []
    for tracked, pose in players_2d:
        if H_inv is not None:
            skel = PlayerSkeleton3D.from_2d(tracked, pose, H_inv)
            if skel:
                players_3d.append(skel)
    
    # Update camera from calibration
    if H_inv is not None:
        H = np.linalg.inv(H_inv)
        renderer.set_camera_from_homography(H)
    
    # Render
    return renderer.render_scene(players_3d)


__all__ = ["PlayerSkeleton3D", "SceneRenderer", "render_3d_scene"]
