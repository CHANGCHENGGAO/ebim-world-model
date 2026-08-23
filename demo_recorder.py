#!/usr/bin/env python3
"""
EBiM Task3 Demo Recorder — Headless Fixed Dual-Camera System
=============================================================

Standard competition recording setup for dual-arm tabletop manipulation:
  - Camera 1: Global 45° overview (captures both FR3 arms + all table props)
  - Camera 2: Grasp close-up (captures manipulation details)

Both cameras are FIXED — no dynamic pose changes during simulation.
This eliminates camera jitter, orbit drift, and angle misalignment.

Backend options:
  - "replicator"    : omni.replicator.core      [DEFAULT, headless-proven]
  - "isaac_sensor"  : omni.isaac.sensor.Camera  (alternative)

Integration with scene_room.py / task rollout:
    from demo_recorder import demo_recorder

    # 1. After stage is built (after room_scene.build_stage):
    demo_recorder.initialize(mode="isaac_sensor")

    # 2. In the simulation step callback (every world.step()):
    demo_recorder.capture_frame()

    # 3. After all 4 stages complete:
    video_paths = demo_recorder.finalize()

    # 4. ffmpeg commands are printed for manual re-encoding if needed

Camera position tuning:
    Edit CAMERA_CONFIGS below. Each camera has:
      - position : (x, y, z) world coordinates of camera
      - target   : (x, y, z) point the camera looks at
      - fov      : horizontal field of view in degrees
    The recorder computes the correct quaternion automatically.

Scene reference (Task 3 room):
    Kitchen area  : ~(-5.2, -1.4)
    Dining area   : ~(-2.8,  1.7)
    Table height  : ~0.78 m
    Robot base    : floor level (z=0)
    Table X range : -5.5 to -2.0  (3.5 m wide)
    Table Y range : -2.5 to  2.5  (5.0 m long)
"""

import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# ================================================================
# 🔧 CAMERA CONFIGURATION — TUNE THESE PARAMETERS
# ================================================================
# Adjust position/target to fine-tune framing.
# Position = where the camera is located in world coords.
# Target   = the point the camera is looking at.
# Both are (x, y, z) in meters.

CAMERA_CONFIGS: List[dict] = [
    {
        # ============================================================
        # Camera 1: Global 45° Overview (Primary Camera)
        # ============================================================
        # Positioned at the negX / posY corner (kitchen side / dining side),
        # looking diagonally across the table toward the kitchen area.
        #
        # VERIFIED via diagnostic (cam_05 corner_posY_negX):
        #   ✓ Shows BOTH FR3 robot arms clearly
        #   ✓ Shows the full table length (kitchen → dining)
        #   ✓ Shows kitchen counter, tray, bowl, all props
        #   ✓ Shows dining area with the human mannequin
        #   ✓ 45° elevated perspective gives good depth perception
        #
        # Scene layout (top-down):
        #
        #   Y=+3.5                    Dining table + mannequin
        #                              Robot base at Y≈2.7
        #     ● Camera (-6.5, 3.5)    ║
        #        \                    ║  TABLE
        #         \                   ║  X: -5.5 to -2.0
        #          \                  ║
        #           \                 ║
        #   Y=0      ---------------- ║
        #                             ║
        #   Y=-2.5                   ─┘
        #                             Kitchen (X≈-5.2, Y≈-1.4)
        #
        "name": "global_overview",
        # --- TUNABLE PARAMS ---
        # NOTE: These values are verified working via diagnostic test
        "position": (-6.5, 3.5, 3.0),   # (x, y, z) camera location
        "target":   (-4.0, 1.0, 0.78),  # (x, y, z) look-at point
        "fov": 60.0,                       # horizontal FOV (degrees)
        # --- FIXED PARAMS ---
        "resolution": (640, 360),         # 360p (memory-efficient for headless)
    },
    {
        # ============================================================
        # Camera 2: Kitchen / Grasp Close-up
        # ============================================================
        # Positioned at the negX / negY corner (kitchen side / far side),
        # looking at the kitchen manipulation zone.
        #
        # VERIFIED via diagnostic (cam_04 corner_negY_negX):
        #   ✓ Shows kitchen counter with induction stove and mixing bowl
        #   ✓ Shows tray with plate, spoon, food bowl, cup
        #   ✓ Shows robot arm in upper part of frame
        #   ✓ Good for showing grasp and manipulation details
        #
        "name": "kitchen_closeup",
        # --- TUNABLE PARAMS ---
        # NOTE: These values are verified working via diagnostic test
        "position": (-6.5, -3.5, 3.0),  # (x, y, z) camera location
        "target":   (-4.0, -1.0, 0.78), # (x, y, z) look-at point (kitchen)
        "fov": 60.0,                       # FOV for detailed manipulation view
        # --- FIXED PARAMS ---
        "resolution": (640, 360),         # 360p (memory-efficient for headless)
    },
]

# ================================================================
# 🎬 VIDEO OUTPUT SETTINGS
# ================================================================

VIDEO_FPS = 30          # frames per second
VIDEO_CODEC = "libx264"  # H.264
VIDEO_PIX_FMT = "yuv420p"  # compatible with all players
VIDEO_QP = 2             # quality (0=lossless, 51=worst; 2 is nearly lossless)
VIDEO_PRESET = "slow"    # encoding speed/efficiency tradeoff


# ================================================================
# 📹 DemoRecorder Class
# ================================================================

class DemoRecorder:
    """
    Fixed dual-camera demo recorder for Isaac Sim headless.

    Creates two fixed cameras at initialization time. Each simulation
    step, captures one RGB frame from each camera and saves to disk.
    After simulation, encodes frames to MP4 via ffmpeg.

    No dynamic camera pose changes — cameras are fixed at init time.
    No MovieCapture GUI dependency — works in pure headless mode.
    """

    def __init__(
        self,
        output_root: str = "/root/demo_output",
        capture_every_n_steps: int = 1,
        mode: str = "replicator",  # "replicator" (proven headless) or "isaac_sensor"
    ):
        """
        Args:
            output_root: Root directory for frames and videos
            capture_every_n_steps: Capture every N physics steps (1 = every step)
            mode: Camera backend — "isaac_sensor" (default) or "replicator"
        """
        self.output_root = Path(output_root)
        self.capture_every_n_steps = capture_every_n_steps
        self.mode = mode

        # Frame directories
        self.frame_dirs: dict = {}
        for cfg in CAMERA_CONFIGS:
            d = self.output_root / "frames" / cfg["name"]
            d.mkdir(parents=True, exist_ok=True)
            self.frame_dirs[cfg["name"]] = d

        # State
        self._cameras: dict = {}
        self._step_count = 0
        self._frame_count = 0
        self._initialized = False
        self._start_time = 0.0

        # Clear old frames
        self._clear_old_frames()

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self, mode: Optional[str] = None) -> bool:
        """
        Create all cameras on the stage. Call AFTER the stage is fully loaded
        (after room_scene.build_stage() and world.reset()).

        Args:
            mode: Override backend mode ("isaac_sensor" or "replicator")
        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            self._log("Already initialized, skipping")
            return True

        if mode is not None:
            self.mode = mode

        self._log(f"Initializing demo recorder (mode={self.mode})...")
        self._start_time = time.time()

        success = False
        if self.mode == "replicator":
            success = self._init_replicator()
        else:
            success = self._init_isaac_sensor()

        if success:
            self._initialized = True
            self._log(f"✓ Ready: {len(self._cameras)} cameras initialized")
            self._print_camera_summary()
        else:
            self._log("✗ Initialization failed")

        return success

    def _init_isaac_sensor(self) -> bool:
        """Initialize cameras using omni.isaac.sensor.Camera (proven headless).

        Uses USD-level transform setup via Gf.Matrix4d.SetLookAt for
        rock-solid camera orientation. The Camera API's orientation
        parameter can be flaky, so we set the transform directly on
        the USD prim after camera creation.
        """
        try:
            # Camera.initialize() internally uses replicator annotators
            # (e.g. ReferenceTime), so we must enable the extension first.
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("omni.replicator.core")
            enable_extension("omni.replicator.isaac")
            enable_extension("omni.syntheticdata")
            enable_extension("omni.kit.window.viewport")  # may provide ReferenceTime

            from omni.isaac.sensor import Camera
            from pxr import UsdGeom, Gf, Sdf
            from omni.isaac.core.utils.prims import get_prim_at_path
        except ImportError as e:
            self._log(f"ERROR: Required modules not available: {e}")
            return False

        for cfg in CAMERA_CONFIGS:
            name = cfg["name"]
            try:
                cam_path = f"/World/DemoCam_{name}"

                # Create camera first (with identity orientation)
                cam = Camera(
                    prim_path=cam_path,
                    resolution=cfg["resolution"],
                    translation=(0, 0, 0),  # placeholder, set via USD below
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # identity
                )
                cam.initialize()

                # --- Set camera transform via USD (most reliable method) ---
                prim = get_prim_at_path(cam_path)
                if not prim:
                    raise RuntimeError(f"Camera prim not found at {cam_path}")

                xformable = UsdGeom.Xformable(prim)

                eye = Gf.Vec3d(*cfg["position"])
                center = Gf.Vec3d(*cfg["target"])
                up = Gf.Vec3d(0, 0, 1)  # Z-up world

                # Compute view matrix (world -> camera) then invert for camera -> world
                view_matrix = Gf.Matrix4d().SetLookAt(eye, center, up)
                world_matrix = view_matrix.GetInverse()

                # Extract translation and rotation
                translation = world_matrix.ExtractTranslation()
                rotation = world_matrix.ExtractRotation()  # Gf.Rotation
                quat_gf = rotation.GetQuat()  # Gf.Quatd

                # Set the transform by modifying the existing xform ops,
                # or by adding a transform op if needed.
                ops = xformable.GetOrderedXformOps()

                # Try to find translate op
                translate_op = None
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break

                # Try to find rotation op (any type)
                rotate_op = None
                rotate_op_type = None
                for op in ops:
                    t = op.GetOpType()
                    if t in (UsdGeom.XformOp.TypeOrient,
                             UsdGeom.XformOp.TypeRotateXYZ,
                             UsdGeom.XformOp.TypeRotateX,
                             UsdGeom.XformOp.TypeRotateY,
                             UsdGeom.XformOp.TypeRotateZ):
                        rotate_op = op
                        rotate_op_type = t
                        break

                if translate_op is not None and rotate_op is not None:
                    # Modify existing ops (match their precision)
                    # Translate: try double first, fall back to float
                    try:
                        translate_op.Set(Gf.Vec3d(translation))
                    except Exception:
                        translate_op.Set(Gf.Vec3f(float(translation[0]),
                                                  float(translation[1]),
                                                  float(translation[2])))

                    # Orient: try double first, fall back to float
                    if rotate_op_type == UsdGeom.XformOp.TypeOrient:
                        try:
                            rotate_op.Set(quat_gf)  # Gf.Quatd
                        except Exception:
                            rotate_op.Set(Gf.Quatf(
                                float(quat_gf.GetReal()),
                                float(quat_gf.GetImaginary()[0]),
                                float(quat_gf.GetImaginary()[1]),
                                float(quat_gf.GetImaginary()[2]),
                            ))
                    else:
                        # For Euler rotation ops
                        euler = rotation.Decompose(Gf.Vec3d(1, 0, 0),
                                                   Gf.Vec3d(0, 1, 0),
                                                   Gf.Vec3d(0, 0, 1))
                        try:
                            rotate_op.Set(Gf.Vec3d(euler))
                        except Exception:
                            rotate_op.Set(Gf.Vec3f(
                                float(euler[0]), float(euler[1]), float(euler[2])))

                else:
                    # Fallback: add a single transform op (matrix)
                    xformable.ClearXformOpOrder()
                    transform_op = xformable.AddTransformOp()
                    transform_op.Set(world_matrix)

                # --- Set FOV (horizontal) ---
                cam_geom = UsdGeom.Camera(prim)
                # horizontalAperture in mm, default ~20.955mm for 16:9 1080p
                # We set it explicitly and compute focal length from FOV
                horiz_aperture = 36.0  # mm (standard 35mm film width)
                cam_geom.GetHorizontalApertureAttr().Set(horiz_aperture)
                fov_rad = np.radians(cfg["fov"])
                # fov = 2 * atan(aperture/2 / focal_length)
                # focal_length = (aperture/2) / tan(fov/2)
                focal = (horiz_aperture / 2.0) / np.tan(fov_rad / 2.0)
                cam_geom.GetFocalLengthAttr().Set(focal)

                # Store quat in numpy [w, x, y, z] for debugging
                quat = np.array([
                    quat_gf.GetReal(),
                    quat_gf.GetImaginary()[0],
                    quat_gf.GetImaginary()[1],
                    quat_gf.GetImaginary()[2],
                ])

                self._cameras[name] = cam
                self._log(f"  ✓ [{name}] created")
                self._log(f"      pos = ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})")
                self._log(f"      quat = [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")

            except Exception as e:
                self._log(f"  ✗ [{name}] failed: {e}")
                import traceback
                traceback.print_exc()

        return len(self._cameras) > 0

    def _init_replicator(self) -> bool:
        """Initialize cameras using omni.replicator.core."""
        try:
            import omni.replicator.core as rep
            from pxr import UsdGeom, Gf, Sdf
        except ImportError as e:
            self._log(f"ERROR: Required modules not available: {e}")
            return False

        for cfg in CAMERA_CONFIGS:
            name = cfg["name"]
            try:
                cam_path = f"/World/DemoCam_{name}"

                # Step 1: Create camera prim FIRST (render_product needs it)
                import omni.usd
                stage = omni.usd.get_context().get_stage()

                prim = stage.DefinePrim(Sdf.Path(cam_path), "Camera")
                xformable = UsdGeom.Xformable(prim)

                eye = Gf.Vec3d(*cfg["position"])
                center = Gf.Vec3d(*cfg["target"])
                up = Gf.Vec3d(0, 0, 1)

                view_matrix = Gf.Matrix4d().SetLookAt(eye, center, up)
                world_matrix = view_matrix.GetInverse()
                translation = world_matrix.ExtractTranslation()
                rotation = world_matrix.ExtractRotation()
                quat_gf = rotation.GetQuat()

                # Set transform ops
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()
                try:
                    translate_op.Set(Gf.Vec3d(translation))
                except Exception:
                    translate_op.Set(Gf.Vec3f(float(translation[0]),
                                              float(translation[1]),
                                              float(translation[2])))
                orient_op = xformable.AddOrientOp()
                try:
                    orient_op.Set(quat_gf)
                except Exception:
                    orient_op.Set(Gf.Quatf(
                        float(quat_gf.GetReal()),
                        float(quat_gf.GetImaginary()[0]),
                        float(quat_gf.GetImaginary()[1]),
                        float(quat_gf.GetImaginary()[2]),
                    ))

                # Set FOV
                cam_geom = UsdGeom.Camera(prim)
                horiz_ap = 36.0
                cam_geom.GetHorizontalApertureAttr().Set(horiz_ap)
                fov_rad = np.radians(cfg["fov"])
                focal = (horiz_ap / 2.0) / np.tan(fov_rad / 2.0)
                cam_geom.GetFocalLengthAttr().Set(focal)

                # Step 2: Create render product from the camera prim
                rp = rep.create.render_product(
                    cam_path,
                    resolution=cfg["resolution"],
                )

                # Step 3: Create and attach RGB annotator
                rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annotator.attach([rp])

                self._cameras[name] = {
                    "render_product": rp,
                    "rgb_annotator": rgb_annotator,
                    "path": cam_path,
                }
                self._log(f"  ✓ [{name}] replicator camera created")

            except Exception as e:
                self._log(f"  ✗ [{name}] failed: {e}")
                import traceback
                traceback.print_exc()

        return len(self._cameras) > 0

    # ----------------------------------------------------------------
    # Per-step frame capture
    # ----------------------------------------------------------------

    def capture_frame(self) -> None:
        """
        Capture one frame from each camera. Call every simulation step
        (after world.step() or in the physics step callback).

        Frames are saved as PNG files with zero-padded sequential names.
        """
        if not self._initialized or not self._cameras:
            return

        self._step_count += 1
        if self._step_count % self.capture_every_n_steps != 0:
            return

        self._frame_count += 1
        frame_idx = self._frame_count

        if self.mode == "replicator":
            self._capture_replicator(frame_idx)
        else:
            self._capture_isaac_sensor(frame_idx)

        # Progress logging
        if self._frame_count % 300 == 0:
            elapsed = time.time() - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            self._log(f"  Captured {self._frame_count} frames ({fps:.1f} fps)")

    def _capture_isaac_sensor(self, frame_idx: int) -> None:
        """Capture frames using omni.isaac.sensor.Camera."""
        for name, cam in self._cameras.items():
            try:
                rgba = cam.get_rgba()
                if rgba is None:
                    continue

                img = np.array(rgba, dtype=np.uint8)
                # RGBA -> RGB
                if img.ndim == 3 and img.shape[2] >= 3:
                    img = img[:, :, :3]

                # Save as PNG
                filepath = self.frame_dirs[name] / f"frame_{frame_idx:06d}.png"
                self._save_png(img, str(filepath))

            except Exception as e:
                if frame_idx <= 5:
                    self._log(f"  Capture error [{name}]: {e}")

    def _capture_replicator(self, frame_idx: int) -> None:
        """Capture frames using omni.replicator.core annotators."""
        for name, cam_info in self._cameras.items():
            try:
                annotator = cam_info["rgb_annotator"]
                data = annotator.get_data()
                if data is None:
                    continue

                # data is typically a dict with "data" key containing the image
                if isinstance(data, dict) and "data" in data:
                    img = np.array(data["data"], dtype=np.uint8)
                else:
                    img = np.array(data, dtype=np.uint8)

                # RGBA -> RGB if needed
                if img.ndim == 3 and img.shape[2] >= 3:
                    img = img[:, :, :3]

                filepath = self.frame_dirs[name] / f"frame_{frame_idx:06d}.png"
                self._save_png(img, str(filepath))

            except Exception as e:
                if frame_idx <= 5:
                    self._log(f"  Capture error [{name}]: {e}")

    def _save_png(self, img_rgb: np.ndarray, filepath: str) -> None:
        """Save RGB image as PNG. Tries cv2 first, then PIL."""
        # Try cv2 (fast)
        try:
            import cv2
            img_bgr = img_rgb[:, :, ::-1].copy()  # RGB -> BGR
            cv2.imwrite(filepath, img_bgr)
            return
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: PIL
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            pil_img.save(filepath)
        except Exception as e:
            if self._frame_count <= 3:
                self._log(f"  PNG save error: {e}")

    # ----------------------------------------------------------------
    # Finalize — encode videos
    # ----------------------------------------------------------------

    def finalize(self) -> dict:
        """
        Stop recording and encode MP4 videos using ffmpeg.

        Returns:
            dict mapping camera name -> video file path (or None if failed)
        """
        elapsed = time.time() - self._start_time
        self._log(f"Finalizing — {self._frame_count} frames in {elapsed:.1f}s")

        results = {}

        for cfg in CAMERA_CONFIGS:
            name = cfg["name"]
            frame_dir = self.frame_dirs[name]

            # Count actual frames
            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ])

            if not frame_files:
                self._log(f"  [{name}]: no frames captured, skipped")
                results[name] = None
                continue

            self._log(f"  [{name}]: {len(frame_files)} frames → encoding MP4...")

            video_path = self.output_root / f"demo_{name}.mp4"

            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(VIDEO_FPS),
                "-pattern_type", "glob",
                "-i", f"{frame_dir}/frame_*.png",
                "-c:v", VIDEO_CODEC,
                "-preset", VIDEO_PRESET,
                "-pix_fmt", VIDEO_PIX_FMT,
                "-qp", str(VIDEO_QP),
                "-movflags", "+faststart",
                str(video_path),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0 and video_path.exists():
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    self._log(f"  ✓ [{name}]: {video_path} ({size_mb:.1f} MB)")
                    results[name] = str(video_path)
                else:
                    self._log(f"  ✗ [{name}]: ffmpeg failed (code {result.returncode})")
                    if result.stderr:
                        # Print last few lines of error
                        stderr_lines = result.stderr.strip().split("\n")
                        for line in stderr_lines[-5:]:
                            self._log(f"    ffmpeg: {line}")
                    results[name] = None

            except subprocess.TimeoutExpired:
                self._log(f"  ✗ [{name}]: ffmpeg timed out")
                results[name] = None
            except Exception as e:
                self._log(f"  ✗ [{name}]: encoding error: {e}")
                results[name] = None

        # Print summary commands for manual use
        self._print_ffmpeg_commands()

        total_videos = sum(1 for v in results.values() if v is not None)
        self._log(f"Done: {total_videos}/{len(CAMERA_CONFIGS)} videos created")

        return results

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def _clear_old_frames(self) -> None:
        """Delete old frame PNG files from previous runs."""
        for name, frame_dir in self.frame_dirs.items():
            count = 0
            for f in os.listdir(frame_dir):
                if f.startswith("frame_") and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(frame_dir, f))
                        count += 1
                    except OSError:
                        pass
            if count > 0:
                self._log(f"  Cleared {count} old frames from {name}")

    def _print_camera_summary(self) -> None:
        """Print a summary of all camera configurations."""
        self._log("--- Camera Configuration ---")
        for cfg in CAMERA_CONFIGS:
            pos = np.array(cfg["position"])
            tgt = np.array(cfg["target"])
            dist = np.linalg.norm(tgt - pos)
            dx, dy, dz = tgt - pos

            # Compute angles for debugging
            pitch_deg = np.degrees(np.arcsin(-dz / dist)) if dist > 0 else 0
            yaw_deg = np.degrees(np.arctan2(dy, dx))

            self._log(f"  {cfg['name']}:")
            self._log(f"    position : ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
            self._log(f"    target   : ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}) m")
            self._log(f"    distance : {dist:.2f} m")
            self._log(f"    pitch    : {pitch_deg:.1f}° (downward angle)")
            self._log(f"    yaw      : {yaw_deg:.1f}°")
            self._log(f"    FOV      : {cfg['fov']:.0f}°")
            self._log(f"    resolution: {cfg['resolution'][0]}x{cfg['resolution'][1]}")

    def _print_ffmpeg_commands(self) -> None:
        """Print ffmpeg commands for manual re-encoding."""
        self._log("--- Manual ffmpeg commands ---")
        for cfg in CAMERA_CONFIGS:
            name = cfg["name"]
            frame_dir = self.frame_dirs[name]
            video_path = self.output_root / f"demo_{name}.mp4"
            cmd = (
                f"ffmpeg -y -framerate {VIDEO_FPS} "
                f"-pattern_type glob -i '{frame_dir}/frame_*.png' "
                f"-c:v {VIDEO_CODEC} -preset {VIDEO_PRESET} "
                f"-pix_fmt {VIDEO_PIX_FMT} -qp {VIDEO_QP} -movflags +faststart "
                f"{video_path}"
            )
            self._log(f"  {name}:")
            self._log(f"    {cmd}")

    def _log(self, msg: str) -> None:
        """Log message with flush (important for headless)."""
        print(f"[DemoRecorder] {msg}", flush=True)

    @staticmethod
    def _look_at_quat(cam_pos: Tuple[float, float, float],
                      target_pos: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute [w, x, y, z] quaternion for a camera at cam_pos looking at target_pos.

        Isaac Sim camera convention:
          - Default forward direction: +X axis
          - Default up direction: +Z axis
          - Rotation order: pitch (Y-axis) then yaw (Z-axis)

        Args:
            cam_pos: (x, y, z) camera position
            target_pos: (x, y, z) target point to look at

        Returns:
            numpy array [w, x, y, z] quaternion
        """
        dx = target_pos[0] - cam_pos[0]
        dy = target_pos[1] - cam_pos[1]
        dz = target_pos[2] - cam_pos[2]

        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        # Normalized forward direction
        fx, fy, fz = dx / dist, dy / dist, dz / dist

        # Pitch: rotation around Y axis
        # sin(pitch) = -fz (pitch down when looking down = negative fz)
        pitch = np.arcsin(np.clip(-fz, -1.0, 1.0))
        cos_pitch = np.cos(pitch)

        # Yaw: rotation around Z axis
        # After pitch rotation, forward in XY plane is (cos(pitch), 0)
        # After yaw, it becomes (cos(pitch)*cos(yaw), cos(pitch)*sin(yaw)) = (fx, fy)
        if abs(cos_pitch) < 1e-6:
            yaw = 0.0
        else:
            yaw = np.arctan2(fy, fx)

        # Quaternion for pitch (around Y): qy = [cos(hp), 0, sin(hp), 0]
        # Quaternion for yaw (around Z): qz = [cos(hz), 0, 0, sin(hz)]
        # Combined: q = qy * qz (pitch first, then yaw)
        hp, hz = pitch / 2.0, yaw / 2.0
        cp, sp = np.cos(hp), np.sin(hp)
        cz, sz = np.cos(hz), np.sin(hz)

        w = cp * cz
        x = sp * sz
        y = sp * cz
        z = cp * sz

        return np.array([w, x, y, z])


# ================================================================
# 🎯 Singleton instance for easy import
# ================================================================

demo_recorder = DemoRecorder()


# ================================================================
# 🧪 Quick test — print camera configs and validate quaternions
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EBiM Task3 Demo Recorder — Configuration Preview")
    print("=" * 60)
    print(f"Mode: isaac_sensor (default)")
    print(f"Output: /root/demo_output/")
    print(f"Video: {VIDEO_FPS}fps, {VIDEO_CODEC}, QP={VIDEO_QP}")
    print()

    for i, cfg in enumerate(CAMERA_CONFIGS):
        pos = np.array(cfg["position"])
        tgt = np.array(cfg["target"])
        dist = np.linalg.norm(tgt - pos)
        quat = DemoRecorder._look_at_quat(pos, tgt)

        dx, dy, dz = tgt - pos
        pitch_deg = np.degrees(np.arcsin(-dz / dist))
        yaw_deg = np.degrees(np.arctan2(dy, dx))

        print(f"[{i}] {cfg['name']}")
        print(f"    Position:  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
        print(f"    Target:    ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}) m")
        print(f"    Distance:  {dist:.2f} m")
        print(f"    Pitch:     {pitch_deg:.1f}° (downward angle)")
        print(f"    Yaw:       {yaw_deg:.1f}°")
        print(f"    FOV:       {cfg['fov']}°")
        print(f"    Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print()

    print("=" * 60)
    print("To use in your task rollout:")
    print("  from demo_recorder import demo_recorder")
    print("  demo_recorder.initialize()           # after stage loaded")
    print("  demo_recorder.capture_frame()        # every sim step")
    print("  demo_recorder.finalize()             # after all stages")
    print("=" * 60)
