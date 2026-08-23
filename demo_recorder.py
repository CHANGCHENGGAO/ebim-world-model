#!/usr/bin/env python3
"""
Demo Recorder — captures simulation camera frames for video generation.

Runs as a tick callback inside Isaac Sim, captures camera images
periodically, and saves them as PNG files. After the run, ffmpeg
combines them into an MP4 video.
"""

import os
import time
from typing import Optional


class DemoRecorder:
    """Captures camera frames during simulation for demo video."""

    def __init__(self, output_dir: str = "/root/demo_frames", fps: int = 10):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_capture = 0.0
        self.frame_count = 0
        self.camera = None
        self._initialized = False

        os.makedirs(output_dir, exist_ok=True)
        # Clear previous frames
        for f in os.listdir(output_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(output_dir, f))

        print(f"[DemoRecorder] Output dir: {output_dir}, target FPS: {fps}")

    def _init_camera(self):
        """Create a camera sensor at a vantage point overlooking the scene."""
        if self._initialized:
            return
        try:
            from isaacsim.sensors.camera import Camera
            import omni.usd
            from pxr import Gf

            # Create camera at a vantage point
            cam_path = "/World/DemoCamera"
            self.camera = Camera(
                prim_path=cam_path,
                position=(-3.5, 0.5, 3.0),
                orientation=Gf.Quatf(0.7071, 0.0, 0.7071, 0.0),  # 90° Y rotation = looking down
                resolution=(1280, 720),
            )
            self.camera.initialize()
            self._initialized = True
            print(f"[DemoRecorder] Camera initialized at {cam_path}")
        except Exception as e:
            print(f"[DemoRecorder] Camera init failed: {e}")
            try:
                # Fallback: use viewport camera
                import omni.kit.viewport.utility as vp_utils
                viewport = vp_utils.get_active_viewport_window()
                if viewport:
                    self.camera = viewport
                    self._initialized = True
                    print("[DemoRecorder] Using viewport camera")
            except Exception as e2:
                print(f"[DemoRecorder] Viewport fallback also failed: {e2}")

    def tick(self, sim_time: float):
        """Called every simulation tick — capture frame if interval elapsed."""
        if not self._initialized:
            self._init_camera()

        if sim_time - self.last_capture < self.frame_interval:
            return
        self.last_capture = sim_time

        if self.camera is None:
            return

        try:
            # Try to get RGBA image from camera sensor
            if hasattr(self.camera, "get_rgba"):
                rgba = self.camera.get_rgba()
                if rgba is not None:
                    import numpy as np
                    from PIL import Image

                    img = Image.fromarray((rgba[:, :, :3] * 255).astype('uint8'))
                    filepath = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.png")
                    img.save(filepath)
                    self.frame_count += 1

                    if self.frame_count % 50 == 0:
                        print(f"[DemoRecorder] Captured {self.frame_count} frames")
            elif hasattr(self.camera, 'get_image'):
                # Alternative API
                img_data = self.camera.get_image()
                if img_data is not None:
                    import numpy as np
                    from PIL import Image

                    if img_data.dtype == np.uint8:
                        img = Image.fromarray(img_data)
                    else:
                        img = Image.fromarray((img_data * 255).astype('uint8'))

                    filepath = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.png")
                    img.save(filepath)
                    self.frame_count += 1
        except Exception as e:
            if self.frame_count == 0:
                print(f"[DemoRecorder] Capture failed: {e}")
            self.frame_count += 0  # don't increment on failure

    def finalize(self):
        """Called after simulation ends — create video with ffmpeg."""
        if self.frame_count == 0:
            print("[DemoRecorder] No frames captured — skipping video")
            return

        video_path = "/root/demo_video.mp4"
        cmd = (
            f"ffmpeg -y -framerate {self.fps} "
            f"-pattern_type glob -i '{self.output_dir}/frame_*.png' "
            f"-c:v libx264 -pix_fmt yuv420p -q:v 2 "
            f"{video_path}"
        )
        print(f"[DemoRecorder] Creating video: {cmd}")
        ret = os.system(cmd)
        if ret == 0:
            print(f"[DemoRecorder] Video saved: {video_path} ({self.frame_count} frames)")
        else:
            print(f"[DemoRecorder] ffmpeg failed with code {ret}")
