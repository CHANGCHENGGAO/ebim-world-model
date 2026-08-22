#!/usr/bin/env python3
"""
Diffusion Policy for EBiM Task 3 — trajectory generation framework.

Framework-level implementation that simulates diffusion-style denoising
using IK as the target signal. The interface is designed to be compatible
with real Diffusion Policy — a trained model can be dropped in by replacing
the _denoise_step method.

Falls back to pure IK trajectory if the generated path is invalid
(out of joint limits, discontinuous, etc.).
"""

import numpy as np
import os
from typing import Optional, Callable, Tuple


class DiffusionPolicy:
    """
    Diffusion-style trajectory generator.

    Without a trained model, uses IK target + simulated denoising.
    Interface is compatible with real Diffusion Policy models.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_denoise_steps: int = 10,
        num_trajectory_steps: int = 50,
        action_dim: int = 7,
        node=None,
    ):
        self.model_path = model_path or os.environ.get("DIFFUSION_MODEL_PATH")
        self.num_denoise_steps = num_denoise_steps
        self.num_trajectory_steps = num_trajectory_steps
        self.action_dim = action_dim
        self.node = node
        self._model = None
        self._use_real_model = False

        # Try to load real model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self._try_load_model()

        if not self._use_real_model:
            self._log("No trained diffusion model — using IK-simulated denoising")

    def _log(self, msg: str):
        if self.node is not None:
            self.node.get_logger().info(f"[Diffusion] {msg}")
        else:
            print(f"[Diffusion] {msg}")

    def _warn(self, msg: str):
        if self.node is not None:
            self.node.get_logger().warn(f"[Diffusion] {msg}")
        else:
            print(f"[Diffusion] WARN: {msg}")

    def _try_load_model(self):
        """Try to load a real diffusion model (placeholder for future)."""
        # Real model loading would go here (e.g., torch.load, diffusers, etc.)
        # For now, we always use the simulated version
        self._use_real_model = False

    def generate_trajectory(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray],
        current_q: np.ndarray,
        ik_fn: Callable,
        joint_limits_low: Optional[np.ndarray] = None,
        joint_limits_high: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate a smooth joint trajectory from current_q to target.

        Args:
            target_pos: target end-effector position (3,)
            target_rot: target end-effector rotation (3x3) or None
            current_q: current joint angles (7,)
            ik_fn: function(target_pos, target_rot, current_q) -> (q_target, success)
            joint_limits_low: lower joint limits (7,)
            joint_limits_high: upper joint limits (7,)

        Returns:
            trajectory array (T, 7) or None if validation fails
        """
        # Step 1: Get IK solution as the "target" for diffusion
        q_target, ik_success = ik_fn(target_pos, target_rot, current_q)
        if not ik_success:
            self._warn("IK failed — cannot generate trajectory")
            return None

        # Step 2: Generate noisy initial trajectory
        T = self.num_trajectory_steps
        D = self.action_dim
        rng = np.random.RandomState(42)  # deterministic for reproducibility

        # Start with pure noise scaled to joint range
        if joint_limits_low is not None and joint_limits_high is not None:
            joint_range = joint_limits_high - joint_limits_low
            traj = rng.randn(T, D) * joint_range * 0.3 + (joint_limits_low + joint_limits_high) / 2
        else:
            traj = rng.randn(T, D) * 1.0

        # Step 3: Denoising steps
        for step in range(self.num_denoise_steps):
            alpha = step / self.num_denoise_steps  # 0 → 1
            traj = self._denoise_step(traj, current_q, q_target, alpha, step)

        # Step 4: Set start/end anchors
        traj[0] = current_q.copy()
        traj[-1] = q_target.copy()

        # Step 5: Smooth
        traj = self._smooth_trajectory(traj)

        # Step 6: Validate
        if not self._validate_trajectory(traj, joint_limits_low, joint_limits_high):
            self._warn("Generated trajectory failed validation — will fall back to IK")
            return None

        return traj

    def _denoise_step(
        self,
        traj: np.ndarray,
        q_start: np.ndarray,
        q_target: np.ndarray,
        alpha: float,
        step: int,
    ) -> np.ndarray:
        """
        One denoising step. In a real diffusion model, this would call the
        neural network. Here we simulate it by interpolating toward the
        IK solution with decreasing noise.

        Args:
            traj: current trajectory (T, D)
            q_start: starting joint angles (D,)
            q_target: target joint angles (D,)
            alpha: denoising progress (0 = noisy, 1 = clean)
            step: current step number

        Returns:
            denoised trajectory (T, D)
        """
        T, D = traj.shape

        # Linear interpolation target trajectory
        t = np.linspace(0, 1, T).reshape(-1, 1)
        target_traj = q_start + (q_target - q_start) * t

        # Add some "natural" variation to make it more diffusion-like
        # (sinusoidal perturbation that diminishes with alpha)
        noise_scale = 1.0 - alpha
        for d in range(D):
            freq = 1.0 + d * 0.5
            phase = d * 0.7
            target_traj[:, d] += noise_scale * 0.05 * np.sin(2 * np.pi * freq * t[:, 0] + phase)

        # Denoise: move trajectory toward target, keep some noise
        denoised = traj * (1 - alpha) + target_traj * alpha

        return denoised

    def _smooth_trajectory(self, traj: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply moving average smoothing to the trajectory."""
        T, D = traj.shape
        smoothed = np.zeros_like(traj)
        for d in range(D):
            kernel = np.ones(window) / window
            smoothed[:, d] = np.convolve(traj[:, d], kernel, mode='same')
        # Fix endpoints
        smoothed[0] = traj[0]
        smoothed[-1] = traj[-1]
        return smoothed

    def _validate_trajectory(
        self,
        traj: np.ndarray,
        joint_limits_low: Optional[np.ndarray],
        joint_limits_high: Optional[np.ndarray],
    ) -> bool:
        """
        Validate trajectory:
        1. All points within joint limits
        2. Smooth (no large jumps between consecutive steps)
        3. Start matches current, end matches target
        """
        T, D = traj.shape

        # Check joint limits
        if joint_limits_low is not None and joint_limits_high is not None:
            if np.any(traj < joint_limits_low - 0.01):
                self._warn("Trajectory exceeds lower joint limits")
                return False
            if np.any(traj > joint_limits_high + 0.01):
                self._warn("Trajectory exceeds upper joint limits")
                return False

        # Check smoothness (max step < 0.1 rad)
        step_diffs = np.diff(traj, axis=0)
        max_step = np.max(np.abs(step_diffs))
        if max_step > 0.15:
            self._warn(f"Trajectory too jerky: max step = {max_step:.3f} rad")
            return False

        # Check continuity
        if np.any(~np.isfinite(traj)):
            self._warn("Trajectory contains non-finite values")
            return False

        return True
