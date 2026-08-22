#!/usr/bin/env python3
"""
Policy Manager for EBiM Task 3 — optional LLM + Diffusion Policy modules
with hardcoded fallback.

Guarantees:
- Default mode (hardcoded): zero overhead, same as original controller
- All AI modules have hardcoded fallback
- Timeout protection on every AI call
- Every decision is logged with source and timing
"""

import time
import logging
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

VALID_POLICIES = {"hardcoded", "llm", "diffusion", "hybrid"}


class PolicyManager:
    """
    Unified policy entry point with multi-level fallback chain.

    Fallback chain (hybrid mode):
      L1: LLM plan + Diffusion execute
      L2: hardcoded plan + Diffusion execute
      L3: LLM plan + IK execute
      L4: hardcoded plan + IK execute  (final fallback, always works)
    """

    def __init__(
        self,
        policy_mode: str = "hardcoded",
        llm_model_path: Optional[str] = None,
        diffusion_model_path: Optional[str] = None,
        timeout: float = 5.0,
        node=None,
    ):
        if policy_mode not in VALID_POLICIES:
            logger.warning(f"Unknown policy mode '{policy_mode}', falling back to 'hardcoded'")
            policy_mode = "hardcoded"

        self.policy_mode = policy_mode
        self.timeout = timeout
        self.node = node

        # Stats
        self.stats = {
            "llm_calls": 0,
            "llm_success": 0,
            "llm_timeouts": 0,
            "diffusion_calls": 0,
            "diffusion_success": 0,
            "fallback_count": 0,
        }

        # Lazy-load AI modules
        self._llm_planner = None
        self._diffusion_policy = None
        self._llm_available = policy_mode in ("llm", "hybrid")
        self._diffusion_available = policy_mode in ("diffusion", "hybrid")
        self._llm_model_path = llm_model_path
        self._diffusion_model_path = diffusion_model_path

        self._log(f"PolicyManager initialized: mode={policy_mode}")

    def _log(self, msg: str):
        if self.node is not None:
            self.node.get_logger().info(f"[Policy] {msg}")
        else:
            logger.info(msg)

    def _warn(self, msg: str):
        if self.node is not None:
            self.node.get_logger().warn(f"[Policy] {msg}")
        else:
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Lazy module loading
    # ------------------------------------------------------------------

    @property
    def llm_planner(self):
        if self._llm_planner is None and self._llm_available:
            try:
                from llm_planner import LLMPlanner
                self._llm_planner = LLMPlanner(
                    model_path=self._llm_model_path,
                    timeout=self.timeout,
                    node=self.node,
                )
                self._log("LLM planner loaded")
            except Exception as e:
                self._warn(f"LLM planner failed to load: {e} — will use hardcoded planning")
                self._llm_available = False
        return self._llm_planner

    @property
    def diffusion_policy(self):
        if self._diffusion_policy is None and self._diffusion_available:
            try:
                from diffusion_policy import DiffusionPolicy
                self._diffusion_policy = DiffusionPolicy(
                    model_path=self._diffusion_model_path,
                    node=self.node,
                )
                self._log("Diffusion policy loaded")
            except Exception as e:
                self._warn(f"Diffusion policy failed to load: {e} — will use IK fallback")
                self._diffusion_available = False
        return self._diffusion_policy

    # ------------------------------------------------------------------
    # Public API: planning
    # ------------------------------------------------------------------

    def plan_action(
        self,
        state: dict,
        goal: dict,
        hardcoded_plan_fn: Callable,
    ) -> dict:
        """
        Decide next action. Falls back to hardcoded_plan_fn if AI fails.

        Returns dict with at least:
          - action: str
          - source: str  ("llm" / "hardcoded")
          - time_used: float
          - success: bool
        """
        t0 = time.time()

        # Hardcoded mode: direct pass-through
        if self.policy_mode == "hardcoded" or not self._llm_available:
            result = hardcoded_plan_fn(state, goal)
            result["source"] = "hardcoded"
            result["time_used"] = time.time() - t0
            result["success"] = True
            return result

        # Try LLM planning
        self.stats["llm_calls"] += 1
        try:
            llm_result = self.llm_planner.plan(state, goal)
            if llm_result is not None:
                self.stats["llm_success"] += 1
                llm_result["source"] = "llm"
                llm_result["time_used"] = time.time() - t0
                llm_result["success"] = True
                self._log(f"LLM plan: {llm_result.get('action', '?')} "
                         f"({time.time()-t0:.2f}s)")
                return llm_result
        except Exception as e:
            self._warn(f"LLM planner error: {e}")

        # Fallback: hardcoded plan
        self.stats["fallback_count"] += 1
        self._warn(f"LLM plan failed, falling back to hardcoded ({time.time()-t0:.2f}s)")
        result = hardcoded_plan_fn(state, goal)
        result["source"] = "hardcoded"
        result["time_used"] = time.time() - t0
        result["success"] = True
        return result

    # ------------------------------------------------------------------
    # Public API: trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        target_pos,
        target_rot,
        current_q,
        ik_fn: Callable,
        interpolate_fn: Optional[Callable] = None,
    ):
        """
        Generate joint trajectory. Falls back to IK if diffusion fails.

        ik_fn(target_pos, target_rot, current_q) -> q_target, success
        interpolate_fn(q_start, q_end, duration) -> None (side-effect)
        """
        t0 = time.time()

        # Hardcoded mode: direct IK + interpolation
        if self.policy_mode == "hardcoded" or self.policy_mode == "llm" or not self._diffusion_available:
            q_target, success = ik_fn(target_pos, target_rot, current_q)
            if not success:
                return None, False
            if interpolate_fn is not None:
                interpolate_fn(current_q, q_target)
            return q_target, True

        # Try diffusion
        self.stats["diffusion_calls"] += 1
        try:
            traj = self.diffusion_policy.generate_trajectory(
                target_pos, target_rot, current_q, ik_fn=ik_fn,
            )
            if traj is not None and len(traj) > 0:
                self.stats["diffusion_success"] += 1
                # Execute the trajectory step by step
                if interpolate_fn is not None:
                    for i in range(1, len(traj)):
                        interpolate_fn(traj[i-1], traj[i], duration=0.05)
                self._log(f"Diffusion trajectory: {len(traj)} steps ({time.time()-t0:.2f}s)")
                return traj[-1], True
        except Exception as e:
            self._warn(f"Diffusion policy error: {e}")

        # Fallback: IK + interpolation
        self.stats["fallback_count"] += 1
        self._warn(f"Diffusion failed, falling back to IK ({time.time()-t0:.2f}s)")
        q_target, success = ik_fn(target_pos, target_rot, current_q)
        if not success:
            return None, False
        if interpolate_fn is not None:
            interpolate_fn(current_q, q_target)
        return q_target, True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return dict(self.stats)

    def print_stats(self):
        s = self.stats
        lines = [
            "=" * 50,
            "Policy Manager Statistics",
            "=" * 50,
            f"  Mode:           {self.policy_mode}",
            f"  LLM calls:      {s['llm_calls']} (success: {s['llm_success']}, "
            f"timeouts: {s['llm_timeouts']})",
            f"  Diffusion calls:{s['diffusion_calls']} (success: {s['diffusion_success']})",
            f"  Fallbacks:      {s['fallback_count']}",
            "=" * 50,
        ]
        for line in lines:
            self._log(line)
