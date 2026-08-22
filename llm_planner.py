#!/usr/bin/env python3
"""
LLM Planner for EBiM Task 3 — local LLM inference via llama-cpp-python.

Given current state and a goal, outputs a structured action command in JSON.
Fails gracefully (returns None) on timeout, parse error, or invalid output.
"""

import json
import time
import threading
import os
from typing import Optional, Dict, Any

# Valid action types (whitelist)
VALID_ACTIONS = {"grasp", "place", "move_arm", "navigate", "wait"}

SYSTEM_PROMPT = """你是一个机器人操作规划专家。你控制一个双臂移动机器人，需要完成桌面操作任务。

【机器人能力】
- 左右两个7自由度机械臂
- 移动底座（前后/左右/旋转）
- 夹爪开合
- 视觉系统可检测物体位置

【输出要求】
1. 只输出一个JSON对象，不要任何其他文字、解释或markdown格式
2. JSON必须包含: action, arm, target_object, params, reasoning
3. action只能是: grasp, place, move_arm, navigate, wait
4. arm只能是: left, right, both

【JSON格式示例】
{"action": "grasp", "arm": "left", "target_object": "bowl2", "params": {"approach_height": 0.15}, "reasoning": "用左手抓取碗，路径更短"}
"""


class LLMPlanner:
    """Local LLM planner with llama-cpp-python backend."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        timeout: float = 5.0,
        n_ctx: int = 2048,
        n_gpu_layers: Optional[int] = None,
        node=None,
    ):
        self.model_path = model_path or os.environ.get(
            "LLM_MODEL_PATH",
            "/root/models/qwen2.5-3b-instruct-q4_k_m.gguf"
        )
        self.timeout = timeout
        self.n_ctx = int(os.environ.get("LLM_N_CTX", n_ctx))
        if n_gpu_layers is None:
            n_gpu_layers = int(os.environ.get("LLM_N_GPU_LAYERS", -1))
        self.n_gpu_layers = n_gpu_layers
        self.node = node
        self._model = None
        self._load_model()

    def _log(self, msg: str):
        if self.node is not None:
            self.node.get_logger().info(f"[LLM] {msg}")
        else:
            print(f"[LLM] {msg}")

    def _warn(self, msg: str):
        if self.node is not None:
            self.node.get_logger().warn(f"[LLM] {msg}")
        else:
            print(f"[LLM] WARN: {msg}")

    def _load_model(self):
        """Load the LLM model. Returns True on success."""
        try:
            from llama_cpp import Llama
        except ImportError:
            self._warn("llama-cpp-python not installed — LLM planner unavailable")
            return False

        if not os.path.exists(self.model_path):
            self._warn(f"Model file not found: {self.model_path} — LLM planner unavailable")
            return False

        try:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            self._log(f"Model loaded: {self.model_path}")
            return True
        except Exception as e:
            self._warn(f"Failed to load model: {e}")
            self._model = None
            return False

    @property
    def available(self) -> bool:
        return self._model is not None

    def plan(self, state: Dict[str, Any], goal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run LLM planning with timeout.

        Args:
            state: Current robot/environment state dict
            goal: Goal description dict with 'description' key

        Returns:
            Parsed action dict, or None on failure
        """
        if not self.available:
            return None

        prompt = self._build_prompt(state, goal)
        response_text = None

        def _inference():
            nonlocal response_text
            try:
                output = self._model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=512,
                    temperature=0.3,
                    stop=["\n\n", "```"],
                )
                if output and "choices" in output and len(output["choices"]) > 0:
                    response_text = output["choices"][0]["message"]["content"]
            except Exception as e:
                self._warn(f"Inference error: {e}")
                response_text = None

        # Run inference in a thread with timeout
        thread = threading.Thread(target=_inference, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            self._warn(f"Inference timed out after {self.timeout}s")
            return None

        if response_text is None:
            return None

        return self._parse_response(response_text)

    def _build_prompt(self, state: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """Build the user prompt from state and goal."""
        lines = []

        lines.append("【当前状态】")

        # Item positions
        items = state.get("item_positions", {})
        if items:
            lines.append("物品位置:")
            for name, pos in items.items():
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    lines.append(f"  - {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Robot base
        base_pos = state.get("base_pos")
        base_yaw = state.get("base_yaw")
        if base_pos is not None:
            lines.append(f"底座位置: ({base_pos[0]:.3f}, {base_pos[1]:.3f}), yaw={base_yaw:.1f}°")

        # Gripper state
        gripper = state.get("gripper", {})
        lines.append(f"夹爪状态: 左={'闭合' if gripper.get('left', 0) > 0.4 else '张开'}, "
                     f"右={'闭合' if gripper.get('right', 0) > 0.4 else '张开'}")

        # Goal
        lines.append("\n【当前目标】")
        lines.append(goal.get("description", "完成当前任务"))

        if "available_objects" in goal:
            lines.append(f"可操作物体: {', '.join(goal['available_objects'])}")

        lines.append("\n请输出下一步动作的JSON:")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into a structured action dict."""
        text = text.strip()

        # Try to extract JSON from code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # Try direct JSON parse
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    self._warn(f"Failed to parse JSON: {text[:100]}...")
                    return None
            else:
                self._warn(f"No JSON found in response: {text[:100]}...")
                return None

        # Validate required fields
        if not isinstance(result, dict):
            self._warn("Response is not a JSON object")
            return None

        action = result.get("action")
        if action not in VALID_ACTIONS:
            self._warn(f"Invalid action: {action}")
            return None

        arm = result.get("arm", "left")
        if arm not in ("left", "right", "both"):
            self._warn(f"Invalid arm: {arm}")
            return None

        # Ensure params is a dict
        if "params" not in result or not isinstance(result["params"], dict):
            result["params"] = {}

        return result
