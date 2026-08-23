# Repository Submission — 提交内容

## 操作步骤

1. 打开: https://github.com/EBiM-Benchmark/submissions/issues/new/choose
2. 选择 **"Repository Submission"** 表单（不是 Technical Report）
3. 复制下面每个字段的内容到对应输入框
4. 勾选 Acknowledgement 复选框
5. 点击 "Submit new issue"

---

## 字段 1: Team name

```
world model
```

## 字段 2: Point-of-contact email

```
1373851641@qq.com
```

## 字段 3: Task

选择: **Task 3 — Assisted Living & Feeding**

## 字段 4: Public GitHub repository URL

```
https://github.com/CHANGCHENGGAO/ebim-world-model
```

## 字段 5: Did this submission use the simulator's ground-truth object poses?

选择: **Yes — we use the simulator's ground-truth object poses**

(注：同时也实现了 YOLO 视觉检测，用于交叉验证和防瞬移安全检查，但主要坐标来源是仿真真值以保证得分可靠性)

## 字段 6: Submission requirements

三个复选框全部勾选:
- ☑ The repository is public.
- ☑ It contains a Dockerfile encapsulating our work.
- ☑ It contains a README explaining how to run it.

## 字段 7: Optional supplementary links

```
Technical Report: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/Technical_Report_World_model.md
Policy module design: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/docs/policy_modules_design.md
Judge evaluation: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/docs/judge_evaluation.html
```

## 字段 8: Notes (optional)

```
All four stages run fully autonomously at full score (18/18 development grading, 16/16 official).

Key features:
• Bimanual coordination — Stage 2: right arm steadies bowl, left arm scoops and feeds
• ISO/TS 15066 safety monitoring — 140N head force threshold with real-time checking
• Closed-loop navigation — odometry-corrected with 3-attempt correction loop
• YOLO vision detection — real-time object detection from Isaac Sim camera
• Local LLM planning — Qwen2.5-3B GGUF, llama-cpp-python, 0.2-0.3s/call (GPU)
• Diffusion Policy framework — optional trajectory generation with IK fallback
• 4-level fallback chain — verified: all AI failures degrade gracefully without score loss

This submission supersedes our earlier Technical Report (Issue #22). The code has been upgraded from a design-phase report to a fully working 18/18 autonomous controller.
```

## 字段 9: Acknowledgement

勾选:
- ☑ I understand source code is not required, but a Dockerfile and a run README are.
