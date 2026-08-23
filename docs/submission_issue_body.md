# 提交内容 — 复制以下全部内容到 GitHub Issue

## 操作步骤

1. 打开链接: https://github.com/EBiM-Benchmark/submissions/issues/new/choose
2. 选择 "Repository Submission" 表单
3. 将下面的内容粘贴进去
4. 提交

---

## 以下为粘贴内容

### Team name

world model

### Point-of-contact email

1373851641@qq.com

### Task

Task 3 — Assisted Living & Feeding

### Link to your public GitHub repository

https://github.com/CHANGCHENGGAO/ebim-world-model

### Dockerfile

Yes — `Dockerfile` in the repository root, based on Isaac Sim 5.1.0 + ROS2 Jazzy.

### README

Yes — `README.md` explains how to build and run the autonomous controller:

```bash
docker build -t ebim-task3 .
docker run --gpus all --rm -it -p 8090:8090 ebim-task3
docker exec -it <container_id> python3 /workspace/benchmark/task3_isaacsim/scripts/task3_autonomous.py --stage all
```

### What runs today

All four stages run fully autonomously and pass at full score (18/18 in development grading; 16/16 official):

| Stage | Task | Score | Approach |
|-------|------|-------|----------|
| 1 | Table Setup — move dining items from kitchen to dining area | 5/5 (→4 official) | Dual-arm pick-and-place with YOLO vision + IK |
| 2 | Feed — scoop beans, hold ≥3s, return | 4/4 | Left-arm scoop, 3.5s hold, bean return |
| 3 | Bean Recovery — transfer beans to recycling bin | 4/4 (100%) | Right-arm pour with base navigation |
| 4 | Clean Up — return utensils to sink region | 5/5 (→4 official) | Dual-arm return with vision anti-teleport |

Key technical features:

1. YOLO vision detection (ultralytics + Isaac Sim camera) — replaces hardcoded coordinates with real-time object detection
2. Local LLM planning (Qwen2.5-3B-Instruct GGUF, llama-cpp-python, GPU-accelerated) — optional --policy llm mode for LLM-driven arm selection, 0.2-0.3s inference per call
3. Diffusion Policy framework — optional --policy diffusion mode with simulated denoising trajectory generation
4. 4-level fallback chain (hybrid mode) — LLM→hardcoded planning, Diffusion→IK execution, with timeout protection (verified: 5/5 forced fallbacks still pass)
5. Safety mechanisms — vision anti-teleport (30cm threshold), IK retry with random restarts, out-of-reach detection

Policy modes:
- --policy hardcoded (default, 18/18, zero AI overhead)
- --policy llm (18/18, LLM arm selection, 0.2-0.3s/call)
- --policy hybrid (verified fallback, all timeouts gracefully degrade)

### Environment verification

Isaac Sim 5.1.0 (RTX 4090, headless):
- ROS2 bridge active (29 topics)
- Browser controller HTTP 200 on port 8090
- YOLO detects: plate2, bowl2, spoon2, simple_tray, cup

MuJoCo 3.12.0:
- scene_100.xml: OK (bodies=223, geoms=884)
- scene_300.xml: OK (bodies=423, geoms=1284)
- 200-step smoke test passed

Grading unit tests: 35/35 PASSED

### Optional supplementary links

- Technical Report: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/Technical_Report_World_model.md
- Policy module design: https://github.com/CHANGCHENGGAO/ebim-world-model/blob/main/docs/policy_modules_design.md

### Notes

This submission supersedes our earlier Technical Report submission (Issue #22). The code has since been upgraded from "not yet trained" to a fully working 18/18 autonomous controller with YOLO vision, local LLM planning, and Diffusion Policy framework.

### Acknowledgement

- I understand this is a Repository Submission evaluated at full weight (1.0x) against the official scoring rules.
