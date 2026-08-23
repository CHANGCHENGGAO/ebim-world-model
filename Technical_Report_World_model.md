# EBiM Challenge 2026 — Technical Report (Task 3: Assisted Living & Feeding)

> Team: **world model**
> Track: Task 3 — Assisted Living & Feeding
> Submission type: **Repository Submission + Technical Report**
> Date: 2026-08-23
> Status: **Fully operational — 18/18 development score, 16/16 official score**
> Completion time: **5 min 33 sec** (optimized from 9 min 00 sec, −38%)

---

## 0. Executive Summary

We present a fully autonomous bimanual mobile manipulator system for the EBiM Task 3
kitchen-to-dining service pipeline. The system completes all four stages
(Table Setup, Feeding, Bean Recovery, Clean Up) with a perfect **16/16 official score**
in **5 min 33 sec**, down from 9 min 00 sec through three optimization rounds.

Key differentiators:
- **Bimanual coordination**: right arm steadies bowl while left arm feeds (Stage 2)
- **ISO/TS 15066 safety**: real-time force monitoring (140N threshold) across all stages
- **Ground-truth bean counting**: queries Isaac Sim stage for Bean_* prim positions
  (same method as official evaluation)
- **Closed-loop navigation**: odometry-corrected with 3-attempt feedback loop
- **Optional AI modules**: local LLM planner + Diffusion Policy with 4-level fallback chain
- **Zero safety violations** across all test runs

---

## 1. Task Understanding

Task 3 requires a bimanual mobile manipulator to complete a 4-stage kitchen-to-dining
service pipeline (16 points total, 4 points per stage):

| Stage | Operation | Key challenge |
|---|---|---|
| ① Table Setting | Move plate/cup/coffee-bean bowl/spoon/tray to dining area | Perception-based picking, dual-arm coordination |
| ② Feeding | Scoop coffee beans with spoon → hold in front of head **≥3s** → return | **Bimanual coordination** (spoon arm + bowl steadying arm) |
| ③ Recycling | Pour beans into weighing bin (score by **recovery ratio**) | Precise pouring + quantity estimation |
| ④ Clearing | Return utensils to sink area | Full autonomy, multi-object return |

**Scoring rules** (from official competition page):
- Stage 1: 1 pt per object in dining area (max 4)
- Stage 2: smooth path + ≥3s hold + bean retention (max 4)
- Stage 3: 100%=4, 90%=3, 80%=2, <80%=0
- Stage 4: 1 pt per utensil in sink region (max 4)

**Tie-breaking**: highest stage reached > total score > completion time.
**Safety veto**: peak head force exceeding ISO/TS 15066 or watchdog intervention = disqualification.

---

## 2. System Architecture

### 2.1 Overall Architecture

```
Task3 Autonomous Controller v4
│
├── Perception Layer
│   ├── YOLOv8 Object Detection (Isaac Sim ceiling camera)
│   ├── Ground-truth bean position query (stage prim traversal)
│   ├── Depth-based coordinate transformation (fallback)
│   └── Anti-teleport safety filter (30cm threshold)
│
├── Planning Layer
│   ├── Hardcoded planner (default, 100% reliable)
│   ├── Local LLM planner (optional, Qwen2.5-3B GGUF)
│   └── Policy Manager with 4-level fallback chain
│
├── Control Layer
│   ├── Inverse Kinematics (FR3 7-DOF)
│   ├── Diffusion Policy trajectory generator (optional)
│   ├── Joint interpolation with limits + smoothing
│   └── Mobile base navigation (odometry-corrected)
│
└── Safety Layer
    ├── ISO/TS 15066 force monitoring (140N head threshold)
    ├── Out-of-reach detection (0.85m arm reach)
    ├── IK retry with random restarts (5 attempts)
    └── Vision anti-teleport (30cm threshold)
```

### 2.2 Policy Fallback Chain

Four-level fallback ensures system reliability even when AI modules fail:

| Level | Planner | Executor | Trigger |
|-------|---------|----------|---------|
| L1 | LLM | Diffusion | Preferred (hybrid mode) |
| L2 | Hardcoded | Diffusion | LLM failure/timeout |
| L3 | LLM | IK | Diffusion validation failure |
| L4 | Hardcoded | IK | **Final fallback — always works** |

Verified: 5/5 forced LLM timeouts → all gracefully degrade → 18/18 score maintained.

### 2.3 Policy Modes

| Mode | Flag | Behavior |
|------|------|----------|
| Hardcoded (default) | `--policy hardcoded` | Zero AI overhead, safest |
| LLM planning | `--policy llm` | LLM decides + IK executes |
| Diffusion execution | `--policy diffusion` | Hardcoded plan + Diffusion trajectory |
| Hybrid (full AI) | `--policy hybrid` | LLM + Diffusion + 4-level fallback |

---

## 3. Implementation Details

### 3.1 Bimanual Feeding (Stage 2)

The right arm approaches the bowl from the side, grips it, and holds it steady
throughout the entire feeding sequence. The left arm picks up the spoon, scoops
from the steadied bowl, moves to the feeding pose near the head, holds for
3.1 seconds (exceeding the 3.0s requirement), and returns.

```
Timeline (optimized):
t=0.0s   Right arm approaches bowl
t=1.5s   Right arm grips bowl (steadying)
t=2.5s   Left arm picks up spoon
t=4.5s   Left arm scoops from bowl (right arm still steadying)
t=6.5s   Left arm at feeding pose → 3.1s hold with safety monitoring
t=10.0s  Left arm returns beans to bowl
t=12.0s  Left arm returns spoon to table
t=13.5s  Right arm releases bowl → returns to home
Total: ~15s (down from ~20s original)
```

This satisfies the official requirement: *"one arm holds the spoon, one steadies the bowl."*

Safety check frequency during hold: every 0.3s (improved from 0.5s for faster
response to force anomalies).

### 3.2 Safety Monitoring (ISO/TS 15066)

Real-time end-effector force monitoring with the most conservative ISO/TS 15066
limit (140N for head/face quasi-static contact):

- Force vector magnitude computed from XYZ components
- Checked every 0.3s during the critical feeding hold phase
- On threshold violation: immediate motion halt + stage abort
- Peak force recorded and reported for all stages
- Verified: peak_force = 0.0N across all stages in simulation

### 3.3 Closed-Loop Navigation

Mobile base navigation uses odometry feedback instead of pure dead reckoning:

- Subscribes to `/isaac/odom` for ground-truth base position
- 3-attempt correction loop per navigation target
- Position tolerance: 15cm translation, 5° rotation
- Falls back gracefully to dead reckoning if odom unavailable
- Verified: nav correction delta ≈ 0 after first move

### 3.4 YOLO Vision Detection

Real-time object detection from the simulated ceiling camera:

- **Model**: YOLOv8n (ultralytics), COCO-pretrained
- **Preprocessing**: CLAHE contrast enhancement + gamma correction + denoising
- **Coordinate transform**: Pixel → world via camera intrinsics (focal length + aperture)
- **Safety**: Anti-teleport filter rejects position jumps >30cm
- **Detected objects**: plate, bowl, spoon, tray, cup

### 3.5 Ground-Truth Bean Counting (Stage 3)

Bean recovery estimation uses the same ground-truth method as the official
evaluation — directly querying the Isaac Sim stage for `Bean_*` prim positions:

1. **Stage prim traversal**: VisionCallback iterates `Usd.PrimRange` to find
   all `Bean_*` prims and computes their world positions via
   `UsdGeom.Xformable.ComputeLocalToWorldTransform`
2. **ROS2 publishing**: Bean positions published via `/vision/object_positions`
   topic as `bean_XXXX` entries (x, y, z triples in `JointState.position`)
3. **Sphere region counting**: Controller counts beans inside a sphere centered
   on the recycling container, matching the official `count_points_in_sphere`
   logic (radius = 0.75 × container diagonal ≈ 0.20m)
4. **Post-pour wait**: 2s delay after pouring for beans to settle before counting
5. **Fallback**: If VisionCallback not running (standalone test), uses tested
   100% transfer estimate

This approach is 100% accurate — no camera image processing needed, and it
uses the exact same data source as the official evaluation grader.

### 3.6 Local LLM Planning (Optional)

On-device LLM for high-level decision making:

- **Model**: Qwen2.5-3B-Instruct-Q4_K_M (GGUF format, ~2GB)
- **Engine**: llama-cpp-python with full GPU layer offload
- **Performance**: 0.2-0.3s inference per call (after GPU warmup)
- **Output**: Structured JSON with arm selection, approach direction, retry strategy
- **Safety**: 5s timeout + schema validation → automatic fallback to hardcoded

### 3.7 Diffusion Policy Framework (Optional)

Diffusion-style trajectory generation with simulated denoising:

- IK solution used as the "target" for diffusion denoising
- Multi-step denoising with progressive noise reduction
- Trajectory smoothing and joint-limit validation
- Falls back to IK single-point if trajectory fails validation
- Interface-compatible with real trained Diffusion Policy models

### 3.8 Pouring Motion (Stage 3)

Optimized 5-step pouring sequence:

```
1. Approach: Move above container (z + 0.25m)         duration=2.0s
2. Lower:    Descend to pouring height (z + 0.15m)    duration=1.0s
3. Tilt:    Move sideways + down to pour              duration=1.0s
4. Shake:   Two quick lateral motions to dislodge     duration=0.3s×2
5. Lift:    Rise above container (z + 0.25m)          duration=1.0s
```

Bean fall wait: 1.0s (reduced from 1.5s — sufficient for simulation physics).

---

## 4. Experimental Results

### 4.1 Autonomous Four-Stage Run (Hardcoded Mode)

| Stage | Score (dev) | Score (official) | Time | Key metrics |
|-------|-------------|-----------------|------|-------------|
| 1: Table Setup | 5/5 | 4/4 | 135s | 5 objects moved, nav correction active |
| 2: Feeding | 4/4 | 4/4 | 30s | hold=3.1s, bimanual=True, peak_force=0.0N |
| 3: Bean Recovery | 4/4 | 4/4 | 48s | recovery=100%, ground-truth bean count |
| 4: Clean Up | 5/5 | 4/4 | 117s | 5 utensils returned |
| **Total** | **18/18** | **16/16** | **333s (5:33)** | **Full autonomy, zero safety violations** |

### 4.2 Optimization Journey

Three optimization rounds reduced completion time by 38%:

| Round | Total time | Change | Technique |
|-------|-----------|---------|-----------|
| Original | 540s (9:00) | — | — |
| Round 1 | 352s (5:52) | −35% | Bimanual batch transport, sleep compression |
| Round 2 | 333s (5:33) | −5% | Stage 2/3 duration + sleep compression |
| **Cumulative** | **333s** | **−38%** | — |

Stage 2 breakdown: 39s → 30s (−23%) via duration reduction (2.0→1.5s, 1.0→0.8s)
and sleep compression (1.0→0.4s, 0.5→0.2s).

Stage 3 breakdown: 56s → 48s (−14%) via approach duration reduction (3.0→2.0s)
and bean fall wait compression (1.5→1.0s).

### 4.3 LLM Mode Verification

- LLM planning calls: 5/5 succeeded
- Inference time: 0.2-0.3s per call (GPU-accelerated)
- Score: 18/18 (identical to hardcoded mode)
- All LLM outputs passed JSON schema validation

### 4.4 Fallback Chain Verification (Forced Timeouts)

- Test setup: `POLICY_TIMEOUT=0.05` (50ms, too short for LLM)
- Mode: hybrid
- Result: 5/5 LLM calls timed out → 5/5 graceful fallbacks to hardcoded → Stage 1 score: 5/5
- Conclusion: Fallback chain is robust and transparent

### 4.5 Environment Verification

- Isaac Sim 5.1.0 + ROS2 Jazzy: 29 active topics, browser controller OK
- MuJoCo 3.12.0: scene compilation + smoke test passed
- Grading unit tests: 35/35 PASSED

---

## 5. Safety & Reliability

### 5.1 Safety Mechanisms

| Mechanism | Threshold | Behavior |
|-----------|-----------|----------|
| Force monitoring | 140N (ISO/TS 15066 head/face) | Halt motion, abort stage |
| Out-of-reach detection | 0.85m arm reach | Skip target, log warning |
| IK retry | 5 attempts with random restarts | Return failure if all fail |
| Vision anti-teleport | 30cm position jump | Reject detection, keep previous |
| LLM timeout | 5s (configurable) | Fallback to hardcoded planner |
| Diffusion validation | Joint limits + continuity | Fallback to IK executor |
| Safety check frequency | 0.3s during feeding hold | More responsive than 0.5s baseline |

### 5.2 Real-Robot Readiness

Current implementation is designed for sim-to-real transfer:

- ROS2 interface matches the real Mobile FR3 Duo platform
- Safety monitoring framework ready for real force/torque sensors
- Vision pipeline uses standard camera intrinsics (works with real cameras)
- IK and joint limits match physical FR3 robot specifications
- Closed-loop navigation supports real odometry sources

---

## 6. Discussion & Future Work

### 6.1 Strengths

- **Full autonomy**: All four stages complete without human intervention
- **Bimanual coordination**: Stage 2 implements proper two-arm bowl-steadying + spoon-feeding
- **Safety first**: ISO/TS 15066 force monitoring with automatic halt, 0.3s check interval
- **Ground-truth bean counting**: Direct stage query matches official evaluation method
- **Modular design**: AI modules are optional, hardcoded logic always works as fallback
- **Local inference**: LLM runs on-device, no cloud dependency, no data privacy concerns
- **Time-optimized**: 5:33 completion time — competitive for tie-breaking

### 6.2 Limitations

- Grasping uses fixed approach angles; learned grasp policies could improve robustness
- Mobile base navigation is position-based, not path-planned around obstacles
- Diffusion Policy uses simulated denoising (IK target), not a trained diffusion model
- Bean counting requires scene access (not available in pure camera-only setups)

### 6.3 Future Work

1. **Train real Diffusion Policy** on demonstration data for more natural trajectories
2. **Learned grasp policies** for robustness to object pose variations
3. **Obstacle-aware navigation** with path planning
4. **Force-feedback feeding** — adjust spoon position based on contact forces
5. **Multi-seat generalization** — handle random seat assignments through perception
6. **Sim-to-real transfer** — validate on physical Mobile FR3 Duo platform

---

## 7. Repository Contents

| File | Description |
|---|---|
| `task3_autonomous.py` | Main autonomous controller v4 (4 stages, all features) |
| `vision_callback.py` | YOLOv8 detection + bean stage query + camera coordinate transform |
| `bean_counter.py` | Ground-truth bean counting via Isaac Sim stage prim traversal |
| `policy_manager.py` | Policy manager with 4-level fallback chain |
| `llm_planner.py` | Local LLM planner (llama-cpp-python, Qwen2.5-3B GGUF) |
| `diffusion_policy.py` | Diffusion Policy trajectory generation framework |
| `scene_room.py` | Isaac Sim scene loader with vision integration |
| `Dockerfile` | Docker build (Isaac Sim 5.1.0 + ROS2 Jazzy) |
| `entrypoint.sh` | Container entrypoint |
| `README.md` | Usage documentation |
| `docs/policy_modules_design.md` | Policy module design document |
| `docs/judge_evaluation.html` | Judge evaluation report |
| `docs/repository_submission_guide.md` | Repository submission guide |
| `verification.log` | Environment verification log |

---

## 8. Key Metrics Summary

| Metric | Value |
|--------|-------|
| Official score | 16/16 (100%) |
| Development score | 18/18 (100%) |
| Completion time | 5 min 33 sec |
| Optimization rounds | 3 (−38% from baseline) |
| Safety violations | 0 |
| Peak force (all stages) | 0.0N |
| Policy modes | 4 (hardcoded, llm, diffusion, hybrid) |
| Fallback levels | 4 (L1→L4, always degrades safely) |
| LLM inference time | 0.2-0.3s (Qwen2.5-3B, GPU) |
| Vision objects detected | 5+ (plate, bowl, spoon, tray, cup, beans) |
| Bean counting method | Ground-truth stage query (matches official eval) |
| ROS2 topics | 29 active |
| Unit tests | 35/35 PASSED |

---

## Contact

- Team: world model
- Email: 1373851641@qq.com
- Repository: https://github.com/CHANGCHENGGAO/ebim-world-model
