# EBiM Challenge 2026 — Technical Report (Task 3: Assisted Living & Feeding)

> Team: **world model**
> Track: Task 3 — Assisted Living & Feeding
> Submission type: **Technical Report**
> Date: 2026-08-21

---

## 1. Task Understanding

Task 3 requires a bimanual mobile manipulator to complete a 4-stage kitchen-to-dining
service pipeline (16 points total, 4 points per stage):

| Stage | Operation | Key challenge |
|---|---|---|
| ① Table Setting | Move plate/cup/coffee-bean bowl/spoon to **3 randomly chosen seats** out of 6 | Randomized targets require robust perception + re-planning |
| ② Feeding | Scoop coffee beans with spoon → hold in front of head **≥3s** → return | **Bimanual coordination** (spoon arm + bowl steadying arm) |
| ③ Recycling | Pour beans into weighing bin (score by **recovery ratio**) | Precise pouring + quantity estimation |
| ④ Clearing | Return 4 utensils to sink area | Full autonomy |

Tie-breaking: highest stage reached > total score > completion time.
**Safety veto**: peak head force exceeding ISO/TS 15066 or watchdog intervention = disqualification.

---

## 2. Proposed Approach

We propose a **modular, LLM-orchestrated pipeline** combining classical motion primitives
with learned perception and imitation-learned bimanual skills:

### 2.1 Perception Layer
- RGB-D perception (front-mounted + wrist cameras)
- Object detection/segmentation (fine-tuned YOLO/DETR on the provided bean/utensil assets)
- 6D pose estimation for grasping targets (coffee bean bowl, spoon, plate, cup)
- Optional use of MuJoCo ground-truth poses as fallback (reduced scoring weight)

### 2.2 Task Planning Layer (LLM-driven)
- Natural-language instruction parsing + **hierarchical task decomposition** into subtask
  sequences (e.g. `set_table → pick_spoon → scoop_beans → hold_3s → return → pour → clear`)
- LLM selects action primitives from a library and validates feasibility against
  current scene state (seat occupancy, utensil availability)
- Falls back to a hand-coded Finite-State-Machine (FSM) policy when LLM is unavailable

### 2.3 Low-Level Control
- **Action primitives**: pre-recorded keyframe sequences + inverse kinematics (IK) for
  reach-and-place motions; force-limited gripper control for compliant grasping
- **Imitation learning**: collect demonstrations via teleoperation (keyboard / GELLO / VR)
  → train bimanual policies (Diffusion Policy / ACT) for the feeding stage
- **Safety**: per-stage force/torque limits; continuous head-area force monitoring;
  watchdog-triggered abort & recover

### 2.4 Sim-to-Real & Evaluation
- Train in MuJoCo (fast, deterministic), validate in Isaac Sim preview
- Score-driven reward shaping aligned with official evaluation (stage completion, recovery ratio)

---

## 3. Current Implementation Status

| Component | Status |
|---|---|
| Task 3 MuJoCo environment (scene_100 / scene_300) | ✅ Running (assets, deps, scene compilation verified) |
| Perception pipeline design | 📝 Spec ready |
| Teleoperation data collection workflow | 🔧 In progress (keyboard teleop verified) |
| LLM task planner (prompt + FSM fallback) | 📝 Spec ready |
| Bimanual feeding policy (Diffusion Policy / ACT) | 📝 Training plan defined |
| Docker packaging | 🔧 Drafting |

---

## 4. Team Background & Differentiators

- **Strong algorithmic engineering** from quantitative finance (data pipelines, model
  engineering, rigorous evaluation methodology) — directly transferable to perception
  model fine-tuning, reward design, and systematic ablation testing
- **LLM application expertise** (prompt engineering, structured output, tool-calling) —
  core to the LLM-orchestrated planning layer
- Pragmatic, milestone-driven development; prioritize **stage-by-stage scoring** with
  robust fallback behaviors rather than fragile end-to-end policies

---

## 5. Development Roadmap

| Phase | Timeframe | Deliverable |
|---|---|---|
| 1. Environment & data | Wk 1 | MuJoCo env verified; teleop dataset collected (100+ demos) |
| 2. Perception | Wk 1–2 | Detector + pose estimator fine-tuned; grasping success ≥90% |
| 3. Action primitives | Wk 2 | IK-based reach/place/feed primitives for stages ①③④ |
| 4. Feeding policy | Wk 3 | Bimanual imitation policy trained & validated (≥3s hold) |
| 5. Full pipeline | Wk 4 | End-to-end 4-stage run; safety checks; final tuning |

---

*Contact / team email: 1373851641@qq.com*
