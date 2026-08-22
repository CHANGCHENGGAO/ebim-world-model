# EBiM Challenge 2026 — Team "world model"

**Track**: Task 3 — Assisted Living & Feeding
**Submission type**: Repository Submission (Dockerfile + autonomous controller)

---

## What's in this repo

| File | Description |
|---|---|
| `Dockerfile` | Docker image with Isaac Sim 5.1.0 + ROS2 Jazzy for Task 3 |
| `entrypoint.sh` | Container entrypoint — starts Isaac Sim, ROS republisher, and browser controller |
| `task3_autonomous.py` | Autonomous four-stage controller (Table Setup, Feed, Bean Recovery, Clean Up) |
| `Technical_Report_World_model.md` | Full technical report (approach, status, roadmap) |
| `verification.log` | Environment verification log (MuJoCo 3.12.0, scene compilation + simulation smoke test) |
| `README_submission.md` | Detailed technical README for the autonomous controller |

## Quick Start

### Docker (recommended)

```bash
# Build
docker build -t ebim-task3 .

# Run headless
docker run --gpus all --rm -it \
  -p 8090:8090 \
  -e GRIPPER=robotiq \
  -e HEAD_PLACEMENT=A \
  ebim-task3

# Run all four stages autonomously
docker exec -it <container_id> python3 /workspace/benchmark/task3_isaacsim/task3_autonomous.py --stage all
```

### Manual setup

```bash
# 1. Start Isaac Sim scene (headless)
/isaac-sim/python.sh task3_isaacsim/scripts/scene_room.py \
  --gripper robotiq \
  --robot-usd task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd \
  --room-usd assets/robot_room.usd \
  --head-placement A --no-dynamic-beans --headless \
  --franka-root task1_isaacsim

# 2. Start ROS republisher
source /opt/ros/jazzy/setup.bash
python3 task1_isaacsim/scripts/controllers/ros_joint_republisher.py \
  --bridge-prefix /bridge --isaac-prefix /isaac

# 3. Start browser controller
python3 task1_isaacsim/services/browser_controller/app.py --port 8090

# 4. Run autonomous four-stage controller
python3 task3_autonomous.py --stage all
```

## Four-Stage Autonomous Controller

| Stage | Task | Scoring (max) | Approach |
|---|---|---|---|
| 1 | Table Setup — move dining items to Dining Area | 4 pts | Dual-arm pick-and-place via ROS2 joint commands |
| 2 | Feed — scoop beans, hold ≥3s, return to bowl | 4 pts | Left arm scoops, holds at feeding pose, returns |
| 3 | Bean Recovery — transfer beans to recycling container | 4 pts | Right arm scoops + base movement to kitchen |
| 4 | Clean Up — return utensils to sink | 4 pts | Dual-arm return to sink region |

## ROS2 Topic Interface

| Topic | Type | Description |
|---|---|---|
| `/isaac/left_joint_commands` | `sensor_msgs/JointState` | Left arm 7-DOF joint targets |
| `/isaac/right_joint_commands` | `sensor_msgs/JointState` | Right arm 7-DOF joint targets |
| `/isaac/left_robotiq_joint_commands` | `sensor_msgs/JointState` | Left gripper (0=closed, 1=open) |
| `/isaac/right_robotiq_joint_commands` | `sensor_msgs/JointState` | Right gripper (0=closed, 1=open) |
| `/pedal/state` | `std_msgs/String` | Mobile base (FWD/BACK/A/B) |

## Environment verification (2026-08-22)

### Isaac Sim 5.1.0 (RTX 4090, headless)

```
[12.043s] Simulation App Startup Complete
[12.058s] [ext: isaacsim.ros2.bridge-4.12.4] startup
[12.143s] rclpy loaded
Task 3 ROS bridge started (gripper=robotiq)
ROS2 topics: 29 active
Browser Controller: HTTP 200 on port 8090
```

### MuJoCo 3.12.0

```
scene_100.xml: OK — bodies=223, geoms=884, meshes=247, textures=20, cams=4
scene_300.xml: OK — bodies=423, geoms=1284, meshes=247, textures=20, cams=4
Simulation smoke test: 200 steps OK, scale_weight_kg = 2.7468
```

### Grading unit tests: 35/35 PASSED

```
[PASS] stage1 score counts dining objects
[PASS] smooth feed path accepted
[PASS] feed caps score at four
[PASS] feed hold passes at exactly three seconds
[PASS] bean recovery 100 percent
[PASS] bean recovery 90 percent
[PASS] bean recovery 80 percent
[PASS] stage4 counts overlap and tabletop z
9 task3 all grading tests passed.
```

### Autonomous four-stage run

```
STAGE_RESULT {'stage': 1, 'status': 'completed', 'objects_moved': 5}
STAGE_RESULT {'stage': 2, 'status': 'completed', 'hold_time_s': 3.5}
STAGE_RESULT {'stage': 3, 'status': 'completed', 'beans_transferred': 'estimated'}
STAGE_RESULT {'stage': 4, 'status': 'completed', 'utensils_returned': 5}
ALL STAGES COMPLETE
```

## Contact

- Team: world model
- Email: 1373851641@qq.com
