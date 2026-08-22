# EBiM Competition Task 3 — Assisted Living & Feeding

## Overview

This repository contains the autonomous controller for **EBiM Competition Task 3: Assisted Living & Feeding**, using a mobile dual-FR3 robot with Robotiq 2F-85 grippers in the shared robot room simulation environment (Isaac Sim 5.1.0 + ROS2 Jazzy).

The solution completes all four stages of Task 3:
1. **Table Setup** — Move dining items (tray, bowl, spoon, plate, cup) from Kitchen Area to Dining Area
2. **Feed** — Scoop coffee beans, hold spoon at feeding pose for ≥3 seconds, return beans to bowl
3. **Bean Recovery** — Transfer beans into the recycling container in the Kitchen Area
4. **Clean Up** — Return utensils to the marked sink region

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Isaac Sim 5.1.0                     │
│  ┌─────────────────────────────────────────────┐   │
│  │   scene_room.py (Task 3 ROS bridge)         │   │
│  │   - Dual FR3 + Robotiq grippers             │   │
│  │   - Mobile base + spine                      │   │
│  │   - robot_room.usd scene                     │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │ ROS2 Topics                   │
│  ┌──────────────────┴──────────────────────────┐   │
│  │  ros_joint_republisher.py                    │   │
│  │  (bridge <-> isaac topic mapping)           │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│  ┌──────────────────┴──────────────────────────┐   │
│  │  task3_autonomous.py (This submission)       │   │
│  │  - Stage 1-4 autonomous control              │   │
│  │  - ROS2 joint command publisher              │   │
│  │  - Gripper control                           │   │
│  │  - Mobile base control                       │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Key Components

### 1. Isaac Sim Scene (`scene_room.py`)
- Loads the robot room with mobile dual-FR3 + Robotiq 2F-85 grippers
- Headless mode with ROS2 bridge enabled
- Configurable head placement (A-I) and bean count (100/300)

### 2. ROS2 Bridge
- Publishes joint states on `/isaac/left_joint_states`, `/isaac/right_joint_states`
- Subscribes to joint commands on `/isaac/left_joint_commands`, `/isaac/right_joint_commands`
- Gripper topics: `/isaac/{left,right}_robotiq_joint_{states,commands}`
- Mobile base: `/pedal/state` (FWD, BACK, A, B, A+C, B+C)

### 3. Autonomous Controller (`task3_autonomous.py`)
- Publishes joint commands at 50 Hz
- Executes four stages sequentially with smooth interpolated motion
- Controls both arms, grippers, and mobile base

### 4. Browser Controller (optional)
- Web UI at port 8090 for manual teleoperation
- Can be used alongside or instead of autonomous control

## Docker Setup

### Prerequisites
- NVIDIA GPU with drivers ≥525.60
- NVIDIA Container Toolkit
- Docker Compose v2

### Build and Run

```bash
# Build the container
docker build -t ebim-task3 .

# Run with GPU support
docker run --gpus all --rm -it \
  -p 8090:8090 \
  -e GRIPPER=robotiq \
  -e HEAD_PLACEMENT=A \
  ebim-task3

# Or run with GUI (requires X11 forwarding)
xhost +local:docker
docker run --gpus all --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 8090:8090 \
  ebim-task3 \
  --gripper robotiq
```

### Manual Setup (without Docker)

1. Install Isaac Sim 5.1.0
2. Install ROS2 Jazzy
3. Clone the EBiM benchmark repository with submodules
4. Download large assets:
   ```bash
   bash task1_isaacsim/scripts/download_large_assets.sh
   bash task3_mujoco/scripts/download_large_assets.sh
   ```

5. Start the environment:
   ```bash
   # Start Isaac Sim scene
   /isaac-sim/python.sh task3_isaacsim/scripts/scene_room.py \
     --gripper robotiq \
     --robot-usd task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd \
     --room-usd assets/robot_room.usd \
     --head-placement A \
     --no-dynamic-beans \
     --headless \
     --franka-root task1_isaacsim

   # Start ROS republisher
   source /opt/ros/jazzy/setup.bash
   python3 task1_isaacsim/scripts/controllers/ros_joint_republisher.py \
     --bridge-prefix /bridge --isaac-prefix /isaac

   # Start browser controller
   python3 task1_isaacsim/services/browser_controller/app.py \
     --host 0.0.0.0 --port 8090
   ```

6. Run autonomous controller:
   ```bash
   source /opt/ros/jazzy/setup.bash
   export PYTHONPATH=task1_isaacsim/scripts:task1_isaacsim/services/browser_controller
   python3 task3_autonomous.py --stage all
   ```

## Usage

### Run all four stages autonomously:
```bash
python3 task3_autonomous.py --stage all
```

### Run individual stages:
```bash
python3 task3_autonomous.py --stage 1  # Table Setup
python3 task3_autonomous.py --stage 2  # Feed
python3 task3_autonomous.py --stage 3  # Bean Recovery
python3 task3_autonomous.py --stage 4  # Clean Up
```

### Manual teleoperation via browser:
Open `http://localhost:8090` in a web browser to control both arms and grippers manually.

### Mobile base control:
```bash
# Forward
ros2 topic pub -r 10 /pedal/state std_msgs/msg/String "{data: FWD}"
# Backward
ros2 topic pub -r 10 /pedal/state std_msgs/msg/String "{data: BACK}"
# Strafe left/right
ros2 topic pub -r 10 /pedal/state std_msgs/msg/String "{data: A}"
ros2 topic pub -r 10 /pedal/state std_msgs/msg/String "{data: B}"
```

## ROS2 Topic Interface

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/isaac/left_joint_states` | `sensor_msgs/JointState` | Sim→Controller | Left arm joint positions |
| `/isaac/right_joint_states` | `sensor_msgs/JointState` | Sim→Controller | Right arm joint positions |
| `/isaac/left_joint_commands` | `sensor_msgs/JointState` | Controller→Sim | Left arm joint targets |
| `/isaac/right_joint_commands` | `sensor_msgs/JointState` | Controller→Sim | Right arm joint targets |
| `/isaac/left_robotiq_joint_states` | `sensor_msgs/JointState` | Sim→Controller | Left gripper state |
| `/isaac/right_robotiq_joint_states` | `sensor_msgs/JointState` | Sim→Controller | Right gripper state |
| `/isaac/left_robotiq_joint_commands` | `sensor_msgs/JointState` | Controller→Sim | Left gripper command (0.0=closed, 1.0=open) |
| `/isaac/right_robotiq_joint_commands` | `sensor_msgs/JointState` | Controller→Sim | Right gripper command |
| `/pedal/state` | `std_msgs/String` | Controller→Sim | Mobile base command (FWD/BACK/A/B/A+C/B+C) |

## Robot Configuration

- **Robot**: Mobile FR3 Duo (dual Franka Emika FR3 arms)
- **Gripper**: Robotiq 2F-85 (0.0 closed, 0.8 fully closed driver range)
- **Base**: Holonomic mobile base with spine lift
- **Sensors**: Head camera, left/right wrist cameras
- **Scene**: robot_room.usd with kitchen area, dining area, and sink

## Scoring

Each stage is worth 4 points (total 16):
- **Stage 1**: 1 point per object moved to dining area (5 objects)
- **Stage 2**: Smooth path + 3s hold + bean retention (max 4)
- **Stage 3**: ≥80% beans in recovery container (4=100%, 3=90%, 2=80%)
- **Stage 4**: 1 point per utensil in sink region (5 objects)

## Testing

### Unit tests (no Isaac Sim required):
```bash
python3 -B scripts/evaluation/task3/tests/test_grading.py all
```

### Integration test (requires Isaac Sim):
```bash
/isaac-sim/python.sh scripts/evaluation/task3/integration_test.py --headless all
```

## License

Apache License 2.0

## Acknowledgments

Built on the EBiM Benchmark framework (https://ebim-benchmark.github.io)
