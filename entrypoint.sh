#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ROS_HOME=/tmp/isaac_ros_home

# Optional LLM / Diffusion policy config
# Set these if using --policy llm/diffusion/hybrid
export LLM_MODEL_PATH=${LLM_MODEL_PATH:-/root/models/qwen2.5-7b-instruct-q4_k_m.gguf}
export LLM_N_CTX=${LLM_N_CTX:-2048}
export LLM_N_GPU_LAYERS=${LLM_N_GPU_LAYERS:-0}
export DIFFUSION_MODEL_PATH=${DIFFUSION_MODEL_PATH:-}
export POLICY_TIMEOUT=${POLICY_TIMEOUT:-5.0}

# Isaac Sim uses Python 3.11; ROS2 Jazzy ships rclpy for 3.12.
# Use the internal rclpy bundled with the isaacsim.ros2.bridge extension.
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy:${PYTHONPATH}

cd /workspace/benchmark/task3_isaacsim/scripts

echo "Starting EBiM Task 3 environment..."
echo "Arguments: $@"

# Start Isaac Sim Task 3 scene (dynamic beans enabled by default)
/isaac-sim/python.sh scene_room.py \
    --gripper "${GRIPPER:-robotiq}" \
    --robot-usd /workspace/benchmark/task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd \
    --room-usd /workspace/benchmark/assets/robot_room.usd \
    --head-placement "${HEAD_PLACEMENT:-A}" \
    --headless \
    --franka-root /workspace/benchmark/task1_isaacsim \
    --physics-hz 120 \
    --render-hz 30 &
ISAAC_PID=$!

echo "Isaac Sim PID: $ISAAC_PID"
echo "Waiting for Isaac Sim to start..."
sleep 30

# Start ROS Joint Republisher
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=/workspace/benchmark/task1_isaacsim/scripts:${PYTHONPATH}
python3 /workspace/benchmark/task1_isaacsim/scripts/controllers/ros_joint_republisher.py \
    --bridge-prefix /bridge \
    --isaac-prefix /isaac \
    --gripper-open-position 0.0 \
    --gripper-closed-position 0.8 &
REPUBLISHER_PID=$!

sleep 2

# Start Browser Controller
export PYTHONPATH=/workspace/benchmark/task1_isaacsim/scripts:/workspace/benchmark/task1_isaacsim/services/browser_controller:${PYTHONPATH}
cd /workspace/benchmark/task1_isaacsim/services/browser_controller
python3 app.py --host 0.0.0.0 --port 8090 --publish-rate 60.0 &
BROWSER_PID=$!

echo ""
echo "============================================"
echo "  EBiM Task 3 Environment Ready"
echo "============================================"
echo "  Browser Controller: http://localhost:8090"
echo "  Isaac Sim PID: $ISAAC_PID"
echo "  Republisher PID: $REPUBLISHER_PID"
echo "  Browser PID: $BROWSER_PID"
echo "============================================"
echo ""

wait $ISAAC_PID
