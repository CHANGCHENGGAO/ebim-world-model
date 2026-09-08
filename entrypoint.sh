#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/ros/jazzy/setup.bash

export ROS_HOME="${ROS_HOME:-/tmp/ebim_ros_home}"
export PYTHONPATH="/workspace/submission:${PYTHONPATH:-}"

ROBOT_MODE="${ROBOT_MODE:-real}"
POLICY_MODE="${POLICY_MODE:-closed_loop}"
WORKFLOW="${WORKFLOW:-onsite_bowl_cup}"
AUTONOMOUS_ONLY="${AUTONOMOUS_ONLY:-1}"
SENSOR_TIMEOUT="${SENSOR_TIMEOUT:-30}"
ROBOT_IO_CONFIG="${ROBOT_IO_CONFIG:-/workspace/submission/config/robot_io.json}"
VISION_BACKEND="${VISION_BACKEND:-auto}"
ALLOW_UNCONFIRMED_IO="${ALLOW_UNCONFIRMED_IO:-0}"
VISION_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$VISION_PID" ]] && kill -0 "$VISION_PID" 2>/dev/null; then
        kill -TERM "$VISION_PID" 2>/dev/null || true
        wait "$VISION_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

case "$ROBOT_MODE" in
    sim|real) ;;
    *) echo "FATAL: ROBOT_MODE must be sim or real, got '$ROBOT_MODE'" >&2; exit 64 ;;
esac

case "$WORKFLOW" in
    full_task3|onsite_bowl_cup) ;;
    *) echo "FATAL: WORKFLOW must be full_task3 or onsite_bowl_cup, got '$WORKFLOW'" >&2; exit 64 ;;
esac
if [[ "$WORKFLOW" == "onsite_bowl_cup" && "$ROBOT_MODE" != "real" ]]; then
    echo "FATAL: onsite_bowl_cup is a real-robot calibration workflow" >&2
    exit 64
fi

export AUTONOMOUS_ONLY POLICY_MODE
python3 /workspace/submission/autonomy_guard.py \
    --policy-file /workspace/submission/task3_autonomous.py \
    --launch-file "$0" --json

CONTRACT_ARGS=(--config "$ROBOT_IO_CONFIG" --mode "$ROBOT_MODE")
if [[ "$ALLOW_UNCONFIRMED_IO" == "1" ]]; then
    CONTRACT_ARGS+=(--allow-unconfirmed)
fi

python3 /workspace/submission/b2_contract.py "${CONTRACT_ARGS[@]}" >/tmp/ebim_contract.json

echo "============================================"
echo "  EBiM Task 3 — B2 policy container"
echo "  ROBOT_MODE=$ROBOT_MODE POLICY_MODE=$POLICY_MODE WORKFLOW=$WORKFLOW"
echo "  VISION_BACKEND=$VISION_BACKEND"
echo "  ROBOT_IO_CONFIG=$ROBOT_IO_CONFIG"
echo "============================================"

if [[ "$VISION_BACKEND" == "internal" ]] || \
   { [[ "$VISION_BACKEND" == "auto" ]] && [[ -f "${VISION_MODEL_PATH:-}" ]]; }; then
    if [[ ! -f "${VISION_MODEL_PATH:-}" ]]; then
        echo "FATAL: internal vision requires a validated Task 3 checkpoint at VISION_MODEL_PATH; refusing generic YOLO fallback" >&2
        exit 66
    fi
    echo "[vision] starting internal perception"
    python3 /workspace/submission/vision_callback.py \
        --robot-mode "$ROBOT_MODE" \
        --config "$ROBOT_IO_CONFIG" \
        --model-path "${VISION_MODEL_PATH:-}" &
    VISION_PID=$!
elif [[ "$VISION_BACKEND" == "external" ]] || [[ "$VISION_BACKEND" == "auto" ]]; then
    echo "[vision] waiting for external perception publisher"
else
    echo "FATAL: VISION_BACKEND must be auto, internal or external" >&2
    exit 64
fi

mapfile -t REQUIRED_TOPICS < <(
    python3 /workspace/submission/b2_contract.py \
        "${CONTRACT_ARGS[@]}" --print-required-topics
)

for topic in "${REQUIRED_TOPICS[@]}"; do
    echo "[readiness] waiting for $topic"
    if ! timeout "$SENSOR_TIMEOUT" bash -c \
        'topic="$1"; until ros2 topic list 2>/dev/null | grep -Fxq "$topic"; do sleep 0.5; done' \
        _ "$topic"; then
        echo "FATAL: required topic '$topic' unavailable after ${SENSOR_TIMEOUT}s" >&2
        exit 70
    fi
    if [[ -n "$VISION_PID" ]] && ! kill -0 "$VISION_PID" 2>/dev/null; then
        echo "FATAL: internal vision process exited during readiness" >&2
        wait "$VISION_PID" || true
        exit 71
    fi
done

echo "[controller] all required topics discovered; starting policy"
set +e
python3 /workspace/submission/task3_autonomous.py \
    --stage "${STAGE:-all}" \
    --workflow "$WORKFLOW" \
    --policy "$POLICY_MODE" \
    --robot-mode "$ROBOT_MODE" \
    --config "$ROBOT_IO_CONFIG" \
    --sensor-timeout "$SENSOR_TIMEOUT"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
    echo "FATAL: controller exited with status $status" >&2
fi
exit "$status"
