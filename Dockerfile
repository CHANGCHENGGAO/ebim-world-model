FROM ros:jazzy-ros-base

LABEL maintainer="EBiM Task 3 Team"
LABEL description="Phase II B2 policy-only container; connects to external ROS2 sim/robot topics"

ARG INSTALL_INTERNAL_VISION=1
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    ROBOT_MODE=real \
    WORKFLOW=onsite_bowl_cup \
    POLICY_MODE=closed_loop \
    SENSOR_TIMEOUT=30 \
    SENSOR_MAX_AGE=1.0 \
    DETECTION_MAX_AGE=3.0 \
    FINE_DETECTION_MAX_AGE=0.8 \
    VISION_REACQUIRE_TIMEOUT=12.0 \
    VISION_HEARTBEAT_MAX_AGE=3.0 \
    VISION_PERIOD_SECONDS=0.5 \
    VISION_BACKEND=external \
    VISION_MODEL_PATH=/workspace/submission/models/ebim_task3.pt \
    ROBOT_IO_CONFIG=/workspace/submission/config/robot_io.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-sensor-msgs \
      ros-jazzy-std-msgs \
      ros-jazzy-geometry-msgs \
      ros-jazzy-nav-msgs \
      ros-jazzy-tf2-ros \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/submission

COPY requirements.txt requirements-vision.txt ./
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_INTERNAL_VISION" = "1" ]; then \
      pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" && \
      pip3 install --break-system-packages --no-cache-dir -r requirements-vision.txt; \
    fi

COPY autonomy_guard.py b2_contract.py b2_evaluator.py task3_autonomous.py vision_callback.py yolo_detector.py navigation_safety.py table_leg_navigation.py ./
COPY config ./config
COPY models ./models
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh autonomy_guard.py b2_contract.py task3_autonomous.py vision_callback.py yolo_detector.py navigation_safety.py table_leg_navigation.py

ENTRYPOINT ["/entrypoint.sh"]
