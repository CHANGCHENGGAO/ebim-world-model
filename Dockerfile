FROM nvcr.io/nvidia/isaac-sim:5.1.0

LABEL maintainer="EBiM Task3 Team"
LABEL description="EBiM Competition Task 3 - Assisted Living & Feeding"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV DISPLAY=:20
ENV QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos \
    -o /tmp/ros2.repos || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-sensor-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-geometry-msgs \
    python3-flask \
    python3-numpy \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    flask \
    flask-socketio \
    numpy \
    scipy \
    matplotlib

WORKDIR /workspace

COPY . /workspace/benchmark/

ENV LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy:/workspace/benchmark/task1_isaacsim/scripts:/workspace/benchmark/task1_isaacsim/services/browser_controller:/workspace/benchmark/scripts/scenes:/workspace/benchmark/scripts/common:/workspace/benchmark/scripts/evaluation/task3:${PYTHONPATH}

RUN chmod +x /workspace/benchmark/task3_isaacsim/scripts/run_isaacsim_teleop.sh \
    /workspace/benchmark/task3_isaacsim/scripts/run_helper_containers.sh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8090

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--gripper", "robotiq", "--headless"]
