# Dev image for the guidance workspace: ROS 2 Jazzy + Nav2 + robot_localization + OpenCV.
FROM ros:jazzy

ARG USERNAME=ubuntu
ARG RUFF_VERSION=0.8.4

ENV DEBIAN_FRONTEND=noninteractive

# Retries: a dropped connection resumes the download instead of failing the build.
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-robot-localization \
        ros-jazzy-cv-bridge \
        ros-jazzy-image-publisher \
        python3-opencv \
        python3-pip \
        python3-pytest \
        git \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system Python as externally managed; ruff is a standalone binary.
RUN pip3 install --no-cache-dir --break-system-packages "ruff==${RUFF_VERSION}"

# ubuntu:24.04 already ships the `ubuntu` user (uid/gid 1000), so use it as the dev user.
RUN echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && echo 'source /opt/ros/jazzy/setup.bash' >> /home/${USERNAME}/.bashrc \
    && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /home/${USERNAME}/.bashrc \
    && mkdir -p /ws && chown ${USERNAME}:${USERNAME} /ws

USER ${USERNAME}
WORKDIR /ws
