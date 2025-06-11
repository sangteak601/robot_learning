ARG ISAACSIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG ISAACLAB_VERSION=2.1.0
ENV SHELL=/bin/bash

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    ncurses-term \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Isaac Lab
RUN git clone -b v${ISAACLAB_VERSION} https://github.com/isaac-sim/IsaacLab.git /isaac-lab
RUN ln -s /isaac-sim /isaac-lab/_isaac_sim
RUN /isaac-lab/isaaclab.sh --install

CMD ["/bin/bash"]