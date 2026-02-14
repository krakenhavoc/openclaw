FROM node:22-bookworm

# Install Bun from official image (pinned, no curl|bash)
COPY --from=oven/bun:1.2.4-slim /usr/local/bin/bun /usr/local/bin/bun

# System dependencies for native modules (sharp, node-pty, sqlite-vec)
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  python3 \
  git \
  curl \
  ca-certificates \
  sudo \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Let the 'node' user use sudo without a password
RUN echo "node ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/node \
  && chmod 0440 /etc/sudoers.d/node

# Enable corepack for pnpm
RUN corepack enable

USER node
WORKDIR /workspace
