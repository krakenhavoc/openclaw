FROM node:22-bookworm

COPY --from=oven/bun:1.2.4-slim /usr/local/bin/bun /usr/local/bin/bun

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential python3 git curl ca-certificates sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "node ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/node

RUN corepack enable

USER node
WORKDIR /workspace
