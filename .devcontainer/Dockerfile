FROM mcr.microsoft.com/devcontainers/python:3.12

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set uv environment variables
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12

# Install system dependencies and Node.js as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set up npm global directory for vscode user and install Claude Code
USER vscode
ENV NPM_CONFIG_PREFIX=/home/vscode/.npm-global \
    PATH=/home/vscode/.npm-global/bin:$PATH
RUN mkdir -p /home/vscode/.npm-global \
    && npm install -g @anthropic-ai/claude-code

# Set working directory
WORKDIR /workspaces/llm-protein-experiments
