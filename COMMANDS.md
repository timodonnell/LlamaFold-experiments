#!/bin/bash

set -e
set -x

docker build -t llm-protein-experiments -f .devcontainer/Dockerfile .
docker run -it --gpus all -v /home/bizon/git/llm-protein-experiments:/workspaces/llm-protein-experiments -v ~/.claude:/root/.claude llm-protein-experiments bash
