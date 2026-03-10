#!/bin/bash
set -e
cd ~/sky_workdir
export PATH="$HOME/.kiro/bin:$PATH"

# Verify auth
kiro-cli whoami || { echo "Not logged in. Run: kiro-cli login --use-device-flow"; exit 1; }

# Launch kiro-cli agent for automated research
echo "=== Starting kiro-cli research agent ==="
kiro-cli chat --model claude-opus-4.6 --trust-all-tools --no-interactive "$(cat ~/sky_workdir/research_prompt.md)"
