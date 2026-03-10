#!/bin/bash
cd ~/sky_workdir
git add scripts/ results/ research_prompt.md -f 2>/dev/null
git add *.py *.md *.json -f 2>/dev/null
git commit -m "${1:-auto: experiment checkpoint}" 2>/dev/null
git push clawrouter mar10-ec2 2>&1
