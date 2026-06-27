#!/usr/bin/env bash
# run_research.sh — autonomous strategy research loop
# Usage: bash run_research.sh [N_BATCHES]
# Each batch = 3 experiments via Claude Code. Default: 10 batches = 30 experiments.

set -e

N=${1:-10}
echo "Starting $N research batches (3 experiments each = $((N * 3)) total)"
echo "Log: run.log  |  Experiments: results.tsv  |  Progress: git log"
echo ""

for i in $(seq 1 $N); do
    echo "══════════════════════════════════════════"
    echo "  Batch $i / $N"
    echo "══════════════════════════════════════════"
    claude --print --dangerously-skip-permissions "$(cat research_step.md)"
    echo ""
done

echo "Research complete. $((N * 3)) experiments run."
echo "Review: cat results.tsv | sort -t$'\t' -k2 -rn | head -10"
