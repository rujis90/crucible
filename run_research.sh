#!/bin/bash
# run_research.sh — runs the Crucible loop as stateless 3-experiment batches
# Each batch is a fresh Claude invocation — no context accumulation
# Usage: bash run_research.sh [num_batches]

cd "$(dirname "$0")"

BATCHES=${1:-20}   # default 20 batches = 60 experiments

echo "╔══════════════════════════════════════════════╗"
echo "║  CRUCIBLE — Autonomous Strategy Research     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Batches:     $BATCHES × 3 experiments = up to $((BATCHES * 3)) experiments"
echo "Working dir: $(pwd)"
echo "Current best: $(grep 'keep' results.tsv 2>/dev/null | tail -1 || echo 'none yet')"
echo ""

for i in $(seq 1 $BATCHES); do
    echo "════════════════════════════════════════"
    echo "  BATCH $i / $BATCHES  ($(date '+%H:%M:%S'))"
    echo "════════════════════════════════════════"

    unset CLAUDECODE && claude --dangerously-skip-permissions --print "$(cat research_step.md)"

    echo ""
    echo "Batch $i done. Current best: $(grep 'keep' results.tsv 2>/dev/null | tail -1 || echo 'none yet')"
    echo ""
done

echo "All batches complete."
echo ""
echo "Final results:"
cat results.tsv
