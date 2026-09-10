#!/usr/bin/env bash
# Reproduce everything: train both models, run the detector on each, run the tests.
# Total wall time on a laptop CPU (one core used) is around 16 minutes.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

STEPS="${STEPS:-1200}"
N_PAIRS="${N_PAIRS:-300}"
N_ABLATION="${N_ABLATION:-100}"
SEED="${SEED:-0}"

echo "== plant: training control and planted models ($STEPS steps each)"
python3 plant.py --steps "$STEPS" --seed "$SEED" --out outputs

echo "== sniff: planted model"
python3 sniff.py --model planted --n-pairs "$N_PAIRS" --n-ablation "$N_ABLATION" --seed "$SEED" --out outputs

echo "== sniff: control model"
python3 sniff.py --model control --n-pairs "$N_PAIRS" --n-ablation "$N_ABLATION" --seed "$SEED" --out outputs

echo "== tests"
python3 -m pytest tests -q
