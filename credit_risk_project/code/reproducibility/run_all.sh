#!/usr/bin/env bash
# Reproduces every number in the deck. Usage: ./run_all.sh UCI_Credit_Card.csv
set -e
CSV=${1:?path to UCI_Credit_Card.csv}
python3 gen_shared.py "$CSV" shared42 42            # split + init + shuffles, seed 42 (byte-identical every time)
g++ -O3 -march=native -std=c++17 -o repro_float  credit_mlp_repro.cpp
g++ -O3 -march=native -std=c++17 -DREAL=double -o repro_double credit_mlp_repro.cpp
./repro_float  shared42                              # the C++ MLP (float)
./repro_double shared42                              # same maths in double -> matches Python exactly
python3 credit_mlp_py_repro.py shared42 80           # pure Python (~80-130 s)
python3 bench_libs_repro.py shared42                 # NumPy, scikit-learn, XGBoost on the same state
# extra figures (curves, ROC, threshold sweep, paired 10 seeds); needs ./repro_float built above
mkdir -p dump42 && ./repro_float shared42 80 dump42
python3 make_figs2.py "$CSV" figs_out
