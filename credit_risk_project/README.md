# High-Performance Credit Risk Assessment
SIMD-Accelerated C++ Multi-Layer Perceptron for Low-Latency Default Prediction
Arya H. Raval, 25EC415 | 204ES: Object Oriented Programming

## Folder map
| Folder | Contents |
| --- | --- |
| `code/main/` | **Final code**: `credit_mlp_fast.cpp`, the single-file C++ MLP (23-16-1, 401 parameters). This is the file explained line by line in the guide |
| `code/reproducibility/` | Scripts and the saved shared state (`shared42/`) that reproduce every number in the deck: C++ float and double builds, pure Python, NumPy, scikit-learn, XGBoost, figures |
| `code/python_baseline/` | Pure-Python version and the benchmark scripts |
| `data/` | `UCI_Credit_Card.csv`, the UCI Default of Credit Card Clients dataset (30,000 rows, 23 features + label) |
| `results/` | Test-set probabilities, per-epoch curves and the library benchmark table |
| `presentation/` | Final slide deck (PDF) and its LaTeX source with figures and logos |

## Run the final code
```
g++ -O3 -march=native -o credit_mlp_fast code/main/credit_mlp_fast.cpp
./credit_mlp_fast data/UCI_Credit_Card.csv
```
The program reads the CSV, does a stratified 80/20 split (seed 42), standardizes with training statistics, trains for 80 epochs with per-sample SGD (learning rate 0.01) and prints accuracy, precision, recall, F1 and the confusion matrix.

**Expected output of this standalone file: about 80.8% accuracy** (confusion matrix 4388 / 285 / 867 / 461 on this build). It draws its own split, initial weights and shuffles, so it lands slightly below the deck figure.

## Get the deck's numbers (81.95%)
The deck's results come from `credit_mlp_repro.cpp`: the identical `FastMLP` class, but the split, initial weights and shuffle orders are read from the saved `shared42/` folder so that C++, Python, NumPy and the libraries all start from the same state.
```
cd code/reproducibility
g++ -O3 -march=native -std=c++17 -o repro_float credit_mlp_repro.cpp
./repro_float shared42
```
Expected: accuracy 0.8195, precision 0.6848, recall 0.3419, F1 0.4561, confusion matrix 4464 / 209 / 874 / 454.

## Reproduce the full benchmark
```
cd code/reproducibility
./run_all.sh ../../data/UCI_Credit_Card.csv
```
Needs Python 3 with numpy, pandas, scikit-learn, xgboost and matplotlib, plus g++. Timings vary by machine; see `README.txt` there.

## Build the slides
```
cd presentation/source
pdflatex credit_risk_beamer.tex   # run twice
```

## Notes
- The line-by-line explanation of the code is kept as a separate document (not in this archive).
- Dataset: Yeh and Lien, UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/350).
