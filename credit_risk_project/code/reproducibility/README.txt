Reproducibility design
- gen_shared.py draws the stratified split, standardizes with TRAIN statistics, draws the initial weights
  and 80 per-epoch shuffle orders from ONE numpy generator (seed 42) and writes them as binary files.
- C++, pure Python and NumPy then only READ those files, so they start from identical data/weights and apply
  identical updates in identical order. Same binary -> identical output every run.
- C++ built with -DREAL=double and the pure-Python version give the SAME confusion matrix
  (4464/209/879/449 on seed 42). The float build differs by a handful of samples (4464/209/874/454)
  purely from 32- vs 64-bit rounding.
- scikit-learn / XGBoost share the split and data; their internal initialisation uses random_state=42.
- Timings vary run to run (noisy shared machine): report medians.
