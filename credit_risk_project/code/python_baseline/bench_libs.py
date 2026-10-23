"""Library-based versions of the credit MLP, timed against the C++ / pure-Python results.
Same data, stratified 80/20 split (seed 42), standardized with train statistics.
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1])
X = df.iloc[:, 1:24].values.astype(np.float32)
y = df.iloc[:, 24].values.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)

rows = []


def single_latency(predict_one, n=2000):
    t = time.perf_counter()
    for i in range(n):
        predict_one(Xte[i:i + 1])
    return (time.perf_counter() - t) / n * 1e6


def report(name, train_s, epochs, proba_fn, one_fn, note=""):
    t = time.perf_counter()
    p = proba_fn(Xte)
    batch_s = time.perf_counter() - t
    pred = (p >= 0.5).astype(int)
    rows.append(dict(model=name, train_s=train_s, epochs=epochs,
                     s_per_epoch=train_s / epochs if epochs else float("nan"),
                     single_us=single_latency(one_fn),
                     batch_us=batch_s / len(Xte) * 1e6,
                     acc=accuracy_score(yte, pred), prec=precision_score(yte, pred),
                     rec=recall_score(yte, pred), f1=f1_score(yte, pred),
                     auc=roc_auc_score(yte, p), note=note))
    print(rows[-1], flush=True)


# ---- 1. NumPy, identical per-sample SGD to the C++ code (23-16-1, tanh, sigmoid) ----
class NumpyMLP:
    def __init__(self, n_in=23, n_h=16, lr=0.01):
        r = np.random.default_rng(42)
        self.W1 = (r.standard_normal((n_h, n_in)) * np.sqrt(2 / n_in)).astype(np.float32)
        self.b1 = np.zeros(n_h, np.float32)
        self.W2 = (r.standard_normal(n_h) * np.sqrt(2 / n_h)).astype(np.float32)
        self.b2 = np.float32(0)
        self.lr = np.float32(lr)

    def proba(self, X):
        h = np.tanh(X @ self.W1.T + self.b1)
        return 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

    def step_one(self, x, y):
        h = np.tanh(self.W1 @ x + self.b1)
        p = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))
        dz = p - y
        dh = dz * self.W2 * (1 - h * h)
        self.W1 -= self.lr * np.outer(dh, x)
        self.b1 -= self.lr * dh
        self.W2 -= self.lr * dz * h
        self.b2 -= self.lr * dz

    def step_batch(self, X, y):
        h = np.tanh(X @ self.W1.T + self.b1)
        p = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))
        dz = (p - y) / len(y)
        dh = dz[:, None] * self.W2 * (1 - h * h)
        self.W1 -= self.lr * dh.T @ X
        self.b1 -= self.lr * dh.sum(0)
        self.W2 -= self.lr * h.T @ dz
        self.b2 -= self.lr * dz.sum()


EP = 80
net = NumpyMLP()
rng = np.random.default_rng(0)
t = time.perf_counter()
for e in range(EP):
    for i in rng.permutation(len(Xtr)):
        net.step_one(Xtr[i], ytr[i])
report("NumPy per-sample SGD (same algo as C++)", time.perf_counter() - t, EP,
       net.proba, lambda x: net.proba(x), "80 epochs, batch=1")

net = NumpyMLP(lr=0.1)
t = time.perf_counter()
for e in range(EP):
    idx = rng.permutation(len(Xtr))
    for s in range(0, len(idx), 32):
        b = idx[s:s + 32]
        net.step_batch(Xtr[b], ytr[b])
report("NumPy mini-batch SGD (batch 32, lr 0.1)", time.perf_counter() - t, EP,
       net.proba, lambda x: net.proba(x), "different algorithm: vectorized batches")

# ---- 2. scikit-learn ----
m = MLPClassifier((16,), activation="tanh", solver="sgd", batch_size=1, learning_rate_init=0.01,
                  momentum=0.0, nesterovs_momentum=False, alpha=0.0, max_iter=5, tol=0,
                  n_iter_no_change=100, shuffle=True, random_state=42)
t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
report("sklearn MLP, SGD batch=1 (5 epochs only)", ts, 5,
       lambda X: m.predict_proba(X)[:, 1], lambda x: m.predict_proba(x), "closest to C++ algorithm; only 5 epochs, so accuracy not comparable")

m = MLPClassifier((16,), activation="tanh", solver="adam", max_iter=80, tol=0, n_iter_no_change=100,
                  random_state=42)
t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
report("sklearn MLP, Adam batch=200 (80 epochs)", ts, 80,
       lambda X: m.predict_proba(X)[:, 1], lambda x: m.predict_proba(x), "typical library usage")

# ---- 3. XGBoost (strong tabular baseline) ----
c = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, n_jobs=2, random_state=42)
t = time.perf_counter(); c.fit(Xtr, ytr); ts = time.perf_counter() - t
report("XGBoost (200 trees, depth 4)", ts, 0,
       lambda X: c.predict_proba(X)[:, 1], lambda x: c.predict_proba(x), "gradient boosting baseline")

pd.DataFrame(rows).to_csv("/home/claude/bench_libs.csv", index=False)
