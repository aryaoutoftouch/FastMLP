"""Library versions on the SAME shared state (split, standardized data, init weights, shuffles)."""
import sys, time, warnings, json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
warnings.filterwarnings("ignore")
d = sys.argv[1]
ntr, nte, nin, nh, EP = map(int, open(f"{d}/meta.txt").read().split())
Xtr = np.fromfile(f"{d}/Xtr.f32", np.float32).reshape(ntr, nin); ytr = np.fromfile(f"{d}/ytr.i32", np.int32)
Xte = np.fromfile(f"{d}/Xte.f32", np.float32).reshape(nte, nin); yte = np.fromfile(f"{d}/yte.i32", np.int32)
perm = np.fromfile(f"{d}/perm.i32", np.int32).reshape(EP, ntr)
W1_0 = np.fromfile(f"{d}/W1.f32", np.float32).reshape(nh, nin); W2_0 = np.fromfile(f"{d}/W2.f32", np.float32)
out = []

def single_us(fn, n=2000):
    t = time.perf_counter()
    for i in range(n): fn(Xte[i:i + 1])
    return (time.perf_counter() - t) / n * 1e6

def rep(name, train_s, epochs, proba, one):
    t = time.perf_counter(); p = proba(Xte); b = (time.perf_counter() - t) / nte * 1e6
    pr = (p >= 0.5).astype(int)
    r = dict(model=name, train_s=train_s, epochs=epochs, single_us=single_us(one), batch_us=b,
             acc=accuracy_score(yte, pr), prec=precision_score(yte, pr), rec=recall_score(yte, pr),
             f1=f1_score(yte, pr), auc=roc_auc_score(yte, p))
    out.append(r); print(r, flush=True)

class Np:
    def __init__(s, lr):
        s.W1, s.W2, s.b1, s.b2, s.lr = W1_0.copy(), W2_0.copy(), np.zeros(nh, np.float32), np.float32(0), np.float32(lr)
    def proba(s, X):
        return 1 / (1 + np.exp(-(np.tanh(X @ s.W1.T + s.b1) @ s.W2 + s.b2)))
    def one(s, x, y):
        h = np.tanh(s.W1 @ x + s.b1); p = 1 / (1 + np.exp(-(h @ s.W2 + s.b2))); dz = p - y
        dh = dz * s.W2 * (1 - h * h)
        s.W1 -= s.lr * np.outer(dh, x); s.b1 -= s.lr * dh; s.W2 -= s.lr * dz * h; s.b2 -= s.lr * dz
    def batch(s, X, y):
        h = np.tanh(X @ s.W1.T + s.b1); p = 1 / (1 + np.exp(-(h @ s.W2 + s.b2))); dz = (p - y) / len(y)
        dh = dz[:, None] * s.W2 * (1 - h * h)
        s.W1 -= s.lr * dh.T @ X; s.b1 -= s.lr * dh.sum(0); s.W2 -= s.lr * h.T @ dz; s.b2 -= s.lr * dz.sum()

n = Np(0.01); t = time.perf_counter()
for e in range(EP):
    for i in perm[e]: n.one(Xtr[i], ytr[i])
rep("NumPy per-sample SGD", time.perf_counter() - t, EP, n.proba, n.proba)

n = Np(0.1); t = time.perf_counter()
for e in range(EP):
    for s in range(0, ntr, 32):
        b = perm[e][s:s + 32]; n.batch(Xtr[b], ytr[b])
rep("NumPy mini-batch (32)", time.perf_counter() - t, EP, n.proba, n.proba)

m = MLPClassifier((nh,), activation="tanh", solver="sgd", batch_size=1, learning_rate_init=0.01, momentum=0.0,
                  nesterovs_momentum=False, alpha=0.0, max_iter=5, tol=0, n_iter_no_change=100, random_state=42)
t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
rep("sklearn MLP SGD batch=1 (5 ep)", ts, 5, lambda X: m.predict_proba(X)[:, 1], lambda x: m.predict_proba(x))

m = MLPClassifier((nh,), activation="tanh", solver="adam", max_iter=80, tol=0, n_iter_no_change=100, random_state=42)
t = time.perf_counter(); m.fit(Xtr, ytr); ts = time.perf_counter() - t
rep("sklearn MLP Adam batch=200", ts, 80, lambda X: m.predict_proba(X)[:, 1], lambda x: m.predict_proba(x))

c = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=2, random_state=42)
t = time.perf_counter(); c.fit(Xtr, ytr); ts = time.perf_counter() - t
rep("XGBoost 200 trees", ts, 0, lambda X: c.predict_proba(X)[:, 1], lambda x: c.predict_proba(x))
json.dump(out, open("libs_repro.json", "w"), indent=1)
