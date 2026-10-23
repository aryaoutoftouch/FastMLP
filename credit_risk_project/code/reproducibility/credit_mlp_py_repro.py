"""Deliberately unoptimized pure-Python port (no numpy, plain loops), reading the SAME shared
state as credit_mlp_repro.cpp. usage: python3 credit_mlp_py_repro.py shared42 [epochs]"""
import math, sys, time
from array import array

d = sys.argv[1] if len(sys.argv) > 1 else "shared42"
ntr, nte, nin, nh, EP = map(int, open(f"{d}/meta.txt").read().split())
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EP

def rd(name, typ, n):
    a = array(typ); a.fromfile(open(f"{d}/{name}", "rb"), n); return a
Xtr, ytr = rd("Xtr.f32", "f", ntr * nin), rd("ytr.i32", "i", ntr)
Xte, yte = rd("Xte.f32", "f", nte * nin), rd("yte.i32", "i", nte)
perm = rd("perm.i32", "i", EP * ntr)
train_rows = [[Xtr[i * nin + j] for j in range(nin)] for i in range(ntr)]
test = [[Xte[i * nin + j] for j in range(nin)] for i in range(nte)]
w1f = rd("W1.f32", "f", nh * nin); W2 = list(rd("W2.f32", "f", nh))
W1 = [[w1f[j * nin + i] for i in range(nin)] for j in range(nh)]

def forward(x, W1, b1, W2, b2):
    h = []
    for j in range(nh):
        s = b1[j]; row = W1[j]
        for i in range(nin):
            s += row[i] * x[i]
        h.append(math.tanh(s))
    z = b2
    for j in range(nh):
        z += W2[j] * h[j]
    return 1.0 / (1.0 + math.exp(-z)), h

def train(W1, W2, epochs, lr=0.01):
    b1, b2 = [0.0] * nh, 0.0
    for e in range(epochs):
        for k in range(ntr):
            idx = perm[e * ntr + k]; x = trainX[idx]; y = ytr[idx]
            p, h = forward(x, W1, b1, W2, b2); dz = p - y
            for j in range(nh):
                dh = dz * W2[j] * (1.0 - h[j] * h[j]); row = W1[j]
                for i in range(nin):
                    row[i] -= lr * dh * x[i]
                b1[j] -= lr * dh
                W2[j] -= lr * (dz * h[j])
            b2 -= lr * dz
    return b1, b2

def main():
    global trainX
    trainX = train_rows
    t0 = time.perf_counter()
    b1, b2 = train(W1, W2, epochs)
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter(); tp = tn = fp = fn = 0
    for x, y in zip(test, yte):
        pr = forward(x, W1, b1, W2, b2)[0] >= 0.5
        if pr and y: tp += 1
        elif pr: fp += 1
        elif y: fn += 1
        else: tn += 1
    t_inf = time.perf_counter() - t1
    acc = (tp + tn) / nte; pre = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0
    print(f"RESULT impl=python epochs={epochs} train_s={t_train:.4f} infer_us={t_inf / nte * 1e6:.3f} "
          f"acc={acc:.4f} prec={pre:.4f} rec={rec:.4f} f1={f1:.4f} tn={tn} fp={fp} fn={fn} tp={tp}")

main()
