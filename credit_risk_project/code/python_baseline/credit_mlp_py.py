"""Deliberately unoptimized pure-Python port of credit_mlp_fast.cpp.
Same architecture (23-16-1, tanh + sigmoid), SGD, lr, init scheme, stratified 80/20 split.
No numpy, nested lists, plain loops. Usage: python3 credit_mlp_py.py data.csv [epochs]
"""
import csv, math, random, sys, time


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class MLP:
    def __init__(self, n_in, n_h, lr=0.01):
        self.n_in, self.n_h, self.lr = n_in, n_h, lr
        rnd = random.Random(42)
        s1 = math.sqrt(2.0 / n_in)
        s2 = math.sqrt(2.0 / n_h)
        self.W1 = [[rnd.gauss(0, s1) for _ in range(n_in)] for _ in range(n_h)]
        self.b1 = [0.0] * n_h
        self.W2 = [rnd.gauss(0, s2) for _ in range(n_h)]
        self.b2 = 0.0

    def forward(self, x):
        h = []
        for j in range(self.n_h):
            s = self.b1[j]
            row = self.W1[j]
            for i in range(self.n_in):
                s += row[i] * x[i]
            h.append(math.tanh(s))
        z = self.b2
        for j in range(self.n_h):
            z += self.W2[j] * h[j]
        return sig(z), h

    def predict(self, x):
        return self.forward(x)[0]

    def train_one(self, x, y):
        p, h = self.forward(x)
        dz = p - y
        for j in range(self.n_h):
            dh = dz * self.W2[j] * (1.0 - h[j] * h[j])
            row = self.W1[j]
            for i in range(self.n_in):
                row[i] -= self.lr * dh * x[i]
            self.b1[j] -= self.lr * dh
            self.W2[j] -= self.lr * (dz * h[j])
        self.b2 -= self.lr * dz


def load_csv(path):
    data = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for c in r:
            if len(c) < 25:
                continue
            data.append(([float(v) for v in c[1:24]], int(c[24])))
    return data


def standardize(train, test):
    d = len(train[0][0])
    n = len(train)
    mu = [0.0] * d
    for x, _ in train:
        for j in range(d):
            mu[j] += x[j]
    mu = [m / n for m in mu]
    sd = [0.0] * d
    for x, _ in train:
        for j in range(d):
            sd[j] += (x[j] - mu[j]) ** 2
    sd = [math.sqrt(v / n) or 1.0 for v in sd]
    for ds in (train, test):
        for k, (x, y) in enumerate(ds):
            ds[k] = ([(x[j] - mu[j]) / sd[j] for j in range(d)], y)


def evaluate(net, data):
    tp = tn = fp = fn = 0
    for x, y in data:
        pred = net.predict(x) >= 0.5
        if pred and y: tp += 1
        elif pred: fp += 1
        elif y: fn += 1
        else: tn += 1
    return tp, tn, fp, fn


def main():
    path = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    t0 = time.perf_counter()
    data = load_csv(path)
    zero = [s for s in data if s[1] == 0]
    one = [s for s in data if s[1] == 1]
    rng = random.Random(42)
    rng.shuffle(zero); rng.shuffle(one)
    zc, oc = len(zero) * 4 // 5, len(one) * 4 // 5
    train = zero[:zc] + one[:oc]
    test = zero[zc:] + one[oc:]
    standardize(train, test)
    rng.shuffle(train)
    t_load = time.perf_counter() - t0

    net = MLP(23, 16, 0.01)
    t1 = time.perf_counter()
    for e in range(1, epochs + 1):
        rng.shuffle(train)
        te = time.perf_counter()
        for x, y in train:
            net.train_one(x, y)
        print(f"epoch {e:3d} took {time.perf_counter() - te:.2f}s", flush=True)
    t_train = time.perf_counter() - t1

    t2 = time.perf_counter()
    tp, tn, fp, fn = evaluate(net, test)
    t_eval = time.perf_counter() - t2

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"PY_RESULT epochs={epochs} load_s={t_load:.3f} train_s={t_train:.3f} "
          f"train_per_epoch_s={t_train / epochs:.3f} eval_s={t_eval:.4f} "
          f"infer_us_per_sample={t_eval / len(test) * 1e6:.2f} "
          f"acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")


if __name__ == "__main__":
    main()
