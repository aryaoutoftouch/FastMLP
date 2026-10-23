"""Writes ONE shared experiment state (split, standardized features, initial weights,
per-epoch shuffle orders) so C++, Python, NumPy etc. all start from identical data.
usage: python3 gen_shared.py data.csv outdir seed"""
import sys, os, numpy as np, pandas as pd
csv, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
EP, NIN, NH = 80, 23, 16
os.makedirs(out, exist_ok=True)
df = pd.read_csv(csv)
X = df.iloc[:, 1:24].values.astype(np.float64)
y = df.iloc[:, 24].values.astype(np.int32)
rng = np.random.default_rng(seed)
zero, one = np.where(y == 0)[0], np.where(y == 1)[0]
rng.shuffle(zero); rng.shuffle(one)
zc, oc = len(zero) * 4 // 5, len(one) * 4 // 5
tr = np.concatenate([zero[:zc], one[:oc]]); te = np.concatenate([zero[zc:], one[oc:]])
rng.shuffle(tr)
mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-12] = 1.0     # training statistics only
Z = ((X - mu) / sd).astype(np.float32)
Z[tr].tofile(f"{out}/Xtr.f32"); y[tr].tofile(f"{out}/ytr.i32")
Z[te].tofile(f"{out}/Xte.f32"); y[te].tofile(f"{out}/yte.i32")
(rng.standard_normal((NH, NIN)) * np.sqrt(2 / NIN)).astype(np.float32).tofile(f"{out}/W1.f32")
(rng.standard_normal(NH) * np.sqrt(2 / NH)).astype(np.float32).tofile(f"{out}/W2.f32")
np.stack([rng.permutation(len(tr)) for _ in range(EP)]).astype(np.int32).tofile(f"{out}/perm.i32")
open(f"{out}/meta.txt", "w").write(f"{len(tr)} {len(te)} {NIN} {NH} {EP}\n")
print(f"seed {seed}: train {len(tr)} test {len(te)}  test default rate {y[te].mean():.4f}")
