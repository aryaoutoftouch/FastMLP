"""Extra figures from REAL outputs (C++ dump mode + sklearn/XGBoost on the same shared state).
usage: python3 make_figs2.py data.csv outdir"""
import sys, os, json, subprocess, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import xgboost as xgb
warnings.filterwarnings("ignore")

CSV, OUT = sys.argv[1], sys.argv[2]
NAVY, RED, GREEN, GREY, INK = "#0F2040", "#DC3232", "#228B22", "#9AA3B2", "#1B2538"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

def load_state(d):
    ntr, nte, nin, nh, EP = map(int, open(f"{d}/meta.txt").read().split())
    Xtr = np.fromfile(f"{d}/Xtr.f32", np.float32).reshape(ntr, nin); ytr = np.fromfile(f"{d}/ytr.i32", np.int32)
    Xte = np.fromfile(f"{d}/Xte.f32", np.float32).reshape(nte, nin); yte = np.fromfile(f"{d}/yte.i32", np.int32)
    return Xtr, ytr, Xte, yte

def lib_probs(Xtr, ytr, Xte):
    m = MLPClassifier((16,), activation="tanh", solver="adam", max_iter=80, tol=0, n_iter_no_change=100, random_state=42)
    m.fit(Xtr, ytr)
    c = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          n_jobs=2, random_state=42).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], c.predict_proba(Xte)[:, 1]

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig)

# ======================= seed 42 =======================
Xtr, ytr, Xte, yte = load_state("shared42")
pr = pd.read_csv("dump42/probs.csv"); assert (pr.y.values == yte).all()
p_cpp = pr.p.values
p_skl, p_xgb = lib_probs(Xtr, ytr, Xte)
auc = {"C++ MLP": roc_auc_score(yte, p_cpp), "sklearn MLP": roc_auc_score(yte, p_skl), "XGBoost": roc_auc_score(yte, p_xgb)}
print("AUC seed 42:", auc)

# ---- 1. training curves (C++) ----
ep = pd.read_csv("dump42/epochs.csv")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 3.9), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
a1.plot(ep.epoch, ep.train_loss, color=NAVY, lw=2); a1.plot(ep.epoch, ep.test_loss, color=RED, lw=2, ls="--")
a1.text(81, ep.train_loss.iloc[-1], "train", color=INK, va="center", fontsize=11)
a1.text(81, ep.test_loss.iloc[-1] + 0.0025, "test", color=INK, va="center", fontsize=11)
a1.set_ylabel("Log loss"); a1.set_xlim(0, 92)
a2.plot(ep.epoch, ep.train_acc * 100, color=NAVY, lw=2); a2.plot(ep.epoch, ep.test_acc * 100, color=RED, lw=2, ls="--")
a2.text(81, ep.train_acc.iloc[-1] * 100, "train", color=INK, va="center", fontsize=11)
a2.text(81, ep.test_acc.iloc[-1] * 100 - 0.15, "test", color=INK, va="center", fontsize=11)
a2.set_ylabel("Accuracy (%)"); a2.set_xlabel("Epoch")
for a in (a1, a2):
    a.grid(axis="y", color="#E3E6EC", lw=0.8); a.set_axisbelow(True)
save(fig, "fig_curves")

# ---- 2. ROC ----
fig, ax = plt.subplots(figsize=(4.6, 3.9))
for name, p, col, ls in [("XGBoost", p_xgb, GREEN, ":"), ("sklearn MLP", p_skl, NAVY, "--"), ("C++ MLP", p_cpp, RED, "-")]:
    fpr, tpr, _ = roc_curve(yte, p)
    ax.plot(fpr, tpr, color=col, lw=2.2, ls=ls, label=f"{name}  AUC {auc[name]:.3f}")
ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="-")
ax.text(0.55, 0.47, "chance", color=GREY, fontsize=10, rotation=33)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
ax.grid(color="#E3E6EC", lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
save(fig, "fig_roc")

# ---- 3. threshold sweep (C++) ----
th = np.linspace(0.05, 0.95, 91)
P, R, F, A = [], [], [], []
for t in th:
    pred = p_cpp >= t; tp = (pred & (yte == 1)).sum(); fp = (pred & (yte == 0)).sum(); fn = ((~pred) & (yte == 1)).sum()
    pre = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn)
    P.append(pre); R.append(rec); F.append(2 * pre * rec / (pre + rec) if pre == pre and pre + rec else np.nan); A.append((pred == yte).mean())
F = np.array(F); tbest = th[np.nanargmax(F)]
print(f"best-F1 threshold {tbest:.2f} F1 {np.nanmax(F):.4f}; recall there {R[np.nanargmax(F)]:.3f}; acc there {A[np.nanargmax(F)]:.4f}")
fig, ax = plt.subplots(figsize=(4.6, 3.9))
for y, col, ls, lab in [(A, NAVY, "-", "Accuracy"), (P, RED, "--", "Precision"), (R, GREEN, "-.", "Recall"), (F, INK, ":", "F1")]:
    ax.plot(th, y, color=col, lw=2.2, ls=ls, label=lab)
ax.axvline(0.5, color=GREY, lw=1); ax.text(0.505, 0.03, "0.5", color=GREY, fontsize=10)
ax.axvline(tbest, color=INK, lw=0.8, ls=":"); ax.text(tbest + 0.01, 0.03, f"best F1\n{tbest:.2f}", color=INK, fontsize=10)
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric"); ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.0)
ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.12), handlelength=1.6, columnspacing=1.0)
ax.grid(axis="y", color="#E3E6EC", lw=0.8); ax.set_axisbelow(True)
save(fig, "fig_threshold")

# ---- 4. paired seeds 1..10 ----
rows = []
for s in range(1, 11):
    d = f"shared_s{s}"; dd = f"dump_s{s}"; os.makedirs(dd, exist_ok=True)
    subprocess.run(["python3", "gen_shared.py", CSV, d, str(s)], check=True, capture_output=True)
    subprocess.run(["./repro_float", d, "80", dd], check=True, capture_output=True)
    Xtr_, ytr_, Xte_, yte_ = load_state(d)
    pc = pd.read_csv(f"{dd}/probs.csv").p.values
    ps, px = lib_probs(Xtr_, ytr_, Xte_)
    r = dict(seed=s, cpp=accuracy_score(yte_, pc >= .5), skl=accuracy_score(yte_, ps >= .5), xgb=accuracy_score(yte_, px >= .5),
             cpp_auc=roc_auc_score(yte_, pc), skl_auc=roc_auc_score(yte_, ps), xgb_auc=roc_auc_score(yte_, px))
    rows.append(r); print(r, flush=True)
    subprocess.run(["rm", "-rf", d, dd])
df = pd.DataFrame(rows); df.to_csv("seeds_paired.csv", index=False)
summ = {k: (df[k].mean(), df[k].std(ddof=1)) for k in df.columns if k != "seed"}
print({k: f"{m:.4f} +/- {s:.4f}" for k, (m, s) in summ.items()})
gap_s = df.skl - df.cpp; gap_x = df.xgb - df.cpp
print(f"paired acc gap sklearn-C++: mean {gap_s.mean()*100:.2f} pt, sd {gap_s.std(ddof=1)*100:.2f}, wins {(gap_s>0).sum()}/10")
print(f"paired acc gap XGB-C++:     mean {gap_x.mean()*100:.2f} pt, sd {gap_x.std(ddof=1)*100:.2f}, wins {(gap_x>0).sum()}/10")
print(f"paired AUC gap XGB-C++: mean {(df.xgb_auc-df.cpp_auc).mean():.4f}; sklearn-C++: mean {(df.skl_auc-df.cpp_auc).mean():.4f}")

fig, ax = plt.subplots(figsize=(4.6, 3.9))
for k, col, mk, ls, lab in [("xgb", GREEN, "^", ":", "XGBoost"), ("skl", NAVY, "s", "--", "sklearn MLP"), ("cpp", RED, "o", "-", "C++ MLP")]:
    ax.plot(df.seed, df[k] * 100, color=col, marker=mk, ms=6, lw=1.6, ls=ls,
            label=f"{lab}  {df[k].mean()*100:.2f} ± {df[k].std(ddof=1)*100:.2f}")
ax.set_xlabel("Seed (split + init + shuffles)"); ax.set_ylabel("Test accuracy (%)"); ax.set_xticks(range(1, 11))
ax.legend(loc="upper center", frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.2), ncol=1)
ax.grid(axis="y", color="#E3E6EC", lw=0.8); ax.set_axisbelow(True)
save(fig, "fig_seeds")
json.dump({"auc42": auc, "tbest": float(tbest), "summary": {k: list(v) for k, v in summ.items()}}, open("figs2_stats.json", "w"), indent=1)
print("done")
