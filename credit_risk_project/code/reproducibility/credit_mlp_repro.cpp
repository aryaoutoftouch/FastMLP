// Reproducible build of credit_mlp_fast.cpp: identical FastMLP class, but data split,
// initial weights and per-epoch shuffle orders are READ from a shared directory
// (written by gen_shared.py), so every language starts from the same state.
// Build float (default):  g++ -O3 -march=native -std=c++17 credit_mlp_repro.cpp
// Build double:           add -DREAL=double
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#ifndef REAL
#define REAL float
#endif
using namespace std;
typedef REAL real;

inline real sig(real x) { return 1.0f / (1.0f + exp(-x)); }

struct Sample { vector<real> x; int y; };

class FastMLP {
    int nIn, nH;
    vector<real> W1, b1, W2;
    real b2, lr;
    mutable vector<real> h_buf;
    vector<real> dh_buf, dW2_buf;
public:
    FastMLP(int in, int hidden, real rate) : nIn(in), nH(hidden), W1(hidden * in), b1(hidden, 0),
        W2(hidden), b2(0), lr(rate), h_buf(hidden), dh_buf(hidden), dW2_buf(hidden) {}
    void load(const string& dir) {
        vector<float> a(nH * nIn), b(nH);
        ifstream f1(dir + "/W1.f32", ios::binary), f2(dir + "/W2.f32", ios::binary);
        f1.read((char*)a.data(), a.size() * 4); f2.read((char*)b.data(), b.size() * 4);
        for (size_t i = 0; i < a.size(); ++i) W1[i] = a[i];
        for (int j = 0; j < nH; ++j) W2[j] = b[j];
    }
    real forward(const real* x, vector<real>& h) const {
        for (int j = 0; j < nH; ++j) {
            real sum = b1[j];
            const real* w_row = &W1[j * nIn];
            for (int i = 0; i < nIn; ++i) sum += w_row[i] * x[i];
            h[j] = tanh(sum);
        }
        real z = b2;
        for (int j = 0; j < nH; ++j) z += W2[j] * h[j];
        return sig(z);
    }
    real predict(const vector<real>& x) const { return forward(x.data(), h_buf); }
    void trainOne(const Sample& s) {
        real p = forward(s.x.data(), h_buf);
        real dz = p - s.y;
        for (int j = 0; j < nH; ++j) {
            dW2_buf[j] = dz * h_buf[j];
            dh_buf[j] = dz * W2[j] * (1.0f - h_buf[j] * h_buf[j]);
        }
        for (int j = 0; j < nH; ++j) {
            real dh = dh_buf[j];
            real* w_row = &W1[j * nIn];
            const real* x_ptr = s.x.data();
            for (int i = 0; i < nIn; ++i) w_row[i] -= lr * dh * x_ptr[i];
            b1[j] -= lr * dh;
            W2[j] -= lr * dW2_buf[j];
        }
        b2 -= lr * dz;
    }
};

template <class T> vector<T> readBin(const string& p, size_t n) {
    vector<T> v(n); ifstream f(p, ios::binary);
    if (!f) { cerr << "cannot open " << p << "\n"; exit(1); }
    f.read((char*)v.data(), n * sizeof(T)); return v;
}
vector<Sample> loadSet(const string& dir, const char* X, const char* Y, int n, int d) {
    auto x = readBin<float>(dir + "/" + X, (size_t)n * d); auto y = readBin<int>(dir + "/" + Y, n);
    vector<Sample> s(n);
    for (int i = 0; i < n; ++i) { s[i].y = y[i]; s[i].x.assign(x.begin() + (size_t)i * d, x.begin() + (size_t)(i + 1) * d); }
    return s;
}

int main(int argc, char** argv) {
    string dir = argc > 1 ? argv[1] : "shared42";
    int ntr, nte, nin, nh, EP; { ifstream m(dir + "/meta.txt"); m >> ntr >> nte >> nin >> nh >> EP; }
    int epochs = argc > 2 ? atoi(argv[2]) : EP;
    auto train = loadSet(dir, "Xtr.f32", "ytr.i32", ntr, nin);
    auto test  = loadSet(dir, "Xte.f32", "yte.i32", nte, nin);
    auto perm = readBin<int>(dir + "/perm.i32", (size_t)EP * ntr);

    string dump = argc > 3 ? argv[3] : "";   // optional: directory for curves/probabilities (not used for timing)
    FastMLP net(nin, nh, 0.01f); net.load(dir);
    ofstream ecsv; if (!dump.empty()) { ecsv.open(dump + "/epochs.csv"); ecsv << "epoch,train_loss,test_loss,train_acc,test_acc\n"; }
    auto lossAcc = [&](const vector<Sample>& d, double& loss, double& acc) {
        loss = 0; int ok = 0;
        for (auto& s : d) { double p = net.predict(s.x); p = min(max(p, 1e-7), 1 - 1e-7);
            loss += -(s.y ? log(p) : log(1 - p)); ok += ((p >= 0.5) == (s.y == 1)); }
        loss /= d.size(); acc = double(ok) / d.size(); };
    using clk = chrono::steady_clock;
    auto sec = [](clk::time_point a, clk::time_point b) { return chrono::duration<double>(b - a).count(); };

    auto t1 = clk::now();
    for (int e = 0; e < epochs; ++e)
    {
        for (int i = 0; i < ntr; ++i) net.trainOne(train[perm[(size_t)e * ntr + i]]);
        if (!dump.empty()) { double l1, a1, l2, a2; lossAcc(train, l1, a1); lossAcc(test, l2, a2);
            ecsv << e + 1 << "," << l1 << "," << l2 << "," << a1 << "," << a2 << "\n"; }
    }
    double t_train = sec(t1, clk::now());

    if (!dump.empty()) { ofstream pc(dump + "/probs.csv"); pc.precision(9); pc << "y,p\n";
        for (auto& s : test) pc << s.y << "," << net.predict(s.x) << "\n"; }
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (auto& s : test) { int pr = net.predict(s.x) >= 0.5f;
        if (pr && s.y) ++tp; else if (pr) ++fp; else if (s.y) ++fn; else ++tn; }
    const int reps = 50; volatile real sink = 0;
    auto t2 = clk::now();
    for (int r = 0; r < reps; ++r) for (auto& s : test) sink = sink + net.predict(s.x);
    double t_inf = sec(t2, clk::now());

    double acc = double(tp + tn) / (tp + tn + fp + fn), pre = tp + fp ? double(tp) / (tp + fp) : 0,
           rec = tp + fn ? double(tp) / (tp + fn) : 0, f1 = pre + rec ? 2 * pre * rec / (pre + rec) : 0;
    printf("RESULT impl=cpp_%s epochs=%d train_s=%.4f infer_us=%.3f acc=%.4f prec=%.4f rec=%.4f f1=%.4f tn=%d fp=%d fn=%d tp=%d\n",
           sizeof(real) == 4 ? "float" : "double", epochs, t_train, t_inf / (reps * (double)nte) * 1e6,
           acc, pre, rec, f1, tn, fp, fn, tp);
}
