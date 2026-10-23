#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

inline float sig(float x) { 
    return 1.0f / (1.0f + exp(-x)); 
}

struct Sample {
    vector<float> x;
    int y;
};

class FastMLP {
    int nIn, nH;
    vector<float> W1; // Contiguous 1D array: nH * nIn
    vector<float> b1; // Size: nH
    vector<float> W2; // Size: nH
    float b2, lr;

    // Pre-allocated workspace buffers to eliminate heap allocations inside hot loops
    mutable vector<float> h_buf;
    vector<float> dh_buf;
    vector<float> dW2_buf;
    vector<float> dW1_buf;

public:
    FastMLP(int in, int hidden, float rate = 0.01f) :
        nIn(in), nH(hidden), W1(hidden * in), b1(hidden, 0.0f),
        W2(hidden), b2(0.0f), lr(rate),
        h_buf(hidden), dh_buf(hidden), dW2_buf(hidden), dW1_buf(hidden * in) 
    {
        mt19937 gen(42);
        normal_distribution<float> d1(0.0f, sqrt(2.0f / in));
        for (auto &w : W1) w = d1(gen);
        
        normal_distribution<float> d2(0.0f, sqrt(2.0f / hidden));
        for (auto &w : W2) w = d2(gen);
    }

    // Forward pass optimized for compiler auto-vectorization (SIMD)
    float forward(const float* x, vector<float>& h) const {
        for (int j = 0; j < nH; ++j) {
            float sum = b1[j];
            const float* w_row = &W1[j * nIn];
            
            // Vectorized dot product loop
            for (int i = 0; i < nIn; ++i) {
                sum += w_row[i] * x[i];
            }
            h[j] = tanh(sum);
        }

        float z = b2;
        for (int j = 0; j < nH; ++j) {
            z += W2[j] * h[j];
        }
        return sig(z);
    }

    float predict(const vector<float>& x) const {
        return forward(x.data(), h_buf);
    }

    void trainOne(const Sample& s) {
        float p = forward(s.x.data(), h_buf);
        float dz = p - s.y;

        for (int j = 0; j < nH; ++j) {
            dW2_buf[j] = dz * h_buf[j];
            dh_buf[j] = dz * W2[j] * (1.0f - h_buf[j] * h_buf[j]);
        }

        // Parameter updates with zero dynamic allocations
        for (int j = 0; j < nH; ++j) {
            float dh = dh_buf[j];
            float* w_row = &W1[j * nIn];
            const float* x_ptr = s.x.data();
            
            for (int i = 0; i < nIn; ++i) {
                w_row[i] -= lr * dh * x_ptr[i];
            }
            b1[j] -= lr * dh;
            W2[j] -= lr * dW2_buf[j];
        }
        b2 -= lr * dz;
    }
};

vector<string> split(const string& line) {
    vector<string> a; string s; stringstream ss(line);
    while (getline(ss, s, ',')) a.push_back(s);
    return a;
}

vector<Sample> loadCSV(const string& file) {
    ifstream f(file);
    if (!f) { cerr << "Cannot open " << file << "\n"; exit(1); }

    string line; getline(f, line); // header
    vector<Sample> data;

    while (getline(f, line)) {
        auto c = split(line);
        if (c.size() < 25) continue;

        Sample s;
        for (int i = 1; i <= 23; ++i)
            s.x.push_back(stof(c[i]));
        s.y = stoi(c[24]);
        data.push_back(s);
    }
    return data;
}

void standardize(vector<Sample>& train, vector<Sample>& test) {
    int d = train[0].x.size();
    vector<float> mu(d, 0.0f), sd(d, 0.0f);

    for (const auto &s : train)
        for (int j = 0; j < d; ++j) mu[j] += s.x[j];
    for (float &v : mu) v /= train.size();

    for (const auto &s : train)
        for (int j = 0; j < d; ++j)
            sd[j] += (s.x[j] - mu[j]) * (s.x[j] - mu[j]);
    for (float &v : sd) {
        v = sqrt(v / train.size());
        if (v < 1e-12f) v = 1.0f;
    }

    for (auto *set : {&train, &test})
        for (auto &s : *set)
            for (int j = 0; j < d; ++j)
                s.x[j] = (s.x[j] - mu[j]) / sd[j];
}

struct Metrics {
    int tp=0, tn=0, fp=0, fn=0;
    float accuracy() const { return float(tp+tn)/(tp+tn+fp+fn); }
    float precision() const { return tp + fp ? float(tp)/(tp+fp) : 0; }
    float recall() const { return tp + fn ? float(tp)/(tp+fn) : 0; }
    float f1() const {
        float p = precision(), r = recall();
        return p + r ? 2 * p * r / (p + r) : 0;
    }
};

Metrics evaluate(const FastMLP& net, const vector<Sample>& data) {
    Metrics m;
    for (const auto &s : data) {
        int pred = net.predict(s.x) >= 0.5f;
        if (pred && s.y) ++m.tp;
        else if (pred && !s.y) ++m.fp;
        else if (!pred && s.y) ++m.fn;
        else ++m.tn;
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./credit_mlp_fast credit.csv\n";
        return 1;
    }

    auto data = loadCSV(argv[1]);

    vector<Sample> zero, one;
    for (auto &s : data) (s.y ? one : zero).push_back(s);

    mt19937 rng(42);
    shuffle(zero.begin(), zero.end(), rng);
    shuffle(one.begin(), one.end(), rng);

    vector<Sample> train, test;
    size_t zcut = zero.size() * 4 / 5, ocut = one.size() * 4 / 5;
    train.insert(train.end(), zero.begin(), zero.begin() + zcut);
    train.insert(train.end(), one.begin(), one.begin() + ocut);
    test.insert(test.end(), zero.begin() + zcut, zero.end());
    test.insert(test.end(), one.begin() + ocut, one.end());

    standardize(train, test);
    shuffle(train.begin(), train.end(), rng);

    FastMLP net(23, 16, 0.01f);

    const int epochs = 80;
    for (int e = 1; e <= epochs; ++e) {
        shuffle(train.begin(), train.end(), rng);
        for (const auto& s : train) net.trainOne(s);

        if (e % 10 == 0) {
            auto m = evaluate(net, test);
            cout << "Epoch " << setw(3) << e
                 << " | Accuracy " << fixed << setprecision(4)
                 << m.accuracy() << "\n";
        }
    }

    auto m = evaluate(net, test);
    cout << "\n=== FINAL TEST RESULTS (SIMD OPTIMIZED) ===\n";
    cout << "Accuracy : " << m.accuracy() * 100.0f << "%\n";
    cout << "Precision: " << m.precision() * 100.0f << "%\n";
    cout << "Recall   : " << m.recall() * 100.0f << "%\n";
    cout << "F1 Score : " << m.f1() * 100.0f << "%\n\n";
    cout << "Confusion Matrix\n";
    cout << "                 Pred 0   Pred 1\n";
    cout << "Actual 0        " << setw(7) << m.tn << " " << setw(9) << m.fp << "\n";
    cout << "Actual 1        " << setw(7) << m.fn << " " << setw(9) << m.tp << "\n";
    
    return 0;
}
