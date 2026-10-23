// Timing harness: reuses the ORIGINAL credit_mlp_fast.cpp unchanged (its main is renamed).
#define main orig_main
#include "credit_mlp_fast.cpp"
#undef main
#include <chrono>

int main(int argc, char** argv) {
    int epochs = argc > 2 ? atoi(argv[2]) : 80;
    using clk = chrono::steady_clock;
    auto sec = [](clk::time_point a, clk::time_point b) { return chrono::duration<double>(b - a).count(); };

    auto t0 = clk::now();
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
    double t_load = sec(t0, clk::now());

    FastMLP net(23, 16, 0.01f);
    auto t1 = clk::now();
    for (int e = 0; e < epochs; ++e) {
        shuffle(train.begin(), train.end(), rng);
        for (const auto& s : train) net.trainOne(s);
    }
    double t_train = sec(t1, clk::now());

    // inference: repeat the test set so the timer resolution is meaningful
    const int reps = 50;
    volatile float sink = 0;
    auto t2 = clk::now();
    for (int r = 0; r < reps; ++r)
        for (const auto& s : test) sink = sink + net.predict(s.x);
    double t_inf = sec(t2, clk::now());

    auto m = evaluate(net, test);
    printf("CPP_RESULT epochs=%d load_s=%.3f train_s=%.4f train_per_epoch_s=%.5f infer_us_per_sample=%.3f acc=%.4f prec=%.4f rec=%.4f\n",
           epochs, t_load, t_train, t_train / epochs, t_inf / (reps * test.size()) * 1e6,
           m.accuracy(), m.precision(), m.recall());
    return 0;
}
