#include <algorithm>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_num_threads = 8;

enum class PartitionStrategy {
    Cyclic,
    Block,
};

struct TimingStats {
    std::atomic<long long> division_ns;
    std::atomic<long long> elimination_ns;
    std::atomic<long long> barrier_ns;
    long long backsub_ns;

    TimingStats() : division_ns(0), elimination_ns(0), barrier_ns(0), backsub_ns(0) {}
};

struct SharedConfig {
    int n;
    float* A;
    int num_threads;
    bool use_neon;
    PartitionStrategy strategy;
    TimingStats* stats;
};

struct ThreadContext {
    SharedConfig* shared;
    int tid;
};

pthread_barrier_t g_barrier_division;
pthread_barrier_t g_barrier_elimination;

inline long long ns_between(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

inline double ns_to_ms(long long ns) {
    return static_cast<double>(ns) / 1e6;
}

inline void add_ns(std::atomic<long long>& dst, long long value) {
    dst.fetch_add(value, std::memory_order_relaxed);
}

inline std::pair<int, int> block_range(int row_begin, int row_end, int tid, int num_threads) {
    const int total = std::max(0, row_end - row_begin);
    const int base = total / num_threads;
    const int extra = total % num_threads;
    const int offset = tid * base + std::min(tid, extra);
    const int count = base + (tid < extra ? 1 : 0);
    const int begin = row_begin + offset;
    return std::make_pair(begin, begin + count);
}

inline void normalize_row_scalar(int n, float* A, int k) {
    const float pivot = A[k * (n + 1) + k];
    for (int j = k + 1; j <= n; ++j) {
        A[k * (n + 1) + j] /= pivot;
    }
    A[k * (n + 1) + k] = 1.0f;
}

inline void normalize_row_neon(int n, float* A, int k) {
    const float pivot = A[k * (n + 1) + k];
    float32x4_t vt = vdupq_n_f32(pivot);
    int j = k + 1;
    for (; j + 4 <= n + 1; j += 4) {
        float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
        va = vdivq_f32(va, vt);
        vst1q_f32(&A[k * (n + 1) + j], va);
    }
    for (; j <= n; ++j) {
        A[k * (n + 1) + j] /= pivot;
    }
    A[k * (n + 1) + k] = 1.0f;
}

inline void eliminate_row_scalar(int n, float* A, int k, int i) {
    const float factor = A[i * (n + 1) + k];
    for (int j = k + 1; j <= n; ++j) {
        A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
    }
    A[i * (n + 1) + k] = 0.0f;
}

inline void eliminate_row_neon(int n, float* A, int k, int i) {
    const float factor_scalar = A[i * (n + 1) + k];
    float32x4_t factor = vdupq_n_f32(factor_scalar);
    int j = k + 1;
    for (; j + 4 <= n + 1; j += 4) {
        float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
        float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
        row_i = vsubq_f32(row_i, vmulq_f32(factor, row_k));
        vst1q_f32(&A[i * (n + 1) + j], row_i);
    }
    for (; j <= n; ++j) {
        A[i * (n + 1) + j] -= factor_scalar * A[k * (n + 1) + j];
    }
    A[i * (n + 1) + k] = 0.0f;
}

void back_substitution_serial(int n, float* A) {
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        }
        A[i * (n + 1) + n] = sum;
    }
}

void gauss_serial_impl(int n, float* A, bool use_neon, TimingStats* stats) {
    for (int k = 0; k < n; ++k) {
        Clock::time_point t0 = Clock::now();
        if (use_neon) {
            normalize_row_neon(n, A, k);
        } else {
            normalize_row_scalar(n, A, k);
        }
        Clock::time_point t1 = Clock::now();
        add_ns(stats->division_ns, ns_between(t0, t1));

        t0 = Clock::now();
        for (int i = k + 1; i < n; ++i) {
            if (use_neon) {
                eliminate_row_neon(n, A, k, i);
            } else {
                eliminate_row_scalar(n, A, k, i);
            }
        }
        t1 = Clock::now();
        add_ns(stats->elimination_ns, ns_between(t0, t1));
    }

    Clock::time_point t0 = Clock::now();
    back_substitution_serial(n, A);
    Clock::time_point t1 = Clock::now();
    stats->backsub_ns = ns_between(t0, t1);
}

void run_pthread_task(const ThreadContext& ctx) {
    SharedConfig& cfg = *ctx.shared;
    const int n = cfg.n;
    float* A = cfg.A;

    for (int k = 0; k < n; ++k) {
        if (ctx.tid == 0) {
            Clock::time_point t0 = Clock::now();
            if (cfg.use_neon) {
                normalize_row_neon(n, A, k);
            } else {
                normalize_row_scalar(n, A, k);
            }
            Clock::time_point t1 = Clock::now();
            add_ns(cfg.stats->division_ns, ns_between(t0, t1));
        }

        Clock::time_point b0 = Clock::now();
        pthread_barrier_wait(&g_barrier_division);
        Clock::time_point b1 = Clock::now();
        add_ns(cfg.stats->barrier_ns, ns_between(b0, b1));

        Clock::time_point e0 = Clock::now();
        if (cfg.strategy == PartitionStrategy::Cyclic) {
            for (int i = k + 1 + ctx.tid; i < n; i += cfg.num_threads) {
                if (cfg.use_neon) {
                    eliminate_row_neon(n, A, k, i);
                } else {
                    eliminate_row_scalar(n, A, k, i);
                }
            }
        } else {
            std::pair<int, int> range = block_range(k + 1, n, ctx.tid, cfg.num_threads);
            for (int i = range.first; i < range.second; ++i) {
                if (cfg.use_neon) {
                    eliminate_row_neon(n, A, k, i);
                } else {
                    eliminate_row_scalar(n, A, k, i);
                }
            }
        }
        Clock::time_point e1 = Clock::now();
        add_ns(cfg.stats->elimination_ns, ns_between(e0, e1));

        b0 = Clock::now();
        pthread_barrier_wait(&g_barrier_elimination);
        b1 = Clock::now();
        add_ns(cfg.stats->barrier_ns, ns_between(b0, b1));
    }
}

void* pthread_worker(void* arg) {
    ThreadContext* ctx = static_cast<ThreadContext*>(arg);
    run_pthread_task(*ctx);
    return nullptr;
}

void gauss_pthread_impl(int n, float* A, bool use_neon, PartitionStrategy strategy, TimingStats* stats) {
    const int num_threads = std::max(1, g_num_threads);
    const int child_threads = std::max(0, num_threads - 1);
    std::vector<pthread_t> threads(child_threads);
    SharedConfig shared = {n, A, num_threads, use_neon, strategy, stats};
    std::vector<ThreadContext> contexts(num_threads);

    pthread_barrier_init(&g_barrier_division, NULL, num_threads);
    pthread_barrier_init(&g_barrier_elimination, NULL, num_threads);

    for (int tid = 0; tid < num_threads; ++tid) {
        contexts[tid].shared = &shared;
        contexts[tid].tid = tid;
    }

    for (int tid = 0; tid < child_threads; ++tid) {
        pthread_create(&threads[tid], NULL, pthread_worker, &contexts[tid]);
    }

    run_pthread_task(contexts[num_threads - 1]);

    for (int tid = 0; tid < child_threads; ++tid) {
        pthread_join(threads[tid], NULL);
    }

    pthread_barrier_destroy(&g_barrier_division);
    pthread_barrier_destroy(&g_barrier_elimination);

    Clock::time_point t0 = Clock::now();
    back_substitution_serial(n, A);
    Clock::time_point t1 = Clock::now();
    stats->backsub_ns = ns_between(t0, t1);
}

std::vector<float> generate_matrix(int n) {
    std::vector<float> A(n * (n + 1), 0.0f);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float val = static_cast<float>(rand() % 100) / 100.0f;
                A[i * (n + 1) + j] = val;
                sum += std::abs(val);
            }
        }
        A[i * (n + 1) + i] = sum + 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        float b = 0.0f;
        for (int j = 0; j < n; ++j) {
            b += A[i * (n + 1) + j] * (j + 1);
        }
        A[i * (n + 1) + n] = b;
    }
    return A;
}

float check_error(int n, const float* A) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = i + 1.0f;
        float computed = A[i * (n + 1) + n];
        float err = std::abs(expected - computed);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

struct RunResult {
    double total_ms;
    double division_ms;
    double elimination_ms;
    double barrier_ms;
    double backsub_ms;
    float err;
};

template <typename Func>
RunResult benchmark_variant(int n, const std::vector<float>& src, Func func) {
    std::vector<float> A = src;
    TimingStats stats;
    Clock::time_point t0 = Clock::now();
    func(n, A.data(), &stats);
    Clock::time_point t1 = Clock::now();

    RunResult r;
    r.total_ms = ns_to_ms(ns_between(t0, t1));
    r.division_ms = ns_to_ms(stats.division_ns.load(std::memory_order_relaxed));
    r.elimination_ms = ns_to_ms(stats.elimination_ns.load(std::memory_order_relaxed));
    r.barrier_ms = ns_to_ms(stats.barrier_ns.load(std::memory_order_relaxed));
    r.backsub_ms = ns_to_ms(stats.backsub_ns);
    r.err = check_error(n, A.data());
    return r;
}

void print_result(const std::string& name, const RunResult& r, double serial_ms) {
    std::cout << name << "_ms=" << r.total_ms << " err=" << r.err << std::endl;
    std::cout << name << "_division_ms=" << r.division_ms << std::endl;
    std::cout << name << "_elimination_ms=" << r.elimination_ms << std::endl;
    std::cout << name << "_barrier_ms=" << r.barrier_ms << std::endl;
    std::cout << name << "_backsub_ms=" << r.backsub_ms << std::endl;
    std::cout << "speedup_" << name << "=" << serial_ms / r.total_ms << std::endl;
}

std::string strategy_name(PartitionStrategy strategy) {
    return strategy == PartitionStrategy::Cyclic ? "cyclic" : "block";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::vector<int> sizes;
    sizes.push_back(1024);
    sizes.push_back(2048);

    std::vector<int> thread_counts;
    thread_counts.push_back(2);
    thread_counts.push_back(4);
    thread_counts.push_back(8);

    if (argc > 1) {
        sizes.clear();
        sizes.push_back(std::atoi(argv[1]));
    }
    if (argc > 2) {
        thread_counts.clear();
        thread_counts.push_back(std::atoi(argv[2]));
    }

    for (size_t si = 0; si < sizes.size(); ++si) {
        const int n = sizes[si];
        std::vector<float> A0 = generate_matrix(n);

        for (size_t ti = 0; ti < thread_counts.size(); ++ti) {
            g_num_threads = thread_counts[ti];
            std::cout << "n=" << n << " threads=" << g_num_threads << std::endl;

            RunResult serial = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) { gauss_serial_impl(n0, a0, false, s0); });
            print_result("serial", serial, serial.total_ms);

            RunResult neon = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) { gauss_serial_impl(n0, a0, true, s0); });
            print_result("neon_full", neon, serial.total_ms);

            RunResult pthread_cyclic = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_pthread_impl(n0, a0, false, PartitionStrategy::Cyclic, s0);
                });
            print_result("pthread_cyclic", pthread_cyclic, serial.total_ms);

            RunResult pthread_block = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_pthread_impl(n0, a0, false, PartitionStrategy::Block, s0);
                });
            print_result("pthread_block", pthread_block, serial.total_ms);

            RunResult pthread_neon_cyclic = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_pthread_impl(n0, a0, true, PartitionStrategy::Cyclic, s0);
                });
            print_result("pthread_neon_cyclic", pthread_neon_cyclic, serial.total_ms);

            RunResult pthread_neon_block = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_pthread_impl(n0, a0, true, PartitionStrategy::Block, s0);
                });
            print_result("pthread_neon_block", pthread_neon_block, serial.total_ms);

            std::cout << "best_partition_plain="
                      << (pthread_cyclic.total_ms <= pthread_block.total_ms ? strategy_name(PartitionStrategy::Cyclic)
                                                                            : strategy_name(PartitionStrategy::Block))
                      << std::endl;
            std::cout << "best_partition_neon="
                      << (pthread_neon_cyclic.total_ms <= pthread_neon_block.total_ms
                              ? strategy_name(PartitionStrategy::Cyclic)
                              : strategy_name(PartitionStrategy::Block))
                      << std::endl;
        }
    }

    return 0;
}