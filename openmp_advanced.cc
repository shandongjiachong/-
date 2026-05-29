#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_num_threads = 8;

enum class OmpScheduleKind {
    Static,
    Dynamic,
    Guided,
};

struct TimingStats {
    double division_ms;
    double elimination_ms;
    double backsub_ms;

    TimingStats() : division_ms(0.0), elimination_ms(0.0), backsub_ms(0.0) {}
};

inline long long ns_between(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

inline double ns_to_ms(long long ns) {
    return static_cast<double>(ns) / 1e6;
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
        stats->division_ms += ns_to_ms(ns_between(t0, t1));

        t0 = Clock::now();
        for (int i = k + 1; i < n; ++i) {
            if (use_neon) {
                eliminate_row_neon(n, A, k, i);
            } else {
                eliminate_row_scalar(n, A, k, i);
            }
        }
        t1 = Clock::now();
        stats->elimination_ms += ns_to_ms(ns_between(t0, t1));
    }

    Clock::time_point t0 = Clock::now();
    back_substitution_serial(n, A);
    Clock::time_point t1 = Clock::now();
    stats->backsub_ms = ns_to_ms(ns_between(t0, t1));
}

void gauss_openmp_impl(int n, float* A, bool use_neon, OmpScheduleKind schedule_kind, TimingStats* stats) {
    omp_set_num_threads(std::max(1, g_num_threads));
    if (schedule_kind == OmpScheduleKind::Static) {
        omp_set_schedule(omp_sched_static, 0);
    } else if (schedule_kind == OmpScheduleKind::Dynamic) {
        omp_set_schedule(omp_sched_dynamic, 1);
    } else {
        omp_set_schedule(omp_sched_guided, 1);
    }

    double division_acc = 0.0;
    double elimination_acc = 0.0;

#pragma omp parallel reduction(+ : division_acc, elimination_acc) default(shared)
    {
        for (int k = 0; k < n; ++k) {
#pragma omp single
            {
                Clock::time_point t0 = Clock::now();
                if (use_neon) {
                    normalize_row_neon(n, A, k);
                } else {
                    normalize_row_scalar(n, A, k);
                }
                Clock::time_point t1 = Clock::now();
                division_acc += ns_to_ms(ns_between(t0, t1));
            }

            Clock::time_point e0 = Clock::now();
#pragma omp for schedule(runtime)
            for (int i = k + 1; i < n; ++i) {
                if (use_neon) {
                    eliminate_row_neon(n, A, k, i);
                } else {
                    eliminate_row_scalar(n, A, k, i);
                }
            }
            Clock::time_point e1 = Clock::now();
            elimination_acc += ns_to_ms(ns_between(e0, e1));
        }
    }

    stats->division_ms = division_acc;
    stats->elimination_ms = elimination_acc;

    Clock::time_point t0 = Clock::now();
    back_substitution_serial(n, A);
    Clock::time_point t1 = Clock::now();
    stats->backsub_ms = ns_to_ms(ns_between(t0, t1));
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
    r.division_ms = stats.division_ms;
    r.elimination_ms = stats.elimination_ms;
    r.backsub_ms = stats.backsub_ms;
    r.err = check_error(n, A.data());
    return r;
}

void print_result(const std::string& name, const RunResult& r, double serial_ms) {
    std::cout << name << "_ms=" << r.total_ms << " err=" << r.err << std::endl;
    std::cout << name << "_division_ms=" << r.division_ms << std::endl;
    std::cout << name << "_elimination_ms=" << r.elimination_ms << std::endl;
    std::cout << name << "_backsub_ms=" << r.backsub_ms << std::endl;
    std::cout << "speedup_" << name << "=" << serial_ms / r.total_ms << std::endl;
}

std::string schedule_name(OmpScheduleKind kind) {
    if (kind == OmpScheduleKind::Static) {
        return "static";
    }
    if (kind == OmpScheduleKind::Dynamic) {
        return "dynamic";
    }
    return "guided";
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

            RunResult openmp_static = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, false, OmpScheduleKind::Static, s0);
                });
            print_result("openmp_static", openmp_static, serial.total_ms);

            RunResult openmp_dynamic = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, false, OmpScheduleKind::Dynamic, s0);
                });
            print_result("openmp_dynamic", openmp_dynamic, serial.total_ms);

            RunResult openmp_guided = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, false, OmpScheduleKind::Guided, s0);
                });
            print_result("openmp_guided", openmp_guided, serial.total_ms);

            RunResult openmp_neon_static = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, true, OmpScheduleKind::Static, s0);
                });
            print_result("openmp_neon_static", openmp_neon_static, serial.total_ms);

            RunResult openmp_neon_dynamic = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, true, OmpScheduleKind::Dynamic, s0);
                });
            print_result("openmp_neon_dynamic", openmp_neon_dynamic, serial.total_ms);

            RunResult openmp_neon_guided = benchmark_variant(
                n, A0, [](int n0, float* a0, TimingStats* s0) {
                    gauss_openmp_impl(n0, a0, true, OmpScheduleKind::Guided, s0);
                });
            print_result("openmp_neon_guided", openmp_neon_guided, serial.total_ms);

            std::vector<std::pair<double, OmpScheduleKind> > plain_candidates;
            plain_candidates.push_back(std::make_pair(openmp_static.total_ms, OmpScheduleKind::Static));
            plain_candidates.push_back(std::make_pair(openmp_dynamic.total_ms, OmpScheduleKind::Dynamic));
            plain_candidates.push_back(std::make_pair(openmp_guided.total_ms, OmpScheduleKind::Guided));
            std::sort(plain_candidates.begin(), plain_candidates.end());

            std::vector<std::pair<double, OmpScheduleKind> > neon_candidates;
            neon_candidates.push_back(std::make_pair(openmp_neon_static.total_ms, OmpScheduleKind::Static));
            neon_candidates.push_back(std::make_pair(openmp_neon_dynamic.total_ms, OmpScheduleKind::Dynamic));
            neon_candidates.push_back(std::make_pair(openmp_neon_guided.total_ms, OmpScheduleKind::Guided));
            std::sort(neon_candidates.begin(), neon_candidates.end());

            std::cout << "best_schedule_plain=" << schedule_name(plain_candidates.front().second) << std::endl;
            std::cout << "best_schedule_neon=" << schedule_name(neon_candidates.front().second) << std::endl;
        }
    }

    return 0;
}