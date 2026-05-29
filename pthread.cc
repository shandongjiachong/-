#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <vector>

namespace {

int g_num_threads = 1;

struct ThreadContext {
    int n;
    float* A;
    int tid;
    int num_threads;
    bool use_neon;
};

pthread_barrier_t g_barrier_division;
pthread_barrier_t g_barrier_elimination;

inline void normalize_row_scalar(int n, float* A, int k) {
    float pivot = A[k * (n + 1) + k];
    for (int j = k + 1; j <= n; ++j) {
        A[k * (n + 1) + j] /= pivot;
    }
    A[k * (n + 1) + k] = 1.0f;
}

inline void normalize_row_neon(int n, float* A, int k) {
    float pivot = A[k * (n + 1) + k];
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
    float factor = A[i * (n + 1) + k];
    for (int j = k + 1; j <= n; ++j) {
        A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
    }
    A[i * (n + 1) + k] = 0.0f;
}

inline void eliminate_row_neon(int n, float* A, int k, int i) {
    float factor_scalar = A[i * (n + 1) + k];
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

void run_pthread_task(const ThreadContext& ctx) {
    const int n = ctx.n;
    float* A = ctx.A;

    for (int k = 0; k < n; ++k) {
        if (ctx.tid == 0) {
            if (ctx.use_neon) {
                normalize_row_neon(n, A, k);
            } else {
                normalize_row_scalar(n, A, k);
            }
        }

        pthread_barrier_wait(&g_barrier_division);

        for (int i = k + 1 + ctx.tid; i < n; i += ctx.num_threads) {
            if (ctx.use_neon) {
                eliminate_row_neon(n, A, k, i);
            } else {
                eliminate_row_scalar(n, A, k, i);
            }
        }

        pthread_barrier_wait(&g_barrier_elimination);
    }
}

void* pthread_worker(void* arg) {
    auto* ctx = static_cast<ThreadContext*>(arg);
    run_pthread_task(*ctx);
    return nullptr;
}

void gauss_pthread_impl(int n, float* A, bool use_neon) {
    const int num_threads = std::max(1, g_num_threads);
    const int child_threads = std::max(0, num_threads - 1);
    std::vector<pthread_t> threads(child_threads);
    std::vector<ThreadContext> contexts(num_threads);

    pthread_barrier_init(&g_barrier_division, nullptr, num_threads);
    pthread_barrier_init(&g_barrier_elimination, nullptr, num_threads);

    for (int tid = 0; tid < num_threads; ++tid) {
        contexts[tid] = ThreadContext{n, A, tid, num_threads, use_neon};
    }

    for (int tid = 0; tid < child_threads; ++tid) {
        pthread_create(&threads[tid], nullptr, pthread_worker, &contexts[tid]);
    }

    run_pthread_task(contexts[num_threads - 1]);

    for (int tid = 0; tid < child_threads; ++tid) {
        pthread_join(threads[tid], nullptr);
    }

    pthread_barrier_destroy(&g_barrier_division);
    pthread_barrier_destroy(&g_barrier_elimination);

    back_substitution_serial(n, A);
}

}  // namespace

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

void gauss_serial(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * (n + 1) + k];
        for (int j = k + 1; j <= n; ++j) {
            A[k * (n + 1) + j] /= pivot;
        }
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * (n + 1) + k];
            for (int j = k + 1; j <= n; ++j) {
                A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
            }
            A[i * (n + 1) + k] = 0.0f;
        }
    }

    back_substitution_serial(n, A);
}

void gauss_neon_divonly(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);
        }
        for (; j <= n; ++j) {
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
        }
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * (n + 1) + k];
            for (int j = k + 1; j <= n; ++j) {
                A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
            }
            A[i * (n + 1) + k] = 0.0f;
        }
    }

    back_substitution_serial(n, A);
}

void gauss_neon_elimonly(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * (n + 1) + k];
        for (int j = k + 1; j <= n; ++j) {
            A[k * (n + 1) + j] /= pivot;
        }
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float32x4_t factor = vdupq_n_f32(A[i * (n + 1) + k]);
            int j = k + 1;
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
                float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
                float32x4_t tmp = vmulq_f32(factor, row_k);
                row_i = vsubq_f32(row_i, tmp);
                vst1q_f32(&A[i * (n + 1) + j], row_i);
            }
            for (; j <= n; ++j) {
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
            }
            A[i * (n + 1) + k] = 0.0f;
        }
    }

    back_substitution_serial(n, A);
}

void gauss_neon_full(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);
        }
        for (; j <= n; ++j) {
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
        }
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float32x4_t factor = vdupq_n_f32(A[i * (n + 1) + k]);
            int j = k + 1;
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
                float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
                float32x4_t tmp = vmulq_f32(factor, row_k);
                row_i = vsubq_f32(row_i, tmp);
                vst1q_f32(&A[i * (n + 1) + j], row_i);
            }
            for (; j <= n; ++j) {
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
            }
            A[i * (n + 1) + k] = 0.0f;
        }
    }

    back_substitution_serial(n, A);
}

void gauss_pthread(int n, float* A) {
    gauss_pthread_impl(n, A, false);
}

void gauss_pthread_neon(int n, float* A) {
    gauss_pthread_impl(n, A, true);
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

int main(int argc, char* argv[]) {
    int n = 2048;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    std::vector<float> A0 = generate_matrix(n);
    std::vector<float> A(n * (n + 1));

    auto run = [&](const char* name, void (*func)(int, float*)) {
        std::copy(A0.begin(), A0.end(), A.begin());
        auto t1 = std::chrono::high_resolution_clock::now();
        func(n, A.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        float err = check_error(n, A.data());
        std::cout << name << "_ms=" << ms << " err=" << err << std::endl;
        return ms;
    };

    std::cout << "n=" << n << " threads=" << g_num_threads << std::endl;
    double t_serial = run("serial", gauss_serial);
    double t_divonly = run("neon_divonly", gauss_neon_divonly);
    double t_elimonly = run("neon_elimonly", gauss_neon_elimonly);
    double t_full = run("neon_full", gauss_neon_full);
    double t_pthread = run("pthread", gauss_pthread);
    double t_pthread_neon = run("pthread_neon", gauss_pthread_neon);

    std::cout << "speedup_divonly=" << t_serial / t_divonly << std::endl;
    std::cout << "speedup_elimonly=" << t_serial / t_elimonly << std::endl;
    std::cout << "speedup_full=" << t_serial / t_full << std::endl;
    std::cout << "speedup_pthread=" << t_serial / t_pthread << std::endl;
    std::cout << "speedup_pthread_neon=" << t_serial / t_pthread_neon << std::endl;

    return 0;
}