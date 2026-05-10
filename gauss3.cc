#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <arm_neon.h>

//矩阵生成（对角占优，已知解 x[i] = i+1）
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
        for (int j = 0; j < n; ++j)
            b += A[i * (n + 1) + j] * (j + 1);
        A[i * (n + 1) + n] = b;
    }
    return A;
}

//串行版本 
void gauss_serial(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * (n + 1) + k];
        for (int j = k + 1; j <= n; ++j)
            A[k * (n + 1) + j] /= pivot;
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * (n + 1) + k];
            for (int j = k + 1; j <= n; ++j)
                A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
            A[i * (n + 1) + k] = 0.0f;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}

//仅除法向量化（消去部分仍为串行）
void gauss_neon_divonly(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        // 除法部分向量化
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);
        }
        for (; j <= n; ++j)
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
        A[k * (n + 1) + k] = 1.0f;

        // 消去部分保持串行
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * (n + 1) + k];
            for (int j = k + 1; j <= n; ++j)
                A[i * (n + 1) + j] -= factor * A[k * (n + 1) + j];
            A[i * (n + 1) + k] = 0.0f;
        }
    }
    // 回代串行
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}

//仅消去向量化（除法部分仍为串行）
void gauss_neon_elimonly(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        // 除法部分串行
        float pivot = A[k * (n + 1) + k];
        for (int j = k + 1; j <= n; ++j)
            A[k * (n + 1) + j] /= pivot;
        A[k * (n + 1) + k] = 1.0f;

        // 消去部分向量化
        for (int i = k + 1; i < n; ++i) {
            float32x4_t factor = vdupq_n_f32(A[i * (n + 1) + k]);
            int j = k + 1;
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
                float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
                float32x4_t tmp   = vmulq_f32(factor, row_k);
                row_i = vsubq_f32(row_i, tmp);
                vst1q_f32(&A[i * (n + 1) + j], row_i);
            }
            for (; j <= n; ++j)
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
            A[i * (n + 1) + k] = 0.0f;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}

//完整向量化（除法+消去均使用NEON）
void gauss_neon_full(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);
        }
        for (; j <= n; ++j)
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
        A[k * (n + 1) + k] = 1.0f;

        for (int i = k + 1; i < n; ++i) {
            float32x4_t factor = vdupq_n_f32(A[i * (n + 1) + k]);
            int j = k + 1;
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
                float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
                float32x4_t tmp   = vmulq_f32(factor, row_k);
                row_i = vsubq_f32(row_i, tmp);
                vst1q_f32(&A[i * (n + 1) + j], row_i);
            }
            for (; j <= n; ++j)
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
            A[i * (n + 1) + k] = 0.0f;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}

//  误差检查 
float check_error(int n, const float* A) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = i + 1.0f;
        float computed = A[i * (n + 1) + n];
        float err = std::abs(expected - computed);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// 测试主函数
int main(int argc, char* argv[]) {
    int n = 2048;
    if (argc > 1) n = std::atoi(argv[1]);

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

    std::cout << "n=" << n << std::endl;
    double t_serial    = run("serial", gauss_serial);
    double t_divonly   = run("neon_divonly", gauss_neon_divonly);
    double t_elimonly  = run("neon_elimonly", gauss_neon_elimonly);
    double t_full      = run("neon_full", gauss_neon_full);

    std::cout << "speedup_divonly=" << t_serial / t_divonly << std::endl;
    std::cout << "speedup_elimonly=" << t_serial / t_elimonly << std::endl;
    std::cout << "speedup_full=" << t_serial / t_full << std::endl;

    return 0;
}



