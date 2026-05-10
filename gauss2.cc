#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
// 自行添加需要的头文件

#include <arm_neon.h>

// ---------- 矩阵生成 ----------
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

// ---------- 串行版本 ----------
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
    // 回代
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}

// ---------- NEON 未对齐版本 ----------
void gauss_neon_unaligned(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        // 除法部分向量化（4路）
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]); // 未对齐加载
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);              // 未对齐存储
        }
        for (; j <= n; ++j)
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
        A[k * (n + 1) + k] = 1.0f;

        // 消去部分向量化（4路）
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
    // 回代（串行）
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}
// NEON 对齐版本
void gauss_neon_aligned(int n, float* A) {
    for (int k = 0; k < n; ++k) {
        // ========== 除法部分（对齐） ==========
        float32x4_t vt = vdupq_n_f32(A[k * (n + 1) + k]);
        int j = k + 1;

        // 推进到 16 字节对齐：&A[k][j] 地址必须 16 字节对齐
        // float* 地址对齐条件：(uintptr_t)(&A[k*(n+1) + j]) % 16 == 0
        while (j <= n && ((uintptr_t)(&A[k * (n + 1) + j]) & 15)) {
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];
            j++;
        }

        // 对齐主体：一次处理 4 个 float
        for (; j + 4 <= n + 1; j += 4) {
            float32x4_t va = vld1q_f32(&A[k * (n + 1) + j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k * (n + 1) + j], va);
        }

        // 尾部标量
        for (; j <= n; ++j)
            A[k * (n + 1) + j] /= A[k * (n + 1) + k];

        A[k * (n + 1) + k] = 1.0f;

        // ========== 消去部分（对齐） ==========
        for (int i = k + 1; i < n; ++i) {
            float32x4_t factor = vdupq_n_f32(A[i * (n + 1) + k]);

            int j = k + 1;

            // 对齐推进（同样以第k行地址为准）
            while (j <= n && ((uintptr_t)(&A[k * (n + 1) + j]) & 15)) {
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
                j++;
            }

            // 对齐主体
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t row_k = vld1q_f32(&A[k * (n + 1) + j]);
                float32x4_t row_i = vld1q_f32(&A[i * (n + 1) + j]);
                float32x4_t tmp   = vmulq_f32(factor, row_k);
                row_i = vsubq_f32(row_i, tmp);
                vst1q_f32(&A[i * (n + 1) + j], row_i);
            }

            // 尾部标量
            for (; j <= n; ++j)
                A[i * (n + 1) + j] -= A[i * (n + 1) + k] * A[k * (n + 1) + j];
            A[i * (n + 1) + k] = 0.0f;
        }
    }

    // 回代（串行，不变）
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        A[i * (n + 1) + n] = sum;
    }
}
// ---------- 误差检查 ----------
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

int main(int argc, char *argv[])
{
    int n = 1024;
    if (argc > 1) n = std::atoi(argv[1]);

    std::vector<float> A0 = generate_matrix(n);
    std::vector<float> A(n * (n + 1));

    // 串行测试
    std::copy(A0.begin(), A0.end(), A.begin());
    auto t1 = std::chrono::high_resolution_clock::now();
    gauss_serial(n, A.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    float err_serial = check_error(n, A.data());

    // NEON 测试
    std::copy(A0.begin(), A0.end(), A.begin());
    t1 = std::chrono::high_resolution_clock::now();
    gauss_neon_unaligned(n, A.data());
    t2 = std::chrono::high_resolution_clock::now();
    double neon_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    float err_neon = check_error(n, A.data());

    std::copy(A0.begin(), A0.end(), A.begin());
    t1 = std::chrono::high_resolution_clock::now();
    gauss_neon_aligned(n, A.data());
    t2 = std::chrono::high_resolution_clock::now();
    double neon_aligned_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    float err_neon_aligned = check_error(n, A.data());
    // 输出
    std::cout << "n=" << n << std::endl;
    std::cout << "serial_ms=" << serial_ms << " err=" << err_serial << std::endl;
    std::cout << "neon_ms=" << neon_ms << " err=" << err_neon << std::endl;
    std::cout << "neon_aligned_ms=" << neon_aligned_ms << " err=" << err_neon_aligned << std::endl;
    std::cout << "speedup1=" << serial_ms / neon_ms << std::endl;
    std::cout << "speedup2=" << serial_ms / neon_aligned_ms << std::endl;
    return 0;
}
