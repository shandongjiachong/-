#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t e = cmd; \
        if (e != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(e) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

std::vector<float> generate_matrix(int n) {
    std::vector<float> A(n * (n + 1), 0.0f);
    srand(0);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float val = static_cast<float>(rand() % 100) / 100.0f;
                A[i * (n + 1) + j] = val;
                sum += std::fabs(val);
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

    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        }
        A[i * (n + 1) + n] = sum;
    }
}

__global__ void normalize_row_kernel(float* A, int n, int k) {
    __shared__ float pivot;
    int tid = threadIdx.x;
    int width = n + 1;

    if (tid == 0) {
        pivot = A[k * width + k];
    }
    __syncthreads();

    for (int j = k + 1 + tid; j <= n; j += blockDim.x) {
        A[k * width + j] /= pivot;
    }
    __syncthreads();

    if (tid == 0) {
        A[k * width + k] = 1.0f;
    }
}

__global__ void eliminate_kernel(float* A, int n, int k) {
    int row = k + 1 + blockIdx.x;
    int tid = threadIdx.x;
    int width = n + 1;

    if (row >= n) return;

    __shared__ float factor;
    if (tid == 0) {
        factor = A[row * width + k];
    }
    __syncthreads();

    for (int col = k + 1 + tid; col <= n; col += blockDim.x) {
        A[row * width + col] -= factor * A[k * width + col];
    }
    __syncthreads();

    if (tid == 0) {
        A[row * width + k] = 0.0f;
    }
}

void back_substitution(int n, float* A) {
    for (int i = n - 1; i >= 0; --i) {
        float sum = A[i * (n + 1) + n];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * (n + 1) + j] * A[j * (n + 1) + n];
        }
        A[i * (n + 1) + n] = sum;
    }
}

double gauss_gpu_basic(int n, std::vector<float>& A) {
    int width = n + 1;
    size_t bytes = static_cast<size_t>(n) * width * sizeof(float);
    float* d_A = nullptr;

    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMemcpy(d_A, A.data(), bytes, hipMemcpyHostToDevice));

    const int threads = 256;

    for (int k = 0; k < n; ++k) {
        hipLaunchKernelGGL(normalize_row_kernel, dim3(1), dim3(threads), 0, 0, d_A, n, k);
        HIP_CHECK(hipGetLastError());

        if (k + 1 < n) {
            int rows_left = n - k - 1;
            hipLaunchKernelGGL(eliminate_kernel, dim3(rows_left), dim3(threads), 0, 0, d_A, n, k);
            HIP_CHECK(hipGetLastError());
        }
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(A.data(), d_A, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_A));

    back_substitution(n, A.data());

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

float check_error(int n, const float* A) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = i + 1.0f;
        float computed = A[i * (n + 1) + n];
        float err = std::fabs(expected - computed);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main(int argc, char* argv[]) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::vector<int> test_sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            test_sizes.push_back(std::atoi(argv[i]));
        }
    } else {
        test_sizes = {512, 1024, 2048};
    }

    const int repeat = 3;

    std::cout << "===== HIP Gaussian Elimination Basic Test =====" << std::endl;
    std::cout << "device=" << prop.name << std::endl;
    std::cout << "repeat=" << repeat << std::endl;
    std::cout << "format: n trial serial_ms gpu_ms speedup err" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    for (int n : test_sizes) {
        for (int t = 1; t <= repeat; ++t) {
            std::vector<float> A0 = generate_matrix(n);
            std::vector<float> A_serial = A0;
            std::vector<float> A_gpu = A0;

            auto ts1 = std::chrono::high_resolution_clock::now();
            gauss_serial(n, A_serial.data());
            auto ts2 = std::chrono::high_resolution_clock::now();

            double serial_ms = std::chrono::duration<double, std::milli>(ts2 - ts1).count();
            double gpu_ms = gauss_gpu_basic(n, A_gpu);
            float err = check_error(n, A_gpu.data());

            std::cout << "n=" << n
                      << " trial=" << t
                      << " serial_ms=" << serial_ms
                      << " gpu_ms=" << gpu_ms
                      << " speedup=" << serial_ms / gpu_ms
                      << " err=" << err
                      << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}