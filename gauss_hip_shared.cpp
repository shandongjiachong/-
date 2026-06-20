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

struct GpuResult {
    double total_ms;
    double kernel_ms;
};

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

__global__ void eliminate_shared_kernel(float* A, int n, int k) {
    int width = n + 1;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = k + 1 + blockIdx.y * blockDim.y + ty;
    int col = k + 1 + blockIdx.x * blockDim.x + tx;

    bool valid_row = (row < n);
    bool valid_col = (col <= n);

    extern __shared__ float sdata[];
    float* sPivot = sdata;
    float* sFactor = sdata + blockDim.x;

    if (ty == 0) {
        sPivot[tx] = valid_col ? A[k * width + col] : 0.0f;
    }

    if (tx == 0) {
        sFactor[ty] = valid_row ? A[row * width + k] : 0.0f;
    }

    __syncthreads();

    if (valid_row && valid_col) {
        A[row * width + col] -= sFactor[ty] * sPivot[tx];
    }
}

__global__ void zero_column_kernel(float* A, int n, int k) {
    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        A[row * (n + 1) + k] = 0.0f;
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

void warmup_gpu() {
    float* d = nullptr;
    HIP_CHECK(hipMalloc(&d, 1024 * sizeof(float)));
    HIP_CHECK(hipMemset(d, 0, 1024 * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d));
}

GpuResult gauss_gpu_shared(int n, std::vector<float>& A) {
    int width = n + 1;
    size_t bytes = static_cast<size_t>(n) * width * sizeof(float);
    float* d_A = nullptr;

    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMemcpy(d_A, A.data(), bytes, hipMemcpyHostToDevice));

    hipEvent_t ek1, ek2;
    HIP_CHECK(hipEventCreate(&ek1));
    HIP_CHECK(hipEventCreate(&ek2));
    HIP_CHECK(hipEventRecord(ek1, 0));

    const int norm_threads = 256;
    dim3 block2d(32, 8);

    for (int k = 0; k < n; ++k) {
        hipLaunchKernelGGL(normalize_row_kernel, dim3(1), dim3(norm_threads), 0, 0, d_A, n, k);
        HIP_CHECK(hipGetLastError());

        if (k + 1 < n) {
            int cols = n - k;
            int rows = n - k - 1;

            dim3 grid2d((cols + block2d.x - 1) / block2d.x,
                        (rows + block2d.y - 1) / block2d.y);

            size_t shared_bytes = (block2d.x + block2d.y) * sizeof(float);

            hipLaunchKernelGGL(eliminate_shared_kernel,
                               grid2d,
                               block2d,
                               shared_bytes,
                               0,
                               d_A,
                               n,
                               k);
            HIP_CHECK(hipGetLastError());

            const int zero_threads = 256;
            int zero_blocks = (rows + zero_threads - 1) / zero_threads;

            hipLaunchKernelGGL(zero_column_kernel,
                               dim3(zero_blocks),
                               dim3(zero_threads),
                               0,
                               0,
                               d_A,
                               n,
                               k);
            HIP_CHECK(hipGetLastError());
        }
    }

    HIP_CHECK(hipEventRecord(ek2, 0));
    HIP_CHECK(hipEventSynchronize(ek2));

    float kernel_ms_f = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&kernel_ms_f, ek1, ek2));

    HIP_CHECK(hipMemcpy(A.data(), d_A, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipEventDestroy(ek1));
    HIP_CHECK(hipEventDestroy(ek2));

    back_substitution(n, A.data());

    auto t2 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    return {total_ms, static_cast<double>(kernel_ms_f)};
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

    warmup_gpu();

    std::cout << "===== HIP Gaussian Elimination Shared Test =====" << std::endl;
    std::cout << "device=" << prop.name << std::endl;
    std::cout << "repeat=" << repeat << std::endl;
    std::cout << "format: n trial serial_ms total_ms kernel_ms total_speedup kernel_speedup err" << std::endl;
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
            GpuResult res = gauss_gpu_shared(n, A_gpu);
            float err = check_error(n, A_gpu.data());

            std::cout << "n=" << n
                      << " trial=" << t
                      << " serial_ms=" << serial_ms
                      << " total_ms=" << res.total_ms
                      << " kernel_ms=" << res.kernel_ms
                      << " total_speedup=" << serial_ms / res.total_ms
                      << " kernel_speedup=" << serial_ms / res.kernel_ms
                      << " err=" << err
                      << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}