#include <mpi.h>
#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace std;

static inline int owner_of_row(int row, int n, int size) {
    int base = n / size;
    int rem = n % size;
    if (row < (base + 1) * rem) {
        return row / (base + 1);
    }
    return rem + (row - (base + 1) * rem) / base;
}

static inline int local_index_of_row(int row, int n, int size) {
    int base = n / size;
    int rem = n % size;
    int owner = owner_of_row(row, n, size);
    int start;
    if (owner < rem) {
        start = owner * (base + 1);
    } else {
        start = rem * (base + 1) + (owner - rem) * base;
    }
    return row - start;
}

static inline int start_row_of_rank(int rank, int n, int size) {
    int base = n / size;
    int rem = n % size;
    if (rank < rem) return rank * (base + 1);
    return rem * (base + 1) + (rank - rem) * base;
}

static inline int rows_of_rank(int rank, int n, int size) {
    int base = n / size;
    int rem = n % size;
    return rank < rem ? (base + 1) : base;
}

vector<float> generate_matrix(int n) {
    vector<float> A(n * (n + 1), 0.0f);
    srand(0);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                float val = static_cast<float>(rand() % 100) / 100.0f;
                A[i * (n + 1) + j] = val;
                sum += fabs(val);
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

void gauss_serial(int n, vector<float>& A) {
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

void gauss_mpi_block(int n, vector<float>& A_global, int rank, int size) {
    int local_rows = rows_of_rank(rank, n, size);
    int start_row = start_row_of_rank(rank, n, size);
    int width = n + 1;

    vector<float> local_A(local_rows * width, 0.0f);

    vector<int> sendcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rows_of_rank(r, n, size) * width;
        displs[r] = start_row_of_rank(r, n, size) * width;
    }

    MPI_Scatterv(rank == 0 ? A_global.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_FLOAT,
                 local_A.data(),
                 local_rows * width,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    vector<float> pivot_row(width, 0.0f);

    for (int k = 0; k < n; ++k) {
        int owner = owner_of_row(k, n, size);

        if (rank == owner) {
            int lk = local_index_of_row(k, n, size);
            float pivot = local_A[lk * width + k];
            for (int j = k + 1; j <= n; ++j) {
                local_A[lk * width + j] /= pivot;
            }
            local_A[lk * width + k] = 1.0f;

            for (int j = 0; j <= n; ++j) {
                pivot_row[j] = local_A[lk * width + j];
            }
        }

        MPI_Bcast(pivot_row.data(), width, MPI_FLOAT, owner, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; ++i) {
            int global_i = start_row + i;
            if (global_i > k) {
                float factor = local_A[i * width + k];
                for (int j = k + 1; j <= n; ++j) {
                    local_A[i * width + j] -= factor * pivot_row[j];
                }
                local_A[i * width + k] = 0.0f;
            }
        }
    }

    MPI_Gatherv(local_A.data(),
                local_rows * width,
                MPI_FLOAT,
                rank == 0 ? A_global.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = n - 1; i >= 0; --i) {
            float sum = A_global[i * width + n];
            for (int j = i + 1; j < n; ++j) {
                sum -= A_global[i * width + j] * A_global[j * width + n];
            }
            A_global[i * width + n] = sum;
        }
    }
}

void gauss_mpi_block_neon(int n, vector<float>& A_global, int rank, int size) {
    int local_rows = rows_of_rank(rank, n, size);
    int start_row = start_row_of_rank(rank, n, size);
    int width = n + 1;

    vector<float> local_A(local_rows * width, 0.0f);

    vector<int> sendcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rows_of_rank(r, n, size) * width;
        displs[r] = start_row_of_rank(r, n, size) * width;
    }

    MPI_Scatterv(rank == 0 ? A_global.data() : nullptr,
                 sendcounts.data(),
                 displs.data(),
                 MPI_FLOAT,
                 local_A.data(),
                 local_rows * width,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    vector<float> pivot_row(width, 0.0f);

    for (int k = 0; k < n; ++k) {
        int owner = owner_of_row(k, n, size);

        if (rank == owner) {
            int lk = local_index_of_row(k, n, size);
            float pivot = local_A[lk * width + k];
            float32x4_t vpivot = vdupq_n_f32(pivot);

            int j = k + 1;
            for (; j + 4 <= n + 1; j += 4) {
                float32x4_t vrow = vld1q_f32(&local_A[lk * width + j]);
                vrow = vdivq_f32(vrow, vpivot);
                vst1q_f32(&local_A[lk * width + j], vrow);
            }
            for (; j <= n; ++j) {
                local_A[lk * width + j] /= pivot;
            }
            local_A[lk * width + k] = 1.0f;

            for (int t = 0; t <= n; ++t) {
                pivot_row[t] = local_A[lk * width + t];
            }
        }

        MPI_Bcast(pivot_row.data(), width, MPI_FLOAT, owner, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; ++i) {
            int global_i = start_row + i;
            if (global_i > k) {
                float factor = local_A[i * width + k];
                float32x4_t vfactor = vdupq_n_f32(factor);

                int j = k + 1;
                for (; j + 4 <= n + 1; j += 4) {
                    float32x4_t vi = vld1q_f32(&local_A[i * width + j]);
                    float32x4_t vk = vld1q_f32(&pivot_row[j]);
                    vi = vsubq_f32(vi, vmulq_f32(vfactor, vk));
                    vst1q_f32(&local_A[i * width + j], vi);
                }
                for (; j <= n; ++j) {
                    local_A[i * width + j] -= factor * pivot_row[j];
                }
                local_A[i * width + k] = 0.0f;
            }
        }
    }

    MPI_Gatherv(local_A.data(),
                local_rows * width,
                MPI_FLOAT,
                rank == 0 ? A_global.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = n - 1; i >= 0; --i) {
            float sum = A_global[i * width + n];
            for (int j = i + 1; j < n; ++j) {
                sum -= A_global[i * width + j] * A_global[j * width + n];
            }
            A_global[i * width + n] = sum;
        }
    }
}

float check_error(int n, const vector<float>& A) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = i + 1.0f;
        float computed = A[i * (n + 1) + n];
        float err = fabs(expected - computed);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int test_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int repeat = 3;

    if (rank == 0) {
        cout << "processes=" << size << endl;
        cout << "repeat=" << repeat << endl;
        cout << fixed << setprecision(3);
    }

    for (int s = 0; s < num_sizes; ++s) {
        int n = test_sizes[s];

        for (int t = 1; t <= repeat; ++t) {
            vector<float> A0;
            vector<float> A_serial;
            vector<float> A_block;
            vector<float> A_neon;

            double serial_ms = 0.0;

            if (rank == 0) {
                A0 = generate_matrix(n);
                A_serial = A0;
                A_block = A0;
                A_neon = A0;

                auto ts1 = chrono::high_resolution_clock::now();
                gauss_serial(n, A_serial);
                auto ts2 = chrono::high_resolution_clock::now();
                serial_ms = chrono::duration<double, milli>(ts2 - ts1).count();
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double tb1 = MPI_Wtime();
            gauss_mpi_block(n, A_block, rank, size);
            double tb2 = MPI_Wtime();

            MPI_Barrier(MPI_COMM_WORLD);
            double tn1 = MPI_Wtime();
            gauss_mpi_block_neon(n, A_neon, rank, size);
            double tn2 = MPI_Wtime();

            if (rank == 0) {
                double block_ms = (tb2 - tb1) * 1000.0;
                double neon_ms = (tn2 - tn1) * 1000.0;
                double speedup_block = serial_ms / block_ms;
                double speedup_neon = serial_ms / neon_ms;
                double neon_gain = block_ms / neon_ms;

                float err_block = check_error(n, A_block);
                float err_neon = check_error(n, A_neon);

                cout << "n=" << n
                     << " trial=" << t
                     << " serial_ms=" << serial_ms
                     << " block_ms=" << block_ms
                     << " block_speedup=" << speedup_block
                     << " neon_ms=" << neon_ms
                     << " neon_speedup=" << speedup_neon
                     << " neon_gain=" << neon_gain
                     << " block_err=" << err_block
                     << " neon_err=" << err_neon
                     << endl;
            }
        }

        if (rank == 0) {
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}