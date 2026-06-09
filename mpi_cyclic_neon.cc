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

static inline int rows_of_rank_cyclic(int rank, int n, int size) {
    if (rank >= n) return 0;
    return (n - 1 - rank) / size + 1;
}

static inline int owner_of_row_cyclic(int row, int size) {
    return row % size;
}

static inline int local_index_of_row_cyclic(int row, int size) {
    return row / size;
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

vector<float> pack_cyclic(const vector<float>& A, int n, int size) {
    int width = n + 1;
    vector<float> packed;
    packed.reserve(n * width);
    for (int rank = 0; rank < size; ++rank) {
        for (int row = rank; row < n; row += size) {
            for (int j = 0; j < width; ++j) {
                packed.push_back(A[row * width + j]);
            }
        }
    }
    return packed;
}

void unpack_cyclic(const vector<float>& packed, vector<float>& A, int n, int size) {
    int width = n + 1;
    int pos = 0;
    for (int rank = 0; rank < size; ++rank) {
        for (int row = rank; row < n; row += size) {
            for (int j = 0; j < width; ++j) {
                A[row * width + j] = packed[pos++];
            }
        }
    }
}

void gauss_mpi_cyclic(int n, vector<float>& A_global, int rank, int size) {
    int width = n + 1;
    int local_rows = rows_of_rank_cyclic(rank, n, size);

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rows_of_rank_cyclic(r, n, size) * width;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    vector<float> packed_A;
    if (rank == 0) {
        packed_A = pack_cyclic(A_global, n, size);
    }

    vector<float> local_A(local_rows * width, 0.0f);

    MPI_Scatterv(rank == 0 ? packed_A.data() : nullptr,
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
        int owner = owner_of_row_cyclic(k, size);

        if (rank == owner) {
            int lk = local_index_of_row_cyclic(k, size);
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
            int global_i = rank + i * size;
            if (global_i > k) {
                float factor = local_A[i * width + k];
                for (int j = k + 1; j <= n; ++j) {
                    local_A[i * width + j] -= factor * pivot_row[j];
                }
                local_A[i * width + k] = 0.0f;
            }
        }
    }

    vector<float> gathered_A;
    if (rank == 0) {
        gathered_A.resize(n * width);
    }

    MPI_Gatherv(local_A.data(),
                local_rows * width,
                MPI_FLOAT,
                rank == 0 ? gathered_A.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        unpack_cyclic(gathered_A, A_global, n, size);

        for (int i = n - 1; i >= 0; --i) {
            float sum = A_global[i * width + n];
            for (int j = i + 1; j < n; ++j) {
                sum -= A_global[i * width + j] * A_global[j * width + n];
            }
            A_global[i * width + n] = sum;
        }
    }
}

void gauss_mpi_cyclic_neon(int n, vector<float>& A_global, int rank, int size) {
    int width = n + 1;
    int local_rows = rows_of_rank_cyclic(rank, n, size);

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rows_of_rank_cyclic(r, n, size) * width;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    vector<float> packed_A;
    if (rank == 0) {
        packed_A = pack_cyclic(A_global, n, size);
    }

    vector<float> local_A(local_rows * width, 0.0f);

    MPI_Scatterv(rank == 0 ? packed_A.data() : nullptr,
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
        int owner = owner_of_row_cyclic(k, size);

        if (rank == owner) {
            int lk = local_index_of_row_cyclic(k, size);
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
            int global_i = rank + i * size;
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

    vector<float> gathered_A;
    if (rank == 0) {
        gathered_A.resize(n * width);
    }

    MPI_Gatherv(local_A.data(),
                local_rows * width,
                MPI_FLOAT,
                rank == 0 ? gathered_A.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        unpack_cyclic(gathered_A, A_global, n, size);

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
            vector<float> A_cyclic;
            vector<float> A_neon;

            double serial_ms = 0.0;

            if (rank == 0) {
                A0 = generate_matrix(n);
                A_serial = A0;
                A_cyclic = A0;
                A_neon = A0;

                auto ts1 = chrono::high_resolution_clock::now();
                gauss_serial(n, A_serial);
                auto ts2 = chrono::high_resolution_clock::now();
                serial_ms = chrono::duration<double, milli>(ts2 - ts1).count();
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double tc1 = MPI_Wtime();
            gauss_mpi_cyclic(n, A_cyclic, rank, size);
            double tc2 = MPI_Wtime();

            MPI_Barrier(MPI_COMM_WORLD);
            double tn1 = MPI_Wtime();
            gauss_mpi_cyclic_neon(n, A_neon, rank, size);
            double tn2 = MPI_Wtime();

            if (rank == 0) {
                double cyclic_ms = (tc2 - tc1) * 1000.0;
                double neon_ms = (tn2 - tn1) * 1000.0;
                double cyclic_speedup = serial_ms / cyclic_ms;
                double neon_speedup = serial_ms / neon_ms;
                double neon_gain = cyclic_ms / neon_ms;

                float err_cyclic = check_error(n, A_cyclic);
                float err_neon = check_error(n, A_neon);

                cout << "n=" << n
                     << " trial=" << t
                     << " serial_ms=" << serial_ms
                     << " cyclic_ms=" << cyclic_ms
                     << " cyclic_speedup=" << cyclic_speedup
                     << " neon_ms=" << neon_ms
                     << " neon_speedup=" << neon_speedup
                     << " neon_gain=" << neon_gain
                     << " cyclic_err=" << err_cyclic
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