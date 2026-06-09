#include <mpi.h>
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
        cout << "format: n trial serial_ms mpi_ms speedup efficiency err" << endl;
        cout << fixed << setprecision(3);
    }

    for (int s = 0; s < num_sizes; ++s) {
        int n = test_sizes[s];

        for (int t = 1; t <= repeat; ++t) {
            vector<float> A0;
            vector<float> A_serial;
            vector<float> A_mpi;

            double serial_ms = 0.0;
            float serial_err = 0.0f;

            if (rank == 0) {
                A0 = generate_matrix(n);
                A_serial = A0;
                A_mpi = A0;

                auto ts1 = chrono::high_resolution_clock::now();
                gauss_serial(n, A_serial);
                auto ts2 = chrono::high_resolution_clock::now();
                serial_ms = chrono::duration<double, milli>(ts2 - ts1).count();
                serial_err = check_error(n, A_serial);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            double tm1 = MPI_Wtime();
            gauss_mpi_block(n, A_mpi, rank, size);
            double tm2 = MPI_Wtime();

            if (rank == 0) {
                double mpi_ms = (tm2 - tm1) * 1000.0;
                float mpi_err = check_error(n, A_mpi);
                double speedup = serial_ms / mpi_ms;
                double efficiency = speedup / size;

                cout << "n=" << n
                     << " trial=" << t
                     << " serial_ms=" << serial_ms
                     << " mpi_ms=" << mpi_ms
                     << " speedup=" << speedup
                     << " efficiency=" << efficiency
                     << " err=" << mpi_err
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