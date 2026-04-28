#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int n, al, ar, cl, cr;
double X = 0.35;
double U;
double hy;

struct GpuContext {
    bool available = false;
    std::string status_message;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
};

void check_cuda(cudaError_t status, const std::string &message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const std::string &message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": cublas status " + std::to_string(static_cast<int>(status)));
    }
}

void check_cusolver(cusolverStatus_t status, const std::string &message) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(message + ": cusolver status " + std::to_string(static_cast<int>(status)));
    }
}

void destroy_gpu_context(GpuContext *context) {
    if (context->cusolver != nullptr) {
        cusolverDnDestroy(context->cusolver);
        context->cusolver = nullptr;
    }
    if (context->cublas != nullptr) {
        cublasDestroy(context->cublas);
        context->cublas = nullptr;
    }
}

GpuContext init_gpu_context() {
    GpuContext context;

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count <= 0) {
        context.status_message =
            "CUDA недоступна: " +
            std::string(cudaGetErrorString(count_status));
        cudaGetLastError();
        return context;
    }

    try {
        check_cuda(cudaSetDevice(0), "cudaSetDevice failed");
        check_cublas(cublasCreate(&context.cublas), "cublasCreate failed");
        check_cusolver(cusolverDnCreate(&context.cusolver), "cusolverDnCreate failed");

        context.available = true;
        context.status_message =
            "CUDA устройств доступно: " + std::to_string(device_count) +
            ", используется GPU-решатель";
        return context;
    } catch (const std::exception &ex) {
        destroy_gpu_context(&context);
        context.available = false;
        context.status_message =
            "Инициализация CUDA не удалась: " + std::string(ex.what());
        cudaGetLastError();
        return context;
    }
}

void print_matrix(std::vector<double> *vec, std::ofstream *out, int row_a, int col_b) {
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            *out << (*vec)[i * col_b + j] << " ";
        }
        *out << std::endl;
    }
    *out << std::endl;
}

double calc_i_katod(int i, std::vector<double> &fi) {
    return -X * (fi[i * n + n - 1] - fi[i * n + n - 2]) / hy;
}

double calc_Fc(int i, std::vector<double> &fi) {
    const double ic = calc_i_katod(i, fi);
    return 1.347 + 0.0016 * ic * ic + 0.055 * ic;
}

double calc_dFc_dic(int i, std::vector<double> &fi) {
    const double ic = calc_i_katod(i, fi);
    return 0.0032 * ic + 0.055;
}

std::vector<std::vector<double>> make_yakob(std::vector<double> &fi) {
    std::vector<std::vector<double>> yakob(n * n, std::vector<double>(n * n, 0.0));
    int line = 0;

    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++, line++) {
            yakob[line][i * n + j] -= 4.0;
            yakob[line][(i - 1) * n + j] += 1.0;
            yakob[line][(i + 1) * n + j] += 1.0;
            yakob[line][i * n + j - 1] += 1.0;
            yakob[line][i * n + j + 1] += 1.0;
        }
    }
    for (int j = 1; j < n - 1; j++, line++) {
        yakob[line][j] -= 1.0;
        yakob[line][n + j] += 1.0;
    }
    for (int j = 1; j < n - 1; j++, line++) {
        yakob[line][(n - 2) * n + j] -= 1.0;
        yakob[line][(n - 1) * n + j] += 1.0;
    }
    for (int i = 0; i < al - 1; i++, line++) {
        yakob[line][i * n + 1] += 1.0;
        yakob[line][i * n] -= 1.0;
    }
    for (int i = ar; i < n; i++, line++) {
        yakob[line][i * n + 1] += 1.0;
        yakob[line][i * n] -= 1.0;
    }
    for (int i = 0; i < cl - 1; i++, line++) {
        yakob[line][i * n + n - 1] += 1.0;
        yakob[line][i * n + n - 2] -= 1.0;
    }
    for (int i = cr; i < n; i++, line++) {
        yakob[line][i * n + n - 1] += 1.0;
        yakob[line][i * n + n - 2] -= 1.0;
    }
    for (int i = al - 1; i < ar; i++, line++) {
        yakob[line][i * n] += 1.0;
    }
    for (int i = cl - 1; i < cr; i++, line++) {
        const double factor = calc_dFc_dic(i, fi) * X / hy;

        yakob[line][i * n + n - 1] += 1.0 + factor;
        yakob[line][i * n + n - 2] -= factor;
    }

    return yakob;
}

std::vector<double> create_fi() {
    std::vector<double> fi(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fi[i * n + j] = U * static_cast<double>(n - 1 - j) / static_cast<double>(n - 1);
        }
    }
    return fi;
}

double lapl(int i, int j, std::vector<double> &fi) {
    return (fi[(i - 1) * n + j] - 2.0 * fi[i * n + j] + fi[(i + 1) * n + j]) +
           (fi[i * n + j - 1] - 2.0 * fi[i * n + j] + fi[i * n + j + 1]);
}

double regional_1(int j, std::vector<double> &fi) {
    return fi[n + j] - fi[j];
}

double regional_2(int j, std::vector<double> &fi) {
    return fi[(n - 1) * n + j] - fi[(n - 2) * n + j];
}

double regional_3(int i, std::vector<double> &fi) {
    return fi[i * n + 1] - fi[i * n];
}

double regional_4(int i, std::vector<double> &fi) {
    return fi[i * n + n - 1] - fi[i * n + n - 2];
}

double regional_anod(int i, std::vector<double> &fi) {
    return fi[i * n] + 1.2 - U;
}

double regional_katod(int i, std::vector<double> &fi) {
    return fi[i * n + n - 1] - calc_Fc(i, fi);
}

double calc_thickness(int i, std::vector<double> &fi) {
    const double kme = 1.22;
    const double rho = 7.133;
    const double dt = 40.0 / 60.0;
    const double diff = fi[i * n + n - 2] - fi[i * n + n - 1];
    const double j = X * (diff / hy);   // А / дм^2

    return (kme / rho) * j * dt * 100.0;
}

std::vector<double> calc_fi(std::vector<double> &fi) {
    std::vector<double> res;
    res.reserve(n * n);

    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            res.push_back(lapl(i, j, fi));
        }
    }
    for (int j = 1; j < n - 1; j++) {
        res.push_back(regional_1(j, fi));
    }
    for (int j = 1; j < n - 1; j++) {
        res.push_back(regional_2(j, fi));
    }
    for (int i = 0; i < al - 1; i++) {
        res.push_back(regional_3(i, fi));
    }
    for (int i = ar; i < n; i++) {
        res.push_back(regional_3(i, fi));
    }
    for (int i = 0; i < cl - 1; i++) {
        res.push_back(regional_4(i, fi));
    }
    for (int i = cr; i < n; i++) {
        res.push_back(regional_4(i, fi));
    }
    for (int i = al - 1; i < ar; i++) {
        res.push_back(regional_anod(i, fi));
    }
    for (int i = cl - 1; i < cr; i++) {
        res.push_back(regional_katod(i, fi));
    }

    return res;
}

bool is_finite_vector(const std::vector<double> &values) {
    for (double value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

double l1_norm(const std::vector<double> &values) {
    double norm = 0.0;
    for (double value : values) {
        norm += std::abs(value);
    }
    return norm;
}

std::vector<double> to_column_major(const std::vector<std::vector<double>> &matrix) {
    const int size = static_cast<int>(matrix.size());
    std::vector<double> result(size * size, 0.0);

    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            result[row + col * size] = matrix[row][col];
        }
    }

    return result;
}

std::vector<double> solve_linear_system_gpu(const std::vector<std::vector<double>> &matrix,
                                            const std::vector<double> &rhs,
                                            GpuContext *context,
                                            double *device_solution_norm) {
    const int size = static_cast<int>(rhs.size());
    const int lda = size;
    const int ldb = size;
    const int nrhs = 1;

    std::vector<double> matrix_column_major = to_column_major(matrix);
    std::vector<double> solution(rhs);

    double *d_matrix = nullptr;
    double *d_rhs = nullptr;
    double *d_work = nullptr;
    int *d_pivots = nullptr;
    int *d_info = nullptr;
    int lwork = 0;

    try {
        check_cuda(cudaMalloc(&d_matrix, matrix_column_major.size() * sizeof(double)),
                   "cudaMalloc for matrix failed");
        check_cuda(cudaMalloc(&d_rhs, solution.size() * sizeof(double)),
                   "cudaMalloc for rhs failed");
        check_cuda(cudaMemcpy(d_matrix, matrix_column_major.data(),
                              matrix_column_major.size() * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy for matrix failed");
        check_cuda(cudaMemcpy(d_rhs, solution.data(), solution.size() * sizeof(double),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy for rhs failed");

        check_cusolver(cusolverDnDgetrf_bufferSize(context->cusolver, size, size, d_matrix, lda, &lwork),
                       "cusolverDnDgetrf_bufferSize failed");
        check_cuda(cudaMalloc(&d_work, lwork * sizeof(double)),
                   "cudaMalloc for LU workspace failed");
        check_cuda(cudaMalloc(&d_pivots, size * sizeof(int)),
                   "cudaMalloc for pivot array failed");
        check_cuda(cudaMalloc(&d_info, sizeof(int)),
                   "cudaMalloc for info failed");

        check_cusolver(cusolverDnDgetrf(context->cusolver, size, size, d_matrix, lda, d_work, d_pivots, d_info),
                       "cusolverDnDgetrf failed");

        int info = 0;
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy for LU info failed");
        if (info < 0) {
            throw std::runtime_error("LU factorization failed: illegal parameter " + std::to_string(-info));
        }
        if (info > 0) {
            throw std::runtime_error("LU factorization failed: singular pivot at index " + std::to_string(info));
        }

        check_cusolver(cusolverDnDgetrs(context->cusolver, CUBLAS_OP_N, size, nrhs, d_matrix, lda, d_pivots,
                                        d_rhs, ldb, d_info),
                       "cusolverDnDgetrs failed");
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy for solve info failed");
        if (info < 0) {
            throw std::runtime_error("Linear solve failed: illegal parameter " + std::to_string(-info));
        }

        check_cublas(cublasDasum(context->cublas, size, d_rhs, 1, device_solution_norm),
                     "cublasDasum failed");

        check_cuda(cudaMemcpy(solution.data(), d_rhs, solution.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy for solution failed");
    } catch (...) {
        if (d_info != nullptr) cudaFree(d_info);
        if (d_pivots != nullptr) cudaFree(d_pivots);
        if (d_work != nullptr) cudaFree(d_work);
        if (d_rhs != nullptr) cudaFree(d_rhs);
        if (d_matrix != nullptr) cudaFree(d_matrix);
        throw;
    }

    cudaFree(d_info);
    cudaFree(d_pivots);
    cudaFree(d_work);
    cudaFree(d_rhs);
    cudaFree(d_matrix);

    return solution;
}

std::vector<double> solve_linear_system_host(const std::vector<std::vector<double>> &matrix,
                                             const std::vector<double> &rhs,
                                             double *solution_norm) {
    const int size = static_cast<int>(rhs.size());
    std::vector<std::vector<double>> a = matrix;
    std::vector<double> b = rhs;

    for (int col = 0; col < size; col++) {
        int pivot = col;
        for (int row = col + 1; row < size; row++) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }

        if (std::abs(a[pivot][col]) < 1e-14) {
            throw std::runtime_error("Host linear solve failed: singular matrix");
        }

        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }

        for (int row = col + 1; row < size; row++) {
            const double multiplier = a[row][col] / a[col][col];
            a[row][col] = 0.0;
            for (int k = col + 1; k < size; k++) {
                a[row][k] -= multiplier * a[col][k];
            }
            b[row] -= multiplier * b[col];
        }
    }

    std::vector<double> solution(size, 0.0);
    for (int row = size - 1; row >= 0; row--) {
        double sum = b[row];
        for (int col = row + 1; col < size; col++) {
            sum -= a[row][col] * solution[col];
        }
        solution[row] = sum / a[row][row];
    }

    *solution_norm = l1_norm(solution);
    return solution;
}

std::vector<double> solve_linear_system(const std::vector<std::vector<double>> &matrix,
                                        const std::vector<double> &rhs,
                                        GpuContext *context,
                                        double *solution_norm) {
    if (context->available) {
        try {
            return solve_linear_system_gpu(matrix, rhs, context, solution_norm);
        } catch (const std::exception &ex) {
            context->available = false;
            context->status_message =
                "GPU-решатель отключен после ошибки, используется host-реализация: " + std::string(ex.what());
            destroy_gpu_context(context);
        }
    }

    return solve_linear_system_host(matrix, rhs, solution_norm);
}

int main() {
    std::ifstream in("input.txt");
    std::ofstream out("output.txt");

    if (!in.is_open()) {
        std::cerr << "Cannot open input.txt\n";
        return 1;
    }
    if (!out.is_open()) {
        std::cerr << "Cannot open output.txt\n";
        return 1;
    }

    out << std::fixed << std::setprecision(4);

    in >> n >> al >> ar >> cl >> cr >> U;
    if (n < 2) {
        std::cerr << "Grid size must be at least 2\n";
        return 1;
    }
    hy = 0.99 / static_cast<double>(n - 1);

    GpuContext gpu_context = init_gpu_context();
    out << gpu_context.status_message << std::endl << std::endl;

    const auto algorithm_start = std::chrono::steady_clock::now();
    std::vector<double> fi = create_fi();

    out << "Приближенное распределение потенциала:\n";
    print_matrix(&fi, &out, n, n);

    double eps = 1.0;
    const double threshold = 0.001;
    const int max_iterations = 200;
    int iteration = 0;

    try {
        while (eps > threshold && iteration < max_iterations) {
            std::vector<double> residual = calc_fi(fi);
            eps = l1_norm(residual);
            if (eps <= threshold) {
                break;
            }

            std::vector<std::vector<double>> yakob = make_yakob(fi);
            double correction_norm = 0.0;
            std::vector<double> correction =
                solve_linear_system(yakob, residual, &gpu_context, &correction_norm);

            if (!is_finite_vector(correction)) {
                throw std::runtime_error("Potential correction contains NaN or Inf");
            }

            double step = 1.0;
            double next_eps = std::numeric_limits<double>::infinity();
            std::vector<double> next_fi;
            std::vector<double> next_residual;

            while (step >= 1e-6) {
                next_fi = fi;
                for (int i = 0; i < n * n; i++) {
                    next_fi[i] -= step * correction[i];
                }

                if (!is_finite_vector(next_fi)) {
                    step *= 0.5;
                    continue;
                }

                next_residual = calc_fi(next_fi);
                next_eps = l1_norm(next_residual);

                if (std::isfinite(next_eps) && next_eps < eps) {
                    break;
                }

                step *= 0.5;
            }

            if (step < 1e-6) {
                throw std::runtime_error("Newton damping failed to reduce residual");
            }

            fi = std::move(next_fi);
            eps = next_eps;
            iteration++;
            out << "Распределение потенциала на очередной итерации:\n";
            print_matrix(&fi, &out, n, n);
            out << "Норма невязки: " << eps << std::endl;
            out << "Норма коррекции: " << correction_norm << std::endl;
            out << "Демпфирование шага: " << step << std::endl;
        }
    } catch (const std::exception &ex) {
        out << "Ошибка вычислений: " << ex.what() << std::endl;
    }

    const auto algorithm_end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(algorithm_end - algorithm_start).count();

    print_matrix(&fi, &out, n, n);
    out << "Толщина покрытия на катоде:" << std::endl;
    for (int i = cl - 1; i < cr; i++) {
        out << calc_thickness(i, fi) << " ";
    }
    out << std::endl << std::endl;
    out << "Время работы основного алгоритма: " << elapsed_ms << " мс" << std::endl;

    destroy_gpu_context(&gpu_context);
    in.close();
    out.close();

    const int python_status = std::system("python3 main.py");
    if (python_status != 0) {
        std::cerr << "main.py finished with status " << python_status << std::endl;
        return 1;
    }

    return 0;
}
