#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>

#include "arith/assign.cuh"
#include "arith/add.cuh"
#include "arith/sub.cuh"
#include "arith/mul.cuh"
#include "arith/muld.cuh"
#include "arith/div.cuh"
#include "arith/cmp.cuh"

#include "mparray.cuh"
#include "blas/gemm.cuh"
#include "blas/mblas_enum.cuh"

mp_float_t width;
int N, al, ar, cl, cr;
mp_float_t X, U, hx, hy;

static inline mp_float_t mp_from_d(double v) {
    mp_float_t t;
    mp_set_d(&t, v);
    return t;
}

static inline double mp_to_d(const mp_float_t &v) {
    return mp_get_d(v);
}

static inline void mp_iadd(mp_float_t *a, const mp_float_t &b) {
    mp_add(a, *a, b);
}

static inline void mp_isub(mp_float_t *a, const mp_float_t &b) {
    mp_sub(a, *a, b);
}

static inline void mp_imul(mp_float_t *a, const mp_float_t &b) {
    mp_float_t r;
    mp_mul(&r, *a, b);
    *a = r;
}

static inline void mp_imul_d(mp_float_t *a, double b) {
    mp_mul_d(a, *a, b);
}

static inline void mp_idiv(mp_float_t *a, const mp_float_t &b) {
    mp_float_t r;
    mp_float_t aa = *a;
    mp_div(&r, &aa, &b);
    *a = r;
}

static inline mp_float_t mp_abs_hp(const mp_float_t &a) {
    if (mp_to_d(a) < 0.0) {
        mp_float_t r;
        mp_mul_d(&r, a, -1.0);
        return r;
    }
    return a;
}

static mp_float_t mp_exp_hp(const mp_float_t &x) {
    double xd = mp_to_d(x);
    mp_float_t xr = x;
    int m = 0;
    while (std::abs(xd) > 0.5 && m < 24) {
        mp_mul_d(&xr, xr, 0.5);
        xd *= 0.5;
        m++;
    }

    mp_float_t sum, term;
    mp_set_d(&sum, 1.0);
    mp_set_d(&term, 1.0);

    for (int k = 1; k <= 60; k++) {
        mp_float_t tmp;
        mp_mul(&tmp, term, xr);
        mp_mul_d(&term, tmp, 1.0 / (double)k);
        mp_add(&sum, sum, term);
        if (std::abs(mp_to_d(term)) < 1e-25) break;
    }

    for (int i = 0; i < m; i++) {
        mp_float_t tmp;
        mp_mul(&tmp, sum, sum);
        sum = tmp;
    }
    return sum;
}

static mp_float_t ch_hp(const mp_float_t &x) {
    mp_float_t nx;
    mp_mul_d(&nx, x, -1.0);
    mp_float_t ex = mp_exp_hp(x);
    mp_float_t enx = mp_exp_hp(nx);
    mp_float_t s;
    mp_add(&s, ex, enx);
    mp_mul_d(&s, s, 0.5);
    return s;
}

static mp_float_t th_hp(const mp_float_t &x) {
    mp_float_t two_x;
    mp_mul_d(&two_x, x, 2.0);
    mp_float_t e = mp_exp_hp(two_x);

    mp_float_t one = mp_from_d(1.0);
    mp_float_t num, den, res;

    mp_sub(&num, e, one);
    mp_add(&den, e, one);

    mp_float_t nnum = num;
    mp_float_t dden = den;
    mp_div(&res, &nnum, &dden);
    return res;
}

void vector_copy(std::vector<mp_float_t> *vec_1, std::vector<mp_float_t> *vec_2) {
    (*vec_2).clear();
    for (auto it : *vec_1) (*vec_2).push_back(it);
}

void vector_copy(std::vector<mp_float_t> vec_1, std::vector<mp_float_t> *vec_2) {
    (*vec_2).clear();
    for (auto it : vec_1) (*vec_2).push_back(it);
}

mp_float_t calc_i_katod(int i, std::vector<mp_float_t> *fi) {
    mp_float_t a = (*fi)[i * N + (N - 1)];
    mp_float_t b = (*fi)[i * N + (N - 2)];
    mp_float_t diff;
    mp_sub(&diff, a, b);

    mp_float_t frac = diff;
    mp_idiv(&frac, hy);

    mp_float_t prod;
    mp_mul(&prod, X, frac);

    mp_float_t res;
    mp_mul_d(&res, prod, -1.0);
    return res;
}

static void matrixMultiplyCuBLAS(mp_array_t A_d, mp_array_t B_d, mp_array_t C_d,
                                 int M, int K, int P,
                                 mp_array_t alpha_d, mp_array_t beta_d, mp_array_t buffer_d) {
    constexpr int BLOCK1X = 16;
    constexpr int BLOCK1Y = 16;
    constexpr int GRID2X = 64;
    constexpr int GRID2Y = 64;
    constexpr int BLOCK3 = 16;

    cuda::mp_gemm<BLOCK1X, BLOCK1Y, GRID2X, GRID2Y, BLOCK3>(
        mblas_no_trans, mblas_no_trans,
        M, P, K,
        alpha_d, A_d, M,
        B_d, K,
        beta_d, C_d, M,
        buffer_d
    );
}

static void to_mp_column_major(const std::vector<mp_float_t> &rm, int rows, int cols, std::vector<mp_float_t> &cm) {
    cm.resize(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cm[i + j * rows] = rm[i * cols + j];
        }
    }
}

static void from_mp_column_major_to_row_major(const std::vector<mp_float_t> &cm, int rows, int cols, std::vector<mp_float_t> &rm) {
    rm.resize(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            rm[i * cols + j] = cm[i + j * rows];
        }
    }
}

void mul(std::vector<mp_float_t> *h_A, std::vector<mp_float_t> *h_B,
         const int row, const int col_row, const int col) {
    std::vector<mp_float_t> A_cm, B_cm, C_cm;
    to_mp_column_major(*h_A, row, col_row, A_cm);
    to_mp_column_major(*h_B, col_row, col, B_cm);

    C_cm.assign(row * col, mp_from_d(0.0));

    mp_float_t alpha_h = mp_from_d(1.0);
    mp_float_t beta_h  = mp_from_d(0.0);

    mp_array_t A_d, B_d, C_d, alpha_d, beta_d, buffer_d;
    cuda::mp_array_init(A_d, row * col_row);
    cuda::mp_array_init(B_d, col_row * col);
    cuda::mp_array_init(C_d, row * col);
    cuda::mp_array_init(alpha_d, 1);
    cuda::mp_array_init(beta_d, 1);
    cuda::mp_array_init(buffer_d, row * col);

    cuda::mp_array_host2device(A_d, A_cm.data(), row * col_row);
    cuda::mp_array_host2device(B_d, B_cm.data(), col_row * col);
    cuda::mp_array_host2device(C_d, C_cm.data(), row * col);
    cuda::mp_array_host2device(alpha_d, &alpha_h, 1);
    cuda::mp_array_host2device(beta_d, &beta_h, 1);

    matrixMultiplyCuBLAS(A_d, B_d, C_d, row, col_row, col, alpha_d, beta_d, buffer_d);
    cudaDeviceSynchronize();

    cuda::mp_array_device2host(C_cm.data(), C_d, row * col);

    std::vector<mp_float_t> C_rm;
    from_mp_column_major_to_row_major(C_cm, row, col, C_rm);

    for (int i = 0; i < row * col; i++) {
        (*h_B)[i] = C_rm[i];
    }
}

void print_matrix(std::vector<mp_float_t> *vec, std::ofstream *out, int row_a, int col_b) {
    (*out) << std::setprecision(17);
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            (*out) << mp_to_d((*vec)[i * col_b + j]) << " ";
        }
        (*out) << std::endl;
    }
    (*out) << std::endl;
}

mp_float_t get_sp(std::vector<mp_float_t> *vec, int n) {
    mp_float_t res = mp_from_d(0.0);
    for (int i = 0; i < n; i++) {
        mp_iadd(&res, (*vec)[i * n + i]);
    }
    return res;
}

mp_float_t get_p(std::vector<mp_float_t> *p, std::vector<mp_float_t> *sp, int n, int idx) {
    mp_float_t res = (*sp)[idx];
    for (int i = 1; i < idx; i++) {
        mp_float_t tmp;
        mp_mul(&tmp, (*p)[i], (*sp)[idx - i]);
        mp_isub(&res, tmp);
    }
    mp_mul_d(&res, res, 1.0 / (double)idx);
    return res;
}

void koef_step(std::vector<mp_float_t> *v) {
    for (auto &it : (*v)) {
        mp_mul_d(&it, it, 0.5);
    }
}

std::vector<mp_float_t> get_obr(std::vector<mp_float_t> h_A, int row, int col) {
    std::vector<std::vector<mp_float_t>> list_matrix(row + 1, h_A);

    for (int i = 0; i < row * row; i++) mp_set_d(&list_matrix[0][i], 0.0);
    for (int i = 0; i < row; i++) mp_set_d(&list_matrix[0][i * row + i], 1.0);

    std::vector<mp_float_t> sp;
    sp.push_back(mp_from_d(0.0));
    sp.push_back(get_sp(&h_A, row));

    std::vector<mp_float_t> p;
    p.push_back(mp_from_d(0.0));
    p.push_back(get_p(&p, &sp, row, 1));

    vector_copy(&h_A, &list_matrix[1]);

    for (int i = 2; i <= row; i++) {
        mul(&list_matrix[i - 1], &list_matrix[i], row, col, col);
        koef_step(&list_matrix[i]);
        sp.push_back(get_sp(&list_matrix[i], row));
        p.push_back(get_p(&p, &sp, row, i));
    }

    vector_copy(&list_matrix[row - 1], &h_A);

    for (int i = 1; i < row; i++) {
        for (int j = 0; j < (int)list_matrix[row].size(); j++) {
            mp_float_t tmp;
            mp_mul(&tmp, p[i], list_matrix[row - i - 1][j]);
            mp_isub(&h_A[j], tmp);
        }
    }

    mp_float_t denom = p[row];
    for (int i = 0; i < row * row; i++) {
        mp_float_t tmp = h_A[i];
        mp_div(&h_A[i], &tmp, &denom);
    }

    std::cout << mp_to_d(p[row]) << std::endl;
    return h_A;
}

std::vector<std::vector<mp_float_t>> make_yakob(std::vector<mp_float_t> *fi) {
    std::vector<std::vector<mp_float_t>> yakob(N * N, std::vector<mp_float_t>(N * N));
    for (int i = 0; i < N * N; i++) {
        for (int j = 0; j < N * N; j++) {
            mp_set_d(&yakob[i][j], 0.0);
        }
    }

    int line = 0;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++, line++) {
            mp_iadd(&yakob[line][i * N + j], mp_from_d(-4.0));
            mp_iadd(&yakob[line][(i - 1) * N + j], mp_from_d(1.0));
            mp_iadd(&yakob[line][(i + 1) * N + j], mp_from_d(1.0));
            mp_iadd(&yakob[line][i * N + j - 1], mp_from_d(1.0));
            mp_iadd(&yakob[line][i * N + j + 1], mp_from_d(1.0));
        }
    }
    for (int j = 1; j < N - 1; j++, line++) {
        mp_iadd(&yakob[line][j], mp_from_d(-1.0));
        mp_iadd(&yakob[line][N + j], mp_from_d(1.0));
    }
    for (int j = 1; j < N - 1; j++, line++) {
        mp_iadd(&yakob[line][(N - 2) * N + j], mp_from_d(-1.0));
        mp_iadd(&yakob[line][(N - 1) * N + j], mp_from_d(1.0));
    }
    for (int i = 0; i < al - 1; i++, line++) {
        mp_iadd(&yakob[line][i * N + 1], mp_from_d(1.0));
        mp_iadd(&yakob[line][i * N], mp_from_d(-1.0));
    }
    for (int i = ar; i < N; i++, line++) {
        mp_iadd(&yakob[line][i * N + 1], mp_from_d(1.0));
        mp_iadd(&yakob[line][i * N], mp_from_d(-1.0));
    }
    for (int i = 0; i < cl - 1; i++, line++) {
        mp_iadd(&yakob[line][i * N + N - 1], mp_from_d(1.0));
        mp_iadd(&yakob[line][i * N + N - 2], mp_from_d(-1.0));
    }
    for (int i = cr; i < N; i++, line++) {
        mp_iadd(&yakob[line][i * N + N - 1], mp_from_d(1.0));
        mp_iadd(&yakob[line][i * N + N - 2], mp_from_d(-1.0));
    }
    for (int i = al - 1; i < ar; i++, line++) {
        mp_iadd(&yakob[line][i * N], mp_from_d(1.0));
    }
    for (int i = cl - 1; i < cr; i++, line++) {
        mp_float_t ic = calc_i_katod(i, fi);

        mp_float_t ic2;
        mp_mul(&ic2, ic, ic);

        mp_float_t t;
        mp_mul_d(&t, ic2, 0.0016);

        mp_float_t c = ch_hp(t);
        mp_float_t c2;
        mp_mul(&c2, c, c);

        mp_float_t one = mp_from_d(1.0);
        mp_float_t inv_c2;
        mp_float_t c2c = c2;
        mp_div(&inv_c2, &one, &c2c);

        mp_float_t factor = inv_c2;
        mp_imul_d(&factor, 0.0032);
        mp_imul(&factor, ic);
        mp_imul(&factor, X);
        mp_idiv(&factor, hy);

        mp_float_t add_to;
        mp_add(&add_to, one, factor);

        mp_iadd(&yakob[line][i * N + N - 1], add_to);
        mp_isub(&yakob[line][i * N + N - 2], factor);
    }

    return yakob;
}

std::vector<mp_float_t> create_fi() {
    std::vector<mp_float_t> fi(N * N);
    mp_float_t halfU;
    mp_mul_d(&halfU, U, 0.5);
    for (int i = 0; i < N * N; i++) fi[i] = halfU;
    return fi;
}

mp_float_t lapl(int i, int j, std::vector<mp_float_t> *fi) {
    mp_float_t a1 = (*fi)[(i - 1) * N + j];
    mp_float_t a2 = (*fi)[i * N + j];
    mp_float_t a3 = (*fi)[(i + 1) * N + j];

    mp_float_t b1 = (*fi)[i * N + j - 1];
    mp_float_t b2 = (*fi)[i * N + j];
    mp_float_t b3 = (*fi)[i * N + j + 1];

    mp_float_t twoa2, twob2;
    mp_mul_d(&twoa2, a2, 2.0);
    mp_mul_d(&twob2, b2, 2.0);

    mp_float_t part1, part2, res;
    mp_sub(&part1, a1, twoa2);
    mp_iadd(&part1, a3);

    mp_sub(&part2, b1, twob2);
    mp_iadd(&part2, b3);

    mp_add(&res, part1, part2);
    return res;
}

mp_float_t regional_1(int j, std::vector<mp_float_t> *fi) {
    mp_float_t res;
    mp_sub(&res, (*fi)[N + j], (*fi)[j]);
    return res;
}

mp_float_t regional_2(int j, std::vector<mp_float_t> *fi) {
    mp_float_t res;
    mp_sub(&res, (*fi)[(N - 1) * N + j], (*fi)[(N - 2) * N + j]);
    return res;
}

mp_float_t regional_3(int i, std::vector<mp_float_t> *fi) {
    mp_float_t res;
    mp_sub(&res, (*fi)[i * N + 1], (*fi)[i * N]);
    return res;
}

mp_float_t regional_4(int i, std::vector<mp_float_t> *fi) {
    mp_float_t res;
    mp_sub(&res, (*fi)[i * N + N - 1], (*fi)[i * N + N - 2]);
    return res;
}

mp_float_t regional_anod(int i, std::vector<mp_float_t> *fi) {
    mp_float_t res;
    mp_add(&res, (*fi)[i * N], mp_from_d(1.2));
    mp_isub(&res, U);
    return res;
}

mp_float_t calc_Fc(int i, std::vector<mp_float_t> *fi) {
    mp_float_t ic = calc_i_katod(i, fi);
    mp_float_t ic2;
    mp_mul(&ic2, ic, ic);

    mp_float_t t;
    mp_mul_d(&t, ic2, 0.0016);
    mp_iadd(&t, mp_from_d(0.12));

    mp_float_t thv = th_hp(t);
    mp_float_t res;
    mp_add(&res, mp_from_d(1.347), thv);
    return res;
}

mp_float_t regional_katod(int i, std::vector<mp_float_t> *fi) {
    mp_float_t Fc = calc_Fc(i, fi);
    mp_float_t res;
    mp_sub(&res, (*fi)[i * N + N - 1], Fc);
    return res;
}

std::vector<mp_float_t> calc_fi(std::vector<mp_float_t> *fi) {
    std::vector<mp_float_t> res;
    res.reserve(N * N);

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            res.push_back(lapl(i, j, fi));
        }
    }
    for (int j = 1; j < N - 1; j++) res.push_back(regional_1(j, fi));
    for (int j = 1; j < N - 1; j++) res.push_back(regional_2(j, fi));
    for (int i = 0; i < al - 1; i++) res.push_back(regional_3(i, fi));
    for (int i = ar; i < N; i++) res.push_back(regional_3(i, fi));
    for (int i = 0; i < cl - 1; i++) res.push_back(regional_4(i, fi));
    for (int i = cr; i < N; i++) res.push_back(regional_4(i, fi));
    for (int i = al - 1; i < ar; i++) res.push_back(regional_anod(i, fi));
    for (int i = cl - 1; i < cr; i++) res.push_back(regional_katod(i, fi));

    return res;
}

mp_float_t calc_thickness(int i, std::vector<mp_float_t> *fi) {
    mp_float_t kme = mp_from_d(1.22);
    mp_float_t dn  = mp_from_d(7.133);
    mp_float_t dt  = mp_from_d(420.0);

    mp_float_t diff;
    mp_sub(&diff, (*fi)[i * N + N - 2], (*fi)[i * N + N - 1]);

    mp_float_t frac = diff;
    mp_idiv(&frac, hy);

    mp_float_t res;
    mp_mul(&res, kme, X);
    mp_idiv(&res, dn);
    mp_imul(&res, frac);
    mp_imul(&res, dt);
    return res;
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

    rns_const_init();
    mp_const_init();

    double Ud;
    in >> N >> al >> ar >> cl >> cr >> Ud;

    mp_set_d(&width, 0.99);
    mp_set_d(&X, 0.35);
    mp_set_d(&U, Ud);

    int start_time = clock();

    mp_set_d(&hx, 1.0);
    mp_set_d(&hy, 9.9 / (double)N);

    out << "Размер сетки: " << N << " X " << N << std::endl;
    out << "Координаты анода: " << al << " " << ar << std::endl;
    out << "Координаты катода: " << cl << " " << cr << std::endl;
    out << "Напряжение на аноде: " << mp_to_d(U) << std::endl;
    out << "Шаг hx и hy: " << mp_to_d(hx) << " " << mp_to_d(hy) << std::endl << std::endl;

    std::vector<mp_float_t> fi = create_fi();

    out << "Приближенное распределение потенциала:\n";
    print_matrix(&fi, &out, N, N);
    out << std::endl;

    mp_float_t eps = mp_from_d(1.0);
    mp_float_t thr = mp_from_d(0.001);

    while (mp_cmp(eps, thr) > 0) {
        std::vector<std::vector<mp_float_t>> yakob = make_yakob(&fi);

        std::vector<mp_float_t> yakob_in_line;
        yakob_in_line.reserve(N * N * N * N);
        for (int i = 0; i < N * N; i++)
            for (int j = 0; j < N * N; j++)
                yakob_in_line.push_back(yakob[i][j]);

        yakob_in_line = get_obr(yakob_in_line, N * N, N * N);
        std::vector<mp_float_t> fi_ = calc_fi(&fi);

        out << "Ошибка потенциала на очередной итерации до умножения на якобиан:\n";
        print_matrix(&fi_, &out, N, N);

        mul(&yakob_in_line, &fi_, N * N, N * N, 1);

        out << "Ошибка потенциала на очередной итерации после умножения на якобиан:\n";
        print_matrix(&fi_, &out, N, N);

        eps = mp_from_d(0.0);
        for (int i = 0; i < N * N; i++) {
            mp_float_t tmp;
            mp_sub(&tmp, fi[i], fi_[i]);
            fi[i] = tmp;

            mp_float_t a = mp_abs_hp(fi_[i]);
            mp_iadd(&eps, a);
        }

        out << "Распределение потенциала на очередной итерации:\n";
        print_matrix(&fi, &out, N, N);
    }

    return 0;
}