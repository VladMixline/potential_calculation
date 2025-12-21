#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>


std::ofstream out("output.txt");


void vector_copy(std::vector < double > *vec_1,
    std:: vector < double > *vec_2){
    (*vec_2).clear();
    for(auto it:*vec_1){
        (*vec_2).push_back(it);
    }
}

void vector_copy(std::vector < double > vec_1,
    std:: vector < double > *vec_2){
    (*vec_2).clear();
    for(auto it:vec_1){
        (*vec_2).push_back(it);
    }
}

// Умножение матриц с использованием cuBLAS
void matrixMultiplyCuBLAS(double* A, double* B, double* C, int M, int N, int P) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const double alpha = 1.0f;  // Коэффициент для матрицы A*B
    const double beta = 0.0f;   // Коэффициент для матрицы C (обнуляем существующее значение)
    
    cublasDgemm_v2(handle, 
                CUBLAS_OP_N,      // Без транспонирования матрицы A
                CUBLAS_OP_N,      // Без транспонирования матрицы B
                P, M, N,          // Размеры для column-major порядка
                &alpha, 
                B, P,             // Матрица B и ее leading dimension
                A, N,             // Матрица A и ее leading dimension
                &beta, 
                C, P);            // Матрица C (результат) и ее leading dimension
    
    cublasDestroy(handle);
}

// Основная функция умножения матриц с работой с GPU
void mul(std::vector<double>* h_A, std::vector<double>* h_B, 
         const int row, const int col) {

    // Выделение памяти на GPU 
    double *d_A, *d_B;
    cudaMalloc(&d_A, row * col * sizeof(double));
    cudaMalloc(&d_B, row * col * sizeof(double));
    
    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, (*h_A).data(), row * col * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*h_B).data(), row * col * sizeof(double), cudaMemcpyHostToDevice);
    
    // Вычисление произведения матриц с использованием cuBLAS
    matrixMultiplyCuBLAS(d_A, d_B, d_B, row, col, col);
    
    // Копирование результата с GPU на CPU
    cudaMemcpy((*h_B).data(), d_B, row * col * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Освобождение памяти GPU
    cudaFree(d_A);
    cudaFree(d_B);
}

void print_matrix(std::vector < double > *vec, std::ofstream *out, int row_a, int col_b){
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            *out << (*vec)[i * col_b + j] << " ";
        }
        *out << std::endl;
    }
    *out<<std::endl;
}

double get_sp(std::vector < double > *vec, int n){
    double res = 0;
    for(int i = 0; i < n; i++){
        res += (*vec)[i*n+i];
    }
    return res;
}

double get_p(std::vector < double > *p, std::vector < double > *sp, int n, int idx){
    double res = (*sp)[idx];
    for(int i = 1; i < idx; i++){
        res -= (*p)[i] * (*sp)[idx-i];
    }
    return res/idx;
}

std::vector < double > get_obr(std :: vector < double > h_A, int row, int col){
    
    std:: vector < std::vector < double > > list_matrix(row+1, h_A);
    for(int i = 0; i < row*row; i++)
        list_matrix[0][i] = 0;
    for(int i = 0;i < row; i++)
        list_matrix[0][i*row+i] = 1;

    std:: vector < double > sp = {0, get_sp(&h_A, row)};

    std:: vector < double > p = {0};
    p.push_back(get_p(&p,&sp,row, 1));

    vector_copy(&h_A, &list_matrix[1]);
    for(int i = 2; i <= row; i++){
        mul(&list_matrix[i-1], &list_matrix[i], row, col);
        sp.push_back(get_sp(&list_matrix[i], row));
        p.push_back(get_p(&p, &sp, row, i));        
    }

    vector_copy(&list_matrix[row - 1], &h_A);

    for(int i = 1; i < row; i++){
        for(int j = 0; j < list_matrix[row].size(); j++){
            h_A[j] -= p[i] * list_matrix[row - i - 1][j];
        }
    }


    for(int i = 0; i < row * row; i++)
        h_A[i] /= p[row];

    return h_A;
}

double lap(double fi, int h_x, int h_y){
    
}

int main() {
    //Открытие файлов для ввода и вывода данных
    std::ifstream in("input.txt");
    std::ofstream out("output.txt");

    // int row;          // Количество строк матрицы A
    // int col;    // Количество столбцов A и строк B
    // // Ввод размеров первой матрицы
    // in >> row >> col;

    // //Инициализация и ввод первой матрицы
    // std::vector<double> h_A(row * col);
    // for(int i = 0; i < row * col; i++) {
    //     in >> h_A[i];
    // }
    // h_A = get_obr(h_A, row, col);
    // print_matrix(&h_A, &out, row, col);

    int width_c, width_a, U;
    in >> width_c >> width_a >> U;


}