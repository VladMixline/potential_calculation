#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

// Умножение матриц с использованием cuBLAS
void matrixMultiplyCuBLAS(float* A, float* B, float* C, int M, int N, int P) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;  // Коэффициент для матрицы A*B
    const float beta = 0.0f;   // Коэффициент для матрицы C (обнуляем существующее значение)
    
    cublasSgemm(handle, 
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
void mul(std::vector<float>* h_A, std::vector<float>* h_B, std::vector<float>* h_C, 
         const int row_a, const int col_a_row_b, const int col_b) {
    
    // Выделение памяти на GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, row_a * col_a_row_b * sizeof(float));
    cudaMalloc(&d_B, col_a_row_b * col_b * sizeof(float));
    cudaMalloc(&d_C, row_a * col_b * sizeof(float));
    
    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, (*h_A).data(), row_a * col_a_row_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*h_B).data(), col_a_row_b * col_b * sizeof(float), cudaMemcpyHostToDevice);
    
    // Вычисление произведения матриц с использованием cuBLAS
    matrixMultiplyCuBLAS(d_A, d_B, d_C, row_a, col_a_row_b, col_b);
    
    // Копирование результата с GPU на CPU
    cudaMemcpy((*h_C).data(), d_C, row_a * col_b * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Освобождение памяти GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Открытие файлов для ввода и вывода данных
    std::ifstream in("input.txt");
    std::ofstream out("output.txt");

    // Переменные для размеров матриц
    int row_a;          // Количество строк матрицы A
    int col_a_row_b;    // Количество столбцов A и строк B
    int col_b;          // Количество столбцов матрицы B

    // Ввод размеров первой матрицы
    in >> row_a >> col_a_row_b;

    // Инициализация и ввод первой матрицы
    std::vector<float> h_A(row_a * col_a_row_b);
    for(int i = 0; i < row_a * col_a_row_b; i++) {
        in >> h_A[i];
    }
    
    // Ввод размеров второй матрицы
    in >> col_a_row_b >> col_b;

    // Инициализация и ввод второй матрицы
    std::vector<float> h_B(col_a_row_b * col_b);
    for(int i = 0; i < col_b * col_a_row_b; i++) {
        in >> h_B[i];
    }
    
    // Инициализация вектора для результата умножения
    std::vector<float> h_C(row_a * col_b, 0);
    
    // Умножение матриц
    mul(&h_A, &h_B, &h_C, row_a, col_a_row_b, col_b);

    // Вывод результата в файл
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            out << h_C[i * col_b + j] << " ";
        }
        out << std::endl;
    }
    
    return 0;
}