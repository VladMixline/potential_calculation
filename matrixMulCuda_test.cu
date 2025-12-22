#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>


double N, al, ar, cl, cr, X = 0.35, U, hx = 0.3, hy = 0.3;

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
void matrixMultiplyCuBLAS(double* A, double* B, double* C, int M, int N_, int P) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const double alpha = 1.0f;  // Коэффициент для матрицы A*B
    const double beta = 0.0f;   // Коэффициент для матрицы C (обнуляем существующее значение)
    
    cublasDgemm_v2(handle, 
                CUBLAS_OP_N,      // Без транспонирования матрицы A
                CUBLAS_OP_N,      // Без транспонирования матрицы B
                P, M, N_,          // Размеры для column-major порядка
                &alpha, 
                B, P,             // Матрица B и ее leading dimension
                A, N_,             // Матрица A и ее leading dimension
                &beta, 
                C, P);            // Матрица C (результат) и ее leading dimension
    
    cublasDestroy(handle);
}

// Основная функция умножения матриц с работой с GPU
void mul(std::vector<double>* h_A, std::vector<double>* h_B, 
         const int row, const int col_row, const int col) {

    // Выделение памяти на GPU 
    double *d_A, *d_B;
    cudaMalloc(&d_A, row * col_row * sizeof(double));
    cudaMalloc(&d_B, col_row * col * sizeof(double));
    
    // Копирование данных с CPU на GPU
    cudaMemcpy(d_A, (*h_A).data(), row * col_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*h_B).data(), col_row * col * sizeof(double), cudaMemcpyHostToDevice);
    
    // Вычисление произведения матриц с использованием cuBLAS
    matrixMultiplyCuBLAS(d_A, d_B, d_B, row, col_row, col);
    
    // Копирование результата с GPU на CPU
    cudaMemcpy((*h_B).data(), d_B, col_row * col * sizeof(double), cudaMemcpyDeviceToHost);
    
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
        mul(&list_matrix[i-1], &list_matrix[i], row, col, col);
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


std::vector < std::vector < double > > make_yakob(std::vector < double > *fi) {
	std::vector < std::vector < double > > yakob(N * N, std::vector < double >(N * N, 0));
	int line = 0;
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++, line++) {
			yakob[line][i * N + j] -= 4;
			yakob[line][(i - 1) * N + j] += 1;
			yakob[line][(i + 1) * N + j] += 1;
			yakob[line][i * N + j - 1] += 1;    
			yakob[line][i * N + j + 1] += 1;
		}
	}
	for (int j = 1; j < N - 1; j++, line++) {
		yakob[line][j] -= 1;
		yakob[line][N + j] += 1;
	}
	for (int j = 1; j < N - 1; j++, line++) {
		yakob[line][(N - 2) * N + j] -= 1;
		yakob[line][(N - 1) * N + j] += 1;
	}
	for (int i = 0; i < al - 1; i++, line++) {
		yakob[line][i * N + 1] += 1;
		yakob[line][i * N] -= 1;
	}
	for (int i = ar; i < N; i++, line++) {
		yakob[line][i * N + 1] += 1;
		yakob[line][i * N] -= 1;
	}
	for (int i = 0; i < cl - 1; i++, line++) {
		yakob[line][i * N + N - 1] += 1;
		yakob[line][i * N + N - 2] -= 1;
	}
	for (int i = cr; i < N; i++, line++) {
		yakob[line][i * N + N - 1] += 1;
		yakob[line][i * N + N - 2] -= 1;
	}
	for (int i = al - 1; i < ar; i++, line++) {
		yakob[line][i * N] += 1;
	}
	for (int i = cl - 1; i < cr; i++, line++) {
		yakob[line][i * N + N - 1] += 1.0 - (0.0032 * -1 * X * (*fi)[i * N + N - 1] / hy + 0.055) * (-1 * X / hy);
		yakob[line][i * N + N - 2] -= (0.0032 * X * (*fi)[i * N + N - 2] / hy + 0.055) * (X / hy);
	}
	return yakob;
}

std::vector < double > create_fi() {
	std::vector < double > fi(N * N, 0);
	for (int i = 0; i < N; i++)
		fi[i * N] = U;
	for (int i = 0; i < N; i++)
		for (int j = 1; j < N - 1; j++) {
			fi[i * N + j] = U * (N - j - 1) / (N - 1);
		}
	return fi;
}


double lapl(int i, int j, std::vector<double> *fi){
    return ((*fi)[(i - 1) * N + j] - 2 * (*fi)[i * N + j] + (*fi)[(i + 1) * N + j]) / hx + \
    ((*fi)[i * N + j - 1] - 2 * (*fi)[i * N + j] + (*fi)[i * N + j + 1]) / hy;
}


double regional_1(int j, std::vector<double> *fi){
    return ((*fi)[N + j] - (*fi)[j]) / hx;
}


double regional_2(int j, std::vector<double> *fi){
    return ((*fi)[(N - 1) * N + j] - (*fi)[(N - 2) * N + j]) / hx;
}


double regional_3(int i, std::vector<double> *fi){
    return ((*fi)[i * N + 1] - (*fi)[i * N]) / hy;
}


double regional_4(int i, std::vector<double> *fi){
    return ((*fi)[i * N + N - 1] - (*fi)[i * N + N - 2]) / hy;
}


double regional_anod(int i, std::vector<double> *fi){
    return (*fi)[i * N] + 1.2 -  U;
}


double calc_i_katod(int i, std::vector<double> *fi){
    return -1 * X * ((*fi)[i * N + N - 1] - (*fi)[i * N + N - 2]) / hy;
}


double calc_Fc(int i, std::vector<double> *fi){
    double ic = calc_i_katod(i, fi);
    return 0.0016 * ic * ic + 0.055 * ic + 1.347;
}


double regional_katod(int i, std::vector<double> *fi){
    return (*fi)[i * N + N - 1] - calc_Fc(i, fi);
}


std::vector < double > calc_fi(std::vector < double > *fi){
    std::vector < double > res;
    for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			res.push_back(lapl(i, j, fi));
		}
	}
	for (int j = 1; j < N - 1; j++) {
        res.push_back(regional_1(j, fi));
	}
	for (int j = 1; j < N - 1; j++) {
        res.push_back(regional_2(j, fi));
	}
	for (int i = 0; i < al - 1; i++) {
		res.push_back(regional_3(i, fi));
	}
	for (int i = ar; i < N; i++) {
		res.push_back(regional_3(i, fi));
	}
	for (int i = 0; i < cl - 1; i++) {
        res.push_back(regional_4(i, fi));
	}
	for (int i = cr; i < N; i++) {
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


double calc_thickness(int i, std::vector < double > *fi){
    double kme = 1.22, dn = 7.133, dt = 420;
    return kme * X / dn * ((*fi)[i * N + N - 2] - (*fi)[i * N + N - 1]) / hy * dt;
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

    
    in >> N >> al >> ar >> cl >> cr >> U;
    //N = 4, al = 2, ar = 3, cl = 2, cr = 3, U = 5;
	std::vector < double > fi = create_fi();

    print_matrix(&fi, &out, N, N);
    out << std::endl;
    double eps = 1;
    while(eps > 0.001){
	    std::vector < std::vector < double > > yakob = make_yakob(&fi);
        std::vector < double > yakob_in_line;
        for(int i = 0; i < N * N; i++)
            for(int j = 0; j < N * N; j++)
                yakob_in_line.push_back(yakob[i][j]);
        //print_matrix(&yakob_in_line,&out, N*N, N*N);
        yakob_in_line = get_obr(yakob_in_line, N * N, N * N);
        std::vector < double > fi_ = calc_fi(&fi);
        mul(&yakob_in_line, &fi_, N * N, N * N, 1);
        for(auto it:fi_){
            out << std::setw(15) << it;
        }
        out << std::endl << fi_.size() << std::endl;
        eps = 0;
        for(int i = 0; i < N * N; i++){
            fi[i] -= fi_[i];
            eps += abs(fi_[i]);
        }
        print_matrix(&fi, &out, N, N);
    }


    std::vector < double > thickness;
    for (int i = 0; i < N; i++) {
        thickness.push_back(calc_thickness(i, &fi));
	}

    for(auto it : thickness)
        out << it << "  "; 

    //print_matrix(&fi, &out, N, N);

    
    //вывод распределения потенциала, анод сверху, катод снизу, т.е. у = 0 (j) для анода находится сверху
	// for (int j = 0; j < N; j++) {
	// 	for (int i = 0; i < N; i++) {
	// 		std::cout << fi[i * N + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// for (int i = 0; i < N * N; i++) {
	// 	for (auto it : yakob[i])
	// 		std::cout << std::setw(10) << it;
	// 	std::cout << std::endl;
	// }


    // int row,col_row,col;
    // in>>row>>col_row>>col;
    // std::vector<double> h_A(row * col_row);
    // std::vector<double> h_B(col_row * col);
    // for(int i = 0; i < row * col_row; i++) {
    //     in >> h_A[i];
    // }
    // for(int i = 0; i < col_row * col; i++) {
    //     in >> h_B[i];
    // }

    // mul(&h_A,&h_B,row, col_row,col);
    // print_matrix(&h_B, &out, col_row, col);
}