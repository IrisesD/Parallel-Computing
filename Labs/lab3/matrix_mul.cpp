%%cu
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <chrono>
#include <climits>

using namespace std;
#define TEST_SIZE 256
#define BLOCKSIZE 16
double RandomGenerateNumber() {
    return rand() % 10;
}

__global__ void multiply(const int *a, const int *b, int *c, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    int sum = 0;
    if (row < n && col < n) {
        for (k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main(){
    
    int* A = (int*)malloc(TEST_SIZE * TEST_SIZE * sizeof(int));
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < TEST_SIZE; j++) {
            A[i * TEST_SIZE + j] = static_cast<int>(RandomGenerateNumber());
        }
    }
    int* B = (int*)malloc(TEST_SIZE * TEST_SIZE * sizeof(int));
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < TEST_SIZE; j++) {
            B[i * TEST_SIZE + j] = static_cast<int>(RandomGenerateNumber());
        }
    }
    int* C = (int*)malloc((TEST_SIZE * TEST_SIZE) * sizeof(int));
    auto start = chrono::steady_clock::now();
    //串行矩阵乘法
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < TEST_SIZE; j++) {
            C[i * TEST_SIZE + j] = 0;
            for (int k = 0; k < TEST_SIZE; k++) {
                C[i * TEST_SIZE + j] += A[i * TEST_SIZE + k] * B[k * TEST_SIZE + j];
            }
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    //检验矩阵乘法结果
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < TEST_SIZE; j++) {
            int sum = 0;
            for (int k = 0; k < TEST_SIZE; k++) {
                sum += A[i * TEST_SIZE + k] * B[k * TEST_SIZE + j];
            }
            if (sum != C[i * TEST_SIZE + j]) {
                cout << "Wrong Answer!" << endl;
                return 0;
            }
        }
    }
    cout << "串行时间:" << double(duration.count()) * chrono::microseconds::period::num << endl;
    //基于cuda的并行矩阵乘法
    int* cuda_A, * cuda_B, * cuda_C;
    int* res = (int*)malloc(TEST_SIZE * TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&cuda_A, TEST_SIZE * TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&cuda_B, TEST_SIZE * TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&cuda_C, TEST_SIZE * TEST_SIZE * sizeof(int));
    cudaMemcpy(cuda_A, A, TEST_SIZE * TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, TEST_SIZE * TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    double num = ceil(pow((double)TEST_SIZE,2) / pow((double)BLOCKSIZE, 2));
    int gridsize = (int)ceil(sqrt(num));
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(gridsize, gridsize, 1);
    auto start1 = chrono::steady_clock::now();
    multiply<<<dimGrid, dimBlock>>>(cuda_A, cuda_B, cuda_C, TEST_SIZE);
    cudaDeviceSynchronize();
    auto end1 = chrono::steady_clock::now();
    cudaMemcpy(res, cuda_C, TEST_SIZE * TEST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    //检验矩阵乘法结果
    for (int i = 0; i < TEST_SIZE; i++) {
        for (int j = 0; j < TEST_SIZE; j++) {
            int sum = 0;
            for (int k = 0; k < TEST_SIZE; k++) {
                sum += A[i * TEST_SIZE + k] * B[k * TEST_SIZE + j];
            }
            if (sum != res[i * TEST_SIZE + j]) {
                cout << "Wrong Answer!" << endl;
                return 0;
            }
        }
    }
    cout << "并行时间:" << double(duration1.count()) * chrono::microseconds::period::num << endl;
    cout << "加速比:" << double(duration.count()) * chrono::microseconds::period::num / double(duration1.count()) * chrono::microseconds::period::num << endl;

    return 0;
}