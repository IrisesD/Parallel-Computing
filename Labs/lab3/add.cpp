%%cu
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <chrono>
#include <climits>

using namespace std;
#define TEST_SIZE 100000
#define BLOCKSIZE 16
double RandomGenerateNumber() {
    return rand() % 10;
}
__global__ void add(const int *a, const int *b, int *c, int n) {
    int i = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(){
    int* A = (int*)malloc(TEST_SIZE * sizeof(int));
    for (int i = 0; i < TEST_SIZE; i++) {
        A[i] = static_cast<int>(RandomGenerateNumber());
    }
    int* B = (int*)malloc(TEST_SIZE * sizeof(int));
    for (int i = 0; i < TEST_SIZE; i++) {
        B[i] = static_cast<int>(RandomGenerateNumber());
    }
    int* C = (int*)malloc((TEST_SIZE) * sizeof(int));
    //串行加法
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < TEST_SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    //检验加法结果是否正确
    for (int i = 0; i < TEST_SIZE; i++) {
        if (C[i] != (A[i] + B[i])) {
            cout << "Wrong Answer!" << endl;
            return 0;
        }
    }
    cout << "串行时间:" << double(duration.count()) * chrono::microseconds::period::num << endl;
    //基于cuda的并行加法
    int* cuda_A, * cuda_B, * cuda_C;
    int* res = (int*)malloc((TEST_SIZE) * sizeof(int));
    cudaMalloc((void**)&cuda_A, TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&cuda_B, TEST_SIZE * sizeof(int));
    cudaMalloc((void**)&cuda_C, (TEST_SIZE+1) * sizeof(int));
    cudaMemcpy(cuda_A, A, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    int gridsize = (int)ceil(sqrt(ceil(TEST_SIZE / (BLOCKSIZE * BLOCKSIZE))));
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(gridsize, gridsize, 1);
    auto start1 = chrono::steady_clock::now();
    add << <dimGrid, dimBlock >> > (cuda_A, cuda_B, cuda_C, TEST_SIZE);
    cudaDeviceSynchronize();
    auto end1 = chrono::steady_clock::now();
    cudaMemcpy(res, cuda_C, (TEST_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);
    auto duration1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
    //检验加法结果是否正确
    for (int i = 0; i < TEST_SIZE; i++) {
        if (res[i] != (A[i] + B[i])) {
            cout << "Wrong Answer" << endl;
            return 0;
        }
    }
    cout << "并行时间:" << double(duration1.count()) * chrono::microseconds::period::num << endl;
    cout << "加速比:" << double(duration.count()) * chrono::microseconds::period::num / double(duration1.count()) * chrono::microseconds::period::num << endl;

    return 0;
}