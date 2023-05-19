#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <vector>
#include <chrono>
#include <climits>
#include <map>
#include <algorithm>

#define NUM_THREADS 4

#define TEST_SIZE 16000000

double RandomGenerateNumber() {
    return rand() % INT16_MAX;
}
int swap_map[TEST_SIZE] = {0};

void PSRS_Sort(std::vector<int>& array, int length) {
    int sample[NUM_THREADS * NUM_THREADS];  //共p^2个样本元素
    int range = length / NUM_THREADS;  //每个线程处理的元素个数 即n/p
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        std::sort(array.begin() + id * range, array.begin() + (id + 1) * range);    //局部排序
        for(int i = 0; i < NUM_THREADS; i++) {
            sample[id * NUM_THREADS + i] = array[id * range + i * range / NUM_THREADS];  //每个线程选取p个样本元素
        }
    }   
    
    std::sort(sample, sample + NUM_THREADS * NUM_THREADS);  //对样本元素进行排序

    int pivot[NUM_THREADS - 1];  //选取p-1个主元
    for(int i = 0; i < NUM_THREADS - 1; i++) {
        pivot[i] = sample[(i + 1) * NUM_THREADS];
    }
    
    #pragma omp parallel
    {   //主元划分
        int id = omp_get_thread_num();
        int start = id * range;
        int end = (id + 1) * range;
        for (int i = start; i < end; i++){
            int j = 0;
            
            while (j < NUM_THREADS - 1) {
                if (array[i] <= pivot[j]) {
                    break;
                }
                j++;
            }
            swap_map[i] = j;
        }
    }

    int count[NUM_THREADS] = {0};
    
    for (int i = 0; i < length; i++){  //统计每个段的元素个数
        count[swap_map[i]]++;
    }
    
    int offset[NUM_THREADS] = {0};
    offset[0] = 0;
    for(int i = 1; i < NUM_THREADS; i++){  //计算每个段的起始位置
        offset[i] = offset[i-1] + count[i-1];
    }
    
    std::vector<int> temp(array);
    int cnt[NUM_THREADS] = {0};
    
    for(int i = 0; i < length; i++){  //根据map进行交换
        array[offset[swap_map[i]] + cnt[swap_map[i]]] = temp[i];
        cnt[swap_map[i]]++;
    }
    
    #pragma omp parallel
    { 
        int id = omp_get_thread_num();
        if(id != NUM_THREADS - 1)
            std::sort(array.begin() + offset[id], array.begin() + offset[id+1]);    //局部排序
        else
            std::sort(array.begin() + offset[id], array.end());    //局部排序
    }
    
}

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(NULL)));
    std::vector<int> arr(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        arr[i] = static_cast<int>(RandomGenerateNumber());
    }
    
    auto start = std::chrono::system_clock::now();
    PSRS_Sort(arr, TEST_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << "Is sorted:" << bool(is_sorted(arr.begin(), arr.end())) << std::endl;
    auto duration = duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "并行时间:" << double(duration.count()) * std::chrono::microseconds::period::num << std::endl;
    
    for (int i = 0; i < TEST_SIZE; i++) {
        arr[i] = static_cast<int>(RandomGenerateNumber());
    }
    auto start1 = std::chrono::system_clock::now();
    std::sort(arr.begin(), arr.end());
    auto end1 = std::chrono::system_clock::now();
    std::cout << "Is sorted:" << bool(is_sorted(arr.begin(), arr.end())) << std::endl;
    auto duration1 = duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "串行时间:" << double(duration1.count()) * std::chrono::microseconds::period::num << std::endl;

    std::cout << "加速比:" << double(duration1.count()) * std::chrono::microseconds::period::num / double(duration.count()) * std::chrono::microseconds::period::num << std::endl;
    
    return 0;
}
