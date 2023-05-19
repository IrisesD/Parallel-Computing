#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <chrono>
#include <climits>
#include <map>
#include <mpi.h>
#include <algorithm>

#define TEST_SIZE 16000000

double RandomGenerateNumber() {
    return rand() % INT16_MAX;
}

void PSRS_Sort(int* array, int id, int num_procs) {    
    int range = TEST_SIZE / num_procs;

    // 局部排序
    int* local_arr = (int*)malloc(range * sizeof(int));
    MPI_Scatter(array, range, MPI_INT, local_arr, range, MPI_INT, 0, MPI_COMM_WORLD);
    std::sort(local_arr, local_arr + range);
    MPI_Gather(local_arr, range, MPI_INT, array, range, MPI_INT, 0, MPI_COMM_WORLD);
    free(local_arr);

    int* pivot = (int*)malloc((num_procs - 1) * sizeof(int));
    
    // 选取样本
    if (id == 0) {
        int* sample = (int*)malloc(num_procs * num_procs * sizeof(int));
        for (int i = 0; i < num_procs * num_procs; ++i) {
            sample[i] = array[i * range / num_procs];
        }

        // 样本排序
        std::sort(sample, sample + num_procs * num_procs);

        // 选择主元
        for (int i = 0; i < num_procs - 1; ++i) {
            pivot[i] = sample[(i + 1) * num_procs];
        }
        free(sample);
    }

    MPI_Bcast(pivot, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);    // 广播主元
    
    // 主元划分
    MPI_Bcast(array, TEST_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    int* swap_map = (int*)malloc(range * sizeof(int));
    for (int i = 0; i < range; ++i) {
        int j = 0;
        while (j < num_procs - 1) {
                if (array[id * range + i] <= pivot[j]) {
                    break;
                }
                j++;
            }
        swap_map[i] = j;
    }
    
    
    int* swap = (int*)malloc(TEST_SIZE * sizeof(int));
    MPI_Gather(swap_map, range, MPI_INT, swap, range, MPI_INT, 0, MPI_COMM_WORLD);
    int* count = (int*)malloc(num_procs * sizeof(int));
    for (size_t i = 0; i < num_procs; i++){
        count[i] = 0;
    }
    
    int* offset = (int*)malloc(num_procs * sizeof(int));
    free(swap_map);
    // 全局交换
    
    if (id == 0) {
        
        for (int i = 0; i < TEST_SIZE; i++){  //统计每个段的元素个数
            count[swap[i]]++;
        }
        
        offset[0] = 0;
        for(int i = 1; i < num_procs; i++){  //计算每个段的起始位置
            offset[i] = offset[i-1] + count[i-1];
        }
        
        int* temp = (int*)malloc(TEST_SIZE * sizeof(int));
        for (int i = 0; i < TEST_SIZE; i++){
            temp[i] = array[i];
        }
        int* cnt = (int*)malloc(num_procs * sizeof(int));
        for(int i = 0; i < num_procs; i++){  //初始化计数器
            cnt[i] = 0;
        }
        
        for(int i = 0; i < TEST_SIZE; i++){  //根据map进行交换
            array[offset[swap[i]] + cnt[swap[i]]] = temp[i];
            cnt[swap[i]]++;
        }
        
    }
    // 归并排序
    MPI_Bcast(offset, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(count, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(array, TEST_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if(id != num_procs - 1)
        std::sort(array + offset[id], array + offset[id+1]);
    else{
        std::sort(array + offset[id], array + TEST_SIZE);
    }

    if (id != 0) {
        MPI_Send(array + offset[id], count[id], MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < num_procs; ++i) {
            MPI_Recv(array + offset[i], count[i], MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
}

int main(int argc, char* argv[]) {
    int id, num_procs;  // 进程号和总进程数

    int* arr = (int*)malloc(TEST_SIZE * sizeof(int));
    for (int i = 0; i < TEST_SIZE; i++) {
        arr[i] = static_cast<int>(RandomGenerateNumber());
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    srand(static_cast<unsigned int>(time(NULL)));
    auto start = std::chrono::system_clock::now();
    PSRS_Sort(arr, id, num_procs);
    auto end = std::chrono::system_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - start);
    if(id == 0){
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "并行 Is sorted:" << std::is_sorted(arr, arr + TEST_SIZE) << std::endl;
        std::cout << "并行时间:" << double(duration.count()) * std::chrono::microseconds::period::num << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < TEST_SIZE; i++) {
        arr[i] = static_cast<int>(RandomGenerateNumber());
    }
    if(id == 0){
        auto start1 = std::chrono::system_clock::now();
        std::sort(arr, arr + TEST_SIZE);
        auto end1 = std::chrono::system_clock::now();
        std::cout << "串行 Is sorted:" << std::is_sorted(arr, arr + TEST_SIZE) << std::endl;
        auto duration1 = duration_cast<std::chrono::microseconds>(end1 - start1);
        std::cout << "串行时间:" << double(duration1.count()) * std::chrono::microseconds::period::num << std::endl;
        std::cout << "加速比:" << double(duration1.count()) / double(duration.count()) << std::endl;
    }
    MPI_Finalize();
    free(arr);
    return 0;
}