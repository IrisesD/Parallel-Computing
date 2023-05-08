#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <vector>

#define NUM_THREADS 5

#define RANDOM_LIMIT 100
#define TEST_SIZE 50

using namespace std;
const int p = 10;
double RandomGenerateNumber() {
    return rand() % INT16_MAX;
}

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(vector<int>& array, int left, int right) {
    int x = array[right];
    int i = left - 1;
    for (int j = left; j < right; j++) {
        if (array[j] <= x) {
            swap(array[++i], array[j]);
        }
    }
    swap(array[i + 1], array[right]);
    return i + 1;
}

void quickSort(vector<int>& array, int left, int right) {
    if (left < right) {
        int q = partition(array, left, right);
        quickSort(array, left, q - 1);
        quickSort(array, q + 1, right);
    }
}

void PSRSSort(vector<int>& array, int length) {
    int base = length / NUM_THREADS;
    vector<int> sample(NUM_THREADS * NUM_THREADS);
    vector<int> pivot(NUM_THREADS - 1);
    vector<vector<int> > count(NUM_THREADS, vector<int>(NUM_THREADS, 0));
    vector<vector<vector<int> > > pivotArray(NUM_THREADS, vector<vector<int> >(NUM_THREADS, vector<int>(50, 0)));

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        quickSort(array, id * base, (id + 1) * base - 1);   // 局部排序

        for (int j = 0; j < NUM_THREADS; j++) {
            sample[id * NUM_THREADS + j] = array[id * base + (j + 1) * base / (NUM_THREADS + 1)];   //选取样本
        }

        #pragma omp barrier
        #pragma omp master
        {
            quickSort(sample, 0, NUM_THREADS * NUM_THREADS - 1);    //样本排序

            for (int i = 1; i < NUM_THREADS; i++) {
                pivot[i - 1] = sample[i * NUM_THREADS]; //选择主元
            }
        }

        #pragma omp barrier

        for (int k = 0, m = 0; k < base; k++) {
            if (array[id * base + k] < pivot[m]) {
                pivotArray[id][m][count[id][m]++] = array[id * base + k];   
            } else {
                m != NUM_THREADS - 1 ? m++ : 0;
                pivotArray[id][m][count[id][m]++] = array[id * base + k];
            }
        }   //主元划分

        #pragma omp barrier

        for (int k = 0; k < NUM_THREADS; k++) {
            if (k != id) {
                memcpy(pivotArray[id][id].data() + count[id][id], pivotArray[k][id].data(), sizeof(int) * count[k][id]);
                count[id][id] += count[k][id];
            }
        }   //全局交换

        quickSort(pivotArray[id][id], 0, count[id][id] - 1);    //归并排序
    }


    cout << "The Sorted Array is:" << endl;
    for (int x = 0; x < NUM_THREADS; x++) {
        for (int y = 0; y < count[x][x]; y++) {
            cout << pivotArray[x][x][y] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(NULL)));
    vector<int> arr(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        arr[i] = static_cast<int>(RandomGenerateNumber());
    }

    cout << "The Original Array is:" << endl;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < TEST_SIZE / p; j++) {
            cout << arr[i * (TEST_SIZE / p) + j] << " ";
        }
        cout << endl;
    }
    PSRSSort(arr, TEST_SIZE);

    return 0;
}
