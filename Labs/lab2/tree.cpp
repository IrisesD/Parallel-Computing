#include <mpi.h>
#include <iostream>
int A[8] = {7, 4, 3, 2, 5, 3, 2, 4};
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int sum = A[id];
    int temp_sum = 0;

    for (int i = 1; i < num_procs; i *= 2) {
        if (id % (2 * i) == 0) {
            if (id + i < num_procs) {
                MPI_Recv(&temp_sum, 1, MPI_INT, id + i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += temp_sum;
            }
        } else {
            MPI_Send(&sum, 1, MPI_INT, id - i, 0, MPI_COMM_WORLD);
            break;
        }
    }

    MPI_Bcast(&sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(id == 0 || id == 1 || id == 2 || id == 3 ||
    id == 4 || id == 5 || id == 6 || id == 7)
        std::cout << "The sum is " << sum << std::endl;

    MPI_Finalize();
    return 0;
}
