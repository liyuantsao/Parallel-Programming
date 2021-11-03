#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <algorithm>
#include "mpi.h"

u_int32_t* counting_sort(u_int32_t* data_arr, int local_n, int shift_bits){
    // b is the array stores the result
    u_int32_t* b = new u_int32_t[local_n];
    // array size is 256 because we want to sort an 8-bit number
    u_int32_t* c = new u_int32_t[256];
    
    // Initialize counters
    memset(c, 0, 256*sizeof(u_int32_t));

    // count the number of occurrence of a particular value
    for(int i=0; i<local_n; i++){
        c[(data_arr[i] >> shift_bits) & 0xFF]++;
    }

    for(int i=1; i<256; i++){
        c[i] += c[i-1];
    }

    for(int i=local_n-1; i>=0; i--){
        b[(c[(data_arr[i] >> shift_bits) & 0xFF]--) - 1] = data_arr[i];
    }

    return b;
}

u_int32_t* radix_sort(u_int32_t* data_for_sort, int local_n){
    u_int32_t* temp = new u_int32_t[local_n];
    temp = data_for_sort;

    for(int i=0; i<4; i++){
        // shift 8 bits after doing one round of counting sort to sort another digit
        int shift_bits = i * 8;
        temp = counting_sort(temp, local_n, shift_bits);
    }
    return temp;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
    int n = atoi(argv[1]);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_File f;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

    // The size for each process
    int local_n = n / size; 
    // If we assign (n / size) numbers to each process, there are still (n % size) numbers are waiting for assignment.
    int remainder = n % size; 

    // Assign one remaining numbers to processes whose rank is less than the number of remaining process
    if(rank < remainder) local_n++; 
    
    // Store how many numbers should be handled by the process on the left hand side
    int left_n = local_n; 
    // Store how many numbers should be handled by the process on the right hand side
    int right_n = local_n; 
    
    if(rank == remainder) left_n++; 
    if(rank == remainder - 1) right_n--;

    float* data = new float[local_n];
    u_int32_t* data_for_sorting = new u_int32_t[local_n];
    float* sorted_buf = new float[local_n];
    float* left_buf = new float[left_n];
    float* right_buf = new float[right_n];
    float* tmp = new float[local_n];

    if(n <= size){ // Handle the case which has few data
        if(rank == 0){
            MPI_File_read_at(f, 0, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
            std::sort(data, data+n);

            MPI_File f_out;
            MPI_File_open(MPI_COMM_SELF, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_out);
            MPI_File_write_at(f_out, 0, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);

            return 0;
        }
        else{
            return 0;
        }
    }

    // The position that a process read data from
    int offset; 

    if(rank < remainder){
        offset = sizeof(float) * local_n * rank;
    }
    else{
        // Because processed whose rank < remainder reads sizeof(float) bytes more, the offset of others should
        // be local_n * rank * sizeof(float) + remainder * sizeof(float)
        offset = sizeof(float) * ((local_n * rank) + (remainder)); 
    }

    MPI_File_read_at(f, offset, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);

    // convert floating point to 32 bit unsigned int
    for(int i=0; i<local_n; i++){ 
        u_int32_t radix_num = *(u_int32_t*)&data[i];
        if(radix_num >> 31 == 1){
            radix_num *= -1;
            radix_num ^= (1 << 31);
        }
        radix_num ^= (1 << 31);
        data_for_sorting[i] = radix_num;
    }

    // radix sort
    data_for_sorting = radix_sort(data_for_sorting, local_n);

    // convert 32 bit unsigned int back to floating point
    for(int i=0; i<local_n; i++){
        u_int32_t radix_num = data_for_sorting[i];
        radix_num ^= (1 << 31);
        if((radix_num >> 31) == 1){
            radix_num ^= (1 << 31);
            radix_num *= -1;
        }
        data[i] = *(float*)&radix_num;
    }

    if(size == 1){
        MPI_File f_out;
        MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_out);
        MPI_File_write_at(f_out, 0, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);

        return 0;
    }
    // std::sort(data, data + local_n);

    bool isSorted = false;
    bool sortResult;
    int data_index, left_index, right_index;
    
    while (!isSorted) {
        isSorted = true;
        // round even
        if(rank % 2 == 1){
            MPI_Sendrecv(data, local_n, MPI_FLOAT, rank-1, 0, 
                            left_buf, left_n, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // merge
            data_index = local_n - 1;
            left_index = left_n - 1;
            for(int i=0; i<local_n; i++){
                if(data[data_index] >= left_buf[left_index]){
                    sorted_buf[local_n - 1 - i] = data[data_index];
                    data_index--;
                }
                else{
                    sorted_buf[local_n - 1 - i] = left_buf[left_index];
                    left_index--;
                    isSorted = false;
                }
            }
            // Swap pointer of old/sorted data array
            tmp = data;
            data = sorted_buf;
            sorted_buf = tmp;
        }
        // rank % 2 == 0
        else{ 
            // If the last even number process is the last process, do nothing this round
            if(rank == size-1){ 
                // Do nothing
            }
            else{
                MPI_Sendrecv(data, local_n, MPI_FLOAT, rank+1, 0, 
                            right_buf, right_n, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //merge
                data_index = 0;
                right_index = 0;
                for(int i=0; i<local_n; i++){
                    if(data[data_index] <= right_buf[right_index] || (right_index == right_n)){
                        sorted_buf[i] = data[data_index];
                        data_index++;
                    }
                    else{
                        sorted_buf[i] = right_buf[right_index];
                        right_index++;
                        isSorted = false;
                    }
                }
                // Swap pointer of old/sorted data array
                tmp = data;
                data = sorted_buf;
                sorted_buf = tmp;
            }
        }

        // round odd
        if(rank % 2 == 0){
            if(rank == 0){
                // Do nothing
            }
            else{
                // Send/recv data to/from the process on the left hand side
                MPI_Sendrecv(data, local_n, MPI_FLOAT, rank-1, 0, 
                            left_buf, left_n, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // merge (collect bigger data)
                data_index = local_n - 1;
                left_index = left_n - 1;
                for(int i=0; i<local_n; i++){
                    if(data[data_index] >= left_buf[left_index]){
                        sorted_buf[local_n - 1 - i] = data[data_index];
                        data_index--;
                    }
                    else{
                        sorted_buf[local_n - 1 - i] = left_buf[left_index];
                        left_index--;
                        isSorted = false;
                    }
                }
                // Swap pointer of old/sorted data array
                tmp = data;
                data = sorted_buf;
                sorted_buf = tmp;
            }
        }
        // rank % 2 == 1
        else{ 
            // If the last odd number process is the last process, do nothing this round
            if(rank == size-1){ 
                //Do nothing
            }
            else{
                MPI_Sendrecv(data, local_n, MPI_FLOAT, rank+1, 0, 
                            right_buf, right_n, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //merge
                data_index = 0;
                right_index = 0;
                for(int i=0; i<local_n; i++){
                    if((data[data_index] <= right_buf[right_index]) || (right_index == right_n)){
                        sorted_buf[i] = data[data_index];
                        data_index++;
                    }
                    else{
                        sorted_buf[i] = right_buf[right_index];
                        right_index++;
                        isSorted = false;
                    }
                }
                // Swap pointer of old/sorted data array
                tmp = data;
                data = sorted_buf;
                sorted_buf = tmp;
            }
        }
        // Collect sorting result of all processes
        MPI_Allreduce(&isSorted, &sortResult, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        isSorted = sortResult;
    }

    MPI_File f_out;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_out);
    MPI_File_write_at(f_out, offset, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);

	MPI_Finalize();
}
