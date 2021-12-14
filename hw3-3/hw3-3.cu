#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define B 64
#define half_B 32

const int INF = ((1 << 30) - 1);

void input(char* inFileName);
void output(char* outFileName);

__global__ void block_FW_Phase1(int* dist_d, int n, int round);
__global__ void block_FW_Phase2(int* dist_d, int n, int round);
__global__ void block_FW_Phase3(int* dist_d, int n, int round, int start_idx_y);

int ceil(int a, int b){
    return (a + b - 1) / b;
}

int input_n, n, m;
int* Dist;

int main(int argc, char* argv[]) {
    // int num_gpus = -1, num_cpus = -1, gpu_id = -1;

    // cudaGetDeviceCount(&num_gpus);
    // num_cpus = omp_get_max_threads();
    // omp_set_num_threads(num_cpus);
    
    // unsigned int cpu_thread_id;
    // unsigned int num_cpu_threads;
    
    // #pragma omp parallel
    // {
    //     cpu_thread_id = omp_get_thread_num(); 
    //     num_cpu_threads = omp_get_num_threads(); 
    //     cudaSetDevice(cpu_thread_id);
    //     cudaGetDevice(&gpu_id);
    //     // printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
    // }

    FILE* file = fopen(argv[1], "rb");

    fread(&input_n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    int round = ceil(input_n, B);
    n = B * round; // avoid invalid memory access

    size_t matrix_size = n * n * sizeof(int);

    Dist = (int*)malloc(matrix_size);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } 
            else {
                Dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; i++) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);

    int* dist_d[2];
    cudaHostRegister(Dist, matrix_size, cudaHostRegisterDefault);

    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num(); 
        unsigned int num_threads = omp_get_num_threads();
        cudaSetDevice(cpu_thread_id);

        cudaMalloc(&dist_d[cpu_thread_id], matrix_size);
        // cudaMemcpy(dist_d, Dist, matrix_size, cudaMemcpyHostToDevice);

        int round_per_GPU = round / 2;
        if(cpu_thread_id == 1){
            round_per_GPU += round % 2;
        } 

        int start_idx_y;
        size_t copy_pos;
        size_t copy_size;

        if(cpu_thread_id == 0){
            start_idx_y = 0;
            copy_pos = 0;
            copy_size = round_per_GPU * B * n * sizeof(int);
        } 
        else{
            start_idx_y = round_per_GPU - round % 2;
            copy_pos = start_idx_y * B * n;
            copy_size = round_per_GPU * B * n * sizeof(int);
        }

        dim3 grid2(round - 1, 2);
        dim3 grid3(round, round_per_GPU);
        dim3 block(32, 32);

        // copy data in Dist to the corresponding device
        cudaMemcpy(dist_d[cpu_thread_id], Dist, matrix_size, cudaMemcpyHostToDevice);

        for(int r = 0; r < round; r++){
            // copy the row of pivot is enough
            if(r >= start_idx_y && r < start_idx_y + round_per_GPU){
                cudaMemcpy(Dist + r * B * n, dist_d[cpu_thread_id] + r * B * n, B * n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            #pragma omp barrier

            cudaMemcpy(dist_d[cpu_thread_id] + r * B * n, Dist + r * B * n, B * n * sizeof(int), cudaMemcpyHostToDevice);

            block_FW_Phase1<<<1, block>>>(dist_d[cpu_thread_id], n, r);
            block_FW_Phase2<<<grid2, block>>>(dist_d[cpu_thread_id], n, r);
            block_FW_Phase3<<<grid3, block>>>(dist_d[cpu_thread_id], n, r, start_idx_y);
        }
        cudaMemcpy(Dist + copy_pos, dist_d[cpu_thread_id] + copy_pos, copy_size, cudaMemcpyDeviceToHost);
        #pragma omp barrier
    }
    
    // cudaMemcpy(Dist, dist_d, matrix_size, cudaMemcpyDeviceToHost);

    FILE* outfile = fopen(argv[2], "w");
    for(int i = 0; i < input_n; i++){
        fwrite(&Dist[i * n], sizeof(int), input_n, outfile);
    }
    fclose(outfile);

    free(Dist);
    cudaFree(dist_d);

    return 0;
}

__global__ void block_FW_Phase1(int* dist_d, int n, int round) {
    __shared__ int s_dist[B][B];

    // ori_i and ori_j stores the position in the n*n matrix
    int ori_i = round * B + threadIdx.y;
    int ori_j = round * B + threadIdx.x;

    // i and j represents the position in the block
    int i = threadIdx.y;
    int j = threadIdx.x;

    s_dist[i][j] = dist_d[ori_i * n + ori_j];
    s_dist[i][j + half_B] = dist_d[ori_i * n + (ori_j + half_B)];
    s_dist[i + half_B][j] = dist_d[(ori_i + half_B) * n + ori_j];
    s_dist[i + half_B][j + half_B] = dist_d[(ori_i + half_B) * n + (ori_j + half_B)];

    __syncthreads();

    for (int k = 0; k < B; k++) {
        int temp1 = s_dist[i][k] + s_dist[k][j];
        if(temp1 < s_dist[i][j]){
            s_dist[i][j] = temp1;
        }
        int temp2 = s_dist[i][k] + s_dist[k][j + half_B];
        if(temp2 < s_dist[i][j + half_B]){
            s_dist[i][j + half_B] = temp2;
        }
        int temp3 = s_dist[i + half_B][k] + s_dist[k][j];
        if(temp3 < s_dist[i + half_B][j]){
            s_dist[i + half_B][j] = temp3;
        }
        int temp4 = s_dist[i + half_B][k] + s_dist[k][j + half_B];
        if(temp4 < s_dist[i + half_B][j + half_B]){
            s_dist[i + half_B][j + half_B] = temp4;
        }
        __syncthreads();
    }

    dist_d[ori_i * n + ori_j] = s_dist[i][j];
    dist_d[ori_i * n + (ori_j + half_B)] = s_dist[i][j + half_B];
    dist_d[(ori_i + half_B) * n + ori_j] = s_dist[i + half_B][j];
    dist_d[(ori_i + half_B) * n + (ori_j + half_B)] = s_dist[i + half_B][j + half_B];
}

__global__ void block_FW_Phase2(int* dist_d, int n, int round){
    __shared__ int s_dist[2][B][B];

    int ori_i, ori_j;

    if(blockIdx.y == 0){ // block with blockIdx.y == 0 handles pivot row
        ori_i = round * B + threadIdx.y;
        ori_j = blockIdx.x * B + threadIdx.x + ((blockIdx.x >= round) * B);
    }
    else{ // block with blockIdx.y == 1 handles pivot column
        ori_i = blockIdx.x * B + threadIdx.y + ((blockIdx.x >= round) * B);
        ori_j = round * B + threadIdx.x;
    }

    int pivot_i = round * B + threadIdx.y;
    int pivot_j = round * B + threadIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    // s_dist[0][x][y] store value in the pivot row/col blocK itself
    s_dist[0][i][j] = dist_d[ori_i * n + ori_j]; 
    s_dist[0][i][j + half_B] = dist_d[ori_i * n + (ori_j + half_B)];
    s_dist[0][i + half_B][j] = dist_d[(ori_i + half_B) * n + ori_j];
    s_dist[0][i + half_B][j + half_B] = dist_d[(ori_i + half_B) * n + (ori_j + half_B)];

    // s_dist[1][x][y] store value in pivot block
    s_dist[1][i][j] = dist_d[pivot_i * n + pivot_j]; 
    s_dist[1][i][j + half_B] = dist_d[pivot_i * n + (pivot_j + half_B)];
    s_dist[1][i + half_B][j] = dist_d[(pivot_i + half_B) * n + pivot_j];
    s_dist[1][i + half_B][j + half_B] = dist_d[(pivot_i + half_B) * n + (pivot_j + half_B)];
    
    __syncthreads();

    for (int k = 0; k < B; k++) {
        int temp1 = s_dist[!blockIdx.y][i][k] + s_dist[blockIdx.y][k][j];
        if(temp1 < s_dist[0][i][j]){
            s_dist[0][i][j] = temp1;
        }
        int temp2 = s_dist[!blockIdx.y][i][k] + s_dist[blockIdx.y][k][j + half_B];
        if(temp2 < s_dist[0][i][j + half_B]){
            s_dist[0][i][j + half_B] = temp2;
        }
        int temp3 = s_dist[!blockIdx.y][i + half_B][k] + s_dist[blockIdx.y][k][j];
        if(temp3 < s_dist[0][i + half_B][j]){
            s_dist[0][i + half_B][j] = temp3;
        }
        int temp4 = s_dist[!blockIdx.y][i + half_B][k] + s_dist[blockIdx.y][k][j + half_B];
        if(temp4 < s_dist[0][i + half_B][j + half_B]){
            s_dist[0][i + half_B][j + half_B] = temp4;
        }
        // __syncthreads();
    }
    dist_d[ori_i * n + ori_j] = s_dist[0][i][j];
    dist_d[ori_i * n + (ori_j + half_B)] = s_dist[0][i][j + half_B];
    dist_d[(ori_i + half_B) * n + ori_j] = s_dist[0][i + half_B][j];
    dist_d[(ori_i + half_B) * n + (ori_j + half_B)] = s_dist[0][i + half_B][j + half_B];
}

__global__ void block_FW_Phase3(int* dist_d, int n, int round, int start_idx_y){
    __shared__ int s_dist[2][B][B];

    int block_y = blockIdx.y + start_idx_y;

    if(blockIdx.x == round || block_y == round) return;
    // if blockIdx.y >= round or blockIdx.x >= round, the position of i or j adds B more to move across the pivot row/col 
    // int ori_i = blockIdx.y * B + threadIdx.y + ((blockIdx.y >= round) * B);
    // int ori_j = blockIdx.x * B + threadIdx.x + ((blockIdx.x >= round) * B);
    int ori_i = block_y * B + threadIdx.y;
    int ori_j = blockIdx.x * B + threadIdx.x;

    int pivot_row_i = round * B + threadIdx.y;
    int pivot_row_j = ori_j;
    int pivot_col_i = ori_i;
    int pivot_col_j = round * B + threadIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;
    
    // s_dist[0][x][y] store things of the pivot row block
    s_dist[0][i][j] = dist_d[pivot_row_i * n + pivot_row_j]; 
    s_dist[0][i][j + half_B] = dist_d[pivot_row_i * n + (pivot_row_j + half_B)];
    s_dist[0][i + half_B][j] = dist_d[(pivot_row_i + half_B) * n + pivot_row_j];
    s_dist[0][i + half_B][j + half_B] = dist_d[(pivot_row_i + half_B) * n + (pivot_row_j + half_B)];

    // s_dist[1][x][y] store things of the pivot column block
    s_dist[1][i][j] = dist_d[pivot_col_i * n + pivot_col_j]; 
    s_dist[1][i][j + half_B] = dist_d[pivot_col_i * n + (pivot_col_j + half_B)];
    s_dist[1][i + half_B][j] = dist_d[(pivot_col_i + half_B) * n + pivot_col_j];
    s_dist[1][i + half_B][j + half_B] = dist_d[(pivot_col_i + half_B) * n + (pivot_col_j + half_B)];

    __syncthreads();

    int self_dis1 = dist_d[ori_i * n + ori_j];
    int self_dis2 = dist_d[ori_i * n + (ori_j + half_B)];
    int self_dis3 = dist_d[(ori_i + half_B) * n + ori_j];
    int self_dis4 = dist_d[(ori_i + half_B) * n + (ori_j + half_B)];

    for (int k = 0; k < B; k++) {
        int temp1 = s_dist[1][i][k] + s_dist[0][k][j];
        if(temp1 < self_dis1){
            self_dis1 = temp1;
        }
        int temp2 = s_dist[1][i][k] + s_dist[0][k][j + half_B];
        if(temp2 < self_dis2){
            self_dis2 = temp2;
        }
        int temp3 = s_dist[1][i + half_B][k] + s_dist[0][k][j];
        if(temp3 < self_dis3){
            self_dis3 = temp3;
        }
        int temp4 = s_dist[1][i + half_B][k] + s_dist[0][k][j + half_B];
        if(temp4 < self_dis4){
            self_dis4 = temp4;
        }
        // __syncthreads();
    }
    dist_d[ori_i * n + ori_j] = self_dis1;
    dist_d[ori_i * n + (ori_j + half_B)] = self_dis2;
    dist_d[(ori_i + half_B) * n + ori_j] = self_dis3;
    dist_d[(ori_i + half_B) * n + (ori_j + half_B)] = self_dis4;
}