#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define B 64
#define half_B 32

const int INF = ((1 << 30) - 1);

void input(char* inFileName);
void output(char* outFileName);

__global__ void block_FW_Phase1(int* dist_d, int n, int round);
__global__ void block_FW_Phase2(int* dist_d, int n, int round);
__global__ void block_FW_Phase3(int* dist_d, int n, int round);

int ceil(int a, int b){
    return (a + b - 1) / b;
}

int input_n, n, m;
int* Dist;

int main(int argc, char* argv[]) {

    FILE* file = fopen(argv[1], "rb");

    fread(&input_n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    int round = ceil(input_n, B);
    n = B * round; // avoid invalid memory access

    size_t matrix_size = n * n * sizeof(int);

    Dist = (int*)malloc(matrix_size);

    // printf("n: %d, m: %d\n", n, m);
    
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
    // int B = ceil(sqrt(n));
    int* dist_d;

    cudaMalloc(&dist_d, matrix_size);
    cudaMemcpy(dist_d, Dist, matrix_size, cudaMemcpyHostToDevice);

    // printf("input_n: %d, m: %d, B: %d, round: %d\n", input_n, m, B, round);

    dim3 grid2(round - 1, 2);
    dim3 grid3(round - 1, round - 1);
    dim3 block(32, 32);

    size_t base_smem_size = B * B * sizeof(int);

    for(int r = 0; r < round; r++){
        block_FW_Phase1<<<1, block>>>(dist_d, n, r);
        block_FW_Phase2<<<grid2, block>>>(dist_d, n, r);
        block_FW_Phase3<<<grid3, block>>>(dist_d, n, r);
    }
    
    cudaMemcpy(Dist, dist_d, matrix_size, cudaMemcpyDeviceToHost);

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
        // if(blockIdx.x == round){
        //     printf("pivot_j: %d, ori_j: %d\n", round * B + threadIdx.x, ori_j);
        // }
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
        __syncthreads();
    }
    dist_d[ori_i * n + ori_j] = s_dist[0][i][j];
    dist_d[ori_i * n + (ori_j + half_B)] = s_dist[0][i][j + half_B];
    dist_d[(ori_i + half_B) * n + ori_j] = s_dist[0][i + half_B][j];
    dist_d[(ori_i + half_B) * n + (ori_j + half_B)] = s_dist[0][i + half_B][j + half_B];
}

__global__ void block_FW_Phase3(int* dist_d, int n, int round){
    __shared__ int s_dist[2][B][B];

    // if blockIdx.y >= round or blockIdx.x >= round, the position of i or j adds B more to move across the pivot row/col 
    int ori_i = blockIdx.y * B + threadIdx.y + ((blockIdx.y >= round) * B);
    int ori_j = blockIdx.x * B + threadIdx.x + ((blockIdx.x >= round) * B);

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

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <time.h>
// #include <cmath>

// #define B 32

// const int INF = ((1 << 30) - 1);

// void input(char* inFileName);
// void output(char* outFileName);

// __global__ void block_FW_Phase1(int* dist_d, int n, int round);
// __global__ void block_FW_Phase2(int* dist_d, int n, int round);
// __global__ void block_FW_Phase3(int* dist_d, int n, int round);

// int ceil_cus(int a, int b){
//     return (a + b - 1) / b;
// }

// int n, m;
// int* Dist;

// int main(int argc, char* argv[]) {

//     // struct timespec start, end, temp;
//     // double time_used;
//     // clock_gettime(CLOCK_MONOTONIC, &start);

//     FILE* file = fopen(argv[1], "rb");

//     fread(&n, sizeof(int), 1, file);
//     fread(&m, sizeof(int), 1, file);

//     size_t matrix_size = n * n * sizeof(int);

//     Dist = (int*)malloc(matrix_size);

//     // printf("n: %d, m: %d\n", n, m);
    
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             if (i == j) {
//                 Dist[i * n + j] = 0;
//             } 
//             else {
//                 Dist[i * n + j] = INF;
//             }
//         }
//     }

//     int pair[3];
//     for (int i = 0; i < m; i++) {
//         fread(pair, sizeof(int), 3, file);
//         Dist[pair[0] * n + pair[1]] = pair[2];
//     }
//     fclose(file);
//     // int B = ceil(sqrt(n));
//     int* dist_d;

//     cudaMalloc(&dist_d, matrix_size);
//     cudaMemcpy(dist_d, Dist, matrix_size, cudaMemcpyHostToDevice);

//     int round = ceil_cus(n, B);

//     // printf("n: %d, m: %d, B: %d, round: %d\n", n, m, B, round);

//     dim3 grid2(round - 1, 2);
//     dim3 grid3(round - 1, round - 1);
//     dim3 block(32, 32);

//     size_t base_smem_size = B * B * sizeof(int);

//     for(int r = 0; r < round; r++){
//         block_FW_Phase1<<<1, block, base_smem_size>>>(dist_d, n, r);
//         // printf("return from kernel 1, round: %d\n", r);
//         block_FW_Phase2<<<grid2, block, 2 * base_smem_size>>>(dist_d, n, r);
//         // printf("return from kernel 2, round: %d\n", r);
//         block_FW_Phase3<<<grid3, block, 2 * base_smem_size>>>(dist_d, n, r);
//         // printf("return from kernel 3, round: %d\n", r);
//     }
    
//     cudaMemcpy(Dist, dist_d, matrix_size, cudaMemcpyDeviceToHost);

//     // for (int i = 0; i < n; i++) {
//     //     for (int j = 0; j < n; j++) {
//     //         printf("%3d ", Dist[i * n + j]);
//     //     }
//     //     printf("\n");
//     // }

//     FILE* outfile = fopen(argv[2], "w");
//     fwrite(Dist, sizeof(int), n*n, outfile);
//     fclose(outfile);

//     // clock_gettime(CLOCK_MONOTONIC, &end);
//     // if ((end.tv_nsec - start.tv_nsec) < 0) {
//     //     temp.tv_sec = end.tv_sec-start.tv_sec-1;
//     //     temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
//     // } else {
//     //     temp.tv_sec = end.tv_sec - start.tv_sec;
//     //     temp.tv_nsec = end.tv_nsec - start.tv_nsec;
//     // }
//     // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    
//     // printf("%f second\n", time_used);
//     free(Dist);
//     cudaFree(dist_d);

//     return 0;
// }

// __global__ void block_FW_Phase1(int* dist_d, int n, int round) {
//     __shared__ int s_dist[B][B];

//     int ori_i = round * B + threadIdx.y;
//     int ori_j = round * B + threadIdx.x;
//     int i = threadIdx.y;
//     int j = threadIdx.x;

//     if(ori_i >= n || ori_j >= n) return;

//     s_dist[i][j] = dist_d[ori_i * n + ori_j];

//     __syncthreads();

//     #pragma unroll 32
//     for (int k = 0; k < B && (k + (round * B)) < n; k++) {
//         if((s_dist[i][k] + s_dist[k][j]) < s_dist[i][j]){
//             s_dist[i][j] = s_dist[i][k] + s_dist[k][j];
//         }
//         __syncthreads();
//     }

//     dist_d[ori_i * n + ori_j] = s_dist[i][j];
// }

// __global__ void block_FW_Phase2(int* dist_d, int n, int round){
//     // for z == 0, it stores things of the block itself
//     // for z == 1, it stores things of the pivot block
//     __shared__ int s_dist[2][B][B];

//     int ori_i, ori_j;

//     if(blockIdx.y == 0){ // pivot row
//         ori_i = round * B + threadIdx.y;
//         ori_j = blockIdx.x * B + threadIdx.x + (blockIdx.x >= round) * B;
//     }
//     else{ // pivot column
//         ori_i = blockIdx.x * B + threadIdx.y + (blockIdx.x >= round) * B;
//         ori_j = round * B + threadIdx.x;
//     }

//     int pivot_i = round * B + threadIdx.y;
//     int pivot_j = round * B + threadIdx.x;

//     int i = threadIdx.y;
//     int j = threadIdx.x;

//     if(!(ori_i >= n || ori_j >= n)){
//         s_dist[0][i][j] = dist_d[ori_i * n + ori_j]; // store value in the pivot row/col blocj itself
//     }

//     // if we return when either pivot_i or pivot_j >= n, we may lose some information of pivot row/col

//     if(pivot_i < n && pivot_j < n){
//         s_dist[1][i][j] = dist_d[pivot_i * n + pivot_j]; // store value in pivot block
//     }
    
//     __syncthreads();

//     if(ori_i >= n || ori_j >= n) return;

//     #pragma unroll 32
//     for (int k = 0; k < B && (k + (round * B)) < n; k++) {
//         if(s_dist[0][i][j] > (s_dist[!blockIdx.y][i][k] + s_dist[blockIdx.y][k][j])){
//             s_dist[0][i][j] = s_dist[!blockIdx.y][i][k] + s_dist[blockIdx.y][k][j];
//         }
//         __syncthreads();
//     }
    
//     dist_d[ori_i * n + ori_j] = s_dist[0][i][j];
// }

// __global__ void block_FW_Phase3(int* dist_d, int n, int round){
//     __shared__ int s_dist[2][B][B];

//     int ori_i = blockIdx.y * B + threadIdx.y + ((blockIdx.y >= round) * B);
//     int ori_j = blockIdx.x * B + threadIdx.x + ((blockIdx.x >= round) * B);
//     // if(blockIdx.x >= round) ori_j += B;
//     // if(blockIdx.y >= round) ori_i += B;

//     int pivot_row_i = round * B + threadIdx.y;
//     int pivot_row_j = ori_j;
//     int pivot_col_i = ori_i;
//     int pivot_col_j = round * B + threadIdx.x;

//     int i = threadIdx.y;
//     int j = threadIdx.x;

//     // if(ori_i >= n || ori_j >= n) return;

//     // if(!(ori_i >= n || ori_j >= n)){
//     //     tmp = dist_d[ori_i * n + ori_j]; // store things of the block itself
//     // }
    
//     if(pivot_row_i < n && pivot_row_j < n){
//         s_dist[0][i][j] = dist_d[pivot_row_i * n + pivot_row_j]; // store things of the pivot row block
//     } 
//     if(pivot_col_i < n && pivot_col_j < n){
//         s_dist[1][i][j] = dist_d[pivot_col_i * n + pivot_col_j]; // store things of the pivot column block
//     }
    
//     __syncthreads();

//     if(ori_i >= n || ori_j >= n) return;

//     int self_dis = dist_d[ori_i * n + ori_j];

//     #pragma unroll 32
//     for (int k = 0; k < B && (k + (round * B)) < n; k++) {
//         int temp = s_dist[1][i][k] + s_dist[0][k][j];
//         if(self_dis > temp){
//             self_dis = temp;
//         }
//         // __syncthreads();
//     }
//     dist_d[ori_i * n + ori_j] = self_dis;
// }