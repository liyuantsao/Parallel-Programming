#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int INF = ((1 << 30) - 1);
// const int V;
void input(char* inFileName);
void output(char* outFileName);

__global__ void block_FW(int B, int* dist_d, int n);

__host__ __device__ int ceil(int a, int b){
    return (a + b - 1) / b;
}
__device__ void cal(int B, int* dist_d, int n, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
int* Dist;

int thread = 128;

int main(int argc, char* argv[]) {
    // input(argv[1]);
    FILE* file = fopen(argv[1], "rb");

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    int matrix_size = n * n * sizeof(int);

    Dist = (int*)malloc(matrix_size);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
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

    int B = ceil(n, 3);
    int* dist_d;

    cudaMalloc(&dist_d, matrix_size);
    cudaMemcpy(dist_d, Dist, matrix_size, cudaMemcpyHostToDevice);

    int grid_dim = ceil(n, B);

    dim3 grid(grid_dim, grid_dim);
    dim3 block(thread, 1);

    block_FW<<<grid, block>>>(B, dist_d, n);
    cudaMemcpy(Dist, dist_d, matrix_size, cudaMemcpyDeviceToHost);

    FILE* outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        }
        fwrite(&Dist[i * n], sizeof(int), n, outfile);
    }
    fclose(outfile);

    return 0;
}

// void input(char* infile) {
//     FILE* file = fopen(infile, "rb");
//     fread(&n, sizeof(int), 1, file);
//     fread(&m, sizeof(int), 1, file);

//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             if (i == j) {
//                 Dist[i][j] = 0;
//             } else {
//                 Dist[i][j] = INF;
//             }
//         }
//     }

//     int pair[3];
//     for (int i = 0; i < m; ++i) {
//         fread(pair, sizeof(int), 3, file);
//         Dist[pair[0]][pair[1]] = pair[2];
//     }
//     fclose(file);
// }

// void output(char* outFileName) {
//     FILE* outfile = fopen(outFileName, "w");
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             if (Dist[i][j] >= INF) Dist[i][j] = INF;
//         }
//         fwrite(Dist[i], sizeof(int), n, outfile);
//     }
//     fclose(outfile);
// }

// int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void block_FW(int B, int* dist_d, int n) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        cal(B, dist_d, n, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, dist_d, n, r, r, 0, r, 1);
        cal(B, dist_d, n, r, r, r + 1, round - r - 1, 1);
        cal(B, dist_d, n, r, 0, r, 1, r);
        cal(B, dist_d, n, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, dist_d, n, r, 0, 0, r, r);
        cal(B, dist_d, n, r, 0, r + 1, round - r - 1, r);
        cal(B, dist_d, n, r, r + 1, 0, r, round - r - 1);
        cal(B, dist_d, n, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

__device__ void cal(
    int B, int* dist_d, int n, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (dist_d[i*n + k] + dist_d[k*n + j] < dist_d[i*n + j]) {
                            dist_d[i*n + j] = dist_d[i*n + k] + dist_d[k*n + j];
                        }
                    }
                }
            }
        }
    }
}
