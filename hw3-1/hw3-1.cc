#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
// #include "mpi.h"

const int INF = ((1 << 30) - 1);
const int V = 6666;
void input(char* inFileName);
void output(char* outFileName);
void* compute(void* t);

int n, m, ncpus, k;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    // int rc, rank, size;
	// rc = MPI_Init(&argc, &argv);
	// MPI_Comm_size(MPI_COMM_WORLD, &size);
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    // printf("ncpus: %d\n", ncpus);

    // double start, end;
    
    input(argv[1]);

    // start = MPI_Wtime();

    pthread_t threads[ncpus];
    int rc2;
	int ID[ncpus];


    for(k = 0; k < n; k++){
        for (int t = 0; t < ncpus; t++) {
            ID[t] = t;
            rc2 = pthread_create(&threads[t], NULL, compute, (void*)&ID[t]);
        }
        for(int i = 0; i < ncpus; i++){
            pthread_join(threads[i], NULL);
        }
    }
    
    // for(int k = 0; k < n; k++){
    //     #pragma omp parallel for
    //     for(int i = 0; i < n; i++){
    //         for(int j = 0; j < n; j++){
    //             if(Dist[i][j] > Dist[i][k] + Dist[k][j]){
    //                 Dist[i][j] = Dist[i][k] + Dist[k][j];
    //             }
    //         }
    //     }
    // }

    // end = MPI_Wtime();
    // printf("time: %f\n", end - start);

    output(argv[2]);

    pthread_exit(NULL);
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void* compute(void *t){
    int* tid = (int*)t;

    for(int i = *tid; i < n; i+=ncpus){
        for(int j = 0; j < n; j++){
            if(Dist[i][j] > Dist[i][k] + Dist[k][j]){
                Dist[i][j] = Dist[i][k] + Dist[k][j];
            }
        }
    }

    pthread_exit(NULL);
}