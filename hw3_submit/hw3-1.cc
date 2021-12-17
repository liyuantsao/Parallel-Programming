#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <algorithm> 

const int INF = ((1 << 30) - 1);
const int V = 6666;
void input(char* inFileName);
void output(char* outFileName);
void* compute(void* t);

int n, m, ncpus, k;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);
    
    input(argv[1]);
    
    for(int k = 0; k < n; k++){
        #pragma omp parallel for
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(Dist[i][j] > Dist[i][k] + Dist[k][j]){
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
                }
            }
        }
    }

    output(argv[2]);

    return 0;
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
