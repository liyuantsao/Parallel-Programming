#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define CHUNK 10

int main(int argc, char** argv) {
	int rc, rank, size;
	rc = MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unsigned long long r = atoll(argv[1]);
	unsigned long long r_square = r * r;
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long result = 0;

	#pragma omp parallel for schedule(guided, CHUNK) reduction(+:pixels)
		for (unsigned long long x = rank; x < r; x+=size) {
			unsigned long long y = ceil(sqrtl(r_square - x*x));
			pixels += y;
		}
		pixels %= k;

	MPI_Reduce(&pixels, &result, size, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	result %= k;
	if(rank == 0){
		printf("%llu\n", (4 * result) % k);
	}
}
