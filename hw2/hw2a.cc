#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>
#include <time.h>


int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int ncpus;

/* allocate memory for image */
int* image;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* compute(void* t){
    int* tid = (int*)t;

    for (int j = *tid; j < height; j += ncpus) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; i += 2) {
            if(i == width - 1){
                double x0 = i * ((right - left) / width) + left;

                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                double x_square = 0;
                double y_square = 0;
                while (repeats < iters && length_squared < 4) {
                    y = 2 * x * y + y0;
                    x = x_square - y_square + x0;
                    x_square = x * x;
                    y_square = y * y;
                    length_squared = x_square + y_square;
                    ++repeats;
                }
                image[j * width + i] = repeats;
            }
            else{
                __m128d x0;
                x0[0] = i * ((right - left) / width) + left;
                x0[1] = (i+1) * ((right - left) / width) + left;
                
                int repeat0 = 0;
                int repeat1 = 0;
                int flag0 = 0;
                int flag1 = 0;
                // int repeats1 = 0;
                
                __m128d x;
                __m128d y;
                __m128d length_squared;
                __m128d x_square;
                __m128d y_square;
                __m128d two;
                __m128d Y0;
                
                x = _mm_set_pd(0.0, 0.0);
                y = _mm_set_pd(0.0, 0.0);
                length_squared = _mm_set_pd(0.0, 0.0);
                x_square = _mm_set_pd(0.0, 0.0);
                y_square = _mm_set_pd(0.0, 0.0);
                two = _mm_set_pd(2.0, 2.0);
                Y0 = _mm_set_pd(y0, y0);

                while((repeat0 < iters && !flag0) || (repeat1 < iters && !flag1)){
                    if(length_squared[0] >= 4 && !flag0){
                        image[j * width + i] = repeat0;
                        flag0 = 1;
                    }
                    if(length_squared[1] >= 4 && !flag1){
                        image[j * width + (i + 1)] = repeat1;
                        flag1 = 1;
                    }
                    y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), two), Y0);
                    x = _mm_add_pd(_mm_sub_pd(x_square, y_square), x0);
                    x_square = _mm_mul_pd(x, x);
                    y_square = _mm_mul_pd(y, y);
                    length_squared = _mm_add_pd(x_square, y_square);
                    repeat0++;
                    repeat1++;
                    // y = 2 * x * y + y0;
                    // x = x_square - y_square + x0;
                    // x_square = x * x;
                    // y_square = y * y;
                    // length_squared = x_square + y_square;
                    // ++repeats;
                }
                // image[j * width + i] = repeats;
            }
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    time_t time_start;
    time_t time_end;
    time(&time_start);
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[ncpus];
    int rc;
	int ID[ncpus];

    for (int t = 0; t < ncpus; t++) {
		ID[t] = t;
        rc = pthread_create(&threads[t], NULL, compute, (void*)&ID[t]);
    }

    for(int i=0; i<ncpus; i++){
		pthread_join(threads[i], NULL);
	}
    /* mandelbrot set */
    // for (int j = 0; j < height; ++j) {
    //     double y0 = j * ((upper - lower) / height) + lower;
    //     for (int i = 0; i < width; ++i) {
    //         double x0 = i * ((right - left) / width) + left;

    //         int repeats = 0;
    //         double x = 0;
    //         double y = 0;
    //         double length_squared = 0;
    //         while (repeats < iters && length_squared < 4) {
    //             double temp = x * x - y * y + x0;
    //             y = 2 * x * y + y0;
    //             x = temp;
    //             length_squared = x * x + y * y;
    //             ++repeats;
    //         }
    //         image[j * width + i] = repeats;
    //     }
    // }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    time(&time_end);
    // printf("elapsed time: %f\n", difftime(time_end, time_start));
    pthread_exit(NULL);
}
