/**
 *
 * CENG305 Homework-1
 *
 * Histogram equalization with pthreads
 *
 * Usage:  main <input.jpg> <numthreads> 
 *
 * @author  Selcuk Ustun
 *
 * @version 1.1, 02 November 2024
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

// Global Variables - Do not touch them
int hist[256];
int cumhistogram[256];
int alpha[256];
pthread_mutex_t mutex_hist;

void seq_histogram_equalizer(uint8_t* rgb_image, int width, int height);
void par_histogram_equalizer(uint8_t* rgb_image, int width, int height);
void *thread_histogram(void *arg);
void *thread_apply_equalization(void *arg);

uint8_t* rgb_image;
int width, height, num_threads;

int main(int argc, char* argv[]) 
{		
    int bpp;
    clock_t start, end;
    double elapsed_time;

    num_threads = atol(argv[2]);

    rgb_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);
    uint8_t* rgb_image_for_par = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);

    printf("Width: %d  Height: %d \n", width, height);
    printf("Input: %s , threads: %d \n", argv[1], num_threads);

    start = clock();
    seq_histogram_equalizer(rgb_image, width, height);
    end = clock();
    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Sequential elapsed time: %lf ms\n", elapsed_time);

    stbi_write_jpg("sequential_output.jpg", width, height, CHANNEL_NUM, rgb_image, 100);

    start = clock();
    par_histogram_equalizer(rgb_image_for_par, width, height);
    end = clock();
    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Multi-thread elapsed time with %d threads: %lf ms\n", num_threads, elapsed_time);

    stbi_write_jpg("multithread_output.jpg", width, height, CHANNEL_NUM, rgb_image_for_par, 100);

    stbi_image_free(rgb_image);
    stbi_image_free(rgb_image_for_par);
    return 0;
}

void par_histogram_equalizer(uint8_t* rgb_image, int width, int height) {
    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    pthread_mutex_init(&mutex_hist, NULL);

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_histogram, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex_hist);

    cumhistogram[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cumhistogram[i] = hist[i] + cumhistogram[i - 1];
    }

    for (int i = 0; i < 256; i++) {
        alpha[i] = round((double)cumhistogram[i] * (255.0 / (width * height)));
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, thread_apply_equalization, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void *thread_histogram(void *arg) {
    int thread_id = *(int *)arg;
    int start_row = (height / num_threads) * thread_id;
    int end_row = (thread_id == num_threads - 1) ? height : start_row + (height / num_threads);
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            pthread_mutex_lock(&mutex_hist);
            hist[rgb_image[i * width + j]]++;
            pthread_mutex_unlock(&mutex_hist);
        }
    }
    return NULL;
}

void *thread_apply_equalization(void *arg) {
    int thread_id = *(int *)arg;
    int start_row = (height / num_threads) * thread_id;
    int end_row = (thread_id == num_threads - 1) ? height : start_row + (height / num_threads);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            rgb_image[i * width + j] = alpha[rgb_image[i * width + j]];
        }
    }
    return NULL;
}

void seq_histogram_equalizer(uint8_t* rgb_image, int width, int height) {			
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            hist[rgb_image[i * width + j]]++;
        }
    }	
    double size = width * height;
   
    cumhistogram[0] = hist[0];
    for(int i = 1; i < 256; i++) {
        cumhistogram[i] = hist[i] + cumhistogram[i - 1];
    }    
	    
    for(int i = 0; i < 256; i++) {
        alpha[i] = round((double)cumhistogram[i] * (255.0 / size));
    }
			
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            rgb_image[y * width + x] = alpha[rgb_image[y * width + x]];
        }
    }
}
