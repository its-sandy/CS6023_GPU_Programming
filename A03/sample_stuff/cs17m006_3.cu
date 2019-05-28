#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>

#define MAXWORDS 5000000
#define MAXLEN 10

char h_words[MAXWORDS * (MAXLEN + 1)];  // Additional byte for '\0' character

__global__ void hkernel_1(char *d_words, int *d_lengths, int M, int binCount, int N) {
    // Kernel definition here
}

__global__ void hkernel_2(int *d_lengths, int *d_binValues, int M, int binCount, int N) {
    // Kernel definition here
}

// Function to read input on host
void readInput(int N, char * filename) {
    FILE * ipf = fopen(filename, "r");
    int totalWordCount = 0;
    while (fscanf(ipf, "%s ",  &words[totalWordCount*(MAXLEN + 1)]) != EOF
                && totalWordCount < N
                && totalWordCount < MAXWORDS) {
        totalWordCount += 1;
    }   
    fclose(ipf);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: ./<executable> <M> <N> <input_file_name>\n");
        return 1;
    }

    // Input - start //

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];
    readInput(M, filename);
    
    // Input - end //
    
    // Memory management - start //

    // do memory allocation here
    char *d_words;
    int *d_binValues, *h_binValues, *d_lengths, *h_lengths;
    int binCount = pow(10, N);
    int blocks, threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_total = 0.0, time_temp = 0;

    // memcpy data to device
    cudaMemcpy(d_words, h_words, M * (MAXLEN + 1) * sizeof(char), cudaMemcpyHostToDevice);

    // Memory management - end //

    // Kernel calls - start //

    cudaEventRecord(start);
    hkernel_1<<<blocks, threadsPerBlock>>>(d_words, d_lengths, M, binCount, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time_total += time_temp;

    cudaEventRecord(start);
    hkernel_2<<<blocks, threadsPerBlock>>>(d_lengths, d_binValues, M, binCount, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_temp, start, stop);
    time_total += time_temp;

    // Kernel calls - end //

    printf("time_total : %f\n", time_total);

    // memcpy result from device to host
    cudaMemcpy(h_binValues, d_binValues, sizeof(int) * binCount, cudaMemcpyDeviceToHost);

    // Output result to file
    FILE* opf = fopen("cs17m006_out_3.txt", "w");
    for (int i = 0; i < binCount; i++) {
        // for each of the bins
        //      print 'N' space-separated word lengths
        // then
        fprintf(opf, "%i\n", h_binValues[i]);
    }
    fclose(opf);

    return 0;
}
