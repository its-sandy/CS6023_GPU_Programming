#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAXWORDS 5000000

char *h_words; //word length capped to 10

void readInput(int M, char *filename)
{
    FILE *ipf = fopen(filename, "r");
    int totalWordCount = 0;
    while(totalWordCount < M && totalWordCount < MAXWORDS && fscanf(ipf, "%s", &h_words[totalWordCount*11]) != EOF)
    {
        totalWordCount++;
    }
    fclose(ipf);
}

__global__ void computeLengths(char *d_words, int *d_lengths, int M)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < M)
    {
        for(int i=0; i<11; i++)
            if(d_words[id*11 + i] == '\0')
            {
                d_lengths[id] = i;
                break;
            }
    }
}

__global__ void computeNCountGrams(int *d_lengths, int *d_bins, int M, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = 0;
    if(id < M-N+1)
    {
        for(int i=0; i<N; i++)
            bid = bid*10 + (d_lengths[id+i]-1); //we use -1 so that range becomes [0,9]
        atomicAdd(d_bins+bid, 1);
    }
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];
    int bins_size = pow(10,N);

    float milliseconds;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    h_words = (char*)malloc(sizeof(char)*11*M);
    readInput(M, filename);

    char *d_words; cudaMalloc((void**)&d_words, sizeof(char)*11*M);
    cudaMemcpy(d_words, h_words, sizeof(char)*11*M, cudaMemcpyHostToDevice);

    int *d_lengths; cudaMalloc((void**)&d_lengths, sizeof(int)*M);

    int *h_bins = (int*)malloc(sizeof(int)*bins_size);
    int *d_bins; cudaMalloc((void**)&d_bins, sizeof(int)*bins_size);    

    for(int i=10; i>=5; i--)
    {
        float total_time = 0; 
        for(int j=0; j<20; j++)
        {
            cudaMemset(d_bins, 0, sizeof(int)*bins_size);

            cudaEventRecord(start);
            computeLengths<<<(M+(1<<i)-1)/(1<<i), (1<<i)>>>(d_words, d_lengths, M);
            computeNCountGrams<<<(M+N-1 +(1<<i)-1)/(1<<i), (1<<i)>>>(d_lengths, d_bins, M, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;
        }
        printf("Threads per Block = %d, elapsed time = %f ms\n",(1<<i), total_time/20);
    }

    cudaMemcpy(h_bins, d_bins, sizeof(int)*bins_size, cudaMemcpyDeviceToHost);
    
    FILE* fp = fopen("cs16b107_out_2.txt","w");
    for(int i=0; i<bins_size; i++)
    {
        for(int j=N-1; j>=0; j--)
            fprintf(fp,"%d ", 1 + (i/((int)pow(10,j)))%10);
        fprintf(fp,"%d\n", h_bins[i]);
    }
    fclose(fp);

    cudaFree(d_bins);
    cudaFree(d_lengths);
    cudaFree(d_words);
    free(h_bins);
    free(h_words);
    return 0;
}