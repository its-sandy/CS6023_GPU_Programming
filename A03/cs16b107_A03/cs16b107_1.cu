#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];

    float milliseconds;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    h_words = (char*)malloc(sizeof(char)*11*M);
    readInput(M, filename);

    char *d_words; cudaMalloc((void**)&d_words, sizeof(char)*11*M);
    cudaMemcpy(d_words, h_words, sizeof(char)*11*M, cudaMemcpyHostToDevice);

    int *h_lengths = (int*)malloc(sizeof(int)*M);
    int *d_lengths; cudaMalloc((void**)&d_lengths, sizeof(int)*M);

    for(int i=10; i>=5; i--)
    {
        float total_time = 0; 
        for(int j=0; j<20; j++)
        {
            cudaEventRecord(start);
            computeLengths<<<(M+(1<<i)-1)/(1<<i), (1<<i)>>>(d_words, d_lengths, M);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;
        }
        printf("Threads per Block = %d, elapsed time = %f ms\n",(1<<i), total_time/20);
    }

    cudaMemcpy(h_lengths, d_lengths, sizeof(int)*M, cudaMemcpyDeviceToHost);
    FILE* fp = fopen("cs16b107_out_1.txt","w");
    for(int i=0; i<M; i++)
    {
        fprintf(fp,"%d\n",h_lengths[i]);
    }
    fclose(fp);
    
    cudaFree(d_lengths);
    cudaFree(d_words);
    free(h_lengths);
    free(h_words);
    return 0;
}