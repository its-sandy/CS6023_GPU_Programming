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

__global__ void computeNCountGrams(int *d_lengths, int *d_bins, int bins_size, int shared_bins_size, int M, int N)
{
    extern __shared__ int shared_bins[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int groupid = threadIdx.x/8;
    int bid = 0, i, j, len, shared_bid, local_bid = 0;
    //shared_bins_size is per group of 8 threads

    //initialization
    for(shared_bid=threadIdx.x; shared_bid<shared_bins_size*(256/8); shared_bid+=blockDim.x)
        shared_bins[shared_bid] = 0;
    __syncthreads();

    bool shared_update = true;
    //update local bins
    if(id < M-N+1)
    {
        for(i=0; i<N; i++)
        {
            len = d_lengths[id+i];
            bid = bid*10 + (len-1); //we use -1 so that range becomes [0,9]
            local_bid = local_bid*3 + (len-2); //we use -2 so that range becomes [0,2]
            if(len < 2 || len > 4)
                shared_update = false;
        }
        if(shared_update)
            atomicAdd(shared_bins + groupid*shared_bins_size + local_bid, 1);
        else
            atomicAdd(d_bins+bid, 1);
    }
    __syncthreads();

    //update global bins
    for(shared_bid=threadIdx.x; shared_bid<shared_bins_size*(256/8); shared_bid+=blockDim.x) //uses many threads
    {
        local_bid = shared_bid%shared_bins_size;
        i = shared_bins_size; // pow(3,N);
        bid = 0;
        for(j=0; j<N; j++)
        {
            i/=3;
            bid = bid*10 + ((local_bid/i)%3 +2-1);
        }
        atomicAdd(d_bins+bid, shared_bins[shared_bid]);
    }         
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];
    int bins_size = pow(10,N);
    int shared_bins_size = pow(3,N);

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

    float total_time = 0; 
    for(int j=0; j<20; j++)
    {
        // 256 threads per block
        cudaMemset(d_bins, 0, sizeof(int)*bins_size);
        cudaEventRecord(start);
        computeLengths<<<(M+255)/256, 256>>>(d_words, d_lengths, M);
        computeNCountGrams<<<(M+N-1 + 255)/256, 256, sizeof(int)*shared_bins_size*(256/8)>>>(d_lengths, d_bins, bins_size, shared_bins_size, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    printf("Threads per Block = 256, elapsed time = %f ms\n", total_time/20);

    cudaMemcpy(h_bins, d_bins, sizeof(int)*bins_size, cudaMemcpyDeviceToHost);
    
    FILE* fp = fopen("cs16b107_out_5.txt","w");
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