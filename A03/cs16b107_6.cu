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

__global__ void computeNCountGrams(int *d_lengths, int *d_bins, int bins_size, int local_bins_size, int M, int N)
{
    extern __shared__ int shared_bins[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid, i, j, len, local_bid;
    int total_num_threads = gridDim.x*blockDim.x;

    int *local_bins = shared_bins + threadIdx.x*local_bins_size; //pointer
    
    //initialization
    for(i=0; i<local_bins_size; i++)
        local_bins[i] = 0;

    bool shared_update = true;
    //update local bins
    for(j=id; j<M-N+1; j += total_num_threads)
    {
        bid = local_bid = 0;
        for(i=0; i<N; i++)
        {
            len = d_lengths[j+i];
            bid = bid*10 + (len-1); //we use -1 so that range becomes [0,9]
            local_bid = local_bid*4 + (len-2); //we use -2 so that range becomes [0,3]
            if(len < 2 || len > 5)
                shared_update = false;
        }
        if(shared_update)
            atomicAdd(local_bins+local_bid, 1);
        else
            atomicAdd(d_bins+bid, 1);
    }

    //update global bins
    for(local_bid=0; local_bid<local_bins_size; local_bid++)
    {
        i = local_bins_size; // pow(4,N);
        bid = 0;
        for(j=0; j<N; j++)
        {
            i/=4;
            bid = bid*10 + ((local_bid/i)%4 +2-1);
        }
        atomicAdd(d_bins+bid, local_bins[local_bid]);
    }         
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];
    int bins_size = pow(10,N);
    int local_bins_size = pow(4,N);

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

    int ops_per_thread, min_num_threads; 

    float total_time = 0; 
    for(int j=0; j<20; j++)
    {
        ops_per_thread = 1000;
        min_num_threads = (M+N-1 +ops_per_thread-1)/ops_per_thread;
        // 32 threads per block
        cudaMemset(d_bins, 0, sizeof(int)*bins_size);
        cudaEventRecord(start);
        computeLengths<<<(M+255)/256, 256>>>(d_words, d_lengths, M);
        computeNCountGrams<<<(min_num_threads+31)/32, 32, sizeof(int)*local_bins_size*32>>>(d_lengths, d_bins, bins_size, local_bins_size, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    printf("Threads per Block = 32, elapsed time = %f ms\n", total_time/20);

    cudaMemcpy(h_bins, d_bins, sizeof(int)*bins_size, cudaMemcpyDeviceToHost);
    
    FILE* fp = fopen("cs16b107_out_6.txt","w");
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