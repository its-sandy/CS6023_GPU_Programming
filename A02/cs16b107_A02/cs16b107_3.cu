#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#define N 32
#define BLOCK_WIDTH 16

void fill_matrix(double *mat, unsigned numRows, unsigned numCols) 
{ 
    for(unsigned i=0; i < numRows; i++) 
        for(unsigned j=0; j < numCols; j++)    
            mat[i*numCols + j] = i*2.1f + j*3.2f;    
}

void print_matrix_to_file(double *mat, unsigned numRows, unsigned numCols) 
{ 
    const char *fname = "assignment2_out"; 
    FILE *f = fopen(fname, "w"); 
 
    for(unsigned i=0; i < numRows; i++) 
    { 
        for(unsigned j=0; j < numCols; j++) 
            fprintf(f,"%4.4f ", mat[i*numCols + j]); 
        fprintf(f,"\n"); 
    } 
    fclose(f); 
}

__global__ void matmul(double* d_a, double* d_b, double* d_c)
{
    // fastest varying index is y

    __shared__ double ds_a[BLOCK_WIDTH][N];
    __shared__ double ds_b[N][BLOCK_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int p, blocksperdim = N/blockDim.x;

    for(p=0; p<blocksperdim; p++)
    {
        ds_a[ty][p*blockDim.x + tx] = d_a[row*N + p*blockDim.x + tx];
        ds_b[p*blockDim.y + ty][tx] = d_b[(p*blockDim.y + ty)*N + col];
    }
    __syncthreads();

	double res = 0;
    for(p=0; p<N; p++)
        res += ds_a[ty][p]*ds_b[p][tx];
    d_c[row*N + col] = res;
}

int main()
{	
	srand(time(NULL));
	int size = N*N;

    float milliseconds;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    double *h_a = (double*)malloc(sizeof(double)*size); fill_matrix(h_a, N, N);
    double *h_b = (double*)malloc(sizeof(double)*size); fill_matrix(h_b, N, N);
    double *h_c = (double*)malloc(sizeof(double)*size);

    double *d_a; cudaMalloc((void**)&d_a, sizeof(double)*size);
    double *d_b; cudaMalloc((void**)&d_b, sizeof(double)*size);
    double *d_c; cudaMalloc((void**)&d_c, sizeof(double)*size); 

    cudaMemcpy(d_a, h_a, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double)*size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    matmul<<<dim3(N/16,N/16),dim3(16,16)>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time = %f ms\n", milliseconds);

    cudaMemcpy(h_c, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost);
    print_matrix_to_file(h_c, N, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

	return 0;
}