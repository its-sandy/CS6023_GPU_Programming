#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

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

__global__ void kernel2(double* d_a, double* d_b, double* d_c, int p, int q, int r)
{
    // fastest varying index is y
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	double res = 0;
    for(int i=0; i<q; i++)
        res += d_a[row*q + i]*d_b[i*r + col];
    d_c[row*r + col] = res;
}

int main()
{	
	srand(time(NULL));
    int p = 4096, q = 8192, r = 16384; // matrix A is pxq, matrix B is qxr

    double *h_a = (double*)malloc(sizeof(double)*p*q); fill_matrix(h_a, p, q);
    double *h_b = (double*)malloc(sizeof(double)*q*r); fill_matrix(h_b, q, r);
    double *h_c = (double*)malloc(sizeof(double)*p*r);

    double *d_a; cudaMalloc((void**)&d_a, sizeof(double)*p*q);
    double *d_b; cudaMalloc((void**)&d_b, sizeof(double)*q*r);
    double *d_c; cudaMalloc((void**)&d_c, sizeof(double)*p*r); 

    cudaMemcpy(d_a, h_a, sizeof(double)*p*q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double)*q*r, cudaMemcpyHostToDevice);

    float milliseconds;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel2<<<dim3(r/16,p/16),dim3(16,16)>>>(d_a, d_b, d_c, p, q, r);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel 2  (fastest varying index as .y) elapsed time = %f ms\n", milliseconds);

    cudaMemcpy(h_c, d_c, sizeof(double)*p*r, cudaMemcpyDeviceToHost);
    print_matrix_to_file(h_c, p, r);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

	return 0;
}