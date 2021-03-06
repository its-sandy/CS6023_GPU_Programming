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

__global__ void matrix_multiplication(double* d_a, double* d_b, double* d_c, int n)
{
    // fastest varying index is y
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	double res = 0;
    for(int i=0; i<n; i++)
        res += d_a[row*n + i]*d_b[i*n + col];
    d_c[row*n + col] = res;
}

int main()
{	
	srand(time(NULL));
	int size, n, i;
    n = 2048; // 8192 = 1<<13
    size = n*n;

    float milliseconds;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    double *h_a = (double*)malloc(sizeof(double)*size); fill_matrix(h_a, n, n);
    double *h_b = (double*)malloc(sizeof(double)*size); fill_matrix(h_b, n, n);
    double *h_c = (double*)malloc(sizeof(double)*size);

    double *d_a; cudaMalloc((void**)&d_a, sizeof(double)*size);
    double *d_b; cudaMalloc((void**)&d_b, sizeof(double)*size);
    double *d_c; cudaMalloc((void**)&d_c, sizeof(double)*size); 

    cudaMemcpy(d_a, h_a, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(double)*size, cudaMemcpyHostToDevice);

    /* we try
    (1<<11, 1<<11), (1<<2, 1<<2)
    (1<<11, 1<<10), (1<<2, 1<<3)
    (1<<10, 1<<10), (1<<3, 1<<3)
    (1<<10, 1<<9), (1<<3, 1<<4)
    (1<<9, 1<<9), (1<<4, 1<<4)
    (1<<9, 1<<8), (1<<4, 1<<5)
    (1<<8, 1<<8), (1<<5, 1<<5)

    (1<<9, 1<<7), (1<<4, 1<<6)
    */

    printf("Run Times for n = %d\n",n);
    for(i=4; i<=10 /*change according to half of max threads per block*/; i++)
	{
        cudaEventRecord(start);
        matrix_multiplication<<<dim3(n/(1<<(i/2)),n/(1<<((i+1)/2))),dim3(1<<(i/2),1<<((i+1)/2))>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Grid Structure: (%d, %d); Block Structure: (%d, %d); Time (ms): %f\n", n/(1<<(i/2)), n/(1<<((i+1)/2)), 1<<(i/2), 1<<((i+1)/2), milliseconds);
	}

    cudaEventRecord(start);
    matrix_multiplication<<<dim3(n/(1<<4),n/(1<<6)),dim3(1<<4,1<<6)>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Grid Structure: (%d, %d); Block Structure: (%d, %d); Time (ms): %f\n", n/(1<<4), n/(1<<6), 1<<4, 1<<6, milliseconds);


    cudaMemcpy(h_c, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost);
    // print_matrix_to_file(h_c, n, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

	return 0;
}