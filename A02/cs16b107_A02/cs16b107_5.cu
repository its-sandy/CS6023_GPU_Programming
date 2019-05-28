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

__global__ void matmul(double* d_a, double* d_b, double* d_c, int n, int tile_width)
{
    // fastest varying index is y

    extern __shared__ double ds[];
    double *ds_a = ds;
    double *ds_b = &ds_a[tile_width*tile_width];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double res = 0;
    for(int p=0; p<n/tile_width; p++)
    {
        ds_a[ty*tile_width + tx] = d_a[row*n + p*tile_width + tx];
        ds_b[ty*tile_width + tx] = d_b[(p*tile_width + ty)*n + col];
        __syncthreads();

        for(int i=0; i<tile_width; i++)
            res += ds_a[ty*tile_width + i]*ds_b[i*tile_width + tx];
        __syncthreads();
    }

    d_c[row*n + col] = res;
}

int main()
{	
	srand(time(NULL));
    int n = 8192;
	int size = n*n;

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

    for(int tile_width = 4; tile_width<=32; tile_width*=2)
    {
        cudaEventRecord(start);
        matmul<<<dim3(n/tile_width,n/tile_width), dim3(tile_width,tile_width), 2*sizeof(double)*tile_width*tile_width>>>(d_a, d_b, d_c, n, tile_width);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("tile size = %dx%d; elapsed time = %f ms\n", tile_width, tile_width, milliseconds);
    }

    cudaMemcpy(h_c, d_c, sizeof(double)*size, cudaMemcpyDeviceToHost);
    print_matrix_to_file(h_c, n, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

	return 0;
}