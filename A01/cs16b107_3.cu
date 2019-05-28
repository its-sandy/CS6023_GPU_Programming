#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

__global__ void vecAdd(int* a, int* b, int* c)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__host__ void generateRandomVector(int* vec, int size)
{
	int i;

	for(i=0; i<size; i++)
		vec[i] = rand() % (INT_MAX/2); // so that addition does not cause overflow
}

__host__ void printResult(int* a, int* b, int* c, int size)
{
	int i;
	for(i=0; i<size; i++)
		printf("%d\n%d\n%d\n", a[i], b[i], c[i]);
}

int main()
{	
	int size = (1<<15);

	srand(time(NULL));
	int *h_a = (int*)malloc(sizeof(int)*size); generateRandomVector(h_a, size);
	int *h_b = (int*)malloc(sizeof(int)*size); generateRandomVector(h_b, size);
	int *h_c = (int*)malloc(sizeof(int)*size);

	int *d_a; cudaMalloc((void**)&d_a, sizeof(int)*size);
	int *d_b; cudaMalloc((void**)&d_b, sizeof(int)*size);
	int *d_c; cudaMalloc((void**)&d_c, sizeof(int)*size); 

	cudaMemcpy(d_a, h_a, sizeof(int)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int)*size, cudaMemcpyHostToDevice);

	vecAdd<<<(1<<7),(1<<8)>>>(d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost);

	printResult(h_a, h_b, h_c, size);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}