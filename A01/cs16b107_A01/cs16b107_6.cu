#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

__global__ void vecMultiAdd(int* a, int* b, int* c, int numOps)
{
	int threadId, l, r;
	threadId = blockDim.x * blockIdx.x + threadIdx.x;
	l = threadId*numOps;
	r = (threadId+1)*numOps;

	for(; l<r; l++)
		c[l] = a[l] + b[l];
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
		printf("%d %d %d\n", a[i], b[i], c[i]);
}

__host__ float computeTime(int* a, int* b, int* c, int blocks, int threadsPerBlock, int numOps, int iterations)
{
	int i;
	float totalTime = 0, milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(i=0; i<iterations; i++)
	{
		cudaEventRecord(start);
		vecMultiAdd<<<blocks,threadsPerBlock>>>(a, b, c, numOps);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
		totalTime += milliseconds;
	}
	return totalTime/iterations;
}

int main()
{	
	srand(time(NULL));
	int size, i;
	for(i=15; i<=20; i++)
	{
		size = 1<<i;

		int *h_a = (int*)malloc(sizeof(int)*size); generateRandomVector(h_a, size);
		int *h_b = (int*)malloc(sizeof(int)*size); generateRandomVector(h_b, size);
		int *h_c = (int*)malloc(sizeof(int)*size);

		int *d_a; cudaMalloc((void**)&d_a, sizeof(int)*size);
		int *d_b; cudaMalloc((void**)&d_b, sizeof(int)*size);
		int *d_c; cudaMalloc((void**)&d_c, sizeof(int)*size); 

		cudaMemcpy(d_a, h_a, sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(int)*size, cudaMemcpyHostToDevice);

	
		printf("size: (1<<%d), Blocks: (1<<%d), threadsPerBlock: (1<<%d), numOps: (1<<%d), Average Time (ms): %f\n", i, i-7, 7, 0, computeTime(d_a, d_b, d_c, (1<<(i-7)), (1<<7), (1<<0), 100));

		cudaMemcpy(h_c, d_c, sizeof(int)*size, cudaMemcpyDeviceToHost);

		// printResult(h_a, h_b, h_c, size);

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_a);
		free(h_b);
		free(h_c);
	}

	return 0;
}	