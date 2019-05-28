#include <stdio.h>
#include <stdlib.h>

void print_matrix_to_file(int *mat, unsigned numRows, unsigned numCols) 
{ 
    const char *fname = "assignment2_out"; 
    FILE *f = fopen(fname, "w"); 
 
    for(unsigned i=0; i < numRows; i++) 
    { 
        for(unsigned j=0; j < numCols; j++) 
            fprintf(f,"%d ", mat[i*numCols + j]); 
        fprintf(f,"\n"); 
    } 
    fclose(f); 
}

__global__ void matrix_multiplication(int *m1, int *m2, int *result, int N) 
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if(row < N && col < N)
    {
        int res = 0;
        for(int i=0; i<N; i++)
            res += m1[row*N + i]*m2[i*N + col];
        result[row*N + col] = res;
    }
}

int main() 
{
    // Taking input graph
    int N, M;
    printf("Enter N:\n");
    scanf("%d", &N);

    int *h_adj;
    int memsize = N * N * sizeof(int);
    h_adj = (int *)malloc(memsize);
    memset(h_adj, memsize, 0);

    printf("Enter M:\n");
    scanf("%d", &M);
    printf("Enter u,v pairs:\n");
    for (int i = 0; i < M; i++)
    {
        int u, v;
        scanf("%d %d", &u, &v);
        h_adj[(u-1)*N + v-1] = h_adj[(v-1)*N + u-1] = 1;
    }

    printf("Computing\n");
    int *d_a; cudaMalloc((void**)&d_a, memsize);
    int *d_b; cudaMalloc((void**)&d_b, memsize);
    int *d_c; cudaMalloc((void**)&d_c, memsize);

    cudaMemcpy(d_a, h_adj, memsize, cudaMemcpyHostToDevice);

    matrix_multiplication<<<dim3(N/16,N/16),dim3(16,16)>>>(d_a, d_a, d_b, N);
    matrix_multiplication<<<dim3(N/16,N/16),dim3(16,16)>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_adj, d_c, memsize, cudaMemcpyDeviceToHost);
    // print_matrix_to_file(h_adj, N, N);

    int answer = 0;
    for(int i=0; i<N; i++)
        answer += h_adj[i*N + i];
    printf("Number of 3-cliques = %d\n", answer/6);
    return 0;
}