#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
__global__ void vectorAddKernel(int* A, int* B, int*C, int N){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        C[index] = A [index] + B[index];
    }
    printf("Thread %d, BlockIdx %d, Dim %d: index: %d\n A: %d, B: %d, C: %d\n", threadIdx.x, blockIdx.x, blockDim.x, index, A[index], B[index], C[index]);
    
}

int main() {
    int N = 1024;
    int size = N * sizeof(int);

    int* A,* B, *C;
    int *d_A, *d_B, *d_C;

    A = new int[N];
    B = new int[N];
    C = new int[N];

    for (int i = 0; i< N; i++){
        A[i] = i;
        B[i] = N-i;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i< N; i++){
        if (C[i] != N){
            std::cout << "Problem: "<< C[i] <<" is not " << N << std::endl;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;

}
