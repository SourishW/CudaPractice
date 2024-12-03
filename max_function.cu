#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <limits>
// max over an array function
__global__ void maxKernel(int* a, int * high_out, int N){

    extern __shared__ float max_shared[];
    // __shared__ int max_shared[16*16];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N){
        max_shared[threadIdx.x] = INT_MIN;
    }
    else {
        max_shared[threadIdx.x] = a[index];
    }

    

    int total_length = 16*16;
    __syncthreads();
    while (total_length!= 1){
        int half_length = total_length /2;
        if (threadIdx.x < half_length) {
            max_shared[threadIdx.x] = max(max_shared[threadIdx.x], max_shared[threadIdx.x + half_length]);
        }
        total_length = half_length;
        __syncthreads();
    }
    if (threadIdx.x == 0 && index < N){
        high_out[blockIdx.x] = max_shared[0];
    }    
}

int findMaxNormal(int* array, int N){
    int max = INT_MIN;
    for (int i = 0; i<N; i++){
        max = std::max(array[i], max);
    }
    return max;
}

int findMaxCuda(int* array, int N, int kernel_size){
    // allocation of array
    int arraySize = N * sizeof(int);
    int* a_device;
    cudaMalloc( (void**) &a_device, arraySize);
    cudaMemcpy( a_device, array, arraySize, cudaMemcpyHostToDevice);

    // allocation of high out
    int num_blocks = (N + kernel_size -1) / kernel_size;
    int* high_out = new int[num_blocks];
    int* ho_device;
    int ho_size = num_blocks*sizeof(int);
    cudaMalloc((void **) &ho_device, ho_size);
    cudaMemcpy( ho_device, high_out, ho_size, cudaMemcpyHostToDevice);

    // computation
    dim3 blocks_per_grid(num_blocks);
    dim3 kernels_per_block(kernel_size);
    size_t shared_mem_size = kernels_per_block.x * kernels_per_block.y * sizeof(int) ; 
    maxKernel<<<blocks_per_grid, kernels_per_block, shared_mem_size>>>(a_device, ho_device, N);
    cudaMemcpy(high_out, ho_device, ho_size, cudaMemcpyDeviceToHost);
    
    int answer = findMaxNormal(high_out, num_blocks);

    // cleanup
    cudaFree(a_device);
    cudaFree(ho_device);
    delete [] high_out;
    return answer;
}



void main_helper(int N){
    int* assymetric_array = new int[N];
    int kernel_size = 16*16;
    
    // initialize assymetric array:
    for (int i = 0; i<N; i++){
        assymetric_array[i] = (i*i*(i/3)) % N;
    }
    int normalMax = findMaxNormal(assymetric_array, N);
    int cudaMax = findMaxCuda(assymetric_array, N, kernel_size);
    
    std::string status = (cudaMax == normalMax)? "Good":"Bad";
    std::cout << status << ", answers cuda " << cudaMax <<", normal " << normalMax << std::endl;
    delete [] assymetric_array;
}

int main(){
    for (int i = 10; i< 10000; i ++) {
        main_helper(i);
    }

}