#include <iostream>

__global__ void mapKernel(int* A, int c, int* out, int N){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N){
        out[index] = A[index] + c;
    }   

}

int main(){
    int N = 1024;
    int* a = new int[N];
    int* out = new int[N];
    
    int arrays_size = N * sizeof(int);
    for (int i = 0; i< N; i++){
        a[i] = i;
    }
    int c = 1;

    int* a_device;
    int* out_device;
    cudaMalloc((void**)&a_device, arrays_size);
    cudaMalloc((void**)&out_device, arrays_size);

    cudaMemcpy(a_device, a, arrays_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 16* 16;
    int numKernelsNeeded = (N+threadsPerBlock-1) / threadsPerBlock;


    mapKernel<<<numKernelsNeeded, threadsPerBlock>>>(a_device, c, out_device, N);

    cudaMemcpy(out, out_device, arrays_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; i++){
        if (a[i]+ c != out[i]){
            std::cout << "A[" << i <<"] = " << a[i] << ", Out[" << i <<"] = " << out[i] << std::endl;

        }
        else {
            std::cout << "y" ;
        }
        
    }
    std::cout <<std::endl;

    cudaFree(a_device);
    cudaFree(out_device);
    free(a);
    free(out);
    return 0;
}