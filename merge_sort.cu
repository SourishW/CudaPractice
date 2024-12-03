#include <limits>
#include <iostream>
__device__ void merge(int* array, int N, int* result){
    int* arrayl = array;
    int* arrayr = &array[N/2];



    int l = 0;
    int r = 0;
    int result_i = 0;
    int half_length = N /2;
    
    while (l != half_length || r != half_length){
        if (r == half_length || arrayl[l] < arrayr[r]){
            result[result_i] = arrayl[l];
            l++;            
        }
        else {
            result[result_i] = arrayr[r];
            r++;
        }
        result_i++;
    }
    return;

}

__global__ void merge_sort(int* array, int* out, int N){
    extern __shared__ int shared_allocated[];

    int* local_array = &shared_allocated[0];
    int* scratch_pad = &shared_allocated[blockDim.x];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    
    if (index < N){
        local_array[tidx] = array[index];
    }
    else {
        local_array[tidx] = INT_MAX;
    }
    
    __syncthreads();
    for (int sort_size = 2; sort_size!= 2*blockDim.x; sort_size*=2){
        if (tidx % sort_size == 0 ){
            // need to sort from tid to tid + sort_size (end not included)
            merge(&local_array[tidx], sort_size, &scratch_pad[tidx]);
        }
        __syncthreads();
        int* temp = scratch_pad;
        scratch_pad = local_array;
        local_array = temp;
        __syncthreads();
    }

    if (index < N){
        out[index] = local_array[tidx];
    }
    __syncthreads();
}

int main(){
    int N = 8*8*8;
    int* a_host = new int[N];
    int* out_host = new int[N];
    size_t array_size = N * sizeof(int);

    std::cout << "Before a:" << std::endl;
    for (int i = 0; i< N; i++){
        a_host[i] = N -i-1;
        std::cout << a_host[i] << "," ;
    }
    std::cout << std::endl;

    int* a_device;
    int* out_device;
    cudaMalloc((void**) &a_device, array_size);
    cudaMalloc((void**) &out_device, array_size);
    cudaMemcpy(a_device, a_host, array_size, cudaMemcpyHostToDevice);
    dim3 blocks_per_grid(1);
    dim3 threads_per_block(N);
    merge_sort<<<blocks_per_grid, threads_per_block, N*2*sizeof(int)>>>(a_device, out_device, N);
    cudaMemcpy(out_host, out_device, array_size, cudaMemcpyDeviceToHost);

    std::cout << "After a:" << std::endl;
    for (int i = 0; i<N; i++){
        std::cout << out_host[i] << ",";
    }
    std::cout << std::endl;
    
    delete[] a_host;
    delete[] out_host;
    cudaFree(a_device);
    cudaFree(out_device);


}