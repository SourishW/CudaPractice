#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void mapKernel2D(int* a, int c, int* out, int xlen, int ylen){
    int x_index = threadIdx.x + blockIdx.x * blockDim.x;
    int y_index = threadIdx.y + blockIdx.y * blockDim.y;

    int real_index = x_index* xlen + y_index;

    if (!(x_index < xlen && y_index < ylen)){
        return;
    }
    out[real_index] = a[real_index] *c;

}

int main(){
    int xlen = 3440;
    int ylen = 4443;

    int kernels_per_blockx = 16;
    int kernels_per_blocky = 16;

    int blocks_per_gridx = (xlen + kernels_per_blockx-1) / kernels_per_blockx;
    int blocks_per_gridy = (ylen + kernels_per_blocky-1) / kernels_per_blocky;

    int* a = new int[xlen * ylen];
    int* out = new int[xlen * ylen];
    int* a_device;
    int* out_device;
    int c = 2;

    int total_size = sizeof(int)*xlen*ylen;

    for (int row = 0; row<xlen; row++){
        for (int col = 0; col<ylen; col++){
            a[row*xlen + col] = row + col;
        }
    }

    cudaMalloc((void**)&a_device, total_size);
    cudaMalloc((void**)&out_device, total_size);


    cudaMemcpy(a_device, a, total_size, cudaMemcpyHostToDevice);

    dim3 blockSize(kernels_per_blockx, kernels_per_blocky);
    dim3 gridSize(blocks_per_gridx, blocks_per_gridy);

    mapKernel2D<<<gridSize, blockSize>>>(a_device, c, out_device, xlen, ylen);

    cudaMemcpy(out, out_device, total_size, cudaMemcpyDeviceToHost);

    for (int row = 0; row < xlen; row++) {
        // std::cout << "Processing row: " << row << std::endl;
        std::cout << "*" ;
        for (int col = 0; col < ylen; col++) {
            if (out[row * xlen + col] != a[row * xlen + col] * c) {
                std::cout << "row: " << row << ", col: " << col << ", a: " << a[row * xlen + col] << ", out: " << out[row * xlen + col] << std::endl;
            }
        }
    }
    std::cout<<std::endl;


   
    delete[] a;
    delete[] out;

    cudaFree(a_device);
    cudaFree(out_device);
}