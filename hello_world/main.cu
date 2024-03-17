#include <stdio.h>

// This kernel runs on the GPU and prints the thread's identifiers
__global__ void kernel() {
  printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
  // Launch the kernel on the GPU with four blocks of six threads each
  kernel<<<4,6>>>();

  // Check for CUDA errors
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
  return 0;
}