/* This program will performe a reduce vectorA (size N)
* with the + operation.
+---------+ 
|111111111| 
+---------+
     |
     N

vectorA   = all Ones
N = Sum of vectorA
*/
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


// CUDA macro wrapper for checking errors
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


// CPU reduce
void reduce(int* vectorA, int* sum, int size)
{
    sum[0] = 0;
    for (int i = 0; i < size; i++)
        sum[0] += vectorA[i];
}


// EXERCISE
// Read: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
// Implement the reduce kernel based on the information
// of the Nvidia blog post.
// Implement both options, using no shared mem at all but global atomics
// and using shared mem for the seconds recution phase.
__global__ void cudaEvenFasterReduceAddition() {
    //ToDo
}


// Already optimized reduce kernel using shared memory.
__global__ void cudaReduceAddition(int* vectorA, int* sum)
{
    int globalIdx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int shmArray[];

    shmArray[threadIdx.x] = vectorA[globalIdx];
    shmArray[threadIdx.x + blockDim.x] = vectorA[globalIdx + blockDim.x];

    for (int stride = blockDim.x; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmArray[threadIdx.x] += shmArray[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[blockIdx.x] = shmArray[0];
    }
}


// Compare result vectors
int compareResultVec(int* vectorCPU, int* vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "No errors. All good!" << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


int main(void)
{
    // Define the size of the vector: 1048576 elements
    const int N = 1 << 20;
    const int NBR_BLOCK = 512;

    // Allocate and prepare input
    int* hostVectorA = new int[N];
    int hostSumCPU[1];
    int hostSumGPU[1];
    for (int i = 0; i < N; i++) {
        hostVectorA[i] = 1;
    }

    // Alloc N times size of int at address of deviceVector[A-C]
    int* deviceVectorA;
    int* deviceSum;
    gpuErrCheck(cudaMalloc(&deviceVectorA, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceSum, NBR_BLOCK* sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVectorA, hostVectorA, N * sizeof(int), cudaMemcpyHostToDevice));

    // Run the vector kernel on the CPU
    reduce(hostVectorA, hostSumCPU, N);

    // Run kernel on all elements on the GPU
    cudaReduceAddition <<<NBR_BLOCK, 1024, 2 * 1024 * sizeof(int)>>> (deviceVectorA, deviceSum);
    gpuErrCheck(cudaPeekAtLastError());
    cudaReduceAddition <<<1, NBR_BLOCK / 2, NBR_BLOCK * sizeof(int) >> > (deviceSum, deviceSum);
    gpuErrCheck(cudaPeekAtLastError());

    // Copy the result stored in device_y back to host_y
    gpuErrCheck(cudaMemcpy(hostSumGPU, deviceSum, sizeof(int), cudaMemcpyDeviceToHost));

    // Check for errors
    auto isValid = compareResultVec(hostSumCPU, hostSumGPU, 1);

    // Free memory on device
    gpuErrCheck(cudaFree(deviceVectorA));
    gpuErrCheck(cudaFree(deviceSum));

    // Free memory on host
    delete[] hostVectorA;

    return isValid;
}
