#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define MAX_CUDA_THREADS_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Startup{
    int random_range = INT_MAX;
    int threads_per_block = MAX_CUDA_THREADS_PER_BLOCK;
} startup;

struct DataSet{
    float* values;
    int  size;
};

struct Result{
    float MaxValue;
    float KernelExecutionTime;
};

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (float*)malloc(sizeof(float)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = (float)(rand()%startup.random_range);

    return data;
}

__global__ void Max_Interleaved_Addressing_Global(float* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size){
        for(int stride=1; stride < data_size; stride *= 2) {
            if (idx % (2*stride) == 0) {
                float lhs = data[idx];
                float rhs = data[idx + stride];
                data[idx] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
}

__global__ void Max_Interleaved_Addressing_Shared(float* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=1; stride < blockDim.x; stride *= 2) {
            if (threadIdx.x % (2*stride) == 0) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}


__global__ void Max_Sequential_Addressing_Shared(float* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[MAX_CUDA_THREADS_PER_BLOCK];
    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}

const int Algorithm_Count = 3;
typedef void (*Kernel)(float *, int);
const char* Algorithm_Name[Algorithm_Count]= {"Max_Interleaved_Addressing_Global", "Max_Interleaved_Addressing_Shared", "Max_Sequential_Addressing_Shared"};
const Kernel Algorithm[Algorithm_Count]    = { Max_Interleaved_Addressing_Global,   Max_Interleaved_Addressing_Shared,   Max_Sequential_Addressing_Shared};

Result calculateMaxValue(DataSet data, Kernel algorithm){
    float* device_data;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    gpuErrchk(cudaMalloc((void **)&device_data,  sizeof(float)*data.size));
    gpuErrchk(cudaMemcpy(device_data, data.values, sizeof(float)*data.size, cudaMemcpyHostToDevice));


    int threads_needed = data.size;
    cudaEventRecord(start);
    algorithm<<< threads_needed/ startup.threads_per_block + 1, startup.threads_per_block>>>(device_data, data.size);
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float max_value;
    gpuErrchk(cudaMemcpy(&max_value, device_data, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_data));

    Result r = {max_value, milliseconds};
    return r;
}

Result calculateMaxValue(DataSet data){
    return calculateMaxValue(data, Algorithm[Algorithm_Count - 1]);
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.6g, ", data.values[i]);
    printf("\n");
}

void benchmarkCSV(){
    /*Print Headers*/
    printf("Elements, ");
    for (int algoID = 0; algoID < Algorithm_Count; algoID++)
        printf("%s, ", Algorithm_Name[algoID]);
    printf("\n");
    /*Benchmark*/
    for (int dataSize = 1024; dataSize < INT_MAX; dataSize*=2){
        DataSet random = generateRandomDataSet(dataSize);
        printf("%d, ", dataSize);
        for (int algoID = 0; algoID < Algorithm_Count; algoID++) {
            Result r = calculateMaxValue(random, Algorithm[algoID]);
            printf("%g, ", r.KernelExecutionTime);
        }
        printf("\n");
        free(random.values);
    }
}

int main(int argc, char** argv){
    srand(time(nullptr));
    benchmarkCSV();
}