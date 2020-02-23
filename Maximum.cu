#include <stdio.h>
#include <cuda.h>
#include <time.h>

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
    int random_range = 100;
    int threads_per_block = 1024;
} startup;

struct DataSet{
    float* values;
    int  size;
};

/*
DataSet* createDeviceDataset(DataSet host){
    DataSet* device_dataset = (DataSet*)malloc(sizeof(DataSet));
    gpuErrchk(cudaMalloc((void **)&host_copy,  sizeof(DataSet)));
    gpuErrchk(cudaMalloc((void **)&device_dataset,  sizeof(DataSet)));
    gpuErrchk(cudaMemcpy(device_d, input.values, sizeOfDataSet(input) , cudaMemcpyHostToDevice));
    return device_dataset;

}*/

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
    extern __shared__ float sdata[];
    /*copy to shared memory*/
    sdata[threadIdx.x] = data[idx];
    __syncthreads();

    if (idx < data_size){
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

float calculateMaxValue(DataSet data){
    float* device_data;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    gpuErrchk(cudaMalloc((void **)&device_data,  sizeof(float)*data.size));
    gpuErrchk(cudaMemcpy(device_data, data.values, sizeof(float)*data.size, cudaMemcpyHostToDevice));


    int threads_needed = data.size;
    cudaEventRecord(start);
    Max_Interleaved_Addressing_Shared<<< threads_needed/ startup.threads_per_block + 1, startup.threads_per_block, sizeof(float)*startup.threads_per_block>>>(device_data, data.size);
    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Executed in %.6g\n", milliseconds);

    float max_value;
    gpuErrchk(cudaMemcpy(&max_value, device_data, sizeof(float), cudaMemcpyDeviceToHost));
    return max_value;
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.6g, ", data.values[i]);
    printf("\n");
}

 


int main(int argc, char** argv){
    srand(time(nullptr));
    DataSet random = generateRandomDataSet(10);
    printDataSet(random);
    float max = calculateMaxValue(random);
    printf("The maximum value is: %g", max);
}