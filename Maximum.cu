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
    int random_range = 50;
    int threads_per_block = 1024;
} startup;

struct DataSet{
    float* values;
    int  size;
};

DataSet* createDeviceDataset(DataSet host){
    DataSet host_copy = {nullptr, host.size};
    gpuErrchk(cudaMalloc((void **)&host_copy.values,  host.size*sizeof(float));
    gpuErrchk(cudaMalloc((void **)&host_copy,  host.size*sizeof(float));

    DataSet* device_dataset = (DataSet*)malloc(sizeof(DataSet));
    gpuErrchk(cudaMalloc((void **)&host_copy,  sizeof(DataSet));
    gpuErrchk(cudaMalloc((void **)&device_dataset,  sizeof(DataSet));
    gpuErrchk(cudaMemcpy(device_d, input.values, sizeOfDataSet(input) , cudaMemcpyHostToDevice));
    return device_dataset;

}

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (float*)malloc(sizeof(float)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = (float)(rand()%startup.random_range);

    return data;
}

__global__ void MaxValue_1(DataSet data){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //input[idx] > input[idx+1]
}

float calculateMaxValue(DataSet data){
    float* device_data;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    int threads_needed = data.size;
    cudaEventRecord(start);
    MaxValue_1<<< threads_needed/ startup.threads_per_block + 1, startup.threads_per_block >>>(device_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Executed in %.6g\n", milliseconds);
    return milliseconds;
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.6g, ", data.values[i]);
    printf("\n");
}

 


int main(int argc, char** argv){
    DataSet random = generateRandomDataSet(10000);
    float max = calculateMaxValue(random);
    //printDataSet(random);
}