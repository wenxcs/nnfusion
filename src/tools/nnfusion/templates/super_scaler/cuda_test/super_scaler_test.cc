#include "super_scaler.h"

#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void display_device_content(float* device, float* host, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        host[i] = 0;
    }
    cudaMemcpy(host, device, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        printf("%lf ", host[i]);
    }
    printf("\n");
}

int main()
{
    int lrank;
    int grank;
    cudaStream_t* comm_stream;

    sc_init();
    sc_get_global_rank(&grank);
    sc_get_local_rank(&lrank);
    std::string planPath = "./plan/execution_plan/" + std::to_string(grank) + ".cfg";
    sc_load_plan(planPath.c_str());
    sc_get_comm_stream(&comm_stream);
    printf("[Grank: %d Lrank: %d] is running with stream %p\n", grank, lrank, comm_stream);

    const char* tensorname = "AllReduce_1";
    //size_t size = 16 * 1024 * 1024;
    size_t size = 5;
    float* gradients = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++)
    {
        gradients[i] = (lrank + 1) * 2 * i;
    }

    float* sendbuff = NULL;
    cudaSetDevice(grank);
    cudaMalloc((void**)(&sendbuff), size * sizeof(float));
    cudaMemset(sendbuff, 0, size * sizeof(float));
    cudaMemcpy(sendbuff, gradients, size * sizeof(float), cudaMemcpyHostToDevice);
    printf("Before allReduce: ");
    display_device_content(sendbuff, gradients, size);
    sc_allreduce(tensorname, sendbuff, size);
    printf("After allReduce: ");
    display_device_content(sendbuff, gradients, size);

    cudaFree(sendbuff);
    free(gradients);
    sc_finalize();
    printf("Done!\n");
    return 0;
}
