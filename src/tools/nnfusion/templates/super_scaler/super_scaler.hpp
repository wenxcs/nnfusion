// Microsoft (c) 2019, NNFusion Team
#pragma once

#include <cstdio>
#include <thread>
// todo: unistd.h is only for linux
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define SS_MPICHECK(cmd)                                                                           \
    do                                                                                             \
    {                                                                                              \
        int e = cmd;                                                                               \
        if (e != MPI_SUCCESS)                                                                      \
        {                                                                                          \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define SS_CUDACHECK(cmd)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess)                                                                      \
        {                                                                                          \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define SS_NCCLCHECK(cmd)                                                                          \
    do                                                                                             \
    {                                                                                              \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess)                                                                      \
        {                                                                                          \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

static uint64_t super_scaler_getHostHash(const char* string)
{
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void super_scaler_getHostName(char* hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

__global__ static void gradientsAverage(float* gradients, int size, int super_scaler_nRanks)
{
    for (int i = 0; i < size; i++)
    {
        gradients[i] /= super_scaler_nRanks;
    }
}

int super_scaler_myRank = 0;
int super_scaler_nRanks = 0;
int super_scaler_localRank = 0;

void super_scaler_initialization()
{
    //initializing MPI
    SS_MPICHECK(MPI_Init(NULL, NULL));
    SS_MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &super_scaler_myRank));
    SS_MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &super_scaler_nRanks));

    //calculating super_scaler_localRank which is used in selecting a GPU
    uint64_t hostHashs[super_scaler_nRanks];
    char hostname[1024];
    super_scaler_getHostName(hostname, 1024);
    hostHashs[super_scaler_myRank] = super_scaler_getHostHash(hostname);
    SS_MPICHECK(MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    for (int p = 0; p < super_scaler_nRanks; p++)
    {
        if (p == super_scaler_myRank)
        {
            break;
        }
        if (hostHashs[p] == hostHashs[super_scaler_myRank])
        {
            super_scaler_localRank++;
        }
    }
}

void super_scaler_finalization()
{
    //finalizing MPI
    SS_MPICHECK(MPI_Finalize());
}

void super_scaler_all_reduce_device(float* gradients,
                                    float* out_gradients,
                                    int size,
                                    void (*callback)(cudaStream_t*, cudaEvent_t*),
                                    cudaStream_t* exe_s,
                                    cudaEvent_t* eve)
{
    //each process use 1 GPU
    int nDev = 1;

    //initializing GPU memery based on super_scaler_localRank
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i)
    {
        SS_CUDACHECK(cudaSetDevice(super_scaler_localRank * nDev + i));
        SS_CUDACHECK(cudaStreamCreate(s + i));
    }

    //generating NCCL unique ID at one process and broadcasting it to all
    ncclUniqueId id;
    if (super_scaler_myRank == 0)
    {
        ncclGetUniqueId(&id);
    }
    SS_MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    //initializing NCCL, group API is required around ncclCommInitRank as it is called across multiple GPUs in each thread/process
    ncclComm_t comms[nDev];
    SS_NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        SS_CUDACHECK(cudaSetDevice(super_scaler_localRank * nDev + i));
        SS_NCCLCHECK(ncclCommInitRank(
            comms + i, super_scaler_nRanks * nDev, id, super_scaler_myRank * nDev + i));
    }
    SS_NCCLCHECK(ncclGroupEnd());

    //calling NCCL communication API. Group API is required when using multiple devices per thread/process
    SS_NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
    {
        SS_NCCLCHECK(ncclAllReduce((const void*)gradients,
                                   (void*)out_gradients,
                                   size,
                                   ncclFloat,
                                   ncclSum,
                                   comms[i],
                                   s[i]));
    }
    SS_NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA stream to complete NCCL communication
    for (int i = 0; i < nDev; i++)
    {
        SS_CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // todo:This is buggy
    /*
    //get gradients after allreduce
    for (int i = 0; i < nDev; i++)
    {
        SS_CUDACHECK(cudaSetDevice(super_scaler_localRank * nDev + i));
        gradientsAverage<<<1, 1, 0>>>(gradients, size, super_scaler_nRanks);
    }
    */

    //finalizing NCCL
    for (int i = 0; i < nDev; i++)
    {
        ncclCommDestroy(comms[i]);
    }

    //call back
    (*callback)(exe_s, eve);
}

void super_scaler_all_reduce_device_async(float* gradients,
                                          float* out_gradients,
                                          int size,
                                          void (*callback)(cudaStream_t*, cudaEvent_t*),
                                          cudaStream_t* exe_s,
                                          cudaEvent_t* eve)
{
    std::thread ss_allreduce(
        super_scaler_all_reduce_device, gradients, out_gradients, size, callback, exe_s, eve);
    //Use Cuda Sync All Device to sync
    ss_allreduce.detach();
}
