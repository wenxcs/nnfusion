#ifndef SUPERSCALER_H_
#define SUPERSCALER_H_

#include<stdio.h>
#include<cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum SC_STATUS{
    SC_STATUS_SUCCESS,
    SC_STATUS_FAILED,
    SC_STATUS_ERROR
};

#define SCCHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != SC_STATUS_SUCCESS)                        \
        {                                            \
            printf("Failed: SC error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

int sc_init();
int sc_get_world_size(int*);
int sc_get_global_rank(int*);
int sc_get_local_rank(int*);
int sc_get_comm_stream(cudaStream_t** stream);
int sc_load_plan(const char* plan_path);
int sc_finalize();


//in-place allreduce
int sc_allreduce(const char* tensor_name, float* ioput, size_t size);
int sc_send(const char* tensor_name, unsigned char* input, size_t size);
int sc_recv(const char* tensor_name, unsigned char** output, size_t* size);


#ifdef __cplusplus
}
#endif


#endif
