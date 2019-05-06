// Microsoft (c) 2019, Wenxiang
#include "../rocm_langunit.hpp"

using namespace nnfusion::rocm;

LU_DEFINE(file::cublas_v2_h,
          "rocm::file::nnfusion_hip",
          R"(#ifndef __HIPIFY_H__
#define __HIPIFY_H__

#include <assert.h>
#include <hipblas.h>
#include <miopen/miopen.h>

#define cudnnDataType_t miopenDataType_t
#define cudnnHandle_t miopenHandle_t
#define cublasHandle_t hipblasHandle_t
#define cudnnStatus_t miopenStatus_t

#define CUDNN_BATCHNORM_SPATIAL miopenBNSpatial
#define cudnnDeriveBNTensorDescriptor miopenDeriveBNTensorDescriptor
#define cudnnBatchNormalizationForwardInference(a,b,c,d,e,f,g,h,i,j,k,l,m,n) \
        miopenBatchNormalizationForwardInference(a,b,(void*)(c),(void*)(d),e,f,g,h,i,(void*)(j),(void*)(k),(void*)(l),(void*)(m),n)

#define CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING miopenPoolingAverage

#define CUDNN_STATUS_SUCCESS miopenStatusSuccess
#define cudnnGetErrorString miopenGetErrorString
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#define cudnnTensorFormat_t int
#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_TENSOR_NCHW 0
#define cudnnConvolutionMode_t miopenConvolutionMode_t
#define CUDNN_CROSS_CORRELATION miopenConvolution
#define cudnnConvolutionFwdAlgo_t miopenConvFwdAlgorithm_t
#define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM miopenConvolutionFwdAlgoGEMM
#define cudnnPoolingMode_t miopenPoolingMode_t
#define CUDNN_POOLING_MAX miopenPoolingMax

#define cudnnTensorDescriptor_t miopenTensorDescriptor_t
#define cudnnFilterDescriptor_t miopenTensorDescriptor_t
#define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
#define cudnnPoolingDescriptor_t miopenPoolingDescriptor_t

#define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
#define cudnnCreateFilterDescriptor miopenCreateTensorDescriptor
#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define cudnnCreatePoolingDescriptor miopenCreatePoolingDescriptor
#define cudnnDestroyTensorDescriptor miopenDestroyTensorDescriptor
#define cudnnDestroyFilterDescriptor miopenDestroyTensorDescriptor
#define cudnnDestroyConvolutionDescriptor miopenDestroyConvolutionDescriptor
#define cudnnDestroyPoolingDescriptor miopenDestroyPoolingDescriptor

#define cudnnGetConvolutionForwardWorkspaceSize(h, x, w, c, y, algo, size) \
	(*(size) = (1LU << 30), miopenStatusSuccess) // miopenConvolutionForwardGetWorkSpaceSize(h, w, x, c, y, size)
#define cudnnConvolutionForward(h, a, x, d_x, w, d_w, c, algo, ws, size, b, y, d_y) \
	miopenConvolutionForward(h, a, x, d_x, w, d_w, c, algo, b, y, d_y, ws, size)
#define cudnnSetPooling2dDescriptor(desc, mode, nanOpt, k0, k1, p0, p1, s0, s1) \
	miopenSet2dPoolingDescriptor(desc, mode, k0, k1, p0, p1, s0, s1)
#define cudnnPoolingForward(h, p, a, x, d_x, b, y, d_y) \
	miopenPoolingForward(h, p, a, x, d_x, b, y, d_y, false, NULL, 0LU)
#define cudnnSetTensor4dDescriptor(desc, fmt, type, n, c, h, w) \
	miopenSet4dTensorDescriptor(desc, type, n, c, h, w)
	// (assert((fmt) == CUDNN_TENSOR_NCHW), miopenSet4dTensorDescriptor(desc, type, n, c, h, w))
#define cudnnSetFilter4dDescriptor(desc, type, fmt, k, c, h, w) \
       miopenSet4dTensorDescriptor(desc, type, k, c, h, w)
       // (assert((fmt) == CUDNN_TENSOR_NCHW), miopenSet4dTensorDescriptor(desc, type, k, c, h, w))	
#define cudnnSetConvolution2dDescriptor(d, p0, p1, s0, s1, d0, d1, mode, type) \
	miopenInitConvolutionDescriptor(d, mode, p0, p1, s0, s1, d0, d1)

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cublasSgemm hipblasSgemm
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cudnnCreate miopenCreate
#define cudnnDestroy miopenDestroy
#define cublasSetStream hipblasSetStream
#define cudnnSetStream miopenSetStream

/////////////////////////////////////////////////////////////////

#include <unordered_map>
#include <vector>

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#define LOGGING(...)    fprintf(__VA_ARGS__)
#else
#define LOGGING(...)
#endif

namespace {
  using std::unordered_map;
  using std::vector;

  unordered_map<void*, size_t> ptr2len;
  unordered_map<size_t, vector<void*>> len2ptrs;

  inline void metric_spot(int line) {
    if (!ENABLE_DEBUG)
      return;
    hipStreamSynchronize(0);
    static bool init = true;
    static std::chrono::time_point<std::chrono::high_resolution_clock> past;
    std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();
    double diff = init ? 0 : (now - past).count() * 1e-9;
    printf("[DBG] Line-%04d: %.4lfs\n", line, diff);
    past = std::move(now);
    init = false;
  }

  inline hipError_t __managed_malloc(void **dptr, size_t bytes) {
    LOGGING(stderr, "[DBG] Managed Alloc: %zd\n", bytes);
    auto &it = len2ptrs[bytes];
    if (it.size()) {
      // LOGGING(stderr, "[DBG] Reusing existing buffer.\n");
      *dptr = it.back();
      it.pop_back();
      return hipSuccess;
    }
    assert(hipSuccess == hipMalloc(dptr, bytes));
    ptr2len[*dptr] = bytes;
    return hipSuccess;
  }

  inline hipError_t __managed_free(void *dptr) {
    LOGGING(stderr, "[DBG] Managed Free: %p\n", dptr);
    size_t bytes = ptr2len[dptr];
    assert(bytes > 0);
    len2ptrs[bytes].push_back(dptr);
    return hipSuccess;

    assert(hipSuccess == hipFree(dptr));
    return hipSuccess;
  }

} // namespace

#define hipMalloc __managed_malloc
#define hipFree __managed_free

#endif
)")