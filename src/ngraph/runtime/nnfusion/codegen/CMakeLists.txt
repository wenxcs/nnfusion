# Microsoft (c) 2019, Wenxiang Hu

#Added Codegen Pass for Cuda
set(NNF_CUDA_CODEGEN_SRC
    cuda/cuda_codegen.cpp 
    cuda/cuda_codegenop.cpp 
    cuda/cuda_helper.cpp 
    cuda/cuda_testutil.cpp 
    cuda/cuda_langunit.cpp
    cuda/Result.cpp
)

add_library(nnfusion_cuda_codegen STATIC ${NNF_CUDA_CODEGEN_SRC})
target_include_directories(nnfusion_cuda_codegen SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

target_compile_options(nnfusion_cuda_codegen PRIVATE "-fPIC")

#Added Codegen Pass for ROCm 
#set(NNF_ROCM_CODEGEN_SRC
#)
# add_library(cuda_codegen SHARED $(CUDA_CODEGEN_SRC))
# target_include_directories(cuda_codegen SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})