# Microsoft (c) 2019, Wenxiang Hu

#Added Codegen Pass for Cuda
set(SRC
    cuda_codegen.cpp 
    cuda_function.cpp 
    cuda_helper.cpp 
    cuda_testutil.cpp 
    cuda_langunit.cpp
    cuda_cudnn.cpp
)

set(OP
    op/constant_naive.cpp
    op/convolution.cpp
    op/result.cpp
    op/anyop.cpp
    op/dot.cpp
    op/pad.cpp
    op/reshape.cpp
    op/broadcast.cpp
    op/max_pool.cpp
    op/avg_pool.cpp
    op/batchnorm.cpp
    op/concat.cpp
    op/slice.cpp
)

add_library(nnfusion_cuda STATIC ${SRC} ${OP})
target_include_directories(nnfusion_cuda SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_cuda PRIVATE "-fPIC")

#Added Codegen Pass for ROCm 
#set(NNF_ROCM_CODEGEN_SRC
#)
# add_library(cuda_codegen SHARED $(CUDA_CODEGEN_SRC))
# target_include_directories(cuda_codegen SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})