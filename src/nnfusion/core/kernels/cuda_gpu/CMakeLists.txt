# Microsoft (c) 2019, NNFusion Team

set(SRC
    cuda_emitter.cpp
    cuda_cudnn.cpp
    cuda_helper.cpp
    cuda_langunit.cpp
    cuda_testutil.cpp
)

set(UTIL 
    util/gpu_util.cpp)

set(KERNELS
    kernels/pad.cpp
    kernels/anyop.cpp
    kernels/batch_matmul.cpp
    kernels/dot.cpp
    kernels/elementwise.cpp
)

add_library(kernels_cuda STATIC ${SRC} ${UTIL} ${KERNELS})
target_include_directories(kernels_cuda SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(kernels_cuda PRIVATE "-fPIC")

target_link_libraries(kernels_cuda  PRIVATE
    kernels_registration
)