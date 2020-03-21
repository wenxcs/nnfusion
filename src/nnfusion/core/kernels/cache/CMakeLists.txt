# Microsoft (c) 2019, NNFusion Team

set(SRC
    blockcuda.cpp
)

add_library(kernels_cache STATIC ${SRC})
target_include_directories(kernels_cache SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(kernels_cache PRIVATE "-fPIC")

target_link_libraries(kernels_cache PRIVATE
    kernels_registration gflags
)
