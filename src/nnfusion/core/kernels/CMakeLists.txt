# Microsoft (c) 2019, NNFusion Team

# #-------lib_kernels_common----------
# set(SRC
#     kernel_emitter.cpp
# )
# add_library(kernels_common STATIC ${SRC})
# target_include_directories(kernels_common SYSTEM PUBLIC
#     ${NGRAPH_INCLUDE_PATH}
# )
# target_compile_options(kernels_common PRIVATE "-fPIC")
# target_link_libraries(kernels_common ngraph codegen nnfusion_common nnfusion_util )


#-------lib_kernels_registration----------
set(SRC
    kernel_emitter.cpp
    kernel_registration.cpp
    common_langunit.cpp
)
add_library(kernels_registration STATIC ${SRC})
target_include_directories(kernels_registration SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_link_libraries(kernels_registration PRIVATE
    ngraph codegen nnfusion_common nnfusion_util
)
target_compile_options(kernels_registration PRIVATE "-fPIC")

add_subdirectory(cuda_gpu)
add_subdirectory(cpu)

#-------kernel_test----------
set(TEST
    main.cpp
)
add_executable(kernel_test ${TEST})

target_link_libraries(kernel_test 
    kernels_registration
    -Wl,--whole-archive 
    kernels_cuda 
    -Wl,--no-whole-archive
)
