# Microsoft (c) 2019, Wenxiang Hu

if (NGRAPH_GPU_ENABLE)
    find_package(CUDA 9 QUIET)
    if (CUDA_FOUND)
        set(CUDA9_FOUND TRUE)
        message(STATUS "Found CUDA 9")
    else()
        find_package(CUDA 8 REQUIRED)
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        --compiler-options -fPIC;
        -arch=sm_30;
        -gencode=arch=compute_35,code=sm_35;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_52,code=sm_52;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_61,code=sm_61;
        -gencode=arch=compute_61,code=compute_61)
    if (CUDA9_FOUND)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
            -gencode=arch=compute_62,code=sm_62;
            -gencode=arch=compute_70,code=sm_70;
            -gencode=arch=compute_70,code=compute_70)
    endif()

    find_library(CUDA_nvrtc_LIBRARY nvrtc /usr/local/cuda/lib64)
    find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
    find_library(CUDA_cudart_LIBRARY libcudart_static.a /usr/local/cuda/lib64)

    # Make pass and op object
    add_subdirectory(pass)
    add_subdirectory(op)

    # This is the source file shared by devices
    set(CORE
        core/codegenerator.cpp
        core/interpreter.cpp
        core/languageunit.cpp
        core/op.cpp 
        core/type_info.cpp
        core/tensorwrapper.cpp
    )

    # Add cuda backend
    if (CUDA_FOUND)
        add_subdirectory(cuda)
        set(CUDA_CODEGEN_SRC
            nnfusion_cudacodegen.cpp
            ${CORE}
        )

        add_library(cuda_codegen_backend SHARED
            ${CUDA_CODEGEN_SRC}
        )

        target_include_directories(cuda_codegen_backend SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
        set_target_properties(cuda_codegen_backend PROPERTIES VERSION ${NGRAPH_VERSION})
        target_link_libraries(cuda_codegen_backend PRIVATE
            nnfusion_cuda
            nnfusion_op
            nnfusion_pass
            ${CUDA_cuda_LIBRARY}
            ${CUDA_cudart_LIBRARY}
            ${CUDA_LIBRARIES}
        )
        set_target_properties(cuda_codegen_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

        install(TARGETS cuda_codegen_backend
            LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
            ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
        )
    endif()
endif()
