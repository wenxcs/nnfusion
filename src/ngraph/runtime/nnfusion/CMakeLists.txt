# Microsoft (c) 2019, Wenxiang Hu

if (NGRAPH_INTERPRETER_ENABLE)
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

    add_subdirectory(codegen)
    add_subdirectory(intermediate)
    add_subdirectory(pass)

    set(COMMON_SRC
        nnfusion_functiontranslator.cpp
        nnfusion_tensorwrapper.cpp
        nnfusion_codegenerator.cpp
        nnfusion_languageunit.cpp
        nnfusion_op.cpp
    )

    set(CUDA_CODEGEN_SRC
        nnfusion_cudacodegen.cpp
        ${COMMON_SRC}
    )

    add_library(cuda_codegen_backend SHARED
        ${CUDA_CODEGEN_SRC}
    )

    target_include_directories(cuda_codegen_backend SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    set_target_properties(cuda_codegen_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(cuda_codegen_backend PUBLIC
        nnfusion_cuda_codegen
        nnfusion_intermediate
        nnfusion_trans_pass
        nnfusion_codegen_pass
        ngraph
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
