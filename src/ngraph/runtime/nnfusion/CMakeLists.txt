# Microsoft (c) 2019, Wenxiang Hu

if (NNFUSION_ENABLE)

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
    add_subdirectory(cuda)
    set(CUDA_CODEGEN_SRC
        nnfusion_cudacodegen.cpp
        ${CORE}
    )

    add_library(cuda_backend SHARED
        ${CUDA_CODEGEN_SRC}
    )

    set_target_properties(cuda_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(cuda_backend PRIVATE
        nnfusion_cuda
        nnfusion_op
        nnfusion_pass
        codegen
    )
    set_target_properties(cuda_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS cuda_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )

    # add rocm backend
    add_subdirectory(rocm)
    set(ROCM_CODEGEN_SRC
        nnfusion_rocmcodegen.cpp
        ${CORE}
    )

    add_library(rocm_backend SHARED
        ${ROCM_CODEGEN_SRC}
    )

    set_target_properties(rocm_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(rocm_backend PRIVATE
        nnfusion_cuda
        nnfusion_op
        nnfusion_pass
        nnfusion_rocm
        codegen
    )
    set_target_properties(rocm_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS rocm_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )

endif()