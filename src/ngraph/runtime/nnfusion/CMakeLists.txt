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

    add_library(cuda_codegen_backend SHARED
        ${CUDA_CODEGEN_SRC}
    )

    set_target_properties(cuda_codegen_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(cuda_codegen_backend PRIVATE
        nnfusion_cuda
        nnfusion_op
        nnfusion_pass
    )
    set_target_properties(cuda_codegen_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS cuda_codegen_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
