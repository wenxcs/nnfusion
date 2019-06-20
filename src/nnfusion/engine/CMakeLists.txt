# Microsoft (c) 2019, Wenxiang Hu

if (NNFUSION_ENABLE)

    # Make pass and op object
    add_subdirectory(pass)

    # This is the source file shared by devices
    set(ENGINE
        interpreter.cpp
        op.cpp 
        external/nnfusion_cudacodegen.cpp
    )

    add_library(nnfusion_backend SHARED
        ${ENGINE}
    )

    set_target_properties(nnfusion_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(nnfusion_backend PRIVATE
        nnfusion_engine_pass
        nnfusion_common
        nnfusion_util
        codegen
        # kernels
        kernels_registration
        -Wl,--whole-archive 
        kernels_cuda 
        kernels_cpu
        -Wl,--no-whole-archive
    )

    set_target_properties(nnfusion_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS nnfusion_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )

endif()