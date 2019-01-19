# Microsoft (c) 2019
# Wenxiang Hu

if (NGRAPH_INTERPRETER_ENABLE)
    add_library(nnfusion_backend SHARED
        nnfusion_backend.cpp
        nnfusion_functiontranslator.cpp
        nnfusion_tensorwrapper.cpp
        nnfusion_codegenerator.cpp
        nnfusion_op.cpp
        )
    set_target_properties(nnfusion_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(nnfusion_backend PUBLIC ngraph)
    set_target_properties(nnfusion_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS nnfusion_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
