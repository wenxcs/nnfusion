# Microsoft (c) 2019, Wenxiang Hu

add_subdirectory(op)

set(SRC
    extract_function_signature.cpp
    extract_graph_signature.cpp
    ngraph_function_pass.cpp
    cpu_codegenerator.cpp
    cuda_codegenerator.cpp
    rocm_codegenerator.cpp
    device_dispatcher.cpp
    kernel_selection.cpp
    create_fusion_block.cpp
)
add_library(nnfusion_engine_pass STATIC ${SRC})
target_include_directories(nnfusion_engine_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass PRIVATE "-fPIC")
