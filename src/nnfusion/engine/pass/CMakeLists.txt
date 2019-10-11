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
    elementwise_kernel_fusion.cpp
    # tensor related
    tensor/tensor_memory_layout.cpp
    tensor/host_tensor_allocation.cpp
    tensor/liveness_analysis.cpp
    # train related
    train/async_execution.cpp
    # fusion
    create_fusion_block.cpp
)
add_library(nnfusion_engine_pass STATIC ${SRC})
target_include_directories(nnfusion_engine_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass PRIVATE "-fPIC")
