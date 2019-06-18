# Microsoft (c) 2019, NNFusion Team

set(SRC
    manager.cpp
)

add_library(nnfusion_graph_pass STATIC ${SRC})
target_include_directories(nnfusion_graph_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_graph_pass PRIVATE "-fPIC")