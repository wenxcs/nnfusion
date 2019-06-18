# Microsoft (c) 2019, NNFusion Team

set(SRC
    graph.cpp
    node.cpp
    edge.cpp
)

add_library(nnfusion_graph STATIC ${SRC})
target_include_directories(nnfusion_graph SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_graph PRIVATE "-fPIC")