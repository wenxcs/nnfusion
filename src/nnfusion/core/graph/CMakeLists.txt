# Microsoft (c) 2019, NNFusion Team

add_subdirectory(pass)

set(SRC
    graph.cpp
    gnode.cpp
    edge.cpp
    graph_util.cpp
)

add_library(nnfusion_graph STATIC ${SRC})
target_include_directories(nnfusion_graph SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_graph PRIVATE "-fPIC")