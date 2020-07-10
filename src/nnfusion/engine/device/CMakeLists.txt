# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    cuda.cpp
    graphcore.cpp
    hlsl.cpp
    rocm.cpp
    reversed_dfs_visitor.cpp
)
add_library(nnfusion_engine_device STATIC ${SRC})
target_include_directories(nnfusion_engine_device SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_link_libraries(nnfusion_engine_device PRIVATE
    nnfusion_graph ${CURL_LIBRARIES}
)
set_target_properties(nnfusion_engine_device PROPERTIES VERSION ${NGRAPH_VERSION})
target_compile_options(nnfusion_engine_device PRIVATE "-fPIC")