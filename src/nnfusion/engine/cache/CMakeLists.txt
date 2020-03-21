# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    manager.cpp
)
add_library(nnfusion_cache_manager STATIC ${SRC})
target_include_directories(nnfusion_cache_manager SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_link_libraries(nnfusion_cache_manager
    nnfusion_graph sqlite3
)
set_target_properties(nnfusion_cache_manager PROPERTIES VERSION ${NGRAPH_VERSION})
target_compile_options(nnfusion_cache_manager PRIVATE "-fPIC")