# Microsoft (c) 2019, Wenxiang Hu
# This will moved to src/nnfusion

set(SRC
    #Log unit
    log.cpp
)
add_library(nnfusion_util_ STATIC ${SRC})
target_include_directories(nnfusion_util_ SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_util_ PRIVATE "-fPIC")