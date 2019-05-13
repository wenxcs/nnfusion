# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    ir.cpp
)
add_library(nnfusion_core STATIC ${SRC})
target_include_directories(nnfusion_core SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_core PRIVATE "-fPIC")