# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    IR/main.cpp
)
add_library(nnfusion_core_ir STATIC ${SRC})
target_include_directories(nnfusion_core_ir SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_core_ir PRIVATE "-fPIC")