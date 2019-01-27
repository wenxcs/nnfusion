# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    interpreter/extract_function_signature.cpp
    interpreter/ngraph_function_pass.cpp
    codegen/naive_unit_test_dump.cpp
)
add_library(nnfusion_pass STATIC ${SRC})
target_include_directories(nnfusion_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_pass PRIVATE "-fPIC")