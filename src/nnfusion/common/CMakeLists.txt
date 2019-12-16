# Microsoft (c) 2019, MNFusion Team

set(SRC
    languageunit.cpp
    type_info.cpp
    descriptor/layout/dense_tensor_layout.cpp
    descriptor/layout/tensor_layout.cpp
    descriptor/tensor.cpp
    code_writer.cpp
)

add_library(nnfusion_common STATIC ${SRC})
target_include_directories(nnfusion_common SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_common PRIVATE "-fPIC")
target_link_libraries(nnfusion_common PRIVATE ngraph nnfusion_util gflags)
