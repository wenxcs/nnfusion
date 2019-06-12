# Microsoft (c) 2019, NNFusion Team

set(SRC
    op_registration.cpp
)

add_library(op_registration STATIC ${SRC})
target_include_directories(op_registration SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_link_libraries(op_registration PRIVATE
    ngraph codegen nnfusion_common nnfusion_util
)
target_compile_options(op_registration PRIVATE "-fPIC")

#-------kernel_test----------
# TODO: ..

