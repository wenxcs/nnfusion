# Microsoft (c) 2019, NNFusion Team

file(GLOB op_define op_define/*.cpp)

set(SRC
    op_registration.cpp
    ${op_define}
)

add_library(op_registration SHARED ${SRC})
target_include_directories(op_registration SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_link_libraries(op_registration PRIVATE
    ngraph nnfusion_common
)
target_compile_options(op_registration PRIVATE "-fPIC")


