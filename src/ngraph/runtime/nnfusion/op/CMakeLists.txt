# Microsoft (c) 2019, Wenxiang Hu

#Added Intermediate OPs for nnfusion
set(NNF_INTERMEDIATE
    result.cpp
    broadcast.cpp
    pad.cpp
    reshape.cpp
    util/gpu_util.cpp
)

add_library(nnfusion_op STATIC ${NNF_INTERMEDIATE})

target_include_directories(nnfusion_op SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_op PRIVATE "-fPIC")