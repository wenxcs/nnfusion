# Microsoft (c) 2019, Wenxiang Hu

#Added Intermediate OPs for nnfusion
set(NNF_INTERMEDIATE
    pad.cpp
    external/gpu_util.cpp
)

add_library(nnfusion_engine_pass_op STATIC ${NNF_INTERMEDIATE})

target_include_directories(nnfusion_engine_pass_op SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_engine_pass_op PRIVATE "-fPIC")