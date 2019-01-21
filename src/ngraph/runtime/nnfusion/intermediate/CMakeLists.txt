# Microsoft (c) 2019, Wenxiang Hu

#Added Intermediate OPs for nnfusion
set(NNF_INTERMEDIATE
    Result.cpp
)

add_library(nnfusion_intermediate STATIC ${NNF_INTERMEDIATE})

target_include_directories(nnfusion_intermediate SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_intermediate PRIVATE "-fPIC")