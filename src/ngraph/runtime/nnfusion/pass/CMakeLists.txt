# Microsoft (c) 2019, Wenxiang Hu

#Added Codegen Pass for Cuda
set(NNF_TRANS_PASS
    translator/extract_function_signature.cpp
    translator/ngraph_function_pass.cpp
)

add_library(nnfusion_trans_pass STATIC ${NNF_TRANS_PASS})
target_include_directories(nnfusion_trans_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_trans_pass PRIVATE "-fPIC")