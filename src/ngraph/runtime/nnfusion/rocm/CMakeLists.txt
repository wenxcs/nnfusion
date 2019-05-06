# Microsoft (c) 2019, Wenxiang Hu

#Added Codegen Pass for Cuda
set(SRC
# rocc file
    rocc_file/cublas_v2.h.file.cpp
    rocc_file/cudnn.h.file.cpp
    rocc_file/hipify-rocc.file.cpp
    rocc_file/rocc.file.cpp
# codegen file
    rocm_codegen.cpp
    rocm_langunit.cpp
# rocm pass
    pass/cuda_to_rocm_pass.cpp
)

add_library(nnfusion_rocm STATIC ${SRC})
target_include_directories(nnfusion_rocm SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)

target_compile_options(nnfusion_rocm PRIVATE "-fPIC")