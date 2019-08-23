# Microsoft (c) 2019, Wenxiang Hu
set(NNFUSION_TESTS
    #op
    nnfusion/inventory/pad.cpp
    #kernels
    nnfusion/kernels/sample.cpp
    nnfusion/core/tagable.cpp
    PARENT_SCOPE)
