# Microsoft (c) 2019, Wenxiang Hu
set(NNFUSION_TESTS
    #op
    nnfusion/inventory/pad.cpp
    #kernels
    nnfusion/kernels/sample.cpp
    nnfusion/kernels/batch_test.cpp
    nnfusion/core/tagable.cpp
    nnfusion/engine/profiler.cpp
    PARENT_SCOPE)
