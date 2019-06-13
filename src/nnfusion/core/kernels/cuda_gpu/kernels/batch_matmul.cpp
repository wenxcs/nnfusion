// Microsoft (c) 2019, NNFusion Team
#pragma once

// This is the 2rd generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style:
//
// 3 files to change:
//   [a] ./new_kernel_0.cpp
//   [b] ./new_kernel_0.hpp
//   [c] ../../../ops/op_registration.cpp

#include "batch_matmul.hpp"
