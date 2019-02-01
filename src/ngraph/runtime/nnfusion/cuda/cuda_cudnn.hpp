// Microsoft (c) 2019, Wenxiang
#pragma once

#include "cuda_helper.hpp"

namespace nnfusion
{
    namespace cuda
    {
        std::vector<int> compute_strides(const std::vector<int>& shape);
        std::string get_cudnn_datatype(std::string dtype);
        LanguageUnit_p cudnn_tensor_descriptor_from_shape(const ngraph::Shape& shape, string desc);
    }
}