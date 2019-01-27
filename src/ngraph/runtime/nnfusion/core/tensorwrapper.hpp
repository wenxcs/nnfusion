// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "common.hpp"

namespace nnfusion
{
    class TensorWrapper;
}

class nnfusion::TensorWrapper
{
public:
    TensorWrapper(const shared_ptr<descriptor::Tensor>&, const string& alias = "");

    size_t get_size() const;
    size_t get_offset() const;
    const Shape& get_shape() const;
    Strides get_strides() const;
    const element::Type& get_element_type() const;
    const string& get_name() const;
    const string& get_type() const;

private:
    shared_ptr<descriptor::Tensor> m_tensor;
    string m_alias;
};