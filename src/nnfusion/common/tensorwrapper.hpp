// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "common.hpp"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
namespace nnfusion
{
    class TensorWrapper;
}

// descriptor::Tensor is just descriptor.
class nnfusion::TensorWrapper
{
public:
    using Pointer = shared_ptr<nnfusion::TensorWrapper>;
    TensorWrapper(const shared_ptr<nnfusion::descriptor::Tensor>&, const string& alias = "");

    bool is_host() const;
    bool is_persistent() const;
    void set_host_tensor(bool value = true);
    size_t get_size() const;
    size_t get_offset() const;
    const Shape& get_shape() const;
    const nnfusion::descriptor::Tensor& get_tensor() const;
    Strides get_strides() const;
    const element::Type& get_element_type() const;
    const string& get_name() const;
    const string& get_type() const;

private:
    shared_ptr<nnfusion::descriptor::Tensor> m_tensor;
    string m_alias;
    bool m_ishost;
};