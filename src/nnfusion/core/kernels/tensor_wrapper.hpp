// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "common.hpp"

namespace nnfusion
{
    class TensorWrapper;
}

class nnfusion::TensorWrapper
{
public:
    TensorWrapper(const shared_ptr<ngraph::descriptor::Tensor>&, const string& alias = "");

    size_t get_size() const;
    size_t get_offset() const;
    const ngraph::Shape& get_shape() const;
    const ngraph::descriptor::Tensor& get_tensor() const;
    ngraph::Strides get_strides() const;
    const ngraph::element::Type& get_element_type() const;
    const string& get_name() const;
    const string& get_type() const;

private:
    shared_ptr<ngraph::descriptor::Tensor> m_tensor;
    string m_alias;
};