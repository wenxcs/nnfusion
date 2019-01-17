#pragma once

#include <memory>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class TensorWrapper;
        }
    }
}

class ngraph::runtime::nnfusion::TensorWrapper
{
public:
    TensorWrapper(const std::shared_ptr<descriptor::Tensor>&, const std::string& alias = "");

    size_t get_size() const;
    const Shape& get_shape() const;
    Strides get_strides() const;
    const element::Type& get_element_type() const;
    const std::string& get_name() const;
    const std::string& get_type() const;

private:
    std::shared_ptr<descriptor::Tensor> m_tensor;
    std::string m_alias;
};