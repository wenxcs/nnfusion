// Microsoft (c) 2019, Wenxiang Hu
#include "tensorwrapper.h"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"

using namespace std;
using namespace ngraph;
using namespace nnfusion;

TensorWrapper::TensorWrapper(const shared_ptr<descriptor::Tensor>& tv, const string& alias)
    : m_tensor(tv)
    , m_alias(alias)
{
}

size_t TensorWrapper::get_size() const
{
    //\todo
    // need to variry the size;
    if (m_tensor->get_tensor_layout() != nullptr)
    {
        return m_tensor->get_tensor_layout()->get_size();
    }
    else
    {
        return shape_size(m_tensor->get_shape());
    }
}

size_t TensorWrapper::get_offset() const
{
    return m_tensor->get_pool_offset();
}

const Shape& TensorWrapper::get_shape() const
{
    //\todo
    // return m_tensor->get_tensor_layout()->get_shape();
    if (m_tensor->get_tensor_layout() != nullptr)
    {
        return m_tensor->get_tensor_layout()->get_shape();
    }
    else
    {
        return m_tensor->get_shape();
    }
}

const descriptor::Tensor& TensorWrapper::get_tensor() const
{
    return *m_tensor;
}

Strides TensorWrapper::get_strides() const
{
    return m_tensor->get_tensor_layout()->get_strides();
}

const element::Type& TensorWrapper::get_element_type() const
{
    // \todo was return m_tensor->get_tensor_layout()->get_element_type(),
    // What is the side effect.
    if (m_tensor->get_tensor_layout() != nullptr)
    {
        return m_tensor->get_tensor_layout()->get_element_type();
    }
    else
    {
        return m_tensor->get_element_type();
    }
}

const std::string& TensorWrapper::get_name() const
{
    if (m_alias.empty())
    {
        return m_tensor->get_name();
    }
    else
    {
        return m_alias;
    }
}

const std::string& TensorWrapper::get_type() const
{
    return get_element_type().c_type_string();
}
