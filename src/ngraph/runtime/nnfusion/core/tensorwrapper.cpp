// Microsoft (c) 2019, Wenxiang Hu
#include "tensorwrapper.hpp"
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
    return m_tensor->get_tensor_layout()->get_size();
}

size_t TensorWrapper::get_offset() const
{
    return m_tensor->get_pool_offset();
}

const Shape& TensorWrapper::get_shape() const
{
    return m_tensor->get_tensor_layout()->get_shape();
}

Strides TensorWrapper::get_strides() const
{
    return m_tensor->get_tensor_layout()->get_strides();
}

const element::Type& TensorWrapper::get_element_type() const
{
    return m_tensor->get_tensor_layout()->get_element_type();
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
