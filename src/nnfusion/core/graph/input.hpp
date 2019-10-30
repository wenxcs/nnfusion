// Microsoft (c) 2019, NNFusion Team

#pragma once
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Input
        {
        public:
            Input(const ngraph::element::Type& element_type, const ngraph::Shape& shape)
                : m_element_type(element_type)
                , m_shape(shape)
            {
            }

            void set_input_type(const ngraph::element::Type& element_type)
            {
                m_element_type = element_type;
            }

            void set_input_shape(const ngraph::Shape& shape) { m_shape = shape; }
            const ngraph::element::Type& get_element_type() const { return m_element_type; }
            const ngraph::Shape& get_shape() const { return m_shape; };
        private:
            ngraph::element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            ngraph::Shape m_shape;
        };
    }
}