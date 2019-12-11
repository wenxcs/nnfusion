// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <memory>

#include "ngraph/descriptor/tensor.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Output
        {
        public:
            /// \param tensor The view of this tensor; where the value will be written
            Output(const std::shared_ptr<ngraph::descriptor::Tensor>& tensor)
                : m_tensor(tensor)
            {
            }

            ngraph::descriptor::Tensor& get_tensor() const { return *m_tensor; }
            std::shared_ptr<ngraph::descriptor::Tensor> get_tensor_ptr() const { return m_tensor; }
            void set_tensor_ptr(const std::shared_ptr<ngraph::descriptor::Tensor>& tensor)
            {
                m_tensor = tensor;
            }

            /// \return the element type of the output
            const ngraph::element::Type& get_element_type() const
            {
                return m_tensor->get_element_type();
            }
            /// \return the shape of the output
            const ngraph::Shape& get_shape() const { return m_tensor->get_shape(); }
            const ngraph::PartialShape& get_partial_shape() const
            {
                return m_tensor->get_partial_shape();
            }

            void set_type_and_shape(const ngraph::element::Type& element_type,
                                    const ngraph::PartialShape& pshape)
            {
                m_tensor->set_tensor_type(element_type, pshape);
            }

        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_tensor;

        private:
            Output(const Output&) = delete;
            Output(Output&&) = delete;
            Output& operator=(const Output&) = delete;
        };
    }
}
