//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

            std::shared_ptr<ngraph::descriptor::Tensor> get_tensor_ptr() const { return m_tensor; }
            void set_tensor_ptr(const std::shared_ptr<ngraph::descriptor::Tensor>& tensor)
            {
                m_tensor = tensor;
            }
            ngraph::descriptor::Tensor& get_tensor() const { return *m_tensor; }
            /// \return the shape of the output
            const ngraph::Shape& get_shape() const { return m_tensor->get_shape(); }
            /// \return the element type of the output
            const ngraph::element::Type& get_element_type() const
            {
                return m_tensor->get_element_type();
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
