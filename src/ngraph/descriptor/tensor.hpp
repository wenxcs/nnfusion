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
#include <string>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        namespace layout
        {
            class TensorLayout;
        }       

        /// \brief Compile-time descriptor of a first-class value that is a view of a tensor.
        class Tensor
        {
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

        public:
            enum DeviceType
            {
                CUDA_GPU,
                ROCM_GPU,
                GENERIC_CPU
            };

            Tensor(const element::Type& element_type,
                   const PartialShape& pshape,
                   const std::string& name,
                   bool is_persistant = false,
                   bool is_constant = false,
                   bool is_RDMA_tensor = false,
                   size_t group_id = -1,
                   DeviceType device_type = CUDA_GPU,
                   size_t device_id = 0);

            const std::string& get_name() const { return m_name; }
            void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);

            const element::Type& get_element_type() const { return m_element_type; }
            const Shape& get_shape() const;
            const PartialShape& get_partial_shape() const { return m_partial_shape; }
            const std::shared_ptr<layout::TensorLayout>& get_tensor_layout() const
            {
                return m_tensor_layout;
            }

            void set_tensor_layout(const std::shared_ptr<layout::TensorLayout>& tensor_layout);

            void set_pool_offset(size_t);
            size_t get_pool_offset() const;
            size_t size() const;
            // Persistant tensors exist in all iterations, and do not reuse any memory space.
            bool is_persistent() const { return m_persistent; }
            // Constant tensors contain immutable data.
            bool is_constant() const { return m_constant; }
            bool is_RDMA_tensor() const { return m_RDMA; }
            void set_persistent() { m_persistent = true; }
            void set_constant() { m_constant = true; }
            void set_RDMA() { m_RDMA = true; }
            //The default group_id is -1, which means the tensor does not belong to any specific group.
            void set_group_id(size_t group_id) { m_group_id = group_id; }
            size_t get_group_id() const { return m_group_id; }
            void set_device_type(DeviceType device_type) { m_device_type = device_type; }
            DeviceType get_device_type() const { return m_device_type; }
            void set_device_id(size_t device_id) { m_device_id = device_id; }
            size_t get_device_id() const { return m_device_id; }
        protected:
            element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            Shape m_shape;
            PartialShape m_partial_shape;

            std::string m_name;
            std::shared_ptr<layout::TensorLayout> m_tensor_layout;
            size_t m_pool_offset{0};
            bool m_persistent;
            bool m_constant;
            bool m_RDMA;
            size_t m_group_id;
            DeviceType m_device_type;
            size_t m_device_id;
        };

        std::ostream& operator<<(std::ostream&, const ngraph::descriptor::Tensor&);
    }
}
