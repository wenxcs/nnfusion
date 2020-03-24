// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief An abstract base class for tensor operations, such as Constant, Placeholder or Variable.
        ///
        class TensorOp : public Op
        {
        public:
            /// \brief Constructions a tensor view-typed op.
            ///
            /// \param node_type The node type of the tensor op.
            /// \param element_type The element type of the tensor.
            /// \param shape The shape of the tensor.
            TensorOp(const std::string& node_type,
                     const nnfusion::element::Type& element_type,
                     const nnfusion::Shape& shape);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            bool is_tensor_op() const override { return true; }
        protected:
            nnfusion::Shape m_shape{};
            nnfusion::element::Type m_element_type;
        };
    }
}