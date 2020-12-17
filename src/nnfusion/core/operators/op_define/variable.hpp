// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../util/tensor_op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Class for variables.
        ///
        class Variable : public TensorOp
        {
        public:
            /// \brief Constructions a tensor view-typed variable node.
            ///
            /// \param element_type The element type of the variable.
            /// \param pshape The partial shape of the variable.
            Variable(const nnfusion::element::Type& element_type, const nnfusion::Shape& shape);
            bool is_variable() const override { return true; }
        };
    }
}
