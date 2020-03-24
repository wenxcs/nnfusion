// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../util/tensor_op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief A graph parameter.
        ///
        /// Parameters are nodes that represent the arguments that will be passed to user-defined graphs.
        /// Function creation requires a sequence of parameters.
        /// Basic graph operations do not need parameters attached to a graph.
        class Parameter : public TensorOp
        {
        public:
            /// \brief Constructions a tensor view-typed parameter node.
            ///
            /// \param element_type The element type of the parameter.
            /// \param shape The shape of the parameter.
            /// \param cacheable True if the parameter is not expected to be frequently updated.
            Parameter(const nnfusion::element::Type& element_type,
                      const nnfusion::Shape& shape,
                      const bool cacheable = false);

            bool get_cacheable() const { return m_cacheable; }
            bool is_parameter() const override { return true; }
        protected:
            bool m_cacheable;
        };
    }
}
