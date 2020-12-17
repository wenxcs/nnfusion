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
            /// \param require_grad True if the parameter requires grad.
            Parameter(const nnfusion::element::Type& element_type,
                      const nnfusion::Shape& shape,
                      const bool cacheable = false,
                      bool require_grad = false);

            bool get_cacheable() const { return m_cacheable; }
            bool is_parameter() const override { return true; }
            bool require_grad() const { return m_require_grad; }
            void set_require_grad(bool value = true) { m_require_grad = value; }
        protected:
            bool m_cacheable;
            bool m_require_grad;
        };
    }
}
