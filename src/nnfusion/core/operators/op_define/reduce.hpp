// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace graph
    {
        class Graph;
    }
    namespace op
    {
        /// \brief Tensor reduction operation.
        ///
        /// Element-wise reduces the input tensor, eliminating the specified reduction axes, given a reduction graph that maps two scalars to a scalar.
        /// For example, if the reduction graph \f$f(x,y) = x+y\f$:
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{0\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 3 + 5), (2 + 4 + 6) \right] =
        ///     \left[ 9, 12 \right]~~~\text{(dimension 0 (rows) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///     \left[ (1 + 2), (3 + 4), (5 + 6) \right] =
        ///     \left[ 3, 7, 11 \right]~~~\text{(dimension 1 (columns) is eliminated)}
        /// \f]
        ///
        /// \f[
        ///     \mathit{reduce}\left(f,\{0,1\},
        ///         \left[ \begin{array}{ccc}
        ///                1 & 2 \\ 3 & 4 \\ 5 & 6 \end{array} \right]\right) =
        ///      (1 + 2) + (3 + 4) + (5 + 6) =
        ///      21~~~\text{(both dimensions (rows and columns) are eliminated)}
        /// \f]
        ///
        /// It is assumed that \f$f\f$ is associative. In other words, the order of operations is undefined. In the case where a collapsed dimension is 0,
        /// the value of `arg_init` will be substituted.
        ///
        /// Note that the parameter `reduction_axes` specifies which axes are to be <i>eliminated</i>, which can be a bit counterintuitive. For example,
        /// as seen above, eliminating the column dimension results in the <i>rows</i> being summed, not the columns.
        ///
        /// ## Parameters
        ///
        /// |                      | Description                                                                                                               |
        /// | -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
        /// | `reduction_graph` | The scalar graph used to reduce the input tensor. Must take two arguments of type \f$E[]\f$ and return type \f$E[]\f$. |
        /// | `reduction_axes`     | The axes to eliminate through reduction.                                                                                  |
        ///
        /// ## Inputs
        ///
        /// |                | Type                              | Description                                                                                           |
        /// | -------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
        /// | `arg_reductee` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any shape, with the element type matching that expected by the reduction graph. |
        /// | `arg_init`     | \f$E[]\f$                         | An scalar to be used as a substitute output value on zero-sized axes.                                 |
        ///
        /// ## Output
        ///
        /// | Type                                      | Description                                                                                                      |
        /// | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[\textit{delete}(A,d_1,\dots,d_n)]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with the `reduction_axes` \f$A\f$ eliminated by reduction. |
        class Reduce : public Op
        {
        public:
            /// \brief Constructs a reduction operation.
            ///
            /// \param arg_reductee The tensor view to be reduced.
            /// \param arg_init The initial value for reduction.
            /// \param reduction_graph The reduction graph to use.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Reduce(const std::shared_ptr<graph::Graph>& reduction_graph,
                   const nnfusion::AxisSet& reduction_axes);

            /// \return A one-element vector containing the graph to use for reduction.
            std::vector<std::shared_ptr<graph::Graph>> get_graphs() const
            {
                return std::vector<std::shared_ptr<graph::Graph>>{m_reduction_graph};
            }
            /// \return The axis positions (0-based) to be eliminated through reduction.
            const nnfusion::AxisSet& get_reduction_axes() const { return m_reduction_axes; }
        protected:
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            std::shared_ptr<graph::Graph> m_reduction_graph;
            nnfusion::AxisSet m_reduction_axes;
        };
    }
}