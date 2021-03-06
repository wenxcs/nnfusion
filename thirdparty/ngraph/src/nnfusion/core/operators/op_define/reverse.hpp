//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Axis-reverse operation.
        ///
        /// Reverses the direction of zero or more axes in a tensor, where "reversing" an axis means that at the output tensor.
        ///
        /// ## Parameters
        ///
        /// |                 | Description              |
        /// | --------------- | ------------------------ |
        /// | `reversed_axes` | The axes to be reversed. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                            |
        /// | ----- | --------------------------------- | -------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | An input tensor of any type and shape. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                                                                               |
        /// | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg}[j_1,\dots,j_n]\f$ and \f$j_k = d_k - i_k - 1\f$ if axis \f$k\f$ is in the reverse set; else \f$j_k = i_k\f$. |
        class Reverse : public Op
        {
        public:
            /// \brief Constructs a reverse operation.
            ///
            /// \param reversed_axes The axes to reverse.
            Reverse(const nnfusion::AxisSet& reversed_axes);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The set of axes to reverse.
            const nnfusion::AxisSet& get_reversed_axes() const { return m_reversed_axes; }
        protected:
            const nnfusion::AxisSet m_reversed_axes;
        };
    }
}
