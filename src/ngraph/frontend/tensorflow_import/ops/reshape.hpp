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

#include "../tensorflow_base.hpp"
#include "../util/util.hpp"
#include "ngraph/node_vector.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace tensorflow_import
        {
            NamedNodeVector TranslateReshapeOp(const tensorflow::NodeDef&,
                                               const NodeMap&,
                                               ngraph::op::ParameterVector&);
        } // namespace tensorflow_import
    }     // namespace frontend
} // namespace ngraph
