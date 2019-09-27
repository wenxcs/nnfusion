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

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::Broadcast::Broadcast(const std::string& name,
                         const NodeVector& args,
                         const Shape& shape,
                         const AxisSet& broadcast_axes)
    : Op(name, check_single_output_args(args))
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
    constructor_validate_and_infer_types();
}

op::Broadcast::Broadcast(const shared_ptr<Node>& arg,
                         const Shape& shape,
                         const AxisSet& broadcast_axes)
    : Broadcast("Broadcast", {arg}, shape, broadcast_axes)
{
}

void op::Broadcast::validate_and_infer_types()
{
    infer_shape();

    inner_or_outer_broadcast();

    for (auto axis : m_broadcast_axes)
    {
        NODE_VALIDATION_ASSERT(this, axis < m_shape.size())
            << "Broadcast axis index (" << axis << ") exceeds specified output shape rank "
            << "(broadcast axes: " << m_broadcast_axes << ", output shape: " << m_shape << ").";
    }

    Shape required_input_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        required_input_shape.erase(required_input_shape.begin() + *i);
    }

    // TODO(amprocte): We can probably have a more helpful error message here.
    // There are two things that can go wrong, which are being picked up in
    // one fell swoop by this check: either the number of broadcast axes is not
    // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(0).compatible(required_input_shape))
        << "Broadcast argument shape, specified output shape, and axes are incompatible "
        << "(argument shape: " << get_input_partial_shape(0) << ", output shape: " << m_shape
        << ", broadcast axes: " << m_broadcast_axes << ").";

    set_output_type(0, get_input_element_type(0), m_shape);
}

void op::Broadcast::inner_or_outer_broadcast()
{
    AxisSet outer_axes;
    size_t rest_size = 1;
    bool count_size_only = false;
    for (size_t i = 0; i < m_shape.size(); i++)
    {
        if (m_broadcast_axes.count(i) > 0 && !count_size_only)
            outer_axes.insert(i);
        else
        {
            count_size_only = true;
            rest_size *= m_shape[i];
        }
    }
    if (outer_axes.size() == m_broadcast_axes.size())
    {
        m_is_outer_broadcast = true;
        m_outer_bc_size = rest_size;
        return;
    }

    AxisSet inner_axes;
    for (size_t i = m_shape.size() - 1; i >= 0; i--)
    {
        if (m_broadcast_axes.count(i) > 0)
            inner_axes.insert(i);
        else
            break;
    }
    if (inner_axes.size() == m_broadcast_axes.size())
    {
        m_is_inner_broadcast = true;
        size_t size = 1;
        for (auto d : inner_axes)
            size *= m_shape[d];
        m_inner_bc_size = size;
        return;
    }
}

shared_ptr<Node> op::Broadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
}

void op::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, m_broadcast_axes));
}

op::BroadcastLike::BroadcastLike(const std::shared_ptr<Node>& arg,
                                 const std::shared_ptr<Node>& like_arg,
                                 const AxisSet& broadcast_axes)
    : Broadcast("BroadcastLike", {arg, like_arg}, {}, {})
    , m_initial_broadcast_axes(broadcast_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::BroadcastLike::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<BroadcastLike>(new_args.at(0), new_args.at(1), m_initial_broadcast_axes);
}

void op::BroadcastLike::infer_shape()
{
    const Shape& in_shape = get_input_shape(0);
    m_shape = get_input_shape(1);
    m_broadcast_axes = m_initial_broadcast_axes;
    if (m_broadcast_axes.size() == 0)
    {
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            if (i < in_shape.size())
            {
                if (in_shape.at(i) == 1 && m_shape.at(i) > 1)
                {
                    m_broadcast_axes.insert(i);
                }
            }
            else
            {
                m_broadcast_axes.insert(i);
            }
        }
    }
}
