// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "result.hpp"
#include "util/gpu_util.hpp"
#include "util/nvshape.hpp"

using namespace ngraph::runtime::gpu;

namespace nnfusion
{
    namespace ir
    {
        class Broadcast : public Operator
        {
        public:
            Shape arg_shape;
            Shape result_shape;
            bool isMemcpy;
            AxisSet axes;

            // calculate strides
            ngraph::NVShape strides;
            // precacluate invariants for integer division via multiplication
            std::vector<int> stride_magic;
            std::vector<int> stride_shift;
            // calculate reduced tensor strides with 0s inserted for reduced axes
            ngraph::NVShape reduced_shape;
            ngraph::NVShape reduced_strides;

            // TODO: blending factors are not currently implemented
            float alpha = 1.0f;
            float beta = 0.0f;

            size_t rank;

        public:
            Broadcast(shared_ptr<Node> node);
            static Operator_p translate(shared_ptr<Node> node);
        };

        using Broadcast_p = shared_ptr<Broadcast>;
    }
}