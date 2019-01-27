// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../core/op.hpp"
#include "noop.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Result : public Operator
        {
        public:
            Result(shared_ptr<Node> node);

            static Operator_p translate(shared_ptr<Node> node);
        };

        using Result_p = shared_ptr<Result>;
    }
}