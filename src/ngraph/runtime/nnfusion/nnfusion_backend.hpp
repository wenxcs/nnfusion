// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            // This is an abstract class for NNFusion Backend
            class nnfusion_Backend : public Backend
            {
            public:
                nnfusion_Backend()
                    : runtime::Backend(){};
                virtual bool codegen(std::shared_ptr<Function> func) = 0;
            };
        }
    }
}
