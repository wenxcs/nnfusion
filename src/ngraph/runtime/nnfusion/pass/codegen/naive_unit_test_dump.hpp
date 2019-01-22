// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                class NaiveUnitTestDump : public ICodeGeneratorPass
                {
                public:
                    bool run(std::shared_ptr<ngraph::runtime::nnfusion::IntermediateOP>& inter_op);
                };
            }
        }
    }
}