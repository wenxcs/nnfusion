// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/codegen/cuda/Elementwise.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/Result.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

using namespace std;
namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class CudaCodeGen : public ICodeGeneratorPass
            {
            public:
                bool run(std::shared_ptr<ngraph::runtime::nnfusion::IntermediateOP>& inter_op);
            };
        }
    }
}