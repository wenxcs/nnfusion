// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    class CudaCodeGenOP : public CodeGenOP
                    {
                    public:
                        CudaCodeGenOP() {}
                        CudaCodeGenOP(shared_ptr<IntermediateOP> inter_op)
                            : CodeGenOP(inter_op)
                        {
                        }

                        shared_ptr<LanguageUnit> codegen_test() override;
                    };
                }
            }
        }
    }
}