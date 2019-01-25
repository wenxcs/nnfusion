// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegenop.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nnfusion::codegen::cuda;

#define LU_DEFINE(NAME) extern shared_ptr<LanguageUnit> NAME;

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
                    namespace header
                    {
                        LU_DEFINE(cuda);
                        LU_DEFINE(stdio);
                    }

                    namespace macro
                    {
                        LU_DEFINE(NNFUSION_DEBUG);
                    }

                    namespace declaration
                    {
                        LU_DEFINE(typedef_int);
                    }
                }
            }
        }
    }
}

#undef LU_DEFINE