// Microsoft (c) 2019, Wenxiang
#pragma once

#include "../../core/codegenerator.hpp"
#include "../../core/common.hpp"
#include "../../core/op.hpp"

namespace nnfusion
{
    namespace rocm
    {
        class CudatoROCM : public ICodeGeneratorPass
        {
        public:
            bool run(ir::Operator_p& inter_op);
            static bool convert(LanguageUnit_p p);
        };
    }
}