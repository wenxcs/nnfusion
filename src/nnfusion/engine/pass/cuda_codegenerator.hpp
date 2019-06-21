// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    class CudaCodeGenerator : public IInterpreterPass
    {
    public:
        bool run(std::shared_ptr<InterpreterContext> ctx,
                 std::shared_ptr<TranslationUnit> tu) override;

    private:
        virtual bool projgen();
        virtual bool setpwd();
        LanguageUnit_p lu_cmakefile, lu_nnfusion_rt, lu_header, lu_main;
    };
}