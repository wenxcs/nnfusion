// Microsoft (c) 2019, Wenxiang
#pragma once
#include "cuda_function.hpp"
#include "cuda_ops.hpp"

namespace nnfusion
{
    class CudaCodeGenPass : public ICodeGeneratorPass
    {
    public:
        bool run(ir::Operator_p& inter_op);
    };

    class CudaCodeGenerator : public CodeGenerator
    {
    public:
        CudaCodeGenerator()
            : CodeGenerator()
        {
            append_pass(shared_ptr<ICodeGeneratorPass>(new CudaCodeGenPass()));
        }

        CudaCodeGenerator(shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_mgr_ref,
                          shared_ptr<CodeGeneratorContext> ctx)
            : CodeGenerator(pass_mgr_ref, ctx)
        {
            //<TODO:wenxh> need to check wether CudaCodeGenPass exists.
            append_pass(shared_ptr<ICodeGeneratorPass>(new CudaCodeGenPass()));
        }

        bool codegen(shared_ptr<TranslationUnit> tu) override;
    };

    class NaiveCudaCodeGenerator : public CodeGenerator
    {
    public:
        NaiveCudaCodeGenerator()
            : CodeGenerator()
        {
            append_pass(shared_ptr<ICodeGeneratorPass>(new CudaCodeGenPass()));
        }

        NaiveCudaCodeGenerator(shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_mgr_ref,
                               shared_ptr<CodeGeneratorContext> ctx)
            : CodeGenerator(pass_mgr_ref, ctx)
        {
            //<TODO:wenxh> need to check wether CudaCodeGenPass exists.
            append_pass(shared_ptr<ICodeGeneratorPass>(new CudaCodeGenPass()));
        }

        bool codegen(shared_ptr<TranslationUnit> tu) override;
    };
}