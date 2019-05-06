// Microsoft (c) 2019, Wenxiang
#include "../cuda/cuda_codegen.hpp"
#include "pass/cuda_to_rocm_pass.hpp"

using namespace nnfusion::cuda;

namespace nnfusion
{
    namespace rocm
    {
        class ROCM_NaiveCudaCodeGenerator : public nnfusion::NaiveCudaCodeGenerator
        {
        public:
            ROCM_NaiveCudaCodeGenerator()
                : nnfusion::NaiveCudaCodeGenerator()
            {
                append_pass(shared_ptr<ICodeGeneratorPass>(new CudatoROCM()));
            }

            ROCM_NaiveCudaCodeGenerator(
                shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_mgr_ref,
                shared_ptr<CodeGeneratorContext> ctx)
                : nnfusion::NaiveCudaCodeGenerator(pass_mgr_ref, ctx)
            {
                append_pass(shared_ptr<ICodeGeneratorPass>(new CudatoROCM()));
            }

            bool codegen(shared_ptr<TranslationUnit> tu) override;
            bool projgen() override;
            bool setpwd() override;
        };
    }
}