// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegen.hpp"

bool ngraph::runtime::nnfusion::CodeGenerator::codegen(std::shared_ptr<IntermediateOP>& inter_op)
{
    return ICodeGeneratorPass::run_passes(this->pass_manager, inter_op);
}

bool ngraph::runtime::nnfusion::CodeGenerator::codegen(
    std::shared_ptr<std::vector<std::shared_ptr<IntermediateOP>>> inter_ops)
{
    bool rc = true;
    for (auto& op : *inter_ops)
    {
        rc = this->codegen(op);
        if(!rc) break;
    }
    return rc;
}

bool ngraph::runtime::nnfusion::CodeGenerator::codegen(std::shared_ptr<TranslationUnitMap> tum)
{
    
}

ngraph::runtime::nnfusion::CodeGenerator::CodeGenerator()
    : default_ctx(new CodeGeneratorContext)
{
    pass_manager.push_back(std::shared_ptr<ICodeGeneratorPass>(new CudaCodeGen()));
}