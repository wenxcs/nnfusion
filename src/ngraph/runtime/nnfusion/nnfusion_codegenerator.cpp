// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegen.hpp"

bool ngraph::runtime::nnfusion::CodeGenerator::codegen(std::shared_ptr<IntermediateOP>& inter_op)
{
    ICodeGeneratorPass::run_passes(this->pass_manager, inter_op);
    return true;
}

bool ngraph::runtime::nnfusion::CodeGenerator::codegen(
    std::shared_ptr<std::vector<std::shared_ptr<IntermediateOP>>> inter_ops)
{
    for (auto& op : *inter_ops)
    {
        this->codegen(op);
    }
    return true;
}

ngraph::runtime::nnfusion::CodeGenerator::CodeGenerator()
    : default_ctx(new CodeGeneratorContext)
{
    pass_manager.push_back(std::shared_ptr<ICodeGeneratorPass>(new CudaCodeGen()));
}