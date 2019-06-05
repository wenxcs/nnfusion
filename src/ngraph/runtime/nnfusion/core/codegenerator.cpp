// Microsoft (c) 2019, Wenxiang Hu
#include "codegenerator.hpp"

bool CodeGenerator::codegen(ir::Operator_p& inter_op)
{
    return ICodeGeneratorPass::run_passes(*(this->pass_manager), inter_op);
}

bool CodeGenerator::codegen(shared_ptr<vector<ir::Operator_p>> inter_ops)
{
    bool rc = true;
    for (auto& op : *inter_ops)
    {
        rc = this->codegen(op);
        if (!rc)
            break;
    }
    return rc;
}

bool CodeGenerator::codegen(shared_ptr<TranslationUnitMap> tum)
{
    enforce_not_nullptr(tum);
    auto& tu_map = *tum;
    for (auto& it : tu_map)
    {
        if (codegen(it.second))
            return false;
    }
    return true;
}

CodeGenerator::CodeGenerator()
    : default_ctx(new CodeGeneratorContext)
    , pass_manager(new vector<shared_ptr<ICodeGeneratorPass>>())
{
}

CodeGenerator::CodeGenerator(shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_mgr_ref,
                             shared_ptr<CodeGeneratorContext> ctx)
{
    this->pass_manager = pass_mgr_ref;
    this->default_ctx = ctx;
}

bool CodeGenerator::append_pass(shared_ptr<ICodeGeneratorPass> pass)
{
    enforce_not_nullptr(this->pass_manager);
    if (pass == nullptr)
    {
        LOG_WARN << "The pass manager is adding a null pass.";
        return false;
    }
    this->pass_manager->push_back(pass);
    return true;
}