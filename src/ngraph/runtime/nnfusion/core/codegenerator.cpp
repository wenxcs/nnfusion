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
    assert_nullptr(tum);
    auto& tu_map = *tum;
    for (auto& it : tu_map)
    {
        codegen(it.second);
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
    assert_nullptr(this->pass_manager);
    assert_nullptr(pass);
    this->pass_manager->push_back(pass);
}