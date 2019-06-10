// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/engine/engine.h"
#include "nnfusion/engine/interpreter.h"
#include "nnfusion/engine/op.h"

namespace nnfusion
{
    class ICodeGeneratorPass
    {
    public:
        virtual bool run(ir::Operator_p& inter_op) = 0;

        static bool run_passes(const vector<shared_ptr<ICodeGeneratorPass>>& passes,
                               ir::Operator_p& inter_op)
        {
            bool rc = true;
            for (const auto& pass : passes)
            {
                rc = pass->run(inter_op);
                if (!rc)
                    break;
            }
            return rc;
        }
    };

    class CodeGeneratorContext
    {
    public:
        unordered_map<ir::Operator*, string> fun_src_buffer;
        unordered_map<string, string> m_variable_name_map;
        map<string, size_t> m_name_index_map;
        /*unordered_map<shared_ptr<Function>, list<shared_ptr<Node>>>
            m_function_ordered_ops;*/
    };

    class CodeGenerator
    {
    public:
        CodeGenerator();
        CodeGenerator(shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_mgr_ref,
                      shared_ptr<CodeGeneratorContext> ctx);

        virtual bool codegen(shared_ptr<TranslationUnitMap> inter_op);
        virtual bool codegen(shared_ptr<TranslationUnit> tu) = 0;
        virtual bool codegen(ir::Operator_p& inter_op);
        virtual bool codegen(shared_ptr<vector<ir::Operator_p>> inter_ops);
        bool append_pass(shared_ptr<ICodeGeneratorPass> pass);

    protected:
        shared_ptr<CodeGeneratorContext> default_ctx;
        shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_manager;
    };
}