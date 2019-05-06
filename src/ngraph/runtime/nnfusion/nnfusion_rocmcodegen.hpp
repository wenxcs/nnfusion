// Microsoft (c) 2019, Wenxiang Hu, Wei Cui
#pragma once

#include "core/backend.hpp"
#include "core/codegenerator.hpp"
#include "core/common.hpp"
#include "core/interpreter.hpp"

namespace nnfusion
{
    class rocm_codegen : public nnfusion_Backend
    {
    public:
        rocm_codegen();
        bool codegen(shared_ptr<Function> func);
        bool compile(shared_ptr<Function> func);
        bool call(shared_ptr<Function> func,
                  const vector<shared_ptr<runtime::Tensor>>& outputs,
                  const vector<shared_ptr<runtime::Tensor>>& inputs);
        shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                  const Shape& shape);
        shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                  const Shape& shape,
                                                  void* memory_pointer);

    private:
        map<shared_ptr<Function>, TranslationUnit> m_function_map;

    protected:
        shared_ptr<CodeGenerator> m_codegen;
        shared_ptr<FunctionTranslator> m_functrans;
    };
}