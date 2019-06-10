// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "backend.hpp"
#include "nnfusion/engine/engine.h"
#include "nnfusion/engine/interpreter.h"

namespace nnfusion
{
    class cuda_codegen : public nnfusion_Backend
    {
    public:
        cuda_codegen();
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
        shared_ptr<Interpreter> m_functrans;
    };
}