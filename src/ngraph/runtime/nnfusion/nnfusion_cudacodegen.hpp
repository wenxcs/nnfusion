// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_backend.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class cuda_codegen : public nnfusion_Backend
            {
            public:
                cuda_codegen();
                bool codegen(std::shared_ptr<Function> func);
                bool compile(std::shared_ptr<Function> func);
                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);
                std::shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                               const Shape& shape);
                std::shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                               const Shape& shape,
                                                               void* memory_pointer);

            private:
                std::map<std::shared_ptr<Function>, TranslationUnit> m_function_map;

            protected:
                std::shared_ptr<CodeGenerator> m_codegen;
                std::shared_ptr<FunctionTranslator> m_functrans;
            };
        }
    }
}