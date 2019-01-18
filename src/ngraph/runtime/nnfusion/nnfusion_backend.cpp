// Microsoft (c) 2019
// Wenxiang Hu

#include "ngraph/runtime/nnfusion/nnfusion_backend.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nnfusion;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::nnfusion::cuda_codegen();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::nnfusion::cuda_codegen::cuda_codegen()
    : nnfusion_Backend()
    , m_functrans(new FunctionTranslator)
    , m_codegen(new CodeGenerator)
{
}

bool runtime::nnfusion::cuda_codegen::codegen(std::shared_ptr<Function> func)
{
    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(func);
        assert_bool(m_codegen->codegen(tus));
    }
    return true;
}

// Unimplement Functions for codegen backend
bool runtime::nnfusion::cuda_codegen::compile(std::shared_ptr<Function> func)
{
    std::cout << "Unimplemented function compile() for cuda_codegen backend;" << std::endl;
    return this->codegen(func);
}

bool runtime::nnfusion::cuda_codegen::call(
    std::shared_ptr<Function> func,
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    std::cout << "Unimplemented function call() for cuda_codegen backend;" << std::endl;
    bool rc = true;

    validate_call(func, outputs, inputs);

    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        rc = compile(func);
    }

    return rc;
}

std::shared_ptr<runtime::Tensor>
    runtime::nnfusion::cuda_codegen::create_tensor(const element::Type& element_type,
                                                   const Shape& shape)
{
    std::cout << "Unimplemented function create_tensor() for cuda_codegen backend;" << std::endl;
    return nullptr;
}

std::shared_ptr<runtime::Tensor> runtime::nnfusion::cuda_codegen::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    std::cout << "Unimplemented function create_tensor() for cuda_codegen backend;" << std::endl;
    return nullptr;
}