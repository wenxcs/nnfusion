// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_cudacodegen.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegen.hpp"
#include "ngraph/runtime/nnfusion/pass/codegen/naive_unit_test_dump.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nnfusion;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    runtime::Backend* backend = nullptr;
    string type(configuration_string);

    auto colon = type.find(":");
    if (colon != type.npos)
    {
        string config = type.substr(colon + 1, type.length() - colon);
        if (config == "naive_unittest")
        {
            class cuda_codegen_naive_unittest : public cuda_codegen
            {
            public:
                cuda_codegen_naive_unittest()
                    : cuda_codegen()
                {
                    assert_nullptr(this->m_codegen);
                    this->m_codegen->append_pass(shared_ptr<ICodeGeneratorPass>(
                        new ngraph::runtime::nnfusion::codegen::NaiveUnitTestDump()));
                }
            };
            backend = new cuda_codegen_naive_unittest();
        }
        else
        {
            assert_bool(false) << "Unknown config for cuda_codegen backend.";
        }
    }
    else
    {
        backend = new runtime::nnfusion::cuda_codegen();
    }

    return backend;
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
    shared_ptr<CodeGeneratorContext> ctx(new CodeGeneratorContext);
    shared_ptr<vector<shared_ptr<ICodeGeneratorPass>>> pass_manager(
        new vector<shared_ptr<ICodeGeneratorPass>>());
    pass_manager->push_back(std::shared_ptr<ICodeGeneratorPass>(new CudaCodeGen()));

    std::shared_ptr<CodeGenerator> m_codegen(new CodeGenerator(pass_manager, ctx));
    this->m_codegen = m_codegen;
}

bool runtime::nnfusion::cuda_codegen::codegen(std::shared_ptr<Function> func)
{
    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(func);
        assert_nullptr(tus);
        assert_bool(m_codegen->codegen(tus));
    }
    return true;
}

// Unimplement Functions for codegen backend
bool runtime::nnfusion::cuda_codegen::compile(std::shared_ptr<Function> func)
{
    NGRAPH_DEBUG << "Unimplemented function compile() for cuda_codegen backend;" << std::endl;
    return this->codegen(func);
}

bool runtime::nnfusion::cuda_codegen::call(
    std::shared_ptr<Function> func,
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    NGRAPH_DEBUG << "Unimplemented function call() for cuda_codegen backend;" << std::endl;
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
    NGRAPH_DEBUG << "Unimplemented function create_tensor() for cuda_codegen backend;" << std::endl;
    return nullptr;
}

std::shared_ptr<runtime::Tensor> runtime::nnfusion::cuda_codegen::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    NGRAPH_DEBUG << "Unimplemented function create_tensor() for cuda_codegen backend;" << std::endl;
    return nullptr;
}