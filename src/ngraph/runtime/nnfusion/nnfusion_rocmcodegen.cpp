// Microsoft (c) 2019, Wenxiang Hu, Wei Cui
#include "nnfusion_rocmcodegen.hpp"
#include "cuda/cuda_codegen.hpp"
#include "cuda/cuda_langunit.hpp"
#include "pass/codegen/naive_unit_test_dump.hpp"
#include "rocm/rocm_codegen.hpp"

extern "C" const char* get_ngraph_version_string()
{
    return "nnfusion_rocm_codegen";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    runtime::Backend* backend = nullptr;
    string type(configuration_string);

    class rocm_codegen_naive_graph_test : public rocm_codegen
    {
    public:
        rocm_codegen_naive_graph_test()
            : rocm_codegen()
        {
            this->m_codegen = shared_ptr<rocm::ROCM_NaiveCudaCodeGenerator>(
                new rocm::ROCM_NaiveCudaCodeGenerator);
        }
    };
    backend = new rocm_codegen_naive_graph_test();
    return backend;
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

rocm_codegen::rocm_codegen()
    : nnfusion_Backend()
    , m_functrans(new FunctionTranslator)
    , m_codegen(new CudaCodeGenerator)
{
}

bool rocm_codegen::codegen(shared_ptr<Function> func)
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
bool rocm_codegen::compile(shared_ptr<Function> func)
{
    NGRAPH_DEBUG << "Unimplemented function compile() for rocm_codegen backend;" << endl;
    return this->codegen(func);
}

bool rocm_codegen::call(shared_ptr<Function> func,
                        const vector<shared_ptr<runtime::Tensor>>& outputs,
                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    NGRAPH_DEBUG << "Unimplemented function call() for rocm_codegen backend;" << endl;
    bool rc = true;

    validate_call(func, outputs, inputs);

    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        rc = compile(func);
    }

    return rc;
}

shared_ptr<runtime::Tensor> rocm_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape)
{
    NGRAPH_DEBUG << "Unimplemented function create_tensor() for rocm_codegen backend;" << endl;
    return nullptr;
}

shared_ptr<runtime::Tensor> rocm_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape,
                                                        void* memory_pointer)
{
    NGRAPH_DEBUG << "Unimplemented function create_tensor() for rocm_codegen backend;" << endl;
    return nullptr;
}