// Microsoft (c) 2019, Wenxiang Hu
#include "nnfusion_cudacodegen.hpp"
#include "cuda/cuda_codegen.hpp"
#include "pass/codegen/naive_unit_test_dump.hpp"

extern "C" const char* get_ngraph_version_string()
{
    return "nnfusion_cuda_codegen";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    runtime::Backend* backend = nullptr;
    string type(configuration_string);

    class cuda_codegen_naive_graph_test : public cuda_codegen
    {
    public:
        cuda_codegen_naive_graph_test()
            : cuda_codegen()
        {
            this->m_codegen = shared_ptr<NaiveCudaCodeGenerator>(new NaiveCudaCodeGenerator);
        }
    };
    backend = new cuda_codegen_naive_graph_test();

    return backend;
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

cuda_codegen::cuda_codegen()
    : nnfusion_Backend()
    , m_functrans(new FunctionTranslator)
    , m_codegen(new CudaCodeGenerator)
{
}

bool cuda_codegen::codegen(shared_ptr<Function> func)
{
    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(func);
        enforce_not_nullptr(tus);
        enforce(m_codegen->codegen(tus));
    }
    return true;
}

// Unimplement Functions for codegen backend
bool cuda_codegen::compile(shared_ptr<Function> func)
{
    LOG_INFO << "Unimplemented function compile() for cuda_codegen backend;" << endl;
    return this->codegen(func);
}

bool cuda_codegen::call(shared_ptr<Function> func,
                        const vector<shared_ptr<runtime::Tensor>>& outputs,
                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    LOG_INFO << "Unimplemented function call() for cuda_codegen backend;" << endl;
    bool rc = true;

    validate_call(func, outputs, inputs);

    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        rc = compile(func);
    }

    return rc;
}

shared_ptr<runtime::Tensor> cuda_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape)
{
    LOG_INFO << "Unimplemented function create_tensor() for cuda_codegen backend;" << endl;
    return nullptr;
}

shared_ptr<runtime::Tensor> cuda_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape,
                                                        void* memory_pointer)
{
    LOG_INFO << "Unimplemented function create_tensor() for cuda_codegen backend;" << endl;
    return nullptr;
}