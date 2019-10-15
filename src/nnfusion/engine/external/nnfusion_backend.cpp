// Microsoft (c) 2019, Wenxiang Hu
#include "nnfusion_backend.hpp"

extern "C" const char* get_ngraph_version_string()
{
    return "nnfusion_engine";
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    runtime::Backend* backend = nullptr;
    string type(configuration_string);

    auto colon = type.find(":");
    if (colon != type.npos)
    {
        string config = type.substr(colon + 1, type.length() - colon);
        if (config == "generic_cpu")
            default_device = nnfusion::GENERIC_CPU;
        else if (config == "rocm_gpu")
            default_device = nnfusion::ROCM_GPU;
        else
            default_device = nnfusion::CUDA_GPU;
    }
    else
        default_device = nnfusion::CUDA_GPU;

    backend = new cuda_codegen();
    return backend;
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

cuda_codegen::cuda_codegen()
    : nnfusion_Backend()
    , m_functrans(new Interpreter)
{
}

bool cuda_codegen::codegen(shared_ptr<Function> func)
{
    TranslationUnit& func_unit = m_function_map[func];
    if (func_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(func);
        CHECK_NOT_NULLPTR(tus);
    }
    return true;
}

bool cuda_codegen::codegen(shared_ptr<graph::Graph> graph)
{
    TranslationUnit& graph_unit = m_graph_map[graph];
    if (graph_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(graph);
        CHECK_NOT_NULLPTR(tus);
    }
    return true;
}

// Unimplement Functions for codegen backend
bool cuda_codegen::compile(shared_ptr<Function> func)
{
    LOG(INFO) << "Unimplemented function compile() for cuda_codegen backend;" << endl;
    return this->codegen(func);
}

bool cuda_codegen::call(shared_ptr<Function> func,
                        const vector<shared_ptr<runtime::Tensor>>& outputs,
                        const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    LOG(INFO) << "Unimplemented function call() for cuda_codegen backend;" << endl;
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
    LOG(INFO) << "Unimplemented function create_tensor() for cuda_codegen backend;" << endl;
    return nullptr;
}

shared_ptr<runtime::Tensor> cuda_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape,
                                                        void* memory_pointer)
{
    LOG(INFO) << "Unimplemented function create_tensor() for cuda_codegen backend;" << endl;
    return nullptr;
}