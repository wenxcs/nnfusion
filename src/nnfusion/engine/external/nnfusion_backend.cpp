// Microsoft (c) 2019, Wenxiang Hu
#include "nnfusion_backend.hpp"

DEFINE_int32(min_log_level,
             0,
             "Minimum logging level: 0 - debug; 1 - info; 2 - warning; 3 - error; 4 - fatal;");

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

bool cuda_codegen::codegen(shared_ptr<graph::Graph> graph)
{
    TranslationUnit& graph_unit = m_graph_map[graph];
    if (graph_unit.m_is_translated == false)
    {
        auto tus = m_functrans->translate(graph);
        NNFUSION_CHECK_NOT_NULLPTR(tus);
    }
    return true;
}

shared_ptr<runtime::Tensor> cuda_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape)
{
    NNFUSION_LOG(INFO) << "Unimplemented function create_tensor() for cuda_codegen backend;";
    return nullptr;
}

shared_ptr<runtime::Tensor> cuda_codegen::create_tensor(const element::Type& element_type,
                                                        const Shape& shape,
                                                        void* memory_pointer)
{
    NNFUSION_LOG(INFO) << "Unimplemented function create_tensor() for cuda_codegen backend;";
    return nullptr;
}