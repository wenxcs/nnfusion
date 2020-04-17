// Microsoft (c) 2019, NNFusion Team
#include "cuda_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

LanguageUnit_p cuda::CudaEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    set_launch_config();

    auto gnode = m_context->gnode;
    string stream_name = "0";
    if (gnode && (*gnode)["Async_info"].is_valid())
    {
        auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
        if (async_info.execution_stream != nullptr)
            stream_name = async_info.execution_stream->get_name();
    }

    //set stream during codegen
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
       << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, " << stream_name
       << ">>>(" << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    set_launch_config();
    emit_function_body();
    lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
       << ") __global__ void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::BlockCudaEmitter::emit_device_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }
    params.push_back("int thread_id");
    params.push_back("int block_id");
    params.push_back("char *shared_buffer");

    lu << "__device__ __noinline__ void " << m_kernel_name << "_block_kernel"
       << "(" << join(params, ", ") << ")";
    return _lu;
}

LanguageUnit_p cuda::BlockCudaEmitter::emit_device_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_body"));
    auto& lu = *_lu;

    int block_size = m_blockDim.x * m_blockDim.y * m_blockDim.z;
    int block_num = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    is_emitting_block_kernel = true;
    FunctionUnit_p fu = this->get_or_emit_source();
    is_emitting_block_kernel = false;

    lu << "if (thread_id >= " << block_size << ")";
    lu.block_begin();
    if (num_local_thread_sync > 0)
    {
        lu << "for (int i = 0; i < " << num_local_thread_sync << "; i++) __syncthreads();\n";
    }
    lu << "return;\n";
    lu.block_end();

    lu << "const dim3 blockDim(" << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z
       << ");\n";
    lu << "const dim3 gridDim(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z
       << ");\n";

    if (m_blockDim.y != 1 && m_blockDim.z == 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", thread_id / "
           << m_blockDim.x << ", 0);\n";
    }
    else if (m_blockDim.y == 1 && m_blockDim.z != 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", 0, thread_id / "
           << m_blockDim.x << ");\n";
    }
    else if (m_blockDim.y != 1 && m_blockDim.z != 1)
    {
        lu << "const dim3 threadIdx(thread_id % " << m_blockDim.x << ", thread_id / "
           << m_blockDim.x << " % " << m_blockDim.y << ", thread_id / "
           << m_blockDim.x * m_blockDim.y << ");\n";
    }

    if (m_gridDim.y == 1 && m_gridDim.z == 1)
    {
        lu << "const dim3 blockIdx(block_id, 0, 0);\n";
    }
    else if (m_gridDim.z == 1)
    {
        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / " << m_gridDim.x
           << ", 0);\n";
    }
    else
    {
        lu << "const dim3 blockIdx(block_id % " << m_gridDim.x << ", block_id / " << m_gridDim.x
           << " % " << m_gridDim.y << ", block_id / " << m_gridDim.x * m_gridDim.y << ");\n";
    }

    lu << fu->body_unit->get_code() << "\n";

    return _lu;
}

const std::unordered_map<std::string, size_t> cuda::BlockCudaEmitter::size_of_str_type{
    {"char", sizeof(char)},
    {"float", sizeof(float)},
    {"double", sizeof(double)},
    {"int8_t", sizeof(int8_t)},
    {"int16_t", sizeof(int16_t)},
    {"int32_t", sizeof(int32_t)},
    {"int64_t", sizeof(int64_t)},
    {"uint8_t", sizeof(uint8_t)},
    {"uint16_t", sizeof(uint16_t)},
    {"uint32_t", sizeof(uint32_t)},
    {"uint64_t", sizeof(uint64_t)}};
