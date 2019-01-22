// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_helper.hpp"

using namespace ngraph::runtime::nnfusion::codegen;

shared_ptr<LanguageUnit> cuda::get_math_kernel(const std::string& name,
                                               const std::string& math_kernel,
                                               const std::vector<std::string>& data_types)
{
    shared_ptr<LanguageUnit> cw(new LanguageUnit("function_def_inline_" + name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "__device__ __forceinline__ " << data_types[num_inputs] << " " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; i++)
        {
            writer << data_types[i] << " x" << i << ", ";
        }
        writer << data_types[num_inputs - 1] << " x" << num_inputs - 1;
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "return " + math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}

uint32_t cuda::align_to_block_size(uint32_t threads, uint32_t block_size)
{
    if (threads > (1u << 31) - 1)
    {
        throw std::runtime_error("Cuda can't handle threads > 2^31 - 1.");
    }
    uint32_t r = (threads + block_size - 1) / block_size;
    return r;
}

void cuda::emit_memcpyDtD(CodeWriter& writer,
                          const TensorWrapper& dst,
                          const TensorWrapper& src,
                          size_t buffer_size)
{
    if (buffer_size == 0)
    {
        writer << "cudaMemcpy(" << dst.get_name() << ", " << src.get_name() << ", "
               << dst.get_size() << " * " << dst.get_element_type().size()
               << ", cudaMemcpyHostToHost);\n";
        return;
    }
    writer << "cudaMemcpy(" << dst.get_name() << ", " << src.get_name() << ", " << buffer_size
           << ", cudaMemcpyHostToHost);\n";
    return;
}