// Microsoft (c) 2019, Wenxiang
#include "cuda_helper.hpp"

using CodeWriter = ngraph::codegen::CodeWriter;

LanguageUnit_p cuda::get_math_kernel(const std::string& name,
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
        writer << "CUDA_SAFE_CALL(cudaMemcpy(" << dst.get_name() << ", " << src.get_name() << ", "
               << dst.get_size() << " * " << dst.get_element_type().size()
               << ", cudaMemcpyDeviceToDevice));\n";
        return;
    }
    writer << "CUDA_SAFE_CALL(cudaMemcpy(" << dst.get_name() << ", " << src.get_name() << ", "
           << buffer_size << ", cudaMemcpyDeviceToDevice));\n";
    return;
}

void cuda::coordinate_transform_to_multi_d(CodeWriter& writer,
                                           std::string i_strides,
                                           std::string i_stride_magic,
                                           std::string i_stride_shift,
                                           std::string i_coord_product,
                                           std::string o_coordinates,
                                           size_t rank,
                                           bool register_arguments)
{
    std::string brace_open = (register_arguments) ? "" : "[";
    std::string brace_close = (register_arguments) ? "" : "]";

    // Translation from flat index to dense tensor coordinates:
    // Given tensor shape [d0 d1 ... dN] with strides [d1*...*dN, d2*...*dN, ... 1],
    // calculate coordinates as:
    //
    //  product = tid
    //  d0 = product/stride[0]
    //  product = product % stride[0]
    //  d1 = product/stride[1]
    //  ...
    writer << "int coordinate_product = " << i_coord_product << ";\n";
    for (size_t i = 0; i < rank; i++)
    {
        if (i != 0)
        {
            writer << "coordinate_product -= (" << o_coordinates << i - 1 << " * " << i_strides
                   << brace_open << i - 1 << brace_close << ");\n";
        }
        writer << "int " << o_coordinates << i << " = division_by_invariant_multiplication("
               << "coordinate_product, " << i_stride_magic << brace_open << i << brace_close << ", "
               << i_stride_shift << brace_open << i << brace_close << ");\n";
    }
}

std::string cuda::collective_coordinate_transform_helper(CodeWriter& writer,
                                                         std::string i_thread_index,
                                                         std::string i_strides,
                                                         std::string i_stride_magic,
                                                         std::string i_stride_shift,
                                                         std::string i_reduced_strides,
                                                         std::string o_coordinates,
                                                         size_t rank,
                                                         bool register_arguments)
{
    coordinate_transform_to_multi_d(writer,
                                    i_strides,
                                    i_stride_magic,
                                    i_stride_shift,
                                    i_thread_index,
                                    o_coordinates,
                                    rank,
                                    register_arguments);

    std::string brace_open = (register_arguments) ? "" : "[";
    std::string brace_close = (register_arguments) ? "" : "]";

    // index into reduced tensor from coordinates of non-reduced tensor
    std::string reduced_idx = "reduced_idx";
    writer << "int " << reduced_idx << " = 0;\n";
    for (size_t i = 0; i < rank; i++)
    {
        writer << "reduced_idx += " << o_coordinates << i << " * " << i_reduced_strides
               << brace_open << i << brace_close << ";\n";
    }

    return reduced_idx;
}