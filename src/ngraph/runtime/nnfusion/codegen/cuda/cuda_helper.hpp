
// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_kernelops.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    void get_math_kernel(CodeWriter& writer,
                                         const std::string& name,
                                         const std::string& math_kernel,
                                         const std::vector<std::string>& data_types)
                    {
                        if (math_kernel.size())
                        {
                            auto num_inputs = data_types.size() - 1;
                            writer << "__device__ __forceinline__ " << data_types[num_inputs] << " "
                                   << name << "(";
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
                            writer << "}\n\n";
                        }
                        return;
                    }
                }
            }
        }
    }
}