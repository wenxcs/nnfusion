// Microsoft (c) 2019, Wenxiang
//#pragma once

#include "ngraph/runtime/nnfusion/codegen/cuda/Elementwise.hpp"

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
                    template <>
                    shared_ptr<LanguageUnit> cuda::Elementwise<ngraph::op::Relu>::codegen_test()
                    {
                        std::string name = codegen_test_name();
                        shared_ptr<LanguageUnit> cw(new LanguageUnit(name));
                        LanguageUnit& writer = *cw;
                        writer << "// Relu Test\n";

                        writer << "extern \"C\" bool " << name << "()";
                        writer.block_begin();
                        {
                            vector<float> data;
                            // Malloc
                            for (auto& arg : inter_op->args)
                            {
                                data = test_hostData(writer, arg);
                                test_cudaMalloc(writer, arg);
                                test_cudaMemcpyHtoD(writer, arg);
                            }

                            for (int i = 0; i < data.size(); i++)
                            {
                                if (data[i] < 0)
                                    data[i] = 0;
                            }
                            test_hostData(writer, inter_op->out[0], data);
                            test_cudaMalloc(writer, inter_op->out[0]);
                            writer << codegen_function_call()->get_code();
                            test_cudaMemcpyDtoH(writer, inter_op->out[0]);
                            test_compare(writer, inter_op->out[0]);
                            writer << "printf(\"SUCCEED\\n\");\n";
                            writer << "return true;\n";
                        }
                        writer.block_end();
                        return cw;
                    }
                }
            }
        }
    }
}