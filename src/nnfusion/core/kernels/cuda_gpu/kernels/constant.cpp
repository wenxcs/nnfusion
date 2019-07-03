// Microsoft (c) 2019, NNFusion Team
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

#include <bits/stdc++.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static bool create_dir(std::string tar_path)
{
    bool flag;
    int mkdir_status;
    struct stat s;
    int err = stat(tar_path.c_str(), &s);
    if (-1 == err)
    {
        mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == mkdir_status)
        {
            printf("Error creating directory: %s", (tar_path).c_str());
            flag = false;
        }
        else
            flag = true;
    }
    else
    {
        flag = true;
    }
    return flag;
}

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Constant : public KernelEmitter
            {
            public:
                Constant(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda_sp")
                {
                    op = static_pointer_cast<ngraph::op::Constant>(ctx->node);
                    enforce_not_nullptr(op) << "Node type is not Constant.";

                    folder = "./Constant/";
                    create_dir(folder);
                    const_name = ctx->outputs[0].get_name();
                    ofstream bin_file(folder + const_name + ".bin", ios::out | ios::binary);
                    bin_file.write((const char*)op->get_data_ptr(), op->get_data_size());
                    bin_file.close();

                    std::stringstream tag;
                    tag << "_" << const_name;
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;
                    writer << "std::ifstream bin_file(\"" << folder + const_name
                           << ".bin\" , std::ios::in | std::ios::binary);\n"
                           // << "cudaMalloc((void**)out, " << op->get_data_size() << ");\n"
                           << "char* tmp_mem = new char[" << op->get_data_size() << "];\n"
                           << "bin_file.read(tmp_mem, " << op->get_data_size() << ");\n"
                           << "cudaMemcpy(output0, tmp_mem, " << op->get_data_size()
                           << ", cudaMemcpyHostToDevice);\n"
                           << "bin_file.close();\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::fstream);
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<ngraph::op::Constant> op;
                string folder;
                string const_name;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Constant",                                //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT), //attrs
                        cuda::Constant)                            // constructor