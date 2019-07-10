// Microsoft (c) 2019, NNFusion Team
#include <iostream>
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

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
        namespace cpu
        {
            class Constant : public KernelEmitter
            {
            public:
                Constant(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cpu")
                {
                    op = static_pointer_cast<ngraph::op::Constant>(ctx->node);
                    enforce_not_nullptr(op) << "Node type is not Constant.";

                    folder = "./Constant/";
                    create_dir(folder);
                    const_name = ctx->outputs[0].get_name();
                    ofstream bin_file(folder + const_name + ".bin", ios::out | ios::binary);
                    bin_file.write((const char*)op->get_data_ptr(), op->get_data_size());
                    bin_file.close();
                    op->get_friendly_name();

                    std::stringstream tag;
                    tag << "load_" << const_name;
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;
                    writer << "std::ifstream bin_file(\"" << folder + const_name
                           << ".bin\" , std::ios::in | std::ios::binary);\n"
                           << "bin_file.read((char*)output0, " << op->get_data_size() << ");\n"
                           << "bin_file.close();\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
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
REGISTER_KERNEL_EMITTER("Constant",                                   //op_name
                        Device(GENERIC_CPU).TypeConstraint(DT_FLOAT), //attrs
                        cpu::Constant)                                // constructor