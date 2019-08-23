// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Basic Test Example for AvgPool;
 * \author wenxh
 */

#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/pad.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

TEST(nnfusion_core_kernels, sample)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Pad>();
    EXPECT_TRUE(node != nullptr);

    // Filter out the kernels meeting the requirement;
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(node->description(), CUDA_GPU, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(node));

    EXPECT_GT(kernel_regs.size(), 0);
    bool has_valid_kernel = false;
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;

            LOG_INFO << "Now on Kernel Emitter:\t" << kernel->get_function_name();

            // Data
            auto input = nnfusion::inventory::generate_input<op::Pad, float>(0);
            auto output = nnfusion::inventory::generate_output<op::Pad, float>(0);
            vector<vector<float>> inputs;
            inputs.push_back(vector<float>{/*a*/ 1, 2, 3, 4, 5, 6});
            inputs.push_back(vector<float>{/*b*/ 9});
            vector<vector<float>> outputs;
            outputs.push_back(output);
            // Context
            nnfusion::profiler::ProfilingContext::Pointer pctx =
                make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            auto rocm_runtime = nnfusion::profiler::RocmDefaultRuntime::Runtime();
            // Rocm
            if (rocm_runtime->check_env())
            {
                LOG_INFO << "Test ROCM runtime of Pad operator:";
                nnfusion::profiler::Profiler prof(rocm_runtime, pctx);
                prof.execute();
                LOG_INFO << "Avg Host duration:" << pctx->result.get_host_avg();
                LOG_INFO << "Avg Device duration:" << pctx->result.get_device_avg();
                auto res = prof.execute(inputs);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(ngraph::test::all_close_f(res[i], outputs[i]));
            }
            // Cuda
            if (true /*Check Cuda Runtime*/)
            {
                // Now we use the tool to profile and test the kernel;
                pctx->reset();
                LOG_INFO << "Test Cuda runtime of Pad operator:";
                nnfusion::profiler::Profiler prof(nnfusion::profiler::CudaDefaultRuntime::Runtime(),
                                                  pctx);
                prof.execute();
                LOG_INFO << "Avg Host duration:" << pctx->result.get_host_avg();
                LOG_INFO << "Avg Device duration:" << pctx->result.get_device_avg();

                auto res = prof.execute(inputs);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(ngraph::test::all_close_f(res[i], outputs[i]));
            }

            // Cpu Reference
            if (true /*Must support*/)
            {
                pctx->reset();
                LOG_INFO << "Test CPU Reference runtime of Pad operator:";
                nnfusion::profiler::Profiler ref_prof(
                    nnfusion::profiler::ReferenceRuntime::Runtime(), pctx);
                auto res = ref_prof.execute(inputs);
                EXPECT_EQ(res.size(), outputs.size());
                for (int i = 0; i < res.size(); i++)
                    EXPECT_TRUE(ngraph::test::all_close_f(res[i], outputs[i]));
            }
        }
    }

    EXPECT_TRUE(has_valid_kernel);
}