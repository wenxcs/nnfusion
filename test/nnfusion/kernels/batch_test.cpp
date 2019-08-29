// Microsoft (c) 2019, MSRA/NNFUSION Team
///\brief Batch tests for our kernels.
///
///\author wenxh, ziming

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/pad.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::inventory;
using namespace nnfusion::profiler;

namespace nnfusion
{
    namespace test
    {
        ///\todo Maybe a better/general way
        IProfilingRuntime::Pointer gen_runtime(DeviceType dev_t)
        {
            IProfilingRuntime::Pointer ip = nullptr;
            switch (dev_t)
            {
            case CUDA_GPU: ip = CudaDefaultRuntime::Runtime(); break;
            case ROCM_GPU: ip = CudaDefaultRuntime::Runtime(); break;
            case GENERIC_CPU: ip = ReferenceRuntime::Runtime(); break;
            }
            if (ip != nullptr && ip->check_env())
                return ip;
            return nullptr;
        }

        bool check_kernel(shared_ptr<ngraph::Node> node,
                          DeviceType dev_t,
                          const vector<float>& IN,
                          const vector<float>& OUT)
        {
            auto rt = CudaDefaultRuntime::Runtime();
            std::vector<shared_ptr<const KernelRegistration>> available_kernels =
                KernelRegistry::Global()->FindKernelRegistrations(
                    node->description(), CUDA_GPU, DT_FLOAT);
            shared_ptr<KernelContext> ctx(new KernelContext(node));
            bool kernel_found = false;
            for (auto& kernel_reg : available_kernels)
            {
                auto kernel = kernel_reg->m_factory(ctx);
                if (kernel->get_or_emit_source())
                {
                    kernel_found = true;
                    auto pctx = make_shared<ProfilingContext>(kernel);
                    Profiler prof(rt, pctx);

                    // The execute() will return a vector of vector,
                    // we only compare the first one with our ground
                    // truth
                    auto res = prof.unsafe_execute<float>((void*)IN.data());
                    if (res.empty())
                        return false;
                    auto& res_first = res[0];

                    if (res_first.size() != OUT.size())
                        return false;

                    if (!ngraph::test::all_close_f(res_first, OUT))
                        return false;

                    LOG_INFO << "Kernel pass unit-test.";
                }
                else
                {
                    LOG_WARN << "Kernel is not available.";
                }
            }
            if (!kernel_found)
                LOG_WARN << "No available found!";
            return kernel_found;
        }

        template <typename T, typename val_t = float>
        bool check_kernels(DeviceType dev_t, DataType data_t)
        {
            auto rt = gen_runtime(dev_t);
            if (rt == nullptr)
                return false;

            for (int case_id = 0;; case_id++)
            {
                auto node = create_object<T, val_t>(case_id);
                if (node == nullptr)
                    break;
                auto input = generate_input<T, val_t>(case_id);
                auto output = generate_output<T, val_t>(case_id);
                shared_ptr<KernelContext> ctx(new KernelContext(node));
                auto available_kernels = KernelRegistry::Global()->FindKernelRegistrations(
                    node->description(), dev_t, data_t);

                for (auto& kernel_reg : available_kernels)
                {
                    auto kernel = kernel_reg->m_factory(ctx);
                    if (kernel->get_or_emit_source())
                    {
                        auto pctx = make_shared<ProfilingContext>(kernel);
                        Profiler prof(rt, pctx);

                        // The execute() will return a vector of vector,
                        // we only compare the first one with our ground
                        // truth
                        auto res = prof.unsafe_execute<val_t>((void*)input.data());
                        if (res.empty())
                            return false;
                        auto& res_first = res[0];

                        if (res_first.size() != output.size())
                            return false;

                        if (!ngraph::test::all_close_f(res_first, output))
                            return false;
                    }
                    else
                    {
                        LOG_WARN << "Kernel is not available";
                    }
                }
            }
            return true;
        }
    }
}

///param: node, device_type, data_type ... etc
TEST(nnfusion_core_kernels, batch_kernel_tests)
{
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Pad>(GENERIC_CPU, DT_FLOAT));
    EXPECT_TRUE(nnfusion::test::check_kernels<op::Pad>(CUDA_GPU, DT_FLOAT));
}