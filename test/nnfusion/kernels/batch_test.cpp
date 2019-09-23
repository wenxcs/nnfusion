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
        template <>
        bool all_close<float>(const std::vector<float>& a, const std::vector<float>& b)
        {
            return ngraph::test::all_close_f(a, b);
        }

        template <>
        bool all_close<int>(const std::vector<int>& a, const std::vector<int>& b)
        {
            if (a.size() == b.size())
            {
                for (size_t i = 0; i < a.size(); i++)
                    if (a[i] != b[i])
                        return false;
                return true;
            }
            return false;
        }

        ///\todo Maybe a better/general way

        template <typename T, typename val_t = float>
        bool check_kernels(DeviceType dev_t, DataType data_t)
        {
            auto rt = get_default_runtime(dev_t);
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