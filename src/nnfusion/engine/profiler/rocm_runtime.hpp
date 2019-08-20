// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Profiler::RocmRuntime for creating a Compiler to profile the kernel.
 * This runtime will overwiter some methods of Cuda runtime
 * \author wenxh
 */

#include "cuda_runtime.hpp"

namespace nnfusion
{
    namespace profiler
    {
        class RocmDefaultRuntime : public CudaDefaultRuntime
        {
        public:
            using Pointer = shared_ptr<RocmDefaultRuntime>;

        public:
            static Pointer Runtime();
            double
                execute(const ProfilingContext::Pointer& ke, void** input, void** output) override;
            bool check_env();

        private:
            // Tiny codegen function for runtime
            // bool codegen(const ProfilingContext::Pointer& ke) override;
            bool hipfy(const ProfilingContext::Pointer& ke);
            bool compile(const ProfilingContext::Pointer& ke) override;
        };
    }
}
