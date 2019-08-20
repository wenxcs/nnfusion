// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#pragma once
#include "binary_utils.hpp"
#include "profiling_runtime.hpp"

namespace nnfusion
{
    namespace profiler
    {
        class CudaDefaultRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<CudaDefaultRuntime>;

        public:
            static Pointer Runtime();
            double
                execute(const ProfilingContext::Pointer& ke, void** input, void** output) override;

        protected:
            // Tiny codegen function for runtime
            virtual bool codegen(const ProfilingContext::Pointer& ke);
            virtual bool compile(const ProfilingContext::Pointer& ke);
        };
    }
}