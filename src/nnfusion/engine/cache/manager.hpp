// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <string>

#include <pwd.h>
#include <sqlite3.h>

#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;

namespace nnfusion
{
    namespace cache
    {
        // presently only kernel cache database supported
        // Todo: integrate the interfaces of profiling cache database
        class KernelCacheManager
        {
        public:
            KernelCacheManager();
            ~KernelCacheManager();

            std::string fetch(std::string identifier, std::string tag);
            inline bool is_valid() { return valid; }
        private:
            bool valid;
            std::string m_path;
            static sqlite3* kernel_cache;
        };
    } //namespace cache
} //namespace nnfusion