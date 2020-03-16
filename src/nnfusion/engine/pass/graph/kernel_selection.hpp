// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class ProfilingBasedKernelSelector : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

                pair<NNFusion_DeiveType, nnfusion::kernels::KernelEmitter::Pointer>
                    profiling_best(shared_ptr<GNode> gnode,
                                   NNFusion_DeiveType devtype,
                                   nnfusion::profiler::IProfilingRuntime::Pointer runtime);
            };

            class DefaultKernelSelector : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;
                pair<NNFusion_DeiveType, nnfusion::kernels::KernelEmitter::Pointer>
                    pick_first(shared_ptr<GNode> gnode, NNFusion_DeiveType devtype);
                pair<NNFusion_DeiveType, nnfusion::kernels::KernelEmitter::Pointer>
                    pick_first_rocm(shared_ptr<GNode> gnode);
            };
        }
    }

} // namespace nnfusion