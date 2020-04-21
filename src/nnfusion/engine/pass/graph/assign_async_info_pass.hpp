// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class AssignAsyncInfoPass : public GraphPassBase
            {
            public:
                AssignAsyncInfoPass();
                bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph) override;

            private:
                void assign_thread_info(nnfusion::async::AsyncManager* CPU_async_manager,
                                        std::shared_ptr<Graph>& graph);
                void naive_assign_stream_info(nnfusion::async::AsyncManager* async_manager,
                                              std::shared_ptr<Graph>& graph);
                void assign_event_info(nnfusion::async::AsyncManager* CUDA_async_manager,
                                       nnfusion::async::AsyncManager* CPU_async_manager,
                                       std::shared_ptr<Graph>& graph);
                KernelEmitter::Pointer get_kernel(std::shared_ptr<nnfusion::graph::GNode> gnode);
                NNFusion_DeviceType m_device;
            };
        }
    }
}