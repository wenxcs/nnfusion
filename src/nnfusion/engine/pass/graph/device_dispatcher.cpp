// Microsoft (c) 2019, NNFusion Team
#include "device_dispatcher.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

DECLARE_string(fdefault_device);

bool DefaultDeviceDispatcher::run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
{
    auto dev_name = FLAGS_fdefault_device.c_str();
    NNFusion_DeviceType dt = nnfusion::get_device_type(dev_name);
    /* for debug purpose
    switch(default_device)
    {
        case GENERIC_CPU:
        NNFUSION_LOG(INFO) << "GENERIC_CPU";
        break;
        case  ROCM_GPU:
        NNFUSION_LOG(INFO) << "ROCM_GPU";
        break;
        case CUDA_GPU:
        NNFUSION_LOG(INFO) << "CUDA_GPU";
    }
    */

    std::vector<std::shared_ptr<GNode>> nodes = graph->get_nodes();
    for (auto it : nodes)
    {
        it->Set<NNFusion_DeviceType>("Device", move(dt));
    }

    return true;
}