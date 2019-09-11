// Microsoft (c) 2019, NNFusion Team

#include "graph_pass.hpp"
#include "manager.hpp"

#include "assign_layout_pass.hpp"
#include "constant_folding_pass.hpp"
#include "generate_result_op_pass.hpp"
#include "gradient_weight_mapping_pass.hpp"
#include "liveness_pass.hpp"
#include "memory_layout_pass.hpp"
#include "reshape_inplace_pass.hpp"
#include "runtime_const_folding_pass.hpp"
#include "vector_dot_transpose_pass.hpp"

using namespace nnfusion::graph::pass;
using namespace std;

bool GraphPass::run(std::shared_ptr<Graph> graph)
{
    GraphPassManager pass_manager;
    // GenerateResultOpPass must before LivenessPass
    // Generate result is implemented in gradient weight mapping pass
    pass_manager.register_pass<GradientWeightMappingPass>();
    //pass_manager.register_pass<GenerateResultOpPass>();
    //pass_manager.register_pass<ConstantFoldingPass>();
    pass_manager.register_pass<RuntimeConstantFoldingPass>();
    pass_manager.register_pass<VectorDotTransposePass>();
    pass_manager.register_pass<AssignLayoutPass>();
    pass_manager.register_pass<LivenessPass>();
    pass_manager.register_pass<MemoryLayoutPass>(64);
    pass_manager.register_pass<ReshapeInplacePass>();
    pass_manager.run_passes(graph);

    return true;
}
