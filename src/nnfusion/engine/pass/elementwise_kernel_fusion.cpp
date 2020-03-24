// Microsoft (c) 2019, NNFUSION TEAM
#include "elementwise_kernel_fusion.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace std;
using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool ElementwiseKernelFusion::run(std::shared_ptr<InterpreterContext> ctx,
                                  std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    for (auto block_iter : p)
    {
        if (block_iter->hasAttribute("fusion_group_id"))
        {
            std::vector<shared_ptr<KernelEmitter>> block_kernels;
            bool all_kernel_emitted = true;
            NNFusion_DeviceType dev_type;
            for (auto ins : *block_iter)
            {
                auto gnode = ins->getGNode();
                NNFUSION_CHECK(!gnode->get_op_ptr()->is_tensor_op());

                auto emitted_kernels =
                    (*ins)["Kernel_Selection_Result"]
                        .as<vector<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>>();
                auto emitter_iter =
                    find_if(emitted_kernels.begin(),
                            emitted_kernels.end(),
                            [this](pair<NNFusion_DeviceType, KernelEmitter::Pointer>& i) {
                                return (i.first == NNFusion_DeviceType::CUDA_GPU ||
                                        i.first == NNFusion_DeviceType::ROCM_GPU);
                            });

                KernelEmitter::Pointer kernel = nullptr;

                if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr ||
                    emitter_iter->second->get_or_emit_source() == nullptr)
                {
                    NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                                   << gnode->get_name();
                    all_kernel_emitted = false;
                    break;
                }
                else
                {
                    kernel = emitter_iter->second;
                    dev_type = emitter_iter->first;
                    block_kernels.push_back(kernel);
                }
            }

            if (all_kernel_emitted)
            {
                auto kernel_reg = KernelRegistry::Global()->FindKernelRegistration(
                    "ElementWiseFused", CUDA_GPU, DT_FLOAT);
                NNFUSION_CHECK_NOT_NULLPTR(kernel_reg);
                auto ctx = std::make_shared<KernelContext>();
                ctx->kernels = block_kernels;
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();

                nnfusion::ir::Instruction::Pointer ins(new nnfusion::ir::Instruction);
                ins->setName("fused_kernel");
                auto fused_op = std::make_shared<op::NoOp>("fused_kernel");
                auto fused_node = std::make_shared<nnfusion::graph::GNode>(
                    fused_op, nnfusion::graph::GNodeVector());
                ins->setGNode(fused_node);
                (*ins)["Kernel_Selection_Result"] =
                    vector<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
                auto& res = (*ins)["Kernel_Selection_Result"]
                                .as<vector<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>>();
                res.push_back(std::make_pair(dev_type, kernel));
                block_iter->clear();
                block_iter->push_back(ins);
            }
        }
    }

    return true;
}