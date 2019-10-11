// Microsoft (c) 2019, NNFUSION TEAM
#include "elementwise_kernel_fusion.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "ngraph/op/noop.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

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
            DeviceType dev_type;
            for (auto ins : *block_iter)
            {
                auto node = ins->operatorDef();
                enforce(!(node->is_parameter()) && !(node->is_constant()));

                auto emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                           .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                auto emitter_iter = find_if(emitted_kernels.begin(),
                                            emitted_kernels.end(),
                                            [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                                return (i.first == DeviceType::CUDA_GPU ||
                                                        i.first == DeviceType::ROCM_GPU);
                                            });

                KernelEmitter::Pointer kernel = nullptr;

                if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr ||
                    emitter_iter->second->get_or_emit_source() == nullptr)
                {
                    LOG_WARN << "Kernel should be emitted before this pass:" << node->get_name();
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
                enforce_not_nullptr(kernel_reg);
                auto ctx = std::make_shared<KernelContext>();
                ctx->kernels = block_kernels;
                auto kernel = kernel_reg->m_factory(ctx);
                kernel->get_or_emit_source();

                nnfusion::ir::Instruction::Pointer ins(new nnfusion::ir::Instruction);
                ins->setName("fused_kernel");
                auto fused_op = std::make_shared<op::NoOp>("fused_kernel");
                ins->setOperatorDef(fused_op);
                (*ins)["Kernel_Selection_Result"] =
                    vector<pair<DeviceType, KernelEmitter::Pointer>>();
                auto& res = (*ins)["Kernel_Selection_Result"]
                                .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                res.push_back(std::make_pair(dev_type, kernel));
                block_iter->clear();
                block_iter->push_back(ins);
            }
        }
    }

    return true;
}