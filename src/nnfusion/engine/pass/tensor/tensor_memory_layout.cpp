// Microsoft (c) 2019, NNFUSION TEAM
#include "tensor_memory_layout.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "ngraph/log.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/util.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace std;
using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool AssignTensorMemoryLayout::run(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    MemoryAllocatorFactory maf(m_alignment, m_disable_memory_sharing);

    auto is_same_dev = [](const descriptor::Tensor* a, const descriptor::Tensor* b) {
        return (a->get_device_type() == b->get_device_type()) &&
               (a->get_device_id() == b->get_device_id());
    };

    std::unordered_set<descriptor::Tensor*> persistent_tensors;

    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            auto node = ins->operatorDef();

            auto emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                       .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
            auto emitter_iter =
                find_if(emitted_kernels.begin(),
                        emitted_kernels.end(),
                        [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                            return (i.first == CUDA_GPU || i.first == DeviceType::ROCM_GPU);
                        });

            KernelEmitter::Pointer kernel = nullptr;

            if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr)
                // Can assign tensor layout even kernel is not emitted.
                LOG(WARNING) << "Kernel should be emitted before this pass:" << node->get_name();
            else
                kernel = emitter_iter->second;

            // Tensors should be considered
            // Node: inputs outputs
            // Kernel Context: +tensors

            std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
            std::set<const descriptor::Tensor*> reused_inputs;
            std::unordered_set<descriptor::Tensor*> alloc_temp;

            if (kernel != nullptr)
            {
                CHECK_NOT_NULLPTR(kernel->m_context);
                // Allocate NoneResuseable Space for Persistent Tensors
                for (auto& tensorwrapper : kernel->m_context->tensors)
                {
                    // todo: make get_tensor() interface return un-const variable.
                    auto& tensor = (descriptor::Tensor&)tensorwrapper.get_tensor();
                    if (tensor.is_persistent())
                        persistent_tensors.insert(&tensor);
                    else
                        alloc_temp.insert(&tensor);
                }
            }

            if (auto op = std::dynamic_pointer_cast<op::Op>(node))
            {
                // concat in_place_oi should be treated differently
                if (!std::dynamic_pointer_cast<op::Concat>(node))
                {
                    if (auto op_annotations = op->get_op_annotations())
                    {
                        for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                        {
                            auto output = &node->get_outputs().at(oi_pair.output).get_tensor();
                            auto input = &node->get_inputs().at(oi_pair.input).get_tensor();
                            auto input_node =
                                node->get_inputs().at(oi_pair.input).get_output().get_node();

                            // should not overwrite constant tensor
                            // if (std::dynamic_pointer_cast<op::Constant>(input_node))
                            //     continue;

                            if (!is_same_dev(input, output))
                            {
                                LOG(WARNING)
                                    << "Tensor inplace oi pairs are not in same device, ignored.";
                                continue;
                            }

                            // For destructive kernel, this should be the last use
                            // Non-destructive kernels can pass through if memory sharing is disabled
                            if ((node->liveness_free_list.count(input) != 0 ||
                                 (m_disable_memory_sharing && !oi_pair.destructive &&
                                  !input_node->is_parameter() && !input_node->is_constant())) &&
                                node->liveness_new_list.count(output) != 0)
                            {
                                in_place_outputs.insert({output, input});
                                reused_inputs.insert(input);
                            }
                        }
                    }
                }
            }

            unordered_set<descriptor::Tensor*> newlist(alloc_temp);
            newlist.insert(node->liveness_new_list.begin(), node->liveness_new_list.end());
            for (descriptor::Tensor* tensor : newlist)
            {
                auto allocator = maf.get_allocator(tensor);
                if (in_place_outputs.count(tensor))
                {
                    size_t offset = in_place_outputs.at(tensor)->get_pool_offset();
                    allocator->allocate(tensor, offset);
                }
                else
                {
                    allocator->allocate(tensor);
                }
            }

            if (!m_disable_memory_sharing)
            {
                unordered_set<descriptor::Tensor*> freelist(alloc_temp);
                freelist.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
                for (descriptor::Tensor* tensor : freelist)
                {
                    if (reused_inputs.count(tensor) == 0)
                    {
                        auto allocator = maf.get_allocator(tensor);
                        allocator->free(tensor);
                    }
                }
            }
        }
    }
    return true;
}
