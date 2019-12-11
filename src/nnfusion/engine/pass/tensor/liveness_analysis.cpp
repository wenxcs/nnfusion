// Microsoft (c) 2019, NNFUSION TEAM
#include "liveness_analysis.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

//#include "ngraph/op/constant.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"

using namespace std;
using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool TensorLivenessAnalysis::run(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    // persistent_tensors: can't be modified and can't reuse other tensors's memory
    std::unordered_set<ngraph::descriptor::Tensor*> persistent_tensors;
    // Deprecate
    std::unordered_set<ngraph::descriptor::Tensor*> output_tensors;
    // constant_tensors: can't be modified while can reuse other tensors's momory
    std::unordered_set<ngraph::descriptor::Tensor*> constant_tensors;
    std::unordered_map<std::shared_ptr<nnfusion::graph::GNode>, KernelEmitter::Pointer> op_kernels;

    auto& p = tu->program;
    for (auto block_iter : p)
    {
        for (auto ins : *block_iter)
        {
            auto gnode = ins->getGNode();
            if (!gnode->get_op_ptr()->is_parameter() && !gnode->get_op_ptr()->is_output() &&
                !gnode->get_op_ptr()->is_constant())
            {
                auto emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                           .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
                auto emitter_iter = find_if(emitted_kernels.begin(),
                                            emitted_kernels.end(),
                                            [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                                return (i.first == DeviceType::CUDA_GPU ||
                                                        i.first == DeviceType::ROCM_GPU);
                                            });

                if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr ||
                    emitter_iter->second->get_or_emit_source() == nullptr)
                {
                    CHECK_FAIL() << "Kernel should be emitted before this pass:"
                                 << gnode->get_name();
                }
                op_kernels[gnode] = emitter_iter->second;
            }

            if (gnode->get_op_ptr()->is_parameter())
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = gnode->get_output_tensor(i);
                    persistent_tensors.insert(&tensor);
                }
            }
            if (gnode->get_op_ptr()->is_output())
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = gnode->get_output_tensor(i);
                    persistent_tensors.insert(&tensor);
                    output_tensors.insert(&tensor);
                }
                for (size_t i = 0; i < gnode->get_input_size(); ++i)
                {
                    auto& tensor = gnode->get_input_tensor(i);
                    constant_tensors.insert(&tensor);
                }
            }
            if (auto constant_node =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(gnode->get_op_ptr()))
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = gnode->get_output_tensor(i);
                    constant_tensors.insert(&tensor);
                }
            }
        }
    }

    std::unordered_set<ngraph::descriptor::Tensor*> currently_live;

    // traverse instructions in reverse order
    for (auto block_it = p.rbegin(); block_it != p.rend(); block_it++)
    {
        auto block_p = *block_it;
        for (auto ins_it = block_p->rbegin(); ins_it != block_p->rend(); ins_it++)
        {
            auto ins = *ins_it;

            auto gnode = ins->getGNode();
            gnode->liveness_new_list.clear();
            gnode->liveness_free_list.clear();

            std::unordered_set<ngraph::descriptor::Tensor*> input_tensor_decls;
            std::unordered_set<ngraph::descriptor::Tensor*> output_tensor_decls;

            if (gnode->get_op_ptr()->is_parameter() || gnode->get_op_ptr()->is_output() ||
                gnode->is_constant())
            {
                for (size_t i = 0; i < gnode->get_input_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = gnode->get_input_tensor(i);
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        input_tensor_decls.insert(&tensor);
                    }
                }

                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = gnode->get_output_tensor(i);
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        output_tensor_decls.insert(&tensor);
                    }
                }
            }
            else
            {
                auto kernel = op_kernels[gnode];
                auto kernel_context = kernel->m_context;

                for (size_t i = 0; i < kernel_context->inputs.size(); i++)
                {
                    auto& tw = kernel_context->inputs[i];
                    auto& tensor = (descriptor::Tensor&)tw.get_tensor();
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        input_tensor_decls.insert(&tensor);
                    }
                }

                for (size_t i = 0; i < kernel_context->outputs.size(); i++)
                {
                    auto& tw = kernel_context->outputs[i];
                    auto& tensor = (descriptor::Tensor&)tw.get_tensor();
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        output_tensor_decls.insert(&tensor);
                    }
                }
            }

            std::unordered_set<ngraph::descriptor::Tensor*> free_tensor_decls;
            std::unordered_set<ngraph::descriptor::Tensor*> new_tensor_decls;
            std::unordered_set<ngraph::descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
            all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

            for (ngraph::descriptor::Tensor* tensor_decl : all_tensor_decls)
            {
                if (currently_live.find(tensor_decl) == currently_live.end())
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_decl);
                    if (output_tensors.find(tensor_decl) == output_tensors.end() &&
                        constant_tensors.find(tensor_decl) == constant_tensors.end())
                    {
                        // Don't free output tensors and constant tensors
                        free_tensor_decls.insert(tensor_decl);
                    }
                }
            }

            for (ngraph::descriptor::Tensor* output_decl : output_tensor_decls)
            {
                auto currently_live_it = currently_live.find(output_decl);
                if (currently_live_it != currently_live.end())
                {
                    new_tensor_decls.insert(output_decl);
                    currently_live.erase(currently_live_it);
                }
            }
            gnode->liveness_free_list = free_tensor_decls;
            gnode->liveness_new_list = new_tensor_decls;
        }
    }

    return true;
}