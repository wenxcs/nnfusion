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
    std::unordered_set<ngraph::descriptor::Tensor*> persistent_tensors;
    std::unordered_set<ngraph::descriptor::Tensor*> output_tensors;
    std::unordered_set<ngraph::descriptor::Tensor*> constant_tensors;
    std::unordered_map<std::shared_ptr<ngraph::Node>, KernelEmitter::Pointer> op_kernels;

    auto& p = tu->program;
    for (auto block_iter : p)
    {
        for (auto ins : *block_iter)
        {
            auto node = ins->operatorDef();
            if (!node->is_parameter() && !node->is_output() && !node->is_constant())
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
                    enforce(false) << "Kernel should be emitted before this pass:"
                                   << node->get_name();
                }
                op_kernels[node] = emitter_iter->second;
            }

            if (node->is_parameter())
            {
                for (size_t i = 0; i < node->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
                    persistent_tensors.insert(&tensor);
                }
            }
            if (node->is_output())
            {
                for (size_t i = 0; i < node->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
                    persistent_tensors.insert(&tensor);
                    output_tensors.insert(&tensor);
                }
                for (auto& input_decl : node->get_inputs())
                {
                    auto& tensor = input_decl.get_tensor();
                    constant_tensors.insert(&tensor);
                }
            }
            if (auto constant_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node))
            {
                for (size_t i = 0; i < node->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
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

            const std::shared_ptr<ngraph::Node>& node = ins->operatorDef();
            node->liveness_new_list.clear();
            node->liveness_free_list.clear();

            std::unordered_set<ngraph::descriptor::Tensor*> input_tensor_decls;
            std::unordered_set<ngraph::descriptor::Tensor*> output_tensor_decls;

            if (node->is_parameter() || node->is_output() || node->is_constant())
            {
                for (ngraph::descriptor::Input& input_decl : node->get_inputs())
                {
                    ngraph::descriptor::Tensor& tensor = input_decl.get_tensor();
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        input_tensor_decls.insert(&tensor);
                    }
                }

                for (size_t i = 0; i < node->get_output_size(); ++i)
                {
                    ngraph::descriptor::Tensor& tensor = node->get_output_tensor(i);
                    if (persistent_tensors.find(&tensor) == persistent_tensors.end())
                    {
                        output_tensor_decls.insert(&tensor);
                    }
                }
            }
            else
            {
                auto kernel = op_kernels[node];
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
            node->liveness_free_list = free_tensor_decls;
            node->liveness_new_list = new_tensor_decls;
        }
    }

    return true;
}