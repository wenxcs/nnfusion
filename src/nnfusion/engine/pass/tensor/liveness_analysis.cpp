// Microsoft (c) 2019, NNFUSION TEAM
#include "liveness_analysis.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;
using namespace nnfusion::async;

DEFINE_bool(frt_const_folding, false, "Add runtime constant folding.");
DECLARE_string(fdefault_device);

bool TensorLivenessAnalysis::run(std::shared_ptr<InterpreterContext> ctx,
                                 std::shared_ptr<TranslationUnit> tu)
{
    bool enable_rt_const_folding = FLAGS_frt_const_folding;
    std::unordered_map<std::shared_ptr<nnfusion::graph::GNode>, KernelEmitter::Pointer> op_kernels;
    std::unordered_set<shared_ptr<descriptor::Tensor>> persist_candidate;

    auto default_device = FLAGS_fdefault_device.c_str();
    NNFusion_DeviceType device_type = nnfusion::get_device_type(default_device);

    auto& p = tu->program;
    for (auto block_iter : p)
    {
        for (auto ins : *block_iter)
        {
            auto gnode = ins->getGNode();
            if (!(*gnode)["Async_info"].is_valid())
            {
                NNFUSION_CHECK_FAIL() << "Async info should be assigned before this pass："
                                      << gnode->get_name();
            }
            auto& async_info = (*gnode)["Async_info"].as<AsyncExecutionInfo>();
            std::shared_ptr<nnfusion::async::Stream> stream;

            if (device_type == GENERIC_CPU)
                stream = async_info.execution_thread;
            else if (device_type == CUDA_GPU || device_type == ROCM_GPU)
                stream = async_info.execution_stream;

            auto stream_id = stream->get_stream_id();

            if (!gnode->get_op_ptr()->is_tensor_op() && !gnode->get_op_ptr()->is_output())
            {
                auto emitted_kernel = (*ins)["Kernel_Selection_Result"]
                                          .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
                if (emitted_kernel.second->get_or_emit_source() == nullptr)
                {
                    NNFUSION_CHECK_FAIL() << "Kernel should be emitted before this pass:"
                                          << gnode->get_name();
                }
                op_kernels[gnode] = emitted_kernel.second;
                // add cross_stream tensor
                auto kernel_context = op_kernels[gnode]->m_context;
                for (size_t i = 0; i < kernel_context->inputs.size(); i++)
                {
                    auto tensor = kernel_context->inputs[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
                // set output tensor's group id
                for (size_t i = 0; i < kernel_context->outputs.size(); i++)
                {
                    auto tensor = kernel_context->outputs[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
                // set temp tensor's group id
                for (size_t i = 0; i < kernel_context->tensors.size(); i++)
                {
                    auto tensor = kernel_context->tensors[i];
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }

            if (gnode->is_parameter())
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    shared_ptr<descriptor::Tensor> tensor = gnode->get_output_tensor_ptr(i);
                    tensor->set_parameter();
                    // set tensor's group id
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            if (gnode->is_variable())
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    shared_ptr<descriptor::Tensor> tensor = gnode->get_output_tensor_ptr(i);
                    tensor->set_persistent();
                    // set tensor's group id
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            if (gnode->get_op_ptr()->is_output())
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    shared_ptr<descriptor::Tensor> tensor = gnode->get_output_tensor_ptr(i);
                    tensor->set_persistent();
                    // set tensor's group id
                    set_tensor_group(tensor, to_string(stream_id));
                }
                for (size_t i = 0; i < gnode->get_input_size(); ++i)
                {
                    shared_ptr<descriptor::Tensor> tensor = gnode->get_input_tensor_ptr(i);
                    tensor->set_persistent();
                    // add cross_stream tensor
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
            if (auto constant_node =
                    std::dynamic_pointer_cast<nnfusion::op::Constant>(gnode->get_op_ptr()))
            {
                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    shared_ptr<descriptor::Tensor> tensor = gnode->get_output_tensor_ptr(i);
                    if (enable_rt_const_folding)
                    {
                        persist_candidate.insert(tensor);
                    }
                    else
                    {
                        tensor->set_persistent();
                    }
                    // set tensor's group id
                    set_tensor_group(tensor, to_string(stream_id));
                }
            }
        }
    }

    if (enable_rt_const_folding)
    {
        for (auto block_iter : p)
        {
            for (auto ins : *block_iter)
            {
                auto gnode = ins->getGNode();
                if (gnode->get_op_ptr()->is_tensor_op() || gnode->get_op_ptr()->is_output())
                {
                    continue;
                }
                else
                {
                    bool is_all_const = true;
                    std::unordered_set<shared_ptr<descriptor::Tensor>> tmp;
                    auto kernel = op_kernels[gnode];
                    auto kernel_context = kernel->m_context;
                    for (size_t i = 0; i < kernel_context->inputs.size(); i++)
                    {
                        auto tensor = kernel_context->inputs[i];
                        if (persist_candidate.find(tensor) != persist_candidate.end())
                        {
                            tmp.insert(tensor);
                        }
                        else
                        {
                            is_all_const = false;
                        }
                    }

                    if (is_all_const)
                    {
                        for (size_t i = 0; i < kernel_context->outputs.size(); i++)
                        {
                            auto tensor = kernel_context->outputs[i];
                            persist_candidate.insert(tensor);
                        }
                    }
                    else
                    {
                        for (auto tensor : tmp)
                        {
                            tensor->set_persistent();
                        }
                    }
                }
            }
        }
    }

    std::unordered_set<shared_ptr<descriptor::Tensor>> currently_live;

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

            std::unordered_set<std::shared_ptr<descriptor::Tensor>> input_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> output_tensor_decls;

            if (gnode->get_op_ptr()->is_tensor_op() || gnode->get_op_ptr()->is_output())
            {
                for (size_t i = 0; i < gnode->get_input_size(); ++i)
                {
                    std::shared_ptr<descriptor::Tensor> tensor = gnode->get_input_tensor_ptr(i);
                    input_tensor_decls.insert(tensor);
                }

                for (size_t i = 0; i < gnode->get_output_size(); ++i)
                {
                    std::shared_ptr<descriptor::Tensor> tensor = gnode->get_output_tensor_ptr(i);
                    output_tensor_decls.insert(tensor);
                }
            }

            else
            {
                auto kernel = op_kernels[gnode];
                auto kernel_context = kernel->m_context;

                for (size_t i = 0; i < kernel_context->inputs.size(); i++)
                {
                    auto tensor = kernel_context->inputs[i];
                    input_tensor_decls.insert(tensor);
                }

                for (size_t i = 0; i < kernel_context->outputs.size(); i++)
                {
                    auto tensor = kernel_context->outputs[i];
                    output_tensor_decls.insert(tensor);
                }
            }

            std::unordered_set<std::shared_ptr<descriptor::Tensor>> free_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> new_tensor_decls;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> all_tensor_decls =
                input_tensor_decls;
            all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

            for (std::shared_ptr<descriptor::Tensor> tensor_decl : all_tensor_decls)
            {
                if (currently_live.find(tensor_decl) == currently_live.end())
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_decl);
                    if (cross_stream.find(tensor_decl) == cross_stream.end())
                        free_tensor_decls.insert(tensor_decl);
                }
            }

            for (std::shared_ptr<descriptor::Tensor> output_decl : output_tensor_decls)
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
void TensorLivenessAnalysis::set_tensor_group(shared_ptr<descriptor::Tensor> tensor,
                                              const std::string& group)
{
    if (tensor->get_group() == "")
    {
        if (tensor->is_persistent())
        {
            tensor->set_group("persist");
        }
        else
        {
            tensor->set_group(group);
        }
    }
    else if (tensor->get_group() != group)
    {
        cross_stream.insert(tensor);
    }
}