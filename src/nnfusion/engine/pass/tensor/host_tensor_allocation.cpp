// Microsoft (c) 2019, NNFUSION TEAM
#include "host_tensor_allocation.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion;
using namespace nnfusion::pass;

bool HostTensorAllocation::run(std::shared_ptr<InterpreterContext> ctx,
                               std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    unordered_map<descriptor::Tensor*, TensorWrapper*> output_wrappers;
    unordered_map<descriptor::Tensor*, shared_ptr<descriptor::Tensor>> tensor_mirror_map;

    for (auto iterator : p)
    {
        for (auto& ins : *iterator)
        {
            // break the link of node
            auto node = ins->operatorDef();

            auto emitted_kernels = (*ins)["Kernel_Selection_Result"]
                                       .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();
            auto emitter_iter = find_if(emitted_kernels.begin(),
                                        emitted_kernels.end(),
                                        [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                            return i.first == this->m_device;
                                        });

            KernelEmitter::Pointer kernel = nullptr;

            if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr)
            {
                LOG(WARNING) << "Kernel should be emitted before this pass:" << node->get_name();
                continue;
            }
            else
                kernel = emitter_iter->second;

            auto kernel_context = kernel->m_context;

            // We get the host flag from tensor wrapper in kernel context
            size_t tensor_pos = 0;
            for (size_t i = 0; i < kernel_context->inputs.size(); i++)
            {
                auto& tensor = kernel_context->inputs[i];
                auto tdesc = &(descriptor::Tensor&)tensor.get_tensor();
                auto outwrapper = output_wrappers.find(tdesc);
                if (outwrapper != output_wrappers.end())
                {
                    // check same device
                    if (tensor.is_host() == outwrapper->second->is_host())
                        continue;
                    auto newts = tensor_mirror_map.find(tdesc);
                    if (newts == tensor_mirror_map.end())
                    {
                        string suffix = tensor.is_host() ? "_host" : "_device";

                        auto ts = make_shared<descriptor::Tensor>(tdesc->get_element_type(),
                                                                  tdesc->get_partial_shape(),
                                                                  tdesc->get_name() + suffix,
                                                                  tdesc->is_host_tensor(),
                                                                  tdesc->is_persistent());
                        ts->set_tensor_layout(tdesc->get_tensor_layout());
                        ts->set_host_tensor(tensor.is_host());
                        tensor_mirror_map[tdesc] = ts;
                        // Update newts;
                        newts = tensor_mirror_map.find(tdesc);
                    }

                    TensorWrapper new_wrapper(newts->second);
                    LOG(INFO) << "Replacing " << node->get_name() << " " << tdesc->get_name()
                              << " with new host tensor " << newts->second->get_name() << ".";
                    kernel_context->inputs[i] = new_wrapper;

                    // memcpy_pair: des <- src
                    if (!(*ins)["memcpy_pair"].is_valid())
                        (*ins)["memcpy_pair"] = unordered_map<TensorWrapper*, TensorWrapper*>();

                    auto key = &(kernel_context->inputs[i]);
                    auto val = outwrapper->second;
                    (*ins)["memcpy_pair"].as<unordered_map<TensorWrapper*, TensorWrapper*>>()[key] =
                        val;
                }
            }

            for (auto tensor : kernel_context->outputs)
            {
                auto tdesc = &(descriptor::Tensor&)tensor.get_tensor();
                output_wrappers[tdesc] = &tensor;
            }
        }
    }

    // Liveness again, we put those mirror node in the same place.
    for (auto iterator : p)
    {
        for (auto& ins : *iterator)
        {
            auto node = ins->operatorDef();
            for (auto& ts : node->liveness_new_list)
            {
                auto mirror = tensor_mirror_map.find(ts);
                if (mirror != tensor_mirror_map.end())
                    node->liveness_new_list.insert(mirror->second.get());
            }

            for (auto& ts : node->liveness_free_list)
            {
                auto mirror = tensor_mirror_map.find(ts);
                if (mirror != tensor_mirror_map.end())
                    node->liveness_free_list.insert(mirror->second.get());
            }
        }
    }
    return true;
}