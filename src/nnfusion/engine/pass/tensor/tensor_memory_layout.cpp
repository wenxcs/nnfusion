// Microsoft (c) 2019, NNFUSION TEAM
#include "tensor_memory_layout.hpp"

#include <exception>
#include <queue>
#include <sstream>
#include <utility>

#include "nnfusion/common/util.hpp"
#include "nnfusion/engine/memory_allocator.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

#include "nnfusion/core/operators/op_define/concat.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"

using namespace std;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

DEFINE_bool(fmem_trace, false, "Record and dump memory trace.");
DEFINE_string(fmem_log_path, "memory.log", "The file path of memory log.");

bool AssignTensorMemoryLayout::run(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    bool dump_trace = FLAGS_fmem_trace;
    string mem_log_path = FLAGS_fmem_log_path;

    // Open memory log file.
    std::ofstream mem_log;
    if (dump_trace)
        mem_log.open(mem_log_path);

    MemoryAllocatorFactory maf(m_alignment, m_disable_memory_sharing);

    // std::unordered_set<shared_ptr<descriptor::Tensor>> persistent_tensors;
    auto& p = tu->program;

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            auto gnode = ins->getGNode();
            // do not allocate parameter tensors.
            if (gnode->get_op_ptr()->is_parameter())
                continue;
            auto emitted_kernel = (*ins)["Kernel_Selection_Result"]
                                      .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            KernelEmitter::Pointer kernel = nullptr;

            if (emitted_kernel.second->get_or_emit_source() == nullptr)
                // Can assign tensor layout even kernel is not emitted.
                NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                               << gnode->get_name();
            kernel = emitted_kernel.second;
            // Tensors should be considered
            // Node: inputs outputs
            // Kernel Context: +tensors

            // <output, <input, offset>>
            std::map<std::shared_ptr<descriptor::Tensor>,
                     std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>
                in_place_outputs;
            std::unordered_set<std::shared_ptr<descriptor::Tensor>> alloc_temp;

            if (kernel != nullptr)
            {
                NNFUSION_CHECK_NOT_NULLPTR(kernel->m_context);
                // Allocate temp tensors
                for (size_t i = 0; i < kernel->m_context->tensors.size(); i++)
                {
                    auto tensor = kernel->m_context->tensors[i];
                    NNFUSION_CHECK(!tensor->is_persistent());
                    alloc_temp.insert(tensor);
                }

                if ((*ins)["InplaceTensorMapping"].is_valid())
                {
                    in_place_outputs =
                        (*ins)["InplaceTensorMapping"]
                            .as<std::map<std::shared_ptr<descriptor::Tensor>,
                                         std::pair<std::shared_ptr<descriptor::Tensor>, size_t>>>();
                }
            }

            unordered_set<std::shared_ptr<descriptor::Tensor>> newlist(alloc_temp);
            // The output of output nodes refers to the input, so there is NO need
            // to allocate memory space for output of output nodes.
            if (!gnode->get_op_ptr()->is_output())
                newlist.insert(gnode->liveness_new_list.begin(), gnode->liveness_new_list.end());

            // Allocate in two passes to make sure ref-tensors is after non-ref-tensors
            std::vector<std::shared_ptr<descriptor::Tensor>> ref_tensors;
            for (std::shared_ptr<descriptor::Tensor> tensor : newlist)
            {
                if (in_place_outputs.count(tensor))
                {
                    ref_tensors.push_back(tensor);
                }
                else
                {
                    auto allocator = maf.get_allocator(tensor);
                    allocator->allocate(tensor);
                }
            }

            for (std::shared_ptr<descriptor::Tensor> tensor : ref_tensors)
            {
                NNFUSION_CHECK(in_place_outputs.count(tensor) > 0);
                auto root_tensor = in_place_outputs.at(tensor).first;
                size_t tensor_offset = in_place_outputs.at(tensor).second;
                auto allocator = maf.get_allocator(root_tensor);
                allocator->allocate(tensor, root_tensor, tensor_offset);
            }

            if (!m_disable_memory_sharing)
            {
                unordered_set<shared_ptr<descriptor::Tensor>> freelist(alloc_temp);
                freelist.insert(gnode->liveness_free_list.begin(), gnode->liveness_free_list.end());
                for (std::shared_ptr<descriptor::Tensor> tensor : freelist)
                {
                    // persistent tensor will not be reused
                    if (!tensor->is_persistent() && !tensor->is_parameter())
                    {
                        auto root_tensor = tensor->get_root_tensor();
                        auto allocator = maf.get_allocator(root_tensor ? root_tensor : tensor);
                        allocator->free(tensor);
                    }
                }
            }
            //dump memory trace at the time scale of node.
            if (dump_trace)
            {
                mem_log << gnode->get_name() << "\n";
                for (const auto& allocator : MemoryAllocatorFactory::get_allocator_list())
                {
                    allocator.second->dump(mem_log);
                }
                mem_log << "\n";
            }
        }
    }

    if (dump_trace)
    {
        // close memory log file.
        mem_log.close();
    }
    return true;
}
