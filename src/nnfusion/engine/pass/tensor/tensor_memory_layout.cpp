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
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace std;
using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::pass;
using namespace nnfusion::kernels;

bool AssignTensorMemoryLayout::run(std::shared_ptr<InterpreterContext> ctx,
                                   std::shared_ptr<TranslationUnit> tu)
{
    ///\todo Strong Assumption: Only two devices related.
    MemoryManager mm(m_alignment, m_disable_memory_sharing);
    MemoryManager host_mm(m_alignment, m_disable_memory_sharing);
    auto chosen_mm = [&mm, &host_mm](const descriptor::Tensor* t) -> MemoryManager& {
        return t->is_host_tensor() ? host_mm : mm;
    };
    auto is_same_dev = [](const descriptor::Tensor* a, const descriptor::Tensor* b) {
        return a->is_host_tensor() == b->is_host_tensor();
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
            auto emitter_iter = find_if(emitted_kernels.begin(),
                                        emitted_kernels.end(),
                                        [this](pair<DeviceType, KernelEmitter::Pointer>& i) {
                                            return i.first == this->m_device;
                                        });

            KernelEmitter::Pointer kernel = nullptr;

            if (emitter_iter == emitted_kernels.end() || emitter_iter->second == nullptr)
                // Can assign tensor layout even kernel is not emitted.
                LOG_WARN << "Kernel should be emitted before this pass:" << node->get_name();
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
                enforce_not_nullptr(kernel->m_context);
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
                            if (std::dynamic_pointer_cast<op::Constant>(input_node))
                                continue;

                            if (!is_same_dev(input, output))
                            {
                                LOG_WARN
                                    << "Tensor inplace oi pairs are not in same device, ignored.";
                                continue;
                            }

                            // For destructive kernel, this should be the last use
                            // Non-destructive kernels can pass through if memory sharing is disabled
                            if ((node->liveness_free_list.count(input) != 0 ||
                                 (m_disable_memory_sharing && !oi_pair.destructive)) &&
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
                size_t offset = in_place_outputs.count(tensor)
                                    ? in_place_outputs.at(tensor)->get_pool_offset()
                                    : chosen_mm(tensor).allocate(tensor->size());
                tensor->set_pool_offset(offset);
            }

            if (!m_disable_memory_sharing)
            {
                unordered_set<descriptor::Tensor*> freelist(alloc_temp);
                freelist.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
                for (const descriptor::Tensor* tensor : freelist)
                {
                    if (reused_inputs.count(tensor) == 0)
                    {
                        chosen_mm(tensor).free(tensor->get_pool_offset());
                    }
                }
            }
        }
    }

    // Allocate persistent tensors at the end of memory pool
    mm.set_no_reuse(); // set mm into no-reuse mode to append tensors at the back.
    host_mm.set_no_reuse();

    for (auto& persist : persistent_tensors)
    {
        size_t offset = chosen_mm(persist).allocate(persist->size());
        persist->set_pool_offset(offset);
    }

    tu->program.m_context.memory_pool_size = mm.max_allocated();
    tu->program.m_context.host_memory_pool_size = host_mm.max_allocated();
    return true;
}

nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::MemoryManager(size_t alignment,
                                                                       bool disable_memory_reuse)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_max_allocated{0}
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
}

size_t nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::allocate(size_t size)
{
    size_t rc;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    return rc;
}

size_t nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::best_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    size_t min_delta = numeric_limits<size_t>::max();
    auto best_fit = m_node_list.end();
    size_t best_offset = offset;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            size_t delta = it->m_size - size;
            if (delta < min_delta)
            {
                min_delta = delta;
                best_fit = it;
                best_offset = offset;
            }
        }
        offset += it->m_size;
    }

    if (best_fit == m_node_list.end())
    {
        throw bad_alloc();
    }

    if (min_delta == 0)
    {
        // exact fit
        best_fit->m_state = block_state::ALLOCATED;
    }
    else
    {
        m_node_list.insert(best_fit, node{size, block_state::ALLOCATED});
        best_fit->m_size -= size;
    }
    m_max_allocated = max(m_max_allocated, best_offset + size);

    return best_offset;
}

size_t nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::first_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            if (it->m_size > size)
            {
                m_node_list.insert(it, node{size, block_state::ALLOCATED});
                it->m_size -= size;
            }
            else
            {
                // exact fit
                it->m_state = block_state::ALLOCATED;
            }

            found = true;
            break;
        }
        offset += it->m_size;
    }
    if (!found)
    {
        throw bad_alloc();
    }
    m_max_allocated = max(m_max_allocated, offset + size);

    return offset;
}

void nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::free(size_t offset)
{
    size_t search_offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (offset == search_offset)
        {
            list<node>::iterator it_next = next(it);
            if (it == m_node_list.begin())
            {
                // free the first node in the list
                it->m_state = block_state::FREE;
            }
            else
            {
                // node has predecessor
                list<node>::iterator it_prev = prev(it);
                if (it_prev->m_state == block_state::FREE)
                {
                    it->m_size += it_prev->m_size;
                    m_node_list.erase(it_prev);
                }
            }
            if (it_next != m_node_list.end() && it_next->m_state == block_state::FREE)
            {
                // join this node with next
                it->m_size += it_next->m_size;
                m_node_list.erase(it_next);
            }
            it->m_state = block_state::FREE;
            found = true;
            break;
        }
        search_offset += it->m_size;
    }
    if (!found)
    {
        throw runtime_error("bad free");
    }
}

void nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::dump(ostream& out)
{
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

size_t nnfusion::pass::AssignTensorMemoryLayout::MemoryManager::align(size_t size, size_t alignment)
{
    if (size == 0)
    {
        size = alignment;
    }
    else
    {
        auto remainder = size % alignment;
        if (remainder > 0)
        {
            size += (alignment - remainder);
        }
    }
    return size;
}
