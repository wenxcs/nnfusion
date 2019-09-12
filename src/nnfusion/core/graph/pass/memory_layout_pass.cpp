// Microsoft (c) 2019, NNFusion Team

#pragma once

#include <limits>
#include <list>

#include "memory_layout_pass.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "nnfusion/util/error.hpp"

using namespace std;
using namespace nnfusion::graph;
using namespace nnfusion::graph::pass;

MemoryLayoutPass::MemoryLayoutPass(size_t alignment, bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
{
}

bool MemoryLayoutPass::run_on_graph(shared_ptr<Graph>& graph)
{
    MemoryManager mm(m_alignment, m_disable_memory_sharing);
    for (auto gnode : graph->get_ordered_ops())
    {
        auto node = gnode->get_op_ptr();
        map<ngraph::descriptor::Tensor*, ngraph::descriptor::Tensor*> in_place_outputs;
        set<const ngraph::descriptor::Tensor*> reused_inputs;

        if (auto op = dynamic_pointer_cast<ngraph::op::Op>(node))
        {
            // concat in_place_oi should be treated differently
            if (!dynamic_pointer_cast<ngraph::op::Concat>(node))
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
                        if (dynamic_pointer_cast<ngraph::op::Constant>(input_node))
                            continue;

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

        for (ngraph::descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t offset = in_place_outputs.count(tensor)
                                ? in_place_outputs.at(tensor)->get_pool_offset()
                                : mm.allocate(tensor->size());
            tensor->set_pool_offset(offset);
        }

        if (!m_disable_memory_sharing)
        {
            for (const ngraph::descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (reused_inputs.count(tensor) == 0)
                {
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
    }
    graph->set_temporary_pool_size(mm.max_allocated());

    return true;
}

MemoryManager::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

MemoryManager::MemoryManager(size_t alignment, bool disable_memory_reuse)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_max_allocated{0}
{
    if (m_alignment == 0)
    {
        // TODO: how to handle error
        throw invalid_argument("Memory alignment must be > 0");
    }
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
}

size_t MemoryManager::allocate(size_t size)
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

size_t MemoryManager::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t MemoryManager::best_fit(size_t size)
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

size_t MemoryManager::first_fit(size_t size)
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

void MemoryManager::free(size_t offset)
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

void MemoryManager::dump(ostream& out)
{
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

size_t MemoryManager::align(size_t size, size_t alignment)
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