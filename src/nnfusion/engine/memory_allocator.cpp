// Microsoft (c) 2019, NNFUSION TEAM
#include "memory_allocator.hpp"

nnfusion::MemoryAllocator::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

nnfusion::MemoryAllocator::MemoryAllocator(size_t alignment,
                                           bool disable_memory_reuse,
                                           DeviceType device_type,
                                           size_t device_id)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_device_type(device_type)
    , m_device_id(device_id)
    , m_max_allocated{0}
{
    CHECK_WITH_EXCEPTION(m_alignment > 0, errors::InvalidArgument)
        << "Memory alignment must be > 0";
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
    if (record_trace)
    {
        m_trace << this->get_name() << ": \n";
        m_trace << "memory allocation trace: \n";
    }
}

void nnfusion::MemoryAllocator::allocate(std::vector<ngraph::descriptor::Tensor*>& tensors)
{
    size_t rc;
    size_t total_size = 0;

    for (auto tensor : tensors)
    {
        total_size += tensor->size();
    }

    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(total_size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(total_size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(total_size); break;
    }
    for (auto tensor : tensors)
    {
        tensor->set_pool_offset(rc);
        // add tensor allocated by this allocator
        m_allocated_tensors.push_back(tensor);
        rc += tensor->size();
        if (record_trace)
        {
            this->record("[allocate]", tensor);
        }
    }
}

void nnfusion::MemoryAllocator::allocate(ngraph::descriptor::Tensor* tensor)
{
    size_t rc;
    size_t size = tensor->size();
    ;

    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    tensor->set_pool_offset(rc);
    // add tensor allocated by this allocator
    m_allocated_tensors.push_back(tensor);
    if (record_trace)
    {
        this->record("[allocate]", tensor);
    }
}

void nnfusion::MemoryAllocator::allocate(ngraph::descriptor::Tensor* tensor, size_t offset)
{
    tensor->set_pool_offset(offset);
    m_allocated_tensors.push_back(tensor);
    if (record_trace)
    {
        this->record("[allocate]", tensor);
    }
}

size_t nnfusion::MemoryAllocator::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t nnfusion::MemoryAllocator::best_fit(size_t size)
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

size_t nnfusion::MemoryAllocator::first_fit(size_t size)
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

void nnfusion::MemoryAllocator::free(ngraph::descriptor::Tensor* tensor)
{
    size_t offset = tensor->get_pool_offset();
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
    if (record_trace)
    {
        this->record("[free]", tensor);
    }
    CHECK(found) << "bad free";
}

void nnfusion::MemoryAllocator::dump(ofstream& out)
{
    out << m_trace.str();
    out << "max allocated memory:\n" << m_max_allocated << "\n";
    out << "current allocated memory:\n" << this->cur_allocated() << "\n";
    out << "current memory in use: \n" << this->memory_in_use() << "\n";
    out << "memory block state: \n";
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

void nnfusion::MemoryAllocator::record(string symbol, ngraph::descriptor::Tensor* tensor)
{
    m_trace << symbol << " name: " << tensor->get_name()
            << "  offset: " << tensor->get_pool_offset() << "  size: " << tensor->size() << "\n";
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_init()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_init"));
    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        lu << "char* " << this->get_name() << "_memory_pool;\n";

        for (auto tensor : m_allocated_tensors)
        {
            lu << tensor->get_element_type().c_type_string() << "* " << tensor->get_name() << ";\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_alloc()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_alloc"));
    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        lu << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << this->get_name() << "_memory_pool,"
           << m_max_allocated << "));\n";
        lu << "CUDA_SAFE_CALL(cudaMemset((void*)" << this->get_name() << "_memory_pool, 0, "
           << m_max_allocated << "));\n";

        for (auto tensor : m_allocated_tensors)
        {
            lu << tensor->get_name() << " = (" << tensor->get_element_type().c_type_string()
               << "*)(" << this->get_name() << "_memory_pool+" << tensor->get_pool_offset()
               << ");\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::MemoryAllocator::emit_memory_free()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_free"));
    auto& lu = *_lu;
    lu << "CUDA_SAFE_CALL(cudaFree(" << this->get_name() + "_memory_pool));\n";
    return _lu;
}

size_t nnfusion::MemoryAllocator::align(size_t size, size_t alignment)
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

std::string nnfusion::MemoryAllocator::get_name()
{
    std::stringstream m_name;
    m_name << (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[m_device_type] << "_"
           << m_device_id << "_allocator";
    return m_name.str();
}

size_t nnfusion::MemoryAllocator::cur_allocated()
{
    return (prev(m_node_list.end())->m_state == block_state::FREE)
               ? numeric_limits<size_t>::max() - prev(m_node_list.end())->m_size
               : numeric_limits<size_t>::max();
}

size_t nnfusion::MemoryAllocator::memory_in_use()
{
    size_t allocated = 0;
    for (const node& n : m_node_list)
    {
        if (n.m_state == block_state::ALLOCATED)
            allocated += n.m_size;
    }
    return allocated;
}
LanguageUnit_p nnfusion::HostMemoryAllocator::emit_memory_alloc()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_alloc"));
    auto& lu = *_lu;
    if (m_max_allocated > 0)
    {
        lu << this->get_name() << "_memory_pool = new char[" << m_max_allocated << "];\n";
        for (auto tensor : m_allocated_tensors)
        {
            lu << tensor->get_name() << " = (" << tensor->get_element_type().c_type_string()
               << "*)(" << this->get_name() << "_memory_pool+" << tensor->get_pool_offset()
               << ");\n";
        }
    }
    return _lu;
}

LanguageUnit_p nnfusion::HostMemoryAllocator::emit_memory_free()
{
    LanguageUnit_p _lu(new LanguageUnit(this->get_name() + "_free"));
    auto& lu = *_lu;
    lu << "delete[] " << this->get_name() + "_memory_pool;\n";
    return _lu;
}

std::string nnfusion::RDMAMemoryAllocator::get_name()
{
    std::stringstream m_name;
    m_name << "RDMA_" << (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[m_device_type]
           << "_" << m_device_id;

    return m_name.str();
}

nnfusion::MemoryAllocatorFactory::MemoryAllocatorFactory(size_t alignment, bool disable_reuse)
    : m_alignment(alignment)
    , m_disable_reuse(disable_reuse)

{
    CHECK_WITH_EXCEPTION(m_alignment > 0, errors::InvalidArgument)
        << "Memory alignment must be > 0";
}
std::unordered_map<std::string, MemoryAllocator*>
    nnfusion::MemoryAllocatorFactory::MemoryAllocatorFactory::m_allocator_list;

MemoryAllocator* nnfusion::MemoryAllocatorFactory::get_allocator(ngraph::descriptor::Tensor* tensor)
{
    std::string device_name = this->get_device_name(tensor);
    if (m_allocator_list.find(device_name) != m_allocator_list.end())
    {
        return m_allocator_list[device_name];
    }
    else
    {
        if (tensor->is_RDMA_tensor())
        {
            auto t_device_type = tensor->get_device_type();
            DeviceType a_device_type =
                (const DeviceType[]){CUDA_GPU, ROCM_GPU, GENERIC_CPU}[t_device_type];
            RDMAMemoryAllocator* allocator = new RDMAMemoryAllocator(
                m_alignment, m_disable_reuse, a_device_type, tensor->get_device_id());
            m_allocator_list[device_name] = allocator;
            return allocator;
        }
        else
        {
            MemoryAllocator* allocator = nullptr;
            switch (tensor->get_device_type())
            {
            case ngraph::descriptor::Tensor::DeviceType::CUDA_GPU:
            {
                allocator = new CUDAMemoryAllocator(
                    m_alignment, m_disable_reuse, CUDA_GPU, tensor->get_device_id());
                break;
            }
            case ngraph::descriptor::Tensor::DeviceType::ROCM_GPU:
            {
                allocator = new RocmMemoryAllocator(
                    m_alignment, m_disable_reuse, ROCM_GPU, tensor->get_device_id());
                break;
            }
            case ngraph::descriptor::Tensor::DeviceType::GENERIC_CPU:
            {
                allocator = new HostMemoryAllocator(
                    m_alignment, m_disable_reuse, GENERIC_CPU, tensor->get_device_id());
                break;
            }
            default: LOG(ERROR) << "No valid allocator found: " << device_name; break;
            }
            if (allocator != nullptr)
                m_allocator_list[device_name] = allocator;
            return allocator;
        }
    }
}

std::string nnfusion::MemoryAllocatorFactory::get_device_name(ngraph::descriptor::Tensor* tensor)
{
    std::stringstream device_name;

    auto device_type = tensor->get_device_type();
    size_t device_id = tensor->get_device_id();
    if (tensor->is_RDMA_tensor())
    {
        device_name << "RDMA_";
    }
    device_name << (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[device_type] << "_"
                << device_id;
    return device_name.str();
}
