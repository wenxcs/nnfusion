// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/descriptor/tensor.hpp"
#include "nnfusion/common/languageunit.hpp"

#include <list>

DECLARE_bool(fmem_trace);

namespace nnfusion
{
    class MemoryAllocator
    {
    public:
        enum class block_state
        {
            FREE,
            ALLOCATED
        };

        enum class allocation_scheme
        {
            FIRST_FIT,
            BEST_FIT,
            NO_REUSE
        };

        class node
        {
        public:
            node(size_t size, block_state state);

            bool is_free() const { return m_state == block_state::FREE; }
            size_t m_size;
            block_state m_state;
        };

        MemoryAllocator(size_t alignment = 1,
                        bool disable_reuse = false,
                        DeviceType device_type = CUDA_GPU,
                        size_t device_id = 0);
        // allocate a set of tensors.
        virtual void allocate(std::vector<nnfusion::descriptor::Tensor*>& tensors);
        // allocate one tensor.
        virtual void allocate(nnfusion::descriptor::Tensor* tensor);
        // allocate tensor with specified offset.
        virtual void allocate(nnfusion::descriptor::Tensor* tensor, size_t offset);
        virtual void free(nnfusion::descriptor::Tensor* tensor);

        void dump(std::ofstream&);
        void record(string symbol, nnfusion::descriptor::Tensor* tensor);
        virtual LanguageUnit_p emit_memory_init();
        virtual LanguageUnit_p emit_memory_alloc();
        virtual LanguageUnit_p emit_memory_free();

        static size_t align(size_t x, size_t alignment);

        std::list<node>::iterator begin() { return m_node_list.begin(); }
        std::list<node>::iterator end() { return m_node_list.end(); }
        std::list<node>::const_iterator begin() const { return m_node_list.cbegin(); }
        std::list<node>::const_iterator end() const { return m_node_list.cend(); }
        const std::list<node>& get_node_list() const { return m_node_list; }
        size_t max_allocated() const { return m_max_allocated; }
        size_t cur_allocated();
        size_t memory_in_use();
        void set_alloc_scheme(allocation_scheme alloc_schem) { m_scheme = alloc_schem; }
        allocation_scheme get_alloc_scheme() const { return m_scheme; }
        void set_alignment(size_t alignment) { m_alignment = alignment; }
        size_t get_alignment() const { return m_alignment; }
        virtual std::string get_name();
        size_t get_device_id() const { return m_device_id; }
        DeviceType get_device_type() const { return m_device_type; }
    protected:
        size_t first_fit(size_t size);
        size_t best_fit(size_t size);
        size_t no_reuse_allocator(size_t size);

        std::list<node> m_node_list;
        size_t m_alignment;
        allocation_scheme m_scheme;
        DeviceType m_device_type;
        size_t m_device_id;
        size_t m_max_allocated;
        std::vector<nnfusion::descriptor::Tensor*> m_allocated_tensors;
        std::stringstream m_trace;
        bool record_trace = FLAGS_fmem_trace;
    };

    class CUDAMemoryAllocator : public MemoryAllocator
    {
    public:
        CUDAMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            DeviceType device_type = CUDA_GPU,
                            size_t device_id = 0)
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id)
        {
        }
    };

    class HostMemoryAllocator : public MemoryAllocator
    {
    public:
        HostMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            DeviceType device_type = GENERIC_CPU,
                            size_t device_id = 0)
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id)
        {
        }
        LanguageUnit_p emit_memory_alloc() override;
        LanguageUnit_p emit_memory_free() override;
    };

    class RocmMemoryAllocator : public MemoryAllocator
    {
    public:
        RocmMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            DeviceType device_type = ROCM_GPU,
                            size_t device_id = 0)
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id)
        {
        }
    };

    class RDMAMemoryAllocator : public MemoryAllocator
    {
    public:
        RDMAMemoryAllocator(size_t alignment = 1,
                            bool disable_reuse = false,
                            DeviceType device_type = CUDA_GPU,
                            size_t device_id = 0)
            : MemoryAllocator(alignment, disable_reuse, device_type, device_id)
        {
        }

        std::string get_name() override;
    };

    class MemoryAllocatorFactory
    {
    public:
        MemoryAllocatorFactory(size_t alignment = 1, bool disable_reuse = false);
        MemoryAllocator* get_allocator(nnfusion::descriptor::Tensor* tensor);
        std::string get_device_name(nnfusion::descriptor::Tensor* tensor);
        size_t get_alignment() const { return m_alignment; }
        static const std::unordered_map<std::string, MemoryAllocator*>& get_allocator_list()
        {
            return m_allocator_list;
        }

    private:
        size_t m_alignment;
        bool m_disable_reuse;
        // map from names to allocators
        static std::unordered_map<std::string, MemoryAllocator*> m_allocator_list;
    };
}
