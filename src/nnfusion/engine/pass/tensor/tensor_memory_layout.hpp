// Microsoft (c) 2019, NNFUSION TEAM
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"
#include "nnfusion/engine/op.hpp"

namespace nnfusion
{
    namespace pass
    {
        ///\brief Obsoleted MemoryLayout pass doesn't support tensor allcoated
        /// by KernelEmitter, thus this class is to fix the problem.
        /// Follows this order:
        /// KernelSelected -> AssignTensorMemoryLayout -> Codegen
        class AssignTensorMemoryLayout : public IInterpreterPass
        {
            ///\brief MemoryManager is from Ngraph' Memory Layout pass.
            class MemoryManager
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

                MemoryManager(size_t alignment = 1, bool disable_reuse = false);
                // memory_manager& alignment(size_t a);

                size_t allocate(size_t size);
                void free(size_t offset);

                void dump(std::ostream&);

                static size_t align(size_t x, size_t alignment);

                std::list<node>::iterator begin() { return m_node_list.begin(); }
                std::list<node>::iterator end() { return m_node_list.end(); }
                std::list<node>::const_iterator begin() const { return m_node_list.cbegin(); }
                std::list<node>::const_iterator end() const { return m_node_list.cend(); }
                const std::list<node>& get_node_list() const { return m_node_list; }
                size_t max_allocated() const { return m_max_allocated; }
                void set_no_reuse() { m_scheme = allocation_scheme::NO_REUSE; }
            private:
                size_t first_fit(size_t size);
                size_t best_fit(size_t size);
                size_t no_reuse_allocator(size_t size);

                std::list<node> m_node_list;
                size_t m_alignment;
                allocation_scheme m_scheme;
                size_t m_max_allocated;
            };

        public:
            AssignTensorMemoryLayout(size_t alignment = 64,
                                     bool disable_memory_sharing = false,
                                     DeviceType dt = CUDA_GPU)
                : m_alignment(alignment)
                , m_disable_memory_sharing(disable_memory_sharing)
                , m_device(dt)
            {
            }

            bool run(std::shared_ptr<InterpreterContext> ctx,
                     std::shared_ptr<TranslationUnit> tu) override;

        private:
            size_t m_alignment;
            bool m_disable_memory_sharing;
            DeviceType m_device;
        };
    }
}