// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag
#pragma once

#include "dependency.hpp"
#include "instruction.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Instruction;
        class Value final
        {
            DISALLOW_COPY_AND_ASSIGN(Value);

        public:
            Value(Instruction* node_, size_t offset_);
            use_list uses_;

        private:
            Node* node_;
            size_t offset_;
            size_t unique_ = 0; // unique id
            size_t stage_ = 0;  // 0-forward, 1-backward, 2-double-backward,...
            bool has_unique_name_;
            std::string unique_name_;
            int32_t elem_type_;
            bool has_sizes_;
            std::vector<Dimension> sizes_;

        public:
            Value* setElemType(int32_t elem_type)
            {
                elem_type_ = elem_type;
                return this;
            }
            int32_t elemType() const { return elem_type_; }
            bool has_sizes() const { return has_sizes_; }
            Value* setSizes(std::vector<Dimension> sizes)
            {
                has_sizes_ = true;
                sizes_ = std::move(sizes);
                return this;
            }
            const std::vector<Dimension>& sizes() const { return sizes_; }
            size_t unique() const { return unique_; }
            bool has_unique_name() const { return has_unique_name_; }
            std::string uniqueName() const
            {
                if (has_unique_name())
                    return unique_name_;
                return std::to_string(unique());
            }
            Value* setUniqueName(std::string name)
            {
                has_unique_name_ = true;
                unique_name_ = std::move(name);
                return this;
            }
            Value* setStage(size_t s)
            {
                stage_ = s;
                return this;
            }
            size_t stage() const { return stage_; }
            Node* node() { return node_; }
            size_t offset() const { return offset_; }
            const Node* node() const { return node_; }
            // Graph* owningGraph();
            // const Graph* owningGraph() const;
            // TODO: make this more const correct
            const use_list& uses() const { return uses_; }
            // Replaces all uses of this node with 'newValue'.
            //
            // Given:   %3 = f(%1, %2)
            //          %4 = g(%3)
            //          %5 = h(%3, %3)
            // Execute: %3.replaceAllUsesWith(%6)
            // Result:  %3 = f(%1, %2)
            //          %4 = g(%6)
            //          %5 = h(%6, %6)
            void replaceAllUsesWith(Value* newValue);

            Value* copyMetadata(Value* from)
            {
                setElemType(from->elemType());
                setSizes(from->sizes());
                if (from->has_unique_name())
                {
                    setUniqueName(from->uniqueName());
                }
                return this;
            }
        };
    }
}