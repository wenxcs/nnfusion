// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"
#include "dependency.hpp"
#include "instruction.hpp"
#include "value.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Module final
        {
            DISALLOW_COPY_AND_ASSIGN(Module);
            friend class Instruction;
            friend class Value;

        private:
            // only used to keep track of allocated nodes
            // actual representation of Graph is done with
            // inputs, outputs, nodes

            std::unordered_set<const Instruction*> all_nodes;
            std::unordered_set<const Value*> all_values;
            size_t next_unique_;

            size_t new_node_stage_;

            // holds outputs in a way that can be reflected
            // as a Use object
            // also used as the beginning/end of the circular node list to avoid
            // having corner cases where the list is empty.
            Instruction* const output_;
            Instruction* const input_;

            std::vector<Tensor> initializers_;
            std::vector<std::string> initializer_names_;

            bool has_name_;
            std::string name_;
            bool has_doc_string_;
            std::string doc_string_;

        public:
            Module()
                : next_unique_(0)
                , new_node_stage_(0)
                // , output_(initOutput(create(kReturn, 0)))
                // , input_(create(kParam, 0))
                , has_name_(false)
                , has_doc_string_(false)
            {
            }

            bool has_doc_string() { return has_doc_string_; }
            const std::string& docString() { return doc_string_; }
            void setDocString(std::string doc_string)
            {
                has_doc_string_ = true;
                doc_string_ = std::move(doc_string);
            }

            void addInitializer(Tensor initializer, std::string name)
            {
                initializers_.push_back(std::move(initializer));
                initializer_names_.push_back(std::move(name));
            }

            /*
            void eraseInitializer(std::string name)
            {
                initializers_.erase(std::remove_if(initializers_.begin(),
                                                   initializers_.end(),
                                                   [&name](Tensor& initializer) {
                                                       return initializer.name() == name;
                                                   }),
                                    initializers_.end());
                initializer_names_.erase(
                    std::remove(initializer_names_.begin(), initializer_names_.end(), name),
                    initializer_names_.end());
            }
            */

            void clearInitializers()
            {
                initializers_.clear();
                initializer_names_.clear();
            }
            const std::vector<Tensor>& initializers() { return initializers_; }
            const std::vector<std::string>& initializer_names() { return initializer_names_; }
            std::vector<Tensor>::const_iterator getInitializer(const std::string& name)
            {
                for (auto it = initializers_.cbegin(); it != initializers_.cend(); ++it)
                {
                    if (name == it->name())
                    {
                        return it;
                    }
                }
                return initializers_.end();
            }

            /*
            ArrayRef<Value*> inputs() { return input_->outputs(); }
            ArrayRef<const Value*> inputs() const
            {
                const auto& inputs = input_->outputs();
                return {inputs.data(), inputs.size()};
            }
            ArrayRef<Value*> outputs() { return output_->inputs(); }
            ArrayRef<const Value*> outputs() const
            {
                return static_cast<const Node*>(output_)->inputs();
            }
            graph_node_list nodes() { return graph_node_list(output_, kNextDirection); }
            const_graph_node_list nodes() const
            {
                return const_graph_node_list(output_, kNextDirection);
            }

            // These invocations of begin() on output of function are OK
            // because graph_node_list is non-owning, so it doesn't matter
            // if it immediately dies after the invocation.
            graph_node_list_iterator begin() { return nodes().begin(); }
            const_graph_node_list_iterator begin() const { return nodes().begin(); }
            graph_node_list_iterator end() { return nodes().end(); }
            const_graph_node_list_iterator end() const { return nodes().end(); }
            graph_node_list_iterator rbegin() { return nodes().rbegin(); }
            const_graph_node_list_iterator rbegin() const { return nodes().rbegin(); }
            graph_node_list_iterator rend() { return nodes().rend(); }
            const_graph_node_list_iterator rend() const { return nodes().rend(); }
            */

            Instruction* return_node() { return output_; }
            const Instruction* return_node() const { return output_; }
            Value* addInput() { return input_->addOutput(); }
            void eraseInput(size_t i) { input_->eraseOutput(i); }
            void advanceStage() { new_node_stage_++; }
            void setStage(size_t new_stage) { new_node_stage_ = new_stage; }
            size_t stage() const { return new_node_stage_; }
            ResourceGuard setStageTemporary(size_t s)
            {
                auto prev_stage = new_node_stage_;
                new_node_stage_ = s;
                return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
            }

            size_t registerOutput(Value* n)
            {
                output_->addInput(n);
                return outputs.size() - 1;
            }

            Instruction* create(NodeKind kind, size_t num_outputs = 1)
            {
                // NB: Node constructor adds node to all_nodes
                auto n = new Node(this, kind);
                for (size_t i = 0; i < num_outputs; i++)
                    n->addOutput();
                return n;
            }

            Instruction* create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs = 1)
            {
                auto n = create(kind, num_outputs);
                for (auto i : inputs)
                    n->addInput(i);
                return n;
            }

            Instruction* appendNode(Instruction* n)
            {
                ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
                n->insertBefore(output_);
                return n;
            }

            Instruction* prependNode(Instruction* n)
            {
                ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
                n->insertAfter(output_);
                return n;
            }

            //Adds to graph initializer list, initializer names list, and as a graph input
            //Also syncs the initializer name, tensor name, and value name
            Value* addInitializerAndInput(const Tensor& initializer, std::string name)
            {
                Tensor initializerCopy = initializer;
                std::vector<Dimension> dim_sizes{initializerCopy.sizes().cbegin(),
                                                 initializerCopy.sizes().cend()};
                Value* new_init = addInput();
                initializerCopy.setName(name);
                new_init->setUniqueName(name);
                new_init->setSizes(dim_sizes);
                new_init->setElemType(initializerCopy.elem_type());
                addInitializer(std::move(initializerCopy), name);
                return new_init;
            }

            Value* addInitializerAndInput(const Tensor& initializer)
            {
                return addInitializerAndInput(initializer, std::to_string(next_unique_++));
            }

            //Erases from graph initializer list, initializer names list, and as a graph input
            //Must have no uses
            void eraseInitializerAndInput(Value* v)
            {
                eraseInitializer(v->uniqueName());
                eraseInput(v->offset());
            }

            ~Graph()
            {
                for (const Instruction* n : all_nodes)
                    delete n;
                for (const Value* v : all_values)
                    delete v;
            }

            std::string toString() const
            {
                std::ostringstream oss;
                oss << *this;
                return oss.str();
            }

            bool has_name() const { return has_name_; }
            const std::string& name() const { return name_; }
            void setName(std::string name)
            {
                has_name_ = true;
                name_ = name;
            }

            friend std::ostream& operator<<(std::ostream& out, const Graph& g);

        private:
            // should only be called in the constructor
            Instruction* initOutput(Instruction* p)
            {
                p->next() = p;
                p->prev() = p;
                return p;
            }

            void freeNode(Instruction* n)
            {
                auto it = all_nodes.find(n);
                enforce(it != all_nodes.end());
                delete *it;
                all_nodes.erase(it);
            }
            void freeValue(Value* v)
            {
                auto it = all_values.find(v);
                enforce(it != all_values.end());
                all_values.erase(it);
            }
        };
    }
}