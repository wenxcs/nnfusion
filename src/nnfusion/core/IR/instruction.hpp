// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"
#include "dependency.hpp"
#include "value.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Instruction: // public Attributes<Node>
        {
            friend class Value;
            friend class Use;
            DISALLOW_COPY_AND_ASSIGN(Instruction);

        private:
            // each node but Return/Param
            // is associated with exactly one place in the node list...
            // of the graph_
            // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the list
            // such that the list never has null pointers
            // next_in_graph[0] is next pointer
            // next_in_graph[1] is prev pointer
            // using an array to allow the same iterator class for forward and reverse node lists
            // This list represents a topological sort

            Instruction* next_in_graph[2] = {nullptr, nullptr};
            Instruction*& next() { return next_in_graph[kNextDirection]; }
            Instruction*& prev() { return next_in_graph[kPrevDirection]; }
            Instruction* const& next() const { return next_in_graph[kNextDirection]; }
            Instruction* const& prev() const { return next_in_graph[kPrevDirection]; }
            const NodeKind kind_;
            std::vector<Value*> inputs_;
            std::vector<Value*> outputs_;
            bool has_name_;
            std::string name_;
            bool has_domain_;
            std::string domain_;
            bool has_doc_string_;
            std::string doc_string_;

        protected:
            Instruction(NodeKind kind_); //defined after graph

        public:
            bool has_name() { return has_name_; }
            const std::string& name() const { return name_; }
            void setName(std::string name)
            {
                has_name_ = true;
                name_ = std::move(name);
            }
            bool has_domain() { return has_domain_; }
            const std::string& domain() const { return domain_; }
            void setDomain(std::string domain)
            {
                has_domain_ = true;
                domain_ = std::move(domain);
            }
            bool has_doc_string() const { return has_doc_string_; }
            const std::string& docString() { return doc_string_; }
            void setDocString(std::string doc_string)
            {
                has_doc_string_ = true;
                doc_string_ = std::move(doc_string);
            }

            // NB: This returns an ArrayRef; that means that it will
            // get invalidated if you resize inputs (e.g., using addInput)
            // We can't return a std::vector<Node*>& because there's no
            // way to soundly cast to std::vector<const Node*> (an insane
            // implementation of std::vector could make this representationally
            // different.)
            /*
            ArrayRef<Value*> inputs() { return inputs_; }
            ArrayRef<const Value*> inputs() const
            {
                // Vectors are not convertible in const-ness of elements, but
                // raw pointers are.
                return {inputs_.data(), inputs_.size()};
            }
            */
            // NB: This returns an ArrayRef; that means that it will
            // get invalidated if you resize inputs (e.g., using addInput)
            // We can't return a std::vector<Node*>& because there's no
            // way to soundly cast to std::vector<const Node*> (an insane
            // implementation of std::vector could make this representationally
            // different.)
            /*
            ArrayRef<Value*> outputs() { return outputs_; }
            ArrayRef<const Value*> outputs() const
            {
                // Vectors are not convertible in const-ness of elements, but
                // raw pointers are.
                return {outputs_.data(), outputs_.size()};
            }
            */
            bool hasUses() const
            {
                for (auto o : outputs_)
                {
                    if (o->uses().size() > 0)
                        return true;
                }
                return false;
            }
            void replaceAllUsesWith(Instruction* n)
            {
                enforce(outputs_.size() == n->outputs_.size());
                size_t nOutputs = outputs_.size();
                for (size_t i = 0; i < nOutputs; i++)
                {
                    outputs_[i]->replaceAllUsesWith(n->outputs_[i]);
                }
            }
            // lots of things like chunk have a single input or single output, so we have a
            // helper to make accessing it easier
            Value* input()
            {
                enforce(inputs_.size() == 1);
                return inputs_.at(0);
            }
            Value* output()
            {
                enforce(outputs_.size() == 1);
                return outputs_.at(0);
            }
            const Value* input() const
            {
                enforce(inputs_.size() == 1);
                return inputs_.at(0);
            }
            // Access a particular input.  This is a checked index.
            Value* input(size_t i) { return inputs_.at(i); }
            const Value* input(size_t i) const { return inputs_.at(i); }
            // Graphs

            // Note [Topological invariant]
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            // We always maintain an up-to-date topological ordering of all nodes via
            // the next()/prev() links.  All transformations to graphs must preserve
            // this topological ordering: for example, it is only valid to 'addInput'
            // with an input which is topologically before the current node.
            //
            // Usually, it is obvious whether or not topological order is maintained;
            // for example, if you are adding nodes to the end of the topsort, it's
            // impossible for them to refer to inputs that are not in the topsort.
            // If it is not obvious, please comment accordingly.

            // Add 'node' as an input to 'this' at the end of existing
            // arguments.  Returns the added node for ease of chaining.
            //
            // Given:   %3 = f(%1, %2)
            // Execute: %3.addInput(%4)
            // Result:  %3 = f(%1, %2, %4)
            Value* addInput(Value* node)
            {
                /*
                enforce(graph_ == node->owningGraph());
                node->uses_.emplace_back(this, inputs_.size());
                inputs_.push_back(node);
                return node;
                */
                enforce(false) << "Not implemented.";
            }

            // Replace the input of 'this' at position 'i' with
            // 'newValue', returning the old node.
            //
            // Given:   %3 = f(%1, %2)
            // Execute: %3.replaceInput(1, %4)
            // Result:  %3 = f(%1, %4)
            Value* replaceInput(size_t i, Value* newValue)
            {
                /*
                ONNX_ASSERT(newValue->owningGraph() == graph_);
                Value* old = dropInput(i);
                inputs_[i] = newValue;
                newValue->uses_.emplace_back(this, i);
                return old;
                */
                enforce(false) << "Not implemented.";
            }

            // Replace all occurrences of 'from' in the inputs of this
            // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
            //
            // Given:   %3 = f(%1, %2, %1)
            // Execute: %3.replaceInputWith(%1, %4)
            // Result:  %3 = f(%4, %2, %4)
            void replaceInputWith(Value* from, Value* to)
            {
                /*
                ONNX_ASSERT(from->owningGraph() == graph_);
                ONNX_ASSERT(to->owningGraph() == graph_);
                size_t i = 0;
                for (auto input : inputs())
                {
                    if (input == from)
                        replaceInput(i, to);
                    i++;
                }
                */
                enforce(false) << "Not implemented.";
            }

            Value* addOutput()
            {
                outputs_.push_back(new Value(this, outputs_.size()));
                return outputs_.back();
            }

            void eraseOutput(size_t i);

            // Insert unattached 'this' node after 'n' in the topological order.
            // Returns this (for chaining).
            //
            // Given:   %3 = f(%1, %2)
            //          %4 = g(%3)
            // and unattached: %5 = h(%1)
            // Execute: %5.insertBefore(%4)
            // Result:  %3 = f(%1, %2)
            //          %5 = h(%1)
            //          %4 = g(%3)
            Node* insertBefore(Node* n)
            {
                /*
                ONNX_ASSERT(n->inGraphList());
                insertAfter(n->prev());
                return this;
                */
                enforce(false) << "Not implemented.";
            }

            // Insert unattached 'this' node after 'n' in the topological order.
            // Returns this (for chaining).
            //
            // Given: %3 = f(%1, %2)
            //        %4 = g(%3)
            // and unattached: %5 = h(%1)
            // Execute: %5.insertAfter(%4)
            // Result:  %3 = f(%1, %2)
            //          %4 = g(%3)
            //          %5 = h(%1)
            Node* insertAfter(Node* n)
            {
                /*
                ONNX_ASSERT(!inGraphList() && n->inGraphList());
                Node* next = n->next();
                n->next() = this;
                this->prev() = n;
                this->next() = next;
                next->prev() = this;
                return this;
                */
                enforce(false) << "Not implemented.";
            }

            // Move 'this' (already in the graph) after 'n' in the topological order.
            //
            // Given: %2 = f(%1)
            //        %3 = g(%1)
            // Execute: %2.moveAfter(%3)
            // Result: %3 = g(%1)
            //         %2 = f(%1)
            //
            void moveAfter(Node* n)
            {
                removeFromList();
                insertAfter(n);
            }

            // Move a node 'n' (already in the graph) before 'this' in the topological order.
            //
            // Given: %2 = f(%1)
            //        %3 = g(%1)
            // Execute: %3.moveBefore(%2)
            // Result: %3 = g(%1)
            //         %2 = f(%1)
            void moveBefore(Node* n)
            {
                removeFromList();
                insertBefore(n);
            }

            // Remove the input at 'i' from this node.
            //
            // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
            // removeInput.
            //
            // Given: %3 = f(%1, %2)
            // Execute: %3.removeInput(1)
            // Result: %3 = f(%1)
            void removeInput(size_t i)
            {
                dropInput(i);
                // everything after this input shifts left,
                // so we need to update their use offsets to match
                for (size_t j = i + 1; j < inputs_.size(); j++)
                {
                    auto it = findUseForInput(j);
                    it->offset--;
                }
                inputs_.erase(inputs_.begin() + i);
            }

            // Remove all inputs from a node.
            //
            // Given: %3 = f(%1, %2)
            // Execute: %3.removeAllInputs()
            // Result: %3 = f()
            void removeAllInputs()
            {
                for (size_t i = 0; i < inputs_.size(); ++i)
                    dropInput(i);
                inputs_.clear();
            }

            // Check whether this node is before node n in the graph.
            bool isBefore(Node* n);

            // iterators of the node list starting at this node
            // useful for resuming a search starting at this node
            /*
            graph_node_list_iterator iterator();
            graph_node_list_iterator reverseIterator();
            const_graph_node_list_iterator iterator() const;
            const_graph_node_list_iterator reverseIterator() const;
            */

            // Remove 'this' from the instruction list and deallocate it.
            //
            // Invariant: no outputs of 'this' may have any uses.
            //
            // Given: %2 = f(%1)
            //        %3 = g(%1)
            // Execute: %2.destroy()
            // Result: %3 = g(%1)
            void destroy();

            // Dynamically cast this node to the subclass indicated by the
            // template variable, returning nullptr if the cast is invalid..
            //
            // Example usage: if(auto s = n.cast<Select>()) { ... }
            //
            // TODO: Make this const correct
            template <typename T>
            T* cast()
            {
                if (T::Kind == kind())
                    return static_cast<T*>(this);
                return nullptr;
            }
            template <typename T>
            T* expect()
            {
                enforce(T::Kind == kind(),
                        "expected a %s but found a %s",
                        T::Kind.toString(),
                        kind().toString());
                return static_cast<T*>(this);
            }

            virtual ~Instruction() = default;

        private:
            // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
            use_list::iterator findUseForInput(size_t i)
            {
                auto& input_uses = inputs_[i]->uses_;
                // O(N) on the use list, but unless we get nodes with +100 uses
                // vector traversal still is probably faster than linked list
                auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
                enforce(use_it != input_uses.end());
                return use_it;
            }

            // remove the use of input i, this sets input i to nullptr, but
            // is only used internally to Node before setting it to a new value
            // or erasing the entry from the list.
            Value* dropInput(size_t i)
            {
                enforce(i < inputs_.size());
                auto input_node = inputs_[i];
                auto use_it = findUseForInput(i);
                input_node->uses_.erase(use_it);
                inputs_[i] = nullptr;
                return input_node;
            }

            bool inGraphList() const
            {
                enforce(next() != nullptr || prev() == nullptr);
                return next() != nullptr;
            }
            void removeFromList()
            {
                enforce(inGraphList());
                Node* next = this->next();
                Node* prev = this->prev();
                prev->next() = next;
                next->prev() = prev;
                this->next() = nullptr;
                this->prev() = nullptr;
            }

        protected:
            // subclasses must override
            // this function is used by createClone to initialize a new version
            // of a node in another graph. It should allocate a new instance of the same
            // concrete type as 'this', but in graph 'g' which might be different
            // than graph_
            // virtual Node* allocNewInstance(Graph* g) { return new Node(kind()); }

            // create a copy of all properties of Node s into this.
            // subclasses should extend if they have additional information to copy.
            // 'this' will be allocated with s->allocNewInstance(g) so it should have
            // the same concrete type as 's'
            //
            // NB: This does NOT clone stages.  You're expected to set the stage correctly
            // if you are going to preserve it.
            // virtual void cloneFrom(Node* s) { copyAttributes(*s); }
        };

        // A class with the same properties as OperatorSetIdProto, but without protobuf
        // overhead, resulting in a simpler and more readable workflow.
        /*
        class OpSetID final
        {
        private:
            std::string domain_;
            int64_t version_;

        public:
            explicit OpSetID(const OperatorSetIdProto& proto)
                : domain_(proto.domain())
                , version_(proto.version())
            {
            }

            // Default Domain Constructor
            explicit OpSetID(const int64_t version)
                : domain_("")
                , version_(version)
            {
            }

            explicit OpSetID(const std::string& domain, int64_t version)
                : domain_(domain)
                , version_(version)
            {
            }

            // target must be in the form "<domain>&<version>"
            std::string toString() const { return domain_ + "$" + std::to_string(version_); }
            // target must be in the form "<domain>&<version>"
            static OpSetID fromString(const std::string& target)
            {
                try
                {
                    std::string new_domain = target.substr(0, target.find("$"));
                    int new_version =
                        std::stoi(target.substr(target.find("$") + 1, target.length()).c_str());
                    return OpSetID(std::move(new_domain), new_version);
                }
                catch (const std::runtime_error& e)
                {
                    enforce(false) << "Error in fromString: " << e.what();
                }
            }

            const std::string& domain() const { return domain_; }
            int64_t version() const { return version_; }
            void incrementVersion(int64_t step) { version_ += step; }
            void setVersion(int64_t newVal) { version_ = newVal; }
        };
        */
    }
}