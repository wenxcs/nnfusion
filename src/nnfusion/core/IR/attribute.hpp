// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "dependency.hpp"

namespace nnfusion
{
    namespace ir
    {
        enum class AttributeKind
        {
            // float, float list, int, int list, string, string list,
            // tensor, tensor list, subgraph, subgraph list
            f,
            fs,
            i,
            is,
            s,
            ss,
            t,
            ts,
            g,
            gs
        };

        static inline const char* toString(AttributeKind kind)
        {
            static constexpr const char* names[] = {
                "f", "fs", "i", "is", "s", "ss", "t", "ts", "g", "gs"};
            enforce(size_t(kind) < sizeof(names) / sizeof(AttributeKind));
            return names[int(kind)];
        }

        struct AttributeValue
        {
            AttributeValue(Symbol name)
                : name(name)
            {
            }
            using Ptr = std::unique_ptr<AttributeValue>;
            Symbol name;
            virtual AttributeKind kind() const = 0;
            virtual Ptr clone() const = 0;
            virtual ~AttributeValue() = default;
        };

        template <typename T, AttributeKind Kind>
        struct ScalarAttributeValue final : public AttributeValue
        {
            using ConstructorType = const T&;
            using ValueType = T;
            ScalarAttributeValue(Symbol name, ConstructorType value_)
                : AttributeValue(name)
                , value_(value_)
            {
            }
            ValueType& value() { return value_; }
            virtual Ptr clone() const override
            {
                return Ptr(new ScalarAttributeValue(name, value_));
            }
            virtual AttributeKind kind() const override { return Kind; }
        private:
            ValueType value_;
        };

        template <typename T, AttributeKind Kind>
        struct VectorAttributeValue final : public AttributeValue
        {
            using ConstructorType = const std::vector<T>&&;
            using ValueType = std::vector<T>;
            VectorAttributeValue(Symbol name, ConstructorType value_)
                : AttributeValue(name)
                , value_(std::move(value_))
            {
            }
            ValueType& value() { return value_; }
            virtual AttributeKind kind() const override { return Kind; }
            virtual std::unique_ptr<AttributeValue> clone() const override
            {
                auto copy = value_;
                return Ptr(new VectorAttributeValue(name, std::move(copy)));
            }

        private:
            ValueType value_;
        };

        using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
        using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
        using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
        using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
        using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
        using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
        // This tensor has different meaning with nnfusion
        // using TensorAttr = ScalarAttributeValue<Tensor, AttributeKind::t>;
        // using TensorsAttr = VectorAttributeValue<Tensor, AttributeKind::ts>;

        // using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>, AttributeKind::g>;
        // using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>, AttributeKind::gs>;

        // CRTP so that Node which inherits Attributes can be return for
        // method chaining e.g:
        // Node * n = g->create(kSelect)->set_i(kOffset,3)->set_f(kValue,3.5);
        // we return Derived* pointers because Nodes are normally held as pointers.
        template <typename Derived>
        class Attributes
        {
        public:
            Attributes() {}
            void copyAttributes(const Attributes& rhs)
            {
                values_.clear();
                values_.reserve(rhs.values_.size());
                for (auto& i : rhs.values_)
                {
                    values_.push_back(i->clone());
                }
            }
            bool hasAttribute(Symbol name) const { return find(name, false) != values_.end(); }
            AttributeKind kindOf(Symbol name) const { return (*find(name, true))->kind(); }
            Derived* removeAttribute(Symbol name)
            {
                values_.erase(find(name, true));
                return This();
            }
            bool hasAttributes() const { return values_.size() > 0; }
            // The names are returned in order, since name actually is the index.
            std::vector<Symbol> attributeNames() const
            {
                std::vector<Symbol> names;
                names.reserve(values_.size());
                for (auto& a : values_)
                    names.push_back(a->name);
                return names;
            }
#define CREATE_ACCESSOR(Kind, method)                                                              \
    Derived* method##_(Symbol name, Kind##Attr::ConstructorType v)                                 \
    {                                                                                              \
        return set<Kind##Attr>(name, std::forward<Kind##Attr::ConstructorType>(v));                \
    }                                                                                              \
    const Kind##Attr::ValueType& method(Symbol name) const { return get<Kind##Attr>(name); }
            CREATE_ACCESSOR(Float, f)
            CREATE_ACCESSOR(Floats, fs)
            CREATE_ACCESSOR(String, s)
            CREATE_ACCESSOR(Strings, ss)
            CREATE_ACCESSOR(Int, i)
            CREATE_ACCESSOR(Ints, is)
/*
            CREATE_ACCESSOR(Tensor, t)
            CREATE_ACCESSOR(Tensors, ts)
            CREATE_ACCESSOR(Graph, g)
            CREATE_ACCESSOR(Graphs, gs)
            */

#undef CREATE_ACCESSOR

        private:
            Derived* This() { return static_cast<Derived*>(this); }
            template <typename T>
            Derived* set(Symbol name, typename T::ConstructorType v)
            {
                auto it = find(name, false);
                auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
                if (it == values_.end())
                {
                    values_.push_back(std::move(nv));
                }
                else
                {
                    *it = std::move(nv);
                }
                return This();
            }
            template <typename T>
            typename T::ValueType& get(Symbol name) const
            {
                auto it = find(name, true);
                T* child = static_cast<T*>(it->get());
                return child->value();
            }
            using AVPtr = AttributeValue::Ptr;
            // NB: For determinism, we use a vector rather than a hash map.  This does
            // mean that lookups are O(n), so you shouldn't use Attributes to store
            // a big pile of messages.
            std::vector<AVPtr> values_;
            using iterator = std::vector<AVPtr>::iterator;
            iterator find(Symbol name, bool required)
            {
                auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
                    return v->name == name;
                });
                enforce(!required || it != values_.end());
                return it;
            }
            using const_iterator = std::vector<AVPtr>::const_iterator;
            const_iterator find(Symbol name, bool required) const
            {
                auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
                    return v->name == name;
                });
                enforce(!required || it != values_.end()) << "required undefined attribute:"
                                                          << name;
                return it;
            }
        };
    }
}