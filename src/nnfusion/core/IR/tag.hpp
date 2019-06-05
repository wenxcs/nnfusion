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
        struct tag_base
        {
            using Ptr = tag_base*;
            Symbol name;
            std::string scope;
            virtual ~tag_base() {}
        };

        // This is class for hoding temporary data's pointer, and take no charge of
        // lifecycle managament.
        template <class StoreType>
        struct tag_pointer : tag_base
        {
            const StoreType* store;
            tag_pointer(const StoreType* store)
                : store(store)
            {
            }
            static const StoreType* cast(const std::shared_ptr<tag_base>& attr)
            {
                const auto p = std::dynamic_pointer_cast<tag_pointer>(attr);
                if (p == nullptr)
                    return nullptr;
                return p->store;
            }
        };

        // This is the class store the real attribute entity.
        template <class StoreType>
        struct tag_entity : tag_base
        {
            const std::shared_ptr<StoreType> store;
            tag_entity(StoreType&& store)
                : store(std::make_shared<StoreType>(store))
            {
            }
            static const std::shared_ptr<StoreType>& cast(const std::shared_ptr<tag_base>& attr)
            {
                const auto p = std::dynamic_pointer_cast<tag_entity>(attr);
                if (p == nullptr)
                    return nullptr;
                return p->store;
            }
        };

#define Attr_Gen_Entity(type, value) (std::make_shared<tag_entity<type>>(std::move(value)))

#define Attr_Get_Entity(type, value) (tag_entity<type>::cast(value))

#define Attr_Gen_Pointer(type, pvalue) (std::make_shared<tag_pointer<type>>(pvalue))

#define Attr_Get_Pointer(type, pvalue) (tag_pointer<type>::cast(pvalue))

        template <typename Derived>
        class Tag
        {
        public:
            Tag() {}
            void copyTag(const Tag& rhs)
            {
                values_.clear();
                values_.reserve(rhs.values_.size());
                for (auto& i : rhs.values_)
                {
                    values_.push_back(i->clone());
                }
            }
            bool hasTag(Symbol name) const { return find(name, false) != values_.end(); }
            // AttributeKind kindOf(Symbol name) const { return (*find(name, true))->kind(); }
            Derived* removeTag(Symbol name)
            {
                values_.erase(find(name, true));
                return This();
            }
            bool hasTags() const { return values_.size() > 0; }
            // The names are returned in order, since name actually is the index.
            std::vector<Symbol> tagNames() const
            {
                std::vector<Symbol> names;
                names.reserve(values_.size());
                for (auto& a : values_)
                    names.push_back(a->name);
                return names;
            }

        private:
            Derived* This() { return static_cast<Derived*>(this); }
            /*
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
            */

            template <typename T>
            typename T::ValueType& get(Symbol name) const
            {
                auto it = find(name, true);
                T* child = static_cast<T*>(it->get());
                return child->value();
            }
            using AVPtr = tag_base::Ptr;
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
                /*
                enforce(!required || it != values_.end()) << "required undefined tag:"
                                                          << name.toString();
                                                          */
                return it;
            }
        };
    }
}