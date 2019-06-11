// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Tags : public Attributes
        {
        public:
            template <typename T>
            Tags* Set(Symbol name, T&& v)
            {
                set<ScalarAttributeValue<T>>(name, v);
                return this;
            }

            template <typename T>
            T& Get(Symbol name) const
            {
                return get<ScalarAttributeValue<T>>(name);
            }
        };
    }
}