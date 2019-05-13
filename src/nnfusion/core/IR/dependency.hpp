// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "nnfusion/util/error.hpp"

//typesdef
#define int64_t int
#define uint8_t char
#define size_t int
#define Symbol int
#define Tensor int
#define Graph int
#define Node int

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                                         \
    TypeName(const TypeName&) = delete;                                                            \
    void operator=(const TypeName&) = delete

namespace nnfusion
{
    namespace ir
    {
        class ResourceGuard final
        {
            std::function<void()> destructor_;
            bool released_;

        public:
            ResourceGuard(std::function<void()> destructor)
                : destructor_(std::move(destructor))
                , released_(false)
            {
            }

            ~ResourceGuard()
            {
                if (!released_)
                    destructor_();
            }

            void release() { released_ = true; }
        };

        class Dimension final
        {
        public:
            Dimension(std::string param)
                : is_int(false)
                , dim(-1)
                , param(std::move(param))
            {
            }
            Dimension(int64_t dim)
                : is_int(true)
                , dim(dim)
            {
            }

            bool is_int;
            int64_t dim;
            std::string param;
        };

        // Each use is represented by this type, see Node::uses()
        // 'user' is the consumer of the value, offset is the index into
        // 'user's input this where the produces will be found.
        class Use final
        {
        public:
            Use(Node* user, size_t offset)
                : user(user)
                , offset(offset)
            {
            }
            Node* user;
            size_t offset;
        };

        static inline bool operator==(const Use& a, const Use& b)
        {
            return a.user == b.user && a.offset == b.offset;
        }

        using node_list = std::vector<Node*>;
        using value_list = std::vector<Value*>;
        using use_list = std::vector<Use>;
        using NodeKind = Symbol;
    }
}