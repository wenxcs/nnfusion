// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/assertion.hpp"
#include "ngraph/except.hpp"

using namespace ngraph;

namespace nnfusion
{
    namespace error
    {
        struct NotSupported : AssertionFailure
        {
            explicit NotSupported(const std::string& what_arg)
                : AssertionFailure(what_arg)
            {
            }
        };

        struct InvalidArgument : AssertionFailure
        {
            explicit InvalidArgument(const std::string& what_arg)
                : AssertionFailure(what_arg)
            {
            }
        };

        struct NullPointer : AssertionFailure
        {
            explicit NullPointer(const std::string& what_arg)
                : AssertionFailure(what_arg)
            {
            }
        };

        struct RuntimeError : AssertionFailure
        {
            explicit RuntimeError(const std::string& what_arg)
                : AssertionFailure(what_arg)
            {
            }
        };
    }

} // namespace  error

#define enforce_not_nullptr(ptr_)                                                                  \
    NGRAPH_ASSERT_STREAM(nnfusion::error::NullPointer, ((ptr_) != nullptr)) << " "

#define enforce(bval) NGRAPH_ASSERT_STREAM(nnfusion::error::RuntimeError, ((bval) == true)) << " "

#define check(ret)                                                                                 \
    if (!ret)                                                                                      \
        return false;

#define status bool
#define ok true
#define fail false