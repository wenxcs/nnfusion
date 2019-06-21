// Microsoft (c) 2019, Wenxiang
#pragma once

#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "inventory.hpp"
#include "library.hpp"

#define EXPECT_POINTER_TYPE(pointer, type, new_pointer)                                            \
    auto new_pointer = static_pointer_cast<type>(pointer);                                         \
    EXPECT_TRUE(new_pointer != nullptr);

using namespace std;
using namespace nnfusion;

template <typename t>
void print_vector(const vector<t>& v, string v_name)
{
    cout << v_name << " = {";
    for (auto& e : v)
        cout << e << ", ";
    cout << "};\n";
}

template <typename t>
void print_set(const set<t>& v, string v_name)
{
    cout << v_name << " = {";
    for (auto& e : v)
        cout << e << ", ";
    cout << "};\n";
}

template <typename t, typename p>
bool compare_vector(const vector<t>& a, const vector<p> b)
{
    if (a.size() != b.size())
        return false;
    for (int i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            return false;
    return true;
}

// from "../../util/all_close_f.hpp"
namespace ngraph
{
    namespace test
    {
        /// \brief Check if the two f32 numbers are close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param mantissa_bits The mantissa width of the underlying number before casting to float
        /// \param tolerance_bits Bit tolerance error
        /// \returns True iff the distance between a and b is within 2 ^ tolerance_bits ULP
        ///
        /// References:
        /// - https://en.wikipedia.org/wiki/Unit_in_the_last_place
        /// - https://randomascii.wordpress.com/2012/01/23/stupid-float-tricks-2
        /// - https://github.com/google/googletest/blob/master/googletest/docs/AdvancedGuide.md#floating-point-comparison
        ///
        /// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
        /// |------------bfloat-----------|
        /// |----------------------------float----------------------------|
        ///
        /// bfloat (s1, e8, m7) has 7 + 1 = 8 bits of mantissa or bit_precision
        /// float (s1, e8, m23) has 23 + 1 = 24 bits of mantissa or bit_precision
        ///
        /// This function uses hard-coded value of 8 bit exponent_bits, so it's only valid for
        /// bfloat and f32.
        bool close_f(float a, float b, int mantissa_bits = 8, int tolerance_bits = 2);

        /// \brief Check if the two floating point vectors are all close
        /// \param a First number to compare
        /// \param b Second number to compare
        /// \param mantissa_bits The mantissa width of the underlying number before casting to float
        /// \param tolerance_bits Bit tolerance error
        /// \returns true iff the two floating point vectors are close
        bool all_close_f(const std::vector<float>& a,
                         const std::vector<float>& b,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);

        /// \brief Check if the two TensorViews are all close in float
        /// \param a First Tensor to compare
        /// \param b Second Tensor to compare
        /// \param mantissa_bits The mantissa width of the underlying number before casting to float
        /// \param tolerance_bits Bit tolerance error
        /// Returns true iff the two TensorViews are all close in float
        bool all_close_f(const std::shared_ptr<runtime::Tensor>& a,
                         const std::shared_ptr<runtime::Tensor>& b,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);

        /// \brief Check if the two vectors of TensorViews are all close in float
        /// \param as First vector of Tensor to compare
        /// \param bs Second vector of Tensor to compare
        /// \param mantissa_bits The mantissa width of the underlying number before casting to float
        /// \param tolerance_bits Bit tolerance error
        /// Returns true iff the two TensorViews are all close in float
        bool all_close_f(const std::vector<std::shared_ptr<runtime::Tensor>>& as,
                         const std::vector<std::shared_ptr<runtime::Tensor>>& bs,
                         int mantissa_bits = 8,
                         int tolerance_bits = 2);
    }
}