// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::Pad
 * \author wenxh
 */

#include "ngraph/op/pad.hpp"
#include "../test_util/common.hpp"
#include "ngraph/op/parameter.hpp"

using namespace ngraph;

namespace nnfusion
{
    namespace inventory
    {
        template <>
        shared_ptr<op::Pad> create_object<op::Pad, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape_a{2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_b{};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                Shape shape_r{7, 6};
                Shape padding_below{1, 0};
                Shape padding_above{2, 1};
                Shape padding_interior{2, 1};
                return make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior);
            }
            case 1:
            {
                Shape shape_a{0, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_b{};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                Shape shape_r{5, 5};
                Shape padding_below{2, 1};
                Shape padding_above{3, 1};
                Shape padding_interior{0, 0};
                return make_shared<op::Pad>(A, B, padding_below, padding_above, padding_interior);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Pad, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{/*a*/ 1, 2, 3, 4, 5, 6, /*b*/ 9};
            case 1: return vector<float>{/*a*/ /*b*/ 2112};
            default: return vector<float>();
            };
        }

        template <>
        vector<float> generate_output<op::Pad, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{9, 9, 9, 9, 9, 9, 1, 9, 2, 9, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                                     9, 9, 9, 4, 9, 5, 9, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
            case 1: return vector<float>(25, 2112);
            default: return vector<float>();
            }
        }
    }
}