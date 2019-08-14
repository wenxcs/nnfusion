// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for Concat
 * \author wenxh
 */

#include "ngraph/op/concat.hpp"
#include "../test_util/common.hpp"
#include "ngraph/node_vector.hpp"

using namespace ngraph;

namespace nnfusion
{
    namespace inventory
    {
        // We use float by default
        template <>
        shared_ptr<op::Concat> create_object(int option)
        {
            switch (option)
            {
            case 0:
                Shape shape_a{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                Shape shape_b{2, 3};
                auto B = make_shared<op::Parameter>(element::f32, shape_b);
                Shape shape_c{2, 3};
                auto C = make_shared<op::Parameter>(element::f32, shape_c);
                Shape shape_r{2, 8};
                return make_shared<op::Concat>(NodeVector{A, B, C}, 1);
            }
        }

        template <>
        vector<float> generate_input<op::Concat, float>(int option)
        {
            switch (option)
            {
            case 0:
                return vector<float>{
                    /*a*/ 2,
                    4,
                    8,
                    16,
                    /*b*/ 1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    /*c*/ 2,
                    3,
                    5,
                    7,
                    11,
                    13,
                };
            };
        }

        template <>
        vector<float> generate_output<op::Concat, float>(int option)
        {
            switch (option)
            {
            case 0: return vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13};
            }
        }
    }
}