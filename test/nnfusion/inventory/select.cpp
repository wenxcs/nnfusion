// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Select
 * \author generated by script
 */

#include "ngraph/op/select.hpp"
#include "../test_util/common.hpp"
#include "ngraph/op/parameter.hpp"
#include "util/ndarray.hpp"

using namespace ngraph;

namespace nnfusion
{
    namespace test
    {
        template <typename T, size_t N>
        using NDArray = ngraph::test::NDArray<T, N>;
    }

    namespace inventory
    {
        template <>
        shared_ptr<op::Select> create_object<op::Select, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                auto C = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Select>(A, B, C);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Select, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{0, 1, 1, 0, 0, 1, 0, 1};
                vector<float> b = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
                vector<float> c = vector<float>{11, 12, 13, 14, 15, 16, 17, 18};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return_vector.insert(return_vector.end(), c.begin(), c.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Select, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{11, 2, 3, 14, 15, 6, 17, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}