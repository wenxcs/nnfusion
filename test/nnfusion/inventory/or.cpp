// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Or
 * \author generated by script
 */

#include "ngraph/op/or.hpp"
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
        shared_ptr<op::Or> create_object<op::Or, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Or>(A, B);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Or, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, 0, 1, 1, 1, 0, 1, 0};
                vector<float> b = vector<float>{0, 0, 1, 0, 0, 1, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Or, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 0, 1, 1, 1, 1, 1, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}