// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Subtract
 * \author generated by script
 */

#include "ngraph/op/subtract.hpp"
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
        shared_ptr<op::Subtract> create_object<op::Subtract, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Subtract>(A, B);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Subtract, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{2, 4, 8, 16};
                vector<float> b = vector<float>{1, 2, 4, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Subtract, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 2, 4, 8};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}