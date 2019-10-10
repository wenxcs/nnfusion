// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Add
 * \author generated by script
 */

#include "ngraph/op/add.hpp"
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
        shared_ptr<op::Add> create_object<op::Add, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto B = make_shared<op::Parameter>(element::f32, shape);
                return make_shared<op::Add>(A, B);
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Add, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector();
                vector<float> b = test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector();
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return_vector.insert(return_vector.end(), b.begin(), b.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Add, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector();
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}