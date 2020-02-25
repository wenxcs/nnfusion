// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::ArgMax
 * \author generated by script
 */

#include "../test_util/common.hpp"

namespace nnfusion
{
    namespace test
    {
        template <typename T, size_t N>
        using NDArray = nnfusion::test::NDArray<T, N>;
    }

    namespace inventory
    {
        template <>
        shared_ptr<graph::GNode> create_object<op::ArgMax, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{4, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::ArgMax>(0, element::f32);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 5, 5};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::ArgMax>(3, element::f32);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::ArgMax, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = test::NDArray<float, 4>({{{{0, 1, 0, 2, 1},
                                                             {0, 3, 2, 0, 0},
                                                             {2, 0, 0, 0, 1},
                                                             {2, 0, 1, 1, 2},
                                                             {0, 2, 1, 0, 0}},
                                                            {{0, 0, 0, 2, 0},
                                                             {0, 2, 3, 0, 1},
                                                             {2, 0, 1, 0, 2},
                                                             {3, 1, 0, 0, 0},
                                                             {2, 0, 0, 0, 0}}},
                                                           {{{0, 2, 1, 1, 0},
                                                             {0, 0, 2, 0, 1},
                                                             {0, 0, 1, 2, 3},
                                                             {2, 0, 0, 3, 0},
                                                             {0, 0, 0, 0, 0}},
                                                            {{2, 1, 0, 0, 1},
                                                             {0, 2, 0, 0, 0},
                                                             {1, 1, 2, 0, 2},
                                                             {1, 1, 1, 0, 1},
                                                             {1, 0, 0, 0, 2}}}})
                                      .get_vector();
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::ArgMax, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 3, 0};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = test::NDArray<float, 3>({{{3, 1, 0, 0, 1}, {3, 2, 0, 0, 0}},
                                                                {{1, 2, 4, 3, 0}, {0, 1, 2, 0, 4}}})
                                           .get_vector();
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}