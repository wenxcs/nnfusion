// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Product
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
        shared_ptr<graph::GNode> create_object<op::Product, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 2:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0, 1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 3:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 4:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 5:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 0};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 6:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{0, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 7:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{0};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 8:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{0, 0};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0, 1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 9:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 10:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{2});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 11:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{0, 1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 12:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 0, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Product>(AxisSet{1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Product, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> a = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> a = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> a = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> a = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> a =
                    vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> a =
                    vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> a =
                    vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 12:
            {
                vector<float> a = vector<float>{};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Product, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result =
                    vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> result = vector<float>{24};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> result = vector<float>{15, 48};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> result = vector<float>{2, 12, 30};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> result = vector<float>{1, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> result = vector<float>{1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> result = vector<float>{1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> result = vector<float>{1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> result = vector<float>{1 * 10 * 19,
                                                     2 * 11 * 20,
                                                     3 * 12 * 21,
                                                     4 * 13 * 22,
                                                     5 * 14 * 23,
                                                     6 * 15 * 24,
                                                     7 * 16 * 25,
                                                     8 * 17 * 26,
                                                     9 * 18 * 27};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> result = vector<float>{1 * 2 * 3,
                                                     4 * 5 * 6,
                                                     7 * 8 * 9,
                                                     10 * 11 * 12,
                                                     13 * 14 * 15,
                                                     16 * 17 * 18,
                                                     19 * 20 * 21,
                                                     22 * 23 * 24,
                                                     25 * 26 * 27};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> result = vector<float>{
                    1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                    2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                    3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 12:
            {
                vector<float> result = vector<float>{1, 1, 1, 1, 1, 1};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}