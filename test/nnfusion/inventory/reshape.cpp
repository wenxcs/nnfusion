// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Reshape
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
        shared_ptr<graph::GNode> create_object<op::Reshape, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{12};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1, 2}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{1, 1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1, 2}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 2:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{1, 1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{};
                auto r = make_shared<op::Reshape>(AxisVector{1, 2, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 3:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 1, 1, 1, 1, 1};
                auto r = make_shared<op::Reshape>(AxisVector{}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 4:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1};
                auto r = make_shared<op::Reshape>(AxisVector{}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 5:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 1};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 6:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 3};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 7:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 3, 1};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 8:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 3};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 9:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 3};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 10:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 3};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 11:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 6};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{12};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 12:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{12};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1, 2}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 13:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{1, 1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1, 2}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 14:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{1, 1, 1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{};
                auto r = make_shared<op::Reshape>(AxisVector{1, 2, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 15:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 1, 1, 1, 1, 1};
                auto r = make_shared<op::Reshape>(AxisVector{}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 16:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1};
                auto r = make_shared<op::Reshape>(AxisVector{}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 17:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 1};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 18:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 3};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 19:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{1, 3, 1};
                auto r = make_shared<op::Reshape>(AxisVector{0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 20:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 3};
                auto r = make_shared<op::Reshape>(AxisVector{0, 1}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 21:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 3};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 22:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 3};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 23:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 6};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{12};
                auto r = make_shared<op::Reshape>(AxisVector{1, 0}, shape_r);
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Reshape, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> a = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> a = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> a = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> a = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 12:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 13:
            {
                vector<float> a = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 14:
            {
                vector<float> a = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 15:
            {
                vector<float> a = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 16:
            {
                vector<float> a = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 17:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 18:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 19:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 20:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 21:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 22:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 23:
            {
                vector<float> a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Reshape, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> result = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> result = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> result = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> result = vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 10:
            {
                vector<float> result = vector<float>{1, 3, 5, 2, 4, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 11:
            {
                vector<float> result = vector<float>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 12:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 13:
            {
                vector<float> result = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 14:
            {
                vector<float> result = vector<float>{6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 15:
            {
                vector<float> result = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 16:
            {
                vector<float> result = vector<float>{42};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 17:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 18:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 19:
            {
                vector<float> result = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 20:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 21:
            {
                vector<float> result = vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 22:
            {
                vector<float> result = vector<float>{1, 3, 5, 2, 4, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 23:
            {
                vector<float> result = vector<float>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}