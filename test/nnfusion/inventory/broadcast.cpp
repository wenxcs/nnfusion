// Microsoft (c) 2019
/**
 * \brief Unit tests for ir::Broadcast
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
        shared_ptr<graph::GNode> create_object<op::Broadcast, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{4};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 1:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 2};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0, 1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 2:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 2, 2};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0, 1, 2});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 3:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape{2, 2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                auto r = make_shared<op::Broadcast>(shape, AxisSet{});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 4:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{3};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 4};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 5:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{4};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 4};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 6:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{1};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{3, 1};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 7:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 2, 2};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{0});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 8:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 2, 2};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{1});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            case 9:
            {
                auto graph = std::make_shared<graph::Graph>();
                Shape shape_a{2, 2};
                auto A = make_shared<op::Parameter>(element::f32, shape_a);
                auto A_gnode = graph->add_node_and_edge(A, GNodeVector({}));
                Shape shape_r{2, 2, 2};
                auto r = make_shared<op::Broadcast>(shape_r, AxisSet{2});
                auto r_gnode = graph->add_node_and_edge(r, {A_gnode});
                return r_gnode;
            }
            default: return nullptr;
            }
        }

        template <>
        vector<float> generate_input<op::Broadcast, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> a = vector<float>{6};
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
                vector<float> a = vector<float>{2, 4, 6, 8, 16, 32, 64, 128};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> a = vector<float>{1, 2, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> a = vector<float>{4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> a = vector<float>{1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), a.begin(), a.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }

        template <>
        vector<float> generate_output<op::Broadcast, float>(int option)
        {
            switch (option)
            {
            case 0:
            {
                vector<float> result = vector<float>{6, 6, 6, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 1:
            {
                vector<float> result = vector<float>{6, 6, 6, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 2:
            {
                vector<float> result = vector<float>{6, 6, 6, 6, 6, 6, 6, 6};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 3:
            {
                vector<float> result = vector<float>{2, 4, 6, 8, 16, 32, 64, 128};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 4:
            {
                vector<float> result = vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 5:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 6:
            {
                vector<float> result = vector<float>{4, 4, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 7:
            {
                vector<float> result = vector<float>{1, 2, 3, 4, 1, 2, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 8:
            {
                vector<float> result = vector<float>{1, 2, 1, 2, 3, 4, 3, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            case 9:
            {
                vector<float> result = vector<float>{1, 1, 2, 2, 3, 3, 4, 4};
                auto return_vector = vector<float>();
                return_vector.insert(return_vector.end(), result.begin(), result.end());
                return return_vector;
            }
            default: return vector<float>();
            }
        }
    }
}