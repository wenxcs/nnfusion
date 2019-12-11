// Microsoft (c) 2019, NNFusion Team

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/engine/pass/graph/graph_pass.hpp"
#include "nnfusion/engine/pass/graph/manager.hpp"
#include "nnfusion/engine/pass/graph/op_inplace_pass.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::pass::graph;
using namespace std;
using namespace nnfusion::profiler;

bool run(std::shared_ptr<nnfusion::graph::Graph> graph)
{
    GraphPassManager pass_manager;

    pass_manager.register_pass<OpInplacePass>();
    pass_manager.run_passes(graph);

    return true;
}

bool check_inplace_oi_pair(shared_ptr<nnfusion::op::Op> node)
{
    if (auto op = dynamic_pointer_cast<nnfusion::op::Op>(node))
    {
        auto annotation = op->get_op_annotations();
        if (annotation && annotation->get_in_place_oi_pairs().size() > 0)
        {
            return true;
        }
    }
    return false;
}

TEST(nnfusion_inplace_op, reshape)
{
    // Create graph
    std::string name = "Reshape";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    ngraph::AxisVector input_order{0, 1};
    Shape output_shape{3, 2};

    // Create node
    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(input_order, output_shape);
    auto reshape_gnode = graph->add_node_and_edge(reshape_op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(reshape_op));
}

TEST(nnfusion_inplace_op, result)
{
    // Create graph
    std::string name = "Result";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Result>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sum)
{
    // Create graph
    std::string name = "Sum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});

    Shape shape_b{2, 3, 1};
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_b);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    ngraph::AxisSet reduce_axesA;
    ngraph::AxisSet reduce_axesB{2};

    // Create node
    auto sum_a_op = std::make_shared<nnfusion::op::Sum>(reduce_axesA);
    auto sum_a_gnode = graph->add_node_and_edge(sum_a_op, {para_a_gnode});
    auto sum_b_op = std::make_shared<nnfusion::op::Sum>(reduce_axesB);
    auto sum_b_gnode = graph->add_node_and_edge(sum_b_op, {para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(sum_a_op));
    EXPECT_TRUE(check_inplace_oi_pair(sum_b_op));
}

TEST(nnfusion_inplace_op, broadcast)
{
    // Create graph
    std::string name = "Broadcast";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    Shape shape_b{2, 3};

    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_b);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    ngraph::AxisSet broadcast_axesA;
    Shape output_shapeA{2, 3};
    ngraph::AxisSet broadcast_axesB{0};
    Shape output_shapeB{1, 2, 3};

    // Create node
    auto a_op = std::make_shared<nnfusion::op::Broadcast>(output_shapeA, broadcast_axesA);
    auto a_gnode = graph->add_node_and_edge(a_op, {para_a_gnode});
    auto b_op = std::make_shared<nnfusion::op::Broadcast>(output_shapeB, broadcast_axesB);
    auto b_gnode = graph->add_node_and_edge(b_op, {para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(a_op));
    EXPECT_TRUE(check_inplace_oi_pair(b_op));
}

TEST(nnfusion_inplace_op, max)
{
    // Create graph
    std::string name = "Max";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});
    ngraph::AxisSet reduction_axes;

    // Create node
    auto op = std::make_shared<nnfusion::op::Max>(reduction_axes);
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, min)
{
    // Create graph
    std::string name = "Min";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});
    ngraph::AxisSet reduction_axes;

    // Create node
    auto op = std::make_shared<nnfusion::op::Min>(reduction_axes);
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, abs)
{
    // Create graph
    std::string name = "Abs";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Abs>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, acos)
{
    // Create graph
    std::string name = "Acos";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Acos>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, asin)
{
    // Create graph
    std::string name = "Asin";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Asin>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, atan)
{
    // Create graph
    std::string name = "Atan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Atan>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, ceiling)
{
    // Create graph
    std::string name = "Ceiling";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Ceiling>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, cos)
{
    // Create graph
    std::string name = "Cos";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Cos>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, cosh)
{
    // Create graph
    std::string name = "Cosh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Cosh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, exp)
{
    // Create graph
    std::string name = "Exp";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Exp>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, floor)
{
    // Create graph
    std::string name = "Floor";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Floor>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, log)
{
    // Create graph
    std::string name = "Log";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Log>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sin)
{
    // Create graph
    std::string name = "Sin";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Sin>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sinh)
{
    // Create graph
    std::string name = "Sinh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Sinh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sqrt)
{
    // Create graph
    std::string name = "Sqrt";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Sqrt>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, tan)
{
    // Create graph
    std::string name = "Tan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Tan>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, tanh)
{
    // Create graph
    std::string name = "Tanh";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Tanh>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, power)
{
    // Create graph
    std::string name = "Power";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Power>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, subtract)
{
    // Create graph
    std::string name = "Subtract";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Subtract>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, divide)
{
    // Create graph
    std::string name = "Divide";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Divide>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, divnonan)
{
    // Create graph
    std::string name = "DivNoNan";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::DivNoNan>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sign)
{
    // Create graph
    std::string name = "Sign";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Sign>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, relu)
{
    // Create graph
    std::string name = "Relu";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Relu>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, negative)
{
    // Create graph
    std::string name = "Negative";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_gnode = graph->add_node_and_edge(para_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Negative>();
    auto gnode = graph->add_node_and_edge(op, {para_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, select)
{
    // Create graph
    std::string name = "Select";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::boolean, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});
    auto para_c_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_c_gnode = graph->add_node_and_edge(para_c_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Select>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode, para_c_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, relubackprop)
{
    // Create graph
    std::string name = "ReluBackprop";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::ReluBackprop>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, add)
{
    // Create graph
    std::string name = "Add";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Add>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, addn)
{
    // Create graph
    std::string name = "AddN";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});
    auto para_c_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_c_gnode = graph->add_node_and_edge(para_c_op, {});

    // Create node
    nnfusion::op::OpConfig::any myConfig;
    auto op = std::make_shared<nnfusion::op::GenericOp>(name, name, myConfig);
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode, para_c_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, multiply)
{
    // Create graph
    std::string name = "Multiply";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Multiply>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, minimum)
{
    // Create graph
    std::string name = "Minimum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Minimum>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, maximum)
{
    // Create graph
    std::string name = "Maximum";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Maximum>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sigmoid)
{
    // Create graph
    std::string name = "Sigmoid";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::Sigmoid>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

TEST(nnfusion_inplace_op, sigmoidbackprop)
{
    // Create graph
    std::string name = "SigmoidBackprop";
    auto graph = std::make_shared<nnfusion::graph::Graph>(name);

    // Prepare inputs
    Shape shape_a{2, 3};
    auto para_a_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_a_gnode = graph->add_node_and_edge(para_a_op, {});
    auto para_b_op = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
    auto para_b_gnode = graph->add_node_and_edge(para_b_op, {});

    // Create node
    auto op = std::make_shared<nnfusion::op::SigmoidBackprop>();
    auto gnode = graph->add_node_and_edge(op, {para_a_gnode, para_b_gnode});

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(op));
}

// TEST(nnfusion_inplace_op, shared_UnaryElementwiseArithmetic)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args = shared_ptr<ngraph::Node>(A);

//     // Create node
//     auto nodeA = std::make_shared<nnfusion::op::Tan>(args);
//     auto nodeB = std::make_shared<nnfusion::op::Sqrt>(args);

//     // Create graph
//     ngraph::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A};
//     std::string name = "UnaryElementwiseArithmetic";
//     auto func = make_shared<ngraph::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }

// TEST(nnfusion_inplace_op, shared_BinaryElementwiseArithmetic)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args0 = shared_ptr<ngraph::Node>(A);
//     auto B = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args1 = shared_ptr<ngraph::Node>(B);

//     // Create node
//     auto nodeA = std::make_shared<nnfusion::op::Power>(args0, args1);
//     auto nodeB = std::make_shared<nnfusion::op::Subtract>(args0, args1);

//     // Create graph
//     ngraph::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A, B};
//     std::string name = "BinaryElementwiseArithmetic";
//     auto func = make_shared<ngraph::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }

// TEST(nnfusion_inplace_op, shared_select_andn)
// {
//     // Prepare inputs
//     Shape shape_a{2, 3};
//     auto A = make_shared<nnfusion::op::Parameter>(element::boolean, shape_a);
//     auto args0 = shared_ptr<ngraph::Node>(A);
//     auto B = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args1 = shared_ptr<ngraph::Node>(B);
//     auto C = make_shared<nnfusion::op::Parameter>(element::f32, shape_a);
//     auto args2 = shared_ptr<ngraph::Node>(C);
//     auto inputs = vector<shared_ptr<ngraph::Node>>{B, C};

//     // Create node
//     string node_type("AddN");
//     auto nodeA = std::make_shared<nnfusion::op::Select>(args0, args1, args2);
//     nnfusion::op::OpConfig::any myConfig;
//     auto nodeB = std::make_shared<nnfusion::op::GenericOp>(node_type, node_type, inputs, myConfig);

//     // Create graph
//     ngraph::NodeVector res{nodeA, nodeB};
//     nnfusion::op::ParameterVector parameters{A, B, C};
//     std::string name = "Select_AddN";
//     auto func = make_shared<ngraph::Function>(res, parameters, name);
//     auto graph = make_shared<nnfusion::graph::Graph>(func, name);

//     run(graph);

//     EXPECT_FALSE(check_inplace_oi_pair(nodeA));
//     EXPECT_FALSE(check_inplace_oi_pair(nodeB));
// }
