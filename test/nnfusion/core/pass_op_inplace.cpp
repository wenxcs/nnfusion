// Microsoft (c) 2019, NNFusion Team

#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../src/nnfusion/core/graph/pass/graph_pass.hpp"
#include "../src/nnfusion/core/graph/pass/manager.hpp"
#include "../src/nnfusion/core/graph/pass/op_inplace_pass.hpp"
#include "gtest/gtest.h"
#include "nnfusion/core/ops/generic_op.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph::pass;
using namespace std;
using namespace nnfusion::profiler;

bool run(std::shared_ptr<nnfusion::graph::Graph> graph)
{
    GraphPassManager pass_manager;

    pass_manager.register_pass<OpInplacePass>();
    pass_manager.run_passes(graph);

    return true;
}

bool check_inplace_oi_pair(shared_ptr<ngraph::op::Op> node)
{
    if (auto op = dynamic_pointer_cast<ngraph::op::Op>(node))
    {
        auto annotation = op->get_op_annotations();
        if (annotation && annotation->get_in_place_oi_pairs().size() > 0)
        {
            return true;
        }
    }
    return false;
}

TEST(nnfusion_op, reshape)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);
    ngraph::AxisVector input_order{0, 1};
    Shape output_shape{3, 2};

    // Create node
    auto node = std::make_shared<ngraph::op::Reshape>(args, input_order, output_shape);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Reshape";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, result)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Result>(args);

    // Create graph
    ngraph::ResultVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Result";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sum)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto argsA = shared_ptr<ngraph::Node>(A);
    ngraph::AxisSet reduce_axesA;

    Shape shape_b{2, 3, 1};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto argsB = shared_ptr<ngraph::Node>(B);
    ngraph::AxisSet reduce_axesB{2};

    // Create node
    auto nodeA = std::make_shared<ngraph::op::Sum>(argsA, reduce_axesA);
    auto nodeB = std::make_shared<ngraph::op::Sum>(argsB, reduce_axesB);

    // Create graph
    ngraph::NodeVector res{nodeA, nodeB};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Sum";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(nodeA));
    EXPECT_TRUE(check_inplace_oi_pair(nodeB));
}

TEST(nnfusion_op, broadcast)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto argsA = shared_ptr<ngraph::Node>(A);
    ngraph::AxisSet broadcast_axesA;
    Shape output_shapeA{2, 3};

    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto argsB = shared_ptr<ngraph::Node>(B);
    ngraph::AxisSet broadcast_axesB{0};
    Shape output_shapeB{1, 2, 3};

    // Create node
    auto nodeA = std::make_shared<ngraph::op::Broadcast>(argsA, output_shapeA, broadcast_axesA);

    auto nodeB = std::make_shared<ngraph::op::Broadcast>(argsB, output_shapeB, broadcast_axesB);

    // Create graph
    ngraph::NodeVector res{nodeA, nodeB};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Broadcast";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(nodeA));
    EXPECT_TRUE(check_inplace_oi_pair(nodeB));
}

TEST(nnfusion_op, max)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);
    ngraph::AxisSet reduction_axes;

    // Create node
    auto node = std::make_shared<ngraph::op::Max>(args, reduction_axes);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Max";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, min)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);
    ngraph::AxisSet reduction_axes;

    // Create node
    auto node = std::make_shared<ngraph::op::Min>(args, reduction_axes);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Min";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, abs)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Abs>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Abs";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, acos)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Acos>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Acos";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, asin)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Asin>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Asin";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, atan)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Atan>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Atan";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, ceiling)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Ceiling>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Ceiling";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, cos)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Cos>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Cos";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, cosh)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Cosh>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Cosh";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, exp)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Exp>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Exp";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, floor)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Floor>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Floor";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, log)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Log>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Log";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sin)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Sin>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Sin";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sinh)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Sinh>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Sinh";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sqrt)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Sqrt>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Sqrt";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, tan)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Tan>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Tan";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, tanh)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Tanh>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Tanh";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, power)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Power>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Power";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, subtract)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Subtract>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Subtract";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, divide)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Divide>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Divide";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, divnonan)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::DivNoNan>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "DivNoNan";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sign)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Sign>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Sign";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, relu)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Relu>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Relu";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, negative)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Negative>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Negative";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, select)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);
    auto C = make_shared<op::Parameter>(element::f32, shape_a);
    auto args2 = shared_ptr<ngraph::Node>(C);

    // Create node
    auto node = std::make_shared<ngraph::op::Select>(args0, args1, args2);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B, C};
    std::string name = "Select";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, relubackprop)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::ReluBackprop>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "ReluBackprop";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, add)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);

    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Add>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Add";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, addn)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{2, 3};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_c{2, 3};
    auto C = make_shared<op::Parameter>(element::f32, shape_c);
    auto inputs = vector<shared_ptr<ngraph::Node>>{A, B, C};

    string node_type("AddN");
    // Create node for AddN
    ngraph::op::OpConfig::any myConfig;
    auto node = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B, C};
    std::string name = "AddN";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, multiply)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);

    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Multiply>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Multiply";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, minimum)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);

    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Minimum>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Minimum";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, maximum)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);

    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::Maximum>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "Maximum";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sigmoid)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto node = std::make_shared<ngraph::op::Sigmoid>(args);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "Sigmoid";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, sigmoidbackprop)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto node = std::make_shared<ngraph::op::SigmoidBackprop>(args0, args1);

    // Create graph
    ngraph::NodeVector res{node};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "SigmoidBackprop";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_TRUE(check_inplace_oi_pair(node));
}

TEST(nnfusion_op, shared_UnaryElementwiseArithmetic)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args = shared_ptr<ngraph::Node>(A);

    // Create node
    auto nodeA = std::make_shared<ngraph::op::Tan>(args);
    auto nodeB = std::make_shared<ngraph::op::Sqrt>(args);

    // Create graph
    ngraph::NodeVector res{nodeA, nodeB};
    ngraph::op::ParameterVector parameters{A};
    std::string name = "UnaryElementwiseArithmetic";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_FALSE(check_inplace_oi_pair(nodeA));
    EXPECT_FALSE(check_inplace_oi_pair(nodeB));
}

TEST(nnfusion_op, shared_BinaryElementwiseArithmetic)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);

    // Create node
    auto nodeA = std::make_shared<ngraph::op::Power>(args0, args1);
    auto nodeB = std::make_shared<ngraph::op::Subtract>(args0, args1);

    // Create graph
    ngraph::NodeVector res{nodeA, nodeB};
    ngraph::op::ParameterVector parameters{A, B};
    std::string name = "BinaryElementwiseArithmetic";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_FALSE(check_inplace_oi_pair(nodeA));
    EXPECT_FALSE(check_inplace_oi_pair(nodeB));
}

TEST(nnfusion_op, shared_select_andn)
{
    // Prepare inputs
    Shape shape_a{2, 3};
    auto A = make_shared<op::Parameter>(element::boolean, shape_a);
    auto args0 = shared_ptr<ngraph::Node>(A);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto args1 = shared_ptr<ngraph::Node>(B);
    auto C = make_shared<op::Parameter>(element::f32, shape_a);
    auto args2 = shared_ptr<ngraph::Node>(C);
    auto inputs = vector<shared_ptr<ngraph::Node>>{B, C};

    // Create node
    string node_type("AddN");
    auto nodeA = std::make_shared<ngraph::op::Select>(args0, args1, args2);
    ngraph::op::OpConfig::any myConfig;
    auto nodeB = std::make_shared<ngraph::op::GenericOp>(node_type, node_type, inputs, myConfig);

    // Create graph
    ngraph::NodeVector res{nodeA, nodeB};
    ngraph::op::ParameterVector parameters{A, B, C};
    std::string name = "Select_AddN";
    auto func = make_shared<ngraph::Function>(res, parameters, name);
    auto graph = make_shared<nnfusion::graph::Graph>(func, name);

    run(graph);

    EXPECT_FALSE(check_inplace_oi_pair(nodeA));
    EXPECT_FALSE(check_inplace_oi_pair(nodeB));
}
