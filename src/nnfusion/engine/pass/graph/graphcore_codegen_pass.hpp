// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion::graph;

DECLARE_string(fdefault_device);

DEFINE_bool(fapply_blockfusion, false, "Whether to apply blockfusion for GraphCore codegen.");

DEFINE_string(fantares_gc_server,
              "10.150.145.98:8883",
              "Antares graphcore server address and port, format: <ip>:<port>");

namespace
{
    template <class T1, class T2>
    inline std::string join_collections(const T1& vect, T2 func, bool skip_empty = false)
    {
        std::stringstream result;
        int idx = 0;
        for (auto& it : vect)
        {
            auto str = func(idx, it);
            if (!str.size() && skip_empty)
                continue;
            if (idx > 0)
                result << ", ";
            result << str;
            ++idx;
        }
        return result.str();
    }
}

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphCoreCodegenPass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    if (FLAGS_fdefault_device != "GraphCore")
                        return true;

                    auto nodes = graph->get_nodes();
                    std::unordered_map<std::shared_ptr<GNode>, int> din, dout;

                    // Count degrees
                    for (auto& it : nodes)
                    {
                        for (auto& in_edge : it->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                                continue;
                            CHECK(in_edge->get_dst() == it);
                            din[it]++;
                            dout[in_edge->get_src()]++;
                        }
                    }

                    // Name nodes, legality checks
                    std::unordered_set<std::shared_ptr<GNode>> visited, vis_pend, blacklist;
                    std::unordered_set<std::string> name_used;
                    std::unordered_map<std::shared_ptr<GNode>, std::string> arg_names;
                    for (auto& it : nodes)
                    {
                        CHECK(it.get() != nullptr);

                        auto arg_name = "Z0_" + it->get_op_ptr()->get_op_type() + "_" +
                                        it->get_op_ptr()->get_name();
                        for (auto& c : arg_name)
                            if (!isalpha(c) && !isdigit(c))
                                c = '_';
                        if (name_used.count(arg_name))
                        {
                            for (int i = 1;; ++i)
                            {
                                auto alter = arg_name + "_" + std::to_string(i);
                                if (!name_used.count(alter))
                                {
                                    arg_name = alter;
                                    break;
                                }
                            }
                        }
                        name_used.insert(arg_name);
                        arg_names[it] = arg_name;

                        if (din[it] == 0 && dout[it] == 0)
                            visited.insert(it), blacklist.insert(it);
                        CHECK(it->get_output_size() == 1);
                    }
                    name_used.clear();

                    // Fill offsetup nodes
                    std::deque<std::shared_ptr<GNode>> gen_q, pend_q;
                    for (auto& it : nodes)
                    {
                        if (visited.count(it))
                            continue;
                        if (din[it] == 0)
                        {
                            gen_q.push_back(it);
                            visited.insert(it);
                        }
                    }

                    std::ofstream fout("nnfusion_rt.h");
                    // Perform blockfusion
                    int offset = 0, step = 0;
                    auto new_super_step = [&]() {
                        while (pend_q.size())
                        {
                            gen_q.push_back(pend_q.front());
                            pend_q.pop_front();
                        }
                        if (offset > 0)
                            ++step, offset = 0;
                    };

                    while (gen_q.size() > 0 || pend_q.size() > 0)
                    {
                        // Move to new super step if satisifed
                        if (!gen_q.size())
                            new_super_step();

                        auto curr = gen_q.front();
                        gen_q.pop_front();
                        visited.insert(curr);

                        // Check its children about whether all inputs are ready
                        for (auto& edge : curr->get_out_edges())
                        {
                            if (edge->is_control_edge())
                                continue;
                            CHECK(edge->get_src() == curr);
                            CHECK(visited.count(edge->get_dst()) == 0);

                            bool ready = true;
                            for (auto& from : edge->get_dst()->get_in_edges())
                            {
                                if (from->is_control_edge())
                                    continue;
                                if (visited.count(from->get_src()) == 0)
                                {
                                    ready = false;
                                    break;
                                }
                            }
                            if (ready)
                            {
                                // Only join pend_q once
                                if (vis_pend.count(edge->get_dst()) == 0)
                                {
                                    vis_pend.insert(edge->get_dst());
                                    pend_q.push_back(edge->get_dst());
                                }
                            }
                        }

                        auto no_scaler = [](const std::string& str) {
                            if (str.size())
                                return str;
                            return std::string("1");
                        };

                        // Print codes for each Op
                        if (curr->get_op_ptr()->get_op_type() == "Constant")
                        {
                            // TODO:
                            // 1) handle more types than float only;
                            assert(curr->get_output_element_type(0) == nnfusion::element::f32);
                            auto p_const =
                                std::dynamic_pointer_cast<op::Constant>(curr->get_op_ptr());
                            CHECK(p_const != nullptr);
                            auto dptr = (float*)p_const->get_data_ptr();
                            auto size = p_const->get_data_size();
                            CHECK(size % sizeof(float) == 0);
                            size /= sizeof(float);

                            std::string ss;
                            for (int i = 0; i < size; ++i)
                                ss += std::to_string(dptr[i]) + "f, ";
                            if (ss.size() >= 2)
                                ss = ss.substr(0, ss.size() - 2);

                            fout << "auto " << arg_names[curr]
                                 << " = g.addConstant<float>(FLOAT, {";
                            fout << no_scaler(join_collections(
                                curr->get_output_shape(0),
                                [](int idx, ssize_t it) { return std::to_string(it); }));
                            fout << "}, {" << ss << "}); place_tensor(g, " << arg_names[curr]
                                 << ");\n";
                        }
                        else if (curr->get_op_ptr()->get_op_type() == "Parameter")
                        {
                            // TODO:
                            // 1) using g.addVariable + stream_HtoD instead of addConstant;
                            // 2) handle more types than float only;
                            assert(curr->get_output_element_type(0) == nnfusion::element::f32);
                            fout << "auto " << arg_names[curr]
                                 << " = g.addConstant<float>(FLOAT, {";
                            auto str_vect = vector_to_string(curr->get_output_shape(0));
                            fout << no_scaler(join_collections(
                                curr->get_output_shape(0),
                                [](int idx, ssize_t it) { return std::to_string(it); }));
                            fout << "}, {1.0f}); place_tensor(g, " << arg_names[curr] << ");\n";
                        }
                        else
                        {
                            bool standard_kernel = true;

                            auto UNHANDLED_CASE = [](std::shared_ptr<GNode>& curr) {
                                printf("## Unhandled case for %s:\n",
                                       curr->get_op_ptr()->get_op_type().c_str());
                                for (int i = 0; i < curr->get_input_size(); ++i)
                                    printf(">> in-%d : %s\n",
                                           i,
                                           vector_to_string(curr->get_input_shape(i)).c_str());
                                for (int i = 0; i < curr->get_output_size(); ++i)
                                    printf(">> out-%d: %s\n",
                                           i,
                                           vector_to_string(curr->get_output_shape(i)).c_str());
                                _exit(1);
                            };

                            auto autogen = [](const std::string& expr) -> std::string {
                                CurlRequest req(FLAGS_fantares_gc_server);
                                req.add_custom_header(("COMPUTE_V1: " + expr).c_str());
                                req.add_custom_header("ARGS: ");

                                printf("[GraphCore] %s\n", expr.c_str());
                                std::string response;
                                CHECK(true == req.send_request(response));
                                CHECK(strncmp(response.c_str(), "[ERROR]", 7) != 0) << expr;
                                return std::move(response);
                            };

                            std::unordered_map<std::string, std::function<std::string()>>
                                elementwise_ops = {
                                    {"Add",
                                     []() {
                                         return "topi=topi.add(args(\"input0\"), args(\"input1\"))";
                                     }},
                                    {"Divide",
                                     []() {
                                         return "topi=topi.divide(args(\"input0\"), "
                                                "args(\"input1\"))";
                                     }},
                                    {"Multiply",
                                     []() {
                                         return "topi=topi.multiply(args(\"input0\"), "
                                                "args(\"input1\"))";
                                     }},
                                    {"Exp", []() { return "topi=topi.exp(args(\"input0\"))"; }},
                                    {"Negative",
                                     []() { return "topi=topi.negative(args(\"input0\"))"; }},
                                    {"Tanh", []() { return "topi=topi.tanh(args(\"input0\"))"; }},
                                };

                            std::string code = "..unimplemented..";
                            if (elementwise_ops.count(curr->get_op_ptr()->get_op_type()))
                            {
                                std::string expr = " -";
                                for (int i = 0; i < curr->get_input_size(); ++i)
                                    expr += " input(\"input" + std::to_string(i) +
                                            "\", @input_shape@);";
                                expr += "output(@input_shape@, " +
                                        elementwise_ops[curr->get_op_ptr()->get_op_type()]() + ");";
                                code = autogen(op::create_code_from_template(
                                    expr,
                                    {{"input_shape", vector_to_string(curr->get_input_shape(0))}}));
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Broadcast")
                            {
                                code = autogen(nnfusion::op::get_translation(curr));
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Reshape")
                            {
                                nnfusion::Shape simp_in, simp_out;
                                for (auto& it : curr->get_input_shape(0))
                                    if (it > 1)
                                        simp_in.push_back(it);
                                for (auto& it : curr->get_output_shape(0))
                                    if (it > 1)
                                        simp_out.push_back(it);
                                if (simp_out == simp_in)
                                { // Memcpy
                                    standard_kernel = false;
                                    fout << "auto &" << arg_names[curr] << " = "
                                         << arg_names[curr->get_in_edge(0)->get_src()] << ";\n";
                                }
                                else
                                {
                                    UNHANDLED_CASE(curr);
                                }
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Slice")
                            {
                                auto _op =
                                    static_pointer_cast<nnfusion::op::Slice>(curr->get_op_ptr());
                                CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                       << curr->get_op_ptr()->get_op_type();

                                code = autogen(op::create_code_from_template(
                                    R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.strided_slice(args("input0"), begin=@begin@, end=@end@, strides=@strides@)); )",
                                    {{"input_shape", vector_to_string(curr->get_input_shape(0))},
                                     {"output_shape", vector_to_string(curr->get_output_shape(0))},
                                     {"begin", vector_to_string(_op->get_lower_bounds())},
                                     {"end", vector_to_string(_op->get_upper_bounds())},
                                     {"strides", vector_to_string(_op->get_strides())}}));
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Dot")
                            {
                                auto _op =
                                    static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
                                CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                       << curr->get_op_ptr()->get_op_type();

                                CHECK(_op->get_transpose_A() == false);
                                CHECK(_op->get_transpose_B() == false);

                                auto shape_0 = curr->get_input_shape(0);
                                auto shape_1 = curr->get_input_shape(1);
                                int N = shape_0[0], K = shape_0[1], M = shape_1[1];

                                code = autogen(op::create_code_from_template(
                                    R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@K@); output(@output_shape@, lambda i, j: tvm.sum(args("input0")[i, k] * args("input1")[k, j], axis=k)); )",
                                    {{"input_shape_0", vector_to_string(shape_0)},
                                     {"input_shape_1", vector_to_string(shape_1)},
                                     {"output_shape", vector_to_string(curr->get_output_shape(0))},
                                     {"K", K}}));
                            }
                            else
                            {
                                UNHANDLED_CASE(curr);
                            }

                            if (standard_kernel)
                            {
                                int tiles = 1, pos = 0, next;
                                while (next = code.find("// [thread_extent] threadIdx_", pos),
                                       next >= 0)
                                {
                                    int eq = code.find(" = ", next);
                                    CHECK(eq >= 0);
                                    tiles *= atoi(code.c_str() + eq + 3);
                                    pos = eq;
                                }

                                // if no enough tiles, then new_super_step()
                                const int max_tiles = 1216;
                                if (FLAGS_fapply_blockfusion == false || offset + tiles > max_tiles)
                                {
                                    new_super_step();
                                    CHECK(offset + tiles <= max_tiles);
                                }

                                fout << "auto " << arg_names[curr] << " = compute_task(g, {";
                                std::vector<int> range(curr->get_input_size());
                                fout << join_collections(
                                            range,
                                            [&](int idx, int val) {
                                                return arg_names[curr->get_in_edge(idx)->get_src()];
                                            })
                                     << "}, R\"(" << code << ")\", ";
                                fout << step << ", " << offset << ", " << offset + tiles << ");\n";
                                offset += tiles;
                            }
                        }
                        fout << std::endl;
                    }

                    // Print Results

                    for (auto& curr : graph->get_outputs()) // Print output nodes
                    // for (auto& curr : graph->get_nodes()) // Print all nodes
                    {
                        if (blacklist.count(curr))
                            continue;
                        fout << "print_tensor(\"Result(" << arg_names[curr] << ")\", "
                             << arg_names[curr] << ");\n";
                        fout << std::endl;
                    }
                    printf("============= %s Codegen Finished =============\n",
                           FLAGS_fdefault_device.c_str());
                    exit(0);
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
