// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

using namespace nnfusion::graph;

DECLARE_string(fdefault_device);

DEFINE_bool(fapply_blockfusion, false, "Whether to apply blockfusion for GraphCore codegen.");

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
                    std::unordered_set<std::shared_ptr<GNode>> visited, vis_pend;
                    std::unordered_set<std::string> name_used;
                    std::unordered_map<std::shared_ptr<GNode>, std::string> arg_names;
                    for (auto& it : nodes)
                    {
                        CHECK(it.get() != nullptr);

                        auto arg_name = "Z0_" + it->get_op_ptr()->get_name();
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
                            visited.insert(it);
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
                            // 2) fill correct values from curr->get_op_ptr();
                            fout << "auto " << arg_names[curr]
                                 << " = g.addConstant<float>(FLOAT, {";
                            fout << no_scaler(join_collections(
                                curr->get_output_shape(0),
                                [](int idx, ssize_t it) { return std::to_string(it); }));
                            fout << "}, {1.0f}); place_tensor(g, " << arg_names[curr] << ");\n";
                        }
                        else if (curr->get_op_ptr()->get_op_type() == "Parameter")
                        {
                            // TODO:
                            // 1) using g.addVariable + stream_HtoD instead of addConstant;
                            // 2) handle more types than float only;
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
                                std::string cmd = "curl -H 'COMPUTE_V1:" + expr +
                                                  "' -H 'ARGS: ' -Ls http://10.150.145.98:8883/ -o "
                                                  "/tmp/current-gc.txt";
                                printf("[EXEC] %s\n", cmd.c_str());
                                CHECK(0 == system(cmd.c_str()));
                                std::ifstream t("/tmp/current-gc.txt");
                                std::string str((std::istreambuf_iterator<char>(t)),
                                                std::istreambuf_iterator<char>());
                                CHECK(strncmp(str.c_str(), "[Error]", 7) != 0);
                                return std::move(str);
                            };

                            std::string code = "..unimplemented..";
                            if (curr->get_op_ptr()->get_op_type() == "Broadcast")
                            {
                                CHECK(curr->get_input_size() == 1);
                                CHECK(curr->get_output_size() == 1);
                                code = autogen(nnfusion::op::get_translation(curr));
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Add")
                            {
                                CHECK(curr->get_input_shape(0) == curr->get_input_shape(1));
                                code = autogen(op::create_code_from_template(
                                    R"( - input("input0", @input_shape@); input("input1", @input_shape@); output(@input_shape@, topi=topi.add(args("input0"), args("input1"))); )",
                                    {{"input_shape", vector_to_string(curr->get_input_shape(0))},
                                     {"hybrid_symb",
                                      join_collections(curr->get_input_shape(0),
                                                       [&](int idx, ssize_t val) {
                                                           return "V" + std::to_string(idx);
                                                       })}}));
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Reshape")
                            {
                                nnfusion::Shape simp_out;
                                for (auto& it : curr->get_output_shape(0))
                                    if (it > 1)
                                        simp_out.push_back(it);
                                if (simp_out == curr->get_input_shape(0))
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

                    // for (auto &curr: graph->get_outputs()) { // Print output nodes
                    for (auto& curr : graph->get_nodes())
                    { // Print all nodes
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

#if 0
        nnfusion::Shape = ..->get_output_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto& axes_order = generic_op->localOpConfig.getRoot()["axes_order"];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
#endif
