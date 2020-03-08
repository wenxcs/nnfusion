// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion::graph;

DECLARE_string(fdefault_device);

DEFINE_bool(fgc_apply_blockfusion, true, "Whether to apply blockfusion for GraphCore codegen.");
DEFINE_string(fantares_gc_server,
              "10.150.145.98:8883",
              "Antares graphcore server address and port, format: <ip>:<port>");

namespace
{
    const int max_tiles = 1216;

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

    inline int get_type_id(nnfusion::element::Type type)
    {
        if (type == nnfusion::element::f32)
            return DT_FLOAT;
        throw std::runtime_error("Not supported element type.");
    }

    inline void UNHANDLED_CASE(std::shared_ptr<GNode>& curr)
    {
        printf("## Unhandled case for %s:\n", curr->get_op_ptr()->get_op_type().c_str());
        for (int i = 0; i < curr->get_input_size(); ++i)
            printf(">> in-%d : %s\n", i, vector_to_string(curr->get_input_shape(i)).c_str());
        for (int i = 0; i < curr->get_output_size(); ++i)
            printf(">> out-%d: %s\n", i, vector_to_string(curr->get_output_shape(i)).c_str());
        _exit(1);
    };
}

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphCoreCodegenPass : public GraphPassBase
            {
                void private_transpose_weight_pass(std::shared_ptr<Graph>& graph)
                {
                    for (auto& curr : graph->get_nodes())
                    {
                        if (curr->get_op_ptr()->get_op_type() == "Dot")
                        {
                            auto _op = static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
                            CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                   << curr->get_op_ptr()->get_op_type();

                            CHECK(_op->get_transpose_A() == false);
                            if (_op->get_transpose_B() == false)
                            {
                                auto weight = curr->get_in_edge(1)->get_src();
                                auto weight_shape = weight->get_output_shape(0);
                                CHECK(weight->get_op_ptr()->get_op_type() == "Constant")
                                    << "Only for constant-weight Dot optimization.";
                                CHECK(weight_shape.size() == 2)
                                    << "Only for 2D weight optimization.";
                                CHECK(weight->get_output_element_type(0) == nnfusion::element::f32);

                                auto p_const =
                                    std::dynamic_pointer_cast<op::Constant>(weight->get_op_ptr());
                                CHECK(p_const != nullptr);

                                _op->get_transpose_B() = true;
                                decltype(weight_shape) trans_shape = {weight_shape[1],
                                                                      weight_shape[0]};

                                std::vector<float> dtmp(weight_shape[0] * weight_shape[1]);
                                auto dptr = (float*)p_const->get_data_ptr();
                                for (int i = 0; i < weight_shape[0]; ++i)
                                    for (int j = 0; j < weight_shape[1]; ++j)
                                        dtmp[i + j * weight_shape[0]] =
                                            dptr[i * weight_shape[1] + j];

                                // Avoid duplicated constant creation
                                static std::unordered_map<uint64_t, std::shared_ptr<op::Constant>>
                                    constant_cache;
                                int hash_digest[2] = {
                                    get_type_id(weight->get_output_element_type(0)), 0};
                                for (int i = 0; i < dtmp.size(); ++i)
                                    hash_digest[i & 1] ^= (int&)dtmp[i];
                                auto it = constant_cache.find(*(uint64_t*)hash_digest);

                                std::shared_ptr<op::Constant> new_constant_op;
                                if (it != constant_cache.end())
                                    new_constant_op = it->second;
                                else
                                    new_constant_op = std::make_shared<op::Constant>(
                                        weight->get_output_element_type(0),
                                        trans_shape,
                                        dtmp.data());
                                dtmp.clear();
                                constant_cache[*(uint64_t*)hash_digest] = new_constant_op;

                                auto new_weight = std::make_shared<nnfusion::graph::GNode>(
                                    new_constant_op, GNodeVector());
                                new_weight->get_op_ptr()->revalidate_and_infer_types(new_weight);

                                if (weight->get_out_edges().size() == 1)
                                {
                                    graph->replace_node(weight, new_weight, false);
                                }
                                else
                                {
                                    graph->add_node(new_weight);
                                    graph->remove_edge(curr->get_in_edge(1));
                                    graph->add_edge(new_weight, 0, curr, 1);
                                }
                            }
                        }
                    }
                }

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    if (FLAGS_fdefault_device != "GraphCore")
                        return true;

                    LOG(INFO) << "GraphCore codegen starts up.";
                    // private_transpose_weight_pass(graph);

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
                        }
                    }

                    CHECK(0 == system("mkdir -p nnfusion_rt/graphcore_codegen"));
                    nnfusion::codegen::copy_file_from_templates(
                        "graphcore/Makefile", "nnfusion_rt/graphcore_codegen/Makefile");
                    nnfusion::codegen::copy_file_from_templates(
                        "graphcore/run_graph.cpp", "nnfusion_rt/graphcore_codegen/run_graph.cpp");

                    std::ofstream fout("nnfusion_rt/graphcore_codegen/nnfusion_rt.h");
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
                            CHECK(curr->get_output_element_type(0) == nnfusion::element::f32);
                            auto p_const =
                                std::dynamic_pointer_cast<op::Constant>(curr->get_op_ptr());
                            CHECK(p_const != nullptr);
                            auto dptr = (float*)p_const->get_data_ptr();
                            auto size = p_const->get_data_size();
                            CHECK(size % sizeof(float) == 0);
                            size /= sizeof(float);

                            CHECK(0 == system("mkdir -p nnfusion_rt/graphcore_codegen/Constant"));
                            FILE* fp =
                                fopen(("nnfusion_rt/graphcore_codegen/Constant/" + arg_names[curr])
                                          .c_str(),
                                      "wb");
                            CHECK(fp != nullptr);
                            CHECK(size == fwrite(dptr, sizeof(float), size, fp));
                            fclose(fp);

                            fout << "Tensor " << arg_names[curr]
                                 << " = g.addConstant<float>(FLOAT, {";
                            fout << no_scaler(join_collections(
                                curr->get_output_shape(0),
                                [](int idx, ssize_t it) { return std::to_string(it); }));
                            fout << "}, load_const<float>(\"" << arg_names[curr]
                                 << "\")); place_tensor(g, " << arg_names[curr] << ");\n";
                        }
                        else if (curr->get_op_ptr()->get_op_type() == "Parameter")
                        {
                            // TODO:
                            // 1) using g.addVariable + stream_HtoD instead of addConstant;
                            // 2) handle more types than float only;
                            assert(curr->get_output_element_type(0) == nnfusion::element::f32);
                            fout << "Tensor " << arg_names[curr]
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
                            std::string code = "..unimplemented..";
                            std::vector<int> shards(1 + curr->get_input_size(), 1);
                            std::vector<std::string> convert_input(curr->get_input_size());

                            auto autogen = [](const std::string& expr) -> std::string {
                                static std::unordered_map<std::string, std::string> code_cache;
                                std::string response;
                                auto it = code_cache.find(expr);
                                if (it == code_cache.end())
                                {
                                    CurlRequest req(FLAGS_fantares_gc_server);
                                    req.add_custom_header(("COMPUTE_V1: " + expr).c_str());
                                    req.add_custom_header("ARGS: ");

                                    printf("[GraphCore] %s\n", expr.c_str());
                                    CHECK(true == req.send_request(response));
                                    CHECK(strncmp(response.c_str(), "[ERROR]", 7) != 0) << expr;
                                    code_cache[expr] = response;
                                    return std::move(response);
                                }
                                else
                                    return it->second;
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

                            if (elementwise_ops.count(curr->get_op_ptr()->get_op_type()))
                            {
                                std::string expr = " -";
                                for (int i = 0; i < curr->get_input_size(); ++i)
                                    expr += " input(\"input" + std::to_string(i) +
                                            "\", @common_shape@);";
                                expr += " output(@common_shape@, " +
                                        elementwise_ops[curr->get_op_ptr()->get_op_type()]() + ");";

                                int num_elements = 1, y;
                                for (auto& it : curr->get_input_shape(0))
                                    num_elements *= it;
                                for (int i = max_tiles; i >= 1; --i)
                                    if (num_elements % i == 0)
                                    {
                                        y = i;
                                        break;
                                    }

                                for (auto& it : shards)
                                    it = y;
                                code = autogen(op::create_code_from_template(
                                    expr,
                                    {{"common_shape",
                                      "[ " + std::to_string(num_elements / y) + " ]"}}));
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
                                    fout << "Tensor " << arg_names[curr] << " = "
                                         << arg_names[curr->get_in_edge(0)->get_src()]
                                         << ".reshape({"
                                         << join_collections(curr->get_output_shape(0),
                                                             [&](int idx, ssize_t val) {
                                                                 return std::to_string(val);
                                                             })
                                         << "});\n";
                                }
                                else
                                {
                                    // Using poplar::Tensor::dimShuffle()
                                    UNHANDLED_CASE(curr);
                                }
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Concat")
                            {
                                auto _op =
                                    static_pointer_cast<nnfusion::op::Concat>(curr->get_op_ptr());
                                CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                       << curr->get_op_ptr()->get_op_type();

                                auto axis = _op->get_concatenation_axis();
                                CHECK(curr->get_input_size() == 2);

                                standard_kernel = false;
                                fout << "Tensor " << arg_names[curr] << " = concat("
                                     << arg_names[curr->get_in_edge(0)->get_src()] << ", "
                                     << arg_names[curr->get_in_edge(1)->get_src()] << ", " << axis
                                     << ");\n";
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Slice")
                            {
                                auto _op =
                                    static_pointer_cast<nnfusion::op::Slice>(curr->get_op_ptr());
                                CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                       << curr->get_op_ptr()->get_op_type();

                                bool builtin_slice = true;
                                for (auto& it : _op->get_strides())
                                    if (it != 1)
                                    {
                                        builtin_slice = false;
                                        break;
                                    }
                                if (builtin_slice)
                                {
                                    standard_kernel = false;
                                    fout << "Tensor " << arg_names[curr] << " = "
                                         << arg_names[curr->get_in_edge(0)->get_src()]
                                         << ".slice(ArrayRef<std::size_t>({"
                                         << join_collections(_op->get_lower_bounds(),
                                                             [&](int idx, ssize_t val) {
                                                                 return std::to_string(val);
                                                             })
                                         << "}), ArrayRef<std::size_t>({"
                                         << join_collections(_op->get_upper_bounds(),
                                                             [&](int idx, ssize_t val) {
                                                                 return std::to_string(val);
                                                             })
                                         << "}));\n";
                                }
                                else
                                {
                                    code = autogen(op::create_code_from_template(
                                        R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.strided_slice(args("input0"), begin=@begin@, end=@end@, strides=@strides@)); )",
                                        {{"input_shape",
                                          vector_to_string(curr->get_input_shape(0))},
                                         {"output_shape",
                                          vector_to_string(curr->get_output_shape(0))},
                                         {"begin", vector_to_string(_op->get_lower_bounds())},
                                         {"end", vector_to_string(_op->get_upper_bounds())},
                                         {"strides", vector_to_string(_op->get_strides())}}));
                                }
                            }
                            else if (curr->get_op_ptr()->get_op_type() == "Dot")
                            {
                                auto _op =
                                    static_pointer_cast<nnfusion::op::Dot>(curr->get_op_ptr());
                                CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                       << curr->get_op_ptr()->get_op_type();

                                CHECK(_op->get_transpose_A() == false);
                                CHECK(_op->get_transpose_B() == false);
                                convert_input[1] = ".transpose()";

                                auto shape_0 = curr->get_input_shape(0);
                                auto shape_1 = curr->get_in_edge(1)->get_src()->get_output_shape(0);
                                int N = shape_0[0], K = shape_0[1], M = shape_1[1];

                                if (N == 1 && M <= max_tiles)
                                {
                                    shards = {1, M, M};
                                    code = autogen(op::create_code_from_template(
                                        R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); k = loop(@K@); output(@output_shape@, lambda i: tvm.sum(args("input0")[k] * args("input1")[k], axis=k)); )",
                                        {{"input_shape_0", "[ " + std::to_string(K) + " ]"},
                                         {"input_shape_1", "[ " + std::to_string(K) + " ]"},
                                         {"output_shape", "[ 1 ]"},
                                         {"K", K}}));
                                }
                                else
                                {
                                    UNHANDLED_CASE(curr);
                                }
                            }
                            else
                            {
                                UNHANDLED_CASE(curr);
                            }

                            if (standard_kernel)
                            {
                                int tiles = shards.back(), pos = 0, next;
                                while (next = code.find("// [thread_extent] threadIdx_", pos),
                                       next >= 0)
                                {
                                    int eq = code.find(" = ", next);
                                    CHECK(eq >= 0);
                                    tiles *= atoi(code.c_str() + eq + 3);
                                    pos = eq;
                                }

                                // if no enough tiles, then new_super_step()
                                if (FLAGS_fgc_apply_blockfusion == false ||
                                    offset + tiles > max_tiles)
                                {
                                    new_super_step();
                                    CHECK(offset + tiles <= max_tiles);
                                }

                                fout << "Tensor " << arg_names[curr] << " = compute_task(g, {";
                                std::vector<int> range(curr->get_input_size());
                                fout
                                    << join_collections(
                                           range,
                                           [&](int idx, int val) {
                                               return arg_names[curr->get_in_edge(idx)->get_src()] +
                                                      convert_input[idx];
                                           })
                                    << "}, R\"(" << code << ")\", ";
                                fout << step << ", " << offset << ", " << offset + tiles << ", {"
                                     << join_collections(
                                            shards,
                                            [](int idx, int val) { return std::to_string(val); })
                                     << "}).reshape({"
                                     << join_collections(curr->get_output_shape(0),
                                                         [&](int idx, ssize_t val) {
                                                             return std::to_string(val);
                                                         })
                                     << "})"
                                     << ";\n";
                                offset += tiles;
                            }
                        }
                        fout << std::endl;

                        // Check its children about whether all inputs are ready (Must be put after any possible new_super_step())
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
                    LOG(INFO) << "GraphCore codegen finished.";
                    exit(0);
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion
