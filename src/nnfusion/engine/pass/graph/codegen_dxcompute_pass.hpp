// Microsoft (c) 2020, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion::graph;

DECLARE_string(fdefault_device);
DECLARE_string(fantares_codegen_server);

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class DirectComputeCodegenPass : public GraphPassBase
            {
                std::string currentBackend;
                std::string autogen(const std::string& expr)
                {
                    if (FLAGS_fantares_codegen_server == "")
                        FLAGS_fantares_codegen_server = "10.150.145.98:8884";
                    static std::unordered_map<std::string, std::string> code_cache;
                    std::string response;
                    auto it = code_cache.find(expr);
                    if (it == code_cache.end())
                    {
                        CurlRequest req(FLAGS_fantares_codegen_server);
                        req.add_custom_header(("COMPUTE_V1: " + expr).c_str());
                        req.add_custom_header("ARGS: ");

                        printf("[Autogen] %s\n", expr.c_str());
                        NNFUSION_CHECK(true == req.send_request(response));
                        NNFUSION_CHECK(strncmp(response.c_str(), "[ERROR]", 7) != 0) << expr << "\n"
                                                                                     << response;
                        code_cache[expr] = response;
                        return std::move(response);
                    }
                    else
                        return it->second;
                }

                template <class T1, class T2>
                inline std::string
                    join_collections(const T1& vect, T2 func, bool skip_empty = false)
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
                    // TODO: fill more type cases
                    if (type == nnfusion::element::f32)
                        return DT_FLOAT;
                    throw std::runtime_error("Not supported element type.");
                }

                template <class T>
                inline std::shared_ptr<T> get_op_object(std::shared_ptr<GNode>& curr)
                {
                    auto _op = static_pointer_cast<T>(curr->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not "
                                                    << curr->get_op_ptr()->get_op_type();
                    return _op;
                }

                inline void UNHANDLED_CASE(std::shared_ptr<GNode>& curr)
                {
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
                    exit(1);
                };

            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override
                {
                    currentBackend = "dxcompute";

                    if (FLAGS_fdefault_device != currentBackend)
                        return true;

                    NNFUSION_LOG(INFO) << "Codegen for " << currentBackend << " starts up.";

                    auto nodes = graph->get_nodes();
                    std::unordered_map<std::shared_ptr<GNode>, int> din, dout;

                    // Count degrees
                    for (auto& it : nodes)
                    {
                        for (auto& in_edge : it->get_in_edges())
                        {
                            if (in_edge->is_control_edge())
                                continue;
                            NNFUSION_CHECK(in_edge->get_dst() == it);
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
                        NNFUSION_CHECK(it.get() != nullptr);

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
                        NNFUSION_CHECK(it->get_output_size() == 1);
                    }
                    NNFUSION_LOG(INFO) << "There are " << blacklist.size()
                                       << " standalone GNode(s) found.";
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

                    NNFUSION_CHECK(
                        0 ==
                        system(("mkdir -p nnfusion_rt/" + currentBackend + "_codegen").c_str()));

                    std::ofstream fout("nnfusion_rt/" + currentBackend + "_codegen/nnfusion_rt.h");

                    fout << "#if 1\n\n";
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

                    auto print_standard_codegen = [&](std::shared_ptr<GNode>& curr,
                                                      std::ofstream& fout,
                                                      const std::string& code) {

                        NNFUSION_CHECK(
                            0 == system(("mkdir -p nnfusion_rt/" + currentBackend + "_codegen/HLSL")
                                            .c_str()));
                        FILE* fp = fopen(("nnfusion_rt/" + currentBackend + "_codegen/HLSL/" +
                                          arg_names[curr] + ".hlsl")
                                             .c_str(),
                                         "wb");
                        NNFUSION_CHECK(fp != nullptr);
                        NNFUSION_CHECK(code.size() == fwrite(code.c_str(), 1, code.size(), fp));
                        fclose(fp);

                        int at_x = code.find("blockIdx.x ="),
                            blockX = (at_x >= 0)
                                         ? atoi(code.c_str() + at_x + sizeof("blockIdx.x ="))
                                         : 1;
                        int at_y = code.find("blockIdx.y ="),
                            blockY = (at_y >= 0)
                                         ? atoi(code.c_str() + at_y + sizeof("blockIdx.y ="))
                                         : 1;
                        int at_z = code.find("blockIdx.z ="),
                            blockZ = (at_z >= 0)
                                         ? atoi(code.c_str() + at_z + sizeof("blockIdx.z ="))
                                         : 1;
                        // "\nStructuredBuffer"

                        fout << "NNfusionTensor " << arg_names[curr] << "(device, {"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [](int idx, ssize_t it) { return std::to_string(it); })
                             << "}, sizeof(" << curr->get_output_element_type(0).c_type_string()
                             << "));\n";

                        fout << "  NNfusionOperator op_" << arg_names[curr]
                             << "(device, cmdQueue, {";
                        for (int i = 0; i < curr->get_input_size(); ++i)
                        {
                            if (i)
                                fout << ", ";
                            fout << arg_names[curr->get_in_edge(i)->get_src()];
                        }
                        fout << "}, { " << arg_names[curr] << " }, {" << blockX << ", " << blockY
                             << ", " << blockZ << "}, L\"" << arg_names[curr] << ".hlsl\");";
                    };

                    auto codegen_for_elementwise = [&](std::shared_ptr<GNode>& curr,
                                                       std::ofstream& fout,
                                                       const std::string& topi) {
                        std::string expr = " -";
                        for (int i = 0; i < curr->get_input_size(); ++i)
                            expr += " input(\"input" + std::to_string(i) + "\", @common_shape@);";
                        expr += " output(@common_shape@, " + topi + ");";

                        int num_elements = 1, y;
                        for (auto& it : curr->get_input_shape(0))
                            num_elements *= it;

                        auto code = autogen(op::create_code_from_template(
                            expr, {{"common_shape", "[ " + std::to_string(num_elements) + " ]"}}));
                        print_standard_codegen(curr, fout, code);
                    };

                    std::unordered_map<std::string,
                                       std::function<void(std::shared_ptr<GNode>&, std::ofstream&)>>
                        kernel_dict;

                    // Elementwise Ops
                    kernel_dict["Add"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.add(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Subtract"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.subtract(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Multiply"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.multiply(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Divide"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.divide(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Power"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.power(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["LessEq"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.less_equal(args(\"input0\"), args(\"input1\"))");
                    };
                    kernel_dict["Exp"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.exp(args(\"input0\"))");
                    };
                    kernel_dict["Negative"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.negative(args(\"input0\"))");
                    };
                    kernel_dict["Tanh"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.tanh(args(\"input0\"))");
                    };
                    kernel_dict["Relu"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.relu(args(\"input0\"))");
                    };
                    kernel_dict["Relu6"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        codegen_for_elementwise(
                            curr, fout, "topi=topi.clip(args(\"input0\"), 0, 6)");
                    };
                    kernel_dict["Sigmoid"] = [&](std::shared_ptr<GNode>& curr,
                                                 std::ofstream& fout) {
                        codegen_for_elementwise(curr, fout, "topi=topi.sigmoid(args(\"input0\"))");
                    };

                    // Other Ops
                    kernel_dict["Constant"] = [&](std::shared_ptr<GNode>& curr,
                                                  std::ofstream& fout) {
                        auto p_const = std::dynamic_pointer_cast<op::Constant>(curr->get_op_ptr());
                        NNFUSION_CHECK(p_const != nullptr);
                        const void* dptr = p_const->get_data_ptr();
                        size_t size = p_const->get_data_size();

                        NNFUSION_CHECK(0 == system(("mkdir -p nnfusion_rt/" + currentBackend +
                                                    "_codegen/Constant")
                                                       .c_str()));
                        FILE* fp = fopen(("nnfusion_rt/" + currentBackend + "_codegen/Constant/" +
                                          arg_names[curr])
                                             .c_str(),
                                         "wb");
                        NNFUSION_CHECK(fp != nullptr);
                        NNFUSION_CHECK(size == fwrite(dptr, 1, size, fp));
                        fclose(fp);

                        fout << "NNfusionTensor " << arg_names[curr] << "(device, {"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [](int idx, ssize_t it) { return std::to_string(it); })
                             << "}, sizeof(" << curr->get_output_element_type(0).c_type_string()
                             << "));\n";

                        fout << "  NNfusionMemcpy op_" << arg_names[curr] << "(device, cmdQueue, "
                             << arg_names[curr] << ", load_data<"
                             << curr->get_output_element_type(0).c_type_string() << ">(\""
                             << arg_names[curr] << "\", " << arg_names[curr]
                             << ".NumElements()).data());\n";
                    };

                    kernel_dict["Parameter"] = [&](std::shared_ptr<GNode>& curr,
                                                   std::ofstream& fout) {
                        fout << "NNfusionTensor " << arg_names[curr] << "(device, {"
                             << join_collections(
                                    curr->get_output_shape(0),
                                    [](int idx, ssize_t it) { return std::to_string(it); })
                             << "}, sizeof(" << curr->get_output_element_type(0).c_type_string()
                             << "));\n";

                        fout << "  NNfusionMemcpy op_" << arg_names[curr] << "(device, cmdQueue, "
                             << arg_names[curr] << ", load_data<"
                             << curr->get_output_element_type(0).c_type_string() << ">(\"\", "
                             << arg_names[curr] << ".NumElements()).data());\n";
                    };

                    kernel_dict["Result"] = [&](std::shared_ptr<GNode>& curr, std::ofstream& fout) {
                        fout << "NNfusionMemcpy " << arg_names[curr]
                             << "(device, cmdQueue, nullptr, "
                             << arg_names[curr->get_in_edge(0)->get_src()] << ");\n";
                    };

                    kernel_dict["Broadcast"] = [&](std::shared_ptr<GNode>& curr,
                                                   std::ofstream& fout) {
                        auto _op = get_op_object<nnfusion::op::Broadcast>(curr);
                        auto axes = _op->get_broadcast_axes();

                        auto original_out_shape = curr->get_output_shape(0);
                        std::vector<int> out_shape(original_out_shape.begin(),
                                                   original_out_shape.end()),
                            in_shape = out_shape;
                        for (auto& it : axes)
                            in_shape[it] = 1;

                        auto code = autogen(op::create_code_from_template(
                            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.broadcast_to(args("input0"), @output_shape@)); )",
                            {{"input_shape", in_shape}, {"output_shape", out_shape}}));
                        print_standard_codegen(curr, fout, code);
                    };

                    while (gen_q.size() > 0 || pend_q.size() > 0)
                    {
                        // Move to new super step if satisifed
                        if (!gen_q.size())
                            new_super_step();

                        auto curr = gen_q.front();
                        gen_q.pop_front();
                        visited.insert(curr);

                        // fout << "DEBUG(\"" << arg_names[curr] << "\");\n";

                        auto entry = kernel_dict.find(curr->get_op_ptr()->get_op_type());
                        if (entry != kernel_dict.end())
                            entry->second(curr, fout);
                        else
                        {
                            UNHANDLED_CASE(curr);
                        }
                        fout << std::endl;

                        // Check its children about whether all inputs are ready (Must be put after any possible new_super_step())
                        for (auto& edge : curr->get_out_edges())
                        {
                            if (edge->is_control_edge())
                                continue;
                            NNFUSION_CHECK(edge->get_src() == curr);
                            NNFUSION_CHECK(visited.count(edge->get_dst()) == 0);

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

                    fout << "#endif\n\n";
                    fout << "device.pCommandQueue->ExecuteCommandLists(cmdQueue.size(), "
                            "cmdQueue.data());\ndevice.AwaitExecution();\n\n";

                    // Print Results
                    for (auto& curr : graph->get_outputs()) // Print output nodes
                    {
                        if (blacklist.count(curr))
                            continue;
                        fout << arg_names[curr] << ".PrintStageBuffer<"
                             << curr->get_output_element_type(0).c_type_string() << ">(device, \""
                             << arg_names[curr] << "\");\n";
                    }
                    fout << std::endl;

                    nnfusion::codegen::copy_file_from_templates(currentBackend + "/run_graph.cpp",
                                                                "nnfusion_rt/" + currentBackend +
                                                                    "_codegen/run_graph.cpp");
                    nnfusion::codegen::copy_file_from_templates(currentBackend + "/d3dx12_helper.h",
                                                                "nnfusion_rt/" + currentBackend +
                                                                    "_codegen/d3dx12_helper.h");
                    nnfusion::codegen::copy_file_from_templates(
                        currentBackend + "/d3dx12_nnfusion.h",
                        "nnfusion_rt/" + currentBackend + "_codegen/d3dx12_nnfusion.h");
                    NNFUSION_LOG(INFO) << currentBackend << " codegen finished.";
                    exit(0);
                    return true;
                }
            };
        } // namespace pass
    }     // namespace graph
} // namespace nnfusion