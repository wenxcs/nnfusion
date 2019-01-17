// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::runtime::nnfusion::FunctionTranslator::translate(
    const std::shared_ptr<ngraph::Function> function)
{
    ngraph::pass::Manager pass_manager;
    /*
#if CUDNN_VERSION >= 7200
    // recurrent network fusion
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::gpu::pass::MultiLayerRNNFusion>();
#else
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
#endif
    pass_manager.register_pass<runtime::gpu::pass::BatchNormCache>();
    pass_manager.register_pass<ngraph::pass::LikeReplacement>();
    pass_manager.register_pass<runtime::gpu::pass::GPULayout>(this);
    pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(s_memory_pool_alignment);
    pass_manager.register_pass<runtime::gpu::pass::TensorMemoryReservation>(
        *allocator, m_tensor_memory_buffers);
    std::string common_function_string;
    auto femitter = bind(&ngraph::runtime::gpu::GPU_ExternalFunction::emit_op_as_function,
                         this,
                         placeholders::_1,
                         placeholders::_2);
    pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
        femitter, m_node_function_map, common_function_string);
    string dump_filename = file_util::path_join(s_output_dir, m_function_name + "_ops.txt");
    pass_manager.register_pass<ngraph::pass::DumpSorted>(dump_filename);
    */
    pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(64);
    pass_manager.run_passes(function);

    for (std::shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        m_trans_ctx->m_function_ordered_ops.emplace(current_function,
                                                    current_function->get_ordered_ops());
    }

    // Iterator through all functions
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        auto current_function = p.first;

        // Get all the output tensors
        set<string> output_names;
        for (std::shared_ptr<Node> op : current_function->get_results())
        {
            std::shared_ptr<ngraph::descriptor::Tensor> tv = op->get_output_tensor_ptr();
            output_names.insert(tv->get_name());
        }

        // Get all the constants
        set<descriptor::Tensor*> constants;
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                constants.insert(tv.get());
            }
        }

        std::cout << "Traslating model(function):\t" << current_function->get_name() << std::endl;

        //(todo) allocate temp memory pool
        // save_temp_mem_pool_allocation(current_function);

        // Save Param info: (type, index of input) => ((type*))(inputs[i])
        size_t arg_index = 0;
        for (shared_ptr<ngraph::op::Parameter> param : current_function->get_parameters())
        {
            for (size_t i = 0; i < param->get_output_size(); ++i)
            {
                shared_ptr<descriptor::Tensor> tv = param->get_output_tensor_ptr(i);
                const element::Type& et = tv->get_element_type();

                string type = et.c_type_string();
                stringstream ss;
                ss << "((" << type << "*)(inputs[" << arg_index << "]))";
                m_trans_ctx->m_variable_name_map[tv->get_name()] = ss.str();
                propagate_in_place_input(&param->get_outputs().at(i), ss.str());

                arg_index++;

                std::cout << "Input:\t" << et.c_type_string() << "\t" << arg_index << std::endl;
            }
        }

        //Save output node info
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();

            string type = tv->get_element_type().c_type_string();
            stringstream ss;
            ss << "((" << type << "*)(outputs[" << i << "]))";
            m_trans_ctx->m_variable_name_map[tv->get_name()] = ss.str();

            auto res = dynamic_pointer_cast<ngraph::op::Result>(op);
            //keep assigning different outputs to a result descriptor
            //op::Result emitter will check if in and out descriptors are the same
            //and skip a copy
            auto input_node = res->get_inputs().at(0).get_output().get_node();

            if (!input_node->is_constant() && !input_node->is_parameter())
            {
                shared_ptr<descriptor::Tensor> itv =
                    res->get_inputs().at(0).get_output().get_tensor_ptr();
                auto output_name = ss.str();
                m_trans_ctx->m_variable_name_map[itv->get_name()] = output_name;
                propagate_in_place_output(&(res->get_inputs().at(0).get_output()), output_name);
            }

            std::cout << "Output:\t" << tv->get_element_type().c_type_string() << std::endl;
        }

        // Translate the Node
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            vector<TensorWrapper> in;
            vector<string> node_input_names;
            vector<string> node_output_names;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                cout << "IN\t" << m_trans_ctx->m_variable_name_map[tv->get_name()] << endl;
                in.push_back(TensorWrapper(tv, m_trans_ctx->m_variable_name_map[tv->get_name()]));
                node_input_names.emplace_back(tv->get_name());
            }
            vector<TensorWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                cout << "OUT\t" << m_trans_ctx->m_variable_name_map[tv->get_name()] << endl;
                out.push_back(TensorWrapper(tv, m_trans_ctx->m_variable_name_map[tv->get_name()]));
                node_output_names.emplace_back(tv->get_name());
            }

            // Output debug info of node
            if (!node->is_parameter() && !node->is_constant())
            {
                std::cout << "Node:\t" << node->get_name() << "\t(";
                vector<string> parameter_nodes = node_input_names;
                parameter_nodes.insert(
                    parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
                std::cout << join(parameter_nodes);
                std::cout << ")\n";
            }

            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(node.get());
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                this->translate_node(node.get(), in, out);
            }
        }

        if (!IFunctionTranslatorPass::run_passes(this->m_passes, current_function))
            return false;
    }
    return true;
}

std::shared_ptr<ngraph::runtime::nnfusion::TranslationUnit>
    ngraph::runtime::nnfusion::FunctionTranslator::get_TranslationUnit()
{
    std::shared_ptr<TranslationUnit> tu(new TranslationUnit());
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        auto current_function = p.first;
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            tu->inter_ops->push_back(m_trans_ctx->m_node_inter_map[node.get()]);
        }
    }
    return tu;
}

bool ngraph::runtime::nnfusion::FunctionTranslator::translate_node(TRANS_ARGS)
{
    /*
    #define NGRAPH_OP(a, b) {type_index(typeid(b::a)), runtime::nnfusion::inter_op_##a##::translate}
        static const map<type_index, function<IntermediateOP(EMIT_ARGS)>> typeid_map{
    #include "ngraph/runtime/gpu/op/op_tbl.hpp"
        };
    #undef NGRAPH_OP
    */
    static const map<type_index, function<std::shared_ptr<IntermediateOP>(TRANS_ARGS)>> typeid_map{
        {type_index(typeid(ngraph::op::Reshape)),
         ngraph::runtime::nnfusion::intermediate::Reshape::translate},
        {type_index(typeid(ngraph::op::Parameter)),
         ngraph::runtime::nnfusion::intermediate::NoTrans::translate},
        {type_index(typeid(ngraph::op::Result)),
         ngraph::runtime::nnfusion::intermediate::Result::translate},
        {type_index(typeid(ngraph::op::Constant)),
         ngraph::runtime::nnfusion::intermediate::NoTrans::translate},
        {type_index(typeid(ngraph::op::Relu)),
         ngraph::runtime::nnfusion::intermediate::elementwise<ngraph::op::Relu>::translate},
        // {type_index(typeid(ngraph::op::Conv2D)), runtime::nnfusion::inter_op_conv2d::translate},
    };

    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        // throw unsupported_op("Unsupported op '" + node->description() + "'");
        cout << "Unsupported op '" + node->description() + "'" << endl;
        return false;
    }
    cout << "Translate op '" + node->description() + "'" << endl;
    m_trans_ctx->m_node_inter_map.emplace(node, it->second(node, args, out));
    return true;
}

void ngraph::runtime::nnfusion::FunctionTranslator::propagate_in_place_input(
    ngraph::descriptor::Output* output, std::string input_name)
{
    std::deque<ngraph::descriptor::Output*> stack;
    stack.push_front(output);

    while (stack.size() > 0)
    {
        ngraph::descriptor::Output* it = stack.front();
        stack.pop_front();
        for (auto input : it->get_inputs())
        {
            auto c_op = std::dynamic_pointer_cast<ngraph::op::Op>(input->get_node());
            if (!c_op || c_op->is_output())
            {
                continue;
            }

            if (auto op_annotations = c_op->get_op_annotations())
            {
                for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                {
                    if (oi_pair.input == input->get_index() && !oi_pair.destructive)
                    {
                        size_t output_index = oi_pair.output;
                        auto& output_tensor = c_op->get_outputs().at(output_index).get_tensor();

                        m_trans_ctx->m_variable_name_map[output_tensor.get_name()] = input_name;

                        NGRAPH_DEBUG << "GPU codegen: Forwarding " << input_name << " through "
                                     << output_tensor.get_name();
                        stack.push_back(&c_op->get_outputs().at(output_index));
                    }
                }
            }
        }
    }
}

void ngraph::runtime::nnfusion::FunctionTranslator::propagate_in_place_output(
    ngraph::descriptor::Output* res_src_output, std::string output_name)
{
    // we start with a particular output
    // which is an argument to a given op::Result
    size_t offset = res_src_output->get_tensor().get_pool_offset();
    auto it = res_src_output;

    bool propagate_further = false;
    do
    {
        propagate_further = false;
        auto arg = std::dynamic_pointer_cast<ngraph::op::Op>(it->get_node());
        if (!arg)
        {
            break;
        }
        if (auto op_annotations = arg->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                if (oi_pair.output == it->get_index())
                {
                    size_t input_index = oi_pair.input;
                    auto& input_tensor = arg->get_inputs().at(input_index).get_tensor();
                    auto tmp_node = arg->get_inputs().at(input_index).get_output().get_node();
                    if (input_tensor.get_pool_offset() == offset && !tmp_node->is_parameter() &&
                        !tmp_node->is_constant())
                    {
                        NGRAPH_DEBUG << "Reusing " << output_name << " for "
                                     << input_tensor.get_name();

                        m_trans_ctx->m_variable_name_map[input_tensor.get_name()] = output_name;

                        it = &arg->get_inputs().at(input_index).get_output();
                        propagate_further = true;
                    }
                }
            }
        }
    } while (propagate_further);
}
