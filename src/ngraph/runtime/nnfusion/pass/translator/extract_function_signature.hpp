// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace translator
            {
                class ExtractFunctionSignature : public IFunctionTranslatorPass
                {
                public:
                    bool extract_result(std::shared_ptr<TranslationUnit> tu,
                                        std::shared_ptr<ngraph::Function> function)
                    {
                        for (std::shared_ptr<Node> op : function->get_results())
                        {
                            std::shared_ptr<ngraph::descriptor::Tensor> tv =
                                op->get_output_tensor_ptr();
                            assert_nullptr(tv);
                            tu->output_names->insert(tv->get_name());

                            NGRAPH_DEBUG << "Result Tensor: " << tv->get_name();
                        }
                        return true;
                    }

                    bool extract_constants(std::shared_ptr<FunctionTranslatorContext> ctx,
                                           std::shared_ptr<TranslationUnit> tu,
                                           std::shared_ptr<ngraph::Function> function)
                    {
                        for (shared_ptr<Node> node : ctx->m_function_ordered_ops.at(function))
                        {
                            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
                            {
                                shared_ptr<descriptor::Tensor> tv =
                                    node->get_outputs()[0].get_tensor_ptr();
                                assert_nullptr(tv);
                                tu->constants->insert(tv);

                                NGRAPH_DEBUG << "Constant Tensor: " << tv->get_name();
                            }
                        }
                        return true;
                    }

                    void propagate_in_place_input(std::shared_ptr<FunctionTranslatorContext> ctx, ngraph::descriptor::Output* output,
                                                  std::string input_name)
                    {
                        std::deque<ngraph::descriptor::Output*> stack;
                        stack.push_front(output);

                        while (stack.size() > 0)
                        {
                            ngraph::descriptor::Output* it = stack.front();
                            stack.pop_front();
                            for (auto input : it->get_inputs())
                            {
                                auto c_op =
                                    std::dynamic_pointer_cast<ngraph::op::Op>(input->get_node());
                                if (!c_op || c_op->is_output())
                                {
                                    continue;
                                }

                                if (auto op_annotations = c_op->get_op_annotations())
                                {
                                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                                    {
                                        if (oi_pair.input == input->get_index() &&
                                            !oi_pair.destructive)
                                        {
                                            size_t output_index = oi_pair.output;
                                            auto& output_tensor =
                                                c_op->get_outputs().at(output_index).get_tensor();

                                            ctx
                                                ->m_variable_name_map[output_tensor.get_name()] =
                                                input_name;

                                            NGRAPH_DEBUG << "GPU codegen: Forwarding " << input_name
                                                         << " through " << output_tensor.get_name();
                                            stack.push_back(&c_op->get_outputs().at(output_index));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    void propagate_in_place_output(std::shared_ptr<FunctionTranslatorContext> ctx, ngraph::descriptor::Output* res_src_output,
                                                   std::string output_name)
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
                                        auto& input_tensor =
                                            arg->get_inputs().at(input_index).get_tensor();
                                        auto tmp_node = arg->get_inputs()
                                                            .at(input_index)
                                                            .get_output()
                                                            .get_node();
                                        if (input_tensor.get_pool_offset() == offset &&
                                            !tmp_node->is_parameter() && !tmp_node->is_constant())
                                        {
                                            NGRAPH_DEBUG << "Reusing " << output_name << " for "
                                                         << input_tensor.get_name();

                                            ctx
                                                ->m_variable_name_map[input_tensor.get_name()] =
                                                output_name;

                                            it = &arg->get_inputs().at(input_index).get_output();
                                            propagate_further = true;
                                        }
                                    }
                                }
                            }
                        } while (propagate_further);
                    }

                    bool extract_args(std::shared_ptr<FunctionTranslatorContext> ctx,
                                      std::shared_ptr<TranslationUnit> tu,
                                      std::shared_ptr<ngraph::Function> function)
                    {
                        size_t arg_index = 0;
                        for (shared_ptr<ngraph::op::Parameter> param : function->get_parameters())
                        {
                            for (size_t i = 0; i < param->get_output_size(); ++i)
                            {
                                shared_ptr<descriptor::Tensor> tv = param->get_output_tensor_ptr(i);
                                assert_nullptr(tv);
                                const element::Type& et = tv->get_element_type();

                                string type = et.c_type_string();
                                stringstream ss;
                                ss << "((" << type << "*)(inputs[" << arg_index << "]))";
                                ctx->m_variable_name_map[tv->get_name()] = ss.str();
                                propagate_in_place_input(ctx, &param->get_outputs().at(i), ss.str());

                                arg_index++;

                                NGRAPH_DEBUG << "Param Tensor:\t" << tv->get_name()
                                             << "\twith id: " << ss.str();
                            }
                        }
                        return true;
                    }

                    bool extract_output(std::shared_ptr<FunctionTranslatorContext> ctx,
                                        std::shared_ptr<TranslationUnit> tu,
                                        std::shared_ptr<ngraph::Function> function)
                    {
                        for (size_t i = 0; i < function->get_output_size(); ++i)
                        {
                            shared_ptr<Node> op = function->get_output_op(i);
                            assert_nullptr(op);
                            shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
                            assert_nullptr(tv);

                            string type = tv->get_element_type().c_type_string();
                            stringstream ss;
                            ss << "((" << type << "*)(outputs[" << i << "]))";
                            ctx->m_variable_name_map[tv->get_name()] = ss.str();

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
                                ctx->m_variable_name_map[itv->get_name()] = output_name;
                                propagate_in_place_output(ctx, &(res->get_inputs().at(0).get_output()),
                                                          output_name);
                                NGRAPH_DEBUG << "Output Tensor:\t" << itv->get_name()
                                             << "\t with id:" << output_name;
                            }
                        }
                        return true;
                    }

                    bool run(std::shared_ptr<FunctionTranslatorContext> ctx,
                             std::shared_ptr<TranslationUnit> tu,
                             std::shared_ptr<ngraph::Function> function) override
                    {
                        assert_bool(extract_result(tu, function));
                        assert_bool(extract_constants(ctx, tu, function));
                        assert_bool(extract_args(ctx, tu, function));
                        assert_bool(extract_output(ctx, tu, function));
                        return true;
                    }
                };
            }
        }
    }
}