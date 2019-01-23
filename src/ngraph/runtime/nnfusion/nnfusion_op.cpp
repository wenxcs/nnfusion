// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

unordered_map<string, shared_ptr<LanguageUnit>> CodeGenOP::definition_pool;

IntermediateOP::IntermediateOP()
    : m_name("Null")
    , isTranslated(false)
    , node(nullptr)
{
}

IntermediateOP::IntermediateOP(shared_ptr<Node> node)
    : IntermediateOP()
{
    vector<TensorWrapper> in;
    vector<string> node_input_names;
    vector<string> node_output_names;
    for (const descriptor::Input& input : node->get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        assert_nullptr(tv);
        in.push_back(TensorWrapper(tv, tv->get_name()));
        node_input_names.emplace_back(tv->get_name());
    }
    vector<TensorWrapper> out;
    for (const descriptor::Output& output : node->get_outputs())
    {
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        assert_nullptr(tv);
        out.push_back(TensorWrapper(tv, tv->get_name()));
        node_output_names.emplace_back(tv->get_name());
    }

    // Output debug info of node
    if (!node->is_parameter() && !node->is_constant())
    {
        NGRAPH_DEBUG << "Node:\t" << node->get_name() << "\t(";
        vector<string> parameter_nodes = node_input_names;
        parameter_nodes.insert(
            parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
        NGRAPH_DEBUG << join(parameter_nodes);
        NGRAPH_DEBUG << ")\n";
    }
    this->node = node;
    this->args = in;
    this->arg_names = node_input_names;
    this->out = out;
    this->out_names = node_output_names;
}

CodeGenOP::CodeGenOP()
    : definition_unit(nullptr)
    , inter_op(nullptr)
    , call_unit(nullptr)
    , test_unit(nullptr)
    , dep_unit(nullptr)
    , source_unit(nullptr)
    , isCodeGened(false)
{
}

CodeGenOP::CodeGenOP(shared_ptr<IntermediateOP> inter_op)
    : CodeGenOP()
{
    assert_nullptr(this->inter_op = inter_op);
}

shared_ptr<LanguageUnit> CodeGenOP::codegen_test()
{
    shared_ptr<LanguageUnit> _lu(new LanguageUnit(codegen_test_name()));
    auto& writer = *_lu;

    // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
    //{
    //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
    //}

    assert_nullptr(inter_op);

    auto& arg = inter_op->args;
    auto& out = inter_op->out;

    writer << "extern \"C\" int " << _lu->get_symbol() << "(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i].get_type() << "* " << arg[i].get_name() << "_host, ";
    }
    if (!arg.empty())
    {
        writer << arg.back().get_type() << "* " << arg.back().get_name();
        if (!out.empty())
            writer << "_host, ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i].get_type() << "* " << out[i].get_name() << "_host, ";
    }
    if (!out.empty())
    {
        writer << out.back().get_type() << "* " << out.back().get_name() << "_host";
    }
    writer << ")\n";

    writer.block_begin();
    {
        for (size_t i = 0; i < arg.size(); i++)
        {
            auto& tensor = arg[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";

            writer << "cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name() << "_host, "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyHostToDevice);\n";
        }
        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";
        }

        assert_nullptr(this->call_unit);
        writer << this->call_unit->get_code();

        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << "cudaMemcpy(" << tensor.get_name() << "_host, " << tensor.get_name() << ", "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyDeviceToHost);\n";
        }

        writer << "return 0;\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" int " << _lu->get_symbol() << "_simple(void** args)";

    writer.block_begin();
    {
        writer << "return " << _lu->get_symbol() << "(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)"
                   << "args[" << i << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)"
                   << "args[" << i << "]";
        }
        writer << ");\n";
    }
    writer.block_end();
    return _lu;
}

shared_ptr<LanguageUnit> CodeGenOP::codegen_source()
{
    assert_bool(isCodeGened == false) << "Code only generated once.";
    assert_nullptr(this->dep_unit = codegen_dependency());
    if (definition_pool.find(codegen_function_name()) != definition_pool.end())
    {
        assert_nullptr(this->definition_unit = definition_pool[codegen_function_name()]);
    }
    else
    {
        assert_nullptr(this->definition_unit = codegen_function_definition());
        this->definition_unit->require(this->dep_unit);
    }
    assert_nullptr(this->call_unit = codegen_function_call());
    assert_bool(this->call_unit->require(this->definition_unit));

    assert_nullptr(this->test_unit = codegen_test());
    assert_bool(this->test_unit->require(this->definition_unit));

    isCodeGened = true;
    return this->call_unit;
}