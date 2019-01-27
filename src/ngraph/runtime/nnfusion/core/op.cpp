// Microsoft (c) 2019, Wenxiang
#include "op.hpp"

using namespace nnfusion::ir;

unordered_map<string, LanguageUnit_p> ir::Function::definition_pool;

Operator::Operator()
    : m_name("Null")
    , isTranslated(false)
    , node(nullptr)
{
}

Operator::Operator(shared_ptr<Node> node)
    : Operator()
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

ir::Function::Function()
    : definition_unit(nullptr)
    , op(nullptr)
    , call_unit(nullptr)
    , test_unit(nullptr)
    , dep_unit(nullptr)
    , source_unit(nullptr)
    , isCodeGened(false)
{
}

ir::Function::Function(shared_ptr<Operator> op)
    : Function()
{
    assert_nullptr(this->op = op);
}

LanguageUnit_p ir::Function::codegen_source()
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