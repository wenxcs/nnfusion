// Microsoft (c) 2019, Wenxiang
#include "op.h"

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
        enforce_not_nullptr(tv);
        in.push_back(TensorWrapper(tv, tv->get_name()));
        node_input_names.emplace_back(tv->get_name());
    }
    vector<TensorWrapper> out;
    for (const descriptor::Output& output : node->get_outputs())
    {
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        enforce_not_nullptr(tv);
        out.push_back(TensorWrapper(tv, tv->get_name()));
        node_output_names.emplace_back(tv->get_name());
    }

    // Output debug info of node
    if (!node->is_parameter() && !node->is_constant())
    {
        LOG_INFO << "Node:\t" << node->get_name() << "\t(";
        vector<string> parameter_nodes = node_input_names;
        parameter_nodes.insert(
            parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
        LOG_INFO << join(parameter_nodes);
        LOG_INFO << ")\n";
    }

    for (auto& arg : in)
    {
        this->dtypes.push_back(arg.get_type());
    }

    for (auto& ou : out)
    {
        this->dtypes.push_back(ou.get_type());
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
    enforce_not_nullptr(this->op = op);
}

LanguageUnit_p ir::Function::codegen_source()
{
    enforce(isCodeGened == false) << "Code only generated once.";
    enforce_not_nullptr(this->dep_unit = codegen_dependency());
    if (definition_pool.find(codegen_function_name()) != definition_pool.end())
    {
        enforce_not_nullptr(this->definition_unit = definition_pool[codegen_function_name()]);
    }
    else
    {
        enforce_not_nullptr(this->definition_unit = codegen_function_definition());
    }
    enforce_not_nullptr(this->call_unit = codegen_function_call());
    enforce_not_nullptr(this->test_unit = codegen_test());
    // Pass other to dep_unit
    for (auto& it : call_unit->local_symbol)
        dep_unit->require(it.second);
    for (auto& it : definition_unit->local_symbol)
        dep_unit->require(it.second);
    for (auto& it : test_unit->local_symbol)
        dep_unit->require(it.second);
    call_unit->clean_require();
    definition_unit->clean_require();
    test_unit->clean_require();

    // orgaize dep
    this->definition_unit->require(this->dep_unit);
    enforce(this->call_unit->require(this->definition_unit));
    enforce(this->test_unit->require(this->definition_unit));

    isCodeGened = true;
    return this->call_unit;
}