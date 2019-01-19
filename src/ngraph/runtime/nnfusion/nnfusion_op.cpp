// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

unordered_map<string, shared_ptr<CodeWriter>> CodeGenOP::definition_pool;

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
    this->out = out;
}

CodeGenOP::CodeGenOP()
    : definition_writer(nullptr)
    , inter_op(nullptr)
    , call_writer(nullptr)
    , _dep(new CodeGenOP::CodeGenOPDep())
    , isCodeGened(false)
{
}

CodeGenOP::CodeGenOP(shared_ptr<IntermediateOP> inter_op)
    : CodeGenOP()
{
    assert_nullptr(this->inter_op = inter_op);
}

shared_ptr<CodeWriter> CodeGenOP::codegen_source()
{
    if (definition_pool.find(codegen_function_name()) != definition_pool.end())
    {
        assert_nullptr(this->definition_writer = definition_pool[codegen_function_name()]);
    }
    else
    {
        assert_nullptr(this->definition_writer = codegen_function_definition());
    }
    assert_nullptr(this->call_writer = codegen_function_call());
    assert_nullptr(this->test_writer = codegen_test());

    shared_ptr<CodeWriter> codewriter(new CodeWriter());
    auto& cw = *codewriter;
    assert_nullptr(this->source_writer = codewriter);
    assert_nullptr(this->dep_writer = codegen_dependency());

    cw << this->dep_writer->get_code() << "\n";
    cw << this->definition_writer->get_code() << "\n";
    //codewriter<<this->codegen_function_call.get_code()<<"\n";

    cw << "int main()";
    cw.block_begin();
    cw << this->test_writer->get_code() << "\n";
    cw << "return 0;\n";
    cw.block_end();

    // Save the function
    string filename = codegen_source_name() + "_test.cu";
    ofstream out(filename);
    out << cw.get_code();
    out.close();

    return codewriter;
}