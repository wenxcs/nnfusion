// Microsoft (c) 2019, NNFusion Team

#include "kernel_emitter.hpp"
#include <string>

using namespace nnfusion;
using namespace nnfusion::kernels;

KernelContext::KernelContext(shared_ptr<Node> node)
    : node(node)
    , gpu_num_sm(20)
{
    // extract input tensors
    for (const descriptor::Input& input : node->get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        enforce_not_nullptr(tv);
        inputs.push_back(TensorWrapper(tv, tv->get_name()));
        input_names.emplace_back(tv->get_name());
    }

    // extract output tensors
    for (const descriptor::Output& output : node->get_outputs())
    {
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        enforce_not_nullptr(tv);
        outputs.push_back(TensorWrapper(tv, tv->get_name()));
        output_names.emplace_back(tv->get_name());
    }

    for (auto& arg : inputs)
    {
        this->dtypes.push_back(arg.get_type());
    }

    for (auto& out : outputs)
    {
        this->dtypes.push_back(out.get_type());
    }
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx)
    : m_context(ctx)
    , m_is_emitted(false)
    , dep_unit(nullptr)
    , body_unit(nullptr)
    , call_unit(nullptr)
    , signature_unit(nullptr)
    , test_unit(nullptr)
    , test_call_unit(nullptr)
    , source_unit(nullptr)
{
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type)
    : m_context(ctx)
    , m_is_emitted(false)
    , dep_unit(nullptr)
    , body_unit(nullptr)
    , call_unit(nullptr)
    , signature_unit(nullptr)
    , test_unit(nullptr)
    , test_call_unit(nullptr)
    , source_unit(nullptr)
    , m_kernel_type(kernel_type)
{
}

string KernelEmitter::get_function_name()
{
    if (m_function_name.empty())
    {
        std::stringstream func_name;
        func_name << m_context->node->description() << "_" << join(m_context->dtypes, "_") << "_"
                  << m_kernel_type << "_" << custom_tag;
        m_function_name = func_name.str();
    }

    return m_function_name;
}

LanguageUnit_p KernelEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    lu << get_function_name() << "(" << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p KernelEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i].get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i].get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    lu << "void " << get_function_name() << "(" << join(params, ", ") << ")";
    return _lu;
}

string KernelEmitter::emit_comments()
{
    LanguageUnit lu;
    lu << "// Node name:\t" << m_context->node->get_name() << "\n";
    lu << "// Description:\t" << m_context->node->description() << "\n";
    lu << "// Input:\n";
    for (auto& in : m_context->inputs)
    {
        lu << "//\t- name: " << in.get_name();
        lu << "\ttype: " << in.get_type();
        lu << "\tshape: " << in.get_shape();
        lu << "\n";
    }

    lu << "// Output:\n";
    for (auto& out : m_context->outputs)
    {
        lu << "//\t- name: " << out.get_name();
        lu << "\ttype: " << out.get_type();
        lu << "\tshape: " << out.get_shape();
        lu << "\n";
    }
    return lu.get_code();
}

LanguageUnit_p KernelEmitter::emit_source()
{
    enforce(m_is_emitted == false) << "Code only generated once.";

    if (this->m_function_name.empty())
    {
        get_function_name();
    }
    if (kernel_definitions.find(this->m_function_name) != kernel_definitions.end())
    {
        enforce_not_nullptr(this->body_unit = kernel_definitions[this->m_function_name]);
    }
    else
    {
        this->body_unit = this->emit_function_body();
        if (!this->body_unit)
        {
            return nullptr;
        }
    }
    enforce_not_nullptr(this->dep_unit = emit_dependency());
    enforce_not_nullptr(this->call_unit = emit_function_call());
    enforce_not_nullptr(this->signature_unit = emit_function_signature());
    // Pass other to dep_unit
    for (auto& it : call_unit->local_symbol)
        dep_unit->require(it.second);
    for (auto& it : body_unit->local_symbol)
        dep_unit->require(it.second);
    call_unit->clean_require();
    body_unit->clean_require();

    // orgnize dep
    this->body_unit->require(this->dep_unit);
    enforce(this->call_unit->require(this->body_unit));

    m_is_emitted = true;
    return this->call_unit;
}
