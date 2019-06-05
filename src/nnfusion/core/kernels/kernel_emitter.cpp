// Microsoft (c) 2019, NNFusion Team

#include "kernel_emitter.hpp"
#include <string>

using namespace nnfusion;
using namespace nnfusion::kernels;

KernelContext::KernelContext(shared_ptr<Node> node)
    : node(node)
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
    , m_dependency(nullptr)
    , m_function_body(nullptr)
    , m_function_call(nullptr)
    , m_test(nullptr)
    , m_test_call(nullptr)
    , m_source(nullptr)
{
}

KernelEmitter::KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type)
    : m_context(ctx)
    , m_is_emitted(false)
    , m_dependency(nullptr)
    , m_function_body(nullptr)
    , m_function_call(nullptr)
    , m_test(nullptr)
    , m_test_call(nullptr)
    , m_source(nullptr)
    , m_kernel_type(kernel_type)
{
}

string KernelEmitter::get_function_name()
{
    if (m_function_name.empty())
    {
        std::stringstream func_name;
        func_name << m_context->node->description() << "_" << join(m_context->dtypes, "_")
                  << m_kernel_type << "_" << custom_tag;
        m_function_name = func_name.str();
    }

    return m_function_name;
}

LanguageUnit_p KernelEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_call"));
    auto& lu = *_lu;
    lu << get_function_name() << "(" << join(m_context->input_names, ", ") << ","
       << join(m_context->output_names, ", ") << ");\n";

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
    enforce_not_nullptr(this->m_dependency = emit_dependency());
    if (this->m_function_name.empty())
    {
        get_function_name();
    }
    if (kernel_definitions.find(this->m_function_name) != kernel_definitions.end())
    {
        enforce_not_nullptr(this->m_function_body = kernel_definitions[this->m_function_name]);
    }
    else
    {
        enforce_not_nullptr(this->m_function_body = emit_function_body());
    }
    enforce_not_nullptr(this->m_function_call = emit_function_call());
    //enforce_not_nullptr(this->m_test = emit_test());

    // Pass other to dep_unit
    for (auto& it : m_function_call->local_symbol)
        m_dependency->require(it.second);
    for (auto& it : m_function_body->local_symbol)
        m_dependency->require(it.second);
    for (auto& it : m_test->local_symbol)
        m_dependency->require(it.second);
    m_function_call->clean_require();
    m_function_body->clean_require();
    m_test->clean_require();

    // orgnize dep
    this->m_function_body->require(this->m_dependency);
    enforce(this->m_function_call->require(this->m_function_body));
    enforce(this->m_test->require(this->m_function_body));

    m_is_emitted = true;
    return this->m_function_call;
}