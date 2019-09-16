// Microsoft (c) 2019, NNFusion Team
#include "elementwise_fused.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::kernels::cuda;

int ElementWiseFused::unique_func_id = 0;

ElementWiseFused::ElementWiseFused(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    for (auto kernel : ctx->kernels)
    {
        auto cuda_kernel = std::dynamic_pointer_cast<CudaElementwiseEmitter>(kernel);
        enforce_not_nullptr(cuda_kernel) << "kernel type:"
                                         << kernel->m_context->node->description();
        m_kernels.push_back(cuda_kernel);
    }
    enforce_not_nullptr(FuseContext());
}

std::shared_ptr<KernelContext> ElementWiseFused::FuseContext()
{
    std::shared_ptr<KernelContext> ctx = this->m_context;
    // output
    std::unordered_map<std::string, size_t> node_outputs;
    std::unordered_map<std::string, TensorWrapper> tensor_wrappers;

    for (auto kernel_emitter : m_kernels)
    {
        auto node = kernel_emitter->m_context->node;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            enforce_not_nullptr(tv);
            auto iter = node_outputs.find(tv->get_name());
            if (iter == node_outputs.end())
            {
                ctx->inputs.push_back(TensorWrapper(tv, tv->get_name()));
                ctx->input_names.push_back(tv->get_name());
            }
            else
            {
                enforce(iter->second > 0);
                node_outputs[tv->get_name()] = node_outputs[tv->get_name()] - 1;
            }
        }

        for (const descriptor::Output& output : node->get_outputs())
        {
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            enforce_not_nullptr(tv);
            enforce(node_outputs.find(tv->get_name()) == node_outputs.end());
            node_outputs[tv->get_name()] = node->get_outputs()[0].get_inputs().size();
            tensor_wrappers.insert(
                std::make_pair(tv->get_name(), TensorWrapper(tv, tv->get_name())));
        }
    }

    for (auto& iter : node_outputs)
    {
        if (iter.second > 0)
        {
            ctx->output_names.push_back(iter.first);
            auto tw = tensor_wrappers.find(iter.first);
            enforce(tw != tensor_wrappers.end());
            ctx->outputs.push_back(tw->second);
        }
    }

    for (auto& arg : ctx->inputs)
    {
        ctx->dtypes.push_back(arg.get_type());
    }

    for (auto& out : ctx->outputs)
    {
        ctx->dtypes.push_back(out.get_type());
    }

    return ctx;
}

LanguageUnit_p ElementWiseFused::emit_function_body()
{
    create_ptr(LanguageUnit, lu_, get_function_name());
    LanguageUnit& lu = *lu_;

    std::unordered_map<std::string, std::string> args, local_tensors;
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        args[tensor.get_name()] = "input" + std::to_string(i) + "[tid]";
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        args[tensor.get_name()] = "output" + std::to_string(i) + "[tid]";
    }

    size_t temp_tensor_id = 0;

    uint32_t nthreads =
        static_cast<uint32_t>(ngraph::shape_size(m_context->outputs[0].get_shape()));

    int grids, blocks, bound;
    compute_best_config(grids, blocks, bound);

    if (grids == 1)
    {
        lu << "int tid = threadIdx.x;\n";
    }
    else
    {
        lu << "int tid = blockIdx.x * " << std::to_string(blocks) << " + threadIdx.x;\n";
    }
    if (bound)
    {
        lu << "if (tid >= " << bound << ") return;\n";
    }

    for (auto kernel_emitter : m_kernels)
    {
        auto op_kernel = kernel_emitter->get_op_kernel();
        if (op_kernel.second != nullptr)
        {
            lu.require(op_kernel.second);
        }

        auto out_tw = kernel_emitter->m_context->outputs[0];
        if (args.count(out_tw.get_name()) > 0)
        {
            lu << args[out_tw.get_name()] << " = ";
        }
        else
        {
            if (local_tensors.count(out_tw.get_name()) == 0)
            {
                local_tensors[out_tw.get_name()] = "temp" + std::to_string(temp_tensor_id++);
            }
            lu << out_tw.get_type() << " " << local_tensors[out_tw.get_name()] << " = ";
        }

        std::vector<std::string> input_args;
        for (int i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
        {
            auto& in_tw = kernel_emitter->m_context->inputs[i];
            if (args.count(in_tw.get_name()) > 0)
            {
                input_args.push_back(args[in_tw.get_name()]);
            }
            else
            {
                enforce(local_tensors.count(in_tw.get_name()) > 0);
                input_args.push_back(local_tensors[in_tw.get_name()]);
            }
        }

        lu << op_kernel.first << "(" << join(input_args, ", ") << ");\n";
    }

    return lu_;
}

LanguageUnit_p ElementWiseFused::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::stdio);

    return _lu;
}

LanguageUnit_p ElementWiseFused::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit("function_name"));
    auto& lu = *_lu;

    std::vector<std::string> names;
    for (auto kernel : m_kernels)
    {
        names.push_back(kernel->m_context->node->description());
    }

    lu << "FusedKernel_" << join(m_context->dtypes, "_") << "_" << m_kernel_type << "_"
       << join(names, "_") << "_" << ElementWiseFused::unique_func_id++; //<< custom_tag;

    return _lu;
}

LanguageUnit_p ElementWiseFused::emit_comments()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_comments"));
    auto& lu = *_lu;
    lu << "// Node name:\t Elementwise Kernel Fusion"
       << "\n";
    //lu << "// Description:\t" << m_context->node->description() << "\n";
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
    return _lu;
}

void ElementWiseFused::set_launch_config()
{
    int grids, blocks, bound;
    compute_best_config(grids, blocks, bound);

    m_gridDim = dim3(grids, 1, 1);
    m_blockDim = dim3(blocks, 1, 1);
}

void ElementWiseFused::compute_best_config(int& grids, int& blocks, int& bound)
{
    uint32_t num_ele = static_cast<uint32_t>(ngraph::shape_size(m_context->outputs[0].get_shape()));
    for (int i = 1024; i >= 64; i >>= 1)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    for (int i = 1024; i >= 32; i--)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    if (num_ele < 32)
        grids = 1, blocks = num_ele, bound = 0;
    else
        grids = (num_ele + 255) / 256, blocks = 256, bound = 1;
}

REGISTER_KERNEL_EMITTER("ElementWiseFused",                                           // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::ElementWiseFused)