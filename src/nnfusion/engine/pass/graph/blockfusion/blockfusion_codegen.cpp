// Microsoft (c) 2019, NNFusion Team

#include "blockfusion_codegen.hpp"
#include "nnfusion/core/kernels/common_langunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/engine/async_manager.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels;

using namespace std;

int BlockFusionCudaCodegen::unique_func_id = 0;

BlockFusionCudaCodegen::BlockFusionCudaCodegen(shared_ptr<KernelContext> ctx,
                                               const BlockExecutorProgram& _block_executor_program)
    : CudaEmitter(ctx)
    , block_executor_program(_block_executor_program)
{
    NNFUSION_CHECK_NOT_NULLPTR(FuseContext());

    // control whether dedupe block_kernels
    this->is_dedupe_block_kernels = true;

    deduped_kernel_id_map.clear();
    for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size(); kernel_id++)
    {
        deduped_kernel_id_map[kernel_id] = kernel_id;
    }

    // dedupe block_kernels
    if (this->is_dedupe_block_kernels == true)
    {
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            std::string block_kernel_body = block_executor_program.block_kernels[kernel_id]
                                                ->emit_device_function_body()
                                                ->get_code();
            for (int deduped_kernel_id = 0; deduped_kernel_id < kernel_id; deduped_kernel_id++)
            {
                std::string deduped_kernel_body =
                    block_executor_program.block_kernels[deduped_kernel_id]
                        ->emit_device_function_body()
                        ->get_code();
                if (block_kernel_body == deduped_kernel_body)
                {
                    deduped_kernel_id_map[kernel_id] = deduped_kernel_id;
                    break;
                }
            }
        }
    }
}

std::shared_ptr<KernelContext> BlockFusionCudaCodegen::FuseContext()
{
    std::shared_ptr<KernelContext> ctx = this->m_context;

    ctx->kernels.clear();
    for (auto block_kernel : block_executor_program.block_kernels)
    {
        auto kernel = std::dynamic_pointer_cast<KernelEmitter>(block_kernel);
        NNFUSION_CHECK_NOT_NULLPTR(kernel);
        ctx->kernels.push_back(kernel);
    }

    // output
    // std::unordered_map<std::string, size_t> node_outputs;
    // std::unordered_map<std::string, shared_ptr<nnfusion::descriptor::Tensor>> tensors;

    // // analyze input and output
    // for (auto kernel_emitter : ctx->kernels)
    // {
    //     auto gnode = kernel_emitter->m_context->gnode;
    //     for (size_t i = 0; i < gnode->get_input_size(); i++)
    //     {
    //         auto tv = gnode->get_input_tensor_ptr(i);
    //         NNFUSION_CHECK_NOT_NULLPTR(tv);

    //         ctx->inputs.push_back(tv);
    //         ctx->input_names.push_back(tv->get_name());

    //         // auto iter = node_outputs.find(tv->get_name());
    //         // if (iter == node_outputs.end())
    //         // {
    //         //     ctx->inputs.push_back(tv);
    //         //     ctx->input_names.push_back(tv->get_name());
    //         // }
    //         // else
    //         // {
    //         //     CHECK(iter->second > 0);
    //         //     node_outputs[tv->get_name()] = node_outputs[tv->get_name()] - 1;
    //         // }
    //     }

    //     for (size_t i = 0; i < gnode->get_output_size(); i++)
    //     {
    //         auto tv = gnode->get_output_tensor_ptr(i);
    //         NNFUSION_CHECK_NOT_NULLPTR(tv);

    //         ctx->outputs.push_back(tv);
    //         ctx->output_names.push_back(tv->get_name());

    //         // CHECK(node_outputs.find(tv->get_name()) == node_outputs.end());
    //         // node_outputs[tv->get_name()] = gnode->get_output_users(0).size();
    //         // tensors.insert(std::make_pair(tv->get_name(), tv));
    //     }
    // }

    // // for (auto& iter : node_outputs)
    // // {
    // //     if (iter.second > 0)
    // //     {
    // //         ctx->output_names.push_back(iter.first);
    // //         auto tw = tensors.find(iter.first);
    // //         CHECK(tw != tensors.end());
    // //         ctx->outputs.push_back(tw->second);
    // //     }
    // // }

    std::unordered_map<std::string, size_t> node_inputs;
    std::unordered_map<std::string, size_t> node_outputs;
    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        for (size_t i = 0; i < gnode->get_output_size(); i++)
        {
            auto tv = gnode->get_output_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);

            if (node_outputs.find(tv->get_name()) == node_outputs.end())
            {
                node_outputs[tv->get_name()] = 1;
                ctx->outputs.push_back(tv);
                ctx->output_names.push_back(tv->get_name());
            }
            else
            {
                node_outputs[tv->get_name()]++;
            }
        }
    }
    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        for (size_t i = 0; i < gnode->get_input_size(); i++)
        {
            auto tv = gnode->get_input_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);

            if (node_outputs.find(tv->get_name()) == node_outputs.end())
            {
                if (node_inputs.find(tv->get_name()) == node_inputs.end())
                {
                    node_inputs[tv->get_name()] = 1;
                    ctx->inputs.push_back(tv);
                    ctx->input_names.push_back(tv->get_name());
                }
                else
                {
                    node_inputs[tv->get_name()]++;
                }
            }
        }
    }

    for (auto arg : ctx->inputs)
    {
        ctx->dtypes.push_back(arg->get_element_type().c_type_string());
    }

    for (auto out : ctx->outputs)
    {
        ctx->dtypes.push_back(out->get_element_type().c_type_string());
    }

    // if be_program has group_sync instructions (step_to, wait_for), set is_group_sync=true
    this->is_group_sync = false;
    for (auto be : block_executor_program.block_executor_instructions)
    {
        for (auto be_instruction : be)
        {
            if ((nullptr !=
                 std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be_instruction)) ||
                (nullptr !=
                 std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(be_instruction)))
            {
                this->is_group_sync = true;
                break;
            }
        }
        if (this->is_group_sync)
        {
            break;
        }
    }

    // allocate be_state_buffer for group_sync
    if (this->is_group_sync)
    {
        std::shared_ptr<nnfusion::descriptor::Tensor> be_state_buffer(
            new nnfusion::descriptor::Tensor(
                nnfusion::element::i32,
                nnfusion::PartialShape({block_executor_program.num_bes}),
                "BlockFusionKernel_" + std::to_string(BlockFusionCudaCodegen::unique_func_id) +
                    "_be_state_buffer",
                nnfusion::NNFusion_DeviceType::CUDA_GPU));

        ctx->tensors.push_back(be_state_buffer);

        ctx->dtypes.push_back(be_state_buffer->get_element_type().c_type_string());

        // ctx->inputs.push_back(be_state_buffer);
        // ctx->input_names.push_back(be_state_buffer->get_name());
    }

    return ctx;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_step_to_device_function()
{
    LanguageUnit_p _lu(new LanguageUnit("declaration::BlockFusion_step_to_device_function"));
    auto& lu = *_lu;

    lu << "__device__ __forceinline__ void BlockFusion_step_to_device_function(int* "
          "be_state_buffer, int be_id, int step_id)\n";
    lu.block_begin();
    lu << "__syncthreads();\n";
    lu << "if (threadIdx.x == 0)\n";
    lu.block_begin();
    lu << "be_state_buffer[be_id] = step_id;\n";
    lu.block_end();
    // lu << "__threadfence();\n";
    // lu << "__syncthreads();\n";
    lu.block_end();

    return _lu;
}

// LanguageUnit_p BlockFusionCudaCodegen::emit_wait_for_device_function()
// {
//     LanguageUnit_p _lu(new LanguageUnit("declaration::BlockFusion_wait_for_device_function"));
//     auto& lu = *_lu;

//     return _lu;
// }

LanguageUnit_p BlockFusionCudaCodegen::emit_block_kernel_functions()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_block_kernel_function"));
    LanguageUnit& lu = *_lu;

    lu << "\n";

    if (this->is_dedupe_block_kernels == true)
    {
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            if (deduped_kernel_id_map[kernel_id] == kernel_id)
            {
                auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
                NNFUSION_CHECK_NOT_NULLPTR(kernel_emitter);
                lu << kernel_emitter->emit_block_kernel()->get_code();
            }
        }
    }
    else
    {
        for (auto kernel_emitter : block_executor_program.block_kernels)
        {
            NNFUSION_CHECK_NOT_NULLPTR(kernel_emitter);
            lu << kernel_emitter->emit_block_kernel()->get_code();
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_execute_block(
    std::shared_ptr<BlockExecutorInstructionExecuteBlock> be_ins_execute_block)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_execute_block);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    auto kernel_id = be_ins_execute_block->kernel_id;
    auto deduped_kernel_id = kernel_id;
    if (this->is_dedupe_block_kernels == true)
    {
        deduped_kernel_id = deduped_kernel_id_map[kernel_id];
    }
    auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
    auto deduped_kernel_emitter = block_executor_program.block_kernels[deduped_kernel_id];

    std::vector<std::string> params;
    for (size_t i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
    {
        std::stringstream ss;
        ss << all_args[kernel_emitter->m_context->inputs[i]->get_name()];
        params.push_back(ss.str());
    }
    for (size_t i = 0; i < kernel_emitter->m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << all_args[kernel_emitter->m_context->outputs[i]->get_name()];
        params.push_back(ss.str());
    }
    params.push_back("threadIdx.x");
    params.push_back(std::to_string(be_ins_execute_block->kernel_block_id));

    if (this->is_shared_buffer == false)
    {
        params.push_back("NULL");
    }
    else
    {
        params.push_back("shared_buffer");
    }

    lu << deduped_kernel_emitter->get_function_name() << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_step_to(
    std::shared_ptr<BlockExecutorInstructionStepTo> be_ins_step_to)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_step_to);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    lu << "BlockFusion_step_to_device_function(be_state_buffer, "
       << std::to_string(be_ins_step_to->be_id) << ", " << std::to_string(be_ins_step_to->step_id)
       << ");\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor_instruction_wait_for(
    std::shared_ptr<BlockExecutorInstructionWaitFor> be_ins_wait_for)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_ins_wait_for);

    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    lu.block_begin();
    // lu << "__syncthreads();\n";
    lu << "if (threadIdx.x == 0)\n";
    lu.block_begin();
    for (auto be_predecessor : be_ins_wait_for->bes_predecessor)
    {
        lu << "while (be_state_buffer[" << be_predecessor << "] < " << be_ins_wait_for->step_id
           << ");\n";
    }
    lu.block_end();
    lu << "__syncthreads();\n";
    lu.block_end();

    return _lu;
}

LanguageUnit_p
    BlockFusionCudaCodegen::emit_block_executor_instruction(BEInstruction_p be_instruction)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_instruction);

    LanguageUnit_p _lu(new LanguageUnit("be_instruction"));
    LanguageUnit& lu = *_lu;

    if (auto ins_execute_block =
            std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction))
    {
        lu << emit_block_executor_instruction_execute_block(ins_execute_block)->get_code();
    }
    else if (auto ins_step_to =
                 std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(be_instruction))
    {
        lu << emit_block_executor_instruction_step_to(ins_step_to)->get_code();
    }
    else if (auto ins_wait_for =
                 std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(be_instruction))
    {
        lu << emit_block_executor_instruction_wait_for(ins_wait_for)->get_code();
    }
    else
    {
        NNFUSION_LOG(ERROR)
            << "BlockFusionCudaCodegen: do not support this BlockExecutorInstruction";
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_block_executor(int be_id)
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_be_" + std::to_string(be_id)));
    LanguageUnit& lu = *_lu;

    auto be = block_executor_program.block_executor_instructions[be_id];

    for (auto be_instruction : be)
    {
        lu << emit_block_executor_instruction(be_instruction)->get_code();
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    // be_state_buffer for group_sync
    if (this->is_group_sync)
    {
        params.push_back("int* be_state_buffer");
    }

    lu << "extern \"C\" __global__  void "
       << "(" << join(params, ", ") << ")";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_alloc_shared()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_alloc_shared"));
    LanguageUnit& lu = *_lu;

    size_t kernel_shared_size = 0;

    for (auto kernel : m_context->kernels)
    {
        auto block_kernel = std::dynamic_pointer_cast<BlockCudaEmitter>(kernel);
        NNFUSION_CHECK_NOT_NULLPTR(block_kernel);
        kernel_shared_size = std::max(kernel_shared_size, block_kernel->get_shared_memory_size());
    }

    // avoid allocate shared_memory when no kernels use shared_memory
    if (kernel_shared_size == 0)
    {
        this->is_shared_buffer = false;
    }
    else
    {
        this->is_shared_buffer = true;
        lu << "__shared__ char shared_buffer[" << std::to_string(kernel_shared_size) << "];"
           << "\n";
    }

    // alloc shared for block sync

    lu << "\n";

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_range_branch(int kernel_id,
                                                         int be_st,
                                                         int be_ed,
                                                         bool flag_first_branch)
{
    LanguageUnit_p _lu(new LanguageUnit());
    auto& lu = *_lu;

    auto deduped_kernel_id = kernel_id;
    if (this->is_dedupe_block_kernels == true)
    {
        deduped_kernel_id = deduped_kernel_id_map[kernel_id];
    }
    auto kernel_emitter = block_executor_program.block_kernels[kernel_id];
    auto deduped_kernel_emitter = block_executor_program.block_kernels[deduped_kernel_id];

    std::vector<std::string> params;
    for (size_t i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
    {
        std::stringstream ss;
        ss << all_args[kernel_emitter->m_context->inputs[i]->get_name()];
        params.push_back(ss.str());
    }
    for (size_t i = 0; i < kernel_emitter->m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << all_args[kernel_emitter->m_context->outputs[i]->get_name()];
        params.push_back(ss.str());
    }
    params.push_back("threadIdx.x");
    params.push_back("blockIdx.x - " + std::to_string(be_st));

    if (this->is_shared_buffer == false)
    {
        params.push_back("NULL");
    }
    else
    {
        params.push_back("shared_buffer");
    }

    if (flag_first_branch)
    {
        lu << "if ((int)blockIdx.x >= " << std::to_string(be_st)
           << " && (int)blockIdx.x <= " << std::to_string(be_ed) << ")\n";
    }
    else
    {
        lu << "else if ((int)blockIdx.x >= " << std::to_string(be_st)
           << " && (int)blockIdx.x <= " << std::to_string(be_ed) << ")\n";
    }
    lu.block_begin();
    lu << deduped_kernel_emitter->get_function_name() << "_block_kernel"
       << "(" << join(params, ", ") << ");"
       << "\n";
    lu.block_end();

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body_with_range_branch()
{
    NNFUSION_LOG(INFO) << "BlockFusionCudaCodegen: Try range_branch codegen style.";

    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() > 1)
        {
            return nullptr;
        }
        if (block_executor_program.block_executor_instructions[be_id].size() > 0)
        {
            auto be_instruction = block_executor_program.block_executor_instructions[be_id][0];
            auto ins_execute_block =
                std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction);
            if (ins_execute_block == nullptr)
            {
                return nullptr;
            }
        }
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    LanguageUnit& lu = *_lu;

    // convert tensor name format
    all_args.clear();
    // in_args.clear();
    // out_args.clear();
    // local_tensors.clear();
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        all_args[tensor->get_name()] = "input" + std::to_string(i);
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        all_args[tensor->get_name()] = "output" + std::to_string(i);
    }

    lu << emit_alloc_shared()->get_code();

    bool flag_first_be_id = true;
    int kernel_block_schedule_checker = 0;
    int emit_be_st = 0;
    int emit_be_ed = 0;
    int emit_kernel_id = -1;
    int current_kernel_id = -1;
    int current_kernel_block_id = 0;
    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() == 0)
        {
            current_kernel_id = -1;
            if (emit_kernel_id != -1)
            {
                emit_be_ed = be_id - 1;
                lu << emit_range_branch(emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                          ->get_code();
                flag_first_be_id = false;
            }
            emit_kernel_id = -1;
            emit_be_st = emit_be_ed = be_id;
            kernel_block_schedule_checker = 0;
        }
        else
        {
            auto be_instruction = block_executor_program.block_executor_instructions[be_id][0];
            auto ins_execute_block =
                std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(be_instruction);
            current_kernel_id = ins_execute_block->kernel_id;
            current_kernel_block_id = ins_execute_block->kernel_block_id;
            if (emit_kernel_id == current_kernel_id)
            {
                emit_be_ed = be_id;
                // check schedule policy
                if (current_kernel_block_id - be_id != kernel_block_schedule_checker)
                {
                    return nullptr;
                }
            }
            else
            {
                if (emit_kernel_id != -1)
                {
                    emit_be_ed = be_id - 1;
                    lu << emit_range_branch(
                              emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                              ->get_code();
                    flag_first_be_id = false;
                }
                emit_kernel_id = current_kernel_id;
                emit_be_st = be_id;
                emit_be_ed = be_id;
                kernel_block_schedule_checker = current_kernel_block_id - be_id;
            }
        }
    }
    if (emit_kernel_id != -1)
    {
        lu << emit_range_branch(emit_kernel_id, emit_be_st, emit_be_ed, flag_first_be_id)
                  ->get_code();
        flag_first_be_id = false;
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_body()
{
    LanguageUnit_p _lu_with_range_branch = emit_function_body_with_range_branch();
    if (_lu_with_range_branch != nullptr)
    {
        return _lu_with_range_branch;
    }
    NNFUSION_LOG(INFO) << "BlockFusionCudaCodegen: range_branch codegen style failed, fallback.";

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    LanguageUnit& lu = *_lu;

    // convert tensor name format
    all_args.clear();
    // in_args.clear();
    // out_args.clear();
    // local_tensors.clear();
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        all_args[tensor->get_name()] = "input" + std::to_string(i);
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        all_args[tensor->get_name()] = "output" + std::to_string(i);
    }

    lu << emit_alloc_shared()->get_code();

    bool flag_first_be_id = true;
    for (int be_id = 0; be_id < block_executor_program.block_executor_instructions.size(); be_id++)
    {
        // skip empty BEs when there are no kernels in some BEs.
        if (block_executor_program.block_executor_instructions[be_id].size() > 0)
        {
            if (flag_first_be_id)
            {
                lu << "if (blockIdx.x == " << std::to_string(be_id) << ")\n";
                flag_first_be_id = false;
            }
            else
            {
                lu << "else if (blockIdx.x == " << std::to_string(be_id) << ")\n";
            }
            lu.block_begin();
            lu << emit_block_executor(be_id)->get_code();
            lu.block_end();
        }
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::stdio);

    // keep each kernel's dependency
    for (auto kernel : m_context->kernels)
    {
        auto kernel_dep = kernel->get_or_emit_source()->dep_unit;
        for (auto& it : kernel_dep->local_symbol)
        {
            _lu->require(it.second);
        }
    }

    if (this->is_group_sync)
    {
        _lu->require(emit_step_to_device_function());
        // _lu->require(emit_wait_for_device_function());
    }

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_function_name"));
    LanguageUnit& lu = *_lu;

    std::vector<std::string> names;
    for (auto kernel : m_context->kernels)
    {
        names.push_back(kernel->m_context->gnode->get_op_type());
    }

    lu << "BlockFusionKernel_" << join(m_context->dtypes, "_") << "_" << m_kernel_type << "_"
       << join(names, "_") << "_" << BlockFusionCudaCodegen::unique_func_id++; //<< custom_tag;

    return _lu;
}

LanguageUnit_p BlockFusionCudaCodegen::emit_comments()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_comments"));
    LanguageUnit& lu = *_lu;

    lu << "// Node name:\t BlockFusion"
       << "\n";
    //lu << "// Description:\t" << m_context->node->description() << "\n";
    lu << "// Input:\n";
    for (auto in : m_context->inputs)
    {
        lu << "//\t- name: " << in->get_name();
        lu << "\ttype: " << in->get_element_type().c_type_string();
        lu << "\tshape: " << in->get_shape();
        lu << "\n";
    }

    lu << "// Output:\n";
    for (auto out : m_context->outputs)
    {
        lu << "//\t- name: " << out->get_name();
        lu << "\ttype: " << out->get_element_type().c_type_string();
        lu << "\tshape: " << out->get_shape();
        lu << "\n";
    }

    if (!m_context->tensors.empty())
    {
        lu << "// Other tensors in use:\n";
        for (auto persist : m_context->tensors)
        {
            lu << "//\t- name: " << persist->get_name();
            lu << "\ttype: " << persist->get_element_type().c_type_string();
            lu << "\tshape: " << persist->get_shape();
            lu << "\n";
        }
    }

    lu << "// Fused functions:\n";
    for (auto kernel : m_context->kernels)
    {
        lu << "// " << kernel->get_or_emit_source()->name_unit->get_code()
           << kernel->get_or_emit_source()->call_unit->get_code();
    }

    if (is_dedupe_block_kernels == true)
    {
        lu << "// Deduped function map: <src_function_name : deduped_function_name>\n";
        for (int kernel_id = 0; kernel_id < block_executor_program.block_kernels.size();
             kernel_id++)
        {
            if (kernel_id != deduped_kernel_id_map[kernel_id])
            {
                lu << "// " << block_executor_program.block_kernels[kernel_id]->get_function_name()
                   << " : "
                   << block_executor_program.block_kernels[deduped_kernel_id_map[kernel_id]]
                          ->get_function_name()
                   << "\n";
            }
        }
    }

    // emit block kernel functions here
    lu << emit_block_kernel_functions()->get_code();

    return _lu;
}

void BlockFusionCudaCodegen::set_launch_config()
{
    int grids, blocks, bound;
    compute_launch_config(grids, blocks, bound);

    m_gridDim = dim3(grids, 1, 1);
    m_blockDim = dim3(blocks, 1, 1);
}

void BlockFusionCudaCodegen::compute_launch_config(int& grids, int& blocks, int& bound)
{
    grids = block_executor_program.num_bes;
    // launch less thread_blocks when there are no kernels in some BEs.
    for (int be_id = block_executor_program.num_bes - 1; be_id >= 0; be_id--)
    {
        if (block_executor_program.block_executor_instructions[be_id].size() == 0)
        {
            grids = be_id;
        }
        else
        {
            break;
        }
    }

    blocks = 0;
    for (auto kernel : m_context->kernels)
    {
        auto block_kernel = std::dynamic_pointer_cast<BlockCudaEmitter>(kernel);
        NNFUSION_CHECK_NOT_NULLPTR(block_kernel);
        dim3 kernel_block_dim = block_kernel->get_block_dim();
        blocks = std::max(blocks, kernel_block_dim.x * kernel_block_dim.y * kernel_block_dim.z);
    }
}

LanguageUnit_p BlockFusionCudaCodegen::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    set_launch_config();

    string stream_name = "0";
    auto gnode = m_context->gnode;
    if (gnode != nullptr)
    {
        NNFUSION_CHECK_NOT_NULLPTR(gnode);
        if ((*gnode)["Async_info"].is_valid())
        {
            auto& async_info = (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
            if (async_info.execution_stream != nullptr)
                stream_name = async_info.execution_stream->get_name();
        }
    }

    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    // for group_sync
    if (this->is_group_sync)
    {
        names.push_back(this->m_context->tensors[0]->get_name());
    }

    lu << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", " << m_gridDim.z << "), dim3("
       << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z << "), 0, " << stream_name
       << ">>>(" << join(names, ", ") << ");\n";

    return _lu;
}

// FunctionUnit_p BlockFusionCudaCodegen::emit_source()
// {
//     FunctionUnit_p fu(new FunctionUnit());

//     if (this->m_kernel_name.empty())
//     {
//         fu->name_unit = emit_function_name();
//         this->m_kernel_name = fu->name_unit->get_code();
//     }

//     if (kernel_definitions.find(this->m_kernel_name) != kernel_definitions.end())
//     {
//         NNFUSION_CHECK_NOT_NULLPTR(fu = kernel_definitions[this->m_kernel_name]);
//         return fu;
//     }

//     // emit function units
//     NNFUSION_CHECK_NOT_NULLPTR(fu->signature_unit = emit_function_signature());
//     fu->body_unit = emit_function_body();
//     if (!fu->body_unit)
//     {
//         return nullptr;
//     }
//     NNFUSION_CHECK_NOT_NULLPTR(fu->call_unit = emit_function_call());
//     NNFUSION_CHECK_NOT_NULLPTR(fu->dep_unit = emit_dependency());
//     NNFUSION_CHECK_NOT_NULLPTR(fu->comment_unit = emit_comments());

//     // Pass other to dep_unit
//     for (auto& it : fu->call_unit->local_symbol)
//         fu->dep_unit->require(it.second);
//     for (auto& it : fu->body_unit->local_symbol)
//         fu->dep_unit->require(it.second);
//     fu->call_unit->clean_require();
//     fu->body_unit->clean_require();

//     // orgnize dep
//     CHECK(fu->body_unit->require(fu->dep_unit));
//     CHECK(fu->call_unit->require(fu->body_unit));

//     return fu;
// }
