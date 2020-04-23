// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "common.hpp"

using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels;

class BlockFusionCudaCodegen : public CudaEmitter
{
public:
    using Pointer = shared_ptr<BlockFusionCudaCodegen>;
    BlockFusionCudaCodegen(shared_ptr<KernelContext> ctx,
                           const BlockExecutorProgram& _block_executor_program);

    std::shared_ptr<KernelContext> get_kernel_context() { return this->m_context; }
    // FunctionUnit_p emit_source() override;

private:
    LanguageUnit_p emit_function_signature() override;
    LanguageUnit_p emit_function_body() override;
    LanguageUnit_p emit_dependency() override;
    LanguageUnit_p emit_function_name() override;
    LanguageUnit_p emit_comments() override;
    LanguageUnit_p emit_function_call() override;

    LanguageUnit_p emit_range_branch(int kernel_id, int be_st, int be_ed, bool flag_first_branch);
    LanguageUnit_p emit_function_body_with_range_branch();

    LanguageUnit_p emit_step_to_device_function();
    // LanguageUnit_p emit_wait_for_device_function();

    LanguageUnit_p emit_block_executor_instruction_execute_block(
        std::shared_ptr<BlockExecutorInstructionExecuteBlock> be_ins_execute_block);
    LanguageUnit_p emit_block_executor_instruction_step_to(
        std::shared_ptr<BlockExecutorInstructionStepTo> be_ins_step_to);
    LanguageUnit_p emit_block_executor_instruction_wait_for(
        std::shared_ptr<BlockExecutorInstructionWaitFor> be_ins_wait_for);

    LanguageUnit_p emit_alloc_shared();
    LanguageUnit_p emit_block_kernel_functions();
    LanguageUnit_p emit_block_executor_instruction(BEInstruction_p be_instruction);
    LanguageUnit_p emit_block_executor(int be_id);

    std::shared_ptr<KernelContext> FuseContext();

    void set_launch_config() override;
    void compute_launch_config(int& grids, int& blocks, int& bound);

private:
    static int unique_func_id;
    BlockExecutorProgram block_executor_program;

    // std::unordered_map<std::string, std::string> in_args;
    // std::unordered_map<std::string, std::string> out_args;
    // std::unordered_map<std::string, std::string> local_tensors;
    std::unordered_map<std::string, std::string> all_args;

    std::unordered_map<int, int> deduped_kernel_id_map; // <src_id, deduped_id>

    bool is_shared_buffer;
    bool is_group_sync;

    bool is_dedupe_block_kernels;
};
