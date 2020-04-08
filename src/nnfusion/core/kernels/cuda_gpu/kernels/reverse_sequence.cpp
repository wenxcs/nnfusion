// Microsoft (c) 2019, NNFusion Team

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ReverseSequence : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                vector<int64_t> seq_lengths;
                vector<int64_t> strides;
                Shape out_shape;
                uint32_t seq_axis, max_seq_len, batch_axis, threads;

            public:
                ReverseSequence(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    seq_lengths =
                        (*ctx->gnode)["ReverseSequenceOp::seq_lengths"].as<vector<int64_t>>();
                    seq_axis = generic_op->localOpConfig.get("seq_axis");
                    batch_axis = generic_op->localOpConfig.get("batch_axis");
                    out_shape = ctx->outputs[0]->get_shape();
                    threads = shape_size(out_shape);

                    strides.push_back(1);
                    for (int i = 0; i < out_shape.size(); i++)
                        strides.push_back(strides[i] * out_shape[out_shape.size() - i - 1]);
                    std::reverse(strides.begin(), strides.end());
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto code = nnfusion::op::create_code_from_template(
                        R"(uint32_t seq_lengths[] = {@seq_lengths@};
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < @threads@) {
    uint32_t seq_dim = tid / @seq_stride@ % @seq_dim_size@;
    uint32_t batch_dim = tid / @batch_stride@ % @batch_dim_size@;
    uint32_t new_dim = (seq_dim <= (seq_lengths[batch_dim] - 1) / 2) ? seq_lengths[batch_dim] - 1 - seq_dim : seq_dim;
    uint32_t from_tid = tid + (new_dim - seq_dim) * @seq_stride@;
    // Just copy
    output0[tid] = input0[tid];
    // Swap
    if(new_dim > seq_dim)
    {
        auto t = output0[tid];
        output0[tid] = output0[from_tid];
        output0[from_tid] = t;
    }
})",
                        {{"seq_lengths", join(seq_lengths)},
                         {"threads", threads},
                         {"seq_stride", strides[seq_axis + 1]},
                         {"seq_dim_size", out_shape[seq_axis]},
                         {"batch_stride", strides[batch_axis + 1]},
                         {"batch_dim_size", out_shape[batch_axis]}});
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name(), code));
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x =
                        align_to_block_size(static_cast<uint32_t>(threads), block_size_x);
                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("ReverseSequence",                                            // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::ReverseSequence)                                        // constructor