// Microsoft (c) 2019, NNFusion Team

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/ops/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class UnsortedSegmentSum : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;
                size_t threads;
                size_t input_count;
                int stride0;
                int rows;
                ngraph::element::Type dtype;

            public:
                UnsortedSegmentSum(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
                    threads = ctx->outputs.front().get_size();
                    input_count = ctx->inputs.size();
                    dtype = ngraph::element::Type(ctx->outputs[0].get_element_type());
                    auto in_shape = ctx->inputs.front().get_shape();
                    stride0 = ngraph::row_major_strides(in_shape)[0];
                    rows = in_shape[0];
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    auto code = ngraph::op::create_code_from_template(
                        R"(
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid >= @nthreads@) return;
size_t stride0 = @stride0@;
size_t rowid = tid / stride0;
size_t colid = tid - rowid * stride0;
@typestring@ acc = 0;
for(size_t row = 0; row < @row_number@; row++)
{
    if(input1[row] == rowid)
        acc += input0[row * stride0 + colid];
}
output0[tid] = acc;
)",
                        {{"nthreads", threads},
                         {"stride0", stride0},
                         {"typestring", dtype.c_type_string()},
                         {"row_number", rows}});

                    lu << code;
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::stdio);
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 512;
                    size_t block_cnt = align_to_block_size(threads, block_size_x);

                    m_gridDim = dim3(block_cnt, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("UnsortedSegmentSum",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::UnsortedSegmentSum)
