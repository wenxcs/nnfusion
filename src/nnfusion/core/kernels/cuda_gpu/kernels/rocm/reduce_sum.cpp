// Microsoft (c) 2019, NNFusion Team

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class RocmReduce : public CudaEmitter
            {
            public:
                RocmReduce(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                }

                LanguageUnit_p put_source(const std::string& code)
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << code;
                    return _lu;
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& ctx = m_context;
                    auto _op = dynamic_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr());

                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    auto reduce_axis = _op->get_reduction_axes();

                    int min_axis = input_shape.size(), max_axis = -1, reduce_scale = 1;
                    for (auto& axis : reduce_axis)
                    {
                        min_axis = min(min_axis, (int)axis);
                        max_axis = max(max_axis, (int)axis);
                        reduce_scale *= input_shape[axis];
                    }
                    size_t tensor_size = std::accumulate(
                        input_shape.begin(), input_shape.end(), 1LU, std::multiplies<int>());
                    if (reduce_scale == 1 || min_axis > max_axis) // as memcpy
                    {
                        int blocks = tensor_size, threads = 1;
                        for (int i = 1024; i > 1; --i)
                        {
                            if (tensor_size % i == 0)
                            {
                                threads = i;
                                blocks = tensor_size / i;
                                break;
                            }
                        }
                        m_gridDim = dim3(blocks, 1, 1);
                        m_blockDim = dim3(threads, 1, 1);

                        return put_source(nnfusion::op::create_code_from_template(
                            R"(output0[((int)blockIdx.x) * @num_threads@ + ((int)threadIdx.x)] = input0[((int)blockIdx.x) * @num_threads@ + ((int)threadIdx.x)];)",
                            {{"num_threads", threads}}));
                    }
                    else // ReduceSum 2D
                    {
                        int groups, samples, stride_group, stride_sample;
                        if (min_axis == 0 &&
                            max_axis ==
                                input_shape.size() - reduce_axis.size() - 1) // A[X][Y] -> B[Y]
                        {
                            samples = std::accumulate(input_shape.begin(),
                                                      input_shape.begin() + reduce_axis.size(),
                                                      1LU,
                                                      std::multiplies<int>());
                            groups = tensor_size / samples;
                            stride_group = 1;
                            stride_sample = groups;
                        }
                        else if (min_axis == input_shape.size() - reduce_axis.size() &&
                                 max_axis == input_shape.size() - 1) // A[X][Y] -> B[X]
                        {
                            samples = std::accumulate(input_shape.end() - reduce_axis.size(),
                                                      input_shape.end(),
                                                      1LU,
                                                      std::multiplies<int>());
                            groups = tensor_size / samples;
                            stride_group = samples;
                            stride_sample = 1;
                        }
                        else
                            return nullptr;

                        int numThreads = 4; // tunable: 2, 4, 8, 16, 32, 64, 128, 512, 1024
                        m_gridDim = dim3(1, groups, 1);
                        m_blockDim = dim3(numThreads, 1, 1);

                        return put_source(nnfusion::op::create_code_from_template(
                            R"(
    constexpr int numThreads = @num_threads@;
    extern __shared__ float Isdata[numThreads];
    int tid = threadIdx.x;
    Isdata[tid] = 0;

    int i = tid;
    #pragma unroll
    while (i < @samples@) { Isdata[tid] += input0[i * @stride_sample@ + ((int)blockIdx.y) * @stride_group@]; i += numThreads; }
    if (numThreads >= 128) { __syncthreads(); }

    if (numThreads >= 512) { if (tid < 256) { Isdata[tid] += Isdata[tid + 256]; } __syncthreads(); }
    if (numThreads >= 256) { if (tid < 128) { Isdata[tid] += Isdata[tid + 128]; } __syncthreads(); }

    volatile float *__sdata = (volatile float *)Isdata;
    if (numThreads >= 128) __sdata[tid] += __sdata[tid + 64];
    if (numThreads >= 64) __sdata[tid] += __sdata[tid + 32];
    if (numThreads >= 32) __sdata[tid] += __sdata[tid + 16];
    if (numThreads >= 16) __sdata[tid] += __sdata[tid + 8];
    if (numThreads >= 8) __sdata[tid] += __sdata[tid + 4];
    if (numThreads >= 4) __sdata[tid] += __sdata[tid + 2];
    if (numThreads >= 2) __sdata[tid] += __sdata[tid + 1];
    if (tid == 0) output0[((int)blockIdx.y)] = Isdata[0];
)",
                            {{"samples", samples},
                             {"stride_sample", stride_sample},
                             {"stride_group", stride_group},
                             {"num_threads", numThreads}}));
                    }
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
// Microsoft (c) 2019, NNFusion Team

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_GPU_KERNEL(KEY, OP_NAME)                                                          \
    REGISTER_KERNEL_EMITTER(KEY,                                                                   \
                            Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("PRIORITY_2" #OP_NAME),  \
                            cuda::RocmReduce<nnfusion::op::OP_NAME>)

REGISTER_GPU_KERNEL("Sum", Add)
