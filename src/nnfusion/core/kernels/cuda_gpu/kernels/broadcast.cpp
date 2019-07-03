// Microsoft (c) 2019, NNFusion Team
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Broadcast : public CudaEmitter
            {
            public:
                Broadcast(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                    auto _op = static_pointer_cast<ngraph::op::Broadcast>(ctx->node);
                    enforce_not_nullptr(_op) << "Node type is not Broadcast.";
                    auto& axes = _op->get_broadcast_axes();
                    if (axes.empty())
                    {
                        isMemcpy = true;
                    }
                    else
                    {
                        this->axes = AxisSet(axes);
                    }

                    result_shape = ctx->outputs[0].get_shape();

                    // calculate strides
                    strides = ngraph::row_major_strides(result_shape);
                    // precacluate invariants for integer division via multiplication
                    stride_magic;
                    stride_shift;
                    for (int i = 0; i < strides.size(); i++)
                    {
                        int magic;
                        int shift;
                        std::tie(magic, shift) = idiv_magic_u64(strides[i]);
                        stride_magic.push_back(magic);
                        stride_shift.push_back(shift);
                    }
                    // calculate reduced tensor strides with 0s inserted for reduced axes
                    reduced_shape = result_shape;
                    for (auto const& axis : axes)
                    {
                        reduced_shape[axis] = 1;
                    }
                    reduced_strides = ngraph::row_major_strides(reduced_shape);
                    for (auto const& axis : axes)
                    {
                        reduced_strides[axis] = 0;
                    }

                    rank = result_shape.size();

                    std::stringstream tag;
                    tag << "_s" << join(result_shape, "_") << "_rs" << join(this->axes, "_");
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));

                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
                    auto& writer = *_lu;

                    auto expand_vector_int = [](string name, vector<int>& d) {
                        stringstream ss;
                        for (int i = 0; i < d.size(); i++)
                            ss << "int " << name << i << " = " << to_string(d[i]) << ";\n";
                        return ss.str();
                    };

                    auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
                        stringstream ss;
                        for (int i = 0; i < d.size(); i++)
                            ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
                        return ss.str();
                    };

                    writer << "size_t nthreads = " << shape_size(result_shape) << ";";

                    writer << expand_vector_uint32("strides", strides)
                           << expand_vector_int("stride_magic", stride_magic)
                           << expand_vector_int("stride_shift", stride_shift)
                           << expand_vector_uint32("reduced_strides", reduced_strides)
                           << "const int tid = blockDim.x*blockIdx.x + threadIdx.x;\n";
                    writer << "if (tid < nthreads)\n";
                    writer.block_begin();
                    {
                        // calculate tensor coordinates (inverse tensor reduction)
                        std::string reduced_idx =
                            collective_coordinate_transform_helper(writer,
                                                                   "tid",
                                                                   "strides",
                                                                   "stride_magic",
                                                                   "stride_shift",
                                                                   "reduced_strides",
                                                                   "coordinate",
                                                                   rank,
                                                                   true);
                        writer << "out[tid] = load(in, " << reduced_idx << ");\n";
                    }
                    writer.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(declaration::division_by_invariant_multiplication);
                    _lu->require(declaration::load);
                    return _lu;
                }

                void set_launch_config() override
                {
                    size_t nthreads = shape_size(result_shape);
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x =
                        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);
                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<ngraph::op::Pad> _op;

                // calculate strides
                ngraph::NVShape strides;
                // precacluate invariants for integer division via multiplication
                std::vector<int> stride_magic;
                std::vector<int> stride_shift;
                // calculate reduced tensor strides with 0s inserted for reduced axes
                ngraph::NVShape reduced_shape;
                ngraph::NVShape reduced_strides;
                ngraph::Shape result_shape;
                size_t rank;
                AxisSet axes;
                bool isMemcpy = false;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Broadcast",                               //op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT), //attrs
                        cuda::Broadcast)                           // constructor