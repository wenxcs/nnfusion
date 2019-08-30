
namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class __KernelUniqueClassName__ : public CudaEmitter
            {
                shared_ptr<ngraph::op::GenericOp> generic_op;

            public:
                __KernelUniqueClassName__(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(static_pointer_cast<ngraph::op::GenericOp>(ctx->node))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    generic_op->validate_and_infer_types();

                    size_t num_in = generic_op->get_input_size(),
                           num_out = generic_op->get_output_size();
                    std::vector<ngraph::Shape> input_shapes, output_shapes;
                    for (int i = 0; i < num_in; ++i)
                        input_shapes.push_back(generic_op->get_input_shape(i));
                    for (int i = 0; i < num_out; ++i)
                        output_shapes.push_back(generic_op->get_output_shape(i));

                    auto res = generate_kernel_code(
                        input_shapes, output_shapes, generic_op->localOpConfig.getRoot());
                    if (res.is_null())
                        return nullptr;

                    m_blockDim =
                        dim3(res["block_dim"][0], res["block_dim"][1], res["block_dim"][2]);
                    m_gridDim = dim3(res["grid_dim"][0], res["grid_dim"][1], res["grid_dim"][2]);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu.block_begin();
                    lu << (std::string)res["source_code"] << "\n";
                    lu.block_end();
                    return _lu;
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

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(__KernelOpType__,                                             // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::__KernelUniqueClassName__)                              // constructor
