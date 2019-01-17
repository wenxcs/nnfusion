// Microsoft (c) 2019, Wenxiang
#pragma once
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            namespace codegen
            {
                namespace cuda
                {
                    class Reshape : public CodeGenOP
                    {
                    public:
                        static std::shared_ptr<CodeGenOP>
                            codegen(std::shared_ptr<IntermediateOP> ireshape)
                        {
                            writer.block_begin();
                            {
                                bool same_layout =
                                    is_sorted(new_input_order.begin(), new_input_order.end());
                                if (same_layout)
                                {
                                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                                }
                                // If there *is* a layout change in the 2D case, we transpose the input.
                                else
                                {
                                    writer << "void* input[] = {" << node_names(args) << "};\n";
                                    writer << "void* output[] = {" << node_names(out) << "};\n";
                                    auto& cuda_emitter = external_function->get_primitive_emitter()
                                                             ->get_cuda_emitter();
                                    size_t index;
                                    if (new_rank == 2)
                                    {
                                        index = cuda_emitter->build_reshape_2d(
                                            {{args[0].get_type(), out[0].get_type()}},
                                            new_arg_shape,
                                            new_input_order);
                                    }
                                    // If there *is* a layout change in the 3D case, we do 3D tiled reshape.
                                    else if (new_rank == 3)
                                    {
                                        index = cuda_emitter->build_reshape_3d(
                                            {{args[0].get_type(), out[0].get_type()}},
                                            new_arg_shape,
                                            new_input_order);
                                    }
                                    // Other cases (reordering of axes for tensors with rank>3).
                                    else
                                    {
                                        index = cuda_emitter->build_reshape(
                                            {{args[0].get_type(), out[0].get_type()}},
                                            new_arg_shape,
                                            new_input_order);
                                    }
                                    writer << "gpu::invoke_primitive(ctx, " << index
                                           << ", input, output);\n";
                                }
                            }
                            writer.block_end();
                        }
                    };
                }
            }
        }
    }
}