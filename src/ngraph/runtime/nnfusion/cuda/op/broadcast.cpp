// Microsoft (c) 2019, Wenxiang
#include "broadcast.hpp"

using namespace nnfusion::cuda;
Broadcast::Broadcast(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    _op = static_pointer_cast<ir::Broadcast>(inter_op);
    enforce_not_nullptr(_op) << "Wrong Operator was given.";
}

string Broadcast::codegen_function_name()
{
    // assumes NC{d1,...,dn} format
    std::string kernel_name =
        "broadcast_" + join(_op->dtypes, "_") + "_r" + std::to_string(_op->result_shape.size());
    std::replace(kernel_name.begin(), kernel_name.end(), ' ', '_');
    std::stringstream ss;
    ss << "cuda_" << kernel_name << "_s" << join(_op->result_shape, "_") << "_rs"
       << join(_op->axes, "_");
    return ss.str();
}

string Broadcast::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p Broadcast::codegen_function_definition()
{
    LanguageUnit_p lu(new LanguageUnit(codegen_function_name()));
    auto& writer = *lu;
    writer << "extern \"C\" __global__ void " << lu->symbol;
    //Adding parameters
    writer << "(" << _op->dtypes[0] << "* in, " << _op->dtypes[1] << "* out, size_t nthreads)\n";
    writer.block_begin();
    {
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

        writer << expand_vector_uint32("strides", _op->strides)
               << expand_vector_int("stride_magic", _op->stride_magic)
               << expand_vector_int("stride_shift", _op->stride_shift)
               << expand_vector_uint32("reduced_strides", _op->reduced_strides)
               << "const int tid = blockDim.x*blockIdx.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            // calculate tensor coordinates (inverse tensor reduction)
            std::string reduced_idx = collective_coordinate_transform_helper(writer,
                                                                             "tid",
                                                                             "strides",
                                                                             "stride_magic",
                                                                             "stride_shift",
                                                                             "reduced_strides",
                                                                             "coordinate",
                                                                             _op->rank,
                                                                             true);
            writer << "out[tid] = load(in, " << reduced_idx << ");\n";
        }
        writer.block_end();
    }
    writer.block_end();
    writer.require(declaration::division_by_invariant_multiplication);
    writer.require(declaration::load);
    return lu;
}

LanguageUnit_p Broadcast::codegen_function_call()
{
    LanguageUnit_p lu(new LanguageUnit(codegen_function_name() + "_call"));
    size_t nthreads = shape_size(_op->result_shape);
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    *lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
        << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
        << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ", " << nthreads
        << ");\n";

    return lu;
}

LanguageUnit_p Broadcast::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit);
    cw->require(header::cuda);
    return cw;
}

CudaFunction_p Broadcast::codegen(ir::Operator_p inter_op)
{
    create_ptr(Broadcast, cop, inter_op);
    LOG_INFO << "Codegen for Broadcast function:" << cop->codegen_function_name() << endl;
    return cop;
}