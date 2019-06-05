// Microsoft (c) 2019, Wenxiang
#include "avg_pool.hpp"
#include "../../core/type_info.hpp"
#include "../cuda_cudnn.hpp"

cuda::AvgPool::AvgPool(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    enforce_not_nullptr(this->op = static_pointer_cast<ir::AvgPool>(inter_op));
}

cuda::CudaFunction_p cuda::AvgPool::codegen(ir::Operator_p inter_op)
{
    auto op = static_pointer_cast<ir::AvgPool>(inter_op);
    enforce_not_nullptr(op);
    if (op->input_shape.size() == 3)
    {
        // AvgPool1d of cuda code
        AvgPool_p cop(new AvgPool1D(inter_op));
        LOG_INFO << "Codegen for AvgPool function:" << cop->codegen_function_name() << endl;
        return cop;
    }
    else
    {
        // AvgPoolmD for cudnn code
        create_ptr(AvgPoolmD, avg_pool_op, inter_op);
        return avg_pool_op;
    }
}

cuda::pooling_op_shape cuda::AvgPool::avgpool_shape(
    NVShape in, NVShape out, NVShape window, NVShape strides, NVShape pad)
{
    pooling_op_shape shape;
    shape.N = in[0];
    shape.C = in[1];
    shape.K = shape.C; // pooling feature maps is
    shape.J = shape.C; // not currently supported
    if (in.size() == 3)
    {
        shape.D = 1;
        shape.H = 1;
        shape.W = in[2];
        shape.M = 1;
        shape.P = 1;
        shape.Q = out[2];
        shape.T = 1;
        shape.R = 1;
        shape.S = window[0];
        shape.STRIDE_D = 0;
        shape.STRIDE_H = 0;
        shape.STRIDE_W = strides[0];
        shape.PAD_D = 0;
        shape.PAD_H = 0;
        shape.PAD_W = pad[0];
    }
    else if (in.size() == 4)
    {
        shape.D = 1;
        shape.H = in[2];
        shape.W = in[3];
        shape.M = 1;
        shape.P = out[2];
        shape.Q = out[3];
        shape.T = 1;
        shape.R = window[0];
        shape.S = window[1];
        shape.STRIDE_D = 0;
        shape.STRIDE_H = strides[0];
        shape.STRIDE_W = strides[1];
        shape.PAD_D = 0;
        shape.PAD_H = pad[0];
        shape.PAD_W = pad[1];
    }
    else if (in.size() == 5)
    {
        shape.D = in[2];
        shape.H = in[3];
        shape.W = in[4];
        shape.M = out[2];
        shape.P = out[3];
        shape.Q = out[4];
        shape.T = window[0];
        shape.R = window[1];
        shape.S = window[2];
        shape.STRIDE_D = strides[0];
        shape.STRIDE_H = strides[1];
        shape.STRIDE_W = strides[2];
        shape.PAD_D = pad[0];
        shape.PAD_H = pad[1];
        shape.PAD_W = pad[2];
    }
    else
    {
        throw std::runtime_error("AvgPool currently supports up to 3 spatial dimensions.");
    }
    return shape;
}

cuda::AvgPool1D::AvgPool1D(ir::Operator_p inter_op)
    : AvgPool(inter_op)
{
    shape = cuda::AvgPool::avgpool_shape(
        op->input_shape, op->result_shape, op->window_shape, op->window_stride, op->padding_below);
    // precompute for fast constant memory access
    HW = shape.H * shape.W;
    DHW = shape.D * HW;
    CDHW = shape.C * DHW;
    PQ = shape.P * shape.Q;
    MPQ = shape.M * PQ;
    KMPQ = shape.K * MPQ;
    RS = shape.R * shape.S;
    TRS = shape.T * RS;

    // precompute magic numbers and shifts for fast integer division
    std::tie(magic_N, shift_N) = idiv_magic_u64(shape.N);
    std::tie(magic_P, shift_P) = idiv_magic_u64(shape.P);
    std::tie(magic_S, shift_S) = idiv_magic_u64(shape.S);
    std::tie(magic_RS, shift_RS) = idiv_magic_u64(RS);

    // TODO: blending factors are not currently implemented
    alpha = 1.0f;
    beta = 0.0f;
}

string cuda::AvgPool1D::codegen_function_name()
{
    std::string kernel_name = "cuda_avgpool";
    std::stringstream ss;
    ss << kernel_name << "_s" << join(op->input_shape, "_") << "_r" << join(op->result_shape, "_")
       << "_st" << join(op->window_stride, "_") << "_ip" << int(op->include_pad);
    return ss.str();
}

string cuda::AvgPool1D::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::AvgPool1D::codegen_function_definition()
{
    create_ptr(LanguageUnit, lu, codegen_function_name());
    auto& writer = *lu;
    // In the pooling operation out = P(in) where in: NCDHW -> out: NKMPQ
    // via pooling window: JTRS. Currently feature pooling
    // is not supported and so K = C and J is unused
    writer << "extern \"C\" __global__ void " << writer.symbol << "(" << op->dtypes[0] << "* in, "
           << op->dtypes[1] << "* out)\n";
    /*
    << "float alpha, float beta, "
    << "int N, int C, int D, int H, int W, "
    << "int HW, int DHW, int CDHW, int magic_N, int shift_N, "
    << "int P, int Q, int magic_P, int shift_P, "
    << "int PQ, int MPQ, int KMPQ, "
    << "int S, int RS, int TRS, "
    << "int magic_S, int shift_S, int magic_RS, int shift_RS, "
    << "int str_d, int str_h, int str_w, "
    << "int pad_d, int pad_h, int pad_w"
    << ")\n";
    */
    writer.block_begin();
    {
        /*CONST*/
        writer << "float alpha = " << alpha << ";\n";
        writer << "float beta = " << beta << ";\n";
        writer << "int N = " << shape.N << ";\n";
        writer << "int C = " << shape.C << ";\n";
        writer << "int D = " << shape.D << ";\n";
        writer << "int H = " << shape.H << ";\n";
        writer << "int W = " << shape.W << ";\n";

        writer << "int HW = " << HW << ";\n";
        writer << "int DHW = " << DHW << ";\n";
        writer << "int CDHW = " << CDHW << ";\n";
        writer << "int magic_N = " << magic_N << ";\n";
        writer << "int shift_N = " << shift_N << ";\n";
        writer << "int P = " << shape.P << ";\n";
        writer << "int Q = " << shape.Q << ";\n";
        writer << "int magic_P = " << magic_P << ";\n";
        writer << "int shift_P = " << shift_P << ";\n";

        writer << "int PQ = " << PQ << ";\n";
        writer << "int MPQ = " << MPQ << ";\n";
        writer << "int KMPQ = " << KMPQ << ";\n";
        writer << "int S = " << shape.S << ";\n";
        writer << "int RS = " << RS << ";\n";
        writer << "int TRS = " << TRS << ";\n";

        writer << "int magic_S = " << magic_S << ";\n";
        writer << "int shift_S = " << shift_S << ";\n";
        writer << "int magic_RS = " << magic_RS << ";\n";
        writer << "int shift_RS = " << shift_RS << ";\n";

        writer << "int str_d = " << shape.STRIDE_D << ";\n";
        writer << "int str_h = " << shape.STRIDE_H << ";\n";
        writer << "int str_w = " << shape.STRIDE_W << ";\n";
        writer << "int pad_d = " << shape.PAD_D << ";\n";
        writer << "int pad_h = " << shape.PAD_H << ";\n";
        writer << "int pad_w = " << shape.PAD_W << ";\n";
        /*CONST*/

        writer << "const int tid = threadIdx.x;\n";
        writer << "if (tid < 32)\n";
        writer.block_begin();
        {
            writer << "const int q = blockIdx.x;\n";
            writer << "const int mp = blockIdx.y;\n";
            writer << "const int nk = blockIdx.z;\n";
            writer << "const int k = division_by_invariant_multiplication(nk, magic_N, "
                      "shift_N);\n";
            writer << "const int n = nk - k * N;\n";
            writer << "const int m = division_by_invariant_multiplication(mp, magic_P, "
                      "shift_P);\n";
            writer << "const int p = mp - m * P;\n";
            writer << "out += n*KMPQ + k*MPQ + m*PQ + mad16(p, Q, q);\n";

            // coordinate transform factors from MPQ to DHW
            writer << "int qs = q * str_w - pad_w;\n";
            writer << "int pr = p * str_h - pad_h;\n";
            writer << "int mt = m * str_d - pad_d;\n";

            writer << "int pool_size = ";
            auto pool_size = op->include_pad ? "TRS" : "0";
            writer << pool_size << ";\n";

            writer << "float sum = 0.0f;\n";
            writer << "float rcp_pool_size = 1.0f;\n";
            // each warp operates on a single pooling window and
            // reduces the contents of the window within the warp
            writer << "for (int trs = tid; trs < TRS; trs += 32)\n";
            writer.block_begin();
            {
                writer << "int t = division_by_invariant_multiplication(trs, magic_RS, "
                          "shift_RS);\n";
                writer << "int rs = mod16(trs, t, RS);\n";
                writer << "int r  = division_by_invariant_multiplication(rs, magic_S, shift_S);\n";
                writer << "int s  = mod16(rs, r, S);\n";

                // coordinate transformation from TRS to DHW
                // via MPQ transform factors above
                writer << "int x = qs + s;\n";
                writer << "int y = pr + r;\n";
                writer << "int z = mt + t;\n";

                // helper to check participating threads
                writer << "bool bounds_x = (x >= 0) && (x < W);\n";
                writer << "bool bounds_y = (y >= 0) && (y < H);\n";
                writer << "bool bounds_z = (z >= 0) && (z < D);\n";
                writer << "bool within_tensor_bounds = bounds_x && bounds_y && bounds_z;\n";

                if (op->include_pad == false)
                {
                    // count the number of (non-padded) elements
                    writer << "pool_size += __popc(__ballot_sync(0xffffffff, "
                              "within_tensor_bounds));\n";
                }
                // this will need to change to k->c once
                // feature pooling support is added
                writer << "int idx = n*CDHW + k*DHW + z*HW + y*W + x;\n";
                writer << "sum += load(in,idx,within_tensor_bounds);\n";
            }
            writer.block_end();

            writer << "rcp_pool_size = 1.0f / (float)pool_size;\n";
            // reduce pooling window within warp.
            // this could be improved by calculating the
            // pooling windows each thread can partake in to
            // reduce loads and increase coalescing. in that case,
            // multiple warps per block would be required and the
            // warp reduced sums would need to be accumulated in
            // shared memory
            writer << "for (int i = 16; i > 0; i >>= 1)\n";
            writer.block_begin();
            {
                writer << "sum += __shfl_xor_sync(0xffffffff,sum,i,32);\n";
            }
            writer.block_end();
            // write result to output
            writer << "if (tid == 0)\n";
            writer.block_begin();
            {
                writer << "*out = sum * rcp_pool_size;\n";
            }
            writer.block_end();
        }
        writer.block_end();
    }
    writer.block_end();
    return lu;
}

LanguageUnit_p cuda::AvgPool1D::codegen_function_call()
{
    LanguageUnit_p lu(new LanguageUnit(codegen_function_name() + "_call"));

    *lu << codegen_function_name() << "<<<dim3(" << shape.Q << ", " << shape.M * shape.P << ", "
        << shape.N * shape.K << "), dim3(" << 32 << ", " << 1 << ", " << 1 << "), " << 0 << ", "
        << 0 << ">>>"
        << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ");\n";

    return lu;
}

LanguageUnit_p cuda::AvgPool1D::codegen_dependency()
{
    LanguageUnit_p cw(new LanguageUnit);
    cw->require(header::cuda);
    cw->require(declaration::division_by_invariant_multiplication);
    cw->require(declaration::load);
    cw->require(declaration::mad16);
    cw->require(declaration::mod16);
    return cw;
}

string cuda::AvgPoolmD::codegen_function_name()
{
    std::stringstream ss;
    string dtype = op->out[0].get_element_type().c_type_string();
    ss << "cudnn_avgpool_dtype_" << dtype << "_i" << join(op->input_shape, "_") << "_o"
       << join(op->result_shape, "_") << "_ws" << join(op->window_shape, "_") << "_wst"
       << join(op->window_stride, "_") << "_pb" << join(op->padding_below, "_") << "_pb"
       << join(op->padding_above, "_");
    return ss.str();
}

string cuda::AvgPoolmD::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p cuda::AvgPoolmD::codegen_function_definition()
{
    auto cudnn_avg_type = op->include_pad ? "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
                                          : "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING";
    enforce(op->input_shape.size() == 4 || op->input_shape.size() == 5)
        << "Cudnn Pooling wrong input.";
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu.require(macro::CUDNN_SAFE_CALL);

    lu << "void " << lu.symbol << "(cudnnHandle_t cudnn_handle, " << op->dtypes[0] << "* in, "
       << op->dtypes[1] << "* out)\n";
    lu.block_begin();
    {
        auto input_desc = cudnn_tensor_descriptor_from_shape(op->input_shape, "input_desc");
        auto output_desc = cudnn_tensor_descriptor_from_shape(op->result_shape, "output_desc");

        lu << input_desc->get_code();
        lu << output_desc->get_code();

        lu << "cudnnPoolingDescriptor_t desc;\n";
        lu << "cudnnCreatePoolingDescriptor(&desc);\n";
        if (op->input_shape.size() == 4)
        {
            lu << "CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,"
               << " " << cudnn_avg_type << ","
               << " CUDNN_NOT_PROPAGATE_NAN," << static_cast<int>(op->window_shape[0]) << ", "
               << static_cast<int>(op->window_shape[1]) << ", "
               << static_cast<int>(op->padding_below[0]) << ", "
               << static_cast<int>(op->padding_below[1]) << ", "
               << static_cast<int>(op->window_stride[0]) << ", "
               << static_cast<int>(op->window_stride[1]) << "));\n";
        }
        else /*op->input_shape.size() == 5*/
        {
            std::vector<int> w_strides(op->window_stride.size());
            std::vector<int> w_shape(op->window_shape.size());
            std::vector<int> w_padding(op->padding_below.size());
            for (int i = 0; i < op->window_shape.size(); i++)
            {
                w_shape[i] = static_cast<int>(op->window_shape[i]);
                w_strides[i] = static_cast<int>(op->window_stride[i]);
                w_padding[i] = static_cast<int>(op->padding_below[i]);
            }

            auto expand_vector_int = [](string name, vector<int>& d) {
                stringstream ss;
                enforce(d.size() > 0);
                ss << "int " << name << "[] = {";
                for (int i = 0; i + 1 < d.size(); i++)
                    ss << to_string(d[i]) << ", ";
                ss << to_string(d.back()) << "}\n";
                return ss.str();
            };

            lu << expand_vector_int("w_shape", w_shape);
            lu << expand_vector_int("w_strides", w_strides);
            lu << expand_vector_int("w_padding", w_padding);

            lu << "CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc, "
               << " " << cudnn_avg_type << ","
               << "CUDNN_NOT_PROPAGATE_NAN, "
               << "3, w_shape, w_padding, w_strides));\n";
        }

        lu << "const float alpha = 1.0;\n";
        lu << "const float beta = 0.0;\n";

        lu << "CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle,"
           << " desc,"
           << " &alpha,"
           << " input_desc,"
           << " in,"
           << " &beta,"
           << " output_desc,"
           << " out));\n";

        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));\n";
    }
    lu.block_end();
    return plu;
}

LanguageUnit_p cuda::AvgPoolmD::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name());
    auto& lu = *plu;
    lu << codegen_function_name() << "(global_cudnn_handle, " << op->arg_names[0] << ", "
       << op->out_names[0] << ");\n";
    return plu;
}

LanguageUnit_p cuda::AvgPoolmD::codegen_dependency()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name() + "_dep");
    auto& lu = *_lu;
    lu.require(header::cudnn);
    lu.require(declaration::global_cudnn_handle);
    return _lu;
}