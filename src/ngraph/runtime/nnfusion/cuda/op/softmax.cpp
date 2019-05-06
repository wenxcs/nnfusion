// Microsoft (c) 2019, Yuchao
#include "softmax.hpp"

using namespace nnfusion::cuda;

Softmax::Softmax(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    enforce_not_nullptr(this->op = static_pointer_cast<ir::Softmax>(inter_op));
}

CudaFunction_p Softmax::codegen(ir::Operator_p inter_op)
{
    auto irop = static_pointer_cast<ir::Softmax>(inter_op);
    enforce_not_nullptr(irop) << "Input operator is invalid.";
    auto iter = irop->reduce_axis.begin();
    NVShape reduce_axis;
    while (iter != irop->reduce_axis.end())
    {
        reduce_axis.push_back(*iter);
        ++iter;
    }
    NVShape simplified_reduce_axis;
    NVShape simplified_input_shape;
    ngraph::NVShape non_reduce_shape;
    ngraph::NVShape non_reduce_strides;
    ngraph::NVShape non_reduce_strides_in_input;
    ngraph::NVShape reduce_shape;
    ngraph::NVShape reduce_strides;
    ngraph::NVShape reduce_strides_in_input;
    simplify_reduce_shape(
        irop->input_shape, reduce_axis, simplified_input_shape, simplified_reduce_axis);

    get_reduce_strides(simplified_input_shape,
                       simplified_reduce_axis,
                       non_reduce_shape,
                       non_reduce_strides,
                       non_reduce_strides_in_input,
                       reduce_shape,
                       reduce_strides,
                       reduce_strides_in_input);
    if (reduce_strides_in_input.back() != 1)
    {
        // if reduce not include last axis, this is a heuristic to choose by reduce axis for better cache
        // a more accurate but slow way is to tune with actual kernel
        create_ptr(SoftmaxStridesBackNotOne, cop, inter_op);
        LOG_INFO << cop->codegen_function_name() << endl;
        return cop;
    }
    else
    {
        create_ptr(SoftmaxStridesBackOne, cop, inter_op);
        LOG_INFO << cop->codegen_function_name() << endl;
        return cop;
    }
}

string Softmax::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p SoftmaxStridesBackOne::codegen_dependency()
{
    std::string name = codegen_function_name() + "_dep";
    create_ptr(LanguageUnit, cw, name);

    cw->require(header::cuda);
    cw->require(declaration::division_by_invariant_multiplication);
    cw->require(declaration::load);
    return cw;
}

LanguageUnit_p SoftmaxStridesBackNotOne::codegen_dependency()
{
    std::string name = codegen_function_name() + "_dep";
    create_ptr(LanguageUnit, cw, name);

    cw->require(header::cuda);
    return cw;
}

SoftmaxStridesBackNotOne::SoftmaxStridesBackNotOne(ir::Operator_p inter_op)
    : Softmax(inter_op)
{
    NVShape simplified_reduce_axis;
    NVShape simplified_input_shape;
    auto iter = op->reduce_axis.begin();
    NVShape reduce_axis;
    while (iter != op->reduce_axis.end())
    {
        reduce_axis.push_back(*iter);
        ++iter;
    }
    simplify_reduce_shape(
        op->input_shape, reduce_axis, simplified_input_shape, simplified_reduce_axis);
    rank = simplified_input_shape.size();
    reduce_rank = simplified_reduce_axis.size();
    non_reduce_rank = rank - reduce_rank;
    out_rank = non_reduce_rank;

    get_reduce_strides(simplified_input_shape,
                       simplified_reduce_axis,
                       non_reduce_shape,
                       non_reduce_strides,
                       non_reduce_strides_in_input,
                       reduce_shape,
                       reduce_strides,
                       reduce_strides_in_input);
    nthreads = static_cast<uint32_t>(shape_size(non_reduce_shape));
}

string SoftmaxStridesBackNotOne::codegen_function_name()
{
    // kernel_name is used to check if the cuda kernel has been previously compiled
    std::stringstream kernel_name;
    kernel_name << "cuda"
                << "_softmax"
                << "_" << join(op->dtypes, "_") << "_i_" << join(op->input_shape, "_") << "_axis_"
                << join(op->reduce_axis, "_");
    return kernel_name.str();
}

LanguageUnit_p SoftmaxStridesBackNotOne::codegen_function_definition()
{
    create_ptr(LanguageUnit, cw, codegen_function_name());
    LanguageUnit& writer = *cw;

    auto stable_sum_lambda = [&]() {
        writer << "input_i = exp(input_i - r_max);\n";
        writer << "y = input_i - c;\n";
        writer << "t = r_sum + y;\n";
        writer << "c = (t - r_sum) - y;\n";
        writer << "r_sum = t;\n";
    };

    auto max_lambda = [&]() { writer << "r_max = r_max > input_i ? r_max : input_i;\n"; };

    auto divide_lambda = [&]() {
        writer << "input_i = exp(input_i - r_max) / r_sum;\n";
        writer << "out[reduce_idx] = input_i;\n";
    };

    writer << "extern \"C\" __global__ void " << writer.symbol << "(" << op->dtypes[0] << "* in, "
           << op->dtypes[1] << "* out, size_t nthreads)\n";
    writer.block_begin();
    {
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        writer << expand_vector_uint32("non_reduce_strides", non_reduce_strides);
        writer << expand_vector_uint32("non_reduce_strides_in_input", non_reduce_strides_in_input);
        writer << expand_vector_uint32("reduce_shape", reduce_shape);
        writer << expand_vector_uint32("reduce_strides_in_input", reduce_strides_in_input);

        writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        writer << "if (tid < nthreads)\n";
        writer.block_begin();
        {
            if (out_rank > 0)
            {
                writer << "uint32_t dim_idx_generator = tid;\n";
            }
            writer << "uint32_t in_idx = 0;\n";

            // loop through all reduction axis
            for (int64_t i = 0; i < static_cast<int64_t>(out_rank); i++)
            {
                writer << "in_idx += (dim_idx_generator / non_reduce_strides" << i
                       << ") * non_reduce_strides_in_input" << i << ";\n";
                writer << "dim_idx_generator %= non_reduce_strides" << i << ";\n";
            }
            writer << "uint32_t init_in_idx = in_idx;\n";
            int64_t last_r_idx = static_cast<int64_t>(reduce_rank) - 1;

            //find max
            writer << op->dtypes[1] << " r_max = in[init_in_idx];\n";
            writer << op->dtypes[1] << " input_i;\n";

            writer.block_begin();
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape" << j << "; idx"
                       << j << "++)\n";
                writer.block_begin();
            }
            {
                writer << "uint32_t reduce_idx = in_idx;\n";
                for (int64_t j = 0; j < last_r_idx; j++)
                {
                    writer << "reduce_idx += idx" << j << " * reduce_strides_in_input" << j
                           << ";\n";
                }
                writer << "uint32_t step = reduce_strides_in_input" << last_r_idx << ";\n";
                writer << "if(reduce_idx != init_in_idx)\n";
                writer.block_begin();
                {
                    writer << "input_i = in[reduce_idx];\n";
                    max_lambda();
                }
                writer.block_end();
                writer << "reduce_idx += step;\n";
                writer << "int idx" << last_r_idx << " = 1;\n";
                // unroll last reduction axis
                uint32_t unroll_num = 8;
                writer << "for(; idx" << last_r_idx << " + " << unroll_num << " - 1 < reduce_shape"
                       << last_r_idx << "; idx" << last_r_idx << " += " << unroll_num << ")\n";
                writer.block_begin();
                {
                    for (int k = 0; k < unroll_num; k++)
                    {
                        writer << "input_i = in[reduce_idx];\n";
                        max_lambda();
                        writer << "reduce_idx += step;\n";
                    }
                }
                writer.block_end();
                writer << "for(; idx" << last_r_idx << " < reduce_shape" << last_r_idx << "; idx"
                       << last_r_idx << "++)\n";
                writer.block_begin();
                {
                    writer << "input_i = in[reduce_idx];\n";
                    max_lambda();
                    writer << "reduce_idx += step;\n";
                }
                writer.block_end();
            }
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer.block_end();
            }
            writer.block_end();

            //exp and sum , https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            writer << op->dtypes[1] << " r_sum = 0;\n";
            writer << op->dtypes[1] << " c = 0;\n";
            writer << op->dtypes[1] << " y;\n";
            writer << op->dtypes[1] << " t;\n";
            writer.block_begin();
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape" << j << "; idx"
                       << j << "++)\n";
                writer.block_begin();
            }
            {
                writer << "uint32_t reduce_idx = in_idx;\n";
                for (int64_t j = 0; j < last_r_idx; j++)
                {
                    writer << "reduce_idx += idx" << j << " * reduce_strides_in_input" << j
                           << ";\n";
                }
                writer << "uint32_t step = reduce_strides_in_input" << last_r_idx << ";\n";
                writer << "int idx" << last_r_idx << " = 0;\n";
                // unroll last reduction axis
                uint32_t unroll_num = 8;
                writer << "for(; idx" << last_r_idx << " + " << unroll_num << " - 1 < reduce_shape"
                       << last_r_idx << "; idx" << last_r_idx << " += " << unroll_num << ")\n";
                writer.block_begin();
                {
                    for (int k = 0; k < unroll_num; k++)
                    {
                        writer << "input_i = in[reduce_idx];\n";
                        stable_sum_lambda();
                        writer << "reduce_idx += step;\n";
                    }
                }
                writer.block_end();
                writer << "for(; idx" << last_r_idx << " < reduce_shape" << last_r_idx << "; idx"
                       << last_r_idx << "++)\n";
                writer.block_begin();
                {
                    writer << "input_i = in[reduce_idx];\n";
                    stable_sum_lambda();
                    writer << "reduce_idx += step;\n";
                }
                writer.block_end();
            }
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer.block_end();
            }
            writer.block_end();

            //divide
            writer.block_begin();
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape" << j << "; idx"
                       << j << "++)\n";
                writer.block_begin();
            }
            {
                writer << "uint32_t reduce_idx = in_idx;\n";
                for (int64_t j = 0; j < last_r_idx; j++)
                {
                    writer << "reduce_idx += idx" << j << " * reduce_strides_in_input" << j
                           << ";\n";
                }
                writer << "uint32_t step = reduce_strides_in_input" << last_r_idx << ";\n";
                writer << "int idx" << last_r_idx << " = 0;\n";
                // unroll last reduction axis
                uint32_t unroll_num = 8;
                writer << "for(; idx" << last_r_idx << " + " << unroll_num << " - 1 < reduce_shape"
                       << last_r_idx << "; idx" << last_r_idx << " += " << unroll_num << ")\n";
                writer.block_begin();
                {
                    for (int k = 0; k < unroll_num; k++)
                    {
                        writer << "input_i = in[reduce_idx];\n";
                        divide_lambda();
                        writer << "reduce_idx += step;\n";
                    }
                }
                writer.block_end();
                writer << "for(; idx" << last_r_idx << " < reduce_shape" << last_r_idx << "; idx"
                       << last_r_idx << "++)\n";
                writer.block_begin();
                {
                    writer << "input_i = in[reduce_idx];\n";
                    divide_lambda();
                    writer << "reduce_idx += step;\n";
                }
                writer.block_end();
            }
            for (int64_t j = 0; j < last_r_idx; j++)
            {
                writer.block_end();
            }
            writer.block_end();
        }
        writer.block_end();
    }
    writer.block_end();
    return cw;
}

LanguageUnit_p SoftmaxStridesBackNotOne::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    auto& lu = *plu;

    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
       << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0 << ", " << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ", " << nthreads
       << ");\n";
    return plu;
}

SoftmaxStridesBackOne::SoftmaxStridesBackOne(ir::Operator_p inter_op)
    : Softmax(inter_op)
{
    NVShape simplified_reduce_axis;
    NVShape simplified_input_shape;
    auto iter = op->reduce_axis.begin();
    NVShape reduce_axis;
    while (iter != op->reduce_axis.end())
    {
        reduce_axis.push_back(*iter);
        ++iter;
    }
    simplify_reduce_shape(
        op->input_shape, reduce_axis, simplified_input_shape, simplified_reduce_axis);
    rank = simplified_input_shape.size();
    reduce_rank = simplified_reduce_axis.size();
    non_reduce_rank = rank - reduce_rank;

    get_reduce_strides(simplified_input_shape,
                       simplified_reduce_axis,
                       non_reduce_shape,
                       non_reduce_strides,
                       non_reduce_strides_in_input,
                       reduce_shape,
                       reduce_strides,
                       reduce_strides_in_input);
    nthreads = static_cast<uint32_t>(shape_size(non_reduce_shape));

    div_to_mul(reduce_strides, reduce_strides_magic, reduce_strides_shift);
    div_to_mul(non_reduce_strides, non_reduce_strides_magic, non_reduce_strides_shift);
    reduce_count = static_cast<uint32_t>(shape_size(reduce_shape));
}

string SoftmaxStridesBackOne::codegen_function_name()
{
    // kernel_name is used to check if the cuda kernel has been previously compiled
    std::stringstream kernel_name;
    kernel_name << "cuda"
                << "_softmax"
                << "_" << join(op->dtypes, "_") << "_i_" << join(op->input_shape, "_") << "_axis_"
                << join(op->reduce_axis, "_");
    return kernel_name.str();
}

LanguageUnit_p SoftmaxStridesBackOne::codegen_function_definition()
{
    create_ptr(LanguageUnit, cw, codegen_function_name());
    LanguageUnit& writer = *cw;

    auto get_reduce_input_lambda = [&]() {
        collective_coordinate_transform_helper(writer,
                                               "reduce_idx",
                                               "reduce_strides",
                                               "reduce_strides_magic",
                                               "reduce_strides_shift",
                                               "reduce_strides_in_input",
                                               "reduce_coordinate",
                                               reduce_rank,
                                               true,
                                               "reduce_input_index");
        writer << "input_idx = reduce_input_index + non_reduce_input_index;\n";
        writer << "input_i = load(in, input_idx);\n";
    };

    writer << "extern \"C\" __global__ void " << writer.symbol << "(" << op->dtypes[0] << "* in, "
           << op->dtypes[1] << "* out)\n";

    auto stable_sum_lambda = [&]() {
        writer << "input_i = exp(input_i - r_max);\n";
        writer << "y = input_i - c;\n";
        writer << "t = r_sum + y;\n";
        writer << "c = (t - r_sum) - y;\n";
        writer << "r_sum = t;\n";
    };

    auto max_lambda = [&]() { writer << "r_max = r_max > input_i ? r_max : input_i;\n"; };

    auto divide_lambda = [&]() {
        writer << "input_i = exp(input_i - r_max) / r_sum;\n";
        writer << "out[input_idx] = input_i;\n";
    };

    uint32_t block_size_x = 1;
    while ((block_size_x << 1) <= fmin(512, reduce_count))
    {
        block_size_x <<= 1;
    }

    const uint32_t WARPSIZE = 32;
    writer.block_begin();
    {
        writer << "uint32_t reduce_count = " << reduce_count << ";\n";
        auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << "int " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        };

        writer << expand_vector_uint32("non_reduce_strides", non_reduce_strides);
        writer << expand_vector_uint32("non_reduce_strides_in_input", non_reduce_strides_in_input);
        writer << expand_vector_uint32("reduce_shape", reduce_shape);
        writer << expand_vector_uint32("reduce_strides_in_input", reduce_strides_in_input);

        writer << expand_vector_int("reduce_strides_magic", reduce_strides_magic);
        writer << expand_vector_int("reduce_strides_shift", reduce_strides_shift);
        writer << expand_vector_int("non_reduce_strides_magic", non_reduce_strides_magic);
        writer << expand_vector_int("non_reduce_strides_shift", non_reduce_strides_shift);

        writer << "extern __shared__ " << op->dtypes[1] << " sdata[];\n";
        if (non_reduce_rank > 0)
        {
            writer << "uint32_t bid = blockIdx.x;\n";
            collective_coordinate_transform_helper(writer,
                                                   "bid",
                                                   "non_reduce_strides",
                                                   "non_reduce_strides_magic",
                                                   "non_reduce_strides_shift",
                                                   "non_reduce_strides_in_input",
                                                   "non_reduce_coordinate",
                                                   non_reduce_rank,
                                                   true,
                                                   "non_reduce_input_index");
        }
        writer << "uint32_t tid = threadIdx.x;\n";
        writer << "uint32_t step = blockDim.x;\n";
        writer << "uint32_t input_idx;\n";
        writer << "uint32_t reduce_idx = tid;\n";
        writer << op->dtypes[1] << " r_max;\n";
        writer << op->dtypes[1] << " input_i;\n";

        // find max
        writer.block_begin();
        {
            get_reduce_input_lambda();
            writer << "r_max = input_i;\n";
            writer << "reduce_idx += step;\n";
        }
        writer.block_end();
        writer << "while (reduce_idx + 7 * step < reduce_count)\n";
        writer.block_begin();
        {
            for (int i = 0; i < 8; i++)
            {
                writer.block_begin();
                get_reduce_input_lambda();
                max_lambda();
                writer << "reduce_idx += step;\n";
                writer.block_end();
            }
        }
        writer.block_end();

        writer << "while (reduce_idx < reduce_count)\n";
        writer.block_begin();
        {
            writer.block_begin();
            get_reduce_input_lambda();
            max_lambda();
            writer << "reduce_idx += step;\n";
            writer.block_end();
        }
        writer.block_end();
        // reduction max
        // accumulate WARPSIZE = 32 threads for each warp
        for (int i = (WARPSIZE >> 1); i >= 1; i >>= 1)
        {
            if (block_size_x > i)
            {
                writer << "input_i = __shfl_down_sync(0xffffffff, r_max, " << i << ", " << WARPSIZE
                       << ");\n";
                max_lambda();
            }
        }
        if (block_size_x > WARPSIZE)
        {
            writer << "uint32_t lane_idx = threadIdx.x & " << WARPSIZE - 1 << ";\n";
            writer << "uint32_t warp_idx = threadIdx.x >> 5;\n";
            writer << "if(lane_idx == 0)\n";
            writer.block_begin();
            {
                writer << "sdata[warp_idx] = r_max;\n";
            }
            writer.block_end();
            writer << "__syncthreads();\n";

            uint32_t num_of_warp = block_size_x >> 5;
            writer << "if(tid < " << num_of_warp << ")\n";
            writer.block_begin();
            {
                writer << "r_max = sdata[tid];\n";
            }
            writer.block_end();
            //accumulate WARPSIZE threads
            for (int i = (WARPSIZE >> 1); i >= 1; i >>= 1)
            {
                if (num_of_warp > i)
                {
                    // Todo(wenxh): __shfl_down_sync needs at least cuda 9.0
                    writer << "input_i = __shfl_down_sync(0xffffffff, r_max, " << i << ", "
                           << WARPSIZE << ");\n";
                    max_lambda();
                }
            }
        }
        // save and broadcast
        writer << "if(tid == 0)\n";
        writer.block_begin();
        {
            writer << "sdata[0] = r_max;\n";
            ;
        }
        writer.block_end();
        writer << "__syncthreads();\n";
        writer << "r_max = sdata[0];\n";

        //exp and sum , https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        writer << op->dtypes[1] << " r_sum = 0;\n";
        writer << op->dtypes[1] << " c = 0;\n";
        writer << op->dtypes[1] << " y;\n";
        writer << op->dtypes[1] << " t;\n";
        writer << "reduce_idx = tid;\n";
        writer << "while (reduce_idx + 7 * step < reduce_count)\n";
        writer.block_begin();
        {
            for (int i = 0; i < 8; i++)
            {
                writer.block_begin();
                get_reduce_input_lambda();
                stable_sum_lambda();
                writer << "reduce_idx += step;\n";
                writer.block_end();
            }
        }
        writer.block_end();

        writer << "while (reduce_idx < reduce_count)\n";
        writer.block_begin();
        {
            writer.block_begin();
            get_reduce_input_lambda();
            stable_sum_lambda();
            writer << "reduce_idx += step;\n";
            writer.block_end();
        }
        writer.block_end();

        // reduction sum
        // accumulate WARPSIZE = 32 threads for each warp
        for (int i = (WARPSIZE >> 1); i >= 1; i >>= 1)
        {
            if (block_size_x > i)
            {
                writer << "r_sum += __shfl_down_sync(0xffffffff, r_sum, " << i << ", " << WARPSIZE
                       << ");\n";
            }
        }
        if (block_size_x > WARPSIZE)
        {
            writer << "if(lane_idx == 0)\n";
            writer.block_begin();
            {
                writer << "sdata[warp_idx] = r_sum;\n";
            }
            writer.block_end();
            writer << "__syncthreads();\n";

            uint32_t num_of_warp = block_size_x >> 5;
            writer << "if(tid < " << num_of_warp << ")\n";
            writer.block_begin();
            {
                writer << "r_sum = sdata[tid];\n";
            }
            writer.block_end();
            //accumulate WARPSIZE = 32 threads
            for (int i = (WARPSIZE >> 1); i >= 1; i >>= 1)
            {
                if (num_of_warp > i)
                {
                    writer << "r_sum += __shfl_down_sync(0xffffffff, r_sum, " << i << ", "
                           << WARPSIZE << ");\n";
                }
            }
        }
        // save and broadcast
        writer << "__syncthreads();\n";
        writer << "if(tid == 0)\n";
        writer.block_begin();
        {
            writer << "sdata[0] = r_sum;\n";
            ;
        }
        writer.block_end();
        writer << "__syncthreads();\n";
        writer << "r_sum = sdata[0];\n";

        // divide
        writer << "reduce_idx = tid;\n";
        writer << "while (reduce_idx + 7 * step < reduce_count)\n";
        writer.block_begin();
        {
            for (int i = 0; i < 8; i++)
            {
                writer.block_begin();
                get_reduce_input_lambda();
                divide_lambda();
                writer << "reduce_idx += step;\n";
                writer.block_end();
            }
        }
        writer.block_end();

        writer << "while (reduce_idx < reduce_count)\n";
        writer.block_begin();
        {
            writer.block_begin();
            get_reduce_input_lambda();
            divide_lambda();
            writer << "reduce_idx += step;\n";
            writer.block_end();
        }
        writer.block_end();
    }
    writer.block_end();
    return cw;
}

LanguageUnit_p SoftmaxStridesBackOne::codegen_function_call()
{
    create_ptr(LanguageUnit, plu, codegen_function_name() + "_call");
    auto& lu = *plu;

    uint32_t block_size_x = 1;
    while ((block_size_x << 1) <= fmin(512, reduce_count))
    {
        block_size_x <<= 1;
    }
    uint32_t shared_data_bytes = block_size_x * static_cast<uint32_t>(op->dtypes[0].size());
    uint32_t aligned_grid_size_x = nthreads;

    lu << codegen_function_name() << "<<<dim3(" << aligned_grid_size_x << ", " << 1 << ", " << 1
       << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << shared_data_bytes << ", "
       << 0 << ">>>"
       << "(" << join(op->arg_names, ", ") << ", " << join(op->out_names, ", ") << ");\n";
    return plu;
}