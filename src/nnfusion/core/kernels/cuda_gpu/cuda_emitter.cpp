// Microsoft (c) 2019, NNFusion Team
#include "cuda_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

LanguageUnit_p cuda::CudaEmitter::emit_function_call()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_call"));
    auto& lu = *_lu;
    vector<string> names;
    names.insert(names.end(), m_context->input_names.begin(), m_context->input_names.end());
    names.insert(names.end(), m_context->output_names.begin(), m_context->output_names.end());
    lu << get_function_name() << "<<<dim3(" << m_gridDim.x << ", " << m_gridDim.y << ", "
       << m_gridDim.z << "), dim3(" << m_blockDim.x << ", " << m_blockDim.y << ", " << m_blockDim.z
       << ") >>>"
       << "(" << join(names, ", ") << ");\n";

    return _lu;
}

LanguageUnit_p cuda::CudaEmitter::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i].get_element_type().c_type_string() << "* ";
        ss << "input" << i;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i].get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    lu << "extern \"C\" __global__  void " << get_function_name() << "(" << join(params, ", ")
       << ")";
    return _lu;
}

// LanguageUnit_p CudaEmitter::emit_test()
// {
//     create_ptr(LanguageUnit, _lu, emit_test_name());
//     auto& writer = *_lu;

//     // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
//     //{
//     //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
//     //}

//     enforce_not_nullptr(op);

//     auto& arg = op->args;
//     auto& out = op->out;

//     writer << "extern \"C\" int " << _lu->get_symbol() << "(";
//     for (size_t i = 0; i + 1 < arg.size(); i++)
//     {
//         writer << arg[i].get_type() << "* " << arg[i].get_name() << "_host, ";
//     }
//     if (!arg.empty())
//     {
//         writer << arg.back().get_type() << "* " << arg.back().get_name();
//         if (!out.empty())
//             writer << "_host, ";
//     }

//     for (size_t i = 0; i + 1 < out.size(); i++)
//     {
//         writer << out[i].get_type() << "* " << out[i].get_name() << "_host, ";
//     }
//     if (!out.empty())
//     {
//         writer << out.back().get_type() << "* " << out.back().get_name() << "_host";
//     }
//     writer << ")\n";

//     writer.block_begin();
//     {
//         if (dep_unit->local_symbol.count("declaration::global_cublas_handle") > 0)
//         {
//             writer << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
//         }

//         if (dep_unit->local_symbol.count("declaration::global_cudnn_handle") > 0)
//         {
//             writer << "CUDNN_SAFE_CALL(cudnnCreate(&global_cudnn_handle));\n";
//         }

//         if (dep_unit->local_symbol.count("declaration::num_SMs") > 0)
//         {
//             writer << "CUDA_SAFE_CALL(cudaDeviceGetAttribute(&num_SMs, "
//                       "cudaDevAttrMultiProcessorCount, 0));\n";
//         }

//         for (size_t i = 0; i < arg.size(); i++)
//         {
//             auto& tensor = arg[i];
//             writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
//                    << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
//                    << " * " << tensor.get_element_type().size() << ");\n";

//             writer << "cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name() << "_host, "
//                    << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
//                    << "cudaMemcpyHostToDevice);\n";
//         }
//         for (size_t i = 0; i < out.size(); i++)
//         {
//             auto& tensor = out[i];
//             writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
//                    << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
//                    << " * " << tensor.get_element_type().size() << ");\n";
//         }

//         enforce_not_nullptr(this->call_unit);
//         writer << this->call_unit->get_code();

//         for (size_t i = 0; i < out.size(); i++)
//         {
//             auto& tensor = out[i];
//             writer << "cudaMemcpy(" << tensor.get_name() << "_host, " << tensor.get_name() << ", "
//                    << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
//                    << "cudaMemcpyDeviceToHost);\n";
//         }

//         for (size_t i = 0; i < arg.size(); i++)
//         {
//             auto& tensor = arg[i];
//             writer << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
//         }

//         for (size_t i = 0; i < out.size(); i++)
//         {
//             auto& tensor = out[i];
//             writer << "CUDA_SAFE_CALL(cudaFree(" << tensor.get_name() << "));\n";
//         }

//         if (dep_unit->local_symbol.count("declaration::global_cublas_handle") > 0)
//         {
//             writer << "CUBLAS_SAFE_CALL(cublasDestroy(global_cublas_handle));\n";
//         }

//         if (dep_unit->local_symbol.count("declaration::global_cudnn_handle") > 0)
//         {
//             writer << "CUDNN_SAFE_CALL(cudnnDestroy(global_cudnn_handle));\n";
//         }
//         writer << "return 0;\n";
//     }
//     writer.block_end();

//     writer << "\n";

//     writer << "extern \"C\" int " << _lu->get_symbol() << "_simple(void** args)";

//     writer.block_begin();
//     {
//         writer << "return " << _lu->get_symbol() << "(";
//         for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
//         {
//             string type = i < arg.size()
//                               ? arg[i].get_type()
//                               : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
//             writer << "(" << type << "*)"
//                    << "args[" << i << "], ";
//         }
//         if (arg.size() + out.size() > 0)
//         {
//             int i = arg.size() + out.size() - 1;
//             string type = i < arg.size()
//                               ? arg[i].get_type()
//                               : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
//             writer << "(" << type << "*)"
//                    << "args[" << i << "]";
//         }
//         writer << ");\n";
//     }
//     writer.block_end();
//     return _lu;
// }