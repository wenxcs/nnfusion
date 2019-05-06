// Microsoft (c) 2019, Wenxiang
#include "dot.hpp"

using namespace nnfusion::cuda;

Dot::Dot(ir::Operator_p inter_op)
    : CudaFunction(inter_op)
{
    _op = static_pointer_cast<ir::Dot>(inter_op);
    enforce_not_nullptr(_op) << "Wrong Operator was given.";
}

string Dot::codegen_function_name()
{
    std::stringstream ss;
    ss << "cublas_dot_op"
       << "_dtype_" << _op->dtype.c_type_string() << "_reduction_axes_count_" << _op->reduction_axes
       << "_i_" << join(_op->arg0_shape, "_") << "_i_" << join(_op->arg1_shape, "_");
    return ss.str();
}

string Dot::codegen_test_name()
{
    return codegen_function_name() + "_test";
}

LanguageUnit_p Dot::codegen_function_definition()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name());
    auto& lu = *_lu;
    lu.require(macro::CUBLAS_SAFE_CALL);

    // Generate the signture
    // void cublas_dot_op_dtype_..._reduction_axes_count_...(float* in0, float*in1, float* out)
    lu << "void " << lu.symbol << "(cublasHandle_t handle, " << _op->args[0].get_type() << "* in0, "
       << _op->out[0].get_type() << "* in1, " << _op->dtype.c_type_string() << "* out)\n";
    lu.block_begin();
    {
        // case 1: Scalar * Tensor
        if (_op->arg0_shape.empty() || _op->arg1_shape.empty())
        {
            auto& second = (_op->arg0_shape.empty() ? _op->arg1_shape : _op->arg0_shape);
            size_t count = ngraph::shape_size(second);

            string firstarg = (_op->arg0_shape.empty() ? "in0" : "in1");
            string secondarg = (_op->arg0_shape.empty() ? "in1" : "in0");

            lu << "CUBLAS_SAFE_CALL(cublasScopy(handle, " << count << ", static_cast<const float*>("
               << firstarg << "), 1, static_cast<float*>(out),1));\n";
            lu << "CUBLAS_SAFE_CALL(cublasSscal(handle, " << count << ", static_cast<const float*>("
               << secondarg << ", static_cast<float*>(out),1));\n";
        }
        // case 2: 1d Dot
        else if ((_op->arg0_shape.size() == _op->arg1_shape.size()) &&
                 (_op->arg0_shape.size() == _op->reduction_axes))
        {
            for (int i = 0; i < _op->arg0_shape.size(); i++)
            {
                if (_op->arg0_shape[i] != _op->arg1_shape[i])
                {
                    std::vector<std::string> arg_vec{"arg0", "arg1"};
                    std::vector<ngraph::Shape> shape_vec{_op->arg0_shape, _op->arg1_shape};

                    std::stringstream ss_err;
                    ss_err << ngraph::join(arg_vec) << " with " << ngraph::join(shape_vec)
                           << " respectively, at Node " << _op->node->get_name()
                           << ", do not match for dot op";

                    enforce(false) << ss_err.str();
                }
            }

            size_t count = ngraph::shape_size(_op->arg0_shape);
            lu << "CUBLAS_SAFE_CALL(cublasSdot(handle, " << count
               << ", static_cast<const float*>(in0), 1, static_cast<const float*>(in0), 1, "
                  "static_cast<float*>(out)));\n";
        }
        // matrix * vector
        else if ((_op->arg0_shape.size() == 2) && (_op->arg1_shape.size() == 1) &&
                 (_op->reduction_axes == 1))
        {
            /*
            lu << "const float alpha = 1.0;\n const float beta = 0;\n";
              <<"CUBLAS_SAFE_CALL(
                    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
              \n ";
              */
            lu << "CUBLAS_SAFE_CALL(cublasSgemv(handle, "
               << "CUBLAS_OP_T, " << _op->arg0_shape[1] << ", " << _op->arg0_shape[0] << ", "
               << " &alpha,"
               << " static_cast<const float*>(in0)," << _op->arg0_shape[1] << ", "
               << " static_cast<const float*>(in1),"
               << " 1,"
               << " &beta,"
               << " static_cast<float*>(out),"
               << " 1));\n";
            /*
      lu << "CUBLAS_SAFE_CALL(
                   cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        \n ";
        */
        }
        else
        {
            size_t axes_for_m_count = _op->arg0_shape.size() - _op->reduction_axes;
            size_t axes_for_n_count = _op->arg1_shape.size() - _op->reduction_axes;
            size_t axes_for_k_count = _op->reduction_axes;
            size_t m = 1;
            size_t n = 1;
            size_t k = 1;

            // check if input and output size correct
            // check and calculate k for arg0 and arg1
            size_t arg0_k_idx = axes_for_m_count; // first axe in arg0 for k
            size_t arg1_k_idx = 0;                // first axe in arg1 for k

            for (size_t i = 0; i < axes_for_k_count; i++)
            {
                k *= _op->arg0_shape[arg0_k_idx];
                if (_op->arg0_shape[arg0_k_idx++] != _op->arg1_shape[arg1_k_idx++])
                {
                    std::vector<std::string> arg_vec{"arg0", "arg1"};
                    std::vector<ngraph::Shape> shape_vec{_op->arg0_shape, _op->arg1_shape};

                    std::stringstream ss_err;
                    ss_err << ngraph::join(arg_vec) << " with " << ngraph::join(shape_vec)
                           << " respectively, at Node " << _op->node->get_name()
                           << ", do not match for dot op";

                    enforce(false) << ss_err.str();
                }
            }
            // check and calculate m for arg0 and out
            size_t arg0_m_idx = 0; // first axe in arg0 for m
            size_t out_m_idx = 0;  // first axe in out for m
            for (size_t i = 0; i < axes_for_m_count; i++)
            {
                m *= _op->arg0_shape[arg0_m_idx];
                if (_op->arg0_shape[arg0_m_idx++] != _op->out_shape[out_m_idx++])
                {
                    std::vector<std::string> arg_vec{"arg0", "output"};
                    std::vector<ngraph::Shape> shape_vec{_op->arg0_shape, _op->out_shape};

                    std::stringstream ss_err;
                    ss_err << ngraph::join(arg_vec) << " with " << ngraph::join(shape_vec)
                           << " respectively, at Node " << _op->node->get_name()
                           << ", do not match for dot op";

                    enforce(false) << ss_err.str();
                }
            }
            // check and calculate n for arg1 and out
            size_t arg1_n_idx = axes_for_k_count; // first axe in arg1 for n
            size_t out_n_idx = axes_for_m_count;  // first axe in arg1 for n
            for (size_t i = 0; i < axes_for_n_count; i++)
            {
                n *= _op->arg1_shape[arg1_n_idx];
                if (_op->arg1_shape[arg1_n_idx++] != _op->out_shape[out_n_idx++])
                {
                    std::vector<std::string> arg_vec{"arg1", "output"};
                    std::vector<ngraph::Shape> shape_vec{_op->arg1_shape, _op->out_shape};

                    std::stringstream ss_err;
                    ss_err << ngraph::join(arg_vec) << " with " << ngraph::join(shape_vec)
                           << " respectively, at Node " << _op->node->get_name()
                           << ", do not match for dot op";

                    enforce(false) << ss_err.str();
                }
            }

            lu << "const float alpha = 1.0;\n const float beta = 0;\n";

            lu << "CUBLAS_SAFE_CALL(cublasSgemm(handle,"
               << " CUBLAS_OP_N,"
               << " CUBLAS_OP_N,"
               << " " << n << ","
               << " " << m << ","
               << " " << k << ","
               << " &alpha,"
               << " static_cast<const float*>(in1),"
               << " " << n << ","
               << " static_cast<const float*>(in0),"
               << " " << k << ","
               << " &beta,"
               << " static_cast<float*>(out),"
               << " " << n << "));\n";
        }
    }
    lu.block_end();
    return _lu;
}

LanguageUnit_p Dot::codegen_function_call()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name() + "_call");
    auto& lu = *_lu;

    lu << codegen_function_name() << "(global_cublas_handle, " << _op->arg_names[0] << ", "
       << _op->arg_names[1] << ", " << _op->out_names[0] << ");\n";

    return _lu;
}

LanguageUnit_p Dot::codegen_dependency()
{
    create_ptr(LanguageUnit, _lu, codegen_function_name() + "_dep");
    auto& lu = *_lu;
    lu.require(header::cublas);
    lu.require(header::stdexcept);
    lu.require(header::sstream);
    lu.require(declaration::global_cublas_handle);
    return _lu;
}

CudaFunction_p Dot::codegen(ir::Operator_p inter_op)
{
    Dot_p cop(new Dot(inter_op));
    LOG_INFO << "Codegen for Dot function:" << cop->codegen_function_name() << endl;
    return cop;
}