// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegenop.hpp"

using namespace ngraph::runtime::nnfusion::codegen::cuda;

shared_ptr<LanguageUnit> CudaCodeGenOP::codegen_test()
{
    shared_ptr<LanguageUnit> _lu(new LanguageUnit(codegen_test_name()));
    auto& writer = *_lu;

    // extern "C" void cuda_some_op_test(type* in0, ..., type* out0, ....)
    //{
    //   call_global_func<<<(1, 1, 1), (1, 1, 1), 0, 0>>(in0, ..., out0, ...)
    //}

    assert_nullptr(inter_op);

    auto& arg = inter_op->args;
    auto& out = inter_op->out;

    writer << "extern \"C\" int " << _lu->get_symbol() << "(";
    for (size_t i = 0; i + 1 < arg.size(); i++)
    {
        writer << arg[i].get_type() << "* " << arg[i].get_name() << "_host, ";
    }
    if (!arg.empty())
    {
        writer << arg.back().get_type() << "* " << arg.back().get_name();
        if (!out.empty())
            writer << "_host, ";
    }

    for (size_t i = 0; i + 1 < out.size(); i++)
    {
        writer << out[i].get_type() << "* " << out[i].get_name() << "_host, ";
    }
    if (!out.empty())
    {
        writer << out.back().get_type() << "* " << out.back().get_name() << "_host";
    }
    writer << ")\n";

    writer.block_begin();
    {
        for (size_t i = 0; i < arg.size(); i++)
        {
            auto& tensor = arg[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";

            writer << "cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name() << "_host, "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyHostToDevice);\n";
        }
        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << tensor.get_type() << "* " << tensor.get_name() << ";\n"
                   << "cudaMalloc((void**)&" << tensor.get_name() << "," << tensor.get_size()
                   << " * " << tensor.get_element_type().size() << ");\n";
        }

        assert_nullptr(this->call_unit);
        writer << this->call_unit->get_code();

        for (size_t i = 0; i < out.size(); i++)
        {
            auto& tensor = out[i];
            writer << "cudaMemcpy(" << tensor.get_name() << "_host, " << tensor.get_name() << ", "
                   << tensor.get_size() << " * " << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyDeviceToHost);\n";
        }

        writer << "return 0;\n";
    }
    writer.block_end();

    writer << "\n";

    writer << "extern \"C\" int " << _lu->get_symbol() << "_simple(void** args)";

    writer.block_begin();
    {
        writer << "return " << _lu->get_symbol() << "(";
        for (size_t i = 0; i + 1 < arg.size() + out.size(); i++)
        {
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)"
                   << "args[" << i << "], ";
        }
        if (arg.size() + out.size() > 0)
        {
            int i = arg.size() + out.size() - 1;
            string type = i < arg.size()
                              ? arg[i].get_type()
                              : (i - arg.size() < out.size() ? out[i - arg.size()].get_type() : "");
            writer << "(" << type << "*)"
                   << "args[" << i << "]";
        }
        writer << ");\n";
    }
    writer.block_end();
    return _lu;
}