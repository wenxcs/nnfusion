// Microsoft (c) 2019, NNFusion Team
#include "cpu_helper.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion::kernels;

LanguageUnit_p cpu::get_eigen_math_kernel(const std::string& name,
                                          const std::string& math_kernel,
                                          size_t data_size,
                                          const std::vector<std::string>& data_types)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_inline_" + name;
    // TODO: handle data_types containing underline, like long_long
    // Output type should be ignore
    for (size_t i = 0; i < data_types.size() - 1; i++)
    {
        mangled_name += "-" + data_types[i];
    }
    mangled_name += "-" + std::to_string(data_size);
    shared_ptr<LanguageUnit> cw(new LanguageUnit(mangled_name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "inline void " << name << "_" << data_size << "(";
        for (size_t i = 0; i < num_inputs; ++i)
        {
            writer << data_types[i] << "* x" << i << ", ";
        }
        writer << data_types[num_inputs] << "* y0";
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            for (size_t i = 0; i < num_inputs; ++i)
            {
                writer << "Eigen::Map<Eigen::Array<" << data_types[i] << ", " << data_size
                       << ", 1>> in" << i << "(x" << i << ");\n";
            }
            writer << "Eigen::Map<Eigen::Array<" << data_types[num_inputs] << ", " << data_size
                   << ", 1>> out(y0);\n";
            writer << "out = " << math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}

LanguageUnit_p cpu::get_simd_math_kernel(const std::string& name,
                                         const std::string& math_kernel,
                                         size_t data_size,
                                         const std::vector<std::string>& data_types)
{
    NNFUSION_CHECK(std::count(name.begin(), name.end(), '-') == 0);
    std::string mangled_name = "declaration::function_def_inline_" + name;
    // TODO: handle data_types containing underline, like long_long
    // Output type should be ignore
    for (size_t i = 0; i < data_types.size() - 1; i++)
    {
        mangled_name += "-" + data_types[i];
    }
    shared_ptr<LanguageUnit> cw(new LanguageUnit(mangled_name));
    auto& writer = *cw;
    if (math_kernel.size())
    {
        auto num_inputs = data_types.size() - 1;
        writer << "inline __m256 " << name << "(";
        for (size_t i = 0; i < num_inputs - 1; ++i)
        {
            writer << "__m256 in" << i << ", ";
        }
        writer << "__m256 in" << num_inputs - 1;
        writer << ")\n";
        writer << "{\n";
        writer.indent++;
        {
            writer << "return " << math_kernel << ";\n";
        }
        writer.indent--;
        writer << "}\n";
    }
    return cw;
}
