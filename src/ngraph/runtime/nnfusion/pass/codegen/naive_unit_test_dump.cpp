// Microsoft (c) 2019, Wenxiang Hu
#include "naive_unit_test_dump.hpp"
#include "../../cuda/cuda_langunit.hpp"

using namespace nnfusion::codegen;

bool NaiveUnitTestDump::run(ir::Operator_p& inter_op)
{
    auto cop = static_pointer_cast<ir::Function>(inter_op);
    if (cop != nullptr && cop->is_codegened())
    {
        cop->dep_unit->require(cuda::declaration::typedef_int);
        string filename = cop->codegen_function_name() + "_test.cu";
        LanguageUnit_p codewriter(new LanguageUnit(filename));
        auto& cw = *codewriter;
        cw << cop->test_unit->collect_code();
        // Save the function
        ofstream out(filename);
        out << cw.get_code();
        out.close();
    }

    return true;
}