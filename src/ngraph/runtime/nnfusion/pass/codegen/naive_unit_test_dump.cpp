// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/pass/codegen/naive_unit_test_dump.hpp"

using namespace ngraph::runtime::nnfusion::codegen;

bool NaiveUnitTestDump::run(shared_ptr<IntermediateOP>& inter_op)
{
    shared_ptr<CodeGenOP> cop = static_pointer_cast<CodeGenOP>(inter_op);
    if (cop != nullptr && cop->isCodeGened)
    {
        string filename = cop->codegen_function_name() + "_test.cu";
        shared_ptr<LanguageUnit> codewriter(new LanguageUnit(filename));
        auto& cw = *codewriter;
        cw << cop->test_call_unit->collect_required_code() << "\n";
        cw << "int main()";
        cw.block_begin();
        cw << cop->test_call_unit->get_code();
        cw << "return 0;\n";
        cw.block_end();

        // Save the function
        ofstream out(filename);
        out << cw.get_code();
        out.close();
    }

    return true;
}