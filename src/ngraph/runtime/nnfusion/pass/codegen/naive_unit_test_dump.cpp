// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/pass/codegen/naive_unit_test_dump.hpp"

using namespace ngraph::runtime::nnfusion::codegen;

bool NaiveUnitTestDump::run(shared_ptr<IntermediateOP>& inter_op)
{
    shared_ptr<CodeGenOP> cop = static_pointer_cast<CodeGenOP>(inter_op);
    if (cop != nullptr && cop->isCodeGened)
    {
        shared_ptr<LanguageUnit> common_typdef(new LanguageUnit("common_typedef"));
        auto& ss = *common_typdef;

        // add modern type definitions
        ss << "typedef signed char int8_t;\n";
        ss << "typedef signed short int16_t;\n";
        ss << "typedef signed int int32_t;\n";
        ss << "typedef signed long int int64_t;\n";
        ss << "typedef unsigned char uint8_t;\n";
        ss << "typedef unsigned short uint16_t;\n";
        ss << "typedef unsigned int uint32_t;\n";
        ss << "typedef unsigned long int uint64_t;\n";

        cop->dep_unit->require(common_typdef);
        string filename = cop->codegen_function_name() + "_test.cu";
        shared_ptr<LanguageUnit> codewriter(new LanguageUnit(filename));
        auto& cw = *codewriter;
        cw << cop->test_unit->collect_code();
        // Save the function
        ofstream out(filename);
        out << cw.get_code();
        out.close();
    }

    return true;
}