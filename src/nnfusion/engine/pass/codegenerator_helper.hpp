// Microsoft(c)2019, NNFusion Team

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"

namespace nnfusion
{
    namespace codegenerator
    {
        nnfusion::LanguageUnit_p extern_function(nnfusion::LanguageUnit_p lu);
        nnfusion::LanguageUnit_p extern_variable(nnfusion::LanguageUnit_p lu);

        class FunctionFile : public LanguageUnit
        {
        public:
            using FunctionFile_p = shared_ptr<FunctionFile>;
            static FunctionFile_p convert_from(FunctionUnit_p fu);
            FunctionFile(string extern_declare, LanguageUnit_p file_context);
            FunctionFile() { extern_declare = ""; }
            string get_extern_declare() { return extern_declare; };
            void save_file();
            void merge_from(FunctionFile_p func);

        private:
            string extern_declare;

            /*
            Origninal FunctionUnit includes:
            LanguageUnit_p name_unit;
            LanguageUnit_p signature_unit; // void (float* input0, float* input1, float* output0)
            LanguageUnit_p body_unit;
            LanguageUnit_p dep_unit;
            LanguageUnit_p call_unit; // (tensor1, tensor2, tensor3)
            LanguageUnit_p comment_unit

            Based on the profiler's codegen:
                1. put dep_unit into extern; ------------------> in to function.cu
                2. put sig & body into single file; -----------^
                3. generate extern function def from sig; -----> Replacing original funciton def

            In the cmakelist files:
                1. Compiling <function>.cu in to objects <function>;
                2. Compiling nnfusion_rt.cu/hpp into objects "nnfusion_rt";
                3. Link them together.
            */

            //\todo: IS THIS WAY GENERAL? Any C-series(*cc) compiler will support this way.
        };

        using FunctionFile_p = FunctionFile::FunctionFile_p;
    } // namespace codegenerator
} // namespace nnfusion