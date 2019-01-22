// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_languageunit.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            // Store the caculated intermediated data
            class IntermediateOP
            {
            protected:
                // Common Data
                std::string m_name;
                bool isTranslated;

            public:
                shared_ptr<ngraph::Node> node;
                std::vector<TensorWrapper> args;
                std::vector<string> arg_names;
                std::vector<TensorWrapper> out;
                std::vector<string> out_names;

                IntermediateOP();
                IntermediateOP(shared_ptr<Node> node);
                ~IntermediateOP(){};
            };

            class GroupOp : public IntermediateOP
            {
            private:
                // std::set<IntermediateOP> operators;
            };

            // Generate Solution files
            class CodeGenOP : public IntermediateOP
            {
            protected:
                // Common Data
                bool isCodeGened;
                shared_ptr<IntermediateOP> inter_op;
                shared_ptr<LanguageUnit> definition_unit;
                shared_ptr<LanguageUnit> call_unit;
                shared_ptr<LanguageUnit> source_unit;
                shared_ptr<LanguageUnit> dep_unit;
                shared_ptr<LanguageUnit> test_unit;
                shared_ptr<LanguageUnit> test_call_unit;

                // mapping: kernel name -> kernel definition
                static unordered_map<string, shared_ptr<LanguageUnit>> definition_pool;

            public:
                // Get the property of some CodeGenOP
                virtual string codegen_function_name() = 0;
                virtual string codegen_test_name() = 0;

            private:
                // Interface for Generate code pieces
                virtual shared_ptr<LanguageUnit> codegen_dependency() = 0;
                virtual shared_ptr<LanguageUnit> codegen_function_definition() = 0;
                virtual shared_ptr<LanguageUnit> codegen_function_call() = 0;
                virtual shared_ptr<LanguageUnit> codegen_test() = 0;
                virtual shared_ptr<LanguageUnit> codegen_test_call() = 0;

            public:
                virtual shared_ptr<LanguageUnit> codegen_source();

                CodeGenOP();
                CodeGenOP(shared_ptr<IntermediateOP> inter_op);
            };

            // Through JIT(NVRTC)
            class CompiledOP : public IntermediateOP
            {
            protected:
                std::string source;
                bool isCompiled;

            public:
                virtual std::shared_ptr<CompiledOP> compile() = 0;
            };
        }
    }
}