// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

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
                std::vector<TensorWrapper> out;

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
            public:
                class CodeGenOPDep
                {
                    set<string> headers;
                    set<shared_ptr<CodeGenOP>> codegen_op;
                };

            protected:
                // Common Data
                bool isCodeGened;
                shared_ptr<IntermediateOP> inter_op;
                shared_ptr<codegen::CodeWriter> definition_writer;
                shared_ptr<codegen::CodeWriter> call_writer;
                shared_ptr<codegen::CodeWriter> source_writer;
                shared_ptr<codegen::CodeWriter> dep_writer;
                shared_ptr<codegen::CodeWriter> test_writer;
                unique_ptr<CodeGenOPDep> _dep;

                // mapping: kernel name -> kernel definition
                static unordered_map<string, shared_ptr<codegen::CodeWriter>> definition_pool;

            public:
                // Get the property of some CodeGenOP
                virtual string codegen_function_name() = 0;
                virtual string codegen_source_name() = 0;

                // Interface for Generate code pieces
                virtual shared_ptr<codegen::CodeWriter> codegen_dependency() = 0;
                virtual shared_ptr<codegen::CodeWriter> codegen_function_definition() = 0;
                virtual shared_ptr<codegen::CodeWriter> codegen_function_call() = 0;
                virtual shared_ptr<codegen::CodeWriter> codegen_test() = 0;

                virtual shared_ptr<codegen::CodeWriter> codegen_source();

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