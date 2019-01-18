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
                // Link to compiled Op
                bool isTranslated;
                // Store the tensor wrappers
                // <todo:wenxh>
            public:
                const ngraph::Node* n;
                std::vector<TensorWrapper> args;
                std::vector<TensorWrapper> out;

            public:
                // virtual std::shared_ptr<IntermediateOP> translate(TRANS_ARGS) = 0; const std::string& get_name() const;
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

            public:
                virtual std::shared_ptr<CodeGenOP> codegen(IntermediateOP* op) = 0;
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