// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/common/tensorwrapper.hpp"

namespace nnfusion
{
    namespace kernels
    {
        class KernelContext
        {
        public:
            KernelContext(shared_ptr<ngraph::Node> node);

            // The node this OpKernel corresponds to
            shared_ptr<ngraph::Node> node;

            // The input tensor descriptions
            vector<TensorWrapper> inputs;

            // The output tensor descriptions
            vector<TensorWrapper> outputs;

            // The input tensor names
            vector<string> input_names;

            // The output tensor names
            vector<string> output_names;

            // The list of input and output data types
            vector<string> dtypes;
        };

        // OpKernel defines the interfaces of generating a specific computation kernel
        // for an operator
        class KernelEmitter
        {
        public:
            KernelEmitter(shared_ptr<KernelContext> ctx);

            KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type);

            string get_kernel_type() { return m_kernel_type; }
            // Generate function name for this kernel, the default name is:
            // "op_name + args_shapes + data_type + device + custom_tag"
            // Note that it's kernel developer's responsibility to avoid confilit
            string get_function_name();

            // Emit entire source code
            LanguageUnit_p emit_source();

            // Emit comments
            string emit_comments();

            bool is_emitted() { return m_is_emitted; }

            // Interfaces for generating the kernel code of an operator

            // Emit the function body of a specific kernel for this operator
            // the order of function args is following the order of inputs/outputs
            // in KernelContext. The function signature looks like:
            // void fname(dtypes[0]* input0, dtypes[1]* input1, …, dtypes[k] *output0, …)
            virtual LanguageUnit_p emit_function_body() = 0;

            // Emit the dependency of this kernel code
            // e.g., the cudnn convolution kernel depends on the cudnn lib,
            // thus it needs to add a header of "#include <cudnn>
            virtual LanguageUnit_p emit_dependency() = 0;

        public:
            // Emitted code units
            LanguageUnit_p dep_unit;
            LanguageUnit_p body_unit;
            LanguageUnit_p call_unit;
            LanguageUnit_p signature_unit;
            LanguageUnit_p test_unit;
            LanguageUnit_p test_call_unit;
            LanguageUnit_p source_unit;

            // The context for this kernel
            shared_ptr<KernelContext> m_context;


        protected:
            // Emit function call
            virtual LanguageUnit_p emit_function_call();

            // Emit function signature
            virtual LanguageUnit_p emit_function_signature();

            
            
            // A kernel only emits kernel code once
            bool m_is_emitted;

            // kernel type: e.g., CUDA, CUDNN, ROCM, CPU, etc.
            const string m_kernel_type;

            // custom kernel tag
            string custom_tag;

            // mapping: kernel name -> kernel definition
            unordered_map<string, LanguageUnit_p> kernel_definitions;

            // Reserved for simplified representation
            nlohmann::json attr;

            // Emitted code units
            string m_function_name;
        };
    } // namespace kernels
} // namespace nnfusion
