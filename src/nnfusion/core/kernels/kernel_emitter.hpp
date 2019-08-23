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
            using Pointer = shared_ptr<KernelContext>;

            KernelContext(shared_ptr<ngraph::Node> node);

            // The node this OpKernel corresponds to
            shared_ptr<ngraph::Node> node;

            // The input tensor descriptions
            vector<nnfusion::TensorWrapper> inputs;

            // The output tensor descriptions
            vector<nnfusion::TensorWrapper> outputs;

            // The input tensor names
            vector<string> input_names;

            // The output tensor names
            vector<string> output_names;

            // The list of input and output data types
            vector<string> dtypes;

            // The number of gpu streaming multiprocessor
            uint32_t gpu_num_sm;
        };

        // OpKernel defines the interfaces of generating a specific computation kernel
        // for an operator
        class KernelEmitter
        {
        public:
            using Pointer = shared_ptr<KernelEmitter>;

            KernelEmitter(shared_ptr<KernelContext> ctx);

            KernelEmitter(shared_ptr<KernelContext> ctx, string kernel_type);

            // Emit entire source code
            virtual FunctionUnit_p get_or_emit_source();

            string get_kernel_type() { return m_kernel_type; }
            string get_function_name() { return this->m_kernel_name; }
            bool is_emitted() { return m_is_emitted; }
            // function declaration will be deduplicated only if the kernel function is
            // not static
            virtual bool is_static_function() { return false; }
            // The context for this kernel
            shared_ptr<KernelContext> m_context;

        protected:
            // Generate function name for this kernel, the default name is:
            // "op_name + args_shapes + data_type + device + custom_tag"
            LanguageUnit_p emit_function_name();

            // Emit the function body of a specific kernel for this operator
            // the order of function args is following the order of inputs/outputs
            // in KernelContext. The function signature looks like:
            // void fname(dtypes[0]* input0, dtypes[1]* input1, …, dtypes[k] *output0, …)
            virtual LanguageUnit_p emit_function_body() = 0;

            // Emit function signature
            virtual LanguageUnit_p emit_function_signature();

            // Emit the dependency of this kernel code
            // e.g., the cudnn convolution kernel depends on the cudnn lib,
            // thus it needs to add a header of "#include <cudnn>
            virtual LanguageUnit_p emit_dependency() = 0;

            // Emit function call
            virtual LanguageUnit_p emit_function_call();

            // Emit comments
            virtual LanguageUnit_p emit_comments();

            // A kernel only emits kernel code once
            bool m_is_emitted;

            // kernel type: e.g., CUDA, CUDNN, ROCM, CPU, etc.
            const string m_kernel_type;

            // kernel name.
            string m_kernel_name;

            // custom kernel tag
            string custom_tag;

            // mapping: kernel name -> kernel definition
            unordered_map<string, shared_ptr<FunctionUnit>> kernel_definitions;

            // Reserved for simplified representation
            nlohmann::json attr;

            // emitted function unit
            FunctionUnit_p m_function_unit;
        };
    } // namespace kernels
} // namespace nnfusion
