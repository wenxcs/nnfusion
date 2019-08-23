// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Reshape : public CudaEmitter
            {
            public:
                Reshape(shared_ptr<KernelContext> ctx);

                //LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                //void set_launch_config() override;

            protected:
                ngraph::Shape arg_shape;
                size_t arg_rank;
                ngraph::Shape result_shape;
                ngraph::AxisVector input_order;
                shared_ptr<ngraph::op::Reshape> reshape;
                bool is_memcpy;
                bool is_noop;
            };

            class Reshape2D : public Reshape
            {
            public:
                Reshape2D(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                void set_launch_config() override;

            private:
                uint32_t block_size;
                NVShape input_strides;
                NVShape output_strides;
                NVShape trans_strides;
            };

            class Reshape3D : public Reshape
            {
            public:
                Reshape3D(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                void set_launch_config() override;

            private:
                std::vector<uint32_t> block_size;
                uint32_t block_size_x;
                NVShape input_strides, output_strides, trans_strides;
            };

            class ReshapehD : public Reshape
            {
            public:
                ReshapehD(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                void set_launch_config() override;

            private:
                uint32_t block_size_x;
                NVShape input_strides;
                NVShape output_strides;
                NVShape trans_strides;
            };

            class ReshapeMemcpy : public CudaLibEmitter
            {
            public:
                ReshapeMemcpy(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                ngraph::Shape arg_shape;
                size_t arg_rank;
                ngraph::Shape result_shape;
                ngraph::AxisVector input_order;
                shared_ptr<ngraph::op::Reshape> reshape;
                bool is_memcpy;
                bool is_noop;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion