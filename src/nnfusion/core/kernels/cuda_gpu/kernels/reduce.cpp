// Microsoft (c) 2019, NNFusion Team

#include "reduce.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Max",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER("Max",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Max>)

REGISTER_KERNEL_EMITTER("Min",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER("Min",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Min>)

REGISTER_KERNEL_EMITTER("Product",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER("Product",                                                 // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Multiply>)

REGISTER_KERNEL_EMITTER("Sum",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::Reduce<nnfusion::op::Add>)

REGISTER_KERNEL_EMITTER("Sum",                                                     // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_lib"), // attrs
                        cuda::ReduceMemcpy<nnfusion::op::Add>)
