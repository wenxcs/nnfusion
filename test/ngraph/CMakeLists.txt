# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(NGRAPH_TESTS
    ngraph/algebraic_simplification.cpp
    ngraph/all_close_f.cpp
    ngraph/assertion.cpp
    ngraph/build_graph.cpp
    ngraph/builder_autobroadcast.cpp
    ngraph/constant_folding.cpp
    ngraph/control_dependencies.cpp
    ngraph/coordinate.cpp
    ngraph/copy.cpp
    ngraph/cpio.cpp
    ngraph/cse.cpp
    ngraph/element_type.cpp
    ngraph/file_util.cpp
    ngraph/graph_partition.cpp
    ngraph/includes.cpp
    ngraph/inliner.cpp
    ngraph/input_output_assign.cpp
    ngraph/nop_elimination.cpp
    ngraph/op.cpp
    ngraph/partial_shape.cpp
    ngraph/pass_liveness.cpp
    ngraph/pass_manager.cpp
    ngraph/pass_memory_layout.cpp
    ngraph/pattern.cpp
    ngraph/reshape_elimination.cpp
    ngraph/serialize.cpp
    ngraph/shape.cpp
    ngraph/tensor.cpp
    ngraph/type_prop.cpp
    ngraph/util.cpp
    ngraph/uuid.cpp
    ngraph/zero_dim_tensor_elimination.cpp
)

if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    list(APPEND NGRAPH_TESTS ngraph/tensorflow_import.cpp)
    #if (NNFUSION_ENABLE)
        # list(APPEND NGRAPH_TESTS ngraph/nnfusion_backend.cpp)
    #endif()
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    list(APPEND NGRAPH_TESTS ngraph/onnx_import.cpp)
    if (NGRAPH_ONNXIFI_ENABLE)
        list(APPEND NGRAPH_TESTS ngraph/onnxifi.cpp)
    endif()
endif()

set(ACTIVE_BACKEND_LIST "")

if (NGRAPH_INTERPRETER_ENABLE)
    list(APPEND NGRAPH_TESTS
        ngraph/backend_debug_api.cpp
        ngraph/builder.cpp
        ngraph/backend_api.cpp
        ngraph/hybrid_backend.cpp
        ngraph/hybrid_utils.cpp)
    list(APPEND ACTIVE_BACKEND_LIST INTERPRETER)
endif()

if (NGRAPH_HYBRID_ENABLE)
    set(NGRAPH_TESTS ${NGRAPH_TESTS} ngraph/hybrid_backend.cpp)
endif()

if (NGRAPH_CPU_ENABLE)
    list(APPEND NGRAPH_TESTS ngraph/core_fusion.cpp ngraph/builder_quantization.cpp)
    list(APPEND NGRAPH_TESTS ngraph/backend_performance.cpp ngraph/cpu_fusion.cpp ngraph/cpu_test.cpp ngraph/cpu_reshape_sinking.cpp ngraph/cpu_debugger.cpp)
    if (NGRAPH_HALIDE)
        list(APPEND NGRAPH_TESTS ngraph/halide.cpp)
    endif()
    list(APPEND ACTIVE_BACKEND_LIST CPU)
endif()

if(NGRAPH_GPU_ENABLE)
    list(APPEND NGRAPH_TESTS ngraph/cudnn.cpp ngraph/gpu_test.cpp ngraph/gpu_fusion.cpp)
    list(APPEND ACTIVE_BACKEND_LIST GPU)
endif()

if (NGRAPH_INTELGPU_ENABLE)
    list(APPEND ACTIVE_BACKEND_LIST INTELGPU)
endif()

if (NGRAPH_HYBRID_ENABLE)
    list(APPEND ACTIVE_BACKEND_LIST HYBRID)
endif()

if (NGRAPH_PLAIDML_ENABLE)
    list(APPEND ACTIVE_BACKEND_LIST PlaidML)
endif()

# backend specific test files must meet the following requirements:
# 1) The must be named <name>.in.cpp
# 2) They must be in the test directory
# 3) add the line `static string s_manifest = "${MANIFEST}";` to your cpp file
# All such files are configured via cmake which replaces all instances of cmake variables
# such as ${BACKEND_NAME} with their values, such as CPU, GPU, or INTERPRETER.
set(MULTI_TEST_SRC
    ngraph/autodiff.in.cpp
    ngraph/backend_binary_elementwise.in.cpp
    ngraph/backend_broadcast.in.cpp
    ngraph/backend_comparison.in.cpp
    ngraph/backend_dot.in.cpp
    ngraph/backend_one_hot.in.cpp
    ngraph/backend_pool.in.cpp
    ngraph/backend_reduce.in.cpp
    ngraph/backend_reshape.in.cpp
    ngraph/backend_sum.in.cpp
    ngraph/backend_topk.in.cpp
    ngraph/backend_arg_reduce.in.cpp
    ngraph/backend_test.in.cpp
    ngraph/backend_unary_elementwise.in.cpp
    ngraph/convolution_test.in.cpp
)
if(NGRAPH_DISTRIBUTED_ENABLE)
    list(APPEND MULTI_TEST_SRC ngraph/distributed.in.cpp)
endif()

set(MULTI_TEST_SRC ${MULTI_TEST_SRC} PARENT_SCOPE)
set(NGRAPH_TESTS ${NGRAPH_TESTS} PARENT_SCOPE)
set(ACTIVE_BACKEND_LIST ${ACTIVE_BACKEND_LIST} PARENT_SCOPE)
