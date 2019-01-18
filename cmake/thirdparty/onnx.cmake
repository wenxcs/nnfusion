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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# ONNX.proto definition version
#------------------------------------------------------------------------------

set(ONNX_VERSION 1.3.0)

#------------------------------------------------------------------------------
# Download and install libonnx ...
#------------------------------------------------------------------------------

ExternalProject_Add(
    ext_onnx
    PREFIX ext_onnx
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DONNX_GEN_PB_TYPE_STUBS=OFF
               -DPROTOBUF_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
               -DPROTOBUF_LIBRARY=${Protobuf_LIBRARY}
               -DPROTOBUF_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
               -DPROTOBUF_SRC_ROOT_FOLDER=${Protobuf_SRC_ROOT_FOLDER}
               -DONNX_ML=TRUE
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/onnx/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/onnx/stamp"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/onnx"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/onnx"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/onnx"
    BUILD_BYPRODUCTS ${NNFUSION_THIRDPARTY_FOLDER}/build/onnx/bin/libonnx_proto.a
                     ${NNFUSION_THIRDPARTY_FOLDER}/build/onnx/bin/libonnx.a
    EXCLUDE_FROM_ALL TRUE
    )

# -----------------------------------------------------------------------------

ExternalProject_Get_Property(ext_onnx SOURCE_DIR BINARY_DIR)

set(ONNX_INCLUDE_DIR ${SOURCE_DIR}/onnx)
set(ONNX_PROTO_INCLUDE_DIR ${BINARY_DIR}/onnx)
set(ONNX_LIBRARY ${BINARY_DIR}/libonnx.a)
set(ONNX_PROTO_LIBRARY ${BINARY_DIR}/libonnx_proto.a)
set(ONNX_LIBRARIES ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY})

if (NOT TARGET onnx::libonnx)
    add_library(onnx::libonnx UNKNOWN IMPORTED)
    set_target_properties(onnx::libonnx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_LIBRARY})
    add_dependencies(onnx::libonnx ext_onnx)
endif()

if (NOT TARGET onnnx::libonnx_proto)
    add_library(onnx::libonnx_proto UNKNOWN IMPORTED)
    set_target_properties(onnx::libonnx_proto PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${ONNX_PROTO_INCLUDE_DIR}
            IMPORTED_LOCATION ${ONNX_PROTO_LIBRARY})
    add_dependencies(onnx::libonnx_proto ext_onnx)
endif()