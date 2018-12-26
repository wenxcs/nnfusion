#-----------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
#-----------------------------------------------------------------------------------------------

# This file is used to Update protobuf generated source files each time when compile

set(TF_PROTOS
    attr_value.proto
    function.proto
    graph.proto
    node_def.proto
    op_def.proto
    resource_handle.proto
    tensor_shape.proto
    tensor.proto
    types.proto
    versions.proto
)

protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HDRS ${TF_PROTOS})

add_library(tensorflow_import_proto STATIC
        ${TF_PROTO_SRCS}
        ${TF_PROTO_HDRS}
)

target_include_directories(tensorflow_import_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIR})

set(TFPROTO_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "Location for generated tensorflow protobuf headers." )