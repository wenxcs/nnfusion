#-----------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
#-----------------------------------------------------------------------------------------------

# This file is used to Update protobuf generated source files each time when compile

if(NGRAPH_USE_SYSTEM_PROTOBUF)
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
else()

    set(TF_PROTO_SRCS
        attr_value.pb.cc
        function.pb.cc
        node_def.pb.cc
        graph.pb.cc
        op_def.pb.cc
        resource_handle.pb.cc
        tensor.pb.cc
        tensor_shape.pb.cc
        types.pb.cc
        versions.pb.cc
    )

    set(TF_PROTO_HDRS
        attr_value.pb.h
        function.pb.h
        node_def.pb.h
        graph.pb.h
        op_def.pb.h
        resource_handle.pb.h
        tensor.pb.h
        tensor_shape.pb.h
        types.pb.h
        versions.pb.h
    )

    add_library(tensorflow_import_proto STATIC
            ${TF_PROTO_SRCS}
            ${TF_PROTO_HDRS}
    )
    add_dependencies(tensorflow_import_proto protobuf::libprotobuf)
    target_include_directories(tensorflow_import_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIR})
    set(TFPROTO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location for generated tensorflow protobuf headers." )
endif()

install(DIRECTORY
    ${TFPROTO_INCLUDE}
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend/tensorflow_import/"
    FILES_MATCHING PATTERN "*.pb.h"
    PATTERN "CMakeFiles" EXCLUDE
)
