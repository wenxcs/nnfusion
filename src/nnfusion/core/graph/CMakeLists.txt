# Microsoft (c) 2019, NNFusion Team

set(SRC
    graph.cpp
    gnode.cpp
    gedge.cpp
    graph_util.cpp
    util/autobroadcast.cpp
    util/numpy_transpose.cpp
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
if(protobuf_VERBOSE)
  message(STATUS "Using Protocol Buffers ${Protobuf_VERSION}")
endif()

foreach(item graph_def node_def attr_value tensor_shape)
  set(${item}_PROTOS ${item}.proto)
  #Code Generation
  protobuf_generate_cpp(${item}_PROTO_SRCS ${item}_PROTO_HDRS ${${item}_PROTOS})
  list(APPEND SRC ${${item}_PROTO_SRCS} ${${item}_PROTO_HDRS} ${${item}_PROTOS})
  message(STATUS "generate code for ${item}.proto, including: ${SRC}")
endforeach()

add_library(nnfusion_graph STATIC ${SRC})
target_include_directories(nnfusion_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(nnfusion_graph ${Protobuf_LIBRARIES})
target_compile_options(nnfusion_graph PRIVATE "-fPIC")
