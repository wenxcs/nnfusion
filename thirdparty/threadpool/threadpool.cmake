include(ExternalProject)

ExternalProject_Add(hwloc
  PREFIX "hwloc"
  URL "https://download.open-mpi.org/release/hwloc/v2.1/hwloc-2.1.0.tar.gz"
  CONFIGURE_COMMAND ../src/configure --prefix "${CMAKE_CURRENT_BINARY_DIR}/hwloc/include"
  BUILD_COMMAND make -j${J} install
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/stamp"
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/download"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/build"
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc"
)

add_library(libhwloc INTERFACE)
add_dependencies(libhwloc hwloc)
target_include_directories(libhwloc SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/hwloc/include/include)
target_link_libraries(libhwloc INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/hwloc/include/lib/libhwloc.so)

set(threadpool_srcs
  ${CMAKE_CURRENT_LIST_DIR}/util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/threadpool.cpp
  ${CMAKE_CURRENT_LIST_DIR}/numa_aware_threadpool.cpp
)

add_library(threadpool STATIC ${threadpool_srcs})
target_include_directories(threadpool SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(threadpool PRIVATE ${EIGEN_DIR})
target_link_libraries(threadpool libhwloc)
