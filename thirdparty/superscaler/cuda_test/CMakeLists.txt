project(superscaler_test)
cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")


enable_testing()

find_package(CUDA)
if (CUDA_FOUND)
  link_directories(/usr/local/cuda/lib64)
  include_directories(${CUDA_INCLUDE_DIRS})
  find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
  find_library(CUDA_cudart_LIBRARY libcudart.so /usr/local/cuda/lib64)
  file(GLOB superscaler_test_files *.cc)
  add_executable(superscaler_test ${superscaler_test_files})
  if (NOT TARGET superscaler)
  include(superscaler/superscaler.cmake)
  endif()
  target_link_libraries(superscaler_test PRIVATE superscaler) 
  target_link_libraries(superscaler_test PRIVATE ${CUDA_cudart_LIBRARY}) 
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_CURRENT_SOURCE_DIR}/plan ${CMAKE_CURRENT_BINARY_DIR}/plan)

add_test(NAME sc_test COMMAND mpirun -np 2 $<TARGET_FILE:superscaler_test>)
