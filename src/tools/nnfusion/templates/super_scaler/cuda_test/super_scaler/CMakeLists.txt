project(super_scaler_lib)

execute_process(
COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


#platform-aware setting
set(TARGET_GPU_PLATFORM "CUDA" CACHE STRING "Choose your GPU platform: CUDA or ROCm")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	set(SUPERSCALER_NAME "libsuper_scaler.${CMAKE_SYSTEM_NAME}.${CMAKE_SYSTEM_PROCESSOR}.${TARGET_GPU_PLATFORM}.so")
	set(SUPERSCALER_PLATFORM_NAME "libsuperscaler.so")
else()
	message(FATAL "Not Supported Yet")
endif()

execute_process(
COMMAND ${CMAKE_COMMAND} -E create_symlink ${SUPERSCALER_NAME} ${SUPERSCALER_PLATFORM_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(super_scaler INTERFACE)
target_include_directories(super_scaler SYSTEM INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(super_scaler INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/${SUPERSCALER_PLATFORM_NAME}")

