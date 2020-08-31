#platform-aware setting
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
	set(SUPERSCALER_NAME "libsuperscaler.${CMAKE_SYSTEM_NAME}.${CMAKE_SYSTEM_PROCESSOR}.${TARGET_GPU_PLATFORM}.so")
	set(SUPERSCALER_PLATFORM_NAME "libsuperscaler.so")
else()
	message(FATAL "Not Supported Yet")
endif()

execute_process(
	COMMAND ${CMAKE_COMMAND} -E create_symlink ${SUPERSCALER_NAME} ${SUPERSCALER_PLATFORM_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
option(OMIT_SC_PLANGEN "ignore plan when build if on, defaults to off" OFF)

if(OMIT_SC_PLANGEN)
else()
find_package(PythonInterp REQUIRED)
execute_process(
	COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/plan_gen.py" "${CMAKE_CURRENT_SOURCE_DIR}/nnfusion_rt.cu" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
endif(OMIT_SC_PLANGEN)

add_library(superscaler INTERFACE)
target_include_directories(superscaler SYSTEM INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(superscaler INTERFACE "${CMAKE_CURRENT_LIST_DIR}/${SUPERSCALER_PLATFORM_NAME}")
