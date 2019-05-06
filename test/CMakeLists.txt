add_subdirectory(models)
add_subdirectory(files)
add_subdirectory(util)

add_subdirectory(ngraph)
set(SRC main.cpp)

#Todo(wenxh): move this foreach into the subfolder;
foreach(BACKEND_NAME ${ACTIVE_BACKEND_LIST})
    # Some---but not all---autodiff tests go through multiple iterations with
    # different random seeds. On the CPU backend this is currently very slow
    # because the autodiff tests recompile with each iteration. That behavior
    # can be changed, but it's a bit involved, so for the time being we just
    # reduce the number of test iterations on non-INTERPRETER backends.
    if(${BACKEND_NAME} MATCHES ^INTERPRETER$)
        set(TEST_LOOPS 100)
    else()
        set(TEST_LOOPS 2)
    endif()

    string(TOLOWER ${BACKEND_NAME} BACKEND_DIR)
    set(MANIFEST ${PROJECT_SOURCE_DIR}/src/ngraph/runtime/${BACKEND_DIR}/unit_test.manifest)

    foreach(TEST_SRC ${MULTI_TEST_SRC})
        string(REPLACE ".in." "_${BACKEND_NAME}." TARGET_NAME ${TEST_SRC})
        configure_file(${TEST_SRC} ${TARGET_NAME})
        list(APPEND NGRAPH_TESTS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
    endforeach()

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

list(APPEND SRC ${NGRAPH_TESTS})

if (NNFUSION_ENABLE)
    add_subdirectory(nnfusion)
    list(APPEND SRC ${NNFUSION_TESTS})
endif()

set_source_files_properties(includes.cpp PROPERTIES COMPILE_DEFINITIONS
    NGRAPH_INCLUDES="${PROJECT_SOURCE_DIR}/src/ngraph")

add_executable(unit-test ${SRC})

target_include_directories(unit-test PRIVATE ".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJSON_INCLUDES=\\\"${JSON_INCLUDE_DIR}\\\"")

if(NGRAPH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    target_compile_definitions(unit-test PRIVATE NGRAPH_DISTRIBUTED)
    target_include_directories(unit-test
        SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(unit-test PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

target_link_libraries(unit-test PRIVATE ngraph_test_util)
target_link_libraries(unit-test PRIVATE ngraph libgtest libjson Threads::Threads)
target_link_libraries(unit-test PRIVATE ${CMAKE_DL_LIBS})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(unit-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

if (NGRAPH_CPU_ENABLE)
    # The INTERPRETER backend is required for graph_partition, convolution, and backwards unit tests
    target_link_libraries(unit-test PRIVATE cpu_backend interpreter_backend)
    target_link_libraries(unit-test PRIVATE libmkldnn)
endif()

if (NGRAPH_PLAIDML_ENABLE)
    target_link_libraries(unit-test PRIVATE plaidml_backend)
endif()

if (NGRAPH_TBB_ENABLE)
    target_compile_definitions(unit-test PRIVATE NGRAPH_TBB_ENABLE)
endif()

if (NGRAPH_INTERPRETER_ENABLE)
    target_compile_definitions(unit-test PRIVATE NGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(unit-test PRIVATE interpreter_backend)
endif()

if (NGRAPH_HYBRID_ENABLE)
    target_link_libraries(unit-test PRIVATE hybrid_backend)
endif()

if (NGRAPH_GPU_ENABLE)
    target_link_libraries(unit-test PRIVATE gpu_backend)
endif()

target_include_directories(unit-test PRIVATE ${Protobuf_INCLUDE_DIR})

if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    target_include_directories(unit-test PRIVATE ${TFPROTO_INCLUDE})
    target_link_libraries(unit-test PRIVATE tensorflow_import_interface tensorflow_import tensorflow_import_proto)
    target_link_libraries(unit-test PRIVATE cuda_backend)
endif()

if (NGRAPH_ONNXIFI_ENABLE)
    target_include_directories(unit-test SYSTEM PUBLIC ${ONNX_INCLUDE_DIR})
    target_link_libraries(unit-test PRIVATE onnxifi-ngraph)
endif()

target_link_libraries(unit-test PRIVATE codegen)

add_custom_target(unit-test-check
    COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
    DEPENDS unit-test
)

add_custom_target(check
    DEPENDS
    style-check
    unit-test-check
)
