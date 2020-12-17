# Microsoft (c) 2019

if (ONNX_FRONTEND)
    add_subdirectory(onnx_import)
endif()

if (TENSORFLOW_FRONTEND)
    add_subdirectory(tensorflow_import)
endif()

if (TORCHSCRIPT_FRONTEND)
    add_subdirectory(torchscript_import)
endif()

add_subdirectory(util)