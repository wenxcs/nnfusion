# Microsoft (c) 2019

if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    add_subdirectory(tensorflow_import)
    add_subdirectory(onnx_import)
endif()