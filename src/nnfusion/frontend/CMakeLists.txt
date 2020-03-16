# Microsoft (c) 2019

if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    add_subdirectory(tensorflow_import)
    if (NGRAPH_ONNX_IMPORT_ENABLE)
        add_subdirectory(onnx_import)
    endif()
endif()

if (NNFUSION_TORCHSCRIPT_IMPORT_ENABLE)
    add_subdirectory(torchscript_import)
endif()

add_subdirectory(util)