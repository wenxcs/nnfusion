# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

find_package(ZLIB REQUIRED)

# Override default LLVM binaries
if(NOT DEFINED LLVM_TARBALL_URL)
    if(APPLE)
        set(LLVM_TARBALL_URL "${NNFUSION_THIRDPARTY_FOLDER}/llvm_prebuilt/clang+llvm-5.0.2-x86_64-apple-darwin.tar.xz")
    else()
        set(LLVM_TARBALL_URL "${NNFUSION_THIRDPARTY_FOLDER}/llvm_prebuilt/clang+llvm-5.0.2-x86_64-linux-gnu-ubuntu-16.04.tar.xz")
    endif()
endif()

set(LLVM_CORE "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt/lib/libLLVMCore.a")
if(EXISTS "${LLVM_CORE}")
    message(STATUS "Found Prebuilt LLVM: ${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt")
    ExternalProject_Add(
        ext_llvm
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        DOWNLOAD_NO_PROGRESS TRUE
        SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt"
        BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt/lib/libLLVMCore.a"
        EXCLUDE_FROM_ALL TRUE
    )
else()
    message(STATUS "Not Found Prebuilt LLVM")
    ExternalProject_Add(
        ext_llvm
        URL ${LLVM_TARBALL_URL}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        DOWNLOAD_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt"
        SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt"
        BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/build/llvm_prebuilt/lib/libLLVMCore.a"
        EXCLUDE_FROM_ALL TRUE
    )
endif()

add_library(libllvm INTERFACE)
ExternalProject_Get_Property(ext_llvm SOURCE_DIR)
add_dependencies(libllvm ext_llvm)

set(LLVM_LINK_LIBS
    ${SOURCE_DIR}/lib/libclangTooling.a
    ${SOURCE_DIR}/lib/libclangFrontendTool.a
    ${SOURCE_DIR}/lib/libclangFrontend.a
    ${SOURCE_DIR}/lib/libclangDriver.a
    ${SOURCE_DIR}/lib/libclangSerialization.a
    ${SOURCE_DIR}/lib/libclangCodeGen.a
    ${SOURCE_DIR}/lib/libclangParse.a
    ${SOURCE_DIR}/lib/libclangSema.a
    ${SOURCE_DIR}/lib/libclangStaticAnalyzerFrontend.a
    ${SOURCE_DIR}/lib/libclangStaticAnalyzerCheckers.a
    ${SOURCE_DIR}/lib/libclangStaticAnalyzerCore.a
    ${SOURCE_DIR}/lib/libclangAnalysis.a
    ${SOURCE_DIR}/lib/libclangARCMigrate.a
    ${SOURCE_DIR}/lib/libclangRewriteFrontend.a
    ${SOURCE_DIR}/lib/libclangEdit.a
    ${SOURCE_DIR}/lib/libclangAST.a
    ${SOURCE_DIR}/lib/libclangLex.a
    ${SOURCE_DIR}/lib/libclangBasic.a
    ${SOURCE_DIR}/lib/libLLVMLTO.a
    ${SOURCE_DIR}/lib/libLLVMPasses.a
    ${SOURCE_DIR}/lib/libLLVMObjCARCOpts.a
    ${SOURCE_DIR}/lib/libLLVMSymbolize.a
    ${SOURCE_DIR}/lib/libLLVMDebugInfoPDB.a
    ${SOURCE_DIR}/lib/libLLVMDebugInfoDWARF.a
    ${SOURCE_DIR}/lib/libLLVMMIRParser.a
    ${SOURCE_DIR}/lib/libLLVMCoverage.a
    ${SOURCE_DIR}/lib/libLLVMTableGen.a
    ${SOURCE_DIR}/lib/libLLVMDlltoolDriver.a
    ${SOURCE_DIR}/lib/libLLVMOrcJIT.a
    ${SOURCE_DIR}/lib/libLLVMObjectYAML.a
    ${SOURCE_DIR}/lib/libLLVMLibDriver.a
    ${SOURCE_DIR}/lib/libLLVMOption.a
    ${SOURCE_DIR}/lib/libLLVMX86Disassembler.a
    ${SOURCE_DIR}/lib/libLLVMX86AsmParser.a
    ${SOURCE_DIR}/lib/libLLVMX86CodeGen.a
    ${SOURCE_DIR}/lib/libLLVMGlobalISel.a
    ${SOURCE_DIR}/lib/libLLVMSelectionDAG.a
    ${SOURCE_DIR}/lib/libLLVMAsmPrinter.a
    ${SOURCE_DIR}/lib/libLLVMDebugInfoCodeView.a
    ${SOURCE_DIR}/lib/libLLVMDebugInfoMSF.a
    ${SOURCE_DIR}/lib/libLLVMX86Desc.a
    ${SOURCE_DIR}/lib/libLLVMMCDisassembler.a
    ${SOURCE_DIR}/lib/libLLVMX86Info.a
    ${SOURCE_DIR}/lib/libLLVMX86AsmPrinter.a
    ${SOURCE_DIR}/lib/libLLVMX86Utils.a
    ${SOURCE_DIR}/lib/libLLVMMCJIT.a
    ${SOURCE_DIR}/lib/libLLVMLineEditor.a
    ${SOURCE_DIR}/lib/libLLVMInterpreter.a
    ${SOURCE_DIR}/lib/libLLVMExecutionEngine.a
    ${SOURCE_DIR}/lib/libLLVMRuntimeDyld.a
    ${SOURCE_DIR}/lib/libLLVMCodeGen.a
    ${SOURCE_DIR}/lib/libLLVMTarget.a
    ${SOURCE_DIR}/lib/libLLVMCoroutines.a
    ${SOURCE_DIR}/lib/libLLVMipo.a
    ${SOURCE_DIR}/lib/libLLVMInstrumentation.a
    ${SOURCE_DIR}/lib/libLLVMVectorize.a
    ${SOURCE_DIR}/lib/libLLVMScalarOpts.a
    ${SOURCE_DIR}/lib/libLLVMLinker.a
    ${SOURCE_DIR}/lib/libLLVMIRReader.a
    ${SOURCE_DIR}/lib/libLLVMAsmParser.a
    ${SOURCE_DIR}/lib/libLLVMInstCombine.a
    ${SOURCE_DIR}/lib/libLLVMTransformUtils.a
    ${SOURCE_DIR}/lib/libLLVMBitWriter.a
    ${SOURCE_DIR}/lib/libLLVMAnalysis.a
    ${SOURCE_DIR}/lib/libLLVMProfileData.a
    ${SOURCE_DIR}/lib/libLLVMObject.a
    ${SOURCE_DIR}/lib/libLLVMMCParser.a
    ${SOURCE_DIR}/lib/libLLVMMC.a
    ${SOURCE_DIR}/lib/libLLVMBitReader.a
    ${SOURCE_DIR}/lib/libLLVMCore.a
    ${SOURCE_DIR}/lib/libLLVMBinaryFormat.a
    ${SOURCE_DIR}/lib/libLLVMSupport.a
    ${SOURCE_DIR}/lib/libLLVMDemangle.a
)

if(APPLE)
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} curses z m)
else()
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} tinfo z m)
endif()

target_include_directories(libllvm SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_link_libraries(libllvm INTERFACE ${LLVM_LINK_LIBS})