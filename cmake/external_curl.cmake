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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install curl...
#------------------------------------------------------------------------------

SET(curl_GIT_REPO_URL https://github.com/curl/curl.git)
SET(curl_GIT_LABEL curl-7_68_0)

ExternalProject_Add(
    ext_curl
    PREFIX curl
    GIT_REPOSITORY ${curl_GIT_REPO_URL}
    GIT_TAG ${curl_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/curl/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/curl/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/curl/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/curl/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/curl/build"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/curl"
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_curl SOURCE_DIR BINARY_DIR)

add_library(libcurl INTERFACE)
add_dependencies(libcurl ext_curl)
target_include_directories(libcurl SYSTEM INTERFACE ${SOURCE_DIR}/include/curl)
target_link_libraries(libcurl INTERFACE ${BINARY_DIR}/lib/libcurl.so)
