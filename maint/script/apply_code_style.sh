#!/bin/bash

CLANG_FORMAT="clang-format-3.9"
echo "Using ${CLANG_FORMAT} formating source code tree..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null
find . -path './thirdparty' -prune -o -path './build' -prune -o \
    -type f -and \( -name '*.cpp' -or -name '*.hpp' \) \
    | xargs "${CLANG_FORMAT}" -i -style=file 2>&1 \
    | grep -v "Is a directory"
echo "Done."
popd > /dev/null