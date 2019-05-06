// Microsoft (c) 2019, Wenxiang
#include "../rocm_langunit.hpp"

using namespace nnfusion::rocm;

LU_DEFINE(file::rocc,
          "rocm::file::rocc",
          R"(#!/bin/sh -e
# [e.g.] rocc sample.cu [..]
# HIP_DB=api ./a.out

if [ "$#" = "0" ]; then
  exit 1
fi

ARCH=${ARCH:-902}
CODE=$1 && shift
TEMP_CODE=$(mktemp).cc

WS=$(dirname $0)
LIBS=${LD_LIBRARY_PATH}
unset LD_LIBRARY_PATH || true

${WS}/hipify-rocc "$CODE" > $TEMP_CODE

/opt/rocm/bin/hipcc --amdgpu-target=gfx$ARCH -lhipblas -lMIOpen -lpthread -I${WS} -I/opt/rocm/include -O2 -std=c++11 -Wno-unused-value $TEMP_CODE "$@"

rm -f $TEMP_CODE
# perl -pi -e 's/opt\/rocm/opt\/rocr/g' ${OUT..}
)");