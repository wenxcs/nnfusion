#!/bin/bash

set -ex

cd $(dirname $0)

[ -e kernels/ ]

FLAGS=""

for I in kernels/*.cpp; do
	FLAGS="$FLAGS -l$(basename $I .cpp)"
	/opt/rocm/bin/hipcc $I -fPIC -shared -O3 -lrocblas -lMIOpen -I/opt/rocm/include -o kernels/lib$(basename $I .cpp).so &
done

while wait -n; do true; done

/opt/rocm/bin/hipcc *.cpp -O3 -lrocblas -lMIOpen -I/opt/rocm/include -L$(pwd)/kernels $FLAGS -o main_test

LD_LIBRARY_PATH=$(pwd)/kernels ./main_test
