#!/bin/bash -e

if [[ "$@" != "--direct" ]]; then
	docker kill nnfusion >/dev/null 2>&1 || true
	docker rm nnfusion >/dev/null 2>&1 || true
	docker run --name nnfusion -it -d --net=host -e EXEC_BASH=1 -v `pwd`:/mnt -w /root ubuntu:18.04 bash
	docker exec -it nnfusion /mnt/autogen.sh --direct
	docker exec -it nnfusion bash
	exit 0
fi

DEB_PACKAGES="build-essential cmake clang-3.9 clang-format-3.9 git curl zlib1g zlib1g-dev libtinfo-dev unzip autoconf automake libtool ca-certificates"

if ! dpkg -L $DEB_PACKAGES >/dev/null 2>&1; then
	if [[ "$(whoami)" != "root" ]]; then
		SUDO=sudo
	fi
	$SUDO sh -c "apt update && apt install -y --no-install-recommends $DEB_PACKAGES"
fi

git config --global http.sslVerify false

[ -e NNFusion ] || git clone https://sysdnn.visualstudio.com/NNFusion/_git/NNFusion --branch nnfusion_backend --single-branch
cd NNFusion && git config credential.helper store && cd ..
[ -e Thirdparty ] || git clone https://sysdnn.visualstudio.com/NNFusion/_git/Thirdparty
cd Thirdparty && git config credential.helper store && cd ..

mkdir -p NNFusion/build && cd NNFusion/build
cmake .. -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DNNFUSION_THIRDPARTY_FOLDER=$(pwd)/../../Thirdparty
make -j

echo
echo "[DONE] Enfusion Example:"
echo
echo "Tensorflow Graph:  $(pwd)/src/tools/nnfusion/nnfusion ./NNFusion/test/models/tensorflow/frozen_op_graph/frozen_add_graph.pb -f tensorflow -b nnfusion"
echo "   ONNX    Graph:  $(pwd)/src/tools/nnfusion/nnfusion ./NNFusion/build/src/tools/nnfusion/nnfusion ./NNFusion/test/models/onnx/softmax.onnx -f onnx -b nnfusion"
echo

if [[ "$EXEC_BASH" == "1" ]];then
	exec bash
fi
