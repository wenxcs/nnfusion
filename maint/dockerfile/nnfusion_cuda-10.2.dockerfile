# tag: nnfusion/base
# docker build -t nnfusion/cuda:10.2-devel-ubuntu18.04 --build-arg GITUSER=${} --build-arg GITPASSWD=${} -f nnfusion_cuda-10.2.dockerfile .
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ARG GITUSER
ARG GITPASSWD
RUN apt update && apt install -y git
RUN git clone https://$GITUSER:$GITPASSWD@sysdnn.visualstudio.com/NNFusion/_git/NNFusion /root/nnfusion --branch wenxh/opensource --single-branch
# - Install Requirements 
RUN bash /root/nnfusion/maint/script/install_dependency.sh
# - Make Install
RUN cd /root/nnfusion/ && mkdir build && cd build && cmake .. && make -j6 && make install
# - Execute command
RUN LD_LIBRARY_PATH=/usr/local/lib nnfusion /root/nnfusion/test/models/tensorflow/frozen_op_graph/frozen_abs_graph.pb
RUN apt install -y python3 python3-pip
RUN pip3 install numpy