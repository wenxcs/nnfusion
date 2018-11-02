# Wolong | NNFusion [![Build Status][build-status-badge]][build-status]

Welcome to the open-source repository for the **Wolong NNFusion Project**. 
NNFusion is a compiler for optimizing and simplifying deep learning computation, 
especially for model inference.  It aims to provide an automatic way to apply graph 
optimizations, memory optimizations (i.e., quantization and sparsity optimization), 
and code generation for DNN computation on variance devices like CPU, GPU and 
others. NNFusion can generate end-to-end optimized and fused source code, which 
completely avoids framework scheduling overhead and kernel launch overhead. 
Furthermore, we provide a flexible way to allow user to customize an operator or 
a subgraph. With this compiler, we can significantly remove the repeated human 
effort for rewriting and hand-optimizing a trained model and automate the whole 
optimization pipeline. Our current prototype has demonstrated significant 
performance improvement on a series of deep learning workloads, without any 
hands optimization effort. 


![NNFusion architechture][nnfusion-arch]

The **nGraph Compiler** is Intel's graph compiler for Artificial Neural Networks. 
Documentation in this repo describes how you can program any framework 
to run training and inference computations on a variety of Backends including 
Intel® Architecture Processors (CPUs), Intel® Nervana™ Neural Network Processors 
(NNPs), cuDNN-compatible graphics cards (GPUs), custom VPUs like [Movidius], and
many others. The default CPU Backend also provides an interactive *Interpreter* 
mode that can be used to zero in on a DL model and create custom nGraph 
optimizations that can be used to further accelerate training or inference, in 
whatever scenario you need.  

nGraph provides both  a C++ API for framework developers and a Python API which 
can run inference on models imported from ONNX. 

See the [Release Notes] for recent changes.


|  Framework     | bridge available? | ONNX support?  |
|----------------|-------------------|----------------|
| TensorFlow*    | yes               | yes            |
| MXNet*         | yes               | yes            |
| PaddlePaddle   | yes               | yes            |   
| PyTorch*       | no                | yes            |
| Chainer*       | no                | yes            |
| CNTK*          | no                | yes            |
| Caffe2*        | no                | yes            |


| Backend                                       | current support   | future support |
|-----------------------------------------------|-------------------|----------------|
| Intel® Architecture CPU                       | yes               | yes            |
| Intel® Nervana™ Neural Network Processor (NNP)| yes               | yes            |
| Intel [Movidius™ Myriad™ 2] VPUs              | coming soon       | yes            |
| Intel® Architecture GPUs                      | via PlaidML       | yes            |
| AMD* GPUs                                     | via PlaidML       | yes            |
| NVIDIA* GPUs                                  | via PlaidML       | some           | 
| Field Programmable Gate Arrays (FPGA)         | no                | yes            |


## Documentation

See our [install] docs for how to get started.

For this early release, we provide [framework integration guides] to
compile MXNet and TensorFlow-based projects. If you already have a
trained model, we've put together a getting started guide for
[how to import] a deep learning model and start working with the nGraph
APIs.

## Support

Please submit your questions, feature requests and bug reports via
[GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea how
to improve the Library:

* See the [contrib guide] for code formatting and style guidelines.
* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* Make sure your PR passes all CI tests. Note: our [Travis-CI][build-status] service
  runs only on a CPU backend on Linux. We will run additional tests
  in other environments.
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.

[install]: http://ngraph.nervanasys.com/docs/latest/buildlb.html
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[release notes]: http://ngraph.nervanasys.com/docs/latest/project/release-notes.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[contrib guide]: http://ngraph.nervanasys.com/docs/latest/project/code-contributor-README.html
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: http://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph-ecosystem]: doc/sphinx/source/graphics/599px-Intel-ngraph-ecosystem.png "nGraph Ecosystem"
[build-status]: https://travis-ci.org/NervanaSystems/ngraph/branches
[build-status-badge]: https://travis-ci.org/NervanaSystems/ngraph.svg?branch=master
[develop-without-lockin]: doc/sphinx/source/graphics/develop-without-lockin.png "Develop on any part of the stack wtihout lockin"
[Movidius™ Myriad™ 2]:https://www.movidius.com/solutions/vision-processing-unit
