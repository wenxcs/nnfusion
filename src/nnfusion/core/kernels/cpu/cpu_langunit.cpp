// Microsoft (c) 2019, NNFusion Team
#include "cpu_langunit.hpp"

using namespace nnfusion::kernels;

// Header
LU_DEFINE(header::thread, "#include <thread>\n");
LU_DEFINE(header::eigen_tensor, "#include <unsupported/Eigen/CXX11/Tensor>\n");

// Macro

// Declaration
LU_DEFINE(declaration::eigen_global_thread_pool, "extern Eigen::ThreadPool global_thread_pool;\n");
LU_DEFINE(declaration::eigen_global_thread_pool_device,
          "extern Eigen::ThreadPoolDevice global_thread_pool_device;\n");