//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef WIN32
#include <windows.h>
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#define DLIB_SUFFIX ".dll"
#define DL_HANDLE HMODULE
#else
#include <dlfcn.h>
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#define DLIB_SUFFIX ".so"
#define DL_HANDLE void*
#endif

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/tensorflow_import/tensorflow.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

#include "ngraph/file_util.hpp"

using namespace ngraph;

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;
using Model = std::vector<std::shared_ptr<Function>>;

namespace nnfusion_test
{
    // This doodad finds the full path of the containing shared library
    static std::string find_my_file()
    {
#ifdef WIN32
        return ".";
#else
        Dl_info dl_info;
        dladdr(reinterpret_cast<void*>(find_my_file), &dl_info);
        return dl_info.dli_fname;
#endif
    }

    bool file_exsits(std::string filename)
    {
        std::ifstream ifile(filename.c_str());
        return (bool)ifile;
    }

    DL_HANDLE get_library(std::string func_name)
    {
        std::string filename = func_name + ".cu";
        assert(file_exsits(filename));

        std::string objname = func_name + DLIB_SUFFIX;
        std::string my_directory = file_util::get_directory(find_my_file());
        std::string library_path = file_util::path_join(my_directory, objname);

        int ret = system(("nvcc\t--compiler-options\t'-fPIC\t-lcudnn'\t--shared\t-gencode\tarch="
                          "compute_60,code=sm_60\t-O3\t" +
                          filename + "\t-o\t" + library_path)
                             .c_str());
        assert(file_exsits(library_path));

        DL_HANDLE handle;
#ifdef WIN32
        handle = LoadLibrary(library_path.c_str());
#else
        handle = dlopen(library_path.c_str(), RTLD_NOW);
#endif

        return handle;
    }

    void* get_funcion_pointer(std::string func_name, DL_HANDLE handle)
    {
        void* fhdl = DLSYM(handle, (func_name + "_simple").c_str());
        assert(fhdl != nullptr);
        return fhdl;
    }

    void close_dhhandel(DL_HANDLE& handle) { CLOSE_LIBRARY(handle); }
    template <class T>
    T* create_tensor(std::vector<T> data)
    {
        T* t = new T[data.size()];
        for (int i = 0; i < data.size(); i++)
            t[i] = data[i];
        return t;
    }

    template <class T>
    T* create_empty_tensor(std::vector<T> data)
    {
        T* t = new T[data.size()];
        return t;
    }

    template <class T>
    std::vector<T> create_vector(T* t, size_t size)
    {
        std::vector<T> vec;
        for (int i = 0; i < size; i++)
            vec.push_back(t[i]);
        return vec;
    }

    template <class T, class T1>
    std::vector<std::vector<T1>> execute_op(const std::shared_ptr<ngraph::Function>& function,
                                            std::string test_name,
                                            std::vector<std::vector<T>> args,
                                            std::vector<std::vector<T1>> out,
                                            std::string config)
    {
        std::vector<std::vector<T1>> vec_rc;
        auto backend = ngraph::runtime::Backend::create(config);
        backend->compile(function);
        DL_HANDLE handle = nnfusion_test::get_library(test_name);
        assert(handle != nullptr);
        auto func_simple = reinterpret_cast<bool (*)(void**)>(
            nnfusion_test::get_funcion_pointer(test_name, handle));

        size_t args_cnt = args.size() + out.size();
        void** arg = new void*[args_cnt];
        for (int i = 0; i < args.size(); i++)
            arg[i] = create_tensor(args[i]);
        for (int i = args.size(); i < out.size() + args.size(); i++)
            arg[i] = create_empty_tensor(out[i - args.size()]);

        func_simple(arg);

        for (int i = args.size(); i < out.size() + args.size(); i++)
            vec_rc.push_back(create_vector((T1*)(arg[i]), out[i - args.size()].size()));

        //Release Resources
        nnfusion_test::close_dhhandel(handle);
        for (int i = 0; i < args.size(); i++)
            delete (T*)arg[i];
        for (int i = args.size(); i < out.size() + args.size(); i++)
            delete (T1*)arg[i];
        delete arg;
        return vec_rc;
    }
}

/* example for test one function
TEST(nnfusion_backend, relu_fun)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_relu_graph.pb"));

    Inputs inputs{{-1, -0.00001, 0, 0.00001, 2}};
    Outputs expected_outputs{{0, 0, 0, 0.00001, 2}};

    Outputs outputs{nnfusion_test::execute_op(model[0],
                                              "cuda_ew_relu_float_float_test",
                                              inputs,
                                              expected_outputs,
                                              "CUDA_CODEGEN:naive_unittest")};

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}
*/

TEST(nnfusion_backend, relu_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_relu_graph.pb"));

    Inputs inputs{{-1, -0.00001, 0, 0.00001, 2}};
    Outputs expected_outputs{{0, 0, 0, 0.00001, 2}};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(nnfusion_backend, abs_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_abs_graph.pb"));

    std::vector<std::vector<int64_t>> inputs{};
    std::vector<std::vector<int64_t>> expected_outputs{{2147483649}};

    // constant input is -2147483649
    std::vector<std::vector<int64_t>> outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_EQ(outputs.size(), 1);
    // This abs is through fabs, so it's maybe inaccurate;
    EXPECT_TRUE(abs(outputs[0][0] - expected_outputs[0][0]) < 10);
}

TEST(nnfusion_backend, add_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_add_graph.pb"));

    Inputs inputs{{2}};
    Outputs expected_outputs{{3.0, 4.0, 5.0}};

    // constant input is -2147483649
    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(nnfusion_backend, bias_add_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_bias_add_graph.pb"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>{{1.8, 2.2}, {-1.3, -0.04}, {3.0, -12}}.get_vector());
    inputs.emplace_back(test::NDArray<float, 1>{100, -100}.get_vector());
    std::vector<std::vector<float>> expected_outputs{
        test::NDArray<float, 2>{{101.8, -97.8}, {98.7, -100.04}, {103, -112}}.get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(nnfusion_backend, matmul_op)
{
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_matmul_graph.pb"));

    Inputs inputs;
    inputs.emplace_back(
        test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    inputs.emplace_back(
        test::NDArray<float, 2>({{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}})
            .get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 2>({{190, 200, 210}, {470, 496, 522}, {750, 792, 834}}).get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(nnfusion_backend, max_pool_2d_op)
{
    // Pooling with strides=2 and padding=1
    auto model = frontend::load_tensorflow_model(
        file_util::path_join(SERIALIZED_ZOO, "tensorflow/frozen_op_graph/max_pool_2d_pads.pb"));

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(test::NDArray<float, 4>({{{{0.f, 1.f, 2.f, 3.f},
                                                {4.f, 5.f, 6.f, 7.f},
                                                {8.f, 9.f, 10.f, 11.f},
                                                {12.f, 13.f, 14.f, 15.f}}}})
                         .get_vector());
    // (1, 1, 2, 2)
    Outputs expected_outputs{test::NDArray<float, 4>({{{{5.f, 7.f}, {13.f, 15.f}}}}).get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

/* <TODO> Add maxpool3d in tensorflow_importer
TEST(nnfusion_backend, max_pool_3d_op)
{
    // Pooling with strides=2 and padding=1
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_max_pool_3d_pads.pb"));

    Inputs inputs;
    inputs.push_back(
        test::NDArray<float, 1>({0.,  -1., 0.,  1.,  -3., -4., -5., 4.,  4.,  -1., -4., -3., -1.,
                                 2.,  -2., -4., -3., -3., 0.,  -4., -5., -4., -3., -2., 4.,  -1.,
                                 3.,  -2., -5., -4., 0.,  4.,  -3., 2.,  -4., 3.,  -3., -3., -5.,
                                 0.,  -3., -4., 2.,  -3., 3.,  -3., 0.,  -4., 1.,  -3., 1.,  -4.,
                                 -5., 4.,  -5., -2., -5., -4., -4., 2.,  -5., -2., -4., 2.})
            .get_vector());

    Outputs expected_outputs{test::NDArray<float, 1>({0., 4., 4., 4., 4., 3., 3., 2.}).get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}
*/

TEST(nnfusion_backend, reshape_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_reshape_int64_graph.pb"));
    Inputs inputs{test::NDArray<float, 3>{
        {{1, 1, 1}, {2, 2, 2}}, {{3, 3, 3}, {4, 4, 4}}, {{5, 5, 5}, {6, 6, 6}}}
                      .get_vector()};
    Outputs expected_outputs{test::NDArray<float, 3>{{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}},
                                                     {{4, 4, 4}, {5, 5, 5}, {6, 6, 6}}}
                                 .get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}

TEST(nnfusion_backend, conv2d_op)
{
    auto model = frontend::load_tensorflow_model(file_util::path_join(
        SERIALIZED_ZOO, "tensorflow/frozen_op_graph/frozen_conv2d_nhwc_graph_2.pb"));
    Inputs inputs;
    inputs.emplace_back(test::NDArray<float, 1>(
                            {3., 4., 4., 0., 0., -5., -3., 1., -2., -3., 4., 4., -1., 3., 0., 4.})
                            .get_vector());
    Outputs expected_outputs{
        test::NDArray<float, 1>({1., -11., -17., 4., 4., 5., -1., 12., -20.}).get_vector()};

    Outputs outputs{nnfusion_test::execute_op(
        model[0], "naive_test", inputs, expected_outputs, "CUDA_CODEGEN:naive_graphtest")};

    EXPECT_TRUE(test::all_close_f(expected_outputs.front(), outputs.front()));
}