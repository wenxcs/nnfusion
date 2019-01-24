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

        int ret = system(
            ("nvcc\t--compiler-options '-fPIC'\t--shared\t" + filename + "\t-o\t" + library_path)
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

TEST(nnfusion_backend, relu_op)
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