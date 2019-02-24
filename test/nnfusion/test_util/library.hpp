// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */
#pragma once

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/nnfusion/core/op.hpp"

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

namespace nnfusion
{
    namespace library
    {
        // test.cu contains test_simple(void** args) entry point;
        // test.cu -> test.so
        // This doodad finds the full path of the containing shared library
        std::string find_my_file();

        bool file_exsits(std::string filename);

        DL_HANDLE get_library(std::string func_name);

        void* get_funcion_pointer(std::string func_name, DL_HANDLE handle);

        void close_dhhandel(DL_HANDLE& handle);

        void dump_test_code(nnfusion::ir::Function_p func);

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
        std::vector<std::vector<T1>> execute_op(std::string test_name,
                                                std::vector<std::vector<T>> args,
                                                std::vector<std::vector<T1>> out)
        {
            std::vector<std::vector<T1>> vec_rc;
            DL_HANDLE handle = get_library(test_name);
            if (handle == nullptr)
                return vec_rc;
            auto func_simple =
                reinterpret_cast<bool (*)(void**)>(get_funcion_pointer(test_name, handle));

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
            close_dhhandel(handle);
            for (int i = 0; i < args.size(); i++)
                delete (T*)arg[i];
            for (int i = args.size(); i < out.size() + args.size(); i++)
                delete (T1*)arg[i];
            delete arg;
            return vec_rc;
        }
    }
}