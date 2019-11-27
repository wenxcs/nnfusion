// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include <assert.h>
#include <execinfo.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#define ktrace()                                                                                   \
    {                                                                                              \
        void* array[10];                                                                           \
        size_t size = backtrace(array, sizeof(array) / sizeof(*array));                            \
        char** strings = backtrace_symbols(array, size);                                           \
        if (NULL == strings)                                                                       \
        {                                                                                          \
            perror("backtrace_symbols");                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        LOG(INFO) << " - Obtained " + size + " stack frames.";                                     \
        for (int i = 0; i < size; i++)                                                             \
            LOG(INFO) << "    # " + strings[i];                                                    \
        free(strings);                                                                             \
    }

namespace nnfusion
{
    namespace codegen
    {
        inline bool create_folder(std::string tar_path)
        {
            bool flag;
            int mkdir_status;
            struct stat s;
            int err = stat(tar_path.c_str(), &s);
            if (-1 == err)
            {
                mkdir_status = mkdir((tar_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                if (-1 == mkdir_status)
                {
                    LOG(INFO) << "Error creating directory: " + tar_path;
                    flag = false;
                }
                else
                    flag = true;
            }
            else
            {
                //LOG(INFO) << "Directory " << tar_path.c_str() << " already exists";
                flag = true;
            }
            return flag;
        }

        inline std::string get_file_from_templates(const std::string& rel_path)
        {
            static std::string abs_path;
            if (abs_path.size() == 0)
            {
                char exepath[1024];
                auto ret = readlink("/proc/self/exe", exepath, sizeof(exepath));
                CHECK(ret > 0);
                for (int i = strlen(exepath) - 1; i >= 0; --i)
                    if (exepath[i] == '/')
                    {
                        exepath[i] = 0;
                        break;
                    }
                abs_path = std::string(exepath) + "/templates/";
            }
            return abs_path + rel_path;
        }

        inline std::string get_content_from_templates(const std::string& rel_path)
        {
            std::ifstream in(get_file_from_templates(rel_path), std::ios::binary);
            std::string str((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            return str;
        }

        inline bool copy_file_from_templates(const std::string& rel_path,
                                             const std::string& target_name)
        {
            int at = 1, next;
            while (next = target_name.find('/', at), next >= 0)
            {
                create_folder(target_name.substr(0, next));
                at = next + 1;
            }
            std::ifstream in(get_file_from_templates(rel_path), std::ios::binary);
            std::ofstream out(target_name, std::ios::binary);
            out << in.rdbuf();
            return true;
        }
    } // namespace codegen
} // namespace nnfusion

using namespace std;
using namespace nnfusion;

#include "code_writer.hpp"
#include "gflags/gflags.h"
#include "nlohmann/json.hpp"
#include "nnfusion/util/util.hpp"
#include "type_info.hpp"

#define create_ptr(type, name, arg) shared_ptr<type> name(new type(arg))

// Uncomment this for quick debug
// #undef LOG(INFO)INFO
// #define LOG(INFO)INFO std::cout

namespace nnfusion
{
    enum DeviceType
    {
        CUDA_GPU,
        ROCM_GPU,
        GENERIC_CPU
    };
}
