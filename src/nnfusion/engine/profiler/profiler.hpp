// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Use this Profiler to run each operator
 * \author wenxh
 * \todo This profiler only support linux since it will invoke native commands.
 */
#pragma once

#include <algorithm>
#include <string>

#include "cpu_runtime.hpp"
#include "cuda_runtime.hpp"
#include "ngraph/file_util.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "profiling_runtime.hpp"
#include "rocm_runtime.hpp"

//Support Linux for now.
#include <dlfcn.h>
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#define DLIB_SUFFIX ".so"
#define DL_HANDLE void*

using namespace std;
using namespace nnfusion::graph;
using namespace nnfusion::kernels;

namespace nnfusion
{
    namespace profiler
    {
        ///\brief Profiler will profile a operator or a subgraph. This Profiler class should be treated as interface for Host.
        //Profiler will use the Runtime to run the subject.
        ///\todo To support a subgraph
        class Profiler
        {
        public:
            Profiler(IProfilingRuntime::Pointer rt, ProfilingContext::Pointer context);
            bool execute();
            bool find_best();
            bool execute_all();
            double execute(void** input, void** output);

            ///\brief T should be basic date type: int, float, double;
            template <typename T>
            vector<vector<T>> execute(const vector<vector<T>>& inputs)
            {
                auto& kernel_mem = pctx->kernel_memory;
                kernel_mem->load_inputs(inputs);

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    LOG(ERROR) << "Failed execute the kernel.";
                    return vector<vector<T>>();
                }

                return kernel_mem->save_outputs<T>();
            }

            // multiple inputs (or outputs) may have different element types
            bool mixed_type_execute(const vector<vector<char>>& inputs,
                                    vector<vector<char>>& outputs)
            {
                auto& kernel_mem = pctx->kernel_memory;
                auto kctx = pctx->kernel->m_context;
                CHECK(inputs.size() == kctx->inputs.size());

                for (size_t i = 0; i < kctx->inputs.size(); i++)
                {
                    auto& t = kctx->inputs[i];
                    size_t _size = t.get_size() * t.get_element_type().size();
                    CHECK(inputs[i].size() == _size);

                    kernel_mem->load_input_from(i, inputs[i].data(), _size);
                }

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    LOG(ERROR) << "Failed execute the kernel.";
                    return false;
                }

                outputs.clear();
                void** ptrs = kernel_mem->unsafe_outputs();
                for (size_t i = 0; i < kctx->outputs.size(); ++i)
                {
                    auto& t = kctx->outputs[i];
                    size_t _size = t.get_size() * t.get_element_type().size();

                    CHECK(ptrs[i] != nullptr);
                    vector<char> output(_size);
                    memcpy(output.data(), ptrs[i], _size);

                    outputs.push_back(move(output));
                }
                return true;
            }

            ///\brief simple interface for execute
            template <typename T>
            vector<vector<T>> unsafe_execute(const void* val)
            {
                auto& kernel_mem = pctx->kernel_memory;

                size_t offset = 0;
                auto kctx = pctx->kernel->m_context;
                for (size_t i = 0; i < kctx->inputs.size(); i++)
                {
                    auto& t = kctx->inputs[i];
                    size_t _size = t.get_size() * t.get_element_type().size();
                    void* newval = (void*)((char*)val + offset);
                    kernel_mem->load_input_from(i, newval, _size);
                    offset += _size;
                }

                if (rt->execute(pctx, kernel_mem->unsafe_inputs(), kernel_mem->unsafe_outputs()) <
                    0)
                {
                    LOG(ERROR) << "Failed execute the kernel.";
                    return vector<vector<T>>();
                }

                return kernel_mem->save_outputs<T>();
            }

            // HOST TENSOR Operations
            ///\brief Allocate spaces for output tensors, but tensors need to be same type.
            /*
            template <typename T>
            vector<vector<T>> allocate_outputs()
            {
                auto& kctx = pctx->kernel->m_context;
                vector<vector<T>> res;
                for (int i = 0; i < kctx->outputs.size(); i++)
                {
                    res.push_back(vector<T>());
                    res[i].resize(kctx->outputs[i].get_size());
                }
                return move(res);
            }

            template <class T>
            std::vector<T> create_vector(T* t, size_t size)
            {
                std::vector<T> vec;
                for (int i = 0; i < size; i++)
                    vec.push_back(t[i]);
                return vec;
            }

            template <class T>
            void* create_empty_tensor(size_t size)
            {
                T* t = new T[size];
                memset(t, 0, sizeof(T) * size);
                return t;
            }

            template <class T>
            void* create_zeros_tensor(size_t size)
            {
                T* t = new T[size];
                for (size_t i = 0; i < size; i++)
                    t[i] = 1;
                return t;
            }

            template <class T>
            void* create_tensor(T* t, size_t size)
            {
                return t;
            }

            template <class T>
            void* create_tensor(std::vector<T> data)
            {
                T* t = new T[data.size()];
                for (int i = 0; i < data.size(); i++)
                    t[i] = data[i];
                return t;
            }
            */

        private:
            ProfilingContext::Pointer pctx;
            IProfilingRuntime::Pointer rt;
        };

        ///\brief Evaluation for (sub)graph, the subgraph should have none undetermined input.
        class GraphEvaluate
        {
        public:
            GraphEvaluate(shared_ptr<nnfusion::graph::Graph> graph)
                : gctx(GraphEvaluationContext(graph))
            {
                rt = ReferenceRuntime::Runtime();
            }

            unordered_map<string, ProfilingContext::Pointer> eval();

        private:
            GraphEvaluationContext gctx;
            ReferenceRuntime::Pointer rt;

            void create_profiling_contexts(shared_ptr<GNode> node);
            void connect_nodes(shared_ptr<GNode> node);
        };

        IProfilingRuntime::Pointer get_default_runtime(DeviceType dev_t);
    };
}
