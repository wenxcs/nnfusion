// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Basic Datastructure used in profiling
 * \author wenxh
 */
#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include "ngraph/node.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
using namespace std;
using namespace ngraph;
using namespace nnfusion;
using namespace nnfusion::graph;

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
    namespace profiler

    {
        //\bief Use this to store the result or other profiling details.
        struct ProfilingResult
        {
        private:
            vector<double> device_duration;
            vector<double> host_duration;
            bool ready = false;

        public:
            bool is_ready() const { return ready; }
            void set_ready() { ready = true; }
            //Get average time cost of host.
            double get_host_avg()
            {
                return host_duration.empty()
                           ? 0.0
                           : std::accumulate(host_duration.begin(), host_duration.end(), 0.0) /
                                 host_duration.size();
            }
            //Get average time cost inside the runtime.
            double get_device_avg()
            {
                return device_duration.empty()
                           ? 0.0
                           : std::accumulate(device_duration.begin(), device_duration.end(), 0.0) /
                                 device_duration.size();
            }

            const vector<double>& get_device_durations() { return device_duration; }
            const vector<double>& get_host_durations() { return host_duration; }
            void reset()
            {
                device_duration.clear();
                host_duration.clear();
            }

            void record_device_duration(double du) { device_duration.push_back(du); }
            void record_host_duration(double du) { host_duration.push_back(du); }
            using Pointer = shared_ptr<ProfilingResult>;
        };

        //\brief The cache to store the time cost data, this should be connected
        // to some database tool.
        struct ProfilingCache
        {
        };

        //\brief Use this to manage the memory of kernels.
        class KernelMemory
        {
        public:
            using Pointer = unique_ptr<KernelMemory>;
            KernelMemory(kernels::KernelContext::Pointer kctx)
            {
                this->kctx = kctx;
                raw_input.clear();
                raw_output.clear();

                for (auto& t : kctx->inputs)
                {
                    shared_ptr<char> i(new char[t.get_size() * t.get_element_type().size()],
                                       [](char* p) { delete[] p; });
                    raw_input.push_back(move(i));
                }
                for (auto& t : kctx->outputs)
                {
                    shared_ptr<char> i(new char[t.get_size() * t.get_element_type().size()],
                                       [](char* p) { delete[] p; });
                    raw_output.push_back(move(i));
                };
            }

            const KernelMemory& forward(int output_id, KernelMemory::Pointer& km, int input_id)
            {
                if (output_id >= raw_output.size() || input_id >= km->raw_output.size())
                {
                    LOG_WARN << "Invalid forward function.";
                    return *this;
                }
                km->raw_input[input_id] = raw_output[output_id];
                return *this;
            }

            bool load_input_from(int input_id, const void* data, size_t size)
            {
                auto buffsize = kctx->inputs[input_id].get_size() *
                                kctx->inputs[input_id].get_element_type().size();
                // Check if the buffer is same size;
                if (input_id >= kctx->inputs.size() || size != buffsize)
                {
                    LOG_ERR << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_input(input_id), data, buffsize);

                if (status == nullptr)
                {
                    LOG_ERR << "Memcpy failed.";
                    return false;
                }
                return true;
            }

            bool set_output_from(int output_id, const void* data, size_t size)
            {
                auto buffsize = kctx->outputs[output_id].get_size() *
                                kctx->outputs[output_id].get_element_type().size();
                // Check if the buffer is same size;
                if (output_id >= kctx->outputs.size() || size != buffsize)
                {
                    LOG_ERR << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_output(output_id), data, buffsize);

                if (status == nullptr)
                {
                    LOG_ERR << "Memcpy failed.";
                    return false;
                }
                return true;
            }

            template <typename T>
            bool load_input_from(const vector<T>& data, int input_id)
            {
                auto buffsize = kctx->inputs[input_id].get_size() *
                                kctx->inputs[input_id].get_element_type().size();
                // Check if the buffer is same size;
                if (input_id >= kctx->inputs.size() || sizeof(T) * data.size() != buffsize)
                {
                    LOG_ERR << "Input data size and memory buffer size don't match:";
                    return false;
                }
                auto status = memcpy(unsafe_input(input_id), (void*)data.data(), buffsize);

                if (status == nullptr)
                {
                    LOG_ERR << "Memcpy failed.";
                    return false;
                }

                return true;
            }

            template <typename T>
            bool load_inputs(const vector<vector<T>>& data)
            {
                if (data.size() != kctx->inputs.size())
                {
                    LOG_ERR << "Data items missmatch.";
                    return false;
                }
                for (int i = 0; i < data.size(); i++)
                {
                    if (load_input_from(data[i], i) == false)
                        return false;
                }
                return true;
            }

            template <typename T>
            vector<T> save_output(int output_id)
            {
                if (output_id > raw_output.size())
                {
                    LOG_ERR << "Index exceeded the limit of vector.";
                    return vector<T>();
                }
                auto base = (T*)unsafe_output(output_id);
                auto buffsize = kctx->outputs[output_id].get_size();
                vector<T> res(base, base + buffsize);
                return move(res);
            }

            template <typename T>
            vector<vector<T>> save_outputs()
            {
                vector<vector<T>> res;
                for (int i = 0; i < kctx->outputs.size(); i++)
                    res.push_back(save_output<T>(i));
                return res;
            }

            void* unsafe_input(int n)
            {
                if (n > raw_input.size())
                {
                    LOG_ERR << "Index exceeded the limit of vector.";
                    return nullptr;
                }
                return raw_input[n].get();
            }

            void* unsafe_output(int n)
            {
                if (n > raw_output.size())
                {
                    LOG_ERR << "Index exceeded the limit of vector.";
                    return nullptr;
                }
                return raw_output[n].get();
            }

            //\brief At last, returned pointer shoule be translated into "T*[]" at runtime.
            //\todo (wenxh)potential bug here, pointer may be used but deallocated.
            void** unsafe_inputs()
            {
                raw_inputs.reset(new char*[kctx->inputs.size()]);
                for (int i = 0; i < kctx->inputs.size(); i++)
                    raw_inputs.get()[i] = (char*)unsafe_input(i);
                return (void**)raw_inputs.get();
            }

            void** unsafe_outputs()
            {
                raw_outputs.reset(new char*[kctx->outputs.size()]);
                for (int i = 0; i < kctx->outputs.size(); i++)
                    raw_outputs.get()[i] = (char*)unsafe_output(i);
                return (void**)raw_outputs.get();
            }

        private:
            kernels::KernelContext::Pointer kctx;
            unique_ptr<char *> raw_inputs, raw_outputs;
            vector<shared_ptr<char>> raw_input, raw_output;
        };

        //\brief The Context will have some basic info like:
        // -Input: Zeros, Ones, Randoms or Other Data.
        // -Output(optional): To check the output is right.
        // -Subject: Profile what subject.
        // -(Warmup)Times: .
        struct ProfilingContext
        {
        public:
            using Pointer = shared_ptr<ProfilingContext>;
            string working_dir = "profile/";
            const size_t warmup_times = 5;
            const size_t host_times = 1;
            const size_t runtime_times = 10000;
            // This emitter includes the kernel context;
            ProfilingResult result;
            kernels::KernelEmitter::Pointer kernel;
            ProfilingContext(kernels::KernelEmitter::Pointer kernel)
            {
                this->kernel = kernel;
                kernel_memory.reset(new KernelMemory(kernel->m_context));
            }
            //\todo source code and function pointer need moved into cache;
            LanguageUnit_p source_code = nullptr;
            double (*entry_point)(void**, void**) = nullptr;
            //\todo To be deprecated in future;
            // ProfilingContext(shared_ptr<ngraph::Node> node) { ; }
            KernelMemory::Pointer kernel_memory;

            void reset()
            {
                source_code = nullptr;
                entry_point = nullptr;
                result.reset();
                // kernel_memory.release();
            }
        };

        //\brief Restricted feature: Only support evaluation of result insteading of profiling.
        //\todo (wenxh) support full-feature profiling, this to be done with new codegen.
        struct GraphEvaluationContext
        {
            shared_ptr<Graph> graph = nullptr;
            GraphEvaluationContext(shared_ptr<Graph> pGraph) { graph = pGraph; };
            void reset() { graph = nullptr; }
            //\brief This function will generate a reference kernel for the GNode
            void set_profiling_context(shared_ptr<GNode> gnode, ProfilingContext::Pointer kctx)
            {
                // Need to check unique_name wether it works.
                if (prof_cache.find(gnode->get_unique_name()) != prof_cache.end())
                    prof_cache[gnode->get_unique_name()] = kctx;
            }

            ProfilingContext::Pointer get_profiling_context(shared_ptr<GNode> gnode)
            {
                if (prof_cache.find(gnode->get_unique_name()) != prof_cache.end())
                    return prof_cache[gnode->get_unique_name()];
                else
                {
                    LOG_ERR << "No valid Profiling Context for this node.";
                    return nullptr;
                }
            }

        private:
            //\brief To store the output constant by the kernel.
            unordered_map<string, ProfilingContext::Pointer> prof_cache;
        };

        //\brief The inteface for profiler runtime, which is binding to Device type.
        // Each device type should have one or more runtime.
        class IProfilingRuntime
        {
        public:
            //\todo This interface is not safe, may access invlid memory address.
            bool execute(const ProfilingContext::Pointer& ke);
            virtual double
                execute(const ProfilingContext::Pointer& ke, void** input, void** output);
            // Get the result of last run;
            using Pointer = shared_ptr<IProfilingRuntime>;

            /*
            //\todo To be provided in future, since we cannot use runtime api here.
            // We use Profiler class as Host here.
            virtual void* create_tensor(size_t bytes_size) = 0;
            virtual bool memcpyHtoD(void* host, void* device, size_t bytes_size) = 0;
            virtual bool memcpyDtoH(void* device, void* host, size_t bytes_size) = 0;
            */
        };
    }
}