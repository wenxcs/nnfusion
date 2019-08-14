// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Basic Datastructure used in profiling
 * \author wenxh
 */

#include "profiling_runtime.hpp"

using namespace nnfusion::profiler;

bool IProfilingRuntime::execute(const ProfilingContext::Pointer& ke)
{
    auto kctx = ke->kernel->m_context;
    size_t buffer_size = 0;
    for (auto& t : kctx->inputs)
        buffer_size += t.get_size() * t.get_element_type().size();
    for (auto& t : kctx->outputs)
        buffer_size += t.get_size() * t.get_element_type().size();

    char* buffer = new char[buffer_size];
    void** inputs = new void*[kctx->inputs.size()];
    void** outputs = new void*[kctx->inputs.size()];

    // Assign all the tensor in the buffer space.
    size_t offset = 0;
    size_t index = 0;
    for (auto& t : kctx->inputs)
    {
        inputs[index++] = buffer + offset;
        offset += t.get_size() * t.get_element_type().size();
    }
    index = 0;
    for (auto& t : kctx->outputs)
    {
        outputs[index++] = buffer + offset;
        offset += t.get_size() * t.get_element_type().size();
    }

    bool ret = execute(ke, inputs, outputs);

    delete inputs;
    delete outputs;
    delete buffer;

    return ret;
}

double IProfilingRuntime::execute(const ProfilingContext::Pointer& ke, void** input, void** output)
{
    if (ke->entry_point == nullptr)
        return -1.0;
    return ke->entry_point(input, output);
}