// Microsoft (c) 2019, Wenxiang Hu
#include "kernel_selection.hpp"

#include <queue>
#include <utility>

using namespace nnfusion;
using namespace nnfusion::profiler;

// Register_Tag(Enable_Kernel_Selection, bool);
// Register_Tag(Kernel_Selection_Device, DeviceType);
// Register_Tag(Kernel_Selection_Result, vector<pair<DeviceType, KernelEmitter>>);

DEFINE_bool(fkernel_selection, true, "Select kernel before codegen.");
DEFINE_bool(fkernel_tunning, false, "Tunning and choose best kernel when do kernel selection.");

pair<DeviceType, kernels::KernelEmitter::Pointer> ProfilingBasedKernelSelector::profiling_best(
    shared_ptr<ngraph::Node> node, DeviceType devtype, IProfilingRuntime::Pointer runtime)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(node->description(), devtype, DT_FLOAT);

    // Skip since only one candidate or constant
    if (kernel_regs.size() == 1 || node->is_constant())
        return std::make_pair(devtype, nullptr);

    shared_ptr<KernelContext> ctx(new KernelContext(node));

    bool has_valid_kernel = false;
    LOG(INFO) << "Start profiling...";
    auto comparef = [](const ProfilingContext::Pointer& a, const ProfilingContext::Pointer& b) {
        return a->result.get_device_avg() > b->result.get_device_avg();
    };
    priority_queue<ProfilingContext::Pointer, vector<ProfilingContext::Pointer>, decltype(comparef)>
        prof_res(comparef);
    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (kernel->get_or_emit_source())
        {
            has_valid_kernel = true;
            auto pctx = make_shared<nnfusion::profiler::ProfilingContext>(kernel);
            nnfusion::profiler::Profiler prof(runtime, pctx);

            if (!prof.execute())
                LOG(INFO) << "Kernel Failed.";
            else
            {
                LOG(INFO) << "Kernel Emitter#" << prof_res.size()
                          << " time cost(ms):" << pctx->result.get_device_avg();
                prof_res.push(pctx);
            }
        }
    }

    while (!prof_res.empty())
    {
        auto best = prof_res.top();
        prof_res.pop();
        ///\todo Check if the result is ready.
        if (!best->result.is_ready())
            continue;
        LOG(INFO) << "Best kernel time cost(ms):" << best->result.get_device_avg();
        return std::make_pair(devtype, move(best->kernel));
    }
    return std::make_pair(devtype, nullptr);
}

bool ProfilingBasedKernelSelector::run(std::shared_ptr<InterpreterContext> ctx,
                                       std::shared_ptr<TranslationUnit> tu)
{
    bool enable_tuning = FLAGS_fkernel_tunning;
    if (!enable_tuning)
        return true;

    // Config area
    vector<string> white_list{"Broadcast"};
    bool all_device = false;
    DeviceType the_device = ROCM_GPU;

    // Currently *ONLY* has BroadCast Selection
    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            auto opname = ins->operatorDef()->get_name();
            for (auto& rule : white_list)
                if (opname.find(rule) < opname.size())
                {
                    (*ins)["Enable_Kernel_Selection"] = true;
                    if (!all_device)
                        (*ins)["Kernel_Selection_Device"] = the_device;
                }
        }
    }

    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            if ((*ins)["Enable_Kernel_Selection"].is_valid() &&
                (*ins)["Enable_Kernel_Selection"].as<bool>())
            {
                (*ins)["Kernel_Selection_Result"] =
                    vector<pair<DeviceType, KernelEmitter::Pointer>>();
                auto& res = (*ins)["Kernel_Selection_Result"]
                                .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();

                vector<DeviceType> dev_type{CUDA_GPU, ROCM_GPU, GENERIC_CPU};
                for (auto t : dev_type)
                {
                    if ((*ins)["Kernel_Selection_Device"].is_valid() &&
                        (*ins)["Kernel_Selection_Device"].as<DeviceType>() != t)
                        continue;

                    auto ans = profiling_best(ins->operatorDef(), t, get_default_runtime(t));

                    if (ans.second != nullptr)
                        res.push_back(ans);
                }
            }
        }
    }
    return true;
}

pair<DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first(shared_ptr<ngraph::Node> node, DeviceType devtype)
{
    std::vector<shared_ptr<const KernelRegistration>> kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(node->description(), devtype, DT_FLOAT);
    shared_ptr<KernelContext> ctx(new KernelContext(node));

    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        // constant kernel emitter will write file to save weights, skip to do it when codegen.
        if (node->is_constant() || kernel->get_or_emit_source())
        {
            // if(kernel->get_or_emit_source() != nullptr)
            //    LOG(WARNING) << "Valid kernel found:" << node->get_name();
            return std::make_pair(devtype, kernel);
        }
    }
    LOG(ERROR) << "No valid kernel found:" << node->get_name();
    return std::make_pair(devtype, nullptr);
}

pair<DeviceType, kernels::KernelEmitter::Pointer>
    DefaultKernelSelector::pick_first_rocm(shared_ptr<ngraph::Node> node)
{
    shared_ptr<KernelContext> ctx(new KernelContext(node));
    auto kernel_regs =
        KernelRegistry::Global()->FindKernelRegistrations(node->description(), ROCM_GPU, DT_FLOAT);
    if (!kernel_regs.size())
        kernel_regs = KernelRegistry::Global()->FindKernelRegistrations(
            node->description(), CUDA_GPU, DT_FLOAT);
    else
    {
        auto priority = [](const std::string& tag) -> int {
            static char sym_prio[] = "PRIORITY_";
            int at = tag.find(sym_prio);
            return (at != 0) ? 0 : atoi(tag.substr(sizeof(sym_prio) - 1).c_str());
        };

        std::sort(kernel_regs.begin(),
                  kernel_regs.end(),
                  [&](const shared_ptr<const KernelRegistration>& x,
                      const shared_ptr<const KernelRegistration>& y) {
                      auto x_prio = priority(x->m_tag), y_prio = priority(y->m_tag);
                      if (x_prio != y_prio)
                          return x_prio > y_prio;

                      auto x_type = x->m_factory(ctx)->get_kernel_type();
                      auto y_type = y->m_factory(ctx)->get_kernel_type();
                      if (x_type != y_type)
                          return x_type < y_type;

                      return false;
                  });
    }

    for (auto kernel_reg : kernel_regs)
    {
        auto kernel = kernel_reg->m_factory(ctx);
        if (node->is_constant() || kernel->get_or_emit_source())
        {
            return std::make_pair(ROCM_GPU, kernel);
        }
    }
    LOG(ERROR) << "No valid kernel found:" << node->get_name();
    return std::make_pair(ROCM_GPU, nullptr);
}

bool DefaultKernelSelector::run(std::shared_ptr<InterpreterContext> ctx,
                                std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;
    for (auto iterator : p)
    {
        for (auto ins : *iterator)
        {
            if (!(*ins)["Kernel_Selection_Result"].is_valid())
                (*ins)["Kernel_Selection_Result"] =
                    vector<pair<DeviceType, KernelEmitter::Pointer>>();
            auto& res = (*ins)["Kernel_Selection_Result"]
                            .as<vector<pair<DeviceType, KernelEmitter::Pointer>>>();

            vector<DeviceType> dev_type{CUDA_GPU, ROCM_GPU, GENERIC_CPU};
            for (auto t : dev_type)
            {
                if ((*ins)["Kernel_Selection_Device"].is_valid() &&
                    (*ins)["Kernel_Selection_Device"].as<DeviceType>() != t)
                    continue;

                bool selected = false;
                for (auto& p : res)
                {
                    if (p.first == t)
                    {
                        selected = true;
                        break;
                    }
                }
                if (selected)
                    continue;

                if (t == ROCM_GPU)
                {
                    auto ans = pick_first_rocm(ins->operatorDef());
                    res.push_back(ans);
                }
                else
                {
                    auto ans = pick_first(ins->operatorDef(), t);
                    res.push_back(ans);
                }
            }
        }
    }
    return true;
}
