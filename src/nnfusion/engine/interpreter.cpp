// Microsoft (c) 2019, Wenxiang Hu
#include "interpreter.hpp"
#include "nnfusion/engine/pass/cpu_codegenerator.hpp"
#include "nnfusion/engine/pass/cuda_codegenerator.hpp"
#include "nnfusion/engine/pass/extract_graph_signature.hpp"
#include "nnfusion/engine/pass/rocm_codegenerator.hpp"

#include <strings.h>
#include "nnfusion/common/descriptor/layout/dense_tensor_layout.hpp"
#include "nnfusion/engine/async_manager.hpp"
#include "pass/tensor/inplace_tensor_analysis.hpp"
#include "pass/tensor/liveness_analysis.hpp"
#include "pass/tensor/tensor_device_dispatcher.hpp"
#include "pass/tensor/tensor_memory_layout.hpp"
using namespace nnfusion::pass;

DECLARE_string(fdefault_device);
DEFINE_bool(fcuda_kernels_as_files, false, "Saving cuda kernels as standalone source code files.");
DEFINE_int64(fcuda_kernels_files_number,
             -1,
             "Saving cuda kernels into how many source code files.");

DEFINE_bool(fkernels_as_files, false, "Saving kernels as standalone source code files.");
DEFINE_int64(fkernels_files_number, -1, "Saving kernels into how many source code files.");

Interpreter::Interpreter()
    : m_trans_ctx(new InterpreterContext())
    , m_passes(new vector<shared_ptr<IInterpreterPass>>())
{
    // Todo: find another way
    auto dev_name = FLAGS_fdefault_device.c_str();
    NNFusion_DeviceType default_device = nnfusion::get_device_type(dev_name);

    // To be compatible with former cli
    //Todo(wenxh): Remove this;
    FLAGS_fkernels_as_files = FLAGS_fkernels_as_files || FLAGS_fcuda_kernels_as_files;
    FLAGS_fkernels_files_number =
        max(FLAGS_fkernels_files_number, FLAGS_fcuda_kernels_files_number);

    // kernel selection
    // m_passes->push_back(make_shared<DefaultDeviceDispatcher>());
    // m_passes->push_back(make_shared<ProfilingBasedKernelSelector>());
    // m_passes->push_back(make_shared<DefaultKernelSelector>());
    m_passes->push_back(make_shared<TensorDeviceDispatcher>());
    m_passes->push_back(make_shared<TensorLivenessAnalysis>());
    m_passes->push_back(make_shared<InplaceTensorAnalysis>());
    m_passes->push_back(make_shared<AssignTensorMemoryLayout>(64, false));

    switch (default_device)
    {
    case CUDA_GPU: m_passes->push_back(make_shared<CudaCodeGenerator>()); break;

    case GENERIC_CPU: m_passes->push_back(make_shared<CpuCodeGenerator>()); break;

    case ROCM_GPU:
        FLAGS_fcuda_kernels_as_files = false;
        m_passes->push_back(nnfusion::make_rocm_codegenerator());
        break;

    default: m_passes->push_back(make_shared<CudaCodeGenerator>()); break;
    }
}

Interpreter::Interpreter(shared_ptr<vector<shared_ptr<IInterpreterPass>>> passes,
                         shared_ptr<InterpreterContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

bool Interpreter::translate(TranslationUnit::Pointer tu)
{
    NNFUSION_CHECK_NOT_NULLPTR(m_passes);
    return IInterpreterPass::run_passes(*m_passes, m_trans_ctx, tu);
}

shared_ptr<TranslationUnitMap> Interpreter::translate(shared_ptr<graph::Graph> graph)
{
    // run graph passes
    nnfusion::pass::graph::GraphPass graph_passes;
    NNFUSION_CHECK(graph_passes.run(graph));

    // TODO : multi graph ?
    m_trans_ctx->m_graphs.insert(graph);

    // Iterator through all nodes
    static interpreter::ExtractGraphSignature extract_global;
    shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());

    // Deal with translation unit's program
    for (const auto& current_graph : m_trans_ctx->m_graphs)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        _tus->emplace(current_graph, _tu);
        NNFUSION_LOG(INFO) << "Translating graph:\t" << current_graph->get_name();

        _tu->program = nnfusion::ir::Program::create_single_basic_block_program();
        _tu->m_graph = current_graph;
        auto bb_main = _tu->program.get_entry();

        // extract output_names/constants/arg/out for _tu, m_variable_name_map for m_trans_ctx
        NNFUSION_CHECK(extract_global.run(m_trans_ctx, _tu))
            << "Error when extract global graph info.";

        // Translate the Node
        for (auto gnode : graph->get_ordered_ops())
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(gnode);
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                nnfusion::ir::Instruction::Pointer ir(new nnfusion::ir::Instruction);
                ir->setGNode(gnode);

                // Tag example
                {
                    auto& INS = *ir;
                    INS["DEBUG"] = 1;
                    auto res = INS["DEBUG"].as<int>();
                }

                // move all tags on the node to the intruction
                {
                    ir->copy_tags_from(*gnode);
                }

                ir->setName(gnode->get_name());
                bb_main->push_back(ir);

                // add memcpy ir and async info if the gnode and its output gnodes are in diffrent devices
                add_memcpy_ir(gnode, bb_main);
            }
        }

        /*

        for (auto& ins : *bb_main)
        {
            std::stringstream ss;
            ss << ins->name() << "\t { ";
            ss << "INPUT:{";
            for(auto& in: ins->Attr().ts("INPUT"))
            {
                ss << in.get_name() << ", ";
            }
            ss << "}, ";
            ss << "OUTPUT:{";
            for(auto& in: ins->Attr().ts("OUTPUT"))
            {
                ss << in.get_name() << ", ";
            }
            ss << "}, (tag:)";
            ss << " DEBUG : " << ins->Tag().Get<int>("DEBUG") << " }";
            NNFUSION_LOG(INFO) << ss.str();
        }
         */
        translate(_tu);
    }
    NNFUSION_LOG(INFO) << "-------------------translate graph done.";
    return _tus;
}

void Interpreter::add_memcpy_ir(shared_ptr<nnfusion::graph::GNode> gnode,
                                nnfusion::ir::BasicBlock::Pointer bb_main)
{
    auto CUDA_async_manager = nnfusion::async::AsyncManagerFactory::get_async_manager(CUDA_GPU);
    auto CPU_async_manager = nnfusion::async::AsyncManagerFactory::get_async_manager(GENERIC_CPU);
    std::unordered_map<int, nnfusion::ir::Instruction::Pointer> dev_ir;
    auto n_device_type = (*gnode)["DeviceType"].as<NNFusion_DeviceType>();
    auto n_device_id = (*gnode)["DeviceID"].as<int>();
    for (auto& out_edge : gnode->get_out_edges())
    {
        if (!out_edge->is_control_edge())
        {
            auto out_gnode = out_edge->get_dst();

            auto out_device_type = (*out_gnode)["DeviceType"].as<NNFusion_DeviceType>();
            auto out_device_id = (*out_gnode)["DeviceID"].as<int>();

            if (n_device_type != out_device_type ||
                (n_device_id != out_device_id && n_device_type == GENERIC_CPU))
            {
                throw nnfusion::errors::NotSupported("Cross-DeviceType ir is not supported.");
            }
            else if (n_device_id != out_device_id)
            {
                nnfusion::ir::Instruction::Pointer memcpy_ir;
                auto idx = out_edge->get_dst_input();
                if (dev_ir.find(out_device_id) == dev_ir.end())
                {
                    auto& async_info =
                        (*gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                    auto thread = async_info.execution_thread;
                    auto stream = async_info.execution_stream;
                    memcpy_ir = std::make_shared<nnfusion::ir::Instruction>();
                    memcpy_ir->setName("Memcpy");

                    // set device id and device type
                    (*memcpy_ir)["DeviceType"] = n_device_type;
                    (*memcpy_ir)["DeviceID"] = n_device_id;

                    auto tensor = out_gnode->get_input_tensor_ptr(idx);
                    auto element_type = tensor->get_element_type();
                    auto pshape = tensor->get_partial_shape();
                    auto name = tensor->get_name();
                    std::string new_name = name + "_" + get_device_str(out_device_type) +
                                           std::to_string(out_device_id);
                    // create new tensor
                    std::shared_ptr<descriptor::Tensor> new_tensor =
                        std::make_shared<descriptor::Tensor>(element_type, pshape, new_name);
                    // set tensor layout
                    auto layout = std::make_shared<nnfusion::descriptor::layout::DenseTensorLayout>(
                        *new_tensor);
                    new_tensor->set_tensor_layout(layout);

                    auto& inputs = memcpy_ir->get_inputs();
                    NNFUSION_CHECK(inputs.empty());
                    inputs.push_back(tensor);

                    auto& outputs = memcpy_ir->get_outputs();
                    NNFUSION_CHECK(outputs.empty());
                    outputs.push_back(new_tensor);

                    // add async info
                    (*memcpy_ir)["Async_info"] = nnfusion::async::AsyncExecutionInfo();
                    auto& memcpy_async_info =
                        (*memcpy_ir)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                    // set thread and stream
                    if (gnode->get_op_ptr()->is_tensor_op())
                    {
                        new_tensor->set_persistent();
                    }
                    if (gnode->is_constant() || gnode->is_variable())
                    {
                        (*memcpy_ir)["Memcpy_Constant_or_Variable"] = true;
                        // constant ops are in xxx_init(),
                        // so thre is no need to add event or barrier.
                        NNFUSION_CHECK(thread->is_default_stream());
                        NNFUSION_CHECK(stream->is_default_stream());
                        // use default thread
                        memcpy_async_info.execution_thread = thread;
                        // use default stream
                        memcpy_async_info.execution_stream = stream;
                    }
                    else
                    {
                        // currently use the same thread, could use a new different thread as well.
                        if (gnode->is_parameter())
                            memcpy_async_info.execution_thread =
                                CPU_async_manager->set_stream(n_device_id, "memcpy");
                        else
                            memcpy_async_info.execution_thread = thread;
                        // use a new different stream.
                        memcpy_async_info.execution_stream =
                            CUDA_async_manager->set_stream(n_device_id, "memcpy_" + new_name);
                    }
                    if (memcpy_async_info.execution_thread != thread &&
                        async_info.notify_barrier != nullptr)
                    {
                        memcpy_async_info.wait_barriers.push_back(async_info.notify_barrier);
                    }
                    if (memcpy_async_info.execution_stream != stream &&
                        async_info.record_event != nullptr)
                    {
                        memcpy_async_info.wait_events.push_back(async_info.record_event);
                    }
                    // gnode->liveness_new_list.insert(new_tensor);
                    bb_main->push_back(memcpy_ir);
                    dev_ir[out_device_id] = memcpy_ir;
                }
                else
                {
                    memcpy_ir = dev_ir[out_device_id];
                }
                auto& outputs = memcpy_ir->get_outputs();
                NNFUSION_CHECK(outputs.size() == 1);
                auto new_tensor = outputs[0];
                auto new_name = new_tensor->get_name();

                // assumption: tensor's index in gnode is the same as in kernel
                if (!out_gnode->get_op_ptr()->is_parameter() &&
                    !out_gnode->get_op_ptr()->is_output() &&
                    !out_gnode->get_op_ptr()->is_constant())
                {
                    auto emitted_kernel =
                        (*out_gnode)["Kernel_Selection_Result"]
                            .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
                    if (emitted_kernel.second->get_or_emit_source() == nullptr)
                    {
                        NNFUSION_CHECK_FAIL() << "Kernel should be emitted before this pass:"
                                              << out_gnode->get_name();
                    }
                    auto out_kernel = emitted_kernel.second;
                    auto& out_kernel_inputs = out_kernel->m_context->inputs;
                    auto& out_kernel_input_names = out_kernel->m_context->input_names;

                    out_kernel_inputs.erase(out_kernel_inputs.begin() + idx);
                    out_kernel_inputs.insert(out_kernel_inputs.begin() + idx, new_tensor);
                    out_kernel_input_names.erase(out_kernel_input_names.begin() + idx);
                    out_kernel_input_names.insert(out_kernel_input_names.begin() + idx, new_name);
                }
                else
                {
                    // parameter and constant nodes have no inputs,
                    // and output nodes kernel is simply a reference operation,
                    // so there should be no memcpy between these nodes and their input nodes.
                }

                // add waiting event and barrier to the out gnodes
                // constant or parameter memcpy ir are in xxx_init(),
                // so there is no need to add event or barrier.
                if ((*memcpy_ir)["Memcpy_Constant_or_Variable"].is_valid() &&
                    (*memcpy_ir)["Memcpy_Constant_or_Variable"].as<bool>())
                    continue;
                auto& memcpy_async_info =
                    (*memcpy_ir)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                auto& out_async_info =
                    (*out_gnode)["Async_info"].as<nnfusion::async::AsyncExecutionInfo>();
                if (memcpy_async_info.execution_thread != out_async_info.execution_thread)
                {
                    if (!memcpy_async_info.notify_barrier)
                    {
                        memcpy_async_info.notify_barrier = CPU_async_manager->set_event(
                            memcpy_async_info.execution_thread, "memcpy_" + new_name);
                    }
                    out_async_info.wait_barriers.push_back(memcpy_async_info.notify_barrier);
                }
                if (memcpy_async_info.execution_stream != out_async_info.execution_stream)
                {
                    if (!memcpy_async_info.record_event)
                    {
                        memcpy_async_info.record_event = CUDA_async_manager->set_event(
                            memcpy_async_info.execution_stream, "memcpy_" + new_name);
                    }
                    out_async_info.wait_events.push_back(memcpy_async_info.record_event);
                }
            }
        }
    }
}
