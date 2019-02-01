// Microsoft (c) 2019, Wenxiang
#include "cuda_codegen.hpp"

using namespace nnfusion::cuda;

bool CudaCodeGenPass::run(ir::Operator_p& inter_op)
{
    const std::map<type_index, function<CudaFunction_p(ir::Operator_p)>> typeid_map{
        {type_index(typeid(ngraph::op::Result)), Result::codegen},
        {type_index(typeid(ngraph::op::Parameter)), Noop::codegen},
        {type_index(typeid(ngraph::op::Constant)), ConstantNaive::codegen},
        {type_index(typeid(ngraph::op::Broadcast)), Broadcast::codegen},
        {type_index(typeid(ngraph::op::MaxPool)), MaxPool::codegen},
        {type_index(typeid(ngraph::op::Dot)), Dot::codegen},
        {type_index(typeid(ngraph::op::Reshape)), Reshape::codegen},
        {type_index(typeid(ngraph::op::Relu)), Elementwise<ngraph::op::Relu>::codegen},
        {type_index(typeid(ngraph::op::Add)), Elementwise<ngraph::op::Add>::codegen},
        {type_index(typeid(ngraph::op::Abs)), Elementwise<ngraph::op::Abs>::codegen},
        {type_index(typeid(ngraph::op::Subtract)), Elementwise<ngraph::op::Subtract>::codegen},
        {type_index(typeid(ngraph::op::Multiply)), Elementwise<ngraph::op::Multiply>::codegen},
    };
    auto& node = *(inter_op->node);
    auto it = typeid_map.find(type_index(typeid(node)));
    CudaFunction_p cop(nullptr);
    if (it == typeid_map.end())
    {
        NGRAPH_DEBUG << "Unsupported op '" << node.description() << "', using Anyop." << endl;
        cop = Anyop::codegen(inter_op);
    }
    else
    {
        NGRAPH_DEBUG << "Codegen op '" << node.description() << "'" << endl;
        cop = it->second(inter_op);
    }
    assert_nullptr(cop);
    auto cw = cop->codegen_source();
    assert_nullptr(cw);
    //Replacing the inter_op with CodegenOP
    inter_op = cop;
    return true;
}

bool CudaCodeGenerator::codegen(shared_ptr<TranslationUnit> tu)
{
    bool rc = true;
    auto& inter_ops = *(tu->inter_ops);
    for (auto& op : inter_ops)
    {
        auto base = static_cast<CodeGenerator*>(this);
        rc = base->codegen(op);
        assert_bool(rc);
        if (!rc)
            return rc;
    }
}

bool NaiveCudaCodeGenerator::codegen(shared_ptr<TranslationUnit> tu)
{
    bool rc = true;
    auto& inter_ops = *(tu->inter_ops);
    for (auto& op : inter_ops)
    {
        auto base = static_cast<CodeGenerator*>(this);
        rc = base->codegen(op);
        assert_bool(rc);
        if (!rc)
            return rc;
    }

    NGRAPH_DEBUG << "Start dump whole source file...\n";
    // Code Gen
    LanguageUnit lu("naive_test.cu");
    lu << "// Microsoft (c) 2019, Wenxiang\n";

    // Collect Requirement
    unordered_set<string> global_required;
    {
        LanguageUnit re("REQUIREMENT");
        re.require(declaration::typedef_int);
        re.require(header::stdexcept);
        re.require(header::sstream);
        re.require(macro::CUDA_SAFE_CALL);
        for (auto& op : inter_ops)
        {
            auto base = static_pointer_cast<CudaFunction>(op);
            if (base == nullptr || base->is_codegened() == false)
                return false;
            for (auto& it : base->dep_unit->local_symbol)
            {
                re.require(it.second);
                global_required.insert(it.second->symbol);
            }
        }

        lu << re.collect_required_code();
    }

    lu << "#include <assert.h>\n";

    // Collect Function Definition
    {
        unordered_set<string> declared;
        LanguageUnit def("FUNCTIONS");
        for (auto& op : inter_ops)
        {
            auto base = static_pointer_cast<CudaFunction>(op);
            if (base == nullptr || base->is_codegened() == false)
                return false;
            for (auto& it : base->definition_unit->local_symbol)
            {
                if (it.second != base->dep_unit)
                    def.require(it.second);
            }
            def << base->gen_comments();
            if (declared.count(base->definition_unit->symbol) == 0)
            {
                def << base->definition_unit->get_code() << "\n";
                declared.insert(base->definition_unit->symbol);
            }
            else
            {
                def << "// Function declared:" << base->definition_unit->symbol << "\n\n";
            }
        }
        lu << def.collect_code() << "\n";
    }

    // Generate function body
    {
        unordered_set<string> allocated;
        lu << "extern \"C\" int naive_entry(";
        // Add param
        {
            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name();
                allocated.insert(tv->get_name());
                params.push_back(ss.str());
            }

            lu << join(params, ", ");
        }
        lu << ")\n";
        lu.block_begin();

        //Planning
        {
            // assert_bool(tu->memory_pool_size > 0) << "GPU Memory pool size cannot be zero.";
            lu << "char* _memory_pool;\n"
               << "CUDA_SAFE_CALL(cudaMalloc((void**)&_memory_pool, " << tu->memory_pool_size
               << "));\n";

            for (auto& op : inter_ops)
            {
                auto base = static_pointer_cast<CudaFunction>(op);
                if (base == nullptr || base->is_codegened() == false)
                    return false;
                for (auto& it : base->op->args)
                {
                    if (allocated.count(it.get_name()) > 0)
                        continue;
                    lu << it.get_type() << "* " << it.get_name() << " = (" << it.get_type()
                       << "*)(_memory_pool+" << it.get_offset() << ");\n";
                    allocated.insert(it.get_name());
                }

                for (auto& it : base->op->out)
                {
                    if (allocated.count(it.get_name()) > 0)
                        continue;
                    lu << it.get_type() << "* " << it.get_name() << " = (" << it.get_type()
                       << "*)(_memory_pool+" << it.get_offset() << ");\n";
                    allocated.insert(it.get_name());
                }
            }
        }

        //Function Call
        {
            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu << "CUBLAS_SAFE_CALL(cublasCreate(&global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu << "CUDNN_SAFE_CALL(cudnnCreate(&global_cudnn_handle));\n";
            }
            for (auto& op : inter_ops)
            {
                auto base = static_pointer_cast<CudaFunction>(op);
                lu << base->call_unit->get_code();
                // lu << "assert(cudaSuccess == cudaGetLastError());\n";
            }
            if (global_required.count("declaration::global_cublas_handle") > 0)
            {
                lu << "CUBLAS_SAFE_CALL(cublasDestroy(global_cublas_handle));\n";
            }
            if (global_required.count("declaration::global_cudnn_handle") > 0)
            {
                lu << "CUDNN_SAFE_CALL(cudnnDestroy(global_cudnn_handle));\n";
            }
        }
        lu << "return 0;\n";
        lu.block_end();
    }

    lu << "\n";

    // Test function
    {
        lu << "extern \"C\" int naive_test(";
        // Add param
        {
            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name() << "_host";
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << type << "* " << tv->get_name() << "_host";
                params.push_back(ss.str());
            }

            lu << join(params, ", ");
        }
        lu << ")\n";
        lu.block_begin();
        {
            for (size_t i = 0; i < tu->arg.size(); i++)
            {
                auto& tensor = *tu->arg[i];
                lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                   << ";\n"
                   << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                   << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << "));\n";

                lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << ", " << tensor.get_name()
                   << "_host, " << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyHostToDevice));\n";
            }

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu << tensor.get_element_type().c_type_string() << "* " << tensor.get_name()
                   << ";\n"
                   << "CUDA_SAFE_CALL(cudaMalloc((void**)&" << tensor.get_name() << ","
                   << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << "));\n";
            }

            vector<string> params;
            for (int i = 0; i < tu->arg.size(); i++)
            {
                auto& tv = tu->arg[i];
                params.push_back(tv->get_name());
            }

            for (int i = 0; i < tu->out.size(); i++)
            {
                auto& tv = tu->out[i];
                params.push_back(tv->get_name());
            }

            lu << "naive_entry(" << join(params, ", ") << ");\n";

            for (size_t i = 0; i < tu->out.size(); i++)
            {
                auto& tensor = *tu->out[i];
                lu << "CUDA_SAFE_CALL(cudaMemcpy(" << tensor.get_name() << "_host, "
                   << tensor.get_name() << ", " << tensor.get_tensor_layout()->get_size() << " * "
                   << tensor.get_element_type().size() << ", "
                   << "cudaMemcpyDeviceToHost));\n";
            }
        }
        lu << "return 0;\n";
        lu.block_end();
    }

    lu << "\n";

    // Test function 2
    {
        lu << "extern \"C\" int naive_test_simple(void** args)\n";
        // Add param
        lu.block_begin();
        {
            lu << "return naive_test(";
            vector<string> params;
            int acc = 0;
            for (int i = 0; i < tu->arg.size(); i++, acc++)
            {
                auto tv = tu->arg[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << "(" << type << "*)args[" << acc << "]";
                params.push_back(ss.str());
            }

            for (int i = 0; i < tu->out.size(); i++, acc++)
            {
                auto tv = tu->out[i];
                string type = tv->get_element_type().c_type_string();
                stringstream ss;
                ss << "(" << type << "*)args[" << acc << "]";
                params.push_back(ss.str());
            }
            lu << join(params, ", ");
            lu << ");\n";
        }
        lu.block_end();
    }

    ofstream source_file(lu.symbol);
    source_file << lu.get_code();
    source_file.close();
    return rc;
}
