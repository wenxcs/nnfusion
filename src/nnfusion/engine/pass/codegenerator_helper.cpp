// Microsoft(c)2019, NNFusion Team

#include "codegenerator_helper.hpp"

#include "nnfusion/core/kernels/cuda_gpu/cuda_langunit.hpp"
using namespace nnfusion;
using namespace nnfusion::codegenerator;
using namespace nnfusion::kernels;

static std::string suffix_str = ".cu";

LanguageUnit_p extern_function(LanguageUnit_p lu)
{
}

LanguageUnit_p extern_variable(LanguageUnit_p lu)
{
}

void FunctionFile::save_file()
{
    LanguageUnit def_re("require");
    def_re << "// Microsoft (c) 2019, NNFusion\n";

    // Write Dependency
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("header::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";

    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("macro::") != string::npos)
            def_re << it.second->get_code();
    def_re << "\n";
    for (auto& it : this->local_symbol)
        if (it.second->symbol.find("declaration::") != string::npos)
        {
            if (it.second->get_symbol() == "declaration::global_cublas_handle" ||
                it.second->get_symbol() == "declaration::global_cudnn_handle" ||
                it.second->get_symbol() == "declaration::num_SMs" ||
                it.second->get_symbol() == "declaration::allreduce_stream" ||
                it.second->get_symbol() == "declaration::applygradient_stream")
            {
                def_re << "extern ";
            }
            def_re << it.second->get_code();
        }
    def_re << "\n";

    string fname = this->get_symbol();
    if (fname.length() > 128)
    {
        size_t hashcode = std::hash<std::string>{}(fname);
        fname = "kernels/compressed_src_" + std::to_string(hashcode) + suffix_str;
    }
    else
        fname = "kernels/" + this->get_symbol() + suffix_str;

    ofstream src(fname);
    src << def_re.get_code();
    src << "\n";
    src << this->get_code();
    src.close();
}

void FunctionFile::merge_from(FunctionFile_p func)
{
    // Merge required symbols;
    for (auto& sym : func->local_symbol)
        require(sym.second);
    // Merge source code;
    (*this) << "\n" << func->get_code();
    // Merge symbol name;
    if (get_symbol() == "")
        change_symbol(func->get_symbol());
    else
        change_symbol(get_symbol() + "_" + func->get_symbol());
    extern_declare = extern_declare + "\n" + func->extern_declare;
}

FunctionFile::FunctionFile(string extern_declare, LanguageUnit_p file_context)
{
    // Get requirement
    this->clean_require();
    for (auto& sym : file_context->local_symbol)
        this->require(sym.second);
    // Get source code
    (*this) << file_context->get_code();
    change_symbol(file_context->get_symbol());
    this->extern_declare = extern_declare;
}

FunctionFile_p FunctionFile::convert_from(FunctionUnit_p fu)
{
    LanguageUnit_p lu = make_shared<LanguageUnit>();
    LanguageUnit& def = *lu;
    def.require(header::assert);
    def.require(header::stdexcept);
    def.require(header::sstream);
    def.require(header::cuda);
    def.require(header::cublas);
    def.require(header::cudnn);
    def.require(macro::CUDA_SAFE_CALL);
    def.require(macro::CUDNN_SAFE_CALL);
    def.require(macro::CUBLAS_SAFE_CALL);
    def.require(declaration::typedef_int);

    for (auto& sym : fu->dep_unit->local_symbol)
        def.require(sym.second);

    // This for cudalib call or __global__ functions;
    def << fu->comment_unit->get_code();
    def << fu->get_specialized_signature() << "\n";
    def.block_begin();
    def << fu->body_unit->get_code() << "\n";
    def.block_end();

    // This is for function wrapper, nnfusion_rt.o will have symbol of this;
    std::string sig = fu->get_specialized_signature();
    int pos = sig.find(" __global__ "), next;
    if (pos >= 0)
    {
        while (pos < sig.size() && sig[pos] == ' ')
            ++pos;
        sig = sig.substr(pos + 12);
        pos = sig.find("(");
        if (pos >= 0)
        {
            std::string args = sig.substr(pos);
            assert(args.size() > 0 && args[args.size() - 1] == ')');
            args[args.size() - 1] = ',';

            sig.insert(pos, "_Call");
            sig.insert(pos + 6,
                       "const dim3 &grids, const dim3 &blocks, unsigned mem, "
                       "cudaStream_t stream, ");
            def << "\n" << sig << "\n{\n";
            def << "    return " << fu->name_unit->get_code()
                << "<<<grids, blocks, mem, stream>>>(";

            std::vector<std::string> params;
            for (pos = 0; next = args.find(',', pos), next >= 0; pos = next + 1)
            {
                int start = next - 1;
                while (start >= 0 &&
                       (isalpha(args[start]) || isdigit(args[start]) || args[start] == '_'))
                    --start;
                params.push_back(args.substr(start + 1, next - start - 1));
            }
            def << join(params, ", ") << ");\n";
            def << "}\n";
        }
    }

    LanguageUnit dec("dec");
    {
        std::string sig = fu->get_specialized_signature();
        int pos = sig.find(" __global__ "), next;
        if (pos >= 0)
        {
            while (pos < sig.size() && sig[pos] == ' ')
                ++pos;
            sig = sig.substr(pos + 12); // Remove __global__
            pos = sig.find("(");
            if (pos >= 0)
            {
                std::string args = sig.substr(pos);
                assert(args.size() > 0 && args[args.size() - 1] == ')');
                args[args.size() - 1] = ',';

                sig.insert(pos, "_Call");
                sig.insert(pos + 6,
                           "const dim3 &grids, const dim3 &blocks, unsigned mem, "
                           "cudaStream_t stream, ");
            }
        }
        dec << "\nextern " << sig << ";\n";
    }

    string sname = fu->name_unit->get_code();
    def.change_symbol(sname);
    return make_shared<FunctionFile>(dec.get_code(), lu);
}