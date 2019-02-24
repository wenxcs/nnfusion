#include "library.hpp"
#include "ngraph/runtime/nnfusion/cuda/cuda_langunit.hpp"

namespace nnfusion
{
    namespace library
    {
        std::string find_my_file()
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
            if (!file_exsits(filename))
                return nullptr;

            std::string objname = func_name + DLIB_SUFFIX;
            std::string my_directory = file_util::get_directory(find_my_file());
            std::string library_path = file_util::path_join(my_directory, objname);

            int ret =
                system(("nvcc\t--compiler-options\t'-fPIC\t-lcudnn'\t--shared\t-gencode\tarch="
                        "compute_60,code=sm_60\t-O3\t" +
                        filename + "\t-o\t" + library_path)
                           .c_str());
            if (!file_exsits(library_path))
                return nullptr;

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
            return fhdl;
        }

        void close_dhhandel(DL_HANDLE& handle) { CLOSE_LIBRARY(handle); }
        void dump_test_code(nnfusion::ir::Function_p func)
        {
            LanguageUnit lu(func->codegen_test_name() + ".cu");
            lu << "// Microsoft (c) 2019, Wenxiang\n";

            auto& re = func->dep_unit;
            using namespace nnfusion::cuda;
            re->require(header::assert);
            re->require(header::stdexcept);
            re->require(header::sstream);
            re->require(macro::CUDA_SAFE_CALL);
            re->require(declaration::typedef_int);

            // Write Dependency
            for (auto& it : re->local_symbol)
                if (it.second->symbol.find("header::") != string::npos)
                    lu << it.second->get_code();
            lu << "\n";
            for (auto& it : re->local_symbol)
                if (it.second->symbol.find("macro::") != string::npos)
                    lu << it.second->get_code();
            lu << "\n";
            for (auto& it : re->local_symbol)
                if (it.second->symbol.find("declaration::") != string::npos)
                    lu << it.second->get_code();
            lu << "\n";

            // Write function definition
            lu << func->definition_unit->get_code() << "\n";

            // Write Test Calls
            lu << func->test_unit->get_code() << "\n";

            // Save the file
            ofstream source_file(lu.symbol);
            source_file << lu.get_code();
            source_file.close();
        }

        string trim(string str)
        {
            str.erase(std::remove_if(str.begin(),
                                     str.end(),
                                     [](char c) -> bool {
                                         return std::isspace<char>(c, std::locale::classic());
                                     }),
                      str.end());
            return str;
        }
    }
}