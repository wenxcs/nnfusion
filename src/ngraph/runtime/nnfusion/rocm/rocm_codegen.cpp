// Microsoft (c) 2019, Wenxiang
#include "rocm_codegen.hpp"
#include "rocm_langunit.hpp"

#include <bits/stdc++.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace nnfusion
{
    namespace rocm
    {
        bool save_file(LanguageUnit_p lu, string file)
        {
            std::ofstream out(file);
            out << lu->get_code();
            out.close();
            return true;
        }

        bool save_file(string str, string file)
        {
            std::ofstream out(file);
            out << str;
            out.close();
            return true;
        }

        bool save_file(LanguageUnit_p lu)
        {
            std::ofstream out(lu->symbol);
            out << lu->get_code();
            out.close();
            return true;
        }

        bool folder_exists(string folder)
        {
            struct stat sb;
            return (stat(folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
        }

        bool file_exists(string folder)
        {
            struct stat sb;
            return stat(folder.c_str(), &sb) == 0;
        }

        bool create_dir(std::string tar_path)
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
                    printf("Error creating directory: %s", (tar_path).c_str());
                    flag = false;
                }
                else
                    flag = true;
            }
            else
            {
                LOG(INFO) << "Directory " << tar_path.c_str() << " already exists";
                flag = true;
            }
            return flag;
        }

        std::string generate_cmakelists(void)
        {
            LanguageUnit lu;
            lu << R"(project(main_test)
cmake_minimum_required(VERSION 3.5)

set(ENV{HIP_PLATFORM} hcc)
link_directories(/opt/rocm/lib)

SET(CMAKE_C_COMPILER /opt/rocm/bin/hipcc)
SET(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

add_compile_options(-std=c++11 -O2)

add_library(nnfusion_naive_rt nnfusion_rt.cc)
target_include_directories(nnfusion_naive_rt
    SYSTEM PUBLIC
    MIOpen hipblas
)

target_link_libraries(nnfusion_naive_rt
)

add_executable(main_test main_test.cpp)
target_link_libraries(main_test nnfusion_naive_rt MIOpen hipblas))";
            return lu.get_code();
        }

        std::string generate_rt_h(LanguageUnit_p rth)
        {
            string mainStr = rth->get_code();
            string toErase = cuda::header::cuda->get_code();
            size_t pos = mainStr.find(toErase);

            if (pos != std::string::npos)
            {
                // If found then erase it from string
                mainStr.erase(pos, toErase.length());
            }
            return mainStr + "\n" + rocm::header::nnfusion_hip->get_code();
        }

        bool ROCM_NaiveCudaCodeGenerator::projgen()
        {
            this->lu_cmakefile = LanguageUnit_p(new LanguageUnit("CMakeLists.txt"));
            (*this->lu_cmakefile) << generate_cmakelists();

            this->lu_header->symbol = "nnfusion_rt.t.h";
            this->lu_main->symbol = "main_test.t.cpp";

            save_file(this->lu_cmakefile);
            save_file(this->lu_nnfusion_rt);
            save_file(generate_rt_h(this->lu_header), this->lu_header->symbol);
            save_file(this->lu_main);

            if (!file_exists("nnfusion_hip.h"))
                save_file(rocm::file::cublas_v2_h, "nnfusion_hip.h");
        }

        bool ROCM_NaiveCudaCodeGenerator::setpwd()
        {
            std::string tar_path = "./rocm_codegen/";
            create_dir(tar_path);
            chdir(tar_path.c_str());
        }

        bool ROCM_NaiveCudaCodeGenerator::codegen(shared_ptr<TranslationUnit> tu)
        {
            string rocm_folder = "/opt/rocm/";
            string rocc_folder = "/opt/rocm/rocc";
            string cuda_file = "nnfusion_rt.cu";
            string rocm_file = "nnfusion_rt.cc";
            string hipify_rocc = "/opt/rocm/rocc/hipify-rocc";
            string rocc_file = "/opt/rocm/rocc/rocc";
            string nnfusion_hip = "nnfusion_hip.h";

            if (!folder_exists(rocm_folder))
                throw std::runtime_error("ROCM not installed in /opt/rocm;");

            { // Write down the rocc
                if (!folder_exists(rocc_folder))
                {
                    mkdir(rocc_folder.c_str(), 0774);

                    if (!file_exists(hipify_rocc))
                        save_file(rocm::file::hipify_rocc, hipify_rocc);

                    if (!file_exists(rocc_file))
                        save_file(rocm::file::rocc, rocc_file);

                    if (folder_exists(rocc_folder))
                    {
                        std::cerr << "Installed the files into /opt/rocm/rocc, pleas re-run your "
                                     "command to codegen."
                                  << std::endl;

                        system("sudo chmod -R 777 /opt/rocm/rocc");
                        return true;
                    }
                }

                if (!folder_exists(rocc_folder) || !file_exists(hipify_rocc) ||
                    !file_exists(rocc_file))
                    throw std::runtime_error(
                        "File not ready for this backend. Please run your command with sudo.");
            }

            bool st = NaiveCudaCodeGenerator::codegen(tu);
            if (!st)
                return false;

            if (0 !=
                system(("/opt/rocm/rocc/hipify-rocc " + cuda_file + " > " + rocm_file).c_str()))
                throw std::runtime_error("Failed coverting cuda file;\n");

            if (0 != system(("/opt/rocm/rocc/hipify-rocc nnfusion_rt.t.h > nnfusion_rt.h")))
                throw std::runtime_error("Failed coverting cuda file;\n");

            if (0 != system(("/opt/rocm/rocc/hipify-rocc main_test.t.cpp > main_test.cpp")))
                throw std::runtime_error("Failed coverting cuda file;\n");

            remove(cuda_file.c_str());
            remove("main_test.t.cpp");
            remove("nnfusion_rt.t.h");

            ///\todo(wenxh) Generate CMakeFileList Here
            return true;
        }
    }
}