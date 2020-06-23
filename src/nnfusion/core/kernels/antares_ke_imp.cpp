#include "antares_ke_imp.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_string(fantares_codegen_server);

std::unordered_map<std::string, std::string> AntaresKEImp::code_cache;

std::string AntaresKEImp::autogen(const std::string& expr, bool antares_quick_codegen)
{
    if (FLAGS_fantares_codegen_server == "")
        return ""; // FLAGS_fantares_codegen_server = "10.150.145.98:8881";

    std::string response;
    auto it = code_cache.find(expr);
    if (it == code_cache.end())
    {
        CurlRequest req(FLAGS_fantares_codegen_server);
        req.add_custom_header(("COMPUTE_V1: " + expr).c_str());

        if (!req.send_request(response))
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Curl request Antares kernel failed.";
            return "";
        }
        if (strncmp(response.c_str(), "[ERROR]", 7) == 0)
        {
            NNFUSION_LOG(ERROR) << expr << "\n" << response;
            return "";
        }
        bool tuned = response.find("\n// Saved Perf =") != std::string::npos;
        bool choose = true;
        if (!antares_quick_codegen && !tuned)
        {
            response = "";
            choose = false;
        }

        NNFUSION_LOG(INFO) << "[Autogen] " << expr << " (tuned = " << tuned
                           << ", choose = " << choose << ")";
        code_cache[expr] = response;
        return std::move(response);
    }
    else
        return it->second;
}