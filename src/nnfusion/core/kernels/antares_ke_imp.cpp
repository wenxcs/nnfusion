#include "antares_ke_imp.hpp"
#include "nnfusion/util/curl_request.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_string(fantares_codegen_server);

std::unordered_map<std::string, std::string> AntaresKEImp::code_cache;

std::string AntaresKEImp::autogen(const std::string& expr)
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
        bool select = int(response.find("\n// CONFIG: {")) >= 0;
        printf("[Autogen] %s (select = %d)\n", expr.c_str(), select);
        if (!select)
            response = "";
        code_cache[expr] = response;
        return std::move(response);
    }
    else
        return it->second;
}