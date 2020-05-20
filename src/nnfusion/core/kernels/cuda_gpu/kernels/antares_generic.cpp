// Microsoft (c) 2019, NNFusion Team

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/util/curl_request.hpp"

DECLARE_string(fantares_codegen_server);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class AntaresCuda : public CudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

                std::string autogen(const std::string& expr)
                {
                    if (FLAGS_fantares_codegen_server == "")
                        return ""; // FLAGS_fantares_codegen_server = "10.150.145.98:8881";
                    static std::unordered_map<std::string, std::string> code_cache;
                    std::string response;
                    auto it = code_cache.find(expr);
                    if (it == code_cache.end())
                    {
                        CurlRequest req(FLAGS_fantares_codegen_server);
                        req.add_custom_header(("COMPUTE_V1: " + expr).c_str());

                        if (!req.send_request(response))
                            return "";
                        NNFUSION_CHECK(strncmp(response.c_str(), "[ERROR]", 7) != 0) << expr << "\n"
                                                                                     << response;
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

            public:
                AntaresCuda(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    auto& ctx = m_context;

                    auto ir = nnfusion::op::get_translation(ctx->gnode);
                    if (ir == "")
                        return nullptr;
                    auto str = autogen(ir);
                    if (str == "")
                        return nullptr;
                    int start = str.find(") {\n"), end = str.find("\n}\n");
                    NNFUSION_CHECK(start >= 0 && end >= 0);
                    str = str.substr(start + 4, end - start - 4);

                    int at_bx = str.find("// [thread_extent] blockIdx.x = "),
                        blockX = (at_bx >= 0)
                                     ? std::atoi(str.data() + at_bx +
                                                 sizeof("// [thread_extent] blockIdx.x = ") - 1)
                                     : 1;
                    int at_by = str.find("// [thread_extent] blockIdx.y = "),
                        blockY = (at_by >= 0)
                                     ? std::atoi(str.data() + at_by +
                                                 sizeof("// [thread_extent] blockIdx.y = ") - 1)
                                     : 1;
                    int at_bz = str.find("// [thread_extent] blockIdx.z = "),
                        blockZ = (at_bz >= 0)
                                     ? std::atoi(str.data() + at_bz +
                                                 sizeof("// [thread_extent] blockIdx.z = ") - 1)
                                     : 1;
                    int at_tx = str.find("// [thread_extent] threadIdx.x = "),
                        threadX = (at_tx >= 0)
                                      ? std::atoi(str.data() + at_tx +
                                                  sizeof("// [thread_extent] threadIdx.x = ") - 1)
                                      : 1;
                    int at_ty = str.find("// [thread_extent] threadIdx.y = "),
                        threadY = (at_ty >= 0)
                                      ? std::atoi(str.data() + at_ty +
                                                  sizeof("// [thread_extent] threadIdx.y = ") - 1)
                                      : 1;
                    int at_tz = str.find("// [thread_extent] threadIdx.z = "),
                        threadZ = (at_tz >= 0)
                                      ? std::atoi(str.data() + at_tz +
                                                  sizeof("// [thread_extent] threadIdx.z = ") - 1)
                                      : 1;

                    m_gridDim = dim3(blockX, blockY, blockZ);
                    m_blockDim = dim3(threadX, threadY, threadZ);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << str << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REG_ANTARES_KERNEL(OP_NAME)                                                                \
    REGISTER_KERNEL_EMITTER(                                                                       \
        #OP_NAME, Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("PRIORITY_9"), cuda::AntaresCuda)

#if 0
REG_ANTARES_KERNEL(Concat)
REG_ANTARES_KERNEL(Reshape)
REG_ANTARES_KERNEL(Dot)
REG_ANTARES_KERNEL(BatchMatMul)
REG_ANTARES_KERNEL(Convolution)
REG_ANTARES_KERNEL(Sum)
#endif
