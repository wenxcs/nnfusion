// Microsoft (c) 2019, NNFusion Team

#include "generic_op.hpp"

namespace nnfusion
{
    namespace op
    {
        std::unordered_map<std::string, OpConfig>& get_op_configs()
        {
            static std::unordered_map<std::string, OpConfig> __op_configs;
            return __op_configs;
        }

        // empty string result when translation is not available for a certain op
        std::string get_translation(std::shared_ptr<nnfusion::graph::GNode>& gnode)
        {
            auto& configs = get_op_configs();
            auto it = configs.find(gnode->get_op_ptr()->get_op_type());
            if (it == configs.end() || it->second.f_translate == nullptr)
                return "";
            auto result = it->second.f_translate(gnode);
            return std::move(result);
        }
    } // namespace op
} // namespace nnfusion
