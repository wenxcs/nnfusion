// Microsoft (c) 2019, NNFusion Team
#pragma once

#include <iomanip>
#include <limits>
#include "nnfusion/common/common.hpp"

#define REGISTER_OP(op_x)                                                                          \
    static nnfusion::op::OpConfig __register_op_##op_x = nnfusion::op::build_op_config(#op_x)
#define GENERIC_OP_LOGGING()                                                                       \
    NNFUSION_LOG(DEBUG) << "[GENERIC_OP_LOGGING] " << __FILE__ << ": " << __PRETTY_FUNCTION__;

namespace nnfusion
{
    namespace op
    {
        class OpConfig;
        class GenericOp;

        class OpConfig
        {
        public:
            using any = nlohmann::json;
            using constrait_func_t = bool (*)(const OpConfig::any& config);
            using infershape_func_t = void (*)(std::shared_ptr<graph::GNode> gnode);
            using translate_func_t = std::string (*)(std::shared_ptr<graph::GNode> gnode);

            // OpConfig(): f_infershape(infershape::copy_shape_from_inputs) { }

            template <typename T>
            OpConfig& attr(const std::string& name, const T& val = T())
            {
                getRoot()[name] = val;
                return *this;
            }

            OpConfig& check_constrait()
            {
                NNFUSION_CHECK(is_legal()) << "OpConfig::check_constrait() not passed!";
                return *this;
            }

            OpConfig& constrait(const constrait_func_t& func)
            {
                f_constraits.push_back(func);
                return *this;
            }

            OpConfig& infershape(const infershape_func_t& func)
            {
                f_infershape = func;
                return *this;
            }

            OpConfig& translate(const translate_func_t& func)
            {
                f_translate = func;
                return *this;
            }

            OpConfig& show()
            {
                NNFUSION_LOG(INFO) << getRoot();
                return *this;
            }

            bool is_legal()
            {
                if (!f_infershape)
                    return false;
                for (auto& func : f_constraits)
                    if (!func(getRoot()))
                        return false;
                return true;
            }

            OpConfig::any& getRoot() { return this->j_attrs["config"]; }
            OpConfig::any& get(std::string key) { return getRoot()[key]; }
            std::vector<constrait_func_t> f_constraits;
            infershape_func_t f_infershape;
            translate_func_t f_translate;
            OpConfig::any j_attrs;
        };

        std::unordered_map<std::string, OpConfig>& get_op_configs();
        std::string get_translation(std::shared_ptr<nnfusion::graph::GNode>& gnode);

        inline const OpConfig& lookup_op_config(const std::string& opname)
        {
            auto it = get_op_configs().find(opname);
            NNFUSION_CHECK(it != get_op_configs().end())
                << "No config-definition found for op type `" + opname + "`";
            return it->second;
        }

        inline OpConfig& build_op_config(const std::string& opname)
        {
            NNFUSION_CHECK(get_op_configs().find(opname) == get_op_configs().end())
                << "OpConfig for opname `" + opname + "` is registered more than once.";
            NNFUSION_LOG(INFO) << "Registering opname `" << opname << "`";
            return get_op_configs()[opname];
        }

        template <typename T>
        std::string expand_vector(string name, vector<T>& d, std::string typestring)
        {
            stringstream ss;
            for (int i = 0; i < d.size(); i++)
                ss << typestring << " " << name << i << " = " << to_string(d[i]) << ";\n";
            return ss.str();
        }

        inline std::string create_code_from_template(std::string templ,
                                                     const OpConfig::any& feed_dict)
        {
            for (auto& it : feed_dict.items())
            {
                std::string placeholder = "@" + it.key() + "@";
                int at = 0;
                while (true)
                {
                    at = templ.find(placeholder, at);
                    if (at < 0)
                        break;
                    std::string value;
                    if (it.value().is_string())
                        value = it.value();
                    else if (it.value().is_null())
                        value = "NULL";
                    else if (it.value().is_number_float())
                    {
                        std::stringstream ss;
                        ss.flags(std::ios_base::scientific);
                        ss << std::setprecision(std::numeric_limits<double>::digits)
                           << (double)it.value();
                        value = ss.str();
                    }
                    else
                    {
                        std::stringstream ss;
                        ss << it.value();
                        value = ss.str();
                    }
                    templ = templ.substr(0, at) + value + templ.substr(at + placeholder.size());
                    at += value.size();
                }
            }
            return std::move(templ);
        };

        class GenericOp : public Op
        {
        public:
            GenericOp(const std::string& name,
                      const std::string& opname,
                      const OpConfig::any& customOpConfig)
                : Op(opname)
            {
                // Merge customOpConfig into default config
                localOpConfig = lookup_op_config(opname);
                std::unordered_set<std::string> keyset;
                for (auto& item : localOpConfig.getRoot().items())
                    keyset.insert(item.key());

                for (auto& item : customOpConfig.items())
                {
                    NNFUSION_CHECK(keyset.find(item.key()) != keyset.end())
                        << "Invalid attribution `" + item.key() + "` not recognized by op type `" +
                               opname + "`";
                    localOpConfig.getRoot()[item.key()] = item.value();
                }

                set_name(name);
                NNFUSION_LOG(INFO) << "Managing GenericOp for Opeartor: type = " << opname
                                   << ", name = " << name;

                localOpConfig.check_constrait();
            }

            virtual void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override
            {
                localOpConfig.check_constrait();
                localOpConfig.f_infershape(gnode);

                if (localOpConfig.f_translate != nullptr && !m_expression.size())
                {
                    m_expression = localOpConfig.f_translate(gnode);
                }
            }

            mutable OpConfig localOpConfig;
            std::string m_expression;
        };

        namespace infershape
        {
            // Provide default infershape function: output_shapes[*] = input_shapes[*];
            inline void copy_shape_from_inputs(std::shared_ptr<graph::GNode> gnode)
            {
                for (int i = 0; i < gnode->get_input_size(); ++i)
                {
                    gnode->set_output_type_and_shape(
                        0, gnode->get_input_element_type(i), gnode->get_input_shape(i));
                }
            }

            // unimplemented that will notify exception when migrating to op_v2 mode
            inline void unimplemented_and_not_used(std::shared_ptr<graph::GNode> gnode)
            {
                throw std::runtime_error(
                    ("Not implemented infershape for Op: " + gnode->get_op_ptr()->get_op_type())
                        .c_str());
            }
        } // namespace infershape
    }     // namespace op
} // namespace nnfusion