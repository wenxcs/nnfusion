// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/common/common.hpp"

#define REGISTER_OP(op_x) static ngraph::op::OpConfig __register_op_##op_x = ngraph::op::build_op_config(#op_x)


namespace ngraph
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
            using infershape_func_t = void (*)(GenericOp& target_op);

            // OpConfig(): f_infershape(infershape::copy_shape_from_inputs) { }

            template <typename T>
            OpConfig& attr(const std::string& name, const T& val = T())
            {
                getRoot()[name] = val;
                return *this;
            }

            OpConfig& check_constrait()
            {
                if (!is_legal())
                    throw std::runtime_error("OpConfig::check_constrait() not passed!");
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

            OpConfig& show()
            {
                std::cout << getRoot() << std::endl;
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
            OpConfig::any j_attrs;
        };

        extern std::unordered_map<std::string, OpConfig> __op_configs;

        inline OpConfig& build_op_config(const std::string &opname) {
            if (__op_configs.count(opname) > 0)
                throw std::runtime_error((std::string("OpConfig for opname `") + opname + "` is registered more than once.").c_str());
            std::cout << "Registering opname `" << opname << "`;\n";
            return __op_configs[opname];
        }

        inline const OpConfig& lookup_op_config(const std::string &opname) {
            auto it = __op_configs.find(opname);
            if (it == __op_configs.end())
                throw std::runtime_error(
                    (std::string("No config-definition found for op type `") + opname + "`")
                        .c_str());
            return it->second;
        }

        class GenericOp : public Op
        {
        public:
            GenericOp(const std::string& name,
                      const std::string& opname,
                      const std::vector<std::shared_ptr<Node>>& inputs,
                      const OpConfig::any& customOpConfig)
                : Op("GenericOp", check_single_output_args(inputs))
                , name(name)
                , opname(opname)
                , inputs(inputs)
            {
                std::cout << "Constructing new op `" << opname << "` with name `" << name
                          << "`, input size = " << inputs.size() << ";\n";

                // Merge customOpConfig into default config
                localOpConfig = lookup_op_config(opname);
                std::unordered_set<std::string> keyset;
                for (auto& item : localOpConfig.getRoot().items())
                    keyset.insert(item.key());

                for (auto& item : customOpConfig.items())
                {
                    if (keyset.find(item.key()) != keyset.end())
                        localOpConfig.getRoot()[item.key()] = item.value();
                    else
                        throw std::runtime_error((std::string("Invalid attribution `") +
                                                  item.key() + "` not recognized by op type `" +
                                                  opname + "`")
                                                     .c_str());
                }

                localOpConfig.check_constrait();
            }

            virtual void validate_and_infer_types() override
            {
                localOpConfig.check_constrait();
                localOpConfig.f_infershape(*this);
            }

            OpConfig localOpConfig;
            std::string name, opname;
            std::vector<std::shared_ptr<Node>> inputs;
        };

        namespace infershape
        {
            // Provide default infershape function: output_shapes[*] = input_shapes[*];
            void copy_shape_from_inputs(GenericOp& target_op)
            {
                for (int i = 0; i < target_op.get_input_size(); ++i)
                    target_op.set_output_type(
                        0, target_op.get_input_element_type(i), target_op.get_input_shape(i));
            }
        }
    }
}
