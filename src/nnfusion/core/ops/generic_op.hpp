// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/common/common.hpp"

#define REGISTER_OP(op_x) static ngraph::op::OpConfig __register_op_##op_x = ngraph::op::build_op_config(#op_x)
#define GENERIC_OP_LOGGING()  std::cout << "[GENERIC_OP_LOGGING] " << __FILE__ << ": " << __PRETTY_FUNCTION__ << std::endl;


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

		std::unordered_map<std::string, ngraph::op::OpConfig> &get_op_configs();

        inline const OpConfig& lookup_op_config(const std::string &opname) {
			GENERIC_OP_LOGGING();

            auto it = get_op_configs().find(opname);
            if (it == get_op_configs().end())
                throw std::runtime_error(
                    (std::string("No config-definition found for op type `") + opname + "`")
                        .c_str());
            return it->second;
        }

        inline OpConfig& build_op_config(const std::string &opname) {
			GENERIC_OP_LOGGING();

            if (get_op_configs().find(opname) != get_op_configs().end())
                throw std::runtime_error((std::string("OpConfig for opname `") + opname + "` is registered more than once.").c_str());
            std::cout << "Registering opname `" << opname << "`;\n";
            return get_op_configs()[opname];
        }

		inline std::string create_code_from_template(std::string templ, const ngraph::op::OpConfig::any &feed_dict) {
			for (auto &it: feed_dict.items()) {
				std::string placeholder = "@" + it.key() + "@";
				int at = 0;
				while (true) {
					at = templ.find(placeholder, at);
					if (at < 0)
						break;
					std::string value;
					if (it.value().is_string())
						value = it.value();
					else if (it.value().is_null())
						value = "NULL";
					else {
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
                      const std::vector<std::shared_ptr<Node>>& inputs,
                      const OpConfig::any& customOpConfig)
                : Op(opname, check_single_output_args(inputs))
                , name(name)
                , opname(opname)
                , inputs(inputs)
            {
				GENERIC_OP_LOGGING();

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
				validate_and_infer_types();
            }

			virtual std::shared_ptr<ngraph::Node> copy_with_new_args(const NodeVector& new_args) const override
			{
				throw std::runtime_error("Not expected to reach here: copy_with_new_args().");
				return std::make_shared<GenericOp>(name, opname, new_args, localOpConfig.getRoot());
			}

            virtual void validate_and_infer_types() override
            {
				GENERIC_OP_LOGGING();
                localOpConfig.check_constrait();
                localOpConfig.f_infershape(*this);
            }

            mutable OpConfig localOpConfig;
            std::string name, opname;
            std::vector<std::shared_ptr<Node>> inputs;
        };

        namespace infershape
        {
            // Provide default infershape function: output_shapes[*] = input_shapes[*];
            inline void copy_shape_from_inputs(GenericOp& target_op)
            {
                for (int i = 0; i < target_op.get_input_size(); ++i)
                    target_op.set_output_type(
                        0, target_op.get_input_element_type(i), target_op.get_input_shape(i));
            }
        }
    }
}
