// Microsoft (c) 2019, NNFusion Team

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        class Fused : public Op
        {
        public:
            Fused(const std::string& name, const std::string& opname)
                : Op(opname){};

            void register_ir2(std::vector<std::shared_ptr<graph::GNode>>& gnodes);
            std::string get_fused_ir2() { return fused_op_ir2; };
            std::string get_plan_rule() { return plan_rule; };
        protected:
            std::string fused_op_ir2;
            std::string plan_rule;
        };
    }
}
