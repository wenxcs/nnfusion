// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "graph_pass_base.hpp"

namespace nnfusion
{
    namespace graph
    {
        namespace pass
        {
            class OpInplacePass : public GraphPassBase
            {
            public:
                bool run_on_graph(std::shared_ptr<Graph>& graph) override;

            private:
                bool shared_in_nodes(std::shared_ptr<GNode>& node);

                template <class T>
                void AddInplace(T op, size_t output, size_t input)
                {
                    auto op_annotations = op->get_op_annotations();
                    if (op_annotations)
                    {
                        // pass-through
                        op_annotations->add_in_place_oi_pair({output, input, false});
                    }
                    else
                    {
                        op_annotations = std::make_shared<ngraph::op::util::OpAnnotations>();
                        // pass-through
                        op_annotations->add_in_place_oi_pair({output, input, false});
                        op->set_op_annotations(op_annotations);
                    }
                }
            };
        }
    }
}
