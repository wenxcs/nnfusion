// Microsoft (c) 2019, NNFusion Team

//*****************************************************************************

#include <sstream>

#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/op_define/reshape.hpp"
#include "nnfusion/util/util.hpp"
#include "numpy_transpose.hpp"

#include "ngraph/util.hpp"

namespace nnfusion
{
    namespace graph
    {
        std::string numpy_transpose_error_str(const ngraph::AxisVector& order,
                                              const ngraph::Shape& in_shape)
        {
            std::ostringstream os;
            os << "The axes order ";
            os << "[ " << ngraph::join(order) << " ]";
            os << " is incompatible with the input shape ";
            os << "[ " << ngraph::join(in_shape) << " ]";
            os << " during numpy_transpose.";
            return os.str();
        }

        std::shared_ptr<GNode> numpy_transpose(const std::shared_ptr<GNode>& gnode,
                                               ngraph::AxisVector order)
        {
            auto in_shape = gnode->get_shape();
            // default, reverse the order of the axes
            if (order.size() == 0)
            {
                auto n = in_shape.size();
                order = ngraph::AxisVector(n);
                std::generate(order.begin(), order.end(), [&n]() { return --n; });
            }
            else if (order.size() == in_shape.size())
            {
                // validate that the axes order is valid, i.e., unique and the right size
                std::unordered_set<ngraph::AxisVector::value_type> axes;
                for (auto o : order)
                {
                    if (o < in_shape.size() && !axes.count(o))
                    {
                        axes.insert(o);
                    }
                    else
                    {
                        CHECK_FAIL() << numpy_transpose_error_str(order, in_shape);
                    }
                }
            }
            else
            {
                CHECK_FAIL() << numpy_transpose_error_str(order, in_shape);
            }

            // create output shape
            ngraph::Shape out_shape;
            for (size_t i = 0; i < in_shape.size(); ++i)
                out_shape.push_back(in_shape[order[i]]);

            // do the reshaping with the order
            auto reshape_op = std::make_shared<op::Reshape>(order, out_shape);
            auto reshape_gnode = std::make_shared<GNode>(reshape_op, GNodeVector({gnode}));
            reshape_op->revalidate_and_infer_types(reshape_gnode->shared_from_this());
            return reshape_gnode;
        }

    } // namespace builder
} // namespace ngraph
