// Microsoft (c) 2019, NNFusion Team

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "nnfusion/core/graph/gnode.hpp"
#include "op.hpp"

using namespace std;
using namespace nnfusion::op;

atomic<size_t> Op::m_next_instance_id(0);

Op::Op(const std::string& op_type)
    : m_op_type(op_type)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name(get_op_type() + "_" + to_string(m_instance_id))
{
}

// While we are still doing validation and type inference in the constructor, this is true
// It can be set to false to debug doing validation/inference after construction. When that
// is working, these two functions will be removed.
static bool in_transition = true;

void Op::constructor_validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (in_transition)
    {
        validate_and_infer_types(gnode);
    }
}

void Op::delayed_validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (!in_transition)
    {
        validate_and_infer_types(gnode);
    }
}

void Op::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
}

void Op::set_output_type_and_shape(std::shared_ptr<graph::GNode> gnode,
                                   size_t i,
                                   const ngraph::element::Type& element_type,
                                   const ngraph::PartialShape& pshape)
{
    gnode->get_outputs().at(i)->set_type_and_shape(element_type, pshape);
}

bool Op::is_parameter() const
{
    return false;
}

bool Op::is_output() const
{
    return false;
}

bool Op::is_constant() const
{
    return false;
}

const std::string& Op::get_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Op::get_unique_name() const
{
    return m_unique_name;
}

void Op::set_name(const string& name)
{
    CHECK(m_name.empty()) << "Op name may be set exactly once";
    m_name = name;
}

ngraph::Placement Op::get_placement() const
{
    return m_placement;
}

void Op::set_placement(ngraph::Placement placement)
{
    m_placement = placement;
}

std::tuple<ngraph::element::Type, ngraph::PartialShape>
    Op::validate_and_infer_elementwise_args(std::shared_ptr<graph::GNode> gnode)
{
    size_t input_size = gnode->get_input_size();
    auto element_type = gnode->get_input_element_type(0);
    auto pshape = gnode->get_input_partial_shape(0);

    if (input_size > 1)
    {
        for (size_t i = 1; i < input_size; ++i)
        {
            OP_VALIDATION(this,
                          ngraph::element::Type::merge(
                              element_type, element_type, gnode->get_input_element_type(i)))
                << "Argument element types are inconsistent.";

            OP_VALIDATION(
                this, ngraph::PartialShape::merge_into(pshape, gnode->get_input_partial_shape(i)))
                << "Argument shapes are inconsistent.";
        }
    }

    return std::make_tuple(element_type, pshape);
}

Op::~Op()
{
}

void Op::Clear()
{
    m_id = -1;
}

std::string nnfusion::op_validation_string(const Op* op)
{
    std::stringstream ss;
    ss << "While validating op '" << op->get_name() << "' of type '" << op->get_op_type() << "'";
    return ss.str();
}
