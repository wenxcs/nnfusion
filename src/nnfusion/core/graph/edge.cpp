// Microsoft (c) 2019, NNFusion Team

#include <sstream>

#include "edge.hpp"

using namespace nnfusion::graph;

std::string Edge::DebugString() const
{
    std::stringstream ss;
    ss << "[id=" << m_id << " " << m_src->get_name().c_str() << ":" << m_src_output << " -> "
       << m_dst->get_name().c_str() << ":" << m_dst_input << "]";
    return ss.str();
}