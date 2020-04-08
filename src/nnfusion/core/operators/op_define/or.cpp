// Microsoft (c) 2019, NNFusion Team

#include "or.hpp"

using namespace std;
using namespace nnfusion::op;

Or::Or()
    : BinaryElementwiseLogical("Or")
{
}

ReduceAny::ReduceAny(const nnfusion::AxisSet& reduction_axes)
    : ArithmeticReduction("ReduceAny", reduction_axes)
{
}
