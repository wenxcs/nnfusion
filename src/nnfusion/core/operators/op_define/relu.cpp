// Microsoft (c) 2019, NNFusion Team

#include "relu.hpp"

using namespace std;
using namespace nnfusion::op;

Relu::Relu()
    : ElementwiseArithmetic("Relu")
{
}

ReluBackprop::ReluBackprop()
    : ElementwiseArithmetic("ReluBackprop")
{
}

Relu6::Relu6()
    : ElementwiseArithmetic("Relu6")
{
}

Relu6Backprop::Relu6Backprop()
    : ElementwiseArithmetic("Relu6Backprop")
{
}