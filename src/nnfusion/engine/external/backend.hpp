// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "nnfusion/common/common.hpp"

using Backend = ngraph::runtime::Backend;

namespace nnfusion
{
    // This is an abstract class for NNFusion Backend
    class nnfusion_Backend : public Backend
    {
    public:
        nnfusion_Backend()
            : Backend(){};
        virtual bool codegen(std::shared_ptr<Function> func) = 0;
        virtual bool codegen(std::shared_ptr<graph::Graph> graph) = 0;
    };
}
