// Microsoft (c) 2019, Wenxiang
#pragma once

// #include "ngraph/runtime/nnfusion/codegen/cuda/Reshape.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/Elementwise.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/Result.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_codegenerator.hpp"
#include "ngraph/runtime/nnfusion/nnfusion_common.hpp"

using namespace std;
namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            class CudaCodeGen : public ICodeGeneratorPass
            {
            public:
                bool run(std::shared_ptr<ngraph::runtime::nnfusion::IntermediateOP>& inter_op)
                {
                    static const std::map<
                        type_index,
                        function<std::shared_ptr<CodeGenOP>(std::shared_ptr<IntermediateOP>)>>
                        typeid_map{
                            {type_index(typeid(ngraph::op::Result)),
                             codegen::cuda::Result::codegen},
                            {type_index(typeid(ngraph::op::Relu)),
                             codegen::cuda::Elementwise<ngraph::op::Relu>::codegen},
                        };
                    auto& node = *(inter_op->node);
                    auto it = typeid_map.find(type_index(typeid(node)));
                    if (it == typeid_map.end())
                    {
                        NGRAPH_DEBUG << "Unsupported op '" << node.description() << "'" << endl;
                        return true;
                    }
                    NGRAPH_DEBUG << "Codegen op '" << node.description() << "'" << endl;
                    auto cop = it->second(inter_op);
                    assert_nullptr(cop);
                    auto cw = cop->codegen_source();
                    assert_nullptr(cw);
                    return true;
                }
            };
        }
    }
}