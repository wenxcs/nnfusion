// Microsoft (c) 2019, Wenxiang
#pragma once

#include "ngraph/runtime/nnfusion/codegen/cuda/Reshape.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/Result.hpp"
#include "ngraph/runtime/nnfusion/codegen/cuda/elementwise.hpp"
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
                             ngraph::runtime::nnfusion::codegen::cuda::elementwise<
                                 ngraph::op::Relu>::codegen},
                        };
                    auto& node = *(inter_op->n);
                    auto it = typeid_map.find(type_index(typeid(node)));
                    if (it == typeid_map.end())
                    {
                        cout << "Unsupported op '"<<node.description()
                             << "'" << endl;
                        return false;
                    }
                    cout << "Codegen op '" <<node.description()
                         << "'" << endl;
                    it->second(inter_op);
                    return true;
                }
            };
        }
    }
}