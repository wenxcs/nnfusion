// Microsoft (c) 2019, Wenxiang
#include "ngraph/runtime/nnfusion/codegen/cuda/cuda_codegen.hpp"

using namespace std;
namespace ngraph
{
    namespace runtime
    {
        namespace nnfusion
        {
            bool CudaCodeGen::run(
                std::shared_ptr<ngraph::runtime::nnfusion::IntermediateOP>& inter_op)
            {
                static const std::map<
                    type_index,
                    function<std::shared_ptr<CodeGenOP>(std::shared_ptr<IntermediateOP>)>>
                    typeid_map{
                        {type_index(typeid(ngraph::op::Result)), codegen::cuda::Result::codegen},
                        {type_index(typeid(ngraph::op::Parameter)), codegen::cuda::Noop::codegen},
                        {type_index(typeid(ngraph::op::Relu)),
                         codegen::cuda::Elementwise<ngraph::op::Relu>::codegen},
                    };
                auto& node = *(inter_op->node);
                auto it = typeid_map.find(type_index(typeid(node)));
                if (it == typeid_map.end())
                {
                    NGRAPH_DEBUG << "Unsupported op '" << node.description() << "'" << endl;
                    return false;
                }
                NGRAPH_DEBUG << "Codegen op '" << node.description() << "'" << endl;
                auto cop = it->second(inter_op);
                assert_nullptr(cop);
                auto cw = cop->codegen_source();
                assert_nullptr(cw);
                //Replacing the inter_op with CodegenOP
                inter_op = cop;
                return true;
            }
        }
    }
}