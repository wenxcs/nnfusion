// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_functiontranslator.hpp"
#include "ngraph/runtime/nnfusion/intermediate/op_tbl.hpp"
#include "ngraph/runtime/nnfusion/pass/translator/extract_function_signature.hpp"
#include "ngraph/runtime/nnfusion/pass/translator/ngraph_function_pass.hpp"

using namespace ngraph::runtime::nnfusion;

ngraph::runtime::nnfusion::FunctionTranslator::FunctionTranslator()
    : m_trans_ctx(new FunctionTranslatorContext())
    , m_passes(new vector<std::shared_ptr<IFunctionTranslatorPass>>())
{
}

ngraph::runtime::nnfusion::FunctionTranslator::FunctionTranslator(
    shared_ptr<vector<shared_ptr<IFunctionTranslatorPass>>> passes,
    shared_ptr<FunctionTranslatorContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

std::shared_ptr<TranslationUnitMap> ngraph::runtime::nnfusion::FunctionTranslator::translate(
    std::shared_ptr<ngraph::Function> function)
{
    static translator::NgraphFunctionPass ngraph_passes;
    static translator::ExtractFunctionSignature extract_global;
    std::shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());
    assert_bool(ngraph_passes.run(m_trans_ctx, nullptr, function));

    // Iterator through all functions
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        std::shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        auto current_function = p.first;
        _tus->emplace(p.first, _tu);
        NGRAPH_DEBUG << "Translating function:\t" << current_function->get_name() << std::endl;

        assert_bool(extract_global.run(m_trans_ctx, _tu, current_function))
            << "Error when extract global graph info.";
        assert_bool(IFunctionTranslatorPass::run_passes(
            *(this->m_passes), m_trans_ctx, _tu, current_function))
            << "Error when apply passes on functions.";

        // Translate the Node
        for (shared_ptr<Node> node : m_trans_ctx->m_function_ordered_ops.at(current_function))
        {
            // Generate Translated OP
            // <todo> not sure translated
            auto it = m_trans_ctx->m_node_inter_map.find(node);
            shared_ptr<IntermediateOP> iop = nullptr;
            if (it == m_trans_ctx->m_node_inter_map.end())
            {
                iop = this->translate_node(node);
                m_trans_ctx->m_node_inter_map.emplace(node, iop);
            }
            else
            {
                iop = m_trans_ctx->m_node_inter_map[node];
            }
            assert_nullptr(iop);
            _tu->inter_ops->push_back(iop);
        }
    }
    return _tus;
}

shared_ptr<IntermediateOP>
    ngraph::runtime::nnfusion::FunctionTranslator::translate_node(shared_ptr<Node> node)
{
    static const map<type_index, function<std::shared_ptr<IntermediateOP>(shared_ptr<Node>)>>
        typeid_map{
            {type_index(typeid(ngraph::op::Parameter)), intermediate::Noop::translate},
            {type_index(typeid(ngraph::op::Result)), intermediate::Result::translate},
            {type_index(typeid(ngraph::op::Relu)),
             intermediate::Elementwise<ngraph::op::Relu>::translate},
        };

    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        NGRAPH_DEBUG << "Unsupported op '" + node->description() + "'" << endl;
        return nullptr;
    }
    NGRAPH_DEBUG << "Translate op '" + node->description() + "'" << endl;
    return it->second(node);
}