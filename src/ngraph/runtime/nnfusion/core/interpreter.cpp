// Microsoft (c) 2019, Wenxiang Hu
#include "interpreter.hpp"
#include "../op/alist.hpp"
#include "../pass/interpreter/extract_function_signature.hpp"
#include "../pass/interpreter/ngraph_function_pass.hpp"

FunctionTranslator::FunctionTranslator()
    : m_trans_ctx(new FunctionTranslatorContext())
    , m_passes(new vector<shared_ptr<IFunctionTranslatorPass>>())
{
}

FunctionTranslator::FunctionTranslator(
    shared_ptr<vector<shared_ptr<IFunctionTranslatorPass>>> passes,
    shared_ptr<FunctionTranslatorContext> ctx)
{
    this->m_passes = passes;
    this->m_trans_ctx = ctx;
}

shared_ptr<TranslationUnitMap> FunctionTranslator::translate(shared_ptr<ngraph::Function> function)
{
    static interpreter::NgraphFunctionPass ngraph_passes;
    static interpreter::ExtractFunctionSignature extract_global;
    shared_ptr<TranslationUnitMap> _tus(new TranslationUnitMap());
    assert_bool(ngraph_passes.run(m_trans_ctx, nullptr, function));
    // Iterator through all functions
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        auto current_function = p.first;
        _tus->emplace(p.first, _tu);
        NGRAPH_DEBUG << "Translating function:\t" << current_function->get_name() << endl;

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
            ir::Operator_p iop = nullptr;
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

ir::Operator_p FunctionTranslator::translate_node(shared_ptr<Node> node)
{
    const map<type_index, function<ir::Operator_p(shared_ptr<Node>)>> typeid_map{
        {type_index(typeid(ngraph::op::Parameter)), ir::Noop::translate},
        {type_index(typeid(ngraph::op::Result)), ir::Result::translate},
        {type_index(typeid(ngraph::op::Constant)), ir::Constant::translate},
        {type_index(typeid(ngraph::op::Broadcast)), ir::Broadcast::translate},
        {type_index(typeid(ngraph::op::Dot)), ir::Dot::translate},
        {type_index(typeid(ngraph::op::Relu)), ir::Elementwise<ngraph::op::Relu>::translate},
        {type_index(typeid(ngraph::op::Abs)), ir::Elementwise<ngraph::op::Abs>::translate},
        {type_index(typeid(ngraph::op::Add)), ir::Elementwise<ngraph::op::Add>::translate},
        {type_index(typeid(ngraph::op::Multiply)),
         ir::Elementwise<ngraph::op::Multiply>::translate},
        {type_index(typeid(ngraph::op::Subtract)),
         ir::Elementwise<ngraph::op::Subtract>::translate},
    };
    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        NGRAPH_DEBUG << "Unsupported op '" + node->description() + "', using Anyop instead."
                     << endl;
        return ir::Anyop::translate(node);
    }
    NGRAPH_DEBUG << "Translate op '" + node->description() + "'" << endl;
    return it->second(node);
}