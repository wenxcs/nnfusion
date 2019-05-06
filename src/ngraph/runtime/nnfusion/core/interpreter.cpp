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
    enforce(ngraph_passes.run(m_trans_ctx, nullptr, function));
    // Iterator through all functions
    for (const auto& p : m_trans_ctx->m_function_ordered_ops)
    {
        shared_ptr<TranslationUnit> _tu(new TranslationUnit());
        auto current_function = p.first;
        _tus->emplace(p.first, _tu);
        LOG_INFO << "Translating function:\t" << current_function->get_name() << endl;

        enforce(extract_global.run(m_trans_ctx, _tu, current_function))
            << "Error when extract global graph info.";

        enforce(IFunctionTranslatorPass::run_passes(
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
            enforce_not_nullptr(iop);
            _tu->inter_ops->push_back(iop);
        }
    }
    return _tus;
}

ir::Operator_p FunctionTranslator::translate_node(shared_ptr<Node> node)
{
    const map<type_index, function<ir::Operator_p(shared_ptr<Node>)>> typeid_map{
        {type_index(typeid(ngraph::op::Parameter)), ir::Noop::translate},
        {type_index(typeid(ngraph::op::AvgPool)), ir::AvgPool::translate},
        {type_index(typeid(ngraph::op::BatchNormInference)), ir::BatchNorm::translate},
        {type_index(typeid(ngraph::op::Result)), ir::Result::translate},
        {type_index(typeid(ngraph::op::Constant)), ir::Constant::translate},
        {type_index(typeid(ngraph::op::Concat)), ir::Concat::translate},
        {type_index(typeid(ngraph::op::Broadcast)), ir::Broadcast::translate},
        {type_index(typeid(ngraph::op::MaxPool)), ir::MaxPool::translate},
        {type_index(typeid(ngraph::op::Dot)), ir::Dot::translate},
        {type_index(typeid(ngraph::op::Pad)), ir::Pad::translate},
        {type_index(typeid(ngraph::op::Sum)), ir::Sum::translate},
        {type_index(typeid(ngraph::op::Slice)), ir::Slice::translate},
        {type_index(typeid(ngraph::op::Reshape)), ir::Reshape::translate},
        {type_index(typeid(ngraph::op::Convolution)), ir::Convolution::translate},
        {type_index(typeid(ngraph::op::Relu)), ir::Elementwise<ngraph::op::Relu>::translate},
        {type_index(typeid(ngraph::op::Abs)), ir::Elementwise<ngraph::op::Abs>::translate},
        {type_index(typeid(ngraph::op::Divide)), ir::Elementwise<ngraph::op::Divide>::translate},
        {type_index(typeid(ngraph::op::Negative)),
         ir::Elementwise<ngraph::op::Negative>::translate},
        {type_index(typeid(ngraph::op::Exp)), ir::Elementwise<ngraph::op::Exp>::translate},
        {type_index(typeid(ngraph::op::Add)), ir::Elementwise<ngraph::op::Add>::translate},
        {type_index(typeid(ngraph::op::Sigmoid)), ir::Elementwise<ngraph::op::Sigmoid>::translate},
        {type_index(typeid(ngraph::op::Power)), ir::Elementwise<ngraph::op::Power>::translate},
        {type_index(typeid(ngraph::op::Tanh)), ir::Elementwise<ngraph::op::Tanh>::translate},
        {type_index(typeid(ngraph::op::Multiply)),
         ir::Elementwise<ngraph::op::Multiply>::translate},
        {type_index(typeid(ngraph::op::Subtract)),
         ir::Elementwise<ngraph::op::Subtract>::translate},
        {type_index(typeid(ngraph::op::Softmax)), ir::Softmax::translate},
    };
    auto it = typeid_map.find(type_index(typeid(*node)));
    if (it == typeid_map.end())
    {
        LOG_WARN << "Unsupported op '" + node->description() + "', using Anyop instead." << endl;
        return ir::Anyop::translate(node);
    }
    LOG_INFO << "Translate op '" + node->description() + "'" << endl;
    return it->second(node);
}