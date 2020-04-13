//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "op/batch_norm.hpp"
#include "op/binaryop.hpp"
#include "op/cast.hpp"
#include "op/clip.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/index_reduce.hpp"
#include "op/pool.hpp"
#include "op/unaryop.hpp"

#include "ops_bridge.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace detail
            {
                const ConvertFunc& find(const std::string& name,
                                        std::int64_t version,
                                        const std::string& domain,
                                        const std::map<std::int64_t, ConvertFunc>& map)
                {
                    while (version > 0)
                    {
                        const auto it = map.find(version--);
                        if (it != std::end(map))
                        {
                            return it->second;
                        }
                    }
                    NNFUSION_CHECK_FAIL()
                        << "Unsupported version: " << (domain.empty() ? "" : domain + ".") << name
                        << ":" << std::to_string(version);
                }
            }

            void OperatorsBridge::_register_operator(const std::string& name,
                                                     std::int64_t version,
                                                     const std::string& domain,
                                                     ConvertFunc fn)
            {
                m_map[domain][name].emplace(version, std::move(fn));
            }

            ConvertFuncMap OperatorsBridge::_get_convert_func_map(std::int64_t version,
                                                                  const std::string& domain)
            {
                ConvertFuncMap result;
                auto dm = m_map.find(domain);
                NNFUSION_CHECK(dm != std::end(m_map)) << "Unknown Domain: " << domain;

                for (const auto& op : dm->second)
                {
                    result.emplace(op.first, detail::find(op.first, version, domain, op.second));
                }
                return result;
            }

#define REGISTER_OPERATOR(name_, ver_, fn_)                                                        \
    m_map[""][name_].emplace(                                                                      \
        ver_,                                                                                      \
        std::bind(                                                                                 \
            set_##ver_::fn_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3))

            OperatorsBridge::OperatorsBridge()
            {
                REGISTER_OPERATOR("Abs", 1, TranslateUnaryOp<op::Abs>);
                REGISTER_OPERATOR("Acos", 1, TranslateUnaryOp<op::Acos>);
                REGISTER_OPERATOR("Add", 1, TranslateLegacyBinaryOp<op::Add>);
                REGISTER_OPERATOR("Add", 7, TranslateBinaryOp<op::Add>);
                REGISTER_OPERATOR("And", 1, TranslateBinaryOp<op::And>);
                REGISTER_OPERATOR("ArgMin", 1, TranslateIndexReductionOp<op::ArgMin>);
                REGISTER_OPERATOR("ArgMax", 1, TranslateIndexReductionOp<op::ArgMax>);
                REGISTER_OPERATOR("Asin", 1, TranslateUnaryOp<op::Asin>);
                REGISTER_OPERATOR("Atan", 1, TranslateUnaryOp<op::Atan>);
                REGISTER_OPERATOR("AveragePool", 1, TranslatePoolOp<op::AvgPool>);
                REGISTER_OPERATOR("BatchNormalization", 1, TranslateBatchNormOp);
                REGISTER_OPERATOR("Cast", 1, TranslateCastOp);
                REGISTER_OPERATOR("Ceil", 1, TranslateUnaryOp<op::Ceiling>);
                REGISTER_OPERATOR("Clip", 1, TranslateClipOp);
                REGISTER_OPERATOR("Concat", 1, TranslateConcatOp);
                REGISTER_OPERATOR("Constant", 1, TranslateConstantOp);
                //REGISTER_OPERATOR("Conv", 1, conv);
                REGISTER_OPERATOR("Cos", 1, TranslateUnaryOp<op::Cos>);
                REGISTER_OPERATOR("Div", 1, TranslateLegacyBinaryOp<op::Divide>);
                REGISTER_OPERATOR("Div", 7, TranslateBinaryOp<op::Divide>);
                //REGISTER_OPERATOR("Dropout", 1, identity);
                //REGISTER_OPERATOR("Elu", 1, elu);
                REGISTER_OPERATOR("Equal", 1, TranslateBinaryOp<op::Equal>);
                REGISTER_OPERATOR("Exp", 1, TranslateUnaryOp<op::Exp>);
                //REGISTER_OPERATOR("Flatten", 1, flatten);
                REGISTER_OPERATOR("Floor", 1, TranslateUnaryOp<op::Floor>);
                //REGISTER_OPERATOR("Gemm", 1, gemm);
                REGISTER_OPERATOR("GlobalAveragePool", 1, TranslatePoolOp<op::AvgPool>);
                REGISTER_OPERATOR("GlobalMaxPool", 1, TranslatePoolOp<op::MaxPool>);
                REGISTER_OPERATOR("Greater", 1, TranslateBinaryOp<op::Greater>);
                //REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
                //REGISTER_OPERATOR("Identity", 1, identity);
                //REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
                REGISTER_OPERATOR("Less", 1, TranslateBinaryOp<op::Less>);
                REGISTER_OPERATOR("Log", 1, TranslateUnaryOp<op::Log>);
                //REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
                //REGISTER_OPERATOR("LRN", 1, lrn);
                //REGISTER_OPERATOR("MatMul", 1, matmul);
                REGISTER_OPERATOR("MaxPool", 1, TranslatePoolOp<op::MaxPool>);
                //REGISTER_OPERATOR("Max", 1, max);
                //REGISTER_OPERATOR("Mean", 1, mean);
                //REGISTER_OPERATOR("Min", 1, min);
                REGISTER_OPERATOR("Mul", 1, TranslateLegacyBinaryOp<op::Multiply>);
                REGISTER_OPERATOR("Mul", 7, TranslateBinaryOp<op::Multiply>);
                //REGISTER_OPERATOR("Neg", 1, neg);
                //REGISTER_OPERATOR("Not", 1, TranslateUnaryOp<op::Not>);
                REGISTER_OPERATOR("Or", 1, TranslateBinaryOp<op::Or>);
                REGISTER_OPERATOR("Pow", 1, TranslateBinaryOp<op::Power>);
                //REGISTER_OPERATOR("PRelu", 1, prelu);
                //REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
                //REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
                //REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
                //REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
                //REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
                //REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
                //REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
                //REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
                //REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
                //REGISTER_OPERATOR("ReduceSum", 1, reduce_sum);
                //REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
                REGISTER_OPERATOR("Relu", 1, TranslateUnaryOp<op::Relu>);
                //REGISTER_OPERATOR("Reshape", 1, reshape);
                //REGISTER_OPERATOR("Selu", 1, selu);
                //REGISTER_OPERATOR("Shape", 1, shape);
                REGISTER_OPERATOR("Sigmoid", 1, TranslateUnaryOp<op::Sigmoid>);
                REGISTER_OPERATOR("Sin", 1, TranslateUnaryOp<op::Sin>);
                //REGISTER_OPERATOR("Slice", 1, slice);
                //REGISTER_OPERATOR("Softmax", 1, softmax);
                //REGISTER_OPERATOR("Softplus", 1, softplus);
                //REGISTER_OPERATOR("Softsign", 1, softsign);
                //REGISTER_OPERATOR("Split", 1, split);
                REGISTER_OPERATOR("Sqrt", 1, TranslateUnaryOp<op::Sqrt>);
                //REGISTER_OPERATOR("Squeeze", 1, squeeze);
                REGISTER_OPERATOR("Sub", 1, TranslateLegacyBinaryOp<op::Subtract>);
                REGISTER_OPERATOR("Sub", 7, TranslateBinaryOp<op::Subtract>);
                //REGISTER_OPERATOR("Sum", 1, sum);
                REGISTER_OPERATOR("Tan", 1, TranslateUnaryOp<op::Tan>);
                REGISTER_OPERATOR("Tanh", 1, TranslateUnaryOp<op::Tanh>);
                /*
                REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
                REGISTER_OPERATOR("Transpose", 1, transpose);
                REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
                REGISTER_OPERATOR("Xor", 1, logical_xor);
                */
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
