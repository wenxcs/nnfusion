//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "op/abs.hpp"

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
    m_map[""][name_].emplace(ver_,                                                                 \
                             std::bind(op::set_##ver_::fn_,                                        \
                                       std::placeholders::_1,                                      \
                                       std::placeholders::_2,                                      \
                                       std::placeholders::_3))

            OperatorsBridge::OperatorsBridge()
            {
                REGISTER_OPERATOR("Abs", 1, abs);
                /*
                REGISTER_OPERATOR("Acos", 1, acos);
                REGISTER_OPERATOR("Add", 1, add);
                REGISTER_OPERATOR("Add", 7, add);
                REGISTER_OPERATOR("And", 1, logical_and);
                REGISTER_OPERATOR("ArgMin", 1, argmin);
                REGISTER_OPERATOR("ArgMax", 1, argmax);
                REGISTER_OPERATOR("Asin", 1, asin);
                REGISTER_OPERATOR("Atan", 1, atan);
                REGISTER_OPERATOR("AveragePool", 1, average_pool);
                REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
                REGISTER_OPERATOR("Cast", 1, cast);
                REGISTER_OPERATOR("Ceil", 1, ceil);
                REGISTER_OPERATOR("Clip", 1, clip);
                REGISTER_OPERATOR("Concat", 1, concat);
                REGISTER_OPERATOR("Constant", 1, constant);
                REGISTER_OPERATOR("Conv", 1, conv);
                REGISTER_OPERATOR("Cos", 1, cos);
                REGISTER_OPERATOR("Div", 1, div);
                REGISTER_OPERATOR("Div", 7, div);
                REGISTER_OPERATOR("Dropout", 1, identity);
                REGISTER_OPERATOR("Elu", 1, elu);
                REGISTER_OPERATOR("Equal", 1, equal);
                REGISTER_OPERATOR("Exp", 1, exp);
                REGISTER_OPERATOR("Flatten", 1, flatten);
                REGISTER_OPERATOR("Floor", 1, floor);
                REGISTER_OPERATOR("Gemm", 1, gemm);
                REGISTER_OPERATOR("GlobalAveragePool", 1, global_average_pool);
                REGISTER_OPERATOR("GlobalMaxPool", 1, global_max_pool);
                REGISTER_OPERATOR("Greater", 1, greater);
                REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
                REGISTER_OPERATOR("Identity", 1, identity);
                REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
                REGISTER_OPERATOR("Less", 1, less);
                REGISTER_OPERATOR("Log", 1, log);
                REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
                REGISTER_OPERATOR("LRN", 1, lrn);
                REGISTER_OPERATOR("MatMul", 1, matmul);
                REGISTER_OPERATOR("MaxPool", 1, max_pool);
                REGISTER_OPERATOR("Max", 1, max);
                REGISTER_OPERATOR("Mean", 1, mean);
                REGISTER_OPERATOR("Min", 1, min);
                REGISTER_OPERATOR("Mul", 1, mul);
                REGISTER_OPERATOR("Mul", 7, mul);
                REGISTER_OPERATOR("Neg", 1, neg);
                REGISTER_OPERATOR("Not", 1, logical_not);
                REGISTER_OPERATOR("Or", 1, logical_or);
                REGISTER_OPERATOR("Pow", 1, pow);
                REGISTER_OPERATOR("PRelu", 1, prelu);
                REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
                REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
                REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
                REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
                REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
                REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
                REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
                REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
                REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
                REGISTER_OPERATOR("ReduceSum", 1, reduce_sum);
                REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
                REGISTER_OPERATOR("Relu", 1, relu);
                REGISTER_OPERATOR("Reshape", 1, reshape);
                REGISTER_OPERATOR("Selu", 1, selu);
                REGISTER_OPERATOR("Shape", 1, shape);
                REGISTER_OPERATOR("Sigmoid", 1, sigmoid);
                REGISTER_OPERATOR("Sin", 1, sin);
                REGISTER_OPERATOR("Slice", 1, slice);
                REGISTER_OPERATOR("Softmax", 1, softmax);
                REGISTER_OPERATOR("Softplus", 1, softplus);
                REGISTER_OPERATOR("Softsign", 1, softsign);
                REGISTER_OPERATOR("Split", 1, split);
                REGISTER_OPERATOR("Sqrt", 1, sqrt);
                REGISTER_OPERATOR("Squeeze", 1, squeeze);
                REGISTER_OPERATOR("Sub", 1, sub);
                REGISTER_OPERATOR("Sub", 7, sub);
                REGISTER_OPERATOR("Sum", 1, sum);
                REGISTER_OPERATOR("Tan", 1, tan);
                REGISTER_OPERATOR("Tanh", 1, tanh);
                REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
                REGISTER_OPERATOR("Transpose", 1, transpose);
                REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
                REGISTER_OPERATOR("Xor", 1, logical_xor);
                */
            }

        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
