// Microsoft (c) 2019, NNFusion Team

#pragma once

namespace nnfusion
{
    namespace op
    {
        class Abs;
        class Acos;
        class Add;
        class Asin;
        class Atan;
        class Ceiling;
        class Cos;
        class Cosh;
        class Exp;
        class Floor;
        class Log;
        class Sin;
        class Sinh;
        class Tan;
        class Tanh;
        class Power;
        class Subtract;
        class Divide;
        class Sign;
        class Maximum;
        class Minimum;
        class Multiply;
        class Convert;
        class Equal;
        class NotEqual;
        class Greater;
        class GreaterEq;
        class Less;
        class LessEq;
        class Not;
        class Relu;
        class ReluBackprop;
        class Max;
        class Min;
        class Negative;
        class Not;
        class Sqrt;
        class Select;
        class And;
        class Or;
        class Nop;
        class Sigmoid;
        class SigmoidBackprop;
    }
}

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            struct TvmOpMap;

            template <>
            struct TvmOpMap<nnfusion::op::Abs>
            {
                static constexpr const char* op = "topi.abs";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Acos>
            {
                static constexpr const char* op = "acosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct TvmOpMap<nnfusion::op::Asin>
            {
                static constexpr const char* op = "asinf";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Atan>
            {
                static constexpr const char* op = "atanf";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Ceiling>
            {
                static constexpr const char* op = "topi.ceil";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Cos>
            {
                static constexpr const char* op = "topi.cos";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Cosh>
            {
                static constexpr const char* op = "coshf";
                static constexpr const char* math_kernel = nullptr;
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Exp>
            {
                static constexpr const char* op = "topi.exp";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Floor>
            {
                static constexpr const char* op = "topi.floor";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Log>
            {
                static constexpr const char* op = "topi.log";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Max>
            {
                static constexpr const char* op = "topi.maximum";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Min>
            {
                static constexpr const char* op = "topi.minimum";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Sin>
            {
                static constexpr const char* op = "topi.sin";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Sinh>
            {
                static constexpr const char* op = "sinhf";
                static constexpr const char* math_kernel = nullptr;
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Sqrt>
            {
                static constexpr const char* op = "topi.sqrt";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Tan>
            {
                static constexpr const char* op = "tanf";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Tanh>
            {
                static constexpr const char* op = "topi.tanh";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Power>
            {
                static constexpr const char* op = "topi.power";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Subtract>
            {
                static constexpr const char* op = "topi.subtract";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Divide>
            {
                static constexpr const char* op = "topi.divide";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::DivNoNan>
            {
                static constexpr const char* op = "divnonan";
                static constexpr const char* math_kernel = "x1 != 0 ? fdividef(x0, x1) : 0";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Sign>
            {
                static constexpr const char* op = "topi.sign";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Convert>
            {
                static constexpr const char* op = "convert";
                static constexpr const char* math_kernel = "x0";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Equal>
            {
                static constexpr const char* op = "topi.equal";
            };

            template <>
            struct TvmOpMap<nnfusion::op::NotEqual>
            {
                static constexpr const char* op = "topi.not_equal";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Greater>
            {
                static constexpr const char* op = "topi.greater";
            };

            template <>
            struct TvmOpMap<nnfusion::op::GreaterEq>
            {
                static constexpr const char* op = "topi.greater_equal";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Less>
            {
                static constexpr const char* op = "topi.less";
            };

            template <>
            struct TvmOpMap<nnfusion::op::LessEq>
            {
                static constexpr const char* op = "topi.less_equal";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Relu>
            {
                static constexpr const char* op = "topi.nn.relu";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Not>
            {
                static constexpr const char* op = "topi.logical_not";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Negative>
            {
                static constexpr const char* op = "topi.negative";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::ReluBackprop>
            {
                static constexpr const char* op = "relu_backprop";
                static constexpr const char* math_kernel = "x1 * int(x0 > 0)";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::And>
            {
                static constexpr const char* op = "topi.logical_and";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Or>
            {
                static constexpr const char* op = "topi.logical_or";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Add>
            {
                static constexpr const char* op = "topi.add";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Multiply>
            {
                static constexpr const char* op = "topi.multiply";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Minimum>
            {
                static constexpr const char* op = "topi.minimum";
            };

            template <>
            struct TvmOpMap<nnfusion::op::Maximum>
            {
                static constexpr const char* op = "topi.maximum";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::Nop>
            {
                static constexpr const char* op = "";
                static constexpr const char* math_kernel = "";
                static constexpr const char* atomic = "";
            };
*/

            template <>
            struct TvmOpMap<nnfusion::op::Sigmoid>
            {
                static constexpr const char* op = "topi.sigmoid";
            };

            /*
            template <>
            struct TvmOpMap<nnfusion::op::SigmoidBackprop>
            {
                static constexpr const char* op = "sigmoid_backprop";
                static constexpr const char* math_kernel = "x1 / (2 + expf(-x0) + expf(x0))";
            };
*/
        }
    }
}
