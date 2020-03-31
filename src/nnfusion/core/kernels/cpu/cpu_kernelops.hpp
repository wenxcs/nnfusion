// Microsoft (c) 2019, NNFusion Team

#pragma once

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            struct CpuOpMap;

            template <>
            struct CpuOpMap<nnfusion::op::Abs>
            {
                static constexpr const char* antares_op = "topi.abs";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Acos>
            {
                static constexpr const char* antares_op = "acosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CpuOpMap<nnfusion::op::Asin>
            {
                static constexpr const char* antares_op = "asinf";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Atan>
            {
                static constexpr const char* antares_op = "atanf";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Ceiling>
            {
                static constexpr const char* antares_op = "topi.ceil";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Cos>
            {
                static constexpr const char* antares_op = "topi.cos";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Cosh>
            {
                static constexpr const char* antares_op = "coshf";
                static constexpr const char* math_kernel = nullptr;
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Exp>
            {
                static constexpr const char* antares_op = "topi.exp";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Floor>
            {
                static constexpr const char* antares_op = "topi.floor";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Log>
            {
                static constexpr const char* antares_op = "topi.log";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Maximum>
            {
                static constexpr const char* antares_op = "topi.maximum";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Minimum>
            {
                static constexpr const char* antares_op = "topi.minimum";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Sin>
            {
                static constexpr const char* antares_op = "topi.sin";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Sinh>
            {
                static constexpr const char* antares_op = "sinhf";
                static constexpr const char* math_kernel = nullptr;
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sqrt>
            {
                static constexpr const char* antares_op = "topi.sqrt";
                static constexpr const char* eigen_op = "sqrt";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Rsqrt>
            {
                static constexpr const char* antares_op = "topi.rsqrt";
                static constexpr const char* eigen_op = "rsqrt";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Square>
            {
                static constexpr const char* antares_op = nullptr;
                static constexpr const char* eigen_op = "square";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Tan>
            {
                static constexpr const char* antares_op = "tanf";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Tanh>
            {
                static constexpr const char* antares_op = "topi.tanh";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Power>
            {
                static constexpr const char* antares_op = "topi.power";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Subtract>
            {
                static constexpr const char* antares_op = "topi.subtract";
                static constexpr const char* eigen_op = "-";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Divide>
            {
                static constexpr const char* antares_op = "topi.divide";
                static constexpr const char* eigen_op = "/";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::DivNoNan>
            {
                static constexpr const char* antares_op = "divnonan";
                static constexpr const char* math_kernel = "x1 != 0 ? fdividef(x0, x1) : 0";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sign>
            {
                static constexpr const char* antares_op = "topi.sign";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Convert>
            {
                static constexpr const char* antares_op = "convert";
                static constexpr const char* math_kernel = "x0";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Equal>
            {
                static constexpr const char* antares_op = "topi.equal";
            };

            template <>
            struct CpuOpMap<nnfusion::op::NotEqual>
            {
                static constexpr const char* antares_op = "topi.not_equal";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Greater>
            {
                static constexpr const char* antares_op = "topi.greater";
            };

            template <>
            struct CpuOpMap<nnfusion::op::GreaterEq>
            {
                static constexpr const char* antares_op = "topi.greater_equal";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Less>
            {
                static constexpr const char* antares_op = "topi.less";
            };

            template <>
            struct CpuOpMap<nnfusion::op::LessEq>
            {
                static constexpr const char* antares_op = "topi.less_equal";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Relu>
            {
                static constexpr const char* antares_op = "topi.nn.relu";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Not>
            {
                static constexpr const char* antares_op = "topi.logical_not";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Negative>
            {
                static constexpr const char* antares_op = "topi.negative";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::ReluBackprop>
            {
                static constexpr const char* antares_op = "relu_backprop";
                static constexpr const char* math_kernel = "x1 * int(x0 > 0)";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::And>
            {
                static constexpr const char* antares_op = "topi.logical_and";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Or>
            {
                static constexpr const char* antares_op = "topi.logical_or";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Add>
            {
                static constexpr const char* antares_op = "topi.add";
                static constexpr const char* eigen_op = "+";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Multiply>
            {
                static constexpr const char* antares_op = "topi.multiply";
                static constexpr const char* eigen_op = "*";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::Nop>
            {
                static constexpr const char* antares_op = "";
                static constexpr const char* math_kernel = "";
                static constexpr const char* atomic = "";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sigmoid>
            {
                static constexpr const char* antares_op = "topi.sigmoid";
            };

            /*
            template <>
            struct CpuOpMap<nnfusion::op::SigmoidBackprop>
            {
                static constexpr const char* antares_op = "sigmoid_backprop";
                static constexpr const char* math_kernel = "x1 / (2 + expf(-x0) + expf(x0))";
            };
*/

            template <>
            struct CpuOpMap<nnfusion::op::Sum>
            {
                static constexpr const char* antares_op = "topi.sum";
                static constexpr const char* eigen_op = "sum";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Product>
            {
                static constexpr const char* antares_op = "topi.prod";
                static constexpr const char* eigen_op = "prod";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Max>
            {
                static constexpr const char* antares_op = "topi.max";
                static constexpr const char* eigen_op = "maxCoeff";
            };

            template <>
            struct CpuOpMap<nnfusion::op::Min>
            {
                static constexpr const char* antares_op = "topi.min";
                static constexpr const char* eigen_op = "minCoeff";
            };
        }
    }
}
