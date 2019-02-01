// Microsoft (c) 2019, Wenxiang
#include "cuda_cudnn.hpp"

std::vector<int> cuda::compute_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }
    return strides;
}

std::string cuda::get_cudnn_datatype(std::string dtype)
{
    static const std::unordered_map<std::string, std::string> datatype_map{
        {"float", "CUDNN_DATA_FLOAT"},
        {"double", "CUDNN_DATA_DOUBLE"},
        {"int8_t", "CUDNN_DATA_INT8"},
        {"int32_t", "CUDNN_DATA_INT32"}};
    auto p = datatype_map.find(dtype);
    if (p == datatype_map.end())
    {
        std::string err = dtype + "is not supported by cuDNN";
        throw std::runtime_error(err);
    }
    return p->second;
}

LanguageUnit_p cuda::cudnn_tensor_descriptor_from_shape(const ngraph::Shape& shape, string desc)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;
    string data_type = "CUDNN_DATA_FLOAT"; //cuda::get_cudnn_datatype(type);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnTensorDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&" << desc << "));\n";
    if (shape.size() < 4)
    {
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(shape[i]);
        }
        lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
           << data_type << ", " << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2]
           << ", " << dimensions[3] << "));\n";
    }
    else if (shape.size() == 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
           << data_type << ", " << static_cast<int>(shape[0]) << ", " << static_cast<int>(shape[1])
           << ", " << static_cast<int>(shape[2]) << ", " << static_cast<int>(shape[3]) << "));\n";
    }
    else
    {
        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            assert_bool(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}\n";
            return ss.str();
        };

        std::vector<int> dimensions(shape.size());
        for (auto i = 0u; i < shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(shape[i]);
        }
        vector<int> strides = cuda::compute_strides(dimensions);

        lu << expand_vector_int(desc + "_dim", dimensions);
        lu << expand_vector_int(desc + "_strides", strides);

        lu << "CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(" << desc << ", " << data_type << ", "
           << static_cast<int>(dimensions.size()) << ", " << desc << "_dim, " << desc << "_strides"
           << "));\n";
    }

    return _lu;
}