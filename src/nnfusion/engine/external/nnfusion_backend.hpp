// Microsoft (c) 2019, Wenxiang Hu
#pragma once

#include "backend.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/engine/interpreter.hpp"

nnfusion::DeviceType default_device;

namespace nnfusion
{
    class cuda_codegen : public nnfusion_Backend
    {
    public:
        cuda_codegen();
        bool codegen(shared_ptr<graph::Graph> graph);

        shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                  const Shape& shape);
        shared_ptr<runtime::Tensor> create_tensor(const element::Type& element_type,
                                                  const Shape& shape,
                                                  void* memory_pointer);

    private:
        map<shared_ptr<graph::Graph>, TranslationUnit> m_graph_map;

    protected:
        shared_ptr<Interpreter> m_functrans;
    };
}