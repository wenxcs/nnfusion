// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "instruction.hpp"

namespace nnfusion
{
    namespace ir
    {
        class Scope : public Instruction
        {
        };
    }
}