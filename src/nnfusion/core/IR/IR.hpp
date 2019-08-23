// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include "attribute.hpp"
#include "instruction.hpp"

namespace nnfusion
{
    namespace ir
    {
        //\brief The Basic Block is to store a block of instructions.
        class BasicBlock : public std::vector<ir::Instruction::Pointer>, public Tagable
        {
        public:
            BasicBlock()
            {
                next = nullptr;
                prior = nullptr;
            }
            using Pointer = std::shared_ptr<BasicBlock>;
            using Pointers = std::shared_ptr<vector<BasicBlock>>;

            Pointer get_next();
            Pointer get_prior();

        private:
            Pointer next, prior;
        };

        //\todo multi-i/o BasicBlock;

        //\brief The Program is a set of Basic blocks.
        class Program : public std::vector<BasicBlock::Pointer>, public Tagable
        {
        public:
            Program() {}
            using Pointer = std::shared_ptr<Program>;
            static Program create_single_basic_block_program();
            BasicBlock::Pointer create_empty_bb();
            Program(std::initializer_list<BasicBlock::Pointer>);

            BasicBlock::Pointer get_entry();
            BasicBlock::Pointer get_exit();

        private:
            BasicBlock::Pointer entry, exit;
        };
    }
}