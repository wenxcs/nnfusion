// Microsoft (c) 2019, NNFusion Team

#include "create_fusion_block.hpp"

using namespace nnfusion;

std::shared_ptr<std::vector<ir::BasicBlock::Pointer>> SplitBlock(ir::BasicBlock::Pointer block_p)
{
    auto blocks = std::make_shared<std::vector<ir::BasicBlock::Pointer>>();
    ir::BasicBlock::Pointer sub_block = nullptr;

    int current_group_id = -2;

    auto AppendCurrentBlock = [&blocks, &sub_block, &current_group_id]() {
        if (sub_block)
        {
            // treat as a fusion_block as only if it has at least 2 ins
            if (current_group_id >= 0 && sub_block->size() >= 2)
            {
                sub_block->Set("fusion_group_id", std::move(current_group_id));
            }
            blocks->push_back(sub_block);
            sub_block = nullptr;
        }
    };

    for (auto ins : *block_p)
    {
        auto group_id =
            (*ins)["fusion_group_id"].is_valid() ? (*ins)["fusion_group_id"].as<int>() : -1;

        if (group_id == current_group_id)
        {
            enforce_not_nullptr(sub_block);
            sub_block->push_back(ins);
        }
        else
        {
            AppendCurrentBlock();
            sub_block = std::make_shared<ir::BasicBlock>();
            sub_block->push_back(ins);
            current_group_id = group_id;
        }
    }
    AppendCurrentBlock();

    return blocks;
}

bool CreateFusionBlock::run(std::shared_ptr<InterpreterContext> ctx,
                            std::shared_ptr<TranslationUnit> tu)
{
    auto& p = tu->program;

    std::vector<ir::BasicBlock::Pointer> main_blocks(p.begin(), p.end());

    tu->program.clear();

    for (auto block : main_blocks)
    {
        auto sub_blocks = SplitBlock(block);
        tu->program.insert(tu->program.begin(), sub_blocks->begin(), sub_blocks->end());
    }

    // debug
    for (auto block : tu->program)
    {
        int group_id = -1;
        if (block->hasAttribute("fusion_group_id"))
        {
            group_id = block->Get<int>("fusion_group_id");
            LOG_INFO << "----------fusion group: " << group_id;
            for (auto ins : *block)
            {
                LOG_INFO << ins->name();
            }
            LOG_INFO << "----------";
        }
    }

    return true;
}