// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "../test_util/common.hpp"
#include "gtest/gtest.h"
#include "ngraph/op/pad.hpp"
#include "nnfusion/core/IR/IR.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"

TEST(nnfusion_core_ir, tagable)
{
    // Check Tag-able interface for instruction;
    nnfusion::ir::Instruction ins;
    ins["Example"].set<std::string>("Yes");
    EXPECT_TRUE(ins["Example"].is_valid());
    EXPECT_TRUE(ins["Example"].get<std::string>() == std::string("Yes"));

    // Check Tag-able interface for GNode;
    nnfusion::graph::GNode gnode;
    gnode["Example"].set<std::string>("Yes");
    EXPECT_TRUE(gnode["Example"].is_valid());
    EXPECT_TRUE(gnode["Example"].get<std::string>() == std::string("Yes"));

    // How to copy tags;
    gnode["Grouped"].set<bool>(true);
    ins.copy_tags_from(gnode);
    EXPECT_TRUE(ins["Grouped"].is_valid() && ins["Grouped"].get<bool>());
}