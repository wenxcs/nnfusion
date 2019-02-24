// Microsoft (c) 2019, Wenxiang
/**
 * \brief Unit tests for ir::anyop
 * \author wenxh
 */

#include "ngraph/runtime/nnfusion/cuda/op/convolution.hpp"
#include "../test_util/common.hpp"

TEST(nnfusion_cuda_op, convolution)
{
    // Prepare
    auto node = nnfusion::inventory::create_object<op::Convolution>(0);
    EXPECT_TRUE(node != nullptr);

    // Static Method
    auto translated = nnfusion::ir::Convolution::translate(node);
    EXPECT_TRUE(translated != nullptr);

    // Initialized Normally
    auto op = static_pointer_cast<nnfusion::ir::Convolution>(translated);
    EXPECT_TRUE(op != nullptr);

    auto cuda_op = nnfusion::cuda::Convolution::codegen(op);

    // output
    /*
    std::cout<<cuda_op->codegen_test_name();
    std::cout<<cuda_op->definition_unit->get_code()<<endl;
    std::cout<<cuda_op->call_unit->get_code()<<endl;
    */
}