// Microsoft (c) 2019, Wenxiang
/**
 * \brief Predefined Operator Inventory for Unit Tests
 * \author wenxh
 */
#include "ngraph/runtime/nnfusion/core/common.hpp"

namespace nnfusion
{
    namespace inventory
    {
        template <class T>
        shared_ptr<T> create_object(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_input(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_output(int option = 0);
        template <class T, class dtype>
        vector<dtype> generate_param(int option = 0);
    }
}