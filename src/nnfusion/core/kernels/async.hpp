// Microsoft (c) 2019, NNFusion Team
#pragma once

#include "nnfusion/common/common.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/common/tensorwrapper.hpp"

namespace nnfusion
{
    namespace kernels
    {
        /*
        Execution Stream Info
         stream default: ------- kernel#0 ----------------
                                   |
                                   |--> trigger: event#0
                                                 ^ 
                                   wait: event#0 | 
                                                 |
         stream      #1: -------------------- kernel#1---
         Its the stream who wait for the event.

        Design purpose aync allreduce:
         stream: default | copy:d2h | allreduce | copy:h2d
                   |          |          |           |
                  op          |          |           |
                   |->grad    |          |           |
                   |   |--->memcpy(*)    |           |
                   |          |----->SuperScaler     |
                   |          |          |-(option)->memcpy(*)-->grad
                   |          |          |-(option)-------------> |
                  ...         x          x                        |
                   |                                              |
           event:iteration end                                    |
                   |                                              |
               apply_grad<-------(grad ready event)---------------
                   |
               apply_grad_other_0 (Apply grad for other op)
                   |
               apply_grad_other_1
                   |
                   x <----next interation


        (*) means this operation works stimulously with default stream.
        */

        struct AsyncExecutionInfo
        {
            struct Stream
            {
                uint32_t number = 0;
                std::string name = "";

                bool is_default_stream() { return number == 0; }
            };

            struct Event
            {
                uint32_t number = 0;
                std::string name = "";

                bool is_invalid_event() { return number == 0; }
            };
            Stream execution_stream;                 // 0 is default stream;
            Event wait_for_event, trigger_off_event; // 0 for none event;
        };
    }
}