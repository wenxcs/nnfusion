// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/util/util.hpp"

namespace nnfusion
{
    namespace op
    {
        struct oi_pair
        {
            size_t output;
            size_t input;
            bool destructive;
        };

        /// \brief Base class for annotations added to graph ops
        class OpAnnotations
        {
        public:
            virtual ~OpAnnotations() = default;

            void add_in_place_oi_pair(const struct oi_pair& oi)
            {
                for (auto e : m_in_place_oi_pairs)
                {
                    NNFUSION_CHECK(e.input != oi.input && e.output != oi.output)
                        << "In_place hint conflicts with an existing entry";
                }
                m_in_place_oi_pairs.emplace_back(oi);
            }

            const std::vector<struct oi_pair>& get_in_place_oi_pairs() const
            {
                return m_in_place_oi_pairs;
            }

        private:
            // map of output-input pairs for which in-place computation is valid
            std::vector<struct oi_pair> m_in_place_oi_pairs;
        };
    }
}