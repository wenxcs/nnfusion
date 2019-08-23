#include "attribute.hpp"

namespace nnfusion
{
    namespace ir
    {
        TagProxy Tags::operator[](Symbol sym) { return TagProxy(this, sym); }
    }
}