// Microsoft (c) 2019, Wenxiang
// Metagraph IR, which is to guide the codegen procedcure.
// This IR is based on ONNIX::ir's interface, but
// Instructions has attribute, namespace, and tag

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "attribute.hpp"
#include "dependency.hpp"
#include "value.hpp"
#include "tag.hpp"

//typesdef
#define int64_t long long
#define uint8_t unsigned char