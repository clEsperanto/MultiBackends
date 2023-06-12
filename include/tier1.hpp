#ifndef __INCLUDE_TIER1_HPP
#define __INCLUDE_TIER1_HPP

#include "array.hpp"
#include "backend.hpp"
#include "execution.hpp"


namespace cle::tier1
{

using DevicePtr = std::shared_ptr<cle::Device>;
using ParameterList = std::vector<std::pair<std::string, std::variant<Array, float, int>>>;
using ConstantList = std::vector<std::pair<std::string, int>>;
using KernelInfo = std::pair<std::string, std::string>;
using RangeArray = std::array<size_t, 3>;

auto
absolute_func(const Array & src, const Array & dst, const DevicePtr & device) -> void;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP