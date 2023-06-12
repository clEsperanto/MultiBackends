#ifndef __INCLUDE_TIER1_HPP
#define __INCLUDE_TIER1_HPP

#include "array.hpp"
#include "backend.hpp"
#include "execution.hpp"


namespace cle::tier1
{

auto
absolute_func(const Array & src, const Array & dst, const Device::Pointer & device) -> void;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP
