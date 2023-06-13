#ifndef __INCLUDE_TIER1_HPP
#define __INCLUDE_TIER1_HPP

#include "array.hpp"
#include "backend.hpp"
#include "execution.hpp"


namespace cle::tier1
{

auto
execute_separable_func(const Array &           src,
                       const Array &           dst,
                       const float &           sigma,
                       const int &             radius,
                       const Device::Pointer & device) -> void;
auto
gaussian_blur_func(const Array &           src,
                   const Array &           dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z,
                   const Device::Pointer & device) -> void;

auto
absolute_func(const Array & src, const Array & dst, const Device::Pointer & device) -> void;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP
