#ifndef __INCLUDE_TIER1_HPP
#define __INCLUDE_TIER1_HPP

#include "array.hpp"
#include "device.hpp"
#include "execution.hpp"

namespace cle::tier1
{

auto
execute_separable_func(const Device::Pointer & device,
                       const KernelInfo &      kernel,
                       const Array &           src,
                       const Array &           dst,
                       const float &           sigma,
                       const int &             radius) -> void;
auto
gaussian_blur_func(const Device::Pointer & device,
                   const Array &           src,
                   const Array &           dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z) -> void;

auto
absolute_func(const Device::Pointer & device, const Array & src, const Array & dst) -> void;

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array &           src1,
                         const Array &           src2,
                         const Array &           dst,
                         const float &           factor1,
                         const float &           factor2) -> void;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP
