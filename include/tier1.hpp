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
                       const Array::Pointer &  src,
                       const Array::Pointer &  dst,
                       const float &           sigma,
                       const int &             radius) -> void;
auto
gaussian_blur_func(const Device::Pointer & device,
                   const Array::Pointer &  src,
                   const Array::Pointer &  dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z) -> void;

auto
absolute_func(const Device::Pointer & device, const Array::Pointer & src, const Array::Pointer & dst) -> void;

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array::Pointer &  src0,
                         const Array::Pointer &  src1,
                         const Array::Pointer &  dst,
                         const float &           factor0,
                         const float &           factor1) -> void;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP
