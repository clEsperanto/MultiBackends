#ifndef __INCLUDE_TIER1_HPP
#define __INCLUDE_TIER1_HPP

#include "tier0.hpp"

namespace cle::tier1
{


auto
gaussian_blur_func(const Device::Pointer & device,
                   const Array::Pointer &  src,
                   Array::Pointer          dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z) -> Array::Pointer;

auto
absolute_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst) -> Array::Pointer;

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array::Pointer &  src0,
                         const Array::Pointer &  src1,
                         Array::Pointer          dst,
                         const float &           factor0,
                         const float &           factor1) -> Array::Pointer;

} // namespace cle::tier1

#endif // __INCLUDE_TIER1_HPP
