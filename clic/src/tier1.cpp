#include "tier1.hpp"
#include "utils.hpp"

#include "cle_absolute.h"
#include "cle_add_images_weighted.h"
#include "cle_gaussian_blur_separable.h"

namespace cle::tier1
{

auto
absolute_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst) -> Array::Pointer
{
  tier0::create_like(src, dst);
  const KernelInfo    kernel = { "absolute", kernel::absolute };
  const ConstantList  constants = {};
  const ParameterList parameters = { { "src", src }, { "dst", dst } };
  const RangeArray    global_range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, parameters, constants, global_range);
  return dst;
}

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array::Pointer &  src0,
                         const Array::Pointer &  src1,
                         Array::Pointer          dst,
                         const float &           factor0,
                         const float &           factor1) -> Array::Pointer
{
  tier0::create_like(src0, dst);
  const KernelInfo    kernel = { "add_images_weighted", kernel::add_images_weighted };
  const ConstantList  constants = {};
  const ParameterList parameters = {
    { "src0", src0 }, { "src1", src1 }, { "dst", dst }, { "factor0", factor0 }, { "factor1", factor1 }
  };
  const RangeArray global_range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, parameters, constants, global_range);
  return dst;
}

auto
gaussian_blur_func(const Device::Pointer & device,
                   const Array::Pointer &  src,
                   Array::Pointer          dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z) -> Array::Pointer
{
  tier0::create_like(src, dst);
  const KernelInfo kernel = { "gaussian_blur_separable", kernel::gaussian_blur_separable };
  tier0::execute_separable_func(device,
                                kernel,
                                src,
                                dst,
                                { sigma_x, sigma_y, sigma_z },
                                { sigma2radius(sigma_x), sigma2radius(sigma_y), sigma2radius(sigma_z) });
  return dst;
}

} // namespace cle::tier1
