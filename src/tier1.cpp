#include "tier1.hpp"

#include "cle_absolute.h"
#include "cle_gaussian_blur_separable.h"

namespace cle::tier1
{

auto
absolute_func(const Array & src, const Array & dst, const DevicePtr & device) -> void
{
  const KernelInfo    kernel = { "absolute", kernel::absolute };
  const ConstantList  constants = {};
  const ParameterList parameters = { { "src", src }, { "dst", dst } };
  const RangeArray    global_range = { dst.width(), dst.height(), dst.depth() };
  execute(device, kernel, parameters, constants, global_range);
}

auto
separable_func(const Array &      src,
               const Array &      dst,
               const float &      sigma,
               const int &        radius,
               const int &        dimension,
               const KernelInfo & kernel,
               const DevicePtr &  device) -> void
{
  const ParameterList parameters = {
    { "src", src }, { "dst", dst }, { "dim", dimension }, { "N", radius }, { "s", sigma }
  };
  const RangeArray   global_range = { dst.width(), dst.height(), dst.depth() };
  const ConstantList constants = {};
  execute(device, kernel, parameters, constants, global_range);
}

auto
execute_separable_func(const Array &                src,
                       const Array &                dst,
                       const std::array<float, 3> & sigma,
                       const std::array<int, 3> &   radius,
                       const KernelInfo &           kernel,
                       const DevicePtr &            device) -> void
{
  Array tmp1(dst.width(), dst.height(), dst.depth(), dst.dtype(), dst.mtype(), dst.device());
  Array tmp2(dst.width(), dst.height(), dst.depth(), dst.dtype(), dst.mtype(), dst.device());

  if (dst.width() > 1 && sigma[0] > 0)
  {
    separable_func(src, tmp1, sigma[0], radius[0], 0, kernel, device);
  }
  else
  {
    src.copy(tmp1);
  }
  if (dst.height() > 1 && sigma[1] > 0)
  {
    separable_func(tmp1, tmp2, sigma[1], radius[1], 1, kernel, device);
  }
  else
  {
    tmp1.copy(tmp2);
  }
  if (dst.depth() > 1 && sigma[2] > 0)
  {
    separable_func(tmp2, dst, sigma[2], radius[2], 2, kernel, device);
  }
  else
  {
    tmp2.copy(dst);
  }
}

auto
gaussian_blur_func(const Array &     src,
                   const Array &     dst,
                   const float &     sigma_x,
                   const float &     sigma_y,
                   const float &     sigma_z,
                   const DevicePtr & device) -> void
{
  const KernelInfo kernel = { "gaussian_blur_separable", kernel::gaussian_blur_separable };
  execute_separable_func(src,
                         dst,
                         { sigma_x, sigma_y, sigma_z },
                         { sigma2radius(sigma_x), sigma2radius(sigma_y), sigma2radius(sigma_z) },
                         kernel,
                         device);
}

} // namespace cle::tier1
