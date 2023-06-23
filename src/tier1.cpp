#include "tier1.hpp"

#include "cle_absolute.h"
#include "cle_gaussian_blur_separable.h"

namespace cle::tier1
{

auto
absolute_func(const Device::Pointer & device, const Array & src, const Array & dst) -> void
{
  const KernelInfo    kernel = { "absolute", kernel::absolute };
  const ConstantList  constants = {};
  const ParameterList parameters = { { "src", src.ptr() }, { "dst", dst.ptr() } };
  const RangeArray    global_range = { dst.width(), dst.height(), dst.depth() };
  execute(device, kernel, parameters, constants, global_range);
}

auto
execute_separable_func(const Device::Pointer &      device,
                       const KernelInfo &           kernel,
                       const Array &                src,
                       const Array &                dst,
                       const std::array<float, 3> & sigma,
                       const std::array<int, 3> &   radius) -> void
{
  const ConstantList constants = {};
  const RangeArray   global_range = { dst.width(), dst.height(), dst.depth() };

  Array tmp1(dst);
  Array tmp2(dst);

  if (dst.width() > 1 && sigma[0] > 0)
  {
    const ParameterList parameters = {
      { "src", src.ptr() }, { "dst", tmp1.ptr() }, { "dim", 0 }, { "N", radius[0] }, { "s", sigma[0] }
    };
    execute(device, kernel, parameters, constants, global_range);
  }
  else
  {
    src.copy(tmp1);
  }
  if (dst.height() > 1 && sigma[1] > 0)
  {
    const ParameterList parameters = {
      { "src", tmp1.ptr() }, { "dst", tmp2.ptr() }, { "dim", 1 }, { "N", radius[1] }, { "s", sigma[1] }
    };
    execute(device, kernel, parameters, constants, global_range);
  }
  else
  {
    tmp1.copy(tmp2);
  }
  if (dst.depth() > 1 && sigma[2] > 0)
  {
    const ParameterList parameters = {
      { "src", tmp2.ptr() }, { "dst", dst.ptr() }, { "dim", 2 }, { "N", radius[2] }, { "s", sigma[2] }
    };
    execute(device, kernel, parameters, constants, global_range);
  }
  else
  {
    tmp2.copy(dst);
  }
}

auto
gaussian_blur_func(const Device::Pointer & device,
                   const Array &           src,
                   const Array &           dst,
                   const float &           sigma_x,
                   const float &           sigma_y,
                   const float &           sigma_z) -> void
{
  const KernelInfo kernel = { "gaussian_blur_separable", kernel::gaussian_blur_separable };
  execute_separable_func(device,
                         kernel,
                         src,
                         dst,
                         { sigma_x, sigma_y, sigma_z },
                         { sigma2radius(sigma_x), sigma2radius(sigma_y), sigma2radius(sigma_z) }

  );
}

} // namespace cle::tier1
