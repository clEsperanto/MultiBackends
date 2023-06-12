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
  const ParameterList parameters = { { "src", std::cref(src) }, { "dst", std::cref(dst) } };
  const RangeArray    global_range = { dst.width(), dst.height(), dst.depth() };
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
  std::vector<float> src_data(src.nbElements());
  src.read(src_data.data());
  std::cout << "src: ";
  for (auto i : src_data)
  {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  const ConstantList constants = {};
  const RangeArray   global_range = { dst.width(), dst.height(), dst.depth() };

  std::cout << sigma[0] << " " << sigma[1] << " " << sigma[2] << std::endl;
  std::cout << radius[0] << " " << radius[1] << " " << radius[2] << std::endl;

  Array tmp1(dst.width(), dst.height(), dst.depth(), dst.dtype(), dst.mtype(), dst.device());
  Array tmp2(dst.width(), dst.height(), dst.depth(), dst.dtype(), dst.mtype(), dst.device());

  if (dst.width() > 1 && sigma[0] > 0)
  {
    const ParameterList parameters = {
      { "src", std::cref(src) }, { "dst", std::cref(tmp1) }, { "dim", 0 }, { "N", radius[0] }, { "s", sigma[0] }
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
      { "src", std::cref(tmp1) }, { "dst", std::cref(tmp2) }, { "dim", 1 }, { "N", radius[1] }, { "s", sigma[1] }
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
      { "src", std::cref(tmp2) }, { "dst", std::cref(dst) }, { "dim", 2 }, { "N", radius[2] }, { "s", sigma[2] }
    };
    execute(device, kernel, parameters, constants, global_range);
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
  std::vector<float> src_data(src.nbElements());
  src.read(src_data.data());
  std::cout << "src: ";
  for (auto i : src_data)
  {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  const KernelInfo kernel = { "gaussian_blur_separable", kernel::gaussian_blur_separable };
  execute_separable_func(src,
                         dst,
                         { sigma_x, sigma_y, sigma_z },
                         { sigma2radius(sigma_x), sigma2radius(sigma_y), sigma2radius(sigma_z) },
                         kernel,
                         device);
}

} // namespace cle::tier1
