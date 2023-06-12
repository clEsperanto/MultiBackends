#include "tier1.hpp"

#include "cle_absolute.h"

namespace cle::tier1
{

auto
absolute_func(const Array & src, const Array & dst, const DevicePtr & device) -> void
{
  const KernelInfo    kernel_func = { "absolute", kernel::absolute };
  const ConstantList  constants = {};
  const ParameterList parameters = { { "src", src }, { "dst", dst } };
  const RangeArray    global_range = { dst.width(), dst.height(), dst.depth() };
  execute(device, kernel_func, parameters, constants, global_range);
}

} // namespace cle::tier1
