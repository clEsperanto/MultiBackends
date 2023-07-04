
#include "tier0.hpp"

namespace cle::tier0
{

auto
create_like(const Array::Pointer & src, Array::Pointer & dst) -> void
{
  if (dst != nullptr)
  {
    return;
  }
  dst = Array::create(src);
}

auto
create_one(const Array::Pointer & src, Array::Pointer & dst) -> void
{
  if (dst != nullptr)
  {
    return;
  }
  dst = Array::create(1, 1, 1, src->dtype(), src->mtype(), src->device());
}

auto
create_xy(const Array::Pointer & src, Array::Pointer & dst) -> void
{
  if (dst != nullptr)
  {
    return;
  }
  dst = Array::create(src->width(), src->height(), 1, src->dtype(), src->mtype(), src->device());
}

auto
create_yz(const Array::Pointer & src, Array::Pointer & dst) -> void
{
  if (dst != nullptr)
  {
    return;
  }
  dst = Array::create(src->height(), src->depth(), 1, src->dtype(), src->mtype(), src->device());
}

auto
create_xz(const Array::Pointer & src, Array::Pointer & dst) -> void
{
  if (dst != nullptr)
  {
    return;
  }
  dst = Array::create(src->width(), src->depth(), 1, src->dtype(), src->mtype(), src->device());
}

auto
execute_separable_func(const Device::Pointer &      device,
                       const KernelInfo &           kernel,
                       const Array::Pointer &       src,
                       const Array::Pointer &       dst,
                       const std::array<float, 3> & sigma,
                       const std::array<int, 3> &   radius) -> void
{
  const RangeArray global_range = { dst->width(), dst->height(), dst->depth() };

  auto tmp1 = Array::create(dst);
  auto tmp2 = Array::create(dst);

  if (dst->width() > 1 && sigma[0] > 0)
  {
    const ParameterList parameters = {
      { "src", src }, { "dst", tmp1 }, { "dim", 0 }, { "N", radius[0] }, { "s", sigma[0] }
    };
    execute(device, kernel, parameters, global_range);
  }
  else
  {
    src->copy(tmp1);
  }
  if (dst->height() > 1 && sigma[1] > 0)
  {
    const ParameterList parameters = {
      { "src", tmp1 }, { "dst", tmp2 }, { "dim", 1 }, { "N", radius[1] }, { "s", sigma[1] }
    };
    execute(device, kernel, parameters, global_range);
  }
  else
  {
    tmp1->copy(tmp2);
  }
  if (dst->depth() > 1 && sigma[2] > 0)
  {
    const ParameterList parameters = {
      { "src", tmp2 }, { "dst", dst }, { "dim", 2 }, { "N", radius[2] }, { "s", sigma[2] }
    };
    execute(device, kernel, parameters, global_range);
  }
  else
  {
    tmp2->copy(dst);
  }
}

} // namespace cle::tier0